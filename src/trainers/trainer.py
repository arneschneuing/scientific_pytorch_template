import os
from src.utilities import util
from src.loggers.logger import Logger
from src.trainers.monitor import Monitor
from src.utilities.build_components import *


class Trainer:
    """
    Trainer class to handle training run specified in config file.
    :param result_dir: string | relative path to result dir where all outputs
    are saved
    :param cfg: config dict
    """
    def __init__(self, result_dir, cfg):

        # Set experiment-level config
        self._cfg = cfg

        # Create result dir
        self._result_dir = result_dir
        os.makedirs(result_dir, exist_ok=True)
        self._logger = Logger(result_dir,
                              cfg['Logging'].get('write_file', default=True),
                              cfg['Logging'].get('write_tb', default=True))

        # Build components
        self._data_loaders = build_dataloaders(cfg)
        self._model = build_model(cfg)
        self._criterion = build_criterion(cfg)
        self._metric_tracker = build_metric_tracker(cfg)
        self._optimizer = build_optimizer(cfg, self._model.parameters())

        # Use GPUs if available
        self._device, device_ids = self._prepare_device(cfg['n_gpu'])
        self._model = self._model.to(self._device)  # Send model to device
        if len(device_ids) > 1:
            self._model = torch.nn.DataParallel(self._model,
                                                device_ids=device_ids)

        # Resume from checkpoint if possible
        ckpt_path = self._resume_checkpoint()

        if ckpt_path is None:
            # Start new experiment
            os.makedirs(self._result_dir, exist_ok=True)

            # Create monitor for current training run
            self._monitor = Monitor(cfg, len(self._data_loaders['train']),
                                    log_fn=self._logger.log_string)

            # Log start of new training run
            log_string_1 = '#### Start new training! ####'
            log_string_2 = f'Result Dir: {self._result_dir}'
            self._logger.log_string(log_string_1 + '\n' + log_string_2)

        else:

            # Load state of relevant components
            model_state_dict, optim_state_dict, self._monitor = \
                self._load_checkpoint(ckpt_path)
            self._model.load_state_dict(model_state_dict)
            self._optimizer.load_state_dict(optim_state_dict)

            # Log resumption of training run
            log_string_1 = f'Result Dir: {self._result_dir}'
            log_string_2 = f'Resume training at iteration ' \
                           f'{self._monitor.it}!'
            self._logger.log_string(log_string_1 + '\n' + log_string_2)

    def train(self):

        # Perform training until monitor indicates end of training
        while not self._monitor.flags.end_training:

            # Perform single training iteration incl. forward pass, loss
            # computation and parameter update
            rtn_dict, tb_dict = self._train_it()

            # Tensorboard Logging
            self._logger.tb.train()  # Set tb logger to train mode
            for key, value in tb_dict.items():
                if isinstance(value, (int, float)):
                    self._logger.tb.add_scalar(key, value, self._monitor.it)

            # Perform validation if scheduled by the monitor
            if self._monitor.do_validation():

                # Validate on the whole validation dataset
                val_score = self.evaluate(split='val')

                # Update monitor with the latest validation score
                self._monitor(val_score)

                # Perform model saving according to monitor flags
                if self._monitor.flags.save_checkpoint:
                    self._save_checkpoint()
                if self._monitor.flags.new_best_model:
                    self._save_checkpoint(save_best=True)

            # Increase monitor's internal iteration counter
            self._monitor.update()

        # Print summary of the training run
        self._monitor.log_summary()

    def evaluate(self, split):

        # Assert that a valid split is provided
        assert split in ['train', 'val']

        # Set model to eval mode
        self._model.eval()

        # Accumulate loss
        loss = 0

        # Iterate over validation dataset
        for batch_id, (batch_data, batch_target) in \
                enumerate(self._data_loaders['split']):

            # Disable gradient taping for validation
            with torch.no_grad():

                # Perform forward pass
                batch_output = self._model(batch_data)

                # Compute loss
                loss += self._criterion(batch_output, batch_target)

        # Return average loss
        loss /= (len(self._data_loaders[split]))

        return loss

    def _train_it(self):

        # Set model to train mode
        self._model.train()

        # Reset gradients
        self._model.zero_grad()

        # Get batch data from data loader
        batch_data, batch_target = next(self._data_loaders['train_it'])

        # Perform forward pass
        batch_output = self._model(batch_data)

        # Compute loss
        loss = self._criterion(batch_output, batch_target)

        # Perform backward pass
        loss.backward()

        # Update parameters
        self._optimizer.step()

        return loss

    def _prepare_device(self, n_gpu):
        """
        Manage GPU setup
        Code adapted from:
        https://github.com/victoresque/pytorch-template/blob/master/base/
        base_trainer.py
        :param n_gpu: intended number of GPUs
        """

        # Get number of available GPUs
        n_gpu_available = torch.cuda.device_count()

        # Use CPU if no GPU available
        if n_gpu_available == 0 and n_gpu > 0:
            self._logger.log_string("Warning: no GPU available, using CPU...")
            n_gpu = 0

        # Use maximum available number of GPUs
        if n_gpu > n_gpu_available:
            self._logger.log_string(f"Warning: not enough GPUs available, "
                                    f"using {n_gpu_available} GPUs")
            n_gpu = n_gpu_available

        # Set device
        device = torch.device('cuda:0' if n_gpu > 0 else 'cpu')

        # Get list of available device indices
        list_ids = list(range(n_gpu))

        return device, list_ids

    def _save_checkpoint(self, save_best=False):
        """
        :param save_best: if True, rename the saved checkpoint to
        'best_model.pth'
        """

        # Create checkpoint dir
        state = {'state_dict': self._model.state_dict(),
                 'optimizer': self._optimizer.state_dict(),
                 'monitor': self._monitor}

        # Create checkpoint path
        ckpt_dir = os.path.join(self._result_dir, 'Checkpoints')

        # Set filename
        if save_best:
            ckpt_path = os.path.join(ckpt_dir, 'best_model.pth')
        else:
            if self._monitor.epoch_based:
                epoch = (self._monitor.it + 1) // \
                        len(self._data_loaders['train'])
                filename = f'ckpt_e{epoch}.pth'
            else:
                filename = f'ckpt_i{self._monitor.it + 1}.pth'
            ckpt_path = os.path.join(ckpt_dir, filename)

        # Save checkpoint
        torch.save(state, ckpt_path)

        # Log saving
        self._logger.log_string(f'Saving checkpoint: {ckpt_path} ...')

    def _resume_checkpoint(self):
        """
        Resume training from saved checkpoints.
        """

        # Set checkpoint directory
        ckpt_dir = os.path.join(self._result_dir, 'Checkpoints')

        # Check if checkpoint exists
        if os.path.isdir(ckpt_dir):

            # Get checkpoint identifier
            identifier = '_e' if self._monitor.epoch_based else '_i'

            # Get filename of latest checkpoint
            ckpt_filename = util.get_latest_version(ckpt_dir, identifier)

            # Verify training resumption with user input
            if ckpt_filename is not None:
                print(f'Found checkpoint {ckpt_filename}. '
                      f'Resume/Exit? [r]/e')
                c = input()
                if c == 'r' or c == '':
                    return ckpt_filename
                elif c == 'e':
                    print('Exiting...')
                    exit()
                else:
                    print('Wrong user input. Exiting...')
                    exit(-1)

        # Return None if no checkpoint found
        else:
            return None

    @staticmethod
    def _load_checkpoint(ckpt_path):

        checkpoint = torch.load(ckpt_path)
        monitor = checkpoint['monitor']
        model_state_dict = checkpoint['state_dict']
        optim_state_dict = checkpoint['optimizer']

        return model_state_dict, optim_state_dict, monitor
