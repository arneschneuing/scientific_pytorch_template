import os
import yaml
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

        # Set result dir
        self._result_dir = result_dir

        # Create logger
        self._logger = Logger(result_dir,
                              cfg['Logging'].get('write_file', True),
                              cfg['Logging'].get('write_tb', True))

        # Build components
        self._data_loaders = build_dataloaders(cfg)
        self._model = build_model(cfg)
        self._criterion = build_criterion(cfg)
        self._metric_trackers = build_metric_trackers(cfg)
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
            os.makedirs(result_dir, exist_ok=True)

            # Create monitor for current training run
            self._monitor = Monitor(cfg, len(self._data_loaders['train']))

            # Log start of new training run
            log_string_1 = '#### Start new training! ####'
            log_string_2 = f'Result Dir: {result_dir}'
            self._logger.log_string(log_string_1 + '\n' + log_string_2)

        else:

            # Load state of relevant components
            model_state_dict, optim_state_dict, self._monitor = \
                self._load_checkpoint(ckpt_path)
            self._model.load_state_dict(model_state_dict)
            self._optimizer.load_state_dict(optim_state_dict)

            # Prepare for next iteration of training
            self._monitor.update()

            # Log resumption of training run
            log_string_1 = f'Result Dir: {result_dir}'
            log_string_2 = f'Resume training at iteration ' \
                           f'{self._monitor.it + 1}!'
            self._logger.log_string(log_string_1 + '\n' + log_string_2)

        # Write config file
        config_path = os.path.join(result_dir, 'config.yaml')
        with open(config_path, 'w') as outfile:
            yaml.dump(cfg, outfile, default_flow_style=False)

    def train(self):

        # Perform training until monitor indicates end of training
        while not self._monitor.flags.end_training:

            # Perform single training iteration incl. forward pass, loss
            # computation and parameter update
            self._train_it()

            # Perform validation if scheduled by the monitor
            if self._monitor.do_logging():

                # Get metrics
                log_dict = self._metric_trackers['train'].get_metrics()

                # Log progress
                self._log_progress(log_dict)

                # Tensorboard Logging
                self._logger.tb.train()  # Set tb logger to train mode
                for key, value in log_dict.items():
                    self._logger.tb.add_scalar(key, value, self._monitor.it)

                # Flush tensorboard
                # self._logger.tb.flush()

                # Reset metric tracker
                self._metric_trackers['train'].reset()

            # Perform validation if scheduled by the monitor
            if self._monitor.do_validation():

                # Inform user about evaluation status
                if self._monitor.epoch_based:
                    epochs_trained = (self._monitor.it + 1) // \
                                     self._monitor.batches_per_epoch
                    self._logger.log_string(
                        f'#### Evaluation ({epochs_trained} Epochs trained)')
                else:
                    self._logger.log_string(
                        f'#### Evaluation ({self._monitor.it + 1} '
                        f'Iterations trained)')

                # Validate on the whole validation dataset
                self.evaluate(split='val')

                # Get metrics
                val_dict = self._metric_trackers['val'].get_metrics()

                # Tensorboard Logging
                self._logger.tb.val()  # Set tb logger to val mode
                for key, value in val_dict.items():
                    self._logger.tb.add_scalar(key, value, self._monitor.it)

                # Update monitor with the latest validation score
                self._monitor(val_dict['acc'])

                # Perform model saving according to monitor flags
                if self._monitor.flags.save_checkpoint:
                    self._save_checkpoint()
                if self._monitor.flags.new_best_model:
                    self._save_checkpoint(save_best=True)

                # Reset metric tracker
                self._metric_trackers['val'].reset()

                self._logger.log_string(f'Evaluation finished with '
                                        f'score: {val_dict["acc"]:.4f}!')

            # Increase monitor's internal iteration counter
            self._monitor.update()

        # Print summary of the training run
        self._logger.log_string(self._monitor.summary_string())

        # Close tensorboard loggers
        self._logger.close()

    def evaluate(self, split):

        # Assert that a valid split is provided
        assert split in ['train', 'val'], 'Incorrect split name provided!'

        # Set model to eval mode
        self._model.eval()

        # Iterate over validation dataset
        for batch_id, (batch_data, batch_target) in \
                enumerate(self._data_loaders[split]):

            # Disable gradient taping for validation
            with torch.no_grad():

                # Perform forward pass
                batch_output = self._model(batch_data)

                # Compute loss
                loss = self._criterion(batch_output, batch_target)

                # Update metric tracker
                self._metric_trackers['val'].update(batch_output, batch_target,
                                                    loss)

                # Perform validation if scheduled by the monitor
                if ((batch_id + 1) % self._monitor.LOG_FREQ) == 0:
                    self._logger.log_string(
                        f'#### Batch ID: {batch_id+1}/'
                        f'{len(self._data_loaders[split])} ####')

    def _train_it(self):

        # Set model to train mode
        self._model.train()

        # Reset gradients
        self._model.zero_grad()

        # Get batch data from data loader
        batch_data, batch_target = next(self._data_loaders['train'])

        # Perform forward pass
        batch_output = self._model(batch_data)

        # Compute loss
        loss = self._criterion(batch_output, batch_target)

        # Update metric tracker
        self._metric_trackers['train'].update(batch_output, batch_target, loss)

        # Perform backward pass
        loss.backward()

        # Update parameters
        self._optimizer.step()

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
        os.makedirs(ckpt_dir, exist_ok=True)

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

            # Get filename of latest checkpoint
            ckpt_candidates = [util.get_latest_version(ckpt_dir, '_i'),
                               util.get_latest_version(ckpt_dir, '_e')]
            if ckpt_candidates[0] is None and ckpt_candidates[1] is None:
                ckpt_filename = None
            else:
                ckpt_filename = ckpt_candidates[0] \
                    if ckpt_candidates[0] is not None else ckpt_candidates[1]

            # Verify training resumption with user input
            if ckpt_filename is not None:
                print(f'Found checkpoint {ckpt_filename}. '
                      f'Resume/Exit? [r]/e')
                c = input()
                if c == 'r' or c == '':
                    return os.path.join(ckpt_dir, ckpt_filename)
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

    def _log_progress(self, log_dict):
        """
        Log training progress according to specified training scheme.
        """

        if self._monitor.epoch_based:

            # Get number of current epoch and iteration
            current_epoch, epoch_it = self._monitor.i2e()

            # Set progress string
            prog_string = f'#### Epoch: {current_epoch} | ' \
                          f'Iteration: {epoch_it}/' \
                          f'{self._monitor.batches_per_epoch} ####'
        else:
            # Set progress string
            prog_string = f'#### Iteration: {self._monitor.it + 1}/' \
                          f'{self._monitor.num_iterations} ####'

        # Log string
        self._logger.log_string(prog_string)

        # Log content of log dict
        for key, item in log_dict.items():
            self._logger.log_string(f'{key:<15}: {item:.4f}')
