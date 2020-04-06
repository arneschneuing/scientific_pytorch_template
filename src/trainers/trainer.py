import os
import torch
from src.loggers.logger import Logger
from src.trainers.monitor import Monitor
from src.utilities import util
import shutil


class Trainer:
    """
    Trainer class
    :param result_dir:
    :param model:
    :param criterion:
    :param metric_tracker:
    :param optimizer:
    :param cfg:
    :param data_loaders:
    """
    def __init__(self, result_dir, model, criterion, metric_tracker, optimizer,
                 cfg, data_loaders):
        self._cfg = cfg
        self._result_dir = result_dir
        os.mkdir(result_dir)
        self._logger = Logger(result_dir,
                              cfg['Logging'].get('write_file', default=True),
                              cfg['Logging'].get('write_tb', default=True))
        self._data_loaders = data_loaders

        # Use GPUs if available
        self._device, device_ids = self._prepare_device(cfg['n_gpu'])
        self._model = model.to(self._device)
        if len(device_ids) > 1:
            self._model = torch.nn.DataParallel(self._model,
                                                device_ids=device_ids)

        self._criterion = criterion
        self._metric_tracker = metric_tracker
        self._optimizer = optimizer

        ckpt_path = self._resume_checkpoint()

        if ckpt_path is None:
            # Start new experiment
            os.makedirs(self._result_dir, exist_ok=True)
            self._monitor = Monitor(cfg, len(self._data_loaders['train']),
                                    log_fn=self._logger.log_string)
            self._logger.log_string('Start training!')

        else:
            model_state_dict, optim_state_dict, self._monitor = \
                self._load_checkpoint(ckpt_path)
            self._model.load_state_dict(model_state_dict)
            self._optimizer.load_state_dict(optim_state_dict)
            self._logger.log_string(f'Resume training at iteration '
                                    f'{self._monitor.it}!')

    def train(self):

        while not self._monitor.flags.end_training:

            rtn_dict, tb_dict = self._train_it()

            # Tensorboard Logging
            self._logger.tb.train()
            for key, value in tb_dict.items():
                self._logger.tb.add_scalar(key, value, self._monitor.it)

            # Validate
            if self._monitor.do_validation():
                score = self.evaluate(split='val')

                # Update monitor
                self._monitor(score)

                # Perform model saving
                if self._monitor.flags.save_checkpoint:
                    self._save_checkpoint()
                if self._monitor.flags.new_best_model:
                    self._save_checkpoint(save_best=True)

            # Increase monitor's internal iteration counter
            self._monitor.update()

        self._monitor.log_summary()

    def _train_it(self):
        pass

    def _prepare_device(self, n_gpu):
        """
        Manage GPU setup
        Code adapted from:
        https://github.com/victoresque/pytorch-template/blob/master/base/
        base_trainer.py
        :param n_gpu: intended number of GPUs
        """
        n_gpu_available = torch.cuda.device_count()
        if n_gpu_available == 0 and n_gpu > 0:
            self._logger.log_string("Warning: no GPU available, using CPU...")
            n_gpu = 0
        if n_gpu > n_gpu_available:
            self._logger.log_string(f"Warning: not enough GPUs available, "
                                    f"using {n_gpu_available} GPUs")
            n_gpu = n_gpu_available
        device = torch.device('cuda:0' if n_gpu > 0 else 'cpu')
        list_ids = list(range(n_gpu))
        return device, list_ids

    def _save_checkpoint(self, save_best=False):
        """
        Saving checkpoints
        :param save_best: if True, rename the saved checkpoint to
        'best_model.pth'
        """

        state = {
            'state_dict': self._model.state_dict(),
            'optimizer': self._optimizer.state_dict(),
            'monitor': self._monitor,
        }

        ckpt_dir = os.path.join(self._result_dir, 'Checkpoints')
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
        torch.save(state, ckpt_path)

        self._logger.log_string(f'Saving checkpoint: {ckpt_path} ...')

    def _resume_checkpoint(self):
        """
        Resume from saved checkpoints
        """

        ckpt_dir = os.path.join(self._result_dir, 'Checkpoints')
        if os.path.isdir(ckpt_dir):
            identifier = '_e' if self._monitor.epoch_based else '_i'
            ckpt_filename = util.get_latest_version(ckpt_dir, identifier)

            if ckpt_filename is not None:
                print(f'Found checkpoint {ckpt_filename}. '
                      f'Resume/Overwrite/Exit? [r]/o/e')
                c = input()
                if c == 'r' or c == '':
                    return ckpt_filename
                elif c == 'o':
                    shutil.rmtree(self._result_dir)
                    print(f'Remove directory {self._result_dir}')
                    return None
                elif c == 'e':
                    print('Exiting...')
                    exit()
                else:
                    print('Wrong user input. Exiting...')
                    exit(-1)
        else:
            return None

    @staticmethod
    def _load_checkpoint(ckpt_path):

        checkpoint = torch.load(ckpt_path)
        monitor = checkpoint['monitor']
        model_state_dict = checkpoint['state_dict']
        optim_state_dict = checkpoint['optimizer']

        return model_state_dict, optim_state_dict, monitor
