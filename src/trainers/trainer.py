# Code adapted from:
# https://github.com/victoresque/pytorch-template/blob/master/base/
# base_trainer.py

import os
import torch
from src.loggers.logger import Logger
from src.trainers.monitor import Monitor
from src.utilities import util
import shutil


class Trainer:

    LOG_FREQ = 10

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

        self._num_iterations, self._checkpoint_freq, self._val_freq, \
            self.epoch_based = self._get_scheduling_params(cfg)

        patience = self._cfg['Monitor'].get('patience',
                                            default=self._num_iterations)
        self._monitor = Monitor(self._checkpoint_freq, patience)

        ckpt_path = self._resume_checkpoint()

        if ckpt_path is None:
            # Start new experiment
            os.makedirs(self._result_dir, exist_ok=True)
            self._it = 0
            self._logger.log_string('Start training!')

        else:
            self._it, model_state_dict, optim_state_dict, self._monitor = \
                self._load_checkpoint(ckpt_path)
            self._model.load_state_dict(model_state_dict)
            self._optimizer.load_state_dict(optim_state_dict)
            self._logger.log_string(f'Resume training at iteration '
                                    f'{self._it}!')

    def _get_scheduling_params(self, cfg):
        # Read from config file
        num_epochs = cfg['Train'].get('num_epochs', default=None)
        num_iterations = cfg['Train'].get('num_iterations', default=None)
        checkpoint_freq = cfg['Train'].get('checkpoint_freq', default=None)
        val_freq = cfg['Train'].get('val_freq', default=None)

        if num_epochs is None and num_iterations is None:
            self._logger.log_string('Error: please specify the training '
                                    'duration using num_epochs or '
                                    'num_iterations in config file')
            exit(-1)
        if num_epochs is not None and num_iterations is not None:
            self._logger.log_string('Error: training duration is ambiguous, '
                                    'num_epochs and num_iterations are '
                                    'specified in config file')
            exit(-1)

        if num_iterations is None:
            # epoch-based training
            num_iterations = len(self._data_loaders['train']) * num_epochs
            checkpoint_freq = num_iterations if checkpoint_freq is None \
                else len(self._data_loaders['train']) * checkpoint_freq
            val_freq = len(self._data_loaders['train']) if val_freq is None \
                else len(self._data_loaders['train']) * val_freq
            epoch_based = True
        else:
            # iteration-based training
            epoch_based = False

        return num_iterations, checkpoint_freq, val_freq, epoch_based

    def train(self):

        while self._it < self._num_iterations:

            rtn_dict, tb_dict = self._train_it()

            # Tensorboard Logging
            self._logger.tb.train()
            for key, value in tb_dict.items():
                self._logger.tb.add_scalar(key, value, self._it)

            # Log Progress
            if (self._it + 1) % self.LOG_FREQ == 0:
                self._log_progress()

            # Validate
            if (self._it + 1) % self._val_freq == 0:
                if self.epoch_based:
                    epochs_trained = (self._it + 1) // \
                                     len(self._data_loaders['train'])
                    self._logger.log_string(f'#### Evaluation ('
                                            f'{epochs_trained} Epochs '
                                            f'trained)')
                else:
                    self._logger.log_string(f'#### Evaluation ('
                                            f'{self._it + 1} Iterations '
                                            f'trained)')
                score = self.evaluate(split='val')

                # Update monitor
                monitor_flags = self._monitor(score, self._it)

                # Perform model saving
                if monitor_flags.save_checkpoint:
                    self._save_checkpoint()
                if monitor_flags.new_best_model:
                    self._save_checkpoint(save_best=True)
                if monitor_flags.end_training:
                    break

        self._logger.log_string('End Training')

    def _train_it(self):
        pass

    def _prepare_device(self, n_gpu):
        """
        Manage GPU setup
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
            'iteration:': self._it,
            'state_dict': self._model.state_dict(),
            'optimizer': self._optimizer.state_dict(),
            'monitor': self._monitor,
        }

        ckpt_dir = os.path.join(self._result_dir, 'Checkpoints')
        if save_best:
            ckpt_path = os.path.join(ckpt_dir, 'best_model.pth')
        else:
            if self.epoch_based:
                epoch = (self._it + 1) // len(self._data_loaders['train'])
                filename = f'ckpt_e{epoch}.pth'
            else:
                filename = f'ckpt_i{self._it+1}.pth'
            ckpt_path = os.path.join(ckpt_dir, filename)
        torch.save(state, ckpt_path)

        self._logger.log_string(f'Saving checkpoint: {ckpt_path} ...')

    def _resume_checkpoint(self):
        """
        Resume from saved checkpoints
        """

        ckpt_dir = os.path.join(self._result_dir, 'Checkpoints')
        if os.path.isdir(ckpt_dir):
            identifier = '_e' if self.epoch_based else '_i'
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
        start_iteration = checkpoint['iteration'] + 1
        monitor = checkpoint['monitor']
        model_state_dict = checkpoint['state_dict']
        optim_state_dict = checkpoint['optimizer']

        return start_iteration, model_state_dict, optim_state_dict, monitor

    def _log_progress(self):
        """
        Log training progress according to specified training scheme.
        """

        if self.epoch_based:

            # Get number of current epoch
            current_epoch = (self._it + 1) // \
                            len(self._data_loaders['train']) + 1

            # Get current iteration within epoch
            epoch_it = (self._it + 1) % len(self._data_loaders['train'])

            # Set progress string
            prog_string = f'#### Epoch: {current_epoch} | ' \
                          f'Iteration: {epoch_it}/' \
                          f'{len(self._data_loaders["train"])} ####'
        else:
            # Set progress string
            prog_string = f'#### Iteration: {self._it + 1}/' \
                          f'{self._num_iterations} ####'

        # Log string
        self._logger.log_string(prog_string)