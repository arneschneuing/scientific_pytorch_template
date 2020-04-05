from collections import namedtuple


class Monitor:
    """
    Class to monitor training process. Handles checkpoint saving and early
    stopping.
    :param cfg: config file
    :param batches_per_epoch: number of batches per epoch
    :param log_fn: function used for logging
    """

    LOG_FREQ = 10
    MonitorFlags = namedtuple('MonitorFlags', ['save_ckpt', 'new_best_model',
                                               'end_training'])

    def __init__(self, cfg, batches_per_epoch, log_fn):

        self._cfg = cfg
        self._batches_per_epoch = batches_per_epoch
        self._log_fn = log_fn

        self.it = 0
        self._patience = self._cfg['Monitor'].get('patience',
                                                  default=self._num_iterations)
        self._counter = 0
        self._best_score = None
        self._best_iteration = None

        self._num_iterations, self._ckpt_freq, self._val_freq, \
            self.epoch_based = self._get_scheduling_params(cfg)

    def __call__(self, score):

        # Initialize monitor flags
        save_checkpoint = False
        new_best_model = False
        stop_training = False

        # Inform user about evaluation status
        if self.epoch_based:
            epochs_trained = (self.it + 1) // self._batches_per_epoch
            self._log_fn(f'#### Evaluation ({epochs_trained} Epochs trained)')
        else:
            self._log_fn(f'#### Evaluation ({self.it + 1} Iterations trained)')

        # Save checkpoint according to specified frequency
        if ((self.it + 1) % self._ckpt_freq) == 0:
            save_checkpoint = True

        # First call
        if self._best_score is None:
            self._best_score = score
            new_best_model = True

        # Score didn't increase
        elif score <= self._best_score:

            # Increment counter
            self._counter += 1

            # Stop training if patience is reached
            if self._counter > self._patience:
                stop_training = True

        # Validation score increased
        else:
            self._best_score = score
            new_best_model = True

            # Reset counter
            self.counter = 0

        # Check if validation phase left before end of scheduled training
        if (self.it + 1) > (self._num_iterations - self._val_freq):
            stop_training = True

        return self.MonitorFlags(save_checkpoint, new_best_model,
                                 stop_training)

    def update(self, step=1):
        """
        Update state of the monitor
        :param step: update step size
        """
        # Log Progress
        if (self.it + 1) % self.LOG_FREQ == 0:
            self._log_progress()

        self.it += 1

    def _get_scheduling_params(self, cfg):
        # Read from config file
        num_epochs = cfg['Train'].get('num_epochs', default=None)
        num_iterations = cfg['Train'].get('num_iterations', default=None)
        checkpoint_freq = cfg['Train'].get('checkpoint_freq', default=None)
        val_freq = cfg['Train'].get('val_freq', default=None)

        if num_epochs is None and num_iterations is None:
            self._log_fn('Error: please specify the training duration using '
                         'num_epochs or num_iterations in config file')
            exit(-1)
        if num_epochs is not None and num_iterations is not None:
            self._log_fn('Error: training duration is ambiguous, num_epochs and '
                         'num_iterations are specified in config file')
            exit(-1)

        if num_iterations is None:
            # epoch-based training
            num_iterations = self._batches_per_epoch * num_epochs
            checkpoint_freq = num_iterations if checkpoint_freq is None \
                else self._batches_per_epoch * checkpoint_freq
            val_freq = self._batches_per_epoch if val_freq is None \
                else self._batches_per_epoch * val_freq
            epoch_based = True
        else:
            # iteration-based training
            epoch_based = False

        return num_iterations, checkpoint_freq, val_freq, epoch_based

    def _log_progress(self):
        """
        Log training progress according to specified training scheme.
        """

        if self.epoch_based:

            # Get number of current epoch
            current_epoch = (self.it + 1) // self._batches_per_epoch + 1

            # Get current iteration within epoch
            epoch_it = (self.it + 1) % self._batches_per_epoch

            # Set progress string
            prog_string = f'#### Epoch: {current_epoch} | ' \
                          f'Iteration: {epoch_it}/' \
                          f'{self._batches_per_epoch} ####'
        else:
            # Set progress string
            prog_string = f'#### Iteration: {self.it + 1}/' \
                          f'{self._num_iterations} ####'

        # Log string
        self._log_fn(prog_string)