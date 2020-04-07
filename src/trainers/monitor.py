from collections import namedtuple

# Flags to influence training flow
MonitorFlags = namedtuple('MonitorFlags', ['save_checkpoint',
                                           'new_best_model',
                                           'end_training'])


class Monitor:
    """
    Class to monitor training process. Handles checkpoint saving and early
    stopping.
    :param cfg: config file
    :param batches_per_epoch: number of batches per epoch
    """

    # Logging period in iterations
    LOG_FREQ = 10

    def __init__(self, cfg, batches_per_epoch):

        # Set config
        self._cfg = cfg

        # Set number of batches per epoch (number of batches in dataset)
        self.batches_per_epoch = batches_per_epoch

        # Set counter of non-improved validation steps
        self._counter = 0

        # Initialization best state for model saving
        self._best_score = None
        self._best_iteration = None

        # Initialize iteration to zero
        self.it = 0

        # Convert all relevant scheduling parameters to iteration-base
        self.num_iterations, self._ckpt_freq, self._val_freq, \
            self.epoch_based = self._get_scheduling_params(cfg)

        # Set early stopping patience if specified
        # Default: No early stopping
        self._patience = self._cfg['Train'].get('patience',
                                                self.num_iterations)

        # Initialize monitor flags
        self.flags = MonitorFlags(False, False, False)

    def __call__(self, score):

        # Initialize monitor flags
        save_checkpoint = False
        new_best_model = False
        stop_training = False

        # Save checkpoint according to specified frequency
        if ((self.it + 1) % self._ckpt_freq) == 0:
            save_checkpoint = True

        # First call
        if self._best_score is None:
            self._best_score = score
            self._best_iteration = self.it
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
            self._best_iteration = self.it
            new_best_model = True

            # Reset counter
            self.counter = 0

        # Check if validation phase left before end of scheduled training
        if (self.it + 1) > (self.num_iterations - self._val_freq):
            stop_training = True

        # Update monitor flags
        self.flags = MonitorFlags(save_checkpoint, new_best_model,
                                  stop_training)

    def do_validation(self):
        """
        Trigger validation.
        :return: True/False
        """
        return (self.it + 1) % self._val_freq == 0

    def do_logging(self):
        """
        Trigger logging.
        :return: True/False
        """

        if self.epoch_based:
            _, epoch_it = self.i2e()
            return epoch_it % self.LOG_FREQ == 0
        else:
            return (self.it + 1) % self.LOG_FREQ == 0

    def update(self):
        """
        Update state of the monitor
        """

        if not self.flags.end_training:
            self.it += 1

    def _get_scheduling_params(self, cfg):

        # Read from config file
        num_epochs = cfg['Train'].get('num_epochs', None)
        num_iterations = cfg['Train'].get('num_iterations', None)
        checkpoint_freq = cfg['Train'].get('checkpoint_freq', None)
        val_freq = cfg['Train'].get('val_freq', None)

        # Check for missing information
        if num_epochs is None and num_iterations is None:
            print('Error: Please specify the training duration using '
                  'num_epochs or num_iterations in config file.')
            exit(-1)
        if num_epochs is not None and num_iterations is not None:
            print('Error: Training duration is ambiguous, num_epochs '
                  'and num_iterations are specified in config file.')
            exit(-1)

        # Set unit for training
        if num_iterations is None:
            # epoch-based training
            num_iterations = self.batches_per_epoch * num_epochs
            checkpoint_freq = num_iterations if checkpoint_freq is None \
                else self.batches_per_epoch * checkpoint_freq
            val_freq = self.batches_per_epoch if val_freq is None \
                else self.batches_per_epoch * val_freq
            epoch_based = True
        else:
            # iteration-based training
            epoch_based = False

        return num_iterations, checkpoint_freq, val_freq, epoch_based

    def i2e(self, iteration=None):
        """
        Convert number of iterations to epoch-based representation
        :return: current epoch and iteration in epoch
        """
        if iteration is None:
            iteration = self.it

        # Get number of current epoch
        epoch = iteration // self.batches_per_epoch + 1

        # Get current iteration within epoch
        epoch_it = (iteration % self.batches_per_epoch) + 1

        return epoch, epoch_it

    def summary_string(self):
        """
        Log a summary string
        """

        if self.epoch_based:
            # Get number of current epoch and iteration
            epoch, epoch_it = self.i2e()

            # Set progress string
            prog_string = f'Epoch: {epoch} | Iteration: {epoch_it}/' \
                          f'{self.batches_per_epoch}'

            # Set best model string
            best_epoch, best_epoch_it = self.i2e(self._best_iteration)
            if best_epoch_it != self.batches_per_epoch:
                epoch_string = f'{best_epoch-1} epochs and {best_epoch_it} ' \
                               f'iterations '
            else:
                epoch_string = f'{best_epoch} epochs '
            best_model_string = f'Best model trained for ' \
                                + epoch_string + \
                                f'({self._best_iteration + 1} ' \
                                f'iterations in total) and achieved score of' \
                                f' {self._best_score:.4f}.\n'
        else:
            # Set progress string
            prog_string = f'Iteration: {self.it + 1}/{self.num_iterations}'

            # Set best model string
            best_model_string = f'Best model trained for ' \
                                f'{self._best_iteration + 1} iterations ' \
                                f'achieved score of {self._best_score:.4f}.\n'

        if self._counter > self._patience:
            cause_of_stop = f"Training stopped because validation score " \
                            f"did not increase for {self._counter} " \
                            f"validation cycles.\n"
        else:
            cause_of_stop = "Training stopped because number of outstanding " \
                            "iterations went below the length of one " \
                            "validation period.\n"

        # Assemble output string
        summary_string = f"\nEnd of training ({prog_string})\n" + \
                         cause_of_stop + best_model_string

        return summary_string