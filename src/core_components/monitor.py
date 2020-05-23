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

    def __init__(self, cfg, batches_per_epoch):

        # Set config
        self._cfg = cfg

        # Set number of batches per epoch (number of batches in dataset)
        self.batches_per_epoch = batches_per_epoch

        # Set counter of non-improved validation steps
        self.counter = 0

        # Initialization best state for model saving
        self._best_score = None
        self._best_iteration = 0

        # Initialize iteration to zero
        self.it = 0

        # Convert all relevant scheduling parameters to iteration-base
        self.num_iterations, self._ckpt_freq, self._val_freq, self._lr_freq, \
            self._log_freq, self.epoch_based = self._get_scheduling_params(cfg)

        # Set early stopping patience if specified
        # Default: No early stopping
        self.patience = self._cfg.get('patience', self.num_iterations)

        # Initialize monitor flags
        end_training = (self.num_iterations - self._val_freq < 0)
        self.flags = MonitorFlags(False, False, end_training)

        # Initialize test score (only used for summary printing if
        # "eval_after_training" is set)
        self._test_score = None

    def register_val_result(self, score):

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
            self.counter += 1

            # Stop training if patience is reached
            if self.counter >= self.patience:
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

    def do_logging(self, it=None):
        """
        Trigger logging.
        :param it: current iteration (e.g. within epoch) | default: None
        :return: True/False
        """

        # Use global iteration counter for logging during training
        if it is None:

            # Use inter-epoch iteration for logging if epoch-based training
            if self.epoch_based:
                _, epoch_it = self.i2e()
                return epoch_it % self._log_freq == 0
            # Use global iteration for logging if iteration-based training
            else:
                return (self.it + 1) % self._log_freq == 0

        # Provide current iteration during evaluation
        else:
            return it % self._log_freq == 0

    def do_lr_step(self):
        """
        Trigger learning rate scheduler step.
        :return: True/False
        """
        return (self.it + 1) % self._lr_freq == 0

    def step(self):
        """
        Update state of the monitor.
        """
        self.it += 1

    def _get_scheduling_params(self, cfg):
        """
        Extract all scheduling-related parameters from the config file and
        convert them to iterations since training is performed on an iteration
        basis.
        :param cfg: config dict
        :return:
            num_iterations: maximum number of training iterations
            checkpoint_freq: checkpoint frequency in iterations
            val_freq: validation frequency in iterations
            epoch_based: Flag indicating if training is to be performed in an
                epoch-based fashion
        """

        # Read from config file
        num_epochs = cfg.get('num_epochs', None)
        num_iterations = cfg.get('num_iterations', None)
        checkpoint_freq = cfg.get('checkpoint_freq', None)
        val_freq = cfg.get('val_freq', None)
        log_freq = cfg.get('log_freq', None)
        lr_freq = cfg.get('lr_freq', None)

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
            epoch_based = True
            num_iterations = self.batches_per_epoch * num_epochs
            checkpoint_freq = num_iterations if checkpoint_freq is None \
                else self.batches_per_epoch * checkpoint_freq
            val_freq = self.batches_per_epoch if val_freq is None \
                else self.batches_per_epoch * val_freq
            lr_freq = self.batches_per_epoch if lr_freq is None \
                else self.batches_per_epoch * lr_freq
        else:
            # iteration-based training
            epoch_based = False
            checkpoint_freq = num_iterations if checkpoint_freq is None \
                else checkpoint_freq
            val_freq = num_iterations if val_freq is None else val_freq
            lr_freq = 1 if lr_freq is None else lr_freq

        # Set logging frequency to 10 if no value provided
        if log_freq is None:
            log_freq = 10

        # Check validity of provided frequencies
        assert checkpoint_freq % val_freq == 0, 'Checkpoint frequency ' \
                                                'has to be an integer ' \
                                                'multiple of the ' \
                                                'validation frequency!'

        return num_iterations, checkpoint_freq, val_freq, lr_freq, log_freq, \
               epoch_based

    def i2e(self, iteration=None):
        """
        Convert number of iterations to epoch-based representation.
        :return: current epoch and inter-epoch iteration
        """
        if iteration is None:
            iteration = self.it

        # Get number of current epoch
        epoch = iteration // self.batches_per_epoch + 1

        # Get current iteration within epoch
        epoch_it = (iteration % self.batches_per_epoch) + 1

        return epoch, epoch_it

    def register_test_result(self, score):
        """
        Register test result for summary printing.
        :param score: test score
        """
        self._test_score = score

    def early_stopping(self):
        """
        Return True if early stopping is enabled.
        """
        early_stopping = True if self.patience < self.num_iterations \
            else False

        return early_stopping

    def result_dict(self):
        """
        Create dict containing the final scores.
        """

        result_dict = {'val_score': self._best_score}

        if self._test_score is not None:
            result_dict['test_score'] = self._test_score

        return result_dict

    def summary_string(self):
        """
        Create a summary string
        """

        # Set score string
        if self._test_score is not None:
            score_string = f'and achieved a score ' \
                           f'of Val: {self._best_score:.4g} | ' \
                           f'Test: {self._test_score:.4g}.\n'
        else:
            score_string = f'and achieved a score ' \
                           f'of {self._best_score:.4g}.\n'

        # Set summary string
        if self.epoch_based:
            # Get number of current epoch and iteration
            if self.it == 0:
                epoch, epoch_it = 0, 0
            else:
                epoch, epoch_it = self.i2e(self.it - 1)

            # Set progress string
            prog_string = f'Epoch: {epoch} | Iteration: {epoch_it}/' \
                          f'{self.batches_per_epoch}'

            # Set best model string
            best_epoch, best_epoch_it = self.i2e(self._best_iteration)
            if best_epoch_it != self.batches_per_epoch:
                epoch_string = f'{best_epoch-1} epoch(s) and {best_epoch_it} ' \
                               f'iteration(s) '
            else:
                epoch_string = f'{best_epoch} epoch(s) '
            best_model_string = f'Best model trained for ' \
                                + epoch_string + \
                                f'({self._best_iteration + 1} ' \
                                f'iteration(s) in total) ' + score_string
        else:
            # Set progress string
            prog_string = f'Iteration: {self.it}/{self.num_iterations}'

            # Set best model string
            if self.it == 0:
                best_iteration = 0
            else:
                best_iteration = self._best_iteration + 1
            best_model_string = f'Best model trained for ' \
                                f'{best_iteration} iterations ' \
                                + score_string

        if self.counter > self.patience:
            cause_of_stop = f"Training stopped because validation score " \
                            f"did not increase for {self.counter} " \
                            f"validation cycles.\n"
        else:
            if self.it == self.num_iterations:
                cause_of_stop = ''
            else:
                cause_of_stop = "Training stopped because number of " \
                                "remaining iterations is less than " \
                                "length of one validation period.\n"

        # Assemble output string
        summary_string = f"\nEnd of training ({prog_string})\n" + \
                         cause_of_stop + best_model_string

        return summary_string