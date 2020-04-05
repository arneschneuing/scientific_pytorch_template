from collections import namedtuple


class Monitor:
    """
    Early stops the training if validation loss doesn't improve after a given
    patience.
    """

    MonitorFlags = namedtuple('MonitorFlags', ['save_ckpt', 'new_best_model',
                                               'end_training'])

    def __init__(self, ckpt_freq, patience):
        """
        Class to monitor training process. Handles checkpoint saving and early
        stopping.
        :param ckpt_freq: frequency of checkpoint savings in iterations
        (default: None). If provided, checkpoint file-names will indicate
        current epoch instead of iteration.
        :param patience: number of permitted validation steps without score
        improvement before training is stopped.
        """

        self._patience = patience
        self._counter = 0
        self._ckpt_freq = ckpt_freq
        self._best_score = None

    def __call__(self, score, iteration):

        # Initialize monitor flags
        save_checkpoint = False
        new_best_model = False
        stop_training = False

        # Save checkpoint according to specified frequency
        if ((iteration + 1) % self._ckpt_freq) == 0:
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

        # validation loss decreased
        else:
            self._best_score = score
            new_best_model = True

            # Reset counter
            self.counter = 0

        return self.MonitorFlags(save_checkpoint, new_best_model,
                                 stop_training)