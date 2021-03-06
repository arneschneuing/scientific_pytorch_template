# Code adapted from:
# https://github.com/victoresque/pytorch-template/blob/master/logger
# /visualization.py

import os
try:
    from tensorboardX import SummaryWriter
    TB_AVAILABLE = True
except ModuleNotFoundError:
    try:
        from torch.utils.tensorboard import SummaryWriter
        TB_AVAILABLE = True
    except ImportError:
        TB_AVAILABLE = False
        print("Warning: TensorBoard not available.")


class Logger:
    def __init__(self, log_dir, cfg):
        self.log_dir = log_dir

        self.write_file = cfg.get('write_file', True)
        self.write_tb = cfg.get('write_tb', False)

        # Create log_dir
        os.makedirs(self.log_dir, exist_ok=True)

        # Create TB log_dir if tensorboard logging enabled
        if self.write_tb:

            if TB_AVAILABLE:
                tb_log_dir = os.path.join(self.log_dir, 'tensorboard_logs')
                os.makedirs(tb_log_dir, exist_ok=True)

                # Create tensorboard writer
                self.tb = TensorboardWriter(tb_log_dir)

            else:
                print("Error: TensorBoard is configured to be used but it is "
                      "not available.")
                exit(-1)

        # Create log file if file logging enabled
        if self.write_file:
            self.log_file = open(os.path.join(self.log_dir, 'log_train.txt'),
                                 'a')

    def log_string(self, log_str):
        """
        Write string to log file and print to console.
        """
        if self.write_file:
            self.log_file.write(log_str + '\n')
            self.log_file.flush()
        print(log_str)

    def close(self):

        # Close txt file
        if self.write_file:
            self.log_file.close()

        # Close tensorboard loggers
        if self.write_tb:
            for tb_writer in self.tb.writers.values():
                tb_writer.close()


class TensorboardWriter:
    def __init__(self, log_dir):
        """
        Tensorboard Writer containing two separate SummaryWriters to allow for
        logging of training and validation metrics.
        :param log_dir: string | relative path to tensorboard logging directory
        """

        # Create tb writer for train mode
        tb_train_dir = os.path.join(log_dir, 'train')
        os.makedirs(tb_train_dir, exist_ok=True)
        self.writers = {'train': SummaryWriter(tb_train_dir)}

        # Create tb writer for val mode
        tb_val_dir = os.path.join(log_dir, 'val')
        os.makedirs(tb_val_dir, exist_ok=True)
        self.writers['val'] = SummaryWriter(tb_val_dir)

        # Set tb writer mode
        self._mode = 'train'

    def __getattr__(self, name):
        """
        Return respective tensorboard function if available.
        Otherwise, return empty function handle that does nothing.
        """

        # Get tb function handle
        add_data = getattr(self.writers[self._mode], name, None)

        # Check if requested function is supported
        if add_data is not None:
            return add_data

        else:
            # Default action for returning attributes defined in this class
            try:
                attr = self.__getattribute__(name)
            except AttributeError:
                raise AttributeError(
                    "Type object '{}' has no attribute '{}'".format(
                        self.__class__.__name__, name))
            return attr

    def train(self):
        """
        Set TB logger mode to train.
        """
        self._mode = 'train'

    def val(self):
        """
        Set TB logger mode to val.
        """
        self._mode = 'val'

    def flush(self):
        for tb_writer in self.writers.values():
            tb_writer.flush()


if __name__ == '__main__':

    logger = Logger('results',
                    {'Logging': {'write_file': True, 'write_tb': True}})
    max_iter = 100
    for i in range(max_iter):
        logger.log_string(f'Iteration: {i}')
        logger.tb.train()
        logger.tb.add_scalar(tag='Iteration', data=i, step=i)
        logger.tb.val()
        logger.tb.add_scalar(tag='Iteration', data=i, step=i)
