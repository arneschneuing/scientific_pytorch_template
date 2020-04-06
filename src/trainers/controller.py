import yaml
import os
from itertools import product
from src.utilities.config_iterator import CfgIterator
from src.trainers.trainer import Trainer


class Controller:
    """
    Controller class to handle the scheduling of one or more training runs.
    The controller takes a path to a config file that can contain parameter
    lists specifying values for several runs, e.g. for hyper-parameter
    optimization. Single training runs will be started for each parameter
    configuration.
    """
    def __init__(self, cfg_path, result_dir, setup=None):
        """
        :param cfg_path: string |relative path to the YAML config file
        :param result_dir: string | relative path to the result directory where
        logs, checkpoints and other outputs will be saved
        :param setup: string | name of the setup to continue experiments |
        Default: None -> Create new setup directory
        """

        # Get ID of next experiment to be performed
        self._experiment_id = self._create_folder_structure(result_dir, setup)

        # Set config file for current setup
        self._cfg = self._read_config_file(cfg_path)

        # Extract config files for each parameter configuration in current
        # setup
        self._experiment_cfgs = self._split_config()

    @staticmethod
    def _read_config_file(cfg_path):
        # Read-in cfg file
        with open(cfg_path, 'r') as f:
            cfg = yaml.safe_load(f)
        return cfg

    @staticmethod
    def _create_folder_structure(result_dir, setup):
        """
        Create required directories for the saving of experimental results.
        :param result_dir: string | relative path to result directory
        :param setup: string | name of the current setup
        :return:
        """

        # Create result dir
        os.makedirs(result_dir, exist_ok=True)

        # Check if setup dir already exists
        setup_path = os.path.join(result_dir, setup)
        if os.path.isdir(setup_path):
            print(f'Directory {setup_path} already exists. Continue '
                  f'training? [y]/n')
            c = input()
            if c == 'n':
                print('Exiting...')
                exit()
            elif c == 'y' or c == '':
                print(f'Continue experiments in {setup_path}.')

        else:
            os.makedirs(setup_path)
            print(f'Create new setup in directory {setup_path}.')

        return len(os.listdir(setup_path)) + 1

    def _split_config(self):
        """
        Split setup-level cfg dict into experiment-level cfg dicts.
        :return: iterator of experiment-level cfg dicts
        """

        def get_param_lists(cfg_dict, param_lists=None, param_keys=None,
                            key_prefix=None):
            """
            Recursively get a list of all parameters for which a list of values
            was specified in the cfg file. Required to construct all
            possible parameter permutations.
            :param cfg_dict: dict from which to extract parameter lists
            :param param_lists: list to which to append parameter lists
            :param param_keys: list to which to append key tuple
            :param key_prefix: prefix key to indicate higher level dict keys
            :return: param_lists, param_keys
            """

            # Create empty lists on first call
            if param_lists is None and param_keys is None:
                param_lists, param_keys = [], []

            # Iterate over all first-level items of the current dict
            for param_key, param_item in cfg_dict.items():

                # Append list of parameter values to param_lists
                # Append tuple of necessary keys to access param in original
                # cfg dict
                if isinstance(param_item, list) and param_key[-1] == "_":
                    param_lists.append(param_item)
                    if key_prefix is not None:
                        param_keys.append(key_prefix + tuple([param_key]))
                    else:
                        param_keys.append(tuple([param_key]))

                # Call function recursively to extract parameter lists at
                # arbitrary levels of the cfg dict
                if isinstance(param_item, dict):
                    if key_prefix is None:
                        sub_prefix = tuple([param_key])
                    else:
                        sub_prefix = key_prefix + tuple([param_key])

                    param_lists, param_keys = get_param_lists(param_item,
                                                              param_lists,
                                                              param_keys,
                                                              sub_prefix)
            return param_lists, param_keys

        # Get list of all parameter lists specified in the config file
        # Note: Parameter lists for separate runs have to contain a trailing
        # underscore.
        param_lists, param_keys = get_param_lists(self._cfg)

        # Get all possible parameter combinations from the parameter lists
        combinations = list(product(*param_lists))

        # Return configurations iterator object starting at the current
        # experiment ID
        return CfgIterator(self._cfg, combinations, param_keys,
                           self._experiment_id - 1)

    def start(self):
        """
        Start scheduling training runs.
        """

        # Iterate over config dicts for all parameter combinations specified in
        # setup-level config file
        for cfg in self._experiment_cfgs:

            # Set path to result dir for current experiment
            experiment_dir = f"experiment_{self._experiment_id}"

            # Start training run for current config
            Trainer(experiment_dir, cfg).train()

            print(f'Schedule Experiment {self._experiment_id}!')

            # Increment experiment counter
            self._experiment_id += 1
