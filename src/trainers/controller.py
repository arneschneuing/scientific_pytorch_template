import yaml
import os
from itertools import product
from src.utilities.config_iterator import CfgIterator
from src.trainers.trainer import Trainer


class Controller:
    def __init__(self, cfg_path, result_dir, setup):
        self._experiment_id = self._create_folder_structure(result_dir, setup)
        self._cfg = self._read_config_file(cfg_path)
        self._experiment_cfgs = self._split_config()

    @staticmethod
    def _read_config_file(cfg_path):
        # Read-in cfg file
        with open(cfg_path, 'r') as f:
            cfg = yaml.safe_load(f)
        return cfg

    @staticmethod
    def _create_folder_structure(result_dir, setup):

        # Create result dir
        os.makedirs(result_dir, exist_ok=True)

        # Check if setup dir already exists
        setup_path = os.path.join(result_dir, setup)
        if os.path.isdir(setup_path):
                print(f'Directory {setup_path} already exists. Append? [y]/n')
                c = input()
                if c == 'n':
                    print('Exiting...')
                    exit()
                elif c == 'y' or c == '':
                    print(f'Append experiments to {setup_path}.')

        else:
            os.makedirs(setup_path)
            print(f'Create directory {setup_path}.')

        return len(os.listdir(setup_path)) + 1

    def _split_config(self):
        """
        Split setup-level cfg dict into experiment-level cfg dicts.
        :return: list of experiment-level cfg dicts
        """

        def get_param_lists(cfg_dict, param_lists=None, param_keys=None,
                            prefix=None):
            """
            Recursively get a list of all parameters for which a list of values
            was specified in the cfg file. Required to construct all
            possible parameter permutations.
            :param cfg_dict: dict from which to extract parameter lists
            :param param_lists: list to which to append parameter lists
            :param param_keys: list to which to append key tuple
            :param prefix: prefix key to indicate higher level dict keys
            :return: param_lists, param_keys
            """

            if param_lists is None and param_keys is None:
                param_lists, param_keys = [], []

            for param_name, param_value in cfg_dict.items():

                # Append list of parameter values to param_lists
                # Append tuple of necessary keys to access param in original
                # cfg dict
                if isinstance(param_value, list) and param_name[-1] == "_":
                    param_lists.append(param_value)
                    if prefix is not None:
                        param_keys.append(prefix + tuple([param_name]))
                    else:
                        param_keys.append(tuple([param_name]))

                # Call function recursively to extract parameter lists at
                # arbitrary levels of the cfg dict
                if isinstance(param_value, dict):
                    if prefix is None:
                        sub_prefix = tuple([param_name])
                    else:
                        sub_prefix = prefix + tuple([param_name])

                    param_lists, param_keys = get_param_lists(param_value,
                                                              param_lists,
                                                              param_keys,
                                                              sub_prefix)
            return param_lists, param_keys

        param_lists, param_keys = get_param_lists(self._cfg)
        combinations = list(product(*param_lists))

        # Return configurations iterator object starting at the current
        # experiment ID
        return CfgIterator(self._cfg, combinations, param_keys,
                           self._experiment_id - 1)

    def start(self):
        for cfg in self._experiment_cfgs:
            experiment_dir = f"Experiment {self._experiment_id}"
            Trainer(experiment_dir, model, criterion, metric_tracker,
                    optimizer, cfg, data_loaders).train()
            self._experiment_id += 1
