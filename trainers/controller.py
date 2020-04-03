import yaml
import os
from itertools import product


class Controller:
    def __init__(self, cfg_path, result_dir, setup):
        self._experiment_id = self._create_folder_structure(result_dir, setup)
        self._cfg = self._read_config_file(cfg_path)
        self._experiment_cfgs = self._split_config()

    @staticmethod
    def _read_config_file(cfg_path):
        # Read-in config file
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
        Split setup-level config dict into experiment-level config dicts.
        :return: list of experiment-level config dicts
        """

        def get_param_lists(cfg_dict, param_lists=None, param_keys=None,
                            prefix=None):
            """
            Recursively get a list of all parameters for which a list of values
            was specified in the config file. Required to construct all
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
                # config dict
                if isinstance(param_value, list):
                    param_lists.append(param_value)
                    if prefix is not None:
                        param_keys.append(prefix + tuple([param_name]))
                    else:
                        param_keys.append(tuple([param_name]))

                # Call function recursively to extract parameter lists at
                # arbitrary levels of the config dict
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
        print(param_lists)
        print(param_keys)
        combinations = list(product(*param_lists))
        n_combinations = len(combinations)
        experiment_dicts = [self._cfg.copy() for _ in range(n_combinations)]

        # Iterate over all combinations
        for comb_id, comb in enumerate(combinations):

            # Get dict container for current combination
            tmp_dict = experiment_dicts[comb_id]

            # Iterate over all parameters
            for param_id, param_key in enumerate(param_keys):

                # Iterate over all dict levels
                tmp_dicts = []
                for level, level_key in enumerate(param_key):
                    # Get dict at current level
                    if level == (len(param_key) - 1):
                        tmp_dict[level_key] = comb[param_id]
                    else:
                        print(tmp_dict)
                        tmp_dict = tmp_dict[level_key]
                    tmp_dicts.append(tmp_dict)

            for tmp_dict_id in range(len(tmp_dicts)-1, 1, -1):
                tmp_dicts[tmp_dict_id-1].update(tmp_dicts[tmp_dict_id])

            # Update experiment config for current combination
            #experiment_dicts[comb_id].update(tmp_dicts[0])

        for exp_dict in experiment_dicts:
            print(exp_dict)