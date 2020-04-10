import yaml
import os
from shutil import copyfile, rmtree
from itertools import product
from src.utilities.config_iterator import CfgIterator
from src.trainers.trainer import Trainer
from src.utilities.util import get_latest_version, copy_code


class Controller:
    """
    Controller class to handle the scheduling of one or more training runs.
    The controller takes a path to a config file that can contain parameter
    lists specifying values for several runs, e.g. for hyper-parameter
    optimization. Single training runs will be started for each parameter
    configuration.
    """
    def __init__(self, cfg_path, result_dir, session, overwrite):
        """
        :param cfg_path: string | relative path to the YAML config file
        :param result_dir: string | relative path to the result directory where
        logs, checkpoints and other outputs will be saved
        :param session: string | name of the session
        :param overwrite: bool | overwrite session directory
        """

        # Get ID of next experiment to be performed
        self._experiment_id, self._session_path = \
            self._create_folder_structure(result_dir, session, overwrite)

        # Set config file for current session
        self._config_path = cfg_path
        self._cfg = self._read_config_file(cfg_path)

        # Extract config files for each parameter configuration in current
        # session
        self._experiment_cfgs = self._split_config()

        # Copy main configuration file
        cfg_copy_path = os.path.join(self._session_path,
                                     os.path.basename(cfg_path))
        copyfile(cfg_path, cfg_copy_path)

        # Copy code to session folder
        if self._cfg.get('copy_code', False):
            code_copy_path = os.path.join(self._session_path, 'code')
            copy_code(code_copy_path)

    @staticmethod
    def _read_config_file(cfg_path):
        with open(cfg_path, 'r') as f:
            cfg = yaml.safe_load(f)
        return cfg

    @staticmethod
    def _create_folder_structure(result_dir, session, overwrite):
        """
        Create required directories for the saving of experimental results.
        :param result_dir: string | relative path to result directory
        :param session: string | name of the current session
        :param overwrite: bool | overwrite session directory
        :return:
            experiment_id: int | ID of the next experiment (1 if first)
            session_path: string | path to session directory
        """

        # Create result dir
        os.makedirs(result_dir, exist_ok=True)

        # Make session path
        session_path = os.path.join(result_dir, session)

        # Check if overwrite mode is turned on
        if overwrite:
            try:
                rmtree(session_path)
            except FileNotFoundError:
                # nothing to overwrite
                pass

        # Check if session dir already exists
        if os.path.isdir(session_path) and len(os.listdir(session_path)) > 0:
            print(f'Directory {session_path} already exists. Continue '
                  f'training? [y]/n')
            c = input()
            if c == 'y' or c == '':
                print(f'Continue experiments in {session_path}.')
            else:
                print('Exiting...')
                exit()

            # Get experiment ID as ID of last available experiment
            experiment_id = int(get_latest_version(session_path, 'experiment_')
                                .strip('experiment_'))

        else:

            # Create session directory
            os.makedirs(session_path, exist_ok=True)

            print(f'Create new session in directory {session_path}.')

            # Set initial experiment ID to 1
            experiment_id = 1

        return experiment_id, session_path

    def _split_config(self):
        """
        Split session-level cfg dict into experiment-level cfg dicts.
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

        # Verify combinations before starting training runs
        print(f'Found {len(combinations)} parameter combinations in config '
              f'file {self._config_path}. Start experiments? [y]/n')
        c = input()
        if c == '' or c == 'y':
            pass
        else:
            print('Exiting...')
            exit(-1)

        # Return configurations iterator object starting at the current
        # experiment ID
        return CfgIterator(self._cfg, combinations, param_keys,
                           self._experiment_id - 1)

    def start(self):
        """
        Start scheduling training runs.
        """

        # Iterate over config dicts for all parameter combinations specified in
        # session-level config file
        for cfg in self._experiment_cfgs:

            # Set path to result dir for current experiment
            experiment_dir = f"experiment_{self._experiment_id}"

            # Get full experiment path
            experiment_path = os.path.join(self._session_path, experiment_dir)

            print(f'Schedule experiment {self._experiment_id}!')

            # Start training run for current config
            Trainer(experiment_path, cfg).train()

            # Increment experiment counter
            self._experiment_id += 1
