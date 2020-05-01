import csv
import yaml
from itertools import product
from shutil import copyfile, rmtree
from collections import OrderedDict
from src.utilities.template_utils import *
from src.core_components.trainer import Trainer


class Controller:
    """
    Controller class to handle the scheduling of one or more training runs.
    The controller takes a path to a config file that can contain parameter
    lists specifying values for several runs, e.g. for hyper-parameter
    optimization. Single training runs will be started for each parameter
    configuration.
    """
    def __init__(self, cfg_path, result_dir, session, result_filename,
                 overwrite):
        """
        :param cfg_path: string | relative path to the YAML config file
        :param result_dir: string | relative path to the result directory where
        logs, checkpoints and other outputs will be saved
        :param session: string | name of the session
        :param result_filename: string | filename of csv file for result saving
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

        # Set path to result file
        self._result_filename = os.path.join(self._session_path,
                                             result_filename)

        # Copy main configuration file
        cfg_copy_path = os.path.join(self._session_path,
                                     os.path.basename(cfg_path))
        copyfile(cfg_path, cfg_copy_path)

        # Copy code to session folder
        if flatten_cfg(self._cfg).get('copy_code', False):
            try:
                code_copy_path = os.path.join(self._session_path, 'code')
                copy_code(code_copy_path)
            except FileExistsError:
                print("Code folder already exists.")

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
                elif not isinstance(param_item, list) and param_key[-1] == "_":
                    print(f'Error in {self._config_path}. Parameters with '
                          f'trailing underscore must be lists. Exiting...')
                    exit()

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
        print(f'Found {len(combinations)} parameter combination(s) in config '
              f'file {self._config_path}:')

        # Print information about all experiments if more than a one
        if len(combinations) > 1:
            self._print_experiment_overview(param_keys, combinations)

        print('Start experiments? [y]/n')
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

    @staticmethod
    def _print_experiment_overview(param_keys, combinations):

        # Set desired number of empty spaces after each parameter
        n_empty_spaces = 5

        # Get maximum column width for each parameter
        col_widths = [0] * len(param_keys)
        for combination_id, combination in enumerate(combinations):
            for param_id in range(len(param_keys)):
                name_width = len(str(param_keys[param_id][-1][:-1]))
                value_width = len(str(combination[param_id]))
                col_widths[param_id] = max(name_width, value_width,
                                           col_widths[param_id])
        col_widths = [col_width + n_empty_spaces for col_width in col_widths]

        # Print parameters for each combination
        for exp_id, combination in enumerate(combinations):

            # Print header string in first iteration
            if exp_id == 0:
                header_string = f'-' * (sum(col_widths) + 10) + f'\n'
                header_string += f'{"Experiment":<{10 + n_empty_spaces}}'
                for param_id in range(len(param_keys)):
                    header_string += \
                        f'{str(param_keys[param_id][-1][:-1]):<{col_widths[param_id]}}'
                header_string += f'\n' + f'-' * (sum(col_widths) + 10)
                print(header_string)

            # Concatenate information about all parameters for current
            # experiment
            param_string = f'{exp_id + 1:<{10 + n_empty_spaces}}'
            for param_id in range(len(param_keys)):
                param_string += \
                    f'{str(combination[param_id]):<{col_widths[param_id]}}'
            print(param_string)

    def write_result_file(self, result_dict):
        """
        Write training results to csv file.
        :param result_dict: dict containing results of finished training
        """

        # Create file dict
        file_dict = OrderedDict()

        # Add current experiment ID
        file_dict['experiment'] = self._experiment_id

        # Add parameters of current experiment
        for param_id in range(len(self._experiment_cfgs._keys)):
            param_name = self._experiment_cfgs._keys[param_id][-1][:-1]
            comb = self._experiment_cfgs._comb[self._experiment_id-1]
            param_value = comb[param_id]
            file_dict[param_name] = param_value

        # Add result dict to file dict
        file_dict.update(result_dict)

        # Open result file. Create new file for first experiment
        if self._experiment_id == 1:
            f = open(self._result_filename, "w")
        else:
            f = open(self._result_filename, "a+")

        # Create header
        fieldnames = list(file_dict.keys())

        # Create writer
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator='\n')

        # Writer header for first experiment
        if self._experiment_id == 1:
            writer.writeheader()

        # Add current results to file
        writer.writerow(file_dict)

        f.close()

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
            result_dict = Trainer(experiment_path, cfg).train()

            # Add training results to session-level result file
            self.write_result_file(result_dict)

            # Increment experiment counter
            self._experiment_id += 1
