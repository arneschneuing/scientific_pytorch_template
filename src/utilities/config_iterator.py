from copy import deepcopy


class CfgIterator:
    """
    Iterate over different configurations defined by a set of parameter lists
    and corresponding keys.
    :param base_cfg: Base configuration from which selected parameters are
        modified
    :param param_combinations: Parameter lists
    :param param_keys: List of key tuples defining configurations parameters
    :param start_id: ID of the first parameter combination
    """
    def __init__(self, base_cfg, param_combinations, param_keys, start_id=0):
        self._base_cfg = base_cfg
        # TODO: self._cfg = deepcopy(self._base_cfg) - relevant?
        self._comb = param_combinations
        self._keys = param_keys
        self._k = start_id

    def __iter__(self, k=0):
        return self

    def __next__(self):

        # Iterate over parameter combinations
        if self._k < len(self._comb):
            # Return base config if there is only a single parameter
            # combination
            if len(self._comb) > 1:

                # Re-initialize config file
                self._cfg = deepcopy(self._base_cfg)

                # Update all variable parameters
                for i in range(len(self._keys)):
                    self._set_in_dict(self._cfg, self._keys[i],
                                      self._comb[self._k][i])

            # Increment counter
            self._k += 1

            return self._cfg
        else:
            raise StopIteration

    @staticmethod
    def _set_in_dict(data_dict, key_list, value):
        for key in key_list[:-1]:
            data_dict = data_dict[key]

        # Remove trailing underscore and assign value
        data_dict[key_list[-1].strip("_")] = value
        # Delete hyper-parameter list
        del data_dict[key_list[-1]]
