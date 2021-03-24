import os
import re
import shutil
import fnmatch
from copy import deepcopy


def get_number(s, token):
    """
    Get number that follows token
    :param s: string
    :param token: token string
    :return: number
    """
    match = re.search(rf'{token}\d+', s)
    if match:
        return int(match.group().strip(token))
    else:
        return None


def get_latest_version(dir, token):
    """
    Get latest version of a file or folder as indicated by the number following
    "token" in the file's name
    :param dir: directory to search
    :param token: token string
    :return: name of the latest file/folder version
    """
    curr_max_count = 0
    curr_latest = None
    for filename in os.listdir(dir):
        count = get_number(filename, token)
        if count is None:
            continue
        if count > curr_max_count:
            curr_max_count = count
            curr_latest = filename
    return curr_latest


def get_components(path):
    """
    Helper function to split path into its components
    :param path: input path
    :return: list of components
    """
    components = []
    head, tail = os.path.split(path)
    while tail != '':
        components.append(tail)
        head, tail = os.path.split(head)
    return components


def safe_copytree(src, dst, ignore_list=None):
    """
    Perform shutil.copytree only when it is safe to do so.
    Recursively copying the entire directory tree rooted at src could result in
    an infinite recursion if dst is contained in src.
    :param src: path of the source directory
    :param dst: path of the destination
    :param ignore_list: list of items to ignore while copying the tree
    :return: path of the new directory
    """
    # Get the canonical paths
    src_path = os.path.realpath(src)
    dst_path = os.path.realpath(dst)

    # Get the longest common sub-path
    common_path = os.path.commonpath([src_path, dst_path])

    # Check if source directory is parent of destination directory
    src_is_parent = (src_path == common_path)

    # Determine whether dst will be ignored while copying
    if ignore_list is None:
        dst_is_ignored = False
    else:
        path_diff = dst_path[len(common_path):]
        path_split = get_components(path_diff)
        dst_is_ignored = any([x in ignore_list for x in path_split])

    if src_is_parent and not dst_is_ignored:
        raise RuntimeError('Possibly infinite recursion '
                           'detected in safe_copytree')
    else:
        return shutil.copytree(src, dst,
                               ignore=shutil.ignore_patterns(*ignore_list))


def copy_recursively(src, dst, include, ignore):
    """
    Copy files recursively form src to dst
    :param src: source directory
    :param dst: destination directory
    :param include: list of files to include (can be specified by patterns)
    :param ignore: list of files and directories to ignore (can be specified by
        patterns)
    """
    # Get the canonical paths
    src_path = os.path.realpath(src)
    dst_path = os.path.realpath(dst)

    for name in os.listdir(src_path):
        src_sub_path = os.path.realpath(os.path.join(src_path, name))
        dst_sub_path = os.path.realpath(os.path.join(dst_path, name))

        # if file/dir is in ignore list do nothing
        if any([fnmatch.fnmatch(name, pat) for pat in ignore]):
            continue

        # call function recursively but avoid infinite recursion
        elif os.path.isdir(src_sub_path) and src_sub_path != dst_path:
            copy_recursively(src_sub_path, dst_sub_path, include, ignore)

        # copy file if included
        elif any([fnmatch.fnmatch(name, pat) for pat in include]):
            # create directory tree
            os.makedirs(dst_path, exist_ok=True)
            # copy file
            shutil.copy2(src_sub_path, dst_sub_path)

        else:
            continue


def copy_code(dest_dir):
    """
    Copy source code to dest_dir
    :param dest_dir: destination
    """
    # Get base directory
    base_dir = os.path.dirname(
        os.path.realpath(os.path.join(__file__, '..', '..')))

    # Decide what files to copy
    include = ['*.py']

    # Decide what files to ignore
    ignore = ['results']

    # Get the canonical paths
    src_path = os.path.realpath(base_dir)
    dst_path = os.path.realpath(dest_dir)

    # create code directory
    os.makedirs(dst_path, exist_ok=False)

    # copy files recursively
    copy_recursively(src_path, dst_path, include, ignore)


def flatten_cfg(cfg):
    """
    Flatten config dictionary. Removing sub-dicts allows for unified access to
    all parameters. Unique keys are required for all parameters.
    :param cfg: config dictionary with potentially several sub-dicts
    :return: cfg_f: flatten config dict containing no other dicts
    """

    cfg_f = {}

    def recurse(t, key=''):

        # Flatten dictionary recursively
        if isinstance(t, dict):
            for k, v in t.items():
                recurse(v, k)

        # Add item if no dictionary
        else:
            # Check for ambiguous parameter names
            if key in cfg_f.keys():
                print(f'Error occurred while flattening config file. Key '
                      f'"{key}" is provided more than once. \n'
                      f'Unique keys are '
                      f'required for all parameters. Exiting...')
                exit()

            cfg_f[key] = t

    # Start recursion
    recurse(cfg)

    return cfg_f


def boolean_string(s):
    if s in {'True', 'true', 'yes', 'y'}:
        return True
    elif s in {'False', 'false', 'no', 'n'}:
        return False
    else:
        raise ValueError('Not a valid boolean string')


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
        self._cfg = deepcopy(self._base_cfg)
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


if __name__ == "__main__":
    # safe_copytree('../..', '../../results/code',
    #               ignore_list=['tmp', '.git', 'results'])
    copy_code('../../results/mnist_test/code')
