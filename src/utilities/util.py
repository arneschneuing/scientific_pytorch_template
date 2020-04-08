import os
import re
import shutil


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


def copy_code(dest_dir):
    """
    Copy source code to dest_dir
    :param dest_dir: destination
    """

    # Get base directory
    base_dir = os.path.dirname(
        os.path.realpath(os.path.join(__file__, '../..')))

    # Make sure important files are present
    assert os.path.isdir(os.path.join(base_dir, 'src'))
    assert os.path.isfile(os.path.join(base_dir, 'train.py'))

    # Read gitignore
    gitignore = os.path.join(base_dir, '.gitignore')
    with open(gitignore) as f:
        ignore = f.readlines()
    ignore = [x.strip('\n').strip('/') for x in ignore]

    # Add further items to ignore
    ignore += ['.gitignore', '.git', '*.yaml', 'README.md']

    # Copy files to new location
    safe_copytree(base_dir, dest_dir, ignore)


if __name__ == "__main__":
    safe_copytree('../..', '../../results/code',
                  ignore_list=['tmp', '.git', 'results'])
