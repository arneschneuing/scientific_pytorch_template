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
    ignore += ['.gitignore', '.git', '*.yaml']

    # Avoid infinite recursion
    ignore += [os.path.basename(dest_dir.strip('/'))]

    # Copy files to new location
    shutil.copytree(base_dir, dest_dir, ignore=shutil.ignore_patterns(*ignore))
