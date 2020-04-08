import os
import re


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
