# *coding:utf-8 *

import os.path as osp
import sys


def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)


def __init_path():
    this_dir = osp.dirname(__file__)

    # Add lib to PYTHONPATH
    lib_path = osp.join(this_dir, 'lib')
    add_path(lib_path)

if __name__ == '__main__':
    __init_path()
    print("system path:")
    print(sys.path)
