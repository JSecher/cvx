import os
import sys
import time

RESULT_DIR = 'results'
THIS_RESULT_DIR = RESULT_DIR

def create_output_dir(child_dir=None):
    global THIS_RESULT_DIR
    if not os.path.exists(RESULT_DIR):
        os.makedirs(RESULT_DIR)
    if child_dir is not None:
        child_dir = os.path.join(RESULT_DIR, child_dir)
        if not os.path.exists(child_dir):
            os.makedirs(child_dir)
    THIS_RESULT_DIR = child_dir
    return child_dir

def get_file_path(file_name):
    return os.path.join(THIS_RESULT_DIR, file_name)
