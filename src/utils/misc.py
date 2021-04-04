import errno
import os
import shutil
import sys
import torch
import imageio
import numpy as np
#To plot pretty picture
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.switch_backend("agg") #release "RuntimeError: main thread is not in main loop" error

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

def to_np(var: torch.Tensor):
    """Exports torch.Tensor to Numpy array.
    """
    return var.detach().cpu().numpy()

def create_folder(folder_path):
    """Create a folder if it does not exist.
    """
    try:
        os.makedirs(folder_path)
    except OSError as _e:
        if _e.errno != errno.EEXIST:
            raise

def clear_and_create_folder(folder_path):
    """Clear all contents recursively if the folder exists.
    Create the folder if it has been accidently deleted.
    """
    create_folder(folder_path)
    for the_file in os.listdir(folder_path):
        _file_path = os.path.join(folder_path, the_file)
        try:
            if os.path.isfile(_file_path):
                os.unlink(_file_path)
            elif os.path.isdir(_file_path):
                shutil.rmtree(_file_path)
        except OSError as _e:
            print(_e)

