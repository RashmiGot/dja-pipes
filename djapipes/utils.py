import os
import numpy as np
from astropy.table import Table

def path_to_filters():
    """
    Get path to the filters directory inside the module
    """
    
    data_path = os.path.join(os.path.dirname(__file__), "data/filters")
    return data_path


def read_filter_list(file="filt_list.txt"):
    """
    Read a list of filters
    """
    
    path_ = path_to_filters()
    with open(os.path.join(path_, file)) as fp:
        lines = fp.readlines()
    
    lines = [os.path.join(path_, file.strip()) for file in lines]
    return np.array(lines)


def load_zeropoints(file="zeropoints.csv"):
    """
    Load zeropoints table
    """
    zpoints = Table.read(
        os.path.join(path_to_filters(), file),
        format='csv'
    )
    return zpoints
    