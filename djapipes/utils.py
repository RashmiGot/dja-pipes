import os
import numpy as np
from astropy.table import Table
import astropy.io.fits as pyfits

def preferred_catalogs():
    """
    Get ordered list of preferred photometric catalogs
    """
    data_path = os.path.join(os.path.dirname(__file__), "data/photcats_latest.csv")
    photcats_latest = Table.read(data_path, format="csv")
    return photcats_latest


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


def load_prism_dispersion(scale_disp=1.3):
    """
    Get prism dispersion arrays
    """
    data_path = os.path.join(
        os.path.dirname(__file__),
        "data/jwst_nirspec_prism_disp.fits"
    )
    
    with pyfits.open(data_path) as hdul:
        resData = np.c_[10000*hdul[1].data["WAVELENGTH"], hdul[1].data["R"]*scale_disp]
    
    return resData

    