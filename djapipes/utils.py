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


def load_calib_curve(file_spec, suffix, sfh="continuity", dust="salim"):
    """
    Load calibration curve from posterior catalogue
    """

    runName = file_spec.split('.spec.fits')[0]

    # posterior models and output catalogue
    postmodels = Table.read(f"pipes/cats/{runName}/{runName}_{sfh}_{dust}_{suffix}_postmodels.csv", format="csv")
    postcat = Table.read(f"pipes/cats/{runName}/{runName}_{sfh}_{dust}_{suffix}_postcat.csv", format="csv")

    # extract wavelength grid and calibration coefficients
    wave = postmodels["wave_sed"]
    wave_min, wave_max = np.nanmin(wave), np.nanmax(wave)
    c_coeffs = np.array([postcat["calib_0_50"][0], postcat["calib_1_50"][0], postcat["calib_2_50"][0]])

    # reconstruct calibration curve
    calib_curve = generate_calib_curve(c_coeffs, wave, wave_min, wave_max)

    return(wave, calib_curve)


def generate_calib_curve(coeffs, x_arr, x_min, x_max):
    """
    Generate calibration curve from Chebyshev coefficients
    """

    x_trans = 2 * (x_arr - x_min) / (x_max - x_min) - 1
    calib_curve = np.polynomial.chebyshev.chebval(x_trans, coeffs)

    return calib_curve