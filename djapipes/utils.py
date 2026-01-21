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


def generate_calzetti_curve(wave):
    """
    Generates Calzetti+00 dust attenuation curve; modified from Bagpipes (see dust_attenuation_model.py)

    wave : array
        rest-frame wavelengths in microns
    """

    A_lambda = np.zeros_like(wave)

    mask1 = (wave < 0.12)
    mask2 = (wave < 0.63) & (wave >= 0.12)
    mask3 = (wave < 3.1) & (wave >= 0.63)

    A_lambda[mask1] = ((wave[mask1]/0.12)**-0.77
                        * (4.05 + 2.695*(- 2.156 + 1.509/0.12
                                        - 0.198/0.12**2 + 0.011/0.12**3)))

    A_lambda[mask2] = (4.05 + 2.695*(- 2.156
                                        + 1.509/wave[mask2]
                                        - 0.198/wave[mask2]**2
                                        + 0.011/wave[mask2]**3))

    A_lambda[mask3] = 2.659*(-1.857 + 1.040/wave[mask3]) + 4.05

    A_lambda /= 4.05  # normalize to A_V

    return A_lambda

def generate_salim_dust_curve(file_spec, suffix, sfh="continuity", dust="salim"):
    """
    generate dust curve from posterior catalogue
    """

    runName = file_spec.split('.spec.fits')[0]

    # posterior models and output catalogue
    postcat = Table.read(f"pipes/cats/{runName}/{runName}_{sfh}_{dust}_{suffix}_postcat.csv", format="csv")

    # read dust model parameters
    redshift = postcat["redshift_50"][0]
    delta = postcat["dust_delta_50"][0]
    B = postcat["dust_B_50"][0]

    wave = np.linspace(1, 1e5, int(1e5))  # example wavelength range in angstroms

    # calzetti curve (Rv = ratio of total to selective extinction in V-band)
    Rv_cal = 4.05
    A_lambda_calz = generate_calzetti_curve(wave/1e4)  # in magnitudes

    # modified Rv (Salim+18)
    Rv_m = Rv_cal/((Rv_cal+1)*((4400./5500.)**delta) - Rv_cal)

    D_lambda = B*wave**2*350.**2
    D_lambda /= (wave**2 - 2175**2)**2 + wave**2*350.**2
    
    A_cont = A_lambda_calz * Rv_m * (wave/5500.)**delta
    A_tot = A_cont + D_lambda
    A_tot /= Rv_m  # normalize by modified Rv

    return(wave, A_tot)