# ------- LIBRARIES ------ #
import os

from astropy.table import Table
import numpy as np
import grizli.utils as utils
import eazy
import msaexp.spectrum 
# from msaexp import spectrum

from . import utils as djautils

import matplotlib.pyplot as plt

# --------------------------------------------------------------
# ------------------------- EFFECTIVE WAVELENGTH OF PHOT FILTERS
# --------------------------------------------------------------

def calc_eff_wavs(filt_list):
    """
    Calculates effective wavelengths of photometric filters
    
    Parameters
    ----------
    filt_list : array of paths to filter files, each element is a string

    Returns
    -------
    eff_wavs : effective wavelengths of the input filters, format=numpy array
    """

    eff_wavs = np.zeros(len(filt_list))

    for i in range(len(eff_wavs)):

        filt_dict = np.loadtxt(filt_list[i], usecols=(0, 1))

        dlambda = np.diff(filt_dict[:,0]) # calculating delta lambda
        dlambda = np.append(dlambda, dlambda[-1]) # adding dummy delta lambda value to make dlambda array shape equal to filter array shape 
        filt_weights = dlambda*filt_dict[:,1] # filter weights

        # effective wav
        eff_wavs[i] = np.sqrt(np.sum(filt_weights*filt_dict[:,0]) 
                                            / np.sum(filt_weights
                                            / filt_dict[:,0]))
        
    return eff_wavs



# --------------------------------------------------------------
# ------------------- CONVERT FLUX UNITS FROM CGS TO MICROJANSKY
# --------------------------------------------------------------
def convert_mujy2cgs(fnuObs_mujy,lamObs_A):
    """
    Converts fluxes from microjansky to cgs units
    
    Parameters
    ----------
    fnuObs_mujy : array of fluxes in units of microjansky, format=numpy array
    lamObs_A : array of wavelengths in units of angstrom, format=numpy array

    Returns
    -------
    flamObs_cgs : array of fluxes in cgs units, format=numpy array
    """
    
    flamObs_cgs = fnuObs_mujy * (10**-29*2.9979*10**18/lamObs_A**2)
    
    return flamObs_cgs


# --------------------------------------------------------------
# ------------------- CONVERT FLUX UNITS FROM MICROJANSKY tO CGS
# --------------------------------------------------------------
def convert_cgs2mujy(flamObs_cgs,lamObs_A):
    """
    Converts fluxes from cgs to microjansky units
    
    Parameters
    ----------
    flamObs_cgs : array of fluxes in cgs units, format=numpy array
    lamObs_A : array of wavelengths in units of angstrom, format=numpy array

    Returns
    -------
    fnuObs_mujy : array of fluxes in units of microjansky, format=numpy array
    """
    
    fnuObs_mujy = flamObs_cgs / (10**-29*2.9979*10**18/lamObs_A**2)
    
    return fnuObs_mujy


# --------------------------------------------------------------
# -------------------------------------------------- FILTER LIST
# --------------------------------------------------------------
def updated_filt_list(ID):
    """
    Updates filterlist
    
    Parameters
    ----------
    ID : number assigned to fitting run, format=int

    Returns
    -------
    filt_list[phot_flux_mask] : masked filter list, format=numpy array
    """

    id = int(ID)

    speclist_cat = Table.read('spec_cat_temp.csv', format='csv')

    fname_phot_out = speclist_cat[speclist_cat["id"]==id]["fname"][0]+'.phot.cat'
    phot_tab = Table.read(f'files/{fname_phot_out}', format='ascii.commented_header')

    # jwst filter list
    # filt_list = np.loadtxt(
    #     os.path.join(djautils.path_to_filters(), "filt_list.txt"),
    #     dtype="str"
    # )
    
    filt_list = djautils.read_filter_list("filt_list.txt")
    
    # extract fluxes from cat (muJy); '{filter}_tot_1'==0.5'' aperture
    flux_colNames = [
        os.path.basename(filt_list_i).split('.')[0]+'_tot_1'
        for filt_list_i in filt_list
    ]

    eflux_colNames = [
        os.path.basename(filt_list_i).split('.')[0]+'_etot_1'
        for filt_list_i in filt_list
    ]

    fluxes_muJy = np.lib.recfunctions.structured_to_unstructured(np.array(phot_tab[list(flux_colNames)]))[0]
    efluxes_muJy = np.lib.recfunctions.structured_to_unstructured(np.array(phot_tab[list(eflux_colNames)]))[0]

    phot_flux_mask = (fluxes_muJy>-90) & (efluxes_muJy>0)

    return filt_list[phot_flux_mask]




# --------------------------------------------------------------
# ------------------------------- CALCULATE SYNTHETIC PHOTOMETRY
# --------------------------------------------------------------
def synthetic_photometry(filt_list, spectrum):
    """
    Calculates synthetic photometry for a spectrum for given filters
    
    Parameters
    ----------
    filt_list : array of paths to filter files, each element is a string, format=numpy array
    spectrum : spectrum array containing spectral wavelengths, fluxes and uncertainties, format=numpy array

    Returns
    -------
    syn_phot : synthetic photometry, format=numpy array
    """

    eff_wavs = calc_eff_wavs(filt_list=filt_list) # effective wavelengths

    syn_phot = np.zeros_like(eff_wavs) # initialise array for synthetic photometry
    
    for i in range(len(filt_list)):
        filt_i = np.loadtxt(filt_list[i], usecols=(0, 1)) # load filter profile
        spec_wav_mask = (spectrum[:,0] > filt_i[:,0].min()) & (spectrum[:,0] < filt_i[:,0].max()) # mask for spectrum in filter wav range
        spectrum_masked = spectrum[spec_wav_mask] # masked spectrum
        filt_i_interp = np.interp(spectrum_masked[:,0], filt_i[:,0], filt_i[:,1]) # interpolating filter profile onto spec wavs
        filt_int = np.trapz(filt_i[:,1], filt_i[:,0]) # integral of filter
        syn_phot[i] = np.trapz(spectrum_masked[:,1]*filt_i_interp/filt_int, spectrum_masked[:,0]) # synthetic photometry from spectrum

    return syn_phot


# --------------------------------------------------------------
# ------------------------------- CALCULATE SYNTHETIC PHOTOMETRY
# --------------------------------------------------------------
def synthetic_photometry_msa(z, filt_list, spec_tab):
    """
    Calculates synthetic photometry for a spectrum for given filters with msaexp
    
    Parameters
    ----------
    z : redshift, format=float
    filt_list : array of paths to filter files, each element is a string, format=numpy array
    spectrum : spectrum array containing spectral wavelengths, fluxes and uncertainties, format=numpy array

    Returns
    -------
    syn_phot : synthetic photometry, format=numpy array
    """

    spec_tab = Table([spec_tab[:,0]/10000, spec_tab[:,1], spec_tab[:,2]], names=['wave', 'flux', 'err'])

    syn_phot = np.zeros(len(filt_list)) # initialise array for synthetic photometry
    syn_phot_err = np.zeros(len(filt_list)) # initialise array for synthetic photometry errors
    
    for i in range(len(syn_phot)):
        filt_tab_i = Table.read(filt_list[i], format='ascii')
        eazy_filt = eazy.filters.FilterDefinition(wave=filt_tab_i["col1"]/(1+z), throughput=filt_tab_i["col2"])

        syn_phot_tab = msaexp.spectrum.integrate_spectrum_filter(spec_tab, eazy_filt, z=z)

        syn_phot[i], syn_phot_err[i] = syn_phot_tab[2], syn_phot_tab[3]

    return syn_phot, syn_phot_err



# --------------------------------------------------------------
# -------------------------------------------- CALIB PRIOR GUESS
# --------------------------------------------------------------
def guess_calib(ID, z, plot=False, phot_xpos=None, spec_xpos=None):
    """
    Estimates calibration function to inform the calib prior in the pipes fitting routine
    
    Parameters
    ----------
    ID : number assigned to fitting run, format=int
    plot : plots calibration curve, format=bool

    Returns
    -------
    coeffs : coefficients of the calibration polynomial, format=numpy array
    """

    id = int(ID)

    # filter list 
    filt_list = updated_filt_list(id) # filt list
    eff_wavs = calc_eff_wavs(filt_list=filt_list) # effective wavelengths
    spec_fluxes, phot_fluxes = load_both(id) # spec and phot fluxes
    syn_phot, syn_phot_err = synthetic_photometry_msa(z=z, filt_list=filt_list, spec_tab=spec_fluxes) # synthetic photometry

    y = phot_fluxes[:,0] / syn_phot # ratio of real to synthetic photometry
    yerr = np.abs(y * np.sqrt((phot_fluxes[:,1] / phot_fluxes[:,0])**2 + (syn_phot_err / syn_phot)**2)) # error on ratio of real to synthetic photometry

    # transforming spec axis such that it runs from -1 to 1
    xfull = spec_fluxes[:, 0]
    xfull_trans = 2.*(xfull - (xfull[0] + (xfull[-1] - xfull[0])/2.))/(xfull[-1] - xfull[0])
    x_trans = np.interp(eff_wavs, spec_fluxes[:, 0], xfull_trans)

    # "design matrix"
    A = np.polynomial.chebyshev.chebvander(x_trans, 2)
    Afull = np.polynomial.chebyshev.chebvander(xfull_trans, 2)
    lsq_coeffs = np.linalg.lstsq((A.T/yerr).T, y/yerr, rcond=None)

    # design matrix with weights
    Ax = (A.T/yerr).T

    # covariance matrix
    covar = utils.safe_invert(np.dot(Ax.T, Ax))
    param_uncertainties = np.sqrt(covar.diagonal())

    if plot:
        cfit = np.polynomial.chebyshev.chebfit(x_trans, y, deg=2, w=1./yerr)

        random_coeffs = np.random.multivariate_normal(lsq_coeffs[0], covar, size=100)
        random_models = Afull.dot(random_coeffs.T)

        post = np.percentile(random_models.T, (16, 50, 84), axis=0).T

        plt.errorbar(phot_xpos, y, yerr=yerr, marker='o', color="grey", zorder=10, ls=' ', lw=0.8)
        plt.plot(spec_xpos, post[:, 0], color="grey", zorder=10, lw=0.1)
        plt.plot(spec_xpos, post[:, 1], color="grey", zorder=10, label='Prior calib guess')
        plt.plot(spec_xpos, post[:, 2], color="grey", zorder=10, lw=0.1)
        plt.fill_between(spec_xpos, post[:, 0], post[:, 2], lw=0,
                        color="grey", alpha=0.3, zorder=9)

    return lsq_coeffs[0], param_uncertainties, covar


def check_spec(ID, valid_threshold=400):
    """
    Searches for valid flux datapoints in spectrum
    
    Parameters
    ----------
    ID : number assigned to fitting run, format=int

    Returns
    -------
    num_valid : number of valid datapoints in spectrum, format=int
    is_valid : True/False depending on whether spectrum has breaks, format=bool
    """

    id = int(ID)

    speclist_cat = Table.read('spec_cat_temp.csv', format='csv')
    fname_spec = speclist_cat[speclist_cat["id"]==id]["fname"][0]+'.spec.fits'
    file_path='files/'

    msaexp_spectrum = msaexp.spectrum.read_spectrum(f"{file_path}{fname_spec}")
    valid = msaexp_spectrum["valid"]

    num_valid = sum(valid*1)

    if num_valid<valid_threshold:
        is_valid=False
    else:
        is_valid=True

    return num_valid, is_valid


# --------------------------------------------------------------
# ---------------------------------------------- LOAD PHOTOMETRY
# --------------------------------------------------------------
def load_phot(ID):
    """
    Load photometry from catalogue.
    
    Parameters
    ----------
    ID : number assigned to fitting run, format=int

    Returns
    -------
    photometry : table containing photometric fluxes and flux uncertainties, format=numpy array
    """

    id = int(ID)

    speclist_cat = Table.read('spec_cat_temp.csv', format='csv')

    fname_phot_out = speclist_cat[speclist_cat["id"]==id]["fname"][0]+'.phot.cat'
    phot_tab = Table.read(f'files/{fname_phot_out}', format='ascii.commented_header')

    # jwst filter list
    # filt_list = np.loadtxt("../filters/filt_list.txt", dtype="str")
    # filt_list = djautils.read_filter_list("filt_list.txt")
    filt_list = updated_filt_list(id)

    # extract fluxes from cat (muJy); '{filter}_tot_1'==0.5'' aperture
    # flux_colNames = [filt_list_i.split('/')[-1].split('.')[0]+'_tot_1' for filt_list_i in filt_list]
    # eflux_colNames = [filt_list_i.split('/')[-1].split('.')[0]+'_etot_1' for filt_list_i in filt_list]

    flux_colNames = [
        os.path.basename(filt_list_i).split('.')[0]+'_tot_1'
        for filt_list_i in filt_list
    ]

    eflux_colNames = [
        os.path.basename(filt_list_i).split('.')[0]+'_etot_1'
        for filt_list_i in filt_list
    ]

    # zeropoints table
    zpoints = djautils.load_zeropoints()
    
    zpoints_sub = zpoints[zpoints["root"]==phot_tab["file_phot"][0].split('_phot')[0]]
    zp_array = [zpoints_sub[zpoints_sub["f_name"]==flux_colName]["zp"][0] for flux_colName in flux_colNames]

    # make flux arrays and correct for zeropoints
    fluxes_muJy_no_zp = np.lib.recfunctions.structured_to_unstructured(np.array(phot_tab[list(flux_colNames)]))[0]
    fluxes_muJy = fluxes_muJy_no_zp * zp_array
    efluxes_muJy = np.lib.recfunctions.structured_to_unstructured(np.array(phot_tab[list(eflux_colNames)]))[0]

    phot_flux_mask = (fluxes_muJy>-90) & (efluxes_muJy>0)

    # effective wavelengths of photometric filters
    phot_wavs_temp = (calc_eff_wavs(filt_list))
    phot_wavs = np.array(phot_wavs_temp)

    # convert fluxes from muJy to cgs
    fluxes = convert_mujy2cgs(fluxes_muJy,phot_wavs)
    efluxes = convert_mujy2cgs(efluxes_muJy,phot_wavs)
    
    # turn these into a 2D array
    photometry = np.c_[fluxes,efluxes][phot_flux_mask]
            
    # enforce a maximum SNR of 20, or 10 in the IRAC channels
    # for i in range(len(photometry)):
    #     if i < 10:
    #         max_snr = 20.
    #     else:
    #         max_snr = 10.

    #     if photometry[i, 0]/photometry[i, 1] > max_snr:
    #         photometry[i, 1] = photometry[i, 0]/max_snr
    sys=0.03
    photometry[:, 1] = np.sqrt((photometry[:, 1])**2 + (sys*photometry[:, 0])**2)

    return photometry



# --------------------------------------------------------------
# ------------------------------------------------- BIN SPECTRUM
# --------------------------------------------------------------

def bin_spec(spectrum, binn):
    """ Bins up two or three column spectral data by a specified factor.
    
    Parameters
    ----------
    spectrum : table containing spectral wavelengths, fluxes and flux uncertainties, format=numpy array
    binn : binning factor, format=int

    Returns
    -------
    binspec : table containing binned spectral wavelengths, fluxes and flux uncertainties, format=numpy array
    """

    binn = int(binn)
    nbins = len(spectrum) // binn
    binspec = np.zeros((nbins, spectrum.shape[1]))

    for i in range(binspec.shape[0]):
        spec_slice = spectrum[i*binn:(i+1)*binn, :]
        binspec[i, 0] = np.mean(spec_slice[:, 0])
        binspec[i, 1] = np.mean(spec_slice[:, 1])

        if spectrum.shape[1] == 3:
            binspec[i,2] = (1./float(binn)
                            *np.sqrt(np.sum(spec_slice[:, 2]**2)))

    return binspec


# --------------------------------------------------------------
# ------------------------------------------------ LOAD SPECTRUM
# --------------------------------------------------------------

def load_spec(ID):
    """
    Loads spectrum from fits file, converts to numpy array and returns binned spectrum
    
    Parameters
    ----------
    ID : number assigned to fitting run, format=int

    Returns
    -------
    binned_spectrum : table containing binned spectral wavelengths, fluxes and flux uncertainties, format=numpy array
    """

    id = int(ID)

    speclist_cat = Table.read('spec_cat_temp.csv', format='csv')

    fname_spec_out = speclist_cat[speclist_cat["id"]==id]["fname"][0]+'.spec.fits'
    spec_tab = Table.read(f'files/{fname_spec_out}', hdu=1)

    spec_wavs = np.array(spec_tab['wave'])*1e4 # convert wavs to angstrom

    # filt_list = updated_filt_list(id) # filt list
    # wav_min, wav_max = Table.read(filt_list[0], format="ascii")[0][0], Table.read(filt_list[-1], format="ascii")[-1][0]

    # spec_wavs_mask = (spec_wavs>7836) & (spec_wavs<50994) # wavs within photometric filter limits
    # spec_wavs_mask = (spec_wavs>wav_min) & (spec_wavs<wav_max)
    spec_wavs_mask = (spec_wavs>(spec_wavs.min()+100)) & (spec_wavs<(spec_wavs.max()-100))

    flux_muJy = np.array(spec_tab['flux'])
    fluxerr_muJy = np.array(spec_tab['err'])
    fluxerr_muJy[np.invert(spec_tab['line_mask'])] = np.nanmean(fluxerr_muJy) * 1e3
    flux_cgs = convert_mujy2cgs(flux_muJy,spec_wavs)
    fluxerr_cgs = convert_mujy2cgs(fluxerr_muJy,spec_wavs)

    # constructing spectrum table
    spectrum = np.c_[spec_wavs[spec_wavs_mask],
                     flux_cgs[spec_wavs_mask],
                     fluxerr_cgs[spec_wavs_mask]]

    # binning spectrum
    binned_spectrum = bin_spec(spectrum, 1)

    return binned_spectrum


# --------------------------------------------------------------
# ----------------------------------- LOAD PHOTOMETRY & SPECTRUM
# --------------------------------------------------------------

def load_both(ID):
    """
    Loads spectrum and photometry of source as numpy arrays
    
    Parameters
    ----------
    ID : number assigned to fitting run, format=int

    Returns
    -------
    spectrum : table containing binned spectral wavelengths, fluxes and flux uncertainties, format=numpy array
    phot : table containing photometric fluxes and flux uncertainties, format=numpy array
    """

    id = int(ID)
    
    spectrum = load_spec(id)
    phot = load_phot(id)

    return spectrum, phot




# --------------------------------------------------------------
# ------------------------------------------ MASK EMISSION LINES
# --------------------------------------------------------------

def mask_emission_lines(fname_spec, z_spec, file_path='files/', mask_lines=False, line_wavs=None, delta_lam=None):
    """
    Masks spectrum, writes spectrum to a fits file
    
    Parameters
    ----------
    fname_spec : filename of spectrum, e.g. 'rubies-uds3-v3_prism-clear_4233_62812.spec.fits', format=str
    z_spec : spectroscopic redshift, format=float
    file_path : path to folder in which outfile is stored, format=str
    mask_lines : boolean to determine whether or not to mask lines, format=bool
    line_wavs : array of emission line wavelengths in angstrom, format=numpy array
    delta_lam : +-wavelength range to mask out in angstrom, format=float

    Returns
    -------
    None
    """

    # spectrum
    spec_tab = Table.read(f'{file_path}{fname_spec}', hdu=1)
    spec_wavs = spec_tab['wave'] # in microns
    
    # making mask to identify regions around emission lines
    line_mask = np.array([True]*len(spec_wavs))
    if mask_lines:
        for i in range(len(line_wavs)):
            mask_i = (spec_wavs>(line_wavs[i]-delta_lam)/1e4*(1+z_spec)) & (spec_wavs<(line_wavs[i]+delta_lam)/1e4*(1+z_spec))
            line_mask = line_mask & np.invert(mask_i)

    # rewriting new spec table to original spectrum file
    if "line_mask" in spec_tab.colnames:
        spec_tab.replace_column("line_mask", line_mask)
    elif "line_mask" not in spec_tab.colnames:
        spec_tab.add_column(line_mask, index=-1, name="line_mask")
    spec_tab.write(f'{file_path}{fname_spec}', format='fits', overwrite=True)

    return None


