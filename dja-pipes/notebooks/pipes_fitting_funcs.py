# ------- LIBRARIES ------ #
from astropy.table import Table
import numpy as np


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

    fname_phot_out = 'file_for_pipes.phot.cat'
    phot_tab = Table.read(f'../files/{fname_phot_out}', format='ascii.commented_header')

    # jwst filter list
    filt_list = np.loadtxt("../filters/filt_list.txt", dtype="str")

    # extract fluxes from cat (muJy); '{filter}_tot_1'==0.5'' aperture
    flux_colNames = [filt_list_i.split('/')[-1].split('.')[0]+'_tot_1' for filt_list_i in filt_list]
    eflux_colNames = [filt_list_i.split('/')[-1].split('.')[0]+'_etot_1' for filt_list_i in filt_list]

    fluxes_muJy = np.lib.recfunctions.structured_to_unstructured(np.array(phot_tab[list(flux_colNames)]))[0]
    efluxes_muJy = np.lib.recfunctions.structured_to_unstructured(np.array(phot_tab[list(eflux_colNames)]))[0]

    phot_wavs = np.array([0.9,1.15,1.5,2.0,2.77,3.56,4.44])*1e4 # convert photometric filter wavs to angstrom

    # convert fluxes from muJy to cgs
    fluxes = convert_mujy2cgs(fluxes_muJy,phot_wavs)
    efluxes = convert_mujy2cgs(efluxes_muJy,phot_wavs)
    
    # turn these into a 2D array
    photometry = np.c_[fluxes,efluxes]

    # blow up the errors associated with any missing fluxes
    for i in range(len(photometry)):
        if (photometry[i, 0] == 0.) or (photometry[i, 1] <= 0):
            photometry[i,:] = [0., 9.9*10**99.]
            
    # enforce a maximum SNR of 20, or 10 in the IRAC channels
    for i in range(len(photometry)):
        if i < 10:
            max_snr = 20.
        else:
            max_snr = 10.

        if photometry[i, 0]/photometry[i, 1] > max_snr:
            photometry[i, 1] = photometry[i, 0]/max_snr

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

    fname_spec_out = 'file_for_pipes.spec.fits'
    spec_tab = Table.read(f'../files/{fname_spec_out}', hdu=1)

    spec_wavs = np.array(spec_tab['wave'])*1e4 # convert wavs to angstrom

    spec_wavs_mask = (spec_wavs>7836) & (spec_wavs<50994) # wavs within photometric filter limits
    
    flux_muJy = np.array(spec_tab['flux'])
    fluxerr_muJy = np.array(spec_tab['err'])
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







