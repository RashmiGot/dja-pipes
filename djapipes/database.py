# ------- LIBRARIES ------ #
import wget
import os
import urllib.request
import urllib.error

from astropy.table import Table

from . import utils as djautils

# --------------------------------------------------------------
# ---------------------------------- PULL SPECTRUM FROM DATABASE
# --------------------------------------------------------------

def pull_spec_from_db(fname_spec, file_path='files/'):
    """
    Downloads spectrum from DJA AWS database, writes spectrum to a fits file
    
    Parameters
    ----------
    fname_spec : filename of spectrum, e.g. 'rubies-uds3-v3_prism-clear_4233_62812.spec.fits', format=str
    file_path : path to folder in which outfile is stored, format=str

    Returns
    -------
    None
    """

    url_path_dja = 'https://s3.amazonaws.com/msaexp-nirspec/extractions'
    url_path_canucs = 'https://s3.amazonaws.com/grizli-canucs/nirspec'
    root_i = fname_spec.split('_')[0]
    file_i = fname_spec

    urls_to_try = [
        f"{url_path_dja}/{root_i}/{file_i}",
        f"{url_path_canucs}/{root_i}/{file_i}"
    ]

    # checks if file exists; if yes, deletes existing file
    if os.path.exists(file_path+fname_spec):
        os.remove(file_path+fname_spec)

    for i, url in enumerate(urls_to_try):
        try:
            # downloads spectrum from aws database
            wget.download(url=url, out=file_path+fname_spec)
            print(f"\nSuccessfully downloaded {fname_spec} from DJA")
            return None
        except Exception as e:
            print(f"Error downloading {url}: {e}")

# --------------------------------------------------------------
# -------------------------------- PULL PHOTOMETRY FROM DATABASE
# --------------------------------------------------------------

def pull_phot_from_db(fname_spec, fname_phot, file_path='files/'):
    """
    Downloads photometry from DJA AWS database, writes photometry to an ascii file
    
    Parameters
    ----------
    fname_spec : filename of spectrum for which corresponding photometry needs to be found,
                 e.g. 'rubies-uds3-v3_prism-clear_4233_62812.spec.fits', format=str
    fname_phot : outfile name of photometry, e.g. 'rubies-uds3-v3_prism-clear_4233_62812.phot.cat', format=str
    file_path : path to folder in which outfile is stored, format=str

    Returns
    -------
    None
    """

    url = 'https://grizli-cutout.herokuapp.com/grizli_photometry?file_spec='
    nrp_gr_match = Table.read(url+fname_spec, format='csv')

    # choose best photometry
    photcats_latest = djautils.preferred_catalogs()
    
    matching_index = []
    for i in range(len(nrp_gr_match)):
        try:
            matching_index.append(list(photcats_latest["file_phot"]).index(list(nrp_gr_match["file_phot"])[i]))
        except ValueError:
            matching_index.append(1e9)
    photcat_index = matching_index.index(min(matching_index))

    # write SQL query output to astropy Table
    phot_tab = Table(nrp_gr_match[photcat_index])
    # write table to output file
    phot_tab.write(f'{file_path}{fname_phot}', format='ascii.commented_header', overwrite=True)




# --------------------------------------------------------------
# -------------------- PULL SPECTROSCPOIC REDSHIFT FROM DATABASE
# --------------------------------------------------------------

def pull_zspec_from_db(fname_spec):
    """
    Downloads spectroscopic redshift from DJA AWS database
    
    Parameters
    ----------
    fname_spec : filename of spectrum, e.g. 'rubies-uds3-v3_prism-clear_4233_62812.spec.fits', format=str
    Returns
    -------
    z_spec : spectroscopic redshift, format=float
    """

    z_spec_url = 'https://grizli-cutout.herokuapp.com/nirspec_file_redshift?file='
    z_spec_tab = Table.read(z_spec_url+fname_spec, format='csv')

    if 'z_prism' in z_spec_tab.colnames:
        zdiff = abs(z_spec_tab["z_prism"][0]-z_spec_tab["z"][0])
        if zdiff<0.1:
            z_spec = z_spec_tab["z_prism"][0]
        else:
            z_spec = z_spec_tab["z"][0]
    else:
        z_spec = z_spec_tab["z"][0]

    return z_spec