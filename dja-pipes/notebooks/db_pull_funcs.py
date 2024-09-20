# ------- LIBRARIES ------ #
import wget
import os

from astropy.table import Table


# --------------------------------------------------------------
# ---------------------------------- PULL SPECTRUM FROM DATABASE
# --------------------------------------------------------------

def pull_spec_from_db(fname_spec, fname_spec_out, file_path='../files/'):
    """
    Downloads spectrum from DJA AWS database, writes spectrum to a fits file
    
    Parameters
    ----------
    fname_spec : filename of spectrum, e.g. 'rubies-uds3-v3_prism-clear_4233_62812.spec.fits', format=str
    fname_spec_out : outfile name of spectrum, e.g. 'file_for_pipes.spec.fits', format=str
    file_path : path to folder in which outfile is stored, format=str

    Returns
    -------
    None
    """

    url_path_dja = 'https://s3.amazonaws.com/msaexp-nirspec/extractions'
    root_i = fname_spec.split('_')[0]
    file_i = fname_spec

    # checks if file exists; if yes, deletes existing file
    if os.path.exists(file_path+fname_spec_out):
        os.remove(file_path+fname_spec_out)
    # downloads spectrum from aws database
    wget.download(url=url_path_dja+'/'+root_i+'/'+file_i, out=file_path+fname_spec_out)




# --------------------------------------------------------------
# -------------------------------- PULL PHOTOMETRY FROM DATABASE
# --------------------------------------------------------------

def pull_phot_from_db(fname_spec, fname_phot_out, file_path='../files/'):
    """
    Downloads photometry from DJA AWS database, writes photometry to an ascii file
    
    Parameters
    ----------
    fname_spec : filename of spectrum for which corresponding photometry needs to be found,
                 e.g. 'rubies-uds3-v3_prism-clear_4233_62812.spec.fits', format=str
    fname_phot_out : outfile name of photometry, e.g. 'file_for_pipes.phot.cat', format=str
    file_path : path to folder in which outfile is stored, format=str

    Returns
    -------
    None
    """

    url = 'https://grizli-cutout.herokuapp.com/grizli_photometry?file_spec='
    nrp_gr_match = Table.read(url+fname_spec, format='csv')

    # write SQL query output to astropy Table
    phot_tab = Table(nrp_gr_match[0])
    # write table to output file
    phot_tab.write(f'{file_path}{fname_phot_out}', format='ascii.commented_header', overwrite=True)




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
    None
    """

    z_spec_url = 'https://grizli-cutout.herokuapp.com/nirspec_file_redshift?file='
    z_spec = Table.read(z_spec_url+fname_spec, format='csv')['z'][0]

    return z_spec