# ------- LIBRARIES ------ #
import wget
import os

from grizli.aws import db
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

    # SQL query with all required parameters
    nrp_gr_match = db.SQL(f"""
    SELECT ne.root as root_spec, ne.file as file_spec, nr.z as z_spec, nrm.grade,
           nrp.file_phot, nrp.id_phot, nrp.dr,
           gr.flux_radius,
           gr.f090w_tot_0, gr.f090w_etot_0, gr.f090w_tot_1, gr.f090w_etot_1,
           gr.f115w_tot_0, gr.f115w_etot_0, gr.f115w_tot_1, gr.f115w_etot_1,
           gr.f150w_tot_0, gr.f150w_etot_0, gr.f150w_tot_1, gr.f150w_etot_1,
           gr.f200w_tot_0, gr.f200w_etot_0, gr.f200w_tot_1, gr.f200w_etot_1,
           gr.f277w_tot_0, gr.f277w_etot_0, gr.f277w_tot_1, gr.f277w_etot_1,
           gr.f356w_tot_0, gr.f356w_etot_0, gr.f356w_tot_1, gr.f356w_etot_1,
           gr.f444w_tot_0, gr.f444w_etot_0, gr.f444w_tot_1, gr.f444w_etot_1,
           gr.file_zout, 
           gr.z_phot
    FROM nirspec_extractions ne,
         nirspec_redshifts nr,
         (SELECT nrm.file, nrm.z, nrm.grade          /* most recent grade */
                 FROM nirspec_redshifts_manual nrm,
                      (SELECT nrm2.file, MAX(nrm2.ctime) as max_ctime
                             FROM nirspec_redshifts_manual nrm2
                             GROUP BY nrm2.file
                      ) nrm_sorted
                 WHERE nrm.ctime >= nrm_sorted.max_ctime
                 AND nrm.file = nrm_sorted.file
                 AND nrm.z > 0
         ) nrm,
         nirspec_phot_match nrp,
         grizli_photometry gr
    WHERE (ne.root LIKE '%%v3')
    --AND (nrp.file_phot LIKE '%%v7.2%%')
    AND ne.file = nr.file
    AND ne.file LIKE '{fname_spec}'
    AND nrp.dr < 0.2 
    AND (nrp.file_phot = gr.file_phot AND nrp.id_phot = gr.id_phot) /* match to photometry */
    AND ne.file = nrp.file_spec
    AND ne.file = nrm.file
    ORDER BY grating
    """)

    # write SQL query output to astropy Table
    phot_tab = Table(nrp_gr_match[-1])
    # write table to output file
    phot_tab.write(f'{file_path}{fname_phot_out}', format='ascii.commented_header', overwrite=True)

