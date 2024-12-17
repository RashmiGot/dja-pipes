# ------- LIBRARIES ------ #
import bagpipes as pipes
import numpy as np

from astropy.io import fits
from astropy.cosmology import Planck13 as cosmo

import os
import grizli.utils

from . import database
from . import fitting
from . import plotting
from . import utils as djautils

# check if 'files' directory exists, else make one
if not os.path.exists("./files"):
    os.mkdir("./files")

# defines bagpipes fit_instructions dictionary 
def fitting_params(runid, z_spec, sfh="continuity", n_age_bins=10, scale_disp=1.3, dust_type="kriek", 
                   use_msa_resamp=False, fit_agn=False, fit_dla=False, fit_mlpoly=False):

    fit_instructions = {}
    
    ## ---------- ## double power-law sfh (parametric)
    dblplaw = {}                        
    dblplaw["tau"] = (0., 15.)            
    dblplaw["alpha"] = (0.01, 1000.)
    dblplaw["beta"] = (0.01, 1000.)
    dblplaw["alpha_prior"] = "log_10"
    dblplaw["beta_prior"] = "log_10"
    dblplaw["massformed"] = (1., 13.)
    dblplaw["metallicity"] = (0.003, 2.)
    dblplaw["metallicity_prior"] = "log_10"

    
    ## ---------- ## delayed-tau sfh (parametric)
    delayed = {}                         # Delayed Tau model t*e^-(t/tau)
    delayed["age"] = (0.1, 9.)           # Time since SF began: Gyr
    delayed["tau"] = (0.1, 9.)           # Timescale of decrease: Gyr
    delayed["massformed"] = (6., 13.)
    delayed["metallicity"] = (0.2, 1.)   # in Zsun

    
    # ## ---------- ## continuity sfh (non-parametric)
    def get_age_bins(z, n_age_bins=10):

        # sets max age to age of universe relative to age at z=30 (which is ~100 Myr after BB)
        max_age = cosmo.age(0).to('Myr').value - cosmo.age(30).to('Myr').value
        age_at_z = cosmo.age(z).to('Myr').value - cosmo.age(30).to('Myr').value

        age_bins = [0., 10., 50] # sets initial two age bin edges
        for i in np.logspace(np.log10(100), np.log10(max_age), n_age_bins):
            age_bins.append(i)
        age_bins=np.array(age_bins)

        # indeces of any age bin edges that are greater than max_age at z_obs
        last_index = np.where(age_bins > age_at_z)[0][0]

        # conditionally extends last age bin edge to age_at_z
        if (np.log10(age_at_z)-np.log10(age_bins[last_index-1]))<0.5*(np.log10(age_bins[last_index])-np.log10(age_bins[last_index-1])):
            age_bins[last_index-1] = age_at_z
            final_age_bins = age_bins[:last_index]
        elif (np.log10(age_at_z)-np.log10(age_bins[last_index-1]))>=0.5*(np.log10(age_bins[last_index])-np.log10(age_bins[last_index-1])):
            age_bins[last_index] = age_at_z
            final_age_bins = age_bins[:(last_index+1)]

        return final_age_bins
    
    continuity = {}
    continuity["massformed"] = (6., 14.)
    continuity["massformed_prior"] = "uniform"
    continuity["metallicity"] = (0.01, 2.5)
    continuity["metallicity_prior"] = "uniform"
    continuity["bin_edges"] = get_age_bins(z_spec, n_age_bins=n_age_bins).tolist()
    
    for i in range(1, len(continuity["bin_edges"])-1):
        continuity["dsfr" + str(i)] = (-10., 10.)
        continuity["dsfr" + str(i) + "_prior"] = "student_t"
        continuity["dsfr" + str(i) + "_prior_scale"] = 1.0  # Defaults to 0.3 as in Leja19, but can be set - 1 is bursty continuity prior from Tacchella+21
        continuity["dsfr" + str(i) + "_prior_df"] = 2       # Defaults to this value as in Leja19, but can be set
    continuity["age_min"] = 0

    # setting the preferred sfh
    if sfh=="continuity":
        fit_instructions["continuity"] = continuity
    elif sfh=="dblplaw":
        fit_instructions["dblplaw"] = dblplaw
    elif sfh=="delayed":
        fit_instructions["delayed"] = delayed
    
    ## ---------- ## nebular emisison, logU
    nebular = {}
    nebular["logU"] = (-4., -1.)
    nebular["logU_prior"] = "uniform"
    fit_instructions["nebular"] = nebular
    
    ## ---------- ## dust law
    dust = {}
    if dust_type=="calzetti":
        dust["type"] = "Calzetti"            # Shape of the attenuation curve
        dust["Av"] = (0., 6.)                # Vary Av between 0 and 4 magnitudes
    elif dust_type=="CF00":
        dust["type"] = "CF00"
        dust["eta"] = 2.
        dust["Av"] = (0., 6.0)
        dust["n"] = (0.3, 2.5)
        dust["n_prior"] = "Gaussian"
        dust["n_prior_mu"] = 0.7
        dust["n_prior_sigma"] = 0.3
    elif dust_type=="salim":
        dust["type"] = "Salim"
        dust["Av"] = (0., 6.)                # Vary Av magnitude
        dust["delta"] = (-0.3, 0.3)          # Vary att. slope
        dust["delta_prior"] = "Gaussian"     # prior on att. slope
        dust["delta_prior_mu"] = 0           # avg. of prior on att. slope
        dust["delta_prior_sigma"] = 0.1      # standard dev. of prior on att. slope
        dust["B"] = (0., 3)                  # Vary 2175A bump strength
        dust["B_prior"] = "uniform"          # prior on 2175A bump strength
    elif dust_type=="kriek":
        dust["type"] = "Salim"               # Specify parameters within the "Salim" model to match Kriek & Conroy 2013
        dust["Av"] = (0., 6.)                # Vary Av magnitude
        dust["eta"] = 2.0#1.0/0.4 - 1        # Multiplicative factor on Av for stars in birth clouds
        dust["delta"] = -0.2                 # Similar to Kriek & Conroy 2013
        dust["B"] = 1                        # Similar to Kriek & Conroy 2013
    fit_instructions["dust"] = dust

    ## ---------- ## max age of birth clouds: Gyr
    fit_instructions["t_bc"] = 0.01
    
    ## ---------- ## agn component
    agn = {}
    agn["alphalam"] = (-2., 2.)
    agn["betalam"] = (-2., 2.)
    agn["f5100A"] = (0, 1e-19)
    agn["sigma"] = (1e3, 5e3)
    agn["hanorm"] = (0,2.5e-17)
    if fit_agn:
        fit_instructions["agn"] = agn
    
    ## ---------- ## tight redshift prior around z_spec
    fit_instructions["redshift"] = (
        z_spec - 0.005*(1+z_spec),
        z_spec + 0.005*(1+z_spec)
    )

    fit_instructions["redshift_prior"] = "Gaussian"
    fit_instructions["redshift_prior_mu"] = z_spec
    fit_instructions["redshift_prior_sigma"] = 0.001 * (1+z_spec)

    ## ---------- ## jwst prism resolution curve
    fit_instructions["R_curve"] = djautils.load_prism_dispersion(scale_disp=scale_disp)

    ## ---------- ## boolean for using msa resampling
    fit_instructions["use_msa_resamp"] = use_msa_resamp

    ## ---------- ## fixed velocity dispersion
    fit_instructions["veldisp"] = 100.   # km/s

    ## ---------- ## dla component (doesn't work?)
    dla = {}
    dla["zabs"] = z_spec
    dla["t"] = 22.
    if fit_dla:
        fit_instructions["dla"] = dla

    ## ---------- ## calibration curve (2nd order polynomial)
    cfit, cfit_err, _ = fitting.guess_calib(runid, z=z_spec) # guessing initial calibration coefficients

    calib = {}
    calib["type"] = "polynomial_bayesian"
    
    calib["0"] = (cfit[0]-5.*cfit_err[0], cfit[0]+5.*cfit_err[0])
    calib["0_prior"] = "Gaussian"
    calib["0_prior_mu"] = cfit[0]
    calib["0_prior_sigma"] = cfit_err[0]
    
    calib["1"] = (cfit[1]-5.*cfit_err[1], cfit[1]+5.*cfit_err[1])
    calib["1_prior"] = "Gaussian"
    calib["1_prior_mu"] = cfit[1]
    calib["1_prior_sigma"] = cfit_err[1]
    
    calib["2"] = (cfit[2]-5.*cfit_err[2], cfit[2]+5.*cfit_err[2])
    calib["2_prior"] = "Gaussian"
    calib["2_prior_mu"] = cfit[2]
    calib["2_prior_sigma"] = cfit_err[2]
    fit_instructions["calib"] = calib
    
    # ## ---------- ##
    if fit_mlpoly:
        mlpoly = {}
        mlpoly["type"] = "polynomial_max_like"
        mlpoly["order"] = 4
        fit_instructions["calib"] = mlpoly

    ## ---------- ## white noise scaling
    noise = {}
    noise["type"] = "white_scaled"
    noise["scaling"] = (1.,10.)
    noise["scaling_prior"] = "log_10"
    fit_instructions["noise"] = noise

    return(fit_instructions)


def run_pipes_on_dja_spec(file_spec="rubies-egs61-v3_prism-clear_4233_42328.spec.fits", sfh="continuity", n_age_bins=10, scale_disp=1.3, dust_type="kriek", 
                   use_msa_resamp=False, fit_agn=False, fit_dla=False, fit_mlpoly=False, **kwargs):
    """
    Runs bagpipes on spectrum from DJA AWS database, saves posteriors as .h5 files and plots as .pdf files
    
    Parameters
    ----------
    file_spec : DJA spectrum name, format=str

    Returns
    -------
    None
    """

    # name of bagpipes run
    runName = file_spec.split('.spec.fits')[0]
    
    # make pipes folders if they don't already exist
    pipes.utils.make_dirs(run=runName)

    ##################################
    # ---- PULLING DATA FROM DB ---- #
    ##################################

    # id and dja name of spectrum
    runid = runName.split('_')[-1]

    print(runid, runName)
    
    ##############
    # temp catalog, though not clear why this needs to be a catalog file at all
    row = [runid, runName]
    tab = grizli.utils.GTable(names=["id", "fname"], rows=[row])
    tab.write("spec_cat_temp.csv", overwrite=True)
    
    # spectrum and photometry filenames
    fname_spec = runName+'.spec.fits'
    fname_phot = runName+'.phot.cat'

    # path to store spectrum and photometry files
    filePath = 'files/'

    # pull spectrum and photometry from the aws database
    database.pull_spec_from_db(fname_spec, filePath)

    try:
        database.pull_phot_from_db(fname_spec, fname_phot, filePath)
    except IndexError:
        print("No photometry found")
        return None

    # spectroscopic redshift
    z_spec = database.pull_zspec_from_db(fname_spec)

    ##################################
    # -------- BAGPIPES RUN -------- #
    ##################################

    # jwst filter list
    filt_list = fitting.updated_filt_list(runid)

    # making galaxy object
    galaxy = pipes.galaxy(runid, fitting.load_both, filt_list=filt_list,
                        spec_units='ergscma',
                        phot_units='ergscma',
                        out_units="ergscma")

    # generating fit instructions
    fit_instructions = fitting_params(runid, z_spec, sfh=sfh, n_age_bins=n_age_bins, scale_disp=scale_disp,
                                      dust_type=dust_type,
                                      use_msa_resamp=use_msa_resamp, fit_agn=fit_agn, fit_dla=fit_dla, fit_mlpoly=fit_mlpoly)

    # check if posterior file exists
    suffix = sfh + '_' + dust_type
    
    full_posterior_file = f'pipes/posterior/{runName}/{runName}_{suffix}.h5'
    run_posterior_file = f'pipes/posterior/{runName}/{runid}.h5'
    
    if os.path.isfile(full_posterior_file):
        os.rename(full_posterior_file, run_posterior_file)

    # making fit object
    fit = pipes.fit(galaxy, fit_instructions, run=runName)

    # fitting  spectrum and photometry with bagpipes
    fit.fit(verbose=False, sampler='nautilus', pool=10)

    ##################################
    # ---------- PLOTTING ---------- #
    ##################################

    # fitted model
    _, plotlims_flam = plotting.plot_fitted_spectrum(fit, fname_spec, z_spec=z_spec, suffix=suffix,
                                                     f_lam=True, save=True, return_plotlims=True)
    _, plotlims_fnu = plotting.plot_fitted_spectrum(fit, fname_spec, z_spec=z_spec, suffix=suffix,
                                                     f_lam=False, save=True, return_plotlims=True)
    # # save data to plot
    _ = plotting.plot_spec_phot_data(fname_spec, fname_phot, z_spec=z_spec, suffix=suffix,
                                     f_lam=True, show=False, save=True, run=runName, plotlims=plotlims_flam)
    _ = plotting.plot_spec_phot_data(fname_spec, fname_phot, z_spec=z_spec, suffix=suffix,
                                     f_lam=False, show=False, save=True, run=runName, plotlims=plotlims_fnu)
    # # star-formation history
    _ = plotting.plot_fitted_sfh(fit, fname_spec, z_spec=z_spec, suffix=suffix, save=True)
    # # posterior corner plot
    _ = plotting.plot_corner(fit, fname_spec, z_spec, fit_instructions, filt_list, suffix=suffix, save=True)
    # # calibration curve
    _ = plotting.plot_calib(fit, fname_spec, z_spec=z_spec, suffix=suffix,
                            save=True, plot_xlims=[plotlims_flam[0],plotlims_flam[1]])

    # # saving posterior quantities to table
    _ = plotting.save_posterior_sample_dists(fit, fname_spec, suffix=suffix, save=True)
    # # saving calib curve to table
    _ = plotting.save_calib(fit, fname_spec, suffix=suffix, save=True)

    # rename posterior
    os.rename(run_posterior_file, full_posterior_file)

    return fit