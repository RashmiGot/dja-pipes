# ------- LIBRARIES ------ #
import bagpipes as pipes
import numpy as np
from eazy import filters

from astropy.io import fits
from astropy.cosmology import Planck13 as cosmo
from astropy.table import Table

import re

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
                   use_msa_resamp=False, fit_agn=False, fit_dla=False, fit_mlpoly=False,
                   spec_only=False):

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
    delayed = {}                         # delayed Tau model t*e^-(t/tau)
    delayed["age"] = (0.1, 9.)           # time since SF began: Gyr
    delayed["tau"] = (0.1, 9.)           # timescale of decrease: Gyr
    delayed["massformed"] = (6., 13.)
    delayed["metallicity"] = (0.2, 1.)   # in Zsun

    
    # ## ---------- ## continuity sfh (non-parametric)
    def get_age_bins(z, n_age_bins=10):

        # sets max age to age of universe relative to age at z=30 (which is ~100 Myr after BB)
        max_age = cosmo.age(0).to('Myr').value - cosmo.age(30).to('Myr').value
        age_at_z = cosmo.age(z).to('Myr').value - cosmo.age(30).to('Myr').value

        age_bins = [0., 3., 10., 25., 50.] # sets initial two age bin edges
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
        
        # set width of Student's t-distribution
        if sfh=="continuity":
            continuity["dsfr" + str(i) + "_prior_scale"] = 0.3  # as in Leja+19 (smooth continuity prior)
        elif sfh=="bursty_continuity":
            continuity["dsfr" + str(i) + "_prior_scale"] = 1.0  # as in Tacchella+22 (bursty continuity prior)
        
        continuity["dsfr" + str(i) + "_prior_df"] = 2           # defaults to this value as in Leja+19

    # setting the preferred sfh
    if sfh=="continuity" or sfh=="bursty_continuity":
        fit_instructions["continuity"] = continuity
    elif sfh=="dblplaw":
        fit_instructions["dblplaw"] = dblplaw
    elif sfh=="delayed":
        fit_instructions["delayed"] = delayed
    
    ## ---------- ## nebular emisison, logU
    nebular = {}
    nebular["logU"] = (-4.0, -1.0)
    nebular["logU_prior"] = "uniform"
    fit_instructions["nebular"] = nebular
    
    ## ---------- ## dust law
    dust = {}
    if dust_type=="calzetti":
        dust["type"] = "Calzetti"            # shape of the attenuation curve
        dust["Av"] = (0., 6.)                # vary Av between 0 and 6 mag
    elif dust_type=="CF00":
        dust["type"] = "CF00"
        dust["eta"] = 2.
        dust["Av"] = (0., 6.0)
        dust["n"] = (0.3, 2.5)
        dust["n_prior"] = "Gaussian"
        dust["n_prior_mu"] = 0.7
        dust["n_prior_sigma"] = 0.3
    elif dust_type=="salim":                 # parameters taken from Carnall+24 (EXCELS)
        dust["type"] = "Salim"
        dust["Av"] = (0., 6.)                # vary Av mag
        dust["delta"] = (-1.0, 0.3)          # vary att. curve slope (-1.0 taken roughly from Salim+18 from high-z analogues, see eq. 7)
        dust["delta_prior"] = "Gaussian"     # prior on att. curve slope
        dust["delta_prior_mu"] = 0           # avg. of prior on att. curve slope
        dust["delta_prior_sigma"] = 0.1      # standard dev. of prior on att. curve slope
        dust["B"] = (0., 5)                  # vary 2175A bump strength
        dust["B_prior"] = "uniform"          # prior on 2175A bump strength
    elif dust_type=="salim_fixed":           # see pg. 13 of Salim+18 (average curve of all SFR galaxies)
        dust["type"] = "Salim"
        dust["Av"] = (0., 6.)                # vary Av mag
        dust["delta"] = -0.4                 # vary att. curve slope
        dust["B"] = 1.3                      # vary 2175A bump strength
    elif dust_type=="kriek":
        dust["type"] = "Salim"               # Specify parameters within the "Salim" model to match Kriek & Conroy 2013
        dust["Av"] = (0., 6.)                # vary Av magnitude
        dust["eta"] = 2.0#1.0/0.4 - 1        # multiplicative factor on Av for stars in birth clouds
        dust["delta"] = -0.2                 # similar to Kriek & Conroy 2013
        dust["B"] = 1                        # similar to Kriek & Conroy 2013
    fit_instructions["dust"] = dust

    ## ---------- ## max age of birth clouds: Gyr
    fit_instructions["t_bc"] = 0.025         # 25 Myr birth cloud age
    
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

    if not spec_only:
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
    
        ## ---------- ##
        if fit_mlpoly:
            mlpoly = {}
            mlpoly["type"] = "polynomial_max_like"
            mlpoly["order"] = 4
            fit_instructions["calib"] = mlpoly

    ## ---------- ## white noise scaling (parameters chosen as in Carnall+24 - EXCELS)
    noise = {}
    noise["type"] = "white_scaled"
    noise["scaling"] = (0.1, 10.)
    noise["scaling_prior"] = "log_10"
    fit_instructions["noise"] = noise

    return(fit_instructions)


def run_pipes_on_dja_spec(file_spec="rubies-egs61-v3_prism-clear_4233_42328.spec.fits",
                          valid_threshold=400,
                          spec_only=False,
                          mask_lines=False, line_wavs=np.array([4970, 6562.81]), delta_lam=0,
                          sfh="continuity", n_age_bins=10, scale_disp=1.3, dust_type="kriek",
                          msa_line_components=None,
                          extended_prism_wavs=False,
                          use_msa_resamp=False, fit_agn=False, fit_dla=False, fit_mlpoly=False,
                          pool=4,
                          make_plots=True, save_tabs=True,
                          suffix=None,
                          **kwargs):
    """
    Runs bagpipes on spectrum from DJA AWS database, saves posteriors as .h5 files and plots as .pdf files
    
    Parameters
    ----------
    file_spec : DJA spectrum name, format=str
    valid_threshold : minimum number of valid datapoints to have in spectrum datafile
    spec_only : fit spectrum only (ignore photometry), format=bool
    mask_lines : to mask lines or not, format=bool
    line_wavs : rest-frame wavelengths of lines to mask in angstroms, format=numpy array
    delta_lam : width of masking region in angstroms, format=float
    sfh : name of star-formation history, format=string
    n_age_bins : number of bins for SFH, only needed if sfh="continuity", format=int
    scale_disp : multiplicative factor on LSF, format=float
    dust_type : specifies dust law; can be 'calzetti', 'CF00', 'salim', 'kriek', format=str
    msa_line_components : names of line components to add to bagpipes model when loglikelihood is calculated between model and data, format=list of strings
    extended_prism_wavs : extend PRISM wavelengths for match DJA v4, format=bool
    use_msa_resamp : use resampling function from msaexp, format=bool
    fit_agn : add AGN prior in bagpipes, format=bool
    fit_dla : add DLA prior in bagpipes, format=bool
    fit_mpoly : add mploy prior in bagpipes for polynomial scaling, format=bool
    make_plots : save plots, format=bool
    save_tabs : save bagpipes outputs to tables, format=bool
    suffix : add suffix to file names of saved outputs, format=str

    Returns
    -------
    fit : bagpipes fit object
    """

    # name of bagpipes run
    runName = file_spec.split('.spec.fits')[0]
    
    # make pipes folders if they don't already exist
    pipes.utils.make_dirs(run=runName)

    ##################################
    # ---- PULLING DATA FROM DB ---- #
    ##################################

    # id and dja name of spectrum
    runid0 = runName.split('_')[-1]
    runid = re.findall(r'\d+', runid0)[0]

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

    # checks spectrum for missing flux datapoints
    num_valid, is_valid = fitting.check_spec(ID=runid, valid_threshold=valid_threshold)

    if not is_valid:
        print("Spectrum not valid")
        return None
    
    # spectroscopic redshift
    z_spec = database.pull_zspec_from_db(fname_spec)

    _ = fitting.mask_emission_lines(fname_spec, z_spec, mask_lines=mask_lines, line_wavs=line_wavs, delta_lam=delta_lam)

    # photometry
    if not spec_only:
        try:
            database.pull_phot_from_db(fname_spec, fname_phot, filePath)
        except (IndexError, ValueError):
            print("No photometry found, fitting only spectrum")
            spec_only = True

    ##################################
    # -------- BAGPIPES RUN -------- #
    ##################################

    if spec_only:
        filt_list = None

        galaxy = pipes.galaxy(runid, fitting.load_spec, filt_list=filt_list,
                              spectrum_exists=True,
                              photometry_exists=False,
                              spec_units='ergscma',
                              out_units='ergscma')
    elif not spec_only:
        # jwst filter list
        filt_list = fitting.updated_filt_list(runid)

        # making galaxy object
        galaxy = pipes.galaxy(runid, fitting.load_both, filt_list=filt_list,
                              spec_units='ergscma',
                              phot_units='ergscma',
                              out_units='ergscma')
        
        # store filter normalisations to galaxy object
        spec = galaxy.spectrum
        spec_tab = Table([spec[:,0]/1e4, spec[:,1], spec[:,2]], names=('wave', 'flux', 'err'))
        galaxy.filter_int_arr, galaxy.filt_norm_arr, galaxy.filt_valid = fitting.calc_filt_int(filt_list, spec_tab, z_spec)
    
    # generating fit instructions
    fit_instructions = fitting_params(runid, z_spec, sfh=sfh, n_age_bins=n_age_bins, scale_disp=scale_disp,
                                      dust_type=dust_type,
                                      use_msa_resamp=use_msa_resamp, fit_agn=fit_agn, fit_dla=fit_dla, fit_mlpoly=fit_mlpoly,
                                      spec_only=spec_only)
    
    # interpolated prism resolution curve 
    R_curve_interp = np.interp(galaxy.spectrum[:, 0]/10000,
                               fit_instructions["R_curve"][:,0]/10000,
                               fit_instructions["R_curve"][:,1])
    galaxy.R_curve_interp = R_curve_interp

    galaxy.msa_line_components = msa_line_components
    galaxy.msa_phot = None
    if msa_line_components is not None:
        # add msa line components to galaxy object
        galaxy.msa_model = []
        galaxy.msa_phot = []
        # add component to store least-squares coeffs. from msaexp fit
        galaxy.lsq_coeffs = []

    # specify whether or not to use extended wavelength range
    galaxy.extended_prism_wavs = extended_prism_wavs

    # suffix to add to saved file names
    if suffix is None:
        suffix = sfh + '_' + dust_type
    if suffix is not None:
        suffix = sfh + '_' + dust_type + '_' + suffix
    
    # check if posterior file exists
    full_posterior_file = f'pipes/posterior/{runName}/{runName}_{suffix}.h5'
    run_posterior_file = f'pipes/posterior/{runName}/{runid}.h5'
    
    if os.path.isfile(full_posterior_file):
        os.rename(full_posterior_file, run_posterior_file)

    # making fit object
    fit = pipes.fit(galaxy, fit_instructions, run=runName)

    # fitting  spectrum and photometry with bagpipes
    fit.fit(verbose=False, sampler='nautilus', pool=pool)

    ##################################
    # ---------- PLOTTING ---------- #
    ##################################

    if make_plots:
        # plot fitted model
        _, plotlims_flam = plotting.plot_fitted_spectrum(fit, fname_spec, z_spec=z_spec, suffix=suffix,
                                                        spec_only=spec_only, f_lam=True, save=True, return_plotlims=True)
        _, plotlims_fnu = plotting.plot_fitted_spectrum(fit, fname_spec, z_spec=z_spec, suffix=suffix,
                                                        spec_only=spec_only, f_lam=False, save=True, return_plotlims=True)
        # plot data
        _ = plotting.plot_spec_phot_data(runid, fname_spec, fname_phot, z_spec=z_spec, suffix=suffix,
                                        spec_only=spec_only, f_lam=True, show=False, save=True, run=runName, plotlims=plotlims_flam)
        _ = plotting.plot_spec_phot_data(runid, fname_spec, fname_phot, z_spec=z_spec, suffix=suffix,
                                        spec_only=spec_only, f_lam=False, show=False, save=True, run=runName, plotlims=plotlims_fnu)
        # plot star-formation history
        _ = plotting.plot_fitted_sfh(fit, fname_spec, z_spec=z_spec, suffix=suffix, save=True)
        # plot posterior corner plot
        _ = plotting.plot_corner(fit, fname_spec, z_spec, fit_instructions, filt_list, suffix=suffix,
                                 spec_only=spec_only, save=True)
        if not spec_only:
            # plot calibration curve
            _ = plotting.plot_calib(fit, fname_spec, z_spec=z_spec, suffix=suffix,
                                    save=True, plot_xlims=[plotlims_flam[0],plotlims_flam[1]])

    if save_tabs:
        # save posterior quantities to table
        _ = plotting.save_posterior_sample_dists(fit, fname_spec, spec_only, suffix=suffix, save=True)
        _ = plotting.save_posterior_line_fluxes(fit, fname_spec, suffix=suffix, save=True)
        if not spec_only:
            # save calib curve to table
            _ = plotting.save_calib(fit, fname_spec, suffix=suffix, save=True)
        if msa_line_components is not None:
            _ = plotting.save_posterior_msa_lsq_line_fluxes(fit, fname_spec, suffix=suffix, save=True)

    # rename posterior
    os.rename(run_posterior_file, full_posterior_file)

    return fit