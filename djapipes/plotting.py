# !!! ADAPTED FROM DEFAULT BAGPIPES PLOTTING FUNCTIONS !!!

# ------- LIBRARIES ------ #
import numpy as np
from astropy.cosmology import Planck13 as cosmo
from astropy.table import Table, hstack
from grizli.utils import figure_timestamp
from grizli.utils import MPL_COLORS

import corner
import bagpipes as pipes
import copy
import os

from . import fitting
from . import utils as djautils

# ------- PLOTTING & FORMATTING ------- #
import matplotlib.pyplot as plt
from matplotlib import colors

params = {'legend.fontsize':12,'axes.labelsize':16,'axes.titlesize':10,
          'xtick.labelsize':16,'ytick.labelsize':16,
          'xtick.top':True,'ytick.right':True,'xtick.direction':'in','ytick.direction':'in',
          "xtick.major.size":10,"xtick.minor.size":5,"ytick.major.size":10,"ytick.minor.size":5}
plt.rcParams.update(params)
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'


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


def prism_wav_xticks(xmin=1, xmax=5, dx=0.5):
    return np.arange(xmin,xmax+dx,dx)


# --------------------------------------------------------------
# ---------------------- PLOT SPECTROSCOPIC AND PHOTOMETRIC DATA
# --------------------------------------------------------------
def plot_spec_phot_data(fname_spec, fname_phot, z_spec, suffix, f_lam=False, show=False, save=False, run='.', plotlims=None):
    """
    Plots spectrum and photometry of given source
    
    Parameters
    ----------
    fname_spec : filename of spectrum e.g. 'rubies-uds3-v3_prism-clear_4233_62812.spec.fits', format=str
    fname_phot : filename of photometry e.g. 'rubies-uds3-v3_prism-clear_4233_62812.phot.cat', format=str
    z_spec : spectroscopic redshift, format=float
    f_lam : if True, output plot is in f_lambda, if False, in f_nu, format=bool
    show : specifies wherether or not to display the image, format=bool
    save : specifies wherether or not to save the image, format=bool
    run : name of bagpipes run, defaults to no run name, format=str

    Returns
    -------
    None
    """
    
    # spectrum
    spec_tab = Table.read(f'files/{fname_spec}', hdu=1)
    spec_fluxes = spec_tab['flux']
    spec_efluxes = spec_tab['err']
    spec_wavs = spec_tab['wave']

    # photometry
    phot_tab = Table.read(f'files/{fname_phot}', format='ascii.commented_header')
    

    # jwst filter list
    # filt_list = np.loadtxt("../filters/filt_list.txt", dtype="str")
    #
    # # extract fluxes from cat
    # flux_colNames = [filt_list_i.split('/')[-1].split('.')[0]+'_tot_1' for filt_list_i in filt_list]
    # eflux_colNames = [filt_list_i.split('/')[-1].split('.')[0]+'_etot_1' for filt_list_i in filt_list]
    #
    # # zeropoints table
    # zpoints = Table.read('zeropoints.csv', format='csv')

    filt_list = djautils.read_filter_list("filt_list.txt")

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
    
    phot_fluxes_temp = np.lib.recfunctions.structured_to_unstructured(np.array(phot_tab[list(flux_colNames)]))
    phot_fluxes_temp = phot_fluxes_temp * zp_array
    phot_efluxes_temp = np.lib.recfunctions.structured_to_unstructured(np.array(phot_tab[list(eflux_colNames)]))

    phot_flux_mask = (phot_fluxes_temp>-90) & (phot_efluxes_temp>0)

    phot_fluxes = phot_fluxes_temp[phot_flux_mask]
    phot_efluxes = phot_efluxes_temp[phot_flux_mask]
    
    # effective wavelengths of photometric filters
    phot_wavs_temp = (calc_eff_wavs(filt_list) / 10000)
    phot_wavs = np.array(phot_wavs_temp)[phot_flux_mask[0]]   

    # plotting spectrum
    if f_lam:
        spec_fluxes = fitting.convert_mujy2cgs(spec_fluxes, spec_wavs*10000)
        spec_efluxes = fitting.convert_mujy2cgs(spec_efluxes, spec_wavs*10000)
        
        phot_fluxes = fitting.convert_mujy2cgs(phot_fluxes, phot_wavs*10000)
        phot_efluxes = fitting.convert_mujy2cgs(phot_efluxes, phot_wavs*10000)

    fig,ax = plt.subplots(figsize=(10,4.5))

    if plotlims==None:
        xmin_plot, xmax_plot = np.min(spec_wavs), np.max(spec_wavs)
        ymin_plot, ymax_plot = -0.1*np.max(spec_fluxes), 1.1*np.max(spec_fluxes)
    elif plotlims!=None:
        xmin_plot, xmax_plot, ymin_plot, ymax_plot = plotlims

    ax.hlines(y=0, xmin=spec_wavs.min(), xmax=spec_wavs.max(), lw=1.0, color='gainsboro', zorder=-1)

    ##################################
    # ------------ DATA ------------ #
    ##################################
    
    # ---------- SPECTRUM ---------- #
    ax.plot(spec_wavs, spec_fluxes,
            zorder=-1, color='slategrey', alpha=0.7, lw=1.2)
    ax.fill_between(spec_wavs, spec_fluxes-spec_efluxes, spec_fluxes+spec_efluxes,
                    zorder=-1, color='slategrey', alpha=0.1)

    # --------- PHOTOMETRY --------- #
    ax.errorbar(phot_wavs, phot_fluxes, yerr=phot_efluxes,
                fmt='o', ms=8, color='gainsboro', markeredgecolor='k', ecolor='grey', elinewidth=1, markeredgewidth=1.,
                zorder=1, alpha=1.)
            
    add_lines_msa(z_spec=z_spec)
    
    ##################################
    # --------- FORMATTING --------- #
    ##################################
    
    ax.set_xlabel('$\lambda_{\\rm obs}{\\rm \\ [\mu m]}$')
    ax.set_ylabel('${f_{\\nu}} {\\rm\\ [\\mu Jy]}$')
    if f_lam:
        ax.set_ylabel('$f_{\\lambda} {\\rm\\ [erg\ s^{-1} cm^{-2} \AA^{-1}]}$')

    ax.set_xlim(xmin_plot, xmax_plot)
    ax.set_ylim(ymin_plot, ymax_plot)

    ax.set_xticks(prism_wav_xticks())
    
    # ax.legend(loc='upper left')
    fname = fname_spec.split('.spec')[0]
    ax.set_title(fname+'          $z=$'+str(np.round(z_spec,4)), loc='right')

    figure_timestamp(fig, x=0.97, y=0.04, fontsize=8)
    
    plt.tight_layout()

    if save:
        # path to location where plots are saved
        if run != ".":
            plotPath = "pipes/plots/" + run
        elif run == ".":
            plotPath = "pipes/plots"

        str_ext = '_flam'
        if not f_lam:
            str_ext = '_fnu'
        imname = fname + '_' + suffix + '_data' + str_ext
        # plt.savefig(f'{plotPath}/{imname}.pdf', transparent=True)
        plt.savefig(f'{plotPath}/{imname}.png', transparent=True)
        plt.close(fig)

    if show:
        plt.show()
        plt.close(fig)

    return fig


# --------------------------------------------------------------
# ----------------------------------- PLOT FITTED SPECTRAL MODEL
# --------------------------------------------------------------
def plot_fitted_spectrum(fit, fname_spec, z_spec, suffix, f_lam=False, show=False, save=False, return_plotlims=False):
    """
    Plots fitted BAGPIPES spectral model, observed spectrum and observed photometry of given source
    
    Parameters
    ----------
    fit : fit object from BAGPIPES (where fit = pipes.fit(galaxy, fit_instructions))
    fname_spec : filename of spectrum e.g. 'rubies-uds3-v3_prism-clear_4233_62812.spec.fits', format=str
    z_spec : spectroscopic redshift, format=float
    f_lam : if True, output plot is in f_lambda, if False, in f_nu, format=bool
    show : specifies wherether or not to display the image, format=bool
    save : specifies wherether or not to save the image, format=bool

    Returns
    -------
    None
    """

    fit.posterior.get_advanced_quantities()

    ymax = 1.05*np.max(fit.galaxy.spectrum[:, 1])
    
    y_scale = float(int(np.log10(ymax))-1)
    
    wavs = fit.galaxy.spectrum[:, 0]/10000
    
    spec_post = np.copy(fit.posterior.samples["spectrum"])
    
    # if "calib" in list(fit.posterior.samples):
    #     spec_post /= fit.posterior.samples["calib"]
    
    if "noise" in list(fit.posterior.samples):
        spec_post += fit.posterior.samples["noise"]
    
    post = np.percentile(spec_post, (16, 50, 84), axis=0).T#*10**-y_scale

    calib_50 = np.percentile(fit.posterior.samples["calib"], 50, axis=0).T

    phot_post = np.percentile(fit.posterior.samples["photometry"], (16, 50, 84), axis=0).T

    spec_fluxes = fit.galaxy.spectrum[:,1]*calib_50
    spec_efluxes = fit.galaxy.spectrum[:,2]*calib_50

    spec_fluxes_model = post[:,1]
    spec_fluxes_model_lo = post[:,0]
    spec_fluxes_model_hi = post[:,2]

    phot_flux_mask = fit.galaxy.photometry[:,2]<1e90

    phot_wavs = (fit.galaxy.filter_set.eff_wavs/10000)[phot_flux_mask]
    phot_fluxes = (fit.galaxy.photometry[:,1])[phot_flux_mask]
    phot_efluxes = (fit.galaxy.photometry[:,2])[phot_flux_mask]

    phot_fluxes_model = (phot_post[:,1])[phot_flux_mask]
    phot_fluxes_model_lo = (phot_post[:,0])[phot_flux_mask]
    phot_fluxes_model_hi = (phot_post[:,2])[phot_flux_mask]

    if not f_lam:
        spec_fluxes = fitting.convert_cgs2mujy(spec_fluxes, wavs*10000)
        spec_efluxes = fitting.convert_cgs2mujy(spec_efluxes, wavs*10000)

        spec_fluxes_model = fitting.convert_cgs2mujy(spec_fluxes_model, wavs*10000)
        spec_fluxes_model_lo = fitting.convert_cgs2mujy(spec_fluxes_model_lo, wavs*10000)
        spec_fluxes_model_hi = fitting.convert_cgs2mujy(spec_fluxes_model_hi, wavs*10000)

        phot_fluxes = fitting.convert_cgs2mujy(phot_fluxes, phot_wavs*10000)
        phot_efluxes = fitting.convert_cgs2mujy(phot_efluxes, phot_wavs*10000)

        phot_fluxes_model = fitting.convert_cgs2mujy(phot_fluxes_model, phot_wavs*10000)
        phot_fluxes_model_lo = fitting.convert_cgs2mujy(phot_fluxes_model_lo, phot_wavs*10000)
        phot_fluxes_model_hi = fitting.convert_cgs2mujy(phot_fluxes_model_hi, phot_wavs*10000)
    
    # plotting spectrum
    fig,ax = plt.subplots(figsize=(10,4.5))

    ax.hlines(y=0, xmin=wavs.min(), xmax=wavs.max(), lw=1.0, color='gainsboro', zorder=-1)
    
    ##################################
    # ------------ DATA ------------ #
    ##################################
    
    # ---------- SPECTRUM ---------- #
    ax.plot(wavs, spec_fluxes,
            zorder=-1, color='slategrey', alpha=0.7, lw=1, label='Spectrum (scaled)')
    ax.fill_between(wavs,
                    spec_fluxes-spec_efluxes,
                    spec_fluxes+spec_efluxes,
                    zorder=-1, color='slategrey', alpha=0.1)
    
    # ---------- PHOTOMETRY ---------- #
    ax.errorbar(phot_wavs, phot_fluxes, yerr=phot_efluxes,
                fmt='o', ms=8, color='gainsboro', markeredgecolor='k', ecolor='grey', elinewidth=0.5, markeredgewidth=1.,
                zorder=1, alpha=1., label='Photometry')
    
    ##################################
    # -------- FITTED MODEL -------- #
    ##################################
    
    # ---------- SPECTRUM ---------- #
    ax.plot(wavs, spec_fluxes_model,
            zorder=-1, color='firebrick', alpha=0.7, lw=1.5, label='Model spectrum')
    ax.fill_between(wavs,
                    spec_fluxes_model_lo, spec_fluxes_model_hi,
                    zorder=-1, color='firebrick', alpha=0.1)
    
    # ---------- PHOTOMETRY ---------- #
    ax.errorbar(phot_wavs, phot_fluxes_model, yerr=[phot_fluxes_model-phot_fluxes_model_lo, phot_fluxes_model_hi-phot_fluxes_model],
                fmt='o', ms=7, color='firebrick', markeredgecolor='k', ecolor='grey', elinewidth=0.5, markeredgewidth=.5,
                zorder=1, alpha=0.7, label='Model photometry')

    ##################################
    # ------- EMISSION LINES ------- #
    ##################################

    add_lines_msa(z_spec=z_spec)
    
    ##################################
    # --------- FORMATTING --------- #
    ##################################
    
    ax.set_xlabel('$\lambda_{\\rm obs}{\\rm \\ [\mu m]}$')
    ax.set_ylabel('${f_{\\lambda}}{\\rm\\ [erg\ s^{-1} cm^{-2} \AA^{-1}]}$')
    if not f_lam:
        ax.set_ylabel('${f_{\\nu}} {\\rm\\ [\\mu Jy]}$')

    xmin_plot, xmax_plot = np.min(wavs), np.max(wavs)
    ymin_plot, ymax_plot = -0.1*np.max(spec_fluxes), 1.1*np.max(spec_fluxes)

    ax.set_xlim(xmin_plot, xmax_plot)
    ax.set_ylim(ymin_plot, ymax_plot)

    ax.set_xticks(prism_wav_xticks())
    
    ax.legend(loc='upper left', framealpha=0.5)

    fname = fname_spec.split('.spec')[0]
    ax.set_title(fname+'          $z=$'+str(np.round(z_spec,4)), loc='right')

    figure_timestamp(fig, x=0.97, y=0.04, fontsize=8)
    
    plt.tight_layout()
    
    if save:
        str_ext = '_flam'
        if not f_lam:
            str_ext = '_fnu'
        plotpath = "pipes/plots/" + fit.run + "/" + fname +  '_' + suffix + '_specfit' + str_ext
        # plt.savefig(plotpath+'.pdf', transparent=True)
        plt.savefig(plotpath+'.png', transparent=True)
        plt.close(fig)

    if show:
        plt.show()
        plt.close(fig)

    if return_plotlims:
        return fig, [xmin_plot, xmax_plot, ymin_plot, ymax_plot]
    else:
        return fig


# --------------------------------------------------------------
# ---------------------------------- PLOT STAR-FORMATION HISTORY
# --------------------------------------------------------------
def plot_fitted_sfh(fit, fname_spec, z_spec, suffix, show=False, save=False):
    """
    Plots star-formation history from fitted BAGPIPES model
    
    Parameters
    ----------
    fit : fit object from BAGPIPES (where fit = pipes.fit(galaxy, fit_instructions))
    fname_spec : filename of spectrum e.g. 'rubies-uds3-v3_prism-clear_4233_62812.spec.fits', format=str
    z_spec : spectroscopic redshift, format=float
    show : specifies wherether or not to display the image, format=bool
    save : specifies wherether or not to save the image, format=bool

    Returns
    -------
    None
    """
    
    fit.posterior.get_advanced_quantities()
    
    color1 = "firebrick"
    color2 = "firebrick"
    alpha = 0.6
    zorder=4
    label=None
    # zvals=[0, 0.5, 1, 2, 4, 10]
    # z_axis=True

    z_array = np.arange(0., 100., 0.01)
    age_at_z = cosmo.age(z_array).value

    age_of_universe_lim = cosmo.age(3).value
    if z_spec<3:
        age_of_universe_lim = cosmo.age(1).value
    
    # Calculate median redshift and median age of Universe
    if "redshift" in fit.fitted_model.params:
        redshift = np.median(fit.posterior.samples["redshift"])
    else:
        redshift = fit.fitted_model.model_components["redshift"]

    age_of_universe = cosmo.age(z_spec).value#np.interp(redshift, z_array, age_at_z)

    # Calculate median and confidence interval for SFH posterior
    post = np.percentile(fit.posterior.samples["sfh"], (16, 50, 84), axis=0).T

    # Plot the SFH
    # x = age_of_universe - fit.posterior.sfh.ages*10**-9
    x = fit.posterior.sfh.ages*10**-9
    xmask = x<age_of_universe

    fig,ax = plt.subplots(figsize=(10,4.5))

    ax.plot(x[xmask], post[:, 1][xmask], color=color1, zorder=zorder+1)
    ax.fill_between(x[xmask], post[:, 0][xmask], post[:, 2][xmask], color=color2,
                    alpha=alpha, zorder=zorder, lw=0, label=label)

    # ax.set_ylim(0., np.max([ax.get_ylim()[1], 1.1*np.max(post[:, 2])]))
    # ax.set_xlim(age_of_universe, 0)

    ymin, ymax = ax.get_ylim()
    ax.vlines(age_of_universe_lim, ymin=ymin, ymax=ymax, color='grey', ls='-')
    if z_spec<3:
        ax.text(0.97*age_of_universe_lim, 0.5*ymax, "Age of Universe at $z=1$", fontsize=12, rotation=90)
    else:
        ax.text(0.97*age_of_universe_lim, 0.5*ymax, "Age of Universe at $z=3$", fontsize=12, rotation=90)
    ax.vlines(age_of_universe, ymin=ymin, ymax=ymax, color='grey', ls='--')
    # ax.text(age_of_universe-0.03*age_of_universe_lim, 0.5*ymax, "Age of Universe at $z_{\\rm spec}$", fontsize=12, rotation=90)
    
    ax.set_xlim(-0.04*age_of_universe_lim, 1.04*age_of_universe_lim)

    # Set axis labels
    ax.set_ylabel("${\\rm SFR \ [ M_\\odot yr^{-1}]}$")
    ax.set_xlabel("${\\rm Lookback\\ time\\ [Gyr]}$")

    # uncomment line below for log scale on y-axis
    # ax.set_yscale("log")

    fname = fname_spec.split('.spec')[0]
    ax.set_title(fname+'          $z=$'+str(np.round(z_spec,4)), loc='right')

    figure_timestamp(fig, x=0.97, y=0.04, fontsize=8)

    plt.tight_layout()

    if save:
        plotpath = "pipes/plots/" + fit.run + "/" + fname + '_' + suffix + '_sfh'
        # plt.savefig(plotpath+'.pdf', transparent=True)
        plt.savefig(plotpath+'.png', transparent=True)
        plt.close(fig)

    if show:
        plt.show()
        plt.close(fig)

    return fig



# --------------------------------------------------------------
# ---------------------------------------- POSTERIOR CORNER PLOT
# --------------------------------------------------------------

def plot_corner(fit, fname_spec, z_spec, fit_instructions, filt_list, suffix, show=False, save=False, bins=25):
    """
    Makes corner plot of the fitted parameters
    
    Parameters
    ----------
    fit : fit object from BAGPIPES (where fit = pipes.fit(galaxy, fit_instructions))
    fname_spec : filename of spectrum e.g. 'rubies-uds3-v3_prism-clear_4233_62812.spec.fits', format=str
    z_spec : spectroscopic redshift, format=float
    fit_instructions : dictionary of bagpipes input parameters
    filt_list : array of paths to filter files, each element is a string, format=numpy array
    show : specifies wherether or not to display the image, format=bool
    save : specifies wherether or not to save the image, format=bool

    Returns
    -------
    None
    """

    names = fit.fitted_model.params
    samples = np.copy(fit.posterior.samples2d)

    # Set up axis labels
    # labels = fit.fitted_model.params.copy()
    labels = pipes.plotting.general.fix_param_names(names)

    # Log any parameters with log_10 priors to make them easier to see
    for i in range(fit.fitted_model.ndim):
        if fit.fitted_model.pdfs[i] == "log_10":
            samples[:, i] = np.log10(samples[:, i])

            labels[i] = "log_10(" + labels[i] + ")"

    # Replace any r parameters for Dirichlet distributions with t_x vals
    j = 0
    for i in range(fit.fitted_model.ndim):
        if "dirichlet" in fit.fitted_model.params[i]:
            comp = fit.fitted_model.params[i].split(":")[0]
            n_x = fit.fitted_model.model_components[comp]["bins"]
            t_percentile = int(np.round(100*(j+1)/n_x))

            samples[:, i] = fit.posterior.samples[comp + ":tx"][:, j]
            j += 1

            labels[i] = "t" + str(t_percentile) + " / Gyr"

    # Make the corner plot
    fig = corner.corner(samples, bins=bins, labels=labels, color="k",
                        quantiles=[0.16, 0.5, 0.84],
                        show_titles=True, smooth=1., title_kwargs={"fontsize": 13},
                        hist_kwargs={"density": True, "histtype": "stepfilled",
                                     "color": "firebrick", "edgecolor": "firebrick", "lw": 2, "alpha": 0.3})
                        #smooth1d=1.)
    
    # overplot priors
    fit_instructions_temp = copy.deepcopy(fit_instructions)
    del fit_instructions_temp['R_curve']
    priors = pipes.fitting.check_priors(fit_instructions=fit_instructions_temp, filt_list=filt_list, n_draws=5000)
    priors.get_advanced_quantities()

    # Access the axes of the figure for additional customization
    axes = fig.get_axes()
    
    # loop of each histogram
    for i in range(len(names)):
        ax = axes[i * (len(names) + 1)]  # spacing of diagonal axes
        ax.hist(priors.samples[names[i]],
                bins=bins, density=True,
                histtype='stepfilled', ls='-', lw=2, edgecolor="steelblue", zorder=-1, alpha=0.3)
        

    # fname = fname_spec.split('.spec')[0]
    # plt.title(fname+'          $z=$'+str(np.round(z_spec,4)), loc='right')

    # figure_timestamp(fig, fontsize=8, ha='right', va='top')

    plt.tight_layout()

    if save:
        fname = fname_spec.split('.spec')[0]
        plotpath = "pipes/plots/" + fit.run + "/" + fname + '_' + suffix + '_corner'
        # plt.savefig(plotpath+'.pdf', transparent=True)
        plt.savefig(plotpath+'.png', transparent=True)
        plt.close(fig)

    if show:
        plt.show()
        plt.close(fig)

    return fig



# --------------------------------------------------------------
# --------------------------------------------- CALIBRATION PLOT
# --------------------------------------------------------------

def plot_calib(fit, fname_spec, z_spec, suffix, show=False, save=False, plot_xlims=None):
    """
    Makes plot of the calibration curve
    
    Parameters
    ----------
    fit : fit object from BAGPIPES (where fit = pipes.fit(galaxy, fit_instructions))
    fname_spec : filename of spectrum e.g. 'rubies-uds3-v3_prism-clear_4233_62812.spec.fits', format=str
    z_spec : spectroscopic redshift, format=float
    show : specifies wherether or not to display the image, format=bool
    save : specifies wherether or not to save the image, format=bool

    Returns
    -------
    None
    """

    fig = plt.figure(figsize=(10,4.5))
    ax = plt.subplot()

    ID = int(fit.galaxy.ID)
    _, _, _ = fitting.guess_calib(ID, z_spec, plot=True)

    fit.posterior.get_advanced_quantities()

    wavs = fit.galaxy.spectrum[:, 0]
    samples = fit.posterior.samples["calib"]
    post = np.percentile(samples, (16, 50, 84), axis=0).T

    ax.plot(wavs/10000, post[:, 0], color="firebrick", zorder=10, lw=0.1)
    ax.plot(wavs/10000, post[:, 1], color="firebrick", zorder=10, label='Posterior calib curve')
    ax.plot(wavs/10000, post[:, 2], color="firebrick", zorder=10, lw=0.1)
    ax.fill_between(wavs/10000, post[:, 0], post[:, 2], lw=0,
                    color="firebrick", alpha=0.3, zorder=9)

    ymin, ymax = ax.get_ylim()
    if ymax<2:
        ymax=2
    yticks = np.arange(0, ymax+0.5, 0.5)

    if plot_xlims==None:
        ax.set_xlim(wavs[0]/10000, wavs[-1]/10000)
    elif plot_xlims!=None:
        ax.set_xlim(plot_xlims)
    ax.set_ylim(0, ymax)

    ax.set_xticks(prism_wav_xticks())
    # ax.set_yticks(yticks)

    ax.set_xlabel("$\lambda_{\\rm obs}{\\rm \\ [\mu m]}$")
    ax.set_ylabel("$\\mathrm{Spectrum\\ multiplied\\ by}$")

    plt.legend(loc='upper left')

    fname = fname_spec.split('.spec')[0]
    ax.set_title(fname+'          $z=$'+str(np.round(z_spec,4)), loc='right')

    figure_timestamp(fig, x=0.97, y=0.04, fontsize=8)

    plt.tight_layout()

    if save:
        fname = fname_spec.split('.spec')[0]
        plotpath = "pipes/plots/" + fit.run + "/" + fname + '_' + suffix + '_calib'
        # plt.savefig(plotpath+'.pdf', transparent=True)
        plt.savefig(plotpath+'.png', transparent=True)
        plt.close(fig)

    if show:
        plt.show()
        plt.close(fig)

    return fig




# --------------------------------------------------------------
# -------------------------------- TABLE OF POSTERIOR PROPERTIES
# --------------------------------------------------------------
def save_posterior_sample_dists(fit, fname_spec, suffix, save=False):
    """
    Makes table of 16th, 50th, and 84th percentile values of all posterior quantities from BAGPIPES fit, saves table to csv file
    
    Parameters
    ----------
    fit : fit object from BAGPIPES (where fit = pipes.fit(galaxy, fit_instructions))
    fname_spec : filename of spectrum e.g. 'rubies-uds3-v3_prism-clear_4233_62812.spec.fits', format=str
    suffix : string containing sfh and dust information to be appended to output file name, format=str
    save : specifies wherether or not to save the table, format=bool

    Returns
    -------
    None
    """

    fit.posterior.get_advanced_quantities()

    samples = fit.posterior.samples
    keys = list(samples.keys())
    to_remove = ['photometry', 'spectrum', 'spectrum_full', 'dust_curve', 'calib', 'uvj']
    keys_mod = [key for key in keys if key not in to_remove]

    tab = []

    for i in range(len(keys_mod)):
    
        tab.append(list([keys_mod[i],
                         np.percentile(samples[keys_mod[i]],(16)),
                         np.percentile(samples[keys_mod[i]],(50)),
                         np.percentile(samples[keys_mod[i]],(84))]))
        
    # posterior table colnames
    tab_colnames = []
    post_ext = ['_16', '_50', '_84']
    [[tab_colnames.append(tab_col_i.replace(':','_') + post_ext_i) for post_ext_i in post_ext] for tab_col_i in np.array(tab)[:,0]]

    # row values of table
    tab_row_vals = []
    [[tab_row_vals.append(float(val_j)) for val_j in tab_row_i[1:4]] for tab_row_i in np.array(tab)]

    # making an astropy Table
    post_tab_temp = Table(np.array(tab_row_vals), names=tab_colnames)

    ### add UVJ to post  tab ###
    post_uvj = np.percentile(fit.posterior.samples["uvj"], (16, 50, 84), axis=0).T
    to_add = 'restU', 'restV', 'restJ'
    to_add_full = [to_add_j + post_ext_i for to_add_j in to_add for post_ext_i in post_ext]
    post_tab_add = Table(post_uvj.flatten(), names=to_add_full)

    # hstack tables
    post_tab = hstack([post_tab_temp,post_tab_add])

    ### saving posterior to csv table ###
    fname = fname_spec.split('.spec')[0]
    fname_phot = fname + '.phot.cat'

    phot_colnames = ['file_spec', 'file_phot', 'file_zout', 'id_phot', 'dr']
    phot_cols = Table.read(f'files/{fname_phot}', format='ascii.commented_header')[phot_colnames]

    # number of photometric filters
    id = int(fit.galaxy.ID)
    filt_list = fitting.updated_filt_list(id) # list of valid filters
    filt_num = len(filt_list)

    phot_cols.add_columns([filt_num], indexes=[-1], names=['filt_num'])

    tab_stacked = hstack([phot_cols, post_tab])

    if not os.path.exists("./pipes/cats/" + fit.run):
        os.mkdir("./pipes/cats/" + fit.run)

    tabpath = "pipes/cats/" + fit.run + "/" + fname + '_' + suffix + '_postcat.csv'
    tab_stacked.write(tabpath, format='csv', overwrite=True)

    return None



# --------------------------------------------------------------
# -------------------------------------- CALIBRATION CURVE TABLE
# --------------------------------------------------------------
def save_calib(fit, fname_spec, suffix, save=False):
    """
    Makes table of 16th, 50th, and 84th percentile values of calibratoion curve from BAGPIPES fit, saves table to csv file
    
    Parameters
    ----------
    fit : fit object from BAGPIPES (where fit = pipes.fit(galaxy, fit_instructions))
    fname_spec : filename of spectrum e.g. 'rubies-uds3-v3_prism-clear_4233_62812.spec.fits', format=str
    save : specifies wherether or not to save the table, format=bool

    Returns
    -------
    None
    """

    fit.posterior.get_advanced_quantities()

    wavs = fit.galaxy.spectrum[:, 0]
    samples = fit.posterior.samples["calib"]
    post = np.percentile(samples, (16, 50, 84), axis=0).T

    tab = Table([wavs, post[:,0], post[:,1], post[:,2]], names=['wavs', 'calib_16', 'calib_50', 'calib_84'], units=['Angstrom', '', '', ''])

    ### saving posterior to csv table ###
    fname = fname_spec.split('.spec')[0]

    if not os.path.exists("./pipes/cats/" + fit.run):
        os.mkdir("./pipes/cats/" + fit.run)

    tabpath = "pipes/cats/" + fit.run + "/" + fname + '_' + suffix + '_calibcurve.csv'
    tab.write(tabpath, format='csv', overwrite=True)

    return None



# --------------------------------------------------------------
# ----------------------------------- EMISSION LINES FROM MSAEXP
# --------------------------------------------------------------
def add_lines_msa(z_spec):
    """
    Plots emission lines in colours consistent with msaexp plotting format
    
    Parameters
    ----------
    z_spec : spectroscopic redshift, format=float

    Returns
    -------
    None
    """

    if z_spec is not None:
        cc = MPL_COLORS
        for w, c in zip(
            [
                1216.0,
                1909.0,
                2799.0,
                3727,
                4101,
                4340,
                4860,
                5007,
                6565,
                9070,
                9530,
                1.094e4,
                1.282e4,
                1.875e4,
                1.083e4,
            ],
            [
                "purple",
                "olive",
                "skyblue",
                cc["purple"],
                cc["g"],
                cc["g"],
                cc["g"],
                cc["b"],
                cc["g"],
                "darkred",
                "darkred",
                cc["pink"],
                cc["pink"],
                cc["pink"],
                cc["orange"],
            ],
        ):
            wz = w * (1 + z_spec) / 1.0e4
            dw = 0.02

            plt.fill_between(
                [wz - dw, wz + dw],
                [0, 0],
                [100, 100],
                color=c,
                alpha=0.07,
                zorder=-100,
            )
            



