# !!! ADAPTED FROM DEFAULT BAGPIPES PLOTTING FUNCTIONS !!!

# ------- LIBRARIES ------ #
import numpy as np
from astropy.cosmology import Planck13 as cosmo
from astropy.table import Table, hstack
from astropy import units as u
import re
from grizli.utils import figure_timestamp
from grizli.utils import MPL_COLORS

import corner
import bagpipes as pipes
import copy
import os

from . import fitting
from . import database
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



# --------------------------------------------------------------
# ----------------------------------- DEFINING X-TICKS FOR PRISM
# --------------------------------------------------------------
def prism_wav_xticks(xmin=0.5, xmax=5.5, dx=0.5):
    return np.arange(xmin,xmax+dx,dx)


# --------------------------------------------------------------
# ---------------------------- CONVERTING WAVELENGTHS TO INDECES
# --------------------------------------------------------------
def wav_to_idx(wavs_to_conv, wavs):
    """
    Maps wavelengths to index values
    
    Parameters
    ----------
    wavs_to_conv : Wavelengths for which to get index values; format=numpy array
    wavs : Wavelengths of spectrum; format=numpy array

    Returns
    -------
    idx : Index values of wavs_to_conv argument
    """

    idx_wavs = np.arange(len(wavs))
    idx = np.interp(wavs_to_conv, wavs, idx_wavs)

    return(idx)


# --------------------------------------------------------------
# ---------------------- PLOT SPECTROSCOPIC AND PHOTOMETRIC DATA
# --------------------------------------------------------------
def plot_spec_phot_data(runid, fname_spec, fname_phot, z_spec, suffix, spec_only=False, f_lam=False, show=False, save=False, run='.', plotlims=None, ymax=None):
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

    if not spec_only:
        # photometry
        phot_tab = Table.read(f'files/{fname_phot}', format='ascii.commented_header')
        
        filt_list = fitting.updated_filt_list(runid) # list of filters

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
        phot_fluxes_temp *= zp_array
        phot_efluxes_temp = np.lib.recfunctions.structured_to_unstructured(np.array(phot_tab[list(eflux_colNames)]))
        phot_efluxes_temp *= zp_array

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
        
        if not spec_only:
            phot_fluxes = fitting.convert_mujy2cgs(phot_fluxes, phot_wavs*10000)
            phot_efluxes = fitting.convert_mujy2cgs(phot_efluxes, phot_wavs*10000)

    fig,ax = plt.subplots(figsize=(10,4.5))

    # from https://github.com/gbrammer/msaexp/blob/main/msaexp/utils.py#L1846
    ymax_percentile = 90
    ymax_scale = 1.5
    ymax_sigma_scale = 7
    if ymax is None:
        _msk = (spec_efluxes > 0) & np.isfinite(spec_efluxes) & np.isfinite(spec_efluxes)
        if _msk.sum() == 0:
            ymax = 1.0
        else:
            ymax = np.nanpercentile(spec_fluxes[_msk], ymax_percentile) * ymax_scale
            ymax = np.maximum(ymax, ymax_sigma_scale * np.nanmedian(spec_efluxes[_msk]))
    # ymax*=np.nanmax(calib_50)

    # x ticks 
    xpos = wav_to_idx(wavs_to_conv=spec_wavs, wavs=spec_wavs)
    if not spec_only:
        phot_xpos = wav_to_idx(wavs_to_conv=phot_wavs, wavs=spec_wavs)

    ax.hlines(y=0, xmin=xpos.min(), xmax=xpos.max(), lw=1.0, color='gainsboro', zorder=-1)

    if plotlims==None:
        xmin_plot, xmax_plot = np.min(spec_wavs), np.max(spec_wavs)
        ymin_plot, ymax_plot = -0.1*np.max(spec_fluxes), ymax#1.1*np.max(spec_fluxes)
    elif plotlims!=None:
        xmin_plot, xmax_plot, ymin_plot, ymax_plot = plotlims

    ax.hlines(y=0, xmin=spec_wavs.min(), xmax=spec_wavs.max(), lw=1.0, color='gainsboro', zorder=-1)

    ##################################
    # ------------ DATA ------------ #
    ##################################
    
    # ---------- SPECTRUM ---------- #
    ax.step(xpos, spec_fluxes,
            zorder=-1, color='slategrey', alpha=0.7, lw=1)
    ax.fill_between(xpos, spec_fluxes-spec_efluxes, spec_fluxes+spec_efluxes,
                    zorder=-1, color='slategrey', alpha=0.1, step="mid")

    if not spec_only:
        # --------- PHOTOMETRY --------- #
        ax.errorbar(phot_xpos, phot_fluxes, yerr=phot_efluxes,
                    fmt='o', ms=8, color='gainsboro', markeredgecolor='k', ecolor='grey', elinewidth=1, markeredgewidth=1.,
                    zorder=1, alpha=1.)
            
    add_lines_msa(z_spec=z_spec, wavs_for_scaling=spec_wavs)
    
    ##################################
    # --------- FORMATTING --------- #
    ##################################
    
    ax.set_xlabel('$\lambda_{\\rm obs}{\\rm \\ [\mu m]}$')
    ax.set_ylabel('${f_{\\nu}} {\\rm\\ [\\mu Jy]}$')
    if f_lam:
        ax.set_ylabel('$f_{\\lambda} {\\rm\\ [erg\ s^{-1} cm^{-2} \AA^{-1}]}$')

    ax.set_xlim(xmin_plot, xmax_plot)
    ax.set_ylim(ymin_plot, ymax)

    ax.set_xticks(prism_wav_xticks())

    xlmin_1, xlmax_1 =  np.ceil(spec_wavs.min() * 10) / 10, np.floor(spec_wavs.max() * 10) / 10
    xlmin_5, xlmax_5 =  np.ceil(spec_wavs.min() * 2) / 2, np.floor(spec_wavs.max() * 2) / 2

    xtmin = wav_to_idx(wavs_to_conv=prism_wav_xticks(xmin=xlmin_1, xmax=xlmax_1, dx=0.1), wavs=spec_wavs)
    xtmaj = wav_to_idx(wavs_to_conv=prism_wav_xticks(xmin=xlmin_5, xmax=xlmax_5, dx=0.5), wavs=spec_wavs)

    ax.set_xticks(xtmin, minor=True)
    ax.set_xticks(xtmaj, minor=False)
    ax.set_xticklabels(prism_wav_xticks(xmin=xlmin_5, xmax=xlmax_5, dx=0.5))

    ax.grid(lw=0.5, ls="dotted", color="grey")
    
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
        plt.savefig(f'{plotPath}/{imname}.pdf', transparent=True)
        plt.savefig(f'{plotPath}/{imname}.png', transparent=True)
        plt.close(fig)

    if show:
        plt.show()
        plt.close(fig)

    return fig


# --------------------------------------------------------------
# ----------------------------------- PLOT FITTED SPECTRAL MODEL
# --------------------------------------------------------------
def plot_fitted_spectrum(fit, fname_spec, z_spec, suffix, spec_only=False, f_lam=False, show=False, save=False, return_plotlims=False, ymax=None):
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
    
    wavs = fit.galaxy.spectrum[:, 0]/10000
    
    spec_post = np.copy(fit.posterior.samples["spectrum"])
    
    if "noise" in list(fit.posterior.samples):
        spec_post += fit.posterior.samples["noise"]
    
    post = np.percentile(spec_post, (16, 50, 84), axis=0).T

    if spec_only:
        spec_fluxes = fit.galaxy.spectrum[:,1]
        spec_efluxes = fit.galaxy.spectrum[:,2]

    elif not spec_only:
        calib_50 = np.percentile(fit.posterior.samples["calib"], 50, axis=0).T

        spec_fluxes = fit.galaxy.spectrum[:,1]*calib_50
        spec_efluxes = fit.galaxy.spectrum[:,2]*calib_50

        # phot_post = np.percentile(fit.posterior.samples["photometry"], (16, 50, 84), axis=0).T
        if fit.galaxy.msa_phot is not None:
            phot_post = np.percentile(fit.galaxy.msa_phot, (16, 50, 84), axis=0).T
        else:
            phot_post = np.percentile(fit.posterior.samples["photometry"], (16, 50, 84), axis=0).T

        phot_flux_mask = fit.galaxy.photometry[:,2]<1e90

        phot_wavs = (fit.galaxy.filter_set.eff_wavs/10000)[phot_flux_mask]
        phot_fluxes = (fit.galaxy.photometry[:,1])[phot_flux_mask]
        phot_efluxes = (fit.galaxy.photometry[:,2])[phot_flux_mask]

        phot_fluxes_model = (phot_post[:,1])[phot_flux_mask]
        phot_fluxes_model_lo = (phot_post[:,0])[phot_flux_mask]
        phot_fluxes_model_hi = (phot_post[:,2])[phot_flux_mask]

        phot_residual = (phot_fluxes - phot_fluxes_model) / phot_efluxes

    if "noise:scaling" in list(fit.posterior.samples):
        spec_efluxes *= np.percentile(fit.posterior.samples["noise:scaling"], 50)

    if fit.galaxy.msa_line_components is not None:
        msa_lsqfit_fluxes = np.percentile(fit.galaxy.msa_model, 50, axis=0).T

    spec_fluxes_model_no_msalsq = post[:,1]
    spec_fluxes_model_no_msalsq_lo = post[:,0]
    spec_fluxes_model_no_msalsq_hi = post[:,2]

    spec_residual_no_msalsq = (spec_fluxes - spec_fluxes_model_no_msalsq) / spec_efluxes

    if fit.galaxy.msa_line_components is not None:
        if spec_only:
            spec_fluxes_model = post[:,1] + msa_lsqfit_fluxes
            spec_fluxes_model_lo = post[:,0] + msa_lsqfit_fluxes
            spec_fluxes_model_hi = post[:,2] + msa_lsqfit_fluxes
        elif not spec_only:
            spec_fluxes_model = post[:,1] + msa_lsqfit_fluxes*calib_50
            spec_fluxes_model_lo = post[:,0] + msa_lsqfit_fluxes*calib_50
            spec_fluxes_model_hi = post[:,2] + msa_lsqfit_fluxes*calib_50
    else:
        spec_fluxes_model = post[:,1]
        spec_fluxes_model_lo = post[:,0]
        spec_fluxes_model_hi = post[:,2]

    spec_residual = (spec_fluxes - spec_fluxes_model) / spec_efluxes

    if not f_lam:
        spec_fluxes = fitting.convert_cgs2mujy(spec_fluxes, wavs*10000)
        spec_efluxes = fitting.convert_cgs2mujy(spec_efluxes, wavs*10000)

        spec_fluxes_model_no_msalsq = fitting.convert_cgs2mujy(spec_fluxes_model_no_msalsq, wavs*10000)
        spec_fluxes_model_no_msalsq_lo = fitting.convert_cgs2mujy(spec_fluxes_model_no_msalsq_lo, wavs*10000)
        spec_fluxes_model_no_msalsq_hi = fitting.convert_cgs2mujy(spec_fluxes_model_no_msalsq_hi, wavs*10000)

        spec_residual_no_msalsq = (spec_fluxes - spec_fluxes_model_no_msalsq) / spec_efluxes

        spec_fluxes_model = fitting.convert_cgs2mujy(spec_fluxes_model, wavs*10000)
        spec_fluxes_model_lo = fitting.convert_cgs2mujy(spec_fluxes_model_lo, wavs*10000)
        spec_fluxes_model_hi = fitting.convert_cgs2mujy(spec_fluxes_model_hi, wavs*10000)

        spec_residual = (spec_fluxes - spec_fluxes_model) / spec_efluxes

        if not spec_only:
            phot_fluxes = fitting.convert_cgs2mujy(phot_fluxes, phot_wavs*10000)
            phot_efluxes = fitting.convert_cgs2mujy(phot_efluxes, phot_wavs*10000)

            phot_fluxes_model = fitting.convert_cgs2mujy(phot_fluxes_model, phot_wavs*10000)
            phot_fluxes_model_lo = fitting.convert_cgs2mujy(phot_fluxes_model_lo, phot_wavs*10000)
            phot_fluxes_model_hi = fitting.convert_cgs2mujy(phot_fluxes_model_hi, phot_wavs*10000)

            phot_residual = (phot_fluxes - phot_fluxes_model) / phot_efluxes
    
    # plotting spectrum
    fig = plt.figure(1, figsize=(10,5.0))
    ax = fig.add_axes((.1,.3,.85,.6))

    # from https://github.com/gbrammer/msaexp/blob/main/msaexp/utils.py#L1846
    ymax_percentile = 90
    ymax_scale = 1.5
    ymax_sigma_scale = 7
    if ymax is None:
        _msk = (spec_efluxes > 0) & np.isfinite(spec_efluxes) & np.isfinite(spec_efluxes)
        if _msk.sum() == 0:
            ymax = 1.0
        else:
            ymax = np.nanpercentile(spec_fluxes[_msk], ymax_percentile) * ymax_scale
            ymax = np.maximum(ymax, ymax_sigma_scale * np.nanmedian(spec_efluxes[_msk]))
    # ymax*=np.nanmax(calib_50)

    # x ticks 
    xpos = wav_to_idx(wavs_to_conv=wavs, wavs=wavs)
    if not spec_only:
        phot_xpos = wav_to_idx(wavs_to_conv=phot_wavs, wavs=wavs)

    ax.hlines(y=0, xmin=xpos.min(), xmax=xpos.max(), lw=1.0, color='gainsboro', zorder=-1)
    
    ##################################
    # ------------ DATA ------------ #
    ##################################
    
    # ---------- SPECTRUM ---------- #
    ax.step(xpos, spec_fluxes,
            zorder=-1, color='slategrey', alpha=0.7, lw=1, label='Spectrum (scaled)', where="mid")
    ax.fill_between(xpos,
                    spec_fluxes-spec_efluxes,
                    spec_fluxes+spec_efluxes,
                    zorder=-1, color='slategrey', alpha=0.1, step="mid")
    
    if not spec_only:
        # ---------- PHOTOMETRY ---------- #
        ax.errorbar(phot_xpos, phot_fluxes, yerr=phot_efluxes,
                    fmt='o', ms=8, color='gainsboro', markeredgecolor='k', ecolor='grey', elinewidth=0.5, markeredgewidth=1.,
                    zorder=1, alpha=1., label='Photometry')
    
    ##################################
    # -------- FITTED MODEL -------- #
    ##################################

    # ------- PIPES SPECTRUM ------- #
    ax.step(xpos, spec_fluxes_model_no_msalsq,
            zorder=-1, color='steelblue', alpha=0.7, lw=1.5, where="mid")
    ax.fill_between(xpos,
                    spec_fluxes_model_no_msalsq_lo, spec_fluxes_model_no_msalsq_hi,
                    zorder=-1, color='steelblue', alpha=0.1, step="mid")
    
    # ---------- SPECTRUM ---------- #
    ax.step(xpos, spec_fluxes_model,
            zorder=-1, color='firebrick', alpha=0.7, lw=1.5, label='Model spectrum', where="mid")
    ax.fill_between(xpos,
                    spec_fluxes_model_lo, spec_fluxes_model_hi,
                    zorder=-1, color='firebrick', alpha=0.1, step="mid")
    
    if not spec_only:
        # ---------- PHOTOMETRY ---------- #
        ax.errorbar(phot_xpos, phot_fluxes_model, yerr=[phot_fluxes_model-phot_fluxes_model_lo, phot_fluxes_model_hi-phot_fluxes_model],
                    fmt='o', ms=7, color='firebrick', markeredgecolor='k', ecolor='grey', elinewidth=0.5, markeredgewidth=.5,
                    zorder=1, alpha=0.7, label='Model photometry')

    ##################################
    # ------- EMISSION LINES ------- #
    ##################################

    add_lines_msa(z_spec=z_spec, wavs_for_scaling=wavs)

    ##################################
    # --------- RESIDUALS ---------- #
    ##################################

    ax_res = fig.add_axes((.1,.1,.85,.2))

    ax_res.step(xpos, spec_residual_no_msalsq,
                zorder=-1, color='steelblue', alpha=0.5, lw=1., where="mid")

    ax_res.step(xpos, spec_residual,
                zorder=-1, color='slategrey', alpha=0.7, lw=1., where="mid")
    
    if not spec_only:
        ax_res.errorbar(phot_xpos, phot_residual,
                        fmt='o', ms=8, color='gainsboro', markeredgecolor='k', ecolor='grey', elinewidth=0.5, markeredgewidth=1.,
                        zorder=1, alpha=1.)
    
    ax_res.hlines(y=0, xmin=xpos.min(), xmax=xpos.max(), lw=1.0, color='grey', zorder=-1)
    
    ##################################
    # --------- FORMATTING --------- #
    ##################################
    
    # ax.set_xlabel('$\lambda_{\\rm obs}{\\rm \\ [\mu m]}$')
    ax.set_ylabel('${f_{\\lambda}}{\\rm\\ [erg\ s^{-1} cm^{-2} \AA^{-1}]}$')
    if not f_lam:
        ax.set_ylabel('${f_{\\nu}} {\\rm\\ [\\mu Jy]}$')

    ax_res.set_xlabel('$\lambda_{\\rm obs}{\\rm \\ [\mu m]}$')
    ax_res.set_ylabel('$\chi$')

    xmin_plot, xmax_plot = np.min(xpos), np.max(xpos)
    ymin_plot, ymax_plot = -0.1*np.max(spec_fluxes), 1.1*np.max(spec_fluxes)

    ax.set_xlim(xmin_plot, xmax_plot)
    ax.set_ylim(-0.1*ymax, ymax)

    ax.set_xticks([])

    ax_res.set_xlim(xmin_plot, xmax_plot)
    if np.abs(spec_residual).max()<=10:
        ax_res_ylim = np.abs(spec_residual).max()
    else:
        ax_res_ylim = 10
    ax_res.set_ylim(-1.1*ax_res_ylim, 1.1*ax_res_ylim)

    xlmin_1, xlmax_1 =  np.ceil(wavs.min() * 10) / 10, np.floor(wavs.max() * 10) / 10
    xlmin_5, xlmax_5 =  np.ceil(wavs.min() * 2) / 2, np.floor(wavs.max() * 2) / 2

    xtmin = wav_to_idx(wavs_to_conv=prism_wav_xticks(xmin=xlmin_1, xmax=xlmax_1, dx=0.1), wavs=wavs)
    xtmaj = wav_to_idx(wavs_to_conv=prism_wav_xticks(xmin=xlmin_5, xmax=xlmax_5, dx=0.5), wavs=wavs)

    ax.set_xticks(xtmin, minor=True)
    ax.set_xticks(xtmaj, minor=False)
    ax.set_xticklabels([])
    
    ax_res.set_xticks(xtmin, minor=True)
    ax_res.set_xticks(xtmaj, minor=False)
    ax_res.set_xticklabels(prism_wav_xticks(xmin=xlmin_5, xmax=xlmax_5, dx=0.5))

    ax.grid(lw=0.5, ls="dotted", color="grey")
    ax_res.grid(lw=0.5, ls="dotted", color="grey")
    
    ax.legend(loc='upper left', framealpha=0.5)

    fname = fname_spec.split('.spec')[0]
    ax.set_title(fname+'          $z=$'+str(np.round(z_spec,4)), loc='right')

    figure_timestamp(fig, x=0.95, y=0.002, fontsize=8)
    
    # plt.tight_layout()
    
    if save:
        str_ext = '_flam'
        if not f_lam:
            str_ext = '_fnu'
        plotpath = "pipes/plots/" + fit.run + "/" + fname +  '_' + suffix + '_specfit' + str_ext
        plt.savefig(plotpath+'.pdf', transparent=True)
        plt.savefig(plotpath+'.png', transparent=True)
        plt.close(fig)

    if show:
        plt.show()
        plt.close(fig)

    if return_plotlims:
        return fig, [xmin_plot, xmax_plot, -0.1*ymax, ymax]
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
        plt.savefig(plotpath+'.pdf', transparent=True)
        plt.savefig(plotpath+'.png', transparent=True)
        plt.close(fig)

    if show:
        plt.show()
        plt.close(fig)

    return fig



# --------------------------------------------------------------
# ---------------------------------------- POSTERIOR CORNER PLOT
# --------------------------------------------------------------

def plot_corner(fit, fname_spec, z_spec, fit_instructions, filt_list, suffix, corner_full=True, corner_sci=True, spec_only=False, show=False, save=False, bins=25):
    """
    Makes corner plot of the fitted parameters
    
    Parameters
    ----------
    fit : fit object from BAGPIPES (where fit = pipes.fit(galaxy, fit_instructions))
    fname_spec : filename of spectrum e.g. 'rubies-uds3-v3_prism-clear_4233_62812.spec.fits', format=str
    z_spec : spectroscopic redshift, format=float
    fit_instructions : dictionary of bagpipes input parameters
    filt_list : array of paths to filter files, each element is a string, format=numpy array
    corner_full : if True, plot all fitted parameters, if False, only plot science parameters, format=bool
    corner_sci : if True, plot only select science parameters (e.g. not dsfr params from SFH), format=bool
    show : specifies wherether or not to display the image, format=bool
    save : specifies wherether or not to save the image, format=bool

    Returns
    -------
    None
    """

    if not spec_only:
        # overplot priors
        fit_instructions_temp = copy.deepcopy(fit_instructions)
        del fit_instructions_temp['R_curve']

        priors = pipes.fitting.check_priors(fit_instructions=fit_instructions_temp, filt_list=filt_list, n_draws=5000)
        priors.get_advanced_quantities()

    if corner_full: # plot full corner, with all fitted parameters

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
        fig1 = corner.corner(samples, bins=bins, labels=labels, color="k",
                            quantiles=[0.16, 0.5, 0.84],
                            show_titles=True, smooth=1., title_kwargs={"fontsize": 13},
                            hist_kwargs={"density": True, "histtype": "stepfilled",
                                        "color": "firebrick", "edgecolor": "firebrick", "lw": 2, "alpha": 0.3})
                            #smooth1d=1.)
        
        # overplot priors
        if not spec_only:
            # Access the axes of the figure for additional customization
            axes1 = fig1.get_axes()
            
            # loop of each histogram
            for i in range(len(names)):
                ax = axes1[i * (len(names) + 1)]  # spacing of diagonal axes
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
            plt.savefig(plotpath+'.pdf', transparent=True)
            plt.savefig(plotpath+'.png', transparent=True)
            plt.close(fig1)

        if show:
            plt.show()
            plt.close(fig1)

    
    if corner_sci: # plot corner with only select science parameters

        names = ['calib:1', 'stellar_mass', 'continuity:metallicity', 'sfr', 't50', 'ssfr', 'mass_weighted_age', 'dust:Av', 'nebular:logU']
        if spec_only:
            names.remove('calib:1')
        labels = pipes.plotting.general.fix_param_names(names)

        corner_samples = np.zeros((fit.posterior.n_samples,len(names)))
        pipes_samplenames = fit.posterior.samples.keys()

        for i, name in enumerate(names):
            if name in pipes_samplenames:
                corner_samples[:, i] = fit.posterior.samples[name]
            else:
                # If the parameter is not in the samples, you might want to handle it differently
                t50_arr = calc_sf_timescales(fit, fit.fit_instructions["redshift_prior_mu"], [50], return_full=True)
                corner_samples[:, i] = t50_arr[:,0]

        # Make the corner plot
        fig2 = corner.corner(corner_samples, bins=25, labels=labels, color="k",
                            quantiles=[0.16, 0.5, 0.84],
                            show_titles=True, smooth=1., title_kwargs={"fontsize": 13},
                            hist_kwargs={"density": True, "histtype": "stepfilled",
                                        "color": "firebrick", "edgecolor": "firebrick", "lw": 2, "alpha": 0.3})

        # overplot priors
        if not spec_only:
            # Access the axes of the figure for additional customization
            axes2 = fig2.get_axes()

            # loop of each histogram
            for i in range(len(names)):
                if names[i] in pipes_samplenames:
                    ax = axes2[i * (len(names) + 1)]  # spacing of diagonal axes
                    ax.hist(priors.samples[names[i]],
                            bins=25, density=True,
                            histtype='stepfilled', ls='-', lw=2, edgecolor="steelblue", zorder=-1, alpha=0.3)


        plt.tight_layout()

        if save:
            fname = fname_spec.split('.spec')[0]
            plotpath = "pipes/plots/" + fit.run + "/" + fname + '_' + suffix + '_sci_corner'
            plt.savefig(plotpath+'.pdf', transparent=True)
            plt.savefig(plotpath+'.png', transparent=True)
            plt.close(fig2)

        if show:
            plt.show()
            plt.close(fig2)

        if corner_full and corner_sci:
            return fig1, fig2
        elif corner_full and not corner_sci:
            return fig1
        elif not corner_full and corner_sci:
            return fig2



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

    fit.posterior.get_advanced_quantities()

    wavs = fit.galaxy.spectrum[:, 0] / 10000
    samples = fit.posterior.samples["calib"]
    post = np.percentile(samples, (16, 50, 84), axis=0).T

    # x ticks 
    xpos = wav_to_idx(wavs_to_conv=wavs, wavs=wavs)

    phot_flux_mask = fit.galaxy.photometry[:,2]<1e90
    phot_wavs = (fit.galaxy.filter_set.eff_wavs/10000)[phot_flux_mask]
    phot_xpos = wav_to_idx(wavs_to_conv=phot_wavs, wavs=wavs)

    ax.plot(xpos, post[:, 0], color="firebrick", zorder=10, lw=0.1)
    ax.plot(xpos, post[:, 1], color="firebrick", zorder=10, label='Posterior calib curve')
    ax.plot(xpos, post[:, 2], color="firebrick", zorder=10, lw=0.1)
    ax.fill_between(xpos, post[:, 0], post[:, 2], lw=0,
                    color="firebrick", alpha=0.3, zorder=9)
    
    _, _, _ = fitting.guess_calib(fit.galaxy.ID, z_spec, plot=True, phot_xpos=phot_xpos, spec_xpos=xpos)

    ymin, ymax = ax.get_ylim()
    if ymax<2:
        ymax=2
    yticks = np.arange(0, ymax+0.5, 0.5)

    if plot_xlims==None:
        ax.set_xlim(wavs[0]/10000, wavs[-1]/10000)
    elif plot_xlims!=None:
        ax.set_xlim(plot_xlims)
    ax.set_ylim(0, ymax)

    xlmin_1, xlmax_1 =  np.ceil(wavs.min() * 10) / 10, np.floor(wavs.max() * 10) / 10
    xlmin_5, xlmax_5 =  np.ceil(wavs.min() * 2) / 2, np.floor(wavs.max() * 2) / 2

    xtmin = wav_to_idx(wavs_to_conv=prism_wav_xticks(xmin=xlmin_1, xmax=xlmax_1, dx=0.1), wavs=wavs)
    xtmaj = wav_to_idx(wavs_to_conv=prism_wav_xticks(xmin=xlmin_5, xmax=xlmax_5, dx=0.5), wavs=wavs)
    
    ax.set_xticks(xtmin, minor=True)
    ax.set_xticks(xtmaj, minor=False)
    ax.set_xticklabels(prism_wav_xticks(xmin=xlmin_5, xmax=xlmax_5, dx=0.5))

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
        plt.savefig(plotpath+'.pdf', transparent=True)
        plt.savefig(plotpath+'.png', transparent=True)
        plt.close(fig)

    if show:
        plt.show()
        plt.close(fig)

    return fig




# --------------------------------------------------------------
# -------------------------------- TABLE OF POSTERIOR PROPERTIES
# --------------------------------------------------------------
def save_posterior_sample_dists(fit, fname_spec, spec_only, suffix, save=False):
    """
    Makes table of 16th, 50th, and 84th percentile values of all posterior quantities from BAGPIPES fit, saves table to csv file
    
    Parameters
    ----------
    fit : fit object from BAGPIPES (where fit = pipes.fit(galaxy, fit_instructions))
    fname_spec : filename of spectrum e.g. 'rubies-uds3-v3_prism-clear_4233_62812.spec.fits', format=str
    spec_only : fit spectrum only (ignore photometry), format=bool
    suffix : string containing sfh and dust information to be appended to output file name, format=str
    save : specifies wherether or not to save the table, format=bool

    Returns
    -------
    None
    """

    fit.posterior.get_advanced_quantities()

    samples = fit.posterior.samples
    keys = list(samples.keys())
    to_remove = ['photometry', 'spectrum', 'spectrum_full', 'dust_curve', 'calib', 'uvj', 'line_fluxes']
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

    # add SFR on 10 Myr timescale
    sfr_timescale=10
    post_sfr10 = calc_sfrX(fit, sfr_timescale=sfr_timescale)
    colname_sfr10 = [f"sfr_{sfr_timescale}m" + post_ext_i for post_ext_i in post_ext]
    post_sfr10_tab = Table(post_sfr10.flatten(), names=colname_sfr10)

    ### add UVJ to post  tab ###
    post_uvj = np.percentile(fit.posterior.samples["uvj"], (16, 50, 84), axis=0).T
    to_add = 'restU', 'restV', 'restJ'
    to_add_full = [to_add_j + post_ext_i for to_add_j in to_add for post_ext_i in post_ext]
    post_tab_add = Table(post_uvj.flatten(), names=to_add_full)

    # hstack tables
    post_tab = hstack([post_tab_temp,post_sfr10_tab,post_tab_add])

    fname = fname_spec.split('.spec')[0]
    
    if not spec_only:
        fname_phot = fname + '.phot.cat'

        phot_colnames = ['file_spec', 'file_phot', 'file_zout', 'id_phot', 'dr']
        phot_cols = Table.read(f'files/{fname_phot}', format='ascii.commented_header')[phot_colnames]

        # number of photometric filters
        filt_list = fitting.updated_filt_list(fit.galaxy.ID) # list of valid filters
        filt_num = len(filt_list)

        phot_cols.add_columns([filt_num], indexes=[-1], names=['filt_num'])

    # calculating and tabulating sf timescales
    timescales=[10,20,50,80,90]
    z_spec = database.pull_zspec_from_db(f"{fname}.spec.fits")
    timescale_arr = calc_sf_timescales(fit, z_spec, timescales=timescales)
    post_ext = ['_16', '_50', '_84']
    timescales_names = []
    [[timescales_names.append(f"t{timescales_i}{post_ext_j}") for post_ext_j in post_ext] for timescales_i in timescales]
    timescales_tab = Table(data=timescale_arr.flatten(), names=timescales_names)

    # calculating and tabulating balmer and d4000 breaks
    break_values = calc_BB_and_D4000(fname_spec, z_spec)
    break_tab = Table(data=np.array(break_values), names=["bb", "bb_err", "d4000", "d4000_err"])

    # tabulating modelled line fluxes from BAGPIPES posterior
    # linefluxes_tab = tabulate_modelled_line_fluxes(fit)

    ### saving posterior to csv table ###
    if not spec_only:
        tab_stacked = hstack([phot_cols, post_tab, timescales_tab, break_tab])
    elif spec_only:
        tab_stacked = hstack([post_tab, timescales_tab, break_tab])

    # save runtime in table
    if "runtime" in fit.results.keys():
        runtime = np.round(fit.results["runtime"], 2)
        tab_stacked.add_column(runtime, name="runtime")

    if save:
        if not os.path.exists("./pipes/cats/" + fit.run):
            os.mkdir("./pipes/cats/" + fit.run)

        tabpath = "pipes/cats/" + fit.run + "/" + fname + '_' + suffix + '_postcat.csv'
        tab_stacked.write(tabpath, format='csv', overwrite=True)

    return None

# --------------------------------------------------------------
# ------------------------------- TABLE OF POSTERIOR LINE FLUXES
# --------------------------------------------------------------
def save_posterior_line_fluxes(fit, fname_spec, suffix, save=False):
    """
    Extracts line fluxes from the fit posterior samples

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
    
    if "line_fluxes" in fit.posterior.samples.keys():
    
        line_fluxes_post = fit.posterior.samples["line_fluxes"]

        line_fluxes_array = np.array([list(line_dict.values()) for line_dict in line_fluxes_post])
        line_fluxes_perc = np.percentile(line_fluxes_array, (16, 50, 84), axis=0).T

        line_fluxes_unit = u.erg/u.second/u.centimeter/u.centimeter

        line_fluxes_tab = Table([list(fit.posterior.samples["line_fluxes"][0].keys()), line_fluxes_perc[:,0,], line_fluxes_perc[:,1,], line_fluxes_perc[:,2,]],
                                names=["line_name", "perc_16", "perc_50", "perc_84"],
                                units=["", line_fluxes_unit, line_fluxes_unit, line_fluxes_unit])
        
        if save:
            if not os.path.exists("./pipes/cats/" + fit.run):
                os.mkdir("./pipes/cats/" + fit.run)

            fname = fname_spec.split('.spec')[0]
            tabpath = "pipes/cats/" + fit.run + "/" + fname + '_' + suffix + '_postcat_pipes_linefluxes.csv'
            line_fluxes_tab.write(tabpath, format='csv', overwrite=True)
    
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
    if save:
        tab.write(tabpath, format='csv', overwrite=True)

    return None



# --------------------------------------------------------------
# ----------------------------------- EMISSION LINES FROM MSAEXP
# --------------------------------------------------------------
def save_posterior_msa_lsq_line_fluxes(fit, fname_spec, suffix, save=False):
    """
    Makes table of 16th, 50th, and 84th percentile values of residual line fluxes from lsq fit, saves table to csv file
    
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

    fname = fname_spec.split('.spec')[0]

    # names of lines that were fit to bagpipes residual
    msa_comp_names = fit.galaxy.msa_line_components

    # column names for line fluxes
    name_ext = ["_16", "_50", "_84"]
    msa_comp_names_ext = [name_i+name_ext_i for name_i in msa_comp_names for name_ext_i in name_ext]

    # extracting 16th, 50th, 84th perc line fluxes from bagpipes posterior
    line_fluxes = np.percentile(np.asarray([lsq_i[0] for lsq_i in fit.galaxy.lsq_coeffs]), (16, 50, 84), axis=0).T
    line_fluxes_flat = line_fluxes.flatten()

    line_fluxes_tab = Table(data=line_fluxes_flat, names=msa_comp_names_ext)

    # save table
    tabpath = "pipes/cats/" + fit.run + "/" + fname + '_' + suffix + '_postcat_lsqfluxes.csv'
    if save:
        line_fluxes_tab.write(tabpath, format='csv', overwrite=True)

    return None


# --------------------------------------------------------------
# ----------------------------------- EMISSION LINES FROM MSAEXP
# --------------------------------------------------------------
def add_lines_msa(z_spec, wavs_for_scaling=None):
    """
    Plots emission lines in colours consistent with msaexp plotting format
    
    Parameters
    ----------
    z_spec : spectroscopic redshift, format=float
    wavs_for_scaling : spectrum wavelengths

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
            if wavs_for_scaling is None:
                wz = w * (1 + z_spec) / 1.0e4
                dw = 0.02
            elif wavs_for_scaling is not None:
                wz = wav_to_idx(wavs_to_conv=w * (1 + z_spec) / 1.0e4, wavs=wavs_for_scaling)
                dw = 2

            plt.fill_between(
                [wz - dw, wz + dw],
                [0, 0],
                [100, 100],
                color=c,
                alpha=0.07,
                zorder=-100,
            )
            

# --------------------------------------------------------------
# ----------------------------------- POSTERIOR BEST MODELS
# --------------------------------------------------------------
def save_posterior_seds(fit, fname_spec, suffix=None, save=False):
    """
    Saves 50th percentile SED model max-L model and MAP model, projected on original wavelength array
    
    Parameters
    ----------
    fit : fit object from BAGPIPES (where fit = pipes.fit(galaxy, fit_instructions))
    fname_spec : filename of spectrum e.g. 'rubies-uds3-v3_prism-clear_4233_62812.spec.fits', format=str
    suffix : string containing sfh and dust information to be appended to output file name, format=str
    save : specifies wherether or not to save the table, format=bool

    Returns
    -------
    tab : astropy table containing wavelength, flux, and flux_50th percentile, max-L and MAP model columns
    """

    fit.posterior.get_advanced_quantities()

    # spectrum
    spec_tab = Table.read(f'files/{fname_spec}', hdu=1)    
    # tab = Table(spec_tab["wave", "flux", "err"])

    # posterior
    wavs = fit.galaxy.spectrum[:, 0]/10000
    spec_post = np.copy(fit.posterior.samples["spectrum"])
    post = np.percentile(spec_post, (16, 50, 84), axis=0).T

    post_cgs = Table([wavs, post[:,0], post[:,1], post[:,2]], names=["wave_sed", "sed_16_cgs", "sed_50_cgs", "sed_84_cgs"])
    post_mujy = Table([fitting.convert_cgs2mujy(post_cgs["sed_16_cgs"], wavs*1e4),
                       fitting.convert_cgs2mujy(post_cgs["sed_50_cgs"], wavs*1e4),
                       fitting.convert_cgs2mujy(post_cgs["sed_84_cgs"], wavs*1e4)],
                       names=["sed_16_mujy", "sed_50_mujy", "sed_84_mujy"])

    tab = hstack([post_cgs, post_mujy], join_type='exact')

    if fit.galaxy.msa_line_components is not None:
        lsqfit_model = np.percentile(fit.galaxy.msa_model, 50, axis=0).T
        sed_50_flexilines_cgs = tab["sed_50_cgs"] + lsqfit_model
        sed_50_flexilines_mujy = fitting.convert_cgs2mujy(sed_50_flexilines_cgs, wavs*1e4)
        tab = hstack([tab, Table([sed_50_flexilines_cgs, sed_50_flexilines_mujy], names=["sed_50_flexilines_cgs", "sed_50_flexilines_mujy"])], join_type='exact')

    # append calibration curves
    post_calib = Table(np.percentile(fit.posterior.samples["calib"], (16, 50, 84), axis=0).T,
                       names=["calib_16", "calib_50", "calib_84"])
    tab = hstack([tab, post_calib], join_type='exact')

    # find overlapping wavelengths
    spec_mask = np.isin(np.round(spec_tab["wave"], 5), np.round(wavs, 5))
    # initialise table with nan values
    tab_full = np.full((len(spec_tab), len(tab.colnames)), np.nan, dtype=float)
    for col in tab.colnames:
        tab_full[:, tab.colnames.index(col)][spec_mask] = tab[col]
    tab_full = Table(tab_full, names=tab.colnames)

    tab_full = hstack([spec_tab["wave", "flux", "err"], tab_full], join_type='exact')

    # append dust curves
    post_dust = np.percentile(fit.posterior.samples["dust_curve"], (16, 50, 84), axis=0).T
    zfit = np.percentile(fit.posterior.samples["redshift"], 50) # need redshift to convert from rest to observed frame
    dust_16 = np.interp(spec_tab["wave"], fit.posterior.model_galaxy.wavelengths / 1e4 * (1 + zfit), post_dust[:,0])
    dust_50 = np.interp(spec_tab["wave"], fit.posterior.model_galaxy.wavelengths / 1e4 * (1 + zfit), post_dust[:,1])
    dust_84 = np.interp(spec_tab["wave"], fit.posterior.model_galaxy.wavelengths / 1e4 * (1 + zfit), post_dust[:,2])
    post_dust_interp = Table([dust_16, dust_50, dust_84], names=["dust_curve_16", "dust_curve_50", "dust_curve_84"])
    
    tab_full = hstack([tab_full, post_dust_interp], join_type='exact') 

    ##########################
    # maximum likelihood model
    ##########################
    maxL_index = np.argmax(fit.results["lnlike"])

    fit.fitted_model._update_model_components(fit.results["samples2d"][maxL_index, :])

    maxL_model_components = fit.fitted_model.model_components
    maxL_model_components_temp = copy.deepcopy(maxL_model_components)
    if "use_msa_resamp" in maxL_model_components_temp.keys():
        del maxL_model_components_temp['use_msa_resamp']

    maxL_model_galaxy = pipes.model_galaxy(model_components=maxL_model_components_temp,
                                            filt_list=fit.galaxy.filt_list,
                                            spec_wavs=tab_full["wave"]*1e4)
    
    maxL_sed_cgs = maxL_model_galaxy.spectrum[:,1]
    maxL_sed_mujy = fitting.convert_cgs2mujy(maxL_sed_cgs, tab_full["wave"]*1e4)

    tab_full = hstack([tab_full, Table([maxL_sed_cgs,maxL_sed_mujy], names=["sed_maxL_cgs", "sed_maxL_mujy"])], join_type='exact')

    if save:
        tabpath = "pipes/cats/" + fit.run + "/"
        if suffix is None:
            outname = fname_spec.replace('.spec.fits', '_postmodels.csv')
        else:
            outname = fname_spec.replace('.spec.fits', f'_{suffix}_postmodels.csv')
        tab_full.write(tabpath + outname, format='csv', overwrite=True)
        print(tabpath + outname)

    return tab_full



# --------------------------------------------------------------
# ------------------------------------ STAR-FORMATION TIMESCALES
# --------------------------------------------------------------
def calc_sf_timescales(fit, z_spec, timescales=[10,20,50,80,90], return_full=False):
    """
    Calculates SF timescales t10, t50, t90
    tX: the look-back time at which X% of the stellar mass was already formed (Belli et al. 2019)
    
    Parameters
    ----------
    fit : fit object from BAGPIPES (where fit = pipes.fit(galaxy, fit_instructions))
    z_spec : spectroscopic redshift, format=float
    timescales : sequence of percentage timescales to calculate, i.e. [X, Y, Z] to calculate [tX, tY, tZ], format=list
    return_full : if True, returns full array of timescales for each sample, format=bool

    Returns
    -------
    t_arr : array of timescales in the order [t10, t50, t90], format=numpy array
    """

    age_of_universe = cosmo.age(z_spec).value # universe age

    # SFH ages [Gyr]
    ages = fit.posterior.sfh.ages*1e-9
    ages_mask = ages<age_of_universe
    ages_interp = np.arange(np.min(ages[ages_mask]), np.max(ages[ages_mask]),0.0005)

    # SFH posterior [M_sol/yr]
    sfh_post = fit.posterior.samples["sfh"]

    timescale_ages = np.zeros((len(sfh_post), len(timescales)))
    for i in range(len(sfh_post)):
        sfh_interp = np.interp(ages_interp, ages, sfh_post[i])

        Mstar_sum = np.cumsum(sfh_interp[::-1]*ages_interp[::-1]*1e9)[::-1] # stellar mass as func. of lookback time
        Mstar_totformed = np.sum(sfh_interp*ages_interp*1e9) # total formed mass
        Mstar_percs = 0.01*np.array(timescales)*Mstar_totformed
        indices = [np.abs(Mstar_sum - M).argmin() for M in Mstar_percs]

        timescale_ages[i] = ages_interp[indices] # Gyr

    if return_full:
        return timescale_ages

    timescale_arr = np.zeros((len(timescales),3))
    for j in range(len(timescales)):
        timescale_arr[j] = np.percentile(timescale_ages[:,j], (16, 50, 84), axis=0).T

    return timescale_arr



# --------------------------------------------------------------
# --------------------------------------------- WEIGHTED AVERAGE
# --------------------------------------------------------------
def weighted_avg(values, errors):
    """
    Calculates weighted mean of array "values" with corresponding uncertainties "errors"
    
    Parameters
    ----------
    values : format=numpy array
    errors : format=numpy array

    Returns
    -------
    mean, err : weighted mean and uncertainty, format=float
    """
    
    values = np.array(values)
    errors = np.array(errors)
    mean = (np.sum((values/errors**2))/(np.sum(1/errors**2))) # from def 4.6 Barlow
    var = (1/(np.sum(1/errors**2)))
    err = np.sqrt(var)
    
    return mean, err


# --------------------------------------------------------------
# --------------------------------------- FLUX RATIO CALCULATION
# --------------------------------------------------------------
def calc_flux_ratio(spec_wavs, spec_fluxes, spec_efluxes, l1, l2, l3, l4):
    """
    Calculates flux ratio for a spectrum between wavs of l1->l2 to l3->l4
    
    Parameters
    ----------
    spec_wavs : wavelength array in microns
    spec_fluxes : flux array in microJansky
    spec_efluxes : flux errors array in microJansky
    l1, l2, l3, l4 : wav limits in Angstrom, format=float

    Returns
    -------
    ratio, ratio_err : ratio and error on ratio, format=float
    """

    blue_idx = (spec_wavs >= l1) & (spec_wavs < l2)
    red_idx = (spec_wavs >= l3) & (spec_wavs < l4)

    if blue_idx is None or red_idx is None:
        raise ValueError("Spectrum does not cover the break.")

    blue_mean, blue_err = weighted_avg(spec_fluxes[blue_idx], spec_efluxes[blue_idx])
    red_mean, red_err = weighted_avg(spec_fluxes[red_idx], spec_efluxes[red_idx])
    
    ratio = red_mean / blue_mean
    ratio_err = np.abs(ratio) * np.sqrt((blue_err/blue_mean)**2 + (red_err/red_mean)**2)

    return(ratio, ratio_err)


# --------------------------------------------------------------
# ---------------------------------------- BALMER & D4000 BREAKS
# --------------------------------------------------------------
def calc_BB_and_D4000(fname_spec, z_spec):
    """
    Calculates Balmer break and D4000 break 
    
    Parameters
    ----------
    fname_spec : filename of spectrum e.g. 'rubies-uds3-v3_prism-clear_4233_62812.spec.fits', format=str
    z_spec : spectroscopic redshift, format=float

    Returns
    -------
    bb, bb_err, d4000, d4000_err : balmer break value and uncertainty, d4000 break value and uncertainty, format=float
    """

    # load spectrum
    spec_tab = Table.read(f'files/{fname_spec}', hdu=1)
    spec_fluxes = spec_tab['flux']
    spec_efluxes = spec_tab['err']
    spec_wavs = spec_tab['wave'] * 1e4 / (1+z_spec)

    # balmer break (defn from Wang+24 : https://iopscience.iop.org/article/10.3847/2041-8213/ad55f7/pdf,
    #               also in Weibel+25 : https://arxiv.org/pdf/2409.03829)
    bb, bb_err = calc_flux_ratio(spec_wavs, spec_fluxes, spec_efluxes, 3620, 3720, 4000, 4100)
    
    # d4000 (defn from Bruzual 1973 : https://articles.adsabs.harvard.edu/pdf/1983ApJ...273..105B)
    d4000, d4000_err = calc_flux_ratio(spec_wavs, spec_fluxes, spec_efluxes, 3750, 3950, 4050, 4250)

    return bb, bb_err, d4000, d4000_err



# --------------------------------------------------------------
# -------------------------------- TABULATE EMISSION LINE FLUXES
# --------------------------------------------------------------
def tabulate_modelled_line_fluxes(fit):
    """
    Tabulates modelled line fluxes from BAGPIPES fit, returns table
    
    Parameters
    ----------
    fit : fit object from BAGPIPES (where fit = pipes.fit(galaxy, fit_instructions))

    Returns
    -------
    linefluxes_tab : format=astropy Table
    """

    line_fluxes = fit.posterior.model_galaxy.line_fluxes
    keys = line_fluxes.keys()
    keys_colnames = [re.sub('\s+', '_', keys_i) for keys_i in list(keys)]
    line_flux_values = np.array(list(line_fluxes.values()))
    linefluxes_tab = Table(data=line_flux_values, names=keys_colnames, units=[u.erg/u.second/u.centimeter/u.centimeter] * len(keys_colnames))

    return linefluxes_tab


# --------------------------------------------------------------
# ------------------------------ TABULATE SFR ON X Myr TIEMSCALE
# --------------------------------------------------------------
def calc_sfrX(fit, sfr_timescale=100):
    """
    Calculate the SFR on a given timescale from the SFH posterior samples

    Parameters:
    -----------
    fit : fit object from BAGPIPES (where fit = pipes.fit(galaxy, fit_instructions))
    sfr_timescale : timescale in Myr over which to calculate the SFR, format=float
    
    Returns:
    --------
    sfrX_perc : tuple of 16th, 50th, and 84th percentiles of the SFR on the given timescale
    """

    sfr_timescale *= 1e6

    sfh_obj = fit.posterior.sfh
    sfh_post = fit.posterior.samples["sfh"]

    age_mask = (sfh_obj.ages < sfr_timescale)

    sfrX = np.sum(sfh_post[:,age_mask]*sfh_obj.age_widths[age_mask], axis=1) / (sfh_obj.age_widths[age_mask].sum())
    sfrX_perc = np.percentile(sfrX, (16,50,84))

    return(sfrX_perc)


# --------------------------------------------------------------
# ------------------------------------------- FULL POSTERIOR SED
# --------------------------------------------------------------
def save_full_posterior_sed(fit, fname_spec, suffix=None, save=False):

    """
    Saves full Bagpipes posterior SED model
    
    Parameters
    ----------
    fit : fit object from BAGPIPES (where fit = pipes.fit(galaxy, fit_instructions))
    fname_spec : filename of spectrum e.g. 'rubies-uds3-v3_prism-clear_4233_62812.spec.fits', format=str
    suffix : string containing sfh and dust information to be appended to output file name, format=str
    save : specifies wherether or not to save the table, format=bool

    Returns
    -------
    tab : astropy table containing wavelength and 16th, 50th and 84th percentile values of the full posterior SED, format=astropy.table.Table
    """

    fit.posterior.get_advanced_quantities()

    wave_full = fit.posterior.model_galaxy.wavelengths

    post = np.percentile(fit.posterior.samples["spectrum_full"], (16,50, 84), axis=0).T
    posttab = Table([wave_full, post[:,0], post[:,1], post[:,2]], names=["wave_full", "sed_full_16_cgs", "sed_full_50_cgs", "sed_full_84_cgs"])

    if save:
        tabpath = "pipes/cats/" + fit.run + "/"
        if suffix is None:
            outname = fname_spec.replace('.spec.fits', '_postsed_full.csv')
        else:
            outname = fname_spec.replace('.spec.fits', f'_{suffix}_postsed_full.csv')
        posttab.write(tabpath + outname, format='csv', overwrite=True)
        print(tabpath + outname)

    return posttab