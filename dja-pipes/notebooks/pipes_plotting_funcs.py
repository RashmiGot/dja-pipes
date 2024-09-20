# !!! ADAPTED FROM DEFAULT BAGPIPES PLOTTING FUNCTIONS !!!

# ------- LIBRARIES ------ #
import numpy as np
from astropy.cosmology import Planck13 as cosmo
from astropy.table import Table
from grizli.utils import get_line_wavelengths
from pipes_fitting_funcs import convert_cgs2mujy
from pipes_fitting_funcs import convert_mujy2cgs

# ------- PLOTTING & FORMATTING ------- #
import matplotlib.pyplot as plt
from matplotlib import colors

params = {'legend.fontsize':12,'axes.labelsize':16,'axes.titlesize':20,
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
# ---------------------- PLOT SPECTROSCOPIC AND PHOTOMETRIC DATA
# --------------------------------------------------------------
def plot_spec_phot_data(fname_spec_out, fname_phot_out, f_lam=False):
    """
    Plots spectrum and photometry of given source
    
    Parameters
    ----------
    fname_spec_out : filename of spectrum stored '../files/', format=str
    fname_phot_out : filename of photometry stored '../files/', format=str
    f_lam : if True, output plot is in f_lambda, if False, in f_nu, format=bool

    Returns
    -------
    None
    """
    
    # spectrum
    spec_tab = Table.read(f'../files/{fname_spec_out}', hdu=1)
    spec_fluxes = spec_tab['flux']
    spec_efluxes = spec_tab['err']
    spec_wavs = spec_tab['wave']

    # photometry
    phot_tab = Table.read(f'../files/{fname_phot_out}', format='ascii.commented_header')
    z_spec = phot_tab['z_spec'].value[0]

    # jwst filter list
    filt_list = np.loadtxt("../filters/filt_list.txt", dtype="str")

    # extract fluxes from cat
    flux_colNames = [filt_list_i.split('/')[-1].split('.')[0]+'_tot_1' for filt_list_i in filt_list]
    eflux_colNames = [filt_list_i.split('/')[-1].split('.')[0]+'_etot_1' for filt_list_i in filt_list]
    
    phot_fluxes = np.lib.recfunctions.structured_to_unstructured(np.array(phot_tab[list(flux_colNames)]))[0]
    phot_efluxes = np.lib.recfunctions.structured_to_unstructured(np.array(phot_tab[list(eflux_colNames)]))[0]
    
    # effective wavelengths of photometric filters
    phot_wavs = calc_eff_wavs(filt_list) / 10000
    

    # plotting spectrum
    if f_lam:
        spec_fluxes = convert_mujy2cgs(spec_fluxes, spec_wavs*10000)
        spec_efluxes = convert_mujy2cgs(spec_efluxes, spec_wavs*10000)
        
        phot_fluxes = convert_mujy2cgs(phot_fluxes, phot_wavs*10000)
        phot_efluxes = convert_mujy2cgs(phot_efluxes, phot_wavs*10000)

    fig,ax = plt.subplots(figsize=(10,4.5))

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

    # emission lines for overplotting
    line_wavelengths, line_ratios = get_line_wavelengths()
    line_names = ["Ha+NII", "OIII+Hb"]

    # plot emisison lines
    for i, line in enumerate(line_names):
        ax.vlines((np.array(line_wavelengths[line]))*(1+z_spec)/10000,
                  np.min(spec_fluxes), np.max(spec_fluxes),
                  color='slategrey', ls='--', lw=1, alpha=0.5)
        ax.text((np.array(line_wavelengths[line][-1])-200)*(1+z_spec)/10000,
                np.max(spec_fluxes)-0.15*np.max(spec_fluxes),
                line, rotation=90, color='slategrey', alpha=0.5)

    
    ##################################
    # --------- FORMATTING --------- #
    ##################################
    
    ax.set_xlabel('Wavelength [$\mathrm{\mu m}$]')
    ax.set_ylabel('${\\rm \mathrm{f_{\\nu}}\\ [\\mu Jy]}$')
    if f_lam:
        ax.set_ylabel('${\\rm \mathrm{f_{\\lambda}}\\ [erg\ s^{-1} cm^{-2} \AA^{-1}]}$')

    ax.set_xlim(np.min(spec_wavs), np.max(spec_wavs))
    
    # ax.legend(loc='upper left')
    
    plt.tight_layout()

    imname = 'x'
    # plt.savefig('images/{imname}.pdf', transparent=True)
    plt.show()



# --------------------------------------------------------------
# ----------------------------------- PLOT FITTED SPECTRAL MODEL
# --------------------------------------------------------------
def plot_fitted_spectrum(fit, f_lam=False):
    """
    Plots fitted BAGPIPES spectral model, observed spectrum and observed photometry of given source
    
    Parameters
    ----------
    fit : fit object from BAGPIPES (where fit = pipes.fit(galaxy, fit_instructions))
    f_lam : if True, output plot is in f_lambda, if False, in f_nu, format=bool

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

    phot_wavs = fit.galaxy.filter_set.eff_wavs/10000
    phot_fluxes = fit.galaxy.photometry[:,1]
    phot_efluxes = fit.galaxy.photometry[:,2]

    phot_fluxes_model = phot_post[:,1]
    phot_fluxes_model_lo = phot_post[:,0]
    phot_fluxes_model_hi = phot_post[:,2]

    if not f_lam:
        spec_fluxes = convert_cgs2mujy(spec_fluxes, wavs*10000)
        spec_efluxes = convert_cgs2mujy(spec_efluxes, wavs*10000)

        spec_fluxes_model = convert_cgs2mujy(spec_fluxes_model, wavs*10000)
        spec_fluxes_model_lo = convert_cgs2mujy(spec_fluxes_model_lo, wavs*10000)
        spec_fluxes_model_hi = convert_cgs2mujy(spec_fluxes_model_hi, wavs*10000)

        phot_fluxes = convert_cgs2mujy(phot_fluxes, phot_wavs*10000)
        phot_efluxes = convert_cgs2mujy(phot_efluxes, phot_wavs*10000)

        phot_fluxes_model = convert_cgs2mujy(phot_fluxes_model, phot_wavs*10000)
        phot_fluxes_model_lo = convert_cgs2mujy(phot_fluxes_model_lo, phot_wavs*10000)
        phot_fluxes_model_hi = convert_cgs2mujy(phot_fluxes_model_hi, phot_wavs*10000)
    
    # plotting spectrum
    fig,ax = plt.subplots(figsize=(10,4.5))
    
    ##################################
    # ------------ DATA ------------ #
    ##################################
    
    # ---------- SPECTRUM ---------- #
    ax.plot(wavs, spec_fluxes,
            zorder=-1, color='slategrey', alpha=0.7, lw=1, label='spectrum (scaled)')
    ax.fill_between(wavs,
                    spec_fluxes-spec_efluxes,
                    spec_fluxes+spec_efluxes,
                    zorder=-1, color='slategrey', alpha=0.1)
    
    # ---------- PHOTOMETRY ---------- #
    ax.errorbar(phot_wavs, phot_fluxes, yerr=phot_efluxes,
                fmt='o', ms=8, color='gainsboro', markeredgecolor='k', ecolor='grey', elinewidth=0.5, markeredgewidth=1.,
                zorder=1, alpha=1., label='photometry')
    
    
    ##################################
    # -------- FITTED MODEL -------- #
    ##################################
    
    # ---------- SPECTRUM ---------- #
    ax.plot(wavs, spec_fluxes_model,
            zorder=-1, color='forestgreen', alpha=0.7, lw=1.5, label='model spectrum (scaled)')
    ax.fill_between(wavs,
                    spec_fluxes_model_lo, spec_fluxes_model_hi,
                    zorder=-1, color='forestgreen', alpha=0.1)
    
    # ---------- PHOTOMETRY ---------- #
    ax.errorbar(phot_wavs, phot_fluxes_model, #yerr=[phot_fluxes_model_lo, phot_fluxes_model_hi],
                fmt='o', ms=8, color='cornflowerblue', markeredgecolor='cornflowerblue', ecolor='grey', elinewidth=0.5, markeredgewidth=1.,
                zorder=1, alpha=0.9, label='model photometry')
    
    
    ##################################
    # --------- FORMATTING --------- #
    ##################################
    
    ax.set_xlabel('Wavelength [$\mathrm{\mu m}$]')
    ax.set_ylabel('${\\rm \mathrm{f_{\\lambda}}\\ [erg\ s^{-1} cm^{-2} \AA^{-1}]}$')
    if not f_lam:
        ax.set_ylabel('${\\rm \mathrm{f_{\\nu}}\\ [\\mu Jy]}$')
    
    ax.legend()
    
    plt.tight_layout()
    
    imname = ''
    # plt.savefig(f'images/{imname}.pdf', transparent=True)
    plt.show()



# --------------------------------------------------------------
# ---------------------------------- PLOT STAR-FORMATION HISTORY
# --------------------------------------------------------------
def plot_fitted_sfh(fit):
    """
    Plots star-formation history from fitted BAGPIPES model
    
    Parameters
    ----------
    fit : fit object from BAGPIPES (where fit = pipes.fit(galaxy, fit_instructions))

    Returns
    -------
    None
    """
    
    fit.posterior.get_advanced_quantities()
    
    color1 = "slategrey"
    color2 = "slategrey"
    alpha = 0.6
    zorder=4
    label=None
    # zvals=[0, 0.5, 1, 2, 4, 10]
    # z_axis=True

    z_array = np.arange(0., 100., 0.01)
    age_at_z = cosmo.age(z_array).value
    
    
    # Calculate median redshift and median age of Universe
    if "redshift" in fit.fitted_model.params:
        redshift = np.median(fit.posterior.samples["redshift"])

    else:
        redshift = fit.fitted_model.model_components["redshift"]

    age_of_universe = np.interp(redshift, z_array, age_at_z)

    # Calculate median and confidence interval for SFH posterior
    post = np.percentile(fit.posterior.samples["sfh"], (16, 50, 84), axis=0).T

    # Plot the SFH
    x = age_of_universe - fit.posterior.sfh.ages*10**-9

    fig,ax = plt.subplots(figsize=(10,4.5))

    ax.plot(x, post[:, 1], color=color1, zorder=zorder+1)
    ax.fill_between(x, post[:, 0], post[:, 2], color=color2,
                    alpha=alpha, zorder=zorder, lw=0, label=label)

    ax.set_ylim(0., np.max([ax.get_ylim()[1], 1.1*np.max(post[:, 2])]))
    ax.set_xlim(age_of_universe, 0)


    # Set axis labels
    ax.set_ylabel("${\\rm SFR \ [ M_\\odot yr^{-1}]}$")
    ax.set_xlabel("Age of Universe [Gyr]")


    plt.tight_layout()


# --------------------------------------------------------------
# -------------------------------- TABLE OF POSTERIOR PROPERTIES
# --------------------------------------------------------------
def get_posterior_sample_dists(fit):
    """
    Makes table of 16th, 50th, and 84th percentile values of all posterior quantities from BAGPIPES fit
    
    Parameters
    ----------
    fit : fit object from BAGPIPES (where fit = pipes.fit(galaxy, fit_instructions))

    Returns
    -------
    tab_info : list with dimensions n*4 (where n is number of posterior quantities)
    """

    fit.posterior.get_advanced_quantities()

    samples = fit.posterior.samples
    keys = list(samples.keys())

    tab_info = []

    for i in range(len(keys)):
    
        tab_info.append(list([keys[i],
                              np.round(np.percentile(samples[keys[i]],(16)),3),
                              np.round(np.percentile(samples[keys[i]],(50)),3),
                              np.round(np.percentile(samples[keys[i]],(84)),3)]))

    return(tab_info)
