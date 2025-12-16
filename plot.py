#! venv/bin/python

import numpy as np
import matplotlib.pyplot as plt
from myplot import *
from pathlib import Path
from tqdm import tqdm

import astropy.units as u
from astropy.time import Time
from astropy.io import fits
from scipy.stats import sigmaclip, kurtosis, kurtosistest
from scipy import signal

def bin_time(data, bin_size):
    T, F = data.shape
    n_bins = T // bin_size
    trimmed = data[:n_bins * bin_size]
    reshaped = trimmed.reshape(n_bins, bin_size, F)
    
    return reshaped.sum(axis=1)

def read_psr(hdul):
    """
    Read a fold-mode PSRFITS file and return dynamic spectrum as a 3D array:
    (nsubint, nchan, nbin)
    Also returns frequency array (MHz) and time array (s).
    """
    # Check observation mode
    primary_hdr = hdul[0].header
    if primary_hdr.get('OBS_MODE', '').strip() != 'PSR':
        raise ValueError("File is not in fold-mode (OBS_MODE != 'PSR')")

    subint_hdu = hdul['SUBINT']
    header = subint_hdu.header

    nsubint = header['NAXIS2']
    nchan   = header['NCHAN']
    nbin    = header['NBIN']
    npol    = header['NPOL']

    print(f"File contains {nsubint} subints, {nchan} channels, {nbin} bins, {npol} pols")

    # Read data and metadata
    si_data = subint_hdu.data
    data     = si_data['DATA']           # Shape: (nsubint, nbin, nchan, npol)
    dat_freq = si_data['DAT_FREQ']       # Shape: (nsubint, nchan)
    dat_wts  = si_data['DAT_WTS']        # Shape: (nsubint, nchan)
    dat_scl  = si_data['DAT_SCL']        # Shape: (nsubint, nchan*npol)
    dat_offs = si_data['DAT_OFFS']       # Shape: (nsubint, nchan*npol)
    tsubint  = si_data['TSUBINT']        # Duration of each subint
    tau      = header['TBIN']

    if (nchan == 1):
        dat_scl = dat_scl[:, np.newaxis]
        dat_offs = dat_offs[:, np.newaxis]
        dat_wts = dat_wts[:, np.newaxis]


    real_data = data.astype(np.float64) 
    real_data = (real_data * dat_scl[np.newaxis, :, :, np.newaxis] + dat_offs[np.newaxis, :, :, np.newaxis]) * dat_wts[np.newaxis, :, :, np.newaxis]

    real_data = np.transpose(real_data, (0, 1, 3, 2))

    # Collapse polarization (if needed) — assume Stokes I = sum of pols
    if npol > 1:
        real_data = real_data.sum(axis=1)  # Shape: (nsubint, nbin, nchan)
    else:
        real_data = real_data[:, 0, :, :]       # Shape: (nsubint, nbin, nchan)



    # Compute time axis (seconds from start)
    offs_sub = si_data['OFFS_SUB']  # Center time of each subint
    t0 = offs_sub - tsubint / 2.0       # Start time of each subint
    time_s = t0 + np.arange(nbin)*tau

    # Use frequency from first subint (assumed constant)
    freq_mhz = dat_freq[0]  # Shape: (nchan,)

    return real_data[0], freq_mhz, time_s

def read_search(hdul):
    """
    Read a fold-mode PSRFITS file and return dynamic spectrum as a 3D array:
    (nsubint, nchan, nbin)
    Also returns frequency array (MHz) and time array (s).
    """
    # Check observation mode
    primary_hdr = hdul[0].header
    if primary_hdr.get('OBS_MODE', '').strip() != 'SEARCH':
        raise ValueError("File is not in search-mode (OBS_MODE != 'SEARCH')")

    subint_hdu = hdul['SUBINT']
    header = subint_hdu.header

    nsubint = header['NAXIS2']
    nchan   = header['NCHAN']
    nbin    = header['NSBLK']
    nstot   = header['NSTOT']
    npol    = header['NPOL']
    signint = header['SIGNINT']        # Is the data stored as signed ints 
    tau     = header['TBIN']

    print(f"File contains {nsubint} subints, {nchan} channels, {nbin} bins, {npol} pols")

    # Read data and metadata
    si_data = subint_hdu.data
    data     = si_data['DATA']           # Shape: (nsubint, nbin, nchan, npol)
    dat_freq = si_data['DAT_FREQ']       # Shape: (nsubint, nchan)
    dat_wts  = si_data['DAT_WTS']        # Shape: (nsubint, nchan)
    dat_scl  = si_data['DAT_SCL']        # Shape: (nsubint, nchan*npol)
    dat_offs = si_data['DAT_OFFS']       # Shape: (nsubint, nchan*npol)
    tsubint  = si_data['TSUBINT']        # Duration of each subint

    if (signint == 1 and header['NBITS'] == 8):
        data = data.astype(np.int8)



    if (nchan == 1):
        dat_scl = dat_scl[:, np.newaxis]
        dat_offs = dat_offs[:, np.newaxis]
        dat_wts = dat_wts[:, np.newaxis]

    # Cast DATA to float and apply scale/offset
    # PSRFITS stores data in (bin, chan, pol) order
    real_data = data.astype(np.float64) 
    real_data = (real_data * dat_scl[:, np.newaxis, np.newaxis, :] + dat_offs[:, np.newaxis, np.newaxis, :]) * dat_wts[:, np.newaxis, np.newaxis, :]

    #plt.figure()
    #plt.plot(dat_offs[0])
    #plt.plot(np.mean(real_data[0, :, 0, :], axis = 0))
    #plt.figure()
    #plt.plot(dat_offs[0] - np.mean(real_data[0, :, 0, :], axis = 0))

    real_data = real_data.reshape(-1, *real_data.shape[-2:])
    real_data = real_data[:nstot, :, :]

    # Collapse polarization (if needed) — assume Stokes I = sum of pols
    if npol > 1:
        real_data = real_data.sum(axis=-2)  # Shape: (nsubint, nbin, nchan)
    else:
        real_data = real_data[:, 0, :]       # Shape: (nsubint, nbin, nchan)



    # Use frequency from first subint (assumed constant)
    freq_mhz = dat_freq[0]  # Shape: (nchan,)

    time_s = np.arange(real_data.shape[0])*tau

    return real_data, freq_mhz, time_s


def read(filename):
    with fits.open(filename, memmap = True) as hdul:

        mode = hdul[0].header.get('OBS_MODE', '').strip()
        print(f"Observational mode: {mode}")

        match mode:
            case "PSR":
                return read_psr(hdul)
            case "SEARCH":
                return read_search(hdul)
            case _:
                raise ValueError(f"Unknown OBS_MODE: {mode}")



if __name__ == "__main__":
    path = Path(".")
    path = Path("data")
    files = np.sort(list(path.glob("*.fits")))

    for filename in files:
        print(f"Processing {filename.stem}")

        binning = 2**0

        data_2d, freqs, tl = read(filename)
        data_2d = data_2d.T

        tl = tl * u.s
        freqs = freqs * u.MHz

        data_2d = bin_time(data_2d.T, binning).T
        tl = tl[::binning][:data_2d.shape[1]]

        if (data_2d.shape[0] > 1):

            data_2d[data_2d == 0.0] = np.median(data_2d[data_2d > 0])


            nchan = data_2d.shape[0]
            fr = np.sum(data_2d, axis = 1)


            fig, ax = plt.subplots(2, 2, 
                                   width_ratios = [2, 1],
                                   height_ratios = [2, 1],
                                   )

            fig.suptitle(filename.stem)

            extent = [tl[0].to_value(u.ms), tl[-1].to_value(u.ms), 
                      freqs[0].to_value(u.MHz), freqs[-1].to_value(u.MHz)]

            ax[1, 1].set_visible(False)
            ax[0, 0].sharex(ax[1, 0]) 



            sig = 3
            clipped, lower, upper = sigmaclip(data_2d, sig, sig)

            while len(data_2d[(data_2d > lower) & (data_2d < upper)])/len(data_2d.ravel()) < .5:
                sig += 1
                clipped, lower, upper = sigmaclip(data_2d, sig, sig)
                print(sig, upper, lower)

                if sig > 10:
                    break

            ax[0, 0].imshow(data_2d, 
                            origin = "lower", 
                            aspect = "auto",
                            cmap = "Greys",
                            extent = extent,
                            vmin = lower,
                            vmax = upper,
                            #interpolation = "none",
                            )
            ax[1, 0].plot(tl.to(u.ms), np.nanmean(data_2d, axis = 0))


            ax[0, 1].set_xscale("log")
            ax[0, 1].plot(fr, freqs)
            ax[0, 1].set_ylim(freqs[0].to_value(u.MHz), freqs[-1].to_value(u.MHz))



            #clipped, l, u = sigmaclip(np.log(fr[fr>0]), 3, 3)
            #mask = (np.log(fr) > l) & (np.log(fr) < u)
            #fr[~mask] = np.nan
            #ax[0, 1].plot(fr, freqs)


            ax[0, 1].set_yticks([])

            ax[0, 0].set_ylabel("Frequency, MHz")
            ax[0, 1].set_xlabel("FR Intensity")
            ax[1, 0].set_xlabel("Time, ms")
            ax[1, 0].set_ylabel("Integal Intensity")



        #ax[0, 0].set_xlim(280, 380)
        else:


            fig, ax = plt.subplots(1, 1)
            fig.suptitle(filename.stem)
    
            #M = (10e-6 * u.s) / (tl[1] - tl[0])
            #M = int(M.to_value(u.dimensionless_unscaled))
            #data_2d[0] = signal.fftconvolve(data_2d[0], np.ones(M)/M, "same")

            val = u.ms

            ax.plot(tl.to(val), data_2d[0])
            ax.set_xlabel(f"Time, {val}")
            ax.set_ylabel("Integal Intensity")

            tmax = tl[np.argmax(data_2d[0])]
            dt = 5*u.us
            #ax.set_xlim((tmax - dt).to_value(val), (tmax + dt).to_value(val))




    save_image("plot.pdf", tight = True)
