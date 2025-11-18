#! venv/bin/python

import numpy as np
import matplotlib.pyplot as plt
from myplot import *
from pathlib import Path
from tqdm import tqdm

import astropy.units as u
from astropy.time import Time
from astropy.io import fits
from scipy.stats import sigmaclip

def bin_time(data, bin_size):
    T, F = data.shape
    n_bins = T // bin_size
    trimmed = data[:n_bins * bin_size]
    reshaped = trimmed.reshape(n_bins, bin_size, F)
    
    return reshaped.sum(axis=1)

def read_dyn(filename):
    """
    Read a fold-mode PSRFITS file and return dynamic spectrum as a 3D array:
    (nsubint, nchan, nbin)
    Also returns frequency array (MHz) and time array (s).
    """
    with fits.open(filename, memmap=True) as hdul:
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

        print(f"File contains {len(hdul)} subints, {nchan} channels, {nbin} bins, {npol} pols")

        # Read data and metadata
        data     = subint_hdu.data['DATA']           # Shape: (nsubint, nbin, nchan, npol)
        dat_freq = subint_hdu.data['DAT_FREQ']       # Shape: (nsubint, nchan)
        dat_wts  = subint_hdu.data['DAT_WTS']        # Shape: (nsubint, nchan)
        dat_scl  = subint_hdu.data['DAT_SCL']        # Shape: (nsubint, nchan*npol)
        dat_offs = subint_hdu.data['DAT_OFFS']       # Shape: (nsubint, nchan*npol)
        tsubint  = subint_hdu.data['TSUBINT']        # Duration of each subint
        tau      = subint_hdu.header['TBIN']

        # Reconstruct real-valued data
        # Reshape scale/offset to (nsubint, nchan, npol)
        data = data.reshape(nsubint, nbin, nchan, npol)
        dat_wts = dat_wts.reshape(nsubint, nchan)
        dat_scl  = dat_scl.reshape(nsubint, nchan, npol)
        dat_offs = dat_offs.reshape(nsubint, nchan, npol)

        # Cast DATA to float and apply scale/offset
        # PSRFITS stores data in (bin, chan, pol) order
        real_data = data.astype(np.float64) 
        real_data = real_data * dat_scl + dat_offs

        # Collapse polarization (if needed) — assume Stokes I = sum of pols
        if npol > 1:
            real_data = real_data.sum(axis=-1)  # Shape: (nsubint, nbin, nchan)
        else:
            real_data = real_data[..., 0]       # Shape: (nsubint, nbin, nchan)

        # Transpose to (nsubint, nchan, nbin) for easier plotting
        dynamic_spectrum = np.transpose(real_data, (0, 2, 1))  # (subint, chan, bin)

        # Compute time axis (seconds from start)
        offs_sub = subint_hdu.data['OFFS_SUB']  # Center time of each subint
        t0 = offs_sub - tsubint / 2.0       # Start time of each subint
        time_s = t0 + np.arange(nbin)*tau

        # Use frequency from first subint (assumed constant)
        freq_mhz = dat_freq[0]  # Shape: (nchan,)

        return dynamic_spectrum, freq_mhz, time_s



if __name__ == "__main__":
    path = Path(".")
    path = Path("data")

    for filename in path.glob("*.psrfits"):
        print(f"Processing {filename.stem}")

        binning = 1

        data_2d, freqs, tl = read_dyn(filename)
        data_2d = data_2d[0, :, :]
        data_2d = bin_time(data_2d, binning)

        tl = tl * u.s
        freqs = freqs * u.MHz
        
        nchan = data_2d.shape[0]
        fr = np.sum(data_2d, axis = 1)
        #tails_mask = fr > .5*np.median(fr)

        #diff = np.diff(np.log(fr))
        #diff = np.concat([diff, [0.0]])
        #masked, lower, upper = sigmaclip(diff, 3, 3)
        #mask = (diff > lower) & (diff < upper)

        #mask = mask & tails_mask
        ##mask[:] = 1


        #data_2d = data_2d / np.sum(data_2d, axis = 1)[:, np.newaxis]
        #data_2d[~mask, :] = np.nan

        



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
                        interpolation = "none",
                        )

        ax[1, 0].plot(tl.to(u.ms), np.nanmean(data_2d, axis = 0))


        ax[0, 1].set_xscale("log")
        ax[0, 1].plot(fr, freqs)
        #fr[~mask] = np.nan
        #ax[0, 1].plot(fr, freqs)


        ax[0, 1].set_yticks([])

        ax[0, 0].set_ylabel("Frequency, MHz")
        ax[0, 1].set_xlabel("FR Intensity")
        ax[1, 0].set_xlabel("Time, ms")
        ax[1, 0].set_ylabel("Integal Intensity")



        #ax[0, 0].set_xlim(280, 380)




    save_image("plot.pdf")
