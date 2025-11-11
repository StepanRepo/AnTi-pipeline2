#! venv/bin/python

import numpy as np
import matplotlib.pyplot as plt
from myplot import *
from pathlib import Path
from tqdm import tqdm

import astropy.units as u
from astropy.time import Time
from scipy.stats import sigmaclip

def bin_time(data, bin_size, method='mean'):
    T, F = data.shape
    n_bins = T // bin_size
    trimmed = data[:n_bins * bin_size]
    reshaped = trimmed.reshape(n_bins, bin_size, F)
    
    return reshaped.sum(axis=1)




if __name__ == "__main__":
    path = Path(".")

    for filename in path.glob("*.bin"):
        print(f"Processing {filename.stem}")

        freq_num = 512
        binning = 1

# Read as flat array, then reshape
        data = np.fromfile(filename, dtype=np.float64)
        data_2d = data.reshape(-1, freq_num)
        data_2d = bin_time(data_2d, binning).T
        
        


        #fmax = 1772.00 
        fmin = 0 * u.MHz
        fmax = 1 * u.MHz
        sampling = 5*u.MHz 

        nchan = data_2d.shape[0]
        tau = (nchan / sampling).to(u.us) * binning * 2
        delta_f = (fmax - fmin) / (nchan)

        tl = np.arange(data_2d.shape[1])*tau
        freqs = fmax - np.arange(nchan)[::-1] * delta_f

        fr = np.sum(data_2d, axis = 1)
        tails_mask = fr > .5*np.median(fr)

        diff = np.diff(np.log(fr))
        diff = np.concat([diff, [0.0]])
        masked, lower, upper = sigmaclip(diff, 3, 3)
        mask = (diff > lower) & (diff < upper)

        mask = mask & tails_mask
        #mask[:] = 1





        data_2d = data_2d / np.sum(data_2d, axis = 1)[:, np.newaxis]
        data_2d[~mask, :] = np.nan

        



        fig, ax = plt.subplots(2, 2, 
                               width_ratios = [2, 1],
                               height_ratios = [2, 1],
                               )

        fig.suptitle(filename.stem)

        extent = [tl[0].to_value(u.ms), tl[-1].to_value(u.ms), 
                  freqs[0].to_value(u.MHz), freqs[-1].to_value(u.MHz)]

        ax[1, 1].set_visible(False)
        ax[0, 0].sharex(ax[1, 0]) 






        data_2d[~mask, :] = np.nanmedian(data_2d)

        sig = 3
        clipped, lower, upper = sigmaclip(data_2d, sig, sig)

        while len(data_2d[(data_2d > lower) & (data_2d < upper)])/len(data_2d.ravel()) < .5:
            sig += 1
            clipped, lower, upper = sigmaclip(data_2d, sig, sig)
            print(sig, upper, lower)

            if sig > 100:
                break
            



        ax[0, 0].imshow(data_2d, 
                        origin = "lower", 
                        aspect = "auto",
                        cmap = "Greys",
                        extent = extent,
                        vmin = lower,
                        vmax = upper,
                        )


        ax[1, 0].plot(tl.to(u.ms), np.nanmean(data_2d, axis = 0))




        ax[0, 1].set_xscale("log")
        ax[0, 1].plot(fr, freqs)
        fr[~mask] = np.nan
        ax[0, 1].plot(fr, freqs)


        ax[0, 1].set_yticks([])

        ax[0, 0].set_ylabel("Frequency, MHz")
        ax[0, 1].set_xlabel("FR Intensity")
        ax[1, 0].set_xlabel("Time, ms")
        ax[1, 0].set_ylabel("Integal Intensity")



        #ax[0, 0].set_xlim(280, 380)




    save_image("123.pdf")
