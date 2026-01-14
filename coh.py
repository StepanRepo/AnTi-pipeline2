#! venv/bin/python

import numpy as np
import matplotlib.pyplot as plt
from myplot import *
from pathlib import Path
from tqdm import tqdm
from plot import bin_time

import astropy.units as u
from astropy.time import Time
from scipy.stats import sigmaclip





if __name__ == "__main__":
    
    path = Path("data")

    for filename in path.glob("*.bin"):
        print(f"Processing {filename.stem}")
        binning = 2**2
        tau = 9.765625e-7*binning*2
        tau = 0.000200089*binning*2
        print(f"tau:   {tau*1e-3:} s")


        data = np.fromfile(filename, dtype=np.float64)
        data = data.reshape(-1, 2048)
        print(f"shape: {data.shape}")
        data = bin_time(data, binning)

        sig = 3.5
        clipped, lower, upper = sigmaclip(data, sig, sig)

        fig, ax = plt.subplots(2, 1, sharex = True, height_ratios = [3, 1])
        fig.suptitle(f"{filename.stem} $\\tau = $ {tau*1e6:.3f} ns")

        ax[0].imshow(data.T,
                     origin = "lower",
                     cmap = "Greys",
                     vmin = lower * (1 + .2),
                     vmax = upper * (1 - .2),
                     aspect = "auto")

        ax[0].axvline(12030 / binning)
        ax[0].axvline(12030*2 / binning)
        ax[1].plot(np.sum(data, axis = 1))




    save_image("plot.pdf")
