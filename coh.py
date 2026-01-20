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

    for filename in path.glob("sum*.bin"):
        print(f"Processing {filename.stem}")
        binning = 2**8
        tau = 9.765625e-7*binning*2
        tau = 0.000200089*binning*2
        print(f"tau:   {tau*1e-3:} s")


        data = np.fromfile(filename, dtype=np.float64)
        #data = data.reshape(-1, 4096)
        print(f"shape: {data.shape}")
        #data = bin_time(data, binning)

        sig = 3.5
        clipped, lower, upper = sigmaclip(data, sig, sig)

        plt.figure()
        plt.plot(data)


        #plt.imshow(data.T,
        #           origin = "lower",
        #           cmap = "Greys",
        #           vmin = lower * (1 + .2),
        #           vmax = upper * (1 - .2),
        #           aspect = "auto")




    save_image("plot.pdf")
