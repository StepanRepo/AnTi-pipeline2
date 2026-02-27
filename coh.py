#! venv/bin/python

import numpy as np
import matplotlib.pyplot as plt
from myplot import *
from pathlib import Path
from tqdm import tqdm
from plot import bin_time

import astropy.units as u
from astropy.time import Time
from scipy.stats import alpha, median_abs_deviation, sigmaclip 


def norm(data):
    mu = np.median(data)
    mad = median_abs_deviation(data)
    sig = 1.4826*mad

    return (data-mu)/sig



if __name__ == "__main__":
    
    path = Path("data")

    # summ = np.fromfile(path/"sum.bin", np.float64)
    # conv = np.fromfile(path/"conv.bin", np.float64)
    # ker  = np.fromfile(path/"ker.bin", np.float64)

    # plt.figure()
    # plt.plot(norm(summ))
    # plt.plot(conv, alpha = .7, zorder = -1)

    # plt.show()


    for filename in path.glob("*.bin"):
        print(f"Processing {filename.stem}")
        binning = 2**0
        tau = 12.4928e-3
        nchann = 1
        print(f"tau:   {tau*1e-3:} s")


        data = np.fromfile(filename, dtype=np.float64)
        data = data.reshape(-1, nchann)
        n, _ = data.shape
        data = bin_time(data, binning)

        #data = data.reshape(*(data.shape[::-1]))


        print(f"shape: {data.shape}")
        print()
        if (nchann == 1):
            plt.figure()
            plt.title(filename.stem)
            plt.plot(data[:])

        else:

            sig = 3.5
            clipped, lower, upper = sigmaclip(data, sig, sig)



            plt.figure()
            plt.title(filename.stem)
            plt.imshow(data.T,
                       origin = "lower",
                       cmap = "Greys",
                       vmin = lower * (1 + .2),
                       vmax = upper * (1 - .2),
                       aspect = "auto")

            #plt.figure()
            #plt.plot(np.mean(data, axis = 1))
            #plt.figure()
            #plt.plot(np.mean(data, axis = 0))




    save_image("plot.pdf")
