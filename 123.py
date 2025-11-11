#! /bin/python

import numpy as np
import matplotlib.pyplot as plt
from myplot import *
from pathlib import Path
from tqdm import tqdm

import astropy.units as u
from astropy.time import Time
from scipy.stats import sigmaclip

def bin_time(data, bin_size):
    T, = data.shape
    n_bins = T // bin_size
    trimmed = data[:n_bins * bin_size]
    reshaped = trimmed.reshape(n_bins, bin_size)
    
    return reshaped.mean(axis=1)




if __name__ == "__main__":
    
    path = Path(".")

    for filename in path.glob("*.bin"):
        print(f"Processing {filename.stem}")
        binning = 1024
        tau = 0.000200089*binning
        print(tau)

# Read as flat array, then reshape
        data = np.fromfile(filename, dtype=np.float64)
        data = bin_time(data, binning)

        t = np.arange(len(data)) * tau

        plt.figure()
        plt.title(filename.stem)
        plt.plot(t, data)
        plt.xlabel("t, ms")

        P = 0.71446991958999994665
        obs_window = int(P*1e3 / tau);
        int_prf = np.zeros(obs_window)
        skip = 0

        plt.figure()
        for i in range(len(data)//obs_window):
            skip = int(i*(P*1e3/tau - obs_window) + .5)
            temp = data[i*obs_window + skip: (i+1)*obs_window + skip]
            int_prf += temp

            plt.plot(t[:obs_window], temp)

        plt.figure()
        plt.plot(t[:obs_window], int_prf)
    
    save_image("123.pdf")
