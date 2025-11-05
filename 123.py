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
    filename = "data.bin"
    binning = 1

# Read as flat array, then reshape
    data = np.fromfile(filename, dtype=np.float64)
    data = bin_time(data, binning)

    tau = 2000e-9/1024 * binning
    tau = 200e-9*binning
    o = np.argmax(data)
    t = np.arange(len(data)) * tau
    t -= t[o]
    #data = data[7200000 : 7_300_000]
    print(np.any(np.isnan(data)))

    plt.figure()
    plt.plot(t*1e6, data)
    plt.xlabel("t, us")
    delta = 10
    plt.xlim(-delta, +delta)

    plt.figure()
    plt.plot(t*1e6, data)
    plt.xlabel("t, us")
    delta = 100
    plt.xlim(-delta, +delta)

    plt.figure()
    plt.plot(t*1e6, data)
    plt.xlabel("t, us")
    delta = 1000
    #plt.xlim(-delta, +delta)

    
    save_image("123.pdf")
