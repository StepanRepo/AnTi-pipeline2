#! /bin/python

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
    filename = "data.bin"
    obs_window = 1000000

# Read as flat array, then reshape
    data = np.fromfile(filename, dtype=np.float64)
    #data = data[7200000 : 7_300_000]
    print(np.any(np.isnan(data)))

    plt.figure()
    plt.plot(np.log(data))
    
    save_image("123.pdf")
