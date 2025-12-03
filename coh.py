#! venv/bin/python

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
    
    path = Path("data")

    for filename in path.glob("*.bin"):
        print(f"Processing {filename.stem}")
        binning = 2**0
        tau = 9.765625e-7*binning*2
        #tau = 0.000200089*binning*2
        print(f"{tau*1e-3:} s")


        data = np.fromfile(filename, dtype=np.float64)
        print(f"shape: {data.shape}")
        data = bin_time(data, binning)
        print(f"mean:  {np.mean(data)}")
        print(f"STD:   {np.std(data)}")

        t = np.arange(len(data)) * tau

        plt.figure()
        plt.title(f"{filename.stem} $\\tau = $ {tau*1e6:.3f} ns")
        plt.plot(t, data)
        plt.xlabel("t, ms")

        tmax = t[np.argmax(data)]
        dt = 1e-1
        #plt.xlim(tmax-dt, tmax+dt)
        #plt.ylim(-5, 40)


#        P = 0.71446991958999994665
#        obs_window = int(P*1e3 / tau);
#        int_prf = np.zeros(obs_window)
#        skip = 0
#        summed = 0
#
#        plt.figure()
#        for i in range(int(t[-1]*1e-3 / P)):
#            print(f"rev: {i}")
#            skip = int(i*(P*1e3/tau - obs_window) + .5)
#            temp = data[i*obs_window + skip: (i+1)*obs_window + skip]
#            int_prf += temp
#
#            plt.plot(t[:obs_window], temp, label = f"{i}")
#            #plt.ylim(1.850, 1.950)
#
#            summed += 1
#
#        #plt.legend()
#
#        int_prf = int_prf/summed
#
#        plt.figure()
#        plt.plot(t[:obs_window], int_prf)
#        #plt.ylim(1.850, 1.950)

    save_image("plot.pdf")
