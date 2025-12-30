#! venv/bin/python

import numpy as np
import matplotlib.pyplot as plt
from myplot import *
import pandas as pd
from plot import read, bin_time
from libstempo import tempopulsar

binning = 2**10
dt = 2
name = "rup103_bv_055-0550_ch01"
name = "rup103_bv_055-0612_ch01"
df = pd.read_csv(f"rup103/{name}.vdif.csv", comment = "#", sep = ";", header = None)

files = np.unique(df.iloc[:, 0])
counter = 0
toas = []

for n in files:
    lines = df.iloc[:, 0] == n

    if np.sum(lines) > 1:
        continue

    if np.all(df[lines][4] < 9):
        continue

    toas.append(df[lines][1].to_numpy()[0])


    conv, _, tlc = read(f"rup103/conv_{name}_{n}.fits") 
    sum, _, tls = read(f"rup103/sum_{name}_{n}.fits") 

    tlc = tlc*1e3
    tls = tls*1e3
    sum = bin_time(sum, binning)
    tls = tls[::binning]
    tls = tls[:len(sum)]
    tmax = tlc[np.argmax(conv)]



    fig, ax = plt.subplots(2, 1, sharex = True)
    fig.suptitle(f"sum_{name}_{n}.fits")
    ax[0].plot(tlc, conv)
    ax[0].axhline(7, c = "k", ls = "--")

    ax[1].plot(tls, sum)
    ax[1].axvline(tmax, c = "k", lw = .5, ls = "--", zorder = -1)

    ax[1].set_xlim(tmax - dt, tmax + dt)

    counter += 1

    if counter == 70:
        break

toas = np.array(toas)

psr = tempopulsar(parfile = "rup103/0531.par",
                  toas = toas,
                  toaerrs = [1e-9]*len(toas),
                  obsfreq = 2675.8,
                  observatory = "bad13m")
psr.savetim("tim")

plt.figure()
plt.plot(psr.toas(), psr.residuals(), "o")

save_image("plot.pdf")

