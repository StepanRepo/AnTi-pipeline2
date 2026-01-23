#! venv/bin/python

import numpy as np
import matplotlib.pyplot as plt
from myplot import *
import pandas as pd
from plot import read, bin_time
from libstempo import tempopulsar
from pathlib import Path
import gc

binning = 2**0
dt = 10e-3
name = "rup103_bv_055-0550_ch01"
name = "rup103_bv_055-0612_ch01"

path = Path("rup103")
df = pd.read_csv(path / f"{name}.vdif.csv", comment = "#", sep = ";", header = None)

files = np.unique(df.iloc[:, 1])
counter = 0
toas = []

for n in files:

    if n!=33:
        continue

    lines = df.iloc[:, 1] == n

    if np.sum(lines) > 1:
        continue

    if np.all(df[lines][5] < 8):
        continue

    left = df[lines][2].to_numpy()[0]
    right = df[lines][3].to_numpy()[0]
    toas.append((left+right)/2)



    conv, _, tlc = read(path / f"conv_{name}_{n}.fits") 
    sum, _, tls = read(path / f"sum_{name}_{n}.fits") 

    tlc = tlc*1e3
    tls = tls*1e3
    sum = bin_time(sum, binning)
    tls = tls[::binning]
    tls = tls[:len(sum)]
    tmax = tlc[np.argmax(conv)]
    tmax = tls[np.argmax(sum)]



    fig, ax = plt.subplots(2, 1, sharex = True)
    fig.suptitle(f"sum_{name}_{n}.fits")

    ax[0].plot(tls, sum)
    ax[0].axvline(tmax, c = "k", lw = .5, ls = "--", zorder = -1)

    ax[1].plot(tlc, conv)
    ax[1].axhline(7, c = "k", ls = "--")
    ax[1].set_xlim(tmax - dt, tmax + dt)

    del conv, tlc, sum, tls
    gc.collect()


    counter += 1

    #if counter == 70:
    #    break

#psr = tempopulsar(parfile = "rup103/0531.par",
#                  timfile = "tim",
#                  dofit = False
#                  )


#toas = np.array(toas)
#psr = tempopulsar(parfile = "rup103/0531.par",
#                  toas = toas,
#                  toaerrs = [1e-6]*len(toas),
#                  obsfreq = 2675.8,
#                  observatory = "bad13m")
#psr.savetim("tim")
#
#
#plt.figure()
#plt.title(f"{psr.name} ($\\sigma = {np.std(psr.residuals()*1e6):.3f}$ us)")
#
#plt.plot(psr.toas(), psr.residuals()*1e3, "o")
#
#for i in range(len(toas)):
#    plt.annotate(i+1, (psr.toas()[i], psr.residuals()[i]*1e3))
#
#plt.ylabel("Timing Residuals, ms")
#plt.xlabel("Times of Arrival, MJD")
#
#
#df = pd.read_csv("/mnt/prognoz-nas/data/n08_pulsar_proc/m5b/rup103_bd_no0003")
#df = df[df["snrr"] > 8]
#
#toas1 = 60359.258344907407409 + df["tr"].to_numpy()/86400
#
#psr1 = tempopulsar(parfile = "rup103/0531.par",
#                   toas = toas1,
#                   toaerrs = [1e-6]*len(toas1),
#                   obsfreq = 2675.8,
#                   observatory = "bad13m")
#
#delta = (np.median(psr1.residuals()) - psr.residuals()[1])*1e3
#plt.plot(psr1.toas(), psr1.residuals()*1e3 - delta, "o", zorder = -1)



save_image("plot.pdf")

