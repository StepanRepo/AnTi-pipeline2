#! venv/bin/python

import numpy as np
import matplotlib.pyplot as plt
from myplot import *
import pandas as pd
from plot import read
from scipy.stats import sigmaclip

data, _, _ = read("data/raw_r6326e_sw_326-1702_ch01.vdif_7195.fits")
data = data.T
fr = np.sum(data, axis = 1)
fr = (fr - fr.min()) / (fr.max() - fr.min())


frk = fr.copy()
mask = np.abs(kurt) < 20*dev
frk[~mask] = 0.0

data[mask, :] = np.nan
#data /= fr[:, None]


sig = 3
clipped, vmin, vmax = sigmaclip(data, sig, sig)

fig, ax = plt.subplots(1, 2)
ax[0].imshow(data, 
             origin = "lower", 
             cmap = "Greys", 
             aspect = "auto",
             vmin = vmin,
             vmax = vmax,
           )
ax[1].plot(fr)
ax[1].plot(frk)

save_image("plot.pdf")
