#! venv/bin/python

import numpy as np
import matplotlib.pyplot as plt
from myplot import *

def bin_time(data, bin_size):
    T, F = data.shape
    n_bins = T // bin_size
    trimmed = data[:n_bins * bin_size]
    reshaped = trimmed.reshape(n_bins, bin_size, F)
    
    return reshaped.sum(axis=1)

data = np.fromfile("./dyn_1804289383.bin", dtype = np.float64)
data = data.reshape(-1, 512)
data = bin_time(data, 16).T

print(np.any(np.isnan(data)))
print(np.mean(data))

plt.figure()
plt.imshow(np.log(data),
           origin = "lower",
           aspect = "auto",
           interpolation = "none",
           cmap = "Greys")

plt.figure()
plt.plot(np.sum(data, axis = 0))

plt.figure()
plt.plot(np.sum(data, axis = 1))


save_image("plot.pdf")
