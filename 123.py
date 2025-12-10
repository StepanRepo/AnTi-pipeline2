#! venv/bin/python

import numpy as np
import matplotlib.pyplot as plt
from myplot import *
import pandas as pd
from plot import read, bin_time

import numpy as np


import numpy as np

def running_mean_subtract(x, window, center=True):
    """
    Subtract running mean from a 1D array.

    Parameters
    ----------
    x : array_like
        Input 1D array.
    window : int
        Window size (must be positive and <= len(x)).
    center : bool, default=True
        If True, center the window (include future points).
        If False, use trailing window (past points only).

    Returns
    -------
    y : ndarray
        Array with running mean subtracted.
        Same length as input. Edge values use partial windows.
    """
    x = np.asarray(x)
    n = len(x)
    if window <= 0 or window > n:
        raise ValueError("window must be > 0 and <= len(x)")

    # Force window to be odd when centering for true symmetry
    if center and window % 2 == 0:
        window += 1
        if window > n:
            window -= 2  # revert if too big

    cumsum = np.cumsum(np.concatenate([[0], x]))

    if not center:
        # Trailing window: mean at i uses x[i-window+1:i+1]
        means = (cumsum[window:] - cumsum[:-window]) / window
        # Pad left edge with partial means
        left_means = cumsum[1:window] / np.arange(1, window)
        running_mean = np.concatenate([left_means, means])
    else:
        # Centered window: mean at i uses x[i-half:i+half+1]
        half = window // 2
        # Full window means
        means = (cumsum[window:] - cumsum[:-window]) / window
        # Left edge (0 to half-1)
        left_idx = np.arange(1, half + 1)
        left_means = cumsum[1:half + 1] / left_idx
        # Right edge (n-half to n-1)
        right_idx = np.arange(half, 0, -1)
        right_means = (cumsum[-1] - cumsum[-half - 1:-1]) / right_idx
        running_mean = np.concatenate([left_means, means, right_means])

    return running_mean


x = np.fromfile("data/test1.bin")
sub = running_mean_subtract(x, 100000)
print((1e5 * 1000*1e-9/1024)*1e6)



x = x.reshape(-1, 1)
x = bin_time(x, 2**14)
x = x[:, 0]

sub = sub.reshape(-1, 1)
sub = bin_time(sub, 2**14)
sub = sub[:len(x), 0]


plt.figure()
plt.plot(x)
plt.plot(sub)
plt.xlim(900, 1100)

plt.figure()
plt.plot(x-sub)







save_image("plot.pdf")
