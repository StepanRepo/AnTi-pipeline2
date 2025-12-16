#! venv/bin/python

import numpy as np
import matplotlib.pyplot as plt
from myplot import *
import pandas as pd
from plot import read, bin_time

from scipy import signal
from scipy import fft



x = np.fromfile("data/test1.bin")
mf = signal.medfilt(x, int(.1/400e-9)+1)
x_new = x - mf



plt.figure()
plt.plot(x)
plt.plot(x_new)
plt.plot(x - x_new)





save_image("plot.pdf")
