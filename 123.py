#! venv/bin/python

import numpy as np
import matplotlib.pyplot as plt
from myplot import *
import psrchive


arch = psrchive.Archive.load("data/010818_0329+54_00adc.psrfits")
data = arch.get_data()

s = data.shape
#data = data.reshape(*s[::-1]).T

print(data.shape)



plt.figure()
plt.imshow(data[0, 0],
           cmap = "Greys",
           origin = "lower",
           aspect = "auto",
           interpolation = "none")
plt.figure()
plt.plot(np.sum(data[0, 0], axis = 0))
plt.figure()
plt.plot(np.sum(data[0, 0], axis = 1))

save_image("plot.pdf")
