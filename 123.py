#! venv/bin/python

import numpy as np
import matplotlib.pyplot as plt
from myplot import *
import pandas as pd

data = pd.read_csv("data/Crab_pulse/r6326e_bv_326-1702_ch01.vdif_7183.vdif.csv", comment = "#", sep = ";")

pwr = data.iloc[:, 2]

plt.figure()
plt.hist(pwr, bins = 100)

save_image("plot.pdf")
