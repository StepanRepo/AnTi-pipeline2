import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm
import numpy as np
import os
from pathlib import Path


def save_image(filename, tight = False, *args, **kwargs):
	# PdfPages is a wrapper around pdf
	# file so there is no clash and create
	# files with no error.
    with PdfPages(filename) as p:

        # get_fignums Return list of existing
        # figure numbers
        fig_nums = plt.get_fignums()
        figs = [plt.figure(n) for n in fig_nums]

        if tight:
            bbox_inches = "tight"
        else:
            bbox_inches = None

        # iterating over the numbers in list
        for fig in tqdm(figs):

            # and saving the files
            fig.savefig(p, format='pdf', 
                        bbox_inches=bbox_inches, 
                        *args,
                        **kwargs)

            plt.close(fig)




def save_eps(dirname, tight=False, *args, **kwargs):
    """
    Save all open figures as EPS vector files in the specified directory.
    Creates the directory if it doesn't exist.

    Parameters:
    - dirname: Directory where EPS files will be saved
    - tight: If True, uses bbox_inches='tight' for tight layout
    - *args, **kwargs: Additional arguments passed to savefig
    """
    # Create directory if it doesn't exist
    Path(dirname).mkdir(parents=True, exist_ok=True)

    # Get all open figures
    fig_nums = plt.get_fignums()
    figs = [plt.figure(n) for n in fig_nums]

    if tight:
        bbox_inches = "tight"
    else:
        bbox_inches = None


    # Save each figure as EPS
    for i, fig in enumerate(tqdm(figs)):
        # Generate filename with sequential numbering
        eps_path = os.path.join(dirname, f"figure_{i+1}.eps")

        fig.savefig(eps_path, format = "eps",
                   bbox_inches=bbox_inches,
                   *args,
                   **kwargs)

        plt.close(fig)

def set_tex():

    plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.sans-serif": "serif",
    "font.size"   : 12
    })
    

def fwhm(x, y):
    half = y.max() / 2.0

    # find the first and the last enterance of +1
    pos = np.where(y > half)[0]

    left, right = pos[0], pos[-1]

    f = np.polynomial.polynomial.Polynomial.fit(
            x[left-1:left+1], y[left-1:left+1]-half, 1)

    ll = f.roots()[0]

    g = np.polynomial.polynomial.Polynomial.fit(
            x[right:right+2], y[right:right+2]-half, 1)

    rr = g.roots()[0]

    return rr-ll

def a4(width = 1, heigh = 1, vpad = 4, hpad = 4, horizontal = False):
    if horizontal:
        return heigh*(29.7-vpad)/2.54, width*(21-hpad)/2.54
    else:
        return width*(21-hpad)/2.54, heigh*(29.7-vpad)/2.54

