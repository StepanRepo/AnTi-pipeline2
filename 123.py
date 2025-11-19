#! venv/bin/python

import numpy as np
import matplotlib.pyplot as plt
from your.formats.psrfits import PsrfitsFile


with PsrfitsFile("./data/010818_0329+54_00adc.psrfits") as psrfits_file:
        # Check observation mode
        primary_hdr = hdul[0].header
        subint_hdu = hdul['SUBINT']
        header = subint_hdu.header

        nsubint = header['NAXIS2']
        nchan   = header['NCHAN']
        nbin    = header['NBIN']
        npol    = header['NPOL']

        print(f"File contains {len(hdul)} subints, {nchan} channels, {nbin} bins, {npol} pols")

        # Read data and metadata
        data     = subint_hdu.data['DATA']           # Shape: (nsubint, nbin, nchan, npol)
        dat_freq = subint_hdu.data['DAT_FREQ']       # Shape: (nsubint, nchan)
        dat_wts  = subint_hdu.data['DAT_WTS']        # Shape: (nsubint, nchan)
        dat_scl  = subint_hdu.data['DAT_SCL']        # Shape: (nsubint, nchan*npol)
        dat_offs = subint_hdu.data['DAT_OFFS']       # Shape: (nsubint, nchan*npol)
        tsubint  = subint_hdu.data['TSUBINT']        # Duration of each subint
        tau      = subint_hdu.header['TBIN']

        for i in range(10):
            print(data[0, i], bin(data[0, i]))


