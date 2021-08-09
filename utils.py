"""Basic utilities that are helpful in working with the CIRADA catalogues.
"""

import os
import numpy as np
from astropy import units as u
from astropy.table import Table
from astropy.wcs import WCS
import pandas as pd
import matplotlib.pyplot as plt

import pyink as pu


def add_filename(objname, survey="DECaLS-DR8", format="fits"):
    # Create a filename based on the Component_name
    # Begins with everything after the white space
    name = objname.split(" ")[1]
    filename = f"{name}_{survey}.{format}"
    return filename


def load_catalogue(catalog, flag_data=False, flag_SNR=False, pandas=False, **kwargs):
    fmt = "fits" if catalog.endswith("fits") else "csv"
    rcat = Table.read(catalog, format=fmt)

    if flag_data:
        rcat = rcat[rcat["S_Code"] != "E"]
        rcat = rcat[rcat["Duplicate_flag"] < 2]

    if flag_SNR:
        rcat = rcat[rcat["Peak_flux"] >= 5 * rcat["Isl_rms"]]

    rcat["SNR"] = rcat["Total_flux"] / rcat["Isl_rms"]

    if pandas:
        rcat = rcat.to_pandas()
        if fmt == "fits":
            for col in rcat.columns[rcat.dtypes == object]:
                rcat[col] = rcat[col].str.decode("ascii")

    return rcat


def save_table(tab, outfile, keep_bmu=False):
    tab = tab.copy()
    if keep_bmu:
        bmu_y, bmu_x = zip(*tab.bmu)
        tab["bmu_y"] = bmu_y
        tab["bmu_x"] = bmu_x
    del tab["bmu"]
    Table.from_pandas(tab).write(outfile)


def create_wcs(ra, dec, imgsize, pixsize):
    pixsize = u.Quantity(pixsize, u.deg)
    hdr = fits.Header()
    hdr["CRPIX1"] = imgsize // 2 + 0.5
    hdr["CRPIX2"] = imgsize // 2 + 0.5
    hdr["CDELT1"] = -pixsize.value
    hdr["CDELT2"] = pixsize.value
    hdr["PC1_1"] = 1
    hdr["PC2_2"] = 1
    hdr["CRVAL1"] = ra
    hdr["CRVAL2"] = dec
    hdr["CTYPE1"] = "RA---TAN"
    hdr["CTYPE2"] = "DEC--TAN"
    return WCS(hdr)

