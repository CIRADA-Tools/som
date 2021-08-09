import os, sys
import numpy as np
from astropy import units as u
from astropy.table import Table
import pandas as pd


def vlass_tile(subtiles):
    def tile_info(subtiles, tile):
        t = subtiles[subtiles.Tile == tile]
        return dict(
            Tile=tile,
            min_ra=np.min(t.min_ra),
            max_ra=np.max(t.max_ra),
            min_dec=np.min(t.min_dec),
            max_dec=np.max(t.max_dec),
        )

    st = subtiles[
        [
            "Tile",
            "NAXIS1",
            "NAXIS2",
            "CRVAL1",
            "CDELT1",
            "CRPIX1",
            "CRVAL2",
            "CDELT2",
            "CRPIX2",
        ]
    ].copy()

    st["min_ra"] = st.CRVAL1 + (0 - st.CRPIX1) * np.abs(st.CDELT1)
    st["max_ra"] = st.CRVAL1 + (st.NAXIS1 - st.CRPIX1) * np.abs(st.CDELT1)
    st["min_dec"] = st.CRVAL2 + (0 - st.CRPIX2) * np.abs(st.CDELT2)
    st["max_dec"] = st.CRVAL2 + (st.NAXIS2 - st.CRPIX2) * np.abs(st.CDELT2)

    tiles = [tile_info(st, tile) for tile in np.unique(st.Tile)]
    tiles = pd.DataFrame(tiles)
    return tiles


def load_vlass_catalogue(
    catalog, flag_data=True, flag_SNR=False, pandas=False, complex=False, **kwargs
):
    fmt = "fits" if catalog.endswith("fits") else "csv"
    rcat = Table.read(catalog, format=fmt)

    if flag_data:
        rcat = rcat[rcat["S_Code"] != "E"]
        rcat = rcat[rcat["Duplicate_flag"] < 2]

    if flag_SNR:
        rcat = rcat[rcat["Peak_flux"] >= 5 * rcat["Isl_rms"]]

    rcat["SNR"] = rcat["Total_flux"] / rcat["Isl_rms"]

    if complex:
        rcat = complex_vlass(rcat, **kwargs)

    if pandas:
        rcat = rcat.to_pandas()
        if fmt == "fits":
            for col in rcat.columns[rcat.dtypes == object]:
                rcat[col] = rcat[col].str.decode("ascii")

    return rcat


def complex_vlass(df, NN_dist=72, SNR_min=None):
    """Algorithm to select complex VLASS components.

    Args:
        df ([pandas.DataFrame]): Table of VLASS components.
        NN_dist ([float]], optional): Maximum distance (in arcsec) to the nearest component to be considered complex. Defaults to 72.
        SNR_min ([float]], optional): Minimum signal-to-noise of the component. Defaults to None.

    Returns:
        pandas.DataFrame: Subset of the input DataFrame containing only the complex components.
    """
    mask = (df["S_Code"] == "S") & (df["NN_dist"] < NN_dist)
    mask |= df["S_Code"] == "M"
    mask |= df["S_Code"] == "C"
    df = df[mask]

    if SNR_min is not None:
        df = df[df["SNR"] >= SNR_min]

    return df
