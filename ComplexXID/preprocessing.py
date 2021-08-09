"""Preprocess a sample of VLASS components with corresponding unWISE
images. Outputs an image binary file suitable for PINK.
"""

import os
import pickle
from multiprocessing import Pool, cpu_count
from glob import glob
from collections import Counter
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import astropy.units as u
from tqdm import tqdm as tqdm
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from astropy.modeling.models import Gaussian2D
from astropy.coordinates import SkyCoord, search_around_sky
from astropy.convolution import convolve, Gaussian2DKernel
from reproject import reproject_interp
from photutils.datasets import make_gaussian_sources_image

import pyink as pu

# from vos import Client
# from legacy_survey_cutout_fetcher import grab_vlass_unwise_cutouts

import logging

# logger = logging.getLogger("pyink.binwrap")
# logger.addHandler(logging.StreamHandler().setLevel(logging.DEBUG))

np.seterr(all="ignore")

import warnings
from astropy.utils.exceptions import AstropyWarning

warnings.filterwarnings("ignore", category=AstropyWarning, append=True)


def add_filename(objname, survey="DECaLS-DR8", format="fits"):
    # Just take Julian coords of name to eliminate white space - eliminate prefix
    name = objname.split(" ")[1]
    filename = f"{name}_{survey}.{format}"
    return filename


def radio_preprocess(data, lower=3, min_isl_pix=5):
    data[np.isnan(data)] = 0
    remove_zeros = data[data != 0]  # data.flatten()
    noise = pu.rms_estimate(remove_zeros, mode="mad", clip_rounds=2)

    # empty = ~np.isfinite(data)
    # data[empty] = np.random.normal(0, noise, np.sum(empty))

    img_scale = np.zeros_like(data)
    for i, mask in enumerate(
        pu.island_segmentation(data - np.median(remove_zeros), 3 * noise)
    ):
        if np.sum(mask) <= min_isl_pix:
            continue

        img_scale[mask] = pu.minmax(np.log10(data[mask]))

    img_scale[~np.isfinite(img_scale)] = 0

    return img_scale.astype(np.float32)


def ir_preprocess(data):
    med = np.median(data.flatten())
    mad = np.median(np.abs(data.flatten() - med))

    noise = med + 3 * mad

    img_scale = np.zeros_like(data)
    for i, mask in enumerate(pu.island_segmentation(data, noise)):
        if np.sum(mask) <= 5:
            continue
        img_scale[mask] = pu.minmax(np.log10(data[mask]))

    img_scale[~np.isfinite(img_scale)] = 0

    return img_scale.astype(np.float32)


def preprocess_hull(radio_data, ir_data, sigma=10, threshold=0.05):
    radio_pp = radio_preprocess(radio_data)
    ir_pp = ir_preprocess(ir_data)
    hull = pu.convex_hull_smoothed(radio_pp, sigma, threshold)
    ir_pp = ir_pp * hull
    return radio_pp, ir_pp


# def new_masking(data, sigma=10):
#     sigma = 10
#     g2d_amp = 1 / (2 * np.pi * sigma * sigma)
#     hull = pu.convex_hull(data)
#     mask = convolve(hull.astype(np.float32), Gaussian2DKernel(sigma)) > 0.01
#     gmask = convolve(data, Gaussian2DKernel(sigma)) > 0.5 * g2d_amp
#     mask = pu.convex_hull_smoothed(data, sigma, 0.05)


def comp_img(comps, posns, img_shape, pix_size=0.6):
    src_tab = Table()
    src_tab["amplitude"] = [1] * len(comps)
    src_tab["x_mean"] = posns[0]
    src_tab["y_mean"] = posns[1]
    src_tab["x_stddev"] = comps["Maj"] / pix_size / 2.355
    src_tab["y_stddev"] = comps["Min"] / pix_size / 2.355
    src_tab["theta"] = 90 - comps["PA"]
    return make_gaussian_sources_image(img_shape, src_tab)


def comp_filter(comps_in_img, hdr, threshold=0.01):
    """Create a filter (mask) based on the positions of real components.
    Connect the components by a convex hull"""
    wcs = WCS(hdr)
    x, y = wcs.all_world2pix(comps_in_img["RA"], comps_in_img["DEC"], 0)

    img_shape = (hdr["NAXIS2"], hdr["NAXIS1"])
    pix_size = hdr["CD2_2"] * 3600
    img = comp_img(comps_in_img, (x, y), img_shape, pix_size)
    hull = pu.convex_hull(img, threshold=threshold)
    return hull


# def comp_filter(all_comps, ra, dec, hdr, threshold=0.01):
#     """Create a filter (mask) based on the positions of real components.
#     Connect the components by a convex hull"""
#     coord = SkyCoord(ra, dec, unit=u.deg)
#     all_coords = SkyCoord(
#         np.array(all_comps["RA"]), np.array(all_comps["DEC"]), unit=u.deg
#     )
#     max_sep = 0.5 * hdr["NAXIS2"] * hdr["CD2_2"] * 60 * u.arcmin
#     coord_inds = np.where(coord.separation(all_coords) <= max_sep)
#     comps_in_img = all_comps[coord_inds]

#     wcs = WCS(hdr)
#     x, y = wcs.all_world2pix(comps_in_img["RA"], comps_in_img["DEC"], 0)

#     img_shape = (hdr["NAXIS2"], hdr["NAXIS1"])
#     pix_size = hdr["CD2_2"] * 3600

#     Y, X = np.mgrid[: hdr["NAXIS1"], : hdr["NAXIS2"]]
#     mod = np.zeros(X.shape)
#     for xi, yi, comp in zip(x, y, comps_in_img):
#         major = comp["Maj"] / pix_size
#         minor = comp["Min"] / pix_size
#         pa = 90 - comp["PA"]
#         mod += Gaussian2D(
#             x_mean=xi,
#             y_mean=yi,
#             x_stddev=major / 2.355,
#             y_stddev=minor / 2.355,
#             theta=pa,
#         )(X, Y)
#     hull = pu.convex_hull(mod, threshold=threshold)
#     return hull


def check_radio_data(data, idx):
    if np.sum(~np.isfinite(data)) > 0:
        print(f"Skipping index {idx} due to too many NaN")
        return False
    if np.max(data) <= 0:
        print(f"Skipping index {idx} due to no positive values")
        return False
    return True


def check_ir_data(data, idx):
    if np.sum(~np.isfinite(data)) > 0:
        print(f"Skipping index {idx} due to too many NaN in IR channel")
        return False
    if np.max(data) <= 0:
        print(f"Skipping index {idx} due to no positive values in IR channel")
        return False
    return True


def load_fits(filename, ext=0):
    hdulist = fits.open(filename)
    d = hdulist[ext]
    return d


def load_radio_fits(filename, ext=0):
    hdu = load_fits(filename, ext=ext)
    wcs = WCS(hdu.header).celestial
    hdu.data = np.squeeze(hdu.data)
    hdu.header = wcs.to_header()
    return hdu


def recenter_regrid(hdu, ra, dec, img_size, reproj_hdr=None):
    # Recentering the reference pixel
    if reproj_hdr is None:
        reproj_hdr = hdu.header.copy()
    reproj_hdr["CRVAL1"] = ra
    reproj_hdr["CRVAL2"] = dec
    reproj_hdr["CRPIX1"] = img_size[0] // 2 + 0.5
    reproj_hdr["CRPIX2"] = img_size[1] // 2 + 0.5
    reproj_wcs = WCS(reproj_hdr).celestial

    reproj_data, reproj_footprint = reproject_interp(
        hdu, reproj_wcs, shape_out=img_size
    )
    return reproj_data


def vlass_preprocessing(
    idx,
    tab,
    ir=True,
    img_size=(2, 150, 150),
    radio_path="",
    ir_path="",
    radio_fname_col="filename",
    ir_fname_col="ir_filename",
    all_comps=None,
    comp_idx_map=None,
    ir_weight=0.05,
):
    """Preprocess a single VLASS image. 
    """

    radio_file = tab.loc[idx][radio_fname_col]
    radio_file = os.path.join(radio_path, radio_file)
    ra = tab.loc[idx]["RA"]
    dec = tab.loc[idx]["DEC"]

    try:
        radio_hdu = load_radio_fits(radio_file)
    except:
        print(f"Failed to open image {radio_file}")
        return None

    if radio_hdu.header["NAXIS"] == 0:
        print(f"Empty image: {radio_file}")
        return None

    if not ir:
        # Preprocess the radio only
        radio_data = recenter_regrid(radio_hdu, ra, dec, img_size[1:])
        radio_prepro = radio_preprocess(radio_data)
        if not check_radio_data(radio_prepro, idx):
            return None
        return (idx, np.array([radio_prepro]))

    # Process the IR data
    ir_file = tab.loc[idx][ir_fname_col]
    ir_file = os.path.join(ir_path, ir_file)

    try:
        ir_hdu = load_fits(ir_file)
    except:
        print(f"Failed to open image {ir_file}")
        return None

    reproj_hdr = ir_hdu.header.copy()
    radio_data = recenter_regrid(radio_hdu, ra, dec, img_size[1:], reproj_hdr)
    ir_data = recenter_regrid(ir_hdu, ra, dec, img_size[1:], reproj_hdr)

    radio_prepro = radio_preprocess(radio_data, min_isl_pix=15)
    if not check_radio_data(radio_prepro, idx):
        return None

    ir_prepro = ir_preprocess(ir_data)
    if not check_ir_data(ir_prepro, idx):
        return None
    # ir_prepro *= pu.convex_hull_smoothed(e_new_data, 15, 0.05)

    if all_comps is not None:
        idx1, idx2 = comp_idx_map
        comps_in_img = all_comps[idx2[idx1 == idx]]
        ir_prepro *= comp_filter(comps_in_img, reproj_hdr)

    return (idx, np.array((radio_prepro * (1 - ir_weight), ir_prepro * ir_weight)))


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(
        description="Preprocess a sample of images and output a SOM image binary."
    )
    parser.add_argument(
        dest="sample", help="Data sample to use", type=str,
    )
    parser.add_argument(
        dest="outfile", help="Name for the output image binary file", type=str,
    )
    parser.add_argument(
        "-r",
        "--radio_survey",
        dest="radio_survey",
        help="The radio survey to be processed",
        default="VLASS",
        type=str,
    )
    parser.add_argument(
        "-i",
        "--ir_survey",
        dest="ir_survey",
        help="The optical/IR survey to be processed",
        default="unWISE-NEO4",
        type=str,
    )
    parser.add_argument(
        "-p",
        "--image_path",
        dest="img_path",
        help="Path to the directory containing the raw input images",
        default="images",
        type=str,
    )
    parser.add_argument(
        "-w",
        "--ir_weight",
        dest="ir_weight",
        help="The weighting to apply to the optical/IR channel",
        default=0.05,
        type=float,
    )
    parser.add_argument(
        "-t",
        "--threads",
        dest="threads",
        help="Number of threads to use for multiprocessing",
        default=cpu_count(),
        type=int,
    )
    parser.add_argument(
        "-s",
        "--size",
        dest="img_size",
        help="Number of pixels in the input image.",
        default=300,
        type=int,
    )
    parser.add_argument(
        "-c",
        "--comp_cat",
        dest="comp_cat",
        help="The radio component catalogue to be used for filtering",
        default=None,
        type=str,
    )
    args = parser.parse_args()
    return args


def main(
    sample,
    outfile,
    img_size,
    img_path,
    threads=None,
    parallel=True,
    all_comps=None,
    seplimit=90,
    **kwargs,
):
    kwargs.update(
        dict(
            ir=True,
            img_size=img_size,
            radio_path=img_path,
            ir_path=img_path,
            all_comps=all_comps,
        )
    )

    if all_comps is not None:
        seplimit = u.Quantity(seplimit, u.arcsec)
        coords = SkyCoord(np.array(sample["RA"]), np.array(sample["DEC"]), unit=u.deg)
        all_coords = SkyCoord(
            np.array(all_comps["RA"]), np.array(all_comps["DEC"]), unit=u.deg
        )
        idx1, idx2, sep, dist = search_around_sky(coords, all_coords, seplimit=seplimit)
        kwargs.update(comp_idx_map=(idx1, idx2))

    with pu.ImageWriter(outfile, 0, img_size, clobber=True) as pk_img:
        if not parallel:
            for idx in tqdm(sample.index):
                out = vlass_preprocessing(idx, sample, **kwargs)
                if out is not None:
                    pk_img.add(out[1], attributes=out[0])
        else:
            if threads is None:
                threads = cpu_count()
            pool = Pool(processes=threads)
            results = [
                pool.apply_async(vlass_preprocessing, args=(idx, sample), kwds=kwargs)
                for idx in sample.index
            ]
            for res in tqdm(results):
                out = res.get()
                if out is not None:
                    pk_img.add(out[1], attributes=out[0])


if __name__ == "__main__":
    args = parse_args()

    # Feed DataFrame directly into preprocessing routine
    # Requires filename columns
    sample = pd.read_csv(args.sample)

    if "filename" not in sample:
        sample["filename"] = sample["Component_name"].apply(
            add_filename, survey=args.radio_survey
        )
        sample["ir_filename"] = sample["Component_name"].apply(
            add_filename, survey=args.ir_survey
        )

    all_comps = None
    if args.comp_cat is not None:
        all_comps = Table.read(args.comp_cat)

    parallel = all_comps is None

    main(
        sample,
        args.outfile,
        img_size=(2, args.img_size, args.img_size),
        img_path=args.img_path,
        threads=args.threads,
        parallel=parallel,
        all_comps=all_comps,
        seplimit=90 * u.arcsec,
        ir_weight=args.ir_weight,
    )

