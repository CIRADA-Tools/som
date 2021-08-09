"""Useful functions for plotting images -- both from fits cutouts 
and the preprocessed ones in the IMG binary file.
"""

import os
import numpy as np
from itertools import product
from astropy import units as u
from astropy.table import Table
from astropy.wcs import WCS
from astropy.visualization import ImageNormalize, LogStretch, AsinhStretch
import pandas as pd
import matplotlib.pyplot as plt

import pyink as pu

# from astropy.coordinates import SkyCoord, search_around_sky, match_coordinates_sky
# import seaborn as sns
# from astroquery.vizier import Vizier
# import vlass_data_loader as vdl


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


def scale_img(radio_file):
    """Scale a radio image for convenient plotting"""
    radio_data = load_radio_fits(radio_file).data
    vmin = np.percentile(radio_data, 0.1)
    vmax = np.percentile(radio_data, 99.9)
    norm = ImageNormalize(stretch=AsinhStretch(0.25), vmin=vmin, vmax=vmax)
    img = norm(radio_data)
    return img


def plot_radio(tile, ax=None):
    """Plot a radio image"""
    vlass_hdu = fits.open(tile)
    data = vlass_hdu[0].data
    rms = pu.rms_estimate(data)
    vmin = 0.25 * rms
    vmax = np.nanmax(data)
    norm = ImageNormalize(stretch=LogStretch(100), vmin=vmin, vmax=vmax)
    # norm = ImageNormalize(stretch=AsinhStretch(0.01), vmin=vmin, vmax=vmax)
    if ax is None:
        plt.imshow(data, norm=norm)
    else:
        ax.imshow(data, norm=norm)


def plot_unwise(tile, ax=None):
    """Plot an IR image"""
    unwise_hdu = fits.open(tile)
    data = unwise_hdu[0].data
    # unorm = ImageNormalize(stretch=AsinhStretch(), data=data)
    unorm = ImageNormalize(stretch=LogStretch(400), data=data)
    if ax is None:
        plt.imshow(data, cmap="plasma", norm=unorm)
    else:
        ax.imshow(data, cmap="plasma", norm=unorm)


def get_transformed_images(imgs, somset, radio_cat):
    bmu = somset.mapping.bmu(idx=radio_cat.index.values)
    transform = somset.transform.data[radio_cat.index.values, ...]
    transform = transform[np.arange(0, len(transform), 1), bmu[:, 0], bmu[:, 1]]
    trimgs = imgs.transform_images(idxs=radio_cat.index.values, transforms=transform)
    return trimgs


def plot_zoo(zoo_sample, cutout_path="zoo_cutouts", img_path="zoo_images"):
    """Prepare a sample of components for a Zooniverse project.
    Original image on the left, preprocessed one on the right.

    Args:
        zoo_sample (pd.DataFrame): The sample to be used.
        cutout_path (str, optional): Path to the input image cutouts.
        img_path (str, optional): Path for the output images to be uploaded to Zooniverse.
    """
    for ind, src in zoo_sample.iterrows():
        plt.close("all")
        radio_file = os.path.join(cutout_path, src.filename)
        fig, axes = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True)

        # Could replace this with `scale_data`
        radio_data = load_radio_fits(radio_file).data
        vmin = np.percentile(radio_data, 0.1)
        vmax = np.percentile(radio_data, 99.9)
        # norm = ImageNormalize(stretch=LogStretch(50), vmin=vmin, vmax=vmax)
        norm = ImageNormalize(stretch=AsinhStretch(0.25), vmin=vmin, vmax=vmax)
        axes[0].imshow(radio_data, norm=norm)
        # plot_radio(radio_file, ax=axes[0])

        axes[1].imshow(imgs.data[ind, 0])

        for ax in axes:
            ax.set_yticks([])
            ax.set_xticks([])
            ax.scatter(75, 75, marker="c", c="r", s=200)

        outfile = src.Component_name.split()[1] + "_zoo.png"
        plt.savefig(os.path.join(img_path, outfile))


def plot_neuron_grid(som, start=(0, 0), dim=5, cross=False):
    """Plot a dim x dim grid of neurons beginning at index `start`.
    """
    fig, axes = plt.subplots(
        dim, dim, figsize=(1.5 * dim, 1.5 * dim), sharex=True, sharey=True,
    )
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout(pad=0, h_pad=0, w_pad=0)

    y1, x1 = start
    for yi in range(y1, y1 + dim):
        for xi in range(x1, x1 + dim):
            ax = axes[yi, xi]
            ax.imshow(som[yi, xi][0], cmap="viridis")
            ax.contour(
                som[yi, xi][1],
                colors="w",
                linewidths=0.5,
                levels=0.05 * np.arange(0.5, 1, 0.1),
            )
            if cross:
                ax.axhline(0.5 * som.header[-1][-2], lw=0.5, c="r", ls="--")
                ax.axvline(0.5 * som.header[-1][-1], lw=0.5, c="r", ls="--")


def grid_plot_prepro(sample: pd.DataFrame, imgs: pu.ImageReader):
    """Create a 10x10 grid of preprocessed images. See example below.

    Arguments:
        sample (pd.DataFrame): The table of 100 images to be plotted.
        imgs (pu.ImageReader): The ImageReader binary
    """
    inds = sample.index
    data = imgs.data[inds, 0]
    # data = imgs.data[inds, ..., 0]
    if len(inds) < 100:
        blank = np.zeros((100 - len(inds), 150, 150))
        data = np.vstack([data, blank])
    data = data.reshape((10, 10, 150, 150))
    data = np.moveaxis(data, 2, 1).reshape((1500, 1500))

    plt.figure(figsize=(20, 20), constrained_layout=True)
    plt.imshow(data)
    plt.xticks([])
    plt.yticks([])

    for i, j in product(range(10), range(10)):
        if j * 10 + i >= len(inds):
            continue
        plt.scatter(150 * i + 75, 150 * j + 75, marker="c", c="r", s=400)

    for i in range(1, 10):
        plt.axvline(150 * i, c="w", ls="-", lw=0.5)
        plt.axhline(150 * i, c="w", ls="-", lw=0.5)


# subset = pd.concat(
#     [
#         sample[sample.bmu == neuron].sample(100)
#         for neuron in product(range(10), range(10))
#     ]
# )
#
# for neuron in product(range(10), range(10)):
#     plt.close("all")
#     grid_plot_prepro(subset[subset.bmu == neuron], imgs)
#     plt.savefig(f"grid_all/preprocessed/neuron_{neuron[0]}-{neuron[1]}.png")


def grid_plot_original(
    sample: pd.DataFrame, imgs: pu.ImageReader, cutout_path: str = "zoo_cutouts"
):
    """Create a 10x10 grid from image cutouts.

    Arguments:
        sample (pd.DataFrame): The table of 100 images to be plotted.
        imgs (pu.ImageReader): The ImageReader binary
        cutout_path (str): The path to the image cutouts
    """
    data = np.array(
        [
            scale_img(os.path.join(cutout_path, src.filename))
            for i, src in sample.iterrows()
        ]
    )
    data = data.reshape((10, 10, 150, 150))
    data = np.moveaxis(data, 2, 1).reshape((1500, 1500))

    plt.figure(figsize=(20, 20), constrained_layout=True)
    plt.imshow(data)
    plt.xticks([])
    plt.yticks([])

    for i, j in product(range(10), range(10)):
        plt.scatter(150 * i + 75, 150 * j + 75, marker="c", c="r", s=400)

    for i in range(1, 10):
        plt.axvline(150 * i, c="w", ls="-", lw=0.5)
        plt.axhline(150 * i, c="w", ls="-", lw=0.5)


def download_first(table, fname_col="FIRST", path="FIRST"):
    for idx, row in table.iterrows():
        print(idx)
        coord = SkyCoord(row["RA"], row["DEC"], unit=u.deg)
        try:
            im = First.get_images(coord, image_size=1.5 * u.arcmin)
            fname = row["filename"].replace("VLASS", "FIRST")
            im.writeto(f"{path}/{fname}")
            subset.loc[idx, fname_col] = fname
        except:
            continue
