"""Tools that use the catalogues only."""

import os, sys
import json
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from astropy.table import Table
from astropy.visualization import AsinhStretch, ImageNormalize, LogStretch
import pyink as pu
import legacy_survey_cutout_fetcher as lscf
import crosshair


def trim_neuron(neuron, img_shape=None):
    som_shape = neuron.shape[0]
    if img_shape is None:
        img_shape = int(np.floor(som_shape / np.sqrt(2)))
    b1 = (som_shape - img_shape) // 2
    b2 = b1 + img_shape
    return neuron[b1:b2, b1:b2]


def add_filename(objname, survey="DECaLS-DR8", format="fits"):
    # Take Julian coords of name to eliminate white space - eliminate prefix
    name = objname.split(" ")[1]
    filename = f"{name}_{survey}.{format}"
    return filename


def plot_radio(tile, ax=None):
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
    unwise_hdu = fits.open(tile)
    data = unwise_hdu[0].data
    # unorm = ImageNormalize(stretch=AsinhStretch(), data=data)
    unorm = ImageNormalize(stretch=LogStretch(400), data=data)
    if ax is None:
        plt.imshow(data, cmap="plasma", norm=unorm)
    else:
        ax.imshow(data, cmap="plasma", norm=unorm)


def plot_source(source, img_size=300, file_path="images", use_wcs=False):
    vlass_tile = os.path.join(
        file_path, add_filename(source["Source_name"], survey="VLASS")
    )
    unwise_tile = os.path.join(
        file_path, add_filename(source["Source_name"], survey="unWISE_NEO4")
    )

    # ra = json.loads(source["RA_components"])[0]
    # dec = json.loads(source["DEC_components"])[0]
    ra = source["RA_source"]
    dec = source["DEC_source"]

    lscf.grab_cutout(
        ra, dec, vlass_tile, survey="vlass1.2", imgsize_arcmin=3.0, imgsize_pix=300,
    )

    lscf.grab_cutout(
        ra,
        dec,
        unwise_tile,
        survey="unwise-neo4",
        imgsize_arcmin=3.0,
        imgsize_pix=img_size,
        extra_processing=lscf.process_unwise,
        extra_proc_kwds={"band": "w1"},
    )

    # Plot fits images
    if use_wcs:
        wcs = WCS(vlass_tile)
        fig, axes = plt.subplots(
            1,
            2,
            figsize=(10, 4),
            sharex=True,
            sharey=True,
            subplot_kw={"projection": wcs},
        )
    else:
        fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True,)

    plot_radio(vlass_tile, ax=axes[0])
    plot_unwise(unwise_tile, ax=axes[1])

    for ax in axes:
        ax.scatter(img_size / 2, img_size / 2, marker="c", c="w", s=200)

    if use_wcs:
        ra = json.loads(src.RA_components)
        dec = json.loads(src.DEC_components)
        axes[0].scatter(
            ra, dec, marker=".", c="r", s=100, transform=axes[0].get_transform("world")
        )


def prepare_zoo_image(src_cat, annotation, src_id, file_path="images"):
    # Add channel masks
    print(f"Source ID: {src_id}")
    src = src_cat.iloc[src_id]
    neuron = src["Best_neuron"]
    # neuron = eval(src["Best_neuron"])
    ant = annotation.results[neuron]
    labels = dict(ant.labels)
    print(f"Neuron: {neuron}")

    plot_source(src, file_path=file_path)
    fig = plt.gcf()
    axes = fig.axes

    radio_mask = pu.valid_region(
        ant.filters[0],
        filter_includes=labels["Related Radio"],
        filter_excludes=labels["Sidelobe"],
    ).astype(np.float32)
    radio_mask = pu.pink_spatial_transform(
        radio_mask, (src["Flip"], src["Angle"]), reverse=True
    )
    radio_mask = trim_neuron(radio_mask, 300)
    radio_mask = radio_mask >= 0.1
    axes[0].contour(radio_mask, levels=[1], colors="w")

    ir_mask = pu.valid_region(
        ant.filters[1],
        filter_includes=labels["IR Host"],
        filter_excludes=labels["Sidelobe"],
    ).astype(np.float32)
    ir_mask = pu.pink_spatial_transform(
        ir_mask, (src["Flip"], src["Angle"]), reverse=True
    )
    ir_mask = trim_neuron(ir_mask, 300)
    ir_mask = ir_mask >= 0.1
    axes[1].contour(ir_mask, levels=[1], colors="w")
