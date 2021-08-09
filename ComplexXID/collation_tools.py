"""Functions that are aimed to inspect the results of collation (multiple
components combined into a single source).

This is largely deprecated, as we did not complete a source catalogue 
using the SOM.
"""

import os, sys
import json
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from astropy.table import Table
from astropy.visualization import AsinhStretch, ImageNormalize, LogStretch

# import legacy_survey_cutout_fetcher as lscf

import pyink as pu
from plotting_tools import plot_radio, plot_unwise
import crosshair


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
    radio_mask = pu.trim_neuron(radio_mask, 300)
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
    ir_mask = pu.trim_neuron(ir_mask, 300)
    ir_mask = ir_mask >= 0.1
    axes[1].contour(ir_mask, levels=[1], colors="w")


def source_from_catalogue(
    src_cat,
    radio_cat,
    imbin_path,
    som=None,
    show_nearby=False,
    show_bmu=False,
    idx=None,
):
    if idx is None:
        # idx = np.random.randint(len(src_cat))
        idx = src_cat[src_cat.N_components > 1].sample(1).index[0]
    src = src_cat.loc[idx]

    comp_names = src.Component_names.split(";")
    tile_id = radio_cat[radio_cat.Component_name == comp_names[0]].Tile.iloc[0]
    radio_tile = radio_cat[radio_cat.Tile == tile_id].reset_index(drop=True)
    comps = radio_tile[radio_tile["Component_name"].isin(comp_names)]

    radio_ra = np.array(src.RA_components.split(";"), dtype=float)
    radio_dec = np.array(src.DEC_components.split(";"), dtype=float)
    radio_pos = SkyCoord(radio_ra, radio_dec, unit=u.deg)

    if src.N_host_candidates > 0:
        ir_ra = np.array(src.RA_host_candidates.split(";"), dtype=float)
        ir_dec = np.array(src.DEC_host_candidates.split(";"), dtype=float)
        ir_pos = SkyCoord(ir_ra, ir_dec, unit=u.deg)

    img_file, map_file, trans_file = binary_names(tile_id, imbin_path)
    imgs = pu.ImageReader(img_file)
    mapping = pu.Mapping(map_file)
    transform = pu.Transform(trans_file)

    if som is not None:
        somset = pu.SOMSet(som, mapping, transform)

    img_idx = comps.index[0]
    comp = comps.loc[img_idx]
    npix = imgs.data.shape[-1]
    wcs = create_wcs(comp.RA, comp.DEC, npix, 3 * u.arcmin / npix)

    if show_bmu:
        plot_image(
            imgs,
            img_idx,
            somset=somset,
            wcs=wcs,
            transform_neuron=True,
            show_bmu=show_bmu,
        )
    else:
        plot_image(imgs, img_idx, wcs=wcs, show_bmu=False)
    axes = plt.gcf().axes

    if show_nearby:
        posn = SkyCoord(comp.RA, comp.DEC, unit=u.deg)
        coords = SkyCoord(radio_cat.RA, radio_cat.DEC, unit=u.deg)
        nearby = coords[posn.separation(coords) < 1.5 * u.arcmin]
        for ax in plt.gcf().axes:
            ax.scatter(
                nearby.RA, nearby.DEC, c="w", transform=ax.get_transform("world")
            )

    # for ax in plt.gcf().axes:
    #     ax.scatter(comps.RA, comps.DEC, c="r", transform=ax.get_transform("world"))

    axes[0].scatter(
        radio_pos.ra, radio_pos.dec, c="r", transform=axes[0].get_transform("world")
    )
    if src.N_host_candidates > 0:
        axes[1].scatter(
            ir_pos.ra, ir_pos.dec, c="w", transform=axes[1].get_transform("world")
        )

    plt.suptitle(f"Source ID: {idx}")


def plot_radio_source(imgs, somset, radio_cat, idx, components, ir_srcs, pixsize):
    # Plot the image with the positions of the related radio components
    # and candidate IR hosts
    pixsize = u.Quantity(pixsize, u.arcsec)
    src = radio_cat.loc[idx]
    plot_image(imgs, somset=somset, show_bmu=True, transform_neuron=True, idx=src_idx)
    fig = plt.gcf()
    wcs = create_wcs(src.RA, src.DEC, imgs.data.shape[2], pixsize)

    # Plot radio components
    x, y = wcs.all_world2pix(components.ra, components.dec, 0)
    for ax in plt.gcf().axes:
        ax.scatter(x, y, color="red")

    # Plot IR sources
    x, y = wcs.all_world2pix(ir_srcs.ra, ir_srcs.dec, 0)
    for ax in plt.gcf().axes:
        ax.scatter(x, y, color="white")


def inspect_components(imgs, somset, positions, matches, idx, pixsize=0.36):
    # Transform an image all nearby components to the BMU frame.
    # Check that the components align with the BMU signal.
    # bmu_keys = somset.mapping.bmu(return_idx=True, squeeze=True)
    # bz, by, bx = bmu_keys.T

    pixsize = u.Quantity(pixsize, u.arcsec)
    bmu = somset.mapping.bmu(idx)
    by, bx = bmu

    radio_positions, ir_positions = positions
    radio_matches, ir_matches = matches

    center_pos = radio_positions[idx]

    trans_key = (idx, *bmu)
    flip, angle = somset.transform.data[trans_key]
    src_transform = (flip, angle)

    src_mask = idx == radio_matches[0]
    src_matches = radio_matches[1][src_mask]

    spatial_radio_pos = pu.CoordinateTransformer(
        center_pos, radio_positions[src_matches], src_transform, pixel_scale=pixsize,
    )

    src_img = imgs.data[idx, 0].copy()
    transform_img = pu.pink_spatial_transform(src_img, src_transform)
    cen_pix = src_img.shape[0] // 2

    fig, axes = plt.subplots(
        2, 3, figsize=(10, 7), sharex=True, sharey=True, constrained_layout=True
    )

    axes[0, 0].imshow(src_img)
    axes[0, 1].imshow(transform_img)
    axes[0, 2].imshow(trim_neuron(somset.som[bmu][0], src_img.shape[0]))

    axes[0, 0].plot(
        spatial_radio_pos.coords["offsets-pixel"][0].value + cen_pix,
        spatial_radio_pos.coords["offsets-pixel"][1].value + cen_pix,
        "ro",
    )

    axes[0, 1].plot(
        spatial_radio_pos.coords["offsets-neuron"][0].value + cen_pix,
        spatial_radio_pos.coords["offsets-neuron"][1].value + cen_pix,
        "ro",
    )

    axes[0, 2].plot(
        spatial_radio_pos.coords["offsets-neuron"][0].value + cen_pix,
        spatial_radio_pos.coords["offsets-neuron"][1].value + cen_pix,
        "ro",
    )

    # Plotting the IR channel
    src_img = imgs.data[idx, 1].copy()
    transform_img = pu.pink_spatial_transform(src_img, src_transform)

    src_mask = idx == ir_matches[0]
    src_matches = ir_matches[1][src_mask]

    spatial_ir_pos = pu.CoordinateTransformer(
        center_pos, ir_positions[src_matches], src_transform, pixel_scale=pixsize,
    )

    axes[1, 0].imshow(src_img)
    axes[1, 1].imshow(transform_img)
    axes[1, 2].imshow(trim_neuron(somset.som[bmu][1], src_img.shape[0]))

    axes[1, 0].plot(
        spatial_ir_pos.coords["offsets-pixel"][0].value + cen_pix,
        spatial_ir_pos.coords["offsets-pixel"][1].value + cen_pix,
        "ro",
    )

    axes[1, 1].plot(
        spatial_ir_pos.coords["offsets-neuron"][0].value + cen_pix,
        spatial_ir_pos.coords["offsets-neuron"][1].value + cen_pix,
        "ro",
    )

    axes[1, 2].plot(
        spatial_ir_pos.coords["offsets-neuron"][0].value + cen_pix,
        spatial_ir_pos.coords["offsets-neuron"][1].value + cen_pix,
        "ro",
    )

    axes[0, 0].set(title=f"Original Image")
    axes[0, 1].set(title=f"flip, rot: {src_transform}")
    axes[0, 2].set(title=f"Neuron: {bmu}")


"""
def accumulate(path, k, bmu_keys, som, sky_positions, sky_matches, close=True):
    # Average all images that match to a specified neuron.
    bz, by, bx = bmu_keys.T
    mask = (k[0] == by) & (k[1] == bx)
    argmask = np.argwhere(mask)

    if np.sum(mask) == 0:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    ax1.imshow(np.sqrt(som[k][0]), cmap="Greys")
    ax2.imshow(np.sqrt(som[k][0]), cmap="Greys")

    ax1.set(title=f"{k} - {np.sum(mask)}")

    divider = make_axes_locatable(ax2)
    axHistx = divider.append_axes("top", size="23%", pad="2%", sharex=ax2)
    axHisty = divider.append_axes("right", size="23%", pad="2%", sharey=ax2)

    spatial_pos = []

    for j, src in enumerate(argmask):
        src = src[0]

        center_pos = sky_positions[src]
        src_mask = src == sky_matches[0]
        src_matches = sky_matches[1][src_mask]

        src_transform = transform.data[(src, *k)]
        flip, angle = src_transform

        spatial_emu_pos = pu.CoordinateTransformer(
            center_pos,
            sky_positions[src_matches],
            src_transform,
            pixel_scale=2 * u.arcsecond,
        )

        spatial_pos.append(spatial_emu_pos)

    px = (
        np.concatenate(
            [pos.coords["offsets-neuron"][0].value for pos in spatial_pos]
        ).flatten()
        + 213 / 2
    )
    py = (
        np.concatenate(
            [pos.coords["offsets-neuron"][1].value for pos in spatial_pos]
        ).flatten()
        + 213 / 2
    )

    ax2.plot(px, py, "ro", markersize=2.0, alpha=0.5)

    axHistx.hist(px, bins=30, density=True, histtype="stepfilled", alpha=0.7)
    axHisty.hist(
        py,
        bins=30,
        density=True,
        orientation="horizontal",
        histtype="stepfilled",
        alpha=0.7,
    )
    axHistx.set(ylabel="Density")
    axHisty.set(xlabel="Density")

    # no labels
    plt.setp(axHistx.get_xticklabels(), visible=False)
    plt.setp(axHisty.get_yticklabels(), visible=False)

    fig.tight_layout()
    fig.savefig(f"{path.Cluster}/{k}_cluster.png")
"""


def plot_filter_idx(imgs, filters, idx):
    fig = plt.figure(figsize=(10, 10))
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222, sharex=ax1, sharey=ax1)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224, sharex=ax3, sharey=ax3)

    filter_size = filters.filters[0][idx].neuron.filters[0].shape
    img_size = imgs.data[idx, 0].shape

    filters.filters[0][idx].plot(axes=ax1)
    filters.filters[1][idx].plot(axes=ax2)

    ax3.imshow(imgs.data[idx, 0])
    ax4.imshow(imgs.data[idx, 1])

    for ax in [ax1, ax2, ax3, ax4]:
        ax.grid(which="major", axis="both", color="white", alpha=0.6)
    for ax in [ax1, ax2]:
        ax.axvline(filter_size[0] / 2, color="white", ls="--", alpha=0.5)
        ax.axhline(filter_size[1] / 2, color="white", ls="--", alpha=0.5)
    for ax in [ax3, ax4]:
        ax.axvline(img_size[0] / 2, color="white", ls="--", alpha=0.5)
        ax.axhline(img_size[1] / 2, color="white", ls="--", alpha=0.5)


def get_src_img(pos, survey, angular=None, level=0, **kwargs):
    if level > 5:
        print("Failing")
        raise ValueError("Too many failed attempts. ")

    sv_survey = {"first": "VLA FIRST (1.4 GHz)", "wise": "WISE 3.4"}
    survey = sv_survey[survey] if survey in sv_survey else survey

    if angular is None:
        FITS_SIZE = 5 * u.arcmin
    else:
        FITS_SIZE = (angular).to(u.arcsecond)

    CELL_SIZE = 2.0 * u.arcsec / u.pix
    imsize = FITS_SIZE.to("pix", equivalencies=u.pixel_scale(CELL_SIZE))

    try:
        images = SkyView.get_images(
            pos,
            survey,
            pixels=int(imsize.value),
            width=FITS_SIZE,
            height=FITS_SIZE,
            coordinates="J2000",
        )
    except:
        import time

        time.sleep(4)
        get_src_img(pos, survey, angular=angular, level=level + 1)

    return images[0]

