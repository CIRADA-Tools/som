"""SOM diagnostics"""

import os, sys
from collections import Counter
import argparse
from itertools import product
from typing import Callable, Iterator, Union, List, Set, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.table import Table
from astropy.coordinates import SkyCoord
import pyink as pu


def img_hist_2d(neurons):
    """Given a list of neurons (tuples), count the number of occurrences for each.
    Return a np.nparray with the resulting counts.
    Similar to pu.SOM.bmu_counts, but starting with a list of neurons instead of indices.
    """
    neuron_count = Counter(neurons)
    img = np.zeros(somset.som.som_shape[:-1])
    for neuron in neuron_count:
        img[neuron] = neuron_count[neuron]
    return img


def plot_som_counts(somset, show_counts=False):
    """Plot the number of matches to each neuron in a SOM."""
    # Add an outfile
    counts = somset.mapping.bmu_counts()
    fig, ax = plt.subplots(1, 1, figsize=(10, 8), constrained_layout=True)
    cim = ax.imshow(counts)

    cbar = fig.colorbar(cim)
    cbar.set_label("Counts per Neuron", fontsize=16)
    # cbar.ax.tick_params(labelsize=16)

    if show_counts:
        for row in range(somset.mapping.data.shape[1]):
            for col in range(somset.mapping.data.shape[2]):
                ax.text(
                    col,
                    row,
                    f"{counts[row, col]:.0f}",
                    color="r",
                    horizontalalignment="center",
                    verticalalignment="center",
                )
    return


def dist_hist(
    somset, neuron=None, density=False, bins=100, log=True, ax=None, labels=True,
):
    """Plot the distribution of Euclidean distances for either a SOM
    or one of its neurons.
    """
    bmu = somset.mapping.bmu()
    bmu_ed = somset.mapping.bmu_ed()
    if neuron is not None:
        mask = np.array([bmu_i == neuron for bmu_i in bmu])
        bmu_ed = bmu_ed[mask]

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(7, 6), constrained_layout=True)

    if log:
        bmu_ed = np.log10(bmu_ed)
    ax.hist(bmu_ed, bins=bins, density=density)
    ax.set_yscale("log")
    if labels:
        xlabel = "Euclidean Distance" if not log else f"log(Euclidean Distance)"
        ax.set_xlabel(xlabel, fontsize=16)
        ax.set_ylabel(r"$N$", fontsize=16)


def dist_hist_2d(df, somset, bins=100, loglog=False):
    """Plot the distribution of Euclidean distances for ALL neurons
    in a SOM.
    WARNING: Slow.
    """
    w, h, d = somset.som.som_shape
    fig, axes = plt.subplots(h, w, sharey=True, sharex=True, figsize=(11, 10))
    for row in range(h):
        for col in range(w):
            ax = axes[row, col]
            dist_hist(
                df, neuron=(row, col), density=False, loglog=loglog, ax=ax, labels=False
            )
    fig.subplots_adjust(hspace=0, wspace=0)

    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor="none", top=False, bottom=False, left=False, right=False)
    plt.xlabel("Euclidean Distance", fontsize=16)
    plt.ylabel(r"$N$", fontsize=16)


def dist_stats(somset):
    """Compute the 50th percentile (median) and 1-sigma bounds from the 
    full set of Euclidean distances in a SOM.
    """
    w, h, d = somset.som.som_shape
    neurons = list(product(range(h), range(w)))
    Y, X = np.mgrid[:w, :h]

    dists = np.zeros(Y.shape)
    stds = np.zeros(Y.shape)

    for neuron in neurons:
        # print(neuron)
        inds = somset.mapping.images_with_bmu(neuron)
        if len(inds) == 0:
            continue
        ed = somset.mapping.bmu_ed(inds)
        dists[neuron[0], neuron[1]] = np.percentile(ed, 50)
        stds[neuron[0], neuron[1]] = np.percentile(ed, 67) - np.percentile(ed, 33)

    return dists, stds


def plot_dist_stats(somset):
    """Plot the distribution of median and 1-sigma Euclidean distances 
    for all images matched to a SOM.
    """
    dists, stds = dist_stats(somset)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    p1 = axes[0].imshow(dists)
    cb1 = fig.colorbar(p1, ax=axes[0], aspect=50, shrink=0.95)
    cb1.set_label("Median distance to neuron", size=16)

    p2 = axes[1].imshow(stds)
    cb2 = fig.colorbar(p2, ax=axes[1], aspect=50, shrink=0.95)
    cb2.set_label(r"1$\sigma$ dispersion in distance", size=16)


def distance_sampling(somset, N=10):
    """Sample N images at low, medium, and high Euclidean distances.
    Useful for inspecting the images that are good, okay, and bad
    matches to a neuron.
    """
    inds = np.argsort(somset.mapping.bmu_ed())
    low = inds[:N]
    med = inds[len(inds) // 2 - N // 2 : len(inds) // 2 + N // 2]
    high = inds[-N:][::-1]
    return low, med, high


def neuron_img_comp(somset, imgs, sampler, outpath=""):
    """For each of a set of neurons defined in a pyink.SOMSampler class,
    plot the comparison between the neuron and a collection of images
    that are best-matched to the neuron. The collection of images is 
    taken from low, medium, and high Euclidean distances.
    Somewhat useful for the inspection stage when training a SOM.
    """
    path = pu.PathHelper(outpath)
    logfile = open(f"{path.path}/info.txt", "w")

    bmu_ed = somset.mapping.bmu_ed()

    img_shape = imgs.data.shape[-1]
    neuron_shape = somset.som.neuron_shape[-1]
    b1 = (neuron_shape - img_shape) // 2
    b2 = b1 + img_shape

    for neuron, ind in enumerate(sampler.points):
        # Plot neuron
        somset.som.plot_neuron(ind)
        f1 = plt.gcf()
        plt.xticks([])
        plt.yticks([])

        radio_img = somset.som[ind][0]
        levels = np.linspace(0.25 * radio_img.max(), radio_img.max(), 4)
        f1.axes[1].contour(radio_img, levels=levels, colors="white", linewidths=0.5)

        f1.axes[0].set_xlim([b1, b2])
        f1.axes[0].set_ylim([b2, b1])
        for ax in f1.axes:
            ax.axvline(neuron_shape / 2, c="r", ls="--", lw=1)
            ax.axhline(neuron_shape / 2, c="r", ls="--", lw=1)

        f1.savefig(f"{path.path}/neuron_{neuron}.png")
        plt.close(f1)

        # Plot images
        matches = somset.mapping.images_with_bmu(ind)
        dist = bmu_ed[matches]
        idx1 = matches[np.argmin(dist)]
        idx2 = matches[np.argsort(dist)[len(matches) // 2]]
        for i, idx in enumerate([idx1, idx2]):
            plot_image(
                imgs,
                idx=idx,
                somset=somset,
                apply_transform=True,
                show_index=False,
                grid=True,
            )
            f2 = plt.gcf()
            plt.xticks([])
            plt.yticks([])
            for ax in f2.axes:
                ax.axvline(img_shape / 2, c="r", ls="--", lw=1)
                ax.axhline(img_shape / 2, c="r", ls="--", lw=1)

            bmu_idx = somset.mapping.bmu(idx)
            tkey = somset.transform.data[(idx, *bmu_idx)]
            radio_img = imgs.data[idx, 0]
            radio_img = pu.pink_spatial_transform(radio_img, tkey)
            levels = np.linspace(0.25 * radio_img.max(), radio_img.max(), 4)
            f2.axes[1].contour(radio_img, levels=levels, colors="white", linewidths=0.5)

            f2.savefig(f"{path.path}/neuron_{neuron}_img{i}.png")
            plt.close(f2)

        # Print to a log file
        print(f"Neuron {neuron}: {ind}", file=logfile)
        print(f"Number in neuron: {len(matches)}", file=logfile)
        print(f"img0 ind: {idx1}", file=logfile)
        print(f"img1 ind: {idx2}", file=logfile)
        print("------------------\n", file=logfile)

