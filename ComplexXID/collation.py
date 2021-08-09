"""Collate a sample into a source table and update the component catalogue
with the appropriate source name.

Future considerations:
- Build a table of neuron properties
- Need to decide on the list of global annotations (i.e. the morphologies
and any labels to describe neuron quality).
- How will non-global annotations (e.g. filters based on sidelobes) be
incorporated into the tables.

Still to add:
- Best host
- Host name
- Replace host coords with names for the non-best IDs
- VLASS/EMU tile ID?
"""

import os, sys
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy import units as u
from astropy.wcs import WCS
from astropy.table import Table, QTable
from astropy.coordinates import SkyCoord, search_around_sky
from astropy.nddata.utils import Cutout2D
import matplotlib.pyplot as plt
from collections import Counter, OrderedDict
from astroquery.skyview import SkyView
import networkx as nx
import pyink as pu
import pdb

# from diagnostics import plot_image, plot_source, plot_filter_idx


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


def source_name(ra, dec, aprec=2, dp=5, prefix="VLASS1QLCIR"):
    # Truncated
    ra, dec = np.array(ra), np.array(dec)
    cat = SkyCoord(ra=ra, dec=dec, unit="deg")

    astring = cat.ra.to_string(sep="", unit="hour", precision=dp, pad=True)
    dstring = cat.dec.to_string(sep="", precision=dp, pad=True, alwayssign=True)

    # Truncation index
    tind = aprec + 7
    sname = [f"{prefix} J{r[:tind]}{d[:tind]}" for r, d in zip(astring, dstring)]
    return sname


def retrieve_source(group, radio_positions, ir_posns, idx):
    G = group.graph
    subg = sorted(nx.connected_components(G), key=lambda x: len(x), reverse=True)[idx]
    subg = list(subg)

    data = [d for d in G.edges(nbunch=subg, data=True)]
    data = min(data, key=lambda x: x[2]["count"])

    ir_hosts = data[2]["IR Host"]
    bmu = tuple(data[2]["bmu"])

    components = radio_positions[np.array(subg)]
    ir_srcs = ir_posns[ir_hosts]
    # mean_component = SkyCoord(components.cartesian.mean())

    node_data = [G.nodes(data=True)[d] for d in subg]
    # onejet = any([d["onejet"] for d in node_data if "onejet" in d.keys()])
    return data, bmu, components, ir_srcs


def source_info(
    group,
    subg,
    base_cat,
    match_catalogues,
    radio_name="Component_name",
    host_name="unwise_detid",
):
    G = group.graph
    somset = group.sorter.som_set
    subg = list(subg)
    # subg is for the idx of the radio match_catalogue in pu.FilterSet

    # match_coords = G.filters.match_catalogues
    radio_cat, ir_cat = match_catalogues

    data = [d for d in G.edges(nbunch=subg, data=True)]
    data = min(data, key=lambda x: x[2]["count"])

    src_idx = data[2]["src_idx"]
    ir_hosts = data[2]["IR Host"]
    bmu = tuple(data[2]["bmu"])
    filters = group.filters[src_idx]

    radio_ind_mask = filters[0].coord_label_contains("Related Radio")
    radio_inds = filters[0].coords.src_idx[radio_ind_mask]

    # Radio position info
    best_comp = base_cat.iloc[src_idx]
    # radio_comps = radio_cat.iloc[subg]  # networkx resets numbering
    radio_comps = radio_cat.iloc[radio_inds]  # networkx resets numbering
    comp_ra = list(radio_comps["RA"])
    comp_dec = list(radio_comps["DEC"])
    comp_names = list(radio_comps[radio_name])
    # mean_component = SkyCoord(components.cartesian.mean())

    # IR position info
    ir_comps = ir_cat.iloc[ir_hosts]
    ir_ra = list(ir_comps["RA"])
    ir_dec = list(ir_comps["DEC"])
    ir_host_id = list(ir_comps[host_name])

    # Derived info
    total_flux = radio_comps["Total_flux"].sum()
    e_total_flux = total_flux * np.sum(
        (radio_comps.E_Total_flux / radio_comps.Total_flux) ** 2
    )
    peak_comp = radio_comps.Peak_flux.idxmax()
    euc_dist = somset.mapping.bmu_ed(src_idx)[0]
    # node_data = [G.nodes(data=True)[d] for d in subg]

    source = OrderedDict(
        src_idx=src_idx,
        RA_source=best_comp["RA"],
        DEC_source=best_comp["DEC"],
        N_components=len(radio_comps),
        Best_component=best_comp.Component_name,
        Component_names=comp_names,
        RA_components=comp_ra,
        DEC_components=comp_dec,
        N_host_candidates=len(ir_comps),
        Host_candidates=ir_host_id,
        RA_host_candidates=ir_ra,
        DEC_host_candidates=ir_dec,
        Best_neuron=bmu,
        Euc_dist=euc_dist,
        Total_flux=total_flux,
        E_Total_flux=e_total_flux,
        Peak_flux=radio_comps.loc[peak_comp, "Peak_flux"],
        E_Peak_flux=radio_comps.loc[peak_comp, "E_Peak_flux"],
    )

    # if np.abs(source["RA_source"] - np.mean(source["RA_components"])) > 10:
    # pdb.set_trace()

    return source


def collate(group, *args, **kwargs):
    G = group.graph
    for i, subg in enumerate(
        sorted(nx.connected_components(G), key=lambda x: len(x), reverse=True)
    ):
        yield source_info(group, subg, *args, **kwargs)


def best_host(components, ir_srcs, sep_limit):
    # A host is most likely when it aligns with a radio components
    likely_host = False
    sep_limit = u.Quantity(sep_limit, u.arcsec)

    sky_matches = search_around_sky(components, ir_srcs, seplimit=15 * u.arcsecond)
    if len(sky_matches[0]) > 0:
        best_match = np.argmin(sky_matches[2])
        best_ir_match = ir_srcs[sky_matches[1][best_match]]

        # Obtained from cross-matching a shifted catalogue to WISE
        if sky_matches[2][best_match] < 3.4 * u.arcsecond:
            # ax.scatter(
            #     best_ir_match.ra,
            #     best_ir_match.dec,
            #     transform=ax.get_transform("world"),
            #     color="white",
            #     s=22,
            # )
            likely_host = True
    return likely_host


def main(
    radio_cat,
    ir_cat,
    imgs,
    somset,
    annotation,
    sorter_mode="area_ratio",
    pix_scale=0.6,
    full_radio_cat=None,
    **kwargs,
):
    pix_scale = u.Quantity(pix_scale, u.arcsec)
    seplimit = 0.5 * np.max(imgs.data.shape[-2:]) * pix_scale

    radio_cat = radio_cat.loc[imgs.records]
    if full_radio_cat is None:
        full_radio_cat = radio_cat

    radio_posns = SkyCoord(
        np.array(radio_cat["RA"]), np.array(radio_cat["DEC"]), unit=u.deg
    )

    all_radio_posns = SkyCoord(
        np.array(full_radio_cat["RA"]), np.array(full_radio_cat["DEC"]), unit=u.deg
    )
    # radio_matches = search_around_sky(radio_posns, all_radio_posns, seplimit=seplimit)
    # full_radio_cat = (
    #     full_radio_cat.iloc[radio_matches[1]].drop_duplicates().reset_index(drop=True)
    # )
    # all_radio_posns = SkyCoord(
    #     np.array(full_radio_cat["RA"]), np.array(full_radio_cat["DEC"]), unit=u.deg
    # )

    ir_cat = ir_cat.drop_duplicates().reset_index(drop=True)
    ir_posns = SkyCoord(np.array(ir_cat["RA"]), np.array(ir_cat["DEC"]), unit=u.deg)
    # ir_matches = search_around_sky(radio_posns, ir_posns, seplimit=seplimit)
    # ir_cat = ir_cat.iloc[ir_matches[1]].drop_duplicates().reset_index(drop=True)
    # ir_posns = SkyCoord(np.array(ir_cat["RA"]), np.array(ir_cat["DEC"]), unit=u.deg)

    # Projecting the filters
    filters = pu.FilterSet(
        radio_posns,
        (all_radio_posns, ir_posns),
        # (radio_posns, ir_posns),
        annotation,
        somset,
        seplimit=seplimit,
        progress=True,
        pixel_scale=pix_scale,
    )
    print(f"Number of matches are: {[len(fs[0]) for fs in filters.sky_matches]}")

    # Inspect the assigned and unassigned components for a given index
    sorter = pu.Sorter(somset, annotation, mode=sorter_mode)

    # LINK, UNLINK, RESOLVE, FLAG, PASS, ISOLATE
    # NODE_ATTACH, DATA_ATTACH, TRUE_ATTACH, FALSE_ATTACH
    actions = pu.LabelResolve(
        {
            "Follow up": pu.Action.TRUE_ATTACH,
            "Related Radio": pu.Action.LINK,
            "IR Host": pu.Action.DATA_ATTACH,
            # "Sidelobe": pu.Action.TRUE_ATTACH,
        }
    )

    def src_fn(idx):
        # idx is taken from the Sorter, so one per image
        return {
            "idx": idx,
            "bmu": somset.mapping.bmu(idx, bmu_mask=somset.som.bmu_mask),
            "ra": radio_posns[idx].ra.deg,
            "dec": radio_posns[idx].dec.deg,
        }

    print("Grouping the components")
    group = pu.Grouper(
        filters, annotation, actions, sorter, src_stats_fn=src_fn, progress=True
    )

    print("Creating the source catalogue")
    source_cat = pd.DataFrame(
        collate(group, radio_cat, (full_radio_cat, ir_cat), **kwargs)
    )

    names = source_name(source_cat.RA_source, source_cat.DEC_source, prefix="VLASS1QLC")
    source_cat.insert(0, "Source_name", names)
    # source_cat["Source_name"] = names

    return source_cat


def component_table(radio_cat, somset, name_col="Component_name"):
    bmu = somset.mapping.bmu(bmu_mask=somset.som.bmu_mask)
    bmut = somset.mapping.bmu(return_tuples=True, bmu_mask=somset.som.bmu_mask)
    euc_dist = somset.mapping.bmu_ed(bmu_mask=somset.som.bmu_mask)
    ind = np.arange(somset.transform.data.shape[0])
    trans = somset.transform.data[ind, bmu[:, 0], bmu[:, 1]]
    flip = trans["flip"]
    angle = trans["angle"]
    comp_tab = pd.DataFrame(
        {
            name_col: radio_cat[name_col],
            "bmu": bmut,
            "Best_neuron_y": bmu[:, 0],
            "Best_neuron_x": bmu[:, 1],
            "Euc_dist": euc_dist,
            "Flip": flip,
            "Angle": angle,
        }
    )
    return comp_tab


def neuron_table():
    # Create neuron table
    # bmu, global text labels, any properties derived from filters
    return

