"""Pipeline for the EMUCat Complex-XID process

Input:
- A radio component catalogue
    - One catalogue per pipeline run? Leave chunking to the pipeline?
- A WISE catalogue covering the same field
    - Can I submit an SQL request to get one?
- A SOM and its annotations, created separately
- EMU tiles, which cutouts are extracted from

Intermediate files:
- EMU and WISE image cutouts
- Preprocessed image binary
- Map and transform files from mapping the prepro files through the SOM
- Source table for each chunk that the pipeline must loop through

Output:
- A file containing the annotation information
- A component table, containing:
    - Component name (links to the main component catalogue)
    - Best-matching neuron (x and y ID)
    - Transformations to match to best-matching neuron (flip, angle)
    - Source name (links to the source table)
- A source table, containing:
    - A source name, based on the source position
    - A list of component names belonging to the source
    - Source position, based on either the radio components or host position
    - Host galaxy ID
    - Source flux, area, extent, and major axis, 
      determined from the image using the SOM transformations
    - Source morphology (labels TBD)
"""

import os
import shutil
import subprocess
import numpy as np
import pandas as pd
from astropy.table import Table
import pdb

from legacy_survey_cutout_fetcher import grab_cutouts
import pyink as pu
import preprocessing
import vlass_data_loader as vdl
import collation

from vos import Client as VOSpace

# /cirada/data/EMU/PilotSurvey


def binary_names(unique_id, path=""):
    imbin_file = os.path.join(path, f"IMG_{unique_id}.bin")
    map_file = os.path.join(path, f"MAP_{unique_id}.bin")
    trans_file = os.path.join(path, f"TRANSFORM_{unique_id}.bin")
    return imbin_file, map_file, trans_file


def add_filename(objname, survey="DECaLS-DR8", format="fits"):
    # Take Julian coords of name to eliminate white space - eliminate prefix
    name = objname.split(" ")[1]
    filename = f"{name}_{survey}.{format}"
    return filename


def map_imbin(
    imbin_file,
    som_file,
    map_file,
    trans_file,
    som_width,
    som_height,
    numthreads=4,
    cpu=False,
    nrot=360,
    log=True,
):
    commands = [
        "Pink",
        "--map",
        imbin_file,
        map_file,
        som_file,
        "--numthreads",
        f"{numthreads}",
        "--som-width",
        f"{som_width}",
        "--som-height",
        f"{som_height}",
        "--store-rot-flip",
        trans_file,
        "--euclidean-distance-shape",
        "circular",
        "-n",
        str(nrot),
    ]
    if cpu:
        commands += ["--cuda-off"]

    if log:
        map_logfile = map_file.replace(".bin", ".log")
        with open(map_logfile, "w") as log:
            subprocess.run(commands, stdout=log)
    else:
        subprocess.run(commands)


def reformat_source_table(catalog):
    # Need to modify certain columns so they are in a csv-friendly format
    # `catalog` should be a pandas.DataFrame instance
    catalog["Component_names"] = catalog["Component_names"].apply(lambda x: ";".join(x))
    catalog["RA_components"] = catalog["RA_components"].apply(
        lambda x: ";".join(str(xi) for xi in x)
    )
    catalog["DEC_components"] = catalog["DEC_components"].apply(
        lambda x: ";".join(str(xi) for xi in x)
    )

    catalog["Host_candidates"] = catalog["Host_candidates"].apply(lambda x: ";".join(x))
    catalog["RA_host_candidates"] = catalog["RA_host_candidates"].apply(
        lambda x: ";".join(str(xi) for xi in x)
    )
    catalog["DEC_host_candidates"] = catalog["DEC_host_candidates"].apply(
        lambda x: ";".join(str(xi) for xi in x)
    )

    bmu_y, bmu_x = zip(*catalog.Best_neuron)
    catalog["Best_neuron_y"] = bmu_y
    catalog["Best_neuron_x"] = bmu_x
    # del catalog["best_neuron"]
    return catalog


def stack_catalogues(stack, path=""):
    for tile_cat in stack:
        new_df = Table.read(os.path.join(path, tile_cat)).to_pandas()
        try:
            final_cat = pd.concat([final_cat, new_df])
        except NameError:
            final_cat = new_df.copy()

    if "col0" in final_cat:
        del final_cat["col0"]
    final_cat = final_cat.reset_index(drop=True)
    return final_cat


def download_data(
    download_path,
    tile_id,
    name_col="Component_name",
    imgsize_arcmin=3.0,
    imgsize_pix=150,
    **kwargs,
):
    # Acquire cutouts using the Legacy Skyviewer
    if not os.path.exists(download_path):
        os.makedirs(download_path)

    print(f"Downloading any required image cutouts to the directory: {download_path}")
    grab_cutouts(
        radio_cat,
        output_dir=download_path,
        name_col=name_col,
        imgsize_arcmin=imgsize_arcmin,
        imgsize_pix=imgsize_pix,
        **kwargs,
    )


def preprocess(
    radio_cat,
    imbin_file,
    img_size=(2, 150, 150),
    tile_cutout_path="",
    remove_tile_cutouts=False,
    overwrite=False,
):
    # /// Preprocess Data \\\
    if not os.path.exists(imbin_file) or overwrite:
        preprocessing.main(radio_cat, imbin_file, img_size, tile_cutout_path)

    if remove_tile_cutouts:
        shutil.rmtree(tile_cutout_path)

    imgs = pu.ImageReader(imbin_file)
    return imgs


def map(
    imbin_file,
    som_file,
    map_file,
    trans_file,
    som_width,
    som_height,
    cpu_only=False,
    nrot=360,
    numthreads=10,
    overwrite=False,
    log=True,
):
    # /// Map through SOM \\\
    imgs = pu.ImageReader(imbin_file)
    if not os.path.exists(map_file) or overwrite:
        map_imbin(
            imbin_file,
            som_file,
            map_file,
            trans_file,
            som_width,
            som_height,
            numthreads=numthreads,
            cpu=cpu_only,
            nrot=nrot,
            log=log,
        )

    som = pu.SOM(som_file)
    mapping = pu.Mapping(map_file)
    transform = pu.Transform(trans_file)
    somset = pu.SOMSet(som, mapping, transform)
    return somset


def collate(
    radio_cat,
    ir_cat,
    imgs,
    somset,
    annotation,
    sorter_mode="area_ratio",
    pix_scale=0.6,
    comp_name_col="Component_name",
    **kwargs,
):
    # /// Collate Sources \\\
    src_cat = collation.main(
        radio_cat,
        ir_cat,
        imgs,
        somset,
        annotation,
        sorter_mode=sorter_mode,
        pix_scale=pix_scale,
        **kwargs,
    )

    all_srcs = np.repeat(src_cat.Source_name, src_cat.Component_names.str.len())
    all_comps = np.hstack(src_cat.Component_names)
    comp_src_map = pd.DataFrame({comp_name_col: all_comps, "Source_name": all_srcs})

    # This will miss out on components that did not preprocess properly
    # Probably not a problem, since it can be joined normally with the original
    comp_cat = collation.component_table(radio_cat.loc[imgs.records], somset)
    comp_cat = comp_cat.merge(comp_src_map, on=comp_name_col)
    comp_cat.Flip = comp_cat.Flip.astype("int32")

    src_cat = pd.merge(
        src_cat,
        comp_cat[["Component_name", "Flip", "Angle"]],
        left_on="Best_component",
        right_on="Component_name",
    )
    del src_cat["Component_name"]

    return comp_cat, src_cat


def run_all(
    catalogues,
    som_file,
    unique_id,
    image_cutout_path,
    bin_path="",
    img_size=(2, 150, 150),
    numthreads=10,
    annotation=None,
    annotations_file=None,
    remove_tile_cutouts=False,
    cpu_only=False,
    sorter_mode="area_ratio",
    pix_scale=0.6,
    som_mask=None,
):
    """Run the preprocess, map, and collate steps for a single sample.

    Args:
        catalogues (tuple): DataFrames of the radio and ir catalogues
        som_file (str): Name of the SOM file
        unique_id (str): Unique identifier for the sample (tile id, ssid, etc)
        image_cutout_path (str): Path to the directory containing the image cutouts
        annotations_file (str, optional): Name of the SOM annotations file. Defaults to a name based on the SOM file name.

    Returns:
        [type]: [description]
    """
    radio_cat, ir_cat = catalogues

    # Preprocess
    imbin_file, map_file, trans_file = binary_names(unique_id, bin_path)
    print(f"Preprocessing the sample: {imbin_file}")
    imgs = preprocess(
        radio_cat,
        imbin_file,
        img_size=img_size,
        tile_cutout_path=image_cutout_path,
        remove_tile_cutouts=remove_tile_cutouts,
    )
    print("...done")

    # Map
    print(f"Mapping...")
    som = pu.SOM(som_file)
    w, h, d = som.som_shape
    somset = map(
        imbin_file,
        som_file,
        map_file,
        trans_file,
        w,
        h,
        numthreads=numthreads,
        cpu_only=cpu_only,
    )
    somset.som.bmu_mask = som_mask
    print("...done")

    # Collate
    if annotation is None:
        if annotations_file is None:
            annotations_file = f"{som_file}.results.pkl"
        annotation = pu.Annotator(som.path, results=annotations_file)

    print(f"Collating...")
    comp_cat, src_cat = collate(
        radio_cat,
        ir_cat,
        imgs,
        somset,
        annotation,
        sorter_mode=sorter_mode,
        pix_scale=pix_scale,
    )
    print("...done")

    return comp_cat, src_cat
