"""Pipeline for the VLASS SOM

The SOM is trained on a separate dataset.
"""
import os, sys
import shutil
import subprocess
import numpy as np
import pandas as pd
from astropy.table import Table, vstack

from legacy_survey_cutout_fetcher import grab_vlass_unwise_cutouts
import pyink as pu
import vlass_data_loader as vdl
import complex_xid_core as cxc

# from vos import Client as VOSpace


def add_filename(objname, survey="DECaLS-DR8", format="fits"):
    # Take Julian coords of name to eliminate white space - eliminate prefix
    name = objname.split(" ")[1]
    filename = f"{name}_{survey}.{format}"
    return filename


def src_cat_name(tile_id, path=""):
    cname = f"CIRADA_VLASS1QL_table4_complex_sources_v01_{tile_id}.xml"
    return os.path.join(path, cname)


def comp_cat_name(tile_id, path=""):
    cname = f"component_som_info_{tile_id}.xml"
    return os.path.join(path, cname)


def read_table(infile, pandas=False):
    ext = infile.split(".")[-1]
    format = {"xml": "votable", "csv": "ascii"}
    tab = Table.read(infile, format=format[ext])
    if pandas:
        tab = tab.to_pandas()
    return tab


def stack_catalogues(stack, path=""):
    # Files to stack are in VOTable format
    for tile_cat in stack:
        new_tab = read_table(tile_cat)
        try:
            # final_cat = pd.concat([final_cat, new_tab])
            final_cat = vstack([final_cat, new_tab])
        except NameError:
            final_cat = new_tab.copy()

    # final_cat = final_cat.reset_index(drop=True)
    return final_cat


def load_tile_catalogues(
    out_cat_path, radio_cat, tile_cat, coad_summary, ir_data_path, tile_id
):
    radio_tile_file = os.path.join(out_cat_path, f"vlass_complex_{tile_id}.csv")
    try:
        radio_sample = pd.read_csv(radio_tile_file)
    except IOError:
        radio_sample = load_vlass_tile(radio_cat, tile_id)
        radio_sample.to_csv(radio_tile_file, index=False)

    ir_tile_file = os.path.join(out_cat_path, f"unWISE_{tile_id}.csv")
    try:
        ir_cat = pd.read_csv(ir_tile_file)
    except IOError:
        print(f"Building the IR catalogue for tile {tile_id}")
        ir_cat = load_unwise(tile_cat, coad_summary, ir_data_path, tile_id)
        ir_cat.to_csv(ir_tile_file, index=False)

    return (radio_sample, ir_cat)


def load_vlass_tile(radio_cat, tile_id):
    radio_sample = radio_cat[radio_cat.Tile == tile_id].copy()
    radio_sample = radio_sample.reset_index(drop=True)
    return radio_sample


def load_unwise(tile_cat, coad_summary, ir_data_path, tile_id):
    # Extract relevant IR info
    tile_info = tile_cat[tile_cat.Tile == tile_id].iloc[0]
    ra_range = (tile_info["min_ra"], tile_info["max_ra"])
    dec_range = (tile_info["min_dec"], tile_info["max_dec"])
    ir_cat = vdl.build_ir_cat(coad_summary, ir_data_path, ra_range, dec_range)
    # ir_cat = vdl.load_tile(coad_summary, ir_data_path, tile_id)
    return ir_cat


# /// Input Info \\\
overwrite = False

# Run from `Pipeline` directory
pipe_dir = os.getcwd()
base_dir = os.path.split(pipe_dir)[0]
data_path = os.path.join(base_dir, "Data")
model_path = os.path.join(base_dir, "SOM")

# Radio Input Info
radio_path = os.path.join(data_path, "vlass")
ir_cat_path = os.path.join(data_path, "unwise")
cutout_path = os.path.join(data_path, "image_cutouts")
out_cat_path = os.path.join(pipe_dir, "catalogues")
out_bin_path = os.path.join(pipe_dir, "binaries")
for out_dir in [out_cat_path, out_bin_path]:
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

radio_component_catalogue = "CIRADA_VLASS1QL_table1_components_v01.fits"
vlass_subtiles = "CIRADA_VLASS1QL_table3_subtile_info_v01.csv.gz"
ir_data_path = os.path.join(ir_cat_path, "coads")
coad_summary = os.path.join(ir_cat_path, "unWISE_coad_summary.csv")

# SOM Input Info
som_file = os.path.join(model_path, "complex_300px", "SOM_B3_h35_w35_vlass.bin")
annotations_file = f"{som_file}.results.pkl"

# /// Select Sample (single tile) \\\
print("Loading the VLASS catalogue, specifying complex components")
radio_cat_file = os.path.join(radio_path, radio_component_catalogue)
radio_cat = vdl.load_vlass_catalogue(
    radio_cat_file, complex=True, NN_dist=72, SNR_min=None
)

print("Creating the tile catalogue")
subtile_file = os.path.join(radio_path, vlass_subtiles)
subtile_cat = Table.read(subtile_file, format="csv").to_pandas()
tile_cat = vdl.vlass_tile(subtile_cat)

print("Loading the SOM and annotations")
som = pu.SOM(som_file)
annotation = pu.Annotator(som.path, results=annotations_file)

# sdss = sky_chunk(df, (120, 240), (-10, 50))
tiles = tile_cat.Tile.values

ti = int(sys.argv[1]) if len(sys.argv) >= 2 else 0
num_tiles = int(sys.argv[2]) if len(sys.argv) >= 3 else 1
tiles = tiles[ti : ti + num_tiles]

for tile_id in tiles:
    print(f"Processing tile {tile_id}")
    imbin_file, map_file, trans_file = cxc.binary_names(tile_id, path=out_bin_path)

    # Load VLASS and unWISE catalogues
    radio_sample, ir_cat = load_tile_catalogues(
        out_cat_path, radio_cat, tile_cat, coad_summary, ir_data_path, tile_id
    )

    radio_sample["filename"] = radio_sample["Component_name"].apply(
        add_filename, survey="VLASS"
    )
    radio_sample["ir_filename"] = radio_sample["Component_name"].apply(
        add_filename, survey="unWISE_NEO4"
    )

    # /// Acquire Data \\\
    tile_cutout_path = os.path.join(cutout_path, tile_id)
    if not os.path.exists(imbin_file):
        if not os.path.exists(tile_cutout_path):
            os.makedirs(tile_cutout_path)

        print(
            f"Downloading any required image cutouts to the directory: {tile_cutout_path}"
        )
        grab_vlass_unwise_cutouts(
            radio_sample,
            output_dir=tile_cutout_path,
            name_col="Component_name",
            ra_col="RA",
            dec_col="DEC",
            imgsize_arcmin=3.0,
            imgsize_pix=300,
        )

    # imgs = cxc.preprocess(
    #     radio_sample,
    #     imbin_file,
    #     img_size=(2, 300, 300),
    #     tile_cutout_path=tile_cutout_path,
    #     remove_tile_cutouts=False,
    # )

    # Preprocess, map, and collate
    comp_cat, src_cat = cxc.run_all(
        (radio_sample, ir_cat),
        som_file,
        tile_id,
        tile_cutout_path,
        bin_path=out_bin_path,
        img_size=(2, 300, 300),
        numthreads=8,
        annotation=annotation,
        remove_tile_cutouts=False,
        cpu_only=False,
        sorter_mode="area_ratio",
        pix_scale=0.6,
    )

    # /// Write to file \\\
    print("Writing the results to a file")
    comp_tab = Table.from_pandas(comp_cat)
    cname = comp_cat_name(tile_id, path=out_cat_path)
    comp_tab.write(cname, format="votable", overwrite=True)

    src_tab = Table.from_pandas(src_cat)
    sname = src_cat_name(tile_id, path=out_cat_path)
    src_tab.write(sname, format="votable", overwrite=True)


# /// Memory cleanup \\\
del radio_sample, ir_cat, src_cat
del radio_cat, som, annotation


# /// Stack source catalogues \\\
final_file = "CIRADA_VLASS1QL_table4_complex_sources_v01.xml"
tile_cats = [src_cat_name(tile_id, path=out_cat_path) for tile_id in tiles]
final_cat = stack_catalogues(tile_cats, path="")
final_cat.write(final_file, format="votable", overwrite=True)

# /// Stack component catalogues \\\
final_file = "CIRADA_VLASS1QL_table1_components_som_v01.xml"
tile_cats = [comp_cat_name(tile_id, path=out_cat_path) for tile_id in tiles]
final_cat = stack_catalogues(tile_cats, path="")
final_cat.write(final_file, format="votable", overwrite=True)
