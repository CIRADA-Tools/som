"""Pipeline for the VLASS SOM

The SOM is trained on a separate dataset.
"""
import os, sys
import shutil
import subprocess
import numpy as np
import pandas as pd
from astropy import units as u
from astropy.table import Table, vstack
from astropy.coordinates import SkyCoord, search_around_sky

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
    cname = f"CIRADA_VLASS1QL_table4_complex_sources_v1_{tile_id}.xml"
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


def load_unwise(ir_file, path=""):
    # Should be generated through TOPCAT
    ir_cat = Table.read(os.path.join(sample_path, ir_file))
    ir_cat = ir_cat[["objID", "RAdeg", "DEdeg", "FW1", "FW2"]]
    ir_cat.rename_columns(("RAdeg", "DEdeg"), ("RA", "DEC"))
    ir_cat = ir_cat.to_pandas()
    ir_cat["objID"] = ir_cat["objID"].str.decode("ascii")
    return ir_cat


# /// Input Info \\\
overwrite = False

# Run from `Pipeline` directory
pipe_dir = os.getcwd()
base_dir = os.path.split(pipe_dir)[0]

# Radio Input Info
data_path = os.path.join(base_dir, "Data")
radio_path = os.path.join(data_path, "vlass")
radio_component_catalogue = "CIRADA_VLASS1QL_table1_components_v1.fits"
radio_cat_file = os.path.join(radio_path, radio_component_catalogue)
all_radio_comps = vdl.load_vlass_catalogue(radio_cat_file, complex=False, pandas=True)

# Sample of pairs
sample_path = "/home/adrian/CIRADA/Sample"
sample_file = "Pairs_uniformNNdist_5asecbins_minSN10_cleansep.csv"
radio_cat = vdl.load_vlass_catalogue(
    os.path.join(sample_path, sample_file), pandas=True, complex=False,
)

ir_file = "Pairs_uniformNNdist_5asecbins_minSN10_cleansep_IR.fits"
ir_cat = load_unwise(ir_file, path=sample_path)

imbin_file = "IMG_uniformNNdist_5asecbins_minSN10_cleansep_compfilt_w7525.bin"
imgs = pu.ImageReader(os.path.join(sample_path, imbin_file))

# SOM Input Info
som_path = os.path.join(
    "/home/adrian/CIRADA/SOM", "Pairs", "ComponentFiltered", "w7525"
)
som_file = os.path.join(som_path, "SOM_B3_h10_w10_vlass.bin")

map_file = "MAP_uniformNNdist_5asecbins_minSN10_cleansep_compfilt_w7525_ComponentFiltered_10x10.bin"
trans_file = map_file.replace("MAP", "TRANSFORM")
somset = pu.SOMSet(
    som_file,
    os.path.join(sample_path, map_file),
    os.path.join(sample_path, trans_file),
)
somset.som.bmu_mask = None

# ir_cat_path = os.path.join(data_path, "unwise")
# cutout_path = os.path.join(data_path, "image_cutouts")

out_cat_path = os.path.join(pipe_dir, "catalogues")
# out_bin_path = os.path.join(pipe_dir, "binaries")
for out_dir in [out_cat_path]:
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

print("Loading the SOM annotations")
annotations_file = f"{som_file}.results.pkl"
annotation = pu.Annotator(somset.som.path, results=annotations_file)

# Everything is now loaded in. Run the collation.
comp_cat, src_cat = cxc.collate(
    radio_cat,
    ir_cat,
    imgs,
    somset,
    annotation,
    sorter_mode="area_ratio",
    pix_scale=0.6,
    host_name="objID",
    # full_radio_cat=all_radio_comps,
)

# Even not using full_radio_cat results in a smaller comp_cat
# The src_cat comps do not (closely) match radio_cat
# Some source names are repeated too

"""
Make the following modifications:
1. Find out where the missing components are going
   -- Also duplicate sources
2. Revise the annotations for the 'best' true double neurons
3. Test the bmu_mask
"""

"""
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
"""
