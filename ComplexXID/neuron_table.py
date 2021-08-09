"""Create a table indexed by the SOM neuron

Common columns:
- Reliability flag
- Morphological label
- Channel mask areas
Dependent on combining the image samples:
- Number of images that match the neuron
- All percentiles
"""

import os, sys
import json
import argparse
from collections import Counter, OrderedDict
from itertools import product

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.table import Table, join
from astropy.coordinates import SkyCoord
import pyink as pu
import seaborn as sns
from catalogue_tools import prepare_zoo_image


def sample_percentile(sample, perc, fill_bad=-1):
    if len(sample) == 0:
        return fill_bad
    return np.percentile(sample, perc)


def neuron_info(neuron, som, comp_cat, src_cat=None):
    # comp_cat: pd.DataFrame
    sample = comp_cat[comp_cat["Best_neuron"] == neuron]
    bmu_counts = len(sample)

    radio_sum = som[neuron][0].sum()
    ir_sum = som[neuron][1].sum()

    # Should these be _component_ or _source_ distributions?
    ed_0 = sample_percentile(sample["Euc_dist"], 0)
    ed_25 = sample_percentile(sample["Euc_dist"], 25)
    ed_50 = sample_percentile(sample["Euc_dist"], 50)
    ed_75 = sample_percentile(sample["Euc_dist"], 75)
    ed_100 = sample_percentile(sample["Euc_dist"], 100)

    info_tab = OrderedDict(
        Neuron_ID=neuron,
        N_components=bmu_counts,
        Radio_sum=radio_sum,
        IR_sum=ir_sum,
        Euc_dist_perc_0=ed_0,
        Euc_dist_perc_25=ed_25,
        Euc_dist_perc_50=ed_50,
        Euc_dist_perc_75=ed_75,
        Euc_dist_perc_100=ed_100,
    )

    if src_cat is not None:
        info_tab["N_matches"] = len(src_cat[src_cat.Best_neuron == neuron])

    return info_tab


def neuron_table(som, comp_cat, src_cat):
    w, h, d = som.som_shape
    Y, X = np.mgrid[:w, :h]
    neurons = list(zip(Y.flatten(), X.flatten()))
    for i, neuron in enumerate(neurons):
        stats = neuron_info(neuron, som, comp_cat, src_cat)
        if stats is not None:
            yield stats


def annotation_info(neuron, annotations):
    ant = annotations.results[neuron]
    labels = dict(ant.labels)
    img_area = np.product(ant.filters[0].shape)

    radio_mask = pu.valid_region(
        ant.filters[0],
        filter_includes=labels["Related Radio"],
        filter_excludes=labels["Sidelobe"],
    )
    radio_area = radio_mask.sum() / img_area

    ir_mask = pu.valid_region(
        ant.filters[1],
        filter_includes=labels["IR Host"],
        filter_excludes=labels["Sidelobe"],
    )
    ir_area = ir_mask.sum() / img_area

    ambiguous = np.any(
        pu.valid_region(
            ant.filters[1],
            filter_includes=labels["Ambiguous"],
            filter_excludes=labels["Sidelobe"],
        )
    )
    return OrderedDict(
        Neuron_ID=neuron, Radio_area=radio_area, IR_area=ir_area, Ambiguous_IR=ambiguous
    )


def annotation_table(som, annotation):
    w, h, d = som.som_shape
    Y, X = np.mgrid[:w, :h]
    neurons = list(zip(Y.flatten(), X.flatten()))
    for i, neuron in enumerate(neurons):
        yield annotation_info(neuron, annotation)


def load_as_tuple(string):
    return tuple(json.loads(string.replace("(", "[").replace(")", "]")))


def img_positions(points, som_shape=35):
    img = np.zeros((som_shape, som_shape))
    for p in points:
        img[p] += 1
    return img


comp_cat = Table.read("CIRADA_VLASS1QL_table1_components_som_v01.xml")
src_cat = Table.read("CIRADA_VLASS1QL_table4_complex_sources_v01.xml")
src_cat = src_cat.to_pandas()

comp_cat = comp_cat.to_pandas()
comp_cat["Best_neuron"] = list(
    zip(comp_cat["Best_neuron_y"], comp_cat["Best_neuron_x"])
)
# comp_cat = Table.from_pandas(comp_cat_df)

src_cat["Best_neuron"] = list(map(load_as_tuple, src_cat["Best_neuron"]))
# src_cat["Component_names"] = [
#     json.loads(src["Component_names"].replace("'", '"'))
#     for i, src in src_cat.iterrows()
# ]
# src_cat["Component_name"] = [src["Component_names"][0] for i, src in src_cat.iterrows()]
# src_cat = pd.merge(src_cat, comp_cat[["Component_name", "flip", "angle"]], how="left")

# top_comp = [json.loads(src["Component_names"].replace("'", '"'))[0] for src in src_cat]
# with src_cat.context_rename_columns(['a', 'b'], ['A', 'B']):
#     out = table.join(t1, t2)
# src_cat = join(
#     src_cat, comp_cat[["Component_name", "flip", "angle"]], keys="Component_name"
# )

pipe_dir = os.getcwd()
base_dir = os.path.split(pipe_dir)[0]
data_path = os.path.join(base_dir, "Data")
model_path = os.path.join(base_dir, "SOM")
som_file = os.path.join(model_path, "complex_300px", "SOM_B3_h35_w35_vlass.bin")
annotations_file = f"{som_file}.results.pkl"
som = pu.SOM(som_file)
annotation = pu.Annotator(som.path, results=annotations_file)

neuron_cat = pd.DataFrame(neuron_table(som, comp_cat, src_cat))
ant_cat = pd.DataFrame(annotation_table(som, annotation))
neuron_cat = pd.merge(ant_cat, neuron_cat)

neuron_tab = Table.from_pandas(neuron_cat)
neuron_tab.write("CIRADA_VLASS1QL_table5_neuron_info_v01.xml", format="votable")


img = np.zeros((35, 35))
for i, j in product(range(35), range(35)):
    img[i, j] = neuron_cat.N_matches[neuron_cat.Neuron_ID == (i, j)].iloc[0]

plt.hist(np.log10(neuron_cat.N_matches[neuron_cat.N_matches > 0]), bins=25)
plt.xlabel("Number of sources matched to the neuron")
plt.ylabel("N")
plt.savefig("neuron_matches_hist.png")


no_ir = neuron_cat.IR_area == 0
some_radio = neuron_cat.Radio_area > 0
ambig_mask = ~neuron_cat.Ambiguous_IR
good_neurons = ~no_ir & some_radio & ambig_mask

neuron_cat["Area_ratio"] = neuron_cat.Radio_area / neuron_cat.IR_area
large_ar = neuron_cat.Area_ratio >= 2

matches_mask = neuron_cat.N_matches >= np.percentile(neuron_cat.N_matches, 67)
ir_dist_mask = neuron_cat.IR_area < np.percentile(neuron_cat[~no_ir].IR_area, 50)

large_radio = neuron_cat.Radio_area >= np.percentile(neuron_cat.Radio_area, 75)

possible_pair = (
    (neuron_cat.Area_ratio > 0.5)
    & (neuron_cat.Area_ratio < 1.5)
    & (neuron_cat.IR_sum > np.percentile(neuron_cat.IR_sum, 50))
)


# True doubles:
# Large radio. IR might be large, as it can be smeared out. Area_ratio > 1
# Independent pairs:
# Area_ratio ~ 1. Large IR_sum.

mask = matches_mask & ir_dist_mask & ambig_mask
neuron_cat["good"] = mask

sns.scatterplot(
    data=neuron_cat[good_neurons], x="Area_ratio", y="Radio_area", hue="good"
)


candidates = neuron_cat.Neuron_ID[mask].values
choices = np.random.choice(candidates, 45, replace=False)
print(choices)
plt.imshow(img_positions(choices, som.som_shape[0]))

# from sklearn.cluster import KMeans
# km = KMeans(30).fit(list(candidates))
# choices = [tuple(ki) for ki in km.cluster_centers_.astype(int)]


path = pu.PathHelper("Zoo_revised")
for neuron in choices:
    plt.close("all")
    som.plot_neuron(neuron)
    plt.savefig(f"{path.path}/neuron_{neuron[0]}-{neuron[1]}.png")


doubles = [
    (0, 8),
    # (0, 28),
    (0, 29),
    # (1, 8),
    (4, 27),
    (6, 18),
    (28, 9),
    (30, 25),
    (31, 4),
    (31, 22),
    (31, 23),
    (31, 27),
    (31, 34),
    (32, 21),
    (32, 23),
    (32, 27),
    (32, 29),
    (32, 34),
    (33, 30),
    (33, 31),
    (34, 32),
]

# pairs = []

final = [
    # (0, 10),
    (2, 30),
    (3, 7),
    # (3, 33),
    # (4, 34),
    (5, 32),
    (7, 5),
    # (7, 8),
    # (7, 33),
    # (8, 4),
    # (8, 28),
    # (10, 31),
    (11, 18),
    (11, 33),
    # (12, 12),
    (12, 26),
    (12, 32),
    (13, 3),
    (13, 18),
    # (13, 34),
    (14, 11),
    (14, 15),
    (14, 27),
    (17, 3),
    (18, 20),
    (18, 29),
    # (19, 31),
    (19, 33),
    (20, 24),
    (21, 25),
    # (22, 16),
    (22, 18),
    (22, 30),
    (22, 32),
    (23, 4),
    # (25, 7),
    (25, 16),
    # (26, 1),
    (26, 30),
    (26, 34),
    (27, 0),
    (28, 31),
    (30, 25),
    (31, 15),
    # (32, 32),
]

final = [(32, 29)]

neuron_cat["final"] = neuron_cat.Neuron_ID.isin(final)

for neuron in final:
    plt.close("all")
    som.plot_neuron(neuron)
    plt.savefig(f"{path.final}/neuron_{neuron[0]}-{neuron[1]}.png")


def plot_zoo_source(*args, **kwargs):
    prepare_zoo_image(*args, **kwargs)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()


def filename(src_name):
    return f"{src_name.split()[1]}.png"


def sample_neurons(src_cat, neurons, n_samples=10):
    for neuron in neurons:
        subset = src_cat[src_cat["Best_neuron"] == neuron]
        if len(subset) > n_samples:
            subset = subset.sample(n_samples)
        try:
            stacked = pd.concat([stacked, subset])
        except NameError:
            stacked = subset
    return stacked


# TODO: Randomly sample 10 images for each neuron in `final`.
subset = sample_neurons(src_cat, final)
subset["filename"] = [filename(src.Source_name) for i, src in subset.iterrows()]
subset[["Source_name", "Component_name", "filename"]].to_csv(f"{path.path}/sample.csv")

for idx, src in subset.iterrows():
    plt.close("all")
    plot_zoo_source(src_cat, annotation, idx, file_path="images")
    plt.savefig(f"{path.new_images}/{filename(src.Source_name)}")
