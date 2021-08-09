"""Create a SOM inspection document"""

import os, sys
import numpy as np
import matplotlib.pyplot as plt
import pyink as pu
from diagnostics import dist_hist, plot_dist_stats, plot_som_counts, neuron_img_comp


def latex_figure(plot_title, caption, label, outfile, fullpage=False):
    if fullpage:
        print(f"\\begin{{figure*}}[h]", file=outfile)
    else:
        print(f"\\begin{{figure}}[h]", file=outfile)

    print(f"    \centering", file=outfile)
    if fullpage:
        print(f"    \includegraphics[width=\\textwidth]{{{plot_title}}}", file=outfile)
    else:
        print(f"    \includegraphics[width=\columnwidth]{{{plot_title}}}", file=outfile)
    print(f"    \caption{{{caption}}}", file=outfile)
    print(f"    \label{{fig:{label}}}", file=outfile)

    if fullpage:
        print(f"\end{{figure*}}", file=outfile)
    else:
        print(f"\end{{figure}}", file=outfile)


def neuron_comp_latex(base, caption, label, outfile):
    print(f"\\begin{{figure*}}", file=outfile)
    print(f"  \centering", file=outfile)
    print(f"  \includegraphics[width=0.75\\textwidth]{{{base}.png}}", file=outfile)
    print(f"  \includegraphics[width=0.75\\textwidth]{{{base}_img0.png}}", file=outfile)
    print(f"  \includegraphics[width=0.75\\textwidth]{{{base}_img1.png}}", file=outfile)
    print(f"  \caption{{{caption}}}", file=outfile)
    print(f"  \label{{fig:{label}}}", file=outfile)
    print(f"\end{{figure*}}", file=outfile)


if len(sys.argv) < 5:
    print(f"USAGE: {sys.argv[0]} IMG SOM MAP TRANSFORM")

imgs = pu.ImageReader(sys.argv[1])
som = pu.SOM(sys.argv[2])
mapping = pu.Mapping(sys.argv[3])
trans = pu.Transform(sys.argv[4])
somset = pu.SOMSet(som, mapping, trans)

path = pu.PathHelper("Inspection")
outfile = open("latex_imgs.tex", "w")

### SOM 2D histogram ###
plot_som_counts(somset, True)
plt.savefig(f"som_stats.png")
plt.clf()
caption = f"The total number of images within the {imgs.data.shape[0]} input images mapped to each neuron in the SOM."
latex_figure("som_stats.png", caption, "som_stats", outfile)
print("", file=outfile)
########################

### Euclidean Distance ###
dist_hist(somset, log=False)
plt.savefig(f"EucDist_hist_lin.png")
dist_hist(somset, log=True)
plt.savefig(f"EucDist_hist_log.png")
caption = (
    "Histogram of the Euclidean distance between an image and its best-matching neuron."
)
latex_figure("EucDist_hist_lin.png", caption, "eucdist", outfile)
print("", file=outfile)
plt.clf()
##########################

### Sample neurons ###
npoints = 30
sampler = pu.SOMSampler(som, npoints)
if os.path.exists("sampled_neurons.txt"):
    infile = open("sampled_neurons.txt", "r").read()
    sampler.points = eval(infile)
else:
    with open("sampled_neurons.txt", "w") as outf:
        print(sampler.points, file=outf)

sampler.visualize()
plt.tight_layout()
plt.savefig(f"sampled_neurons.png")
plt.clf()
caption = f"The {npoints} neurons that have been sampled for this analysis are shown in yellow."
latex_figure("sampled_neurons.png", caption, "sampled_neurons", outfile)
print("", file=outfile)
######################

### Median distance and 1-sigma dispersion ###
plot_dist_stats(somset)
plt.savefig("som_bmu_dist.png")
plt.clf()
caption = f"The median distance (left) and $1\sigma$ dispersion in the distance distribution (right) for all objects matched to a given neuron."
latex_figure("som_bmu_dist.png", caption, "som_dist", outfile, fullpage=True)
print("", file=outfile)
##############################################

### Neuron inspection -- comparison with images ###
neuron_img_comp(somset, imgs, sampler, outpath=path.path)
for i in range(len(sampler.points)):
    nid = "-".join(str(s) for s in sampler.points[i])
    neuron_comp_latex(f"neuron_{i}", f"Neuron {nid}", f"neuron{i}", outfile)
    print("", file=outfile)
###################################################
