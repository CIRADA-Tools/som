# som

This repository is a collection of code that has been useful in the development of and experimentation with the VLASS SOM. Some utilities have been directly incorporated into pyink (a requirement for this software), while others are included here.

Note: This code has been consolidated from multiples files in multiple locations. Some files may have an incomplete list of imports and some functions may rely on others that have been moved to different files.

## train_som.sh

The script used to train the SOM. Training is conducted in three stages in order to first establish the rough structure before improving the detail in each neuron.

## utils.py

General utilities to simplify certain processes, like reading in the CIRADA component catalogue.

⋅* `add_filename`: Convert a Component_name into a file name.
⋅* `load_catalogue`: Load the component catalogue (fits or csv) into an astropy Table or pandas DataFrame (pandas=True). Can apply flagging if desired.
⋅* `save_table`: Useful for saving a table when a column contains a tuple for the best-matching neuron (bmu) that needs to be unpacked into separate columns.
⋅* `create_wcs`: Add WCS coordinates to an image based on its central RA and DEC and its pixel size. Useful for IMG*bin images, as their WCS info has been stripped.

## diagnostics.py

A collection of function that can be used to measure properties of a sample tat has been mapped onto a SOM. This includes the median and 1-sigma bounds of the Euclidean distances, plots for the Euclidean distance distribution for both the ensemble and for a single neuron, etc.

## plotting_tools.py

Some utilities to assist in plotting, such as convenience functions to load in fits images, scale them between the 0.1 and 99.9 percentiles, and plot them.

Includes functions to prepare side-by-side panels of images with their preprocessed version for use in a Zooniverse project.

Supplies functions to create a 10x10 grid of images for each neuron for the Sidelobe SOM. This will likely be used again when someone attempts a SOM paper.

## som_inspection_doc.py

A potentially useful tool to consider training a new SOM. Plots various distributions (e.g. the Euclidean distance), then takes a subset of the neurons and plots images with low, medium, and high Euclidean distances to an individual neuron. Compiles all plots into a LaTeX document.

## ComplexXID

A collection of scripts that were originally intended to serve as the VLASS Complex Cross-Identification pipeline. The SOM was to be trained on "Complex" components. Each neuron would then be annotated with a 2D mask to specify the position of components belonging to the same source as the central component. The components are then fed through a somewhat complicated "collation" step in order to produce the source catalogue.

The pipeline for this process is in tact and generally works, though we had difficulty getting the SOM to produce results that were scientifically impactful. Namely, the SOM struggled to differentiate independent components from pairs.

The code here is certainly undercommented. The main process is in `vlass_sidelobe_pipeline.py` with the brunt of the work covered by `preprocessing.py`, `complex_xid_core.py`, and `collation.py`. Read through the code in that order and hopefully it is sufficiently understandable.