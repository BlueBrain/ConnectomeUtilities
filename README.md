# Connectome utilities

Complex network representation and analysis layer

[![DOI](https://zenodo.org/badge/641456590.svg)](https://zenodo.org/badge/latestdoi/641456590)

![ConnectomeUtilities](https://github.com/BlueBrain/ConnectomeUtilities/blob/main/banner-ConnectomeUtilities.jpg?raw=true)

# Table of Contents

1. [What is Connectome Utilities?](#what-is-connectome-utilities-and-what-does-it-provide)
    * [Usage with Sonata-based models](#usage-with-sonata-based-models)
    * [A Non-sonata based example](#a-non-sonata-based-example)
    * [Summary](#summary)
2. [Installation](#installation)
3. [Examples](#examples)
4. [Further information](#further-information)
    * [Contents and overview](#contents-and-overview)
    * [Functionality](#functionality)
    * [Defining analyses in .json configurations](configuration_files.md)
5. [Contribution guidelines](CONTRIBUTING.md)
6. [Citation](#citation)
7. [Acknowledgements and Funding](#acknowledgements--funding)

## What is Connectome Utilities and what does it provide?

The purpose of Connectome Utilities is to simplify running topological analyses on detailed models of networks by providing a bridge between existing analyses and the model representation. The purpose is not to provide the analyses themselves, there are great existing packages and solutions for that. But to simplify their application to the case of complex, non-homogeneous networks associated with interdependent node and edge properties. This comes in the form of two types of functionality: First, loading complex connectomes into a reduced representation that still keeps salient details. Second, automate standard operations, such as extraction of specific subnetworks and generation of statistical controls.

With respect to the first point, loading from [SONATA](https://github.com/BlueBrain/libsonata) models is provided. But once loaded, the representation is independent from Sonata and the second point provides utility also for non-Sonata networks.

### Usage with Sonata-based models
For illustration, we will provide some examples from the field of biologically detailed models of neural circuits, although the methods themselves could be useful in different fields and for more simplified neuronal networks as well.

- [Example 1](https://github.com/BlueBrain/ConnectomeUtilities/blob/main/examples/Examples%201%20and%202%20-%20Analyzing%20pathways%20and%20controls.ipynb): The network of a circuit model is to be analyzed, but the user wants to analyze the various neuron types separately, as it is known that their connectivities are very different. Connectome Utilities provides the automatic application of the same analysis to separate pathways and comparison of the results.

- [Example 2](https://github.com/BlueBrain/ConnectomeUtilities/blob/main/examples/Examples%201%20and%202%20-%20Analyzing%20pathways%20and%20controls.ipynb): A subnetwork of neurons has been identified in the network. The user found that its connectivity is "non-random" with respect to a metric.  But can it be explained by considering a more complex baseline model? Connectome Utilities handles the generation of complex control models, application of analyses, and comparison of results. Note that (like the analyses) the control connectivity model needs to be user-provided, but Connectome Utilities simplifies its execution and analysis.

- [Example 3](https://github.com/BlueBrain/ConnectomeUtilities/blob/main/examples/Example%203%20-%20Generating%20control%20subsamples%20and%20neighborhoods.ipynb): Building on the previous example, the user found that the non-random connectivity trend cannot be explained by complex controls. It is possible that the non-random aspect be explained by the neuronal composition of the subnetwork, e.g. it may be biased towards sampling interneurons. Alternatively, it can be more deeply rooted. To decide this, Connectome Utilities provides the sampling of random control subnetworks from the base network with the same distribution of neuron types as the original sample or the same distribution of neuron locations. It also facilities running the same analysis on the data and controls and comparing their results.

- [Example 4](https://github.com/BlueBrain/ConnectomeUtilities/blob/main/examples/Example%204%20-%20Loading%20atlas%20data.ipynb): A large network model is to be analyzed. The user wants to analyze it in the spatial context provided by the voxelized brain atlas, e.g. are connections more common within than across regions? Connectome Utilities provides loading data from brain atlases and cross-referencing them with network models.

- [Example 5](https://github.com/BlueBrain/ConnectomeUtilities/blob/main/examples/Example%205%20-%20Connectivity%20at%20reduced%20resolution.ipynb): A large micro-scale network model has been loaded that exists in a spatial context. The user wants to know what the network looks like at reduced resolution, such as the voxelized meso-scale, or pathways between layers. Connectome Utilities provides easy partition of networks according to spatial or other properties and generation of (reduced scale) quotient graphs. 

- [Example 6](https://github.com/BlueBrain/ConnectomeUtilities/blob/main/examples/Example%206%20-%20Plastic%20matrices.ipynb): A micro-scale network model is to be analyzed. But as it represents a neuronal network, it changes over time, due to plasticity. Connectome Utilities allows the representation and analysis of networks that change, both structurally or functionally (i.e. only the weights).

### Non-sonata based examples
- [Non-sonata Example 1](https://github.com/BlueBrain/ConnectomeUtilities/blob/main/examples/C%20elegans%20-%20a%20non-sonata-based%20example.ipynb): This is an example of loading connectomics data not from a Sonata model, but instead from an Excel file of the connectivity of the worm at different developmental stages. Source of the data: [Witvliet et al., 2021](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8756380/) and wormatlas.org. The input data is not included in this repository, but instructions on how to download it are given in the notebook.

- [Non-sonata Example 2](https://github.com/BlueBrain/ConnectomeUtilities/blob/main/examples/Fly%20connectome%20-%20a%20non-sonata%20based%20example.ipynb): This is another example of loading connectomics data not from a Sonata model. It loads R-formatted (.rds) data on Drosophila connectivity. Source of the data: [Mehta et al., 2023](https://direct.mit.edu/netn/article/7/1/269/113338/Circuit-analysis-of-the-Drosophila-brain-using). The input data is not included in this repository, but instructions on how to download it are given in the notebook.

### Summary

In summary, when networks are more complex than just nodes and unweighted edges, more confounders have to be considered, and certain decisions have to be made during analysis. We provide the package that makes these decisions explicit and provides a sane default for them. Additionally, the package defines and uses a configuration file format formalizing analyses that combine any or all the complexities outlined above into a json file. This facilitates the reproduction of identical analyses in different networks.

The above is centered around a "ConnectivityMatrix" class and various derived classes, that provide a powerful and detailed representation of a complex network. It also provides save and load operations, enabling efficient data sharing. Some additional, independent functionality is provided:

- Functionality to extract the representation of topographical mapping of cortical connectivity, as parameterized in the [BBP "White Matter" project](github.com/BlueBrain/Long-range-micro-connectome).
- Functionality to represent connectivity at multiple spatial scales simultaneously.
- Functionality to rapidly calculate path distances between synapses on neuronal morphology in a Sonata model.

## Installation
Simply run 
  > pip install .

All dependencies are declared in the setup.py and are available from [pypi](https://pypi.org/)

The package can then be imported as "conntility".
See the various exemplary notebooks for more details.

## Examples
Usage examples can be found in the jupyter notebooks listed [above](#usage-with-sonata-based-models)

## Further information
### Contents and Overview

  - circuit_models: Provides data loaders on top of bluepy to load data specific to 
  BlueBrain models in sonata format.
    - circuit_models.neuron_groups: Provides utility to load the properties of neurons in
    a circuit model, and group them according to various schemes:
      - by values of their properties (by etype, layer, mtype...)
      - by binned values of their properties (by coordinates)
      - by grouping them into a 2d grid (most useful in conjunction with flat coordinates)
    In addition to the default properties that already bluepy can load, it can add flat 
    mapped coordinates, supersampled or not, and neuron depth. It can further add properties
    looked up from any voxel atlas (looking it up by neuron location) and membership in any
    predefined group of neurons as a boolean property. For details, see below.

    - circuit_models.connection_matrix: Provides useful ways of looking up connection matrices.
    Either just a single matrix or matrices between groups of neurons. Or the counts of 
    connections between groups of neurons (reduced resolution connectome). 
    Can use any of the very flexible grouping schemes given by circuit_models.neuron_groups.

  - analysis: Provides functionality for simplifying and automating the analysis of connection matrices. 
  While the basic, atomic analyses are _not_ part of this repository, it provides functionality for
  easily applying it to multiple matrices, to submatrices defined by neuron properties (such as layer, mtypes, etc.)
  and comparing it to random controls.

  - flatmapping: Provides functionality on top of existing flat maps. That is, it does NOT
  provide the ability to create flat map volumes, but to get more out of existing flat maps.
  This begins with creating "region images", i.e. images of flattened versions of a circuit,
  colored by region identity, and extends to supersampling a flat map, i.e. going beyond its
  pixel resolution and even turning these arbitrary pixel coordinates into an um-based
  coordinate system.

  - io: Provides functionality to save / load multiple connection matrices into a single hdf5
  file. The matrices must be the values of a pandas.Series, indexed in any way.
  When such a Series is loaded again using this functionality, a lazy loading scheme is employed,
  i.e. the actual underlying matrices are only loaded when they are first actually accessed.
  This can be useful if you are only interested in a small number of the matrices in the Series.


### Functionality
We can conceptually split a connectivity analysis into several parts:
  - **Loading neurons**: Start by looking up the neurons in a circuit model and their properties
  - **Filtering neurons**: Next, find the neurons you are really interested in by filtering out the uninteresting ones.
  - **Grouping neurons**: This step is optional. But maybe you want to analyze the result with respect to separate
  groups of neurons, such as per layer or mtype.
  - **Loading connectivity**: Next, load the connectivity between the neurons, yielding one or several connection
  matrices
  - **Analyzing connectivity**: Finally, run your analysis.

conntility provides python functions to simplify and standardize any of these steps. 

To enable flexible, reproducible, and re-usable analyses, some of the functionality must be parameterized through the use of .json formatted configuration files. To understand their format and how to use them, refer to [this documentation](configuration_files.md). In that document, I will walk you through a number of use cases and explain the contents of the configurations along the way.

In the remainder of this document, I will simply provide an overview of all functions and their purpose.

  - circuit_models: For interacting with and loading from circuit models
    - neuron_groups: Loading neuron properties from SONATA circuits and defining groups or subsets of them
      - *load_neurons*: Load specified neurons and return their properties in a DataFrame. In addition to basic properties (accessed through bluepy), it provides flatmapped locations, supersampled flat locations, and any property in a volumetric atlas. 
      - *load_projection_locations*: Same, but for projection fibers. For a single specified projection.
      - *load_all_projection_locations*: Same, loads all projections existing in a Circuit model
      - *group_by_properties*: Define a group of neurons based on structural properties
      - *group_by_binned_properties*:  Same, but for numerical values (they are binned)
      - *group_by_grid*: Group by spatial locations, splitting space into hexagonal, triangular, or rhombic grids
    
      - *load_filter*: Access to neuron property loading (see above), configured through a .json config
      - *load_group_filter*: Same, but including access to the grouping functionality
    
    - *simulation_conditions*: Small utility to return simulation conditions (as found in BlueConfig.json) from a Simulation object
    - *input_spikes*: Small utility to return input spikes given to a Simulation in the same format as output spikes (pandas.Series)
    - *input_innervation*: Calculates how strongly any neuron is innervated by input (or recurrent!) spikes in a specified time window of a Simulation
    - *input_innervation_from_matrix*: Same, but provide custom connectivity matrix

    - *circuit_connection_matrix*: Most useful: Look up a connectivity matrix from a Circuit object for any pre- post-synaptic population, including projection fibers; for local connectivity or any projection.
    - *connection_matrix_for_gids*: Same, but directly specify the sonata connectome file instead of Circuit object.
    - *circuit_group_matrices*: Uses the grouping functionality mentioned above; Looks up connection matrices *within* specified groups of neurons.
    - *circuit_cross_group_matrices*: Similar to the previous. But in addition to the matrices within groups (A->A), also returns matrices between groups (A -> B).
    - *circuit_matrix_between_groups*: Uses the grouping functionality mentioned above: Count the number of connections between specified groups of neurons.

- *flatmapping*: Tools to get more out of flat maps. Mostly supersampling
  - *flat_region_image*: Simple utility to generate an RGB image (N x M x 3 numpy.array) of a flat view of a model with regions in different colors.
  - *apply_flatmap_with_translation*: Simple utility that helps with looking up flat locations of projection fibers: Since they are placed outside the volume a direct lookup is impossible, so they will be translated along their direction vectors until they hit the mapped volume.
  - *supersample_flatmap*: Return a copy of a flat map where the locations are no longer integer valued, but in floating point micrometers. That is, each voxel is associated with a flat location in um.
  - *supersampled_locations*: Supersamples (and transforms to um) the flat location of any 3d point or array of 3d points.
  - *estimate_flatmap_pixel_size*: Helper function to estimate the approximate size (in um) that corresponds to a single flatmap pixel.
  - *per_pixel_coordinate_transformation*: Very technical helper that provides access to the intermediate coordinate system between the original 3d coordinates and the supersampled flat coordinates, such as local coordinates "within" each pixel.

- *io*: Input/Output
  - *logging*: Centralized logger for this package
  - *write_toc_plus_payload*: Helper for storing multiple scipy.sparse matrices in a single .hdf5 file in a very efficient way
  - *read_toc_plus_payload*: Helper for reading back the stored matrices. Implements lazy reading for potential speedup.

- *analysis*: Making it easier / faster / better reproducible to apply connectivity analyses
  - *Analysis*: A class instantiating a single connectivity analysis. Can import the analysis dynamically from any file at runtime. Configured through a .json config file. For details, see [this tutorial](configuration_files.md).
  - *get_analyses*: Helper to rapidly read a number of analyses defined in config.
  - *library*: In principle, individual atomic analyses are not supposed to be in this repository. But I put it here temporarily until it is adopted by a different repo.
    - *embed_pathway*: Perform diffusion embedding on a connection matrix (weighted or unweighted, sparse or dense).
  - *analysis_decorators*: Decorators to turn atomic analysis of a connection matrix into more powerful ones. For details, see [this tutorial](configuration_files.md).

  - *ConnectivityMatrix*: High-level class that defines the connectivity of a population of neurons as well as the properties of the neurons. Provides access to multiple connectivity properties, such as strength or weight. Provides powerful filtering functions and generation of stochastic control samples. Best used with loader configs and analysis configs (see [this tutorial](configuration_files.md))
  - *TimeDependentMatrix*: Represents a ConnectivityMatrix where connection properties (weights) change over time.
  - *ConnectivityGroup*: Represents a group of ConnectivityMatrices that are subpopulations of a single larger population.

## Citation
If you use this software, kindly use the following BIBTEX entry for citation:

```
@article {Isbister2023.05.17.541168,
	author = {James B. Isbister and Andr{\'a}s Ecker and Christoph Pokorny and Sirio Bola{\~n}os-Puchet and Daniela Egas Santander and Alexis Arnaudon and Omar Awile and Natali Barros-Zulaica and Jorge Blanco Alonso and Elvis Boci and Giuseppe Chindemi and Jean-Denis Courcol and Tanguy Damart and Thomas Delemontex and Alexander Dietz and Gianluca Ficarelli and Mike Gevaert and Joni Herttuainen and Genrich Ivaska and Weina Ji and Daniel Keller and James King and Pramod Kumbhar and Samuel Lapere and Polina Litvak and Darshan Mandge and Eilif B. Muller and Fernando Pereira and Judit Planas and Rajnish Ranjan and Maria Reva and Armando Romani and Christian R{\"o}ssert and Felix Sch{\"u}rmann and Vishal Sood and Aleksandra Teska and Anil Tuncel and Werner Van Geit and Matthias Wolf and Henry Markram and Srikanth Ramaswamy and Michael W. Reimann},
	title = {Modeling and Simulation of Neocortical Micro- and Mesocircuitry. Part II: Physiology and Experimentation},
	elocation-id = {2023.05.17.541168},
	year = {2023},
	doi = {10.1101/2023.05.17.541168},
	publisher = {Cold Spring Harbor Laboratory},
	\
	URL = {https://www.biorxiv.org/content/early/2023/05/23/2023.05.17.541168},
	eprint = {https://www.biorxiv.org/content/early/2023/05/23/2023.05.17.541168.full.pdf},
	journal = {bioRxiv}
}
```

## Acknowledgements & Funding
The development of this software was supported by funding to the Blue Brain Project, a research center of the École polytechnique fédérale de Lausanne (EPFL), from the Swiss government’s ETH Board of the Swiss Federal Institutes of Technology.

Copyright (c) 2023 Blue Brain Project / EPFL.
