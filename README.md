## Connectome utilities
Provides utility to look up and analyze connectivity between neurons in a circuit model.

  - circuit_models: Provides data loaders on top of bluepy to load data specific to 
  BlueBrain models in sonata format.
    - circuit_models.neuron_groups: Provies utility to load the properties of neurons in
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
  provide the ability to create flat map volumes, but to get more out existing flat maps.
  This begins with creating "region images", i.e. images of flattened versions of a circuit,
  colored by region identity and extends to supersampling a flat map, i.e. going beyond its
  pixel resolution and even turning these arbitrary pixel coordinates into a um-based
  coordinate system.

  - io: Provides functionality to save / load multiple connection matrices into a single hdf5
  file. The matrices must be the values of a pandas.Series, indexed in any way.
  When such a Series is loaded again using this functionality, a lazy loading scheme is employed,
  i.e. the actual underlying matrices are only loaded when they are first actually accessed.
  This can be useful if you are only interested in a small number of the matrices in the Series.

### Installation
Simply run "pip install ."
Can then imported as "conntility".

### Using it
We can conceptually split a connectivity analysis into several parts:
  - **Loading neurons**: Start by looking up the neurons in a circuit model and their properties
  - **Filtering neurons**: Next, find the neurons you are really interested in by filtering out the uninteresting ones.
  - **Grouping neurons**: This step is optional. But maybe you want to analyze the result with respect to separate
  groups of neurons, such as per layer or mtype.
  - **Loading connectivity**: Next, load the connectivity between the neurons, yielding one or several connection
  matrices
  - **Analyzing connectivity**: Finally, run your analysis.

conntility provides python functions to simplify and standardize any of these steps. 

To enable flexible, reproducable and re-usable analyses, some of the functionality must be parameterized through the use of .json formatted configuration files. To understand their format and how to use them, refer to [this documentation](configuration_files.md). In that document, I will walk you through a number of use cases and explain the contents of the configurations along the way.

In the remainder of this document, I will simply provide an overview of all functions and their purpose.
