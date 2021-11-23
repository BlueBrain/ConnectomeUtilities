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
    mapped coordinates, supersampled or not, and neuron depth.
    NOTE: For convenience, it is possible to define which properties of which neurons are loaded, how
    they are grouped, and what filters to apply to them in a .json formatted config file.
    This makes it easy to get the same grouping across use cases. 
    It may also be helpful for command line applications that run a specific analysis on
    specifiable groups of neurons.

    - circuit_models.connection_matrix: Provides useful ways of looking up connection matrices.
    Either just a single matrix or matrices between groups of neurons. Or the counts of 
    connections between groups of neurons (reduced resolution connectome). 
    Can use any of the very flexible grouping schemes given by circuit_models.neuron_groups.

  - analysis: This is where functionality for analyzing connection matrices goes. Currently
  it only contains a function to execute a diffusion embedding of a connection matrix 
  (either sparse or dense).

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

  - randomization: Currently empty. This is where randomization functionality will go.

### Installation
Simply run "pip install ."
Can then imported as "conntility".
