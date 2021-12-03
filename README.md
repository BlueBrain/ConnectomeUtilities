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

  - circuit_models: For interacting with and loading from circuit models
    - neuron_groups: Loading neuron properties from SONATA circuits and defining groups or subsets of them
      - *load_neurons*: Load specified neurons and return their properties in a DataFrame. In addition to basic properties (accessed through bluepy), it provides flatmapped locations, supersampled flat locations, any property in a volumetric atlas. 
      - *load_projection_locations*: Same, but for projection fibers. For a single specified projection.
      - *load_all_projection_locations*: Same, loads all projections existing in a Circuit model
      - *group_by_properties*: Define a group of neurons based on structural properties
      - *group_by_binned_properties*:  Same, but for numerical values (they are binned)
      - *group_by_grid*: Group by spatial locations, splitting space into hexagonal, triangular or rhombic grids
    
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
  - *flat_region_image*: Simple utility to generate a RGB image (N x M x 3 numpy.array) of a flat view of a model with regions in different colors.
  - *apply_flatmap_with_translation*: Simple utility that helps with looking up flat locations of projection fibers: Since they are placed outside the volume a direct lookup is impossible, so they will translated along their direction vectors until they hit the mapped volume.
  - *supersample_flatmap*: Return a copy of a flat map where the locations are no longer integer valued, but in floating point micrometers. That is, each voxel is associated with a flat location in um.
  - *supersampled_locations*: Supersamples (and transforms to um) the flat location of any 3d point or array of 3d points.
  - *estimate_flatmap_pixel_size*: Helper function to estimate the approximate size (in um) that corresponds to a single flatmap pixel.
  - *per_pixel_coordinate_transformation*: Very technical helper that provides access to intermediate coordinate system between the original 3d coordinates and the supersampled flat coordinates, such as local coordinates "within" each pixel.

- *io*: Input/Output
  - *logging*: Centralized logger for this package
  - *write_toc_plus_payload*: Helper for storing multiple scipy.sparse matrices in a single .hdf5 file in a very efficient way
  - *read_toc_plus_payload*: Helper for reading back the stored matrices. Implements lazy reading for potential speedup.

- *analysis*: Making it easier / faster / better reproducible to apply connectivity analyses
  - *Analysis*: A class instantiating a single connectivity analysis. Can import the analysis dynamically from any file at runtime. Configured through a .json config file. For details, see [this tutorial](configuration_files.md).
  - *get_analyses*: Helper to rapidly read a number of analyses defined in a config.
  - *library*: In principle, individual atomic analyses are not supposed to be in this repository. But I put it here temporarilily until it is adopted by a different repo.
    - *embed_pathway*: Perform diffusion embedding on a connection matrix (weighted or unweighted, sparse or dense).
  - *analysis_decorators*: Decorators to turn atomic analysis of a connection matrix into more powerful ones. For details, see [this tutorial](configuration_files.md).

  - *ConnectivityMatrix*: High-level class that defines the connectivity of a population of neurons as well as the properties of the neurons. Provides access to multiple connectivity properties, such as strength or weight. Provides powerful filtering functions and generation of stochastic control samples. Best used with loader configs and analysis configs (see [this tutorial](configuration_files.md))
  - *TimeDependentMatrix*: Represents a ConnectivityMatrix where connection properties (weights) change over time.
  - *ConnectivityGroup*: Represents a group of ConnectivityMatrices that are subpopulations of a single larger population.
