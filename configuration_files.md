###
Much of the functionality of *Connectome utilities* requires the use of .json formatted configuration files. In this document, I will explain how they are formatted and used. To that end, I will walk you through a number of examples, from simple to more complex use cases.

In general, we use two types of configuration files:
  - **Loader configurations**: Define which set of neurons to load, and (optionally) how to group them into subsets
  - **Analysis configuration**: Defines which analysis to run on a connection matrix, and where to find the code for that analysis.

First, let's look at the *loader configuration*

#### Loader configuration
A loader configuration is an abstract description of a group of neurons, and a list of properties of interest. That is,
it allows you to describe which neurons you want to analyze, and which of their properties are relevant for your analysis.
The file is .json formatted containing up to three entries: *loading*, *filtering* and *grouping*. Any of these entries is
optional. 

Let's begin with a simple example of *loading* and *filtering* only:
```
{
    "loading": {
        "properties": ["x", "y", "z", "etype", "mtype", "layer"],
    },
    "filtering": [
        {
            "column": "etype",
            "values": ["bNAC", "cNAC"]
        },
        {
            "column": "layer",
            "value": 3
        }
    ]
}
```
Here, the *loading* block contains a singe entry "*properties*" that specifies the list of neuron properties to load -
the x, y, z coordinates, their e-types, m-types and layers. What other properties are available? Unfortunately, that 
depends on the circuit model - different models expose different properties. For the SSCx-v7 model this list is:
```
'etype', 'exc_mini_frequency', 'inh_mini_frequency', 'layer',
'me_combo', 'morph_class', 'morphology', 'mtype', 'orientation',
'region', 'synapse_class', 'x', 'y', 'z'
```
In addition to these basic properties directly exposed by the circuit model, four more *derived properties* are available:
```
'flat_x', 'flat_y', 'ss_flat_x', 'ss_flat_y'
```
*flat_x*, *flat_y* are the x, y coordinates in a flattened coordinate system (top-down view). They are given in an integer-valued
coordinate system with no specific units. *ss_flat_x*, *ss_flat_y* are given in a float-valued coordinate-sysyem in units 
that are roughly one micrometer. In order to load the derived properties certain prerequisites need to be fulfilled:
  - For *flat_x*, *flat_y* a **flatmap** atlas needs to exist in the circuits atlas directory. Additionally, the basic properties
  'x', 'y', 'z' must be loaded.
  - For *ss_flat_x*, *ss_flat_y* a **flatmap** atlas and an **orientation** atlas both need to exist in the circuits atlas directory.

**Note**: "properties" and in fact the entire "loading" block are optional. If you leave it out, then *all* available properties are
going to be loaded.


Next, the *filtering* block describes the subset of neurons you are interested. It is a list of of conditions and neurons failing
*any* of the conditions will be filtered out. That is, in the example above only "bNAC" and "cNAC" neurons in layer 3 will remain!

The subset is described by specifying the neuron property based on which you want to filter ("column") and the range
of valid values (here: "values"). The "*column*" given **must** be one of the properties that is specified in "loading". The valid
values can be specified in several ways:
  - A list of valid values:
  ```
  "values": *list*
  ```
  - A single valid value:
  ```
  "value": *scalar value*
  ```
  - For numerical properties: A valid interval:
  ```
  "interval": [*min value*, *max value*]
  ```

#### Using a loader config to load a connection matrix
Let's use our example loader config above to load a connection matrix:
```
import bluepy
from conntility.connectivity import ConnectivityMatrix

CIRC_FN =  "/gpfs/bbp.cscs.ch/project/proj83/circuits/Bio_M/20200805/CircuitConfig"
circ = bluepy.Circuit(CIRC_FN)

M = ConnectivityMatrix.from_bluepy(circ, loader_cfg)

[...]

print(M)
<conntility.connectivity.ConnectivityMatrix at 0x7ff4d1236700>

```

Here, "*loader_cfg*" is assumed to be either the path to a json file holding the example above, or the example itself,
i.e. a dict object with the *loading* and *filtering* entries. Both are supported.

Extracting the connectivity will take a minute, but you have a progressbar to keep you company.

The resulting object, *M* will give you access to the loaded neuron properties and the connectivity between the neurons.
First, we can list which properties have been loaded:
```
print(M.vertex_properties)
['etype' 'layer' 'mtype' 'x' 'y' 'z']
```
As specified by the loader config, we have the three coordinates, etype, mtype and layer. These can be directly accessed as
properties of the object:
```
print(M.etype)
['cNAC', 'cNAC', 'cNAC', 'cNAC', 'cNAC', ..., 'bNAC', 'bNAC', 'bNAC', 'bNAC', 'bNAC']
Length: 17570
Categories (11, object): ['bAC', 'bIR', 'bNAC', 'bSTUT', ..., 'cNAC', 'cSTUT', 'dNAC', 'dSTUT']

print(M.layer)
[3 3 3 ... 3 3 3]
```
As specified by the filtering, all neurons are in layer 3 and have an etype of either 'cNAC' or 'bNAC'.

The number of neurons is the length of the object:
```
print(len(M))
17570
```

##### Accessing matrices and submatrices
But what about the connectivity? It can be accessed as a sparse matrix, dense matrix or numpy.array as a property of the object.
The order of the rows / columns of the matrix matches the order of the entries of the loaded properties (layer, etype, etc.)
```
M.matrix 
<17570x17570 sparse matrix of type '<class 'numpy.bool_'>'
        with 17483 stored elements in COOrdinate format>

M.dense_matrix
matrix([[False, False, False, ..., False, False, False],
        [False, False, False, ..., False, False, False],
        [False, False, False, ..., False, False, False],
        ...,
        [False, False, False, ..., False, False, False],
        [False, False, False, ..., False, False, False],
        [False, False, False, ..., False, False, False]])

M.array
array([[False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       ...,
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False]])
```
You always have access to a global identifier of each of the neurons in the matrix:
```
print(M.gids)
[462330 462334 462341 ... 522514 522516 522517]

``` 
You can use the gids to define subpopulations or directly access submatrices:
```
print(M.subpopulation(M.gids[1000:2000]))
<conntility.connectivity.ConnectivityMatrix object at 0x7fffcb6fa9a0>  # object defining the subpopulation

print(len(M.subpopulation(M.gids[1000:2000])))
1000  # Because 2000 - 1000 = 1000

A = M.subpopulation(M.gids[1000:2000]).matrix  # matrix of the subpopulation
B = M.submatrix(M.gids[1000:2000])  # or direct access to the submatrix
print((A == B).mean())
1.0  # They are identical.
```

##### Subpopulations based on properties
Still, defining subpopulations by gid is a rather manual process. You first have to figure out which gids you are interested in
after all. There is functionality to help you with that. You can perform filtering based on neuron properties:
```
subpopulation = M.index("etype").eq("bNAC")  # Define a subpopulation based on "etype" and return the one equal to "bNAC"
print(len(subpopulation))  # Only 7212 of them are bNAC
7212

print(subpopulation.matrix)  # And the matrix of the subpopulation 
<7212x7212 sparse matrix of type '<class 'numpy.bool_'>'
        with 2474 stored elements in COOrdinate format>
```
This functionality works by first specifying which property you want to use in the call to .index; then give the range of valid values.
In a nutshell, it works similar to the *filtering* keyword in the loader config.
The range of valid values can be specified in different ways, each corresponding to a different function call:
  - .eq: *equal*. See above, matches a specified value. Works like using "value" in the *filtering* stage of a loader config
  - .isin: Matches a list of valid values. Works like using "values" in *filtering*.
  - .lt: *less than*. Only works with numerical values
  - .le: *less or equal*
  - .gt: *greater than*
  - .ge: *greater or equal*.
The equivalent of using "interval" in *filtering* is a chain of .lt and .ge.

##### Random subpopulations
Additionally, you can generate *constrained random subpopulations* that can serve as stochastic controls. For example:
```
indexer = M.index("mtype")
reference = M.subpopulation(M.gids[:1000])  # As an example, simply the first 1000 neurons.
rnd_sample = indexer.random_categorical(reference.gids)
print(len(rnd_sample))
1000
```
generates a random subsample of *M* with the same length as the reference_gids (here simply the first 1000 gids). Additionally, the random 
sample is constrained in the its distribution of "*mtype*" values will match the reference distribution!
```
(reference.mtype.value_counts() == rnd_sample.mtype.value_counts()).all()
True
```
For categorical properties use .random_categorical. For numerical values use .random_numerical. In that case, the distribution of the numerical value in discrete bins will match the reference. The number of bins can be specified:
```
rnd_sample = M.index("x").random_numerical(reference.gids, n_bins=10)
rnd_sample.x.mean(), rnd_sample.x.std()
(4389.180429069136, 1138.2503319186505)

reference.x.mean(), reference.x.std()
(4390.739357584497, 1143.8032496901749)  # Approximate match
```

#### More features of the loader configuration
Additionally, the following entries can be added to a loader config. They are of course all optional:
**base-target**
```
{
    "loading": {
        "base_target": "central_column_4_region_700um", 
    [...]
}
```
The *base_target* limits the neurons to a pre-defined group. This is, of course, similar to filtering. It acts differently though in that the target needs to have been pre-defined, and the filtering is applied *before* anything is loaded. Thus, it can be faster than applying a filter afterwards.

**atlas**
```
{
    "loading": {
        "atlas": [
            {"data": "distance", "properties": ["distance"]}
        ],
    [...]
```
The *atlas* loads additional data from one or several voxel atlases in .nrrd format. Value is a list of dicts that each specify the atlas to use ("data") and what to call the resulting property or properties ("properties"). What this does is to look up for each neuron the value in the atlas at the neurons x, y, z location. Therefore, to use this feature, the "x", "y" and "z" properties **must** also be loaded!

Note that the value of "properties" must be a list. This is because a voxel atlas can associate each voxel with more than one value (multi-dimensional atlas). The number of entries in the list must match the number of values associated with a voxel in the atlas.

**groups**
```
{
    "loading": {
        "groups":[
            {
                "filtering": [{"column": "etype", "values": ["bNAC", "cNAC"]}],
                "name": "is_bNAC_cNAC"
            }
        ]
    [...]
```
This associates an additional property with the neurons based on whether they pass afiltering check or not. The syntax for the entry under "filtering" is equal for the syntax of "filtering" at the root level. And it works in the same way: It defines a list of tests that a neuron passes or not. Only that non-passing neurons are not removed. Instead, a new property is added to the neurons, which is *True* if all checks are passed and False otherwise. This can be thought of as ad-hoc defined groups of neurons, where the value of the property denotes membership in the group. The name of the property to be added is explicitly set (*"name"*).

**include**
```
{
    "loading": {
        "groups":[
            {"include": "my_group.json"}
        ]
    [...]
```
The *include* keyword can be used at *any* location in a loader config. The way it works is very simple: If any element in the config is a dict with a single entry that is called "include", then the entire dict will be replaced with the contents of the .json file referenced by the value of "include". That is, if the contents of "my_group.json" is:
```
{
    "filtering": [{"column": "etype", "values": ["bNAC", "cNAC"]}],
    "name": "is_bNAC_cNAC"
}
```
Then the resulting behavior of the example using "include" is equal to the example above that.

The "include" allows you to re-use certain part of a config file that are useful without resorting to copy-paste. This can be used anywhere, but it is most useful for storing custom groups of neurons that cannot be readily assembled from their structural properties. Assemblies detected from simulated spiking activity is one possible use case.

To do this, simply perform "filtering" based on the column called "gid". The "gid" is a unique identifier of a neuron and is always loaded. Therefore, you can specify any group by giving a list of valid "gid"s:
```
# Contents of my_group.json
[
    {
        "filtering": [{"column": "gid", "values": 
            [521473, 468766, 477276, 483514, 495251, 499767, 520998, 474925,
                476827, 521806, 497750, 497824, 494201, 474007, 485759, 500136,
                502532, 470080, 482381, 477501]
                }],
        "name": "in_assembly1"
    },
    {
        "filtering": [{"column": "gid", "values": [517953, 485154, 479110, 482221, 485291, 506824, 469236, 488711,
            493455, 495871, 492482, 486045, 521636, 479584, 503382, 463251,
            521711, 488329, 463244, 502105]
            }],
        "name": "in_assembly2"
    }
]
```
Since the list of gids can grow very large it makes the file hard to read as a human. Therefore, it might be a good idea to keep these groups in a separate file that is referenced through an "include".


**Let's put all of that together**:
```
{
    "loading": {
        "base_target": "central_column_4_region_700um", 
        "properties": ["x", "y", "z", "etype", "mtype", "layer"],
        "atlas": [
            {"data": "distance", "properties": ["distance"]}
        ],
        "groups":[
            {
                "include": "my_group.json"
            },
            {
                "filtering": [{"column": "etype", "values": ["bNAC", "cNAC"]}],
                "name": "is_bNAC_cNAC"
            }
        ]
    },
    "filtering": [
        {
            "column": "etype",
            "values": ["bNAC", "cNAC"]
        },
        {
            "column": "layer",
            "value": 3
        }
    ]
}
```
```
M = ConnectivityMatrix.from_bluepy(circ, loader_cfg)
print(len(M))
1051
print(M.vertex_properties)
array(['etype', 'layer', 'mtype', 'x', 'y', 'z', 'distance',
       'in_assembly1', 'in_assembly2', 'is_bNAC_cNAC'], dtype=object)
```
We see that fewer neurons than in the earlier example are loaded, because we limited everything to the target "central_column_4_region_700um".

We also have additional vertex (neuron) properties: "distance" comes from the "atlas" property loaded, "in_assembly1", "in_assembly2" and "is_bNAC_cNAC" are boolean properties of group membership.

Now we can for example generate random control samples that match the "distance" distribution of our pre-defined assemblies:
```
assembly1 = M.index("in_assembly1").eq(True)
rnd_sample = M.index("distance").random_numerical(assembly1.gids)
```

#### Grouping in a loader config
So far, the general approach outlined was to load the entire connectivity matrix of a population, then access submatrices of interest using the .index function. There is one more tweak that allows you to define the submatrices you are interested in already in the loader config:
```
{
    "loading":
    [...]
    "filtering":
    [...]
    "grouping": [
        {
            "method": "group_by_properties",
            "columns": ["mtype", "etype"]
        }
    ]
}
```
The "grouping" keyword defines groups of neurons where you are interested in their submatrices. The value of "grouping" is a list, where each entry of the list yields a partition of the neurons into subgroups. The final groups used are then the product of the individual partitions, i.e. the intersections of all combinations of partitions.

The above example simply partitions neurons into groups based on their values of "mtype" and "etype", i.e. one group would be "L23_BP, bNAC", another "L23_NBC, cNAC". As a side note, this is equavalent, to a list of two separate groupings, one by "mtype", one by "etype" (but more compact):
```
{
    [...]
    "grouping": [
        {
            "method": "group_by_properties",
            "columns": ["mtype"]
        },
        {
            "method": "group_by_properties",
            "columns": ["etype"]
        }
    ]
}
```

#### Using a grouped loader config
```
from conntility.connectivity import ConnectivityGroup

G = ConnectivityGroup.from_bluepy(circ, loader_cfg)

print(G.index)
MultiIndex([( 'L23_BP', 'bNAC'),
            ( 'L23_BP', 'cNAC'),
            ('L23_BTC', 'bNAC'),
            ('L23_BTC', 'cNAC'),
            ('L23_CHC', 'cNAC'),
            ('L23_DBC', 'bNAC'),
            ('L23_LBC', 'bNAC'),
            ('L23_LBC', 'cNAC'),
            ( 'L23_MC', 'bNAC'),
            ( 'L23_MC', 'cNAC'),
            ('L23_NBC', 'bNAC'),
            ('L23_NBC', 'cNAC'),
            ('L23_NGC', 'bNAC'),
            ('L23_NGC', 'cNAC'),
            ('L23_SBC', 'bNAC')],
           names=['idx-mtype', 'idx-etype'])

print(G["L23_BP", "bNAC"])
<conntility.connectivity.ConnectivityMatrix at 0x7ff4cba65850>

print(G["L23_BP", "bNAC"].matrix.shape)
(70, 70)
```
As we can see, this results in an object that contains the subpopulations of interest. It can be indexed by the "mtype" and "etype" of the subpopulations, corresponding to the properties we have specified in the "grouping" of the loader config. Indexing returns a representation of the subpopulation with all the features described above.

At this point, this is all the *"ConnectivityGroup"* can do, but more features are planned in the future.

#### Reduced loader config
As a note: In the examples above I assumed that a loader config exists as a separate .json file that is referenced in a function call. Instead, one can **always** also directly specify the dict that is otherwise contained in the file.

Additionally, large parts of the config can be left out completely. Leaving out "filtering" applies no filters; leaving out "loading" will load all available properties etc. If you want to specify only one of "loading", "filtering" or "grouping", you can leave out that keyword completely and just provide its value:
```
reduced_config = {  # This will be understood to be the contents of "loading":
        "base_target": "central_column_4_region_700um", 
        "properties": ["x", "y", "z", "etype", "mtype", "layer"]
}
M = ConnectivityMatrix.from_bluepy(circ, reduced_config)
```
### Analyzing connectivity matrices
At the root of all analysis is what we call *atomic analysis functions*. This can be any function that takes as inputs:
  - A scipy.sparse.matrix (M x M) specifying the connectivity of a group of neurons
  - A pandas.DataFrame (length M) with columns specifying neuron properties
  - any number of additional arguments (*args)
  - any number of keyword arguments (**kwargs)
and returns either
  - A scalar value
or
  - A pandas.Series

An example of the first return type would be an analysis that returns the average connection probability, an example of the second would be the number of simplices in all dimensions, returned as a Series with dimension as index.

#### Decorating atomic analyses
Conntility provides functionality to "get more" from such atomic analyses. To that end it provides function decorators that turn it into a more complicated, more involved analysis. Currently, the following decorators exist:
  - **grouped_by_grouping_config**: This decorator uses a reference to a *loader config* and accesses the "grouping" entry in the config. Then it turns any analysis into one that is instead applied separately to the submatrices of the groupings defined by the config.
  - **grouped_by_filtering_config**: This works very similarly to the previous one, but allows a bit more flexibility. It takes a list of *loader configs* and acceses their "filtering" entries. Then, each thusly defined filtering will be considered a group comprising the neurons passing the filter. Then the submatrices of these groups will be separately analyzed
  - **control_by_randomization**: This takes an analysis and performs it once on the actual matrix, then several times on a randomized shuffled control. Then returns the actual result and the mean of the controls. To generate the controls, it takes a reference to a randomization function that can exist anywhere on the file system.

You can find examples for all of these further down.
#### Dynamic import of analyses
Conntility provides functionality to import an atomic analysis or randomization function from any file at runtime. That is, the analysis does not need to be part of any package that is installed anywhere. This allows the user to quickly make use of the analysis without just copy-pasting it into their own code.

This is a separation of the actual payload of an analysis and the tubing and infrastructure required to efficiently use it - Think about the atomic analysis as a bullet and conntility as the gun.

This functionality requires the analysis be specified in an **analysis config**, i.e. where the file can be found, which function within the file to use and what additional arguments it might use.
Just as before, the *analysis config* can either exist as a separate .json file, or be directly provided as a dict.

To explain the format, I will walk you through some examples:

#### Examples

##### grouped_by_grouping_config
As an example, let's consider simplex counts. First the basic analysis:
```
from conntility.analysis import Analysis
from conntility.connectivity import ConnectivityMatrix

M = ConnectivityMatrix.from_bluepy(circ, loader_cfg)  # As in the previous examples above

analysis_specs = {
          "source": "/gpfs/bbp.cscs.ch/project/proj83/home/sood/analyses/manuscript/topological-analysis-subvolumes/topologists_connectome_analysis/library/topology.py",
          "method": "simplex-counts",
          "output": "scalar"
        }
A = Analysis("simplex_counts", analysis_specs)
A.apply(M.matrix)

dim
0   1051
1   1150
2   ...
...
```
Of course you can manually just apply the analysis also to subtargets:
```
 A.apply(M.index("distance").lt(1600).matrix) 
dim
0    500
1    342
2    ...
...
```
Or we use a decorator to apply it to subtargets separately. To that end, add an entry "decorators" to the *analysis config*:
```
analysis_specs = {
            "source": "/gpfs/bbp.cscs.ch/project/proj83/home/sood/analyses/manuscript/topological-analysis-subvolumes/topologists_connectome_analysis/library/topology.py",
            "method": "simplex_counts",
            "decorators": [
                {
                    "name": "grouped_by_grouping_config",
                    "args": [{"columns": ["mtype"], "method": "group_by_properties"}]
                }
            ],
            "output": "scalar"
}
A = Analysis("simplex_counts", analysis_specs)
A.apply(M.matrix.to_csc(), M.vertices)

idx-mtype  dim
L23_BP     0      102
           1        0
           2       85
           3      161
           4        8
L23_BTC    0      126
           1       30
           2       44
[...]
L23_SBC    0      119
           1       29
           2      244
           3      182
           4       37
```
We see that the analysis has now been performed for all mtypes separately. As a sanity check we can do some filtering beforehand:
```
subM = M.index("mtype").eq("L23_SBC")
A.apply(subM.matrix.tocsc(), subM.vertices)

idx-mtype  dim
L23_SBC    0      119
           1       29
           2      224
           3      182
           4       37
```
As expected, now only a single result consistent with the previous one is returned due to the prior filtering by m-type.

##### grouped_by_filtering_config
The previous decorator has one disadvantage: The groupings defined as above are all partitions. That is, each neuron of the base (filtered) population is part of one and exactly one group. This certainly does not cover all use cases.

A more flexible (albeit slightly more complicated) decorator is *group_by_filtering_config*. The user provides a list of "filtering" blocks of loader configs. Each provided "filtering" block is then considered one group, comprising the neurons that pass the filter. Instead of providing only the "filtering" block, a user can also provide an entire loader config; in that case all but the "filtering" block will be ignored.

Let's look at an example with overlapping groups:
```
analysis_specs = {
            "source": "/gpfs/bbp.cscs.ch/project/proj83/home/sood/analyses/manuscript/topological-analysis-subvolumes/topologists_connectome_analysis/library/topology.py",
            "method": "simplex_counts",
            "decorators": [
                {
                    "name": "grouped_by_filtering_config",
                    "args": [
                        [
                        {"filtering": [
                            {
                                "column": "mtype",
                                "values": ["L23_BP", "L23_SBC"]
                            }
                        ]},
                        {"filtering": [
                            {
                                "column": "mtype",
                                "values": ["L23_LBC", "L23_SBC"]
                            }
                        ]}
                        ]
                    ]
                }
            ],
            "output": "scalar"
}

A = Analysis("simplex_counts", analysis_specs)

A.apply(M.matrix.tocsc(), M.vertices)

mtype                   dim
['L23_BP', 'L23_SBC']   0      221
                        1       34
                        2      180
                        3       10
                        4        7
['L23_LBC', 'L23_SBC']  0      350
                        1      185
                        2       91
                        3       45
                        4        1
dtype: int64
```

We see that the analysis was applied to the submatrices of L23_BP + L23_SBC and to L23_LBC + L23_SBC, which overlap in L23_SBC. As before, we can use an "include" statement to use pre-defined neuron groups from another file and thus drastically simplify the configuration.
```
analysis_specs = {
            "source": "/gpfs/bbp.cscs.ch/project/proj83/home/sood/analyses/manuscript/topological-analysis-subvolumes/topologists_connectome_analysis/library/topology.py",
            "method": "simplex_counts",
            "decorators": [
                {
                    "name": "grouped_by_filtering_config",
                    "args": [
                        {"include": "more_groups.json"}
                    ]
                }
            ],
            "output": "scalar"
}

A = Analysis("simplex_counts", analysis_specs)

A.apply(M.matrix.tocsc(), M.vertices)

group name    dim
in_assembly1  0       20
              1        0
in_assembly2  0       20
              1        0
dtype: int64
```

Not many edges in our exemplary assemblies...
Like in most powerful tool boxes, there are several equivalent ways of doing the same thing. Remember that we have already loaded membership in our exemplary assemblies 1 and 2 as part of the loader config. That is, we have vertex proprties that are True if a neuron is a member of the respective assembly:
```
print(M.in_assembly1)
[False, False, False, ..., False, False, False]

print(M.in_assembly2)
[False False False ... False False False]
```
We can use that property instead of loading the assemblies again in the *analysis config*. 
```
analysis_specs = {
            "source": "/gpfs/bbp.cscs.ch/project/proj83/home/sood/analyses/manuscript/topological-analysis-subvolumes/topologists_connectome_analysis/library/topology.py",
            "method": "simplex_counts",
            "decorators": [
                {
                    "name": "grouped_by_filtering_config",
                    "args": [
                        [
                        {"filtering": [
                            {
                                "column": "in_assembly1",
                                "value": 0
                            },
                        ]},
                        {"filtering": [
                            {
                                "column": "in_assembly2",
                                "value": 0
                            }
                        ]}
                        ]
                    ]
                }
            ],
            "output": "scalar"
}
A = Analysis("simplex_counts", analysis_specs)

A.apply(M.matrix.tocsc(), M.vertices)
in_assembly1  in_assembly2  dim
1             nan           0       20
                            1        0
nan           1             0       20
                            1        0
```
The output is equivalent, albeit slightly differently formatted, as now membership in each assembly is its own column.
We can also use this to analyze the group of neurons that are neither part of assembl1 nor assembly2. Or in their intersection.

Also, you might have noted that the output of these analyses uses a MultiIndex that is automatically assembled from the filter specification. It uses the values of "column" and the valid values. In some cases, it is more useful to directly specify a name for the group defined by a filtering. For example, when you specify a long list of gids, the automatically assembled name would be excessively long. You can do this, as previously (see above) by providing an entry "name" as the same level as "filtering":
```
analysis_specs = {
            "source": "/gpfs/bbp.cscs.ch/project/proj83/home/sood/analyses/manuscript/topological-analysis-subvolumes/topologists_connectome_analysis/library/topology.py",
            "method": "simplex_counts",
            "decorators": [
                {
                    "name": "grouped_by_filtering_config",
                    "args": [
                        [
                        {"filtering": [
                            {
                                "column": "in_assembly1",
                                "value": 0
                            },
                            {
                                "column": "in_assembly2",
                                "value": 0
                            }
                        ],
                        "name": "in_neither"},
                        {"filtering": [
                            {
                                "column": "in_assembly1",
                                "value": 1
                            },
                            {
                                "column": "in_assembly2",
                                "value": 0
                            }
                        ],
                        "name": "in_intersection"}
                        ]
                    ]
                }
            ],
            "output": "scalar"
}

A = Analysis("simplex_counts", analysis_specs)

A.apply(M.matrix.tocsc(), M.vertices)
Out[15]: 
group name       dim
in_neither       0      1011
                 1      1073
                 2        39
                 3        23
                 4        42
in_intersection  0         0
dtype: int64
```

##### control_by_randomization

This decorator executes an analysis as usual on a connection matrix, then calls a specified randomization function on the matrix and repeats the analysis on the randomized control. The matrix is randomized and analyzed a specifiable number of times, and the mean result is reported in addition to the result for the original matrix.

**TODO**: Provide example.

To make full use of this randomization, it is important to understand that the decorators can be chained and nested! For example, it is possible to first split a connectivity matrix by m-type, analyze each submatrix, then randomize the submatrices and analyze them again. This is done by first listing the grouping decorator, then the randomization decorator:
```
[...]
            "decorators": [
                {
                    "name": "grouped_by_grouping_config",
                    "args": [{"columns": ["mtype"], "method": "group_by_properties"}]
                },
                {
                    "name": "control_by_randomization",
                    "analysis_arg":
                    {
                        "random_er":{
                            "source": "random_er.py",
                            "method": "random_er",
                            "args": [],
                            "kwargs": {},
                            "output": "Matrix"
                        }
                    },
                    "args": [],
                    "kwargs": {"n_randomizations": 3}
                }
            ],
[...]
A.apply(M.matrix.tocsc(), M.vertices)
Control    idx-mtype  dim
data       L23_BP     0      102.000000
                      1        0.000000
                                ...    
random_er  L23_SBC    0      119.000000
                      1       18.333333
                      2        2.000000
Length: 90, dtype: float64
```

We see that we now get a breakdown both by mtype and by data vs. control.

And it gets even more powerful: Let's build an analysis where the entire matrix is analyzed as one, but in the control we apply an ER-randomization separately for submatrices of each m-type pathway (stochastic block model):

**TODO**: This needs to be implemented.
