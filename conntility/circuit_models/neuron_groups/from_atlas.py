# SPDX-License-Identifier: Apache-2.0
import os
import numpy

import pandas

from .defaults import XYZ


def atlas_property(df_in, atlas, circ=None, column_names=None):
    """
    Adds simple flatmapped locations to a DataFrame holding neuron information.
    Input:
    df_in (pandas.DataFrame): Holds already loaded information on neurons or fibers. Must contain at
    least three columns with x, y and z coordinates. If it also contains u, v, w columns, they may be
    used to translate x, y, z locations into the flat mapped volume (required for fibers).
    atlas: Specifies the atlas to load properties from.
    Must be one of: 
      - Path to atlas (nrrd-format) file to use
      - A voxcell.VoxelData object
      - A string denoting the name of the atlas to be found within the Circuit's atlas directory
    circ (bluepysnap.Circuit, optional): Must be provided if the third option for specifying atlas is used.
    """
    try:
        import voxcell
    except ImportError as e:
        print("Optional dependency voxcell not found!")
        raise e

    assert numpy.all([col in df_in for col in XYZ]), "Must load base x, y, z coordinates first"

    if not isinstance(atlas, voxcell.VoxelData):
        if os.path.isfile(atlas):
            if column_names is None:
                column_names = [os.path.splitext(os.path.split(atlas)[1])[0]]
            atlas = voxcell.VoxelData.load_nrrd(atlas)
        else:
            assert circ is not None, "Must specify Circuit to load data from its atlas directory!"
            from .sonata_extensions import load_atlas_data
            if column_names is None:
                column_names = [atlas]
            atlas = load_atlas_data(circ, atlas)
    else:
        assert isinstance(column_names, list) or isinstance(column_names, numpy.ndarray), "Must specify column names!"
    
    xyz = df_in[XYZ].values
    atlas_data = atlas.lookup(xyz)
    shape = atlas_data.ndim
    if shape > 1: shape = atlas_data.shape[1]

    assert shape == len(column_names), "Size mismatch: Data has size {0}, column names {1}".format(
        shape, len(column_names)
    )
    
    return pandas.DataFrame(atlas_data,
                            columns=column_names,
                            index=df_in.index)
