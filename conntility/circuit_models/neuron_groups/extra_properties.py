# SPDX-License-Identifier: Apache-2.0
# Functionality for adding additional columns to the DataFrame holding neuron information
# That is: information that is not already easily available using Circuit.cells.get
import pandas
import numpy

from ...flatmapping import supersampled_locations, apply_flatmap_with_translation
from .defaults import XYZ, UVW, DEFAULT_PIXEL_SZ, SS_COORDINATES, FLAT_COORDINATES


def supersampled_locations_wrapper(df_in, circ=None, fm=None, orient=None, pixel_sz=DEFAULT_PIXEL_SZ):
    """
    Wraps the lookup of supersampled flat locations and depth of fibers and neurons. Can be used for
    both fiber and neuron locations.
    Input:
    df_in (pandas.DataFrame): Holds already loaded information on neurons or fibers. Must contain at
    least three columns with x, y and z coordinates. If it also contains u, v, w columns, they may be
    used to translate x, y, z locations into the flat mapped volume (required for fibers).
    fm (str, optional): Path to flatmap file to use
    orient (str, optional): Path to orientation volume to use
    circ (bluepysnap.Circuit, optional): Can be provided instead of fm and orient. In that case flatmap and
    orientation are loaded from Circuit.atlas
    pixel_sz (float): Assumed approximate size in um of a single flatmap pixel. If set to None, an 
    approximate value will be estimated

    Returns:
    a copy of df_in with two additional columns holding the supersampled flat coordinates and depth
    """
    assert numpy.all([col in df_in for col in XYZ]), "Must load base x, y, z coordinates first"
    columns_uvw = None
    if numpy.all([col in df_in for col in UVW]):
        columns_uvw = UVW
    return supersampled_locations(df_in, XYZ, columns_uvw=columns_uvw, circ=circ, fm=fm, orient=orient,
                                  pixel_sz=pixel_sz, columns_out=SS_COORDINATES, include_depth=True)


def flat_locations(df_in, circ, fm=None):
    """
    Adds simple flatmapped locations to a DataFrame holding neuron information.
    Input:
    df_in (pandas.DataFrame): Holds already loaded information on neurons or fibers. Must contain at
    least three columns with x, y and z coordinates. If it also contains u, v, w columns, they may be
    used to translate x, y, z locations into the flat mapped volume (required for fibers).
    fm (str, optional): Path to flatmap file to use
    circ (bluepysnap.Circuit): Can be provided instead of fm. In that case flatmap is loaded from
    Circuit.atlas. If fm is provided, then circ can be None.
    """
    assert numpy.all([col in df_in for col in XYZ]), "Must load base x, y, z coordinates first"
    if fm is None:
        from .sonata_extensions import load_atlas_data
        fm = load_atlas_data(circ, "flatmap")
    xyz = df_in[XYZ].values
    uvw = None
    if numpy.all([x in df_in for x in UVW]):
        uvw = df_in[UVW].values
    flat_coords = apply_flatmap_with_translation(xyz, uvw, fm)
    return pandas.DataFrame(flat_coords,
                            columns=FLAT_COORDINATES,
                            index=df_in.index)


AVAILABLE_EXTRAS = {
    flat_locations: FLAT_COORDINATES,
    supersampled_locations_wrapper: SS_COORDINATES
}


def add_extra_properties(df_in, circ, lst_properties, fm=None):
    """
    Adds additional properties (like flat loactions) to a DataFrame holding neuron or fiber information. 
    List of available properties:
    flat_x: Flat mapped x coordinate (pixel resolution)
    flat_y: Flat mapped y coordinate (pixel resolution)
    ss_flat_x: Supersampled flat mapped x coordinate (approximately in um)
    ss_flat_y: Supersampled flat mapped y coordinate (approximately in um)
    depth: Approximate cortical depth of a neuron in um

    Input:
    df_in (pandas.DataFrame): Holds already loaded information on neurons or fibers. Must contain at
    least three columns with x, y and z coordinates. If it also contains u, v, w columns, they may be
    used to translate x, y, z locations into the flat mapped volume (required for fibers).
    lst_properties (list-like): List of properties to add. For available properties: see above.
    circ (bluepysnap.Circuit): Circuit that the neurons or fibers originate from. Must have a "orientation"
    atlas in Circuit.atlas!
    fm (str, optional): Path to flat map file to use. If not provided, then circ must have a "flatmap"
    atlas in Circuit.atlas!

    Returns:
    a copy of df_in with additional columns holding the specified additional properties.
    """
    lst_properties = list(lst_properties)
    for extra_fun, extra_props in AVAILABLE_EXTRAS.items():
        props = []
        for p in extra_props:
            if p in lst_properties:
                props.append(lst_properties.pop(lst_properties.index(p)))
        if len(props) > 0:
            new_df = extra_fun(df_in, circ, fm=fm)
            df_in = pandas.concat([df_in, new_df[props]], axis=1, copy=False)
    if len(lst_properties) > 0:
        raise ValueError("Unknown properties: {0}".format(lst_properties))
    return df_in
