import pandas
import numpy

from ...flatmapping import supersampled_locations, apply_flatmap_with_translation
from .defaults import XYZ, UVW, GID, DEFAULT_PIXEL_SZ, FIBER_GID, SS_FLAT_FIBER, SS_FLAT_NEURON, FLAT_COORDINATES

# TODO: All these functions assume you have already loaded x y z coordinates. should check that.


def supersampled_neuron_locations(df_in, circ=None, fm=None, orient=None, pixel_sz=DEFAULT_PIXEL_SZ):
    return supersampled_locations(df_in, XYZ, circ=circ, fm=fm, orient=orient, pixel_sz=pixel_sz,
                                  column_index=GID, columns_out=SS_FLAT_NEURON)


def supersampled_fiber_locations(df_in, circ=None, fm=None, orient=None, pixel_sz=DEFAULT_PIXEL_SZ):
    return supersampled_locations(df_in, XYZ, circ=circ, fm=fm, orient=orient, pixel_sz=pixel_sz,
                                  column_index=FIBER_GID, columns_out=SS_FLAT_FIBER)


def flat_locations(df_in, circ, fm=None):
    if fm is None:
        fm = circ.atlas.load_data("flatmap")
    xyz = df_in[XYZ]
    uvw = None
    if numpy.all([x in df_in for x in UVW]):
        uvw = df_in[UVW]
    flat_coords = apply_flatmap_with_translation(xyz.values, uvw.values, fm)
    return pandas.DataFrame(flat_coords,
                            columns=FLAT_COORDINATES,
                            index=xyz.index)


AVAILABLE_EXTRAS = {
    flat_locations: FLAT_COORDINATES,
    supersampled_neuron_locations: SS_FLAT_NEURON,
    supersampled_fiber_locations: SS_FLAT_FIBER
}


def add_extra_properties(df_in, circ, lst_properties, fm=None):
    lst_properties = list(lst_properties)
    for extra_fun, extra_props in AVAILABLE_EXTRAS.items():
        props = []
        for p in extra_props:
            if p in lst_properties:
                props.append(lst_properties.pop(lst_properties.index(p)))
        if len(props) > 0:
            new_df = extra_fun(df_in, circ, fm=fm)
            df_in = pandas.concat([df_in, new_df[props]], axis=1)
    if len(lst_properties) > 0:
        raise ValueError("Unknown properties: {0}".format(lst_properties))
    return df_in
