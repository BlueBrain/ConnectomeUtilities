import pandas
import numpy

from ...flatmapping import supersampled_locations, apply_flatmap_with_translation
from .defaults import XYZ, UVW, DEFAULT_PIXEL_SZ, SS_COORDINATES, FLAT_COORDINATES


def supersampled_locations_wrapper(df_in, circ=None, fm=None, orient=None, pixel_sz=DEFAULT_PIXEL_SZ):
    assert numpy.all([col in df_in for col in XYZ]), "Must load base x, y, z coordinates first"
    columns_uvw = None
    if numpy.all([col in df_in for col in UVW]):
        columns_uvw = UVW
    return supersampled_locations(df_in, XYZ, columns_uvw=columns_uvw, circ=circ, fm=fm, orient=orient,
                                  pixel_sz=pixel_sz, columns_out=SS_COORDINATES)


def flat_locations(df_in, circ, fm=None):
    assert numpy.all([col in df_in for col in XYZ]), "Must load base x, y, z coordinates first"
    if fm is None:
        fm = circ.atlas.load_data("flatmap")
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
