# SPDX-License-Identifier: Apache-2.0
import numpy

from .supersampling import supersample_flatmap


def _loader(recipe):
    if isinstance(recipe, str):
        import yaml
        with open(recipe, "r") as fid:
            mp = yaml.load(fid, Loader=yaml.SafeLoader)
        return mp
    return recipe


def regions_of_population(nm, recipe):
    tmp = [x for x in recipe["populations"] if x["name"] == nm]
    assert len(tmp) == 1
    regions = tmp[0]['atlas_region']
    if isinstance(regions, list):
        return [region["name"] for region in regions]
    return [regions['name']]


def ids_of_regions(names_region, hier):
    reg_ids = set()
    for region in names_region:
        reg_ids.update(hier.find(region, "acronym", with_descendants=True))
    return list(reg_ids)


def twod2rgb(flat_coords, x, y):
    pts = flat_coords - numpy.array([[x[2], y[2]]])
    T = numpy.array([[x[0] - x[2], x[1] - x[2]],
                    [y[0] - y[2], y[1] - y[2]]])
    Tinv = numpy.linalg.inv(T)
    l_0_1 = numpy.dot(Tinv, pts.transpose())
    l = numpy.vstack([l_0_1, 1.0 - numpy.sum(l_0_1, axis=0, keepdims=True)]).transpose()
    l[l < 0] = 0.0; l[l > 1] = 1.0
    return l


def twod2mapping_coords(flat_coords, x, y):
    pts = flat_coords - numpy.array([[x[2], y[2]]])
    v0 = numpy.array([x[0], y[0]]) - numpy.array([x[2], y[2]])
    v1 = numpy.array([x[1], y[1]]) - numpy.array([x[2], y[2]])
    v0 = v0 / numpy.linalg.norm(v0); v1 = v1 / numpy.linalg.norm(v1)
    x_out = numpy.dot(pts, v0.reshape((2, 1)))  # N x 2 * 2 x 1
    y_out = numpy.dot(pts, v1.reshape((2, 1)))
    return numpy.hstack([x_out, y_out])


def atlas_of_mapping_coordinates(recipe, circ=None, fm=None, orient=None, hier=None, ann=None,
                                 supersample=False):
    import voxcell
    from ..circuit_models.neuron_groups.sonata_extensions import load_atlas_data, load_atlas_hierarchy

    if fm is None or orient is None or hier is None or ann is None:
        assert circ is not None, "Must provide all of fm, orient, hier and ann or a Circuit!"
        if fm is None:
            fm = load_atlas_data(circ, "flatmap")
        if orient is None:
            orient = load_atlas_data(circ, "orientation")
        if hier is None:
            hier = load_atlas_hierarchy(circ)
        if ann is None:
            ann = load_atlas_data(circ, "brain_regions")
    elif circ is not None:
        print("Provided Circuit will be ignored since all of fm, orient, hier and ann were provided")
    
    if supersample:
        fm = supersample_flatmap(fm, orient, pixel_sz=1.0)
    recipe = _loader(recipe)

    touched_srcs = []
    out_raw = -numpy.ones_like(fm.raw, dtype=int)

    for proj in recipe["projections"]:
        src_str = proj["source"].split("_")[0]
        if src_str in touched_srcs:
            continue
        touched_srcs.append(src_str)

        pop_strs = regions_of_population(proj["source"], recipe)
        pop_ids = ids_of_regions(pop_strs, hier)
        pop_mask = numpy.in1d(ann.raw.flat, pop_ids).reshape(ann.raw.shape)
        pop_flat_coords = fm.raw[pop_mask]

        x = numpy.array(proj["mapping_coordinate_system"]["x"])
        y = numpy.array(proj["mapping_coordinate_system"]["y"])

        out_raw[pop_mask] = twod2mapping_coords(pop_flat_coords, x, y)
    
    return voxcell.VoxelData(out_raw, fm.voxel_dimensions, offset=fm.offset)



