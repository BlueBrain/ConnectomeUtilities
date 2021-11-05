import numpy


def colored_points_to_image(flat_coords, cols, extent=None):
    flat_coords = flat_coords.astype(int)
    if extent is None:
        extent = flat_coords.max(axis=0) + 1
    img = numpy.NaN * numpy.ones(tuple(extent) + (3, ))
    for xy, col in zip(flat_coords, cols):
        img[xy[0], xy[1], :] = col
    return img


def _flatmap_extent(fm, subsample=None):
    mx = fm.raw.max(axis=(0, 1, 2))
    if subsample is not None:
        mx = numpy.floor(mx / subsample).astype(int)
    return mx + 1


def _flat_coordinates_of_regions(names_region, fm, hier, ann, make_unique=True, subsample=None):
    reg_ids = set()
    for region in names_region:
        reg_ids.update(hier.find(region, "acronym", with_descendants=True))
    lst_ids = list(reg_ids)
    view3d = numpy.in1d(ann.raw.flat, lst_ids).reshape(ann.raw.shape)
    view2d = fm.raw[view3d]
    view2d = view2d[numpy.all(view2d >= 0, axis=1)]

    if subsample is not None:
        view2d = numpy.floor(view2d / subsample).astype(int)

    if not make_unique:
        return view2d
    mul = view2d[:, 0].max() + 1
    view_comb = view2d[:, 0] + mul * view2d[:, 1]
    unique_comb = numpy.unique(view_comb)
    return numpy.vstack([
        numpy.mod(unique_comb, mul),
        numpy.floor(unique_comb / mul)
    ]).transpose()


def flat_coordinates_of_regions(names_regions, fm, *args, make_unique=False, subsample=None):
    if len(args) == 2:
        return _flat_coordinates_of_regions(names_regions, fm, *args, make_unique=make_unique, subsample=subsample)
    elif len(args) == 1:
        circ = args[0]
        atlas = circ.atlas
        hier = atlas.load_region_map()
        ann = atlas.load_data("brain_regions")
        return _flat_coordinates_of_regions(names_regions, fm, hier, ann, make_unique=make_unique, subsample=subsample)
    else:
        raise ValueError()


def flat_region_image(lst_regions, fm, *args, extent=None, subsample=None):
    if extent is None:
        extent = _flatmap_extent(fm, subsample=subsample)
    xbins = numpy.arange(extent[0] + 1)
    ybins = numpy.arange(extent[1] + 1)
    counters = []
    for reg in lst_regions:
        if not isinstance(reg, list):
            reg = [reg]
        A = flat_coordinates_of_regions(reg, fm, *args, make_unique=False, subsample=subsample)
        H = numpy.histogram2d(A[:, 0], A[:, 1], bins=(xbins, ybins))[0]
        counters.append(H)

    region_lookup = numpy.argmax(numpy.dstack(counters), axis=2)
    region_lookup[numpy.sum(numpy.dstack(counters), axis=2) == 0] = -1
    return region_lookup
