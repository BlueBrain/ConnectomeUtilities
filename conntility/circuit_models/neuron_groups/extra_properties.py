import pandas

from ...flatmapping import supersampled_neuron_locations


def flat_neuron_locations(circ, fm=None):
    if fm is None:
        fm = circ.atla.load_data("flatmap")
    xyz = circ.cells.get(properties=["x", "y", "z"])
    return pandas.DataFrame(fm.lookup(xyz.values),
                            columns=["flat x", "flat y"])


AVAILABLE_EXTRAS = {
    flat_neuron_locations: ["flat x", "flat y"],
    supersampled_neuron_locations: ["ss flat x", "ss flat y"]
}


def add_extra_properties(df_in, circ, lst_properties, fm=None):
    for extra_fun, extra_props in AVAILABLE_EXTRAS.items():
        props = []
        for p in extra_props:
            if p in lst_properties:
                props.append(lst_properties.pop(lst_properties.index(p)))
        if len(props) > 0:
            new_df = extra_fun(circ, fm=fm)
            df_in = pandas.concat([df_in, new_df[props]])
    if len(lst_properties) > 0:
        raise ValueError("Unknown properties: {0}".format(lst_properties))
    return df_in
