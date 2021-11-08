import pandas


def add_extra_properties(df_in, circ, lst_properties, **kwargs):
    for prop in lst_properties:
        if prop == "supersampled_flat_coordinates":
            from ...flatmapping import supersampled_neuron_locations
            flat_locs = supersampled_neuron_locations(circ, **kwargs)
            df_in = pandas.concat([df_in, flat_locs], axis=1)
        elif prop == "flat_coordinates":
            fm = kwargs.get("fm", None)
            if fm is None:
                fm = circ.atlas.load_data("flatmap")
            df_in = pandas.concat([df_in, pandas.DataFrame(fm.lookup(df_in["x", "y", "z"]),
                                                           columns=["flat x", "flat y"])])
        else:
            raise ValueError()
    return df_in
