# SPDX-License-Identifier: Apache-2.0
"""
Contributed by Vishal Sood
Modified by Michael W. Reimann
Last changed: 2021/12/15
"""

import os

from ..plugins import import_module

from ..io.logging import get_logger
from ..circuit_models.neuron_groups.grouping_config import _resolve_includes

LOG = get_logger("Topology Pipeline Analysis", "INFO")


def widen_by_index(level, dataframe):
    """Widen a dataframe by an index level."""
    import pandas as pd
    groups = dataframe.groupby(level)
    return pd.concat([g.droplevel(level) for _, g in groups], axis=1,
                     keys=[i for i, _ in groups], names=[level])


def get_analyses(in_config):
    """..."""
    if isinstance(in_config, str) or isinstance(in_config, os.PathLike):
        import json
        _root = os.path.split(os.path.abspath(in_config))[0]
        with open(in_config, "r") as fid:
            in_config = json.load(fid)
    else:
        _root = None
    analyses = in_config["analyses"]
    return collect_plugins_of_type(SingleMethodAnalysisFromSource,
                                   in_config=analyses, resolve_at=_root)


def collect_plugins_of_type(T, in_config, **kwargs):
    """..."""
    return {T(name, description, **kwargs) for name, description in in_config.items()}


class SingleMethodAnalysisFromSource:
    """Algorithms defined as such in the config:

    "analyze-connectivity": {
      "analyses": {
        "simplex_counts": {
          "source": "/gpfs/bbp.cscs.ch/project/proj83/home/sood/analyses/manuscript/topological-analysis-subvolumes/topologists_connectome_analysis/analysis/simplex_counts.py",
          "args": [],
          "kwargs": {},
          "method": "simplex-counts",
          "output": "scalar"
        }
      }
    }

    """
    @staticmethod
    def read_method(description):
        """..."""
        return description.get("method", "shuffle")

    @staticmethod
    def read_source(description):
        """..."""
        return description["source"]

    @staticmethod
    def read_args(description):
        """..."""
        return description.get("args", [])

    @staticmethod
    def read_kwargs(description):
        """..."""
        return description.get("kwargs", {})

    @staticmethod
    def read_output_type(description):
        """..."""
        return description["output"]

    @staticmethod
    def read_collection(description):
        """..."""
        policy = description.get("collect", None)
        if not policy:
            return None
        return policy
    
    @staticmethod
    def read_decorators(description):
        decorators = description.get("decorators", [])
        if not isinstance(decorators, list): return [decorators]
        return decorators

    def __init__(self, name, description, resolve_at=None):
        """..."""
        self._name = name
        self._root = resolve_at
        self._description = _resolve_includes(description, resolve_at=self._root)
        self._source = self.read_source(description)
        self._args = self.read_args(description)
        self._kwargs = self.read_kwargs(description)
        self._method = self.read_method(description)
        self._decoration = self.read_decorators(description)
        self._output_type = self.read_output_type(description)
        self._analysis = self.load(description)
        self._collection_policy = self.read_collection(description)

    @property
    def name(self):
        """..."""
        return self._name

    @property
    def description(self):
        return self._description

    @property
    def quantity(self):
        """To name the column in a dataframe, or an item in a series."""
        return self._description.get("quantity",
                                     self._description.get("method",
                                                           self._analysis.__name__))

    def load(self, description):
        """..."""
        source = self.read_source(description)
        method = self.read_method(description)

        try:
           run = getattr(source, method)
        except AttributeError:
            pass
        else:
            self._module = source
            return run

        if callable(source):
            #TODO: inspect source
            return source
        
        if not os.path.isfile(source):
            if self._root is not None and not os.path.isabs(source):
                source = os.path.join(self._root, source)

        module, method = import_module(from_path=source, with_method=method)
        self._module = module
        return method
    
    def decorate(self, analysis, lst_decorators, **kwargs):
        from . import analysis_decorators
        for decorator in lst_decorators:
            dec_func = getattr(analysis_decorators, decorator["name"])
            args = decorator.get("args", []).copy()

            for i in range(len(args)):  # To be used to insert dynamic arguments to decorators
                if isinstance(args[i], str):
                    if args[i] in kwargs:
                        args.insert(i, kwargs[args.pop(i)])
            
            if "analysis_arg" in decorator:
                args = [
                    SingleMethodAnalysisFromSource(k, v, resolve_at=self._root)
                    for k, v in decorator["analysis_arg"].items()
                ] + args
            analysis = dec_func(*args, **decorator.get("kwargs", {}))(analysis)
        return analysis

    def apply(self, adjacency, node_properties=None, log_info=None):
        """..."""
        from .. import ConnectivityMatrix
        if log_info:
            LOG.info("APPLY %s", log_info)
        decoration_kwargs = {}
        if isinstance(adjacency, ConnectivityMatrix):
            decoration_kwargs["ConnectivityMatrix"] = adjacency
            node_properties = adjacency.vertices
            adjacency = adjacency.matrix.tocsc()
        elif hasattr(adjacency, "matrix"):
            adjacency = adjacency.matrix

        if node_properties is not None:
            assert node_properties.shape[0] == adjacency.shape[0]

        result = self.decorate(self._analysis, self._decoration, **decoration_kwargs)(
            adjacency, node_properties, *self._args, **self._kwargs
        )

        if log_info:
            LOG.info("Done %s", log_info)

        return result

    @staticmethod
    def collect(data):
        """collect data...
        TODO: We could have the scientist provide a collect method.
        """
        return data
