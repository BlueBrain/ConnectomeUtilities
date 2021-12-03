"""
Contributed by Vishal Sood
Last changed: 2021/11/29
"""
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
    analyses = in_config["analyses"]
    return collect_plugins_of_type(SingleMethodAnalysisFromSource,
                                   in_config=analyses)

def collect_plugins_of_type(T, in_config):
    """..."""
    return {T(name, description) for name, description in in_config.items()}


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

    def __init__(self, name, description):
        """..."""
        self._name = name
        self._description = _resolve_includes(description)
        self._source = self.read_source(description)
        self._args = self.read_args(description)
        self._kwargs = self.read_kwargs(description)
        self._method = self.read_method(description)
        self._output_type = self.read_output_type(description)
        self._analysis = self.decorate(
            self.load(description),
            description
        )
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

        module, method = import_module(from_path=source, with_method=method)
        self._module = module
        return method
    
    def decorate(self, analysis, description):
        from . import analysis_decorators
        lst_decorators = self.read_decorators(description)
        for decorator in lst_decorators:
            dec_func = getattr(analysis_decorators, decorator["name"])
            args = decorator.get("args", [])
            if "analysis_arg" in decorator:
                args = [
                    SingleMethodAnalysisFromSource(k, v) for k, v in decorator["analysis_arg"].items()
                ] + args
            analysis = dec_func(*args, **decorator.get("kwargs", {}))(analysis)
        return analysis

    def apply(self, adjacency, node_properties=None, log_info=None):
        """..."""
        if log_info:
            LOG.info("APPLY %s", log_info)
        try:
            matrix = adjacency.matrix
        except AttributeError:
            matrix = adjacency

        if node_properties is not None:
            assert node_properties.shape[0] == matrix.shape[0]

        result = self._analysis(matrix, node_properties,
                              *self._args, **self._kwargs)

        if log_info:
            LOG.info("Done %s", log_info)

        return result


    @staticmethod
    def collect(data):
        """collect data...
        TODO: We could have the scientist provide a collect method.
        """
        return data
