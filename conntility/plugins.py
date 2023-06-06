# SPDX-License-Identifier: Apache-2.0
"""
Contributed by Vishal Sood
Last changed: 2021/11/29
"""
import importlib
from collections import OrderedDict
from pathlib import Path

import pandas as pd

def import_module(from_path, with_method=None):
    """..."""
    path = Path(from_path)

    assert path.exists

    assert path.suffix == ".py", f"Not a python file {path}!"

    spec = importlib.util.spec_from_file_location(path.stem, path)

    module = importlib.util.module_from_spec(spec)

    spec.loader.exec_module(module)

    if with_method:
        if not hasattr(module, with_method):
            raise TypeError(f"No method to {with_method}")
        return (module, getattr(module, with_method))
    return module


def import_module_with_name(n):
    """Must be in the environment."""
    assert isinstance(n, str)
    return importlib.import_module(n)


def load_module_from_path(p):
    """Load a module from a path.
    """
    path = Path(p)

    if not path.exists:
        raise FileNotFoundError(p.as_posix())

    if  path.suffix != ".py":
        raise ValueError(f"Not a python file {path}!")

    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module) #

    return module


def get_module(from_object, with_function=None):
    """Get module from an object.
    Read the code to see what object can be resolved to a module.
    If `with_method`, look for the method in the module.
    """
    def iterate(functions):
        """..."""
        if isinstance(functions, str):
            return [functions]
        try:
            items = iter(functions)
        except TypeError:
            items = [functions]

        return items

    def check(module, has_function=None):
        """..."""
        if not has_function:
            return module

        def get_method(function):
            """..."""
            try:
                method = getattr(module, function)
            except AttributeError:
                raise TypeError(f" {module} is missing required method {function}.")
            return method

        if isinstance(has_function, str):
            methods = get_method(has_function)

        methods = {f: get_method(f) for f in iterate(has_function)}

        return (module, methods)

    try:
        module = import_module_with_name(str(from_object))
    except ModuleNotFoundError:
        module = load_module_from_path(p=from_object)
        if not module:
            raise ModuleNotFoundError(f"that was specified by {from_object}")

    return check(module)


def collect_plugins_of_type(T, in_config):
    """..."""
    return {T(name, description) for name, description in items()}
