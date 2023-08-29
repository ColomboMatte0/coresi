# License: Apache 2.0
# Carl Kadie
# https://fastlmm.github.io/

import logging
import os

import numpy as np

_warn_array_module_once = False


def array_module(xp=None):
    """
    Find the array module to use, for example **numpy** or **cupy**.

    :param xp: The array module to use, for example, 'numpy'
               (normal CPU-based module) or 'cupy' (GPU-based module).
               If not given, will try to read
               from the ARRAY_MODULE environment variable. If not given and
               ARRAY_MODULE is not set,
               will use numpy. If 'cupy' is requested, will
               try to 'import cupy'. If that import fails, will
               revert to numpy.
    :type xp: optional, string or Python module
    :rtype: Python module

    >>> from pysnptools.util import array_module
    >>> xp = array_module() # will look at environment variable
    >>> print(xp.zeros((3)))
    [0. 0. 0.]
    >>> xp = array_module('cupy') # will try to import 'cupy'
    >>> print(xp.zeros((3)))
    [0. 0. 0.]
    """
    xp = xp or os.environ.get("ARRAY_MODULE", "numpy")

    if xp == "numpy":
        return np

    if xp == "cupy":
        try:
            import cupy as cp

            return cp
        except ModuleNotFoundError as e:
            global _warn_array_module_once
            if not _warn_array_module_once:
                logging.warning(f"Using numpy. ({e})")
                _warn_array_module_once = True
            return np

    raise ValueError(f"Don't know ARRAY_MODULE '{xp}'")
