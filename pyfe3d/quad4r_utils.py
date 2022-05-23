r"""
Quad4R useful functions (:mod:`pyfe3d.quad4r_utils`)
====================================================

.. currentmodule:: pyfe3d.quad4r_utils

"""
import numpy as np

from pyfe3d import DOF
from .quad4r import Quad4R
from .coord import CoordR


def quad4r_coord(quad: Quad4R, ncoords: np.ndarray):
    r"""Determine the coordinate system of a :class:`.Quad4R` element

    The element coordinate system is determined identically what is explained
    in Nastran's quick reference guide for the CQUAD4 element, as illustrated
    below.

    .. image:: ../figures/nastran_cquad4.svg

    Parameters
    ----------
    quad : :class:`.Quad4R` object
        The quadilateral element.
    ncoords : :class:`.np.ndarray` object
        An array with shape `N, 3` for a system with `N` nodes.
    """
    x1 = ncoords[quad.c1//DOF]
    x2 = ncoords[quad.c2//DOF]
    x3 = ncoords[quad.c3//DOF]
    x4 = ncoords[quad.c4//DOF]
    v13 = x3 - x1
    v42 = x2 - x4
    center = (x1 + x2 + x3 + x4)/4.
    xaxis = (v13 + v42)/2.
    zaxis = np.cross(v42, v13)
    vecxz = xaxis + zaxis
    return CoordR(quad.eid, center, zaxis, vecxz)
