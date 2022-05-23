r"""
Beam useful functions (:mod:`pyfe3d.beam_utils`)
==================================================

.. currentmodule:: pyfe3d.beam_utils

"""
import numpy as np

from pyfe3d import DOF
from .coord import CoordR


def beam_coord(beam, vecxy, ncoords: np.ndarray):
    r"""Determine the coordinate system of a beam element

    The element coordinate system is determined identically what is explained
    in Nastran's quick reference guide for the CBEAM element.

    Parameters
    ----------
    beam : :class:`.BeamC` or :class:`.BeamLR` object
        The beam element.
    vecxy : array-like
        Vector on the xy plane of the element coordinate system.
    ncoords : :class:`.np.ndarray` object
        An array with shape `N, 3` for a system with `N` nodes.
    """
    x1 = ncoords[beam.c1//DOF]
    x2 = ncoords[beam.c2//DOF]
    xaxis = x2 - x1
    center = x1
    zaxis = np.cross(xaxis, vecxy)
    vecxz = xaxis + zaxis
    return CoordR(beam.eid, center, zaxis, vecxz)
