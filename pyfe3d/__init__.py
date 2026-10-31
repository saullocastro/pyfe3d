r"""
================================================
Finite elements for 3D problems in Python/Cython
================================================

The ``pyfe3d`` namespace gives the element classes :class:`.Quad4`,
:class:`.Quad4R`, :class:`.Tria3R`, :class:`.Tria3DSG`, :class:`.BeamC`,
:class:`.BeamLR`, :class:`.Truss` and :class:`.Spring`, each with its
``Data`` and ``Probe`` classes (e.g. :class:`.BeamCData` and
:class:`.BeamCProbe`), the classes of the element-independent co-rotational
(EICR) layer for large rotations of :mod:`pyfe3d.corotational`,
:class:`.CorotationalModel`, which assembles a model, and the wrappers of
one element each, :class:`.CorotationalShell`, :class:`.CorotationalBeam`,
:class:`.CorotationalTruss` and :class:`.CorotationalSpring`, and the
module :mod:`pyfe3d.solver`. The finite-rotation utilities of the
co-rotational layer, e.g. :func:`pyfe3d.corotational.expso3`, are imported
from :mod:`pyfe3d.corotational`. It also defines:

- ``DOF = 6``: number of degrees of freedom per node;
- ``INT``: NumPy integer type of the C ``long`` of the platform,
  ``np.int64`` when a C ``long`` has 8 bytes and ``np.int32`` otherwise,
  matching the ``long`` arrays of row and column indices of the ``update_*``
  methods of the elements;
- ``DOUBLE = np.float64``: NumPy type of the arrays of values.

"""
import ctypes

import numpy as np

from .version import __version__
from .quad4 import Quad4, Quad4Data, Quad4Probe
from .quad4r import Quad4R, Quad4RData, Quad4RProbe
from .tria3r import Tria3R, Tria3RData, Tria3RProbe
from .tria3dsg import Tria3DSG, Tria3DSGData, Tria3DSGProbe
from .beamc import BeamC, BeamCData, BeamCProbe
from .beamlr import BeamLR, BeamLRData, BeamLRProbe
from .truss import Truss, TrussData, TrussProbe
from .spring import Spring, SpringData, SpringProbe
from .corotational import (CorotationalModel, CorotationalShell,
                           CorotationalBeam, CorotationalTruss,
                           CorotationalSpring)
from . import solver

DOF = 6

if ctypes.sizeof(ctypes.c_long) == 8:
    # here the C long will correspond to np.int64
    INT = np.int64
else:
    # here the C long will correspond to np.int32
    INT = np.int32

DOUBLE = np.float64
