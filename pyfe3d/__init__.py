"""
Finite elements for 3D problems in Python/Cython
================================================

"""
import numpy as np

from .quad4r import Quad4R, Quad4RData, Quad4RProbe
from .beamc import BeamC, BeamCData, BeamCProbe
DOF = 6
INT = np.int64
DOUBLE = np.float64
