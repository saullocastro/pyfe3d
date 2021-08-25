r"""
Finite elements for 3D problems in Python/Cython
================================================

"""
import numpy as np
from alg3dpy.vector import asvector
from alg3dpy.constants import Z, O

from .quad4r import Quad4R, Quad4RData, Quad4RProbe
from .beamc import BeamC, BeamCData, BeamCProbe
from .beamlr import BeamLR, BeamLRData, BeamLRProbe
from .truss import Truss, TrussData, TrussProbe
from .spring import Spring
DOF = 6
INT = np.int64
DOUBLE = np.float64

from .coord import CoordR
vecxz = asvector([1.,0.,1.])
CSYSGLOBAL = CoordR(0, O, Z, vecxz)
