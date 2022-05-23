import sys
sys.path.append('..')

import numpy as np
from numpy import isclose
from scipy.sparse.linalg import eigsh, spsolve, cg
from scipy.sparse import coo_matrix

from pyfe3d.shellprop_utils import isotropic_plate
from pyfe3d import Quad4R, Quad4RData, Quad4RProbe, INT, DOUBLE, DOF
from pyfe3d.quad4r_utils import quad4r_coord


def test_quad4r_coord():
    data = Quad4RData()
    probe = Quad4RProbe()
    ncoords = np.array([
        [0, 0, 0],
        [15, 0, 0],
        [15, 15, 0],
        [0, 15, 0],
        ])
    E = 70.e9 # Pa
    nu = 0.33
    h = 0.002 # m
    prop = isotropic_plate(thickness=h, E=E, nu=nu, calc_scf=True)
    quad = Quad4R(probe)
    quad.c1 = DOF*0
    quad.c2 = DOF*1
    quad.c3 = DOF*2
    quad.c4 = DOF*3
    coord = quad4r_coord(quad, ncoords)
    cosa, cosb, cosg = coord.cosines_to_global()
    assert np.allclose([cosa, cosb, cosg], [1, 1, 1])

    ncoords = np.array([
        [0, 0, 0],
        [15, 0, 0],
        [15, 0, 15],
        [0, 0, 15],
        ])
    coord = quad4r_coord(quad, ncoords)
    cosa, cosb, cosg = coord.cosines_to_global()
    assert np.allclose([cosa, cosb, cosg], [0, 1, 1])

    ncoords = np.array([
        [0, 0, 0],
        [0, 15, 0],
        [-15, 15, 0],
        [-15, 0, 0],
        ])
    coord = quad4r_coord(quad, ncoords)
    cosa, cosb, cosg = coord.cosines_to_global()
    assert np.allclose([cosa, cosb, cosg], [1, 1, 0])

    ncoords = np.array([
        [0, 0, 0],
        [0, 0, 15],
        [0, -15, 15],
        [0, -15, 0],
        ])
    coord = quad4r_coord(quad, ncoords)
    cosa, cosb, cosg = coord.cosines_to_global()
    assert np.allclose([cosa, cosb, cosg], [-1, 0, 1], atol=1e-6)

    ncoords = np.array([
        [0, 0, 0],
        [0, 15, 0],
        [0, 15, 15],
        [0, 0, 15],
        ])
    coord = quad4r_coord(quad, ncoords)
    cosa, cosb, cosg = coord.cosines_to_global()
    # NOTE Gimbal lock in the determination of the angles
    # NOTE reason why we are using rij terms directly
    # assert np.allclose([cosa, cosb, cosg], [0, 0, 1])


if __name__ == '__main__':
    test_quad4r_coord()
