import sys
sys.path.append('..')

import numpy as np
from numpy import isclose
from scipy.sparse.linalg import eigsh, spsolve, cg
from scipy.sparse import coo_matrix

from pyfe3d.shellprop_utils import isotropic_plate
from pyfe3d import BeamC, BeamCData, BeamCProbe, INT, DOUBLE, DOF
from pyfe3d.beam_utils import beam_coord


def test_beam_coord():
    data = BeamCData()
    probe = BeamCProbe()
    ncoords = np.array([
        [0, 0, 0],
        [1, 0, 0],
        ])
    E = 70.e9 # Pa
    nu = 0.33
    h = 0.002 # m
    prop = isotropic_plate(thickness=h, E=E, nu=nu, calc_scf=True)
    beam = BeamC(probe)
    beam.c1 = DOF*0
    beam.c2 = DOF*1
    coord = beam_coord(beam, [1, 1, 0], ncoords)
    cosa, cosb, cosg = coord.cosines_to_global()
    assert np.allclose([cosa, cosb, cosg], [1, 1, 1])

    ncoords = np.array([
        [0, 0, 0],
        [0, 0, 1],
        ])
    coord = beam_coord(beam, [0, 1, 1], ncoords)
    cosa, cosb, cosg = coord.cosines_to_global()
    #NOTE Gimbal lock
    #assert np.allclose([cosa, cosb, cosg], [1, 0, 1])

    ncoords = np.array([
        [0, 0, 0],
        [1, 0, 1],
        ])
    coord = beam_coord(beam, [1, 1, 1], ncoords)
    cosa, cosb, cosg = coord.cosines_to_global()
    assert np.allclose([cosa, cosb, cosg], [1, np.cos(-np.pi/4), 1])

    ncoords = np.array([
        [0, 0, 0],
        [-1, 0, -1],
        ])
    coord = beam_coord(beam, [-1, 1, -1], ncoords)
    print(coord)
    cosa, cosb, cosg = coord.cosines_to_global()
    #NOTE Gimbal lock
    #assert np.allclose([cosa, cosb, cosg], [1, np.cos(-np.pi/4+np.pi), 1])


if __name__ == '__main__':
    test_beam_coord()
