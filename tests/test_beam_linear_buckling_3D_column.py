"""Linear buckling of a clamped-free column free to buckle in both planes

Each bending plane buckles at its own Euler load, pi**2*E*I/(4*L**2). The
geometric stiffness matrix has to keep the two planes uncoupled: built from
the square of the summed slopes, (v,x + w,x)**2, instead of the sum of the
squares, v,x**2 + w,x**2, it couples them and the lowest load of a column
with a square cross section drops to half of the Euler load.
"""
import sys
sys.path.append('..')

import numpy as np
import pytest
from numpy import pi
from scipy.linalg import eigh
from scipy.sparse import coo_matrix

from pyfe3d.beamprop import BeamProp
from pyfe3d import (BeamC, BeamCData, BeamCProbe, BeamLR, BeamLRData,
                    BeamLRProbe, DOF, INT, DOUBLE)

BEAMS = {
    'BeamC': (BeamC, BeamCProbe, BeamCData, 20),
    'BeamLR': (BeamLR, BeamLRProbe, BeamLRData, 100),
}


def buckling_loads(name, b, h, direction, vxy):
    """Two lowest buckling loads, and the Euler loads of both planes"""
    cls, probecls, datacls, num_elements = BEAMS[name]
    L = 3.
    E = 203.e9
    prop = BeamProp()
    prop.A = b*h
    prop.E = E
    prop.G = 5/6*E/2/1.3
    prop.Izz = b*h**3/12
    prop.Iyy = b**3*h/12
    prop.J = prop.Izz + prop.Iyy

    direction = np.asarray(direction, dtype=float)
    direction /= np.linalg.norm(direction)
    s = np.linspace(0, L, num_elements + 1)
    x = (s[:, None]*direction[None, :]).flatten()
    N = DOF*(num_elements + 1)

    data = datacls()
    probe = probecls()
    KC0r = np.zeros(data.KC0_SPARSE_SIZE*num_elements, dtype=INT)
    KC0c = np.zeros(data.KC0_SPARSE_SIZE*num_elements, dtype=INT)
    KC0v = np.zeros(data.KC0_SPARSE_SIZE*num_elements, dtype=DOUBLE)
    KGr = np.zeros(data.KG_SPARSE_SIZE*num_elements, dtype=INT)
    KGc = np.zeros(data.KG_SPARSE_SIZE*num_elements, dtype=INT)
    KGv = np.zeros(data.KG_SPARSE_SIZE*num_elements, dtype=DOUBLE)
    beams = []
    for i in range(num_elements):
        beam = cls(probe)
        beam.n1, beam.n2 = i + 1, i + 2
        beam.c1, beam.c2 = DOF*i, DOF*(i + 1)
        beam.init_k_KC0 = i*data.KC0_SPARSE_SIZE
        beam.init_k_KG = i*data.KG_SPARSE_SIZE
        beam.update_rotation_matrix(vxy[0], vxy[1], vxy[2], x)
        beam.update_probe_xe(x)
        beam.update_KC0(KC0r, KC0c, KC0v, prop)
        beams.append(beam)
    KC0 = coo_matrix((KC0v, (KC0r, KC0c)), shape=(N, N)).toarray()

    bk = np.zeros(N, dtype=bool)
    bk[:DOF] = True
    bu = ~bk

    # unit compressive load at the tip
    fext = np.zeros(N)
    fext[-DOF:-DOF + 3] = -direction
    u = np.zeros(N)
    u[bu] = np.linalg.solve(KC0[np.ix_(bu, bu)], fext[bu])
    for beam in beams:
        beam.update_probe_ue(u)
        beam.update_KG(KGr, KGc, KGv, prop)
    KG = coo_matrix((KGv, (KGr, KGc)), shape=(N, N)).toarray()

    # KC0*phi = -lambda*KG*phi, solved as -KG*phi = mu*KC0*phi, lambda = 1/mu
    mu = eigh(-KG[np.ix_(bu, bu)], KC0[np.ix_(bu, bu)], eigvals_only=True)
    loads = np.sort(1/mu[mu > 1.e-12])
    euler_z = pi**2*E*prop.Izz/(4*L**2)
    euler_y = pi**2*E*prop.Iyy/(4*L**2)
    return loads[:2], min(euler_z, euler_y), max(euler_z, euler_y)


@pytest.mark.parametrize('name', sorted(BEAMS))
@pytest.mark.parametrize('b, h', [(0.05, 0.05), (0.10, 0.05)])
@pytest.mark.parametrize('direction, vxy', [((1, 0, 0), (0, 1, 0)),
                                            ((1, 2, 3), (0, 0, 1))])
def test_column_buckles_at_the_euler_load(name, b, h, direction, vxy):
    loads, euler_weak, euler_strong = buckling_loads(name, b, h, direction,
                                                     vxy)
    assert np.isclose(loads[0], euler_weak, rtol=0.01)
    # square section: the second plane buckles at the same load; rectangular
    # section: the next load is the lower of the Euler load of the strong
    # plane and the second mode of the weak plane, at nine times its load
    expected = euler_weak if b == h else min(euler_strong, 9*euler_weak)
    assert np.isclose(loads[1], expected, rtol=0.01)


if __name__ == '__main__':
    for name in sorted(BEAMS):
        for b, h in [(0.05, 0.05), (0.10, 0.05)]:
            loads, weak, strong = buckling_loads(name, b, h, (1, 2, 3),
                                                 (0, 0, 1))
            print(name, b, h, 'Pcr/Euler_weak =', loads/weak,
                  'Euler_strong/Euler_weak =', strong/weak)
