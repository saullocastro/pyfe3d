r"""Linear buckling of a clamped-free column free to buckle in both planes

Each bending plane buckles at its own Euler load, ``pi**2*E*I/(4*L**2)``,
Eq. (2-4) of Timoshenko, S. P., Gere, J. M., Theory of Elastic Stability,
2nd ed., McGraw-Hill, 1961, Sec. 2.1, p. 48, where the second mode of a
plane buckles at nine times this load (p. 48). The geometric stiffness
matrix has to keep the two planes uncoupled: built from the square of the
summed slopes, (v,x + w,x)**2, instead of the sum of the squares, v,x**2 +
w,x**2, it couples them and the lowest load of a column with a square cross
section drops to half of the Euler load.

With shear deformation the load ``P_e`` of each mode becomes ``P_e/(1 +
n*P_e/(A*G))``, Eq. (2-57), Sec. 2.17, p. 133, of Timoshenko and Gere
(1961), with ``n = 1.2`` for a rectangular cross section (p. 132), i.e. the
shear correction factor ``5/6`` of the tests. For these columns it lowers
the loads by 0.018 % (both planes of the square section and weak plane of
the rectangular one) and by 0.071 % (strong plane of the rectangular
section). BeamC reproduces Eq. (2-57) to 4.2e-7 with 20 elements (1.9e-5
with 5 elements). BeamLR converges to the Euler loads of Eq. (2-4),
without the reduction due to shear deformation: with 100 elements its loads
are 0.020 % (square section and weak plane) and 0.073 % (strong plane)
above Eq. (2-57), with 50 elements 0.026 and 0.080 %, hence its larger
tolerance.
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

# NOTE element, probe, data, number of elements, rtol
BEAMS = {
    'BeamC': (BeamC, BeamCProbe, BeamCData, 20, 1e-5),
    'BeamLR': (BeamLR, BeamLRProbe, BeamLRData, 100, 2e-3),
}


def buckling_loads(name, b, h, direction, vxy):
    """Two lowest buckling loads, the Euler loads of both planes, and the
    shear stiffness ``kappa*G*A``"""
    cls, probecls, datacls, num_elements, _ = BEAMS[name]
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
    # NOTE Eq. (2-4) of Timoshenko and Gere (1961)
    euler_z = pi**2*E*prop.Izz/(4*L**2)
    euler_y = pi**2*E*prop.Iyy/(4*L**2)
    return (loads[:2], min(euler_z, euler_y), max(euler_z, euler_y),
            prop.G*prop.A)


@pytest.mark.parametrize('name', sorted(BEAMS))
@pytest.mark.parametrize('b, h', [(0.05, 0.05), (0.10, 0.05)])
@pytest.mark.parametrize('direction, vxy', [((1, 0, 0), (0, 1, 0)),
                                            ((1, 2, 3), (0, 0, 1))])
def test_column_buckles_at_the_euler_load(name, b, h, direction, vxy):
    loads, euler_weak, euler_strong, kGA = buckling_loads(name, b, h,
                                                          direction, vxy)
    rtol = BEAMS[name][4]
    # NOTE Eq. (2-57) of Timoshenko and Gere (1961), n/(A*G) = 1/(kappa*G*A)
    shear = lambda Pe: Pe/(1 + Pe/kGA)
    assert np.isclose(loads[0], shear(euler_weak), rtol=rtol)
    # square section: the second plane buckles at the same load; rectangular
    # section: the next load is the lower of the Euler load of the strong
    # plane and the second mode of the weak plane, at nine times its load
    expected = euler_weak if b == h else min(euler_strong, 9*euler_weak)
    assert np.isclose(loads[1], shear(expected), rtol=rtol)


if __name__ == '__main__':
    for name in sorted(BEAMS):
        for b, h in [(0.05, 0.05), (0.10, 0.05)]:
            loads, weak, strong, kGA = buckling_loads(name, b, h, (1, 2, 3),
                                                      (0, 0, 1))
            print(name, b, h, 'Pcr/Euler_weak =', loads/weak,
                  'Euler_strong/Euler_weak =', strong/weak)
