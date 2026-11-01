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
with 5 elements). BeamLR converges to Eq. (2-57) as the square of the
element length: its loads are above it by 1.03e-3, 1.65e-4, 4.11e-5 and
1.03e-5 with 20, 50, 100 and 200 elements, in both planes and for both
sections and directions. Its tolerance with 100 elements, 1e-4, is below
the reduction due to shear deformation, 1.8e-4, which the test therefore
resolves: up to pyfe3d 0.11, BeamLR took the slopes of its geometric
stiffness matrix from the rotations, converged to the Euler load of Eq.
(2-4) instead, and its loads with 100 elements were 2.0e-4 and 7.3e-4
above Eq. (2-57).

:func:`test_shear_flexible_column` checks Eq. (2-57) where the shear
deformation dominates, with ``P_e/(A*G/n) = 0.2`` and ``0.6``.
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
    'BeamLR': (BeamLR, BeamLRProbe, BeamLRData, 100, 1e-4),
}


def buckling_loads(name, b, h, direction, vxy, shear_ratio=None,
                   num_elements=None):
    """Two lowest buckling loads, the Euler loads of both planes, and the
    shear stiffness ``kappa*G*A``

    With ``shear_ratio`` the shear modulus is set such that the Euler load
    of the weak plane divided by ``kappa*G*A`` equals ``shear_ratio``;
    ``num_elements`` replaces the number of elements of ``BEAMS``.
    """
    cls, probecls, datacls, default_num_elements, _ = BEAMS[name]
    if num_elements is None:
        num_elements = default_num_elements
    L = 3.
    E = 203.e9
    prop = BeamProp()
    prop.A = b*h
    prop.E = E
    prop.G = 5/6*E/2/1.3
    prop.Izz = b*h**3/12
    prop.Iyy = b**3*h/12
    prop.J = prop.Izz + prop.Iyy
    if shear_ratio is not None:
        # NOTE Eq. (2-4) of Timoshenko and Gere (1961), weak plane
        prop.G = pi**2*E*min(prop.Izz, prop.Iyy)/(4*L**2)/(shear_ratio*prop.A)

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


@pytest.mark.parametrize('name, num_elements', [('BeamC', 20),
                                                ('BeamLR', 40)])
@pytest.mark.parametrize('shear_ratio', [0.2, 0.6])
def test_shear_flexible_column(name, num_elements, shear_ratio):
    r"""Clamped-free column whose buckling load is governed by the shear
    deformation, as in sandwich and composite columns

    The shear modulus is set such that ``P_e/(A*G/n)`` is ``0.2`` or ``0.6``,
    ``P_e`` being the Euler load of Eq. (2-4), so that Eq. (2-57) of
    Timoshenko and Gere (1961), Sec. 2.17, p. 133, ``P_cr = P_e/(1 +
    n*P_e/(A*G))``, lowers the load by 16.7 % and 37.5 %. The square
    section along the direction ``(1, 2, 3)`` buckles at this load in both
    planes. Measured errors, which fall as the square of the element length:

    ======  ========  ===========  ===========
    beam    elements  ratio 0.2    ratio 0.6
    ======  ========  ===========  ===========
    BeamC   10        +2.86e-4     +4.82e-4
    BeamC   20        +7.14e-5     +1.21e-4
    BeamC   40        +1.79e-5     +3.01e-5
    BeamLR  20        +8.57e-4     +6.43e-4
    BeamLR  40        +2.14e-4     +1.61e-4
    BeamLR  80        +5.36e-5     +4.02e-5
    BeamLR  160       +1.34e-5     +1.00e-5
    ======  ========  ===========  ===========

    The tolerance, 1e-3, is above these errors and two orders of magnitude
    below the reduction due to shear deformation that the test checks. Up to
    pyfe3d 0.11, BeamLR took the slopes of its geometric stiffness matrix
    from the rotations and converged to ``P_e`` instead, 20 % and 60 % above
    Eq. (2-57).
    """
    loads, euler, _, kGA = buckling_loads(name, 0.05, 0.05, (1, 2, 3),
                                          (0, 0, 1), shear_ratio=shear_ratio,
                                          num_elements=num_elements)
    assert np.isclose(euler/kGA, shear_ratio, rtol=1e-12)
    # NOTE Eq. (2-57) of Timoshenko and Gere (1961), n/(A*G) = 1/(kappa*G*A)
    expected = euler/(1 + euler/kGA)
    assert np.allclose(loads, expected, rtol=1e-3)


if __name__ == '__main__':
    for name in sorted(BEAMS):
        for b, h in [(0.05, 0.05), (0.10, 0.05)]:
            loads, weak, strong, kGA = buckling_loads(name, b, h, (1, 2, 3),
                                                      (0, 0, 1))
            print(name, b, h, 'Pcr/Euler_weak =', loads/weak,
                  'Euler_strong/Euler_weak =', strong/weak)
