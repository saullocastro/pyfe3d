import sys
sys.path.append('..')

import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.sparse import coo_matrix

from pyfe3d.beamprop import BeamProp
from pyfe3d import BeamLR, BeamLRData, BeamLRProbe, DOF, INT, DOUBLE


def test_static_point_load():
    r"""Tip deflection of a cantilever under a transverse point load at
    ``x = a``, in both planes of its square cross section

    The Euler-Bernoulli tip deflection is ``P*a**2*(3*L - a)/(6*E*I)``,
    Table G-1, Case 5, p. 985, of Gere, J. M., Goodno, B. J., Mechanics of
    Materials, 7th ed., Cengage Learning, 2009. The shear deformation adds
    ``P*a/(kappa*G*A)``, the second term of Eq. (1.2), p. 5, of Bathe, K.
    J., Finite Element Procedures, 2nd ed., 2014, applied to the loaded
    length ``a``, beyond which the shear force is zero. For this beam the
    shear term is 4.1e-5 of the total.

    BeamLR converges quadratically from below: the tip deflection is 0.53,
    0.13, 0.058 and 0.015 % below the reference with 8, 16, 24 and 48
    elements.
    """
    nnodes = 49 # NOTE must be multiple of 8 + 1 due to the position of the load
    L = 8
    a = 5
    P = -10.e3

    E = 203.e9 # Pa
    nu = 0.3
    rho = 7.83e3 # kg/m3

    x = np.linspace(0, L, nnodes)
    y = np.ones_like(x)
    hy = 0.05 # m
    hz = 0.05 # m
    A = hy*hz
    Izz = hz*hy**3/12
    Iyy = hz**3*hy/12
    print('E', E)
    print('Iyy', Iyy)
    print('Izz', Izz)

    ncoords = np.vstack((x, y, np.zeros_like(x))).T
    nids = 1 + np.arange(ncoords.shape[0])
    nid_pos = dict(zip(nids, np.arange(len(nids))))

    n1s = nids[0:-1]
    n2s = nids[1:]

    num_elements = len(n1s)
    print('num_elements', num_elements)

    p = BeamLRProbe()
    data = BeamLRData()

    KC0r = np.zeros(data.KC0_SPARSE_SIZE*num_elements, dtype=INT)
    KC0c = np.zeros(data.KC0_SPARSE_SIZE*num_elements, dtype=INT)
    KC0v = np.zeros(data.KC0_SPARSE_SIZE*num_elements, dtype=DOUBLE)
    N = DOF*nnodes
    print('num_DOF', N)

    prop = BeamProp()
    prop.A = A
    prop.E = E
    scf = 5/6.
    prop.G = scf*E/2/(1+nu)
    prop.Izz = Izz
    prop.Iyy = Iyy
    prop.J = Izz + Iyy

    ncoords_flatten = ncoords.flatten()

    beams = []
    init_k_KC0 = 0
    for n1, n2 in zip(n1s, n2s):
        pos1 = nid_pos[n1]
        pos2 = nid_pos[n2]
        beam = BeamLR(p)
        beam.init_k_KC0 = init_k_KC0
        beam.n1 = n1
        beam.n2 = n2
        beam.c1 = DOF*pos1
        beam.c2 = DOF*pos2
        beam.update_rotation_matrix(1., 1., 0, ncoords_flatten)
        beam.update_probe_xe(ncoords_flatten)
        beam.update_KC0(KC0r, KC0c, KC0v, prop)
        beams.append(beam)
        init_k_KC0 += data.KC0_SPARSE_SIZE

    print('elements created')

    KC0 = coo_matrix((KC0v, (KC0r, KC0c)), shape=(N, N)).tocsc()

    print('sparse KC0 created')

    bk = np.zeros(N, dtype=bool)

    left_end = np.isclose(x, 0.)
    right_end = np.isclose(x, L)

    bk[0::DOF][left_end] = True
    bk[1::DOF][left_end] = True
    bk[2::DOF][left_end] = True
    bk[3::DOF][left_end] = True
    bk[4::DOF][left_end] = True
    bk[5::DOF][left_end] = True

    bu = ~bk

    fext = np.zeros(N)
    load_pos = np.isclose(x, a)
    fext[1::DOF][load_pos] = P
    fext[2::DOF][load_pos] = -P
    print('np.where(load_pos)', np.where(load_pos))

    Kuu = KC0[bu, :][:, bu]

    u = np.zeros(N)
    u[bu] = spsolve(Kuu, fext[bu])

    # NOTE Table G-1, Case 5, of Gere and Goodno (2009), plus the shear
    #      deflection under the load, Eq. (1.2) of Bathe (2014)
    ref_bending = -P*a**2*(3*L - a)/(6*E*Izz)
    ref_value = ref_bending - P*a/(prop.G*A)
    print('tip deflections', u[1::DOF].min(), u[2::DOF].max())
    print('reference', ref_value, 'bending only', ref_bending)
    assert np.isclose(u[1::DOF].min(), -ref_value, rtol=1e-3)
    assert np.isclose(u[2::DOF].max(), +ref_value, rtol=1e-3)

    fint = np.zeros(N)
    for beam in beams:
        beam.update_probe_ue(u)
        beam.update_fint(fint, prop)

    # NOTE adding reaction forces to external force vector
    Kku = KC0[bk, :][:, bu]
    fext[bk] = Kku @ u[bu]
    assert np.allclose(fint, fext, atol=1e-5)
    #print(np.where(np.logical_not(np.isclose(fint, fext, atol=1e-5))))


if __name__ == '__main__':
    test_static_point_load()
