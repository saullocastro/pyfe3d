import sys
sys.path.append('..')

import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.sparse import coo_matrix

from pyfe3d.beamprop import BeamProp
from pyfe3d import Truss, TrussData, TrussProbe, DOF, INT, DOUBLE

def test_truss_static():
    L = 3.

    E = 203.e9 # Pa
    rho = 7.83e3 # kg/m3

    # example with rectangular cross section
    A = 3.e-4 # m^2


    # 3D tetrahedron
    #
    # ^ y
    # |
    # |  /\
    # | /  \
    # |/    \
    # |---------> x
    x = np.array([0, L, L/2, L/2])
    y = np.array([0, 0, L*(3**0.5)/2, L*(3**0.5)/6])
    z = np.array([0, 0, 0, L*(6**0.5)/3])
    ncoords = np.vstack((x, y, z)).T
    nids = 1 + np.arange(ncoords.shape[0])
    nid_pos = dict(zip(nids, np.arange(len(nids))))
    n1s = [1, 1, 1, 2, 2, 3]
    n2s = [2, 3, 4, 3, 4, 4]

    num_elements = len(n1s)
    print('num_elements', num_elements)

    p = TrussProbe()
    data = TrussData()

    KC0r = np.zeros(data.KC0_SPARSE_SIZE*num_elements, dtype=INT)
    KC0c = np.zeros(data.KC0_SPARSE_SIZE*num_elements, dtype=INT)
    KC0v = np.zeros(data.KC0_SPARSE_SIZE*num_elements, dtype=DOUBLE)
    N = DOF*len(x)
    print('num_DOF', N)

    prop = BeamProp()
    prop.A = A
    prop.E = E
    scf = 5/6.
    prop.G = scf*E/2/(1+0.3)
    prop.intrho = rho*A
    prop.intrhoy2 = 0 # used to calculate torsional constant
    prop.intrhoz2 = 0 # used to calculate torsional constant

    ncoords_flatten = ncoords.flatten()

    trusses = []
    init_k_KC0 = 0
    for n1, n2 in zip(n1s, n2s):
        pos1 = nid_pos[n1]
        pos2 = nid_pos[n2]
        truss = Truss(p)
        truss.init_k_KC0 = init_k_KC0
        truss.n1 = n1
        truss.n2 = n2
        truss.c1 = DOF*pos1
        truss.c2 = DOF*pos2
        truss.update_rotation_matrix(ncoords_flatten)
        truss.update_probe_xe(ncoords_flatten)
        truss.update_KC0(KC0r, KC0c, KC0v, prop)
        trusses.append(truss)
        init_k_KC0 += data.KC0_SPARSE_SIZE

    print('elements created')

    KC0 = coo_matrix((KC0v, (KC0r, KC0c)), shape=(N, N)).tocsc()

    print('sparse KC0 created')

    # applying boundary conditions
    bk = np.zeros(N, dtype=bool)

    # supporting base of the tetrahedron
    base = np.isclose(z, 0.)
    top = np.isclose(z, z.max())
    bk[0::DOF][base] = True # u
    bk[1::DOF][base] = True # v
    bk[2::DOF][base] = True # w

    # removing rotations
    bk[3::DOF] = True # rx
    bk[4::DOF] = True # ry
    bk[5::DOF] = True # rz

    bu = ~bk

    Kuu = KC0[bu, :][:, bu]

    # defining external force
    P = -7.
    fext = np.zeros(N)
    fext[2::DOF][top] = P

    # solving
    Kuu = KC0[bu, :][:, bu]

    u = np.zeros(N)
    u[bu] = spsolve(Kuu, fext[bu])

    # verification
    ref_value = P*L/(2*A*E)
    assert np.isclose(ref_value, u[2::DOF].min())

    fint = np.zeros(N)
    for truss in trusses:
        truss.update_probe_ue(u)
        truss.update_fint(fint, prop)

    # NOTE adding reaction forces to external force vector
    Kku = KC0[bk, :][:, bu]
    fext[bk] = Kku @ u[bu]
    assert np.allclose(fint, fext)


if __name__ == '__main__':
    test_truss_static()
