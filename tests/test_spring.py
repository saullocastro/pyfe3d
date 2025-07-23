import sys
sys.path.append('..')

import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.sparse import coo_matrix

from pyfe3d import Spring, SpringData, SpringProbe, INT, DOUBLE, DOF


def test_spring():
    springdata = SpringData()
    springprobe = SpringProbe()

    r"""
    building the following spring setup

          node 1     node 2   node 3
    fixed ___o___ k ___o___ x ___o --> force or moment
              \__ k __/

    where this is done for all 6 DOFs, and for each a force or moment is
    applied in the last node

    """
    NUM_ELEMENTS = 3
    NUM_NODES = 3

    KC0r = np.zeros(springdata.KC0_SPARSE_SIZE*NUM_ELEMENTS, dtype=INT)
    KC0c = np.zeros(springdata.KC0_SPARSE_SIZE*NUM_ELEMENTS, dtype=INT)
    KC0v = np.zeros(springdata.KC0_SPARSE_SIZE*NUM_ELEMENTS, dtype=DOUBLE)
    N = DOF*NUM_NODES

    ncoords = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [3, 0, 0],
    ], dtype=np.float64)
    ncoords_flatten = ncoords.flatten()

    n1s = [0, 0, 1]
    n2s = [1, 1, 2]

    init_k_KC0 = 0


    k = 1.
    kr = 3.

    springs = []
    for n1, n2 in zip(n1s, n2s):
        spring = Spring(springprobe)
        spring.init_k_KC0 = init_k_KC0
        spring.n1 = n1
        spring.n2 = n2
        spring.c1 = n1*DOF
        spring.c2 = n2*DOF
        spring.kxe = spring.kye = spring.kze = k
        spring.krxe = spring.krye = spring.krze = kr
        spring.update_rotation_matrix(1, 0, 0, 1, 1, 0) # global coordinates
        spring.update_KC0(KC0r, KC0c, KC0v)
        init_k_KC0 += springdata.KC0_SPARSE_SIZE
        springs.append(spring)

    print('elements created')

    KC0 = coo_matrix((KC0v, (KC0r, KC0c)), shape=(N, N)).tocsc()

    print('sparse KC0 created')

    # supporting first node
    bk = np.zeros(N, dtype=bool)

    check = np.array([1, 0, 0], dtype=bool)
    bk[0::DOF][check] = True
    bk[1::DOF][check] = True
    bk[2::DOF][check] = True
    bk[3::DOF][check] = True
    bk[4::DOF][check] = True
    bk[5::DOF][check] = True

    bu = ~bk

    Kuu = KC0[bu, :][:, bu]

    fext = np.zeros(N)
    load_pos = np.array([0, 0, 1], dtype=bool)
    fext[0::DOF][load_pos] = 3.
    fext[1::DOF][load_pos] = 5.
    fext[2::DOF][load_pos] = 7.
    fext[3::DOF][load_pos] = 11.
    fext[4::DOF][load_pos] = 13.
    fext[5::DOF][load_pos] = 17.

    u = np.zeros(N)
    u[bu] = spsolve(Kuu, fext[bu])

    # checking displacements and rotations of last node
    # u1 = np.zeros(6)
    stiff = np.array([k, k, k, kr, kr, kr])
    f3 = fext[2*DOF:3*DOF]
    u2 = f3/(2*stiff)
    u3 = 2*stiff*u2/stiff + u2

    assert np.allclose(u2, u[1*DOF:2*DOF])
    assert np.allclose(u3, u[2*DOF:3*DOF])

    fint = np.zeros(N)
    for spring in springs:
        spring.update_probe_ue(u)
        spring.update_fint(fint)

    # NOTE adding reaction forces to external force vector
    Kku = KC0[bk, :][:, bu]
    fext[bk] = Kku @ u[bu]
    assert np.allclose(fint, fext)

if __name__ == '__main__':
    test_spring()
