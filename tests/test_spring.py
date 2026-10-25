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


def test_spring_arbitrary_orientation():
    r"""Spring in an arbitrary orientation against the analytical matrix

    The global matrix must be `[T] [K_e] [T]^T`, with `[K_e]` of the Notes
    of :class:`.Spring` and `[T]` block diagonal with four copies of the
    `3 \times 3` matrix of the entries `T_{ij}` stored in ``r11, ...,
    r33``, whose columns are the element axes (``R`` and ``Rb`` below); the
    displacements of the probe must be `[T]^T u`, the internal forces
    `[K] u`, and the matrix symmetric, without energy for a rigid
    translation and for equal rotations of the two nodes.

    """
    springdata = SpringData()
    springprobe = SpringProbe()
    N = DOF*3
    KC0r = np.zeros(springdata.KC0_SPARSE_SIZE, dtype=INT)
    KC0c = np.zeros(springdata.KC0_SPARSE_SIZE, dtype=INT)
    KC0v = np.zeros(springdata.KC0_SPARSE_SIZE, dtype=DOUBLE)

    stiff = np.array([1.1, 2.3, 3.7, 5.3, 7.1, 11.3])
    spring = Spring(springprobe)
    spring.init_k_KC0 = 0
    # NOTE nodes 2 and 0 of a model with 3 nodes, to check the positions
    spring.c1 = 2*DOF
    spring.c2 = 0*DOF
    (spring.kxe, spring.kye, spring.kze,
     spring.krxe, spring.krye, spring.krze) = stiff
    xaxis = np.array([1., -2., 0.5])
    vxy = np.array([0.3, 0.7, 2.])
    spring.update_rotation_matrix(*xaxis, *vxy)
    spring.update_KC0(KC0r, KC0c, KC0v)
    KC0 = coo_matrix((KC0v, (KC0r, KC0c)), shape=(N, N)).toarray()

    # analytical element coordinate system
    ex = xaxis/np.linalg.norm(xaxis)
    ez = np.cross(ex, vxy)
    ez /= np.linalg.norm(ez)
    ey = np.cross(ez, ex)
    R = np.column_stack((ex, ey, ez))
    assert np.allclose(R, np.array([[spring.r11, spring.r12, spring.r13],
                                    [spring.r21, spring.r22, spring.r23],
                                    [spring.r31, spring.r32, spring.r33]]))
    assert np.allclose(R.T @ R, np.eye(3))
    assert np.isclose(np.linalg.det(R), 1.)
    # vxy is in the element xy plane
    assert np.isclose(ez @ vxy, 0.)

    k = np.diag(stiff)
    Ke = np.block([[k, -k], [-k, k]])
    Rb = np.kron(np.eye(4), R)
    Kel = Rb @ Ke @ Rb.T
    pos = np.r_[2*DOF:3*DOF, 0:DOF]
    Kref = np.zeros((N, N))
    Kref[np.ix_(pos, pos)] = Kel
    assert np.allclose(KC0, Kref, rtol=0, atol=1e-13)
    assert np.allclose(KC0, KC0.T, rtol=0, atol=1e-13)
    # the middle node is not connected
    assert np.allclose(KC0[DOF:2*DOF, :], 0)
    assert np.allclose(KC0[:, DOF:2*DOF], 0)

    # rigid translation and equal rotations of the two nodes: no energy
    rng = np.random.default_rng(7)
    t = rng.random(3)
    th = rng.random(3)
    u = np.zeros(N)
    for c in (0, 2*DOF):
        u[c:c+3] = t
        u[c+3:c+6] = th
    assert np.allclose(KC0 @ u, 0, atol=1e-13)

    # local displacements and internal forces against the analytical ones
    u = rng.random(N)
    spring.update_probe_ue(u)
    ue_ref = Rb.T @ u[pos]
    assert np.allclose(springprobe.ue, ue_ref)
    fint = np.zeros(N)
    spring.update_fint(fint)
    assert np.allclose(springprobe.finte, Ke @ ue_ref)
    assert np.allclose(fint, KC0 @ u)

    # one stiffness at a time: the relative displacement along the element
    # axis of that stiffness, k_i (e_i . (u2 - u1))
    for i in range(6):
        spring.kxe, spring.kye, spring.kze = 0., 0., 0.
        spring.krxe, spring.krye, spring.krze = 0., 0., 0.
        setattr(spring, ('kxe', 'kye', 'kze', 'krxe', 'krye', 'krze')[i], stiff[i])
        KC0v[:] = 0
        spring.update_KC0(KC0r, KC0c, KC0v, update_KC0v_only=1)
        KC0 = coo_matrix((KC0v, (KC0r, KC0c)), shape=(N, N)).toarray()
        e = R[:, i % 3]
        off = 0 if i < 3 else 3
        d = np.zeros(N)
        d[2*DOF+off:2*DOF+off+3] = -e
        d[0+off:0+off+3] = e
        # relative displacement 2 e, energy k_i (2)**2/2
        assert np.isclose(0.5*d @ KC0 @ d, 0.5*stiff[i]*2**2)
        # a relative displacement normal to e_i gives no energy
        for j in range(3):
            if j == i % 3:
                continue
            d = np.zeros(N)
            d[0+off:0+off+3] = R[:, j]
            assert np.isclose(d @ KC0 @ d, 0., atol=1e-13)


def test_spring_rigid_rotation_nodes_apart():
    r"""Rigid rotation of a spring whose nodes are at different positions

    The nodal coordinates are not used by :class:`.Spring`: a rigid rotation
    `\theta` of the two nodes at `p_1` and `p_2` gives the relative
    translation `\theta \times (p_2 - p_1)`, normal to the line of the
    nodes, which has no energy when the element `x` axis is along that line
    and only ``kxe`` and the rotational stiffnesses are non-zero, and has
    energy when ``kye`` or ``kze`` is non-zero.

    """
    springdata = SpringData()
    springprobe = SpringProbe()
    N = DOF*2
    KC0r = np.zeros(springdata.KC0_SPARSE_SIZE, dtype=INT)
    KC0c = np.zeros(springdata.KC0_SPARSE_SIZE, dtype=INT)
    KC0v = np.zeros(springdata.KC0_SPARSE_SIZE, dtype=DOUBLE)
    p1 = np.array([0.2, -0.1, 0.4])
    p2 = np.array([1.3, 0.8, -0.6])
    spring = Spring(springprobe)
    spring.c1 = 0
    spring.c2 = DOF
    spring.kxe = 3.
    spring.krxe = spring.krye = spring.krze = 5.
    spring.update_rotation_matrix(*(p2 - p1), 0., 0., 1.)
    spring.update_KC0(KC0r, KC0c, KC0v)
    KC0 = coo_matrix((KC0v, (KC0r, KC0c)), shape=(N, N)).toarray()
    theta = np.array([0.3, -0.2, 0.5])
    u = np.zeros(N)
    u[0:3] = np.cross(theta, p1)
    u[3:6] = theta
    u[DOF:DOF+3] = np.cross(theta, p2)
    u[DOF+3:DOF+6] = theta
    assert np.allclose(KC0 @ u, 0, atol=1e-13)

    spring.kye = 2.
    KC0v[:] = 0
    spring.update_KC0(KC0r, KC0c, KC0v, update_KC0v_only=1)
    KC0 = coo_matrix((KC0v, (KC0r, KC0c)), shape=(N, N)).toarray()
    assert u @ KC0 @ u > 1e-3


if __name__ == '__main__':
    test_spring()
    test_spring_arbitrary_orientation()
    test_spring_rigid_rotation_nodes_apart()
