r"""Check of the rigid offsets of warped Quad4 and Quad4R (warp_offsets)

The flat element lies in the mean plane of the nodes and node `a` is at the
height `z_a` above its node. The degrees-of-freedom of the flat element are
`J u_e`, `u_f = u - z_a r_y`, `v_f = v + z_a r_x`, the others unchanged
(derived here, from `\delta u^f_a = \delta u_a + \mathrm{Spin}(r_a) \delta
\psi_a` with `r_a = z_a e_3`, the first order of the offsets of
``pyfe3d.corotational.CorotationalShell``).

Checked symbolically here, for the post-steps written in ``quad4.pyx`` and
``quad4r.pyx``, reimplemented line by line in Python:

1. ``_warp_ue`` is `J u_e` and ``_warp_f`` is `J^T f`;
2. ``_warp_K`` (in place) and ``K + _warp_dK`` are `J^T K J` for a general
   symmetric 24x24 `K`;
3. the substitution of ``update_M``, ``intrhoz`` -> ``intrhoz - intrho
   z_j`` in the entries between a translation and the rotation of node `j`
   and ``intrhoz2`` -> ``intrhoz2 - intrhoz (z_i + z_j) + intrho z_i z_j``
   in the entries between the rotations of nodes `i` and `j`, is `J^T M J`;
4. ``_warp_KE`` is the second derivative of the work of the nodal forces
   `n_a` of the flat element on the second-order displacement of the flat
   node, `-(R_a - I) r_a`, at `\psi_a = 0`, and is equal to
   ``dLambdaT_m(0, n x r) - Spin(n) Spin(r)`` of ``pyfe3d.corotational``
   (numerically);
5. `J` is ``CorotationalShell._offset_jacobian`` at `\psi = 0`, in element
   coordinates (numerically);
6. ``_warp_KE_sub``, with which ``update_KCNL`` subtracts the eccentricity
   term from `J^T K_{CNL} J` such that the von Kármán tangent ``KC0 + KCNL
   + KG`` stays exact, removes from a general 24x24 matrix the element
   coordinates ``_warp_KE`` on the rotations of each node;
7. the post-step of ``Quad4Probe.update_BL`` is `B J` for strain rows `B`
   whose `r_x`, `r_y` columns are zero when they have `u` or `v` entries,
   the structure of the rows of the flat element.

Run from the root of the repository: ``python
theory/deriving_equations/quad4r/verify_quad4r_warp_offsets.py``.
"""
import numpy as np
import sympy

DOF = 6


def J_matrix(z):
    J = sympy.eye(24)
    for a in range(4):
        J[DOF*a + 0, DOF*a + 4] = -z[a]
        J[DOF*a + 1, DOF*a + 3] = z[a]
    return J


# NOTE line-by-line copies of the Cython helpers
def warp_ue(z, ue):
    uf = list(ue)
    for a in range(4):
        uf[6*a + 0] = ue[6*a + 0] - z[a]*ue[6*a + 4]
        uf[6*a + 1] = ue[6*a + 1] + z[a]*ue[6*a + 3]
    return uf


def warp_f(z, f):
    f = list(f)
    for a in range(4):
        f[6*a + 3] += z[a]*f[6*a + 1]
        f[6*a + 4] -= z[a]*f[6*a + 0]
    return f


def warp_K(z, K):
    K = list(K)
    for a in range(4):
        for p in range(24):
            K[24*p + 6*a + 3] += z[a]*K[24*p + 6*a + 1]
            K[24*p + 6*a + 4] -= z[a]*K[24*p + 6*a + 0]
    for a in range(4):
        for p in range(24):
            K[24*(6*a + 3) + p] += z[a]*K[24*(6*a + 1) + p]
            K[24*(6*a + 4) + p] -= z[a]*K[24*(6*a + 0) + p]
    return K


def warp_dK(z, K):
    dK = [0]*576
    for a in range(4):
        for p in range(24):
            dK[24*p + 6*a + 3] = z[a]*K[24*p + 6*a + 1]
            dK[24*p + 6*a + 4] = -z[a]*K[24*p + 6*a + 0]
    for a in range(4):
        for p in range(24):
            dK[24*(6*a + 3) + p] += z[a]*(K[24*(6*a + 1) + p]
                                          + dK[24*(6*a + 1) + p])
            dK[24*(6*a + 4) + p] -= z[a]*(K[24*(6*a + 0) + p]
                                          + dK[24*(6*a + 0) + p])
    return dK


def warp_KE_local(z, n):
    nx, ny, nz = n
    return sympy.Matrix([[z*nz, 0, -z*nx/2],
                         [0, z*nz, -z*ny/2],
                         [-z*nx/2, -z*ny/2, 0]])


def warp_KE_sub(z, f, K):
    K = list(K)
    for a in range(4):
        p = 24*(6*a + 3) + 6*a + 3
        K[p] -= z[a]*f[6*a + 2]
        K[p + 2] += 0.5*z[a]*f[6*a + 0]
        K[p + 24 + 1] -= z[a]*f[6*a + 2]
        K[p + 24 + 2] += 0.5*z[a]*f[6*a + 1]
        K[p + 48] += 0.5*z[a]*f[6*a + 0]
        K[p + 48 + 1] += 0.5*z[a]*f[6*a + 1]
    return K


def update_BL_post(z, B):
    B = list(B)
    for i in range(4):
        B[6*i + 3] = z[i]*B[6*i + 1]
        B[6*i + 4] = -z[i]*B[6*i + 0]
    return B


def spin(v):
    return sympy.Matrix([[0, -v[2], v[1]], [v[2], 0, -v[0]],
                         [-v[1], v[0], 0]])


def main():
    z = sympy.symbols('z1:5')
    J = J_matrix(z)

    # 1. displacements and forces
    ue = sympy.symbols('u0:24')
    f = sympy.symbols('f0:24')
    assert sympy.Matrix(warp_ue(z, ue)) - J*sympy.Matrix(ue) == sympy.zeros(24, 1)
    assert sympy.Matrix(warp_f(z, f)) - J.T*sympy.Matrix(f) == sympy.zeros(24, 1)
    print('1. _warp_ue = J ue, _warp_f = J^T f: OK')

    # 2. stiffness, general symmetric K
    K = sympy.zeros(24, 24)
    for i in range(24):
        for j in range(i, 24):
            K[i, j] = K[j, i] = sympy.Symbol('K_%d_%d' % (i, j))
    JKJ = (J.T*K*J).expand()
    Kl = list(K)
    K1 = sympy.Matrix(24, 24, warp_K(z, Kl))
    assert (K1.expand() - JKJ) == sympy.zeros(24, 24)
    dK = sympy.Matrix(24, 24, warp_dK(z, Kl))
    assert ((K + dK).expand() - JKJ) == sympy.zeros(24, 24)
    print('2. _warp_K and K + _warp_dK = J^T K J: OK')

    # 3. mass, element coordinates, M_ij = h_ij m6
    rho, rhoz, rhoz2 = sympy.symbols('intrho intrhoz intrhoz2')
    h = sympy.Matrix(4, 4, lambda i, j: sympy.Symbol('h%d%d' % tuple(sorted((i + 1, j + 1)))))

    def m6(mzr, mzc, mzz):
        r"""Block of the Notes of update_M, with the intrhoz of the
        translation rows (mzr), of the translation columns (mzc) and the
        intrhoz2 (mzz)"""
        return sympy.Matrix([
            [rho, 0, 0, 0, mzr, 0],
            [0, rho, 0, -mzr, 0, 0],
            [0, 0, rho, 0, 0, 0],
            [0, -mzc, 0, mzz, 0, 0],
            [mzc, 0, 0, 0, mzz, 0],
            [0, 0, 0, 0, 0, 0]])
    M = sympy.zeros(24, 24)
    Mw = sympy.zeros(24, 24)
    for i in range(4):
        for j in range(4):
            M[6*i:6*i + 6, 6*j:6*j + 6] = h[i, j]*m6(rhoz, rhoz, rhoz2)
            # NOTE trans-rot: node of the rotation j; rot-trans: node i
            Mw[6*i:6*i + 6, 6*j:6*j + 6] = h[i, j]*m6(
                rhoz - rho*z[j], rhoz - rho*z[i],
                rhoz2 - rhoz*(z[i] + z[j]) + rho*z[i]*z[j])
    assert ((J.T*M*J).expand() - Mw.expand()) == sympy.zeros(24, 24)
    print('3. substitution of update_M = J^T M J: OK')

    # 4. eccentricity geometric stiffness
    za = sympy.Symbol('z')
    n = sympy.Matrix(sympy.symbols('n_x n_y n_z'))
    psi = sympy.Matrix(sympy.symbols('psi_x psi_y psi_z'))
    r = sympy.Matrix([0, 0, za])
    # NOTE the flat node is at x - R r, its displacement u - (R - I) r;
    #      second order of R - I: Spin(psi) + Spin(psi)^2/2
    Sp = spin(psi)
    u2 = -(Sp*Sp/2)*r
    W = (n.T*u2)[0, 0]
    H = sympy.hessian(W, list(psi))
    assert (H - warp_KE_local(za, n)).expand() == sympy.zeros(3, 3)
    # NOTE and the first order is J: -Spin(psi) r = Spin(r) psi
    assert ((-Sp*r) - spin(r)*psi).expand() == sympy.zeros(3, 1)
    print('4. _warp_KE = Hessian of the work of n on the second-order '
          'offset: OK')

    from pyfe3d.corotational import dLambdaT_m
    from pyfe3d.corotational import spin as spin_np
    rng = np.random.default_rng(1)
    for _ in range(5):
        nn = rng.standard_normal(3)
        zz = rng.standard_normal()
        rr = np.array([0., 0., zz])
        KE_cr = dLambdaT_m(np.zeros(3), np.cross(nn, rr)) - spin_np(nn) @ spin_np(rr)
        KE = np.array(warp_KE_local(zz, nn).tolist(), dtype=float)
        assert np.abs(KE_cr - KE).max() < 1e-14*max(1., np.abs(KE).max())
    print('   equal to dLambdaT_m(0, n x r) - Spin(n) Spin(r): OK')

    # 5. J against CorotationalShell._offset_jacobian at psi = 0
    import warnings
    from pyfe3d import Quad4R, Quad4RProbe
    from pyfe3d.shellprop_utils import isotropic_plate
    from pyfe3d.corotational import CorotationalShell, Lambda
    X0 = np.array([[0., 0, 0], [1.0, 0.05, 0.03], [1.1, 0.9, -0.02],
                   [-0.05, 1.0, 0.04]])
    q = Quad4R(Quad4RProbe())
    q.n1, q.n2, q.n3, q.n4 = 0, 1, 2, 3
    q.c1, q.c2, q.c3, q.c4 = 0, 6, 12, 18
    q.update_rotation_matrix(X0.ravel())
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        w = CorotationalShell(q, isotropic_plate(thickness=0.1, E=1., nu=0.3),
                              X0.ravel())
    Jcr, _ = w._offset_jacobian(np.zeros((4, 3)), [Lambda(np.zeros(3))]*4)
    T = np.kron(np.eye(8), w.T0)
    Jloc = T.T @ Jcr @ T
    zl = w.xtilde[:, 2]
    Jn = np.array(J.subs(dict(zip(z, zl))).tolist(), dtype=float)
    assert np.abs(Jloc - Jn).max() < 1e-15
    print('5. J = CorotationalShell._offset_jacobian at psi = 0: OK')

    # 6. K_E subtracted by update_KCNL
    K6 = sympy.Matrix(24, 24, warp_KE_sub(z, f, Kl))
    KE = sympy.zeros(24, 24)
    for a in range(4):
        KE[6*a + 3:6*a + 6, 6*a + 3:6*a + 6] = warp_KE_local(
            z[a], f[6*a:6*a + 3])
    assert (K6 - (K - KE)).expand() == sympy.zeros(24, 24)
    print('6. _warp_KE_sub = K - K_E: OK')

    # 7. rows of update_BL: membrane and drilling rows, with u, v, w and r_z
    #    entries and zero r_x, r_y columns
    for name in ('exx', 'eyy', 'gxy', 'drilling'):
        B = [0]*24
        for a in range(4):
            for c in (0, 1, 2, 5):
                B[6*a + c] = sympy.Symbol('B%s_%d' % (name, 6*a + c))
        BJ = (sympy.Matrix(1, 24, B)*J).expand()
        assert BJ - sympy.Matrix(1, 24, update_BL_post(z, B)) == sympy.zeros(1, 24)
    print('7. post-step of Quad4Probe.update_BL = B J: OK')


if __name__ == '__main__':
    main()
