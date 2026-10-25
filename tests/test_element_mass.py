r"""Element mass matrices: total mass, matrices and rigid-body inertia

Every element with ``update_M`` is placed in a generic orientation in space
and its mass matrix `\boldsymbol{M}`, in global coordinates, is projected on the six
rigid-body motions, the three translations and the three rotations about the
global origin, whose nodal values are the columns of `\boldsymbol{\Phi}`. The 6 x 6
matrix `\boldsymbol{\Phi}^\top \boldsymbol{M} \boldsymbol{\Phi}` contains the mass `m`, the first moment of the
mass `\boldsymbol{Q}` and the inertia tensor of the element,

.. math::

    \boldsymbol{\Phi}^\top \boldsymbol{M} \boldsymbol{\Phi} = \begin{bmatrix}
        m \boldsymbol{I} & -\tilde{\boldsymbol{Q}} \\
        \tilde{\boldsymbol{Q}} & \mathrm{tr}(\boldsymbol{I}_2) \boldsymbol{I} - \boldsymbol{I}_2
    \end{bmatrix}

with `\tilde{\boldsymbol{Q}}` the skew-symmetric matrix of `\boldsymbol{Q}` and `\boldsymbol{I}_2
= \int \rho \boldsymbol{p} \boldsymbol{p}^\top dV`. For the shells, `\boldsymbol{p} = \boldsymbol{x} + z
\boldsymbol{n}` and the integrals through the thickness are ``intrho``, ``intrhoz``
and ``intrhoz2`` of the property, with an offset, such that ``intrhoz`` is
not zero. Each scheme of ``update_M`` integrates the kinetic energy with a
rule of points `\boldsymbol{x}_q` and weights `w_q` over the area, for which

.. math::

    m = I_\rho \sum_q w_q, \quad
    \boldsymbol{Q} = I_\rho \sum_q w_q \boldsymbol{x}_q + I_{\rho z} \boldsymbol{n} \sum_q w_q,

    \boldsymbol{I}_2 = I_\rho \sum_q w_q \boldsymbol{x}_q \boldsymbol{x}_q^\top
        + I_{\rho z} \sum_q w_q (\boldsymbol{x}_q \boldsymbol{n}^\top + \boldsymbol{n} \boldsymbol{x}_q^\top)
        + I_{\rho z^2} \boldsymbol{n} \boldsymbol{n}^\top \sum_q w_q

and the exact values of the shell are those of a rule exact for the
quadratic integrands. The lumped mass matrices are the nodal quadrature, with
the points at the nodes.

"""
import sys
sys.path.append('..')

import numpy as np
import pytest
from scipy.sparse import coo_matrix

from pyfe3d.shellprop_utils import isotropic_plate
from pyfe3d.beamprop import BeamProp
from pyfe3d import (Tria3R, Tria3RData, Tria3RProbe, Tria3DSG, Tria3DSGData,
                    Tria3DSGProbe, Quad4R, Quad4RData, Quad4RProbe, Quad4,
                    Quad4Data, Quad4Probe, BeamC, BeamCData, BeamCProbe,
                    BeamLR, BeamLRData, BeamLRProbe, Truss, TrussData,
                    TrussProbe, DOF, INT, DOUBLE)


SHELLS = {
    'Tria3R': (Tria3R, Tria3RData, Tria3RProbe, (0, 1, 2)),
    'Tria3DSG': (Tria3DSG, Tria3DSGData, Tria3DSGProbe, (0, 1)),
    'Quad4R': (Quad4R, Quad4RData, Quad4RProbe, (0, 1, 2)),
    'Quad4': (Quad4, Quad4Data, Quad4Probe, (0, 1, 2)),
}
BEAMS = {
    'BeamC': (BeamC, BeamCData, BeamCProbe),
    'BeamLR': (BeamLR, BeamLRData, BeamLRProbe),
    'Truss': (Truss, TrussData, TrussProbe),
}
TRIANGLES = {
    'right': np.array([[1., 0.5], [2.5, 0.5], [2.5, 3.1]]),
    'distorted': np.array([[1., 0.5], [2.5, 0.7], [3.5, 3.1]]),
}
QUADS = {
    'rectangle': np.array([[1., 0.5], [2.5, 0.5], [2.5, 3.1], [1., 3.1]]),
    'parallelogram': np.array([[0., 0.], [2., 0.3], [2.6, 1.8], [0.6, 1.5]]),
    'distorted': np.array([[1., 0.5], [2.5, 0.7], [3.5, 3.1], [1.5, 3.2]]),
}


def rotation(a, b, c):
    ca, sa, cb, sb, cc, sc = (np.cos(a), np.sin(a), np.cos(b), np.sin(b),
                              np.cos(c), np.sin(c))
    Rz = np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]])
    Ry = np.array([[cb, 0, sb], [0, 1, 0], [-sb, 0, cb]])
    Rx = np.array([[1, 0, 0], [0, cc, -sc], [0, sc, cc]])
    return Rz @ Ry @ Rx


# NOTE a generic orientation and position in space
Q = rotation(0.4, -0.7, 1.1)
X0 = np.array([0.3, -1.2, 2.0])


def skew(p):
    return np.array([[0, -p[2], p[1]], [p[2], 0, -p[0]], [-p[1], p[0], 0]])


def rigid_modes(X):
    r"""Nodal values of the rigid-body motions, translations and rotations
    about the origin, with the nodal coordinates ``X`` of shape (n, 3)"""
    n = X.shape[0]
    Phi = np.zeros((DOF*n, 6))
    for i in range(n):
        Phi[DOF*i:DOF*i+3, 0:3] = np.eye(3)
        Phi[DOF*i:DOF*i+3, 3:6] = -skew(X[i])
        Phi[DOF*i+3:DOF*i+6, 3:6] = np.eye(3)
    return Phi


def rigid_inertia(m, S, S2):
    r"""6 x 6 rigid-body mass matrix from the mass, the first moment and
    the second moments `\int \rho \boldsymbol{p} \boldsymbol{p}^\top dV`"""
    Mrb = np.zeros((6, 6))
    Mrb[:3, :3] = m*np.eye(3)
    Mrb[:3, 3:] = -skew(S)
    Mrb[3:, :3] = skew(S)
    Mrb[3:, 3:] = np.trace(S2)*np.eye(3) - S2
    return Mrb


def mass_moments(Mrb):
    r"""Mass, first moment and inertia tensor about the centre of mass from
    the 6 x 6 rigid-body mass matrix"""
    m = Mrb[0, 0]
    S = np.array([Mrb[5, 1], Mrb[3, 2], Mrb[4, 0]])
    c = S/m
    # parallel-axis theorem, from the origin to the centre of mass
    Jc = Mrb[3:, 3:] - m*(c @ c*np.eye(3) - np.outer(c, c))
    return m, S, Jc


def relerr(A, B):
    return np.abs(A - B).max()/np.abs(B).max()


def rotation_blocks(e, nnodes):
    r = np.array([[e.r11, e.r12, e.r13],
                  [e.r21, e.r22, e.r23],
                  [e.r31, e.r32, e.r33]])
    Rb = np.zeros((DOF*nnodes, DOF*nnodes))
    for i in range(2*nnodes):
        Rb[3*i:3*i+3, 3*i:3*i+3] = r
    return r, Rb


def dense_M(e, data, prop, mtype, nnodes):
    Mr = np.zeros(data.M_SPARSE_SIZE, dtype=INT)
    Mc = np.zeros(data.M_SPARSE_SIZE, dtype=INT)
    Mv = np.zeros(data.M_SPARSE_SIZE, dtype=DOUBLE)
    e.update_M(Mr, Mc, Mv, prop, mtype=mtype)
    return coo_matrix((Mv, (Mr, Mc)),
                      shape=(DOF*nnodes, DOF*nnodes)).toarray()


def shell_element(name, xy):
    Elem, Data, Probe, _ = SHELLS[name]
    nnodes = xy.shape[0]
    Xg = (Q @ np.column_stack([xy, np.zeros(nnodes)]).T).T + X0
    x = Xg.flatten()
    e = Elem(Probe())
    for i in range(nnodes):
        setattr(e, 'n%d' % (i+1), i+1)
        setattr(e, 'c%d' % (i+1), DOF*i)
    e.init_k_M = 0
    e.update_rotation_matrix(x)
    e.update_probe_xe(x)
    return e, Data(), Xg


def quad_N_detJ(xy, xi, eta):
    N = np.array([(1 - xi)*(1 - eta), (1 + xi)*(1 - eta),
                  (1 + xi)*(1 + eta), (1 - xi)*(1 + eta)])/4
    dNxi = np.array([-(1 - eta), (1 - eta), (1 + eta), -(1 + eta)])/4
    dNeta = np.array([-(1 - xi), -(1 + xi), (1 + xi), (1 - xi)])/4
    J = np.array([[dNxi @ xy[:, 0], dNxi @ xy[:, 1]],
                  [dNeta @ xy[:, 0], dNeta @ xy[:, 1]]])
    return N, np.linalg.det(J)


def area(xy):
    if xy.shape[0] == 3:
        (x1, y1), (x2, y2), (x3, y3) = xy
        return ((x2 - x1)*(y3 - y1) - (x3 - x1)*(y2 - y1))/2
    (x1, y1), (x2, y2), (x3, y3), (x4, y4) = xy
    return ((x3 - x1)*(y4 - y2) - (x4 - x2)*(y3 - y1))/2


def area_rule(xy, scheme):
    r"""Points, as values of the shape functions, and weights over the area

    ``scheme`` is ``'exact'``, a rule exact for the integrands of degree 2
    of the shell, ``'nodes'``, the nodal quadrature of ``mtype=1``, or
    ``'centroid'``, the one-point rule of ``mtype=2``.
    """
    nnodes = xy.shape[0]
    A = area(xy)
    if scheme == 'centroid':
        return [(np.ones(nnodes)/nnodes, A)]
    if nnodes == 3:
        if scheme == 'exact':
            # Cowper (1973), Table I, first row, degree of precision 2
            return [(np.array(N), A/3) for N in ([2/3, 1/6, 1/6],
                                                 [1/6, 2/3, 1/6],
                                                 [1/6, 1/6, 2/3])]
        return [(N, A/3) for N in np.eye(3)]
    if scheme == 'exact':
        out = []
        points, weights = np.polynomial.legendre.leggauss(3)
        for xi, wi in zip(points, weights):
            for eta, wj in zip(points, weights):
                N, detJ = quad_N_detJ(xy, xi, eta)
                out.append((N, wi*wj*detJ))
        return out
    # two-point Gauss-Lobatto in each direction, points at the nodes
    return [(N, quad_N_detJ(xy, xi, eta)[1]) for N, (xi, eta) in
            zip(np.eye(4), [(-1, -1), (1, -1), (1, 1), (-1, 1)])]


SCHEME = {0: 'exact', 1: 'nodes', 2: 'centroid'}


def shell_rigid_inertia(Xg, n, prop, rule):
    I0, I1, I2 = prop.intrho, prop.intrhoz, prop.intrhoz2
    m = 0.
    S = np.zeros(3)
    S2 = np.zeros((3, 3))
    for N, w in rule:
        X = N @ Xg
        m += I0*w
        S += w*(I0*X + I1*n)
        S2 += w*(I0*np.outer(X, X) + I1*(np.outer(X, n) + np.outer(n, X))
                 + I2*np.outer(n, n))
    return rigid_inertia(m, S, S2)


def shell_local_matrix(H, prop):
    r"""`h_{ij}` times the inertia of the section, in element coordinates"""
    I0, I1, I2 = prop.intrho, prop.intrhoz, prop.intrhoz2
    B = np.zeros((DOF, DOF))
    B[0, 0] = B[1, 1] = B[2, 2] = I0
    B[3, 3] = B[4, 4] = I2
    B[0, 4] = B[4, 0] = I1
    B[1, 3] = B[3, 1] = -I1
    return np.kron(H, B)


def shell_cases():
    for name, (_, _, _, mtypes) in SHELLS.items():
        geoms = TRIANGLES if name.startswith('Tria') else QUADS
        for gname in geoms:
            for mtype in mtypes:
                yield name, gname, mtype


@pytest.mark.parametrize('name, gname, mtype', list(shell_cases()))
def test_shell_mass(name, gname, mtype):
    xy = TRIANGLES[gname] if name.startswith('Tria') else QUADS[gname]
    nnodes = xy.shape[0]
    A = area(xy)
    # NOTE offset 0.02, such that intrhoz is not zero
    prop = isotropic_plate(thickness=0.05, E=70e9, nu=0.3, rho=2700.,
                           offset=0.02)
    assert prop.intrhoz != 0
    e, data, Xg = shell_element(name, xy)
    r, Rb = rotation_blocks(e, nnodes)
    n = r[:, 2]
    assert np.allclose(np.abs(n), np.abs(Q[:, 2]))
    M = dense_M(e, data, prop, mtype, nnodes)
    assert relerr(M, M.T) < 1e-14

    # NOTE the matrix of the Notes of update_M, h_ij by the scheme of mtype
    rule = area_rule(xy, SCHEME[mtype])
    H = sum(w*np.outer(N, N) for N, w in rule)
    Ml = Rb.T @ M @ Rb
    assert relerr(Ml, shell_local_matrix(H, prop)) < 1e-13
    if mtype == 1:
        # block-diagonal per node, the coupling of intrhoz kept
        assert np.allclose(H, np.diag(np.diag(H)))
        for i in range(nnodes):
            assert np.isclose(Ml[DOF*i, DOF*i + 4], prop.intrhoz*H[i, i])
            assert np.isclose(Ml[DOF*i + 1, DOF*i + 3], -prop.intrhoz*H[i, i])

    Phi = rigid_modes(Xg)
    Mrb = Phi.T @ M @ Phi
    # total mass, exact for every scheme
    assert np.allclose(np.diag(Mrb)[:3], prop.intrho*A, rtol=1e-13)
    # rigid-body inertia of the points and weights of the scheme
    assert relerr(Mrb, shell_rigid_inertia(Xg, n, prop, rule)) < 1e-13
    # against the exact inertia of the shell
    Mrb_exact = shell_rigid_inertia(Xg, n, prop, area_rule(xy, 'exact'))
    S = mass_moments(Mrb)[1]
    S_exact = mass_moments(Mrb_exact)[1]
    # first moment through the thickness, with the offset, for every scheme
    assert np.isclose(S @ n, S_exact @ n, rtol=1e-13)
    if mtype == 0:
        assert relerr(Mrb, Mrb_exact) < 1e-13
    elif name.startswith('Tria') or gname != 'distorted':
        # first moment exact, inertia not
        assert relerr(S, S_exact) < 1e-13
        assert relerr(Mrb[3:, 3:], Mrb_exact[3:, 3:]) > 1e-3
    else:
        # the quadrilateral distorted: the in-plane first moment is not
        # exact with mtype 1 and 2
        assert relerr(S, S_exact) > 1e-3


@pytest.mark.parametrize('name, gname, ratio', [
    ('Tria3R', 'right', 4.), ('Tria3R', 'distorted', 4.),
    ('Tria3DSG', 'right', 4.), ('Tria3DSG', 'distorted', 4.),
    ('Quad4R', 'rectangle', 3.), ('Quad4R', 'parallelogram', 3.),
    ('Quad4', 'rectangle', 3.), ('Quad4', 'parallelogram', 3.),
    ])
def test_lumped_rotary_inertia_about_the_normal(name, gname, ratio):
    r"""The lumped inertia about the normal through the centre of mass

    The drilling rotation has no inertia and the inertia about the normal
    comes only from the distribution of ``intrho`` over the area: lumped at
    the nodes, it is 4 times the exact value for a triangle and 3 times for
    a parallelogram.
    """
    xy = TRIANGLES[gname] if name.startswith('Tria') else QUADS[gname]
    nnodes = xy.shape[0]
    prop = isotropic_plate(thickness=0.05, E=70e9, nu=0.3, rho=2700.,
                           offset=0.02)
    e, data, Xg = shell_element(name, xy)
    n = rotation_blocks(e, nnodes)[0][:, 2]
    Phi = rigid_modes(Xg)
    Mrb_exact = shell_rigid_inertia(Xg, n, prop, area_rule(xy, 'exact'))

    def J_normal(Mrb):
        Jc = mass_moments(Mrb)[2]
        return n @ Jc @ n

    M = dense_M(e, data, prop, 1, nnodes)
    lumped = J_normal(Phi.T @ M @ Phi)
    exact = J_normal(Mrb_exact)
    print(name, gname, 'lumped/exact inertia about the normal', lumped/exact)
    assert np.isclose(lumped/exact, ratio, rtol=1e-12)


def beam_prop(first_moments):
    p = BeamProp()
    b, h, rho = 0.05, 0.1, 7850.
    p.A = b*h
    p.E = 210e9
    p.G = p.E/2.6
    p.Izz = b*h**3/12
    p.Iyy = h*b**3/12
    p.J = p.Izz + p.Iyy
    p.intrho = rho*p.A
    ey, ez = (0.03, -0.02) if first_moments else (0., 0.)
    # NOTE the centroid of the section at (ey, ez) of the element axis
    p.intrhoy = rho*p.A*ey
    p.intrhoz = rho*p.A*ez
    p.intrhoy2 = rho*(p.Izz + p.A*ey**2)
    p.intrhoz2 = rho*(p.Iyy + p.A*ez**2)
    p.intrhoyz = rho*p.A*ey*ez
    return p


def beam_local_matrix(p, L, lumped, truss):
    r"""The matrices of the Notes of update_M of BeamLR and Truss and the
    lumped matrix of BeamC, in element coordinates"""
    H = np.eye(2)*L/2 if lumped else L/6*np.array([[2, 1], [1, 2]])
    B = np.zeros((DOF, DOF))
    B[0, 0] = B[1, 1] = B[2, 2] = p.intrho
    B[3, 3] = p.intrhoy2 + p.intrhoz2
    if not truss:
        B[4, 4] = p.intrhoz2
        B[5, 5] = p.intrhoy2
    if not lumped:
        B[1, 3] = B[3, 1] = -p.intrhoz
        B[2, 3] = B[3, 2] = p.intrhoy
        if not truss:
            B[0, 5] = B[5, 0] = -p.intrhoy
            B[0, 4] = B[4, 0] = p.intrhoz
            B[4, 5] = B[5, 4] = -p.intrhoyz
    return np.kron(H, B)


@pytest.mark.parametrize('name', list(BEAMS))
@pytest.mark.parametrize('mtype', [0, 1])
@pytest.mark.parametrize('first_moments', [False, True])
def test_beam_mass(name, mtype, first_moments):
    Elem, Data, Probe = BEAMS[name]
    p = beam_prop(first_moments)
    x = np.array([0.1, -0.2, 0.3, 1.3, 0.7, -0.4])
    e = Elem(Probe())
    e.n1, e.n2 = 1, 2
    e.c1, e.c2 = 0, DOF
    e.init_k_M = 0
    if name == 'Truss':
        e.update_rotation_matrix(x)
    else:
        e.update_rotation_matrix(0.3, 1.0, 0.5, x)
    e.update_probe_xe(x)
    L = e.length
    M = dense_M(e, Data(), p, mtype, 2)
    assert relerr(M, M.T) < 1e-14
    r, Rb = rotation_blocks(e, 2)
    Ml = Rb.T @ M @ Rb
    if name != 'BeamC' or mtype == 1:
        Mexp = beam_local_matrix(p, L, lumped=(mtype == 1),
                                 truss=(name == 'Truss'))
        assert relerr(Ml, Mexp) < 1e-13

    # rigid-body inertia in element coordinates, node 1 at the origin
    Mrb = rigid_modes(np.array([[0., 0., 0.], [L, 0., 0.]])).T @ Ml @ \
        rigid_modes(np.array([[0., 0., 0.], [L, 0., 0.]]))
    m = p.intrho*L
    assert np.allclose(np.diag(Mrb)[:3], m, rtol=1e-13)
    S_exact = np.array([m*L/2, p.intrhoy*L, p.intrhoz*L])
    S2_exact = np.array([[m*L**2/3, p.intrhoy*L**2/2, p.intrhoz*L**2/2],
                         [p.intrhoy*L**2/2, p.intrhoy2*L, p.intrhoyz*L],
                         [p.intrhoz*L**2/2, p.intrhoyz*L, p.intrhoz2*L]])
    Mrb_exact = rigid_inertia(m, S_exact, S2_exact)
    if mtype == 0 and name != 'Truss':
        assert relerr(Mrb, Mrb_exact) < 1e-13
    elif mtype == 1:
        # two point masses at the nodes, with the rotary inertia of the
        # section and without its first moments
        S2 = np.array([[m*L**2/2, 0, 0], [0, 0, 0], [0, 0, 0]])
        Mrb_lumped = rigid_inertia(m, np.array([m*L/2, 0, 0]), S2)
        Jsec = L*np.diag([p.intrhoy2 + p.intrhoz2,
                          0. if name == 'Truss' else p.intrhoz2,
                          0. if name == 'Truss' else p.intrhoy2])
        Mrb_lumped[3:, 3:] += Jsec
        assert relerr(Mrb, Mrb_lumped) < 1e-13


# NOTE one numbering for every element, 2 only where the element has a
#      reduced-integration mass matrix
MTYPES = {name: mtypes for name, (_, _, _, mtypes) in SHELLS.items()}
MTYPES.update({name: (0, 1) for name in BEAMS})
LABELS = {0: 'consistent', 1: 'lumped', 2: 'reduced integration'}


def mass_element(name):
    r"""One element of each type in a generic orientation, with its data,
    property, number of nodes, area or length, and, for the shells, the
    coordinates in the plane of the element"""
    if name in SHELLS:
        xy = TRIANGLES['right'] if name.startswith('Tria') else QUADS['rectangle']
        e, data, Xg = shell_element(name, xy)
        prop = isotropic_plate(thickness=0.05, E=70e9, nu=0.3, rho=2700.)
        return e, data, prop, xy.shape[0], area(xy), xy
    Elem, Data, Probe = BEAMS[name]
    e = Elem(Probe())
    e.n1, e.n2 = 1, 2
    e.c1, e.c2 = 0, DOF
    e.init_k_M = 0
    x = np.array([0.1, -0.2, 0.3, 1.3, 0.7, -0.4])
    if name == 'Truss':
        e.update_rotation_matrix(x)
    else:
        e.update_rotation_matrix(0.3, 1.0, 0.5, x)
    e.update_probe_xe(x)
    return e, Data(), beam_prop(False), 2, e.length, None


def numbering_cases():
    for name, mtypes in MTYPES.items():
        for mtype in mtypes:
            yield name, mtype


@pytest.mark.parametrize('name, mtype', list(numbering_cases()))
def test_mtype_numbering(name, mtype):
    r"""``mtype`` 0 consistent, 1 lumped and 2 reduced integration

    The integrals `h_{ij}` between the translations `u` of the nodes `i`
    and `j`, in element coordinates, identify the matrix: with ``mtype=0``
    the exact integrals of `S_i S_j`; with ``mtype=1`` no coupling between
    different nodes, a weight at each node; with ``mtype=2``, the one-point
    rule at the centre, every `h_{ij}` equal. All give the exact mass.
    """
    e, data, prop, nnodes, size, xy = mass_element(name)
    M = dense_M(e, data, prop, mtype, nnodes)
    Rb = rotation_blocks(e, nnodes)[1]
    Ml = Rb.T @ M @ Rb
    H =Ml[0::DOF, 0::DOF]/prop.intrho
    assert np.isclose(H.sum(), size, rtol=1e-13)
    if mtype == 0:
        if xy is None:
            H_exact = size/6*np.array([[2., 1.], [1., 2.]])
        else:
            H_exact = sum(w*np.outer(N, N) for N, w in area_rule(xy, 'exact'))
        assert relerr(H, H_exact) < 1e-13
    elif mtype == 1:
        for i in range(nnodes):
            for j in range(nnodes):
                if i != j:
                    block = Ml[DOF*i:DOF*(i + 1), DOF*j:DOF*(j + 1)]
                    assert np.abs(block).max() < 1e-14*np.abs(Ml).max()
        # NOTE the right triangle, the rectangle and the beams: equal weights
        assert np.allclose(np.diag(H), size/nnodes, rtol=1e-13)
    else:
        assert np.allclose(H, size/nnodes**2, rtol=1e-13)


def unsupported():
    for name, mtypes in MTYPES.items():
        for mtype in (-1, 2, 3):
            if mtype not in mtypes:
                yield name, mtype


@pytest.mark.parametrize('name, mtype', list(unsupported()))
def test_unsupported_mtype_raises(name, mtype):
    e, data, prop, nnodes, _, _ = mass_element(name)
    Mr = np.zeros(data.M_SPARSE_SIZE, dtype=INT)
    Mc = np.zeros(data.M_SPARSE_SIZE, dtype=INT)
    Mv = np.zeros(data.M_SPARSE_SIZE, dtype=DOUBLE)
    with pytest.raises(ValueError) as excinfo:
        e.update_M(Mr, Mc, Mv, prop, mtype=mtype)
    # NOTE the message names the element, the value and the valid options
    msg = str(excinfo.value)
    print(msg)
    assert msg.startswith('mtype=%d is not supported by %s.update_M, use '
                          % (mtype, name))
    for value, label in LABELS.items():
        assert ('%d (%s)' % (value, label) in msg) == (value in MTYPES[name])
    if 2 not in MTYPES[name]:
        assert ('%s has no reduced-integration mass matrix (2)' % name) in msg
    # nothing written
    assert not Mr.any() and not Mc.any() and not Mv.any()


if __name__ == '__main__':
    for case in shell_cases():
        test_shell_mass(*case)
    for name in BEAMS:
        for mtype in (0, 1):
            for first_moments in (False, True):
                test_beam_mass(name, mtype, first_moments)
    for case in numbering_cases():
        test_mtype_numbering(*case)
    for case in unsupported():
        test_unsupported_mtype_raises(*case)
