r"""Shells pressed down by a displacement prescribed at a point, traced with
Riks' arc-length method

The deflection is prescribed with ``Analysis(..., bu=bu, uk=uk)`` and
``calc_fext`` returning zeros, and the reaction is ``calc_fint(u)`` at the
prescribed dof. The load-deflection paths of two cases are compared with
literature results:

1. Clamped spherical cap under a central point load, the NAFEMS benchmark of
   Lyons and Holsgrove (2005), reference values as tabulated in the COMSOL
   verification model "Spherical Cap with Central Point Load"; one quarter
   modelled with :class:`.Tria3R` and symmetry conditions.

2. Hinged shallow spherical shell on a square base under a central point
   load, Horrigmoe and Bergan (1978), Fig. 17, and Bergan et al. (1978), Fig.
   4, against the series solution of Leicester (1968) read from these
   figures; one quarter modelled with :class:`.Tria3R`.

Both cases are also traced with the co-rotational kinematics of
:class:`pyfe3d.corotational.CorotationalModel`, ``kinematics='corotational'``,
with :class:`.Quad4R`: :func:`run_cap_corotational` and
:func:`run_leicester_corotational`. The cap is also traced restricted to
the axisymmetric deformations, a sector of a polar mesh with the nodal dofs
``u = T q`` of three dofs per ring, :func:`run_cap_axisymmetric`.

Notes
-----
Case 1: radius ``a = 10`` m, thickness ``th = 0.20384`` m, half-angle 45
degrees (p. 2 of the COMSOL model), ``E = 210`` GPa, ``nu = 0.3``
(Parameters 1, p. 9), clamped edge, nondimensional load ``F a/(2 pi E
th**3)`` and deflection ``w/th`` (Variables, p. 9), reference shell results
in Tables 3 and 4, p. 7. The elements use the von Kármán strains, valid for
moderate rotations: the path agrees with the reference within 5 % up to
``w/th = 6.72``, where the largest rotation of the meridian is about 30
degrees, and departs from it beyond.

Case 2: ``R = 2540`` mm, half-side of the base ``a = 784.9`` mm, thickness
``h = 99.45`` mm, ``E = 68.95`` N/mm2, ``nu = 0.3``, all edges hinged and
immovable (p. 30 of Horrigmoe and Bergan, 1978); Fig. 17 is on p. 33 of
Horrigmoe and Bergan (1978) and Fig. 4 on p. 1688 of Bergan et al. (1978).
The series solution of shallow shell theory of Leicester (1968) and the
limit loads of the flat triangles of Horrigmoe and Bergan are read from the
figures, rendered at 400 dpi, as the mean of the two readings.

References
----------
Lyons, P., Holsgrove, S., Finite Element Benchmarks for 2D Beams and
Axisymmetric Shells Involving Geometric Non-Linearity, NAFEMS, 2005.

COMSOL Multiphysics 6.4 Application Library, Structural Mechanics Module,
Spherical Cap with Central Point Load,
https://doc.comsol.com/6.4/doc/com.comsol.help.models.sme.spherical_cap_with_central_point_load/models.sme.spherical_cap_with_central_point_load.pdf.

Horrigmoe, G., Bergan, P. G., Nonlinear analysis of free-form shells by
flat finite elements, Computer Methods in Applied Mechanics and
Engineering, 16(1), 1978, pp. 11-35,
https://doi.org/10.1016/0045-7825(78)90030-0.

Bergan, P. G., Horrigmoe, G., Krakeland, B., Soreide, T. H., Solution
techniques for non-linear finite element problems, International Journal
for Numerical Methods in Engineering, 12(11), 1978, pp. 1677-1696,
https://doi.org/10.1002/nme.1620121106.

Leicester, R. H., Finite deformations of shallow shells, Journal of the
Engineering Mechanics Division, ASCE, 94(EM6), 1968, pp. 1409-1423.
"""
import sys
sys.path.append('..')

import warnings
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.sparse import coo_matrix

from pyfe3d import (Quad4, Quad4Data, Quad4Probe, Quad4R, Quad4RData,
                    Quad4RProbe, Tria3R, Tria3RData, Tria3RProbe, DOF)
from pyfe3d.shellprop_utils import isotropic_plate

from pyfe3d.solver import Analysis
from pyfe3d.corotational import CorotationalModel

from test_solver_nonlinear_static import Model

#: Parameters 1, p. 9 of the COMSOL model: radius, thickness, Young's
#: modulus and Poisson's ratio; half-angle of the cap, p. 2
CAP = dict(a=10., th=0.20384, E=210e9, nu=0.3, alpha=np.pi/4)

#: Tables 3 and 4, p. 7 of the COMSOL model: reference shell results of
#: the NAFEMS benchmark, nondimensional load and crown deflection ``w/th``,
#: in the order of the path
TABLE_SHELL_REF = np.array([
    [0.332, 2.326], [0.580, 6.720], [0.994, 13.642], [1.502, 18.487],
    [1.757, 20.887], [1.678, 25.668], [3.705, 28.680]])
#: Table 4, p. 7: the COMSOL shell results with 16 elements, the crown
#: deflection computed at the loads of the reference, except at the lowest
#: load after the limit point, 1.717 instead of 1.678 (p. 8)
TABLE_SHELL_COMSOL16 = np.array([
    [0.332, 2.445], [0.580, 6.800], [0.994, 13.800], [1.502, 18.945],
    [1.757, 21.640], [1.717, 25.500], [3.705, 29.410]])
#: Annotations of Fig. 3, p. 5 of the COMSOL model, the shell with 16
#: elements: crown loads F2 = -2.01827e9, -1.91879e9 and -3.11455e9 N at the
#: crown displacements -4.7, -5.2 and -5.8 m (Results, Total Displacement,
#: 3D (shell), pp. 21-22), as ``(F a/(2 pi E th**3), w/th)``; the first is
#: near the maximum of the curve of Fig. 4, p. 6. The mid-surface radius of
#: this shell is ``a + th/2``: the arc of radius ``a + th`` with a layer of
#: thickness ``th`` inwards (Geometry 1, p. 10) and the shell offset to put
#: its top surface on that arc (Thickness and Offset 1, pp. 13-14)
FIG3_SHELL_COMSOL16 = np.array([[1.806, 23.057], [1.717, 25.510],
                                [2.787, 28.454]])
#: Table 1, p. 6: reference solid results of the NAFEMS benchmark
TABLE_SOLID_REF = np.array([
    [0.320, 2.165], [0.584, 6.769], [0.975, 13.335], [1.624, 19.706],
    [1.808, 22.073], [1.758, 24.398], [1.962, 26.788], [4.699, 29.851]])

#: Case 2 of the module docstring: radius, half-side of the square base, thickness [mm], Young's
#: modulus [N/mm2] and Poisson's ratio
LEICESTER = dict(R=2540., a=784.9, h=99.45, E=68.95, nu=0.3)
#: Series solution of case 2, read from the two figures of the module
#: docstring, the mean of the two readings: central load P [kN] and central deflection
#: w [mm] of the rising branch, and the upper and lower limit points as
#: ``(P, w)``; reading uncertainty +-0.4 kN and +-5 mm
LEICESTER_RISING = np.array([[10., 13.9], [15., 21.2], [20., 29.3],
                             [25., 38.1], [30., 47.6]])
LEICESTER_UPPER = (49.25, 138.)
LEICESTER_LOWER = (36.0, 237.)
#: Limit loads [kN] of the flat triangles of case 2, read from the same
#: figures, +-0.4 kN
HORRIGMOE_LIMIT_LOADS = (50.2, 38.1)

ELEMENTS = {'Quad4': (Quad4, Quad4Data, Quad4Probe),
            'Quad4R': (Quad4R, Quad4RData, Quad4RProbe),
            'Tria3R': (Tria3R, Tria3RData, Tria3RProbe)}


def _model(conn, x, prop, element):
    r""":class:`Model` of the elements of connectivity ``conn``"""
    cls, datacls, probecls = ELEMENTS[element]
    data = datacls()
    probe = probecls()
    elements = []
    for k, nodes in enumerate(conn):
        elem = cls(probe)
        for j, nj in enumerate(nodes):
            setattr(elem, 'n%d' % (j + 1), nj)
            setattr(elem, 'c%d' % (j + 1), DOF*nj)
        elem.init_k_KC0 = k*data.KC0_SPARSE_SIZE
        elem.init_k_KCNL = k*data.KCNL_SPARSE_SIZE
        elem.init_k_KG = k*data.KG_SPARSE_SIZE
        elem.update_rotation_matrix(x)
        elem.update_probe_xe(x)
        elements.append(elem)
    N = x.size//3*DOF
    return Model(elements, data, prop, x, N, np.zeros(N))


def _split(quads, alternate):
    r"""Two triangles per quadrilateral, along the diagonal 1-3 where
    ``alternate`` is even and 2-4 elsewhere"""
    conn = []
    for (n1, n2, n3, n4), k in zip(quads, alternate):
        if k % 2 == 0:
            conn += [(n1, n2, n3), (n1, n3, n4)]
        else:
            conn += [(n1, n2, n4), (n2, n3, n4)]
    return conn


def _symmetry(bk, X, Y):
    r"""Symmetry conditions on the planes ``X = 0``, `u = r_y = r_z = 0`,
    and ``Y = 0``, `v = r_x = r_z = 0`"""
    for i in np.flatnonzero(np.isclose(X, 0.)):
        bk[DOF*i + 0] = bk[DOF*i + 4] = bk[DOF*i + 5] = True
    for i in np.flatnonzero(np.isclose(Y, 0.)):
        bk[DOF*i + 1] = bk[DOF*i + 3] = bk[DOF*i + 5] = True


def quarter_disc(n, s=0.5):
    r"""Quadrilaterals of the quarter of the unit disc `x, y \ge 0`, in three
    blocks of ``n x n`` cells: the square ``[0, s]^2`` and the two blocks
    between its edges ``x = s`` and ``y = s`` and the arc, the rows of
    these interpolated linearly between the edge and the arc

    Returns the points, shape ``(num_points, 2)``, and the counterclockwise
    connectivity, shape ``(3*n**2, 4)``.
    """
    pts = []
    quads = []

    def block(P):
        base = sum(len(p) for p in pts)
        ni, nj = P.shape[0] - 1, P.shape[1] - 1
        pts.append(P.reshape(-1, 2))
        ids = base + np.arange((ni + 1)*(nj + 1)).reshape(ni + 1, nj + 1)
        for i in range(ni):
            for j in range(nj):
                quads.append([ids[i, j], ids[i + 1, j], ids[i + 1, j + 1],
                              ids[i, j + 1]])

    t = np.linspace(0, 1, n + 1)
    X, Y = np.meshgrid(s*t, s*t, indexing='ij')
    block(np.stack((X, Y), axis=-1))
    inner = np.stack((s + 0*t, s*t), axis=-1)
    outer = np.stack((np.cos(t*np.pi/4), np.sin(t*np.pi/4)), axis=-1)
    PA = (inner[None, :, :]*(1 - t[:, None, None])
          + outer[None, :, :]*t[:, None, None])
    block(PA)
    # NOTE mirror about y = x, transposed to keep the counterclockwise order
    block(PA[..., ::-1].transpose(1, 0, 2))
    pts = np.vstack(pts)
    # NOTE merge the points shared by the blocks
    key = np.round(pts*1e9).astype(np.int64)
    _, idx, inv = np.unique(key[:, 0]*10**12 + key[:, 1], return_index=True,
                            return_inverse=True)
    return pts[idx], inv.ravel()[np.array(quads)]


def cap_model(n, R, element='Tria3R'):
    r"""Quarter of the clamped spherical cap of radius ``R`` and half-angle
    ``CAP['alpha']``, with the crown at ``(0, 0, R)``

    The points of :func:`quarter_disc` at the polar coordinates `\rho,
    \varphi` are mapped to the sphere at the polar angle `\beta = \rho
    \alpha` and the azimuth `\varphi`. With ``'Tria3R'`` each quadrilateral
    is split in two along alternating diagonals.

    Returns the :class:`Model` with a null load vector, the mask of the free
    dofs ``bu``, with the edge clamped, the symmetry conditions on ``X = 0``
    and ``Y = 0`` and the deflection `w` of the crown known, and the index
    of that dof.
    """
    pq, quads = quarter_disc(n)
    rho = np.hypot(pq[:, 0], pq[:, 1])
    phi = np.arctan2(pq[:, 1], pq[:, 0])
    beta = rho*CAP['alpha']
    x = np.vstack((R*np.sin(beta)*np.cos(phi), R*np.sin(beta)*np.sin(phi),
                   R*np.cos(beta))).T.flatten()
    conn = (_split(quads, range(len(quads))) if element == 'Tria3R'
            else quads)
    prop = isotropic_plate(thickness=CAP['th'], E=CAP['E'], nu=CAP['nu'])
    model = _model(conn, x, prop, element)
    bk = np.zeros(model.N, dtype=bool)
    for i in np.flatnonzero(np.isclose(rho, 1.)):
        bk[DOF*i:DOF*(i + 1)] = True
    _symmetry(bk, pq[:, 0], pq[:, 1])
    dofw = DOF*int(np.flatnonzero(rho == 0.)[0]) + 2
    bk[dofw] = True
    return model, ~bk, dofw


def polar_cap(nr, nsec, dphi, R, shear_factor=1., grading=1.):
    r"""Polar mesh of the sector `0 \le \varphi \le` ``nsec*dphi`` of the
    clamped spherical cap of radius ``R`` and half-angle ``CAP['alpha']``,
    with the crown at ``(0, 0, R)``

    ``nr`` rings at the polar angles `\beta_i = \alpha (i/n_r)^g`, `i = 1
    \dots n_r`, `g` = ``grading``, equally spaced with the default ``1.``
    and refined towards the crown with ``grading > 1``, each with ``nsec +
    1`` nodes at the azimuths `\varphi_j = j \Delta\varphi`, the crown being node 0. The
    first ring is joined to the crown by :class:`.Tria3R` and the other
    rings by :class:`.Quad4R`; each quadrilateral is an isosceles
    trapezoid, hence flat. ``shear_factor`` multiplies the transverse shear
    stiffnesses ``A44``, ``A45`` and ``A55``.

    Returns the nodal coordinates, the :class:`.ShellProp`, the elements,
    the ring index of each node (0 at the crown) and the azimuth of each
    node.
    """
    beta = CAP['alpha']*np.linspace(0, 1, nr + 1)**grading
    phi = np.arange(nsec + 1)*dphi
    B, P = np.meshgrid(beta[1:], phi, indexing='ij')
    xyz = np.vstack(([0., 0., R], np.column_stack((
        R*np.sin(B.ravel())*np.cos(P.ravel()),
        R*np.sin(B.ravel())*np.sin(P.ravel()), R*np.cos(B.ravel())))))
    ring = np.concatenate(([0], np.repeat(np.arange(1, nr + 1), nsec + 1)))
    azimuth = np.concatenate(([0.], P.ravel()))
    ids = 1 + np.arange(nr*(nsec + 1)).reshape(nr, nsec + 1)
    tria = [(0, ids[0, j], ids[0, j + 1]) for j in range(nsec)]
    quads = [(ids[i, j], ids[i + 1, j], ids[i + 1, j + 1], ids[i, j + 1])
             for i in range(nr - 1) for j in range(nsec)]
    x = xyz.flatten()
    prop = isotropic_plate(thickness=CAP['th'], E=CAP['E'], nu=CAP['nu'])
    prop.A44 *= shear_factor
    prop.A45 *= shear_factor
    prop.A55 *= shear_factor
    elements = (_model(tria, x, prop, 'Tria3R').elements
                + _model(quads, x, prop, 'Quad4R').elements)
    return x, prop, elements, ring, azimuth


def axisymmetric_T(ring, azimuth, nr):
    r"""Map ``u = T q`` from the dofs of the axisymmetric deformations to
    the nodal dofs of :func:`polar_cap`

    ``q[0]`` is the deflection `u_z` of the crown, whose other dofs are
    zero, and ``q[1 + 3*(i - 1):4 + 3*(i - 1)]`` are the radial
    displacement `u_r`, the axial displacement `u_z` and the rotation
    `\theta` about the hoop direction `\boldsymbol{e}_\varphi = (-\sin
    \varphi, \cos \varphi, 0)` of the ring `i = 1 \dots n_r - 1`, the ring
    ``nr`` being clamped. A node at the azimuth `\varphi` gets `(u_x, u_y,
    u_z) = (u_r \cos \varphi, u_r \sin \varphi, u_z)` and the rotation vector
    `\theta \boldsymbol{e}_\varphi`.

    Notes
    -----
    The rotational dofs of :class:`.CorotationalModel` are total rotation
    vectors, `\boldsymbol{R} = \exp(\mathrm{Spin}(\boldsymbol{\psi}))`, see
    :mod:`pyfe3d.corotational`. The rotation of the director of a
    torsionless axisymmetric deformation is a rotation by `\theta` about
    the fixed axis `\boldsymbol{e}_\varphi`, whose rotation vector is
    exactly `\theta \boldsymbol{e}_\varphi`, without a drilling part since
    `\boldsymbol{e}_\varphi` is normal to the meridian plane, which holds
    the shell normal. The map is therefore exact for finite displacements
    and rotations, `|\theta| < 2 \pi`.
    """
    rows, cols, vals = [2], [0], [1.]
    for k in np.flatnonzero((ring > 0) & (ring < nr)):
        c = 1 + 3*(ring[k] - 1)
        cp, sp = np.cos(azimuth[k]), np.sin(azimuth[k])
        rows += [DOF*k + d for d in (0, 1, 2, 3, 4)]
        cols += [c, c, c + 1, c + 2, c + 2]
        vals += [cp, sp, 1., -sp, cp]
    return coo_matrix((vals, (rows, cols)),
                      shape=(DOF*ring.size, 1 + 3*(nr - 1))).tocsr()


class Restricted(object):
    r"""Model ``cr`` restricted to the displacements ``u = T q``

    The internal force vector and the tangent stiffness matrices of ``q``
    are ``T.T @ fint(T @ q)`` and ``T.T @ K(T @ q) @ T``, the gradient and
    the Hessian of the strain energy restricted to the subspace, with the
    interface that :class:`.Analysis` expects.
    """
    def __init__(self, cr, T):
        self.cr = cr
        self.T = T
        self.N = T.shape[1]
        self.KC0 = (T.T @ cr.KC0 @ T).tocsr()

    def calc_fext(self, inc=1., silent=True):
        return np.zeros(self.N)

    def calc_fint(self, c, inc=1., silent=True):
        return self.T.T @ self.cr.calc_fint(self.T @ c)

    def calc_kC(self, c=None, NLgeom=False, silent=True):
        if c is None or not NLgeom:
            return self.KC0
        return (self.T.T @ self.cr.calc_kC(c=self.T @ c, NLgeom=True)
                @ self.T).tocsr()

    def calc_kG(self, c=None, NLgeom=False, silent=True):
        if c is None or not NLgeom:
            return 0*self.KC0
        return (self.T.T @ self.cr.calc_kG(c=self.T @ c, NLgeom=True)
                @ self.T).tocsr()


def run_cap_axisymmetric(nr, R, w_max=30., dphi=np.radians(1.),
                         shear_factor=1., grading=1., inc=0.01):
    r"""Path of the cap of :func:`polar_cap`, one sector of angle ``dphi``,
    ``shear_factor`` and ``grading`` as there, with
    ``kinematics='corotational'``, restricted to the axisymmetric
    deformations with :func:`axisymmetric_T` and :class:`Restricted`, the
    crown deflection prescribed up to ``w/th = w_max`` with Riks' method of
    constant arc length ``inc``

    Returns ``w/th`` and the nondimensional crown reaction of the whole cap,
    the reaction of the sector times ``2 pi/dphi``, at the unloaded state
    and at the converged increments, and the :class:`.Analysis`.
    """
    x, prop, elements, ring, azimuth = polar_cap(nr, 1, dphi, R,
                                                 shear_factor, grading)
    cr = CorotationalModel(x, x.size//3*DOF)
    for elem in elements:
        cr.add(elem, prop, kinematics='corotational')
    red = Restricted(cr, axisymmetric_T(ring, azimuth, nr))
    bu = np.ones(red.N, dtype=bool)
    bu[0] = False
    uk = np.zeros(red.N)
    uk[0] = -w_max*CAP['th']
    an = Analysis(red.calc_fext, red.calc_fint, red.calc_kC, red.calc_kG,
                  bu=bu, uk=uk)
    an.NL_method = 'arc_length_riks'
    an.initialInc = an.maxInc = inc
    an.maxArcLength = 1.e6
    with warnings.catch_warnings():
        # NOTE the warning of nodes shared by elements of two kinematics
        #      does not apply, the model has a single kinematics
        warnings.simplefilter('ignore', RuntimeWarning)
        _, cs = an.static(NLgeom=True, silent=True)
    w = np.array([0.] + [-c[0] for c in cs])
    F = np.array([0.] + [-red.calc_fint(c)[0] for c in cs])*2*np.pi/dphi
    return w/CAP['th'], load_factor(F), an


def leicester_model(n, shear_factor=1., element='Tria3R'):
    r"""Quarter `0 \le X, Y \le a` of the hinged spherical shell of
    :data:`LEICESTER`, ``n x n`` cells of the plan, with the nodes on the
    sphere `Z = \sqrt{R^2 - X^2 - Y^2}`; with ``'Tria3R'`` each cell is split
    in two along alternating diagonals

    The edges ``X = a`` and ``Y = a`` are hinged and immovable, `u = v = w =
    0` with the rotations free; the planes ``X = 0`` and ``Y = 0`` have the
    symmetry conditions. ``shear_factor`` multiplies the transverse shear
    stiffnesses ``A44``, ``A45`` and ``A55`` of the :class:`.ShellProp`.

    Returns the :class:`Model` with a null load vector, the mask of the free
    dofs, the central deflection being known, and the index of that dof.
    """
    R, a = LEICESTER['R'], LEICESTER['a']
    xs = np.linspace(0, a, n + 1)
    X, Y = np.meshgrid(xs, xs, indexing='ij')
    X, Y = X.ravel(), Y.ravel()
    x = np.vstack((X, Y, np.sqrt(R**2 - X**2 - Y**2))).T.flatten()
    ids = np.arange(X.size).reshape(n + 1, n + 1)
    quads = np.vstack((ids[:-1, :-1].ravel(), ids[1:, :-1].ravel(),
                       ids[1:, 1:].ravel(), ids[:-1, 1:].ravel())).T
    conn = (_split(quads, [sum(divmod(k, n)) for k in range(len(quads))])
            if element == 'Tria3R' else quads)
    prop = isotropic_plate(thickness=LEICESTER['h'], E=LEICESTER['E'],
                           nu=LEICESTER['nu'])
    prop.A44 *= shear_factor
    prop.A45 *= shear_factor
    prop.A55 *= shear_factor
    model = _model(conn, x, prop, element)
    bk = np.zeros(model.N, dtype=bool)
    for i in np.flatnonzero(np.isclose(X, a) | np.isclose(Y, a)):
        bk[DOF*i:DOF*i + 3] = True
    _symmetry(bk, X, Y)
    dofw = DOF*int(np.flatnonzero((X == 0) & (Y == 0))[0]) + 2
    bk[dofw] = True
    return model, ~bk, dofw


def load_factor(F):
    r"""Nondimensional load ``F a/(2 pi E th**3)`` of the force ``F`` on the
    whole cap, Variables, p. 9 of the COMSOL model"""
    return F*CAP['a']/(2*np.pi*CAP['E']*CAP['th']**3)


def prescribed_path(model, bu, dofw, w_max, initialInc=0.02, maxInc=0.02,
                    NL_method='arc_length_riks', **kwargs):
    r"""Deflection of the dof ``dofw`` prescribed downwards up to ``w_max``,
    without load, traced with ``NL_method``

    Returns the deflections and the reactions of the quarter model times 4,
    at the unloaded state and at the converged increments, the linear
    reaction of the deflection ``w_max``, ``KC0 u`` at the dof since
    ``calc_fint`` gives the non-linear internal forces, and the
    :class:`.Analysis`.
    """
    uk = np.zeros(model.N)
    uk[dofw] = -w_max
    an = model.analysis(bu, uk=uk, NL_method=NL_method,
                        initialInc=initialInc, maxInc=maxInc,
                        maxArcLength=1.e6, **kwargs)
    _, cs = an.static(NLgeom=False, silent=True)
    F_lin = -4*(model.KC0 @ cs[0])[dofw]
    _, cs = an.static(NLgeom=True, silent=True)
    w = np.array([0.] + [-c[dofw] for c in cs])
    F = np.array([0.] + [-4*model.calc_fint(c)[dofw] for c in cs])
    return w, F, F_lin, an


def run_cap(n, R, w_max=30., element='Tria3R', **kwargs):
    r"""Path of the cap of :func:`cap_model` with :func:`prescribed_path`
    up to ``w/th = w_max``

    Returns ``w/th`` and the nondimensional crown reaction of the whole cap,
    the slope of the linear reaction ``dF_n/d(w/th)``, the
    :class:`.Analysis`, the :class:`Model` and the index of the crown dof.
    """
    model, bu, dofw = cap_model(n, R, element)
    th = CAP['th']
    w, F, F_lin, an = prescribed_path(model, bu, dofw, w_max*th, **kwargs)
    return w/th, load_factor(F), load_factor(F_lin)/w_max, an, model, dofw


def run_leicester(n, shear_factor=1., w_max=300., element='Tria3R',
                  **kwargs):
    r"""Path of the shell of :func:`leicester_model` with
    :func:`prescribed_path` up to ``w = w_max`` mm

    Returns the central deflection [mm] and load [kN], the slope of the
    linear reaction [kN/mm], the :class:`.Analysis`, the :class:`Model` and
    the index of the central dof.
    """
    model, bu, dofw = leicester_model(n, shear_factor, element)
    w, F, F_lin, an = prescribed_path(model, bu, dofw, w_max, **kwargs)
    return w, F/1e3, F_lin/1e3/w_max, an, model, dofw


def prescribed_path_corotational(model, bu, dofw, w_max, initialInc=0.02,
                                  maxInc=0.02, NL_method='arc_length_riks'):
    r"""As :func:`prescribed_path`, with the elements of ``model`` added to a
    :class:`.CorotationalModel` with ``kinematics='corotational'`` and the
    default convergence criterion; the co-rotational kinematics take the
    nodal positions relative to the element centroid, so no ``absTOL`` is
    needed against the round-off of the internal force

    Returns the deflections and the reactions of the quarter model times 4,
    the linear reaction of the deflection ``w_max``, the
    :class:`.Analysis` and the :class:`.CorotationalModel`.
    """
    cr = CorotationalModel(model.x, model.N)
    for elem in model.elements:
        cr.add(elem, model.prop, kinematics='corotational')
    uk = np.zeros(cr.N)
    uk[dofw] = -w_max
    an = Analysis(cr.calc_fext, cr.calc_fint, cr.calc_kC, cr.calc_kG, bu=bu,
                  uk=uk)
    an.NL_method = NL_method
    an.initialInc = initialInc
    an.maxInc = maxInc
    an.maxArcLength = 1.e6
    _, cs = an.static(NLgeom=False, silent=True)
    F_lin = -4*(cr.KC0 @ cs[0])[dofw]
    with warnings.catch_warnings():
        # NOTE the warning of nodes shared by elements of two kinematics
        #      does not apply, the model has a single kinematics
        warnings.simplefilter('ignore', RuntimeWarning)
        _, cs = an.static(NLgeom=True, silent=True)
    w = np.array([0.] + [-c[dofw] for c in cs])
    F = np.array([0.] + [-4*cr.calc_fint(c)[dofw] for c in cs])
    return w, F, F_lin, an, cr


def run_cap_corotational(n, R, w_max=30., **kwargs):
    r"""Path of the cap of :func:`cap_model` with :class:`.Quad4R` and
    :func:`prescribed_path_corotational`, returned as by :func:`run_cap`"""
    model, bu, dofw = cap_model(n, R, 'Quad4R')
    th = CAP['th']
    w, F, F_lin, an, cr = prescribed_path_corotational(model, bu, dofw,
                                                       w_max*th, **kwargs)
    return w/th, load_factor(F), load_factor(F_lin)/w_max, an, model, dofw


def run_leicester_corotational(n, shear_factor=1., w_max=300., **kwargs):
    r"""Path of the shell of :func:`leicester_model` with :class:`.Quad4R`
    and :func:`prescribed_path_corotational`, returned as by
    :func:`run_leicester`"""
    model, bu, dofw = leicester_model(n, shear_factor, 'Quad4R')
    w, F, F_lin, an, cr = prescribed_path_corotational(model, bu, dofw,
                                                       w_max, **kwargs)
    return w, F/1e3, F_lin/1e3/w_max, an, model, dofw


def branches(w):
    r"""Slices of the first and of the last part of a path along which the
    deflection ``w`` increases monotonically, before its first and after
    its last decrease"""
    back = np.flatnonzero(np.diff(w) <= 0)
    if back.size == 0:
        return slice(None), slice(None)
    return slice(0, back[0] + 1), slice(back[-1] + 1, None)


def limit_points(w, P, w_split):
    r"""Largest load before ``w_split`` and smallest load after it, as
    ``(P, w)`` pairs of the sampled path"""
    before = w < w_split
    i_max = np.flatnonzero(before)[np.argmax(P[before])]
    i_min = np.flatnonzero(~before)[np.argmin(P[~before])]
    return (P[i_max], w[i_max]), (P[i_min], w[i_min])


def line_y0(model):
    r"""Nodes on the plane ``Y = 0`` ordered by ``X``"""
    xyz = model.x.reshape(-1, 3)
    line = np.flatnonzero(np.isclose(xyz[:, 1], 0.))
    return line[np.argsort(xyz[line, 0])]


def meridian_rotation(model, c, line):
    r"""Largest change of slope, in degrees, of the segments between the
    nodes ``line`` of the plane ``Y = 0``, ordered along it, from the
    undeformed positions to the positions displaced by ``c``"""
    xyz = model.x.reshape(-1, 3)
    xd = xyz + c.reshape(-1, DOF)[:, :3]
    s0 = np.arctan2(np.diff(xyz[line, 2]), np.diff(xyz[line, 0]))
    s1 = np.arctan2(np.diff(xd[line, 2]), np.diff(xd[line, 0]))
    return np.degrees(np.abs(s1 - s0).max())


def compare(w, Fn, table=TABLE_SHELL_REF):
    r"""Relative difference of the load of the path, interpolated linearly
    at the deflections of ``table`` that the path reaches, to the loads of
    ``table``; the path is monotonic in ``w``"""
    assert np.all(np.diff(w) > 0)
    reached = table[:, 1] <= w[-1]
    return np.interp(table[reached, 1], w, Fn)/table[reached, 0] - 1


@pytest.mark.parametrize('offset', [0., 0.5])
def test_cap_moderate_rotations(offset):
    r"""12x12 cells of Tria3R pairs on the quarter, mid-surface radius ``a +
    offset*th``, up to ``w/th = 7``: the load at ``w/th = 2.326`` and
    ``6.720`` of Table 3 within the 5 % tolerance, the published results
    scattering by about 4 %; measured on 2026-10-30, +2.5 % and +2.1 % with
    ``offset = 0``, +1.6 % and +1.2 % with ``offset = 0.5``, and -0.5 % and
    -1.9 % with 24x24 cells and ``offset = 0``. The edge
    reaction balances the crown reaction along ``Z``, the vertical rigid
    translation being an exact zero-energy mode of the von Kármán strains,
    and the rotation of the meridian stays below 35 degrees."""
    w, Fn, slope, an, model, dofw = run_cap(12, CAP['a'] + offset*CAP['th'],
                                            w_max=7., initialInc=0.05,
                                            maxInc=0.05)
    assert an.increments[-1] == 1.
    rel = compare(w, Fn)
    assert rel.size == 2
    assert np.all(np.abs(rel) < 0.05), rel
    fint = model.calc_fint(an.cs[-1])
    assert abs(fint[2::DOF].sum()) < 1e-9*abs(fint[dofw])
    # NOTE secant stiffness below the linear one: the cap softens
    assert Fn[-1] < slope*w[-1]
    assert meridian_rotation(model, an.cs[-1], line_y0(model)) < 35.


@pytest.mark.slow
def test_cap_von_karman_whole_path():
    r"""4x4 cells of Tria3R pairs on the quarter, mid-surface radius ``a``,
    von Kármán strains, the crown deflection prescribed up to ``w/th = 30``,
    beyond the inversion of the cap, with Riks' method: the path reaches the
    load factor 1 with ``w`` increasing monotonically, the edge reaction
    balances the crown reaction along ``Z`` at ``w/th = 30``, the vertical
    rigid translation being an exact zero-energy mode of the von Kármán
    strains, the secant stiffness is below the linear one, and the crown
    reaction is that of the code.

    Notes
    -----
    A coarse mesh, so the reaction is a regression value of the code, not a
    literature value: ``F_n = 2.9986`` at ``w/th = 30``, against the linear
    ``16.00``; measured on hpc12 on 2026-11-12, with the vertical balance
    to 6.8e-14 of the crown reaction.
    """
    w, Fn, slope, an, model, dofw = run_cap(4, CAP['a'], initialInc=0.05,
                                            maxInc=0.05)
    assert an.increments[-1] == 1.
    assert np.all(np.diff(w) > 0)
    fint = model.calc_fint(an.cs[-1])
    assert abs(fint[2::DOF].sum()) < 1e-9*abs(fint[dofw])
    assert Fn[-1] < slope*w[-1]
    assert np.isclose(Fn[-1], 2.9986, rtol=1e-3), Fn[-1]


def test_leicester_upper_limit_load():
    r"""Hinged shell of :data:`LEICESTER`, 16x16 cells of Tria3R pairs on
    the quarter, transverse shear stiffness times 1000, the Kirchhoff limit
    of the shallow shell theory of the references, up to ``w = 300`` mm:
    snap-through, with the upper limit load within 3 % of Leicester (1968),
    the reading uncertainty of 0.8 % plus the 1.9 % of the flat triangles of
    Horrigmoe and Bergan (1978); measured on 2026-10-30: 50.22 kN, +2.0 %."""
    w, P, slope, an, model, dofw = run_leicester(16, shear_factor=1000.)
    assert an.increments[-1] == 1.
    assert np.all(np.diff(w) > 0)
    (Pu, wu), (Pl, wl) = limit_points(w, P, 190.)
    assert abs(Pu/LEICESTER_UPPER[0] - 1) < 0.03, Pu
    assert Pl < 0.85*Pu
    assert P[-1] > Pu


@pytest.mark.slow
def test_cap_corotational_snap_back():
    r"""8x8 cells of :class:`.Quad4R` with ``kinematics='corotational'``,
    mid-surface radius ``a``, the crown deflection prescribed up to ``w/th
    = 30`` with Riks' method of arc length 0.1: past the limit load the
    prescribed deflection turns back, a snap-back of the known dof with a
    decreasing load factor, and the path reaches the load factor 1 on a
    last branch along which ``w`` increases and the inverted cap stiffens,
    the final reaction above twice the limit load. The edge reaction
    balances the crown reaction along ``Z`` at ``w/th = 30``, the vertical
    rigid translation being a zero-energy mode of the co-rotational
    kinematics, and the limit load is that of the code.

    Notes
    -----
    A coarse mesh, so the limit load is a regression value of the code, not
    a literature value: ``F_n = 1.9525`` at ``w/th = 23.74``, the
    deflection turning back from 24.92 to 24.38, and ``F_n = 4.751`` at
    ``w/th = 30``; measured on hpc12 on 2026-11-12. The limit load decreases
    with the mesh, 1.954, 1.915 and 1.894 with 8x8, 10x10 and 12x12 cells
    and the arc length 0.05, measured on the same day.
    """
    model, bu, dofw = cap_model(8, CAP['a'], 'Quad4R')
    th = CAP['th']
    w, F, F_lin, an, cr = prescribed_path_corotational(
        model, bu, dofw, 30.*th, initialInc=0.1, maxInc=0.1)
    w, Fn = w/th, load_factor(F)
    assert an.increments[-1] == 1.
    assert np.any(np.diff(w) < 0)
    first, last = branches(w)
    i_lim = np.argmax(Fn[first])
    F_lim = Fn[first][i_lim]
    assert np.isclose(F_lim, 1.9525, rtol=2e-3), F_lim
    assert Fn[i_lim:].min() < F_lim
    assert np.all(np.diff(w[last]) > 0)
    assert w[last][-1] == pytest.approx(30.)
    assert Fn[-1] > 2*F_lim, Fn[-1]
    fint = cr.calc_fint(an.cs[-1])
    assert abs(fint[2::DOF].sum()) < 1e-8*abs(fint[dofw])


@pytest.mark.slow
def test_leicester_corotational():
    r"""Hinged shell of :data:`LEICESTER`, 16x16 :class:`.Quad4R` with
    ``kinematics='corotational'`` on the quarter, transverse shear
    stiffness times 1000, up to ``w = 300`` mm: the upper limit load within
    3 % and the lower within 7 % of Leicester (1968), the reading
    uncertainty of 1.1 % plus the 5.8 % between Leicester and the flat
    triangles of Horrigmoe and Bergan (1978); measured on 2026-10-30:
    50.38 kN, +2.3 %, and 37.73 kN, +4.8 %, against 49.95 and 38.09 kN
    with ``kinematics='von_karman'``, i.e. :func:`run_leicester` with the
    same mesh of Quad4R."""
    w, P, slope, an, model, dofw = run_leicester_corotational(
        16, shear_factor=1000.)
    assert an.increments[-1] == 1.
    assert np.all(np.diff(w) > 0)
    (Pu, wu), (Pl, wl) = limit_points(w, P, 190.)
    assert abs(Pu/LEICESTER_UPPER[0] - 1) < 0.03, Pu
    assert abs(Pl/LEICESTER_LOWER[0] - 1) < 0.07, Pl


def test_cap_axisymmetric_map_exact():
    r"""The map of :func:`axisymmetric_T` is exact for finite rotations: the
    quarter of :func:`polar_cap`, 16 rings and 4 sectors of 22.5 degrees,
    unrestricted, with ``kinematics='corotational'``, the symmetry
    conditions on ``X = 0`` and ``Y = 0`` and Riks' method, follows the path
    of :func:`run_cap_axisymmetric` with one sector of the same mesh up to
    ``w/th = 20``; the polar mesh is invariant under the rotations by the
    sector angle, so the axisymmetric path is a solution of the
    unrestricted model. Measured on 2026-10-31: largest difference of the
    loads, interpolated at the same ``w``, 4.6e-7 of the largest load; the
    largest rotation of the meridian ``Y = 0`` at ``w/th = 20``, from
    :func:`meridian_rotation`, 61 degrees."""
    nr, nsec, dphi = 16, 4, np.pi/8
    w_max = 20.
    x, prop, elements, ring, azimuth = polar_cap(nr, nsec, dphi, CAP['a'])
    N = x.size//3*DOF
    bk = np.zeros(N, dtype=bool)
    for i in np.flatnonzero(ring == nr):
        bk[DOF*i:DOF*(i + 1)] = True
    _symmetry(bk, x[0::3], x[1::3])
    dofw = 2
    bk[dofw] = True
    # NOTE what prescribed_path_corotational uses of a Model
    model = SimpleNamespace(x=x, N=N, elements=elements, prop=prop)
    w, F, F_lin, an, cr = prescribed_path_corotational(model, ~bk, dofw,
                                                       w_max*CAP['th'])
    assert an.increments[-1] == 1.
    w, Fn = w/CAP['th'], load_factor(F)
    w_ax, Fn_ax, an_ax = run_cap_axisymmetric(nr, CAP['a'], w_max=w_max,
                                              dphi=dphi, inc=0.02)
    assert an_ax.increments[-1] == 1.
    assert np.all(np.diff(w) > 0)
    assert np.abs(np.interp(w, w_ax, Fn_ax) - Fn).max() < 1e-5*Fn.max()


def test_cap_axisymmetric():
    r"""NAFEMS cap of :data:`CAP` restricted to the axisymmetric
    deformations, :func:`run_cap_axisymmetric` with 64 rings, mid-surface
    radius ``a + th/2`` as in the COMSOL model, up to ``w/th = 30``: no
    snap-back, ``w`` increases monotonically (compare
    :func:`test_cap_corotational_snap_back`), and the three points of
    :data:`FIG3_SHELL_COMSOL16` are matched: the limit load within 2 %, the
    load at ``w/th = 25.51`` within 2 % and the deflection at the load
    2.787 within 1 %. Measured on 2026-10-31: the limit load 1.835 at
    ``w/th = 23.21``, +1.6 %, decreasing to 1.828 with 256 rings, +1.2 %,
    see :func:`test_cap_axisymmetric_mesh_trend`; 1.737 at 25.51, +1.1 %;
    ``w/th = 28.61`` at 2.787, +0.6 %."""
    w, Fn, an = run_cap_axisymmetric(64, CAP['a'] + CAP['th']/2, inc=0.02)
    assert an.increments[-1] == 1.
    assert np.all(np.diff(w) > 0)
    (F_max, w_max), (F_min, w_min) = limit_points(w, Fn, 24.5)
    assert abs(F_max/FIG3_SHELL_COMSOL16[0, 0] - 1) < 0.02, F_max
    assert 22. < w_max < 24.5, w_max
    assert F_min < FIG3_SHELL_COMSOL16[1, 0]*1.02, F_min
    F_2 = np.interp(FIG3_SHELL_COMSOL16[1, 1], w, Fn)
    assert abs(F_2/FIG3_SHELL_COMSOL16[1, 0] - 1) < 0.02, F_2
    stiff = w > w_min
    w_3 = np.interp(FIG3_SHELL_COMSOL16[2, 0], Fn[stiff], w[stiff])
    assert abs(w_3/FIG3_SHELL_COMSOL16[2, 1] - 1) < 0.01, w_3


@pytest.mark.slow
def test_cap_axisymmetric_mesh_trend():
    r"""NAFEMS cap of :data:`CAP` restricted to the axisymmetric
    deformations, :func:`run_cap_axisymmetric` with 16, 32 and 64 rings,
    mid-surface radius ``a + th/2``, Riks' method of arc length 0.05, up to
    ``w/th = 30``: on every mesh ``w`` increases monotonically, and the
    limit load decreases with the mesh at second order, the difference
    between the meshes falling by a factor above 3; with the transverse
    shear stiffness times 1000 (``shear_factor``), the Kirchhoff limit, and
    32 rings the limit load is 1.5 to 3 % above that of the actual
    stiffness and the minimum after it lower.

    Notes
    -----
    Values of the code, measured on hpc12 on 2026-11-12: limit loads
    1.9513, 1.8575 and 1.8335, differences 0.094 and 0.024; Kirchhoff limit
    1.8990, +2.2 %, with the minimum 1.7109 against 1.7342. The limit load
    converges to 1.828 with 256 rings and the arc length 0.01 (2026-10-31).
    Under a point load the crown deflection of a shell with transverse
    shear has no converged value, its linear compliance growing by `\ln 2
    \, (h/a) \, 2 (1 + \nu)/\kappa` at each halving of the crown elements
    (derived from the point-load shear deflection `F \ln(r)/(2 \pi \kappa G
    h)`), hence the loads, not the deflections, are compared.
    """
    R = CAP['a'] + CAP['th']/2
    limits = []
    for nr in (16, 32, 64):
        w, Fn, an = run_cap_axisymmetric(nr, R, inc=0.05)
        assert an.increments[-1] == 1.
        assert np.all(np.diff(w) > 0)
        limits.append(limit_points(w, Fn, 24.5))
    d = -np.diff([upper[0] for upper, lower in limits])
    assert np.all(d > 0) and d[0] > 3*d[1], limits
    w, Fn, an = run_cap_axisymmetric(32, R, shear_factor=1000., inc=0.05)
    assert an.increments[-1] == 1.
    assert np.all(np.diff(w) > 0)
    (F_K, w_K), (F_Kmin, w_Kmin) = limit_points(w, Fn, 24.5)
    assert 0.015 < F_K/limits[1][0][0] - 1 < 0.03, F_K
    assert F_Kmin < limits[1][1][0], F_Kmin


if __name__ == '__main__':
    test_cap_moderate_rotations(0.)
    test_cap_moderate_rotations(0.5)
    test_cap_von_karman_whole_path()
    test_leicester_upper_limit_load()
    test_cap_corotational_snap_back()
    test_leicester_corotational()
    test_cap_axisymmetric_map_exact()
    test_cap_axisymmetric()
    test_cap_axisymmetric_mesh_trend()
