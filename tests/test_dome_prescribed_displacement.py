r"""Shells pressed down by a displacement prescribed at a point, traced with
Riks' arc-length method

The deflection is prescribed with ``Analysis(..., bu=bu, uk=uk)`` and
``calc_fext`` returning zeros, and the reaction is ``calc_fint(u)`` at the
prescribed dof, see the notebook
``doc/source/tut-prescribed-displacement-dome-riks.ipynb``. Two cases:

1. Clamped spherical cap under a central point load, the benchmark of Lyons,
   P. and Holsgrove, S., Finite Element Benchmarks for 2D Beams and
   Axisymmetric Shells Involving Geometric Non-Linearity, NAFEMS, 2005,
   whose reference values are tabulated in the verification model
   "Spherical Cap with Central Point Load" of the COMSOL Multiphysics 6.4
   Application Library, Structural Mechanics Module,
   https://doc.comsol.com/6.4/doc/com.comsol.help.models.sme.spherical_cap_with_central_point_load/models.sme.spherical_cap_with_central_point_load.pdf:
   radius ``a = 10`` m, thickness ``th = 0.20384`` m, half-angle 45 degrees
   (p. 2), ``E = 210`` GPa, ``nu = 0.3`` (Parameters 1, p. 9), clamped edge,
   nondimensional load ``F a/(2 pi E th**3)`` and deflection ``w/th``
   (Variables, p. 9), reference shell results in Tables 3 and 4, p. 7. One
   quarter is modelled with :class:`.Tria3R` and symmetry conditions. The
   elements use the von Kármán strains, valid for moderate rotations: the
   path agrees with the reference within 5 % up to ``w/th = 6.72``, where
   the largest rotation of the meridian is about 30 degrees, and departs
   from it beyond.

2. Hinged shallow spherical shell on a square base under a central point
   load, Fig. 17 (p. 33) of Horrigmoe, G. and Bergan, P. G., Nonlinear
   analysis of free-form shells by flat finite elements, Computer Methods in
   Applied Mechanics and Engineering, 16(1), 11-35, 1978,
   https://doi.org/10.1016/0045-7825(78)90030-0, and Fig. 4 (p. 1688) of
   Bergan, P. G., Horrigmoe, G., Krakeland, B. and Soreide, T. H., Solution
   techniques for non-linear finite element problems, International Journal
   for Numerical Methods in Engineering, 12(11), 1677-1696, 1978,
   https://doi.org/10.1002/nme.1620121106: ``R = 2540`` mm, half-side of
   the base ``a = 784.9`` mm, thickness ``h = 99.45`` mm, ``E = 68.95``
   N/mm2, ``nu = 0.3``, all edges hinged and immovable (p. 30 of Horrigmoe
   and Bergan), compared with the series solution of shallow shell theory of
   Leicester, R. H., Finite deformations of shallow shells, Journal of the
   Engineering Mechanics Division, ASCE, 94(EM6), 1409-1423, 1968, read from
   the figures. One quarter is modelled with :class:`.Tria3R`.

Both cases are also traced with the co-rotational kinematics of
:class:`pyfe3d.corotational.CorotationalModel`, ``kinematics='corotational'``,
with :class:`.Quad4R`: :func:`run_cap_corotational` and
:func:`run_leicester_corotational`.
"""
import sys
sys.path.append('..')

import warnings

import numpy as np
import pytest

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
#: Lyons and Holsgrove (2005), nondimensional load ``F a/(2 pi E th**3)``
#: and crown deflection ``w/th``, in the order of the path
TABLE_SHELL_REF = np.array([
    [0.332, 2.326], [0.580, 6.720], [0.994, 13.642], [1.502, 18.487],
    [1.757, 20.887], [1.678, 25.668], [3.705, 28.680]])
#: Table 4, p. 7: the COMSOL shell results with 16 elements, the crown
#: deflection computed at the loads of the reference, except at the lowest
#: load after the limit point, 1.717 instead of 1.678 (p. 8)
TABLE_SHELL_COMSOL16 = np.array([
    [0.332, 2.445], [0.580, 6.800], [0.994, 13.800], [1.502, 18.945],
    [1.757, 21.640], [1.717, 25.500], [3.705, 29.410]])
#: Table 1, p. 6: reference solid results of Lyons and Holsgrove (2005)
TABLE_SOLID_REF = np.array([
    [0.320, 2.165], [0.584, 6.769], [0.975, 13.335], [1.624, 19.706],
    [1.808, 22.073], [1.758, 24.398], [1.962, 26.788], [4.699, 29.851]])

#: Fig. 17 of Horrigmoe and Bergan (1978) and Fig. 4 of Bergan et al.
#: (1978): radius, half-side of the square base, thickness [mm], Young's
#: modulus [N/mm2] and Poisson's ratio
LEICESTER = dict(R=2540., a=784.9, h=99.45, E=68.95, nu=0.3)
#: Series solution of Leicester (1968), read from Fig. 17 of Horrigmoe and
#: Bergan (1978) and Fig. 4 of Bergan et al. (1978) rendered at 400 dpi,
#: the mean of the two readings: central load P [kN] and central deflection
#: w [mm] of the rising branch, and the upper and lower limit points as
#: ``(P, w)``; reading uncertainty +-0.4 kN and +-5 mm
LEICESTER_RISING = np.array([[10., 13.9], [15., 21.2], [20., 29.3],
                             [25., 38.1], [30., 47.6]])
LEICESTER_UPPER = (49.25, 138.)
LEICESTER_LOWER = (36.0, 237.)
#: Limit loads [kN] of the flat triangles of Horrigmoe and Bergan (1978),
#: read from the same figures, +-0.4 kN
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
def test_cap_von_karman_departure():
    r"""16x16 cells of Tria3R pairs, mid-surface radius ``a``, up to ``w/th =
    30`` with Riks' method: the path reaches the load factor 1, agrees with
    Table 3 within 5 % at ``w/th = 2.326`` and ``6.720`` and is more than
    15 % below it at ``18.487`` and ``20.887``, past the moderate rotations of
    the von Kármán strains; measured on 2026-10-30: +0.8, -0.2, -7.1, -17.9,
    -19.5, +1.7 and -53.0 % at the seven points of Table 3."""
    w, Fn, slope, an, model, dofw = run_cap(16, CAP['a'])
    assert an.increments[-1] == 1.
    rel = compare(w, Fn)
    assert rel.size == 7
    assert np.all(np.abs(rel[:2]) < 0.05), rel
    assert np.all(rel[3:5] < -0.15), rel


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
def test_leicester_limit_loads():
    r"""As :func:`test_leicester_upper_limit_load` with 32x32 cells: the
    upper limit load within 3 % and the lower within 7 % of Leicester
    (1968), the reading uncertainty of 1.1 % plus the 5.8 % between Leicester
    and the flat triangles of Horrigmoe and Bergan (1978), and the rotations
    of the line ``Y = 0`` below 30 degrees; measured on 2026-10-30: 49.70
    kN, +0.9 %, and 37.61 kN, +4.5 %, the largest rotation 15 degrees at the
    upper and 22 degrees at the lower limit point, 27 degrees at ``w = 300``
    mm."""
    w, P, slope, an, model, dofw = run_leicester(32, shear_factor=1000.)
    (Pu, wu), (Pl, wl) = limit_points(w, P, 190.)
    assert abs(Pu/LEICESTER_UPPER[0] - 1) < 0.03, Pu
    assert abs(Pl/LEICESTER_LOWER[0] - 1) < 0.07, Pl
    rot = max(meridian_rotation(model, c, line_y0(model)) for c in an.cs)
    assert rot < 30., rot


@pytest.mark.slow
def test_cap_corotational():
    r"""12x12 cells of :class:`.Quad4R` with ``kinematics='corotational'``,
    mid-surface radius ``a``, up to ``w/th = 30`` with Riks' method: the
    path reaches the load factor 1 and agrees with Table 3 within the 5 %
    of :func:`test_cap_moderate_rotations` at its first five points, up to
    ``w/th = 20.887``, where the rotation of the meridian is 57 degrees; the
    von Kármán strains of :func:`test_cap_von_karman_departure` are 19.5 %
    below the reference there. The point 1.757 at 20.887 of Table 3 is not
    the limit load: the shell of COMSOL with 16 elements gives 1.806 at
    ``w/th = 23.06`` (annotation of Fig. 3, p. 5, -2.01827e9 N at the crown
    displacement -4.7 m), near the maximum of its curve (Fig. 4, p. 6).

    Just past the limit load the axisymmetric path of the quarter model
    becomes unstable: an eigenvalue of the free-free tangent stiffness
    matrix whose eigenvector is antisymmetric about the plane ``X = Y``, a
    `\cos 2\varphi` mode, crosses zero at ``w/th = 24.6``, then one of a
    `\cos 4\varphi` mode at 25.4, and the path follows these
    non-axisymmetric modes, which the axisymmetric reference does not
    have: the crown deflection turns back (a snap-back of the prescribed
    dof) before the cap inverts and stiffens, and the two last points of
    Table 3 are compared on the last branch along which ``w`` increases.
    Measured on 2026-10-30: +0.7, +2.1, +2.5, -0.2 and -0.5 % at the first
    five points; the limit load 1.895 at ``w/th = 23.34``, +4.9 % on the
    1.806 of COMSOL, decreasing with the mesh to 1.873 with 16x16 cells;
    at the last two points +7.8 and -15.9 %, outside the tolerance, the
    inverted cap being so stiff that the deflection at the load of the
    last point, 3.705, is 29.26, 2.0 % above the reference 28.680 and 0.5 %
    below the 29.41 of COMSOL with 16 elements (Table 4)."""
    w, Fn, slope, an, model, dofw = run_cap_corotational(12, CAP['a'])
    assert an.increments[-1] == 1.
    first, last = branches(w)
    rel = compare(w[first], Fn[first])
    assert rel.size == 5
    assert np.all(np.abs(rel) < 0.05), rel
    # NOTE deflection at the load of the last point of Table 3
    w_last = np.interp(TABLE_SHELL_REF[-1, 0], Fn[last], w[last])
    assert abs(w_last/TABLE_SHELL_REF[-1, 1] - 1) < 0.05, w_last


@pytest.mark.slow
def test_leicester_corotational():
    r"""Hinged shell of :data:`LEICESTER`, 16x16 :class:`.Quad4R` with
    ``kinematics='corotational'`` on the quarter, transverse shear
    stiffness times 1000, up to ``w = 300`` mm: the limit loads within the
    3 % and 7 % of :func:`test_leicester_limit_loads`; measured on
    2026-10-30: 50.38 kN, +2.3 %, and 37.73 kN, +4.8 %, against 49.95 and
    38.09 kN with ``kinematics='von_karman'``, i.e. :func:`run_leicester`
    with the same mesh of Quad4R."""
    w, P, slope, an, model, dofw = run_leicester_corotational(
        16, shear_factor=1000.)
    assert an.increments[-1] == 1.
    assert np.all(np.diff(w) > 0)
    (Pu, wu), (Pl, wl) = limit_points(w, P, 190.)
    assert abs(Pu/LEICESTER_UPPER[0] - 1) < 0.03, Pu
    assert abs(Pl/LEICESTER_LOWER[0] - 1) < 0.07, Pl


if __name__ == '__main__':
    test_cap_moderate_rotations(0.)
    test_cap_moderate_rotations(0.5)
    test_cap_von_karman_departure()
    test_leicester_upper_limit_load()
    test_leicester_limit_loads()
    test_cap_corotational()
    test_leicester_corotational()
