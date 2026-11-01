r"""Large-rotation benchmarks of Sze, Liu and Lo (2004) with the co-rotational
(EICR) layer, :mod:`pyfe3d.corotational`

Sze, K. Y., Liu, X. H. and Lo, S. H., Popular benchmark problems for
geometric nonlinear analysis of shells, Finite Elements in Analysis and
Design, 40(11), 1551-1569, 2004, https://doi.org/10.1016/j.finel.2003.11.001
(below "Sze"). The geometry, material, mesh, load and reference values of
each case are read from the paper, at the figure, table and page given in
the docstrings:

1. Section 3.1, Fig. 2a, Table 2a (p. 1556): cantilever under an end shear
   force, BeamC, Quad4R and Tria3R, Newton-Raphson;
2. Section 3.2, Fig. 3a, Table 3a (p. 1557): cantilever under an end moment,
   rolled up into a full circle, BeamC, Quad4R and Tria3R, Newton-Raphson;
3. Section 3.3, Fig. 4a, Table 4a (p. 1558): slit annular plate under a
   lifting line force, Quad4R, Newton-Raphson;
4. Section 3.4, Fig. 5a, Table 5a (p. 1559): hemispherical shell with an
   18 degree hole under alternating radial forces, Quad4R, Newton-Raphson;
5. Section 3.8, Fig. 9a, Tables 9a and 9d (pp. 1564-1565): hinged
   cylindrical roof, Quad4R, Crisfield's arc-length method, against Sze
   and against the von Kármán path of
   ``tests/test_nonlinear_benchmarks_sze2004.py``.

Cases 1, 3 and 5 are also run with ``kinematics='corotational_von_karman'``
and compared with ``'corotational'``.

The models are assembled with :class:`pyfe3d.corotational.CorotationalModel`
and solved with :class:`pyfe3d.solver.Analysis`, unchanged. The rotational
degrees-of-freedom are total rotation vectors, see
:mod:`pyfe3d.corotational`.

Convergence tolerance. The analyses use the default criterion of
:class:`pyfe3d.solver.Analysis`, ``relTOL = 1e-9`` with the round-off
floor of :mod:`pyfe3d.solver.convergence`, and no ``absTOL``. Up to
2026-10-30 the co-rotational elements computed the current nodal positions
as `\boldsymbol{X} + \boldsymbol{u}`, with a round-off of the order of
`\varepsilon |\boldsymbol{K}_{C0}| |\boldsymbol{X}|`, `\boldsymbol{X}` the
nodal coordinates, above the round-off floor: the iterations of the first
increments of the cantilever with 64 BeamC elements, or with 16x1 Tria3R,
stagnated at `\zeta` of 3e-9 to 1e-8, and this module set ``absTOL`` to
ten times `\varepsilon \max_j (|\boldsymbol{K}_{C0}| \boldsymbol{z})_j`,
see :func:`round_off_absTOL`. The positions are now taken relative to the
base centroid of each element, see :class:`pyfe3d.corotational.CorotationalShell`,
and ``test_cantilever_end_shear_translated`` checks that the iterations
no longer depend on the origin of the coordinates.

Policy, as in ``tests/test_nonlinear_benchmarks_sze2004.py``: when the
reference is reached only on fine meshes, the test uses a coarse mesh,
compares its path with the values pre-determined by this same test with
``rtol = 1e-4``, for the platform round-off, and with Sze with the grades
of MacNeal, R. H. and Harder, R. L., A proposed standard set of problems to
test finite element accuracy, Finite Elements in Analysis and Design, 1(1),
3-20, 1985, https://doi.org/10.1016/0168-874X(85)90003-4, p. 14: A for a
normalised error within 2 %, B within 10 %. The convergence with the mesh,
measured once, is given in the docstrings. Times measured on 2026-10-31
with ``OMP_NUM_THREADS=1`` and the compiled :mod:`pyfe3d.corotational`,
several runs in parallel.
"""
import sys
import time
import warnings
sys.path.append('..')

import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from pyfe3d import (BeamC, BeamCProbe, Quad4R, Quad4RProbe, Tria3R,
                    Tria3RProbe, DOF)
from pyfe3d.beamprop import BeamProp
from pyfe3d.shellprop_utils import isotropic_plate
from pyfe3d.solver import Analysis
from pyfe3d.corotational import CorotationalModel

from test_nonlinear_benchmarks_sze2004 import (
    CANTILEVER, TABLE_3A_EXACT, TABLE_9A, TABLE_9D, elastica_tip,
    von_karman_tip, hinged_roof, run_hinged_roof, compare_with_table_9a,
    limit_points, thin_roof_path)

# NOTE Fig. 2a of Sze: L = 10, b = 1, h = 0.1, E = 1.2e6, nu = 0,
#      P_o = E*I/L**2 = 1, P_max = 4*P_o = 4
CANTILEVER_SHEAR = dict(L=10., b=1., h=0.1, E=1.2e6, Pmax=4.)
# NOTE Table 2a of Sze, p. 1556, 16x1 S4R elements: P/P_max, -U_tip, W_tip
#      (the headers "P/P_o" of its second and third column groups are
#      P/P_max, the load levels running from 0.05 to 1.00)
TABLE_2A = np.array([
    [0.05, 0.026, 0.663], [0.10, 0.103, 1.309], [0.15, 0.224, 1.922],
    [0.20, 0.381, 2.493], [0.25, 0.563, 3.015], [0.30, 0.763, 3.488],
    [0.35, 0.971, 3.912], [0.40, 1.184, 4.292], [0.45, 1.396, 4.631],
    [0.50, 1.604, 4.933], [0.55, 1.807, 5.202], [0.60, 2.002, 5.444],
    [0.65, 2.190, 5.660], [0.70, 2.370, 5.855], [0.75, 2.541, 6.031],
    [0.80, 2.705, 6.190], [0.85, 2.861, 6.335], [0.90, 3.010, 6.467],
    [0.95, 3.151, 6.588], [1.00, 3.286, 6.698]])
# NOTE Table 3a of Sze, p. 1557, columns S4R (16x1 S4R elements): M/M_max,
#      -U_tip, W_tip; the exact columns are TABLE_3A_EXACT
TABLE_3A_S4R = np.array([
    [0.05, 0.196, 1.870], [0.10, 0.773, 3.648], [0.15, 1.698, 5.249],
    [0.20, 2.916, 6.600], [0.25, 4.357, 7.643], [0.30, 5.942, 8.338],
    [0.35, 7.582, 8.671], [0.40, 9.191, 8.646], [0.45, 10.687, 8.291],
    [0.50, 12.000, 7.652], [0.55, 13.075, 6.788], [0.60, 13.875, 5.772],
    [0.65, 14.384, 4.678], [0.70, 14.603, 3.583], [0.75, 14.556, 2.556],
    [0.80, 14.280, 1.656], [0.85, 13.826, 0.931], [0.90, 13.254, 0.407],
    [0.95, 12.625, 0.099], [1.00, 12.000, 0.000]])
# NOTE Fig. 4a of Sze: R_i = 6, R_o = 10, h = 0.03, E = 21e6, nu = 0,
#      P_max = 0.8 per unit length
SLIT_PLATE = dict(Ri=6., Ro=10., h=0.03, E=21e6, nu=0., Pmax=0.8)
# NOTE Table 4a of Sze, p. 1558, 10x80 S4R elements: P/P_max, W_A, W_B
TABLE_4A = np.array([
    [0.025, 1.305, 1.789], [0.05, 2.455, 3.370], [0.075, 3.435, 4.720],
    [0.10, 4.277, 5.876], [0.125, 5.007, 6.872], [0.15, 5.649, 7.736],
    [0.20, 6.725, 9.160], [0.25, 7.602, 10.288], [0.30, 8.340, 11.213],
    [0.35, 8.974, 11.992], [0.40, 9.529, 12.661], [0.45, 10.023, 13.247],
    [0.50, 10.468, 13.768], [0.55, 10.876, 14.240], [0.60, 11.257, 14.674],
    [0.65, 11.620, 15.081], [0.70, 11.970, 15.469], [0.75, 12.310, 15.842],
    [0.80, 12.642, 16.202], [0.85, 12.966, 16.550], [0.90, 13.282, 16.886],
    [0.95, 13.590, 17.212], [1.00, 13.891, 17.528]])
# NOTE Fig. 5a of Sze: R = 10, h = 0.04, E = 6.825e7, nu = 0.3, hole of 18
#      degrees at the pole, P_max = 400 for each of the four point forces
HEMISPHERE = dict(R=10., h=0.04, E=6.825e7, nu=0.3, hole=18., Pmax=400.)
# NOTE Table 5a of Sze, p. 1559, 16x16 S4R elements on one quarter: P/P_max,
#      V_A, -U_B
TABLE_5A = np.array([
    [0.05, 0.855, 0.955], [0.10, 1.499, 1.840], [0.15, 1.969, 2.604],
    [0.20, 2.321, 3.261], [0.25, 2.596, 3.833], [0.30, 2.819, 4.339],
    [0.35, 3.002, 4.790], [0.40, 3.158, 5.196], [0.45, 3.291, 5.565],
    [0.50, 3.406, 5.902], [0.55, 3.508, 6.212], [0.60, 3.598, 6.497],
    [0.65, 3.678, 6.761], [0.70, 3.750, 7.006], [0.75, 3.816, 7.234],
    [0.80, 3.875, 7.448], [0.85, 3.929, 7.647], [0.90, 3.979, 7.835],
    [0.95, 4.025, 8.011], [1.00, 4.067, 8.178]])


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def round_off_absTOL(model, bu, factor=10.):
    r"""``factor`` times `\varepsilon \max_j (|\boldsymbol{K}_{C0}|
    \boldsymbol{z})_j` over the free degrees-of-freedom, `\boldsymbol{z}`
    the absolute nodal coordinates at the translations and 1 at the
    rotations: the round-off of the co-rotational internal force computed
    from `\boldsymbol{X} + \boldsymbol{u}`, up to 2026-10-30, see the module
    documentation (derived here). No longer used by :func:`analysis`, kept
    for the ``absTOL`` argument of :func:`analysis` and for the other test
    modules that import it."""
    z = np.ones(model.N)
    for i in range(3):
        z[i::DOF] = np.abs(model.x[i::3])
    return factor*np.finfo(float).eps*(abs(model.KC0) @ z)[bu].max()


def analysis(model, bu, inc, NL_method='NR', maxInc=None, absTOL=None):
    r"""Non-linear static analysis of ``model`` with
    :class:`pyfe3d.solver.Analysis`, increments ``inc``, default convergence
    criterion, ``relTOL = 1e-9`` with the round-off floor, and ``absTOL``

    Returns the load factors and the configurations of the converged
    increments, and the time of the analysis in seconds.
    """
    an = Analysis(model.calc_fext, model.calc_fint, model.calc_kC,
                  model.calc_kG, bu=bu)
    an.NL_method = NL_method
    an.initialInc = inc
    an.maxInc = inc if maxInc is None else maxInc
    an.absTOL = absTOL
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        # NOTE the warning of nodes shared by elements of two kinematics
        #      does not apply, each model has a single kinematics
        warnings.simplefilter('ignore', RuntimeWarning)
        increments, cs = an.static(NLgeom=True, silent=True)
    return np.array(increments), cs, time.perf_counter() - t0


def at_levels(increments, levels):
    r"""Indices of the converged increments at the load factors ``levels``,
    which must all be reached"""
    idx = []
    for level in levels:
        k = np.flatnonzero(np.isclose(increments, level, rtol=0,
                                      atol=1e-12))
        assert k.size == 1, 'load factor %g not reached' % level
        idx.append(k[0])
    return np.array(idx)


def add_shells(model, quads, prop, element, kinematics='corotational'):
    r"""Add to ``model`` one Quad4R per quadrilateral ``(n1, n2, n3, n4)``,
    or two Tria3R, ``(n1, n2, n3)`` and ``(n1, n3, n4)``"""
    for n1, n2, n3, n4 in quads:
        if element == 'Quad4R':
            groups = [(n1, n2, n3, n4)]
        elif element == 'Tria3R':
            groups = [(n1, n2, n3), (n1, n3, n4)]
        else:
            raise ValueError(element)
        for nodes in groups:
            elem = (Quad4R(Quad4RProbe()) if len(nodes) == 4
                    else Tria3R(Tria3RProbe()))
            for i, n in enumerate(nodes):
                setattr(elem, 'n%d' % (i + 1), n)
                setattr(elem, 'c%d' % (i + 1), DOF*n)
            elem.update_rotation_matrix(model.x)
            model.add(elem, prop, kinematics=kinematics)


def cantilever(element, num_elements, L, b=1., h=0.1, E=1.2e6,
               kinematics='corotational', origin=(0., 0., 0.)):
    r"""Cantilever along X, clamped at ``X = 0``, bending in the XZ plane,
    ``nu = 0``: ``num_elements`` BeamC, or a strip of ``num_elements x 1``
    Quad4R or ``2 x num_elements x 1`` Tria3R; all the nodal coordinates
    are translated by ``origin``

    Returns the :class:`CorotationalModel`, the mask of the free
    degrees-of-freedom and the tip nodes.
    """
    if element == 'BeamC':
        prop = BeamProp()
        prop.A = b*h
        prop.E = E
        prop.G = E/2 # nu = 0
        prop.Iyy = b*h**3/12 # bending in the XZ plane
        prop.Izz = h*b**3/12
        prop.J = b*h**3/3
        xs = np.linspace(0, L, num_elements + 1)
        x = (np.vstack((xs, 0*xs, 0*xs)).T + origin).ravel()
        model = CorotationalModel(x)
        probe = BeamCProbe()
        for k in range(num_elements):
            elem = BeamC(probe)
            elem.n1, elem.n2 = k, k + 1
            elem.c1, elem.c2 = DOF*k, DOF*(k + 1)
            elem.update_rotation_matrix(0., 1., 0., x)
            model.add(elem, prop, kinematics=kinematics)
        root, tip = [0], [num_elements]
    else:
        # NOTE nu = 0: the bending stiffness per unit width times b is E*I
        prop = isotropic_plate(thickness=h, E=E, nu=0.)
        nx, ny = num_elements + 1, 2
        xs, ys = np.meshgrid(np.linspace(0, L, nx), np.linspace(0, b, ny),
                             indexing='ij')
        x = (np.vstack((xs.ravel(), ys.ravel(), 0*xs.ravel())).T
             + origin).ravel()
        nids = np.arange(nx*ny).reshape(nx, ny)
        model = CorotationalModel(x)
        add_shells(model, zip(nids[:-1, :-1].ravel(), nids[1:, :-1].ravel(),
                              nids[1:, 1:].ravel(), nids[:-1, 1:].ravel()),
                   prop, element, kinematics)
        root, tip = list(nids[0]), list(nids[-1])
    bu = np.ones(model.N, dtype=bool)
    for n in root:
        bu[DOF*n:DOF*(n + 1)] = False
    return model, bu, tip


def tip_deflections(cs, tip):
    r"""Mean ``-U`` and ``W`` of the tip nodes for each configuration"""
    U = np.array([np.mean([u[DOF*n] for n in tip]) for u in cs])
    W = np.array([np.mean([u[DOF*n + 2] for n in tip]) for u in cs])
    return -U, W


def elastica_end_force(alpha2):
    r"""Tip deflections ``-U/L, W/L`` of the inextensible Euler-Bernoulli
    cantilever under an end force of fixed direction, perpendicular to the
    undeformed axis, ``alpha2 = P L**2/(E I)``

    The moment equilibrium of the part beyond the arc length `s`, `E I
    \theta' = P (x_{tip} - x)`, gives `\theta'' = -\alpha^2 \cos\theta`, `x'
    = \cos\theta`, `z' = \sin\theta` in the arc length `s/L`, with
    `\theta(0) = 0` and `\theta'(1) = 0` (derived here). It is solved by
    shooting on `\theta'(0) = \alpha^2 x_{tip}/L`, bracketed by `(0,
    \alpha^2]`. The small-load limit is `W/L = \alpha^2/3`.
    """
    def ode(s, y):
        theta, kappa, xs, zs = y
        return [kappa, -alpha2*np.cos(theta), np.cos(theta), np.sin(theta)]

    def shoot(kappa0):
        sol = solve_ivp(ode, (0., 1.), [0., kappa0, 0., 0.], rtol=1e-12,
                        atol=1e-14)
        return sol.y[:, -1]

    kappa0 = brentq(lambda k0: shoot(k0)[1], 1e-12, alpha2, xtol=1e-15)
    _, _, x_tip, z_tip = shoot(kappa0)
    return 1 - x_tip, z_tip


def polygon_tip(m, num_elements):
    r"""Tip deflections ``-U/L, W/L`` of the cantilever of Sze, Section 3.2,
    discretised with ``num_elements`` elements, under ``M = m*M_max``: the
    axial and shear forces vanish, every element keeps its length ``l0 =
    L/num_elements`` and turns by ``phi/num_elements``, ``phi = 2*pi*m``,
    the nodes lying on the polygon inscribed in the circle of the exact
    solution (derived here; see ``test_sze_cantilever_half_circle`` in
    ``tests/test_corotational_core.py``)"""
    phi = 2*np.pi*np.atleast_1d(m)[:, None]
    j = np.arange(num_elements) + 0.5
    U = np.cos(j*phi/num_elements).sum(axis=1)/num_elements - 1
    W = np.sin(j*phi/num_elements).sum(axis=1)/num_elements
    return -U, W


# ---------------------------------------------------------------------------
# Section 3.1: cantilever under an end shear force
# ---------------------------------------------------------------------------
def run_cantilever_end_shear(element, num_elements, inc=0.05,
                             kinematics='corotational', origin=(0., 0., 0.)):
    r"""Cantilever of Fig. 2a of Sze under the end force ``P_max = 4`` along
    +Z, of fixed direction, Newton-Raphson in increments of ``inc*P_max``,
    nodal coordinates translated by ``origin``

    Returns ``P/P_max``, ``-U_tip``, ``W_tip`` of the converged increments
    and the time in seconds.
    """
    L, Pmax = CANTILEVER_SHEAR['L'], CANTILEVER_SHEAR['Pmax']
    model, bu, tip = cantilever(element, num_elements, L,
                                kinematics=kinematics, origin=origin)
    for n in tip:
        model.fext[DOF*n + 2] = Pmax/len(tip)
    increments, cs, t = analysis(model, bu, inc)
    U, W = tip_deflections(cs, tip)
    return increments, U, W, t


def test_elastica_end_force():
    r"""The elastica of :func:`elastica_end_force` against its small-load
    limit, ``W/L = alpha2/3``, and against Table 2a of Sze, computed with
    16x1 S4R elements, at ``P_max``: ``P L**2/(E I) = 4``, ``-U = 3.286``,
    ``W = 6.698`` with ``L = 10``; the elastica gives 3.2894 and 6.6996"""
    U, W = elastica_end_force(1e-4)
    assert np.isclose(W, 1e-4/3, rtol=1e-6)
    U, W = elastica_end_force(4.)
    L = CANTILEVER_SHEAR['L']
    print('elastica at P_max: -U %.5f, W %.5f' % (L*U, L*W))
    assert abs(L*U - 3.286) <= 0.005
    assert abs(L*W - 6.698) <= 0.005


@pytest.mark.parametrize('element, num_elements', [
    ('BeamC', 16), ('Quad4R', 16), ('Tria3R', 16)])
def test_cantilever_end_shear(element, num_elements):
    r"""Cantilever of Sze, Section 3.1, Fig. 2a, under the end shear force,
    against Table 2a (16x1 S4R) at its 20 load levels and against the
    elastica of :func:`elastica_end_force`

    The tip rotates by 64 degrees at ``P_max``. Measured with the 16
    elements of Table 2a and increments of ``0.05*P_max`` (no cut-back),
    the largest differences over the 20 levels: from Table 2a, 0.004 (-U)
    and 0.005 (W) with BeamC, within the rounding of its three decimals
    with Quad4R and Tria3R; from the elastica at ``P_max``, -0.02 % and
    +0.05 % (BeamC), -0.10 % and -0.02 % (Quad4R and Tria3R). Times 0.1 s
    (BeamC), 0.2 s (Quad4R), 0.3 s (Tria3R), measured on 2026-10-31 with
    the compiled :mod:`pyfe3d.corotational`. The convergence of BeamC is
    in :func:`test_cantilever_end_shear_convergence`.

    Tolerance: 1 % of the value of Table 2a plus 0.0005 for its rounding,
    and 0.2 % from the elastica at ``P_max``.
    """
    m, U, W, t = run_cantilever_end_shear(element, num_elements)
    print('%s %dx1: %.1f s, %d increments' % (element, num_elements, t,
                                              m.size))
    idx = at_levels(m, TABLE_2A[:, 0])
    for (Pr, Ur, Wr), k in zip(TABLE_2A, idx):
        print('P/Pmax %.2f  -U %.4f (Sze %.3f)  W %.4f (Sze %.3f)'
              % (Pr, U[k], Ur, W[k], Wr))
    np.testing.assert_allclose(U[idx], TABLE_2A[:, 1], rtol=0.01,
                               atol=0.0005)
    np.testing.assert_allclose(W[idx], TABLE_2A[:, 2], rtol=0.01,
                               atol=0.0005)
    L = CANTILEVER_SHEAR['L']
    Ue, We = elastica_end_force(4.)
    assert np.isclose(m[-1], 1.)
    assert abs(U[-1]/(L*Ue) - 1) <= 0.002
    assert abs(W[-1]/(L*We) - 1) <= 0.002


def test_cantilever_end_shear_convergence():
    r"""Tip deflections of the cantilever of Sze, Section 3.1, at ``P_max``
    with 8, 16, 32 and 64 BeamC against the elastica of
    :func:`elastica_end_force`

    Measured: the error of ``-U`` -0.070, -0.017, -0.003 and +0.001 %, of
    observed order 2.07 from 8 to 16 elements; that of ``W`` +0.166,
    +0.046, +0.017 and +0.009 %, of observed order 1.85 from 8 to 16 and
    tending to a floor of about +0.01 %, the transverse shear flexibility
    of BeamC (Timoshenko), absent from the elastica, of the order of ``P
    L/(k G A)/W = 1e-4``. Sze report 8x1 S4R as adequate. Total time 0.6
    s.
    """
    L = CANTILEVER_SHEAR['L']
    Ue, We = elastica_end_force(4.)
    errors = []
    for num_elements in (8, 16, 32, 64):
        m, U, W, t = run_cantilever_end_shear('BeamC', num_elements)
        assert np.isclose(m[-1], 1.)
        errors.append((U[-1]/(L*Ue) - 1, W[-1]/(L*We) - 1))
        print('%2d BeamC: %.1f s, -U %.5f, W %.5f, errors %+.4f %%, '
              '%+.4f %%' % (num_elements, t, U[-1], W[-1],
                            100*errors[-1][0], 100*errors[-1][1]))
    errors = np.array(errors)
    assert np.all(np.diff(np.abs(errors[:, 0])) < 0)
    assert np.all(np.diff(np.abs(errors[:, 1])) < 0)
    order = np.log2(errors[0, 0]/errors[1, 0])
    print('observed order of -U, 8 to 16 elements: %.2f' % order)
    assert abs(order - 2) < 0.2
    assert np.abs(errors[-1]).max() <= 1.5e-4


@pytest.mark.parametrize('element, num_elements', [
    ('BeamC', 16), ('Tria3R', 8)])
def test_cantilever_end_shear_translated(element, num_elements):
    r"""The cantilever of :func:`test_cantilever_end_shear` with its nodal
    coordinates translated by ``(1000, -1000, 1000)``, default criterion
    (``relTOL = 1e-9``, round-off floor, no ``absTOL``): the same converged
    increments and tip deflections as at the origin

    A translation of the origin leaves the linear stiffness matrix, the
    internal and external force vectors and thus the convergence criterion
    unchanged, but not the round-off of positions computed as `\boldsymbol{X}
    + \boldsymbol{u}`, of the order of `\varepsilon |\boldsymbol{X}|`.
    Measured on 2026-10-30 with 64 BeamC elements and the positions computed
    as `\boldsymbol{X} + \boldsymbol{u}`: the iterations of the first
    increment stagnated at `\zeta` of about 1e-8, 1e-7 and 1e-6 with the
    root at ``X = 0``, 100 and 1000, at 0.12 to 0.19 times `\varepsilon \|
    |\boldsymbol{K}_{C0}| |\boldsymbol{X}| \|_{\boldsymbol{d}}`, and no
    increment converged. With the positions relative to the base centroid
    of each element, see :class:`pyfe3d.corotational.CorotationalShell`, the
    iterations are the same at both origins.

    Tolerance: 1e-9 on the tip deflections relative to ``W`` at ``P_max``,
    the order of ``relTOL``.
    """
    m0, U0, W0, _ = run_cantilever_end_shear(element, num_elements)
    m1, U1, W1, _ = run_cantilever_end_shear(element, num_elements,
                                             origin=(1000., -1000., 1000.))
    print('%s %d: %d and %d increments, max differences -U %.1e, W %.1e'
          % (element, num_elements, m0.size, m1.size,
             np.abs(U1 - U0).max(), np.abs(W1 - W0).max()))
    assert np.isclose(m0[-1], 1.) and np.isclose(m1[-1], 1.)
    np.testing.assert_array_equal(m1, m0)
    np.testing.assert_allclose(U1, U0, rtol=0, atol=1e-9*W0[-1])
    np.testing.assert_allclose(W1, W0, rtol=0, atol=1e-9*W0[-1])


def test_cantilever_end_shear_von_karman():
    r"""The same strip of 16x1 Quad4R with ``kinematics='von_karman'``:
    without axial force the von Kármán cantilever bends as the linear one,
    ``W = P L**3/(3 E I)``, and ``u_x = -w_x**2/2`` gives ``U = -(P/(E
    I))**2 L**5/15`` (derived here). At ``P_max`` this is ``W = 13.33``,
    twice Table 2a, and ``U = -10.67``; it is 0.5 % above Table 2a in
    ``W`` at ``0.05*P_max`` (tip rotation 0.1 rad) and 1.8 % at
    ``0.1*P_max``, whereas the co-rotational strip stays within the
    rounding of Table 2a at every level, see
    :func:`test_cantilever_end_shear`. Measured: the von Kármán strip
    within 0.1 % of these closed forms in ``W`` (the transverse shear
    flexibility of Quad4R) and 0.25 % in ``U``, 0.1 s.
    """
    m, U, W, t = run_cantilever_end_shear('Quad4R', 16,
                                          kinematics='von_karman')
    L, b, h, E, Pmax = (CANTILEVER_SHEAR[k]
                        for k in ('L', 'b', 'h', 'E', 'Pmax'))
    EI = E*b*h**3/12
    P = m*Pmax
    W_vk = P*L**3/(3*EI)
    U_vk = (P/EI)**2*L**5/15
    idx = at_levels(m, TABLE_2A[:, 0])
    for (Pr, Ur, Wr), k in zip(TABLE_2A, idx):
        print('P/Pmax %.2f  -U %.4f (closed form %.4f, Sze %.3f)  W %.4f '
              '(closed form %.4f, Sze %.3f)' % (Pr, U[k], U_vk[k], Ur, W[k],
                                                W_vk[k], Wr))
    np.testing.assert_allclose(W, W_vk, rtol=2e-3)
    np.testing.assert_allclose(U, U_vk, rtol=5e-3)
    # NOTE the von Kármán path leaves Table 2a at the second level
    assert W[idx[1]]/TABLE_2A[1, 2] - 1 > 0.015
    assert np.isclose(W[-1]/TABLE_2A[-1, 2], 2., rtol=0.01)


@pytest.mark.parametrize('element', ['BeamC', 'Quad4R', 'Tria3R'])
def test_cantilever_end_shear_corotational_von_karman(element):
    r"""The cantilever of :func:`test_cantilever_end_shear`, 16 elements,
    with ``kinematics='corotational_von_karman'``, against
    ``'corotational'``, Table 2a and the elastica

    Quad4R and Tria3R: their von Kármán terms vanish in the CR frame (see
    :class:`pyfe3d.corotational.CorotationalShell`), so the path is that of
    ``'corotational'`` to round-off. BeamC: its von Kármán axial strain,
    averaged over the element, adds in the CR frame the local
    "higher-order terms" of Crisfield, Vol. 2, Eq. (17.56a), see
    :class:`pyfe3d.corotational.CorotationalBeam`, which make the coarse
    meshes more accurate. Measured at ``P_max`` against the elastica (``-U
    = 3.2894``, ``W = 6.6996``):

    ===========  ================  =========================
    BeamC        corotational      corotational_von_karman
    ===========  ================  =========================
    4 elements   -0.24 %, +0.68 %  -0.013 %, -0.010 %
    8            -0.07 %, +0.17 %  +0.001 %, +0.006 %
    16           -0.02 %, +0.05 %  +0.002 %, +0.007 %
    ===========  ================  =========================

    the same, to the five digits printed, as a prototype of the local
    element with the strain of Eq. (17.56a). With 16 elements the errors
    are at the level of the transverse shear flexibility of BeamC, absent
    from the elastica, see :func:`test_cantilever_end_shear_convergence`.
    With 16 BeamC the largest differences from Table 2a over the 20 levels
    are 0.0035 (``-U``) and 0.0029 (``W``), within the tolerance of
    :func:`test_cantilever_end_shear`; ``-U`` is above and ``W`` below
    that of ``'corotational'`` at every level, by at most 0.08 % and 0.04
    %. The von Kármán path, see
    :func:`test_cantilever_end_shear_von_karman`, ends at twice the ``W``
    of Table 2a. Times 0.3 s (BeamC), 0.6 s (Quad4R), 0.8 s (Tria3R), and
    0.2, 0.3 and 0.4 s with ``'corotational'``.

    Following the policy of the module documentation, BeamC is compared
    with its values at ``P_max`` pre-determined by this same test with
    ``rtol = 1e-4``. Up to pyfe3d 0.11 the pointwise von Kármán strain of
    BeamC locked in membrane, with -59.0 %, -33.3 % (4 elements), -15.6 %,
    -7.5 % (8) and -1.48 %, -0.69 % (16).
    """
    L = CANTILEVER_SHEAR['L']
    Ue, We = elastica_end_force(4.)
    m, U, W, t = run_cantilever_end_shear(
        element, 16, kinematics='corotational_von_karman')
    mc, Uc, Wc, tc = run_cantilever_end_shear(element, 16)
    print('%s 16x1 corotational_von_karman: %.1f s, corotational: %.1f s'
          % (element, t, tc))
    idx = at_levels(m, TABLE_2A[:, 0])
    idxc = at_levels(mc, TABLE_2A[:, 0])
    for (Pr, Ur, Wr), k, kc in zip(TABLE_2A, idx, idxc):
        print('P/Pmax %.2f  -U %.4f (corotational %.4f, Sze %.3f)  W %.4f '
              '(corotational %.4f, Sze %.3f)' % (Pr, U[k], Uc[kc], Ur, W[k],
                                                 Wc[kc], Wr))
    print('at P_max against the elastica: -U %+.3f %%, W %+.3f %%'
          % (100*(U[-1]/(L*Ue) - 1), 100*(W[-1]/(L*We) - 1)))
    assert np.isclose(m[-1], 1.)
    if element == 'BeamC':
        # NOTE values pre-determined with 16 BeamC, see the docstring
        assert np.isclose(U[-1], 3.28948, rtol=1e-4)
        assert np.isclose(W[-1], 6.70008, rtol=1e-4)
        # NOTE slightly more flexible than the elastica, by the transverse
        #      shear flexibility of BeamC, +0.002 % and +0.007 %, and closer
        #      to it than 'corotational'
        assert 0 < U[-1]/(L*Ue) - 1 < 1e-4
        assert 0 < W[-1]/(L*We) - 1 < 1e-4
        assert abs(U[-1]/(L*Ue) - 1) < abs(Uc[-1]/(L*Ue) - 1)
        assert abs(W[-1]/(L*We) - 1) < abs(Wc[-1]/(L*We) - 1)
        assert np.all(U[idx] > Uc[idxc])
        assert np.all(W[idx] < Wc[idxc])
        np.testing.assert_allclose(U[idx], TABLE_2A[:, 1], rtol=0.01,
                                   atol=0.0005)
        np.testing.assert_allclose(W[idx], TABLE_2A[:, 2], rtol=0.01,
                                   atol=0.0005)
    else:
        np.testing.assert_allclose(U[idx], Uc[idxc], rtol=1e-9)
        np.testing.assert_allclose(W[idx], Wc[idxc], rtol=1e-9)
        np.testing.assert_allclose(U[idx], TABLE_2A[:, 1], rtol=0.01,
                                   atol=0.0005)
        np.testing.assert_allclose(W[idx], TABLE_2A[:, 2], rtol=0.01,
                                   atol=0.0005)

# ---------------------------------------------------------------------------
# Section 3.2: cantilever under an end moment, rolled up into a circle
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('element, inc', [
    ('BeamC', 0.05), ('Quad4R', 0.025), ('Tria3R', 0.05)])
def test_cantilever_end_moment_full_circle(element, inc):
    r"""Cantilever of Sze, Section 3.2, Fig. 3a (``L = 12``, ``b = 1``,
    ``h = 0.1``, ``E = 1.2e6``, ``nu = 0``), 16 elements as Table 3a, under
    the end moment ``M_max = 2*pi*E*I/L`` that rolls it up into a full
    circle, the tip rotation reaching `2 \pi`, Newton-Raphson

    The rotation-vector map is singular at `|\boldsymbol{\psi}| = 2 \pi`,
    see :mod:`pyfe3d.corotational`. Following decision D1 of the design,
    the out-of-plane rotations `\psi_X` and `\psi_Z` are restrained at all
    the nodes, which is exact for this plane problem. The moment is
    conjugate to the rotation vector, which is the moment about the fixed
    axis -Y here, since `\boldsymbol{\Lambda}^T \boldsymbol{n} =
    \boldsymbol{n}` for `\boldsymbol{\psi} \parallel \boldsymbol{n}`, see
    :meth:`CorotationalModel.add_moment`. (Measured: without the restraint
    the paths are the same, the out-of-plane residuals vanishing by
    symmetry.)

    The discretised solution is the polygon of :func:`polygon_tip`, which
    differs from the exact circle of Table 3a by at most 0.0011 L. Measured
    at the 20 levels of Table 3a: all three elements on the polygon within
    1e-8 L, and within 0.0005 of the S4R columns of Table 3a, also 16x1;
    0.2 s (BeamC), 0.6 s (Quad4R, increments of 0.025: with 0.05 Newton-
    Raphson cuts back three increments and misses the levels), 0.4 s
    (Tria3R).

    The von Kármán path of ``tests/test_nonlinear_benchmarks_sze2004.py``,
    :func:`von_karman_tip`, verified there with the same strip of Quad4R,
    is printed for comparison: it leaves the exact solution by 0.6 % of L
    at ``0.1*M_max`` and ends at ``W = pi*L`` instead of 0.
    """
    L, b, h, E = (CANTILEVER[k] for k in ('L', 'b', 'h', 'E'))
    EI = E*b*h**3/12
    Mmax = 2*np.pi*EI/L
    num_elements = 16
    model, bu, tip = cantilever(element, num_elements, L, b, h, E)
    # NOTE the moment about -Y bends the cantilever towards +Z
    for n in tip:
        model.add_moment(n, [0., -Mmax/len(tip), 0.])
    bu[3::DOF] = False
    bu[5::DOF] = False
    m, cs, t = analysis(model, bu, inc)
    print('%s %dx1: %.1f s, %d increments' % (element, num_elements, t,
                                              m.size))
    U, W = tip_deflections(cs, tip)
    U, W = U/L, W/L
    idx = at_levels(m, TABLE_3A_EXACT[:, 0])
    mt = TABLE_3A_EXACT[:, 0]
    U_ex, W_ex = elastica_tip(mt)
    U_pol, W_pol = polygon_tip(mt, num_elements)
    U_vk, W_vk = von_karman_tip(mt)
    for i, k in enumerate(idx):
        print('M/Mmax %.2f  -U/L %.5f (polygon %.5f, exact %.5f, von Kármán '
              '%.5f)  W/L %.5f (polygon %.5f, exact %.5f, von Kármán %.5f)'
              % (mt[i], U[k], U_pol[i], U_ex[i], U_vk[i], W[k], W_pol[i],
                 W_ex[i], W_vk[i]))
    assert np.isclose(m[-1], 1.)
    # NOTE the tip turns by 2*pi
    assert np.isclose(np.mean([cs[-1][DOF*n + 4] for n in tip]), -2*np.pi,
                      rtol=1e-8)
    np.testing.assert_allclose(U[idx], U_pol, rtol=0, atol=1e-7)
    np.testing.assert_allclose(W[idx], W_pol, rtol=0, atol=1e-7)
    # NOTE the polygon against the exact solution of Table 3a, and the
    #      computed tip against the S4R columns of Table 3a
    assert np.abs(U_pol - U_ex).max() <= 0.0015
    assert np.abs(W_pol - W_ex).max() <= 0.0015
    np.testing.assert_allclose(L*U[idx], TABLE_3A_S4R[:, 1], rtol=0,
                               atol=0.001)
    np.testing.assert_allclose(L*W[idx], TABLE_3A_S4R[:, 2], rtol=0,
                               atol=0.001)
    # NOTE the von Kármán path leaves the exact one at 0.1*M_max
    second = np.isclose(mt, 0.10)
    assert abs(W_vk[second] - W_ex[second])[0] > 0.005
    assert abs(W[idx][second] - W_ex[second])[0] < 0.001
    assert np.isclose(W_vk[-1], np.pi) and abs(W[-1]) < 1e-7


# ---------------------------------------------------------------------------
# Section 3.3: slit annular plate under a lifting line force
# ---------------------------------------------------------------------------
def slit_annular_plate(nr, nt, element='Quad4R', kinematics='corotational'):
    r"""Slit annular plate of Fig. 4a of Sze, ``nr`` elements along the
    radius and ``nt`` around, Quad4R or two Tria3R per quadrilateral

    The slit is at ``theta = 0``: the edge ``theta = 0`` is clamped, the
    edge ``theta = 2*pi`` carries the line force ``P_max`` per unit length
    along +Z, lumped as the consistent nodal forces of a uniform line load
    on linear edges. A and B are the inner and outer corners of the loaded
    edge.

    Returns the :class:`CorotationalModel`, the mask of the free
    degrees-of-freedom and the nodes A and B.
    """
    Ri, Ro, h, E, nu, Pmax = (SLIT_PLATE[k]
                              for k in ('Ri', 'Ro', 'h', 'E', 'nu', 'Pmax'))
    prop = isotropic_plate(thickness=h, E=E, nu=nu)
    rs, ts = np.meshgrid(np.linspace(Ri, Ro, nr + 1),
                         np.linspace(0, 2*np.pi, nt + 1), indexing='ij')
    x = np.vstack(((rs*np.cos(ts)).ravel(), (rs*np.sin(ts)).ravel(),
                   0*rs.ravel())).T.ravel()
    nids = np.arange((nr + 1)*(nt + 1)).reshape(nr + 1, nt + 1)
    model = CorotationalModel(x)
    add_shells(model, zip(nids[:-1, :-1].ravel(), nids[1:, :-1].ravel(),
                          nids[1:, 1:].ravel(), nids[:-1, 1:].ravel()),
               prop, element, kinematics)
    bu = np.ones(model.N, dtype=bool)
    for n in nids[:, 0]:
        bu[DOF*n:DOF*(n + 1)] = False
    le = (Ro - Ri)/nr
    for i, n in enumerate(nids[:, -1]):
        model.fext[DOF*n + 2] = Pmax*le*(0.5 if i in (0, nr) else 1.)
    return model, bu, nids[0, -1], nids[-1, -1]


@pytest.mark.slow
def test_slit_annular_plate():
    r"""Slit annular plate of Sze, Section 3.3, Fig. 4a, lifted by the line
    force, with 6x30 Quad4R elements and Newton-Raphson in 40 increments of
    ``0.025*P_max``, against Table 4a (10x80 S4R) at its 23 load levels

    The plate twists into a helix with large rotations, Fig. 4c of Sze.
    Measured at ``P_max`` (Table 4a: ``W_A = 13.891``, ``W_B =
    17.528``), Newton-Raphson, with ``OMP_NUM_THREADS=1`` and the compiled
    :mod:`pyfe3d.corotational` (times of 2026-10-31):

    ============  ================  ================  =====
    mesh          W_A               W_B               time
    ============  ================  ================  =====
    6x30 Quad4R   13.294 (-4.29 %)  16.941 (-3.35 %)  7.5 s
    8x40          13.523 (-2.65 %)  17.166 (-2.06 %)   11 s
    12x60         13.693 (-1.43 %)  17.333 (-1.11 %)   32 s
    6x30 Tria3R   12.418 (-10.6 %)  16.029 (-8.55 %)  8.5 s
    12x60         13.488 (-2.90 %)  17.126 (-2.29 %)   37 s
    ============  ================  ================  =====

    (the meshes after 6x30 Quad4R with increments of 0.05 and cut-backs;
    Tria3R: 2 x nr x nt elements). The Quad4R errors decrease with an
    observed order of 1.9, and the Richardson extrapolation of the three
    Quad4R meshes is 0.36 % (W_A) and 0.30 % (W_B) below Table 4a. Sze
    report 6x30 S4R within 0.1 % of 10x80; Quad4R and Tria3R are stiffer
    on coarse meshes. The largest error of 6x30 Quad4R over the 23 levels
    is that at ``P_max``; at ``0.025*P_max`` it is -1.2 % and -1.0 %.

    Following the policy of the module documentation, this test compares
    the 6x30 path with the values pre-determined by this same test at
    ``P_max`` and with Table 4a with grade B.
    """
    model, bu, A, B = slit_annular_plate(6, 30)
    m, cs, t = analysis(model, bu, 0.025)
    print('6x30 Quad4R: %.1f s, %d increments' % (t, m.size))
    WA = np.array([u[DOF*A + 2] for u in cs])
    WB = np.array([u[DOF*B + 2] for u in cs])
    idx = at_levels(m, TABLE_4A[:, 0])
    for (Pr, WAr, WBr), k in zip(TABLE_4A, idx):
        print('P/Pmax %.3f  W_A %.4f (Sze %.3f, %+.2f %%)  W_B %.4f (Sze '
              '%.3f, %+.2f %%)' % (Pr, WA[k], WAr, 100*(WA[k]/WAr - 1),
                                   WB[k], WBr, 100*(WB[k]/WBr - 1)))
    assert np.isclose(m[-1], 1.)
    # NOTE values pre-determined with the 6x30 mesh, see the docstring
    assert np.isclose(WA[-1], 13.2944, rtol=1e-4)
    assert np.isclose(WB[-1], 16.9411, rtol=1e-4)
    # NOTE grade B against Table 4a at every level
    np.testing.assert_allclose(WA[idx], TABLE_4A[:, 1], rtol=0.10)
    np.testing.assert_allclose(WB[idx], TABLE_4A[:, 2], rtol=0.10)


@pytest.mark.slow
def test_slit_annular_plate_corotational_von_karman():
    r"""The slit annular plate of :func:`test_slit_annular_plate`, 6x30
    Quad4R, with ``kinematics='corotational_von_karman'``

    The von Kármán terms of Quad4R vanish in its CR frame, see
    :class:`pyfe3d.corotational.CorotationalShell`, so the path is that of
    ``'corotational'``. Measured at ``P_max``: ``W_A = 13.294417``, ``W_B
    = 16.941076``, as with ``'corotational'``, in 15 s (7.4 s with
    ``'corotational'``).

    Following the policy of the module documentation, the values of
    ``'corotational'`` pre-determined in :func:`test_slit_annular_plate`
    are used, with ``rtol = 1e-4``, and grade B against Table 4a.
    """
    model, bu, A, B = slit_annular_plate(
        6, 30, kinematics='corotational_von_karman')
    assert all(w.local == 'von_karman' for w in model.corotational)
    m, cs, t = analysis(model, bu, 0.025)
    print('6x30 Quad4R corotational_von_karman: %.1f s, %d increments'
          % (t, m.size))
    WA = np.array([u[DOF*A + 2] for u in cs])
    WB = np.array([u[DOF*B + 2] for u in cs])
    idx = at_levels(m, TABLE_4A[:, 0])
    for (Pr, WAr, WBr), k in zip(TABLE_4A, idx):
        print('P/Pmax %.3f  W_A %.4f (Sze %.3f)  W_B %.4f (Sze %.3f)'
              % (Pr, WA[k], WAr, WB[k], WBr))
    assert np.isclose(m[-1], 1.)
    assert np.isclose(WA[-1], 13.2944, rtol=1e-4)
    assert np.isclose(WB[-1], 16.9411, rtol=1e-4)
    np.testing.assert_allclose(WA[idx], TABLE_4A[:, 1], rtol=0.10)
    np.testing.assert_allclose(WB[idx], TABLE_4A[:, 2], rtol=0.10)

# ---------------------------------------------------------------------------
# Section 3.4: hemispherical shell under alternating radial forces
# ---------------------------------------------------------------------------
def hemisphere(n, element='Quad4R'):
    r"""Quarter of the hemispherical shell of Fig. 5a of Sze, ``0 <= phi <=
    90`` degrees in azimuth, from the equator to the hole at 18 degrees from
    the pole, meshed with ``n x n`` Quad4R or ``2 x n x n`` Tria3R

    The pole is along +Z. Point A, ``(0, R, 0)``, is pulled outwards (+Y)
    and point B, ``(R, 0, 0)``, pushed inwards (-X), each by half of
    ``P_max`` on the quarter. The symmetry planes ``Y = 0`` (``u_Y``,
    `\psi_X`, `\psi_Z` restrained) and ``X = 0`` (``u_X``, `\psi_Y`,
    `\psi_Z`), exact for rotation vectors; ``u_Z`` restrained at one node of
    the hole edge to remove the rigid translation, without reaction.
    The quadrilaterals of the grid in colatitude and azimuth are isosceles
    trapezoids, flat.

    Returns the :class:`CorotationalModel`, the mask of the free
    degrees-of-freedom and the nodes A and B.
    """
    R, h, E, nu, hole, Pmax = (HEMISPHERE[k] for k in ('R', 'h', 'E', 'nu',
                                                       'hole', 'Pmax'))
    prop = isotropic_plate(thickness=h, E=E, nu=nu)
    th, ph = np.meshgrid(np.linspace(np.pi/2, np.deg2rad(hole), n + 1),
                         np.linspace(0, np.pi/2, n + 1), indexing='ij')
    x = np.vstack(((R*np.sin(th)*np.cos(ph)).ravel(),
                   (R*np.sin(th)*np.sin(ph)).ravel(),
                   (R*np.cos(th)).ravel())).T.ravel()
    nids = np.arange((n + 1)**2).reshape(n + 1, n + 1)
    model = CorotationalModel(x)
    # NOTE outward normals
    add_shells(model, zip(nids[:-1, :-1].ravel(), nids[:-1, 1:].ravel(),
                          nids[1:, 1:].ravel(), nids[1:, :-1].ravel()),
               prop, element)
    bu = np.ones(model.N, dtype=bool)
    for nid in nids[:, 0]:
        bu[DOF*nid + 1] = bu[DOF*nid + 3] = bu[DOF*nid + 5] = False
    for nid in nids[:, -1]:
        bu[DOF*nid + 0] = bu[DOF*nid + 4] = bu[DOF*nid + 5] = False
    bu[DOF*nids[-1, n//2] + 2] = False
    A, B = nids[0, -1], nids[0, 0]
    model.fext[DOF*A + 1] = Pmax/2
    model.fext[DOF*B + 0] = -Pmax/2
    return model, bu, A, B


@pytest.mark.slow
def test_hemisphere():
    r"""Hemispherical shell of Sze, Section 3.4, Fig. 5a, with 16x16 Quad4R
    on the quarter, the mesh of Table 5a (S4R), and Newton-Raphson in 20
    increments of ``0.05*P_max``, against Table 5a at its 20 load levels

    Measured at ``P_max`` (Table 5a: ``V_A = 4.067``, ``-U_B = 8.178``),
    Newton-Raphson with increments of 0.05 and no cut-back, with
    ``OMP_NUM_THREADS=1``, and the linear radial deflection at A under
    forces of 2 on the full shell, 1 on the quarter, against that with
    64x64, 0.09348:

    =====  ===============  ================  ======  =====
    mesh   V_A              -U_B              linear  time
    =====  ===============  ================  ======  =====
    12x12  3.7243 (-8.4 %)  7.1621 (-12.4 %)  -9.3 %  2.5 s
    16x16  3.9107 (-3.8 %)  7.6877 (-6.0 %)   -3.5 %  6.8 s
    24x24  4.0269 (-1.0 %)  8.0120 (-2.0 %)   -0.9 %   22 s
    32x32  4.0536 (-0.3 %)  8.0872 (-1.1 %)   -0.4 %   50 s
    =====  ===============  ================  ======  =====

    The non-linear paths converge to Table 5a, at the rate of the linear
    solution, i.e. the error comes from the flat Quad4R on the doubly
    curved shell, not from the co-rotational kinematics. Sze report 12x12
    S4R as reasonably accurate. Tria3R is stiffer still: linear -14 % with
    16x16 and -1.3 % with 32x32.

    Following the policy of the module documentation, this test compares
    the 16x16 path with the values pre-determined by this same test at
    ``P_max`` and with Table 5a with grade B.
    """
    model, bu, A, B = hemisphere(16)
    m, cs, t = analysis(model, bu, 0.05)
    print('16x16 Quad4R: %.1f s, %d increments' % (t, m.size))
    VA = np.array([u[DOF*A + 1] for u in cs])
    UB = -np.array([u[DOF*B + 0] for u in cs])
    idx = at_levels(m, TABLE_5A[:, 0])
    for (Pr, VAr, UBr), k in zip(TABLE_5A, idx):
        print('P/Pmax %.2f  V_A %.4f (Sze %.3f, %+.2f %%)  -U_B %.4f (Sze '
              '%.3f, %+.2f %%)' % (Pr, VA[k], VAr, 100*(VA[k]/VAr - 1),
                                   UB[k], UBr, 100*(UB[k]/UBr - 1)))
    assert np.isclose(m[-1], 1.)
    # NOTE values pre-determined with the 16x16 mesh, see the docstring
    assert np.isclose(VA[-1], 3.9107, rtol=1e-4)
    assert np.isclose(UB[-1], 7.6877, rtol=1e-4)
    # NOTE grade B against Table 5a at every level
    np.testing.assert_allclose(VA[idx], TABLE_5A[:, 1], rtol=0.10)
    np.testing.assert_allclose(UB[idx], TABLE_5A[:, 2], rtol=0.10)


# ---------------------------------------------------------------------------
# Section 3.8: hinged cylindrical roof
# ---------------------------------------------------------------------------
def run_corotational_roof(n=8, h=12.7, load_factor=1.2, inc=0.05,
                          kinematics='corotational'):
    r"""The quarter of the hinged roof of :func:`hinged_roof`, ``n x n``
    Quad4R with their default stabilisation, assembled with
    ``kinematics``, up to ``load_factor`` times ``P_max``
    with Crisfield's arc-length method and increments ``inc``, as
    :func:`run_hinged_roof` for the von Kármán path

    Returns ``P/P_max`` and ``-W_C`` of the unloaded state and of the
    converged increments, and the time in seconds.
    """
    vk, bu, dofC = hinged_roof(n=n, h=h)
    model = CorotationalModel(vk.x, vk.N)
    for elem in vk.elements:
        model.add(elem, vk.prop, kinematics=kinematics, **vk.elem_kwargs)
    model.fext[:] = load_factor*vk.fext
    m, cs, t = analysis(model, bu, inc, NL_method='arc_length_crisfield')
    P = np.concatenate(([0.], load_factor*m))
    W = np.concatenate(([0.], -np.array([u[dofC] for u in cs])))
    return P, W, t


@pytest.mark.slow
def test_hinged_roof_thick_corotational():
    r"""Snap-through of the 12.7 thick roof of Sze, Section 3.8, Fig. 9a,
    with co-rotational 8x8 Quad4R and Crisfield's arc-length method,
    against Table 9a (16x16 S4R) and against the von Kármán path of
    :func:`run_hinged_roof`, same mesh and increments

    The rotations stay small, and the two paths agree up to the upper limit
    point, ``-W_C = 11``, within 0.001 P_max; they then part, by up to
    0.006 P_max in the snap-through and 0.020 P_max at the end of Table 9a,
    ``-W_C = 29.4``, where the deflection is about two thicknesses.
    Measured, the limit loads and the load at the last point of Table 9a
    (0.7421, 0.1729 and 1.0835):

    ============  ======================  ======================  =====
    mesh          co-rotational           von Kármán              time
    ============  ======================  ======================  =====
    8x8 Quad4R    0.74211 0.17306 1.1003  0.74293 0.17183 1.0800   2.7 s
    16x16         0.74159 0.16947 1.0878  0.74206 0.16831 1.0679    13 s
    ============  ======================  ======================  =====

    With the mesh of Table 9a the co-rotational path is 0.4 % above it at
    the last point, the von Kármán one 1.4 % below: S4R, whose kinematics
    are those of finite rotations, follows the co-rotational path.

    Following the policy of the module documentation, this test uses the
    8x8 mesh, its two limit loads pre-determined with ``rtol = 1e-4``, and
    the grades A (upper) and B (lower) against Table 9a, as
    ``test_hinged_roof_thick`` of ``tests/test_nonlinear_benchmarks_sze2004.py``.
    """
    P, W, t = run_corotational_roof(n=8)
    Pv, Wv = run_hinged_roof(n=8)
    print('8x8 Quad4R co-rotational: %.1f s, %d increments' % (t, P.size - 1))
    P_calc = compare_with_table_9a(P, W)
    Pv_calc = compare_with_table_9a(Pv, Wv)
    for (Pr, Wr), Pc, Pvc in zip(TABLE_9A, P_calc, Pv_calc):
        print('-W_C %7.3f  P/Pmax Sze %.4f  co-rotational %.4f  von Kármán '
              '%.4f' % (Wr, Pr, Pc, Pvc))
    (Pmax, _), (Pmin, _) = limit_points(P, W, 15.)
    (Pmax_ref, _), (Pmin_ref, _) = limit_points(TABLE_9A[:, 0],
                                                TABLE_9A[:, 1], 15.)
    print('limit loads %.5f, %.5f; Table 9a %.4f, %.4f'
          % (Pmax, Pmin, Pmax_ref, Pmin_ref))
    # NOTE values pre-determined with the 8x8 mesh, see the docstring
    assert np.isclose(Pmax, 0.74211, rtol=1e-4)
    assert np.isclose(Pmin, 0.17306, rtol=1e-4)
    # NOTE grades A and B against Table 9a
    assert abs(Pmax/Pmax_ref - 1) <= 0.02
    assert abs(Pmin/Pmin_ref - 1) <= 0.10
    # NOTE the two kinematics agree up to the upper limit point
    before = TABLE_9A[:, 1] <= 11.3
    assert np.abs(P_calc - Pv_calc)[before].max() <= 0.001
    assert np.abs(P_calc - Pv_calc).max() <= 0.03


@pytest.mark.slow
def test_hinged_roof_thin_corotational():
    r"""Snap-through and snap-back of the 6.35 thick roof of Sze, Section
    3.8, with co-rotational 8x8 Quad4R and Crisfield's arc-length method in
    increments of 0.075 up to ``1.1*P_max``, against Table 9d (24x24 S4R)
    and against the von Kármán path of :func:`thin_roof_path`, same mesh
    and increments

    Measured, the upper and lower limit loads (Table 9d: 0.1953 and
    -0.1288):

    =====  ==================  ==================  =====
    mesh   co-rotational       von Kármán          time
    =====  ==================  ==================  =====
    8x8    0.19821 -0.12606    0.19878 -0.12908    1.5 s
    16x16  0.19537 -0.12828    0.19552 -0.12930    6.9 s
    =====  ==================  ==================  =====

    The co-rotational limit loads converge to Table 9d as the von Kármán
    ones, see ``test_hinged_roof_thin_snap_back`` of
    ``tests/test_nonlinear_benchmarks_sze2004.py``; the sampling with the
    increments 0.075 changes the lower one by up to 2 %, see that test.

    Following the policy of the module documentation, this test uses the
    8x8 mesh, its limit loads pre-determined with ``rtol = 1e-4``, grade A
    against Table 9d for the upper one and grade B for the lower one.
    """
    P, W, t = run_corotational_roof(n=8, h=6.35, load_factor=1.1, inc=0.075)
    Pv, Wv = thin_roof_path()
    print('8x8 Quad4R co-rotational: %.1f s, %d increments' % (t, P.size - 1))
    assert np.isclose(P[-1], 1.1)
    assert np.any(np.diff(W) < 0)
    (Pmax, _), _ = limit_points(P, W, 13.5)
    (Pvmax, _), _ = limit_points(Pv, Wv, 13.5)
    (Pmax_ref, _), _ = limit_points(TABLE_9D[:, 0], TABLE_9D[:, 1], 13.5)
    Pmin, Pvmin, Pmin_ref = P.min(), Pv.min(), TABLE_9D[:, 0].min()
    print('limit loads: co-rotational %.5f, %.5f; von Kármán %.5f, %.5f; '
          'Table 9d %.4f, %.4f' % (Pmax, Pmin, Pvmax, Pvmin, Pmax_ref,
                                   Pmin_ref))
    # NOTE values pre-determined with the 8x8 mesh, see the docstring
    assert np.isclose(Pmax, 0.19821, rtol=1e-4)
    assert np.isclose(Pmin, -0.12606, rtol=1e-4)
    assert abs(Pmax/Pmax_ref - 1) <= 0.02
    assert abs(Pmin/Pmin_ref - 1) <= 0.10
    assert abs(Pmax - Pvmax) <= 0.002


@pytest.mark.slow
@pytest.mark.parametrize('h', [12.7, 6.35])
def test_hinged_roof_corotational_von_karman(h):
    r"""The hinged roofs of :func:`test_hinged_roof_thick_corotational`
    (``h = 12.7``, up to ``1.2*P_max`` in increments of 0.05) and
    :func:`test_hinged_roof_thin_corotational` (``h = 6.35``, ``1.1*P_max``,
    0.075), 8x8 Quad4R, with ``kinematics='corotational_von_karman'``,
    against ``'corotational'``

    The von Kármán terms of Quad4R vanish in its CR frame, see
    :class:`pyfe3d.corotational.CorotationalShell`, so the arc-length path
    is that of ``'corotational'``: measured, the same number of increments
    and differences of at most 6e-13 in ``P/P_max`` and 1e-11 in ``-W_C``;
    limit loads 0.74211, 0.17306 (``h = 12.7``) and 0.19821, -0.12606
    (``h = 6.35``), against 0.74293, 0.17183 and 0.19878, -0.12908 with
    ``'von_karman'`` and 0.7421, 0.1729 (Table 9a) and 0.1953, -0.1288
    (Table 9d) of Sze. Times 4.6 s and 2.7 s, and 2.4 s and 1.5 s with
    ``'corotational'``.
    """
    if h == 12.7:
        kw = dict(n=8, h=h)
    else:
        kw = dict(n=8, h=h, load_factor=1.1, inc=0.075)
    P, W, t = run_corotational_roof(kinematics='corotational_von_karman',
                                    **kw)
    Pc, Wc, tc = run_corotational_roof(**kw)
    print('h %.2f, 8x8 Quad4R: corotational_von_karman %.1f s, %d '
          'increments; corotational %.1f s, %d increments'
          % (h, t, P.size - 1, tc, Pc.size - 1))
    assert P.size == Pc.size
    print('max differences: P/Pmax %.1e, -W_C %.1e'
          % (np.abs(P - Pc).max(), np.abs(W - Wc).max()))
    assert np.abs(P - Pc).max() <= 1e-9
    assert np.abs(W - Wc).max() <= 1e-8*np.abs(Wc).max()
    if h == 12.7:
        (Pmax, _), (Pmin, _) = limit_points(P, W, 15.)
        assert np.isclose(Pmax, 0.74211, rtol=1e-4)
        assert np.isclose(Pmin, 0.17306, rtol=1e-4)
    else:
        (Pmax, _), _ = limit_points(P, W, 13.5)
        Pmin = P.min()
        assert np.isclose(Pmax, 0.19821, rtol=1e-4)
        assert np.isclose(Pmin, -0.12606, rtol=1e-4)
    print('limit loads %.5f, %.5f' % (Pmax, Pmin))
