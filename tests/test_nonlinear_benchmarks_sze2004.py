r"""Geometrically non-linear benchmarks of Sze, Liu and Lo (2004)

Sze, K. Y., Liu, X. H. and Lo, S. H., Popular benchmark problems for
geometric nonlinear analysis of shells, Finite Elements in Analysis and
Design, 40(11), 1551-1569, 2004, https://doi.org/10.1016/j.finel.2003.11.001:

1. Section 3.2, Fig. 3 and Table 3: cantilever under an end moment, ``L =
   12``, ``b = 1``, ``h = 0.1``, ``E = 1.2e6``, ``nu = 0``, ``M_o = E*I/L =
   25/3`` and ``M_max = 2*pi*M_o`` (Fig. 3a), at which the cantilever is bent
   into a full circle. Table 3a lists the exact tip deflections, ``U/L =
   M_o/M*sin(M/M_o) - 1`` and ``W/L = M_o/M*(1 - cos(M/M_o))``, and those of
   16x1 S4R elements.

2. Section 3.8, Fig. 9 and Tables 9a and 9d: hinged cylindrical roof under a
   central point load, ``R = 2540``, ``L = 254``, ``theta = 0.1`` rad, ``h =
   12.7`` or ``6.35``, ``E = 3102.75``, ``nu = 0.3``, ``P_max = 3000``; one
   quarter modelled with 16x16 (``h = 12.7``) or 24x24 (``h = 6.35``) S4R
   elements and Riks' method. The hinged edges have all the translations
   restrained, the curved edges are free.

The elements of pyfe3d use the von Karman strains, valid for moderate
rotations. In the first problem the von Karman cantilever has no axial force,
its deflection is that of the linear theory, ``W = M*L**2/(2*E*I)``, and its
axial strain ``u_x + w_x**2/2 = 0`` gives ``U = -(M/(E*I))**2*L**3/6``: it
agrees with the exact solution only while the tip rotation ``M*L/(E*I)`` is
small and never rolls up into a circle. In the second problem the rotations
stay small and the snap-through is traced with the arc-length method.

The roof is modelled with Quad4R and its default stabilisation, see
:mod:`pyfe3d.quad4r`: since 0.12.0 the bending and the transverse shear have
the physical stabilisation of Belytschko and Leviathan (1994), which passes
the constant-twist patch test, see ``test_constant_twist_patch_test`` in
``tests/test_quad4r_hourglass_control.py``. Up to 0.11 the hourglass mode
of `w` was controlled with Brockman (1987), ``0.10*E*h**3/(1 + 1/A)``,
which added 73 to 78 % to the energy of a uniform twist in this model,
whatever the mesh; the lower limit load of the thinner roof then
converged 11 % below Table 9d. The sensitivity of the roof to the
stabilisation, scaled with the ``hgfactor_physical`` of Quad4R, is measured
by :func:`test_hinged_roof_hourglass_sensitivity`, against Tria3DSG, which
has no hourglass control, and against the shallow-shell Kirchhoff solution
of :func:`ritz_roof_WC`.
"""
import functools
import sys
sys.path.append('..')

import numpy as np
import pytest
from scipy.sparse.linalg import spsolve

from pyfe3d.beamprop import BeamProp
from pyfe3d.shellprop_utils import isotropic_plate
from pyfe3d import (BeamC, BeamCData, BeamCProbe, BeamLR, BeamLRData,
                    BeamLRProbe, Quad4R, Quad4RData, Quad4RProbe, Tria3DSG,
                    Tria3DSGData, Tria3DSGProbe, DOF)

from ritz import vecf, vecf_xi, vecf_xixi
from test_solver_nonlinear_static import Model

# NOTE Fig. 3a of Sze et al. (2004)
CANTILEVER = dict(L=12., b=1., h=0.1, E=1.2e6)
# NOTE Table 3a of Sze et al. (2004): M/M_max, exact -U_tip and W_tip
TABLE_3A_EXACT = np.array([
    [0.05, 0.196, 1.870], [0.10, 0.774, 3.648], [0.15, 1.699, 5.248],
    [0.20, 2.918, 6.598], [0.25, 4.361, 7.639], [0.30, 5.945, 8.333],
    [0.35, 7.585, 8.664], [0.40, 9.194, 8.637], [0.45, 10.688, 8.281],
    [0.50, 12.000, 7.639], [0.55, 13.073, 6.775], [0.60, 13.871, 5.758],
    [0.65, 14.377, 4.665], [0.70, 14.595, 3.571], [0.75, 14.546, 2.546],
    [0.80, 14.270, 1.650], [0.85, 13.818, 0.926], [0.90, 13.247, 0.405],
    [0.95, 12.621, 0.098], [1.00, 12.000, 0.000]])

# NOTE Fig. 9a of Sze et al. (2004)
ROOF = dict(R=2540., L=254., theta=0.1, E=3102.75, nu=0.3, Pmax=3000.)
# NOTE Table 9a of Sze et al. (2004), 12.7 thick isotropic roof, 16x16 S4R
#      elements on one quarter: P/P_max and -W_C, in the order of the path
TABLE_9A = np.array([
    [0.0877, 0.693], [0.1980, 1.638], [0.3473, 3.087], [0.4686, 4.477],
    [0.5647, 5.802], [0.6381, 7.057], [0.6908, 8.237], [0.7246, 9.339],
    [0.7412, 10.358], [0.7421, 11.293], [0.7286, 12.141], [0.7023, 12.903],
    [0.6649, 13.583], [0.6182, 14.188], [0.5643, 14.728], [0.5055, 15.217],
    [0.4442, 15.676], [0.3830, 16.125], [0.3245, 16.590], [0.2717, 17.094],
    [0.2272, 17.657], [0.1940, 18.299], [0.1750, 19.028], [0.1729, 19.852],
    [0.1905, 20.771], [0.2303, 21.780], [0.2950, 22.875], [0.3871, 24.049],
    [0.4443, 24.663], [0.5093, 25.293], [0.5826, 25.940], [0.6644, 26.601],
    [0.7551, 27.276], [0.8549, 27.964], [0.9643, 28.663], [1.0835, 29.374]])
# NOTE Table 9d of Sze et al. (2004), 6.35 thick isotropic roof, 24x24 S4R
#      elements on one quarter: P/P_max and -W_C, in the order of the path
TABLE_9D = np.array([
    [0.0517, 1.846], [0.1182, 5.271], [0.1583, 8.257], [0.1837, 10.799],
    [0.1914, 11.904], [0.1953, 12.892], [0.1950, 13.752], [0.1901, 14.472],
    [0.1806, 15.050], [0.1671, 15.501], [0.1323, 16.145], [0.0923, 16.602],
    [0.0504, 16.915], [0.0083, 17.008], [-0.0312, 16.697], [-0.0622, 15.780],
    [-0.0739, 15.206], [-0.0861, 14.767], [-0.1001, 14.520],
    [-0.1142, 14.451], [-0.1247, 14.862], [-0.1288, 15.778],
    [-0.1271, 16.961], [-0.1196, 18.320], [-0.1055, 19.817],
    [-0.0825, 21.420], [-0.0484, 23.100], [-0.0006, 24.824],
    [0.0626, 26.565], [0.1427, 28.302], [0.2403, 30.023], [0.3559, 31.720],
    [0.4898, 33.388], [0.6417, 35.024], [0.8114, 36.626], [1.0313, 38.450]])


def elastica_tip(m):
    r"""Exact tip deflections ``-U/L, W/L`` of the cantilever under the end
    moment ``M = m*M_max``, ``M_max = 2*pi*E*I/L``, Section 3.2 of Sze et
    al. (2004)"""
    phi = 2*np.pi*np.asarray(m, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        U = np.where(phi == 0, 0., np.sin(phi)/phi - 1)
        W = np.where(phi == 0, 0., (1 - np.cos(phi))/phi)
    return -U, W


def von_karman_tip(m):
    r"""Tip deflections ``-U/L, W/L`` of the cantilever under the end moment
    ``M = m*M_max`` with the von Karman axial strain: ``W/L = phi/2`` and
    ``U/L = -phi**2/6``, ``phi = M*L/(E*I)``"""
    phi = 2*np.pi*np.asarray(m, dtype=float)
    return phi**2/6, phi/2


def cantilever_end_moment(name, num_elements=16):
    r"""Cantilever of Fig. 3a of Sze et al. (2004) under the end moment
    ``M_max``, with BeamC or BeamLR elements or with a strip of
    ``num_elements x 1`` Quad4R elements

    Returns the :class:`Model`, the mask of the free degrees-of-freedom, the
    function that gives the tip deflections ``-U`` and ``W`` from the
    displacements, and ``M_max``.
    """
    L, b, h, E = (CANTILEVER[k] for k in ('L', 'b', 'h', 'E'))
    EI = E*b*h**3/12
    Mmax = 2*np.pi*EI/L
    if name in ('BeamC', 'BeamLR'):
        cls, datacls, probecls = {
            'BeamC': (BeamC, BeamCData, BeamCProbe),
            'BeamLR': (BeamLR, BeamLRData, BeamLRProbe)}[name]
        prop = BeamProp()
        prop.A = b*h
        prop.E = E
        prop.G = E/2 # nu = 0
        prop.Iyy = b*h**3/12 # bending in the xz plane
        prop.Izz = h*b**3/12
        prop.J = b*h**3/3
        num_nodes = num_elements + 1
        xs = np.linspace(0, L, num_nodes)
        x = np.vstack((xs, 0*xs, 0*xs)).T.flatten()
        data = datacls()
        probe = probecls()
        elements = []
        for k in range(num_elements):
            elem = cls(probe)
            elem.n1, elem.n2 = k, k + 1
            elem.c1, elem.c2 = DOF*k, DOF*(k + 1)
            elem.init_k_KC0 = k*data.KC0_SPARSE_SIZE
            elem.init_k_KCNL = k*data.KCNL_SPARSE_SIZE
            elem.init_k_KG = k*data.KG_SPARSE_SIZE
            elem.update_rotation_matrix(0., 1., 0., x)
            elem.update_probe_xe(x)
            elements.append(elem)
        root = [0]
        tip = [num_nodes - 1]
    elif name == 'Quad4R':
        # NOTE nu = 0, the bending stiffness per unit width E*h**3/12 times
        #      the width b = 1 is the E*I of the beam
        prop = isotropic_plate(thickness=h, E=E, nu=0.)
        nx, ny = num_elements + 1, 2
        xs, ys = np.meshgrid(np.linspace(0, L, nx), np.linspace(0, b, ny),
                             indexing='ij')
        x = np.vstack((xs.flatten(), ys.flatten(),
                       0*xs.flatten())).T.flatten()
        nids = np.arange(nx*ny).reshape(nx, ny)
        data = Quad4RData()
        probe = Quad4RProbe()
        elements = []
        for k, (n1, n2, n3, n4) in enumerate(zip(
                nids[:-1, :-1].flatten(), nids[1:, :-1].flatten(),
                nids[1:, 1:].flatten(), nids[:-1, 1:].flatten())):
            elem = Quad4R(probe)
            elem.n1, elem.n2, elem.n3, elem.n4 = n1, n2, n3, n4
            elem.c1, elem.c2, elem.c3, elem.c4 = (DOF*n1, DOF*n2, DOF*n3,
                                                  DOF*n4)
            elem.init_k_KC0 = k*data.KC0_SPARSE_SIZE
            elem.init_k_KCNL = k*data.KCNL_SPARSE_SIZE
            elem.init_k_KG = k*data.KG_SPARSE_SIZE
            elem.update_rotation_matrix(x)
            elem.update_probe_xe(x)
            elements.append(elem)
        root = list(nids[0, :])
        tip = list(nids[-1, :])
    else:
        raise ValueError(name)
    N = x.shape[0]//3*DOF
    bk = np.zeros(N, dtype=bool)
    for n in root:
        bk[DOF*n:DOF*(n + 1)] = True
    bu = ~bk
    # NOTE the moment about -y bends the cantilever towards +z
    fext = np.zeros(N)
    for n in tip:
        fext[DOF*n + 4] = -Mmax/len(tip)
    model = Model(elements, data, prop, x, N, fext)

    def tip_deflections(u):
        U = np.mean([u[DOF*n + 0] for n in tip])
        W = np.mean([u[DOF*n + 2] for n in tip])
        return -U, W

    return model, bu, tip_deflections, Mmax


def run_cantilever_end_moment(name, num_elements=16, inc=0.05, m_max=1.):
    r"""Newton-Raphson analysis up to ``m_max*M_max`` in equal increments
    of ``inc*m_max*M_max``

    Returns ``M/M_max`` and the tip deflections ``-U/L`` and ``W/L`` of the
    converged increments.
    """
    model, bu, tip_deflections, Mmax = cantilever_end_moment(name,
                                                             num_elements)
    model.fext *= m_max
    an = model.analysis(bu, NL_method='NR', initialInc=inc, maxInc=inc)
    increments, cs = an.static(NLgeom=True, silent=True)
    L = CANTILEVER['L']
    tips = np.array([tip_deflections(u) for u in cs])/L
    return m_max*np.array(increments), tips[:, 0], tips[:, 1]


def test_elastica_table_3a():
    r"""The closed-form solution reproduces the exact values of Table 3a of
    Sze et al. (2004), published with three decimals"""
    L = CANTILEVER['L']
    U, W = elastica_tip(TABLE_3A_EXACT[:, 0])
    np.testing.assert_allclose(L*U, TABLE_3A_EXACT[:, 1], rtol=0,
                               atol=0.0005)
    np.testing.assert_allclose(L*W, TABLE_3A_EXACT[:, 2], rtol=0,
                               atol=0.0005)


def test_cantilever_end_moment_quad4r():
    r"""The 16x1 strip of Quad4R elements gives the von Karman solution up
    to ``M_max``, which agrees with the exact solution only for small tip
    rotations and does not roll up into a circle"""
    m, U, W = run_cantilever_end_moment('Quad4R')
    assert np.isclose(m[-1], 1.)
    U_vk, W_vk = von_karman_tip(m)
    U_ex, W_ex = elastica_tip(m)
    for mi, Ui, Wi, Uvk, Wvk, Uex, Wex in zip(m, U, W, U_vk, W_vk, U_ex,
                                              W_ex):
        print('M/Mmax %.3f  -U/L %.5f (von Karman %.5f, exact %.5f)'
              '  W/L %.5f (von Karman %.5f, exact %.5f)'
              % (mi, Ui, Uvk, Uex, Wi, Wvk, Wex))
    np.testing.assert_allclose(W, W_vk, rtol=1.e-6)
    # NOTE the axial displacement converges with the mesh, 0.1 % with 16x1
    np.testing.assert_allclose(U, U_vk, rtol=2.e-3)
    # NOTE tip rotation 2*pi*0.05 = 0.31 rad: 1 % from the exact solution
    first = np.isclose(m, 0.05)
    assert np.allclose(W[first], W_ex[first], rtol=0.01)
    assert np.allclose(U[first], U_ex[first], rtol=0.01)
    # NOTE at M_max the exact tip is back at the root, the von Karman tip
    #      is at W = pi*L
    assert np.isclose(W[-1], np.pi, rtol=1.e-6)
    assert W_ex[-1] < 1.e-12


@pytest.mark.slow
@pytest.mark.parametrize('name', ['BeamC'])
def test_cantilever_end_moment_beams_membrane_locking(name):
    r"""BeamC interpolates the axial displacement linearly and integrates
    the von Karman axial strain ``u_x + w_x**2/2`` at Gauss points along
    the element, where ``w_x**2`` is not constant, so that the
    axial strain cannot vanish and the cantilever is too stiff (membrane
    locking, Stolarski, H. and Belytschko, T., Membrane locking and reduced
    integration for curved elements, Journal of Applied Mechanics, 49(1),
    172-176, 1982, https://doi.org/10.1115/1.3161961)

    Under ``M = 0.05*M_max`` the von Karman curvature is uniform, ``kappa =
    M/(E*I)``, and the slope ``theta = kappa*x``. In an element of length
    ``l_e`` and mid-point slope ``theta_m`` the linear part of
    ``theta**2/2``, ``theta_m*kappa*s``, cannot be balanced by the constant
    ``u_x``; its energy, ``E*A*theta_m**2*kappa**2*l_e**3/24``, summed over
    the elements, ``sum(theta_m**2*l_e) ~ theta_tip**2*L/3``, and
    differentiated with respect to ``kappa``, gives ``M = E*I*kappa*(1 +
    e)``, ``e = 2/3*theta_tip**2*(l_e/h)**2``, ``theta_tip = M*L/(E*I)``,
    for a uniform curvature imposed on the mesh. The excess stiffness ``W_vk/W
    - 1`` therefore tends to zero as ``l_e**2``, i.e. as ``num_elements**-2``.

    Measured ``W/W_vk - 1`` (BeamLR up to pyfe3d 0.11 agreed within 0.1 %,
    see :func:`test_cantilever_end_moment_beamlr_no_membrane_locking`): -43.6 %,
    -26.9 %, -12.6 %, -4.33 %, -1.21 % and -0.313 % with 16, 32, 64, 128, 256
    and 512 elements; ``W_vk/W - 1`` = 0.772, 0.368, 0.144, 0.0452, 0.01227
    and 0.003138, of observed orders 1.07, 1.35, 1.67, 1.88 and 1.97, which
    tend to 2; the deviation from 2 decreases by a factor of 3.5 from the
    pair 128/256 to the pair 256/512, the tolerance 0.1 on the order of the
    last pair is below the deviation of the pair 128/256, 0.12. With 512
    elements the excess, 0.00314, is 0.87 times the estimate with a uniform
    curvature, 0.00361; the finite elements relax the curvature where the
    locking is largest, near the tip, and are less stiff than the estimate.

    With the 16 elements of Table 3a of Sze et al. (2004) the error is
    worse than grade C of MacNeal and Harder (1985) (``> 20 %``), and with
    512 elements of grade A (``<= 2 %``); MacNeal, R. H. and Harder, R. L., A
    proposed standard set of problems to test finite element accuracy,
    Finite Elements in Analysis and Design, 1(1), 3-20, 1985,
    https://doi.org/10.1016/0168-874X(85)90003-4, p. 14.
    """
    L, h = CANTILEVER['L'], CANTILEVER['h']
    nums = (16, 64, 256, 512)
    errors = []
    excess = []
    for num_elements in nums:
        m, U, W = run_cantilever_end_moment(name, num_elements, inc=0.25,
                                            m_max=0.05)
        U_vk, W_vk = von_karman_tip(m[-1])
        errors.append(abs(W[-1] - W_vk)/W_vk)
        excess.append(W_vk/W[-1] - 1)
        print('%s %3d elements, M/Mmax %.2f: W/L %.5f von Karman %.5f, '
              'error %.3f %%, W_vk/W - 1 = %.5f'
              % (name, num_elements, m[-1], W[-1], W_vk, 100*errors[-1],
                 excess[-1]))
    assert np.isclose(m[-1], 0.05)
    assert np.all(np.diff(errors) < 0)
    assert np.all(np.array(excess) > 0)
    assert errors[0] > 0.20
    assert errors[-1] <= 0.02
    order = np.log(excess[-2]/excess[-1])/np.log(nums[-1]/nums[-2])
    print('observed order %.3f' % order)
    assert abs(order - 2) < 0.1
    theta_tip = 2*np.pi*m[-1]
    estimate = 2/3*theta_tip**2*(L/nums[-1]/h)**2
    print('excess %.5f, uniform-curvature estimate %.5f'
          % (excess[-1], estimate))
    assert 0 < excess[-1] < estimate


@pytest.mark.parametrize('num_elements', [4, 16])
def test_cantilever_end_moment_beamlr_no_membrane_locking(num_elements):
    r"""BeamLR takes the slope ``w_x`` of the von Karman axial strain from
    its linear interpolation of ``w``, constant in the element as ``u_x``,
    so that the axial strain ``u_x + w_x**2/2`` can vanish in each element
    and there is no membrane locking.

    Under ``M = 0.05*M_max`` the von Karman cantilever has no axial force,
    the curvature ``kappa = M/(E*I)`` is uniform, the rotation linear and
    ``w = kappa*x**2/2`` quadratic. The linear rotations of BeamLR give the
    uniform curvature exactly, and the transverse shear strain, sampled at
    the element centre, vanishes for the nodal values of the quadratic
    ``w``: the nodal deflections are exact, ``W = W_vk``. The axial
    displacement follows from ``u_x = -w_x**2/2`` with the mean slope of
    each element, whose square misses the variance ``kappa**2*l_e**2/12``
    of the linear slope, so that ``U - U_vk = kappa**2*L*l_e**2/24`` and
    ``U/U_vk - 1 = -1/(4*num_elements**2)``, with ``U_vk =
    -kappa**2*L**3/6``.

    Measured with 4, 16 and 64 elements: ``W/W_vk - 1`` = 3.9e-12, -7.7e-11
    and -5.7e-10, at the level of the convergence criterion of the
    Newton-Raphson increments, ``relTOL = 1e-9``; ``U/U_vk - 1`` =
    -1.5625e-2, -9.766e-4 and -6.104e-5, the closed form above. Up to pyfe3d
    0.11 BeamLR took the slopes from the rotations, linear in the element,
    and locked as BeamC: ``W/W_vk - 1`` = -43.6 % with 16 elements.
    """
    m, U, W = run_cantilever_end_moment('BeamLR', num_elements, inc=0.25,
                                        m_max=0.05)
    U_vk, W_vk = von_karman_tip(m[-1])
    assert np.isclose(m[-1], 0.05)
    assert np.isclose(W[-1], W_vk, rtol=1e-8)
    assert np.isclose(U[-1]/U_vk - 1, -1/(4*num_elements**2), rtol=1e-4)


def hinged_roof(n=16, h=12.7, element='Quad4R', hgfactor_physical=1.,
                shear_factor=1.):
    r"""Quarter of the hinged cylindrical roof of Fig. 9a of Sze et al.
    (2004) meshed with ``n x n`` Quad4R elements, or ``2 x n x n`` Tria3DSG
    elements

    The axis of the cylinder is along X, the quarter covers ``0 <= X <= L``
    and ``0 <= theta <= 0.1``, the point C at ``X = 0, theta = 0``. The
    reference load is ``P_max`` on the whole roof, a quarter of it on the
    quarter. ``hgfactor_physical`` multiplies the stabilisation of the bending and
    of the transverse shear of Quad4R, ``1.`` being the default of Quad4R; ``shear_factor`` multiplies
    the transverse shear stiffnesses ``A44``, ``A45`` and ``A55``, e.g.
    ``100.`` to approach the Kirchhoff solution of :func:`ritz_roof_WC`.

    Returns the :class:`Model`, the mask of the free degrees-of-freedom and
    the index of the degree-of-freedom of the deflection ``W`` at C.
    """
    R, L, theta = ROOF['R'], ROOF['L'], ROOF['theta']
    prop = isotropic_plate(thickness=h, E=ROOF['E'], nu=ROOF['nu'])
    prop.A44 *= shear_factor
    prop.A45 *= shear_factor
    prop.A55 *= shear_factor
    Xs, thetas = np.meshgrid(np.linspace(0, L, n + 1),
                             np.linspace(0, theta, n + 1), indexing='ij')
    Xs = Xs.flatten()
    thetas = thetas.flatten()
    x = np.vstack((Xs, R*np.sin(thetas), R*np.cos(thetas))).T.flatten()
    nids = np.arange((n + 1)**2).reshape(n + 1, n + 1)
    quads = zip(nids[:-1, :-1].flatten(), nids[1:, :-1].flatten(),
                nids[1:, 1:].flatten(), nids[:-1, 1:].flatten())
    if element == 'Quad4R':
        cls, data, probe = Quad4R, Quad4RData(), Quad4RProbe()
        conn = list(quads)
        elem_kwargs = dict(hgfactor_physical=hgfactor_physical)
    elif element == 'Tria3DSG':
        cls, data, probe = Tria3DSG, Tria3DSGData(), Tria3DSGProbe()
        conn = []
        for n1, n2, n3, n4 in quads:
            conn += [(n1, n2, n3), (n1, n3, n4)]
        elem_kwargs = None
    else:
        raise ValueError(element)
    elements = []
    for k, nodes in enumerate(conn):
        elem = cls(probe)
        for i, nid in enumerate(nodes):
            setattr(elem, 'n%d' % (i + 1), nid)
            setattr(elem, 'c%d' % (i + 1), DOF*nid)
        elem.init_k_KC0 = k*data.KC0_SPARSE_SIZE
        elem.init_k_KCNL = k*data.KCNL_SPARSE_SIZE
        elem.init_k_KG = k*data.KG_SPARSE_SIZE
        elem.update_rotation_matrix(x)
        elem.update_probe_xe(x)
        elements.append(elem)
    N = DOF*(n + 1)**2
    bk = np.zeros(N, dtype=bool)
    # NOTE hinged straight edge, theta = 0.1: all translations restrained
    for nid in nids[:, -1]:
        bk[DOF*nid:DOF*nid + 3] = True
    # NOTE symmetry plane X = 0: u_X, r_Y and r_Z restrained
    for nid in nids[0, :]:
        bk[DOF*nid + 0] = bk[DOF*nid + 4] = bk[DOF*nid + 5] = True
    # NOTE symmetry plane Y = 0: u_Y, r_X and r_Z restrained
    for nid in nids[:, 0]:
        bk[DOF*nid + 1] = bk[DOF*nid + 3] = bk[DOF*nid + 5] = True
    bu = ~bk
    dofC = DOF*nids[0, 0] + 2
    fext = np.zeros(N)
    fext[dofC] = -ROOF['Pmax']/4
    model = Model(elements, data, prop, x, N, fext, elem_kwargs=elem_kwargs)
    return model, bu, dofC


def linear_roof_WC(n=16, h=12.7, element='Quad4R', hgfactor_physical=1.,
                   shear_factor=1.):
    r"""Linear deflection ``-W_C`` of :func:`hinged_roof` under ``P_max``"""
    model, bu, dofC = hinged_roof(n, h, element, hgfactor_physical, shear_factor)
    u = np.zeros(model.N)
    u[bu] = spsolve(model.KC0[bu, :][:, bu].tocsc(), model.fext[bu])
    return -u[dofC]


def ritz_roof_WC(m=20, h=12.7):
    r"""Linear deflection ``-W_C`` under ``P_max`` of the quarter of the roof
    of :func:`hinged_roof` with the shallow-shell (Donnell-Marguerre)
    Kirchhoff theory and a Ritz method

    The quarter is the rectangle ``0 <= x <= L``, ``0 <= y <= b`` of the
    developed surface, ``b = R*theta``, with the membrane strains ``u_x``,
    ``v_y + w/R`` and ``u_y + v_x``, the curvatures ``-w_xx``, ``-w_yy`` and
    ``-2*w_xy``, and ``w`` positive outwards. Each of ``u``, ``v`` and ``w``
    is approximated with the products of ``m`` functions of
    :func:`ritz.vecf` in ``x`` and in ``y``, without those that violate the
    boundary conditions: symmetry at ``x = 0`` (``u = w_x = 0``) and at ``y =
    0`` (``v = w_y = 0``), hinged edge at ``y = b`` (``u = v = w = 0``), free
    edge at ``x = L``. The integrals are evaluated with a Gauss-Legendre
    quadrature of 200 points in each direction, exact for the polynomial
    integrands. The solution has no transverse shear deformation and is
    independent of the finite elements of pyfe3d.
    """
    R, L, E, nu = ROOF['R'], ROOF['L'], ROOF['E'], ROOF['nu']
    b = R*ROOF['theta']
    xi, weights = np.polynomial.legendre.leggauss(200)

    def basis(flags, a):
        F = [vecf(m, xi, *flags).T, vecf_xi(m, xi, *flags).T*(2/a),
             vecf_xixi(m, xi, *flags).T*(2/a)**2]
        # NOTE the functions removed by the flags are identically zero
        keep = np.abs(F[0]).sum(axis=0) + np.abs(F[1]).sum(axis=0) > 1e-14
        return [Fi[:, keep] for Fi in F], weights*a/2, keep

    # NOTE flags of vecf: translation and rotation at xi = -1 (x = 0 or
    #      y = 0) and at xi = +1 (x = L or y = b), 1 free and 0 restrained
    flags = {'u': ((0, 1, 1, 1), (1, 1, 0, 1)),
             'v': ((1, 1, 1, 1), (0, 1, 0, 1)),
             'w': ((1, 0, 1, 1), (1, 0, 0, 1))}
    B = {field: basis(fx, L) + basis(fy, b)
         for field, (fx, fy) in flags.items()}
    A = E*h/(1 - nu**2)*np.array([[1, nu, 0], [nu, 1, 0],
                                  [0, 0, (1 - nu)/2]])
    D = A*h**2/12
    # NOTE each strain component as a sum of (field, order of the derivative
    #      in x, order of the derivative in y, coefficient)
    membrane = [[('u', 1, 0, 1.)], [('v', 0, 1, 1.), ('w', 0, 0, 1/R)],
                [('u', 0, 1, 1.), ('v', 1, 0, 1.)]]
    bending = [[('w', 2, 0, -1.)], [('w', 0, 2, -1.)], [('w', 1, 1, -2.)]]
    fields = ['u', 'v', 'w']
    size = {f: B[f][0][0].shape[1]*B[f][3][0].shape[1] for f in fields}
    offset = dict(zip(fields, np.cumsum([0] + [size[f] for f in fields])))
    K = np.zeros((sum(size.values()), sum(size.values())))
    for ops, C in ((membrane, A), (bending, D)):
        for s in range(3):
            for t in range(3):
                if C[s, t] == 0:
                    continue
                for fa, pa, qa, ca in ops[s]:
                    for fb, pb, qb, cb in ops[t]:
                        Fxa, wx, _, Fya, wy, _ = B[fa]
                        Fxb, _, _, Fyb, _, _ = B[fb]
                        Ix = (Fxa[pa]*wx[:, None]).T @ Fxb[pb]
                        Iy = (Fya[qa]*wy[:, None]).T @ Fyb[qb]
                        K[offset[fa]:offset[fa] + size[fa],
                          offset[fb]:offset[fb] + size[fb]] += (
                              C[s, t]*ca*cb*np.kron(Ix, Iy))
    fx, fy = flags['w']
    phiC = np.kron(vecf(m, -1., *fx)[B['w'][2]],
                   vecf(m, -1., *fy)[B['w'][5]])
    f = np.zeros(K.shape[0])
    f[offset['w']:] = -ROOF['Pmax']/4*phiC
    c = np.linalg.solve(K, f)
    return -phiC @ c[offset['w']:]


def run_hinged_roof(n=16, h=12.7, load_factor=1.2, initialInc=0.05,
                    maxInc=0.05, NL_method='arc_length_crisfield',
                    element='Quad4R', hgfactor_physical=1., **kwargs):
    r"""Non-linear analysis of :func:`hinged_roof` up to ``load_factor``
    times ``P_max``, by default with Crisfield's arc-length method

    Returns ``P/P_max`` and ``-W_C`` of the unloaded state and of the
    converged increments.
    """
    model, bu, dofC = hinged_roof(n, h, element, hgfactor_physical)
    model.fext *= load_factor
    an = model.analysis(bu, NL_method=NL_method, initialInc=initialInc,
                        maxInc=maxInc, **kwargs)
    increments, cs = an.static(NLgeom=True, silent=True)
    P = load_factor*np.array(increments)
    W = -np.array([u[dofC] for u in cs])
    return np.concatenate(([0.], P)), np.concatenate(([0.], W))


def compare_with_table_9a(P, W):
    r"""Load ``P/P_max`` of the computed path at the deflections ``-W_C`` of
    Table 9a, interpolated linearly along the path, which is monotonic in
    ``-W_C``"""
    assert np.all(np.diff(W) > 0)
    assert W[-1] > TABLE_9A[-1, 1]
    return np.interp(TABLE_9A[:, 1], W, P)


def limit_points(P, W, W_split):
    r"""Upper limit load before ``W_split`` and lower limit load after it,
    ``(P, W)`` pairs of the sampled path"""
    before = W < W_split
    i_max = np.flatnonzero(before)[np.argmax(P[before])]
    after = ~before
    i_min = np.flatnonzero(after)[np.argmin(P[after])]
    return (P[i_max], W[i_max]), (P[i_min], W[i_min])


@pytest.mark.slow
def test_hinged_roof_thick():
    r"""Snap-through of the 12.7 thick roof with Quad4R elements and
    Crisfield's arc-length method against Table 9a of Sze et al. (2004),
    computed with 16x16 S4R elements on the quarter

    The path of :func:`run_hinged_roof` against the mesh, ``initialInc =
    maxInc = 0.05``, measured on 2026-10-29: the limit loads, relative to
    Table 9a, 0.7421 and 0.1729, the largest difference of the load at the
    deflections of Table 9a, and the relative difference at its first
    point, 0.0877 P_max (times with ``OMP_NUM_THREADS=1``; Tria3DSG
    measured with three such runs in parallel):

    ============  ===============  ===============  ======  ======  =====
    mesh          upper            lower            max dP  first   time
    ============  ===============  ===============  ======  ======  =====
    8x8 Quad4R    0.74293 (+0.1%)  0.17183 (-0.6%)  0.0066  +0.8%     3 s
    12x12         0.74231 (+0.0%)  0.16912 (-2.2%)  0.012   +0.1%     7 s
    16x16         0.74206 (-0.0%)  0.16831 (-2.7%)  0.016   -0.2%    16 s
    24x24         0.74184 (-0.0%)  0.16735 (-3.2%)  0.020   -0.5%    59 s
    8x8 Tria3DSG  0.74253 (+0.1%)  0.17063 (-1.3%)  0.027   +3.4%    10 s
    16x16         0.74198 (-0.0%)  0.16793 (-2.9%)  0.0087  +0.8%    49 s
    32x32         0.74175 (-0.1%)  0.16691 (-3.5%)  0.016   -0.2%   331 s
    ============  ===============  ===============  ======  ======  =====

    (Tria3DSG: 2 x n x n elements.) Quad4R and Tria3DSG converge to the
    same path: the upper limit load within 0.1 % of Table 9a, the lower one
    about 3.5 % below it, Table 9a being itself a 16x16 S4R result. Up to
    0.11, with the hourglass stiffnesses of Brockman (1987), Quad4R was 6.7
    to 9.1 % stiffer at the first point, with the upper limit loads 0.5 to
    0.7 % above Table 9a and the lower ones +1.6 to -1.4 %; since 0.12.0 the
    bending and the transverse shear have the physical stabilisation of
    Belytschko and Leviathan (1994), see ``test_constant_twist_patch_test``
    in ``tests/test_quad4r_hourglass_control.py``.

    Following the policy of the tests whose reference is reached only on
    fine meshes, this test uses the 8x8 mesh and compares the two limit
    loads, the largest difference along the path and the difference at the
    first point with the values pre-determined by this same test with that
    mesh, with ``rtol = 1e-4`` for the platform round-off (fixed arc-length
    increments, each converged to ``relTOL = 1e-9``; two runs on one machine
    are identical to the bit). The limit loads are also checked against
    Table 9a with the grades of MacNeal and Harder (1985), see
    :func:`test_hinged_roof_thin_snap_back`: A for the upper one, B for the
    lower one, whose converged value differs from Table 9a by 3.5 %.
    """
    P, W = run_hinged_roof(n=8)
    P_calc = compare_with_table_9a(P, W)
    for (Pr, Wr), Pc in zip(TABLE_9A, P_calc):
        print('-W_C %7.3f  P/Pmax Sze et al. %.4f  pyfe3d %.4f'
              % (Wr, Pr, Pc))
    (Pmax, Wmax), (Pmin, Wmin) = limit_points(P, W, 15.)
    print('limit points: P/Pmax %.4f at -W_C %.3f, %.4f at %.3f'
          % (Pmax, Wmax, Pmin, Wmin))
    (Pmax_ref, _), (Pmin_ref, _) = limit_points(TABLE_9A[:, 0],
                                                TABLE_9A[:, 1], 15.)
    dP = np.abs(P_calc - TABLE_9A[:, 0]).max()
    first = P_calc[0]/TABLE_9A[0, 0] - 1
    # NOTE values pre-determined with the 8x8 mesh, see the docstring
    assert np.isclose(Pmax, 0.7429301050846626, rtol=1e-4)
    assert np.isclose(Pmin, 0.17182699153797426, rtol=1e-4)
    assert np.isclose(dP, 0.0066299675401739755, rtol=1e-4)
    assert np.isclose(first, 0.007793632923613236, rtol=1e-4)
    # NOTE grades A and B against Table 9a
    assert abs(Pmax/Pmax_ref - 1) <= 0.02
    assert abs(Pmin/Pmin_ref - 1) <= 0.10


@functools.lru_cache(maxsize=None)
def thin_roof_path():
    r"""Path of the 6.35 thick roof with 8x8 Quad4R elements on the quarter
    up to ``1.1*P_max``, ``initialInc = maxInc = 0.075``, computed once for
    the tests that use it

    The mesh of Table 9d of Sze et al. (2004) is 24x24; the convergence
    from 8x8 to 48x48 is in :func:`test_hinged_roof_thin_snap_back`.
    """
    return run_hinged_roof(n=8, h=6.35, load_factor=1.1, initialInc=0.075,
                           maxInc=0.075)


@pytest.mark.slow
def test_hinged_roof_thin_snap_back():
    r"""Snap-through and snap-back of the 6.35 thick roof with Quad4R
    elements and Crisfield's arc-length method against Table 9d of Sze et
    al. (2004), computed with 24x24 S4R elements on the quarter

    The path turns back in ``-W_C`` twice. Limit loads against the mesh,
    with the arc-length increments ``initialInc = maxInc = 0.02``, which
    sample the lower limit point more finely than the 0.075 of
    :func:`thin_roof_path`, measured on 2026-10-29 with ``OMP_NUM_THREADS=1``
    and three runs in parallel:

    =====  ===============  ================  =====
    mesh   upper, P/P_max   lower, P/P_max    time
    =====  ===============  ================  =====
    8x8    0.19915 (+2.0%)  -0.13134 (+2.0%)  7 s
    12x12  0.19683 (+0.8%)  -0.13010 (+1.0%)  18 s
    16x16  0.19601 (+0.4%)  -0.12964 (+0.7%)  40 s
    24x24  0.19542 (+0.1%)  -0.12922 (+0.3%)  129 s
    32x32  0.19521 (-0.0%)  -0.12908 (+0.2%)  228 s
    48x48  0.19505 (-0.1%)  -0.12883 (+0.0%)  451 s
    =====  ===============  ================  =====

    the percentages relative to Table 9d, 0.1953 and -0.1288 (a positive
    one for the lower limit load means a more negative load). Tria3DSG, 2 x
    n x n elements, gives 0.19558 and -0.12916 with 16x16, 0.19511 and
    -0.12903 with 32x32. Both limit loads converge to Table 9d, see
    :func:`test_hinged_roof_thin_lower_limit_load`.

    Following the policy of the tests whose reference is reached only on
    fine meshes, this test uses the path of :func:`thin_roof_path`, 8x8
    elements and the increments 0.075, and compares its limit loads with
    the values pre-determined by this same test, 0.19877860583652432 and
    -0.1290755052637671, with ``rtol = 1e-4`` for the platform round-off
    (the path is sampled at fixed arc-length increments and each increment
    converges to ``relTOL = 1e-9``; two runs on one machine are identical to
    the bit). The upper limit load is also checked against Table 9d with the
    tolerance of grade A of MacNeal, R. H. and Harder, R. L., A proposed
    standard set of problems to test finite element accuracy, Finite
    Elements in Analysis and Design, 1(1), 3-20, 1985,
    https://doi.org/10.1016/0168-874X(85)90003-4, p. 14: a normalised
    error within 2 %. Until 0.12.0 this test used the 24x24 mesh, about 23
    s.
    """
    P, W = thin_roof_path()
    assert np.isclose(P[-1], 1.1)
    # NOTE snap-back: -W_C decreases along part of the path
    dW = np.diff(W)
    assert np.any(dW < 0)
    (Pmax, Wmax), _ = limit_points(P, W, 13.5)
    Pmin = P.min()
    (Pmax_ref, _), _ = limit_points(TABLE_9D[:, 0], TABLE_9D[:, 1], 13.5)
    Pmin_ref = TABLE_9D[:, 0].min()
    print('limit loads: P/Pmax %.4f, %.4f; Table 9d: %.4f, %.4f'
          % (Pmax, Pmin, Pmax_ref, Pmin_ref))
    # NOTE values pre-determined with the 8x8 mesh, see the docstring
    assert np.isclose(Pmax, 0.19877860583652432, rtol=1e-4)
    assert np.isclose(Pmin, -0.1290755052637671, rtol=1e-4)
    # NOTE grade A against Table 9d
    assert abs(Pmax/Pmax_ref - 1) <= 0.02
    # NOTE the lower limit load is negative, the roof snaps back
    assert Pmin < 0


@pytest.mark.slow
def test_hinged_roof_thin_lower_limit_load():
    r"""Lower limit load of the 6.35 thick roof with the Quad4R elements of
    :func:`thin_roof_path` against Table 9d of Sze et al. (2004)

    The lower limit load converges to Table 9d, -0.1288 P_max, see the table
    of :func:`test_hinged_roof_thin_snap_back`; Tria3DSG gives -0.12903 with
    32x32 elements. Up to 0.11 this was an expected failure: Quad4R
    controlled the hourglass mode of `w` with Brockman's operator, which
    added 73 to 78 % to the energy of a uniform twist whatever the mesh in
    the unit of length of this model, and the lower limit load converged to
    -0.1435 to -0.1440, 11.4 to 11.8 % below Table 9d, from 8x8 to 48x48
    elements; see ``test_constant_twist_patch_test`` in
    ``tests/test_quad4r_hourglass_control.py``. Since 0.12.0 the bending and
    the transverse shear of Quad4R have the physical stabilisation of
    Belytschko and Leviathan (1994), which passes that patch test.

    With the 8x8 mesh and the increments 0.075 of :func:`thin_roof_path`
    the lower limit load is -0.12908, 0.2 % below Table 9d; sampled with the
    increments 0.02 it is -0.13134, 2.0 % below. It is compared with the
    value pre-determined by :func:`test_hinged_roof_thin_snap_back`, and
    with Table 9d with grade B of MacNeal and Harder (1985), see that test,
    as a bound that also holds for the finer sampling.
    """
    P, W = thin_roof_path()
    Pmin = P.min()
    Pmin_ref = TABLE_9D[:, 0].min()
    print('lower limit load: P/Pmax %.4f; Table 9d: %.4f' % (Pmin, Pmin_ref))
    assert np.isclose(Pmin, -0.1290755052637671, rtol=1e-4)
    # NOTE grade B against Table 9d
    assert abs(Pmin/Pmin_ref - 1) <= 0.10


@pytest.mark.slow
def test_hinged_roof_hourglass_sensitivity():
    r"""Linear deflection ``-W_C`` of the 12.7 thick roof under ``P_max``
    against the physical stabilisation of the bending and of the transverse
    shear of Quad4R, ``hgfactor_physical`` times the default, with 16x16
    meshes

    Since 0.12.0 the bending and the transverse shear of Quad4R have the
    physical stabilisation of Belytschko and Leviathan (1994), which has no
    parameter; ``hgfactor_physical`` multiplies it. The deflection grows
    monotonically as ``hgfactor_physical`` decreases from 1 to 0.002, by
    1.7 %. With the transverse shear stiffnesses multiplied by 100, close
    to the Kirchhoff theory, Tria3DSG and Quad4R converge to the
    shallow-shell Kirchhoff solution of :func:`ritz_roof_WC`, itself
    converged within 0.01 % (7.47913 and 7.47981 with ``m = 15`` and 20).
    Measured on 2026-10-29, relative to the Ritz solution with ``m = 20``,
    transverse shear x100 (time of all ten meshes of a row, with
    ``OMP_NUM_THREADS=1``):

    =====  ========  ===============  ===============  ======
    mesh   Tria3DSG  Quad4R, default  Quad4R, 0.002    time
    =====  ========  ===============  ===============  ======
    16x16  -1.07 %   -0.164 %         +0.008 %          0.5 s
    32x32  -0.23 %   +0.012 %         +0.084 %          4.5 s
    64x64  -0.005 %  +0.067 %         +0.107 %         27 s
    =====  ========  ===============  ===============  ======

    the two elements ending about 0.1 % above the Ritz solution with
    128x128 elements, 0.061 % (Tria3DSG) and 0.085 % (Quad4R), the residual
    transverse shear flexibility of the stiffnesses x100. With the actual
    transverse shear stiffness Quad4R with the default is 1.05, 0.51 and
    0.34 % more flexible than Tria3DSG on the three meshes. Up to 0.11,
    with the hourglass stiffnesses of Brockman (1987), Quad4R with the
    default was 6.4 to 7.1 % stiffer than Tria3DSG and 6.95, 6.35 and 5.95 %
    stiffer than the Ritz solution with the shear x100, an excess that
    refinement did not remove, see ``test_constant_twist_patch_test`` in
    ``tests/test_quad4r_hourglass_control.py``.

    Following the policy of the tests whose reference is reached only on
    fine meshes, this test uses the 16x16 meshes and compares the ten
    deflections with the values pre-determined by this same test with these
    meshes, with ``rtol = 1e-6`` for the round-off of the sparse linear
    solution, and Tria3DSG and Quad4R with the shear x100 with the Ritz
    solution with grade A of MacNeal and Harder (1985), see
    :func:`test_hinged_roof_thin_snap_back`.
    """
    W_ritz = [ritz_roof_WC(m) for m in (15, 20)]
    hgs = [1., 0.1, 0.01, 0.002]
    W_tria = linear_roof_WC(16, element='Tria3DSG')
    W_quad = [linear_roof_WC(16, hgfactor_physical=hg) for hg in hgs]
    W_tria_K = linear_roof_WC(16, element='Tria3DSG', shear_factor=100.)
    W_quad_K = [linear_roof_WC(16, hgfactor_physical=hg, shear_factor=100.)
                for hg in hgs]
    print('-W_C, Ritz Kirchhoff: %.4f (m = 15), %.4f (m = 20)' % tuple(W_ritz))
    print('-W_C, 16x16 Tria3DSG: %.4f; transverse shear x100: %.4f'
          % (W_tria, W_tria_K))
    for hg, W, W_K in zip(hgs, W_quad, W_quad_K):
        print('-W_C, 16x16 Quad4R, hgfactor_physical = %g: %.4f; transverse '
              'shear x100: %.4f' % (hg, W, W_K))
    # NOTE values pre-determined with the 16x16 meshes, see the docstring:
    #      Tria3DSG, Quad4R with hgfactor_physical = 1, 0.1, 0.01 and
    #      0.002, with the actual transverse shear stiffness and with it x100
    predetermined = [7.547952062067695, 7.6271559336711245,
                     7.683069380030956, 7.728435615397175, 7.7563567602480505,
                     7.399604085019292, 7.4675106211328295, 7.477330085984253,
                     7.479558040946031, 7.480410578960498]
    np.testing.assert_allclose([W_tria] + W_quad + [W_tria_K] + W_quad_K,
                               predetermined, rtol=1e-6)
    assert abs(W_ritz[0]/W_ritz[1] - 1) < 2e-4
    assert np.all(np.diff(W_quad) > 0)
    assert np.all(np.diff(W_quad_K) > 0)
    # NOTE grade A against the Ritz solution
    assert abs(W_tria_K/W_ritz[1] - 1) <= 0.02
    assert abs(W_quad_K[0]/W_ritz[1] - 1) <= 0.02
    assert abs(W_quad_K[-1]/W_ritz[1] - 1) <= 0.02
