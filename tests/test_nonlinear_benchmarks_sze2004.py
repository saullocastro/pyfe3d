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

The roof is modelled with Quad4R and its default hourglass control, of
Brockman (1987), see :mod:`pyfe3d.quad4r`. The transverse hourglass
stiffness, ``0.10*E*h**3/(1 + 1/A)``, is ``0.94*0.1*E*h**3`` to
``0.1*E*h**3`` for the element areas ``A`` of the 64x64 to 8x8 meshes of the
quarter, 16 to 1008 in the unit of length of the model, and it also stiffens
the twist curvature, by about ``3*0.1 = 30 %``. The sensitivity of the roof
to this stiffness, scaled with the ``hgfactor_w`` of Quad4R, is measured by
:func:`test_hinged_roof_hourglass_sensitivity`, against Tria3DSG, which has
no hourglass control, and against the shallow-shell Kirchhoff solution of
:func:`ritz_roof_WC`. The hourglass stiffness that suits a problem depends on
the problem; the default is kept.
"""
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


@pytest.mark.parametrize('name', ['BeamC', 'BeamLR'])
def test_cantilever_end_moment_beams_membrane_locking(name):
    r"""The beam elements interpolate the axial displacement linearly and
    integrate the von Karman axial strain ``u_x + w_x**2/2`` at Gauss
    points along the element, where ``w_x**2`` is not constant, so that the
    axial strain cannot vanish and the cantilever is too stiff (membrane
    locking); the error decreases with the number of elements"""
    errors = []
    for num_elements in (16, 64, 256):
        m, U, W = run_cantilever_end_moment(name, num_elements, inc=0.25,
                                            m_max=0.05)
        U_vk, W_vk = von_karman_tip(m[-1])
        errors.append(abs(W[-1] - W_vk)/W_vk)
        print('%s %3d elements, M/Mmax %.2f: W/L %.5f von Karman %.5f, '
              'error %.2f %%' % (name, num_elements, m[-1], W[-1], W_vk,
                                 100*errors[-1]))
    assert errors[0] > 0.4
    assert errors[0] > errors[1] > errors[2]
    assert errors[2] < 0.015


def hinged_roof(n=16, h=12.7, element='Quad4R', hgfactor_w=1.,
                shear_factor=1.):
    r"""Quarter of the hinged cylindrical roof of Fig. 9a of Sze et al.
    (2004) meshed with ``n x n`` Quad4R elements, or ``2 x n x n`` Tria3DSG
    elements

    The axis of the cylinder is along X, the quarter covers ``0 <= X <= L``
    and ``0 <= theta <= 0.1``, the point C at ``X = 0, theta = 0``. The
    reference load is ``P_max`` on the whole roof, a quarter of it on the
    quarter. ``hgfactor_w`` multiplies the transverse hourglass stiffness of
    Quad4R, ``1.`` being the default of Quad4R; ``shear_factor`` multiplies
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
        elem_kwargs = dict(hgfactor_w=hgfactor_w)
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


def linear_roof_WC(n=16, h=12.7, element='Quad4R', hgfactor_w=1.,
                   shear_factor=1.):
    r"""Linear deflection ``-W_C`` of :func:`hinged_roof` under ``P_max``"""
    model, bu, dofC = hinged_roof(n, h, element, hgfactor_w, shear_factor)
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
                    element='Quad4R', hgfactor_w=1., **kwargs):
    r"""Non-linear analysis of :func:`hinged_roof` up to ``load_factor``
    times ``P_max``, by default with Crisfield's arc-length method

    Returns ``P/P_max`` and ``-W_C`` of the unloaded state and of the
    converged increments.
    """
    model, bu, dofC = hinged_roof(n, h, element, hgfactor_w)
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


def test_hinged_roof_thick():
    r"""Snap-through of the 12.7 thick roof with 8x8 Quad4R elements and
    Crisfield's arc-length method against Table 9a of Sze et al. (2004)

    The limit loads agree within 1.5 %, the load at a given deflection
    within 0.07 P_max along the path; pyfe3d is stiffer at the start of the
    path, about 9 % at the first point of Table 9a, see
    :func:`test_hinged_roof_hourglass_sensitivity`.
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
    assert abs(Pmax - Pmax_ref) < 0.015*Pmax_ref
    assert abs(Pmin - Pmin_ref) < 0.015*Pmin_ref
    assert np.abs(P_calc - TABLE_9A[:, 0]).max() < 0.07
    assert 0.05 < P_calc[0]/TABLE_9A[0, 0] - 1 < 0.12


def test_hinged_roof_thin_snap_back():
    r"""Snap-through and snap-back of the 6.35 thick roof with 16x16 Quad4R
    elements and Crisfield's arc-length method against Table 9d of Sze et
    al. (2004)

    The path turns back in ``-W_C`` twice; the upper limit load agrees
    within 1 % with the largest load of Table 9d, the lower limit load,
    negative, is 0.015 P_max lower than the smallest load of Table 9d.
    """
    P, W = run_hinged_roof(n=16, h=6.35, load_factor=1.1)
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
    assert abs(Pmax - Pmax_ref) < 0.01*Pmax_ref
    assert -0.02 < Pmin - Pmin_ref < -0.01


def test_hinged_roof_hourglass_sensitivity():
    r"""Linear deflection ``-W_C`` of the 12.7 thick roof under ``P_max``
    against the transverse hourglass stiffness of Quad4R, ``hgfactor_w``
    times the default, with 32x32 meshes

    The deflection grows monotonically as ``hgfactor_w`` decreases from 1 to
    0.002, and with the default the Quad4R mesh is more than 5 % stiffer
    than the Tria3DSG mesh, which has no hourglass control. With the
    transverse shear stiffnesses multiplied by 100, close to the Kirchhoff
    theory, Tria3DSG agrees within 0.5 % with the shallow-shell Kirchhoff
    solution of :func:`ritz_roof_WC`, converged within 0.02 %, Quad4R with
    the default is more than 5 % stiffer and approaches it as
    ``hgfactor_w`` decreases. The values of ``hgfactor_w`` measure the
    sensitivity of this problem; they are not recommendations.
    """
    W_ritz = [ritz_roof_WC(m) for m in (15, 20)]
    hgs = [1., 0.1, 0.01, 0.002]
    W_tria = linear_roof_WC(32, element='Tria3DSG')
    W_quad = [linear_roof_WC(32, hgfactor_w=hg) for hg in hgs]
    W_tria_K = linear_roof_WC(32, element='Tria3DSG', shear_factor=100.)
    W_quad_K = [linear_roof_WC(32, hgfactor_w=hg, shear_factor=100.)
                for hg in hgs]
    print('-W_C, Ritz Kirchhoff: %.4f (m = 15), %.4f (m = 20)' % tuple(W_ritz))
    print('-W_C, 32x32 Tria3DSG: %.4f; transverse shear x100: %.4f'
          % (W_tria, W_tria_K))
    for hg, W, W_K in zip(hgs, W_quad, W_quad_K):
        print('-W_C, 32x32 Quad4R, hgfactor_w = %g: %.4f; transverse shear '
              'x100: %.4f' % (hg, W, W_K))
    assert abs(W_ritz[0]/W_ritz[1] - 1) < 2e-4
    assert np.all(np.diff(W_quad) > 0)
    assert np.all(np.diff(W_quad_K) > 0)
    assert W_tria/W_quad[0] - 1 > 0.05
    assert abs(W_tria_K/W_ritz[1] - 1) < 0.005
    assert 1 - W_quad_K[0]/W_ritz[1] > 0.05
    assert abs(W_quad_K[-1]/W_ritz[1] - 1) < 0.005
