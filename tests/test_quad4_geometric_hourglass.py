r"""Quad4: slopes of `w` in the geometric stiffness

The lowest buckling load of a hard simply supported square sandwich-like
plate under uniaxial compression, with reduced transverse shear stiffness,
is compared with Eq. (1) of Seide and Stowell (1950) for the rules of the
slopes of `w` (``shear_model = 0``, ``centroid_slopes = 1`` and the default);
the geometric stiffness of the pre-buckling state, the slope rule of KG, the
nonlinear terms of the hourglass pattern and the tangent stiffness against
the Jacobian of the internal forces are checked as well.

Notes
-----
With ``shear_model = 0`` the elements of :class:`pyfe3d.Quad4` with
`h/\ell < 1` integrate the transverse shear with one point at the centroid,
after Hughes et al. (1977), such that the hourglass pattern of `w`, `w =
\xi \eta` with zero rotations, has no stiffness in KC0, see
``tests/test_quad4_spurious_shear_mode.py``. If the geometric stiffness KG,
and the geometrically nonlinear terms that share its quadrature, took the
slopes `w_{,x}` and `w_{,y}` at the two-by-two Gauss-Legendre points with
that transverse shear, the pattern would have the geometric stiffness of
the full quadrature. In a supported plate the pattern then costs only the
transverse shear energy of its smooth envelope, independent of the mesh,
whereas its geometric work grows with the square of the number of elements
per side, and a linear buckling analysis can return a spurious mode of `w`
alternating in sign from node to node, with rotations close to zero, below
the physical buckling load, once the transverse shear stiffness is low
compared with the bending stiffness, as in a sandwich plate.

The slopes therefore depend on ``shear_model``, see the Notes of
:meth:`pyfe3d.Quad4.update_KG`: with ``shear_model = 0`` they are taken at
the centroid, where the pattern has no slope, and with the default
``shear_model = 1``, the assumed transverse shear strains of the MITC4
element, which give the pattern its transverse shear stiffness, at the
two-by-two points; ``centroid_slopes = 1`` takes them at the centroid with
both.

Hard simply supported square plate, side 0.5, `b/h = 100`, uniaxial
compression, `A_{44}` and `A_{55}` reduced to a fraction of the isotropic
value, lowest buckling load against Eq. (1) of Seide and Stowell (1950),

.. math::

    N_{cr} = k \frac{\pi^2 D}{b^2}, \qquad
    k = \min_m \frac{(m b/a + a/(m b))^2}{1 + r ((m b/a)^2 + 1)}, \qquad
    r = \frac{\pi^2 D}{b^2 D_Q}

with the transverse shear stiffness `D_Q = A_{55}`, `m = 1` for these
plates: with the one-point transverse shear of ``shear_model = 0`` and the
slopes at the two-by-two points (``2x2``, a combination that
:class:`pyfe3d.Quad4` does not offer) and at the centroid
(``shear_model = 0``), and with the MITC4 strains of ``shear_model = 1`` and
the slopes at the centroid (``centroid_slopes = 1``) and at the two-by-two
points (default)::

    A44, A55   mesh   h/l    2x2       shear_model=0  centroid   default
    1/100      8x8    0.08   +1.98 %     +3.33 %      +3.41 %    +2.06 %
               16x16  0.16  -49.18 %     +0.81 %      +0.83 %    +0.51 %
               32x32  0.32  -87.08 %     +0.20 %      +0.21 %    +0.13 %
    1/1000     8x8    0.08  -71.39 %     +2.19 %      +2.70 %    +1.36 %
               16x16  0.16  -92.37 %     +0.54 %      +0.66 %    +0.34 %
               32x32  0.32  -98.06 %     +0.13 %      +0.17 %    +0.08 %

The negative errors are the spurious mode: 0.70 to 0.79 of `w` in it is the
pattern alternating in sign from node to node, and the rotations times the
element size are below 4e-3 of the largest `w`. With the slopes at the
centroid the pattern has no geometric stiffness, and with the MITC4 strains
it has the transverse shear stiffness of the element at every refinement, so
that the default has no spurious mode either: the error is positive and
decreases with the square of the element size. The ratio of the errors of
two successive meshes from 4x4 to 64x64 is, for 1/100, 4.36, 4.08, 4.02 and
4.01 with ``shear_model = 0``, +0.051 % at 64x64, 4.38, 4.09, 4.02 and 4.01
with ``centroid_slopes = 1``, +0.052 %, and 4.22, 4.05, 4.01 and 4.00 with
the default, +0.032 %; for 1/1000, 4.21, 4.05, 4.01 and 4.00, +0.034 %,
4.32, 4.08, 4.02 and 4.00, +0.041 %, and 4.13, 4.03, 4.01 and 4.00,
+0.021 % (``2x2`` measured with an element that takes the slopes at the
two-by-two points, the other columns on 2026-11-01).
:func:`test_low_transverse_shear_stiffness` asserts this convergence for the
three rules: the error is positive on every mesh, the ratio of the errors of
the 16x16 and 32x32 meshes lies between 3.8 and 4.2, the order 2 of the
element, and the 32x32 mesh is within 0.5 % of the reference. The case 1/100
with 8x8 elements is not affected by the spurious mode, whose load decreases
as the square of the element size.

References
----------
Seide, P., and Stowell, E. Z., Elastic and plastic buckling of simply
supported solid-core sandwich plates in compression, NACA Report 967, 1950,
Eq. (1), p. 2.

Hughes, T. J. R., Taylor, R. L., and Kanoknukulchai, W., A simple and
efficient finite element for plate bending, International Journal for
Numerical Methods in Engineering, 11(10), 1977, pp. 1529-1543,
https://doi.org/10.1002/nme.1620111005.

"""
import sys
sys.path.append('..')

import numpy as np
import pytest
from numpy import isclose
from scipy.sparse import coo_matrix

from pyfe3d.shellprop_utils import isotropic_plate
from pyfe3d import Quad4, Quad4Data, Quad4Probe, INT, DOUBLE, DOF
from pyfe3d.solver import linear_buckling

E = 203.e9
NU = 0.33
SIDE = 0.5
H = 0.005


def square_mesh(n):
    xtmp = np.linspace(0, SIDE, n)
    xmesh, ymesh = np.meshgrid(xtmp, xtmp)
    x = xmesh.T.flatten()
    y = ymesh.T.flatten()
    nids = np.arange(n*n).reshape(n, n)
    elements = list(zip(nids[:-1, :-1].flatten(), nids[1:, :-1].flatten(),
                        nids[1:, 1:].flatten(), nids[:-1, 1:].flatten()))
    return x, y, elements


def assemble(x, y, elements, prop, Nxx=-1., given_stress=True,
             shear_model=1, centroid_slopes=0):
    r"""KC0 and KG; with ``given_stress=False``, KG is computed by
    ``update_KG`` from the uniform pre-buckling state with `N_{xx} =` ``Nxx``
    and `N_{yy} = N_{xy} = 0`"""
    data = Quad4Data()
    probe = Quad4Probe()
    coords = np.vstack((x, y, 0*x)).T.flatten()
    num_elements = len(elements)
    N = DOF*len(x)
    KC0r = np.zeros(data.KC0_SPARSE_SIZE*num_elements, dtype=INT)
    KC0c = np.zeros_like(KC0r)
    KC0v = np.zeros(data.KC0_SPARSE_SIZE*num_elements, dtype=DOUBLE)
    KGr = np.zeros(data.KG_SPARSE_SIZE*num_elements, dtype=INT)
    KGc = np.zeros_like(KGr)
    KGv = np.zeros(data.KG_SPARSE_SIZE*num_elements, dtype=DOUBLE)
    u0 = np.zeros(N)
    eps = Nxx/(E*prop.h)
    u0[0::DOF] = eps*x
    u0[1::DOF] = -NU*eps*y
    for k, (n1, n2, n3, n4) in enumerate(elements):
        quad = Quad4(probe)
        quad.shear_model = shear_model
        quad.centroid_slopes = centroid_slopes
        quad.n1, quad.n2, quad.n3, quad.n4 = n1, n2, n3, n4
        quad.c1, quad.c2, quad.c3, quad.c4 = DOF*n1, DOF*n2, DOF*n3, DOF*n4
        quad.init_k_KC0 = k*data.KC0_SPARSE_SIZE
        quad.init_k_KG = k*data.KG_SPARSE_SIZE
        quad.update_rotation_matrix(coords)
        quad.update_probe_xe(coords)
        quad.update_KC0(KC0r, KC0c, KC0v, prop)
        if given_stress:
            quad.update_KG_given_stress(Nxx, 0., 0., KGr, KGc, KGv)
        else:
            quad.update_probe_ue(u0)
            quad.update_KG(KGr, KGc, KGv, prop)
    KC0 = coo_matrix((KC0v, (KC0r, KC0c)), shape=(N, N)).tocsc()
    KG = coo_matrix((KGv, (KGr, KGc)), shape=(N, N)).tocsc()
    return KC0, KG


def seide_stowell(a, b, D, DQ):
    r"""Buckling load per unit width of a simply supported plate under
    uniaxial compression with transverse shear deformation, Eq. (1) of
    Seide and Stowell (1950), minimum over the number `m` of half-waves
    along the load, with `r = \pi^2 D/(b^2 D_Q)`"""
    r = np.pi**2*D/(b**2*DQ)
    m = np.arange(1, 101)
    k = (m*b/a + a/(m*b))**2/(1 + r*((m*b/a)**2 + 1))
    return k.min()*np.pi**2*D/b**2


def buckling_ss_plate(n, shear_factor, given_stress=True, shear_model=1,
                      centroid_slopes=0):
    r"""Lowest buckling load of a hard simply supported square plate under
    uniaxial compression, relative to Seide and Stowell (1950), and the
    checkerboard content of `w` in its mode, between 0 and 1"""
    prop = isotropic_plate(E=E, nu=NU, thickness=H)
    prop.A44 *= shear_factor
    prop.A55 *= shear_factor
    x, y, elements = square_mesh(n)
    KC0, KG = assemble(x, y, elements, prop, given_stress=given_stress,
                       shear_model=shear_model,
                       centroid_slopes=centroid_slopes)
    bk = np.zeros(KC0.shape[0], dtype=bool)
    edge = (isclose(x, 0.) | isclose(x, SIDE) | isclose(y, 0.)
            | isclose(y, SIDE))
    for i in range(3):
        bk[i::DOF] = edge
    bk[3::DOF] |= isclose(x, 0.) | isclose(x, SIDE)
    bk[4::DOF] |= isclose(y, 0.) | isclose(y, SIDE)
    bu = ~bk
    eigvals, eigvecs = linear_buckling(KC0[bu][:, bu], KG[bu][:, bu],
                                       num_eigvalues=2, tol=1e-9)
    u = np.zeros(KC0.shape[0])
    u[bu] = eigvecs[:, 0]
    w = u[2::DOF].reshape(n, n)
    sign = (-1.)**np.add.outer(np.arange(n), np.arange(n))
    checkerboard = abs((w*sign).sum())/np.sqrt((w**2).sum()*n*n)
    D = E*H**3/(12*(1 - NU**2))
    N_ref = seide_stowell(SIDE, SIDE, D, prop.A55)
    return eigvals[0]/N_ref - 1, checkerboard


#: rules of the slopes of w: (shear_model, centroid_slopes)
RULES = [(0, 0), (1, 1), (1, 0)]


@pytest.mark.slow
def test_low_transverse_shear_stiffness():
    r"""Sandwich plate of the module documentation, the three rules of the
    slopes: no spurious mode and second-order convergence to Seide and
    Stowell (1950), Eq. (1)"""
    for shear_factor in (1e-2, 1e-3):
        for shear_model, centroid_slopes in RULES:
            errs = []
            for n in (9, 17, 33):
                err, checkerboard = buckling_ss_plate(
                    n, shear_factor, shear_model=shear_model,
                    centroid_slopes=centroid_slopes)
                print('A44 = A55 = %g of the isotropic value, shear_model '
                      '%d, centroid_slopes %d, %dx%d elements: %+.2f %%, '
                      'checkerboard %.3f'
                      % (shear_factor, shear_model, centroid_slopes, n - 1,
                         n - 1, 100*err, checkerboard))
                # NOTE convergence from above, without the spurious mode
                assert err > 0
                assert checkerboard < 0.05
                errs.append(err)
            # NOTE second-order convergence to the reference
            assert 3.8 < errs[1]/errs[2] < 4.2
            assert errs[2] < 0.005


def test_KG_of_the_pre_buckling_state():
    r"""``update_KG`` with the stress resultants of a uniform pre-buckling
    state gives the load of ``update_KG_given_stress``, for the three rules
    of the slopes"""
    for shear_model, centroid_slopes in RULES:
        err_given, _ = buckling_ss_plate(
            17, 1e-2, given_stress=True, shear_model=shear_model,
            centroid_slopes=centroid_slopes)
        err_state, checkerboard = buckling_ss_plate(
            17, 1e-2, given_stress=False, shear_model=shear_model,
            centroid_slopes=centroid_slopes)
        assert isclose(err_state, err_given, rtol=0, atol=1e-8)
        assert checkerboard < 0.05


def test_thick_plate_two_by_two_slopes():
    r"""Hard simply supported square plate with `b/h = 10` under uniaxial
    compression, 12 by 12 elements, `h/\ell = 1.2`, against the Navier
    solution of the Mindlin plate,

    Notes
    -----
    .. math::

        N_{cr} = \frac{N_{cr,K}}{1 + D \alpha^2/A_{55}}, \qquad \alpha^2 =
        \left(\frac{\pi}{a}\right)^2 + \left(\frac{\pi}{b}\right)^2

    5.56 % below the Kirchhoff value `N_{cr,K} = 4 \pi^2 D/b^2`, see
    ``tests/test_quad4_linear_buckling_plate_rotated_thick_elements.py``:
    the default, the MITC4 transverse shear with the slopes of `w` at the
    two-by-two points, is +0.91 % above it, and +1.49 % with the slopes at
    the centroid, ``centroid_slopes = 1`` or ``shear_model = 0``, whose
    elements with `h/\ell \ge 1` have the MITC4 strains as well (measured
    on 2026-11-01); +0.23 % and +0.37 % with 24 by 24 elements"""
    h = 0.05
    prop = isotropic_plate(E=E, nu=NU, thickness=h)
    D = E*h**3/(12*(1 - NU**2))
    N_K = 4*np.pi**2*D/SIDE**2
    N_ref = N_K/(1 + D*2*(np.pi/SIDE)**2/prop.A55)
    assert isclose(1 - N_ref/N_K, 0.0556, atol=5e-5)
    x, y, elements = square_mesh(13)
    edge = (isclose(x, 0.) | isclose(x, SIDE) | isclose(y, 0.)
            | isclose(y, SIDE))
    errs = {}
    for shear_model, centroid_slopes in RULES:
        KC0, KG = assemble(x, y, elements, prop, shear_model=shear_model,
                           centroid_slopes=centroid_slopes)
        bk = np.zeros(KC0.shape[0], dtype=bool)
        for i in range(3):
            bk[i::DOF] = edge
        bk[3::DOF] |= isclose(x, 0.) | isclose(x, SIDE)
        bk[4::DOF] |= isclose(y, 0.) | isclose(y, SIDE)
        bu = ~bk
        eigvals, _ = linear_buckling(KC0[bu][:, bu], KG[bu][:, bu],
                                     num_eigvalues=2, tol=1e-9)
        errs[shear_model, centroid_slopes] = eigvals[0]/N_ref - 1
        print('shear_model %d, centroid_slopes %d: %+.3f %%'
              % (shear_model, centroid_slopes,
                 100*errs[shear_model, centroid_slopes]))
    assert isclose(errs[1, 0], 0.00906, atol=5e-5)
    assert isclose(errs[1, 1], 0.01489, atol=5e-5)
    assert isclose(errs[0, 0], errs[1, 1], rtol=1e-8)


def single_element(xy, h=0.01, shear_model=1, centroid_slopes=0):
    x, y = xy[:, 0], xy[:, 1]
    prop = isotropic_plate(E=E, nu=NU, thickness=h)
    data = Quad4Data()
    probe = Quad4Probe()
    quad = Quad4(probe)
    quad.shear_model = shear_model
    quad.centroid_slopes = centroid_slopes
    quad.n1, quad.n2, quad.n3, quad.n4 = 0, 1, 2, 3
    quad.c1, quad.c2, quad.c3, quad.c4 = 0, DOF, 2*DOF, 3*DOF
    quad.init_k_KC0 = 0
    quad.init_k_KCNL = 0
    quad.init_k_KG = 0
    coords = np.vstack((x, y, 0*x)).T.flatten()
    quad.update_rotation_matrix(coords)
    quad.update_probe_xe(coords)
    return quad, data, prop, coords


def test_KG_slope_rule():
    r"""KG with constant stress resultants of a distorted element, with
    ``update_KG_given_stress`` and with ``update_KG`` of a uniform strain
    state, for `h/\ell < 1` and `h/\ell \ge 1`: with ``shear_model = 0`` or
    ``centroid_slopes = 1`` it equals the element area times the integrand
    at the centroid, and the hourglass pattern of `w` has no geometric
    stiffness; with the default it equals the two-by-two Gauss-Legendre
    quadrature with the slopes at its points, computed here with numpy, and
    the pattern has geometric stiffness"""
    for h in (0.01, 2.):
        for shear_model, centroid_slopes in RULES:
            check_KG_slope_rule(h, shear_model, centroid_slopes)


def shape_derivatives(xy, s, t):
    r"""Derivatives of the bilinear shape functions with respect to the
    coordinates ``xy``, shape ``(2, 4)``, and the determinant of the
    Jacobian, at the natural coordinates ``s, t``"""
    xi = np.array([-1., 1., 1., -1.])
    eta = np.array([-1., -1., 1., 1.])
    dxi = xi*(1 + eta*t)/4
    deta = eta*(1 + xi*s)/4
    J = np.array([[dxi @ xy[:, 0], dxi @ xy[:, 1]],
                  [deta @ xy[:, 0], deta @ xy[:, 1]]])
    return np.linalg.solve(J, np.vstack((dxi, deta))), np.linalg.det(J)


def check_KG_slope_rule(h, shear_model, centroid_slopes):
    xy = np.array([[0., 0.], [2., 0.], [2.3, 1.4], [0.2, 1.1]])
    quad, data, prop, coords = single_element(xy, h, shear_model,
                                              centroid_slopes)
    Nxx, Nyy, Nxy = -3., 1.5, 0.7
    # NOTE stress resultants from the element axes, the columns of r, to the
    #      global axes
    r = np.array([[quad.r11, quad.r12], [quad.r21, quad.r22]])
    Nmat = r @ np.array([[Nxx, Nxy], [Nxy, Nyy]]) @ r.T
    at_centroid = shear_model == 0 or centroid_slopes != 0
    if at_centroid:
        dN, _ = shape_derivatives(xy, 0., 0.)
        Kww_ref = quad.area*dN.T @ Nmat @ dN
    else:
        Kww_ref = np.zeros((4, 4))
        for s in (-1/np.sqrt(3), 1/np.sqrt(3)):
            for t in (-1/np.sqrt(3), 1/np.sqrt(3)):
                dN, detJ = shape_derivatives(xy, s, t)
                Kww_ref += detJ*dN.T @ Nmat @ dN
    hourglass = np.array([1., -1., 1., -1.])

    def check(Kww, rtol):
        assert np.allclose(Kww, Kww_ref, rtol=rtol,
                           atol=rtol*abs(Kww_ref).max())
        if at_centroid:
            assert abs(hourglass @ Kww @ hourglass) < 1e-12*abs(Kww).max()
        else:
            assert abs(hourglass @ Kww @ hourglass) > 1e-2*abs(Kww).max()

    KGr = np.zeros(data.KG_SPARSE_SIZE, dtype=INT)
    KGc = np.zeros_like(KGr)
    KGv = np.zeros(data.KG_SPARSE_SIZE, dtype=DOUBLE)
    quad.update_KG_given_stress(Nxx, Nyy, Nxy, KGr, KGc, KGv)
    KG = coo_matrix((KGv, (KGr, KGc)), shape=(4*DOF, 4*DOF)).toarray()
    check(KG[2::DOF, 2::DOF], 1e-12)

    # NOTE uniform membrane strains, in global axes, giving Nmat
    A = np.array([[prop.A11, prop.A12, prop.A16],
                  [prop.A12, prop.A22, prop.A26],
                  [prop.A16, prop.A26, prop.A66]])
    eps = np.linalg.solve(A, [Nmat[0, 0], Nmat[1, 1], Nmat[0, 1]])
    u = np.zeros(4*DOF)
    u[0::DOF] = eps[0]*xy[:, 0] + eps[2]/2*xy[:, 1]
    u[1::DOF] = eps[2]/2*xy[:, 0] + eps[1]*xy[:, 1]
    quad.update_probe_ue(u)
    KGv[:] = 0
    quad.update_KG(KGr, KGc, KGv, prop)
    KG = coo_matrix((KGv, (KGr, KGc)), shape=(4*DOF, 4*DOF)).toarray()
    check(KG[2::DOF, 2::DOF], 1e-10)


def test_hourglass_pattern_nonlinear_terms():
    r"""In a distorted element with `h/\ell < 1`, the hourglass pattern of
    `w` with zero rotations has no nonlinear membrane strains with the
    slopes at the centroid, ``shear_model = 0`` or ``centroid_slopes = 1``,
    such that its nonlinear internal forces and KCNL vanish; with the
    default, the slopes at the two-by-two points, it has both. With
    ``shear_model = 0``, the one-point transverse shear of Hughes et al.
    (1977), it has no linear internal forces either, so no energy at all;
    with ``shear_model = 1`` the assumed transverse shear strains of the
    MITC4 element give it the linear forces ``KC0 u``"""
    xy = np.array([[0., 0.], [2., 0.], [2.3, 1.4], [0.2, 1.1]])
    for shear_model, centroid_slopes in RULES:
        quad, data, prop, coords = single_element(
            xy, shear_model=shear_model, centroid_slopes=centroid_slopes)
        assert prop.h/quad.area**0.5 < 1
        u = np.zeros(4*DOF)
        u[2::DOF] = 0.1*np.array([1., -1., 1., -1.])
        quad.update_probe_ue(u)
        fint = np.zeros(4*DOF)
        quad.update_fint(fint, prop, nonlinear=1)
        fint_linear = np.zeros(4*DOF)
        quad.update_fint(fint_linear, prop, nonlinear=0)
        KCNLr = np.zeros(data.KCNL_SPARSE_SIZE, dtype=INT)
        KCNLc = np.zeros_like(KCNLr)
        KCNLv = np.zeros(data.KCNL_SPARSE_SIZE, dtype=DOUBLE)
        quad.update_KCNL(KCNLr, KCNLc, KCNLv, prop)
        KC0r = np.zeros(data.KC0_SPARSE_SIZE, dtype=INT)
        KC0c = np.zeros_like(KC0r)
        KC0v = np.zeros(data.KC0_SPARSE_SIZE, dtype=DOUBLE)
        quad.update_KC0(KC0r, KC0c, KC0v, prop)
        KC0 = coo_matrix((KC0v, (KC0r, KC0c)), shape=(4*DOF, 4*DOF))
        # NOTE scale of the internal forces of the same amplitude of a
        #      smooth w pattern, w = x, which stretches the element
        u_ref = np.zeros(4*DOF)
        u_ref[2::DOF] = 0.1*xy[:, 0]
        quad.update_probe_ue(u_ref)
        fint_ref = np.zeros(4*DOF)
        quad.update_fint(fint_ref, prop, nonlinear=1)
        assert abs(fint_linear - KC0 @ u).max() < 1e-12*abs(fint_ref).max()
        if shear_model == 0 or centroid_slopes:
            assert abs(fint - fint_linear).max() < 1e-12*abs(fint_ref).max()
            assert abs(KCNLv).max() < 1e-12*abs(KC0v).max()
        else:
            assert abs(fint - fint_linear).max() > 1e-6*abs(fint_ref).max()
            assert abs(KCNLv).max() > 1e-6*abs(KC0v).max()
        if shear_model == 0:
            assert abs(fint).max() < 1e-12*abs(fint_ref).max()
        else:
            assert abs(fint).max() > 1e-3*abs(fint_ref).max()


def test_tangent_is_the_jacobian_of_the_internal_forces():
    r"""``KC0 + KCNL + KG`` against central differences of the nonlinear
    internal forces of a distorted element, for `h/\ell < 1` and `h/\ell
    \ge 1` and the three rules of the slopes; the difference decreases as
    `\epsilon^2`, as it does only when the tangent is the exact Jacobian"""
    xy = np.array([[0., 0.], [2., 0.], [2.3, 1.4], [0.2, 1.1]])
    rng = np.random.default_rng(0)
    for h, (shear_model, centroid_slopes) in [(h, rule) for h in (0.01, 2.)
                                              for rule in RULES]:
        quad, data, prop, coords = single_element(xy, h, shear_model,
                                                  centroid_slopes)
        assert (prop.h/quad.area**0.5 < 1) == (h < 1)
        u = 0.05*rng.standard_normal(4*DOF)
        d = rng.standard_normal(4*DOF)

        def fint(u):
            quad.update_probe_ue(u)
            f = np.zeros(4*DOF)
            quad.update_fint(f, prop, nonlinear=1)
            return f

        def matrix(update, size):
            r = np.zeros(size, dtype=INT)
            c = np.zeros_like(r)
            v = np.zeros(size, dtype=DOUBLE)
            update(r, c, v)
            return coo_matrix((v, (r, c)), shape=(4*DOF, 4*DOF)).toarray()

        quad.update_probe_ue(u)
        KT = (matrix(lambda r, c, v: quad.update_KC0(r, c, v, prop),
                     data.KC0_SPARSE_SIZE)
              + matrix(lambda r, c, v: quad.update_KCNL(r, c, v, prop),
                       data.KCNL_SPARSE_SIZE)
              + matrix(lambda r, c, v: quad.update_KG(r, c, v, prop),
                       data.KG_SPARSE_SIZE))
        # NOTE relative difference 2.7e-5, 2.7e-7, 2.7e-9 and 2.7e-11 with
        #      eps = 1e-2, 1e-3, 1e-4 and 1e-5, decreasing as eps**2; an
        #      inexact tangent leaves a difference independent of eps
        errs = []
        for eps in (1e-3, 1e-4):
            fd = (fint(u + eps*d) - fint(u - eps*d))/(2*eps)
            errs.append(np.linalg.norm(fd - KT @ d)/np.linalg.norm(KT @ d))
        assert errs[1] < 1e-7
        assert 80 < errs[0]/errs[1] < 120


if __name__ == '__main__':
    test_low_transverse_shear_stiffness()
    test_KG_of_the_pre_buckling_state()
    test_KG_slope_rule()
    test_hourglass_pattern_nonlinear_terms()
    test_tangent_is_the_jacobian_of_the_internal_forces()
