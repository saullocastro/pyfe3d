r"""Verification of the transverse shear stiffness and of how elements use it

The transverse shear stiffness `A_{ts} = [[A_{44}, A_{45}], [A_{45},
A_{55}]]` is a property of the laminate alone. Its equilibrium-based form is
computed by
:meth:`pyfe3d.shellprop.ShellProp.calc_transverse_shear_stiffness` following

    Rohwer, K., 1988, "Improved transverse shear stiffness for layered
    finite elements", DFVLR-FB 88-32.

    Vlachoutsis, S., 1992, "Shear correction factors for plates and
    shells", International Journal for Numerical Methods in Engineering,
    33(7), pp. 1537-1552. https://doi.org/10.1002/nme.1620330712

There is no element size anywhere in that formula, so `A_{ts}` must not
depend on the mesh, and an element that uses it correctly must reproduce a
constant transverse shear state exactly at any element size. These tests
check both halves.

"""
import sys
sys.path.append('..')

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

from pyfe3d.shellprop_utils import isotropic_plate, laminated_plate
from pyfe3d import (Quad4, Quad4Data, Quad4Probe, Quad4R, Quad4RData,
                    Quad4RProbe, Tria3R, Tria3RData, Tria3RProbe,
                    INT, DOUBLE, DOF)

E_ISO = 210.e9
NU_ISO = 0.3
# NOTE laminate of cylinder Z22, Table 4 of Castro et al. (2014)
LAMINAPROP = (123.55e9, 8.708e9, 0.319, 5.595e9, 5.595e9, 5.595e9)
STACK = [49, -49, 36, -36, 0, 0]
PLYT = 0.125e-3


def test_homogeneous_plate_gives_five_sixths():
    r"""The equilibrium approach must give `k = 5/6` for a single ply

    For a homogeneous plate the transverse shear stress of the equilibrium
    solution is the parabola `\tau(z) = 3Q/(2h)(1 - 4z^2/h^2)`, whose
    complementary energy gives `A_{55} = 5/6 G h` exactly. That is a pure
    number, so it must come out the same at every thickness.

    """
    for h in (1.e-3, 1.e-2, 0.1, 1.0):
        prop = isotropic_plate(thickness=h, E=E_ISO, nu=NU_ISO)
        G = E_ISO/(2.*(1. + NU_ISO))
        print('h = %-8g k13 = %.12f  k23 = %.12f  A55/(G*h) = %.12f'
              % (h, prop.scf_k13, prop.scf_k23, prop.A55/(G*h)))
        assert np.isclose(prop.scf_k13, 5./6., rtol=1e-12)
        assert np.isclose(prop.scf_k23, 5./6., rtol=1e-12)
        assert np.isclose(prop.A55, 5./6.*G*h, rtol=1e-12)
        assert np.isclose(prop.A44, 5./6.*G*h, rtol=1e-12)


def test_stiffness_is_invariant_to_ply_subdivision():
    r"""Splitting a ply into identical sub-plies is the same laminate

    The integrand of the complementary energy is a polynomial of degree 4 in
    `z` within each ply, so the 3-point Gauss-Legendre rule used per ply is
    exact and subdividing must change nothing. This is the check that the
    through-thickness integration is exact and not merely accurate.

    """
    base = laminated_plate(stack=[0, 45, -45, 90], plyt=0.25e-3,
                           laminaprop=LAMINAPROP)
    for n in (2, 4, 8):
        stack = []
        for angle in [0, 45, -45, 90]:
            stack += [angle]*n
        prop = laminated_plate(stack=stack, plyt=0.25e-3/n,
                               laminaprop=LAMINAPROP)
        print('n = %d  A44 %.10e  A55 %.10e' % (n, prop.A44, prop.A55))
        assert np.isclose(prop.A44, base.A44, rtol=1e-12)
        assert np.isclose(prop.A55, base.A55, rtol=1e-12)


def test_stiffness_is_invariant_to_the_offset():
    r"""Documented property of the equilibrium approach

    The tractions vanish at both faces whatever the reference surface is,
    so the recovered distribution, and hence `A_{ts}`, cannot depend on the
    offset.

    """
    base = laminated_plate(stack=[0, 45, -45, 90], plyt=0.25e-3,
                           laminaprop=LAMINAPROP)
    for offset in (1.e-4, 5.e-4, -3.e-4):
        prop = laminated_plate(stack=[0, 45, -45, 90], plyt=0.25e-3,
                               laminaprop=LAMINAPROP, offset=offset)
        assert np.isclose(prop.A44, base.A44, rtol=1e-12)
        assert np.isclose(prop.A55, base.A55, rtol=1e-12)


def shear_energy(cls, Data, Probe, L, h, triangle=False):
    r"""Energy of `w = cx` with every rotation and in-plane DOF at zero

    That is a state of constant transverse shear, `\gamma_{xz} = c` and
    `\gamma_{yz} = 0`, with zero curvature because the rotations are
    constant, and zero membrane strain, so the whole energy is

    .. math::
        U = \frac{1}{2} A_{55} c^2 A

    Returned as the ratio of computed to exact.

    """
    prop = isotropic_plate(thickness=h, E=E_ISO, nu=NU_ISO)
    if triangle:
        ncoords = np.array([[0., 0., 0.], [L, 0., 0.], [L, L, 0.]])
    else:
        ncoords = np.array([[0., 0., 0.], [L, 0., 0.], [L, L, 0.],
                            [0., L, 0.]])
    nn = len(ncoords)
    flat = ncoords.flatten()
    data = Data()
    el = cls(Probe())
    for k in range(nn):
        setattr(el, 'n%d' % (k + 1), k + 1)
        setattr(el, 'c%d' % (k + 1), DOF*k)
    el.init_k_KC0 = 0
    N = DOF*nn
    KC0r = np.zeros(data.KC0_SPARSE_SIZE, dtype=INT)
    KC0c = np.zeros(data.KC0_SPARSE_SIZE, dtype=INT)
    KC0v = np.zeros(data.KC0_SPARSE_SIZE, dtype=DOUBLE)
    el.update_rotation_matrix(flat, 0., 0., 1.)
    el.update_probe_xe(flat)
    el.update_area()
    el.update_KC0(KC0r, KC0c, KC0v, prop)
    K = coo_matrix((KC0v, (KC0r, KC0c)), shape=(N, N)).toarray()
    c = 1.e-4
    ue = np.zeros(N)
    ue[2::DOF] = c*ncoords[:, 0]
    return (0.5*ue.dot(K.dot(ue)))/(0.5*prop.A55*c**2*el.area)


SIZES = [(1.0, 0.01), (0.1, 0.01), (0.01, 0.01), (0.005, 0.01),
         (0.01, 0.1), (100., 0.01)]


def test_quads_reproduce_a_constant_transverse_shear_state():
    r"""Quad4 and Quad4R use `A_{ts}` with no element-size dependence

    The sizes span `h/\sqrt{A}` from `10^{-4}` to `10`, which crosses the
    ``prop.h/length >= 1`` switch of :mod:`pyfe3d.quad4` that selects the
    quadrature of the transverse shear gradient term, so this also checks
    that the switch does not spoil the state on either side of it.

    """
    for label, cls, Data, Probe in [('Quad4', Quad4, Quad4Data, Quad4Probe),
                                    ('Quad4R', Quad4R, Quad4RData,
                                     Quad4RProbe)]:
        for L, h in SIZES:
            ratio = shear_energy(cls, Data, Probe, L, h)
            print('%-7s L = %-8g h = %-6g h/L = %-9.4g U/exact = %.15f'
                  % (label, L, h, h/L, ratio))
            assert np.isclose(ratio, 1., rtol=1e-12), (label, L, h, ratio)


def test_tria3r_scales_the_shear_stiffness_with_the_element_size():
    r"""Tria3R does not reproduce the state, by construction

    Its ``alpha_shear_locking`` stabilisation divides `A_{44}`, `A_{45}` and
    `A_{55}` by `1 + \alpha \ell^2/h^2` with `\ell` the longest edge, so the
    energy of a constant transverse shear state comes out too small by
    exactly that factor. The stabilisation is the Stenberg-type scheme of
    Bischoff and Bletzinger (2004) as modified by Castro et al. (2019), and
    in the thin limit the transverse shear energy is asymptotically
    negligible, which is what justifies it there. It is not negligible when
    transverse shear carries load, which is measured for a cylinder in
    ``test_quad4_linear_buckling_cylinder_displ.py``.

    This test pins the factor down so that any future change to the
    stabilisation is visible, and it is the reason Tria3R is excluded from
    :func:`test_quads_reproduce_a_constant_transverse_shear_state`.

    """
    alpha = 0.7
    for L, h in SIZES:
        ratio = shear_energy(Tria3R, Tria3RData, Tria3RProbe, L, h,
                             triangle=True)
        # NOTE the right triangle with legs L has its longest edge on the
        #      diagonal, so maxl = L*sqrt(2)
        maxl = L*2.**0.5
        expected = 1./(1. + alpha*maxl**2/h**2)
        print('Tria3R  L = %-8g h = %-6g U/exact = %.6e  1/(1+factor) = %.6e'
              % (L, h, ratio, expected))
        assert np.isclose(ratio, expected, rtol=1e-10), (L, h, ratio)


def test_rohwer_differs_from_the_constant_factor_on_a_real_laminate():
    r"""What the equilibrium approach buys over `k = 5/6`

    On the laminate of cylinder Z22 the equilibrium approach gives
    `k_{13} = 0.721` and `k_{23} = 0.726`, so it is noticeably more
    compliant than `5/6`, and it produces a non-zero `A_{45}` that neither
    the constant factor nor the Vlachoutsis factors can represent, both of
    them leaving `A_{45} = 0` because `\bar{A}_{45} = 0` for this stacking.

    """
    got = {}
    for mode in ('rohwer', 'vlachoutsis', 'constant', None):
        prop = laminated_plate(stack=STACK, plyt=PLYT,
                               laminaprop=LAMINAPROP, shear_correction=mode)
        got[mode] = (prop.A44, prop.A45, prop.A55)
        print('%-12s A44 %.6e  A45 %+.6e  A55 %.6e'
              % (mode, prop.A44, prop.A45, prop.A55))
    assert np.isclose(got['rohwer'][0]/got[None][0], 0.7262, rtol=1e-3)
    assert np.isclose(got['rohwer'][2]/got[None][2], 0.7208, rtol=1e-3)
    # NOTE only the equilibrium approach couples the two directions here
    assert abs(got['rohwer'][1]) > 1.e-4*got['rohwer'][0]
    assert got['vlachoutsis'][1] == 0.
    assert got['constant'][1] == 0.
    for i in (0, 2):
        assert np.isclose(got['constant'][i], 5./6.*got[None][i], rtol=1e-12)




def patch_stiffness(cls, Data, Probe, n, L, h, triangles=False):
    r"""Assembled `K_{C_0}` of an `n` by `n` patch of side `L`"""
    prop = isotropic_plate(thickness=h, E=E_ISO, nu=NU_ISO)
    xs = np.linspace(0., L, n + 1)
    X, Y = np.meshgrid(xs, xs, indexing='ij')
    nid = np.arange((n + 1)**2).reshape(n + 1, n + 1)
    X = X.ravel()
    Y = Y.ravel()
    nn = len(X)
    flat = np.vstack((X, Y, np.zeros(nn))).T.flatten()
    cells = []
    for i in range(n):
        for j in range(n):
            q = (nid[i, j], nid[i+1, j], nid[i+1, j+1], nid[i, j+1])
            if triangles:
                cells += [(q[0], q[1], q[2]), (q[0], q[2], q[3])]
            else:
                cells.append(q)
    data = Data()
    probe = Probe()
    N = DOF*nn
    ne = len(cells)
    KC0r = np.zeros(data.KC0_SPARSE_SIZE*ne, dtype=INT)
    KC0c = np.zeros(data.KC0_SPARSE_SIZE*ne, dtype=INT)
    KC0v = np.zeros(data.KC0_SPARSE_SIZE*ne, dtype=DOUBLE)
    init_k_KC0 = 0
    for e in cells:
        el = cls(probe)
        for a in range(len(e)):
            setattr(el, 'n%d' % (a + 1), e[a] + 1)
            setattr(el, 'c%d' % (a + 1), DOF*e[a])
        el.init_k_KC0 = init_k_KC0
        el.update_rotation_matrix(flat, 0., 0., 1.)
        el.update_probe_xe(flat)
        el.update_area()
        el.update_KC0(KC0r, KC0c, KC0v, prop)
        init_k_KC0 += data.KC0_SPARSE_SIZE
    K = coo_matrix((KC0v, (KC0r, KC0c)), shape=(N, N)).toarray()
    return K, X, Y, L/n


def shear_only_subspace(cls, Data, Probe, n, L, h, triangles=False):
    r"""Smallest `u^T K u/u^T u` with only the interior `w` free

    With every rotation and in-plane degree-of-freedom held at zero the
    curvature `\kappa = \nabla \theta` vanishes identically, so this
    subspace carries no bending or membrane energy and is resisted by
    transverse shear alone.

    """
    K, X, Y, el_len = patch_stiffness(cls, Data, Probe, n, L, h, triangles)
    N = K.shape[0]
    free = np.zeros(N, dtype=bool)
    free[2::DOF] = True
    edge = (np.isclose(X, 0.) | np.isclose(X, L) | np.isclose(Y, 0.)
            | np.isclose(Y, L))
    free[2::DOF] &= ~edge
    Kw = K[np.ix_(free, free)]
    return np.linalg.eigvalsh(Kw).min(), el_len


def test_shear_only_subspace_is_the_mechanism_the_stabilisation_opens():
    r"""Where the price of ``alpha_shear_locking`` is actually paid

    A Donnell-type geometric stiffness matrix works on
    `\partial w/\partial x`, which is precisely the subspace isolated by
    :func:`shear_only_subspace`, so an element that is soft there gives poor
    thin-shell buckling loads however well it does on plate bending. That is
    measured for a cylinder in
    ``test_quad4_linear_buckling_cylinder_displ.py``.

    Quad4 and Quad4R agree with each other here, and Tria3R is softer by
    exactly its stabilisation factor.

    """
    n = 8
    L = 0.1
    for h in (0.01, 0.002, 0.0005):
        ref, el_len = shear_only_subspace(Quad4R, Quad4RData, Quad4RProbe,
                                          n, L, h)
        q4, _ = shear_only_subspace(Quad4, Quad4Data, Quad4Probe, n, L, h)
        t3, _ = shear_only_subspace(Tria3R, Tria3RData, Tria3RProbe, n, L, h,
                                    triangles=True)
        maxl = el_len*2.**0.5
        expected = 1./(1. + 0.7*maxl**2/h**2)
        print('l/h = %-7.4g Quad4R %.6e  Quad4/Quad4R %.6f  '
              'Tria3R/Quad4R %.4e  1/(1+factor) %.4e'
              % (el_len/h, ref, q4/ref, t3/ref, expected))
        # NOTE the two quadrilaterals use Ats as given, so they agree
        assert np.isclose(q4, ref, rtol=1e-5)
        # NOTE and Tria3R is softened by its factor, to within the
        #      difference between a quadrilateral and two triangles
        assert np.isclose(t3/ref, expected, rtol=0.2), (h, t3/ref, expected)


def cylindrical_bending_strip(cls, Data, Probe, L, b, h, nx, ny,
                              triangles=False, alpha=None):
    r"""Tip deflection of a strip constrained to cylindrical bending

    `v = r_x = r_z = u = 0` at every node and a unit transverse force spread
    over the tip edge, which makes the problem exactly a Timoshenko beam of
    bending stiffness `D_{11} b` and shear stiffness `A_{55} b`, so

    .. math::
        \delta = \frac{P L^3}{3 D_{11} b} + \frac{P L}{A_{55} b}

    Returns the computed tip deflection and the two exact contributions.

    """
    prop = isotropic_plate(thickness=h, E=E_ISO, nu=NU_ISO)
    xs = np.linspace(0., L, nx + 1)
    ys = np.linspace(0., b, ny + 1)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    nid = np.arange((nx + 1)*(ny + 1)).reshape(nx + 1, ny + 1)
    X = X.ravel()
    Y = Y.ravel()
    nn = len(X)
    flat = np.vstack((X, Y, np.zeros(nn))).T.flatten()
    cells = []
    for i in range(nx):
        for j in range(ny):
            q = (nid[i, j], nid[i+1, j], nid[i+1, j+1], nid[i, j+1])
            if triangles:
                cells += [(q[0], q[1], q[2]), (q[0], q[2], q[3])]
            else:
                cells.append(q)
    data = Data()
    probe = Probe()
    N = DOF*nn
    ne = len(cells)
    KC0r = np.zeros(data.KC0_SPARSE_SIZE*ne, dtype=INT)
    KC0c = np.zeros(data.KC0_SPARSE_SIZE*ne, dtype=INT)
    KC0v = np.zeros(data.KC0_SPARSE_SIZE*ne, dtype=DOUBLE)
    init_k_KC0 = 0
    for e in cells:
        el = cls(probe)
        for a in range(len(e)):
            setattr(el, 'n%d' % (a + 1), e[a] + 1)
            setattr(el, 'c%d' % (a + 1), DOF*e[a])
        el.init_k_KC0 = init_k_KC0
        if alpha is not None:
            el.alpha_shear_locking = alpha
        el.update_rotation_matrix(flat, 0., 0., 1.)
        el.update_probe_xe(flat)
        el.update_area()
        el.update_KC0(KC0r, KC0c, KC0v, prop)
        init_k_KC0 += data.KC0_SPARSE_SIZE
    K = coo_matrix((KC0v, (KC0r, KC0c)), shape=(N, N)).tocsc()
    bk = np.zeros(N, dtype=bool)
    root = np.isclose(X, 0.)
    for d in range(DOF):
        bk[d::DOF] |= root
    bk[0::DOF] = True
    bk[1::DOF] = True
    bk[3::DOF] = True
    bk[5::DOF] = True
    bu = ~bk
    f = np.zeros(N)
    tip = np.where(np.isclose(X, L))[0]
    f[DOF*tip + 2] = 1./len(tip)
    u = np.zeros(N)
    u[bu] = spsolve(K[bu, :][:, bu], f[bu])
    return (u[DOF*tip + 2].mean(), L**3/(3.*prop.D11*b), L/(prop.A55*b))


def test_the_stabilisation_error_is_a_discretisation_error():
    r"""Why the scheme is legitimate despite the factor being large

    The stabilisation inflates the shear part of a load-driven response by
    ``factor``, so the relative error is `factor \times f_s` with `f_s` the
    shear fraction. For a strip of span `L` cut into `n` elements,
    `factor \sim \alpha (L/n)^2/h^2` while `f_s \sim c (h/L)^2`, so the
    thickness cancels and the product is `\alpha c/n^2`. This test measures
    that product over three decades of slenderness and shows it depends on
    `n` alone, falling as `1/n^2`.

    """
    h = 0.01
    alpha = 0.7
    products = {}
    for L_over_h in (5., 20., 100.):
        L = L_over_h*h
        for nx in (8, 32):
            b = L/nx
            _, eb, es = cylindrical_bending_strip(Quad4, Quad4Data,
                                                  Quad4Probe, L, b, h, nx, 1)
            fs = es/(eb + es)
            maxl = ((L/nx)**2 + b**2)**0.5
            factor = alpha*maxl**2/h**2
            products[(L_over_h, nx)] = factor*fs
            print('L/h = %-6g n = %-3d shear fraction %.4e  factor %.4g  '
                  'product %.5f' % (L_over_h, nx, fs, factor, factor*fs))
    for nx in (8, 32):
        vals = [products[(s, nx)] for s in (5., 20., 100.)]
        spread = (max(vals) - min(vals))/np.mean(vals)
        print('n = %-3d product spread over the three thicknesses: %.3f'
              % (nx, spread))
        # NOTE the same number whatever the thickness, to within the
        #      approximation f_s ~ c (h/L)**2 which is exact only for L >> h
        assert spread < 0.05, (nx, vals)
    ratio = np.mean([products[(s, 8)] for s in (5., 20., 100.)])/np.mean(
        [products[(s, 32)] for s in (5., 20., 100.)])
    print('product at n = 8 over product at n = 32: %.2f, expected 16' % ratio)
    assert np.isclose(ratio, 16., rtol=0.05)


def test_small_alpha_makes_tria3r_lock():
    r"""``alpha_shear_locking`` is doing the unlocking, not fine tuning

    In Lyly et al. (1993), Bischoff and Bletzinger (2004) and Castro et al.
    (2019) the factor is applied on top of a discrete shear gap formulation,
    which is already free of shear locking, and `\alpha` is near `0.1`. This
    element has no such formulation, its transverse shear coming from a
    single point at the centroid, so reducing `\alpha` towards the
    literature value makes it lock. Measured here on the constrained strip,
    whose exact answer is elementary.

    """
    h = 0.01
    L = 100.*h
    nx = 16
    b = L/nx
    ratios = []
    for alpha in (0.7, 0.1, 0.01):
        d, eb, es = cylindrical_bending_strip(Tria3R, Tria3RData,
                                             Tria3RProbe, L, b, h, nx, 1,
                                             triangles=True, alpha=alpha)
        ratios.append(d/(eb + es))
        print('alpha = %-6g Tria3R tip deflection / exact = %.6f'
              % (alpha, d/(eb + es)))
    quad, eb, es = cylindrical_bending_strip(Quad4, Quad4Data, Quad4Probe,
                                             L, b, h, nx, 1)
    print('Quad4 tip deflection / exact = %.6f' % (quad/(eb + es)))
    # NOTE monotonically stiffer as alpha falls, which is shear locking
    assert ratios[0] > ratios[1] > ratios[2]
    # NOTE and the quadrilateral, which needs no stabilisation, is close
    assert abs(quad/(eb + es) - 1.) < 0.01

if __name__ == '__main__':
    test_homogeneous_plate_gives_five_sixths()
    test_stiffness_is_invariant_to_ply_subdivision()
    test_stiffness_is_invariant_to_the_offset()
    test_quads_reproduce_a_constant_transverse_shear_state()
    test_tria3r_scales_the_shear_stiffness_with_the_element_size()
    test_rohwer_differs_from_the_constant_factor_on_a_real_laminate()
    test_shear_only_subspace_is_the_mechanism_the_stabilisation_opens()
    test_the_stabilisation_error_is_a_discretisation_error()
    test_small_alpha_makes_tria3r_lock()
