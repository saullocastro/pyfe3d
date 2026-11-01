r"""Thick Quad4 elements, `h/\ell \ge 1`: twist, bending and rank

Up to version 0.11, :class:`pyfe3d.Quad4` integrated the transverse shear of
the elements with `h/\ell \ge 1` with the "modified one-point shear" rule of

    Hughes, T. J. R., Taylor, R. L. and Kanoknukulchai, W., 1977, "A simple
    and efficient finite element for plate bending", IJNME, 11(10), pp.
    1529-1543, pp. 1541-1542. doi:10.1002/nme.1620111005

two-by-two quadrature for the `(\partial w/\partial x)^2` and `(\partial
w/\partial y)^2` terms and one point for the rest. In a state of pure twist,
`w = \kappa x y`, `r_x = \kappa x`, `r_y = -\kappa y`, the transverse shear
strains `\gamma_{xz} = w_{,x} + r_y` and `\gamma_{yz} = w_{,y} - r_x` vanish
everywhere, but the rule integrates `w_{,x}^2 = \kappa^2 y^2` with two-by-two
and the terms that cancel it at the centroid only, storing `A_{55}
\ell^4 \kappa^2/12` in a square element of side `\ell`, i.e. `(5/12)(\ell/h)^2`
times its twist energy for an isotropic plate, 42 % at the gate. Measured
with the old rule:

    h/l       U_twist/U_exact
    0.99         1.00000
    1.00         1.41667
    1.20         1.28935
    2.00         1.10417

A mesh refined across the gate therefore became abruptly stiffer in torsion.
The tip twist (x100) of the isotropic I-beam of Yu et al. (2005), Table 5,
as a cantilever of length 10 under a unit tip torque, meshed with 20 + 20
elements across the flanges and 20 along the web, was 3.005, 2.764 and 2.368
with 100, 150 and 200 elements along the length, the web and then the
flanges crossing `h/\ell = 1`, against 3.003 for Quad4R and 2.987 for
Tria3DSG. With the assumed transverse shear strains of the MITC4 element,

    Dvorkin, E. N. and Bathe, K. J., 1984, "A continuum mechanics based
    four-node shell element for general non-linear analysis", Engineering
    Computations, 1(1), pp. 77-88. doi:10.1108/eb023562

    Bathe, K. J. and Dvorkin, E. N., 1985, "A four-node plate bending element
    based on Mindlin/Reissner plate theory and a mixed interpolation", IJNME,
    21(2), pp. 367-383. doi:10.1002/nme.1620210213

used since 0.12.0 for `h/\ell \ge 1`, the I-beam gives 3.005, 3.005 and
2.999, and the thin elements, `h/\ell < 1`, are unchanged, with the
attribute ``shear_model = 0`` of :class:`pyfe3d.Quad4`. With the default
``shear_model = 1``, also since 0.12.0, the thin elements take the assumed
strains as well and there is no gate; the tests below check both.

"""
import sys
sys.path.append('..')

import numpy as np
import pytest
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

from pyfe3d.shellprop_utils import isotropic_plate
from pyfe3d import Quad4, Quad4Data, Quad4Probe, INT, DOUBLE, DOF

DISTORTED = np.array([[0., 0.], [2., 0.], [2.3, 1.4], [0.2, 1.1]])


def element_K(xy, prop, drilling_model=0, shear_model=1):
    coords = np.zeros((4, 3))
    coords[:, :2] = xy
    cflat = coords.flatten()
    data = Quad4Data()
    probe = Quad4Probe()
    q = Quad4(probe)
    q.drilling_model = drilling_model
    q.shear_model = shear_model
    r = np.zeros(data.KC0_SPARSE_SIZE, dtype=INT)
    c = np.zeros(data.KC0_SPARSE_SIZE, dtype=INT)
    v = np.zeros(data.KC0_SPARSE_SIZE, dtype=DOUBLE)
    q.n1, q.n2, q.n3, q.n4 = 0, 1, 2, 3
    q.c1, q.c2, q.c3, q.c4 = 0, DOF, 2*DOF, 3*DOF
    q.init_k_KC0 = 0
    q.update_rotation_matrix(cflat, 1., 0., 0.)
    q.update_probe_xe(cflat)
    q.update_KC0(r, c, v, prop)
    K = coo_matrix((v, (r, c)), shape=(24, 24)).toarray()
    return 0.5*(K + K.T)


def nodal(xy, f):
    u = np.zeros(24)
    for i, (x, y) in enumerate(xy):
        u[DOF*i:DOF*i + DOF] = f(x, y)
    return u


def test_pure_twist_and_bending_are_exact_on_both_sides_of_the_gate():
    r"""Rectangles centred at the origin, both drilling models

    Pure twist, `w = \kappa x y`, and pure bending in both directions are
    exact for the bilinear fields of a rectangle, and the element must store
    exactly the energy of the plate, `D_{66} (2\kappa)^2/2` and `D
    \kappa^2/2` per unit area, for any `h/\ell` and both values of
    ``shear_model``.

    """
    E, nu = 1., 0.3
    G = E/2/(1 + nu)
    kap = 1.
    for lx, ly in ((1., 1.), (3., 1.)):
        xy = np.array([[-lx/2, -ly/2], [lx/2, -ly/2], [lx/2, ly/2],
                       [-lx/2, ly/2]])
        area = lx*ly
        twist = nodal(xy, lambda x, y: (0, 0, kap*x*y, kap*x, -kap*y, 0))
        bendx = nodal(xy, lambda x, y: (0, 0, -kap*x**2/2, 0, kap*x, 0))
        bendy = nodal(xy, lambda x, y: (0, 0, -kap*y**2/2, -kap*y, 0, 0))
        ell = np.sqrt(area)
        for hl in (0.5, 0.99, 1., 1.2, 2., 10.):
            h = hl*ell
            prop = isotropic_plate(thickness=h, E=E, nu=nu)
            D = E*h**3/12/(1 - nu**2)
            D66 = G*h**3/12
            for drilling_model, shear_model in ((0, 1), (1, 1), (0, 0),
                                                (1, 0)):
                K = element_K(xy, prop, drilling_model, shear_model)
                Ut = 0.5*twist@K@twist/(0.5*D66*(2*kap)**2*area)
                Ux = 0.5*bendx@K@bendx/(0.5*D*kap**2*area)
                Uy = 0.5*bendy@K@bendy/(0.5*D*kap**2*area)
                print(lx, ly, hl, drilling_model, shear_model, Ut, Ux, Uy)
                assert abs(Ut - 1) < 1.e-10, (lx, hl, shear_model, Ut)
                assert abs(Ux - 1) < 1.e-10, (lx, hl, shear_model, Ux)
                assert abs(Uy - 1) < 1.e-10, (lx, hl, shear_model, Uy)


def test_constant_transverse_shear_on_a_distorted_element():
    r"""`w = a x + b y` with zero rotations, a constant transverse shear

    The energy is `(A_{55} a^2 + A_{44} b^2) A/2` for both rules, the
    one-point rule of the thin elements with ``shear_model = 0`` and the
    assumed strains, also on a distorted element.

    """
    a, b = 0.3, -0.7
    xy = DISTORTED
    x, y = xy[:, 0], xy[:, 1]
    area = 0.5*abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    ell = np.sqrt(area)
    u = nodal(xy, lambda x, y: (0, 0, a*x + b*y, 0, 0, 0))
    for hl in (0.1, 0.5, 1., 2.):
        prop = isotropic_plate(thickness=hl*ell, E=1., nu=0.3)
        for shear_model in (0, 1):
            K = element_K(xy, prop, shear_model=shear_model)
            U = 0.5*u@K@u
            Uref = 0.5*(prop.A55*a**2 + prop.A44*b**2)*area
            print(hl, shear_model, U/Uref)
            assert abs(U/Uref - 1) < 1.e-10, (hl, shear_model, U/Uref)


def test_rank_of_the_thick_element():
    r"""Only the six rigid-body modes for `h/\ell \ge 1`, and for any
    `h/\ell` with the default ``shear_model = 1``

    The one-point rule of the thin elements with ``shear_model = 0`` leaves
    the two zero-energy modes of Hughes et al. (1977), see
    ``tests/test_quad4_spurious_shear_mode.py``.

    """
    square = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
    rectangle = np.array([[0., 0.], [3., 0.], [3., 1.], [0., 1.]])
    for xy in (square, rectangle, DISTORTED):
        x, y = xy[:, 0], xy[:, 1]
        area = 0.5*abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
        for hl, expected in ((0.5, 8), (1., 6), (2., 6)):
            prop = isotropic_plate(thickness=hl*np.sqrt(area), E=1., nu=0.3)
            for drilling_model in (0, 1):
                K = element_K(xy, prop, drilling_model, shear_model=0)
                ev = np.linalg.eigvalsh(K)
                nz = np.sum(np.abs(ev) < 1.e-10*ev.max())
                print(hl, drilling_model, nz)
                if hl >= 1.:
                    assert nz == expected, (hl, drilling_model, nz)
                else:
                    assert nz >= 7, (hl, drilling_model, nz)
                # NOTE the default, MITC4 at any h/l
                K = element_K(xy, prop, drilling_model)
                ev = np.linalg.eigvalsh(K)
                nz = np.sum(np.abs(ev) < 1.e-10*ev.max())
                assert nz == 6, (hl, drilling_model, nz)


def strip_twist(nx, ny, Lx=2., b=0.5, h=0.04, E=10.e6, nu=0.3,
                shear_model=1):
    r"""Tip twist of a strip clamped at `x = 0` under a unit end torque"""
    prop = isotropic_plate(thickness=h, E=E, nu=nu)
    xs = np.linspace(0., Lx, nx + 1)
    ys = np.linspace(-b/2, b/2, ny + 1)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    ncoords = np.column_stack((X.ravel(), Y.ravel(), np.zeros(X.size)))
    cflat = ncoords.flatten()
    ids = np.arange(X.size).reshape(nx + 1, ny + 1)
    data = Quad4Data()
    probe = Quad4Probe()
    ne = nx*ny
    r = np.zeros(data.KC0_SPARSE_SIZE*ne, dtype=INT)
    c = np.zeros(data.KC0_SPARSE_SIZE*ne, dtype=INT)
    v = np.zeros(data.KC0_SPARSE_SIZE*ne, dtype=DOUBLE)
    k = 0
    for i in range(nx):
        for j in range(ny):
            q = Quad4(probe)
            q.shear_model = shear_model
            q.n1, q.n2, q.n3, q.n4 = (ids[i, j], ids[i + 1, j],
                                      ids[i + 1, j + 1], ids[i, j + 1])
            q.c1, q.c2, q.c3, q.c4 = [DOF*n for n in (q.n1, q.n2, q.n3, q.n4)]
            q.init_k_KC0 = k
            q.update_rotation_matrix(cflat)
            q.update_probe_xe(cflat)
            q.update_KC0(r, c, v, prop)
            k += data.KC0_SPARSE_SIZE
    N = DOF*X.size
    K = coo_matrix((v, (r, c)), shape=(N, N)).tocsc()
    bk = np.zeros(N, dtype=bool)
    for d in range(DOF):
        bk[DOF*ids[0, :] + d] = True
    # NOTE unit torque as normal forces proportional to y on the tip edge
    tip = ids[-1, :]
    w = np.ones(ny + 1)
    w[0] = w[-1] = 0.5
    f = np.zeros(N)
    f[DOF*tip + 2] = w*ys/np.sum(w*ys**2)
    bu = ~bk
    u = np.zeros(N)
    u[bu] = spsolve(K[bu][:, bu], f[bu])
    return np.polyfit(ys, u[DOF*tip + 2], 1)[0]


@pytest.mark.slow
def test_strip_torsion_is_continuous_across_the_gate():
    r"""Clamped strip 2 x 0.5, h = 0.04, square elements, unit end torque

    Tip twist with 12 elements across the width, `h/\ell = 0.96`, and with
    14, `h/\ell = 1.12`: 0.048122 and 0.048278 with the default
    ``shear_model = 1``, the assumed strains of the MITC4 element on both
    meshes; 0.048880 and 0.048278 with ``shear_model = 0``, one point for
    the thin elements; 0.048880 and 0.036544 with the modified one-point
    shear rule of Hughes et al. (1977) above the gate, as up to 0.11, whose
    spurious twist stiffness made the finer mesh 25 % stiffer. Continuity
    across the gate is asserted within 2 % for both values of
    ``shear_model``, the grade A of MacNeal and Harder (1985), see below;
    the old rule fails it by 25 %.

    The reference is the Saint-Venant twist rate `\theta' = T/(G J)` of the
    `b \times h` rectangle, `b/h = 12.5`, with the series of

        Timoshenko, S. P. and Goodier, J. N., Theory of Elasticity, 2nd ed.,
        McGraw-Hill, 1951, Art. 95, "Torsion of rectangular bars", p. 278,
        Eq. (160), with `2a = h` and `2b = b`,

    `J = (b h^3/3)(1 - (192/\pi^5)(h/b) \sum_{n=1,3,5...} \tanh(n \pi b/2h)
    /n^5) = 0.94958 b h^3/3`; for a narrow rectangle it becomes `(b h^3/3)(1
    - 0.630 h/b)`, Eq. (161), the correction of the `b h^3/3` of Art. 84,
    p. 273, Eq. (155), due to the stress free short edges. The Reissner-Mindlin plate reproduces it through the
    boundary layer of the transverse shear at the free edges: with `w =
    \theta' x y`, `r_y = -\theta' f(y)`, `r_x = \theta' x`, the equilibrium
    `f'' = \lambda^2 (f - y)`, `\lambda^2 = 12 k/h^2`, and the free edges,
    `f'(\pm b/2) = -1`, give `J = (b h^3/3)(1 - (2/\lambda b) \tanh(\lambda
    b/2)) = 0.94940 b h^3/3` for `k = 5/6`, 0.02 % from Saint-Venant, against
    `b h^3/3` for the Kirchhoff plate, 5.3 % stiffer.

    The tip twist itself also contains the effect of the clamped root, which
    prevents the `w_{,x} = \theta' y` of the free twist, and of the tip load,
    applied as normal forces proportional to `y`, both decaying within the
    width. They cancel in the difference of the tip twists of the 2 and 4
    long strips: the twist rate from the lengths 1 and 2, 2 and 4, 4 and 6
    agree within 0.003 %, and the clamped root reduces the tip twist of the
    2 long strip by 0.0024, 4.8 %, on all the meshes below.

    `G J \theta'/T` of the twist rate, measured, against Saint-Venant:

        elements across   h/l    G J theta'/T - 1
        12                0.96      +1.45 %  (-0.02 % with shear_model = 0)
        14                1.12      +1.18 %
        24                1.92      +0.50 %
        48                3.84      +0.13 %

    the MITC4 elements converging to the reference; 12 and 14 are asserted
    within 2 %, the grade A of

        MacNeal, R. H. and Harder, R. L., 1985, "A proposed standard set of
        problems to test finite element accuracy", Finite Elements in
        Analysis and Design, 1(1), pp. 3-20, p. 14.
        doi:10.1016/0168-874X(85)90003-4

    """
    E, nu, b, h = 10.e6, 0.3, 0.5, 0.04
    G = E/2/(1 + nu)
    n = np.arange(1, 200, 2)
    J = b*h**3/3*(1 - 192/np.pi**5*h/b
                  *np.sum(np.tanh(n*np.pi*b/(2*h))/n**5))
    rate_ref = 1./(G*J)
    for sm in (1, 0):
        t12 = strip_twist(48, 12, shear_model=sm)
        t14 = strip_twist(56, 14, shear_model=sm)
        print(sm, t12, t14, t14/t12)
        assert abs(t14/t12 - 1) < 0.02
        for ny, t2 in ((12, t12), (14, t14)):
            t4 = strip_twist(8*ny, ny, Lx=4., shear_model=sm)
            rate = (t4 - t2)/2.
            print('%d across: twist rate %.6f, Saint-Venant %.6f, J_FE/J %.5f'
                  % (ny, rate, rate_ref, rate_ref/rate))
            assert abs(rate_ref/rate - 1) < 0.02, (ny, rate_ref/rate)
