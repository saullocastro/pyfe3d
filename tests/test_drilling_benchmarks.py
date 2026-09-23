r"""Literature benchmarks for the drilling degree-of-freedom, on Quad4

C1, Cook's skew membrane, is the standard test of a membrane element with
drilling rotations, and the convergence sequence is compared against Table
III of:

    Ibrahimbegovic, A., Taylor, R. L. and Wilson, E. L., 1990, "A robust
    quadrilateral membrane finite element with drilling degrees of freedom",
    IJNME, 30(3), pp. 445-457. doi:10.1002/nme.1620300305

C2 and C3 are the straight and the curved cantilever of:

    MacNeal, R. H. and Harder, R. L., 1985, "A proposed standard set of
    problems to test finite element accuracy", Finite Elements in Analysis and
    Design, 1(1), pp. 3-20. doi:10.1016/0168-874X(85)90003-4

C4 is the cantilever of Section 8.3 and Table 3 of:

    Allman, D. J., 1984, "A compatible triangular element including vertex
    rotations for plane elasticity analysis", Computers & Structures, 19(1-2),
    pp. 1-8. doi:10.1016/0045-7949(84)90197-4

The reference values quoted in the comments were taken from those tables. The
values of the penalty model are recorded next to them because the point of
the comparison is the in-plane bending response, which is where the drilling
rotation carries the load and where the two models differ by an order of
magnitude on a coarse mesh.

A limitation of the element shows up in two of these cases and is asserted
here so that it is on record. With a single element across the width, a
strip of Quad4 elements has almost no torsional stiffness, because of the
zero-energy mode of the transverse shear documented in
``test_quad4_spurious_shear_mode.py``, which is Mode 1 of Figure 10 of Hughes,
Taylor and Kanoknukulchai (1977), the paper the integration scheme of the
element comes from. The twist case of C2 and the out-of-plane case of
C3 both load the strip in torsion and are therefore meaningless at the 6 by 1
mesh that MacNeal and Harder specify, identically for both drilling models.
Two elements across the width are enough to recover them. That mode lives
entirely in the out-of-plane block and has nothing to do with the drilling
rotation.

"""
import sys
sys.path.append('..')

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

from pyfe3d.shellprop_utils import isotropic_plate
from pyfe3d import Quad4, Quad4Data, Quad4Probe, INT, DOUBLE, DOF


def assemble(ncoords, conn, prop, drilling_model, K6ROT=100.):
    ncoords = np.asarray(ncoords, dtype=float)
    N = DOF*ncoords.shape[0]
    ncf = ncoords.flatten()
    data = Quad4Data()
    probe = Quad4Probe()
    ne = len(conn)
    KC0r = np.zeros(data.KC0_SPARSE_SIZE*ne, dtype=INT)
    KC0c = np.zeros(data.KC0_SPARSE_SIZE*ne, dtype=INT)
    KC0v = np.zeros(data.KC0_SPARSE_SIZE*ne, dtype=DOUBLE)
    k = 0
    for (i1, i2, i3, i4) in conn:
        q = Quad4(probe)
        q.n1, q.n2, q.n3, q.n4 = i1 + 1, i2 + 1, i3 + 1, i4 + 1
        q.c1, q.c2, q.c3, q.c4 = DOF*i1, DOF*i2, DOF*i3, DOF*i4
        q.init_k_KC0 = k
        q.drilling_model = drilling_model
        q.K6ROT = K6ROT
        q.update_rotation_matrix(ncf)
        q.update_probe_xe(ncf)
        q.update_area()
        q.update_KC0(KC0r, KC0c, KC0v, prop)
        k += data.KC0_SPARSE_SIZE
    return coo_matrix((KC0v, (KC0r, KC0c)), shape=(N, N)).tocsc(), N


def solve(K, N, bk, fext):
    bu = ~bk
    u = np.zeros(N)
    u[bu] = spsolve(K[bu, :][:, bu], fext[bu])
    return u


def structured_quads(X, Y):
    r"""Nodes and connectivity of a structured grid, counter-clockwise"""
    nx, ny = X.shape[0] - 1, X.shape[1] - 1
    ncoords = np.column_stack((X.ravel(), Y.ravel(), np.zeros(X.size)))
    ids = np.arange(X.size).reshape(nx + 1, ny + 1)
    conn = [(ids[i, j], ids[i + 1, j], ids[i + 1, j + 1], ids[i, j + 1])
            for i in range(nx) for j in range(ny)]
    n = np.cross(ncoords[conn[0][1]] - ncoords[conn[0][0]],
                 ncoords[conn[0][2]] - ncoords[conn[0][1]])
    if n[2] < 0:
        conn = [(a, d, c, b) for (a, b, c, d) in conn]
    return ncoords, ids, conn


# ------------------------------------------------------------------- C1
COOK_CORNERS = np.array([[0., 0.], [48., 44.], [48., 60.], [0., 44.]])
# Table III of Ibrahimbegovic et al. (1990), displacement-type formulation
COOK_ITW_DTYPE = {1: 14.065, 2: 20.682, 4: 22.984, 8: 23.626}
COOK_REF = 23.91


def cook(n, drilling_model):
    s = np.linspace(0., 1., n + 1)
    S, T = np.meshgrid(s, s, indexing='ij')
    P = ((1 - S)[..., None]*(1 - T)[..., None]*COOK_CORNERS[0]
         + S[..., None]*(1 - T)[..., None]*COOK_CORNERS[1]
         + S[..., None]*T[..., None]*COOK_CORNERS[2]
         + (1 - S)[..., None]*T[..., None]*COOK_CORNERS[3])
    ncoords, ids, conn = structured_quads(P[..., 0], P[..., 1])
    prop = isotropic_plate(thickness=1., E=1., nu=1./3.)
    K, N = assemble(ncoords, conn, prop, drilling_model)
    x = ncoords[:, 0]
    y = ncoords[:, 1]
    bk = np.zeros(N, dtype=bool)
    # membrane problem, the out-of-plane degrees-of-freedom are removed
    for d in (2, 3, 4):
        bk[d::DOF] = True
    clamped = np.isclose(x, 0.)
    for d in (0, 1, 5):
        bk[d::DOF] |= clamped
    fext = np.zeros(N)
    loaded = ids[-1, :]
    for k, nid in enumerate(loaded):
        w = 0.5 if k in (0, len(loaded) - 1) else 1.
        fext[DOF*nid + 1] = w/n
    assert np.isclose(fext.sum(), 1.)
    u = solve(K, N, bk, fext)
    ys = y[loaded]
    vs = u[1::DOF][loaded]
    o = np.argsort(ys)
    return np.interp(52., ys[o], vs[o])


def test_C1_cook_skew_membrane():
    r"""Tip displacement at (48, 52), reference 23.91

    Measured, against Table III of Ibrahimbegovic et al. (1990):

        mesh    penalty   enriched   ITW D-type
        1x1      5.9685    13.6808     14.065
        2x2     11.8451    20.7527     20.682
        4x4     18.2991    22.8264     22.984
        8x8     22.0792    23.5420     23.626
       16x16    23.4304    23.8111        -
       32x32    23.8176    23.9082        -

    The remaining gap to the published values comes from the interior bubble
    that Ibrahimbegovic et al. condense out of their element, their Eq. 24,
    which pyfe3d does not have.

    """
    for n in (1, 2, 4, 8):
        p = cook(n, 1)
        e = cook(n, 0)
        print('%2d penalty %.4f enriched %.4f ITW %.3f'
              % (n, p, e, COOK_ITW_DTYPE[n]))
        # within 6% of the published element on every mesh
        assert abs(e/COOK_ITW_DTYPE[n] - 1.) < 0.06, (n, e)
        # and far better than the penalty where it matters, on coarse meshes
        if n <= 2:
            assert e > 1.7*p, (n, e, p)
        else:
            assert e > p, (n, e, p)

    # convergence to the reference
    assert abs(cook(32, 0)/COOK_REF - 1.) < 0.005


# ------------------------------------------------------------------- C2
# Table 3 of MacNeal and Harder (1985), straight cantilever
MH_STRAIGHT_REF = {'extension': 3.0e-5, 'in-plane shear': 0.1081,
                   'out-of-plane shear': 0.4321, 'twist': 0.03208}


def straight_cantilever(shape, load, drilling_model, nx=6, ny=1):
    r"""MacNeal and Harder Fig. 4: L = 6, width 0.2, thickness 0.1

    The interior node lines are slanted at 45 degrees, which for a width of
    0.2 displaces each end of a line by 0.1 from its nominal station. The
    root and the tip stay straight and every element keeps the same area, as
    the figure requires. ``trapezoidal`` alternates the direction of the
    slant and ``parallelogram`` keeps it the same.

    """
    L, w = 6., 0.2
    xs = np.arange(nx + 1)*L/nx
    ys = np.linspace(0., w, ny + 1)
    X = np.empty((nx + 1, ny + 1))
    Y = np.empty((nx + 1, ny + 1))
    for i, x in enumerate(xs):
        if i == 0 or i == nx or shape == 'regular':
            d = 0.
        elif shape == 'parallelogram':
            d = 0.1
        elif shape == 'trapezoidal':
            d = 0.1 if (i % 2) else -0.1
        else:
            raise ValueError(shape)
        # the slant is linear across the width, from -d at y=0 to +d at y=w
        X[i, :] = x + d*(2.*ys/w - 1.)
        Y[i, :] = ys
    ncoords, ids, conn = structured_quads(X, Y)
    prop = isotropic_plate(thickness=0.1, E=1.e7, nu=0.3)
    K, N = assemble(ncoords, conn, prop, drilling_model)
    x = ncoords[:, 0]
    bk = np.zeros(N, dtype=bool)
    root = np.isclose(x, 0.)
    for d in range(DOF):
        bk[d::DOF] |= root
    tip = ids[-1, :]
    fext = np.zeros(N)
    if load == 'twist':
        # a unit torque about x, as a couple on the two extreme tip nodes
        fext[DOF*tip[0] + 2] = -1./w
        fext[DOF*tip[-1] + 2] = +1./w
        u = solve(K, N, bk, fext)
        return (u[DOF*tip[-1] + 2] - u[DOF*tip[0] + 2])/w
    comp = {'extension': 0, 'in-plane shear': 1,
            'out-of-plane shear': 2}[load]
    fext[DOF*tip + comp] = 1./len(tip)
    u = solve(K, N, bk, fext)
    return u[DOF*tip + comp].mean()


def test_C2_macneal_harder_straight_cantilever():
    r"""Tip displacement of the 6 by 1 cantilever, three element shapes

    Measured, with the references of Table 3 in brackets:

        load                  mesh            penalty    enriched     ref
        extension             regular       2.9863e-5   2.9955e-5   3.0e-5
        extension             trapezoidal   2.9872e-5   2.9942e-5
        extension             parallelogram 2.9871e-5   2.9942e-5
        in-plane shear        regular         0.01009     0.09767   0.1081
        in-plane shear        trapezoidal     0.00291     0.08226
        in-plane shear        parallelogram   0.00369     0.09057
        out-of-plane shear    regular         0.42349     0.42349   0.4321
        out-of-plane shear    trapezoidal     0.42410     0.42410
        out-of-plane shear    parallelogram   0.42427     0.42427

    In-plane shear is the drilling-sensitive case, and the one that motivates
    the enrichment: the unenriched element reaches 9%, 3% and 3% of the
    reference on the three meshes, against 90%, 76% and 84% enriched. The
    residual error of the enriched element on the distorted meshes is the
    over-stiffness that Allman's formulation is known to keep.

    """
    for shape in ('regular', 'trapezoidal', 'parallelogram'):
        # extension is a membrane state that the enrichment leaves almost
        # unchanged, and both models must be accurate
        for drilling_model in (0, 1):
            v = straight_cantilever(shape, 'extension', drilling_model)
            assert abs(v/MH_STRAIGHT_REF['extension'] - 1.) < 0.01

        # out-of-plane shear involves no drilling rotation at all, so the two
        # models must give exactly the same answer
        vp = straight_cantilever(shape, 'out-of-plane shear', 1)
        ve = straight_cantilever(shape, 'out-of-plane shear', 0)
        assert np.isclose(vp, ve, rtol=1.e-12)
        assert abs(ve/MH_STRAIGHT_REF['out-of-plane shear'] - 1.) < 0.03

        # in-plane shear, the drilling-sensitive case
        vp = straight_cantilever(shape, 'in-plane shear', 1)
        ve = straight_cantilever(shape, 'in-plane shear', 0)
        print(shape, 'in-plane shear: penalty %.5f enriched %.5f' % (vp, ve))
        assert ve > 0.75*MH_STRAIGHT_REF['in-plane shear']
        assert ve < 1.02*MH_STRAIGHT_REF['in-plane shear']
        assert ve > 8.*vp


def test_C2_twist_needs_two_elements_across_the_width():
    r"""The torsional response of a one-element-wide strip is unusable

    Asserted as a limitation, not as an accuracy result, and it is identical
    for both drilling models. See the module docstring and
    ``test_quad4_spurious_shear_mode.py``.

        elements across width    twist          eigmin/eigmax
        1                        12.106         2.0e-9
        2                         0.03268       6.4e-9
        4                         0.03398       1.5e-9

    """
    ref = MH_STRAIGHT_REF['twist']
    for drilling_model in (0, 1):
        bad = straight_cantilever('regular', 'twist', drilling_model, ny=1)
        assert bad > 100.*ref
    # the two models agree, so the defect is not in the drilling stiffness
    assert np.isclose(straight_cantilever('regular', 'twist', 0, ny=1),
                      straight_cantilever('regular', 'twist', 1, ny=1),
                      rtol=1.e-10)
    for ny in (2, 4):
        for drilling_model in (0, 1):
            good = straight_cantilever('regular', 'twist', drilling_model,
                                       ny=ny)
            print('ny=%d twist %.5f, reference %.5f' % (ny, good, ref))
            assert abs(good/ref - 1.) < 0.07


# ------------------------------------------------------------------- C3
MH_CURVED_REF = {'in-plane': 0.08734, 'out-of-plane': 0.5022}


def curved_beam(load, drilling_model, nx=6, ny=1):
    r"""MacNeal and Harder Fig. 5: 90 degree arc, radii 4.12 and 4.32"""
    ri, ro = 4.12, 4.32
    th = np.linspace(0., np.pi/2., nx + 1)
    rr = np.linspace(ri, ro, ny + 1)
    T, R = np.meshgrid(th, rr, indexing='ij')
    ncoords, ids, conn = structured_quads(R*np.cos(T), R*np.sin(T))
    prop = isotropic_plate(thickness=0.1, E=1.e7, nu=0.25)
    K, N = assemble(ncoords, conn, prop, drilling_model)
    bk = np.zeros(N, dtype=bool)
    root = np.isclose(ncoords[:, 1], 0., atol=1.e-9)
    for d in range(DOF):
        bk[d::DOF] |= root
    tip = ids[-1, :]
    comp = 1 if load == 'in-plane' else 2
    fext = np.zeros(N)
    fext[DOF*tip + comp] = 1./len(tip)
    u = solve(K, N, bk, fext)
    return u[DOF*tip + comp].mean()


def test_C3_macneal_harder_curved_beam():
    r"""Tip displacement of the 6 by 1 curved cantilever

    Measured, references 0.08734 in-plane and 0.5022 out-of-plane:

        elements across width    load           penalty   enriched
        1                        in-plane       0.006414   0.079346
        2                        in-plane       0.006434   0.082802
        4                        in-plane       0.006440   0.083937
        2                        out-of-plane   0.474      0.474
        4                        out-of-plane   0.476      0.476

    In-plane is the drilling-sensitive case: the penalty reaches 7% of the
    reference, the enriched element 91% on the benchmark mesh and 96% with
    four elements across the width. The out-of-plane case loads the strip in
    torsion, so it needs at least two elements across the width for the
    reason given in the module docstring, and it is then identical for both
    models.

    """
    vp = curved_beam('in-plane', 1)
    ve = curved_beam('in-plane', 0)
    print('in-plane: penalty %.6f enriched %.6f' % (vp, ve))
    assert ve > 0.88*MH_CURVED_REF['in-plane']
    assert ve < 1.02*MH_CURVED_REF['in-plane']
    assert ve > 10.*vp
    assert curved_beam('in-plane', 0, ny=4) > 0.95*MH_CURVED_REF['in-plane']

    # out-of-plane, torsion dominated, unusable with one element across
    for drilling_model in (0, 1):
        assert curved_beam('out-of-plane', drilling_model, ny=1) > 10.
    for ny in (2, 4):
        vp = curved_beam('out-of-plane', 1, ny=ny)
        ve = curved_beam('out-of-plane', 0, ny=ny)
        assert np.isclose(vp, ve, rtol=1.e-10)
        print('ny=%d out-of-plane %.5f' % (ny, ve))
        assert abs(ve/MH_CURVED_REF['out-of-plane'] - 1.) < 0.07


# ------------------------------------------------------------------- C4
# Table 3 of Allman (1984), his own triangular element, and Table II of
# Ibrahimbegovic et al. (1990) for the same problem
ALLMAN_TABLE = {(4, 1): 0.2696, (8, 2): 0.3261, (16, 4): 0.3471}
ALLMAN_REF = 0.3558


def allman_cantilever(nx, ny, drilling_model):
    r"""Allman (1984) Fig. 3: L = 48, H = 12, t = 1, E = 30000, nu = 0.25

    The tip load W = 40 is applied as the parabolic shear traction
    `\tau_{xy} = \frac{3W}{2Ht}\left[1 - 4(y/H)^2\right]` of his Eq. 8.4,
    integrated exactly into consistent nodal loads.

    """
    L, H, t, E, nu, W = 48., 12., 1., 30000., 0.25, 40.
    xs = np.linspace(0., L, nx + 1)
    ys = np.linspace(-H/2., H/2., ny + 1)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    ncoords, ids, conn = structured_quads(X, Y)
    prop = isotropic_plate(thickness=t, E=E, nu=nu)
    K, N = assemble(ncoords, conn, prop, drilling_model)
    x = ncoords[:, 0]
    y = ncoords[:, 1]
    bk = np.zeros(N, dtype=bool)
    for d in (2, 3, 4):
        bk[d::DOF] = True
    root = np.isclose(x, 0.)
    for d in (0, 1, 5):
        bk[d::DOF] |= root

    fext = np.zeros(N)
    gp = np.array([-np.sqrt(3./5.), 0., np.sqrt(3./5.)])
    gw = np.array([5./9., 8./9., 5./9.])
    tipcol = ids[-1, :]
    for j in range(ny):
        na, nb = tipcol[j], tipcol[j + 1]
        ya, yb = y[na], y[nb]
        for xi, wq in zip(gp, gw):
            N1 = 0.5*(1. - xi)
            N2 = 0.5*(1. + xi)
            yq = N1*ya + N2*yb
            tau = 3.*W/(2.*H*t)*(1. - 4.*(yq/H)**2)
            jac = (yb - ya)/2.
            fext[DOF*na + 1] += wq*jac*N1*tau*t
            fext[DOF*nb + 1] += wq*jac*N2*tau*t
    assert np.isclose(fext.sum(), W, rtol=1.e-10)

    u = solve(K, N, bk, fext)
    pos = np.where(np.isclose(x, L))[0]
    o = np.argsort(y[pos])
    return np.interp(0., y[pos][o], u[1::DOF][pos][o])


def test_C4_allman_cantilever():
    r"""Tip deflection of Allman's cantilever, reference 0.3558

    Measured, with Allman's own triangular element from his Table 3 and the
    displacement-type element of Ibrahimbegovic et al. Table II:

        mesh    penalty   enriched   Allman tri   ITW D-type
        4x1     0.24242    0.32812     0.2696       0.3445
        8x2     0.31626    0.34733     0.3261       0.3504
       16x4     0.34472    0.35327     0.3471       0.3543
       32x8        -       0.35515
       64x16       -       0.35576

    At mesh III, 16 by 4, the enriched element gives 0.3533 with 255 in-plane
    degrees-of-freedom, against 0.3556 for the six-node linear strain
    triangle with 594 degrees-of-freedom in the same table of Allman. That is
    99.4% of the higher-order result at 43% of the unknowns, which is the
    claim that the enrichment is meant to support.

    """
    for (nx, ny), allman in ALLMAN_TABLE.items():
        p = allman_cantilever(nx, ny, 1)
        e = allman_cantilever(nx, ny, 0)
        print('%dx%d penalty %.5f enriched %.5f Allman %.4f'
              % (nx, ny, p, e, allman))
        # better than the unenriched element, and than Allman's triangles on
        # the same grid
        assert e > p
        assert e > allman
        assert e < 1.01*ALLMAN_REF
    # the coarsest mesh already reaches 92% of the converged value
    assert allman_cantilever(4, 1, 0) > 0.92*ALLMAN_REF
    # and the sequence converges onto it
    assert abs(allman_cantilever(64, 16, 0)/ALLMAN_REF - 1.) < 0.002


if __name__ == '__main__':
    test_C1_cook_skew_membrane()
    test_C2_macneal_harder_straight_cantilever()
    test_C2_twist_needs_two_elements_across_the_width()
    test_C3_macneal_harder_curved_beam()
    test_C4_allman_cantilever()
