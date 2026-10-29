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

The accuracy on the coarse meshes is graded with the rule of MacNeal and
Harder (1985), p. 14, applied to the absolute error normalised by the
reference value: A for an error up to 2 %, B up to 10 %, C up to 20 %, D up to
50 % and F, failing, above 50 %. Their test problems are meant to be solved
on coarse meshes, one element across the depth of the beams, and the grade is
their measure of the accuracy that can be expected there, see
:func:`mh_grade`. The convergence with mesh refinement is checked separately.

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
import pytest
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


def mh_grade(value, ref):
    r"""Letter grade of MacNeal and Harder (1985), p. 14, of ``value``
    against ``ref``: A for an absolute normalised error up to 2 %, B up to
    10 %, C up to 20 %, D up to 50 %, F above"""
    error = abs(value/ref - 1.)
    for grade, limit in zip('ABCD', (0.02, 0.10, 0.20, 0.50)):
        if error <= limit:
            return grade
    return 'F'


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
# Table III of Ibrahimbegovic et al. (1990), p. 454, displacement-type
# formulation
COOK_ITW_DTYPE = {1: 14.065, 2: 20.682, 4: 22.984, 8: 23.626}
# NOTE Ibrahimbegovic et al. (1990), Section 4.4, p. 454: "the reference value
#      23.91, obtained by numerical analysis for a refined model"
COOK_REF = 23.91
# NOTE Felippa, C. A., 2003, "A study of optimal membrane triangles with
#      drilling freedoms", CMAME, 192(16-18), pp. 2125-2168,
#      doi:10.1016/S0045-7825(03)00253-6, Section 9.3, p. 2157: "There is no
#      known analytical solution", the extrapolation of his 64x64 results
#      with Wynn's epsilon algorithm gives 23.956
COOK_FELIPPA = 23.956


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


@pytest.mark.slow
def test_C1_cook_skew_membrane():
    r"""Tip displacement at (48, 52), reference 23.91

    Measured, against Table III of Ibrahimbegovic et al. (1990), p. 454, with
    the grade of MacNeal and Harder (1985), p. 14, against ``COOK_REF`` =
    23.91 in brackets:

        mesh    penalty      enriched     ITW D-type
        1x1      5.9685 (F)  13.6808 (D)  14.065 (D)
        2x2     11.8451 (F)  20.7527 (C)  20.682 (C)
        4x4     18.2991 (D)  22.8264 (B)  22.984 (B)
        8x8     22.0792 (B)  23.5420 (A)  23.626 (A)
       16x16    23.4304      23.8111        -
       32x32    23.8176      23.9082        -
       64x64    23.9245      23.9440        -
      128x128   23.9548      23.9579        -

    The enriched element earns the grade of the displacement-type element of
    Ibrahimbegovic et al. on every mesh, and one or two grades more than the
    penalty. The remaining gap to their values comes from the interior bubble
    that Ibrahimbegovic et al. condense out of their element, their Eq. 24,
    which pyfe3d does not have.

    The reference is not exact. Ibrahimbegovic et al. obtained 23.91 with a
    refined model, Felippa (2003), p. 2157, 23.956 by extrapolation. Both
    models of pyfe3d approach 23.956 monotonically, errors -1.73, -0.60,
    -0.20 and -0.05 % enriched and -7.83, -2.19, -0.58 and -0.13 % penalty
    from 8x8 to 64x64, and at 64x64 they lie between the two published
    values. The Aitken extrapolation of the 32x32, 64x64 and 128x128 results
    gives 23.967 for both models, 0.04 % above the value of Felippa and
    0.24 % above the 23.91 of Ibrahimbegovic et al.

    """
    for n in (1, 2, 4, 8):
        p = cook(n, 1)
        e = cook(n, 0)
        gp = mh_grade(p, COOK_REF)
        ge = mh_grade(e, COOK_REF)
        gitw = mh_grade(COOK_ITW_DTYPE[n], COOK_REF)
        print('%2d penalty %.4f (%s) enriched %.4f (%s) ITW %.3f (%s)'
              % (n, p, gp, e, ge, COOK_ITW_DTYPE[n], gitw))
        # the grade of the published element on every mesh
        assert ge == gitw, (n, e, ge, gitw)
        # and at least one grade better than the penalty
        assert ge < gp, (n, e, ge, p, gp)

    # convergence to the reference
    errors = {0: [], 1: []}
    for n in (8, 16, 32, 64):
        for drilling_model in (0, 1):
            v = cook(n, drilling_model)
            errors[drilling_model].append(abs(v/COOK_FELIPPA - 1.))
            if n == 64:
                # within the spread of the published reference values
                assert COOK_REF <= v <= COOK_FELIPPA, (drilling_model, v)
    for drilling_model in (0, 1):
        print('model %d, errors against %.3f:' % (drilling_model,
                                                  COOK_FELIPPA),
              errors[drilling_model])
        assert np.all(np.diff(errors[drilling_model]) < 0)


# ------------------------------------------------------------------- C2
# Table 3 of MacNeal and Harder (1985), p. 10, straight cantilever
MH_STRAIGHT_REF = {'extension': 3.0e-5, 'in-plane shear': 0.1081,
                   'out-of-plane shear': 0.4321, 'twist': 0.03208}
# Table 8 of MacNeal and Harder (1985), p. 15, normalised in-plane shear tip
# displacement of the MSC/NASTRAN QUAD4 on the 6 by 1 meshes
MH_QUAD4_IN_PLANE_SHEAR = {'regular': 0.904, 'trapezoidal': 0.071,
                           'parallelogram': 0.080}


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

    Measured, with the references of Table 3, p. 10, in brackets:

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
    the enrichment. Normalised by the reference and graded with the rule of
    MacNeal and Harder (1985), p. 14, next to the QUAD4 of MSC/NASTRAN of
    their Table 8, p. 15:

        mesh            penalty     enriched    QUAD4 (Table 8)
        regular         0.093 (F)   0.904 (B)   0.904 (B)
        trapezoidal     0.027 (F)   0.761 (D)   0.071 (F)
        parallelogram   0.034 (F)   0.838 (C)   0.080 (F)

    The unenriched element fails on every mesh, the enriched element passes
    on every mesh and is never worse than the QUAD4 of Table 8. The residual
    error of the enriched element on the distorted meshes is the
    over-stiffness that Allman's formulation is known to keep. On the
    regular mesh refined with the same aspect ratio the enriched element
    converges monotonically from below: 0.904, 0.971, 0.991 and 0.997 with 6
    by 1, 12 by 2, 24 by 4 and 48 by 8 elements, the penalty 0.093, 0.291 and
    0.621 up to 24 by 4.

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
        ref = MH_STRAIGHT_REF['in-plane shear']
        vp = straight_cantilever(shape, 'in-plane shear', 1)
        ve = straight_cantilever(shape, 'in-plane shear', 0)
        gp = mh_grade(vp, ref)
        ge = mh_grade(ve, ref)
        gq = mh_grade(MH_QUAD4_IN_PLANE_SHEAR[shape], 1.)
        print(shape, 'in-plane shear: penalty %.5f (%s) enriched %.5f (%s), '
              'QUAD4 of Table 8 (%s)' % (vp, gp, ve, ge, gq))
        assert gp == 'F'
        assert ge != 'F'
        assert ge <= gq

    # mesh refinement of the regular mesh, in-plane shear
    ref = MH_STRAIGHT_REF['in-plane shear']
    ves = [straight_cantilever('regular', 'in-plane shear', 0, nx=6*k, ny=k)
           for k in (1, 2, 4)]
    print('in-plane shear, regular 6x1, 12x2, 24x4, enriched:',
          np.array(ves)/ref)
    assert np.all(np.diff(ves) > 0)
    assert ves[-1] < ref
    assert mh_grade(ves[-1], ref) == 'A'


def test_C2_twist_needs_two_elements_across_the_width():
    r"""The torsional response of a one-element-wide strip is unusable

    Asserted as a limitation, not as an accuracy result, and it is identical
    for both drilling models. See the module docstring and
    ``test_quad4_spurious_shear_mode.py``.

        elements across width    twist          eigmin/eigmax
        1                        12.106         2.0e-9
        2                         0.03268       6.4e-9
        4                         0.03398       1.5e-9

    The 1.9 % and 5.9 % from the reference 0.03208 with two and four
    elements across are not mesh errors that refinement removes: the strip
    converges to about 1.067 times 0.03208, 0.3 % above the Saint-Venant
    twist, see the table of ``test_torsion_of_a_one_element_wide_strip`` in
    ``test_quad4_spurious_shear_mode.py``. Following the policy of the tests
    whose reference is not reached on the coarse mesh used, the twists are
    compared with the values pre-determined by this same test on
    2026-10-29, 0.032684006858721734 and 0.033975948654968564, with ``rtol =
    1e-6`` for the round-off of the linear solution, whose condition
    number is about 1e9 (eigmin/eigmax above; the same strip assembled in
    ``test_quad4_spurious_shear_mode.py`` agrees within 1e-12), and with grade B of MacNeal and Harder (1985), p. 14, at most
    10 %, against 0.03208 as a bound.

    """
    ref = MH_STRAIGHT_REF['twist']
    predetermined = {2: 0.032684006858721734, 4: 0.033975948654968564}
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
            assert np.isclose(good, predetermined[ny], rtol=1e-6)
            assert abs(good/ref - 1.) <= 0.10


# ------------------------------------------------------------------- C3
# Table 3 of MacNeal and Harder (1985), p. 10, curved beam
MH_CURVED_REF = {'in-plane': 0.08734, 'out-of-plane': 0.5022}
# Table 9 of MacNeal and Harder (1985), p. 15, normalised in-plane tip
# displacement of the MSC/NASTRAN QUAD4
MH_QUAD4_CURVED_IN_PLANE = 0.833


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

    Measured, references 0.08734 in-plane and 0.5022 out-of-plane, Table 3,
    p. 10, of MacNeal and Harder (1985):

        elements across width    load           penalty   enriched
        1                        in-plane       0.006414   0.079346
        2                        in-plane       0.006434   0.082802
        4                        in-plane       0.006440   0.083937
        2                        out-of-plane   0.474      0.474
        4                        out-of-plane   0.476      0.476

    In-plane is the drilling-sensitive case. On the benchmark mesh the
    penalty reaches 0.073 of the reference, grade F of MacNeal and Harder
    (1985), p. 14, the enriched element 0.908, grade B, against 0.833, grade
    C, for the QUAD4 of their Table 9, p. 15. The error of the enriched
    element decreases monotonically with the elements across the width, 9.2,
    5.2 and 3.9 %. Refining along the arc as well, 12 by 2, 24 by 4 and 48 by
    8 elements give 0.991, 1.008 and 1.012 of the reference, which is reached
    from below and then passed. MacNeal and Harder call their value
    theoretical without saying how it was obtained, and here the tip load is
    applied as equal nodal forces, so the 1.2 % is a difference of models
    that is not resolved here, and the refinement in both directions is not
    asserted.

    The out-of-plane case loads the strip in torsion, so it needs at least
    two elements across the width for the reason given in the module
    docstring, and it is then identical for both models.

    """
    ref = MH_CURVED_REF['in-plane']
    vp = curved_beam('in-plane', 1)
    ve = curved_beam('in-plane', 0)
    print('in-plane: penalty %.6f (%s) enriched %.6f (%s)'
          % (vp, mh_grade(vp, ref), ve, mh_grade(ve, ref)))
    assert mh_grade(vp, ref) == 'F'
    assert mh_grade(ve, ref) == 'B'
    assert mh_grade(ve, ref) < mh_grade(MH_QUAD4_CURVED_IN_PLANE, 1.)
    ves = [ve] + [curved_beam('in-plane', 0, ny=ny) for ny in (2, 4)]
    print('in-plane, enriched, 1, 2 and 4 elements across:', ves)
    assert np.all(np.diff(ves) > 0)
    assert ves[-1] < ref

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
# Table 3 of Allman (1984), p. 7, his own triangular element, and Table II of
# Ibrahimbegovic et al. (1990) for the same problem
ALLMAN_TABLE = {(4, 1): 0.2696, (8, 2): 0.3261, (16, 4): 0.3471}
# NOTE the reference of Table 3 of Allman (1984), p. 7, from Olsen and
#      Bearden, his reference 9
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
            S1 = 0.5*(1. - xi)
            S2 = 0.5*(1. + xi)
            yq = S1*ya + S2*yb
            tau = 3.*W/(2.*H*t)*(1. - 4.*(yq/H)**2)
            jac = (yb - ya)/2.
            fext[DOF*na + 1] += wq*jac*S1*tau*t
            fext[DOF*nb + 1] += wq*jac*S2*tau*t
    assert np.isclose(fext.sum(), W, rtol=1.e-10)

    u = solve(K, N, bk, fext)
    pos = np.where(np.isclose(x, L))[0]
    o = np.argsort(y[pos])
    return np.interp(0., y[pos][o], u[1::DOF][pos][o])


@pytest.mark.slow
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

    On the coarsest mesh, 4 by 1, the error of the enriched element is
    7.8 %, grade B of MacNeal and Harder (1985), p. 14, against 24.2 %, grade
    D, for the triangles of Allman and 31.9 %, grade D, for the penalty. The
    reference is itself numerical; the Aitken extrapolation of the 16 by 4,
    32 by 8 and 64 by 16 results of the enriched element is 0.3561, 0.07 %
    above it.

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
    # the coarsest mesh is graded B, Allman's triangles and the penalty D
    assert mh_grade(allman_cantilever(4, 1, 0), ALLMAN_REF) == 'B'
    assert mh_grade(ALLMAN_TABLE[4, 1], ALLMAN_REF) == 'D'
    assert mh_grade(allman_cantilever(4, 1, 1), ALLMAN_REF) == 'D'
    # and the sequence converges onto it
    assert abs(allman_cantilever(64, 16, 0)/ALLMAN_REF - 1.) < 0.002


if __name__ == '__main__':
    test_C1_cook_skew_membrane()
    test_C2_macneal_harder_straight_cantilever()
    test_C2_twist_needs_two_elements_across_the_width()
    test_C3_macneal_harder_curved_beam()
    test_C4_allman_cantilever()
