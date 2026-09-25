r"""Rank and conditioning of the element with the physics-based drilling stiffness

    Allman, D. J., 1984, "A compatible triangular element including vertex
    rotations for plane elasticity analysis", Computers & Structures, 19(1-2),
    pp. 1-8. doi:10.1016/0045-7949(84)90197-4

    Hughes, T. J. R. and Brezzi, F., 1989, "On drilling degrees of freedom",
    CMAME, 72(1), pp. 105-121. doi:10.1016/0045-7825(89)90124-2

    Ibrahimbegovic, A., Taylor, R. L. and Wilson, E. L., 1990, "A robust
    quadrilateral membrane finite element with drilling degrees of freedom",
    IJNME, 30(3), pp. 445-457. doi:10.1002/nme.1620300305

Allman's enrichment on its own is rank-deficient by one. Setting `u_i = v_i =
0` and `{r_z}_i = \omega_0` makes every edge amplitude `a_k =
(\ell_k/8)({r_z}_i - {r_z}_j)` vanish, so the in-plane displacement field
vanishes identically and the state produces no membrane strain energy,
although it is not a rigid-body motion. Over the `3n` in-plane
degrees-of-freedom the enriched membrane therefore has rank `3n - 4` instead
of `3n - 3`. The Hughes-Brezzi term gives that state the energy
`\frac{1}{2}\gamma_{r_z} A_e \omega_0^2` and restores the rank, which is why
the two must be used together.

The counts below are taken over the in-plane subspace `\{u, v, r_z\}`, which
is where the drilling formulation is responsible for the rank. The full
element matrix of a single free element has more zero eigenvalues than the
six rigid-body modes, one more for a rectangle and two more for a distorted
quadrilateral, identically for both drilling models. The mode that a
rectangle carries is `r_x = 2x - 1`, `r_y = 2y - 1`, `w = 0`, which has zero
curvature everywhere and whose transverse shear rotation part vanishes at the
centroid, so the one-point sampled rotation term of the transverse shear does
not see it. Those modes live entirely in the out-of-plane block `\{w, r_x,
r_y\}`, which is exactly decoupled from the in-plane one for a flat element
with `B = 0`, and the drilling enrichment acts only on the in-plane
translations. They are therefore a property of the mixed integration of the
plate formulation, documented as Figure 10 of Hughes, Taylor and
Kanoknukulchai (1977) and measured in ``test_quad4_spurious_shear_mode.py``.
They predate the drilling work and are unrelated to it, which is why the
assertion below compares the two drilling models against each other rather
than against an absolute number.

"""
import sys
sys.path.append('..')

import numpy as np
from scipy.sparse import coo_matrix

from pyfe3d.shellprop_utils import isotropic_plate, laminated_plate
from pyfe3d import Quad4, Quad4Data, Quad4Probe, INT, DOUBLE, DOF

SQUARE = np.array([[0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.]])
DISTORTED = np.array([[0., 0., 0.], [2., 0., 0.], [2.3, 1.4, 0.],
                      [0.2, 1.1, 0.]])
# in-plane degrees-of-freedom of a four node element: u, v and r_z per node
INPLANE = np.array([i for n in range(4) for i in (DOF*n, DOF*n + 1, DOF*n + 5)])


def shoelace_area(ncoords):
    x = ncoords[:, 0]
    y = ncoords[:, 1]
    return 0.5*abs(x.dot(np.roll(y, -1)) - y.dot(np.roll(x, -1)))


def element_KC0(ncoords, prop, drilling_model=0, K6ROT=100., gamma_rz=-1.):
    data = Quad4Data()
    probe = Quad4Probe()
    el = Quad4(probe)
    ncoords_flatten = ncoords.flatten()
    for i in range(4):
        setattr(el, 'n%d' % (i + 1), i + 1)
        setattr(el, 'c%d' % (i + 1), DOF*i)
    el.init_k_KC0 = 0
    el.drilling_model = drilling_model
    el.K6ROT = K6ROT
    el.gamma_rz = gamma_rz
    el.update_rotation_matrix(ncoords_flatten)
    el.update_probe_xe(ncoords_flatten)
    el.update_area()
    N = DOF*4
    KC0r = np.zeros(data.KC0_SPARSE_SIZE, dtype=INT)
    KC0c = np.zeros(data.KC0_SPARSE_SIZE, dtype=INT)
    KC0v = np.zeros(data.KC0_SPARSE_SIZE, dtype=DOUBLE)
    el.update_KC0(KC0r, KC0c, KC0v, prop)
    return coo_matrix((KC0v, (KC0r, KC0c)), shape=(N, N)).toarray()


def num_zeros(K, tol=1.e-10):
    ev = np.linalg.eigvalsh(0.5*(K + K.T))
    return int((ev < tol*ev.max()).sum()), ev


def test_B1_rank_of_a_single_free_element():
    r"""Three zero eigenvalues over the in-plane degrees-of-freedom

    Those are the two in-plane translations and the rotation about the
    element normal. Any further zero would be a spurious in-plane mechanism.

    """
    for prop in (isotropic_plate(thickness=1., E=1., nu=0.3),
                 isotropic_plate(thickness=0.01, E=1., nu=0.3),
                 laminated_plate(stack=[0, 30, -60, 90], plyt=0.001,
                                 laminaprop=(127.6e9, 11.3e9, 0.3, 6e9, 6e9,
                                             6e9))):
        for ncoords in (SQUARE, DISTORTED):
            for drilling_model in (0, 1):
                K = element_KC0(ncoords, prop, drilling_model)
                nz, ev = num_zeros(K[np.ix_(INPLANE, INPLANE)])
                print(drilling_model, nz, ev[3]/ev.max())
                assert nz == 3, (drilling_model, nz)
                # the first non-zero in-plane eigenvalue must be clear of the
                # round-off floor
                assert ev[3]/ev.max() > 1.e-11

    # NOTE the null space of the whole element matrix is only counted for the
    #      thick, well scaled case. For a thin plate the spectrum spans a
    #      factor of the order of (l/h)**2 between the membrane and the
    #      bending modes, so the softest physical bending mode falls under
    #      any fixed relative threshold and the count stops being meaningful,
    #      which is the conditioning issue of a thin shell and not a rank
    #      defect
    prop = isotropic_plate(thickness=1., E=1., nu=0.3)
    for ncoords in (SQUARE, DISTORTED):
        counts = {}
        for drilling_model in (0, 1):
            K = element_KC0(ncoords, prop, drilling_model)
            nzf, _ = num_zeros(K)
            counts[drilling_model] = nzf
            print('full element, drilling_model %d: %d zeros'
                  % (drilling_model, nzf))
        # the enrichment must not add any zero mode of its own. The absolute
        # count is a property of the transverse shear of the underlying plate
        # formulation and depends on the element shape: seven for a
        # rectangle, eight for a distorted quadrilateral, against the six
        # rigid-body modes. Those extra modes live entirely in the
        # out-of-plane block, which the enrichment leaves untouched, so the
        # meaningful assertion is that the two models agree
        assert counts[0] == counts[1], counts
        assert counts[0] in (7, 8), counts


def test_B2_spurious_allman_mode_is_removed():
    r"""Rank 3n-4 without the Hughes-Brezzi term, 3n-3 with it

    The energy of the uniform drilling state is also checked in closed form
    against `\frac{1}{2}\gamma_{r_z} A_e \omega_0^2`.

    """
    prop = isotropic_plate(thickness=1., E=1., nu=0.3)
    for ncoords in (SQUARE, DISTORTED):
        area = shoelace_area(ncoords)

        # Allman enrichment alone
        K = element_KC0(ncoords, prop, 0, gamma_rz=0.)
        nz, _ = num_zeros(K[np.ix_(INPLANE, INPLANE)])
        print('gamma_rz = 0:', nz)
        assert nz == 4, nz
        # and the extra mode is the uniform drilling state
        w0 = 0.7
        m = np.zeros(DOF*4)
        m[5::DOF] = w0
        assert abs(0.5*m.dot(K.dot(m)))/np.abs(K).max() < 1.e-14

        # with the Hughes-Brezzi term
        for gamma_ratio in (1.e-2, 1., 1.e2):
            gamma = gamma_ratio*prop.A66
            K = element_KC0(ncoords, prop, 0, gamma_rz=gamma)
            nz, _ = num_zeros(K[np.ix_(INPLANE, INPLANE)])
            print('gamma_rz/A66 = %g:' % gamma_ratio, nz)
            assert nz == 3, (gamma_ratio, nz)
            U = 0.5*m.dot(K.dot(m))
            U_ref = 0.5*gamma*area*w0**2
            print('  U = %.10e, closed form %.10e' % (U, U_ref))
            assert np.isclose(U, U_ref, rtol=1.e-12)

        # the default takes gamma_rz = A66
        K_default = element_KC0(ncoords, prop, 0)
        K_explicit = element_KC0(ncoords, prop, 0, gamma_rz=prop.A66)
        assert np.allclose(K_default, K_explicit, rtol=0., atol=0.)


def test_B3_coplanar_mesh_is_not_singular_without_a_user_parameter():
    r"""A flat mesh loses its drilling singularity with K6ROT = 0

    On a coplanar mesh the drilling rotation appears in no strain measure of
    the unenriched formulation, so with the penalty switched off the rows and
    columns of every drilling degree-of-freedom are exactly zero and the
    global matrix has one zero eigenvalue per node on top of the rigid-body
    modes. The physics-based stiffness removes exactly those, without any
    user parameter.

    The assertion is a difference of counts rather than positive
    definiteness of the constrained matrix, because this element has a
    further zero mode of its own, in the transverse shear, that has nothing
    to do with the drilling rotation, as recorded in the module docstring.

    """
    prop = isotropic_plate(thickness=0.01, E=200e9, nu=0.3)
    nx = ny = 5
    a = b = 1.
    xs = np.linspace(0., a, nx)
    ys = np.linspace(0., b, ny)
    xmesh, ymesh = np.meshgrid(xs, ys)
    ncoords = np.vstack((xmesh.T.flatten(), ymesh.T.flatten(),
                         np.zeros(nx*ny))).T
    ncoords_flatten = ncoords.flatten()
    nids = 1 + np.arange(nx*ny)
    nid_pos = dict(zip(nids, np.arange(len(nids))))
    nids_mesh = nids.reshape(nx, ny)
    n1s = nids_mesh[:-1, :-1].flatten()
    n2s = nids_mesh[1:, :-1].flatten()
    n3s = nids_mesh[1:, 1:].flatten()
    n4s = nids_mesh[:-1, 1:].flatten()
    N = DOF*nx*ny
    num_nodes = nx*ny

    data = Quad4Data()
    probe = Quad4Probe()
    ne = len(n1s)

    def assemble(drilling_model):
        KC0r = np.zeros(data.KC0_SPARSE_SIZE*ne, dtype=INT)
        KC0c = np.zeros(data.KC0_SPARSE_SIZE*ne, dtype=INT)
        KC0v = np.zeros(data.KC0_SPARSE_SIZE*ne, dtype=DOUBLE)
        init_k_KC0 = 0
        for n1, n2, n3, n4 in zip(n1s, n2s, n3s, n4s):
            q = Quad4(probe)
            q.n1, q.n2, q.n3, q.n4 = n1, n2, n3, n4
            q.c1 = DOF*nid_pos[n1]
            q.c2 = DOF*nid_pos[n2]
            q.c3 = DOF*nid_pos[n3]
            q.c4 = DOF*nid_pos[n4]
            q.init_k_KC0 = init_k_KC0
            q.drilling_model = drilling_model
            # NOTE the penalty is switched off, so only the physics-based
            #      stiffness can remove the drilling singularity
            q.K6ROT = 0.
            q.update_rotation_matrix(ncoords_flatten)
            q.update_probe_xe(ncoords_flatten)
            q.update_area()
            q.update_KC0(KC0r, KC0c, KC0v, prop)
            init_k_KC0 += data.KC0_SPARSE_SIZE
        K = coo_matrix((KC0v, (KC0r, KC0c)), shape=(N, N)).toarray()
        return 0.5*(K + K.T)

    K_penalty = assemble(1)
    K_enriched = assemble(0)

    rz = np.zeros(N, dtype=bool)
    rz[5::DOF] = True

    # with the penalty switched off the drilling rows are exactly empty
    assert np.abs(K_penalty[rz, :]).max() == 0.
    # the physics-based stiffness populates them, with the scale of A66 times
    # an element area
    diag = np.diag(K_enriched)[rz]
    area = (a/(nx - 1))*(b/(ny - 1))
    print('rz diagonal / (A66*area):', diag.min()/(prop.A66*area),
          diag.max()/(prop.A66*area))
    assert diag.min() > 0.
    assert 1.e-2 < diag.min()/(prop.A66*area) < 1.e2
    assert 1.e-2 < diag.max()/(prop.A66*area) < 1.e2

    nz_penalty, _ = num_zeros(K_penalty, tol=1.e-12)
    nz_enriched, _ = num_zeros(K_enriched, tol=1.e-12)
    print('zero eigenvalues: penalty %d, enriched %d, nodes %d'
          % (nz_penalty, nz_enriched, num_nodes))
    # exactly one drilling singularity per node is removed
    assert nz_penalty - nz_enriched == num_nodes


def test_B5_symmetry():
    r"""The element matrix must be symmetric to machine precision"""
    prop = laminated_plate(stack=[0, 30, -60, 90], plyt=0.001,
                           laminaprop=(127.6e9, 11.3e9, 0.3, 6e9, 6e9, 6e9))
    for ncoords in (SQUARE, DISTORTED):
        for drilling_model in (0, 1):
            K = element_KC0(ncoords, prop, drilling_model)
            # NOTE not bit equality: the two triangles of the matrix are
            #      accumulated in a different order, so they agree to
            #      round-off and not exactly
            asym = np.abs(K - K.T).max()/np.abs(K).max()
            print('drilling_model %d: asymmetry %.3e' % (drilling_model, asym))
            assert asym < 1.e-14, (drilling_model, asym)


def quad4r_KC0(ncoords, prop, drilling_model=0, K6ROT=100., gamma_rz=-1.):
    from pyfe3d import Quad4R, Quad4RData, Quad4RProbe
    data = Quad4RData()
    probe = Quad4RProbe()
    el = Quad4R(probe)
    ncoords_flatten = ncoords.flatten()
    for i in range(4):
        setattr(el, 'n%d' % (i + 1), i + 1)
        setattr(el, 'c%d' % (i + 1), DOF*i)
    el.init_k_KC0 = 0
    el.drilling_model = drilling_model
    el.K6ROT = K6ROT
    el.gamma_rz = gamma_rz
    el.update_rotation_matrix(ncoords_flatten)
    el.update_probe_xe(ncoords_flatten)
    el.update_area()
    N = DOF*4
    KC0r = np.zeros(data.KC0_SPARSE_SIZE, dtype=INT)
    KC0c = np.zeros(data.KC0_SPARSE_SIZE, dtype=INT)
    KC0v = np.zeros(data.KC0_SPARSE_SIZE, dtype=DOUBLE)
    el.update_KC0(KC0r, KC0c, KC0v, prop)
    return coo_matrix((KC0v, (KC0r, KC0c)), shape=(N, N)).toarray()


def test_B6_quad4r_drilling_rank_and_energy():
    r"""The drilling stiffness of :class:`pyfe3d.Quad4R`

    That element applies the regularised operator of Hughes and Brezzi
    (1989) without the enrichment of Allman (1984), see the module
    documentation of :mod:`pyfe3d.quad4r`, with either the physics-based
    coefficient `\gamma_{r_z} = A_{66}` or the fictitious `K6ROT \cdot
    10^{-6} A_{66}`. The operator and its quadrature, two points per
    direction, are the same in both cases, so both must give the in-plane
    block the same rank, and both must reproduce the closed-form energy of
    the uniform drilling state.

    Over the twelve in-plane degrees-of-freedom `u`, `v` and `r_z`, the
    correct count of zero-energy modes is three, the in-plane rigid-body
    motions. Setting `\gamma_{r_z} = 0` removes the drilling stiffness
    altogether and leaves seven, the three rigid-body motions plus the four
    unconstrained drilling rotations, which is the deficiency the term
    exists to remove.

    """
    prop = isotropic_plate(thickness=0.01, E=200.e9, nu=0.3)
    for name, ncoords in [('square', SQUARE), ('distorted', DISTORTED)]:
        area = shoelace_area(ncoords)
        ue = np.zeros(DOF*4)
        ue[5::DOF] = 1.
        cases = [
            ('Hughes-Brezzi', dict(drilling_model=0), prop.A66),
            ('penalty', dict(drilling_model=1, K6ROT=100.),
             100.*1.e-6*prop.A66),
        ]
        for label, kw, gamma in cases:
            KC0 = quad4r_KC0(ncoords, prop, **kw)
            nzeros, ev = num_zeros(KC0[np.ix_(INPLANE, INPLANE)])
            U = 0.5*ue.dot(KC0.dot(ue))
            print(name, label, 'in-plane zeros', nzeros, 'energy', U,
                  'closed form', 0.5*gamma*area)
            assert nzeros == 3, (name, label, nzeros, ev)
            assert np.isclose(U, 0.5*gamma*area, rtol=1e-12), (name, label)
            asym = np.abs(KC0 - KC0.T).max()/np.abs(KC0).max()
            assert asym < 1e-14, (name, label, asym)
        # NOTE without the drilling term the four drilling rotations are free
        KC0 = quad4r_KC0(ncoords, prop, drilling_model=0, gamma_rz=0.)
        nzeros, ev = num_zeros(KC0[np.ix_(INPLANE, INPLANE)])
        print(name, 'gamma_rz = 0, in-plane zeros', nzeros)
        assert nzeros == 7, (name, nzeros, ev)
        assert np.isclose(0.5*ue.dot(KC0.dot(ue)), 0., atol=1e-20), name


if __name__ == '__main__':
    test_B1_rank_of_a_single_free_element()
    test_B2_spurious_allman_mode_is_removed()
    test_B3_coplanar_mesh_is_not_singular_without_a_user_parameter()
    test_B5_symmetry()
    test_B6_quad4r_drilling_rank_and_energy()
