r"""Verification of Tria3DSG, the discrete shear gap triangle

The element is described in :mod:`pyfe3d.tria3dsg` and follows

    Bletzinger, K.-U., Bischoff, M., and Ramm, E., 2000, "A unified approach
    for shear-locking-free triangular and rectangular shell finite
    elements", Computers & Structures, 75(3), pp. 321-334.
    https://doi.org/10.1016/S0045-7949(99)00140-6

It exists because :class:`pyfe3d.tria3r.Tria3R` cures its shear locking by
dividing the transverse shear stiffnesses by `1 + \alpha \ell^2/h^2`, which
leaves it unable to reproduce a constant transverse shear state and makes it
converge to the wrong answer on bending problems. The tests below check the
properties the DSG buys and contrast them with Tria3R throughout, so the
comparison is part of the record rather than a claim.

"""
import sys
sys.path.append('..')

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

from pyfe3d.shellprop_utils import isotropic_plate, laminated_plate
from pyfe3d.solver import linear_buckling, natural_frequency
from pyfe3d import (Tria3DSG, Tria3DSGData, Tria3DSGProbe,
                    Tria3R, Tria3RData, Tria3RProbe,
                    Quad4, Quad4Data, Quad4Probe,
                    Quad4R, Quad4RData, Quad4RProbe,
                    INT, DOUBLE, DOF)

E_ISO = 210.e9
NU_ISO = 0.3
# NOTE a deliberately irregular triangle, so that no result below can be an
#      accident of symmetry
IRREGULAR = np.array([[0.13, -0.07, 0.], [1.21, 0.31, 0.], [0.44, 0.93, 0.]])


def element_K(cls, Data, Probe, ncoords, h, drilling_model=0, alpha=None):
    r"""Single-element stiffness matrix in global coordinates

    ``alpha``, when given, sets ``alpha_shear_locking`` on elements that
    have it; left alone otherwise, so every existing caller is unaffected.
    """
    prop = isotropic_plate(thickness=h, E=E_ISO, nu=NU_ISO)
    nn = len(ncoords)
    flat = ncoords.flatten()
    data = Data()
    el = cls(Probe())
    for k in range(nn):
        setattr(el, 'n%d' % (k + 1), k + 1)
        setattr(el, 'c%d' % (k + 1), DOF*k)
    el.init_k_KC0 = 0
    el.drilling_model = drilling_model
    if alpha is not None:
        el.alpha_shear_locking = alpha
    N = DOF*nn
    KC0r = np.zeros(data.KC0_SPARSE_SIZE, dtype=INT)
    KC0c = np.zeros(data.KC0_SPARSE_SIZE, dtype=INT)
    KC0v = np.zeros(data.KC0_SPARSE_SIZE, dtype=DOUBLE)
    el.update_rotation_matrix(flat, 0., 0., 1.)
    el.update_probe_xe(flat)
    el.update_area()
    el.update_KC0(KC0r, KC0c, KC0v, prop)
    K = coo_matrix((KC0v, (KC0r, KC0c)), shape=(N, N)).toarray()
    return K, el, prop


def mesh_patch(n, L, triangles=True):
    r"""``n`` by ``n`` patch of side ``L``, each cell split into triangles"""
    xs = np.linspace(0., L, n + 1)
    X, Y = np.meshgrid(xs, xs, indexing='ij')
    nid = np.arange((n + 1)**2).reshape(n + 1, n + 1)
    X = X.ravel()
    Y = Y.ravel()
    cells = []
    for i in range(n):
        for j in range(n):
            q = (nid[i, j], nid[i+1, j], nid[i+1, j+1], nid[i, j+1])
            if triangles:
                cells += [(q[0], q[1], q[2]), (q[0], q[2], q[3])]
            else:
                cells.append(q)
    return X, Y, cells


def assemble(cls, Data, Probe, X, Y, cells, h):
    prop = isotropic_plate(thickness=h, E=E_ISO, nu=NU_ISO)
    nn = len(X)
    flat = np.vstack((X, Y, np.zeros(nn))).T.flatten()
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
    return coo_matrix((KC0v, (KC0r, KC0c)), shape=(N, N)).toarray(), N


def test_rigid_body_modes_are_exact():
    r"""All six rigid-body modes must carry no energy

    The discrete shear gap is built by integrating the rotations along the
    edges, which is itself an exact statement for a rigid motion, so this
    holds to machine precision rather than approximately.

    """
    K, el, prop = element_K(Tria3DSG, Tria3DSGData, Tria3DSGProbe,
                            IRREGULAR, 0.01)
    scale = np.abs(K).max()
    assert np.abs(K - K.T).max()/scale < 1e-14
    for a in range(3):
        ue = np.zeros(18)
        ue[a::DOF] = 1.
        energy = abs(ue.dot(K.dot(ue)))/scale
        print('translation %d energy/max|K| = %.3e' % (a, energy))
        assert energy < 1e-14
    for axis in range(3):
        ue = np.zeros(18)
        omega = np.zeros(3)
        omega[axis] = 1.
        for i in range(3):
            ue[DOF*i:DOF*i + 3] = np.cross(omega, IRREGULAR[i])
            ue[DOF*i + 3:DOF*i + 6] = omega
        energy = abs(ue.dot(K.dot(ue)))/scale
        print('rotation %d energy/max|K| = %.3e' % (axis, energy))
        assert energy < 1e-14


def test_rank_matches_tria3r_and_assembly_removes_the_spurious_modes():
    r"""Seven zero eigenvalues alone, six once assembled

    A three-node element has 18 degrees-of-freedom and only eleven
    independent constant strain measures, three membrane, three curvature,
    two transverse shear and three drilling, so a single element has seven
    zero eigenvalues of which one is spurious. Tria3R has the same count for
    the same reason.

    The drilling term is what makes the difference between eleven and nine:
    it is integrated with three points, since a single point supplies one
    constraint and leaves two `r_z` combinations free, which showed up
    during development as eight zero eigenvalues in an assembled patch
    instead of six.

    """
    for label, cls, Data, Probe in [('Tria3DSG', Tria3DSG, Tria3DSGData,
                                     Tria3DSGProbe),
                                    ('Tria3R', Tria3R, Tria3RData,
                                     Tria3RProbe)]:
        K, el, prop = element_K(cls, Data, Probe, IRREGULAR, 0.01)
        ev = np.linalg.eigvalsh(K)/np.abs(K).max()
        nzeros = int((np.abs(ev) < 1e-11).sum())
        print('%-9s single element zero eigenvalues: %d' % (label, nzeros))
        assert nzeros == 7, (label, nzeros)
        for n in (2, 3, 4):
            X, Y, cells = mesh_patch(n, 0.4)
            Kp, N = assemble(cls, Data, Probe, X, Y, cells, 0.01)
            evp = np.linalg.eigvalsh(Kp)/np.abs(Kp).max()
            nzp = int((np.abs(evp) < 1e-11).sum())
            print('%-9s patch n = %d, %d DOF, zero eigenvalues: %d'
                  % (label, n, N, nzp))
            assert nzp == 6, (label, n, nzp)


def test_constant_transverse_shear_state_is_exact():
    r"""The property Tria3R cannot deliver

    With `w = cx` and every rotation at zero the curvature vanishes, so the
    whole energy is `A_{55} c^2 A/2`. The DSG reproduces it exactly at any
    element size, while Tria3R is short by its stabilisation factor, which
    reaches `7 \times 10^{-9}` for the largest element below.

    """
    for L, h in [(1.0, 0.01), (0.1, 0.01), (0.01, 0.01), (0.01, 0.1),
                 (100., 0.01)]:
        nc = np.array([[0., 0., 0.], [L, 0., 0.], [L, L, 0.]])
        got = {}
        for label, cls, Data, Probe in [('Tria3DSG', Tria3DSG, Tria3DSGData,
                                         Tria3DSGProbe),
                                        ('Tria3R', Tria3R, Tria3RData,
                                         Tria3RProbe)]:
            K, el, prop = element_K(cls, Data, Probe, nc, h)
            c = 1.e-4
            ue = np.zeros(18)
            ue[2::DOF] = c*nc[:, 0]
            got[label] = ((0.5*ue.dot(K.dot(ue)))
                          / (0.5*prop.A55*c**2*el.area))
        print('L = %-8g h = %-6g Tria3DSG %.15f   Tria3R %.6e'
              % (L, h, got['Tria3DSG'], got['Tria3R']))
        assert np.isclose(got['Tria3DSG'], 1., rtol=1e-12), (L, h)


def test_kirchhoff_states_produce_no_parasitic_shear():
    r"""The locking-free property, and the reason no parameter is needed

    For each of the three constant-curvature states, with the rotations set
    to the Kirchhoff values `\pmb{\phi} = \nabla w`, the exact transverse
    shear strain is zero. A displacement-based three-node triangle cannot
    represent the quadratic `w` in its interior and so develops parasitic
    shear, which is what locks it. The DSG operator uses only the nodal
    values and gives exactly zero, so the element energy equals the bending
    energy `\pmb{\kappa}^T [D] \pmb{\kappa} A/2` to machine precision.

    Tria3R is 8 to 23 per cent above that on the same triangle, which is the
    parasitic shear its stabilisation is there to suppress.

    """
    kap = 1.e-3
    X, Y = IRREGULAR[:, 0], IRREGULAR[:, 1]
    states = {
        'kxx': (0.5*kap*X**2, 0.*X, -kap*X, np.array([kap, 0., 0.])),
        'kyy': (0.5*kap*Y**2, kap*Y, 0.*X, np.array([0., -kap, 0.])),
        'twist': (kap*X*Y, kap*X, -kap*Y, np.array([0., 0., -2.*kap])),
    }
    for label, cls, Data, Probe in [('Tria3DSG', Tria3DSG, Tria3DSGData,
                                     Tria3DSGProbe),
                                    ('Tria3R', Tria3R, Tria3RData,
                                     Tria3RProbe)]:
        K, el, prop = element_K(cls, Data, Probe, IRREGULAR, 0.01)
        D = np.array([[prop.D11, prop.D12, prop.D16],
                      [prop.D12, prop.D22, prop.D26],
                      [prop.D16, prop.D26, prop.D66]])
        for name, (wv, rxv, ryv, kvec) in states.items():
            ue = np.zeros(18)
            ue[2::DOF] = wv
            ue[3::DOF] = rxv
            ue[4::DOF] = ryv
            U = 0.5*ue.dot(K.dot(ue))
            exact = 0.5*kvec.dot(D.dot(kvec))*el.area
            print('%-9s %-6s U/exact = %.10f' % (label, name, U/exact))
            if label == 'Tria3DSG':
                assert np.isclose(U, exact, rtol=1e-11), (name, U/exact)
            else:
                # NOTE recorded so that the contrast is part of the record
                assert U/exact > 1.05, (name, U/exact)


def test_membrane_constant_strain_patch_test():
    r"""Constant in-plane strain must be exact, the triangle being a CST

    The drilling constraint is part of the element, so the test field has to
    satisfy it: `r_z` is set to `	heta_z`, otherwise the Hughes and Brezzi
    term correctly charges energy for the mismatch, which it did at 0.8 per
    cent while this test was being written.

    """
    K, el, prop = element_K(Tria3DSG, Tria3DSGData, Tria3DSGProbe,
                            IRREGULAR, 0.01)
    X, Y = IRREGULAR[:, 0], IRREGULAR[:, 1]
    exx, eyy, gxy = 1.e-4, -0.5e-4, 0.3e-4
    ue = np.zeros(18)
    ue[0::DOF] = exx*X + gxy*Y
    ue[1::DOF] = eyy*Y
    # NOTE the drilling degree-of-freedom has to follow the in-plane
    #      rotation, r_z = theta_z = (v_,x - u_,y)/2, or the Hughes and
    #      Brezzi term legitimately adds energy and the state is not a
    #      constant-strain one at all
    ue[5::DOF] = -gxy/2.
    A = np.array([[prop.A11, prop.A12, prop.A16],
                  [prop.A12, prop.A22, prop.A26],
                  [prop.A16, prop.A26, prop.A66]])
    evec = np.array([exx, eyy, gxy])
    U = 0.5*ue.dot(K.dot(ue))
    exact = 0.5*evec.dot(A.dot(evec))*el.area
    print('membrane U/exact = %.12f' % (U/exact))
    assert np.isclose(U, exact, rtol=1e-11)


def test_node_numbering_invariance():
    r"""The symmetrised gap makes the element independent of node ordering

    Bletzinger et al. build the shear gap by integrating the rotations away
    from one node, which makes the plain operator depend on which node that
    is. Measured on the irregular triangle below while this element was
    being written, the plain operator moved the element matrix by 14 and 17
    per cent under the two cyclic relabellings, against 3e-16 for Tria3R,
    whose shear comes from a symmetric centroid evaluation. A mesh generator
    orders the nodes of a triangle as it pleases, so that was not
    acceptable.

    The element therefore averages the three operators, one per starting
    node. Each of them is separately exact for the rigid-body modes, for
    constant transverse shear and for the Kirchhoff curvature states, and
    all of those are linear in the operator, so the mean keeps them, which
    the other tests in this file check. What the averaging adds is the
    invariance measured here.

    """
    for label, cls, Data, Probe in [('Tria3DSG', Tria3DSG, Tria3DSGData,
                                     Tria3DSGProbe),
                                    ('Tria3R', Tria3R, Tria3RData,
                                     Tria3RProbe)]:
        ref, _, _ = element_K(cls, Data, Probe, IRREGULAR, 0.01)
        worst = 0.
        for shift in (1, 2):
            ncoords = np.roll(IRREGULAR, shift, axis=0)
            K, _, _ = element_K(cls, Data, Probe, ncoords, 0.01)
            # NOTE original node p sits at new index (p + shift) % 3
            perm = [(p + shift) % 3 for p in range(3)]
            idx = np.concatenate([np.arange(DOF) + DOF*p for p in perm])
            worst = max(worst, np.abs(K[np.ix_(idx, idx)] - ref).max()
                        / np.abs(ref).max())
        print('%-9s worst relative change over node orderings: %.3e'
              % (label, worst))
        assert worst < 1e-13, (label, worst)


def plate_point_load(cls, Data, Probe, n, h, triangles=True):
    r"""Central deflection of a simply supported square plate, point load

    The thin-plate reference is `w = 0.01160 P a^2/D` for `\nu = 0.3`:

        Timoshenko, S., and Woinowsky-Krieger, S., 1959, "Theory of Plates
        and Shells", 2nd ed., McGraw-Hill, Table 8.

    """
    L = 1.0
    P = 1000.
    prop = isotropic_plate(thickness=h, E=E_ISO, nu=NU_ISO)
    X, Y, cells = mesh_patch(n, L, triangles)
    nn = len(X)
    flat = np.vstack((X, Y, np.zeros(nn))).T.flatten()
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
    K = coo_matrix((KC0v, (KC0r, KC0c)), shape=(N, N)).tocsc()
    edge = (np.isclose(X, 0.) | np.isclose(X, L) | np.isclose(Y, 0.)
            | np.isclose(Y, L))
    bk = np.zeros(N, dtype=bool)
    bk[0::DOF] = edge
    bk[1::DOF] = edge
    bk[2::DOF] = edge
    bu = ~bk
    f = np.zeros(N)
    mid = np.where(np.isclose(X, L/2) & np.isclose(Y, L/2))[0]
    f[DOF*mid + 2] = P/len(mid)
    u = np.zeros(N)
    u[bu] = spsolve(K[bu, :][:, bu], f[bu])
    Dpl = E_ISO*h**3/(12.*(1. - NU_ISO**2))
    return u[2::DOF].max()/(0.01160*P*L**2/Dpl)


def test_converges_where_tria3r_converges_to_the_wrong_answer():
    r"""The point of the element, on a problem with an analytical value

    Simply supported square plate under a central point load. Tria3DSG is
    stiff on coarse meshes, which is the expected behaviour of a
    constant-strain triangle on a problem with a singular shear field, but
    it converges to the reference. Tria3R looks better on the coarsest mesh,
    because its soft transverse shear happens to compensate, and then
    settles about ten per cent away from the reference and stays there.

    """
    ratios = {'Tria3DSG': [], 'Tria3R': []}
    for n in (8, 16, 32, 48):
        d = plate_point_load(Tria3DSG, Tria3DSGData, Tria3DSGProbe, n, 0.01)
        t = plate_point_load(Tria3R, Tria3RData, Tria3RProbe, n, 0.01)
        ratios['Tria3DSG'].append(d)
        ratios['Tria3R'].append(t)
        print('n = %-4d Tria3DSG %.6f   Tria3R %.6f' % (n, d, t))
    dsg = ratios['Tria3DSG']
    t3r = ratios['Tria3R']
    # NOTE monotone approach to the reference from below, and within one per
    #      cent of it on the finest mesh
    for k in range(len(dsg) - 1):
        assert dsg[k + 1] > dsg[k]
    assert abs(dsg[-1] - 1.) < 0.01, dsg[-1]
    # NOTE whereas Tria3R settles away from it and the last two meshes agree
    #      with each other far better than either agrees with the reference
    assert abs(t3r[-1] - 1.) > 0.05, t3r[-1]
    assert abs(t3r[-1] - t3r[-2]) < 0.1*abs(t3r[-1] - 1.)


# NOTE node 1 at the origin and node 2 on the +x axis, with node 3 placed
#      so that the normal is +z, which makes the element coordinate system
#      of update_rotation_matrix() coincide with the global one and lets the
#      closed forms below be written directly in global coordinates. Node 3
#      is off centre so that nothing depends on symmetry
ALIGNED = np.array([[0., 0., 0.], [1.3, 0., 0.], [0.37, 0.81, 0.]])
NUM_NODES_TRI = 3


def single_element(cls, Data, Probe, ncoords, h, prop=None):
    r"""One element of ``cls``, ready to be given displacements"""
    if prop is None:
        prop = isotropic_plate(thickness=h, E=E_ISO, nu=NU_ISO)
    nn = len(ncoords)
    flat = ncoords.flatten()
    el = cls(Probe())
    for k in range(nn):
        setattr(el, 'n%d' % (k + 1), k + 1)
        setattr(el, 'c%d' % (k + 1), DOF*k)
    el.init_k_KC0 = 0
    el.init_k_KCNL = 0
    el.init_k_KG = 0
    el.update_rotation_matrix(flat)
    el.update_probe_xe(flat)
    return el, prop, Data(), flat


def element_matrix(el, data, prop, u, which):
    r"""One element matrix, ``KC0``, ``KCNL`` or ``KG``, as a dense array"""
    size = getattr(data, '%s_SPARSE_SIZE' % which)
    n = DOF*NUM_NODES_TRI
    r = np.zeros(size, dtype=INT)
    c = np.zeros(size, dtype=INT)
    v = np.zeros(size, dtype=DOUBLE)
    el.update_probe_ue(u)
    if which == 'KC0':
        el.update_KC0(r, c, v, prop)
    elif which == 'KCNL':
        el.update_KCNL(r, c, v, prop)
    else:
        el.update_KG(r, c, v, prop)
    return coo_matrix((v, (r, c)), shape=(n, n)).toarray()


def shape_function_gradients(ncoords):
    r"""``area``, `\partial N_i/\partial x` and `\partial N_i/\partial y`"""
    x = ncoords[:, 0]
    y = ncoords[:, 1]
    detJ = (x[0]*(y[1] - y[2]) + x[1]*(y[2] - y[0])
            + x[2]*(y[0] - y[1]))
    Nx = np.array([y[1] - y[2], y[2] - y[0], y[0] - y[1]])/detJ
    Ny = np.array([x[2] - x[1], x[0] - x[2], x[1] - x[0]])/detJ
    return abs(detJ)/2., Nx, Ny


def test_nonlinear_internal_forces_of_an_imposed_slope():
    r"""The von Karman terms, against virtual work on a state with a closed form

    The field `w = c x`, `r_y = -c`, with everything else zero, is the
    linearised rigid rotation about the `y` axis. Every linear strain of the
    element vanishes for it: the curvatures because `r_y` is constant, and
    the discrete shear gap exactly, since
    `\Delta w_q = c(x_q - x_p) - c(x_q - x_p) = 0` edge by edge. So
    ``update_fint`` with ``nonlinear=0`` returns zero and everything left is
    the von Karman contribution, which is what this test is after. That a
    rotation is charged any energy at all is the known price of the von
    Karman kinematics, not a defect of the element.

    The only nonlinear strain is `\epsilon_{NL} = \{c^2/2, 0, 0\}`, giving
    the uniform membrane stress `\{N\} = [A] \{\epsilon_{NL}\}`, and the
    internal forces are checked by virtual work against three fields whose
    virtual strains are elementary:

    - `\delta u = a x` gives `\delta \epsilon_{xx} = a`, work `N_{xx} a A`
    - `\delta v = a y` gives `\delta \epsilon_{yy} = a`, work `N_{yy} a A`
    - `\delta w = b x` gives `\delta \epsilon_{xx} = w_{,x} \delta w_{,x} =
      c b`, work `N_{xx} c b A`

    The third one is the term that carries the factor `w_{,x}` and not
    `w_{,x}/2`, the classic place to get the von Karman variation wrong.

    """
    h = 0.02
    c = 0.03
    el, prop, data, flat = single_element(Tria3DSG, Tria3DSGData,
                                         Tria3DSGProbe, ALIGNED, h)
    area, Nx, Ny = shape_function_gradients(ALIGNED)
    n = DOF*NUM_NODES_TRI

    u = np.zeros(n)
    u[2::DOF] = c*ALIGNED[:, 0]
    u[4::DOF] = -c

    # NOTE isotropic, so the laminate matrix is the same in the element and
    #      in the material coordinate systems whatever the orientation
    Nxx = prop.A11*c**2/2.
    Nyy = prop.A12*c**2/2.

    linear = np.zeros(n)
    el.update_probe_ue(u)
    el.update_fint(linear, prop, nonlinear=0)
    scale = abs(prop.A11*c*area)
    print('linear fint of the rotation: %.3e, relative to %.3e'
          % (abs(linear).max(), scale))
    # NOTE a linearised rigid rotation is a zero energy state of KC0
    assert abs(linear).max()/scale < 1e-14

    fint = np.zeros(n)
    el.update_probe_ue(u)
    el.update_fint(fint, prop, nonlinear=1)

    a = 0.7
    b = 0.5
    cases = []
    du = np.zeros(n)
    du[0::DOF] = a*ALIGNED[:, 0]
    cases.append(('du = a x', du.copy(), Nxx*a*area))
    du[:] = 0.
    du[1::DOF] = a*ALIGNED[:, 1]
    cases.append(('dv = a y', du.copy(), Nyy*a*area))
    du[:] = 0.
    du[2::DOF] = b*ALIGNED[:, 0]
    cases.append(('dw = b x', du.copy(), Nxx*c*b*area))
    for label, du, expected in cases:
        got = fint @ du
        print('%-10s virtual work %.12e  exact %.12e  ratio %.14f'
              % (label, got, expected, got/expected))
        assert np.isclose(got, expected, rtol=1e-13), (label, got, expected)

    # NOTE the nonlinear membrane strain involves no rotation, so the
    #      rotations and the drilling take none of the internal force
    for d, name in ((3, 'rx'), (4, 'ry'), (5, 'rz')):
        assert abs(fint[d::DOF]).max()/scale < 1e-14, name
    # NOTE and a uniform stress state is self equilibrated
    for d in range(3):
        assert abs(fint[d::DOF].sum())/scale < 1e-14, d

    # NOTE the exact nodal forces, entry by entry, from the same uniform
    #      stress: Bm.T*N for the in-plane rows and w_x*G.T*N for the
    #      transverse one
    expected = np.zeros(n)
    expected[0::DOF] = area*Nx*Nxx
    expected[1::DOF] = area*Ny*Nyy
    expected[2::DOF] = area*c*Nx*Nxx
    assert np.allclose(fint, expected, rtol=1e-13, atol=1e-13*scale)


def test_transverse_shear_stays_out_of_the_nonlinear_terms():
    r"""The discrete shear gap enters KC0 alone

    First-order shear deformation theory gives the transverse shear strains
    no von Karman terms, so scaling `A_{44}, A_{45}, A_{55}` must leave
    ``KCNL`` and ``KG`` untouched while it changes ``KC0``. If the shear
    operator ever leaked into the nonlinear tangent, the element would stop
    being the exact Jacobian of its own internal forces, which
    ``tests/test_tangent_consistency.py`` would then catch only indirectly.

    """
    h = 0.02
    el, prop, data, flat = single_element(Tria3DSG, Tria3DSGData,
                                         Tria3DSGProbe, ALIGNED, h)
    rng = np.random.default_rng(19)
    u = 0.01*rng.standard_normal(DOF*NUM_NODES_TRI)

    before = {w: element_matrix(el, data, prop, u, w)
              for w in ('KC0', 'KCNL', 'KG')}
    prop.A44 *= 1000.
    prop.A45 *= 1000.
    prop.A55 *= 1000.
    after = {w: element_matrix(el, data, prop, u, w)
             for w in ('KC0', 'KCNL', 'KG')}

    for w in ('KCNL', 'KG'):
        change = abs(after[w] - before[w]).max()
        print('%-5s max change under Ats x 1000: %.3e  (norm %.3e)'
              % (w, change, abs(before[w]).max()))
        assert abs(before[w]).max() > 0.
        assert change == 0., w
    grew = abs(after['KC0'] - before['KC0']).max()/abs(before['KC0']).max()
    print('KC0   relative change under Ats x 1000: %.3f' % grew)
    assert grew > 1.


# NOTE an unsymmetric laminate, so that the B block carries weight. Every
#      other test in this file uses isotropic_plate, which makes B = 0 and
#      makes A and D invariant under a rotation of the material direction,
#      and that is exactly why the two tests below had to be added: they
#      cover the only part of this element that an isotropic plate cannot
#      exercise. See test_material_direction_matches_tria3r.
UNSYM_STACK = [+49, -49, +36, -36, 0, 0]
UNSYM_PLYT = 0.125e-3
UNSYM_LAMINAPROP = (123.55e9, 8.708e9, 0.319, 5.595e9, 5.595e9, 5.595e9)


def unsymmetric_prop():
    return laminated_plate(stack=UNSYM_STACK, plyt=UNSYM_PLYT,
                           laminaprop=UNSYM_LAMINAPROP,
                           shear_correction='constant')


def rotation_matrix(seed):
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.standard_normal((3, 3)))
    if np.linalg.det(q) < 0.:
        q[:, 0] *= -1.
    return q


def material_rotation(cls, Probe, ncoords, xmat):
    r"""``m11, m12, m21, m22`` of one element for a material direction"""
    nn = len(ncoords)
    flat = np.asarray(ncoords, dtype=float).flatten()
    el = cls(Probe())
    for k in range(nn):
        setattr(el, 'n%d' % (k + 1), k + 1)
        setattr(el, 'c%d' % (k + 1), DOF*k)
    el.update_rotation_matrix(flat, xmat[0], xmat[1], xmat[2])
    return np.array([el.m11, el.m12, el.m21, el.m22])


def test_material_direction_matches_tria3r():
    r"""The in-plane material rotation, against the reference elements

    :class:`pyfe3d.quad4.Quad4` and :class:`pyfe3d.tria3r.Tria3R` build
    ``m11, m12, m21, m22`` identically, line for line, and this element must
    agree with them: the same laminate and the same material direction have
    to give the same element-frame stiffness whichever element is used.

    Three parts of that construction are easy to get wrong and an earlier
    hand-written version of :meth:`pyfe3d.tria3dsg.Tria3DSG.\
update_rotation_matrix` got all three wrong at once:

    * the material direction must be projected onto the element plane and
      renormalised before the angle is taken, otherwise any component along
      the element normal shrinks the cosine and ``m`` stops being a rotation;
    * the sine is negative when the material direction has a positive
      component along the element `y` axis. The opposite convention mirrors
      every angle ply, which is invisible for an isotropic plate;
    * a material direction parallel to the element normal leaves no
      projection, and ``m`` must then stay at the identity. A caller that
      passes the normal of a flat mesh hits this, and for triangles it
      matters more than for quadrilaterals, because the two triangles of a
      diagonally split cell have element frames 45 degrees apart, so any
      fallback that depends on the element frame would orient the laminate
      differently in each of them.

    Checked on an arbitrarily oriented element, so that the element normal
    is along no global axis and the projection is not trivially satisfied.

    """
    Q = rotation_matrix(4)
    tri = IRREGULAR @ Q.T
    # NOTE the second triangle of a diagonally split cell, whose node 1 to
    #      node 2 edge, and therefore element x axis, runs along the diagonal
    diag = np.array([[0., 0., 0.], [1., 1., 0.], [0., 1., 0.]]) @ Q.T
    directions = {
        'in-plane, along the element x axis': Q @ np.array([1., 0., 0.]),
        'in-plane, +30 deg': Q @ np.array([np.cos(np.pi/6), np.sin(np.pi/6),
                                           0.]),
        'in-plane, -30 deg': Q @ np.array([np.cos(-np.pi/6),
                                           np.sin(-np.pi/6), 0.]),
        'tilted out of the element plane': Q @ np.array([0.8, 0.4, 0.6]),
        'parallel to the element normal': Q @ np.array([0., 0., 1.]),
        'the zero vector': np.zeros(3),
    }
    for label, xmat in directions.items():
        for nodes, which in ((tri, 'x along an edge'),
                             (diag, 'x along a diagonal')):
            ref = material_rotation(Tria3R, Tria3RProbe, nodes, xmat)
            got = material_rotation(Tria3DSG, Tria3DSGProbe, nodes, xmat)
            print('%-36s %-20s Tria3R %s  Tria3DSG %s'
                  % (label, which,
                     np.array2string(ref, precision=6, suppress_small=True),
                     np.array2string(got, precision=6, suppress_small=True)))
            assert np.allclose(got, ref, atol=1e-13), (label, which, ref, got)
            # NOTE and it must be a rotation matrix, which the missing
            #      projection broke
            assert np.isclose(got[0]**2 + got[1]**2, 1., atol=1e-13), label
            assert np.isclose(got[0], got[3], atol=1e-13), label
            assert np.isclose(got[1], -got[2], atol=1e-13), label


def combined_state_energy(cls, Data, Probe, n, prop, matangle,
                          Lx=0.37, Ly=0.29):
    r"""Strain energy of a constant membrane *and* curvature state

    The field is

    .. math::
        u = \epsilon_{xx} x + \gamma_{xy} y/2, \quad
        v = \gamma_{xy} x/2 + \epsilon_{yy} y, \quad
        w = (c_1 x^2 + c_2 y^2)/2 + c_3 x y

    with `r_x = w_{,y}`, `r_y = -w_{,x}` and `r_z = 0`, which gives the
    constant strains `\{\epsilon\}` and the constant curvatures
    `\{\kappa\} = \{-c_1, -c_2, -2 c_3\}` and zero transverse shear.

    The zero shear is exact for the discrete shear gap even though `w` is
    quadratic and the element interpolates it linearly: the gap integrates a
    linear rotation field along a straight edge with the trapezoidal rule,
    which is exact for a linear integrand, and the result equals the exact
    difference of `w` between the two ends. That is the same property as
    :func:`test_kirchhoff_states_produce_no_parasitic_shear`.

    """
    exx, eyy, gxy = 3.1e-4, -1.7e-4, 2.3e-4
    c1, c2, c3 = 0.11, -0.07, 0.05
    num_nodes = 4 if 'Quad' in cls.__name__ else 3
    xs = np.linspace(0., Lx, n + 1)
    ys = np.linspace(0., Ly, n + 1)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    nid = np.arange((n + 1)**2).reshape(n + 1, n + 1)
    X = X.ravel()
    Y = Y.ravel()
    cells = []
    for i in range(n):
        for j in range(n):
            q = (nid[i, j], nid[i+1, j], nid[i+1, j+1], nid[i, j+1])
            if num_nodes == 4:
                cells.append(q)
            else:
                cells += [(q[0], q[1], q[2]), (q[0], q[2], q[3])]
    nn = len(X)
    flat = np.vstack((X, Y, np.zeros(nn))).T.flatten()
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
        el.update_rotation_matrix(flat, np.cos(matangle), np.sin(matangle),
                                  0.)
        el.update_probe_xe(flat)
        el.update_area()
        el.update_KC0(KC0r, KC0c, KC0v, prop)
        init_k_KC0 += data.KC0_SPARSE_SIZE
    K = coo_matrix((KC0v, (KC0r, KC0c)), shape=(N, N)).toarray()
    u = np.zeros(N)
    u[0::DOF] = exx*X + gxy*Y/2.
    u[1::DOF] = gxy*X/2. + eyy*Y
    u[2::DOF] = (c1*X**2 + c2*Y**2)/2. + c3*X*Y
    u[3::DOF] = c2*Y + c3*X
    u[4::DOF] = -(c1*X + c3*Y)
    e = np.array([exx, eyy, gxy])
    k = np.array([-c1, -c2, -2.*c3])
    return 0.5*u @ (K @ u), e, k, Lx*Ly


def test_unsymmetric_laminate_couples_membrane_and_bending_exactly():
    r"""The B block, against a closed form and against Quad4

    For a state of constant membrane strain and constant curvature the
    strain energy is

    .. math::
        U = \frac{A}{2} \left( \{\epsilon\}^T [A] \{\epsilon\}
            + 2 \{\epsilon\}^T [B] \{\kappa\}
            + \{\kappa\}^T [D] \{\kappa\} \right)

    and the cross term is the only place `[B]` appears, so this is the one
    state that tests it. With the material direction along the element `x`
    axis the closed form can be written straight from the laminate, and it
    is checked first. At other material directions the laminate has to be
    rotated, and rather than repeat that rotation here the element is
    checked against :class:`pyfe3d.quad4.Quad4`, which reproduces the same
    state exactly and is the element this one was written from.

    Both parts caught the material-rotation defects described in
    :func:`test_material_direction_matches_tria3r`: the energy was wrong by
    a mesh-independent factor, 0.714 in the case first measured, while every
    isotropic test in this file stayed exact.

    """
    prop = unsymmetric_prop()
    A = np.array([[prop.A11, prop.A12, prop.A16],
                  [prop.A12, prop.A22, prop.A26],
                  [prop.A16, prop.A26, prop.A66]])
    B = np.array([[prop.B11, prop.B12, prop.B16],
                  [prop.B12, prop.B22, prop.B26],
                  [prop.B16, prop.B26, prop.B66]])
    D = np.array([[prop.D11, prop.D12, prop.D16],
                  [prop.D12, prop.D22, prop.D26],
                  [prop.D16, prop.D26, prop.D66]])
    assert abs(B).max() > 0., 'the laminate must be unsymmetric'

    for n in (1, 3):
        U, e, k, area = combined_state_energy(Tria3DSG, Tria3DSGData,
                                              Tria3DSGProbe, n, prop, 0.)
        memb = 0.5*area*(e @ A @ e)
        coup = area*(e @ B @ k)
        bend = 0.5*area*(k @ D @ k)
        exact = memb + coup + bend
        print('n = %d  energy %.12e  exact %.12e  ratio %.12f   '
              '(coupling is %.2f%% of the exact total)'
              % (n, U, exact, U/exact, 100*coup/exact))
        # NOTE the coupling has to be a real part of the total, or the test
        #      would pass with B ignored altogether
        assert abs(coup/exact) > 0.05
        assert np.isclose(U, exact, rtol=1e-12), (n, U, exact)

    for matangle in (np.radians(20.), np.radians(-35.)):
        for n in (1, 3):
            got, _, _, _ = combined_state_energy(
                Tria3DSG, Tria3DSGData, Tria3DSGProbe, n, prop, matangle)
            ref, _, _, _ = combined_state_energy(
                Quad4, Quad4Data, Quad4Probe, n, prop, matangle)
            print('material at %+6.1f deg, n = %d  Tria3DSG %.12e  '
                  'Quad4 %.12e  ratio %.12f'
                  % (np.degrees(matangle), n, got, ref, got/ref))
            assert np.isclose(got, ref, rtol=1e-12), (matangle, n, got, ref)


def triangulated_plate_buckling(cls, Data, Probe, n, h, second='diagonal',
                                rotate_prestress=True):
    r"""Uniaxial buckling load of a triangulated simply supported plate

    Each cell is split into ``(n1, n2, n3)`` and a second triangle whose node
    numbering is chosen by ``second``:

    * ``'diagonal'`` gives ``(n1, n3, n4)``, whose node 1 to node 2 edge, and
      therefore element `x` axis, runs along the cell diagonal;
    * ``'edge'`` gives ``(n3, n4, n1)``, a cyclic relabelling of the same
      triangle whose element `x` axis runs along a mesh edge instead.

    The two describe the same triangle, and for an element whose matrices do
    not depend on the node numbering they give the identical `K_{C_0}`. What
    differs is the element coordinate system, and therefore what
    ``update_KG_given_stress`` has to be given.

    ``rotate_prestress`` selects how the uniaxial state `N_{xx}` along the
    global `x` axis is handed over. The correct way is a tensor rotation:
    with `[R]` the element-to-global rotation and
    `\{a\} = [R]^T \{1, 0, 0\}` the loading direction in element
    coordinates, the state `N_{xx} \{a\} \{a\}^T` has components

    .. math::
        N^e_{xx} = N_{xx} a_x^2, \quad
        N^e_{yy} = N_{xx} a_y^2, \quad
        N^e_{xy} = N_{xx} a_x a_y

    which reduces to `(N_{xx}, 0, 0)` when the element `x` axis is along the
    load.

    """
    L = 1.0
    prop = isotropic_plate(thickness=h, E=E_ISO, nu=NU_ISO)
    xs = np.linspace(0., L, n + 1)
    X, Y = np.meshgrid(xs, xs, indexing='ij')
    nid = np.arange((n + 1)**2).reshape(n + 1, n + 1)
    X = X.ravel()
    Y = Y.ravel()
    cells = []
    for i in range(n):
        for j in range(n):
            q = (nid[i, j], nid[i+1, j], nid[i+1, j+1], nid[i, j+1])
            cells.append((q[0], q[1], q[2]))
            if second == 'diagonal':
                cells.append((q[0], q[2], q[3]))
            else:
                cells.append((q[2], q[3], q[0]))
    nn = len(X)
    flat = np.vstack((X, Y, np.zeros(nn))).T.flatten()
    data = Data()
    probe = Probe()
    N = DOF*nn
    ne = len(cells)
    KC0r = np.zeros(data.KC0_SPARSE_SIZE*ne, dtype=INT)
    KC0c = np.zeros(data.KC0_SPARSE_SIZE*ne, dtype=INT)
    KC0v = np.zeros(data.KC0_SPARSE_SIZE*ne, dtype=DOUBLE)
    KGr = np.zeros(data.KG_SPARSE_SIZE*ne, dtype=INT)
    KGc = np.zeros(data.KG_SPARSE_SIZE*ne, dtype=INT)
    KGv = np.zeros(data.KG_SPARSE_SIZE*ne, dtype=DOUBLE)
    Nxx = -1.
    init_k_KC0 = 0
    init_k_KG = 0
    for e in cells:
        el = cls(probe)
        for a in range(3):
            setattr(el, 'n%d' % (a + 1), e[a] + 1)
            setattr(el, 'c%d' % (a + 1), DOF*e[a])
        el.init_k_KC0 = init_k_KC0
        el.init_k_KG = init_k_KG
        el.update_rotation_matrix(flat, 1., 0., 0.)
        el.update_probe_xe(flat)
        el.update_area()
        el.update_KC0(KC0r, KC0c, KC0v, prop)
        if rotate_prestress:
            Rt = np.array([[el.r11, el.r21, el.r31],
                           [el.r12, el.r22, el.r32],
                           [el.r13, el.r23, el.r33]])
            ax, ay, _ = Rt @ np.array([1., 0., 0.])
            el.update_KG_given_stress(Nxx*ax*ax, Nxx*ay*ay, Nxx*ax*ay,
                                      KGr, KGc, KGv)
        else:
            el.update_KG_given_stress(Nxx, 0., 0., KGr, KGc, KGv)
        init_k_KC0 += data.KC0_SPARSE_SIZE
        init_k_KG += data.KG_SPARSE_SIZE
    KC0 = coo_matrix((KC0v, (KC0r, KC0c)), shape=(N, N)).tocsc()
    KG = coo_matrix((KGv, (KGr, KGc)), shape=(N, N)).tocsc()
    bk = np.zeros(N, dtype=bool)
    bk[2::DOF] = (np.isclose(X, 0.) | np.isclose(X, L) | np.isclose(Y, 0.)
                  | np.isclose(Y, L))
    bk[0::DOF] = np.isclose(X, L/2.) & (np.isclose(Y, 0.)
                                        | np.isclose(Y, L))
    bk[1::DOF] = np.isclose(Y, L/2.) & (np.isclose(X, 0.)
                                        | np.isclose(X, L))
    bk[5::DOF] = True
    bu = ~bk
    eigvals, _ = linear_buckling(KC0[bu, :][:, bu], KG[bu, :][:, bu],
                                num_eigvalues=1, tol=1e-9)
    Dpl = E_ISO*h**3/(12.*(1. - NU_ISO**2))
    exact = 4.*np.pi**2*Dpl/L**2
    return eigvals[0]*abs(Nxx)*L, exact, KC0[bu, :][:, bu]


def test_buckling_load_does_not_depend_on_triangle_node_numbering():
    r"""A given stress state has to be rotated into the element frame

    :meth:`pyfe3d.tria3dsg.Tria3DSG.update_KG_given_stress` takes the
    membrane stress resultants **in the element coordinate system**, as its
    docstring says. That is easy to overlook with triangles, because the
    element `x` axis runs from node 1 to node 2, so the second triangle of a
    diagonally split cell has its frame about 45 degrees away from the mesh
    direction while the first has it along the mesh. Handing the same
    ``(Nxx, 0, 0)`` to both then loads half the elements along the cell
    diagonal instead of along the intended axis.

    The consequence is a result that depends on how the mesh generator
    happened to number the nodes of each triangle, which is not acceptable:
    a cyclic relabelling of the second triangle leaves `K_{C_0}` untouched,
    as :func:`test_node_numbering_invariance` shows, and must therefore
    leave the buckling load untouched too.

    Measured on the cylinder Z22 of
    ``tests/test_quad4_linear_buckling_cylinder_displ.py``, where the cells
    are nearly square so the diagonal is at 44.3 degrees, the same mistake
    moves the buckling load of Tria3R by a factor of seven. It is worth a
    test of its own.

    """
    n = 12
    h = 0.01
    for cls, Data, Probe, name in ((Tria3DSG, Tria3DSGData, Tria3DSGProbe,
                                    'Tria3DSG'),
                                   (Tria3R, Tria3RData, Tria3RProbe,
                                    'Tria3R')):
        got = {}
        mats = {}
        for second in ('diagonal', 'edge'):
            for rotate in (True, False):
                Pcr, exact, K = triangulated_plate_buckling(
                    cls, Data, Probe, n, h, second=second,
                    rotate_prestress=rotate)
                got[(second, rotate)] = Pcr/exact
                mats[second] = K
        # NOTE the two numberings describe the same triangles, so the
        #      stiffness matrix must be identical
        dK = abs(mats['diagonal'] - mats['edge']).max()/abs(
            mats['diagonal']).max()
        print('%-9s KC0 under the relabelling: %.3e relative' % (name, dK))
        assert dK < 1e-12, (name, dK)

        print('%-9s rotated  : diagonal %.6f  edge %.6f' %
              (name, got[('diagonal', True)], got[('edge', True)]))
        print('%-9s as-is    : diagonal %.6f  edge %.6f' %
              (name, got[('diagonal', False)], got[('edge', False)]))
        # NOTE with the prestress rotated into each element frame the answer
        #      is the same whichever way the triangles are numbered
        assert np.isclose(got[('diagonal', True)], got[('edge', True)],
                          rtol=1e-10), (name, got)
        # NOTE and the edge numbering needs no rotation, its element x axis
        #      already being along the load, so the two agree there as well
        assert np.isclose(got[('edge', False)], got[('edge', True)],
                          rtol=1e-10), (name, got)
        # NOTE whereas passing the components unrotated to the diagonally
        #      numbered triangles gives a different answer, which is the
        #      mistake this test exists to catch
        assert not np.isclose(got[('diagonal', False)],
                              got[('diagonal', True)], rtol=1e-3), (name, got)



def test_optional_stabilisation_is_off_by_default_and_mild_when_on():
    r"""``alpha_shear_locking`` on top of the discrete shear gap

    Source: the stabilisation is Eq. (26) of

        Castro, S. G. P., Donadon, M. V., and Guimaraes, T. A. M., 2019,
        "ES-PIM applied to buckling of variable angle tow laminates",
        Composite Structures, 209, pp. 67-78.
        https://doi.org/10.1016/j.compstruct.2018.10.058

    following

        Bischoff, M., and Bletzinger, K.-U., 2004, "Improving stability and
        accuracy of Reissner-Mindlin plate finite elements via algebraic
        subgrid scale stabilization", Computer Methods in Applied Mechanics
        and Engineering, 193(15-16), pp. 1517-1528.
        https://doi.org/10.1016/j.cma.2003.12.036

    In both papers the factor `1/(1 + \alpha \ell^2/h^2)` is applied on top
    of a discrete shear gap field, which is already free of locking, and the
    recommended `\alpha` is near 0.1. Tria3R applies the same factor to a
    centroid-sampled shear field, where it has to do the unlocking itself,
    which is why its default is 0.7. This test pins both halves of that
    statement, on the plate of tests/test_tria3r_natural_freq.py so that the
    geometry, material, mesh and analytical reference are the recorded ones.

    Two things are checked. First that the default is a strict no-op, to the
    bit, so that nothing about Tria3DSG changed by the attribute existing.
    Second that on top of the DSG the parameter is mild, while on the
    centroid-sampled element it is load bearing.

    """
    nx, ny = 9, 11
    a, b = 0.3, 0.5
    E, nu, rho, h = 203.e9, 0.33, 7.83e3, 0.01

    D = 2*h**3*E/(3*(1 - nu**2))
    wmn = (1/a**2 + 1/b**2)*np.sqrt(D*np.pi**4/(2*rho*h))/2

    def assemble(cls, datacls, probecls, alpha):
        data = datacls()
        probe = probecls()
        xtmp = np.linspace(0, a, nx)
        ytmp = np.linspace(0, b, ny)
        xmesh, ymesh = np.meshgrid(xtmp, ytmp)
        ncoords = np.vstack((xmesh.T.flatten(), ymesh.T.flatten(),
                             np.zeros_like(ymesh.T.flatten()))).T
        x, y = ncoords[:, 0], ncoords[:, 1]
        ncoords_flatten = ncoords.flatten()

        nids = 1 + np.arange(ncoords.shape[0])
        nid_pos = dict(zip(nids, np.arange(len(nids))))
        nids_mesh = nids.reshape(nx, ny)
        n1s = nids_mesh[:-1, :-1].flatten()
        n2s = nids_mesh[1:, :-1].flatten()
        n3s = nids_mesh[1:, 1:].flatten()
        n4s = nids_mesh[:-1, 1:].flatten()

        num_elements = 2*len(n1s)
        KC0r = np.zeros(data.KC0_SPARSE_SIZE*num_elements, dtype=INT)
        KC0c = np.zeros(data.KC0_SPARSE_SIZE*num_elements, dtype=INT)
        KC0v = np.zeros(data.KC0_SPARSE_SIZE*num_elements, dtype=DOUBLE)
        Mr = np.zeros(data.M_SPARSE_SIZE*num_elements, dtype=INT)
        Mc = np.zeros(data.M_SPARSE_SIZE*num_elements, dtype=INT)
        Mv = np.zeros(data.M_SPARSE_SIZE*num_elements, dtype=DOUBLE)
        N = DOF*nx*ny

        prop = isotropic_plate(thickness=h, E=E, nu=nu, rho=rho)

        init_k_KC0 = 0
        init_k_M = 0
        for n1, n2, n3, n4 in zip(n1s, n2s, n3s, n4s):
            for (a1, a2, a3) in ((n1, n2, n3), (n1, n3, n4)):
                el = cls(probe)
                el.n1, el.n2, el.n3 = a1, a2, a3
                el.c1 = DOF*nid_pos[a1]
                el.c2 = DOF*nid_pos[a2]
                el.c3 = DOF*nid_pos[a3]
                el.init_k_KC0 = init_k_KC0
                el.init_k_M = init_k_M
                if alpha is not None:
                    el.alpha_shear_locking = alpha
                el.update_rotation_matrix(ncoords_flatten)
                el.update_probe_xe(ncoords_flatten)
                el.update_KC0(KC0r, KC0c, KC0v, prop)
                el.update_M(Mr, Mc, Mv, prop, mtype=0)
                init_k_KC0 += data.KC0_SPARSE_SIZE
                init_k_M += data.M_SPARSE_SIZE

        KC0 = coo_matrix((KC0v, (KC0r, KC0c)), shape=(N, N)).tocsc()
        M = coo_matrix((Mv, (Mr, Mc)), shape=(N, N)).tocsc()

        bk = np.zeros(N, dtype=bool)
        check = (np.isclose(x, 0.) | np.isclose(x, a)
                 | np.isclose(y, 0.) | np.isclose(y, b))
        bk[0::DOF] = check
        bk[1::DOF] = check
        bk[2::DOF] = check
        bu = ~bk
        return KC0, M, bu, KC0v

    def error(cls, datacls, probecls, alpha):
        KC0, M, bu, _ = assemble(cls, datacls, probecls, alpha)
        omegan, _ = natural_frequency(KC0[bu, :][:, bu], M[bu, :][:, bu],
                                      num_eigvalues=3, tol=1e-9)
        return 100*(omegan[0] - wmn)/wmn

    # NOTE the default has to be invisible: assembling with the attribute
    #      untouched and with it explicitly zero must give the same bits
    _, _, _, v_untouched = assemble(Tria3DSG, Tria3DSGData, Tria3DSGProbe, None)
    _, _, _, v_zero = assemble(Tria3DSG, Tria3DSGData, Tria3DSGProbe, 0.)
    assert np.array_equal(v_untouched, v_zero)

    sweep = (0., 0.07, 0.1, 0.7)
    e_dsg = {al: error(Tria3DSG, Tria3DSGData, Tria3DSGProbe, al)
             for al in sweep}
    e_r = {al: error(Tria3R, Tria3RData, Tria3RProbe, al)
           for al in sweep}
    print('alpha      Tria3DSG   Tria3R')
    for al in sweep:
        print('%5.2f    %+8.2f%% %+8.2f%%' % (al, e_dsg[al], e_r[al]))

    # NOTE the discrete shear gap is already accurate with no parameter,
    #      whereas the centroid-sampled element is nearly twice too stiff.
    #      This is the locking, and it is what alpha is curing in Tria3R
    assert abs(e_dsg[0.]) < 5.
    assert e_r[0.] > 80.

    # NOTE at the literature value the stabilisation is a mild improvement
    #      on top of the DSG, and nowhere near enough on its own
    assert abs(e_dsg[0.1]) < abs(e_dsg[0.])
    assert e_r[0.1] > 25.

    # NOTE while Tria3R's default over-softens the DSG element, the two
    #      being different jobs for the same symbol
    assert e_dsg[0.7] < -3.
    assert abs(e_r[0.7]) < 5.

    # NOTE 0.07 is the value the attribute documentation recommends for a
    #      coarse mesh of a thin plate, being the lower bound of the range
    #      of Castro et al. It has to sit between alpha = 0 and 0.1 and to
    #      improve on alpha = 0, which is the whole basis of recommending
    #      it, and this is what pins the table in that docstring
    assert e_dsg[0.] > e_dsg[0.07] > e_dsg[0.1]
    assert abs(e_dsg[0.07]) < abs(e_dsg[0.])

    # NOTE and the price of it, which is why the default is still zero: the
    #      constant transverse shear state of
    #      test_constant_transverse_shear_state_is_exact is exact on this
    #      element, and any non-zero alpha degrades it by exactly
    #      factor/(1 + factor). Recorded here so that the cost quoted in
    #      the docstring cannot drift away from the code
    L, hh = 1.0, 0.01
    nc = np.array([[0., 0., 0.], [L, 0., 0.], [L, L, 0.]])
    ratios = {}
    for al in (0., 0.07):
        K, el, prop = element_K(Tria3DSG, Tria3DSGData, Tria3DSGProbe, nc,
                                hh, alpha=al)
        assert el.alpha_shear_locking == al
        c = 1.e-4
        ue = np.zeros(18)
        ue[2::DOF] = c*nc[:, 0]
        ratios[al] = ((0.5*ue.dot(K.dot(ue)))
                      / (0.5*prop.A55*c**2*el.area))
    factor = 0.07*(L*2**0.5)**2/hh**2
    print('constant shear energy ratio: alpha=0 %.15f  alpha=0.07 %.6e'
          % (ratios[0.], ratios[0.07]))
    print('predicted 1/(1 + factor), factor = %.1f: %.6e'
          % (factor, 1./(1. + factor)))
    assert np.isclose(ratios[0.], 1., rtol=1e-12)
    assert np.isclose(ratios[0.07], 1./(1. + factor), rtol=1e-10)

if __name__ == '__main__':
    test_rigid_body_modes_are_exact()
    test_rank_matches_tria3r_and_assembly_removes_the_spurious_modes()
    test_constant_transverse_shear_state_is_exact()
    test_kirchhoff_states_produce_no_parasitic_shear()
    test_membrane_constant_strain_patch_test()
    test_node_numbering_invariance()
    test_converges_where_tria3r_converges_to_the_wrong_answer()
    test_nonlinear_internal_forces_of_an_imposed_slope()
    test_transverse_shear_stays_out_of_the_nonlinear_terms()
    test_material_direction_matches_tria3r()
    test_unsymmetric_laminate_couples_membrane_and_bending_exactly()
    test_buckling_load_does_not_depend_on_triangle_node_numbering()
    test_optional_stabilisation_is_off_by_default_and_mild_when_on()
