r"""General 6x6 sectional stiffness matrix ``BeamProp.C`` of the beam elements

``BeamProp.C`` is the sectional stiffness matrix of the strains
`\{\bar{\varepsilon}, \bar{\gamma}_y, \bar{\gamma}_z, \kappa_x, \kappa_y,
\kappa_z\}`, with its 21 independent entries; when it is not set, it is
built from the 11 stiffness attributes, see the notes of
:class:`pyfe3d.beamprop.BeamProp`.

1. The default ``C`` is the matrix of the 11 attributes given in the notes
   of :class:`pyfe3d.beamprop.BeamProp`, and setting ``C`` to it gives the
   matrices and internal forces of the default bit by bit.
2. Cantilever clamped at `x = 0` with the loads ``f`` at the tip `x = L`,
   against the closed form of :func:`tip_closed_form`: with the
   extension-twist coupling ``C[0, 3]`` alone both elements reproduce it
   with any mesh; BeamC reproduces the six load cases with one element for
   any coupling, and BeamLR converges as the square of the element length.
3. Thin-walled box beam CUS of Yu et al. (2012), Tables 1-2 (VABS 3.2
   column), converted with :func:`pyfe3d.vabs.yu_to_pyfe3d` and
   :func:`pyfe3d.vabs.beamprop_from_vabs`: extension-twist and
   bending-shear couplings in a cantilever, against the same closed form.
4. ``KC0 + KCNL + KG`` is the Jacobian of the nonlinear internal forces for
   a section with all 21 entries of ``C`` non-zero.
5. Without couplings other than ``C[0, 3]``, BeamC keeps the closed-form
   expressions of the shape functions of Luo (2008); the general
   evaluation, triggered by a coupling of 1e-30 relative to the diagonal,
   gives the same arrays to round-off.
6. Mass matrices with couplings: the kinetic energy of the rigid-body
   motions is exact, and BeamC and BeamLR converge to the same natural
   frequencies of the CUS box cantilever.

Notes
-----
The closed form of item 2 is derived in :func:`tip_closed_form`: the
sectional forces are linear in `x`, the strains are ``inv(C)`` times them,
and the tip displacements follow by integrating the strains. With the
extension-twist coupling ``C[0, 3]`` alone, the strains of an axial load or
of a torque are constant and the tip twist under an axial load ``P`` is
``S[3, 0]*P*L``, ``S = inv(C)``. BeamC is exact for any coupling because its
shape functions are derived from the same equations for any ``C``, see the
notes of :class:`pyfe3d.beamc.BeamC`.

References
----------
Yu, W., Hodges, D. H., Ho, J. C., Variational asymptotic beam sectional
analysis - An updated version, International Journal of Engineering
Science, 59, 2012, pp. 40-64, https://doi.org/10.1016/j.ijengsci.2012.03.006.

Luo, Y., An Efficient 3D Timoshenko Beam Element with Consistent Shape
Functions, Adv. Theor. Appl. Mech., 1(3), 2008, pp. 95-106.
"""
import copy
import pickle
import sys
sys.path.append('..')

import numpy as np
import pytest
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

from pyfe3d.beamprop import BeamProp
from pyfe3d.vabs import yu_to_pyfe3d, beamprop_from_vabs
from pyfe3d import (BeamC, BeamCData, BeamCProbe, BeamLR, BeamLRData,
                    BeamLRProbe, Truss, TrussData, TrussProbe, DOF, INT,
                    DOUBLE)

BEAMS = {
    'BeamC': (BeamC, BeamCData, BeamCProbe),
    'BeamLR': (BeamLR, BeamLRData, BeamLRProbe),
}
LOADS = ['Px', 'Py', 'Pz', 'T', 'My', 'Mz']


def general_prop():
    """Property with every stiffness attribute non-zero, as in
    ``tests/test_beam_shear_moduli.py``, with two shear moduli"""
    p = BeamProp()
    p.A = 0.0032
    p.E = 71.e9
    p.G = 26.e9
    p.G12 = 21.e9
    p.G13 = 17.e9
    p.GJ = 3.1e4
    p.Iyy = 2.1e-6
    p.Izz = 3.7e-6
    p.Iyz = 0.4e-6
    p.J = 1.3e-6
    p.Ay = 0.9e-5
    p.Az = -0.6e-5
    p.intrho = 8.6
    p.intrhoy = 0.024
    p.intrhoz = -0.016
    p.intrhoy2 = 0.012
    p.intrhoz2 = 0.0072
    p.intrhoyz = 0.0011
    return p


def rectangle_C():
    """Diagonal sectional stiffness of a rectangle of 50 mm by 30 mm,
    aluminium, shear correction 5/6, torsional stiffness 0.2*G*(Iyy + Izz)"""
    b, h = 0.05, 0.03
    E, G = 70e9, 26e9
    A = b*h
    Izz = b*h**3/12
    Iyy = b**3*h/12
    return np.diag([E*A, 5/6*G*A, 5/6*G*A, 0.2*G*(Izz + Iyy), E*Izz, E*Iyy])


def coupled(C0, i, j, ratio=0.3):
    """``C0`` with the coupling ``C[i, j] = ratio*sqrt(C[i, i]*C[j, j])``"""
    C = C0.copy()
    C[i, j] = C[j, i] = ratio*np.sqrt(C0[i, i]*C0[j, j])
    return C


def CUS_C():
    r"""Sectional stiffness of the CUS box, convention of pyfe3d

    Table 2 of Yu et al. (2012), VABS 3.2 column, in lb, in., in the
    convention `\{\gamma_{11}, 2\gamma_{12}, 2\gamma_{13}, \kappa_1,
    \kappa_2, \kappa_3\}` of :mod:`pyfe3d.vabs`; "the other components are
    negligible", p. 56. The same entries are computed in
    ``tests/test_vabs_composite.py``.
    """
    Kt = np.zeros((6, 6))
    ref = {(0, 0): 1.437e6, (0, 3): 1.074e5, (1, 1): 9.027e4,
           (1, 4): -5.201e4, (2, 2): 3.943e4, (2, 5): -5.635e4,
           (3, 3): 1.679e4, (4, 4): 6.621e4, (5, 5): 1.725e5}
    for (i, j), v in ref.items():
        Kt[i, j] = Kt[j, i] = v
    return yu_to_pyfe3d(Kt)


def tip_closed_form(C, L, f):
    r"""Tip displacements of a cantilever with the sectional stiffness ``C``

    Derived here. Clamped at `x = 0`, loaded at `x = L` by ``f`` `=
    \{P_x, P_y, P_z, T, M_y, M_z\}`, work-conjugate to `\{u, v, w, r_x,
    r_y, r_z\}`. The strain energy `\frac{1}{2} \int_0^L
    \boldsymbol{\epsilon}^\top \boldsymbol{C} \boldsymbol{\epsilon} dx`,
    with `\boldsymbol{\epsilon} = \{u_{,x}, v_{,x} - r_z, w_{,x} + r_y,
    {r_x}_{,x}, -{r_z}_{,x}, {r_y}_{,x}\}`, gives with the virtual work the
    equilibrium of the sectional forces `\boldsymbol{F} = \boldsymbol{C}
    \boldsymbol{\epsilon} = \{N, V_y, V_z, M_x, M_4, M_5\}`: `N = P_x`, `V_y
    = P_y`, `V_z = P_z`, `M_x = T`, `M_4' = V_y`, `M_4(L) = -M_z`, `M_5' =
    V_z`, `M_5(L) = M_y`. Hence `\boldsymbol{F} = \boldsymbol{F}_0 +
    \boldsymbol{F}_1 x` and `\boldsymbol{\epsilon} = \boldsymbol{a} +
    \boldsymbol{b} x` with `\boldsymbol{a} = \boldsymbol{S}
    \boldsymbol{F}_0`, `\boldsymbol{b} = \boldsymbol{S} \boldsymbol{F}_1`,
    `\boldsymbol{S} = \boldsymbol{C}^{-1}`, and, integrating with zero
    displacements at `x = 0`, `r_z = -\int \epsilon_4`, `r_y = \int
    \epsilon_5`, `v = \int (\epsilon_1 + r_z)` and `w = \int (\epsilon_2 -
    r_y)`. Without couplings, `v(L) = P_y L^3/(3 C_{44}) + P_y L/C_{11}`,
    Eq. (1.2) of Bathe, Finite Element Procedures, 2014, p. 5.
    """
    S = np.linalg.inv(C)
    Px, Py, Pz, T, My, Mz = f
    F0 = np.array([Px, Py, Pz, T, -Mz - Py*L, My - Pz*L])
    F1 = np.array([0., 0., 0., 0., Py, Pz])
    a = S @ F0
    b = S @ F1
    u = a[0]*L + b[0]*L**2/2
    rx = a[3]*L + b[3]*L**2/2
    rz = -(a[4]*L + b[4]*L**2/2)
    ry = a[5]*L + b[5]*L**2/2
    v = a[1]*L + b[1]*L**2/2 - (a[4]*L**2/2 + b[4]*L**3/6)
    w = a[2]*L + b[2]*L**2/2 - (a[5]*L**2/2 + b[5]*L**3/6)
    return np.array([u, v, w, rx, ry, rz])


def tip_fe(name, n, prop, L, f):
    """Tip displacements of a cantilever of ``n`` elements along `x`"""
    cls, datacls, probecls = BEAMS[name]
    x = np.zeros(3*(n + 1))
    x[0::3] = np.linspace(0, L, n + 1)
    N = DOF*(n + 1)
    data = datacls()
    probe = probecls()
    size = n*data.KC0_SPARSE_SIZE
    r = np.zeros(size, dtype=INT)
    c = np.zeros(size, dtype=INT)
    v = np.zeros(size, dtype=DOUBLE)
    for i in range(n):
        elem = cls(probe)
        elem.n1, elem.n2 = i, i + 1
        elem.c1, elem.c2 = DOF*i, DOF*(i + 1)
        elem.init_k_KC0 = i*data.KC0_SPARSE_SIZE
        elem.update_rotation_matrix(0., 1., 0., x)
        elem.update_probe_xe(x)
        elem.update_KC0(r, c, v, prop)
    K = coo_matrix((v, (r, c)), shape=(N, N)).tocsc()
    fg = np.zeros(N)
    fg[-DOF:] = f
    bu = np.ones(N, dtype=bool)
    bu[:DOF] = False
    u = np.zeros(N)
    u[bu] = spsolve(K[bu][:, bu], fg[bu])
    return u[-DOF:]


def tip_error(name, n, C, L, load):
    """Largest error of the six tip displacements, relative to the largest"""
    prop = BeamProp()
    prop.C = C
    f = np.zeros(6)
    f[LOADS.index(load)] = 1.
    exact = tip_closed_form(C, L, f)
    return np.abs(tip_fe(name, n, prop, L, f) - exact).max()/np.abs(exact).max()


def element_arrays(name, prop):
    """All matrices and internal forces of one element in a generic
    orientation, as in ``tests/test_beam_shear_moduli.py``"""
    ncoords = np.array([0.1, -0.2, 0.05, 0.9, 0.35, -0.4])
    u = np.array([1.2e-3, -0.7e-3, 2.1e-3, 0.03, -0.02, 0.025,
                  -0.4e-3, 1.5e-3, -1.1e-3, -0.015, 0.035, -0.01])
    cls, datacls, probecls = dict(BEAMS, Truss=(Truss, TrussData,
                                                TrussProbe))[name]
    data = datacls()
    probe = probecls()
    elem = cls(probe)
    elem.n1, elem.n2 = 0, 1
    elem.c1, elem.c2 = 0, DOF
    if name == 'Truss':
        elem.update_rotation_matrix(ncoords)
    else:
        elem.update_rotation_matrix(0.3, 1., -0.2, ncoords)
    elem.update_probe_xe(ncoords)
    elem.update_probe_ue(u)
    out = {}

    def sparse(method, size, *args):
        r = np.zeros(size, dtype=INT)
        c = np.zeros(size, dtype=INT)
        v = np.zeros(size, dtype=DOUBLE)
        method(r, c, v, prop, *args)
        return v

    out['KC0v'] = sparse(elem.update_KC0, data.KC0_SPARSE_SIZE)
    out['M0v'] = sparse(elem.update_M, data.M_SPARSE_SIZE, 0)
    out['M1v'] = sparse(elem.update_M, data.M_SPARSE_SIZE, 1)
    if name == 'Truss':
        fint = np.zeros(2*DOF)
        elem.update_fint(fint, prop)
        out['fint'] = fint
        return out
    out['KGv'] = sparse(elem.update_KG, data.KG_SPARSE_SIZE)
    out['KCNLv'] = sparse(elem.update_KCNL, data.KCNL_SPARSE_SIZE)
    for nonlinear in [0, 1]:
        fint = np.zeros(2*DOF)
        elem.update_fint(fint, prop, nonlinear)
        out['fint%d' % nonlinear] = fint
    return out


def test_default_C_is_matrix_of_11_attributes():
    p = general_prop()
    E, A = p.E, p.A
    expected = np.zeros((6, 6))
    expected[0, 0] = E*A
    expected[0, 4] = expected[4, 0] = E*p.Ay
    expected[0, 5] = expected[5, 0] = E*p.Az
    expected[1, 1] = p.G12*A
    expected[2, 2] = p.G13*A
    expected[1, 3] = expected[3, 1] = -p.G12*p.Az
    expected[2, 3] = expected[3, 2] = p.G13*p.Ay
    expected[3, 3] = p.GJ
    expected[4, 4] = E*p.Izz
    expected[4, 5] = expected[5, 4] = E*p.Iyz
    expected[5, 5] = E*p.Iyy
    assert np.array_equal(p.C, expected)
    # NOTE C follows the attributes until it is set
    p.E = 2*E
    assert np.array_equal(p.C[0], 2*expected[0])
    assert np.array_equal(BeamProp().C, np.zeros((6, 6)))


@pytest.mark.parametrize('name', ['BeamC', 'BeamLR', 'Truss'])
def test_explicit_C_reproduces_default(name):
    """Setting ``C`` to its default gives the same arrays bit by bit, and
    deleting it returns to the attributes"""
    default = element_arrays(name, general_prop())
    p = general_prop()
    p.C = p.C
    explicit = element_arrays(name, p)
    for key in default:
        assert np.array_equal(explicit[key], default[key]), key
    # NOTE once set, C no longer follows the attributes
    p.E *= 2
    p.G12 *= 3
    p.Iyz *= 5
    for key, value in element_arrays(name, p).items():
        if not key.startswith('M'):
            assert np.array_equal(value, default[key]), key
    del p.C
    changed = element_arrays(name, p)
    assert not np.array_equal(changed['KC0v'], default['KC0v'])


def test_C_set_get_copy_and_errors():
    """``BeamProp.C`` returns a copy, stores the symmetric part, survives
    pickling and ``copy.deepcopy``, is reset to zero by ``del``, and raises
    ``ValueError`` for a wrong shape, an unsymmetric or a non-finite
    matrix"""
    p = BeamProp()
    C = rectangle_C()
    C[0, 3] = C[3, 0] = 1.e3
    p.C = C
    out = p.C
    assert np.array_equal(out, C)
    out[0, 0] = -1.
    assert p.C[0, 0] == C[0, 0]
    # NOTE the symmetric part is stored
    Cn = C.copy()
    Cn[0, 3] *= 1 + 1e-12
    p.C = Cn
    assert p.C[0, 3] == p.C[3, 0]
    for p2 in [pickle.loads(pickle.dumps(p)), copy.deepcopy(p)]:
        assert np.array_equal(p2.C, p.C)
        del p2.C
        assert np.array_equal(p2.C, np.zeros((6, 6)))
    with pytest.raises(ValueError, match='shape'):
        p.C = np.eye(5)
    with pytest.raises(ValueError, match='symmetric'):
        Cn[0, 3] *= 2
        p.C = Cn
    with pytest.raises(ValueError, match='finite'):
        p.C = np.full((6, 6), np.nan)


@pytest.mark.parametrize('name', ['BeamC', 'BeamLR'])
@pytest.mark.parametrize('num_elements', [1, 3])
def test_extension_twist_any_mesh(name, num_elements):
    """Extension-twist coupling ``C[0, 3]``: the tip twist of an axial load
    is ``S[3, 0]*P*L`` and the tip extension of a torque ``S[0, 3]*T*L``
    with any mesh; one BeamC reproduces all six load cases"""
    C = coupled(rectangle_C(), 0, 3)
    S = np.linalg.inv(C)
    L = 0.5
    prop = BeamProp()
    prop.C = C
    tip = tip_fe(name, num_elements, prop, L, [1., 0, 0, 0, 0, 0])
    assert np.isclose(tip[3], S[3, 0]*L, rtol=1e-12)
    assert np.isclose(tip[0], S[0, 0]*L, rtol=1e-12)
    tip = tip_fe(name, num_elements, prop, L, [0, 0, 0, 1., 0, 0])
    assert np.isclose(tip[0], S[0, 3]*L, rtol=1e-12)
    assert np.isclose(tip[3], S[3, 3]*L, rtol=1e-12)
    if name == 'BeamC':
        for load in LOADS:
            assert tip_error(name, num_elements, C, L, load) < 1e-12, load


def test_BeamC_exact_with_any_coupling():
    """One BeamC gives the closed form :func:`tip_closed_form` of the six
    tip loads for each single coupling of 0.3 times the square root of the
    product of the diagonal entries, for a ``C`` with all 21 entries
    non-zero and for the CUS box; measured on 2026-10-31, the largest
    error is 2.7e-15 of the largest tip displacement for the single
    couplings, 2.0e-15 for the 21 entries and 5.4e-16 for the CUS box,
    against 7.5e-3 to 8.9e-2 (single couplings other than ``C[0, 3]``) and
    0.42 (CUS box) with the closed-form shape functions of Luo (2008),
    which BeamC uses only without couplings other than ``C[0, 3]``, see
    :func:`test_general_shape_functions_uncoupled`"""
    C0 = rectangle_C()
    for i in range(6):
        for j in range(i + 1, 6):
            C = coupled(C0, i, j)
            err = max(tip_error('BeamC', 1, C, 0.5, load) for load in LOADS)
            assert err < 1e-12, (i, j, err)
    _, _, prop, x = make_coupled_element('BeamC')
    assert np.all(prop.C != 0)
    for C, L in [(prop.C, 0.4), (CUS_C(), 10.)]:
        err = max(tip_error('BeamC', 1, C, L, load) for load in LOADS)
        assert err < 1e-12, err


@pytest.mark.parametrize('i, j, load', [(1, 5, 'Pz'), (1, 5, 'My'),
                                        (1, 3, 'T'), (0, 4, 'Py'),
                                        (3, 5, 'Pz'), (4, 5, 'Py')])
def test_convergence_with_couplings(i, j, load):
    """BeamC is exact with any mesh, BeamLR converges as the square of the
    element length; measured errors of the bending-shear coupling ``C[1,
    5]`` with the tip load ``Pz``: BeamLR 8.3e-2, 5.2e-3, 3.3e-4 and 2.0e-5
    with 1, 4, 16 and 64 elements; BeamC at most 9.8e-13 with 16
    elements, the round-off of the solution, as BeamLR under ``My``"""
    C = coupled(rectangle_C(), i, j)
    L = 0.5
    for n in [1, 4, 16]:
        assert tip_error('BeamC', n, C, L, load) < 1e-11, n
    errLR = [tip_error('BeamLR', n, C, L, load) for n in [4, 16]]
    assert errLR[1] < 1e-12 or 14 < errLR[0]/errLR[1] < 18, errLR
    assert errLR[1] < 4e-4


@pytest.mark.parametrize('name', ['BeamC', 'BeamLR'])
def test_CUS_box_cantilever(name):
    """CUS box of Yu et al. (2012): ``beamprop_from_vabs`` keeps the
    extension-twist coupling, whose tip twist under an axial load is
    ``S[3, 0]*P*L`` with any mesh, and the bending-shear couplings, with
    which the tip displacements of transverse loads are exact with BeamC
    and converge to the closed form with BeamLR: measured error 2.3e-4
    with 32 elements of BeamLR"""
    Cs = CUS_C()
    # NOTE extension-twist and bending-shear couplings, convention of pyfe3d
    assert Cs[0, 3] > 0 and Cs[1, 5] < 0 and Cs[2, 4] > 0
    prop = beamprop_from_vabs(Cs, E=20.59e6)
    assert np.array_equal(prop.C, Cs)
    S = np.linalg.inv(Cs)
    L = 10.
    P = 100.
    for n in [1, 4]:
        tip = tip_fe(name, n, prop, L, [P, 0, 0, 0, 0, 0])
        assert np.isclose(tip[3], S[3, 0]*P*L, rtol=1e-12)
    n, tol = {'BeamC': (1, 1e-12), 'BeamLR': (32, 3e-4)}[name]
    for load in ['Py', 'Pz', 'My', 'Mz']:
        assert tip_error(name, n, Cs, L, load) < tol, load
    # NOTE without C, i.e. with the 11 attributes, the couplings are lost
    del prop.C
    tip = tip_fe(name, 4, prop, L, [P, 0, 0, 0, 0, 0])
    assert tip[3] == 0


def make_coupled_element(name):
    """One element arbitrarily oriented in space, with a sectional matrix
    whose 21 entries are all non-zero, positive definite"""
    rng = np.random.default_rng(5)
    q, _ = np.linalg.qr(rng.standard_normal((3, 3)))
    origin = np.array([0.3, -0.2, 0.5])
    x = np.concatenate([origin, origin + 0.4*q[:, 0]])
    C0 = rectangle_C()
    d = np.sqrt(np.diag(C0))
    R = 0.25*rng.uniform(-1, 1, (6, 6))
    R = (R + R.T)/2
    np.fill_diagonal(R, 1.)
    assert np.all(np.linalg.eigvalsh(R) > 0)
    prop = BeamProp()
    prop.C = R*np.outer(d, d)
    cls, datacls, probecls = BEAMS[name]
    elem = cls(probecls())
    elem.n1, elem.n2 = 1, 2
    elem.c1, elem.c2 = 0, DOF
    vxy = q[:, 1] + 0.3*q[:, 0]
    elem.update_rotation_matrix(vxy[0], vxy[1], vxy[2], x)
    elem.update_probe_xe(x)
    return elem, datacls(), prop, x


@pytest.mark.parametrize('name', ['BeamC', 'BeamLR'])
def test_tangent_coupled_section(name):
    """``KC0 + KCNL + KG`` against a central difference of
    ``update_fint(..., nonlinear=1)``, as in
    ``tests/test_tangent_consistency.py``, and ``fint`` with
    ``nonlinear=0`` against ``KC0 @ u``"""
    elem, data, prop, x = make_coupled_element(name)
    n = 2*DOF

    def assemble(size, update):
        r = np.zeros(size, dtype=INT)
        c = np.zeros(size, dtype=INT)
        v = np.zeros(size, dtype=DOUBLE)
        update(r, c, v)
        return coo_matrix((v, (r, c)), shape=(n, n)).toarray()

    def fint(u, nonlinear=1):
        f = np.zeros(n)
        elem.update_probe_ue(u)
        elem.update_fint(f, prop, nonlinear=nonlinear)
        return f

    rng = np.random.default_rng(7)
    u = 0.01*rng.standard_normal(n)
    KC0 = assemble(data.KC0_SPARSE_SIZE,
                   lambda r, c, v: elem.update_KC0(r, c, v, prop))
    assert np.linalg.norm(fint(u, 0) - KC0 @ u) < 1e-12*np.linalg.norm(KC0 @ u)
    elem.update_probe_ue(u)
    KCNL = assemble(data.KCNL_SPARSE_SIZE,
                    lambda r, c, v: elem.update_KCNL(r, c, v, prop))
    KG = assemble(data.KG_SPARSE_SIZE,
                  lambda r, c, v: elem.update_KG(r, c, v, prop))
    KT = KC0 + KCNL + KG
    step = 1e-8
    J = np.empty((n, n))
    for j in range(n):
        e = np.zeros(n)
        e[j] = step
        J[:, j] = (fint(u + e) - fint(u - e))/(2*step)
    err = np.linalg.norm(KT - J)/np.linalg.norm(J)
    assert err < 1e-6, err
    assert np.linalg.norm(KT - KT.T) < 1e-12*np.linalg.norm(KT)
    # NOTE the couplings of the axial strain with the shear strains and the
    #      twist, absent from the 11 attributes, carry weight in KG
    C = prop.C
    C[0, 1:4] = C[1:4, 0] = 0
    prop.C = C
    elem.update_probe_ue(u)
    KG0 = assemble(data.KG_SPARSE_SIZE,
                   lambda r, c, v: elem.update_KG(r, c, v, prop))
    assert np.linalg.norm(KG0 - KG) > 1e-4*np.linalg.norm(KG)


@pytest.mark.parametrize('base', ['diagonal', 'extension-twist',
                                  'attributes'])
def test_general_shape_functions_uncoupled(base):
    """Without couplings other than ``C[0, 3]``, BeamC uses the closed-form
    shape functions of Luo (2008); a coupling ``C[1, 5]`` of 1e-30 times
    the square root of ``C[1, 1]*C[5, 5]`` switches to their general
    evaluation from ``inv(C)``, see the notes of :class:`pyfe3d.beamc.BeamC`,
    which gives the same matrices and internal forces to round-off;
    measured on 2026-10-31 up to 1.3e-14 of the largest entry, for the
    nonlinear stiffness matrix KCNL"""
    p = general_prop()
    if base == 'attributes':
        # NOTE the 11 attributes without the couplings of Ay, Az and Iyz
        p.Ay = p.Az = p.Iyz = 0.
        C0 = p.C
    else:
        C0 = rectangle_C()
        if base == 'extension-twist':
            C0 = coupled(C0, 0, 3)
    p.C = C0
    closed = element_arrays('BeamC', p)
    Ct = C0.copy()
    Ct[1, 5] = Ct[5, 1] = 1e-30*np.sqrt(C0[1, 1]*C0[5, 5])
    p.C = Ct
    general = element_arrays('BeamC', p)
    for key in closed:
        err = np.abs(general[key] - closed[key]).max()/np.abs(closed[key]).max()
        assert err < 1e-13, (key, err)


def update_calls(name, elem, data, prop):
    """The methods of ``elem`` that read ``C``: every method of BeamC, and
    all but ``update_M`` for BeamLR"""
    size = data.KC0_SPARSE_SIZE
    r = np.zeros(size, dtype=INT)
    c = np.zeros(size, dtype=INT)
    v = np.zeros(size, dtype=DOUBLE)
    calls = [lambda: elem.update_KC0(r, c, v, prop),
             lambda: elem.update_KG(r, c, v, prop),
             lambda: elem.update_KCNL(r, c, v, prop),
             lambda: elem.update_fint(np.zeros(2*DOF), prop, 1)]
    if name == 'BeamC':
        calls += [lambda: elem.update_M(r, c, v, prop, 0),
                  lambda: elem.update_M(r, c, v, prop, 1)]
    return calls


@pytest.mark.parametrize('name', ['BeamC', 'BeamLR'])
def test_C_not_positive_semidefinite(name):
    """``C`` must be positive semi-definite, see
    :class:`pyfe3d.beamprop.BeamProp`: an indefinite ``C``, a negative
    diagonal entry and a zero diagonal entry with a coupling raise in every
    method that reads ``C``, whether ``C`` is set or built from the 11
    attributes; a planar property with ``GJ = 0`` and ``Iyy = 0`` does not
    raise"""
    elem, data, prop, x = make_coupled_element(name)
    calls = update_calls(name, elem, data, prop)
    C0 = rectangle_C()
    negative = C0.copy()
    negative[3, 3] = -negative[3, 3]
    zero = C0.copy()
    zero[3, 3] = 0.
    zero[1, 3] = zero[3, 1] = 1e-3*C0[1, 1]
    for C in [coupled(C0, 1, 5, ratio=1.5), negative, zero]:
        prop.C = C
        for call in calls:
            with pytest.raises(ValueError, match='positive semi-definite'):
                call()
    del prop.C
    prop.A, prop.E, prop.G, prop.Izz = 1e-3, 70e9, 26e9, 1e-7
    prop.GJ = -1.
    for call in calls:
        with pytest.raises(ValueError, match='positive semi-definite'):
            call()
    del prop.GJ
    assert prop.GJ == 0 and prop.Iyy == 0
    for call in calls:
        call()


@pytest.mark.parametrize('name', ['BeamC', 'BeamLR'])
def test_coupled_C_singular(name):
    """A positive semi-definite but singular ``C``, here with ``C[1, 5] =
    sqrt(C[1, 1]*C[5, 5])``, is accepted by BeamLR; BeamC, whose shape
    functions with couplings other than ``C[0, 3]`` need ``inv(C)``,
    raises, except in its lumped mass matrix, which does not use them"""
    elem, data, prop, x = make_coupled_element(name)
    prop.C = coupled(rectangle_C(), 1, 5, ratio=1.)
    calls = update_calls(name, elem, data, prop)
    if name == 'BeamC':
        calls.pop()()
    for call in calls:
        if name == 'BeamC':
            with pytest.raises(ValueError, match='must be positive definite '
                               'when it has couplings'):
                call()
        else:
            call()


def coupled_mass(prop):
    """Mass integrals with all couplings non-zero"""
    prop.intrho = 8.6
    prop.intrhoy = 0.024
    prop.intrhoz = -0.016
    prop.intrhoy2 = 0.012
    prop.intrhoz2 = 0.0072
    prop.intrhoyz = 0.0011


@pytest.mark.parametrize('name', ['BeamC', 'BeamLR'])
def test_coupled_mass_rigid_body(name):
    r"""The consistent mass matrix gives the exact kinetic energy of the six
    rigid-body motions about node 1 for the section with 21 non-zero
    entries of ``C``: ``Phi.T @ M @ Phi`` equals the integral along the
    element of ``D.T @ Mrho @ D``, computed with Simpson's rule, exact for
    the quadratic integrand, where ``D`` gives the rigid-body displacements
    `\{u, v, w, r_x, r_y, r_z\}` at `x` and ``Mrho`` holds the mass
    integrals of the kinetic energy in the notes of
    :meth:`pyfe3d.beamc.BeamC.update_M`"""
    elem, data, prop, x = make_coupled_element(name)
    coupled_mass(prop)
    size = data.M_SPARSE_SIZE
    r = np.zeros(size, dtype=INT)
    c = np.zeros(size, dtype=INT)
    v = np.zeros(size, dtype=DOUBLE)
    elem.update_M(r, c, v, prop, 0)
    M = coo_matrix((v, (r, c)), shape=(2*DOF, 2*DOF)).toarray()
    Mrho = np.zeros((6, 6))
    Mrho[0, 0] = Mrho[1, 1] = Mrho[2, 2] = prop.intrho
    Mrho[3, 3] = prop.intrhoy2 + prop.intrhoz2
    Mrho[4, 4] = prop.intrhoz2
    Mrho[5, 5] = prop.intrhoy2
    Mrho[4, 5] = Mrho[5, 4] = -prop.intrhoyz
    Mrho[0, 5] = Mrho[5, 0] = -prop.intrhoy
    Mrho[0, 4] = Mrho[4, 0] = prop.intrhoz
    Mrho[1, 3] = Mrho[3, 1] = -prop.intrhoz
    Mrho[2, 3] = Mrho[3, 2] = prop.intrhoy

    def D(xi):
        """element displacements of the translations and rotations about
        node 1, in element coordinates, at the distance ``xi`` from node 1"""
        out = np.eye(6)
        out[1, 5] = xi
        out[2, 4] = -xi
        return out

    L = elem.length
    exact = L/6*(D(0).T @ Mrho @ D(0) + 4*D(L/2).T @ Mrho @ D(L/2)
                 + D(L).T @ Mrho @ D(L))
    R = np.array([[elem.r11, elem.r12, elem.r13],
                  [elem.r21, elem.r22, elem.r23],
                  [elem.r31, elem.r32, elem.r33]])
    T = np.kron(np.eye(4), R)
    Phi = T @ np.vstack([D(0), D(L)])
    err = np.abs(Phi.T @ M @ Phi - exact).max()/np.abs(exact).max()
    assert err < 1e-12, err


def test_coupled_frequencies_CUS():
    """Natural frequencies of the CUS box cantilever, 10 in. long, with the
    mass integrals of :func:`coupled_mass` scaled by 1e-5: BeamC converges
    as the square of the element length to the frequencies of BeamLR,
    extrapolated with Richardson's rule from 256 and 1024 elements;
    measured on 2026-10-31, the relative errors of the first eight
    frequencies are 3.2e-5 to 1.7e-2 with 16 BeamC elements and 2.0e-6 to
    1.1e-3 with 64, the error of 1024 BeamLR elements being at most 1.1e-5.
    Without the couplings, i.e. with the diagonal of ``C`` and of the mass
    integrals, BeamC, with the shape functions of Luo (2008), converges at
    the same rate"""
    from scipy.sparse.linalg import eigsh
    prop = beamprop_from_vabs(CUS_C(), E=20.59e6)
    coupled_mass(prop)
    for attr in ['intrho', 'intrhoy', 'intrhoz', 'intrhoy2', 'intrhoz2',
                 'intrhoyz']:
        setattr(prop, attr, 1e-5*getattr(prop, attr))
    L = 10.

    def frequencies(name, n):
        cls, datacls, probecls = BEAMS[name]
        x = np.zeros(3*(n + 1))
        x[0::3] = np.linspace(0, L, n + 1)
        N = DOF*(n + 1)
        data = datacls()
        probe = probecls()
        size = n*data.KC0_SPARSE_SIZE
        arrays = {key: (np.zeros(size, dtype=INT), np.zeros(size, dtype=INT),
                        np.zeros(size, dtype=DOUBLE)) for key in 'KM'}
        for i in range(n):
            elem = cls(probe)
            elem.c1, elem.c2 = DOF*i, DOF*(i + 1)
            elem.init_k_KC0 = elem.init_k_M = i*data.KC0_SPARSE_SIZE
            elem.update_rotation_matrix(0., 1., 0., x)
            elem.update_probe_xe(x)
            elem.update_KC0(*arrays['K'], prop)
            elem.update_M(*arrays['M'], prop, 0)
        K, M = [coo_matrix((v, (r, c)), shape=(N, N)).tocsc()[DOF:, DOF:]
                for r, c, v in [arrays['K'], arrays['M']]]
        w2 = eigsh(K, k=8, M=M, sigma=0., which='LM',
                   return_eigenvectors=False)
        return np.sqrt(np.sort(w2))

    ref = (16*frequencies('BeamLR', 1024) - frequencies('BeamLR', 256))/15
    err16 = np.abs(frequencies('BeamC', 16)/ref - 1)
    err64 = np.abs(frequencies('BeamC', 64)/ref - 1)
    assert np.all(err64 < 1.5e-3), err64
    assert np.all((14 < err16/err64) & (err16/err64 < 18)), err16/err64


if __name__ == '__main__':
    C0 = rectangle_C()
    for i, j, load in [(1, 5, 'My'), (1, 5, 'Pz')]:
        C = coupled(C0, i, j)
        for name in BEAMS:
            print(name, i, j, load, ' '.join(
                '%.2e' % tip_error(name, n, C, 0.5, load)
                for n in [1, 4, 16, 64, 256]))
