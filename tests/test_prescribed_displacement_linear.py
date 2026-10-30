r"""Linear static analysis with a prescribed displacement: a cantilever of
:class:`.BeamC` elements, clamped at `x = 0`, with the transverse deflection
`\delta` prescribed at the free end and no applied load.

The tip reaction, the clamp reactions and the deflected shape are compared
with the closed form of the Timoshenko beam, derived in
``doc/source/tut-prescribed-displacement-cantilever.ipynb`` and checked with
SymPy in :func:`test_closed_form_timoshenko_cantilever`. :class:`.BeamC`
solves the homogeneous equilibrium equations of the Timoshenko beam exactly,
such that the nodal values agree to round-off.
"""
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

from pyfe3d import BeamC, BeamCData, BeamCProbe, DOF, INT, DOUBLE
from pyfe3d.beamprop import BeamProp
from pyfe3d.solver import solve, Analysis


def cantilever_prop():
    r"""Rectangular aluminium cross section, `L/h = 5`

    Returns the property, with the shear correction factor `k = 5/6` of the
    rectangle applied to ``G13``, see :class:`.BeamProp`, the length ``L``
    and the prescribed deflection ``delta``, in SI units.
    """
    L = 1. # m
    b, h = 0.1, 0.2 # m, width along y, height along z
    E, nu = 70.e9, 0.3 # Pa
    k = 5/6
    prop = BeamProp()
    prop.A = b*h
    prop.E = E
    prop.G = E/(2*(1 + nu))
    prop.G12 = prop.G13 = k*prop.G
    prop.Iyy = b*h**3/12 # bending with w, in the plane xz
    prop.Izz = h*b**3/12
    prop.J = prop.Iyy + prop.Izz
    delta = 0.01 # m
    return prop, L, delta


def cantilever(prop, L, num_elements=10):
    r"""Nodes, elements and ``KC0`` of a cantilever along `X`

    Returns ``x`` (nodal positions along `X`), ``ncoords``, the list of
    elements, the probe and ``KC0`` (CSC), of size ``N = DOF*(n + 1)``.
    """
    data = BeamCData()
    probe = BeamCProbe()
    nodes = np.linspace(0, L, num_elements + 1)
    ncoords = np.vstack((nodes, 0*nodes, 0*nodes)).T.flatten()
    N = DOF*(num_elements + 1)
    KC0r = np.zeros(data.KC0_SPARSE_SIZE*num_elements, dtype=INT)
    KC0c = np.zeros(data.KC0_SPARSE_SIZE*num_elements, dtype=INT)
    KC0v = np.zeros(data.KC0_SPARSE_SIZE*num_elements, dtype=DOUBLE)
    elements = []
    for i in range(num_elements):
        beam = BeamC(probe)
        beam.init_k_KC0 = i*data.KC0_SPARSE_SIZE
        beam.n1, beam.n2 = i, i + 1
        beam.c1, beam.c2 = DOF*i, DOF*(i + 1)
        # NOTE element y along global Y, element z along global Z
        beam.update_rotation_matrix(0., 1., 0., ncoords)
        beam.update_probe_xe(ncoords)
        beam.update_KC0(KC0r, KC0c, KC0v, prop)
        elements.append(beam)
    KC0 = coo_matrix((KC0v, (KC0r, KC0c)), shape=(N, N)).tocsc()
    return nodes, ncoords, elements, KC0


def boundary_conditions(N, delta):
    r"""Clamp at the first node and `w = \delta` at the last one

    Returns ``bk``, ``bu`` and ``uk`` (size ``N``, with ``delta`` at the
    `w` of the tip and zeros elsewhere), and the index ``tip_w`` of that dof.
    """
    bk = np.zeros(N, dtype=bool)
    bk[:DOF] = True # clamp: all six dofs of node 0
    tip_w = N - DOF + 2
    bk[tip_w] = True
    bu = ~bk
    uk = np.zeros(N)
    uk[tip_w] = delta
    return bk, bu, uk, tip_w


def timoshenko_cantilever(x, P, prop, L):
    r"""Closed form of a Timoshenko cantilever with a tip force `P`

    Returns the deflection `w(x) = \frac{P}{EI}\left(\frac{L x^2}{2} -
    \frac{x^3}{6}\right) + \frac{P x}{k G A}` and the rotation of the cross
    section `\theta(x) = \frac{P}{EI}\left(L x - \frac{x^2}{2}\right)`, with
    `w_{,x} = \theta + \gamma`, `\gamma = P/(k G A)`, `EI` = ``E*Iyy`` and
    `kG` = ``G13``. In pyfe3d, `r_y = -\theta`, see :doc:`conventions`.
    """
    EI = prop.E*prop.Iyy
    kGA = prop.G13*prop.A
    w = P/EI*(L*x**2/2 - x**3/6) + P*x/kGA
    theta = P/EI*(L*x - x**2/2)
    return w, theta


def tip_force(delta, prop, L):
    r"""`P = \delta / (L^3/(3EI) + L/(kGA))`, and the Euler-Bernoulli value
    `3 E I \delta / L^3`"""
    EI = prop.E*prop.Iyy
    kGA = prop.G13*prop.A
    return delta/(L**3/(3*EI) + L/kGA), 3*EI*delta/L**3


def internal_forces(elements, prop, u, ncoords):
    r"""``update_fint`` of every element, linear, in global coordinates"""
    fint = np.zeros_like(u)
    for beam in elements:
        beam.update_probe_xe(ncoords)
        beam.update_probe_ue(u)
        beam.update_fint(fint, prop)
    return fint


def solve_by_partition(KC0, bu, bk, uk, f):
    r"""`K_{uu} u_u = f_u - K_{uk} u_k`, then `u` of size ``N``"""
    KC0uu = KC0[bu, :][:, bu]
    KC0uk = KC0[bu, :][:, bk]
    u = np.zeros(KC0.shape[0])
    u[bk] = uk[bk]
    u[bu] = solve(KC0uu, f[bu] - KC0uk @ uk[bk])
    return u


def check_reactions(R, nodes, L, P, rtol):
    r"""Tip reaction `P` and equilibrium of the reactions of the clamp"""
    N = R.size
    tip_w = N - DOF + 2
    assert np.isclose(R[tip_w], P, rtol=rtol)
    # NOTE forces: the clamp balances the tip reaction
    Fclamp = R[0:3]
    Mclamp = R[3:6]
    assert np.allclose(Fclamp, [0, 0, -P], rtol=rtol, atol=rtol*P)
    # NOTE moments about the origin: Mclamp + (L, 0, 0) x (0, 0, P) = 0,
    #      i.e. My of the clamp is L*P
    assert np.allclose(Mclamp + np.cross([L, 0, 0], [0, 0, R[tip_w]]), 0,
                       atol=rtol*P*L)
    assert np.isclose(Mclamp[1], P*L, rtol=rtol)
    # NOTE no reaction at the free dofs
    free = np.ones(N, dtype=bool)
    free[:DOF] = False
    free[tip_w] = False
    assert np.allclose(R[free], 0, atol=1e-9*P*L)


def test_partition_by_hand():
    r"""Partition with SciPy: the tip reaction, the clamp reactions and the
    deflected shape agree with the closed form of the Timoshenko beam to
    round-off, and the Euler-Bernoulli tip force is 3.1 % larger"""
    prop, L, delta = cantilever_prop()
    nodes, ncoords, elements, KC0 = cantilever(prop, L)
    N = KC0.shape[0]
    bk, bu, uk, tip_w = boundary_conditions(N, delta)
    f = np.zeros(N) # no applied load
    u = solve_by_partition(KC0, bu, bk, uk, f)
    P, P_EB = tip_force(delta, prop, L)
    # NOTE reactions from the second row of the partition and from
    #      update_fint, step 7 of the overview
    R_partition = np.zeros(N)
    R_partition[bk] = (KC0[bk, :][:, bu] @ u[bu] + KC0[bk, :][:, bk] @ uk[bk]
                       - f[bk])
    R_fint = internal_forces(elements, prop, u, ncoords) - f
    print('P = %.10e, FE %.10e, Euler-Bernoulli %.10e' % (P, R_fint[tip_w],
                                                          P_EB))
    assert np.allclose(R_partition[bk], R_fint[bk], rtol=1e-12,
                       atol=1e-12*P*L)
    check_reactions(R_fint, nodes, L, P, rtol=1e-9)
    w_ref, theta_ref = timoshenko_cantilever(nodes, P, prop, L)
    assert np.allclose(u[2::DOF], w_ref, rtol=0, atol=1e-10*delta)
    assert np.allclose(u[4::DOF], -theta_ref, rtol=0,
                       atol=1e-10*abs(theta_ref).max())
    # NOTE other dofs remain null
    for i in [0, 1, 3, 5]:
        assert np.allclose(u[i::DOF], 0, atol=1e-14*delta)
    # NOTE shear flexibility: 3 E Iyy/(L**2 kGA) = (E/(4 k G)) (h/L)**2
    assert np.isclose(P_EB/P - 1, 0.0312, atol=0.0001)


def test_analysis_bu_uk():
    r"""The linear :meth:`.Analysis.static` with ``bu`` and ``uk`` gives the
    solution of the partition by hand, with the prescribed displacement in
    ``cs[0]``"""
    prop, L, delta = cantilever_prop()
    nodes, ncoords, elements, KC0 = cantilever(prop, L)
    N = KC0.shape[0]
    bk, bu, uk, tip_w = boundary_conditions(N, delta)

    def calc_fext(inc=1., silent=True):
        return np.zeros(N)

    def calc_kC(c=None, NLgeom=False, silent=True):
        return KC0

    def calc_fint(c, silent=True):
        return internal_forces(elements, prop, c, ncoords)

    def calc_kG(c=None, NLgeom=False, silent=True):
        return csr_matrix((N, N))

    an = Analysis(calc_fext, calc_fint, calc_kC, calc_kG, bu=bu, uk=uk)
    increments, cs = an.static(NLgeom=False, silent=True)
    assert increments == [1.]
    u = cs[0]
    assert u[tip_w] == delta
    u_hand = solve_by_partition(KC0, bu, bk, uk, np.zeros(N))
    assert np.allclose(u, u_hand, rtol=1e-12, atol=1e-14*delta)
    R = calc_fint(u) - calc_fext()
    P, _ = tip_force(delta, prop, L)
    check_reactions(R, nodes, L, P, rtol=1e-9)


def test_closed_form_timoshenko_cantilever():
    r"""SymPy check of :func:`timoshenko_cantilever`: `EI \theta_{,x} = M =
    P (L - x)`, `k G A (w_{,x} - \theta) = P`, `w(0) = \theta(0) = 0`, and
    `w(L) = \delta` gives :func:`tip_force`"""
    import sympy as sp
    x, L, P, EI, kGA = sp.symbols('x L P EI kGA', positive=True)
    w = P/EI*(L*x**2/2 - x**3/6) + P*x/kGA
    theta = P/EI*(L*x - x**2/2)
    assert sp.simplify(EI*sp.diff(theta, x) - P*(L - x)) == 0
    assert sp.simplify(kGA*(sp.diff(w, x) - theta) - P) == 0
    # NOTE equilibrium of the shear force: dQ/dx = 0, dM/dx = -Q
    assert sp.simplify(sp.diff(kGA*(sp.diff(w, x) - theta), x)) == 0
    assert sp.simplify(sp.diff(EI*sp.diff(theta, x), x) + P) == 0
    assert w.subs(x, 0) == 0 and theta.subs(x, 0) == 0
    delta = sp.symbols('delta', positive=True)
    Psol = sp.solve(sp.Eq(w.subs(x, L), delta), P)[0]
    assert sp.simplify(Psol - delta/(L**3/(3*EI) + L/kGA)) == 0
    assert sp.limit(Psol, kGA, sp.oo) == 3*EI*delta/L**3


if __name__ == '__main__':
    test_partition_by_hand()
    test_analysis_bu_uk()
    test_closed_form_timoshenko_cantilever()
