r"""Snap-through of a low sinusoidal arch of BeamC elements, Fung and Kaplan (1952)

Fung and Kaplan [1]_ give the equilibrium of a low arch hinged at both ends,
whose initial and deflected center lines are `y_0 = \sum a_m L \sin(m \pi
x/L)` and `y = \sum b_m L \sin(m \pi x/L)`, in terms of the nondimensional
rises and load of their Eq. (14),

.. math::

    \lambda_m = \frac{a_m L}{2}\sqrt{\frac{A}{I}}, \qquad
    B_m = \frac{b_m L}{2}\sqrt{\frac{A}{I}}, \qquad
    R = \frac{q_0 L^4}{2 \pi^4 E I}\sqrt{\frac{A}{I}}

For the sinusoidal arch under the sinusoidal load `q = q_0 \sin(\pi x/L)`,
positive downward, their Eq. (21), with rigid supports and without initial
thrust, the symmetric path is their Eq. (23), `B_1^3 - B_1(\lambda_1^2 - 1)
= \lambda_1 - R`, and its upper limit point their Eq. (25), `R_{cr} =
\lambda_1 + \sqrt{4(\lambda_1^2 - 1)^3/27}`; the lower limit point of Eq.
(23), at `B_1 = -\sqrt{(\lambda_1^2 - 1)/3}`, is at `\lambda_1 -
\sqrt{4(\lambda_1^2 - 1)^3/27}`. With `\lambda_1^2 < 5.5` the arch buckles
at the limit point, their Eq. (29).

The model: span ``L = 100``, square cross-section ``1 x 1``, ``E = 1e7``,
``nu = 0.3``, `\lambda_1 = 2.2`, i.e. a rise ``a_1 L = 1.27``, 40 BeamC
elements in the ``xz`` plane, the out-of-plane displacement and the rotations
about ``x`` and ``z`` restrained at every node, ``u`` and ``w`` restrained at
both ends; the load is lumped at the nodes with the weights of the
trapezoidal rule. `B_1` is computed from the deflection at mid-span. The
reference load is ``1.25`` times `R_{cr}`, such that the load factor 1.0 is
reached on the inverted configuration only, after both limit points and two
null loads.

At the default ``relTOL = 1e-9`` the arc-length solvers pass the null load
between the limit points only with the round-off floor of the convergence
criteria, see :mod:`pyfe3d.solver.convergence`: without it, they stop near
the load factor 0.03, the residual of the large self-equilibrated axial
forces being then larger than ``relTOL`` times the references of the
criteria.

References
----------
.. [1] Fung, Y. C. and Kaplan, A., Buckling of low arches or curved beams of
   small curvature, NACA Technical Note 2840, National Advisory Committee
   for Aeronautics, Washington, 1952.
   https://ntrs.nasa.gov/citations/19930083579

"""
import sys
sys.path.append('..')

import numpy as np
import pytest

from pyfe3d import BeamC, BeamCData, BeamCProbe, DOF
from pyfe3d.beamprop import BeamProp
from pyfe3d.solver import arc_length, convergence, newton_raphson

from test_solver_nonlinear_static import Model

METHODS = ['arc_length_riks', 'arc_length_crisfield']
CONVERGENCE = ['crisfield', 'relative']


def limit_loads(lam1):
    r"""Upper and lower limit loads of the symmetric path, Eq. (25) of Fung
    and Kaplan and the other extreme of their Eq. (23)"""
    d = np.sqrt(4/27*(lam1**2 - 1)**3)
    return lam1 + d, lam1 - d


def arch(lam1, num_elements=40, L=100., t=1., b=1., E=1.e7, nu=0.3,
         R_ref=1.):
    r"""Pinned sinusoidal arch of BeamC elements under a sinusoidal load

    Returns the :class:`Model`, the mask of the free degrees-of-freedom, the
    function that gives `B_1` from the displacements and the load `R` of the
    reference load vector, ``R_ref``.
    """
    A = b*t
    I = b*t**3/12
    rA = np.sqrt(A/I)
    a1L = 2*lam1/rA
    q0 = R_ref*2*np.pi**4*E*I/(L**4*rA)
    prop = BeamProp()
    prop.A = A
    prop.E = E
    prop.G = E/2/(1 + nu)
    prop.Iyy = I # bending in the xz plane
    prop.Izz = t*b**3/12
    prop.J = b*t**3/3
    num_nodes = num_elements + 1
    xs = np.linspace(0, L, num_nodes)
    x = np.vstack((xs, 0*xs, a1L*np.sin(np.pi*xs/L))).T.flatten()
    data = BeamCData()
    probe = BeamCProbe()
    elements = []
    for k in range(num_elements):
        elem = BeamC(probe)
        elem.n1, elem.n2 = k, k + 1
        elem.c1, elem.c2 = DOF*k, DOF*(k + 1)
        elem.init_k_KC0 = k*data.KC0_SPARSE_SIZE
        elem.init_k_KCNL = k*data.KCNL_SPARSE_SIZE
        elem.init_k_KG = k*data.KG_SPARSE_SIZE
        elem.update_rotation_matrix(0., 1., 0., x)
        elem.update_probe_xe(x)
        elements.append(elem)
    N = DOF*num_nodes
    bk = np.zeros(N, dtype=bool)
    # NOTE planar problem: v, rx and rz restrained at every node
    bk[1::DOF] = True
    bk[3::DOF] = True
    bk[5::DOF] = True
    # NOTE hinged ends, u and w restrained
    for n in (0, num_nodes - 1):
        bk[DOF*n + 0] = True
        bk[DOF*n + 2] = True
    fext = np.zeros(N)
    weights = np.full(num_nodes, L/num_elements)
    weights[[0, -1]] /= 2
    fext[2::DOF] = -q0*np.sin(np.pi*xs/L)*weights
    model = Model(elements, data, prop, x, N, fext)
    mid = DOF*(num_elements//2) + 2

    def B1(u):
        return (a1L + u[mid])/2*rA

    return model, ~bk, B1


@pytest.mark.parametrize('convergence_criterion', CONVERGENCE)
@pytest.mark.parametrize('method', METHODS)
def test_fung_kaplan_beam_arch(method, convergence_criterion, monkeypatch):
    r"""Symmetric path through both limit points at the default tolerances

    The limit loads of 40 elements differ from those of Fung and Kaplan by
    -0.0094 (upper) and +0.040 (lower), measured with arc-length increments
    of 0.025; with 80 and 160 elements these differences decrease to
    -0.0053 and +0.0087, and to -0.0042 and +0.0007. The converged point of
    extreme load is within about `3 B_{1,lim} h^2 + h^3 = 0.006` of the
    limit load of the model, with `B_{1,lim} = 1.13` and `h = 0.042` the
    largest increment of `B_1` of a step, as in Eq. (23). Measured on
    2026-10-29 with the increments of this test, 0.05, the same for both
    methods and both criteria within 2e-10 (time of one analysis on a
    loaded machine):

    ========  ==========  ==========  ======
    elements  upper - R   lower - R   time
    ========  ==========  ==========  ======
    40        -0.0108     +0.0406     3.5 s
    80        -0.0067     +0.0086     5.0 s
    160       -0.0056     +0.0007     7.8 s
    320       -0.0054     -0.0013     10 s
    ========  ==========  ==========  ======

    the lower limit load converging with the mesh, the upper one to the
    sampling difference above. Following the policy of the tests whose
    reference is reached only on fine meshes, the limit loads of the 40
    elements are compared with the values pre-determined by this same test,
    5.0854743909 and -0.6557002018, with ``rtol = 1e-6`` (the methods and
    criteria differ by 2e-10 relative), and with Fung and Kaplan with the
    grades of MacNeal, R. H. and Harder, R. L., A proposed standard set of
    problems to test finite element accuracy, Finite Elements in Analysis
    and Design, 1(1), 3-20, 1985,
    https://doi.org/10.1016/0168-874X(85)90003-4, p. 14, as bounds: A (at
    most 2 %) for the upper limit load, 0.21 % measured, and B (at most
    10 %) for the lower one, 5.8 % measured. Until 0.12.0 both were
    compared with ``0.01*R_cr = 0.051``, close to the 0.046 of the lower
    one.

    The round-off floor of the convergence criteria must accept at least one
    iteration, see the module documentation.

    """
    lam1 = 2.2
    R_cr, R_low = limit_loads(lam1)
    B1_lim = np.sqrt((lam1**2 - 1)/3)
    R_ref = 1.25*R_cr
    model, bu, B1 = arch(lam1, R_ref=R_ref)

    roundoff = []

    def recording_check_convergence(*args, **kwargs):
        out = convergence.check_convergence(*args, **kwargs)
        if kwargs.get('full_output'):
            roundoff.append(out[3])
        return out

    monkeypatch.setattr(newton_raphson, 'check_convergence',
                        recording_check_convergence)
    monkeypatch.setattr(arc_length, 'check_convergence',
                        recording_check_convergence)

    an = model.analysis(bu, NL_method=method, initialInc=0.05, maxInc=0.05,
                        convergence=convergence_criterion, maxArcLength=1e6)
    assert an.relTOL == 1e-9 and an.absTOL is None
    increments, cs = an.static(NLgeom=True, silent=True)
    assert increments[-1] == 1.0
    assert any(roundoff)
    R = R_ref*np.asarray(increments)
    B1s = np.array([B1(c) for c in cs])
    # NOTE continuous path, without doubling back, to the inverted arch
    assert np.all(np.diff(np.r_[lam1, B1s]) < 0)
    assert B1s[-1] < -2.
    kmin = np.argmin(R)
    kmax = np.argmax(R[:kmin])
    assert B1s[kmax] > 0 > B1s[kmin]
    assert abs(B1s[kmax] - B1_lim) < 0.05
    assert abs(B1s[kmin] + B1_lim) < 0.05
    # NOTE values pre-determined with 40 elements, see the docstring
    assert np.isclose(R[kmax], 5.0854743909, rtol=1e-6)
    assert np.isclose(R[kmin], -0.6557002018, rtol=1e-6)
    # NOTE grades A and B against Fung and Kaplan
    assert abs(R[kmax]/R_cr - 1) <= 0.02
    assert abs(R[kmin]/R_low - 1) <= 0.10
    # NOTE two null loads, between the limit points and after the lower one
    assert np.count_nonzero(R[:-1]*R[1:] < 0) == 2
