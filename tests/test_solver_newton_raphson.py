"""Tests for pyfe3d.solver.newton_raphson using analytic problems

All problems have an exact tangent stiffness matrix ``kT = kC + kG`` that is
the Jacobian of ``fint``, such that full Newton-Raphson must converge
quadratically near the solution.

"""
import numpy as np

from analytic_problems import (hardening_springs, von_mises_truss,
                               linear_problem, convergence_orders, Problem)


def check_increments_and_equilibrium(p, rtol=1e-5):
    an = p.an
    inc = np.asarray(an.increments)
    assert len(an.increments) == len(an.cs)
    assert len(inc) >= 1
    assert inc[-1] == 1.0
    assert np.all(np.diff(inc) > 0)
    assert np.all(inc > 0)
    for lbd, c in zip(an.increments, an.cs):
        R = p.fint(c) - lbd*p.fext
        assert np.linalg.norm(R) <= rtol*np.linalg.norm(lbd*p.fext)


def test_nr_full_load_quadratic_convergence():
    p = hardening_springs()
    p.analysis(initialInc=1.)
    p.an.static(NLgeom=True, silent=True)
    check_increments_and_equilibrium(p)
    assert len(p.an.increments) == 1

    ref = p.solve_reference()
    errors = [np.linalg.norm(c - ref)/np.linalg.norm(ref)
              for _, c in p.fint_calls]
    orders = convergence_orders(errors)
    assert len(orders) >= 1, errors
    assert min(orders) > 1.6, (orders, errors)
    # plain Newton from the linear predictor needs 9 residual evaluations to
    # reach machine precision; relTOL requires at most that
    assert len(p.fint_calls) <= 9


def test_nr_line_search_does_not_spoil_quadratic_convergence():
    p0 = hardening_springs()
    p0.analysis(initialInc=1., line_search=False)
    p0.an.static(NLgeom=True, silent=True)

    p = hardening_springs()
    p.analysis(initialInc=1., line_search=True)
    p.an.static(NLgeom=True, silent=True)
    check_increments_and_equilibrium(p)

    ref = p.solve_reference()
    errors = [np.linalg.norm(c - ref)/np.linalg.norm(ref)
              for _, c in p.fint_calls]
    orders = convergence_orders(errors)
    assert len(orders) >= 1, errors
    assert min(orders) > 1.6, (orders, errors)
    # the Newton step is accepted as is, without extra fint evaluations
    assert len(p.fint_calls) == len(p0.fint_calls)


def test_nr_incremental_hardening():
    p = hardening_springs()
    p.analysis(initialInc=0.1)
    p.an.static(NLgeom=True, silent=True)
    check_increments_and_equilibrium(p)
    # a good predictor with an exact tangent needs few iterations per step
    assert max(p.calls_per_step()) <= 5, p.calls_per_step()


def test_nr_modified_NR_option():
    p = hardening_springs()
    p.analysis(initialInc=0.1, modified_NR=True, compute_every_n=3)
    p.an.static(NLgeom=True, silent=True)
    check_increments_and_equilibrium(p)


def test_nr_load_factor_bookkeeping():
    """Load steps must grow at most by the growth factor and end at 1.0"""
    for initialInc in [0.1, 0.25, 0.3, 0.5, 0.7, 1.]:
        p = linear_problem()
        p.analysis(initialInc=initialInc)
        p.an.static(NLgeom=True, silent=True)
        check_increments_and_equilibrium(p, rtol=1e-12)
        inc = np.asarray(p.an.increments)
        assert np.isclose(inc[0], initialInc)
        steps = np.diff(np.concatenate(([0.], inc)))
        assert np.all(steps[1:] <= 1.1111*steps[:-1] + 1e-12), inc


def test_nr_predictor_scaled_with_increment():
    """On a linear problem a scaled predictor is the exact solution

    Hence every step must converge at the first residual evaluation.

    """
    p = linear_problem()
    p.analysis(initialInc=0.1)
    p.an.static(NLgeom=True, silent=True)
    check_increments_and_equilibrium(p, rtol=1e-12)
    assert len(p.an.increments) > 1
    assert p.calls_per_step() == [1]*len(p.an.increments)


def exact_apex_deflection(frac):
    r"""Apex deflection of :func:`von_mises_truss` under ``frac`` times the
    limit load, the smallest positive root of ``w**3 - 3*w**2 + 2*w =
    frac*2/(3*sqrt(3))``, before the limit point at ``w = 1 - 1/sqrt(3)``"""
    roots = np.roots([1., -3., 2., -frac*2/(3*np.sqrt(3))])
    roots = roots[np.isreal(roots)].real
    return roots[roots > 0].min()


def test_nr_softening_with_step_cutting():
    p = von_mises_truss(frac=0.99)
    # plain Newton needs 5 corrections to reach relTOL=1e-6 at full load, a
    # budget of 4 corrections forces the solver to cut the step
    p.analysis(initialInc=1., maxNumIter=4)
    p.an.static(NLgeom=True, silent=True)
    check_increments_and_equilibrium(p)
    assert len(p.an.increments) >= 2
    ref = p.solve_reference(c0=p.an.cs[-1])
    np.testing.assert_allclose(p.an.cs[-1], ref, rtol=1e-6)
    # the converged apex deflection is before the limit point
    assert p.an.cs[-1][0] < 1 - 1/np.sqrt(3)
    # NOTE rtol of the residual 1e-6, the tangent at 0.99 of the limit load
    #      is about 1/4 of the initial one: the error in w is below 1e-5
    np.testing.assert_allclose(p.an.cs[-1][0], exact_apex_deflection(0.99),
                               rtol=1e-5)


def test_nr_softening_incremental():
    p = von_mises_truss(frac=0.99)
    p.analysis(initialInc=0.1)
    p.an.static(NLgeom=True, silent=True)
    check_increments_and_equilibrium(p)
    assert p.an.cs[-1][0] < 1 - 1/np.sqrt(3)
    np.testing.assert_allclose(p.an.cs[-1][0], exact_apex_deflection(0.99),
                               rtol=1e-5)


def test_nr_non_monotonic_residual_is_not_divergence():
    """Newton converges although the residual increases at iteration 5

    ``fint = u + 0.8 sin(u)`` has a positive tangent and a unique solution,
    but the Newton iterates oscillate before locking in, similarly to what
    happens in deep postbuckling.

    """
    b = 0.8
    p = Problem(np.array([8.]), lambda u: u + b*np.sin(u),
                np.array([[1. + b]]), lambda u: np.array([[b*np.cos(u[0]) - b]]))
    p.analysis(initialInc=1.)
    p.an.static(NLgeom=True, silent=True)
    check_increments_and_equilibrium(p)
    # solved in a single step, without step cutting
    assert len(p.an.increments) == 1
    assert len(p.fint_calls) <= 9
    # the residual did increase after the third iteration
    Rs = [abs(8. - p.fint(c)[0]) for _, c in p.fint_calls]
    assert any(Rs[i] > Rs[i-1] for i in range(3, len(Rs)))


def tangent_error(p, c, eps=1e-6):
    """Relative difference between the tangent stiffness matrix of the
    callables and the central finite differences of ``calc_fint`` at ``c``"""
    an = p.analysis()
    kT = (an.calc_kC(c, NLgeom=True) + an.calc_kG(c, NLgeom=True)).toarray()
    kT_fd = np.zeros_like(kT)
    for j in range(c.size):
        dc = np.zeros_like(c)
        dc[j] = eps*max(1., abs(c[j]))
        kT_fd[:, j] = (p.fint(c + dc) - p.fint(c - dc))/(2*dc[j])
    return np.linalg.norm(kT - kT_fd)/np.linalg.norm(kT_fd)


def test_tangent_consistency_finite_differences():
    """The tangent stiffness matrix must be the derivative of the internal
    force vector, which a finite-difference check verifies at any
    configuration"""
    c = np.array([0.1, -0.2, 0.3, 0.5])
    assert tangent_error(hardening_springs(), c) < 1e-8
    assert tangent_error(von_mises_truss(), c[:2]) < 1e-8
    # without kG the tangent is inconsistent
    p = hardening_springs()
    p.kG = lambda u: np.zeros((4, 4))
    assert tangent_error(p, c) > 1e-2


def test_inconsistent_tangent_converges_linearly():
    """An inconsistent tangent stiffness matrix converges linearly, with many
    more iterations per step, and here not at all at the higher load
    factors: the analysis stops before the load factor 1.0"""
    p = hardening_springs()
    p.analysis(initialInc=0.1)
    p.an.static(NLgeom=True, silent=True)
    check_increments_and_equilibrium(p)
    assert max(p.calls_per_step()) <= 5

    q = hardening_springs()
    ref = q.solve_reference(lbd=0.1)
    q.kG = lambda u: np.zeros((4, 4))
    q.analysis(initialInc=0.1)
    q.an.static(NLgeom=True, silent=True)
    assert q.an.increments[-1] < 1.
    assert max(q.calls_per_step()) > 20
    # the first attempt, at the load factor 0.1, stops after maxNumIter
    # corrections without reaching relTOL, the step is then cut
    assert q.an.increments[0] < 0.1
    errors = [np.linalg.norm(c - ref)/np.linalg.norm(ref)
              for _, c in q.fint_calls[:q.an.maxNumIter]]
    orders = convergence_orders(errors)
    assert len(orders) >= 1
    assert max(orders) < 1.2, orders


def test_nr_unreachable_tolerance_stops_without_increments(monkeypatch):
    """A tolerance below the round-off cannot be met without the round-off
    floor of the convergence criteria, disabled here: every step is cut
    until the increment is smaller than minInc, and no increment is
    stored"""
    from pyfe3d.solver import convergence
    monkeypatch.setattr(convergence, 'ROUNDOFF_FACTOR', 0.)
    p = hardening_springs()
    p.analysis(relTOL=1e-20)
    p.an.static(NLgeom=True, silent=True)
    assert p.an.increments == []
    assert p.an.cs == []


def test_nr_jumps_over_limit_point():
    """Beyond the limit load, load control converges to another branch of
    the equilibrium path: the run finishes at the load factor 1.0, with a
    jump of the solution that the arc-length methods do not have"""
    p = von_mises_truss(frac=1.5)
    p.analysis(initialInc=0.05)
    p.an.static(NLgeom=True, silent=True)
    check_increments_and_equilibrium(p)
    w = np.array([c[0] for c in p.an.cs])
    assert np.diff(w).max() > 1.
    # the same final state as the arc-length methods, on the inverted
    # configuration
    assert w[-1] > 2.

    q = von_mises_truss(frac=1.5)
    q.analysis(NL_method='arc_length_riks', initialInc=0.05)
    q.an.static(NLgeom=True, silent=True)
    wq = np.array([c[0] for c in q.an.cs])
    assert np.diff(wq).max() < 0.5
    np.testing.assert_allclose(p.an.cs[-1], q.an.cs[-1], rtol=1e-5)



def inconsistent_tangent(factor=0.1):
    """The hardening springs with the exact tangent stiffness matrix
    multiplied by ``factor``: the Newton direction is ``1/factor`` times too
    long, the linear stiffness matrix is unchanged"""
    h = hardening_springs()
    return Problem(h.fext, h.fint, h.kC,
                   lambda c: factor*(h.kC + h.kG(c)) - h.kC), h


def test_nr_line_search_shortens_a_too_long_direction(capsys):
    r"""With a direction 10 times too long, `\phi(1) = 81 \phi(0)` for a
    nearly linear residual, and the minimum of the quadratic interpolation
    is cut to its safeguard, `\eta = 0.1`, which recovers the Newton step:
    the run converges to the reference solution, logging the step length"""
    p, h = inconsistent_tangent()
    p.analysis(initialInc=1., line_search=True)
    p.an.static(NLgeom=True, silent=False)
    out = capsys.readouterr().out
    check_increments_and_equilibrium(p, rtol=1e-8)
    np.testing.assert_allclose(p.an.cs[-1], h.solve_reference(), rtol=1e-8)
    assert 'Line-search: eta = 0.10000' in out
    assert 'maximum number of iterations' not in out


def test_nr_line_search_without_iterations(capsys):
    """With ``max_iter_line_search = 0`` the full step of the too long
    direction is kept, with a warning: the residual diverges, every step is
    cut down to ``minInc`` and no increment is accepted"""
    p, h = inconsistent_tangent()
    p.analysis(initialInc=1., line_search=True, max_iter_line_search=0,
               minInc=0.1)
    p.an.static(NLgeom=True, silent=False)
    out = capsys.readouterr().out
    assert p.an.increments == []
    assert ('WARNING: Line-search: maximum number of iterations achieved'
            in out)
    assert 'Line-search: eta' not in out


def test_nr_non_finite_residual_cuts_the_step(capsys):
    """A residual that is not finite, here an internal force vector that is
    not defined beyond 0.42, stops the iterations of the step, which is
    repeated with a smaller increment"""
    def fint(c):
        return np.where(c > 0.42, np.nan, c + c**3)

    p = Problem(np.array([0.45]), fint, np.array([[1.]]),
                lambda c: np.diag(3*c**2))
    p.analysis(initialInc=1.)
    p.an.static(NLgeom=True, silent=False)
    out = capsys.readouterr().out
    assert 'WARNING: Diverged - residual is not finite' in out
    check_increments_and_equilibrium(p, rtol=1e-9)
    # NOTE the first attempt, at the load factor 1, is cut
    assert p.an.increments[0] == 0.5
    c = p.an.cs[-1][0]
    np.testing.assert_allclose(c + c**3, 0.45, rtol=1e-9)


def test_nr_modified_kT_initial_state_false():
    """``kT_initial_state = False``: the first iteration of the first step
    of the modified Newton-Raphson method uses the linear stiffness matrix,
    instead of the tangent stiffness matrix at the predictor, and the run
    converges to the same solution"""
    def run(kT_initial_state):
        p = hardening_springs()
        events = []
        calc_fint, calc_kG = p.calc_fint, p.calc_kG
        p.calc_fint = lambda c, silent=True: (events.append('fint'),
                                             calc_fint(c, silent))[1]
        p.calc_kG = lambda c=None, NLgeom=False, silent=True: (
            events.append('kG'), calc_kG(c, NLgeom, silent))[1]
        p.analysis(initialInc=0.2, modified_NR=True,
                   kT_initial_state=kT_initial_state)
        p.an.static(NLgeom=True, silent=True)
        check_increments_and_equilibrium(p)
        np.testing.assert_allclose(p.an.cs[-1], p.solve_reference(),
                                   rtol=1e-7)
        return events

    assert run(True)[:2] == ['fint', 'kG']
    assert run(False)[:2] == ['fint', 'fint']


def test_norm_without_weights():
    """The Euclidean norm when the weights of Crisfield's criterion are not
    given, the scaled norm otherwise"""
    from pyfe3d.solver.newton_raphson import _norm

    v = np.array([3., -4.])
    assert _norm(v, None) == 5.
    assert _norm(v, np.array([1., 0.25])) == np.sqrt(9. + 4.)


def test_logger_prints_unless_silent(capsys):
    """The messages are printed after ``level`` tabs, the warnings with the
    prefix ``WARNING:``, and returned; nothing is printed when silent"""
    from pyfe3d.solver.logger import msg, warn

    assert msg('text', level=2) == 'text'
    assert warn('careful', level=1) == 'WARNING: careful'
    assert capsys.readouterr().out == '\t\ttext\n\tWARNING: careful\n'
    msg('text', level=2, silent=True)
    warn('careful', silent=True)
    assert capsys.readouterr().out == ''
