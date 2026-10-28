"""Tests for the arc-length solvers using analytic problems

Both arc-length methods trace the equilibrium path until the load factor
reaches exactly 1.0, the cumulative arc length reaches ``maxArcLength`` or the
arc-length increment becomes smaller than ``minInc``.

"""
import numpy as np
import pytest

from pyfe3d.solver import arc_length

from analytic_problems import (hardening_springs, von_mises_truss,
                               linear_problem, convergence_orders, Problem)

METHODS = ['arc_length_riks', 'arc_length_crisfield']

#: apex deflection at the limit point of the von Mises truss
W_LIMIT = 1 - 1/np.sqrt(3)
#: limit load of the von Mises truss, see :func:`von_mises_truss`
F_LIMIT = 2/(3*np.sqrt(3))


def check_limit_points(p, frac, which=(0, 1), rtol=1e-5):
    r"""The load factors sampled around the two limit points of the von
    Mises truss of :func:`von_mises_truss`, of exact values ``1/frac`` at
    ``w = 1 - 1/sqrt(3)`` and ``-1/frac`` at ``w = 1 + 1/sqrt(3)``

    The arc-length method samples the path at discrete points, and the
    largest (smallest) sampled load factor is below (above) the limit load
    factor by an amount that depends on the step. Along the equilibrium path
    ``lambda = F(w)/P``, ``P = frac*F_LIMIT`` and ``F(w) = w**3 - 3*w**2 +
    2*w``. With ``w_k <= w_lim <= w_k+1`` the samples bracketing the limit
    point, one of them is within ``dw/2 = (w_k+1 - w_k)/2`` of ``w_lim``,
    and Taylor's theorem with ``dF/dw(w_lim) = 0`` and ``|d2F/dw2| = 6*|w -
    1| <= 6*m`` in the bracket, ``m = max(|w_k - 1|, |w_k+1 - 1|)``, bounds
    the gap:
    ``|lambda_lim| - max(|lambda_k|, |lambda_k+1|) <= 3*m*dw**2/(4*P)``.
    ``rtol`` is the tolerance of :func:`check_equilibrium`. The apex
    deflection must increase along the path.
    """
    P = frac*F_LIMIT
    inc = np.asarray(p.an.increments)
    w = np.array([c[0] for c in p.an.cs])
    assert np.all(np.diff(w) > 0)
    limits = ((W_LIMIT, 1/frac), (1 + 1/np.sqrt(3), -1/frac))
    for i in which:
        w_lim, lbd_lim = limits[i]
        k = np.searchsorted(w, w_lim)
        assert 0 < k < len(w)
        dw = w[k] - w[k-1]
        m = max(abs(w[k] - 1), abs(w[k-1] - 1))
        bound = 3*m*dw**2/(4*P)
        gap = abs(lbd_lim) - np.abs(inc[k-1:k+1]).max()
        assert np.sign(inc[k-1:k+1]).tolist() == [np.sign(lbd_lim)]*2
        assert -rtol*abs(lbd_lim) <= gap <= bound + rtol*abs(lbd_lim), (
            i, gap, bound)


def check_equilibrium(p, rtol=1e-5):
    an = p.an
    assert len(an.increments) == len(an.cs)
    assert len(an.increments) >= 1
    fext_norm = np.linalg.norm(p.fext)
    for lbd, c in zip(an.increments, an.cs):
        fint = p.fint(c)
        R = fint - lbd*p.fext
        ref = max(np.linalg.norm(lbd*p.fext), np.linalg.norm(fint))
        assert np.linalg.norm(R) <= rtol*ref + 1e-9*fext_norm, (lbd, R)


def scaled_problem(p, force, disp):
    """Same problem with forces multiplied by ``force`` and displacements
    divided by ``disp``"""
    return Problem(force*p.fext,
                   lambda c: force*p.fint(disp*c),
                   force*disp*p.kC,
                   lambda c: force*disp*p.kG(disp*c))


@pytest.mark.parametrize('method', METHODS)
def test_arc_length_linear_problem(method):
    p = linear_problem()
    p.analysis(NL_method=method, initialInc=0.1)
    p.an.static(NLgeom=True, silent=True)
    check_equilibrium(p, rtol=1e-10)
    inc = np.asarray(p.an.increments)
    assert inc[-1] == 1.0
    assert len(inc) > 1
    assert np.all(np.diff(inc) > 0)
    # the tangent predictor is exact for a linear problem, the last step
    # includes the correction that lands exactly at a load factor of 1.0
    assert max(p.calls_per_step()) <= 2, p.calls_per_step()


@pytest.mark.parametrize('method', METHODS)
def test_arc_length_hardening(method):
    p = hardening_springs()
    p.analysis(NL_method=method, initialInc=0.1)
    p.an.static(NLgeom=True, silent=True)
    check_equilibrium(p)
    inc = np.asarray(p.an.increments)
    assert inc[-1] == 1.0
    assert np.all(np.diff(inc) > 0)
    np.testing.assert_allclose(p.an.cs[-1], p.solve_reference(), rtol=1e-5)
    assert max(p.calls_per_step()) <= 8, p.calls_per_step()


@pytest.mark.parametrize('method', METHODS)
def test_arc_length_quadratic_convergence(method):
    p = hardening_springs()
    p.analysis(NL_method=method, initialInc=0.3, relTOL=1e-12)
    p.an.static(NLgeom=True, silent=True)
    check_equilibrium(p, rtol=1e-10)
    assert p.an.increments[0] < 1.
    # iterates of the first step, the last one is the converged solution
    cs = [c for step, c in p.fint_calls if step == 0]
    np.testing.assert_allclose(cs[-1], p.an.cs[0])
    ref = p.an.cs[0]
    errors = [np.linalg.norm(c - ref)/np.linalg.norm(ref) for c in cs[:-1]]
    orders = convergence_orders(errors)
    assert len(orders) >= 1, errors
    assert min(orders) > 1.6, (orders, errors)


@pytest.mark.parametrize('method', METHODS)
def test_arc_length_snap_through(method):
    """Load of 1.5 times the limit load of the von Mises truss

    The path goes through the limit point, the load factor becomes negative
    during the snap-through, and it increases again until the load factor
    reaches 1.0 on the inverted configuration.

    """
    p = von_mises_truss(frac=1.5)
    p.analysis(NL_method=method, initialInc=0.05)
    p.an.static(NLgeom=True, silent=True)
    check_equilibrium(p)
    inc = np.asarray(p.an.increments)
    w = np.array([c[0] for c in p.an.cs])
    assert inc[-1] == 1.0
    # limit load factor is 1/1.5, the minimum along the path is -1/1.5
    assert inc[inc < 1.].max() < 1/1.5 + 1e-6
    check_limit_points(p, frac=1.5)
    assert inc.min() < 0
    assert np.all(inc[:-1] < 1.)
    # continuous path along the apex deflection, without jumps
    assert np.all(np.diff(w) > 0)
    assert np.diff(w).max() < 0.5
    assert w[-1] > 2.
    np.testing.assert_allclose(p.an.cs[-1],
                               p.solve_reference(c0=p.an.cs[-1], tol=1e-12),
                               rtol=1e-6)


@pytest.mark.parametrize('convergence, relTOL', [('relative', 1.e-6),
                                                  ('crisfield', 1.e-9)])
@pytest.mark.parametrize('method', METHODS)
def test_arc_length_snap_back(method, convergence, relTOL):
    """A soft spring in series with the truss causes snap-back

    The displacement of the loaded point decreases along part of the path.
    The relative criterion with ``relTOL=1e-6``, and Crisfield's criterion
    with its default, ``1e-9``.

    """
    p = von_mises_truss(frac=1.5, ks=0.5)
    p.analysis(NL_method=method, initialInc=0.05, convergence=convergence,
               relTOL=relTOL)
    p.an.static(NLgeom=True, silent=True)
    check_equilibrium(p)
    inc = np.asarray(p.an.increments)
    cs = np.asarray(p.an.cs)
    assert inc[-1] == 1.0
    assert inc.min() < 0
    # NOTE the loaded point snaps back, the apex deflection does not, and
    #      the load factor at the apex deflection is that of the truss alone
    check_limit_points(p, frac=1.5)
    assert np.any(np.diff(cs[:, 1]) < 0)
    assert np.all(np.diff(cs[:, 0]) > 0)
    assert cs[-1, 0] > 2.


@pytest.mark.parametrize('convergence, relTOL', [('relative', 1.e-6),
                                                  ('crisfield', 1.e-9)])
@pytest.mark.parametrize('method', METHODS)
def test_arc_length_no_doubling_back(method, convergence, relTOL):
    """Long steps through the limit point at 10 times the limit load

    The previous converged point is at the distance of the arc length from
    the last one, and the iterations of the second step may converge to it,
    which would trace the path backwards, the load factor decreasing below
    zero without ever reaching 1.0. Such a step points against the tangent
    predictor, and it is repeated with a shorter arc length.

    """
    p = von_mises_truss(frac=10., ks=1.)
    p.analysis(NL_method=method, initialInc=0.2, convergence=convergence,
               relTOL=relTOL)
    p.an.static(NLgeom=True, silent=True)
    check_equilibrium(p)
    inc = np.asarray(p.an.increments)
    cs = np.asarray(p.an.cs)
    assert inc[-1] == 1.0
    assert np.all(np.diff(np.r_[0., cs[:, 0]]) > 0)
    np.testing.assert_allclose(cs[-1],
                               p.solve_reference(c0=cs[-1], tol=1e-12),
                               rtol=1e-6)


@pytest.mark.parametrize('method', METHODS)
def test_arc_length_step_cutting(method):
    p = von_mises_truss(frac=1.5)
    p.analysis(NL_method=method, initialInc=0.2, maxNumIter=3)
    p.an.static(NLgeom=True, silent=True)
    check_equilibrium(p)
    assert p.an.increments[-1] == 1.0
    assert p.an.cs[-1][0] > 2.


@pytest.mark.parametrize('method', METHODS)
def test_arc_length_modified_NR(method):
    p = von_mises_truss(frac=1.5)
    p.analysis(NL_method=method, initialInc=0.05, modified_NR=True,
               compute_every_n=3)
    p.an.static(NLgeom=True, silent=True)
    check_equilibrium(p)
    assert p.an.increments[-1] == 1.0
    assert p.an.cs[-1][0] > 2.


@pytest.mark.parametrize('method', METHODS)
def test_arc_length_maxArcLength(method):
    p = von_mises_truss(frac=1.5)
    p.analysis(NL_method=method, initialInc=0.05, maxArcLength=0.3)
    p.an.static(NLgeom=True, silent=True)
    check_equilibrium(p)
    inc = np.asarray(p.an.increments)
    assert 0 < inc[-1] < 1.


@pytest.mark.parametrize('method', METHODS)
def test_arc_length_silent(method, capsys):
    p = von_mises_truss(frac=1.5)
    p.analysis(NL_method=method, initialInc=0.05)
    p.an.static(NLgeom=True, silent=True)
    assert capsys.readouterr().out == ''


@pytest.mark.parametrize('method', METHODS + ['NR'])
def test_units_independence(method):
    frac = 1.5 if method != 'NR' else 0.99
    results = []
    for force, disp in [(1., 1.), (1.e6, 1.), (1.e-4, 1.e-3)]:
        p = scaled_problem(von_mises_truss(frac=frac), force, disp)
        p.analysis(NL_method=method, initialInc=0.05)
        p.an.static(NLgeom=True, silent=True)
        check_equilibrium(p)
        results.append((np.asarray(p.an.increments),
                        disp*np.asarray(p.an.cs), len(p.fint_calls)))
    inc0, cs0, calls0 = results[0]
    assert inc0[-1] == 1.0
    for inc, cs, calls in results[1:]:
        assert calls == calls0
        np.testing.assert_allclose(inc, inc0, rtol=1e-6, atol=1e-9)
        np.testing.assert_allclose(cs, cs0, rtol=1e-6, atol=1e-9)



def test_arc_length_invalid_method():
    p = linear_problem()
    an = p.analysis()
    with pytest.raises(ValueError, match='Invalid arc-length method: newton'):
        arc_length._solver_arc_length(an, an, 'newton', silent=True)


@pytest.mark.parametrize('method', METHODS)
def test_arc_length_lands_exactly_on_load_factor_one(method):
    """A converged step whose load factor is exactly 1.0 ends the analysis
    without the correction to the load factor 1.0"""
    p = linear_problem()
    p.analysis(NL_method=method, initialInc=1.)
    p.an.static(NLgeom=True, silent=True)
    assert p.an.increments == [1.]
    np.testing.assert_allclose(p.kC @ p.an.cs[0], p.fext, rtol=1e-12)
    assert p.calls_per_step() == [1]


@pytest.mark.parametrize('method', METHODS)
def test_arc_length_maximum_number_of_steps(method, monkeypatch, capsys):
    monkeypatch.setattr(arc_length, 'MAX_NUM_STEPS', 3)
    p = linear_problem()
    p.analysis(NL_method=method, initialInc=0.1)
    p.an.static(NLgeom=True, silent=False)
    out = capsys.readouterr().out
    assert len(p.an.increments) == 3
    assert p.an.increments[-1] < 1.
    check_equilibrium(p, rtol=1e-10)
    assert 'WARNING: Maximum number of steps of 3 achieved' in out


@pytest.mark.parametrize('method', METHODS)
def test_arc_length_modified_NR_kT_initial_state_false(method):
    """``kT_initial_state = False``: the predictor of the first step uses the
    linear stiffness matrix, without evaluating the tangent stiffness matrix
    at the initial state"""
    def run(kT_initial_state):
        p = von_mises_truss(frac=1.5)
        events = []
        calc_fint, calc_kG = p.calc_fint, p.calc_kG
        p.calc_fint = lambda c, silent=True: (events.append('fint'),
                                             calc_fint(c, silent))[1]
        p.calc_kG = lambda c=None, NLgeom=False, silent=True: (
            events.append('kG'), calc_kG(c, NLgeom, silent))[1]
        p.analysis(NL_method=method, initialInc=0.05, modified_NR=True,
                   kT_initial_state=kT_initial_state)
        p.an.static(NLgeom=True, silent=True)
        check_equilibrium(p)
        assert p.an.increments[-1] == 1.0
        assert p.an.cs[-1][0] > 2.
        return events

    assert run(True)[0] == 'kG'
    assert run(False)[0] == 'fint'


def singular_tangent(at_origin):
    """2 DOFs whose tangent stiffness matrix ``[[1, 1], [1, 1]]`` is
    singular, at the origin too or only away from it, with a regular linear
    stiffness matrix"""
    kC = np.array([[2., 1.], [1., 2.]])

    def kG(c):
        if at_origin or (c is not None and np.any(c != 0)):
            return -np.eye(2)
        return np.zeros((2, 2))

    return Problem(np.array([1., 0.]), lambda c: kC @ c + 0.1*c**3, kC, kG)


@pytest.mark.parametrize('method', METHODS)
def test_arc_length_singular_tangent_at_converged_state(method, capsys):
    """The analysis stops when the tangent stiffness matrix of the last
    converged state is singular: at once from the initial state, after the
    first step when the linear stiffness matrix is used for it"""
    p = singular_tangent(at_origin=True)
    p.analysis(NL_method=method, initialInc=0.1)
    p.an.static(NLgeom=True, silent=False)
    out = capsys.readouterr().out
    assert p.an.increments == []
    assert ('WARNING: Singular tangent stiffness matrix at the last '
            'converged state! Analysis stopped') in out

    p = singular_tangent(at_origin=True)
    p.analysis(NL_method=method, initialInc=0.1, modified_NR=True,
               kT_initial_state=False)
    p.an.static(NLgeom=True, silent=False)
    out = capsys.readouterr().out
    assert len(p.an.increments) == 1
    check_equilibrium(p, rtol=1e-9)
    assert 'at the last converged state! Analysis stopped' in out


@pytest.mark.parametrize('method', METHODS)
def test_arc_length_singular_tangent_in_the_iterations(method, capsys):
    """A singular tangent stiffness matrix at an iterate stops the
    iterations of the step, which is repeated with a shorter arc length
    until it is smaller than ``minInc``"""
    p = singular_tangent(at_origin=False)
    p.analysis(NL_method=method, initialInc=0.1, minInc=1e-3)
    p.an.static(NLgeom=True, silent=False)
    out = capsys.readouterr().out
    assert p.an.increments == []
    assert '\t\tWARNING: Singular tangent stiffness matrix\n' in out
    assert 'WARNING: Minimum arc-length increment of 0.00100 achieved' in out


def sharp_peak(cp=0.05*(1 - 1e-12)):
    """1 DOF with a unit stiffness up to the peak load ``cp`` at the
    displacement ``cp``, and the stiffness -1 beyond it"""
    def fint(c):
        return np.where(c < cp, c, 2*cp - c)

    def kG(c):
        return np.array([[0. if c is None or c[0] < cp else -2.]])

    return Problem(np.array([1.]), fint, np.array([[1.]]), kG)


def test_arc_length_riks_at_a_sharp_peak(capsys):
    r"""Riks' linearized constraint has no solution when the tangent
    `\Delta \boldsymbol{c}_q` of the iterate is orthogonal to the increment, here
    when the stiffness changes sign from 1 to -1: the step is repeated with a
    shorter arc length. The predictor of the next step is orthogonal to the
    last increment, and the sign of the predictor is then positive, along
    `\Delta \boldsymbol{c}_q = \{-1\}`; the analysis stops at the peak when the arc
    length becomes smaller than ``minInc``"""
    p = sharp_peak()
    p.analysis(NL_method='arc_length_riks', initialInc=0.1, minInc=1e-3)
    p.an.static(NLgeom=True, silent=False)
    out = capsys.readouterr().out
    np.testing.assert_allclose(p.an.increments, [0.05], rtol=1e-10)
    np.testing.assert_allclose(p.an.cs[0], [0.05], rtol=1e-10)
    check_equilibrium(p, rtol=1e-10)
    assert ('WARNING: Diverged - arc-length constraint cannot be satisfied'
            in out)
    assert 'WARNING: Minimum arc-length increment of 0.00100 achieved' in out
    second_step = [c[0] for step, c in p.fint_calls if step == 1]
    assert second_step and second_step[0] < p.an.cs[0][0]


def test_arc_length_crisfield_complex_roots(capsys):
    """Without real roots of Crisfield's quadratic constraint, here with the
    modified Newton-Raphson method, the linearized constraint is used and
    the analysis still reaches the load factor 1.0"""
    p = hardening_springs()
    p.analysis(NL_method='arc_length_crisfield', initialInc=0.3,
               modified_NR=True)
    p.an.static(NLgeom=True, silent=False)
    out = capsys.readouterr().out
    assert 'Complex roots, using the linearized constraint' in out
    check_equilibrium(p)
    assert p.an.increments[-1] == 1.0
