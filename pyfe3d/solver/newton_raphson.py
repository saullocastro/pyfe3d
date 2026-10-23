r"""
===========================================================
Newton-Raphson solver (:mod:`pyfe3d.solver.newton_raphson`)
===========================================================

.. currentmodule:: pyfe3d.solver.newton_raphson

Load-controlled Newton-Raphson solver of the non-linear static analyses of
:class:`.Analysis`, selected with ``Analysis.NL_method = 'NR'``. The
module-level constants control the step size and the divergence checks, and
they are shared with the arc-length solvers of :mod:`.arc_length`.

"""
import numpy as np

from .logger import msg, warn
from .static import solve, _factorize
from .convergence import check_convergence, crisfield_weights, scaled_norm
from . import callbacks


#: Number of iterations of a step before the divergence and too-slow checks
#: are activated. A non-monotonic residual in the first iterations is normal,
#: especially in postbuckling
NUM_ITER_BEFORE_DIVERGENCE_CHECK = 5
#: A step diverged when its residual norm becomes larger than this factor
#: times the residual norm at the start of the step
DIVERGENCE_FACTOR = 1.e3
#: The convergence is too slow when the smallest residual norm of the last
#: ``TOO_SLOW_WINDOW`` iterations is not at least ``too_slow_TOL`` (relative)
#: smaller than the smallest residual norm of the previous iterations
TOO_SLOW_WINDOW = 5
#: Constant of the sufficient-decrease (Armijo) test of the line search
LINE_SEARCH_ALPHA = 1.e-4
#: Factor used to increase the load increment after a step that converged
#: without being cut
INC_GROWTH_FACTOR = 1.1111
#: Factor used to reduce the load increment when a step fails to converge
INC_CUT_FACTOR = 0.5


def _weights(an, kC0):
    r"""Weights of Crisfield's criterion from the linear stiffness matrix
    ``kC0``, see :func:`.crisfield_weights`, or ``None`` with
    ``an.convergence = 'relative'``"""
    if an.convergence == 'crisfield':
        return crisfield_weights(kC0)
    return None


def _norm(v, w):
    r"""Norm of the convergence criterion: `\|v\|_{\boldsymbol{d}}` with the
    weights ``w`` of Crisfield's criterion, the Euclidean norm when ``w`` is
    ``None``"""
    if w is None:
        return float(np.linalg.norm(v))
    return scaled_norm(v, w)


def _log_iteration(an, iteration, R, zeta, silent, lbd=None):
    r"""Message of one equilibrium iteration, with `\zeta` of Crisfield's
    criterion or the relative residual of the relative criterion"""
    name = 'zeta' if an.convergence == 'crisfield' else 'relative ||R||'
    load = '' if lbd is None else ', lbd %1.5f' % lbd
    msg('Iteration: %d%s, max(|R|) = %1.3e, %s = %1.3e'
        % (iteration, load, np.abs(R).max(), name, zeta), level=2,
        silent=silent)


def _check_divergence(an, Rnorm, Rnorms, iteration, silent=False):
    r"""Check if the iterations of a step must be stopped without convergence

    The iterations are stopped when the residual is not finite, when
    ``an.maxNumIter`` corrections were done, and, after
    ``NUM_ITER_BEFORE_DIVERGENCE_CHECK`` iterations, when the residual
    diverges or when the convergence is too slow.

    Parameters
    ----------
    an : :class:`.Analysis`
        Analysis object with the attributes ``maxNumIter`` and
        ``too_slow_TOL``.
    Rnorm : float
        Norm of the residual at the current iteration, the norm of the
        convergence criterion, see :func:`.check_convergence`.
    Rnorms : list
        Norms of the residual at the previous iterations of the step; the
        current norm is appended to it.
    iteration : int
        Current iteration, starting at 1.
    silent : bool, optional
        Whether the log messages are suppressed. Default ``False``.

    Returns
    -------
    stop : bool
        Whether the iterations must be stopped.

    Notes
    -----
    With `r_i` the residual norm at the iteration `i` of the step, `r_1`
    being the residual of the predictor, the iterations are stopped at the
    iteration `j` when:

    - `r_j` is not finite;
    - `j >` ``an.maxNumIter``, i.e. ``an.maxNumIter`` corrections were done;
    - `j >` ``NUM_ITER_BEFORE_DIVERGENCE_CHECK`` and `r_j >`
      ``DIVERGENCE_FACTOR`` `r_1` (divergence);
    - `j >` ``NUM_ITER_BEFORE_DIVERGENCE_CHECK``, `j > W` and
      `\min(r_{j-W+1}, ..., r_j) > (1 -` ``an.too_slow_TOL``
      `) \min(r_1, ..., r_{j-W})`, with `W` = ``TOO_SLOW_WINDOW`` (too slow
      convergence).

    The divergence and too-slow checks are delayed because a non-monotonic
    residual in the first iterations is normal, especially in postbuckling.
    The checks only compare norms of the same step with each other, and with
    the scaled norm of Crisfield's criterion they do not depend on the units
    of the degrees-of-freedom.

    """
    if not np.isfinite(Rnorm):
        warn('Diverged - residual is not finite', level=2, silent=silent)
        return True
    if iteration > an.maxNumIter:
        warn('Maximum number of iterations achieved!', level=2, silent=silent)
        return True
    Rnorms.append(Rnorm)
    if iteration > NUM_ITER_BEFORE_DIVERGENCE_CHECK:
        if Rnorm > DIVERGENCE_FACTOR*Rnorms[0]:
            warn('Diverged - residual increased %1.0e times since the start '
                 'of the step' % DIVERGENCE_FACTOR, level=2, silent=silent)
            return True
        if (len(Rnorms) > TOO_SLOW_WINDOW and
                min(Rnorms[-TOO_SLOW_WINDOW:]) >
                (1 - an.too_slow_TOL)*min(Rnorms[:-TOO_SLOW_WINDOW])):
            warn('Diverged - convergence too slow', level=2, silent=silent)
            return True
    return False


def _NR_iterations(an, pb, c, dc, total, fext, w, kT0=None, silent=False,
                   lbd_dependent=False, q=None):
    r"""Newton-Raphson iterations at a fixed load factor

    Corrects the predictor ``dc`` at the load factor ``total`` until the
    convergence criterion of :func:`.check_convergence` is satisfied, or
    until the iterations are stopped by :func:`_check_divergence` or by a
    singular tangent stiffness matrix. The tangent stiffness matrix is
    ``calc_kC + calc_kG`` at the current iterate, see
    :func:`.callbacks.calc_kT`; its updates and the line search are
    controlled by ``an.modified_NR``, ``an.compute_every_n``,
    ``an.line_search`` and ``an.max_iter_line_search``.

    Parameters
    ----------
    an : :class:`.Analysis`
        Analysis object with the parameters of the solver.
    pb : object
        Object with the callables, ``an`` itself or a :class:`.FreeDofs`.
    c : np.ndarray
        Solution at the last converged state.
    dc : np.ndarray
        Predictor of the solution increment.
    total : float
        Load factor, dimensionless.
    fext : np.ndarray
        External force vector for a unit load factor.
    w : np.ndarray or None
        Weights of Crisfield's criterion, see :func:`.crisfield_weights`,
        ``None`` with the relative criterion.
    kT0 : sparse_matrix or None, optional
        Tangent stiffness matrix used in the first iteration when
        ``an.modified_NR=True``. If ``None`` it is calculated at ``c + dc``.
        Default ``None``.
    silent : bool, optional
        Whether the log messages are suppressed. Default ``False``.
    lbd_dependent : bool, optional
        Whether ``calc_fext`` accepts ``lbd``, see :mod:`.callbacks`;
        ``total*q``, with `\{q\} = \partial R/\partial \lambda`, is then an
        additional reference of the convergence criterion. `\{q\}` is
        evaluated together with the tangent stiffness matrix, at the same
        iterate, and the last one evaluated is the reference until the next
        update of the tangent. Default ``False``.
    q : np.ndarray or None, optional
        The last `\{q\}` evaluated before these iterations, the reference
        until the first update of the tangent stiffness matrix, used only
        when ``lbd_dependent=True``. Default ``None``.

    Returns
    -------
    converged : bool
        Whether the iterations converged.
    dc : np.ndarray
        Solution increment, such that ``c + dc`` is the solution.
    q : np.ndarray or None
        The last `\{q\}` evaluated, ``None`` when ``lbd_dependent=False``.

    Notes
    -----
    At each iteration, with the load factor `\lambda` = ``total`` fixed,

    .. math::

        \{R\} = \lambda \{F_{ext}\} - \{F_{int}(c + \Delta c, \lambda)\},
        \qquad \{\delta c\} = [K_T]^{-1}\{R\}, \qquad
        \{\Delta c\} \leftarrow \{\Delta c\} + \eta \{\delta c\}

    With full Newton-Raphson, `[K_T]` is computed and factorized at every
    iteration at `c + \Delta c`, and the convergence is quadratic when it is
    the exact Jacobian `-\partial R/\partial c` ([1]_, Section 1.2.2,
    p. 9). With ``an.modified_NR = True``, `[K_T]` is computed at the first
    iteration, or given as ``kT0``, and then every ``an.compute_every_n``
    iterations, its LU factorization being reused in the other iterations.

    `\eta = 1` without line search. With ``an.line_search = True``, with
    `\phi(\eta) = \|R(c + \Delta c + \eta \, \delta c)\|^2` in the norm of
    the convergence criterion, the scaled norm `\|\cdot\|_{\boldsymbol{d}}`
    of Crisfield's criterion or the Euclidean norm, the step `\eta` is
    accepted when it satisfies the sufficient-decrease (Armijo) test

    .. math::

        \phi(\eta) \le (1 - 2 \alpha \eta) \phi(0)

    with `\alpha` = ``LINE_SEARCH_ALPHA``, which assumes `\phi'(0) =
    -2\phi(0)`, exact for a Newton direction in any norm whose weights do
    not change along the line, i.e. when `[K_T]` is the exact Jacobian at
    `c + \Delta c`; with ``an.modified_NR = True`` it is an approximation.
    The full step `\eta = 1` is tried first. Otherwise `\eta` is replaced by
    the minimizer `\phi(0)/A` of the quadratic interpolation of `\phi(0)`,
    `\phi'(0)` and `\phi(\eta)`, with `A = (\phi(\eta) - \phi(0) +
    2\phi(0)\eta)/\eta^2`, or by `\eta/2` when `A` is not finite or not
    positive, safeguarded in `[0.1 \eta, 0.5 \eta]`. After
    ``an.max_iter_line_search`` reductions the last `\eta` is accepted. The
    internal force vector of the accepted point is reused in the residual of
    the next iteration.

    The line search is adapted from the backtracking line search of [2]_
    (Section 6.3.2, Algorithm 6.3.5, pp. 126-129, and Algorithm A6.3.1,
    pp. 325-327), applied to nonlinear equations with the merit function
    `f = \frac{1}{2}\|D_F F\|^2` ([2]_, Section 6.5, Eq. (6.5.3), p. 147,
    and Algorithm NEFN, p. 303), whose directional derivative along the
    Newton step is `-F^T F = -2f` (p. 148, with `D_F` the identity). With
    `F = R` and `\phi = 2f`, `D_F` being the identity with the Euclidean
    norm or `\mathrm{diag}(\sqrt{w_j})` with Crisfield's criterion, the test
    above is their condition (6.3.3) (p. 118) with their `\alpha = 10^{-4}`
    (p. 126), and the first reduction is their quadratic backtrack,
    Eq. (6.3.17) (p. 127). It differs from Algorithm A6.3.1 in that: the
    quadratic interpolation through the last trial `\eta` is used at every
    reduction, whereas [2]_ uses, from the second backtrack on, the cubic
    interpolation of `f(0)`, `f'(0)` and the last two trial values,
    Eq. (6.3.18) (p. 128); the bound `0.5 \eta` is also imposed at the
    first reduction, where [2]_ imposes only the lower bound `0.1`, the
    quadratic backtrack being then smaller than `1/(2(1 - \alpha))`
    (p. 127; Algorithm A6.3.1, steps 10.3c.1E.4 and 10.3c.4); and [2]_ has
    no maximum number of reductions, but limits the length of the step to
    ``maxstep`` and stops when the step length becomes smaller than a
    minimum, returning the current point.

    References
    ----------
    .. [1] Crisfield, M. A., Non-linear Finite Element Analysis of Solids and
       Structures, Volume 1: Essentials, John Wiley & Sons, Chichester,
       1991.
    .. [2] Dennis, J. E., Jr. and Schnabel, R. B., Numerical Methods for
       Unconstrained Optimization and Nonlinear Equations, SIAM,
       Philadelphia, 1996 (first published by Prentice-Hall, Englewood
       Cliffs, NJ, 1983), pages of the SIAM edition.
       https://doi.org/10.1137/1.9781611971200

    """
    modified_NR = an.modified_NR
    fext_total = total*fext
    if not lbd_dependent:
        q = None
    q_ref = None
    Rnorms = []
    iter_kT = 0
    solve_kT = None
    fint = None

    iteration = 0
    while True:
        iteration += 1
        if fint is None:
            fint = callbacks.calc_fint(pb, c + dc, total)
        R = fext_total - fint
        if q is not None:
            q_ref = total*q

        conv, Rnorm, zeta = check_convergence(an, R, fint, fext_total,
                                              q_ref=q_ref, w=w)
        _log_iteration(an, iteration, R, zeta, silent)
        if conv:
            return True, dc, q
        if _check_divergence(an, Rnorm, Rnorms, iteration, silent=silent):
            return False, dc, q

        # NOTE the factorization of the tangent stiffness matrix is reused
        #      while the matrix is not updated
        try:
            if modified_NR and iteration > 1 and iter_kT < an.compute_every_n:
                iter_kT += 1
            elif modified_NR and iteration == 1 and kT0 is not None:
                solve_kT = _factorize(kT0)
                iter_kT = 1
            else:
                solve_kT = _factorize(callbacks.calc_kT(pb, c + dc, total))
                iter_kT = 1
                if lbd_dependent:
                    # NOTE dR/dlbd at the iterate of the tangent stiffness
                    #      matrix
                    q = callbacks.calc_load_vector(pb, c + dc, fext, total)
        except RuntimeError:
            warn('Singular tangent stiffness matrix', level=2, silent=silent)
            return False, dc, q

        varc = solve_kT(R)

        eta = 1.
        fint = None
        if an.line_search:
            # NOTE sufficient decrease of phi(eta) = ||R(eta)||**2, assuming
            #      phi'(0) = -2*phi(0), exact for a Newton direction
            fint = callbacks.calc_fint(pb, c + dc + varc, total)
            phi0 = Rnorm**2
            phi1 = _norm(fext_total - fint, w)**2
            iter_line_search = 0
            while not phi1 <= (1 - 2*LINE_SEARCH_ALPHA*eta)*phi0:
                if iter_line_search == an.max_iter_line_search:
                    warn('Line-search: maximum number of iterations achieved',
                         level=3, silent=silent)
                    break
                iter_line_search += 1
                # NOTE minimum of the quadratic interpolation of phi,
                #      safeguarded
                A = (phi1 - phi0 + 2*phi0*eta)/eta**2
                eta_new = phi0/A if (np.isfinite(A) and A > 0) else 0.5*eta
                eta = min(max(eta_new, 0.1*eta), 0.5*eta)
                fint = callbacks.calc_fint(pb, c + dc + eta*varc, total)
                phi1 = _norm(fext_total - fint, w)**2
            if eta != 1.:
                msg('Line-search: eta = %1.5f' % eta, level=3, silent=silent)
        dc = dc + eta*varc


def _solver_NR(an, pb, silent=False, initialInc=None):
    r"""Newton-Raphson solver with load control

    Used by :meth:`.Analysis.static` when ``NL_method='NR'``. The load factor
    goes from zero to one in increments; at each increment the equilibrium
    of the internal force vector with the external force vector scaled by
    the load factor is solved with the iterations of :func:`_NR_iterations`,
    starting from a predictor extrapolated from the previous converged
    increment, and the converged load factors and solutions are appended to
    ``an.increments`` and ``an.cs``.

    The variant of the iterations is selected by ``an.modified_NR``,
    ``an.compute_every_n``, ``an.kT_initial_state``, ``an.line_search`` and
    ``an.max_iter_line_search``, and the convergence criterion by
    ``an.convergence``, see :func:`.check_convergence`. A step that does
    not converge within ``an.maxNumIter`` iterations, that diverges or that
    converges too slowly is repeated with the load increment multiplied by
    ``INC_CUT_FACTOR``, see :func:`_check_divergence`. After a step that
    converged without being cut, the load increment is multiplied by
    ``INC_GROWTH_FACTOR``, limited by ``an.maxInc``. The last step is
    adjusted such that the analysis finishes at a load factor of exactly
    ``1.0``. The analysis stops earlier, keeping the converged increments,
    when the load increment becomes smaller than ``an.minInc``.

    Parameters
    ----------
    an : :class:`.Analysis`
        Analysis object; the converged load factors and solutions are
        appended to ``an.increments`` and ``an.cs``.
    pb : object
        Object with the callables, ``an`` itself or a :class:`.FreeDofs`.
    silent : bool, optional
        Whether the log messages are suppressed. Default ``False``.
    initialInc : float or None, optional
        Initial load increment, dimensionless; ``an.initialInc`` is used if
        ``None``. Default ``None``.

    Notes
    -----
    The load factor `\lambda` goes from zero to one in increments, solving
    at each increment

    .. math::

        \{R\} = \lambda \{F_{ext}\} - \{F_{int}(c)\} = \{0\}

    Each increment starts from a predictor that extrapolates the previous
    converged increment of the solution, scaled by the ratio between the new
    and the previous load increments. The first increment uses the linear
    solution as predictor: `\Delta\lambda [K_{C0}]^{-1}\{F_{ext}\}`, with
    `[K_{C0}]` the linear stiffness matrix ``calc_kC()``, which also gives
    the scaling of Crisfield's criterion, computed once for the whole
    analysis, see :mod:`.convergence`.

    The callables that accept the keyword argument ``inc`` receive the load
    factor `\lambda` of the step, see :mod:`.callbacks`, which allows
    configuration-dependent loads such as follower pressures.

    When ``calc_fext`` accepts the keyword argument ``lbd``, e.g. for
    prescribed displacements imposed exactly, the load vector `\{q\} =
    \partial R/\partial \lambda` at `c = 0`, `\lambda = 0` gives the
    predictor of the first increment, its linear solution `[K_{C0}]^{-1}
    \{q\}`, instead of the linear solution of the reference vector, which is
    null under a prescribed displacement alone, and `\lambda \{q\}` is an
    additional reference of the convergence criterion. `\{q\}` is evaluated
    together with the tangent stiffness matrix, at the same iterate, i.e.
    ``calc_fext`` is called once per update of the tangent stiffness matrix,
    and the last `\{q\}` evaluated is the reference of the next convergence
    checks, also in the next step.

    By default full Newton-Raphson is used, i.e. the tangent stiffness matrix
    is rebuilt at every iteration, which gives quadratic convergence when it
    is the exact Jacobian of the internal force vector. With
    ``an.modified_NR = True`` it is rebuilt only at the first iteration of
    each step and every ``an.compute_every_n`` iterations, and its LU
    factorization is reused in the other iterations, at the cost of a
    linear convergence rate; at the first step with ``an.kT_initial_state =
    False``, `[K_{C0}]` is used instead of the tangent at the first
    iteration. The full and modified Newton-Raphson methods, combined with
    incremental predictors, are described in [1]_ (Sections 1.2.2 and
    1.2.3, pp. 8-13), with the quadratic and the linear convergence rates.

    The step that would end within ``max(an.minInc, 1e-12)`` of the load
    factor 1.0, or beyond it, is replaced by the step to exactly 1.0.

    References
    ----------
    .. [1] Crisfield, M. A., Non-linear Finite Element Analysis of Solids and
       Structures, Volume 1: Essentials, John Wiley & Sons, Chichester,
       1991.

    """
    msg('___________________________________________', level=1, silent=silent)
    msg('                                           ', level=1, silent=silent)
    msg('Newton-Raphson solver', level=1, silent=silent)
    msg('___________________________________________', level=1, silent=silent)
    msg('Initializing...', level=1, silent=silent)

    if initialInc is None:
        initialInc = an.initialInc
    inc = min(initialInc, 1.)
    min_last_inc = max(an.minInc, 1.e-12)

    fext = pb.calc_fext(inc=1., silent=True)
    kC0 = pb.calc_kC(silent=True)
    w = _weights(an, kC0)
    q = None
    lbd_dependent = callbacks.is_load_factor_dependent(pb)
    if lbd_dependent:
        # NOTE dR/dlbd at c = 0 and lbd = 0
        q = callbacks.calc_load_vector(pb, np.zeros_like(fext), fext, 0.)
    # NOTE predictor of the first step: linear solution scaled by the load
    #      factor
    dc_last = solve(kC0, fext if q is None else q)
    inc_last = 1.
    c = np.zeros_like(dc_last)
    total_last = 0.
    step_cut = False

    step_num = 1

    while True:
        if total_last + inc >= 1. - min_last_inc:
            total = 1.
            inc = 1. - total_last
        else:
            total = total_last + inc
        msg('Step %d, attempting load factor %1.5f' % (step_num, total),
            level=1, silent=silent)

        dc = (inc/inc_last)*dc_last
        kT0 = None
        if an.modified_NR and step_num == 1 and not an.kT_initial_state:
            kT0 = kC0
        converged, dc, q_new = _NR_iterations(an, pb, c, dc, total, fext, w,
                kT0=kT0, silent=silent, lbd_dependent=lbd_dependent, q=q)

        if converged:
            msg('Converged at load factor %1.5f' % total, level=2,
                silent=silent)
            # NOTE the dR/dlbd of a step that failed is not kept
            q = q_new
            c = c + dc
            total_last = total
            an.cs.append(c.copy())
            an.increments.append(total)
            if total == 1.:
                break
            dc_last = dc
            inc_last = inc
            if not step_cut:
                inc = min(INC_GROWTH_FACTOR*inc, an.maxInc)
            step_cut = False
            step_num += 1

        else:
            inc *= INC_CUT_FACTOR
            step_cut = True
            msg('Reseting step with reduced load increment %1.5f' % inc,
                level=1, silent=silent)
            if inc < an.minInc:
                warn('Minimum step size of %1.5f achieved! Analysis stopped '
                     'at load factor %1.5f' % (an.minInc, total_last),
                     level=1, silent=silent)
                break

    msg('Finished Non-Linear Static Analysis', silent=silent)
    msg('    load factor %1.5f' % total_last, level=1, silent=silent)
