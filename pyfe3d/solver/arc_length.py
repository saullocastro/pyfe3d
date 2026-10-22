r"""
Arc-length solvers (:mod:`pyfe3d.solver.arc_length`)
====================================================

.. currentmodule:: pyfe3d.solver.arc_length

The arc-length methods of Riks and of Crisfield share the implementation of
:func:`_solver_arc_length`, and are selected with ``Analysis.NL_method =
'arc_length_riks'`` or ``Analysis.NL_method = 'arc_length_crisfield'``. The
module-level constants control the arc-length scaling and adaptation.

"""
import numpy as np

from .logger import msg, warn
from .static import solve, _factorize
from .convergence import check_convergence
from .newton_raphson import (_check_divergence, _NR_iterations, _weights,
                             _log_iteration, INC_CUT_FACTOR)
from . import callbacks


#: Scaling factor of the load-factor term in the arc-length constraint
PSI = 1.
#: Desired number of corrections per step, used to adapt the arc length
DESIRED_NUM_ITER = 4
#: Maximum factor to increase the arc length after a converged step
MAX_GROWTH_FACTOR = 1.5
#: Minimum factor to reduce the arc length after a converged step
MIN_GROWTH_FACTOR = 0.5
#: Maximum number of converged steps of an arc-length analysis
MAX_NUM_STEPS = 1000

#: Names of the arc-length methods, see :func:`_solver_arc_length`
METHODS = ('riks', 'crisfield')


def _solver_arc_length(an, pb, method, silent=False):
    r"""Arc-length solver

    Traces the equilibrium path of the model with an arc-length method, in
    which the load factor is an unknown of each step besides the solution
    vector, such that limit points can be passed. ``method`` selects how the
    load factor of each iteration is computed. Used by
    :meth:`.Analysis.static` when ``NL_method`` is ``'arc_length_riks'`` or
    ``'arc_length_crisfield'``.

    The steps are controlled by the attributes ``initialInc``, ``minInc``,
    ``maxInc`` and ``maxArcLength`` of ``an``, and by the module-level
    constants; the iterations of each step by ``convergence``, ``relTOL``,
    ``absTOL``, ``maxNumIter`` and ``too_slow_TOL``, with the checks of
    :func:`.check_convergence` and
    :func:`.newton_raphson._check_divergence`; and ``an.modified_NR``,
    ``an.compute_every_n`` and ``an.kT_initial_state`` control how often the
    tangent stiffness matrix is updated. A step that fails is repeated with
    the arc length multiplied by ``INC_CUT_FACTOR``. The analysis stops
    when:

    - the load factor reaches 1.0, the last solution being computed at a load
      factor of exactly 1.0;
    - the cumulative arc length reaches ``an.maxArcLength``;
    - the arc length becomes smaller than ``an.minInc``;
    - ``MAX_NUM_STEPS`` steps converged;
    - the tangent stiffness matrix at the last converged state is singular.

    Parameters
    ----------
    an : :class:`.Analysis`
        Analysis object; the converged load factors and solutions are
        appended to ``an.increments`` and ``an.cs``.
    pb : object
        Object with the callables, ``an`` itself or a :class:`.FreeDofs`.
    method : str
        ``'riks'`` or ``'crisfield'``.
    silent : bool, optional
        Whether the log messages are suppressed. Default ``False``.

    Raises
    ------
    ValueError
        When ``method`` is not one of :data:`METHODS`.

    Notes
    -----
    The equilibrium path `\{F_{int}(c)\} = \lambda \{F_{ext}\}` is traced with
    the arc-length constraint

    .. math::

        \frac{\{\Delta c\}^T\{\Delta c\}}{\{u_{ref}\}^T\{u_{ref}\}}
        + \psi^2 \Delta\lambda^2 = \Delta s^2

    where `\{\Delta c\}` and `\Delta\lambda` are the increments of the current
    step and `\{u_{ref}\}` is the linear solution for `\lambda=1`, which makes
    the arc length `\Delta s` independent of a uniform change of the units
    of the displacements; it is not independent of a change of the units of
    the rotations only. In the linear regime `\Delta s \approx \sqrt{1 +
    \psi^2} \Delta\lambda`, see [4]_ for the arc-length methods. `\psi` is
    the constant ``PSI``, and `\{u_{ref}\}^T\{u_{ref}\}` is replaced by 1
    when it is null. With the inner product

    .. math::

        \langle (a, \alpha), (b, \beta) \rangle =
        \frac{\{a\}^T\{b\}}{\{u_{ref}\}^T\{u_{ref}\}} + \psi^2 \alpha \beta

    the constraint reads `g = \frac{1}{2}(\langle (\Delta c, \Delta\lambda),
    (\Delta c, \Delta\lambda) \rangle - \Delta s^2) = 0`.

    .. rubric:: Predictor

    Each step starts from a tangent predictor, whose direction follows the
    previous increment, allowing limit points to be passed. With
    `\{\delta c_q\} = [K_T]^{-1}\{F_{ext}\}` at the last converged state,
    the predictor

    .. math::

        \Delta\lambda = \pm \frac{\Delta s}{\sqrt{\langle (\delta c_q, 1),
        (\delta c_q, 1) \rangle}}, \qquad
        \{\Delta c\} = \Delta\lambda \{\delta c_q\}

    satisfies the constraint, with the sign of `\langle (\delta c_q, 1),
    (\Delta c_{prev}, \Delta\lambda_{prev}) \rangle`, where
    `(\Delta c_{prev}, \Delta\lambda_{prev})` is the increment of the
    previous converged step, and a positive sign at the first step. The
    initial arc length, `\Delta s_0 = initialInc \sqrt{\langle (\delta c_q,
    1), (\delta c_q, 1) \rangle}` at the origin, corresponds to a load factor
    increment of ``an.initialInc`` along the initial tangent.

    .. rubric:: Corrections

    At each iteration, the residual is `\{R\} = (\lambda + \Delta\lambda)
    \{F_{ext}\} - \{F_{int}(c + \Delta c)\}`, and the bordered system is
    solved with two solutions of the tangent stiffness matrix,
    `\{\delta c_R\} = [K_T]^{-1}\{R\}` and `\{\delta c_q\} =
    [K_T]^{-1}\{F_{ext}\}`; the increment is updated to `(\Delta c +
    \delta c_R + \delta\lambda \, \delta c_q, \Delta\lambda +
    \delta\lambda)`, where the correction of the load factor
    `\delta\lambda` comes from the constraint:

    - ``method='riks'``: the constraint is linearized at the current
      increment (updated normal plane), which is Newton-Raphson applied to
      the augmented system and gives quadratic convergence, see [1]_ and
      [3]_:

      .. math::

          \delta\lambda = -\frac{g + \langle (\Delta c, \Delta\lambda),
          (\delta c_R, 0) \rangle}{\langle (\Delta c, \Delta\lambda),
          (\delta c_q, 1) \rangle}

    - ``method='crisfield'``: the quadratic (spherical) constraint is solved
      exactly, choosing the root with the smallest angle to the current
      increment. When the roots are complex, the linearized constraint is
      used, see [2]_. With `\{u\} = \{\Delta c\} + \{\delta c_R\}`, the
      constraint `\langle (u + \delta\lambda \, \delta c_q, \Delta\lambda +
      \delta\lambda), (u + \delta\lambda \, \delta c_q, \Delta\lambda +
      \delta\lambda) \rangle = \Delta s^2` is the quadratic equation
      `a \, \delta\lambda^2 + b \, \delta\lambda + c = 0` with
      `a = \langle (\delta c_q, 1), (\delta c_q, 1) \rangle`,
      `b = 2 \langle (u, \Delta\lambda), (\delta c_q, 1) \rangle` and
      `c = \langle (u, \Delta\lambda), (u, \Delta\lambda) \rangle -
      \Delta s^2`. Its roots are computed as `q/a` and `c/q`, with
      `q = -\frac{1}{2}(b + \mathrm{sign}(b) \sqrt{b^2 - 4 a c})`, and the
      root that maximizes the inner product of the new increment with the
      current one `(\Delta c, \Delta\lambda)` is chosen, i.e. the smallest
      angle, since both have the length `\Delta s`.

    The step fails when `\delta\lambda` cannot be computed or is not
    finite. With full Newton-Raphson, `[K_T]` is updated at every iteration
    at the current iterate `(c + \Delta c, \lambda + \Delta\lambda)`; with
    ``an.modified_NR = True`` the factorization at the last converged state
    is used for the first ``an.compute_every_n`` iterations and the tangent
    is then updated every ``an.compute_every_n`` iterations; at the first
    step with ``an.kT_initial_state = False`` the linear constitutive
    stiffness matrix replaces the tangent of the initial state.

    The convergence criterion is the one of the Newton-Raphson solver, see
    :func:`.check_convergence`, at the current load factor `\lambda +
    \Delta\lambda`, the arc-length method changing only how the load factor
    is updated and not the residual. Both criteria also consider the
    external force vector of the increment, `\Delta\lambda
    \{F_{ext}\}`, as a reference.

    .. rubric:: Adaptation of the arc length

    After each converged step, the arc length is multiplied by
    ``sqrt(DESIRED_NUM_ITER/num_iter)``, bounded by ``MIN_GROWTH_FACTOR`` and
    ``MAX_GROWTH_FACTOR`` and limited by ``an.maxInc``, where ``num_iter`` is
    the number of corrections of the step. The factor is at most 1 after a
    step that was repeated, and the limit is in fact
    ``max(an.maxInc, initial arc length)``. When a step fails, the arc length
    is multiplied by ``INC_CUT_FACTOR`` and the step is repeated from the
    same tangent predictor direction. The square root of the ratio is the
    rule recommended by Ramm [3]_ (p. 76), who found that the ratio itself,
    as used by Crisfield [2]_ (Eq. 15), makes the number of iterations
    oscillate from step to step.

    When the load factor 1.0 is exceeded within a converged step, from
    `\lambda < 1` to `\lambda + \Delta\lambda > 1`, the step is replaced by
    Newton-Raphson iterations at exactly `\lambda = 1`, see
    :func:`.newton_raphson._NR_iterations`, from the predictor
    `t \{\Delta c\}` with `t = (1 - \lambda)/\Delta\lambda`. When these
    iterations do not converge, the step is repeated with a shorter arc
    length.

    .. rubric:: Configuration- and load-factor-dependent loads

    When ``calc_fext`` accepts the keyword argument ``c``, `\{F_{ext}\}` in
    `\{\delta c_q\}` and in the tangent predictor is the load vector of the
    current configuration ``calc_fext(inc=1., c=c)``, the derivative
    `\partial R/\partial \lambda` of the residual, whereas `\{u_{ref}\}`
    remains the linear solution of the reference vector. The callables that
    accept ``inc`` receive the current load factor, see :mod:`.callbacks`.

    When ``calc_fext`` also accepts the keyword argument ``lbd``, e.g. for
    prescribed displacements imposed exactly, it receives the load factor
    together with the configuration, `\{F_{ext}\}` above being then
    `\partial R/\partial \lambda` at the current `c` and `\lambda`;
    `\{u_{ref}\}` is the linear solution of `\partial R/\partial \lambda`
    at `c = 0`, `\lambda = 0`, since the reference vector may be null, e.g.
    under a prescribed displacement alone; and `\lambda \, \partial
    R/\partial \lambda` is an additional reference of the convergence
    criterion. `\partial R/\partial \lambda` is evaluated together with the
    tangent stiffness matrix, at the same iterate, i.e. once per update of
    the tangent stiffness matrix, and used in the bordered solution; the
    last one evaluated is the reference of the next convergence checks. At
    the converged state it is kept with the tangent stiffness matrix when
    the step is repeated with a shorter arc length.

    References
    ----------
    .. [1] Riks, E., An incremental approach to the solution of snapping and
       buckling problems, International Journal of Solids and Structures,
       15(7), 529-551, 1979.
       https://doi.org/10.1016/0020-7683(79)90081-7
    .. [2] Crisfield, M. A., A fast incremental/iterative solution procedure
       that handles "snap-through", Computers & Structures, 13(1-3), 55-62,
       1981. https://doi.org/10.1016/0045-7949(81)90108-5
    .. [3] Ramm, E., Strategies for tracing the nonlinear response near
       limit points, in: Wunderlich, W., Stein, E., Bathe, K. J. (eds.),
       Nonlinear Finite Element Analysis in Structural Mechanics, Springer,
       63-89, 1981. https://doi.org/10.1007/978-3-642-81589-8_5
    .. [4] Crisfield, M. A., Non-linear Finite Element Analysis of Solids and
       Structures, Volume 1: Essentials, John Wiley & Sons, Chichester,
       1991.

    """
    if method not in METHODS:
        raise ValueError('Invalid arc-length method: %s' % method)
    msg('___________________________________________', level=1, silent=silent)
    msg('                                           ', level=1, silent=silent)
    msg('Arc-length solver, %s method' % method.capitalize(), level=1,
        silent=silent)
    msg('___________________________________________', level=1, silent=silent)
    msg('Initializing...', level=1, silent=silent)
    modified_NR = an.modified_NR

    fext = pb.calc_fext(inc=1., silent=True)
    kC0 = pb.calc_kC(silent=True)
    w = _weights(an, kC0)
    lbd_dependent = callbacks.is_load_factor_dependent(pb)
    if lbd_dependent:
        # NOTE dR/dlbd at c = 0 and lbd = 0
        q0 = callbacks.calc_load_vector(pb, np.zeros_like(fext), fext, 0.)
        u_ref = solve(kC0, q0)
    else:
        u_ref = solve(kC0, fext)
    scale2 = u_ref.dot(u_ref)
    if not scale2 > 0:
        scale2 = 1.
    psi2 = PSI**2

    def inner(dc1, dlbd1, dc2, dlbd2):
        return dc1.dot(dc2)/scale2 + psi2*dlbd1*dlbd2

    c = np.zeros_like(u_ref)
    lbd = 0.
    dc_last = None
    dlbd_last = None
    arc_length = None
    total_arc_length = 0.
    step_cut = False
    solve_kT_conv = None

    step_num = 1

    while True:
        if step_num > MAX_NUM_STEPS:
            warn('Maximum number of steps of %d achieved! Analysis stopped at '
                 'load factor %1.5f' % (MAX_NUM_STEPS, lbd), level=1,
                 silent=silent)
            break

        # NOTE tangent stiffness matrix at the last converged state
        if solve_kT_conv is None:
            if step_num == 1 and modified_NR and not an.kT_initial_state:
                kT = kC0
            else:
                kT = callbacks.calc_kT(pb, c, lbd)
            try:
                solve_kT_conv = _factorize(kT)
            except RuntimeError:
                warn('Singular tangent stiffness matrix at the last converged '
                     'state! Analysis stopped', level=1, silent=silent)
                break
            # NOTE load vector of the converged configuration, dR/dlbd, kept
            #      with the tangent stiffness matrix when the step is repeated
            if step_num == 1 and lbd_dependent:
                q_conv = q0
            else:
                q_conv = callbacks.calc_load_vector(pb, c, fext, lbd)
        dc_q_conv = solve_kT_conv(q_conv)
        norm_q = np.sqrt(inner(dc_q_conv, 1., dc_q_conv, 1.))
        if arc_length is None:
            arc_length = an.initialInc*norm_q
            max_arc_length_inc = max(an.maxInc, arc_length)

        # NOTE tangent predictor
        sign = 1.
        if dc_last is not None:
            sign = np.sign(inner(dc_q_conv, 1., dc_last, dlbd_last))
            if sign == 0:
                sign = 1.
        dlbd = sign*arc_length/norm_q
        dc = dlbd*dc_q_conv

        msg('Step %d, lbd %1.5f, arc-length increment %1.5f'
            % (step_num, lbd, arc_length), level=1, silent=silent)

        converged = False
        Rnorms = []
        solve_kT = solve_kT_conv
        dc_q = dc_q_conv
        # NOTE the last dR/dlbd evaluated, with the tangent stiffness matrix
        q_iter = q_conv
        iter_kT = 0
        iteration = 0
        while True:
            iteration += 1
            fint = callbacks.calc_fint(pb, c + dc, lbd + dlbd)
            fext_total = (lbd + dlbd)*fext
            R = fext_total - fint
            q_ref = None
            if lbd_dependent:
                q_ref = (lbd + dlbd)*q_iter
            conv, Rnorm, zeta = check_convergence(an, R, fint, fext_total,
                                                  fext_ref=dlbd*fext,
                                                  q_ref=q_ref, w=w)
            _log_iteration(an, iteration, R, zeta, silent, lbd=lbd + dlbd)
            if conv:
                converged = True
                break
            if _check_divergence(an, Rnorm, Rnorms, iteration, silent=silent):
                break

            # NOTE with modified_NR the tangent of the last converged state
            #      is used first, otherwise it is updated at every iteration
            if modified_NR and iter_kT < an.compute_every_n:
                iter_kT += 1
            else:
                try:
                    solve_kT = _factorize(callbacks.calc_kT(pb, c + dc,
                                                            lbd + dlbd))
                except RuntimeError:
                    warn('Singular tangent stiffness matrix', level=2,
                         silent=silent)
                    break
                if lbd_dependent:
                    # NOTE dR/dlbd at the iterate of the tangent stiffness
                    #      matrix
                    q_iter = callbacks.calc_load_vector(pb, c + dc, fext,
                                                        lbd + dlbd)
                    dc_q = solve_kT(q_iter)
                else:
                    dc_q = solve_kT(callbacks.calc_load_vector(pb, c + dc,
                                                               fext))
                iter_kT = 1
            dc_R = solve_kT(R)

            # NOTE linearized constraint
            den = inner(dc, dlbd, dc_q, 1.)
            g = 0.5*(inner(dc, dlbd, dc, dlbd) - arc_length**2)
            varlbd = None
            if den != 0:
                varlbd = -(g + inner(dc, dlbd, dc_R, 0.))/den
            if method == 'crisfield':
                u = dc + dc_R
                a = inner(dc_q, 1., dc_q, 1.)
                b = 2*inner(u, dlbd, dc_q, 1.)
                cc = inner(u, dlbd, u, dlbd) - arc_length**2
                disc = b**2 - 4*a*cc
                if disc >= 0:
                    sq = np.sqrt(disc)
                    qq = -0.5*(b + np.copysign(sq, b))
                    roots = [qq/a]
                    if qq != 0:
                        roots.append(cc/qq)
                    cosines = [inner(dc, dlbd, u + r*dc_q, dlbd + r)
                               for r in roots]
                    varlbd = roots[int(np.argmax(cosines))]
                else:
                    msg('Complex roots, using the linearized constraint',
                        level=3, silent=silent)
            if varlbd is None or not np.isfinite(varlbd):
                warn('Diverged - arc-length constraint cannot be satisfied',
                     level=2, silent=silent)
                break
            dc = dc + dc_R + varlbd*dc_q
            dlbd = dlbd + varlbd

        finished = False
        if converged and lbd + dlbd > 1. and lbd < 1.:
            msg('Load factor 1.0 exceeded, correcting to load factor 1.0...',
                level=1, silent=silent)
            t = (1. - lbd)/dlbd
            converged, dc1, _ = _NR_iterations(an, pb, c, t*dc, 1., fext, w,
                    kT0=None, silent=silent, lbd_dependent=lbd_dependent,
                    q=q_iter)
            if converged:
                msg('Converged at load factor 1.00000', level=2,
                    silent=silent)
                an.cs.append((c + dc1).copy())
                an.increments.append(1.)
                finished = True

        if finished:
            break

        if converged:
            c = c + dc
            lbd = lbd + dlbd
            total_arc_length += arc_length
            msg('Converged at load factor %1.5f, total arc length %1.5f'
                % (lbd, total_arc_length), level=2, silent=silent)
            an.cs.append(c.copy())
            an.increments.append(lbd)
            if lbd == 1.:
                break
            if total_arc_length >= an.maxArcLength:
                msg('Maximum specified arc-length of %1.5f achieved'
                    % an.maxArcLength, level=1, silent=silent)
                break
            dc_last = dc
            dlbd_last = dlbd
            factor = np.sqrt(DESIRED_NUM_ITER/max(iteration - 1, 1))
            factor = min(max(factor, MIN_GROWTH_FACTOR), MAX_GROWTH_FACTOR)
            if step_cut:
                factor = min(factor, 1.)
            arc_length = min(factor*arc_length, max_arc_length_inc)
            step_cut = False
            solve_kT_conv = None
            step_num += 1

        else:
            arc_length *= INC_CUT_FACTOR
            step_cut = True
            msg('Reseting step with reduced arc-length increment %1.5f'
                % arc_length, level=1, silent=silent)
            if arc_length < an.minInc:
                warn('Minimum arc-length increment of %1.5f achieved! '
                     'Analysis stopped at load factor %1.5f'
                     % (an.minInc, lbd), level=1, silent=silent)
                break

    msg('Finished Non-Linear Static Analysis', silent=silent)
    msg('    load factor %1.5f, total arc length %1.5f'
        % (lbd if not an.increments else an.increments[-1],
           total_arc_length), level=1, silent=silent)
