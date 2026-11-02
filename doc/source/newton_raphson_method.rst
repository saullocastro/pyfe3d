The Newton-Raphson method
=========================

``Analysis.NL_method = 'NR'``, the default, solves the non-linear static
problem of :doc:`nonlinear_analyses` under load control: the load factor
`\lambda` goes from zero to one in increments, and at each increment the
equilibrium `\boldsymbol{R}(\boldsymbol{u}, \lambda) = \boldsymbol{0}` is solved for the
configuration by Newton's method, with the solver
:func:`pyfe3d.solver.newton_raphson._solver_NR` and the iterations of
:func:`pyfe3d.solver.newton_raphson._NR_iterations`. The same iterations
correct the increments of the arc-length methods, see
:doc:`arc_length_methods`.


Newton's method at a fixed load factor
--------------------------------------

Expanding the residual about the current iterate `\boldsymbol{u}^{(i)}` and
truncating after the linear term gives the correction that would make the
residual vanish if it were linear:

.. math::
    :label: nr-iteration

    \boldsymbol{K}_T^{(i)} \, \delta \boldsymbol{u} = -\boldsymbol{R}^{(i)}, \qquad
    \boldsymbol{u}^{(i+1)} = \boldsymbol{u}^{(i)} + \eta \, \delta \boldsymbol{u}, \qquad
    \boldsymbol{R}^{(i)} = \boldsymbol{F}_{int}(\boldsymbol{u}^{(i)}, \lambda) - \lambda \boldsymbol{F}_{ext}

with `\boldsymbol{K}_T^{(i)} = \partial \boldsymbol{R}/\partial \boldsymbol{u}` at
`\boldsymbol{u}^{(i)}`, given by ``calc_kC + calc_kG``, and `\eta = 1` without
line search. When `\boldsymbol{K}_T` is the exact Jacobian, the error is squared
at each iteration near the solution, quadratic convergence (Crisfield 1991,
Sec. 1.2.2, p. 9), and Crisfield's `\zeta` of :doc:`convergence_criteria`
falls as `\zeta^{(i+1)} \sim (\zeta^{(i)})^2`. The tangent stiffness matrix
controls only the rate of convergence; the converged configuration is
controlled by the residual alone. An approximate tangent, e.g. a missing
``calc_kG`` or the symmetric part of an unsymmetric load stiffness, still
converges to the same equilibrium when it converges, linearly and in more
iterations, which ``tests/test_solver_newton_raphson.py`` measures.

The tangent stiffness matrix is factorized with the sparse LU
factorization of :func:`scipy.sparse.linalg.splu`, valid for unsymmetric
matrices, see :func:`pyfe3d.solver.static._factorize`: the load stiffness
of a follower load needs no symmetrisation. A singular tangent, e.g. at a
limit point, does not raise; it stops the iterations of the step, which is
then cut, see below.


Full and modified Newton-Raphson
--------------------------------

With full Newton-Raphson, ``an.modified_NR = False``, the default,
`\boldsymbol{K}_T` is evaluated and factorized at every iteration. With
``an.modified_NR = True``, it is evaluated at the first iteration of each
step and then every ``an.compute_every_n`` iterations, 6 by default, and
its factorization is reused in between, each iteration costing only one
evaluation of the internal forces and a forward and backward substitution;
the convergence is then linear (Crisfield 1991, Secs. 1.2.2 and 1.2.3, pp.
8-13). With ``an.kT_initial_state = False``, the first iteration of the
first step uses `\boldsymbol{K}_{C0}` instead of the tangent of the initial state,
which differ when the initial state is already stressed or deformed, e.g.
with initial imperfections in the callables. The modified method pays off
when the factorization dominates the cost and the path is mildly
non-linear; near limit points and with tight tolerances the full method is
the robust choice.


Increments
----------

The increments of the load factor are controlled by the attributes of
:class:`.Analysis` and by the module constants of
:mod:`pyfe3d.solver.newton_raphson`:

1. The first increment is ``an.initialInc``, 0.1 by default, and its
   predictor is the linear solution scaled by it, `\Delta \boldsymbol{u} = \Delta
   \lambda \, \boldsymbol{K}_{C0}^{-1} \boldsymbol{F}_{ext}`, the solution of the first
   Newton iteration from the undeformed state. When ``calc_fext`` declares
   ``lbd``, e.g. under prescribed displacements, `\boldsymbol{F}_{ext}` is
   replaced by `-\partial \boldsymbol{R}/\partial \lambda` at `\boldsymbol{u} =
   \boldsymbol{0}`, `\lambda = 0`, which is not null under a prescribed
   displacement alone, see :doc:`configuration_dependent_loads`.
2. Each next increment starts from the converged increment of the
   configuration of the previous step, scaled by the ratio of the new and
   the previous load increments, a secant extrapolation of the path.
3. After a step that converged without being cut, the load increment is
   multiplied by ``INC_GROWTH_FACTOR`` `= 1.1111`, up to ``an.maxInc``.
4. A step that fails is repeated from the last converged state with the
   load increment multiplied by ``INC_CUT_FACTOR`` `= 0.5`; the analysis
   stops, keeping the converged increments, when the increment falls below
   ``an.minInc``.
5. A step that would end within ``max(an.minInc, 1e-12)`` of the load
   factor 1.0, or beyond it, is replaced by the step to 1.0 exactly, such
   that a completed analysis ends with ``an.increments[-1] == 1.``.

A step fails when the iterations are stopped by
:func:`pyfe3d.solver.newton_raphson._check_divergence` or by a singular
tangent. With `r_i` the norm of the residual at the iteration `i` of the
step, in the norm of the convergence criterion, the iterations stop at
iteration `j` when `r_j` is not finite; when ``an.maxNumIter``
corrections, 30 by default, were done; and, after
``NUM_ITER_BEFORE_DIVERGENCE_CHECK`` `= 5` iterations, when the residual
diverges, `r_j > 10^3 r_1`, or decreases too slowly,

.. math::
    :label: nr-too-slow

    \min(r_{j-W+1}, \ldots, r_j) > (1 - \tau) \min(r_1, \ldots, r_{j-W})

with `W` = ``TOO_SLOW_WINDOW`` `= 5` and `\tau` = ``an.too_slow_TOL``,
0.005 by default. The checks are delayed because a non-monotonic residual
in the first iterations is normal, especially in post-buckling, and they
compare norms of the same step only, which with Crisfield's criterion do
not depend on the units of the dofs.

Beyond a limit point of the load, load control has no nearby equilibrium:
the step is cut until ``an.minInc``, or the iterations converge to a
configuration on another part of the path. Such paths are traced with
:doc:`arc_length_methods`.


Line search
-----------

With ``an.line_search = True``, the correction `\delta \boldsymbol{u}` of
:eq:`nr-iteration` is a direction, along which the step `\eta` is
chosen to reduce the merit function

.. math::
    :label: nr-merit

    \phi(\eta) = \| \boldsymbol{R}(\boldsymbol{u}^{(i)} + \eta \, \delta \boldsymbol{u}) \|^2

in the norm of the convergence criterion, the scaled norm
`\|\cdot\|_{\boldsymbol{d}}` of Crisfield's criterion or the Euclidean norm,
such that the line search and the convergence check measure the residual
in the same way, see :func:`pyfe3d.solver.newton_raphson._norm`. Along a
Newton direction, `\phi'(0) = -2 \phi(0)` in any norm whose weights do not
change along the line, exactly when `\boldsymbol{K}_T` is the exact Jacobian
at `\boldsymbol{u}^{(i)}` and approximately with the modified method. The step
is accepted when it satisfies the sufficient-decrease (Armijo) condition

.. math::
    :label: nr-armijo

    \phi(\eta) \le (1 - 2 \alpha \eta) \, \phi(0), \qquad \alpha = 10^{-4}

with `\alpha` = ``LINE_SEARCH_ALPHA``. The full step `\eta = 1` is tried
first, such that the line search does not change the quadratic convergence
of the full method near the solution. Otherwise `\eta` is replaced by the
minimizer of the parabola through `\phi(0)`, `\phi'(0) = -2 \phi(0)` and
`\phi(\eta)`,

.. math::
    :label: nr-backtrack

    \eta \leftarrow \frac{\phi(0)}{A}, \qquad
    A = \frac{\phi(\eta) - \phi(0) + 2 \phi(0) \eta}{\eta^2}

or by `\eta/2` when `A` is not finite or not positive, safeguarded in
`[0.1 \eta, 0.5 \eta]`; after ``an.max_iter_line_search`` reductions, 20 by
default, the last `\eta` is accepted. The internal force vector of the
accepted point is reused in the next residual. This is the backtracking
line search of Dennis and Schnabel (1996, Sec. 6.3.2, Algorithm 6.3.5, and
Sec. 6.5, Eq. (6.5.3)) applied to non-linear equations with the merit
function `\frac{1}{2}\|\boldsymbol{D}_F \boldsymbol{R}\|^2`, with `\boldsymbol{D}_F`
the identity or `\mathrm{diag}(\sqrt{w_j})` of Crisfield's criterion, with
three differences: the quadratic interpolation is used at every reduction,
instead of the cubic one from the second reduction on; the bound `0.5
\eta` is imposed at the first reduction too; and the number of reductions
is limited instead of the length of the step, see
:func:`pyfe3d.solver.newton_raphson._NR_iterations`. The line search
safeguards the iterations when the full correction overshoots, e.g. with
strongly non-linear springs or long increments; it does not make load
control pass a limit point.


Verification
------------

``tests/test_solver_newton_raphson.py`` checks the iterations on analytic
problems with exact tangents, hardening springs, the shallow von Mises
truss of Crisfield (1991, Sec. 1.2, Eq. (1.7)) and a linear problem:
equilibrium, the solution of a dense Newton-Raphson reference, the measured
convergence orders, quadratic with the exact tangent and with the line
search, linear with the modified method and with an inconsistent tangent,
the step cutting and the bookkeeping of the load factor.
``tests/test_solver_nonlinear_static.py`` and
``tests/test_nonlinear_static_newton_raphson.py`` check the quadratic
convergence of the models of plates and beams of the elements.


References
----------

- Crisfield, M. A., Non-linear Finite Element Analysis of Solids and
  Structures, Volume 1: Essentials, John Wiley & Sons, Chichester, 1991.
- Dennis, J. E., Jr. and Schnabel, R. B., Numerical Methods for
  Unconstrained Optimization and Nonlinear Equations, SIAM, Philadelphia,
  1996 (first published by Prentice-Hall, Englewood Cliffs, NJ, 1983).
  https://doi.org/10.1137/1.9781611971200
