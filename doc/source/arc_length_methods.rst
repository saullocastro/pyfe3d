Arc-length methods
==================

Under load control, the load factor of each increment is prescribed, and
the Newton-Raphson iterations of :doc:`newton_raphson_method` fail at a
limit point of the load, where the tangent stiffness matrix is singular and
no equilibrium exists near the current configuration for a larger load.
The arc-length methods make the load factor an unknown of each increment,
together with the configuration, and add one equation that fixes the
length of the increment along the equilibrium path. They trace limit
points, snap-through and snap-back. ``Analysis.NL_method =
'arc_length_riks'`` and ``'arc_length_crisfield'`` select the methods of
Riks (1979) and Crisfield (1981), which share the implementation of
:func:`pyfe3d.solver.arc_length._solver_arc_length` and differ only in how
the correction of the load factor is computed.


The arc-length constraint
-------------------------

With `\Delta \boldsymbol{u}` and `\Delta\lambda` the increments of the
configuration and of the load factor from the last converged point, the
step satisfies, besides the equilibrium `\boldsymbol{R}(\boldsymbol{u} + \Delta
\boldsymbol{u}, \lambda + \Delta\lambda) = \boldsymbol{0}`, the constraint

.. math::
    :label: al-constraint

    \frac{\Delta \boldsymbol{u}^\top \Delta \boldsymbol{u}}{\boldsymbol{u}_{ref}^\top \boldsymbol{u}_{ref}}
    + \psi^2 \Delta\lambda^2 = \Delta s^2

with `\Delta s` the arc length of the step, `\psi` = ``PSI`` `= 1`, and
`\boldsymbol{u}_{ref} = \boldsymbol{K}_{C0}^{-1} \boldsymbol{F}_{ext}` the linear solution of
the reference load, which makes `\Delta s` dimensionless and independent of
a uniform change of the units of the displacements, though not of a change
of the units of the rotations only. In the linear regime, `\Delta
\boldsymbol{u} = \Delta\lambda \, \boldsymbol{u}_{ref}` and `\Delta s = \sqrt{1 +
\psi^2} \, \Delta\lambda`. This is the spherical constraint of Crisfield
(1991, Sec. 9.3.2, Eq. (9.19)), whose displacement term is here scaled and
whose load term, there `\psi^2 \Delta\lambda^2 \boldsymbol{F}_{ext}^\top
\boldsymbol{F}_{ext}`, is not. It is convenient to write it with the inner
product of the space of the configurations and the load factor,

.. math::
    :label: al-inner

    \langle (\boldsymbol{a}, \alpha), (\boldsymbol{b}, \beta) \rangle =
    \frac{\boldsymbol{a}^\top \boldsymbol{b}}{\boldsymbol{u}_{ref}^\top \boldsymbol{u}_{ref}} + \psi^2 \alpha \beta,
    \qquad
    g = \tfrac{1}{2} \left( \langle (\Delta \boldsymbol{u}, \Delta\lambda),
    (\Delta \boldsymbol{u}, \Delta\lambda) \rangle - \Delta s^2 \right) = 0

When `\boldsymbol{u}_{ref}` is null, its squared norm is replaced by 1; when
``calc_fext`` declares ``lbd``, e.g. under prescribed displacements,
`\boldsymbol{u}_{ref}` is the linear solution of `-\partial \boldsymbol{R}/\partial
\lambda` at `\boldsymbol{u} = \boldsymbol{0}`, `\lambda = 0`, see
:doc:`configuration_dependent_loads`.


Predictor
---------

Each step starts from the tangent at the last converged point, `\delta
\boldsymbol{u}_q = \boldsymbol{K}_T^{-1} \boldsymbol{q}`, with `\boldsymbol{q} = -\partial
\boldsymbol{R}/\partial \lambda`, equal to `\boldsymbol{F}_{ext}` for a load that
depends neither on the configuration nor on the load factor:

.. math::
    :label: al-predictor

    \Delta\lambda = \pm \frac{\Delta s}{\sqrt{\langle (\delta \boldsymbol{u}_q, 1),
    (\delta \boldsymbol{u}_q, 1) \rangle}}, \qquad
    \Delta \boldsymbol{u} = \Delta\lambda \, \delta \boldsymbol{u}_q

which satisfies the constraint :eq:`al-constraint`. The sign is that of
`\langle (\delta \boldsymbol{u}_q, 1), (\Delta \boldsymbol{u}_{prev},
\Delta\lambda_{prev}) \rangle`, with the increment of the previous step,
positive at the first step: the predictor continues in the direction of
the path already traced, which makes `\Delta\lambda` negative after a
limit point of the load. The first arc length, `\Delta s_0 =`
``an.initialInc`` `\sqrt{\langle (\delta \boldsymbol{u}_q, 1), (\delta
\boldsymbol{u}_q, 1) \rangle}` at the origin, corresponds to a load factor
increment of ``an.initialInc`` along the initial tangent.


Corrections
-----------

At each iteration, the residual is `\boldsymbol{R} = \boldsymbol{F}_{int}(\boldsymbol{u} +
\Delta \boldsymbol{u}) - (\lambda + \Delta\lambda) \boldsymbol{F}_{ext}`, and the
linearized equilibrium is the bordered system `\boldsymbol{K}_T \delta \boldsymbol{u} -
\delta\lambda \, \boldsymbol{q} = -\boldsymbol{R}`. It is solved with one
factorization and two substitutions,

.. math::
    :label: al-bordered

    \delta \boldsymbol{u}_R = -\boldsymbol{K}_T^{-1} \boldsymbol{R}, \qquad
    \delta \boldsymbol{u}_q = \boldsymbol{K}_T^{-1} \boldsymbol{q}, \qquad
    \delta \boldsymbol{u} = \delta \boldsymbol{u}_R + \delta\lambda \, \delta \boldsymbol{u}_q

and the increment becomes `(\Delta \boldsymbol{u} + \delta \boldsymbol{u}, \Delta\lambda
+ \delta\lambda)`, with `\delta\lambda` from the constraint:

- **Riks**, ``'arc_length_riks'``: the constraint is linearized at the
  current increment, `g + \langle (\Delta \boldsymbol{u}, \Delta\lambda), (\delta
  \boldsymbol{u}, \delta\lambda) \rangle = 0`, which gives

  .. math::
      :label: al-riks

      \delta\lambda = -\frac{g + \langle (\Delta \boldsymbol{u}, \Delta\lambda),
      (\delta \boldsymbol{u}_R, 0) \rangle}{\langle (\Delta \boldsymbol{u}, \Delta\lambda),
      (\delta \boldsymbol{u}_q, 1) \rangle}

  This is Newton's method applied to the system of the equilibrium
  equations augmented with the constraint, with quadratic convergence when
  `\boldsymbol{K}_T` is the exact Jacobian and is updated at every iteration
  (Crisfield 1991, Sec. 9.3.2, p. 272). Riks (1979, Secs. 3.1 to 3.3)
  augments the equilibrium equations with a linear equation, the plane
  normal to the tangent of the path at the last converged point at the
  distance `\Delta s` from it, and solves them by Newton's method with
  quadratic convergence; with `g = 0` in :eq:`al-riks`, the constraint is
  the updated normal plane of Ramm (1981, p. 75); Crisfield (1991, Sec.
  9.3.2.2, Eq. (9.29), pp. 274-275) attributes the mismatch term to
  Schweizerhof and Wriggers and to Riks. With it, each iteration satisfies
  the linearized constraint about the current increment, and the iterates
  return to the sphere of :eq:`al-constraint` as they converge.
- **Crisfield**, ``'arc_length_crisfield'``: the spherical constraint is
  solved exactly for `\delta\lambda`. With `\boldsymbol{v} = \Delta \boldsymbol{u} +
  \delta \boldsymbol{u}_R`, it is the quadratic equation

  .. math::
      :label: al-crisfield

      a \, \delta\lambda^2 + b \, \delta\lambda + c = 0, \qquad
      a = \langle (\delta \boldsymbol{u}_q, 1), (\delta \boldsymbol{u}_q, 1) \rangle, \quad
      b = 2 \langle (\boldsymbol{v}, \Delta\lambda), (\delta \boldsymbol{u}_q, 1) \rangle, \quad
      c = \langle (\boldsymbol{v}, \Delta\lambda), (\boldsymbol{v}, \Delta\lambda) \rangle - \Delta s^2

  whose roots, computed as `q/a` and `c/q` with `q = -\frac{1}{2}(b +
  \mathrm{sign}(b) \sqrt{b^2 - 4 a c})` to avoid cancellation, give two
  new increments of the length `\Delta s`. The root whose increment makes
  the smallest angle with the current one, i.e. the largest inner product,
  is chosen, which avoids the "doubling back" along the path already
  traced (Crisfield 1981, p. 56). When the roots are complex, the
  linearized constraint :eq:`al-riks` is used for that iteration.

The convergence criterion is that of the Newton-Raphson method, see
:doc:`convergence_criteria`, at the current load factor `\lambda +
\Delta\lambda`, with the external force vector of the increment,
`\Delta\lambda \boldsymbol{F}_{ext}`, as an additional reference, which keeps the
criterion meaningful where the path crosses a null load. The step also
fails when `\delta\lambda` cannot be computed or is not finite, when the
divergence checks of :doc:`newton_raphson_method` stop the iterations, and
when the converged increment points against the tangent predictor of the
step, `\langle (\Delta \boldsymbol{u}, \Delta\lambda), (\Delta \boldsymbol{u}_{pred},
\Delta\lambda_{pred}) \rangle \le 0`: the previous converged point is also
at the distance `\Delta s` from the last one, and with long steps or near a
limit point the iterations may converge to it. The accepted points are
thus on the side of the predictor of the hyperplane normal to it through
the last converged point, as the points of the plane of Riks.

With ``an.modified_NR = True``, the factorization of the tangent at the
last converged point is used for the first ``an.compute_every_n``
iterations and the tangent is then updated every ``an.compute_every_n``
iterations; with ``an.kT_initial_state = False``, `\boldsymbol{K}_{C0}` replaces
the tangent of the initial state at the first step.


Step length and termination
---------------------------

After each converged step, the arc length is multiplied by
`\sqrt{I_d/n_{it}}`, with `n_{it}` the number of corrections of the step
and `I_d` = ``DESIRED_NUM_ITER`` `= 4`, bounded by ``MIN_GROWTH_FACTOR``
`= 0.5` and ``MAX_GROWTH_FACTOR`` `= 1.5`, at most 1 after a step that was
repeated, and limited by the larger of ``an.maxInc`` and the first arc
length. The square root is the rule of Ramm (1981, p. 76), who found that
the ratio `I_d/n_{it}` itself, used by Crisfield (1981, Eq. (15)), makes
the number of iterations oscillate from step to step. A step that fails is
repeated from the same converged point and predictor direction with the
arc length multiplied by ``INC_CUT_FACTOR`` `= 0.5`.

When a converged step crosses the load factor 1.0, from `\lambda < 1` to
`\lambda + \Delta\lambda > 1`, it is replaced by Newton-Raphson iterations
at `\lambda = 1` exactly, from the predictor `t \Delta \boldsymbol{u}` with `t = (1
- \lambda)/\Delta\lambda`; when they do not converge, the step is repeated
with a shorter arc length. The analysis stops when the load factor 1.0 is
reached, when the cumulative arc length reaches ``an.maxArcLength``, 18 by
default, when the arc length falls below ``an.minInc``, after
``MAX_NUM_STEPS`` `= 1000` converged steps, or when the tangent stiffness
matrix at the last converged point is singular. A path whose load factor never reaches 1.0, e.g. the
post-buckling path of a structure whose load is beyond its limit load and
does not recover, ends at ``an.maxArcLength``: the attribute bounds the
length of the path traced.


Choice between the two methods
------------------------------

Both methods trace the same path, with the same predictor, step control
and convergence criterion. Riks' method is Newton's method on the
augmented system, with quadratic convergence and a linear equation for
`\delta\lambda` that always has a solution while its denominator is not
null. Crisfield's method keeps the iterates on the sphere of radius
`\Delta s` and needs the choice between two roots, with the fall-back to
the linearized constraint when they are complex. ``tests/test_solver_arc_length.py``
checks both on analytic problems with exact tangents: the limit points of
the shallow von Mises truss, snap-through and snap-back, the quadratic
convergence, the step cutting, the modified Newton-Raphson, the landing on
the load factor 1.0, the independence of the units of forces and
displacements, and a sharp peak and complex roots;
``tests/test_solver_arc_length_fung_kaplan.py`` traces the snap-through of
the low sinusoidal arch of Fung and Kaplan (1952) through both limit
points with :class:`.BeamC` elements.


Loads that depend on the configuration or on the load factor
------------------------------------------------------------

When ``calc_fext`` declares ``c``, `\boldsymbol{q}` in the predictor and in the
bordered system is the load vector of the current configuration,
``calc_fext(inc=1., c=c)``, i.e. `-\partial \boldsymbol{R}/\partial \lambda` at the
iterate, whereas `\boldsymbol{u}_{ref}` remains the linear solution of the
reference vector; with it, the iterations of Riks' method keep their
quadratic convergence under follower loads, see
``tests/test_solver_follower_loads.py``. When ``calc_fext`` also declares
``lbd``, `\boldsymbol{q}` is evaluated at the current configuration and load
factor, together with the tangent stiffness matrix, i.e. once per update of
the tangent, and `\lambda \boldsymbol{q}` is an additional reference of the
convergence criterion, see :doc:`configuration_dependent_loads`.


References
----------

- Crisfield, M. A., A fast incremental/iterative solution procedure that
  handles "snap-through", Computers & Structures, 13(1-3), 55-62, 1981.
  https://doi.org/10.1016/0045-7949(81)90108-5
- Crisfield, M. A., Non-linear Finite Element Analysis of Solids and
  Structures, Volume 1: Essentials, John Wiley & Sons, Chichester, 1991.
- Fung, Y. C. and Kaplan, A., Buckling of low arches or curved beams of
  small curvature, NACA Technical Note 2840, National Advisory Committee
  for Aeronautics, Washington, 1952.
  https://ntrs.nasa.gov/citations/19930083579
- Ramm, E., Strategies for tracing the nonlinear response near limit
  points, in: Wunderlich, W., Stein, E. and Bathe, K. J. (eds.), Nonlinear
  Finite Element Analysis in Structural Mechanics, Springer, 63-89, 1981.
  https://doi.org/10.1007/978-3-642-81589-8_5
- Riks, E., An incremental approach to the solution of snapping and
  buckling problems, International Journal of Solids and Structures,
  15(7), 529-551, 1979. https://doi.org/10.1016/0020-7683(79)90081-7
