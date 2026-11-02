Non-linear static analyses
==========================

When the displacements change the geometry enough to change the
equilibrium, e.g. the membrane stiffening of a plate under pressure, the
softening of a compressed shell after buckling, or the snap-through of an
arch, the equilibrium equations are non-linear in the displacements and
are solved incrementally and iteratively. This chapter states the
equations that pyfe3d solves, how a model is passed to the solvers, and how
to choose the solution method and the kinematics. The algorithms are in
:doc:`newton_raphson_method` and :doc:`arc_length_methods`, the stopping
test of the iterations in :doc:`convergence_criteria`, and the loads that
depend on the configuration or on the load factor in
:doc:`configuration_dependent_loads`.


Equilibrium, residual and tangent stiffness
-------------------------------------------

The external loads are applied proportionally to the load factor
`\lambda`, with `\boldsymbol{F}_{ext}` the reference external force vector and
`\lambda = 1` the full load. An equilibrium configuration `\boldsymbol{u}` makes
the residual null, the principle of virtual work in discrete form:

.. math::
    :label: nl-residual

    \boldsymbol{R}(\boldsymbol{u}, \lambda) = \boldsymbol{F}_{int}(\boldsymbol{u}) - \lambda \boldsymbol{F}_{ext} = \boldsymbol{0},
    \qquad
    \boldsymbol{F}_{int} = \int_{\Omega} (\boldsymbol{B}_L + \boldsymbol{B}_{NL})^\top \boldsymbol{\sigma} \, d\Omega

with the strains `\boldsymbol{\varepsilon} = (\boldsymbol{B}_L + \frac{1}{2}
\boldsymbol{B}_{NL}) \boldsymbol{u}` and their variations `\delta
\boldsymbol{\varepsilon} = (\boldsymbol{B}_L + \boldsymbol{B}_{NL}) \delta \boldsymbol{u}`, where
`\boldsymbol{B}_{NL}` depends on `\boldsymbol{u}`. The sign convention is that of
the residual `\boldsymbol{R} = \boldsymbol{F}_{int} - \lambda \boldsymbol{F}_{ext}`, whose
Jacobian is the tangent stiffness matrix

.. math::
    :label: nl-tangent

    \boldsymbol{K}_T = \frac{\partial \boldsymbol{R}}{\partial \boldsymbol{u}}
    = \boldsymbol{K}_{C0} + \boldsymbol{K}_{CNL}(\boldsymbol{u}) + \boldsymbol{K}_{GNL}(\boldsymbol{u})
    + \boldsymbol{K}_G(\boldsymbol{u}) + \boldsymbol{K}_F

with the linear constitutive stiffness matrix `\boldsymbol{K}_{C0}` of
`\boldsymbol{B}_L`, the non-linear constitutive stiffness matrix
`\boldsymbol{K}_{CNL}` of the terms of `\boldsymbol{B}_{NL}`, the geometric stiffness
matrices of the stresses of the linear strains, `\boldsymbol{K}_G`, and of the
non-linear strains, `\boldsymbol{K}_{GNL}`, and the load stiffness matrix
`\boldsymbol{K}_F = -\partial (\lambda \boldsymbol{F}_{ext})/\partial \boldsymbol{u}`, null for
dead loads, see :doc:`configuration_dependent_loads`. Without `\boldsymbol{K}_F`,
`\boldsymbol{K}_T` is symmetric: the constitutive terms have the same operator
`\boldsymbol{B}_L + \boldsymbol{B}_{NL}` on both sides of the symmetric
`\boldsymbol{C}`, and the geometric terms contain the mixed second
derivatives of the strains. The elements group the terms as their ``update_*`` methods compute them:

- ``KC0``, of ``update_KC0``, is `\boldsymbol{K}_{C0}`;
- ``KCNL``, of ``update_KCNL``, is `\boldsymbol{K}_{CNL} + \boldsymbol{K}_{GNL}`;
- ``KG``, of ``update_KG``, is `\boldsymbol{K}_G`, which is linear in
  `\boldsymbol{u}` and therefore scales with the load multiplier of the linear
  buckling analysis, see :doc:`eigenvalue_analyses`;
- ``fint``, of ``update_fint(..., nonlinear=1)``, is `\boldsymbol{F}_{int}`.

Their sum ``KC0 + KCNL + KG`` is the exact derivative of ``fint`` for every
element with ``update_KCNL``, which ``tests/test_tangent_consistency.py``
checks with a Taylor test and finite differences on distorted elements in
arbitrary orientations. The internal force vector is not `\boldsymbol{K}_C
\boldsymbol{u}`: because of the factor `\frac{1}{2}` of the strains, it must be
computed from the stresses, as ``update_fint`` does. The strains are the
von Kármán strains of each element, valid for small strains and small to
moderate rotations; for large rotations, see the kinematics below.


The model as four callables
---------------------------

:class:`.Analysis` receives the model as four callables and stores the
solution in ``an.increments``, the converged load factors, and ``an.cs``,
the converged configurations:

.. list-table::
    :header-rows: 1

    * - Callable
      - Returns
      - With the elements
    * - ``calc_fext(inc=1., silent=True)``
      - `\boldsymbol{F}_{ext}`, scaled by ``inc``
      - the nodal loads of the model
    * - ``calc_fint(c, silent=True)``
      - `\boldsymbol{F}_{int}(\boldsymbol{c})`
      - ``update_fint(fint, prop, nonlinear=1)`` of every element, after
        ``update_probe_xe(x)`` and ``update_probe_ue(c)``
    * - ``calc_kC(c=None, NLgeom=False, silent=True)``
      - `\boldsymbol{K}_{C0}`, or `\boldsymbol{K}_{C0} + \boldsymbol{K}_{CNL}(\boldsymbol{c}) +
        \boldsymbol{K}_{GNL}(\boldsymbol{c})` with ``NLgeom=True``
      - ``update_KC0`` once, ``update_KCNL`` at ``c``
    * - ``calc_kG(c=None, NLgeom=False, silent=True)``
      - `\boldsymbol{K}_G(\boldsymbol{c})`
      - ``update_KG`` at ``c``

The solvers call ``calc_fext(inc=1.)`` and scale the vector by the load
factor themselves; ``calc_kC()``, the linear stiffness matrix, gives the
predictor of the first increment, the scaling of the arc length and the
scaling of the convergence criterion; and the tangent stiffness matrix is
``calc_kC(c, NLgeom=True) + calc_kG(c, NLgeom=True)``, see
:func:`pyfe3d.solver.callbacks.calc_kT`. A callable may declare the
optional keyword arguments of :mod:`pyfe3d.solver.callbacks`, for loads
that depend on the configuration or on the load factor, see
:doc:`configuration_dependent_loads`. The ``Model`` class of
``tests/test_solver_nonlinear_static.py`` wraps a list of elements of one
type into these callables.

With ``Analysis(..., bu=bu, uk=uk)``, the callables work with the vectors
and matrices of all the ``N`` dofs and :class:`.FreeDofs` restricts them to
the free dofs, with the prescribed displacements `\lambda
\bar{\boldsymbol{u}}_k` imposed exactly, see :doc:`boundary_conditions`; the
vectors of ``an.cs`` then have size ``N``. Without ``bu``, the callables
work with the free dofs only.


Incremental-iterative solution
------------------------------

:meth:`.Analysis.static` with ``NLgeom=True`` traces the equilibrium path
`\boldsymbol{R}(\boldsymbol{u}, \lambda) = \boldsymbol{0}` from the undeformed state,
`\lambda = 0`, to `\lambda = 1`, in increments, each corrected by
iterations of Newton's method until the residual is small. ``an.NL_method``
selects how the load factor of each increment is set:

- ``'NR'``, the default, load control: `\lambda` is prescribed at each
  increment and the iterations solve for `\boldsymbol{u}`, see
  :func:`pyfe3d.solver.newton_raphson._solver_NR` and
  :doc:`newton_raphson_method`;
- ``'arc_length_riks'`` and ``'arc_length_crisfield'``, path following:
  `\lambda` is an unknown of each increment, together with `\boldsymbol{u}`, and
  an arc-length constraint fixes the size of the increment, see
  :func:`pyfe3d.solver.arc_length._solver_arc_length` and
  :doc:`arc_length_methods`.

Both stop at the load factor 1.0 exactly, or earlier, keeping the
converged increments, when the increment falls below ``an.minInc`` or, for
the arc-length methods, when the cumulative arc length reaches
``an.maxArcLength``; an analysis that did not finish is recognised by
``an.increments[-1] != 1.``. The linear analysis, ``an.static(NLgeom=False)``,
returns a single increment, see :doc:`linear_analyses`.

At a limit point of the load, `\boldsymbol{K}_T` is singular and the load factor
reaches a maximum along the path. Under load control, no equilibrium
exists just above the limit load near the current configuration, and the
iterations either fail, the increment being cut until ``an.minInc``, or
converge to a distant configuration on another part of the path, which
hides the snap-through. The arc-length methods
pass limit points of the load, snap-through, and also points where the
displacements turn back, snap-back, since they do not prescribe either of
them.


Choice of the method
--------------------

.. list-table::
    :header-rows: 1

    * - Problem
      - Method
      - Remarks
    * - path without limit points, e.g. membrane stiffening, moderate
        softening
      - ``'NR'``, full Newton-Raphson
      - quadratic convergence with the consistent tangent of the elements;
        the default
    * - same, expensive factorizations
      - ``'NR'`` with ``an.modified_NR = True``
      - linear convergence, see :doc:`newton_raphson_method`
    * - corrections that overshoot, e.g. strongly non-linear springs or
        large increments
      - ``'NR'`` with ``an.line_search = True``
      - the full step is kept when it reduces the residual enough
    * - limit points, snap-through, snap-back, post-buckling paths
      - ``'arc_length_riks'`` or ``'arc_length_crisfield'``
      - Riks: Newton's method on the augmented system, quadratic
        convergence; Crisfield: spherical constraint solved exactly, see
        :doc:`arc_length_methods`
    * - an analysis controlled by a prescribed displacement
      - ``bu`` and ``uk`` without load, with any method
      - load control traces the reaction as long as the prescribed
        displacement grows along the path; a snap-back of that
        displacement needs an arc-length method

The tolerance of the iterations, ``an.relTOL = 1e-9`` by default, is that
of Crisfield's criterion with a consistent tangent stiffness matrix, see
:doc:`convergence_criteria`. A tight tolerance matters when the converged
state feeds an eigenvalue analysis, since the geometric stiffness matrix of
:doc:`eigenvalue_analyses` is built from the stresses of that state.


Choosing the kinematics
-----------------------

The callables of the table above give the von Kármán kinematics of the
elements. For large rotations, :class:`.CorotationalModel` of
:mod:`pyfe3d.corotational` assembles a model in which each element is added
with one of four kinematics, and provides the four callables of
:class:`.Analysis`, which is used unchanged:

.. list-table::
    :header-rows: 1

    * - ``kinematics``
      - Assumes
      - Limits
      - Elements
    * - ``'linear'``
      - small displacements, `\boldsymbol{K}_{C0} \boldsymbol{u}`
      - small rotations
      - all
    * - ``'von_karman'``
      - the von Kármán strains of each element in the fixed element axes,
        ``update_fint(..., nonlinear=1)``, ``update_KCNL`` and
        ``update_KG``; small strains
      - small to moderate rotations
      - :class:`.Quad4`, :class:`.Quad4R`, :class:`.Tria3R`,
        :class:`.Tria3DSG`, :class:`.BeamC`, :class:`.BeamLR`
    * - ``'corotational'``
      - the element-independent co-rotational formulation of Felippa and
        Haugen (2005): the rigid motion of each element is removed, and its
        linear local stiffness acts on the deformational displacements;
        small strains
      - large displacements and rotations, each nodal rotation vector
        below one full revolution, `|\boldsymbol{\psi}_a| < 2 \pi`
      - all, :class:`.Truss` and :class:`.Spring` included
    * - ``'corotational_von_karman'``
      - as ``'corotational'``, with the von Kármán strains of the element
        in its co-rotational frame, Crisfield (1997), Sec. 17.1.5
      - as ``'corotational'``
      - as ``'corotational'``; it differs only for :class:`.BeamC`

The rotational dofs of a :class:`.CorotationalModel` are total rotation
vectors `\boldsymbol{\psi}_a`, whose nodal rotation matrices are
`\exp(\mathrm{Spin}(\boldsymbol{\psi}_a))`, see :func:`.expso3`; the solvers
add their corrections like those of the translations, without a state
kept between the calls of the callables. Concentrated moments about fixed
axes or following the nodal triads, :meth:`.CorotationalModel.add_moment`,
are configuration-dependent loads and give an unsymmetric tangent
stiffness matrix, which the solvers support, see
:doc:`configuration_dependent_loads`. The theory of the co-rotational layer,
its frames, its treatment of warped quadrilaterals and its verification
are given with :mod:`pyfe3d.corotational`.


Log messages
------------

The solvers print the increments and, at each iteration, the largest
absolute residual and the `\zeta` of Crisfield's criterion, with
:mod:`pyfe3d.solver.logger`, unless ``silent=True`` is passed to
:meth:`.Analysis.static`; the callables must accept ``silent``. The number
of iterations per increment is the first diagnostic of a model: with a
consistent tangent, two to four iterations per moderate increment, see
:doc:`convergence_criteria`.


References
----------

- Crisfield, M. A., Non-linear Finite Element Analysis of Solids and
  Structures, Volume 1: Essentials, John Wiley & Sons, Chichester, 1991.
- Crisfield, M. A., Non-linear Finite Element Analysis of Solids and
  Structures, Volume 2: Advanced Topics, John Wiley & Sons, Chichester,
  1997.
- Felippa, C. A. and Haugen, B., A unified formulation of small-strain
  corotational finite elements: I. Theory, Computer Methods in Applied
  Mechanics and Engineering, 194(21-24), 2285-2335, 2005.
  https://doi.org/10.1016/j.cma.2004.07.035
