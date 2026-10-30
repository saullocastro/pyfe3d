Non-linear static analyses
==========================

The geometrically non-linear static analyses are run with
:class:`.Analysis`, which receives four callables of the model and finds
the configurations `\boldsymbol{u}` in equilibrium with the external load scaled by
the load factor `\lambda`:

.. math::

    \boldsymbol{R}(\boldsymbol{u}, \lambda) = \boldsymbol{F}_{int}(\boldsymbol{u}) - \lambda \boldsymbol{F}_{ext} = \boldsymbol{0}

The elements give the terms of the residual and of its Jacobian, the
solvers of :mod:`pyfe3d.solver` the increments and the iterations.


From the elements to the callables
----------------------------------

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
      - `\boldsymbol{K}_{C0}`, or `\boldsymbol{K}_{C0} + \boldsymbol{K}_{CNL}(\boldsymbol{c}) + \boldsymbol{K}_{GNL}(\boldsymbol{c})` with ``NLgeom=True``
      - ``update_KC0`` once, ``update_KCNL`` at ``c``
    * - ``calc_kG(c=None, NLgeom=False, silent=True)``
      - `\boldsymbol{K}_G(\boldsymbol{c})`
      - ``update_KG`` at ``c``

``calc_kC() + calc_kG()`` is then the exact Jacobian of ``calc_fint``,
`\boldsymbol{K}_T = \partial \boldsymbol{R}/\partial \boldsymbol{u}`, and the full Newton-Raphson method
converges quadratically. The class ``Model`` of
``tests/test_solver_nonlinear_static.py``, see :doc:`ex_nonlinear_static`,
wraps a list of elements of one type into these four callables.
``calc_kC()``, the linear stiffness matrix, also gives the scaling of the
convergence criterion.


Boundary conditions and prescribed displacements
------------------------------------------------

With ``Analysis(..., bu=bu)``, the callables receive and return the
vectors and matrices of all the ``N`` dofs, as assembled by the elements,
and :class:`.FreeDofs` restricts them to the free dofs ``bu``. The solution
vectors ``an.cs`` have size ``N``, with zeros at the constrained dofs.

With ``uk``, the constrained dofs have the prescribed displacements
`\lambda \boldsymbol{u}_k`, imposed exactly, proportional to the load factor:
``uk[~bu]`` are the values at `\lambda = 1`, and a load-free analysis,
``calc_fext`` returning zeros, is displacement controlled. The reaction at
a prescribed dof is ``calc_fint(u)`` at that dof minus the load applied
there.

In ``test_prescribed_displacement_and_reaction`` of
:doc:`ex_nonlinear_static`, a point-loaded plate is first solved under load
control; its central deflection ``w_c``, at the dof ``dof``, is then
prescribed without load, with the callables of the full model::

    bu_d = bu.copy()
    bu_d[dof] = False        # the loaded dof becomes a known dof
    uk = np.zeros(N)
    uk[dof] = w_c            # value at the load factor 1

    def calc_fext(inc=1., silent=True):
        return np.zeros(N)   # no load: displacement control

    an = Analysis(calc_fext, calc_fint, calc_kC, calc_kG, bu=bu_d, uk=uk)
    an.NL_method = 'arc_length_riks'   # or 'NR'
    increments, cs = an.static(NLgeom=True)
    reaction = calc_fint(cs[-1])[dof]  # minus the load at dof, here none

Each ``cs[i][dof]`` is ``increments[i]*w_c``, the displacement field at
`\lambda = 1` is that of the load-controlled analysis, and the reaction
equals the load. The same ``bu`` and ``uk`` serve the linear analysis,
``an.static(NLgeom=False)``, see :doc:`linear_analyses` and the worked
example :doc:`tut-prescribed-displacement-cantilever`.

Without ``bu``, the callables return the vectors and matrices of the free
dofs only. The solvers do not remove null rows and columns: a null row or
column of the linear stiffness matrix, a dof without stiffness, raises a
:class:`numpy.linalg.LinAlgError` naming the dofs, see
:func:`.check_null_rows_cols`.


Solvers
-------

``an.NL_method`` selects the solver, and :meth:`.Analysis.static` with
``NLgeom=True`` runs it, returning the converged load factors
``an.increments`` and solutions ``an.cs``:

- ``'NR'``, the default: load control with the Newton-Raphson method,
  :func:`.newton_raphson._solver_NR`, full or modified
  (``an.modified_NR``), see Crisfield (1991), Sections 1.2.2 and 1.2.3,
  with an optional line search (``an.line_search``). The first increment
  is ``an.initialInc``; a step
  that fails is repeated with half the increment, and the increment grows
  after a step that converged without being cut, up to ``an.maxInc``.
  Beyond a limit point of the load it may converge to another branch of
  the equilibrium path.
- ``'arc_length_riks'`` and ``'arc_length_crisfield'``: the arc-length
  methods following Riks (1979) and Crisfield (1981),
  :func:`.arc_length._solver_arc_length`, in which the load factor is an
  unknown of each step, such that limit points, snap-through and
  snap-back are traced. The first arc length corresponds to the load
  factor increment ``an.initialInc`` along the initial tangent.

Both stop at the load factor 1.0, or earlier when the increment falls
below ``an.minInc`` or, for the arc-length methods, when the cumulative
arc length reaches ``an.maxArcLength``. A run that did not finish is
detected with::

    increments, cs = an.static(NLgeom=True)
    if not increments or increments[-1] != 1.:
        ...  # stopped at increments[-1], see the log messages

The linear static analysis, ``an.static(NLgeom=False)``, returns a single
increment, see :doc:`linear_analyses`.


Convergence criterion
---------------------

The iterations of each increment stop when the residual is small according
to ``an.convergence``, see :mod:`.convergence`:

- ``'crisfield'``, the default: the residual scaled with the diagonal of
  the linear stiffness matrix of the free dofs, `d_j = K_{{C0}_{jj}}`, the
  entries of the diagonal scaling vector `\boldsymbol{d}`,

  .. math::

      \zeta = \frac{\|\boldsymbol{R}\|_{\boldsymbol{d}}}{\max(
      \|\lambda \boldsymbol{F}_{ext}\|_{\boldsymbol{d}}, \|\boldsymbol{F}_{int}\|_{\boldsymbol{d}},
      \ldots)} \le \text{relTOL}, \qquad
      \|v\|_{\boldsymbol{d}} = \sqrt{\sum_j v_j^2/|d_j|}

  with ``an.relTOL = 1e-9`` by default, see :func:`.crisfield_zeta` for
  the references in the denominator. It does not depend on the units of
  each dof, e.g. on whether the rotations are in radians or in
  milliradians, and it ignores the support reactions;
- ``'relative'``: the Euclidean norm of the residual over the largest
  Euclidean norm of the references, which adds the squares of forces and
  of moments.

``an.absTOL`` also accepts the iterations whose largest absolute residual
is smaller, in the units of force of the model. With both criteria, an
iteration has also converged at the round-off floor of the residual,

.. math::

    \|\boldsymbol{R}\|_{\boldsymbol{d}} \le 3 \varepsilon \, \| |\boldsymbol{K}_{C0}| \, |\boldsymbol{u}| \|_{\boldsymbol{d}}

with `\varepsilon` the machine precision, `|\boldsymbol{K}_{C0}|` and `|\boldsymbol{u}|` the
elementwise absolute values of the linear stiffness matrix and of the
displacements of the free dofs, and the Euclidean norm with
``'relative'``, see :func:`.roundoff_floor`: the residual cannot be
computed more accurately, and, under large self-equilibrated axial forces,
e.g. of an arch near a null load, `\zeta` would otherwise stagnate above
``relTOL`` and the analysis stop, see ``test_fung_kaplan_beam_arch`` in
:doc:`ex_nonlinear_static`. The log messages mark the iterations converged
at the round-off floor. With a consistent tangent
stiffness matrix and full Newton-Raphson, `\zeta` falls quadratically, as
verified in :doc:`ex_nonlinear_static`; a slow decrease of `\zeta` reveals
an inconsistent tangent stiffness matrix, e.g. a missing ``calc_kG``, which
changes the iterations but not the converged solution.


Configuration-dependent loads
-----------------------------

A load that depends on the configuration, e.g. a follower pressure or the
aerodynamic load of the piston theory, is given through optional keyword
arguments, passed only to the callables that declare them, see
:mod:`.callbacks`:

- ``calc_fint(c, inc=...)`` receives the load factor and includes the
  configuration-dependent part of the load, `-\lambda \boldsymbol{F}(\boldsymbol{c})`, and
  ``calc_kC(c, NLgeom=True, inc=...)`` its derivative, the load stiffness
  matrix `\boldsymbol{K}_F = \lambda \boldsymbol{K}_{F_{ref}}`, with `\boldsymbol{K}_{F_{ref}} = -\partial \boldsymbol{F}/\partial \boldsymbol{c}`, which is
  unsymmetric in general, see below; the solvers use LU factorizations;
- ``calc_fext(inc=1., c=...)`` returns the load vector of the
  configuration ``c`` for a unit load factor, `-\partial \boldsymbol{R}/\partial
  \lambda`, used by the arc-length methods.

The aerodynamic load of the piston theory, `\lambda \beta
\boldsymbol{K}_{A_\beta} \boldsymbol{u}`, with ``update_KA_beta`` of :class:`.Quad4R` or
:class:`.Quad4`, is such a load, see ``test_piston_theory_load_unsymmetric_tangent``
in :doc:`ex_nonlinear_static`. The keyword argument ``lbd`` of
``calc_fext`` gives a load vector that depends on the load factor, which is
how :class:`.FreeDofs` imposes the prescribed displacements.

`\boldsymbol{K}_F` is symmetric, and the load conservative, only in particular cases.
For a pressure normal to the deformed surface, Hibbitt (1979, p. 938)
splits the load stiffness of a uniform pressure into a symmetric surface
integral and an antisymmetric integral along the boundary of the loaded
surface, which vanishes where the displacements are prescribed: a uniform
pressure on a fully enclosed volume, or on a surface whose boundaries have
prescribed displacements, gives a symmetric `\boldsymbol{K}_F`, whereas free edges,
where the pressure changes, and a non-uniform pressure give unsymmetric
terms (p. 939). In Table 2 of Schweizerhof and Ramm (1984), the domain
terms are symmetric for a uniform pressure, and for a non-uniform one only
when the pressure field is fixed in space, and the boundary terms vanish
under one of the conditions of their Sec. 3.2.1, Eqs. (32)-(35): the
pressure is zero on the whole boundary; at least two displacement
components are prescribed at the edge; the edge is restrained normal to a
continuous surface, e.g. the undeformed surface, `w = 0` for a plate on
the `XY` plane; or the boundary terms add up to zero. pyfe3d does not
symmetrise `\boldsymbol{K}_F`: the tangent stiffness matrix, with the matrix returned
by ``calc_kC``, is factorized with :func:`scipy.sparse.linalg.splu`, as in
:func:`.solve`, such that a consistent unsymmetric `\boldsymbol{K}_F` keeps the
quadratic convergence of the Newton-Raphson method, as verified by
``test_newton_raphson_quadratic_with_follower_load`` of
``tests/test_solver_follower_loads.py``.


Log messages
------------

The solvers print their progress, the increments and the `\zeta` of each
iteration, with :mod:`pyfe3d.solver.logger`, unless ``silent=True`` is
passed to :meth:`.Analysis.static`. The callables must accept ``silent``.


References
----------

- Riks, E., An incremental approach to the solution of snapping and
  buckling problems, International Journal of Solids and Structures,
  15(7), 529-551, 1979. https://doi.org/10.1016/0020-7683(79)90081-7
- Crisfield, M. A., A fast incremental/iterative solution procedure that
  handles "snap-through", Computers & Structures, 13(1-3), 55-62, 1981.
  https://doi.org/10.1016/0045-7949(81)90108-5
- Crisfield, M. A., Non-linear Finite Element Analysis of Solids and
  Structures, Volume 1: Essentials, John Wiley & Sons, Chichester, 1991.
- Hibbitt, H. D., Some follower forces and load stiffness, International
  Journal for Numerical Methods in Engineering, 14(6), 937-941, 1979.
  https://doi.org/10.1002/nme.1620140613
- Schweizerhof, K. and Ramm, E., Displacement dependent pressure loads in
  nonlinear finite element analyses, Computers & Structures, 18(6),
  1099-1114, 1984. https://doi.org/10.1016/0045-7949(84)90154-8
