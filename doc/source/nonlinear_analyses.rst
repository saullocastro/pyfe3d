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

The worked example :doc:`tut-prescribed-displacement-dome-riks` traces with
Riks' method the reaction of a clamped spherical cap and of a hinged shallow
shell pressed down by a deflection prescribed at their crown.

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


Choosing the kinematics
-----------------------

The callables of the table above give the von Kármán kinematics of the
elements. For large rotations, :class:`.CorotationalModel` of
:mod:`pyfe3d.corotational` assembles a model in which each element is
added with one of four kinematics, and provides the four callables of
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
      - the von Kármán strains of each element, ``update_fint(...,
        nonlinear=1)``, ``update_KCNL`` and ``update_KG``, in the fixed
        element axes; small strains
      - small to moderate rotations
      - :class:`.Quad4`, :class:`.Quad4R`, :class:`.Tria3R`,
        :class:`.Tria3DSG`, :class:`.BeamC`, :class:`.BeamLR`
    * - ``'corotational'``
      - the element-independent co-rotational (EICR) formulation of Felippa
        and Haugen (2005): the rigid motion of each element is removed, and
        its local linear stiffness matrix acts on the deformational
        displacements; small strains
      - large displacements and rotations, each nodal rotation vector
        below one full revolution, `|\boldsymbol{\psi}_a| < 2 \pi`
      - all, :class:`.Truss` and :class:`.Spring` included
    * - ``'corotational_von_karman'``
      - as ``'corotational'``, with the von Kármán strains of the element
        in its co-rotational frame instead of its linear strains,
        Crisfield (1997), Section 17.1.5
      - as ``'corotational'``
      - as ``'corotational'``; it differs only for :class:`.BeamC`

The von Kármán terms of the shells and of :class:`.BeamLR` vanish in their
co-rotational frames, so ``'corotational_von_karman'`` gives the results of
``'corotational'`` for them, to round-off; :class:`.Truss` and
:class:`.Spring`, which have no von Kármán strains, take it as
``'corotational'``. For :class:`.BeamC` it adds the higher-order terms of
the local element, with its element-averaged axial strain, see
:class:`.BeamC`: on the cantilever under an end force of Sze, Liu and Lo
(2004), Section 3.1, the tip deflections `-U` and `W` of 16 elements differ
from the elastica by -0.02 % and +0.05 % with ``'corotational'``, and by
+0.002 % and +0.007 % with ``'corotational_von_karman'``, see
:class:`.CorotationalBeam`. ``'von_karman'`` follows the elastica only up
to moderate rotations: on the cantilever rolled up by an end moment, Sze,
Liu and Lo (2004), Section 3.2, its tip leaves the exact one by 0.6 % of
the length at a tenth of the moment that closes the circle, see
``test_cantilever_end_moment_full_circle`` of
``tests/test_corotational_benchmarks.py``.

The rotational dofs of a :class:`.CorotationalModel` are total rotation
vectors `\boldsymbol{\psi}_a`, whose nodal rotation matrices are
`\exp(\mathrm{Spin}(\boldsymbol{\psi}_a))`, see :func:`.expso3`, added by
the solvers like the translations, without a state kept between the
calls. The map is singular at `|\boldsymbol{\psi}_a| = 2 \pi`; the
benchmark above reaches it at the tip only because the out-of-plane
rotations are restrained, which is exact for that plane problem. A node
shared by a co-rotational element and an element of another kinematics
is valid only while its rotations remain small, and a warning is issued.

A model is built as with the elements alone: set the connectivity, call
``update_rotation_matrix`` with the coordinates ``x`` of the model, and
pass each element to :meth:`.CorotationalModel.add`; the ``init_k_*``
attributes are not needed. The cantilever of Sze, Liu and Lo (2004),
Section 3.2, with 16 :class:`.BeamC` elements, rolled up into half a
circle by an end moment that follows the tip::

    import numpy as np
    from pyfe3d import BeamC, BeamCProbe, DOF
    from pyfe3d.beamprop import BeamProp
    from pyfe3d.solver import Analysis
    from pyfe3d.corotational import CorotationalModel

    L, b, h, E = 12., 1., 0.1, 1.2e6
    prop = BeamProp()
    prop.A, prop.E, prop.G = b*h, E, E/2
    prop.Iyy, prop.Izz, prop.J = b*h**3/12, h*b**3/12, b*h**3/3
    n = 16
    xs = np.linspace(0, L, n + 1)
    x = np.vstack((xs, 0*xs, 0*xs)).T.ravel()
    model = CorotationalModel(x)
    probe = BeamCProbe()
    for k in range(n):
        elem = BeamC(probe)
        elem.n1, elem.n2 = k, k + 1
        elem.c1, elem.c2 = DOF*k, DOF*(k + 1)
        elem.update_rotation_matrix(0., 1., 0., x)
        model.add(elem, prop, kinematics='corotational')

    M = np.pi*E*prop.Iyy/L   # half of the moment that closes the circle
    model.add_moment(n, [0., -M, 0.], kind='follower')
    bu = np.ones(model.N, dtype=bool)
    bu[:DOF] = False         # clamped at X = 0

    an = Analysis(model.calc_fext, model.calc_fint, model.calc_kC,
                  model.calc_kG, bu=bu)
    an.initialInc = an.maxInc = 0.1
    increments, cs = an.static(NLgeom=True, silent=True)
    tip = cs[-1][DOF*n:DOF*(n + 1)]   # -L, 0, 7.6517, 0, -pi, 0

The tip reaches the end of the polygon of 16 equal chords inscribed in the
circle of radius `L/\pi`, `W = 7.6517`, to round-off, against `2 L/\pi =
7.6394` for the exact circle, and turns by `-\pi` about `Y`.

Applied moments. The forces and moments in the attribute ``fext`` of the
model are constant; a moment in ``fext`` is conjugate to the rotation
vector, and equals a moment about fixed axes only while the rotation
vector keeps a fixed direction, e.g. in plane problems.
:meth:`.CorotationalModel.add_moment` adds a concentrated moment with
``kind='rotation_vector'`` (into ``fext``), ``'fixed_axes'``, a moment
about the fixed global axes, which is non-conservative and gives an
unsymmetric tangent stiffness matrix, or ``'follower'``, a moment that
rotates with the nodal triad. The last two depend on the configuration
and enter ``calc_fint``, ``calc_kG`` and ``calc_fext(c=...)`` as described
in the section above; pass ``model.calc_fext`` to :class:`.Analysis` after
adding them, since the callable it returns depends on them.

Shells and warping. :class:`.CorotationalShell` links a warped
quadrilateral to its nodes with rigid offsets, ``warping='offsets'`` by
default, and warns above the warp ratio of 0.07, up to which the offsets
were verified; with ``warping=None`` it warns above 0.01, see
:doc:`element_choice`. The frame of the triangles is along their side 1-2,
so that their solution depends on the numbering of the nodes, see the
Notes of :class:`.CorotationalShell`. The keyword arguments of
:meth:`.CorotationalModel.add` reach the wrappers, e.g. ``warping`` and the
``frame`` of :class:`.CorotationalTruss` and :class:`.CorotationalSpring`,
and the ``update_KC0`` of the element, e.g. the ``hgfactor_*`` of
:class:`.Quad4R`.

Convergence tolerance. The default criterion, ``relTOL = 1e-9`` with the
round-off floor of the section above, applies without ``absTOL``. The
co-rotational elements compute the current nodal positions relative to the
base centroid of each element, see the Notes of
:class:`.CorotationalShell`, so that the round-off of the internal force
does not depend on the origin of the coordinates, which changes neither
`\boldsymbol{K}_{C0}` nor the force vectors of the criterion. Computed as
`\boldsymbol{X} + \boldsymbol{u}`, with `\boldsymbol{X}` the nodal
coordinates, the positions had a round-off of the order of `\varepsilon
|\boldsymbol{X}|`, above the round-off floor `3 \varepsilon \|
|\boldsymbol{K}_{C0}| |\boldsymbol{u}| \|_{\boldsymbol{d}}` on fine meshes:
up to 2026-10-30, the iterations of the cantilever with 64 :class:`.BeamC`
elements stagnated at `\zeta \approx 10^{-8}` and needed ``an.absTOL``, and
with its root at 1000 from the origin at `\zeta \approx 10^{-6}`, see
``test_cantilever_end_shear_translated`` in
``tests/test_corotational_benchmarks.py``.

Post-processing. :meth:`.CorotationalModel.nodal_rotations` gives the
rotation matrices of the nodes and
:meth:`.CorotationalModel.deformational_displacements` the deformational
displacements of each co-rotational element in its frame;
:meth:`.CorotationalShell.update_probe` writes them into the probe of the
element, whose own routines then recover the stresses. ``model.KC0`` is
the linear stiffness matrix of the elements; for a warped quadrilateral
added with ``'corotational'`` or ``'corotational_von_karman'`` it is that
of the flat element, since the wrapper switches off the offsets of the
element, see the Notes of :class:`.CorotationalShell`.

Benchmarks. ``tests/test_corotational_benchmarks.py`` reproduces cases 3.1
(cantilever under an end force), 3.2 (cantilever rolled up into a full
circle by an end moment), 3.3 (slit annular plate), 3.4 (hemispherical
shell) and 3.8 (hinged cylindrical roof, with the arc-length method) of
Sze, Liu and Lo (2004) with :class:`.BeamC`, :class:`.Quad4R` and
:class:`.Tria3R`, and ``tests/test_corotational_core.py`` checks the
finite-rotation utilities, the wrappers and their tangent stiffness
matrices by finite differences. The worked example
:doc:`tut-corotational-large-rotations` shows such analyses step by step.


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
- Crisfield, M. A., Non-linear Finite Element Analysis of Solids and
  Structures, Volume 2: Advanced Topics, John Wiley & Sons, Chichester,
  1997.
- Felippa, C. A. and Haugen, B., A unified formulation of small-strain
  corotational finite elements: I. Theory, Computer Methods in Applied
  Mechanics and Engineering, 194(21-24), 2285-2335, 2005.
  https://doi.org/10.1016/j.cma.2004.07.035
- Hibbitt, H. D., Some follower forces and load stiffness, International
  Journal for Numerical Methods in Engineering, 14(6), 937-941, 1979.
  https://doi.org/10.1002/nme.1620140613
- Schweizerhof, K. and Ramm, E., Displacement dependent pressure loads in
  nonlinear finite element analyses, Computers & Structures, 18(6),
  1099-1114, 1984. https://doi.org/10.1016/0045-7949(84)90154-8
- Sze, K. Y., Liu, X. H. and Lo, S. H., Popular benchmark problems for
  geometric nonlinear analysis of shells, Finite Elements in Analysis and
  Design, 40(11), 1551-1569, 2004.
  https://doi.org/10.1016/j.finel.2003.11.001
