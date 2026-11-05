Linear static analyses
======================

A linear static analysis assumes small displacements and rotations, a
linear elastic material and loads that do not change with the
deformation. The equilibrium is then a single linear system, solved in one
step, which is also the first iteration of every non-linear analysis. This
page derives the system, describes how pyfe3d solves it and checks its
matrix, and treats the case of a load that depends on the configuration of
a structure that otherwise remains linear.


The linear static system
------------------------

The total potential energy of an elastic body under dead loads, `V = U -
W_{ext}`, is stationary at equilibrium, `\delta V = \delta U - \delta
W_{ext} = 0`. With the linear strains of the elements, `\boldsymbol{\varepsilon}
= \boldsymbol{B}_L \boldsymbol{u}`, the stresses `\boldsymbol{\sigma} = \boldsymbol{C}
\boldsymbol{\varepsilon}` and the displacement field interpolated from the nodal
dofs `\boldsymbol{u}`, the first variation must vanish for every
`\delta \boldsymbol{u}`, which gives

.. math::
    :label: ls-system

    \boldsymbol{R} = \boldsymbol{F}_{int} - \boldsymbol{F}_{ext} = \boldsymbol{K}_{C0} \boldsymbol{u} - \boldsymbol{F}_{ext} = \boldsymbol{0},
    \qquad
    \boldsymbol{K}_{C0} = \int_{\Omega} \boldsymbol{B}_L^\top \boldsymbol{C} \boldsymbol{B}_L \, d\Omega

with `\boldsymbol{K}_{C0}` the linear constitutive stiffness matrix, ``KC0``,
assembled with the ``update_KC0`` methods of the elements, see
:doc:`overview`, and `\boldsymbol{F}_{ext}` the external force vector, assembled
by the user. The residual `\boldsymbol{R}` keeps the sign convention of the
non-linear analyses, `\boldsymbol{R} = \boldsymbol{F}_{int} - \boldsymbol{F}_{ext}`, whose
Jacobian is the tangent stiffness matrix, here `\boldsymbol{K}_{C0}`, see
:doc:`nonlinear_analyses`. The linear internal force vector of the
elements, ``update_fint(..., nonlinear=0)``, is `\boldsymbol{K}_{C0}
\boldsymbol{u}`.

`\boldsymbol{K}_{C0}` of an unsupported structure is symmetric and positive
semi-definite, with the rigid-body modes in its null space. The supports
are imposed by the partition of :doc:`boundary_conditions`, which gives the
system of the free dofs, Eq. :eq:`bc-free`,

.. math::
    :label: ls-free

    \boldsymbol{K}_{{C0}_{uu}} \boldsymbol{u}_u = \boldsymbol{F}_{{ext}_u} - \boldsymbol{K}_{{C0}_{uk}} \bar{\boldsymbol{u}}_k

whose matrix is positive definite when the supports suppress every
rigid-body mode (Bathe 2014, Sec. 8.2.5, pp. 726-727), and the reactions,
Eq. :eq:`bc-reactions`. A structure that has no supports is solved with
inertia relief, see :doc:`inertia_relief`.

Constant terms that are not proportional to the displacements of the
analysis belong to the matrix: e.g. the geometric stiffness matrix of a
known preload, ``KC0uu + KG0uu``, see :doc:`eigenvalue_analyses`, or the
aerodynamic stiffness matrix of the piston theory of :class:`.Quad4` and
:class:`.Quad4R`; a follower load affine in the displacements,
`\boldsymbol{F}(\boldsymbol{u}) = \boldsymbol{F}_0 - \boldsymbol{K}_F \boldsymbol{u}`, gives the unsymmetric
system `(\boldsymbol{K}_{C0} + \boldsymbol{K}_F) \boldsymbol{u} = \boldsymbol{F}_0`, see
:func:`.linear_static` and :doc:`configuration_dependent_loads`.


Solvers
-------

Three functions solve the system :eq:`ls-free`:

- :func:`.solve` solves a sparse system with the LU factorization of
  :func:`scipy.sparse.linalg.splu`, symmetric or not, with a real or
  complex right-hand side, and raises a :class:`numpy.linalg.LinAlgError`
  for a null row or column, see :func:`.check_null_rows_cols`, or for an
  exactly singular matrix, where :func:`scipy.sparse.linalg.spsolve` would
  return NaN values;
- :func:`.linear_static` solves ``K @ u = fext`` of the free dofs, after
  checking that a symmetric ``K`` is positive definite, and returns
  ``[1.]`` and ``[u]``, the outputs of :meth:`.Analysis.static`;
- :meth:`.Analysis.static` with ``NLgeom=False`` solves the same system
  from the callables of a model, ``calc_kC()`` and ``calc_fext()``, see
  :doc:`nonlinear_analyses`; with ``Analysis(..., bu=bu, uk=uk)`` the
  callables work with all the ``N`` dofs, the right-hand side of
  :eq:`ls-free` is formed by :meth:`.FreeDofs.linear_fext`, and ``cs[0]``
  has size ``N``, with ``uk[bk]`` at the known dofs.

The check of positive definiteness costs nothing: a symmetric matrix is
factorized once with a symmetric permutation and without row interchanges,
`\boldsymbol{\Pi} \boldsymbol{K} \boldsymbol{\Pi}^\top = \boldsymbol{L} \boldsymbol{D} \boldsymbol{L}^\top`,
whose number of negative pivots in `\boldsymbol{D}` is the number of negative
eigenvalues of `\boldsymbol{K}` by Sylvester's law of inertia, see
:func:`.is_positive_definite`. When all the pivots are positive, the same
factorization, then that of Cholesky up to the scaling of the rows, gives
the solution, see :func:`pyfe3d.solver.static._solve_linear_static`. A
negative or a zero pivot raises a :class:`numpy.linalg.LinAlgError` naming
the likely causes: insufficient boundary conditions, a mechanism, distorted
elements, e.g. re-entrant quadrilaterals, or a preload beyond the buckling
load included in the matrix. An unsymmetric matrix is solved with
:func:`.solve` without this check.

The factorization is the cost of the analysis; the reactions,
Eq. :eq:`bc-reactions`, and the element forces of the post-processing,
``update_probe_ue`` and ``update_fint`` of the elements, are products by
the assembled matrices or loops over the elements.


Loads that depend on the configuration
--------------------------------------

A load whose direction or magnitude follows the deformation, e.g. a
pressure normal to the deformed surface, makes the equilibrium of a
structure that is otherwise linear, `\boldsymbol{K}_{C0} \boldsymbol{u} =
\boldsymbol{F}_{ext}(\boldsymbol{u})`, a non-linear system. When ``calc_fext``
declares the keyword argument ``c``, see
:doc:`configuration_dependent_loads`, :meth:`.Analysis.static` with
``NLgeom=False`` solves it with the load correction iterations

.. math::
    :label: ls-load-correction

    \boldsymbol{u}_{j+1} = \boldsymbol{u}_j + \boldsymbol{K}_{C0}^{-1} \left( \boldsymbol{F}_{ext}(\boldsymbol{u}_j)
    - \boldsymbol{K}_{C0} \boldsymbol{u}_j \right), \qquad
    \boldsymbol{u}_0 = \boldsymbol{K}_{C0}^{-1} \boldsymbol{F}_{ext}

which start from the solution of the reference load ``calc_fext()`` and
reuse one factorization of `\boldsymbol{K}_{C0}`, until the convergence
criterion of the non-linear solvers is satisfied, see
:doc:`convergence_criteria`, or ``maxNumIter`` iterations are done; a
warning is printed and the last iterate returned when they do not
converge. This is a fixed-point iteration without the load stiffness in
the iteration matrix. For a load affine in the displacements,
`\boldsymbol{F}_{ext}(\boldsymbol{u}) = \boldsymbol{F}_0 - \boldsymbol{K}_F \boldsymbol{u}`, its limit is the
solution of `(\boldsymbol{K}_{C0} + \boldsymbol{K}_F) \boldsymbol{u} = \boldsymbol{F}_0`, and the
error is multiplied at each iteration by `-\boldsymbol{K}_{C0}^{-1} \boldsymbol{K}_F`,
whose spectral radius is `1/\min|\lambda|` over the load multipliers of
`(\boldsymbol{K}_{C0} + \lambda \boldsymbol{K}_F) \boldsymbol{u} = \boldsymbol{0}`, negative and complex
ones included: the iterations converge, linearly, below the load of the
smallest `|\lambda|` and diverge beyond it, see :meth:`.Analysis.static`.
For an affine load whose `\boldsymbol{K}_F` is known, solving the unsymmetric
system with :func:`.linear_static` avoids the iterations; with geometric
non-linearity, the load stiffness enters the tangent stiffness matrix
instead, see :doc:`configuration_dependent_loads`.


Verification
------------

``tests/test_solver_static.py`` checks :func:`.solve` and
:func:`.linear_static` against exact solutions of small systems and the
deflection of a Ritz plate against the Navier solution of Timoshenko and
Woinowsky-Krieger (1959), Art. 29;
``tests/test_prescribed_displacement_linear.py`` checks a cantilever with a
prescribed tip deflection, and ``tests/test_solver_analysis.py`` and
``tests/test_solver_follower_loads.py`` the linear analysis of
:class:`.Analysis` with prescribed displacements and with a
configuration-dependent load.


References
----------

- Bathe, K. J., Finite Element Procedures, 2nd ed., K. J. Bathe,
  Watertown, MA, 2014.
- Timoshenko, S. and Woinowsky-Krieger, S., Theory of Plates and Shells,
  2nd ed., McGraw-Hill, 1959.
