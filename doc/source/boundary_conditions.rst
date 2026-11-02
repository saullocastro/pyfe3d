Boundary conditions and prescribed displacements
================================================

The elements assemble the matrices and vectors of all the ``N`` dofs of
the model, see :doc:`overview`, without any support. The supports and the
prescribed displacements are imposed afterwards, by partitioning the dofs
into the free ones, whose displacements are unknown, and the known ones,
whose displacements are prescribed. This page gives the partition in the
linear analyses, its extension to the non-linear analyses, where the
prescribed displacements grow with the load factor and are imposed
exactly, the reactions, and the errors that the solvers report when the
supports are insufficient.


The partition of the dofs
-------------------------

At each dof either the external force or the displacement is known. The
Boolean array ``bu`` of size ``N`` marks the free dofs, with the subscript
`u`, and its complement ``bk = ~bu`` the known ones, with the subscript
`k`, whose prescribed displacements `\bar{\boldsymbol{u}}_k` are given in
``uk[bk]``; a fixed support is the particular case `\bar{\boldsymbol{u}}_k =
\boldsymbol{0}`. The dofs are not reordered: the blocks are selected with the
masks, e.g. ``KC0uu = KC0[bu, :][:, bu]`` and ``KC0uk = KC0[bu, :][:,
bk]``. Written as if the free dofs came first, the linear static system
`\boldsymbol{K}_{C0} \boldsymbol{u} = \boldsymbol{F}_{ext}` of :doc:`linear_analyses`
reads (Bathe 2014, Sec. 4.2.2, Eq. (4.42), with other subscripts)

.. math::
    :label: bc-partition

    \begin{bmatrix} \boldsymbol{K}_{{C0}_{uu}} & \boldsymbol{K}_{{C0}_{uk}} \\
                    \boldsymbol{K}_{{C0}_{ku}} & \boldsymbol{K}_{{C0}_{kk}} \end{bmatrix}
    \begin{Bmatrix} \boldsymbol{u}_u \\ \bar{\boldsymbol{u}}_k \end{Bmatrix}
    =
    \begin{Bmatrix} \boldsymbol{F}_{{ext}_u} \\ \boldsymbol{F}_{{ext}_k} + \boldsymbol{R}_k \end{Bmatrix}

with `\boldsymbol{F}_{{ext}_k}` the loads applied at the known dofs and
`\boldsymbol{R}_k` the unknown reactions. The first block row is the system of
the free dofs, in which the prescribed displacements act as the equivalent
forces `-\boldsymbol{K}_{{C0}_{uk}} \bar{\boldsymbol{u}}_k` (Eq. (4.43)), and the
second gives the reactions (Eq. (4.45)):

.. math::
    :label: bc-free

    \boldsymbol{K}_{{C0}_{uu}} \boldsymbol{u}_u = \boldsymbol{F}_{{ext}_u} - \boldsymbol{K}_{{C0}_{uk}} \bar{\boldsymbol{u}}_k

.. math::
    :label: bc-reactions

    \boldsymbol{R}_k = \boldsymbol{K}_{{C0}_{ku}} \boldsymbol{u}_u + \boldsymbol{K}_{{C0}_{kk}} \bar{\boldsymbol{u}}_k
    - \boldsymbol{F}_{{ext}_k}

Only `\boldsymbol{K}_{{C0}_{uu}}` is factorized. In terms of the residual
`\boldsymbol{R} = \boldsymbol{F}_{int} - \boldsymbol{F}_{ext}`, with
`\boldsymbol{F}_{int} = \boldsymbol{K}_{C0} \boldsymbol{u}` in a linear analysis, the
first row is the equilibrium `\boldsymbol{R}_u = \boldsymbol{0}` of the free dofs
and `\boldsymbol{R}_k` is the residual at the known ones. In code, with the
assembled ``KC0``, ``fext`` and ``uk``::

    u = np.zeros(N)
    u[bk] = uk[bk]
    u[bu] = solve(KC0[bu, :][:, bu], fext[bu] - KC0[bu, :][:, bk] @ uk[bk])
    Rk = KC0[bk, :] @ u - fext[bk]

where ``KC0[bk, :] @ u`` is the internal force vector at the known dofs,
which the ``update_fint`` methods of the elements also give, see step 7 of
:doc:`overview`. The matrices of the eigenvalue analyses are partitioned in
the same way and only their free-free blocks are used, see
:doc:`eigenvalue_analyses`: the eigenvalue problems are homogeneous, and the
known dofs have null displacements in the buckling and vibration modes.


Prescribed displacements in the non-linear analyses
---------------------------------------------------

:class:`.Analysis` performs the partition itself when it receives ``bu``,
and imposes the prescribed displacements ``uk`` exactly. The callables of
the model then receive and return the vectors and matrices of all the
``N`` dofs, as assembled by the elements, and :class:`.FreeDofs` restricts
them to the free dofs for the solvers. The prescribed displacements grow
with the load factor `\lambda`, `\boldsymbol{u}_k = \lambda \bar{\boldsymbol{u}}_k`,
with ``uk[bk]`` the values `\bar{\boldsymbol{u}}_k` at `\lambda = 1`, such that
the configuration of the full model is the null-space reduction

.. math::
    :label: bc-null-space

    \boldsymbol{u} = \boldsymbol{Z} \boldsymbol{c} + \lambda \bar{\boldsymbol{u}}_k, \qquad
    \boldsymbol{Z} = \begin{bmatrix} \boldsymbol{I}_{uu} \\ \boldsymbol{0}_{ku} \end{bmatrix}

with `\boldsymbol{c} = \boldsymbol{u}_u` the unknowns of the solvers, see
:meth:`.FreeDofs.full`, and `\bar{\boldsymbol{u}}_k` placed at the known dofs.
The solvers see the reduced residual, its Jacobian and its derivative with
respect to the load factor:

.. math::
    :label: bc-reduced-residual

    \boldsymbol{R}_u(\boldsymbol{c}, \lambda) = \boldsymbol{F}_{{int}_u}(\boldsymbol{u}) - \lambda \boldsymbol{F}_{{ext}_u},
    \qquad
    \frac{\partial \boldsymbol{R}_u}{\partial \boldsymbol{c}} = \boldsymbol{K}_{T_{uu}},
    \qquad
    -\frac{\partial \boldsymbol{R}_u}{\partial \lambda} = \boldsymbol{F}_{{ext}_u} - \boldsymbol{K}_{T_{uk}} \bar{\boldsymbol{u}}_k

with `\boldsymbol{K}_T` the tangent stiffness matrix of the full model at the
current configuration and load factor, whose blocks are returned by
:meth:`.FreeDofs.calc_kC` and :meth:`.FreeDofs.calc_kG`. The last vector,
the out-of-balance force that a unit increment of the load factor produces
at the free dofs held fixed, depends on the configuration and on the load
factor. It is passed to the solvers as a load vector that depends on the
load factor, ``calc_fext(inc=1., c=c, lbd=lbd)``, see
:doc:`configuration_dependent_loads`, and it is what drives an analysis
under prescribed displacements alone, where `\boldsymbol{F}_{ext}` is null: it
gives the predictor of the first Newton-Raphson increment, the scaling of
the arc length and a reference of the convergence criteria, see
:doc:`newton_raphson_method`, :doc:`arc_length_methods` and
:doc:`convergence_criteria`. In the linear analysis, with
`\boldsymbol{K}_T = \boldsymbol{K}_{C0}`, the system solved is
:eq:`bc-free`, with the right-hand side of :meth:`.FreeDofs.linear_fext`.

The prescribed values are imposed exactly at every iteration: the
corrections of the solvers vanish at the known dofs, and the norms of the
convergence criteria are evaluated over the free dofs only. At a converged
configuration the reactions are the residual at the known dofs,

.. math::
    :label: bc-reactions-nl

    \boldsymbol{R}_k = \boldsymbol{F}_{{int}_k}(\boldsymbol{u}) - \lambda \boldsymbol{F}_{{ext}_k}

given by the ``calc_fint`` of the full model, which reduces to
:eq:`bc-reactions` in a linear analysis. The solution vectors ``an.cs``
of :meth:`.Analysis.static` have size ``N``, with `\lambda
\bar{\boldsymbol{u}}_k` at the known dofs. A model without load,
``calc_fext`` returning zeros, and with a prescribed displacement is
displacement controlled; with the arc-length methods, the load factor then
measures the prescribed displacement, which can pass a limit point of the
reaction, see ``tests/test_dome_prescribed_displacement.py``.

The same reduction applies to any linear constraint `\boldsymbol{u} =
\boldsymbol{Z} \boldsymbol{c} + \boldsymbol{u}_{0} + \lambda \boldsymbol{u}_{1}`, e.g.
symmetry conditions or a restriction of a model to some of its modes,
implemented by the user in the callables, see :mod:`pyfe3d.solver.callbacks`;
:class:`.FreeDofs` is the case of `\boldsymbol{Z}` made of columns of the
identity matrix.

Without ``bu``, the callables return the vectors and matrices of the free
dofs only, and the partition is done by the user; ``uk`` then cannot be
given.


Errors of the supports
----------------------

The solvers of :mod:`pyfe3d.solver` do not remove null rows and columns. A
dof left without stiffness, e.g. an unattached node, an unstiffened drilling
rotation or a missing boundary condition, is a modelling error that makes
the matrix singular, and it raises a :class:`numpy.linalg.LinAlgError`
naming the dofs, see :func:`.check_null_rows_cols`. Insufficient supports
that leave a rigid-body mode free, a mechanism, or distorted elements, e.g.
re-entrant quadrilaterals, leave `\boldsymbol{K}_{{C0}_{uu}}` with a null or a
negative eigenvalue. For a symmetric matrix this is detected from the
signs of the pivots of its factorization, by Sylvester's law of inertia
(:func:`.is_positive_definite`), and :func:`.linear_static`, the linear
:meth:`.Analysis.static` and :func:`.linear_buckling` raise a
:class:`numpy.linalg.LinAlgError` that names these likely causes, see
:doc:`linear_analyses`. A singular matrix whose round-off pivots all come
out positive is not detected by this test.

A structure without supports, e.g. in flight, has a singular
`\boldsymbol{K}_{C0}` by nature. It is solved with inertia relief, whose
reference support is a set of prescribed null displacements chosen to
suppress the rigid-body modes without restraining any deformation, see
:doc:`inertia_relief`.


References
----------

- Bathe, K. J., Finite Element Procedures, 2nd ed., K. J. Bathe,
  Watertown, MA, 2014.
