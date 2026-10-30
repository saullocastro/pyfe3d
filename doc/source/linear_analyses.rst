Linear static analyses
======================

A linear static analysis solves `\boldsymbol{K}_{C0} \boldsymbol{u} = \boldsymbol{f}` for the
dofs whose displacements are unknown, with the matrix ``KC0`` assembled as
in :doc:`overview`. This page gives the partition of the dofs, with zero or
prescribed displacements, the reactions, and the solvers.


Partition of the dofs
---------------------

At each dof either the load or the displacement is known. The Boolean array
``bu`` marks the free dofs, `u`, whose displacements are unknown, and its
complement ``bk = ~bu`` the known ones, `k`, with the prescribed
displacements ``uk[bk]``, `\boldsymbol{u}_k`, zero at a fixed support. Bathe (2014,
Sec. 4.2.2, Eqs. (4.42) to (4.45)) writes the same partition with other
subscripts. Without reordering the dofs, the system reads

.. math::

    \begin{bmatrix} \boldsymbol{K}_{{C0}_{uu}} & \boldsymbol{K}_{{C0}_{uk}} \\
                    \boldsymbol{K}_{{C0}_{ku}} & \boldsymbol{K}_{{C0}_{kk}} \end{bmatrix}
    \begin{Bmatrix} \boldsymbol{u}_u \\ \boldsymbol{u}_k \end{Bmatrix}
    =
    \begin{Bmatrix} \boldsymbol{f}_u \\ \boldsymbol{f}_k + \boldsymbol{R}_k \end{Bmatrix}

with `\boldsymbol{f}_k` the loads applied at the known dofs and `\boldsymbol{R}_k` the
unknown reactions. The first row gives the system of the free dofs, where
the prescribed displacements act as the forces `-\boldsymbol{K}_{{C0}_{uk}}
\boldsymbol{u}_k`, and the second row the reactions:

.. math::

    \boldsymbol{K}_{{C0}_{uu}} \boldsymbol{u}_u = \boldsymbol{f}_u - \boldsymbol{K}_{{C0}_{uk}} \boldsymbol{u}_k,
    \qquad
    \boldsymbol{R}_k = \boldsymbol{K}_{{C0}_{ku}} \boldsymbol{u}_u + \boldsymbol{K}_{{C0}_{kk}} \boldsymbol{u}_k
    - \boldsymbol{f}_k

which, with SciPy::

    KC0uu = KC0[bu, :][:, bu]
    KC0uk = KC0[bu, :][:, bk]
    u = np.zeros(N)
    u[bk] = uk[bk]
    u[bu] = solve(KC0uu, f[bu] - KC0uk @ uk[bk])
    Rk = KC0[bk, :] @ u - f[bk]

``KC0[bk, :] @ u`` is `\boldsymbol{K}_{{C0}_{ku}} \boldsymbol{u}_u + \boldsymbol{K}_{{C0}_{kk}}
\boldsymbol{u}_k`, the internal forces at the known dofs, which ``update_fint`` of
the elements also gives, `\boldsymbol{F}_{{int}_k} - \boldsymbol{f}_k`, see step 7 of
:doc:`overview`. For fixed supports, ``uk`` is zero and the system is
`\boldsymbol{K}_{{C0}_{uu}} \boldsymbol{u}_u = \boldsymbol{f}_u`. The same partition gives the
matrices of the eigenvalue analyses, see :doc:`eigenvalue_analyses`.


Solvers
-------

- :func:`.solve` solves a sparse system with an LU factorization and
  raises an error for a singular matrix, where
  :func:`scipy.sparse.linalg.spsolve` returns NaN values.
- :func:`.linear_static` solves ``K @ u = fext`` of the free dofs, checks
  that a symmetric ``K`` is positive definite and returns ``[1.]`` and
  ``[u]``, the outputs of :meth:`.Analysis.static`.
- :meth:`.Analysis.static` with ``NLgeom=False`` solves the same system
  from the callables of a model, see :doc:`nonlinear_analyses`, in a single
  increment. With ``Analysis(..., bu=bu, uk=uk)``, the callables work with
  all the ``N`` dofs, the system of the free dofs above is formed by
  :meth:`.FreeDofs.linear_fext`, and ``cs[0]`` has size ``N``, with
  ``uk[bk]`` at the known dofs; the reactions are ``calc_fint(u) -
  calc_fext()`` at the known dofs. A configuration-dependent load is
  corrected by iterations, see :meth:`.Analysis.static`.

The worked example
:doc:`tut-prescribed-displacement-cantilever` solves a cantilever with a
prescribed tip deflection both ways, and :doc:`ex_linear_static` a plate
under a point load.


Errors of the supports
----------------------

The solvers of :mod:`pyfe3d.solver` do not remove null rows and columns: a
dof left without stiffness, e.g. an unattached node, an unstiffened
drilling rotation or a missing boundary condition, raises a
:class:`numpy.linalg.LinAlgError` naming the dofs, see
:func:`.check_null_rows_cols`. Insufficient boundary conditions, a
mechanism or re-entrant elements leave `\boldsymbol{K}_{{C0}_{uu}}` not positive
definite, which :func:`.linear_static`, :func:`.linear_buckling` and the
linear :meth:`.Analysis.static` report with a
:class:`numpy.linalg.LinAlgError` naming the likely causes, see
:doc:`eigenvalue_analyses`.


References
----------

- Bathe, K. J., Finite Element Procedures, 2nd ed., K. J. Bathe,
  Watertown, MA, 2014.
