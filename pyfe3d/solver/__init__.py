r"""
==============================
Solvers (:mod:`pyfe3d.solver`)
==============================

.. currentmodule:: pyfe3d.solver

Solvers for the matrices and vectors that the elements of this package
assemble, all available from the ``pyfe3d.solver`` namespace:

- linear static solution, :mod:`pyfe3d.solver.static`: :func:`.solve` and
  :func:`.linear_static`;
- linear static analysis of unsupported structures with inertia relief,
  :mod:`pyfe3d.solver.inertia_relief`: :func:`.inertia_relief`;
- linear and non-linear static analyses of a model given by callables,
  :mod:`pyfe3d.solver.analysis`: :class:`.Analysis`, with the Newton-Raphson
  method, full or modified and with an optional line search,
  :mod:`pyfe3d.solver.newton_raphson`, and the arc-length methods of Riks
  and of Crisfield, :mod:`pyfe3d.solver.arc_length`; configuration- and
  load-factor-dependent loads, :mod:`pyfe3d.solver.callbacks`, and
  prescribed displacements imposed exactly, :class:`.FreeDofs`;
- Crisfield's convergence criterion, the default of the non-linear
  solvers, :mod:`pyfe3d.solver.convergence`: :func:`.crisfield_weights` and
  :func:`.crisfield_zeta`;
- eigenvalue analyses, :mod:`pyfe3d.solver.eigen`: :func:`.linear_buckling`,
  :func:`.natural_frequency` and, for damped systems,
  :func:`.complex_frequency`, with the helpers
  :func:`.diagonal_preconditioner`, :func:`.estimate_cayley_sigma`,
  :func:`.is_positive_definite` and :func:`.check_eigenpairs`, for
  symmetric and unsymmetric matrices, with the sparse solvers first and a
  dense solver as a last resort, :class:`.DenseFallbackWarning`;
- the cap of ARPACK's ``ncv`` against the faulty ``dsteqr`` of some Intel
  MKL versions, :mod:`pyfe3d.solver.arpackutils`;
- sparse matrix utilities, :mod:`pyfe3d.solver.sparseutils`:
  :func:`.check_null_rows_cols` and :func:`.is_symmetric`.

The matrices given to the solvers are those of the free degrees-of-freedom,
e.g. ``KC0uu = KC0[bu, :][:, bu]``: a null row or column of the stiffness
matrix, a degree-of-freedom without stiffness, raises a
:class:`numpy.linalg.LinAlgError` naming the degrees-of-freedom, see
:func:`.check_null_rows_cols`. A symmetric stiffness matrix that is not
positive definite, e.g. with insufficient boundary conditions, a mechanism
or re-entrant elements, raises a :class:`numpy.linalg.LinAlgError` that
says so, with the likely causes, in :func:`.linear_buckling`,
:func:`.linear_static` and the linear :meth:`.Analysis.static`, and the
``RuntimeError`` of :func:`.natural_frequency` starts with that diagnosis
when its solvers fail.

The log messages of the non-linear static analyses are printed by
:mod:`pyfe3d.solver.logger`, unless ``silent=True``.

"""
from .eigen import (DenseFallbackWarning, diagonal_preconditioner,
                    estimate_cayley_sigma, is_positive_definite,
                    check_eigenpairs, linear_buckling, natural_frequency,
                    complex_frequency)
from .static import solve, linear_static
from .inertia_relief import inertia_relief
from .analysis import Analysis, FreeDofs
from .convergence import crisfield_weights, crisfield_zeta
from .sparseutils import check_null_rows_cols, is_symmetric

__all__ = ['DenseFallbackWarning', 'diagonal_preconditioner', 'estimate_cayley_sigma',
           'is_positive_definite', 'check_eigenpairs', 'linear_buckling',
           'natural_frequency', 'complex_frequency', 'solve', 'linear_static', 'inertia_relief', 'Analysis',
           'FreeDofs', 'crisfield_weights', 'crisfield_zeta',
           'check_null_rows_cols', 'is_symmetric']
