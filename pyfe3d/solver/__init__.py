r"""
Solvers (:mod:`pyfe3d.solver`)
==============================

.. currentmodule:: pyfe3d.solver

Solvers for the matrices and vectors that the elements of this package
assemble, all available from the ``pyfe3d.solver`` namespace:

- linear static solution, :mod:`pyfe3d.solver.static`: :func:`solve`, which
  removes the null rows and columns of the matrix, and
  :func:`linear_static`;
- eigenvalue analyses, :mod:`pyfe3d.solver.eigen`: :func:`linear_buckling`
  and :func:`natural_frequency`, with the helpers
  :func:`diagonal_preconditioner`, :func:`estimate_cayley_sigma`,
  :func:`is_positive_definite` and :func:`check_eigenpairs`;
- sparse matrix utilities, :mod:`pyfe3d.solver.sparseutils`:
  :func:`remove_null_cols` and :func:`is_symmetric`.

"""
from .eigen import (diagonal_preconditioner, estimate_cayley_sigma,
                    is_positive_definite, check_eigenpairs, linear_buckling,
                    natural_frequency)
from .static import solve, linear_static
from .sparseutils import remove_null_cols, is_symmetric

__all__ = ['diagonal_preconditioner', 'estimate_cayley_sigma',
           'is_positive_definite', 'check_eigenpairs', 'linear_buckling',
           'natural_frequency', 'solve', 'linear_static',
           'remove_null_cols', 'is_symmetric']
