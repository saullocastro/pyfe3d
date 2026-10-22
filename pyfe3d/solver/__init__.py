r"""
Solvers (:mod:`pyfe3d.solver`)
==============================

.. currentmodule:: pyfe3d.solver

Solvers for the matrices and vectors that the elements of this package
assemble, all available from the ``pyfe3d.solver`` namespace:

- eigenvalue analyses, :mod:`pyfe3d.solver.eigen`: :func:`linear_buckling`
  and :func:`natural_frequency`, with the helpers
  :func:`diagonal_preconditioner`, :func:`estimate_cayley_sigma`,
  :func:`is_positive_definite` and :func:`check_eigenpairs`.

"""
from .eigen import (diagonal_preconditioner, estimate_cayley_sigma,
                    is_positive_definite, check_eigenpairs, linear_buckling,
                    natural_frequency)

__all__ = ['diagonal_preconditioner', 'estimate_cayley_sigma',
           'is_positive_definite', 'check_eigenpairs', 'linear_buckling',
           'natural_frequency']
