r"""
====================================================
Linear static solution (:mod:`pyfe3d.solver.static`)
====================================================

.. currentmodule:: pyfe3d.solver.static

:func:`solve` solves a sparse linear system, raising an error for a singular
matrix, and :func:`linear_static` is the linear static analysis based on it,
with the outputs of :meth:`.Analysis.static`. The linear and non-linear
static analyses of a model given by callables are done with
:class:`.Analysis`.

"""
import warnings

import numpy as np
from numpy.linalg import LinAlgError
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve, splu, MatrixRankWarning

from .sparseutils import check_null_rows_cols


def solve(A, b, **kwargs):
    r"""Solve a linear system, raising an error for a singular matrix

    Wrapper of :func:`scipy.sparse.linalg.spsolve`. ``A`` must have only
    the free degrees-of-freedom, e.g. ``KC0uu = KC0[bu, :][:, bu]``: a null
    row or column, see :func:`.check_null_rows_cols`, and an exactly
    singular matrix raise an error instead of giving a solution with NaN
    values.

    Parameters
    ----------
    A : sparse_matrix or np.ndarray
        Square matrix, shape ``(n, n)``. It may be unsymmetric, e.g. with the
        load stiffness of a follower load, since
        :func:`scipy.sparse.linalg.spsolve` uses an LU factorization.
    b : np.ndarray
        Right-hand side, 1-D array of size ``n``.
    kwargs : keyword arguments, optional
        Passed to :func:`scipy.sparse.linalg.spsolve`.

    Returns
    -------
    x : np.ndarray
        Solution, 1-D array of size ``n``.

    Raises
    ------
    numpy.linalg.LinAlgError
        When ``A`` has a null row or column, with the indices of these
        degrees-of-freedom, see :func:`.check_null_rows_cols`, or when it is
        exactly singular, i.e. when :func:`scipy.sparse.linalg.spsolve`
        warns with :class:`scipy.sparse.linalg.MatrixRankWarning`.

    See Also
    --------
    linear_static : the linear static analysis based on it.

    Notes
    -----
    The system `[A]\{x\} = \{b\}` is solved as given. In a finite element
    model the known displacements are partitioned out beforehand, and a null
    row or column of `[A]` is a degree-of-freedom without stiffness, i.e. a
    modelling error, which is reported instead of being removed. This
    deliberately differs from structsolve, whose solvers remove the null
    rows and columns left by the boundary conditions of its semi-analytical
    models, see :func:`.check_null_rows_cols`.

    """
    A = csc_matrix(A)
    check_null_rows_cols(A, 'A')
    with warnings.catch_warnings():
        warnings.simplefilter('error', MatrixRankWarning)
        try:
            return spsolve(A, np.asarray(b), **kwargs)
        except MatrixRankWarning as e:
            raise LinAlgError('A is exactly singular: %s' % e) from e


def _factorize(kT):
    r"""LU factorization of a tangent stiffness matrix, to be reused

    The matrix is factorized with :func:`scipy.sparse.linalg.splu`, after
    checking that it has no null row or column, as in :func:`solve`.

    Parameters
    ----------
    kT : sparse_matrix or np.ndarray
        Square matrix, shape ``(n, n)``, symmetric or not.

    Returns
    -------
    solve_kT : callable
        ``solve_kT(b)`` returns the solution ``x`` of ``kT @ x = b``, a 1-D
        array of size ``n``.

    Raises
    ------
    numpy.linalg.LinAlgError
        When ``kT`` has a null row or column, with the indices of these
        degrees-of-freedom, see :func:`.check_null_rows_cols`, or when
        :func:`scipy.sparse.linalg.splu` finds it exactly singular.

    Notes
    -----
    The matrix is factorized once with SuperLU, a sparse LU factorization
    with row and column permutations, valid for unsymmetric matrices. Each
    call of ``solve_kT`` is then only a forward and a backward substitution,
    which is why the non-linear solvers reuse the factorization while the
    tangent stiffness matrix is not updated, e.g. in the modified
    Newton-Raphson method and in the two solutions of the bordered system of
    the arc-length methods.

    """
    kT = csc_matrix(kT)
    check_null_rows_cols(kT, 'kT')
    try:
        lu = splu(kT)
    except RuntimeError as e:
        raise LinAlgError('kT is exactly singular: %s' % e) from e

    def solve_kT(b):
        return lu.solve(np.ascontiguousarray(b))

    return solve_kT


def linear_static(K, fext):
    r"""Linear static analysis

    Solves ``K @ u = fext`` with :func:`solve` and returns the solution with
    the outputs of :meth:`.Analysis.static`, a single increment of load
    factor ``1.``.

    Parameters
    ----------
    K : sparse_matrix
        Stiffness matrix of the free degrees-of-freedom, shape ``(n, n)``.
        It should include the constant terms that apply, e.g. a constant
        geometric stiffness matrix of a preload or an aerodynamic matrix. It
        may be unsymmetric, e.g. ``KC0 + KL`` for a follower load affine in
        the displacements on a geometrically linear structure, whose load
        stiffness is ``KL = -dfext/du``.
    fext : np.ndarray
        External force vector, 1-D array of size ``n``.

    Returns
    -------
    increments : list
        ``[1.]``, the load factor of the solution.
    cs : list
        List with the solution vector, 1-D array of size ``n``.

    Raises
    ------
    numpy.linalg.LinAlgError
        When ``K`` is singular, see :func:`solve`.

    Notes
    -----
    The system solved is `[K]\{u\} = \{F_{ext}\}`, the first Newton-Raphson
    iteration of the non-linear problem from `\{u\} = \{0\}`, where the
    tangent stiffness matrix is the linear constitutive stiffness matrix
    `[K_{C0}]`. For a follower load affine in the displacements, `\{F(u)\}
    = \{F_0\} - [K_L]\{u\}`, the linear solution is that of `([K_{C0}] +
    [K_L])\{u\} = \{F_0\}`.

    """
    return [1.], [solve(K, fext)]
