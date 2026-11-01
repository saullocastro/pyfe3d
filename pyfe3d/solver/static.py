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
import numpy as np
from numpy.linalg import LinAlgError
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu

from .sparseutils import (check_null_rows_cols, is_symmetric,
                          _not_positive_definite_message)


def solve(A, b, **kwargs):
    r"""Solve a linear system, raising an error for a singular matrix

    The system is solved with the LU factorization of
    :func:`scipy.sparse.linalg.splu`. ``A`` must have only the free
    degrees-of-freedom, e.g. ``KC0uu = KC0[bu, :][:, bu]``: a null row or
    column, see :func:`.check_null_rows_cols`, and an exactly singular
    matrix raise an error instead of giving a solution with NaN values, as
    :func:`scipy.sparse.linalg.spsolve` does.

    Parameters
    ----------
    A : sparse_matrix or np.ndarray
        Square matrix, shape ``(n, n)``. It may be unsymmetric, e.g. with the
        load stiffness of a follower load, since the factorization is an LU
        factorization.
    b : np.ndarray
        Right-hand side, real or complex, 1-D array of size ``n``.
    kwargs : keyword arguments, optional
        Passed to :func:`scipy.sparse.linalg.splu`, e.g. ``permc_spec``.

    Returns
    -------
    x : np.ndarray
        Solution, 1-D array of size ``n``.

    Raises
    ------
    numpy.linalg.LinAlgError
        When ``A`` has a null row or column, with the indices of these
        degrees-of-freedom, see :func:`.check_null_rows_cols`, or when it is
        exactly singular, i.e. when :func:`scipy.sparse.linalg.splu` raises
        a ``RuntimeError``.

    See Also
    --------
    linear_static : the linear static analysis based on it.

    Notes
    -----
    The system `\boldsymbol{A} \boldsymbol{x} = \boldsymbol{b}` is solved as given. In a finite element
    model the known displacements are partitioned out beforehand, and a null
    row or column of `\boldsymbol{A}` is a degree-of-freedom without stiffness, i.e. a
    modelling error, which is reported instead of being removed. This
    deliberately differs from structsolve, whose solvers remove the null
    rows and columns left by the boundary conditions of its semi-analytical
    models, see :func:`.check_null_rows_cols`.

    An exactly singular matrix is detected by the factorization, which
    raises, instead of the :class:`scipy.sparse.linalg.MatrixRankWarning` and
    the NaN values of :func:`scipy.sparse.linalg.spsolve`. Turning that
    warning into an error needs a :func:`warnings.catch_warnings` block,
    whose exit resets the registry of the warnings already shown, of every
    module, such that the warnings of the caller shown once per location
    would be shown again after each call.

    """
    A = csc_matrix(A)
    check_null_rows_cols(A, 'A')
    try:
        lu = splu(A, **kwargs)
    except RuntimeError as e:
        raise LinAlgError('A is exactly singular: %s' % e) from e
    b = np.asarray(b)
    if np.iscomplexobj(b) and not np.iscomplexobj(A.data):
        return (lu.solve(np.ascontiguousarray(b.real, dtype=float))
                + 1j*lu.solve(np.ascontiguousarray(b.imag, dtype=float)))
    return lu.solve(np.ascontiguousarray(b, dtype=np.result_type(A.dtype,
                                                                  b.dtype,
                                                                  float)))


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


def _solve_linear_static(K, fext, analysis='Linear static analysis'):
    r"""Solution of a linear static analysis: :func:`solve`, with a check
    that a symmetric ``K`` is positive definite

    Parameters
    ----------
    K : sparse_matrix or np.ndarray
        Stiffness matrix, shape ``(n, n)``.
    fext : np.ndarray
        External force vector, 1-D array of size ``n``.
    analysis : str, optional
        Name of the analysis in the message of the error. Default
        ``'Linear static analysis'``.

    Returns
    -------
    u : np.ndarray
        Solution, 1-D array of size ``n``.

    Raises
    ------
    numpy.linalg.LinAlgError
        As :func:`solve`, and when ``K`` is symmetric, see
        :func:`.is_symmetric`, and not positive definite, with the likely
        causes, see :func:`.sparseutils._not_positive_definite_message`.

    Notes
    -----
    A symmetric ``K`` is factorized once, with
    :func:`.eigen._symmetric_lu`, a symmetric permutation without row
    interchanges, whose number of negative pivots is its number of negative
    eigenvalues by Sylvester's law of inertia, see
    :func:`.eigen._negative_pivots`; when they are all positive, i.e. ``K``
    is positive definite, this factorization, which is then that of
    Cholesky up to the scaling of the rows, gives the solution. An
    unsymmetric ``K``, e.g. with the load stiffness of a follower load, is
    solved with :func:`solve` without this check.

    """
    from .eigen import _symmetric_lu
    if not is_symmetric(K):
        return solve(K, fext)
    K = csc_matrix(K)
    check_null_rows_cols(K, 'A')
    lu, count = _symmetric_lu(K)
    if count != 0:
        raise LinAlgError(_not_positive_definite_message(analysis, 'K',
                                                         count))
    b = np.asarray(fext)
    if np.iscomplexobj(b) and not np.iscomplexobj(K.data):
        return (lu.solve(np.ascontiguousarray(b.real, dtype=float))
                + 1j*lu.solve(np.ascontiguousarray(b.imag, dtype=float)))
    return lu.solve(np.ascontiguousarray(b, dtype=np.result_type(K.dtype,
                                                                  b.dtype,
                                                                  float)))


def linear_static(K, fext):
    r"""Linear static analysis

    Solves ``K @ u = fext`` with :func:`solve` and returns the solution with
    the outputs of :meth:`.Analysis.static`, a single increment of load
    factor ``1.``. A symmetric ``K`` must be positive definite, which is
    checked with its factorization, see :func:`_solve_linear_static`.

    Parameters
    ----------
    K : sparse_matrix
        Stiffness matrix of the free degrees-of-freedom, shape ``(n, n)``.
        It should include the constant terms that apply, e.g. a constant
        geometric stiffness matrix of a preload or an aerodynamic matrix. It
        may be unsymmetric, e.g. ``KC0 + KF`` for a follower load affine in
        the displacements on a geometrically linear structure, whose load
        stiffness is ``KF = -dfext/du``.
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
        When ``K`` is singular, see :func:`solve`, or when it is symmetric
        and not positive definite, with the likely causes: insufficient
        boundary conditions, a mechanism, distorted elements, e.g.
        re-entrant quadrilaterals, or a preload beyond the buckling load in
        ``K``, see :func:`_solve_linear_static`.

    Notes
    -----
    The system solved is `\boldsymbol{K} \boldsymbol{u} = \boldsymbol{F}_{ext}`, the first Newton-Raphson
    iteration of the non-linear problem from `\boldsymbol{u} = \boldsymbol{0}`, where the
    tangent stiffness matrix is the linear constitutive stiffness matrix
    `\boldsymbol{K}_{C0}`. For a follower load affine in the displacements, `\boldsymbol{F}(\boldsymbol{u})
    = \boldsymbol{F}_0 - \boldsymbol{K}_F \boldsymbol{u}`, the linear solution is that of `(\boldsymbol{K}_{C0} +
    \boldsymbol{K}_F) \boldsymbol{u} = \boldsymbol{F}_0`.

    """
    return [1.], [_solve_linear_static(K, fext)]
