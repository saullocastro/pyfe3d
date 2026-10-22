r"""
Linear static solution (:mod:`pyfe3d.solver.static`)
====================================================

.. currentmodule:: pyfe3d.solver.static

:func:`solve` solves a sparse linear system after removing the null rows and
columns of its matrix, and :func:`linear_static` is the linear static
analysis based on it, with the outputs of :meth:`.Analysis.static`. The
linear and non-linear static analyses of a model given by callables are done
with :class:`.Analysis`.

"""
import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve, splu

from .sparseutils import remove_null_cols


def solve(A, b, **kwargs):
    r"""Solve a linear system removing the null rows and columns

    Wrapper of :func:`scipy.sparse.linalg.spsolve`. The degrees-of-freedom
    whose row and column of ``A`` are both null, typically the constrained
    degrees-of-freedom of a matrix assembled with their rows and columns
    zeroed, are removed with :func:`.remove_null_cols` before solving, and
    the corresponding values of the solution are zero.

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
        Solution, 1-D array of size ``n`` with the ``dtype`` of ``b``, null
        at the removed degrees-of-freedom.

    See Also
    --------
    linear_static : the linear static analysis based on it.

    Notes
    -----
    With `u` the set of the degrees-of-freedom whose row or column of `[A]`
    is not null, the reduced system `[A_{uu}]\{x_u\} = \{b_u\}` is solved
    and `x = 0` at the other degrees-of-freedom. The entries of `\{b\}` at
    the removed degrees-of-freedom are ignored: `\{x\}` solves the full
    system only when they are null, otherwise the full system has no
    solution, a null row and column making `[A]` singular.

    """
    A, used = remove_null_cols(A)
    b = np.asarray(b)
    x = np.zeros(b.shape[0], dtype=b.dtype)
    x[used] = spsolve(csc_matrix(A), b[used], **kwargs)
    return x


def _factorize(kT):
    r"""LU factorization of a tangent stiffness matrix, to be reused

    The null rows and columns of ``kT`` are removed, as in :func:`solve`,
    and the reduced matrix is factorized with
    :func:`scipy.sparse.linalg.splu`.

    Parameters
    ----------
    kT : sparse_matrix or np.ndarray
        Square matrix, shape ``(n, n)``, symmetric or not.

    Returns
    -------
    solve_kT : callable
        ``solve_kT(b)`` returns the solution ``x`` of ``kT @ x = b``, a 1-D
        array of size ``n``, null at the removed degrees-of-freedom.

    Raises
    ------
    RuntimeError
        Raised by :func:`scipy.sparse.linalg.splu` when the reduced matrix
        is exactly singular.

    Notes
    -----
    The reduced matrix is factorized once with SuperLU, a sparse LU
    factorization with row and column permutations, valid for unsymmetric
    matrices. Each call of ``solve_kT`` is then only a forward and a
    backward substitution, which is why the non-linear solvers reuse the
    factorization while the tangent stiffness matrix is not updated, e.g. in
    the modified Newton-Raphson method and in the two solutions of the
    bordered system of the arc-length methods.

    """
    kT_red, used = remove_null_cols(kT)
    lu = splu(csc_matrix(kT_red))

    def solve_kT(b):
        b = np.asarray(b)
        x = np.zeros(b.shape[0], dtype=np.result_type(b, lu.U.dtype))
        x[used] = lu.solve(np.ascontiguousarray(b[used]))
        return x

    return solve_kT


def linear_static(K, fext):
    r"""Linear static analysis

    Solves ``K @ u = fext`` with :func:`solve`, i.e. removing the null rows
    and columns of ``K``, and returns the solution with the outputs of
    :meth:`.Analysis.static`, a single increment of load factor ``1.``.

    Parameters
    ----------
    K : sparse_matrix
        Stiffness matrix, shape ``(n, n)``. It should include the constant
        terms that apply, e.g. a constant geometric stiffness matrix of a
        preload or an aerodynamic matrix. It may be unsymmetric, e.g. ``KC0 +
        KL`` for a follower load affine in the displacements on a
        geometrically linear structure, whose load stiffness is ``KL =
        -dfext/du``.
    fext : np.ndarray
        External force vector, 1-D array of size ``n``.

    Returns
    -------
    increments : list
        ``[1.]``, the load factor of the solution.
    cs : list
        List with the solution vector, 1-D array of size ``n``.

    Notes
    -----
    The system solved is `[K]\{u\} = \{F_{ext}\}`, the first Newton-Raphson
    iteration of the non-linear problem from `\{u\} = \{0\}`, where the
    tangent stiffness matrix is the linear constitutive stiffness matrix
    `[K_{C_0}]`. For a follower load affine in the displacements, `\{F(u)\}
    = \{F_0\} - [K_L]\{u\}`, the linear solution is that of `([K_{C_0}] +
    [K_L])\{u\} = \{F_0\}`.

    """
    return [1.], [solve(K, fext)]
