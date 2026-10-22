r"""
================================================================
Utilities for sparse matrices (:mod:`pyfe3d.solver.sparseutils`)
================================================================

.. currentmodule:: pyfe3d.solver.sparseutils

Removal of the null rows and columns of the system matrices, used by the
solvers before factorizing them, and a cheap test of symmetry, used to
select the eigenvalue solvers of symmetric or unsymmetric matrices.

"""
import numpy as np
from scipy.sparse import csr_matrix


def remove_null_cols(*args):
    r"""Remove the null rows and columns of square sparse matrices

    A degree-of-freedom is removed when both its row and its column of the
    first matrix are null, and the same rows and columns are removed from
    all the matrices. An unsymmetric matrix, e.g. with the load stiffness of
    a follower load, thus never loses an equation whose row is not null. For
    a matrix that is not square, the null columns are removed.

    Parameters
    ----------
    args : sparse_matrix or np.ndarray
        One or more matrices of the same shape ``(n, n)``. The first one
        tells which rows and columns are removed from all of them.

    Returns
    -------
    out : list
        The reduced matrices, converted to :class:`scipy.sparse.csr_matrix`
        and in the order of ``args``, followed by the 1-D integer array
        ``used`` of the indices of the rows and columns that were kept, in
        increasing order.

    Notes
    -----
    With `u` the set of the degrees-of-freedom whose row or column of the
    first matrix `[A]` has a non-zero entry, the reduced matrices are
    `[A_{uu}]`. A null row and column are what the boundary conditions leave
    in a matrix assembled with the constrained degrees-of-freedom zeroed, so
    that removing them is the partition of the free degrees-of-freedom. A
    row that is not null with a null column, or the opposite, is kept, and
    the reduced matrix is then singular, as the full one.

    """
    first = csr_matrix(args[0])
    rows, cols = first.nonzero()
    if first.shape[0] == first.shape[1]:
        used = np.unique(np.concatenate((rows, cols)))
    else:
        used = np.unique(cols)
    out = []
    for i, arg in enumerate(args):
        m = first if i == 0 else csr_matrix(arg)
        out.append(m[used, :][:, used])
    out.append(used)
    return out


def is_symmetric(A, rtol=1.e-10):
    r"""Tell whether a square matrix is symmetric, in the Frobenius norm

    The matrix is symmetric when ``norm(A - A.T) <= rtol*norm(A)``, both in
    the Frobenius norm. The cost is of the order of the number of stored
    entries of ``A``.

    Parameters
    ----------
    A : sparse_matrix or np.ndarray
        Square matrix, converted to :class:`scipy.sparse.csr_matrix`.
    rtol : float, optional
        Relative tolerance, dimensionless. Default ``1.e-10``.

    Returns
    -------
    symmetric : bool
        ``True`` when ``A`` is symmetric to the relative tolerance ``rtol``.
        A null matrix is symmetric.

    Notes
    -----
    The test is `\|[A] - [A]^T\|_F \le rtol \|[A]\|_F`. The relative
    tolerance absorbs the round-off of matrices that are symmetric in exact
    arithmetic but assembled from terms evaluated in a different order on
    each side of the diagonal, whereas a genuinely unsymmetric term, such as
    the load stiffness of a follower load or the aerodynamic matrix of
    piston theory, gives a relative asymmetry that is not a round-off.

    """
    A = csr_matrix(A)
    normA = np.sqrt(np.sum(np.abs(A.data)**2))
    D = (A - A.T).tocsr()
    normD = np.sqrt(np.sum(np.abs(D.data)**2))
    return bool(normD <= rtol*normA)
