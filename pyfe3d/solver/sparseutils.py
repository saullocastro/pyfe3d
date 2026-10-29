r"""
================================================================
Utilities for sparse matrices (:mod:`pyfe3d.solver.sparseutils`)
================================================================

.. currentmodule:: pyfe3d.solver.sparseutils

The check that a matrix given to the solvers has no null row or column,
i.e. no degree-of-freedom without stiffness, done by the solvers before
factorizing it, and a cheap test of symmetry, used to select the eigenvalue
solvers of symmetric or unsymmetric matrices.

"""
import numpy as np
from numpy.linalg import LinAlgError
from scipy.sparse import csr_matrix

#: Largest number of degrees-of-freedom listed by the error of
#: :func:`check_null_rows_cols`
NUM_LISTED_DOFS = 10


def check_null_rows_cols(A, name='A'):
    r"""Raise an error when a square matrix has a null row or column

    Called by the solvers of :mod:`pyfe3d.solver` before factorizing the
    matrix, since a null row or column makes it singular.

    Parameters
    ----------
    A : sparse_matrix or np.ndarray
        Square matrix, shape ``(n, n)``.
    name : str, optional
        Name of the matrix in the error message. Default ``'A'``.

    Raises
    ------
    numpy.linalg.LinAlgError
        When a row or a column of ``A`` is null. The message gives the
        number of these degrees-of-freedom, the first
        :data:`NUM_LISTED_DOFS` of their indices in ``A``, and how to correct
        the model.

    Notes
    -----
    A row or column is null when none of its stored entries is non-zero,
    explicitly stored zeros being ignored; the cost is of the order of the
    number of stored entries of `\boldsymbol{A}`.

    In a finite element model, a null row or column of the matrix given to
    the solvers is a degree-of-freedom without stiffness, e.g. an unattached
    node, an unstiffened drilling rotation or a missing boundary condition,
    i.e. a modelling error, which is reported instead of being hidden. The
    known displacements are partitioned out before calling the solvers,
    e.g. ``KC0uu = KC0[bu, :][:, bu]`` with the boolean mask ``bu`` of the
    free degrees-of-freedom, or with ``Analysis(..., bu=bu)``, see
    :class:`.FreeDofs`. This deliberately differs from structsolve, whose
    semi-analytical Ritz models apply the boundary conditions by zeroing
    approximation functions, which leaves null rows and columns that its
    solvers remove.

    """
    A = csr_matrix(A)
    n = A.shape[0]
    rows, cols = A.nonzero()
    null = ((np.bincount(rows, minlength=n) == 0)
            | (np.bincount(cols, minlength=n) == 0))
    dofs = np.flatnonzero(null)
    if dofs.size == 0:
        return
    listed = ', '.join(str(i) for i in dofs[:NUM_LISTED_DOFS])
    if dofs.size > NUM_LISTED_DOFS:
        listed += ', ...'
    raise LinAlgError('%s is singular: null row or column at %d '
        'degree(s)-of-freedom, of indices %s. A degree-of-freedom without '
        'stiffness, e.g. an unattached node, an unstiffened drilling '
        'rotation or a missing boundary condition, is a modelling error. '
        'Partition out the constrained degrees-of-freedom, e.g. '
        'KC0uu = KC0[bu, :][:, bu] with the boolean mask bu of the free '
        'ones, or Analysis(..., bu=bu), or add the missing stiffness or '
        'boundary conditions' % (name, dofs.size, listed))


def _not_positive_definite_message(analysis, name, count):
    r"""Message of the error raised when a stiffness matrix is not positive
    definite

    Parameters
    ----------
    analysis : str
        Name of the analysis, e.g. ``'Linear buckling analysis'``.
    name : str
        Name of the matrix, e.g. ``'K'``.
    count : int or None
        Number of negative pivots of its factorization, see
        :func:`.eigen._negative_pivots`, ``None`` when the factorization met
        a zero pivot or failed.

    Returns
    -------
    message : str
        The message, in plain words, with the likely causes.

    Notes
    -----
    By Sylvester's law of inertia, the number of negative pivots of the
    factorization `\boldsymbol{K} = \boldsymbol{L} \boldsymbol{D} \boldsymbol{L}^\top` is the number of negative
    eigenvalues of `\boldsymbol{K}`, i.e. of the deformation modes with a negative
    strain energy `\frac{1}{2}\boldsymbol{u}^\top \boldsymbol{K} \boldsymbol{u}`. A singular `\boldsymbol{K}`, e.g.
    with an unrestrained rigid-body mode, has null eigenvalues, which the
    round-off turns into pivots that are null, or tiny and of either sign:
    the count then includes them, and a singular matrix whose pivots all
    come out positive is not detected.

    """
    if count is None:
        what = ('its factorization met a zero pivot, i.e. %s is singular or '
                'indefinite' % name)
    else:
        what = ('its factorization has %d negative pivot(s), i.e. %s has %d '
                'eigenvalue(s) that are negative, or null up to the '
                'round-off: %d deformation mode(s) without a positive strain '
                'energy' % (count, name, count, count))
    return ('%s: the stiffness matrix %s is not positive definite, %s. '
            'Likely causes: insufficient boundary conditions, which leave a '
            'rigid-body motion of the model or of a part of it unrestrained; '
            'a mechanism, i.e. parts that can move without deforming; '
            'distorted elements, e.g. re-entrant (non-convex) '
            'quadrilaterals, which can make it indefinite; or, when %s '
            'includes the '
            'geometric stiffness of a preload, a preload beyond the buckling '
            'load' % (analysis, name, what, name))


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
    The test is `\|\boldsymbol{A} - \boldsymbol{A}^\top\|_F \le rtol \|\boldsymbol{A}\|_F`. The relative
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
