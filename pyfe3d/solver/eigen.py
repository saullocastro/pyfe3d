r"""
================================================
Eigenvalue analyses (:mod:`pyfe3d.solver.eigen`)
================================================

.. currentmodule:: pyfe3d.solver.eigen

Linear buckling, with the constitutive and geometric stiffness matrices, and
natural frequency, with the constitutive stiffness and mass matrices, the
eigenvalue problems that the element matrices of this package are usually
assembled for. Neither problem is safe to hand to
:func:`scipy.sparse.linalg.eigsh` without preparation:

- :func:`diagonal_preconditioner` equilibrates the diagonal of the matrices,
  which does not change the eigenvalues;
- :func:`estimate_cayley_sigma` estimates, from the matrices themselves, a
  safe shift for the Cayley mode used for linear buckling;
- :func:`check_eigenpairs` verifies the result: it checks the residual of
  every eigenpair and, when the constitutive stiffness matrix is positive
  definite, that no load multiplier was missed below the highest one found,
  by counting the negative pivots of ``K + s*KG`` just below it, see
  :func:`is_positive_definite`.

:func:`linear_buckling` and :func:`natural_frequency` put them together and
are what most users should call. They also solve unsymmetric problems, e.g.
with the load stiffness of a follower load or with the aerodynamic matrix of
piston theory, and they follow the same solver policy: the sparse solvers
run first, their eigenpairs are verified, and a dense solver is used only as
a last resort, with a :class:`DenseFallbackWarning`. :func:`complex_frequency`
follows the same policy for the free vibrations of damped systems, with the
stiffness, damping and mass matrices, a quadratic eigenvalue problem.

.. note:: ARPACK, used by :func:`scipy.sparse.linalg.eigsh`, has been
    reported to return wrong eigenpairs at random, without raising, when
    linked against Intel MKL 2024.2.0 to 2025.0.0, which includes some
    Anaconda builds of SciPy, whenever its parameter ``ncv`` exceeds 32.
    MKL 2025.0.1 or newer is not affected. The faulty versions are detected
    and ``ncv`` is then capped at 32, see :mod:`pyfe3d.solver.arpackutils`.
    The verification of :func:`check_eigenpairs` catches such a failure,
    which is another reason to keep it on.

Notes
-----
The eigenvalue problems are linear buckling

.. math::
    (\boldsymbol{K}_{C0} + \lambda \boldsymbol{K}_G) \boldsymbol{u} = \boldsymbol{0}

and natural frequency

.. math::
    (\boldsymbol{K}_{C0} - \omega_n^2 \boldsymbol{M}) \boldsymbol{u} = \boldsymbol{0}

The reason this module exists is that neither problem is safe to hand to
:func:`scipy.sparse.linalg.eigsh` without preparation, and the two traps are
easy to fall into and silent when they bite:

1. **The matrices are badly scaled.** A shell element mixes translations and
   rotations, so the diagonal of `\boldsymbol{K}_{C0}` spans many orders of magnitude,
   and the drilling stiffness makes that worse: with the fictitious penalty
   of ``K6ROT`` the drilling entries are some six orders of magnitude below
   the membrane ones, and a condition number of `10^{14}` is easy to reach.
   :func:`diagonal_preconditioner` equilibrates the diagonal, which does not
   change the eigenvalues at all and usually decides whether the iterative
   solver converges.

2. **The shift of the Cayley mode has to be chosen, not guessed.** With
   ``eigsh(A=KG, M=K, sigma=sigma, mode='cayley', which='SM')`` the returned
   eigenvalues are the ones with the smallest
   `|(\mu + \sigma)/(\mu - \sigma)|`, which are the most negative `\mu`, and
   therefore the lowest positive load multipliers `\lambda = -1/\mu`, **only
   while** `\sigma` is larger than `|\mu|` of those critical eigenvalues.
   Otherwise the eigenvalues closest to `-\sigma` come back instead, with no
   warning. :func:`estimate_cayley_sigma` estimates a safe shift from the
   matrices themselves.

The inertia check of :func:`check_eigenpairs`, the number of negative pivots
of `\boldsymbol{K}_{C0} + s \boldsymbol{K}_G` just below the highest load multiplier found, is the
only cheap way to know that no mode was skipped, and it is what distinguishes
a converged answer from a plausible-looking one.

.. rubric:: Solver policy

With ``sparse_solver=True`` the sparse solvers run first for every problem
size, and each result must pass the verification. For symmetric matrices,
when the inertia check finds eigenvalues missing below the highest one
returned by the first sparse solver, e.g. copies of repeated eigenvalues,
which the Lanczos process may miss, they are computed by deflated passes
before any other solver, see :func:`_complete_repeated`. Only when all of
them failed, e.g. a failed verification, an ARPACK error or a singular
factorization, a dense solver is used, if the dense problem has at most
``max_dense_size`` degrees-of-freedom, with a :class:`DenseFallbackWarning`
giving the reasons of the failures; its result is verified as well. A
``RuntimeError`` listing every attempted solver is raised otherwise. See
:func:`_sparse_first`.

.. rubric:: Positive definite stiffness matrix

The Cayley mode of :func:`linear_buckling` and its inertia check require a
positive definite `\boldsymbol{K}_{C0}` for symmetric matrices: with ``check=True``
a `\boldsymbol{K}_{C0}` that is not raises a :class:`numpy.linalg.LinAlgError` that
says so, with its likely causes, see
:func:`.sparseutils._not_positive_definite_message`.
:func:`natural_frequency` accepts an indefinite `\boldsymbol{K}_{C0}`, e.g. with a
preload beyond the buckling load, and a singular one, with unconstrained
rigid-body modes; when its solvers fail and `\boldsymbol{K}_{C0}` is not positive
definite, its ``RuntimeError`` starts with the same diagnosis.

.. rubric:: Order of the load multipliers

:func:`linear_buckling` returns the positive load multipliers first, in
increasing order, followed by the negative ones closest to zero, i.e. the
critical load multipliers of the reversed load, then the complex ones by
increasing modulus. For symmetric matrices, the negative ones are computed
with `-\boldsymbol{K}_G` when fewer than ``num_eigvalues`` positive load multipliers
exist, see :func:`_eigsh_both_signs`.

The shift estimate, the verification, the dense fallback, the solvers of
unsymmetric matrices, the order of the load multipliers and the cap of
``ncv`` depend only on NumPy and SciPy. The ``dsteqr`` routine of the Intel
MKL versions in the note above returns wrong eigenvectors for matrices
larger than 32 by 32, see :mod:`pyfe3d.solver.arpackutils`.

"""
import sys
import warnings

import numpy as np
from scipy.sparse import csc_matrix, csr_matrix, diags
from scipy.sparse.linalg import eigs, splu, LinearOperator
from scipy.linalg import eigh, eig, LinAlgError

from .sparseutils import (check_null_rows_cols, is_symmetric,
                          _not_positive_definite_message)
from .arpackutils import (arpack_max_ncv, capped_eigsh, deflated_eigsh,
                          start_kwargs, _rayleigh_ritz)
from .logger import msg

__all__ = ['DenseFallbackWarning', 'diagonal_preconditioner',
           'estimate_cayley_sigma', 'is_positive_definite',
           'check_eigenpairs', 'linear_buckling', 'natural_frequency',
           'complex_frequency']

#: An eigenvalue `\mu` of `\boldsymbol{K}_G \boldsymbol{u} = \mu \boldsymbol{K} \boldsymbol{u}` with `|\mu|` not
#: larger than this fraction of the largest `|\mu|` is null, i.e. its load
#: multiplier is infinite up to the round-off, e.g. at the
#: degrees-of-freedom where `\boldsymbol{K}_G` is null; also the fraction of the scale
#: of the null `\nu` of :func:`_eigs_shifted`
NULL_MU_RTOL = 1.e-12
#: An eigenvalue `\mu` of the dense solvers is also null when `|\mu|` is
#: not larger than this factor times its first-order perturbation bound
#: for a backward error of the machine precision, see
#: :func:`_null_mu_pencil`
NULL_MU_PERTURBATION_FACTOR = 10.
#: The load multipliers of each sign are counted with the inertia of
#: `\boldsymbol{K} + s \boldsymbol{K}_G` below the modulus `|s|`, this factor divided by the
#: largest `|\mu|`, i.e. this factor times the lowest modulus that a load
#: multiplier may have
INERTIA_COUNT_FACTOR = 1.e6
#: An eigenpair of :func:`natural_frequency` is a rigid-body mode, null up
#: to the round-off, when `\|\boldsymbol{K} \boldsymbol{u}\| + |\omega_n^2| \, \|\boldsymbol{M} \boldsymbol{u}\|` is
#: not larger than this fraction of the matrix-norm scale `(\|\boldsymbol{K}\|_F +
#: |\omega_n^2| \, \|\boldsymbol{M}\|_F)\|\boldsymbol{u}\|`, 100 times the machine precision,
#: see :func:`_relative_residuals`
RIGID_BODY_RTOL = 100*np.finfo(float).eps
#: An eigenpair of :func:`natural_frequency` or :func:`complex_frequency`
#: whose relative residual is larger than ``check_rtol`` passes the residual
#: check when its normwise backward error is not larger than this value, the
#: machine precision, i.e. when it is accurate up to the round-off, see
#: :func:`_backward_errors`
BACKWARD_ERROR_RTOL = np.finfo(float).eps
#: Minimum number of eigenvalues per step of the shift search of
#: unsymmetric problems: few eigenvalues give small discs and many steps
SHIFT_SEARCH_MIN_EIGVALUES = 6
#: Maximum number of steps of the shift search of unsymmetric problems
SHIFT_SEARCH_MAX_STEPS = 40
#: The shift search fails when the shift exceeds this factor times the
#: radius of its first step
SHIFT_SEARCH_MAX_GROWTH = 1.e10
#: Relative gap below the critical load multiplier of the parity check of
#: the shift search
PARITY_REL_GAP = 1.e-4
#: Maximum number of deflated passes of :func:`_complete_repeated`; in
#: exact arithmetic each pass finds one more copy of every repeated
#: eigenvalue whose copies were missed, such that three passes complete
#: eigenvalues of multiplicity up to four found once
MAX_COMPLETION_PASSES = 3
#: Relative gap below the highest eigenvalue found of the inertia counts of
#: :func:`_complete_repeated`, the default ``min_rel_gap`` of
#: :func:`check_eigenpairs` and :func:`_check_freq`
COMPLETION_REL_GAP = 1.e-4
#: Seed of the starting vector of the first deflated pass of
#: :func:`_complete_repeated`, incremented for each pass; it differs from the
#: seed ``42`` of :func:`.start_vector`, see :func:`.deflated_eigsh`
COMPLETION_SEED = 43


class DenseFallbackWarning(RuntimeWarning):
    r"""Issued by :func:`linear_buckling`, :func:`natural_frequency` and
    :func:`complex_frequency` when the sparse eigenvalue solvers failed and
    a dense solver was used as a last resort

    A subclass of :class:`RuntimeWarning`, issued with :func:`warnings.warn`,
    whose message gives the reasons of the failures of the sparse solvers.
    It can be turned into an error with
    ``warnings.simplefilter('error', DenseFallbackWarning)``.

    """


class _NotApplicable(Exception):
    r"""Raised by a strategy of :func:`_sparse_first` that does not apply,
    e.g. the completion of :func:`_complete_repeated` when no eigenvalue is
    missing; the strategy is skipped without being listed among the
    failures"""


def diagonal_preconditioner(K, floor=1.e-30):
    r"""Jacobi preconditioner of a stiffness matrix

    Returns the diagonal matrix ``P`` with the entries
    ``1/sqrt(max(abs(K[i, i]), floor))``, such that ``P @ K @ P`` has a
    diagonal of ones, or minus ones for negative entries, when the moduli of
    the diagonal entries of ``K`` are larger than ``floor``.
    For a generalized eigenvalue problem both matrices are transformed,
    ``P @ A @ P`` and ``P @ B @ P``, which leaves the eigenvalues unchanged,
    and the eigenvectors of the original problem are recovered with
    ``u = P @ u_scaled``.

    Parameters
    ----------
    K : sparse_matrix
        Matrix whose diagonal sets the scaling, normally the constitutive
        stiffness matrix with the boundary conditions already applied. Shape
        ``(n, n)``.
    floor : float, optional
        Lower bound for the diagonal entries, which guards against the null
        diagonal of an unconstrained degree-of-freedom. Same dimension as the
        diagonal entries of ``K``. Default ``1.e-30``.

    Returns
    -------
    P : sparse_matrix
        Diagonal preconditioner, created with :func:`scipy.sparse.diags`,
        shape ``(n, n)``.

    Notes
    -----
    The diagonal matrix `\boldsymbol{P}` has the entries

    .. math::
        P_{ii} = \frac{1}{\sqrt{max(|K_{{C0}_{ii}}|, floor)}}

    such that `\boldsymbol{P} \boldsymbol{K}_{C0} \boldsymbol{P}` has a unit diagonal, up to the sign of the
    negative entries, which appear e.g. in a stiffness matrix with the load
    stiffness of a follower load beyond the critical load. For a generalized
    eigenvalue problem both matrices are transformed, `\boldsymbol{P} \boldsymbol{A} \boldsymbol{P}` and
    `\boldsymbol{P} \boldsymbol{B} \boldsymbol{P}`, which leaves the eigenvalues unchanged and maps the
    eigenvectors to `\boldsymbol{P}^{-1} \boldsymbol{u}`: `\boldsymbol{P} \boldsymbol{A} \boldsymbol{P} \boldsymbol{y} = w \boldsymbol{P} \boldsymbol{B} \boldsymbol{P} \boldsymbol{y}` is
    `\boldsymbol{A} \boldsymbol{u} = w \boldsymbol{B} \boldsymbol{u}` with `\boldsymbol{u} = \boldsymbol{P} \boldsymbol{y}`, `\boldsymbol{P}` being
    non-singular. The equilibration removes the spread of the diagonal
    entries, which usually reduces the condition number, see the notes of
    :mod:`pyfe3d.solver.eigen`.

    Examples
    --------
    >>> P = diagonal_preconditioner(Kuu)                  # doctest: +SKIP
    >>> eigvals, eigvecs = eigsh(A=P@KGuu@P, M=P@Kuu@P)   # doctest: +SKIP
    >>> eigvecs = P @ eigvecs                             # doctest: +SKIP

    """
    d = np.abs(np.asarray(K.diagonal(), dtype=float))
    return diags(1.0/np.sqrt(np.maximum(d, floor)))


def _estimate_mu_max(K, KG, max_iter=50, rel_tol=1.e-3):
    r"""Largest `|\mu|` of `\boldsymbol{K}_G \boldsymbol{u} = \mu \boldsymbol{K} \boldsymbol{u}` by power iterations

    Parameters
    ----------
    K, KG : sparse_matrix
        Constitutive and geometric stiffness matrices, shape ``(n, n)``.
    max_iter : int, optional
        Maximum number of power iterations. Default ``50``.
    rel_tol : float, optional
        Relative increment below which the iterations stop, dimensionless.
        Default ``1.e-3``.

    Returns
    -------
    mu_max : float or None
        The estimate, or ``None`` when ``K`` is singular or not positive
        definite, when the linear solution is inaccurate, or when the ratio
        of norms is not finite or zero.

    Notes
    -----
    See :func:`estimate_cayley_sigma`, which multiplies this estimate by a
    safety factor.

    """
    try:
        lu = splu(csc_matrix(K))
        x = np.random.RandomState(42).randn(K.shape[0])
        Kx = K @ x
        mu_max = 0.
        for i in range(max_iter):
            xKx = x @ Kx
            if not np.isfinite(xKx) or xKx <= 0:
                return None
            x = x/np.sqrt(xKx)
            rhs = KG @ x
            y = lu.solve(rhs)
            Ky = K @ y
            if i == 0:
                residual = np.linalg.norm(Ky - rhs)
                if not residual <= 1.e-6*np.linalg.norm(rhs):
                    return None
            ratio = np.sqrt(abs(y @ Ky))
            if not np.isfinite(ratio) or ratio == 0:
                return None
            converged = i > 0 and ratio - mu_max <= rel_tol*ratio
            mu_max = max(mu_max, ratio)
            if converged:
                break
            x, Kx = y, Ky
        return float(mu_max)
    except Exception:
        return None


def estimate_cayley_sigma(K, KG, safety=10., max_iter=50, rel_tol=1.e-3):
    r"""Safe shift for the Cayley mode of :func:`scipy.sparse.linalg.eigsh`

    The largest modulus of the eigenvalues ``mu`` of the generalized
    eigenvalue problem ``KG @ u = mu * K @ u`` is estimated with power
    iterations, and the result is multiplied by ``safety``.

    The fallback value ``1.`` is returned when ``K`` is singular or not
    positive definite, or when the linear solution is inaccurate.

    ``K`` is factorized with :func:`scipy.sparse.linalg.splu` and the power
    iteration starts from a vector drawn from
    :class:`numpy.random.RandomState` with the seed ``42``, such that the
    result is reproducible. The fallback value ``1.`` is also returned when
    the ratio of norms is not finite or zero, and when any exception is
    raised during the estimate.

    Parameters
    ----------
    K, KG : sparse_matrix
        Constitutive and geometric stiffness matrices, with the boundary
        conditions already applied. Shape ``(n, n)``.
    safety : float, optional
        Multiplier applied to the estimated largest ``abs(mu)``. Dimensionless.
        Default ``10.``.
    max_iter : int, optional
        Maximum number of power iterations. Default ``50``.
    rel_tol : float, optional
        Relative increment below which the power iteration is stopped.
        Dimensionless. Default ``1.e-3``.

    Returns
    -------
    sigma : float
        Shift to be passed to ``eigsh(..., sigma=sigma, mode='cayley')``, of
        the dimension of the eigenvalues ``mu``.

    Notes
    -----
    The shift has to exceed `|\mu|` of the critical eigenvalues of
    `\boldsymbol{K}_G \boldsymbol{u} = \mu \boldsymbol{K}_{C0} \boldsymbol{u}`, see the module documentation. The
    largest `|\mu|` is estimated with power iterations on
    `\boldsymbol{K}_{C0}^{-1} \boldsymbol{K}_G`, whose ratio of norms in the `\boldsymbol{K}_{C0}`-norm grows
    towards it, and the result is multiplied by ``safety``.

    At each iteration `\boldsymbol{x}` is normalized such that `\boldsymbol{x}^\top \boldsymbol{K}_{C0}
    \boldsymbol{x} = 1`, `\boldsymbol{y} = \boldsymbol{K}_{C0}^{-1} \boldsymbol{K}_G \boldsymbol{x}` is computed and the
    ratio of norms is `\sqrt{|\boldsymbol{y}^\top \boldsymbol{K}_{C0} \boldsymbol{y}|}`; `\boldsymbol{y}` becomes
    the next `\boldsymbol{x}`. The estimate is the largest ratio found, and the
    iterations stop when a ratio exceeds the previous estimate by at most
    ``rel_tol`` times the ratio, or after ``max_iter`` iterations. For a
    symmetric positive definite `\boldsymbol{K}_{C0}` and a symmetric `\boldsymbol{K}_G`,
    `\boldsymbol{K}_{C0}^{-1} \boldsymbol{K}_G` is self-adjoint in the inner product of
    `\boldsymbol{K}_{C0}`, such that the ratio tends to its spectral radius
    `\max |\mu|`. At the first iteration the relative residual of the linear
    solution must not exceed ``1e-6``.

    The fallback value ``1.`` is returned when `\boldsymbol{K}_{C0}` is singular or not
    positive definite, or when the linear solution is inaccurate, since in
    those cases the power iteration says nothing.

    """
    mu_max = _estimate_mu_max(K, KG, max_iter=max_iter, rel_tol=rel_tol)
    if mu_max is None:
        return 1.
    return safety*mu_max


def _negative_pivots(A):
    r"""Number of negative eigenvalues of a symmetric sparse matrix

    Parameters
    ----------
    A : sparse_matrix
        Symmetric matrix, shape ``(n, n)``. The symmetry is not checked.

    Returns
    -------
    num : int or None
        The number of negative pivots, ``None`` when it is unknown: ``A`` is
        exactly singular, a zero pivot forced a row interchange, or a pivot
        is null or not finite.

    Notes
    -----
    ``A`` is factorized with :func:`scipy.sparse.linalg.splu`, with the
    ordering ``'MMD_AT_PLUS_A'``, ``diag_pivot_thresh=0.`` and the symmetric
    mode, i.e. with a symmetric permutation and without row interchanges,
    `\boldsymbol{\Pi} \boldsymbol{A} \boldsymbol{\Pi}^\top = \boldsymbol{L} \boldsymbol{D} \boldsymbol{L}^\top` with the pivots `\boldsymbol{D}` on the diagonal of
    `\boldsymbol{U} = \boldsymbol{D} \boldsymbol{L}^\top`. By Sylvester's law of inertia, the number of negative
    pivots is the number of negative eigenvalues of `\boldsymbol{A}`.

    """
    return _symmetric_lu(A)[1]


def _symmetric_lu(A):
    r"""Factorization of a symmetric sparse matrix without row interchanges
    and its number of negative pivots

    Parameters
    ----------
    A : sparse_matrix
        Symmetric matrix, shape ``(n, n)``. The symmetry is not checked.

    Returns
    -------
    lu : SuperLU or None
        The factorization of :func:`_negative_pivots`, ``None`` when
        :func:`scipy.sparse.linalg.splu` found ``A`` exactly singular.
    num : int or None
        The number of negative pivots, ``None`` when it is unknown, see
        :func:`_negative_pivots`.

    """
    try:
        lu = splu(csc_matrix(A), permc_spec='MMD_AT_PLUS_A',
                  diag_pivot_thresh=0., options=dict(SymmetricMode=True))
    except RuntimeError:
        # NOTE exactly singular
        return None, None
    if not np.array_equal(lu.perm_r, lu.perm_c):
        # NOTE a zero pivot forced a row interchange
        return lu, None
    d = lu.U.diagonal()
    if not np.all(np.isfinite(d)) or np.any(d == 0):
        return lu, None
    return lu, int(np.count_nonzero(d < 0))


def is_positive_definite(A):
    r"""Tells whether the symmetric sparse matrix ``A`` is positive definite

    ``A`` is factorized with a symmetric permutation and without row
    interchanges, and it is considered positive definite when all the pivots
    of the factorization are positive.

    The factorization uses :func:`scipy.sparse.linalg.splu` with the
    ordering ``'MMD_AT_PLUS_A'``, ``diag_pivot_thresh=0.`` and the symmetric
    mode. The result is ``False`` when the factorization fails because ``A``
    is exactly singular, when a row interchange was nevertheless made, or
    when a pivot is not positive. The symmetry of ``A`` is not checked.

    Parameters
    ----------
    A : sparse_matrix
        Symmetric matrix, shape ``(n, n)``.

    Returns
    -------
    positive_definite : bool
        ``True`` when all the pivots are positive.

    See Also
    --------
    check_eigenpairs

    Notes
    -----
    The Gaussian elimination of a positive definite matrix never meets a zero
    or a negative pivot, and by Sylvester's law of inertia a negative pivot
    means a negative eigenvalue: without row interchanges, the factorization
    of the symmetric matrix is `\boldsymbol{L} \boldsymbol{D}_U \boldsymbol{L}^\top`, with the pivots on the
    diagonal `\boldsymbol{D}_U` of `\boldsymbol{U}`, and `\boldsymbol{A}` has as many negative eigenvalues as
    `\boldsymbol{D}_U` has negative entries, see :func:`_negative_pivots`.

    """
    return _negative_pivots(A) == 0


def _frobenius(A):
    r"""Frobenius norm of a dense or sparse matrix"""
    data = A.data if hasattr(A, 'nnz') else np.asarray(A)
    return float(np.sqrt(np.sum(np.abs(data)**2)))


def _relative_residuals(A, B, lam, u, rigid_rtol=None):
    r"""Relative residuals of the eigenpairs of `(\boldsymbol{A} + \lambda \boldsymbol{B}) \boldsymbol{u} =
    \boldsymbol{0}`

    Parameters
    ----------
    A, B : sparse_matrix or np.ndarray
        Matrices, shape ``(n, n)``.
    lam : array_like
        Finite eigenvalues, real or complex, 1-D array of size ``k``.
    u : array_like
        Eigenvectors, shape ``(n, k)``.
    rigid_rtol : float or None, optional
        ``None`` gives the relative residual ``norm(A @ u + lam*(B @
        u))/(norm(A @ u) + abs(lam)*norm(B @ u))`` of every eigenpair.
        Otherwise the eigenpairs with ``norm(A @ u) + abs(lam)*norm(B @ u)
        <= rigid_rtol*(norm_F(A) + abs(lam)*norm_F(B))*norm(u)``, null up to
        the round-off, e.g. the rigid-body modes of :func:`natural_frequency`
        with ``rigid_rtol=RIGID_BODY_RTOL``, give ``norm(A @ u + lam*(B @
        u))/((norm_F(A) + abs(lam)*norm_F(B))*norm(u))``, ``norm_F`` being
        the Frobenius norm. Default ``None``.

    Returns
    -------
    residual : np.ndarray
        One relative residual per eigenpair, dimensionless, ``inf`` when the
        denominator is null.

    Notes
    -----
    The relative residual

    .. math::

        \frac{\|\boldsymbol{A} \boldsymbol{u} + \lambda \boldsymbol{B} \boldsymbol{u}\|}{\|\boldsymbol{A} \boldsymbol{u}\|
        + |\lambda| \, \|\boldsymbol{B} \boldsymbol{u}\|}

    compares the residual with the terms of the eigenvalue problem for the
    eigenvector itself. For an eigenvector in the null space of `\boldsymbol{A}` with a
    null eigenvalue, e.g. a rigid-body mode of :func:`natural_frequency`,
    both the residual and the denominator are round-off errors, and their
    ratio is meaningless, of the order of 1. With ``rigid_rtol`` `=
    \epsilon_r`, an eigenpair with

    .. math::

        \|\boldsymbol{A} \boldsymbol{u}\| + |\lambda| \, \|\boldsymbol{B} \boldsymbol{u}\| \le \epsilon_r
        (\|\boldsymbol{A}\|_F + |\lambda| \, \|\boldsymbol{B}\|_F) \|\boldsymbol{u}\|

    is null up to the round-off: its residual is relative to the
    matrix-norm scale, i.e. it is the normwise backward error
    `\|\boldsymbol{r}\|/((\|\boldsymbol{A}\|_F + |\lambda| \, \|\boldsymbol{B}\|_F) \|\boldsymbol{u}\|)`, with
    `\boldsymbol{r}` the residual vector, not larger than `\epsilon_r`. Every other
    eigenpair keeps the relative residual, such that the check of a genuine
    eigenpair is never loosened.

    A backward error at the round-off does not tell that a genuine
    eigenpair is accurate when `\|\boldsymbol{A}\|_F` is set by stiff springs, e.g. the
    penalty connections of multi-domain models: the eigenpairs that satisfy
    the connections give `\|\boldsymbol{A} \boldsymbol{u}\|` many orders of magnitude smaller
    than `\|\boldsymbol{A}\|_F \|\boldsymbol{u}\|`. For the lowest load multipliers of a
    blade-stiffened panel with penalty connections, `\|\boldsymbol{K}\|_F = 5.1 \cdot
    10^{12}`, :func:`linear_buckling` with ``precondition=False`` and
    ``check=False`` returns eigenpairs with relative residuals of 1 to 2 %
    and a lowest load multiplier 0.08 % too low, whose backward errors are
    `6.5 \cdot 10^{-14}` to `2.7 \cdot 10^{-13}` and whose `(\|\boldsymbol{K} \boldsymbol{u}\| +
    |\lambda| \, \|\boldsymbol{K}_G \boldsymbol{u}\|)/((\|\boldsymbol{K}\|_F + |\lambda| \, \|\boldsymbol{K}_G\|_F)
    \|\boldsymbol{u}\|)` are `6.5 \cdot 10^{-12}` to `1.6 \cdot 10^{-11}`. A floor
    `\epsilon_f (\|\boldsymbol{A}\|_F + |\lambda| \, \|\boldsymbol{B}\|_F)\|\boldsymbol{u}\|` of the
    denominator with `\epsilon_f = 10^{-8}` would accept them. The
    rigid-body modes of :func:`natural_frequency` give `(\|\boldsymbol{K} \boldsymbol{u}\| +
    |\omega_n^2| \, \|\boldsymbol{M} \boldsymbol{u}\|)/((\|\boldsymbol{K}\|_F + |\omega_n^2| \, \|\boldsymbol{M}\|_F)
    \|\boldsymbol{u}\|)` below the machine precision, with the sparse and the dense
    solvers, for the chains of springs of the tests, also when a penalty
    spring `10^6` or `10^8` times stiffer than the others connects two
    chains, hence ``RIGID_BODY_RTOL``.
    :func:`check_eigenpairs` uses the relative residual only: a finite load
    multiplier is never null. The eigenpairs of :func:`natural_frequency`
    whose relative residual is larger than ``check_rtol`` are also accepted
    when their backward error is at the round-off, see
    :func:`_backward_errors`.

    """
    Au = A @ u
    Bu = B @ u
    lam = np.asarray(lam)
    num = np.linalg.norm(Au + Bu*lam, axis=0)
    den = np.linalg.norm(Au, axis=0) + np.abs(lam)*np.linalg.norm(Bu, axis=0)
    if rigid_rtol is not None:
        scale = (_frobenius(A) + np.abs(lam)*_frobenius(B)
                 )*np.linalg.norm(u, axis=0)
        den = np.where(den <= rigid_rtol*scale, scale, den)
    residual = np.full(lam.shape, np.inf)
    np.divide(num, den, out=residual, where=den > 0)
    return residual


def _backward_errors(A, B, lam, u):
    r"""Normwise backward errors of the eigenpairs of `(\boldsymbol{A} + \lambda \boldsymbol{B})
    \boldsymbol{u} = \boldsymbol{0}`

    Parameters
    ----------
    A, B : sparse_matrix or np.ndarray
        Matrices, shape ``(n, n)``.
    lam : array_like
        Finite eigenvalues, real or complex, 1-D array of size ``k``.
    u : array_like
        Eigenvectors, shape ``(n, k)``.

    Returns
    -------
    backward_error : np.ndarray
        ``norm(A @ u + lam*(B @ u))`` divided by ``(norm_F(A) +
        abs(lam)*norm_F(B))*norm(u)`` for each eigenpair, ``norm_F`` being
        the Frobenius norm, dimensionless, ``inf`` when the denominator is
        null.

    Notes
    -----
    The normwise backward error of an approximate eigenpair `(\lambda,
    \boldsymbol{u})`

    .. math::

        \eta = \frac{\|\boldsymbol{A} \boldsymbol{u} + \lambda \boldsymbol{B} \boldsymbol{u}\|}{(\|\boldsymbol{A}\|_F + |\lambda|
        \, \|\boldsymbol{B}\|_F) \|\boldsymbol{u}\|}

    is the smallest `\epsilon` such that `(\lambda, \boldsymbol{u})` is an exact
    eigenpair of `(\boldsymbol{A} + \Delta \boldsymbol{A}) + \lambda (\boldsymbol{B} + \Delta \boldsymbol{B})` with
    `\|\Delta \boldsymbol{A}\| \le \epsilon \|\boldsymbol{A}\|_F` and `\|\Delta \boldsymbol{B}\| \le \epsilon
    \|\boldsymbol{B}\|_F` ([1]_, Theorem 1, Eq. (2.3), with the tolerances `\|\boldsymbol{A}\|_F`
    and `\|\boldsymbol{B}\|_F`). The perturbations that attain it are of rank one, so
    that their Frobenius norm is their 2-norm, and for symmetric matrices and
    a real `\lambda` the minimum is the same with symmetric perturbations
    ([1]_, Theorem 2).

    The relative residual of :func:`_relative_residuals` is `\eta` times
    `(\|\boldsymbol{A}\|_F + |\lambda| \, \|\boldsymbol{B}\|_F)\|\boldsymbol{u}\|/(\|\boldsymbol{A} \boldsymbol{u}\| + |\lambda|
    \, \|\boldsymbol{B} \boldsymbol{u}\|)`, a ratio that is large for an eigenvalue close to zero
    of a nearly singular `\boldsymbol{A}`, e.g. a natural frequency pre-stressed close
    to buckling, whose eigenvalue is then sensitive to the round-off of the
    entries of `\boldsymbol{A}`. :func:`natural_frequency` accepts an eigenpair whose
    relative residual is larger than ``check_rtol`` when `\eta \le`
    ``BACKWARD_ERROR_RTOL`` `= \epsilon`, the machine precision: the
    eigenpair is exact for matrices that differ from `\boldsymbol{K}` and `\boldsymbol{M}` by
    perturbations of the order of the round-off of their entries, and it is
    as accurate as the matrices allow; see :func:`_quadratic_backward_errors`
    for :func:`complex_frequency`.

    Measured with SciPy 1.17.1 and Intel MKL 2025.0.0 for the cantilever of
    ``tests/test_beamc_natural_freq_cantilever_pre_stress.py``, pre-stressed
    to 0.9999 of its buckling load, whose two lowest eigenvalues are
    `\omega_n^2 = 0.0895`: the relative residuals of these eigenpairs are
    `2.3 \cdot 10^{-6}` to `1.3 \cdot 10^{-5}`, larger than the default
    ``check_rtol=1e-6``, with ``eigsh``, ``eigs`` and the dense solvers,
    with and without the preconditioner, and `(\|\boldsymbol{K} \boldsymbol{u}\| + |\omega_n^2|
    \, \|\boldsymbol{M} \boldsymbol{u}\|)/((\|\boldsymbol{K}\|_F + |\omega_n^2| \, \|\boldsymbol{M}\|_F)\|\boldsymbol{u}\|) =
    5.0 \cdot 10^3 \epsilon`, while their backward errors are `0.011
    \epsilon` to `0.063 \epsilon`, i.e. ``BACKWARD_ERROR_RTOL`` is 16 times
    the largest one. The inaccurate eigenpairs of ``eigsh`` for the lowest load
    multipliers of the blade-stiffened panel with penalty connections of
    ``tests/data/blade_stiffened_penalty_*.npz``, see
    :func:`_relative_residuals`, and of six variants with other blade
    heights, numbers of terms and 1000 times the penalty constants, have
    relative residuals of `1.6 \cdot 10^{-3}` to `9.7 \cdot 10^{-2}` and
    backward errors of `14 \epsilon` to `1200 \epsilon`: the smallest one is
    14 times ``BACKWARD_ERROR_RTOL``, and they are rejected. The backward
    error of an inaccurate eigenpair is its relative residual times the
    ratio above, which decreases as the penalty constants increase; the
    smallest one, `14 \epsilon`, was measured with 1000 times the penalty
    constants of the panel of ``tests/data``.

    :func:`check_eigenpairs` keeps the relative residual only. For the
    cantilever above, with `\boldsymbol{K} = \boldsymbol{K}_{C0} + f \lambda_{cr} \boldsymbol{K}_G`, the load
    multipliers `(1 - f) \lambda_{cr}` close to zero of the dense solver
    have relative residuals from `4.7 \cdot 10^{-7}` for `f = 0.9999` to
    `5.1 \cdot 10^{-4}` for `f = 1 - 10^{-7}`, below the default
    ``check_rtol=1e-3`` of :func:`linear_buckling`. With a tolerance of
    `10^{-3}` the backward error would only accept eigenpairs whose ratio
    above is lower than `1000 \epsilon`, the range of the penalty
    connections with 1000 times the penalty constants, where the inaccurate
    eigenpairs above lie.

    References
    ----------
    .. [1] Tisseur, F., Backward error and condition of polynomial eigenvalue
       problems, Linear Algebra and its Applications, 309(1-3), 339-361,
       2000. https://doi.org/10.1016/S0024-3795(99)00063-4

    """
    Au = A @ u
    Bu = B @ u
    lam = np.asarray(lam)
    num = np.linalg.norm(Au + Bu*lam, axis=0)
    scale = (_frobenius(A) + np.abs(lam)*_frobenius(B)
             )*np.linalg.norm(u, axis=0)
    backward_error = np.full(lam.shape, np.inf)
    np.divide(num, scale, out=backward_error, where=scale > 0)
    return backward_error


def check_eigenpairs(K, KG, eigvals, eigvecs, rtol=1.e-3, min_rel_gap=1.e-4,
                     check_inertia=True, K_positive_definite=None):
    r"""Verify the eigenpairs of a linear buckling problem

    The eigenpairs, load multipliers and buckling modes, of the linear
    buckling problem of ``K`` and ``KG``, see :func:`linear_buckling`, are
    verified with two checks:

    - the relative residual of every eigenpair with a finite load multiplier
      ``lam`` and eigenvector ``u``, ``norm(K @ u + lam*(KG @ u))/(norm(K @
      u) + abs(lam)*norm(KG @ u))``, see :func:`_relative_residuals`, must
      not exceed ``rtol``;
    - with ``check_inertia=True`` and when ``K`` is positive definite, no
      positive load multiplier is missing below the highest positive one
      returned, ``lam_n``: the number of negative pivots of ``K + s*KG``,
      with ``s = lam_n*(1 - min_rel_gap)``, must be the number of returned
      load multipliers in ``(0, s)``. When this count is unknown, see
      :func:`_negative_pivots`, the weaker check that ``K + s*KG`` with ``s
      = lam_1*(1 - min_rel_gap)`` is positive definite, ``lam_1`` being the
      lowest positive load multiplier, is done.

    The eigenpairs may be real or complex; the inertia check ignores the
    complex load multipliers and requires symmetric matrices. The eigenpairs
    with a non-finite load multiplier are skipped. When no load multiplier
    is finite, the check fails at once. The inertia check is skipped when no
    load multiplier is positive or when ``K`` is not positive definite, see
    :func:`is_positive_definite`; missing negative load multipliers are not
    detected, see :func:`_check_both_signs`. Nothing is raised or printed;
    the result tells which check failed.

    Parameters
    ----------
    K, KG : sparse_matrix
        Constitutive and geometric stiffness matrices. Shape ``(n, n)``.
    eigvals : array_like
        Load multipliers, dimensionless, real or complex. 1-D array of size
        ``k``.
    eigvecs : array_like
        Eigenvectors, one per column. Shape ``(n, k)``.
    rtol : float, optional
        Largest acceptable relative residual, dimensionless. Default
        ``1.e-3``.
    min_rel_gap : float, optional
        Relative margin below the load multipliers at which the inertia is
        tested, dimensionless. Default ``1.e-4``.
    check_inertia : bool, optional
        Whether the inertia check is done, only valid for symmetric matrices.
        Default ``True``.
    K_positive_definite : bool, callable or None, optional
        Whether ``K`` is positive definite, or a function of ``K`` returning
        it, e.g. to verify several solutions with one factorization of
        ``K``. ``None`` uses :func:`is_positive_definite`. Default ``None``.

    Returns
    -------
    error : str or None
        Description of the check that failed, or ``None`` when all passed.

    See Also
    --------
    linear_buckling

    Notes
    -----
    The eigenpairs are those of `(\boldsymbol{K}_{C0} + \lambda \boldsymbol{K}_G) \boldsymbol{u} = \boldsymbol{0}`.
    The inertia check is the one that catches a solver that converged to an
    interior part of the spectrum, which is silent otherwise. For symmetric
    `\boldsymbol{K}_{C0}` and `\boldsymbol{K}_G` with a positive definite `\boldsymbol{K}_{C0}`,
    `\boldsymbol{K}_{C0} + s \boldsymbol{K}_G` is congruent to `\boldsymbol{I} + s
    \boldsymbol{K}_{C0}^{-1/2} \boldsymbol{K}_G \boldsymbol{K}_{C0}^{-1/2}`, whose eigenvalues are `1 + s
    \mu_i = 1 - s/\lambda_i`, with `\mu_i = -1/\lambda_i` the eigenvalues of
    `\boldsymbol{K}_G \boldsymbol{u} = \mu \boldsymbol{K}_{C0} \boldsymbol{u}`. For `s > 0`, `1 - s/\lambda_i < 0`
    exactly when `0 < \lambda_i < s`, hence the number of negative
    eigenvalues of `\boldsymbol{K}_{C0} + s \boldsymbol{K}_G`, equal to its number of negative
    pivots by Sylvester's law of inertia, is the number of load multipliers
    in `(0, s)`. This is the Sturm sequence check of the eigenvalue solvers
    of structural analysis ([1]_, Section 10.2.2, p. 849, Section 11.4.3,
    p. 943, and Section 11.6, p. 954), where the number of negative
    elements of `\boldsymbol{D}` in the factorization `\boldsymbol{L} \boldsymbol{D} \boldsymbol{L}^\top` of `\boldsymbol{K} - \mu
    \boldsymbol{M}` is the number of eigenvalues of the pencil `(\boldsymbol{K}, \boldsymbol{M})` smaller
    than `\mu`. When only one
    positive load multiplier was returned, none is expected in `(0, s)` and
    the weaker check is done. The gap ``min_rel_gap`` keeps `s` away from
    `\lambda_n`, where `\boldsymbol{K}_{C0} + s \boldsymbol{K}_G` is singular.

    References
    ----------
    .. [1] Bathe, K. J., Finite Element Procedures, 2nd ed., K. J. Bathe,
       Watertown, MA, 2014, ISBN 978-0-9790049-5-7.

    """
    eigvals = np.asarray(eigvals)
    eigvecs = np.asarray(eigvecs)
    finite = np.isfinite(eigvals)
    if not np.any(finite):
        return 'no finite load multiplier was found'
    lam = eigvals[finite]
    u = eigvecs[:, finite]
    residual = _relative_residuals(K, KG, lam, u)
    if not np.all(residual <= rtol):
        i = int(np.argmax(np.nan_to_num(residual, nan=np.inf)))
        return ('relative residual %.2e > %.1e for the load multiplier %r'
                % (residual[i], rtol, lam[i]))
    if not check_inertia:
        return None
    lam = lam[np.isreal(lam)].real
    positive = lam[lam > 0]
    if K_positive_definite is None:
        K_positive_definite = is_positive_definite
    if not positive.size:
        return None
    if not (K_positive_definite(K) if callable(K_positive_definite)
            else K_positive_definite):
        return None
    K = csr_matrix(K)
    KG = csr_matrix(KG)
    s = positive.max()*(1 - min_rel_gap)
    expected = int(np.count_nonzero(positive < s))
    count = _negative_pivots(K + s*KG) if expected else None
    if count is not None:
        if count > expected:
            return ('%d load multiplier(s) lower than %g are missing'
                    % (count - expected, s))
        if count < expected:
            return ('%d load multipliers lower than %g were found, but only '
                    '%d exist' % (expected, s, count))
        return None
    s = positive.min()*(1 - min_rel_gap)
    if not is_positive_definite(K + s*KG):
        return ('at least one load multiplier lower than %g is missing'
                % positive.min())
    return None


def _null_mu(mu, scale=None):
    r"""Tell which eigenvalues `\mu` are null

    Parameters
    ----------
    mu : array_like
        Eigenvalues of `\boldsymbol{K}_G \boldsymbol{u} = \mu \boldsymbol{K} \boldsymbol{u}`, real or complex.
    scale : float or None, optional
        Reference modulus; the largest finite `|\mu|` when ``None``. Default
        ``None``.

    Returns
    -------
    null : np.ndarray
        Boolean array, ``True`` where `|\mu| \le` ``NULL_MU_RTOL`` ``scale``.

    """
    absmu = np.abs(np.asarray(mu))
    finite = np.isfinite(absmu)
    if scale is None:
        scale = absmu[finite].max() if np.any(finite) else 0.
    return finite & (absmu <= NULL_MU_RTOL*scale)


def _null_mu_pencil(a, b, mu, vl, vr):
    r"""Tell which eigenvalues `\mu` of `[a]\boldsymbol{v} = \mu [b]\boldsymbol{v}` are null up
    to the round-off of a dense solver

    Parameters
    ----------
    a : np.ndarray
        Dense matrix, shape ``(n, n)``.
    b : np.ndarray or None
        Dense matrix, shape ``(n, n)``, or ``None`` for the identity.
    mu : np.ndarray
        Eigenvalues, shape ``(m,)``.
    vl, vr : np.ndarray
        Left and right eigenvectors, ``vl[:, i]`` and ``vr[:, i]`` those of
        ``mu[i]``, shape ``(n, m)``; ``vl = vr`` for symmetric matrices.

    Returns
    -------
    null : np.ndarray
        Boolean array, shape ``(m,)``, ``True`` where `\mu` is null for
        :func:`_null_mu` or where `|\mu| \le c\,\epsilon\,\|[a]\|_F\,\kappa`,
        with `c` = ``NULL_MU_PERTURBATION_FACTOR``, `\epsilon` the machine
        precision and `\kappa = \|\boldsymbol{y}\| \|\boldsymbol{x}\| / |\boldsymbol{y}^H [b] \boldsymbol{x}|`
        the condition number of `\mu`, `\boldsymbol{y}` and `\boldsymbol{x}` its left and
        right eigenvectors.

    Notes
    -----
    A backward stable solver, e.g. the QZ algorithm of LAPACK ``zggev``,
    returns the exact eigenvalues of `[a] + [\delta a]`, `[b] + [\delta b]`
    with `\|[\delta a]\| \approx \gamma \epsilon \|[a]\|`, `\gamma` of order
    one. Multiplying `([a] + [\delta a] - (\mu + \delta\mu)([b] + [\delta
    b]))(\boldsymbol{x} + \delta \boldsymbol{x}) = 0` by `\boldsymbol{y}^H`, with `\boldsymbol{y}^H ([a] - \mu
    [b]) = 0`, gives to first order `\delta\mu = \boldsymbol{y}^H ([\delta a] - \mu
    [\delta b]) \boldsymbol{x} / (\boldsymbol{y}^H [b] \boldsymbol{x})`: a null eigenvalue moves by up
    to `\gamma \epsilon \|[a]\| \kappa`.

    A null eigenvalue of a Jordan block of size `m`, e.g. `m = 2` for the
    nilpotent `\boldsymbol{K}^{-1} \boldsymbol{K}_G` of Ziegler's double pendulum, is split by a
    perturbation of size `\eta` into `m` eigenvalues with `\mu^m \propto
    \eta`, i.e. `|\mu| \approx (\gamma \epsilon)^{1/m} \|[a]\|`, and
    `d\mu/d\eta = \mu/(m \eta)`: their condition number gives `\epsilon
    \|[a]\| \kappa \approx |\mu|/(m \gamma)`, i.e. `|\mu|` is about `m
    \gamma` times the bound whatever `m`. Measured with ``zggev``: 0.14 for
    Ziegler's double pendulum, `|\mu| = 9.4 \cdot 10^{-10}`, and 0.005 to
    2.5 for Jordan blocks of size 2 to 4 in pencils of size 2 to 600,
    where the genuine eigenvalues down to `10^{-8} \max |\mu|` were at
    least 73 times their bound, except those within the modulus of the
    split null eigenvalues, which the round-off does not determine and
    which are null as well. A well-conditioned genuine eigenvalue is null
    only when `|\mu| \le 2.2 \cdot 10^{-15} \|[a]\|`, below the
    :func:`_null_mu` threshold when `\|[a]\|` is of the order of `\max
    |\mu|`. For symmetric matrices no Jordan block exists, and `\boldsymbol{x}^\top [b]
    \boldsymbol{x} = 1` of :func:`scipy.linalg.eigh` gives `\kappa = \|\boldsymbol{x}\|^2`.

    """
    mu = np.asarray(mu)
    null = _null_mu(mu)
    vl = np.asarray(vl)
    vr = np.asarray(vr)
    bvr = vr if b is None else np.asarray(b) @ vr
    den = np.abs(np.einsum('ij,ij->j', vl.conj(), bvr))
    num = np.linalg.norm(vl, axis=0)*np.linalg.norm(vr, axis=0)
    with np.errstate(divide='ignore', invalid='ignore'):
        kappa = np.where(den > 0, num/den, np.inf)
    bound = (NULL_MU_PERTURBATION_FACTOR*np.finfo(float).eps
             * np.linalg.norm(np.asarray(a))*kappa)
    return null | (np.isfinite(mu) & (np.abs(mu) <= bound))


def _lb_order(mu, scale=None):
    r"""Order of the load multipliers `\lambda = -1/\mu` returned by
    :func:`linear_buckling`

    Parameters
    ----------
    mu : array_like
        Eigenvalues of `\boldsymbol{K}_G \boldsymbol{u} = \mu \boldsymbol{K} \boldsymbol{u}`, real or complex.
    scale : float or None, optional
        Reference modulus of :func:`_null_mu`. Default ``None``.

    Returns
    -------
    order : np.ndarray
        Integer indices of ``mu``: the positive real `\lambda` first, in
        increasing order, then the negative real `\lambda` by increasing
        modulus, i.e. the critical load multipliers of the reversed load
        first, then the complex `\lambda` by increasing modulus, and at the
        end the null `\mu`, see :func:`_null_mu`, and the eigenvalues that
        are not finite.

    """
    mu = np.asarray(mu)
    null = _null_mu(mu, scale)
    finite = np.isfinite(mu)
    real = np.isreal(mu) & finite & ~null
    re = np.real(mu)
    neg = np.flatnonzero(real & (re < 0))
    neg = neg[np.argsort(re[neg], kind='stable')]
    pos = np.flatnonzero(real & (re > 0))
    pos = pos[np.argsort(-re[pos], kind='stable')]
    cplx = np.flatnonzero(~np.isreal(mu) & finite & ~null)
    cplx = cplx[np.argsort(-np.abs(mu[cplx]), kind='stable')]
    first = np.concatenate((neg, pos, cplx))
    rest = np.setdiff1d(np.arange(mu.size), first)
    return np.concatenate((first, rest)).astype(int)


def _sort_eigenpairs(mu, eigvecs, scale=None):
    r"""Eigenpairs of `\boldsymbol{K}_G \boldsymbol{u} = \mu \boldsymbol{K} \boldsymbol{u}` in the order of
    :func:`_lb_order`, returned as ``(mu, eigvecs)``"""
    order = _lb_order(mu, scale)
    return mu[order], eigvecs[:, order]


def _eigsh_cayley(K, KG, k, tol, sigma):
    r"""Eigenpairs with :func:`scipy.sparse.linalg.eigsh` in Cayley mode

    Parameters
    ----------
    K, KG : sparse_matrix
        Symmetric matrices, ``K`` positive definite, shape ``(n, n)``.
    k : int
        Number of eigenpairs, at most ``n - 2``.
    tol : float
        Tolerance of :func:`scipy.sparse.linalg.eigsh`.
    sigma : float
        Shift of the Cayley mode, see :func:`estimate_cayley_sigma`.

    Returns
    -------
    mu, eigvecs : np.ndarray
        The ``k`` eigenvalues `\mu` of `\boldsymbol{K}_G \boldsymbol{u} = \mu \boldsymbol{K} \boldsymbol{u}` selected
        by the Cayley mode and their eigenvectors, sorted by
        :func:`_sort_eigenpairs`. :func:`.capped_eigsh` keeps ARPACK's
        ``ncv`` within the cap of the faulty Intel MKL versions, see
        :mod:`pyfe3d.solver.arpackutils`, and its fixed starting vector
        makes the results reproducible.

    Notes
    -----
    In the Cayley mode, ARPACK computes the eigenvalues `(\mu + \sigma)/(\mu
    - \sigma)` of the operator `(\boldsymbol{K}_G - \sigma \boldsymbol{K})^{-1}(\boldsymbol{K}_G + \sigma
    \boldsymbol{K})` ([1]_, Appendix A, Section A.1, "Cayley Transformation Mode", with
    `\boldsymbol{A} = \boldsymbol{K}_G` and `\boldsymbol{M} = \boldsymbol{K}`), and ``which='SM'`` selects those of
    smallest modulus. When `\sigma` is larger than `|\mu|` of all the
    eigenvalues, `(\mu + \sigma)/(\mu - \sigma)` increases in modulus with
    `\mu` in `(-\sigma, \sigma)`, such that the most negative
    `\mu`, i.e. the lowest positive load multipliers, are selected first,
    then the null and the positive `\mu`.

    References
    ----------
    .. [1] Lehoucq, R. B., Sorensen, D. C. and Yang, C., ARPACK Users'
       Guide: Solution of Large-Scale Eigenvalue Problems with Implicitly
       Restarted Arnoldi Methods, SIAM, Philadelphia, 1998.
       https://doi.org/10.1137/1.9780898719628

    """
    mu, eigvecs = capped_eigsh(A=KG, k=k, M=K, sigma=sigma, which='SM',
                               mode='cayley', tol=tol)
    return _sort_eigenpairs(mu, eigvecs)


def _count_multipliers(K, KG, mu_max, K_positive_definite):
    r"""Number of positive load multipliers with a modulus lower than
    ``INERTIA_COUNT_FACTOR/mu_max``

    Parameters
    ----------
    K, KG : sparse_matrix
        Symmetric matrices, shape ``(n, n)``; ``-KG`` counts the negative
        load multipliers.
    mu_max : float or None
        Estimate of the largest `|\mu|`, see :func:`_estimate_mu_max`.
    K_positive_definite : callable
        Function of ``K`` telling whether it is positive definite.

    Returns
    -------
    num : int or None
        The count, ``None`` when ``mu_max`` is unknown, ``K`` is not
        positive definite, or the inertia is unknown, see
        :func:`_negative_pivots`.

    Notes
    -----
    By Sylvester's law of inertia, for a positive definite `\boldsymbol{K}`, the number
    of negative pivots of `\boldsymbol{K} + s \boldsymbol{K}_G` is the number of load multipliers
    in `(0, s)` for `s > 0`, see :func:`check_eigenpairs`.

    """
    if mu_max is None or not K_positive_definite(K):
        return None
    s = INERTIA_COUNT_FACTOR/mu_max
    return _negative_pivots(csr_matrix(K) + s*csr_matrix(KG))


def _eigsh_both_signs(K, KG, k, tol, sigma, mu_max, K_positive_definite):
    r"""The lowest positive load multipliers, completed by the negative ones
    closest to zero, with :func:`_eigsh_cayley`

    Parameters
    ----------
    K, KG : sparse_matrix
        Symmetric matrices, ``K`` positive definite, shape ``(n, n)``.
    k : int
        Number of eigenpairs, at most ``n - 2``.
    tol : float
        Tolerance of :func:`scipy.sparse.linalg.eigsh`.
    sigma : float
        Shift of the Cayley mode.
    mu_max : float or None
        Estimate of the largest `|\mu|`, the scale of the null `\mu`, see
        :func:`_null_mu`, and of the counts of :func:`_count_multipliers`.
    K_positive_definite : callable
        Function of ``K`` telling whether it is positive definite.

    Returns
    -------
    mu, eigvecs : np.ndarray
        The eigenpairs of at most ``k`` finite load multipliers, in the order
        of :func:`_sort_eigenpairs`.

    Raises
    ------
    RuntimeError
        When no finite load multiplier was found.

    Notes
    -----
    The Cayley mode with a shift larger than all the `|\mu|` selects all
    the negative `\mu` before the null and the positive ones, see
    :func:`_eigsh_cayley`. ARPACK may not converge when fewer negative
    `\mu` exist than the number of requested eigenvalues, the remaining
    ones being then in the cluster of the null `\mu`, e.g. for a tensile
    reference load. The number of requested eigenvalues of each sign is
    therefore limited by the number of load multipliers of that sign,
    counted with the inertia of `\boldsymbol{K} + s \boldsymbol{K}_G` below the modulus
    ``INERTIA_COUNT_FACTOR`` divided by the largest `|\mu|`, see
    :func:`_count_multipliers`:

    1. ``KG`` gives the positive load multipliers `\lambda = -1/\mu`,
       `\mu < 0`, at most as many as counted. The null `\mu`, see
       :func:`_null_mu`, and the positive `\mu` are discarded.
    2. When fewer than ``k`` positive load multipliers were found, ``-KG``,
       i.e. the reversed load, gives the remaining ones, the negative load
       multipliers closest to zero, at most as many as counted.

    When the counts are unknown, e.g. ``K`` is not positive definite, ``k``
    eigenvalues are requested. A tensile reference load, without any
    positive load multiplier, thus gives the negative load multipliers
    closest to zero without a dense solution.

    """
    num_pos = _count_multipliers(K, KG, mu_max, K_positive_definite)
    mu = np.zeros(0)
    eigvecs = np.zeros((K.shape[0], 0))
    if num_pos != 0:
        k1 = k if num_pos is None else min(k, num_pos)
        mu, eigvecs = _eigsh_cayley(K, KG, k1, tol, sigma)
        keep = (mu < 0) & ~_null_mu(mu, mu_max)
        mu, eigvecs = mu[keep], eigvecs[:, keep]
        if mu.size >= k:
            return mu[:k], eigvecs[:, :k]
    # NOTE the reversed load
    num_neg = _count_multipliers(K, -KG, mu_max, K_positive_definite)
    if num_neg != 0:
        k2 = k - mu.size
        if num_neg is not None:
            k2 = min(k2, num_neg)
        mu2, eigvecs2 = _eigsh_cayley(K, -KG, k2, tol, sigma)
        keep2 = (mu2 < 0) & ~_null_mu(mu2, mu_max)
        mu = np.concatenate((mu, -mu2[keep2]))
        eigvecs = np.concatenate((eigvecs, eigvecs2[:, keep2]), axis=1)
    if mu.size == 0:
        raise RuntimeError('no finite load multiplier was found')
    return mu, eigvecs


def _check_both_signs(K, KG, mu, eigvecs, rtol, K_positive_definite=None):
    r"""Verify the eigenpairs of both signs of :func:`linear_buckling`

    The positive load multipliers are verified with :func:`check_eigenpairs`
    and ``KG``, the negative ones with ``-KG``, i.e. the reversed load, such
    that the inertia check also tells that no negative load multiplier closer
    to zero than the returned ones is missing. The null `\mu`, see
    :func:`_null_mu`, are not verified.

    Parameters
    ----------
    K, KG : sparse_matrix
        Symmetric matrices, shape ``(n, n)``.
    mu : array_like
        Eigenvalues of `\boldsymbol{K}_G \boldsymbol{u} = \mu \boldsymbol{K} \boldsymbol{u}`, 1-D array of size ``k``.
    eigvecs : array_like
        Eigenvectors, shape ``(n, k)``.
    rtol : float
        Largest acceptable relative residual, dimensionless.
    K_positive_definite : bool, callable or None, optional
        See :func:`check_eigenpairs`. Default ``None``.

    Returns
    -------
    error : str or None
        Description of the check that failed, or ``None``.

    """
    mu = np.asarray(mu)
    kept = np.isfinite(mu) & ~_null_mu(mu)
    if not np.any(kept):
        return 'no finite load multiplier was found'
    with np.errstate(divide='ignore', invalid='ignore'):
        lam = -1./mu
    positive = kept & (lam > 0)
    negative = kept & (lam < 0)
    if np.any(positive):
        error = check_eigenpairs(K, KG, lam[positive], eigvecs[:, positive],
                                 rtol, K_positive_definite=K_positive_definite)
        if error is not None:
            return error
    if np.any(negative):
        error = check_eigenpairs(K, -KG, -lam[negative],
                                 eigvecs[:, negative], rtol,
                                 K_positive_definite=K_positive_definite)
        if error is not None:
            return 'reversed load: ' + error
    return None


def _condense(K, KG, active):
    r"""Schur complement of the degrees-of-freedom where ``KG`` is null

    Parameters
    ----------
    K, KG : sparse_matrix
        Matrices, shape ``(n, n)``.
    active : np.ndarray
        Boolean mask of the degrees-of-freedom kept, `b`; the others, `a`,
        are condensed out.

    Returns
    -------
    S : np.ndarray
        Dense Schur complement `\boldsymbol{K}_{bb} - \boldsymbol{K}_{ba} \boldsymbol{K}_{aa}^{-1} \boldsymbol{K}_{ab}`.
    KGbb : np.ndarray
        Dense block of ``KG``.
    back : callable
        ``back(ub)`` returns the full eigenvectors, with `\boldsymbol{u}_a =
        -\boldsymbol{K}_{aa}^{-1} \boldsymbol{K}_{ab} \boldsymbol{u}_b`, for real or complex ``ub``.

    Raises
    ------
    ValueError
        When no degree-of-freedom is active, i.e. ``KG`` is null.

    """
    b = np.flatnonzero(active)
    a = np.flatnonzero(~active)
    if b.size == 0:
        raise ValueError('KG is a null matrix')
    K = csr_matrix(K)
    KGbb = csr_matrix(KG)[b, :][:, b].toarray()
    S = K[b, :][:, b].toarray()
    lu = None
    if a.size:
        Kab = csc_matrix(K[a, :][:, b])
        Kba = K[b, :][:, a]
        lu = splu(csc_matrix(K[a, :][:, a]))
        # NOTE blocks of right-hand sides with at most ~1e7 entries
        chunk = max(1, int(1e7)//a.size)
        for j0 in range(0, b.size, chunk):
            j1 = min(j0 + chunk, b.size)
            X = lu.solve(Kab[:, j0:j1].toarray())
            S[:, j0:j1] -= Kba @ X

    def back(ub):
        eigvecs = np.zeros((K.shape[0], ub.shape[1]), dtype=ub.dtype)
        eigvecs[b, :] = ub
        if a.size:
            eigvecs[a, :] = -_lu_solve(lu, np.asarray(Kab @ ub))
        return eigvecs

    return S, KGbb, back


def _eigh_condensed(K, KG, k):
    r"""Dense solution after condensing out the degrees-of-freedom where
    ``KG`` is null, for symmetric matrices

    Parameters
    ----------
    K, KG : sparse_matrix
        Symmetric matrices, shape ``(n, n)``.
    k : int
        Number of eigenpairs returned.

    Returns
    -------
    mu, eigvecs : np.ndarray
        The first ``k`` eigenvalues `\mu` of `\boldsymbol{K}_G \boldsymbol{u} = \mu \boldsymbol{K} \boldsymbol{u}` in
        the order of :func:`_sort_eigenpairs`, i.e. the positive load
        multipliers first, and the eigenvectors, shape ``(n, k)``.

    Notes
    -----
    With the degrees-of-freedom split in `b`, where `\boldsymbol{K}_G` has non-null
    rows, and `a`, where it does not, the static condensation of the `a`
    degrees-of-freedom gives

    .. math::

        (\boldsymbol{K}_{bb} - \boldsymbol{K}_{ba} \boldsymbol{K}_{aa}^{-1} \boldsymbol{K}_{ab}
         + \lambda \boldsymbol{K}_{G_{bb}}) \boldsymbol{u}_b = \boldsymbol{0}, \qquad
        \boldsymbol{u}_a = -\boldsymbol{K}_{aa}^{-1} \boldsymbol{K}_{ab} \boldsymbol{u}_b

    solved with :func:`scipy.linalg.eigh`, which requires a positive
    definite Schur complement. The null `\mu` of :func:`_null_mu_pencil`
    are set to 0.

    """
    active = np.asarray(abs(csr_matrix(KG)).sum(axis=1)).ravel() > 0
    S, KGbb, back = _condense(K, KG, active)
    KGbb = 0.5*(KGbb + KGbb.T)
    S = 0.5*(S + S.T)
    mu, ub = eigh(a=KGbb, b=S)
    mu[_null_mu_pencil(KGbb, S, mu, ub, ub)] = 0.
    mu, ub = _sort_eigenpairs(mu, ub)
    k = min(k, ub.shape[1])
    return mu[:k], back(ub[:, :k])


def _lu_solve(lu, b):
    r"""``lu.solve(b)`` for a real or complex right-hand side ``b``

    The SuperLU factorization of a real matrix rejects a complex ``b``, whose
    real and imaginary parts are then solved separately.

    """
    b = np.asarray(b)
    if np.iscomplexobj(b):
        return (lu.solve(np.ascontiguousarray(b.real))
                + 1j*lu.solve(np.ascontiguousarray(b.imag)))
    return lu.solve(np.ascontiguousarray(b))


def _eig_complex(a, b=None, left=False):
    r"""All the eigenpairs of `[a]\boldsymbol{v} = w [b]\boldsymbol{v}` with
    :func:`scipy.linalg.eig`, through the complex LAPACK drivers

    The matrices are converted to complex, such that ``zgeev`` or ``zggev``
    is used instead of ``dgeev`` or ``dggev``: with Intel MKL 2025.0.0 the
    real drivers crash the Python process when computing the eigenvectors
    of some matrices, which cannot be caught, e.g. with SciPy 1.17.1
    :func:`scipy.linalg.eig` corrupts the heap and the process exits with
    the code ``0xc0000374`` for the real 48 x 48 matrices of
    ``test_eig_complex_mkl_crash_regression`` in
    ``tests/test_solver_linear_buckling.py``. The complex drivers are not
    affected, at the cost of the complex arithmetic. In the browser,
    ``sys.platform == 'emscripten'`` (Pyodide), the real drivers are used.

    Parameters
    ----------
    a : array_like
        Square matrix, shape ``(n, n)``.
    b : array_like or None, optional
        Square matrix, shape ``(n, n)``, or ``None`` for the identity.
        Default ``None``.
    left : bool, optional
        Also return the left eigenvectors. Default ``False``.

    Returns
    -------
    w, v : np.ndarray
        Eigenvalues and eigenvectors of :func:`scipy.linalg.eig`, or ``w,
        vl, v`` with the left eigenvectors ``vl`` when ``left=True``.

    """
    if sys.platform != 'emscripten':
        a = np.asarray(a, dtype=complex)
        if b is not None:
            b = np.asarray(b, dtype=complex)
    return eig(a=a, b=b, left=left, right=True)


def _real_if_close(w, eigvecs, rtol=1.e-8):
    r"""Real eigenpairs when the imaginary parts are negligible

    Each eigenvector of an eigenvalue whose imaginary part is not larger
    than ``rtol`` times its modulus is rotated in the complex plane to make
    its largest component real, and its imaginary part is dropped, as the
    imaginary part of the eigenvalue. Complex arrays are returned.

    Parameters
    ----------
    w : array_like
        Eigenvalues, 1-D array of size ``k``.
    eigvecs : array_like
        Eigenvectors, shape ``(n, k)``.
    rtol : float, optional
        Relative tolerance, dimensionless. Default ``1.e-8``.

    Returns
    -------
    w, eigvecs : np.ndarray
        Complex arrays.

    """
    w = np.array(w, dtype=complex)
    eigvecs = np.array(eigvecs, dtype=complex)
    real = np.abs(w.imag) <= rtol*np.abs(w)
    w[real] = w[real].real
    for i in np.flatnonzero(real):
        v = eigvecs[:, i]
        j = np.argmax(np.abs(v))
        if v[j] != 0:
            v = v*(abs(v[j])/v[j])
        eigvecs[:, i] = v.real
    return w, eigvecs


def _general_eigenpairs(mu, eigvecs, k=None):
    r"""The first ``k`` finite eigenpairs of a general eigenvalue solver

    Parameters
    ----------
    mu : array_like
        Eigenvalues `\mu` of `\boldsymbol{K}_G \boldsymbol{u} = \mu \boldsymbol{K} \boldsymbol{u}`, real or complex.
    eigvecs : array_like
        Eigenvectors, one per column.
    k : int or None, optional
        Number of eigenpairs kept, all with ``None``. Default ``None``.

    Returns
    -------
    mu, eigvecs : np.ndarray
        The eigenpairs made real by :func:`_real_if_close` when possible,
        sorted with :func:`_sort_eigenpairs`. Real arrays are returned when
        the kept eigenpairs are real, even when discarded ones are complex.

    """
    mu = np.asarray(mu)
    finite = np.isfinite(mu)
    mu, eigvecs = _real_if_close(mu[finite], np.asarray(eigvecs)[:, finite])
    mu, eigvecs = _sort_eigenpairs(mu, eigvecs)
    if k is not None:
        mu = mu[:k]
        eigvecs = eigvecs[:, :k]
    if np.all(mu.imag == 0):
        return mu.real, eigvecs.real
    return mu, eigvecs


def _eig_condensed(K, KG, k):
    r"""As :func:`_eigh_condensed`, for unsymmetric matrices

    The degrees-of-freedom where both the row and the column of ``KG`` are
    null are condensed out with an LU factorization of ``K``, and the
    condensed problem is solved with :func:`scipy.linalg.eig` through
    :func:`_eig_complex`.

    Parameters
    ----------
    K, KG : sparse_matrix
        Matrices, shape ``(n, n)``.
    k : int
        Number of eigenpairs returned.

    Returns
    -------
    mu, eigvecs : np.ndarray
        The ``k`` eigenpairs of :func:`_general_eigenpairs`, real when they
        are all real; the null `\mu` of :func:`_null_mu_pencil`, e.g. the
        round-off splitting of a defective null eigenvalue, are set to 0.

    """
    KG = csr_matrix(KG)
    active = ((np.asarray(abs(KG).sum(axis=1)).ravel() > 0)
              | (np.asarray(abs(KG).sum(axis=0)).ravel() > 0))
    S, KGbb, back = _condense(K, KG, active)
    mu, vl, ub = _eig_complex(a=KGbb, b=S, left=True)
    mu[_null_mu_pencil(KGbb, S, mu, vl, ub)] = 0.
    mu, ub = _general_eigenpairs(mu, ub, k)
    return mu, back(ub)


def _splu(A):
    r"""SuperLU factorization of ``A`` with partial pivoting, raising a
    ``RuntimeError`` when ``A`` is exactly singular

    :func:`scipy.sparse.linalg.splu` warns only when it converts its input
    to the CSC format, :class:`scipy.sparse.SparseEfficiencyWarning`, which
    the conversion here avoids. As everywhere in :mod:`pyfe3d.solver`, no
    :func:`warnings.catch_warnings` block is used: its exit resets the
    registry of the warnings already shown, of every module, and the
    warnings of the caller shown once per location would be shown again
    after each call.

    """
    return splu(csc_matrix(A))


def _permutation_parity(p):
    r"""``1`` for an even permutation ``p``, ``-1`` for an odd one"""
    p = np.asarray(p)
    seen = np.zeros(p.size, dtype=bool)
    odd = False
    for i in range(p.size):
        if seen[i]:
            continue
        j = i
        length = 0
        while not seen[j]:
            seen[j] = True
            j = p[j]
            length += 1
        odd ^= bool((length - 1) % 2)
    return -1 if odd else 1


def _det_sign(lu):
    r"""Sign of the determinant of a real matrix from its SuperLU
    factorization `\boldsymbol{\Pi}_r \boldsymbol{A} \boldsymbol{\Pi}_c = \boldsymbol{L} \boldsymbol{U}`, with a unit `\boldsymbol{L}`"""
    d = lu.U.diagonal()
    return (int(np.prod(np.sign(d))) * _permutation_parity(lu.perm_r)
            * _permutation_parity(lu.perm_c))


def _norm_estimate(lu, KG, v):
    r"""Estimate from below of `\|\boldsymbol{A}^{-1} \boldsymbol{K}_G\|_2`, with `\boldsymbol{A}` the matrix
    factorized in ``lu``

    Parameters
    ----------
    lu : SuperLU
        Factorization of `\boldsymbol{A}`.
    KG : sparse_matrix
        Matrix `\boldsymbol{K}_G`, shape ``(n, n)``.
    v : np.ndarray
        Starting vector, shape ``(n,)``.

    Returns
    -------
    norm : float
        `\|\boldsymbol{O}^H \boldsymbol{O} \boldsymbol{v}\|/\|\boldsymbol{O} \boldsymbol{v}\|`, with `\boldsymbol{O} = \boldsymbol{A}^{-1} \boldsymbol{K}_G`, or
        ``0.`` when `\boldsymbol{O} \boldsymbol{v} = 0`.

    Notes
    -----
    One step of the power iteration of `\boldsymbol{O}^H \boldsymbol{O}`, whose largest
    eigenvalue is `\|\boldsymbol{O}\|_2^2`, with two solutions with ``lu``. The
    estimate is not larger than `\|\boldsymbol{O}\|_2` and, by the Cauchy-Schwarz
    inequality `\|\boldsymbol{O} \boldsymbol{v}\|^2 = \boldsymbol{v}^H \boldsymbol{O}^H \boldsymbol{O} \boldsymbol{v} \le \|\boldsymbol{v}\|
    \|\boldsymbol{O}^H \boldsymbol{O} \boldsymbol{v}\|`, not smaller than `\|\boldsymbol{O} \boldsymbol{v}\|/\|\boldsymbol{v}\|`. The
    vector `\boldsymbol{O} \boldsymbol{v}` is in the range of `\boldsymbol{O}`, i.e. of `\boldsymbol{A}^{-1} \boldsymbol{K}_G`,
    such that the estimate does not decrease with the number of
    degrees-of-freedom where `\boldsymbol{K}_G` is null, as `\|\boldsymbol{O} \boldsymbol{v}\|/\|\boldsymbol{v}\|`
    does.

    """
    y = _lu_solve(lu, KG @ v)
    norm_y = np.linalg.norm(y)
    if norm_y == 0:
        return 0.
    z = KG.conj().T @ lu.solve(np.ascontiguousarray(y), trans='H')
    return float(np.linalg.norm(z)/norm_y)


def _eigs_shifted(K, KG, s, k, tol, lu):
    r"""The ``k`` load multipliers nearest the shift ``s``, for real
    unsymmetric matrices

    Parameters
    ----------
    K, KG : sparse_matrix
        Matrices, shape ``(n, n)``.
    s : float
        Shift, a load multiplier, dimensionless.
    k : int
        Number of eigenpairs, at most ``n - 2``.
    tol : float
        Tolerance of :func:`scipy.sparse.linalg.eigs`.
    lu : SuperLU
        Factorization of `\boldsymbol{K} + s \boldsymbol{K}_G`.

    Returns
    -------
    mu, eigvecs : np.ndarray
        The eigenpairs, with `\mu = -1/\lambda`, sorted by
        :func:`_general_eigenpairs`, empty when all the eigenvalues found
        are null. The eigenvalues `\nu = -1/(\lambda - s)` with `|\nu| \le`
        ``NULL_MU_RTOL`` times the largest of `\max |\nu|` and the estimate
        of :func:`_norm_estimate` of the norm of `(\boldsymbol{K} + s \boldsymbol{K}_G)^{-1}
        \boldsymbol{K}_G` from the starting vector of ARPACK are null, i.e. infinite
        load multipliers, and are not returned.
    radius : float
        `\max |\lambda - s|` of the returned load multipliers: all the load
        multipliers in the disc `|\lambda - s| <` ``radius`` are returned.
        ``inf`` when a null `\nu` was found, i.e. all the finite load
        multipliers were found, none when all the `\nu` are null.

    Notes
    -----
    With `\lambda = s + \delta`, `(\boldsymbol{K} + s \boldsymbol{K}_G + \delta \boldsymbol{K}_G) \boldsymbol{u} = 0`
    gives the standard problem `\boldsymbol{O} \boldsymbol{u} = \nu \boldsymbol{u}`, `\boldsymbol{O} = (\boldsymbol{K} + s
    \boldsymbol{K}_G)^{-1} \boldsymbol{K}_G`, `\nu = -1/(\lambda - s)`, whose ``k`` eigenvalues
    of largest `|\nu|` are found with :func:`scipy.sparse.linalg.eigs`.
    Then `\mu = \nu/(1 - s \nu)`. The eigenvalues `|\nu| \le \epsilon_0
    \max(\max |\nu|, \|\boldsymbol{O}\|)`, with `\epsilon_0` = ``NULL_MU_RTOL``, are
    taken as null, e.g. those of the null rows of `\boldsymbol{K}_G`.

    The null `\nu` are found as round-off: each product with `\boldsymbol{O}` is
    computed with an error of the order of the machine precision times
    `\|\boldsymbol{O}\|`, and a null `\nu` comes out with a modulus up to this order,
    or larger for a defective null eigenvalue. When `\boldsymbol{O}` has non-null
    eigenvalues, `\max |\nu|` is a scale of the round-off; when it has
    none, e.g. when `\boldsymbol{K}^{-1} \boldsymbol{K}_G` is nilpotent as with the follower load
    of Ziegler's double pendulum, all the `\nu` are round-off, e.g. about
    `10^{-33}` or `10^{-17}` instead of 0 depending on the restarts of
    ARPACK, and `\max |\nu|` is not a scale: the largest `\nu` would not be
    null and their spurious load multipliers, about `10^{33}` or
    `10^{17}`, would be returned. The norm `\|\boldsymbol{O}\|` is a scale that does
    not depend on the eigenvalues. The estimate is not larger than `\max
    |\nu|` for a normal `\boldsymbol{O}`, whose largest `|\nu|` is its norm, and both
    scales then give the same null `\nu`.

    """
    n = K.shape[0]
    OP = LinearOperator((n, n), matvec=lambda x: _lu_solve(lu, KG @ x),
                        dtype=np.result_type(K.dtype, KG.dtype))
    kwargs = start_kwargs(n)
    nu, eigvecs = eigs(OP, k=k, which='LM', tol=tol, **kwargs)
    scale = max(float(np.abs(nu).max()), _norm_estimate(lu, KG, kwargs['v0']))
    null = np.abs(nu) <= NULL_MU_RTOL*scale
    if np.any(null):
        radius = np.inf
        nu = nu[~null]
        eigvecs = eigvecs[:, ~null]
    else:
        radius = float(np.max(1./np.abs(nu)))
    with np.errstate(divide='ignore', invalid='ignore'):
        mu = nu/(1 - s*nu)
    mu, eigvecs = _general_eigenpairs(mu, eigvecs)
    return mu, eigvecs, radius


def _covered(discs):
    r"""Largest `c` such that `(0, c)` is inside the union of the intervals
    `(s - r, s + r)` of the discs ``(s, r)`` on the real axis"""
    c = 0.
    for left, right in sorted((s - r, s + r) for s, r in discs):
        if left > c:
            break
        c = max(c, right)
    return c


def _eigs_shift_search(K, KG, k, tol, check_rtol):
    r"""Lowest positive real load multiplier of unsymmetric matrices, with a
    search of shifts along the positive real axis

    Parameters
    ----------
    K, KG : sparse_matrix
        Real matrices, shape ``(n, n)``.
    k : int
        Number of eigenpairs per shift, at least
        ``SHIFT_SEARCH_MIN_EIGVALUES`` and at most ``n - 2``.
    tol : float
        Tolerance of :func:`scipy.sparse.linalg.eigs`.
    check_rtol : float
        Largest relative residual of the eigenpairs of each shift, see
        :func:`check_eigenpairs` with ``check_inertia=False``.

    Returns
    -------
    mu, eigvecs : np.ndarray
        The eigenpairs of the step that found `\lambda_{cr}`, i.e. the ``k``
        multipliers nearest its shift, sorted by :func:`_general_eigenpairs`
        with `\lambda_{cr}` first.

    Raises
    ------
    RuntimeError
        When ``K`` is singular, a verification failed, no positive real load
        multiplier was found, or the parity check failed.

    Notes
    -----
    :func:`scipy.sparse.linalg.eigs` finds the load multipliers nearest a
    shift, see :func:`_eigs_shifted`, but with follower loads many negative
    or complex multipliers may have a smaller modulus than the critical one.
    The search therefore moves the shift `s` along the positive real axis:

    1. `s_0 = 0`: the ``k`` multipliers nearest `s_0`, within the radius
       `r_0`, are found with :func:`_eigs_shifted`. The residual of every
       step is verified.
    2. Each step `j` covers the interval `(s_j - r_j, s_j + r_j)` of the real
       axis, inside which all the multipliers were found. With `c` the end of
       the union of these intervals starting at 0, i.e. `(0, c)` is covered,
       the next shift is `s_{j+1} = c + r_j`. The shift is perturbed when
       `\boldsymbol{K} + s \boldsymbol{K}_G` is exactly singular.
    3. The search stops when the lowest positive real multiplier found,
       `\lambda_{cr}`, is not larger than `c`: no multiplier of `(0,
       \lambda_{cr})` was missed. It fails after
       ``SHIFT_SEARCH_MAX_STEPS`` steps, or when the shift exceeds
       ``SHIFT_SEARCH_MAX_GROWTH`` times `r_0`.
    4. Parity check: `\det(\boldsymbol{K} + (1 - \epsilon_p) \lambda_{cr} \boldsymbol{K}_G)`, with
       `\epsilon_p` = ``PARITY_REL_GAP``, must have the sign of `\det \boldsymbol{K}`,
       i.e. an even number, normally zero, of real multipliers in `(0, (1 -
       \epsilon_p) \lambda_{cr})`. The signs come from the SuperLU
       factorizations, see :func:`_det_sign`. This check cannot detect two
       missing real multipliers, nor a missing complex pair.

    The parity check follows from `\det(\boldsymbol{K} + s \boldsymbol{K}_G) = \det \boldsymbol{K}
    \prod_i (1 + s \mu_i)`, with `\mu_i = -1/\lambda_i` the eigenvalues of
    `\boldsymbol{K}^{-1} \boldsymbol{K}_G`: a complex conjugate pair gives the positive factor
    `|1 + s \mu_i|^2`, a null `\mu_i` the factor 1, and a real `\mu_i` a
    negative factor exactly when `0 < \lambda_i < s`.

    """
    K = csc_matrix(K)
    KG = csr_matrix(KG)
    n = K.shape[0]
    k = min(max(k, SHIFT_SEARCH_MIN_EIGVALUES), n - 2)
    if k < 1:
        raise ValueError('eigs() requires at least 3 degrees-of-freedom')
    try:
        lu0 = _splu(K)
    except RuntimeError:
        raise RuntimeError('K is singular')
    discs = []
    best = None
    s = 0.
    lu = lu0
    for step in range(SHIFT_SEARCH_MAX_STEPS):
        mu, eigvecs, radius = _eigs_shifted(K, KG, s, k, tol, lu)
        with np.errstate(divide='ignore', invalid='ignore'):
            lam = -1./mu
        error = check_eigenpairs(K, KG, lam, eigvecs, check_rtol,
                                 check_inertia=False)
        if error is not None:
            raise RuntimeError('shift %g: %s' % (s, error))
        discs.append((s, radius))
        real = np.isfinite(lam) & (np.imag(lam) == 0)
        positive = np.real(lam[real])
        positive = positive[positive > 0]
        if positive.size and (best is None or positive.min() < best[0]):
            best = (positive.min(), mu, eigvecs)
        covered = _covered(discs)
        if best is not None and best[0] <= covered:
            break
        if not np.isfinite(covered):
            raise RuntimeError('no positive real load multiplier exists, all '
                               'the finite ones were found')
        if step == 0:
            r0 = radius
        s_next = covered + radius
        if s_next > SHIFT_SEARCH_MAX_GROWTH*r0:
            raise RuntimeError('no positive real load multiplier was found '
                               'below %g' % covered)
        for perturbation in (0., 1.e-6, 1.e-3):
            s = s_next + perturbation*radius
            try:
                lu = _splu(K + s*KG)
                break
            except RuntimeError:
                continue
        else:
            raise RuntimeError('K + s KG is singular for s=%g' % s)
    else:
        raise RuntimeError('no positive real load multiplier was found below '
                           '%g after %d shifts' % (covered,
                                                   SHIFT_SEARCH_MAX_STEPS))
    lam_cr, mu, eigvecs = best
    s = (1 - PARITY_REL_GAP)*lam_cr
    if _det_sign(_splu(K + s*KG)) != _det_sign(lu0):
        raise RuntimeError('parity check failed: an odd number of real load '
                           'multipliers lower than %g was missed' % lam_cr)
    return mu, eigvecs


def _complete_repeated(A, M, w, V, shift, tol, num_missing, wanted, k):
    r"""Complete the lowest eigenpairs of `\boldsymbol{A} \boldsymbol{x} = w \boldsymbol{M} \boldsymbol{x}` with
    deflated passes, e.g. the missing copies of repeated eigenvalues

    Parameters
    ----------
    A, M : sparse_matrix
        Symmetric matrices, shape ``(n, n)``, ``M`` positive definite on the
        eigenvectors, ``A - shift*M`` non-singular.
    w : np.ndarray
        Eigenvalues found, 1-D array of size ``p``, the lowest ones but for
        those missing.
    V : np.ndarray
        Their `\boldsymbol{M}`-orthonormal eigenvectors, shape ``(n, p)``.
    shift : float
        Shift of the shift-invert mode, lower than the eigenvalues sought,
        which are those nearest it.
    tol : float
        Tolerance of :func:`scipy.sparse.linalg.eigsh`.
    num_missing : callable
        ``num_missing(w)`` returns the number of eigenvalues missing below
        the ``k``-th lowest one of ``w``, counted with an inertia, ``0``
        when none is missing or when the count is unknown, see
        :func:`_missing_counter`.
    wanted : callable
        ``wanted(w)`` returns the boolean mask of the eigenvalues of a pass
        that are kept.
    k : int
        Number of eigenpairs requested, the largest number computed by a
        pass.

    Returns
    -------
    w, V : np.ndarray
        The Ritz values by increasing value and the `\boldsymbol{M}`-orthonormal Ritz
        vectors of the eigenvectors found and of those added, see
        :func:`.arpackutils._rayleigh_ritz`.
    num_added : int
        Number of eigenpairs added.

    Notes
    -----
    The Lanczos process of ARPACK, started from a single vector, finds in
    exact arithmetic only one eigenvector of each eigenspace, and the other
    copies of a repeated eigenvalue only through the rounding errors,
    whereas the structures with a symmetry have repeated eigenvalues equal
    up to the round-off, e.g. the pairs of modes `\cos n\theta` and `\sin
    n\theta` of a cylinder with a rotationally symmetric mesh, see
    :func:`.deflated_eigsh`. Measured with SciPy 1.17.1 and Intel MKL
    2025.0.0 for the torsion of the cylinder Z33 of
    ``tests/test_quad4_linear_buckling_cylinder_Nxy.py`` with 80 elements
    around the circumference: ``eigsh()`` in Cayley mode with 4 load
    multipliers requested returns one copy of each of the four lowest
    pairs, 3 copies are missing, and with 2 or 3 requested 1 or 2. The
    inertia counts of the verification, see
    :func:`check_eigenpairs` and :func:`_check_freq`, detect the missing
    copies below the highest eigenvalue found. Each pass counts them, with
    the relative gap ``COMPLETION_REL_GAP`` below the ``k``-th lowest
    eigenvalue found, as the verification does below the highest one
    returned, and computes as many eigenpairs, at most ``k``, with
    :func:`.deflated_eigsh`, about ``shift``, with the eigenvectors already
    found deflated and the starting vector drawn with the seed
    ``COMPLETION_SEED`` plus the number of the pass, i.e. the eigenvalues
    nearest the shift not found yet, which are the missing ones. The
    eigenvectors found and added are combined with a Rayleigh-Ritz
    projection, which keeps them `\boldsymbol{M}`-orthonormal. At most
    ``MAX_COMPLETION_PASSES`` passes are made, each also limited to the cap
    of ``ncv`` of :func:`.arpack_max_ncv`, and the passes stop when no
    eigenvalue is missing or when a pass adds nothing. The result is
    verified as any other.

    """
    n = A.shape[0]
    lu = None
    num_added = 0
    for i in range(MAX_COMPLETION_PASSES):
        m = num_missing(w)
        if not m:
            break
        k_pass = min(m, k, n - V.shape[1] - 2)
        max_ncv = arpack_max_ncv()
        if max_ncv is not None:
            k_pass = min(k_pass, (max_ncv - 1)//2)
        if k_pass < 1:
            break
        if lu is None:
            lu = _splu(A - shift*M)
        v0 = np.random.RandomState(COMPLETION_SEED + i).uniform(-1., 1., n)
        wi, Vi = deflated_eigsh(A, k_pass, M, shift, V, tol=tol, lu=lu,
                                v0=v0)
        keep = wanted(wi)
        if not np.any(keep):
            break
        w, V = _rayleigh_ritz(A, M, np.hstack((V, Vi[:, keep])))
        num_added += int(np.count_nonzero(keep))
    return w, V, num_added


def _missing_counter(K, B, s_of, k):
    r"""``num_missing`` of :func:`_complete_repeated` from the inertia of
    `\boldsymbol{K} + s \boldsymbol{B}`

    Parameters
    ----------
    K, B : sparse_matrix
        Symmetric matrices, shape ``(n, n)``: ``B = KG`` or ``-KG`` for
        linear buckling, `\boldsymbol{K}` being positive definite, ``B = -M`` for
        natural frequency.
    s_of : callable
        ``s_of(w)`` returns the eigenvalues found as the values `s` of the
        inertia, the load multipliers `\lambda = -1/\mu` or the `\omega_n^2`,
        whose number below `s` is the number of negative pivots of `\boldsymbol{K} +
        s \boldsymbol{B}`.
    k : int
        Number of eigenvalues requested: the count is made below the
        ``k``-th lowest one found, the highest one that will be returned.

    Returns
    -------
    num_missing : callable
        See :func:`_complete_repeated`.

    """
    def num_missing(w):
        s_found = np.sort(s_of(w))
        if not s_found.size:
            return 0
        top = s_found[min(k, s_found.size) - 1]
        s = top - COMPLETION_REL_GAP*abs(top)
        if s == 0:
            return 0
        expected = int(np.count_nonzero(s_found < s))
        count = _negative_pivots(csr_matrix(K) + s*csr_matrix(B))
        if count is None or count <= expected:
            return 0
        return count - expected
    return num_missing


def _complete_both_signs(K, KG, mu, eigvecs, k, tol, sigma, mu_max):
    r"""Completion of the eigenpairs of :func:`_eigsh_both_signs` with
    :func:`_complete_repeated`, for each sign of the load multipliers

    Parameters
    ----------
    K, KG : sparse_matrix
        Symmetric matrices, ``K`` positive definite, shape ``(n, n)``.
    mu, eigvecs : np.ndarray
        Eigenpairs of :func:`_eigsh_both_signs`, the eigenvectors
        `\boldsymbol{K}`-orthonormal.
    k : int
        Number of eigenpairs.
    tol : float
        Tolerance of :func:`scipy.sparse.linalg.eigsh`.
    sigma : float
        Shift of the Cayley mode that found the eigenpairs; the deflated
        passes use the shift `-\sigma`, below all the `\mu`.
    mu_max : float or None
        Estimate of the largest `|\mu|`, the scale of the null `\mu`.

    Returns
    -------
    mu, eigvecs : np.ndarray
        At most ``k`` eigenpairs, in the order of :func:`_sort_eigenpairs`.
    num_added : int
        Number of eigenpairs added.

    Notes
    -----
    The positive load multipliers, `\mu < 0`, are completed with ``KG``,
    with the count of :func:`check_eigenpairs`, and, when fewer than ``k``
    of them exist, the negative ones, `\mu > 0`, with ``-KG``, i.e. the
    reversed load, as in :func:`_eigsh_both_signs`. With `\sigma` larger
    than all `|\mu|`, `\boldsymbol{K}_G + \sigma \boldsymbol{K}` is positive definite and the
    eigenvalues nearest `-\sigma` not found yet are the most negative ones,
    i.e. the lowest load multipliers missing.

    """
    mu = np.asarray(mu)
    kept = np.isfinite(mu) & ~_null_mu(mu, mu_max)
    parts_mu = []
    parts_vecs = []
    num_added = 0
    for sign in (1., -1.):
        sel = kept & (sign*mu < 0)
        if not np.any(sel) or sum(p.size for p in parts_mu) >= k:
            continue
        B = sign*csr_matrix(KG)
        num_missing = _missing_counter(K, B, lambda w: -1./w, k)
        w, vecs, added = _complete_repeated(
            B, K, sign*mu[sel], eigvecs[:, sel], -sigma, tol, num_missing,
            lambda w: (w < 0) & ~_null_mu(w, mu_max), k)
        parts_mu.append(sign*w)
        parts_vecs.append(vecs)
        num_added += added
    mu = np.concatenate(parts_mu) if parts_mu else np.zeros(0)
    eigvecs = (np.hstack(parts_vecs) if parts_vecs
               else np.zeros((K.shape[0], 0)))
    mu, eigvecs = _sort_eigenpairs(mu, eigvecs, mu_max)
    return mu[:k], eigvecs[:, :k], num_added


def _sparse_first(analysis, strategies, verify, dense, dense_size,
                  max_dense_size, hint='', diagnose=None):
    r"""Solver policy: the sparse solvers first, the dense solver as a last
    resort

    Parameters
    ----------
    analysis : str
        Name of the analysis, used in the messages.
    strategies : list
        The sparse strategies as ``(name, solver)`` pairs, tried in this
        order. ``solver()`` returns the eigenpairs as a tuple, or raises
        :class:`_NotApplicable` when the strategy does not apply, which is
        then skipped without being listed among the failures.
    verify : callable or None
        ``verify(*eigenpairs)`` returns ``None`` when the eigenpairs pass
        the verification, otherwise the description of the failure. With
        ``None`` the first strategy is returned without verification and
        without fallback.
    dense : tuple
        The dense solver, a ``(name, solver)`` pair, used when all the
        strategies failed and ``0 < dense_size <= max_dense_size``. A
        :class:`DenseFallbackWarning` is then issued with the reasons of the
        failures, and the dense eigenpairs are verified as well.
    dense_size, max_dense_size : int
        Size of the dense problem and its maximum.
    hint : str, optional
        Appended to the message of the ``RuntimeError``. Default ``''``.
    diagnose : callable or None, optional
        ``diagnose()``, called only when every solver failed, returns a
        description of the likely cause, put at the start of the message of
        the ``RuntimeError``, or ``None``; e.g. a stiffness matrix that is
        not positive definite, see :func:`natural_frequency`. Default
        ``None``.

    Returns
    -------
    eigenpairs : tuple
        The returned value of the solver that passed the verification.
    name : str
        The name of this solver.

    Raises
    ------
    RuntimeError
        When no solver passed the verification, listing every attempted
        solver and the reason of its failure.

    Notes
    -----
    The dense fallback keeps the analyses robust where ARPACK is
    unreliable, whereas normal runs never pay for a dense solution.

    """
    if verify is None:
        name, solver = strategies[0]
        return solver(), name
    errors = []

    def attempt(name, solver):
        try:
            result = solver()
            error = verify(*result)
        except (LinAlgError, RuntimeError, ValueError, TypeError,
                ArithmeticError) as e:
            result = None
            error = '%s: %s' % (type(e).__name__, e)
        if error is not None:
            errors.append('%s: %s' % (name, error))
        return result, error

    for name, solver in strategies:
        try:
            result, error = attempt(name, solver)
        except _NotApplicable:
            continue
        if error is None:
            return result, name
    name, solver = dense
    if max_dense_size <= 0:
        errors.append('%s: dense fallback disabled with max_dense_size=%d'
                      % (name, max_dense_size))
    elif dense_size > max_dense_size:
        errors.append('%s: dense fallback skipped, size %d > '
                      'max_dense_size=%d' % (name, dense_size,
                                             max_dense_size))
    else:
        text = ('%s: the sparse solvers failed (%s), using the dense solver '
                '%s of size %d as a last resort' % (analysis,
                    '; '.join(errors), name, dense_size))
        warnings.warn(text, DenseFallbackWarning, stacklevel=3)
        result, error = attempt(name, solver)
        if error is None:
            return result, name
    text = ('%s failed, no eigenvalue solver passed the verification. '
            'Attempted: %s.%s' % (analysis, '; '.join(errors), hint))
    diagnosis = diagnose() if diagnose is not None else None
    if diagnosis is not None:
        text = '%s. %s' % (diagnosis, text)
    raise RuntimeError(text)


def linear_buckling(K, KG, num_eigvalues=6, tol=0, sigma=None,
                    precondition=True, check=True, check_rtol=1.e-3,
                    symmetric=None, sparse_solver=True, max_dense_size=2000):
    r"""Linear buckling analysis with a verified spectrum

    Computes the load multipliers and the buckling modes of the linear
    buckling problem of ``K`` and ``KG``, the lowest positive load
    multipliers being sought. ``K`` must have no null row or column, see
    :func:`.check_null_rows_cols`, whereas ``KG`` may have null rows and
    columns. ``K`` and ``KG`` are symmetric when ``symmetric=None`` finds
    them so, see :func:`.is_symmetric`.

    **Symmetric matrices.** ``K`` must be positive definite, which is
    checked with ``check=True``. The sparse solvers are
    :func:`scipy.sparse.linalg.eigsh` in Cayley mode, using the shift of
    :func:`estimate_cayley_sigma`, or ``sigma`` when given; when its
    accurate eigenpairs miss load multipliers below the highest one, e.g.
    copies of repeated ones, the same completed by deflated passes, see
    :func:`_complete_both_signs`; and then :func:`scipy.sparse.linalg.eigsh`
    with a 10 times larger shift. Each one computes the lowest positive
    load multipliers and, when fewer than ``num_eigvalues`` exist, the
    negative ones closest to zero, i.e. the critical load multipliers of the
    reversed load, with ``-KG``, see :func:`_eigsh_both_signs`. The
    eigenpairs are verified with the residual and inertia checks of
    :func:`check_eigenpairs`, for both signs, see :func:`_check_both_signs`.
    The dense fallback condenses out the degrees-of-freedom where ``KG`` has
    null rows and solves the condensed problem with
    :func:`scipy.linalg.eigh`, see :func:`_eigh_condensed`.

    **Unsymmetric matrices**, e.g. ``KG`` with the load stiffness of a
    follower load, may have complex load multipliers. The sparse solver is a
    search of shifts along the positive real axis with
    :func:`scipy.sparse.linalg.eigs`, see :func:`_eigs_shift_search`, which
    finds the lowest positive real load multiplier; the returned eigenpairs
    are the ``num_eigvalues`` load multipliers nearest the last shift of the
    search. Only the residuals are verified. The dense fallback,
    :func:`_eig_condensed`, keeps the degrees-of-freedom where either the
    row or the column of ``KG`` is not null; when none of its load
    multipliers is positive and real, they are returned with a
    :class:`RuntimeWarning`. Eigenvalues whose imaginary part is below
    ``1e-8`` of their modulus are returned as real, with real eigenvectors.

    The sparse solvers run first and the dense solver is a last resort, see
    the solver policy of :mod:`pyfe3d.solver.eigen`.

    Parameters
    ----------
    K, KG : sparse_matrix
        Constitutive and geometric stiffness matrices of the free
        degrees-of-freedom, i.e. with the constrained ones partitioned out,
        e.g. ``KC0uu = KC0[bu, :][:, bu]``. ``K`` should include the
        constant terms that apply, e.g. the geometric stiffness matrix of a
        preload, and ``KG`` is the matrix multiplied by the load multiplier.
        Shape ``(n, n)``.
    num_eigvalues : int, optional
        Number of load multipliers to compute, at most ``n - 2`` with the
        sparse solvers. Default ``6``.
    tol : float, optional
        Tolerance passed to :func:`scipy.sparse.linalg.eigsh` and
        :func:`scipy.sparse.linalg.eigs`, ``0`` meaning machine precision.
        Default ``0``.
    sigma : float or None, optional
        Shift of the Cayley mode of the symmetric sparse solver. With
        ``None`` it is estimated from the matrices, which is the recommended
        use. Default ``None``.
    precondition : bool, optional
        Whether to equilibrate the diagonal before solving, see
        :func:`diagonal_preconditioner`. This does not change the
        eigenvalues. Default ``True``.
    check : bool, optional
        Whether to verify the eigenpairs of the sparse solvers, and, for
        symmetric matrices, that ``K`` is positive definite. With
        ``False``, the first sparse solver is returned without verification
        and without dense fallback. Default ``True``.
    check_rtol : float, optional
        Largest acceptable relative residual of the eigenpairs,
        dimensionless. Default ``1.e-3``.
    symmetric : bool or None, optional
        Whether ``K`` and ``KG`` are symmetric. ``None`` checks it with
        :func:`.is_symmetric`, ``True`` uses the solvers of symmetric
        matrices, ``False`` the solvers of unsymmetric matrices. Default
        ``None``.
    sparse_solver : bool, optional
        With ``True`` the sparse solvers run first, the dense one being a
        last resort. With ``False``, the full problem is explicitly solved
        with a dense solver, :func:`scipy.linalg.eigh`, which requires a
        positive definite ``K``, or :func:`scipy.linalg.eig` for unsymmetric
        matrices, returning all the load multipliers, without verification;
        it is a reference for validations. Default ``True``.
    max_dense_size : int, optional
        Maximum size of the dense problem solved as a last resort, the
        number of non-null rows of ``KG``, or of non-null rows or columns for
        unsymmetric matrices; ``0`` disables the dense fallback. Default
        ``2000``.

    Returns
    -------
    eigvals : np.ndarray
        Load multipliers, dimensionless: the positive ones first in
        increasing order, followed by the negative ones in order of
        increasing modulus, then the complex ones by increasing modulus, see
        :func:`_lb_order`. Fewer than ``num_eigvalues`` are returned by the
        symmetric sparse solvers when fewer finite load multipliers exist,
        the infinite ones, e.g. of the null rows of ``KG``, being discarded.
        The dense solvers return as ``-inf``, last, the load multipliers of
        the eigenvalues null up to the round-off, see
        :func:`_null_mu_pencil`. Complex only when complex load multipliers
        are returned.
    eigvecs : np.ndarray
        Eigenvectors, one per column, in the same order. Shape ``(n, k)``.

    Raises
    ------
    numpy.linalg.LinAlgError
        When ``K`` has a null row or column, with the indices of these
        degrees-of-freedom, see :func:`.check_null_rows_cols`; for symmetric
        matrices, when ``check=True`` or ``sparse_solver=False`` and ``K`` is
        not positive definite, i.e. its factorization has a negative or a
        zero pivot, with the likely causes: insufficient boundary
        conditions, a mechanism, distorted elements, e.g. re-entrant
        quadrilaterals, or a preload beyond the buckling load included in
        ``K``, see :func:`.sparseutils._not_positive_definite_message`.
    RuntimeError
        When ``check=True`` and no eigenvalue solver passed the
        verification, with the list of the attempted solvers and their
        failures.

    Warns
    -----
    DenseFallbackWarning
        When the sparse solvers failed and the dense solver was used.
    RuntimeWarning
        For unsymmetric matrices, when the dense fallback found no positive
        real load multiplier.

    Notes
    -----
    The problem solved is `(\boldsymbol{K}_{C0} + \lambda \boldsymbol{K}_G) \boldsymbol{u} = \boldsymbol{0}`, written
    as `\boldsymbol{K}_G \boldsymbol{u} = \mu \boldsymbol{K}_{C0} \boldsymbol{u}` with `\lambda = -1/\mu`, whose lowest
    positive load multipliers are the most negative `\mu`, returned by the
    Cayley mode while the shift is large enough, see the notes of
    :mod:`pyfe3d.solver.eigen`. With ``precondition=True`` the problem
    solved is the one of `\boldsymbol{P} \boldsymbol{K}_{C0} \boldsymbol{P}` and `\boldsymbol{P} \boldsymbol{K}_G \boldsymbol{P}`, with the same
    eigenvalues, see :func:`diagonal_preconditioner`; the verification
    always uses the original matrices.

    A null row or column of `\boldsymbol{K}_{C0}` is a degree-of-freedom without
    stiffness, a modelling error: with the same null row or column of
    `\boldsymbol{K}_G`, `\det(\boldsymbol{K}_{C0} + \lambda \boldsymbol{K}_G) = 0` for every `\lambda`, a
    singular pencil whose computed eigenvalues are arbitrary, and otherwise
    `\lambda = 0` is a spurious load multiplier. It is therefore reported
    instead of being removed, unlike in structsolve, see
    :func:`.check_null_rows_cols`. A null row of `\boldsymbol{K}_G`, at a
    degree-of-freedom that the load does not destabilize, only gives an
    infinite load multiplier.

    The load multipliers of a cylindrical shell with a rotationally
    symmetric mesh and load, e.g. torsion, come in pairs equal up to the
    round-off, of which the Lanczos process of ARPACK may return one
    copy only; the missing copies, detected by the inertia check, are
    computed by deflated passes, see :func:`_complete_repeated`. A
    failure of the verification that remains usually means that the mesh,
    and not the solver, needs attention, e.g. a spectrum with many nearly
    coincident eigenvalues.

    Non-conservative systems, e.g. under follower loads, may lose stability
    by flutter, which this static criterion does not detect, see
    :func:`natural_frequency`. When no returned load multiplier is positive
    and real, no divergence load was found.

    """
    check_null_rows_cols(K, 'K')
    K, KG = csr_matrix(K), csr_matrix(KG)
    n = K.shape[0]
    if symmetric is None:
        symmetric = is_symmetric(K) and is_symmetric(KG)
    if precondition:
        P = diagonal_preconditioner(K)
        Ks = csr_matrix(P @ K @ P)
        KGs = csr_matrix(P @ KG @ P)
    else:
        P = None
        Ks, KGs = K, KG

    def unscale(result):
        mu, eigvecs = result
        if P is not None:
            eigvecs = np.asarray(P @ eigvecs)
        return mu, eigvecs

    if not sparse_solver:
        a, b = KGs.toarray(), Ks.toarray()
        if symmetric:
            try:
                mu, eigvecs = eigh(a=a, b=b)
            except LinAlgError as e:
                # NOTE eigh() needs the Cholesky factorization of b
                count = _negative_pivots(K)
                if count == 0:
                    raise
                raise LinAlgError(_not_positive_definite_message(
                    'Linear buckling analysis', 'K', count)
                    + ' (scipy.linalg.eigh: %s)' % e) from e
            mu[_null_mu_pencil(a, b, mu, eigvecs, eigvecs)] = 0.
            mu, eigvecs = _sort_eigenpairs(mu, eigvecs)
        else:
            mu, vl, eigvecs = _eig_complex(a=a, b=b, left=True)
            mu[_null_mu_pencil(a, b, mu, vl, eigvecs)] = 0.
            mu, eigvecs = _general_eigenpairs(mu, eigvecs)
        mu, eigvecs = unscale((mu, eigvecs))
    elif symmetric:
        k = min(num_eigvalues, n - 2)
        once = {}

        def mu_max():
            if 'mu_max' not in once:
                once['mu_max'] = _estimate_mu_max(Ks, KGs)
            return once['mu_max']

        def K_positive_definite(A=None):
            if 'pd' not in once:
                once['pd'] = is_positive_definite(K)
            return once['pd']

        if check and not K_positive_definite():
            raise LinAlgError(_not_positive_definite_message(
                'Linear buckling analysis', 'K', _negative_pivots(K)))

        def cayley(factor):
            def solver():
                if sigma is None:
                    m = mu_max()
                    s = 1. if m is None else 10.*m
                else:
                    s = sigma
                result = _eigsh_both_signs(Ks, KGs, k, tol, factor*s,
                                           mu_max(), K_positive_definite)
                once['cayley', factor] = result + (factor*s,)
                return unscale(result)
            return solver

        def completed(factor):
            def solver():
                if ('cayley', factor) not in once:
                    raise _NotApplicable()
                mu, eigvecs, s = once['cayley', factor]
                # NOTE the deflated passes about -s find the lowest missing
                #      load multipliers only when s exceeds all |mu|, see
                #      _complete_both_signs
                if mu_max() is not None and s <= mu_max():
                    raise _NotApplicable()
                # NOTE only accurate eigenpairs are completed
                mu_u, eigvecs_u = unscale((mu, eigvecs))
                kept = np.isfinite(mu_u) & ~_null_mu(mu_u)
                if not np.any(kept):
                    raise _NotApplicable()
                with np.errstate(divide='ignore'):
                    lam = -1./mu_u[kept]
                if not np.all(_relative_residuals(K, KG, lam,
                        eigvecs_u[:, kept]) <= check_rtol):
                    raise _NotApplicable()
                mu, eigvecs, num_added = _complete_both_signs(Ks, KGs, mu,
                        eigvecs, k, tol, s, mu_max())
                if not num_added:
                    raise _NotApplicable()
                return unscale((mu, eigvecs))
            return solver

        def verify(mu, eigvecs):
            return _check_both_signs(K, KG, mu, eigvecs, check_rtol,
                                     K_positive_definite=K_positive_definite)

        num_active = int(np.count_nonzero(
            np.asarray(abs(KG).sum(axis=1)).ravel()))
        strategies = [('eigsh() in Cayley mode', cayley(1.)),
                      ('eigsh() in Cayley mode completed by deflated passes',
                       completed(1.)),
                      ('eigsh() in Cayley mode with a 10 times larger shift',
                       cayley(10.))]
        dense = ('eigh() with condensed degrees-of-freedom',
                 lambda: unscale(_eigh_condensed(Ks, KGs, k)))
        (mu, eigvecs), _ = _sparse_first('Linear buckling analysis',
                strategies, verify if check else None, dense, num_active,
                max_dense_size, hint=' Try sparse_solver=False or a larger '
                'max_dense_size.')
    else:
        k = num_eigvalues
        num_active = int(np.count_nonzero(
            (np.asarray(abs(KG).sum(axis=1)).ravel() > 0)
            | (np.asarray(abs(KG).sum(axis=0)).ravel() > 0)))

        def verify(mu, eigvecs):
            with np.errstate(divide='ignore', invalid='ignore'):
                return check_eigenpairs(K, KG, -1./mu, eigvecs, check_rtol,
                                        check_inertia=False)

        strategies = [('eigs() shift search along the positive real axis',
                       lambda: unscale(_eigs_shift_search(Ks, KGs, k, tol,
                                                          check_rtol)))]
        dense = ('eig() with condensed degrees-of-freedom',
                 lambda: unscale(_eig_condensed(Ks, KGs, k)))
        (mu, eigvecs), name = _sparse_first(
            'Linear buckling analysis of unsymmetric matrices', strategies,
            verify if check else None, dense, num_active, max_dense_size,
            hint=' Try sparse_solver=False or a larger max_dense_size.')
        if name == dense[0] and not np.any(np.isreal(mu) & (np.real(mu) < 0)):
            warnings.warn('Linear buckling analysis of unsymmetric matrices: '
                          'no positive real load multiplier exists, i.e. no '
                          'divergence load, the returned load multipliers are '
                          'negative or complex', RuntimeWarning, stacklevel=2)
        mu, eigvecs = _general_eigenpairs(mu, eigvecs, k)

    with np.errstate(divide='ignore', invalid='ignore'):
        eigvals = -1./mu
    eigvals[mu == 0] = -np.inf
    return eigvals, eigvecs


def _check_freq(K, M, omega2, eigvecs, rtol, symmetric, min_rel_gap=1.e-4):
    r"""Verify the eigenpairs of the natural frequency problem

    Parameters
    ----------
    K, M : sparse_matrix
        Stiffness and mass matrices, shape ``(n, n)``.
    omega2 : array_like
        Eigenvalues `\omega_n^2`, real or complex, 1-D array of size ``k``.
    eigvecs : array_like
        Eigenvectors, shape ``(n, k)``.
    rtol : float
        Largest acceptable relative residual, dimensionless.
    symmetric : bool
        Whether the inertia check is done, valid for symmetric matrices.
    min_rel_gap : float, optional
        Relative gap below the highest eigenvalue at which the inertia is
        tested, dimensionless. Default ``1.e-4``.

    Returns
    -------
    error : str or None
        Description of the check that failed, ``None`` when all passed.

    Notes
    -----
    - The relative residual `\|\boldsymbol{K} \boldsymbol{u} - \omega_n^2 \boldsymbol{M} \boldsymbol{u}\|/(\|\boldsymbol{K} \boldsymbol{u}\| +
      |\omega_n^2| \|M u\|)` of every finite eigenpair must not be larger
      than ``rtol``. For a rigid-body mode, whose `\|\boldsymbol{K} \boldsymbol{u}\| + |\omega_n^2|
      \|M u\|` is not larger than ``RIGID_BODY_RTOL`` times the
      matrix-norm scale `(\|\boldsymbol{K}\|_F + |\omega_n^2| \|\boldsymbol{M}\|_F) \|\boldsymbol{u}\|`, the
      residual is relative to this scale, see :func:`_relative_residuals`.
      An eigenpair with a larger relative residual also passes when its
      normwise backward error, the residual relative to the matrix-norm
      scale, is not larger than ``BACKWARD_ERROR_RTOL``, the machine
      precision, i.e. when it is accurate up to the round-off, e.g. an
      eigenvalue close to zero of a model pre-stressed close to buckling,
      see :func:`_backward_errors`.
    - With ``symmetric=True``, no eigenvalue is missing below the highest
      one returned, `\omega^2_k`: the number of negative pivots of `\boldsymbol{K} - s
      \boldsymbol{M}`, with `s = \omega^2_k - min\_rel\_gap |\omega^2_k|`, must be the
      number of returned eigenvalues lower than `s`. For symmetric `\boldsymbol{K}` and
      a positive definite `\boldsymbol{M}`, `\boldsymbol{K} - s \boldsymbol{M}` is congruent to
      `\boldsymbol{M}^{-1/2} \boldsymbol{K} \boldsymbol{M}^{-1/2} - s \boldsymbol{I}`, whose eigenvalues are `\omega_i^2 -
      s`, hence its number of negative eigenvalues, equal to its number of
      negative pivots by Sylvester's law of inertia, is the number of
      `\omega_i^2 < s`. The check is skipped when the count is unknown, see
      :func:`_negative_pivots`, or when `s` is zero.

    """
    omega2 = np.asarray(omega2)
    eigvecs = np.asarray(eigvecs)
    if not eigvecs.shape[1]:
        return 'no eigenpair was found'
    finite = np.isfinite(omega2)
    u = eigvecs[:, finite]
    w2 = omega2[finite]
    res = _relative_residuals(K, M, -w2, u, rigid_rtol=RIGID_BODY_RTOL)
    backward_error = _backward_errors(K, M, -w2, u)
    failed = ~((res <= rtol) | (backward_error <= BACKWARD_ERROR_RTOL))
    if np.any(failed):
        i = int(np.argmax(np.where(failed, np.nan_to_num(res, nan=np.inf),
                                   -np.inf)))
        return ('relative residual %.2e > %.1e and backward error %.1e > '
                '%.1e for the eigenvalue omega**2=%r'
                % (res[i], rtol, backward_error[i], BACKWARD_ERROR_RTOL,
                   w2[i]))
    if symmetric:
        w2 = np.real(w2)
        if not w2.size:
            return 'no finite eigenvalue was found'
        s = w2.max() - min_rel_gap*abs(w2.max())
        expected = int(np.count_nonzero(w2 < s))
        count = (_negative_pivots(csr_matrix(K) - s*csr_matrix(M))
                 if s != 0 else None)
        if count is not None and count != expected:
            return ('%d eigenvalues omega**2 lower than %g exist, %d were '
                    'found' % (count, s, expected))
    return None


def _dense_freq(K, M, k, sigma, symmetric):
    r"""Dense eigenpairs of the natural frequency problem

    Parameters
    ----------
    K, M : sparse_matrix
        Stiffness and mass matrices, shape ``(n, n)``.
    k : int or None
        Number of eigenpairs returned, all with ``None``.
    sigma : float
        Shift of the sparse solver; the unsymmetric solver returns the
        eigenvalues nearest it.
    symmetric : bool
        Whether :func:`scipy.linalg.eigh` is used.

    Returns
    -------
    omega2, eigvecs : np.ndarray
        Eigenvalues `\omega_n^2` and eigenvectors.

    Notes
    -----
    With ``symmetric=True``, `\boldsymbol{M} \boldsymbol{u} = (1/\omega_n^2) \boldsymbol{K} \boldsymbol{u}` is solved
    with :func:`scipy.linalg.eigh`, which gives the lowest frequencies with
    smaller residuals than `\boldsymbol{K} \boldsymbol{u} = \omega_n^2 \boldsymbol{M} \boldsymbol{u}` for stiff models,
    e.g. relative residuals of at most 1.6e-10 against 1.4e-8 for the 10
    lowest frequencies of the Quad4 plate of
    ``test_dense_symmetric_lowest_frequencies`` in
    ``tests/test_solver_natural_frequency.py``, and accepts a singular
    `\boldsymbol{M}`, but requires a positive definite `\boldsymbol{K}`; otherwise `\boldsymbol{K} \boldsymbol{u} =
    \omega_n^2 \boldsymbol{M} \boldsymbol{u}` is solved. The eigenvectors are normalized with
    respect to `\boldsymbol{M}`. With ``symmetric=False``, :func:`scipy.linalg.eig` is
    used through :func:`_eig_complex`, and the finite `\omega_n^2` nearest
    ``sigma`` are returned.

    """
    Kd = csr_matrix(K).toarray()
    Md = csr_matrix(M).toarray()
    if symmetric:
        try:
            ev, eigvecs = eigh(a=Md, b=Kd)
            positive = np.flatnonzero(ev > 0)[::-1]
            if k is not None:
                positive = positive[:k]
            w2 = 1./ev[positive]
            eigvecs = eigvecs[:, positive]
            eigvecs /= np.sqrt(np.einsum('ij,ij->j', eigvecs, Md @ eigvecs))
        except LinAlgError:
            w2, eigvecs = eigh(a=Kd, b=Md)
            if k is not None:
                w2, eigvecs = w2[:k], eigvecs[:, :k]
        return w2, eigvecs
    ev, eigvecs = _eig_complex(a=Md, b=Kd)
    with np.errstate(divide='ignore', invalid='ignore'):
        w2 = 1./ev
    finite = np.flatnonzero(np.isfinite(w2))
    order = finite[np.argsort(np.abs(w2[finite] - sigma), kind='stable')]
    if k is not None:
        order = order[:k]
    return w2[order], eigvecs[:, order]


def natural_frequency(K, M, num_eigvalues=6, tol=0, precondition=True,
                      symmetric=None, check=True, check_rtol=1.e-6,
                      sparse_solver=True, max_dense_size=2000):
    r"""Natural frequencies from the stiffness and mass matrices

    ``K`` must have no null row or column, see :func:`.check_null_rows_cols`,
    whereas ``M`` may have null rows and columns, e.g. a lumped mass matrix
    without rotary inertia. The matrices are optionally preconditioned
    with :func:`diagonal_preconditioner`, and the eigenvalues, the squares
    of the circular frequencies, are computed in shift-invert mode with
    ``sigma=-1.`` and ``which='LM'``, with :func:`scipy.sparse.linalg.eigsh`
    for symmetric matrices and :func:`scipy.sparse.linalg.eigs` otherwise,
    e.g. with the aerodynamic matrix of piston theory or the load stiffness
    of a follower load in ``K``. ``K`` and ``M`` are symmetric when
    ``symmetric=None`` finds them so, see :func:`.is_symmetric`. The number
    of computed frequencies is ``k = min(num_eigvalues, n - 2)``.

    The eigenpairs are verified, see :func:`_check_freq`: the relative
    residual of each eigenpair must not be larger than ``check_rtol``,
    unless its backward error is at the round-off, and, for symmetric
    matrices, no eigenvalue lower than the highest one returned may be
    missing. For symmetric matrices with a ``K`` without negative pivots,
    accurate eigenpairs that miss eigenvalues, e.g. copies of repeated
    ones, are completed by deflated passes, see :func:`_complete_repeated`.
    When the verification still fails or the sparse
    solver raises an error, the dense solver of :func:`_dense_freq` is used
    as a last resort, with a :class:`DenseFallbackWarning`, see the solver
    policy of :mod:`pyfe3d.solver.eigen`.

    The eigenpairs are sorted together by increasing frequency, the
    eigenvectors are transformed back to the original degrees-of-freedom,
    and negative eigenvalues of symmetric matrices give a zero frequency.

    Parameters
    ----------
    K, M : sparse_matrix
        Constitutive stiffness and mass matrices of the free
        degrees-of-freedom, i.e. with the constrained ones partitioned out,
        e.g. ``KC0uu = KC0[bu, :][:, bu]``. ``K`` should include the
        constant terms that apply, e.g. the geometric stiffness matrix of a
        preload. Shape ``(n, n)``.
    num_eigvalues : int, optional
        Number of frequencies to compute. Default ``6``.
    tol : float, optional
        Tolerance passed to :func:`scipy.sparse.linalg.eigsh` or
        :func:`scipy.sparse.linalg.eigs`, ``0`` meaning machine precision.
        Default ``0``.
    precondition : bool, optional
        Whether to equilibrate the diagonal of ``K`` before solving.
        Default ``True``.
    symmetric : bool or None, optional
        Whether ``K`` and ``M`` are symmetric. ``None`` checks it with
        :func:`.is_symmetric`. Default ``None``.
    check : bool, optional
        Whether to verify the eigenpairs of the sparse solver. With
        ``False`` they are returned without verification and without dense
        fallback. Default ``True``.
    check_rtol : float, optional
        Largest acceptable relative residual of the eigenpairs,
        dimensionless. An eigenpair with a larger one is also accepted when
        its normwise backward error is not larger than
        ``BACKWARD_ERROR_RTOL``, the machine precision, e.g. an eigenvalue
        close to zero of a model pre-stressed close to buckling, see
        :func:`_backward_errors`. Default ``1.e-6``.
    sparse_solver : bool, optional
        With ``False``, the full problem is solved with a dense solver,
        returning all the frequencies, without verification. Default
        ``True``.
    max_dense_size : int, optional
        Maximum size of the dense problem solved as a last resort; ``0``
        disables the dense fallback. Default ``2000``.

    Returns
    -------
    omegan : np.ndarray
        Circular frequencies in rad/s, in increasing order. 1-D array of size
        ``k``. For unsymmetric matrices whose eigenvalues `\omega_n^2` are
        complex, the complex `\omega_n = \sqrt{\omega_n^2}` (principal
        root), sorted by increasing real part of `\omega_n^2`, then by
        increasing imaginary part.
    eigvecs : np.ndarray
        Eigenvectors, one per column, in the same order. Shape ``(n, k)``.

    Raises
    ------
    numpy.linalg.LinAlgError
        When ``K`` has a null row or column, with the indices of these
        degrees-of-freedom, see :func:`.check_null_rows_cols`.
    RuntimeError
        When ``check=True`` and no eigenvalue solver passed the
        verification. For symmetric matrices with a ``K`` that is not
        positive definite, the message starts with that diagnosis and its
        likely causes, see
        :func:`.sparseutils._not_positive_definite_message`: an indefinite
        ``K`` is accepted, e.g. with a preload beyond the buckling load,
        but its negative eigenvalues may be far below the shift and missed
        by the sparse solver.

    Warns
    -----
    DenseFallbackWarning
        When the sparse solver failed and the dense solver was used.

    Notes
    -----
    The problem solved is `(\boldsymbol{K}_{C0} - \omega_n^2 \boldsymbol{M}) \boldsymbol{u} = \boldsymbol{0}`, and
    the frequencies of symmetric matrices are `\omega_n = \sqrt{\max(
    \omega_n^2, 0)}`. In the shift-invert mode, the sparse solvers find the
    eigenvalues closest to the shift ``sigma=-1.``, i.e. the lowest ones
    when `\boldsymbol{K}_{C0}` is positive semi-definite and `\boldsymbol{M}` positive definite,
    factorizing `\boldsymbol{K}_{C0} + \boldsymbol{M}`, which is not singular when `\boldsymbol{K}_{C0}` is
    singular only because of unconstrained rigid-body modes that have mass.

    A null row or column of `\boldsymbol{K}_{C0}` is a degree-of-freedom without
    stiffness, a modelling error: with the same null row or column of
    `\boldsymbol{M}`, `\det(\boldsymbol{K}_{C0} - \omega_n^2 \boldsymbol{M}) = 0` for every `\omega_n^2`,
    a singular pencil whose computed eigenvalues are arbitrary, and
    otherwise `\omega_n^2 = 0` is a spurious frequency. It is therefore
    reported instead of being removed, unlike in structsolve, see
    :func:`.check_null_rows_cols`.

    A mass matrix without rotary inertia, and one without inertia associated
    with the drilling rotation, is singular, so the eigenvalues associated
    with those degrees-of-freedom are infinite. The shift-invert mode used
    here looks for the eigenvalues closest to zero and is not troubled by
    them, but a lumped mass matrix will place the massless degrees-of-freedom
    wherever the factorization puts them, so prefer a consistent mass matrix
    when the lowest frequencies matter.

    Non-conservative systems, e.g. under follower loads, may lose stability
    by flutter, which the static criterion of :func:`linear_buckling` does
    not detect: with ``K`` the tangent stiffness matrix at a load level,
    including the load stiffness, the eigenvalues `\omega_n^2` are real and
    positive below the flutter load and become complex conjugate pairs above
    it, when two natural frequencies coalesce, a double root of the
    frequency equation. This is the dynamic method of [1]_ (Introduction,
    Section 6, pp. 12-15, and Section 7, pp. 15-17), where `\omega_n` is
    the frequency `\Omega` of the characteristic exponent `s = i\Omega`. A
    real negative `\omega_n^2`, i.e. a real positive `s`, indicates
    divergence ([1]_, p. 17).

    References
    ----------
    .. [1] Bolotin, V. V., Nonconservative Problems of the Theory of Elastic
       Stability, Pergamon Press, Oxford, 1963.

    """
    check_null_rows_cols(K, 'K')
    K, M = csr_matrix(K), csr_matrix(M)
    n = K.shape[0]
    if symmetric is None:
        symmetric = is_symmetric(K) and is_symmetric(M)
    if precondition:
        P = diagonal_preconditioner(K)
        Ks = csr_matrix(P @ K @ P)
        Ms = csr_matrix(P @ M @ P)
    else:
        P = None
        Ks, Ms = K, M
    sigma = -1.

    def unscale(result):
        w2, eigvecs = result
        if P is not None:
            eigvecs = np.asarray(P @ eigvecs)
        return w2, eigvecs

    if not sparse_solver:
        w2, eigvecs = unscale(_dense_freq(Ks, Ms, None, sigma, symmetric))
    else:
        k = min(num_eigvalues, n - 2)
        once = {}
        if symmetric:
            def solver():
                result = capped_eigsh(A=Ks, k=k, M=Ms, sigma=sigma,
                                      which='LM', tol=tol)
                once['eigsh'] = result
                return unscale(result)

            def completed():
                if 'eigsh' not in once:
                    raise _NotApplicable()
                w2, eigvecs = once['eigsh']
                # NOTE only accurate eigenpairs are completed, and the
                #      missing eigenvalues of an indefinite K may be far
                #      below the shift, i.e. not the nearest to it
                if (_check_freq(K, M, *unscale((w2, eigvecs)), check_rtol,
                                False) is not None
                        or K_negative_pivots() != 0):
                    raise _NotApplicable()
                num_missing = _missing_counter(Ks, -Ms, lambda w: w, k)
                w2, eigvecs, num_added = _complete_repeated(Ks, Ms, w2,
                        eigvecs, sigma, tol, num_missing, np.isfinite, k)
                if not num_added:
                    raise _NotApplicable()
                return unscale((w2[:k], eigvecs[:, :k]))

            strategies = [('eigsh() in shift-invert mode', solver),
                          ('eigsh() in shift-invert mode completed by '
                           'deflated passes', completed)]
        else:
            strategies = [('eigs() in shift-invert mode',
                           lambda: unscale(eigs(A=Ks, k=k, M=Ms, sigma=sigma,
                                                which='LM', tol=tol,
                                                **start_kwargs(n))))]
        dense = ('eigh()' if symmetric else 'eig()',
                 lambda: unscale(_dense_freq(Ks, Ms, k, sigma, symmetric)))

        def K_negative_pivots():
            if 'count' not in once:
                once['count'] = _negative_pivots(K)
            return once['count']

        def diagnose():
            if not symmetric or K_negative_pivots() == 0:
                return None
            return (_not_positive_definite_message('Frequency analysis',
                    'K', K_negative_pivots()) + '; negative eigenvalues '
                    'omega**2 are expected only in the latter case')

        def verify(w2, eigvecs):
            return _check_freq(K, M, w2, eigvecs, check_rtol, symmetric)

        (w2, eigvecs), _ = _sparse_first('Frequency analysis',
                strategies, verify if check else None, dense, n,
                max_dense_size, hint=' Try sparse_solver=False or a larger '
                'max_dense_size.', diagnose=diagnose)
    if symmetric:
        w2 = np.real(w2)
        eigvecs = np.real(eigvecs)
        order = np.argsort(w2, kind='stable')
        w2 = w2[order]
        omegan = np.sqrt(np.maximum(w2, 0.))
    else:
        w2, eigvecs = _real_if_close(w2, eigvecs)
        order = np.lexsort((w2.imag, w2.real))
        w2 = w2[order]
        if np.all(w2.imag == 0):
            w2 = w2.real
            eigvecs = eigvecs.real
            omegan = np.sqrt(np.maximum(w2, 0.))
        else:
            omegan = np.sqrt(w2)
    return omegan, eigvecs[:, order]


def _quadratic_residuals(K, C, M, lam, u, rigid_rtol=None):
    r"""Relative residuals of the eigenpairs of `(\lambda^2 \boldsymbol{M} + \lambda \boldsymbol{C}
    + \boldsymbol{K}) \boldsymbol{u} = \boldsymbol{0}`

    Parameters
    ----------
    K, C, M : sparse_matrix or np.ndarray
        Stiffness, damping and mass matrices, shape ``(n, n)``.
    lam : array_like
        Finite eigenvalues, real or complex, 1-D array of size ``k``.
    u : array_like
        Eigenvectors, shape ``(n, k)``.
    rigid_rtol : float or None, optional
        ``None`` gives the relative residual ``norm(lam**2*(M @ u) + lam*(C
        @ u) + K @ u)/(abs(lam)**2*norm(M @ u) + abs(lam)*norm(C @ u) +
        norm(K @ u))`` of every eigenpair. Otherwise the eigenpairs whose
        denominator is not larger than ``rigid_rtol*(abs(lam)**2*norm_F(M) +
        abs(lam)*norm_F(C) + norm_F(K))*norm(u)``, null up to the round-off,
        e.g. the rigid-body modes, give the residual relative to this
        matrix-norm scale, ``norm_F`` being the Frobenius norm. Default
        ``None``.

    Returns
    -------
    residual : np.ndarray
        One relative residual per eigenpair, dimensionless, ``inf`` when the
        denominator is null.

    Notes
    -----
    The relative residual

    .. math::

        \frac{\|(\lambda^2 \boldsymbol{M} + \lambda \boldsymbol{C} + \boldsymbol{K}) \boldsymbol{u}\|}{|\lambda|^2
        \|\boldsymbol{M} \boldsymbol{u}\| + |\lambda| \, \|\boldsymbol{C} \boldsymbol{u}\| + \|\boldsymbol{K} \boldsymbol{u}\|}

    is the quadratic counterpart of the one of :func:`_relative_residuals`,
    with the same treatment of the eigenpairs null up to the round-off: with
    ``rigid_rtol``, their residual is the normwise backward error of the
    quadratic eigenvalue problem in the Frobenius norm, `\|\boldsymbol{r}\|/
    ((|\lambda|^2 \|\boldsymbol{M}\|_F + |\lambda| \, \|\boldsymbol{C}\|_F + \|\boldsymbol{K}\|_F)
    \|\boldsymbol{u}\|)`, see [1]_ (Section 4.2, Eq. (4.7), pp. 264-265).

    References
    ----------
    .. [1] Tisseur, F. and Meerbergen, K., The quadratic eigenvalue
       problem, SIAM Review, 43(2), 235-286, 2001.
       https://doi.org/10.1137/S0036144500381988

    """
    Ku = K @ u
    Cu = C @ u
    Mu = M @ u
    lam = np.asarray(lam)
    num = np.linalg.norm(Ku + Cu*lam + Mu*lam**2, axis=0)
    den = (np.linalg.norm(Ku, axis=0) + np.abs(lam)*np.linalg.norm(Cu, axis=0)
           + np.abs(lam)**2*np.linalg.norm(Mu, axis=0))
    if rigid_rtol is not None:
        scale = (_frobenius(K) + np.abs(lam)*_frobenius(C)
                 + np.abs(lam)**2*_frobenius(M))*np.linalg.norm(u, axis=0)
        den = np.where(den <= rigid_rtol*scale, scale, den)
    residual = np.full(lam.shape, np.inf)
    np.divide(num, den, out=residual, where=den > 0)
    return residual


def _quadratic_backward_errors(K, C, M, lam, u):
    r"""Normwise backward errors of the eigenpairs of `(\lambda^2 \boldsymbol{M} +
    \lambda \boldsymbol{C} + \boldsymbol{K}) \boldsymbol{u} = \boldsymbol{0}`

    Parameters
    ----------
    K, C, M : sparse_matrix or np.ndarray
        Stiffness, damping and mass matrices, shape ``(n, n)``.
    lam : array_like
        Finite eigenvalues, real or complex, 1-D array of size ``k``.
    u : array_like
        Eigenvectors, shape ``(n, k)``.

    Returns
    -------
    backward_error : np.ndarray
        ``norm(lam**2*(M @ u) + lam*(C @ u) + K @ u)/((abs(lam)**2*norm_F(M)
        + abs(lam)*norm_F(C) + norm_F(K))*norm(u))`` of each eigenpair,
        ``norm_F`` being the Frobenius norm, dimensionless, ``inf`` when the
        denominator is null.

    Notes
    -----
    The quadratic counterpart of :func:`_backward_errors`, the normwise
    backward error of [1]_ (Theorem 1, Eq. (2.3)) with the tolerances
    `\|\boldsymbol{K}\|_F`, `\|\boldsymbol{C}\|_F` and `\|\boldsymbol{M}\|_F`, see also [2]_ (Section 4.2,
    Eq. (4.7), pp. 264-265). :func:`complex_frequency` accepts an eigenpair
    whose relative residual of :func:`_quadratic_residuals` is larger than
    ``check_rtol`` when its backward error is not larger than
    ``BACKWARD_ERROR_RTOL``, the machine precision. Measured with SciPy
    1.17.1 and Intel MKL 2025.0.0 for the cantilever of
    ``tests/test_beamc_natural_freq_cantilever_pre_stress.py`` pre-stressed
    to 0.9999 of its buckling load, with `\boldsymbol{C}` null, `10^{-6} \boldsymbol{K}_{C0}` or
    `10^{-4} \boldsymbol{K}_{C0} + 10^{-2} \boldsymbol{M}`: the lowest eigenpairs of ``eigs`` on
    the linearization, `|\lambda| = 0.30`, have relative residuals of `2.6
    \cdot 10^{-6}` to `7.0 \cdot 10^{-6}`, larger than the default
    ``check_rtol=1e-6``, and backward errors of `0.022 \epsilon` to `0.035
    \epsilon`.

    References
    ----------
    .. [1] Tisseur, F., Backward error and condition of polynomial eigenvalue
       problems, Linear Algebra and its Applications, 309(1-3), 339-361,
       2000. https://doi.org/10.1016/S0024-3795(99)00063-4
    .. [2] Tisseur, F. and Meerbergen, K., The quadratic eigenvalue
       problem, SIAM Review, 43(2), 235-286, 2001.
       https://doi.org/10.1137/S0036144500381988

    """
    Ku = K @ u
    Cu = C @ u
    Mu = M @ u
    lam = np.asarray(lam)
    num = np.linalg.norm(Ku + Cu*lam + Mu*lam**2, axis=0)
    scale = (_frobenius(K) + np.abs(lam)*_frobenius(C)
             + np.abs(lam)**2*_frobenius(M))*np.linalg.norm(u, axis=0)
    backward_error = np.full(lam.shape, np.inf)
    np.divide(num, scale, out=backward_error, where=scale > 0)
    return backward_error


def _companion_vectors(K, C, M, lam, Z):
    r"""Eigenvectors of the quadratic eigenvalue problem from those of its
    first companion linearization

    Parameters
    ----------
    K, C, M : sparse_matrix
        Stiffness, damping and mass matrices, shape ``(n, n)``.
    lam : np.ndarray
        Finite eigenvalues, 1-D array of size ``k``.
    Z : np.ndarray
        Eigenvectors `\boldsymbol{z} = \{\boldsymbol{u}^\top, \lambda \boldsymbol{u}^\top\}^\top` of the
        linearization, shape ``(2n, k)``.

    Returns
    -------
    U : np.ndarray
        Complex eigenvectors, shape ``(n, k)``, of unit Euclidean norm, with
        the component of largest modulus real and positive.

    Notes
    -----
    In finite precision the first and the last `n` components of `\boldsymbol{z}`
    are not equivalent approximations of `\boldsymbol{u}`; the one with the smaller
    relative residual, see :func:`_quadratic_residuals`, is kept, as in
    Algorithm 5.1 of [1]_ (Section 5.1, p. 267). E.g. the last `n`
    components vanish with `\lambda`.

    References
    ----------
    .. [1] Tisseur, F. and Meerbergen, K., The quadratic eigenvalue
       problem, SIAM Review, 43(2), 235-286, 2001.
       https://doi.org/10.1137/S0036144500381988

    """
    n = K.shape[0]
    Z = np.asarray(Z, dtype=complex)
    U1, U2 = Z[:n, :], Z[n:, :]
    r1 = _quadratic_residuals(K, C, M, lam, U1)
    r2 = _quadratic_residuals(K, C, M, lam, U2)
    U = np.where(r2 < r1, U2, U1)
    return _normalize_vectors(U)


def _normalize_vectors(U):
    r"""Columns of ``U`` scaled to a unit Euclidean norm, with the component
    of largest modulus real and positive"""
    U = np.asarray(U, dtype=complex)
    norm = np.linalg.norm(U, axis=0)
    U = U/np.where(norm > 0, norm, 1.)
    pivot = U[np.argmax(np.abs(U), axis=0), np.arange(U.shape[1])]
    phase = np.ones_like(pivot)
    nonnull = pivot != 0
    phase[nonnull] = np.abs(pivot[nonnull])/pivot[nonnull]
    return U*phase


def _qep_modes(lam, U, k, sigma):
    r"""One eigenpair per complex conjugate pair, the ``k`` nearest the
    shift, sorted by increasing damped circular frequency

    Parameters
    ----------
    lam : array_like
        Eigenvalues of a quadratic eigenvalue problem with real matrices,
        1-D array of size ``m``; the eigenvalues that are not finite are
        discarded.
    U : array_like
        Eigenvectors, shape ``(n, m)``.
    k : int or None
        Number of eigenpairs returned, all with ``None``.
    sigma : complex
        Shift: the eigenpairs kept are those nearest it.

    Returns
    -------
    lam, U : np.ndarray
        Complex arrays: the eigenvalues with `Im(\lambda) \ge 0`, sorted by
        increasing `Im(\lambda)`, then by increasing `|\lambda|`, and their
        eigenvectors.

    Notes
    -----
    The eigenvalues of real matrices are real or come in complex conjugate
    pairs `(\lambda, \lambda^*)`, with the eigenvectors `\boldsymbol{u}` and
    `\boldsymbol{u}^*` ([1]_, Table 1.1, p. 237). The eigenvalues whose imaginary
    part is not larger than ``1e-8`` times their modulus are made real with
    :func:`_real_if_close`. Each eigenvalue with `Im(\lambda) < 0` is then
    paired with the nearest unpaired one with `Im(\lambda) > 0` within
    ``1e-6`` times its modulus, which represents the pair, or else replaced
    by its conjugate, with the conjugate eigenvector, its partner not having
    been computed, e.g. with a complex shift, or when the solver kept one
    eigenvalue of the last pair. The ``k`` eigenvalues nearest the shift,
    `\min(|\lambda - \sigma|, |\lambda^* - \sigma|)`, are kept.

    References
    ----------
    .. [1] Tisseur, F. and Meerbergen, K., The quadratic eigenvalue
       problem, SIAM Review, 43(2), 235-286, 2001.
       https://doi.org/10.1137/S0036144500381988

    """
    lam = np.asarray(lam)
    finite = np.isfinite(lam)
    lam, U = _real_if_close(lam[finite], np.asarray(U)[:, finite])
    upper = list(np.flatnonzero(lam.imag > 0))
    keep = np.ones(lam.size, dtype=bool)
    for i in np.flatnonzero(lam.imag < 0):
        target = np.conj(lam[i])
        if upper:
            dist = np.abs(lam[upper] - target)
            j = int(np.argmin(dist))
            if dist[j] <= 1.e-6*abs(target):
                upper.pop(j)
                keep[i] = False
                continue
        lam[i] = target
        U[:, i] = np.conj(U[:, i])
    lam, U = lam[keep], U[:, keep]
    dist = np.minimum(np.abs(lam - sigma), np.abs(np.conj(lam) - sigma))
    order = np.argsort(dist, kind='stable')
    if k is not None:
        order = order[:k]
    lam, U = lam[order], U[:, order]
    order = np.lexsort((np.abs(lam), lam.imag))
    return lam[order], U[:, order]


def _companion(K, C, M):
    r"""Dense first companion linearization ``(A, B)`` of `(\lambda^2 \boldsymbol{M} +
    \lambda \boldsymbol{C} + \boldsymbol{K}) \boldsymbol{u} = \boldsymbol{0}`, see :func:`complex_frequency`"""
    n = K.shape[0]
    I = np.eye(n)
    Z = np.zeros((n, n))
    A = np.block([[Z, I],
                  [-csr_matrix(K).toarray(), -csr_matrix(C).toarray()]])
    B = np.block([[I, Z], [Z, csr_matrix(M).toarray()]])
    return A, B


def _dense_qep(K, C, M, k, sigma):
    r"""Dense eigenpairs of the quadratic eigenvalue problem

    Parameters
    ----------
    K, C, M : sparse_matrix
        Stiffness, damping and mass matrices, shape ``(n, n)``.
    k : int or None
        Number of eigenpairs returned, all with ``None``.
    sigma : complex
        Shift; the ``k`` eigenpairs nearest it are returned.

    Returns
    -------
    lam, U : np.ndarray
        Eigenvalues and eigenvectors, see :func:`_qep_modes`.

    Notes
    -----
    All the eigenpairs of the first companion linearization, see
    :func:`_companion`, are computed with :func:`scipy.linalg.eig`
    through :func:`_eig_complex`; the infinite eigenvalues of a singular
    `\boldsymbol{M}` are discarded.

    """
    A, B = _companion(K, C, M)
    w, Z = _eig_complex(a=A, b=B)
    finite = np.isfinite(w)
    w, Z = w[finite], Z[:, finite]
    U = _companion_vectors(K, C, M, w, Z)
    return _qep_modes(w, U, k, sigma)


def _eigs_qep(K, C, M, k, tol, sigma):
    r"""Eigenpairs of the quadratic eigenvalue problem nearest the shift,
    with :func:`scipy.sparse.linalg.eigs` in shift-invert mode

    Parameters
    ----------
    K, C, M : sparse_matrix
        Real stiffness, damping and mass matrices, shape ``(n, n)``.
    k : int
        Number of eigenvalues of the linearization computed by ARPACK, at
        most ``2*n - 2``.
    tol : float
        Tolerance of :func:`scipy.sparse.linalg.eigs`.
    sigma : complex
        Shift, not an eigenvalue.

    Returns
    -------
    lam, U : np.ndarray
        The finite eigenvalues found and their eigenvectors, see
        :func:`_companion_vectors`, to be passed to :func:`_qep_modes`.

    Raises
    ------
    RuntimeError
        When `\lambda^2 \boldsymbol{M} + \lambda \boldsymbol{C} + \boldsymbol{K}` is exactly singular at
        `\lambda = \sigma`.

    Notes
    -----
    With the first companion linearization `\boldsymbol{A} - \lambda \boldsymbol{B}`, see
    :func:`complex_frequency`, the operator of the shift-invert mode is
    `\boldsymbol{O} = (\boldsymbol{A} - \sigma \boldsymbol{B})^{-1} \boldsymbol{B}`, whose eigenvalues are `\nu =
    1/(\lambda - \sigma)`, and those of largest modulus are found. For
    `\boldsymbol{y} = \boldsymbol{O} \boldsymbol{z}`, with the halves `\boldsymbol{z}_1`, `\boldsymbol{z}_2` and `\boldsymbol{y}_1`,
    `\boldsymbol{y}_2`:

    .. math::

        \boldsymbol{y}_1 = -\boldsymbol{K}_{dyn}(\sigma)^{-1} \left( \boldsymbol{M} \boldsymbol{z}_2 + (\boldsymbol{C} + \sigma \boldsymbol{M})
        \boldsymbol{z}_1 \right), \qquad \boldsymbol{y}_2 = \boldsymbol{z}_1 + \sigma \boldsymbol{y}_1

    with `\boldsymbol{K}_{dyn}(\sigma) = \sigma^2 \boldsymbol{M} + \sigma \boldsymbol{C} + \boldsymbol{K}`, such that only the
    `n \times n` matrix `\boldsymbol{K}_{dyn}(\sigma)` is factorized, not the `2n \times 2n`
    matrix `\boldsymbol{A} - \sigma \boldsymbol{B}` ([1]_, Section 6.1, Eq. (6.3), p. 273). A
    complex shift gives a complex operator. The `\nu` with `|\nu| \le`
    ``NULL_MU_RTOL`` `\max |\nu|`, infinite eigenvalues of a singular
    `\boldsymbol{M}`, are discarded.

    References
    ----------
    .. [1] Tisseur, F. and Meerbergen, K., The quadratic eigenvalue
       problem, SIAM Review, 43(2), 235-286, 2001.
       https://doi.org/10.1137/S0036144500381988

    """
    n = K.shape[0]
    Kdyn = csc_matrix(sigma**2*M + sigma*C + K)
    try:
        lu = _splu(Kdyn)
    except RuntimeError:
        raise RuntimeError('lambda**2*M + lambda*C + K is singular at the '
                           'shift sigma=%r' % sigma)
    CsM = csr_matrix(C + sigma*M)
    dtype = np.result_type(Kdyn.dtype, float)

    def matvec(z):
        z = np.ravel(z)
        z1, z2 = z[:n], z[n:]
        rhs = np.asarray(M @ z2 + CsM @ z1)
        if np.iscomplexobj(Kdyn.data):
            y1 = -lu.solve(np.ascontiguousarray(rhs, dtype=complex))
        else:
            y1 = -_lu_solve(lu, rhs)
        return np.concatenate((y1, z1 + sigma*y1))

    OP = LinearOperator((2*n, 2*n), matvec=matvec, dtype=dtype)
    kwargs = start_kwargs(2*n)
    kwargs['v0'] = np.asarray(kwargs['v0'], dtype=dtype)
    nu, Z = eigs(OP, k=k, which='LM', tol=tol, **kwargs)
    nonnull = np.abs(nu) > NULL_MU_RTOL*np.abs(nu).max()
    nu, Z = nu[nonnull], Z[:, nonnull]
    lam = sigma + 1./nu
    return lam, _companion_vectors(K, C, M, lam, Z)


def _check_qep(K, C, M, lam, U, rtol):
    r"""Verify the eigenpairs of the quadratic eigenvalue problem

    Parameters
    ----------
    K, C, M : sparse_matrix
        Stiffness, damping and mass matrices, shape ``(n, n)``.
    lam : array_like
        Finite eigenvalues, 1-D array of size ``k``.
    U : array_like
        Eigenvectors, shape ``(n, k)``.
    rtol : float
        Largest acceptable relative residual, dimensionless.

    Returns
    -------
    error : str or None
        Description of the check that failed, ``None`` when all passed.

    Notes
    -----
    The relative residual of every eigenpair, see
    :func:`_quadratic_residuals` with ``rigid_rtol=RIGID_BODY_RTOL``, must
    not be larger than ``rtol``, unless its normwise backward error, see
    :func:`_quadratic_backward_errors`, is not larger than
    ``BACKWARD_ERROR_RTOL``, the machine precision. Whether an eigenvalue
    nearer the shift was missed is not verified, see
    :func:`complex_frequency`.

    """
    lam = np.asarray(lam)
    if not lam.size:
        return 'no eigenpair was found'
    U = np.asarray(U)
    res = _quadratic_residuals(K, C, M, lam, U, rigid_rtol=RIGID_BODY_RTOL)
    backward_error = _quadratic_backward_errors(K, C, M, lam, U)
    failed = ~((res <= rtol) | (backward_error <= BACKWARD_ERROR_RTOL))
    if np.any(failed):
        i = int(np.argmax(np.where(failed, np.nan_to_num(res, nan=np.inf),
                                   -np.inf)))
        return ('relative residual %.2e > %.1e and backward error %.1e > '
                '%.1e for the eigenvalue lambda=%r'
                % (res[i], rtol, backward_error[i], BACKWARD_ERROR_RTOL,
                   lam[i]))
    return None


def complex_frequency(K, C, M, num_eigvalues=6, tol=0, sigma=None,
                      precondition=True, check=True, check_rtol=1.e-6,
                      sparse_solver=True, max_dense_size=2000, silent=True):
    r"""Complex eigenvalues of a damped system from the stiffness, damping
    and mass matrices

    Solves the quadratic eigenvalue problem of the free vibrations of a
    damped system, ``(lam**2*M + lam*C + K) @ u = 0``, whose eigenvalues
    ``lam`` give the damped circular frequencies ``omegad = abs(lam.imag)``
    and the damping ratios ``zeta = -lam.real/abs(lam)``. ``K``, ``C`` and
    ``M`` must be real and may be unsymmetric, e.g. ``K`` with the
    aerodynamic matrices of the piston theory and ``C`` with its damping
    matrix, or ``C`` with gyroscopic terms. ``K`` must have no null row or
    column, see :func:`.check_null_rows_cols`, whereas ``C`` and ``M`` may
    have null rows and columns. The matrices are optionally preconditioned
    with :func:`diagonal_preconditioner`.

    The problem is linearized into a generalized eigenvalue problem of size
    ``2*n``, the first companion form, solved with
    :func:`scipy.sparse.linalg.eigs` in shift-invert mode, which finds the
    eigenvalues nearest the shift ``sigma`` and factorizes the ``n`` by
    ``n`` matrix ``sigma**2*M + sigma*C + K``, see :func:`_eigs_qep`. The
    eigenvalues of real matrices are real or come in complex conjugate
    pairs, and one eigenvalue per pair, the one with a positive imaginary
    part, is returned, with its eigenvector. The ``num_eigvalues``
    eigenvalues nearest the shift are returned, sorted by increasing
    imaginary part, then by increasing modulus, see :func:`_qep_modes`;
    the real ones, e.g. of overdamped modes, come first.

    The relative residual of each eigenpair must not be larger than
    ``check_rtol``, unless its backward error is at the round-off, see
    :func:`_check_qep`. When the verification fails or
    the sparse solver raises an error, the dense solver of
    :func:`_dense_qep`, :func:`scipy.linalg.eig` on the linearization, is
    used as a last resort, with a :class:`DenseFallbackWarning`, see the
    solver policy of :mod:`pyfe3d.solver.eigen`. Whether an eigenvalue
    nearer the shift than the returned ones was missed is not verified.

    With ``C`` null, the eigenvalues are ``1j*omegan``, with the natural
    frequencies ``omegan`` of :func:`natural_frequency`.

    Parameters
    ----------
    K, C, M : sparse_matrix
        Stiffness, damping and mass matrices of the free
        degrees-of-freedom, i.e. with the constrained ones partitioned out,
        e.g. ``KC0uu = KC0[bu, :][:, bu]``. ``K`` should include the
        constant terms that apply, e.g. the geometric stiffness matrix of a
        preload. Real, shape ``(n, n)``.
    num_eigvalues : int, optional
        Number of eigenvalues to compute, one per complex conjugate pair.
        Default ``6``.
    tol : float, optional
        Tolerance passed to :func:`scipy.sparse.linalg.eigs`, ``0`` meaning
        machine precision. Default ``0``.
    sigma : float, complex or None, optional
        Shift, in rad/s, near which the eigenvalues are sought; it must not
        be an eigenvalue. ``None`` uses ``1.``, which gives the eigenvalues
        of lowest modulus, see the Notes. Default ``None``.
    precondition : bool, optional
        Whether to equilibrate the diagonal of ``K`` before solving, the
        same transformation being applied to ``C`` and ``M``. Default
        ``True``.
    check : bool, optional
        Whether to verify the eigenpairs of the sparse solver. With
        ``False`` they are returned without verification and without dense
        fallback. Default ``True``.
    check_rtol : float, optional
        Largest acceptable relative residual of the eigenpairs,
        dimensionless. An eigenpair with a larger one is also accepted when
        its normwise backward error is not larger than
        ``BACKWARD_ERROR_RTOL``, the machine precision, see
        :func:`_quadratic_backward_errors`. Default ``1.e-6``.
    sparse_solver : bool, optional
        With ``False``, the full problem is solved with the dense solver,
        returning all the finite eigenvalues, one per complex conjugate
        pair, without verification. Default ``True``.
    max_dense_size : int, optional
        Maximum size of the dense problem solved as a last resort, i.e. of
        the linearization, ``2*n``; ``0`` disables the dense fallback.
        Default ``2000``.
    silent : bool, optional
        With ``False``, the solver used and the eigenvalues, damped circular
        frequencies and damping ratios are printed, see
        :mod:`pyfe3d.solver.logger`. Default ``True``.

    Returns
    -------
    eigvals : np.ndarray
        Complex eigenvalues ``lam``, in rad/s, with ``lam.imag >= 0``.
        1-D array of size ``k``, at most ``num_eigvalues``.
    eigvecs : np.ndarray
        Complex eigenvectors, one per column, in the same order, of unit
        Euclidean norm with the component of largest modulus real and
        positive. Shape ``(n, k)``.

    Raises
    ------
    numpy.linalg.LinAlgError
        When ``K`` has a null row or column, with the indices of these
        degrees-of-freedom, see :func:`.check_null_rows_cols`.
    ValueError
        When ``K``, ``C`` or ``M`` is complex.
    RuntimeError
        When ``check=True`` and no eigenvalue solver passed the
        verification.

    Warns
    -----
    DenseFallbackWarning
        When the sparse solver failed and the dense solver was used.

    Notes
    -----
    The problem solved is

    .. math::

        (\lambda^2 \boldsymbol{M} + \lambda \boldsymbol{C} + \boldsymbol{K}) \boldsymbol{u} = \boldsymbol{0}

    the free vibrations `\boldsymbol{q} = \boldsymbol{u} e^{\lambda t}` of `\boldsymbol{M} \ddot{\boldsymbol{q}} +
    \boldsymbol{C} \dot{\boldsymbol{q}} + \boldsymbol{K} \boldsymbol{q} = \boldsymbol{0}`. With `\lambda = -\zeta \omega_n \pm
    i \omega_d`, `\omega_d = |Im(\lambda)|` is the damped circular frequency
    and `\zeta = -Re(\lambda)/|\lambda|` the damping ratio: a single
    degree-of-freedom, `m \lambda^2 + c \lambda + k = 0`, has the roots
    `\lambda = -\zeta \omega_n \pm i \omega_n \sqrt{1 - \zeta^2}` for
    `\zeta < 1`, with `\omega_n = \sqrt{k/m} = |\lambda|` and `\zeta =
    c/(2 \sqrt{k m})`. A real `\lambda`, e.g. of an overdamped mode, gives
    `\omega_d = 0`, and a positive `Re(\lambda)` a growing motion, i.e. an
    unstable system, by flutter when `\lambda` is complex. `\zeta` is not
    defined for `\lambda = 0`. With `\boldsymbol{C} = \boldsymbol{0}`, `\lambda^2 = -\omega_n^2`
    of :func:`natural_frequency`.

    .. rubric:: Linearization

    With `\boldsymbol{z} = \{\boldsymbol{u}^\top, \lambda \boldsymbol{u}^\top\}^\top`, the problem is the
    generalized eigenvalue problem of size `2n`

    .. math::

        \begin{bmatrix} \boldsymbol{0} & \boldsymbol{I} \\ -\boldsymbol{K} & -\boldsymbol{C} \end{bmatrix} \boldsymbol{z} =
        \lambda \begin{bmatrix} \boldsymbol{I} & \boldsymbol{0} \\ \boldsymbol{0} & \boldsymbol{M} \end{bmatrix} \boldsymbol{z}

    the first companion form `\boldsymbol{A} - \lambda \boldsymbol{B}` of [1]_ (Section 3.4, Eq.
    (3.6) with `\boldsymbol{N} = \boldsymbol{I}`, p. 253): the first row is `\lambda \boldsymbol{u} =
    \lambda \boldsymbol{u}`, and the second gives the quadratic problem. It is valid
    for any `\boldsymbol{C}`, symmetric or not, singular or not, and it requires
    neither a non-singular `\boldsymbol{K}` nor a non-singular `\boldsymbol{M}`: a singular
    `\boldsymbol{M}`, e.g. a lumped mass matrix without rotary inertia, only gives
    infinite eigenvalues, `\boldsymbol{B}` being singular, which are discarded. Its
    shift-invert operator needs the factorization of the `n \times n`
    matrix `\boldsymbol{K}_{dyn}(\sigma) = \sigma^2 \boldsymbol{M} + \sigma \boldsymbol{C} + \boldsymbol{K}` only, see
    :func:`_eigs_qep`; MSC/Nastran uses this linearization with a
    shift-invert Lanczos algorithm ([1]_, Section 7.1, p. 278). The
    symmetric linearizations of symmetric `\boldsymbol{K}`, `\boldsymbol{C}` and `\boldsymbol{M}` give
    pencils that are indefinite, unless the system is overdamped ([1]_,
    Section 3.8.2, p. 258), for which the symmetric solver of ARPACK cannot
    be used, and they exclude the unsymmetric matrices of the piston
    theory. The eigenvectors `\boldsymbol{u}`
    are taken from the first or the last `n` rows of `\boldsymbol{z}`, see
    :func:`_companion_vectors`.

    .. rubric:: Shift

    The default shift `\sigma = 1` rad/s is the counterpart of the shift
    `\omega_n^2 = -1` of :func:`natural_frequency`, `\lambda^2 =
    -\omega_n^2 = 1`: with `\boldsymbol{C} = \boldsymbol{0}` the factorized matrix is the same,
    `\boldsymbol{K} + \boldsymbol{M}`, and `|\lambda - 1|^2 = 1 + \omega_n^2` for `\lambda = \pm
    i \omega_n`, such that the lowest natural frequencies are found first.
    With `\lambda = \omega_n(-\zeta \pm i \sqrt{1 - \zeta^2})`, `|\lambda -
    1|^2 = 1 + 2 \zeta \omega_n + \omega_n^2`, which increases with
    `\omega_n` for `\omega_n > -\zeta`. When `\boldsymbol{M}` is non-singular,
    `\det \boldsymbol{K}_{dyn}(\sigma) = \det \boldsymbol{M} \prod_i (\sigma - \lambda_i)` over the
    `2n` eigenvalues, such that `\boldsymbol{K}_{dyn}(1)` is singular only when `\lambda =
    1` is an eigenvalue, i.e. never for the rigid-body modes, `\lambda =
    0`, nor for a stable system, `Re(\lambda) \le 0`. A complex shift gives
    a complex operator, and the conjugate of each eigenvalue found with
    `Im(\lambda) < 0` is returned ([1]_, Section 6.1, p. 273).

    .. rubric:: Verification

    The relative residual `\|(\lambda^2 \boldsymbol{M} + \lambda \boldsymbol{C} + \boldsymbol{K}) \boldsymbol{u}\| /
    (|\lambda|^2 \|\boldsymbol{M} \boldsymbol{u}\| + |\lambda| \, \|\boldsymbol{C} \boldsymbol{u}\| + \|\boldsymbol{K} \boldsymbol{u}\|)`
    of each eigenpair is checked, the rigid-body modes being relative to the
    matrix-norm scale, see :func:`_quadratic_residuals`. Unlike
    :func:`natural_frequency`, no inertia check tells that no eigenvalue
    nearer the shift was missed: by Sylvester's law of inertia, the
    negative pivots of `\boldsymbol{K} - s \boldsymbol{M}` count the real eigenvalues below `s`
    of a symmetric pencil with a positive definite `\boldsymbol{M}`, whereas the
    eigenvalues of a damped system are complex, and the symmetric
    linearizations are definite only for overdamped systems, whose
    eigenvalues are real ([1]_, Sections 3.8.2 and 3.9, p. 258). No count
    of the eigenvalues of a quadratic eigenvalue problem inside a region of
    the complex plane is implemented, and that ARPACK found the eigenvalues
    of largest `|1/(\lambda - \sigma)|` is therefore not verified; the
    dense solution, ``sparse_solver=False``, is the reference. With `\boldsymbol{C} =
    \boldsymbol{0}` and symmetric matrices, :func:`natural_frequency` verifies it with
    the inertia check.

    A rigid-body mode, `\boldsymbol{K} \boldsymbol{u} = \boldsymbol{0}`, gives `\lambda = 0`, a double
    eigenvalue when `\boldsymbol{C} \boldsymbol{u} = \boldsymbol{0}`, which the solvers return as two
    eigenvalues null up to the round-off.

    References
    ----------
    .. [1] Tisseur, F. and Meerbergen, K., The quadratic eigenvalue
       problem, SIAM Review, 43(2), 235-286, 2001.
       https://doi.org/10.1137/S0036144500381988

    """
    check_null_rows_cols(K, 'K')
    K, C, M = csr_matrix(K), csr_matrix(C), csr_matrix(M)
    if any(np.iscomplexobj(A.data) for A in (K, C, M)):
        raise ValueError('K, C and M must be real')
    n = K.shape[0]
    if sigma is None:
        sigma = 1.
    if precondition:
        P = diagonal_preconditioner(K)
        Ks = csr_matrix(P @ K @ P)
        Cs = csr_matrix(P @ C @ P)
        Ms = csr_matrix(P @ M @ P)
    else:
        P = None
        Ks, Cs, Ms = K, C, M

    def unscale(result):
        lam, U = result
        if P is not None:
            U = _normalize_vectors(P @ U)
        return lam, U

    if not sparse_solver:
        name = 'eig() of the linearization'
        lam, U = unscale(_dense_qep(Ks, Cs, Ms, None, sigma))
    else:
        k = num_eigvalues

        def solver():
            lam, U = _eigs_qep(Ks, Cs, Ms, min(2*k, 2*n - 2), tol, sigma)
            return unscale(_qep_modes(lam, U, k, sigma))

        dense = ('eig() of the linearization',
                 lambda: unscale(_dense_qep(Ks, Cs, Ms, k, sigma)))

        def verify(lam, U):
            return _check_qep(K, C, M, lam, U, check_rtol)

        (lam, U), name = _sparse_first('Complex frequency analysis',
                [('eigs() in shift-invert mode', solver)],
                verify if check else None, dense, 2*n, max_dense_size,
                hint=' Try sparse_solver=False or a larger max_dense_size.')
    msg('Complex frequency analysis with %s, %d eigenvalue(s):'
        % (name, lam.size), silent=silent)
    for li in lam:
        zeta = -li.real/abs(li) if li != 0 else np.nan
        msg('lambda = %s, omegad = %g rad/s, zeta = %g'
            % (li, abs(li.imag), zeta), level=1, silent=silent)
    return lam, U
