r"""
Solvers and preconditioners (:mod:`pyfe3d.solver`)
==================================================

.. currentmodule:: pyfe3d.solver

Helpers for the eigenvalue problems that the element matrices of this package
are usually assembled for, namely linear buckling, with the constitutive and
geometric stiffness matrices, and natural frequency, with the constitutive
stiffness and mass matrices. Neither problem is safe to hand to
:func:`scipy.sparse.linalg.eigsh` without preparation:

- :func:`diagonal_preconditioner` equilibrates the diagonal of the matrices,
  which does not change the eigenvalues;
- :func:`estimate_cayley_sigma` estimates, from the matrices themselves, a
  safe shift for the Cayley mode used for linear buckling;
- :func:`check_eigenpairs` verifies the result: it checks the residual of
  every eigenpair and, when the constitutive stiffness matrix is positive
  definite, that no load multiplier was missed below the lowest one found,
  by testing whether ``K + s*KG`` is still positive definite just below it,
  see :func:`is_positive_definite`.

:func:`linear_buckling` and :func:`natural_frequency` put the three together
and are what most users should call.

.. note:: ARPACK, used by :func:`scipy.sparse.linalg.eigsh`, has been
    reported to return wrong eigenpairs at random, without raising, when
    linked against Intel MKL 2024.2.0 to 2025.0.0, which includes some
    Anaconda builds of SciPy. MKL 2025.0.1 or newer is not affected. The
    verification of :func:`check_eigenpairs` catches such a failure, which
    is another reason to keep it on.

Notes
-----
The eigenvalue problems are linear buckling

.. math::
    ([K_{C_0}] + \lambda [K_G])\{u\} = \{0\}

and natural frequency

.. math::
    ([K_{C_0}] - \omega_n^2 [M])\{u\} = \{0\}

The reason this module exists is that neither problem is safe to hand to
:func:`scipy.sparse.linalg.eigsh` without preparation, and the two traps are
easy to fall into and silent when they bite:

1. **The matrices are badly scaled.** A shell element mixes translations and
   rotations, so the diagonal of `[K_{C_0}]` spans many orders of magnitude,
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

The second check of :func:`check_eigenpairs`, that `[K_{C_0}] + s [K_G]` is
still positive definite just below the lowest load multiplier found, is the
only cheap way to know that the lowest mode really is the lowest, and it is
what distinguishes a converged answer from a plausible-looking one.

The mathematics of the shift estimate and of the verification follow the
implementation of ``structsolve.linear_buckling.lb`` by the same author [1]_,

    https://github.com/saullocastro/structsolve

which should be preferred when its additional solvers, such as the static
condensation of the degrees-of-freedom where `[K_G]` vanishes, are wanted.
The versions here are self-contained so that this package keeps depending
only on NumPy and SciPy.

The ``dsteqr`` routine of the Intel MKL versions in the note above returns
wrong eigenvectors for matrices larger than 32 by 32.

References
----------
.. [1] Castro, S. G. P., structsolve,
   https://github.com/saullocastro/structsolve

"""
import warnings

import numpy as np
from scipy.sparse import csc_matrix, csr_matrix, diags
from scipy.sparse.linalg import eigsh, splu

__all__ = ['diagonal_preconditioner', 'estimate_cayley_sigma',
           'is_positive_definite', 'check_eigenpairs', 'linear_buckling',
           'natural_frequency']


def diagonal_preconditioner(K, floor=1.e-30):
    r"""Jacobi preconditioner of a stiffness matrix

    Returns the diagonal matrix ``D`` with the entries
    ``1/sqrt(max(K[i, i], floor))``, such that ``D @ K @ D`` has a unit
    diagonal when the diagonal entries of ``K`` are larger than ``floor``.
    For a generalized eigenvalue problem both matrices are transformed,
    ``D @ A @ D`` and ``D @ B @ D``, which leaves the eigenvalues unchanged,
    and the eigenvectors of the original problem are recovered with
    ``u = D @ u_scaled``.

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
    D : sparse_matrix
        Diagonal preconditioner, created with :func:`scipy.sparse.diags`,
        shape ``(n, n)``.

    Notes
    -----
    The diagonal matrix `[D]` has the entries

    .. math::
        D_{ii} = \frac{1}{\sqrt{max({K_{C_0}}_{ii}, floor)}}

    such that `[D][K_{C_0}][D]` has a unit diagonal. For a generalized
    eigenvalue problem both matrices are transformed, `[D][A][D]` and
    `[D][B][D]`, which leaves the eigenvalues unchanged and maps the
    eigenvectors to `[D]^{-1}\{u\}`: `[D][A][D]\{y\} = w [D][B][D]\{y\}` is
    `[A]\{u\} = w [B]\{u\}` with `\{u\} = [D]\{y\}`, `[D]` being
    non-singular. The equilibration removes the spread of the diagonal
    entries, which usually reduces the condition number, see the notes of
    :mod:`pyfe3d.solver`.

    Examples
    --------
    >>> D = diagonal_preconditioner(Kuu)                  # doctest: +SKIP
    >>> eigvals, eigvecs = eigsh(A=D@KGuu@D, M=D@Kuu@D)   # doctest: +SKIP
    >>> eigvecs = D @ eigvecs                             # doctest: +SKIP

    """
    d = np.asarray(K.diagonal(), dtype=float)
    return diags(1.0/np.sqrt(np.maximum(d, floor)))


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
    `[K_G]\{u\} = \mu [K_{C_0}]\{u\}`, see the module documentation. The
    largest `|\mu|` is estimated with power iterations on
    `[K_{C_0}]^{-1}[K_G]`, whose ratio of norms in the `[K_{C_0}]`-norm grows
    towards it, and the result is multiplied by ``safety``.

    At each iteration `\{x\}` is normalized such that `\{x\}^T [K_{C_0}]
    \{x\} = 1`, `\{y\} = [K_{C_0}]^{-1} [K_G] \{x\}` is computed and the
    ratio of norms is `\sqrt{|\{y\}^T [K_{C_0}] \{y\}|}`; `\{y\}` becomes
    the next `\{x\}`. The estimate is the largest ratio found, and the
    iterations stop when a ratio exceeds the previous estimate by at most
    ``rel_tol`` times the ratio, or after ``max_iter`` iterations. For a
    symmetric positive definite `[K_{C_0}]` and a symmetric `[K_G]`,
    `[K_{C_0}]^{-1}[K_G]` is self-adjoint in the inner product of
    `[K_{C_0}]`, such that the ratio tends to its spectral radius
    `\max |\mu|`. At the first iteration the relative residual of the linear
    solution must not exceed ``1e-6``.

    The fallback value ``1.`` is returned when `[K_{C_0}]` is singular or not
    positive definite, or when the linear solution is inaccurate, since in
    those cases the power iteration says nothing.

    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            lu = splu(csc_matrix(K))
        x = np.random.RandomState(42).randn(K.shape[0])
        Kx = K @ x
        mu_max = 0.
        for i in range(max_iter):
            xKx = x @ Kx
            if not np.isfinite(xKx) or xKx <= 0:
                return 1.
            x = x/np.sqrt(xKx)
            rhs = KG @ x
            y = lu.solve(rhs)
            Ky = K @ y
            if i == 0:
                residual = np.linalg.norm(Ky - rhs)
                if not residual <= 1.e-6*np.linalg.norm(rhs):
                    return 1.
            ratio = np.sqrt(abs(y @ Ky))
            if not np.isfinite(ratio) or ratio == 0:
                return 1.
            converged = i > 0 and ratio - mu_max <= rel_tol*ratio
            mu_max = max(mu_max, ratio)
            if converged:
                break
            x, Kx = y, Ky
        return safety*mu_max
    except Exception:
        return 1.


def is_positive_definite(A):
    r"""Tells whether the symmetric sparse matrix ``A`` is positive definite

    ``A`` is factorized with a symmetric permutation and without row
    interchanges, and it is considered positive definite when all the pivots
    of the factorization are positive.

    The factorization uses :func:`scipy.sparse.linalg.splu` with the
    ordering ``'MMD_AT_PLUS_A'``, ``diag_pivot_thresh=0.`` and the symmetric
    mode. The result is ``False`` when the factorization fails because ``A``
    is exactly singular, when a row interchange was nevertheless made, or
    when a diagonal entry of the factor ``U`` is not positive. The symmetry
    of ``A`` is not checked.

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
    of the symmetric matrix is `[L][D_U][L]^T`, with the pivots on the
    diagonal `[D_U]` of `[U]`, and `[A]` has as many negative eigenvalues as
    `[D_U]` has negative entries.

    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            lu = splu(csc_matrix(A), permc_spec='MMD_AT_PLUS_A',
                      diag_pivot_thresh=0., options=dict(SymmetricMode=True))
    except RuntimeError:
        # NOTE exactly singular
        return False
    if not np.array_equal(lu.perm_r, lu.perm_c):
        # NOTE a zero pivot forced a row interchange
        return False
    return bool(np.all(lu.U.diagonal() > 0))


def check_eigenpairs(K, KG, eigvals, eigvecs, rtol=1.e-3, min_rel_gap=1.e-4):
    r"""Verify the eigenpairs of a linear buckling problem

    The eigenpairs, load multipliers and buckling modes, of the linear
    buckling problem of ``K`` and ``KG``, see :func:`linear_buckling`, are
    verified with two checks:

    - the relative residual of every eigenpair with a finite load multiplier
      ``lam`` and eigenvector ``u``, ``norm(K @ u + lam*(KG @ u))/(norm(K @
      u) + abs(lam)*norm(KG @ u))``, must not exceed ``rtol``;
    - no load multiplier is missing below the lowest positive one returned,
      ``lam1``: when ``K`` is positive definite, ``K + s*KG`` with
      ``s = lam1*(1 - min_rel_gap)`` must also be positive definite.

    The eigenpairs with a non-finite load multiplier are skipped. When no
    load multiplier is finite, the check fails at once. The second check is
    skipped when no load multiplier is positive or when ``K`` is not
    positive definite, see :func:`is_positive_definite`. Nothing is raised
    or printed; the result tells which check failed.

    Parameters
    ----------
    K, KG : sparse_matrix
        Constitutive and geometric stiffness matrices. Shape ``(n, n)``.
    eigvals : array_like
        Load multipliers, dimensionless. 1-D array of size ``k``.
    eigvecs : array_like
        Eigenvectors, one per column. Shape ``(n, k)``.
    rtol : float, optional
        Largest acceptable relative residual, dimensionless. Default
        ``1.e-3``.
    min_rel_gap : float, optional
        Relative margin below the lowest positive load multiplier at which
        the completeness of the spectrum is tested, dimensionless. Default
        ``1.e-4``.

    Returns
    -------
    error : str or None
        Description of the check that failed, or ``None`` when all passed.

    See Also
    --------
    linear_buckling

    Notes
    -----
    The eigenpairs are those of `([K_{C_0}] + \lambda [K_G])\{u\} = \{0\}`,
    and the two checks are:

    - the relative residual
      `||K u + \lambda K_G u||/(||K u|| + |\lambda| ||K_G u||)` of every
      eigenpair with a finite `\lambda` must not exceed ``rtol``;
    - no load multiplier is missing below the lowest positive `\lambda_1`
      returned: when `[K_{C_0}]` is positive definite,
      `[K_{C_0}] + s [K_G]` with `s = \lambda_1 (1 - min\_rel\_gap)` must also
      be positive definite.

    The second check is the one that catches a solver that converged to an
    interior part of the spectrum, which is silent otherwise. With
    `[K_{C_0}]` positive definite, the eigenvalues of `[K_{C_0}] + s [K_G]`
    relative to `[K_{C_0}]` are `1 + s \mu = 1 - s/\lambda`, where `\mu =
    -1/\lambda` are the eigenvalues of `[K_G]\{u\} = \mu [K_{C_0}]\{u\}`,
    and by Sylvester's law of inertia the number of negative eigenvalues of
    `[K_{C_0}] + s [K_G]` is the number of load multipliers `\lambda` in
    `(0, s)`. The matrix is therefore positive definite when no load
    multiplier lies in `(0, s]`.

    """
    eigvals = np.asarray(eigvals, dtype=float)
    eigvecs = np.asarray(eigvecs, dtype=float)
    finite = np.isfinite(eigvals)
    if not np.any(finite):
        return 'no finite load multiplier was found'
    lam = eigvals[finite]
    u = eigvecs[:, finite]
    Ku = K @ u
    KGu = KG @ u
    num = np.linalg.norm(Ku + KGu*lam, axis=0)
    den = np.linalg.norm(Ku, axis=0) + np.abs(lam)*np.linalg.norm(KGu, axis=0)
    residual = np.full(lam.shape, np.inf)
    np.divide(num, den, out=residual, where=den > 0)
    if not np.all(residual <= rtol):
        i = int(np.argmax(np.nan_to_num(residual, nan=np.inf)))
        return ('relative residual %.2e > %.1e for the load multiplier %r'
                % (residual[i], rtol, lam[i]))
    positive = lam[lam > 0]
    if positive.size and is_positive_definite(K):
        s = positive.min()*(1 - min_rel_gap)
        if not is_positive_definite(csr_matrix(K) + s*csr_matrix(KG)):
            return ('at least one load multiplier lower than %r is missing'
                    % positive.min())
    return None


def linear_buckling(K, KG, num_eigvalues=6, tol=0, sigma=None,
                    precondition=True, check=True, check_rtol=1.e-3):
    r"""Linear buckling analysis with a verified spectrum

    Computes the load multipliers and the buckling modes of the linear
    buckling problem of ``K`` and ``KG``, the lowest positive load
    multipliers being sought, with
    :func:`scipy.sparse.linalg.eigsh` in Cayley mode, using the shift of
    :func:`estimate_cayley_sigma`, the preconditioner of
    :func:`diagonal_preconditioner` and the verification of
    :func:`check_eigenpairs`.

    ``K`` and ``KG`` are converted to CSR matrices. The number of computed
    load multipliers is ``k = min(num_eigvalues, n - 2)``. The eigenvalues
    ``mu`` returned by ``eigsh(A=KG, M=K, which='SM', mode='cayley')`` are
    converted into the load multipliers ``-1/mu``, a zero ``mu`` giving an
    infinite load multiplier. With
    ``precondition=True`` the eigenvectors are transformed back to the
    original degrees-of-freedom, and the verification always uses the
    original ``K`` and ``KG``.

    Parameters
    ----------
    K, KG : sparse_matrix
        Constitutive and geometric stiffness matrices, with the boundary
        conditions already applied, i.e. only the free degrees-of-freedom.
        Shape ``(n, n)``.
    num_eigvalues : int, optional
        Number of load multipliers to compute. Default ``6``.
    tol : float, optional
        Tolerance passed to :func:`scipy.sparse.linalg.eigsh`, ``0`` meaning
        machine precision. Default ``0``.
    sigma : float or None, optional
        Shift of the Cayley mode. With ``None`` it is estimated from the
        matrices, which is the recommended use. Default ``None``.
    precondition : bool, optional
        Whether to equilibrate the diagonal before solving. This does not
        change the eigenvalues. Default ``True``.
    check : bool, optional
        Whether to verify the eigenpairs and raise on failure. Default
        ``True``.
    check_rtol : float, optional
        Largest acceptable relative residual of the eigenpairs,
        dimensionless. Default ``1.e-3``.

    Returns
    -------
    eigvals : np.ndarray
        Load multipliers, the positive ones first in increasing
        order, followed by the negative ones. Dimensionless, 1-D array of
        size ``k``.
    eigvecs : np.ndarray
        Eigenvectors, one per column, in the same order. Shape ``(n, k)``.

    Raises
    ------
    RuntimeError
        When ``check=True`` and the verification fails.

    Notes
    -----
    The problem solved is `([K_{C_0}] + \lambda [K_G])\{u\} = \{0\}`, written
    as `[K_G]\{u\} = \mu [K_{C_0}]\{u\}` with `\lambda = -1/\mu`, whose lowest
    positive load multipliers are the most negative `\mu`, returned by the
    Cayley mode while the shift is large enough, see the notes of
    :mod:`pyfe3d.solver`. With ``precondition=True`` the problem solved is
    the one of `[D][K_{C_0}][D]` and `[D][K_G][D]`, with the same
    eigenvalues, see :func:`diagonal_preconditioner`.

    A failure of the verification usually means that the mesh, and not the
    solver, needs attention: a spectrum with many nearly coincident
    eigenvalues, as in the buckling of a cylindrical shell, is the typical
    case.

    """
    K = csr_matrix(K)
    KG = csr_matrix(KG)
    if precondition:
        D = diagonal_preconditioner(K)
        Ks = D @ K @ D
        KGs = D @ KG @ D
    else:
        D = None
        Ks, KGs = K, KG
    if sigma is None:
        sigma = estimate_cayley_sigma(Ks, KGs)
    k = min(num_eigvalues, Ks.shape[0] - 2)
    mu, eigvecs = eigsh(A=KGs, k=k, which='SM', M=Ks, tol=tol, sigma=sigma,
                        mode='cayley')
    if D is not None:
        eigvecs = np.asarray(D @ eigvecs)
    with np.errstate(divide='ignore'):
        eigvals = -1./mu
    # NOTE positive load multipliers first, in increasing order
    order = np.argsort(np.where(eigvals > 0, eigvals, np.inf), kind='stable')
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    if check:
        error = check_eigenpairs(K, KG, eigvals, eigvecs, rtol=check_rtol)
        if error is not None:
            raise RuntimeError('linear buckling: ' + error)
    return eigvals, eigvecs


def natural_frequency(K, M, num_eigvalues=6, tol=0, precondition=True):
    r"""Natural frequencies from the stiffness and mass matrices

    ``K`` and ``M`` are converted to CSR matrices, optionally preconditioned
    with :func:`diagonal_preconditioner`, and the eigenvalues, the squares of
    the circular frequencies, are computed with
    :func:`scipy.sparse.linalg.eigsh` in
    shift-invert mode with ``sigma=-1.`` and ``which='LM'``. The number of
    computed frequencies is ``k = min(num_eigvalues, n - 2)``. The
    eigenpairs are sorted by increasing eigenvalue, the eigenvectors are
    transformed back to the original degrees-of-freedom when
    ``precondition=True``, and negative eigenvalues give a zero frequency.
    Unlike :func:`linear_buckling`, the eigenpairs are not verified.

    Parameters
    ----------
    K, M : sparse_matrix
        Constitutive stiffness and mass matrices, with the boundary
        conditions already applied. Shape ``(n, n)``.
    num_eigvalues : int, optional
        Number of frequencies to compute. Default ``6``.
    tol : float, optional
        Tolerance passed to :func:`scipy.sparse.linalg.eigsh`, ``0`` meaning
        machine precision. Default ``0``.
    precondition : bool, optional
        Whether to equilibrate the diagonal of ``K`` before solving.
        Default ``True``.

    Returns
    -------
    omegan : np.ndarray
        Circular frequencies in rad/s, in increasing order. 1-D array of size
        ``k``.
    eigvecs : np.ndarray
        Eigenvectors, one per column. Shape ``(n, k)``.

    Notes
    -----
    The problem solved is `([K_{C_0}] - \omega_n^2 [M])\{u\} = \{0\}`, and
    the frequencies are `\omega_n = \sqrt{\max(\omega_n^2, 0)}`. In the
    shift-invert mode, :func:`scipy.sparse.linalg.eigsh` finds the
    eigenvalues closest to the shift ``sigma=-1.``, i.e. the lowest ones
    when `[K_{C_0}]` is positive semi-definite and `[M]` positive definite,
    factorizing `[K_{C_0}] + [M]`, which is not singular when `[K_{C_0}]` is
    singular only because of unconstrained rigid-body modes that have mass.

    A mass matrix without rotary inertia, and one without inertia associated
    with the drilling rotation, is singular, so the eigenvalues associated
    with those degrees-of-freedom are infinite. The shift-invert mode used
    here looks for the eigenvalues closest to zero and is not troubled by
    them, but a lumped mass matrix will place the massless degrees-of-freedom
    wherever the factorization puts them, so prefer a consistent mass matrix
    when the lowest frequencies matter.

    """
    K = csr_matrix(K)
    M = csr_matrix(M)
    if precondition:
        D = diagonal_preconditioner(K)
        Ks = D @ K @ D
        Ms = D @ M @ D
    else:
        D = None
        Ks, Ms = K, M
    k = min(num_eigvalues, Ks.shape[0] - 2)
    eigvals, eigvecs = eigsh(A=Ks, k=k, M=Ms, sigma=-1., which='LM', tol=tol)
    if D is not None:
        eigvecs = np.asarray(D @ eigvecs)
    order = np.argsort(eigvals)
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    omegan = np.sqrt(np.maximum(eigvals, 0.))
    return omegan, eigvecs
