r"""
Solvers and preconditioners (:mod:`pyfe3d.solver`)
==================================================

.. currentmodule:: pyfe3d.solver

Helpers for the eigenvalue problems that the element matrices of this package
are usually assembled for, namely linear buckling

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

On top of that, :func:`check_eigenpairs` verifies the result: it checks the
residual of every eigenpair and, when `[K_{C_0}]` is positive definite, that
no load multiplier was missed below the lowest one found, by testing whether
`[K_{C_0}] + s [K_G]` is still positive definite just below it. That second
check is the only cheap way to know that the lowest mode really is the lowest,
and it is what distinguishes a converged answer from a plausible-looking one.

:func:`linear_buckling` and :func:`natural_frequency` put the three together
and are what most users should call.

The mathematics of the shift estimate and of the verification follow the
implementation of ``structsolve.linear_buckling.lb`` by the same author,

    https://github.com/saullocastro/structsolve

which should be preferred when its additional solvers, such as the static
condensation of the degrees-of-freedom where `[K_G]` vanishes, are wanted.
The versions here are self-contained so that this package keeps depending
only on NumPy and SciPy.

.. note:: ARPACK, used by :func:`scipy.sparse.linalg.eigsh`, has been
    reported to return wrong eigenpairs at random, without raising, when
    linked against Intel MKL 2024.2.0 to 2025.0.0, which includes some
    Anaconda builds of SciPy. The ``dsteqr`` routine of those versions
    returns wrong eigenvectors for matrices larger than 32 by 32. MKL
    2025.0.1 or newer is not affected. The verification of
    :func:`check_eigenpairs` catches such a failure, which is another reason
    to keep it on.

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

    Returns the diagonal matrix `[D]` with

    .. math::
        D_{ii} = \frac{1}{\sqrt{max({K_{C_0}}_{ii}, floor)}}

    such that `[D][K_{C_0}][D]` has a unit diagonal. For a generalized
    eigenvalue problem both matrices are transformed, `[D][A][D]` and
    `[D][B][D]`, which leaves the eigenvalues unchanged and maps the
    eigenvectors to `[D]^{-1}\{u\}`, so the eigenvectors of the original
    problem are recovered with ``u = D @ u_scaled``.

    Parameters
    ----------
    K : sparse_matrix
        Matrix whose diagonal sets the scaling, normally the constitutive
        stiffness matrix with the boundary conditions already applied.
    floor : float, optional
        Lower bound for the diagonal entries, which guards against the null
        diagonal of an unconstrained degree-of-freedom.

    Returns
    -------
    D : sparse_matrix
        Diagonal preconditioner.

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

    The shift has to exceed `|\mu|` of the critical eigenvalues of
    `[K_G]\{u\} = \mu [K_{C_0}]\{u\}`, see the module documentation. The
    largest `|\mu|` is estimated with power iterations on
    `[K_{C_0}]^{-1}[K_G]`, whose ratio of norms in the `[K_{C_0}]`-norm grows
    towards it, and the result is multiplied by ``safety``.

    The fallback value ``1.`` is returned when `[K_{C_0}]` is singular or not
    positive definite, or when the linear solution is inaccurate, since in
    those cases the power iteration says nothing.

    Parameters
    ----------
    K, KG : sparse_matrix
        Constitutive and geometric stiffness matrices, with the boundary
        conditions already applied.
    safety : float, optional
        Multiplier applied to the estimated largest `|\mu|`.
    max_iter : int, optional
        Maximum number of power iterations.
    rel_tol : float, optional
        Relative increment below which the power iteration is stopped.

    Returns
    -------
    sigma : float
        Shift to be passed to ``eigsh(..., sigma=sigma, mode='cayley')``.

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
    interchanges. The Gaussian elimination of a positive definite matrix
    never meets a zero or a negative pivot, and by Sylvester's law of inertia
    a negative pivot means a negative eigenvalue.

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
    r"""Verify the eigenpairs of `([K_{C_0}] + \lambda [K_G])\{u\} = \{0\}`

    Two checks are made:

    - the relative residual
      `||K u + \lambda K_G u||/(||K u|| + |\lambda| ||K_G u||)` of every
      eigenpair with a finite `\lambda` must not exceed ``rtol``;
    - no load multiplier is missing below the lowest positive `\lambda_1`
      returned: when `[K_{C_0}]` is positive definite,
      `[K_{C_0}] + s [K_G]` with `s = \lambda_1 (1 - min\_rel\_gap)` must also
      be positive definite.

    The second check is the one that catches a solver that converged to an
    interior part of the spectrum, which is silent otherwise.

    Parameters
    ----------
    K, KG : sparse_matrix
        Constitutive and geometric stiffness matrices.
    eigvals : array_like
        Load multipliers `\lambda`.
    eigvecs : array_like
        Eigenvectors, one per column.
    rtol : float, optional
        Largest acceptable relative residual.
    min_rel_gap : float, optional
        Relative margin below `\lambda_1` at which the completeness of the
        spectrum is tested.

    Returns
    -------
    error : str or None
        Description of the check that failed, or ``None`` when all passed.

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

    Solves `([K_{C_0}] + \lambda [K_G])\{u\} = \{0\}` with
    :func:`scipy.sparse.linalg.eigsh` in Cayley mode, using the shift of
    :func:`estimate_cayley_sigma`, the preconditioner of
    :func:`diagonal_preconditioner` and the verification of
    :func:`check_eigenpairs`.

    Parameters
    ----------
    K, KG : sparse_matrix
        Constitutive and geometric stiffness matrices, with the boundary
        conditions already applied, i.e. only the free degrees-of-freedom.
    num_eigvalues : int, optional
        Number of load multipliers to compute.
    tol : float, optional
        Tolerance passed to :func:`scipy.sparse.linalg.eigsh`, ``0`` meaning
        machine precision.
    sigma : float or None, optional
        Shift of the Cayley mode. With ``None`` it is estimated from the
        matrices, which is the recommended use.
    precondition : bool, optional
        Whether to equilibrate the diagonal before solving. This does not
        change the eigenvalues.
    check : bool, optional
        Whether to verify the eigenpairs and raise on failure.
    check_rtol : float, optional
        Largest acceptable relative residual of the eigenpairs.

    Returns
    -------
    eigvals : np.ndarray
        Load multipliers `\lambda`, the positive ones first in increasing
        order, followed by the negative ones.
    eigvecs : np.ndarray
        Eigenvectors, one per column, in the same order.

    Raises
    ------
    RuntimeError
        When ``check=True`` and the verification fails, which usually means
        that the mesh, and not the solver, needs attention: a spectrum with
        many nearly coincident eigenvalues, as in the buckling of a
        cylindrical shell, is the typical case.

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
    r"""Natural frequencies of `([K_{C_0}] - \omega_n^2 [M])\{u\} = \{0\}`

    Parameters
    ----------
    K, M : sparse_matrix
        Constitutive stiffness and mass matrices, with the boundary
        conditions already applied.
    num_eigvalues : int, optional
        Number of frequencies to compute.
    tol : float, optional
        Tolerance passed to :func:`scipy.sparse.linalg.eigsh`.
    precondition : bool, optional
        Whether to equilibrate the diagonal of `[K_{C_0}]` before solving.

    Returns
    -------
    omegan : np.ndarray
        Circular frequencies in rad/s, in increasing order.
    eigvecs : np.ndarray
        Eigenvectors, one per column.

    Notes
    -----
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
