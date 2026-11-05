Eigenvalue solvers and their verification
=========================================

The eigenvalue problems of :doc:`eigenvalue_analyses` are large and sparse,
and only a few eigenvalues are wanted: the lowest positive load
multipliers, the lowest frequencies, the eigenvalues of a damped system
nearest a shift. The iterative solvers of ARPACK, through
:func:`scipy.sparse.linalg.eigsh` and :func:`scipy.sparse.linalg.eigs`,
find them from one factorization, but they return the eigenvalues nearest
a shift, not necessarily the lowest ones, may miss copies of repeated
eigenvalues, and do not raise when they do. :mod:`pyfe3d.solver.eigen`
therefore transforms each problem into one that the sparse solvers solve
reliably, verifies every result, and uses a dense solver only as a last
resort. This page describes the transformations, the verifications and the
solver policy shared by :func:`.linear_buckling`,
:func:`.natural_frequency` and :func:`.complex_frequency`.


Equilibration of the diagonal
-----------------------------

The matrices of beam and shell models mix translations and rotations, such
that the diagonal of `\boldsymbol{K}_{C0}` spans many orders of magnitude; the
fictitious drilling stiffness of the shells widens the spread further. With
``precondition=True``, the default, the problems are solved with the
matrices `\boldsymbol{P} \boldsymbol{A} \boldsymbol{P}` and `\boldsymbol{P} \boldsymbol{B} \boldsymbol{P}` of the
Jacobi scaling

.. math::
    :label: es-jacobi

    P_{ii} = \frac{1}{\sqrt{\max(|K_{ii}|, \mathrm{floor})}}

of :func:`.diagonal_preconditioner`, whose `\boldsymbol{P} \boldsymbol{K} \boldsymbol{P}` has a
unit diagonal up to the signs. Since `\boldsymbol{P}` is non-singular, `\boldsymbol{P}
\boldsymbol{A} \boldsymbol{P} \boldsymbol{y} = w \boldsymbol{P} \boldsymbol{B} \boldsymbol{P} \boldsymbol{y}` is `\boldsymbol{A}
\boldsymbol{u} = w \boldsymbol{B} \boldsymbol{u}` with `\boldsymbol{u} = \boldsymbol{P} \boldsymbol{y}`: the eigenvalues
are unchanged, the eigenvectors are transformed back, and the verification
always uses the original matrices. The equilibration usually reduces the
condition number and often decides whether the iterative solver converges.


Linear buckling: the Cayley mode
--------------------------------

:func:`.linear_buckling` writes :eq:`ev-buckling` as

.. math::
    :label: es-mu

    \boldsymbol{K}_G \boldsymbol{u} = \mu \boldsymbol{K}_{C0} \boldsymbol{u}, \qquad \lambda = -\frac{1}{\mu}

whose lowest positive load multipliers are the most negative `\mu`. For
symmetric matrices with a positive definite `\boldsymbol{K}_{C0}`, it calls
:func:`scipy.sparse.linalg.eigsh` in the Cayley mode, in which ARPACK
computes the eigenvalues `(\mu + \sigma)/(\mu - \sigma)` of the operator
`(\boldsymbol{K}_G - \sigma \boldsymbol{K}_{C0})^{-1}(\boldsymbol{K}_G + \sigma \boldsymbol{K}_{C0})`
(Lehoucq et al. 1998, Appendix A, Sec. A.1), and ``which='SM'`` selects
those of smallest modulus. When the shift `\sigma` is larger than `|\mu|`
of all the eigenvalues, `(\mu + \sigma)/(\mu - \sigma)` increases in modulus
with `\mu` in `(-\sigma, \sigma)`, and the most negative `\mu`, i.e. the
lowest positive load multipliers, come first; otherwise the eigenvalues
closest to `-\sigma` come back instead, without warning, see
:func:`pyfe3d.solver.eigen._eigsh_cayley`. The shift is therefore estimated
from the matrices by :func:`.estimate_cayley_sigma`: power iterations on
`\boldsymbol{K}_{C0}^{-1} \boldsymbol{K}_G`, self-adjoint in the inner product of
`\boldsymbol{K}_{C0}`, whose ratio of `\boldsymbol{K}_{C0}`-norms grows towards the
spectral radius `\max|\mu|`, multiplied by a safety factor of 10. The
second sparse attempt uses a 10 times larger shift.

The negative load multipliers, those of the reversed load, are the
positive `\mu`. When fewer than ``num_eigvalues`` positive multipliers
exist, the remaining ones are computed with `-\boldsymbol{K}_G`; the number
requested of each sign is limited by the count of the multipliers of that
sign, from the inertia below, since ARPACK may not converge when it is
asked for more eigenvalues of one sign than exist, see
:func:`pyfe3d.solver.eigen._eigsh_both_signs`.


Verification: residuals and the inertia count
---------------------------------------------

Every eigenpair returned by a sparse solver is verified, see
:func:`.check_eigenpairs`:

- the relative residual of each eigenpair, e.g. for buckling

  .. math::
      :label: es-residual

      \frac{\|\boldsymbol{K}_{C0} \boldsymbol{u} + \lambda \boldsymbol{K}_G \boldsymbol{u}\|}
      {\|\boldsymbol{K}_{C0} \boldsymbol{u}\| + |\lambda| \, \|\boldsymbol{K}_G \boldsymbol{u}\|}

  must not exceed ``check_rtol``, `10^{-3}` for :func:`.linear_buckling`
  and `10^{-6}` for the frequencies; for a rigid-body mode, whose residual
  and denominator are both round-off errors, the residual is taken
  relative to the matrix norms, see
  :func:`pyfe3d.solver.eigen._relative_residuals`;
- for symmetric matrices, the absence of an eigenvalue missed below the
  highest one returned, from the inertia of a shifted matrix. With a
  positive definite `\boldsymbol{K}_{C0}`, `\boldsymbol{K}_{C0} + s \boldsymbol{K}_G` is
  congruent to `\boldsymbol{I} + s \boldsymbol{K}_{C0}^{-1/2} \boldsymbol{K}_G
  \boldsymbol{K}_{C0}^{-1/2}`, whose eigenvalues are `1 - s/\lambda_i`, negative
  exactly when `0 < \lambda_i < s`: the number of negative pivots of the
  factorization of

  .. math::
      :label: es-sturm

      \boldsymbol{K}_{C0} + s \boldsymbol{K}_G \quad \text{(buckling)}, \qquad
      \boldsymbol{K} - s \boldsymbol{M} \quad \text{(frequencies)}

  is, by Sylvester's law of inertia, the number of load multipliers in
  `(0, s)`, or of the `\omega_n^2 < s`, which must equal the number
  returned below `s`, with `s` just below the highest eigenvalue returned.
  This is the Sturm sequence check of the eigenvalue solvers of structural
  analysis (Bathe 2014, Sec. 10.2.2, p. 849, Sec. 11.4.3, p. 943, and
  Sec. 11.6, p. 954), and it is what distinguishes a converged answer from
  a plausible-looking one. The factorization is that of
  :func:`.is_positive_definite`, with a symmetric permutation and without
  row interchanges.

An eigenpair of :func:`.natural_frequency` or :func:`.complex_frequency`
whose relative residual is too large still passes when its normwise
backward error

.. math::
    :label: es-backward

    \eta = \frac{\|\boldsymbol{A} \boldsymbol{u} + \lambda \boldsymbol{B} \boldsymbol{u}\|}{(\|\boldsymbol{A}\|_F + |\lambda|
    \, \|\boldsymbol{B}\|_F) \|\boldsymbol{u}\|}

is not larger than the machine precision: the eigenpair is then exact for
matrices that differ from the given ones by perturbations of the order of
the round-off of their entries (Tisseur 2000, Theorem 1, Eq. (2.3)), e.g. an
eigenvalue close to zero of a model pre-stressed close to buckling, see
:func:`pyfe3d.solver.eigen._backward_errors` and
:func:`pyfe3d.solver.eigen._quadratic_backward_errors`. The buckling
analysis keeps the relative residual only: a finite load multiplier is never
null, and a backward error at the round-off does not tell that an
eigenpair is accurate when the matrix norms are set by stiff penalty
springs.


Repeated eigenvalues
--------------------

Structures with a symmetry have eigenvalues equal up to the round-off,
e.g. the pairs of modes `\cos n\theta` and `\sin n\theta` of a cylinder with
a rotationally symmetric mesh. The Lanczos process of ARPACK, started from
a single vector, finds in exact arithmetic one eigenvector of each
eigenspace only, and the other copies only through the rounding errors.
When the inertia count finds eigenvalues missing below the highest one
returned, they are computed by deflated passes before the next solver, see
:func:`pyfe3d.solver.eigen._complete_repeated`: each pass calls
:func:`.deflated_eigsh`, the shift-invert mode with the eigenvectors already
found deflated from the operator and another starting vector, which returns
the eigenvalues nearest the shift not found yet, and the eigenvectors are
combined with a Rayleigh-Ritz projection that keeps them orthonormal; the
result is verified as any other. ``tests/test_solver_linear_buckling.py``
checks the completion on a :class:`.Quad4` cylinder under torsion.


Natural frequencies: shift-invert mode
--------------------------------------

:func:`.natural_frequency` computes the eigenvalues `\omega_n^2` nearest the
shift `\sigma = -1` in the shift-invert mode, with
:func:`scipy.sparse.linalg.eigsh` for symmetric matrices and
:func:`scipy.sparse.linalg.eigs` otherwise, factorizing `\boldsymbol{K} +
\boldsymbol{M}`: these are the lowest ones when `\boldsymbol{K}` is positive
semi-definite and `\boldsymbol{M}` positive definite, and `\boldsymbol{K} + \boldsymbol{M}` is
not singular when `\boldsymbol{K}` is singular only because of unconstrained
rigid-body modes that have mass. An indefinite `\boldsymbol{K}`, e.g. pre-stressed
beyond buckling, is accepted, its negative `\omega_n^2` giving null
frequencies. The eigenpairs are verified with the residuals, the backward
error and, for symmetric matrices, the inertia count of
:eq:`es-sturm`, see :func:`pyfe3d.solver.eigen._check_freq`.


Unsymmetric problems
--------------------

The load stiffness of a follower load makes the buckling problem
unsymmetric, with possibly complex load multipliers, many of which may have
a smaller modulus than the critical one. :func:`.linear_buckling` then
searches along the positive real axis: :func:`scipy.sparse.linalg.eigs`
finds the multipliers nearest a shift `s_j` within a radius `r_j`; the next
shift is placed beyond the interval of the real axis already covered, until
the lowest positive real multiplier found is not beyond the covered
interval, see :func:`pyfe3d.solver.eigen._eigs_shift_search`. A parity
check completes the verification of the residuals:

.. math::
    :label: es-parity

    \det(\boldsymbol{K} + s \boldsymbol{K}_G) = \det \boldsymbol{K} \prod_i (1 + s \mu_i)

where a complex conjugate pair gives the positive factor `|1 + s
\mu_i|^2` and a real `\mu_i` a negative factor exactly when `0 <
\lambda_i < s`, so `\det(\boldsymbol{K} + s \boldsymbol{K}_G)` just below the critical
multiplier must have the sign of `\det \boldsymbol{K}`. The signs come from the LU
factorizations. This check cannot detect two missing real multipliers, nor
a missing complex pair.


Damped frequencies: linearization
---------------------------------

:func:`.complex_frequency` writes the quadratic problem :eq:`ev-qep`, with
`\boldsymbol{z} = \{\boldsymbol{u}^\top, \lambda \boldsymbol{u}^\top\}^\top`, as the
generalized eigenvalue problem of size `2n`

.. math::
    :label: es-companion

    \begin{bmatrix} \boldsymbol{0} & \boldsymbol{I} \\ -\boldsymbol{K} & -\boldsymbol{C} \end{bmatrix} \boldsymbol{z} =
    \lambda \begin{bmatrix} \boldsymbol{I} & \boldsymbol{0} \\ \boldsymbol{0} & \boldsymbol{M} \end{bmatrix} \boldsymbol{z}

the first companion form of Tisseur and Meerbergen (2001, Sec. 3.4, Eq.
(3.6)), valid for any `\boldsymbol{C}`, symmetric or not, and requiring neither a
non-singular `\boldsymbol{K}` nor a non-singular `\boldsymbol{M}`, whose singularity
only gives infinite eigenvalues, discarded. Its shift-invert operator needs
the factorization of the `n \times n` matrix `\boldsymbol{K}_{dyn}(\sigma) =
\sigma^2 \boldsymbol{M} + \sigma \boldsymbol{C} + \boldsymbol{K}` only (Sec. 6.1, Eq. (6.3)), see
:func:`pyfe3d.solver.eigen._eigs_qep`. The default shift `\sigma = 1` rad/s
is the counterpart of the shift `-1` of :func:`.natural_frequency`: with
`\boldsymbol{C} = \boldsymbol{0}` the factorized matrix is the same, `\boldsymbol{K} +
\boldsymbol{M}`, and, for `\lambda = \omega_n(-\zeta \pm i \sqrt{1 -
\zeta^2})`, `|\lambda - 1|^2 = 1 + 2 \zeta \omega_n + \omega_n^2` grows with
`\omega_n` for `\omega_n > -\zeta`, such that the lowest frequencies are found first. The
symmetric linearizations of symmetric matrices give indefinite pencils,
unless the system is overdamped (Sec. 3.8.2), and exclude the unsymmetric
matrices of the piston theory. The residuals and the backward errors are
verified, but no inertia count applies to the complex eigenvalues of a
damped system: that no eigenvalue nearer the shift was missed is not
verified, and the dense solution, ``sparse_solver=False``, is the
reference.


Solver policy and dense fallback
--------------------------------

With ``sparse_solver=True``, the default, the sparse solvers run first for
every problem size, and each result must pass the verification. Only when
all of them failed, e.g. a failed verification, an ARPACK error or a
singular factorization, a dense solver of SciPy is used, if the dense
problem has at most ``max_dense_size`` dofs, 2000 by default, with a
:class:`.DenseFallbackWarning` giving the reasons of the failures; its
result is verified as well. Otherwise a ``RuntimeError`` lists every
attempted solver, such that no unverified result is returned silently, see
:func:`pyfe3d.solver.eigen._sparse_first`. The dense fallback of the
buckling problem condenses out the dofs where ``KG`` has null rows, see
:func:`pyfe3d.solver.eigen._condense`. ``sparse_solver=False`` solves the
full problem with a dense solver and returns all the eigenvalues, without
verification, as a reference for validations, and ``check=False`` returns
the first sparse result without verification. The fallback can be turned
into an error, e.g. in a test suite, with
``warnings.simplefilter('error', DenseFallbackWarning)``.

For symmetric matrices, :func:`.linear_buckling` requires a positive
definite `\boldsymbol{K}_{C0}`, the matrix of its Cayley mode and of its inertia
count: with ``check=True`` a negative or a zero pivot raises a
:class:`numpy.linalg.LinAlgError` naming the likely causes, insufficient
boundary conditions, a mechanism, distorted elements or a preload beyond
the buckling load, see :doc:`boundary_conditions`. When the solvers of
:func:`.natural_frequency` fail and `\boldsymbol{K}` is not positive definite,
its ``RuntimeError`` starts with the same diagnosis.

A dof without stiffness, a null row or column of `\boldsymbol{K}`, raises a
:class:`numpy.linalg.LinAlgError` naming the dofs, see
:func:`.check_null_rows_cols`: with the same null row of `\boldsymbol{K}_G` or
`\boldsymbol{M}` the pencil is singular, every value being an eigenvalue, and
otherwise it gives a spurious null eigenvalue. `\boldsymbol{K}_G`, `\boldsymbol{C}` and
`\boldsymbol{M}` may have null rows, e.g. a lumped mass matrix without rotary
inertia. :func:`.is_symmetric` selects the solvers of symmetric or
unsymmetric matrices when ``symmetric=None``.


ARPACK and Intel MKL
--------------------

With Intel MKL 2024.2.0 to 2025.0.0, the routine ``dsteqr``, which the
symmetric drivers of ARPACK call on the tridiagonal matrix of the Lanczos
process, returns wrong eigenvectors for matrices larger than 32 by 32, and
:func:`scipy.sparse.linalg.eigsh` returns wrong eigenpairs, without error,
whenever its number of Lanczos vectors ``ncv`` exceeds 32. These versions
are detected, see :func:`.dsteqr_is_faulty` and :func:`.arpack_max_ncv`,
and ``ncv`` is then capped at 32: :func:`.capped_eigsh` computes more than
15 eigenpairs in passes with the eigenvectors already found deflated,
combined with a Rayleigh-Ritz projection, see
:mod:`pyfe3d.solver.arpackutils`, which also describes the override with
the environment variable ``PYFE3D_ARPACK_MAX_NCV``. The verification above
catches such a failure too, which is another reason to keep it on.


References
----------

- Bathe, K. J., Finite Element Procedures, 2nd ed., K. J. Bathe,
  Watertown, MA, 2014.
- Lehoucq, R. B., Sorensen, D. C. and Yang, C., ARPACK Users' Guide:
  Solution of Large-Scale Eigenvalue Problems with Implicitly Restarted
  Arnoldi Methods, SIAM, Philadelphia, 1998.
  https://doi.org/10.1137/1.9780898719628
- Tisseur, F., Backward error and condition of polynomial eigenvalue
  problems, Linear Algebra and its Applications, 309(1-3), 339-361, 2000.
  https://doi.org/10.1016/S0024-3795(99)00063-4
- Tisseur, F. and Meerbergen, K., The quadratic eigenvalue problem, SIAM
  Review, 43(2), 235-286, 2001. https://doi.org/10.1137/S0036144500381988
