r"""
=================================================================================
Convergence criteria of the non-linear solvers (:mod:`pyfe3d.solver.convergence`)
=================================================================================

.. currentmodule:: pyfe3d.solver.convergence

The equilibrium iterations of :class:`.Analysis` stop when the residual
force vector is small, measured by one of two criteria selected with
``Analysis.convergence``:

- ``'crisfield'``, the default: Crisfield's criterion, the residual scaled
  with the diagonal of the linear stiffness matrix and compared with the
  largest scaled norm of the reference force vectors, the external and the
  internal ones and, when they apply, the external force vector of the
  increment of the arc-length solvers and the load factor times
  `\partial R/\partial \lambda`, over the free degrees-of-freedom only, see
  :func:`crisfield_zeta`;
- ``'relative'``: the Euclidean norm of the residual compared with the
  largest Euclidean norm of the force vectors, ``Analysis.relTOL``, or the
  largest absolute residual force, ``Analysis.absTOL``.

Notes
-----
.. rubric:: The need for scaling

In beam, plate and shell models the degrees-of-freedom mix translations and
rotations, such that the residual mixes forces and moments, and its
Euclidean norm `\|R\|` adds the squares of forces and of moments: the
quantity has no physical meaning and its value depends on the system of
units, the moments being scaled by a change of the length unit while the
forces are not [1]_. Each component is therefore scaled with the
corresponding diagonal entry of a reference stiffness matrix, an idea of
[2]_ that [3]_ (Eqs. 27 and 28) used in the convergence test of his
arc-length procedure, with the diagonal of the tangent stiffness matrix at
the beginning of the increment, and that [1]_ recommends for convergence
testing:

.. math::

    d_j = K_{{C0}_{jj}}, \qquad
    \|v\|_{\boldsymbol{d}} = \sqrt{\sum_{j=1}^{n} \frac{v_j^2}{|d_j|}}

with `n` the number of free degrees-of-freedom and `d_j` the entries of
the diagonal scaling vector `\boldsymbol{d}`. `d_j` is a stiffness, of
the units of the generalized force of the degree-of-freedom `j` divided by
those of its generalized displacement, such that each term `v_j^2/|d_j|`
has the units of energy, whether the degree-of-freedom is a translation or
a rotation: `\|v\|_{\boldsymbol{d}}` is the square root of an energy-like
quantity, independent of the units of each degree-of-freedom. The absolute
value guards against negative diagonal entries.

.. rubric:: Crisfield's criterion

The iteration `i` has converged when

.. math::

    \zeta_i = \frac{\|R_i\|_{\boldsymbol{d}}}{\max\left(
    \|\lambda F_{ext}\|_{\boldsymbol{d}}, \|F_{int}(u_i)\|_{\boldsymbol{d}},
    \|\Delta\lambda F_{ext}\|_{\boldsymbol{d}},
    \|\lambda \, \partial R/\partial \lambda\|_{\boldsymbol{d}}\right)}
    \le \epsilon

with `\epsilon` = ``Analysis.relTOL``, `\lambda` the current load factor
and `\{F_{ext}\}` the reference external force vector. The third reference
only applies to the arc-length solvers, `\Delta\lambda` being the
increment of the load factor of the current step, and the fourth one only
to load vectors that depend on the load factor, e.g. prescribed
displacements imposed exactly, see :mod:`.callbacks`. These are also the
references of the relative criterion, see :func:`check_convergence`,
measured here with the scaled norm. The maximum does not degenerate:
under self-equilibrated loads `\|\lambda F_{ext}\|_{\boldsymbol{d}}` may be
negligible whereas `\|F_{int}\|_{\boldsymbol{d}}` reflects the internal
force carried by the structure; at the beginning of
a load-controlled analysis the internal forces are small and the external
load is the reference; when the arc-length solvers pass through a load
factor close to zero, `\Delta\lambda F_{ext}` is the reference; and under
prescribed displacements alone the external and the free part of the
internal force vectors both vanish at equilibrium, whereas `\lambda \,
\partial R/\partial \lambda`, the out-of-balance force that the prescribed
displacements produce with the free degrees-of-freedom held fixed, does
not. The norms are evaluated over the free degrees-of-freedom only, since the
components of the residual at the constrained ones are the support
reactions, not an imbalance. `\boldsymbol{d}` is computed once from the
linear stiffness matrix ``calc_kC()``, assembled once for the whole
analysis, and kept constant during the iterations, such that `\zeta_i`
measures the progress of the iterations and not the evolution of the
tangent stiffness matrix. The internal force vector of the numerator and of
the denominator comes from the same evaluation, at no additional cost.

With a consistent tangent stiffness matrix the Newton-Raphson iterations
converge quadratically, `\zeta_{i+1} \sim \zeta_i^2`, and a tight
tolerance such as `\epsilon = 10^{-9}` is reached in a few iterations; with
the modified Newton-Raphson method the convergence is linear ([1]_,
Sections 1.2.2 and 1.2.3, pp. 8-13). A number of iterations consistently
larger than three or four for moderate increments indicates a tangent
stiffness matrix inconsistent with the internal force vector, which is why
`\zeta_i` is printed at every iteration.

The criterion applies to the arc-length methods [3]_, which only change
how the load factor is updated, not the residual, with the additional
reference `\Delta\lambda F_{ext}`.

.. rubric:: The relative criterion

With ``Analysis.convergence = 'relative'``, the iteration has converged
when `\|R\| \le` ``relTOL`` `\max(\|F_{ext}\|, \|F_{int}\|, ...)`, with the
Euclidean norm over all the degrees-of-freedom of the solver, or when
`\max|R_j| \le` ``absTOL``, which depends on the units of the model, see
:func:`check_convergence`.

References
----------
.. [1] Crisfield, M. A., Non-linear Finite Element Analysis of Solids and
   Structures, Volume 1: Essentials, John Wiley & Sons, Chichester, 1991.
.. [2] Peano, A. and Riccioni, R., Automated discretisation error control in
   finite element analysis, in: Robinson, J. (ed.), Finite Elements in the
   Commercial Environment, Robinson and Associates, Verwood, 368-387, 1978.
.. [3] Crisfield, M. A., A fast incremental/iterative solution procedure
   that handles "snap-through", Computers & Structures, 13(1-3), 55-62,
   1981. https://doi.org/10.1016/0045-7949(81)90108-5

"""
import numpy as np
from scipy.sparse import csr_matrix

#: Names of the convergence criteria accepted by ``Analysis.convergence``
CRITERIA = ('crisfield', 'relative')


def crisfield_weights(K0):
    r"""Weights `1/|d_j|` of the scaled norm of Crisfield's criterion

    The weight of each free degree-of-freedom is the inverse of the modulus
    of the diagonal entry of ``K0``. The free degrees-of-freedom are those
    whose row or column of ``K0`` is not null, as the ones kept by the
    solvers, see :func:`.remove_null_cols`; the others get a null weight,
    i.e. they do not enter the norms.

    Parameters
    ----------
    K0 : sparse_matrix or np.ndarray
        Linear stiffness matrix, shape ``(n, n)``, normally ``calc_kC()``.

    Returns
    -------
    w : np.ndarray
        Weights, 1-D array of size ``n``, of the inverse of the dimension of
        the diagonal entries of ``K0``.

    Raises
    ------
    ValueError
        When the diagonal entry of a free degree-of-freedom is null, which
        leaves its scaling undefined.

    See Also
    --------
    crisfield_zeta : the criterion that uses them.

    Notes
    -----
    `w_j = 1/|K_{{C0}_{jj}}|` at the free degrees-of-freedom, see the
    module documentation, and `w_j = 0` at the degrees-of-freedom whose row
    and column of `[K_{C0}]` are both null, i.e. constrained by zeroing
    them.

    """
    K0 = csr_matrix(K0)
    rows, cols = K0.nonzero()
    free = np.zeros(K0.shape[0], dtype=bool)
    free[rows] = True
    free[cols] = True
    d = np.abs(np.asarray(K0.diagonal(), dtype=float))
    if np.any(free & (d == 0)):
        raise ValueError('Crisfield criterion: null diagonal entry of the '
                         'linear stiffness matrix at a free '
                         'degree-of-freedom, use convergence="relative"')
    w = np.zeros(K0.shape[0])
    w[free] = 1./d[free]
    return w


def scaled_norm(v, w):
    r"""Scaled norm `\|v\|_{\boldsymbol{d}}` of a force-like vector

    Parameters
    ----------
    v : np.ndarray
        Force-like vector, 1-D array of size ``n``.
    w : np.ndarray
        Weights of :func:`crisfield_weights`, 1-D array of size ``n``.

    Returns
    -------
    norm : float
        `\sqrt{\sum_j w_j v_j^2}`, of the dimension of the square root of an
        energy.

    """
    v = np.asarray(v)
    return float(np.sqrt(np.sum(w*np.abs(v)**2)))


def crisfield_zeta(R, fext, fint, w, q=None, fext_inc=None):
    r"""Relative force imbalance `\zeta` of Crisfield's criterion

    Parameters
    ----------
    R : np.ndarray
        Residual force vector, ``fext - fint``, 1-D array of size ``n``.
    fext : np.ndarray
        External force vector at the current load factor, 1-D array of size
        ``n``.
    fint : np.ndarray
        Internal force vector, from the same evaluation as ``R``, 1-D array
        of size ``n``.
    w : np.ndarray
        Weights of :func:`crisfield_weights`, 1-D array of size ``n``.
    q : np.ndarray or None, optional
        Additional reference, the load factor times `\partial R/\partial
        \lambda` when the load vector depends on the load factor, see
        :mod:`.callbacks`. Default ``None``.
    fext_inc : np.ndarray or None, optional
        Additional reference, the external force vector of the increment of
        the arc-length solvers, `\Delta\lambda \{F_{ext}\}`. Default
        ``None``.

    Returns
    -------
    zeta : float
        `\|R\|_{\boldsymbol{d}}` over the largest scaled norm of the
        references, dimensionless, ``0.`` when the residual and all the
        references are null.

    Notes
    -----
    The criterion is `\zeta \le \epsilon`, see the module documentation:
    the reference is `\max(\|F_{ext}\|_{\boldsymbol{d}},
    \|F_{int}\|_{\boldsymbol{d}})`, with also `\|q\|_{\boldsymbol{d}}` and
    `\|\Delta\lambda F_{ext}\|_{\boldsymbol{d}}` when given. When the
    reference is null and the residual is not, `\zeta` is infinite.

    """
    Rn = scaled_norm(R, w)
    ref = max(scaled_norm(fext, w), scaled_norm(fint, w))
    if q is not None:
        ref = max(ref, scaled_norm(q, w))
    if fext_inc is not None:
        ref = max(ref, scaled_norm(fext_inc, w))
    if ref > 0:
        return Rn/ref
    return 0. if Rn == 0 else np.inf


def check_convergence(an, R, fint, fext_total, fext_ref=None, q_ref=None,
                      w=None):
    r"""Check the convergence of the residual force vector

    With ``an.convergence = 'crisfield'`` the convergence is achieved when
    `\zeta \le` ``an.relTOL``, see :func:`crisfield_zeta`, with the weights
    ``w``. With ``an.convergence = 'relative'`` it is achieved when either:

    - ``norm(R) <= an.relTOL*max(norm(fext_total), norm(fint),
      norm(fext_ref), norm(q_ref))``, with the Euclidean norm, if
      ``an.relTOL`` is not ``None``;
    - ``max(abs(R)) <= an.absTOL``, if ``an.absTOL`` is not ``None``.

    ``an.absTOL`` is also accepted as an additional criterion with
    ``'crisfield'``.

    Parameters
    ----------
    an : :class:`.Analysis`
        Analysis object with the attributes ``convergence``, ``relTOL`` and
        ``absTOL``.
    R : np.ndarray
        Residual force vector, 1-D array.
    fint : np.ndarray
        Internal force vector of the same evaluation, 1-D array.
    fext_total : np.ndarray
        External force vector at the current load factor, 1-D array.
    fext_ref : np.ndarray or None, optional
        Additional reference of both criteria, the external force vector of
        the current increment of the arc-length solvers. Default ``None``.
    q_ref : np.ndarray or None, optional
        Additional reference of both criteria, the load factor times
        `\partial R/\partial \lambda` when ``calc_fext`` accepts ``lbd``, see
        :mod:`.callbacks`. Default ``None``.
    w : np.ndarray or None, optional
        Weights of :func:`crisfield_weights`, required with ``'crisfield'``.
        Default ``None``.

    Returns
    -------
    converged : bool
        Whether the criterion is satisfied.
    Rnorm : float
        Norm of the residual used by the criterion,
        `\|R\|_{\boldsymbol{d}}` with ``'crisfield'``, the Euclidean norm
        with ``'relative'``, used by the divergence checks of the solvers.
    zeta : float
        Relative residual, `\zeta` with ``'crisfield'``, ``norm(R)`` over
        the largest Euclidean norm of the references with ``'relative'``.

    Raises
    ------
    ValueError
        When ``an.convergence`` is not one of :data:`CRITERIA`, when
        ``an.relTOL`` and ``an.absTOL`` are both ``None``, or when ``w`` is
        missing with ``'crisfield'``.

    Notes
    -----
    The relative criterion is that of the first versions of these solvers;
    its reference ``fext_ref`` is useful when the load factor is close to
    zero, where ``fext_total`` and ``fint`` are small. When all its
    references are null, the relative residual is ``norm(R)`` itself.

    """
    if an.convergence not in CRITERIA:
        raise ValueError('%r is an invalid convergence criterion, use one of '
                         '%s' % (an.convergence, CRITERIA))
    if an.relTOL is None and an.absTOL is None:
        raise ValueError('At least one of relTOL or absTOL must be defined')
    if an.convergence == 'crisfield':
        if w is None:
            raise ValueError('the weights of the Crisfield criterion are '
                             'missing')
        Rnorm = scaled_norm(R, w)
        zeta = crisfield_zeta(R, fext_total, fint, w, q=q_ref,
                              fext_inc=fext_ref)
    else:
        Rnorm = float(np.linalg.norm(R))
        ref = max(np.linalg.norm(fext_total), np.linalg.norm(fint))
        if fext_ref is not None:
            ref = max(ref, np.linalg.norm(fext_ref))
        if q_ref is not None:
            ref = max(ref, np.linalg.norm(q_ref))
        zeta = Rnorm/ref if ref > 0 else Rnorm
    converged = False
    if an.relTOL is not None:
        if an.convergence == 'crisfield':
            converged = zeta <= an.relTOL
        else:
            converged = Rnorm <= an.relTOL*ref
    if an.absTOL is not None and np.abs(R).max() <= an.absTOL:
        converged = True
    return converged, Rnorm, zeta
