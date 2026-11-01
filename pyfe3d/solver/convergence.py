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
  `-\partial \boldsymbol{R}/\partial \lambda`, over the free degrees-of-freedom only, see
  :func:`crisfield_zeta`;
- ``'relative'``: the Euclidean norm of the residual compared with the
  largest Euclidean norm of the force vectors, ``Analysis.relTOL``, or the
  largest absolute residual force, ``Analysis.absTOL``.

Notes
-----
.. rubric:: The need for scaling

In beam, plate and shell models the degrees-of-freedom mix translations and
rotations, such that the residual mixes forces and moments, and its
Euclidean norm `\|\boldsymbol{R}\|` adds the squares of forces and of moments: the
quantity has no physical meaning and its value depends on the system of
units, the moments being scaled by a change of the length unit while the
forces are not. Each component is therefore scaled with the
corresponding diagonal entry of a reference stiffness matrix, a scaling
that [3]_ (p. 58) attributes to [2]_ and that he used (his Eqs. 27 and 28)
in the convergence test of his arc-length procedure, with the diagonal of
the tangent stiffness matrix at the beginning of the increment. [1]_ (Sec.
9.5.4, Eq. 9.48) presents the same scaling as an alternative to the
Euclidean norm of the residual, with a scaling matrix that could, for
example, be used to ensure that, for a problem involving rotational
variables, all parameters have the same dimensions:

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

    \zeta_i = \frac{\|\boldsymbol{R}_i\|_{\boldsymbol{d}}}{\max\left(
    \|\lambda \boldsymbol{F}_{ext}\|_{\boldsymbol{d}}, \|\boldsymbol{F}_{int}(\boldsymbol{u}_i)\|_{\boldsymbol{d}},
    \|\Delta\lambda \boldsymbol{F}_{ext}\|_{\boldsymbol{d}},
    \|\lambda \, \partial \boldsymbol{R}/\partial \lambda\|_{\boldsymbol{d}}\right)}
    \le \epsilon

with `\epsilon` = ``Analysis.relTOL``, `\lambda` the current load factor
and `\boldsymbol{F}_{ext}` the reference external force vector. The third reference
only applies to the arc-length solvers, `\Delta\lambda` being the
increment of the load factor of the current step, and the fourth one only
to load vectors that depend on the load factor, e.g. prescribed
displacements imposed exactly, see :mod:`.callbacks`. These are also the
references of the relative criterion, see :func:`check_convergence`,
measured here with the scaled norm. The maximum does not degenerate:
under self-equilibrated loads `\|\lambda \boldsymbol{F}_{ext}\|_{\boldsymbol{d}}` may be
negligible whereas `\|\boldsymbol{F}_{int}\|_{\boldsymbol{d}}` reflects the internal
force carried by the structure; at the beginning of
a load-controlled analysis the internal forces are small and the external
load is the reference; when the arc-length solvers pass through a load
factor close to zero, `\Delta\lambda \boldsymbol{F}_{ext}` is the reference; and under
prescribed displacements alone the external and the free part of the
internal force vectors both vanish at equilibrium, whereas `-\lambda \,
\partial \boldsymbol{R}/\partial \lambda`, the out-of-balance force that the prescribed
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
reference `\Delta\lambda \boldsymbol{F}_{ext}`.

.. rubric:: The relative criterion

With ``Analysis.convergence = 'relative'``, the iteration has converged
when `\|\boldsymbol{R}\| \le` ``relTOL`` `\max(\|\boldsymbol{F}_{ext}\|, \|\boldsymbol{F}_{int}\|, ...)`, with the
Euclidean norm over all the degrees-of-freedom of the solver, or when
`\max|R_j| \le` ``absTOL``, which depends on the units of the model, see
:func:`check_convergence`.

.. rubric:: The round-off floor

The residual is the difference of force vectors computed in floating-point
arithmetic. Each component of `\boldsymbol{F}_{int}` sums element contributions of
the order of `\sum_k |K_{{C0}_{jk}}| |c_k|`, which may be much larger than
the component itself, e.g. under large self-equilibrated axial forces,
such that the residual cannot be computed more accurately than about
`\varepsilon \| |\boldsymbol{K}_{C0}| \, |\boldsymbol{c}| \|`, with `\varepsilon` the machine
precision and `|\boldsymbol{K}_{C0}|` and `|\boldsymbol{c}|` the elementwise absolute values of the
linear stiffness matrix and of the solution, which is the componentwise
bound of the rounding error of a matrix-vector product [5]_. When this level is above
`\epsilon` times the references of `\zeta_i`, e.g. near a null load, where
the references, net force vectors, are small and the element contributions
are not, the iterations stagnate and every step fails. The iteration `i`
has therefore also converged when

.. math::

    \|\boldsymbol{R}_i\|_{\boldsymbol{d}} \le C \, \varepsilon \,
    \| |\boldsymbol{K}_{C0}| \, |\boldsymbol{c}_i| \|_{\boldsymbol{d}}

with `C` = ``ROUNDOFF_FACTOR`` `= 3`, `c_i` the solution of the iteration,
and the Euclidean norm instead of `\|\cdot\|_{\boldsymbol{d}}` with the
relative criterion, see :func:`roundoff_floor`. Like ``absTOL``, the floor
is an alternative to ``relTOL`` with both criteria. `|\boldsymbol{K}_{C0}|` is computed
once, with `\boldsymbol{d}`, and the floor adds one product by `|\boldsymbol{c}_i|` per
iteration. It does not change `\zeta_i`, which may then be larger than
`\epsilon`: the log of the iterations marks the iterations converged at the
round-off floor. With prescribed displacements, ``uk``, `|\boldsymbol{K}_{C0}|` and
`|\boldsymbol{c}|` are those of the free degrees-of-freedom, the product
`|K_{ff}| \, |u_f|` omitting the contribution of the prescribed
displacements, `|K_{fp}| \, |u_p|`: the floor is then lower, i.e.
conservative.

The floor assumes that ``calc_fint`` computes the internal force from the
displacements `\boldsymbol{c}`, whose round-off is then of the order of
`\varepsilon |\boldsymbol{K}_{C0}| |\boldsymbol{c}|`. It cannot cover a
round-off that depends on the nodal coordinates `\boldsymbol{X}`, e.g. of
current positions computed as `\boldsymbol{X} + \boldsymbol{u}`: a
translation of the origin changes `|\boldsymbol{X}|` but none of the
quantities the criterion receives, `\boldsymbol{K}_{C0}`, `\boldsymbol{c}`,
`\boldsymbol{F}_{int}` and `\boldsymbol{F}_{ext}`. Measured on 2026-10-30
on the cantilever of Sze et al. under an end shear force with 64
co-rotational :class:`.BeamC` elements, positions computed as
`\boldsymbol{X} + \boldsymbol{u}`: with the root at `X = 0`, 100 and 1000,
the iterations of the first increment stagnated at `\zeta` of about
`10^{-8}`, `10^{-7}` and `10^{-6}`, at 0.12 to 0.19 times `\varepsilon \|
|\boldsymbol{K}_{C0}| |\boldsymbol{X}| \|_{\boldsymbol{d}}`, whereas
`\boldsymbol{K}_{C0}`, the iterates and the floor were the same, see
``test_cantilever_end_shear_translated`` in
``tests/test_corotational_benchmarks.py``. Such a
model should compute the positions relative to a nearby point, as
:mod:`pyfe3d.corotational` does with the base centroid of each element,
or else set ``absTOL``.

Measured on the low sinusoidal arch of Fung and Kaplan [4]_, hinged at both
ends under a sinusoidal load, `\lambda_1 = 2.2`, modelled with 40
:class:`.BeamC` elements, traced with both arc-length methods and both
criteria, with arc-length increments of 0.05 and `\epsilon = 10^{-9}`, and
a reference load of 1.25 times the upper limit load, reached at the load
factor 0.80: without the floor, the four analyses stop at load factors of
0.026 to 0.032, close to the null load between the limit points; after the
last converged step, the residuals of the iterations that follow the first
correction of each attempt remain between 0.29 and 0.47 times
`\varepsilon \| |\boldsymbol{K}_{C0}| \, |\boldsymbol{c}| \|`, with `\zeta_i \ge 1.0 \times 10^{-9}`,
and no residual of the analyses is below 0.21 times this level. With 80
elements they stop at load factors of 0.20 to 0.21, between the limit
points, and with 160 elements at 0.67 to 0.71, before the upper limit
point. With the floor, all of them reach the load factor 1.0, the floor
accepting 9 to 31 of the about 360 iterations with 40 elements, with
`\zeta_i` up to `8.8 \times 10^{-9}` and residuals of 0.30 to 2.9 times
`\varepsilon \| |\boldsymbol{K}_{C0}| \, |\boldsymbol{c}| \|`, and 83 to 86 iterations with 160
elements.

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
.. [4] Fung, Y. C. and Kaplan, A., Buckling of low arches or curved beams of
   small curvature, NACA Technical Note 2840, National Advisory Committee
   for Aeronautics, Washington, 1952.
   https://ntrs.nasa.gov/citations/19930083579
.. [5] Higham, N. J., Accuracy and Stability of Numerical Algorithms, 2nd
   ed., SIAM, 2002, Sec. 3.5, Eqs. (3.12) and (3.13).
   https://doi.org/10.1137/1.9780898718027

"""
import numpy as np
from scipy.sparse import csr_matrix

from .sparseutils import check_null_rows_cols

#: Names of the convergence criteria accepted by ``Analysis.convergence``
CRITERIA = ('crisfield', 'relative')
#: An iteration has also converged when the norm of its residual is not
#: larger than this factor times the machine precision times the norm of
#: `|\boldsymbol{K}_{C0}| |\boldsymbol{c}|`, the round-off floor of the residual, see
#: :func:`roundoff_floor` and :func:`check_convergence`
ROUNDOFF_FACTOR = 3.


def crisfield_weights(K0):
    r"""Weights `1/|d_j|` of the scaled norm of Crisfield's criterion

    The weight of each degree-of-freedom is the inverse of the modulus of
    the diagonal entry of ``K0``. ``K0`` must have no null row or column, as
    the matrices given to the solvers, see :func:`.check_null_rows_cols`.

    Parameters
    ----------
    K0 : sparse_matrix or np.ndarray
        Linear stiffness matrix of the free degrees-of-freedom, shape ``(n,
        n)``, normally ``calc_kC()``.

    Returns
    -------
    w : np.ndarray
        Weights, 1-D array of size ``n``, of the inverse of the dimension of
        the diagonal entries of ``K0``.

    Raises
    ------
    numpy.linalg.LinAlgError
        When ``K0`` has a null row or column, see
        :func:`.check_null_rows_cols`.
    ValueError
        When a diagonal entry is null in a row or column that is not null,
        which leaves the scaling of its degree-of-freedom undefined.

    See Also
    --------
    crisfield_zeta : the criterion that uses them.

    Notes
    -----
    `w_j = 1/|K_{{C0}_{jj}}|`, see the module documentation. The
    constrained degrees-of-freedom are partitioned out before calling the
    solvers, so that every degree-of-freedom of ``K0`` is free and enters
    the norms. A null diagonal entry in a row that is not null, e.g. of a
    Lagrange multiplier, gives no scale to its degree-of-freedom, in which
    case the relative criterion is the alternative,
    ``convergence='relative'``.

    """
    K0 = csr_matrix(K0)
    check_null_rows_cols(K0, 'K0')
    d = np.abs(np.asarray(K0.diagonal(), dtype=float))
    if np.any(d == 0):
        raise ValueError('Crisfield criterion: null diagonal entry of the '
                         'linear stiffness matrix, use convergence="relative"')
    return 1./d


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
        Residual force vector, ``fint - fext``, 1-D array of size ``n``.
    fext : np.ndarray
        External force vector at the current load factor, 1-D array of size
        ``n``.
    fint : np.ndarray
        Internal force vector, from the same evaluation as ``R``, 1-D array
        of size ``n``.
    w : np.ndarray
        Weights of :func:`crisfield_weights`, 1-D array of size ``n``.
    q : np.ndarray or None, optional
        Additional reference, the load factor times `-\partial \boldsymbol{R}/\partial
        \lambda` when the load vector depends on the load factor, see
        :mod:`.callbacks`. Default ``None``.
    fext_inc : np.ndarray or None, optional
        Additional reference, the external force vector of the increment of
        the arc-length solvers, `\Delta\lambda \boldsymbol{F}_{ext}`. Default
        ``None``.

    Returns
    -------
    zeta : float
        `\|\boldsymbol{R}\|_{\boldsymbol{d}}` over the largest scaled norm of the
        references, dimensionless, ``0.`` when the residual and all the
        references are null.

    Notes
    -----
    The criterion is `\zeta \le \epsilon`, see the module documentation:
    the reference is `\max(\|\boldsymbol{F}_{ext}\|_{\boldsymbol{d}},
    \|\boldsymbol{F}_{int}\|_{\boldsymbol{d}})`, with also `\|\boldsymbol{q}\|_{\boldsymbol{d}}` and
    `\|\Delta\lambda \boldsymbol{F}_{ext}\|_{\boldsymbol{d}}` when given. When the
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


def roundoff_floor(Kc_abs, w=None):
    r"""Round-off floor of the residual norm

    Parameters
    ----------
    Kc_abs : np.ndarray
        `|\boldsymbol{K}_{C0}| |\boldsymbol{c}|`, the product of the elementwise absolute values of
        the linear stiffness matrix and of the current solution, 1-D array
        of size ``n``.
    w : np.ndarray or None, optional
        Weights of :func:`crisfield_weights`, for the scaled norm of
        Crisfield's criterion; the Euclidean norm when ``None``. Default
        ``None``.

    Returns
    -------
    floor : float
        ``ROUNDOFF_FACTOR*eps*norm(Kc_abs)``, with ``eps`` the machine
        precision of ``float`` and the norm of the criterion.

    See Also
    --------
    check_convergence : the criterion that uses it.

    Notes
    -----
    `C \varepsilon \| |\boldsymbol{K}_{C0}| |\boldsymbol{c}| \|`, with `C` = ``ROUNDOFF_FACTOR``,
    see the module documentation.

    """
    Kc_abs = np.asarray(Kc_abs)
    if w is None:
        norm = float(np.linalg.norm(Kc_abs))
    else:
        norm = scaled_norm(Kc_abs, w)
    return ROUNDOFF_FACTOR*np.finfo(float).eps*norm


def check_convergence(an, R, fint, fext_total, fext_ref=None, q_ref=None,
                      w=None, Kc_abs=None, full_output=False):
    r"""Check the convergence of the residual force vector

    With ``an.convergence = 'crisfield'`` the convergence is achieved when
    `\zeta \le` ``an.relTOL``, see :func:`crisfield_zeta`, with the weights
    ``w``. With ``an.convergence = 'relative'`` it is achieved when either:

    - ``norm(R) <= an.relTOL*max(norm(fext_total), norm(fint),
      norm(fext_ref), norm(q_ref))``, with the Euclidean norm, if
      ``an.relTOL`` is not ``None``;
    - ``max(abs(R)) <= an.absTOL``, if ``an.absTOL`` is not ``None``.

    ``an.absTOL`` is also accepted as an additional criterion with
    ``'crisfield'``. With both criteria, when ``Kc_abs`` is given, the
    convergence is also achieved when the norm of ``R`` of the criterion is
    not larger than the round-off floor of :func:`roundoff_floor`.

    Parameters
    ----------
    an : :class:`.Analysis`
        Analysis object with the attributes ``convergence``, ``relTOL`` and
        ``absTOL``.
    R : np.ndarray
        Residual force vector, ``fint - fext_total``, 1-D array.
    fint : np.ndarray
        Internal force vector of the same evaluation, 1-D array.
    fext_total : np.ndarray
        External force vector at the current load factor, 1-D array.
    fext_ref : np.ndarray or None, optional
        Additional reference of both criteria, the external force vector of
        the current increment of the arc-length solvers. Default ``None``.
    q_ref : np.ndarray or None, optional
        Additional reference of both criteria, the load factor times
        `-\partial \boldsymbol{R}/\partial \lambda` when ``calc_fext`` accepts ``lbd``, see
        :mod:`.callbacks`. Default ``None``.
    w : np.ndarray or None, optional
        Weights of :func:`crisfield_weights`, required with ``'crisfield'``.
        Default ``None``.
    Kc_abs : np.ndarray or None, optional
        `|\boldsymbol{K}_{C0}| |\boldsymbol{c}|`, the product of the elementwise absolute values of
        the linear stiffness matrix ``calc_kC()`` and of the solution of the
        evaluation of ``R``, 1-D array. ``None`` disables the round-off
        floor. Default ``None``.
    full_output : bool, optional
        Whether to also return ``roundoff``. Default ``False``.

    Returns
    -------
    converged : bool
        Whether the criterion is satisfied.
    Rnorm : float
        Norm of the residual used by the criterion,
        `\|\boldsymbol{R}\|_{\boldsymbol{d}}` with ``'crisfield'``, the Euclidean norm
        with ``'relative'``, used by the divergence checks of the solvers.
    zeta : float
        Relative residual, `\zeta` with ``'crisfield'``, ``norm(R)`` over
        the largest Euclidean norm of the references with ``'relative'``.
    roundoff : bool
        Only with ``full_output=True``: whether the convergence is achieved
        by the round-off floor only, i.e. not by ``an.relTOL`` or
        ``an.absTOL``.

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

    The round-off floor, see the module documentation, does not change
    ``zeta``: an iteration that it accepts may have `\zeta >` ``an.relTOL``.

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
        w = None
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
    roundoff = False
    if (not converged and Kc_abs is not None
            and Rnorm <= roundoff_floor(Kc_abs, w=w)):
        converged = roundoff = True
    converged = bool(converged)
    if full_output:
        return converged, Rnorm, zeta, roundoff
    return converged, Rnorm, zeta
