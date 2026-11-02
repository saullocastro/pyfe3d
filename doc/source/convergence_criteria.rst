Convergence criteria
====================

The iterations of :doc:`newton_raphson_method` and of
:doc:`arc_length_methods` stop when the residual `\boldsymbol{R} =
\boldsymbol{F}_{int} - \lambda \boldsymbol{F}_{ext}` is small enough. The test is on
the residual, the actual imbalance of forces, and not on the size of the
corrections, which measures only how far the last iteration moved.
``Analysis.convergence`` selects one of the two criteria of
:mod:`pyfe3d.solver.convergence`, applied by
:func:`pyfe3d.solver.convergence.check_convergence`: ``'crisfield'``, the
default, or ``'relative'``.


Why the residual must be scaled
-------------------------------

The dofs of beam, plate and shell models mix translations and rotations,
and the residual mixes forces and moments. Its Euclidean norm adds the
squares of forces and of moments, a quantity without physical meaning,
whose value depends on the system of units: a change of the length unit
scales the moments but not the forces, and changes the relative weight of
the two in the norm. Each component is therefore scaled with the diagonal
entry of a reference stiffness matrix, a scaling that Crisfield (1981, p.
58) attributes to Peano and Riccioni (1978) and that Crisfield (1991, Sec.
9.5.4, Eq. (9.48)) presents as an alternative to the Euclidean norm of the
residual. With the diagonal scaling vector `\boldsymbol{d}` of the linear
stiffness matrix of the free dofs,

.. math::
    :label: cc-scaled-norm

    d_j = K_{{C0}_{jj}}, \qquad
    \|\boldsymbol{v}\|_{\boldsymbol{d}} = \sqrt{\sum_{j=1}^{n} \frac{v_j^2}{|d_j|}}

with `n` the number of free dofs, see :func:`.crisfield_weights` and
:func:`pyfe3d.solver.convergence.scaled_norm`. `d_j` is a stiffness, of the
units of the generalized force of the dof `j` divided by those of its
generalized displacement, such that each term `v_j^2/|d_j|` has the units
of energy, whether the dof is a translation or a rotation: the scaled norm
is the square root of an energy-like quantity, independent of the units of
each dof. The absolute value guards against negative diagonal entries. A
null diagonal entry in a row that is not null, e.g. of a Lagrange
multiplier, leaves its dof without a scale and raises a ``ValueError``; the
relative criterion is then the alternative.


Crisfield's criterion
---------------------

The iteration `i` has converged when

.. math::
    :label: cc-zeta

    \zeta_i = \frac{\|\boldsymbol{R}_i\|_{\boldsymbol{d}}}{\max\left(
    \|\lambda \boldsymbol{F}_{ext}\|_{\boldsymbol{d}}, \|\boldsymbol{F}_{int}(\boldsymbol{u}_i)\|_{\boldsymbol{d}},
    \|\Delta\lambda \boldsymbol{F}_{ext}\|_{\boldsymbol{d}},
    \|\lambda \, \partial \boldsymbol{R}/\partial \lambda\|_{\boldsymbol{d}}\right)}
    \le \epsilon

with `\epsilon` = ``Analysis.relTOL``, `10^{-9}` by default, see
:func:`.crisfield_zeta`. The third reference applies only to the
arc-length methods, with `\Delta\lambda` the increment of the load factor
of the current step, and the fourth only to load vectors that depend on the
load factor, e.g. prescribed displacements imposed exactly, see
:doc:`boundary_conditions`. The maximum does not degenerate:

- under self-equilibrated loads `\|\lambda \boldsymbol{F}_{ext}\|_{\boldsymbol{d}}` may
  be negligible, whereas `\|\boldsymbol{F}_{int}\|_{\boldsymbol{d}}` reflects the
  internal forces carried by the structure;
- at the beginning of a load-controlled analysis the internal forces are
  small and the external load is the reference;
- when an arc-length method passes through a load factor close to zero,
  `\Delta\lambda \boldsymbol{F}_{ext}` is the reference;
- under prescribed displacements alone, the external and the free part of
  the internal force vectors both vanish at equilibrium, whereas
  `-\lambda \, \partial \boldsymbol{R}/\partial \lambda`, the out-of-balance force
  that the prescribed displacements produce with the free dofs held fixed,
  does not.

The norms are evaluated over the free dofs only: the components of the
residual at the known dofs are the support reactions, not an imbalance.
`\boldsymbol{d}` is computed once from the linear stiffness matrix
``calc_kC()``, assembled once for the analysis, and kept constant, such
that `\zeta_i` measures the progress of the iterations and not the
evolution of the tangent stiffness matrix; the internal force vector of the
numerator and of the denominator comes from the same evaluation, at no
additional cost. ``tests/test_solver_nonlinear_static.py`` checks that
scaling the rotational dofs of a plate model changes neither the
iterations nor `\zeta`, whereas it changes the relative criterion, and that
the norms ignore the reactions.

The criterion is the same for the arc-length methods, which change how the
load factor is updated but not the residual, with the additional reference
`\Delta\lambda \boldsymbol{F}_{ext}` (Crisfield 1981).


The relative criterion
----------------------

With ``Analysis.convergence = 'relative'``, the iteration has converged
when

.. math::
    :label: cc-relative

    \|\boldsymbol{R}\| \le \epsilon \max\left(\|\lambda \boldsymbol{F}_{ext}\|,
    \|\boldsymbol{F}_{int}\|, \|\Delta\lambda \boldsymbol{F}_{ext}\|,
    \|\lambda \, \partial \boldsymbol{R}/\partial \lambda\|\right)

with the Euclidean norm over the free dofs and the same references, or
`\|\boldsymbol{R}\|` itself when they are all null. It depends on the units of
the dofs, as discussed above, and remains for models whose dofs have no
stiffness scale, e.g. with Lagrange multipliers.

With both criteria, ``Analysis.absTOL``, ``None`` by default, also accepts
the iterations whose largest absolute residual is not larger than it, in
the units of force of the model, and ``relTOL = None`` leaves only that
test.


Choice of the tolerance
-----------------------

With a consistent tangent stiffness matrix, the full Newton-Raphson
iterations converge quadratically, `\zeta_{i+1} \sim \zeta_i^2`, and a
tolerance as tight as `\epsilon = 10^{-9}` is reached in two or three
iterations from the `\zeta` of `10^{-2}` to `10^{-3}` of a moderate
increment; with the modified Newton-Raphson method the convergence is
linear (Crisfield 1991, Secs. 1.2.2 and 1.2.3, pp. 8-13) and looser
tolerances are more realistic. The decay of `\zeta`, printed at every
iteration, is a diagnostic: a number of iterations consistently larger than
three or four for moderate increments reveals a tangent stiffness matrix
inconsistent with the internal force vector, e.g. a missing ``calc_kG``,
which changes the iterations but not the converged solution. A tight
tolerance is recommended when the converged state feeds an eigenvalue
analysis, whose geometric stiffness matrix is built from its stresses.


The round-off floor
-------------------

The tolerance cannot be made arbitrarily small, because the residual is the
difference of force vectors computed in floating-point arithmetic. Each
component of `\boldsymbol{F}_{int}` sums element contributions of the order of
`\sum_k |K_{{C0}_{jk}}| |u_k|`, which may be much larger than the component
itself, e.g. under large self-equilibrated axial forces, and the rounding
error of such a sum is bounded componentwise by the product of the absolute
values times the machine precision `\varepsilon` and a factor of the number
of terms (Higham 2002, Sec. 3.5, Eqs. (3.12) and (3.13)). The residual thus
cannot be computed more accurately than about `\varepsilon \| |\boldsymbol{K}_{C0}|
\, |\boldsymbol{u}| \|`, with `|\cdot|` the elementwise absolute values. When this
level is above `\epsilon` times the references of :eq:`cc-zeta`, e.g. near a
null load, where the references, net force vectors, are small and the
element contributions are not, the iterations stagnate and every step
fails. An iteration has therefore also converged when

.. math::
    :label: cc-roundoff

    \|\boldsymbol{R}_i\|_{\boldsymbol{d}} \le C \, \varepsilon \,
    \| |\boldsymbol{K}_{C0}| \, |\boldsymbol{u}_i| \|_{\boldsymbol{d}}, \qquad C = 3

with `C` = ``ROUNDOFF_FACTOR`` and the Euclidean norm with the relative
criterion, see :func:`pyfe3d.solver.convergence.roundoff_floor`. Like
``absTOL``, the floor is an alternative to ``relTOL``; it does not change
`\zeta_i`, which may then be larger than `\epsilon`, and the log marks the
iterations converged at the round-off floor. `|\boldsymbol{K}_{C0}|` is computed
once with `\boldsymbol{d}`, and the floor costs one product per iteration. With
prescribed displacements, the product is over the free dofs and omits the
contribution of the prescribed ones, which makes the floor lower, i.e.
conservative.

On the low sinusoidal arch of Fung and Kaplan (1952), traced with both
arc-length methods and both criteria with `\epsilon = 10^{-9}`, the
analyses stop close to the null load between the two limit points without
the floor, with residuals at 0.3 to 0.5 times `\varepsilon \| |\boldsymbol{K}_{C0}|
\, |\boldsymbol{u}| \|`, and all of them reach the load factor 1.0 with it, see
the module documentation of :mod:`pyfe3d.solver.convergence` and
``tests/test_solver_arc_length_fung_kaplan.py``.

The floor assumes that ``calc_fint`` computes the internal forces from the
displacements, whose round-off is then of the order of `\varepsilon
|\boldsymbol{K}_{C0}| |\boldsymbol{u}|`. It cannot cover a round-off that depends on the
nodal coordinates `\boldsymbol{X}`, e.g. of current positions computed as
`\boldsymbol{X} + \boldsymbol{u}`, since a translation of the origin changes none of the
quantities the criterion receives. Such a model should compute the
positions relative to a nearby point, as :mod:`pyfe3d.corotational` does
with the base centroid of each element, or else set ``absTOL``.


References
----------

- Crisfield, M. A., A fast incremental/iterative solution procedure that
  handles "snap-through", Computers & Structures, 13(1-3), 55-62, 1981.
  https://doi.org/10.1016/0045-7949(81)90108-5
- Crisfield, M. A., Non-linear Finite Element Analysis of Solids and
  Structures, Volume 1: Essentials, John Wiley & Sons, Chichester, 1991.
- Fung, Y. C. and Kaplan, A., Buckling of low arches or curved beams of
  small curvature, NACA Technical Note 2840, National Advisory Committee
  for Aeronautics, Washington, 1952.
  https://ntrs.nasa.gov/citations/19930083579
- Higham, N. J., Accuracy and Stability of Numerical Algorithms, 2nd ed.,
  SIAM, Philadelphia, 2002. https://doi.org/10.1137/1.9780898718027
- Peano, A. and Riccioni, R., Automated discretisation error control in
  finite element analysis, in: Robinson, J. (ed.), Finite Elements in the
  Commercial Environment, Robinson and Associates, Verwood, 368-387, 1978.
