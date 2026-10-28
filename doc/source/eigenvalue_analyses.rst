Linear buckling and natural frequencies
=======================================

The eigenvalue analyses use the matrices of the free dofs, e.g.
``KC0uu = KC0[bu, :][:, bu]``, and are solved with
:func:`.linear_buckling`, :func:`.natural_frequency` and
:func:`.complex_frequency` of :mod:`pyfe3d.solver.eigen`, which return the
eigenvectors with the size of the matrices received, to be expanded by the
user, ``u[bu] = eigvecs[:, i]``.


Linear buckling
---------------

A linear buckling analysis has three steps:

1. the linear static solution `\boldsymbol{u}_0` of a reference load `\boldsymbol{F}_{ref}`,
   `\boldsymbol{K}_{{C0}_{uu}} \boldsymbol{u}_{0_u} = \boldsymbol{F}_{{ref}_u}`, see :doc:`overview`;
2. the geometric stiffness matrix of the prestress of `\boldsymbol{u}_0`: for each
   element, ``update_probe_xe(x)``, ``update_probe_ue(u0)`` and
   ``update_KG(KGr, KGc, KGv, prop)``. For the shells, a known constant
   membrane state can be given instead with ``update_KG_given_stress(Nxx,
   Nyy, Nxy, KGr, KGc, KGv)``, with the stress resultants in the element
   coordinate system of each element, see :doc:`conventions`;
3. the eigenvalue problem

   .. math::

       (\boldsymbol{K}_{C0} + \lambda \boldsymbol{K}_G) \boldsymbol{u} = \boldsymbol{0}

   solved by ``eigvals, eigvecs = linear_buckling(KC0uu, KGuu,
   num_eigvalues=...)``.

`\boldsymbol{K}_G` is linear in `\boldsymbol{u}_0`, hence in the reference load, and the
critical load is `\lambda_1 \boldsymbol{F}_{ref}`, with `\lambda_1` = ``eigvals[0]``.
:func:`.linear_buckling` returns the positive load multipliers first, in
increasing order, followed, when fewer than ``num_eigvalues`` positive ones
exist, by the negative ones closest to zero, which are the critical load
multipliers of the reversed reference load. A load that is not scaled by
`\lambda`, e.g. a preload with the geometric stiffness matrix `\boldsymbol{K}_{G0}`,
belongs to the first matrix, ``linear_buckling(KC0uu + KG0uu, KGuu)``,
which is also how the critical value of one load component is found under
the other components kept constant.

The beams have ``update_KG`` for the axial force of the prestress, see
``tests/test_beam_linear_buckling_3D_column.py``; :class:`.Truss` and
:class:`.Spring` have no geometric stiffness matrix.

The geometric stiffness matrices of :class:`.Quad4` and :class:`.Quad4R`
integrate the membrane stress resultants with two-by-two quadrature and take
the slopes of `w` at the centroid of the element, where the transverse shear
of :class:`.Quad4R` and of the thin :class:`.Quad4` elements is sampled. The
hourglass pattern of `w`, alternating in sign from node to node with zero
rotations, which that transverse shear does not resist, then has no
geometric stiffness either; with the slopes at the two-by-two points, as up
to version 0.11, it could give a spurious buckling mode below the physical
load when the transverse shear stiffness is low compared with the bending
stiffness, e.g. in sandwich plates. The buckling loads of coarse meshes are
somewhat higher, see the Notes of :meth:`.Quad4.update_KG` and
:meth:`.Quad4R.update_KG`.


Natural frequencies
-------------------

The frequency problem

.. math::

    (\boldsymbol{K} - \omega_n^2 \boldsymbol{M}) \boldsymbol{u} = \boldsymbol{0}

is solved by ``omegan, eigvecs = natural_frequency(Kuu, Muu,
num_eigvalues=...)``, which returns the circular frequencies `\omega_n`,
in rad/s with consistent units, in increasing order. `\boldsymbol{K}` is
`\boldsymbol{K}_{C0}`, plus the geometric stiffness matrix of a prestress when it
applies, e.g. ``KC0uu + KGuu``, see
``tests/test_quad4r_natural_freq_pre_stress.py``. The mass matrix is
assembled with ``update_M``, whose ``mtype`` selects the mass matrix of
each element, see :doc:`element_choice`.

With an unsymmetric `\boldsymbol{K}`, e.g. with the aerodynamic stiffness matrix of
the piston theory or the load stiffness of a follower load, the
frequencies may become complex; ``omegan`` is then the principal square
root of the complex eigenvalues. Two frequencies that coalesce and become
complex indicate the loss of stability by flutter, which
:func:`.linear_buckling` does not detect: this is the dynamic method of
Bolotin (1963), see :func:`.natural_frequency`.

:func:`.natural_frequency` solves the undamped problem.


Damped frequencies
------------------

With a damping matrix `\boldsymbol{C}`, the free vibrations `\boldsymbol{u} e^{\lambda t}`
give the quadratic eigenvalue problem

.. math::

    (\lambda^2 \boldsymbol{M} + \lambda \boldsymbol{C} + \boldsymbol{K}) \boldsymbol{u} = \boldsymbol{0}

solved by ``eigvals, eigvecs = complex_frequency(Kuu, Cuu, Muu,
num_eigvalues=...)``, which returns the complex eigenvalues `\lambda`,
one per complex conjugate pair, with `Im(\lambda) \ge 0`, and the complex
eigenvectors. The damped circular frequency is `\omega_d = |Im(\lambda)|`,
``abs(eigvals.imag)``, the damping ratio `\zeta = -Re(\lambda)/|\lambda|`,
``-eigvals.real/abs(eigvals)``, and a positive `Re(\lambda)` is a growing
motion. With `\boldsymbol{C} = \boldsymbol{0}`, `\lambda = i \omega_n`, with the `\omega_n` of
:func:`.natural_frequency`.

The damping matrix is assembled by the user, e.g. Rayleigh damping
`\boldsymbol{C} = \alpha \boldsymbol{M} + \beta \boldsymbol{K}`, or the aerodynamic damping of the piston
theory, ``Cuu = -mu*CAuu`` with ``update_CA`` of :class:`.Quad4` and
:class:`.Quad4R`, with the aerodynamic stiffness in ``Kuu = KC0uu -
beta*KA_betauu - gamma*KA_gammauu``, see ``update_KA_beta``. A flutter
analysis then follows the largest `Re(\lambda)` over the flow speed: the
aerodynamic damping moves the flutter onset beyond the coalescence of the
frequencies of :func:`.natural_frequency`, see
``tests/test_quad4_piston_theory.py``.


Verification and dense fallback
-------------------------------

The three functions check that `\boldsymbol{K}` has no null row or column, a dof
without stiffness, which raises a :class:`numpy.linalg.LinAlgError` naming
the dofs, see :func:`.check_null_rows_cols`, whereas `\boldsymbol{K}_G`, `\boldsymbol{C}` and
`\boldsymbol{M}` may have null rows, e.g. a lumped mass matrix without rotary
inertia. They equilibrate the diagonal of the matrices with
:func:`.diagonal_preconditioner`, which does not change the eigenvalues,
and :func:`.linear_buckling` and :func:`.natural_frequency` select the
solvers of symmetric or unsymmetric matrices with :func:`.is_symmetric`.
Then:

- the sparse solvers of SciPy run first, in shift-invert or Cayley mode,
  with a shift estimated from the matrices for the linear buckling,
  :func:`.estimate_cayley_sigma`;
- every result is verified: the relative residual of each eigenpair and,
  for symmetric matrices, the absence of an eigenvalue missed below the
  highest one returned, from the count of the negative pivots of a shifted
  matrix, see :func:`.check_eigenpairs` and :func:`.is_positive_definite`;
  an eigenpair of :func:`.natural_frequency` or :func:`.complex_frequency`
  whose relative residual is too large still passes when its backward
  error is at the round-off, e.g. an eigenvalue close to zero of a model
  pre-stressed close to buckling, see
  :func:`pyfe3d.solver.eigen._backward_errors`;
  for the complex eigenvalues of :func:`.complex_frequency` no such count
  is implemented, and only the residuals are verified;
- when the sparse solvers fail, a dense solver of SciPy is used as a last
  resort, when the dense problem has at most ``max_dense_size`` dofs, with a
  :class:`.DenseFallbackWarning` giving the reasons; otherwise a
  ``RuntimeError`` lists every failure, such that no unverified result is
  returned silently;
- ``sparse_solver=False`` solves the full problem with a dense solver,
  returning all the eigenvalues, as a reference for validations.

The fallback can be turned into an error, e.g. in a test suite::

    import warnings
    from pyfe3d.solver import DenseFallbackWarning
    warnings.simplefilter('error', DenseFallbackWarning)

With Intel MKL 2024.2.0 to 2025.0.0, the symmetric solver of ARPACK
returns wrong eigenpairs when its number of Lanczos vectors exceeds 32;
these versions are detected and the number is capped, see
:mod:`pyfe3d.solver.arpackutils`, also for the override with the
environment variable ``PYFE3D_ARPACK_MAX_NCV``.


Examples
--------

The reference load, a preload and the reversed reference load of a
:class:`.Quad4` plate:

.. literalinclude:: ../../tests/test_solver_linear_buckling.py
    :pyobject: test_quad4_plate_reference_load

The frequencies of the same plate under a compressive prestress, whose
lowest one vanishes at the critical load multiplier:

.. literalinclude:: ../../tests/test_solver_natural_frequency.py
    :pyobject: test_quad4_plate_prestress

The damped eigenvalues of a chain of masses with Rayleigh damping, against
the analytical modal values:

.. literalinclude:: ../../tests/test_solver_complex_frequency.py
    :pyobject: test_rayleigh_damping_chain

Complete models are in :doc:`ex_linear_buckling_plate`,
:doc:`ex_linear_buckling_cylinder_torsion`, :doc:`ex_natural_frequency_plate`
and :doc:`ex_natural_frequency_cylinder`.
