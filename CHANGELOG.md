# Changelog

## 0.11.0 (2026-10-24)

### Python versions

- Python 3.9 is no longer supported, it reached its end of life in October 2025: Python 3.10 or newer is required, and the tests and wheels cover Python 3.10 to 3.14.

### Packaging

- Packaging: the metadata, with the version, in `pyproject.toml` (PEP 621), built with `setuptools>=77` and Cython; `setup.py` keeps only the Cython extensions with the compiler flags and writes `pyfe3d/version.py` from `pyproject.toml`. NumPy and SciPy, imported by pyfe3d, are now declared as run-time dependencies and are no longer build requirements; the license is given as the SPDX expression `BSD-3-Clause` instead of a classifier, wheel is no longer a build requirement, and README.md, AUTHORS and LICENSE are no longer installed in the root of the Python environment (`data_files`).

### Non-linear static solvers in `pyfe3d.solver`

`pyfe3d.solver` becomes a package; every import of the previous module keeps
working. The features of structsolve are implemented in it, without
depending on structsolve:

- `pyfe3d.solver.static`: `solve()` and `linear_static()`;
- `pyfe3d.solver.analysis.Analysis`: linear and non-linear static analyses
  of a model given by the callables `calc_fext`, `calc_fint`, `calc_kC` and
  `calc_kG`, with the Newton-Raphson method, full or modified, with an
  optional line search, and the arc-length methods of Riks and of
  Crisfield; configuration-dependent loads with unsymmetric tangent
  stiffness matrices and load-factor-dependent load vectors, see
  `pyfe3d.solver.callbacks`; with `bu` and `uk`, the callables work with
  all the degrees-of-freedom, as assembled by the elements, and prescribed
  displacements are imposed exactly, see `FreeDofs`;
- `pyfe3d.solver.arc_length`, both methods: a converged step whose
  increment points against the tangent predictor, i.e. a negative or null
  inner product of the arc-length constraint, is rejected and repeated with
  a shorter arc length, as in structsolve. The previous converged point is
  at the distance of the arc length from the last one, and the iterations
  could converge to it and then trace the path backwards ("doubling
  back"): in the development version of 0.11.0, the von Mises truss of the
  tests at 10 times its limit load with `initialInc=0.2` was traced
  backwards from the second step, which returned to the origin with Riks'
  method, and ended at load factors of about -15 instead of 1.0, with both
  methods and both convergence criteria. New test
  `test_arc_length_no_doubling_back`;
- `pyfe3d.solver.convergence`: Crisfield's convergence criterion, the
  default of the non-linear solvers, the residual scaled with the diagonal
  of the linear stiffness matrix over the free degrees-of-freedom; the
  relative criterion remains available with `convergence='relative'`;
- `pyfe3d.solver.convergence`, round-off floor, as in structsolve 0.9.3:
  with both criteria, an iteration has also converged when the norm of its
  residual, `||R||_d` with `'crisfield'` and the Euclidean norm with
  `'relative'`, is not larger than `ROUNDOFF_FACTOR = 3` times the machine
  precision times the same norm of `|K_C0| |u|`, the product of the
  elementwise absolute values of the linear stiffness matrix and of the
  displacements of the free degrees-of-freedom, below which the residual
  cannot be computed. Like `absTOL`, it is an alternative to `relTOL`;
  `zeta` is unchanged, and the log of the iterations marks those converged
  at the round-off floor. With prescribed displacements, `|K_ff| |u_f|`
  omits `|K_fp| |u_p|`: the floor is then lower, i.e. conservative. In the
  development version of 0.11.0, the arc-length methods stopped on a low
  sinusoidal arch of 40 `BeamC` elements (Fung and Kaplan, NACA TN 2840,
  1952) at load factors of 0.026 to 0.032, close to the null load between
  the limit points, where the residual of the large self-equilibrated
  axial forces stagnates at 0.29 to 0.47 times `eps ||(|K_C0| |u|)||_d`
  above `relTOL = 1e-9`, and with 160 elements before the upper limit
  point; they now reach the load factor 1.0. New `roundoff_floor()`,
  keyword arguments `Kc_abs` and `full_output` of `check_convergence()`,
  whose defaults keep the former behaviour, `K0_abs` of
  `newton_raphson._NR_iterations` and `newton_raphson._abs_matrix`. Over
  the test suite the floor changes the verdict of 76 iterations: 69 of the
  new test of the arch, 2 of the new unit test, 4 of
  `test_crisfield_criterion_units_invariance[plate-Quad4R]`, whose
  relative criterion with the rotations in micro-units now converges in
  four increments instead of stopping without any, and 1 of
  `test_piston_theory_load_unsymmetric_tangent` (`relTOL = 1e-12`), all
  with residuals of at most 3 times `eps ||(|K_C0| |u|)||`.
  `test_nr_unreachable_tolerance_stops_without_increments` and
  `test_displacement_control_without_lbd_does_not_converge` set
  `ROUNDOFF_FACTOR = 0`, which disables the floor, to test the behaviour
  without it. New tests `tests/test_solver_arc_length_fung_kaplan.py`,
  both arc-length methods and both criteria at the default tolerances: the
  load factor 1.0 is reached, both limit loads within `0.01 R_cr` of Eq.
  (25) of Fung and Kaplan, the 40-element mesh being 0.0094 and 0.040 off
  them, and at least one iteration is accepted by the floor; the modal
  model of the same arch is tested in structsolve,
  `tests/test_arc_length_fung_kaplan.py`; and
  `test_check_convergence_roundoff_floor`;
- singular matrices raise `numpy.linalg.LinAlgError`: the solvers do not
  remove null rows and columns. A null row or column of the matrix given
  to `solve()`, to the factorization of the non-linear solvers, to
  `crisfield_weights()`, or of `K` in `linear_buckling()` and
  `natural_frequency()`, is a degree-of-freedom without stiffness (an
  unattached node, an unstiffened drilling rotation, a missing boundary
  condition), i.e. a modelling error, and the message gives the number of
  these degrees-of-freedom, their first indices and how to correct the
  model, see the new `pyfe3d.solver.check_null_rows_cols()`; `KG` and `M`
  may have null rows and columns, e.g. a lumped mass matrix without rotary
  inertia. An exactly singular matrix also raises in `solve()`, instead of
  the `MatrixRankWarning` and NaN values of
  `scipy.sparse.linalg.spsolve()`. The constrained degrees-of-freedom are
  partitioned out by the user, `KC0uu = KC0[bu, :][:, bu]`, or with
  `Analysis(..., bu=bu)`. This deliberately differs from structsolve, whose
  semi-analytical Ritz models apply the boundary conditions by zeroing
  approximation functions and whose solvers remove the null rows and
  columns that this leaves;
- `linear_buckling()` and `natural_frequency()`: unsymmetric matrices, the
  negative load multipliers closest to zero after the positive ones, the
  verification of every eigenpair with the sparse solvers first and a dense
  solver as a last resort (`DenseFallbackWarning`), new keyword arguments
  after the existing ones, whose defaults are unchanged. `check_eigenpairs()`
  now also detects load multipliers missing below the highest one returned.
  `diagonal_preconditioner()` uses the modulus of the diagonal entries.
- `linear_buckling()`, `check_eigenpairs()` and `natural_frequency()`
  verify the relative residual `||K u + lambda B u||/(||K u|| + |lambda|
  ||B u||)` of each eigenpair, with `B = KG`, or `B = M` and `lambda =
  -omega_n^2`, without any floor of its denominator. Only the residual of a rigid-body mode of
  `natural_frequency()`, whose `||K u|| + |omega_n^2| ||M u||` is not larger
  than `pyfe3d.solver.eigen.RIGID_BODY_RTOL = 100 eps` times the
  matrix-norm scale `(||K||_F + |omega_n^2| ||M||_F) ||u||`, is relative to
  this scale, i.e. the backward error, see
  `pyfe3d.solver.eigen._relative_residuals(rigid_rtol)`: the relative
  residual of the rigid-body modes, `omega_n^2 = 0` and `K u = 0` up to the
  round-off, is about 1. The floor of `1e-8` times the matrix-norm scale
  for every eigenpair, in the development version of 0.11.0, loosened the
  check about 1000 times for models with stiff penalty springs, whose
  `||K||_F` is many orders of magnitude larger than `||K u||`: e.g. for a
  blade-stiffened plate with penalty connections, `linear_buckling(...,
  precondition=False)` accepted eigenpairs of `eigsh` with relative
  residuals of 1 to 2 % and a lowest load multiplier 0.08 % too low; they
  are now rejected and computed with the dense fallback. New test data
  `tests/data/blade_stiffened_penalty_K.npz` and `_KG.npz`.
- `natural_frequency()` and `complex_frequency()`: an eigenpair whose
  relative residual is larger than `check_rtol` passes the verification
  when its normwise backward error, `||K u - omega_n^2 M u||/((||K||_F +
  |omega_n^2| ||M||_F) ||u||)`, or `||(lambda^2 M + lambda C + K) u||/
  ((|lambda|^2 ||M||_F + |lambda| ||C||_F + ||K||_F) ||u||)`, is not larger
  than `pyfe3d.solver.eigen.BACKWARD_ERROR_RTOL`, the machine precision,
  i.e. when it is accurate up to the round-off, as in structsolve (issue
  #49, structsolve issue #14). The genuine eigenpairs with an eigenvalue
  close to zero of a nearly singular `K`, e.g. pre-stressed close to
  buckling, failed the verification of the sparse solvers and of the dense
  fallback, raising a `RuntimeError` unless a larger `check_rtol` was
  given: for the cantilever of
  `tests/test_beamc_natural_freq_cantilever_pre_stress.py` at 0.9999 of
  its buckling load, `omega_n**2 = 0.0895` with relative residuals of
  2.3e-6 to 1.3e-5 and backward errors of at most 0.063 times the machine
  precision; the test no longer needs `check_rtol=1e-4`. The inaccurate
  eigenpairs of the penalty models above, with backward errors of 14 times
  the machine precision and more, are still rejected; `linear_buckling()`
  and `check_eigenpairs()` keep the relative residual only. New
  `pyfe3d.solver.eigen._backward_errors` and `_quadratic_backward_errors`,
  and test data `tests/data/beam_cantilever_K.npz`, `_KG.npz` and `_M.npz`.

### Damped frequencies: `complex_frequency()`

- New `pyfe3d.solver.complex_frequency(K, C, M, ...)`: the complex
  eigenvalues `lambda` of the damped free vibrations, `(lambda**2 M +
  lambda C + K) u = 0`, with the damped circular frequency
  `abs(lambda.imag)` and the damping ratio `-lambda.real/abs(lambda)`;
  one eigenvalue per complex conjugate pair, `lambda.imag >= 0`, the
  `num_eigvalues` nearest the shift, sorted by increasing imaginary part.
  `K`, `C` and `M` are real, symmetric or not, e.g. with the matrices of
  the piston theory or gyroscopic terms. The first companion linearization
  of size `2n` (Tisseur and Meerbergen, SIAM Review, 2001) is solved with
  `eigs` in shift-invert mode, which factorizes only `sigma**2 M + sigma C
  + K`; the default shift `sigma = 1` rad/s is the counterpart of the shift
  `omega_n**2 = -1` of `natural_frequency()`, and a real or complex shift
  can be given. Same conventions and solver policy as
  `natural_frequency()`: no null row or column of `K`, the relative
  residual of every eigenpair verified against `check_rtol`, and the dense
  fallback, `scipy.linalg.eig` on the linearization, of size `2n` at most
  `max_dense_size`, with a `DenseFallbackWarning`. Whether an eigenvalue
  nearer the shift was missed is not verified, no inertia count existing
  for complex eigenvalues. With `C = 0`, `lambda = 1j*omega_n` of
  `natural_frequency()`, which is unchanged. New
  `tests/test_solver_complex_frequency.py`.
- `tests/test_quad4_piston_theory.py` and
  `tests/test_quad4r_piston_theory.py` include the aerodynamic damping `C
  = -mu*CA` and now assert their results: for the flat plate `C` is
  `a = mu/(rho h)` times the mass matrix of `w`, such that the damped
  eigenvalues follow from the undamped ones, `lambda = -a/2 +- sqrt(a**2/4
  - omega**2)`, verified at every Mach number; every mode grows below Mach
  `sqrt(2)`, where `mu < 0`, and the flutter onset moves from the
  coalescence of the frequencies, Mach 2.804 (`Quad4`) and 2.890
  (`Quad4R`), to Mach 2.824 and 2.910 with the damping.
- `pyfe3d.solver` no longer uses `warnings.catch_warnings()`, whose exit
  resets the registry of the warnings already shown, of every module: a
  warning of the caller, shown once per location with the `'default'`
  action, was shown again after each call of `linear_buckling()`,
  `natural_frequency()`, `complex_frequency()`, `check_eigenpairs()`,
  `is_positive_definite()`, `estimate_cayley_sigma()`, `solve()`,
  `linear_static()` or of a non-linear `Analysis`. The blocks around the
  LU factorizations suppressed no warning, `scipy.sparse.linalg.splu`
  warning only for a matrix not in the CSC format, and `solve()` now
  factorizes with `splu`, an exactly singular matrix raising the same
  `LinAlgError` instead of the `MatrixRankWarning` of `spsolve` turned
  into an error; its keyword arguments are passed to `splu`. New
  `tests/test_solver_warnings_shown_once.py`.

### Breaking: `pyfe3d.shellprop` mirrors `composites.Laminate`

`ShellProp`, `Lamina`, `MatLamina`, the lamination-parameter functions and
`GradABD` now have the names and the behaviour of `composites.Laminate`,
`Lamina`, `MatLamina` and of the functions of `composites.core`, implemented
in pyfe3d, which does not depend on composites. The class `ShellProp`, the
functions `shellprop_from_LaminationParameters` and
`shellprop_from_lamination_parameters`, and the element-frame methods
`get_constitutive_element`, `calc_constitutive_element` and
`calc_Ats_element` keep their names. Renamed or removed, without aliases:

- `ShellProp.force_balanced()`, `force_orthotropic()`, `force_symmetric()`
  are `make_balanced()`, `make_orthotropic()`, `make_symmetric()`;
- `force_balanced_LP()`, `force_orthotropic_LP()`, `force_symmetric_LP()`
  are `make_balanced_LP()`, `make_orthotropic_LP()`, `make_symmetric_LP()`;
- `Lamina.get_transf_matrix_displ_to_laminate()` is
  `Lamina.get_transf_matrix_displ_to_lamina()`, the matrix mapping the
  displacements from the laminate axes to the material axes of the ply, as
  before; the old name said the opposite;
- the private `_rohwer()` has no `offset` argument and the private attribute
  `ShellProp._ts_offset` is removed from `shellprop.pxd` and from the pickled
  state.

Changed behaviour:

- `make_orthotropic()` also sets `A45`, `Abar45` and `Abarbar45` (unless
  `nan`, i.e. when a ply has a singular transverse shear stiffness matrix,
  as `make_balanced()`) to zero, and `make_orthotropic_LP()` also sets
  `xiAts2` to zero, the terms that vanish when all plies are at 0 or 90
  degrees;
- a new `ShellProp()` has `scf_k13 = scf_k23 = 5/6` until
  `calc_constitutive_matrix()` is called, instead of 1;
- the transverse shear stiffnesses and the through-thickness distribution of
  `calc_transverse_shear_stress()` are computed with `z` measured from the
  mid-surface, giving the same values to round-off and avoiding the
  round-off growing with `offset` for large offsets.

New:

- `shear_correction` accepts `'whitney'`, `'chow'`, `'birman_bert'` and
  `'thickness_shear'`, besides `'rohwer'`, `'vlachoutsis'`, `'constant'` and
  `None`;
- `ShellProp.make_smeared()`, `calc_equilibrium_transverse_shear()` and
  `calc_aposteriori_energy()`;
- `GradABD` can be pickled.

### Behaviour change: hourglass control of `Quad4R` for laminates

- The hourglass stiffnesses of `w`, `r_x` and `r_y` of `Quad4R`, Brockman's
  (1987) `0.1 E t^3` times the area factors, now use for `E` the equivalent
  bending moduli `E1b = 12/(h^3 d11)` and `E2b = 12/(h^3 d22)`, with `[d] =
  (D - B A^-1 B)^-1` the lower-right block of `[[A, B], [B, D]]^-1` in the
  element coordinate system, `E2b` for `r_x`, `E1b` for `r_y` and their mean
  for `w`. Before, they used the membrane moduli `E1eq = 1/(h a11)` and
  `E2eq = 1/(h a22)`, `[a] = [A]^-1`, which remain those of `u` and `v`.
  `E1b` is the modulus of the homogeneous plate that bends as the laminate
  under `M_xx` alone, as `E1eq` is the one that stretches as the laminate
  under `N_xx` alone; both are `E` for an isotropic plate, with or without
  `offset`, and do not depend on the reference surface. `12 D11/h^3` would
  give `E/(1 - nu^2)`, and `[D]^-1` would depend on `offset`. Isotropic
  results are unchanged to round-off: the element matrices and internal
  forces differ by at most `6e-19` relative to their largest entry. For a
  laminate, the artificial stiffness of the rotations becomes the same
  fraction of its bending stiffness as for an isotropic plate. It decreases
  where `E_b < E_eq`, e.g. `E1b = E2b = 0.36 E1eq` for an unsymmetric
  `[0/90]`, through `[B]`, and increases where `E_b > E_eq`, e.g. `E1b =
  1.7 E1eq` for a symmetric `[0/90]s` (`E2b = 0.31 E2eq`) and `2.7 E1eq`
  for a sandwich. Centre deflection of simply supported plates under
  uniform pressure, 8 by 8 elements, over the FSDT Navier solution, before
  and after: unsymmetric `[0/90]` 0.682 and 0.847, symmetric `[0/90]s`
  0.855 and 0.852, sandwich 0.943 and 0.861, isotropic 0.921 and 0.921.
  The previous stiffnesses are recovered with `hgfactor_w = (E1eq +
  E2eq)/(E1b + E2b)`, `hgfactor_rx = E2eq/E2b` and `hgfactor_ry =
  E1eq/E1b`. Documented in the Notes of `pyfe3d.quad4r` and of
  `Quad4R.update_KC0` and on the page on the choice of the element.
- New tests in `tests/test_quad4r_hourglass_control.py`: the five
  hourglass stiffnesses of six plates, among them an isotropic plate with
  an offset, an unsymmetric angle-ply with an offset and a sandwich, with
  the material direction along the element axis and at 30 degrees from it;
  the ratios of the bending to the membrane moduli; the zero eigenvalues of
  one element, 12 without and 7 with the hourglass control, for these
  plates on a rectangle and on a distorted element; the simply supported
  plates above against the Navier solution.
- `tests/test_quad4r_natural_freq_cylinder.py`, unsymmetric laminate: the
  recorded frequencies are 0.4, 0.7 and 0.2 % lower, and differ from those
  of the same mesh of `Quad4` by 2.0, 2.5 and 1.3 % instead of 2.4, 3.2 and
  1.5 %.
- `theory/deriving_equations/quad4r/derive_quad4r_KC0_sparse.py` and
  `derive_quad4r_fint.py`: the hourglass coefficients with `E1b` and `E2b`;
  `derive_quad4r_fint.py` also with the area factors of the code, where it
  still had Brockman's `1/(1 + 1/A)` for all five coefficients and `Ew =
  (Erx + Ery)/2`.

### `BeamProp`: a transverse shear modulus per plane and the torsional stiffness

- New attributes of `BeamProp`: `G12` and `G13`, the transverse shear
  moduli of the element planes `xy` (bending with `v`, `Izz`) and `xz`
  (bending with `w`, `Iyy`), and `GJ`, the torsional stiffness. Until they
  are set they follow `G`: `G12 = G13 = G` and `GJ = G*J`, with the values
  of `G` and `J` at the time they are read; `del prop.G12` (or `G13`, `GJ`)
  makes the attribute follow `G` again. A property given with `G` and `J`
  only keeps its meaning, and the shear correction factors of the two
  planes can now be applied to `G12` and `G13` without scaling the
  torsional stiffness.
- `BeamC` and `BeamLR` use `G12*A` and `G13*A` for the transverse shear of
  each plane, and in the coupling of the shear strains with the twist
  through `Az` and `Ay`; `BeamC` uses them in `alphay =
  -12*E*Izz/(G12*A*L**2)` and `alphaz = -12*E*Iyy/(G13*A*L**2)`, i.e. in
  `KC0`, `KG`, `M` and the internal forces. `BeamC`, `BeamLR` and `Truss`
  use `GJ` in place of `G*J`. With the rigid cross section the shear block
  of the sectional matrix is `[[G12 A, 0, -G12 Az], [0, G13 A, G13 Ay],
  [-G12 Az, G13 Ay, G12 Iyy + G13 Izz]]`, in which `GJ`, given by the user,
  replaces the last entry.
- With `G12 = G13 = G` and `GJ = G*J`, set or not, the matrices `KC0`,
  `KCNL`, `KG` and `M` of `BeamC`, `BeamLR` and `Truss`, and the internal
  forces of `BeamC` and `Truss`, are bit-identical to those of the previous
  version (Windows build); the internal forces of `BeamLR` agree to
  round-off (`1e-16` relative), because their regenerated lines group the
  terms differently, see "Derivation scripts". New test
  `tests/test_beam_shear_moduli.py`, with the reference arrays of the
  previous version in `tests/data/beam_single_G_reference.npz`, the
  cantilever deflection of Bathe (2014), Eq. (1.2), and the frequencies of
  Petyt (2010), Example 3.10 and Table 3.6, in each plane with its own
  modulus, and the twist `T L/GJ` independent of `G12` and `G13`.

### Bug fixes

- `BeamC`: the shear parameters `alphay` and `alphaz` are
  `-12*E*I/(G*A*L**2)`, the sign with which the consistent shape functions
  of Luo (2008) satisfy the Timoshenko equilibrium, instead of the positive
  sign of Eq. (21) of that paper, in `update_probe_finte`, `update_KC0`,
  `update_fint`, `update_KG` and `update_M`. One element of a cantilever gave
  a tip deflection of 429.07, stiffer than Euler-Bernoulli (493.83), instead
  of the exact Timoshenko value 522.716. New tests
  `tests/test_beamc_timoshenko_shape_functions.py` and
  `tests/test_timoshenko_beam_literature.py`, with benchmarks of Bathe
  (2014) and Petyt (2010).
- `Quad4` and `Quad4R`, `update_KA_beta`: the slope along the flow is
  `N,x*r11 + N,y*r12`, instead of `N,x*r11 + N,y*r21`, which was wrong for
  elements whose `x` axis is not along `X`. New
  `tests/test_piston_theory_flow_direction.py`.
- `MatLamina`: `c11` used `nu23*nu23` instead of `nu23*nu32` (4.26% too high
  for `E2 != E3`) and `q31` used `nu12*nu32` instead of `nu12*nu23` (4.65%
  too high). New `tests/test_matlamina_stiffness.py`, against the inverse of
  the compliance matrix.
- `GradABD.calc_LP_grad` left the derivatives of `A66`, `B66` and `D66` with
  respect to the lamination parameters at zero. New `tests/test_gradabd.py`,
  against central finite differences.
- `ShellProp.make_balanced()` set to zero `B16` and `B26`, which do not
  vanish in a balanced laminate in general, e.g. in an antisymmetric
  `[+45/-45]` laminate, and left `Abar45`, `Abarbar45` and `A45`, which
  vanish in every balanced laminate. It now sets to zero only the terms that
  vanish for any position of the `+theta` and `-theta` plies, the integrals
  of `z^0` of `q16L`, `q26L` and `q45L`: `A16`, `A26`, `Abar45`,
  `Abarbar45` (unless `nan`), and `A45` except with
  `shear_correction='rohwer'`, whose `A45` does not vanish in general, e.g.
  for a `[+30/-30/0]s` laminate. `B16`, `B26`, `D16` and `D26` keep the
  values of `calc_constitutive_matrix()`. `make_balanced_LP()` also sets
  `xiAts2` to zero, as `Abar45`. New tests in `tests/test_shellprop.py`.
- The curved beam frequency tests clamped an extra node with fine meshes;
  the end nodes are now selected by index.
- `linear_buckling()`, `natural_frequency()` and
  `pyfe3d.solver.arpackutils.capped_eigsh`: the calls of ARPACK are
  reproducible with SciPy 1.17 or newer. When the Krylov subspace is
  invariant, e.g. with a nilpotent `K^-1 KG`, ARPACK draws new starting
  vectors from the argument `rng` of `eigs` and `eigsh`, by default from
  the entropy of the operating system: with the same `v0` the results
  differed between calls, e.g. `linear_buckling()` on Ziegler's double
  pendulum raised a `RuntimeError` or returned a spurious load multiplier
  of about `1e16` at random. New `pyfe3d.solver.arpackutils.start_kwargs(n)`,
  the arguments `v0` and, when supported, `rng` with the seed `42`, used by
  every call of ARPACK.
- `linear_buckling()` with unsymmetric matrices, sparse shift search: the
  eigenvalues `nu` of `(K + s KG)^-1 KG` found by `eigs` at each shift are
  null when `|nu| <= NULL_MU_RTOL max(max|nu|, ||(K + s KG)^-1 KG||)`, the
  norm estimated with one step of the power iteration from the starting
  vector of ARPACK, instead of `|nu| <= 1e-12 max|nu|`. When `K^-1 KG` has
  no non-null eigenvalue, e.g. nilpotent as with the follower load of
  Ziegler's double pendulum, all the `nu` are round-off and `max|nu|` is
  not a scale: spurious load multipliers were found, and
  `linear_buckling(max_dense_size=0)` raised the expected `RuntimeError`
  only when the parity check rejected them, depending on the platform and,
  with SciPy < 1.17, whose ARPACK keeps its restart seed from one call to
  the next, on the calls made before in the same process (SciPy 1.15.3: a
  load multiplier of `3e16` to `3e18` returned in 43 of 200 successive
  calls). They are now all null, and the shift search raises "no finite
  load multiplier was found". New `pyfe3d.solver.eigen._norm_estimate`.
  The same rule as structsolve.
- `linear_buckling()`, dense solvers (`sparse_solver=False` and the dense
  fallback, symmetric and unsymmetric): an eigenvalue `mu` of
  `KG u = mu K u` is also null when `|mu| <= 10 eps ||KG||_F kappa`,
  `kappa` its condition number from the left and right eigenvectors, i.e.
  when the round-off does not determine it, instead of only when
  `|mu| <= 1e-12 max|mu|`. The round-off splits a defective null
  eigenvalue, a Jordan block of size `m`, into `m` eigenvalues of modulus
  about `eps^(1/m)`, e.g. `+-9.4e-10` for Ziegler's double pendulum, where
  `K^-1 KG` is nilpotent: `linear_buckling()` returned the spurious load
  multipliers `+-8.05e8`, about `6.4e7` times the flutter load, instead of
  none. The null `mu` are set to 0 and their load multipliers are `-inf`,
  also in complex arrays; the dense fallback then finds no finite load
  multiplier and fails the verification. Well-conditioned eigenvalues are
  null only below `2.2e-15 ||KG||_F`. New
  `pyfe3d.solver.eigen._null_mu_pencil`, `NULL_MU_PERTURBATION_FACTOR`;
  argument `left` of `_eig_complex`. The same rule as structsolve.

### Breaking: one `mtype` numbering for every element

`mtype` of `update_M` has the same meaning for every element: `0`
consistent, `1` lumped and `2` reduced integration, the last only for the
elements that have it, `Quad4`, `Quad4R` and `Tria3R`. In 0.10.x, `mtype=1`
was the reduced-integration matrix of the shells and the lumped matrix of
the beams, and a model with one `mtype` for shells and beams got massless
beams with `mtype=2`. Changed, without a deprecation step:

- `Quad4`, `Quad4R`, `Tria3R`: lumped `2` → `1`, reduced integration
  `1` → `2`;
- `Tria3DSG`: lumped `2` → `1`; `1`, treated as `0` (consistent) in 0.10.x,
  is now the lumped matrix, and `2` raises;
- `BeamC`, `BeamLR`, `Truss`: unchanged, `0` consistent and `1` lumped; `2`
  raises.

Any other value raises a `ValueError` that names the element, the value
and the valid values, see below. Scripts that pass `mtype=1` or `mtype=2`
to the shells must swap the two values: they run without an error and
select the other matrix.

### Mass matrices

- Lumped mass matrices of the shells, `mtype=1` of `update_M` of `Quad4`,
  `Quad4R`, `Tria3R` and `Tria3DSG` (`mtype=2` before, see above): the coupling between the translations
  and the rotations through `intrhoz`, the first moment of the density about
  the reference surface, is kept in the block of each node. The lumped
  matrix is the nodal quadrature of the kinetic energy, the Lobatto
  integration at the nodes of Brockman (1987), which gives the complete
  inertia of the section at each node; it is diagonal only when `intrhoz`
  is zero. With an offset, the centre of mass of each element was on the
  reference surface instead of at the offset; the mass and the first moment
  of the mass are now exact (in the plane, for the triangles and the
  parallelograms). Without offset the matrices are unchanged. `mtype=1`
  writes 90 entries for `Tria3R` (162 before) and 120 for `Quad4` and
  `Quad4R` (288 before), generated with
  `theory/deriving_equations/tria3r/derive_tria3r_M_sparse.py` and
  `theory/deriving_equations/quad4r/derive_quad4r_M_sparse.py`.
- `Tria3DSG.update_M` uses `intrhoz`, coupling `u` with `r_y` and `v` with
  `r_x`, for the consistent and the lumped matrix, consistently with
  `intrhoz2`, which already included the offset.
- `update_M` of every element raises a `ValueError` for an unsupported
  `mtype`, checked before the code without the GIL: `0`, `1` or `2` for
  `Quad4`, `Quad4R` and `Tria3R`; `0` or `1` for `Tria3DSG`, `BeamC`,
  `BeamLR` and `Truss`. Before, `Tria3DSG` treated any value other than its
  lumped `2` as `0`, and the other elements wrote nothing for other values,
  e.g. `mtype=2` gave massless beams. The values are listed on the page on
  the choice of the element.
- New `tests/test_element_mass.py`: total mass, the matrices of the Notes and
  the rigid-body inertia of every element and every `mtype`, with an offset
  and with the first moments of the beam sections; the numbering of
  `mtype` for every element; unsupported `mtype`, `-1`, `3` and `2` for the
  elements without a reduced-integration matrix, and the message.
- New tutorial `tut-shell-lumped-mass.ipynb`: derivation of the lumped
  matrices with the offset and verification with the rigid-body inertia of
  one element and of a mesh.
- `Tria3R`: unused variables removed from `update_M`; comments and Notes
  corrected (the weight of the one-point rule, the vertex rule of the lumped
  matrix). The check scripts `derive_tria3r_mass.py` and
  `derive_quad4r_mass.py` are rewritten: the weights of the rules in the
  reference triangle sum to its area 1/2 (they gave twice the integrals),
  every rule and geometry is evaluated, and `intrhoz` is derived;
  `derive_tria3r_M_sparse.py` prints `valH1 = A/9` instead of `2*A/9`;
  `theory/README.md` lists the scripts that generate no code.

### Derivation scripts

- The SymPy scripts that generate the element code are moved from the
  repository pyfe3d-dev (commit 9b9971c) to `theory/`, with the derivations
  of the piston-theory slope and of the sign of the `BeamC` shear parameters
  corrected as above.
- The scripts of `beamc`, `beamLR` and `truss` use `G12`, `G13` and `GJ`;
  `derive_beamc_KC0_sparse.py` and `derive_beamLR_KC0_sparse.py` check the
  shear block of the sectional matrix against the integral over the rigid
  cross section. The lines of `update_KC0` and `update_probe_finte` of
  `BeamC`, `BeamLR` and `Truss` are regenerated from them.
  `derive_beamLR_fint.py` did not evaluate the shear terms at the centre of
  the element, `xi = 0`, as `derive_beamLR_KC0_sparse.py` and the code do,
  and printed a null global vector (wrong symbol names); fixed, and the
  internal forces of `BeamLR` are now its output.

### Clean-up

- Unused local variables, found with the Cython directives `warn.unused`
  and `warn.unused_result`, are removed from the elements and `shellprop`:
  declarations never used, e.g. the tying-point arrays of an assumed
  transverse shear of `Quad4` with their comment, which `Quad4` does not
  implement, copies of rotation-matrix terms and properties never read, and
  dead assignments. The matrices, internal forces and probe arrays of every
  element are bit-identical before and after.
- `pyfe3d.solver.eigen` no longer imports `scipy.sparse.linalg.eigsh`,
  which it did not use (the solvers call `capped_eigsh` of
  `pyfe3d.solver.arpackutils`).

### Documentation

- `tut-nonlinear-benchmarks.ipynb`: Williams' toggle frame (Williams, QJMAM
  17(4), 451-469, 1964) as reproduced in Fig. 7 of Karamanlidis, Honecker
  and Knothe (1981), with `BeamC` elements and both arc-length methods,
  against the load-deflection curve digitised from the figure; the limit
  loads converge with the mesh to 1.6 % and 2.9 % below those read from
  the figure.
- Functional docstrings of the public API, and Notes and References with
  the theory of the beams, truss, spring, shell elements, `shellprop` and
  the solver. The docstring of `Truss.update_M` had the wrong signs of the
  coupling terms of the consistent mass matrix; the code was right, new
  `tests/test_truss_consistent_mass.py`.
- Reddy's book is cited as Reddy (2003), https://doi.org/10.1201/b12409.
- `pyfe3d.solver.convergence`: the origin of the scaled norm is cited as
  the sources state it: Crisfield (1981, p. 58) attributes the scaling to
  Peano and Riccioni and uses it in his arc-length procedure, and Crisfield
  (1991, Sec. 9.5.4, Eq. 9.48) presents it as an alternative to the
  Euclidean norm of the residual, not as a recommendation.
- Bathe, *Finite Element Procedures*, is cited in its 2nd edition (2014,
  K. J. Bathe, Watertown, MA, ISBN 978-0-9790049-5-7), with the locations
  checked in it: the tip deflection of a cantilever with shear
  deformation, Sec. 1.2, p. 5, Eq. (1.2), Fig. 1.2(b), and the Sturm
  sequence check of `check_eigenpairs`, Sec. 10.2.2, p. 849, Sec. 11.4.3,
  p. 943, and Sec. 11.6, p. 954.
- Sources stated as they are, as in the book that documents the elements:
  the transverse shear stabilisation of `Tria3R` and `Tria3DSG` is the
  factor that Lyly, Stenberg and Vihinen proposed for the MITC4, called
  Stenberg's method by Bischoff and Bletzinger, in the form of Castro et
  al., applied in these references on top of assumed-strain or discrete
  shear gap fields; Eq. (24) of Bischoff and Bletzinger gives
  `0.17/(2(1+nu))`, i.e. `0.065` for a square element, not `0.12`, and
  their Eq. (22) `kappa/(2(1+nu)) = 0.32`; the `0.067` and `7e-8` of
  `Tria3DSG` hold for the leg of the right triangles of its test, `0.125`
  and `1.4e-7` for the longest edge; the shear gap is the part of the
  deflection due to shear, Eq. (3) of Bletzinger, Bischoff and Ramm. In
  `Quad4`, the thick-plate rule of Hughes et al. (1977) integrates with
  two-by-two only the products of the gradients of `w`, and the rank
  deficiency of Allman's enrichment alone is a single global mode that one
  prescribed drilling rotation suppresses (Allman, 1984, Sec. 5). In
  `BeamC`, Eq. (20) of Luo (2008) prints the functions of `w` and `r_y`
  with the signs of those of `v` and `r_z`. In `pyfe3d.solver.convergence`,
  the dependence of the Euclidean norm on the units is no longer attributed
  to Crisfield (1991).
- The load stiffness matrix of a follower load is written `[K_F]` instead
  of `[K_L]`, as in the book.
- The page on the non-linear static analyses gives the cases in which
  `[K_F]` is symmetric, as Hibbitt (1979, pp. 938-939) and Schweizerhof and
  Ramm (1984, Table 2, Sec. 3.2.1, Eqs. 32-35) state them, and that pyfe3d
  keeps the unsymmetric `[K_F]` and factorizes the tangent stiffness matrix
  with LU.
- The transformation matrix of an element is written `[T]` instead of
  `[R]`, which is the residual, as in the book: `[T]` is block diagonal
  with one `3x3` block `[T_{3x3}]` per node and per translations or
  rotations, whose entries `T_{ij}` are stored in the attributes
  `r11, ..., r33` of the elements, with the same indices and unchanged
  names, and whose columns are the element axes in global components,
  such that `{u} = [T]{u_e}`, `{u_e} = [T]^T{u}`, `[K] = [T][K_e][T]^T`
  and `{F} = [T]{F_e}`, in the conventions and overview pages, the Notes
  and attributes of the elements, the code comments and the tests. The
  bubbles of the Allman enrichment are written `\hat{S}_k` instead of
  `N_k`.
- In `pyfe3d.solver`, the basis of the null space of the constraints of
  the prescribed displacements imposed exactly is written `[Z]` instead
  of `[T]`, which is the transformation matrix of the elements:
  `{u} = [Z]{c} + lambda {u_k}` in `Analysis`, `{c} = [Z]{c_r} + {c_p0}
  + lambda {c_p1}` and `[Z]^T` in the reduced residual in
  `pyfe3d.solver.callbacks`, and the variable `Z` of
  `tests/test_solver_load_factor_dependent_load.py`. No public name
  changes: the matrix is not an argument, attribute or function of the
  API.
- Notation of the documentation as in the book, without code changes: the
  scalar shape functions are `S_i` instead of `N_i` (`N` stays for the
  stress resultants); the drilling operator and energy of the penalty
  model are `B_{r_z}` and `U_{r_z}` instead of `B_{drill}` and
  `U_{drill}`, and the drilling stiffness of the physics-based model
  `K_{C_e}^{r_z}`; the sectional constitutive matrix of the beams is
  `[C_{sec}]` instead of `[D]`; the variation of the nonlinear membrane
  strains of the shells is `[B_{NL}]` instead of `[B_{mL}]`, as for the
  beams; bold symbols use `\boldsymbol`, with the accents outside, e.g.
  `\tilde{\boldsymbol{S}}`, instead of `\pmb`, and the transpose is
  `^\top` instead of `^T`, in the docstrings, the user guide and the
  tutorials. A matrix in the docstring of `Tria3R.update_KG_given_stress`
  had a single backslash as row separator.
- User guide pages: overview and workflow, conventions, choice of the
  element, shell and beam properties, non-linear static analyses, linear
  buckling and natural frequencies, installation and changelog. The usage
  examples include the test functions, with the new
  `tests/test_beamc_frame.py` (L-shaped frame against the unit-load
  method) and `test_offset_isotropic_plate`. The module docstrings have
  overlined titles, and Sphinx builds without warnings.
- Sensitivity of the hinged cylindrical roof of Sze, Liu and Lo (2004) to
  the transverse hourglass stiffness of `Quad4R`, in a new section of the
  tutorial `tut-nonlinear-benchmarks.ipynb` and in the new
  `test_hinged_roof_hourglass_sensitivity` of
  `tests/test_nonlinear_benchmarks_sze2004.py`: `hgfactor_w` from `1` (the
  default) to `0.0002`, against `Tria3DSG`, which has no hourglass control,
  and against a shallow-shell Kirchhoff Ritz solution, the new
  `ritz_roof_WC` of the same test module. With the default, the linear
  deflection at the centre is 6 to 7 % smaller than with `Tria3DSG` and,
  with the transverse shear stiffness multiplied by 100, than the Ritz
  solution. The comparison with Tables 9a and 9d of the paper keeps the
  default. The hourglass stiffness is problem-dependent and the default,
  which follows the references documented in `pyfe3d.quad4r`, is unchanged:
  new default values need further studies. Noted in the Notes of
  `pyfe3d.quad4r` and on the page on the choice of the element.
- Documented limitations, replacing developer notes: `Quad4` does not apply
  the correction of Hughes et al. (1977) for extremely thin elements, which
  a comment said was applied; in double precision the deflection of a thin
  plate is unchanged up to an element side of `1.25e5` times the thickness,
  new `tests/test_quad4_thin_limit.py`. The hourglass control of `Quad4R`
  is derived by Brockman (1987) for isotropic plates and applied to
  laminates with equivalent moduli, see the behaviour change above.
  The ply stiffnesses are those of plane stress, and thermal expansion is
  not modelled (`MatLamina.a1`, `a2`, `a3` and `tref` are not used).
  `natural_frequency()` solves the undamped problem; the damped one, e.g.
  with the damping matrix `CA` of the piston theory, is solved by the new
  `complex_frequency()`.
- `Spring` verified in an arbitrary orientation against `[T][K_e][T]^T`,
  with symmetry, rigid-body translation and equal rotations without
  energy, and the internal forces, new tests in `tests/test_spring.py`;
  its Notes state that a rigid rotation of two nodes at different
  positions loads `kye` and `kze`, the nodal coordinates not being used.

## 0.10.0 (2026-09-25)

### New element: `Tria3DSG`, a triangle without a locking parameter

`Tria3DSG`, `Tria3DSGData` and `Tria3DSGProbe` (new, `pyfe3d/tria3dsg.pyx`):
a three-node flat shell element whose transverse shear strains come from the
discrete shear gap method of

> Bletzinger, K.-U., Bischoff, M., and Ramm, E., "A unified approach for
> shear-locking-free triangular and rectangular shell finite elements",
> Computers & Structures, 75(3), 2000, pp. 321-334.
> https://doi.org/10.1016/S0045-7949(99)00140-6

The shear gap `\Delta w` is the function whose gradient is the shear strain.
Its nodal values are line integrals along the element edges from node 1,
`\Delta w_1 = 0` and
`\Delta w_i = (w_i - w_1) - (\phi_1 + \phi_i)\cdot(x_i - x_1)/2`, and the
assumed strain is `\gamma = \sum_i \Delta w_i \nabla N_i`, constant over the
triangle. The locking is removed in the operator rather than in the
constitutive matrix, so the element uses `A_{ts}` exactly as the laminate
gives it and needs no stabilisation parameter, which is what distinguishes it
from `Tria3R`. Verified in `tests/test_tria3dsg.py`:

- the three rigid-body modes involving `w` give `\gamma = 0` exactly;
- a constant transverse shear state is reproduced exactly, to `1e-12` at
  every element size, which `Tria3R` cannot do;
- a constant-curvature Kirchhoff state gives `\gamma = 0` exactly, the
  locking-free property;
- the element rank matches `Tria3R`, and assembly removes the spurious modes.

The gap is measured from one node, so the plain operator depends on which
node carries that label: on an irregular triangle the element matrix moves by
14 and 17 per cent under the two cyclic relabellings. The element therefore
averages the three operators, one per starting node. Each is separately exact
for all the properties above and those are linear in the operator, so the mean
inherits them; invariance is measured at `6e-16`.

The element is geometrically nonlinear, with `update_KC0`, `update_KG`,
`update_KG_given_stress`, `update_KCNL`, `update_probe_finte`, `update_fint`
and `update_M`, and it is registered in `tests/test_tangent_consistency.py`
and `tests/test_nonlinear_static_newton_raphson.py`, where it passes the same
eight checks as the other elements. The tangent is the exact Jacobian of the
internal force vector: the Taylor test error falls by ten for every decade of
the step, over the four steps `1e-1` to `1e-4`, and every entry of
`KC0 + KCNL(u) + KG(u)` matches a central difference of `fint` to better than
`1e-6` in norm. Under refinement its nonlinear plate deflection approaches the
`Quad4R` answer, the gap falling `1.139% -> 0.319% -> 0.180%` for
`nx = 11, 21, 31`, while `Tria3R` stalls, at `2.779% -> 2.398% -> 2.321%`,
its last two meshes agreeing with each other rather than with the reference.

The prestress-frame trap is guarded by
`test_buckling_load_does_not_depend_on_triangle_node_numbering`: the three
cyclic numberings of the same triangle give the same `KC0` to `1e-15` and so
must give the same buckling load, which they do only when the membrane stress
resultants handed to `update_KG_given_stress` are rotated into each element
frame.

`Tria3DSG.alpha_shear_locking` (new, default `0.`) applies the algebraic
stabilisation of Bischoff and Bletzinger (2004) and Castro et al. (2019) on
top of the discrete shear gap field, which is where those papers apply it.
The default disables it and is a strict no-op, bit-identical to the element
without the attribute. It is documented as useful only on a coarse mesh of a
thin plate, where `\alpha = 0.07`, the lower bound of the range recommended by
Castro et al., takes the first natural frequency of a `5x7` mesh from `12.06`
to `7.93` per cent above the analytical value and the buckling load of a thin
plate from `+1.71` to `+0.11` per cent. The buckling error crosses from the
stiff to the soft side between `\alpha = 0.07` and `0.1`, reproducing on a
discrete shear gap field the behaviour Castro et al. report above `0.09`. The
default is nevertheless zero because `\alpha \ell^2/h^2` is unbounded in
`\ell/h`: any non-zero value removes nearly all of the transverse shear
stiffness on a practical shell mesh, reducing the constant shear state to
`1/(1 + factor)` of its energy, which is `0.07` at `\ell/h = 10` and `7e-8` at
`\ell/h = 10^4`, and it adds error monotonically where transverse shear is
real physics.

`Tria3DSG` is now the recommended triangular element, stated in `README.md`
and in `doc/source/fe_overview.rst`. `Tria3R` is kept for continuity with
results obtained before it existed.

### Breaking: physics-based drilling stiffness is now the default

`Quad4`, `Quad4R` and `Tria3R` gain `drilling_model` (new, default `0`).
The default combines the enrichment of the in-plane displacement field of
Allman, so that `r_z` produces membrane strain energy, with the regularised
functional of Hughes and Brezzi tying `r_z` to the rotation of the membrane
field, in the combination of Ibrahimbegovic, Taylor and Wilson for the
quadrilateral:

> Allman, D. J., "A compatible triangular element including vertex rotations
> for plane elasticity analysis", Computers & Structures, 19(1-2), 1984,
> pp. 1-8. https://doi.org/10.1016/0045-7949(84)90197-4
>
> Hughes, T. J. R., and Brezzi, F., "On drilling degrees of freedom",
> Computer Methods in Applied Mechanics and Engineering, 72(1), 1989,
> pp. 105-121. https://doi.org/10.1016/0045-7825(89)90124-2
>
> Ibrahimbegovic, A., Taylor, R. L., and Wilson, E. L., "A robust
> quadrilateral membrane finite element with drilling degrees of freedom",
> International Journal for Numerical Methods in Engineering, 30(3), 1990,
> pp. 445-457. https://doi.org/10.1002/nme.1620300305

The drilling rotation becomes a kinematic variable that carries strain
energy, the recovered nodal moments about the shell normal are physical, and
no user parameter is involved. Any other value of `drilling_model` selects
the fictitious penalty of MSC Nastran and Autodesk Nastran controlled by
`K6ROT`, which was the default up to 0.9.0. Setting
`elem.drilling_model = 1` before `update_KC0` reproduces results obtained
before 0.10.0.

- `gamma_rz` (new): regularisation coefficient of the physics-based model.
  Negative, the default, means use `A_{66}` of the laminate extensional
  stiffness, the value identified by Hughes and Brezzi. It is a modulus and
  not a parameter that needs tuning; the attribute is exposed for the
  sensitivity study the literature recommends. The penalty of MSC and
  Autodesk Nastran corresponds to `gamma_rz = K6ROT*1e-6*A66`.
- `K6ROT` is only read when `drilling_model` is not `0`.

Recorded reference values move on **non-coplanar models only**. On a flat
mesh the drilling row is decoupled and its magnitude cannot change any
displacement. `tests/test_quad4_natural_freq_cylinder.py` moves from
`[1932.79, 2105.30, 2114.87]` to `[2176.66, 2303.52, 2396.45]`, and
`tests/test_quad4r_natural_freq_cylinder.py` from
`[1981.51, 2117.62, 2203.24]` to `[2229.65, 2377.24, 2432.59]`. The physics-based
stiffness is stiffer than the penalty on those coarse meshes, by about 11 per
cent, because the constraint `r_z = \theta_z` does real work on a curved
shell, where the drilling rotation of one element projects onto the bending
rotations of its neighbours. The two models converge towards each other under
refinement, the gap on the first mode falling `11.2% -> 6.0% -> 3.1%` for
refinement `1 -> 2 -> 4`.

### Fixed: the `K6ROT` penalty coefficient had been lost

The fictitious drilling stiffness was assembled with an effective coefficient
of `1` instead of `K6ROT*1e-6*A66`, so `K6ROT` had no effect on the assembled
stiffness. Restoring it moves the frequencies of
`tests/test_quad4_natural_freq_cylinder.py` from
`[1932.79, 2105.30, 2114.87]` to `[1958.18, 2115.51, 2154.20]`, independently
of the change of default above. It went unnoticed because only non-coplanar
models are affected, for the reason given above.

### Quad4 membrane quadrature

The membrane stiffness of `Quad4`, which Hughes et al. (1977) do not specify,
is integrated with a three-by-three quadrature when the default drilling model
is active, and with two-by-two otherwise. The Allman edge modes make the
membrane strain-displacement operator vary linearly, and with two points there
is a combination of drilling rotations that produces no membrane strain at any
of the four points, leaving a zero-energy mode that no value of `gamma_rz`
can remove. The bending term always uses two-by-two, so that the drilling
model does not change the bending response of the element.

### New: `pyfe3d.solver`

`pyfe3d/solver.py` (new module, exposed as `pyfe3d.solver` and documented in
`doc/source/solver.rst`) collects the preparation that the two eigenvalue
problems of this package need and that is easy to get wrong silently:

- `diagonal_preconditioner(K, floor)`: Jacobi equilibration of the diagonal,
  which does not change the eigenvalues and usually decides whether the
  iterative solver converges. A shell element mixes translations and
  rotations, and with the `K6ROT` penalty the drilling entries sit some six
  orders of magnitude below the membrane ones, so condition numbers of
  `10^{14}` are easy to reach.
- `estimate_cayley_sigma(K, KG, ...)`: a safe shift for the Cayley mode of
  `scipy.sparse.linalg.eigsh`. With a shift smaller than `|\mu|` of the
  critical eigenvalues, the eigenvalues closest to `-\sigma` come back
  instead, with no warning.
- `linear_buckling(K, KG, ...)`: load multipliers with a verified spectrum.
- `natural_frequency(K, M, ...)`: circular frequencies, already sorted.
- `check_eigenpairs(K, KG, eigvals, eigvecs, ...)` and
  `is_positive_definite(A)`.

The tests that previously inlined the preconditioning and the `eigsh` call now
use this module, which is the bulk of the diff in the test suite. Covered by
`tests/test_solver.py`.

### Performance: Cython directives and compiler flags

- Every module is compiled with `initializedcheck=False`. Cython no longer
  checks, at each memoryview access inside the element methods, that the
  memoryview has been assigned. Reading an unassigned memoryview attribute
  from Python still raises `AttributeError`, because the property getters
  keep their own check. An unassigned memoryview reached inside an element
  method would now crash instead of raising, which the probes rule out by
  allocating their arrays in `__cinit__`.
- `setup.py` passed `linetrace=True` to every build. Without `CYTHON_TRACE`
  the line tracing itself stays compiled out, but the directive also
  generates profiling hooks that are active by default and run at every call
  of a Python-visible method. The directive is now set only for a coverage
  build, that is when `CYTHON_TRACE_NOGIL` is in the environment.
- Compiler flags: `-O3` and `-fno-math-errno` for GCC and Clang, and an
  explicit `/O2` for MSVC, which setuptools already combines with `/GL`. The
  OpenMP flags are removed, since no module uses `prange`. Flags that change
  floating-point results, such as `/fp:fast` or `-ffast-math`, and flags that
  tie a wheel to the CPU that built it, such as `-march=native`, are
  deliberately not used.

Measured on an AMD x86-64 laptop CPU (family 25, model 117), Windows 11,
Python 3.13, NumPy 2.4, Cython 3.2, MSVC, with the process pinned to one core
at high priority, minimum of 3 interleaved runs of 5 repetitions each. The
GCC and Clang flags were not measured. Cost per element of the element methods
called from Python (`benchmarks/bench_build_flags.py`), in microseconds, for a
flat 40x40 mesh (1600 quadrilaterals or 3200 triangles) and an 8-ply CFRP
laminate, including the Python overhead of the element loop. `fint` is
`update_fint(..., nonlinear=1)`, and each cell gives the cost before and after
this change, with the relative difference:

| Element | KC0 | M | KG | fint | KCNL |
| - | - | - | - | - | - |
| Quad4 | 96.09 / 73.76 (-23%) | 5.08 / 1.81 (-64%) | 1.87 / 1.70 (-9%) | 57.35 / 54.84 (-4%) | 47.41 / 27.34 (-42%) |
| Quad4R | 3.09 / 2.98 (-4%) | 5.08 / 1.81 (-64%) | 1.93 / 1.73 (-10%) | 2.53 / 2.38 (-6%) | 47.77 / 26.31 (-45%) |
| Tria3R | 5.31 / 5.12 (-4%) | 1.46 / 1.42 (-2%) | 1.32 / 1.26 (-5%) | 4.09 / 3.99 (-2%) | 24.47 / 12.97 (-47%) |
| Tria3DSG | 24.13 / 13.02 (-46%) | 10.07 / 10.06 (-0%) | 1.59 / 1.50 (-6%) | 2.65 / 2.37 (-10%) | 25.11 / 13.07 (-48%) |

The gain is largest where a method loops over memoryviews: `update_KCNL` of
every element, and `update_KC0` of `Tria3DSG` and `Quad4`. The consistent mass
matrix of `Quad4` and `Quad4R` is almost three times faster. Methods whose
cost is dominated by generated straight-line expressions change by 2 to 10%.
Results are unchanged: the whole test suite passes with the new build.

### Documentation

- `pyfe3d.tria3r`: the `alpha_shear_locking` documentation now states that no
  single value serves every problem class and that it has to be verified per
  problem. On the plate of `tests/test_tria3r_natural_freq.py`, with the
  consistent mass matrix, the error in the first natural frequency is
  `+0.6%` at `\alpha = 0.7`, `+30.6%` at `0.1`, `+77.3%` at `0.01` and
  `+96.1%` at `0`, the last being the locking itself; `Tria3DSG` on the same
  mesh, with no parameter, gives `+2.8%`. On the cylinder of
  `tests/test_quad4_linear_buckling_cylinder_displ.py` the ordering reverses:
  at `\alpha = 0.7` the critical eigenvalue belongs to a mode at two elements
  per wavelength, the mesh Nyquist limit, carrying 36.6 per cent of its energy
  in transverse shear, which is a numerical mechanism and not a physical mode,
  while at `0.1` the physical mode is critical and agrees to 3 per cent with
  `Quad4` and `Tria3DSG`. The previously quoted figures mixed two mass matrix
  types and are replaced by a consistent set.
- `pyfe3d.tria3r` now points at `pyfe3d.tria3dsg` as the triangle to reach
  for by default.
- A literal `0x07` byte in `pyfe3d/tria3r.pyx`, where `\alpha` was meant, is
  fixed.
- `doc/source/fe_overview.rst` gains a paragraph on which element to use, and
  lists `tria3dsg` before `tria3r`.
- `Quad4.update_KG_given_stress` and its counterparts document that the stress
  resultants are expected in the **element** coordinate system, with the trap
  this creates on a diagonally split quadrilateral mesh, where the second
  triangle of each cell has its `x` axis on the cell diagonal.

### Tests

New files, all citing their source:

- `tests/test_tria3dsg.py`: the new element, 13 tests.
- `tests/test_drilling_patch_tests.py`, `tests/test_drilling_rank.py`,
  `tests/test_drilling_benchmarks.py`: consistency and patch tests, rank and
  conditioning, and the literature benchmarks of the physics-based drilling
  stiffness, including Cook's skew membrane against the table of
  Ibrahimbegovic et al.
- `tests/test_drilling_k6rot_penalty.py`: regression lock for the penalty
  model, which remains available through `drilling_model = 1`.
- `tests/test_transverse_shear_stiffness.py`: the transverse shear stiffness
  as a property of the laminate, and how each element uses it, including the
  measurement that `alpha_shear_locking` is doing the unlocking in `Tria3R`
  rather than fine tuning.
- `tests/test_quad4_spurious_shear_mode.py`: the two zero-energy modes of the
  selectively underintegrated element, recorded as a documented property.
- `tests/test_quad4r_hourglass_control.py`: verification of the hourglass
  control of `Quad4R`.
- `tests/test_solver.py`: the new solver module, including the fallback
  paths of the shift estimate and of the positive definiteness test.

Extended:

- `tests/test_tangent_consistency.py`: `fint == KC0 @ u` and the finite
  difference Jacobian are also checked for an explicit `gamma_rz` and for the
  `K6ROT` penalty on all four shells, since the drilling stiffness is computed
  in both `update_KC0` and `update_probe_finte`. `gamma_rz = A66` must
  reproduce the default exactly and any other value must change the result.
- `tests/test_tria3dsg.py`: a clockwise node ordering gives the same element,
  and the lumped mass matrix puts a third of the mass on each node.

Line coverage is 99.99%. The only lines not covered are the platform
dependent choice of `INT` and three negative-area branches of `Tria3DSG`,
which the element frame makes unreachable and which are marked
`# pragma: no cover`.

### Packaging

- `pyfe3d.Tria3DSG`, `Tria3DSGData` and `Tria3DSGProbe` are exported from
  `pyfe3d`, and `pyfe3d.solver` is imported with the package.
- `setup.py` builds the new `pyfe3d.tria3dsg` extension, and `.gitignore`
  covers its generated `tria3dsg.cpp`.
- `doc/source/tria3dsg.rst` and `doc/source/solver.rst` are new, and
  `doc/source/index.rst` gains a "Solvers and preconditioners" section.
- `Quad4`, `Quad4R` and `Tria3R` have new C attributes (`drilling_model`,
  `gamma_rz`), so packages that `cimport pyfe3d` must be recompiled against
  this version.
- Coverage build (`CYTHON_TRACE_NOGIL` in the environment, or
  `--define CYTHON_TRACE_NOGIL` on the command line): legacy tracing is
  requested with `CYTHON_USE_SYS_MONITORING=0`, since from Python 3.12 Cython
  traces through `sys.monitoring`, which the `Cython.Coverage` plugin does not
  follow, and the report then silently lists no `.pyx` file. `/Od` replaces
  `/O0`, which MSVC ignores. Switching between a coverage build and a normal
  one now regenerates the C++ files, which cythonize would otherwise reuse
  from the other mode.

## 0.9.0 (2026-09-17)

### Breaking: transverse shear stiffness now includes the shear correction

`ShellProp.E44`, `E45` and `E55` were the uncorrected constant-strain stiffness
`sum_k Cs_k h_k`, and the elements multiplied them by the shear correction
factors `scf_k13`, `scf_k23`. They are replaced by `A44`, `A45` and `A55`,
which come **with the correction already applied**, computed by default with
the equilibrium approach of

> Rohwer, K. "Improved transverse shear stiffness for layered finite
> elements", DFVLR-FB 88-32, 1988.

which returns the full 2x2 matrix directly, including the `A45` coupling of
angle-ply laminates. It is valid for unsymmetric laminates and does not depend
on `offset`. For a homogeneous plate it gives exactly `5/6 G h`. This follows
the same change in [composites](https://github.com/saullocastro/composites/pull/15).
No backward compatibility is kept.

- `A44`, `A45`, `A55` (replace `E44`, `E45`, `E55`): corrected stiffness, used
  directly by the elements.
- `ShellProp.Ats` (replaces `ShellProp.E`): `[[A44, A45], [A45, A55]]`, index
  4 <-> yz, 5 <-> xz.
- `ShellProp.ABDE` and `get_ABDE` were removed; use `ABD` and `Ats`.
- `Abar44`, `Abar45`, `Abar55` and `ShellProp.Abar_ts` (new): the
  constant-strain, uncorrected stiffness (what `E44`, `E45`, `E55` used to be).
- `Abarbar44`, `Abarbar45`, `Abarbar55` and `ShellProp.Abarbar_ts` (new): the
  constant-stress stiffness `h^2 [sum_k inv(Cs_k) h_k]^-1`, for comparison.
- `ShellProp.shear_correction` and the `shear_correction` argument of
  `laminated_plate` and `isotropic_plate` (new): `'rohwer'` (default),
  `'vlachoutsis'`, `'constant'` (5/6) or `None` (no correction). They replace
  the `calc_scf` argument, which was removed.
- `ShellProp.calc_scf()` was removed. `ShellProp.calc_transverse_shear_stiffness()`
  (new) is called by `calc_constitutive_matrix()`. It raises `ValueError` if a
  ply has `g13 = 0` or `g23 = 0` (singular `Cs`), or if the ABD matrix is
  singular. Use `shear_correction=None` for plies without transverse shear
  properties.
- `scf_k13`, `scf_k23`: now the **reported ratios** `A55/Abar55` and
  `A44/Abar44`, informative only. The elements no longer read them.
- `ShellProp.calc_transverse_shear_stress(z, Qy, Qx)` (new): recovers
  `(tau_yz, tau_xz)` through the thickness from the same equilibrium
  distribution, zero at the free surfaces and continuous across plies, for use
  in failure criteria.
- Shell properties created from lamination parameters
  (`shellprop_from_lamination_parameters`,
  `shellprop_from_LaminationParameters`) have no ply distribution. They keep
  `A44 = Abar44`, etc., with `shear_correction = None` and
  `scf_k13 = scf_k23 = 1`.
- Lamination parameters and gradients: `xiE1`, `xiE2` renamed to `xiAts1`,
  `xiAts2`; `GradABDE` renamed to `GradABD`; `gradEij` renamed to
  `gradAtsij`, which contains the gradients of `Abar44`, `Abar45`, `Abar55`.
- `ShellProp` has new C attributes, so packages that `cimport pyfe3d` must be
  recompiled against this version.

### Transverse shear stiffness in the element coordinate system

The stiffness of Rohwer (1988) is not invariant to a rotation of the reference
frame, because its two cylindrical bending states are tied to the `x` and `y`
axes. When a material direction is defined (`xmati`, `xmatj`, `xmatk` in
`update_rotation_matrix`), `Quad4`, `Quad4R` and `Tria3R` now use the stiffness
evaluated in the element coordinate system, i.e. for the plies rotated by the
angle between the material and the element `x` axes, such that the assumed
static state and the element kinematics refer to the same pair of directions.
With `'constant'`, `'vlachoutsis'`, `None`, or without plies, the stiffness is
rotated as a second-order tensor.

Re-evaluating the method of Rohwer for each element costs about 330 ns per ply
(see the cost comparison below). Instead, the compliance `S = inv(Ats)` in the
element frame is represented exactly by a Fourier series in `2 theta` with 5
harmonics: the equilibrium distribution `f(z)` is a polynomial of degree 4 in
`cos(theta)`, `sin(theta)` and `inv(Cs)` of degree 2, such that `S` is a
trigonometric polynomial of degree 10 with a period of 180 degrees. The 33
coefficients are computed once per laminate, from 16 evaluations in rotated
frames, and the cost per element is independent of the number of plies. The
harmonics above the 5th were verified to be at machine precision.

- `ShellProp.calc_Ats_element(thetadeg)` (new): `Ats` in an element frame
  whose material direction is at `thetadeg` from the element `x` axis.
- `ShellProp.calc_constitutive_element(thetadeg)` (new): `A`, `B`, `D` and
  `Ats` in that element frame.
- Changes made to `A44`, `A45`, `A55` after `calc_constitutive_matrix()` are
  only used by elements without a material direction when the plies are
  available and `shear_correction='rohwer'`.

### Single function for the constitutive matrices of all shell elements

`ShellProp.get_constitutive_element(m11, m12, m21, m22, A, B, D, Ats)` (new,
`cdef nogil`) is now the only place where the constitutive matrices are brought
to the element coordinate system. It is used by `update_KC0`,
`update_probe_finte`/`update_fint`, `update_KG`, `update_KCNL` and the
nonlinear internal forces of `Quad4`, `Quad4R` and `Tria3R`. It replaces:

- the generated closed-form rotation of `A`, `B`, `D`, repeated in `KC0`,
  `finte` and `KG` of each element;
- the private `_update_AB_element`, used by `KCNL` and the nonlinear internal
  forces, which rotated `A` and `B` with a separate loop-based implementation.

`_update_AB_element` did not handle the transverse shear, but this did not
produce wrong results: the von Karman strains only add nonlinear terms to the
membrane strains, whereas `gamma_yz` and `gamma_xz` remain linear, such that
the transverse shear stiffness only enters `KC0` and the linear internal
forces. The two implementations were consistent (the tangent consistency tests
with a material direction passed before and after), but duplicated, which is
the kind of divergence that led to past bugs.

### Fixed

- The transverse shear stiffness was not rotated from the material to the
  element coordinate system: the elements used `E44*scf_k23`,
  `E45*(scf_k13 + scf_k23)/2` and `E55*scf_k13` in the material system while
  `A`, `B` and `D` were rotated.
- The previous `ShellProp.calc_scf`, which claimed to implement Vlachoutsis
  (1992), returned wrong factors, often above 1, which a shear correction
  factor can never be. Specifically, it:
  1. accumulated the ply bending stiffness over the plies, so the result
     depended on the stacking order: for CFRP plies (`E1 = 138`, `E2 = 9.3`,
     `G12 = G13 = 4.6`, `G23 = 2.3` GPa) `[0, 90]s` gave
     `(k13, k23) = (1.2343, 0.4278)` and `[90, 0]s` gave `(0.4278, 1.2343)`,
     whereas `'rohwer'` now gives `(0.6705, 0.7320)` and `(0.7320, 0.6705)`;
  2. cancelled the transverse shear moduli, so the factors did not depend on
     `g13`, `g23`: a sandwich with 0.1-1.6-0.1 thicknesses, isotropic faces
     100 times stiffer than the core, and a core with `g13 = g23 = g12/100`,
     gave `1.0892` instead of `0.0009`;
  3. rotated `E1`, `E2` with a non-tensorial formula, wrong except at 0 and 90
     degrees;
  4. used `offset` instead of the neutral surface of each direction.

  The `'vlachoutsis'` mode is a corrected implementation. It is exact for
  specially orthotropic plies, and its `A45 = (k13 + k23)/2 Abar45` is ad hoc.
- `ShellProp` can be pickled and deep-copied.

### Computational cost

Measured on an AMD x86-64 laptop CPU (family 25, model 117), Windows 11, Python
3.13, NumPy 2.4, Cython 3.2, MSVC, with the process pinned to one core at high
priority, minimum of 2 interleaved runs of 5 repetitions each. Three versions
are compared:

- **0.8.0**: before this change.
- **Direct**: Rohwer's method re-evaluated for each element with a material
  direction (intermediate implementation, never released).
- **0.9.0**: this change, with the Fourier evaluation.

Cost of the transverse shear stiffness per element call, in a compiled loop
without Python overhead (`benchmarks/bench_constitutive_element.py`), in ns,
for a material direction at 30 degrees:

| Plies | 0.8.0 | Direct | 0.9.0, `Ats` only | 0.9.0, `A`, `B`, `D` and `Ats` |
| - | - | - | - | - |
| 1 | < 1 | 267 | 23 | 48 |
| 8 | < 1 | 1029 | 27 | 49 |
| 32 | < 1 | 2840 | 25 | 51 |
| 128 | < 1 | 10564 | 23 | 51 |

Without a material direction, `get_constitutive_element` costs 10 to 17 ns,
mostly copying the 27 stiffness terms. In 0.8.0 the rotation of `A`, `B`, `D`
was inlined in each element method and is not included in the 0.8.0 column,
whose transverse shear cost was only three multiplications.

Cost per element of the element methods called from Python
(`benchmarks/bench_transverse_shear_cost.py`), in microseconds, for a flat
40x40 mesh (1600 quadrilaterals or 3200 triangles), including the Python
overhead of the element loop. The relative difference to 0.8.0 is in
parentheses; differences within about 3% are measurement noise:

| Element | Material direction | Plies | KC0 0.8.0 | KC0 Direct | KC0 0.9.0 | fint 0.8.0 | fint Direct | fint 0.9.0 | KG 0.8.0 | KG 0.9.0 |
| - | - | - | - | - | - | - | - | - | - | - |
| Quad4 | no | 1 | 75.77 | 75.36 (-1%) | 75.39 (-1%) | 35.76 | 35.67 (-0%) | 35.81 (+0%) | 1.70 | 1.75 (+3%) |
| Quad4 | no | 8 | 75.58 | 75.46 (-0%) | 76.37 (+1%) | 35.67 | 35.88 (+1%) | 35.93 (+1%) | 1.72 | 1.73 (+0%) |
| Quad4 | no | 32 | 75.03 | 75.45 (+1%) | 75.69 (+1%) | 35.19 | 35.72 (+1%) | 35.99 (+2%) | 1.71 | 1.73 (+1%) |
| Quad4 | yes | 1 | 75.63 | 75.67 (+0%) | 76.44 (+1%) | 35.15 | 36.08 (+3%) | 36.03 (+3%) | 1.78 | 2.10 (+18%) |
| Quad4 | yes | 8 | 76.60 | 77.20 (+1%) | 76.32 (-0%) | 35.58 | 36.68 (+3%) | 36.17 (+2%) | 1.79 | 1.72 (-4%) |
| Quad4 | yes | 32 | 75.20 | 79.25 (+5%) | 76.37 (+2%) | 35.38 | 38.83 (+10%) | 35.87 (+1%) | 1.81 | 1.87 (+3%) |
| Quad4R | no | 1 | 3.03 | 3.05 (+1%) | 2.99 (-1%) | 1.95 | 1.96 (+0%) | 2.01 (+3%) | 1.78 | 1.86 (+5%) |
| Quad4R | no | 8 | 3.00 | 2.97 (-1%) | 3.02 (+1%) | 1.97 | 1.97 (+0%) | 2.04 (+3%) | 1.77 | 1.92 (+8%) |
| Quad4R | no | 32 | 3.02 | 2.98 (-1%) | 3.00 (-1%) | 1.97 | 1.97 (+0%) | 2.03 (+3%) | 1.76 | 1.86 (+5%) |
| Quad4R | yes | 1 | 3.05 | 3.30 (+8%) | 3.07 (+1%) | 2.06 | 2.34 (+14%) | 2.07 (+1%) | 1.84 | 1.88 (+2%) |
| Quad4R | yes | 8 | 3.10 | 3.92 (+27%) | 3.04 (-2%) | 2.04 | 2.91 (+43%) | 2.08 (+2%) | 1.87 | 1.88 (+1%) |
| Quad4R | yes | 32 | 3.10 | 5.87 (+90%) | 3.11 (+0%) | 2.03 | 4.85 (+139%) | 2.08 (+3%) | 1.86 | 1.91 (+3%) |
| Tria3R | no | 1 | 2.03 | 2.02 (-0%) | 2.01 (-1%) | 1.40 | 1.40 (+0%) | 1.41 (+1%) | 1.24 | 1.27 (+2%) |
| Tria3R | no | 8 | 2.01 | 2.04 (+2%) | 2.04 (+1%) | 1.39 | 1.40 (+1%) | 1.41 (+2%) | 1.22 | 1.27 (+4%) |
| Tria3R | no | 32 | 2.02 | 2.00 (-1%) | 2.01 (-1%) | 1.39 | 1.41 (+1%) | 1.42 (+1%) | 1.23 | 1.23 (+0%) |
| Tria3R | yes | 1 | 2.11 | 2.37 (+12%) | 2.07 (-2%) | 1.44 | 1.77 (+22%) | 1.45 (+0%) | 1.28 | 1.26 (-2%) |
| Tria3R | yes | 8 | 2.07 | 2.99 (+44%) | 2.05 (-1%) | 1.45 | 2.37 (+64%) | 1.46 (+1%) | 1.30 | 1.27 (-2%) |
| Tria3R | yes | 32 | 2.10 | 4.94 (+136%) | 2.06 (-2%) | 1.45 | 4.32 (+197%) | 1.45 (+0%) | 1.30 | 1.27 (-2%) |

Conclusions:

- Re-evaluating Rohwer's method per element doubled the cost of `KC0` and
  nearly tripled the cost of the internal forces of `Quad4R` and `Tria3R` for
  a 32-ply laminate with a material direction, growing linearly with the
  number of plies. For `Quad4`, whose own cost is about 25 times larger, the
  increase was up to 10%.
- With the Fourier evaluation, the cost of 0.9.0 is the same as 0.8.0 within
  the measurement noise, for all elements, number of plies and with or
  without a material direction.
- Creating a laminate with `laminated_plate` costs about the same as before
  (0.02 ms for 1 ply to 0.95 ms for 128 plies), since the 16 evaluations for
  the Fourier coefficients replace the former `calc_scf`.

### Results

- Laminates with transverse shear correction factors different from the old
  `calc_scf` results give different results. For the thin CFRP cylinders of
  `tests/test_quad4_natural_freq_cylinder.py` and
  `tests/test_quad4r_natural_freq_cylinder.py` the natural frequencies changed
  by about 1e-4, and the reference values were updated.
- Isotropic plates give the same results as before, since both the old
  `calc_scf` and the new default give 5/6.

### Migration

- Replace `prop.E44`, `prop.E45`, `prop.E55` with `prop.A44`, `prop.A45`,
  `prop.A55`, and `prop.E` with `prop.Ats`. When setting them directly, e.g.
  for smeared or optimised properties, **include the shear correction**, since
  the elements no longer apply `scf_k13`, `scf_k23`: `prop.A44 = 5/6*G*h`.
- Replace `calc_scf=True` with the default and `calc_scf=False` with
  `shear_correction=None`. To force `5/6`, use `shear_correction='constant'`
  instead of setting `scf_k13`, `scf_k23`.
- To reproduce the previous uncorrected values of `E44`, `E45`, `E55`, read
  `Abar44`, `Abar45`, `Abar55` (or `prop.Abar_ts`).
- Replace `xiE1`, `xiE2` with `xiAts1`, `xiAts2`, `GradABDE` with `GradABD`
  and `gradEij` with `gradAtsij`.
- Through-thickness transverse shear stresses: use
  `prop.calc_transverse_shear_stress(z, Qy, Qx)` rather than `Cs @ gamma`,
  which is constant within each ply and non-zero at the free surfaces.


## 0.8.0 (2026-09-14)

- KCNL matrix for all elements, with more accurate nonlinear analyses.
- Fixed KC0 and KG for beams.
- Fixed KC0 for Tria3R.

## 0.7.0 (2026-07-14)

- Drilling stiffness re-coupled with the in-plane rotation using K6ROT.
- Fixed drilling stiffness integration.
- Restored missing `pid` attribute.

## 0.6.4 (2026-07-02)

- Internal force vectors (finte) for all elements.
- Fixed drilling stiffness (K6ROT) scaling.
- Added Python 3.14 and removed Python 3.8 support; Linux wheels limited to
  64-bit.
- Exposed `update_probe_finte` to Python.
- New documentation examples.

## 0.5.2 (2024-10-14)

- New Quad4 element.
- K6ROT drilling stiffness.
- `probe.update_BL`.
- Removed the alg3dpy dependency.
- Topology optimization tutorial.

## 0.4.28 (2024-07-22)

- Piston theory aerodynamics for Quad4R.
- Fixed array casting on some systems.
- Quad4R less sensitive to small element sizes.
- New `pid` attribute for elements.
- Exposed element orientation vectors.
- Python 3.12 support.
- Fixed build for Linux and macOS wheels.

## 0.3.24 (2023-03-26)

- New Tria3R element.
- Fixed BeamC formulation.
- Fixed beam torsion stiffness.
- Fixed lamination parameters and added their gradients.
- Fixed lumped mass for Tria3R.
- Removed dependency on the NumPy C API.
- OpenMP build fixes.
- Improved Zenodo metadata and documentation layout.
- Python 3.10 support.

## 0.2.8 (2021-10-15)

- Material coordinate systems for plate elements.
- Fixed Truss mass matrix.
- More aggressive Cython directives.
- License changed to 3-clause BSD; license info in README, author metadata
  updates.

## 0.1.16 (2021-08-25)

- New Spring element.
- Rotation matrices using direct `rij` terms instead of Euler angles.
- **Breaking:** renamed `update_xe` to `update_probe_xe`.
- `probe` made a public attribute.
- KG with a given stress state.
- **Breaking:** renamed `update_ue` to `update_probe_ue`.

## 0.1.3 (2021-07-29)

- First release: Quad4R, BeamC, BeamLR and Truss elements; shell properties;
  linear buckling.
