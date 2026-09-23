# Changelog

## 0.10.0 (2026-09-23)

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
- `tests/test_solver.py`: the new solver module.

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
