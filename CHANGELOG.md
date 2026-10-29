# Changelog

## 0.12.0 (2026-10-29)

### New features

- New module `pyfe3d.vabs`, a variational asymptotic cross-sectional
  analysis (VABS) of prismatic beams, after Cesnik and Hodges (1997), Yu et
  al. (2002) and Yu et al. (2012). `CrossSection` takes an arbitrary mesh of
  the cross section, with 3- and 6-node triangles and 4-, 8- and 9-node
  quadrilaterals, the 6x6 material matrices, created e.g. by `isotropic` or
  `orthotropic`, and the angles `theta1` and `theta3` that orient each
  element, see `rotate_material`; `calc_stiffness` gives the classical 4x4
  and the generalized Timoshenko 6x6 stiffness and flexibility matrices,
  the tension and shear centres, with the warping constraints of Yu et al.
  (2012) or of Yu et al. (2002), and `calc_mass` the mass matrix, the mass
  centre and the mass integrals of `BeamProp`.
- The matrices are in the convention of Yu, `{gamma11, 2gamma12, 2gamma13,
  kappa1, kappa2, kappa3}`; `yu_to_pyfe3d` and `pyfe3d_to_yu` convert them
  to and from that of the beam elements, `{N, Vy, Vz, Mx, Mz, My}`, with
  `kappa2 = kappa_z` and `kappa3 = -kappa_y`, `classical_from_timoshenko`
  condenses the shear strains, and `beamprop_from_vabs` creates a
  `BeamProp` when the matrix has the pattern that the elements can
  represent. The beam elements are unchanged.
- `CrossSection.warping` gives the warping displacements of the nodes for
  given beam strains, in the convention of Yu or of pyfe3d, and
  `CrossSection.triangles` a triangulation of the mesh through all its
  nodes, to plot the warping with any plotting library; pyfe3d does not
  depend on one.
- `CrossSection.calc_vlasov` gives the 5x5 stiffness `Kv` of the
  generalized Vlasov model of Yu et al. (2005), for thin-walled open
  sections, of `{gamma11, kappa1, kappa2, kappa3, kappa1'}` about the shear
  centre, with the warping rigidity; `CrossSection.translated` moves the
  origin of a cross section, e.g. to the shear centre. No beam element
  uses `Kv`. The default warping constraints of `calc_vlasov` are
  `'yu2002'`, which reproduce Yu et al. (2005); `'yu2012'` gives the same
  classical block and a warping rigidity lower by up to 12 per cent for
  strips and within about 0.1 per cent for I- and channel sections.
- Initially twisted and curved beams are not implemented: non-zero `k1`,
  `k2`, `k3` raise `NotImplementedError`.
- `linear_buckling()` and `natural_frequency()` complete the eigenvalues
  that the inertia check finds missing below the highest one returned by
  the sparse solver, e.g. copies of the repeated eigenvalues of a cylinder
  that the Lanczos process missed, with deflated passes before any other
  solver (new `pyfe3d.solver.arpackutils.deflated_eigsh`, at most
  `MAX_COMPLETION_PASSES = 3`); `linear_buckling()` also completes the
  result of its second strategy, `eigsh()` with a 10 times larger shift,
  before the dense fallback. The torsion of the cylinder Z33 with 80
  elements around the circumference raised with 2 to 4 load multipliers
  requested; it now returns them without the dense fallback.
- A symmetric stiffness matrix that is not positive definite raises a
  `numpy.linalg.LinAlgError` that says so, with the likely causes
  (insufficient boundary conditions, a mechanism, re-entrant elements, a
  preload beyond buckling), in `linear_buckling()` (with `check=True` or
  `sparse_solver=False`), `linear_static()` and the linear
  `Analysis.static()`, which previously returned results. The
  `RuntimeError` of `natural_frequency()` starts with the same diagnosis
  when its solvers fail; an indefinite or singular `K` remains accepted
  there.
- `MatLamina.rebuild(plane_stress=False)` keeps the three-dimensional
  constants and invariants (Tsai, 2008, Sec. 3.3), e.g. for continuum
  shells; the choice is stored in the new read-only attribute
  `MatLamina.plane_stress`, `True` by default, see the behaviour change
  below. `trace_normalize_plane_stress()` raises `ValueError` for
  `plane_stress=False`, since the trace of Melo et al. (2017), Eq. (1), is
  that of the plane-stress stiffness matrix.
- Zero Young moduli, e.g. `laminaprop = (0, 0, nu12, 0, G13, G23)` for the
  antiplane core of a sandwich (Allen, 1969, p. 14), are accepted by
  `read_laminaprop()` and `MatLamina.rebuild()`, where `nu21 =
  nu12*e2/e1` raised `ZeroDivisionError`. With `ei = 0` the reciprocal
  Poisson's ratio is stored as zero; a non-zero `nuij` with `ei = 0` and
  `ej != 0`, which violates `|nuij| < (ei/ej)**0.5` (Jones, 1999, Eq.
  (2.51)), raises `ValueError`. Such a core adds only transverse shear
  stiffness. Results with non-zero moduli are unchanged, bit for bit.
- Input checks:
  - `MatLamina.rebuild()`, and therefore `read_laminaprop()` and
    `laminated_plate()`, raise `ValueError` for constants that are not
    positive definite (Jones, 1999, Sec. 2.4.2): negative moduli, Eq.
    (2.45), whose zero limits are kept for antiplane cores;
    `|nu_ij| >= (e_i/e_j)**0.5`, Eq. (2.51); and a non-positive
    determinant, Eq. (2.48). The message names the violated conditions
    and their values. With 2 or 6 entries only the in-plane conditions
    apply. Results with admissible constants are unchanged, bit for bit.
  - `ShellProp.calc_equivalent_properties()`, called by
    `laminated_plate()`, raises `numpy.linalg.LinAlgError`, a
    `ValueError`, for an `ABD` matrix that is singular or ill-conditioned,
    also with a positive diagonal, e.g. a single ply at 45 degrees with
    `g12 = 0`, or a laminate of antiplane cores only. The check is the
    condition number in the 1-norm of the `ABD` matrix about the
    mid-surface, scaled by its diagonal, against `1e12`; about the
    reference surface it would grow as `(offset/h)**2` and reject valid
    laminates with a large offset. A non-finite inverse, as returned by
    the LAPACK of Pyodide for a singular matrix, is also caught.

### Behaviour change: physical hourglass stabilisation of `Quad4R`

- `Quad4R` stabilises the bending and the transverse shear physically,
  after Belytschko and Leviathan (1994), CMAME 113: the assumed transverse
  shear strains of Dvorkin and Bathe, their constant part at the centroid,
  Eq. (4.12), and a stabilisation of their nonconstant part, Eq. (4.13);
  and the nonconstant curvatures, Eq. (3.28), integrated with the bending
  stiffness of the laminate, `D - B A^-1 B`, Eqs. (4.13) and (B.18). This
  replaces the artificial hourglass stiffnesses of Brockman for `w`, `r_x`
  and `r_y`; that of `w` stiffened a uniform twist by up to 78 per cent
  of its energy whatever the mesh, and depended on the length unit.
  `Quad4R` now passes the constant-twist and the thin-plate bending patch
  tests, and a single element is singular only in its six rigid-body
  modes. The membrane keeps the hourglass stiffness of Brockman: the
  physical form of the paper for `u`, `v` is the 2x2-integrated
  quadrilateral, which was stiffer on the in-plane strips of MacNeal and
  Harder and on Cook's membrane. The stabilisation is linear and enters
  `KC0` and the internal forces; `KCNL` and `KG` are unchanged.
- Results, before and after:
  - one-element-wide torsion strip of MacNeal and Harder (1985): 4.78
    (grade F) to 0.941 (grade B); failed tests of the set from 8 to 4;
  - shell obstacle course with N = 48: Scordelis-Lo roof 0.967 to 0.996,
    pinched cylinder 0.940 to 1.001, hemisphere 0.982 to 1.011;
  - thin hinged roof of Sze et al. (2004): the lower limit load converges
    to Table 9d, +0.02 per cent with 48x48 elements, instead of 11 per
    cent below it;
  - laminated plates with 8x8 elements: 0.5 to 3 per cent from the Navier
    solution instead of 8 to 15 per cent;
  - thick rotated plate with 30x14 elements: +1.4 per cent with the
    defaults instead of +10.7 per cent with tuned factors;
  - other plate frequencies and deflections move by at most 0.4 per cent;
    the coarse meshes of the laminated cylinders are stiffer than with the
    hourglass stiffnesses reduced to 1/1000 of the former tests, with the
    same extrapolated limits.
  - Assembling `KC0` costs about 5 to 25 per cent more.
- Breaking: the arguments `hgfactor_u`, `hgfactor_v`, `hgfactor_w`,
  `hgfactor_rx` and `hgfactor_ry` of `Quad4R.update_KC0`,
  `update_probe_finte` and `update_fint` are removed and replaced by
  `hgfactor_membrane`, which multiplies the hourglass stiffness of `u`,
  `v`, and `hgfactor_physical`, which multiplies the stabilisation of
  Belytschko and Leviathan, whose formulation has no parameter, 1 by
  default, other values being for verification and sensitivity studies.
  Positional calls change: `(..., prop, update_KC0v_only,
  hgfactor_membrane, hgfactor_physical)`.

### Behaviour change: transverse shear of `Tria3R`

- `Tria3R` takes, by default, the discrete shear gap transverse shear field
  of `Tria3DSG` (new attribute `shear_model = 1`), keeping its membrane
  enriched with the drilling rotations of Allman, and `alpha_shear_locking`
  defaults to `0.`. Up to 0.11 its transverse shear was sampled at the
  centroid and divided by `1 + 0.7 l^2/h^2`, which does not converge on
  thin plates and shells (Lyly, Stenberg and Vihinen, 1993, p. 348): on a
  simply supported square plate with `a/h = 1000` the centre deflection
  stalled at -14 per cent, the buckling load at +16 per cent and the first
  frequency at +8 per cent under refinement, against 0.02 per cent now;
  the Scordelis-Lo roof stalled at 0.88 to 0.89 and the pinched hemisphere
  at 0.83 of the reference, now 0.996 and 1.008 with N = 48. The former
  field is available with `shear_model = 0` and `alpha_shear_locking =
  0.7`, for reproducing former results only.
- On flat plates in bending, buckling and vibration `Tria3R` and `Tria3DSG`
  now give the same results to round-off; `Tria3R` is better in in-plane
  bending (Cook's membrane 0.86 against 0.46 with 4x4 cells), on the
  Scordelis-Lo roof and on the twisted beam, and assembles `KC0` in about
  half the time, but is stiffer on coarse meshes of the pinched hemisphere
  (0.29 against 0.45 with N = 8, equal from N = 32). `Tria3R` is the
  recommended triangle, `Tria3DSG` the alternative on coarse meshes of
  doubly curved shells.

### Behaviour change: plane-stress invariants of `MatLamina` for any `laminaprop`

- `MatLamina.rebuild()` computes `q11`, `q12`, `q21`, `q22` and the
  invariants `u1` to `u5` with the plane-stress reduced stiffnesses,
  `sigma33 = 0` (Jones, 1999, Eq. (2.66); Reddy, 2003, Eq. (2.4.4)), also
  when `e3`, `nu13` and `nu23` are given, i.e. for a 9-entry `laminaprop`,
  and sets `q13`, `q23`, `q31`, `q32` and `q33` to zero. Before, with `e3
  != 0` they were the three-dimensional stiffnesses, such that
  `shellprop_from_LaminationParameters()`,
  `shellprop_from_lamination_parameters()`, `GradABD.calc_LP_grad()` and
  `trace_normalize_plane_stress()` gave stiffer `A`, `B`, `D` than
  `laminated_plate()` for the same material and layup, e.g. `(lambda +
  2 mu) h` instead of `E h/(1 - nu^2)` for an isotropic material, 32 per
  cent higher for `nu = 0.33`. Both routes now agree to round-off for any
  layup. The results with a 2- or 6-entry `laminaprop` and those of
  `laminated_plate()` are unchanged, bit for bit.

### Fix: lamination parameters with an offset

- `shellprop_from_LaminationParameters()`,
  `shellprop_from_lamination_parameters()` and `GradABD.calc_LP_grad()`
  take a new argument `offset`, default `0.`.
  `ShellProp.calc_lamination_parameters()` integrates `xiB` and `xiD`
  about the reference surface, i.e. including the offset, while the
  reverse route assumed the mid-surface, such that the round trip of a
  laminate with an offset gave wrong `B` and `D`, e.g. `B11` 4.8 times too
  small for a `[0/90]s` laminate with `offset = 0.2 mm` and `0.125 mm`
  plies. Given the same offset, the stiffnesses now satisfy `B = B0 + e A`
  and `D = D0 + 2 e B0 + e^2 A` (Vasiliev and Morozov, 2001, Eq. (5.28),
  p. 235) and reproduce `laminated_plate()` to round-off; the created
  property stores the offset. Results with `offset = 0` are unchanged.

### Behaviour change: transverse shear of thick `Quad4` elements

- The elements with `h/l >= 1`, `l` the square root of the element area,
  integrate the transverse shear with the assumed strains of the MITC4
  element of Dvorkin and Bathe (1984), in the plate form of Bathe and
  Dvorkin (1985), with two-by-two quadrature, instead of the modified
  one-point shear rule of Hughes et al. (1977). That rule integrates the
  products of the gradients of `w` with two-by-two quadrature and the terms
  with the rotations with one point, such that a state of pure twist, whose
  transverse shear strains vanish, stored spurious shear energy,
  `(5/12)(l/h)^2` times the twist energy of an isotropic square element,
  42 per cent at `h/l = 1`, and a mesh refined across `h/l = 1` became
  abruptly stiffer in torsion: the tip twist of a thin-walled I-beam fell by
  21 per cent, and the buckling load of a hard simply supported square
  plate with `b/h = 10` was 2.0 per cent above the Mindlin solution with
  8 x 8 elements, `h/l = 0.8`, and 11.2 per cent above it with 12 x 12,
  `h/l = 1.2`. The thick element is now exact for pure twist and pure bending
  of a rectangle, has only the six rigid-body zero-energy modes, and the
  plate converges monotonically to the Mindlin solution, +0.91 per cent
  with 12 x 12 elements. The elements with `h/l < 1` are unchanged.

### Behaviour change: geometric stiffness of `Quad4R`

- The slopes of `w` in the geometric stiffness `KG`, of `update_KG` and
  `update_KG_given_stress`, and in the geometrically nonlinear terms of
  `update_KCNL` and of the internal forces are taken at the centroid,
  the single integration point of the transverse shear and of the
  curvatures in `KC0`, instead of at the two-by-two quadrature points; the
  stress resultants keep the two-by-two quadrature. The hourglass pattern
  of `w`, which `KC0` resists only with the hourglass stabilisation, then
  has no geometric stiffness. With the slopes at the
  two-by-two points it had the geometric stiffness of the full quadrature,
  and a linear buckling analysis could return a spurious mode of `w`
  alternating in sign from node to node, below the physical buckling load:
  on a hard simply supported square plate with `b/h = 10`, 3.0 per cent
  below the Mindlin solution with 12 x 12 elements, `h/l = 1.2`, 40.8 per
  cent below with 16 x 16 and 85.7 per cent below with 48 x 48; with `b/h =
  100` and the transverse shear stiffness reduced to 1/1000, as in a
  sandwich, 59.9 per cent below with 8 x 8 elements, `h/l = 0.08`. These
  cases now give +4.7, +2.6, +0.3 and +7.0 per cent. The buckling loads of
  coarse meshes of thin plates increase slightly, e.g. from 10.0 to 11.4
  per cent above the Mindlin solution with 8 x 8 elements and `b/h = 100`,
  and `KC0 + KCNL + KG` remains the exact Jacobian of the internal forces.
  These figures were measured with the hourglass stiffness of Brockman;
  with the physical stabilisation above, the first case gives +1.5 per
  cent.

### Behaviour change: geometric stiffness of `Quad4`

- The slopes of `w` in the geometric stiffness `KG`, of `update_KG` and
  `update_KG_given_stress`, and in the geometrically nonlinear terms of
  `update_KCNL` and of the internal forces are taken at the centroid,
  instead of at the two-by-two quadrature points, as for `Quad4R`; the
  stress resultants keep the two-by-two quadrature. The elements with
  `h/l < 1` integrate the transverse shear at the centroid, so the
  hourglass pattern of `w`, with zero rotations, has no stiffness in `KC0`,
  and it now has no geometric stiffness either. With the slopes at the
  two-by-two points it had the geometric stiffness of the full quadrature,
  and a linear buckling analysis could return a spurious mode of `w`
  alternating in sign from node to node, with nearly zero rotations, below
  the physical buckling load, once the transverse shear stiffness is low
  compared with the bending stiffness: on a hard simply supported square
  plate with `b/h = 100` and the transverse shear stiffness reduced to
  1/100, as in a sandwich, 49.2 per cent below the solution of Seide and
  Stowell (1950), NACA Report 967, Eq. (1), with 16 x 16 elements, `h/l =
  0.16`, and 87.1 per cent below with 32 x 32; with 1/1000, 71.4, 92.4 and
  98.1 per cent below with 8 x 8, 16 x 16 and 32 x 32 elements. These
  cases now converge to that solution from above, as the square of the
  element size: +0.81, +0.20, +2.19, +0.54 and +0.13 per cent. The
  elements with `h/l >= 1`, whose MITC4 transverse shear resists the
  pattern, did not have the spurious mode; they take the slopes at the
  centroid as well, such that `KG` does not depend on the thickness. The
  buckling loads of coarse meshes increase: from 2.2 to 4.7 per cent above
  the Kirchhoff solution for the plate with 20 x 6 elements of
  `tests/test_quad4_linear_buckling_plate_rotated.py`, from 0.5 to 1.1 per
  cent with 40 x 12 elements; from 2.0 to 3.3 per cent above the Mindlin
  solution for the plate with `b/h = 10` and 8 x 8 elements, `h/l = 0.8`,
  of the entry above, and from 0.91 to 1.49 per cent with 12 x 12
  elements, `h/l = 1.2`. The laminated cylinder Z11 of Castro et al.
  (2014) under an end shortening, whose buckling load fell under mesh
  refinement through such a mode, to -81.6 kN with 160 elements around the
  circumference, now converges, -297.0 kN. The torsional buckling loads are
  unchanged: with a pure shear prestress the two-by-two and the centroid
  `KG` coincide on rectangles. `KC0 + KCNL + KG` remains the exact Jacobian
  of the internal forces.

### Tests

- `tests/test_vabs_isotropic.py`, `tests/test_vabs_composite.py` and
  `tests/test_vabs_elements.py`: elasticity solutions of rectangles,
  circles and tubes, including the shear correction factors of Eq. (25) of
  Yu and Hodges (2005); the box beam CUS of Yu et al. (2012), Table 2,
  reproduced to its four digits; the pipe of Yu and Hodges (2005), Table 3
  and Eq. (26); the box beam CAS1 of Yu et al. (2002), Table 2; the
  T-section of Yu and Hodges (2005), Table 1. The angle convention of
  `rotate_material` reproduces all signs of the CUS box; the three
  couplings of the pipe and of CAS1 that have the opposite sign with it
  are asserted explicitly and explained in the tests.
- `tests/test_vabs_warping.py`: the torsional warping of an ellipse against
  the Saint-Venant solution, linearity and conventions of
  `CrossSection.warping`, and the triangles of every element type.
- `tests/test_vabs_vlasov.py`: the strips of Yu et al. (2005), Tables 1, 3
  and 4, reproduced to five digits or more, the isotropic I-beam of Table
  5, within 1 per cent for the torsional rigidity and within 2 per cent,
  the grade A of MacNeal and Harder (1985), for the warping rigidity, the
  anisotropic I-beam of Table 6, with the lay-up of Volovoi et al. (1999),
  p. 1038, within 1.1 per cent for the effective torsional rigidity and 4
  per cent for the warping rigidity, the composite I-beam of Pandey, Kabir
  and Sherbourne (1995) against the asymptotic constants given by Hodges
  and Peters (2001), p. 1601, within tolerances built from the error of
  the thin-walled theory and the rounding of the printed data, 3.45 and
  2.90 per cent for the torsional rigidity at 0 and 45 degrees and 1.31 per
  cent for the warping rigidity, with the thin-walled limit checked
  separately, and the torsional warping of a
  thin-walled channel about its shear centre against the principal
  sectorial coordinate,
  `w1 = -omega kappa1`, with its warping rigidity; all with both
  `'yu2002'` and `'yu2012'`.
- `tests/test_quad4_thick_twist.py`: pure twist and pure bending exact for
  thick and thin `Quad4` elements, constant transverse shear on a distorted
  element, the rank of the thick element and the torsion of a strip refined
  across `h/l = 1`.
- `tests/test_quad4_linear_buckling_plate_rotated_thick_elements.py`: the
  plate is hard simply supported and compared with the Mindlin solution,
  within 2 per cent, instead of soft simply supported and compared with the
  Kirchhoff solution, which the spurious twist stiffness of the modified
  one-point shear rule had matched within 3 per cent;
  `tests/test_quad4_spurious_shear_mode.py` and `tests/test_drilling_rank.py`
  record the energies of the two zero-energy modes and the rank of the thick
  element.
- `tests/test_quad4r_geometric_hourglass.py`: buckling of hard simply
  supported square plates with thick elements and with a low transverse
  shear stiffness, without the spurious mode, and the one-point form of
  `KG` with constant stress resultants on a distorted element.
- `tests/test_quad4_geometric_hourglass.py`: buckling of hard simply
  supported square plates with `b/h = 100` and the transverse shear
  stiffness reduced to 1/100 and 1/1000, without the spurious mode,
  against Seide and Stowell (1950), converging from above as the square
  of the element size, within 0.5 per cent with 32 x 32 elements; the
  one-point form of
  `KG` on a distorted element, no nonlinear terms for the hourglass pattern
  of `w`, and the tangent against central differences for `h/l < 1` and
  `h/l >= 1`.
- `tests/test_quad4r_linear_buckling_plate.py` and
  `tests/test_quad4r_linear_buckling_plate_given_stress.py`: with 92x44
  elements, within 1 per cent of the exact buckling load of Leissa
  (1969), NASA SP-160, Eq. (10.46), instead of 5 per cent with 30x14
  elements; `tests/test_quad4r_linear_buckling_plate_rotated.py`: the
  buckling load of the 25 orientations is the same within 1e-6, instead
  of each within 5 per cent of the exact one.
- `tests/test_quad4r_natural_freq_pre_stress.py`: the fundamental
  frequency of a simply supported plate under half of its buckling load
  against the exact solution of Leissa (1969), Eq. (10.44), within 1 per
  cent for the three mass matrices, instead of a regression value with a
  load above the exact buckling load.
- `test_hinged_roof_thick`: with the 16x16 mesh of Table 9a of Sze et
  al. (2004) instead of 8x8, the limit loads within 1 per cent instead
  of 1.5 per cent.
- The expected values of the tests come from the literature or from a
  closed form written in the test, with its source, and no longer from the
  output of the code; where a finite element result is compared with the
  literature, the mesh is refined until the discretisation error, measured
  and stated in the docstring, is well inside the tolerance. Where the
  literature is matched only with an expensive mesh, the test runs a
  coarse mesh against the value pre-determined by the same test, to
  round-off (`rtol = 1e-4` for eigenvalues, `1e-6` for linear results),
  with the refinement table in the docstring and the literature bounded by
  the accuracy grades of MacNeal and Harder (1985), see
  `documentation_levels`. The names of the tests are kept, except
  `test_twisting_mode_is_not_stabilised_and_needs_two_elements` and
  `test_transverse_coefficient_has_no_unit_invariant_constant` of
  `tests/test_quad4r_hourglass_control.py`, which became
  `test_twisting_mode_is_stabilised` and
  `test_one_element_wide_strip_in_torsion`:
  - beams: the cantilever frequencies against the exact Timoshenko beam
    solution (Petyt, 2010, Table 3.6) instead of the Euler-Bernoulli roots
    with `beta*L = 7.885` for 7.8548; the pre-stressed cantilevers at half
    the Euler load (Timoshenko and Gere, 1961, Eqs. (2-4) and (2-57))
    against the exact solution with axial load (Bokaian, 1988; Abramovich,
    2017) instead of regression values; the curved arch against Table 6 of
    Yoon et al. (2006), with and without shear deformation, instead of the
    output of the code; the static point load against Gere and Goodno
    (2009) with the shear term instead of a web calculator; the column
    and the truss with their sources;
  - plates: the point-loaded plate against the series of Timoshenko and
    Woinowsky-Krieger (1959), Eq. (147), instead of the output of the code;
    the frequencies against Leissa (1969), Eq. (4.20); the buckling of thick
    plates against the shear-deformable solution of Seide and Stowell
    (1950), NACA Report 967, Eq. (1), instead of the classical one; the
    rotated plates assert the invariance of the buckling load under 25
    rotations;
  - cylinders: the torsional buckling of the laminates Z11 and Z33 against
    Fig. 13 of Castro et al. (2014), to which `Quad4` and `Quad4R` converge
    within 0.7 per cent once the boundary conditions CC2 of the paper are
    used, and of an isotropic cylinder against Batdorf et al. (1947), NACA
    TN 1344; the frequencies of an isotropic cylinder against Leissa (1973),
    NASA SP-288, Table 2.8, within grade A, instead of regression values;
    the laminate Z11 under end shortening converging to -278.9 kN, and the
    laminate Z22, cited from Castro et al. (2014), Tables 3 and 4, with
    `G12 = G13 = 5.695 GPa`, converging within 0.3 per cent of the 35.27 kN
    of Fig. 6(c), where it fell to 0.166 of it with 480 elements around the
    circumference before the geometric stiffness of this version;
  - shell properties: the ABD matrices against the closed forms and an
    independent lamination sum, and the transverse shear stiffness of
    Rohwer against his Table 2 and an independent implementation of his
    equations, instead of pasted values; the trace of Melo et al. (2017),
    Table 1, to half a unit of its last printed digit;
  - sandwiches: `shear_correction='rohwer'` and `'vlachoutsis'` against the
    factors and energy ratios of Vlachoutsis (1992), Tables I and II, for
    `E_f/E_c` up to 1000, where 5/6 is 42 times too stiff, and his closed
    forms, Eqs. (44), (45) and (48); an antiplane core, `E_c = 0`,
    converging to `G_c d^2/c` of Allen (1969), Eq. (2.15b), within bounds
    derived from the face terms that Allen neglects. Table 4 of Rohwer
    (1988) is not used: its bending stiffnesses contradict the stated
    lay-up, `(D11 - D22)/D13` being 4.2162 by the geometry against 4.709 to
    4.716 printed, and its shear stiffnesses need a foam shear modulus of
    0.04 instead of the stated one; the note is kept in
    `tests/test_shellprop_transverse_shear.py` and in `shell_properties`;
  - material input: constants that are not positive definite, a singular
    `ABD` with a positive diagonal, the conditioning check independent of
    the offset, an inversion that raises or returns inf or nan, antiplane
    cores, the lamination-parameter route with an offset against the
    classical lamination theory to `rtol = 1e-12`, and the plane-stress
    and three-dimensional invariants;
  - solver: completion of repeated eigenvalues, also after the second
    strategy, its stops, and the stiffness matrices that are not positive
    definite;
  - benchmarks: the drilling benchmarks, the strip torsion and the
    optional stabilisation of `Tria3DSG` graded with the accuracy grades of
    MacNeal and Harder (1985); the thin hinged roof with the 24x24 mesh of
    Table 9d of Sze et al. (2004), whose lower limit load converges to the
    table with the physical stabilisation of `Quad4R` and is no longer an
    expected failure; the constant-twist and the Belytschko-Leviathan
    bending patch tests of `Quad4R`, its stabilisation against a closed
    form and an independent implementation; the energies of the
    zero-energy modes of `Quad4` in closed form (Bathe and Dvorkin, 1985);
    the FE/Navier ratios of the
    laminated plates as converging to the Navier solution (Reddy, 2003)
    instead of locked; the von Mises truss and Ziegler's column with their
    sources.
- The tests of `Tria3R` are anchored to the literature with its new
  default; `test_tria3r_linear_buckling_plate_given_stress` rotates the
  given `N_xx` to the element axes, as `update_KG_given_stress` requires;
  the tests of the former field set `shear_model = 0` explicitly.
- The piston-theory tangent test of `Quad4R` requires an aerodynamic
  effect of 1e-4 instead of 1e-3 of the deflection, 0.09996 per cent with
  the physical stabilisation.
- Bugs of the tests: the "distorted" frequency tests copied the nodal
  coordinates before distorting the mesh and analysed the regular mesh,
  and their distortion, by the same random fraction along `x` and `y`,
  made up to 7 per cent of the elements re-entrant, which made the
  stiffness matrix indefinite from 40x50 elements;
  the constant-curvature patch test of the drilling rotations prescribed
  `r_y` with the wrong sign; the displacement-controlled cylinder built
  `KG` from a displacement vector without the prescribed shortening; the
  torsion cylinders restrained the axial displacement of a whole ring,
  which raised the buckling load by 17 per cent; `test_errors` of
  `test_shellprop.py` passed when no exception was raised, and
  `test_trace_normalized` used `Q44` for `Q66`.

### Documentation

- User guide page `cross_sectional_analysis` and API page `vabs`, with
  the figure `doc/figures/vabs_axes.svg` (TikZ source `vabs_axes.tex`) of
  the axes, the angles `theta1` and `theta3`, and the tension and shear
  centres.
- `cross_sectional_analysis` shows how to plot the warping of a channel:
  in 2D with matplotlib, through the plot directive, and in 3D with
  plotly, as an interactive figure written at the start of the build by
  `doc/source/_scripts/vabs_warping_3d.py`. The torsional warping is
  shown about the shear centre, i.e. the principal sectorial coordinate.
  matplotlib and plotly are added to `requirements_doc.txt` only.
- `cross_sectional_analysis` describes the generalized Vlasov model.
- New user guide page `documentation_levels`: the four levels of the
  documentation (functional docstrings, theory in the `Notes` and
  `References` sections, the pages, the notebooks) and what the tests
  verify and against what.
- The example pages of the plate, the column and the cylinders name the
  references of their tests, and the figures of the cylinders are those of
  the new models.
- The Notes of `pyfe3d.quad4r` describe the stabilisation of Belytschko
  and Leviathan (1994) with its equations and pages, the measurement of
  their physical membrane against that of Brockman, and the results before
  and after. `element_choice` and the README are updated for `Quad4R` and
  `Tria3R`, the recommended triangle.
- The Notes of `pyfe3d.tria3r` and `pyfe3d.tria3dsg` and `element_choice`
  give the error of the former transverse shear of `Tria3R`, which stalls
  on thin plates instead of falling as `1/n^2`, and the comparison of the
  two triangles.
- `MatLamina.rebuild` cites Tsai (2008), Theory of Composites Design, Sec.
  3.3, for the three-dimensional constants; `shell_properties` shows the
  sandwich case, cites Allen (1969) and records why Table 4 of Rohwer
  (1988) is not used; `eigenvalue_analyses` describes the completion of
  repeated eigenvalues and the positive definite stiffness matrix;
  `documentation_levels` describes the tests on coarse meshes against
  pre-determined values.
- `tut-bench-macneal-harder`, `tut-bench-shell-obstacle-course`,
  `tut-opt-topology` and `tut-nonlinear-benchmarks` re-executed with
  `Quad4R` and `Tria3R` of this version; with `hgfactor_physical = 1`, the
  default, the paths of the thinner hinged roof follow Table 9d of Sze et
  al. (2004).

### Continuous integration

- The coverage workflow runs in two parts: every test with the optimized
  build, for the Python code, and, for the Cython code, with the
  line-traced build every test except those with the new pytest marker
  `slow` (`-m "not slow"`), which take 40 s or more under line tracing
  and cover no `.pyx` line that the other tests do not, in 4 processes of
  pytest-xdist.

## 0.11.3 (2026-10-27)

Documentation only: no change of the code or of any result.

### Documentation

- `pyfe3d.shellprop`: the tensor rotation of the transverse shear stiffness is
  written as in the book, `T_\gamma^\top A_{ts} T_\gamma`, with `T_\gamma =
  [[m_{22}, m_{12}], [m_{21}, m_{11}]]` the transpose of the former `T_s`,
  in `_rotate_ply`, `get_constitutive_element` and `calc_Ats_element`;
  `calc_Ats_element` gives `T_\gamma` as a function of `\varphi = -\theta`,
  the rotation of the element frame relative to the frame of `A44`, `A45`,
  `A55`. Checked numerically: for `shear_correction='constant'`,
  `calc_Ats_element(theta)` equals this rotation to 1e-16 for an unsymmetric
  laminate.
- Derivation scripts of `BeamLR` and of the curved `BeamLR`: the moments of
  the sectional force vector are listed as `p = [N Mz My Qy Qz Mx]`, with `Mz =
  int(sigma_xx*y)` and `My = int(sigma_xx*z)`, as in Eq. (11) of Luo (2008);
  his Eq. (12), copied before, lists `My` before `Mz`. The matrix `D` is
  unchanged.
- `pyfe3d.solver.convergence`: the round-off floor cites Higham (2002), Sec.
  3.5, Eqs. (3.12) and (3.13), for the componentwise bound of the rounding
  error of a matrix-vector product.

## 0.11.0 (2026-10-26)

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

### Sign convention of the residual

- The residual of the non-linear solvers is `R = Fint - lbd*Fext`, instead
  of `lbd*Fext - Fint`, in the code and the documentation, such that the
  tangent stiffness matrix is its Jacobian, `K_T = dR/dc`, the
  Newton-Raphson corrections solve `K_T dc = -R`, and
  `dR/dlbd = -Fext`; the load vector that `calc_fext(inc=1., c=c)` and
  `calc_fext(inc=1., c=c, lbd=lbd)` return, unchanged, is `-dR/dlbd`. The
  same convention as structsolve and panels. Every operation is the
  negation of the former one, `fint - fext` instead of `fext - fint` and
  `solve_kT(-R)` instead of `solve_kT(R)`, exact in floating-point
  arithmetic: over the test suite, the load factors and the solution
  vectors of the 162 calls of `Analysis.static`, 1979 vectors, among them
  the hinged roofs of Sze et al. (2004), the low arch of Fung and Kaplan
  (1952) and the follower loads, are bitwise identical before and after,
  signs of the zeros included. The argument `R` of
  `convergence.check_convergence` and `convergence.crisfield_zeta` is
  documented as `fint - fext`; they use only norms of `R`, and return the
  same values for either sign. These functions and the non-linear solvers
  are new in this version: no function of 0.10.0 changes.

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

### Breaking: the Jacobi equilibration matrix is `P`

The diagonal matrix returned by `pyfe3d.solver.diagonal_preconditioner()`,
public since 0.10.0, where it was documented as `D`, is now `P`, as in
the book, which keeps `D` for the diagonal of the `LDL^\top` factorization:
the returned value of the docstring, `P @ K @ P` instead of `D @ K @ D`,
its Notes, the variables of `linear_buckling()`, `natural_frequency()` and
`complex_frequency()` and of the tests. The signature and the returned
matrix are unchanged, and code that names it `D` keeps working; only the
documented name changes.

### Breaking: probe attributes `Sw_x`, `Sw_y` and `Sv_x`

The rows of the slopes in the probes, public since 0.10.0, are renamed,
the underscore denoting the partial derivative, as in `w_{,x}`, without a
deprecation step:

- `Quad4Probe`, `Quad4RProbe`, `Tria3RProbe`, `Tria3DSGProbe`: `Gwx` →
  `Sw_x`, `Gwy` → `Sw_y`, the book's `\boldsymbol{G}_x = \partial
  \boldsymbol{S}^w/\partial x` and `\boldsymbol{G}_y = \partial
  \boldsymbol{S}^w/\partial y`;
- `BeamCProbe`: `Gvx` → `Sv_x`, `Gwx` → `Sw_x`, i.e. `\partial
  \boldsymbol{S}^v/\partial x` and `\partial \boldsymbol{S}^w/\partial x`;
- `BeamLRProbe`: `Gvx` → `Sv_x`, `Gwx` → `Sw_x`, the same names as in
  `BeamCProbe`, holding the interpolations of the rotations,
  `\boldsymbol{S}^{r_z}` and `-\boldsymbol{S}^{r_y}`, since `v_{,x} = r_z`
  and `w_{,x} = -r_y`.

The local pointers of the element methods and the derivation scripts
follow. No result changes: all the element outputs, matrices, internal
forces and probe arrays, are bitwise identical before and after.

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
- The local arrays of the variation of the nonlinear membrane strains of
  `Quad4`, `Quad4R`, `Tria3R` and `Tria3DSG`, and the symbols of the
  derivation scripts `derive_quad4r_KC0L_KCL0_KCLL_KGLL_sparse.py` and
  `derive_tria3r_KC0L_KCL0_KCLL_KGLL_sparse.py`, are named as in the book
  and as in the beams: `BmL`, `ABmL` and `sBmL` become `BNL`, `ABNL` and
  `sBNL`. No public name changes. The matrices, internal forces and probe
  arrays of every element are bit-identical before and after.
- The partial derivatives are named with an underscore introducing the
  derivative, as the probe attributes `Sw_x`, `Sw_y` and `Sv_x` above, in
  the elements, the derivation scripts, the tests and the tutorials:
  - `Quad4`, `Quad4R`, `Tria3R`, `Tria3DSG` and the scripts of `quad4r`,
    `tria3r` and `truss`: `N1x`, ..., `N4x`, `N1y`, ..., `N4y` become
    `S1_x`, ..., `S4_x`, `S1_y`, ..., `S4_y`, i.e. `\partial S_i/\partial
    x` and `\partial S_i/\partial y`, and `N1xy`, ... of the hourglass
    control of `Quad4R` become `S1_xy`, ...; the scripts also rename
    `N1xi`, `N1eta`, `N1xxi`, `N1xeta`, `N1yx`, ... to `S1_xi`, `S1_eta`,
    `S1_xxi`, `S1_xeta`, `S1_yx`, ..., the sensitivities `N1x_x1`, ... to
    `S1_x_x1`, ..., and the rows `Nvx`, `Nwx`, `Nwy` and `NwX` to `Sv_x`,
    `Sw_x`, `Sw_y` and `Sw_X`;
  - the derivatives of the bubbles of the Allman enrichment, `Nb1xi`,
    `Nb1eta`, `Nb1x`, `Nb1y`, ... of `pyfe3d.quad4.allman_enrichment` and
    `Nbx`, `Nby` of `pyfe3d.tria3r.allman_enrichment`, become `Sb1_xi`,
    `Sb1_eta`, `Sb1_x`, `Sb1_y`, ... and `Sb_x`, `Sb_y`; the arguments
    `Nx`, `Ny` of `allman_enrichment` and `allman_delta_Ke` of
    `pyfe3d.tria3r`, and the local arrays `Nxv`, `Nyv` of `Tria3R` and
    `Nxa`, `Nya` of `Tria3DSG` that are passed to them or used alike,
    become `S_x`, `S_y`; the local arrays `Nx`, `Ny` of
    `Tria3DSG.update_KG_given_stress`, which hold the `w` entries of the
    probe rows, become `Sw_x`, `Sw_y`;
  - the arguments `grad_x` and `grad_y` of
    `ShellProp.calc_equilibrium_transverse_shear()` and
    `calc_aposteriori_energy()`, new in this version, become `eps_x` and
    `eps_y`, the derivatives `\varepsilon_{,x}` and `\varepsilon_{,y}` of
    the generalized strains, also in `_equilibrium_integrals`, whose
    docstring writes them `\varepsilon^{(0)}_{,x}`, ... instead of
    `g_x^{(0)}`, ...;
  - tests and tutorials: `vecfxi`, `vecfxixi` of `tests/ritz.py` become
    `vecf_xi`, `vecf_xixi`; in the Ritz models of
    `tests/test_solver_linear_buckling.py` and `tests/test_solver_static.py`,
    `Swx`, `Swy`, `Swxx`, `Swyy`, `Swxy` become `Sw_x`, `Sw_y`, `Sw_xx`,
    `Sw_yy`, `Sw_xy`, `Sphixx`, `Sphixy`, `Sphiyx`, `Sphiyy` become
    `Sphix_x`, `Sphix_y`, `Sphiy_x`, `Sphiy_y`, the functions at `xi` and
    `eta`, `P_xi` and `P_eta`, become `Pxi` and `Peta`, and their
    derivatives `Px_xi`, `Pxx_xi`, `Px_eta`, ... become `Pxi_xi`,
    `Pxi_xixi`, `Peta_eta`, ...; `Nvx`, `Nwx`, `dHv1`, `dHr1`, `dHr2` of
    `tests/test_beamc_timoshenko_shape_functions.py` and the Timoshenko
    tutorial become `Sv_x`, `Sw_x`, `Hv1_x`, `Hr1_x`, `Hr2_x`, and `dNrz`
    of the tutorial `Srz_x`; `dNxi`, `dNeta` of `tests/test_element_mass.py`
    and `Nxi`, `Neta` of `tests/test_quad4r_hourglass_control.py` become
    `S_xi`, `S_eta`; `Nx`, `Ny` of `tests/test_tria3dsg.py` become `S_x`,
    `S_y`; in `tests/test_shellprop_shear_corrections.py`, `grad_x`,
    `grad_y` become `eps_x`, `eps_y`, and `gx`, `gy`, the same strains
    about another reference surface, `eps_x0`, `eps_y0`; the sensitivities
    `dKdrho`, `dEdrho`, `dCdrho` of the topology
    optimization tutorial become `K_rho`, `E_rho`, `C_rho`.

  The values of the shape functions, `N1`, ..., `N4`, become `S1`, ...,
  `S4`, as `S_i` in the documentation, in `Quad4`, `Quad4R`, `Tria3R`, the
  derivation scripts of every element and the tests, e.g.
  `tests/test_truss_consistent_mass.py`; the area coordinates `N1`, `N2`
  of the reference triangle in `tria3r/derive_tria3r_mass.py`, the
  integration variables of `h_{ij}`, keep their names. The Jacobian
  entries `j11`, ..., `J22`, the rotation interpolations `Gv1`, `Gr1`,
  ..., the interpolation matrices `Nu`, `Nv`, ..., `Nrz` of the
  derivation scripts and the quadrature weights, e.g. `wxi`, `weta`, keep
  their names. No released name changes. The matrices, internal forces and probe arrays of
  every element and the outputs of `ShellProp` are bit-identical before and
  after. The derivation scripts print the same lines apart from the names,
  except for the order of the factors, which SymPy sorts by name, in some
  lines of the geometric stiffness of `Quad4R` and `Tria3R`, the same in
  `Quad4`; these lines are pasted in the new order, which gives the same
  bits. The 14 lines of the drilling penalty of `Quad4R` and `Tria3R`
  that the renaming of the derivatives alone would have printed with
  another order of the factors are printed as in the `.pyx` again.
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
- The load stiffness matrix of a follower load is written
  `\boldsymbol{K}_F` instead of `[K_L]`, as in the book.
- The page on the non-linear static analyses gives the cases in which
  `\boldsymbol{K}_F` is symmetric, as Hibbitt (1979, pp. 938-939) and Schweizerhof and
  Ramm (1984, Table 2, Sec. 3.2.1, Eqs. 32-35) state them, and that pyfe3d
  keeps the unsymmetric `\boldsymbol{K}_F` and factorizes the tangent stiffness matrix
  with LU.
- The transformation matrix of an element is written `\boldsymbol{T}`
  instead of `[R]`, `\boldsymbol{R}` being the residual, as in the book:
  `\boldsymbol{T}` is block diagonal with one `3x3` block
  `\boldsymbol{T}_{3 \times 3}` per node and per translations or
  rotations, whose entries `T_{ij}` are stored in the attributes
  `r11, ..., r33` of the elements, with the same indices and unchanged
  names, and whose columns are the element axes in global components,
  such that `\boldsymbol{u} = \boldsymbol{T} \boldsymbol{u}_e`,
  `\boldsymbol{u}_e = \boldsymbol{T}^\top \boldsymbol{u}`, `\boldsymbol{K} =
  \boldsymbol{T} \boldsymbol{K}_e \boldsymbol{T}^\top` and `\boldsymbol{F} =
  \boldsymbol{T} \boldsymbol{F}_e`, in the conventions and overview pages, the Notes
  and attributes of the elements, the code comments and the tests. The
  bubbles of the Allman enrichment are written `\hat{S}_k` instead of
  `N_k`.
- In `pyfe3d.solver`, the basis of the null space of the constraints of
  the prescribed displacements imposed exactly is written
  `\boldsymbol{Z}` instead of `[T]`, `\boldsymbol{T}` being the
  transformation matrix of the elements: `\boldsymbol{u} = \boldsymbol{Z}
  \boldsymbol{c} + \lambda \boldsymbol{u}_k` in `Analysis`, `\boldsymbol{c} =
  \boldsymbol{Z} \boldsymbol{c}_r + \boldsymbol{c}_{p0} + \lambda
  \boldsymbol{c}_{p1}` and `\boldsymbol{Z}^\top` in the reduced residual in
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
  `\boldsymbol{C}_{sec}` instead of `[D]`; the variation of the nonlinear
  membrane strains of the shells is `\boldsymbol{B}_{NL}` instead of
  `[B_{mL}]`, as for the beams; bold symbols use `\boldsymbol`, with the accents outside, e.g.
  `\tilde{\boldsymbol{S}}`, instead of `\pmb`, and the transpose is
  `^\top` instead of `^T`, in the docstrings, the user guide and the
  tutorials. A matrix in the docstring of `Tria3R.update_KG_given_stress`
  had a single backslash as row separator.
- In the docstrings of the drilling penalty of `Quad4` and `Quad4R`, the
  interpolation of `v` is `v = S^v u_e`, as `S^u` and `S^{r_z}`, instead
  of `v = S_v u_e`, which reads as a derivative.
- Vectors and matrices as in the book, in the docstrings, the user guide,
  the tutorials, the math of the tests and the derivation scripts: vectors
  are bold lower case and matrices bold upper case, without braces or
  brackets, e.g. `\boldsymbol{u}`, `\Delta \boldsymbol{c}`,
  `\boldsymbol{K}_{C0}` and `\boldsymbol{T}_{3 \times 3}` instead of
  `\{u\}`, `\{\Delta c\}`, `[K_{C0}]` and `[T_{3\times 3}]`; the force
  vectors, the residual and the stress resultants keep their upper case,
  e.g. `\boldsymbol{F}_{ext}`, `\boldsymbol{R}` and `\boldsymbol{N}_{NL}`,
  also in the derivatives and in the scaled norms of the convergence
  criteria, e.g. `\partial \boldsymbol{R}/\partial \lambda` and
  `\|\boldsymbol{R}\|_{\boldsymbol{d}}`; the partitions are subscripts,
  e.g. `\boldsymbol{K}_{T_{fk}}` and `\boldsymbol{F}_{{ext}_f}`; the
  overbar is kept for the prescribed displacements,
  `\bar{\boldsymbol{u}}_k`; explicit lists of components keep their
  braces. The matrix KCNL of `update_KCNL`, which in every element with
  it (`Quad4`, `Quad4R`, `Tria3R`, `Tria3DSG`, `BeamC`, `BeamLR`) contains
  KC0L + KCL0 + KCLL and the geometric term KGNL, is written
  `\boldsymbol{K}_{CNL} + \boldsymbol{K}_{GNL}` (was
  `\boldsymbol{K}_{CNL}`), the book's `\boldsymbol{K}_{CNL}` having only
  the terms of `\boldsymbol{B}_{NL}`, e.g. `\boldsymbol{K}_T =
  \boldsymbol{K}_{C0} + \boldsymbol{K}_{CNL}(\boldsymbol{u}) +
  \boldsymbol{K}_{GNL}(\boldsymbol{u}) + \boldsymbol{K}_G(\boldsymbol{u})`;
  it is called the nonlinear stiffness matrix (was the nonlinear
  constitutive stiffness matrix) in the docstrings and the derivation
  scripts. The gradient matrices of the geometric stiffness are those of
  the book: `\boldsymbol{G}_x = \partial \boldsymbol{S}^w/\partial x` and
  `\boldsymbol{G}_y = \partial \boldsymbol{S}^w/\partial y` for the
  shells, whose `\boldsymbol{K}_G` and KGNL are written with them term by term instead
  of `[G]^\top [N] [G]`, and `\partial \boldsymbol{S}^v/\partial x`,
  `\partial \boldsymbol{S}^w/\partial x` (`BeamC`) or
  `\boldsymbol{S}^{r_z}`, `\boldsymbol{S}^{r_y}` (`BeamLR`) for the beams
  instead of `[G_v]` and `[G_w]`; the probe attributes are renamed
  ``Sw_x``, ``Sw_y`` and ``Sv_x``, see above. The first moment of the mass is
  `\boldsymbol{Q}` in `tests/test_element_mass.py` and
  `tut-shell-lumped-mass.ipynb`, whose shape functions are `S_i` and
  `\boldsymbol{S}`, as in the elements; the rows of the shape functions
  in `tut-timoshenko-beam-benchmarks.ipynb` are
  `\boldsymbol{S}_{r_z,x}` and `\boldsymbol{S}_\gamma`, and the hourglass
  operator of `Quad4R` is `\partial^2 \boldsymbol{S}/\partial x \partial
  y`. The second moments of the mass are `\boldsymbol{I}_2`, by analogy
  with the second moments of area, instead of `\boldsymbol{S}_2`, `S`
  being the shape functions, in `tests/test_element_mass.py` and
  `tut-shell-lumped-mass.ipynb`; the conjugate of a complex eigenvector
  is `\boldsymbol{u}^*` instead of `\bar{\boldsymbol{u}}`, the overbar
  being reserved for prescribed values; the aerodynamic matrices of the
  piston theory are written `\boldsymbol{K}_{A_\beta}`,
  `\boldsymbol{K}_{A_\gamma}` and `\boldsymbol{C}_A` instead of
  `\boldsymbol{KA}_\beta`, `\boldsymbol{K}_{A\beta}`, `\boldsymbol{KA}_\gamma`
  and `\boldsymbol{CA}` in the docstrings of `Quad4` and `Quad4R`, the
  tests and the user guide; the methods and attributes, e.g.
  `update_KA_beta`, `KA_BETA_SPARSE_SIZE` and `init_k_CA`, keep their
  names. In `pyfe3d.solver.eigen`, the permutation matrices of the sparse
  factorizations are `\boldsymbol{\Pi}`, `\boldsymbol{\Pi}_r` and
  `\boldsymbol{\Pi}_c`, `\boldsymbol{\Pi} \boldsymbol{A}
  \boldsymbol{\Pi}^\top = \boldsymbol{L} \boldsymbol{D} \boldsymbol{L}^\top`
  and `\boldsymbol{\Pi}_r \boldsymbol{A} \boldsymbol{\Pi}_c = \boldsymbol{L}
  \boldsymbol{U}`, instead of `\boldsymbol{P}`, `\boldsymbol{P}_r` and
  `\boldsymbol{P}_c`, `\boldsymbol{P}` being the Jacobi equilibration
  matrix; the shift-invert operator of the quadratic eigenvalue problem is
  `\boldsymbol{O} = (\boldsymbol{A} - \sigma \boldsymbol{B})^{-1}
  \boldsymbol{B}`, as the other operators, instead of `\boldsymbol{S}`, and
  the factorized quadratic pencil `\boldsymbol{K}_{dyn}(\sigma) = \sigma^2
  \boldsymbol{M} + \sigma \boldsymbol{C} + \boldsymbol{K}` instead of
  `\boldsymbol{Q}(\sigma)`, with the local variable `Kdyn` instead of `Q`
  in `_eigs_qep`; the conjugate of a complex eigenvalue is `\lambda^*`
  instead of `\bar{\lambda}`. The configuration-dependent load of
  `tests/test_solver_follower_loads.py` is written with the load stiffness
  matrix of the book, `\boldsymbol{K}_F = -\partial \boldsymbol{F}/\partial
  \boldsymbol{c} = \lambda \boldsymbol{K}_{F_{ref}}`, i.e.
  `\boldsymbol{F}(\boldsymbol{c}, \lambda) = \lambda (\boldsymbol{q}_0 -
  \boldsymbol{K}_{F_{ref}} \boldsymbol{c})`, instead of `\lambda
  (\boldsymbol{q}_0 + \boldsymbol{Q} \boldsymbol{c})`; the variable `Q` of
  the test, `-\boldsymbol{K}_{F_{ref}}`, keeps its name. The tutorials were
  not re-executed: only their Markdown changed.
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
- `Spring` verified in an arbitrary orientation against `\boldsymbol{T}
  \boldsymbol{K}_e \boldsymbol{T}^\top`,
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

Re-evaluating the method of Rohwer for each element costs about 80 ns per ply,
2.8 us with 32 plies (see the cost comparison below). Instead, the compliance
`S = inv(Ats)` in the element frame is represented exactly by a Fourier series
in `2 theta` with 5 harmonics: the equilibrium distribution `f(z)` is a polynomial of degree 4 in
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
