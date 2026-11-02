Shell elements: kinematics and formulation
==========================================

The four shell elements of ``pyfe3d``, the quadrilaterals :class:`.Quad4` and
:class:`.Quad4R` and the triangles :class:`.Tria3DSG` and :class:`.Tria3R`,
are flat facets with the six degrees of freedom `u`, `v`, `w`, `r_x`,
`r_y`, `r_z` per node, such that they share the nodes of the beams, of the
springs and of each other, see :doc:`conventions`. A curved shell is
modelled as an assembly of flat facets, each one a plate in which the
membrane and the bending stiffnesses are superimposed and coupled through
the laminate. This chapter gives the theory that the four elements have in
common: the kinematics, the constitutive relation, the interpolation, the
element matrices of the linear and of the geometrically non-linear
analyses, and the reason why each element treats the transverse shear
differently. The pages of the elements give what is specific to each one:
:doc:`shell_quad4`, :doc:`shell_quad4r`, :doc:`shell_tria3dsg` and
:doc:`shell_tria3r`; the stiffness of the drilling rotation and the mass
matrices are treated in :doc:`shell_drilling` and :doc:`shell_mass`, the
properties in :doc:`shell_properties`, and the choice of an element in
:doc:`element_choice`.


Facets and element coordinates
------------------------------

Every element matrix is computed in the element coordinate system `x`,
`y`, `z`, with `z` normal to the facet, and rotated to the global one with
the transformation matrix `\boldsymbol{T}` of the element, see
:doc:`conventions`:

.. math::
    :label: sh-transformation

    \boldsymbol{K} = \boldsymbol{T} \boldsymbol{K}_e \boldsymbol{T}^\top,
    \qquad
    \boldsymbol{F} = \boldsymbol{T} \boldsymbol{F}_e

The three nodes of a triangle define its plane. The four nodes of a
quadrilateral on a curved surface are in general not coplanar, and the
matrices of :class:`.Quad4` and :class:`.Quad4R` are those of the flat
element in the mean plane of the four nodes, which is parallel to both
diagonals; the rigid offsets that link the flat element to the warped
nodes are described in :doc:`element_choice`, section *Warped
quadrilaterals*. The coordinates of the nodes in the element system are
the attribute ``xe`` of the probe of each element, e.g.
:class:`.Quad4Probe`, set by ``update_probe_xe``.


Kinematics of the first-order shear deformation theory
------------------------------------------------------

The elements follow the first-order shear deformation theory (FSDT). The
displacements of a point at the distance `z` from the reference surface
are

.. math::
    :label: sh-displacement-field

    u(x, y, z) = u(x, y) + z \, r_y(x, y), \qquad
    v(x, y, z) = v(x, y) - z \, r_x(x, y), \qquad
    w(x, y, z) = w(x, y)

with `r_x` and `r_y` the right-hand rotations about the element axes. In
terms of the rotations of the transverse normal of the FSDT, `\phi_x` and
`\phi_y`, which give `u + z\phi_x` and `v + z\phi_y`, they are `r_y =
\phi_x` and `r_x = -\phi_y`. Keeping only the linear terms, the in-plane
strains are the sum of a part constant through the thickness and of a
part linear in `z`, and the transverse shear strains are constant through
the thickness:

.. math::
    :label: sh-strains

    \boldsymbol{\varepsilon}^{(0)} =
    \begin{Bmatrix} \varepsilon_{xx} \\ \varepsilon_{yy} \\ \gamma_{xy}
    \end{Bmatrix} =
    \begin{Bmatrix} u_{,x} \\ v_{,y} \\ u_{,y} + v_{,x} \end{Bmatrix},
    \qquad
    \boldsymbol{\varepsilon}^{(1)} =
    \begin{Bmatrix} \kappa_{xx} \\ \kappa_{yy} \\ \kappa_{xy}
    \end{Bmatrix} =
    \begin{Bmatrix} r_{y,x} \\ -r_{x,y} \\ r_{y,y} - r_{x,x}
    \end{Bmatrix},
    \qquad
    \boldsymbol{\gamma}^{(0)} =
    \begin{Bmatrix} \gamma_{yz} \\ \gamma_{xz} \end{Bmatrix} =
    \begin{Bmatrix} w_{,y} - r_x \\ w_{,x} + r_y \end{Bmatrix}

The membrane strains `\boldsymbol{\varepsilon}^{(0)}` depend on `u` and
`v`, the curvatures `\boldsymbol{\varepsilon}^{(1)}`, written
`\boldsymbol{\kappa}` in the Notes of the elements, on `r_x` and `r_y`,
and the transverse shear strains `\boldsymbol{\gamma}^{(0)}` on `w`, `r_x`
and `r_y`. The transverse shear strains are ordered `\{\gamma_{yz},
\gamma_{xz}\}`, as the transverse shear stiffnesses `A_{44}`, `A_{45}`,
`A_{55}` of :class:`.ShellProp`, index 4 for `yz` and 5 for `xz`. The
drilling rotation `r_z` appears in none of the strains of
:eq:`sh-strains`: its stiffness is the subject of :doc:`shell_drilling`.
The rows of these strains are the attributes ``BLexx``, ``BLeyy``,
``BLgxy``, ``BLkxx``, ``BLkyy``, ``BLkxy`` and of the transverse shear of
the probes, see :meth:`.Quad4Probe.update_BL`.


Stress resultants and constitutive relation
-------------------------------------------

With the membrane forces `\boldsymbol{N} = \{N_{xx}, N_{yy}, N_{xy}\}`, the
moments `\boldsymbol{M} = \{M_{xx}, M_{yy}, M_{xy}\}` and the transverse
shear forces `\boldsymbol{Q} = \{Q_y, Q_x\}`, the thickness integrals of
the stresses, of the stresses times `z` and of the transverse shear
stresses, and the plane-stress stiffnesses of the plies, the constitutive
relation of the element is

.. math::
    :label: sh-abd

    \begin{Bmatrix} \boldsymbol{N} \\ \boldsymbol{M} \end{Bmatrix} =
    \begin{bmatrix} \boldsymbol{A} & \boldsymbol{B} \\ \boldsymbol{B} &
    \boldsymbol{D} \end{bmatrix}
    \begin{Bmatrix} \boldsymbol{\varepsilon}^{(0)} \\
    \boldsymbol{\varepsilon}^{(1)} \end{Bmatrix},
    \qquad
    \boldsymbol{Q} = \boldsymbol{A}_{ts} \boldsymbol{\gamma}^{(0)},
    \qquad
    \boldsymbol{A}_{ts} = \begin{bmatrix} A_{44} & A_{45} \\ A_{45} & A_{55}
    \end{bmatrix}

with the extensional, extension-bending coupling and bending stiffness
matrices `\boldsymbol{A}`, `\boldsymbol{B}`, `\boldsymbol{D}` and the
transverse shear stiffness matrix `\boldsymbol{A}_{ts}`, which contains the
shear correction, of a :class:`.ShellProp`, see :doc:`shell_properties`.
The coupling matrix `\boldsymbol{B}` is not to be confused with the
strain-displacement matrices below, which always carry a subscript. The
elements read the four matrices in their own coordinate system through the
rotation ``m11, m12, m21, m22`` from the material direction to the element
`x` axis, see :meth:`.ShellProp.calc_constitutive_element` and
:meth:`.ShellProp.calc_Ats_element`, and apply no shear correction factor
of their own.

The strain energy of the element is

.. math::
    :label: sh-strain-energy

    U = \frac{1}{2} \int_A \left(
    \boldsymbol{\varepsilon}^{(0)\top} \boldsymbol{A}
    \boldsymbol{\varepsilon}^{(0)}
    + 2 \boldsymbol{\varepsilon}^{(0)\top} \boldsymbol{B}
    \boldsymbol{\varepsilon}^{(1)}
    + \boldsymbol{\varepsilon}^{(1)\top} \boldsymbol{D}
    \boldsymbol{\varepsilon}^{(1)}
    + \boldsymbol{\gamma}^{(0)\top} \boldsymbol{A}_{ts}
    \boldsymbol{\gamma}^{(0)} \right) dA + U_{r_z}

with `U_{r_z}` the energy of the drilling term, to which :class:`.Quad4R`
adds the energy of its hourglass control, see :doc:`shell_quad4r`.


Interpolation and element matrices
----------------------------------

All six fields are interpolated with the same shape functions `S_i`, the
bilinear functions of the natural coordinates `\xi`, `\eta` for the
quadrilaterals,

.. math::
    :label: sh-shape-quad

    S_1 = \frac{(1 - \xi)(1 - \eta)}{4}, \quad
    S_2 = \frac{(1 + \xi)(1 - \eta)}{4}, \quad
    S_3 = \frac{(1 + \xi)(1 + \eta)}{4}, \quad
    S_4 = \frac{(1 - \xi)(1 + \eta)}{4}

and the area coordinates `S_1`, `S_2`, `S_3` for the triangles, such that
`u = \boldsymbol{S}^u \boldsymbol{u}_e`, ..., `r_z = \boldsymbol{S}^{r_z}
\boldsymbol{u}_e`, with `\boldsymbol{u}_e` the element displacements in the
order of :doc:`conventions`. The same functions map the geometry of the
quadrilateral, and the Cartesian derivatives follow from the Jacobian
matrix,

.. math::
    :label: sh-jacobian

    \begin{Bmatrix} \partial/\partial x \\ \partial/\partial y
    \end{Bmatrix} = \boldsymbol{J}^{-1}
    \begin{Bmatrix} \partial/\partial \xi \\ \partial/\partial \eta
    \end{Bmatrix},
    \qquad
    \boldsymbol{J} = \begin{bmatrix} x_{,\xi} & y_{,\xi} \\ x_{,\eta} &
    y_{,\eta} \end{bmatrix},
    \qquad dx \, dy = \det \boldsymbol{J} \, d\xi \, d\eta

whereas the derivatives of the area coordinates are constant, e.g.
`S_{1,x} = (y_2 - y_3)/(2 A_e)` and `S_{1,y} = (x_3 - x_2)/(2 A_e)`, with
`A_e` the area of the triangle. The strains :eq:`sh-strains` become
`\boldsymbol{\varepsilon}^{(0)} = \boldsymbol{B}_m \boldsymbol{u}_e`,
`\boldsymbol{\varepsilon}^{(1)} = \boldsymbol{B}_b \boldsymbol{u}_e` and
`\boldsymbol{\gamma}^{(0)} = \boldsymbol{B}_s \boldsymbol{u}_e`, with

.. math::
    :label: sh-B-operators

    \boldsymbol{B}_m = \begin{bmatrix} \boldsymbol{S}^u_{,x} \\
    \boldsymbol{S}^v_{,y} \\ \boldsymbol{S}^u_{,y} + \boldsymbol{S}^v_{,x}
    \end{bmatrix},
    \qquad
    \boldsymbol{B}_b = \begin{bmatrix} \boldsymbol{S}^{r_y}_{,x} \\
    -\boldsymbol{S}^{r_x}_{,y} \\ \boldsymbol{S}^{r_y}_{,y} -
    \boldsymbol{S}^{r_x}_{,x} \end{bmatrix},
    \qquad
    \boldsymbol{B}_s = \begin{bmatrix} \boldsymbol{S}^w_{,y} -
    \boldsymbol{S}^{r_x} \\ \boldsymbol{S}^w_{,x} + \boldsymbol{S}^{r_y}
    \end{bmatrix}

and the constitutive stiffness matrix of the element, the Hessian of
:eq:`sh-strain-energy`, is

.. math::
    :label: sh-KC0

    \boldsymbol{K}_{C0_e} = \int_A
    \begin{bmatrix} \boldsymbol{B}_m \\ \boldsymbol{B}_b \end{bmatrix}^\top
    \begin{bmatrix} \boldsymbol{A} & \boldsymbol{B} \\ \boldsymbol{B} &
    \boldsymbol{D} \end{bmatrix}
    \begin{bmatrix} \boldsymbol{B}_m \\ \boldsymbol{B}_b \end{bmatrix} dA
    + \int_A \boldsymbol{B}_s^\top \boldsymbol{A}_{ts} \boldsymbol{B}_s \, dA
    + \boldsymbol{K}_{C0_e}^{r_z}

computed by ``update_KC0``, e.g. :meth:`.Quad4.update_KC0`, and rotated
with :eq:`sh-transformation`. With `\boldsymbol{B} = \boldsymbol{0}` the
first integral splits into a membrane term on `u`, `v` and a bending term
on `r_x`, `r_y`; with an offset of the reference surface or an
unsymmetric laminate they are coupled. The internal forces of
``update_fint`` are the gradient of :eq:`sh-strain-energy`,
`\boldsymbol{K}_{C0_e} \boldsymbol{u}_e` in the linear case.

The elements differ in the shear strains `\boldsymbol{B}_s`, in the
quadrature of each integral and in the drilling term:

.. list-table:: Quadrature of each term (``drilling_model = 0``, the default)
    :header-rows: 1

    * - Term
      - :class:`.Quad4`
      - :class:`.Quad4R`
      - :class:`.Tria3DSG`
      - :class:`.Tria3R`
    * - membrane, coupling
      - 3x3 Gauss points, with the Allman enrichment
      - centroid, with the hourglass stiffness of Brockman
      - centroid (exact)
      - centroid, plus the Allman terms with 3 points
    * - bending
      - 2x2
      - centroid, with the physical stabilisation
      - centroid (exact)
      - centroid (exact)
    * - transverse shear
      - assumed strains of MITC4, 2x2
      - assumed strains, constant part at the centroid, with the physical
        stabilisation
      - discrete shear gap, constant (exact)
      - discrete shear gap, constant (exact)
    * - drilling
      - Hughes-Brezzi term at the centroid
      - Hughes-Brezzi term, 2x2
      - Hughes-Brezzi term, 3 points
      - Hughes-Brezzi term at the centroid
    * - ``KG``, ``KCNL``, non-linear forces
      - 2x2
      - 2x2, slopes of `w` at the centroid
      - constant (exact)
      - constant (exact)

The triangles are linear: their strains `\boldsymbol{B}_m`,
`\boldsymbol{B}_b` and `\boldsymbol{B}_s` are constant, and one point
integrates them exactly, with the weight `A_e`, i.e. `1/2` times `\det
\boldsymbol{J} = 2 A_e`. The bilinear quadrilateral has strains that vary
over the element, and the choice of the quadrature is what controls its
locking and its spurious modes.


Shear locking and spurious modes
--------------------------------

The transverse shear stiffness scales with `h` and the bending stiffness
with `h^3`, such that their ratio grows as `(\ell/h)^2`, `\ell` a length
of the element. In bending, the FSDT approaches the constraint
`\boldsymbol{\gamma}^{(0)} \to \boldsymbol{0}` of the Kirchhoff plate. If
the interpolated `\boldsymbol{B}_s` cannot vanish in a state of pure
bending, the element stores parasitic shear energy that grows as
`(\ell/h)^2` relative to its bending energy, and becomes far too stiff:
this is shear locking. The bilinear quadrilateral integrated with two by
two points locks, because its displacement-based shear strains contain
bilinear terms that a constant curvature cannot switch off; the linear
triangle locks because its `\nabla w` is a single constant vector while
the rotations are linear, such that no nodal values satisfy
`\boldsymbol{\gamma}^{(0)} = \boldsymbol{0}` in a state of constant
curvature.

Integrating the shear with fewer points removes the parasitic terms, but
leaves zero-energy (hourglass) modes, deformation patterns with no strain
at the integration points: with one point for the shear, the bilinear
quadrilateral has two of them besides the rigid-body modes (Hughes et al.,
1977). ``pyfe3d`` uses four remedies, each with its own page:

- **assumed transverse shear strains**, the mixed interpolation of the
  covariant shear strains of Dvorkin and Bathe (1984), tied at the
  mid-points of the edges: :class:`.Quad4`, free of shear locking and of
  spurious modes, see :doc:`shell_quad4`;
- **one-point integration with physical stabilisation**: :class:`.Quad4R`
  integrates every term at the centroid and gives the hourglass modes of
  the bending and of the transverse shear the stiffness of the non-constant
  part of the strains, after Belytschko and Leviathan (1994), with the same
  assumed shear strains, see :doc:`shell_quad4r`;
- **discrete shear gap** of Bletzinger, Bischoff and Ramm (2000): the shear
  strains of both triangles are the gradient of an interpolated shear gap,
  which vanishes identically in a state of constant curvature, see
  :doc:`shell_tria3dsg`;
- **algebraic rescaling** of the transverse shear stiffness by `1/(1 +
  \alpha \ell^2/h^2)`, Lyly, Stenberg and Vihinen (1993), available on
  the triangles with ``alpha_shear_locking``, ``0.`` by default, see
  :doc:`shell_tria3dsg` and :doc:`shell_tria3r`.

The first three remove the locking in the strain operator, such that the
transverse shear stiffness of the :class:`.ShellProp` is used as the
laminate gives it, and the shear correction keeps its meaning.


Geometrically non-linear terms
------------------------------

The geometrically non-linear analyses use the von Kármán membrane
strains of the facet,

.. math::
    :label: sh-von-karman

    \varepsilon_{xx} = u_{,x} + \frac{1}{2} w_{,x}^2, \qquad
    \varepsilon_{yy} = v_{,y} + \frac{1}{2} w_{,y}^2, \qquad
    \gamma_{xy} = u_{,y} + v_{,x} + w_{,x} w_{,y}

whose non-linear part is `\boldsymbol{\varepsilon}_{NL} = \frac{1}{2}
\boldsymbol{B}_{NL} \boldsymbol{u}_e`, with the rows `w_{,x}
\boldsymbol{G}_x`, `w_{,y} \boldsymbol{G}_y` and `w_{,x} \boldsymbol{G}_y
+ w_{,y} \boldsymbol{G}_x` of `\boldsymbol{B}_{NL}`, `\boldsymbol{G}_x =
\partial \boldsymbol{S}^w/\partial x` and `\boldsymbol{G}_y = \partial
\boldsymbol{S}^w/\partial y` being the rows of the slopes of `w`, the
attributes ``Sw_x`` and ``Sw_y`` of the probes. The curvatures and the
transverse shear strains remain linear. The tangent stiffness matrix of
the internal forces of ``update_fint`` with ``nonlinear=1`` is split into
three matrices,

.. math::
    :label: sh-tangent

    \boldsymbol{K}_T = \boldsymbol{K}_{C0} + \left( \boldsymbol{K}_{CNL}
    + \boldsymbol{K}_{GNL} \right) + \boldsymbol{K}_G

computed by ``update_KC0``, ``update_KCNL`` and ``update_KG``, e.g.
:meth:`.Quad4.update_KCNL`. The geometric stiffness matrix of the membrane
forces `\{N_{xx}, N_{yy}, N_{xy}\} = \boldsymbol{A}
\boldsymbol{\varepsilon}^{(0)} + \boldsymbol{B}
\boldsymbol{\varepsilon}^{(1)}` of the linear strains is

.. math::
    :label: sh-KG

    \boldsymbol{K}_G = \int_A \left( N_{xx} \boldsymbol{G}_x^\top
    \boldsymbol{G}_x + N_{yy} \boldsymbol{G}_y^\top \boldsymbol{G}_y +
    N_{xy} \left( \boldsymbol{G}_x^\top \boldsymbol{G}_y +
    \boldsymbol{G}_y^\top \boldsymbol{G}_x \right) \right) dA

which involves only `w`, the displacement along the normal of the
element, as in the Donnell theory. It is linear in the displacements,
such that `(\boldsymbol{K}_{C0} + \lambda \boldsymbol{K}_G) \boldsymbol{u}
= \boldsymbol{0}` is the linear buckling problem of a pre-buckling state
multiplied by `\lambda`, see :doc:`eigenvalue_analyses`;
``update_KG_given_stress`` computes it for given constant membrane
forces. `\boldsymbol{K}_{CNL}` collects the constitutive terms that
contain `\boldsymbol{B}_{NL}`, `\boldsymbol{B}_m^\top \boldsymbol{A}
\boldsymbol{B}_{NL} + \boldsymbol{B}_b^\top \boldsymbol{B}
\boldsymbol{B}_{NL}`, its transpose and `\boldsymbol{B}_{NL}^\top
\boldsymbol{A} \boldsymbol{B}_{NL}`, and `\boldsymbol{K}_{GNL}` is
:eq:`sh-KG` with the membrane forces `\boldsymbol{A}
\boldsymbol{\varepsilon}_{NL}` of the non-linear strains; it is computed
with `\boldsymbol{K}_{CNL}` so that `\boldsymbol{K}_G` stays linear.
Crisfield (1991), Section 8.1.5, groups the same terms differently, with
the total membrane forces in his initial-stress matrix, and the sum is the
same. Each part of the strain energy is differentiated with its own
quadrature and with the same slopes of `w`, such that :eq:`sh-tangent` is
the exact Jacobian of the internal forces and the Newton-Raphson
iterations of :doc:`nonlinear_analyses` converge quadratically. In
:class:`.Quad4` and :class:`.Quad4R` the point where the slopes of `w` are
taken matters, because of the hourglass pattern `w = \xi \eta`, which has
no slope at the centroid, see :doc:`shell_quad4`.

The von Kármán strains hold for moderate rotations. For large rotations,
every shell element can be wrapped in the element-independent
co-rotational formulation of :class:`.CorotationalShell`, see
:doc:`nonlinear_analyses`.


Aerodynamic matrices of the piston theory
-----------------------------------------

:class:`.Quad4` and :class:`.Quad4R` give the matrices of the aerodynamic
load of a supersonic flow along the global `X` axis, flowing towards the
positive `X`. The linear piston theory, with the correction of Krumhaar
(1963) for a cylinder, gives the load per unit area along the normal of
the element

.. math::
    :label: sh-piston

    q = -\beta \, w_{n,X} + \gamma \, w_n - \mu \, \dot{w}_n

with `w_n = T_{13} u + T_{23} v + T_{33} w` the displacement along the
element normal written with the global translations, whatever the side of
the element on which the flow is. With the bilinear functions
`\boldsymbol{S}` acting on the nodal values of `w_n`, the matrices of
:meth:`.Quad4.update_KA_beta`, :meth:`.Quad4.update_KA_gamma` and
:meth:`.Quad4.update_CA` are

.. math::
    :label: sh-piston-matrices

    \boldsymbol{K}_{A_\beta} = -\int_A \boldsymbol{S}^\top \left( T_{11}
    \boldsymbol{S}_{,x} + T_{12} \boldsymbol{S}_{,y} \right) dA, \qquad
    \boldsymbol{K}_{A_\gamma} = \int_A \boldsymbol{S}^\top \boldsymbol{S}
    \, dA, \qquad
    \boldsymbol{C}_A = -\int_A \boldsymbol{S}^\top \boldsymbol{S} \, dA

`T_{11} \boldsymbol{S}_{,x} + T_{12} \boldsymbol{S}_{,y}` being the
derivative along `X` projected on the element. The matrices contain no
aerodynamic coefficient, and the equations of motion are

.. math::
    :label: sh-flutter

    \boldsymbol{M} \ddot{\boldsymbol{u}} - \mu \boldsymbol{C}_A
    \dot{\boldsymbol{u}} + \left( \boldsymbol{K} - \beta
    \boldsymbol{K}_{A_\beta} - \gamma \boldsymbol{K}_{A_\gamma} \right)
    \boldsymbol{u} = \boldsymbol{0}

with, for the density `\rho_\infty`, the speed `U` and the Mach number `M`
of the flow, `\beta = \rho_\infty U^2/\sqrt{M^2 - 1}`, `\mu = \beta (M^2 -
2)/(U (M^2 - 1))`, and `\gamma = \beta/(2 R \sqrt{M^2 - 1})` for the flow
over the convex side of a cylinder of radius `R`, `\gamma = 0` for a flat
panel. The system has the form of the damped problem of
:func:`.complex_frequency`, see :doc:`eigenvalue_analyses`. The triangles
have no aerodynamic matrices.


Limits of validity
------------------

- **Facets.** The curvature of a shell is represented only by the angles
  between neighbouring facets, and the membrane-bending coupling of a
  curved shell appears only through them; the drilling enrichment of :class:`.Quad4`
  and :class:`.Tria3R` stiffens coarse meshes of doubly curved shells, see
  :doc:`shell_tria3r`. A warped quadrilateral is represented by its flat
  projection, corrected with rigid offsets up to the warping ratios given
  in :doc:`element_choice`.
- **FSDT.** Plane stress in each ply, an inextensible transverse normal and
  transverse shear strains constant through the thickness, whose energy is
  made consistent by the shear correction of the :class:`.ShellProp`; the
  transverse shear stresses must be recovered from the shear forces, see
  :doc:`shell_properties`.
- **Material.** Linear elastic, small strains, no thermal strains.
- **Von Kármán strains.** Moderate rotations about the in-plane axes;
  larger rotations need the co-rotational formulation.
- **Thin limit.** Hughes et al. (1977) note that the transverse shear
  stiffness is of the order of `(\ell/h)^2` times the bending stiffness,
  such that round-off removes the bending stiffness of very thin elements.
  ``pyfe3d`` applies no correction: in double precision the central
  deflection of a square plate with eight by eight :class:`.Quad4`
  elements changes by `10^{-4}` from `\ell/h = 1.25 \times 10^3` to `1.25
  \times 10^5`, and by -33 % at `\ell/h = 1.25 \times 10^7`, where the
  round-off dominates, see the Notes of :mod:`pyfe3d.quad4`.


References
----------

- Bletzinger, K.-U., Bischoff, M. and Ramm, E., A unified approach for
  shear-locking-free triangular and rectangular shell finite elements,
  Computers & Structures, 75(3), 321-334, 2000.
  https://doi.org/10.1016/S0045-7949(99)00140-6
- Belytschko, T. and Leviathan, I., Physical stabilization of the 4-node
  shell element with one point quadrature, Computer Methods in Applied
  Mechanics and Engineering, 113(3-4), 321-350, 1994.
  https://doi.org/10.1016/0045-7825(94)90052-3
- Crisfield, M. A., Non-linear Finite Element Analysis of Solids and
  Structures, Volume 1: Essentials, John Wiley & Sons, Chichester, 1991.
- Dvorkin, E. N. and Bathe, K. J., A continuum mechanics based four-node
  shell element for general non-linear analysis, Engineering Computations,
  1(1), 77-88, 1984. https://doi.org/10.1108/eb023562
- Hughes, T. J. R., Taylor, R. L. and Kanoknukulchai, W., A simple and
  efficient finite element for plate bending, International Journal for
  Numerical Methods in Engineering, 11(10), 1529-1543, 1977.
  https://doi.org/10.1002/nme.1620111005
- Krumhaar, H., The accuracy of linear piston theory when applied to
  cylindrical shells, AIAA Journal, 1(6), 1448-1449, 1963.
- Lyly, M., Stenberg, R. and Vihinen, T., A stable bilinear element for
  the Reissner-Mindlin plate model, Computer Methods in Applied Mechanics
  and Engineering, 110(3-4), 343-357, 1993.
  https://doi.org/10.1016/0045-7825(93)90214-I
