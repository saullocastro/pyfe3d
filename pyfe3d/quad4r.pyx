#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: nonecheck=False
#cython: overflowcheck=False
#cython: initializedcheck=False
#cython: embedsignature=True
#cython: infer_types=False
r"""
==============================================================================
Quad4R - Quadrilateral element with reduced integration (:mod:`pyfe3d.quad4r`)
==============================================================================

.. currentmodule:: pyfe3d.quad4r

The module has three classes: :class:`.Quad4RData`, with the sizes of the
sparse arrays that one element fills; :class:`.Quad4RProbe`, with the
buffers in element coordinates that one or more elements share; and
:class:`.Quad4R`, the element. The ``update_*`` methods of :class:`.Quad4R`
write the element contributions in sparse row, column and value arrays, from
the positions given by its ``init_k_*`` attributes, or add them to a global
internal force vector.

The :class:`.Quad4R` element has 6 degrees-of-freedom (DOF): `u`, `v`, `w`,
`r_x`, `r_y`, `r_z`. All DOF are interpolated bi-linearly between the nodes.

All stiffness terms, besides drilling, are integrated using the reduced
integration scheme for the :class:`.Quad4R` element, with a single point at
the centroid, making it very efficient concerning the time needed to
calculate the internal forces and stiffness matrices, in comparison with the
:class:`.Quad4` element. The hourglass modes of the bending and of the
transverse shear, of `w`, `r_x` and `r_y`, are controlled by the physical
stabilisation of Belytschko and Leviathan (1994), which has no parameter,
and those of the membrane, of `u` and `v`, by the artificial stiffness of
Brockman (1987); ``hgfactor_physical`` and ``hgfactor_membrane`` of
:meth:`.Quad4R.update_KC0`, :meth:`.Quad4R.update_probe_finte` and
:meth:`.Quad4R.update_fint` multiply them.

The drilling rotation `r_z` is tied to the in-plane rotation of the membrane
field by a term integrated with two points per direction, whose coefficient
is selected with the ``drilling_model`` attribute of :class:`.Quad4R`: the
default ``0`` is physics-based, any other value gives the fictitious penalty
of MSC Nastran and Autodesk Nastran controlled by the ``K6ROT`` attribute.
Unlike :class:`.Quad4` and :class:`.Tria3R`, this element does
**not** enrich the in-plane displacement field with the drilling rotations.
Use :class:`.Quad4` to use Allman's enrichment.

The transverse shear stiffnesses `A_{44}`, `A_{45}` and `A_{55}` are read
from the :class:`pyfe3d.shellprop.ShellProp` object with the shear correction
already applied, see :meth:`pyfe3d.shellprop.ShellProp.calc_transverse_shear_stiffness`,
and no shear correction factor is applied by the element. When a material
direction is defined, they are brought to the element coordinate system with
:meth:`pyfe3d.shellprop.ShellProp.calc_Ats_element`.

Notes
-----
The element follows the first-order shear deformation theory (FSDT). The
displacements at a distance `z` from the reference surface are `u + z r_y`,
`v - z r_x` and `w`, which gives, in the element coordinate system, the
membrane strains, the curvatures and the transverse shear strains

.. math::

    \epsilon_{xx} = u_{,x}, \quad \epsilon_{yy} = v_{,y}, \quad
    \gamma_{xy} = u_{,y} + v_{,x}

.. math::

    \kappa_{xx} = {r_y}_{,x}, \quad \kappa_{yy} = -{r_x}_{,y}, \quad
    \kappa_{xy} = {r_y}_{,y} - {r_x}_{,x}

.. math::

    \gamma_{xz} = w_{,x} + r_y, \quad \gamma_{yz} = w_{,y} - r_x

and the strain energy

.. math::

    U = \frac{1}{2} \int_A \left( \boldsymbol{\epsilon}^\top \boldsymbol{A} \boldsymbol{\epsilon}
        + 2 \boldsymbol{\epsilon}^\top \boldsymbol{B} \boldsymbol{\kappa}
        + \boldsymbol{\kappa}^\top \boldsymbol{D} \boldsymbol{\kappa}
        + \boldsymbol{\gamma}^\top \boldsymbol{A}_s \boldsymbol{\gamma} \right) dA + U_{r_z} + U_{hg}

with `\boldsymbol{\epsilon} = \{\epsilon_{xx}, \epsilon_{yy}, \gamma_{xy}\}^\top`,
`\boldsymbol{\kappa} = \{\kappa_{xx}, \kappa_{yy}, \kappa_{xy}\}^\top`,
`\boldsymbol{\gamma} = \{\gamma_{yz}, \gamma_{xz}\}^\top`, `\boldsymbol{A}`, `\boldsymbol{B}` and `\boldsymbol{D}` the
laminate stiffness matrices and `\boldsymbol{A}_s` the matrix of `A_{44}`, `A_{45}` and
`A_{55}`, all in the element coordinate system, `U_{r_z}` the drilling
term and `U_{hg}` the hourglass term described below. All DOF are
interpolated bi-linearly between the nodes, such that any of the DOF
gradients can be constant over the element when the element is rectangular.

.. rubric:: Integration scheme and hourglass control

In the reduced integration scheme used, a single point at the centroid
(`\xi=\eta=0`) and weight `w_{ij}=4`, the membrane, bending and transverse
shear terms leave hourglass modes without stiffness. Those of the bending
and of the transverse shear are controlled with the physical stabilisation
of Belytschko and Leviathan (1994) [9]_, described below, and those of the
membrane with the artificial stiffness of Brockman 1987 [1]_.

.. rubric:: Physical stabilisation of the bending and of the transverse shear

The transverse shear is that of Belytschko and Leviathan (1994) [9]_: the
assumed covariant shear strains of Dvorkin and Bathe (1984) [10]_, tied at
the mid-sides, `e_{\xi z}` at `(0, \pm 1)` and `e_{\eta z}` at
`(\pm 1, 0)`, from `\gamma_{xz} = w_{,x} + r_y` and `\gamma_{yz} = w_{,y} -
r_x`, split into a constant part and a nonconstant part, linear in `\eta`
for `\gamma_{xz}` and in `\xi` for `\gamma_{yz}`, both mapped to Cartesian
components with the Jacobian at the centroid, their Eqs. (3.35)-(3.42),
pp. 328-329. The constant part is integrated at the centroid, their
Eq. (4.12), p. 330, and the nonconstant part gives the stabilisation of
their Eq. (4.13), p. 331, integrated exactly with `\int \xi^2 dA = \int
\eta^2 dA = A/3` and `\int \xi \eta \, dA = 0`, their Eqs. (4.15) and
(4.10), with the transverse shear stiffness `A_{44}`, `A_{45}` and
`A_{55}`.

The bending is stabilised in the same way: with the bilinear shape
functions written as `N_I = \Delta_I + x b_{xI} + y b_{yI} + h \gamma_I`,
their Eq. (3.23), `h = \xi \eta` and `\gamma_I = (h_I - (\boldsymbol{h}^\top
\boldsymbol{x}) b_{xI} - (\boldsymbol{h}^\top \boldsymbol{y}) b_{yI})/4`,
Eq. (3.24), the derivatives are `N_{I,x} = b_{xI} + h_{,x} \gamma_I`,
Eq. (3.25), and the nonconstant part of the curvatures, `\kappa_{xx} =
h_{,x} q_y`, `\kappa_{yy} = -h_{,y} q_x` and `\kappa_{xy} = h_{,y} q_y -
h_{,x} q_x` with `q_x = \boldsymbol{\gamma}^\top \boldsymbol{r}_x` and `q_y =
\boldsymbol{\gamma}^\top \boldsymbol{r}_y`, Eq. (3.28), p. 327, gives the
stabilisation of Eqs. (4.13), (B.17) and (B.18), pp. 331 and 349, with
the integrals `H_{xx}`, `H_{yy}` and `H_{xy}` of `h_{,x}^2`, `h_{,y}^2`
and `h_{,x} h_{,y}` of their Eq. (4.14), here integrated with two by two
Gauss points, and the bending stiffness `\boldsymbol{D} - \boldsymbol{B}
\boldsymbol{A}^{-1} \boldsymbol{B}` of the laminate in the element
coordinate system, which is `\boldsymbol{D}` for a symmetric laminate,
`E t^3/12` times `\hat{\boldsymbol{C}}_b` of Eq. (B.18) for an isotropic
plate, and does not depend on the reference surface set by
``prop.offset``. For a parallelogram the one-point bending plus this
stabilisation is the bending integrated with two by two points.

Neither stabilisation has a parameter; ``hgfactor_physical`` multiplies
both, ``1`` being the formulation of the paper, other values serving
verification and sensitivity studies. The rigid-body motions and every
state of constant strain, of constant curvature, including a uniform twist
`w = kxy`, `r_x = kx`, `r_y = -ky`, and of constant transverse shear have
no nonconstant strain, on any quadrilateral, and the assumed shear strains
of constant curvature vanish there too: the element passes the
constant-twist patch test and the thin-plate bending patch test of
Belytschko and Leviathan, their Section 6.1, Fig. 3 and Tables 2 and 3,
pp. 336-337, for any thickness, length unit and ``hgfactor_physical``, see
``test_constant_twist_patch_test`` and
``test_bending_patch_test_belytschko_leviathan`` in
``test_quad4r_hourglass_control.py``. The hourglass modes of `w`, `r_x`
and `r_y` and the twisting mode of Brockman's Eq. (13) have nonconstant
strains, so a single element is singular only in its six rigid-body
motions. Both stiffnesses scale with the length unit as the physical ones
do, so the response of a model is independent of the unit of length.

.. rubric:: Hourglass control of the membrane

The hourglass modes of `u` and `v` have the artificial stiffness of
Brockman 1987 [1]_, with the orthogonalised hourglass operator of his
Eq. (14), which he quotes from Belytschko and Tsay 1983 [2]_. Brockman
considers only isotropic plates, and his generalized stiffnesses contain
the Young's modulus `E` times `t`; for a laminate, `E` is replaced by the
membrane moduli `E_{1eq}` and `E_{2eq}` of the extensional compliance
`\boldsymbol{A}^{-1}` in the element coordinate system, see
:meth:`.Quad4R.update_KC0`, which reduce to `E` for an isotropic plate.
The generalized stiffnesses of Brockman's Eq. (16a),

.. math::

    E^{(h)}_u = E^{(h)}_v = \frac{0.10 E t}{1 + 1/A}

are not dimensionally homogeneous, `1/A` being an inverse area. The
hourglass term enters the element matrix as `K_{ij} = A E^{(h)} \gamma_i
\gamma_j`, with `\gamma = \partial^2 \boldsymbol{S}/\partial x \partial y`
at the centroid, of dimension `1/L^2`, so `E^{(h)}` must be a force times a
length; Brockman's factor `1/(1 + 1/A)` tends to `A` only for `A \ll 1`,
the regime in which he calibrated it, "over a range of six orders of
magnitude in the planform dimension". This element uses the area itself,

.. math::

    E^{(h)}_u = 0.10 \, f_m E_{1eq} t A
    \qquad
    E^{(h)}_v = 0.10 \, f_m E_{2eq} t A

with `f_m` = ``hgfactor_membrane``, which scales with the length unit as
the extensional stiffness `E t` does: Cook's in-plane bending problem is
reproduced to twelve significant figures over nine orders of magnitude of
length unit, see ``test_hourglass_stiffness_is_independent_of_the_length_unit``.

Belytschko and Leviathan stabilise the membrane physically too, with the
nonconstant part of the membrane strains, their Eqs. (3.27) and (B.13),
which on a flat element is the bilinear membrane integrated with two by two
points and locks in in-plane bending. Measured on 2026-10-29 with that
stabilisation, integrated with two by two points, in place of Brockman's:
the in-plane tip load of the straight cantilever of MacNeal and Harder
(1985) [7]_, 6 by 1 elements, gives 0.076, 0.024 and 0.028 times the
reference on the regular, trapezoidal and parallelogram meshes, against
0.119, 0.033 and 0.064 with Brockman's; Cook's membrane gives 11.32, 18.14,
22.44 and 24.18 with 2x2 to 16x16 elements, against 13.95, 20.28, 23.41 and
24.54 (reference 23.96). Neither is free of locking on the one-element-deep
strips, and the physical form is the stiffer, so the membrane keeps
Brockman's stabilisation. :class:`.Quad4`, with its drilling enrichment,
does not lock in in-plane bending.

Up to 0.11 the hourglass modes of `w`, `r_x` and `r_y` also had Brockman's
artificial stiffnesses, `0.1 E t^3/(1 + 1/A)` for `w` and `0.1 E_{b} t^3
A` for the rotations, with the bending moduli, and the transverse shear was
the displacement-based one at the centroid. The operator `\gamma^\top w`
is `w_{,xy}`, `k` for a uniform twist, and added the fraction `0.6 (1 +
\nu)/(1 + 1/A)` to its energy, 0.78 for `\nu = 0.3` and `A \gg 1`, a
failure of the constant-twist patch test that refinement did not remove
where the element areas were large in the unit of length of the model; the
parameters were ``hgfactor_u``, ``hgfactor_v``, ``hgfactor_w``,
``hgfactor_rx`` and ``hgfactor_ry``. The consequences, measured on
2026-10-29, before and after:

* the one-element-wide torsion strip of MacNeal and Harder (1985) [7]_:
  4.78 times the reference twist, grade F, against 0.941, grade B, the
  value of their QUAD4, see ``test_one_element_wide_strip_in_torsion``;
* the hinged cylindrical roof of Sze et al. (2004) [8]_, `t = 6.35`: the
  lower limit load converged to -0.1435 to -0.1440 `P_{max}`, 11 % below
  the -0.1288 of their Table 9d, against 0.2 % with 32x32 elements on the
  quarter, see ``test_hinged_roof_thin_snap_back``; with `t = 12.7` the
  linear deflection was 6 to 7 % smaller than with :class:`.Tria3DSG`;
* the thin-plate bending patch test above: the interior deflections were
  0.88 to 1.21 times the exact ones;
* simply supported laminated plates under uniform pressure, 8x8 elements:
  the centre deflection was 8 to 15 % below the Navier solution, against
  0.5 to 3 %, see ``test_laminated_plates_against_navier``;
* the thin plates of ``test_quad4r_linear_buckling_plate.py``: the
  buckling load was 0.27 % above Leissa's with 92x44 elements, against
  0.05 %; the thick plate of
  ``test_quad4r_linear_buckling_plate_rotated_thick_elements.py``, 30x14
  elements: 10.7 % above the reference with tuned hourglass factors,
  against 1.4 % with the default.

The operators and the coefficients, as computed, are given in the Notes of
:meth:`.Quad4R.update_KC0`.

.. rubric:: Transverse shear stiffness

When a material direction is defined,
:meth:`pyfe3d.shellprop.ShellProp.calc_Ats_element` re-evaluates the
equilibrium-based stiffness of Rohwer (1988) [6]_ with the plies rotated to
the element coordinate system, such that the assumed cylindrical bending
states are posed along the element axes.

.. rubric:: Drilling stiffness

Two drilling models are available, selected with the ``drilling_model``
attribute, whose default is ``0``, see Hughes and Brezzi [3]_ and
Allman [4]_.

Both use the same operator, the regularised one of Hughes and
Brezzi (1989) that ties the drilling rotation `r_z` to the in-plane rotation
`\theta_z = (v_{,x} - u_{,y})/2`, integrated with two points per direction,
and they differ only in the coefficient `\gamma_{r_z}` that multiplies it:

* ``drilling_model = 0``, the default, uses the physics-based value
  `\gamma_{r_z} = A_{66}` identified by Hughes and Brezzi, which is a
  stiffness per unit length taken from the laminate and not an adjustable
  parameter;
* ``drilling_model = 1`` uses the fictitious value `\gamma_{r_z} = K6ROT
  \cdot 10^{-6} A_{66}` of MSC Nastran and Autodesk Nastran, derived below.

Unlike :class:`.Quad4` and :class:`.Tria3R`, this element does
**not** use the hierarchical edge-mode enrichment of the in-plane
displacement field of Allman (1984) that accompanies the Hughes-Brezzi term
in those two elements. The enrichment exists to remove the excessive
in-plane bending stiffness of the bilinear displacement field, and the
single-point integration of this element already removes it: on Cook's
membrane problem, with a reference tip displacement of 23.96,

.. table::

    ============  ============  ============  ============  ============
    mesh          Quad4R        Quad4         Quad4         Quad4R
                  (no           penalty       enriched      enriched
                  enrichment)                               at 3x3
    ============  ============  ============  ============  ============
    2 by 2        30.14         11.85         20.75         20.74
    4 by 4        24.66         18.30         22.83         22.81
    8 by 8        24.08         22.08         23.54         23.52
    16 by 16      23.97         23.43         23.81         23.79
    ============  ============  ============  ============  ============

so the enrichment recovers for :class:`.Quad4` what the reduced
integration already gives here, and adding it to this element makes it less
accurate and more computationally expensive.

Integrating the enrichment consistently with the single point of the element
was also evaluated and rejected. The enriched membrane operator sampled at
one point supplies at most three independent rows over the twelve in-plane
degrees-of-freedom, and a rectangle then keeps one spurious in-plane
zero-energy mode, namely

.. math::
    u \propto x \quad , \quad v \propto -y \quad , \quad
    r_z \propto \boldsymbol{h} = [1, -1, 1, -1]^\top

in which the uniform membrane strain of the `u` and `v` fields is exactly
cancelled at the centroid by the enrichment strain of the hourglass pattern
of `r_z`. That mode escapes every term of the element: the one-point
membrane term because its centroidal strain vanishes, the hourglass control
of Brockman because `u` and `v` are linear fields and carry no hourglass
content, and the Hughes-Brezzi term because `r_z - \theta_z` vanishes not
only at the integration points but identically over the element, so no
quadrature of that term can reach it. Removing it would require a further
artificial stabilisation of the drilling rotation, of the kind this element
already uses for its other degrees-of-freedom, for a change of the Cook
results above of less than half a per cent.

The ``drilling_model = 1`` branch follows the approach adopted in MSC
Nastran and Autodesk Nastran, using a penalty-based method. It provides a
non-physical drilling stiffness with the objective of removing the
singularity, by creating an artificial stiffness between the drilling
rotation degree-of-freedom of each node and the in-plane rotational strain.
The operator and its quadrature, two points per direction, are the same as
for ``drilling_model = 0``, only the coefficient differing. The penalty
energy is defined per element as:

.. math::

    U_{r_z} = \frac{1}{2} K6ROT \cdot 10^{-6} \cdot \int_A A_{66} (r_z - \theta_z)^2 dA

where `10^{-6}` is a scaling factor suggested by MSC Nastran's approach (CQUAD4) to make the artificial
drilling stiffness sufficiently small. AUTODESK NASTRAN's quick reference guide recommends `K6ROT = 100`
for static analysis. For modal solutions, `K6ROT = 10^4` is suggested. MSC NASTRAN's quick reference guide
states that `K6ROT > 100` should not be used, thus contradicting AUTODESK NASTRAN. The rotation `r_z` represents
the drilling degree-of-freedom in element's coordinates, whereas `\theta_z` the in-plane rotation strain, defined as:

.. math::

    \theta_z = \frac{1}{2}\left(\frac{\partial v}{\partial x} - \frac{\partial u}{\partial y}\right)

The first variation of `U_{r_z}` then becomes:

.. math::

    \delta U_{r_z} = K6ROT \cdot 10^{-6} \cdot \int_A A_{66} (r_z - \theta_z)(\delta r_z - \delta \theta_z) dA

which can be expressed in terms of the shape functions and element degrees-of-freedom (`u_e`) as:
`r_z = S^{r_z} u_e`, `u = S^u u_e` and `v = S^v u_e` as:

.. math::

    \delta U_{r_z} = K6ROT \cdot 10^{-6} \cdot u_e^\top \int_A A_{66} (S^{r_z \top} + 1/2 S^{u \top}_{,y} - 1/2 S^{v \top}_{,x})(\delta S^{r_z} + 1/2 \delta S^u_{,y} - 1/2 S^v_{,x}) dA u_e

or simply as:

.. math::

    \delta U_{r_z} = K6ROT \cdot 10^{-6} \cdot A_{66} u_e^\top \int_A B_{r_z}^\top B_{r_z} dA u_e

with:

.. math::

    B_{r_z} = S^{r_z} + 1/2 S^u_{,y} - 1/2 S^v_{,x}

Note that `A_{66}` is assumed constant over the element, which here has no difference given that a reduced integration approach is used.
The approach herein presented is very similar to the one presented in Eq. 2.20 of Adam et al. [5]_.

References
----------
.. [1] Brockman, R. A., Dynamics of the Bilinear Mindlin Plate Element,
   International Journal for Numerical Methods in Engineering, 24(12),
   2343-2356, 1987. https://doi.org/10.1002/nme.1620241208
.. [2] Belytschko, T., and Tsay, C. S., A stabilization procedure for the
   quadrilateral plate element with one-point quadrature, International
   Journal for Numerical Methods in Engineering, 19(3), 405-419, 1983.
   https://doi.org/10.1002/nme.1620190308
.. [3] Hughes, T. J. R., and Brezzi, F., On drilling degrees of freedom,
   Computer Methods in Applied Mechanics and Engineering, 72(1), 105-121,
   1989. https://doi.org/10.1016/0045-7825(89)90124-2
.. [4] Allman, D. J., A compatible triangular element including vertex
   rotations for plane elasticity analysis, Computers & Structures,
   19(1-2), 1-8, 1984. https://doi.org/10.1016/0045-7949(84)90197-4
.. [5] Adam, F. M., Mohamed, A. E., and Hassaballa, A. E., Degenerated Four
   Nodes Shell Element with Drilling Degree of Freedom, IOSR Journal of
   Engineering, 3(8), 10-20, 2013.
.. [6] Rohwer, K., Improved transverse shear stiffness for layered finite
   elements, DFVLR-FB 88-32, 1988.
.. [7] MacNeal, R. H., and Harder, R. L., A proposed standard set of problems
   to test finite element accuracy, Finite Elements in Analysis and Design,
   1(1), 3-20, 1985. https://doi.org/10.1016/0168-874X(85)90003-4
.. [8] Sze, K. Y., Liu, X. H., and Lo, S. H., Popular benchmark problems
   for geometric nonlinear analysis of shells, Finite Elements in Analysis
   and Design, 40(11), 1551-1569, 2004.
   https://doi.org/10.1016/j.finel.2003.11.001
.. [9] Belytschko, T., and Leviathan, I., Physical stabilization of the
   4-node shell element with one point quadrature, Computer Methods in
   Applied Mechanics and Engineering, 113(3-4), 321-350, 1994.
   https://doi.org/10.1016/0045-7825(94)90052-3
.. [10] Dvorkin, E. N., and Bathe, K. J., A continuum mechanics based
   four-node shell element for general non-linear analysis, Engineering
   Computations, 1(1), 77-88, 1984. https://doi.org/10.1108/eb023562

"""
from libc.math cimport fabs

import numpy as np

from .shellprop cimport ShellProp

cdef int DOF = 6
cdef int NUM_NODES = 4


# NOTE rigid offsets of a warped element, see the attribute warp_offsets of
#      the element class. The flat element lies in the mean plane of the
#      nodes, the plane z = zm of the element coordinates, which is parallel
#      to both diagonals, and node a is at the height z_a = xe[3*a + 2] - zm
#      above its node. The degrees-of-freedom of the flat element are
#      J ue: u_f = u - z_a r_y, v_f = v + z_a r_x, the others unchanged.
#      With z_a = 0 every function below leaves its result unchanged, bit
#      for bit
cdef inline int _warp_z(double *xe, int warp_offsets, double *z) noexcept nogil:
    r"""Heights ``z`` of the nodes above the flat element, returns 1 when
    one of them is not zero and ``warp_offsets`` is not ``0``"""
    cdef int a, warped
    cdef double zm
    warped = 0
    for a in range(4):
        z[a] = 0.
    if warp_offsets == 0:
        return 0
    # NOTE exactly xe[2] when the four heights are equal
    zm = 0.25*((xe[2] + xe[8]) + (xe[5] + xe[11]))
    for a in range(4):
        z[a] = xe[3*a + 2] - zm
        if z[a] != 0.:
            warped = 1
    return warped


cdef inline void _warp_ue(double *z, double *ue, double *uf) noexcept nogil:
    r"""``uf = J ue``, the displacements of the flat element"""
    cdef int i, a
    for i in range(24):
        uf[i] = ue[i]
    for a in range(4):
        uf[6*a + 0] = ue[6*a + 0] - z[a]*ue[6*a + 4]
        uf[6*a + 1] = ue[6*a + 1] + z[a]*ue[6*a + 3]


cdef inline void _warp_f(double *z, double *f) noexcept nogil:
    r"""``f = J^T f`` in place, forces of the flat element to the nodes"""
    cdef int a
    for a in range(4):
        f[6*a + 3] += z[a]*f[6*a + 1]
        f[6*a + 4] -= z[a]*f[6*a + 0]


cdef inline void _warp_K(double *z, double *K) noexcept nogil:
    r"""``K = J^T K J`` in place, ``K`` of size 24x24 stored row by row"""
    cdef int a, p
    for a in range(4):
        for p in range(24):
            K[24*p + 6*a + 3] += z[a]*K[24*p + 6*a + 1]
            K[24*p + 6*a + 4] -= z[a]*K[24*p + 6*a + 0]
    for a in range(4):
        for p in range(24):
            K[24*(6*a + 3) + p] += z[a]*K[24*(6*a + 1) + p]
            K[24*(6*a + 4) + p] -= z[a]*K[24*(6*a + 0) + p]


cdef inline void _warp_dK(double *z, double *K, double *dK) noexcept nogil:
    r"""``dK = J^T K J - K``, without forming ``J^T K J``, 24x24 row by
    row: ``K J - K`` followed by the rows of ``J^T`` applied to ``K J``"""
    cdef int a, p
    for p in range(576):
        dK[p] = 0.
    for a in range(4):
        for p in range(24):
            dK[24*p + 6*a + 3] = z[a]*K[24*p + 6*a + 1]
            dK[24*p + 6*a + 4] = -z[a]*K[24*p + 6*a + 0]
    # NOTE the rows 6*a and 6*a + 1 of dK are not changed below
    for a in range(4):
        for p in range(24):
            dK[24*(6*a + 3) + p] += z[a]*(K[24*(6*a + 1) + p]
                                          + dK[24*(6*a + 1) + p])
            dK[24*(6*a + 4) + p] -= z[a]*(K[24*(6*a + 0) + p]
                                          + dK[24*(6*a + 0) + p])


cdef inline void _rotate_add_24(double *dK, double *R, double *Kv) noexcept nogil:
    r"""``Kv[24*I + J] += (T dK T^T)[I, J]``, with ``T`` block diagonal, the
    3x3 matrix ``R[3*m + i]`` of ``r11, ..., r33`` on each block of three
    degrees-of-freedom"""
    cdef int bi, bj, m, n, i, j
    cdef double s
    for bi in range(8):
        for bj in range(8):
            for m in range(3):
                for n in range(3):
                    s = 0.
                    for i in range(3):
                        for j in range(3):
                            s += R[3*m + i]*dK[24*(3*bi + i) + 3*bj + j]*R[3*n + j]
                    Kv[24*(3*bi + m) + 3*bj + n] += s


cdef inline void _warp_KE_sub(double *z, double *f, double *K) noexcept nogil:
    r"""``K -= K_E`` in element coordinates, ``K`` of size 24x24 stored row
    by row: the eccentricity geometric stiffness of :func:`_warp_KE`, before
    its rotation to global coordinates, on the rotations of each node, from
    the forces ``f`` of the nodes of the flat element in element
    coordinates, see the Notes of ``update_KCNL``"""
    cdef int a, p
    for a in range(4):
        # NOTE row r_x of node a, columns r_x, r_y, r_z of node a
        p = 24*(6*a + 3) + 6*a + 3
        K[p] -= z[a]*f[6*a + 2]
        K[p + 2] += 0.5*z[a]*f[6*a + 0]
        K[p + 24 + 1] -= z[a]*f[6*a + 2]
        K[p + 24 + 2] += 0.5*z[a]*f[6*a + 1]
        K[p + 48] += 0.5*z[a]*f[6*a + 0]
        K[p + 48 + 1] += 0.5*z[a]*f[6*a + 1]


cdef inline void _warp_KE(double *z, double *f, double *R, double *KE) noexcept nogil:
    r"""Eccentricity geometric stiffness of the rotations of each node, in
    global coordinates, ``KE[9*a + 3*m + n]``, from the forces ``f`` of the
    nodes of the flat element in element coordinates: ``(n.r) I - (n r^T +
    r n^T)/2`` with ``r = z_a e_3``, see the Notes of ``update_KG``"""
    cdef int a, m, n, i, j
    cdef double nx, ny, nz, s
    cdef double L[9]
    for a in range(4):
        nx = f[6*a + 0]
        ny = f[6*a + 1]
        nz = f[6*a + 2]
        L[0] = z[a]*nz
        L[1] = 0.
        L[2] = -0.5*z[a]*nx
        L[3] = 0.
        L[4] = z[a]*nz
        L[5] = -0.5*z[a]*ny
        L[6] = -0.5*z[a]*nx
        L[7] = -0.5*z[a]*ny
        L[8] = 0.
        for m in range(3):
            for n in range(3):
                s = 0.
                for i in range(3):
                    for j in range(3):
                        s += R[3*m + i]*L[3*i + j]*R[3*n + j]
                KE[9*a + 3*m + n] = s


cdef inline void _reduced_bending_stiffness(double *Ae, double *Be,
        double *De, double *Ds) noexcept nogil:
    r"""Bending stiffness with zero membrane forces, ``Ds = D - B A^-1 B``

    From the row-major 3x3 matrices ``Ae``, ``Be`` and ``De`` in the element
    coordinate system. It is ``D`` itself when ``B`` is zero and does not
    depend on the reference surface set by ``ShellProp.offset``. Used by the
    physical stabilisation of the bending, see :func:`_bending_bl94`.

    """
    cdef int i, j, k
    cdef double detA
    cdef double Ai[9]
    cdef double BAi[9]
    # NOTE inverse of the extensional stiffness by cofactors
    detA = (Ae[0]*(Ae[4]*Ae[8] - Ae[5]*Ae[7])
            - Ae[1]*(Ae[3]*Ae[8] - Ae[5]*Ae[6])
            + Ae[2]*(Ae[3]*Ae[7] - Ae[4]*Ae[6]))
    Ai[0] = (Ae[4]*Ae[8] - Ae[5]*Ae[7])/detA
    Ai[1] = (Ae[2]*Ae[7] - Ae[1]*Ae[8])/detA
    Ai[2] = (Ae[1]*Ae[5] - Ae[2]*Ae[4])/detA
    Ai[3] = (Ae[5]*Ae[6] - Ae[3]*Ae[8])/detA
    Ai[4] = (Ae[0]*Ae[8] - Ae[2]*Ae[6])/detA
    Ai[5] = (Ae[2]*Ae[3] - Ae[0]*Ae[5])/detA
    Ai[6] = (Ae[3]*Ae[7] - Ae[4]*Ae[6])/detA
    Ai[7] = (Ae[1]*Ae[6] - Ae[0]*Ae[7])/detA
    Ai[8] = (Ae[0]*Ae[4] - Ae[1]*Ae[3])/detA
    for i in range(3):
        for j in range(3):
            BAi[3*i + j] = 0.
            for k in range(3):
                BAi[3*i + j] += Be[3*i + k]*Ai[3*k + j]
    for i in range(3):
        for j in range(3):
            Ds[3*i + j] = De[3*i + j]
            for k in range(3):
                Ds[3*i + j] -= BAi[3*i + k]*Be[3*k + j]


cdef inline void _bending_bl94(double x1, double x2, double x3, double x4,
        double y1, double y2, double y3, double y4, double *Ds, double fH,
        double *K) noexcept nogil:
    r"""Physical stabilisation of the bending of Belytschko and Leviathan
    (1994), added to ``K``

    ``K`` is the 12 x 12 row-major matrix of :func:`_shear_bl94`, on ``w,
    r_x, r_y`` of nodes 1 to 4 in element coordinates, to which this adds
    the rows and columns of ``r_x`` and ``r_y``. The bilinear shape
    functions are written as in their Eq. (3.23), ``N_I = Delta_I + x b_xI +
    y b_yI + h gamma_I``, with ``h = xi*eta`` and ``gamma_I = (h_I - (h.x)
    b_xI - (h.y) b_yI)/4``, Eq. (3.24d), so that ``N_I,x = b_xI + h,x
    gamma_I``, Eq. (3.25). The nonconstant part of the curvatures,
    ``kappa_xx = h,x q_y``, ``kappa_yy = -h,y q_x`` and ``kappa_xy = h,y q_y
    - h,x q_x`` with ``q_x = gamma^T r_x`` and ``q_y = gamma^T r_y``, Eq.
    (3.28), gives the stiffness of their Eqs. (4.13), (B.17) and (B.18),
    with ``H_xx``, ``H_yy`` and ``H_xy`` of Eq. (4.14) integrated with two
    by two Gauss points and the bending stiffness ``Ds`` of the laminate,
    which reduces to ``a**2/12*C_b*a`` of Eq. (B.18) for an isotropic plate;
    multiplied by ``fH``.

    """
    cdef int I, J, p, q
    cdef double xs[4]
    cdef double ys[4]
    cdef double XI[4]
    cdef double ETA[4]
    cdef double h[4]
    cdef double bx[4]
    cdef double by[4]
    cdef double gam[4]
    cdef double pts[2]
    cdef double xi, eta, xxi, yxi, xeta, yeta, detJ, hx, hy
    cdef double hdotx, hdoty, detJ0
    cdef double Hxx, Hyy, Hxy, Qxx, Qyy, Qxy
    cdef double D11, D12, D16, D22, D26, D66
    xs[0] = x1; xs[1] = x2; xs[2] = x3; xs[3] = x4
    ys[0] = y1; ys[1] = y2; ys[2] = y3; ys[3] = y4
    XI[0] = -1.; XI[1] = 1.; XI[2] = 1.; XI[3] = -1.
    ETA[0] = -1.; ETA[1] = -1.; ETA[2] = 1.; ETA[3] = 1.
    D11 = Ds[0]; D12 = Ds[1]; D16 = Ds[2]
    D22 = Ds[4]; D26 = Ds[5]; D66 = Ds[8]
    # NOTE centroidal derivatives b_x, b_y and the projected hourglass
    #      vector gamma of Eqs. (3.24)
    xxi = 0.; yxi = 0.; xeta = 0.; yeta = 0.
    for I in range(4):
        xxi += 0.25*XI[I]*xs[I]
        yxi += 0.25*XI[I]*ys[I]
        xeta += 0.25*ETA[I]*xs[I]
        yeta += 0.25*ETA[I]*ys[I]
    detJ0 = xxi*yeta - yxi*xeta
    hdotx = 0.
    hdoty = 0.
    for I in range(4):
        h[I] = XI[I]*ETA[I]
        bx[I] = (yeta*0.25*XI[I] - yxi*0.25*ETA[I])/detJ0
        by[I] = (-xeta*0.25*XI[I] + xxi*0.25*ETA[I])/detJ0
        hdotx += h[I]*xs[I]
        hdoty += h[I]*ys[I]
    for I in range(4):
        gam[I] = 0.25*(h[I] - hdotx*bx[I] - hdoty*by[I])
    # NOTE H_xx, H_yy, H_xy of Eq. (4.14), per unit thickness, with h,x
    #      and h,y the Cartesian derivatives of h = xi*eta
    pts[0] = -0.5773502691896257645092
    pts[1] = +0.5773502691896257645092
    Hxx = 0.; Hyy = 0.; Hxy = 0.
    for p in range(2):
        for q in range(2):
            xi = pts[p]
            eta = pts[q]
            xxi = 0.; yxi = 0.; xeta = 0.; yeta = 0.
            for I in range(4):
                xxi += 0.25*XI[I]*(1 + ETA[I]*eta)*xs[I]
                yxi += 0.25*XI[I]*(1 + ETA[I]*eta)*ys[I]
                xeta += 0.25*ETA[I]*(1 + XI[I]*xi)*xs[I]
                yeta += 0.25*ETA[I]*(1 + XI[I]*xi)*ys[I]
            detJ = xxi*yeta - yxi*xeta
            # NOTE [h,x, h,y] = J^-1 [h,xi, h,eta] = J^-1 [eta, xi]
            hx = (yeta*eta - yxi*xi)/detJ
            hy = (-xeta*eta + xxi*xi)/detJ
            Hxx += detJ*hx*hx
            Hyy += detJ*hy*hy
            Hxy += detJ*hx*hy
    # NOTE energy H_xx a^T Ds a + 2 H_xy a^T Ds c + H_yy c^T Ds c with
    #      a = [q_y, 0, -q_x] and c = [0, -q_x, q_y], over 2
    Qxx = Hxx*D66 + Hyy*D22 + 2*Hxy*D26
    Qyy = Hxx*D11 + Hyy*D66 + 2*Hxy*D16
    Qxy = -Hxx*D16 - Hyy*D26 - Hxy*(D12 + D66)
    for I in range(4):
        for J in range(4):
            K[12*(3*I + 1) + 3*J + 1] += fH*Qxx*gam[I]*gam[J]
            K[12*(3*I + 2) + 3*J + 2] += fH*Qyy*gam[I]*gam[J]
            K[12*(3*I + 1) + 3*J + 2] += fH*Qxy*gam[I]*gam[J]
            K[12*(3*I + 2) + 3*J + 1] += fH*Qxy*gam[I]*gam[J]


cdef inline void _shear_bl94(double x1, double x2, double x3, double x4,
        double y1, double y2, double y3, double y4, double A44, double A45,
        double A55, double fH, double *K) noexcept nogil:
    r"""Transverse shear of Belytschko and Leviathan (1994) minus the
    one-point displacement-based transverse shear

    Writes in ``K`` the 12 x 12 row-major matrix, on the degrees-of-freedom
    ``w, r_x, r_y`` of nodes 1 to 4 in element coordinates, to be added to
    the stiffness matrix whose transverse shear is
    ``A*B0^T*C_s*B0``, ``B0`` the displacement-based shear strains
    ``gamma_xz = w_x + r_y`` and ``gamma_yz = w_y - r_x`` at the centroid.

    The assumed covariant shear strains of Dvorkin and Bathe (1984), as in
    Belytschko and Leviathan (1994), Eqs. (3.35)-(3.39), are interpolated
    from the tying points ``B (0, -1)``, ``D (0, 1)``, ``A (-1, 0)`` and
    ``C (1, 0)``: ``e_xi = (eB + eD)/2 + eta*(eD - eB)/2`` and
    ``e_eta = (eA + eC)/2 + xi*(eC - eA)/2``, mapped to Cartesian components
    with the Jacobian ``J0`` at the centroid. The constant part replaces the
    displacement-based centroid value, Eq. (4.12), and the nonconstant part
    gives the stabilisation of Eq. (4.13), integrated with
    ``int eta**2 dA = int xi**2 dA = A/3`` and ``int xi*eta dA = 0``, Eqs.
    (4.15) and (4.10), multiplied by ``fH``; the coupling of the constant
    and nonconstant parts is omitted, as in that paper.

    """
    cdef int i, j, k, I
    cdef double xs[4]
    cdef double ys[4]
    cdef double XI[4]
    cdef double ETA[4]
    cdef double tie_xi[4]
    cdef double tie_eta[4]
    cdef double N[4]
    cdef double Nxi[4]
    cdef double Neta[4]
    cdef double xxi, yxi, xeta, yeta
    cdef double J0[4]
    cdef double J0inv[4]
    cdef double detJ0, A
    cdef double e[4][12]
    cdef double ec[2][12]
    cdef double G0m[2][12]
    cdef double G0d[2][12]
    cdef double D1[2][12]
    cdef double D2[2][12]
    cdef double ex[2][12]
    cdef double Cs[4]
    xs[0] = x1; xs[1] = x2; xs[2] = x3; xs[3] = x4
    ys[0] = y1; ys[1] = y2; ys[2] = y3; ys[3] = y4
    XI[0] = -1.; XI[1] = 1.; XI[2] = 1.; XI[3] = -1.
    ETA[0] = -1.; ETA[1] = -1.; ETA[2] = 1.; ETA[3] = 1.
    # NOTE tying points B, D (e_xi) and A, C (e_eta)
    tie_xi[0] = 0.; tie_eta[0] = -1.
    tie_xi[1] = 0.; tie_eta[1] = 1.
    tie_xi[2] = -1.; tie_eta[2] = 0.
    tie_xi[3] = 1.; tie_eta[3] = 0.
    Cs[0] = A55; Cs[1] = A45; Cs[2] = A45; Cs[3] = A44
    # NOTE covariant shear at each tying point, e_xi = w_xi + x_xi*r_y -
    #      y_xi*r_x for B and D, and e_eta = w_eta + x_eta*r_y - y_eta*r_x
    #      for A and C, from gamma_xz = w_x + r_y and gamma_yz = w_y - r_x
    for k in range(5):
        if k < 4:
            for I in range(4):
                N[I] = 0.25*(1 + XI[I]*tie_xi[k])*(1 + ETA[I]*tie_eta[k])
                Nxi[I] = 0.25*XI[I]*(1 + ETA[I]*tie_eta[k])
                Neta[I] = 0.25*ETA[I]*(1 + XI[I]*tie_xi[k])
        else:
            for I in range(4):
                N[I] = 0.25
                Nxi[I] = 0.25*XI[I]
                Neta[I] = 0.25*ETA[I]
        xxi = 0.; yxi = 0.; xeta = 0.; yeta = 0.
        for I in range(4):
            xxi += Nxi[I]*xs[I]
            yxi += Nxi[I]*ys[I]
            xeta += Neta[I]*xs[I]
            yeta += Neta[I]*ys[I]
        if k < 4:
            for I in range(4):
                if k < 2:
                    e[k][3*I + 0] = Nxi[I]
                    e[k][3*I + 1] = -yxi*N[I]
                    e[k][3*I + 2] = xxi*N[I]
                else:
                    e[k][3*I + 0] = Neta[I]
                    e[k][3*I + 1] = -yeta*N[I]
                    e[k][3*I + 2] = xeta*N[I]
        else:
            # NOTE centroid: both covariant components and the Jacobian
            J0[0] = xxi; J0[1] = yxi; J0[2] = xeta; J0[3] = yeta
            for I in range(4):
                ec[0][3*I + 0] = Nxi[I]
                ec[0][3*I + 1] = -yxi*N[I]
                ec[0][3*I + 2] = xxi*N[I]
                ec[1][3*I + 0] = Neta[I]
                ec[1][3*I + 1] = -yeta*N[I]
                ec[1][3*I + 2] = xeta*N[I]
    detJ0 = J0[0]*J0[3] - J0[1]*J0[2]
    A = 4.*detJ0
    J0inv[0] = J0[3]/detJ0
    J0inv[1] = -J0[1]/detJ0
    J0inv[2] = -J0[2]/detJ0
    J0inv[3] = J0[0]/detJ0
    # NOTE Cartesian strains gamma = J0^-1 [e_xi, e_eta]
    for j in range(12):
        # MITC4 constant part
        G0m[0][j] = J0inv[0]*0.5*(e[0][j] + e[1][j]) + J0inv[1]*0.5*(e[2][j] + e[3][j])
        G0m[1][j] = J0inv[2]*0.5*(e[0][j] + e[1][j]) + J0inv[3]*0.5*(e[2][j] + e[3][j])
        # displacement-based centroid value
        G0d[0][j] = J0inv[0]*ec[0][j] + J0inv[1]*ec[1][j]
        G0d[1][j] = J0inv[2]*ec[0][j] + J0inv[3]*ec[1][j]
        # nonconstant parts, multiplying eta and xi
        D1[0][j] = J0inv[0]*0.5*(e[1][j] - e[0][j])
        D1[1][j] = J0inv[2]*0.5*(e[1][j] - e[0][j])
        D2[0][j] = J0inv[1]*0.5*(e[3][j] - e[2][j])
        D2[1][j] = J0inv[3]*0.5*(e[3][j] - e[2][j])
    for i in range(12):
        for j in range(12):
            K[12*i + j] = (
                A*(G0m[0][i]*(Cs[0]*G0m[0][j] + Cs[1]*G0m[1][j])
                   + G0m[1][i]*(Cs[2]*G0m[0][j] + Cs[3]*G0m[1][j])
                   - G0d[0][i]*(Cs[0]*G0d[0][j] + Cs[1]*G0d[1][j])
                   - G0d[1][i]*(Cs[2]*G0d[0][j] + Cs[3]*G0d[1][j]))
                + fH*A/3.*(D1[0][i]*(Cs[0]*D1[0][j] + Cs[1]*D1[1][j])
                           + D1[1][i]*(Cs[2]*D1[0][j] + Cs[3]*D1[1][j])
                           + D2[0][i]*(Cs[0]*D2[0][j] + Cs[1]*D2[1][j])
                           + D2[1][i]*(Cs[2]*D2[0][j] + Cs[3]*D2[1][j])))


cdef inline void _centroid_slopes(double x1, double x2, double x3, double x4,
                                  double y1, double y2, double y3, double y4,
                                  double *Sx, double *Sy) noexcept nogil:
    r"""Derivatives `S_{i,x}` and `S_{i,y}` of the bilinear shape functions at
    the centroid, `\xi = \eta = 0`, written in ``Sx`` and ``Sy``, 4 values
    each, from the node coordinates in element coordinates

    They are the rows of the slopes `w_{,x}` and `w_{,y}` of the geometrically
    nonlinear terms, see the Notes of :meth:`.Quad4R.update_KG`.

    """
    cdef double J11, J12, J21, J22, detJ
    cdef double j11, j12, j21, j22
    J11 = 0.25*(-x1 + x2 + x3 - x4)
    J12 = 0.25*(-y1 + y2 + y3 - y4)
    J21 = 0.25*(-x1 - x2 + x3 + x4)
    J22 = 0.25*(-y1 - y2 + y3 + y4)
    detJ = J11*J22 - J12*J21
    j11 = J22/detJ
    j12 = -J12/detJ
    j21 = -J21/detJ
    j22 = J11/detJ
    Sx[0] = -0.25*j11 - 0.25*j12
    Sx[1] = 0.25*j11 - 0.25*j12
    Sx[2] = 0.25*j11 + 0.25*j12
    Sx[3] = -0.25*j11 + 0.25*j12
    Sy[0] = -0.25*j21 - 0.25*j22
    Sy[1] = 0.25*j21 - 0.25*j22
    Sy[2] = 0.25*j21 + 0.25*j22
    Sy[3] = -0.25*j21 + 0.25*j22



cdef class Quad4RData:
    r"""
    Used to allocate memory for the sparse matrices.

    Each attribute is the number of entries that one :class:`.Quad4R` writes
    in the row, column and value arrays of the corresponding sparse matrix,
    starting at the position given by the matching ``init_k_*`` attribute of
    the element. For ``n`` elements the arrays need ``n`` times these sizes.
    The constructor takes no parameters.

    Attributes
    ----------
    KC0_SPARSE_SIZE, : int
        ``KC0_SPARSE_SIZE = 576``

    KCNL_SPARSE_SIZE, : int
        ``KCNL_SPARSE_SIZE = 576``

    KG_SPARSE_SIZE, : int
        ``KG_SPARSE_SIZE = 180``: 144 entries of the translations and 36 of
        the rotations of each node, the eccentricity geometric stiffness of
        the rigid offsets of a warped element, see ``update_KG``

    M_SPARSE_SIZE, : int
        ``M_SPARSE_SIZE = 480``

    KA_BETA_SPARSE_SIZE, : int
        ``KA_BETA_SPARSE_SIZE = 144``

    KA_GAMMA_SPARSE_SIZE, : int
        ``KA_GAMMA_SPARSE_SIZE = 144``

    CA_SPARSE_SIZE, : int
        ``CA_SPARSE_SIZE = 144``

    """
    cdef public int KC0_SPARSE_SIZE
    cdef public int KCNL_SPARSE_SIZE
    cdef public int KG_SPARSE_SIZE
    cdef public int M_SPARSE_SIZE
    cdef public int KA_BETA_SPARSE_SIZE
    cdef public int KA_GAMMA_SPARSE_SIZE
    cdef public int CA_SPARSE_SIZE

    def __cinit__(Quad4RData self):
        self.KC0_SPARSE_SIZE = 576
        self.KCNL_SPARSE_SIZE = 576
        self.KG_SPARSE_SIZE = 180
        self.M_SPARSE_SIZE = 480
        self.KA_BETA_SPARSE_SIZE = 144
        self.KA_GAMMA_SPARSE_SIZE = 144
        self.CA_SPARSE_SIZE = 144


cdef class Quad4RProbe:
    r"""
    Probe used for local coordinates, local displacements, local stiffness,
    local stresses etc...

    The idea behind using a probe is to avoid allocating larger memory buffers
    per finite element. The memory buffers are allocated per probe, and one
    probe can be shared amongst many finite elements, with the information
    being updated and retrieved on demand.

    .. note:: The probe can be shared amongst more than one finite element, 
              depending on how you defined them. Mind that the probe will
              always safe the values from the last udpate.


    The constructor takes no parameters and allocates every array filled with
    zeros.

    Attributes
    ----------
    xe, : array-like
        Array of size ``NUM_NODES*DOF//2=12`` containing the nodal coordinates
        in the element coordinate system, in the following order `{x_e}_1,
        {y_e}_1, {z_e}_1, `{x_e}_2, {y_e}_2, {z_e}_2`, `{x_e}_3, {y_e}_3,
        {z_e}_3`, `{x_e}_4, {y_e}_4, {z_e}_4`.
        Dimension of length. Updated by :meth:`.Quad4R.update_probe_xe`.
    ue, : array-like
        Array of size ``NUM_NODES*DOF=24`` containing the element displacements
        in the following order `u_{e_1}, v_{e_1}, w_{e_1}, {r_x}_{e_1},
        {r_y}_{e_1}, {r_z}_{e_1}`, `u_{e_2}, v_{e_2}, w_{e_2}, {r_x}_{e_2},
        {r_y}_{e_2}, {r_z}_{e_2}`, `u_{e_3}, v_{e_3}, w_{e_3}, {r_x}_{e_3},
        {r_y}_{e_3}, {r_z}_{e_3}`, `u_{e_4}, v_{e_4}, w_{e_4}, {r_x}_{e_4},
        {r_y}_{e_4}, {r_z}_{e_4}`.
        Translations with dimension of length, rotations dimensionless.
        Updated by :meth:`.Quad4R.update_probe_ue`.
    finte, : array-like
        Array of size ``NUM_NODES*DOF=24`` containing the element internal
        forces corresponding to the degrees-of-freedom described by ``ue``.
        Forces, and moments (force times length) for the rotations. Updated
        by :meth:`.Quad4R.update_probe_finte`.
    BLexx, BLeyy, BLgxy, BLkxx, BLkyy, BLkxy : array-like
        Arrays of size ``NUM_NODES*DOF=24`` with the rows of the linear
        strain-displacement matrix for the membrane strains and curvatures,
        at the last evaluated integration point.
        Their products with ``ue`` give the membrane strains, dimensionless,
        and the curvatures, with dimension of one over length.
    Sw_x, Sw_y : array-like
        Arrays of size ``NUM_NODES*DOF=24`` with the rows giving `w_{,x}` and
        `w_{,y}`, at the last evaluated integration point.
        ``Sw_x`` holds `\boldsymbol{G}_x = \partial \boldsymbol{S}^w/\partial x`
        and ``Sw_y`` holds `\boldsymbol{G}_y = \partial \boldsymbol{S}^w/\partial
        y`, the underscore denoting the partial derivative.
        Their products with ``ue`` are dimensionless.
    KCNLve : array-like
        Array of size ``(NUM_NODES*DOF)**2=576`` with the nonlinear
        constitutive stiffness matrix KCNL in element coordinates, stored row
        by row.
        Updated by :meth:`.Quad4R.update_KCNL`.

    """
    cdef public double [::1] xe
    cdef public double [::1] ue
    cdef public double [::1] finte
    cdef public double [::1] BLexx
    cdef public double [::1] BLeyy
    cdef public double [::1] BLgxy
    cdef public double [::1] BLkxx
    cdef public double [::1] BLkyy
    cdef public double [::1] BLkxy
    cdef public double [::1] Sw_x
    cdef public double [::1] Sw_y
    cdef public double [::1] KCNLve

    def __cinit__(Quad4RProbe self):
        self.xe = np.zeros(NUM_NODES*DOF//2, dtype=np.float64)
        self.ue = np.zeros(NUM_NODES*DOF, dtype=np.float64)
        self.finte = np.zeros(NUM_NODES*DOF, dtype=np.float64)
        self.BLexx = np.zeros(NUM_NODES*DOF, dtype=np.float64)
        self.BLeyy = np.zeros(NUM_NODES*DOF, dtype=np.float64)
        self.BLgxy = np.zeros(NUM_NODES*DOF, dtype=np.float64)
        self.BLkxx = np.zeros(NUM_NODES*DOF, dtype=np.float64)
        self.BLkyy = np.zeros(NUM_NODES*DOF, dtype=np.float64)
        self.BLkxy = np.zeros(NUM_NODES*DOF, dtype=np.float64)
        self.Sw_x = np.zeros(NUM_NODES*DOF, dtype=np.float64)
        self.Sw_y = np.zeros(NUM_NODES*DOF, dtype=np.float64)
        self.KCNLve = np.zeros((NUM_NODES*DOF)**2, dtype=np.float64)


cdef class Quad4R:
    r"""
    Four-node quadrilateral shell element with reduced integration

    Nodal connectivity for the plate element similar to Nastran's CQUAD4::

         ^ y
         |
        4 ________ 3
         |       |
         |       |   --> x
         |       |
         |_______|
        1         2

    The element coordinate system is determined identically what is explained
    in Nastran's quick reference guide for the CQUAD4 element, as illustrated
    below.

    .. image:: ../figures/nastran_cquad4.svg

    The constructor stores the probe and sets the identification numbers and
    the positions ``c1, c2, c3, c4`` to ``-1``, the ``init_k_*`` positions to
    ``0``, the rotation matrix ``r11, ..., r33`` to zero and ``m11, m12, m21,
    m22`` to the identity. Before computing any element matrix, the caller
    sets ``c1, c2, c3, c4`` and the ``init_k_*`` positions, and calls
    :meth:`.update_rotation_matrix` and :meth:`.update_probe_xe`.

    Parameters
    ----------
    p : :class:`.Quad4RProbe`
        Probe used by the element, stored in ``probe``. One probe can be
        shared by many elements.

    Attributes
    ----------
    eid, : int
        Element identification number.
    pid, : int
        Property identification number.
    area, : double
        Element area.
        Dimension of length squared. Updated by :meth:`.update_area`.
    drilling_model, : int
        Selects the coefficient of the drilling stiffness, see the module
        documentation. The default is ``0``, the physics-based coefficient
        set by ``gamma_rz``; with anything else the fictitious penalty of MSC
        Nastran and Autodesk Nastran controlled by ``K6ROT`` is used instead.
        Switching between them changes only the magnitude of the drilling
        stiffness.

        Note that this element does not implement the in-plane drilling
        enrichment that :class:`.Quad4` and :class:`.Tria3R`
        apply, see the module documentation.
    K6ROT, : double
        Dimensionless multiplier for the fictitious drilling stiffness, read
        only when ``drilling_model`` is not ``0``. Default ``100.``.
    gamma_rz, : double
        Coefficient `\gamma_{r_z}` of the Hughes-Brezzi drilling stiffness,
        read only when ``drilling_model`` is ``0``. The default is a negative
        value, which means that `A_{66}` of the laminate extensional
        stiffness matrix is used.
        A non-negative value is ignored when ``A66`` is zero. Dimension of
        force per unit length, as ``A66``.
    warp_offsets, : int
        Rigid offsets of a warped element, see the Notes. The default ``1``
        links the flat element of the matrices, in the mean plane of the
        nodes, to the warped nodes by rigid offsets along its normal, in
        ``update_KC0``, ``update_M``, ``update_KG``,
        ``update_KG_given_stress``, ``update_KCNL`` and the internal forces
        of ``update_probe_finte`` and ``update_fint``, also with
        ``nonlinear``; ``0`` uses the flat element without them, as before
        version 0.12.0. A flat element gives the same results, bit for bit,
        with both values.
    r11, r12, r13, r21, r22, r23, r31, r32, r33 : double
        Entries `T_{ij}` of the transformation matrix `\boldsymbol{T}_{3\times 3}` from
        element to global coordinates, whose columns are the element axes.
    m11, m12, m21, m22 : double
        Rotation matrix only for the constitutive relations. Used when a
        material direction is used instead of the element local coordinates.
    c1, c2, c3, c4 : int
        Position of each node in the global stiffness matrix.
        The coordinates of a node are read at the positions ``c1//2``,
        ``c2//2``, ``c3//2`` and ``c4//2`` of the coordinate array.
    n1, n2, n3, n4 : int
        Node identification number.
    init_k_KC0, init_k_KCNL, init_k_KG, init_k_M : int
        Position in the arrays storing the sparse data for the structural
        matrices.
    init_k_KA_beta, init_k_KA_gamma, init_k_CA : int
        Position in the arrays storing the sparse data for the aerodynamic
        matrices based on the Piston theory.
    probe, : :class:`.Quad4RProbe` object
        Pointer to the probe.

    Notes
    -----
    The physics-based coefficient of ``drilling_model = 0`` is
    `\gamma_{r_z} = A_{66}` of Hughes and Brezzi (1989) [1]_, and the
    fictitious penalty of the other values of ``drilling_model`` corresponds
    to `\gamma_{r_z} = K6ROT \cdot 10^{-6} A_{66}`. Both apply the same
    operator with the same quadrature, so switching between them changes
    only the magnitude of the drilling stiffness, by four orders of
    magnitude for the recommended ``K6ROT = 100.``. Both models are
    described in the Notes of :mod:`pyfe3d.quad4r`.

    This element does not implement the in-plane enrichment of Allman (1984)
    [2]_ that :class:`.Quad4` and :class:`.Tria3R` apply together
    with the Hughes-Brezzi term, because its reduced integration already
    achieves what the enrichment achieves there.

    For ``K6ROT``, AUTODESK NASTRAN's quick reference guide recommends
    ``K6ROT = 100.`` for static analysis. For modal solutions,
    ``K6ROT=1.e4`` is suggested. MSC NASTRAN's quick reference guide states
    that ``K6ROT > 100.`` should not be used, but this is contradicting
    AUTODESK NASTRAN.

    The default ``gamma_rz``, `A_{66}`, is the value identified by Hughes
    and Brezzi (1989) [1]_.

    Warped elements. The matrices are those of the flat projection of the
    element, whose nodes are in the mean plane of the four nodes, the plane of
    the element coordinates parallel to both diagonals: node `a` is at the
    height `\tilde{z}_a` above it, with `\tilde{z}_1 = \tilde{z}_3 =
    -\tilde{z}_2 = -\tilde{z}_4`, and the coordinate `z` of ``probe.xe`` is
    otherwise ignored. The flat element alone does not represent the rigid
    rotations of the warped nodes and is too stiff in warped meshes. With
    ``warp_offsets = 1`` the flat element is linked to each node by the rigid
    offset `\boldsymbol{r}_a = \tilde{z}_a \boldsymbol{e}_3`, the first order
    of the offsets of :class:`pyfe3d.corotational.CorotationalShell` with
    ``warping='offsets'`` (derived here), which gives, in element coordinates,

    .. math::

        u^f_a = u_a - \tilde{z}_a r_{y,a}, \qquad v^f_a = v_a + \tilde{z}_a
        r_{x,a}

    with `w` and the rotations unchanged, `\boldsymbol{u}^f = \boldsymbol{J}
    \boldsymbol{u}_e`. The element matrices and forces are `\boldsymbol{J}^\top
    \boldsymbol{K}_{C0} \boldsymbol{J}`, `\boldsymbol{J}^\top \boldsymbol{M}
    \boldsymbol{J}` and `\boldsymbol{J}^\top \boldsymbol{f}`, with the flat
    element evaluated at `\boldsymbol{J} \boldsymbol{u}_e`, applied in element
    coordinates before the rotation to global coordinates, and the geometric
    stiffness adds the eccentricity term of the offsets, see
    :meth:`.update_KG`. ``KC0`` then annihilates the six rigid motions of the
    warped nodes, is the tangent of ``CorotationalShell`` at the base
    configuration and keeps the number of zero eigenvalues of the flat element;
    for a warp ratio ``max|z|/sqrt(A)`` of 0.022 its non-zero eigenvalues
    change by at most 0.07 %. Other corrections of warped flat quadrilaterals
    balance the moment of the offset membrane forces with forces normal to the
    element instead of nodal moments, MacNeal (1978) [3]_, p. 180, and Cook
    (1994) [4]_, p. 552, or add warping terms to the curvatures, Belytschko et
    al. (1992) [5]_, Section 3.3, Eqs. (30b) and (37d-f). In the twisted beam
    of MacNeal and Harder (1985) [6]_, Fig. 6, 12 by 2 elements with a warp
    ratio of 0.024, the tip displacements of their Table 3 are matched within 1
    % for the load along the width at the tip and 7 % for the other one with
    the offsets, and are 67 to 77 % too small without them, see
    ``tests/test_quad_warp_offsets.py``. Quad4R bends in its plane stiffly with
    one element across the width, which limits the accuracy of the load along
    the width at the root, 0.40 of the beam solution of the untwisted strip
    with 6 by 1 elements and 0.72 with 12 by 2.

    References
    ----------
    .. [1] Hughes, T. J. R., and Brezzi, F., On drilling degrees of freedom,
       Computer Methods in Applied Mechanics and Engineering, 72(1), 105-121,
       1989. https://doi.org/10.1016/0045-7825(89)90124-2
    .. [2] Allman, D. J., A compatible triangular element including vertex
       rotations for plane elasticity analysis, Computers & Structures,
       19(1-2), 1-8, 1984. https://doi.org/10.1016/0045-7949(84)90197-4
    .. [3] MacNeal, R. H., A simple quadrilateral shell element, Computers
       & Structures, 8(2), 175-183, 1978.
       https://doi.org/10.1016/0045-7949(78)90020-2
    .. [4] Cook, R. D., Four-node 'flat' shell element: drilling degrees of
       freedom, membrane-bending coupling, warped geometry, and behavior,
       Computers & Structures, 50(4), 549-555, 1994.
       https://doi.org/10.1016/0045-7949(94)90025-6
    .. [5] Belytschko, T., Wong, B. L., and Chiang, H.-Y., Advances in
       one-point quadrature shell elements, Computer Methods in Applied
       Mechanics and Engineering, 96(1), 93-107, 1992.
       https://doi.org/10.1016/0045-7825(92)90100-X
    .. [6] MacNeal, R. H., and Harder, R. L., A proposed standard set of
       problems to test finite element accuracy, Finite Elements in Analysis
       and Design, 1(1), 3-20, 1985.
       https://doi.org/10.1016/0168-874X(85)90003-4

    """
    cdef public int eid, pid
    cdef public int n1, n2, n3, n4
    cdef public int c1, c2, c3, c4
    cdef public int init_k_KC0, init_k_KCNL, init_k_KG, init_k_M
    cdef public int init_k_KA_beta, init_k_KA_gamma, init_k_CA
    cdef public double area
    cdef public int drilling_model
    cdef public double K6ROT
    cdef public double gamma_rz
    cdef public int warp_offsets
    cdef public double r11, r12, r13, r21, r22, r23, r31, r32, r33
    cdef public double m11, m12, m21, m22
    cdef public Quad4RProbe probe

    def __cinit__(Quad4R self, Quad4RProbe p):
        self.probe = p
        self.eid = -1
        self.pid = -1
        self.n1 = -1
        self.n2 = -1
        self.n3 = -1
        self.n4 = -1
        self.c1 = -1
        self.c2 = -1
        self.c3 = -1
        self.c4 = -1
        self.init_k_KC0 = 0
        self.init_k_KCNL = 0
        self.init_k_KG = 0
        self.init_k_M = 0
        self.init_k_KA_beta = 0
        self.init_k_KA_gamma = 0
        self.init_k_CA = 0
        self.area = 0
        # NOTE the physics-based coefficient of Hughes and Brezzi (1989),
        #      gamma_rz = A66, is the default, as in Quad4 and Tria3R, so
        #      that K6ROT is not read unless it is asked for. The operator
        #      and its quadrature are the same for both models, only the
        #      coefficient differing, see the module documentation
        self.drilling_model = 0
        self.K6ROT = 100. # NOTE default value in MSC Nastran
        self.gamma_rz = -1. # NOTE negative means "use A66"
        self.warp_offsets = 1 # NOTE rigid offsets of warped elements
        self.r11 = self.r12 = self.r13 = 0.
        self.r21 = self.r22 = self.r23 = 0.
        self.r31 = self.r32 = self.r33 = 0.
        self.m11 = 1.
        self.m12 = 0.
        self.m21 = 0.
        self.m22 = 1.


    cpdef void update_rotation_matrix(Quad4R self, double [::1] x,
            double xmati=0., double xmatj=0., double xmatk=0.):
        r"""Update the rotation matrix of the element

        Attributes ``r11,r12,r13,r21,r22,r23,r31,r32,r33`` are updated, with
        the entries `T_{ij}` of the transformation matrix `\boldsymbol{T}_{3\times 3}`
        from element to global coordinates, whose columns are the element axes.

        The element coordinate system is determined, identifying the `ijk`
        components of each axis: `{x_e}_i, {x_e}_j, {x_e}_k`; `{y_e}_i,
        {y_e}_j, {y_e}_k`; `{z_e}_i, {z_e}_j, {z_e}_k`.

        The element ``z`` axis is along the cross product of the vectors from
        node 4 to node 2 and from node 1 to node 3, and the element ``x`` axis
        along the sum of these two vectors. When a material direction is
        given, ``m11, m12, m21, m22`` are updated with the rotation from the
        material direction projected onto the element to the element ``x``
        axis. When it is the zero vector, the default, or parallel to the
        element normal, ``m11, m12, m21, m22`` are not changed.


        Parameters
        ----------
        x : array-like
            Array with global nodal coordinates, for a total of `M` nodes in
            the model, this array will be arranged as: `x_1, y_1, z_1, x_2,
            y_2, z_2, ..., x_M, y_M, z_M`.
            1-D array of size ``3*M``, dimension of length. The coordinates
            of each node are read from the positions ``c1//2``, ``c2//2``,
            ``c3//2`` and ``c4//2``.

        xmati, xmatj, xmatk : double, optional
            Vector in global coordinates representing the material direction.
            This vector is projected onto the plate element, thus becoming the
            material direction. The `ABD` matrix defining the constitutive
            behavior of the element is rotated from the material direction to
            the element `x` axis while calculating the stiffness matrices.
            Only the direction of the vector is used. Default ``0.``, the
            zero vector, which leaves ``m11, m12, m21, m22`` unchanged.

        """
        cdef double xi, xj, xk, yi, yj, yk, zi, zj, zk
        cdef double x1i, x1j, x1k, x2i, x2j, x2k, x3i, x3j, x3k, x4i, x4j, x4k
        cdef double v13i, v13j, v13k, v42i, v42j, v42k
        cdef double tmp, xmatnorm, ymati, ymatj, ymatk
        cdef double tol

        with nogil:
            x1i = x[self.c1//2 + 0]
            x1j = x[self.c1//2 + 1]
            x1k = x[self.c1//2 + 2]
            x2i = x[self.c2//2 + 0]
            x2j = x[self.c2//2 + 1]
            x2k = x[self.c2//2 + 2]
            x3i = x[self.c3//2 + 0]
            x3j = x[self.c3//2 + 1]
            x3k = x[self.c3//2 + 2]
            x4i = x[self.c4//2 + 0]
            x4j = x[self.c4//2 + 1]
            x4k = x[self.c4//2 + 2]

            v13i = x3i - x1i
            v13j = x3j - x1j
            v13k = x3k - x1k
            v42i = x2i - x4i
            v42j = x2j - x4j
            v42k = x2k - x4k

            zi = v42j*v13k - v42k*v13j
            zj = -v42i*v13k + v42k*v13i
            zk = v42i*v13j - v42j*v13i
            tmp = (zi**2 + zj**2 + zk**2)**0.5
            zi /= tmp
            zj /= tmp
            zk /= tmp
            # NOTE defining tolerance to be 1/1e10 of normal vector norm
            tol = tmp/1e10

            xi = (v13i + v42i)/2.
            xj = (v13j + v42j)/2.
            xk = (v13k + v42k)/2.
            tmp = (xi**2 + xj**2 + xk**2)**0.5
            xi /= tmp
            xj /= tmp
            xk /= tmp

            # y = z X x
            yi = zj*xk - zk*xj
            yj = zk*xi - zi*xk
            yk = zi*xj - zj*xi
            tmp = (yi**2 + yj**2 + yk**2)**0.5
            yi /= tmp
            yj /= tmp
            yk /= tmp

            self.r11 = xi
            self.r21 = xj
            self.r31 = xk
            self.r12 = yi
            self.r22 = yj
            self.r32 = yk
            self.r13 = zi
            self.r23 = zj
            self.r33 = zk

            xmatnorm = (xmati**2 + xmatj**2 + xmatk**2)**0.5
            xmati /= xmatnorm
            xmatj /= xmatnorm
            xmatk /= xmatnorm

            if xmatnorm > tol:
                # Project X Material Vector into Element CSYS
                # ymat = z X xmat
                ymati = zj*xmatk - zk*xmatj
                ymatj = zk*xmati - zi*xmatk
                ymatk = zi*xmatj - zj*xmati
                tmp = (ymati**2 + ymatj**2 + ymatk**2)**0.5
                ymati /= tmp
                ymatj /= tmp
                ymatk /= tmp
                if tmp > tol:
                    # NOTE ovewriting xmati,xmatj,xmatk, they now represent the projected xmat axis
                    # xmat_projected = ymat X z
                    xmati = ymatj*zk - ymatk*zj
                    xmatj = ymatk*zi - ymati*zk
                    xmatk = ymati*zj - ymatj*zi
                    tmp = (xmati**2 + xmatj**2 + xmatk**2)**0.5
                    xmati /= tmp
                    xmatj /= tmp
                    xmatk /= tmp

                    # NOTE angle between xmat_projected and xelem
                    # NOTE assuming they are already normalized (no need to normalize)
                    self.m11 = xmati*xi + xmatj*xj + xmatk*xk # costheta
                    self.m22 = self.m11
                    # NOTE sign of costheta
                    #     - the sign is positive when rotating from the material
                    #       to the element coordinate
                    #     - the sign only affects sintheta
                    # xmat dot_product y
                    if (xmati*yi + xmatj*yj + xmatk*yk) > 0:
                        self.m12 = -(1 - self.m11**2)**0.5 # sintheta
                        self.m21 = -self.m12
                    else:
                        # NOTE theta is negative
                        self.m12 = (1 - self.m11**2)**0.5 # sintheta
                        self.m21 = -self.m12


    cpdef void update_probe_ue(Quad4R self, double [::1] u):
        r"""Update the local displacement vector of the probe of the element

        The displacements and rotations of the four nodes, read from ``u`` at
        the positions ``c1``, ``c2``, ``c3`` and ``c4``, are rotated to the
        element coordinate system with the transpose `\boldsymbol{T}_{3\times 3}^\top` of the
        transformation matrix and written to ``probe.ue``.
        :meth:`.update_rotation_matrix` must have been called before.

        .. note:: The ``probe`` attribute of object :class:`.Quad4RProbe` is
                  updated, accessible using ``.probe.ue``.

        Parameters
        ----------
        u : array-like
            Array with global displacements, for a total of `M` nodes in
            the model, this array will be arranged as: `u_1, v_1, w_1, {r_x}_1,
            {r_y}_1, {r_z}_1, u_2, v_2, w_2, {r_x}_2, {r_y}_2, {r_z}_2, ...,
            u_M, v_M, w_M, {r_x}_M, {r_y}_M, {r_z}_M`.
            1-D array of size ``DOF*M``, translations with dimension of
            length and rotations dimensionless.

        """
        cdef int i, j
        cdef int c[4]
        cdef double s1[3]
        cdef double s2[3]
        cdef double s3[3]

        with nogil:
            # positions in the global stiffness matrix
            c[0] = self.c1
            c[1] = self.c2
            c[2] = self.c3
            c[3] = self.c4

            # global to local transformation of displacements (T.T)
            s1[0] = self.r11
            s1[1] = self.r21
            s1[2] = self.r31
            s2[0] = self.r12
            s2[1] = self.r22
            s2[2] = self.r32
            s3[0] = self.r13
            s3[1] = self.r23
            s3[2] = self.r33

            for j in range(NUM_NODES):
                for i in range(DOF):
                    self.probe.ue[j*DOF + i] = 0

            for j in range(NUM_NODES):
                for i in range(DOF//2):
                    # transforming translations
                    self.probe.ue[j*DOF + 0] += s1[i]*u[c[j] + 0 + i]
                    self.probe.ue[j*DOF + 1] += s2[i]*u[c[j] + 0 + i]
                    self.probe.ue[j*DOF + 2] += s3[i]*u[c[j] + 0 + i]
                    # transforming rotations
                    self.probe.ue[j*DOF + 3] += s1[i]*u[c[j] + 3 + i]
                    self.probe.ue[j*DOF + 4] += s2[i]*u[c[j] + 3 + i]
                    self.probe.ue[j*DOF + 5] += s3[i]*u[c[j] + 3 + i]


    cpdef void update_probe_xe(Quad4R self, double [::1] x):
        r"""Update the 3D coordinates of the probe of the element

        The coordinates of the four nodes, read from ``x`` at the positions
        ``c1//2``, ``c2//2``, ``c3//2`` and ``c4//2``, are rotated to the
        element coordinate system with the transpose `\boldsymbol{T}_{3\times 3}^\top` of the
        transformation matrix and written to ``probe.xe``. :meth:`.update_area`
        is then called, updating ``area``. :meth:`.update_rotation_matrix` must
        have been called before.

        .. note:: The ``xe`` attribute of object :class:`.Quad4RProbe` is
                  updated, accessible using ``.probe.xe``.

        Parameters
        ----------
        x : array-like
            Array with global nodal coordinates, for a total of `M` nodes in
            the model, this array will be arranged as: `x_1, y_1, z_1, x_2,
            y_2, z_2, ..., x_M, y_M, z_M`.
            1-D array of size ``3*M``, dimension of length.

        """
        cdef int i, j
        cdef int c[4]
        cdef double s1[3]
        cdef double s2[3]
        cdef double s3[3]

        with nogil:
            # positions in the global stiffness matrix
            c[0] = self.c1
            c[1] = self.c2
            c[2] = self.c3
            c[3] = self.c4

            # global to local transformation of coordinates (T.T)
            s1[0] = self.r11
            s1[1] = self.r21
            s1[2] = self.r31
            s2[0] = self.r12
            s2[1] = self.r22
            s2[2] = self.r32
            s3[0] = self.r13
            s3[1] = self.r23
            s3[2] = self.r33

            for j in range(NUM_NODES):
                for i in range(DOF//2):
                    self.probe.xe[j*DOF//2 + i] = 0

            for j in range(NUM_NODES):
                for i in range(DOF//2):
                    self.probe.xe[j*DOF//2 + 0] += s1[i]*x[c[j]//2 + i]
                    self.probe.xe[j*DOF//2 + 1] += s2[i]*x[c[j]//2 + i]
                    self.probe.xe[j*DOF//2 + 2] += s3[i]*x[c[j]//2 + i]

        self.update_area()


    cpdef void update_area(Quad4R self):
        r"""Update element area

        The ``area`` attribute is updated with the area of the quadrilateral
        given by the element coordinates ``probe.xe``, ignoring the ``z``
        coordinates. Called by :meth:`.update_probe_xe`.

        """
        cdef double x1, x2, x3, x4, y1, y2, y3, y4
        with nogil:
            # NOTE ignoring z in local coordinates
            x1 = self.probe.xe[0]
            y1 = self.probe.xe[1]
            # z1 = self.probe.xe[2]
            x2 = self.probe.xe[3]
            y2 = self.probe.xe[4]
            # z2 = self.probe.xe[5]
            x3 = self.probe.xe[6]
            y3 = self.probe.xe[7]
            # z3 = self.probe.xe[8]
            x4 = self.probe.xe[9]
            y4 = self.probe.xe[10]
            # z4 = self.probe.xe[11]
            self.area = 1/2.*fabs((x1*y2 + x2*y3 + x3*y4 + x4*y1) - (x2*y1 + x3*y2 + x4*y3 + x1*y4))


    cpdef void update_probe_finte(Quad4R self,
                           ShellProp prop,
                           double hgfactor_membrane = 1.,
                           double hgfactor_physical = 1.,
                           int nonlinear=0
                                  ):
        r"""Update the internal force vector of the probe

        The attribute ``finte`` of the object :class:`.Quad4RProbe` is updated,
        which corresponds to the internal forces in local coordinates. While
        using this function, mind that the probe can be shared amongst more
        than one finite element, depending how you defined them, meaning that
        the probe will always safe the values from the last udpate.

        The internal forces are computed from the probe coordinates ``xe``
        and displacements ``ue``, the ``area`` and the constitutive matrices
        of ``prop`` rotated with ``m11, m12, m21, m22``. Before this method is
        called, :meth:`.update_probe_xe` and :meth:`.update_probe_ue` must
        have been used.

        .. note:: The ``finte`` attribute of object :class:`.Quad4RProbe` is
                  updated, accessible using ``.probe.finte``.

        Parameters
        ----------
        prop : :class:`.ShellProp` object
            Shell property object from where the stiffness and mass attributes
            are read from.
        hgfactor_membrane : double, optional
            Dimensionless multiplier of the artificial hourglass stiffness
            of the in-plane displacements ``u`` and ``v``, of Brockman
            (1987). Default ``1.``.
        hgfactor_physical : double, optional
            Dimensionless multiplier of the physical stabilisation of the
            bending and of the transverse shear of Belytschko and Leviathan
            (1994), which controls the hourglass modes of ``w``, ``r_x`` and
            ``r_y``; the default ``1.`` is the formulation of the paper,
            which has no parameter, and other values serve verification and
            sensitivity studies. See the Notes of :mod:`pyfe3d.quad4r`.
        nonlinear : int, optional
            The default ``0`` gives the linear internal forces, ``KC0*u``. Any other
            value adds the geometrically nonlinear terms of the von Kármán strains,
            whose tangent stiffness matrix is ``KC0 + KCNL + KG``, see
            :meth:`.update_KCNL`.

        Notes
        -----
        The internal forces are the gradient of the strain energy of the Notes
        of :mod:`pyfe3d.quad4r` with respect to the element displacements,
        ``finte = KC0e @ ue`` in the linear case, with the element matrix
        ``KC0e`` of :meth:`.update_KC0`, including its hourglass and drilling
        terms. With ``nonlinear`` the terms of
        :meth:`._update_probe_finte_nonlinear` are added, and ``KC0 + KCNL +
        KG`` is the exact Jacobian of the internal forces, see
        :meth:`.update_KCNL`, when KC0 is computed with the same
        ``hgfactor_*``.

        With ``warp_offsets = 1`` and a warped element the internal forces
        are those of the flat element at the displacements `\boldsymbol{J}
        \boldsymbol{u}_e`, taken to the nodes with `\boldsymbol{J}^\top`,
        see the Notes of :class:`.Quad4R`: the linear ones `\boldsymbol{J}^\top
        \boldsymbol{K}_e \boldsymbol{J} \boldsymbol{u}_e` and the terms of
        ``nonlinear``, see :meth:`.update_KCNL`.

        """
        cdef int warped
        cdef double zw[4]
        cdef double uw[24]
        cdef double *ue
        cdef double *finte
        cdef double x1, x2, x3, x4
        cdef double y1, y2, y3, y4
        cdef int pti, ptj
        cdef double wij, detJ, xi, eta
        cdef double points[2]
        cdef double Ae[9]
        cdef double Be[9]
        cdef double De[9]
        cdef double Atse[4]
        cdef double A44, A45, A55
        # NOTE ABD in the element direction
        cdef double A11, A12, A16, A22, A26, A66
        cdef double B11, B12, B16, B22, B26, B66
        cdef double D11, D12, D16, D22, D26, D66
        cdef double E1eq, E2eq
        cdef double K6ROT
        cdef double Eu, Ev
        cdef double Ds[9]
        cdef double Kws[144]
        cdef double a11, a22
        cdef double J11, J12, J21, J22
        cdef double j11, j12, j21, j22
        cdef double S1_x, S2_x, S3_x, S4_x, S1_y, S2_y, S3_y, S4_y
        cdef double S1, S2, S3, S4
        cdef double S1_xy, S2_xy, S3_xy, S4_xy, gamma1, gamma2, gamma3, gamma4
        cdef double hgx, hgy

        cdef double KC0e0000, KC0e0001, KC0e0003, KC0e0004, KC0e0005, KC0e0006, KC0e0007, KC0e0009, KC0e0010, KC0e0011, KC0e0012, KC0e0013, KC0e0015, KC0e0016, KC0e0017, KC0e0018, KC0e0019, KC0e0021, KC0e0022, KC0e0023
        cdef double KC0e0101, KC0e0103, KC0e0104, KC0e0105, KC0e0106, KC0e0107, KC0e0109, KC0e0110, KC0e0111, KC0e0112, KC0e0113, KC0e0115, KC0e0116, KC0e0117, KC0e0118, KC0e0119, KC0e0121, KC0e0122, KC0e0123
        cdef double KC0e0202, KC0e0203, KC0e0204, KC0e0208, KC0e0209, KC0e0210, KC0e0214, KC0e0215, KC0e0216, KC0e0220, KC0e0221, KC0e0222, KC0e0303
        cdef double KC0e0304, KC0e0306, KC0e0307, KC0e0308, KC0e0309, KC0e0310, KC0e0312, KC0e0313, KC0e0314, KC0e0315, KC0e0316, KC0e0318, KC0e0319, KC0e0320, KC0e0321, KC0e0322
        cdef double KC0e0404, KC0e0406, KC0e0407, KC0e0408, KC0e0409, KC0e0410, KC0e0412, KC0e0413, KC0e0414, KC0e0415, KC0e0416, KC0e0418, KC0e0419, KC0e0420, KC0e0421, KC0e0422
        cdef double KC0e0505, KC0e0506, KC0e0507, KC0e0511, KC0e0512, KC0e0513, KC0e0517, KC0e0518, KC0e0519, KC0e0523
        cdef double KC0e0606, KC0e0607, KC0e0609, KC0e0610, KC0e0611, KC0e0612, KC0e0613, KC0e0615, KC0e0616, KC0e0617, KC0e0618, KC0e0619, KC0e0621, KC0e0622, KC0e0623
        cdef double KC0e0707, KC0e0709, KC0e0710, KC0e0711, KC0e0712, KC0e0713, KC0e0715, KC0e0716, KC0e0717, KC0e0718, KC0e0719, KC0e0721, KC0e0722, KC0e0723
        cdef double KC0e0808, KC0e0809, KC0e0810, KC0e0814, KC0e0815, KC0e0816, KC0e0820, KC0e0821, KC0e0822
        cdef double KC0e0909, KC0e0910, KC0e0912, KC0e0913, KC0e0914, KC0e0915, KC0e0916, KC0e0918, KC0e0919, KC0e0920, KC0e0921, KC0e0922
        cdef double KC0e1010, KC0e1012, KC0e1013, KC0e1014, KC0e1015, KC0e1016, KC0e1018, KC0e1019, KC0e1020, KC0e1021, KC0e1022
        cdef double KC0e1111, KC0e1112, KC0e1113, KC0e1117, KC0e1118, KC0e1119, KC0e1123
        cdef double KC0e1212, KC0e1213, KC0e1215, KC0e1216, KC0e1217, KC0e1218, KC0e1219, KC0e1221, KC0e1222, KC0e1223
        cdef double KC0e1313, KC0e1315, KC0e1316, KC0e1317, KC0e1318, KC0e1319, KC0e1321, KC0e1322, KC0e1323
        cdef double KC0e1414, KC0e1415, KC0e1416, KC0e1420, KC0e1421, KC0e1422
        cdef double KC0e1515, KC0e1516, KC0e1518, KC0e1519, KC0e1520, KC0e1521, KC0e1522
        cdef double KC0e1616, KC0e1618, KC0e1619, KC0e1620, KC0e1621, KC0e1622
        cdef double KC0e1717, KC0e1718, KC0e1719, KC0e1723
        cdef double KC0e1818, KC0e1819, KC0e1821, KC0e1822, KC0e1823
        cdef double KC0e1919, KC0e1921, KC0e1922, KC0e1923
        cdef double KC0e2020, KC0e2021, KC0e2022
        cdef double KC0e2121, KC0e2122
        cdef double KC0e2222
        cdef double KC0e2323

        with nogil:
            ue = &self.probe.ue[0]
            # NOTE rigid offsets: the flat element has the displacements
            #      J ue and its forces are taken to the nodes with J^T
            warped = _warp_z(&self.probe.xe[0], self.warp_offsets, zw)
            if warped:
                _warp_ue(zw, ue, uw)
                ue = uw
            finte = &self.probe.finte[0]

            # NOTE constitutive matrices in the element coordinate system,
            #      the same function is used by all element methods
            prop.get_constitutive_element(self.m11, self.m12, self.m21, self.m22, Ae, Be, De, Atse)
            A11 = Ae[0]
            A12 = Ae[1]
            A16 = Ae[2]
            A22 = Ae[4]
            A26 = Ae[5]
            A66 = Ae[8]
            B11 = Be[0]
            B12 = Be[1]
            B16 = Be[2]
            B22 = Be[4]
            B26 = Be[5]
            B66 = Be[8]
            D11 = De[0]
            D12 = De[1]
            D16 = De[2]
            D22 = De[4]
            D26 = De[5]
            D66 = De[8]
            # NOTE transverse shear stiffness with the shear correction already applied
            A44 = Atse[0]
            A45 = Atse[1]
            A55 = Atse[3]

            # NOTE equivalent moduli of the hourglass control, from the
            #      compliance [a] = [A]^-1 of the extensional stiffness in
            #      the element coordinate system, [B] neglected, see the
            #      Notes of update_KC0
            a11 = (-A22*A66 + A26**2)/(-A11*A22*A66 + A11*A26**2 + A12**2*A66 - 2*A12*A16*A26 + A16**2*A22)
            a22 = (-A11*A66 + A16**2)/(-A11*A22*A66 + A11*A26**2 + A12**2*A66 - 2*A12*A16*A26 + A16**2*A22)
            E1eq = 1./(prop.h*a11)
            E2eq = 1./(prop.h*a22)
            # NOTE bending stiffness with zero membrane forces, of the
            #      physical stabilisation of the bending
            _reduced_bending_stiffness(Ae, Be, De, Ds)


            # NOTE both drilling models use the same operator, the one of
            #      Hughes and Brezzi (1989) integrated with two points per
            #      direction below, and differ only in the coefficient that
            #      multiplies it. The generated block carries the factor
            #      K6ROT*1e-6*A66, so the physics-based value gamma_rz =
            #      A66 is obtained with K6ROT = 1e6, and no generated
            #      expression needs to change. See the module documentation
            if self.drilling_model == 0:
                if self.gamma_rz >= 0. and A66 != 0.:
                    K6ROT = 1.e6*self.gamma_rz/A66
                else:
                    K6ROT = 1.e6
            else:
                K6ROT = self.K6ROT

            # NOTE ignoring z in local coordinates
            x1 = self.probe.xe[0]
            y1 = self.probe.xe[1]
            # z1 = self.probe.xe[2]
            x2 = self.probe.xe[3]
            y2 = self.probe.xe[4]
            # z2 = self.probe.xe[5]
            x3 = self.probe.xe[6]
            y3 = self.probe.xe[7]
            # z3 = self.probe.xe[8]
            x4 = self.probe.xe[9]
            y4 = self.probe.xe[10]
            # z4 = self.probe.xe[11]

            # NOTE reduced integration to remove shear locking
            wij = 4.

            # NOTE Brockman's hourglass control of u and v, derived for
            #      isotropic plates, applied to laminates with the membrane
            #      moduli E1eq and E2eq, and with the area in place of his
            #      1/(1 + 1/A) of Eq. (16), see the section "Hourglass
            #      control of the membrane" of the module documentation: the
            #      term enters as K_ij = A*E^(h)*gamma_i*gamma_j with gamma
            #      of dimension 1/L**2, so E^(h) must be F*L
            Eu = hgfactor_membrane*0.1*E1eq*prop.h*self.area
            Ev = hgfactor_membrane*0.1*E2eq*prop.h*self.area
            # NOTE w, rx and ry have no artificial hourglass stiffness: their
            #      hourglass modes are controlled by the physical
            #      stabilisation of the bending and of the transverse shear
            #      below, scaled by hgfactor_physical, see _shear_bl94 and
            #      _bending_bl94

            # NOTE using only one integration point at xi=0, eta=0 to avoid shear locking
            detJ = 0.125*x1*y2 - 0.125*x1*y4 - 0.125*x2*y1 + 0.125*x2*y3 - 0.125*x3*y2 + 0.125*x3*y4 + 0.125*x4*y1 - 0.125*x4*y3

            j11 = 2.0*(-y1 - y2 + y3 + y4)/(x1*y2 - x1*y4 - x2*y1 + x2*y3 - x3*y2 + x3*y4 + x4*y1 - x4*y3)
            j12 = 2.0*(y1 - y2 - y3 + y4)/(x1*y2 - x1*y4 - x2*y1 + x2*y3 - x3*y2 + x3*y4 + x4*y1 - x4*y3)
            j21 = 2.0*(x1 + x2 - x3 - x4)/(x1*y2 - x1*y4 - x2*y1 + x2*y3 - x3*y2 + x3*y4 + x4*y1 - x4*y3)
            j22 = 2.0*(-x1 + x2 + x3 - x4)/(x1*y2 - x1*y4 - x2*y1 + x2*y3 - x3*y2 + x3*y4 + x4*y1 - x4*y3)

            S1 = 0.25
            S2 = 0.25
            S3 = 0.25
            S4 = 0.25

            S1_x = -0.25*j11 - 0.25*j12
            S2_x = 0.25*j11 - 0.25*j12
            S3_x = 0.25*j11 + 0.25*j12
            S4_x = -0.25*j11 + 0.25*j12
            S1_y = -0.25*j21 - 0.25*j22
            S2_y = 0.25*j21 - 0.25*j22
            S3_y = 0.25*j21 + 0.25*j22
            S4_y = -0.25*j21 + 0.25*j22

            S1_xy = 0.25*j11*j22 + 0.25*j12*j21
            S2_xy = -0.25*j11*j22 - 0.25*j12*j21
            S3_xy = 0.25*j11*j22 + 0.25*j12*j21
            S4_xy = -0.25*j11*j22 - 0.25*j12*j21

            # NOTE hourglass operator of Eq. (14) of Brockman 1987, quoted
            #      there from Belytschko and Tsay 1983,
            #
            #          gamma = h - (h.x)*b1 - (h.y)*b2
            #
            #      with h = [1, -1, 1, -1], b1 and b2 the centroidal
            #      derivatives N,x and N,y, scaled by the normalisation of
            #      Brockman's own Eq. (15), gamma = d2N/dxdy at the centroid,
            #      so that the generalized stiffnesses of his Eq. (16) keep
            #      their calibration. The two correction terms are, in his
            #      words, "important for irregular elements, if the hourglass
            #      strains are to vanish in the presence of rigid-body motion
            #      and uniform strain". They vanish identically whenever the
            #      element is a parallelogram, h.x and h.y being zero there,
            #      so this leaves every regular mesh bit-identical, and they
            #      do not change the hourglass energy of the h patterns on
            #      any shape either, because the centroidal sum h.b1 = h.b2
            #      is zero. See tests/test_quad4r_hourglass_control.py
            hgx = x1 - x2 + x3 - x4
            hgy = y1 - y2 + y3 - y4
            gamma1 = S1_xy*(1. - hgx*S1_x - hgy*S1_y)
            gamma2 = S2_xy*(1. + hgx*S2_x + hgy*S2_y)
            gamma3 = S3_xy*(1. - hgx*S3_x - hgy*S3_y)
            gamma4 = S4_xy*(1. + hgx*S4_x + hgy*S4_y)

            KC0e0000 = detJ*wij*(Eu*gamma1**2 + S1_x*(A11*S1_x + A16*S1_y) + S1_y*(A16*S1_x + A66*S1_y))
            KC0e0001 = detJ*wij*(S1_x*(A16*S1_x + A66*S1_y) + S1_y*(A12*S1_x + A26*S1_y))
            KC0e0003 = -detJ*wij*(S1_x*(B16*S1_x + B66*S1_y) + S1_y*(B12*S1_x + B26*S1_y))
            KC0e0004 = detJ*wij*(S1_x*(B11*S1_x + B16*S1_y) + S1_y*(B16*S1_x + B66*S1_y))
            KC0e0006 = detJ*wij*(Eu*gamma1*gamma2 + S2_x*(A11*S1_x + A16*S1_y) + S2_y*(A16*S1_x + A66*S1_y))
            KC0e0007 = detJ*wij*(S2_x*(A16*S1_x + A66*S1_y) + S2_y*(A12*S1_x + A26*S1_y))
            KC0e0009 = -detJ*wij*(S2_x*(B16*S1_x + B66*S1_y) + S2_y*(B12*S1_x + B26*S1_y))
            KC0e0010 = detJ*wij*(S2_x*(B11*S1_x + B16*S1_y) + S2_y*(B16*S1_x + B66*S1_y))
            KC0e0012 = detJ*wij*(Eu*gamma1*gamma3 + S3_x*(A11*S1_x + A16*S1_y) + S3_y*(A16*S1_x + A66*S1_y))
            KC0e0013 = detJ*wij*(S3_x*(A16*S1_x + A66*S1_y) + S3_y*(A12*S1_x + A26*S1_y))
            KC0e0015 = -detJ*wij*(S3_x*(B16*S1_x + B66*S1_y) + S3_y*(B12*S1_x + B26*S1_y))
            KC0e0016 = detJ*wij*(S3_x*(B11*S1_x + B16*S1_y) + S3_y*(B16*S1_x + B66*S1_y))
            KC0e0018 = detJ*wij*(Eu*gamma1*gamma4 + S4_x*(A11*S1_x + A16*S1_y) + S4_y*(A16*S1_x + A66*S1_y))
            KC0e0019 = detJ*wij*(S4_x*(A16*S1_x + A66*S1_y) + S4_y*(A12*S1_x + A26*S1_y))
            KC0e0021 = -detJ*wij*(S4_x*(B16*S1_x + B66*S1_y) + S4_y*(B12*S1_x + B26*S1_y))
            KC0e0022 = detJ*wij*(S4_x*(B11*S1_x + B16*S1_y) + S4_y*(B16*S1_x + B66*S1_y))
            KC0e0101 = detJ*wij*(Ev*gamma1**2 + S1_x*(A26*S1_y + A66*S1_x) + S1_y*(A22*S1_y + A26*S1_x))
            KC0e0103 = -detJ*wij*(S1_x*(B26*S1_y + B66*S1_x) + S1_y*(B22*S1_y + B26*S1_x))
            KC0e0104 = detJ*wij*(S1_x*(B12*S1_y + B16*S1_x) + S1_y*(B26*S1_y + B66*S1_x))
            KC0e0106 = detJ*wij*(S2_x*(A12*S1_y + A16*S1_x) + S2_y*(A26*S1_y + A66*S1_x))
            KC0e0107 = detJ*wij*(Ev*gamma1*gamma2 + S2_x*(A26*S1_y + A66*S1_x) + S2_y*(A22*S1_y + A26*S1_x))
            KC0e0109 = -detJ*wij*(S2_x*(B26*S1_y + B66*S1_x) + S2_y*(B22*S1_y + B26*S1_x))
            KC0e0110 = detJ*wij*(S2_x*(B12*S1_y + B16*S1_x) + S2_y*(B26*S1_y + B66*S1_x))
            KC0e0112 = detJ*wij*(S3_x*(A12*S1_y + A16*S1_x) + S3_y*(A26*S1_y + A66*S1_x))
            KC0e0113 = detJ*wij*(Ev*gamma1*gamma3 + S3_x*(A26*S1_y + A66*S1_x) + S3_y*(A22*S1_y + A26*S1_x))
            KC0e0115 = -detJ*wij*(S3_x*(B26*S1_y + B66*S1_x) + S3_y*(B22*S1_y + B26*S1_x))
            KC0e0116 = detJ*wij*(S3_x*(B12*S1_y + B16*S1_x) + S3_y*(B26*S1_y + B66*S1_x))
            KC0e0118 = detJ*wij*(S4_x*(A12*S1_y + A16*S1_x) + S4_y*(A26*S1_y + A66*S1_x))
            KC0e0119 = detJ*wij*(Ev*gamma1*gamma4 + S4_x*(A26*S1_y + A66*S1_x) + S4_y*(A22*S1_y + A26*S1_x))
            KC0e0121 = -detJ*wij*(S4_x*(B26*S1_y + B66*S1_x) + S4_y*(B22*S1_y + B26*S1_x))
            KC0e0122 = detJ*wij*(S4_x*(B12*S1_y + B16*S1_x) + S4_y*(B26*S1_y + B66*S1_x))
            KC0e0202 = detJ*wij*(S1_x*(A45*S1_y + A55*S1_x) + S1_y*(A44*S1_y + A45*S1_x))
            KC0e0203 = -S1*detJ*wij*(A44*S1_y + A45*S1_x)
            KC0e0204 = S1*detJ*wij*(A45*S1_y + A55*S1_x)
            KC0e0208 = detJ*wij*(S2_x*(A45*S1_y + A55*S1_x) + S2_y*(A44*S1_y + A45*S1_x))
            KC0e0209 = -S2*detJ*wij*(A44*S1_y + A45*S1_x)
            KC0e0210 = S2*detJ*wij*(A45*S1_y + A55*S1_x)
            KC0e0214 = detJ*wij*(S3_x*(A45*S1_y + A55*S1_x) + S3_y*(A44*S1_y + A45*S1_x))
            KC0e0215 = -S3*detJ*wij*(A44*S1_y + A45*S1_x)
            KC0e0216 = S3*detJ*wij*(A45*S1_y + A55*S1_x)
            KC0e0220 = detJ*wij*(S4_x*(A45*S1_y + A55*S1_x) + S4_y*(A44*S1_y + A45*S1_x))
            KC0e0221 = -S4*detJ*wij*(A44*S1_y + A45*S1_x)
            KC0e0222 = S4*detJ*wij*(A45*S1_y + A55*S1_x)
            KC0e0303 = detJ*wij*(A44*S1**2 + S1_x*(D26*S1_y + D66*S1_x) + S1_y*(D22*S1_y + D26*S1_x))
            KC0e0304 = -detJ*wij*(A45*S1**2 + S1_x*(D12*S1_y + D16*S1_x) + S1_y*(D26*S1_y + D66*S1_x))
            KC0e0306 = -detJ*wij*(S2_x*(B12*S1_y + B16*S1_x) + S2_y*(B26*S1_y + B66*S1_x))
            KC0e0307 = -detJ*wij*(S2_x*(B26*S1_y + B66*S1_x) + S2_y*(B22*S1_y + B26*S1_x))
            KC0e0308 = -S1*detJ*wij*(A44*S2_y + A45*S2_x)
            KC0e0309 = detJ*wij*(A44*S1*S2 + S2_x*(D26*S1_y + D66*S1_x) + S2_y*(D22*S1_y + D26*S1_x))
            KC0e0310 = -detJ*wij*(A45*S1*S2 + S2_x*(D12*S1_y + D16*S1_x) + S2_y*(D26*S1_y + D66*S1_x))
            KC0e0312 = -detJ*wij*(S3_x*(B12*S1_y + B16*S1_x) + S3_y*(B26*S1_y + B66*S1_x))
            KC0e0313 = -detJ*wij*(S3_x*(B26*S1_y + B66*S1_x) + S3_y*(B22*S1_y + B26*S1_x))
            KC0e0314 = -S1*detJ*wij*(A44*S3_y + A45*S3_x)
            KC0e0315 = detJ*wij*(A44*S1*S3 + S3_x*(D26*S1_y + D66*S1_x) + S3_y*(D22*S1_y + D26*S1_x))
            KC0e0316 = -detJ*wij*(A45*S1*S3 + S3_x*(D12*S1_y + D16*S1_x) + S3_y*(D26*S1_y + D66*S1_x))
            KC0e0318 = -detJ*wij*(S4_x*(B12*S1_y + B16*S1_x) + S4_y*(B26*S1_y + B66*S1_x))
            KC0e0319 = -detJ*wij*(S4_x*(B26*S1_y + B66*S1_x) + S4_y*(B22*S1_y + B26*S1_x))
            KC0e0320 = -S1*detJ*wij*(A44*S4_y + A45*S4_x)
            KC0e0321 = detJ*wij*(A44*S1*S4 + S4_x*(D26*S1_y + D66*S1_x) + S4_y*(D22*S1_y + D26*S1_x))
            KC0e0322 = -detJ*wij*(A45*S1*S4 + S4_x*(D12*S1_y + D16*S1_x) + S4_y*(D26*S1_y + D66*S1_x))
            KC0e0404 = detJ*wij*(A55*S1**2 + S1_x*(D11*S1_x + D16*S1_y) + S1_y*(D16*S1_x + D66*S1_y))
            KC0e0406 = detJ*wij*(S2_x*(B11*S1_x + B16*S1_y) + S2_y*(B16*S1_x + B66*S1_y))
            KC0e0407 = detJ*wij*(S2_x*(B16*S1_x + B66*S1_y) + S2_y*(B12*S1_x + B26*S1_y))
            KC0e0408 = S1*detJ*wij*(A45*S2_y + A55*S2_x)
            KC0e0409 = -detJ*wij*(A45*S1*S2 + S2_x*(D16*S1_x + D66*S1_y) + S2_y*(D12*S1_x + D26*S1_y))
            KC0e0410 = detJ*wij*(A55*S1*S2 + S2_x*(D11*S1_x + D16*S1_y) + S2_y*(D16*S1_x + D66*S1_y))
            KC0e0412 = detJ*wij*(S3_x*(B11*S1_x + B16*S1_y) + S3_y*(B16*S1_x + B66*S1_y))
            KC0e0413 = detJ*wij*(S3_x*(B16*S1_x + B66*S1_y) + S3_y*(B12*S1_x + B26*S1_y))
            KC0e0414 = S1*detJ*wij*(A45*S3_y + A55*S3_x)
            KC0e0415 = -detJ*wij*(A45*S1*S3 + S3_x*(D16*S1_x + D66*S1_y) + S3_y*(D12*S1_x + D26*S1_y))
            KC0e0416 = detJ*wij*(A55*S1*S3 + S3_x*(D11*S1_x + D16*S1_y) + S3_y*(D16*S1_x + D66*S1_y))
            KC0e0418 = detJ*wij*(S4_x*(B11*S1_x + B16*S1_y) + S4_y*(B16*S1_x + B66*S1_y))
            KC0e0419 = detJ*wij*(S4_x*(B16*S1_x + B66*S1_y) + S4_y*(B12*S1_x + B26*S1_y))
            KC0e0420 = S1*detJ*wij*(A45*S4_y + A55*S4_x)
            KC0e0421 = -detJ*wij*(A45*S1*S4 + S4_x*(D16*S1_x + D66*S1_y) + S4_y*(D12*S1_x + D26*S1_y))
            KC0e0422 = detJ*wij*(A55*S1*S4 + S4_x*(D11*S1_x + D16*S1_y) + S4_y*(D16*S1_x + D66*S1_y))
            KC0e0606 = detJ*wij*(Eu*gamma2**2 + S2_x*(A11*S2_x + A16*S2_y) + S2_y*(A16*S2_x + A66*S2_y))
            KC0e0607 = detJ*wij*(S2_x*(A16*S2_x + A66*S2_y) + S2_y*(A12*S2_x + A26*S2_y))
            KC0e0609 = -detJ*wij*(S2_x*(B16*S2_x + B66*S2_y) + S2_y*(B12*S2_x + B26*S2_y))
            KC0e0610 = detJ*wij*(S2_x*(B11*S2_x + B16*S2_y) + S2_y*(B16*S2_x + B66*S2_y))
            KC0e0612 = detJ*wij*(Eu*gamma2*gamma3 + S3_x*(A11*S2_x + A16*S2_y) + S3_y*(A16*S2_x + A66*S2_y))
            KC0e0613 = detJ*wij*(S3_x*(A16*S2_x + A66*S2_y) + S3_y*(A12*S2_x + A26*S2_y))
            KC0e0615 = -detJ*wij*(S3_x*(B16*S2_x + B66*S2_y) + S3_y*(B12*S2_x + B26*S2_y))
            KC0e0616 = detJ*wij*(S3_x*(B11*S2_x + B16*S2_y) + S3_y*(B16*S2_x + B66*S2_y))
            KC0e0618 = detJ*wij*(Eu*gamma2*gamma4 + S4_x*(A11*S2_x + A16*S2_y) + S4_y*(A16*S2_x + A66*S2_y))
            KC0e0619 = detJ*wij*(S4_x*(A16*S2_x + A66*S2_y) + S4_y*(A12*S2_x + A26*S2_y))
            KC0e0621 = -detJ*wij*(S4_x*(B16*S2_x + B66*S2_y) + S4_y*(B12*S2_x + B26*S2_y))
            KC0e0622 = detJ*wij*(S4_x*(B11*S2_x + B16*S2_y) + S4_y*(B16*S2_x + B66*S2_y))
            KC0e0707 = detJ*wij*(Ev*gamma2**2 + S2_x*(A26*S2_y + A66*S2_x) + S2_y*(A22*S2_y + A26*S2_x))
            KC0e0709 = -detJ*wij*(S2_x*(B26*S2_y + B66*S2_x) + S2_y*(B22*S2_y + B26*S2_x))
            KC0e0710 = detJ*wij*(S2_x*(B12*S2_y + B16*S2_x) + S2_y*(B26*S2_y + B66*S2_x))
            KC0e0712 = detJ*wij*(S3_x*(A12*S2_y + A16*S2_x) + S3_y*(A26*S2_y + A66*S2_x))
            KC0e0713 = detJ*wij*(Ev*gamma2*gamma3 + S3_x*(A26*S2_y + A66*S2_x) + S3_y*(A22*S2_y + A26*S2_x))
            KC0e0715 = -detJ*wij*(S3_x*(B26*S2_y + B66*S2_x) + S3_y*(B22*S2_y + B26*S2_x))
            KC0e0716 = detJ*wij*(S3_x*(B12*S2_y + B16*S2_x) + S3_y*(B26*S2_y + B66*S2_x))
            KC0e0718 = detJ*wij*(S4_x*(A12*S2_y + A16*S2_x) + S4_y*(A26*S2_y + A66*S2_x))
            KC0e0719 = detJ*wij*(Ev*gamma2*gamma4 + S4_x*(A26*S2_y + A66*S2_x) + S4_y*(A22*S2_y + A26*S2_x))
            KC0e0721 = -detJ*wij*(S4_x*(B26*S2_y + B66*S2_x) + S4_y*(B22*S2_y + B26*S2_x))
            KC0e0722 = detJ*wij*(S4_x*(B12*S2_y + B16*S2_x) + S4_y*(B26*S2_y + B66*S2_x))
            KC0e0808 = detJ*wij*(S2_x*(A45*S2_y + A55*S2_x) + S2_y*(A44*S2_y + A45*S2_x))
            KC0e0809 = -S2*detJ*wij*(A44*S2_y + A45*S2_x)
            KC0e0810 = S2*detJ*wij*(A45*S2_y + A55*S2_x)
            KC0e0814 = detJ*wij*(S3_x*(A45*S2_y + A55*S2_x) + S3_y*(A44*S2_y + A45*S2_x))
            KC0e0815 = -S3*detJ*wij*(A44*S2_y + A45*S2_x)
            KC0e0816 = S3*detJ*wij*(A45*S2_y + A55*S2_x)
            KC0e0820 = detJ*wij*(S4_x*(A45*S2_y + A55*S2_x) + S4_y*(A44*S2_y + A45*S2_x))
            KC0e0821 = -S4*detJ*wij*(A44*S2_y + A45*S2_x)
            KC0e0822 = S4*detJ*wij*(A45*S2_y + A55*S2_x)
            KC0e0909 = detJ*wij*(A44*S2**2 + S2_x*(D26*S2_y + D66*S2_x) + S2_y*(D22*S2_y + D26*S2_x))
            KC0e0910 = -detJ*wij*(A45*S2**2 + S2_x*(D12*S2_y + D16*S2_x) + S2_y*(D26*S2_y + D66*S2_x))
            KC0e0912 = -detJ*wij*(S3_x*(B12*S2_y + B16*S2_x) + S3_y*(B26*S2_y + B66*S2_x))
            KC0e0913 = -detJ*wij*(S3_x*(B26*S2_y + B66*S2_x) + S3_y*(B22*S2_y + B26*S2_x))
            KC0e0914 = -S2*detJ*wij*(A44*S3_y + A45*S3_x)
            KC0e0915 = detJ*wij*(A44*S2*S3 + S3_x*(D26*S2_y + D66*S2_x) + S3_y*(D22*S2_y + D26*S2_x))
            KC0e0916 = -detJ*wij*(A45*S2*S3 + S3_x*(D12*S2_y + D16*S2_x) + S3_y*(D26*S2_y + D66*S2_x))
            KC0e0918 = -detJ*wij*(S4_x*(B12*S2_y + B16*S2_x) + S4_y*(B26*S2_y + B66*S2_x))
            KC0e0919 = -detJ*wij*(S4_x*(B26*S2_y + B66*S2_x) + S4_y*(B22*S2_y + B26*S2_x))
            KC0e0920 = -S2*detJ*wij*(A44*S4_y + A45*S4_x)
            KC0e0921 = detJ*wij*(A44*S2*S4 + S4_x*(D26*S2_y + D66*S2_x) + S4_y*(D22*S2_y + D26*S2_x))
            KC0e0922 = -detJ*wij*(A45*S2*S4 + S4_x*(D12*S2_y + D16*S2_x) + S4_y*(D26*S2_y + D66*S2_x))
            KC0e1010 = detJ*wij*(A55*S2**2 + S2_x*(D11*S2_x + D16*S2_y) + S2_y*(D16*S2_x + D66*S2_y))
            KC0e1012 = detJ*wij*(S3_x*(B11*S2_x + B16*S2_y) + S3_y*(B16*S2_x + B66*S2_y))
            KC0e1013 = detJ*wij*(S3_x*(B16*S2_x + B66*S2_y) + S3_y*(B12*S2_x + B26*S2_y))
            KC0e1014 = S2*detJ*wij*(A45*S3_y + A55*S3_x)
            KC0e1015 = -detJ*wij*(A45*S2*S3 + S3_x*(D16*S2_x + D66*S2_y) + S3_y*(D12*S2_x + D26*S2_y))
            KC0e1016 = detJ*wij*(A55*S2*S3 + S3_x*(D11*S2_x + D16*S2_y) + S3_y*(D16*S2_x + D66*S2_y))
            KC0e1018 = detJ*wij*(S4_x*(B11*S2_x + B16*S2_y) + S4_y*(B16*S2_x + B66*S2_y))
            KC0e1019 = detJ*wij*(S4_x*(B16*S2_x + B66*S2_y) + S4_y*(B12*S2_x + B26*S2_y))
            KC0e1020 = S2*detJ*wij*(A45*S4_y + A55*S4_x)
            KC0e1021 = -detJ*wij*(A45*S2*S4 + S4_x*(D16*S2_x + D66*S2_y) + S4_y*(D12*S2_x + D26*S2_y))
            KC0e1022 = detJ*wij*(A55*S2*S4 + S4_x*(D11*S2_x + D16*S2_y) + S4_y*(D16*S2_x + D66*S2_y))
            KC0e1212 = detJ*wij*(Eu*gamma3**2 + S3_x*(A11*S3_x + A16*S3_y) + S3_y*(A16*S3_x + A66*S3_y))
            KC0e1213 = detJ*wij*(S3_x*(A16*S3_x + A66*S3_y) + S3_y*(A12*S3_x + A26*S3_y))
            KC0e1215 = -detJ*wij*(S3_x*(B16*S3_x + B66*S3_y) + S3_y*(B12*S3_x + B26*S3_y))
            KC0e1216 = detJ*wij*(S3_x*(B11*S3_x + B16*S3_y) + S3_y*(B16*S3_x + B66*S3_y))
            KC0e1218 = detJ*wij*(Eu*gamma3*gamma4 + S4_x*(A11*S3_x + A16*S3_y) + S4_y*(A16*S3_x + A66*S3_y))
            KC0e1219 = detJ*wij*(S4_x*(A16*S3_x + A66*S3_y) + S4_y*(A12*S3_x + A26*S3_y))
            KC0e1221 = -detJ*wij*(S4_x*(B16*S3_x + B66*S3_y) + S4_y*(B12*S3_x + B26*S3_y))
            KC0e1222 = detJ*wij*(S4_x*(B11*S3_x + B16*S3_y) + S4_y*(B16*S3_x + B66*S3_y))
            KC0e1313 = detJ*wij*(Ev*gamma3**2 + S3_x*(A26*S3_y + A66*S3_x) + S3_y*(A22*S3_y + A26*S3_x))
            KC0e1315 = -detJ*wij*(S3_x*(B26*S3_y + B66*S3_x) + S3_y*(B22*S3_y + B26*S3_x))
            KC0e1316 = detJ*wij*(S3_x*(B12*S3_y + B16*S3_x) + S3_y*(B26*S3_y + B66*S3_x))
            KC0e1318 = detJ*wij*(S4_x*(A12*S3_y + A16*S3_x) + S4_y*(A26*S3_y + A66*S3_x))
            KC0e1319 = detJ*wij*(Ev*gamma3*gamma4 + S4_x*(A26*S3_y + A66*S3_x) + S4_y*(A22*S3_y + A26*S3_x))
            KC0e1321 = -detJ*wij*(S4_x*(B26*S3_y + B66*S3_x) + S4_y*(B22*S3_y + B26*S3_x))
            KC0e1322 = detJ*wij*(S4_x*(B12*S3_y + B16*S3_x) + S4_y*(B26*S3_y + B66*S3_x))
            KC0e1414 = detJ*wij*(S3_x*(A45*S3_y + A55*S3_x) + S3_y*(A44*S3_y + A45*S3_x))
            KC0e1415 = -S3*detJ*wij*(A44*S3_y + A45*S3_x)
            KC0e1416 = S3*detJ*wij*(A45*S3_y + A55*S3_x)
            KC0e1420 = detJ*wij*(S4_x*(A45*S3_y + A55*S3_x) + S4_y*(A44*S3_y + A45*S3_x))
            KC0e1421 = -S4*detJ*wij*(A44*S3_y + A45*S3_x)
            KC0e1422 = S4*detJ*wij*(A45*S3_y + A55*S3_x)
            KC0e1515 = detJ*wij*(A44*S3**2 + S3_x*(D26*S3_y + D66*S3_x) + S3_y*(D22*S3_y + D26*S3_x))
            KC0e1516 = -detJ*wij*(A45*S3**2 + S3_x*(D12*S3_y + D16*S3_x) + S3_y*(D26*S3_y + D66*S3_x))
            KC0e1518 = -detJ*wij*(S4_x*(B12*S3_y + B16*S3_x) + S4_y*(B26*S3_y + B66*S3_x))
            KC0e1519 = -detJ*wij*(S4_x*(B26*S3_y + B66*S3_x) + S4_y*(B22*S3_y + B26*S3_x))
            KC0e1520 = -S3*detJ*wij*(A44*S4_y + A45*S4_x)
            KC0e1521 = detJ*wij*(A44*S3*S4 + S4_x*(D26*S3_y + D66*S3_x) + S4_y*(D22*S3_y + D26*S3_x))
            KC0e1522 = -detJ*wij*(A45*S3*S4 + S4_x*(D12*S3_y + D16*S3_x) + S4_y*(D26*S3_y + D66*S3_x))
            KC0e1616 = detJ*wij*(A55*S3**2 + S3_x*(D11*S3_x + D16*S3_y) + S3_y*(D16*S3_x + D66*S3_y))
            KC0e1618 = detJ*wij*(S4_x*(B11*S3_x + B16*S3_y) + S4_y*(B16*S3_x + B66*S3_y))
            KC0e1619 = detJ*wij*(S4_x*(B16*S3_x + B66*S3_y) + S4_y*(B12*S3_x + B26*S3_y))
            KC0e1620 = S3*detJ*wij*(A45*S4_y + A55*S4_x)
            KC0e1621 = -detJ*wij*(A45*S3*S4 + S4_x*(D16*S3_x + D66*S3_y) + S4_y*(D12*S3_x + D26*S3_y))
            KC0e1622 = detJ*wij*(A55*S3*S4 + S4_x*(D11*S3_x + D16*S3_y) + S4_y*(D16*S3_x + D66*S3_y))
            KC0e1818 = detJ*wij*(Eu*gamma4**2 + S4_x*(A11*S4_x + A16*S4_y) + S4_y*(A16*S4_x + A66*S4_y))
            KC0e1819 = detJ*wij*(S4_x*(A16*S4_x + A66*S4_y) + S4_y*(A12*S4_x + A26*S4_y))
            KC0e1821 = -detJ*wij*(S4_x*(B16*S4_x + B66*S4_y) + S4_y*(B12*S4_x + B26*S4_y))
            KC0e1822 = detJ*wij*(S4_x*(B11*S4_x + B16*S4_y) + S4_y*(B16*S4_x + B66*S4_y))
            KC0e1919 = detJ*wij*(Ev*gamma4**2 + S4_x*(A26*S4_y + A66*S4_x) + S4_y*(A22*S4_y + A26*S4_x))
            KC0e1921 = -detJ*wij*(S4_x*(B26*S4_y + B66*S4_x) + S4_y*(B22*S4_y + B26*S4_x))
            KC0e1922 = detJ*wij*(S4_x*(B12*S4_y + B16*S4_x) + S4_y*(B26*S4_y + B66*S4_x))
            KC0e2020 = detJ*wij*(S4_x*(A45*S4_y + A55*S4_x) + S4_y*(A44*S4_y + A45*S4_x))
            KC0e2021 = -S4*detJ*wij*(A44*S4_y + A45*S4_x)
            KC0e2022 = S4*detJ*wij*(A45*S4_y + A55*S4_x)
            KC0e2121 = detJ*wij*(A44*S4**2 + S4_x*(D26*S4_y + D66*S4_x) + S4_y*(D22*S4_y + D26*S4_x))
            KC0e2122 = -detJ*wij*(A45*S4**2 + S4_x*(D12*S4_y + D16*S4_x) + S4_y*(D26*S4_y + D66*S4_x))
            KC0e2222 = detJ*wij*(A55*S4**2 + S4_x*(D11*S4_x + D16*S4_y) + S4_y*(D16*S4_x + D66*S4_y))
            # NOTE transverse shear of Belytschko and Leviathan (1994), the
            #      constant part of the assumed strains of Dvorkin and Bathe
            #      (1984) instead of the displacement-based centroid value, and
            #      the physical stabilisation of their nonconstant part,
            #      multiplied by hgfactor_physical, see the module documentation
            _shear_bl94(x1, x2, x3, x4, y1, y2, y3, y4, A44, A45, A55,
                        hgfactor_physical, Kws)
            # NOTE physical stabilisation of the bending of Belytschko and
            #      Leviathan (1994), Eq. (B.18), multiplied by
            #      hgfactor_physical
            _bending_bl94(x1, x2, x3, x4, y1, y2, y3, y4, Ds,
                          hgfactor_physical, Kws)
            KC0e0202 += Kws[0]
            KC0e0203 += Kws[1]
            KC0e0204 += Kws[2]
            KC0e0208 += Kws[3]
            KC0e0209 += Kws[4]
            KC0e0210 += Kws[5]
            KC0e0214 += Kws[6]
            KC0e0215 += Kws[7]
            KC0e0216 += Kws[8]
            KC0e0220 += Kws[9]
            KC0e0221 += Kws[10]
            KC0e0222 += Kws[11]
            KC0e0303 += Kws[13]
            KC0e0304 += Kws[14]
            KC0e0308 += Kws[15]
            KC0e0309 += Kws[16]
            KC0e0310 += Kws[17]
            KC0e0314 += Kws[18]
            KC0e0315 += Kws[19]
            KC0e0316 += Kws[20]
            KC0e0320 += Kws[21]
            KC0e0321 += Kws[22]
            KC0e0322 += Kws[23]
            KC0e0404 += Kws[26]
            KC0e0408 += Kws[27]
            KC0e0409 += Kws[28]
            KC0e0410 += Kws[29]
            KC0e0414 += Kws[30]
            KC0e0415 += Kws[31]
            KC0e0416 += Kws[32]
            KC0e0420 += Kws[33]
            KC0e0421 += Kws[34]
            KC0e0422 += Kws[35]
            KC0e0808 += Kws[39]
            KC0e0809 += Kws[40]
            KC0e0810 += Kws[41]
            KC0e0814 += Kws[42]
            KC0e0815 += Kws[43]
            KC0e0816 += Kws[44]
            KC0e0820 += Kws[45]
            KC0e0821 += Kws[46]
            KC0e0822 += Kws[47]
            KC0e0909 += Kws[52]
            KC0e0910 += Kws[53]
            KC0e0914 += Kws[54]
            KC0e0915 += Kws[55]
            KC0e0916 += Kws[56]
            KC0e0920 += Kws[57]
            KC0e0921 += Kws[58]
            KC0e0922 += Kws[59]
            KC0e1010 += Kws[65]
            KC0e1014 += Kws[66]
            KC0e1015 += Kws[67]
            KC0e1016 += Kws[68]
            KC0e1020 += Kws[69]
            KC0e1021 += Kws[70]
            KC0e1022 += Kws[71]
            KC0e1414 += Kws[78]
            KC0e1415 += Kws[79]
            KC0e1416 += Kws[80]
            KC0e1420 += Kws[81]
            KC0e1421 += Kws[82]
            KC0e1422 += Kws[83]
            KC0e1515 += Kws[91]
            KC0e1516 += Kws[92]
            KC0e1520 += Kws[93]
            KC0e1521 += Kws[94]
            KC0e1522 += Kws[95]
            KC0e1616 += Kws[104]
            KC0e1620 += Kws[105]
            KC0e1621 += Kws[106]
            KC0e1622 += Kws[107]
            KC0e2020 += Kws[117]
            KC0e2021 += Kws[118]
            KC0e2022 += Kws[119]
            KC0e2121 += Kws[130]
            KC0e2122 += Kws[131]
            KC0e2222 += Kws[143]

            # NOTE drilling rotation uses full integration with two-point Gauss-Legendre quadrature
            wij = 1.
            points[0] = -0.5773502691896257645092
            points[1] = +0.5773502691896257645092

            KC0e0005 = 0
            KC0e0011 = 0
            KC0e0017 = 0
            KC0e0023 = 0
            KC0e0105 = 0
            KC0e0111 = 0
            KC0e0117 = 0
            KC0e0123 = 0
            KC0e0505 = 0
            KC0e0506 = 0
            KC0e0507 = 0
            KC0e0511 = 0
            KC0e0512 = 0
            KC0e0513 = 0
            KC0e0517 = 0
            KC0e0518 = 0
            KC0e0519 = 0
            KC0e0523 = 0
            KC0e0611 = 0
            KC0e0617 = 0
            KC0e0623 = 0
            KC0e0711 = 0
            KC0e0717 = 0
            KC0e0723 = 0
            KC0e1111 = 0
            KC0e1112 = 0
            KC0e1113 = 0
            KC0e1117 = 0
            KC0e1118 = 0
            KC0e1119 = 0
            KC0e1123 = 0
            KC0e1217 = 0
            KC0e1223 = 0
            KC0e1317 = 0
            KC0e1323 = 0
            KC0e1717 = 0
            KC0e1718 = 0
            KC0e1719 = 0
            KC0e1723 = 0
            KC0e1823 = 0
            KC0e1923 = 0
            KC0e2323 = 0

            for pti in range(2):
                xi = points[pti]
                for ptj in range(2):
                    eta = points[ptj]

                    J11 = -0.5*x1 + 0.5*x2 + 0.5*(eta + 1)*(0.5*x1 - 0.5*x2 + 0.5*x3 - 0.5*x4)
                    J12 = -0.5*y1 + 0.5*y2 + 0.5*(eta + 1)*(0.5*y1 - 0.5*y2 + 0.5*y3 - 0.5*y4)
                    J21 = -0.5*x1 + 0.5*x4 - 0.25*(-x1 + x2)*(xi + 1) + 0.25*(x3 - x4)*(xi + 1)
                    J22 = -0.5*y1 + 0.5*y4 - 0.25*(xi + 1)*(-y1 + y2) + 0.25*(xi + 1)*(y3 - y4)

                    detJ = J11*J22 - J12*J21

                    j11 = J22/(J11*J22 - J12*J21)
                    j12 = -J12/(J11*J22 - J12*J21)
                    j21 = -J21/(J11*J22 - J12*J21)
                    j22 = J11/(J11*J22 - J12*J21)

                    S1 = eta*xi/4. - eta/4. - xi/4. + 1/4.
                    S2 = -eta*xi/4. - eta/4. + xi/4. + 1/4.
                    S3 = eta*xi/4. + eta/4. + xi/4. + 1/4.
                    S4 = -eta*xi/4. + eta/4. - xi/4. + 1/4.

                    S1_x = 0.25*j11*(eta - 1) + 0.25*j12*(xi - 1)
                    S2_x = -0.25*eta*j11 + 0.25*j11 - 0.25*j12*xi - 0.25*j12
                    S3_x = 0.25*j11*(eta + 1) + 0.25*j12*(xi + 1)
                    S4_x = -0.25*eta*j11 - 0.25*j11 - 0.25*j12*xi + 0.25*j12

                    S1_y = 0.25*j21*(eta - 1) + 0.25*j22*(xi - 1)
                    S2_y = -0.25*eta*j21 + 0.25*j21 - 0.25*j22*xi - 0.25*j22
                    S3_y = 0.25*j21*(eta + 1) + 0.25*j22*(xi + 1)
                    S4_y = -0.25*eta*j21 - 0.25*j21 - 0.25*j22*xi + 0.25*j22

                    KC0e0000 += 2.5e-7*A66*K6ROT*S1_y**2*detJ*wij
                    KC0e0001 += -2.5e-7*A66*K6ROT*S1_x*S1_y*detJ*wij
                    KC0e0005 += 5.0e-7*A66*K6ROT*S1*S1_y*detJ*wij
                    KC0e0006 += 2.5e-7*A66*K6ROT*S1_y*S2_y*detJ*wij
                    KC0e0007 += -2.5e-7*A66*K6ROT*S1_y*S2_x*detJ*wij
                    KC0e0011 += 5.0e-7*A66*K6ROT*S1_y*S2*detJ*wij
                    KC0e0012 += 2.5e-7*A66*K6ROT*S1_y*S3_y*detJ*wij
                    KC0e0013 += -2.5e-7*A66*K6ROT*S1_y*S3_x*detJ*wij
                    KC0e0017 += 5.0e-7*A66*K6ROT*S1_y*S3*detJ*wij
                    KC0e0018 += 2.5e-7*A66*K6ROT*S1_y*S4_y*detJ*wij
                    KC0e0019 += -2.5e-7*A66*K6ROT*S1_y*S4_x*detJ*wij
                    KC0e0023 += 5.0e-7*A66*K6ROT*S1_y*S4*detJ*wij
                    KC0e0101 += 2.5e-7*A66*K6ROT*S1_x**2*detJ*wij
                    KC0e0105 += -5.0e-7*A66*K6ROT*S1*S1_x*detJ*wij
                    KC0e0106 += -2.5e-7*A66*K6ROT*S1_x*S2_y*detJ*wij
                    KC0e0107 += 2.5e-7*A66*K6ROT*S1_x*S2_x*detJ*wij
                    KC0e0111 += -5.0e-7*A66*K6ROT*S1_x*S2*detJ*wij
                    KC0e0112 += -2.5e-7*A66*K6ROT*S1_x*S3_y*detJ*wij
                    KC0e0113 += 2.5e-7*A66*K6ROT*S1_x*S3_x*detJ*wij
                    KC0e0117 += -5.0e-7*A66*K6ROT*S1_x*S3*detJ*wij
                    KC0e0118 += -2.5e-7*A66*K6ROT*S1_x*S4_y*detJ*wij
                    KC0e0119 += 2.5e-7*A66*K6ROT*S1_x*S4_x*detJ*wij
                    KC0e0123 += -5.0e-7*A66*K6ROT*S1_x*S4*detJ*wij
                    KC0e0505 += 1.0e-6*A66*K6ROT*S1**2*detJ*wij
                    KC0e0506 += 5.0e-7*A66*K6ROT*S1*S2_y*detJ*wij
                    KC0e0507 += -5.0e-7*A66*K6ROT*S1*S2_x*detJ*wij
                    KC0e0511 += 1.0e-6*A66*K6ROT*S1*S2*detJ*wij
                    KC0e0512 += 5.0e-7*A66*K6ROT*S1*S3_y*detJ*wij
                    KC0e0513 += -5.0e-7*A66*K6ROT*S1*S3_x*detJ*wij
                    KC0e0517 += 1.0e-6*A66*K6ROT*S1*S3*detJ*wij
                    KC0e0518 += 5.0e-7*A66*K6ROT*S1*S4_y*detJ*wij
                    KC0e0519 += -5.0e-7*A66*K6ROT*S1*S4_x*detJ*wij
                    KC0e0523 += 1.0e-6*A66*K6ROT*S1*S4*detJ*wij
                    KC0e0606 += 2.5e-7*A66*K6ROT*S2_y**2*detJ*wij
                    KC0e0607 += -2.5e-7*A66*K6ROT*S2_x*S2_y*detJ*wij
                    KC0e0611 += 5.0e-7*A66*K6ROT*S2*S2_y*detJ*wij
                    KC0e0612 += 2.5e-7*A66*K6ROT*S2_y*S3_y*detJ*wij
                    KC0e0613 += -2.5e-7*A66*K6ROT*S2_y*S3_x*detJ*wij
                    KC0e0617 += 5.0e-7*A66*K6ROT*S2_y*S3*detJ*wij
                    KC0e0618 += 2.5e-7*A66*K6ROT*S2_y*S4_y*detJ*wij
                    KC0e0619 += -2.5e-7*A66*K6ROT*S2_y*S4_x*detJ*wij
                    KC0e0623 += 5.0e-7*A66*K6ROT*S2_y*S4*detJ*wij
                    KC0e0707 += 2.5e-7*A66*K6ROT*S2_x**2*detJ*wij
                    KC0e0711 += -5.0e-7*A66*K6ROT*S2*S2_x*detJ*wij
                    KC0e0712 += -2.5e-7*A66*K6ROT*S2_x*S3_y*detJ*wij
                    KC0e0713 += 2.5e-7*A66*K6ROT*S2_x*S3_x*detJ*wij
                    KC0e0717 += -5.0e-7*A66*K6ROT*S2_x*S3*detJ*wij
                    KC0e0718 += -2.5e-7*A66*K6ROT*S2_x*S4_y*detJ*wij
                    KC0e0719 += 2.5e-7*A66*K6ROT*S2_x*S4_x*detJ*wij
                    KC0e0723 += -5.0e-7*A66*K6ROT*S2_x*S4*detJ*wij
                    KC0e1111 += 1.0e-6*A66*K6ROT*S2**2*detJ*wij
                    KC0e1112 += 5.0e-7*A66*K6ROT*S2*S3_y*detJ*wij
                    KC0e1113 += -5.0e-7*A66*K6ROT*S2*S3_x*detJ*wij
                    KC0e1117 += 1.0e-6*A66*K6ROT*S2*S3*detJ*wij
                    KC0e1118 += 5.0e-7*A66*K6ROT*S2*S4_y*detJ*wij
                    KC0e1119 += -5.0e-7*A66*K6ROT*S2*S4_x*detJ*wij
                    KC0e1123 += 1.0e-6*A66*K6ROT*S2*S4*detJ*wij
                    KC0e1212 += 2.5e-7*A66*K6ROT*S3_y**2*detJ*wij
                    KC0e1213 += -2.5e-7*A66*K6ROT*S3_x*S3_y*detJ*wij
                    KC0e1217 += 5.0e-7*A66*K6ROT*S3*S3_y*detJ*wij
                    KC0e1218 += 2.5e-7*A66*K6ROT*S3_y*S4_y*detJ*wij
                    KC0e1219 += -2.5e-7*A66*K6ROT*S3_y*S4_x*detJ*wij
                    KC0e1223 += 5.0e-7*A66*K6ROT*S3_y*S4*detJ*wij
                    KC0e1313 += 2.5e-7*A66*K6ROT*S3_x**2*detJ*wij
                    KC0e1317 += -5.0e-7*A66*K6ROT*S3*S3_x*detJ*wij
                    KC0e1318 += -2.5e-7*A66*K6ROT*S3_x*S4_y*detJ*wij
                    KC0e1319 += 2.5e-7*A66*K6ROT*S3_x*S4_x*detJ*wij
                    KC0e1323 += -5.0e-7*A66*K6ROT*S3_x*S4*detJ*wij
                    KC0e1717 += 1.0e-6*A66*K6ROT*S3**2*detJ*wij
                    KC0e1718 += 5.0e-7*A66*K6ROT*S3*S4_y*detJ*wij
                    KC0e1719 += -5.0e-7*A66*K6ROT*S3*S4_x*detJ*wij
                    KC0e1723 += 1.0e-6*A66*K6ROT*S3*S4*detJ*wij
                    KC0e1818 += 2.5e-7*A66*K6ROT*S4_y**2*detJ*wij
                    KC0e1819 += -2.5e-7*A66*K6ROT*S4_x*S4_y*detJ*wij
                    KC0e1823 += 5.0e-7*A66*K6ROT*S4*S4_y*detJ*wij
                    KC0e1919 += 2.5e-7*A66*K6ROT*S4_x**2*detJ*wij
                    KC0e1923 += -5.0e-7*A66*K6ROT*S4*S4_x*detJ*wij
                    KC0e2323 += 1.0e-6*A66*K6ROT*S4**2*detJ*wij

            finte[0] = KC0e0000*ue[0] + KC0e0001*ue[1] + KC0e0003*ue[3] + KC0e0004*ue[4] + KC0e0005*ue[5] + KC0e0006*ue[6] + KC0e0007*ue[7] + KC0e0009*ue[9] + KC0e0010*ue[10] + KC0e0011*ue[11] + KC0e0012*ue[12] + KC0e0013*ue[13] + KC0e0015*ue[15] + KC0e0016*ue[16] + KC0e0017*ue[17] + KC0e0018*ue[18] + KC0e0019*ue[19] + KC0e0021*ue[21] + KC0e0022*ue[22] + KC0e0023*ue[23]
            finte[1] = KC0e0001*ue[0] + KC0e0101*ue[1] + KC0e0103*ue[3] + KC0e0104*ue[4] + KC0e0105*ue[5] + KC0e0106*ue[6] + KC0e0107*ue[7] + KC0e0109*ue[9] + KC0e0110*ue[10] + KC0e0111*ue[11] + KC0e0112*ue[12] + KC0e0113*ue[13] + KC0e0115*ue[15] + KC0e0116*ue[16] + KC0e0117*ue[17] + KC0e0118*ue[18] + KC0e0119*ue[19] + KC0e0121*ue[21] + KC0e0122*ue[22] + KC0e0123*ue[23]
            finte[2] = KC0e0202*ue[2] + KC0e0203*ue[3] + KC0e0204*ue[4] + KC0e0208*ue[8] + KC0e0209*ue[9] + KC0e0210*ue[10] + KC0e0214*ue[14] + KC0e0215*ue[15] + KC0e0216*ue[16] + KC0e0220*ue[20] + KC0e0221*ue[21] + KC0e0222*ue[22]
            finte[3] = KC0e0003*ue[0] + KC0e0103*ue[1] + KC0e0203*ue[2] + KC0e0303*ue[3] + KC0e0304*ue[4] + KC0e0306*ue[6] + KC0e0307*ue[7] + KC0e0308*ue[8] + KC0e0309*ue[9] + KC0e0310*ue[10] + KC0e0312*ue[12] + KC0e0313*ue[13] + KC0e0314*ue[14] + KC0e0315*ue[15] + KC0e0316*ue[16] + KC0e0318*ue[18] + KC0e0319*ue[19] + KC0e0320*ue[20] + KC0e0321*ue[21] + KC0e0322*ue[22]
            finte[4] = KC0e0004*ue[0] + KC0e0104*ue[1] + KC0e0204*ue[2] + KC0e0304*ue[3] + KC0e0404*ue[4] + KC0e0406*ue[6] + KC0e0407*ue[7] + KC0e0408*ue[8] + KC0e0409*ue[9] + KC0e0410*ue[10] + KC0e0412*ue[12] + KC0e0413*ue[13] + KC0e0414*ue[14] + KC0e0415*ue[15] + KC0e0416*ue[16] + KC0e0418*ue[18] + KC0e0419*ue[19] + KC0e0420*ue[20] + KC0e0421*ue[21] + KC0e0422*ue[22]
            finte[5] = KC0e0005*ue[0] + KC0e0105*ue[1] + KC0e0505*ue[5] + KC0e0506*ue[6] + KC0e0507*ue[7] + KC0e0511*ue[11] + KC0e0512*ue[12] + KC0e0513*ue[13] + KC0e0517*ue[17] + KC0e0518*ue[18] + KC0e0519*ue[19] + KC0e0523*ue[23]
            finte[6] = KC0e0006*ue[0] + KC0e0106*ue[1] + KC0e0306*ue[3] + KC0e0406*ue[4] + KC0e0506*ue[5] + KC0e0606*ue[6] + KC0e0607*ue[7] + KC0e0609*ue[9] + KC0e0610*ue[10] + KC0e0611*ue[11] + KC0e0612*ue[12] + KC0e0613*ue[13] + KC0e0615*ue[15] + KC0e0616*ue[16] + KC0e0617*ue[17] + KC0e0618*ue[18] + KC0e0619*ue[19] + KC0e0621*ue[21] + KC0e0622*ue[22] + KC0e0623*ue[23]
            finte[7] = KC0e0007*ue[0] + KC0e0107*ue[1] + KC0e0307*ue[3] + KC0e0407*ue[4] + KC0e0507*ue[5] + KC0e0607*ue[6] + KC0e0707*ue[7] + KC0e0709*ue[9] + KC0e0710*ue[10] + KC0e0711*ue[11] + KC0e0712*ue[12] + KC0e0713*ue[13] + KC0e0715*ue[15] + KC0e0716*ue[16] + KC0e0717*ue[17] + KC0e0718*ue[18] + KC0e0719*ue[19] + KC0e0721*ue[21] + KC0e0722*ue[22] + KC0e0723*ue[23]
            finte[8] = KC0e0208*ue[2] + KC0e0308*ue[3] + KC0e0408*ue[4] + KC0e0808*ue[8] + KC0e0809*ue[9] + KC0e0810*ue[10] + KC0e0814*ue[14] + KC0e0815*ue[15] + KC0e0816*ue[16] + KC0e0820*ue[20] + KC0e0821*ue[21] + KC0e0822*ue[22]
            finte[9] = KC0e0009*ue[0] + KC0e0109*ue[1] + KC0e0209*ue[2] + KC0e0309*ue[3] + KC0e0409*ue[4] + KC0e0609*ue[6] + KC0e0709*ue[7] + KC0e0809*ue[8] + KC0e0909*ue[9] + KC0e0910*ue[10] + KC0e0912*ue[12] + KC0e0913*ue[13] + KC0e0914*ue[14] + KC0e0915*ue[15] + KC0e0916*ue[16] + KC0e0918*ue[18] + KC0e0919*ue[19] + KC0e0920*ue[20] + KC0e0921*ue[21] + KC0e0922*ue[22]
            finte[10] = KC0e0010*ue[0] + KC0e0110*ue[1] + KC0e0210*ue[2] + KC0e0310*ue[3] + KC0e0410*ue[4] + KC0e0610*ue[6] + KC0e0710*ue[7] + KC0e0810*ue[8] + KC0e0910*ue[9] + KC0e1010*ue[10] + KC0e1012*ue[12] + KC0e1013*ue[13] + KC0e1014*ue[14] + KC0e1015*ue[15] + KC0e1016*ue[16] + KC0e1018*ue[18] + KC0e1019*ue[19] + KC0e1020*ue[20] + KC0e1021*ue[21] + KC0e1022*ue[22]
            finte[11] = KC0e0011*ue[0] + KC0e0111*ue[1] + KC0e0511*ue[5] + KC0e0611*ue[6] + KC0e0711*ue[7] + KC0e1111*ue[11] + KC0e1112*ue[12] + KC0e1113*ue[13] + KC0e1117*ue[17] + KC0e1118*ue[18] + KC0e1119*ue[19] + KC0e1123*ue[23]
            finte[12] = KC0e0012*ue[0] + KC0e0112*ue[1] + KC0e0312*ue[3] + KC0e0412*ue[4] + KC0e0512*ue[5] + KC0e0612*ue[6] + KC0e0712*ue[7] + KC0e0912*ue[9] + KC0e1012*ue[10] + KC0e1112*ue[11] + KC0e1212*ue[12] + KC0e1213*ue[13] + KC0e1215*ue[15] + KC0e1216*ue[16] + KC0e1217*ue[17] + KC0e1218*ue[18] + KC0e1219*ue[19] + KC0e1221*ue[21] + KC0e1222*ue[22] + KC0e1223*ue[23]
            finte[13] = KC0e0013*ue[0] + KC0e0113*ue[1] + KC0e0313*ue[3] + KC0e0413*ue[4] + KC0e0513*ue[5] + KC0e0613*ue[6] + KC0e0713*ue[7] + KC0e0913*ue[9] + KC0e1013*ue[10] + KC0e1113*ue[11] + KC0e1213*ue[12] + KC0e1313*ue[13] + KC0e1315*ue[15] + KC0e1316*ue[16] + KC0e1317*ue[17] + KC0e1318*ue[18] + KC0e1319*ue[19] + KC0e1321*ue[21] + KC0e1322*ue[22] + KC0e1323*ue[23]
            finte[14] = KC0e0214*ue[2] + KC0e0314*ue[3] + KC0e0414*ue[4] + KC0e0814*ue[8] + KC0e0914*ue[9] + KC0e1014*ue[10] + KC0e1414*ue[14] + KC0e1415*ue[15] + KC0e1416*ue[16] + KC0e1420*ue[20] + KC0e1421*ue[21] + KC0e1422*ue[22]
            finte[15] = KC0e0015*ue[0] + KC0e0115*ue[1] + KC0e0215*ue[2] + KC0e0315*ue[3] + KC0e0415*ue[4] + KC0e0615*ue[6] + KC0e0715*ue[7] + KC0e0815*ue[8] + KC0e0915*ue[9] + KC0e1015*ue[10] + KC0e1215*ue[12] + KC0e1315*ue[13] + KC0e1415*ue[14] + KC0e1515*ue[15] + KC0e1516*ue[16] + KC0e1518*ue[18] + KC0e1519*ue[19] + KC0e1520*ue[20] + KC0e1521*ue[21] + KC0e1522*ue[22]
            finte[16] = KC0e0016*ue[0] + KC0e0116*ue[1] + KC0e0216*ue[2] + KC0e0316*ue[3] + KC0e0416*ue[4] + KC0e0616*ue[6] + KC0e0716*ue[7] + KC0e0816*ue[8] + KC0e0916*ue[9] + KC0e1016*ue[10] + KC0e1216*ue[12] + KC0e1316*ue[13] + KC0e1416*ue[14] + KC0e1516*ue[15] + KC0e1616*ue[16] + KC0e1618*ue[18] + KC0e1619*ue[19] + KC0e1620*ue[20] + KC0e1621*ue[21] + KC0e1622*ue[22]
            finte[17] = KC0e0017*ue[0] + KC0e0117*ue[1] + KC0e0517*ue[5] + KC0e0617*ue[6] + KC0e0717*ue[7] + KC0e1117*ue[11] + KC0e1217*ue[12] + KC0e1317*ue[13] + KC0e1717*ue[17] + KC0e1718*ue[18] + KC0e1719*ue[19] + KC0e1723*ue[23]
            finte[18] = KC0e0018*ue[0] + KC0e0118*ue[1] + KC0e0318*ue[3] + KC0e0418*ue[4] + KC0e0518*ue[5] + KC0e0618*ue[6] + KC0e0718*ue[7] + KC0e0918*ue[9] + KC0e1018*ue[10] + KC0e1118*ue[11] + KC0e1218*ue[12] + KC0e1318*ue[13] + KC0e1518*ue[15] + KC0e1618*ue[16] + KC0e1718*ue[17] + KC0e1818*ue[18] + KC0e1819*ue[19] + KC0e1821*ue[21] + KC0e1822*ue[22] + KC0e1823*ue[23]
            finte[19] = KC0e0019*ue[0] + KC0e0119*ue[1] + KC0e0319*ue[3] + KC0e0419*ue[4] + KC0e0519*ue[5] + KC0e0619*ue[6] + KC0e0719*ue[7] + KC0e0919*ue[9] + KC0e1019*ue[10] + KC0e1119*ue[11] + KC0e1219*ue[12] + KC0e1319*ue[13] + KC0e1519*ue[15] + KC0e1619*ue[16] + KC0e1719*ue[17] + KC0e1819*ue[18] + KC0e1919*ue[19] + KC0e1921*ue[21] + KC0e1922*ue[22] + KC0e1923*ue[23]
            finte[20] = KC0e0220*ue[2] + KC0e0320*ue[3] + KC0e0420*ue[4] + KC0e0820*ue[8] + KC0e0920*ue[9] + KC0e1020*ue[10] + KC0e1420*ue[14] + KC0e1520*ue[15] + KC0e1620*ue[16] + KC0e2020*ue[20] + KC0e2021*ue[21] + KC0e2022*ue[22]
            finte[21] = KC0e0021*ue[0] + KC0e0121*ue[1] + KC0e0221*ue[2] + KC0e0321*ue[3] + KC0e0421*ue[4] + KC0e0621*ue[6] + KC0e0721*ue[7] + KC0e0821*ue[8] + KC0e0921*ue[9] + KC0e1021*ue[10] + KC0e1221*ue[12] + KC0e1321*ue[13] + KC0e1421*ue[14] + KC0e1521*ue[15] + KC0e1621*ue[16] + KC0e1821*ue[18] + KC0e1921*ue[19] + KC0e2021*ue[20] + KC0e2121*ue[21] + KC0e2122*ue[22]
            finte[22] = KC0e0022*ue[0] + KC0e0122*ue[1] + KC0e0222*ue[2] + KC0e0322*ue[3] + KC0e0422*ue[4] + KC0e0622*ue[6] + KC0e0722*ue[7] + KC0e0822*ue[8] + KC0e0922*ue[9] + KC0e1022*ue[10] + KC0e1222*ue[12] + KC0e1322*ue[13] + KC0e1422*ue[14] + KC0e1522*ue[15] + KC0e1622*ue[16] + KC0e1822*ue[18] + KC0e1922*ue[19] + KC0e2022*ue[20] + KC0e2122*ue[21] + KC0e2222*ue[22]
            finte[23] = KC0e0023*ue[0] + KC0e0123*ue[1] + KC0e0523*ue[5] + KC0e0623*ue[6] + KC0e0723*ue[7] + KC0e1123*ue[11] + KC0e1223*ue[12] + KC0e1323*ue[13] + KC0e1723*ue[17] + KC0e1823*ue[18] + KC0e1923*ue[19] + KC0e2323*ue[23]
            if warped:
                _warp_f(zw, finte)

            if nonlinear:
                self._update_probe_finte_nonlinear(prop)


    cpdef void update_KC0(Quad4R self,
                          long [::1] KC0r,
                          long [::1] KC0c,
                          double [::1] KC0v,
                          ShellProp prop,
                          int update_KC0v_only=0,
                          double hgfactor_membrane = 1.,
                          double hgfactor_physical = 1.,
                          ):
        r"""Update sparse vectors for linear constitutive stiffness matrix KC0

        The element matrix is computed from the coordinates ``probe.xe``, the
        ``area`` and the constitutive matrices of ``prop`` rotated with
        ``m11, m12, m21, m22``, with the hourglass stiffnesses scaled by the
        ``hgfactor_*`` parameters. Its values, in global coordinates, are
        added to ``KC0v`` at the positions ``init_k_KC0`` to ``init_k_KC0 +
        KC0_SPARSE_SIZE - 1``. With ``update_KC0v_only=0`` the global row and
        column indices are written to ``KC0r`` and ``KC0c`` at the same
        positions. Before this method is called,
        :meth:`.update_rotation_matrix` and :meth:`.update_probe_xe` must have
        been used.


        Parameters
        ----------
        KC0r : np.array
            Array to store row positions of sparse values
            1-D array of integers (``INT``) of size at least ``init_k_KC0 +
            KC0_SPARSE_SIZE``.
        KC0c : np.array
            Array to store column positions of sparse values
            Same type and size as ``KC0r``.
        KC0v : np.array
            Array to store sparse values
            1-D array of ``DOUBLE``, same size as ``KC0r``. The values are
            added to the existing ones. Dimension of force per unit length
            between translations, of force between a translation and a
            rotation, and of force times length between rotations.
        prop : :class:`.ShellProp` object
            Shell property object from where the stiffness and mass attributes
            are read from.
        update_KC0v_only : int, optional
            The default ``0`` means that the row and column indices ``KC0r``
            and ``KC0c`` should also be updated. Any other value will only
            update the stiffness matrix values ``KC0v``.
        hgfactor_membrane : double, optional
            Dimensionless multiplier of the artificial hourglass stiffness
            of the in-plane displacements ``u`` and ``v``, of Brockman
            (1987). Default ``1.``.
        hgfactor_physical : double, optional
            Dimensionless multiplier of the physical stabilisation of the
            bending and of the transverse shear of Belytschko and Leviathan
            (1994), which controls the hourglass modes of ``w``, ``r_x`` and
            ``r_y``; the default ``1.`` is the formulation of the paper,
            which has no parameter, and other values serve verification and
            sensitivity studies. See the Notes of :mod:`pyfe3d.quad4r`.

        Notes
        -----
        The element matrix in element coordinates is the Hessian of the strain
        energy of the Notes of :mod:`pyfe3d.quad4r`. The membrane, coupling,
        bending and transverse shear terms, `\boldsymbol{B}^\top \boldsymbol{C} \boldsymbol{B}` with the
        constitutive matrices `\boldsymbol{C}` of
        :meth:`pyfe3d.shellprop.ShellProp.get_constitutive_element` rotated
        with ``m11, m12, m21, m22``, are evaluated at the centroid,
        `\xi = \eta = 0`, with weight `w_{ij} = 4`, `4 \det J` being the
        element area `A`. The transverse shear strains are those of
        Belytschko and Leviathan (1994): the constant part of the assumed
        strains of Dvorkin and Bathe at the centroid, and the stabilisation
        of their nonconstant part,

        .. math::

            \boldsymbol{K}^s_{stab} = f_p \frac{A}{3} \left(
            \boldsymbol{D}_1^\top \boldsymbol{A}_s \boldsymbol{D}_1
            + \boldsymbol{D}_2^\top \boldsymbol{A}_s \boldsymbol{D}_2 \right)

        with `\boldsymbol{D}_1 = \boldsymbol{J}_0^{-1} \{(e^D_{\xi z} - e^B_{\xi
        z})/2, 0\}^\top` and `\boldsymbol{D}_2 = \boldsymbol{J}_0^{-1} \{0, (e^C_{\eta
        z} - e^A_{\eta z})/2\}^\top`, the rows of the covariant shear strains
        at the tying points `A (-1, 0)`, `B (0, -1)`, `C (1, 0)` and `D (0,
        1)`, `\boldsymbol{J}_0` the Jacobian at the centroid and `\boldsymbol{A}_s` the
        matrix of `A_{55}`, `A_{45}` and `A_{44}` for `\gamma_{xz}` and
        `\gamma_{yz}`. The bending has the stabilisation of their Eq. (B.18),

        .. math::

            K^b_{r_{xI} r_{xJ}} = f_p Q_{xx} \gamma_I \gamma_J, \quad
            K^b_{r_{yI} r_{yJ}} = f_p Q_{yy} \gamma_I \gamma_J, \quad
            K^b_{r_{xI} r_{yJ}} = f_p Q_{xy} \gamma_I \gamma_J

        .. math::

            Q_{xx} = H_{xx} D_{66} + H_{yy} D_{22} + 2 H_{xy} D_{26}, \quad
            Q_{yy} = H_{xx} D_{11} + H_{yy} D_{66} + 2 H_{xy} D_{16}

        .. math::

            Q_{xy} = -H_{xx} D_{16} - H_{yy} D_{26} - H_{xy} (D_{12} + D_{66})

        with `\gamma_I = (h_I - (\boldsymbol{h}^\top \boldsymbol{x}) S_{I,x} -
        (\boldsymbol{h}^\top \boldsymbol{y}) S_{I,y})/4`, their Eq. (3.24),
        `H_{xx}`, `H_{yy}` and `H_{xy}` the integrals over the element of
        `h_{,x}^2`, `h_{,y}^2` and `h_{,x} h_{,y}`, `h = \xi \eta`, evaluated
        with two by two Gauss points, and `D_{ij}` the entries of `\boldsymbol{D} -
        \boldsymbol{B} \boldsymbol{A}^{-1} \boldsymbol{B}`, which do not change
        when the reference surface is moved with ``prop.offset``; `f_p` is
        ``hgfactor_physical``. The membrane hourglass term adds, for `u` and
        `v`,

        .. math::

            K_{ij} = A E^{(h)} \gamma_i \gamma_j, \qquad
            \gamma_i = S_{1,xy} \left( h_i - hgx \, S_{i,x} - hgy \, S_{i,y}
            \right)

        with `\boldsymbol{h} = [1, -1, 1, -1]`, ``hgx = x1 - x2 + x3 - x4``, ``hgy =
        y1 - y2 + y3 - y4`` and the derivatives `S_{i,x}`, `S_{i,y}` and
        `S_{1,xy}` of the bilinear shape functions at the centroid. This is
        the orthogonalised operator of Eq. (14) of Brockman 1987 [1]_,
        normalised as in his Eq. (15), `\gamma = \partial^2 \boldsymbol{S}/\partial x
        \partial y` at the centroid, so that the generalized stiffnesses of
        his Eq. (16) keep their calibration. The correction terms of ``hgx``
        and ``hgy`` vanish for a parallelogram. With the thickness `t` of
        ``prop.h``, the generalized stiffnesses are

        .. math::

            E^{(h)}_u = 0.1 \, f_m E_{1eq} t A, \qquad
            E^{(h)}_v = 0.1 \, f_m E_{2eq} t A

        where `f_m` is ``hgfactor_membrane``. The reason for the area factor
        is given in the Notes of :mod:`pyfe3d.quad4r`. The equivalent moduli,
        in the element coordinate system, are

        .. math::

            E_{1eq} = \frac{1}{t a_{11}}, \quad
            E_{2eq} = \frac{1}{t a_{22}}

        with `[a] = \boldsymbol{A}^{-1}`, the coupling matrix `\boldsymbol{B}` being
        neglected. A homogeneous plate of modulus `E` has `a_{11} = 1/(E
        t)`, so `E_{1eq}` is the modulus of the homogeneous plate that
        stretches as the laminate under `N_{xx}` alone with the curvatures
        restrained; analogously in `y`. Both reduce to `E` for an isotropic
        plate, as in Eq. (16) of Brockman, whereas `A_{11}/t` would give `E/(1
        - \nu^2)`, and neither depends on the offset.

        The drilling term, `\gamma_{r_z} \boldsymbol{B}_{r_z}^\top \boldsymbol{B}_{r_z}` with
        `B_{r_z} = S^{r_z} + 1/2 S^u_{,y} - 1/2 S^v_{,x}`, is integrated
        with a two-point Gauss-Legendre quadrature in each direction. For
        ``drilling_model = 0``, `\gamma_{r_z}` is ``gamma_rz`` when it is not
        negative and `A_{66}` is not zero, and `A_{66}` otherwise; for the
        other values, `\gamma_{r_z} = K6ROT \cdot 10^{-6} A_{66}`.

        The element matrix is rotated to global coordinates with `\boldsymbol{K}_g = \boldsymbol{T}
        \boldsymbol{K}_e \boldsymbol{T}^\top`, where `\boldsymbol{T}` is block diagonal, with the matrix
        `\boldsymbol{T}_{3\times 3}` of the entries `T_{ij}` stored in ``r11, ..., r33``,
        whose columns are the element axes, applied to the translations and to
        the rotations of each node.

        With ``warp_offsets = 1`` and a warped element the element matrix is
        `\boldsymbol{J}^\top \boldsymbol{K}_e \boldsymbol{J}`, see the Notes of
        :class:`.Quad4R`, computed as `\boldsymbol{J}^\top \boldsymbol{K}_e
        \boldsymbol{J} - \boldsymbol{K}_e` and added to the entries above once
        rotated to global coordinates.

        References
        ----------
        .. [1] Brockman, R. A., Dynamics of the Bilinear Mindlin Plate Element,
           International Journal for Numerical Methods in Engineering, 24(12),
           2343-2356, 1987. https://doi.org/10.1002/nme.1620241208

        """
        cdef int warped, iw
        cdef double zw[4]
        cdef double Rw[9]
        cdef double Kd[576]
        cdef double dKd[576]
        cdef int c1, c2, c3, c4, k
        cdef double x1, x2, x3, x4
        cdef double y1, y2, y3, y4
        cdef int pti, ptj
        cdef double wij, detJ, xi, eta
        cdef double points[2]
        cdef double Ae[9]
        cdef double Be[9]
        cdef double De[9]
        cdef double Atse[4]
        cdef double A44, A45, A55
        # NOTE ABD in the element direction
        cdef double A11, A12, A16, A22, A26, A66
        cdef double B11, B12, B16, B22, B26, B66
        cdef double D11, D12, D16, D22, D26, D66
        cdef double E1eq, E2eq
        cdef double K6ROT
        cdef double Eu, Ev
        cdef double Ds[9]
        cdef double Kws[144]
        cdef double r11, r12, r13, r21, r22, r23, r31, r32, r33
        cdef double a11, a22
        cdef double J11, J12, J21, J22
        cdef double j11, j12, j21, j22
        cdef double S1_x, S2_x, S3_x, S4_x, S1_y, S2_y, S3_y, S4_y
        cdef double S1, S2, S3, S4
        cdef double S1_xy, S2_xy, S3_xy, S4_xy, gamma1, gamma2, gamma3, gamma4
        cdef double hgx, hgy

        cdef double KC0e0000, KC0e0001, KC0e0003, KC0e0004, KC0e0005, KC0e0006, KC0e0007, KC0e0009, KC0e0010, KC0e0011, KC0e0012, KC0e0013, KC0e0015, KC0e0016, KC0e0017, KC0e0018, KC0e0019, KC0e0021, KC0e0022, KC0e0023
        cdef double KC0e0101, KC0e0103, KC0e0104, KC0e0105, KC0e0106, KC0e0107, KC0e0109, KC0e0110, KC0e0111, KC0e0112, KC0e0113, KC0e0115, KC0e0116, KC0e0117, KC0e0118, KC0e0119, KC0e0121, KC0e0122, KC0e0123
        cdef double KC0e0202, KC0e0203, KC0e0204, KC0e0208, KC0e0209, KC0e0210, KC0e0214, KC0e0215, KC0e0216, KC0e0220, KC0e0221, KC0e0222, KC0e0303
        cdef double KC0e0304, KC0e0306, KC0e0307, KC0e0308, KC0e0309, KC0e0310, KC0e0312, KC0e0313, KC0e0314, KC0e0315, KC0e0316, KC0e0318, KC0e0319, KC0e0320, KC0e0321, KC0e0322
        cdef double KC0e0404, KC0e0406, KC0e0407, KC0e0408, KC0e0409, KC0e0410, KC0e0412, KC0e0413, KC0e0414, KC0e0415, KC0e0416, KC0e0418, KC0e0419, KC0e0420, KC0e0421, KC0e0422
        cdef double KC0e0505, KC0e0506, KC0e0507, KC0e0511, KC0e0512, KC0e0513, KC0e0517, KC0e0518, KC0e0519, KC0e0523
        cdef double KC0e0606, KC0e0607, KC0e0609, KC0e0610, KC0e0611, KC0e0612, KC0e0613, KC0e0615, KC0e0616, KC0e0617, KC0e0618, KC0e0619, KC0e0621, KC0e0622, KC0e0623
        cdef double KC0e0707, KC0e0709, KC0e0710, KC0e0711, KC0e0712, KC0e0713, KC0e0715, KC0e0716, KC0e0717, KC0e0718, KC0e0719, KC0e0721, KC0e0722, KC0e0723
        cdef double KC0e0808, KC0e0809, KC0e0810, KC0e0814, KC0e0815, KC0e0816, KC0e0820, KC0e0821, KC0e0822
        cdef double KC0e0909, KC0e0910, KC0e0912, KC0e0913, KC0e0914, KC0e0915, KC0e0916, KC0e0918, KC0e0919, KC0e0920, KC0e0921, KC0e0922
        cdef double KC0e1010, KC0e1012, KC0e1013, KC0e1014, KC0e1015, KC0e1016, KC0e1018, KC0e1019, KC0e1020, KC0e1021, KC0e1022
        cdef double KC0e1111, KC0e1112, KC0e1113, KC0e1117, KC0e1118, KC0e1119, KC0e1123
        cdef double KC0e1212, KC0e1213, KC0e1215, KC0e1216, KC0e1217, KC0e1218, KC0e1219, KC0e1221, KC0e1222, KC0e1223
        cdef double KC0e1313, KC0e1315, KC0e1316, KC0e1317, KC0e1318, KC0e1319, KC0e1321, KC0e1322, KC0e1323
        cdef double KC0e1414, KC0e1415, KC0e1416, KC0e1420, KC0e1421, KC0e1422
        cdef double KC0e1515, KC0e1516, KC0e1518, KC0e1519, KC0e1520, KC0e1521, KC0e1522
        cdef double KC0e1616, KC0e1618, KC0e1619, KC0e1620, KC0e1621, KC0e1622
        cdef double KC0e1717, KC0e1718, KC0e1719, KC0e1723
        cdef double KC0e1818, KC0e1819, KC0e1821, KC0e1822, KC0e1823
        cdef double KC0e1919, KC0e1921, KC0e1922, KC0e1923
        cdef double KC0e2020, KC0e2021, KC0e2022
        cdef double KC0e2121, KC0e2122
        cdef double KC0e2222
        cdef double KC0e2323

        with nogil:
            # NOTE constitutive matrices in the element coordinate system,
            #      the same function is used by all element methods
            prop.get_constitutive_element(self.m11, self.m12, self.m21, self.m22, Ae, Be, De, Atse)
            A11 = Ae[0]
            A12 = Ae[1]
            A16 = Ae[2]
            A22 = Ae[4]
            A26 = Ae[5]
            A66 = Ae[8]
            B11 = Be[0]
            B12 = Be[1]
            B16 = Be[2]
            B22 = Be[4]
            B26 = Be[5]
            B66 = Be[8]
            D11 = De[0]
            D12 = De[1]
            D16 = De[2]
            D22 = De[4]
            D26 = De[5]
            D66 = De[8]
            # NOTE transverse shear stiffness with the shear correction already applied
            A44 = Atse[0]
            A45 = Atse[1]
            A55 = Atse[3]

            # NOTE equivalent moduli of the hourglass control, from the
            #      compliance [a] = [A]^-1 of the extensional stiffness in
            #      the element coordinate system, [B] neglected, see the
            #      Notes of update_KC0
            a11 = (-A22*A66 + A26**2)/(-A11*A22*A66 + A11*A26**2 + A12**2*A66 - 2*A12*A16*A26 + A16**2*A22)
            a22 = (-A11*A66 + A16**2)/(-A11*A22*A66 + A11*A26**2 + A12**2*A66 - 2*A12*A16*A26 + A16**2*A22)
            E1eq = 1./(prop.h*a11)
            E2eq = 1./(prop.h*a22)
            # NOTE bending stiffness with zero membrane forces, of the
            #      physical stabilisation of the bending
            _reduced_bending_stiffness(Ae, Be, De, Ds)


            # NOTE both drilling models use the same operator, the one of
            #      Hughes and Brezzi (1989) integrated with two points per
            #      direction below, and differ only in the coefficient that
            #      multiplies it. The generated block carries the factor
            #      K6ROT*1e-6*A66, so the physics-based value gamma_rz =
            #      A66 is obtained with K6ROT = 1e6, and no generated
            #      expression needs to change. See the module documentation
            if self.drilling_model == 0:
                if self.gamma_rz >= 0. and A66 != 0.:
                    K6ROT = 1.e6*self.gamma_rz/A66
                else:
                    K6ROT = 1.e6
            else:
                K6ROT = self.K6ROT
            
            # NOTE ignoring z in local coordinates
            x1 = self.probe.xe[0]
            y1 = self.probe.xe[1]
            # z1 = self.probe.xe[2]
            x2 = self.probe.xe[3]
            y2 = self.probe.xe[4]
            # z2 = self.probe.xe[5]
            x3 = self.probe.xe[6]
            y3 = self.probe.xe[7]
            # z3 = self.probe.xe[8]
            x4 = self.probe.xe[9]
            y4 = self.probe.xe[10]
            # z4 = self.probe.xe[11]

            # local to global transformation
            r11 = self.r11
            r12 = self.r12
            r13 = self.r13
            r21 = self.r21
            r22 = self.r22
            r23 = self.r23
            r31 = self.r31
            r32 = self.r32
            r33 = self.r33

            if update_KC0v_only == 0:
                # positions in the global stiffness matrix
                c1 = self.c1
                c2 = self.c2
                c3 = self.c3
                c4 = self.c4

                k = self.init_k_KC0
                KC0r[k] = 0+c1
                KC0c[k] = 0+c1
                k += 1
                KC0r[k] = 0+c1
                KC0c[k] = 1+c1
                k += 1
                KC0r[k] = 0+c1
                KC0c[k] = 2+c1
                k += 1
                KC0r[k] = 0+c1
                KC0c[k] = 3+c1
                k += 1
                KC0r[k] = 0+c1
                KC0c[k] = 4+c1
                k += 1
                KC0r[k] = 0+c1
                KC0c[k] = 5+c1
                k += 1
                KC0r[k] = 0+c1
                KC0c[k] = 0+c2
                k += 1
                KC0r[k] = 0+c1
                KC0c[k] = 1+c2
                k += 1
                KC0r[k] = 0+c1
                KC0c[k] = 2+c2
                k += 1
                KC0r[k] = 0+c1
                KC0c[k] = 3+c2
                k += 1
                KC0r[k] = 0+c1
                KC0c[k] = 4+c2
                k += 1
                KC0r[k] = 0+c1
                KC0c[k] = 5+c2
                k += 1
                KC0r[k] = 0+c1
                KC0c[k] = 0+c3
                k += 1
                KC0r[k] = 0+c1
                KC0c[k] = 1+c3
                k += 1
                KC0r[k] = 0+c1
                KC0c[k] = 2+c3
                k += 1
                KC0r[k] = 0+c1
                KC0c[k] = 3+c3
                k += 1
                KC0r[k] = 0+c1
                KC0c[k] = 4+c3
                k += 1
                KC0r[k] = 0+c1
                KC0c[k] = 5+c3
                k += 1
                KC0r[k] = 0+c1
                KC0c[k] = 0+c4
                k += 1
                KC0r[k] = 0+c1
                KC0c[k] = 1+c4
                k += 1
                KC0r[k] = 0+c1
                KC0c[k] = 2+c4
                k += 1
                KC0r[k] = 0+c1
                KC0c[k] = 3+c4
                k += 1
                KC0r[k] = 0+c1
                KC0c[k] = 4+c4
                k += 1
                KC0r[k] = 0+c1
                KC0c[k] = 5+c4
                k += 1
                KC0r[k] = 1+c1
                KC0c[k] = 0+c1
                k += 1
                KC0r[k] = 1+c1
                KC0c[k] = 1+c1
                k += 1
                KC0r[k] = 1+c1
                KC0c[k] = 2+c1
                k += 1
                KC0r[k] = 1+c1
                KC0c[k] = 3+c1
                k += 1
                KC0r[k] = 1+c1
                KC0c[k] = 4+c1
                k += 1
                KC0r[k] = 1+c1
                KC0c[k] = 5+c1
                k += 1
                KC0r[k] = 1+c1
                KC0c[k] = 0+c2
                k += 1
                KC0r[k] = 1+c1
                KC0c[k] = 1+c2
                k += 1
                KC0r[k] = 1+c1
                KC0c[k] = 2+c2
                k += 1
                KC0r[k] = 1+c1
                KC0c[k] = 3+c2
                k += 1
                KC0r[k] = 1+c1
                KC0c[k] = 4+c2
                k += 1
                KC0r[k] = 1+c1
                KC0c[k] = 5+c2
                k += 1
                KC0r[k] = 1+c1
                KC0c[k] = 0+c3
                k += 1
                KC0r[k] = 1+c1
                KC0c[k] = 1+c3
                k += 1
                KC0r[k] = 1+c1
                KC0c[k] = 2+c3
                k += 1
                KC0r[k] = 1+c1
                KC0c[k] = 3+c3
                k += 1
                KC0r[k] = 1+c1
                KC0c[k] = 4+c3
                k += 1
                KC0r[k] = 1+c1
                KC0c[k] = 5+c3
                k += 1
                KC0r[k] = 1+c1
                KC0c[k] = 0+c4
                k += 1
                KC0r[k] = 1+c1
                KC0c[k] = 1+c4
                k += 1
                KC0r[k] = 1+c1
                KC0c[k] = 2+c4
                k += 1
                KC0r[k] = 1+c1
                KC0c[k] = 3+c4
                k += 1
                KC0r[k] = 1+c1
                KC0c[k] = 4+c4
                k += 1
                KC0r[k] = 1+c1
                KC0c[k] = 5+c4
                k += 1
                KC0r[k] = 2+c1
                KC0c[k] = 0+c1
                k += 1
                KC0r[k] = 2+c1
                KC0c[k] = 1+c1
                k += 1
                KC0r[k] = 2+c1
                KC0c[k] = 2+c1
                k += 1
                KC0r[k] = 2+c1
                KC0c[k] = 3+c1
                k += 1
                KC0r[k] = 2+c1
                KC0c[k] = 4+c1
                k += 1
                KC0r[k] = 2+c1
                KC0c[k] = 5+c1
                k += 1
                KC0r[k] = 2+c1
                KC0c[k] = 0+c2
                k += 1
                KC0r[k] = 2+c1
                KC0c[k] = 1+c2
                k += 1
                KC0r[k] = 2+c1
                KC0c[k] = 2+c2
                k += 1
                KC0r[k] = 2+c1
                KC0c[k] = 3+c2
                k += 1
                KC0r[k] = 2+c1
                KC0c[k] = 4+c2
                k += 1
                KC0r[k] = 2+c1
                KC0c[k] = 5+c2
                k += 1
                KC0r[k] = 2+c1
                KC0c[k] = 0+c3
                k += 1
                KC0r[k] = 2+c1
                KC0c[k] = 1+c3
                k += 1
                KC0r[k] = 2+c1
                KC0c[k] = 2+c3
                k += 1
                KC0r[k] = 2+c1
                KC0c[k] = 3+c3
                k += 1
                KC0r[k] = 2+c1
                KC0c[k] = 4+c3
                k += 1
                KC0r[k] = 2+c1
                KC0c[k] = 5+c3
                k += 1
                KC0r[k] = 2+c1
                KC0c[k] = 0+c4
                k += 1
                KC0r[k] = 2+c1
                KC0c[k] = 1+c4
                k += 1
                KC0r[k] = 2+c1
                KC0c[k] = 2+c4
                k += 1
                KC0r[k] = 2+c1
                KC0c[k] = 3+c4
                k += 1
                KC0r[k] = 2+c1
                KC0c[k] = 4+c4
                k += 1
                KC0r[k] = 2+c1
                KC0c[k] = 5+c4
                k += 1
                KC0r[k] = 3+c1
                KC0c[k] = 0+c1
                k += 1
                KC0r[k] = 3+c1
                KC0c[k] = 1+c1
                k += 1
                KC0r[k] = 3+c1
                KC0c[k] = 2+c1
                k += 1
                KC0r[k] = 3+c1
                KC0c[k] = 3+c1
                k += 1
                KC0r[k] = 3+c1
                KC0c[k] = 4+c1
                k += 1
                KC0r[k] = 3+c1
                KC0c[k] = 5+c1
                k += 1
                KC0r[k] = 3+c1
                KC0c[k] = 0+c2
                k += 1
                KC0r[k] = 3+c1
                KC0c[k] = 1+c2
                k += 1
                KC0r[k] = 3+c1
                KC0c[k] = 2+c2
                k += 1
                KC0r[k] = 3+c1
                KC0c[k] = 3+c2
                k += 1
                KC0r[k] = 3+c1
                KC0c[k] = 4+c2
                k += 1
                KC0r[k] = 3+c1
                KC0c[k] = 5+c2
                k += 1
                KC0r[k] = 3+c1
                KC0c[k] = 0+c3
                k += 1
                KC0r[k] = 3+c1
                KC0c[k] = 1+c3
                k += 1
                KC0r[k] = 3+c1
                KC0c[k] = 2+c3
                k += 1
                KC0r[k] = 3+c1
                KC0c[k] = 3+c3
                k += 1
                KC0r[k] = 3+c1
                KC0c[k] = 4+c3
                k += 1
                KC0r[k] = 3+c1
                KC0c[k] = 5+c3
                k += 1
                KC0r[k] = 3+c1
                KC0c[k] = 0+c4
                k += 1
                KC0r[k] = 3+c1
                KC0c[k] = 1+c4
                k += 1
                KC0r[k] = 3+c1
                KC0c[k] = 2+c4
                k += 1
                KC0r[k] = 3+c1
                KC0c[k] = 3+c4
                k += 1
                KC0r[k] = 3+c1
                KC0c[k] = 4+c4
                k += 1
                KC0r[k] = 3+c1
                KC0c[k] = 5+c4
                k += 1
                KC0r[k] = 4+c1
                KC0c[k] = 0+c1
                k += 1
                KC0r[k] = 4+c1
                KC0c[k] = 1+c1
                k += 1
                KC0r[k] = 4+c1
                KC0c[k] = 2+c1
                k += 1
                KC0r[k] = 4+c1
                KC0c[k] = 3+c1
                k += 1
                KC0r[k] = 4+c1
                KC0c[k] = 4+c1
                k += 1
                KC0r[k] = 4+c1
                KC0c[k] = 5+c1
                k += 1
                KC0r[k] = 4+c1
                KC0c[k] = 0+c2
                k += 1
                KC0r[k] = 4+c1
                KC0c[k] = 1+c2
                k += 1
                KC0r[k] = 4+c1
                KC0c[k] = 2+c2
                k += 1
                KC0r[k] = 4+c1
                KC0c[k] = 3+c2
                k += 1
                KC0r[k] = 4+c1
                KC0c[k] = 4+c2
                k += 1
                KC0r[k] = 4+c1
                KC0c[k] = 5+c2
                k += 1
                KC0r[k] = 4+c1
                KC0c[k] = 0+c3
                k += 1
                KC0r[k] = 4+c1
                KC0c[k] = 1+c3
                k += 1
                KC0r[k] = 4+c1
                KC0c[k] = 2+c3
                k += 1
                KC0r[k] = 4+c1
                KC0c[k] = 3+c3
                k += 1
                KC0r[k] = 4+c1
                KC0c[k] = 4+c3
                k += 1
                KC0r[k] = 4+c1
                KC0c[k] = 5+c3
                k += 1
                KC0r[k] = 4+c1
                KC0c[k] = 0+c4
                k += 1
                KC0r[k] = 4+c1
                KC0c[k] = 1+c4
                k += 1
                KC0r[k] = 4+c1
                KC0c[k] = 2+c4
                k += 1
                KC0r[k] = 4+c1
                KC0c[k] = 3+c4
                k += 1
                KC0r[k] = 4+c1
                KC0c[k] = 4+c4
                k += 1
                KC0r[k] = 4+c1
                KC0c[k] = 5+c4
                k += 1
                KC0r[k] = 5+c1
                KC0c[k] = 0+c1
                k += 1
                KC0r[k] = 5+c1
                KC0c[k] = 1+c1
                k += 1
                KC0r[k] = 5+c1
                KC0c[k] = 2+c1
                k += 1
                KC0r[k] = 5+c1
                KC0c[k] = 3+c1
                k += 1
                KC0r[k] = 5+c1
                KC0c[k] = 4+c1
                k += 1
                KC0r[k] = 5+c1
                KC0c[k] = 5+c1
                k += 1
                KC0r[k] = 5+c1
                KC0c[k] = 0+c2
                k += 1
                KC0r[k] = 5+c1
                KC0c[k] = 1+c2
                k += 1
                KC0r[k] = 5+c1
                KC0c[k] = 2+c2
                k += 1
                KC0r[k] = 5+c1
                KC0c[k] = 3+c2
                k += 1
                KC0r[k] = 5+c1
                KC0c[k] = 4+c2
                k += 1
                KC0r[k] = 5+c1
                KC0c[k] = 5+c2
                k += 1
                KC0r[k] = 5+c1
                KC0c[k] = 0+c3
                k += 1
                KC0r[k] = 5+c1
                KC0c[k] = 1+c3
                k += 1
                KC0r[k] = 5+c1
                KC0c[k] = 2+c3
                k += 1
                KC0r[k] = 5+c1
                KC0c[k] = 3+c3
                k += 1
                KC0r[k] = 5+c1
                KC0c[k] = 4+c3
                k += 1
                KC0r[k] = 5+c1
                KC0c[k] = 5+c3
                k += 1
                KC0r[k] = 5+c1
                KC0c[k] = 0+c4
                k += 1
                KC0r[k] = 5+c1
                KC0c[k] = 1+c4
                k += 1
                KC0r[k] = 5+c1
                KC0c[k] = 2+c4
                k += 1
                KC0r[k] = 5+c1
                KC0c[k] = 3+c4
                k += 1
                KC0r[k] = 5+c1
                KC0c[k] = 4+c4
                k += 1
                KC0r[k] = 5+c1
                KC0c[k] = 5+c4
                k += 1
                KC0r[k] = 0+c2
                KC0c[k] = 0+c1
                k += 1
                KC0r[k] = 0+c2
                KC0c[k] = 1+c1
                k += 1
                KC0r[k] = 0+c2
                KC0c[k] = 2+c1
                k += 1
                KC0r[k] = 0+c2
                KC0c[k] = 3+c1
                k += 1
                KC0r[k] = 0+c2
                KC0c[k] = 4+c1
                k += 1
                KC0r[k] = 0+c2
                KC0c[k] = 5+c1
                k += 1
                KC0r[k] = 0+c2
                KC0c[k] = 0+c2
                k += 1
                KC0r[k] = 0+c2
                KC0c[k] = 1+c2
                k += 1
                KC0r[k] = 0+c2
                KC0c[k] = 2+c2
                k += 1
                KC0r[k] = 0+c2
                KC0c[k] = 3+c2
                k += 1
                KC0r[k] = 0+c2
                KC0c[k] = 4+c2
                k += 1
                KC0r[k] = 0+c2
                KC0c[k] = 5+c2
                k += 1
                KC0r[k] = 0+c2
                KC0c[k] = 0+c3
                k += 1
                KC0r[k] = 0+c2
                KC0c[k] = 1+c3
                k += 1
                KC0r[k] = 0+c2
                KC0c[k] = 2+c3
                k += 1
                KC0r[k] = 0+c2
                KC0c[k] = 3+c3
                k += 1
                KC0r[k] = 0+c2
                KC0c[k] = 4+c3
                k += 1
                KC0r[k] = 0+c2
                KC0c[k] = 5+c3
                k += 1
                KC0r[k] = 0+c2
                KC0c[k] = 0+c4
                k += 1
                KC0r[k] = 0+c2
                KC0c[k] = 1+c4
                k += 1
                KC0r[k] = 0+c2
                KC0c[k] = 2+c4
                k += 1
                KC0r[k] = 0+c2
                KC0c[k] = 3+c4
                k += 1
                KC0r[k] = 0+c2
                KC0c[k] = 4+c4
                k += 1
                KC0r[k] = 0+c2
                KC0c[k] = 5+c4
                k += 1
                KC0r[k] = 1+c2
                KC0c[k] = 0+c1
                k += 1
                KC0r[k] = 1+c2
                KC0c[k] = 1+c1
                k += 1
                KC0r[k] = 1+c2
                KC0c[k] = 2+c1
                k += 1
                KC0r[k] = 1+c2
                KC0c[k] = 3+c1
                k += 1
                KC0r[k] = 1+c2
                KC0c[k] = 4+c1
                k += 1
                KC0r[k] = 1+c2
                KC0c[k] = 5+c1
                k += 1
                KC0r[k] = 1+c2
                KC0c[k] = 0+c2
                k += 1
                KC0r[k] = 1+c2
                KC0c[k] = 1+c2
                k += 1
                KC0r[k] = 1+c2
                KC0c[k] = 2+c2
                k += 1
                KC0r[k] = 1+c2
                KC0c[k] = 3+c2
                k += 1
                KC0r[k] = 1+c2
                KC0c[k] = 4+c2
                k += 1
                KC0r[k] = 1+c2
                KC0c[k] = 5+c2
                k += 1
                KC0r[k] = 1+c2
                KC0c[k] = 0+c3
                k += 1
                KC0r[k] = 1+c2
                KC0c[k] = 1+c3
                k += 1
                KC0r[k] = 1+c2
                KC0c[k] = 2+c3
                k += 1
                KC0r[k] = 1+c2
                KC0c[k] = 3+c3
                k += 1
                KC0r[k] = 1+c2
                KC0c[k] = 4+c3
                k += 1
                KC0r[k] = 1+c2
                KC0c[k] = 5+c3
                k += 1
                KC0r[k] = 1+c2
                KC0c[k] = 0+c4
                k += 1
                KC0r[k] = 1+c2
                KC0c[k] = 1+c4
                k += 1
                KC0r[k] = 1+c2
                KC0c[k] = 2+c4
                k += 1
                KC0r[k] = 1+c2
                KC0c[k] = 3+c4
                k += 1
                KC0r[k] = 1+c2
                KC0c[k] = 4+c4
                k += 1
                KC0r[k] = 1+c2
                KC0c[k] = 5+c4
                k += 1
                KC0r[k] = 2+c2
                KC0c[k] = 0+c1
                k += 1
                KC0r[k] = 2+c2
                KC0c[k] = 1+c1
                k += 1
                KC0r[k] = 2+c2
                KC0c[k] = 2+c1
                k += 1
                KC0r[k] = 2+c2
                KC0c[k] = 3+c1
                k += 1
                KC0r[k] = 2+c2
                KC0c[k] = 4+c1
                k += 1
                KC0r[k] = 2+c2
                KC0c[k] = 5+c1
                k += 1
                KC0r[k] = 2+c2
                KC0c[k] = 0+c2
                k += 1
                KC0r[k] = 2+c2
                KC0c[k] = 1+c2
                k += 1
                KC0r[k] = 2+c2
                KC0c[k] = 2+c2
                k += 1
                KC0r[k] = 2+c2
                KC0c[k] = 3+c2
                k += 1
                KC0r[k] = 2+c2
                KC0c[k] = 4+c2
                k += 1
                KC0r[k] = 2+c2
                KC0c[k] = 5+c2
                k += 1
                KC0r[k] = 2+c2
                KC0c[k] = 0+c3
                k += 1
                KC0r[k] = 2+c2
                KC0c[k] = 1+c3
                k += 1
                KC0r[k] = 2+c2
                KC0c[k] = 2+c3
                k += 1
                KC0r[k] = 2+c2
                KC0c[k] = 3+c3
                k += 1
                KC0r[k] = 2+c2
                KC0c[k] = 4+c3
                k += 1
                KC0r[k] = 2+c2
                KC0c[k] = 5+c3
                k += 1
                KC0r[k] = 2+c2
                KC0c[k] = 0+c4
                k += 1
                KC0r[k] = 2+c2
                KC0c[k] = 1+c4
                k += 1
                KC0r[k] = 2+c2
                KC0c[k] = 2+c4
                k += 1
                KC0r[k] = 2+c2
                KC0c[k] = 3+c4
                k += 1
                KC0r[k] = 2+c2
                KC0c[k] = 4+c4
                k += 1
                KC0r[k] = 2+c2
                KC0c[k] = 5+c4
                k += 1
                KC0r[k] = 3+c2
                KC0c[k] = 0+c1
                k += 1
                KC0r[k] = 3+c2
                KC0c[k] = 1+c1
                k += 1
                KC0r[k] = 3+c2
                KC0c[k] = 2+c1
                k += 1
                KC0r[k] = 3+c2
                KC0c[k] = 3+c1
                k += 1
                KC0r[k] = 3+c2
                KC0c[k] = 4+c1
                k += 1
                KC0r[k] = 3+c2
                KC0c[k] = 5+c1
                k += 1
                KC0r[k] = 3+c2
                KC0c[k] = 0+c2
                k += 1
                KC0r[k] = 3+c2
                KC0c[k] = 1+c2
                k += 1
                KC0r[k] = 3+c2
                KC0c[k] = 2+c2
                k += 1
                KC0r[k] = 3+c2
                KC0c[k] = 3+c2
                k += 1
                KC0r[k] = 3+c2
                KC0c[k] = 4+c2
                k += 1
                KC0r[k] = 3+c2
                KC0c[k] = 5+c2
                k += 1
                KC0r[k] = 3+c2
                KC0c[k] = 0+c3
                k += 1
                KC0r[k] = 3+c2
                KC0c[k] = 1+c3
                k += 1
                KC0r[k] = 3+c2
                KC0c[k] = 2+c3
                k += 1
                KC0r[k] = 3+c2
                KC0c[k] = 3+c3
                k += 1
                KC0r[k] = 3+c2
                KC0c[k] = 4+c3
                k += 1
                KC0r[k] = 3+c2
                KC0c[k] = 5+c3
                k += 1
                KC0r[k] = 3+c2
                KC0c[k] = 0+c4
                k += 1
                KC0r[k] = 3+c2
                KC0c[k] = 1+c4
                k += 1
                KC0r[k] = 3+c2
                KC0c[k] = 2+c4
                k += 1
                KC0r[k] = 3+c2
                KC0c[k] = 3+c4
                k += 1
                KC0r[k] = 3+c2
                KC0c[k] = 4+c4
                k += 1
                KC0r[k] = 3+c2
                KC0c[k] = 5+c4
                k += 1
                KC0r[k] = 4+c2
                KC0c[k] = 0+c1
                k += 1
                KC0r[k] = 4+c2
                KC0c[k] = 1+c1
                k += 1
                KC0r[k] = 4+c2
                KC0c[k] = 2+c1
                k += 1
                KC0r[k] = 4+c2
                KC0c[k] = 3+c1
                k += 1
                KC0r[k] = 4+c2
                KC0c[k] = 4+c1
                k += 1
                KC0r[k] = 4+c2
                KC0c[k] = 5+c1
                k += 1
                KC0r[k] = 4+c2
                KC0c[k] = 0+c2
                k += 1
                KC0r[k] = 4+c2
                KC0c[k] = 1+c2
                k += 1
                KC0r[k] = 4+c2
                KC0c[k] = 2+c2
                k += 1
                KC0r[k] = 4+c2
                KC0c[k] = 3+c2
                k += 1
                KC0r[k] = 4+c2
                KC0c[k] = 4+c2
                k += 1
                KC0r[k] = 4+c2
                KC0c[k] = 5+c2
                k += 1
                KC0r[k] = 4+c2
                KC0c[k] = 0+c3
                k += 1
                KC0r[k] = 4+c2
                KC0c[k] = 1+c3
                k += 1
                KC0r[k] = 4+c2
                KC0c[k] = 2+c3
                k += 1
                KC0r[k] = 4+c2
                KC0c[k] = 3+c3
                k += 1
                KC0r[k] = 4+c2
                KC0c[k] = 4+c3
                k += 1
                KC0r[k] = 4+c2
                KC0c[k] = 5+c3
                k += 1
                KC0r[k] = 4+c2
                KC0c[k] = 0+c4
                k += 1
                KC0r[k] = 4+c2
                KC0c[k] = 1+c4
                k += 1
                KC0r[k] = 4+c2
                KC0c[k] = 2+c4
                k += 1
                KC0r[k] = 4+c2
                KC0c[k] = 3+c4
                k += 1
                KC0r[k] = 4+c2
                KC0c[k] = 4+c4
                k += 1
                KC0r[k] = 4+c2
                KC0c[k] = 5+c4
                k += 1
                KC0r[k] = 5+c2
                KC0c[k] = 0+c1
                k += 1
                KC0r[k] = 5+c2
                KC0c[k] = 1+c1
                k += 1
                KC0r[k] = 5+c2
                KC0c[k] = 2+c1
                k += 1
                KC0r[k] = 5+c2
                KC0c[k] = 3+c1
                k += 1
                KC0r[k] = 5+c2
                KC0c[k] = 4+c1
                k += 1
                KC0r[k] = 5+c2
                KC0c[k] = 5+c1
                k += 1
                KC0r[k] = 5+c2
                KC0c[k] = 0+c2
                k += 1
                KC0r[k] = 5+c2
                KC0c[k] = 1+c2
                k += 1
                KC0r[k] = 5+c2
                KC0c[k] = 2+c2
                k += 1
                KC0r[k] = 5+c2
                KC0c[k] = 3+c2
                k += 1
                KC0r[k] = 5+c2
                KC0c[k] = 4+c2
                k += 1
                KC0r[k] = 5+c2
                KC0c[k] = 5+c2
                k += 1
                KC0r[k] = 5+c2
                KC0c[k] = 0+c3
                k += 1
                KC0r[k] = 5+c2
                KC0c[k] = 1+c3
                k += 1
                KC0r[k] = 5+c2
                KC0c[k] = 2+c3
                k += 1
                KC0r[k] = 5+c2
                KC0c[k] = 3+c3
                k += 1
                KC0r[k] = 5+c2
                KC0c[k] = 4+c3
                k += 1
                KC0r[k] = 5+c2
                KC0c[k] = 5+c3
                k += 1
                KC0r[k] = 5+c2
                KC0c[k] = 0+c4
                k += 1
                KC0r[k] = 5+c2
                KC0c[k] = 1+c4
                k += 1
                KC0r[k] = 5+c2
                KC0c[k] = 2+c4
                k += 1
                KC0r[k] = 5+c2
                KC0c[k] = 3+c4
                k += 1
                KC0r[k] = 5+c2
                KC0c[k] = 4+c4
                k += 1
                KC0r[k] = 5+c2
                KC0c[k] = 5+c4
                k += 1
                KC0r[k] = 0+c3
                KC0c[k] = 0+c1
                k += 1
                KC0r[k] = 0+c3
                KC0c[k] = 1+c1
                k += 1
                KC0r[k] = 0+c3
                KC0c[k] = 2+c1
                k += 1
                KC0r[k] = 0+c3
                KC0c[k] = 3+c1
                k += 1
                KC0r[k] = 0+c3
                KC0c[k] = 4+c1
                k += 1
                KC0r[k] = 0+c3
                KC0c[k] = 5+c1
                k += 1
                KC0r[k] = 0+c3
                KC0c[k] = 0+c2
                k += 1
                KC0r[k] = 0+c3
                KC0c[k] = 1+c2
                k += 1
                KC0r[k] = 0+c3
                KC0c[k] = 2+c2
                k += 1
                KC0r[k] = 0+c3
                KC0c[k] = 3+c2
                k += 1
                KC0r[k] = 0+c3
                KC0c[k] = 4+c2
                k += 1
                KC0r[k] = 0+c3
                KC0c[k] = 5+c2
                k += 1
                KC0r[k] = 0+c3
                KC0c[k] = 0+c3
                k += 1
                KC0r[k] = 0+c3
                KC0c[k] = 1+c3
                k += 1
                KC0r[k] = 0+c3
                KC0c[k] = 2+c3
                k += 1
                KC0r[k] = 0+c3
                KC0c[k] = 3+c3
                k += 1
                KC0r[k] = 0+c3
                KC0c[k] = 4+c3
                k += 1
                KC0r[k] = 0+c3
                KC0c[k] = 5+c3
                k += 1
                KC0r[k] = 0+c3
                KC0c[k] = 0+c4
                k += 1
                KC0r[k] = 0+c3
                KC0c[k] = 1+c4
                k += 1
                KC0r[k] = 0+c3
                KC0c[k] = 2+c4
                k += 1
                KC0r[k] = 0+c3
                KC0c[k] = 3+c4
                k += 1
                KC0r[k] = 0+c3
                KC0c[k] = 4+c4
                k += 1
                KC0r[k] = 0+c3
                KC0c[k] = 5+c4
                k += 1
                KC0r[k] = 1+c3
                KC0c[k] = 0+c1
                k += 1
                KC0r[k] = 1+c3
                KC0c[k] = 1+c1
                k += 1
                KC0r[k] = 1+c3
                KC0c[k] = 2+c1
                k += 1
                KC0r[k] = 1+c3
                KC0c[k] = 3+c1
                k += 1
                KC0r[k] = 1+c3
                KC0c[k] = 4+c1
                k += 1
                KC0r[k] = 1+c3
                KC0c[k] = 5+c1
                k += 1
                KC0r[k] = 1+c3
                KC0c[k] = 0+c2
                k += 1
                KC0r[k] = 1+c3
                KC0c[k] = 1+c2
                k += 1
                KC0r[k] = 1+c3
                KC0c[k] = 2+c2
                k += 1
                KC0r[k] = 1+c3
                KC0c[k] = 3+c2
                k += 1
                KC0r[k] = 1+c3
                KC0c[k] = 4+c2
                k += 1
                KC0r[k] = 1+c3
                KC0c[k] = 5+c2
                k += 1
                KC0r[k] = 1+c3
                KC0c[k] = 0+c3
                k += 1
                KC0r[k] = 1+c3
                KC0c[k] = 1+c3
                k += 1
                KC0r[k] = 1+c3
                KC0c[k] = 2+c3
                k += 1
                KC0r[k] = 1+c3
                KC0c[k] = 3+c3
                k += 1
                KC0r[k] = 1+c3
                KC0c[k] = 4+c3
                k += 1
                KC0r[k] = 1+c3
                KC0c[k] = 5+c3
                k += 1
                KC0r[k] = 1+c3
                KC0c[k] = 0+c4
                k += 1
                KC0r[k] = 1+c3
                KC0c[k] = 1+c4
                k += 1
                KC0r[k] = 1+c3
                KC0c[k] = 2+c4
                k += 1
                KC0r[k] = 1+c3
                KC0c[k] = 3+c4
                k += 1
                KC0r[k] = 1+c3
                KC0c[k] = 4+c4
                k += 1
                KC0r[k] = 1+c3
                KC0c[k] = 5+c4
                k += 1
                KC0r[k] = 2+c3
                KC0c[k] = 0+c1
                k += 1
                KC0r[k] = 2+c3
                KC0c[k] = 1+c1
                k += 1
                KC0r[k] = 2+c3
                KC0c[k] = 2+c1
                k += 1
                KC0r[k] = 2+c3
                KC0c[k] = 3+c1
                k += 1
                KC0r[k] = 2+c3
                KC0c[k] = 4+c1
                k += 1
                KC0r[k] = 2+c3
                KC0c[k] = 5+c1
                k += 1
                KC0r[k] = 2+c3
                KC0c[k] = 0+c2
                k += 1
                KC0r[k] = 2+c3
                KC0c[k] = 1+c2
                k += 1
                KC0r[k] = 2+c3
                KC0c[k] = 2+c2
                k += 1
                KC0r[k] = 2+c3
                KC0c[k] = 3+c2
                k += 1
                KC0r[k] = 2+c3
                KC0c[k] = 4+c2
                k += 1
                KC0r[k] = 2+c3
                KC0c[k] = 5+c2
                k += 1
                KC0r[k] = 2+c3
                KC0c[k] = 0+c3
                k += 1
                KC0r[k] = 2+c3
                KC0c[k] = 1+c3
                k += 1
                KC0r[k] = 2+c3
                KC0c[k] = 2+c3
                k += 1
                KC0r[k] = 2+c3
                KC0c[k] = 3+c3
                k += 1
                KC0r[k] = 2+c3
                KC0c[k] = 4+c3
                k += 1
                KC0r[k] = 2+c3
                KC0c[k] = 5+c3
                k += 1
                KC0r[k] = 2+c3
                KC0c[k] = 0+c4
                k += 1
                KC0r[k] = 2+c3
                KC0c[k] = 1+c4
                k += 1
                KC0r[k] = 2+c3
                KC0c[k] = 2+c4
                k += 1
                KC0r[k] = 2+c3
                KC0c[k] = 3+c4
                k += 1
                KC0r[k] = 2+c3
                KC0c[k] = 4+c4
                k += 1
                KC0r[k] = 2+c3
                KC0c[k] = 5+c4
                k += 1
                KC0r[k] = 3+c3
                KC0c[k] = 0+c1
                k += 1
                KC0r[k] = 3+c3
                KC0c[k] = 1+c1
                k += 1
                KC0r[k] = 3+c3
                KC0c[k] = 2+c1
                k += 1
                KC0r[k] = 3+c3
                KC0c[k] = 3+c1
                k += 1
                KC0r[k] = 3+c3
                KC0c[k] = 4+c1
                k += 1
                KC0r[k] = 3+c3
                KC0c[k] = 5+c1
                k += 1
                KC0r[k] = 3+c3
                KC0c[k] = 0+c2
                k += 1
                KC0r[k] = 3+c3
                KC0c[k] = 1+c2
                k += 1
                KC0r[k] = 3+c3
                KC0c[k] = 2+c2
                k += 1
                KC0r[k] = 3+c3
                KC0c[k] = 3+c2
                k += 1
                KC0r[k] = 3+c3
                KC0c[k] = 4+c2
                k += 1
                KC0r[k] = 3+c3
                KC0c[k] = 5+c2
                k += 1
                KC0r[k] = 3+c3
                KC0c[k] = 0+c3
                k += 1
                KC0r[k] = 3+c3
                KC0c[k] = 1+c3
                k += 1
                KC0r[k] = 3+c3
                KC0c[k] = 2+c3
                k += 1
                KC0r[k] = 3+c3
                KC0c[k] = 3+c3
                k += 1
                KC0r[k] = 3+c3
                KC0c[k] = 4+c3
                k += 1
                KC0r[k] = 3+c3
                KC0c[k] = 5+c3
                k += 1
                KC0r[k] = 3+c3
                KC0c[k] = 0+c4
                k += 1
                KC0r[k] = 3+c3
                KC0c[k] = 1+c4
                k += 1
                KC0r[k] = 3+c3
                KC0c[k] = 2+c4
                k += 1
                KC0r[k] = 3+c3
                KC0c[k] = 3+c4
                k += 1
                KC0r[k] = 3+c3
                KC0c[k] = 4+c4
                k += 1
                KC0r[k] = 3+c3
                KC0c[k] = 5+c4
                k += 1
                KC0r[k] = 4+c3
                KC0c[k] = 0+c1
                k += 1
                KC0r[k] = 4+c3
                KC0c[k] = 1+c1
                k += 1
                KC0r[k] = 4+c3
                KC0c[k] = 2+c1
                k += 1
                KC0r[k] = 4+c3
                KC0c[k] = 3+c1
                k += 1
                KC0r[k] = 4+c3
                KC0c[k] = 4+c1
                k += 1
                KC0r[k] = 4+c3
                KC0c[k] = 5+c1
                k += 1
                KC0r[k] = 4+c3
                KC0c[k] = 0+c2
                k += 1
                KC0r[k] = 4+c3
                KC0c[k] = 1+c2
                k += 1
                KC0r[k] = 4+c3
                KC0c[k] = 2+c2
                k += 1
                KC0r[k] = 4+c3
                KC0c[k] = 3+c2
                k += 1
                KC0r[k] = 4+c3
                KC0c[k] = 4+c2
                k += 1
                KC0r[k] = 4+c3
                KC0c[k] = 5+c2
                k += 1
                KC0r[k] = 4+c3
                KC0c[k] = 0+c3
                k += 1
                KC0r[k] = 4+c3
                KC0c[k] = 1+c3
                k += 1
                KC0r[k] = 4+c3
                KC0c[k] = 2+c3
                k += 1
                KC0r[k] = 4+c3
                KC0c[k] = 3+c3
                k += 1
                KC0r[k] = 4+c3
                KC0c[k] = 4+c3
                k += 1
                KC0r[k] = 4+c3
                KC0c[k] = 5+c3
                k += 1
                KC0r[k] = 4+c3
                KC0c[k] = 0+c4
                k += 1
                KC0r[k] = 4+c3
                KC0c[k] = 1+c4
                k += 1
                KC0r[k] = 4+c3
                KC0c[k] = 2+c4
                k += 1
                KC0r[k] = 4+c3
                KC0c[k] = 3+c4
                k += 1
                KC0r[k] = 4+c3
                KC0c[k] = 4+c4
                k += 1
                KC0r[k] = 4+c3
                KC0c[k] = 5+c4
                k += 1
                KC0r[k] = 5+c3
                KC0c[k] = 0+c1
                k += 1
                KC0r[k] = 5+c3
                KC0c[k] = 1+c1
                k += 1
                KC0r[k] = 5+c3
                KC0c[k] = 2+c1
                k += 1
                KC0r[k] = 5+c3
                KC0c[k] = 3+c1
                k += 1
                KC0r[k] = 5+c3
                KC0c[k] = 4+c1
                k += 1
                KC0r[k] = 5+c3
                KC0c[k] = 5+c1
                k += 1
                KC0r[k] = 5+c3
                KC0c[k] = 0+c2
                k += 1
                KC0r[k] = 5+c3
                KC0c[k] = 1+c2
                k += 1
                KC0r[k] = 5+c3
                KC0c[k] = 2+c2
                k += 1
                KC0r[k] = 5+c3
                KC0c[k] = 3+c2
                k += 1
                KC0r[k] = 5+c3
                KC0c[k] = 4+c2
                k += 1
                KC0r[k] = 5+c3
                KC0c[k] = 5+c2
                k += 1
                KC0r[k] = 5+c3
                KC0c[k] = 0+c3
                k += 1
                KC0r[k] = 5+c3
                KC0c[k] = 1+c3
                k += 1
                KC0r[k] = 5+c3
                KC0c[k] = 2+c3
                k += 1
                KC0r[k] = 5+c3
                KC0c[k] = 3+c3
                k += 1
                KC0r[k] = 5+c3
                KC0c[k] = 4+c3
                k += 1
                KC0r[k] = 5+c3
                KC0c[k] = 5+c3
                k += 1
                KC0r[k] = 5+c3
                KC0c[k] = 0+c4
                k += 1
                KC0r[k] = 5+c3
                KC0c[k] = 1+c4
                k += 1
                KC0r[k] = 5+c3
                KC0c[k] = 2+c4
                k += 1
                KC0r[k] = 5+c3
                KC0c[k] = 3+c4
                k += 1
                KC0r[k] = 5+c3
                KC0c[k] = 4+c4
                k += 1
                KC0r[k] = 5+c3
                KC0c[k] = 5+c4
                k += 1
                KC0r[k] = 0+c4
                KC0c[k] = 0+c1
                k += 1
                KC0r[k] = 0+c4
                KC0c[k] = 1+c1
                k += 1
                KC0r[k] = 0+c4
                KC0c[k] = 2+c1
                k += 1
                KC0r[k] = 0+c4
                KC0c[k] = 3+c1
                k += 1
                KC0r[k] = 0+c4
                KC0c[k] = 4+c1
                k += 1
                KC0r[k] = 0+c4
                KC0c[k] = 5+c1
                k += 1
                KC0r[k] = 0+c4
                KC0c[k] = 0+c2
                k += 1
                KC0r[k] = 0+c4
                KC0c[k] = 1+c2
                k += 1
                KC0r[k] = 0+c4
                KC0c[k] = 2+c2
                k += 1
                KC0r[k] = 0+c4
                KC0c[k] = 3+c2
                k += 1
                KC0r[k] = 0+c4
                KC0c[k] = 4+c2
                k += 1
                KC0r[k] = 0+c4
                KC0c[k] = 5+c2
                k += 1
                KC0r[k] = 0+c4
                KC0c[k] = 0+c3
                k += 1
                KC0r[k] = 0+c4
                KC0c[k] = 1+c3
                k += 1
                KC0r[k] = 0+c4
                KC0c[k] = 2+c3
                k += 1
                KC0r[k] = 0+c4
                KC0c[k] = 3+c3
                k += 1
                KC0r[k] = 0+c4
                KC0c[k] = 4+c3
                k += 1
                KC0r[k] = 0+c4
                KC0c[k] = 5+c3
                k += 1
                KC0r[k] = 0+c4
                KC0c[k] = 0+c4
                k += 1
                KC0r[k] = 0+c4
                KC0c[k] = 1+c4
                k += 1
                KC0r[k] = 0+c4
                KC0c[k] = 2+c4
                k += 1
                KC0r[k] = 0+c4
                KC0c[k] = 3+c4
                k += 1
                KC0r[k] = 0+c4
                KC0c[k] = 4+c4
                k += 1
                KC0r[k] = 0+c4
                KC0c[k] = 5+c4
                k += 1
                KC0r[k] = 1+c4
                KC0c[k] = 0+c1
                k += 1
                KC0r[k] = 1+c4
                KC0c[k] = 1+c1
                k += 1
                KC0r[k] = 1+c4
                KC0c[k] = 2+c1
                k += 1
                KC0r[k] = 1+c4
                KC0c[k] = 3+c1
                k += 1
                KC0r[k] = 1+c4
                KC0c[k] = 4+c1
                k += 1
                KC0r[k] = 1+c4
                KC0c[k] = 5+c1
                k += 1
                KC0r[k] = 1+c4
                KC0c[k] = 0+c2
                k += 1
                KC0r[k] = 1+c4
                KC0c[k] = 1+c2
                k += 1
                KC0r[k] = 1+c4
                KC0c[k] = 2+c2
                k += 1
                KC0r[k] = 1+c4
                KC0c[k] = 3+c2
                k += 1
                KC0r[k] = 1+c4
                KC0c[k] = 4+c2
                k += 1
                KC0r[k] = 1+c4
                KC0c[k] = 5+c2
                k += 1
                KC0r[k] = 1+c4
                KC0c[k] = 0+c3
                k += 1
                KC0r[k] = 1+c4
                KC0c[k] = 1+c3
                k += 1
                KC0r[k] = 1+c4
                KC0c[k] = 2+c3
                k += 1
                KC0r[k] = 1+c4
                KC0c[k] = 3+c3
                k += 1
                KC0r[k] = 1+c4
                KC0c[k] = 4+c3
                k += 1
                KC0r[k] = 1+c4
                KC0c[k] = 5+c3
                k += 1
                KC0r[k] = 1+c4
                KC0c[k] = 0+c4
                k += 1
                KC0r[k] = 1+c4
                KC0c[k] = 1+c4
                k += 1
                KC0r[k] = 1+c4
                KC0c[k] = 2+c4
                k += 1
                KC0r[k] = 1+c4
                KC0c[k] = 3+c4
                k += 1
                KC0r[k] = 1+c4
                KC0c[k] = 4+c4
                k += 1
                KC0r[k] = 1+c4
                KC0c[k] = 5+c4
                k += 1
                KC0r[k] = 2+c4
                KC0c[k] = 0+c1
                k += 1
                KC0r[k] = 2+c4
                KC0c[k] = 1+c1
                k += 1
                KC0r[k] = 2+c4
                KC0c[k] = 2+c1
                k += 1
                KC0r[k] = 2+c4
                KC0c[k] = 3+c1
                k += 1
                KC0r[k] = 2+c4
                KC0c[k] = 4+c1
                k += 1
                KC0r[k] = 2+c4
                KC0c[k] = 5+c1
                k += 1
                KC0r[k] = 2+c4
                KC0c[k] = 0+c2
                k += 1
                KC0r[k] = 2+c4
                KC0c[k] = 1+c2
                k += 1
                KC0r[k] = 2+c4
                KC0c[k] = 2+c2
                k += 1
                KC0r[k] = 2+c4
                KC0c[k] = 3+c2
                k += 1
                KC0r[k] = 2+c4
                KC0c[k] = 4+c2
                k += 1
                KC0r[k] = 2+c4
                KC0c[k] = 5+c2
                k += 1
                KC0r[k] = 2+c4
                KC0c[k] = 0+c3
                k += 1
                KC0r[k] = 2+c4
                KC0c[k] = 1+c3
                k += 1
                KC0r[k] = 2+c4
                KC0c[k] = 2+c3
                k += 1
                KC0r[k] = 2+c4
                KC0c[k] = 3+c3
                k += 1
                KC0r[k] = 2+c4
                KC0c[k] = 4+c3
                k += 1
                KC0r[k] = 2+c4
                KC0c[k] = 5+c3
                k += 1
                KC0r[k] = 2+c4
                KC0c[k] = 0+c4
                k += 1
                KC0r[k] = 2+c4
                KC0c[k] = 1+c4
                k += 1
                KC0r[k] = 2+c4
                KC0c[k] = 2+c4
                k += 1
                KC0r[k] = 2+c4
                KC0c[k] = 3+c4
                k += 1
                KC0r[k] = 2+c4
                KC0c[k] = 4+c4
                k += 1
                KC0r[k] = 2+c4
                KC0c[k] = 5+c4
                k += 1
                KC0r[k] = 3+c4
                KC0c[k] = 0+c1
                k += 1
                KC0r[k] = 3+c4
                KC0c[k] = 1+c1
                k += 1
                KC0r[k] = 3+c4
                KC0c[k] = 2+c1
                k += 1
                KC0r[k] = 3+c4
                KC0c[k] = 3+c1
                k += 1
                KC0r[k] = 3+c4
                KC0c[k] = 4+c1
                k += 1
                KC0r[k] = 3+c4
                KC0c[k] = 5+c1
                k += 1
                KC0r[k] = 3+c4
                KC0c[k] = 0+c2
                k += 1
                KC0r[k] = 3+c4
                KC0c[k] = 1+c2
                k += 1
                KC0r[k] = 3+c4
                KC0c[k] = 2+c2
                k += 1
                KC0r[k] = 3+c4
                KC0c[k] = 3+c2
                k += 1
                KC0r[k] = 3+c4
                KC0c[k] = 4+c2
                k += 1
                KC0r[k] = 3+c4
                KC0c[k] = 5+c2
                k += 1
                KC0r[k] = 3+c4
                KC0c[k] = 0+c3
                k += 1
                KC0r[k] = 3+c4
                KC0c[k] = 1+c3
                k += 1
                KC0r[k] = 3+c4
                KC0c[k] = 2+c3
                k += 1
                KC0r[k] = 3+c4
                KC0c[k] = 3+c3
                k += 1
                KC0r[k] = 3+c4
                KC0c[k] = 4+c3
                k += 1
                KC0r[k] = 3+c4
                KC0c[k] = 5+c3
                k += 1
                KC0r[k] = 3+c4
                KC0c[k] = 0+c4
                k += 1
                KC0r[k] = 3+c4
                KC0c[k] = 1+c4
                k += 1
                KC0r[k] = 3+c4
                KC0c[k] = 2+c4
                k += 1
                KC0r[k] = 3+c4
                KC0c[k] = 3+c4
                k += 1
                KC0r[k] = 3+c4
                KC0c[k] = 4+c4
                k += 1
                KC0r[k] = 3+c4
                KC0c[k] = 5+c4
                k += 1
                KC0r[k] = 4+c4
                KC0c[k] = 0+c1
                k += 1
                KC0r[k] = 4+c4
                KC0c[k] = 1+c1
                k += 1
                KC0r[k] = 4+c4
                KC0c[k] = 2+c1
                k += 1
                KC0r[k] = 4+c4
                KC0c[k] = 3+c1
                k += 1
                KC0r[k] = 4+c4
                KC0c[k] = 4+c1
                k += 1
                KC0r[k] = 4+c4
                KC0c[k] = 5+c1
                k += 1
                KC0r[k] = 4+c4
                KC0c[k] = 0+c2
                k += 1
                KC0r[k] = 4+c4
                KC0c[k] = 1+c2
                k += 1
                KC0r[k] = 4+c4
                KC0c[k] = 2+c2
                k += 1
                KC0r[k] = 4+c4
                KC0c[k] = 3+c2
                k += 1
                KC0r[k] = 4+c4
                KC0c[k] = 4+c2
                k += 1
                KC0r[k] = 4+c4
                KC0c[k] = 5+c2
                k += 1
                KC0r[k] = 4+c4
                KC0c[k] = 0+c3
                k += 1
                KC0r[k] = 4+c4
                KC0c[k] = 1+c3
                k += 1
                KC0r[k] = 4+c4
                KC0c[k] = 2+c3
                k += 1
                KC0r[k] = 4+c4
                KC0c[k] = 3+c3
                k += 1
                KC0r[k] = 4+c4
                KC0c[k] = 4+c3
                k += 1
                KC0r[k] = 4+c4
                KC0c[k] = 5+c3
                k += 1
                KC0r[k] = 4+c4
                KC0c[k] = 0+c4
                k += 1
                KC0r[k] = 4+c4
                KC0c[k] = 1+c4
                k += 1
                KC0r[k] = 4+c4
                KC0c[k] = 2+c4
                k += 1
                KC0r[k] = 4+c4
                KC0c[k] = 3+c4
                k += 1
                KC0r[k] = 4+c4
                KC0c[k] = 4+c4
                k += 1
                KC0r[k] = 4+c4
                KC0c[k] = 5+c4
                k += 1
                KC0r[k] = 5+c4
                KC0c[k] = 0+c1
                k += 1
                KC0r[k] = 5+c4
                KC0c[k] = 1+c1
                k += 1
                KC0r[k] = 5+c4
                KC0c[k] = 2+c1
                k += 1
                KC0r[k] = 5+c4
                KC0c[k] = 3+c1
                k += 1
                KC0r[k] = 5+c4
                KC0c[k] = 4+c1
                k += 1
                KC0r[k] = 5+c4
                KC0c[k] = 5+c1
                k += 1
                KC0r[k] = 5+c4
                KC0c[k] = 0+c2
                k += 1
                KC0r[k] = 5+c4
                KC0c[k] = 1+c2
                k += 1
                KC0r[k] = 5+c4
                KC0c[k] = 2+c2
                k += 1
                KC0r[k] = 5+c4
                KC0c[k] = 3+c2
                k += 1
                KC0r[k] = 5+c4
                KC0c[k] = 4+c2
                k += 1
                KC0r[k] = 5+c4
                KC0c[k] = 5+c2
                k += 1
                KC0r[k] = 5+c4
                KC0c[k] = 0+c3
                k += 1
                KC0r[k] = 5+c4
                KC0c[k] = 1+c3
                k += 1
                KC0r[k] = 5+c4
                KC0c[k] = 2+c3
                k += 1
                KC0r[k] = 5+c4
                KC0c[k] = 3+c3
                k += 1
                KC0r[k] = 5+c4
                KC0c[k] = 4+c3
                k += 1
                KC0r[k] = 5+c4
                KC0c[k] = 5+c3
                k += 1
                KC0r[k] = 5+c4
                KC0c[k] = 0+c4
                k += 1
                KC0r[k] = 5+c4
                KC0c[k] = 1+c4
                k += 1
                KC0r[k] = 5+c4
                KC0c[k] = 2+c4
                k += 1
                KC0r[k] = 5+c4
                KC0c[k] = 3+c4
                k += 1
                KC0r[k] = 5+c4
                KC0c[k] = 4+c4
                k += 1
                KC0r[k] = 5+c4
                KC0c[k] = 5+c4

            # NOTE reduced integration to remove shear locking
            wij = 4.

            # NOTE Brockman's hourglass control of u and v, derived for
            #      isotropic plates, applied to laminates with the membrane
            #      moduli E1eq and E2eq, and with the area in place of his
            #      1/(1 + 1/A) of Eq. (16), see the section "Hourglass
            #      control of the membrane" of the module documentation: the
            #      term enters as K_ij = A*E^(h)*gamma_i*gamma_j with gamma
            #      of dimension 1/L**2, so E^(h) must be F*L
            Eu = hgfactor_membrane*0.1*E1eq*prop.h*self.area
            Ev = hgfactor_membrane*0.1*E2eq*prop.h*self.area
            # NOTE w, rx and ry have no artificial hourglass stiffness: their
            #      hourglass modes are controlled by the physical
            #      stabilisation of the bending and of the transverse shear
            #      below, scaled by hgfactor_physical, see _shear_bl94 and
            #      _bending_bl94

            # NOTE using only one integration point at xi=0, eta=0 to avoid shear locking
            detJ = 0.125*x1*y2 - 0.125*x1*y4 - 0.125*x2*y1 + 0.125*x2*y3 - 0.125*x3*y2 + 0.125*x3*y4 + 0.125*x4*y1 - 0.125*x4*y3

            j11 = 2.0*(-y1 - y2 + y3 + y4)/(x1*y2 - x1*y4 - x2*y1 + x2*y3 - x3*y2 + x3*y4 + x4*y1 - x4*y3)
            j12 = 2.0*(y1 - y2 - y3 + y4)/(x1*y2 - x1*y4 - x2*y1 + x2*y3 - x3*y2 + x3*y4 + x4*y1 - x4*y3)
            j21 = 2.0*(x1 + x2 - x3 - x4)/(x1*y2 - x1*y4 - x2*y1 + x2*y3 - x3*y2 + x3*y4 + x4*y1 - x4*y3)
            j22 = 2.0*(-x1 + x2 + x3 - x4)/(x1*y2 - x1*y4 - x2*y1 + x2*y3 - x3*y2 + x3*y4 + x4*y1 - x4*y3)

            S1 = 0.25
            S2 = 0.25
            S3 = 0.25
            S4 = 0.25

            S1_x = -0.25*j11 - 0.25*j12
            S2_x = 0.25*j11 - 0.25*j12
            S3_x = 0.25*j11 + 0.25*j12
            S4_x = -0.25*j11 + 0.25*j12
            S1_y = -0.25*j21 - 0.25*j22
            S2_y = 0.25*j21 - 0.25*j22
            S3_y = 0.25*j21 + 0.25*j22
            S4_y = -0.25*j21 + 0.25*j22

            S1_xy = 0.25*j11*j22 + 0.25*j12*j21
            S2_xy = -0.25*j11*j22 - 0.25*j12*j21
            S3_xy = 0.25*j11*j22 + 0.25*j12*j21
            S4_xy = -0.25*j11*j22 - 0.25*j12*j21

            # NOTE hourglass operator of Eq. (14) of Brockman 1987, quoted
            #      there from Belytschko and Tsay 1983,
            #
            #          gamma = h - (h.x)*b1 - (h.y)*b2
            #
            #      with h = [1, -1, 1, -1], b1 and b2 the centroidal
            #      derivatives N,x and N,y, scaled by the normalisation of
            #      Brockman's own Eq. (15), gamma = d2N/dxdy at the centroid,
            #      so that the generalized stiffnesses of his Eq. (16) keep
            #      their calibration. The two correction terms are, in his
            #      words, "important for irregular elements, if the hourglass
            #      strains are to vanish in the presence of rigid-body motion
            #      and uniform strain". They vanish identically whenever the
            #      element is a parallelogram, h.x and h.y being zero there,
            #      so this leaves every regular mesh bit-identical, and they
            #      do not change the hourglass energy of the h patterns on
            #      any shape either, because the centroidal sum h.b1 = h.b2
            #      is zero. See tests/test_quad4r_hourglass_control.py
            hgx = x1 - x2 + x3 - x4
            hgy = y1 - y2 + y3 - y4
            gamma1 = S1_xy*(1. - hgx*S1_x - hgy*S1_y)
            gamma2 = S2_xy*(1. + hgx*S2_x + hgy*S2_y)
            gamma3 = S3_xy*(1. - hgx*S3_x - hgy*S3_y)
            gamma4 = S4_xy*(1. + hgx*S4_x + hgy*S4_y)

            KC0e0000 = detJ*wij*(Eu*gamma1**2 + S1_x*(A11*S1_x + A16*S1_y) + S1_y*(A16*S1_x + A66*S1_y))
            KC0e0001 = detJ*wij*(S1_x*(A16*S1_x + A66*S1_y) + S1_y*(A12*S1_x + A26*S1_y))
            KC0e0003 = -detJ*wij*(S1_x*(B16*S1_x + B66*S1_y) + S1_y*(B12*S1_x + B26*S1_y))
            KC0e0004 = detJ*wij*(S1_x*(B11*S1_x + B16*S1_y) + S1_y*(B16*S1_x + B66*S1_y))
            KC0e0006 = detJ*wij*(Eu*gamma1*gamma2 + S2_x*(A11*S1_x + A16*S1_y) + S2_y*(A16*S1_x + A66*S1_y))
            KC0e0007 = detJ*wij*(S2_x*(A16*S1_x + A66*S1_y) + S2_y*(A12*S1_x + A26*S1_y))
            KC0e0009 = -detJ*wij*(S2_x*(B16*S1_x + B66*S1_y) + S2_y*(B12*S1_x + B26*S1_y))
            KC0e0010 = detJ*wij*(S2_x*(B11*S1_x + B16*S1_y) + S2_y*(B16*S1_x + B66*S1_y))
            KC0e0012 = detJ*wij*(Eu*gamma1*gamma3 + S3_x*(A11*S1_x + A16*S1_y) + S3_y*(A16*S1_x + A66*S1_y))
            KC0e0013 = detJ*wij*(S3_x*(A16*S1_x + A66*S1_y) + S3_y*(A12*S1_x + A26*S1_y))
            KC0e0015 = -detJ*wij*(S3_x*(B16*S1_x + B66*S1_y) + S3_y*(B12*S1_x + B26*S1_y))
            KC0e0016 = detJ*wij*(S3_x*(B11*S1_x + B16*S1_y) + S3_y*(B16*S1_x + B66*S1_y))
            KC0e0018 = detJ*wij*(Eu*gamma1*gamma4 + S4_x*(A11*S1_x + A16*S1_y) + S4_y*(A16*S1_x + A66*S1_y))
            KC0e0019 = detJ*wij*(S4_x*(A16*S1_x + A66*S1_y) + S4_y*(A12*S1_x + A26*S1_y))
            KC0e0021 = -detJ*wij*(S4_x*(B16*S1_x + B66*S1_y) + S4_y*(B12*S1_x + B26*S1_y))
            KC0e0022 = detJ*wij*(S4_x*(B11*S1_x + B16*S1_y) + S4_y*(B16*S1_x + B66*S1_y))
            KC0e0101 = detJ*wij*(Ev*gamma1**2 + S1_x*(A26*S1_y + A66*S1_x) + S1_y*(A22*S1_y + A26*S1_x))
            KC0e0103 = -detJ*wij*(S1_x*(B26*S1_y + B66*S1_x) + S1_y*(B22*S1_y + B26*S1_x))
            KC0e0104 = detJ*wij*(S1_x*(B12*S1_y + B16*S1_x) + S1_y*(B26*S1_y + B66*S1_x))
            KC0e0106 = detJ*wij*(S2_x*(A12*S1_y + A16*S1_x) + S2_y*(A26*S1_y + A66*S1_x))
            KC0e0107 = detJ*wij*(Ev*gamma1*gamma2 + S2_x*(A26*S1_y + A66*S1_x) + S2_y*(A22*S1_y + A26*S1_x))
            KC0e0109 = -detJ*wij*(S2_x*(B26*S1_y + B66*S1_x) + S2_y*(B22*S1_y + B26*S1_x))
            KC0e0110 = detJ*wij*(S2_x*(B12*S1_y + B16*S1_x) + S2_y*(B26*S1_y + B66*S1_x))
            KC0e0112 = detJ*wij*(S3_x*(A12*S1_y + A16*S1_x) + S3_y*(A26*S1_y + A66*S1_x))
            KC0e0113 = detJ*wij*(Ev*gamma1*gamma3 + S3_x*(A26*S1_y + A66*S1_x) + S3_y*(A22*S1_y + A26*S1_x))
            KC0e0115 = -detJ*wij*(S3_x*(B26*S1_y + B66*S1_x) + S3_y*(B22*S1_y + B26*S1_x))
            KC0e0116 = detJ*wij*(S3_x*(B12*S1_y + B16*S1_x) + S3_y*(B26*S1_y + B66*S1_x))
            KC0e0118 = detJ*wij*(S4_x*(A12*S1_y + A16*S1_x) + S4_y*(A26*S1_y + A66*S1_x))
            KC0e0119 = detJ*wij*(Ev*gamma1*gamma4 + S4_x*(A26*S1_y + A66*S1_x) + S4_y*(A22*S1_y + A26*S1_x))
            KC0e0121 = -detJ*wij*(S4_x*(B26*S1_y + B66*S1_x) + S4_y*(B22*S1_y + B26*S1_x))
            KC0e0122 = detJ*wij*(S4_x*(B12*S1_y + B16*S1_x) + S4_y*(B26*S1_y + B66*S1_x))
            KC0e0202 = detJ*wij*(S1_x*(A45*S1_y + A55*S1_x) + S1_y*(A44*S1_y + A45*S1_x))
            KC0e0203 = -S1*detJ*wij*(A44*S1_y + A45*S1_x)
            KC0e0204 = S1*detJ*wij*(A45*S1_y + A55*S1_x)
            KC0e0208 = detJ*wij*(S2_x*(A45*S1_y + A55*S1_x) + S2_y*(A44*S1_y + A45*S1_x))
            KC0e0209 = -S2*detJ*wij*(A44*S1_y + A45*S1_x)
            KC0e0210 = S2*detJ*wij*(A45*S1_y + A55*S1_x)
            KC0e0214 = detJ*wij*(S3_x*(A45*S1_y + A55*S1_x) + S3_y*(A44*S1_y + A45*S1_x))
            KC0e0215 = -S3*detJ*wij*(A44*S1_y + A45*S1_x)
            KC0e0216 = S3*detJ*wij*(A45*S1_y + A55*S1_x)
            KC0e0220 = detJ*wij*(S4_x*(A45*S1_y + A55*S1_x) + S4_y*(A44*S1_y + A45*S1_x))
            KC0e0221 = -S4*detJ*wij*(A44*S1_y + A45*S1_x)
            KC0e0222 = S4*detJ*wij*(A45*S1_y + A55*S1_x)
            KC0e0303 = detJ*wij*(A44*S1**2 + S1_x*(D26*S1_y + D66*S1_x) + S1_y*(D22*S1_y + D26*S1_x))
            KC0e0304 = -detJ*wij*(A45*S1**2 + S1_x*(D12*S1_y + D16*S1_x) + S1_y*(D26*S1_y + D66*S1_x))
            KC0e0306 = -detJ*wij*(S2_x*(B12*S1_y + B16*S1_x) + S2_y*(B26*S1_y + B66*S1_x))
            KC0e0307 = -detJ*wij*(S2_x*(B26*S1_y + B66*S1_x) + S2_y*(B22*S1_y + B26*S1_x))
            KC0e0308 = -S1*detJ*wij*(A44*S2_y + A45*S2_x)
            KC0e0309 = detJ*wij*(A44*S1*S2 + S2_x*(D26*S1_y + D66*S1_x) + S2_y*(D22*S1_y + D26*S1_x))
            KC0e0310 = -detJ*wij*(A45*S1*S2 + S2_x*(D12*S1_y + D16*S1_x) + S2_y*(D26*S1_y + D66*S1_x))
            KC0e0312 = -detJ*wij*(S3_x*(B12*S1_y + B16*S1_x) + S3_y*(B26*S1_y + B66*S1_x))
            KC0e0313 = -detJ*wij*(S3_x*(B26*S1_y + B66*S1_x) + S3_y*(B22*S1_y + B26*S1_x))
            KC0e0314 = -S1*detJ*wij*(A44*S3_y + A45*S3_x)
            KC0e0315 = detJ*wij*(A44*S1*S3 + S3_x*(D26*S1_y + D66*S1_x) + S3_y*(D22*S1_y + D26*S1_x))
            KC0e0316 = -detJ*wij*(A45*S1*S3 + S3_x*(D12*S1_y + D16*S1_x) + S3_y*(D26*S1_y + D66*S1_x))
            KC0e0318 = -detJ*wij*(S4_x*(B12*S1_y + B16*S1_x) + S4_y*(B26*S1_y + B66*S1_x))
            KC0e0319 = -detJ*wij*(S4_x*(B26*S1_y + B66*S1_x) + S4_y*(B22*S1_y + B26*S1_x))
            KC0e0320 = -S1*detJ*wij*(A44*S4_y + A45*S4_x)
            KC0e0321 = detJ*wij*(A44*S1*S4 + S4_x*(D26*S1_y + D66*S1_x) + S4_y*(D22*S1_y + D26*S1_x))
            KC0e0322 = -detJ*wij*(A45*S1*S4 + S4_x*(D12*S1_y + D16*S1_x) + S4_y*(D26*S1_y + D66*S1_x))
            KC0e0404 = detJ*wij*(A55*S1**2 + S1_x*(D11*S1_x + D16*S1_y) + S1_y*(D16*S1_x + D66*S1_y))
            KC0e0406 = detJ*wij*(S2_x*(B11*S1_x + B16*S1_y) + S2_y*(B16*S1_x + B66*S1_y))
            KC0e0407 = detJ*wij*(S2_x*(B16*S1_x + B66*S1_y) + S2_y*(B12*S1_x + B26*S1_y))
            KC0e0408 = S1*detJ*wij*(A45*S2_y + A55*S2_x)
            KC0e0409 = -detJ*wij*(A45*S1*S2 + S2_x*(D16*S1_x + D66*S1_y) + S2_y*(D12*S1_x + D26*S1_y))
            KC0e0410 = detJ*wij*(A55*S1*S2 + S2_x*(D11*S1_x + D16*S1_y) + S2_y*(D16*S1_x + D66*S1_y))
            KC0e0412 = detJ*wij*(S3_x*(B11*S1_x + B16*S1_y) + S3_y*(B16*S1_x + B66*S1_y))
            KC0e0413 = detJ*wij*(S3_x*(B16*S1_x + B66*S1_y) + S3_y*(B12*S1_x + B26*S1_y))
            KC0e0414 = S1*detJ*wij*(A45*S3_y + A55*S3_x)
            KC0e0415 = -detJ*wij*(A45*S1*S3 + S3_x*(D16*S1_x + D66*S1_y) + S3_y*(D12*S1_x + D26*S1_y))
            KC0e0416 = detJ*wij*(A55*S1*S3 + S3_x*(D11*S1_x + D16*S1_y) + S3_y*(D16*S1_x + D66*S1_y))
            KC0e0418 = detJ*wij*(S4_x*(B11*S1_x + B16*S1_y) + S4_y*(B16*S1_x + B66*S1_y))
            KC0e0419 = detJ*wij*(S4_x*(B16*S1_x + B66*S1_y) + S4_y*(B12*S1_x + B26*S1_y))
            KC0e0420 = S1*detJ*wij*(A45*S4_y + A55*S4_x)
            KC0e0421 = -detJ*wij*(A45*S1*S4 + S4_x*(D16*S1_x + D66*S1_y) + S4_y*(D12*S1_x + D26*S1_y))
            KC0e0422 = detJ*wij*(A55*S1*S4 + S4_x*(D11*S1_x + D16*S1_y) + S4_y*(D16*S1_x + D66*S1_y))
            KC0e0606 = detJ*wij*(Eu*gamma2**2 + S2_x*(A11*S2_x + A16*S2_y) + S2_y*(A16*S2_x + A66*S2_y))
            KC0e0607 = detJ*wij*(S2_x*(A16*S2_x + A66*S2_y) + S2_y*(A12*S2_x + A26*S2_y))
            KC0e0609 = -detJ*wij*(S2_x*(B16*S2_x + B66*S2_y) + S2_y*(B12*S2_x + B26*S2_y))
            KC0e0610 = detJ*wij*(S2_x*(B11*S2_x + B16*S2_y) + S2_y*(B16*S2_x + B66*S2_y))
            KC0e0612 = detJ*wij*(Eu*gamma2*gamma3 + S3_x*(A11*S2_x + A16*S2_y) + S3_y*(A16*S2_x + A66*S2_y))
            KC0e0613 = detJ*wij*(S3_x*(A16*S2_x + A66*S2_y) + S3_y*(A12*S2_x + A26*S2_y))
            KC0e0615 = -detJ*wij*(S3_x*(B16*S2_x + B66*S2_y) + S3_y*(B12*S2_x + B26*S2_y))
            KC0e0616 = detJ*wij*(S3_x*(B11*S2_x + B16*S2_y) + S3_y*(B16*S2_x + B66*S2_y))
            KC0e0618 = detJ*wij*(Eu*gamma2*gamma4 + S4_x*(A11*S2_x + A16*S2_y) + S4_y*(A16*S2_x + A66*S2_y))
            KC0e0619 = detJ*wij*(S4_x*(A16*S2_x + A66*S2_y) + S4_y*(A12*S2_x + A26*S2_y))
            KC0e0621 = -detJ*wij*(S4_x*(B16*S2_x + B66*S2_y) + S4_y*(B12*S2_x + B26*S2_y))
            KC0e0622 = detJ*wij*(S4_x*(B11*S2_x + B16*S2_y) + S4_y*(B16*S2_x + B66*S2_y))
            KC0e0707 = detJ*wij*(Ev*gamma2**2 + S2_x*(A26*S2_y + A66*S2_x) + S2_y*(A22*S2_y + A26*S2_x))
            KC0e0709 = -detJ*wij*(S2_x*(B26*S2_y + B66*S2_x) + S2_y*(B22*S2_y + B26*S2_x))
            KC0e0710 = detJ*wij*(S2_x*(B12*S2_y + B16*S2_x) + S2_y*(B26*S2_y + B66*S2_x))
            KC0e0712 = detJ*wij*(S3_x*(A12*S2_y + A16*S2_x) + S3_y*(A26*S2_y + A66*S2_x))
            KC0e0713 = detJ*wij*(Ev*gamma2*gamma3 + S3_x*(A26*S2_y + A66*S2_x) + S3_y*(A22*S2_y + A26*S2_x))
            KC0e0715 = -detJ*wij*(S3_x*(B26*S2_y + B66*S2_x) + S3_y*(B22*S2_y + B26*S2_x))
            KC0e0716 = detJ*wij*(S3_x*(B12*S2_y + B16*S2_x) + S3_y*(B26*S2_y + B66*S2_x))
            KC0e0718 = detJ*wij*(S4_x*(A12*S2_y + A16*S2_x) + S4_y*(A26*S2_y + A66*S2_x))
            KC0e0719 = detJ*wij*(Ev*gamma2*gamma4 + S4_x*(A26*S2_y + A66*S2_x) + S4_y*(A22*S2_y + A26*S2_x))
            KC0e0721 = -detJ*wij*(S4_x*(B26*S2_y + B66*S2_x) + S4_y*(B22*S2_y + B26*S2_x))
            KC0e0722 = detJ*wij*(S4_x*(B12*S2_y + B16*S2_x) + S4_y*(B26*S2_y + B66*S2_x))
            KC0e0808 = detJ*wij*(S2_x*(A45*S2_y + A55*S2_x) + S2_y*(A44*S2_y + A45*S2_x))
            KC0e0809 = -S2*detJ*wij*(A44*S2_y + A45*S2_x)
            KC0e0810 = S2*detJ*wij*(A45*S2_y + A55*S2_x)
            KC0e0814 = detJ*wij*(S3_x*(A45*S2_y + A55*S2_x) + S3_y*(A44*S2_y + A45*S2_x))
            KC0e0815 = -S3*detJ*wij*(A44*S2_y + A45*S2_x)
            KC0e0816 = S3*detJ*wij*(A45*S2_y + A55*S2_x)
            KC0e0820 = detJ*wij*(S4_x*(A45*S2_y + A55*S2_x) + S4_y*(A44*S2_y + A45*S2_x))
            KC0e0821 = -S4*detJ*wij*(A44*S2_y + A45*S2_x)
            KC0e0822 = S4*detJ*wij*(A45*S2_y + A55*S2_x)
            KC0e0909 = detJ*wij*(A44*S2**2 + S2_x*(D26*S2_y + D66*S2_x) + S2_y*(D22*S2_y + D26*S2_x))
            KC0e0910 = -detJ*wij*(A45*S2**2 + S2_x*(D12*S2_y + D16*S2_x) + S2_y*(D26*S2_y + D66*S2_x))
            KC0e0912 = -detJ*wij*(S3_x*(B12*S2_y + B16*S2_x) + S3_y*(B26*S2_y + B66*S2_x))
            KC0e0913 = -detJ*wij*(S3_x*(B26*S2_y + B66*S2_x) + S3_y*(B22*S2_y + B26*S2_x))
            KC0e0914 = -S2*detJ*wij*(A44*S3_y + A45*S3_x)
            KC0e0915 = detJ*wij*(A44*S2*S3 + S3_x*(D26*S2_y + D66*S2_x) + S3_y*(D22*S2_y + D26*S2_x))
            KC0e0916 = -detJ*wij*(A45*S2*S3 + S3_x*(D12*S2_y + D16*S2_x) + S3_y*(D26*S2_y + D66*S2_x))
            KC0e0918 = -detJ*wij*(S4_x*(B12*S2_y + B16*S2_x) + S4_y*(B26*S2_y + B66*S2_x))
            KC0e0919 = -detJ*wij*(S4_x*(B26*S2_y + B66*S2_x) + S4_y*(B22*S2_y + B26*S2_x))
            KC0e0920 = -S2*detJ*wij*(A44*S4_y + A45*S4_x)
            KC0e0921 = detJ*wij*(A44*S2*S4 + S4_x*(D26*S2_y + D66*S2_x) + S4_y*(D22*S2_y + D26*S2_x))
            KC0e0922 = -detJ*wij*(A45*S2*S4 + S4_x*(D12*S2_y + D16*S2_x) + S4_y*(D26*S2_y + D66*S2_x))
            KC0e1010 = detJ*wij*(A55*S2**2 + S2_x*(D11*S2_x + D16*S2_y) + S2_y*(D16*S2_x + D66*S2_y))
            KC0e1012 = detJ*wij*(S3_x*(B11*S2_x + B16*S2_y) + S3_y*(B16*S2_x + B66*S2_y))
            KC0e1013 = detJ*wij*(S3_x*(B16*S2_x + B66*S2_y) + S3_y*(B12*S2_x + B26*S2_y))
            KC0e1014 = S2*detJ*wij*(A45*S3_y + A55*S3_x)
            KC0e1015 = -detJ*wij*(A45*S2*S3 + S3_x*(D16*S2_x + D66*S2_y) + S3_y*(D12*S2_x + D26*S2_y))
            KC0e1016 = detJ*wij*(A55*S2*S3 + S3_x*(D11*S2_x + D16*S2_y) + S3_y*(D16*S2_x + D66*S2_y))
            KC0e1018 = detJ*wij*(S4_x*(B11*S2_x + B16*S2_y) + S4_y*(B16*S2_x + B66*S2_y))
            KC0e1019 = detJ*wij*(S4_x*(B16*S2_x + B66*S2_y) + S4_y*(B12*S2_x + B26*S2_y))
            KC0e1020 = S2*detJ*wij*(A45*S4_y + A55*S4_x)
            KC0e1021 = -detJ*wij*(A45*S2*S4 + S4_x*(D16*S2_x + D66*S2_y) + S4_y*(D12*S2_x + D26*S2_y))
            KC0e1022 = detJ*wij*(A55*S2*S4 + S4_x*(D11*S2_x + D16*S2_y) + S4_y*(D16*S2_x + D66*S2_y))
            KC0e1212 = detJ*wij*(Eu*gamma3**2 + S3_x*(A11*S3_x + A16*S3_y) + S3_y*(A16*S3_x + A66*S3_y))
            KC0e1213 = detJ*wij*(S3_x*(A16*S3_x + A66*S3_y) + S3_y*(A12*S3_x + A26*S3_y))
            KC0e1215 = -detJ*wij*(S3_x*(B16*S3_x + B66*S3_y) + S3_y*(B12*S3_x + B26*S3_y))
            KC0e1216 = detJ*wij*(S3_x*(B11*S3_x + B16*S3_y) + S3_y*(B16*S3_x + B66*S3_y))
            KC0e1218 = detJ*wij*(Eu*gamma3*gamma4 + S4_x*(A11*S3_x + A16*S3_y) + S4_y*(A16*S3_x + A66*S3_y))
            KC0e1219 = detJ*wij*(S4_x*(A16*S3_x + A66*S3_y) + S4_y*(A12*S3_x + A26*S3_y))
            KC0e1221 = -detJ*wij*(S4_x*(B16*S3_x + B66*S3_y) + S4_y*(B12*S3_x + B26*S3_y))
            KC0e1222 = detJ*wij*(S4_x*(B11*S3_x + B16*S3_y) + S4_y*(B16*S3_x + B66*S3_y))
            KC0e1313 = detJ*wij*(Ev*gamma3**2 + S3_x*(A26*S3_y + A66*S3_x) + S3_y*(A22*S3_y + A26*S3_x))
            KC0e1315 = -detJ*wij*(S3_x*(B26*S3_y + B66*S3_x) + S3_y*(B22*S3_y + B26*S3_x))
            KC0e1316 = detJ*wij*(S3_x*(B12*S3_y + B16*S3_x) + S3_y*(B26*S3_y + B66*S3_x))
            KC0e1318 = detJ*wij*(S4_x*(A12*S3_y + A16*S3_x) + S4_y*(A26*S3_y + A66*S3_x))
            KC0e1319 = detJ*wij*(Ev*gamma3*gamma4 + S4_x*(A26*S3_y + A66*S3_x) + S4_y*(A22*S3_y + A26*S3_x))
            KC0e1321 = -detJ*wij*(S4_x*(B26*S3_y + B66*S3_x) + S4_y*(B22*S3_y + B26*S3_x))
            KC0e1322 = detJ*wij*(S4_x*(B12*S3_y + B16*S3_x) + S4_y*(B26*S3_y + B66*S3_x))
            KC0e1414 = detJ*wij*(S3_x*(A45*S3_y + A55*S3_x) + S3_y*(A44*S3_y + A45*S3_x))
            KC0e1415 = -S3*detJ*wij*(A44*S3_y + A45*S3_x)
            KC0e1416 = S3*detJ*wij*(A45*S3_y + A55*S3_x)
            KC0e1420 = detJ*wij*(S4_x*(A45*S3_y + A55*S3_x) + S4_y*(A44*S3_y + A45*S3_x))
            KC0e1421 = -S4*detJ*wij*(A44*S3_y + A45*S3_x)
            KC0e1422 = S4*detJ*wij*(A45*S3_y + A55*S3_x)
            KC0e1515 = detJ*wij*(A44*S3**2 + S3_x*(D26*S3_y + D66*S3_x) + S3_y*(D22*S3_y + D26*S3_x))
            KC0e1516 = -detJ*wij*(A45*S3**2 + S3_x*(D12*S3_y + D16*S3_x) + S3_y*(D26*S3_y + D66*S3_x))
            KC0e1518 = -detJ*wij*(S4_x*(B12*S3_y + B16*S3_x) + S4_y*(B26*S3_y + B66*S3_x))
            KC0e1519 = -detJ*wij*(S4_x*(B26*S3_y + B66*S3_x) + S4_y*(B22*S3_y + B26*S3_x))
            KC0e1520 = -S3*detJ*wij*(A44*S4_y + A45*S4_x)
            KC0e1521 = detJ*wij*(A44*S3*S4 + S4_x*(D26*S3_y + D66*S3_x) + S4_y*(D22*S3_y + D26*S3_x))
            KC0e1522 = -detJ*wij*(A45*S3*S4 + S4_x*(D12*S3_y + D16*S3_x) + S4_y*(D26*S3_y + D66*S3_x))
            KC0e1616 = detJ*wij*(A55*S3**2 + S3_x*(D11*S3_x + D16*S3_y) + S3_y*(D16*S3_x + D66*S3_y))
            KC0e1618 = detJ*wij*(S4_x*(B11*S3_x + B16*S3_y) + S4_y*(B16*S3_x + B66*S3_y))
            KC0e1619 = detJ*wij*(S4_x*(B16*S3_x + B66*S3_y) + S4_y*(B12*S3_x + B26*S3_y))
            KC0e1620 = S3*detJ*wij*(A45*S4_y + A55*S4_x)
            KC0e1621 = -detJ*wij*(A45*S3*S4 + S4_x*(D16*S3_x + D66*S3_y) + S4_y*(D12*S3_x + D26*S3_y))
            KC0e1622 = detJ*wij*(A55*S3*S4 + S4_x*(D11*S3_x + D16*S3_y) + S4_y*(D16*S3_x + D66*S3_y))
            KC0e1818 = detJ*wij*(Eu*gamma4**2 + S4_x*(A11*S4_x + A16*S4_y) + S4_y*(A16*S4_x + A66*S4_y))
            KC0e1819 = detJ*wij*(S4_x*(A16*S4_x + A66*S4_y) + S4_y*(A12*S4_x + A26*S4_y))
            KC0e1821 = -detJ*wij*(S4_x*(B16*S4_x + B66*S4_y) + S4_y*(B12*S4_x + B26*S4_y))
            KC0e1822 = detJ*wij*(S4_x*(B11*S4_x + B16*S4_y) + S4_y*(B16*S4_x + B66*S4_y))
            KC0e1919 = detJ*wij*(Ev*gamma4**2 + S4_x*(A26*S4_y + A66*S4_x) + S4_y*(A22*S4_y + A26*S4_x))
            KC0e1921 = -detJ*wij*(S4_x*(B26*S4_y + B66*S4_x) + S4_y*(B22*S4_y + B26*S4_x))
            KC0e1922 = detJ*wij*(S4_x*(B12*S4_y + B16*S4_x) + S4_y*(B26*S4_y + B66*S4_x))
            KC0e2020 = detJ*wij*(S4_x*(A45*S4_y + A55*S4_x) + S4_y*(A44*S4_y + A45*S4_x))
            KC0e2021 = -S4*detJ*wij*(A44*S4_y + A45*S4_x)
            KC0e2022 = S4*detJ*wij*(A45*S4_y + A55*S4_x)
            KC0e2121 = detJ*wij*(A44*S4**2 + S4_x*(D26*S4_y + D66*S4_x) + S4_y*(D22*S4_y + D26*S4_x))
            KC0e2122 = -detJ*wij*(A45*S4**2 + S4_x*(D12*S4_y + D16*S4_x) + S4_y*(D26*S4_y + D66*S4_x))
            KC0e2222 = detJ*wij*(A55*S4**2 + S4_x*(D11*S4_x + D16*S4_y) + S4_y*(D16*S4_x + D66*S4_y))
            # NOTE transverse shear of Belytschko and Leviathan (1994), the
            #      constant part of the assumed strains of Dvorkin and Bathe
            #      (1984) instead of the displacement-based centroid value, and
            #      the physical stabilisation of their nonconstant part,
            #      multiplied by hgfactor_physical, see the module documentation
            _shear_bl94(x1, x2, x3, x4, y1, y2, y3, y4, A44, A45, A55,
                        hgfactor_physical, Kws)
            # NOTE physical stabilisation of the bending of Belytschko and
            #      Leviathan (1994), Eq. (B.18), multiplied by
            #      hgfactor_physical
            _bending_bl94(x1, x2, x3, x4, y1, y2, y3, y4, Ds,
                          hgfactor_physical, Kws)
            KC0e0202 += Kws[0]
            KC0e0203 += Kws[1]
            KC0e0204 += Kws[2]
            KC0e0208 += Kws[3]
            KC0e0209 += Kws[4]
            KC0e0210 += Kws[5]
            KC0e0214 += Kws[6]
            KC0e0215 += Kws[7]
            KC0e0216 += Kws[8]
            KC0e0220 += Kws[9]
            KC0e0221 += Kws[10]
            KC0e0222 += Kws[11]
            KC0e0303 += Kws[13]
            KC0e0304 += Kws[14]
            KC0e0308 += Kws[15]
            KC0e0309 += Kws[16]
            KC0e0310 += Kws[17]
            KC0e0314 += Kws[18]
            KC0e0315 += Kws[19]
            KC0e0316 += Kws[20]
            KC0e0320 += Kws[21]
            KC0e0321 += Kws[22]
            KC0e0322 += Kws[23]
            KC0e0404 += Kws[26]
            KC0e0408 += Kws[27]
            KC0e0409 += Kws[28]
            KC0e0410 += Kws[29]
            KC0e0414 += Kws[30]
            KC0e0415 += Kws[31]
            KC0e0416 += Kws[32]
            KC0e0420 += Kws[33]
            KC0e0421 += Kws[34]
            KC0e0422 += Kws[35]
            KC0e0808 += Kws[39]
            KC0e0809 += Kws[40]
            KC0e0810 += Kws[41]
            KC0e0814 += Kws[42]
            KC0e0815 += Kws[43]
            KC0e0816 += Kws[44]
            KC0e0820 += Kws[45]
            KC0e0821 += Kws[46]
            KC0e0822 += Kws[47]
            KC0e0909 += Kws[52]
            KC0e0910 += Kws[53]
            KC0e0914 += Kws[54]
            KC0e0915 += Kws[55]
            KC0e0916 += Kws[56]
            KC0e0920 += Kws[57]
            KC0e0921 += Kws[58]
            KC0e0922 += Kws[59]
            KC0e1010 += Kws[65]
            KC0e1014 += Kws[66]
            KC0e1015 += Kws[67]
            KC0e1016 += Kws[68]
            KC0e1020 += Kws[69]
            KC0e1021 += Kws[70]
            KC0e1022 += Kws[71]
            KC0e1414 += Kws[78]
            KC0e1415 += Kws[79]
            KC0e1416 += Kws[80]
            KC0e1420 += Kws[81]
            KC0e1421 += Kws[82]
            KC0e1422 += Kws[83]
            KC0e1515 += Kws[91]
            KC0e1516 += Kws[92]
            KC0e1520 += Kws[93]
            KC0e1521 += Kws[94]
            KC0e1522 += Kws[95]
            KC0e1616 += Kws[104]
            KC0e1620 += Kws[105]
            KC0e1621 += Kws[106]
            KC0e1622 += Kws[107]
            KC0e2020 += Kws[117]
            KC0e2021 += Kws[118]
            KC0e2022 += Kws[119]
            KC0e2121 += Kws[130]
            KC0e2122 += Kws[131]
            KC0e2222 += Kws[143]

            # NOTE drilling rotation uses full integration with two-point Gauss-Legendre quadrature
            wij = 1.
            points[0] = -0.5773502691896257645092
            points[1] = +0.5773502691896257645092

            KC0e0005 = 0
            KC0e0011 = 0
            KC0e0017 = 0
            KC0e0023 = 0
            KC0e0105 = 0
            KC0e0111 = 0
            KC0e0117 = 0
            KC0e0123 = 0
            KC0e0505 = 0
            KC0e0506 = 0
            KC0e0507 = 0
            KC0e0511 = 0
            KC0e0512 = 0
            KC0e0513 = 0
            KC0e0517 = 0
            KC0e0518 = 0
            KC0e0519 = 0
            KC0e0523 = 0
            KC0e0611 = 0
            KC0e0617 = 0
            KC0e0623 = 0
            KC0e0711 = 0
            KC0e0717 = 0
            KC0e0723 = 0
            KC0e1111 = 0
            KC0e1112 = 0
            KC0e1113 = 0
            KC0e1117 = 0
            KC0e1118 = 0
            KC0e1119 = 0
            KC0e1123 = 0
            KC0e1217 = 0
            KC0e1223 = 0
            KC0e1317 = 0
            KC0e1323 = 0
            KC0e1717 = 0
            KC0e1718 = 0
            KC0e1719 = 0
            KC0e1723 = 0
            KC0e1823 = 0
            KC0e1923 = 0
            KC0e2323 = 0

            for pti in range(2):
                xi = points[pti]
                for ptj in range(2):
                    eta = points[ptj]

                    J11 = -0.5*x1 + 0.5*x2 + 0.5*(eta + 1)*(0.5*x1 - 0.5*x2 + 0.5*x3 - 0.5*x4)
                    J12 = -0.5*y1 + 0.5*y2 + 0.5*(eta + 1)*(0.5*y1 - 0.5*y2 + 0.5*y3 - 0.5*y4)
                    J21 = -0.5*x1 + 0.5*x4 - 0.25*(-x1 + x2)*(xi + 1) + 0.25*(x3 - x4)*(xi + 1)
                    J22 = -0.5*y1 + 0.5*y4 - 0.25*(xi + 1)*(-y1 + y2) + 0.25*(xi + 1)*(y3 - y4)

                    detJ = J11*J22 - J12*J21

                    j11 = J22/(J11*J22 - J12*J21)
                    j12 = -J12/(J11*J22 - J12*J21)
                    j21 = -J21/(J11*J22 - J12*J21)
                    j22 = J11/(J11*J22 - J12*J21)

                    S1 = eta*xi/4. - eta/4. - xi/4. + 1/4.
                    S2 = -eta*xi/4. - eta/4. + xi/4. + 1/4.
                    S3 = eta*xi/4. + eta/4. + xi/4. + 1/4.
                    S4 = -eta*xi/4. + eta/4. - xi/4. + 1/4.

                    S1_x = 0.25*j11*(eta - 1) + 0.25*j12*(xi - 1)
                    S2_x = -0.25*eta*j11 + 0.25*j11 - 0.25*j12*xi - 0.25*j12
                    S3_x = 0.25*j11*(eta + 1) + 0.25*j12*(xi + 1)
                    S4_x = -0.25*eta*j11 - 0.25*j11 - 0.25*j12*xi + 0.25*j12

                    S1_y = 0.25*j21*(eta - 1) + 0.25*j22*(xi - 1)
                    S2_y = -0.25*eta*j21 + 0.25*j21 - 0.25*j22*xi - 0.25*j22
                    S3_y = 0.25*j21*(eta + 1) + 0.25*j22*(xi + 1)
                    S4_y = -0.25*eta*j21 - 0.25*j21 - 0.25*j22*xi + 0.25*j22

                    KC0e0000 += 2.5e-7*A66*K6ROT*S1_y**2*detJ*wij
                    KC0e0001 += -2.5e-7*A66*K6ROT*S1_x*S1_y*detJ*wij
                    KC0e0005 += 5.0e-7*A66*K6ROT*S1*S1_y*detJ*wij
                    KC0e0006 += 2.5e-7*A66*K6ROT*S1_y*S2_y*detJ*wij
                    KC0e0007 += -2.5e-7*A66*K6ROT*S1_y*S2_x*detJ*wij
                    KC0e0011 += 5.0e-7*A66*K6ROT*S1_y*S2*detJ*wij
                    KC0e0012 += 2.5e-7*A66*K6ROT*S1_y*S3_y*detJ*wij
                    KC0e0013 += -2.5e-7*A66*K6ROT*S1_y*S3_x*detJ*wij
                    KC0e0017 += 5.0e-7*A66*K6ROT*S1_y*S3*detJ*wij
                    KC0e0018 += 2.5e-7*A66*K6ROT*S1_y*S4_y*detJ*wij
                    KC0e0019 += -2.5e-7*A66*K6ROT*S1_y*S4_x*detJ*wij
                    KC0e0023 += 5.0e-7*A66*K6ROT*S1_y*S4*detJ*wij
                    KC0e0101 += 2.5e-7*A66*K6ROT*S1_x**2*detJ*wij
                    KC0e0105 += -5.0e-7*A66*K6ROT*S1*S1_x*detJ*wij
                    KC0e0106 += -2.5e-7*A66*K6ROT*S1_x*S2_y*detJ*wij
                    KC0e0107 += 2.5e-7*A66*K6ROT*S1_x*S2_x*detJ*wij
                    KC0e0111 += -5.0e-7*A66*K6ROT*S1_x*S2*detJ*wij
                    KC0e0112 += -2.5e-7*A66*K6ROT*S1_x*S3_y*detJ*wij
                    KC0e0113 += 2.5e-7*A66*K6ROT*S1_x*S3_x*detJ*wij
                    KC0e0117 += -5.0e-7*A66*K6ROT*S1_x*S3*detJ*wij
                    KC0e0118 += -2.5e-7*A66*K6ROT*S1_x*S4_y*detJ*wij
                    KC0e0119 += 2.5e-7*A66*K6ROT*S1_x*S4_x*detJ*wij
                    KC0e0123 += -5.0e-7*A66*K6ROT*S1_x*S4*detJ*wij
                    KC0e0505 += 1.0e-6*A66*K6ROT*S1**2*detJ*wij
                    KC0e0506 += 5.0e-7*A66*K6ROT*S1*S2_y*detJ*wij
                    KC0e0507 += -5.0e-7*A66*K6ROT*S1*S2_x*detJ*wij
                    KC0e0511 += 1.0e-6*A66*K6ROT*S1*S2*detJ*wij
                    KC0e0512 += 5.0e-7*A66*K6ROT*S1*S3_y*detJ*wij
                    KC0e0513 += -5.0e-7*A66*K6ROT*S1*S3_x*detJ*wij
                    KC0e0517 += 1.0e-6*A66*K6ROT*S1*S3*detJ*wij
                    KC0e0518 += 5.0e-7*A66*K6ROT*S1*S4_y*detJ*wij
                    KC0e0519 += -5.0e-7*A66*K6ROT*S1*S4_x*detJ*wij
                    KC0e0523 += 1.0e-6*A66*K6ROT*S1*S4*detJ*wij
                    KC0e0606 += 2.5e-7*A66*K6ROT*S2_y**2*detJ*wij
                    KC0e0607 += -2.5e-7*A66*K6ROT*S2_x*S2_y*detJ*wij
                    KC0e0611 += 5.0e-7*A66*K6ROT*S2*S2_y*detJ*wij
                    KC0e0612 += 2.5e-7*A66*K6ROT*S2_y*S3_y*detJ*wij
                    KC0e0613 += -2.5e-7*A66*K6ROT*S2_y*S3_x*detJ*wij
                    KC0e0617 += 5.0e-7*A66*K6ROT*S2_y*S3*detJ*wij
                    KC0e0618 += 2.5e-7*A66*K6ROT*S2_y*S4_y*detJ*wij
                    KC0e0619 += -2.5e-7*A66*K6ROT*S2_y*S4_x*detJ*wij
                    KC0e0623 += 5.0e-7*A66*K6ROT*S2_y*S4*detJ*wij
                    KC0e0707 += 2.5e-7*A66*K6ROT*S2_x**2*detJ*wij
                    KC0e0711 += -5.0e-7*A66*K6ROT*S2*S2_x*detJ*wij
                    KC0e0712 += -2.5e-7*A66*K6ROT*S2_x*S3_y*detJ*wij
                    KC0e0713 += 2.5e-7*A66*K6ROT*S2_x*S3_x*detJ*wij
                    KC0e0717 += -5.0e-7*A66*K6ROT*S2_x*S3*detJ*wij
                    KC0e0718 += -2.5e-7*A66*K6ROT*S2_x*S4_y*detJ*wij
                    KC0e0719 += 2.5e-7*A66*K6ROT*S2_x*S4_x*detJ*wij
                    KC0e0723 += -5.0e-7*A66*K6ROT*S2_x*S4*detJ*wij
                    KC0e1111 += 1.0e-6*A66*K6ROT*S2**2*detJ*wij
                    KC0e1112 += 5.0e-7*A66*K6ROT*S2*S3_y*detJ*wij
                    KC0e1113 += -5.0e-7*A66*K6ROT*S2*S3_x*detJ*wij
                    KC0e1117 += 1.0e-6*A66*K6ROT*S2*S3*detJ*wij
                    KC0e1118 += 5.0e-7*A66*K6ROT*S2*S4_y*detJ*wij
                    KC0e1119 += -5.0e-7*A66*K6ROT*S2*S4_x*detJ*wij
                    KC0e1123 += 1.0e-6*A66*K6ROT*S2*S4*detJ*wij
                    KC0e1212 += 2.5e-7*A66*K6ROT*S3_y**2*detJ*wij
                    KC0e1213 += -2.5e-7*A66*K6ROT*S3_x*S3_y*detJ*wij
                    KC0e1217 += 5.0e-7*A66*K6ROT*S3*S3_y*detJ*wij
                    KC0e1218 += 2.5e-7*A66*K6ROT*S3_y*S4_y*detJ*wij
                    KC0e1219 += -2.5e-7*A66*K6ROT*S3_y*S4_x*detJ*wij
                    KC0e1223 += 5.0e-7*A66*K6ROT*S3_y*S4*detJ*wij
                    KC0e1313 += 2.5e-7*A66*K6ROT*S3_x**2*detJ*wij
                    KC0e1317 += -5.0e-7*A66*K6ROT*S3*S3_x*detJ*wij
                    KC0e1318 += -2.5e-7*A66*K6ROT*S3_x*S4_y*detJ*wij
                    KC0e1319 += 2.5e-7*A66*K6ROT*S3_x*S4_x*detJ*wij
                    KC0e1323 += -5.0e-7*A66*K6ROT*S3_x*S4*detJ*wij
                    KC0e1717 += 1.0e-6*A66*K6ROT*S3**2*detJ*wij
                    KC0e1718 += 5.0e-7*A66*K6ROT*S3*S4_y*detJ*wij
                    KC0e1719 += -5.0e-7*A66*K6ROT*S3*S4_x*detJ*wij
                    KC0e1723 += 1.0e-6*A66*K6ROT*S3*S4*detJ*wij
                    KC0e1818 += 2.5e-7*A66*K6ROT*S4_y**2*detJ*wij
                    KC0e1819 += -2.5e-7*A66*K6ROT*S4_x*S4_y*detJ*wij
                    KC0e1823 += 5.0e-7*A66*K6ROT*S4*S4_y*detJ*wij
                    KC0e1919 += 2.5e-7*A66*K6ROT*S4_x**2*detJ*wij
                    KC0e1923 += -5.0e-7*A66*K6ROT*S4*S4_x*detJ*wij
                    KC0e2323 += 1.0e-6*A66*K6ROT*S4**2*detJ*wij

            # NOTE storing the rotated KC0e into the global KC0

            k = self.init_k_KC0
            KC0v[k] += KC0e0202*r13**2 + r11*(KC0e0000*r11 + KC0e0001*r12) + r12*(KC0e0001*r11 + KC0e0101*r12)
            k += 1
            KC0v[k] += KC0e0202*r13*r23 + r21*(KC0e0000*r11 + KC0e0001*r12) + r22*(KC0e0001*r11 + KC0e0101*r12)
            k += 1
            KC0v[k] += KC0e0202*r13*r33 + r31*(KC0e0000*r11 + KC0e0001*r12) + r32*(KC0e0001*r11 + KC0e0101*r12)
            k += 1
            KC0v[k] += r11*(KC0e0003*r11 + KC0e0103*r12 + KC0e0203*r13) + r12*(KC0e0004*r11 + KC0e0104*r12 + KC0e0204*r13) + r13*(KC0e0005*r11 + KC0e0105*r12)
            k += 1
            KC0v[k] += r21*(KC0e0003*r11 + KC0e0103*r12 + KC0e0203*r13) + r22*(KC0e0004*r11 + KC0e0104*r12 + KC0e0204*r13) + r23*(KC0e0005*r11 + KC0e0105*r12)
            k += 1
            KC0v[k] += r31*(KC0e0003*r11 + KC0e0103*r12 + KC0e0203*r13) + r32*(KC0e0004*r11 + KC0e0104*r12 + KC0e0204*r13) + r33*(KC0e0005*r11 + KC0e0105*r12)
            k += 1
            KC0v[k] += KC0e0208*r13**2 + r11*(KC0e0006*r11 + KC0e0106*r12) + r12*(KC0e0007*r11 + KC0e0107*r12)
            k += 1
            KC0v[k] += KC0e0208*r13*r23 + r21*(KC0e0006*r11 + KC0e0106*r12) + r22*(KC0e0007*r11 + KC0e0107*r12)
            k += 1
            KC0v[k] += KC0e0208*r13*r33 + r31*(KC0e0006*r11 + KC0e0106*r12) + r32*(KC0e0007*r11 + KC0e0107*r12)
            k += 1
            KC0v[k] += r11*(KC0e0009*r11 + KC0e0109*r12 + KC0e0209*r13) + r12*(KC0e0010*r11 + KC0e0110*r12 + KC0e0210*r13) + r13*(KC0e0011*r11 + KC0e0111*r12)
            k += 1
            KC0v[k] += r21*(KC0e0009*r11 + KC0e0109*r12 + KC0e0209*r13) + r22*(KC0e0010*r11 + KC0e0110*r12 + KC0e0210*r13) + r23*(KC0e0011*r11 + KC0e0111*r12)
            k += 1
            KC0v[k] += r31*(KC0e0009*r11 + KC0e0109*r12 + KC0e0209*r13) + r32*(KC0e0010*r11 + KC0e0110*r12 + KC0e0210*r13) + r33*(KC0e0011*r11 + KC0e0111*r12)
            k += 1
            KC0v[k] += KC0e0214*r13**2 + r11*(KC0e0012*r11 + KC0e0112*r12) + r12*(KC0e0013*r11 + KC0e0113*r12)
            k += 1
            KC0v[k] += KC0e0214*r13*r23 + r21*(KC0e0012*r11 + KC0e0112*r12) + r22*(KC0e0013*r11 + KC0e0113*r12)
            k += 1
            KC0v[k] += KC0e0214*r13*r33 + r31*(KC0e0012*r11 + KC0e0112*r12) + r32*(KC0e0013*r11 + KC0e0113*r12)
            k += 1
            KC0v[k] += r11*(KC0e0015*r11 + KC0e0115*r12 + KC0e0215*r13) + r12*(KC0e0016*r11 + KC0e0116*r12 + KC0e0216*r13) + r13*(KC0e0017*r11 + KC0e0117*r12)
            k += 1
            KC0v[k] += r21*(KC0e0015*r11 + KC0e0115*r12 + KC0e0215*r13) + r22*(KC0e0016*r11 + KC0e0116*r12 + KC0e0216*r13) + r23*(KC0e0017*r11 + KC0e0117*r12)
            k += 1
            KC0v[k] += r31*(KC0e0015*r11 + KC0e0115*r12 + KC0e0215*r13) + r32*(KC0e0016*r11 + KC0e0116*r12 + KC0e0216*r13) + r33*(KC0e0017*r11 + KC0e0117*r12)
            k += 1
            KC0v[k] += KC0e0220*r13**2 + r11*(KC0e0018*r11 + KC0e0118*r12) + r12*(KC0e0019*r11 + KC0e0119*r12)
            k += 1
            KC0v[k] += KC0e0220*r13*r23 + r21*(KC0e0018*r11 + KC0e0118*r12) + r22*(KC0e0019*r11 + KC0e0119*r12)
            k += 1
            KC0v[k] += KC0e0220*r13*r33 + r31*(KC0e0018*r11 + KC0e0118*r12) + r32*(KC0e0019*r11 + KC0e0119*r12)
            k += 1
            KC0v[k] += r11*(KC0e0021*r11 + KC0e0121*r12 + KC0e0221*r13) + r12*(KC0e0022*r11 + KC0e0122*r12 + KC0e0222*r13) + r13*(KC0e0023*r11 + KC0e0123*r12)
            k += 1
            KC0v[k] += r21*(KC0e0021*r11 + KC0e0121*r12 + KC0e0221*r13) + r22*(KC0e0022*r11 + KC0e0122*r12 + KC0e0222*r13) + r23*(KC0e0023*r11 + KC0e0123*r12)
            k += 1
            KC0v[k] += r31*(KC0e0021*r11 + KC0e0121*r12 + KC0e0221*r13) + r32*(KC0e0022*r11 + KC0e0122*r12 + KC0e0222*r13) + r33*(KC0e0023*r11 + KC0e0123*r12)
            k += 1
            KC0v[k] += KC0e0202*r13*r23 + r11*(KC0e0000*r21 + KC0e0001*r22) + r12*(KC0e0001*r21 + KC0e0101*r22)
            k += 1
            KC0v[k] += KC0e0202*r23**2 + r21*(KC0e0000*r21 + KC0e0001*r22) + r22*(KC0e0001*r21 + KC0e0101*r22)
            k += 1
            KC0v[k] += KC0e0202*r23*r33 + r31*(KC0e0000*r21 + KC0e0001*r22) + r32*(KC0e0001*r21 + KC0e0101*r22)
            k += 1
            KC0v[k] += r11*(KC0e0003*r21 + KC0e0103*r22 + KC0e0203*r23) + r12*(KC0e0004*r21 + KC0e0104*r22 + KC0e0204*r23) + r13*(KC0e0005*r21 + KC0e0105*r22)
            k += 1
            KC0v[k] += r21*(KC0e0003*r21 + KC0e0103*r22 + KC0e0203*r23) + r22*(KC0e0004*r21 + KC0e0104*r22 + KC0e0204*r23) + r23*(KC0e0005*r21 + KC0e0105*r22)
            k += 1
            KC0v[k] += r31*(KC0e0003*r21 + KC0e0103*r22 + KC0e0203*r23) + r32*(KC0e0004*r21 + KC0e0104*r22 + KC0e0204*r23) + r33*(KC0e0005*r21 + KC0e0105*r22)
            k += 1
            KC0v[k] += KC0e0208*r13*r23 + r11*(KC0e0006*r21 + KC0e0106*r22) + r12*(KC0e0007*r21 + KC0e0107*r22)
            k += 1
            KC0v[k] += KC0e0208*r23**2 + r21*(KC0e0006*r21 + KC0e0106*r22) + r22*(KC0e0007*r21 + KC0e0107*r22)
            k += 1
            KC0v[k] += KC0e0208*r23*r33 + r31*(KC0e0006*r21 + KC0e0106*r22) + r32*(KC0e0007*r21 + KC0e0107*r22)
            k += 1
            KC0v[k] += r11*(KC0e0009*r21 + KC0e0109*r22 + KC0e0209*r23) + r12*(KC0e0010*r21 + KC0e0110*r22 + KC0e0210*r23) + r13*(KC0e0011*r21 + KC0e0111*r22)
            k += 1
            KC0v[k] += r21*(KC0e0009*r21 + KC0e0109*r22 + KC0e0209*r23) + r22*(KC0e0010*r21 + KC0e0110*r22 + KC0e0210*r23) + r23*(KC0e0011*r21 + KC0e0111*r22)
            k += 1
            KC0v[k] += r31*(KC0e0009*r21 + KC0e0109*r22 + KC0e0209*r23) + r32*(KC0e0010*r21 + KC0e0110*r22 + KC0e0210*r23) + r33*(KC0e0011*r21 + KC0e0111*r22)
            k += 1
            KC0v[k] += KC0e0214*r13*r23 + r11*(KC0e0012*r21 + KC0e0112*r22) + r12*(KC0e0013*r21 + KC0e0113*r22)
            k += 1
            KC0v[k] += KC0e0214*r23**2 + r21*(KC0e0012*r21 + KC0e0112*r22) + r22*(KC0e0013*r21 + KC0e0113*r22)
            k += 1
            KC0v[k] += KC0e0214*r23*r33 + r31*(KC0e0012*r21 + KC0e0112*r22) + r32*(KC0e0013*r21 + KC0e0113*r22)
            k += 1
            KC0v[k] += r11*(KC0e0015*r21 + KC0e0115*r22 + KC0e0215*r23) + r12*(KC0e0016*r21 + KC0e0116*r22 + KC0e0216*r23) + r13*(KC0e0017*r21 + KC0e0117*r22)
            k += 1
            KC0v[k] += r21*(KC0e0015*r21 + KC0e0115*r22 + KC0e0215*r23) + r22*(KC0e0016*r21 + KC0e0116*r22 + KC0e0216*r23) + r23*(KC0e0017*r21 + KC0e0117*r22)
            k += 1
            KC0v[k] += r31*(KC0e0015*r21 + KC0e0115*r22 + KC0e0215*r23) + r32*(KC0e0016*r21 + KC0e0116*r22 + KC0e0216*r23) + r33*(KC0e0017*r21 + KC0e0117*r22)
            k += 1
            KC0v[k] += KC0e0220*r13*r23 + r11*(KC0e0018*r21 + KC0e0118*r22) + r12*(KC0e0019*r21 + KC0e0119*r22)
            k += 1
            KC0v[k] += KC0e0220*r23**2 + r21*(KC0e0018*r21 + KC0e0118*r22) + r22*(KC0e0019*r21 + KC0e0119*r22)
            k += 1
            KC0v[k] += KC0e0220*r23*r33 + r31*(KC0e0018*r21 + KC0e0118*r22) + r32*(KC0e0019*r21 + KC0e0119*r22)
            k += 1
            KC0v[k] += r11*(KC0e0021*r21 + KC0e0121*r22 + KC0e0221*r23) + r12*(KC0e0022*r21 + KC0e0122*r22 + KC0e0222*r23) + r13*(KC0e0023*r21 + KC0e0123*r22)
            k += 1
            KC0v[k] += r21*(KC0e0021*r21 + KC0e0121*r22 + KC0e0221*r23) + r22*(KC0e0022*r21 + KC0e0122*r22 + KC0e0222*r23) + r23*(KC0e0023*r21 + KC0e0123*r22)
            k += 1
            KC0v[k] += r31*(KC0e0021*r21 + KC0e0121*r22 + KC0e0221*r23) + r32*(KC0e0022*r21 + KC0e0122*r22 + KC0e0222*r23) + r33*(KC0e0023*r21 + KC0e0123*r22)
            k += 1
            KC0v[k] += KC0e0202*r13*r33 + r11*(KC0e0000*r31 + KC0e0001*r32) + r12*(KC0e0001*r31 + KC0e0101*r32)
            k += 1
            KC0v[k] += KC0e0202*r23*r33 + r21*(KC0e0000*r31 + KC0e0001*r32) + r22*(KC0e0001*r31 + KC0e0101*r32)
            k += 1
            KC0v[k] += KC0e0202*r33**2 + r31*(KC0e0000*r31 + KC0e0001*r32) + r32*(KC0e0001*r31 + KC0e0101*r32)
            k += 1
            KC0v[k] += r11*(KC0e0003*r31 + KC0e0103*r32 + KC0e0203*r33) + r12*(KC0e0004*r31 + KC0e0104*r32 + KC0e0204*r33) + r13*(KC0e0005*r31 + KC0e0105*r32)
            k += 1
            KC0v[k] += r21*(KC0e0003*r31 + KC0e0103*r32 + KC0e0203*r33) + r22*(KC0e0004*r31 + KC0e0104*r32 + KC0e0204*r33) + r23*(KC0e0005*r31 + KC0e0105*r32)
            k += 1
            KC0v[k] += r31*(KC0e0003*r31 + KC0e0103*r32 + KC0e0203*r33) + r32*(KC0e0004*r31 + KC0e0104*r32 + KC0e0204*r33) + r33*(KC0e0005*r31 + KC0e0105*r32)
            k += 1
            KC0v[k] += KC0e0208*r13*r33 + r11*(KC0e0006*r31 + KC0e0106*r32) + r12*(KC0e0007*r31 + KC0e0107*r32)
            k += 1
            KC0v[k] += KC0e0208*r23*r33 + r21*(KC0e0006*r31 + KC0e0106*r32) + r22*(KC0e0007*r31 + KC0e0107*r32)
            k += 1
            KC0v[k] += KC0e0208*r33**2 + r31*(KC0e0006*r31 + KC0e0106*r32) + r32*(KC0e0007*r31 + KC0e0107*r32)
            k += 1
            KC0v[k] += r11*(KC0e0009*r31 + KC0e0109*r32 + KC0e0209*r33) + r12*(KC0e0010*r31 + KC0e0110*r32 + KC0e0210*r33) + r13*(KC0e0011*r31 + KC0e0111*r32)
            k += 1
            KC0v[k] += r21*(KC0e0009*r31 + KC0e0109*r32 + KC0e0209*r33) + r22*(KC0e0010*r31 + KC0e0110*r32 + KC0e0210*r33) + r23*(KC0e0011*r31 + KC0e0111*r32)
            k += 1
            KC0v[k] += r31*(KC0e0009*r31 + KC0e0109*r32 + KC0e0209*r33) + r32*(KC0e0010*r31 + KC0e0110*r32 + KC0e0210*r33) + r33*(KC0e0011*r31 + KC0e0111*r32)
            k += 1
            KC0v[k] += KC0e0214*r13*r33 + r11*(KC0e0012*r31 + KC0e0112*r32) + r12*(KC0e0013*r31 + KC0e0113*r32)
            k += 1
            KC0v[k] += KC0e0214*r23*r33 + r21*(KC0e0012*r31 + KC0e0112*r32) + r22*(KC0e0013*r31 + KC0e0113*r32)
            k += 1
            KC0v[k] += KC0e0214*r33**2 + r31*(KC0e0012*r31 + KC0e0112*r32) + r32*(KC0e0013*r31 + KC0e0113*r32)
            k += 1
            KC0v[k] += r11*(KC0e0015*r31 + KC0e0115*r32 + KC0e0215*r33) + r12*(KC0e0016*r31 + KC0e0116*r32 + KC0e0216*r33) + r13*(KC0e0017*r31 + KC0e0117*r32)
            k += 1
            KC0v[k] += r21*(KC0e0015*r31 + KC0e0115*r32 + KC0e0215*r33) + r22*(KC0e0016*r31 + KC0e0116*r32 + KC0e0216*r33) + r23*(KC0e0017*r31 + KC0e0117*r32)
            k += 1
            KC0v[k] += r31*(KC0e0015*r31 + KC0e0115*r32 + KC0e0215*r33) + r32*(KC0e0016*r31 + KC0e0116*r32 + KC0e0216*r33) + r33*(KC0e0017*r31 + KC0e0117*r32)
            k += 1
            KC0v[k] += KC0e0220*r13*r33 + r11*(KC0e0018*r31 + KC0e0118*r32) + r12*(KC0e0019*r31 + KC0e0119*r32)
            k += 1
            KC0v[k] += KC0e0220*r23*r33 + r21*(KC0e0018*r31 + KC0e0118*r32) + r22*(KC0e0019*r31 + KC0e0119*r32)
            k += 1
            KC0v[k] += KC0e0220*r33**2 + r31*(KC0e0018*r31 + KC0e0118*r32) + r32*(KC0e0019*r31 + KC0e0119*r32)
            k += 1
            KC0v[k] += r11*(KC0e0021*r31 + KC0e0121*r32 + KC0e0221*r33) + r12*(KC0e0022*r31 + KC0e0122*r32 + KC0e0222*r33) + r13*(KC0e0023*r31 + KC0e0123*r32)
            k += 1
            KC0v[k] += r21*(KC0e0021*r31 + KC0e0121*r32 + KC0e0221*r33) + r22*(KC0e0022*r31 + KC0e0122*r32 + KC0e0222*r33) + r23*(KC0e0023*r31 + KC0e0123*r32)
            k += 1
            KC0v[k] += r31*(KC0e0021*r31 + KC0e0121*r32 + KC0e0221*r33) + r32*(KC0e0022*r31 + KC0e0122*r32 + KC0e0222*r33) + r33*(KC0e0023*r31 + KC0e0123*r32)
            k += 1
            KC0v[k] += r11*(KC0e0003*r11 + KC0e0004*r12 + KC0e0005*r13) + r12*(KC0e0103*r11 + KC0e0104*r12 + KC0e0105*r13) + r13*(KC0e0203*r11 + KC0e0204*r12)
            k += 1
            KC0v[k] += r21*(KC0e0003*r11 + KC0e0004*r12 + KC0e0005*r13) + r22*(KC0e0103*r11 + KC0e0104*r12 + KC0e0105*r13) + r23*(KC0e0203*r11 + KC0e0204*r12)
            k += 1
            KC0v[k] += r31*(KC0e0003*r11 + KC0e0004*r12 + KC0e0005*r13) + r32*(KC0e0103*r11 + KC0e0104*r12 + KC0e0105*r13) + r33*(KC0e0203*r11 + KC0e0204*r12)
            k += 1
            KC0v[k] += KC0e0505*r13**2 + r11*(KC0e0303*r11 + KC0e0304*r12) + r12*(KC0e0304*r11 + KC0e0404*r12)
            k += 1
            KC0v[k] += KC0e0505*r13*r23 + r21*(KC0e0303*r11 + KC0e0304*r12) + r22*(KC0e0304*r11 + KC0e0404*r12)
            k += 1
            KC0v[k] += KC0e0505*r13*r33 + r31*(KC0e0303*r11 + KC0e0304*r12) + r32*(KC0e0304*r11 + KC0e0404*r12)
            k += 1
            KC0v[k] += r11*(KC0e0306*r11 + KC0e0406*r12 + KC0e0506*r13) + r12*(KC0e0307*r11 + KC0e0407*r12 + KC0e0507*r13) + r13*(KC0e0308*r11 + KC0e0408*r12)
            k += 1
            KC0v[k] += r21*(KC0e0306*r11 + KC0e0406*r12 + KC0e0506*r13) + r22*(KC0e0307*r11 + KC0e0407*r12 + KC0e0507*r13) + r23*(KC0e0308*r11 + KC0e0408*r12)
            k += 1
            KC0v[k] += r31*(KC0e0306*r11 + KC0e0406*r12 + KC0e0506*r13) + r32*(KC0e0307*r11 + KC0e0407*r12 + KC0e0507*r13) + r33*(KC0e0308*r11 + KC0e0408*r12)
            k += 1
            KC0v[k] += KC0e0511*r13**2 + r11*(KC0e0309*r11 + KC0e0409*r12) + r12*(KC0e0310*r11 + KC0e0410*r12)
            k += 1
            KC0v[k] += KC0e0511*r13*r23 + r21*(KC0e0309*r11 + KC0e0409*r12) + r22*(KC0e0310*r11 + KC0e0410*r12)
            k += 1
            KC0v[k] += KC0e0511*r13*r33 + r31*(KC0e0309*r11 + KC0e0409*r12) + r32*(KC0e0310*r11 + KC0e0410*r12)
            k += 1
            KC0v[k] += r11*(KC0e0312*r11 + KC0e0412*r12 + KC0e0512*r13) + r12*(KC0e0313*r11 + KC0e0413*r12 + KC0e0513*r13) + r13*(KC0e0314*r11 + KC0e0414*r12)
            k += 1
            KC0v[k] += r21*(KC0e0312*r11 + KC0e0412*r12 + KC0e0512*r13) + r22*(KC0e0313*r11 + KC0e0413*r12 + KC0e0513*r13) + r23*(KC0e0314*r11 + KC0e0414*r12)
            k += 1
            KC0v[k] += r31*(KC0e0312*r11 + KC0e0412*r12 + KC0e0512*r13) + r32*(KC0e0313*r11 + KC0e0413*r12 + KC0e0513*r13) + r33*(KC0e0314*r11 + KC0e0414*r12)
            k += 1
            KC0v[k] += KC0e0517*r13**2 + r11*(KC0e0315*r11 + KC0e0415*r12) + r12*(KC0e0316*r11 + KC0e0416*r12)
            k += 1
            KC0v[k] += KC0e0517*r13*r23 + r21*(KC0e0315*r11 + KC0e0415*r12) + r22*(KC0e0316*r11 + KC0e0416*r12)
            k += 1
            KC0v[k] += KC0e0517*r13*r33 + r31*(KC0e0315*r11 + KC0e0415*r12) + r32*(KC0e0316*r11 + KC0e0416*r12)
            k += 1
            KC0v[k] += r11*(KC0e0318*r11 + KC0e0418*r12 + KC0e0518*r13) + r12*(KC0e0319*r11 + KC0e0419*r12 + KC0e0519*r13) + r13*(KC0e0320*r11 + KC0e0420*r12)
            k += 1
            KC0v[k] += r21*(KC0e0318*r11 + KC0e0418*r12 + KC0e0518*r13) + r22*(KC0e0319*r11 + KC0e0419*r12 + KC0e0519*r13) + r23*(KC0e0320*r11 + KC0e0420*r12)
            k += 1
            KC0v[k] += r31*(KC0e0318*r11 + KC0e0418*r12 + KC0e0518*r13) + r32*(KC0e0319*r11 + KC0e0419*r12 + KC0e0519*r13) + r33*(KC0e0320*r11 + KC0e0420*r12)
            k += 1
            KC0v[k] += KC0e0523*r13**2 + r11*(KC0e0321*r11 + KC0e0421*r12) + r12*(KC0e0322*r11 + KC0e0422*r12)
            k += 1
            KC0v[k] += KC0e0523*r13*r23 + r21*(KC0e0321*r11 + KC0e0421*r12) + r22*(KC0e0322*r11 + KC0e0422*r12)
            k += 1
            KC0v[k] += KC0e0523*r13*r33 + r31*(KC0e0321*r11 + KC0e0421*r12) + r32*(KC0e0322*r11 + KC0e0422*r12)
            k += 1
            KC0v[k] += r11*(KC0e0003*r21 + KC0e0004*r22 + KC0e0005*r23) + r12*(KC0e0103*r21 + KC0e0104*r22 + KC0e0105*r23) + r13*(KC0e0203*r21 + KC0e0204*r22)
            k += 1
            KC0v[k] += r21*(KC0e0003*r21 + KC0e0004*r22 + KC0e0005*r23) + r22*(KC0e0103*r21 + KC0e0104*r22 + KC0e0105*r23) + r23*(KC0e0203*r21 + KC0e0204*r22)
            k += 1
            KC0v[k] += r31*(KC0e0003*r21 + KC0e0004*r22 + KC0e0005*r23) + r32*(KC0e0103*r21 + KC0e0104*r22 + KC0e0105*r23) + r33*(KC0e0203*r21 + KC0e0204*r22)
            k += 1
            KC0v[k] += KC0e0505*r13*r23 + r11*(KC0e0303*r21 + KC0e0304*r22) + r12*(KC0e0304*r21 + KC0e0404*r22)
            k += 1
            KC0v[k] += KC0e0505*r23**2 + r21*(KC0e0303*r21 + KC0e0304*r22) + r22*(KC0e0304*r21 + KC0e0404*r22)
            k += 1
            KC0v[k] += KC0e0505*r23*r33 + r31*(KC0e0303*r21 + KC0e0304*r22) + r32*(KC0e0304*r21 + KC0e0404*r22)
            k += 1
            KC0v[k] += r11*(KC0e0306*r21 + KC0e0406*r22 + KC0e0506*r23) + r12*(KC0e0307*r21 + KC0e0407*r22 + KC0e0507*r23) + r13*(KC0e0308*r21 + KC0e0408*r22)
            k += 1
            KC0v[k] += r21*(KC0e0306*r21 + KC0e0406*r22 + KC0e0506*r23) + r22*(KC0e0307*r21 + KC0e0407*r22 + KC0e0507*r23) + r23*(KC0e0308*r21 + KC0e0408*r22)
            k += 1
            KC0v[k] += r31*(KC0e0306*r21 + KC0e0406*r22 + KC0e0506*r23) + r32*(KC0e0307*r21 + KC0e0407*r22 + KC0e0507*r23) + r33*(KC0e0308*r21 + KC0e0408*r22)
            k += 1
            KC0v[k] += KC0e0511*r13*r23 + r11*(KC0e0309*r21 + KC0e0409*r22) + r12*(KC0e0310*r21 + KC0e0410*r22)
            k += 1
            KC0v[k] += KC0e0511*r23**2 + r21*(KC0e0309*r21 + KC0e0409*r22) + r22*(KC0e0310*r21 + KC0e0410*r22)
            k += 1
            KC0v[k] += KC0e0511*r23*r33 + r31*(KC0e0309*r21 + KC0e0409*r22) + r32*(KC0e0310*r21 + KC0e0410*r22)
            k += 1
            KC0v[k] += r11*(KC0e0312*r21 + KC0e0412*r22 + KC0e0512*r23) + r12*(KC0e0313*r21 + KC0e0413*r22 + KC0e0513*r23) + r13*(KC0e0314*r21 + KC0e0414*r22)
            k += 1
            KC0v[k] += r21*(KC0e0312*r21 + KC0e0412*r22 + KC0e0512*r23) + r22*(KC0e0313*r21 + KC0e0413*r22 + KC0e0513*r23) + r23*(KC0e0314*r21 + KC0e0414*r22)
            k += 1
            KC0v[k] += r31*(KC0e0312*r21 + KC0e0412*r22 + KC0e0512*r23) + r32*(KC0e0313*r21 + KC0e0413*r22 + KC0e0513*r23) + r33*(KC0e0314*r21 + KC0e0414*r22)
            k += 1
            KC0v[k] += KC0e0517*r13*r23 + r11*(KC0e0315*r21 + KC0e0415*r22) + r12*(KC0e0316*r21 + KC0e0416*r22)
            k += 1
            KC0v[k] += KC0e0517*r23**2 + r21*(KC0e0315*r21 + KC0e0415*r22) + r22*(KC0e0316*r21 + KC0e0416*r22)
            k += 1
            KC0v[k] += KC0e0517*r23*r33 + r31*(KC0e0315*r21 + KC0e0415*r22) + r32*(KC0e0316*r21 + KC0e0416*r22)
            k += 1
            KC0v[k] += r11*(KC0e0318*r21 + KC0e0418*r22 + KC0e0518*r23) + r12*(KC0e0319*r21 + KC0e0419*r22 + KC0e0519*r23) + r13*(KC0e0320*r21 + KC0e0420*r22)
            k += 1
            KC0v[k] += r21*(KC0e0318*r21 + KC0e0418*r22 + KC0e0518*r23) + r22*(KC0e0319*r21 + KC0e0419*r22 + KC0e0519*r23) + r23*(KC0e0320*r21 + KC0e0420*r22)
            k += 1
            KC0v[k] += r31*(KC0e0318*r21 + KC0e0418*r22 + KC0e0518*r23) + r32*(KC0e0319*r21 + KC0e0419*r22 + KC0e0519*r23) + r33*(KC0e0320*r21 + KC0e0420*r22)
            k += 1
            KC0v[k] += KC0e0523*r13*r23 + r11*(KC0e0321*r21 + KC0e0421*r22) + r12*(KC0e0322*r21 + KC0e0422*r22)
            k += 1
            KC0v[k] += KC0e0523*r23**2 + r21*(KC0e0321*r21 + KC0e0421*r22) + r22*(KC0e0322*r21 + KC0e0422*r22)
            k += 1
            KC0v[k] += KC0e0523*r23*r33 + r31*(KC0e0321*r21 + KC0e0421*r22) + r32*(KC0e0322*r21 + KC0e0422*r22)
            k += 1
            KC0v[k] += r11*(KC0e0003*r31 + KC0e0004*r32 + KC0e0005*r33) + r12*(KC0e0103*r31 + KC0e0104*r32 + KC0e0105*r33) + r13*(KC0e0203*r31 + KC0e0204*r32)
            k += 1
            KC0v[k] += r21*(KC0e0003*r31 + KC0e0004*r32 + KC0e0005*r33) + r22*(KC0e0103*r31 + KC0e0104*r32 + KC0e0105*r33) + r23*(KC0e0203*r31 + KC0e0204*r32)
            k += 1
            KC0v[k] += r31*(KC0e0003*r31 + KC0e0004*r32 + KC0e0005*r33) + r32*(KC0e0103*r31 + KC0e0104*r32 + KC0e0105*r33) + r33*(KC0e0203*r31 + KC0e0204*r32)
            k += 1
            KC0v[k] += KC0e0505*r13*r33 + r11*(KC0e0303*r31 + KC0e0304*r32) + r12*(KC0e0304*r31 + KC0e0404*r32)
            k += 1
            KC0v[k] += KC0e0505*r23*r33 + r21*(KC0e0303*r31 + KC0e0304*r32) + r22*(KC0e0304*r31 + KC0e0404*r32)
            k += 1
            KC0v[k] += KC0e0505*r33**2 + r31*(KC0e0303*r31 + KC0e0304*r32) + r32*(KC0e0304*r31 + KC0e0404*r32)
            k += 1
            KC0v[k] += r11*(KC0e0306*r31 + KC0e0406*r32 + KC0e0506*r33) + r12*(KC0e0307*r31 + KC0e0407*r32 + KC0e0507*r33) + r13*(KC0e0308*r31 + KC0e0408*r32)
            k += 1
            KC0v[k] += r21*(KC0e0306*r31 + KC0e0406*r32 + KC0e0506*r33) + r22*(KC0e0307*r31 + KC0e0407*r32 + KC0e0507*r33) + r23*(KC0e0308*r31 + KC0e0408*r32)
            k += 1
            KC0v[k] += r31*(KC0e0306*r31 + KC0e0406*r32 + KC0e0506*r33) + r32*(KC0e0307*r31 + KC0e0407*r32 + KC0e0507*r33) + r33*(KC0e0308*r31 + KC0e0408*r32)
            k += 1
            KC0v[k] += KC0e0511*r13*r33 + r11*(KC0e0309*r31 + KC0e0409*r32) + r12*(KC0e0310*r31 + KC0e0410*r32)
            k += 1
            KC0v[k] += KC0e0511*r23*r33 + r21*(KC0e0309*r31 + KC0e0409*r32) + r22*(KC0e0310*r31 + KC0e0410*r32)
            k += 1
            KC0v[k] += KC0e0511*r33**2 + r31*(KC0e0309*r31 + KC0e0409*r32) + r32*(KC0e0310*r31 + KC0e0410*r32)
            k += 1
            KC0v[k] += r11*(KC0e0312*r31 + KC0e0412*r32 + KC0e0512*r33) + r12*(KC0e0313*r31 + KC0e0413*r32 + KC0e0513*r33) + r13*(KC0e0314*r31 + KC0e0414*r32)
            k += 1
            KC0v[k] += r21*(KC0e0312*r31 + KC0e0412*r32 + KC0e0512*r33) + r22*(KC0e0313*r31 + KC0e0413*r32 + KC0e0513*r33) + r23*(KC0e0314*r31 + KC0e0414*r32)
            k += 1
            KC0v[k] += r31*(KC0e0312*r31 + KC0e0412*r32 + KC0e0512*r33) + r32*(KC0e0313*r31 + KC0e0413*r32 + KC0e0513*r33) + r33*(KC0e0314*r31 + KC0e0414*r32)
            k += 1
            KC0v[k] += KC0e0517*r13*r33 + r11*(KC0e0315*r31 + KC0e0415*r32) + r12*(KC0e0316*r31 + KC0e0416*r32)
            k += 1
            KC0v[k] += KC0e0517*r23*r33 + r21*(KC0e0315*r31 + KC0e0415*r32) + r22*(KC0e0316*r31 + KC0e0416*r32)
            k += 1
            KC0v[k] += KC0e0517*r33**2 + r31*(KC0e0315*r31 + KC0e0415*r32) + r32*(KC0e0316*r31 + KC0e0416*r32)
            k += 1
            KC0v[k] += r11*(KC0e0318*r31 + KC0e0418*r32 + KC0e0518*r33) + r12*(KC0e0319*r31 + KC0e0419*r32 + KC0e0519*r33) + r13*(KC0e0320*r31 + KC0e0420*r32)
            k += 1
            KC0v[k] += r21*(KC0e0318*r31 + KC0e0418*r32 + KC0e0518*r33) + r22*(KC0e0319*r31 + KC0e0419*r32 + KC0e0519*r33) + r23*(KC0e0320*r31 + KC0e0420*r32)
            k += 1
            KC0v[k] += r31*(KC0e0318*r31 + KC0e0418*r32 + KC0e0518*r33) + r32*(KC0e0319*r31 + KC0e0419*r32 + KC0e0519*r33) + r33*(KC0e0320*r31 + KC0e0420*r32)
            k += 1
            KC0v[k] += KC0e0523*r13*r33 + r11*(KC0e0321*r31 + KC0e0421*r32) + r12*(KC0e0322*r31 + KC0e0422*r32)
            k += 1
            KC0v[k] += KC0e0523*r23*r33 + r21*(KC0e0321*r31 + KC0e0421*r32) + r22*(KC0e0322*r31 + KC0e0422*r32)
            k += 1
            KC0v[k] += KC0e0523*r33**2 + r31*(KC0e0321*r31 + KC0e0421*r32) + r32*(KC0e0322*r31 + KC0e0422*r32)
            k += 1
            KC0v[k] += KC0e0208*r13**2 + r11*(KC0e0006*r11 + KC0e0007*r12) + r12*(KC0e0106*r11 + KC0e0107*r12)
            k += 1
            KC0v[k] += KC0e0208*r13*r23 + r21*(KC0e0006*r11 + KC0e0007*r12) + r22*(KC0e0106*r11 + KC0e0107*r12)
            k += 1
            KC0v[k] += KC0e0208*r13*r33 + r31*(KC0e0006*r11 + KC0e0007*r12) + r32*(KC0e0106*r11 + KC0e0107*r12)
            k += 1
            KC0v[k] += r11*(KC0e0306*r11 + KC0e0307*r12 + KC0e0308*r13) + r12*(KC0e0406*r11 + KC0e0407*r12 + KC0e0408*r13) + r13*(KC0e0506*r11 + KC0e0507*r12)
            k += 1
            KC0v[k] += r21*(KC0e0306*r11 + KC0e0307*r12 + KC0e0308*r13) + r22*(KC0e0406*r11 + KC0e0407*r12 + KC0e0408*r13) + r23*(KC0e0506*r11 + KC0e0507*r12)
            k += 1
            KC0v[k] += r31*(KC0e0306*r11 + KC0e0307*r12 + KC0e0308*r13) + r32*(KC0e0406*r11 + KC0e0407*r12 + KC0e0408*r13) + r33*(KC0e0506*r11 + KC0e0507*r12)
            k += 1
            KC0v[k] += KC0e0808*r13**2 + r11*(KC0e0606*r11 + KC0e0607*r12) + r12*(KC0e0607*r11 + KC0e0707*r12)
            k += 1
            KC0v[k] += KC0e0808*r13*r23 + r21*(KC0e0606*r11 + KC0e0607*r12) + r22*(KC0e0607*r11 + KC0e0707*r12)
            k += 1
            KC0v[k] += KC0e0808*r13*r33 + r31*(KC0e0606*r11 + KC0e0607*r12) + r32*(KC0e0607*r11 + KC0e0707*r12)
            k += 1
            KC0v[k] += r11*(KC0e0609*r11 + KC0e0709*r12 + KC0e0809*r13) + r12*(KC0e0610*r11 + KC0e0710*r12 + KC0e0810*r13) + r13*(KC0e0611*r11 + KC0e0711*r12)
            k += 1
            KC0v[k] += r21*(KC0e0609*r11 + KC0e0709*r12 + KC0e0809*r13) + r22*(KC0e0610*r11 + KC0e0710*r12 + KC0e0810*r13) + r23*(KC0e0611*r11 + KC0e0711*r12)
            k += 1
            KC0v[k] += r31*(KC0e0609*r11 + KC0e0709*r12 + KC0e0809*r13) + r32*(KC0e0610*r11 + KC0e0710*r12 + KC0e0810*r13) + r33*(KC0e0611*r11 + KC0e0711*r12)
            k += 1
            KC0v[k] += KC0e0814*r13**2 + r11*(KC0e0612*r11 + KC0e0712*r12) + r12*(KC0e0613*r11 + KC0e0713*r12)
            k += 1
            KC0v[k] += KC0e0814*r13*r23 + r21*(KC0e0612*r11 + KC0e0712*r12) + r22*(KC0e0613*r11 + KC0e0713*r12)
            k += 1
            KC0v[k] += KC0e0814*r13*r33 + r31*(KC0e0612*r11 + KC0e0712*r12) + r32*(KC0e0613*r11 + KC0e0713*r12)
            k += 1
            KC0v[k] += r11*(KC0e0615*r11 + KC0e0715*r12 + KC0e0815*r13) + r12*(KC0e0616*r11 + KC0e0716*r12 + KC0e0816*r13) + r13*(KC0e0617*r11 + KC0e0717*r12)
            k += 1
            KC0v[k] += r21*(KC0e0615*r11 + KC0e0715*r12 + KC0e0815*r13) + r22*(KC0e0616*r11 + KC0e0716*r12 + KC0e0816*r13) + r23*(KC0e0617*r11 + KC0e0717*r12)
            k += 1
            KC0v[k] += r31*(KC0e0615*r11 + KC0e0715*r12 + KC0e0815*r13) + r32*(KC0e0616*r11 + KC0e0716*r12 + KC0e0816*r13) + r33*(KC0e0617*r11 + KC0e0717*r12)
            k += 1
            KC0v[k] += KC0e0820*r13**2 + r11*(KC0e0618*r11 + KC0e0718*r12) + r12*(KC0e0619*r11 + KC0e0719*r12)
            k += 1
            KC0v[k] += KC0e0820*r13*r23 + r21*(KC0e0618*r11 + KC0e0718*r12) + r22*(KC0e0619*r11 + KC0e0719*r12)
            k += 1
            KC0v[k] += KC0e0820*r13*r33 + r31*(KC0e0618*r11 + KC0e0718*r12) + r32*(KC0e0619*r11 + KC0e0719*r12)
            k += 1
            KC0v[k] += r11*(KC0e0621*r11 + KC0e0721*r12 + KC0e0821*r13) + r12*(KC0e0622*r11 + KC0e0722*r12 + KC0e0822*r13) + r13*(KC0e0623*r11 + KC0e0723*r12)
            k += 1
            KC0v[k] += r21*(KC0e0621*r11 + KC0e0721*r12 + KC0e0821*r13) + r22*(KC0e0622*r11 + KC0e0722*r12 + KC0e0822*r13) + r23*(KC0e0623*r11 + KC0e0723*r12)
            k += 1
            KC0v[k] += r31*(KC0e0621*r11 + KC0e0721*r12 + KC0e0821*r13) + r32*(KC0e0622*r11 + KC0e0722*r12 + KC0e0822*r13) + r33*(KC0e0623*r11 + KC0e0723*r12)
            k += 1
            KC0v[k] += KC0e0208*r13*r23 + r11*(KC0e0006*r21 + KC0e0007*r22) + r12*(KC0e0106*r21 + KC0e0107*r22)
            k += 1
            KC0v[k] += KC0e0208*r23**2 + r21*(KC0e0006*r21 + KC0e0007*r22) + r22*(KC0e0106*r21 + KC0e0107*r22)
            k += 1
            KC0v[k] += KC0e0208*r23*r33 + r31*(KC0e0006*r21 + KC0e0007*r22) + r32*(KC0e0106*r21 + KC0e0107*r22)
            k += 1
            KC0v[k] += r11*(KC0e0306*r21 + KC0e0307*r22 + KC0e0308*r23) + r12*(KC0e0406*r21 + KC0e0407*r22 + KC0e0408*r23) + r13*(KC0e0506*r21 + KC0e0507*r22)
            k += 1
            KC0v[k] += r21*(KC0e0306*r21 + KC0e0307*r22 + KC0e0308*r23) + r22*(KC0e0406*r21 + KC0e0407*r22 + KC0e0408*r23) + r23*(KC0e0506*r21 + KC0e0507*r22)
            k += 1
            KC0v[k] += r31*(KC0e0306*r21 + KC0e0307*r22 + KC0e0308*r23) + r32*(KC0e0406*r21 + KC0e0407*r22 + KC0e0408*r23) + r33*(KC0e0506*r21 + KC0e0507*r22)
            k += 1
            KC0v[k] += KC0e0808*r13*r23 + r11*(KC0e0606*r21 + KC0e0607*r22) + r12*(KC0e0607*r21 + KC0e0707*r22)
            k += 1
            KC0v[k] += KC0e0808*r23**2 + r21*(KC0e0606*r21 + KC0e0607*r22) + r22*(KC0e0607*r21 + KC0e0707*r22)
            k += 1
            KC0v[k] += KC0e0808*r23*r33 + r31*(KC0e0606*r21 + KC0e0607*r22) + r32*(KC0e0607*r21 + KC0e0707*r22)
            k += 1
            KC0v[k] += r11*(KC0e0609*r21 + KC0e0709*r22 + KC0e0809*r23) + r12*(KC0e0610*r21 + KC0e0710*r22 + KC0e0810*r23) + r13*(KC0e0611*r21 + KC0e0711*r22)
            k += 1
            KC0v[k] += r21*(KC0e0609*r21 + KC0e0709*r22 + KC0e0809*r23) + r22*(KC0e0610*r21 + KC0e0710*r22 + KC0e0810*r23) + r23*(KC0e0611*r21 + KC0e0711*r22)
            k += 1
            KC0v[k] += r31*(KC0e0609*r21 + KC0e0709*r22 + KC0e0809*r23) + r32*(KC0e0610*r21 + KC0e0710*r22 + KC0e0810*r23) + r33*(KC0e0611*r21 + KC0e0711*r22)
            k += 1
            KC0v[k] += KC0e0814*r13*r23 + r11*(KC0e0612*r21 + KC0e0712*r22) + r12*(KC0e0613*r21 + KC0e0713*r22)
            k += 1
            KC0v[k] += KC0e0814*r23**2 + r21*(KC0e0612*r21 + KC0e0712*r22) + r22*(KC0e0613*r21 + KC0e0713*r22)
            k += 1
            KC0v[k] += KC0e0814*r23*r33 + r31*(KC0e0612*r21 + KC0e0712*r22) + r32*(KC0e0613*r21 + KC0e0713*r22)
            k += 1
            KC0v[k] += r11*(KC0e0615*r21 + KC0e0715*r22 + KC0e0815*r23) + r12*(KC0e0616*r21 + KC0e0716*r22 + KC0e0816*r23) + r13*(KC0e0617*r21 + KC0e0717*r22)
            k += 1
            KC0v[k] += r21*(KC0e0615*r21 + KC0e0715*r22 + KC0e0815*r23) + r22*(KC0e0616*r21 + KC0e0716*r22 + KC0e0816*r23) + r23*(KC0e0617*r21 + KC0e0717*r22)
            k += 1
            KC0v[k] += r31*(KC0e0615*r21 + KC0e0715*r22 + KC0e0815*r23) + r32*(KC0e0616*r21 + KC0e0716*r22 + KC0e0816*r23) + r33*(KC0e0617*r21 + KC0e0717*r22)
            k += 1
            KC0v[k] += KC0e0820*r13*r23 + r11*(KC0e0618*r21 + KC0e0718*r22) + r12*(KC0e0619*r21 + KC0e0719*r22)
            k += 1
            KC0v[k] += KC0e0820*r23**2 + r21*(KC0e0618*r21 + KC0e0718*r22) + r22*(KC0e0619*r21 + KC0e0719*r22)
            k += 1
            KC0v[k] += KC0e0820*r23*r33 + r31*(KC0e0618*r21 + KC0e0718*r22) + r32*(KC0e0619*r21 + KC0e0719*r22)
            k += 1
            KC0v[k] += r11*(KC0e0621*r21 + KC0e0721*r22 + KC0e0821*r23) + r12*(KC0e0622*r21 + KC0e0722*r22 + KC0e0822*r23) + r13*(KC0e0623*r21 + KC0e0723*r22)
            k += 1
            KC0v[k] += r21*(KC0e0621*r21 + KC0e0721*r22 + KC0e0821*r23) + r22*(KC0e0622*r21 + KC0e0722*r22 + KC0e0822*r23) + r23*(KC0e0623*r21 + KC0e0723*r22)
            k += 1
            KC0v[k] += r31*(KC0e0621*r21 + KC0e0721*r22 + KC0e0821*r23) + r32*(KC0e0622*r21 + KC0e0722*r22 + KC0e0822*r23) + r33*(KC0e0623*r21 + KC0e0723*r22)
            k += 1
            KC0v[k] += KC0e0208*r13*r33 + r11*(KC0e0006*r31 + KC0e0007*r32) + r12*(KC0e0106*r31 + KC0e0107*r32)
            k += 1
            KC0v[k] += KC0e0208*r23*r33 + r21*(KC0e0006*r31 + KC0e0007*r32) + r22*(KC0e0106*r31 + KC0e0107*r32)
            k += 1
            KC0v[k] += KC0e0208*r33**2 + r31*(KC0e0006*r31 + KC0e0007*r32) + r32*(KC0e0106*r31 + KC0e0107*r32)
            k += 1
            KC0v[k] += r11*(KC0e0306*r31 + KC0e0307*r32 + KC0e0308*r33) + r12*(KC0e0406*r31 + KC0e0407*r32 + KC0e0408*r33) + r13*(KC0e0506*r31 + KC0e0507*r32)
            k += 1
            KC0v[k] += r21*(KC0e0306*r31 + KC0e0307*r32 + KC0e0308*r33) + r22*(KC0e0406*r31 + KC0e0407*r32 + KC0e0408*r33) + r23*(KC0e0506*r31 + KC0e0507*r32)
            k += 1
            KC0v[k] += r31*(KC0e0306*r31 + KC0e0307*r32 + KC0e0308*r33) + r32*(KC0e0406*r31 + KC0e0407*r32 + KC0e0408*r33) + r33*(KC0e0506*r31 + KC0e0507*r32)
            k += 1
            KC0v[k] += KC0e0808*r13*r33 + r11*(KC0e0606*r31 + KC0e0607*r32) + r12*(KC0e0607*r31 + KC0e0707*r32)
            k += 1
            KC0v[k] += KC0e0808*r23*r33 + r21*(KC0e0606*r31 + KC0e0607*r32) + r22*(KC0e0607*r31 + KC0e0707*r32)
            k += 1
            KC0v[k] += KC0e0808*r33**2 + r31*(KC0e0606*r31 + KC0e0607*r32) + r32*(KC0e0607*r31 + KC0e0707*r32)
            k += 1
            KC0v[k] += r11*(KC0e0609*r31 + KC0e0709*r32 + KC0e0809*r33) + r12*(KC0e0610*r31 + KC0e0710*r32 + KC0e0810*r33) + r13*(KC0e0611*r31 + KC0e0711*r32)
            k += 1
            KC0v[k] += r21*(KC0e0609*r31 + KC0e0709*r32 + KC0e0809*r33) + r22*(KC0e0610*r31 + KC0e0710*r32 + KC0e0810*r33) + r23*(KC0e0611*r31 + KC0e0711*r32)
            k += 1
            KC0v[k] += r31*(KC0e0609*r31 + KC0e0709*r32 + KC0e0809*r33) + r32*(KC0e0610*r31 + KC0e0710*r32 + KC0e0810*r33) + r33*(KC0e0611*r31 + KC0e0711*r32)
            k += 1
            KC0v[k] += KC0e0814*r13*r33 + r11*(KC0e0612*r31 + KC0e0712*r32) + r12*(KC0e0613*r31 + KC0e0713*r32)
            k += 1
            KC0v[k] += KC0e0814*r23*r33 + r21*(KC0e0612*r31 + KC0e0712*r32) + r22*(KC0e0613*r31 + KC0e0713*r32)
            k += 1
            KC0v[k] += KC0e0814*r33**2 + r31*(KC0e0612*r31 + KC0e0712*r32) + r32*(KC0e0613*r31 + KC0e0713*r32)
            k += 1
            KC0v[k] += r11*(KC0e0615*r31 + KC0e0715*r32 + KC0e0815*r33) + r12*(KC0e0616*r31 + KC0e0716*r32 + KC0e0816*r33) + r13*(KC0e0617*r31 + KC0e0717*r32)
            k += 1
            KC0v[k] += r21*(KC0e0615*r31 + KC0e0715*r32 + KC0e0815*r33) + r22*(KC0e0616*r31 + KC0e0716*r32 + KC0e0816*r33) + r23*(KC0e0617*r31 + KC0e0717*r32)
            k += 1
            KC0v[k] += r31*(KC0e0615*r31 + KC0e0715*r32 + KC0e0815*r33) + r32*(KC0e0616*r31 + KC0e0716*r32 + KC0e0816*r33) + r33*(KC0e0617*r31 + KC0e0717*r32)
            k += 1
            KC0v[k] += KC0e0820*r13*r33 + r11*(KC0e0618*r31 + KC0e0718*r32) + r12*(KC0e0619*r31 + KC0e0719*r32)
            k += 1
            KC0v[k] += KC0e0820*r23*r33 + r21*(KC0e0618*r31 + KC0e0718*r32) + r22*(KC0e0619*r31 + KC0e0719*r32)
            k += 1
            KC0v[k] += KC0e0820*r33**2 + r31*(KC0e0618*r31 + KC0e0718*r32) + r32*(KC0e0619*r31 + KC0e0719*r32)
            k += 1
            KC0v[k] += r11*(KC0e0621*r31 + KC0e0721*r32 + KC0e0821*r33) + r12*(KC0e0622*r31 + KC0e0722*r32 + KC0e0822*r33) + r13*(KC0e0623*r31 + KC0e0723*r32)
            k += 1
            KC0v[k] += r21*(KC0e0621*r31 + KC0e0721*r32 + KC0e0821*r33) + r22*(KC0e0622*r31 + KC0e0722*r32 + KC0e0822*r33) + r23*(KC0e0623*r31 + KC0e0723*r32)
            k += 1
            KC0v[k] += r31*(KC0e0621*r31 + KC0e0721*r32 + KC0e0821*r33) + r32*(KC0e0622*r31 + KC0e0722*r32 + KC0e0822*r33) + r33*(KC0e0623*r31 + KC0e0723*r32)
            k += 1
            KC0v[k] += r11*(KC0e0009*r11 + KC0e0010*r12 + KC0e0011*r13) + r12*(KC0e0109*r11 + KC0e0110*r12 + KC0e0111*r13) + r13*(KC0e0209*r11 + KC0e0210*r12)
            k += 1
            KC0v[k] += r21*(KC0e0009*r11 + KC0e0010*r12 + KC0e0011*r13) + r22*(KC0e0109*r11 + KC0e0110*r12 + KC0e0111*r13) + r23*(KC0e0209*r11 + KC0e0210*r12)
            k += 1
            KC0v[k] += r31*(KC0e0009*r11 + KC0e0010*r12 + KC0e0011*r13) + r32*(KC0e0109*r11 + KC0e0110*r12 + KC0e0111*r13) + r33*(KC0e0209*r11 + KC0e0210*r12)
            k += 1
            KC0v[k] += KC0e0511*r13**2 + r11*(KC0e0309*r11 + KC0e0310*r12) + r12*(KC0e0409*r11 + KC0e0410*r12)
            k += 1
            KC0v[k] += KC0e0511*r13*r23 + r21*(KC0e0309*r11 + KC0e0310*r12) + r22*(KC0e0409*r11 + KC0e0410*r12)
            k += 1
            KC0v[k] += KC0e0511*r13*r33 + r31*(KC0e0309*r11 + KC0e0310*r12) + r32*(KC0e0409*r11 + KC0e0410*r12)
            k += 1
            KC0v[k] += r11*(KC0e0609*r11 + KC0e0610*r12 + KC0e0611*r13) + r12*(KC0e0709*r11 + KC0e0710*r12 + KC0e0711*r13) + r13*(KC0e0809*r11 + KC0e0810*r12)
            k += 1
            KC0v[k] += r21*(KC0e0609*r11 + KC0e0610*r12 + KC0e0611*r13) + r22*(KC0e0709*r11 + KC0e0710*r12 + KC0e0711*r13) + r23*(KC0e0809*r11 + KC0e0810*r12)
            k += 1
            KC0v[k] += r31*(KC0e0609*r11 + KC0e0610*r12 + KC0e0611*r13) + r32*(KC0e0709*r11 + KC0e0710*r12 + KC0e0711*r13) + r33*(KC0e0809*r11 + KC0e0810*r12)
            k += 1
            KC0v[k] += KC0e1111*r13**2 + r11*(KC0e0909*r11 + KC0e0910*r12) + r12*(KC0e0910*r11 + KC0e1010*r12)
            k += 1
            KC0v[k] += KC0e1111*r13*r23 + r21*(KC0e0909*r11 + KC0e0910*r12) + r22*(KC0e0910*r11 + KC0e1010*r12)
            k += 1
            KC0v[k] += KC0e1111*r13*r33 + r31*(KC0e0909*r11 + KC0e0910*r12) + r32*(KC0e0910*r11 + KC0e1010*r12)
            k += 1
            KC0v[k] += r11*(KC0e0912*r11 + KC0e1012*r12 + KC0e1112*r13) + r12*(KC0e0913*r11 + KC0e1013*r12 + KC0e1113*r13) + r13*(KC0e0914*r11 + KC0e1014*r12)
            k += 1
            KC0v[k] += r21*(KC0e0912*r11 + KC0e1012*r12 + KC0e1112*r13) + r22*(KC0e0913*r11 + KC0e1013*r12 + KC0e1113*r13) + r23*(KC0e0914*r11 + KC0e1014*r12)
            k += 1
            KC0v[k] += r31*(KC0e0912*r11 + KC0e1012*r12 + KC0e1112*r13) + r32*(KC0e0913*r11 + KC0e1013*r12 + KC0e1113*r13) + r33*(KC0e0914*r11 + KC0e1014*r12)
            k += 1
            KC0v[k] += KC0e1117*r13**2 + r11*(KC0e0915*r11 + KC0e1015*r12) + r12*(KC0e0916*r11 + KC0e1016*r12)
            k += 1
            KC0v[k] += KC0e1117*r13*r23 + r21*(KC0e0915*r11 + KC0e1015*r12) + r22*(KC0e0916*r11 + KC0e1016*r12)
            k += 1
            KC0v[k] += KC0e1117*r13*r33 + r31*(KC0e0915*r11 + KC0e1015*r12) + r32*(KC0e0916*r11 + KC0e1016*r12)
            k += 1
            KC0v[k] += r11*(KC0e0918*r11 + KC0e1018*r12 + KC0e1118*r13) + r12*(KC0e0919*r11 + KC0e1019*r12 + KC0e1119*r13) + r13*(KC0e0920*r11 + KC0e1020*r12)
            k += 1
            KC0v[k] += r21*(KC0e0918*r11 + KC0e1018*r12 + KC0e1118*r13) + r22*(KC0e0919*r11 + KC0e1019*r12 + KC0e1119*r13) + r23*(KC0e0920*r11 + KC0e1020*r12)
            k += 1
            KC0v[k] += r31*(KC0e0918*r11 + KC0e1018*r12 + KC0e1118*r13) + r32*(KC0e0919*r11 + KC0e1019*r12 + KC0e1119*r13) + r33*(KC0e0920*r11 + KC0e1020*r12)
            k += 1
            KC0v[k] += KC0e1123*r13**2 + r11*(KC0e0921*r11 + KC0e1021*r12) + r12*(KC0e0922*r11 + KC0e1022*r12)
            k += 1
            KC0v[k] += KC0e1123*r13*r23 + r21*(KC0e0921*r11 + KC0e1021*r12) + r22*(KC0e0922*r11 + KC0e1022*r12)
            k += 1
            KC0v[k] += KC0e1123*r13*r33 + r31*(KC0e0921*r11 + KC0e1021*r12) + r32*(KC0e0922*r11 + KC0e1022*r12)
            k += 1
            KC0v[k] += r11*(KC0e0009*r21 + KC0e0010*r22 + KC0e0011*r23) + r12*(KC0e0109*r21 + KC0e0110*r22 + KC0e0111*r23) + r13*(KC0e0209*r21 + KC0e0210*r22)
            k += 1
            KC0v[k] += r21*(KC0e0009*r21 + KC0e0010*r22 + KC0e0011*r23) + r22*(KC0e0109*r21 + KC0e0110*r22 + KC0e0111*r23) + r23*(KC0e0209*r21 + KC0e0210*r22)
            k += 1
            KC0v[k] += r31*(KC0e0009*r21 + KC0e0010*r22 + KC0e0011*r23) + r32*(KC0e0109*r21 + KC0e0110*r22 + KC0e0111*r23) + r33*(KC0e0209*r21 + KC0e0210*r22)
            k += 1
            KC0v[k] += KC0e0511*r13*r23 + r11*(KC0e0309*r21 + KC0e0310*r22) + r12*(KC0e0409*r21 + KC0e0410*r22)
            k += 1
            KC0v[k] += KC0e0511*r23**2 + r21*(KC0e0309*r21 + KC0e0310*r22) + r22*(KC0e0409*r21 + KC0e0410*r22)
            k += 1
            KC0v[k] += KC0e0511*r23*r33 + r31*(KC0e0309*r21 + KC0e0310*r22) + r32*(KC0e0409*r21 + KC0e0410*r22)
            k += 1
            KC0v[k] += r11*(KC0e0609*r21 + KC0e0610*r22 + KC0e0611*r23) + r12*(KC0e0709*r21 + KC0e0710*r22 + KC0e0711*r23) + r13*(KC0e0809*r21 + KC0e0810*r22)
            k += 1
            KC0v[k] += r21*(KC0e0609*r21 + KC0e0610*r22 + KC0e0611*r23) + r22*(KC0e0709*r21 + KC0e0710*r22 + KC0e0711*r23) + r23*(KC0e0809*r21 + KC0e0810*r22)
            k += 1
            KC0v[k] += r31*(KC0e0609*r21 + KC0e0610*r22 + KC0e0611*r23) + r32*(KC0e0709*r21 + KC0e0710*r22 + KC0e0711*r23) + r33*(KC0e0809*r21 + KC0e0810*r22)
            k += 1
            KC0v[k] += KC0e1111*r13*r23 + r11*(KC0e0909*r21 + KC0e0910*r22) + r12*(KC0e0910*r21 + KC0e1010*r22)
            k += 1
            KC0v[k] += KC0e1111*r23**2 + r21*(KC0e0909*r21 + KC0e0910*r22) + r22*(KC0e0910*r21 + KC0e1010*r22)
            k += 1
            KC0v[k] += KC0e1111*r23*r33 + r31*(KC0e0909*r21 + KC0e0910*r22) + r32*(KC0e0910*r21 + KC0e1010*r22)
            k += 1
            KC0v[k] += r11*(KC0e0912*r21 + KC0e1012*r22 + KC0e1112*r23) + r12*(KC0e0913*r21 + KC0e1013*r22 + KC0e1113*r23) + r13*(KC0e0914*r21 + KC0e1014*r22)
            k += 1
            KC0v[k] += r21*(KC0e0912*r21 + KC0e1012*r22 + KC0e1112*r23) + r22*(KC0e0913*r21 + KC0e1013*r22 + KC0e1113*r23) + r23*(KC0e0914*r21 + KC0e1014*r22)
            k += 1
            KC0v[k] += r31*(KC0e0912*r21 + KC0e1012*r22 + KC0e1112*r23) + r32*(KC0e0913*r21 + KC0e1013*r22 + KC0e1113*r23) + r33*(KC0e0914*r21 + KC0e1014*r22)
            k += 1
            KC0v[k] += KC0e1117*r13*r23 + r11*(KC0e0915*r21 + KC0e1015*r22) + r12*(KC0e0916*r21 + KC0e1016*r22)
            k += 1
            KC0v[k] += KC0e1117*r23**2 + r21*(KC0e0915*r21 + KC0e1015*r22) + r22*(KC0e0916*r21 + KC0e1016*r22)
            k += 1
            KC0v[k] += KC0e1117*r23*r33 + r31*(KC0e0915*r21 + KC0e1015*r22) + r32*(KC0e0916*r21 + KC0e1016*r22)
            k += 1
            KC0v[k] += r11*(KC0e0918*r21 + KC0e1018*r22 + KC0e1118*r23) + r12*(KC0e0919*r21 + KC0e1019*r22 + KC0e1119*r23) + r13*(KC0e0920*r21 + KC0e1020*r22)
            k += 1
            KC0v[k] += r21*(KC0e0918*r21 + KC0e1018*r22 + KC0e1118*r23) + r22*(KC0e0919*r21 + KC0e1019*r22 + KC0e1119*r23) + r23*(KC0e0920*r21 + KC0e1020*r22)
            k += 1
            KC0v[k] += r31*(KC0e0918*r21 + KC0e1018*r22 + KC0e1118*r23) + r32*(KC0e0919*r21 + KC0e1019*r22 + KC0e1119*r23) + r33*(KC0e0920*r21 + KC0e1020*r22)
            k += 1
            KC0v[k] += KC0e1123*r13*r23 + r11*(KC0e0921*r21 + KC0e1021*r22) + r12*(KC0e0922*r21 + KC0e1022*r22)
            k += 1
            KC0v[k] += KC0e1123*r23**2 + r21*(KC0e0921*r21 + KC0e1021*r22) + r22*(KC0e0922*r21 + KC0e1022*r22)
            k += 1
            KC0v[k] += KC0e1123*r23*r33 + r31*(KC0e0921*r21 + KC0e1021*r22) + r32*(KC0e0922*r21 + KC0e1022*r22)
            k += 1
            KC0v[k] += r11*(KC0e0009*r31 + KC0e0010*r32 + KC0e0011*r33) + r12*(KC0e0109*r31 + KC0e0110*r32 + KC0e0111*r33) + r13*(KC0e0209*r31 + KC0e0210*r32)
            k += 1
            KC0v[k] += r21*(KC0e0009*r31 + KC0e0010*r32 + KC0e0011*r33) + r22*(KC0e0109*r31 + KC0e0110*r32 + KC0e0111*r33) + r23*(KC0e0209*r31 + KC0e0210*r32)
            k += 1
            KC0v[k] += r31*(KC0e0009*r31 + KC0e0010*r32 + KC0e0011*r33) + r32*(KC0e0109*r31 + KC0e0110*r32 + KC0e0111*r33) + r33*(KC0e0209*r31 + KC0e0210*r32)
            k += 1
            KC0v[k] += KC0e0511*r13*r33 + r11*(KC0e0309*r31 + KC0e0310*r32) + r12*(KC0e0409*r31 + KC0e0410*r32)
            k += 1
            KC0v[k] += KC0e0511*r23*r33 + r21*(KC0e0309*r31 + KC0e0310*r32) + r22*(KC0e0409*r31 + KC0e0410*r32)
            k += 1
            KC0v[k] += KC0e0511*r33**2 + r31*(KC0e0309*r31 + KC0e0310*r32) + r32*(KC0e0409*r31 + KC0e0410*r32)
            k += 1
            KC0v[k] += r11*(KC0e0609*r31 + KC0e0610*r32 + KC0e0611*r33) + r12*(KC0e0709*r31 + KC0e0710*r32 + KC0e0711*r33) + r13*(KC0e0809*r31 + KC0e0810*r32)
            k += 1
            KC0v[k] += r21*(KC0e0609*r31 + KC0e0610*r32 + KC0e0611*r33) + r22*(KC0e0709*r31 + KC0e0710*r32 + KC0e0711*r33) + r23*(KC0e0809*r31 + KC0e0810*r32)
            k += 1
            KC0v[k] += r31*(KC0e0609*r31 + KC0e0610*r32 + KC0e0611*r33) + r32*(KC0e0709*r31 + KC0e0710*r32 + KC0e0711*r33) + r33*(KC0e0809*r31 + KC0e0810*r32)
            k += 1
            KC0v[k] += KC0e1111*r13*r33 + r11*(KC0e0909*r31 + KC0e0910*r32) + r12*(KC0e0910*r31 + KC0e1010*r32)
            k += 1
            KC0v[k] += KC0e1111*r23*r33 + r21*(KC0e0909*r31 + KC0e0910*r32) + r22*(KC0e0910*r31 + KC0e1010*r32)
            k += 1
            KC0v[k] += KC0e1111*r33**2 + r31*(KC0e0909*r31 + KC0e0910*r32) + r32*(KC0e0910*r31 + KC0e1010*r32)
            k += 1
            KC0v[k] += r11*(KC0e0912*r31 + KC0e1012*r32 + KC0e1112*r33) + r12*(KC0e0913*r31 + KC0e1013*r32 + KC0e1113*r33) + r13*(KC0e0914*r31 + KC0e1014*r32)
            k += 1
            KC0v[k] += r21*(KC0e0912*r31 + KC0e1012*r32 + KC0e1112*r33) + r22*(KC0e0913*r31 + KC0e1013*r32 + KC0e1113*r33) + r23*(KC0e0914*r31 + KC0e1014*r32)
            k += 1
            KC0v[k] += r31*(KC0e0912*r31 + KC0e1012*r32 + KC0e1112*r33) + r32*(KC0e0913*r31 + KC0e1013*r32 + KC0e1113*r33) + r33*(KC0e0914*r31 + KC0e1014*r32)
            k += 1
            KC0v[k] += KC0e1117*r13*r33 + r11*(KC0e0915*r31 + KC0e1015*r32) + r12*(KC0e0916*r31 + KC0e1016*r32)
            k += 1
            KC0v[k] += KC0e1117*r23*r33 + r21*(KC0e0915*r31 + KC0e1015*r32) + r22*(KC0e0916*r31 + KC0e1016*r32)
            k += 1
            KC0v[k] += KC0e1117*r33**2 + r31*(KC0e0915*r31 + KC0e1015*r32) + r32*(KC0e0916*r31 + KC0e1016*r32)
            k += 1
            KC0v[k] += r11*(KC0e0918*r31 + KC0e1018*r32 + KC0e1118*r33) + r12*(KC0e0919*r31 + KC0e1019*r32 + KC0e1119*r33) + r13*(KC0e0920*r31 + KC0e1020*r32)
            k += 1
            KC0v[k] += r21*(KC0e0918*r31 + KC0e1018*r32 + KC0e1118*r33) + r22*(KC0e0919*r31 + KC0e1019*r32 + KC0e1119*r33) + r23*(KC0e0920*r31 + KC0e1020*r32)
            k += 1
            KC0v[k] += r31*(KC0e0918*r31 + KC0e1018*r32 + KC0e1118*r33) + r32*(KC0e0919*r31 + KC0e1019*r32 + KC0e1119*r33) + r33*(KC0e0920*r31 + KC0e1020*r32)
            k += 1
            KC0v[k] += KC0e1123*r13*r33 + r11*(KC0e0921*r31 + KC0e1021*r32) + r12*(KC0e0922*r31 + KC0e1022*r32)
            k += 1
            KC0v[k] += KC0e1123*r23*r33 + r21*(KC0e0921*r31 + KC0e1021*r32) + r22*(KC0e0922*r31 + KC0e1022*r32)
            k += 1
            KC0v[k] += KC0e1123*r33**2 + r31*(KC0e0921*r31 + KC0e1021*r32) + r32*(KC0e0922*r31 + KC0e1022*r32)
            k += 1
            KC0v[k] += KC0e0214*r13**2 + r11*(KC0e0012*r11 + KC0e0013*r12) + r12*(KC0e0112*r11 + KC0e0113*r12)
            k += 1
            KC0v[k] += KC0e0214*r13*r23 + r21*(KC0e0012*r11 + KC0e0013*r12) + r22*(KC0e0112*r11 + KC0e0113*r12)
            k += 1
            KC0v[k] += KC0e0214*r13*r33 + r31*(KC0e0012*r11 + KC0e0013*r12) + r32*(KC0e0112*r11 + KC0e0113*r12)
            k += 1
            KC0v[k] += r11*(KC0e0312*r11 + KC0e0313*r12 + KC0e0314*r13) + r12*(KC0e0412*r11 + KC0e0413*r12 + KC0e0414*r13) + r13*(KC0e0512*r11 + KC0e0513*r12)
            k += 1
            KC0v[k] += r21*(KC0e0312*r11 + KC0e0313*r12 + KC0e0314*r13) + r22*(KC0e0412*r11 + KC0e0413*r12 + KC0e0414*r13) + r23*(KC0e0512*r11 + KC0e0513*r12)
            k += 1
            KC0v[k] += r31*(KC0e0312*r11 + KC0e0313*r12 + KC0e0314*r13) + r32*(KC0e0412*r11 + KC0e0413*r12 + KC0e0414*r13) + r33*(KC0e0512*r11 + KC0e0513*r12)
            k += 1
            KC0v[k] += KC0e0814*r13**2 + r11*(KC0e0612*r11 + KC0e0613*r12) + r12*(KC0e0712*r11 + KC0e0713*r12)
            k += 1
            KC0v[k] += KC0e0814*r13*r23 + r21*(KC0e0612*r11 + KC0e0613*r12) + r22*(KC0e0712*r11 + KC0e0713*r12)
            k += 1
            KC0v[k] += KC0e0814*r13*r33 + r31*(KC0e0612*r11 + KC0e0613*r12) + r32*(KC0e0712*r11 + KC0e0713*r12)
            k += 1
            KC0v[k] += r11*(KC0e0912*r11 + KC0e0913*r12 + KC0e0914*r13) + r12*(KC0e1012*r11 + KC0e1013*r12 + KC0e1014*r13) + r13*(KC0e1112*r11 + KC0e1113*r12)
            k += 1
            KC0v[k] += r21*(KC0e0912*r11 + KC0e0913*r12 + KC0e0914*r13) + r22*(KC0e1012*r11 + KC0e1013*r12 + KC0e1014*r13) + r23*(KC0e1112*r11 + KC0e1113*r12)
            k += 1
            KC0v[k] += r31*(KC0e0912*r11 + KC0e0913*r12 + KC0e0914*r13) + r32*(KC0e1012*r11 + KC0e1013*r12 + KC0e1014*r13) + r33*(KC0e1112*r11 + KC0e1113*r12)
            k += 1
            KC0v[k] += KC0e1414*r13**2 + r11*(KC0e1212*r11 + KC0e1213*r12) + r12*(KC0e1213*r11 + KC0e1313*r12)
            k += 1
            KC0v[k] += KC0e1414*r13*r23 + r21*(KC0e1212*r11 + KC0e1213*r12) + r22*(KC0e1213*r11 + KC0e1313*r12)
            k += 1
            KC0v[k] += KC0e1414*r13*r33 + r31*(KC0e1212*r11 + KC0e1213*r12) + r32*(KC0e1213*r11 + KC0e1313*r12)
            k += 1
            KC0v[k] += r11*(KC0e1215*r11 + KC0e1315*r12 + KC0e1415*r13) + r12*(KC0e1216*r11 + KC0e1316*r12 + KC0e1416*r13) + r13*(KC0e1217*r11 + KC0e1317*r12)
            k += 1
            KC0v[k] += r21*(KC0e1215*r11 + KC0e1315*r12 + KC0e1415*r13) + r22*(KC0e1216*r11 + KC0e1316*r12 + KC0e1416*r13) + r23*(KC0e1217*r11 + KC0e1317*r12)
            k += 1
            KC0v[k] += r31*(KC0e1215*r11 + KC0e1315*r12 + KC0e1415*r13) + r32*(KC0e1216*r11 + KC0e1316*r12 + KC0e1416*r13) + r33*(KC0e1217*r11 + KC0e1317*r12)
            k += 1
            KC0v[k] += KC0e1420*r13**2 + r11*(KC0e1218*r11 + KC0e1318*r12) + r12*(KC0e1219*r11 + KC0e1319*r12)
            k += 1
            KC0v[k] += KC0e1420*r13*r23 + r21*(KC0e1218*r11 + KC0e1318*r12) + r22*(KC0e1219*r11 + KC0e1319*r12)
            k += 1
            KC0v[k] += KC0e1420*r13*r33 + r31*(KC0e1218*r11 + KC0e1318*r12) + r32*(KC0e1219*r11 + KC0e1319*r12)
            k += 1
            KC0v[k] += r11*(KC0e1221*r11 + KC0e1321*r12 + KC0e1421*r13) + r12*(KC0e1222*r11 + KC0e1322*r12 + KC0e1422*r13) + r13*(KC0e1223*r11 + KC0e1323*r12)
            k += 1
            KC0v[k] += r21*(KC0e1221*r11 + KC0e1321*r12 + KC0e1421*r13) + r22*(KC0e1222*r11 + KC0e1322*r12 + KC0e1422*r13) + r23*(KC0e1223*r11 + KC0e1323*r12)
            k += 1
            KC0v[k] += r31*(KC0e1221*r11 + KC0e1321*r12 + KC0e1421*r13) + r32*(KC0e1222*r11 + KC0e1322*r12 + KC0e1422*r13) + r33*(KC0e1223*r11 + KC0e1323*r12)
            k += 1
            KC0v[k] += KC0e0214*r13*r23 + r11*(KC0e0012*r21 + KC0e0013*r22) + r12*(KC0e0112*r21 + KC0e0113*r22)
            k += 1
            KC0v[k] += KC0e0214*r23**2 + r21*(KC0e0012*r21 + KC0e0013*r22) + r22*(KC0e0112*r21 + KC0e0113*r22)
            k += 1
            KC0v[k] += KC0e0214*r23*r33 + r31*(KC0e0012*r21 + KC0e0013*r22) + r32*(KC0e0112*r21 + KC0e0113*r22)
            k += 1
            KC0v[k] += r11*(KC0e0312*r21 + KC0e0313*r22 + KC0e0314*r23) + r12*(KC0e0412*r21 + KC0e0413*r22 + KC0e0414*r23) + r13*(KC0e0512*r21 + KC0e0513*r22)
            k += 1
            KC0v[k] += r21*(KC0e0312*r21 + KC0e0313*r22 + KC0e0314*r23) + r22*(KC0e0412*r21 + KC0e0413*r22 + KC0e0414*r23) + r23*(KC0e0512*r21 + KC0e0513*r22)
            k += 1
            KC0v[k] += r31*(KC0e0312*r21 + KC0e0313*r22 + KC0e0314*r23) + r32*(KC0e0412*r21 + KC0e0413*r22 + KC0e0414*r23) + r33*(KC0e0512*r21 + KC0e0513*r22)
            k += 1
            KC0v[k] += KC0e0814*r13*r23 + r11*(KC0e0612*r21 + KC0e0613*r22) + r12*(KC0e0712*r21 + KC0e0713*r22)
            k += 1
            KC0v[k] += KC0e0814*r23**2 + r21*(KC0e0612*r21 + KC0e0613*r22) + r22*(KC0e0712*r21 + KC0e0713*r22)
            k += 1
            KC0v[k] += KC0e0814*r23*r33 + r31*(KC0e0612*r21 + KC0e0613*r22) + r32*(KC0e0712*r21 + KC0e0713*r22)
            k += 1
            KC0v[k] += r11*(KC0e0912*r21 + KC0e0913*r22 + KC0e0914*r23) + r12*(KC0e1012*r21 + KC0e1013*r22 + KC0e1014*r23) + r13*(KC0e1112*r21 + KC0e1113*r22)
            k += 1
            KC0v[k] += r21*(KC0e0912*r21 + KC0e0913*r22 + KC0e0914*r23) + r22*(KC0e1012*r21 + KC0e1013*r22 + KC0e1014*r23) + r23*(KC0e1112*r21 + KC0e1113*r22)
            k += 1
            KC0v[k] += r31*(KC0e0912*r21 + KC0e0913*r22 + KC0e0914*r23) + r32*(KC0e1012*r21 + KC0e1013*r22 + KC0e1014*r23) + r33*(KC0e1112*r21 + KC0e1113*r22)
            k += 1
            KC0v[k] += KC0e1414*r13*r23 + r11*(KC0e1212*r21 + KC0e1213*r22) + r12*(KC0e1213*r21 + KC0e1313*r22)
            k += 1
            KC0v[k] += KC0e1414*r23**2 + r21*(KC0e1212*r21 + KC0e1213*r22) + r22*(KC0e1213*r21 + KC0e1313*r22)
            k += 1
            KC0v[k] += KC0e1414*r23*r33 + r31*(KC0e1212*r21 + KC0e1213*r22) + r32*(KC0e1213*r21 + KC0e1313*r22)
            k += 1
            KC0v[k] += r11*(KC0e1215*r21 + KC0e1315*r22 + KC0e1415*r23) + r12*(KC0e1216*r21 + KC0e1316*r22 + KC0e1416*r23) + r13*(KC0e1217*r21 + KC0e1317*r22)
            k += 1
            KC0v[k] += r21*(KC0e1215*r21 + KC0e1315*r22 + KC0e1415*r23) + r22*(KC0e1216*r21 + KC0e1316*r22 + KC0e1416*r23) + r23*(KC0e1217*r21 + KC0e1317*r22)
            k += 1
            KC0v[k] += r31*(KC0e1215*r21 + KC0e1315*r22 + KC0e1415*r23) + r32*(KC0e1216*r21 + KC0e1316*r22 + KC0e1416*r23) + r33*(KC0e1217*r21 + KC0e1317*r22)
            k += 1
            KC0v[k] += KC0e1420*r13*r23 + r11*(KC0e1218*r21 + KC0e1318*r22) + r12*(KC0e1219*r21 + KC0e1319*r22)
            k += 1
            KC0v[k] += KC0e1420*r23**2 + r21*(KC0e1218*r21 + KC0e1318*r22) + r22*(KC0e1219*r21 + KC0e1319*r22)
            k += 1
            KC0v[k] += KC0e1420*r23*r33 + r31*(KC0e1218*r21 + KC0e1318*r22) + r32*(KC0e1219*r21 + KC0e1319*r22)
            k += 1
            KC0v[k] += r11*(KC0e1221*r21 + KC0e1321*r22 + KC0e1421*r23) + r12*(KC0e1222*r21 + KC0e1322*r22 + KC0e1422*r23) + r13*(KC0e1223*r21 + KC0e1323*r22)
            k += 1
            KC0v[k] += r21*(KC0e1221*r21 + KC0e1321*r22 + KC0e1421*r23) + r22*(KC0e1222*r21 + KC0e1322*r22 + KC0e1422*r23) + r23*(KC0e1223*r21 + KC0e1323*r22)
            k += 1
            KC0v[k] += r31*(KC0e1221*r21 + KC0e1321*r22 + KC0e1421*r23) + r32*(KC0e1222*r21 + KC0e1322*r22 + KC0e1422*r23) + r33*(KC0e1223*r21 + KC0e1323*r22)
            k += 1
            KC0v[k] += KC0e0214*r13*r33 + r11*(KC0e0012*r31 + KC0e0013*r32) + r12*(KC0e0112*r31 + KC0e0113*r32)
            k += 1
            KC0v[k] += KC0e0214*r23*r33 + r21*(KC0e0012*r31 + KC0e0013*r32) + r22*(KC0e0112*r31 + KC0e0113*r32)
            k += 1
            KC0v[k] += KC0e0214*r33**2 + r31*(KC0e0012*r31 + KC0e0013*r32) + r32*(KC0e0112*r31 + KC0e0113*r32)
            k += 1
            KC0v[k] += r11*(KC0e0312*r31 + KC0e0313*r32 + KC0e0314*r33) + r12*(KC0e0412*r31 + KC0e0413*r32 + KC0e0414*r33) + r13*(KC0e0512*r31 + KC0e0513*r32)
            k += 1
            KC0v[k] += r21*(KC0e0312*r31 + KC0e0313*r32 + KC0e0314*r33) + r22*(KC0e0412*r31 + KC0e0413*r32 + KC0e0414*r33) + r23*(KC0e0512*r31 + KC0e0513*r32)
            k += 1
            KC0v[k] += r31*(KC0e0312*r31 + KC0e0313*r32 + KC0e0314*r33) + r32*(KC0e0412*r31 + KC0e0413*r32 + KC0e0414*r33) + r33*(KC0e0512*r31 + KC0e0513*r32)
            k += 1
            KC0v[k] += KC0e0814*r13*r33 + r11*(KC0e0612*r31 + KC0e0613*r32) + r12*(KC0e0712*r31 + KC0e0713*r32)
            k += 1
            KC0v[k] += KC0e0814*r23*r33 + r21*(KC0e0612*r31 + KC0e0613*r32) + r22*(KC0e0712*r31 + KC0e0713*r32)
            k += 1
            KC0v[k] += KC0e0814*r33**2 + r31*(KC0e0612*r31 + KC0e0613*r32) + r32*(KC0e0712*r31 + KC0e0713*r32)
            k += 1
            KC0v[k] += r11*(KC0e0912*r31 + KC0e0913*r32 + KC0e0914*r33) + r12*(KC0e1012*r31 + KC0e1013*r32 + KC0e1014*r33) + r13*(KC0e1112*r31 + KC0e1113*r32)
            k += 1
            KC0v[k] += r21*(KC0e0912*r31 + KC0e0913*r32 + KC0e0914*r33) + r22*(KC0e1012*r31 + KC0e1013*r32 + KC0e1014*r33) + r23*(KC0e1112*r31 + KC0e1113*r32)
            k += 1
            KC0v[k] += r31*(KC0e0912*r31 + KC0e0913*r32 + KC0e0914*r33) + r32*(KC0e1012*r31 + KC0e1013*r32 + KC0e1014*r33) + r33*(KC0e1112*r31 + KC0e1113*r32)
            k += 1
            KC0v[k] += KC0e1414*r13*r33 + r11*(KC0e1212*r31 + KC0e1213*r32) + r12*(KC0e1213*r31 + KC0e1313*r32)
            k += 1
            KC0v[k] += KC0e1414*r23*r33 + r21*(KC0e1212*r31 + KC0e1213*r32) + r22*(KC0e1213*r31 + KC0e1313*r32)
            k += 1
            KC0v[k] += KC0e1414*r33**2 + r31*(KC0e1212*r31 + KC0e1213*r32) + r32*(KC0e1213*r31 + KC0e1313*r32)
            k += 1
            KC0v[k] += r11*(KC0e1215*r31 + KC0e1315*r32 + KC0e1415*r33) + r12*(KC0e1216*r31 + KC0e1316*r32 + KC0e1416*r33) + r13*(KC0e1217*r31 + KC0e1317*r32)
            k += 1
            KC0v[k] += r21*(KC0e1215*r31 + KC0e1315*r32 + KC0e1415*r33) + r22*(KC0e1216*r31 + KC0e1316*r32 + KC0e1416*r33) + r23*(KC0e1217*r31 + KC0e1317*r32)
            k += 1
            KC0v[k] += r31*(KC0e1215*r31 + KC0e1315*r32 + KC0e1415*r33) + r32*(KC0e1216*r31 + KC0e1316*r32 + KC0e1416*r33) + r33*(KC0e1217*r31 + KC0e1317*r32)
            k += 1
            KC0v[k] += KC0e1420*r13*r33 + r11*(KC0e1218*r31 + KC0e1318*r32) + r12*(KC0e1219*r31 + KC0e1319*r32)
            k += 1
            KC0v[k] += KC0e1420*r23*r33 + r21*(KC0e1218*r31 + KC0e1318*r32) + r22*(KC0e1219*r31 + KC0e1319*r32)
            k += 1
            KC0v[k] += KC0e1420*r33**2 + r31*(KC0e1218*r31 + KC0e1318*r32) + r32*(KC0e1219*r31 + KC0e1319*r32)
            k += 1
            KC0v[k] += r11*(KC0e1221*r31 + KC0e1321*r32 + KC0e1421*r33) + r12*(KC0e1222*r31 + KC0e1322*r32 + KC0e1422*r33) + r13*(KC0e1223*r31 + KC0e1323*r32)
            k += 1
            KC0v[k] += r21*(KC0e1221*r31 + KC0e1321*r32 + KC0e1421*r33) + r22*(KC0e1222*r31 + KC0e1322*r32 + KC0e1422*r33) + r23*(KC0e1223*r31 + KC0e1323*r32)
            k += 1
            KC0v[k] += r31*(KC0e1221*r31 + KC0e1321*r32 + KC0e1421*r33) + r32*(KC0e1222*r31 + KC0e1322*r32 + KC0e1422*r33) + r33*(KC0e1223*r31 + KC0e1323*r32)
            k += 1
            KC0v[k] += r11*(KC0e0015*r11 + KC0e0016*r12 + KC0e0017*r13) + r12*(KC0e0115*r11 + KC0e0116*r12 + KC0e0117*r13) + r13*(KC0e0215*r11 + KC0e0216*r12)
            k += 1
            KC0v[k] += r21*(KC0e0015*r11 + KC0e0016*r12 + KC0e0017*r13) + r22*(KC0e0115*r11 + KC0e0116*r12 + KC0e0117*r13) + r23*(KC0e0215*r11 + KC0e0216*r12)
            k += 1
            KC0v[k] += r31*(KC0e0015*r11 + KC0e0016*r12 + KC0e0017*r13) + r32*(KC0e0115*r11 + KC0e0116*r12 + KC0e0117*r13) + r33*(KC0e0215*r11 + KC0e0216*r12)
            k += 1
            KC0v[k] += KC0e0517*r13**2 + r11*(KC0e0315*r11 + KC0e0316*r12) + r12*(KC0e0415*r11 + KC0e0416*r12)
            k += 1
            KC0v[k] += KC0e0517*r13*r23 + r21*(KC0e0315*r11 + KC0e0316*r12) + r22*(KC0e0415*r11 + KC0e0416*r12)
            k += 1
            KC0v[k] += KC0e0517*r13*r33 + r31*(KC0e0315*r11 + KC0e0316*r12) + r32*(KC0e0415*r11 + KC0e0416*r12)
            k += 1
            KC0v[k] += r11*(KC0e0615*r11 + KC0e0616*r12 + KC0e0617*r13) + r12*(KC0e0715*r11 + KC0e0716*r12 + KC0e0717*r13) + r13*(KC0e0815*r11 + KC0e0816*r12)
            k += 1
            KC0v[k] += r21*(KC0e0615*r11 + KC0e0616*r12 + KC0e0617*r13) + r22*(KC0e0715*r11 + KC0e0716*r12 + KC0e0717*r13) + r23*(KC0e0815*r11 + KC0e0816*r12)
            k += 1
            KC0v[k] += r31*(KC0e0615*r11 + KC0e0616*r12 + KC0e0617*r13) + r32*(KC0e0715*r11 + KC0e0716*r12 + KC0e0717*r13) + r33*(KC0e0815*r11 + KC0e0816*r12)
            k += 1
            KC0v[k] += KC0e1117*r13**2 + r11*(KC0e0915*r11 + KC0e0916*r12) + r12*(KC0e1015*r11 + KC0e1016*r12)
            k += 1
            KC0v[k] += KC0e1117*r13*r23 + r21*(KC0e0915*r11 + KC0e0916*r12) + r22*(KC0e1015*r11 + KC0e1016*r12)
            k += 1
            KC0v[k] += KC0e1117*r13*r33 + r31*(KC0e0915*r11 + KC0e0916*r12) + r32*(KC0e1015*r11 + KC0e1016*r12)
            k += 1
            KC0v[k] += r11*(KC0e1215*r11 + KC0e1216*r12 + KC0e1217*r13) + r12*(KC0e1315*r11 + KC0e1316*r12 + KC0e1317*r13) + r13*(KC0e1415*r11 + KC0e1416*r12)
            k += 1
            KC0v[k] += r21*(KC0e1215*r11 + KC0e1216*r12 + KC0e1217*r13) + r22*(KC0e1315*r11 + KC0e1316*r12 + KC0e1317*r13) + r23*(KC0e1415*r11 + KC0e1416*r12)
            k += 1
            KC0v[k] += r31*(KC0e1215*r11 + KC0e1216*r12 + KC0e1217*r13) + r32*(KC0e1315*r11 + KC0e1316*r12 + KC0e1317*r13) + r33*(KC0e1415*r11 + KC0e1416*r12)
            k += 1
            KC0v[k] += KC0e1717*r13**2 + r11*(KC0e1515*r11 + KC0e1516*r12) + r12*(KC0e1516*r11 + KC0e1616*r12)
            k += 1
            KC0v[k] += KC0e1717*r13*r23 + r21*(KC0e1515*r11 + KC0e1516*r12) + r22*(KC0e1516*r11 + KC0e1616*r12)
            k += 1
            KC0v[k] += KC0e1717*r13*r33 + r31*(KC0e1515*r11 + KC0e1516*r12) + r32*(KC0e1516*r11 + KC0e1616*r12)
            k += 1
            KC0v[k] += r11*(KC0e1518*r11 + KC0e1618*r12 + KC0e1718*r13) + r12*(KC0e1519*r11 + KC0e1619*r12 + KC0e1719*r13) + r13*(KC0e1520*r11 + KC0e1620*r12)
            k += 1
            KC0v[k] += r21*(KC0e1518*r11 + KC0e1618*r12 + KC0e1718*r13) + r22*(KC0e1519*r11 + KC0e1619*r12 + KC0e1719*r13) + r23*(KC0e1520*r11 + KC0e1620*r12)
            k += 1
            KC0v[k] += r31*(KC0e1518*r11 + KC0e1618*r12 + KC0e1718*r13) + r32*(KC0e1519*r11 + KC0e1619*r12 + KC0e1719*r13) + r33*(KC0e1520*r11 + KC0e1620*r12)
            k += 1
            KC0v[k] += KC0e1723*r13**2 + r11*(KC0e1521*r11 + KC0e1621*r12) + r12*(KC0e1522*r11 + KC0e1622*r12)
            k += 1
            KC0v[k] += KC0e1723*r13*r23 + r21*(KC0e1521*r11 + KC0e1621*r12) + r22*(KC0e1522*r11 + KC0e1622*r12)
            k += 1
            KC0v[k] += KC0e1723*r13*r33 + r31*(KC0e1521*r11 + KC0e1621*r12) + r32*(KC0e1522*r11 + KC0e1622*r12)
            k += 1
            KC0v[k] += r11*(KC0e0015*r21 + KC0e0016*r22 + KC0e0017*r23) + r12*(KC0e0115*r21 + KC0e0116*r22 + KC0e0117*r23) + r13*(KC0e0215*r21 + KC0e0216*r22)
            k += 1
            KC0v[k] += r21*(KC0e0015*r21 + KC0e0016*r22 + KC0e0017*r23) + r22*(KC0e0115*r21 + KC0e0116*r22 + KC0e0117*r23) + r23*(KC0e0215*r21 + KC0e0216*r22)
            k += 1
            KC0v[k] += r31*(KC0e0015*r21 + KC0e0016*r22 + KC0e0017*r23) + r32*(KC0e0115*r21 + KC0e0116*r22 + KC0e0117*r23) + r33*(KC0e0215*r21 + KC0e0216*r22)
            k += 1
            KC0v[k] += KC0e0517*r13*r23 + r11*(KC0e0315*r21 + KC0e0316*r22) + r12*(KC0e0415*r21 + KC0e0416*r22)
            k += 1
            KC0v[k] += KC0e0517*r23**2 + r21*(KC0e0315*r21 + KC0e0316*r22) + r22*(KC0e0415*r21 + KC0e0416*r22)
            k += 1
            KC0v[k] += KC0e0517*r23*r33 + r31*(KC0e0315*r21 + KC0e0316*r22) + r32*(KC0e0415*r21 + KC0e0416*r22)
            k += 1
            KC0v[k] += r11*(KC0e0615*r21 + KC0e0616*r22 + KC0e0617*r23) + r12*(KC0e0715*r21 + KC0e0716*r22 + KC0e0717*r23) + r13*(KC0e0815*r21 + KC0e0816*r22)
            k += 1
            KC0v[k] += r21*(KC0e0615*r21 + KC0e0616*r22 + KC0e0617*r23) + r22*(KC0e0715*r21 + KC0e0716*r22 + KC0e0717*r23) + r23*(KC0e0815*r21 + KC0e0816*r22)
            k += 1
            KC0v[k] += r31*(KC0e0615*r21 + KC0e0616*r22 + KC0e0617*r23) + r32*(KC0e0715*r21 + KC0e0716*r22 + KC0e0717*r23) + r33*(KC0e0815*r21 + KC0e0816*r22)
            k += 1
            KC0v[k] += KC0e1117*r13*r23 + r11*(KC0e0915*r21 + KC0e0916*r22) + r12*(KC0e1015*r21 + KC0e1016*r22)
            k += 1
            KC0v[k] += KC0e1117*r23**2 + r21*(KC0e0915*r21 + KC0e0916*r22) + r22*(KC0e1015*r21 + KC0e1016*r22)
            k += 1
            KC0v[k] += KC0e1117*r23*r33 + r31*(KC0e0915*r21 + KC0e0916*r22) + r32*(KC0e1015*r21 + KC0e1016*r22)
            k += 1
            KC0v[k] += r11*(KC0e1215*r21 + KC0e1216*r22 + KC0e1217*r23) + r12*(KC0e1315*r21 + KC0e1316*r22 + KC0e1317*r23) + r13*(KC0e1415*r21 + KC0e1416*r22)
            k += 1
            KC0v[k] += r21*(KC0e1215*r21 + KC0e1216*r22 + KC0e1217*r23) + r22*(KC0e1315*r21 + KC0e1316*r22 + KC0e1317*r23) + r23*(KC0e1415*r21 + KC0e1416*r22)
            k += 1
            KC0v[k] += r31*(KC0e1215*r21 + KC0e1216*r22 + KC0e1217*r23) + r32*(KC0e1315*r21 + KC0e1316*r22 + KC0e1317*r23) + r33*(KC0e1415*r21 + KC0e1416*r22)
            k += 1
            KC0v[k] += KC0e1717*r13*r23 + r11*(KC0e1515*r21 + KC0e1516*r22) + r12*(KC0e1516*r21 + KC0e1616*r22)
            k += 1
            KC0v[k] += KC0e1717*r23**2 + r21*(KC0e1515*r21 + KC0e1516*r22) + r22*(KC0e1516*r21 + KC0e1616*r22)
            k += 1
            KC0v[k] += KC0e1717*r23*r33 + r31*(KC0e1515*r21 + KC0e1516*r22) + r32*(KC0e1516*r21 + KC0e1616*r22)
            k += 1
            KC0v[k] += r11*(KC0e1518*r21 + KC0e1618*r22 + KC0e1718*r23) + r12*(KC0e1519*r21 + KC0e1619*r22 + KC0e1719*r23) + r13*(KC0e1520*r21 + KC0e1620*r22)
            k += 1
            KC0v[k] += r21*(KC0e1518*r21 + KC0e1618*r22 + KC0e1718*r23) + r22*(KC0e1519*r21 + KC0e1619*r22 + KC0e1719*r23) + r23*(KC0e1520*r21 + KC0e1620*r22)
            k += 1
            KC0v[k] += r31*(KC0e1518*r21 + KC0e1618*r22 + KC0e1718*r23) + r32*(KC0e1519*r21 + KC0e1619*r22 + KC0e1719*r23) + r33*(KC0e1520*r21 + KC0e1620*r22)
            k += 1
            KC0v[k] += KC0e1723*r13*r23 + r11*(KC0e1521*r21 + KC0e1621*r22) + r12*(KC0e1522*r21 + KC0e1622*r22)
            k += 1
            KC0v[k] += KC0e1723*r23**2 + r21*(KC0e1521*r21 + KC0e1621*r22) + r22*(KC0e1522*r21 + KC0e1622*r22)
            k += 1
            KC0v[k] += KC0e1723*r23*r33 + r31*(KC0e1521*r21 + KC0e1621*r22) + r32*(KC0e1522*r21 + KC0e1622*r22)
            k += 1
            KC0v[k] += r11*(KC0e0015*r31 + KC0e0016*r32 + KC0e0017*r33) + r12*(KC0e0115*r31 + KC0e0116*r32 + KC0e0117*r33) + r13*(KC0e0215*r31 + KC0e0216*r32)
            k += 1
            KC0v[k] += r21*(KC0e0015*r31 + KC0e0016*r32 + KC0e0017*r33) + r22*(KC0e0115*r31 + KC0e0116*r32 + KC0e0117*r33) + r23*(KC0e0215*r31 + KC0e0216*r32)
            k += 1
            KC0v[k] += r31*(KC0e0015*r31 + KC0e0016*r32 + KC0e0017*r33) + r32*(KC0e0115*r31 + KC0e0116*r32 + KC0e0117*r33) + r33*(KC0e0215*r31 + KC0e0216*r32)
            k += 1
            KC0v[k] += KC0e0517*r13*r33 + r11*(KC0e0315*r31 + KC0e0316*r32) + r12*(KC0e0415*r31 + KC0e0416*r32)
            k += 1
            KC0v[k] += KC0e0517*r23*r33 + r21*(KC0e0315*r31 + KC0e0316*r32) + r22*(KC0e0415*r31 + KC0e0416*r32)
            k += 1
            KC0v[k] += KC0e0517*r33**2 + r31*(KC0e0315*r31 + KC0e0316*r32) + r32*(KC0e0415*r31 + KC0e0416*r32)
            k += 1
            KC0v[k] += r11*(KC0e0615*r31 + KC0e0616*r32 + KC0e0617*r33) + r12*(KC0e0715*r31 + KC0e0716*r32 + KC0e0717*r33) + r13*(KC0e0815*r31 + KC0e0816*r32)
            k += 1
            KC0v[k] += r21*(KC0e0615*r31 + KC0e0616*r32 + KC0e0617*r33) + r22*(KC0e0715*r31 + KC0e0716*r32 + KC0e0717*r33) + r23*(KC0e0815*r31 + KC0e0816*r32)
            k += 1
            KC0v[k] += r31*(KC0e0615*r31 + KC0e0616*r32 + KC0e0617*r33) + r32*(KC0e0715*r31 + KC0e0716*r32 + KC0e0717*r33) + r33*(KC0e0815*r31 + KC0e0816*r32)
            k += 1
            KC0v[k] += KC0e1117*r13*r33 + r11*(KC0e0915*r31 + KC0e0916*r32) + r12*(KC0e1015*r31 + KC0e1016*r32)
            k += 1
            KC0v[k] += KC0e1117*r23*r33 + r21*(KC0e0915*r31 + KC0e0916*r32) + r22*(KC0e1015*r31 + KC0e1016*r32)
            k += 1
            KC0v[k] += KC0e1117*r33**2 + r31*(KC0e0915*r31 + KC0e0916*r32) + r32*(KC0e1015*r31 + KC0e1016*r32)
            k += 1
            KC0v[k] += r11*(KC0e1215*r31 + KC0e1216*r32 + KC0e1217*r33) + r12*(KC0e1315*r31 + KC0e1316*r32 + KC0e1317*r33) + r13*(KC0e1415*r31 + KC0e1416*r32)
            k += 1
            KC0v[k] += r21*(KC0e1215*r31 + KC0e1216*r32 + KC0e1217*r33) + r22*(KC0e1315*r31 + KC0e1316*r32 + KC0e1317*r33) + r23*(KC0e1415*r31 + KC0e1416*r32)
            k += 1
            KC0v[k] += r31*(KC0e1215*r31 + KC0e1216*r32 + KC0e1217*r33) + r32*(KC0e1315*r31 + KC0e1316*r32 + KC0e1317*r33) + r33*(KC0e1415*r31 + KC0e1416*r32)
            k += 1
            KC0v[k] += KC0e1717*r13*r33 + r11*(KC0e1515*r31 + KC0e1516*r32) + r12*(KC0e1516*r31 + KC0e1616*r32)
            k += 1
            KC0v[k] += KC0e1717*r23*r33 + r21*(KC0e1515*r31 + KC0e1516*r32) + r22*(KC0e1516*r31 + KC0e1616*r32)
            k += 1
            KC0v[k] += KC0e1717*r33**2 + r31*(KC0e1515*r31 + KC0e1516*r32) + r32*(KC0e1516*r31 + KC0e1616*r32)
            k += 1
            KC0v[k] += r11*(KC0e1518*r31 + KC0e1618*r32 + KC0e1718*r33) + r12*(KC0e1519*r31 + KC0e1619*r32 + KC0e1719*r33) + r13*(KC0e1520*r31 + KC0e1620*r32)
            k += 1
            KC0v[k] += r21*(KC0e1518*r31 + KC0e1618*r32 + KC0e1718*r33) + r22*(KC0e1519*r31 + KC0e1619*r32 + KC0e1719*r33) + r23*(KC0e1520*r31 + KC0e1620*r32)
            k += 1
            KC0v[k] += r31*(KC0e1518*r31 + KC0e1618*r32 + KC0e1718*r33) + r32*(KC0e1519*r31 + KC0e1619*r32 + KC0e1719*r33) + r33*(KC0e1520*r31 + KC0e1620*r32)
            k += 1
            KC0v[k] += KC0e1723*r13*r33 + r11*(KC0e1521*r31 + KC0e1621*r32) + r12*(KC0e1522*r31 + KC0e1622*r32)
            k += 1
            KC0v[k] += KC0e1723*r23*r33 + r21*(KC0e1521*r31 + KC0e1621*r32) + r22*(KC0e1522*r31 + KC0e1622*r32)
            k += 1
            KC0v[k] += KC0e1723*r33**2 + r31*(KC0e1521*r31 + KC0e1621*r32) + r32*(KC0e1522*r31 + KC0e1622*r32)
            k += 1
            KC0v[k] += KC0e0220*r13**2 + r11*(KC0e0018*r11 + KC0e0019*r12) + r12*(KC0e0118*r11 + KC0e0119*r12)
            k += 1
            KC0v[k] += KC0e0220*r13*r23 + r21*(KC0e0018*r11 + KC0e0019*r12) + r22*(KC0e0118*r11 + KC0e0119*r12)
            k += 1
            KC0v[k] += KC0e0220*r13*r33 + r31*(KC0e0018*r11 + KC0e0019*r12) + r32*(KC0e0118*r11 + KC0e0119*r12)
            k += 1
            KC0v[k] += r11*(KC0e0318*r11 + KC0e0319*r12 + KC0e0320*r13) + r12*(KC0e0418*r11 + KC0e0419*r12 + KC0e0420*r13) + r13*(KC0e0518*r11 + KC0e0519*r12)
            k += 1
            KC0v[k] += r21*(KC0e0318*r11 + KC0e0319*r12 + KC0e0320*r13) + r22*(KC0e0418*r11 + KC0e0419*r12 + KC0e0420*r13) + r23*(KC0e0518*r11 + KC0e0519*r12)
            k += 1
            KC0v[k] += r31*(KC0e0318*r11 + KC0e0319*r12 + KC0e0320*r13) + r32*(KC0e0418*r11 + KC0e0419*r12 + KC0e0420*r13) + r33*(KC0e0518*r11 + KC0e0519*r12)
            k += 1
            KC0v[k] += KC0e0820*r13**2 + r11*(KC0e0618*r11 + KC0e0619*r12) + r12*(KC0e0718*r11 + KC0e0719*r12)
            k += 1
            KC0v[k] += KC0e0820*r13*r23 + r21*(KC0e0618*r11 + KC0e0619*r12) + r22*(KC0e0718*r11 + KC0e0719*r12)
            k += 1
            KC0v[k] += KC0e0820*r13*r33 + r31*(KC0e0618*r11 + KC0e0619*r12) + r32*(KC0e0718*r11 + KC0e0719*r12)
            k += 1
            KC0v[k] += r11*(KC0e0918*r11 + KC0e0919*r12 + KC0e0920*r13) + r12*(KC0e1018*r11 + KC0e1019*r12 + KC0e1020*r13) + r13*(KC0e1118*r11 + KC0e1119*r12)
            k += 1
            KC0v[k] += r21*(KC0e0918*r11 + KC0e0919*r12 + KC0e0920*r13) + r22*(KC0e1018*r11 + KC0e1019*r12 + KC0e1020*r13) + r23*(KC0e1118*r11 + KC0e1119*r12)
            k += 1
            KC0v[k] += r31*(KC0e0918*r11 + KC0e0919*r12 + KC0e0920*r13) + r32*(KC0e1018*r11 + KC0e1019*r12 + KC0e1020*r13) + r33*(KC0e1118*r11 + KC0e1119*r12)
            k += 1
            KC0v[k] += KC0e1420*r13**2 + r11*(KC0e1218*r11 + KC0e1219*r12) + r12*(KC0e1318*r11 + KC0e1319*r12)
            k += 1
            KC0v[k] += KC0e1420*r13*r23 + r21*(KC0e1218*r11 + KC0e1219*r12) + r22*(KC0e1318*r11 + KC0e1319*r12)
            k += 1
            KC0v[k] += KC0e1420*r13*r33 + r31*(KC0e1218*r11 + KC0e1219*r12) + r32*(KC0e1318*r11 + KC0e1319*r12)
            k += 1
            KC0v[k] += r11*(KC0e1518*r11 + KC0e1519*r12 + KC0e1520*r13) + r12*(KC0e1618*r11 + KC0e1619*r12 + KC0e1620*r13) + r13*(KC0e1718*r11 + KC0e1719*r12)
            k += 1
            KC0v[k] += r21*(KC0e1518*r11 + KC0e1519*r12 + KC0e1520*r13) + r22*(KC0e1618*r11 + KC0e1619*r12 + KC0e1620*r13) + r23*(KC0e1718*r11 + KC0e1719*r12)
            k += 1
            KC0v[k] += r31*(KC0e1518*r11 + KC0e1519*r12 + KC0e1520*r13) + r32*(KC0e1618*r11 + KC0e1619*r12 + KC0e1620*r13) + r33*(KC0e1718*r11 + KC0e1719*r12)
            k += 1
            KC0v[k] += KC0e2020*r13**2 + r11*(KC0e1818*r11 + KC0e1819*r12) + r12*(KC0e1819*r11 + KC0e1919*r12)
            k += 1
            KC0v[k] += KC0e2020*r13*r23 + r21*(KC0e1818*r11 + KC0e1819*r12) + r22*(KC0e1819*r11 + KC0e1919*r12)
            k += 1
            KC0v[k] += KC0e2020*r13*r33 + r31*(KC0e1818*r11 + KC0e1819*r12) + r32*(KC0e1819*r11 + KC0e1919*r12)
            k += 1
            KC0v[k] += r11*(KC0e1821*r11 + KC0e1921*r12 + KC0e2021*r13) + r12*(KC0e1822*r11 + KC0e1922*r12 + KC0e2022*r13) + r13*(KC0e1823*r11 + KC0e1923*r12)
            k += 1
            KC0v[k] += r21*(KC0e1821*r11 + KC0e1921*r12 + KC0e2021*r13) + r22*(KC0e1822*r11 + KC0e1922*r12 + KC0e2022*r13) + r23*(KC0e1823*r11 + KC0e1923*r12)
            k += 1
            KC0v[k] += r31*(KC0e1821*r11 + KC0e1921*r12 + KC0e2021*r13) + r32*(KC0e1822*r11 + KC0e1922*r12 + KC0e2022*r13) + r33*(KC0e1823*r11 + KC0e1923*r12)
            k += 1
            KC0v[k] += KC0e0220*r13*r23 + r11*(KC0e0018*r21 + KC0e0019*r22) + r12*(KC0e0118*r21 + KC0e0119*r22)
            k += 1
            KC0v[k] += KC0e0220*r23**2 + r21*(KC0e0018*r21 + KC0e0019*r22) + r22*(KC0e0118*r21 + KC0e0119*r22)
            k += 1
            KC0v[k] += KC0e0220*r23*r33 + r31*(KC0e0018*r21 + KC0e0019*r22) + r32*(KC0e0118*r21 + KC0e0119*r22)
            k += 1
            KC0v[k] += r11*(KC0e0318*r21 + KC0e0319*r22 + KC0e0320*r23) + r12*(KC0e0418*r21 + KC0e0419*r22 + KC0e0420*r23) + r13*(KC0e0518*r21 + KC0e0519*r22)
            k += 1
            KC0v[k] += r21*(KC0e0318*r21 + KC0e0319*r22 + KC0e0320*r23) + r22*(KC0e0418*r21 + KC0e0419*r22 + KC0e0420*r23) + r23*(KC0e0518*r21 + KC0e0519*r22)
            k += 1
            KC0v[k] += r31*(KC0e0318*r21 + KC0e0319*r22 + KC0e0320*r23) + r32*(KC0e0418*r21 + KC0e0419*r22 + KC0e0420*r23) + r33*(KC0e0518*r21 + KC0e0519*r22)
            k += 1
            KC0v[k] += KC0e0820*r13*r23 + r11*(KC0e0618*r21 + KC0e0619*r22) + r12*(KC0e0718*r21 + KC0e0719*r22)
            k += 1
            KC0v[k] += KC0e0820*r23**2 + r21*(KC0e0618*r21 + KC0e0619*r22) + r22*(KC0e0718*r21 + KC0e0719*r22)
            k += 1
            KC0v[k] += KC0e0820*r23*r33 + r31*(KC0e0618*r21 + KC0e0619*r22) + r32*(KC0e0718*r21 + KC0e0719*r22)
            k += 1
            KC0v[k] += r11*(KC0e0918*r21 + KC0e0919*r22 + KC0e0920*r23) + r12*(KC0e1018*r21 + KC0e1019*r22 + KC0e1020*r23) + r13*(KC0e1118*r21 + KC0e1119*r22)
            k += 1
            KC0v[k] += r21*(KC0e0918*r21 + KC0e0919*r22 + KC0e0920*r23) + r22*(KC0e1018*r21 + KC0e1019*r22 + KC0e1020*r23) + r23*(KC0e1118*r21 + KC0e1119*r22)
            k += 1
            KC0v[k] += r31*(KC0e0918*r21 + KC0e0919*r22 + KC0e0920*r23) + r32*(KC0e1018*r21 + KC0e1019*r22 + KC0e1020*r23) + r33*(KC0e1118*r21 + KC0e1119*r22)
            k += 1
            KC0v[k] += KC0e1420*r13*r23 + r11*(KC0e1218*r21 + KC0e1219*r22) + r12*(KC0e1318*r21 + KC0e1319*r22)
            k += 1
            KC0v[k] += KC0e1420*r23**2 + r21*(KC0e1218*r21 + KC0e1219*r22) + r22*(KC0e1318*r21 + KC0e1319*r22)
            k += 1
            KC0v[k] += KC0e1420*r23*r33 + r31*(KC0e1218*r21 + KC0e1219*r22) + r32*(KC0e1318*r21 + KC0e1319*r22)
            k += 1
            KC0v[k] += r11*(KC0e1518*r21 + KC0e1519*r22 + KC0e1520*r23) + r12*(KC0e1618*r21 + KC0e1619*r22 + KC0e1620*r23) + r13*(KC0e1718*r21 + KC0e1719*r22)
            k += 1
            KC0v[k] += r21*(KC0e1518*r21 + KC0e1519*r22 + KC0e1520*r23) + r22*(KC0e1618*r21 + KC0e1619*r22 + KC0e1620*r23) + r23*(KC0e1718*r21 + KC0e1719*r22)
            k += 1
            KC0v[k] += r31*(KC0e1518*r21 + KC0e1519*r22 + KC0e1520*r23) + r32*(KC0e1618*r21 + KC0e1619*r22 + KC0e1620*r23) + r33*(KC0e1718*r21 + KC0e1719*r22)
            k += 1
            KC0v[k] += KC0e2020*r13*r23 + r11*(KC0e1818*r21 + KC0e1819*r22) + r12*(KC0e1819*r21 + KC0e1919*r22)
            k += 1
            KC0v[k] += KC0e2020*r23**2 + r21*(KC0e1818*r21 + KC0e1819*r22) + r22*(KC0e1819*r21 + KC0e1919*r22)
            k += 1
            KC0v[k] += KC0e2020*r23*r33 + r31*(KC0e1818*r21 + KC0e1819*r22) + r32*(KC0e1819*r21 + KC0e1919*r22)
            k += 1
            KC0v[k] += r11*(KC0e1821*r21 + KC0e1921*r22 + KC0e2021*r23) + r12*(KC0e1822*r21 + KC0e1922*r22 + KC0e2022*r23) + r13*(KC0e1823*r21 + KC0e1923*r22)
            k += 1
            KC0v[k] += r21*(KC0e1821*r21 + KC0e1921*r22 + KC0e2021*r23) + r22*(KC0e1822*r21 + KC0e1922*r22 + KC0e2022*r23) + r23*(KC0e1823*r21 + KC0e1923*r22)
            k += 1
            KC0v[k] += r31*(KC0e1821*r21 + KC0e1921*r22 + KC0e2021*r23) + r32*(KC0e1822*r21 + KC0e1922*r22 + KC0e2022*r23) + r33*(KC0e1823*r21 + KC0e1923*r22)
            k += 1
            KC0v[k] += KC0e0220*r13*r33 + r11*(KC0e0018*r31 + KC0e0019*r32) + r12*(KC0e0118*r31 + KC0e0119*r32)
            k += 1
            KC0v[k] += KC0e0220*r23*r33 + r21*(KC0e0018*r31 + KC0e0019*r32) + r22*(KC0e0118*r31 + KC0e0119*r32)
            k += 1
            KC0v[k] += KC0e0220*r33**2 + r31*(KC0e0018*r31 + KC0e0019*r32) + r32*(KC0e0118*r31 + KC0e0119*r32)
            k += 1
            KC0v[k] += r11*(KC0e0318*r31 + KC0e0319*r32 + KC0e0320*r33) + r12*(KC0e0418*r31 + KC0e0419*r32 + KC0e0420*r33) + r13*(KC0e0518*r31 + KC0e0519*r32)
            k += 1
            KC0v[k] += r21*(KC0e0318*r31 + KC0e0319*r32 + KC0e0320*r33) + r22*(KC0e0418*r31 + KC0e0419*r32 + KC0e0420*r33) + r23*(KC0e0518*r31 + KC0e0519*r32)
            k += 1
            KC0v[k] += r31*(KC0e0318*r31 + KC0e0319*r32 + KC0e0320*r33) + r32*(KC0e0418*r31 + KC0e0419*r32 + KC0e0420*r33) + r33*(KC0e0518*r31 + KC0e0519*r32)
            k += 1
            KC0v[k] += KC0e0820*r13*r33 + r11*(KC0e0618*r31 + KC0e0619*r32) + r12*(KC0e0718*r31 + KC0e0719*r32)
            k += 1
            KC0v[k] += KC0e0820*r23*r33 + r21*(KC0e0618*r31 + KC0e0619*r32) + r22*(KC0e0718*r31 + KC0e0719*r32)
            k += 1
            KC0v[k] += KC0e0820*r33**2 + r31*(KC0e0618*r31 + KC0e0619*r32) + r32*(KC0e0718*r31 + KC0e0719*r32)
            k += 1
            KC0v[k] += r11*(KC0e0918*r31 + KC0e0919*r32 + KC0e0920*r33) + r12*(KC0e1018*r31 + KC0e1019*r32 + KC0e1020*r33) + r13*(KC0e1118*r31 + KC0e1119*r32)
            k += 1
            KC0v[k] += r21*(KC0e0918*r31 + KC0e0919*r32 + KC0e0920*r33) + r22*(KC0e1018*r31 + KC0e1019*r32 + KC0e1020*r33) + r23*(KC0e1118*r31 + KC0e1119*r32)
            k += 1
            KC0v[k] += r31*(KC0e0918*r31 + KC0e0919*r32 + KC0e0920*r33) + r32*(KC0e1018*r31 + KC0e1019*r32 + KC0e1020*r33) + r33*(KC0e1118*r31 + KC0e1119*r32)
            k += 1
            KC0v[k] += KC0e1420*r13*r33 + r11*(KC0e1218*r31 + KC0e1219*r32) + r12*(KC0e1318*r31 + KC0e1319*r32)
            k += 1
            KC0v[k] += KC0e1420*r23*r33 + r21*(KC0e1218*r31 + KC0e1219*r32) + r22*(KC0e1318*r31 + KC0e1319*r32)
            k += 1
            KC0v[k] += KC0e1420*r33**2 + r31*(KC0e1218*r31 + KC0e1219*r32) + r32*(KC0e1318*r31 + KC0e1319*r32)
            k += 1
            KC0v[k] += r11*(KC0e1518*r31 + KC0e1519*r32 + KC0e1520*r33) + r12*(KC0e1618*r31 + KC0e1619*r32 + KC0e1620*r33) + r13*(KC0e1718*r31 + KC0e1719*r32)
            k += 1
            KC0v[k] += r21*(KC0e1518*r31 + KC0e1519*r32 + KC0e1520*r33) + r22*(KC0e1618*r31 + KC0e1619*r32 + KC0e1620*r33) + r23*(KC0e1718*r31 + KC0e1719*r32)
            k += 1
            KC0v[k] += r31*(KC0e1518*r31 + KC0e1519*r32 + KC0e1520*r33) + r32*(KC0e1618*r31 + KC0e1619*r32 + KC0e1620*r33) + r33*(KC0e1718*r31 + KC0e1719*r32)
            k += 1
            KC0v[k] += KC0e2020*r13*r33 + r11*(KC0e1818*r31 + KC0e1819*r32) + r12*(KC0e1819*r31 + KC0e1919*r32)
            k += 1
            KC0v[k] += KC0e2020*r23*r33 + r21*(KC0e1818*r31 + KC0e1819*r32) + r22*(KC0e1819*r31 + KC0e1919*r32)
            k += 1
            KC0v[k] += KC0e2020*r33**2 + r31*(KC0e1818*r31 + KC0e1819*r32) + r32*(KC0e1819*r31 + KC0e1919*r32)
            k += 1
            KC0v[k] += r11*(KC0e1821*r31 + KC0e1921*r32 + KC0e2021*r33) + r12*(KC0e1822*r31 + KC0e1922*r32 + KC0e2022*r33) + r13*(KC0e1823*r31 + KC0e1923*r32)
            k += 1
            KC0v[k] += r21*(KC0e1821*r31 + KC0e1921*r32 + KC0e2021*r33) + r22*(KC0e1822*r31 + KC0e1922*r32 + KC0e2022*r33) + r23*(KC0e1823*r31 + KC0e1923*r32)
            k += 1
            KC0v[k] += r31*(KC0e1821*r31 + KC0e1921*r32 + KC0e2021*r33) + r32*(KC0e1822*r31 + KC0e1922*r32 + KC0e2022*r33) + r33*(KC0e1823*r31 + KC0e1923*r32)
            k += 1
            KC0v[k] += r11*(KC0e0021*r11 + KC0e0022*r12 + KC0e0023*r13) + r12*(KC0e0121*r11 + KC0e0122*r12 + KC0e0123*r13) + r13*(KC0e0221*r11 + KC0e0222*r12)
            k += 1
            KC0v[k] += r21*(KC0e0021*r11 + KC0e0022*r12 + KC0e0023*r13) + r22*(KC0e0121*r11 + KC0e0122*r12 + KC0e0123*r13) + r23*(KC0e0221*r11 + KC0e0222*r12)
            k += 1
            KC0v[k] += r31*(KC0e0021*r11 + KC0e0022*r12 + KC0e0023*r13) + r32*(KC0e0121*r11 + KC0e0122*r12 + KC0e0123*r13) + r33*(KC0e0221*r11 + KC0e0222*r12)
            k += 1
            KC0v[k] += KC0e0523*r13**2 + r11*(KC0e0321*r11 + KC0e0322*r12) + r12*(KC0e0421*r11 + KC0e0422*r12)
            k += 1
            KC0v[k] += KC0e0523*r13*r23 + r21*(KC0e0321*r11 + KC0e0322*r12) + r22*(KC0e0421*r11 + KC0e0422*r12)
            k += 1
            KC0v[k] += KC0e0523*r13*r33 + r31*(KC0e0321*r11 + KC0e0322*r12) + r32*(KC0e0421*r11 + KC0e0422*r12)
            k += 1
            KC0v[k] += r11*(KC0e0621*r11 + KC0e0622*r12 + KC0e0623*r13) + r12*(KC0e0721*r11 + KC0e0722*r12 + KC0e0723*r13) + r13*(KC0e0821*r11 + KC0e0822*r12)
            k += 1
            KC0v[k] += r21*(KC0e0621*r11 + KC0e0622*r12 + KC0e0623*r13) + r22*(KC0e0721*r11 + KC0e0722*r12 + KC0e0723*r13) + r23*(KC0e0821*r11 + KC0e0822*r12)
            k += 1
            KC0v[k] += r31*(KC0e0621*r11 + KC0e0622*r12 + KC0e0623*r13) + r32*(KC0e0721*r11 + KC0e0722*r12 + KC0e0723*r13) + r33*(KC0e0821*r11 + KC0e0822*r12)
            k += 1
            KC0v[k] += KC0e1123*r13**2 + r11*(KC0e0921*r11 + KC0e0922*r12) + r12*(KC0e1021*r11 + KC0e1022*r12)
            k += 1
            KC0v[k] += KC0e1123*r13*r23 + r21*(KC0e0921*r11 + KC0e0922*r12) + r22*(KC0e1021*r11 + KC0e1022*r12)
            k += 1
            KC0v[k] += KC0e1123*r13*r33 + r31*(KC0e0921*r11 + KC0e0922*r12) + r32*(KC0e1021*r11 + KC0e1022*r12)
            k += 1
            KC0v[k] += r11*(KC0e1221*r11 + KC0e1222*r12 + KC0e1223*r13) + r12*(KC0e1321*r11 + KC0e1322*r12 + KC0e1323*r13) + r13*(KC0e1421*r11 + KC0e1422*r12)
            k += 1
            KC0v[k] += r21*(KC0e1221*r11 + KC0e1222*r12 + KC0e1223*r13) + r22*(KC0e1321*r11 + KC0e1322*r12 + KC0e1323*r13) + r23*(KC0e1421*r11 + KC0e1422*r12)
            k += 1
            KC0v[k] += r31*(KC0e1221*r11 + KC0e1222*r12 + KC0e1223*r13) + r32*(KC0e1321*r11 + KC0e1322*r12 + KC0e1323*r13) + r33*(KC0e1421*r11 + KC0e1422*r12)
            k += 1
            KC0v[k] += KC0e1723*r13**2 + r11*(KC0e1521*r11 + KC0e1522*r12) + r12*(KC0e1621*r11 + KC0e1622*r12)
            k += 1
            KC0v[k] += KC0e1723*r13*r23 + r21*(KC0e1521*r11 + KC0e1522*r12) + r22*(KC0e1621*r11 + KC0e1622*r12)
            k += 1
            KC0v[k] += KC0e1723*r13*r33 + r31*(KC0e1521*r11 + KC0e1522*r12) + r32*(KC0e1621*r11 + KC0e1622*r12)
            k += 1
            KC0v[k] += r11*(KC0e1821*r11 + KC0e1822*r12 + KC0e1823*r13) + r12*(KC0e1921*r11 + KC0e1922*r12 + KC0e1923*r13) + r13*(KC0e2021*r11 + KC0e2022*r12)
            k += 1
            KC0v[k] += r21*(KC0e1821*r11 + KC0e1822*r12 + KC0e1823*r13) + r22*(KC0e1921*r11 + KC0e1922*r12 + KC0e1923*r13) + r23*(KC0e2021*r11 + KC0e2022*r12)
            k += 1
            KC0v[k] += r31*(KC0e1821*r11 + KC0e1822*r12 + KC0e1823*r13) + r32*(KC0e1921*r11 + KC0e1922*r12 + KC0e1923*r13) + r33*(KC0e2021*r11 + KC0e2022*r12)
            k += 1
            KC0v[k] += KC0e2323*r13**2 + r11*(KC0e2121*r11 + KC0e2122*r12) + r12*(KC0e2122*r11 + KC0e2222*r12)
            k += 1
            KC0v[k] += KC0e2323*r13*r23 + r21*(KC0e2121*r11 + KC0e2122*r12) + r22*(KC0e2122*r11 + KC0e2222*r12)
            k += 1
            KC0v[k] += KC0e2323*r13*r33 + r31*(KC0e2121*r11 + KC0e2122*r12) + r32*(KC0e2122*r11 + KC0e2222*r12)
            k += 1
            KC0v[k] += r11*(KC0e0021*r21 + KC0e0022*r22 + KC0e0023*r23) + r12*(KC0e0121*r21 + KC0e0122*r22 + KC0e0123*r23) + r13*(KC0e0221*r21 + KC0e0222*r22)
            k += 1
            KC0v[k] += r21*(KC0e0021*r21 + KC0e0022*r22 + KC0e0023*r23) + r22*(KC0e0121*r21 + KC0e0122*r22 + KC0e0123*r23) + r23*(KC0e0221*r21 + KC0e0222*r22)
            k += 1
            KC0v[k] += r31*(KC0e0021*r21 + KC0e0022*r22 + KC0e0023*r23) + r32*(KC0e0121*r21 + KC0e0122*r22 + KC0e0123*r23) + r33*(KC0e0221*r21 + KC0e0222*r22)
            k += 1
            KC0v[k] += KC0e0523*r13*r23 + r11*(KC0e0321*r21 + KC0e0322*r22) + r12*(KC0e0421*r21 + KC0e0422*r22)
            k += 1
            KC0v[k] += KC0e0523*r23**2 + r21*(KC0e0321*r21 + KC0e0322*r22) + r22*(KC0e0421*r21 + KC0e0422*r22)
            k += 1
            KC0v[k] += KC0e0523*r23*r33 + r31*(KC0e0321*r21 + KC0e0322*r22) + r32*(KC0e0421*r21 + KC0e0422*r22)
            k += 1
            KC0v[k] += r11*(KC0e0621*r21 + KC0e0622*r22 + KC0e0623*r23) + r12*(KC0e0721*r21 + KC0e0722*r22 + KC0e0723*r23) + r13*(KC0e0821*r21 + KC0e0822*r22)
            k += 1
            KC0v[k] += r21*(KC0e0621*r21 + KC0e0622*r22 + KC0e0623*r23) + r22*(KC0e0721*r21 + KC0e0722*r22 + KC0e0723*r23) + r23*(KC0e0821*r21 + KC0e0822*r22)
            k += 1
            KC0v[k] += r31*(KC0e0621*r21 + KC0e0622*r22 + KC0e0623*r23) + r32*(KC0e0721*r21 + KC0e0722*r22 + KC0e0723*r23) + r33*(KC0e0821*r21 + KC0e0822*r22)
            k += 1
            KC0v[k] += KC0e1123*r13*r23 + r11*(KC0e0921*r21 + KC0e0922*r22) + r12*(KC0e1021*r21 + KC0e1022*r22)
            k += 1
            KC0v[k] += KC0e1123*r23**2 + r21*(KC0e0921*r21 + KC0e0922*r22) + r22*(KC0e1021*r21 + KC0e1022*r22)
            k += 1
            KC0v[k] += KC0e1123*r23*r33 + r31*(KC0e0921*r21 + KC0e0922*r22) + r32*(KC0e1021*r21 + KC0e1022*r22)
            k += 1
            KC0v[k] += r11*(KC0e1221*r21 + KC0e1222*r22 + KC0e1223*r23) + r12*(KC0e1321*r21 + KC0e1322*r22 + KC0e1323*r23) + r13*(KC0e1421*r21 + KC0e1422*r22)
            k += 1
            KC0v[k] += r21*(KC0e1221*r21 + KC0e1222*r22 + KC0e1223*r23) + r22*(KC0e1321*r21 + KC0e1322*r22 + KC0e1323*r23) + r23*(KC0e1421*r21 + KC0e1422*r22)
            k += 1
            KC0v[k] += r31*(KC0e1221*r21 + KC0e1222*r22 + KC0e1223*r23) + r32*(KC0e1321*r21 + KC0e1322*r22 + KC0e1323*r23) + r33*(KC0e1421*r21 + KC0e1422*r22)
            k += 1
            KC0v[k] += KC0e1723*r13*r23 + r11*(KC0e1521*r21 + KC0e1522*r22) + r12*(KC0e1621*r21 + KC0e1622*r22)
            k += 1
            KC0v[k] += KC0e1723*r23**2 + r21*(KC0e1521*r21 + KC0e1522*r22) + r22*(KC0e1621*r21 + KC0e1622*r22)
            k += 1
            KC0v[k] += KC0e1723*r23*r33 + r31*(KC0e1521*r21 + KC0e1522*r22) + r32*(KC0e1621*r21 + KC0e1622*r22)
            k += 1
            KC0v[k] += r11*(KC0e1821*r21 + KC0e1822*r22 + KC0e1823*r23) + r12*(KC0e1921*r21 + KC0e1922*r22 + KC0e1923*r23) + r13*(KC0e2021*r21 + KC0e2022*r22)
            k += 1
            KC0v[k] += r21*(KC0e1821*r21 + KC0e1822*r22 + KC0e1823*r23) + r22*(KC0e1921*r21 + KC0e1922*r22 + KC0e1923*r23) + r23*(KC0e2021*r21 + KC0e2022*r22)
            k += 1
            KC0v[k] += r31*(KC0e1821*r21 + KC0e1822*r22 + KC0e1823*r23) + r32*(KC0e1921*r21 + KC0e1922*r22 + KC0e1923*r23) + r33*(KC0e2021*r21 + KC0e2022*r22)
            k += 1
            KC0v[k] += KC0e2323*r13*r23 + r11*(KC0e2121*r21 + KC0e2122*r22) + r12*(KC0e2122*r21 + KC0e2222*r22)
            k += 1
            KC0v[k] += KC0e2323*r23**2 + r21*(KC0e2121*r21 + KC0e2122*r22) + r22*(KC0e2122*r21 + KC0e2222*r22)
            k += 1
            KC0v[k] += KC0e2323*r23*r33 + r31*(KC0e2121*r21 + KC0e2122*r22) + r32*(KC0e2122*r21 + KC0e2222*r22)
            k += 1
            KC0v[k] += r11*(KC0e0021*r31 + KC0e0022*r32 + KC0e0023*r33) + r12*(KC0e0121*r31 + KC0e0122*r32 + KC0e0123*r33) + r13*(KC0e0221*r31 + KC0e0222*r32)
            k += 1
            KC0v[k] += r21*(KC0e0021*r31 + KC0e0022*r32 + KC0e0023*r33) + r22*(KC0e0121*r31 + KC0e0122*r32 + KC0e0123*r33) + r23*(KC0e0221*r31 + KC0e0222*r32)
            k += 1
            KC0v[k] += r31*(KC0e0021*r31 + KC0e0022*r32 + KC0e0023*r33) + r32*(KC0e0121*r31 + KC0e0122*r32 + KC0e0123*r33) + r33*(KC0e0221*r31 + KC0e0222*r32)
            k += 1
            KC0v[k] += KC0e0523*r13*r33 + r11*(KC0e0321*r31 + KC0e0322*r32) + r12*(KC0e0421*r31 + KC0e0422*r32)
            k += 1
            KC0v[k] += KC0e0523*r23*r33 + r21*(KC0e0321*r31 + KC0e0322*r32) + r22*(KC0e0421*r31 + KC0e0422*r32)
            k += 1
            KC0v[k] += KC0e0523*r33**2 + r31*(KC0e0321*r31 + KC0e0322*r32) + r32*(KC0e0421*r31 + KC0e0422*r32)
            k += 1
            KC0v[k] += r11*(KC0e0621*r31 + KC0e0622*r32 + KC0e0623*r33) + r12*(KC0e0721*r31 + KC0e0722*r32 + KC0e0723*r33) + r13*(KC0e0821*r31 + KC0e0822*r32)
            k += 1
            KC0v[k] += r21*(KC0e0621*r31 + KC0e0622*r32 + KC0e0623*r33) + r22*(KC0e0721*r31 + KC0e0722*r32 + KC0e0723*r33) + r23*(KC0e0821*r31 + KC0e0822*r32)
            k += 1
            KC0v[k] += r31*(KC0e0621*r31 + KC0e0622*r32 + KC0e0623*r33) + r32*(KC0e0721*r31 + KC0e0722*r32 + KC0e0723*r33) + r33*(KC0e0821*r31 + KC0e0822*r32)
            k += 1
            KC0v[k] += KC0e1123*r13*r33 + r11*(KC0e0921*r31 + KC0e0922*r32) + r12*(KC0e1021*r31 + KC0e1022*r32)
            k += 1
            KC0v[k] += KC0e1123*r23*r33 + r21*(KC0e0921*r31 + KC0e0922*r32) + r22*(KC0e1021*r31 + KC0e1022*r32)
            k += 1
            KC0v[k] += KC0e1123*r33**2 + r31*(KC0e0921*r31 + KC0e0922*r32) + r32*(KC0e1021*r31 + KC0e1022*r32)
            k += 1
            KC0v[k] += r11*(KC0e1221*r31 + KC0e1222*r32 + KC0e1223*r33) + r12*(KC0e1321*r31 + KC0e1322*r32 + KC0e1323*r33) + r13*(KC0e1421*r31 + KC0e1422*r32)
            k += 1
            KC0v[k] += r21*(KC0e1221*r31 + KC0e1222*r32 + KC0e1223*r33) + r22*(KC0e1321*r31 + KC0e1322*r32 + KC0e1323*r33) + r23*(KC0e1421*r31 + KC0e1422*r32)
            k += 1
            KC0v[k] += r31*(KC0e1221*r31 + KC0e1222*r32 + KC0e1223*r33) + r32*(KC0e1321*r31 + KC0e1322*r32 + KC0e1323*r33) + r33*(KC0e1421*r31 + KC0e1422*r32)
            k += 1
            KC0v[k] += KC0e1723*r13*r33 + r11*(KC0e1521*r31 + KC0e1522*r32) + r12*(KC0e1621*r31 + KC0e1622*r32)
            k += 1
            KC0v[k] += KC0e1723*r23*r33 + r21*(KC0e1521*r31 + KC0e1522*r32) + r22*(KC0e1621*r31 + KC0e1622*r32)
            k += 1
            KC0v[k] += KC0e1723*r33**2 + r31*(KC0e1521*r31 + KC0e1522*r32) + r32*(KC0e1621*r31 + KC0e1622*r32)
            k += 1
            KC0v[k] += r11*(KC0e1821*r31 + KC0e1822*r32 + KC0e1823*r33) + r12*(KC0e1921*r31 + KC0e1922*r32 + KC0e1923*r33) + r13*(KC0e2021*r31 + KC0e2022*r32)
            k += 1
            KC0v[k] += r21*(KC0e1821*r31 + KC0e1822*r32 + KC0e1823*r33) + r22*(KC0e1921*r31 + KC0e1922*r32 + KC0e1923*r33) + r23*(KC0e2021*r31 + KC0e2022*r32)
            k += 1
            KC0v[k] += r31*(KC0e1821*r31 + KC0e1822*r32 + KC0e1823*r33) + r32*(KC0e1921*r31 + KC0e1922*r32 + KC0e1923*r33) + r33*(KC0e2021*r31 + KC0e2022*r32)
            k += 1
            KC0v[k] += KC0e2323*r13*r33 + r11*(KC0e2121*r31 + KC0e2122*r32) + r12*(KC0e2122*r31 + KC0e2222*r32)
            k += 1
            KC0v[k] += KC0e2323*r23*r33 + r21*(KC0e2121*r31 + KC0e2122*r32) + r22*(KC0e2122*r31 + KC0e2222*r32)
            k += 1
            KC0v[k] += KC0e2323*r33**2 + r31*(KC0e2121*r31 + KC0e2122*r32) + r32*(KC0e2122*r31 + KC0e2222*r32)

            # NOTE rigid offsets of a warped element: J^T KC0e J - KC0e,
            #      rotated to global coordinates and added, see the Notes
            warped = _warp_z(&self.probe.xe[0], self.warp_offsets, zw)
            if warped:
                for iw in range(576):
                    Kd[iw] = 0.
                Kd[0] = KC0e0000
                Kd[1] = KC0e0001
                Kd[24] = KC0e0001
                Kd[3] = KC0e0003
                Kd[72] = KC0e0003
                Kd[4] = KC0e0004
                Kd[96] = KC0e0004
                Kd[5] = KC0e0005
                Kd[120] = KC0e0005
                Kd[6] = KC0e0006
                Kd[144] = KC0e0006
                Kd[7] = KC0e0007
                Kd[168] = KC0e0007
                Kd[9] = KC0e0009
                Kd[216] = KC0e0009
                Kd[10] = KC0e0010
                Kd[240] = KC0e0010
                Kd[11] = KC0e0011
                Kd[264] = KC0e0011
                Kd[12] = KC0e0012
                Kd[288] = KC0e0012
                Kd[13] = KC0e0013
                Kd[312] = KC0e0013
                Kd[15] = KC0e0015
                Kd[360] = KC0e0015
                Kd[16] = KC0e0016
                Kd[384] = KC0e0016
                Kd[17] = KC0e0017
                Kd[408] = KC0e0017
                Kd[18] = KC0e0018
                Kd[432] = KC0e0018
                Kd[19] = KC0e0019
                Kd[456] = KC0e0019
                Kd[21] = KC0e0021
                Kd[504] = KC0e0021
                Kd[22] = KC0e0022
                Kd[528] = KC0e0022
                Kd[23] = KC0e0023
                Kd[552] = KC0e0023
                Kd[25] = KC0e0101
                Kd[27] = KC0e0103
                Kd[73] = KC0e0103
                Kd[28] = KC0e0104
                Kd[97] = KC0e0104
                Kd[29] = KC0e0105
                Kd[121] = KC0e0105
                Kd[30] = KC0e0106
                Kd[145] = KC0e0106
                Kd[31] = KC0e0107
                Kd[169] = KC0e0107
                Kd[33] = KC0e0109
                Kd[217] = KC0e0109
                Kd[34] = KC0e0110
                Kd[241] = KC0e0110
                Kd[35] = KC0e0111
                Kd[265] = KC0e0111
                Kd[36] = KC0e0112
                Kd[289] = KC0e0112
                Kd[37] = KC0e0113
                Kd[313] = KC0e0113
                Kd[39] = KC0e0115
                Kd[361] = KC0e0115
                Kd[40] = KC0e0116
                Kd[385] = KC0e0116
                Kd[41] = KC0e0117
                Kd[409] = KC0e0117
                Kd[42] = KC0e0118
                Kd[433] = KC0e0118
                Kd[43] = KC0e0119
                Kd[457] = KC0e0119
                Kd[45] = KC0e0121
                Kd[505] = KC0e0121
                Kd[46] = KC0e0122
                Kd[529] = KC0e0122
                Kd[47] = KC0e0123
                Kd[553] = KC0e0123
                Kd[50] = KC0e0202
                Kd[51] = KC0e0203
                Kd[74] = KC0e0203
                Kd[52] = KC0e0204
                Kd[98] = KC0e0204
                Kd[56] = KC0e0208
                Kd[194] = KC0e0208
                Kd[57] = KC0e0209
                Kd[218] = KC0e0209
                Kd[58] = KC0e0210
                Kd[242] = KC0e0210
                Kd[62] = KC0e0214
                Kd[338] = KC0e0214
                Kd[63] = KC0e0215
                Kd[362] = KC0e0215
                Kd[64] = KC0e0216
                Kd[386] = KC0e0216
                Kd[68] = KC0e0220
                Kd[482] = KC0e0220
                Kd[69] = KC0e0221
                Kd[506] = KC0e0221
                Kd[70] = KC0e0222
                Kd[530] = KC0e0222
                Kd[75] = KC0e0303
                Kd[76] = KC0e0304
                Kd[99] = KC0e0304
                Kd[78] = KC0e0306
                Kd[147] = KC0e0306
                Kd[79] = KC0e0307
                Kd[171] = KC0e0307
                Kd[80] = KC0e0308
                Kd[195] = KC0e0308
                Kd[81] = KC0e0309
                Kd[219] = KC0e0309
                Kd[82] = KC0e0310
                Kd[243] = KC0e0310
                Kd[84] = KC0e0312
                Kd[291] = KC0e0312
                Kd[85] = KC0e0313
                Kd[315] = KC0e0313
                Kd[86] = KC0e0314
                Kd[339] = KC0e0314
                Kd[87] = KC0e0315
                Kd[363] = KC0e0315
                Kd[88] = KC0e0316
                Kd[387] = KC0e0316
                Kd[90] = KC0e0318
                Kd[435] = KC0e0318
                Kd[91] = KC0e0319
                Kd[459] = KC0e0319
                Kd[92] = KC0e0320
                Kd[483] = KC0e0320
                Kd[93] = KC0e0321
                Kd[507] = KC0e0321
                Kd[94] = KC0e0322
                Kd[531] = KC0e0322
                Kd[100] = KC0e0404
                Kd[102] = KC0e0406
                Kd[148] = KC0e0406
                Kd[103] = KC0e0407
                Kd[172] = KC0e0407
                Kd[104] = KC0e0408
                Kd[196] = KC0e0408
                Kd[105] = KC0e0409
                Kd[220] = KC0e0409
                Kd[106] = KC0e0410
                Kd[244] = KC0e0410
                Kd[108] = KC0e0412
                Kd[292] = KC0e0412
                Kd[109] = KC0e0413
                Kd[316] = KC0e0413
                Kd[110] = KC0e0414
                Kd[340] = KC0e0414
                Kd[111] = KC0e0415
                Kd[364] = KC0e0415
                Kd[112] = KC0e0416
                Kd[388] = KC0e0416
                Kd[114] = KC0e0418
                Kd[436] = KC0e0418
                Kd[115] = KC0e0419
                Kd[460] = KC0e0419
                Kd[116] = KC0e0420
                Kd[484] = KC0e0420
                Kd[117] = KC0e0421
                Kd[508] = KC0e0421
                Kd[118] = KC0e0422
                Kd[532] = KC0e0422
                Kd[125] = KC0e0505
                Kd[126] = KC0e0506
                Kd[149] = KC0e0506
                Kd[127] = KC0e0507
                Kd[173] = KC0e0507
                Kd[131] = KC0e0511
                Kd[269] = KC0e0511
                Kd[132] = KC0e0512
                Kd[293] = KC0e0512
                Kd[133] = KC0e0513
                Kd[317] = KC0e0513
                Kd[137] = KC0e0517
                Kd[413] = KC0e0517
                Kd[138] = KC0e0518
                Kd[437] = KC0e0518
                Kd[139] = KC0e0519
                Kd[461] = KC0e0519
                Kd[143] = KC0e0523
                Kd[557] = KC0e0523
                Kd[150] = KC0e0606
                Kd[151] = KC0e0607
                Kd[174] = KC0e0607
                Kd[153] = KC0e0609
                Kd[222] = KC0e0609
                Kd[154] = KC0e0610
                Kd[246] = KC0e0610
                Kd[155] = KC0e0611
                Kd[270] = KC0e0611
                Kd[156] = KC0e0612
                Kd[294] = KC0e0612
                Kd[157] = KC0e0613
                Kd[318] = KC0e0613
                Kd[159] = KC0e0615
                Kd[366] = KC0e0615
                Kd[160] = KC0e0616
                Kd[390] = KC0e0616
                Kd[161] = KC0e0617
                Kd[414] = KC0e0617
                Kd[162] = KC0e0618
                Kd[438] = KC0e0618
                Kd[163] = KC0e0619
                Kd[462] = KC0e0619
                Kd[165] = KC0e0621
                Kd[510] = KC0e0621
                Kd[166] = KC0e0622
                Kd[534] = KC0e0622
                Kd[167] = KC0e0623
                Kd[558] = KC0e0623
                Kd[175] = KC0e0707
                Kd[177] = KC0e0709
                Kd[223] = KC0e0709
                Kd[178] = KC0e0710
                Kd[247] = KC0e0710
                Kd[179] = KC0e0711
                Kd[271] = KC0e0711
                Kd[180] = KC0e0712
                Kd[295] = KC0e0712
                Kd[181] = KC0e0713
                Kd[319] = KC0e0713
                Kd[183] = KC0e0715
                Kd[367] = KC0e0715
                Kd[184] = KC0e0716
                Kd[391] = KC0e0716
                Kd[185] = KC0e0717
                Kd[415] = KC0e0717
                Kd[186] = KC0e0718
                Kd[439] = KC0e0718
                Kd[187] = KC0e0719
                Kd[463] = KC0e0719
                Kd[189] = KC0e0721
                Kd[511] = KC0e0721
                Kd[190] = KC0e0722
                Kd[535] = KC0e0722
                Kd[191] = KC0e0723
                Kd[559] = KC0e0723
                Kd[200] = KC0e0808
                Kd[201] = KC0e0809
                Kd[224] = KC0e0809
                Kd[202] = KC0e0810
                Kd[248] = KC0e0810
                Kd[206] = KC0e0814
                Kd[344] = KC0e0814
                Kd[207] = KC0e0815
                Kd[368] = KC0e0815
                Kd[208] = KC0e0816
                Kd[392] = KC0e0816
                Kd[212] = KC0e0820
                Kd[488] = KC0e0820
                Kd[213] = KC0e0821
                Kd[512] = KC0e0821
                Kd[214] = KC0e0822
                Kd[536] = KC0e0822
                Kd[225] = KC0e0909
                Kd[226] = KC0e0910
                Kd[249] = KC0e0910
                Kd[228] = KC0e0912
                Kd[297] = KC0e0912
                Kd[229] = KC0e0913
                Kd[321] = KC0e0913
                Kd[230] = KC0e0914
                Kd[345] = KC0e0914
                Kd[231] = KC0e0915
                Kd[369] = KC0e0915
                Kd[232] = KC0e0916
                Kd[393] = KC0e0916
                Kd[234] = KC0e0918
                Kd[441] = KC0e0918
                Kd[235] = KC0e0919
                Kd[465] = KC0e0919
                Kd[236] = KC0e0920
                Kd[489] = KC0e0920
                Kd[237] = KC0e0921
                Kd[513] = KC0e0921
                Kd[238] = KC0e0922
                Kd[537] = KC0e0922
                Kd[250] = KC0e1010
                Kd[252] = KC0e1012
                Kd[298] = KC0e1012
                Kd[253] = KC0e1013
                Kd[322] = KC0e1013
                Kd[254] = KC0e1014
                Kd[346] = KC0e1014
                Kd[255] = KC0e1015
                Kd[370] = KC0e1015
                Kd[256] = KC0e1016
                Kd[394] = KC0e1016
                Kd[258] = KC0e1018
                Kd[442] = KC0e1018
                Kd[259] = KC0e1019
                Kd[466] = KC0e1019
                Kd[260] = KC0e1020
                Kd[490] = KC0e1020
                Kd[261] = KC0e1021
                Kd[514] = KC0e1021
                Kd[262] = KC0e1022
                Kd[538] = KC0e1022
                Kd[275] = KC0e1111
                Kd[276] = KC0e1112
                Kd[299] = KC0e1112
                Kd[277] = KC0e1113
                Kd[323] = KC0e1113
                Kd[281] = KC0e1117
                Kd[419] = KC0e1117
                Kd[282] = KC0e1118
                Kd[443] = KC0e1118
                Kd[283] = KC0e1119
                Kd[467] = KC0e1119
                Kd[287] = KC0e1123
                Kd[563] = KC0e1123
                Kd[300] = KC0e1212
                Kd[301] = KC0e1213
                Kd[324] = KC0e1213
                Kd[303] = KC0e1215
                Kd[372] = KC0e1215
                Kd[304] = KC0e1216
                Kd[396] = KC0e1216
                Kd[305] = KC0e1217
                Kd[420] = KC0e1217
                Kd[306] = KC0e1218
                Kd[444] = KC0e1218
                Kd[307] = KC0e1219
                Kd[468] = KC0e1219
                Kd[309] = KC0e1221
                Kd[516] = KC0e1221
                Kd[310] = KC0e1222
                Kd[540] = KC0e1222
                Kd[311] = KC0e1223
                Kd[564] = KC0e1223
                Kd[325] = KC0e1313
                Kd[327] = KC0e1315
                Kd[373] = KC0e1315
                Kd[328] = KC0e1316
                Kd[397] = KC0e1316
                Kd[329] = KC0e1317
                Kd[421] = KC0e1317
                Kd[330] = KC0e1318
                Kd[445] = KC0e1318
                Kd[331] = KC0e1319
                Kd[469] = KC0e1319
                Kd[333] = KC0e1321
                Kd[517] = KC0e1321
                Kd[334] = KC0e1322
                Kd[541] = KC0e1322
                Kd[335] = KC0e1323
                Kd[565] = KC0e1323
                Kd[350] = KC0e1414
                Kd[351] = KC0e1415
                Kd[374] = KC0e1415
                Kd[352] = KC0e1416
                Kd[398] = KC0e1416
                Kd[356] = KC0e1420
                Kd[494] = KC0e1420
                Kd[357] = KC0e1421
                Kd[518] = KC0e1421
                Kd[358] = KC0e1422
                Kd[542] = KC0e1422
                Kd[375] = KC0e1515
                Kd[376] = KC0e1516
                Kd[399] = KC0e1516
                Kd[378] = KC0e1518
                Kd[447] = KC0e1518
                Kd[379] = KC0e1519
                Kd[471] = KC0e1519
                Kd[380] = KC0e1520
                Kd[495] = KC0e1520
                Kd[381] = KC0e1521
                Kd[519] = KC0e1521
                Kd[382] = KC0e1522
                Kd[543] = KC0e1522
                Kd[400] = KC0e1616
                Kd[402] = KC0e1618
                Kd[448] = KC0e1618
                Kd[403] = KC0e1619
                Kd[472] = KC0e1619
                Kd[404] = KC0e1620
                Kd[496] = KC0e1620
                Kd[405] = KC0e1621
                Kd[520] = KC0e1621
                Kd[406] = KC0e1622
                Kd[544] = KC0e1622
                Kd[425] = KC0e1717
                Kd[426] = KC0e1718
                Kd[449] = KC0e1718
                Kd[427] = KC0e1719
                Kd[473] = KC0e1719
                Kd[431] = KC0e1723
                Kd[569] = KC0e1723
                Kd[450] = KC0e1818
                Kd[451] = KC0e1819
                Kd[474] = KC0e1819
                Kd[453] = KC0e1821
                Kd[522] = KC0e1821
                Kd[454] = KC0e1822
                Kd[546] = KC0e1822
                Kd[455] = KC0e1823
                Kd[570] = KC0e1823
                Kd[475] = KC0e1919
                Kd[477] = KC0e1921
                Kd[523] = KC0e1921
                Kd[478] = KC0e1922
                Kd[547] = KC0e1922
                Kd[479] = KC0e1923
                Kd[571] = KC0e1923
                Kd[500] = KC0e2020
                Kd[501] = KC0e2021
                Kd[524] = KC0e2021
                Kd[502] = KC0e2022
                Kd[548] = KC0e2022
                Kd[525] = KC0e2121
                Kd[526] = KC0e2122
                Kd[549] = KC0e2122
                Kd[550] = KC0e2222
                Kd[575] = KC0e2323
                _warp_dK(zw, Kd, dKd)
                Rw[0] = self.r11
                Rw[1] = self.r12
                Rw[2] = self.r13
                Rw[3] = self.r21
                Rw[4] = self.r22
                Rw[5] = self.r23
                Rw[6] = self.r31
                Rw[7] = self.r32
                Rw[8] = self.r33
                _rotate_add_24(dKd, Rw, &KC0v[self.init_k_KC0])


    cpdef void update_fint(Quad4R self,
                           double [::1] fint,
                           ShellProp prop,
                           double hgfactor_membrane = 1.,
                           double hgfactor_physical = 1.,
                           int nonlinear=0
                           ):
        r"""Update the internal force vector

        The element internal forces, computed in element coordinates with
        :meth:`.update_probe_finte`, are rotated to global coordinates and
        added to ``fint`` at the positions of the nodes, ``c1``, ``c2``,
        ``c3`` and ``c4``. Before this method is called,
        :meth:`.update_probe_xe` and :meth:`.update_probe_ue` must have been
        used.

        Parameters
        ----------
        fint : np.array
            Array that is updated in place with the internal forces. The
            internal forces stored in ``fint`` are calculated in global
            coordinates. Method :meth:`.update_probe_finte` is called to update
            the parameter ``finte`` of the :class:`.Quad4RProbe` with the
            internal forces in local coordinates.
            1-D array of size ``DOF*M``, with forces for the translations and
            moments (force times length) for the rotations.
        prop : :class:`.ShellProp` object
            Shell property object from where the stiffness and mass attributes
            are read from.
        hgfactor_membrane : double, optional
            Dimensionless multiplier of the artificial hourglass stiffness
            of the in-plane displacements ``u`` and ``v``, of Brockman
            (1987). Default ``1.``.
        hgfactor_physical : double, optional
            Dimensionless multiplier of the physical stabilisation of the
            bending and of the transverse shear of Belytschko and Leviathan
            (1994), which controls the hourglass modes of ``w``, ``r_x`` and
            ``r_y``; the default ``1.`` is the formulation of the paper,
            which has no parameter, and other values serve verification and
            sensitivity studies. See the Notes of :mod:`pyfe3d.quad4r`.
        nonlinear : int, optional
            The default ``0`` gives the linear internal forces, ``KC0*u``. Any other
            value adds the geometrically nonlinear terms of the von Kármán strains,
            whose tangent stiffness matrix is ``KC0 + KCNL + KG``, see
            :meth:`.update_KCNL`.

        Notes
        -----
        The internal forces of each node, forces and moments, are rotated
        from the element to the global coordinates with the rotation matrix
        ``r11, ..., r33``. With ``nonlinear``, ``KC0 + KCNL + KG`` is the
        exact Jacobian of the assembled internal forces, see
        :meth:`.update_probe_finte` and :meth:`.update_KCNL`.

        With ``warp_offsets = 1`` and a warped element the internal forces
        are those of the flat element at the displacements `\boldsymbol{J}
        \boldsymbol{u}_e`, taken to the nodes with `\boldsymbol{J}^\top`,
        see the Notes of :class:`.Quad4R`: the linear ones `\boldsymbol{J}^\top
        \boldsymbol{K}_e \boldsymbol{J} \boldsymbol{u}_e` and the terms of
        ``nonlinear``, see :meth:`.update_KCNL`.

        """
        cdef double *finte

        self.update_probe_finte(prop, hgfactor_membrane, hgfactor_physical, nonlinear)

        with nogil:
            finte = &self.probe.finte[0]

            fint[0+self.c1] += finte[0]*self.r11 + finte[1]*self.r12 + finte[2]*self.r13
            fint[1+self.c1] += finte[0]*self.r21 + finte[1]*self.r22 + finte[2]*self.r23
            fint[2+self.c1] += finte[0]*self.r31 + finte[1]*self.r32 + finte[2]*self.r33
            fint[3+self.c1] += finte[3]*self.r11 + finte[4]*self.r12 + finte[5]*self.r13
            fint[4+self.c1] += finte[3]*self.r21 + finte[4]*self.r22 + finte[5]*self.r23
            fint[5+self.c1] += finte[3]*self.r31 + finte[4]*self.r32 + finte[5]*self.r33
            fint[0+self.c2] += finte[6]*self.r11 + finte[7]*self.r12 + finte[8]*self.r13
            fint[1+self.c2] += finte[6]*self.r21 + finte[7]*self.r22 + finte[8]*self.r23
            fint[2+self.c2] += finte[6]*self.r31 + finte[7]*self.r32 + finte[8]*self.r33
            fint[3+self.c2] += finte[10]*self.r12 + finte[11]*self.r13 + finte[9]*self.r11
            fint[4+self.c2] += finte[10]*self.r22 + finte[11]*self.r23 + finte[9]*self.r21
            fint[5+self.c2] += finte[10]*self.r32 + finte[11]*self.r33 + finte[9]*self.r31
            fint[0+self.c3] += finte[12]*self.r11 + finte[13]*self.r12 + finte[14]*self.r13
            fint[1+self.c3] += finte[12]*self.r21 + finte[13]*self.r22 + finte[14]*self.r23
            fint[2+self.c3] += finte[12]*self.r31 + finte[13]*self.r32 + finte[14]*self.r33
            fint[3+self.c3] += finte[15]*self.r11 + finte[16]*self.r12 + finte[17]*self.r13
            fint[4+self.c3] += finte[15]*self.r21 + finte[16]*self.r22 + finte[17]*self.r23
            fint[5+self.c3] += finte[15]*self.r31 + finte[16]*self.r32 + finte[17]*self.r33
            fint[0+self.c4] += finte[18]*self.r11 + finte[19]*self.r12 + finte[20]*self.r13
            fint[1+self.c4] += finte[18]*self.r21 + finte[19]*self.r22 + finte[20]*self.r23
            fint[2+self.c4] += finte[18]*self.r31 + finte[19]*self.r32 + finte[20]*self.r33
            fint[3+self.c4] += finte[21]*self.r11 + finte[22]*self.r12 + finte[23]*self.r13
            fint[4+self.c4] += finte[21]*self.r21 + finte[22]*self.r22 + finte[23]*self.r23
            fint[5+self.c4] += finte[21]*self.r31 + finte[22]*self.r32 + finte[23]*self.r33


    cdef double _update_probe_BL_G(Quad4R self, double xi,
                                   double eta) noexcept nogil:
        r"""Update the probe rows of the linear strains and of the gradient of `w`

        Evaluated at the natural coordinates ``xi, eta``, in element coordinates:

        - ``BLexx, BLeyy, BLgxy``: membrane strains `\epsilon_{xx}, \epsilon_{yy},
          \gamma_{xy}`
        - ``BLkxx, BLkyy, BLkxy``: curvatures `\kappa_{xx}, \kappa_{yy},
          \kappa_{xy}`
        - ``Sw_x, Sw_y``: `\boldsymbol{G}_x` and `\boldsymbol{G}_y`, the rows of `w_{,x}` and `w_{,y}`,
          at the centroid for every ``xi, eta``

        Returns
        -------
        detJ : double
            Determinant of the Jacobian matrix at ``xi, eta``.

        Notes
        -----
        With the bilinear shape functions `S_i` and their Cartesian
        derivatives `S_{i,x}`, `S_{i,y}`, the products of the rows with
        ``ue`` are the strains of the Notes of :mod:`pyfe3d.quad4r`:
        ``BLexx``, ``BLeyy`` and ``BLgxy`` give `\epsilon_{xx} = u_{,x}`,
        `\epsilon_{yy} = v_{,y}` and `\gamma_{xy} = u_{,y} + v_{,x}`;
        ``BLkxx``, ``BLkyy`` and ``BLkxy`` give `\kappa_{xx} = {r_y}_{,x}`,
        `\kappa_{yy} = -{r_x}_{,y}` and `\kappa_{xy} = {r_y}_{,y} -
        {r_x}_{,x}`; and ``Sw_x``, ``Sw_y`` hold `S_{i,x}`, `S_{i,y}` at the `w`
        columns, evaluated at the centroid, see the Notes of
        :meth:`.update_KG`.

        """
        cdef int i
        cdef double x1, x2, x3, x4, y1, y2, y3, y4
        cdef double J11, J12, J21, J22, detJ
        cdef double j11, j12, j21, j22
        cdef double S1_x, S2_x, S3_x, S4_x, S1_y, S2_y, S3_y, S4_y
        cdef double Scx[4]
        cdef double Scy[4]
        cdef double *BLexx
        cdef double *BLeyy
        cdef double *BLgxy
        cdef double *BLkxx
        cdef double *BLkyy
        cdef double *BLkxy
        cdef double *Sw_x
        cdef double *Sw_y

        BLexx = &self.probe.BLexx[0]
        BLeyy = &self.probe.BLeyy[0]
        BLgxy = &self.probe.BLgxy[0]
        BLkxx = &self.probe.BLkxx[0]
        BLkyy = &self.probe.BLkyy[0]
        BLkxy = &self.probe.BLkxy[0]
        Sw_x = &self.probe.Sw_x[0]
        Sw_y = &self.probe.Sw_y[0]

        # NOTE ignoring z in local coordinates
        x1 = self.probe.xe[0]
        y1 = self.probe.xe[1]
        x2 = self.probe.xe[3]
        y2 = self.probe.xe[4]
        x3 = self.probe.xe[6]
        y3 = self.probe.xe[7]
        x4 = self.probe.xe[9]
        y4 = self.probe.xe[10]

        J11 = -0.5*x1 + 0.5*x2 + 0.5*(eta + 1)*(0.5*x1 - 0.5*x2 + 0.5*x3 - 0.5*x4)
        J12 = -0.5*y1 + 0.5*y2 + 0.5*(eta + 1)*(0.5*y1 - 0.5*y2 + 0.5*y3 - 0.5*y4)
        J21 = -0.5*x1 + 0.5*x4 - 0.25*(-x1 + x2)*(xi + 1) + 0.25*(x3 - x4)*(xi + 1)
        J22 = -0.5*y1 + 0.5*y4 - 0.25*(xi + 1)*(-y1 + y2) + 0.25*(xi + 1)*(y3 - y4)

        detJ = J11*J22 - J12*J21

        j11 = J22/detJ
        j12 = -J12/detJ
        j21 = -J21/detJ
        j22 = J11/detJ

        S1_x = 0.25*j11*(eta - 1) + 0.25*j12*(xi - 1)
        S2_x = -0.25*eta*j11 + 0.25*j11 - 0.25*j12*xi - 0.25*j12
        S3_x = 0.25*j11*(eta + 1) + 0.25*j12*(xi + 1)
        S4_x = -0.25*eta*j11 - 0.25*j11 - 0.25*j12*xi + 0.25*j12

        S1_y = 0.25*j21*(eta - 1) + 0.25*j22*(xi - 1)
        S2_y = -0.25*eta*j21 + 0.25*j21 - 0.25*j22*xi - 0.25*j22
        S3_y = 0.25*j21*(eta + 1) + 0.25*j22*(xi + 1)
        S4_y = -0.25*eta*j21 - 0.25*j21 - 0.25*j22*xi + 0.25*j22

        for i in range(24):
            BLexx[i] = 0.
            BLeyy[i] = 0.
            BLgxy[i] = 0.
            BLkxx[i] = 0.
            BLkyy[i] = 0.
            BLkxy[i] = 0.
            Sw_x[i] = 0.
            Sw_y[i] = 0.

        # exx = u,x
        BLexx[0] = S1_x
        BLexx[6] = S2_x
        BLexx[12] = S3_x
        BLexx[18] = S4_x

        # eyy = v,y
        BLeyy[1] = S1_y
        BLeyy[7] = S2_y
        BLeyy[13] = S3_y
        BLeyy[19] = S4_y

        # gxy = u,y + v,x
        BLgxy[0] = S1_y
        BLgxy[6] = S2_y
        BLgxy[12] = S3_y
        BLgxy[18] = S4_y
        BLgxy[1] = S1_x
        BLgxy[7] = S2_x
        BLgxy[13] = S3_x
        BLgxy[19] = S4_x

        # kxx = ry,x
        BLkxx[4] = S1_x
        BLkxx[10] = S2_x
        BLkxx[16] = S3_x
        BLkxx[22] = S4_x

        # kyy = -rx,y
        BLkyy[3] = -S1_y
        BLkyy[9] = -S2_y
        BLkyy[15] = -S3_y
        BLkyy[21] = -S4_y

        # kxy = ry,y - rx,x
        BLkxy[3] = -S1_x
        BLkxy[9] = -S2_x
        BLkxy[15] = -S3_x
        BLkxy[21] = -S4_x
        BLkxy[4] = S1_y
        BLkxy[10] = S2_y
        BLkxy[16] = S3_y
        BLkxy[22] = S4_y

        # NOTE the slopes of w are taken at the centroid at every integration
        #      point, see the Notes of update_KG
        _centroid_slopes(x1, x2, x3, x4, y1, y2, y3, y4, Scx, Scy)

        # w,x
        Sw_x[2] = Scx[0]
        Sw_x[8] = Scx[1]
        Sw_x[14] = Scx[2]
        Sw_x[20] = Scx[3]

        # w,y
        Sw_y[2] = Scy[0]
        Sw_y[8] = Scy[1]
        Sw_y[14] = Scy[2]
        Sw_y[20] = Scy[3]

        return detJ


    cdef void _update_probe_KCNLve(Quad4R self, ShellProp prop,
                                   double *fw) noexcept nogil:
        r"""Update the probe values of the nonlinear stiffness matrix

        The attribute ``KCNLve`` of the :class:`.Quad4RProbe` is updated with
        KCNL in element coordinates, stored row by row and evaluated at the
        displacements ``ue`` of the probe, with a two-point Gauss-Legendre
        quadrature in each direction and the slopes of `w` at the centroid.
        See :meth:`.update_KCNL`.

        Notes
        -----
        KCNL = KC0L + KCL0 + KCLL + KGNL, see the Notes of
        :meth:`.update_KCNL`. At each integration point, with the rows of
        :meth:`._update_probe_BL_G`, the slopes `w_{,x}` and `w_{,y}` are
        `\boldsymbol{G}_x \boldsymbol{u}_e` and `\boldsymbol{G}_y \boldsymbol{u}_e`, with the rows
        `\boldsymbol{G}_x` ``Sw_x`` and `\boldsymbol{G}_y` ``Sw_y``, the rows of
        `\boldsymbol{B}_{NL}` are `w_{,x} \boldsymbol{G}_x`, `w_{,y} \boldsymbol{G}_y` and
        `w_{,x} \boldsymbol{G}_y + w_{,y} \boldsymbol{G}_x`, and `\boldsymbol{N}_{NL} =
        \{N_{NL,xx}, N_{NL,yy}, N_{NL,xy}\}^\top = \boldsymbol{A} \{w_{,x}^2/2,
        w_{,y}^2/2, w_{,x} w_{,y}\}^\top`, such that

        .. math::

            \boldsymbol{K}_{CNL} + \boldsymbol{K}_{GNL} = \sum w_{ij} \det J \left( (\boldsymbol{A} \boldsymbol{B}_m + \boldsymbol{B} \boldsymbol{B}_b)^\top \boldsymbol{B}_{NL}
                + \boldsymbol{B}_{NL}^\top (\boldsymbol{A} \boldsymbol{B}_m + \boldsymbol{B} \boldsymbol{B}_b)
                + \boldsymbol{B}_{NL}^\top \boldsymbol{A} \boldsymbol{B}_{NL}
                + N_{NL,xx} \boldsymbol{G}_x^\top \boldsymbol{G}_x
                + N_{NL,yy} \boldsymbol{G}_y^\top \boldsymbol{G}_y
                + N_{NL,xy} (\boldsymbol{G}_x^\top \boldsymbol{G}_y
                + \boldsymbol{G}_y^\top \boldsymbol{G}_x) \right)

        `\boldsymbol{A}` and `\boldsymbol{B}` are rotated with ``m11, m12, m21, m22``.

        With ``warp_offsets = 1`` and a warped element, the matrix above is
        that of the flat element at `\boldsymbol{J} \boldsymbol{u}_e`, and
        ``KCNLve`` receives `\boldsymbol{J}^\top \boldsymbol{K}_{CNL}
        \boldsymbol{J} - \boldsymbol{K}_E`, with `\boldsymbol{K}_E` from
        the nodal forces ``fw`` of the flat element, in element
        coordinates, see :meth:`.update_KCNL`; ``fw`` is not read
        otherwise.

        """
        cdef int i, j, a
        cdef int pti, ptj
        cdef double xi, eta
        cdef double points[2]
        cdef double wij, detJ, w_x, w_y
        cdef double A[9]
        cdef double B[9]
        cdef double NNL[3]
        # NOTE products stored row by row, with 3 rows and NUM_NODES*DOF columns
        cdef double ABL[72]
        cdef double BNL[72]
        cdef double ABNL[72]
        cdef int warped
        cdef double zw[4]
        cdef double uw[24]
        cdef double *ue
        cdef double *KCNLve
        cdef double *BLexx
        cdef double *BLeyy
        cdef double *BLgxy
        cdef double *BLkxx
        cdef double *BLkyy
        cdef double *BLkxy
        cdef double *Sw_x
        cdef double *Sw_y

        ue = &self.probe.ue[0]
        # NOTE rigid offsets: the flat element has the displacements J ue
        warped = _warp_z(&self.probe.xe[0], self.warp_offsets, zw)
        if warped:
            _warp_ue(zw, ue, uw)
            ue = uw
        KCNLve = &self.probe.KCNLve[0]
        BLexx = &self.probe.BLexx[0]
        BLeyy = &self.probe.BLeyy[0]
        BLgxy = &self.probe.BLgxy[0]
        BLkxx = &self.probe.BLkxx[0]
        BLkyy = &self.probe.BLkyy[0]
        BLkxy = &self.probe.BLkxy[0]
        Sw_x = &self.probe.Sw_x[0]
        Sw_y = &self.probe.Sw_y[0]

        prop.get_constitutive_element(self.m11, self.m12, self.m21, self.m22, A, B, NULL, NULL)

        for i in range(24*24):
            KCNLve[i] = 0.

        # NOTE same two-point Gauss-Legendre quadrature as in update_KG
        wij = 1.
        points[0] = -0.5773502691896257645092
        points[1] = +0.5773502691896257645092

        for pti in range(2):
            xi = points[pti]
            for ptj in range(2):
                eta = points[ptj]
                detJ = self._update_probe_BL_G(xi, eta)

                w_x = 0.
                w_y = 0.
                for i in range(24):
                    w_x += Sw_x[i]*ue[i]
                    w_y += Sw_y[i]*ue[i]

                # stress resultants of the nonlinear membrane strain,
                # epsNL = {w_x**2/2, w_y**2/2, w_x*w_y}
                for a in range(3):
                    NNL[a] = A[3*a]*w_x*w_x/2. + A[3*a + 1]*w_y*w_y/2. + A[3*a + 2]*w_x*w_y

                # BNL, the variation of the nonlinear membrane strain, and the products
                # A*Bm + B*Bb and A*BNL, all stored row by row with 3 rows
                for i in range(24):
                    BNL[i] = w_x*Sw_x[i]
                    BNL[24 + i] = w_y*Sw_y[i]
                    BNL[48 + i] = w_x*Sw_y[i] + w_y*Sw_x[i]
                    for a in range(3):
                        ABL[24*a + i] = (A[3*a]*BLexx[i] + A[3*a + 1]*BLeyy[i] + A[3*a + 2]*BLgxy[i]
                                       + B[3*a]*BLkxx[i] + B[3*a + 1]*BLkyy[i] + B[3*a + 2]*BLkxy[i])
                        ABNL[24*a + i] = A[3*a]*BNL[i] + A[3*a + 1]*BNL[24 + i] + A[3*a + 2]*BNL[48 + i]

                for i in range(24):
                    for j in range(24):
                        KCNLve[24*i + j] += wij*detJ*(
                            # KC0L = (Bm.T*A + Bb.T*B)*BNL
                              ABL[i]*BNL[j] + ABL[24 + i]*BNL[24 + j] + ABL[48 + i]*BNL[48 + j]
                            # KCL0 = BNL.T*(A*Bm + B*Bb)
                            + BNL[i]*ABL[j] + BNL[24 + i]*ABL[24 + j] + BNL[48 + i]*ABL[48 + j]
                            # KCLL = BNL.T*A*BNL
                            + BNL[i]*ABNL[j] + BNL[24 + i]*ABNL[24 + j] + BNL[48 + i]*ABNL[48 + j]
                            # KGNL = G.T*[NNL]*G
                            + Sw_x[i]*(NNL[0]*Sw_x[j] + NNL[2]*Sw_y[j])
                            + Sw_y[i]*(NNL[2]*Sw_x[j] + NNL[1]*Sw_y[j])
                        )

        if warped:
            # NOTE J^T KCNL J of the flat element, minus the eccentricity
            #      term that update_KG adds, see the Notes of update_KCNL
            _warp_K(zw, KCNLve)
            _warp_KE_sub(zw, fw, KCNLve)


    cdef void _update_probe_finte_nonlinear(Quad4R self,
                                            ShellProp prop) noexcept nogil:
        r"""Add the geometrically nonlinear terms to the probe internal forces

        The attribute ``finte`` of the :class:`.Quad4RProbe` receives the terms
        of the nonlinear part of the von Kármán membrane strains, evaluated at
        the displacements ``ue`` of the probe with a two-point Gauss-Legendre
        quadrature in each direction and the slopes of `w` at the centroid.
        See :meth:`.update_KCNL`.

        Notes
        -----
        With the nonlinear membrane strain `\boldsymbol{\epsilon}_{NL} =
        \{w_{,x}^2/2, w_{,y}^2/2, w_{,x} w_{,y}\}^\top`, ``finte`` becomes the
        gradient of the strain energy whose Hessian is KC0 + KCNL + KG. At
        each integration point, with the rows of :meth:`._update_probe_BL_G`
        and `\boldsymbol{B}_{NL}` of :meth:`._update_probe_KCNLve`, the stress
        resultants of the linear strains `\boldsymbol{N} = \boldsymbol{A} \boldsymbol{\epsilon} +
        \boldsymbol{B} \boldsymbol{\kappa}`, as in :meth:`.update_KG`, and those of the nonlinear
        strain `\boldsymbol{N}_{NL} = \boldsymbol{A} \boldsymbol{\epsilon}_{NL}` and `\boldsymbol{M}_{NL} =
        \boldsymbol{B} \boldsymbol{\epsilon}_{NL}`, the added terms are

        .. math::

            \sum w_{ij} \det J \left( \boldsymbol{B}_m^\top \boldsymbol{N}_{NL} + \boldsymbol{B}_b^\top \boldsymbol{M}_{NL}
                + \boldsymbol{B}_{NL}^\top (\boldsymbol{N} + \boldsymbol{N}_{NL}) \right)

        With ``warp_offsets = 1`` and a warped element, these terms are
        evaluated at the displacements `\boldsymbol{J} \boldsymbol{u}_e` of
        the flat element and multiplied by `\boldsymbol{J}^\top` before they
        are added, see :meth:`.update_KCNL`.

        """
        cdef int i, a
        cdef int pti, ptj
        cdef double xi, eta
        cdef double points[2]
        cdef double wij, detJ, w_x, w_y
        cdef double exx, eyy, gxy, kxx, kyy, kxy
        cdef double A[9]
        cdef double B[9]
        cdef double N[3]
        cdef double NNL[3]
        cdef double MNL[3]
        cdef double epsNL[3]
        cdef int warped
        cdef double zw[4]
        cdef double uw[24]
        cdef double fn[24]
        cdef double *fout
        cdef double *ue
        cdef double *finte
        cdef double *BLexx
        cdef double *BLeyy
        cdef double *BLgxy
        cdef double *BLkxx
        cdef double *BLkyy
        cdef double *BLkxy
        cdef double *Sw_x
        cdef double *Sw_y

        ue = &self.probe.ue[0]
        finte = &self.probe.finte[0]
        # NOTE rigid offsets: the flat element has the displacements J ue,
        #      and its forces, accumulated in fn, are taken to the nodes
        #      with J^T
        fout = finte
        warped = _warp_z(&self.probe.xe[0], self.warp_offsets, zw)
        if warped:
            _warp_ue(zw, ue, uw)
            ue = uw
            for i in range(24):
                fn[i] = 0.
            fout = fn
        BLexx = &self.probe.BLexx[0]
        BLeyy = &self.probe.BLeyy[0]
        BLgxy = &self.probe.BLgxy[0]
        BLkxx = &self.probe.BLkxx[0]
        BLkyy = &self.probe.BLkyy[0]
        BLkxy = &self.probe.BLkxy[0]
        Sw_x = &self.probe.Sw_x[0]
        Sw_y = &self.probe.Sw_y[0]

        prop.get_constitutive_element(self.m11, self.m12, self.m21, self.m22, A, B, NULL, NULL)

        # NOTE same two-point Gauss-Legendre quadrature as in update_KG
        wij = 1.
        points[0] = -0.5773502691896257645092
        points[1] = +0.5773502691896257645092

        for pti in range(2):
            xi = points[pti]
            for ptj in range(2):
                eta = points[ptj]
                detJ = self._update_probe_BL_G(xi, eta)

                exx = 0.
                eyy = 0.
                gxy = 0.
                kxx = 0.
                kyy = 0.
                kxy = 0.
                w_x = 0.
                w_y = 0.
                for i in range(24):
                    exx += BLexx[i]*ue[i]
                    eyy += BLeyy[i]*ue[i]
                    gxy += BLgxy[i]*ue[i]
                    kxx += BLkxx[i]*ue[i]
                    kyy += BLkyy[i]*ue[i]
                    kxy += BLkxy[i]*ue[i]
                    w_x += Sw_x[i]*ue[i]
                    w_y += Sw_y[i]*ue[i]

                epsNL[0] = w_x*w_x/2.
                epsNL[1] = w_y*w_y/2.
                epsNL[2] = w_x*w_y

                for a in range(3):
                    # stress resultants of the linear strains, as in update_KG
                    N[a] = (A[3*a]*exx + A[3*a + 1]*eyy + A[3*a + 2]*gxy
                          + B[3*a]*kxx + B[3*a + 1]*kyy + B[3*a + 2]*kxy)
                    # stress resultants of the nonlinear membrane strain
                    NNL[a] = A[3*a]*epsNL[0] + A[3*a + 1]*epsNL[1] + A[3*a + 2]*epsNL[2]
                    MNL[a] = B[3*a]*epsNL[0] + B[3*a + 1]*epsNL[1] + B[3*a + 2]*epsNL[2]

                for i in range(24):
                    fout[i] += wij*detJ*(
                        # Bm.T*NNL + Bb.T*MNL
                          BLexx[i]*NNL[0] + BLeyy[i]*NNL[1] + BLgxy[i]*NNL[2]
                        + BLkxx[i]*MNL[0] + BLkyy[i]*MNL[1] + BLkxy[i]*MNL[2]
                        # BNL.T*(N + NNL)
                        + w_x*Sw_x[i]*(N[0] + NNL[0])
                        + w_y*Sw_y[i]*(N[1] + NNL[1])
                        + (w_x*Sw_y[i] + w_y*Sw_x[i])*(N[2] + NNL[2])
                    )

        if warped:
            _warp_f(zw, fn)
            for i in range(24):
                finte[i] += fn[i]


    cpdef void update_KCNL(Quad4R self,
                           long [::1] KCNLr,
                           long [::1] KCNLc,
                           double [::1] KCNLv,
                           ShellProp prop,
                           int update_KCNLv_only=0,
                           double hgfactor_membrane=1.,
                           double hgfactor_physical=1.
                           ):
        r"""Update sparse vectors for the nonlinear stiffness matrix KCNL

        KCNL collects the terms of the tangent stiffness matrix of the
        geometrically nonlinear analysis with the von Kármán membrane strains
        that are not in KC0 and in KG, evaluated at the displacements of the
        probe with the quadrature of :meth:`.update_KG`. The tangent stiffness
        matrix of the internal forces of :meth:`.update_fint` with
        ``nonlinear=1`` is ``KC0 + KCNL + KG``.

        Before this function is called, the probe :class:`.Quad4RProbe` attribute
        of the :class:`.Quad4R` object must be updated using
        :meth:`.update_probe_ue` with the current displacements; and
        :meth:`.update_probe_xe` with the node coordinates.

        The element matrix is computed in element coordinates into the probe
        attribute ``KCNLve``, rotated to global coordinates, and its values
        are added to ``KCNLv`` at the positions ``init_k_KCNL`` to
        ``init_k_KCNL + KCNL_SPARSE_SIZE - 1``. With ``update_KCNLv_only=0``
        the global row and column indices are written to ``KCNLr`` and
        ``KCNLc`` at the same positions.

        Parameters
        ----------
        KCNLr : np.array
            Array to store row positions of sparse values
            1-D array of integers (``INT``) of size at least ``init_k_KCNL +
            KCNL_SPARSE_SIZE``.
        KCNLc : np.array
            Array to store column positions of sparse values
            Same type and size as ``KCNLr``.
        KCNLv : np.array
            Array to store sparse values
            1-D array of ``DOUBLE``, same size as ``KCNLr``. The values are
            added to the existing ones, with the dimensions given for
            ``KC0v`` in :meth:`.update_KC0`.
        prop : :class:`.ShellProp` object
            Shell property object from where the stiffness and mass attributes are
            read from.
        update_KCNLv_only : int, optional
            The default ``0`` means that the row and column indices ``KCNLr`` and
            ``KCNLc`` should also be updated. Any other value will only update the
            stiffness matrix values ``KCNLv``.
        hgfactor_membrane, hgfactor_physical : double, optional
            Read only with ``warp_offsets = 1`` and a warped element, for the
            nodal forces of the eccentricity term of the rigid offsets, which
            KCNL subtracts, see the Notes; they must be those passed to
            :meth:`.update_KG`, see its parameters. Default ``1.``.

        Notes
        -----
        KCNL = KC0L + KCL0 + KCLL + KGNL, that is `\boldsymbol{K}_{CNL} +
        \boldsymbol{K}_{GNL}`: the nonlinear constitutive stiffness matrix
        `\boldsymbol{K}_{CNL}`, the terms of `\int (\boldsymbol{B}_L +
        \boldsymbol{B}_{NL})^\top \boldsymbol{C} (\boldsymbol{B}_L +
        \boldsymbol{B}_{NL}) d\Omega` that contain `\boldsymbol{B}_{NL}`, plus
        the geometric stiffness matrix `\boldsymbol{K}_{GNL}` of the stress
        resultants of the nonlinear strains. Crisfield [2]_, Section 8.1.5,
        Eqs. (8.26)-(8.31), groups the same terms differently: his
        `\boldsymbol{K}_{t1}` is `\boldsymbol{K}_{C0} + \boldsymbol{K}_{CNL}`,
        and his initial-stress matrix `\boldsymbol{K}_{t\sigma}`, built with
        the total membrane forces, is `\boldsymbol{K}_G +
        \boldsymbol{K}_{GNL}`; the tangent stiffness matrix is the same. The
        matrices are built from the von Kármán membrane strains

        .. math::
            \epsilon_{xx} = u_{,x} + \frac{1}{2} w_{,x}^2, \quad
            \epsilon_{yy} = v_{,y} + \frac{1}{2} w_{,y}^2, \quad
            \gamma_{xy} = u_{,y} + v_{,x} + w_{,x} w_{,y}

        whose nonlinear part is `\boldsymbol{\epsilon}_{NL} = \frac{1}{2} \boldsymbol{B}_{NL}
        \boldsymbol{u}_e`, with `\boldsymbol{B}_{NL}` its variation. With `\boldsymbol{B}_m` and `\boldsymbol{B}_b` the
        linear membrane and bending strain-displacement matrices, `\boldsymbol{G}_x = \partial \boldsymbol{S}^w / \partial x` and `\boldsymbol{G}_y = \partial \boldsymbol{S}^w / \partial y`
        the gradients of `w`, and `\boldsymbol{A}`, `\boldsymbol{B}` the laminate matrices:

        - KC0L = `\boldsymbol{B}_m^\top \boldsymbol{A} \boldsymbol{B}_{NL} + \boldsymbol{B}_b^\top \boldsymbol{B} \boldsymbol{B}_{NL}`
        - KCL0 = KC0L`^\top`
        - KCLL = `\boldsymbol{B}_{NL}^\top \boldsymbol{A} \boldsymbol{B}_{NL}`
        - KGNL = `N_{NL,xx} \boldsymbol{G}_x^\top \boldsymbol{G}_x + N_{NL,yy}
          \boldsymbol{G}_y^\top \boldsymbol{G}_y + N_{NL,xy} (\boldsymbol{G}_x^\top
          \boldsymbol{G}_y + \boldsymbol{G}_y^\top \boldsymbol{G}_x)`, with
          `\boldsymbol{N}_{NL} = \{N_{NL,xx}, N_{NL,yy}, N_{NL,xy}\}^\top
          = \boldsymbol{A} \boldsymbol{\epsilon}_{NL}`

        The first three groups are the constitutive terms coupling the linear and
        the nonlinear parts of the membrane strain. KGNL is geometric, carrying the
        stress of the nonlinear membrane strain. It is collected here so that
        :meth:`.update_KG` stays homogeneous of degree one in the displacements,
        which is what a linear buckling analysis needs. With it here,

        .. math::
            \boldsymbol{K}_T = \boldsymbol{K}_{C0} + \boldsymbol{K}_{CNL}(\boldsymbol{u}) + \boldsymbol{K}_{GNL}(\boldsymbol{u}) + \boldsymbol{K}_G(\boldsymbol{u})

        is the exact Jacobian of the internal forces of :meth:`.update_fint` with
        ``nonlinear=1``, and a Newton-Raphson iteration built on them converges
        quadratically. The quadrature of :meth:`.update_KG` is used.

        The linear part of the internal forces, ``KC0*u``, keeps the
        quadratures of KC0, whereas the nonlinear terms of
        :meth:`._update_probe_finte_nonlinear`, KCNL and KG use the
        two-by-two quadrature, with the slopes of `w` at the centroid, see
        :meth:`.update_KG`; each part of the strain energy is differentiated
        with its own quadrature, so the Jacobian remains exact.
        The element matrix is computed by :meth:`._update_probe_KCNLve` and
        rotated to global coordinates as in :meth:`.update_KC0`.

        Warped elements, ``warp_offsets = 1``, see the Notes of
        :class:`.Quad4R`. The von Kármán strains are those of the flat element,
        linked to the nodes by the first-order rigid offsets
        `\boldsymbol{u}^f = \boldsymbol{J} \boldsymbol{u}_e` (derived
        here): the strain energy is that of the flat element at
        `\boldsymbol{J} \boldsymbol{u}_e`, the internal forces are
        `\boldsymbol{J}^\top` times those of the flat element, and its exact
        Jacobian is `\boldsymbol{J}^\top (\boldsymbol{K}_{C0} +
        \boldsymbol{K}_{CNL} + \boldsymbol{K}_{GNL} + \boldsymbol{K}_G)
        \boldsymbol{J}` of the flat element at `\boldsymbol{J}
        \boldsymbol{u}_e`, where `\boldsymbol{J}^\top \boldsymbol{K}_G
        \boldsymbol{J} = \boldsymbol{K}_G`. The link is kept to first order in the
        rotations because the von Kármán strains themselves are: with the
        ordering of Sanders (1963), p. 30, Eqs. (67)-(69), and p. 32, Eq.
        (89), strains of order `\epsilon^2`, rotations about the in-plane
        axes of order `\epsilon` and a smaller rotation about the normal,
        the second-order terms of the offsets, `-\frac{1}{2}
        \boldsymbol{\theta}_a \times (\boldsymbol{\theta}_a \times
        \boldsymbol{r}_a)`, change the strains by terms of order
        `(\tilde{z}/\ell) \epsilon^3`, beyond those kept (derived here).
        Measured on the twisted beam of ``tests/test_quad_warp_offsets.py``
        with 12 by 2 elements and rotations up to 0.3, the second-order link
        gives tip displacements 4 to 7 % smaller than the first-order one,
        whose results with Quad4 are within 0.6 % of the converged von
        Kármán solution. The eccentricity
        term `\boldsymbol{K}_E` that :meth:`.update_KG` adds for the linear
        buckling analysis, the second derivative of the second-order terms,
        is therefore subtracted here, with the same nodal forces, those of
        :meth:`.update_probe_finte`, whose ``probe.finte`` is restored,
        such that KCNL is `\boldsymbol{J}^\top (\boldsymbol{K}_{CNL} +
        \boldsymbol{K}_{GNL}) \boldsymbol{J} - \boldsymbol{K}_E`, rotated to
        global coordinates, and ``KC0 + KCNL
        + KG`` remains the exact Jacobian of the internal forces of
        :meth:`.update_fint` with ``nonlinear=1``. A flat element and
        ``warp_offsets = 0`` give the results of the flat element, bit for
        bit.

        References
        ----------
        .. [1] Sanders, J. L., Nonlinear theories for thin shells, Quarterly
           of Applied Mathematics, 21(1), 21-36, 1963.
           https://doi.org/10.1090/qam/147023
        .. [2] Crisfield, M. A., Non-linear Finite Element Analysis of Solids
           and Structures, Volume 1: Essentials, John Wiley & Sons,
           Chichester, 1991.

        """
        cdef int i, j, node_i, node_j, k, ke, m, n
        cdef int c[4]
        cdef double r[6][6]
        cdef int warped, iw
        cdef double zw[4]
        cdef double fsave[24]
        cdef double fw[24]

        # NOTE rigid offsets: the translations of the linear internal forces
        #      of update_probe_finte are the nodal forces of the flat element
        #      of the eccentricity term, as in update_KG; finte of the probe
        #      is restored
        warped = _warp_z(&self.probe.xe[0], self.warp_offsets, zw)
        if warped:
            for iw in range(24):
                fsave[iw] = self.probe.finte[iw]
            self.update_probe_finte(prop, hgfactor_membrane, hgfactor_physical)
            for iw in range(24):
                fw[iw] = self.probe.finte[iw]
                self.probe.finte[iw] = fsave[iw]

        with nogil:
            # local to global transformation
            # translation DOFs
            r[0][0] = self.r11
            r[0][1] = self.r12
            r[0][2] = self.r13
            r[1][0] = self.r21
            r[1][1] = self.r22
            r[1][2] = self.r23
            r[2][0] = self.r31
            r[2][1] = self.r32
            r[2][2] = self.r33
            # rotation DOFs
            r[0+3][0+3] = self.r11
            r[0+3][1+3] = self.r12
            r[0+3][2+3] = self.r13
            r[1+3][0+3] = self.r21
            r[1+3][1+3] = self.r22
            r[1+3][2+3] = self.r23
            r[2+3][0+3] = self.r31
            r[2+3][1+3] = self.r32
            r[2+3][2+3] = self.r33
            # coupled translation-rotation DOFs
            for i in range(3):
                for j in range(3):
                    r[i][j+3] = 0.
                    r[i+3][j] = 0.

            if update_KCNLv_only == 0:
                # positions in the global stiffness matrix
                c[0] = self.c1
                c[1] = self.c2
                c[2] = self.c3
                c[3] = self.c4

                for node_i in range(NUM_NODES):
                    for m in range(DOF):
                        for node_j in range(NUM_NODES):
                            for n in range(DOF):
                                k = self.init_k_KCNL + 24*(node_i*DOF + m) + node_j*DOF + n
                                KCNLr[k] = c[node_i] + m
                                KCNLc[k] = c[node_j] + n

            self._update_probe_KCNLve(prop, fw)

            # NOTE from element to global coordinates:
            #
            # Kg = T @ Ke @ T.T
            #
            # in tensor notation:
            #
            # Kg_{mn} = r_{mi} * Ke_{ij} * r_{nj}
            #
            for node_i in range(NUM_NODES):
                for m in range(DOF):
                    for node_j in range(NUM_NODES):
                        for n in range(DOF):
                            k = self.init_k_KCNL + 24*(node_i*DOF + m) + node_j*DOF + n
                            for i in range(DOF):
                                for j in range(DOF):
                                    ke = 24*(node_i*DOF + i) + node_j*DOF + j
                                    KCNLv[k] += r[m][i]*self.probe.KCNLve[ke]*r[n][j]


    cpdef void update_KG(Quad4R self,
                         long [::1] KGr,
                         long [::1] KGc,
                         double [::1] KGv,
                         ShellProp prop,
                         int update_KGv_only=0,
                         double hgfactor_membrane=1.,
                         double hgfactor_physical=1.
                         ):
        r"""Update sparse vectors for geometric stiffness matrix KG

        The stress resultants are integrated with a two-point Gauss-Legendre
        quadrature in each direction and the slopes of `w` are taken at the
        centroid, see the Notes.

        Before this function is called, the probe :class:`.Quad4RProbe`
        attribute of the :class:`.Quad4R` object must be updated using
        :meth:`.update_probe_ue` with the correct pre-buckling displacements;
        and :meth:`.update_probe_xe` with the node coordinates.

        At each integration point the membrane stress resultants are obtained
        from the linear membrane strains and curvatures of the probe
        displacements ``ue`` and from the extensional and extension-bending
        coupling matrices of ``prop`` rotated with ``m11, m12, m21, m22``.
        Only the translational degrees-of-freedom appear. The values are
        added to ``KGv`` at the positions ``init_k_KG`` to ``init_k_KG +
        KG_SPARSE_SIZE - 1``. With ``update_KGv_only=0`` the global row and
        column indices are written to ``KGr`` and ``KGc`` at the same
        positions.

        Parameters
        ----------
        KGr : np.array
           Array to store row positions of sparse values
           1-D array of integers (``INT``) of size at least ``init_k_KG +
           KG_SPARSE_SIZE``.
        KGc : np.array
           Array to store column positions of sparse values
           Same type and size as ``KGr``.
        KGv : np.array
            Array to store sparse values
            1-D array of ``DOUBLE``, same size as ``KGr``. The values are
            added to the existing ones, with dimension of force per unit
            length.
        prop : :class:`.ShellProp` object
            Shell property object from where the stiffness and mass attributes
            are read from.
        update_KGv_only : int, optional
            The default ``0`` means that the row and column indices ``KGr``
            and ``KGc`` are also updated. Any other value will only update the
            values ``KGv``.
        hgfactor_membrane, hgfactor_physical : double, optional
            Read only with ``warp_offsets = 1`` and a warped element: the
            factors of :meth:`.update_probe_finte` for the nodal forces of the
            eccentricity term, see the Notes, which should be those of
            :meth:`.update_KC0` and :meth:`.update_fint`. The same values must
            be passed to :meth:`.update_KCNL` for the exact tangent of the
            von Kármán internal forces. Default ``1.``.

        Notes
        -----
        The matrix is

        .. math::

            \boldsymbol{K}_G = \sum w_{ij} \det J \left( N_{xx} \boldsymbol{G}_x^\top
            \boldsymbol{G}_x + N_{yy} \boldsymbol{G}_y^\top \boldsymbol{G}_y
            + N_{xy} (\boldsymbol{G}_x^\top \boldsymbol{G}_y
            + \boldsymbol{G}_y^\top \boldsymbol{G}_x) \right)

        with `\boldsymbol{G}_x` and `\boldsymbol{G}_y` the rows of `w_{,x}`
        and `w_{,y}` at the centroid, see :meth:`._update_probe_BL_G`, `w` being the displacement along the
        element normal, i.e. the combination of the global translations with
        ``r13, r23, r33``, and `\{N_{xx}, N_{yy}, N_{xy}\}^\top = \boldsymbol{A} \boldsymbol{\epsilon}
        + \boldsymbol{B} \boldsymbol{\kappa}` the stress resultants of the linear strains of the
        Notes of :mod:`pyfe3d.quad4r`. KG is linear in the displacements
        ``ue``, such that ``KC0 + lambda*KG`` gives the linear buckling problem
        for the load multiplier ``lambda`` of the pre-buckling state.

        The slopes are taken at the centroid, the single integration point of
        the transverse shear and of the curvatures in KC0, whereas the stress
        resultants are integrated with two points in each direction. The
        hourglass pattern of `w`, `w = \xi \eta`, has no gradient at the
        centroid, so it has no geometric stiffness, just as it has no
        stiffness in KC0 besides the stabilisation of the transverse shear,
        see :meth:`.update_KC0`. With the slopes at the two-point quadrature
        points, as before version 0.12.0, that pattern had the geometric
        stiffness of the full quadrature, and a linear buckling analysis could
        return a spurious mode of `w` alternating in sign from node to node,
        below the physical buckling load. On a simply supported square plate
        with `b/h = 10` under uniaxial compression, the lowest buckling load
        was 41 % below the Mindlin solution with 16 by 16 elements, `h/\ell =
        1.6`, and 86 % below with 48 by 48 elements; with `b/h = 100` and the
        transverse shear stiffness reduced to 1/1000 of the isotropic one, as
        in a sandwich, it was 60 % below with 8 by 8 elements, `h/\ell =
        0.08`. The thin elements of the isotropic plate were protected only
        because the transverse shear energy of the smooth envelope of that
        pattern exceeded its geometric work. With the slopes at the centroid
        these cases converge to the Mindlin solution, and the buckling loads
        of coarse meshes of thin plates increase slightly, e.g. 11.4 % instead
        of 10.0 % above the Mindlin solution with 8 by 8 elements, `b/h =
        100` and hard simple supports. See
        ``tests/test_quad4r_geometric_hourglass.py``.

        Warped elements, ``warp_offsets = 1``, see the Notes of
        :class:`.Quad4R`: the stress resultants are those of the displacements
        `\boldsymbol{J} \boldsymbol{u}_e` of the flat element, and the matrix
        above, which has no entries of the in-plane translations, is not
        changed by `\boldsymbol{J}^\top \boldsymbol{K}_G \boldsymbol{J}`. The
        rigid offsets add on the rotations of each node the eccentricity
        geometric stiffness, which Felippa and Haugen (2005), Section 5.1,
        p. 2304, name without giving it (see the references of
        :class:`pyfe3d.corotational.CorotationalShell`),

        .. math::

            \boldsymbol{K}_{E,a} = (\boldsymbol{n}_a \cdot \boldsymbol{r}_a)
            \boldsymbol{I} - \frac{1}{2} \left( \boldsymbol{n}_a
            \boldsymbol{r}_a^\top + \boldsymbol{r}_a \boldsymbol{n}_a^\top
            \right)

        the second derivative, at zero rotation, of the work of the nodal
        forces `\boldsymbol{n}_a` of the flat element on the displacement
        `-(\boldsymbol{R}_a - \boldsymbol{I}) \boldsymbol{r}_a` of its node,
        `\boldsymbol{R}_a` the rotation matrix of the node (derived here),
        equal to ``dLambdaT_m(0, n x r) - Spin(n) Spin(r)`` of
        :mod:`pyfe3d.corotational` and linear in the pre-buckling state. The
        forces `\boldsymbol{n}_a` are the translations of the linear internal
        forces of :meth:`.update_probe_finte` with ``hgfactor_membrane`` and
        ``hgfactor_physical``, which are those of the flat element;
        ``probe.finte`` is restored. :meth:`.update_KCNL` subtracts
        `\boldsymbol{K}_E`, see its Notes, such that the tangent of the von
        Kármán internal forces does not contain it. The 36 entries of
        `\boldsymbol{K}_E` follow the 144 of the translations, ``(r_x, r_y,
        r_z)`` of node 1 to node 4 row by row, and are zero for a flat element
        and with ``warp_offsets = 0``.

        """
        cdef int warped, iw, jw
        cdef double zw[4]
        cdef double uw[24]
        cdef double fw[24]
        cdef double KEw[36]
        cdef double Rw[9]
        cdef double *ue
        cdef int c1, c2, c3, c4, i, j, k
        cdef double x1, x2, x3, x4
        cdef double y1, y2, y3, y4
        cdef double wij, detJ, xi, eta
        cdef double points[2]
        cdef double Ae[9]
        cdef double Be[9]
        # NOTE ABD in the element direction
        cdef double A11, A12, A16, A22, A26, A66
        cdef double B11, B12, B16, B22, B26, B66
        cdef double r13, r23, r33
        cdef double j11, j12, j21, j22
        cdef double S1_x, S2_x, S3_x, S4_x, S1_y, S2_y, S3_y, S4_y
        cdef double Scx[4]
        cdef double Scy[4]
        cdef double Nxx, Nyy, Nxy

        # NOTE rigid offsets: the flat element has the displacements J ue
        #      and the linear internal forces of update_probe_finte,
        #      whose translations are those of the flat element; finte
        #      of the probe is restored
        warped = _warp_z(&self.probe.xe[0], self.warp_offsets, zw)
        if warped:
            for iw in range(24):
                uw[iw] = self.probe.finte[iw]
            self.update_probe_finte(prop, hgfactor_membrane, hgfactor_physical)
            for iw in range(24):
                fw[iw] = self.probe.finte[iw]
                self.probe.finte[iw] = uw[iw]
            _warp_ue(zw, &self.probe.ue[0], uw)

        with nogil:
            # NOTE constitutive matrices in the element coordinate system,
            #      the same function is used by all element methods
            prop.get_constitutive_element(self.m11, self.m12, self.m21, self.m22, Ae, Be, NULL, NULL)
            A11 = Ae[0]
            A12 = Ae[1]
            A16 = Ae[2]
            A22 = Ae[4]
            A26 = Ae[5]
            A66 = Ae[8]
            B11 = Be[0]
            B12 = Be[1]
            B16 = Be[2]
            B22 = Be[4]
            B26 = Be[5]
            B66 = Be[8]

            # local to global transformation
            r13 = self.r13
            r23 = self.r23
            r33 = self.r33

            # NOTE ignoring z in local coordinates
            x1 = self.probe.xe[0]
            y1 = self.probe.xe[1]
            # z1 = self.probe.xe[2]
            x2 = self.probe.xe[3]
            y2 = self.probe.xe[4]
            # z2 = self.probe.xe[5]
            x3 = self.probe.xe[6]
            y3 = self.probe.xe[7]
            # z3 = self.probe.xe[8]
            x4 = self.probe.xe[9]
            y4 = self.probe.xe[10]
            # z4 = self.probe.xe[11]

            ue = &self.probe.ue[0]
            if warped:
                ue = uw

            if update_KGv_only == 0:
                # positions of nodes 1,2,3,4 in the global matrix
                c1 = self.c1
                c2 = self.c2
                c3 = self.c3
                c4 = self.c4

                k = self.init_k_KG
                KGr[k] = 0+c1
                KGc[k] = 0+c1
                k += 1
                KGr[k] = 0+c1
                KGc[k] = 1+c1
                k += 1
                KGr[k] = 0+c1
                KGc[k] = 2+c1
                k += 1
                KGr[k] = 0+c1
                KGc[k] = 0+c2
                k += 1
                KGr[k] = 0+c1
                KGc[k] = 1+c2
                k += 1
                KGr[k] = 0+c1
                KGc[k] = 2+c2
                k += 1
                KGr[k] = 0+c1
                KGc[k] = 0+c3
                k += 1
                KGr[k] = 0+c1
                KGc[k] = 1+c3
                k += 1
                KGr[k] = 0+c1
                KGc[k] = 2+c3
                k += 1
                KGr[k] = 0+c1
                KGc[k] = 0+c4
                k += 1
                KGr[k] = 0+c1
                KGc[k] = 1+c4
                k += 1
                KGr[k] = 0+c1
                KGc[k] = 2+c4
                k += 1
                KGr[k] = 1+c1
                KGc[k] = 0+c1
                k += 1
                KGr[k] = 1+c1
                KGc[k] = 1+c1
                k += 1
                KGr[k] = 1+c1
                KGc[k] = 2+c1
                k += 1
                KGr[k] = 1+c1
                KGc[k] = 0+c2
                k += 1
                KGr[k] = 1+c1
                KGc[k] = 1+c2
                k += 1
                KGr[k] = 1+c1
                KGc[k] = 2+c2
                k += 1
                KGr[k] = 1+c1
                KGc[k] = 0+c3
                k += 1
                KGr[k] = 1+c1
                KGc[k] = 1+c3
                k += 1
                KGr[k] = 1+c1
                KGc[k] = 2+c3
                k += 1
                KGr[k] = 1+c1
                KGc[k] = 0+c4
                k += 1
                KGr[k] = 1+c1
                KGc[k] = 1+c4
                k += 1
                KGr[k] = 1+c1
                KGc[k] = 2+c4
                k += 1
                KGr[k] = 2+c1
                KGc[k] = 0+c1
                k += 1
                KGr[k] = 2+c1
                KGc[k] = 1+c1
                k += 1
                KGr[k] = 2+c1
                KGc[k] = 2+c1
                k += 1
                KGr[k] = 2+c1
                KGc[k] = 0+c2
                k += 1
                KGr[k] = 2+c1
                KGc[k] = 1+c2
                k += 1
                KGr[k] = 2+c1
                KGc[k] = 2+c2
                k += 1
                KGr[k] = 2+c1
                KGc[k] = 0+c3
                k += 1
                KGr[k] = 2+c1
                KGc[k] = 1+c3
                k += 1
                KGr[k] = 2+c1
                KGc[k] = 2+c3
                k += 1
                KGr[k] = 2+c1
                KGc[k] = 0+c4
                k += 1
                KGr[k] = 2+c1
                KGc[k] = 1+c4
                k += 1
                KGr[k] = 2+c1
                KGc[k] = 2+c4
                k += 1
                KGr[k] = 0+c2
                KGc[k] = 0+c1
                k += 1
                KGr[k] = 0+c2
                KGc[k] = 1+c1
                k += 1
                KGr[k] = 0+c2
                KGc[k] = 2+c1
                k += 1
                KGr[k] = 0+c2
                KGc[k] = 0+c2
                k += 1
                KGr[k] = 0+c2
                KGc[k] = 1+c2
                k += 1
                KGr[k] = 0+c2
                KGc[k] = 2+c2
                k += 1
                KGr[k] = 0+c2
                KGc[k] = 0+c3
                k += 1
                KGr[k] = 0+c2
                KGc[k] = 1+c3
                k += 1
                KGr[k] = 0+c2
                KGc[k] = 2+c3
                k += 1
                KGr[k] = 0+c2
                KGc[k] = 0+c4
                k += 1
                KGr[k] = 0+c2
                KGc[k] = 1+c4
                k += 1
                KGr[k] = 0+c2
                KGc[k] = 2+c4
                k += 1
                KGr[k] = 1+c2
                KGc[k] = 0+c1
                k += 1
                KGr[k] = 1+c2
                KGc[k] = 1+c1
                k += 1
                KGr[k] = 1+c2
                KGc[k] = 2+c1
                k += 1
                KGr[k] = 1+c2
                KGc[k] = 0+c2
                k += 1
                KGr[k] = 1+c2
                KGc[k] = 1+c2
                k += 1
                KGr[k] = 1+c2
                KGc[k] = 2+c2
                k += 1
                KGr[k] = 1+c2
                KGc[k] = 0+c3
                k += 1
                KGr[k] = 1+c2
                KGc[k] = 1+c3
                k += 1
                KGr[k] = 1+c2
                KGc[k] = 2+c3
                k += 1
                KGr[k] = 1+c2
                KGc[k] = 0+c4
                k += 1
                KGr[k] = 1+c2
                KGc[k] = 1+c4
                k += 1
                KGr[k] = 1+c2
                KGc[k] = 2+c4
                k += 1
                KGr[k] = 2+c2
                KGc[k] = 0+c1
                k += 1
                KGr[k] = 2+c2
                KGc[k] = 1+c1
                k += 1
                KGr[k] = 2+c2
                KGc[k] = 2+c1
                k += 1
                KGr[k] = 2+c2
                KGc[k] = 0+c2
                k += 1
                KGr[k] = 2+c2
                KGc[k] = 1+c2
                k += 1
                KGr[k] = 2+c2
                KGc[k] = 2+c2
                k += 1
                KGr[k] = 2+c2
                KGc[k] = 0+c3
                k += 1
                KGr[k] = 2+c2
                KGc[k] = 1+c3
                k += 1
                KGr[k] = 2+c2
                KGc[k] = 2+c3
                k += 1
                KGr[k] = 2+c2
                KGc[k] = 0+c4
                k += 1
                KGr[k] = 2+c2
                KGc[k] = 1+c4
                k += 1
                KGr[k] = 2+c2
                KGc[k] = 2+c4
                k += 1
                KGr[k] = 0+c3
                KGc[k] = 0+c1
                k += 1
                KGr[k] = 0+c3
                KGc[k] = 1+c1
                k += 1
                KGr[k] = 0+c3
                KGc[k] = 2+c1
                k += 1
                KGr[k] = 0+c3
                KGc[k] = 0+c2
                k += 1
                KGr[k] = 0+c3
                KGc[k] = 1+c2
                k += 1
                KGr[k] = 0+c3
                KGc[k] = 2+c2
                k += 1
                KGr[k] = 0+c3
                KGc[k] = 0+c3
                k += 1
                KGr[k] = 0+c3
                KGc[k] = 1+c3
                k += 1
                KGr[k] = 0+c3
                KGc[k] = 2+c3
                k += 1
                KGr[k] = 0+c3
                KGc[k] = 0+c4
                k += 1
                KGr[k] = 0+c3
                KGc[k] = 1+c4
                k += 1
                KGr[k] = 0+c3
                KGc[k] = 2+c4
                k += 1
                KGr[k] = 1+c3
                KGc[k] = 0+c1
                k += 1
                KGr[k] = 1+c3
                KGc[k] = 1+c1
                k += 1
                KGr[k] = 1+c3
                KGc[k] = 2+c1
                k += 1
                KGr[k] = 1+c3
                KGc[k] = 0+c2
                k += 1
                KGr[k] = 1+c3
                KGc[k] = 1+c2
                k += 1
                KGr[k] = 1+c3
                KGc[k] = 2+c2
                k += 1
                KGr[k] = 1+c3
                KGc[k] = 0+c3
                k += 1
                KGr[k] = 1+c3
                KGc[k] = 1+c3
                k += 1
                KGr[k] = 1+c3
                KGc[k] = 2+c3
                k += 1
                KGr[k] = 1+c3
                KGc[k] = 0+c4
                k += 1
                KGr[k] = 1+c3
                KGc[k] = 1+c4
                k += 1
                KGr[k] = 1+c3
                KGc[k] = 2+c4
                k += 1
                KGr[k] = 2+c3
                KGc[k] = 0+c1
                k += 1
                KGr[k] = 2+c3
                KGc[k] = 1+c1
                k += 1
                KGr[k] = 2+c3
                KGc[k] = 2+c1
                k += 1
                KGr[k] = 2+c3
                KGc[k] = 0+c2
                k += 1
                KGr[k] = 2+c3
                KGc[k] = 1+c2
                k += 1
                KGr[k] = 2+c3
                KGc[k] = 2+c2
                k += 1
                KGr[k] = 2+c3
                KGc[k] = 0+c3
                k += 1
                KGr[k] = 2+c3
                KGc[k] = 1+c3
                k += 1
                KGr[k] = 2+c3
                KGc[k] = 2+c3
                k += 1
                KGr[k] = 2+c3
                KGc[k] = 0+c4
                k += 1
                KGr[k] = 2+c3
                KGc[k] = 1+c4
                k += 1
                KGr[k] = 2+c3
                KGc[k] = 2+c4
                k += 1
                KGr[k] = 0+c4
                KGc[k] = 0+c1
                k += 1
                KGr[k] = 0+c4
                KGc[k] = 1+c1
                k += 1
                KGr[k] = 0+c4
                KGc[k] = 2+c1
                k += 1
                KGr[k] = 0+c4
                KGc[k] = 0+c2
                k += 1
                KGr[k] = 0+c4
                KGc[k] = 1+c2
                k += 1
                KGr[k] = 0+c4
                KGc[k] = 2+c2
                k += 1
                KGr[k] = 0+c4
                KGc[k] = 0+c3
                k += 1
                KGr[k] = 0+c4
                KGc[k] = 1+c3
                k += 1
                KGr[k] = 0+c4
                KGc[k] = 2+c3
                k += 1
                KGr[k] = 0+c4
                KGc[k] = 0+c4
                k += 1
                KGr[k] = 0+c4
                KGc[k] = 1+c4
                k += 1
                KGr[k] = 0+c4
                KGc[k] = 2+c4
                k += 1
                KGr[k] = 1+c4
                KGc[k] = 0+c1
                k += 1
                KGr[k] = 1+c4
                KGc[k] = 1+c1
                k += 1
                KGr[k] = 1+c4
                KGc[k] = 2+c1
                k += 1
                KGr[k] = 1+c4
                KGc[k] = 0+c2
                k += 1
                KGr[k] = 1+c4
                KGc[k] = 1+c2
                k += 1
                KGr[k] = 1+c4
                KGc[k] = 2+c2
                k += 1
                KGr[k] = 1+c4
                KGc[k] = 0+c3
                k += 1
                KGr[k] = 1+c4
                KGc[k] = 1+c3
                k += 1
                KGr[k] = 1+c4
                KGc[k] = 2+c3
                k += 1
                KGr[k] = 1+c4
                KGc[k] = 0+c4
                k += 1
                KGr[k] = 1+c4
                KGc[k] = 1+c4
                k += 1
                KGr[k] = 1+c4
                KGc[k] = 2+c4
                k += 1
                KGr[k] = 2+c4
                KGc[k] = 0+c1
                k += 1
                KGr[k] = 2+c4
                KGc[k] = 1+c1
                k += 1
                KGr[k] = 2+c4
                KGc[k] = 2+c1
                k += 1
                KGr[k] = 2+c4
                KGc[k] = 0+c2
                k += 1
                KGr[k] = 2+c4
                KGc[k] = 1+c2
                k += 1
                KGr[k] = 2+c4
                KGc[k] = 2+c2
                k += 1
                KGr[k] = 2+c4
                KGc[k] = 0+c3
                k += 1
                KGr[k] = 2+c4
                KGc[k] = 1+c3
                k += 1
                KGr[k] = 2+c4
                KGc[k] = 2+c3
                k += 1
                KGr[k] = 2+c4
                KGc[k] = 0+c4
                k += 1
                KGr[k] = 2+c4
                KGc[k] = 1+c4
                k += 1
                KGr[k] = 2+c4
                KGc[k] = 2+c4
                # NOTE rotations of each node, eccentricity geometric stiffness
                #      of the rigid offsets, see the Notes
                k = self.init_k_KG + 144
                for iw in range(3):
                    for jw in range(3):
                        KGr[k] = 3 + iw + c1
                        KGc[k] = 3 + jw + c1
                        k += 1
                for iw in range(3):
                    for jw in range(3):
                        KGr[k] = 3 + iw + c2
                        KGc[k] = 3 + jw + c2
                        k += 1
                for iw in range(3):
                    for jw in range(3):
                        KGr[k] = 3 + iw + c3
                        KGc[k] = 3 + jw + c3
                        k += 1
                for iw in range(3):
                    for jw in range(3):
                        KGr[k] = 3 + iw + c4
                        KGc[k] = 3 + jw + c4
                        k += 1

            # NOTE the slopes of w are taken at the centroid, see the Notes of
            #      update_KG
            _centroid_slopes(x1, x2, x3, x4, y1, y2, y3, y4, Scx, Scy)

            # NOTE two-point Gauss-Legendre quadrature for the stress resultants
            wij = 1.
            points[0] = -0.5773502691896257645092
            points[1] = +0.5773502691896257645092

            for i in range(2):
                xi = points[i]
                for j in range(2):
                    eta = points[j]

                    detJ = -0.125*eta*x1*y2 + 0.125*eta*x1*y3 + 0.125*eta*x2*y1 - 0.125*eta*x2*y4 - 0.125*eta*x3*y1 + 0.125*eta*x3*y4 + 0.125*eta*x4*y2 - 0.125*eta*x4*y3 - 0.125*x1*xi*y3 + 0.125*x1*xi*y4 + 0.125*x1*y2 - 0.125*x1*y4 + 0.125*x2*xi*y3 - 0.125*x2*xi*y4 - 0.125*x2*y1 + 0.125*x2*y3 + 0.125*x3*xi*y1 - 0.125*x3*xi*y2 - 0.125*x3*y2 + 0.125*x3*y4 - 0.125*x4*xi*y1 + 0.125*x4*xi*y2 + 0.125*x4*y1 - 0.125*x4*y3

                    j11 = 2.0*(-xi*y1 + xi*y2 - xi*y3 + xi*y4 + y1 + y2 - y3 - y4)/(eta*x1*y2 - eta*x1*y3 - eta*x2*y1 + eta*x2*y4 + eta*x3*y1 - eta*x3*y4 - eta*x4*y2 + eta*x4*y3 + x1*xi*y3 - x1*xi*y4 - x1*y2 + x1*y4 - x2*xi*y3 + x2*xi*y4 + x2*y1 - x2*y3 - x3*xi*y1 + x3*xi*y2 + x3*y2 - x3*y4 + x4*xi*y1 - x4*xi*y2 - x4*y1 + x4*y3)
                    j12 = 2.0*(eta*y1 - eta*y2 + eta*y3 - eta*y4 - y1 + y2 + y3 - y4)/(eta*x1*y2 - eta*x1*y3 - eta*x2*y1 + eta*x2*y4 + eta*x3*y1 - eta*x3*y4 - eta*x4*y2 + eta*x4*y3 + x1*xi*y3 - x1*xi*y4 - x1*y2 + x1*y4 - x2*xi*y3 + x2*xi*y4 + x2*y1 - x2*y3 - x3*xi*y1 + x3*xi*y2 + x3*y2 - x3*y4 + x4*xi*y1 - x4*xi*y2 - x4*y1 + x4*y3)
                    j21 = 2.0*(x1*xi - x1 - x2*xi - x2 + x3*xi + x3 - x4*xi + x4)/(eta*x1*y2 - eta*x1*y3 - eta*x2*y1 + eta*x2*y4 + eta*x3*y1 - eta*x3*y4 - eta*x4*y2 + eta*x4*y3 + x1*xi*y3 - x1*xi*y4 - x1*y2 + x1*y4 - x2*xi*y3 + x2*xi*y4 + x2*y1 - x2*y3 - x3*xi*y1 + x3*xi*y2 + x3*y2 - x3*y4 + x4*xi*y1 - x4*xi*y2 - x4*y1 + x4*y3)
                    j22 = 2.0*(-eta*x1 + eta*x2 - eta*x3 + eta*x4 + x1 - x2 - x3 + x4)/(eta*x1*y2 - eta*x1*y3 - eta*x2*y1 + eta*x2*y4 + eta*x3*y1 - eta*x3*y4 - eta*x4*y2 + eta*x4*y3 + x1*xi*y3 - x1*xi*y4 - x1*y2 + x1*y4 - x2*xi*y3 + x2*xi*y4 + x2*y1 - x2*y3 - x3*xi*y1 + x3*xi*y2 + x3*y2 - x3*y4 + x4*xi*y1 - x4*xi*y2 - x4*y1 + x4*y3)

                    S1_x = 0.25*j11*(eta - 1) + 0.25*j12*(xi - 1)
                    S2_x = -0.25*eta*j11 + 0.25*j11 - 0.25*j12*xi - 0.25*j12
                    S3_x = 0.25*j11*(eta + 1) + 0.25*j12*(xi + 1)
                    S4_x = -0.25*eta*j11 - 0.25*j11 - 0.25*j12*xi + 0.25*j12

                    S1_y = 0.25*j21*(eta - 1) + 0.25*j22*(xi - 1)
                    S2_y = -0.25*eta*j21 + 0.25*j21 - 0.25*j22*xi - 0.25*j22
                    S3_y = 0.25*j21*(eta + 1) + 0.25*j22*(xi + 1)
                    S4_y = -0.25*eta*j21 - 0.25*j21 - 0.25*j22*xi + 0.25*j22

                    Nxx = ue[0]*(A11*S1_x + A16*S1_y) + ue[10]*(B11*S2_x + B16*S2_y) + ue[12]*(A11*S3_x + A16*S3_y) + ue[13]*(A12*S3_y + A16*S3_x) - ue[15]*(B12*S3_y + B16*S3_x) + ue[16]*(B11*S3_x + B16*S3_y) + ue[18]*(A11*S4_x + A16*S4_y) + ue[19]*(A12*S4_y + A16*S4_x) + ue[1]*(A12*S1_y + A16*S1_x) - ue[21]*(B12*S4_y + B16*S4_x) + ue[22]*(B11*S4_x + B16*S4_y) - ue[3]*(B12*S1_y + B16*S1_x) + ue[4]*(B11*S1_x + B16*S1_y) + ue[6]*(A11*S2_x + A16*S2_y) + ue[7]*(A12*S2_y + A16*S2_x) - ue[9]*(B12*S2_y + B16*S2_x)
                    Nyy = ue[0]*(A12*S1_x + A26*S1_y) + ue[10]*(B12*S2_x + B26*S2_y) + ue[12]*(A12*S3_x + A26*S3_y) + ue[13]*(A22*S3_y + A26*S3_x) - ue[15]*(B22*S3_y + B26*S3_x) + ue[16]*(B12*S3_x + B26*S3_y) + ue[18]*(A12*S4_x + A26*S4_y) + ue[19]*(A22*S4_y + A26*S4_x) + ue[1]*(A22*S1_y + A26*S1_x) - ue[21]*(B22*S4_y + B26*S4_x) + ue[22]*(B12*S4_x + B26*S4_y) - ue[3]*(B22*S1_y + B26*S1_x) + ue[4]*(B12*S1_x + B26*S1_y) + ue[6]*(A12*S2_x + A26*S2_y) + ue[7]*(A22*S2_y + A26*S2_x) - ue[9]*(B22*S2_y + B26*S2_x)
                    Nxy = ue[0]*(A16*S1_x + A66*S1_y) + ue[10]*(B16*S2_x + B66*S2_y) + ue[12]*(A16*S3_x + A66*S3_y) + ue[13]*(A26*S3_y + A66*S3_x) - ue[15]*(B26*S3_y + B66*S3_x) + ue[16]*(B16*S3_x + B66*S3_y) + ue[18]*(A16*S4_x + A66*S4_y) + ue[19]*(A26*S4_y + A66*S4_x) + ue[1]*(A26*S1_y + A66*S1_x) - ue[21]*(B26*S4_y + B66*S4_x) + ue[22]*(B16*S4_x + B66*S4_y) - ue[3]*(B26*S1_y + B66*S1_x) + ue[4]*(B16*S1_x + B66*S1_y) + ue[6]*(A16*S2_x + A66*S2_y) + ue[7]*(A26*S2_y + A66*S2_x) - ue[9]*(B26*S2_y + B66*S2_x)

                    # NOTE slopes of w at the centroid
                    S1_x = Scx[0]
                    S2_x = Scx[1]
                    S3_x = Scx[2]
                    S4_x = Scx[3]
                    S1_y = Scy[0]
                    S2_y = Scy[1]
                    S3_y = Scy[2]
                    S4_y = Scy[3]

                    k = self.init_k_KG
                    KGv[k] += r13**2*(S1_x*(Nxx*S1_x*detJ*wij + Nxy*S1_y*detJ*wij) + S1_y*(Nxy*S1_x*detJ*wij + Nyy*S1_y*detJ*wij))
                    k += 1
                    KGv[k] += r13*r23*(S1_x*(Nxx*S1_x*detJ*wij + Nxy*S1_y*detJ*wij) + S1_y*(Nxy*S1_x*detJ*wij + Nyy*S1_y*detJ*wij))
                    k += 1
                    KGv[k] += r13*r33*(S1_x*(Nxx*S1_x*detJ*wij + Nxy*S1_y*detJ*wij) + S1_y*(Nxy*S1_x*detJ*wij + Nyy*S1_y*detJ*wij))
                    k += 1
                    KGv[k] += r13**2*(S2_x*(Nxx*S1_x*detJ*wij + Nxy*S1_y*detJ*wij) + S2_y*(Nxy*S1_x*detJ*wij + Nyy*S1_y*detJ*wij))
                    k += 1
                    KGv[k] += r13*r23*(S2_x*(Nxx*S1_x*detJ*wij + Nxy*S1_y*detJ*wij) + S2_y*(Nxy*S1_x*detJ*wij + Nyy*S1_y*detJ*wij))
                    k += 1
                    KGv[k] += r13*r33*(S2_x*(Nxx*S1_x*detJ*wij + Nxy*S1_y*detJ*wij) + S2_y*(Nxy*S1_x*detJ*wij + Nyy*S1_y*detJ*wij))
                    k += 1
                    KGv[k] += r13**2*(S3_x*(Nxx*S1_x*detJ*wij + Nxy*S1_y*detJ*wij) + S3_y*(Nxy*S1_x*detJ*wij + Nyy*S1_y*detJ*wij))
                    k += 1
                    KGv[k] += r13*r23*(S3_x*(Nxx*S1_x*detJ*wij + Nxy*S1_y*detJ*wij) + S3_y*(Nxy*S1_x*detJ*wij + Nyy*S1_y*detJ*wij))
                    k += 1
                    KGv[k] += r13*r33*(S3_x*(Nxx*S1_x*detJ*wij + Nxy*S1_y*detJ*wij) + S3_y*(Nxy*S1_x*detJ*wij + Nyy*S1_y*detJ*wij))
                    k += 1
                    KGv[k] += r13**2*(S4_x*(Nxx*S1_x*detJ*wij + Nxy*S1_y*detJ*wij) + S4_y*(Nxy*S1_x*detJ*wij + Nyy*S1_y*detJ*wij))
                    k += 1
                    KGv[k] += r13*r23*(S4_x*(Nxx*S1_x*detJ*wij + Nxy*S1_y*detJ*wij) + S4_y*(Nxy*S1_x*detJ*wij + Nyy*S1_y*detJ*wij))
                    k += 1
                    KGv[k] += r13*r33*(S4_x*(Nxx*S1_x*detJ*wij + Nxy*S1_y*detJ*wij) + S4_y*(Nxy*S1_x*detJ*wij + Nyy*S1_y*detJ*wij))
                    k += 1
                    KGv[k] += r13*r23*(S1_x*(Nxx*S1_x*detJ*wij + Nxy*S1_y*detJ*wij) + S1_y*(Nxy*S1_x*detJ*wij + Nyy*S1_y*detJ*wij))
                    k += 1
                    KGv[k] += r23**2*(S1_x*(Nxx*S1_x*detJ*wij + Nxy*S1_y*detJ*wij) + S1_y*(Nxy*S1_x*detJ*wij + Nyy*S1_y*detJ*wij))
                    k += 1
                    KGv[k] += r23*r33*(S1_x*(Nxx*S1_x*detJ*wij + Nxy*S1_y*detJ*wij) + S1_y*(Nxy*S1_x*detJ*wij + Nyy*S1_y*detJ*wij))
                    k += 1
                    KGv[k] += r13*r23*(S2_x*(Nxx*S1_x*detJ*wij + Nxy*S1_y*detJ*wij) + S2_y*(Nxy*S1_x*detJ*wij + Nyy*S1_y*detJ*wij))
                    k += 1
                    KGv[k] += r23**2*(S2_x*(Nxx*S1_x*detJ*wij + Nxy*S1_y*detJ*wij) + S2_y*(Nxy*S1_x*detJ*wij + Nyy*S1_y*detJ*wij))
                    k += 1
                    KGv[k] += r23*r33*(S2_x*(Nxx*S1_x*detJ*wij + Nxy*S1_y*detJ*wij) + S2_y*(Nxy*S1_x*detJ*wij + Nyy*S1_y*detJ*wij))
                    k += 1
                    KGv[k] += r13*r23*(S3_x*(Nxx*S1_x*detJ*wij + Nxy*S1_y*detJ*wij) + S3_y*(Nxy*S1_x*detJ*wij + Nyy*S1_y*detJ*wij))
                    k += 1
                    KGv[k] += r23**2*(S3_x*(Nxx*S1_x*detJ*wij + Nxy*S1_y*detJ*wij) + S3_y*(Nxy*S1_x*detJ*wij + Nyy*S1_y*detJ*wij))
                    k += 1
                    KGv[k] += r23*r33*(S3_x*(Nxx*S1_x*detJ*wij + Nxy*S1_y*detJ*wij) + S3_y*(Nxy*S1_x*detJ*wij + Nyy*S1_y*detJ*wij))
                    k += 1
                    KGv[k] += r13*r23*(S4_x*(Nxx*S1_x*detJ*wij + Nxy*S1_y*detJ*wij) + S4_y*(Nxy*S1_x*detJ*wij + Nyy*S1_y*detJ*wij))
                    k += 1
                    KGv[k] += r23**2*(S4_x*(Nxx*S1_x*detJ*wij + Nxy*S1_y*detJ*wij) + S4_y*(Nxy*S1_x*detJ*wij + Nyy*S1_y*detJ*wij))
                    k += 1
                    KGv[k] += r23*r33*(S4_x*(Nxx*S1_x*detJ*wij + Nxy*S1_y*detJ*wij) + S4_y*(Nxy*S1_x*detJ*wij + Nyy*S1_y*detJ*wij))
                    k += 1
                    KGv[k] += r13*r33*(S1_x*(Nxx*S1_x*detJ*wij + Nxy*S1_y*detJ*wij) + S1_y*(Nxy*S1_x*detJ*wij + Nyy*S1_y*detJ*wij))
                    k += 1
                    KGv[k] += r23*r33*(S1_x*(Nxx*S1_x*detJ*wij + Nxy*S1_y*detJ*wij) + S1_y*(Nxy*S1_x*detJ*wij + Nyy*S1_y*detJ*wij))
                    k += 1
                    KGv[k] += r33**2*(S1_x*(Nxx*S1_x*detJ*wij + Nxy*S1_y*detJ*wij) + S1_y*(Nxy*S1_x*detJ*wij + Nyy*S1_y*detJ*wij))
                    k += 1
                    KGv[k] += r13*r33*(S2_x*(Nxx*S1_x*detJ*wij + Nxy*S1_y*detJ*wij) + S2_y*(Nxy*S1_x*detJ*wij + Nyy*S1_y*detJ*wij))
                    k += 1
                    KGv[k] += r23*r33*(S2_x*(Nxx*S1_x*detJ*wij + Nxy*S1_y*detJ*wij) + S2_y*(Nxy*S1_x*detJ*wij + Nyy*S1_y*detJ*wij))
                    k += 1
                    KGv[k] += r33**2*(S2_x*(Nxx*S1_x*detJ*wij + Nxy*S1_y*detJ*wij) + S2_y*(Nxy*S1_x*detJ*wij + Nyy*S1_y*detJ*wij))
                    k += 1
                    KGv[k] += r13*r33*(S3_x*(Nxx*S1_x*detJ*wij + Nxy*S1_y*detJ*wij) + S3_y*(Nxy*S1_x*detJ*wij + Nyy*S1_y*detJ*wij))
                    k += 1
                    KGv[k] += r23*r33*(S3_x*(Nxx*S1_x*detJ*wij + Nxy*S1_y*detJ*wij) + S3_y*(Nxy*S1_x*detJ*wij + Nyy*S1_y*detJ*wij))
                    k += 1
                    KGv[k] += r33**2*(S3_x*(Nxx*S1_x*detJ*wij + Nxy*S1_y*detJ*wij) + S3_y*(Nxy*S1_x*detJ*wij + Nyy*S1_y*detJ*wij))
                    k += 1
                    KGv[k] += r13*r33*(S4_x*(Nxx*S1_x*detJ*wij + Nxy*S1_y*detJ*wij) + S4_y*(Nxy*S1_x*detJ*wij + Nyy*S1_y*detJ*wij))
                    k += 1
                    KGv[k] += r23*r33*(S4_x*(Nxx*S1_x*detJ*wij + Nxy*S1_y*detJ*wij) + S4_y*(Nxy*S1_x*detJ*wij + Nyy*S1_y*detJ*wij))
                    k += 1
                    KGv[k] += r33**2*(S4_x*(Nxx*S1_x*detJ*wij + Nxy*S1_y*detJ*wij) + S4_y*(Nxy*S1_x*detJ*wij + Nyy*S1_y*detJ*wij))
                    k += 1
                    KGv[k] += r13**2*(S1_x*(Nxx*S2_x*detJ*wij + Nxy*S2_y*detJ*wij) + S1_y*(Nxy*S2_x*detJ*wij + Nyy*S2_y*detJ*wij))
                    k += 1
                    KGv[k] += r13*r23*(S1_x*(Nxx*S2_x*detJ*wij + Nxy*S2_y*detJ*wij) + S1_y*(Nxy*S2_x*detJ*wij + Nyy*S2_y*detJ*wij))
                    k += 1
                    KGv[k] += r13*r33*(S1_x*(Nxx*S2_x*detJ*wij + Nxy*S2_y*detJ*wij) + S1_y*(Nxy*S2_x*detJ*wij + Nyy*S2_y*detJ*wij))
                    k += 1
                    KGv[k] += r13**2*(S2_x*(Nxx*S2_x*detJ*wij + Nxy*S2_y*detJ*wij) + S2_y*(Nxy*S2_x*detJ*wij + Nyy*S2_y*detJ*wij))
                    k += 1
                    KGv[k] += r13*r23*(S2_x*(Nxx*S2_x*detJ*wij + Nxy*S2_y*detJ*wij) + S2_y*(Nxy*S2_x*detJ*wij + Nyy*S2_y*detJ*wij))
                    k += 1
                    KGv[k] += r13*r33*(S2_x*(Nxx*S2_x*detJ*wij + Nxy*S2_y*detJ*wij) + S2_y*(Nxy*S2_x*detJ*wij + Nyy*S2_y*detJ*wij))
                    k += 1
                    KGv[k] += r13**2*(S3_x*(Nxx*S2_x*detJ*wij + Nxy*S2_y*detJ*wij) + S3_y*(Nxy*S2_x*detJ*wij + Nyy*S2_y*detJ*wij))
                    k += 1
                    KGv[k] += r13*r23*(S3_x*(Nxx*S2_x*detJ*wij + Nxy*S2_y*detJ*wij) + S3_y*(Nxy*S2_x*detJ*wij + Nyy*S2_y*detJ*wij))
                    k += 1
                    KGv[k] += r13*r33*(S3_x*(Nxx*S2_x*detJ*wij + Nxy*S2_y*detJ*wij) + S3_y*(Nxy*S2_x*detJ*wij + Nyy*S2_y*detJ*wij))
                    k += 1
                    KGv[k] += r13**2*(S4_x*(Nxx*S2_x*detJ*wij + Nxy*S2_y*detJ*wij) + S4_y*(Nxy*S2_x*detJ*wij + Nyy*S2_y*detJ*wij))
                    k += 1
                    KGv[k] += r13*r23*(S4_x*(Nxx*S2_x*detJ*wij + Nxy*S2_y*detJ*wij) + S4_y*(Nxy*S2_x*detJ*wij + Nyy*S2_y*detJ*wij))
                    k += 1
                    KGv[k] += r13*r33*(S4_x*(Nxx*S2_x*detJ*wij + Nxy*S2_y*detJ*wij) + S4_y*(Nxy*S2_x*detJ*wij + Nyy*S2_y*detJ*wij))
                    k += 1
                    KGv[k] += r13*r23*(S1_x*(Nxx*S2_x*detJ*wij + Nxy*S2_y*detJ*wij) + S1_y*(Nxy*S2_x*detJ*wij + Nyy*S2_y*detJ*wij))
                    k += 1
                    KGv[k] += r23**2*(S1_x*(Nxx*S2_x*detJ*wij + Nxy*S2_y*detJ*wij) + S1_y*(Nxy*S2_x*detJ*wij + Nyy*S2_y*detJ*wij))
                    k += 1
                    KGv[k] += r23*r33*(S1_x*(Nxx*S2_x*detJ*wij + Nxy*S2_y*detJ*wij) + S1_y*(Nxy*S2_x*detJ*wij + Nyy*S2_y*detJ*wij))
                    k += 1
                    KGv[k] += r13*r23*(S2_x*(Nxx*S2_x*detJ*wij + Nxy*S2_y*detJ*wij) + S2_y*(Nxy*S2_x*detJ*wij + Nyy*S2_y*detJ*wij))
                    k += 1
                    KGv[k] += r23**2*(S2_x*(Nxx*S2_x*detJ*wij + Nxy*S2_y*detJ*wij) + S2_y*(Nxy*S2_x*detJ*wij + Nyy*S2_y*detJ*wij))
                    k += 1
                    KGv[k] += r23*r33*(S2_x*(Nxx*S2_x*detJ*wij + Nxy*S2_y*detJ*wij) + S2_y*(Nxy*S2_x*detJ*wij + Nyy*S2_y*detJ*wij))
                    k += 1
                    KGv[k] += r13*r23*(S3_x*(Nxx*S2_x*detJ*wij + Nxy*S2_y*detJ*wij) + S3_y*(Nxy*S2_x*detJ*wij + Nyy*S2_y*detJ*wij))
                    k += 1
                    KGv[k] += r23**2*(S3_x*(Nxx*S2_x*detJ*wij + Nxy*S2_y*detJ*wij) + S3_y*(Nxy*S2_x*detJ*wij + Nyy*S2_y*detJ*wij))
                    k += 1
                    KGv[k] += r23*r33*(S3_x*(Nxx*S2_x*detJ*wij + Nxy*S2_y*detJ*wij) + S3_y*(Nxy*S2_x*detJ*wij + Nyy*S2_y*detJ*wij))
                    k += 1
                    KGv[k] += r13*r23*(S4_x*(Nxx*S2_x*detJ*wij + Nxy*S2_y*detJ*wij) + S4_y*(Nxy*S2_x*detJ*wij + Nyy*S2_y*detJ*wij))
                    k += 1
                    KGv[k] += r23**2*(S4_x*(Nxx*S2_x*detJ*wij + Nxy*S2_y*detJ*wij) + S4_y*(Nxy*S2_x*detJ*wij + Nyy*S2_y*detJ*wij))
                    k += 1
                    KGv[k] += r23*r33*(S4_x*(Nxx*S2_x*detJ*wij + Nxy*S2_y*detJ*wij) + S4_y*(Nxy*S2_x*detJ*wij + Nyy*S2_y*detJ*wij))
                    k += 1
                    KGv[k] += r13*r33*(S1_x*(Nxx*S2_x*detJ*wij + Nxy*S2_y*detJ*wij) + S1_y*(Nxy*S2_x*detJ*wij + Nyy*S2_y*detJ*wij))
                    k += 1
                    KGv[k] += r23*r33*(S1_x*(Nxx*S2_x*detJ*wij + Nxy*S2_y*detJ*wij) + S1_y*(Nxy*S2_x*detJ*wij + Nyy*S2_y*detJ*wij))
                    k += 1
                    KGv[k] += r33**2*(S1_x*(Nxx*S2_x*detJ*wij + Nxy*S2_y*detJ*wij) + S1_y*(Nxy*S2_x*detJ*wij + Nyy*S2_y*detJ*wij))
                    k += 1
                    KGv[k] += r13*r33*(S2_x*(Nxx*S2_x*detJ*wij + Nxy*S2_y*detJ*wij) + S2_y*(Nxy*S2_x*detJ*wij + Nyy*S2_y*detJ*wij))
                    k += 1
                    KGv[k] += r23*r33*(S2_x*(Nxx*S2_x*detJ*wij + Nxy*S2_y*detJ*wij) + S2_y*(Nxy*S2_x*detJ*wij + Nyy*S2_y*detJ*wij))
                    k += 1
                    KGv[k] += r33**2*(S2_x*(Nxx*S2_x*detJ*wij + Nxy*S2_y*detJ*wij) + S2_y*(Nxy*S2_x*detJ*wij + Nyy*S2_y*detJ*wij))
                    k += 1
                    KGv[k] += r13*r33*(S3_x*(Nxx*S2_x*detJ*wij + Nxy*S2_y*detJ*wij) + S3_y*(Nxy*S2_x*detJ*wij + Nyy*S2_y*detJ*wij))
                    k += 1
                    KGv[k] += r23*r33*(S3_x*(Nxx*S2_x*detJ*wij + Nxy*S2_y*detJ*wij) + S3_y*(Nxy*S2_x*detJ*wij + Nyy*S2_y*detJ*wij))
                    k += 1
                    KGv[k] += r33**2*(S3_x*(Nxx*S2_x*detJ*wij + Nxy*S2_y*detJ*wij) + S3_y*(Nxy*S2_x*detJ*wij + Nyy*S2_y*detJ*wij))
                    k += 1
                    KGv[k] += r13*r33*(S4_x*(Nxx*S2_x*detJ*wij + Nxy*S2_y*detJ*wij) + S4_y*(Nxy*S2_x*detJ*wij + Nyy*S2_y*detJ*wij))
                    k += 1
                    KGv[k] += r23*r33*(S4_x*(Nxx*S2_x*detJ*wij + Nxy*S2_y*detJ*wij) + S4_y*(Nxy*S2_x*detJ*wij + Nyy*S2_y*detJ*wij))
                    k += 1
                    KGv[k] += r33**2*(S4_x*(Nxx*S2_x*detJ*wij + Nxy*S2_y*detJ*wij) + S4_y*(Nxy*S2_x*detJ*wij + Nyy*S2_y*detJ*wij))
                    k += 1
                    KGv[k] += r13**2*(S1_x*(Nxx*S3_x*detJ*wij + Nxy*S3_y*detJ*wij) + S1_y*(Nxy*S3_x*detJ*wij + Nyy*S3_y*detJ*wij))
                    k += 1
                    KGv[k] += r13*r23*(S1_x*(Nxx*S3_x*detJ*wij + Nxy*S3_y*detJ*wij) + S1_y*(Nxy*S3_x*detJ*wij + Nyy*S3_y*detJ*wij))
                    k += 1
                    KGv[k] += r13*r33*(S1_x*(Nxx*S3_x*detJ*wij + Nxy*S3_y*detJ*wij) + S1_y*(Nxy*S3_x*detJ*wij + Nyy*S3_y*detJ*wij))
                    k += 1
                    KGv[k] += r13**2*(S2_x*(Nxx*S3_x*detJ*wij + Nxy*S3_y*detJ*wij) + S2_y*(Nxy*S3_x*detJ*wij + Nyy*S3_y*detJ*wij))
                    k += 1
                    KGv[k] += r13*r23*(S2_x*(Nxx*S3_x*detJ*wij + Nxy*S3_y*detJ*wij) + S2_y*(Nxy*S3_x*detJ*wij + Nyy*S3_y*detJ*wij))
                    k += 1
                    KGv[k] += r13*r33*(S2_x*(Nxx*S3_x*detJ*wij + Nxy*S3_y*detJ*wij) + S2_y*(Nxy*S3_x*detJ*wij + Nyy*S3_y*detJ*wij))
                    k += 1
                    KGv[k] += r13**2*(S3_x*(Nxx*S3_x*detJ*wij + Nxy*S3_y*detJ*wij) + S3_y*(Nxy*S3_x*detJ*wij + Nyy*S3_y*detJ*wij))
                    k += 1
                    KGv[k] += r13*r23*(S3_x*(Nxx*S3_x*detJ*wij + Nxy*S3_y*detJ*wij) + S3_y*(Nxy*S3_x*detJ*wij + Nyy*S3_y*detJ*wij))
                    k += 1
                    KGv[k] += r13*r33*(S3_x*(Nxx*S3_x*detJ*wij + Nxy*S3_y*detJ*wij) + S3_y*(Nxy*S3_x*detJ*wij + Nyy*S3_y*detJ*wij))
                    k += 1
                    KGv[k] += r13**2*(S4_x*(Nxx*S3_x*detJ*wij + Nxy*S3_y*detJ*wij) + S4_y*(Nxy*S3_x*detJ*wij + Nyy*S3_y*detJ*wij))
                    k += 1
                    KGv[k] += r13*r23*(S4_x*(Nxx*S3_x*detJ*wij + Nxy*S3_y*detJ*wij) + S4_y*(Nxy*S3_x*detJ*wij + Nyy*S3_y*detJ*wij))
                    k += 1
                    KGv[k] += r13*r33*(S4_x*(Nxx*S3_x*detJ*wij + Nxy*S3_y*detJ*wij) + S4_y*(Nxy*S3_x*detJ*wij + Nyy*S3_y*detJ*wij))
                    k += 1
                    KGv[k] += r13*r23*(S1_x*(Nxx*S3_x*detJ*wij + Nxy*S3_y*detJ*wij) + S1_y*(Nxy*S3_x*detJ*wij + Nyy*S3_y*detJ*wij))
                    k += 1
                    KGv[k] += r23**2*(S1_x*(Nxx*S3_x*detJ*wij + Nxy*S3_y*detJ*wij) + S1_y*(Nxy*S3_x*detJ*wij + Nyy*S3_y*detJ*wij))
                    k += 1
                    KGv[k] += r23*r33*(S1_x*(Nxx*S3_x*detJ*wij + Nxy*S3_y*detJ*wij) + S1_y*(Nxy*S3_x*detJ*wij + Nyy*S3_y*detJ*wij))
                    k += 1
                    KGv[k] += r13*r23*(S2_x*(Nxx*S3_x*detJ*wij + Nxy*S3_y*detJ*wij) + S2_y*(Nxy*S3_x*detJ*wij + Nyy*S3_y*detJ*wij))
                    k += 1
                    KGv[k] += r23**2*(S2_x*(Nxx*S3_x*detJ*wij + Nxy*S3_y*detJ*wij) + S2_y*(Nxy*S3_x*detJ*wij + Nyy*S3_y*detJ*wij))
                    k += 1
                    KGv[k] += r23*r33*(S2_x*(Nxx*S3_x*detJ*wij + Nxy*S3_y*detJ*wij) + S2_y*(Nxy*S3_x*detJ*wij + Nyy*S3_y*detJ*wij))
                    k += 1
                    KGv[k] += r13*r23*(S3_x*(Nxx*S3_x*detJ*wij + Nxy*S3_y*detJ*wij) + S3_y*(Nxy*S3_x*detJ*wij + Nyy*S3_y*detJ*wij))
                    k += 1
                    KGv[k] += r23**2*(S3_x*(Nxx*S3_x*detJ*wij + Nxy*S3_y*detJ*wij) + S3_y*(Nxy*S3_x*detJ*wij + Nyy*S3_y*detJ*wij))
                    k += 1
                    KGv[k] += r23*r33*(S3_x*(Nxx*S3_x*detJ*wij + Nxy*S3_y*detJ*wij) + S3_y*(Nxy*S3_x*detJ*wij + Nyy*S3_y*detJ*wij))
                    k += 1
                    KGv[k] += r13*r23*(S4_x*(Nxx*S3_x*detJ*wij + Nxy*S3_y*detJ*wij) + S4_y*(Nxy*S3_x*detJ*wij + Nyy*S3_y*detJ*wij))
                    k += 1
                    KGv[k] += r23**2*(S4_x*(Nxx*S3_x*detJ*wij + Nxy*S3_y*detJ*wij) + S4_y*(Nxy*S3_x*detJ*wij + Nyy*S3_y*detJ*wij))
                    k += 1
                    KGv[k] += r23*r33*(S4_x*(Nxx*S3_x*detJ*wij + Nxy*S3_y*detJ*wij) + S4_y*(Nxy*S3_x*detJ*wij + Nyy*S3_y*detJ*wij))
                    k += 1
                    KGv[k] += r13*r33*(S1_x*(Nxx*S3_x*detJ*wij + Nxy*S3_y*detJ*wij) + S1_y*(Nxy*S3_x*detJ*wij + Nyy*S3_y*detJ*wij))
                    k += 1
                    KGv[k] += r23*r33*(S1_x*(Nxx*S3_x*detJ*wij + Nxy*S3_y*detJ*wij) + S1_y*(Nxy*S3_x*detJ*wij + Nyy*S3_y*detJ*wij))
                    k += 1
                    KGv[k] += r33**2*(S1_x*(Nxx*S3_x*detJ*wij + Nxy*S3_y*detJ*wij) + S1_y*(Nxy*S3_x*detJ*wij + Nyy*S3_y*detJ*wij))
                    k += 1
                    KGv[k] += r13*r33*(S2_x*(Nxx*S3_x*detJ*wij + Nxy*S3_y*detJ*wij) + S2_y*(Nxy*S3_x*detJ*wij + Nyy*S3_y*detJ*wij))
                    k += 1
                    KGv[k] += r23*r33*(S2_x*(Nxx*S3_x*detJ*wij + Nxy*S3_y*detJ*wij) + S2_y*(Nxy*S3_x*detJ*wij + Nyy*S3_y*detJ*wij))
                    k += 1
                    KGv[k] += r33**2*(S2_x*(Nxx*S3_x*detJ*wij + Nxy*S3_y*detJ*wij) + S2_y*(Nxy*S3_x*detJ*wij + Nyy*S3_y*detJ*wij))
                    k += 1
                    KGv[k] += r13*r33*(S3_x*(Nxx*S3_x*detJ*wij + Nxy*S3_y*detJ*wij) + S3_y*(Nxy*S3_x*detJ*wij + Nyy*S3_y*detJ*wij))
                    k += 1
                    KGv[k] += r23*r33*(S3_x*(Nxx*S3_x*detJ*wij + Nxy*S3_y*detJ*wij) + S3_y*(Nxy*S3_x*detJ*wij + Nyy*S3_y*detJ*wij))
                    k += 1
                    KGv[k] += r33**2*(S3_x*(Nxx*S3_x*detJ*wij + Nxy*S3_y*detJ*wij) + S3_y*(Nxy*S3_x*detJ*wij + Nyy*S3_y*detJ*wij))
                    k += 1
                    KGv[k] += r13*r33*(S4_x*(Nxx*S3_x*detJ*wij + Nxy*S3_y*detJ*wij) + S4_y*(Nxy*S3_x*detJ*wij + Nyy*S3_y*detJ*wij))
                    k += 1
                    KGv[k] += r23*r33*(S4_x*(Nxx*S3_x*detJ*wij + Nxy*S3_y*detJ*wij) + S4_y*(Nxy*S3_x*detJ*wij + Nyy*S3_y*detJ*wij))
                    k += 1
                    KGv[k] += r33**2*(S4_x*(Nxx*S3_x*detJ*wij + Nxy*S3_y*detJ*wij) + S4_y*(Nxy*S3_x*detJ*wij + Nyy*S3_y*detJ*wij))
                    k += 1
                    KGv[k] += r13**2*(S1_x*(Nxx*S4_x*detJ*wij + Nxy*S4_y*detJ*wij) + S1_y*(Nxy*S4_x*detJ*wij + Nyy*S4_y*detJ*wij))
                    k += 1
                    KGv[k] += r13*r23*(S1_x*(Nxx*S4_x*detJ*wij + Nxy*S4_y*detJ*wij) + S1_y*(Nxy*S4_x*detJ*wij + Nyy*S4_y*detJ*wij))
                    k += 1
                    KGv[k] += r13*r33*(S1_x*(Nxx*S4_x*detJ*wij + Nxy*S4_y*detJ*wij) + S1_y*(Nxy*S4_x*detJ*wij + Nyy*S4_y*detJ*wij))
                    k += 1
                    KGv[k] += r13**2*(S2_x*(Nxx*S4_x*detJ*wij + Nxy*S4_y*detJ*wij) + S2_y*(Nxy*S4_x*detJ*wij + Nyy*S4_y*detJ*wij))
                    k += 1
                    KGv[k] += r13*r23*(S2_x*(Nxx*S4_x*detJ*wij + Nxy*S4_y*detJ*wij) + S2_y*(Nxy*S4_x*detJ*wij + Nyy*S4_y*detJ*wij))
                    k += 1
                    KGv[k] += r13*r33*(S2_x*(Nxx*S4_x*detJ*wij + Nxy*S4_y*detJ*wij) + S2_y*(Nxy*S4_x*detJ*wij + Nyy*S4_y*detJ*wij))
                    k += 1
                    KGv[k] += r13**2*(S3_x*(Nxx*S4_x*detJ*wij + Nxy*S4_y*detJ*wij) + S3_y*(Nxy*S4_x*detJ*wij + Nyy*S4_y*detJ*wij))
                    k += 1
                    KGv[k] += r13*r23*(S3_x*(Nxx*S4_x*detJ*wij + Nxy*S4_y*detJ*wij) + S3_y*(Nxy*S4_x*detJ*wij + Nyy*S4_y*detJ*wij))
                    k += 1
                    KGv[k] += r13*r33*(S3_x*(Nxx*S4_x*detJ*wij + Nxy*S4_y*detJ*wij) + S3_y*(Nxy*S4_x*detJ*wij + Nyy*S4_y*detJ*wij))
                    k += 1
                    KGv[k] += r13**2*(S4_x*(Nxx*S4_x*detJ*wij + Nxy*S4_y*detJ*wij) + S4_y*(Nxy*S4_x*detJ*wij + Nyy*S4_y*detJ*wij))
                    k += 1
                    KGv[k] += r13*r23*(S4_x*(Nxx*S4_x*detJ*wij + Nxy*S4_y*detJ*wij) + S4_y*(Nxy*S4_x*detJ*wij + Nyy*S4_y*detJ*wij))
                    k += 1
                    KGv[k] += r13*r33*(S4_x*(Nxx*S4_x*detJ*wij + Nxy*S4_y*detJ*wij) + S4_y*(Nxy*S4_x*detJ*wij + Nyy*S4_y*detJ*wij))
                    k += 1
                    KGv[k] += r13*r23*(S1_x*(Nxx*S4_x*detJ*wij + Nxy*S4_y*detJ*wij) + S1_y*(Nxy*S4_x*detJ*wij + Nyy*S4_y*detJ*wij))
                    k += 1
                    KGv[k] += r23**2*(S1_x*(Nxx*S4_x*detJ*wij + Nxy*S4_y*detJ*wij) + S1_y*(Nxy*S4_x*detJ*wij + Nyy*S4_y*detJ*wij))
                    k += 1
                    KGv[k] += r23*r33*(S1_x*(Nxx*S4_x*detJ*wij + Nxy*S4_y*detJ*wij) + S1_y*(Nxy*S4_x*detJ*wij + Nyy*S4_y*detJ*wij))
                    k += 1
                    KGv[k] += r13*r23*(S2_x*(Nxx*S4_x*detJ*wij + Nxy*S4_y*detJ*wij) + S2_y*(Nxy*S4_x*detJ*wij + Nyy*S4_y*detJ*wij))
                    k += 1
                    KGv[k] += r23**2*(S2_x*(Nxx*S4_x*detJ*wij + Nxy*S4_y*detJ*wij) + S2_y*(Nxy*S4_x*detJ*wij + Nyy*S4_y*detJ*wij))
                    k += 1
                    KGv[k] += r23*r33*(S2_x*(Nxx*S4_x*detJ*wij + Nxy*S4_y*detJ*wij) + S2_y*(Nxy*S4_x*detJ*wij + Nyy*S4_y*detJ*wij))
                    k += 1
                    KGv[k] += r13*r23*(S3_x*(Nxx*S4_x*detJ*wij + Nxy*S4_y*detJ*wij) + S3_y*(Nxy*S4_x*detJ*wij + Nyy*S4_y*detJ*wij))
                    k += 1
                    KGv[k] += r23**2*(S3_x*(Nxx*S4_x*detJ*wij + Nxy*S4_y*detJ*wij) + S3_y*(Nxy*S4_x*detJ*wij + Nyy*S4_y*detJ*wij))
                    k += 1
                    KGv[k] += r23*r33*(S3_x*(Nxx*S4_x*detJ*wij + Nxy*S4_y*detJ*wij) + S3_y*(Nxy*S4_x*detJ*wij + Nyy*S4_y*detJ*wij))
                    k += 1
                    KGv[k] += r13*r23*(S4_x*(Nxx*S4_x*detJ*wij + Nxy*S4_y*detJ*wij) + S4_y*(Nxy*S4_x*detJ*wij + Nyy*S4_y*detJ*wij))
                    k += 1
                    KGv[k] += r23**2*(S4_x*(Nxx*S4_x*detJ*wij + Nxy*S4_y*detJ*wij) + S4_y*(Nxy*S4_x*detJ*wij + Nyy*S4_y*detJ*wij))
                    k += 1
                    KGv[k] += r23*r33*(S4_x*(Nxx*S4_x*detJ*wij + Nxy*S4_y*detJ*wij) + S4_y*(Nxy*S4_x*detJ*wij + Nyy*S4_y*detJ*wij))
                    k += 1
                    KGv[k] += r13*r33*(S1_x*(Nxx*S4_x*detJ*wij + Nxy*S4_y*detJ*wij) + S1_y*(Nxy*S4_x*detJ*wij + Nyy*S4_y*detJ*wij))
                    k += 1
                    KGv[k] += r23*r33*(S1_x*(Nxx*S4_x*detJ*wij + Nxy*S4_y*detJ*wij) + S1_y*(Nxy*S4_x*detJ*wij + Nyy*S4_y*detJ*wij))
                    k += 1
                    KGv[k] += r33**2*(S1_x*(Nxx*S4_x*detJ*wij + Nxy*S4_y*detJ*wij) + S1_y*(Nxy*S4_x*detJ*wij + Nyy*S4_y*detJ*wij))
                    k += 1
                    KGv[k] += r13*r33*(S2_x*(Nxx*S4_x*detJ*wij + Nxy*S4_y*detJ*wij) + S2_y*(Nxy*S4_x*detJ*wij + Nyy*S4_y*detJ*wij))
                    k += 1
                    KGv[k] += r23*r33*(S2_x*(Nxx*S4_x*detJ*wij + Nxy*S4_y*detJ*wij) + S2_y*(Nxy*S4_x*detJ*wij + Nyy*S4_y*detJ*wij))
                    k += 1
                    KGv[k] += r33**2*(S2_x*(Nxx*S4_x*detJ*wij + Nxy*S4_y*detJ*wij) + S2_y*(Nxy*S4_x*detJ*wij + Nyy*S4_y*detJ*wij))
                    k += 1
                    KGv[k] += r13*r33*(S3_x*(Nxx*S4_x*detJ*wij + Nxy*S4_y*detJ*wij) + S3_y*(Nxy*S4_x*detJ*wij + Nyy*S4_y*detJ*wij))
                    k += 1
                    KGv[k] += r23*r33*(S3_x*(Nxx*S4_x*detJ*wij + Nxy*S4_y*detJ*wij) + S3_y*(Nxy*S4_x*detJ*wij + Nyy*S4_y*detJ*wij))
                    k += 1
                    KGv[k] += r33**2*(S3_x*(Nxx*S4_x*detJ*wij + Nxy*S4_y*detJ*wij) + S3_y*(Nxy*S4_x*detJ*wij + Nyy*S4_y*detJ*wij))
                    k += 1
                    KGv[k] += r13*r33*(S4_x*(Nxx*S4_x*detJ*wij + Nxy*S4_y*detJ*wij) + S4_y*(Nxy*S4_x*detJ*wij + Nyy*S4_y*detJ*wij))
                    k += 1
                    KGv[k] += r23*r33*(S4_x*(Nxx*S4_x*detJ*wij + Nxy*S4_y*detJ*wij) + S4_y*(Nxy*S4_x*detJ*wij + Nyy*S4_y*detJ*wij))
                    k += 1
                    KGv[k] += r33**2*(S4_x*(Nxx*S4_x*detJ*wij + Nxy*S4_y*detJ*wij) + S4_y*(Nxy*S4_x*detJ*wij + Nyy*S4_y*detJ*wij))

            if warped:
                # NOTE eccentricity geometric stiffness of the rigid offsets
                Rw[0] = self.r11
                Rw[1] = self.r12
                Rw[2] = self.r13
                Rw[3] = self.r21
                Rw[4] = self.r22
                Rw[5] = self.r23
                Rw[6] = self.r31
                Rw[7] = self.r32
                Rw[8] = self.r33
                _warp_KE(zw, fw, Rw, KEw)
                for iw in range(36):
                    KGv[self.init_k_KG + 144 + iw] += KEw[iw]


    cpdef void update_KG_given_stress(Quad4R self,
                                      double Nxx, double Nyy, double Nxy,
                                      long [::1] KGr,
                                      long [::1] KGc,
                                      double [::1] KGv,
                                      int update_KGv_only=0,
                                      ):
        r"""Update sparse vectors for geometric stiffness matrix KG

        .. note:: A constant stress state is assumed within the element,
                  according to the given values of `N_{xx}, N_{yy}, N_{xy}`.

        The slopes of `w` are taken at the centroid, see the Notes of
        :meth:`.update_KG`.

        Before this function is called, the probe :class:`.Quad4RProbe`
        attribute of the :class:`.Quad4R` object must be updated using
        :meth:`.update_probe_xe` with the node coordinates.

        Only the translational degrees-of-freedom appear. The values are added
        to ``KGv`` at the positions ``init_k_KG`` to ``init_k_KG +
        KG_SPARSE_SIZE - 1``. With ``update_KGv_only=0`` the global row and
        column indices are written to ``KGr`` and ``KGc`` at the same
        positions. :meth:`.update_rotation_matrix` must also have been called
        before.

        Parameters
        ----------
        Nxx, Nyy, Nxy : double
            Membrane stress resultants in the element coordinate system,
            force per unit length.
        KGr : np.array
           Array to store row positions of sparse values
           1-D array of integers (``INT``) of size at least ``init_k_KG +
           KG_SPARSE_SIZE``.
        KGc : np.array
           Array to store column positions of sparse values
           Same type and size as ``KGr``.
        KGv : np.array
            Array to store sparse values
            1-D array of ``DOUBLE``, same size as ``KGr``. The values are
            added to the existing ones, with dimension of force per unit
            length.
        update_KGv_only : int, optional
            The default ``0`` means that the row and column indices ``KGr``
            and ``KGc`` are also updated. Any other value will only update the
            values ``KGv``.

        Notes
        -----
        The matrix is the one of :meth:`.update_KG`, with the given constant
        stress resultants ``Nxx, Nyy, Nxy`` instead of the ones computed from
        the displacements of the probe. With constant stress resultants and
        the slopes at the centroid, it is the element area times the
        integrand at the centroid.

        Warped elements, ``warp_offsets = 1``, see the Notes of
        :class:`.Quad4R`: the stress resultants are those of the displacements
        `\boldsymbol{J} \boldsymbol{u}_e` of the flat element, and the matrix
        above, which has no entries of the in-plane translations, is not
        changed by `\boldsymbol{J}^\top \boldsymbol{K}_G \boldsymbol{J}`. The
        rigid offsets add on the rotations of each node the eccentricity
        geometric stiffness, which Felippa and Haugen (2005), Section 5.1,
        p. 2304, name without giving it (see the references of
        :class:`pyfe3d.corotational.CorotationalShell`),

        .. math::

            \boldsymbol{K}_{E,a} = (\boldsymbol{n}_a \cdot \boldsymbol{r}_a)
            \boldsymbol{I} - \frac{1}{2} \left( \boldsymbol{n}_a
            \boldsymbol{r}_a^\top + \boldsymbol{r}_a \boldsymbol{n}_a^\top
            \right)

        the second derivative, at zero rotation, of the work of the nodal
        forces `\boldsymbol{n}_a` of the flat element on the displacement
        `-(\boldsymbol{R}_a - \boldsymbol{I}) \boldsymbol{r}_a` of its node,
        `\boldsymbol{R}_a` the rotation matrix of the node (derived here),
        equal to ``dLambdaT_m(0, n x r) - Spin(n) Spin(r)`` of
        :mod:`pyfe3d.corotational` and linear in the pre-buckling state. The
        forces `\boldsymbol{n}_a` are those of the given stress resultants, the
        area times `N_{xx} S_{a,x} + N_{xy} S_{a,y}` and `N_{xy} S_{a,x} +
        N_{yy} S_{a,y}` with the slopes of the shape functions at the centroid,
        which is their exact integral. The 36 entries of `\boldsymbol{K}_E`
        follow the 144 of the translations, ``(r_x, r_y, r_z)`` of node 1 to
        node 4 row by row, and are zero for a flat element and with
        ``warp_offsets = 0``.

        """
        cdef int warped, iw, jw, a
        cdef double zw[4]
        cdef double fw[24]
        cdef double KEw[36]
        cdef double Rw[9]
        cdef int c1, c2, c3, c4, i, j, k
        cdef double x1, x2, x3, x4
        cdef double y1, y2, y3, y4
        cdef double wij, detJ, xi, eta
        cdef double points[2]
        cdef double r13, r23, r33
        cdef double j11, j12, j21, j22
        cdef double S1_x, S2_x, S3_x, S4_x, S1_y, S2_y, S3_y, S4_y
        cdef double Scx[4]
        cdef double Scy[4]

        with nogil:
            # local to global transformation
            r13 = self.r13
            r23 = self.r23
            r33 = self.r33

            # NOTE ignoring z in local coordinates
            x1 = self.probe.xe[0]
            y1 = self.probe.xe[1]
            # z1 = self.probe.xe[2]
            x2 = self.probe.xe[3]
            y2 = self.probe.xe[4]
            # z2 = self.probe.xe[5]
            x3 = self.probe.xe[6]
            y3 = self.probe.xe[7]
            # z3 = self.probe.xe[8]
            x4 = self.probe.xe[9]
            y4 = self.probe.xe[10]
            # z4 = self.probe.xe[11]

            if update_KGv_only == 0:
                # positions of nodes 1,2,3,4 in the global matrix
                c1 = self.c1
                c2 = self.c2
                c3 = self.c3
                c4 = self.c4

                k = self.init_k_KG
                KGr[k] = 0+c1
                KGc[k] = 0+c1
                k += 1
                KGr[k] = 0+c1
                KGc[k] = 1+c1
                k += 1
                KGr[k] = 0+c1
                KGc[k] = 2+c1
                k += 1
                KGr[k] = 0+c1
                KGc[k] = 0+c2
                k += 1
                KGr[k] = 0+c1
                KGc[k] = 1+c2
                k += 1
                KGr[k] = 0+c1
                KGc[k] = 2+c2
                k += 1
                KGr[k] = 0+c1
                KGc[k] = 0+c3
                k += 1
                KGr[k] = 0+c1
                KGc[k] = 1+c3
                k += 1
                KGr[k] = 0+c1
                KGc[k] = 2+c3
                k += 1
                KGr[k] = 0+c1
                KGc[k] = 0+c4
                k += 1
                KGr[k] = 0+c1
                KGc[k] = 1+c4
                k += 1
                KGr[k] = 0+c1
                KGc[k] = 2+c4
                k += 1
                KGr[k] = 1+c1
                KGc[k] = 0+c1
                k += 1
                KGr[k] = 1+c1
                KGc[k] = 1+c1
                k += 1
                KGr[k] = 1+c1
                KGc[k] = 2+c1
                k += 1
                KGr[k] = 1+c1
                KGc[k] = 0+c2
                k += 1
                KGr[k] = 1+c1
                KGc[k] = 1+c2
                k += 1
                KGr[k] = 1+c1
                KGc[k] = 2+c2
                k += 1
                KGr[k] = 1+c1
                KGc[k] = 0+c3
                k += 1
                KGr[k] = 1+c1
                KGc[k] = 1+c3
                k += 1
                KGr[k] = 1+c1
                KGc[k] = 2+c3
                k += 1
                KGr[k] = 1+c1
                KGc[k] = 0+c4
                k += 1
                KGr[k] = 1+c1
                KGc[k] = 1+c4
                k += 1
                KGr[k] = 1+c1
                KGc[k] = 2+c4
                k += 1
                KGr[k] = 2+c1
                KGc[k] = 0+c1
                k += 1
                KGr[k] = 2+c1
                KGc[k] = 1+c1
                k += 1
                KGr[k] = 2+c1
                KGc[k] = 2+c1
                k += 1
                KGr[k] = 2+c1
                KGc[k] = 0+c2
                k += 1
                KGr[k] = 2+c1
                KGc[k] = 1+c2
                k += 1
                KGr[k] = 2+c1
                KGc[k] = 2+c2
                k += 1
                KGr[k] = 2+c1
                KGc[k] = 0+c3
                k += 1
                KGr[k] = 2+c1
                KGc[k] = 1+c3
                k += 1
                KGr[k] = 2+c1
                KGc[k] = 2+c3
                k += 1
                KGr[k] = 2+c1
                KGc[k] = 0+c4
                k += 1
                KGr[k] = 2+c1
                KGc[k] = 1+c4
                k += 1
                KGr[k] = 2+c1
                KGc[k] = 2+c4
                k += 1
                KGr[k] = 0+c2
                KGc[k] = 0+c1
                k += 1
                KGr[k] = 0+c2
                KGc[k] = 1+c1
                k += 1
                KGr[k] = 0+c2
                KGc[k] = 2+c1
                k += 1
                KGr[k] = 0+c2
                KGc[k] = 0+c2
                k += 1
                KGr[k] = 0+c2
                KGc[k] = 1+c2
                k += 1
                KGr[k] = 0+c2
                KGc[k] = 2+c2
                k += 1
                KGr[k] = 0+c2
                KGc[k] = 0+c3
                k += 1
                KGr[k] = 0+c2
                KGc[k] = 1+c3
                k += 1
                KGr[k] = 0+c2
                KGc[k] = 2+c3
                k += 1
                KGr[k] = 0+c2
                KGc[k] = 0+c4
                k += 1
                KGr[k] = 0+c2
                KGc[k] = 1+c4
                k += 1
                KGr[k] = 0+c2
                KGc[k] = 2+c4
                k += 1
                KGr[k] = 1+c2
                KGc[k] = 0+c1
                k += 1
                KGr[k] = 1+c2
                KGc[k] = 1+c1
                k += 1
                KGr[k] = 1+c2
                KGc[k] = 2+c1
                k += 1
                KGr[k] = 1+c2
                KGc[k] = 0+c2
                k += 1
                KGr[k] = 1+c2
                KGc[k] = 1+c2
                k += 1
                KGr[k] = 1+c2
                KGc[k] = 2+c2
                k += 1
                KGr[k] = 1+c2
                KGc[k] = 0+c3
                k += 1
                KGr[k] = 1+c2
                KGc[k] = 1+c3
                k += 1
                KGr[k] = 1+c2
                KGc[k] = 2+c3
                k += 1
                KGr[k] = 1+c2
                KGc[k] = 0+c4
                k += 1
                KGr[k] = 1+c2
                KGc[k] = 1+c4
                k += 1
                KGr[k] = 1+c2
                KGc[k] = 2+c4
                k += 1
                KGr[k] = 2+c2
                KGc[k] = 0+c1
                k += 1
                KGr[k] = 2+c2
                KGc[k] = 1+c1
                k += 1
                KGr[k] = 2+c2
                KGc[k] = 2+c1
                k += 1
                KGr[k] = 2+c2
                KGc[k] = 0+c2
                k += 1
                KGr[k] = 2+c2
                KGc[k] = 1+c2
                k += 1
                KGr[k] = 2+c2
                KGc[k] = 2+c2
                k += 1
                KGr[k] = 2+c2
                KGc[k] = 0+c3
                k += 1
                KGr[k] = 2+c2
                KGc[k] = 1+c3
                k += 1
                KGr[k] = 2+c2
                KGc[k] = 2+c3
                k += 1
                KGr[k] = 2+c2
                KGc[k] = 0+c4
                k += 1
                KGr[k] = 2+c2
                KGc[k] = 1+c4
                k += 1
                KGr[k] = 2+c2
                KGc[k] = 2+c4
                k += 1
                KGr[k] = 0+c3
                KGc[k] = 0+c1
                k += 1
                KGr[k] = 0+c3
                KGc[k] = 1+c1
                k += 1
                KGr[k] = 0+c3
                KGc[k] = 2+c1
                k += 1
                KGr[k] = 0+c3
                KGc[k] = 0+c2
                k += 1
                KGr[k] = 0+c3
                KGc[k] = 1+c2
                k += 1
                KGr[k] = 0+c3
                KGc[k] = 2+c2
                k += 1
                KGr[k] = 0+c3
                KGc[k] = 0+c3
                k += 1
                KGr[k] = 0+c3
                KGc[k] = 1+c3
                k += 1
                KGr[k] = 0+c3
                KGc[k] = 2+c3
                k += 1
                KGr[k] = 0+c3
                KGc[k] = 0+c4
                k += 1
                KGr[k] = 0+c3
                KGc[k] = 1+c4
                k += 1
                KGr[k] = 0+c3
                KGc[k] = 2+c4
                k += 1
                KGr[k] = 1+c3
                KGc[k] = 0+c1
                k += 1
                KGr[k] = 1+c3
                KGc[k] = 1+c1
                k += 1
                KGr[k] = 1+c3
                KGc[k] = 2+c1
                k += 1
                KGr[k] = 1+c3
                KGc[k] = 0+c2
                k += 1
                KGr[k] = 1+c3
                KGc[k] = 1+c2
                k += 1
                KGr[k] = 1+c3
                KGc[k] = 2+c2
                k += 1
                KGr[k] = 1+c3
                KGc[k] = 0+c3
                k += 1
                KGr[k] = 1+c3
                KGc[k] = 1+c3
                k += 1
                KGr[k] = 1+c3
                KGc[k] = 2+c3
                k += 1
                KGr[k] = 1+c3
                KGc[k] = 0+c4
                k += 1
                KGr[k] = 1+c3
                KGc[k] = 1+c4
                k += 1
                KGr[k] = 1+c3
                KGc[k] = 2+c4
                k += 1
                KGr[k] = 2+c3
                KGc[k] = 0+c1
                k += 1
                KGr[k] = 2+c3
                KGc[k] = 1+c1
                k += 1
                KGr[k] = 2+c3
                KGc[k] = 2+c1
                k += 1
                KGr[k] = 2+c3
                KGc[k] = 0+c2
                k += 1
                KGr[k] = 2+c3
                KGc[k] = 1+c2
                k += 1
                KGr[k] = 2+c3
                KGc[k] = 2+c2
                k += 1
                KGr[k] = 2+c3
                KGc[k] = 0+c3
                k += 1
                KGr[k] = 2+c3
                KGc[k] = 1+c3
                k += 1
                KGr[k] = 2+c3
                KGc[k] = 2+c3
                k += 1
                KGr[k] = 2+c3
                KGc[k] = 0+c4
                k += 1
                KGr[k] = 2+c3
                KGc[k] = 1+c4
                k += 1
                KGr[k] = 2+c3
                KGc[k] = 2+c4
                k += 1
                KGr[k] = 0+c4
                KGc[k] = 0+c1
                k += 1
                KGr[k] = 0+c4
                KGc[k] = 1+c1
                k += 1
                KGr[k] = 0+c4
                KGc[k] = 2+c1
                k += 1
                KGr[k] = 0+c4
                KGc[k] = 0+c2
                k += 1
                KGr[k] = 0+c4
                KGc[k] = 1+c2
                k += 1
                KGr[k] = 0+c4
                KGc[k] = 2+c2
                k += 1
                KGr[k] = 0+c4
                KGc[k] = 0+c3
                k += 1
                KGr[k] = 0+c4
                KGc[k] = 1+c3
                k += 1
                KGr[k] = 0+c4
                KGc[k] = 2+c3
                k += 1
                KGr[k] = 0+c4
                KGc[k] = 0+c4
                k += 1
                KGr[k] = 0+c4
                KGc[k] = 1+c4
                k += 1
                KGr[k] = 0+c4
                KGc[k] = 2+c4
                k += 1
                KGr[k] = 1+c4
                KGc[k] = 0+c1
                k += 1
                KGr[k] = 1+c4
                KGc[k] = 1+c1
                k += 1
                KGr[k] = 1+c4
                KGc[k] = 2+c1
                k += 1
                KGr[k] = 1+c4
                KGc[k] = 0+c2
                k += 1
                KGr[k] = 1+c4
                KGc[k] = 1+c2
                k += 1
                KGr[k] = 1+c4
                KGc[k] = 2+c2
                k += 1
                KGr[k] = 1+c4
                KGc[k] = 0+c3
                k += 1
                KGr[k] = 1+c4
                KGc[k] = 1+c3
                k += 1
                KGr[k] = 1+c4
                KGc[k] = 2+c3
                k += 1
                KGr[k] = 1+c4
                KGc[k] = 0+c4
                k += 1
                KGr[k] = 1+c4
                KGc[k] = 1+c4
                k += 1
                KGr[k] = 1+c4
                KGc[k] = 2+c4
                k += 1
                KGr[k] = 2+c4
                KGc[k] = 0+c1
                k += 1
                KGr[k] = 2+c4
                KGc[k] = 1+c1
                k += 1
                KGr[k] = 2+c4
                KGc[k] = 2+c1
                k += 1
                KGr[k] = 2+c4
                KGc[k] = 0+c2
                k += 1
                KGr[k] = 2+c4
                KGc[k] = 1+c2
                k += 1
                KGr[k] = 2+c4
                KGc[k] = 2+c2
                k += 1
                KGr[k] = 2+c4
                KGc[k] = 0+c3
                k += 1
                KGr[k] = 2+c4
                KGc[k] = 1+c3
                k += 1
                KGr[k] = 2+c4
                KGc[k] = 2+c3
                k += 1
                KGr[k] = 2+c4
                KGc[k] = 0+c4
                k += 1
                KGr[k] = 2+c4
                KGc[k] = 1+c4
                k += 1
                KGr[k] = 2+c4
                KGc[k] = 2+c4
                # NOTE rotations of each node, eccentricity geometric stiffness
                #      of the rigid offsets, see the Notes
                k = self.init_k_KG + 144
                for iw in range(3):
                    for jw in range(3):
                        KGr[k] = 3 + iw + c1
                        KGc[k] = 3 + jw + c1
                        k += 1
                for iw in range(3):
                    for jw in range(3):
                        KGr[k] = 3 + iw + c2
                        KGc[k] = 3 + jw + c2
                        k += 1
                for iw in range(3):
                    for jw in range(3):
                        KGr[k] = 3 + iw + c3
                        KGc[k] = 3 + jw + c3
                        k += 1
                for iw in range(3):
                    for jw in range(3):
                        KGr[k] = 3 + iw + c4
                        KGc[k] = 3 + jw + c4
                        k += 1

            # NOTE the slopes of w are taken at the centroid, see the Notes of
            #      update_KG
            _centroid_slopes(x1, x2, x3, x4, y1, y2, y3, y4, Scx, Scy)

            # NOTE two-point Gauss-Legendre quadrature for the stress resultants
            wij = 1.
            points[0] = -0.5773502691896257645092
            points[1] = +0.5773502691896257645092

            for i in range(2):
                xi = points[i]
                for j in range(2):
                    eta = points[j]

                    detJ = -0.125*eta*x1*y2 + 0.125*eta*x1*y3 + 0.125*eta*x2*y1 - 0.125*eta*x2*y4 - 0.125*eta*x3*y1 + 0.125*eta*x3*y4 + 0.125*eta*x4*y2 - 0.125*eta*x4*y3 - 0.125*x1*xi*y3 + 0.125*x1*xi*y4 + 0.125*x1*y2 - 0.125*x1*y4 + 0.125*x2*xi*y3 - 0.125*x2*xi*y4 - 0.125*x2*y1 + 0.125*x2*y3 + 0.125*x3*xi*y1 - 0.125*x3*xi*y2 - 0.125*x3*y2 + 0.125*x3*y4 - 0.125*x4*xi*y1 + 0.125*x4*xi*y2 + 0.125*x4*y1 - 0.125*x4*y3

                    j11 = 2.0*(-xi*y1 + xi*y2 - xi*y3 + xi*y4 + y1 + y2 - y3 - y4)/(eta*x1*y2 - eta*x1*y3 - eta*x2*y1 + eta*x2*y4 + eta*x3*y1 - eta*x3*y4 - eta*x4*y2 + eta*x4*y3 + x1*xi*y3 - x1*xi*y4 - x1*y2 + x1*y4 - x2*xi*y3 + x2*xi*y4 + x2*y1 - x2*y3 - x3*xi*y1 + x3*xi*y2 + x3*y2 - x3*y4 + x4*xi*y1 - x4*xi*y2 - x4*y1 + x4*y3)
                    j12 = 2.0*(eta*y1 - eta*y2 + eta*y3 - eta*y4 - y1 + y2 + y3 - y4)/(eta*x1*y2 - eta*x1*y3 - eta*x2*y1 + eta*x2*y4 + eta*x3*y1 - eta*x3*y4 - eta*x4*y2 + eta*x4*y3 + x1*xi*y3 - x1*xi*y4 - x1*y2 + x1*y4 - x2*xi*y3 + x2*xi*y4 + x2*y1 - x2*y3 - x3*xi*y1 + x3*xi*y2 + x3*y2 - x3*y4 + x4*xi*y1 - x4*xi*y2 - x4*y1 + x4*y3)
                    j21 = 2.0*(x1*xi - x1 - x2*xi - x2 + x3*xi + x3 - x4*xi + x4)/(eta*x1*y2 - eta*x1*y3 - eta*x2*y1 + eta*x2*y4 + eta*x3*y1 - eta*x3*y4 - eta*x4*y2 + eta*x4*y3 + x1*xi*y3 - x1*xi*y4 - x1*y2 + x1*y4 - x2*xi*y3 + x2*xi*y4 + x2*y1 - x2*y3 - x3*xi*y1 + x3*xi*y2 + x3*y2 - x3*y4 + x4*xi*y1 - x4*xi*y2 - x4*y1 + x4*y3)
                    j22 = 2.0*(-eta*x1 + eta*x2 - eta*x3 + eta*x4 + x1 - x2 - x3 + x4)/(eta*x1*y2 - eta*x1*y3 - eta*x2*y1 + eta*x2*y4 + eta*x3*y1 - eta*x3*y4 - eta*x4*y2 + eta*x4*y3 + x1*xi*y3 - x1*xi*y4 - x1*y2 + x1*y4 - x2*xi*y3 + x2*xi*y4 + x2*y1 - x2*y3 - x3*xi*y1 + x3*xi*y2 + x3*y2 - x3*y4 + x4*xi*y1 - x4*xi*y2 - x4*y1 + x4*y3)

                    S1_x = 0.25*j11*(eta - 1) + 0.25*j12*(xi - 1)
                    S2_x = -0.25*eta*j11 + 0.25*j11 - 0.25*j12*xi - 0.25*j12
                    S3_x = 0.25*j11*(eta + 1) + 0.25*j12*(xi + 1)
                    S4_x = -0.25*eta*j11 - 0.25*j11 - 0.25*j12*xi + 0.25*j12

                    S1_y = 0.25*j21*(eta - 1) + 0.25*j22*(xi - 1)
                    S2_y = -0.25*eta*j21 + 0.25*j21 - 0.25*j22*xi - 0.25*j22
                    S3_y = 0.25*j21*(eta + 1) + 0.25*j22*(xi + 1)
                    S4_y = -0.25*eta*j21 - 0.25*j21 - 0.25*j22*xi + 0.25*j22

                    # NOTE slopes of w at the centroid
                    S1_x = Scx[0]
                    S2_x = Scx[1]
                    S3_x = Scx[2]
                    S4_x = Scx[3]
                    S1_y = Scy[0]
                    S2_y = Scy[1]
                    S3_y = Scy[2]
                    S4_y = Scy[3]

                    k = self.init_k_KG
                    KGv[k] += r13**2*(S1_x*(Nxx*S1_x*detJ*wij + Nxy*S1_y*detJ*wij) + S1_y*(Nxy*S1_x*detJ*wij + Nyy*S1_y*detJ*wij))
                    k += 1
                    KGv[k] += r13*r23*(S1_x*(Nxx*S1_x*detJ*wij + Nxy*S1_y*detJ*wij) + S1_y*(Nxy*S1_x*detJ*wij + Nyy*S1_y*detJ*wij))
                    k += 1
                    KGv[k] += r13*r33*(S1_x*(Nxx*S1_x*detJ*wij + Nxy*S1_y*detJ*wij) + S1_y*(Nxy*S1_x*detJ*wij + Nyy*S1_y*detJ*wij))
                    k += 1
                    KGv[k] += r13**2*(S2_x*(Nxx*S1_x*detJ*wij + Nxy*S1_y*detJ*wij) + S2_y*(Nxy*S1_x*detJ*wij + Nyy*S1_y*detJ*wij))
                    k += 1
                    KGv[k] += r13*r23*(S2_x*(Nxx*S1_x*detJ*wij + Nxy*S1_y*detJ*wij) + S2_y*(Nxy*S1_x*detJ*wij + Nyy*S1_y*detJ*wij))
                    k += 1
                    KGv[k] += r13*r33*(S2_x*(Nxx*S1_x*detJ*wij + Nxy*S1_y*detJ*wij) + S2_y*(Nxy*S1_x*detJ*wij + Nyy*S1_y*detJ*wij))
                    k += 1
                    KGv[k] += r13**2*(S3_x*(Nxx*S1_x*detJ*wij + Nxy*S1_y*detJ*wij) + S3_y*(Nxy*S1_x*detJ*wij + Nyy*S1_y*detJ*wij))
                    k += 1
                    KGv[k] += r13*r23*(S3_x*(Nxx*S1_x*detJ*wij + Nxy*S1_y*detJ*wij) + S3_y*(Nxy*S1_x*detJ*wij + Nyy*S1_y*detJ*wij))
                    k += 1
                    KGv[k] += r13*r33*(S3_x*(Nxx*S1_x*detJ*wij + Nxy*S1_y*detJ*wij) + S3_y*(Nxy*S1_x*detJ*wij + Nyy*S1_y*detJ*wij))
                    k += 1
                    KGv[k] += r13**2*(S4_x*(Nxx*S1_x*detJ*wij + Nxy*S1_y*detJ*wij) + S4_y*(Nxy*S1_x*detJ*wij + Nyy*S1_y*detJ*wij))
                    k += 1
                    KGv[k] += r13*r23*(S4_x*(Nxx*S1_x*detJ*wij + Nxy*S1_y*detJ*wij) + S4_y*(Nxy*S1_x*detJ*wij + Nyy*S1_y*detJ*wij))
                    k += 1
                    KGv[k] += r13*r33*(S4_x*(Nxx*S1_x*detJ*wij + Nxy*S1_y*detJ*wij) + S4_y*(Nxy*S1_x*detJ*wij + Nyy*S1_y*detJ*wij))
                    k += 1
                    KGv[k] += r13*r23*(S1_x*(Nxx*S1_x*detJ*wij + Nxy*S1_y*detJ*wij) + S1_y*(Nxy*S1_x*detJ*wij + Nyy*S1_y*detJ*wij))
                    k += 1
                    KGv[k] += r23**2*(S1_x*(Nxx*S1_x*detJ*wij + Nxy*S1_y*detJ*wij) + S1_y*(Nxy*S1_x*detJ*wij + Nyy*S1_y*detJ*wij))
                    k += 1
                    KGv[k] += r23*r33*(S1_x*(Nxx*S1_x*detJ*wij + Nxy*S1_y*detJ*wij) + S1_y*(Nxy*S1_x*detJ*wij + Nyy*S1_y*detJ*wij))
                    k += 1
                    KGv[k] += r13*r23*(S2_x*(Nxx*S1_x*detJ*wij + Nxy*S1_y*detJ*wij) + S2_y*(Nxy*S1_x*detJ*wij + Nyy*S1_y*detJ*wij))
                    k += 1
                    KGv[k] += r23**2*(S2_x*(Nxx*S1_x*detJ*wij + Nxy*S1_y*detJ*wij) + S2_y*(Nxy*S1_x*detJ*wij + Nyy*S1_y*detJ*wij))
                    k += 1
                    KGv[k] += r23*r33*(S2_x*(Nxx*S1_x*detJ*wij + Nxy*S1_y*detJ*wij) + S2_y*(Nxy*S1_x*detJ*wij + Nyy*S1_y*detJ*wij))
                    k += 1
                    KGv[k] += r13*r23*(S3_x*(Nxx*S1_x*detJ*wij + Nxy*S1_y*detJ*wij) + S3_y*(Nxy*S1_x*detJ*wij + Nyy*S1_y*detJ*wij))
                    k += 1
                    KGv[k] += r23**2*(S3_x*(Nxx*S1_x*detJ*wij + Nxy*S1_y*detJ*wij) + S3_y*(Nxy*S1_x*detJ*wij + Nyy*S1_y*detJ*wij))
                    k += 1
                    KGv[k] += r23*r33*(S3_x*(Nxx*S1_x*detJ*wij + Nxy*S1_y*detJ*wij) + S3_y*(Nxy*S1_x*detJ*wij + Nyy*S1_y*detJ*wij))
                    k += 1
                    KGv[k] += r13*r23*(S4_x*(Nxx*S1_x*detJ*wij + Nxy*S1_y*detJ*wij) + S4_y*(Nxy*S1_x*detJ*wij + Nyy*S1_y*detJ*wij))
                    k += 1
                    KGv[k] += r23**2*(S4_x*(Nxx*S1_x*detJ*wij + Nxy*S1_y*detJ*wij) + S4_y*(Nxy*S1_x*detJ*wij + Nyy*S1_y*detJ*wij))
                    k += 1
                    KGv[k] += r23*r33*(S4_x*(Nxx*S1_x*detJ*wij + Nxy*S1_y*detJ*wij) + S4_y*(Nxy*S1_x*detJ*wij + Nyy*S1_y*detJ*wij))
                    k += 1
                    KGv[k] += r13*r33*(S1_x*(Nxx*S1_x*detJ*wij + Nxy*S1_y*detJ*wij) + S1_y*(Nxy*S1_x*detJ*wij + Nyy*S1_y*detJ*wij))
                    k += 1
                    KGv[k] += r23*r33*(S1_x*(Nxx*S1_x*detJ*wij + Nxy*S1_y*detJ*wij) + S1_y*(Nxy*S1_x*detJ*wij + Nyy*S1_y*detJ*wij))
                    k += 1
                    KGv[k] += r33**2*(S1_x*(Nxx*S1_x*detJ*wij + Nxy*S1_y*detJ*wij) + S1_y*(Nxy*S1_x*detJ*wij + Nyy*S1_y*detJ*wij))
                    k += 1
                    KGv[k] += r13*r33*(S2_x*(Nxx*S1_x*detJ*wij + Nxy*S1_y*detJ*wij) + S2_y*(Nxy*S1_x*detJ*wij + Nyy*S1_y*detJ*wij))
                    k += 1
                    KGv[k] += r23*r33*(S2_x*(Nxx*S1_x*detJ*wij + Nxy*S1_y*detJ*wij) + S2_y*(Nxy*S1_x*detJ*wij + Nyy*S1_y*detJ*wij))
                    k += 1
                    KGv[k] += r33**2*(S2_x*(Nxx*S1_x*detJ*wij + Nxy*S1_y*detJ*wij) + S2_y*(Nxy*S1_x*detJ*wij + Nyy*S1_y*detJ*wij))
                    k += 1
                    KGv[k] += r13*r33*(S3_x*(Nxx*S1_x*detJ*wij + Nxy*S1_y*detJ*wij) + S3_y*(Nxy*S1_x*detJ*wij + Nyy*S1_y*detJ*wij))
                    k += 1
                    KGv[k] += r23*r33*(S3_x*(Nxx*S1_x*detJ*wij + Nxy*S1_y*detJ*wij) + S3_y*(Nxy*S1_x*detJ*wij + Nyy*S1_y*detJ*wij))
                    k += 1
                    KGv[k] += r33**2*(S3_x*(Nxx*S1_x*detJ*wij + Nxy*S1_y*detJ*wij) + S3_y*(Nxy*S1_x*detJ*wij + Nyy*S1_y*detJ*wij))
                    k += 1
                    KGv[k] += r13*r33*(S4_x*(Nxx*S1_x*detJ*wij + Nxy*S1_y*detJ*wij) + S4_y*(Nxy*S1_x*detJ*wij + Nyy*S1_y*detJ*wij))
                    k += 1
                    KGv[k] += r23*r33*(S4_x*(Nxx*S1_x*detJ*wij + Nxy*S1_y*detJ*wij) + S4_y*(Nxy*S1_x*detJ*wij + Nyy*S1_y*detJ*wij))
                    k += 1
                    KGv[k] += r33**2*(S4_x*(Nxx*S1_x*detJ*wij + Nxy*S1_y*detJ*wij) + S4_y*(Nxy*S1_x*detJ*wij + Nyy*S1_y*detJ*wij))
                    k += 1
                    KGv[k] += r13**2*(S1_x*(Nxx*S2_x*detJ*wij + Nxy*S2_y*detJ*wij) + S1_y*(Nxy*S2_x*detJ*wij + Nyy*S2_y*detJ*wij))
                    k += 1
                    KGv[k] += r13*r23*(S1_x*(Nxx*S2_x*detJ*wij + Nxy*S2_y*detJ*wij) + S1_y*(Nxy*S2_x*detJ*wij + Nyy*S2_y*detJ*wij))
                    k += 1
                    KGv[k] += r13*r33*(S1_x*(Nxx*S2_x*detJ*wij + Nxy*S2_y*detJ*wij) + S1_y*(Nxy*S2_x*detJ*wij + Nyy*S2_y*detJ*wij))
                    k += 1
                    KGv[k] += r13**2*(S2_x*(Nxx*S2_x*detJ*wij + Nxy*S2_y*detJ*wij) + S2_y*(Nxy*S2_x*detJ*wij + Nyy*S2_y*detJ*wij))
                    k += 1
                    KGv[k] += r13*r23*(S2_x*(Nxx*S2_x*detJ*wij + Nxy*S2_y*detJ*wij) + S2_y*(Nxy*S2_x*detJ*wij + Nyy*S2_y*detJ*wij))
                    k += 1
                    KGv[k] += r13*r33*(S2_x*(Nxx*S2_x*detJ*wij + Nxy*S2_y*detJ*wij) + S2_y*(Nxy*S2_x*detJ*wij + Nyy*S2_y*detJ*wij))
                    k += 1
                    KGv[k] += r13**2*(S3_x*(Nxx*S2_x*detJ*wij + Nxy*S2_y*detJ*wij) + S3_y*(Nxy*S2_x*detJ*wij + Nyy*S2_y*detJ*wij))
                    k += 1
                    KGv[k] += r13*r23*(S3_x*(Nxx*S2_x*detJ*wij + Nxy*S2_y*detJ*wij) + S3_y*(Nxy*S2_x*detJ*wij + Nyy*S2_y*detJ*wij))
                    k += 1
                    KGv[k] += r13*r33*(S3_x*(Nxx*S2_x*detJ*wij + Nxy*S2_y*detJ*wij) + S3_y*(Nxy*S2_x*detJ*wij + Nyy*S2_y*detJ*wij))
                    k += 1
                    KGv[k] += r13**2*(S4_x*(Nxx*S2_x*detJ*wij + Nxy*S2_y*detJ*wij) + S4_y*(Nxy*S2_x*detJ*wij + Nyy*S2_y*detJ*wij))
                    k += 1
                    KGv[k] += r13*r23*(S4_x*(Nxx*S2_x*detJ*wij + Nxy*S2_y*detJ*wij) + S4_y*(Nxy*S2_x*detJ*wij + Nyy*S2_y*detJ*wij))
                    k += 1
                    KGv[k] += r13*r33*(S4_x*(Nxx*S2_x*detJ*wij + Nxy*S2_y*detJ*wij) + S4_y*(Nxy*S2_x*detJ*wij + Nyy*S2_y*detJ*wij))
                    k += 1
                    KGv[k] += r13*r23*(S1_x*(Nxx*S2_x*detJ*wij + Nxy*S2_y*detJ*wij) + S1_y*(Nxy*S2_x*detJ*wij + Nyy*S2_y*detJ*wij))
                    k += 1
                    KGv[k] += r23**2*(S1_x*(Nxx*S2_x*detJ*wij + Nxy*S2_y*detJ*wij) + S1_y*(Nxy*S2_x*detJ*wij + Nyy*S2_y*detJ*wij))
                    k += 1
                    KGv[k] += r23*r33*(S1_x*(Nxx*S2_x*detJ*wij + Nxy*S2_y*detJ*wij) + S1_y*(Nxy*S2_x*detJ*wij + Nyy*S2_y*detJ*wij))
                    k += 1
                    KGv[k] += r13*r23*(S2_x*(Nxx*S2_x*detJ*wij + Nxy*S2_y*detJ*wij) + S2_y*(Nxy*S2_x*detJ*wij + Nyy*S2_y*detJ*wij))
                    k += 1
                    KGv[k] += r23**2*(S2_x*(Nxx*S2_x*detJ*wij + Nxy*S2_y*detJ*wij) + S2_y*(Nxy*S2_x*detJ*wij + Nyy*S2_y*detJ*wij))
                    k += 1
                    KGv[k] += r23*r33*(S2_x*(Nxx*S2_x*detJ*wij + Nxy*S2_y*detJ*wij) + S2_y*(Nxy*S2_x*detJ*wij + Nyy*S2_y*detJ*wij))
                    k += 1
                    KGv[k] += r13*r23*(S3_x*(Nxx*S2_x*detJ*wij + Nxy*S2_y*detJ*wij) + S3_y*(Nxy*S2_x*detJ*wij + Nyy*S2_y*detJ*wij))
                    k += 1
                    KGv[k] += r23**2*(S3_x*(Nxx*S2_x*detJ*wij + Nxy*S2_y*detJ*wij) + S3_y*(Nxy*S2_x*detJ*wij + Nyy*S2_y*detJ*wij))
                    k += 1
                    KGv[k] += r23*r33*(S3_x*(Nxx*S2_x*detJ*wij + Nxy*S2_y*detJ*wij) + S3_y*(Nxy*S2_x*detJ*wij + Nyy*S2_y*detJ*wij))
                    k += 1
                    KGv[k] += r13*r23*(S4_x*(Nxx*S2_x*detJ*wij + Nxy*S2_y*detJ*wij) + S4_y*(Nxy*S2_x*detJ*wij + Nyy*S2_y*detJ*wij))
                    k += 1
                    KGv[k] += r23**2*(S4_x*(Nxx*S2_x*detJ*wij + Nxy*S2_y*detJ*wij) + S4_y*(Nxy*S2_x*detJ*wij + Nyy*S2_y*detJ*wij))
                    k += 1
                    KGv[k] += r23*r33*(S4_x*(Nxx*S2_x*detJ*wij + Nxy*S2_y*detJ*wij) + S4_y*(Nxy*S2_x*detJ*wij + Nyy*S2_y*detJ*wij))
                    k += 1
                    KGv[k] += r13*r33*(S1_x*(Nxx*S2_x*detJ*wij + Nxy*S2_y*detJ*wij) + S1_y*(Nxy*S2_x*detJ*wij + Nyy*S2_y*detJ*wij))
                    k += 1
                    KGv[k] += r23*r33*(S1_x*(Nxx*S2_x*detJ*wij + Nxy*S2_y*detJ*wij) + S1_y*(Nxy*S2_x*detJ*wij + Nyy*S2_y*detJ*wij))
                    k += 1
                    KGv[k] += r33**2*(S1_x*(Nxx*S2_x*detJ*wij + Nxy*S2_y*detJ*wij) + S1_y*(Nxy*S2_x*detJ*wij + Nyy*S2_y*detJ*wij))
                    k += 1
                    KGv[k] += r13*r33*(S2_x*(Nxx*S2_x*detJ*wij + Nxy*S2_y*detJ*wij) + S2_y*(Nxy*S2_x*detJ*wij + Nyy*S2_y*detJ*wij))
                    k += 1
                    KGv[k] += r23*r33*(S2_x*(Nxx*S2_x*detJ*wij + Nxy*S2_y*detJ*wij) + S2_y*(Nxy*S2_x*detJ*wij + Nyy*S2_y*detJ*wij))
                    k += 1
                    KGv[k] += r33**2*(S2_x*(Nxx*S2_x*detJ*wij + Nxy*S2_y*detJ*wij) + S2_y*(Nxy*S2_x*detJ*wij + Nyy*S2_y*detJ*wij))
                    k += 1
                    KGv[k] += r13*r33*(S3_x*(Nxx*S2_x*detJ*wij + Nxy*S2_y*detJ*wij) + S3_y*(Nxy*S2_x*detJ*wij + Nyy*S2_y*detJ*wij))
                    k += 1
                    KGv[k] += r23*r33*(S3_x*(Nxx*S2_x*detJ*wij + Nxy*S2_y*detJ*wij) + S3_y*(Nxy*S2_x*detJ*wij + Nyy*S2_y*detJ*wij))
                    k += 1
                    KGv[k] += r33**2*(S3_x*(Nxx*S2_x*detJ*wij + Nxy*S2_y*detJ*wij) + S3_y*(Nxy*S2_x*detJ*wij + Nyy*S2_y*detJ*wij))
                    k += 1
                    KGv[k] += r13*r33*(S4_x*(Nxx*S2_x*detJ*wij + Nxy*S2_y*detJ*wij) + S4_y*(Nxy*S2_x*detJ*wij + Nyy*S2_y*detJ*wij))
                    k += 1
                    KGv[k] += r23*r33*(S4_x*(Nxx*S2_x*detJ*wij + Nxy*S2_y*detJ*wij) + S4_y*(Nxy*S2_x*detJ*wij + Nyy*S2_y*detJ*wij))
                    k += 1
                    KGv[k] += r33**2*(S4_x*(Nxx*S2_x*detJ*wij + Nxy*S2_y*detJ*wij) + S4_y*(Nxy*S2_x*detJ*wij + Nyy*S2_y*detJ*wij))
                    k += 1
                    KGv[k] += r13**2*(S1_x*(Nxx*S3_x*detJ*wij + Nxy*S3_y*detJ*wij) + S1_y*(Nxy*S3_x*detJ*wij + Nyy*S3_y*detJ*wij))
                    k += 1
                    KGv[k] += r13*r23*(S1_x*(Nxx*S3_x*detJ*wij + Nxy*S3_y*detJ*wij) + S1_y*(Nxy*S3_x*detJ*wij + Nyy*S3_y*detJ*wij))
                    k += 1
                    KGv[k] += r13*r33*(S1_x*(Nxx*S3_x*detJ*wij + Nxy*S3_y*detJ*wij) + S1_y*(Nxy*S3_x*detJ*wij + Nyy*S3_y*detJ*wij))
                    k += 1
                    KGv[k] += r13**2*(S2_x*(Nxx*S3_x*detJ*wij + Nxy*S3_y*detJ*wij) + S2_y*(Nxy*S3_x*detJ*wij + Nyy*S3_y*detJ*wij))
                    k += 1
                    KGv[k] += r13*r23*(S2_x*(Nxx*S3_x*detJ*wij + Nxy*S3_y*detJ*wij) + S2_y*(Nxy*S3_x*detJ*wij + Nyy*S3_y*detJ*wij))
                    k += 1
                    KGv[k] += r13*r33*(S2_x*(Nxx*S3_x*detJ*wij + Nxy*S3_y*detJ*wij) + S2_y*(Nxy*S3_x*detJ*wij + Nyy*S3_y*detJ*wij))
                    k += 1
                    KGv[k] += r13**2*(S3_x*(Nxx*S3_x*detJ*wij + Nxy*S3_y*detJ*wij) + S3_y*(Nxy*S3_x*detJ*wij + Nyy*S3_y*detJ*wij))
                    k += 1
                    KGv[k] += r13*r23*(S3_x*(Nxx*S3_x*detJ*wij + Nxy*S3_y*detJ*wij) + S3_y*(Nxy*S3_x*detJ*wij + Nyy*S3_y*detJ*wij))
                    k += 1
                    KGv[k] += r13*r33*(S3_x*(Nxx*S3_x*detJ*wij + Nxy*S3_y*detJ*wij) + S3_y*(Nxy*S3_x*detJ*wij + Nyy*S3_y*detJ*wij))
                    k += 1
                    KGv[k] += r13**2*(S4_x*(Nxx*S3_x*detJ*wij + Nxy*S3_y*detJ*wij) + S4_y*(Nxy*S3_x*detJ*wij + Nyy*S3_y*detJ*wij))
                    k += 1
                    KGv[k] += r13*r23*(S4_x*(Nxx*S3_x*detJ*wij + Nxy*S3_y*detJ*wij) + S4_y*(Nxy*S3_x*detJ*wij + Nyy*S3_y*detJ*wij))
                    k += 1
                    KGv[k] += r13*r33*(S4_x*(Nxx*S3_x*detJ*wij + Nxy*S3_y*detJ*wij) + S4_y*(Nxy*S3_x*detJ*wij + Nyy*S3_y*detJ*wij))
                    k += 1
                    KGv[k] += r13*r23*(S1_x*(Nxx*S3_x*detJ*wij + Nxy*S3_y*detJ*wij) + S1_y*(Nxy*S3_x*detJ*wij + Nyy*S3_y*detJ*wij))
                    k += 1
                    KGv[k] += r23**2*(S1_x*(Nxx*S3_x*detJ*wij + Nxy*S3_y*detJ*wij) + S1_y*(Nxy*S3_x*detJ*wij + Nyy*S3_y*detJ*wij))
                    k += 1
                    KGv[k] += r23*r33*(S1_x*(Nxx*S3_x*detJ*wij + Nxy*S3_y*detJ*wij) + S1_y*(Nxy*S3_x*detJ*wij + Nyy*S3_y*detJ*wij))
                    k += 1
                    KGv[k] += r13*r23*(S2_x*(Nxx*S3_x*detJ*wij + Nxy*S3_y*detJ*wij) + S2_y*(Nxy*S3_x*detJ*wij + Nyy*S3_y*detJ*wij))
                    k += 1
                    KGv[k] += r23**2*(S2_x*(Nxx*S3_x*detJ*wij + Nxy*S3_y*detJ*wij) + S2_y*(Nxy*S3_x*detJ*wij + Nyy*S3_y*detJ*wij))
                    k += 1
                    KGv[k] += r23*r33*(S2_x*(Nxx*S3_x*detJ*wij + Nxy*S3_y*detJ*wij) + S2_y*(Nxy*S3_x*detJ*wij + Nyy*S3_y*detJ*wij))
                    k += 1
                    KGv[k] += r13*r23*(S3_x*(Nxx*S3_x*detJ*wij + Nxy*S3_y*detJ*wij) + S3_y*(Nxy*S3_x*detJ*wij + Nyy*S3_y*detJ*wij))
                    k += 1
                    KGv[k] += r23**2*(S3_x*(Nxx*S3_x*detJ*wij + Nxy*S3_y*detJ*wij) + S3_y*(Nxy*S3_x*detJ*wij + Nyy*S3_y*detJ*wij))
                    k += 1
                    KGv[k] += r23*r33*(S3_x*(Nxx*S3_x*detJ*wij + Nxy*S3_y*detJ*wij) + S3_y*(Nxy*S3_x*detJ*wij + Nyy*S3_y*detJ*wij))
                    k += 1
                    KGv[k] += r13*r23*(S4_x*(Nxx*S3_x*detJ*wij + Nxy*S3_y*detJ*wij) + S4_y*(Nxy*S3_x*detJ*wij + Nyy*S3_y*detJ*wij))
                    k += 1
                    KGv[k] += r23**2*(S4_x*(Nxx*S3_x*detJ*wij + Nxy*S3_y*detJ*wij) + S4_y*(Nxy*S3_x*detJ*wij + Nyy*S3_y*detJ*wij))
                    k += 1
                    KGv[k] += r23*r33*(S4_x*(Nxx*S3_x*detJ*wij + Nxy*S3_y*detJ*wij) + S4_y*(Nxy*S3_x*detJ*wij + Nyy*S3_y*detJ*wij))
                    k += 1
                    KGv[k] += r13*r33*(S1_x*(Nxx*S3_x*detJ*wij + Nxy*S3_y*detJ*wij) + S1_y*(Nxy*S3_x*detJ*wij + Nyy*S3_y*detJ*wij))
                    k += 1
                    KGv[k] += r23*r33*(S1_x*(Nxx*S3_x*detJ*wij + Nxy*S3_y*detJ*wij) + S1_y*(Nxy*S3_x*detJ*wij + Nyy*S3_y*detJ*wij))
                    k += 1
                    KGv[k] += r33**2*(S1_x*(Nxx*S3_x*detJ*wij + Nxy*S3_y*detJ*wij) + S1_y*(Nxy*S3_x*detJ*wij + Nyy*S3_y*detJ*wij))
                    k += 1
                    KGv[k] += r13*r33*(S2_x*(Nxx*S3_x*detJ*wij + Nxy*S3_y*detJ*wij) + S2_y*(Nxy*S3_x*detJ*wij + Nyy*S3_y*detJ*wij))
                    k += 1
                    KGv[k] += r23*r33*(S2_x*(Nxx*S3_x*detJ*wij + Nxy*S3_y*detJ*wij) + S2_y*(Nxy*S3_x*detJ*wij + Nyy*S3_y*detJ*wij))
                    k += 1
                    KGv[k] += r33**2*(S2_x*(Nxx*S3_x*detJ*wij + Nxy*S3_y*detJ*wij) + S2_y*(Nxy*S3_x*detJ*wij + Nyy*S3_y*detJ*wij))
                    k += 1
                    KGv[k] += r13*r33*(S3_x*(Nxx*S3_x*detJ*wij + Nxy*S3_y*detJ*wij) + S3_y*(Nxy*S3_x*detJ*wij + Nyy*S3_y*detJ*wij))
                    k += 1
                    KGv[k] += r23*r33*(S3_x*(Nxx*S3_x*detJ*wij + Nxy*S3_y*detJ*wij) + S3_y*(Nxy*S3_x*detJ*wij + Nyy*S3_y*detJ*wij))
                    k += 1
                    KGv[k] += r33**2*(S3_x*(Nxx*S3_x*detJ*wij + Nxy*S3_y*detJ*wij) + S3_y*(Nxy*S3_x*detJ*wij + Nyy*S3_y*detJ*wij))
                    k += 1
                    KGv[k] += r13*r33*(S4_x*(Nxx*S3_x*detJ*wij + Nxy*S3_y*detJ*wij) + S4_y*(Nxy*S3_x*detJ*wij + Nyy*S3_y*detJ*wij))
                    k += 1
                    KGv[k] += r23*r33*(S4_x*(Nxx*S3_x*detJ*wij + Nxy*S3_y*detJ*wij) + S4_y*(Nxy*S3_x*detJ*wij + Nyy*S3_y*detJ*wij))
                    k += 1
                    KGv[k] += r33**2*(S4_x*(Nxx*S3_x*detJ*wij + Nxy*S3_y*detJ*wij) + S4_y*(Nxy*S3_x*detJ*wij + Nyy*S3_y*detJ*wij))
                    k += 1
                    KGv[k] += r13**2*(S1_x*(Nxx*S4_x*detJ*wij + Nxy*S4_y*detJ*wij) + S1_y*(Nxy*S4_x*detJ*wij + Nyy*S4_y*detJ*wij))
                    k += 1
                    KGv[k] += r13*r23*(S1_x*(Nxx*S4_x*detJ*wij + Nxy*S4_y*detJ*wij) + S1_y*(Nxy*S4_x*detJ*wij + Nyy*S4_y*detJ*wij))
                    k += 1
                    KGv[k] += r13*r33*(S1_x*(Nxx*S4_x*detJ*wij + Nxy*S4_y*detJ*wij) + S1_y*(Nxy*S4_x*detJ*wij + Nyy*S4_y*detJ*wij))
                    k += 1
                    KGv[k] += r13**2*(S2_x*(Nxx*S4_x*detJ*wij + Nxy*S4_y*detJ*wij) + S2_y*(Nxy*S4_x*detJ*wij + Nyy*S4_y*detJ*wij))
                    k += 1
                    KGv[k] += r13*r23*(S2_x*(Nxx*S4_x*detJ*wij + Nxy*S4_y*detJ*wij) + S2_y*(Nxy*S4_x*detJ*wij + Nyy*S4_y*detJ*wij))
                    k += 1
                    KGv[k] += r13*r33*(S2_x*(Nxx*S4_x*detJ*wij + Nxy*S4_y*detJ*wij) + S2_y*(Nxy*S4_x*detJ*wij + Nyy*S4_y*detJ*wij))
                    k += 1
                    KGv[k] += r13**2*(S3_x*(Nxx*S4_x*detJ*wij + Nxy*S4_y*detJ*wij) + S3_y*(Nxy*S4_x*detJ*wij + Nyy*S4_y*detJ*wij))
                    k += 1
                    KGv[k] += r13*r23*(S3_x*(Nxx*S4_x*detJ*wij + Nxy*S4_y*detJ*wij) + S3_y*(Nxy*S4_x*detJ*wij + Nyy*S4_y*detJ*wij))
                    k += 1
                    KGv[k] += r13*r33*(S3_x*(Nxx*S4_x*detJ*wij + Nxy*S4_y*detJ*wij) + S3_y*(Nxy*S4_x*detJ*wij + Nyy*S4_y*detJ*wij))
                    k += 1
                    KGv[k] += r13**2*(S4_x*(Nxx*S4_x*detJ*wij + Nxy*S4_y*detJ*wij) + S4_y*(Nxy*S4_x*detJ*wij + Nyy*S4_y*detJ*wij))
                    k += 1
                    KGv[k] += r13*r23*(S4_x*(Nxx*S4_x*detJ*wij + Nxy*S4_y*detJ*wij) + S4_y*(Nxy*S4_x*detJ*wij + Nyy*S4_y*detJ*wij))
                    k += 1
                    KGv[k] += r13*r33*(S4_x*(Nxx*S4_x*detJ*wij + Nxy*S4_y*detJ*wij) + S4_y*(Nxy*S4_x*detJ*wij + Nyy*S4_y*detJ*wij))
                    k += 1
                    KGv[k] += r13*r23*(S1_x*(Nxx*S4_x*detJ*wij + Nxy*S4_y*detJ*wij) + S1_y*(Nxy*S4_x*detJ*wij + Nyy*S4_y*detJ*wij))
                    k += 1
                    KGv[k] += r23**2*(S1_x*(Nxx*S4_x*detJ*wij + Nxy*S4_y*detJ*wij) + S1_y*(Nxy*S4_x*detJ*wij + Nyy*S4_y*detJ*wij))
                    k += 1
                    KGv[k] += r23*r33*(S1_x*(Nxx*S4_x*detJ*wij + Nxy*S4_y*detJ*wij) + S1_y*(Nxy*S4_x*detJ*wij + Nyy*S4_y*detJ*wij))
                    k += 1
                    KGv[k] += r13*r23*(S2_x*(Nxx*S4_x*detJ*wij + Nxy*S4_y*detJ*wij) + S2_y*(Nxy*S4_x*detJ*wij + Nyy*S4_y*detJ*wij))
                    k += 1
                    KGv[k] += r23**2*(S2_x*(Nxx*S4_x*detJ*wij + Nxy*S4_y*detJ*wij) + S2_y*(Nxy*S4_x*detJ*wij + Nyy*S4_y*detJ*wij))
                    k += 1
                    KGv[k] += r23*r33*(S2_x*(Nxx*S4_x*detJ*wij + Nxy*S4_y*detJ*wij) + S2_y*(Nxy*S4_x*detJ*wij + Nyy*S4_y*detJ*wij))
                    k += 1
                    KGv[k] += r13*r23*(S3_x*(Nxx*S4_x*detJ*wij + Nxy*S4_y*detJ*wij) + S3_y*(Nxy*S4_x*detJ*wij + Nyy*S4_y*detJ*wij))
                    k += 1
                    KGv[k] += r23**2*(S3_x*(Nxx*S4_x*detJ*wij + Nxy*S4_y*detJ*wij) + S3_y*(Nxy*S4_x*detJ*wij + Nyy*S4_y*detJ*wij))
                    k += 1
                    KGv[k] += r23*r33*(S3_x*(Nxx*S4_x*detJ*wij + Nxy*S4_y*detJ*wij) + S3_y*(Nxy*S4_x*detJ*wij + Nyy*S4_y*detJ*wij))
                    k += 1
                    KGv[k] += r13*r23*(S4_x*(Nxx*S4_x*detJ*wij + Nxy*S4_y*detJ*wij) + S4_y*(Nxy*S4_x*detJ*wij + Nyy*S4_y*detJ*wij))
                    k += 1
                    KGv[k] += r23**2*(S4_x*(Nxx*S4_x*detJ*wij + Nxy*S4_y*detJ*wij) + S4_y*(Nxy*S4_x*detJ*wij + Nyy*S4_y*detJ*wij))
                    k += 1
                    KGv[k] += r23*r33*(S4_x*(Nxx*S4_x*detJ*wij + Nxy*S4_y*detJ*wij) + S4_y*(Nxy*S4_x*detJ*wij + Nyy*S4_y*detJ*wij))
                    k += 1
                    KGv[k] += r13*r33*(S1_x*(Nxx*S4_x*detJ*wij + Nxy*S4_y*detJ*wij) + S1_y*(Nxy*S4_x*detJ*wij + Nyy*S4_y*detJ*wij))
                    k += 1
                    KGv[k] += r23*r33*(S1_x*(Nxx*S4_x*detJ*wij + Nxy*S4_y*detJ*wij) + S1_y*(Nxy*S4_x*detJ*wij + Nyy*S4_y*detJ*wij))
                    k += 1
                    KGv[k] += r33**2*(S1_x*(Nxx*S4_x*detJ*wij + Nxy*S4_y*detJ*wij) + S1_y*(Nxy*S4_x*detJ*wij + Nyy*S4_y*detJ*wij))
                    k += 1
                    KGv[k] += r13*r33*(S2_x*(Nxx*S4_x*detJ*wij + Nxy*S4_y*detJ*wij) + S2_y*(Nxy*S4_x*detJ*wij + Nyy*S4_y*detJ*wij))
                    k += 1
                    KGv[k] += r23*r33*(S2_x*(Nxx*S4_x*detJ*wij + Nxy*S4_y*detJ*wij) + S2_y*(Nxy*S4_x*detJ*wij + Nyy*S4_y*detJ*wij))
                    k += 1
                    KGv[k] += r33**2*(S2_x*(Nxx*S4_x*detJ*wij + Nxy*S4_y*detJ*wij) + S2_y*(Nxy*S4_x*detJ*wij + Nyy*S4_y*detJ*wij))
                    k += 1
                    KGv[k] += r13*r33*(S3_x*(Nxx*S4_x*detJ*wij + Nxy*S4_y*detJ*wij) + S3_y*(Nxy*S4_x*detJ*wij + Nyy*S4_y*detJ*wij))
                    k += 1
                    KGv[k] += r23*r33*(S3_x*(Nxx*S4_x*detJ*wij + Nxy*S4_y*detJ*wij) + S3_y*(Nxy*S4_x*detJ*wij + Nyy*S4_y*detJ*wij))
                    k += 1
                    KGv[k] += r33**2*(S3_x*(Nxx*S4_x*detJ*wij + Nxy*S4_y*detJ*wij) + S3_y*(Nxy*S4_x*detJ*wij + Nyy*S4_y*detJ*wij))
                    k += 1
                    KGv[k] += r13*r33*(S4_x*(Nxx*S4_x*detJ*wij + Nxy*S4_y*detJ*wij) + S4_y*(Nxy*S4_x*detJ*wij + Nyy*S4_y*detJ*wij))
                    k += 1
                    KGv[k] += r23*r33*(S4_x*(Nxx*S4_x*detJ*wij + Nxy*S4_y*detJ*wij) + S4_y*(Nxy*S4_x*detJ*wij + Nyy*S4_y*detJ*wij))
                    k += 1
                    KGv[k] += r33**2*(S4_x*(Nxx*S4_x*detJ*wij + Nxy*S4_y*detJ*wij) + S4_y*(Nxy*S4_x*detJ*wij + Nyy*S4_y*detJ*wij))

            # NOTE eccentricity geometric stiffness of the rigid offsets,
            #      with the forces of the given stress resultants at the
            #      nodes of the flat element, area times the slopes at the
            #      centroid, the exact integral of the bilinear slopes
            warped = _warp_z(&self.probe.xe[0], self.warp_offsets, zw)
            if warped:
                for a in range(4):
                    fw[6*a + 0] = self.area*(Scx[a]*Nxx + Scy[a]*Nxy)
                    fw[6*a + 1] = self.area*(Scx[a]*Nxy + Scy[a]*Nyy)
                    fw[6*a + 2] = 0.
                Rw[0] = self.r11
                Rw[1] = self.r12
                Rw[2] = self.r13
                Rw[3] = self.r21
                Rw[4] = self.r22
                Rw[5] = self.r23
                Rw[6] = self.r31
                Rw[7] = self.r32
                Rw[8] = self.r33
                _warp_KE(zw, fw, Rw, KEw)
                for iw in range(36):
                    KGv[self.init_k_KG + 144 + iw] += KEw[iw]


    cpdef void update_M(Quad4R self,
                        long [::1] Mr,
                        long [::1] Mc,
                        double [::1] Mv,
                        ShellProp prop,
                        int mtype=0,
                        ):
        r"""Update sparse vectors for mass matrix M

        Different integration schemes are available by means of the ``mtype``
        parameter.

        The element matrix is computed from ``intrho``, ``intrhoz`` and
        ``intrhoz2`` of ``prop``, the coordinates ``probe.xe`` and the
        ``area``, rotated to global coordinates, and its values are added to
        ``Mv`` at the positions starting at ``init_k_M``; the global row and
        column indices are always written to ``Mr`` and ``Mc`` at the same
        positions. ``mtype=0`` and ``mtype=2`` write ``M_SPARSE_SIZE``
        entries and ``mtype=1`` writes the first 120 of them. Before this
        method is called, :meth:`.update_rotation_matrix` and
        :meth:`.update_probe_xe` must have been used.

        Parameters
        ----------
        Mr : np.array
            Array to store row positions of sparse values
            1-D array of integers (``INT``) of size at least ``init_k_M +
            M_SPARSE_SIZE``.
        Mc : np.array
            Array to store column positions of sparse values
            Same type and size as ``Mr``.
        Mv : np.array
            Array to store sparse values
            1-D array of ``DOUBLE``, same size as ``Mr``. The values are added
            to the existing ones. Dimension of mass between translations, of
            mass times length between a translation and a rotation, and of
            mass times length squared between rotations.
        prop : :class:`.ShellProp` object
            Shell property object, from which ``intrho``, ``intrhoz`` and
            ``intrhoz2`` are read.
        mtype : int, optional
            0 for consistent mass matrix
            1 for lumped mass matrix
            2 for reduced integration mass matrix
            Default ``0``.

        Raises
        ------
        ValueError
            If ``mtype`` is not 0, 1 or 2.

        Notes
        -----
        The three mass matrices follow from the kinetic energy of Brockman
        1987 [1]_, Eq. (20), and have the form of its Eq. (21). With the
        displacements `u + z r_y`, `v - z r_x` and `w` of the Notes of
        :mod:`pyfe3d.quad4r`, the kinetic energy integrated through the
        thickness,

        .. math::

            T = \frac{1}{2} \int_A \left[ intrho (\dot{u}^2 + \dot{v}^2 +
            \dot{w}^2) + 2\, intrhoz (\dot{u} \dot{r}_y - \dot{v} \dot{r}_x)
            + intrhoz2 (\dot{r}_x^2 + \dot{r}_y^2) \right] dA

        where ``intrho``, ``intrhoz`` and ``intrhoz2`` include the offset,
        gives, for the nodes `i` and `j`, in element coordinates,

        .. math::

            M_{ij} = h_{ij} \begin{bmatrix}
                intrho & 0 & 0 & 0 & intrhoz & 0 \\
                0 & intrho & 0 & -intrhoz & 0 & 0 \\
                0 & 0 & intrho & 0 & 0 & 0 \\
                0 & -intrhoz & 0 & intrhoz2 & 0 & 0 \\
                intrhoz & 0 & 0 & 0 & intrhoz2 & 0 \\
                0 & 0 & 0 & 0 & 0 & 0
            \end{bmatrix}, \qquad h_{ij} = \int_A S_i S_j dA

        for the DOF `u, v, w, r_x, r_y, r_z`, the drilling rotation carrying
        no inertia. The matrix is rotated to global coordinates as in
        :meth:`.update_KC0`. The integrals `h_{ij}` are computed with:

        - ``mtype=0``: two-point Gauss-Legendre quadrature in each direction;
        - ``mtype=1``: two-point Gauss-Lobatto quadrature in each direction,
          whose points are the nodes, the lumped mass matrix of Brockman
          [1]_, giving `h_{ii}` equal to the Jacobian determinant at node `i`
          and `h_{ij} = 0` for `i \neq j`;
        - ``mtype=2``: a single point at the centroid, where every `S_i =
          1/4`, giving `h_{ij} = A/16` for all `i, j`.

        For ``mtype=0`` the quadrature is exact. With ``mtype=2`` the
        element mass is placed at the average of the node positions, which
        is the centroid of the area only for a parallelogram.

        .. rubric:: Lumped mass matrix

        The Gauss-Lobatto rule of ``mtype=1`` is a nodal quadrature. Applied
        to the whole integrand of `T`, in which `S_i` is 1 at node `i` and 0
        at the other nodes, it couples no different nodes and gives for node
        `i` the block `M_{ii}` above with `h_{ii}`, the coupling through
        ``intrhoz`` included: the matrix is block-diagonal per node and, as
        Brockman [1]_ writes, diagonal only when ``intrhoz`` is zero. The
        weights `h_{ii}` sum to the area `A`, so the lumped matrix has the
        exact mass and the exact first moment of the mass through the
        thickness, ``intrhoz`` `A`, which places the centre of mass at the
        offset from the reference surface. In the plane of the element the
        first moment is exact when `h_{ii} = \int_A S_i dA`, which holds for
        a parallelogram, where `h_{ii} = A/4`, and not for a general
        quadrilateral, whose Jacobian determinant varies. Of the rotary
        inertia, the term ``intrhoz2`` `A` about the axes in the plane of
        the element is exact, while the part from the distribution of
        ``intrho`` over the area, about any axis, is concentrated at the
        nodes: 3 times the exact value about the centroid of a
        parallelogram, an error that decreases with the size of the elements.

        With ``warp_offsets = 1`` and a warped element the matrix is
        `\boldsymbol{J}^\top \boldsymbol{M} \boldsymbol{J}`, see the Notes of
        :class:`.Quad4R`, which keeps the form above for each pair of nodes `i`
        and `j` with ``intrhoz`` replaced by ``intrhoz - intrho*z_j`` in the
        entries between the translations of `i` and the rotations of `j`, and
        ``intrhoz2`` by ``intrhoz2 - intrhoz*(z_i + z_j) + intrho*z_i*z_j`` in
        those between the rotations, `z_i` being the heights of the nodes above
        the flat element (derived here, checked in
        ``theory/deriving_equations/quad4r/verify_quad4r_warp_offsets.py``):
        the moments of the mass about the warped nodes. The numbers of entries
        do not change, and the lumped matrix remains block-diagonal per node.

        References
        ----------
        .. [1] Brockman, R. A., Dynamics of the Bilinear Mindlin Plate Element,
           International Journal for Numerical Methods in Engineering, 24(12),
           2343-2356, 1987. https://doi.org/10.1002/nme.1620241208

        """
        cdef double zw[4]
        cdef double mz1, mz2, mz3, mz4
        cdef double mzz11, mzz12, mzz13, mzz14, mzz22, mzz23, mzz24, mzz33, mzz34, mzz44
        cdef int c1, c2, c3, c4, i, j, k
        cdef double intrho, intrhoz, intrhoz2
        cdef double r11, r12, r13, r21, r22, r23, r31, r32, r33
        cdef double x1, x2, x3, x4
        cdef double y1, y2, y3, y4
        cdef double h11, h12, h13, h14, h22, h23, h24, h33, h34, h44, valH1
        cdef double xi, eta, wij, detJ, S1, S2, S3, S4
        cdef double points[2]

        # NOTE checked once per call, before the code without the GIL
        if mtype != 0 and mtype != 1 and mtype != 2:
            raise ValueError('mtype=%d is not supported by Quad4R.update_M, '
                             'use 0 (consistent), 1 (lumped) or 2 (reduced '
                             'integration)' % mtype)

        with nogil:
            intrho = prop.intrho
            intrhoz = prop.intrhoz
            intrhoz2 = prop.intrhoz2

            # NOTE rigid offsets of a warped element, J^T M J, see the Notes
            _warp_z(&self.probe.xe[0], self.warp_offsets, zw)
            mz1 = intrhoz - intrho*zw[0]
            mz2 = intrhoz - intrho*zw[1]
            mz3 = intrhoz - intrho*zw[2]
            mz4 = intrhoz - intrho*zw[3]
            mzz11 = intrhoz2 - intrhoz*(zw[0] + zw[0]) + intrho*zw[0]*zw[0]
            mzz12 = intrhoz2 - intrhoz*(zw[0] + zw[1]) + intrho*zw[0]*zw[1]
            mzz13 = intrhoz2 - intrhoz*(zw[0] + zw[2]) + intrho*zw[0]*zw[2]
            mzz14 = intrhoz2 - intrhoz*(zw[0] + zw[3]) + intrho*zw[0]*zw[3]
            mzz22 = intrhoz2 - intrhoz*(zw[1] + zw[1]) + intrho*zw[1]*zw[1]
            mzz23 = intrhoz2 - intrhoz*(zw[1] + zw[2]) + intrho*zw[1]*zw[2]
            mzz24 = intrhoz2 - intrhoz*(zw[1] + zw[3]) + intrho*zw[1]*zw[3]
            mzz33 = intrhoz2 - intrhoz*(zw[2] + zw[2]) + intrho*zw[2]*zw[2]
            mzz34 = intrhoz2 - intrhoz*(zw[2] + zw[3]) + intrho*zw[2]*zw[3]
            mzz44 = intrhoz2 - intrhoz*(zw[3] + zw[3]) + intrho*zw[3]*zw[3]

            valH1 = 0.0625*self.area

            # NOTE ignoring z in local coordinates
            x1 = self.probe.xe[0]
            y1 = self.probe.xe[1]
            # z1 = self.probe.xe[2]
            x2 = self.probe.xe[3]
            y2 = self.probe.xe[4]
            # z2 = self.probe.xe[5]
            x3 = self.probe.xe[6]
            y3 = self.probe.xe[7]
            # z3 = self.probe.xe[8]
            x4 = self.probe.xe[9]
            y4 = self.probe.xe[10]
            # z4 = self.probe.xe[11]

            # local to global transformation
            r11 = self.r11
            r12 = self.r12
            r13 = self.r13
            r21 = self.r21
            r22 = self.r22
            r23 = self.r23
            r31 = self.r31
            r32 = self.r32
            r33 = self.r33

            # positions the global matrices
            c1 = self.c1
            c2 = self.c2
            c3 = self.c3
            c4 = self.c4

            if mtype == 0: # M_cons consistent mass matrix, using two-point Gauss-Legendre quadrature
                k = self.init_k_M
                Mr[k] = 0+c1
                Mc[k] = 0+c1
                k += 1
                Mr[k] = 0+c1
                Mc[k] = 1+c1
                k += 1
                Mr[k] = 0+c1
                Mc[k] = 2+c1
                k += 1
                Mr[k] = 0+c1
                Mc[k] = 4+c1
                k += 1
                Mr[k] = 0+c1
                Mc[k] = 5+c1
                k += 1
                Mr[k] = 0+c1
                Mc[k] = 0+c2
                k += 1
                Mr[k] = 0+c1
                Mc[k] = 1+c2
                k += 1
                Mr[k] = 0+c1
                Mc[k] = 2+c2
                k += 1
                Mr[k] = 0+c1
                Mc[k] = 4+c2
                k += 1
                Mr[k] = 0+c1
                Mc[k] = 5+c2
                k += 1
                Mr[k] = 0+c1
                Mc[k] = 0+c3
                k += 1
                Mr[k] = 0+c1
                Mc[k] = 1+c3
                k += 1
                Mr[k] = 0+c1
                Mc[k] = 2+c3
                k += 1
                Mr[k] = 0+c1
                Mc[k] = 4+c3
                k += 1
                Mr[k] = 0+c1
                Mc[k] = 5+c3
                k += 1
                Mr[k] = 0+c1
                Mc[k] = 0+c4
                k += 1
                Mr[k] = 0+c1
                Mc[k] = 1+c4
                k += 1
                Mr[k] = 0+c1
                Mc[k] = 2+c4
                k += 1
                Mr[k] = 0+c1
                Mc[k] = 4+c4
                k += 1
                Mr[k] = 0+c1
                Mc[k] = 5+c4
                k += 1
                Mr[k] = 1+c1
                Mc[k] = 0+c1
                k += 1
                Mr[k] = 1+c1
                Mc[k] = 1+c1
                k += 1
                Mr[k] = 1+c1
                Mc[k] = 2+c1
                k += 1
                Mr[k] = 1+c1
                Mc[k] = 3+c1
                k += 1
                Mr[k] = 1+c1
                Mc[k] = 5+c1
                k += 1
                Mr[k] = 1+c1
                Mc[k] = 0+c2
                k += 1
                Mr[k] = 1+c1
                Mc[k] = 1+c2
                k += 1
                Mr[k] = 1+c1
                Mc[k] = 2+c2
                k += 1
                Mr[k] = 1+c1
                Mc[k] = 3+c2
                k += 1
                Mr[k] = 1+c1
                Mc[k] = 5+c2
                k += 1
                Mr[k] = 1+c1
                Mc[k] = 0+c3
                k += 1
                Mr[k] = 1+c1
                Mc[k] = 1+c3
                k += 1
                Mr[k] = 1+c1
                Mc[k] = 2+c3
                k += 1
                Mr[k] = 1+c1
                Mc[k] = 3+c3
                k += 1
                Mr[k] = 1+c1
                Mc[k] = 5+c3
                k += 1
                Mr[k] = 1+c1
                Mc[k] = 0+c4
                k += 1
                Mr[k] = 1+c1
                Mc[k] = 1+c4
                k += 1
                Mr[k] = 1+c1
                Mc[k] = 2+c4
                k += 1
                Mr[k] = 1+c1
                Mc[k] = 3+c4
                k += 1
                Mr[k] = 1+c1
                Mc[k] = 5+c4
                k += 1
                Mr[k] = 2+c1
                Mc[k] = 0+c1
                k += 1
                Mr[k] = 2+c1
                Mc[k] = 1+c1
                k += 1
                Mr[k] = 2+c1
                Mc[k] = 2+c1
                k += 1
                Mr[k] = 2+c1
                Mc[k] = 3+c1
                k += 1
                Mr[k] = 2+c1
                Mc[k] = 4+c1
                k += 1
                Mr[k] = 2+c1
                Mc[k] = 0+c2
                k += 1
                Mr[k] = 2+c1
                Mc[k] = 1+c2
                k += 1
                Mr[k] = 2+c1
                Mc[k] = 2+c2
                k += 1
                Mr[k] = 2+c1
                Mc[k] = 3+c2
                k += 1
                Mr[k] = 2+c1
                Mc[k] = 4+c2
                k += 1
                Mr[k] = 2+c1
                Mc[k] = 0+c3
                k += 1
                Mr[k] = 2+c1
                Mc[k] = 1+c3
                k += 1
                Mr[k] = 2+c1
                Mc[k] = 2+c3
                k += 1
                Mr[k] = 2+c1
                Mc[k] = 3+c3
                k += 1
                Mr[k] = 2+c1
                Mc[k] = 4+c3
                k += 1
                Mr[k] = 2+c1
                Mc[k] = 0+c4
                k += 1
                Mr[k] = 2+c1
                Mc[k] = 1+c4
                k += 1
                Mr[k] = 2+c1
                Mc[k] = 2+c4
                k += 1
                Mr[k] = 2+c1
                Mc[k] = 3+c4
                k += 1
                Mr[k] = 2+c1
                Mc[k] = 4+c4
                k += 1
                Mr[k] = 3+c1
                Mc[k] = 1+c1
                k += 1
                Mr[k] = 3+c1
                Mc[k] = 2+c1
                k += 1
                Mr[k] = 3+c1
                Mc[k] = 3+c1
                k += 1
                Mr[k] = 3+c1
                Mc[k] = 4+c1
                k += 1
                Mr[k] = 3+c1
                Mc[k] = 5+c1
                k += 1
                Mr[k] = 3+c1
                Mc[k] = 1+c2
                k += 1
                Mr[k] = 3+c1
                Mc[k] = 2+c2
                k += 1
                Mr[k] = 3+c1
                Mc[k] = 3+c2
                k += 1
                Mr[k] = 3+c1
                Mc[k] = 4+c2
                k += 1
                Mr[k] = 3+c1
                Mc[k] = 5+c2
                k += 1
                Mr[k] = 3+c1
                Mc[k] = 1+c3
                k += 1
                Mr[k] = 3+c1
                Mc[k] = 2+c3
                k += 1
                Mr[k] = 3+c1
                Mc[k] = 3+c3
                k += 1
                Mr[k] = 3+c1
                Mc[k] = 4+c3
                k += 1
                Mr[k] = 3+c1
                Mc[k] = 5+c3
                k += 1
                Mr[k] = 3+c1
                Mc[k] = 1+c4
                k += 1
                Mr[k] = 3+c1
                Mc[k] = 2+c4
                k += 1
                Mr[k] = 3+c1
                Mc[k] = 3+c4
                k += 1
                Mr[k] = 3+c1
                Mc[k] = 4+c4
                k += 1
                Mr[k] = 3+c1
                Mc[k] = 5+c4
                k += 1
                Mr[k] = 4+c1
                Mc[k] = 0+c1
                k += 1
                Mr[k] = 4+c1
                Mc[k] = 2+c1
                k += 1
                Mr[k] = 4+c1
                Mc[k] = 3+c1
                k += 1
                Mr[k] = 4+c1
                Mc[k] = 4+c1
                k += 1
                Mr[k] = 4+c1
                Mc[k] = 5+c1
                k += 1
                Mr[k] = 4+c1
                Mc[k] = 0+c2
                k += 1
                Mr[k] = 4+c1
                Mc[k] = 2+c2
                k += 1
                Mr[k] = 4+c1
                Mc[k] = 3+c2
                k += 1
                Mr[k] = 4+c1
                Mc[k] = 4+c2
                k += 1
                Mr[k] = 4+c1
                Mc[k] = 5+c2
                k += 1
                Mr[k] = 4+c1
                Mc[k] = 0+c3
                k += 1
                Mr[k] = 4+c1
                Mc[k] = 2+c3
                k += 1
                Mr[k] = 4+c1
                Mc[k] = 3+c3
                k += 1
                Mr[k] = 4+c1
                Mc[k] = 4+c3
                k += 1
                Mr[k] = 4+c1
                Mc[k] = 5+c3
                k += 1
                Mr[k] = 4+c1
                Mc[k] = 0+c4
                k += 1
                Mr[k] = 4+c1
                Mc[k] = 2+c4
                k += 1
                Mr[k] = 4+c1
                Mc[k] = 3+c4
                k += 1
                Mr[k] = 4+c1
                Mc[k] = 4+c4
                k += 1
                Mr[k] = 4+c1
                Mc[k] = 5+c4
                k += 1
                Mr[k] = 5+c1
                Mc[k] = 0+c1
                k += 1
                Mr[k] = 5+c1
                Mc[k] = 1+c1
                k += 1
                Mr[k] = 5+c1
                Mc[k] = 3+c1
                k += 1
                Mr[k] = 5+c1
                Mc[k] = 4+c1
                k += 1
                Mr[k] = 5+c1
                Mc[k] = 5+c1
                k += 1
                Mr[k] = 5+c1
                Mc[k] = 0+c2
                k += 1
                Mr[k] = 5+c1
                Mc[k] = 1+c2
                k += 1
                Mr[k] = 5+c1
                Mc[k] = 3+c2
                k += 1
                Mr[k] = 5+c1
                Mc[k] = 4+c2
                k += 1
                Mr[k] = 5+c1
                Mc[k] = 5+c2
                k += 1
                Mr[k] = 5+c1
                Mc[k] = 0+c3
                k += 1
                Mr[k] = 5+c1
                Mc[k] = 1+c3
                k += 1
                Mr[k] = 5+c1
                Mc[k] = 3+c3
                k += 1
                Mr[k] = 5+c1
                Mc[k] = 4+c3
                k += 1
                Mr[k] = 5+c1
                Mc[k] = 5+c3
                k += 1
                Mr[k] = 5+c1
                Mc[k] = 0+c4
                k += 1
                Mr[k] = 5+c1
                Mc[k] = 1+c4
                k += 1
                Mr[k] = 5+c1
                Mc[k] = 3+c4
                k += 1
                Mr[k] = 5+c1
                Mc[k] = 4+c4
                k += 1
                Mr[k] = 5+c1
                Mc[k] = 5+c4
                k += 1
                Mr[k] = 0+c2
                Mc[k] = 0+c1
                k += 1
                Mr[k] = 0+c2
                Mc[k] = 1+c1
                k += 1
                Mr[k] = 0+c2
                Mc[k] = 2+c1
                k += 1
                Mr[k] = 0+c2
                Mc[k] = 4+c1
                k += 1
                Mr[k] = 0+c2
                Mc[k] = 5+c1
                k += 1
                Mr[k] = 0+c2
                Mc[k] = 0+c2
                k += 1
                Mr[k] = 0+c2
                Mc[k] = 1+c2
                k += 1
                Mr[k] = 0+c2
                Mc[k] = 2+c2
                k += 1
                Mr[k] = 0+c2
                Mc[k] = 4+c2
                k += 1
                Mr[k] = 0+c2
                Mc[k] = 5+c2
                k += 1
                Mr[k] = 0+c2
                Mc[k] = 0+c3
                k += 1
                Mr[k] = 0+c2
                Mc[k] = 1+c3
                k += 1
                Mr[k] = 0+c2
                Mc[k] = 2+c3
                k += 1
                Mr[k] = 0+c2
                Mc[k] = 4+c3
                k += 1
                Mr[k] = 0+c2
                Mc[k] = 5+c3
                k += 1
                Mr[k] = 0+c2
                Mc[k] = 0+c4
                k += 1
                Mr[k] = 0+c2
                Mc[k] = 1+c4
                k += 1
                Mr[k] = 0+c2
                Mc[k] = 2+c4
                k += 1
                Mr[k] = 0+c2
                Mc[k] = 4+c4
                k += 1
                Mr[k] = 0+c2
                Mc[k] = 5+c4
                k += 1
                Mr[k] = 1+c2
                Mc[k] = 0+c1
                k += 1
                Mr[k] = 1+c2
                Mc[k] = 1+c1
                k += 1
                Mr[k] = 1+c2
                Mc[k] = 2+c1
                k += 1
                Mr[k] = 1+c2
                Mc[k] = 3+c1
                k += 1
                Mr[k] = 1+c2
                Mc[k] = 5+c1
                k += 1
                Mr[k] = 1+c2
                Mc[k] = 0+c2
                k += 1
                Mr[k] = 1+c2
                Mc[k] = 1+c2
                k += 1
                Mr[k] = 1+c2
                Mc[k] = 2+c2
                k += 1
                Mr[k] = 1+c2
                Mc[k] = 3+c2
                k += 1
                Mr[k] = 1+c2
                Mc[k] = 5+c2
                k += 1
                Mr[k] = 1+c2
                Mc[k] = 0+c3
                k += 1
                Mr[k] = 1+c2
                Mc[k] = 1+c3
                k += 1
                Mr[k] = 1+c2
                Mc[k] = 2+c3
                k += 1
                Mr[k] = 1+c2
                Mc[k] = 3+c3
                k += 1
                Mr[k] = 1+c2
                Mc[k] = 5+c3
                k += 1
                Mr[k] = 1+c2
                Mc[k] = 0+c4
                k += 1
                Mr[k] = 1+c2
                Mc[k] = 1+c4
                k += 1
                Mr[k] = 1+c2
                Mc[k] = 2+c4
                k += 1
                Mr[k] = 1+c2
                Mc[k] = 3+c4
                k += 1
                Mr[k] = 1+c2
                Mc[k] = 5+c4
                k += 1
                Mr[k] = 2+c2
                Mc[k] = 0+c1
                k += 1
                Mr[k] = 2+c2
                Mc[k] = 1+c1
                k += 1
                Mr[k] = 2+c2
                Mc[k] = 2+c1
                k += 1
                Mr[k] = 2+c2
                Mc[k] = 3+c1
                k += 1
                Mr[k] = 2+c2
                Mc[k] = 4+c1
                k += 1
                Mr[k] = 2+c2
                Mc[k] = 0+c2
                k += 1
                Mr[k] = 2+c2
                Mc[k] = 1+c2
                k += 1
                Mr[k] = 2+c2
                Mc[k] = 2+c2
                k += 1
                Mr[k] = 2+c2
                Mc[k] = 3+c2
                k += 1
                Mr[k] = 2+c2
                Mc[k] = 4+c2
                k += 1
                Mr[k] = 2+c2
                Mc[k] = 0+c3
                k += 1
                Mr[k] = 2+c2
                Mc[k] = 1+c3
                k += 1
                Mr[k] = 2+c2
                Mc[k] = 2+c3
                k += 1
                Mr[k] = 2+c2
                Mc[k] = 3+c3
                k += 1
                Mr[k] = 2+c2
                Mc[k] = 4+c3
                k += 1
                Mr[k] = 2+c2
                Mc[k] = 0+c4
                k += 1
                Mr[k] = 2+c2
                Mc[k] = 1+c4
                k += 1
                Mr[k] = 2+c2
                Mc[k] = 2+c4
                k += 1
                Mr[k] = 2+c2
                Mc[k] = 3+c4
                k += 1
                Mr[k] = 2+c2
                Mc[k] = 4+c4
                k += 1
                Mr[k] = 3+c2
                Mc[k] = 1+c1
                k += 1
                Mr[k] = 3+c2
                Mc[k] = 2+c1
                k += 1
                Mr[k] = 3+c2
                Mc[k] = 3+c1
                k += 1
                Mr[k] = 3+c2
                Mc[k] = 4+c1
                k += 1
                Mr[k] = 3+c2
                Mc[k] = 5+c1
                k += 1
                Mr[k] = 3+c2
                Mc[k] = 1+c2
                k += 1
                Mr[k] = 3+c2
                Mc[k] = 2+c2
                k += 1
                Mr[k] = 3+c2
                Mc[k] = 3+c2
                k += 1
                Mr[k] = 3+c2
                Mc[k] = 4+c2
                k += 1
                Mr[k] = 3+c2
                Mc[k] = 5+c2
                k += 1
                Mr[k] = 3+c2
                Mc[k] = 1+c3
                k += 1
                Mr[k] = 3+c2
                Mc[k] = 2+c3
                k += 1
                Mr[k] = 3+c2
                Mc[k] = 3+c3
                k += 1
                Mr[k] = 3+c2
                Mc[k] = 4+c3
                k += 1
                Mr[k] = 3+c2
                Mc[k] = 5+c3
                k += 1
                Mr[k] = 3+c2
                Mc[k] = 1+c4
                k += 1
                Mr[k] = 3+c2
                Mc[k] = 2+c4
                k += 1
                Mr[k] = 3+c2
                Mc[k] = 3+c4
                k += 1
                Mr[k] = 3+c2
                Mc[k] = 4+c4
                k += 1
                Mr[k] = 3+c2
                Mc[k] = 5+c4
                k += 1
                Mr[k] = 4+c2
                Mc[k] = 0+c1
                k += 1
                Mr[k] = 4+c2
                Mc[k] = 2+c1
                k += 1
                Mr[k] = 4+c2
                Mc[k] = 3+c1
                k += 1
                Mr[k] = 4+c2
                Mc[k] = 4+c1
                k += 1
                Mr[k] = 4+c2
                Mc[k] = 5+c1
                k += 1
                Mr[k] = 4+c2
                Mc[k] = 0+c2
                k += 1
                Mr[k] = 4+c2
                Mc[k] = 2+c2
                k += 1
                Mr[k] = 4+c2
                Mc[k] = 3+c2
                k += 1
                Mr[k] = 4+c2
                Mc[k] = 4+c2
                k += 1
                Mr[k] = 4+c2
                Mc[k] = 5+c2
                k += 1
                Mr[k] = 4+c2
                Mc[k] = 0+c3
                k += 1
                Mr[k] = 4+c2
                Mc[k] = 2+c3
                k += 1
                Mr[k] = 4+c2
                Mc[k] = 3+c3
                k += 1
                Mr[k] = 4+c2
                Mc[k] = 4+c3
                k += 1
                Mr[k] = 4+c2
                Mc[k] = 5+c3
                k += 1
                Mr[k] = 4+c2
                Mc[k] = 0+c4
                k += 1
                Mr[k] = 4+c2
                Mc[k] = 2+c4
                k += 1
                Mr[k] = 4+c2
                Mc[k] = 3+c4
                k += 1
                Mr[k] = 4+c2
                Mc[k] = 4+c4
                k += 1
                Mr[k] = 4+c2
                Mc[k] = 5+c4
                k += 1
                Mr[k] = 5+c2
                Mc[k] = 0+c1
                k += 1
                Mr[k] = 5+c2
                Mc[k] = 1+c1
                k += 1
                Mr[k] = 5+c2
                Mc[k] = 3+c1
                k += 1
                Mr[k] = 5+c2
                Mc[k] = 4+c1
                k += 1
                Mr[k] = 5+c2
                Mc[k] = 5+c1
                k += 1
                Mr[k] = 5+c2
                Mc[k] = 0+c2
                k += 1
                Mr[k] = 5+c2
                Mc[k] = 1+c2
                k += 1
                Mr[k] = 5+c2
                Mc[k] = 3+c2
                k += 1
                Mr[k] = 5+c2
                Mc[k] = 4+c2
                k += 1
                Mr[k] = 5+c2
                Mc[k] = 5+c2
                k += 1
                Mr[k] = 5+c2
                Mc[k] = 0+c3
                k += 1
                Mr[k] = 5+c2
                Mc[k] = 1+c3
                k += 1
                Mr[k] = 5+c2
                Mc[k] = 3+c3
                k += 1
                Mr[k] = 5+c2
                Mc[k] = 4+c3
                k += 1
                Mr[k] = 5+c2
                Mc[k] = 5+c3
                k += 1
                Mr[k] = 5+c2
                Mc[k] = 0+c4
                k += 1
                Mr[k] = 5+c2
                Mc[k] = 1+c4
                k += 1
                Mr[k] = 5+c2
                Mc[k] = 3+c4
                k += 1
                Mr[k] = 5+c2
                Mc[k] = 4+c4
                k += 1
                Mr[k] = 5+c2
                Mc[k] = 5+c4
                k += 1
                Mr[k] = 0+c3
                Mc[k] = 0+c1
                k += 1
                Mr[k] = 0+c3
                Mc[k] = 1+c1
                k += 1
                Mr[k] = 0+c3
                Mc[k] = 2+c1
                k += 1
                Mr[k] = 0+c3
                Mc[k] = 4+c1
                k += 1
                Mr[k] = 0+c3
                Mc[k] = 5+c1
                k += 1
                Mr[k] = 0+c3
                Mc[k] = 0+c2
                k += 1
                Mr[k] = 0+c3
                Mc[k] = 1+c2
                k += 1
                Mr[k] = 0+c3
                Mc[k] = 2+c2
                k += 1
                Mr[k] = 0+c3
                Mc[k] = 4+c2
                k += 1
                Mr[k] = 0+c3
                Mc[k] = 5+c2
                k += 1
                Mr[k] = 0+c3
                Mc[k] = 0+c3
                k += 1
                Mr[k] = 0+c3
                Mc[k] = 1+c3
                k += 1
                Mr[k] = 0+c3
                Mc[k] = 2+c3
                k += 1
                Mr[k] = 0+c3
                Mc[k] = 4+c3
                k += 1
                Mr[k] = 0+c3
                Mc[k] = 5+c3
                k += 1
                Mr[k] = 0+c3
                Mc[k] = 0+c4
                k += 1
                Mr[k] = 0+c3
                Mc[k] = 1+c4
                k += 1
                Mr[k] = 0+c3
                Mc[k] = 2+c4
                k += 1
                Mr[k] = 0+c3
                Mc[k] = 4+c4
                k += 1
                Mr[k] = 0+c3
                Mc[k] = 5+c4
                k += 1
                Mr[k] = 1+c3
                Mc[k] = 0+c1
                k += 1
                Mr[k] = 1+c3
                Mc[k] = 1+c1
                k += 1
                Mr[k] = 1+c3
                Mc[k] = 2+c1
                k += 1
                Mr[k] = 1+c3
                Mc[k] = 3+c1
                k += 1
                Mr[k] = 1+c3
                Mc[k] = 5+c1
                k += 1
                Mr[k] = 1+c3
                Mc[k] = 0+c2
                k += 1
                Mr[k] = 1+c3
                Mc[k] = 1+c2
                k += 1
                Mr[k] = 1+c3
                Mc[k] = 2+c2
                k += 1
                Mr[k] = 1+c3
                Mc[k] = 3+c2
                k += 1
                Mr[k] = 1+c3
                Mc[k] = 5+c2
                k += 1
                Mr[k] = 1+c3
                Mc[k] = 0+c3
                k += 1
                Mr[k] = 1+c3
                Mc[k] = 1+c3
                k += 1
                Mr[k] = 1+c3
                Mc[k] = 2+c3
                k += 1
                Mr[k] = 1+c3
                Mc[k] = 3+c3
                k += 1
                Mr[k] = 1+c3
                Mc[k] = 5+c3
                k += 1
                Mr[k] = 1+c3
                Mc[k] = 0+c4
                k += 1
                Mr[k] = 1+c3
                Mc[k] = 1+c4
                k += 1
                Mr[k] = 1+c3
                Mc[k] = 2+c4
                k += 1
                Mr[k] = 1+c3
                Mc[k] = 3+c4
                k += 1
                Mr[k] = 1+c3
                Mc[k] = 5+c4
                k += 1
                Mr[k] = 2+c3
                Mc[k] = 0+c1
                k += 1
                Mr[k] = 2+c3
                Mc[k] = 1+c1
                k += 1
                Mr[k] = 2+c3
                Mc[k] = 2+c1
                k += 1
                Mr[k] = 2+c3
                Mc[k] = 3+c1
                k += 1
                Mr[k] = 2+c3
                Mc[k] = 4+c1
                k += 1
                Mr[k] = 2+c3
                Mc[k] = 0+c2
                k += 1
                Mr[k] = 2+c3
                Mc[k] = 1+c2
                k += 1
                Mr[k] = 2+c3
                Mc[k] = 2+c2
                k += 1
                Mr[k] = 2+c3
                Mc[k] = 3+c2
                k += 1
                Mr[k] = 2+c3
                Mc[k] = 4+c2
                k += 1
                Mr[k] = 2+c3
                Mc[k] = 0+c3
                k += 1
                Mr[k] = 2+c3
                Mc[k] = 1+c3
                k += 1
                Mr[k] = 2+c3
                Mc[k] = 2+c3
                k += 1
                Mr[k] = 2+c3
                Mc[k] = 3+c3
                k += 1
                Mr[k] = 2+c3
                Mc[k] = 4+c3
                k += 1
                Mr[k] = 2+c3
                Mc[k] = 0+c4
                k += 1
                Mr[k] = 2+c3
                Mc[k] = 1+c4
                k += 1
                Mr[k] = 2+c3
                Mc[k] = 2+c4
                k += 1
                Mr[k] = 2+c3
                Mc[k] = 3+c4
                k += 1
                Mr[k] = 2+c3
                Mc[k] = 4+c4
                k += 1
                Mr[k] = 3+c3
                Mc[k] = 1+c1
                k += 1
                Mr[k] = 3+c3
                Mc[k] = 2+c1
                k += 1
                Mr[k] = 3+c3
                Mc[k] = 3+c1
                k += 1
                Mr[k] = 3+c3
                Mc[k] = 4+c1
                k += 1
                Mr[k] = 3+c3
                Mc[k] = 5+c1
                k += 1
                Mr[k] = 3+c3
                Mc[k] = 1+c2
                k += 1
                Mr[k] = 3+c3
                Mc[k] = 2+c2
                k += 1
                Mr[k] = 3+c3
                Mc[k] = 3+c2
                k += 1
                Mr[k] = 3+c3
                Mc[k] = 4+c2
                k += 1
                Mr[k] = 3+c3
                Mc[k] = 5+c2
                k += 1
                Mr[k] = 3+c3
                Mc[k] = 1+c3
                k += 1
                Mr[k] = 3+c3
                Mc[k] = 2+c3
                k += 1
                Mr[k] = 3+c3
                Mc[k] = 3+c3
                k += 1
                Mr[k] = 3+c3
                Mc[k] = 4+c3
                k += 1
                Mr[k] = 3+c3
                Mc[k] = 5+c3
                k += 1
                Mr[k] = 3+c3
                Mc[k] = 1+c4
                k += 1
                Mr[k] = 3+c3
                Mc[k] = 2+c4
                k += 1
                Mr[k] = 3+c3
                Mc[k] = 3+c4
                k += 1
                Mr[k] = 3+c3
                Mc[k] = 4+c4
                k += 1
                Mr[k] = 3+c3
                Mc[k] = 5+c4
                k += 1
                Mr[k] = 4+c3
                Mc[k] = 0+c1
                k += 1
                Mr[k] = 4+c3
                Mc[k] = 2+c1
                k += 1
                Mr[k] = 4+c3
                Mc[k] = 3+c1
                k += 1
                Mr[k] = 4+c3
                Mc[k] = 4+c1
                k += 1
                Mr[k] = 4+c3
                Mc[k] = 5+c1
                k += 1
                Mr[k] = 4+c3
                Mc[k] = 0+c2
                k += 1
                Mr[k] = 4+c3
                Mc[k] = 2+c2
                k += 1
                Mr[k] = 4+c3
                Mc[k] = 3+c2
                k += 1
                Mr[k] = 4+c3
                Mc[k] = 4+c2
                k += 1
                Mr[k] = 4+c3
                Mc[k] = 5+c2
                k += 1
                Mr[k] = 4+c3
                Mc[k] = 0+c3
                k += 1
                Mr[k] = 4+c3
                Mc[k] = 2+c3
                k += 1
                Mr[k] = 4+c3
                Mc[k] = 3+c3
                k += 1
                Mr[k] = 4+c3
                Mc[k] = 4+c3
                k += 1
                Mr[k] = 4+c3
                Mc[k] = 5+c3
                k += 1
                Mr[k] = 4+c3
                Mc[k] = 0+c4
                k += 1
                Mr[k] = 4+c3
                Mc[k] = 2+c4
                k += 1
                Mr[k] = 4+c3
                Mc[k] = 3+c4
                k += 1
                Mr[k] = 4+c3
                Mc[k] = 4+c4
                k += 1
                Mr[k] = 4+c3
                Mc[k] = 5+c4
                k += 1
                Mr[k] = 5+c3
                Mc[k] = 0+c1
                k += 1
                Mr[k] = 5+c3
                Mc[k] = 1+c1
                k += 1
                Mr[k] = 5+c3
                Mc[k] = 3+c1
                k += 1
                Mr[k] = 5+c3
                Mc[k] = 4+c1
                k += 1
                Mr[k] = 5+c3
                Mc[k] = 5+c1
                k += 1
                Mr[k] = 5+c3
                Mc[k] = 0+c2
                k += 1
                Mr[k] = 5+c3
                Mc[k] = 1+c2
                k += 1
                Mr[k] = 5+c3
                Mc[k] = 3+c2
                k += 1
                Mr[k] = 5+c3
                Mc[k] = 4+c2
                k += 1
                Mr[k] = 5+c3
                Mc[k] = 5+c2
                k += 1
                Mr[k] = 5+c3
                Mc[k] = 0+c3
                k += 1
                Mr[k] = 5+c3
                Mc[k] = 1+c3
                k += 1
                Mr[k] = 5+c3
                Mc[k] = 3+c3
                k += 1
                Mr[k] = 5+c3
                Mc[k] = 4+c3
                k += 1
                Mr[k] = 5+c3
                Mc[k] = 5+c3
                k += 1
                Mr[k] = 5+c3
                Mc[k] = 0+c4
                k += 1
                Mr[k] = 5+c3
                Mc[k] = 1+c4
                k += 1
                Mr[k] = 5+c3
                Mc[k] = 3+c4
                k += 1
                Mr[k] = 5+c3
                Mc[k] = 4+c4
                k += 1
                Mr[k] = 5+c3
                Mc[k] = 5+c4
                k += 1
                Mr[k] = 0+c4
                Mc[k] = 0+c1
                k += 1
                Mr[k] = 0+c4
                Mc[k] = 1+c1
                k += 1
                Mr[k] = 0+c4
                Mc[k] = 2+c1
                k += 1
                Mr[k] = 0+c4
                Mc[k] = 4+c1
                k += 1
                Mr[k] = 0+c4
                Mc[k] = 5+c1
                k += 1
                Mr[k] = 0+c4
                Mc[k] = 0+c2
                k += 1
                Mr[k] = 0+c4
                Mc[k] = 1+c2
                k += 1
                Mr[k] = 0+c4
                Mc[k] = 2+c2
                k += 1
                Mr[k] = 0+c4
                Mc[k] = 4+c2
                k += 1
                Mr[k] = 0+c4
                Mc[k] = 5+c2
                k += 1
                Mr[k] = 0+c4
                Mc[k] = 0+c3
                k += 1
                Mr[k] = 0+c4
                Mc[k] = 1+c3
                k += 1
                Mr[k] = 0+c4
                Mc[k] = 2+c3
                k += 1
                Mr[k] = 0+c4
                Mc[k] = 4+c3
                k += 1
                Mr[k] = 0+c4
                Mc[k] = 5+c3
                k += 1
                Mr[k] = 0+c4
                Mc[k] = 0+c4
                k += 1
                Mr[k] = 0+c4
                Mc[k] = 1+c4
                k += 1
                Mr[k] = 0+c4
                Mc[k] = 2+c4
                k += 1
                Mr[k] = 0+c4
                Mc[k] = 4+c4
                k += 1
                Mr[k] = 0+c4
                Mc[k] = 5+c4
                k += 1
                Mr[k] = 1+c4
                Mc[k] = 0+c1
                k += 1
                Mr[k] = 1+c4
                Mc[k] = 1+c1
                k += 1
                Mr[k] = 1+c4
                Mc[k] = 2+c1
                k += 1
                Mr[k] = 1+c4
                Mc[k] = 3+c1
                k += 1
                Mr[k] = 1+c4
                Mc[k] = 5+c1
                k += 1
                Mr[k] = 1+c4
                Mc[k] = 0+c2
                k += 1
                Mr[k] = 1+c4
                Mc[k] = 1+c2
                k += 1
                Mr[k] = 1+c4
                Mc[k] = 2+c2
                k += 1
                Mr[k] = 1+c4
                Mc[k] = 3+c2
                k += 1
                Mr[k] = 1+c4
                Mc[k] = 5+c2
                k += 1
                Mr[k] = 1+c4
                Mc[k] = 0+c3
                k += 1
                Mr[k] = 1+c4
                Mc[k] = 1+c3
                k += 1
                Mr[k] = 1+c4
                Mc[k] = 2+c3
                k += 1
                Mr[k] = 1+c4
                Mc[k] = 3+c3
                k += 1
                Mr[k] = 1+c4
                Mc[k] = 5+c3
                k += 1
                Mr[k] = 1+c4
                Mc[k] = 0+c4
                k += 1
                Mr[k] = 1+c4
                Mc[k] = 1+c4
                k += 1
                Mr[k] = 1+c4
                Mc[k] = 2+c4
                k += 1
                Mr[k] = 1+c4
                Mc[k] = 3+c4
                k += 1
                Mr[k] = 1+c4
                Mc[k] = 5+c4
                k += 1
                Mr[k] = 2+c4
                Mc[k] = 0+c1
                k += 1
                Mr[k] = 2+c4
                Mc[k] = 1+c1
                k += 1
                Mr[k] = 2+c4
                Mc[k] = 2+c1
                k += 1
                Mr[k] = 2+c4
                Mc[k] = 3+c1
                k += 1
                Mr[k] = 2+c4
                Mc[k] = 4+c1
                k += 1
                Mr[k] = 2+c4
                Mc[k] = 0+c2
                k += 1
                Mr[k] = 2+c4
                Mc[k] = 1+c2
                k += 1
                Mr[k] = 2+c4
                Mc[k] = 2+c2
                k += 1
                Mr[k] = 2+c4
                Mc[k] = 3+c2
                k += 1
                Mr[k] = 2+c4
                Mc[k] = 4+c2
                k += 1
                Mr[k] = 2+c4
                Mc[k] = 0+c3
                k += 1
                Mr[k] = 2+c4
                Mc[k] = 1+c3
                k += 1
                Mr[k] = 2+c4
                Mc[k] = 2+c3
                k += 1
                Mr[k] = 2+c4
                Mc[k] = 3+c3
                k += 1
                Mr[k] = 2+c4
                Mc[k] = 4+c3
                k += 1
                Mr[k] = 2+c4
                Mc[k] = 0+c4
                k += 1
                Mr[k] = 2+c4
                Mc[k] = 1+c4
                k += 1
                Mr[k] = 2+c4
                Mc[k] = 2+c4
                k += 1
                Mr[k] = 2+c4
                Mc[k] = 3+c4
                k += 1
                Mr[k] = 2+c4
                Mc[k] = 4+c4
                k += 1
                Mr[k] = 3+c4
                Mc[k] = 1+c1
                k += 1
                Mr[k] = 3+c4
                Mc[k] = 2+c1
                k += 1
                Mr[k] = 3+c4
                Mc[k] = 3+c1
                k += 1
                Mr[k] = 3+c4
                Mc[k] = 4+c1
                k += 1
                Mr[k] = 3+c4
                Mc[k] = 5+c1
                k += 1
                Mr[k] = 3+c4
                Mc[k] = 1+c2
                k += 1
                Mr[k] = 3+c4
                Mc[k] = 2+c2
                k += 1
                Mr[k] = 3+c4
                Mc[k] = 3+c2
                k += 1
                Mr[k] = 3+c4
                Mc[k] = 4+c2
                k += 1
                Mr[k] = 3+c4
                Mc[k] = 5+c2
                k += 1
                Mr[k] = 3+c4
                Mc[k] = 1+c3
                k += 1
                Mr[k] = 3+c4
                Mc[k] = 2+c3
                k += 1
                Mr[k] = 3+c4
                Mc[k] = 3+c3
                k += 1
                Mr[k] = 3+c4
                Mc[k] = 4+c3
                k += 1
                Mr[k] = 3+c4
                Mc[k] = 5+c3
                k += 1
                Mr[k] = 3+c4
                Mc[k] = 1+c4
                k += 1
                Mr[k] = 3+c4
                Mc[k] = 2+c4
                k += 1
                Mr[k] = 3+c4
                Mc[k] = 3+c4
                k += 1
                Mr[k] = 3+c4
                Mc[k] = 4+c4
                k += 1
                Mr[k] = 3+c4
                Mc[k] = 5+c4
                k += 1
                Mr[k] = 4+c4
                Mc[k] = 0+c1
                k += 1
                Mr[k] = 4+c4
                Mc[k] = 2+c1
                k += 1
                Mr[k] = 4+c4
                Mc[k] = 3+c1
                k += 1
                Mr[k] = 4+c4
                Mc[k] = 4+c1
                k += 1
                Mr[k] = 4+c4
                Mc[k] = 5+c1
                k += 1
                Mr[k] = 4+c4
                Mc[k] = 0+c2
                k += 1
                Mr[k] = 4+c4
                Mc[k] = 2+c2
                k += 1
                Mr[k] = 4+c4
                Mc[k] = 3+c2
                k += 1
                Mr[k] = 4+c4
                Mc[k] = 4+c2
                k += 1
                Mr[k] = 4+c4
                Mc[k] = 5+c2
                k += 1
                Mr[k] = 4+c4
                Mc[k] = 0+c3
                k += 1
                Mr[k] = 4+c4
                Mc[k] = 2+c3
                k += 1
                Mr[k] = 4+c4
                Mc[k] = 3+c3
                k += 1
                Mr[k] = 4+c4
                Mc[k] = 4+c3
                k += 1
                Mr[k] = 4+c4
                Mc[k] = 5+c3
                k += 1
                Mr[k] = 4+c4
                Mc[k] = 0+c4
                k += 1
                Mr[k] = 4+c4
                Mc[k] = 2+c4
                k += 1
                Mr[k] = 4+c4
                Mc[k] = 3+c4
                k += 1
                Mr[k] = 4+c4
                Mc[k] = 4+c4
                k += 1
                Mr[k] = 4+c4
                Mc[k] = 5+c4
                k += 1
                Mr[k] = 5+c4
                Mc[k] = 0+c1
                k += 1
                Mr[k] = 5+c4
                Mc[k] = 1+c1
                k += 1
                Mr[k] = 5+c4
                Mc[k] = 3+c1
                k += 1
                Mr[k] = 5+c4
                Mc[k] = 4+c1
                k += 1
                Mr[k] = 5+c4
                Mc[k] = 5+c1
                k += 1
                Mr[k] = 5+c4
                Mc[k] = 0+c2
                k += 1
                Mr[k] = 5+c4
                Mc[k] = 1+c2
                k += 1
                Mr[k] = 5+c4
                Mc[k] = 3+c2
                k += 1
                Mr[k] = 5+c4
                Mc[k] = 4+c2
                k += 1
                Mr[k] = 5+c4
                Mc[k] = 5+c2
                k += 1
                Mr[k] = 5+c4
                Mc[k] = 0+c3
                k += 1
                Mr[k] = 5+c4
                Mc[k] = 1+c3
                k += 1
                Mr[k] = 5+c4
                Mc[k] = 3+c3
                k += 1
                Mr[k] = 5+c4
                Mc[k] = 4+c3
                k += 1
                Mr[k] = 5+c4
                Mc[k] = 5+c3
                k += 1
                Mr[k] = 5+c4
                Mc[k] = 0+c4
                k += 1
                Mr[k] = 5+c4
                Mc[k] = 1+c4
                k += 1
                Mr[k] = 5+c4
                Mc[k] = 3+c4
                k += 1
                Mr[k] = 5+c4
                Mc[k] = 4+c4
                k += 1
                Mr[k] = 5+c4
                Mc[k] = 5+c4

                # NOTE two-point Gauss-Legendre quadrature
                wij = 1.
                points[0] = -0.5773502691896257645092
                points[1] = +0.5773502691896257645092
                h11 = 0.
                h12 = 0.
                h13 = 0.
                h14 = 0.
                h22 = 0.
                h23 = 0.
                h24 = 0.
                h33 = 0.
                h34 = 0.
                h44 = 0.
                for i in range(2):
                    xi = points[i]
                    for j in range(2):
                        eta = points[j]
                        detJ = (-2*x1 + 2*x2 + (eta + 1)*(x1 - x2 + x3 - x4))*(-2*y1 + 2*y4 + (xi + 1)*(y1 - y2) + (xi + 1)*(y3 - y4))/16. - (-2*y1 + 2*y2 + (eta + 1)*(y1 - y2 + y3 - y4))*(-2*x1 + 2*x4 + (x1 - x2)*(xi + 1) + (x3 - x4)*(xi + 1))/16.
                        S1 = eta*xi/4. - eta/4. - xi/4. + 1/4.
                        S2 = -eta*xi/4. - eta/4. + xi/4. + 1/4.
                        S3 = eta*xi/4. + eta/4. + xi/4. + 1/4.
                        S4 = -eta*xi/4. + eta/4. - xi/4. + 1/4.
                        h11 += S1**2*detJ*wij
                        h12 += S1*S2*detJ*wij
                        h13 += S1*S3*detJ*wij
                        h14 += S1*S4*detJ*wij
                        h22 += S2**2*detJ*wij
                        h23 += S2*S3*detJ*wij
                        h24 += S2*S4*detJ*wij
                        h33 += S3**2*detJ*wij
                        h34 += S3*S4*detJ*wij
                        h44 += S4**2*detJ*wij

                k = self.init_k_M
                Mv[k] += h11*intrho*r11**2 + h11*intrho*r12**2 + h11*intrho*r13**2
                k += 1
                Mv[k] += h11*intrho*r11*r21 + h11*intrho*r12*r22 + h11*intrho*r13*r23
                k += 1
                Mv[k] += h11*intrho*r11*r31 + h11*intrho*r12*r32 + h11*intrho*r13*r33
                k += 1
                Mv[k] += h11*mz1*r11*r22 - h11*mz1*r12*r21
                k += 1
                Mv[k] += h11*mz1*r11*r32 - h11*mz1*r12*r31
                k += 1
                Mv[k] += h12*intrho*r11**2 + h12*intrho*r12**2 + h12*intrho*r13**2
                k += 1
                Mv[k] += h12*intrho*r11*r21 + h12*intrho*r12*r22 + h12*intrho*r13*r23
                k += 1
                Mv[k] += h12*intrho*r11*r31 + h12*intrho*r12*r32 + h12*intrho*r13*r33
                k += 1
                Mv[k] += h12*mz2*r11*r22 - h12*mz2*r12*r21
                k += 1
                Mv[k] += h12*mz2*r11*r32 - h12*mz2*r12*r31
                k += 1
                Mv[k] += h13*intrho*r11**2 + h13*intrho*r12**2 + h13*intrho*r13**2
                k += 1
                Mv[k] += h13*intrho*r11*r21 + h13*intrho*r12*r22 + h13*intrho*r13*r23
                k += 1
                Mv[k] += h13*intrho*r11*r31 + h13*intrho*r12*r32 + h13*intrho*r13*r33
                k += 1
                Mv[k] += h13*mz3*r11*r22 - h13*mz3*r12*r21
                k += 1
                Mv[k] += h13*mz3*r11*r32 - h13*mz3*r12*r31
                k += 1
                Mv[k] += h14*intrho*r11**2 + h14*intrho*r12**2 + h14*intrho*r13**2
                k += 1
                Mv[k] += h14*intrho*r11*r21 + h14*intrho*r12*r22 + h14*intrho*r13*r23
                k += 1
                Mv[k] += h14*intrho*r11*r31 + h14*intrho*r12*r32 + h14*intrho*r13*r33
                k += 1
                Mv[k] += h14*mz4*r11*r22 - h14*mz4*r12*r21
                k += 1
                Mv[k] += h14*mz4*r11*r32 - h14*mz4*r12*r31
                k += 1
                Mv[k] += h11*intrho*r11*r21 + h11*intrho*r12*r22 + h11*intrho*r13*r23
                k += 1
                Mv[k] += h11*intrho*r21**2 + h11*intrho*r22**2 + h11*intrho*r23**2
                k += 1
                Mv[k] += h11*intrho*r21*r31 + h11*intrho*r22*r32 + h11*intrho*r23*r33
                k += 1
                Mv[k] += -h11*mz1*r11*r22 + h11*mz1*r12*r21
                k += 1
                Mv[k] += h11*mz1*r21*r32 - h11*mz1*r22*r31
                k += 1
                Mv[k] += h12*intrho*r11*r21 + h12*intrho*r12*r22 + h12*intrho*r13*r23
                k += 1
                Mv[k] += h12*intrho*r21**2 + h12*intrho*r22**2 + h12*intrho*r23**2
                k += 1
                Mv[k] += h12*intrho*r21*r31 + h12*intrho*r22*r32 + h12*intrho*r23*r33
                k += 1
                Mv[k] += -h12*mz2*r11*r22 + h12*mz2*r12*r21
                k += 1
                Mv[k] += h12*mz2*r21*r32 - h12*mz2*r22*r31
                k += 1
                Mv[k] += h13*intrho*r11*r21 + h13*intrho*r12*r22 + h13*intrho*r13*r23
                k += 1
                Mv[k] += h13*intrho*r21**2 + h13*intrho*r22**2 + h13*intrho*r23**2
                k += 1
                Mv[k] += h13*intrho*r21*r31 + h13*intrho*r22*r32 + h13*intrho*r23*r33
                k += 1
                Mv[k] += -h13*mz3*r11*r22 + h13*mz3*r12*r21
                k += 1
                Mv[k] += h13*mz3*r21*r32 - h13*mz3*r22*r31
                k += 1
                Mv[k] += h14*intrho*r11*r21 + h14*intrho*r12*r22 + h14*intrho*r13*r23
                k += 1
                Mv[k] += h14*intrho*r21**2 + h14*intrho*r22**2 + h14*intrho*r23**2
                k += 1
                Mv[k] += h14*intrho*r21*r31 + h14*intrho*r22*r32 + h14*intrho*r23*r33
                k += 1
                Mv[k] += -h14*mz4*r11*r22 + h14*mz4*r12*r21
                k += 1
                Mv[k] += h14*mz4*r21*r32 - h14*mz4*r22*r31
                k += 1
                Mv[k] += h11*intrho*r11*r31 + h11*intrho*r12*r32 + h11*intrho*r13*r33
                k += 1
                Mv[k] += h11*intrho*r21*r31 + h11*intrho*r22*r32 + h11*intrho*r23*r33
                k += 1
                Mv[k] += h11*intrho*r31**2 + h11*intrho*r32**2 + h11*intrho*r33**2
                k += 1
                Mv[k] += -h11*mz1*r11*r32 + h11*mz1*r12*r31
                k += 1
                Mv[k] += -h11*mz1*r21*r32 + h11*mz1*r22*r31
                k += 1
                Mv[k] += h12*intrho*r11*r31 + h12*intrho*r12*r32 + h12*intrho*r13*r33
                k += 1
                Mv[k] += h12*intrho*r21*r31 + h12*intrho*r22*r32 + h12*intrho*r23*r33
                k += 1
                Mv[k] += h12*intrho*r31**2 + h12*intrho*r32**2 + h12*intrho*r33**2
                k += 1
                Mv[k] += -h12*mz2*r11*r32 + h12*mz2*r12*r31
                k += 1
                Mv[k] += -h12*mz2*r21*r32 + h12*mz2*r22*r31
                k += 1
                Mv[k] += h13*intrho*r11*r31 + h13*intrho*r12*r32 + h13*intrho*r13*r33
                k += 1
                Mv[k] += h13*intrho*r21*r31 + h13*intrho*r22*r32 + h13*intrho*r23*r33
                k += 1
                Mv[k] += h13*intrho*r31**2 + h13*intrho*r32**2 + h13*intrho*r33**2
                k += 1
                Mv[k] += -h13*mz3*r11*r32 + h13*mz3*r12*r31
                k += 1
                Mv[k] += -h13*mz3*r21*r32 + h13*mz3*r22*r31
                k += 1
                Mv[k] += h14*intrho*r11*r31 + h14*intrho*r12*r32 + h14*intrho*r13*r33
                k += 1
                Mv[k] += h14*intrho*r21*r31 + h14*intrho*r22*r32 + h14*intrho*r23*r33
                k += 1
                Mv[k] += h14*intrho*r31**2 + h14*intrho*r32**2 + h14*intrho*r33**2
                k += 1
                Mv[k] += -h14*mz4*r11*r32 + h14*mz4*r12*r31
                k += 1
                Mv[k] += -h14*mz4*r21*r32 + h14*mz4*r22*r31
                k += 1
                Mv[k] += -h11*mz1*r11*r22 + h11*mz1*r12*r21
                k += 1
                Mv[k] += -h11*mz1*r11*r32 + h11*mz1*r12*r31
                k += 1
                Mv[k] += h11*mzz11*r11**2 + h11*mzz11*r12**2
                k += 1
                Mv[k] += h11*mzz11*r11*r21 + h11*mzz11*r12*r22
                k += 1
                Mv[k] += h11*mzz11*r11*r31 + h11*mzz11*r12*r32
                k += 1
                Mv[k] += -h12*mz1*r11*r22 + h12*mz1*r12*r21
                k += 1
                Mv[k] += -h12*mz1*r11*r32 + h12*mz1*r12*r31
                k += 1
                Mv[k] += h12*mzz12*r11**2 + h12*mzz12*r12**2
                k += 1
                Mv[k] += h12*mzz12*r11*r21 + h12*mzz12*r12*r22
                k += 1
                Mv[k] += h12*mzz12*r11*r31 + h12*mzz12*r12*r32
                k += 1
                Mv[k] += -h13*mz1*r11*r22 + h13*mz1*r12*r21
                k += 1
                Mv[k] += -h13*mz1*r11*r32 + h13*mz1*r12*r31
                k += 1
                Mv[k] += h13*mzz13*r11**2 + h13*mzz13*r12**2
                k += 1
                Mv[k] += h13*mzz13*r11*r21 + h13*mzz13*r12*r22
                k += 1
                Mv[k] += h13*mzz13*r11*r31 + h13*mzz13*r12*r32
                k += 1
                Mv[k] += -h14*mz1*r11*r22 + h14*mz1*r12*r21
                k += 1
                Mv[k] += -h14*mz1*r11*r32 + h14*mz1*r12*r31
                k += 1
                Mv[k] += h14*mzz14*r11**2 + h14*mzz14*r12**2
                k += 1
                Mv[k] += h14*mzz14*r11*r21 + h14*mzz14*r12*r22
                k += 1
                Mv[k] += h14*mzz14*r11*r31 + h14*mzz14*r12*r32
                k += 1
                Mv[k] += h11*mz1*r11*r22 - h11*mz1*r12*r21
                k += 1
                Mv[k] += -h11*mz1*r21*r32 + h11*mz1*r22*r31
                k += 1
                Mv[k] += h11*mzz11*r11*r21 + h11*mzz11*r12*r22
                k += 1
                Mv[k] += h11*mzz11*r21**2 + h11*mzz11*r22**2
                k += 1
                Mv[k] += h11*mzz11*r21*r31 + h11*mzz11*r22*r32
                k += 1
                Mv[k] += h12*mz1*r11*r22 - h12*mz1*r12*r21
                k += 1
                Mv[k] += -h12*mz1*r21*r32 + h12*mz1*r22*r31
                k += 1
                Mv[k] += h12*mzz12*r11*r21 + h12*mzz12*r12*r22
                k += 1
                Mv[k] += h12*mzz12*r21**2 + h12*mzz12*r22**2
                k += 1
                Mv[k] += h12*mzz12*r21*r31 + h12*mzz12*r22*r32
                k += 1
                Mv[k] += h13*mz1*r11*r22 - h13*mz1*r12*r21
                k += 1
                Mv[k] += -h13*mz1*r21*r32 + h13*mz1*r22*r31
                k += 1
                Mv[k] += h13*mzz13*r11*r21 + h13*mzz13*r12*r22
                k += 1
                Mv[k] += h13*mzz13*r21**2 + h13*mzz13*r22**2
                k += 1
                Mv[k] += h13*mzz13*r21*r31 + h13*mzz13*r22*r32
                k += 1
                Mv[k] += h14*mz1*r11*r22 - h14*mz1*r12*r21
                k += 1
                Mv[k] += -h14*mz1*r21*r32 + h14*mz1*r22*r31
                k += 1
                Mv[k] += h14*mzz14*r11*r21 + h14*mzz14*r12*r22
                k += 1
                Mv[k] += h14*mzz14*r21**2 + h14*mzz14*r22**2
                k += 1
                Mv[k] += h14*mzz14*r21*r31 + h14*mzz14*r22*r32
                k += 1
                Mv[k] += h11*mz1*r11*r32 - h11*mz1*r12*r31
                k += 1
                Mv[k] += h11*mz1*r21*r32 - h11*mz1*r22*r31
                k += 1
                Mv[k] += h11*mzz11*r11*r31 + h11*mzz11*r12*r32
                k += 1
                Mv[k] += h11*mzz11*r21*r31 + h11*mzz11*r22*r32
                k += 1
                Mv[k] += h11*mzz11*r31**2 + h11*mzz11*r32**2
                k += 1
                Mv[k] += h12*mz1*r11*r32 - h12*mz1*r12*r31
                k += 1
                Mv[k] += h12*mz1*r21*r32 - h12*mz1*r22*r31
                k += 1
                Mv[k] += h12*mzz12*r11*r31 + h12*mzz12*r12*r32
                k += 1
                Mv[k] += h12*mzz12*r21*r31 + h12*mzz12*r22*r32
                k += 1
                Mv[k] += h12*mzz12*r31**2 + h12*mzz12*r32**2
                k += 1
                Mv[k] += h13*mz1*r11*r32 - h13*mz1*r12*r31
                k += 1
                Mv[k] += h13*mz1*r21*r32 - h13*mz1*r22*r31
                k += 1
                Mv[k] += h13*mzz13*r11*r31 + h13*mzz13*r12*r32
                k += 1
                Mv[k] += h13*mzz13*r21*r31 + h13*mzz13*r22*r32
                k += 1
                Mv[k] += h13*mzz13*r31**2 + h13*mzz13*r32**2
                k += 1
                Mv[k] += h14*mz1*r11*r32 - h14*mz1*r12*r31
                k += 1
                Mv[k] += h14*mz1*r21*r32 - h14*mz1*r22*r31
                k += 1
                Mv[k] += h14*mzz14*r11*r31 + h14*mzz14*r12*r32
                k += 1
                Mv[k] += h14*mzz14*r21*r31 + h14*mzz14*r22*r32
                k += 1
                Mv[k] += h14*mzz14*r31**2 + h14*mzz14*r32**2
                k += 1
                Mv[k] += h12*intrho*r11**2 + h12*intrho*r12**2 + h12*intrho*r13**2
                k += 1
                Mv[k] += h12*intrho*r11*r21 + h12*intrho*r12*r22 + h12*intrho*r13*r23
                k += 1
                Mv[k] += h12*intrho*r11*r31 + h12*intrho*r12*r32 + h12*intrho*r13*r33
                k += 1
                Mv[k] += h12*mz1*r11*r22 - h12*mz1*r12*r21
                k += 1
                Mv[k] += h12*mz1*r11*r32 - h12*mz1*r12*r31
                k += 1
                Mv[k] += h22*intrho*r11**2 + h22*intrho*r12**2 + h22*intrho*r13**2
                k += 1
                Mv[k] += h22*intrho*r11*r21 + h22*intrho*r12*r22 + h22*intrho*r13*r23
                k += 1
                Mv[k] += h22*intrho*r11*r31 + h22*intrho*r12*r32 + h22*intrho*r13*r33
                k += 1
                Mv[k] += h22*mz2*r11*r22 - h22*mz2*r12*r21
                k += 1
                Mv[k] += h22*mz2*r11*r32 - h22*mz2*r12*r31
                k += 1
                Mv[k] += h23*intrho*r11**2 + h23*intrho*r12**2 + h23*intrho*r13**2
                k += 1
                Mv[k] += h23*intrho*r11*r21 + h23*intrho*r12*r22 + h23*intrho*r13*r23
                k += 1
                Mv[k] += h23*intrho*r11*r31 + h23*intrho*r12*r32 + h23*intrho*r13*r33
                k += 1
                Mv[k] += h23*mz3*r11*r22 - h23*mz3*r12*r21
                k += 1
                Mv[k] += h23*mz3*r11*r32 - h23*mz3*r12*r31
                k += 1
                Mv[k] += h24*intrho*r11**2 + h24*intrho*r12**2 + h24*intrho*r13**2
                k += 1
                Mv[k] += h24*intrho*r11*r21 + h24*intrho*r12*r22 + h24*intrho*r13*r23
                k += 1
                Mv[k] += h24*intrho*r11*r31 + h24*intrho*r12*r32 + h24*intrho*r13*r33
                k += 1
                Mv[k] += h24*mz4*r11*r22 - h24*mz4*r12*r21
                k += 1
                Mv[k] += h24*mz4*r11*r32 - h24*mz4*r12*r31
                k += 1
                Mv[k] += h12*intrho*r11*r21 + h12*intrho*r12*r22 + h12*intrho*r13*r23
                k += 1
                Mv[k] += h12*intrho*r21**2 + h12*intrho*r22**2 + h12*intrho*r23**2
                k += 1
                Mv[k] += h12*intrho*r21*r31 + h12*intrho*r22*r32 + h12*intrho*r23*r33
                k += 1
                Mv[k] += -h12*mz1*r11*r22 + h12*mz1*r12*r21
                k += 1
                Mv[k] += h12*mz1*r21*r32 - h12*mz1*r22*r31
                k += 1
                Mv[k] += h22*intrho*r11*r21 + h22*intrho*r12*r22 + h22*intrho*r13*r23
                k += 1
                Mv[k] += h22*intrho*r21**2 + h22*intrho*r22**2 + h22*intrho*r23**2
                k += 1
                Mv[k] += h22*intrho*r21*r31 + h22*intrho*r22*r32 + h22*intrho*r23*r33
                k += 1
                Mv[k] += -h22*mz2*r11*r22 + h22*mz2*r12*r21
                k += 1
                Mv[k] += h22*mz2*r21*r32 - h22*mz2*r22*r31
                k += 1
                Mv[k] += h23*intrho*r11*r21 + h23*intrho*r12*r22 + h23*intrho*r13*r23
                k += 1
                Mv[k] += h23*intrho*r21**2 + h23*intrho*r22**2 + h23*intrho*r23**2
                k += 1
                Mv[k] += h23*intrho*r21*r31 + h23*intrho*r22*r32 + h23*intrho*r23*r33
                k += 1
                Mv[k] += -h23*mz3*r11*r22 + h23*mz3*r12*r21
                k += 1
                Mv[k] += h23*mz3*r21*r32 - h23*mz3*r22*r31
                k += 1
                Mv[k] += h24*intrho*r11*r21 + h24*intrho*r12*r22 + h24*intrho*r13*r23
                k += 1
                Mv[k] += h24*intrho*r21**2 + h24*intrho*r22**2 + h24*intrho*r23**2
                k += 1
                Mv[k] += h24*intrho*r21*r31 + h24*intrho*r22*r32 + h24*intrho*r23*r33
                k += 1
                Mv[k] += -h24*mz4*r11*r22 + h24*mz4*r12*r21
                k += 1
                Mv[k] += h24*mz4*r21*r32 - h24*mz4*r22*r31
                k += 1
                Mv[k] += h12*intrho*r11*r31 + h12*intrho*r12*r32 + h12*intrho*r13*r33
                k += 1
                Mv[k] += h12*intrho*r21*r31 + h12*intrho*r22*r32 + h12*intrho*r23*r33
                k += 1
                Mv[k] += h12*intrho*r31**2 + h12*intrho*r32**2 + h12*intrho*r33**2
                k += 1
                Mv[k] += -h12*mz1*r11*r32 + h12*mz1*r12*r31
                k += 1
                Mv[k] += -h12*mz1*r21*r32 + h12*mz1*r22*r31
                k += 1
                Mv[k] += h22*intrho*r11*r31 + h22*intrho*r12*r32 + h22*intrho*r13*r33
                k += 1
                Mv[k] += h22*intrho*r21*r31 + h22*intrho*r22*r32 + h22*intrho*r23*r33
                k += 1
                Mv[k] += h22*intrho*r31**2 + h22*intrho*r32**2 + h22*intrho*r33**2
                k += 1
                Mv[k] += -h22*mz2*r11*r32 + h22*mz2*r12*r31
                k += 1
                Mv[k] += -h22*mz2*r21*r32 + h22*mz2*r22*r31
                k += 1
                Mv[k] += h23*intrho*r11*r31 + h23*intrho*r12*r32 + h23*intrho*r13*r33
                k += 1
                Mv[k] += h23*intrho*r21*r31 + h23*intrho*r22*r32 + h23*intrho*r23*r33
                k += 1
                Mv[k] += h23*intrho*r31**2 + h23*intrho*r32**2 + h23*intrho*r33**2
                k += 1
                Mv[k] += -h23*mz3*r11*r32 + h23*mz3*r12*r31
                k += 1
                Mv[k] += -h23*mz3*r21*r32 + h23*mz3*r22*r31
                k += 1
                Mv[k] += h24*intrho*r11*r31 + h24*intrho*r12*r32 + h24*intrho*r13*r33
                k += 1
                Mv[k] += h24*intrho*r21*r31 + h24*intrho*r22*r32 + h24*intrho*r23*r33
                k += 1
                Mv[k] += h24*intrho*r31**2 + h24*intrho*r32**2 + h24*intrho*r33**2
                k += 1
                Mv[k] += -h24*mz4*r11*r32 + h24*mz4*r12*r31
                k += 1
                Mv[k] += -h24*mz4*r21*r32 + h24*mz4*r22*r31
                k += 1
                Mv[k] += -h12*mz2*r11*r22 + h12*mz2*r12*r21
                k += 1
                Mv[k] += -h12*mz2*r11*r32 + h12*mz2*r12*r31
                k += 1
                Mv[k] += h12*mzz12*r11**2 + h12*mzz12*r12**2
                k += 1
                Mv[k] += h12*mzz12*r11*r21 + h12*mzz12*r12*r22
                k += 1
                Mv[k] += h12*mzz12*r11*r31 + h12*mzz12*r12*r32
                k += 1
                Mv[k] += -h22*mz2*r11*r22 + h22*mz2*r12*r21
                k += 1
                Mv[k] += -h22*mz2*r11*r32 + h22*mz2*r12*r31
                k += 1
                Mv[k] += h22*mzz22*r11**2 + h22*mzz22*r12**2
                k += 1
                Mv[k] += h22*mzz22*r11*r21 + h22*mzz22*r12*r22
                k += 1
                Mv[k] += h22*mzz22*r11*r31 + h22*mzz22*r12*r32
                k += 1
                Mv[k] += -h23*mz2*r11*r22 + h23*mz2*r12*r21
                k += 1
                Mv[k] += -h23*mz2*r11*r32 + h23*mz2*r12*r31
                k += 1
                Mv[k] += h23*mzz23*r11**2 + h23*mzz23*r12**2
                k += 1
                Mv[k] += h23*mzz23*r11*r21 + h23*mzz23*r12*r22
                k += 1
                Mv[k] += h23*mzz23*r11*r31 + h23*mzz23*r12*r32
                k += 1
                Mv[k] += -h24*mz2*r11*r22 + h24*mz2*r12*r21
                k += 1
                Mv[k] += -h24*mz2*r11*r32 + h24*mz2*r12*r31
                k += 1
                Mv[k] += h24*mzz24*r11**2 + h24*mzz24*r12**2
                k += 1
                Mv[k] += h24*mzz24*r11*r21 + h24*mzz24*r12*r22
                k += 1
                Mv[k] += h24*mzz24*r11*r31 + h24*mzz24*r12*r32
                k += 1
                Mv[k] += h12*mz2*r11*r22 - h12*mz2*r12*r21
                k += 1
                Mv[k] += -h12*mz2*r21*r32 + h12*mz2*r22*r31
                k += 1
                Mv[k] += h12*mzz12*r11*r21 + h12*mzz12*r12*r22
                k += 1
                Mv[k] += h12*mzz12*r21**2 + h12*mzz12*r22**2
                k += 1
                Mv[k] += h12*mzz12*r21*r31 + h12*mzz12*r22*r32
                k += 1
                Mv[k] += h22*mz2*r11*r22 - h22*mz2*r12*r21
                k += 1
                Mv[k] += -h22*mz2*r21*r32 + h22*mz2*r22*r31
                k += 1
                Mv[k] += h22*mzz22*r11*r21 + h22*mzz22*r12*r22
                k += 1
                Mv[k] += h22*mzz22*r21**2 + h22*mzz22*r22**2
                k += 1
                Mv[k] += h22*mzz22*r21*r31 + h22*mzz22*r22*r32
                k += 1
                Mv[k] += h23*mz2*r11*r22 - h23*mz2*r12*r21
                k += 1
                Mv[k] += -h23*mz2*r21*r32 + h23*mz2*r22*r31
                k += 1
                Mv[k] += h23*mzz23*r11*r21 + h23*mzz23*r12*r22
                k += 1
                Mv[k] += h23*mzz23*r21**2 + h23*mzz23*r22**2
                k += 1
                Mv[k] += h23*mzz23*r21*r31 + h23*mzz23*r22*r32
                k += 1
                Mv[k] += h24*mz2*r11*r22 - h24*mz2*r12*r21
                k += 1
                Mv[k] += -h24*mz2*r21*r32 + h24*mz2*r22*r31
                k += 1
                Mv[k] += h24*mzz24*r11*r21 + h24*mzz24*r12*r22
                k += 1
                Mv[k] += h24*mzz24*r21**2 + h24*mzz24*r22**2
                k += 1
                Mv[k] += h24*mzz24*r21*r31 + h24*mzz24*r22*r32
                k += 1
                Mv[k] += h12*mz2*r11*r32 - h12*mz2*r12*r31
                k += 1
                Mv[k] += h12*mz2*r21*r32 - h12*mz2*r22*r31
                k += 1
                Mv[k] += h12*mzz12*r11*r31 + h12*mzz12*r12*r32
                k += 1
                Mv[k] += h12*mzz12*r21*r31 + h12*mzz12*r22*r32
                k += 1
                Mv[k] += h12*mzz12*r31**2 + h12*mzz12*r32**2
                k += 1
                Mv[k] += h22*mz2*r11*r32 - h22*mz2*r12*r31
                k += 1
                Mv[k] += h22*mz2*r21*r32 - h22*mz2*r22*r31
                k += 1
                Mv[k] += h22*mzz22*r11*r31 + h22*mzz22*r12*r32
                k += 1
                Mv[k] += h22*mzz22*r21*r31 + h22*mzz22*r22*r32
                k += 1
                Mv[k] += h22*mzz22*r31**2 + h22*mzz22*r32**2
                k += 1
                Mv[k] += h23*mz2*r11*r32 - h23*mz2*r12*r31
                k += 1
                Mv[k] += h23*mz2*r21*r32 - h23*mz2*r22*r31
                k += 1
                Mv[k] += h23*mzz23*r11*r31 + h23*mzz23*r12*r32
                k += 1
                Mv[k] += h23*mzz23*r21*r31 + h23*mzz23*r22*r32
                k += 1
                Mv[k] += h23*mzz23*r31**2 + h23*mzz23*r32**2
                k += 1
                Mv[k] += h24*mz2*r11*r32 - h24*mz2*r12*r31
                k += 1
                Mv[k] += h24*mz2*r21*r32 - h24*mz2*r22*r31
                k += 1
                Mv[k] += h24*mzz24*r11*r31 + h24*mzz24*r12*r32
                k += 1
                Mv[k] += h24*mzz24*r21*r31 + h24*mzz24*r22*r32
                k += 1
                Mv[k] += h24*mzz24*r31**2 + h24*mzz24*r32**2
                k += 1
                Mv[k] += h13*intrho*r11**2 + h13*intrho*r12**2 + h13*intrho*r13**2
                k += 1
                Mv[k] += h13*intrho*r11*r21 + h13*intrho*r12*r22 + h13*intrho*r13*r23
                k += 1
                Mv[k] += h13*intrho*r11*r31 + h13*intrho*r12*r32 + h13*intrho*r13*r33
                k += 1
                Mv[k] += h13*mz1*r11*r22 - h13*mz1*r12*r21
                k += 1
                Mv[k] += h13*mz1*r11*r32 - h13*mz1*r12*r31
                k += 1
                Mv[k] += h23*intrho*r11**2 + h23*intrho*r12**2 + h23*intrho*r13**2
                k += 1
                Mv[k] += h23*intrho*r11*r21 + h23*intrho*r12*r22 + h23*intrho*r13*r23
                k += 1
                Mv[k] += h23*intrho*r11*r31 + h23*intrho*r12*r32 + h23*intrho*r13*r33
                k += 1
                Mv[k] += h23*mz2*r11*r22 - h23*mz2*r12*r21
                k += 1
                Mv[k] += h23*mz2*r11*r32 - h23*mz2*r12*r31
                k += 1
                Mv[k] += h33*intrho*r11**2 + h33*intrho*r12**2 + h33*intrho*r13**2
                k += 1
                Mv[k] += h33*intrho*r11*r21 + h33*intrho*r12*r22 + h33*intrho*r13*r23
                k += 1
                Mv[k] += h33*intrho*r11*r31 + h33*intrho*r12*r32 + h33*intrho*r13*r33
                k += 1
                Mv[k] += h33*mz3*r11*r22 - h33*mz3*r12*r21
                k += 1
                Mv[k] += h33*mz3*r11*r32 - h33*mz3*r12*r31
                k += 1
                Mv[k] += h34*intrho*r11**2 + h34*intrho*r12**2 + h34*intrho*r13**2
                k += 1
                Mv[k] += h34*intrho*r11*r21 + h34*intrho*r12*r22 + h34*intrho*r13*r23
                k += 1
                Mv[k] += h34*intrho*r11*r31 + h34*intrho*r12*r32 + h34*intrho*r13*r33
                k += 1
                Mv[k] += h34*mz4*r11*r22 - h34*mz4*r12*r21
                k += 1
                Mv[k] += h34*mz4*r11*r32 - h34*mz4*r12*r31
                k += 1
                Mv[k] += h13*intrho*r11*r21 + h13*intrho*r12*r22 + h13*intrho*r13*r23
                k += 1
                Mv[k] += h13*intrho*r21**2 + h13*intrho*r22**2 + h13*intrho*r23**2
                k += 1
                Mv[k] += h13*intrho*r21*r31 + h13*intrho*r22*r32 + h13*intrho*r23*r33
                k += 1
                Mv[k] += -h13*mz1*r11*r22 + h13*mz1*r12*r21
                k += 1
                Mv[k] += h13*mz1*r21*r32 - h13*mz1*r22*r31
                k += 1
                Mv[k] += h23*intrho*r11*r21 + h23*intrho*r12*r22 + h23*intrho*r13*r23
                k += 1
                Mv[k] += h23*intrho*r21**2 + h23*intrho*r22**2 + h23*intrho*r23**2
                k += 1
                Mv[k] += h23*intrho*r21*r31 + h23*intrho*r22*r32 + h23*intrho*r23*r33
                k += 1
                Mv[k] += -h23*mz2*r11*r22 + h23*mz2*r12*r21
                k += 1
                Mv[k] += h23*mz2*r21*r32 - h23*mz2*r22*r31
                k += 1
                Mv[k] += h33*intrho*r11*r21 + h33*intrho*r12*r22 + h33*intrho*r13*r23
                k += 1
                Mv[k] += h33*intrho*r21**2 + h33*intrho*r22**2 + h33*intrho*r23**2
                k += 1
                Mv[k] += h33*intrho*r21*r31 + h33*intrho*r22*r32 + h33*intrho*r23*r33
                k += 1
                Mv[k] += -h33*mz3*r11*r22 + h33*mz3*r12*r21
                k += 1
                Mv[k] += h33*mz3*r21*r32 - h33*mz3*r22*r31
                k += 1
                Mv[k] += h34*intrho*r11*r21 + h34*intrho*r12*r22 + h34*intrho*r13*r23
                k += 1
                Mv[k] += h34*intrho*r21**2 + h34*intrho*r22**2 + h34*intrho*r23**2
                k += 1
                Mv[k] += h34*intrho*r21*r31 + h34*intrho*r22*r32 + h34*intrho*r23*r33
                k += 1
                Mv[k] += -h34*mz4*r11*r22 + h34*mz4*r12*r21
                k += 1
                Mv[k] += h34*mz4*r21*r32 - h34*mz4*r22*r31
                k += 1
                Mv[k] += h13*intrho*r11*r31 + h13*intrho*r12*r32 + h13*intrho*r13*r33
                k += 1
                Mv[k] += h13*intrho*r21*r31 + h13*intrho*r22*r32 + h13*intrho*r23*r33
                k += 1
                Mv[k] += h13*intrho*r31**2 + h13*intrho*r32**2 + h13*intrho*r33**2
                k += 1
                Mv[k] += -h13*mz1*r11*r32 + h13*mz1*r12*r31
                k += 1
                Mv[k] += -h13*mz1*r21*r32 + h13*mz1*r22*r31
                k += 1
                Mv[k] += h23*intrho*r11*r31 + h23*intrho*r12*r32 + h23*intrho*r13*r33
                k += 1
                Mv[k] += h23*intrho*r21*r31 + h23*intrho*r22*r32 + h23*intrho*r23*r33
                k += 1
                Mv[k] += h23*intrho*r31**2 + h23*intrho*r32**2 + h23*intrho*r33**2
                k += 1
                Mv[k] += -h23*mz2*r11*r32 + h23*mz2*r12*r31
                k += 1
                Mv[k] += -h23*mz2*r21*r32 + h23*mz2*r22*r31
                k += 1
                Mv[k] += h33*intrho*r11*r31 + h33*intrho*r12*r32 + h33*intrho*r13*r33
                k += 1
                Mv[k] += h33*intrho*r21*r31 + h33*intrho*r22*r32 + h33*intrho*r23*r33
                k += 1
                Mv[k] += h33*intrho*r31**2 + h33*intrho*r32**2 + h33*intrho*r33**2
                k += 1
                Mv[k] += -h33*mz3*r11*r32 + h33*mz3*r12*r31
                k += 1
                Mv[k] += -h33*mz3*r21*r32 + h33*mz3*r22*r31
                k += 1
                Mv[k] += h34*intrho*r11*r31 + h34*intrho*r12*r32 + h34*intrho*r13*r33
                k += 1
                Mv[k] += h34*intrho*r21*r31 + h34*intrho*r22*r32 + h34*intrho*r23*r33
                k += 1
                Mv[k] += h34*intrho*r31**2 + h34*intrho*r32**2 + h34*intrho*r33**2
                k += 1
                Mv[k] += -h34*mz4*r11*r32 + h34*mz4*r12*r31
                k += 1
                Mv[k] += -h34*mz4*r21*r32 + h34*mz4*r22*r31
                k += 1
                Mv[k] += -h13*mz3*r11*r22 + h13*mz3*r12*r21
                k += 1
                Mv[k] += -h13*mz3*r11*r32 + h13*mz3*r12*r31
                k += 1
                Mv[k] += h13*mzz13*r11**2 + h13*mzz13*r12**2
                k += 1
                Mv[k] += h13*mzz13*r11*r21 + h13*mzz13*r12*r22
                k += 1
                Mv[k] += h13*mzz13*r11*r31 + h13*mzz13*r12*r32
                k += 1
                Mv[k] += -h23*mz3*r11*r22 + h23*mz3*r12*r21
                k += 1
                Mv[k] += -h23*mz3*r11*r32 + h23*mz3*r12*r31
                k += 1
                Mv[k] += h23*mzz23*r11**2 + h23*mzz23*r12**2
                k += 1
                Mv[k] += h23*mzz23*r11*r21 + h23*mzz23*r12*r22
                k += 1
                Mv[k] += h23*mzz23*r11*r31 + h23*mzz23*r12*r32
                k += 1
                Mv[k] += -h33*mz3*r11*r22 + h33*mz3*r12*r21
                k += 1
                Mv[k] += -h33*mz3*r11*r32 + h33*mz3*r12*r31
                k += 1
                Mv[k] += h33*mzz33*r11**2 + h33*mzz33*r12**2
                k += 1
                Mv[k] += h33*mzz33*r11*r21 + h33*mzz33*r12*r22
                k += 1
                Mv[k] += h33*mzz33*r11*r31 + h33*mzz33*r12*r32
                k += 1
                Mv[k] += -h34*mz3*r11*r22 + h34*mz3*r12*r21
                k += 1
                Mv[k] += -h34*mz3*r11*r32 + h34*mz3*r12*r31
                k += 1
                Mv[k] += h34*mzz34*r11**2 + h34*mzz34*r12**2
                k += 1
                Mv[k] += h34*mzz34*r11*r21 + h34*mzz34*r12*r22
                k += 1
                Mv[k] += h34*mzz34*r11*r31 + h34*mzz34*r12*r32
                k += 1
                Mv[k] += h13*mz3*r11*r22 - h13*mz3*r12*r21
                k += 1
                Mv[k] += -h13*mz3*r21*r32 + h13*mz3*r22*r31
                k += 1
                Mv[k] += h13*mzz13*r11*r21 + h13*mzz13*r12*r22
                k += 1
                Mv[k] += h13*mzz13*r21**2 + h13*mzz13*r22**2
                k += 1
                Mv[k] += h13*mzz13*r21*r31 + h13*mzz13*r22*r32
                k += 1
                Mv[k] += h23*mz3*r11*r22 - h23*mz3*r12*r21
                k += 1
                Mv[k] += -h23*mz3*r21*r32 + h23*mz3*r22*r31
                k += 1
                Mv[k] += h23*mzz23*r11*r21 + h23*mzz23*r12*r22
                k += 1
                Mv[k] += h23*mzz23*r21**2 + h23*mzz23*r22**2
                k += 1
                Mv[k] += h23*mzz23*r21*r31 + h23*mzz23*r22*r32
                k += 1
                Mv[k] += h33*mz3*r11*r22 - h33*mz3*r12*r21
                k += 1
                Mv[k] += -h33*mz3*r21*r32 + h33*mz3*r22*r31
                k += 1
                Mv[k] += h33*mzz33*r11*r21 + h33*mzz33*r12*r22
                k += 1
                Mv[k] += h33*mzz33*r21**2 + h33*mzz33*r22**2
                k += 1
                Mv[k] += h33*mzz33*r21*r31 + h33*mzz33*r22*r32
                k += 1
                Mv[k] += h34*mz3*r11*r22 - h34*mz3*r12*r21
                k += 1
                Mv[k] += -h34*mz3*r21*r32 + h34*mz3*r22*r31
                k += 1
                Mv[k] += h34*mzz34*r11*r21 + h34*mzz34*r12*r22
                k += 1
                Mv[k] += h34*mzz34*r21**2 + h34*mzz34*r22**2
                k += 1
                Mv[k] += h34*mzz34*r21*r31 + h34*mzz34*r22*r32
                k += 1
                Mv[k] += h13*mz3*r11*r32 - h13*mz3*r12*r31
                k += 1
                Mv[k] += h13*mz3*r21*r32 - h13*mz3*r22*r31
                k += 1
                Mv[k] += h13*mzz13*r11*r31 + h13*mzz13*r12*r32
                k += 1
                Mv[k] += h13*mzz13*r21*r31 + h13*mzz13*r22*r32
                k += 1
                Mv[k] += h13*mzz13*r31**2 + h13*mzz13*r32**2
                k += 1
                Mv[k] += h23*mz3*r11*r32 - h23*mz3*r12*r31
                k += 1
                Mv[k] += h23*mz3*r21*r32 - h23*mz3*r22*r31
                k += 1
                Mv[k] += h23*mzz23*r11*r31 + h23*mzz23*r12*r32
                k += 1
                Mv[k] += h23*mzz23*r21*r31 + h23*mzz23*r22*r32
                k += 1
                Mv[k] += h23*mzz23*r31**2 + h23*mzz23*r32**2
                k += 1
                Mv[k] += h33*mz3*r11*r32 - h33*mz3*r12*r31
                k += 1
                Mv[k] += h33*mz3*r21*r32 - h33*mz3*r22*r31
                k += 1
                Mv[k] += h33*mzz33*r11*r31 + h33*mzz33*r12*r32
                k += 1
                Mv[k] += h33*mzz33*r21*r31 + h33*mzz33*r22*r32
                k += 1
                Mv[k] += h33*mzz33*r31**2 + h33*mzz33*r32**2
                k += 1
                Mv[k] += h34*mz3*r11*r32 - h34*mz3*r12*r31
                k += 1
                Mv[k] += h34*mz3*r21*r32 - h34*mz3*r22*r31
                k += 1
                Mv[k] += h34*mzz34*r11*r31 + h34*mzz34*r12*r32
                k += 1
                Mv[k] += h34*mzz34*r21*r31 + h34*mzz34*r22*r32
                k += 1
                Mv[k] += h34*mzz34*r31**2 + h34*mzz34*r32**2
                k += 1
                Mv[k] += h14*intrho*r11**2 + h14*intrho*r12**2 + h14*intrho*r13**2
                k += 1
                Mv[k] += h14*intrho*r11*r21 + h14*intrho*r12*r22 + h14*intrho*r13*r23
                k += 1
                Mv[k] += h14*intrho*r11*r31 + h14*intrho*r12*r32 + h14*intrho*r13*r33
                k += 1
                Mv[k] += h14*mz1*r11*r22 - h14*mz1*r12*r21
                k += 1
                Mv[k] += h14*mz1*r11*r32 - h14*mz1*r12*r31
                k += 1
                Mv[k] += h24*intrho*r11**2 + h24*intrho*r12**2 + h24*intrho*r13**2
                k += 1
                Mv[k] += h24*intrho*r11*r21 + h24*intrho*r12*r22 + h24*intrho*r13*r23
                k += 1
                Mv[k] += h24*intrho*r11*r31 + h24*intrho*r12*r32 + h24*intrho*r13*r33
                k += 1
                Mv[k] += h24*mz2*r11*r22 - h24*mz2*r12*r21
                k += 1
                Mv[k] += h24*mz2*r11*r32 - h24*mz2*r12*r31
                k += 1
                Mv[k] += h34*intrho*r11**2 + h34*intrho*r12**2 + h34*intrho*r13**2
                k += 1
                Mv[k] += h34*intrho*r11*r21 + h34*intrho*r12*r22 + h34*intrho*r13*r23
                k += 1
                Mv[k] += h34*intrho*r11*r31 + h34*intrho*r12*r32 + h34*intrho*r13*r33
                k += 1
                Mv[k] += h34*mz3*r11*r22 - h34*mz3*r12*r21
                k += 1
                Mv[k] += h34*mz3*r11*r32 - h34*mz3*r12*r31
                k += 1
                Mv[k] += h44*intrho*r11**2 + h44*intrho*r12**2 + h44*intrho*r13**2
                k += 1
                Mv[k] += h44*intrho*r11*r21 + h44*intrho*r12*r22 + h44*intrho*r13*r23
                k += 1
                Mv[k] += h44*intrho*r11*r31 + h44*intrho*r12*r32 + h44*intrho*r13*r33
                k += 1
                Mv[k] += h44*mz4*r11*r22 - h44*mz4*r12*r21
                k += 1
                Mv[k] += h44*mz4*r11*r32 - h44*mz4*r12*r31
                k += 1
                Mv[k] += h14*intrho*r11*r21 + h14*intrho*r12*r22 + h14*intrho*r13*r23
                k += 1
                Mv[k] += h14*intrho*r21**2 + h14*intrho*r22**2 + h14*intrho*r23**2
                k += 1
                Mv[k] += h14*intrho*r21*r31 + h14*intrho*r22*r32 + h14*intrho*r23*r33
                k += 1
                Mv[k] += -h14*mz1*r11*r22 + h14*mz1*r12*r21
                k += 1
                Mv[k] += h14*mz1*r21*r32 - h14*mz1*r22*r31
                k += 1
                Mv[k] += h24*intrho*r11*r21 + h24*intrho*r12*r22 + h24*intrho*r13*r23
                k += 1
                Mv[k] += h24*intrho*r21**2 + h24*intrho*r22**2 + h24*intrho*r23**2
                k += 1
                Mv[k] += h24*intrho*r21*r31 + h24*intrho*r22*r32 + h24*intrho*r23*r33
                k += 1
                Mv[k] += -h24*mz2*r11*r22 + h24*mz2*r12*r21
                k += 1
                Mv[k] += h24*mz2*r21*r32 - h24*mz2*r22*r31
                k += 1
                Mv[k] += h34*intrho*r11*r21 + h34*intrho*r12*r22 + h34*intrho*r13*r23
                k += 1
                Mv[k] += h34*intrho*r21**2 + h34*intrho*r22**2 + h34*intrho*r23**2
                k += 1
                Mv[k] += h34*intrho*r21*r31 + h34*intrho*r22*r32 + h34*intrho*r23*r33
                k += 1
                Mv[k] += -h34*mz3*r11*r22 + h34*mz3*r12*r21
                k += 1
                Mv[k] += h34*mz3*r21*r32 - h34*mz3*r22*r31
                k += 1
                Mv[k] += h44*intrho*r11*r21 + h44*intrho*r12*r22 + h44*intrho*r13*r23
                k += 1
                Mv[k] += h44*intrho*r21**2 + h44*intrho*r22**2 + h44*intrho*r23**2
                k += 1
                Mv[k] += h44*intrho*r21*r31 + h44*intrho*r22*r32 + h44*intrho*r23*r33
                k += 1
                Mv[k] += -h44*mz4*r11*r22 + h44*mz4*r12*r21
                k += 1
                Mv[k] += h44*mz4*r21*r32 - h44*mz4*r22*r31
                k += 1
                Mv[k] += h14*intrho*r11*r31 + h14*intrho*r12*r32 + h14*intrho*r13*r33
                k += 1
                Mv[k] += h14*intrho*r21*r31 + h14*intrho*r22*r32 + h14*intrho*r23*r33
                k += 1
                Mv[k] += h14*intrho*r31**2 + h14*intrho*r32**2 + h14*intrho*r33**2
                k += 1
                Mv[k] += -h14*mz1*r11*r32 + h14*mz1*r12*r31
                k += 1
                Mv[k] += -h14*mz1*r21*r32 + h14*mz1*r22*r31
                k += 1
                Mv[k] += h24*intrho*r11*r31 + h24*intrho*r12*r32 + h24*intrho*r13*r33
                k += 1
                Mv[k] += h24*intrho*r21*r31 + h24*intrho*r22*r32 + h24*intrho*r23*r33
                k += 1
                Mv[k] += h24*intrho*r31**2 + h24*intrho*r32**2 + h24*intrho*r33**2
                k += 1
                Mv[k] += -h24*mz2*r11*r32 + h24*mz2*r12*r31
                k += 1
                Mv[k] += -h24*mz2*r21*r32 + h24*mz2*r22*r31
                k += 1
                Mv[k] += h34*intrho*r11*r31 + h34*intrho*r12*r32 + h34*intrho*r13*r33
                k += 1
                Mv[k] += h34*intrho*r21*r31 + h34*intrho*r22*r32 + h34*intrho*r23*r33
                k += 1
                Mv[k] += h34*intrho*r31**2 + h34*intrho*r32**2 + h34*intrho*r33**2
                k += 1
                Mv[k] += -h34*mz3*r11*r32 + h34*mz3*r12*r31
                k += 1
                Mv[k] += -h34*mz3*r21*r32 + h34*mz3*r22*r31
                k += 1
                Mv[k] += h44*intrho*r11*r31 + h44*intrho*r12*r32 + h44*intrho*r13*r33
                k += 1
                Mv[k] += h44*intrho*r21*r31 + h44*intrho*r22*r32 + h44*intrho*r23*r33
                k += 1
                Mv[k] += h44*intrho*r31**2 + h44*intrho*r32**2 + h44*intrho*r33**2
                k += 1
                Mv[k] += -h44*mz4*r11*r32 + h44*mz4*r12*r31
                k += 1
                Mv[k] += -h44*mz4*r21*r32 + h44*mz4*r22*r31
                k += 1
                Mv[k] += -h14*mz4*r11*r22 + h14*mz4*r12*r21
                k += 1
                Mv[k] += -h14*mz4*r11*r32 + h14*mz4*r12*r31
                k += 1
                Mv[k] += h14*mzz14*r11**2 + h14*mzz14*r12**2
                k += 1
                Mv[k] += h14*mzz14*r11*r21 + h14*mzz14*r12*r22
                k += 1
                Mv[k] += h14*mzz14*r11*r31 + h14*mzz14*r12*r32
                k += 1
                Mv[k] += -h24*mz4*r11*r22 + h24*mz4*r12*r21
                k += 1
                Mv[k] += -h24*mz4*r11*r32 + h24*mz4*r12*r31
                k += 1
                Mv[k] += h24*mzz24*r11**2 + h24*mzz24*r12**2
                k += 1
                Mv[k] += h24*mzz24*r11*r21 + h24*mzz24*r12*r22
                k += 1
                Mv[k] += h24*mzz24*r11*r31 + h24*mzz24*r12*r32
                k += 1
                Mv[k] += -h34*mz4*r11*r22 + h34*mz4*r12*r21
                k += 1
                Mv[k] += -h34*mz4*r11*r32 + h34*mz4*r12*r31
                k += 1
                Mv[k] += h34*mzz34*r11**2 + h34*mzz34*r12**2
                k += 1
                Mv[k] += h34*mzz34*r11*r21 + h34*mzz34*r12*r22
                k += 1
                Mv[k] += h34*mzz34*r11*r31 + h34*mzz34*r12*r32
                k += 1
                Mv[k] += -h44*mz4*r11*r22 + h44*mz4*r12*r21
                k += 1
                Mv[k] += -h44*mz4*r11*r32 + h44*mz4*r12*r31
                k += 1
                Mv[k] += h44*mzz44*r11**2 + h44*mzz44*r12**2
                k += 1
                Mv[k] += h44*mzz44*r11*r21 + h44*mzz44*r12*r22
                k += 1
                Mv[k] += h44*mzz44*r11*r31 + h44*mzz44*r12*r32
                k += 1
                Mv[k] += h14*mz4*r11*r22 - h14*mz4*r12*r21
                k += 1
                Mv[k] += -h14*mz4*r21*r32 + h14*mz4*r22*r31
                k += 1
                Mv[k] += h14*mzz14*r11*r21 + h14*mzz14*r12*r22
                k += 1
                Mv[k] += h14*mzz14*r21**2 + h14*mzz14*r22**2
                k += 1
                Mv[k] += h14*mzz14*r21*r31 + h14*mzz14*r22*r32
                k += 1
                Mv[k] += h24*mz4*r11*r22 - h24*mz4*r12*r21
                k += 1
                Mv[k] += -h24*mz4*r21*r32 + h24*mz4*r22*r31
                k += 1
                Mv[k] += h24*mzz24*r11*r21 + h24*mzz24*r12*r22
                k += 1
                Mv[k] += h24*mzz24*r21**2 + h24*mzz24*r22**2
                k += 1
                Mv[k] += h24*mzz24*r21*r31 + h24*mzz24*r22*r32
                k += 1
                Mv[k] += h34*mz4*r11*r22 - h34*mz4*r12*r21
                k += 1
                Mv[k] += -h34*mz4*r21*r32 + h34*mz4*r22*r31
                k += 1
                Mv[k] += h34*mzz34*r11*r21 + h34*mzz34*r12*r22
                k += 1
                Mv[k] += h34*mzz34*r21**2 + h34*mzz34*r22**2
                k += 1
                Mv[k] += h34*mzz34*r21*r31 + h34*mzz34*r22*r32
                k += 1
                Mv[k] += h44*mz4*r11*r22 - h44*mz4*r12*r21
                k += 1
                Mv[k] += -h44*mz4*r21*r32 + h44*mz4*r22*r31
                k += 1
                Mv[k] += h44*mzz44*r11*r21 + h44*mzz44*r12*r22
                k += 1
                Mv[k] += h44*mzz44*r21**2 + h44*mzz44*r22**2
                k += 1
                Mv[k] += h44*mzz44*r21*r31 + h44*mzz44*r22*r32
                k += 1
                Mv[k] += h14*mz4*r11*r32 - h14*mz4*r12*r31
                k += 1
                Mv[k] += h14*mz4*r21*r32 - h14*mz4*r22*r31
                k += 1
                Mv[k] += h14*mzz14*r11*r31 + h14*mzz14*r12*r32
                k += 1
                Mv[k] += h14*mzz14*r21*r31 + h14*mzz14*r22*r32
                k += 1
                Mv[k] += h14*mzz14*r31**2 + h14*mzz14*r32**2
                k += 1
                Mv[k] += h24*mz4*r11*r32 - h24*mz4*r12*r31
                k += 1
                Mv[k] += h24*mz4*r21*r32 - h24*mz4*r22*r31
                k += 1
                Mv[k] += h24*mzz24*r11*r31 + h24*mzz24*r12*r32
                k += 1
                Mv[k] += h24*mzz24*r21*r31 + h24*mzz24*r22*r32
                k += 1
                Mv[k] += h24*mzz24*r31**2 + h24*mzz24*r32**2
                k += 1
                Mv[k] += h34*mz4*r11*r32 - h34*mz4*r12*r31
                k += 1
                Mv[k] += h34*mz4*r21*r32 - h34*mz4*r22*r31
                k += 1
                Mv[k] += h34*mzz34*r11*r31 + h34*mzz34*r12*r32
                k += 1
                Mv[k] += h34*mzz34*r21*r31 + h34*mzz34*r22*r32
                k += 1
                Mv[k] += h34*mzz34*r31**2 + h34*mzz34*r32**2
                k += 1
                Mv[k] += h44*mz4*r11*r32 - h44*mz4*r12*r31
                k += 1
                Mv[k] += h44*mz4*r21*r32 - h44*mz4*r22*r31
                k += 1
                Mv[k] += h44*mzz44*r11*r31 + h44*mzz44*r12*r32
                k += 1
                Mv[k] += h44*mzz44*r21*r31 + h44*mzz44*r22*r32
                k += 1
                Mv[k] += h44*mzz44*r31**2 + h44*mzz44*r32**2

            elif mtype == 2: # M_red mass matrix purely by reduced integration
                k = self.init_k_M
                Mr[k] = 0+c1
                Mc[k] = 0+c1
                k += 1
                Mr[k] = 0+c1
                Mc[k] = 1+c1
                k += 1
                Mr[k] = 0+c1
                Mc[k] = 2+c1
                k += 1
                Mr[k] = 0+c1
                Mc[k] = 4+c1
                k += 1
                Mr[k] = 0+c1
                Mc[k] = 5+c1
                k += 1
                Mr[k] = 0+c1
                Mc[k] = 0+c2
                k += 1
                Mr[k] = 0+c1
                Mc[k] = 1+c2
                k += 1
                Mr[k] = 0+c1
                Mc[k] = 2+c2
                k += 1
                Mr[k] = 0+c1
                Mc[k] = 4+c2
                k += 1
                Mr[k] = 0+c1
                Mc[k] = 5+c2
                k += 1
                Mr[k] = 0+c1
                Mc[k] = 0+c3
                k += 1
                Mr[k] = 0+c1
                Mc[k] = 1+c3
                k += 1
                Mr[k] = 0+c1
                Mc[k] = 2+c3
                k += 1
                Mr[k] = 0+c1
                Mc[k] = 4+c3
                k += 1
                Mr[k] = 0+c1
                Mc[k] = 5+c3
                k += 1
                Mr[k] = 0+c1
                Mc[k] = 0+c4
                k += 1
                Mr[k] = 0+c1
                Mc[k] = 1+c4
                k += 1
                Mr[k] = 0+c1
                Mc[k] = 2+c4
                k += 1
                Mr[k] = 0+c1
                Mc[k] = 4+c4
                k += 1
                Mr[k] = 0+c1
                Mc[k] = 5+c4
                k += 1
                Mr[k] = 1+c1
                Mc[k] = 0+c1
                k += 1
                Mr[k] = 1+c1
                Mc[k] = 1+c1
                k += 1
                Mr[k] = 1+c1
                Mc[k] = 2+c1
                k += 1
                Mr[k] = 1+c1
                Mc[k] = 3+c1
                k += 1
                Mr[k] = 1+c1
                Mc[k] = 5+c1
                k += 1
                Mr[k] = 1+c1
                Mc[k] = 0+c2
                k += 1
                Mr[k] = 1+c1
                Mc[k] = 1+c2
                k += 1
                Mr[k] = 1+c1
                Mc[k] = 2+c2
                k += 1
                Mr[k] = 1+c1
                Mc[k] = 3+c2
                k += 1
                Mr[k] = 1+c1
                Mc[k] = 5+c2
                k += 1
                Mr[k] = 1+c1
                Mc[k] = 0+c3
                k += 1
                Mr[k] = 1+c1
                Mc[k] = 1+c3
                k += 1
                Mr[k] = 1+c1
                Mc[k] = 2+c3
                k += 1
                Mr[k] = 1+c1
                Mc[k] = 3+c3
                k += 1
                Mr[k] = 1+c1
                Mc[k] = 5+c3
                k += 1
                Mr[k] = 1+c1
                Mc[k] = 0+c4
                k += 1
                Mr[k] = 1+c1
                Mc[k] = 1+c4
                k += 1
                Mr[k] = 1+c1
                Mc[k] = 2+c4
                k += 1
                Mr[k] = 1+c1
                Mc[k] = 3+c4
                k += 1
                Mr[k] = 1+c1
                Mc[k] = 5+c4
                k += 1
                Mr[k] = 2+c1
                Mc[k] = 0+c1
                k += 1
                Mr[k] = 2+c1
                Mc[k] = 1+c1
                k += 1
                Mr[k] = 2+c1
                Mc[k] = 2+c1
                k += 1
                Mr[k] = 2+c1
                Mc[k] = 3+c1
                k += 1
                Mr[k] = 2+c1
                Mc[k] = 4+c1
                k += 1
                Mr[k] = 2+c1
                Mc[k] = 0+c2
                k += 1
                Mr[k] = 2+c1
                Mc[k] = 1+c2
                k += 1
                Mr[k] = 2+c1
                Mc[k] = 2+c2
                k += 1
                Mr[k] = 2+c1
                Mc[k] = 3+c2
                k += 1
                Mr[k] = 2+c1
                Mc[k] = 4+c2
                k += 1
                Mr[k] = 2+c1
                Mc[k] = 0+c3
                k += 1
                Mr[k] = 2+c1
                Mc[k] = 1+c3
                k += 1
                Mr[k] = 2+c1
                Mc[k] = 2+c3
                k += 1
                Mr[k] = 2+c1
                Mc[k] = 3+c3
                k += 1
                Mr[k] = 2+c1
                Mc[k] = 4+c3
                k += 1
                Mr[k] = 2+c1
                Mc[k] = 0+c4
                k += 1
                Mr[k] = 2+c1
                Mc[k] = 1+c4
                k += 1
                Mr[k] = 2+c1
                Mc[k] = 2+c4
                k += 1
                Mr[k] = 2+c1
                Mc[k] = 3+c4
                k += 1
                Mr[k] = 2+c1
                Mc[k] = 4+c4
                k += 1
                Mr[k] = 3+c1
                Mc[k] = 1+c1
                k += 1
                Mr[k] = 3+c1
                Mc[k] = 2+c1
                k += 1
                Mr[k] = 3+c1
                Mc[k] = 3+c1
                k += 1
                Mr[k] = 3+c1
                Mc[k] = 4+c1
                k += 1
                Mr[k] = 3+c1
                Mc[k] = 5+c1
                k += 1
                Mr[k] = 3+c1
                Mc[k] = 1+c2
                k += 1
                Mr[k] = 3+c1
                Mc[k] = 2+c2
                k += 1
                Mr[k] = 3+c1
                Mc[k] = 3+c2
                k += 1
                Mr[k] = 3+c1
                Mc[k] = 4+c2
                k += 1
                Mr[k] = 3+c1
                Mc[k] = 5+c2
                k += 1
                Mr[k] = 3+c1
                Mc[k] = 1+c3
                k += 1
                Mr[k] = 3+c1
                Mc[k] = 2+c3
                k += 1
                Mr[k] = 3+c1
                Mc[k] = 3+c3
                k += 1
                Mr[k] = 3+c1
                Mc[k] = 4+c3
                k += 1
                Mr[k] = 3+c1
                Mc[k] = 5+c3
                k += 1
                Mr[k] = 3+c1
                Mc[k] = 1+c4
                k += 1
                Mr[k] = 3+c1
                Mc[k] = 2+c4
                k += 1
                Mr[k] = 3+c1
                Mc[k] = 3+c4
                k += 1
                Mr[k] = 3+c1
                Mc[k] = 4+c4
                k += 1
                Mr[k] = 3+c1
                Mc[k] = 5+c4
                k += 1
                Mr[k] = 4+c1
                Mc[k] = 0+c1
                k += 1
                Mr[k] = 4+c1
                Mc[k] = 2+c1
                k += 1
                Mr[k] = 4+c1
                Mc[k] = 3+c1
                k += 1
                Mr[k] = 4+c1
                Mc[k] = 4+c1
                k += 1
                Mr[k] = 4+c1
                Mc[k] = 5+c1
                k += 1
                Mr[k] = 4+c1
                Mc[k] = 0+c2
                k += 1
                Mr[k] = 4+c1
                Mc[k] = 2+c2
                k += 1
                Mr[k] = 4+c1
                Mc[k] = 3+c2
                k += 1
                Mr[k] = 4+c1
                Mc[k] = 4+c2
                k += 1
                Mr[k] = 4+c1
                Mc[k] = 5+c2
                k += 1
                Mr[k] = 4+c1
                Mc[k] = 0+c3
                k += 1
                Mr[k] = 4+c1
                Mc[k] = 2+c3
                k += 1
                Mr[k] = 4+c1
                Mc[k] = 3+c3
                k += 1
                Mr[k] = 4+c1
                Mc[k] = 4+c3
                k += 1
                Mr[k] = 4+c1
                Mc[k] = 5+c3
                k += 1
                Mr[k] = 4+c1
                Mc[k] = 0+c4
                k += 1
                Mr[k] = 4+c1
                Mc[k] = 2+c4
                k += 1
                Mr[k] = 4+c1
                Mc[k] = 3+c4
                k += 1
                Mr[k] = 4+c1
                Mc[k] = 4+c4
                k += 1
                Mr[k] = 4+c1
                Mc[k] = 5+c4
                k += 1
                Mr[k] = 5+c1
                Mc[k] = 0+c1
                k += 1
                Mr[k] = 5+c1
                Mc[k] = 1+c1
                k += 1
                Mr[k] = 5+c1
                Mc[k] = 3+c1
                k += 1
                Mr[k] = 5+c1
                Mc[k] = 4+c1
                k += 1
                Mr[k] = 5+c1
                Mc[k] = 5+c1
                k += 1
                Mr[k] = 5+c1
                Mc[k] = 0+c2
                k += 1
                Mr[k] = 5+c1
                Mc[k] = 1+c2
                k += 1
                Mr[k] = 5+c1
                Mc[k] = 3+c2
                k += 1
                Mr[k] = 5+c1
                Mc[k] = 4+c2
                k += 1
                Mr[k] = 5+c1
                Mc[k] = 5+c2
                k += 1
                Mr[k] = 5+c1
                Mc[k] = 0+c3
                k += 1
                Mr[k] = 5+c1
                Mc[k] = 1+c3
                k += 1
                Mr[k] = 5+c1
                Mc[k] = 3+c3
                k += 1
                Mr[k] = 5+c1
                Mc[k] = 4+c3
                k += 1
                Mr[k] = 5+c1
                Mc[k] = 5+c3
                k += 1
                Mr[k] = 5+c1
                Mc[k] = 0+c4
                k += 1
                Mr[k] = 5+c1
                Mc[k] = 1+c4
                k += 1
                Mr[k] = 5+c1
                Mc[k] = 3+c4
                k += 1
                Mr[k] = 5+c1
                Mc[k] = 4+c4
                k += 1
                Mr[k] = 5+c1
                Mc[k] = 5+c4
                k += 1
                Mr[k] = 0+c2
                Mc[k] = 0+c1
                k += 1
                Mr[k] = 0+c2
                Mc[k] = 1+c1
                k += 1
                Mr[k] = 0+c2
                Mc[k] = 2+c1
                k += 1
                Mr[k] = 0+c2
                Mc[k] = 4+c1
                k += 1
                Mr[k] = 0+c2
                Mc[k] = 5+c1
                k += 1
                Mr[k] = 0+c2
                Mc[k] = 0+c2
                k += 1
                Mr[k] = 0+c2
                Mc[k] = 1+c2
                k += 1
                Mr[k] = 0+c2
                Mc[k] = 2+c2
                k += 1
                Mr[k] = 0+c2
                Mc[k] = 4+c2
                k += 1
                Mr[k] = 0+c2
                Mc[k] = 5+c2
                k += 1
                Mr[k] = 0+c2
                Mc[k] = 0+c3
                k += 1
                Mr[k] = 0+c2
                Mc[k] = 1+c3
                k += 1
                Mr[k] = 0+c2
                Mc[k] = 2+c3
                k += 1
                Mr[k] = 0+c2
                Mc[k] = 4+c3
                k += 1
                Mr[k] = 0+c2
                Mc[k] = 5+c3
                k += 1
                Mr[k] = 0+c2
                Mc[k] = 0+c4
                k += 1
                Mr[k] = 0+c2
                Mc[k] = 1+c4
                k += 1
                Mr[k] = 0+c2
                Mc[k] = 2+c4
                k += 1
                Mr[k] = 0+c2
                Mc[k] = 4+c4
                k += 1
                Mr[k] = 0+c2
                Mc[k] = 5+c4
                k += 1
                Mr[k] = 1+c2
                Mc[k] = 0+c1
                k += 1
                Mr[k] = 1+c2
                Mc[k] = 1+c1
                k += 1
                Mr[k] = 1+c2
                Mc[k] = 2+c1
                k += 1
                Mr[k] = 1+c2
                Mc[k] = 3+c1
                k += 1
                Mr[k] = 1+c2
                Mc[k] = 5+c1
                k += 1
                Mr[k] = 1+c2
                Mc[k] = 0+c2
                k += 1
                Mr[k] = 1+c2
                Mc[k] = 1+c2
                k += 1
                Mr[k] = 1+c2
                Mc[k] = 2+c2
                k += 1
                Mr[k] = 1+c2
                Mc[k] = 3+c2
                k += 1
                Mr[k] = 1+c2
                Mc[k] = 5+c2
                k += 1
                Mr[k] = 1+c2
                Mc[k] = 0+c3
                k += 1
                Mr[k] = 1+c2
                Mc[k] = 1+c3
                k += 1
                Mr[k] = 1+c2
                Mc[k] = 2+c3
                k += 1
                Mr[k] = 1+c2
                Mc[k] = 3+c3
                k += 1
                Mr[k] = 1+c2
                Mc[k] = 5+c3
                k += 1
                Mr[k] = 1+c2
                Mc[k] = 0+c4
                k += 1
                Mr[k] = 1+c2
                Mc[k] = 1+c4
                k += 1
                Mr[k] = 1+c2
                Mc[k] = 2+c4
                k += 1
                Mr[k] = 1+c2
                Mc[k] = 3+c4
                k += 1
                Mr[k] = 1+c2
                Mc[k] = 5+c4
                k += 1
                Mr[k] = 2+c2
                Mc[k] = 0+c1
                k += 1
                Mr[k] = 2+c2
                Mc[k] = 1+c1
                k += 1
                Mr[k] = 2+c2
                Mc[k] = 2+c1
                k += 1
                Mr[k] = 2+c2
                Mc[k] = 3+c1
                k += 1
                Mr[k] = 2+c2
                Mc[k] = 4+c1
                k += 1
                Mr[k] = 2+c2
                Mc[k] = 0+c2
                k += 1
                Mr[k] = 2+c2
                Mc[k] = 1+c2
                k += 1
                Mr[k] = 2+c2
                Mc[k] = 2+c2
                k += 1
                Mr[k] = 2+c2
                Mc[k] = 3+c2
                k += 1
                Mr[k] = 2+c2
                Mc[k] = 4+c2
                k += 1
                Mr[k] = 2+c2
                Mc[k] = 0+c3
                k += 1
                Mr[k] = 2+c2
                Mc[k] = 1+c3
                k += 1
                Mr[k] = 2+c2
                Mc[k] = 2+c3
                k += 1
                Mr[k] = 2+c2
                Mc[k] = 3+c3
                k += 1
                Mr[k] = 2+c2
                Mc[k] = 4+c3
                k += 1
                Mr[k] = 2+c2
                Mc[k] = 0+c4
                k += 1
                Mr[k] = 2+c2
                Mc[k] = 1+c4
                k += 1
                Mr[k] = 2+c2
                Mc[k] = 2+c4
                k += 1
                Mr[k] = 2+c2
                Mc[k] = 3+c4
                k += 1
                Mr[k] = 2+c2
                Mc[k] = 4+c4
                k += 1
                Mr[k] = 3+c2
                Mc[k] = 1+c1
                k += 1
                Mr[k] = 3+c2
                Mc[k] = 2+c1
                k += 1
                Mr[k] = 3+c2
                Mc[k] = 3+c1
                k += 1
                Mr[k] = 3+c2
                Mc[k] = 4+c1
                k += 1
                Mr[k] = 3+c2
                Mc[k] = 5+c1
                k += 1
                Mr[k] = 3+c2
                Mc[k] = 1+c2
                k += 1
                Mr[k] = 3+c2
                Mc[k] = 2+c2
                k += 1
                Mr[k] = 3+c2
                Mc[k] = 3+c2
                k += 1
                Mr[k] = 3+c2
                Mc[k] = 4+c2
                k += 1
                Mr[k] = 3+c2
                Mc[k] = 5+c2
                k += 1
                Mr[k] = 3+c2
                Mc[k] = 1+c3
                k += 1
                Mr[k] = 3+c2
                Mc[k] = 2+c3
                k += 1
                Mr[k] = 3+c2
                Mc[k] = 3+c3
                k += 1
                Mr[k] = 3+c2
                Mc[k] = 4+c3
                k += 1
                Mr[k] = 3+c2
                Mc[k] = 5+c3
                k += 1
                Mr[k] = 3+c2
                Mc[k] = 1+c4
                k += 1
                Mr[k] = 3+c2
                Mc[k] = 2+c4
                k += 1
                Mr[k] = 3+c2
                Mc[k] = 3+c4
                k += 1
                Mr[k] = 3+c2
                Mc[k] = 4+c4
                k += 1
                Mr[k] = 3+c2
                Mc[k] = 5+c4
                k += 1
                Mr[k] = 4+c2
                Mc[k] = 0+c1
                k += 1
                Mr[k] = 4+c2
                Mc[k] = 2+c1
                k += 1
                Mr[k] = 4+c2
                Mc[k] = 3+c1
                k += 1
                Mr[k] = 4+c2
                Mc[k] = 4+c1
                k += 1
                Mr[k] = 4+c2
                Mc[k] = 5+c1
                k += 1
                Mr[k] = 4+c2
                Mc[k] = 0+c2
                k += 1
                Mr[k] = 4+c2
                Mc[k] = 2+c2
                k += 1
                Mr[k] = 4+c2
                Mc[k] = 3+c2
                k += 1
                Mr[k] = 4+c2
                Mc[k] = 4+c2
                k += 1
                Mr[k] = 4+c2
                Mc[k] = 5+c2
                k += 1
                Mr[k] = 4+c2
                Mc[k] = 0+c3
                k += 1
                Mr[k] = 4+c2
                Mc[k] = 2+c3
                k += 1
                Mr[k] = 4+c2
                Mc[k] = 3+c3
                k += 1
                Mr[k] = 4+c2
                Mc[k] = 4+c3
                k += 1
                Mr[k] = 4+c2
                Mc[k] = 5+c3
                k += 1
                Mr[k] = 4+c2
                Mc[k] = 0+c4
                k += 1
                Mr[k] = 4+c2
                Mc[k] = 2+c4
                k += 1
                Mr[k] = 4+c2
                Mc[k] = 3+c4
                k += 1
                Mr[k] = 4+c2
                Mc[k] = 4+c4
                k += 1
                Mr[k] = 4+c2
                Mc[k] = 5+c4
                k += 1
                Mr[k] = 5+c2
                Mc[k] = 0+c1
                k += 1
                Mr[k] = 5+c2
                Mc[k] = 1+c1
                k += 1
                Mr[k] = 5+c2
                Mc[k] = 3+c1
                k += 1
                Mr[k] = 5+c2
                Mc[k] = 4+c1
                k += 1
                Mr[k] = 5+c2
                Mc[k] = 5+c1
                k += 1
                Mr[k] = 5+c2
                Mc[k] = 0+c2
                k += 1
                Mr[k] = 5+c2
                Mc[k] = 1+c2
                k += 1
                Mr[k] = 5+c2
                Mc[k] = 3+c2
                k += 1
                Mr[k] = 5+c2
                Mc[k] = 4+c2
                k += 1
                Mr[k] = 5+c2
                Mc[k] = 5+c2
                k += 1
                Mr[k] = 5+c2
                Mc[k] = 0+c3
                k += 1
                Mr[k] = 5+c2
                Mc[k] = 1+c3
                k += 1
                Mr[k] = 5+c2
                Mc[k] = 3+c3
                k += 1
                Mr[k] = 5+c2
                Mc[k] = 4+c3
                k += 1
                Mr[k] = 5+c2
                Mc[k] = 5+c3
                k += 1
                Mr[k] = 5+c2
                Mc[k] = 0+c4
                k += 1
                Mr[k] = 5+c2
                Mc[k] = 1+c4
                k += 1
                Mr[k] = 5+c2
                Mc[k] = 3+c4
                k += 1
                Mr[k] = 5+c2
                Mc[k] = 4+c4
                k += 1
                Mr[k] = 5+c2
                Mc[k] = 5+c4
                k += 1
                Mr[k] = 0+c3
                Mc[k] = 0+c1
                k += 1
                Mr[k] = 0+c3
                Mc[k] = 1+c1
                k += 1
                Mr[k] = 0+c3
                Mc[k] = 2+c1
                k += 1
                Mr[k] = 0+c3
                Mc[k] = 4+c1
                k += 1
                Mr[k] = 0+c3
                Mc[k] = 5+c1
                k += 1
                Mr[k] = 0+c3
                Mc[k] = 0+c2
                k += 1
                Mr[k] = 0+c3
                Mc[k] = 1+c2
                k += 1
                Mr[k] = 0+c3
                Mc[k] = 2+c2
                k += 1
                Mr[k] = 0+c3
                Mc[k] = 4+c2
                k += 1
                Mr[k] = 0+c3
                Mc[k] = 5+c2
                k += 1
                Mr[k] = 0+c3
                Mc[k] = 0+c3
                k += 1
                Mr[k] = 0+c3
                Mc[k] = 1+c3
                k += 1
                Mr[k] = 0+c3
                Mc[k] = 2+c3
                k += 1
                Mr[k] = 0+c3
                Mc[k] = 4+c3
                k += 1
                Mr[k] = 0+c3
                Mc[k] = 5+c3
                k += 1
                Mr[k] = 0+c3
                Mc[k] = 0+c4
                k += 1
                Mr[k] = 0+c3
                Mc[k] = 1+c4
                k += 1
                Mr[k] = 0+c3
                Mc[k] = 2+c4
                k += 1
                Mr[k] = 0+c3
                Mc[k] = 4+c4
                k += 1
                Mr[k] = 0+c3
                Mc[k] = 5+c4
                k += 1
                Mr[k] = 1+c3
                Mc[k] = 0+c1
                k += 1
                Mr[k] = 1+c3
                Mc[k] = 1+c1
                k += 1
                Mr[k] = 1+c3
                Mc[k] = 2+c1
                k += 1
                Mr[k] = 1+c3
                Mc[k] = 3+c1
                k += 1
                Mr[k] = 1+c3
                Mc[k] = 5+c1
                k += 1
                Mr[k] = 1+c3
                Mc[k] = 0+c2
                k += 1
                Mr[k] = 1+c3
                Mc[k] = 1+c2
                k += 1
                Mr[k] = 1+c3
                Mc[k] = 2+c2
                k += 1
                Mr[k] = 1+c3
                Mc[k] = 3+c2
                k += 1
                Mr[k] = 1+c3
                Mc[k] = 5+c2
                k += 1
                Mr[k] = 1+c3
                Mc[k] = 0+c3
                k += 1
                Mr[k] = 1+c3
                Mc[k] = 1+c3
                k += 1
                Mr[k] = 1+c3
                Mc[k] = 2+c3
                k += 1
                Mr[k] = 1+c3
                Mc[k] = 3+c3
                k += 1
                Mr[k] = 1+c3
                Mc[k] = 5+c3
                k += 1
                Mr[k] = 1+c3
                Mc[k] = 0+c4
                k += 1
                Mr[k] = 1+c3
                Mc[k] = 1+c4
                k += 1
                Mr[k] = 1+c3
                Mc[k] = 2+c4
                k += 1
                Mr[k] = 1+c3
                Mc[k] = 3+c4
                k += 1
                Mr[k] = 1+c3
                Mc[k] = 5+c4
                k += 1
                Mr[k] = 2+c3
                Mc[k] = 0+c1
                k += 1
                Mr[k] = 2+c3
                Mc[k] = 1+c1
                k += 1
                Mr[k] = 2+c3
                Mc[k] = 2+c1
                k += 1
                Mr[k] = 2+c3
                Mc[k] = 3+c1
                k += 1
                Mr[k] = 2+c3
                Mc[k] = 4+c1
                k += 1
                Mr[k] = 2+c3
                Mc[k] = 0+c2
                k += 1
                Mr[k] = 2+c3
                Mc[k] = 1+c2
                k += 1
                Mr[k] = 2+c3
                Mc[k] = 2+c2
                k += 1
                Mr[k] = 2+c3
                Mc[k] = 3+c2
                k += 1
                Mr[k] = 2+c3
                Mc[k] = 4+c2
                k += 1
                Mr[k] = 2+c3
                Mc[k] = 0+c3
                k += 1
                Mr[k] = 2+c3
                Mc[k] = 1+c3
                k += 1
                Mr[k] = 2+c3
                Mc[k] = 2+c3
                k += 1
                Mr[k] = 2+c3
                Mc[k] = 3+c3
                k += 1
                Mr[k] = 2+c3
                Mc[k] = 4+c3
                k += 1
                Mr[k] = 2+c3
                Mc[k] = 0+c4
                k += 1
                Mr[k] = 2+c3
                Mc[k] = 1+c4
                k += 1
                Mr[k] = 2+c3
                Mc[k] = 2+c4
                k += 1
                Mr[k] = 2+c3
                Mc[k] = 3+c4
                k += 1
                Mr[k] = 2+c3
                Mc[k] = 4+c4
                k += 1
                Mr[k] = 3+c3
                Mc[k] = 1+c1
                k += 1
                Mr[k] = 3+c3
                Mc[k] = 2+c1
                k += 1
                Mr[k] = 3+c3
                Mc[k] = 3+c1
                k += 1
                Mr[k] = 3+c3
                Mc[k] = 4+c1
                k += 1
                Mr[k] = 3+c3
                Mc[k] = 5+c1
                k += 1
                Mr[k] = 3+c3
                Mc[k] = 1+c2
                k += 1
                Mr[k] = 3+c3
                Mc[k] = 2+c2
                k += 1
                Mr[k] = 3+c3
                Mc[k] = 3+c2
                k += 1
                Mr[k] = 3+c3
                Mc[k] = 4+c2
                k += 1
                Mr[k] = 3+c3
                Mc[k] = 5+c2
                k += 1
                Mr[k] = 3+c3
                Mc[k] = 1+c3
                k += 1
                Mr[k] = 3+c3
                Mc[k] = 2+c3
                k += 1
                Mr[k] = 3+c3
                Mc[k] = 3+c3
                k += 1
                Mr[k] = 3+c3
                Mc[k] = 4+c3
                k += 1
                Mr[k] = 3+c3
                Mc[k] = 5+c3
                k += 1
                Mr[k] = 3+c3
                Mc[k] = 1+c4
                k += 1
                Mr[k] = 3+c3
                Mc[k] = 2+c4
                k += 1
                Mr[k] = 3+c3
                Mc[k] = 3+c4
                k += 1
                Mr[k] = 3+c3
                Mc[k] = 4+c4
                k += 1
                Mr[k] = 3+c3
                Mc[k] = 5+c4
                k += 1
                Mr[k] = 4+c3
                Mc[k] = 0+c1
                k += 1
                Mr[k] = 4+c3
                Mc[k] = 2+c1
                k += 1
                Mr[k] = 4+c3
                Mc[k] = 3+c1
                k += 1
                Mr[k] = 4+c3
                Mc[k] = 4+c1
                k += 1
                Mr[k] = 4+c3
                Mc[k] = 5+c1
                k += 1
                Mr[k] = 4+c3
                Mc[k] = 0+c2
                k += 1
                Mr[k] = 4+c3
                Mc[k] = 2+c2
                k += 1
                Mr[k] = 4+c3
                Mc[k] = 3+c2
                k += 1
                Mr[k] = 4+c3
                Mc[k] = 4+c2
                k += 1
                Mr[k] = 4+c3
                Mc[k] = 5+c2
                k += 1
                Mr[k] = 4+c3
                Mc[k] = 0+c3
                k += 1
                Mr[k] = 4+c3
                Mc[k] = 2+c3
                k += 1
                Mr[k] = 4+c3
                Mc[k] = 3+c3
                k += 1
                Mr[k] = 4+c3
                Mc[k] = 4+c3
                k += 1
                Mr[k] = 4+c3
                Mc[k] = 5+c3
                k += 1
                Mr[k] = 4+c3
                Mc[k] = 0+c4
                k += 1
                Mr[k] = 4+c3
                Mc[k] = 2+c4
                k += 1
                Mr[k] = 4+c3
                Mc[k] = 3+c4
                k += 1
                Mr[k] = 4+c3
                Mc[k] = 4+c4
                k += 1
                Mr[k] = 4+c3
                Mc[k] = 5+c4
                k += 1
                Mr[k] = 5+c3
                Mc[k] = 0+c1
                k += 1
                Mr[k] = 5+c3
                Mc[k] = 1+c1
                k += 1
                Mr[k] = 5+c3
                Mc[k] = 3+c1
                k += 1
                Mr[k] = 5+c3
                Mc[k] = 4+c1
                k += 1
                Mr[k] = 5+c3
                Mc[k] = 5+c1
                k += 1
                Mr[k] = 5+c3
                Mc[k] = 0+c2
                k += 1
                Mr[k] = 5+c3
                Mc[k] = 1+c2
                k += 1
                Mr[k] = 5+c3
                Mc[k] = 3+c2
                k += 1
                Mr[k] = 5+c3
                Mc[k] = 4+c2
                k += 1
                Mr[k] = 5+c3
                Mc[k] = 5+c2
                k += 1
                Mr[k] = 5+c3
                Mc[k] = 0+c3
                k += 1
                Mr[k] = 5+c3
                Mc[k] = 1+c3
                k += 1
                Mr[k] = 5+c3
                Mc[k] = 3+c3
                k += 1
                Mr[k] = 5+c3
                Mc[k] = 4+c3
                k += 1
                Mr[k] = 5+c3
                Mc[k] = 5+c3
                k += 1
                Mr[k] = 5+c3
                Mc[k] = 0+c4
                k += 1
                Mr[k] = 5+c3
                Mc[k] = 1+c4
                k += 1
                Mr[k] = 5+c3
                Mc[k] = 3+c4
                k += 1
                Mr[k] = 5+c3
                Mc[k] = 4+c4
                k += 1
                Mr[k] = 5+c3
                Mc[k] = 5+c4
                k += 1
                Mr[k] = 0+c4
                Mc[k] = 0+c1
                k += 1
                Mr[k] = 0+c4
                Mc[k] = 1+c1
                k += 1
                Mr[k] = 0+c4
                Mc[k] = 2+c1
                k += 1
                Mr[k] = 0+c4
                Mc[k] = 4+c1
                k += 1
                Mr[k] = 0+c4
                Mc[k] = 5+c1
                k += 1
                Mr[k] = 0+c4
                Mc[k] = 0+c2
                k += 1
                Mr[k] = 0+c4
                Mc[k] = 1+c2
                k += 1
                Mr[k] = 0+c4
                Mc[k] = 2+c2
                k += 1
                Mr[k] = 0+c4
                Mc[k] = 4+c2
                k += 1
                Mr[k] = 0+c4
                Mc[k] = 5+c2
                k += 1
                Mr[k] = 0+c4
                Mc[k] = 0+c3
                k += 1
                Mr[k] = 0+c4
                Mc[k] = 1+c3
                k += 1
                Mr[k] = 0+c4
                Mc[k] = 2+c3
                k += 1
                Mr[k] = 0+c4
                Mc[k] = 4+c3
                k += 1
                Mr[k] = 0+c4
                Mc[k] = 5+c3
                k += 1
                Mr[k] = 0+c4
                Mc[k] = 0+c4
                k += 1
                Mr[k] = 0+c4
                Mc[k] = 1+c4
                k += 1
                Mr[k] = 0+c4
                Mc[k] = 2+c4
                k += 1
                Mr[k] = 0+c4
                Mc[k] = 4+c4
                k += 1
                Mr[k] = 0+c4
                Mc[k] = 5+c4
                k += 1
                Mr[k] = 1+c4
                Mc[k] = 0+c1
                k += 1
                Mr[k] = 1+c4
                Mc[k] = 1+c1
                k += 1
                Mr[k] = 1+c4
                Mc[k] = 2+c1
                k += 1
                Mr[k] = 1+c4
                Mc[k] = 3+c1
                k += 1
                Mr[k] = 1+c4
                Mc[k] = 5+c1
                k += 1
                Mr[k] = 1+c4
                Mc[k] = 0+c2
                k += 1
                Mr[k] = 1+c4
                Mc[k] = 1+c2
                k += 1
                Mr[k] = 1+c4
                Mc[k] = 2+c2
                k += 1
                Mr[k] = 1+c4
                Mc[k] = 3+c2
                k += 1
                Mr[k] = 1+c4
                Mc[k] = 5+c2
                k += 1
                Mr[k] = 1+c4
                Mc[k] = 0+c3
                k += 1
                Mr[k] = 1+c4
                Mc[k] = 1+c3
                k += 1
                Mr[k] = 1+c4
                Mc[k] = 2+c3
                k += 1
                Mr[k] = 1+c4
                Mc[k] = 3+c3
                k += 1
                Mr[k] = 1+c4
                Mc[k] = 5+c3
                k += 1
                Mr[k] = 1+c4
                Mc[k] = 0+c4
                k += 1
                Mr[k] = 1+c4
                Mc[k] = 1+c4
                k += 1
                Mr[k] = 1+c4
                Mc[k] = 2+c4
                k += 1
                Mr[k] = 1+c4
                Mc[k] = 3+c4
                k += 1
                Mr[k] = 1+c4
                Mc[k] = 5+c4
                k += 1
                Mr[k] = 2+c4
                Mc[k] = 0+c1
                k += 1
                Mr[k] = 2+c4
                Mc[k] = 1+c1
                k += 1
                Mr[k] = 2+c4
                Mc[k] = 2+c1
                k += 1
                Mr[k] = 2+c4
                Mc[k] = 3+c1
                k += 1
                Mr[k] = 2+c4
                Mc[k] = 4+c1
                k += 1
                Mr[k] = 2+c4
                Mc[k] = 0+c2
                k += 1
                Mr[k] = 2+c4
                Mc[k] = 1+c2
                k += 1
                Mr[k] = 2+c4
                Mc[k] = 2+c2
                k += 1
                Mr[k] = 2+c4
                Mc[k] = 3+c2
                k += 1
                Mr[k] = 2+c4
                Mc[k] = 4+c2
                k += 1
                Mr[k] = 2+c4
                Mc[k] = 0+c3
                k += 1
                Mr[k] = 2+c4
                Mc[k] = 1+c3
                k += 1
                Mr[k] = 2+c4
                Mc[k] = 2+c3
                k += 1
                Mr[k] = 2+c4
                Mc[k] = 3+c3
                k += 1
                Mr[k] = 2+c4
                Mc[k] = 4+c3
                k += 1
                Mr[k] = 2+c4
                Mc[k] = 0+c4
                k += 1
                Mr[k] = 2+c4
                Mc[k] = 1+c4
                k += 1
                Mr[k] = 2+c4
                Mc[k] = 2+c4
                k += 1
                Mr[k] = 2+c4
                Mc[k] = 3+c4
                k += 1
                Mr[k] = 2+c4
                Mc[k] = 4+c4
                k += 1
                Mr[k] = 3+c4
                Mc[k] = 1+c1
                k += 1
                Mr[k] = 3+c4
                Mc[k] = 2+c1
                k += 1
                Mr[k] = 3+c4
                Mc[k] = 3+c1
                k += 1
                Mr[k] = 3+c4
                Mc[k] = 4+c1
                k += 1
                Mr[k] = 3+c4
                Mc[k] = 5+c1
                k += 1
                Mr[k] = 3+c4
                Mc[k] = 1+c2
                k += 1
                Mr[k] = 3+c4
                Mc[k] = 2+c2
                k += 1
                Mr[k] = 3+c4
                Mc[k] = 3+c2
                k += 1
                Mr[k] = 3+c4
                Mc[k] = 4+c2
                k += 1
                Mr[k] = 3+c4
                Mc[k] = 5+c2
                k += 1
                Mr[k] = 3+c4
                Mc[k] = 1+c3
                k += 1
                Mr[k] = 3+c4
                Mc[k] = 2+c3
                k += 1
                Mr[k] = 3+c4
                Mc[k] = 3+c3
                k += 1
                Mr[k] = 3+c4
                Mc[k] = 4+c3
                k += 1
                Mr[k] = 3+c4
                Mc[k] = 5+c3
                k += 1
                Mr[k] = 3+c4
                Mc[k] = 1+c4
                k += 1
                Mr[k] = 3+c4
                Mc[k] = 2+c4
                k += 1
                Mr[k] = 3+c4
                Mc[k] = 3+c4
                k += 1
                Mr[k] = 3+c4
                Mc[k] = 4+c4
                k += 1
                Mr[k] = 3+c4
                Mc[k] = 5+c4
                k += 1
                Mr[k] = 4+c4
                Mc[k] = 0+c1
                k += 1
                Mr[k] = 4+c4
                Mc[k] = 2+c1
                k += 1
                Mr[k] = 4+c4
                Mc[k] = 3+c1
                k += 1
                Mr[k] = 4+c4
                Mc[k] = 4+c1
                k += 1
                Mr[k] = 4+c4
                Mc[k] = 5+c1
                k += 1
                Mr[k] = 4+c4
                Mc[k] = 0+c2
                k += 1
                Mr[k] = 4+c4
                Mc[k] = 2+c2
                k += 1
                Mr[k] = 4+c4
                Mc[k] = 3+c2
                k += 1
                Mr[k] = 4+c4
                Mc[k] = 4+c2
                k += 1
                Mr[k] = 4+c4
                Mc[k] = 5+c2
                k += 1
                Mr[k] = 4+c4
                Mc[k] = 0+c3
                k += 1
                Mr[k] = 4+c4
                Mc[k] = 2+c3
                k += 1
                Mr[k] = 4+c4
                Mc[k] = 3+c3
                k += 1
                Mr[k] = 4+c4
                Mc[k] = 4+c3
                k += 1
                Mr[k] = 4+c4
                Mc[k] = 5+c3
                k += 1
                Mr[k] = 4+c4
                Mc[k] = 0+c4
                k += 1
                Mr[k] = 4+c4
                Mc[k] = 2+c4
                k += 1
                Mr[k] = 4+c4
                Mc[k] = 3+c4
                k += 1
                Mr[k] = 4+c4
                Mc[k] = 4+c4
                k += 1
                Mr[k] = 4+c4
                Mc[k] = 5+c4
                k += 1
                Mr[k] = 5+c4
                Mc[k] = 0+c1
                k += 1
                Mr[k] = 5+c4
                Mc[k] = 1+c1
                k += 1
                Mr[k] = 5+c4
                Mc[k] = 3+c1
                k += 1
                Mr[k] = 5+c4
                Mc[k] = 4+c1
                k += 1
                Mr[k] = 5+c4
                Mc[k] = 5+c1
                k += 1
                Mr[k] = 5+c4
                Mc[k] = 0+c2
                k += 1
                Mr[k] = 5+c4
                Mc[k] = 1+c2
                k += 1
                Mr[k] = 5+c4
                Mc[k] = 3+c2
                k += 1
                Mr[k] = 5+c4
                Mc[k] = 4+c2
                k += 1
                Mr[k] = 5+c4
                Mc[k] = 5+c2
                k += 1
                Mr[k] = 5+c4
                Mc[k] = 0+c3
                k += 1
                Mr[k] = 5+c4
                Mc[k] = 1+c3
                k += 1
                Mr[k] = 5+c4
                Mc[k] = 3+c3
                k += 1
                Mr[k] = 5+c4
                Mc[k] = 4+c3
                k += 1
                Mr[k] = 5+c4
                Mc[k] = 5+c3
                k += 1
                Mr[k] = 5+c4
                Mc[k] = 0+c4
                k += 1
                Mr[k] = 5+c4
                Mc[k] = 1+c4
                k += 1
                Mr[k] = 5+c4
                Mc[k] = 3+c4
                k += 1
                Mr[k] = 5+c4
                Mc[k] = 4+c4
                k += 1
                Mr[k] = 5+c4
                Mc[k] = 5+c4

                k = self.init_k_M
                Mv[k] += intrho*r11**2*valH1 + intrho*r12**2*valH1 + intrho*r13**2*valH1
                k += 1
                Mv[k] += intrho*r11*r21*valH1 + intrho*r12*r22*valH1 + intrho*r13*r23*valH1
                k += 1
                Mv[k] += intrho*r11*r31*valH1 + intrho*r12*r32*valH1 + intrho*r13*r33*valH1
                k += 1
                Mv[k] += mz1*r11*r22*valH1 - mz1*r12*r21*valH1
                k += 1
                Mv[k] += mz1*r11*r32*valH1 - mz1*r12*r31*valH1
                k += 1
                Mv[k] += intrho*r11**2*valH1 + intrho*r12**2*valH1 + intrho*r13**2*valH1
                k += 1
                Mv[k] += intrho*r11*r21*valH1 + intrho*r12*r22*valH1 + intrho*r13*r23*valH1
                k += 1
                Mv[k] += intrho*r11*r31*valH1 + intrho*r12*r32*valH1 + intrho*r13*r33*valH1
                k += 1
                Mv[k] += mz2*r11*r22*valH1 - mz2*r12*r21*valH1
                k += 1
                Mv[k] += mz2*r11*r32*valH1 - mz2*r12*r31*valH1
                k += 1
                Mv[k] += intrho*r11**2*valH1 + intrho*r12**2*valH1 + intrho*r13**2*valH1
                k += 1
                Mv[k] += intrho*r11*r21*valH1 + intrho*r12*r22*valH1 + intrho*r13*r23*valH1
                k += 1
                Mv[k] += intrho*r11*r31*valH1 + intrho*r12*r32*valH1 + intrho*r13*r33*valH1
                k += 1
                Mv[k] += mz3*r11*r22*valH1 - mz3*r12*r21*valH1
                k += 1
                Mv[k] += mz3*r11*r32*valH1 - mz3*r12*r31*valH1
                k += 1
                Mv[k] += intrho*r11**2*valH1 + intrho*r12**2*valH1 + intrho*r13**2*valH1
                k += 1
                Mv[k] += intrho*r11*r21*valH1 + intrho*r12*r22*valH1 + intrho*r13*r23*valH1
                k += 1
                Mv[k] += intrho*r11*r31*valH1 + intrho*r12*r32*valH1 + intrho*r13*r33*valH1
                k += 1
                Mv[k] += mz4*r11*r22*valH1 - mz4*r12*r21*valH1
                k += 1
                Mv[k] += mz4*r11*r32*valH1 - mz4*r12*r31*valH1
                k += 1
                Mv[k] += intrho*r11*r21*valH1 + intrho*r12*r22*valH1 + intrho*r13*r23*valH1
                k += 1
                Mv[k] += intrho*r21**2*valH1 + intrho*r22**2*valH1 + intrho*r23**2*valH1
                k += 1
                Mv[k] += intrho*r21*r31*valH1 + intrho*r22*r32*valH1 + intrho*r23*r33*valH1
                k += 1
                Mv[k] += -mz1*r11*r22*valH1 + mz1*r12*r21*valH1
                k += 1
                Mv[k] += mz1*r21*r32*valH1 - mz1*r22*r31*valH1
                k += 1
                Mv[k] += intrho*r11*r21*valH1 + intrho*r12*r22*valH1 + intrho*r13*r23*valH1
                k += 1
                Mv[k] += intrho*r21**2*valH1 + intrho*r22**2*valH1 + intrho*r23**2*valH1
                k += 1
                Mv[k] += intrho*r21*r31*valH1 + intrho*r22*r32*valH1 + intrho*r23*r33*valH1
                k += 1
                Mv[k] += -mz2*r11*r22*valH1 + mz2*r12*r21*valH1
                k += 1
                Mv[k] += mz2*r21*r32*valH1 - mz2*r22*r31*valH1
                k += 1
                Mv[k] += intrho*r11*r21*valH1 + intrho*r12*r22*valH1 + intrho*r13*r23*valH1
                k += 1
                Mv[k] += intrho*r21**2*valH1 + intrho*r22**2*valH1 + intrho*r23**2*valH1
                k += 1
                Mv[k] += intrho*r21*r31*valH1 + intrho*r22*r32*valH1 + intrho*r23*r33*valH1
                k += 1
                Mv[k] += -mz3*r11*r22*valH1 + mz3*r12*r21*valH1
                k += 1
                Mv[k] += mz3*r21*r32*valH1 - mz3*r22*r31*valH1
                k += 1
                Mv[k] += intrho*r11*r21*valH1 + intrho*r12*r22*valH1 + intrho*r13*r23*valH1
                k += 1
                Mv[k] += intrho*r21**2*valH1 + intrho*r22**2*valH1 + intrho*r23**2*valH1
                k += 1
                Mv[k] += intrho*r21*r31*valH1 + intrho*r22*r32*valH1 + intrho*r23*r33*valH1
                k += 1
                Mv[k] += -mz4*r11*r22*valH1 + mz4*r12*r21*valH1
                k += 1
                Mv[k] += mz4*r21*r32*valH1 - mz4*r22*r31*valH1
                k += 1
                Mv[k] += intrho*r11*r31*valH1 + intrho*r12*r32*valH1 + intrho*r13*r33*valH1
                k += 1
                Mv[k] += intrho*r21*r31*valH1 + intrho*r22*r32*valH1 + intrho*r23*r33*valH1
                k += 1
                Mv[k] += intrho*r31**2*valH1 + intrho*r32**2*valH1 + intrho*r33**2*valH1
                k += 1
                Mv[k] += -mz1*r11*r32*valH1 + mz1*r12*r31*valH1
                k += 1
                Mv[k] += -mz1*r21*r32*valH1 + mz1*r22*r31*valH1
                k += 1
                Mv[k] += intrho*r11*r31*valH1 + intrho*r12*r32*valH1 + intrho*r13*r33*valH1
                k += 1
                Mv[k] += intrho*r21*r31*valH1 + intrho*r22*r32*valH1 + intrho*r23*r33*valH1
                k += 1
                Mv[k] += intrho*r31**2*valH1 + intrho*r32**2*valH1 + intrho*r33**2*valH1
                k += 1
                Mv[k] += -mz2*r11*r32*valH1 + mz2*r12*r31*valH1
                k += 1
                Mv[k] += -mz2*r21*r32*valH1 + mz2*r22*r31*valH1
                k += 1
                Mv[k] += intrho*r11*r31*valH1 + intrho*r12*r32*valH1 + intrho*r13*r33*valH1
                k += 1
                Mv[k] += intrho*r21*r31*valH1 + intrho*r22*r32*valH1 + intrho*r23*r33*valH1
                k += 1
                Mv[k] += intrho*r31**2*valH1 + intrho*r32**2*valH1 + intrho*r33**2*valH1
                k += 1
                Mv[k] += -mz3*r11*r32*valH1 + mz3*r12*r31*valH1
                k += 1
                Mv[k] += -mz3*r21*r32*valH1 + mz3*r22*r31*valH1
                k += 1
                Mv[k] += intrho*r11*r31*valH1 + intrho*r12*r32*valH1 + intrho*r13*r33*valH1
                k += 1
                Mv[k] += intrho*r21*r31*valH1 + intrho*r22*r32*valH1 + intrho*r23*r33*valH1
                k += 1
                Mv[k] += intrho*r31**2*valH1 + intrho*r32**2*valH1 + intrho*r33**2*valH1
                k += 1
                Mv[k] += -mz4*r11*r32*valH1 + mz4*r12*r31*valH1
                k += 1
                Mv[k] += -mz4*r21*r32*valH1 + mz4*r22*r31*valH1
                k += 1
                Mv[k] += -mz1*r11*r22*valH1 + mz1*r12*r21*valH1
                k += 1
                Mv[k] += -mz1*r11*r32*valH1 + mz1*r12*r31*valH1
                k += 1
                Mv[k] += mzz11*r11**2*valH1 + mzz11*r12**2*valH1
                k += 1
                Mv[k] += mzz11*r11*r21*valH1 + mzz11*r12*r22*valH1
                k += 1
                Mv[k] += mzz11*r11*r31*valH1 + mzz11*r12*r32*valH1
                k += 1
                Mv[k] += -mz1*r11*r22*valH1 + mz1*r12*r21*valH1
                k += 1
                Mv[k] += -mz1*r11*r32*valH1 + mz1*r12*r31*valH1
                k += 1
                Mv[k] += mzz12*r11**2*valH1 + mzz12*r12**2*valH1
                k += 1
                Mv[k] += mzz12*r11*r21*valH1 + mzz12*r12*r22*valH1
                k += 1
                Mv[k] += mzz12*r11*r31*valH1 + mzz12*r12*r32*valH1
                k += 1
                Mv[k] += -mz1*r11*r22*valH1 + mz1*r12*r21*valH1
                k += 1
                Mv[k] += -mz1*r11*r32*valH1 + mz1*r12*r31*valH1
                k += 1
                Mv[k] += mzz13*r11**2*valH1 + mzz13*r12**2*valH1
                k += 1
                Mv[k] += mzz13*r11*r21*valH1 + mzz13*r12*r22*valH1
                k += 1
                Mv[k] += mzz13*r11*r31*valH1 + mzz13*r12*r32*valH1
                k += 1
                Mv[k] += -mz1*r11*r22*valH1 + mz1*r12*r21*valH1
                k += 1
                Mv[k] += -mz1*r11*r32*valH1 + mz1*r12*r31*valH1
                k += 1
                Mv[k] += mzz14*r11**2*valH1 + mzz14*r12**2*valH1
                k += 1
                Mv[k] += mzz14*r11*r21*valH1 + mzz14*r12*r22*valH1
                k += 1
                Mv[k] += mzz14*r11*r31*valH1 + mzz14*r12*r32*valH1
                k += 1
                Mv[k] += mz1*r11*r22*valH1 - mz1*r12*r21*valH1
                k += 1
                Mv[k] += -mz1*r21*r32*valH1 + mz1*r22*r31*valH1
                k += 1
                Mv[k] += mzz11*r11*r21*valH1 + mzz11*r12*r22*valH1
                k += 1
                Mv[k] += mzz11*r21**2*valH1 + mzz11*r22**2*valH1
                k += 1
                Mv[k] += mzz11*r21*r31*valH1 + mzz11*r22*r32*valH1
                k += 1
                Mv[k] += mz1*r11*r22*valH1 - mz1*r12*r21*valH1
                k += 1
                Mv[k] += -mz1*r21*r32*valH1 + mz1*r22*r31*valH1
                k += 1
                Mv[k] += mzz12*r11*r21*valH1 + mzz12*r12*r22*valH1
                k += 1
                Mv[k] += mzz12*r21**2*valH1 + mzz12*r22**2*valH1
                k += 1
                Mv[k] += mzz12*r21*r31*valH1 + mzz12*r22*r32*valH1
                k += 1
                Mv[k] += mz1*r11*r22*valH1 - mz1*r12*r21*valH1
                k += 1
                Mv[k] += -mz1*r21*r32*valH1 + mz1*r22*r31*valH1
                k += 1
                Mv[k] += mzz13*r11*r21*valH1 + mzz13*r12*r22*valH1
                k += 1
                Mv[k] += mzz13*r21**2*valH1 + mzz13*r22**2*valH1
                k += 1
                Mv[k] += mzz13*r21*r31*valH1 + mzz13*r22*r32*valH1
                k += 1
                Mv[k] += mz1*r11*r22*valH1 - mz1*r12*r21*valH1
                k += 1
                Mv[k] += -mz1*r21*r32*valH1 + mz1*r22*r31*valH1
                k += 1
                Mv[k] += mzz14*r11*r21*valH1 + mzz14*r12*r22*valH1
                k += 1
                Mv[k] += mzz14*r21**2*valH1 + mzz14*r22**2*valH1
                k += 1
                Mv[k] += mzz14*r21*r31*valH1 + mzz14*r22*r32*valH1
                k += 1
                Mv[k] += mz1*r11*r32*valH1 - mz1*r12*r31*valH1
                k += 1
                Mv[k] += mz1*r21*r32*valH1 - mz1*r22*r31*valH1
                k += 1
                Mv[k] += mzz11*r11*r31*valH1 + mzz11*r12*r32*valH1
                k += 1
                Mv[k] += mzz11*r21*r31*valH1 + mzz11*r22*r32*valH1
                k += 1
                Mv[k] += mzz11*r31**2*valH1 + mzz11*r32**2*valH1
                k += 1
                Mv[k] += mz1*r11*r32*valH1 - mz1*r12*r31*valH1
                k += 1
                Mv[k] += mz1*r21*r32*valH1 - mz1*r22*r31*valH1
                k += 1
                Mv[k] += mzz12*r11*r31*valH1 + mzz12*r12*r32*valH1
                k += 1
                Mv[k] += mzz12*r21*r31*valH1 + mzz12*r22*r32*valH1
                k += 1
                Mv[k] += mzz12*r31**2*valH1 + mzz12*r32**2*valH1
                k += 1
                Mv[k] += mz1*r11*r32*valH1 - mz1*r12*r31*valH1
                k += 1
                Mv[k] += mz1*r21*r32*valH1 - mz1*r22*r31*valH1
                k += 1
                Mv[k] += mzz13*r11*r31*valH1 + mzz13*r12*r32*valH1
                k += 1
                Mv[k] += mzz13*r21*r31*valH1 + mzz13*r22*r32*valH1
                k += 1
                Mv[k] += mzz13*r31**2*valH1 + mzz13*r32**2*valH1
                k += 1
                Mv[k] += mz1*r11*r32*valH1 - mz1*r12*r31*valH1
                k += 1
                Mv[k] += mz1*r21*r32*valH1 - mz1*r22*r31*valH1
                k += 1
                Mv[k] += mzz14*r11*r31*valH1 + mzz14*r12*r32*valH1
                k += 1
                Mv[k] += mzz14*r21*r31*valH1 + mzz14*r22*r32*valH1
                k += 1
                Mv[k] += mzz14*r31**2*valH1 + mzz14*r32**2*valH1
                k += 1
                Mv[k] += intrho*r11**2*valH1 + intrho*r12**2*valH1 + intrho*r13**2*valH1
                k += 1
                Mv[k] += intrho*r11*r21*valH1 + intrho*r12*r22*valH1 + intrho*r13*r23*valH1
                k += 1
                Mv[k] += intrho*r11*r31*valH1 + intrho*r12*r32*valH1 + intrho*r13*r33*valH1
                k += 1
                Mv[k] += mz1*r11*r22*valH1 - mz1*r12*r21*valH1
                k += 1
                Mv[k] += mz1*r11*r32*valH1 - mz1*r12*r31*valH1
                k += 1
                Mv[k] += intrho*r11**2*valH1 + intrho*r12**2*valH1 + intrho*r13**2*valH1
                k += 1
                Mv[k] += intrho*r11*r21*valH1 + intrho*r12*r22*valH1 + intrho*r13*r23*valH1
                k += 1
                Mv[k] += intrho*r11*r31*valH1 + intrho*r12*r32*valH1 + intrho*r13*r33*valH1
                k += 1
                Mv[k] += mz2*r11*r22*valH1 - mz2*r12*r21*valH1
                k += 1
                Mv[k] += mz2*r11*r32*valH1 - mz2*r12*r31*valH1
                k += 1
                Mv[k] += intrho*r11**2*valH1 + intrho*r12**2*valH1 + intrho*r13**2*valH1
                k += 1
                Mv[k] += intrho*r11*r21*valH1 + intrho*r12*r22*valH1 + intrho*r13*r23*valH1
                k += 1
                Mv[k] += intrho*r11*r31*valH1 + intrho*r12*r32*valH1 + intrho*r13*r33*valH1
                k += 1
                Mv[k] += mz3*r11*r22*valH1 - mz3*r12*r21*valH1
                k += 1
                Mv[k] += mz3*r11*r32*valH1 - mz3*r12*r31*valH1
                k += 1
                Mv[k] += intrho*r11**2*valH1 + intrho*r12**2*valH1 + intrho*r13**2*valH1
                k += 1
                Mv[k] += intrho*r11*r21*valH1 + intrho*r12*r22*valH1 + intrho*r13*r23*valH1
                k += 1
                Mv[k] += intrho*r11*r31*valH1 + intrho*r12*r32*valH1 + intrho*r13*r33*valH1
                k += 1
                Mv[k] += mz4*r11*r22*valH1 - mz4*r12*r21*valH1
                k += 1
                Mv[k] += mz4*r11*r32*valH1 - mz4*r12*r31*valH1
                k += 1
                Mv[k] += intrho*r11*r21*valH1 + intrho*r12*r22*valH1 + intrho*r13*r23*valH1
                k += 1
                Mv[k] += intrho*r21**2*valH1 + intrho*r22**2*valH1 + intrho*r23**2*valH1
                k += 1
                Mv[k] += intrho*r21*r31*valH1 + intrho*r22*r32*valH1 + intrho*r23*r33*valH1
                k += 1
                Mv[k] += -mz1*r11*r22*valH1 + mz1*r12*r21*valH1
                k += 1
                Mv[k] += mz1*r21*r32*valH1 - mz1*r22*r31*valH1
                k += 1
                Mv[k] += intrho*r11*r21*valH1 + intrho*r12*r22*valH1 + intrho*r13*r23*valH1
                k += 1
                Mv[k] += intrho*r21**2*valH1 + intrho*r22**2*valH1 + intrho*r23**2*valH1
                k += 1
                Mv[k] += intrho*r21*r31*valH1 + intrho*r22*r32*valH1 + intrho*r23*r33*valH1
                k += 1
                Mv[k] += -mz2*r11*r22*valH1 + mz2*r12*r21*valH1
                k += 1
                Mv[k] += mz2*r21*r32*valH1 - mz2*r22*r31*valH1
                k += 1
                Mv[k] += intrho*r11*r21*valH1 + intrho*r12*r22*valH1 + intrho*r13*r23*valH1
                k += 1
                Mv[k] += intrho*r21**2*valH1 + intrho*r22**2*valH1 + intrho*r23**2*valH1
                k += 1
                Mv[k] += intrho*r21*r31*valH1 + intrho*r22*r32*valH1 + intrho*r23*r33*valH1
                k += 1
                Mv[k] += -mz3*r11*r22*valH1 + mz3*r12*r21*valH1
                k += 1
                Mv[k] += mz3*r21*r32*valH1 - mz3*r22*r31*valH1
                k += 1
                Mv[k] += intrho*r11*r21*valH1 + intrho*r12*r22*valH1 + intrho*r13*r23*valH1
                k += 1
                Mv[k] += intrho*r21**2*valH1 + intrho*r22**2*valH1 + intrho*r23**2*valH1
                k += 1
                Mv[k] += intrho*r21*r31*valH1 + intrho*r22*r32*valH1 + intrho*r23*r33*valH1
                k += 1
                Mv[k] += -mz4*r11*r22*valH1 + mz4*r12*r21*valH1
                k += 1
                Mv[k] += mz4*r21*r32*valH1 - mz4*r22*r31*valH1
                k += 1
                Mv[k] += intrho*r11*r31*valH1 + intrho*r12*r32*valH1 + intrho*r13*r33*valH1
                k += 1
                Mv[k] += intrho*r21*r31*valH1 + intrho*r22*r32*valH1 + intrho*r23*r33*valH1
                k += 1
                Mv[k] += intrho*r31**2*valH1 + intrho*r32**2*valH1 + intrho*r33**2*valH1
                k += 1
                Mv[k] += -mz1*r11*r32*valH1 + mz1*r12*r31*valH1
                k += 1
                Mv[k] += -mz1*r21*r32*valH1 + mz1*r22*r31*valH1
                k += 1
                Mv[k] += intrho*r11*r31*valH1 + intrho*r12*r32*valH1 + intrho*r13*r33*valH1
                k += 1
                Mv[k] += intrho*r21*r31*valH1 + intrho*r22*r32*valH1 + intrho*r23*r33*valH1
                k += 1
                Mv[k] += intrho*r31**2*valH1 + intrho*r32**2*valH1 + intrho*r33**2*valH1
                k += 1
                Mv[k] += -mz2*r11*r32*valH1 + mz2*r12*r31*valH1
                k += 1
                Mv[k] += -mz2*r21*r32*valH1 + mz2*r22*r31*valH1
                k += 1
                Mv[k] += intrho*r11*r31*valH1 + intrho*r12*r32*valH1 + intrho*r13*r33*valH1
                k += 1
                Mv[k] += intrho*r21*r31*valH1 + intrho*r22*r32*valH1 + intrho*r23*r33*valH1
                k += 1
                Mv[k] += intrho*r31**2*valH1 + intrho*r32**2*valH1 + intrho*r33**2*valH1
                k += 1
                Mv[k] += -mz3*r11*r32*valH1 + mz3*r12*r31*valH1
                k += 1
                Mv[k] += -mz3*r21*r32*valH1 + mz3*r22*r31*valH1
                k += 1
                Mv[k] += intrho*r11*r31*valH1 + intrho*r12*r32*valH1 + intrho*r13*r33*valH1
                k += 1
                Mv[k] += intrho*r21*r31*valH1 + intrho*r22*r32*valH1 + intrho*r23*r33*valH1
                k += 1
                Mv[k] += intrho*r31**2*valH1 + intrho*r32**2*valH1 + intrho*r33**2*valH1
                k += 1
                Mv[k] += -mz4*r11*r32*valH1 + mz4*r12*r31*valH1
                k += 1
                Mv[k] += -mz4*r21*r32*valH1 + mz4*r22*r31*valH1
                k += 1
                Mv[k] += -mz2*r11*r22*valH1 + mz2*r12*r21*valH1
                k += 1
                Mv[k] += -mz2*r11*r32*valH1 + mz2*r12*r31*valH1
                k += 1
                Mv[k] += mzz12*r11**2*valH1 + mzz12*r12**2*valH1
                k += 1
                Mv[k] += mzz12*r11*r21*valH1 + mzz12*r12*r22*valH1
                k += 1
                Mv[k] += mzz12*r11*r31*valH1 + mzz12*r12*r32*valH1
                k += 1
                Mv[k] += -mz2*r11*r22*valH1 + mz2*r12*r21*valH1
                k += 1
                Mv[k] += -mz2*r11*r32*valH1 + mz2*r12*r31*valH1
                k += 1
                Mv[k] += mzz22*r11**2*valH1 + mzz22*r12**2*valH1
                k += 1
                Mv[k] += mzz22*r11*r21*valH1 + mzz22*r12*r22*valH1
                k += 1
                Mv[k] += mzz22*r11*r31*valH1 + mzz22*r12*r32*valH1
                k += 1
                Mv[k] += -mz2*r11*r22*valH1 + mz2*r12*r21*valH1
                k += 1
                Mv[k] += -mz2*r11*r32*valH1 + mz2*r12*r31*valH1
                k += 1
                Mv[k] += mzz23*r11**2*valH1 + mzz23*r12**2*valH1
                k += 1
                Mv[k] += mzz23*r11*r21*valH1 + mzz23*r12*r22*valH1
                k += 1
                Mv[k] += mzz23*r11*r31*valH1 + mzz23*r12*r32*valH1
                k += 1
                Mv[k] += -mz2*r11*r22*valH1 + mz2*r12*r21*valH1
                k += 1
                Mv[k] += -mz2*r11*r32*valH1 + mz2*r12*r31*valH1
                k += 1
                Mv[k] += mzz24*r11**2*valH1 + mzz24*r12**2*valH1
                k += 1
                Mv[k] += mzz24*r11*r21*valH1 + mzz24*r12*r22*valH1
                k += 1
                Mv[k] += mzz24*r11*r31*valH1 + mzz24*r12*r32*valH1
                k += 1
                Mv[k] += mz2*r11*r22*valH1 - mz2*r12*r21*valH1
                k += 1
                Mv[k] += -mz2*r21*r32*valH1 + mz2*r22*r31*valH1
                k += 1
                Mv[k] += mzz12*r11*r21*valH1 + mzz12*r12*r22*valH1
                k += 1
                Mv[k] += mzz12*r21**2*valH1 + mzz12*r22**2*valH1
                k += 1
                Mv[k] += mzz12*r21*r31*valH1 + mzz12*r22*r32*valH1
                k += 1
                Mv[k] += mz2*r11*r22*valH1 - mz2*r12*r21*valH1
                k += 1
                Mv[k] += -mz2*r21*r32*valH1 + mz2*r22*r31*valH1
                k += 1
                Mv[k] += mzz22*r11*r21*valH1 + mzz22*r12*r22*valH1
                k += 1
                Mv[k] += mzz22*r21**2*valH1 + mzz22*r22**2*valH1
                k += 1
                Mv[k] += mzz22*r21*r31*valH1 + mzz22*r22*r32*valH1
                k += 1
                Mv[k] += mz2*r11*r22*valH1 - mz2*r12*r21*valH1
                k += 1
                Mv[k] += -mz2*r21*r32*valH1 + mz2*r22*r31*valH1
                k += 1
                Mv[k] += mzz23*r11*r21*valH1 + mzz23*r12*r22*valH1
                k += 1
                Mv[k] += mzz23*r21**2*valH1 + mzz23*r22**2*valH1
                k += 1
                Mv[k] += mzz23*r21*r31*valH1 + mzz23*r22*r32*valH1
                k += 1
                Mv[k] += mz2*r11*r22*valH1 - mz2*r12*r21*valH1
                k += 1
                Mv[k] += -mz2*r21*r32*valH1 + mz2*r22*r31*valH1
                k += 1
                Mv[k] += mzz24*r11*r21*valH1 + mzz24*r12*r22*valH1
                k += 1
                Mv[k] += mzz24*r21**2*valH1 + mzz24*r22**2*valH1
                k += 1
                Mv[k] += mzz24*r21*r31*valH1 + mzz24*r22*r32*valH1
                k += 1
                Mv[k] += mz2*r11*r32*valH1 - mz2*r12*r31*valH1
                k += 1
                Mv[k] += mz2*r21*r32*valH1 - mz2*r22*r31*valH1
                k += 1
                Mv[k] += mzz12*r11*r31*valH1 + mzz12*r12*r32*valH1
                k += 1
                Mv[k] += mzz12*r21*r31*valH1 + mzz12*r22*r32*valH1
                k += 1
                Mv[k] += mzz12*r31**2*valH1 + mzz12*r32**2*valH1
                k += 1
                Mv[k] += mz2*r11*r32*valH1 - mz2*r12*r31*valH1
                k += 1
                Mv[k] += mz2*r21*r32*valH1 - mz2*r22*r31*valH1
                k += 1
                Mv[k] += mzz22*r11*r31*valH1 + mzz22*r12*r32*valH1
                k += 1
                Mv[k] += mzz22*r21*r31*valH1 + mzz22*r22*r32*valH1
                k += 1
                Mv[k] += mzz22*r31**2*valH1 + mzz22*r32**2*valH1
                k += 1
                Mv[k] += mz2*r11*r32*valH1 - mz2*r12*r31*valH1
                k += 1
                Mv[k] += mz2*r21*r32*valH1 - mz2*r22*r31*valH1
                k += 1
                Mv[k] += mzz23*r11*r31*valH1 + mzz23*r12*r32*valH1
                k += 1
                Mv[k] += mzz23*r21*r31*valH1 + mzz23*r22*r32*valH1
                k += 1
                Mv[k] += mzz23*r31**2*valH1 + mzz23*r32**2*valH1
                k += 1
                Mv[k] += mz2*r11*r32*valH1 - mz2*r12*r31*valH1
                k += 1
                Mv[k] += mz2*r21*r32*valH1 - mz2*r22*r31*valH1
                k += 1
                Mv[k] += mzz24*r11*r31*valH1 + mzz24*r12*r32*valH1
                k += 1
                Mv[k] += mzz24*r21*r31*valH1 + mzz24*r22*r32*valH1
                k += 1
                Mv[k] += mzz24*r31**2*valH1 + mzz24*r32**2*valH1
                k += 1
                Mv[k] += intrho*r11**2*valH1 + intrho*r12**2*valH1 + intrho*r13**2*valH1
                k += 1
                Mv[k] += intrho*r11*r21*valH1 + intrho*r12*r22*valH1 + intrho*r13*r23*valH1
                k += 1
                Mv[k] += intrho*r11*r31*valH1 + intrho*r12*r32*valH1 + intrho*r13*r33*valH1
                k += 1
                Mv[k] += mz1*r11*r22*valH1 - mz1*r12*r21*valH1
                k += 1
                Mv[k] += mz1*r11*r32*valH1 - mz1*r12*r31*valH1
                k += 1
                Mv[k] += intrho*r11**2*valH1 + intrho*r12**2*valH1 + intrho*r13**2*valH1
                k += 1
                Mv[k] += intrho*r11*r21*valH1 + intrho*r12*r22*valH1 + intrho*r13*r23*valH1
                k += 1
                Mv[k] += intrho*r11*r31*valH1 + intrho*r12*r32*valH1 + intrho*r13*r33*valH1
                k += 1
                Mv[k] += mz2*r11*r22*valH1 - mz2*r12*r21*valH1
                k += 1
                Mv[k] += mz2*r11*r32*valH1 - mz2*r12*r31*valH1
                k += 1
                Mv[k] += intrho*r11**2*valH1 + intrho*r12**2*valH1 + intrho*r13**2*valH1
                k += 1
                Mv[k] += intrho*r11*r21*valH1 + intrho*r12*r22*valH1 + intrho*r13*r23*valH1
                k += 1
                Mv[k] += intrho*r11*r31*valH1 + intrho*r12*r32*valH1 + intrho*r13*r33*valH1
                k += 1
                Mv[k] += mz3*r11*r22*valH1 - mz3*r12*r21*valH1
                k += 1
                Mv[k] += mz3*r11*r32*valH1 - mz3*r12*r31*valH1
                k += 1
                Mv[k] += intrho*r11**2*valH1 + intrho*r12**2*valH1 + intrho*r13**2*valH1
                k += 1
                Mv[k] += intrho*r11*r21*valH1 + intrho*r12*r22*valH1 + intrho*r13*r23*valH1
                k += 1
                Mv[k] += intrho*r11*r31*valH1 + intrho*r12*r32*valH1 + intrho*r13*r33*valH1
                k += 1
                Mv[k] += mz4*r11*r22*valH1 - mz4*r12*r21*valH1
                k += 1
                Mv[k] += mz4*r11*r32*valH1 - mz4*r12*r31*valH1
                k += 1
                Mv[k] += intrho*r11*r21*valH1 + intrho*r12*r22*valH1 + intrho*r13*r23*valH1
                k += 1
                Mv[k] += intrho*r21**2*valH1 + intrho*r22**2*valH1 + intrho*r23**2*valH1
                k += 1
                Mv[k] += intrho*r21*r31*valH1 + intrho*r22*r32*valH1 + intrho*r23*r33*valH1
                k += 1
                Mv[k] += -mz1*r11*r22*valH1 + mz1*r12*r21*valH1
                k += 1
                Mv[k] += mz1*r21*r32*valH1 - mz1*r22*r31*valH1
                k += 1
                Mv[k] += intrho*r11*r21*valH1 + intrho*r12*r22*valH1 + intrho*r13*r23*valH1
                k += 1
                Mv[k] += intrho*r21**2*valH1 + intrho*r22**2*valH1 + intrho*r23**2*valH1
                k += 1
                Mv[k] += intrho*r21*r31*valH1 + intrho*r22*r32*valH1 + intrho*r23*r33*valH1
                k += 1
                Mv[k] += -mz2*r11*r22*valH1 + mz2*r12*r21*valH1
                k += 1
                Mv[k] += mz2*r21*r32*valH1 - mz2*r22*r31*valH1
                k += 1
                Mv[k] += intrho*r11*r21*valH1 + intrho*r12*r22*valH1 + intrho*r13*r23*valH1
                k += 1
                Mv[k] += intrho*r21**2*valH1 + intrho*r22**2*valH1 + intrho*r23**2*valH1
                k += 1
                Mv[k] += intrho*r21*r31*valH1 + intrho*r22*r32*valH1 + intrho*r23*r33*valH1
                k += 1
                Mv[k] += -mz3*r11*r22*valH1 + mz3*r12*r21*valH1
                k += 1
                Mv[k] += mz3*r21*r32*valH1 - mz3*r22*r31*valH1
                k += 1
                Mv[k] += intrho*r11*r21*valH1 + intrho*r12*r22*valH1 + intrho*r13*r23*valH1
                k += 1
                Mv[k] += intrho*r21**2*valH1 + intrho*r22**2*valH1 + intrho*r23**2*valH1
                k += 1
                Mv[k] += intrho*r21*r31*valH1 + intrho*r22*r32*valH1 + intrho*r23*r33*valH1
                k += 1
                Mv[k] += -mz4*r11*r22*valH1 + mz4*r12*r21*valH1
                k += 1
                Mv[k] += mz4*r21*r32*valH1 - mz4*r22*r31*valH1
                k += 1
                Mv[k] += intrho*r11*r31*valH1 + intrho*r12*r32*valH1 + intrho*r13*r33*valH1
                k += 1
                Mv[k] += intrho*r21*r31*valH1 + intrho*r22*r32*valH1 + intrho*r23*r33*valH1
                k += 1
                Mv[k] += intrho*r31**2*valH1 + intrho*r32**2*valH1 + intrho*r33**2*valH1
                k += 1
                Mv[k] += -mz1*r11*r32*valH1 + mz1*r12*r31*valH1
                k += 1
                Mv[k] += -mz1*r21*r32*valH1 + mz1*r22*r31*valH1
                k += 1
                Mv[k] += intrho*r11*r31*valH1 + intrho*r12*r32*valH1 + intrho*r13*r33*valH1
                k += 1
                Mv[k] += intrho*r21*r31*valH1 + intrho*r22*r32*valH1 + intrho*r23*r33*valH1
                k += 1
                Mv[k] += intrho*r31**2*valH1 + intrho*r32**2*valH1 + intrho*r33**2*valH1
                k += 1
                Mv[k] += -mz2*r11*r32*valH1 + mz2*r12*r31*valH1
                k += 1
                Mv[k] += -mz2*r21*r32*valH1 + mz2*r22*r31*valH1
                k += 1
                Mv[k] += intrho*r11*r31*valH1 + intrho*r12*r32*valH1 + intrho*r13*r33*valH1
                k += 1
                Mv[k] += intrho*r21*r31*valH1 + intrho*r22*r32*valH1 + intrho*r23*r33*valH1
                k += 1
                Mv[k] += intrho*r31**2*valH1 + intrho*r32**2*valH1 + intrho*r33**2*valH1
                k += 1
                Mv[k] += -mz3*r11*r32*valH1 + mz3*r12*r31*valH1
                k += 1
                Mv[k] += -mz3*r21*r32*valH1 + mz3*r22*r31*valH1
                k += 1
                Mv[k] += intrho*r11*r31*valH1 + intrho*r12*r32*valH1 + intrho*r13*r33*valH1
                k += 1
                Mv[k] += intrho*r21*r31*valH1 + intrho*r22*r32*valH1 + intrho*r23*r33*valH1
                k += 1
                Mv[k] += intrho*r31**2*valH1 + intrho*r32**2*valH1 + intrho*r33**2*valH1
                k += 1
                Mv[k] += -mz4*r11*r32*valH1 + mz4*r12*r31*valH1
                k += 1
                Mv[k] += -mz4*r21*r32*valH1 + mz4*r22*r31*valH1
                k += 1
                Mv[k] += -mz3*r11*r22*valH1 + mz3*r12*r21*valH1
                k += 1
                Mv[k] += -mz3*r11*r32*valH1 + mz3*r12*r31*valH1
                k += 1
                Mv[k] += mzz13*r11**2*valH1 + mzz13*r12**2*valH1
                k += 1
                Mv[k] += mzz13*r11*r21*valH1 + mzz13*r12*r22*valH1
                k += 1
                Mv[k] += mzz13*r11*r31*valH1 + mzz13*r12*r32*valH1
                k += 1
                Mv[k] += -mz3*r11*r22*valH1 + mz3*r12*r21*valH1
                k += 1
                Mv[k] += -mz3*r11*r32*valH1 + mz3*r12*r31*valH1
                k += 1
                Mv[k] += mzz23*r11**2*valH1 + mzz23*r12**2*valH1
                k += 1
                Mv[k] += mzz23*r11*r21*valH1 + mzz23*r12*r22*valH1
                k += 1
                Mv[k] += mzz23*r11*r31*valH1 + mzz23*r12*r32*valH1
                k += 1
                Mv[k] += -mz3*r11*r22*valH1 + mz3*r12*r21*valH1
                k += 1
                Mv[k] += -mz3*r11*r32*valH1 + mz3*r12*r31*valH1
                k += 1
                Mv[k] += mzz33*r11**2*valH1 + mzz33*r12**2*valH1
                k += 1
                Mv[k] += mzz33*r11*r21*valH1 + mzz33*r12*r22*valH1
                k += 1
                Mv[k] += mzz33*r11*r31*valH1 + mzz33*r12*r32*valH1
                k += 1
                Mv[k] += -mz3*r11*r22*valH1 + mz3*r12*r21*valH1
                k += 1
                Mv[k] += -mz3*r11*r32*valH1 + mz3*r12*r31*valH1
                k += 1
                Mv[k] += mzz34*r11**2*valH1 + mzz34*r12**2*valH1
                k += 1
                Mv[k] += mzz34*r11*r21*valH1 + mzz34*r12*r22*valH1
                k += 1
                Mv[k] += mzz34*r11*r31*valH1 + mzz34*r12*r32*valH1
                k += 1
                Mv[k] += mz3*r11*r22*valH1 - mz3*r12*r21*valH1
                k += 1
                Mv[k] += -mz3*r21*r32*valH1 + mz3*r22*r31*valH1
                k += 1
                Mv[k] += mzz13*r11*r21*valH1 + mzz13*r12*r22*valH1
                k += 1
                Mv[k] += mzz13*r21**2*valH1 + mzz13*r22**2*valH1
                k += 1
                Mv[k] += mzz13*r21*r31*valH1 + mzz13*r22*r32*valH1
                k += 1
                Mv[k] += mz3*r11*r22*valH1 - mz3*r12*r21*valH1
                k += 1
                Mv[k] += -mz3*r21*r32*valH1 + mz3*r22*r31*valH1
                k += 1
                Mv[k] += mzz23*r11*r21*valH1 + mzz23*r12*r22*valH1
                k += 1
                Mv[k] += mzz23*r21**2*valH1 + mzz23*r22**2*valH1
                k += 1
                Mv[k] += mzz23*r21*r31*valH1 + mzz23*r22*r32*valH1
                k += 1
                Mv[k] += mz3*r11*r22*valH1 - mz3*r12*r21*valH1
                k += 1
                Mv[k] += -mz3*r21*r32*valH1 + mz3*r22*r31*valH1
                k += 1
                Mv[k] += mzz33*r11*r21*valH1 + mzz33*r12*r22*valH1
                k += 1
                Mv[k] += mzz33*r21**2*valH1 + mzz33*r22**2*valH1
                k += 1
                Mv[k] += mzz33*r21*r31*valH1 + mzz33*r22*r32*valH1
                k += 1
                Mv[k] += mz3*r11*r22*valH1 - mz3*r12*r21*valH1
                k += 1
                Mv[k] += -mz3*r21*r32*valH1 + mz3*r22*r31*valH1
                k += 1
                Mv[k] += mzz34*r11*r21*valH1 + mzz34*r12*r22*valH1
                k += 1
                Mv[k] += mzz34*r21**2*valH1 + mzz34*r22**2*valH1
                k += 1
                Mv[k] += mzz34*r21*r31*valH1 + mzz34*r22*r32*valH1
                k += 1
                Mv[k] += mz3*r11*r32*valH1 - mz3*r12*r31*valH1
                k += 1
                Mv[k] += mz3*r21*r32*valH1 - mz3*r22*r31*valH1
                k += 1
                Mv[k] += mzz13*r11*r31*valH1 + mzz13*r12*r32*valH1
                k += 1
                Mv[k] += mzz13*r21*r31*valH1 + mzz13*r22*r32*valH1
                k += 1
                Mv[k] += mzz13*r31**2*valH1 + mzz13*r32**2*valH1
                k += 1
                Mv[k] += mz3*r11*r32*valH1 - mz3*r12*r31*valH1
                k += 1
                Mv[k] += mz3*r21*r32*valH1 - mz3*r22*r31*valH1
                k += 1
                Mv[k] += mzz23*r11*r31*valH1 + mzz23*r12*r32*valH1
                k += 1
                Mv[k] += mzz23*r21*r31*valH1 + mzz23*r22*r32*valH1
                k += 1
                Mv[k] += mzz23*r31**2*valH1 + mzz23*r32**2*valH1
                k += 1
                Mv[k] += mz3*r11*r32*valH1 - mz3*r12*r31*valH1
                k += 1
                Mv[k] += mz3*r21*r32*valH1 - mz3*r22*r31*valH1
                k += 1
                Mv[k] += mzz33*r11*r31*valH1 + mzz33*r12*r32*valH1
                k += 1
                Mv[k] += mzz33*r21*r31*valH1 + mzz33*r22*r32*valH1
                k += 1
                Mv[k] += mzz33*r31**2*valH1 + mzz33*r32**2*valH1
                k += 1
                Mv[k] += mz3*r11*r32*valH1 - mz3*r12*r31*valH1
                k += 1
                Mv[k] += mz3*r21*r32*valH1 - mz3*r22*r31*valH1
                k += 1
                Mv[k] += mzz34*r11*r31*valH1 + mzz34*r12*r32*valH1
                k += 1
                Mv[k] += mzz34*r21*r31*valH1 + mzz34*r22*r32*valH1
                k += 1
                Mv[k] += mzz34*r31**2*valH1 + mzz34*r32**2*valH1
                k += 1
                Mv[k] += intrho*r11**2*valH1 + intrho*r12**2*valH1 + intrho*r13**2*valH1
                k += 1
                Mv[k] += intrho*r11*r21*valH1 + intrho*r12*r22*valH1 + intrho*r13*r23*valH1
                k += 1
                Mv[k] += intrho*r11*r31*valH1 + intrho*r12*r32*valH1 + intrho*r13*r33*valH1
                k += 1
                Mv[k] += mz1*r11*r22*valH1 - mz1*r12*r21*valH1
                k += 1
                Mv[k] += mz1*r11*r32*valH1 - mz1*r12*r31*valH1
                k += 1
                Mv[k] += intrho*r11**2*valH1 + intrho*r12**2*valH1 + intrho*r13**2*valH1
                k += 1
                Mv[k] += intrho*r11*r21*valH1 + intrho*r12*r22*valH1 + intrho*r13*r23*valH1
                k += 1
                Mv[k] += intrho*r11*r31*valH1 + intrho*r12*r32*valH1 + intrho*r13*r33*valH1
                k += 1
                Mv[k] += mz2*r11*r22*valH1 - mz2*r12*r21*valH1
                k += 1
                Mv[k] += mz2*r11*r32*valH1 - mz2*r12*r31*valH1
                k += 1
                Mv[k] += intrho*r11**2*valH1 + intrho*r12**2*valH1 + intrho*r13**2*valH1
                k += 1
                Mv[k] += intrho*r11*r21*valH1 + intrho*r12*r22*valH1 + intrho*r13*r23*valH1
                k += 1
                Mv[k] += intrho*r11*r31*valH1 + intrho*r12*r32*valH1 + intrho*r13*r33*valH1
                k += 1
                Mv[k] += mz3*r11*r22*valH1 - mz3*r12*r21*valH1
                k += 1
                Mv[k] += mz3*r11*r32*valH1 - mz3*r12*r31*valH1
                k += 1
                Mv[k] += intrho*r11**2*valH1 + intrho*r12**2*valH1 + intrho*r13**2*valH1
                k += 1
                Mv[k] += intrho*r11*r21*valH1 + intrho*r12*r22*valH1 + intrho*r13*r23*valH1
                k += 1
                Mv[k] += intrho*r11*r31*valH1 + intrho*r12*r32*valH1 + intrho*r13*r33*valH1
                k += 1
                Mv[k] += mz4*r11*r22*valH1 - mz4*r12*r21*valH1
                k += 1
                Mv[k] += mz4*r11*r32*valH1 - mz4*r12*r31*valH1
                k += 1
                Mv[k] += intrho*r11*r21*valH1 + intrho*r12*r22*valH1 + intrho*r13*r23*valH1
                k += 1
                Mv[k] += intrho*r21**2*valH1 + intrho*r22**2*valH1 + intrho*r23**2*valH1
                k += 1
                Mv[k] += intrho*r21*r31*valH1 + intrho*r22*r32*valH1 + intrho*r23*r33*valH1
                k += 1
                Mv[k] += -mz1*r11*r22*valH1 + mz1*r12*r21*valH1
                k += 1
                Mv[k] += mz1*r21*r32*valH1 - mz1*r22*r31*valH1
                k += 1
                Mv[k] += intrho*r11*r21*valH1 + intrho*r12*r22*valH1 + intrho*r13*r23*valH1
                k += 1
                Mv[k] += intrho*r21**2*valH1 + intrho*r22**2*valH1 + intrho*r23**2*valH1
                k += 1
                Mv[k] += intrho*r21*r31*valH1 + intrho*r22*r32*valH1 + intrho*r23*r33*valH1
                k += 1
                Mv[k] += -mz2*r11*r22*valH1 + mz2*r12*r21*valH1
                k += 1
                Mv[k] += mz2*r21*r32*valH1 - mz2*r22*r31*valH1
                k += 1
                Mv[k] += intrho*r11*r21*valH1 + intrho*r12*r22*valH1 + intrho*r13*r23*valH1
                k += 1
                Mv[k] += intrho*r21**2*valH1 + intrho*r22**2*valH1 + intrho*r23**2*valH1
                k += 1
                Mv[k] += intrho*r21*r31*valH1 + intrho*r22*r32*valH1 + intrho*r23*r33*valH1
                k += 1
                Mv[k] += -mz3*r11*r22*valH1 + mz3*r12*r21*valH1
                k += 1
                Mv[k] += mz3*r21*r32*valH1 - mz3*r22*r31*valH1
                k += 1
                Mv[k] += intrho*r11*r21*valH1 + intrho*r12*r22*valH1 + intrho*r13*r23*valH1
                k += 1
                Mv[k] += intrho*r21**2*valH1 + intrho*r22**2*valH1 + intrho*r23**2*valH1
                k += 1
                Mv[k] += intrho*r21*r31*valH1 + intrho*r22*r32*valH1 + intrho*r23*r33*valH1
                k += 1
                Mv[k] += -mz4*r11*r22*valH1 + mz4*r12*r21*valH1
                k += 1
                Mv[k] += mz4*r21*r32*valH1 - mz4*r22*r31*valH1
                k += 1
                Mv[k] += intrho*r11*r31*valH1 + intrho*r12*r32*valH1 + intrho*r13*r33*valH1
                k += 1
                Mv[k] += intrho*r21*r31*valH1 + intrho*r22*r32*valH1 + intrho*r23*r33*valH1
                k += 1
                Mv[k] += intrho*r31**2*valH1 + intrho*r32**2*valH1 + intrho*r33**2*valH1
                k += 1
                Mv[k] += -mz1*r11*r32*valH1 + mz1*r12*r31*valH1
                k += 1
                Mv[k] += -mz1*r21*r32*valH1 + mz1*r22*r31*valH1
                k += 1
                Mv[k] += intrho*r11*r31*valH1 + intrho*r12*r32*valH1 + intrho*r13*r33*valH1
                k += 1
                Mv[k] += intrho*r21*r31*valH1 + intrho*r22*r32*valH1 + intrho*r23*r33*valH1
                k += 1
                Mv[k] += intrho*r31**2*valH1 + intrho*r32**2*valH1 + intrho*r33**2*valH1
                k += 1
                Mv[k] += -mz2*r11*r32*valH1 + mz2*r12*r31*valH1
                k += 1
                Mv[k] += -mz2*r21*r32*valH1 + mz2*r22*r31*valH1
                k += 1
                Mv[k] += intrho*r11*r31*valH1 + intrho*r12*r32*valH1 + intrho*r13*r33*valH1
                k += 1
                Mv[k] += intrho*r21*r31*valH1 + intrho*r22*r32*valH1 + intrho*r23*r33*valH1
                k += 1
                Mv[k] += intrho*r31**2*valH1 + intrho*r32**2*valH1 + intrho*r33**2*valH1
                k += 1
                Mv[k] += -mz3*r11*r32*valH1 + mz3*r12*r31*valH1
                k += 1
                Mv[k] += -mz3*r21*r32*valH1 + mz3*r22*r31*valH1
                k += 1
                Mv[k] += intrho*r11*r31*valH1 + intrho*r12*r32*valH1 + intrho*r13*r33*valH1
                k += 1
                Mv[k] += intrho*r21*r31*valH1 + intrho*r22*r32*valH1 + intrho*r23*r33*valH1
                k += 1
                Mv[k] += intrho*r31**2*valH1 + intrho*r32**2*valH1 + intrho*r33**2*valH1
                k += 1
                Mv[k] += -mz4*r11*r32*valH1 + mz4*r12*r31*valH1
                k += 1
                Mv[k] += -mz4*r21*r32*valH1 + mz4*r22*r31*valH1
                k += 1
                Mv[k] += -mz4*r11*r22*valH1 + mz4*r12*r21*valH1
                k += 1
                Mv[k] += -mz4*r11*r32*valH1 + mz4*r12*r31*valH1
                k += 1
                Mv[k] += mzz14*r11**2*valH1 + mzz14*r12**2*valH1
                k += 1
                Mv[k] += mzz14*r11*r21*valH1 + mzz14*r12*r22*valH1
                k += 1
                Mv[k] += mzz14*r11*r31*valH1 + mzz14*r12*r32*valH1
                k += 1
                Mv[k] += -mz4*r11*r22*valH1 + mz4*r12*r21*valH1
                k += 1
                Mv[k] += -mz4*r11*r32*valH1 + mz4*r12*r31*valH1
                k += 1
                Mv[k] += mzz24*r11**2*valH1 + mzz24*r12**2*valH1
                k += 1
                Mv[k] += mzz24*r11*r21*valH1 + mzz24*r12*r22*valH1
                k += 1
                Mv[k] += mzz24*r11*r31*valH1 + mzz24*r12*r32*valH1
                k += 1
                Mv[k] += -mz4*r11*r22*valH1 + mz4*r12*r21*valH1
                k += 1
                Mv[k] += -mz4*r11*r32*valH1 + mz4*r12*r31*valH1
                k += 1
                Mv[k] += mzz34*r11**2*valH1 + mzz34*r12**2*valH1
                k += 1
                Mv[k] += mzz34*r11*r21*valH1 + mzz34*r12*r22*valH1
                k += 1
                Mv[k] += mzz34*r11*r31*valH1 + mzz34*r12*r32*valH1
                k += 1
                Mv[k] += -mz4*r11*r22*valH1 + mz4*r12*r21*valH1
                k += 1
                Mv[k] += -mz4*r11*r32*valH1 + mz4*r12*r31*valH1
                k += 1
                Mv[k] += mzz44*r11**2*valH1 + mzz44*r12**2*valH1
                k += 1
                Mv[k] += mzz44*r11*r21*valH1 + mzz44*r12*r22*valH1
                k += 1
                Mv[k] += mzz44*r11*r31*valH1 + mzz44*r12*r32*valH1
                k += 1
                Mv[k] += mz4*r11*r22*valH1 - mz4*r12*r21*valH1
                k += 1
                Mv[k] += -mz4*r21*r32*valH1 + mz4*r22*r31*valH1
                k += 1
                Mv[k] += mzz14*r11*r21*valH1 + mzz14*r12*r22*valH1
                k += 1
                Mv[k] += mzz14*r21**2*valH1 + mzz14*r22**2*valH1
                k += 1
                Mv[k] += mzz14*r21*r31*valH1 + mzz14*r22*r32*valH1
                k += 1
                Mv[k] += mz4*r11*r22*valH1 - mz4*r12*r21*valH1
                k += 1
                Mv[k] += -mz4*r21*r32*valH1 + mz4*r22*r31*valH1
                k += 1
                Mv[k] += mzz24*r11*r21*valH1 + mzz24*r12*r22*valH1
                k += 1
                Mv[k] += mzz24*r21**2*valH1 + mzz24*r22**2*valH1
                k += 1
                Mv[k] += mzz24*r21*r31*valH1 + mzz24*r22*r32*valH1
                k += 1
                Mv[k] += mz4*r11*r22*valH1 - mz4*r12*r21*valH1
                k += 1
                Mv[k] += -mz4*r21*r32*valH1 + mz4*r22*r31*valH1
                k += 1
                Mv[k] += mzz34*r11*r21*valH1 + mzz34*r12*r22*valH1
                k += 1
                Mv[k] += mzz34*r21**2*valH1 + mzz34*r22**2*valH1
                k += 1
                Mv[k] += mzz34*r21*r31*valH1 + mzz34*r22*r32*valH1
                k += 1
                Mv[k] += mz4*r11*r22*valH1 - mz4*r12*r21*valH1
                k += 1
                Mv[k] += -mz4*r21*r32*valH1 + mz4*r22*r31*valH1
                k += 1
                Mv[k] += mzz44*r11*r21*valH1 + mzz44*r12*r22*valH1
                k += 1
                Mv[k] += mzz44*r21**2*valH1 + mzz44*r22**2*valH1
                k += 1
                Mv[k] += mzz44*r21*r31*valH1 + mzz44*r22*r32*valH1
                k += 1
                Mv[k] += mz4*r11*r32*valH1 - mz4*r12*r31*valH1
                k += 1
                Mv[k] += mz4*r21*r32*valH1 - mz4*r22*r31*valH1
                k += 1
                Mv[k] += mzz14*r11*r31*valH1 + mzz14*r12*r32*valH1
                k += 1
                Mv[k] += mzz14*r21*r31*valH1 + mzz14*r22*r32*valH1
                k += 1
                Mv[k] += mzz14*r31**2*valH1 + mzz14*r32**2*valH1
                k += 1
                Mv[k] += mz4*r11*r32*valH1 - mz4*r12*r31*valH1
                k += 1
                Mv[k] += mz4*r21*r32*valH1 - mz4*r22*r31*valH1
                k += 1
                Mv[k] += mzz24*r11*r31*valH1 + mzz24*r12*r32*valH1
                k += 1
                Mv[k] += mzz24*r21*r31*valH1 + mzz24*r22*r32*valH1
                k += 1
                Mv[k] += mzz24*r31**2*valH1 + mzz24*r32**2*valH1
                k += 1
                Mv[k] += mz4*r11*r32*valH1 - mz4*r12*r31*valH1
                k += 1
                Mv[k] += mz4*r21*r32*valH1 - mz4*r22*r31*valH1
                k += 1
                Mv[k] += mzz34*r11*r31*valH1 + mzz34*r12*r32*valH1
                k += 1
                Mv[k] += mzz34*r21*r31*valH1 + mzz34*r22*r32*valH1
                k += 1
                Mv[k] += mzz34*r31**2*valH1 + mzz34*r32**2*valH1
                k += 1
                Mv[k] += mz4*r11*r32*valH1 - mz4*r12*r31*valH1
                k += 1
                Mv[k] += mz4*r21*r32*valH1 - mz4*r22*r31*valH1
                k += 1
                Mv[k] += mzz44*r11*r31*valH1 + mzz44*r12*r32*valH1
                k += 1
                Mv[k] += mzz44*r21*r31*valH1 + mzz44*r22*r32*valH1
                k += 1
                Mv[k] += mzz44*r31**2*valH1 + mzz44*r32**2*valH1

            elif mtype == 1: # M_lump lumped mass matrix, two-point Gauss-Lobatto quadrature, see the Notes
                k = self.init_k_M
                Mr[k] = 0+c1
                Mc[k] = 0+c1
                k += 1
                Mr[k] = 0+c1
                Mc[k] = 1+c1
                k += 1
                Mr[k] = 0+c1
                Mc[k] = 2+c1
                k += 1
                Mr[k] = 0+c1
                Mc[k] = 4+c1
                k += 1
                Mr[k] = 0+c1
                Mc[k] = 5+c1
                k += 1
                Mr[k] = 1+c1
                Mc[k] = 0+c1
                k += 1
                Mr[k] = 1+c1
                Mc[k] = 1+c1
                k += 1
                Mr[k] = 1+c1
                Mc[k] = 2+c1
                k += 1
                Mr[k] = 1+c1
                Mc[k] = 3+c1
                k += 1
                Mr[k] = 1+c1
                Mc[k] = 5+c1
                k += 1
                Mr[k] = 2+c1
                Mc[k] = 0+c1
                k += 1
                Mr[k] = 2+c1
                Mc[k] = 1+c1
                k += 1
                Mr[k] = 2+c1
                Mc[k] = 2+c1
                k += 1
                Mr[k] = 2+c1
                Mc[k] = 3+c1
                k += 1
                Mr[k] = 2+c1
                Mc[k] = 4+c1
                k += 1
                Mr[k] = 3+c1
                Mc[k] = 1+c1
                k += 1
                Mr[k] = 3+c1
                Mc[k] = 2+c1
                k += 1
                Mr[k] = 3+c1
                Mc[k] = 3+c1
                k += 1
                Mr[k] = 3+c1
                Mc[k] = 4+c1
                k += 1
                Mr[k] = 3+c1
                Mc[k] = 5+c1
                k += 1
                Mr[k] = 4+c1
                Mc[k] = 0+c1
                k += 1
                Mr[k] = 4+c1
                Mc[k] = 2+c1
                k += 1
                Mr[k] = 4+c1
                Mc[k] = 3+c1
                k += 1
                Mr[k] = 4+c1
                Mc[k] = 4+c1
                k += 1
                Mr[k] = 4+c1
                Mc[k] = 5+c1
                k += 1
                Mr[k] = 5+c1
                Mc[k] = 0+c1
                k += 1
                Mr[k] = 5+c1
                Mc[k] = 1+c1
                k += 1
                Mr[k] = 5+c1
                Mc[k] = 3+c1
                k += 1
                Mr[k] = 5+c1
                Mc[k] = 4+c1
                k += 1
                Mr[k] = 5+c1
                Mc[k] = 5+c1
                k += 1
                Mr[k] = 0+c2
                Mc[k] = 0+c2
                k += 1
                Mr[k] = 0+c2
                Mc[k] = 1+c2
                k += 1
                Mr[k] = 0+c2
                Mc[k] = 2+c2
                k += 1
                Mr[k] = 0+c2
                Mc[k] = 4+c2
                k += 1
                Mr[k] = 0+c2
                Mc[k] = 5+c2
                k += 1
                Mr[k] = 1+c2
                Mc[k] = 0+c2
                k += 1
                Mr[k] = 1+c2
                Mc[k] = 1+c2
                k += 1
                Mr[k] = 1+c2
                Mc[k] = 2+c2
                k += 1
                Mr[k] = 1+c2
                Mc[k] = 3+c2
                k += 1
                Mr[k] = 1+c2
                Mc[k] = 5+c2
                k += 1
                Mr[k] = 2+c2
                Mc[k] = 0+c2
                k += 1
                Mr[k] = 2+c2
                Mc[k] = 1+c2
                k += 1
                Mr[k] = 2+c2
                Mc[k] = 2+c2
                k += 1
                Mr[k] = 2+c2
                Mc[k] = 3+c2
                k += 1
                Mr[k] = 2+c2
                Mc[k] = 4+c2
                k += 1
                Mr[k] = 3+c2
                Mc[k] = 1+c2
                k += 1
                Mr[k] = 3+c2
                Mc[k] = 2+c2
                k += 1
                Mr[k] = 3+c2
                Mc[k] = 3+c2
                k += 1
                Mr[k] = 3+c2
                Mc[k] = 4+c2
                k += 1
                Mr[k] = 3+c2
                Mc[k] = 5+c2
                k += 1
                Mr[k] = 4+c2
                Mc[k] = 0+c2
                k += 1
                Mr[k] = 4+c2
                Mc[k] = 2+c2
                k += 1
                Mr[k] = 4+c2
                Mc[k] = 3+c2
                k += 1
                Mr[k] = 4+c2
                Mc[k] = 4+c2
                k += 1
                Mr[k] = 4+c2
                Mc[k] = 5+c2
                k += 1
                Mr[k] = 5+c2
                Mc[k] = 0+c2
                k += 1
                Mr[k] = 5+c2
                Mc[k] = 1+c2
                k += 1
                Mr[k] = 5+c2
                Mc[k] = 3+c2
                k += 1
                Mr[k] = 5+c2
                Mc[k] = 4+c2
                k += 1
                Mr[k] = 5+c2
                Mc[k] = 5+c2
                k += 1
                Mr[k] = 0+c3
                Mc[k] = 0+c3
                k += 1
                Mr[k] = 0+c3
                Mc[k] = 1+c3
                k += 1
                Mr[k] = 0+c3
                Mc[k] = 2+c3
                k += 1
                Mr[k] = 0+c3
                Mc[k] = 4+c3
                k += 1
                Mr[k] = 0+c3
                Mc[k] = 5+c3
                k += 1
                Mr[k] = 1+c3
                Mc[k] = 0+c3
                k += 1
                Mr[k] = 1+c3
                Mc[k] = 1+c3
                k += 1
                Mr[k] = 1+c3
                Mc[k] = 2+c3
                k += 1
                Mr[k] = 1+c3
                Mc[k] = 3+c3
                k += 1
                Mr[k] = 1+c3
                Mc[k] = 5+c3
                k += 1
                Mr[k] = 2+c3
                Mc[k] = 0+c3
                k += 1
                Mr[k] = 2+c3
                Mc[k] = 1+c3
                k += 1
                Mr[k] = 2+c3
                Mc[k] = 2+c3
                k += 1
                Mr[k] = 2+c3
                Mc[k] = 3+c3
                k += 1
                Mr[k] = 2+c3
                Mc[k] = 4+c3
                k += 1
                Mr[k] = 3+c3
                Mc[k] = 1+c3
                k += 1
                Mr[k] = 3+c3
                Mc[k] = 2+c3
                k += 1
                Mr[k] = 3+c3
                Mc[k] = 3+c3
                k += 1
                Mr[k] = 3+c3
                Mc[k] = 4+c3
                k += 1
                Mr[k] = 3+c3
                Mc[k] = 5+c3
                k += 1
                Mr[k] = 4+c3
                Mc[k] = 0+c3
                k += 1
                Mr[k] = 4+c3
                Mc[k] = 2+c3
                k += 1
                Mr[k] = 4+c3
                Mc[k] = 3+c3
                k += 1
                Mr[k] = 4+c3
                Mc[k] = 4+c3
                k += 1
                Mr[k] = 4+c3
                Mc[k] = 5+c3
                k += 1
                Mr[k] = 5+c3
                Mc[k] = 0+c3
                k += 1
                Mr[k] = 5+c3
                Mc[k] = 1+c3
                k += 1
                Mr[k] = 5+c3
                Mc[k] = 3+c3
                k += 1
                Mr[k] = 5+c3
                Mc[k] = 4+c3
                k += 1
                Mr[k] = 5+c3
                Mc[k] = 5+c3
                k += 1
                Mr[k] = 0+c4
                Mc[k] = 0+c4
                k += 1
                Mr[k] = 0+c4
                Mc[k] = 1+c4
                k += 1
                Mr[k] = 0+c4
                Mc[k] = 2+c4
                k += 1
                Mr[k] = 0+c4
                Mc[k] = 4+c4
                k += 1
                Mr[k] = 0+c4
                Mc[k] = 5+c4
                k += 1
                Mr[k] = 1+c4
                Mc[k] = 0+c4
                k += 1
                Mr[k] = 1+c4
                Mc[k] = 1+c4
                k += 1
                Mr[k] = 1+c4
                Mc[k] = 2+c4
                k += 1
                Mr[k] = 1+c4
                Mc[k] = 3+c4
                k += 1
                Mr[k] = 1+c4
                Mc[k] = 5+c4
                k += 1
                Mr[k] = 2+c4
                Mc[k] = 0+c4
                k += 1
                Mr[k] = 2+c4
                Mc[k] = 1+c4
                k += 1
                Mr[k] = 2+c4
                Mc[k] = 2+c4
                k += 1
                Mr[k] = 2+c4
                Mc[k] = 3+c4
                k += 1
                Mr[k] = 2+c4
                Mc[k] = 4+c4
                k += 1
                Mr[k] = 3+c4
                Mc[k] = 1+c4
                k += 1
                Mr[k] = 3+c4
                Mc[k] = 2+c4
                k += 1
                Mr[k] = 3+c4
                Mc[k] = 3+c4
                k += 1
                Mr[k] = 3+c4
                Mc[k] = 4+c4
                k += 1
                Mr[k] = 3+c4
                Mc[k] = 5+c4
                k += 1
                Mr[k] = 4+c4
                Mc[k] = 0+c4
                k += 1
                Mr[k] = 4+c4
                Mc[k] = 2+c4
                k += 1
                Mr[k] = 4+c4
                Mc[k] = 3+c4
                k += 1
                Mr[k] = 4+c4
                Mc[k] = 4+c4
                k += 1
                Mr[k] = 4+c4
                Mc[k] = 5+c4
                k += 1
                Mr[k] = 5+c4
                Mc[k] = 0+c4
                k += 1
                Mr[k] = 5+c4
                Mc[k] = 1+c4
                k += 1
                Mr[k] = 5+c4
                Mc[k] = 3+c4
                k += 1
                Mr[k] = 5+c4
                Mc[k] = 4+c4
                k += 1
                Mr[k] = 5+c4
                Mc[k] = 5+c4

                wij = 1.
                # NOTE two-point Gauss-Lobatto quadrature in each direction,
                #      whose points are the nodes, where S_i is 1 at node i and
                #      0 at the other nodes, such that h_ij = 0 for i != j
                points[0] = -1.
                points[1] = +1.
                h11 = 0.
                h22 = 0.
                h33 = 0.
                h44 = 0.
                for i in range(2):
                    xi = points[i]
                    for j in range(2):
                        eta = points[j]
                        detJ = (-2*x1 + 2*x2 + (eta + 1)*(x1 - x2 + x3 - x4))*(-2*y1 + 2*y4 + (xi + 1)*(y1 - y2) + (xi + 1)*(y3 - y4))/16. - (-2*y1 + 2*y2 + (eta + 1)*(y1 - y2 + y3 - y4))*(-2*x1 + 2*x4 + (x1 - x2)*(xi + 1) + (x3 - x4)*(xi + 1))/16.
                        S1 = eta*xi/4. - eta/4. - xi/4. + 1/4.
                        S2 = -eta*xi/4. - eta/4. + xi/4. + 1/4.
                        S3 = eta*xi/4. + eta/4. + xi/4. + 1/4.
                        S4 = -eta*xi/4. + eta/4. - xi/4. + 1/4.
                        h11 += S1**2*detJ*wij
                        h22 += S2**2*detJ*wij
                        h33 += S3**2*detJ*wij
                        h44 += S4**2*detJ*wij

                k = self.init_k_M
                Mv[k] += h11*intrho*r11**2 + h11*intrho*r12**2 + h11*intrho*r13**2
                k += 1
                Mv[k] += h11*intrho*r11*r21 + h11*intrho*r12*r22 + h11*intrho*r13*r23
                k += 1
                Mv[k] += h11*intrho*r11*r31 + h11*intrho*r12*r32 + h11*intrho*r13*r33
                k += 1
                Mv[k] += h11*mz1*r11*r22 - h11*mz1*r12*r21
                k += 1
                Mv[k] += h11*mz1*r11*r32 - h11*mz1*r12*r31
                k += 1
                Mv[k] += h11*intrho*r11*r21 + h11*intrho*r12*r22 + h11*intrho*r13*r23
                k += 1
                Mv[k] += h11*intrho*r21**2 + h11*intrho*r22**2 + h11*intrho*r23**2
                k += 1
                Mv[k] += h11*intrho*r21*r31 + h11*intrho*r22*r32 + h11*intrho*r23*r33
                k += 1
                Mv[k] += -h11*mz1*r11*r22 + h11*mz1*r12*r21
                k += 1
                Mv[k] += h11*mz1*r21*r32 - h11*mz1*r22*r31
                k += 1
                Mv[k] += h11*intrho*r11*r31 + h11*intrho*r12*r32 + h11*intrho*r13*r33
                k += 1
                Mv[k] += h11*intrho*r21*r31 + h11*intrho*r22*r32 + h11*intrho*r23*r33
                k += 1
                Mv[k] += h11*intrho*r31**2 + h11*intrho*r32**2 + h11*intrho*r33**2
                k += 1
                Mv[k] += -h11*mz1*r11*r32 + h11*mz1*r12*r31
                k += 1
                Mv[k] += -h11*mz1*r21*r32 + h11*mz1*r22*r31
                k += 1
                Mv[k] += -h11*mz1*r11*r22 + h11*mz1*r12*r21
                k += 1
                Mv[k] += -h11*mz1*r11*r32 + h11*mz1*r12*r31
                k += 1
                Mv[k] += h11*mzz11*r11**2 + h11*mzz11*r12**2
                k += 1
                Mv[k] += h11*mzz11*r11*r21 + h11*mzz11*r12*r22
                k += 1
                Mv[k] += h11*mzz11*r11*r31 + h11*mzz11*r12*r32
                k += 1
                Mv[k] += h11*mz1*r11*r22 - h11*mz1*r12*r21
                k += 1
                Mv[k] += -h11*mz1*r21*r32 + h11*mz1*r22*r31
                k += 1
                Mv[k] += h11*mzz11*r11*r21 + h11*mzz11*r12*r22
                k += 1
                Mv[k] += h11*mzz11*r21**2 + h11*mzz11*r22**2
                k += 1
                Mv[k] += h11*mzz11*r21*r31 + h11*mzz11*r22*r32
                k += 1
                Mv[k] += h11*mz1*r11*r32 - h11*mz1*r12*r31
                k += 1
                Mv[k] += h11*mz1*r21*r32 - h11*mz1*r22*r31
                k += 1
                Mv[k] += h11*mzz11*r11*r31 + h11*mzz11*r12*r32
                k += 1
                Mv[k] += h11*mzz11*r21*r31 + h11*mzz11*r22*r32
                k += 1
                Mv[k] += h11*mzz11*r31**2 + h11*mzz11*r32**2
                k += 1
                Mv[k] += h22*intrho*r11**2 + h22*intrho*r12**2 + h22*intrho*r13**2
                k += 1
                Mv[k] += h22*intrho*r11*r21 + h22*intrho*r12*r22 + h22*intrho*r13*r23
                k += 1
                Mv[k] += h22*intrho*r11*r31 + h22*intrho*r12*r32 + h22*intrho*r13*r33
                k += 1
                Mv[k] += h22*mz2*r11*r22 - h22*mz2*r12*r21
                k += 1
                Mv[k] += h22*mz2*r11*r32 - h22*mz2*r12*r31
                k += 1
                Mv[k] += h22*intrho*r11*r21 + h22*intrho*r12*r22 + h22*intrho*r13*r23
                k += 1
                Mv[k] += h22*intrho*r21**2 + h22*intrho*r22**2 + h22*intrho*r23**2
                k += 1
                Mv[k] += h22*intrho*r21*r31 + h22*intrho*r22*r32 + h22*intrho*r23*r33
                k += 1
                Mv[k] += -h22*mz2*r11*r22 + h22*mz2*r12*r21
                k += 1
                Mv[k] += h22*mz2*r21*r32 - h22*mz2*r22*r31
                k += 1
                Mv[k] += h22*intrho*r11*r31 + h22*intrho*r12*r32 + h22*intrho*r13*r33
                k += 1
                Mv[k] += h22*intrho*r21*r31 + h22*intrho*r22*r32 + h22*intrho*r23*r33
                k += 1
                Mv[k] += h22*intrho*r31**2 + h22*intrho*r32**2 + h22*intrho*r33**2
                k += 1
                Mv[k] += -h22*mz2*r11*r32 + h22*mz2*r12*r31
                k += 1
                Mv[k] += -h22*mz2*r21*r32 + h22*mz2*r22*r31
                k += 1
                Mv[k] += -h22*mz2*r11*r22 + h22*mz2*r12*r21
                k += 1
                Mv[k] += -h22*mz2*r11*r32 + h22*mz2*r12*r31
                k += 1
                Mv[k] += h22*mzz22*r11**2 + h22*mzz22*r12**2
                k += 1
                Mv[k] += h22*mzz22*r11*r21 + h22*mzz22*r12*r22
                k += 1
                Mv[k] += h22*mzz22*r11*r31 + h22*mzz22*r12*r32
                k += 1
                Mv[k] += h22*mz2*r11*r22 - h22*mz2*r12*r21
                k += 1
                Mv[k] += -h22*mz2*r21*r32 + h22*mz2*r22*r31
                k += 1
                Mv[k] += h22*mzz22*r11*r21 + h22*mzz22*r12*r22
                k += 1
                Mv[k] += h22*mzz22*r21**2 + h22*mzz22*r22**2
                k += 1
                Mv[k] += h22*mzz22*r21*r31 + h22*mzz22*r22*r32
                k += 1
                Mv[k] += h22*mz2*r11*r32 - h22*mz2*r12*r31
                k += 1
                Mv[k] += h22*mz2*r21*r32 - h22*mz2*r22*r31
                k += 1
                Mv[k] += h22*mzz22*r11*r31 + h22*mzz22*r12*r32
                k += 1
                Mv[k] += h22*mzz22*r21*r31 + h22*mzz22*r22*r32
                k += 1
                Mv[k] += h22*mzz22*r31**2 + h22*mzz22*r32**2
                k += 1
                Mv[k] += h33*intrho*r11**2 + h33*intrho*r12**2 + h33*intrho*r13**2
                k += 1
                Mv[k] += h33*intrho*r11*r21 + h33*intrho*r12*r22 + h33*intrho*r13*r23
                k += 1
                Mv[k] += h33*intrho*r11*r31 + h33*intrho*r12*r32 + h33*intrho*r13*r33
                k += 1
                Mv[k] += h33*mz3*r11*r22 - h33*mz3*r12*r21
                k += 1
                Mv[k] += h33*mz3*r11*r32 - h33*mz3*r12*r31
                k += 1
                Mv[k] += h33*intrho*r11*r21 + h33*intrho*r12*r22 + h33*intrho*r13*r23
                k += 1
                Mv[k] += h33*intrho*r21**2 + h33*intrho*r22**2 + h33*intrho*r23**2
                k += 1
                Mv[k] += h33*intrho*r21*r31 + h33*intrho*r22*r32 + h33*intrho*r23*r33
                k += 1
                Mv[k] += -h33*mz3*r11*r22 + h33*mz3*r12*r21
                k += 1
                Mv[k] += h33*mz3*r21*r32 - h33*mz3*r22*r31
                k += 1
                Mv[k] += h33*intrho*r11*r31 + h33*intrho*r12*r32 + h33*intrho*r13*r33
                k += 1
                Mv[k] += h33*intrho*r21*r31 + h33*intrho*r22*r32 + h33*intrho*r23*r33
                k += 1
                Mv[k] += h33*intrho*r31**2 + h33*intrho*r32**2 + h33*intrho*r33**2
                k += 1
                Mv[k] += -h33*mz3*r11*r32 + h33*mz3*r12*r31
                k += 1
                Mv[k] += -h33*mz3*r21*r32 + h33*mz3*r22*r31
                k += 1
                Mv[k] += -h33*mz3*r11*r22 + h33*mz3*r12*r21
                k += 1
                Mv[k] += -h33*mz3*r11*r32 + h33*mz3*r12*r31
                k += 1
                Mv[k] += h33*mzz33*r11**2 + h33*mzz33*r12**2
                k += 1
                Mv[k] += h33*mzz33*r11*r21 + h33*mzz33*r12*r22
                k += 1
                Mv[k] += h33*mzz33*r11*r31 + h33*mzz33*r12*r32
                k += 1
                Mv[k] += h33*mz3*r11*r22 - h33*mz3*r12*r21
                k += 1
                Mv[k] += -h33*mz3*r21*r32 + h33*mz3*r22*r31
                k += 1
                Mv[k] += h33*mzz33*r11*r21 + h33*mzz33*r12*r22
                k += 1
                Mv[k] += h33*mzz33*r21**2 + h33*mzz33*r22**2
                k += 1
                Mv[k] += h33*mzz33*r21*r31 + h33*mzz33*r22*r32
                k += 1
                Mv[k] += h33*mz3*r11*r32 - h33*mz3*r12*r31
                k += 1
                Mv[k] += h33*mz3*r21*r32 - h33*mz3*r22*r31
                k += 1
                Mv[k] += h33*mzz33*r11*r31 + h33*mzz33*r12*r32
                k += 1
                Mv[k] += h33*mzz33*r21*r31 + h33*mzz33*r22*r32
                k += 1
                Mv[k] += h33*mzz33*r31**2 + h33*mzz33*r32**2
                k += 1
                Mv[k] += h44*intrho*r11**2 + h44*intrho*r12**2 + h44*intrho*r13**2
                k += 1
                Mv[k] += h44*intrho*r11*r21 + h44*intrho*r12*r22 + h44*intrho*r13*r23
                k += 1
                Mv[k] += h44*intrho*r11*r31 + h44*intrho*r12*r32 + h44*intrho*r13*r33
                k += 1
                Mv[k] += h44*mz4*r11*r22 - h44*mz4*r12*r21
                k += 1
                Mv[k] += h44*mz4*r11*r32 - h44*mz4*r12*r31
                k += 1
                Mv[k] += h44*intrho*r11*r21 + h44*intrho*r12*r22 + h44*intrho*r13*r23
                k += 1
                Mv[k] += h44*intrho*r21**2 + h44*intrho*r22**2 + h44*intrho*r23**2
                k += 1
                Mv[k] += h44*intrho*r21*r31 + h44*intrho*r22*r32 + h44*intrho*r23*r33
                k += 1
                Mv[k] += -h44*mz4*r11*r22 + h44*mz4*r12*r21
                k += 1
                Mv[k] += h44*mz4*r21*r32 - h44*mz4*r22*r31
                k += 1
                Mv[k] += h44*intrho*r11*r31 + h44*intrho*r12*r32 + h44*intrho*r13*r33
                k += 1
                Mv[k] += h44*intrho*r21*r31 + h44*intrho*r22*r32 + h44*intrho*r23*r33
                k += 1
                Mv[k] += h44*intrho*r31**2 + h44*intrho*r32**2 + h44*intrho*r33**2
                k += 1
                Mv[k] += -h44*mz4*r11*r32 + h44*mz4*r12*r31
                k += 1
                Mv[k] += -h44*mz4*r21*r32 + h44*mz4*r22*r31
                k += 1
                Mv[k] += -h44*mz4*r11*r22 + h44*mz4*r12*r21
                k += 1
                Mv[k] += -h44*mz4*r11*r32 + h44*mz4*r12*r31
                k += 1
                Mv[k] += h44*mzz44*r11**2 + h44*mzz44*r12**2
                k += 1
                Mv[k] += h44*mzz44*r11*r21 + h44*mzz44*r12*r22
                k += 1
                Mv[k] += h44*mzz44*r11*r31 + h44*mzz44*r12*r32
                k += 1
                Mv[k] += h44*mz4*r11*r22 - h44*mz4*r12*r21
                k += 1
                Mv[k] += -h44*mz4*r21*r32 + h44*mz4*r22*r31
                k += 1
                Mv[k] += h44*mzz44*r11*r21 + h44*mzz44*r12*r22
                k += 1
                Mv[k] += h44*mzz44*r21**2 + h44*mzz44*r22**2
                k += 1
                Mv[k] += h44*mzz44*r21*r31 + h44*mzz44*r22*r32
                k += 1
                Mv[k] += h44*mz4*r11*r32 - h44*mz4*r12*r31
                k += 1
                Mv[k] += h44*mz4*r21*r32 - h44*mz4*r22*r31
                k += 1
                Mv[k] += h44*mzz44*r11*r31 + h44*mzz44*r12*r32
                k += 1
                Mv[k] += h44*mzz44*r21*r31 + h44*mzz44*r22*r32
                k += 1
                Mv[k] += h44*mzz44*r31**2 + h44*mzz44*r32**2


    cpdef void update_KA_beta(Quad4R self,
                        long [::1] KA_betar,
                        long [::1] KA_betac,
                        double [::1] KA_betav,
                        ):
        r"""Update sparse vectors for piston-theory aerodynamic matrix `\boldsymbol{K}_{A_\beta}`

        The matrix is integrated over the element with 2 by 2 Gauss-Legendre
        quadrature, from the coordinates ``probe.xe`` and the rotation matrix,
        and its values are added to ``KA_betav`` at the positions
        ``init_k_KA_beta`` to ``init_k_KA_beta + KA_BETA_SPARSE_SIZE - 1``; the
        global row and column indices are always written to ``KA_betar`` and
        ``KA_betac`` at the same positions. Only the translational
        degrees-of-freedom appear, through their components along the element
        normal. No aerodynamic coefficient enters the values, which depend
        only on the geometry, so the caller multiplies the assembled matrix by
        it. Before this method is called, :meth:`.update_rotation_matrix` and
        :meth:`.update_probe_xe` must have been used.

        Parameters
        ----------
        KA_betar : np.array
            Array to store row positions of sparse values
            1-D array of integers (``INT``) of size at least
            ``init_k_KA_beta + KA_BETA_SPARSE_SIZE``.
        KA_betac : np.array
            Array to store column positions of sparse values
            Same type and size as ``KA_betar``.
        KA_betav : np.array
            Array to store sparse values
            1-D array of ``DOUBLE``, same size as ``KA_betar``. The values are
            added to the existing ones, with dimension of length.

        Notes
        -----
        With `w_n = T_{13} u + T_{23} v + T_{33} w` the displacement along
        the element normal, written with the global translations, the
        computed matrix is

        .. math::

            \boldsymbol{K}_{A_\beta} = -\int_A \boldsymbol{S}^\top \left( T_{11} \boldsymbol{S}_{,x}
                + T_{12} \boldsymbol{S}_{,y} \right) dA

        acting on the nodal values of `w_n`, where `\boldsymbol{S}` holds the bilinear
        shape functions and `\boldsymbol{S}_{,x}`, `\boldsymbol{S}_{,y}` their derivatives along
        the element axes. The element coordinates are `x = T_{11} X + T_{21}
        Y + T_{31} Z` and `y = T_{12} X + T_{22} Y + T_{32} Z`, such that
        `T_{11} \boldsymbol{S}_{,x} + T_{12} \boldsymbol{S}_{,y}` is the derivative along the
        global `X` axis, the direction of the flow, projected on the plane of
        the element.

        .. rubric:: Piston theory

        The matrices of :meth:`.update_KA_beta`, :meth:`.update_KA_gamma`
        and :meth:`.update_CA` are the matrices of the aerodynamic load
        along the element normal of a supersonic flow along the global `X`
        axis, flowing towards the positive `X`. The linear piston theory, with
        Krumhaar's correction for a cylinder [1]_, gives the pressure on the
        side of the flow `p = \beta w_{,X} + \mu \dot{w} - \gamma w`, with
        `w` the displacement towards the flow, and hence the
        load per unit area along the normal

        .. math::

            q = -\beta \, w_{n,X} + \gamma \, w_n - \mu \, \dot{w}_n

        whichever side of the element the flow is on, because `q` and `w_n`
        change sign together. The load vector is then `\boldsymbol{f}_A = \beta
        \boldsymbol{K}_{A_\beta} \boldsymbol{u} + \gamma \boldsymbol{K}_{A_\gamma} \boldsymbol{u} + \mu \boldsymbol{C}_A
        \dot{\boldsymbol{u}}` and the equations of motion are

        .. math::

            \boldsymbol{M} \ddot{\boldsymbol{u}} - \mu \boldsymbol{C}_A \dot{\boldsymbol{u}} + \left(\boldsymbol{K} - \beta
            \boldsymbol{K}_{A_\beta} - \gamma \boldsymbol{K}_{A_\gamma}\right) \boldsymbol{u} = \boldsymbol{0}

        The coefficients are given by the caller; with the density
        `\rho_{\infty}`, speed `U` and Mach number `M` of the flow, `\beta =
        \rho_{\infty} U^2/\sqrt{M^2 - 1}`, `\mu = \beta (M^2 - 2)/(U (M^2 -
        1))`, and `\gamma = \beta/(2 R \sqrt{M^2 - 1})` for the flow over the
        outer, convex, side of a cylinder of radius `R`, an outward
        displacement lowering the pressure, or `\gamma = 0` for a flat
        panel. The positive `\mu` damps the motion, and the positive
        `\gamma` reduces the stiffness.

        References
        ----------
        .. [1] Krumhaar, H., "The accuracy of linear piston theory when
           applied to cylindrical shells", AIAA Journal, 1(6), 1448-1449,
           1963.

        """
        cdef int c1, c2, c3, c4, i, j, k
        cdef double r11, r12, r13, r23, r33
        cdef double x1, x2, x3, x4
        cdef double y1, y2, y3, y4
        cdef double j11, j12, j21, j22
        cdef double S1, S2, S3, S4
        cdef double S1_x, S2_x, S3_x, S4_x
        cdef double S1_y, S2_y, S3_y, S4_y
        cdef double xi, eta, wij, detJ
        cdef double points[2]

        with nogil:

            # NOTE ignoring z in local coordinates
            x1 = self.probe.xe[0]
            y1 = self.probe.xe[1]
            # z1 = self.probe.xe[2]
            x2 = self.probe.xe[3]
            y2 = self.probe.xe[4]
            # z2 = self.probe.xe[5]
            x3 = self.probe.xe[6]
            y3 = self.probe.xe[7]
            # z3 = self.probe.xe[8]
            x4 = self.probe.xe[9]
            y4 = self.probe.xe[10]
            # z4 = self.probe.xe[11]

            # local to global transformation
            r11 = self.r11
            r12 = self.r12
            r13 = self.r13
            r23 = self.r23
            r33 = self.r33

            # positions the global matrices
            c1 = self.c1
            c2 = self.c2
            c3 = self.c3
            c4 = self.c4

            k = self.init_k_KA_beta
            KA_betar[k] = 0+c1
            KA_betac[k] = 0+c1
            k += 1
            KA_betar[k] = 0+c1
            KA_betac[k] = 1+c1
            k += 1
            KA_betar[k] = 0+c1
            KA_betac[k] = 2+c1
            k += 1
            KA_betar[k] = 0+c1
            KA_betac[k] = 0+c2
            k += 1
            KA_betar[k] = 0+c1
            KA_betac[k] = 1+c2
            k += 1
            KA_betar[k] = 0+c1
            KA_betac[k] = 2+c2
            k += 1
            KA_betar[k] = 0+c1
            KA_betac[k] = 0+c3
            k += 1
            KA_betar[k] = 0+c1
            KA_betac[k] = 1+c3
            k += 1
            KA_betar[k] = 0+c1
            KA_betac[k] = 2+c3
            k += 1
            KA_betar[k] = 0+c1
            KA_betac[k] = 0+c4
            k += 1
            KA_betar[k] = 0+c1
            KA_betac[k] = 1+c4
            k += 1
            KA_betar[k] = 0+c1
            KA_betac[k] = 2+c4
            k += 1
            KA_betar[k] = 1+c1
            KA_betac[k] = 0+c1
            k += 1
            KA_betar[k] = 1+c1
            KA_betac[k] = 1+c1
            k += 1
            KA_betar[k] = 1+c1
            KA_betac[k] = 2+c1
            k += 1
            KA_betar[k] = 1+c1
            KA_betac[k] = 0+c2
            k += 1
            KA_betar[k] = 1+c1
            KA_betac[k] = 1+c2
            k += 1
            KA_betar[k] = 1+c1
            KA_betac[k] = 2+c2
            k += 1
            KA_betar[k] = 1+c1
            KA_betac[k] = 0+c3
            k += 1
            KA_betar[k] = 1+c1
            KA_betac[k] = 1+c3
            k += 1
            KA_betar[k] = 1+c1
            KA_betac[k] = 2+c3
            k += 1
            KA_betar[k] = 1+c1
            KA_betac[k] = 0+c4
            k += 1
            KA_betar[k] = 1+c1
            KA_betac[k] = 1+c4
            k += 1
            KA_betar[k] = 1+c1
            KA_betac[k] = 2+c4
            k += 1
            KA_betar[k] = 2+c1
            KA_betac[k] = 0+c1
            k += 1
            KA_betar[k] = 2+c1
            KA_betac[k] = 1+c1
            k += 1
            KA_betar[k] = 2+c1
            KA_betac[k] = 2+c1
            k += 1
            KA_betar[k] = 2+c1
            KA_betac[k] = 0+c2
            k += 1
            KA_betar[k] = 2+c1
            KA_betac[k] = 1+c2
            k += 1
            KA_betar[k] = 2+c1
            KA_betac[k] = 2+c2
            k += 1
            KA_betar[k] = 2+c1
            KA_betac[k] = 0+c3
            k += 1
            KA_betar[k] = 2+c1
            KA_betac[k] = 1+c3
            k += 1
            KA_betar[k] = 2+c1
            KA_betac[k] = 2+c3
            k += 1
            KA_betar[k] = 2+c1
            KA_betac[k] = 0+c4
            k += 1
            KA_betar[k] = 2+c1
            KA_betac[k] = 1+c4
            k += 1
            KA_betar[k] = 2+c1
            KA_betac[k] = 2+c4
            k += 1
            KA_betar[k] = 0+c2
            KA_betac[k] = 0+c1
            k += 1
            KA_betar[k] = 0+c2
            KA_betac[k] = 1+c1
            k += 1
            KA_betar[k] = 0+c2
            KA_betac[k] = 2+c1
            k += 1
            KA_betar[k] = 0+c2
            KA_betac[k] = 0+c2
            k += 1
            KA_betar[k] = 0+c2
            KA_betac[k] = 1+c2
            k += 1
            KA_betar[k] = 0+c2
            KA_betac[k] = 2+c2
            k += 1
            KA_betar[k] = 0+c2
            KA_betac[k] = 0+c3
            k += 1
            KA_betar[k] = 0+c2
            KA_betac[k] = 1+c3
            k += 1
            KA_betar[k] = 0+c2
            KA_betac[k] = 2+c3
            k += 1
            KA_betar[k] = 0+c2
            KA_betac[k] = 0+c4
            k += 1
            KA_betar[k] = 0+c2
            KA_betac[k] = 1+c4
            k += 1
            KA_betar[k] = 0+c2
            KA_betac[k] = 2+c4
            k += 1
            KA_betar[k] = 1+c2
            KA_betac[k] = 0+c1
            k += 1
            KA_betar[k] = 1+c2
            KA_betac[k] = 1+c1
            k += 1
            KA_betar[k] = 1+c2
            KA_betac[k] = 2+c1
            k += 1
            KA_betar[k] = 1+c2
            KA_betac[k] = 0+c2
            k += 1
            KA_betar[k] = 1+c2
            KA_betac[k] = 1+c2
            k += 1
            KA_betar[k] = 1+c2
            KA_betac[k] = 2+c2
            k += 1
            KA_betar[k] = 1+c2
            KA_betac[k] = 0+c3
            k += 1
            KA_betar[k] = 1+c2
            KA_betac[k] = 1+c3
            k += 1
            KA_betar[k] = 1+c2
            KA_betac[k] = 2+c3
            k += 1
            KA_betar[k] = 1+c2
            KA_betac[k] = 0+c4
            k += 1
            KA_betar[k] = 1+c2
            KA_betac[k] = 1+c4
            k += 1
            KA_betar[k] = 1+c2
            KA_betac[k] = 2+c4
            k += 1
            KA_betar[k] = 2+c2
            KA_betac[k] = 0+c1
            k += 1
            KA_betar[k] = 2+c2
            KA_betac[k] = 1+c1
            k += 1
            KA_betar[k] = 2+c2
            KA_betac[k] = 2+c1
            k += 1
            KA_betar[k] = 2+c2
            KA_betac[k] = 0+c2
            k += 1
            KA_betar[k] = 2+c2
            KA_betac[k] = 1+c2
            k += 1
            KA_betar[k] = 2+c2
            KA_betac[k] = 2+c2
            k += 1
            KA_betar[k] = 2+c2
            KA_betac[k] = 0+c3
            k += 1
            KA_betar[k] = 2+c2
            KA_betac[k] = 1+c3
            k += 1
            KA_betar[k] = 2+c2
            KA_betac[k] = 2+c3
            k += 1
            KA_betar[k] = 2+c2
            KA_betac[k] = 0+c4
            k += 1
            KA_betar[k] = 2+c2
            KA_betac[k] = 1+c4
            k += 1
            KA_betar[k] = 2+c2
            KA_betac[k] = 2+c4
            k += 1
            KA_betar[k] = 0+c3
            KA_betac[k] = 0+c1
            k += 1
            KA_betar[k] = 0+c3
            KA_betac[k] = 1+c1
            k += 1
            KA_betar[k] = 0+c3
            KA_betac[k] = 2+c1
            k += 1
            KA_betar[k] = 0+c3
            KA_betac[k] = 0+c2
            k += 1
            KA_betar[k] = 0+c3
            KA_betac[k] = 1+c2
            k += 1
            KA_betar[k] = 0+c3
            KA_betac[k] = 2+c2
            k += 1
            KA_betar[k] = 0+c3
            KA_betac[k] = 0+c3
            k += 1
            KA_betar[k] = 0+c3
            KA_betac[k] = 1+c3
            k += 1
            KA_betar[k] = 0+c3
            KA_betac[k] = 2+c3
            k += 1
            KA_betar[k] = 0+c3
            KA_betac[k] = 0+c4
            k += 1
            KA_betar[k] = 0+c3
            KA_betac[k] = 1+c4
            k += 1
            KA_betar[k] = 0+c3
            KA_betac[k] = 2+c4
            k += 1
            KA_betar[k] = 1+c3
            KA_betac[k] = 0+c1
            k += 1
            KA_betar[k] = 1+c3
            KA_betac[k] = 1+c1
            k += 1
            KA_betar[k] = 1+c3
            KA_betac[k] = 2+c1
            k += 1
            KA_betar[k] = 1+c3
            KA_betac[k] = 0+c2
            k += 1
            KA_betar[k] = 1+c3
            KA_betac[k] = 1+c2
            k += 1
            KA_betar[k] = 1+c3
            KA_betac[k] = 2+c2
            k += 1
            KA_betar[k] = 1+c3
            KA_betac[k] = 0+c3
            k += 1
            KA_betar[k] = 1+c3
            KA_betac[k] = 1+c3
            k += 1
            KA_betar[k] = 1+c3
            KA_betac[k] = 2+c3
            k += 1
            KA_betar[k] = 1+c3
            KA_betac[k] = 0+c4
            k += 1
            KA_betar[k] = 1+c3
            KA_betac[k] = 1+c4
            k += 1
            KA_betar[k] = 1+c3
            KA_betac[k] = 2+c4
            k += 1
            KA_betar[k] = 2+c3
            KA_betac[k] = 0+c1
            k += 1
            KA_betar[k] = 2+c3
            KA_betac[k] = 1+c1
            k += 1
            KA_betar[k] = 2+c3
            KA_betac[k] = 2+c1
            k += 1
            KA_betar[k] = 2+c3
            KA_betac[k] = 0+c2
            k += 1
            KA_betar[k] = 2+c3
            KA_betac[k] = 1+c2
            k += 1
            KA_betar[k] = 2+c3
            KA_betac[k] = 2+c2
            k += 1
            KA_betar[k] = 2+c3
            KA_betac[k] = 0+c3
            k += 1
            KA_betar[k] = 2+c3
            KA_betac[k] = 1+c3
            k += 1
            KA_betar[k] = 2+c3
            KA_betac[k] = 2+c3
            k += 1
            KA_betar[k] = 2+c3
            KA_betac[k] = 0+c4
            k += 1
            KA_betar[k] = 2+c3
            KA_betac[k] = 1+c4
            k += 1
            KA_betar[k] = 2+c3
            KA_betac[k] = 2+c4
            k += 1
            KA_betar[k] = 0+c4
            KA_betac[k] = 0+c1
            k += 1
            KA_betar[k] = 0+c4
            KA_betac[k] = 1+c1
            k += 1
            KA_betar[k] = 0+c4
            KA_betac[k] = 2+c1
            k += 1
            KA_betar[k] = 0+c4
            KA_betac[k] = 0+c2
            k += 1
            KA_betar[k] = 0+c4
            KA_betac[k] = 1+c2
            k += 1
            KA_betar[k] = 0+c4
            KA_betac[k] = 2+c2
            k += 1
            KA_betar[k] = 0+c4
            KA_betac[k] = 0+c3
            k += 1
            KA_betar[k] = 0+c4
            KA_betac[k] = 1+c3
            k += 1
            KA_betar[k] = 0+c4
            KA_betac[k] = 2+c3
            k += 1
            KA_betar[k] = 0+c4
            KA_betac[k] = 0+c4
            k += 1
            KA_betar[k] = 0+c4
            KA_betac[k] = 1+c4
            k += 1
            KA_betar[k] = 0+c4
            KA_betac[k] = 2+c4
            k += 1
            KA_betar[k] = 1+c4
            KA_betac[k] = 0+c1
            k += 1
            KA_betar[k] = 1+c4
            KA_betac[k] = 1+c1
            k += 1
            KA_betar[k] = 1+c4
            KA_betac[k] = 2+c1
            k += 1
            KA_betar[k] = 1+c4
            KA_betac[k] = 0+c2
            k += 1
            KA_betar[k] = 1+c4
            KA_betac[k] = 1+c2
            k += 1
            KA_betar[k] = 1+c4
            KA_betac[k] = 2+c2
            k += 1
            KA_betar[k] = 1+c4
            KA_betac[k] = 0+c3
            k += 1
            KA_betar[k] = 1+c4
            KA_betac[k] = 1+c3
            k += 1
            KA_betar[k] = 1+c4
            KA_betac[k] = 2+c3
            k += 1
            KA_betar[k] = 1+c4
            KA_betac[k] = 0+c4
            k += 1
            KA_betar[k] = 1+c4
            KA_betac[k] = 1+c4
            k += 1
            KA_betar[k] = 1+c4
            KA_betac[k] = 2+c4
            k += 1
            KA_betar[k] = 2+c4
            KA_betac[k] = 0+c1
            k += 1
            KA_betar[k] = 2+c4
            KA_betac[k] = 1+c1
            k += 1
            KA_betar[k] = 2+c4
            KA_betac[k] = 2+c1
            k += 1
            KA_betar[k] = 2+c4
            KA_betac[k] = 0+c2
            k += 1
            KA_betar[k] = 2+c4
            KA_betac[k] = 1+c2
            k += 1
            KA_betar[k] = 2+c4
            KA_betac[k] = 2+c2
            k += 1
            KA_betar[k] = 2+c4
            KA_betac[k] = 0+c3
            k += 1
            KA_betar[k] = 2+c4
            KA_betac[k] = 1+c3
            k += 1
            KA_betar[k] = 2+c4
            KA_betac[k] = 2+c3
            k += 1
            KA_betar[k] = 2+c4
            KA_betac[k] = 0+c4
            k += 1
            KA_betar[k] = 2+c4
            KA_betac[k] = 1+c4
            k += 1
            KA_betar[k] = 2+c4
            KA_betac[k] = 2+c4

            # NOTE full integration for KG with two-point Gauss-Legendre quadrature
            wij = 1.
            points[0] = -0.5773502691896257645092
            points[1] = +0.5773502691896257645092

            for i in range(2):
                xi = points[i]
                for j in range(2):
                    eta = points[j]

                    detJ = -0.125*eta*x1*y2 + 0.125*eta*x1*y3 + 0.125*eta*x2*y1 - 0.125*eta*x2*y4 - 0.125*eta*x3*y1 + 0.125*eta*x3*y4 + 0.125*eta*x4*y2 - 0.125*eta*x4*y3 - 0.125*x1*xi*y3 + 0.125*x1*xi*y4 + 0.125*x1*y2 - 0.125*x1*y4 + 0.125*x2*xi*y3 - 0.125*x2*xi*y4 - 0.125*x2*y1 + 0.125*x2*y3 + 0.125*x3*xi*y1 - 0.125*x3*xi*y2 - 0.125*x3*y2 + 0.125*x3*y4 - 0.125*x4*xi*y1 + 0.125*x4*xi*y2 + 0.125*x4*y1 - 0.125*x4*y3

                    j11 = 2.0*(-xi*y1 + xi*y2 - xi*y3 + xi*y4 + y1 + y2 - y3 - y4)/(eta*x1*y2 - eta*x1*y3 - eta*x2*y1 + eta*x2*y4 + eta*x3*y1 - eta*x3*y4 - eta*x4*y2 + eta*x4*y3 + x1*xi*y3 - x1*xi*y4 - x1*y2 + x1*y4 - x2*xi*y3 + x2*xi*y4 + x2*y1 - x2*y3 - x3*xi*y1 + x3*xi*y2 + x3*y2 - x3*y4 + x4*xi*y1 - x4*xi*y2 - x4*y1 + x4*y3)
                    j12 = 2.0*(eta*y1 - eta*y2 + eta*y3 - eta*y4 - y1 + y2 + y3 - y4)/(eta*x1*y2 - eta*x1*y3 - eta*x2*y1 + eta*x2*y4 + eta*x3*y1 - eta*x3*y4 - eta*x4*y2 + eta*x4*y3 + x1*xi*y3 - x1*xi*y4 - x1*y2 + x1*y4 - x2*xi*y3 + x2*xi*y4 + x2*y1 - x2*y3 - x3*xi*y1 + x3*xi*y2 + x3*y2 - x3*y4 + x4*xi*y1 - x4*xi*y2 - x4*y1 + x4*y3)
                    j21 = 2.0*(x1*xi - x1 - x2*xi - x2 + x3*xi + x3 - x4*xi + x4)/(eta*x1*y2 - eta*x1*y3 - eta*x2*y1 + eta*x2*y4 + eta*x3*y1 - eta*x3*y4 - eta*x4*y2 + eta*x4*y3 + x1*xi*y3 - x1*xi*y4 - x1*y2 + x1*y4 - x2*xi*y3 + x2*xi*y4 + x2*y1 - x2*y3 - x3*xi*y1 + x3*xi*y2 + x3*y2 - x3*y4 + x4*xi*y1 - x4*xi*y2 - x4*y1 + x4*y3)
                    j22 = 2.0*(-eta*x1 + eta*x2 - eta*x3 + eta*x4 + x1 - x2 - x3 + x4)/(eta*x1*y2 - eta*x1*y3 - eta*x2*y1 + eta*x2*y4 + eta*x3*y1 - eta*x3*y4 - eta*x4*y2 + eta*x4*y3 + x1*xi*y3 - x1*xi*y4 - x1*y2 + x1*y4 - x2*xi*y3 + x2*xi*y4 + x2*y1 - x2*y3 - x3*xi*y1 + x3*xi*y2 + x3*y2 - x3*y4 + x4*xi*y1 - x4*xi*y2 - x4*y1 + x4*y3)

                    S1 = 0.25*eta*xi - 0.25*eta - 0.25*xi + 0.25
                    S2 = -0.25*eta*xi - 0.25*eta + 0.25*xi + 0.25
                    S3 = 0.25*eta*xi + 0.25*eta + 0.25*xi + 0.25
                    S4 = -0.25*eta*xi + 0.25*eta - 0.25*xi + 0.25

                    S1_x = 0.25*j11*(eta - 1) + 0.25*j12*(xi - 1)
                    S2_x = -0.25*eta*j11 + 0.25*j11 - 0.25*j12*xi - 0.25*j12
                    S3_x = 0.25*j11*(eta + 1) + 0.25*j12*(xi + 1)
                    S4_x = -0.25*eta*j11 - 0.25*j11 - 0.25*j12*xi + 0.25*j12

                    S1_y = 0.25*j21*(eta - 1) + 0.25*j22*(xi - 1)
                    S2_y = -0.25*eta*j21 + 0.25*j21 - 0.25*j22*xi - 0.25*j22
                    S3_y = 0.25*j21*(eta + 1) + 0.25*j22*(xi + 1)
                    S4_y = -0.25*eta*j21 - 0.25*j21 - 0.25*j22*xi + 0.25*j22

                    k = self.init_k_KA_beta
                    KA_betav[k] += -S1*detJ*r13**2*wij*(S1_x*r11 + S1_y*r12)
                    k += 1
                    KA_betav[k] += -S1*detJ*r13*r23*wij*(S1_x*r11 + S1_y*r12)
                    k += 1
                    KA_betav[k] += -S1*detJ*r13*r33*wij*(S1_x*r11 + S1_y*r12)
                    k += 1
                    KA_betav[k] += -S1*detJ*r13**2*wij*(S2_x*r11 + S2_y*r12)
                    k += 1
                    KA_betav[k] += -S1*detJ*r13*r23*wij*(S2_x*r11 + S2_y*r12)
                    k += 1
                    KA_betav[k] += -S1*detJ*r13*r33*wij*(S2_x*r11 + S2_y*r12)
                    k += 1
                    KA_betav[k] += -S1*detJ*r13**2*wij*(S3_x*r11 + S3_y*r12)
                    k += 1
                    KA_betav[k] += -S1*detJ*r13*r23*wij*(S3_x*r11 + S3_y*r12)
                    k += 1
                    KA_betav[k] += -S1*detJ*r13*r33*wij*(S3_x*r11 + S3_y*r12)
                    k += 1
                    KA_betav[k] += -S1*detJ*r13**2*wij*(S4_x*r11 + S4_y*r12)
                    k += 1
                    KA_betav[k] += -S1*detJ*r13*r23*wij*(S4_x*r11 + S4_y*r12)
                    k += 1
                    KA_betav[k] += -S1*detJ*r13*r33*wij*(S4_x*r11 + S4_y*r12)
                    k += 1
                    KA_betav[k] += -S1*detJ*r13*r23*wij*(S1_x*r11 + S1_y*r12)
                    k += 1
                    KA_betav[k] += -S1*detJ*r23**2*wij*(S1_x*r11 + S1_y*r12)
                    k += 1
                    KA_betav[k] += -S1*detJ*r23*r33*wij*(S1_x*r11 + S1_y*r12)
                    k += 1
                    KA_betav[k] += -S1*detJ*r13*r23*wij*(S2_x*r11 + S2_y*r12)
                    k += 1
                    KA_betav[k] += -S1*detJ*r23**2*wij*(S2_x*r11 + S2_y*r12)
                    k += 1
                    KA_betav[k] += -S1*detJ*r23*r33*wij*(S2_x*r11 + S2_y*r12)
                    k += 1
                    KA_betav[k] += -S1*detJ*r13*r23*wij*(S3_x*r11 + S3_y*r12)
                    k += 1
                    KA_betav[k] += -S1*detJ*r23**2*wij*(S3_x*r11 + S3_y*r12)
                    k += 1
                    KA_betav[k] += -S1*detJ*r23*r33*wij*(S3_x*r11 + S3_y*r12)
                    k += 1
                    KA_betav[k] += -S1*detJ*r13*r23*wij*(S4_x*r11 + S4_y*r12)
                    k += 1
                    KA_betav[k] += -S1*detJ*r23**2*wij*(S4_x*r11 + S4_y*r12)
                    k += 1
                    KA_betav[k] += -S1*detJ*r23*r33*wij*(S4_x*r11 + S4_y*r12)
                    k += 1
                    KA_betav[k] += -S1*detJ*r13*r33*wij*(S1_x*r11 + S1_y*r12)
                    k += 1
                    KA_betav[k] += -S1*detJ*r23*r33*wij*(S1_x*r11 + S1_y*r12)
                    k += 1
                    KA_betav[k] += -S1*detJ*r33**2*wij*(S1_x*r11 + S1_y*r12)
                    k += 1
                    KA_betav[k] += -S1*detJ*r13*r33*wij*(S2_x*r11 + S2_y*r12)
                    k += 1
                    KA_betav[k] += -S1*detJ*r23*r33*wij*(S2_x*r11 + S2_y*r12)
                    k += 1
                    KA_betav[k] += -S1*detJ*r33**2*wij*(S2_x*r11 + S2_y*r12)
                    k += 1
                    KA_betav[k] += -S1*detJ*r13*r33*wij*(S3_x*r11 + S3_y*r12)
                    k += 1
                    KA_betav[k] += -S1*detJ*r23*r33*wij*(S3_x*r11 + S3_y*r12)
                    k += 1
                    KA_betav[k] += -S1*detJ*r33**2*wij*(S3_x*r11 + S3_y*r12)
                    k += 1
                    KA_betav[k] += -S1*detJ*r13*r33*wij*(S4_x*r11 + S4_y*r12)
                    k += 1
                    KA_betav[k] += -S1*detJ*r23*r33*wij*(S4_x*r11 + S4_y*r12)
                    k += 1
                    KA_betav[k] += -S1*detJ*r33**2*wij*(S4_x*r11 + S4_y*r12)
                    k += 1
                    KA_betav[k] += -S2*detJ*r13**2*wij*(S1_x*r11 + S1_y*r12)
                    k += 1
                    KA_betav[k] += -S2*detJ*r13*r23*wij*(S1_x*r11 + S1_y*r12)
                    k += 1
                    KA_betav[k] += -S2*detJ*r13*r33*wij*(S1_x*r11 + S1_y*r12)
                    k += 1
                    KA_betav[k] += -S2*detJ*r13**2*wij*(S2_x*r11 + S2_y*r12)
                    k += 1
                    KA_betav[k] += -S2*detJ*r13*r23*wij*(S2_x*r11 + S2_y*r12)
                    k += 1
                    KA_betav[k] += -S2*detJ*r13*r33*wij*(S2_x*r11 + S2_y*r12)
                    k += 1
                    KA_betav[k] += -S2*detJ*r13**2*wij*(S3_x*r11 + S3_y*r12)
                    k += 1
                    KA_betav[k] += -S2*detJ*r13*r23*wij*(S3_x*r11 + S3_y*r12)
                    k += 1
                    KA_betav[k] += -S2*detJ*r13*r33*wij*(S3_x*r11 + S3_y*r12)
                    k += 1
                    KA_betav[k] += -S2*detJ*r13**2*wij*(S4_x*r11 + S4_y*r12)
                    k += 1
                    KA_betav[k] += -S2*detJ*r13*r23*wij*(S4_x*r11 + S4_y*r12)
                    k += 1
                    KA_betav[k] += -S2*detJ*r13*r33*wij*(S4_x*r11 + S4_y*r12)
                    k += 1
                    KA_betav[k] += -S2*detJ*r13*r23*wij*(S1_x*r11 + S1_y*r12)
                    k += 1
                    KA_betav[k] += -S2*detJ*r23**2*wij*(S1_x*r11 + S1_y*r12)
                    k += 1
                    KA_betav[k] += -S2*detJ*r23*r33*wij*(S1_x*r11 + S1_y*r12)
                    k += 1
                    KA_betav[k] += -S2*detJ*r13*r23*wij*(S2_x*r11 + S2_y*r12)
                    k += 1
                    KA_betav[k] += -S2*detJ*r23**2*wij*(S2_x*r11 + S2_y*r12)
                    k += 1
                    KA_betav[k] += -S2*detJ*r23*r33*wij*(S2_x*r11 + S2_y*r12)
                    k += 1
                    KA_betav[k] += -S2*detJ*r13*r23*wij*(S3_x*r11 + S3_y*r12)
                    k += 1
                    KA_betav[k] += -S2*detJ*r23**2*wij*(S3_x*r11 + S3_y*r12)
                    k += 1
                    KA_betav[k] += -S2*detJ*r23*r33*wij*(S3_x*r11 + S3_y*r12)
                    k += 1
                    KA_betav[k] += -S2*detJ*r13*r23*wij*(S4_x*r11 + S4_y*r12)
                    k += 1
                    KA_betav[k] += -S2*detJ*r23**2*wij*(S4_x*r11 + S4_y*r12)
                    k += 1
                    KA_betav[k] += -S2*detJ*r23*r33*wij*(S4_x*r11 + S4_y*r12)
                    k += 1
                    KA_betav[k] += -S2*detJ*r13*r33*wij*(S1_x*r11 + S1_y*r12)
                    k += 1
                    KA_betav[k] += -S2*detJ*r23*r33*wij*(S1_x*r11 + S1_y*r12)
                    k += 1
                    KA_betav[k] += -S2*detJ*r33**2*wij*(S1_x*r11 + S1_y*r12)
                    k += 1
                    KA_betav[k] += -S2*detJ*r13*r33*wij*(S2_x*r11 + S2_y*r12)
                    k += 1
                    KA_betav[k] += -S2*detJ*r23*r33*wij*(S2_x*r11 + S2_y*r12)
                    k += 1
                    KA_betav[k] += -S2*detJ*r33**2*wij*(S2_x*r11 + S2_y*r12)
                    k += 1
                    KA_betav[k] += -S2*detJ*r13*r33*wij*(S3_x*r11 + S3_y*r12)
                    k += 1
                    KA_betav[k] += -S2*detJ*r23*r33*wij*(S3_x*r11 + S3_y*r12)
                    k += 1
                    KA_betav[k] += -S2*detJ*r33**2*wij*(S3_x*r11 + S3_y*r12)
                    k += 1
                    KA_betav[k] += -S2*detJ*r13*r33*wij*(S4_x*r11 + S4_y*r12)
                    k += 1
                    KA_betav[k] += -S2*detJ*r23*r33*wij*(S4_x*r11 + S4_y*r12)
                    k += 1
                    KA_betav[k] += -S2*detJ*r33**2*wij*(S4_x*r11 + S4_y*r12)
                    k += 1
                    KA_betav[k] += -S3*detJ*r13**2*wij*(S1_x*r11 + S1_y*r12)
                    k += 1
                    KA_betav[k] += -S3*detJ*r13*r23*wij*(S1_x*r11 + S1_y*r12)
                    k += 1
                    KA_betav[k] += -S3*detJ*r13*r33*wij*(S1_x*r11 + S1_y*r12)
                    k += 1
                    KA_betav[k] += -S3*detJ*r13**2*wij*(S2_x*r11 + S2_y*r12)
                    k += 1
                    KA_betav[k] += -S3*detJ*r13*r23*wij*(S2_x*r11 + S2_y*r12)
                    k += 1
                    KA_betav[k] += -S3*detJ*r13*r33*wij*(S2_x*r11 + S2_y*r12)
                    k += 1
                    KA_betav[k] += -S3*detJ*r13**2*wij*(S3_x*r11 + S3_y*r12)
                    k += 1
                    KA_betav[k] += -S3*detJ*r13*r23*wij*(S3_x*r11 + S3_y*r12)
                    k += 1
                    KA_betav[k] += -S3*detJ*r13*r33*wij*(S3_x*r11 + S3_y*r12)
                    k += 1
                    KA_betav[k] += -S3*detJ*r13**2*wij*(S4_x*r11 + S4_y*r12)
                    k += 1
                    KA_betav[k] += -S3*detJ*r13*r23*wij*(S4_x*r11 + S4_y*r12)
                    k += 1
                    KA_betav[k] += -S3*detJ*r13*r33*wij*(S4_x*r11 + S4_y*r12)
                    k += 1
                    KA_betav[k] += -S3*detJ*r13*r23*wij*(S1_x*r11 + S1_y*r12)
                    k += 1
                    KA_betav[k] += -S3*detJ*r23**2*wij*(S1_x*r11 + S1_y*r12)
                    k += 1
                    KA_betav[k] += -S3*detJ*r23*r33*wij*(S1_x*r11 + S1_y*r12)
                    k += 1
                    KA_betav[k] += -S3*detJ*r13*r23*wij*(S2_x*r11 + S2_y*r12)
                    k += 1
                    KA_betav[k] += -S3*detJ*r23**2*wij*(S2_x*r11 + S2_y*r12)
                    k += 1
                    KA_betav[k] += -S3*detJ*r23*r33*wij*(S2_x*r11 + S2_y*r12)
                    k += 1
                    KA_betav[k] += -S3*detJ*r13*r23*wij*(S3_x*r11 + S3_y*r12)
                    k += 1
                    KA_betav[k] += -S3*detJ*r23**2*wij*(S3_x*r11 + S3_y*r12)
                    k += 1
                    KA_betav[k] += -S3*detJ*r23*r33*wij*(S3_x*r11 + S3_y*r12)
                    k += 1
                    KA_betav[k] += -S3*detJ*r13*r23*wij*(S4_x*r11 + S4_y*r12)
                    k += 1
                    KA_betav[k] += -S3*detJ*r23**2*wij*(S4_x*r11 + S4_y*r12)
                    k += 1
                    KA_betav[k] += -S3*detJ*r23*r33*wij*(S4_x*r11 + S4_y*r12)
                    k += 1
                    KA_betav[k] += -S3*detJ*r13*r33*wij*(S1_x*r11 + S1_y*r12)
                    k += 1
                    KA_betav[k] += -S3*detJ*r23*r33*wij*(S1_x*r11 + S1_y*r12)
                    k += 1
                    KA_betav[k] += -S3*detJ*r33**2*wij*(S1_x*r11 + S1_y*r12)
                    k += 1
                    KA_betav[k] += -S3*detJ*r13*r33*wij*(S2_x*r11 + S2_y*r12)
                    k += 1
                    KA_betav[k] += -S3*detJ*r23*r33*wij*(S2_x*r11 + S2_y*r12)
                    k += 1
                    KA_betav[k] += -S3*detJ*r33**2*wij*(S2_x*r11 + S2_y*r12)
                    k += 1
                    KA_betav[k] += -S3*detJ*r13*r33*wij*(S3_x*r11 + S3_y*r12)
                    k += 1
                    KA_betav[k] += -S3*detJ*r23*r33*wij*(S3_x*r11 + S3_y*r12)
                    k += 1
                    KA_betav[k] += -S3*detJ*r33**2*wij*(S3_x*r11 + S3_y*r12)
                    k += 1
                    KA_betav[k] += -S3*detJ*r13*r33*wij*(S4_x*r11 + S4_y*r12)
                    k += 1
                    KA_betav[k] += -S3*detJ*r23*r33*wij*(S4_x*r11 + S4_y*r12)
                    k += 1
                    KA_betav[k] += -S3*detJ*r33**2*wij*(S4_x*r11 + S4_y*r12)
                    k += 1
                    KA_betav[k] += -S4*detJ*r13**2*wij*(S1_x*r11 + S1_y*r12)
                    k += 1
                    KA_betav[k] += -S4*detJ*r13*r23*wij*(S1_x*r11 + S1_y*r12)
                    k += 1
                    KA_betav[k] += -S4*detJ*r13*r33*wij*(S1_x*r11 + S1_y*r12)
                    k += 1
                    KA_betav[k] += -S4*detJ*r13**2*wij*(S2_x*r11 + S2_y*r12)
                    k += 1
                    KA_betav[k] += -S4*detJ*r13*r23*wij*(S2_x*r11 + S2_y*r12)
                    k += 1
                    KA_betav[k] += -S4*detJ*r13*r33*wij*(S2_x*r11 + S2_y*r12)
                    k += 1
                    KA_betav[k] += -S4*detJ*r13**2*wij*(S3_x*r11 + S3_y*r12)
                    k += 1
                    KA_betav[k] += -S4*detJ*r13*r23*wij*(S3_x*r11 + S3_y*r12)
                    k += 1
                    KA_betav[k] += -S4*detJ*r13*r33*wij*(S3_x*r11 + S3_y*r12)
                    k += 1
                    KA_betav[k] += -S4*detJ*r13**2*wij*(S4_x*r11 + S4_y*r12)
                    k += 1
                    KA_betav[k] += -S4*detJ*r13*r23*wij*(S4_x*r11 + S4_y*r12)
                    k += 1
                    KA_betav[k] += -S4*detJ*r13*r33*wij*(S4_x*r11 + S4_y*r12)
                    k += 1
                    KA_betav[k] += -S4*detJ*r13*r23*wij*(S1_x*r11 + S1_y*r12)
                    k += 1
                    KA_betav[k] += -S4*detJ*r23**2*wij*(S1_x*r11 + S1_y*r12)
                    k += 1
                    KA_betav[k] += -S4*detJ*r23*r33*wij*(S1_x*r11 + S1_y*r12)
                    k += 1
                    KA_betav[k] += -S4*detJ*r13*r23*wij*(S2_x*r11 + S2_y*r12)
                    k += 1
                    KA_betav[k] += -S4*detJ*r23**2*wij*(S2_x*r11 + S2_y*r12)
                    k += 1
                    KA_betav[k] += -S4*detJ*r23*r33*wij*(S2_x*r11 + S2_y*r12)
                    k += 1
                    KA_betav[k] += -S4*detJ*r13*r23*wij*(S3_x*r11 + S3_y*r12)
                    k += 1
                    KA_betav[k] += -S4*detJ*r23**2*wij*(S3_x*r11 + S3_y*r12)
                    k += 1
                    KA_betav[k] += -S4*detJ*r23*r33*wij*(S3_x*r11 + S3_y*r12)
                    k += 1
                    KA_betav[k] += -S4*detJ*r13*r23*wij*(S4_x*r11 + S4_y*r12)
                    k += 1
                    KA_betav[k] += -S4*detJ*r23**2*wij*(S4_x*r11 + S4_y*r12)
                    k += 1
                    KA_betav[k] += -S4*detJ*r23*r33*wij*(S4_x*r11 + S4_y*r12)
                    k += 1
                    KA_betav[k] += -S4*detJ*r13*r33*wij*(S1_x*r11 + S1_y*r12)
                    k += 1
                    KA_betav[k] += -S4*detJ*r23*r33*wij*(S1_x*r11 + S1_y*r12)
                    k += 1
                    KA_betav[k] += -S4*detJ*r33**2*wij*(S1_x*r11 + S1_y*r12)
                    k += 1
                    KA_betav[k] += -S4*detJ*r13*r33*wij*(S2_x*r11 + S2_y*r12)
                    k += 1
                    KA_betav[k] += -S4*detJ*r23*r33*wij*(S2_x*r11 + S2_y*r12)
                    k += 1
                    KA_betav[k] += -S4*detJ*r33**2*wij*(S2_x*r11 + S2_y*r12)
                    k += 1
                    KA_betav[k] += -S4*detJ*r13*r33*wij*(S3_x*r11 + S3_y*r12)
                    k += 1
                    KA_betav[k] += -S4*detJ*r23*r33*wij*(S3_x*r11 + S3_y*r12)
                    k += 1
                    KA_betav[k] += -S4*detJ*r33**2*wij*(S3_x*r11 + S3_y*r12)
                    k += 1
                    KA_betav[k] += -S4*detJ*r13*r33*wij*(S4_x*r11 + S4_y*r12)
                    k += 1
                    KA_betav[k] += -S4*detJ*r23*r33*wij*(S4_x*r11 + S4_y*r12)
                    k += 1
                    KA_betav[k] += -S4*detJ*r33**2*wij*(S4_x*r11 + S4_y*r12)


    cpdef void update_KA_gamma(Quad4R self,
                        long [::1] KA_gammar,
                        long [::1] KA_gammac,
                        double [::1] KA_gammav,
                        ):
        r"""Update sparse vectors for piston-theory aerodynamic matrix `\boldsymbol{K}_{A_\gamma}`

        The matrix is integrated over the element with 2 by 2 Gauss-Legendre
        quadrature, from the coordinates ``probe.xe`` and the rotation matrix,
        and its values are added to ``KA_gammav`` at the positions
        ``init_k_KA_gamma`` to ``init_k_KA_gamma + KA_GAMMA_SPARSE_SIZE - 1``;
        the global row and column indices are always written to
        ``KA_gammar`` and ``KA_gammac`` at the same positions. Only the
        translational degrees-of-freedom appear, through their components
        along the element normal. No aerodynamic coefficient enters the
        values, which depend only on the geometry, so the caller multiplies
        the assembled matrix by it. Before this method is called,
        :meth:`.update_rotation_matrix` and :meth:`.update_probe_xe` must have
        been used.

        Parameters
        ----------
        KA_gammar : np.array
            Array to store row positions of sparse values
            1-D array of integers (``INT``) of size at least
            ``init_k_KA_gamma + KA_GAMMA_SPARSE_SIZE``.
        KA_gammac : np.array
            Array to store column positions of sparse values
            Same type and size as ``KA_gammar``.
        KA_gammav : np.array
            Array to store sparse values
            1-D array of ``DOUBLE``, same size as ``KA_gammar``. The values
            are added to the existing ones, with dimension of length squared.

        Notes
        -----
        With `w_n` the displacement along the element normal, as in
        :meth:`.update_KA_beta`, the computed matrix is

        .. math::

            \boldsymbol{K}_{A_\gamma} = \int_A \boldsymbol{S}^\top \boldsymbol{S} dA

        acting on the nodal values of `w_n`. With the piston-theory
        coefficient `\gamma` given by the caller, `\gamma \boldsymbol{K}_{A_\gamma}
        \boldsymbol{u}` is the load of the term `\gamma w_n` of the pressure law, which
        enters the stiffness of the equations of motion as `-\gamma
        \boldsymbol{K}_{A_\gamma}`, see the piston theory in the notes of
        :meth:`.update_KA_beta`.

        """
        cdef int c1, c2, c3, c4, i, j, k
        cdef double r13, r23, r33
        cdef double x1, x2, x3, x4
        cdef double y1, y2, y3, y4
        cdef double S1, S2, S3, S4
        cdef double xi, eta, wij, detJ
        cdef double points[2]

        with nogil:

            # NOTE ignoring z in local coordinates
            x1 = self.probe.xe[0]
            y1 = self.probe.xe[1]
            # z1 = self.probe.xe[2]
            x2 = self.probe.xe[3]
            y2 = self.probe.xe[4]
            # z2 = self.probe.xe[5]
            x3 = self.probe.xe[6]
            y3 = self.probe.xe[7]
            # z3 = self.probe.xe[8]
            x4 = self.probe.xe[9]
            y4 = self.probe.xe[10]
            # z4 = self.probe.xe[11]

            # local to global transformation
            r13 = self.r13
            r23 = self.r23
            r33 = self.r33

            # positions the global matrices
            c1 = self.c1
            c2 = self.c2
            c3 = self.c3
            c4 = self.c4

            k = self.init_k_KA_gamma
            KA_gammar[k] = 0+c1
            KA_gammac[k] = 0+c1
            k += 1
            KA_gammar[k] = 0+c1
            KA_gammac[k] = 1+c1
            k += 1
            KA_gammar[k] = 0+c1
            KA_gammac[k] = 2+c1
            k += 1
            KA_gammar[k] = 0+c1
            KA_gammac[k] = 0+c2
            k += 1
            KA_gammar[k] = 0+c1
            KA_gammac[k] = 1+c2
            k += 1
            KA_gammar[k] = 0+c1
            KA_gammac[k] = 2+c2
            k += 1
            KA_gammar[k] = 0+c1
            KA_gammac[k] = 0+c3
            k += 1
            KA_gammar[k] = 0+c1
            KA_gammac[k] = 1+c3
            k += 1
            KA_gammar[k] = 0+c1
            KA_gammac[k] = 2+c3
            k += 1
            KA_gammar[k] = 0+c1
            KA_gammac[k] = 0+c4
            k += 1
            KA_gammar[k] = 0+c1
            KA_gammac[k] = 1+c4
            k += 1
            KA_gammar[k] = 0+c1
            KA_gammac[k] = 2+c4
            k += 1
            KA_gammar[k] = 1+c1
            KA_gammac[k] = 0+c1
            k += 1
            KA_gammar[k] = 1+c1
            KA_gammac[k] = 1+c1
            k += 1
            KA_gammar[k] = 1+c1
            KA_gammac[k] = 2+c1
            k += 1
            KA_gammar[k] = 1+c1
            KA_gammac[k] = 0+c2
            k += 1
            KA_gammar[k] = 1+c1
            KA_gammac[k] = 1+c2
            k += 1
            KA_gammar[k] = 1+c1
            KA_gammac[k] = 2+c2
            k += 1
            KA_gammar[k] = 1+c1
            KA_gammac[k] = 0+c3
            k += 1
            KA_gammar[k] = 1+c1
            KA_gammac[k] = 1+c3
            k += 1
            KA_gammar[k] = 1+c1
            KA_gammac[k] = 2+c3
            k += 1
            KA_gammar[k] = 1+c1
            KA_gammac[k] = 0+c4
            k += 1
            KA_gammar[k] = 1+c1
            KA_gammac[k] = 1+c4
            k += 1
            KA_gammar[k] = 1+c1
            KA_gammac[k] = 2+c4
            k += 1
            KA_gammar[k] = 2+c1
            KA_gammac[k] = 0+c1
            k += 1
            KA_gammar[k] = 2+c1
            KA_gammac[k] = 1+c1
            k += 1
            KA_gammar[k] = 2+c1
            KA_gammac[k] = 2+c1
            k += 1
            KA_gammar[k] = 2+c1
            KA_gammac[k] = 0+c2
            k += 1
            KA_gammar[k] = 2+c1
            KA_gammac[k] = 1+c2
            k += 1
            KA_gammar[k] = 2+c1
            KA_gammac[k] = 2+c2
            k += 1
            KA_gammar[k] = 2+c1
            KA_gammac[k] = 0+c3
            k += 1
            KA_gammar[k] = 2+c1
            KA_gammac[k] = 1+c3
            k += 1
            KA_gammar[k] = 2+c1
            KA_gammac[k] = 2+c3
            k += 1
            KA_gammar[k] = 2+c1
            KA_gammac[k] = 0+c4
            k += 1
            KA_gammar[k] = 2+c1
            KA_gammac[k] = 1+c4
            k += 1
            KA_gammar[k] = 2+c1
            KA_gammac[k] = 2+c4
            k += 1
            KA_gammar[k] = 0+c2
            KA_gammac[k] = 0+c1
            k += 1
            KA_gammar[k] = 0+c2
            KA_gammac[k] = 1+c1
            k += 1
            KA_gammar[k] = 0+c2
            KA_gammac[k] = 2+c1
            k += 1
            KA_gammar[k] = 0+c2
            KA_gammac[k] = 0+c2
            k += 1
            KA_gammar[k] = 0+c2
            KA_gammac[k] = 1+c2
            k += 1
            KA_gammar[k] = 0+c2
            KA_gammac[k] = 2+c2
            k += 1
            KA_gammar[k] = 0+c2
            KA_gammac[k] = 0+c3
            k += 1
            KA_gammar[k] = 0+c2
            KA_gammac[k] = 1+c3
            k += 1
            KA_gammar[k] = 0+c2
            KA_gammac[k] = 2+c3
            k += 1
            KA_gammar[k] = 0+c2
            KA_gammac[k] = 0+c4
            k += 1
            KA_gammar[k] = 0+c2
            KA_gammac[k] = 1+c4
            k += 1
            KA_gammar[k] = 0+c2
            KA_gammac[k] = 2+c4
            k += 1
            KA_gammar[k] = 1+c2
            KA_gammac[k] = 0+c1
            k += 1
            KA_gammar[k] = 1+c2
            KA_gammac[k] = 1+c1
            k += 1
            KA_gammar[k] = 1+c2
            KA_gammac[k] = 2+c1
            k += 1
            KA_gammar[k] = 1+c2
            KA_gammac[k] = 0+c2
            k += 1
            KA_gammar[k] = 1+c2
            KA_gammac[k] = 1+c2
            k += 1
            KA_gammar[k] = 1+c2
            KA_gammac[k] = 2+c2
            k += 1
            KA_gammar[k] = 1+c2
            KA_gammac[k] = 0+c3
            k += 1
            KA_gammar[k] = 1+c2
            KA_gammac[k] = 1+c3
            k += 1
            KA_gammar[k] = 1+c2
            KA_gammac[k] = 2+c3
            k += 1
            KA_gammar[k] = 1+c2
            KA_gammac[k] = 0+c4
            k += 1
            KA_gammar[k] = 1+c2
            KA_gammac[k] = 1+c4
            k += 1
            KA_gammar[k] = 1+c2
            KA_gammac[k] = 2+c4
            k += 1
            KA_gammar[k] = 2+c2
            KA_gammac[k] = 0+c1
            k += 1
            KA_gammar[k] = 2+c2
            KA_gammac[k] = 1+c1
            k += 1
            KA_gammar[k] = 2+c2
            KA_gammac[k] = 2+c1
            k += 1
            KA_gammar[k] = 2+c2
            KA_gammac[k] = 0+c2
            k += 1
            KA_gammar[k] = 2+c2
            KA_gammac[k] = 1+c2
            k += 1
            KA_gammar[k] = 2+c2
            KA_gammac[k] = 2+c2
            k += 1
            KA_gammar[k] = 2+c2
            KA_gammac[k] = 0+c3
            k += 1
            KA_gammar[k] = 2+c2
            KA_gammac[k] = 1+c3
            k += 1
            KA_gammar[k] = 2+c2
            KA_gammac[k] = 2+c3
            k += 1
            KA_gammar[k] = 2+c2
            KA_gammac[k] = 0+c4
            k += 1
            KA_gammar[k] = 2+c2
            KA_gammac[k] = 1+c4
            k += 1
            KA_gammar[k] = 2+c2
            KA_gammac[k] = 2+c4
            k += 1
            KA_gammar[k] = 0+c3
            KA_gammac[k] = 0+c1
            k += 1
            KA_gammar[k] = 0+c3
            KA_gammac[k] = 1+c1
            k += 1
            KA_gammar[k] = 0+c3
            KA_gammac[k] = 2+c1
            k += 1
            KA_gammar[k] = 0+c3
            KA_gammac[k] = 0+c2
            k += 1
            KA_gammar[k] = 0+c3
            KA_gammac[k] = 1+c2
            k += 1
            KA_gammar[k] = 0+c3
            KA_gammac[k] = 2+c2
            k += 1
            KA_gammar[k] = 0+c3
            KA_gammac[k] = 0+c3
            k += 1
            KA_gammar[k] = 0+c3
            KA_gammac[k] = 1+c3
            k += 1
            KA_gammar[k] = 0+c3
            KA_gammac[k] = 2+c3
            k += 1
            KA_gammar[k] = 0+c3
            KA_gammac[k] = 0+c4
            k += 1
            KA_gammar[k] = 0+c3
            KA_gammac[k] = 1+c4
            k += 1
            KA_gammar[k] = 0+c3
            KA_gammac[k] = 2+c4
            k += 1
            KA_gammar[k] = 1+c3
            KA_gammac[k] = 0+c1
            k += 1
            KA_gammar[k] = 1+c3
            KA_gammac[k] = 1+c1
            k += 1
            KA_gammar[k] = 1+c3
            KA_gammac[k] = 2+c1
            k += 1
            KA_gammar[k] = 1+c3
            KA_gammac[k] = 0+c2
            k += 1
            KA_gammar[k] = 1+c3
            KA_gammac[k] = 1+c2
            k += 1
            KA_gammar[k] = 1+c3
            KA_gammac[k] = 2+c2
            k += 1
            KA_gammar[k] = 1+c3
            KA_gammac[k] = 0+c3
            k += 1
            KA_gammar[k] = 1+c3
            KA_gammac[k] = 1+c3
            k += 1
            KA_gammar[k] = 1+c3
            KA_gammac[k] = 2+c3
            k += 1
            KA_gammar[k] = 1+c3
            KA_gammac[k] = 0+c4
            k += 1
            KA_gammar[k] = 1+c3
            KA_gammac[k] = 1+c4
            k += 1
            KA_gammar[k] = 1+c3
            KA_gammac[k] = 2+c4
            k += 1
            KA_gammar[k] = 2+c3
            KA_gammac[k] = 0+c1
            k += 1
            KA_gammar[k] = 2+c3
            KA_gammac[k] = 1+c1
            k += 1
            KA_gammar[k] = 2+c3
            KA_gammac[k] = 2+c1
            k += 1
            KA_gammar[k] = 2+c3
            KA_gammac[k] = 0+c2
            k += 1
            KA_gammar[k] = 2+c3
            KA_gammac[k] = 1+c2
            k += 1
            KA_gammar[k] = 2+c3
            KA_gammac[k] = 2+c2
            k += 1
            KA_gammar[k] = 2+c3
            KA_gammac[k] = 0+c3
            k += 1
            KA_gammar[k] = 2+c3
            KA_gammac[k] = 1+c3
            k += 1
            KA_gammar[k] = 2+c3
            KA_gammac[k] = 2+c3
            k += 1
            KA_gammar[k] = 2+c3
            KA_gammac[k] = 0+c4
            k += 1
            KA_gammar[k] = 2+c3
            KA_gammac[k] = 1+c4
            k += 1
            KA_gammar[k] = 2+c3
            KA_gammac[k] = 2+c4
            k += 1
            KA_gammar[k] = 0+c4
            KA_gammac[k] = 0+c1
            k += 1
            KA_gammar[k] = 0+c4
            KA_gammac[k] = 1+c1
            k += 1
            KA_gammar[k] = 0+c4
            KA_gammac[k] = 2+c1
            k += 1
            KA_gammar[k] = 0+c4
            KA_gammac[k] = 0+c2
            k += 1
            KA_gammar[k] = 0+c4
            KA_gammac[k] = 1+c2
            k += 1
            KA_gammar[k] = 0+c4
            KA_gammac[k] = 2+c2
            k += 1
            KA_gammar[k] = 0+c4
            KA_gammac[k] = 0+c3
            k += 1
            KA_gammar[k] = 0+c4
            KA_gammac[k] = 1+c3
            k += 1
            KA_gammar[k] = 0+c4
            KA_gammac[k] = 2+c3
            k += 1
            KA_gammar[k] = 0+c4
            KA_gammac[k] = 0+c4
            k += 1
            KA_gammar[k] = 0+c4
            KA_gammac[k] = 1+c4
            k += 1
            KA_gammar[k] = 0+c4
            KA_gammac[k] = 2+c4
            k += 1
            KA_gammar[k] = 1+c4
            KA_gammac[k] = 0+c1
            k += 1
            KA_gammar[k] = 1+c4
            KA_gammac[k] = 1+c1
            k += 1
            KA_gammar[k] = 1+c4
            KA_gammac[k] = 2+c1
            k += 1
            KA_gammar[k] = 1+c4
            KA_gammac[k] = 0+c2
            k += 1
            KA_gammar[k] = 1+c4
            KA_gammac[k] = 1+c2
            k += 1
            KA_gammar[k] = 1+c4
            KA_gammac[k] = 2+c2
            k += 1
            KA_gammar[k] = 1+c4
            KA_gammac[k] = 0+c3
            k += 1
            KA_gammar[k] = 1+c4
            KA_gammac[k] = 1+c3
            k += 1
            KA_gammar[k] = 1+c4
            KA_gammac[k] = 2+c3
            k += 1
            KA_gammar[k] = 1+c4
            KA_gammac[k] = 0+c4
            k += 1
            KA_gammar[k] = 1+c4
            KA_gammac[k] = 1+c4
            k += 1
            KA_gammar[k] = 1+c4
            KA_gammac[k] = 2+c4
            k += 1
            KA_gammar[k] = 2+c4
            KA_gammac[k] = 0+c1
            k += 1
            KA_gammar[k] = 2+c4
            KA_gammac[k] = 1+c1
            k += 1
            KA_gammar[k] = 2+c4
            KA_gammac[k] = 2+c1
            k += 1
            KA_gammar[k] = 2+c4
            KA_gammac[k] = 0+c2
            k += 1
            KA_gammar[k] = 2+c4
            KA_gammac[k] = 1+c2
            k += 1
            KA_gammar[k] = 2+c4
            KA_gammac[k] = 2+c2
            k += 1
            KA_gammar[k] = 2+c4
            KA_gammac[k] = 0+c3
            k += 1
            KA_gammar[k] = 2+c4
            KA_gammac[k] = 1+c3
            k += 1
            KA_gammar[k] = 2+c4
            KA_gammac[k] = 2+c3
            k += 1
            KA_gammar[k] = 2+c4
            KA_gammac[k] = 0+c4
            k += 1
            KA_gammar[k] = 2+c4
            KA_gammac[k] = 1+c4
            k += 1
            KA_gammar[k] = 2+c4
            KA_gammac[k] = 2+c4

            # NOTE full integration for KG with two-point Gauss-Legendre quadrature
            wij = 1.
            points[0] = -0.5773502691896257645092
            points[1] = +0.5773502691896257645092

            for i in range(2):
                xi = points[i]
                for j in range(2):
                    eta = points[j]

                    detJ = -0.125*eta*x1*y2 + 0.125*eta*x1*y3 + 0.125*eta*x2*y1 - 0.125*eta*x2*y4 - 0.125*eta*x3*y1 + 0.125*eta*x3*y4 + 0.125*eta*x4*y2 - 0.125*eta*x4*y3 - 0.125*x1*xi*y3 + 0.125*x1*xi*y4 + 0.125*x1*y2 - 0.125*x1*y4 + 0.125*x2*xi*y3 - 0.125*x2*xi*y4 - 0.125*x2*y1 + 0.125*x2*y3 + 0.125*x3*xi*y1 - 0.125*x3*xi*y2 - 0.125*x3*y2 + 0.125*x3*y4 - 0.125*x4*xi*y1 + 0.125*x4*xi*y2 + 0.125*x4*y1 - 0.125*x4*y3

                    S1 = 0.25*eta*xi - 0.25*eta - 0.25*xi + 0.25
                    S2 = -0.25*eta*xi - 0.25*eta + 0.25*xi + 0.25
                    S3 = 0.25*eta*xi + 0.25*eta + 0.25*xi + 0.25
                    S4 = -0.25*eta*xi + 0.25*eta - 0.25*xi + 0.25

                    k = self.init_k_KA_gamma
                    KA_gammav[k] += S1**2*detJ*r13**2*wij
                    k += 1
                    KA_gammav[k] += S1**2*detJ*r13*r23*wij
                    k += 1
                    KA_gammav[k] += S1**2*detJ*r13*r33*wij
                    k += 1
                    KA_gammav[k] += S1*S2*detJ*r13**2*wij
                    k += 1
                    KA_gammav[k] += S1*S2*detJ*r13*r23*wij
                    k += 1
                    KA_gammav[k] += S1*S2*detJ*r13*r33*wij
                    k += 1
                    KA_gammav[k] += S1*S3*detJ*r13**2*wij
                    k += 1
                    KA_gammav[k] += S1*S3*detJ*r13*r23*wij
                    k += 1
                    KA_gammav[k] += S1*S3*detJ*r13*r33*wij
                    k += 1
                    KA_gammav[k] += S1*S4*detJ*r13**2*wij
                    k += 1
                    KA_gammav[k] += S1*S4*detJ*r13*r23*wij
                    k += 1
                    KA_gammav[k] += S1*S4*detJ*r13*r33*wij
                    k += 1
                    KA_gammav[k] += S1**2*detJ*r13*r23*wij
                    k += 1
                    KA_gammav[k] += S1**2*detJ*r23**2*wij
                    k += 1
                    KA_gammav[k] += S1**2*detJ*r23*r33*wij
                    k += 1
                    KA_gammav[k] += S1*S2*detJ*r13*r23*wij
                    k += 1
                    KA_gammav[k] += S1*S2*detJ*r23**2*wij
                    k += 1
                    KA_gammav[k] += S1*S2*detJ*r23*r33*wij
                    k += 1
                    KA_gammav[k] += S1*S3*detJ*r13*r23*wij
                    k += 1
                    KA_gammav[k] += S1*S3*detJ*r23**2*wij
                    k += 1
                    KA_gammav[k] += S1*S3*detJ*r23*r33*wij
                    k += 1
                    KA_gammav[k] += S1*S4*detJ*r13*r23*wij
                    k += 1
                    KA_gammav[k] += S1*S4*detJ*r23**2*wij
                    k += 1
                    KA_gammav[k] += S1*S4*detJ*r23*r33*wij
                    k += 1
                    KA_gammav[k] += S1**2*detJ*r13*r33*wij
                    k += 1
                    KA_gammav[k] += S1**2*detJ*r23*r33*wij
                    k += 1
                    KA_gammav[k] += S1**2*detJ*r33**2*wij
                    k += 1
                    KA_gammav[k] += S1*S2*detJ*r13*r33*wij
                    k += 1
                    KA_gammav[k] += S1*S2*detJ*r23*r33*wij
                    k += 1
                    KA_gammav[k] += S1*S2*detJ*r33**2*wij
                    k += 1
                    KA_gammav[k] += S1*S3*detJ*r13*r33*wij
                    k += 1
                    KA_gammav[k] += S1*S3*detJ*r23*r33*wij
                    k += 1
                    KA_gammav[k] += S1*S3*detJ*r33**2*wij
                    k += 1
                    KA_gammav[k] += S1*S4*detJ*r13*r33*wij
                    k += 1
                    KA_gammav[k] += S1*S4*detJ*r23*r33*wij
                    k += 1
                    KA_gammav[k] += S1*S4*detJ*r33**2*wij
                    k += 1
                    KA_gammav[k] += S1*S2*detJ*r13**2*wij
                    k += 1
                    KA_gammav[k] += S1*S2*detJ*r13*r23*wij
                    k += 1
                    KA_gammav[k] += S1*S2*detJ*r13*r33*wij
                    k += 1
                    KA_gammav[k] += S2**2*detJ*r13**2*wij
                    k += 1
                    KA_gammav[k] += S2**2*detJ*r13*r23*wij
                    k += 1
                    KA_gammav[k] += S2**2*detJ*r13*r33*wij
                    k += 1
                    KA_gammav[k] += S2*S3*detJ*r13**2*wij
                    k += 1
                    KA_gammav[k] += S2*S3*detJ*r13*r23*wij
                    k += 1
                    KA_gammav[k] += S2*S3*detJ*r13*r33*wij
                    k += 1
                    KA_gammav[k] += S2*S4*detJ*r13**2*wij
                    k += 1
                    KA_gammav[k] += S2*S4*detJ*r13*r23*wij
                    k += 1
                    KA_gammav[k] += S2*S4*detJ*r13*r33*wij
                    k += 1
                    KA_gammav[k] += S1*S2*detJ*r13*r23*wij
                    k += 1
                    KA_gammav[k] += S1*S2*detJ*r23**2*wij
                    k += 1
                    KA_gammav[k] += S1*S2*detJ*r23*r33*wij
                    k += 1
                    KA_gammav[k] += S2**2*detJ*r13*r23*wij
                    k += 1
                    KA_gammav[k] += S2**2*detJ*r23**2*wij
                    k += 1
                    KA_gammav[k] += S2**2*detJ*r23*r33*wij
                    k += 1
                    KA_gammav[k] += S2*S3*detJ*r13*r23*wij
                    k += 1
                    KA_gammav[k] += S2*S3*detJ*r23**2*wij
                    k += 1
                    KA_gammav[k] += S2*S3*detJ*r23*r33*wij
                    k += 1
                    KA_gammav[k] += S2*S4*detJ*r13*r23*wij
                    k += 1
                    KA_gammav[k] += S2*S4*detJ*r23**2*wij
                    k += 1
                    KA_gammav[k] += S2*S4*detJ*r23*r33*wij
                    k += 1
                    KA_gammav[k] += S1*S2*detJ*r13*r33*wij
                    k += 1
                    KA_gammav[k] += S1*S2*detJ*r23*r33*wij
                    k += 1
                    KA_gammav[k] += S1*S2*detJ*r33**2*wij
                    k += 1
                    KA_gammav[k] += S2**2*detJ*r13*r33*wij
                    k += 1
                    KA_gammav[k] += S2**2*detJ*r23*r33*wij
                    k += 1
                    KA_gammav[k] += S2**2*detJ*r33**2*wij
                    k += 1
                    KA_gammav[k] += S2*S3*detJ*r13*r33*wij
                    k += 1
                    KA_gammav[k] += S2*S3*detJ*r23*r33*wij
                    k += 1
                    KA_gammav[k] += S2*S3*detJ*r33**2*wij
                    k += 1
                    KA_gammav[k] += S2*S4*detJ*r13*r33*wij
                    k += 1
                    KA_gammav[k] += S2*S4*detJ*r23*r33*wij
                    k += 1
                    KA_gammav[k] += S2*S4*detJ*r33**2*wij
                    k += 1
                    KA_gammav[k] += S1*S3*detJ*r13**2*wij
                    k += 1
                    KA_gammav[k] += S1*S3*detJ*r13*r23*wij
                    k += 1
                    KA_gammav[k] += S1*S3*detJ*r13*r33*wij
                    k += 1
                    KA_gammav[k] += S2*S3*detJ*r13**2*wij
                    k += 1
                    KA_gammav[k] += S2*S3*detJ*r13*r23*wij
                    k += 1
                    KA_gammav[k] += S2*S3*detJ*r13*r33*wij
                    k += 1
                    KA_gammav[k] += S3**2*detJ*r13**2*wij
                    k += 1
                    KA_gammav[k] += S3**2*detJ*r13*r23*wij
                    k += 1
                    KA_gammav[k] += S3**2*detJ*r13*r33*wij
                    k += 1
                    KA_gammav[k] += S3*S4*detJ*r13**2*wij
                    k += 1
                    KA_gammav[k] += S3*S4*detJ*r13*r23*wij
                    k += 1
                    KA_gammav[k] += S3*S4*detJ*r13*r33*wij
                    k += 1
                    KA_gammav[k] += S1*S3*detJ*r13*r23*wij
                    k += 1
                    KA_gammav[k] += S1*S3*detJ*r23**2*wij
                    k += 1
                    KA_gammav[k] += S1*S3*detJ*r23*r33*wij
                    k += 1
                    KA_gammav[k] += S2*S3*detJ*r13*r23*wij
                    k += 1
                    KA_gammav[k] += S2*S3*detJ*r23**2*wij
                    k += 1
                    KA_gammav[k] += S2*S3*detJ*r23*r33*wij
                    k += 1
                    KA_gammav[k] += S3**2*detJ*r13*r23*wij
                    k += 1
                    KA_gammav[k] += S3**2*detJ*r23**2*wij
                    k += 1
                    KA_gammav[k] += S3**2*detJ*r23*r33*wij
                    k += 1
                    KA_gammav[k] += S3*S4*detJ*r13*r23*wij
                    k += 1
                    KA_gammav[k] += S3*S4*detJ*r23**2*wij
                    k += 1
                    KA_gammav[k] += S3*S4*detJ*r23*r33*wij
                    k += 1
                    KA_gammav[k] += S1*S3*detJ*r13*r33*wij
                    k += 1
                    KA_gammav[k] += S1*S3*detJ*r23*r33*wij
                    k += 1
                    KA_gammav[k] += S1*S3*detJ*r33**2*wij
                    k += 1
                    KA_gammav[k] += S2*S3*detJ*r13*r33*wij
                    k += 1
                    KA_gammav[k] += S2*S3*detJ*r23*r33*wij
                    k += 1
                    KA_gammav[k] += S2*S3*detJ*r33**2*wij
                    k += 1
                    KA_gammav[k] += S3**2*detJ*r13*r33*wij
                    k += 1
                    KA_gammav[k] += S3**2*detJ*r23*r33*wij
                    k += 1
                    KA_gammav[k] += S3**2*detJ*r33**2*wij
                    k += 1
                    KA_gammav[k] += S3*S4*detJ*r13*r33*wij
                    k += 1
                    KA_gammav[k] += S3*S4*detJ*r23*r33*wij
                    k += 1
                    KA_gammav[k] += S3*S4*detJ*r33**2*wij
                    k += 1
                    KA_gammav[k] += S1*S4*detJ*r13**2*wij
                    k += 1
                    KA_gammav[k] += S1*S4*detJ*r13*r23*wij
                    k += 1
                    KA_gammav[k] += S1*S4*detJ*r13*r33*wij
                    k += 1
                    KA_gammav[k] += S2*S4*detJ*r13**2*wij
                    k += 1
                    KA_gammav[k] += S2*S4*detJ*r13*r23*wij
                    k += 1
                    KA_gammav[k] += S2*S4*detJ*r13*r33*wij
                    k += 1
                    KA_gammav[k] += S3*S4*detJ*r13**2*wij
                    k += 1
                    KA_gammav[k] += S3*S4*detJ*r13*r23*wij
                    k += 1
                    KA_gammav[k] += S3*S4*detJ*r13*r33*wij
                    k += 1
                    KA_gammav[k] += S4**2*detJ*r13**2*wij
                    k += 1
                    KA_gammav[k] += S4**2*detJ*r13*r23*wij
                    k += 1
                    KA_gammav[k] += S4**2*detJ*r13*r33*wij
                    k += 1
                    KA_gammav[k] += S1*S4*detJ*r13*r23*wij
                    k += 1
                    KA_gammav[k] += S1*S4*detJ*r23**2*wij
                    k += 1
                    KA_gammav[k] += S1*S4*detJ*r23*r33*wij
                    k += 1
                    KA_gammav[k] += S2*S4*detJ*r13*r23*wij
                    k += 1
                    KA_gammav[k] += S2*S4*detJ*r23**2*wij
                    k += 1
                    KA_gammav[k] += S2*S4*detJ*r23*r33*wij
                    k += 1
                    KA_gammav[k] += S3*S4*detJ*r13*r23*wij
                    k += 1
                    KA_gammav[k] += S3*S4*detJ*r23**2*wij
                    k += 1
                    KA_gammav[k] += S3*S4*detJ*r23*r33*wij
                    k += 1
                    KA_gammav[k] += S4**2*detJ*r13*r23*wij
                    k += 1
                    KA_gammav[k] += S4**2*detJ*r23**2*wij
                    k += 1
                    KA_gammav[k] += S4**2*detJ*r23*r33*wij
                    k += 1
                    KA_gammav[k] += S1*S4*detJ*r13*r33*wij
                    k += 1
                    KA_gammav[k] += S1*S4*detJ*r23*r33*wij
                    k += 1
                    KA_gammav[k] += S1*S4*detJ*r33**2*wij
                    k += 1
                    KA_gammav[k] += S2*S4*detJ*r13*r33*wij
                    k += 1
                    KA_gammav[k] += S2*S4*detJ*r23*r33*wij
                    k += 1
                    KA_gammav[k] += S2*S4*detJ*r33**2*wij
                    k += 1
                    KA_gammav[k] += S3*S4*detJ*r13*r33*wij
                    k += 1
                    KA_gammav[k] += S3*S4*detJ*r23*r33*wij
                    k += 1
                    KA_gammav[k] += S3*S4*detJ*r33**2*wij
                    k += 1
                    KA_gammav[k] += S4**2*detJ*r13*r33*wij
                    k += 1
                    KA_gammav[k] += S4**2*detJ*r23*r33*wij
                    k += 1
                    KA_gammav[k] += S4**2*detJ*r33**2*wij


    cpdef void update_CA(Quad4R self,
                        long [::1] CAr,
                        long [::1] CAc,
                        double [::1] CAv,
                        ):
        r"""Update sparse vectors for piston-theory aerodynamic damping matrix `\boldsymbol{C}_A`

        The matrix is integrated over the element with 2 by 2 Gauss-Legendre
        quadrature, from the coordinates ``probe.xe`` and the rotation matrix,
        and its values are added to ``CAv`` at the positions ``init_k_CA`` to
        ``init_k_CA + CA_SPARSE_SIZE - 1``; the global row and column indices
        are always written to ``CAr`` and ``CAc`` at the same positions. Only
        the translational degrees-of-freedom appear, through their components
        along the element normal. No aerodynamic coefficient enters the
        values, which depend only on the geometry, so the caller multiplies
        the assembled matrix by it. Before this method is called,
        :meth:`.update_rotation_matrix` and :meth:`.update_probe_xe` must have
        been used.

        Parameters
        ----------
        CAr : np.array
            Array to store row positions of sparse values
            1-D array of integers (``INT``) of size at least ``init_k_CA +
            CA_SPARSE_SIZE``.
        CAc : np.array
            Array to store column positions of sparse values
            Same type and size as ``CAr``.
        CAv : np.array
            Array to store sparse values
            1-D array of ``DOUBLE``, same size as ``CAr``. The values are added
            to the existing ones, with dimension of length squared.

        Notes
        -----
        With `w_n` the displacement along the element normal, as in
        :meth:`.update_KA_beta`, the computed matrix is

        .. math::

            \boldsymbol{C}_A = -\int_A \boldsymbol{S}^\top \boldsymbol{S} dA

        acting on the nodal values of the velocity `\dot w_n`. With the
        piston-theory damping coefficient `\mu` given by the caller, `\mu
        \boldsymbol{C}_A \dot{\boldsymbol{u}}` is the load of the term `-\mu \dot{w}_n` of the
        pressure law, which enters the damping of the equations of motion as
        `-\mu \boldsymbol{C}_A`, positive definite, see the piston theory in the notes of
        :meth:`.update_KA_beta`.

        """
        cdef int c1, c2, c3, c4, i, j, k
        cdef double r13, r23, r33
        cdef double x1, x2, x3, x4
        cdef double y1, y2, y3, y4
        cdef double S1, S2, S3, S4
        cdef double xi, eta, wij, detJ
        cdef double points[2]

        with nogil:

            # NOTE ignoring z in local coordinates
            x1 = self.probe.xe[0]
            y1 = self.probe.xe[1]
            # z1 = self.probe.xe[2]
            x2 = self.probe.xe[3]
            y2 = self.probe.xe[4]
            # z2 = self.probe.xe[5]
            x3 = self.probe.xe[6]
            y3 = self.probe.xe[7]
            # z3 = self.probe.xe[8]
            x4 = self.probe.xe[9]
            y4 = self.probe.xe[10]
            # z4 = self.probe.xe[11]

            # local to global transformation
            r13 = self.r13
            r23 = self.r23
            r33 = self.r33

            # positions the global matrices
            c1 = self.c1
            c2 = self.c2
            c3 = self.c3
            c4 = self.c4

            k = self.init_k_CA
            CAr[k] = 0+c1
            CAc[k] = 0+c1
            k += 1
            CAr[k] = 0+c1
            CAc[k] = 1+c1
            k += 1
            CAr[k] = 0+c1
            CAc[k] = 2+c1
            k += 1
            CAr[k] = 0+c1
            CAc[k] = 0+c2
            k += 1
            CAr[k] = 0+c1
            CAc[k] = 1+c2
            k += 1
            CAr[k] = 0+c1
            CAc[k] = 2+c2
            k += 1
            CAr[k] = 0+c1
            CAc[k] = 0+c3
            k += 1
            CAr[k] = 0+c1
            CAc[k] = 1+c3
            k += 1
            CAr[k] = 0+c1
            CAc[k] = 2+c3
            k += 1
            CAr[k] = 0+c1
            CAc[k] = 0+c4
            k += 1
            CAr[k] = 0+c1
            CAc[k] = 1+c4
            k += 1
            CAr[k] = 0+c1
            CAc[k] = 2+c4
            k += 1
            CAr[k] = 1+c1
            CAc[k] = 0+c1
            k += 1
            CAr[k] = 1+c1
            CAc[k] = 1+c1
            k += 1
            CAr[k] = 1+c1
            CAc[k] = 2+c1
            k += 1
            CAr[k] = 1+c1
            CAc[k] = 0+c2
            k += 1
            CAr[k] = 1+c1
            CAc[k] = 1+c2
            k += 1
            CAr[k] = 1+c1
            CAc[k] = 2+c2
            k += 1
            CAr[k] = 1+c1
            CAc[k] = 0+c3
            k += 1
            CAr[k] = 1+c1
            CAc[k] = 1+c3
            k += 1
            CAr[k] = 1+c1
            CAc[k] = 2+c3
            k += 1
            CAr[k] = 1+c1
            CAc[k] = 0+c4
            k += 1
            CAr[k] = 1+c1
            CAc[k] = 1+c4
            k += 1
            CAr[k] = 1+c1
            CAc[k] = 2+c4
            k += 1
            CAr[k] = 2+c1
            CAc[k] = 0+c1
            k += 1
            CAr[k] = 2+c1
            CAc[k] = 1+c1
            k += 1
            CAr[k] = 2+c1
            CAc[k] = 2+c1
            k += 1
            CAr[k] = 2+c1
            CAc[k] = 0+c2
            k += 1
            CAr[k] = 2+c1
            CAc[k] = 1+c2
            k += 1
            CAr[k] = 2+c1
            CAc[k] = 2+c2
            k += 1
            CAr[k] = 2+c1
            CAc[k] = 0+c3
            k += 1
            CAr[k] = 2+c1
            CAc[k] = 1+c3
            k += 1
            CAr[k] = 2+c1
            CAc[k] = 2+c3
            k += 1
            CAr[k] = 2+c1
            CAc[k] = 0+c4
            k += 1
            CAr[k] = 2+c1
            CAc[k] = 1+c4
            k += 1
            CAr[k] = 2+c1
            CAc[k] = 2+c4
            k += 1
            CAr[k] = 0+c2
            CAc[k] = 0+c1
            k += 1
            CAr[k] = 0+c2
            CAc[k] = 1+c1
            k += 1
            CAr[k] = 0+c2
            CAc[k] = 2+c1
            k += 1
            CAr[k] = 0+c2
            CAc[k] = 0+c2
            k += 1
            CAr[k] = 0+c2
            CAc[k] = 1+c2
            k += 1
            CAr[k] = 0+c2
            CAc[k] = 2+c2
            k += 1
            CAr[k] = 0+c2
            CAc[k] = 0+c3
            k += 1
            CAr[k] = 0+c2
            CAc[k] = 1+c3
            k += 1
            CAr[k] = 0+c2
            CAc[k] = 2+c3
            k += 1
            CAr[k] = 0+c2
            CAc[k] = 0+c4
            k += 1
            CAr[k] = 0+c2
            CAc[k] = 1+c4
            k += 1
            CAr[k] = 0+c2
            CAc[k] = 2+c4
            k += 1
            CAr[k] = 1+c2
            CAc[k] = 0+c1
            k += 1
            CAr[k] = 1+c2
            CAc[k] = 1+c1
            k += 1
            CAr[k] = 1+c2
            CAc[k] = 2+c1
            k += 1
            CAr[k] = 1+c2
            CAc[k] = 0+c2
            k += 1
            CAr[k] = 1+c2
            CAc[k] = 1+c2
            k += 1
            CAr[k] = 1+c2
            CAc[k] = 2+c2
            k += 1
            CAr[k] = 1+c2
            CAc[k] = 0+c3
            k += 1
            CAr[k] = 1+c2
            CAc[k] = 1+c3
            k += 1
            CAr[k] = 1+c2
            CAc[k] = 2+c3
            k += 1
            CAr[k] = 1+c2
            CAc[k] = 0+c4
            k += 1
            CAr[k] = 1+c2
            CAc[k] = 1+c4
            k += 1
            CAr[k] = 1+c2
            CAc[k] = 2+c4
            k += 1
            CAr[k] = 2+c2
            CAc[k] = 0+c1
            k += 1
            CAr[k] = 2+c2
            CAc[k] = 1+c1
            k += 1
            CAr[k] = 2+c2
            CAc[k] = 2+c1
            k += 1
            CAr[k] = 2+c2
            CAc[k] = 0+c2
            k += 1
            CAr[k] = 2+c2
            CAc[k] = 1+c2
            k += 1
            CAr[k] = 2+c2
            CAc[k] = 2+c2
            k += 1
            CAr[k] = 2+c2
            CAc[k] = 0+c3
            k += 1
            CAr[k] = 2+c2
            CAc[k] = 1+c3
            k += 1
            CAr[k] = 2+c2
            CAc[k] = 2+c3
            k += 1
            CAr[k] = 2+c2
            CAc[k] = 0+c4
            k += 1
            CAr[k] = 2+c2
            CAc[k] = 1+c4
            k += 1
            CAr[k] = 2+c2
            CAc[k] = 2+c4
            k += 1
            CAr[k] = 0+c3
            CAc[k] = 0+c1
            k += 1
            CAr[k] = 0+c3
            CAc[k] = 1+c1
            k += 1
            CAr[k] = 0+c3
            CAc[k] = 2+c1
            k += 1
            CAr[k] = 0+c3
            CAc[k] = 0+c2
            k += 1
            CAr[k] = 0+c3
            CAc[k] = 1+c2
            k += 1
            CAr[k] = 0+c3
            CAc[k] = 2+c2
            k += 1
            CAr[k] = 0+c3
            CAc[k] = 0+c3
            k += 1
            CAr[k] = 0+c3
            CAc[k] = 1+c3
            k += 1
            CAr[k] = 0+c3
            CAc[k] = 2+c3
            k += 1
            CAr[k] = 0+c3
            CAc[k] = 0+c4
            k += 1
            CAr[k] = 0+c3
            CAc[k] = 1+c4
            k += 1
            CAr[k] = 0+c3
            CAc[k] = 2+c4
            k += 1
            CAr[k] = 1+c3
            CAc[k] = 0+c1
            k += 1
            CAr[k] = 1+c3
            CAc[k] = 1+c1
            k += 1
            CAr[k] = 1+c3
            CAc[k] = 2+c1
            k += 1
            CAr[k] = 1+c3
            CAc[k] = 0+c2
            k += 1
            CAr[k] = 1+c3
            CAc[k] = 1+c2
            k += 1
            CAr[k] = 1+c3
            CAc[k] = 2+c2
            k += 1
            CAr[k] = 1+c3
            CAc[k] = 0+c3
            k += 1
            CAr[k] = 1+c3
            CAc[k] = 1+c3
            k += 1
            CAr[k] = 1+c3
            CAc[k] = 2+c3
            k += 1
            CAr[k] = 1+c3
            CAc[k] = 0+c4
            k += 1
            CAr[k] = 1+c3
            CAc[k] = 1+c4
            k += 1
            CAr[k] = 1+c3
            CAc[k] = 2+c4
            k += 1
            CAr[k] = 2+c3
            CAc[k] = 0+c1
            k += 1
            CAr[k] = 2+c3
            CAc[k] = 1+c1
            k += 1
            CAr[k] = 2+c3
            CAc[k] = 2+c1
            k += 1
            CAr[k] = 2+c3
            CAc[k] = 0+c2
            k += 1
            CAr[k] = 2+c3
            CAc[k] = 1+c2
            k += 1
            CAr[k] = 2+c3
            CAc[k] = 2+c2
            k += 1
            CAr[k] = 2+c3
            CAc[k] = 0+c3
            k += 1
            CAr[k] = 2+c3
            CAc[k] = 1+c3
            k += 1
            CAr[k] = 2+c3
            CAc[k] = 2+c3
            k += 1
            CAr[k] = 2+c3
            CAc[k] = 0+c4
            k += 1
            CAr[k] = 2+c3
            CAc[k] = 1+c4
            k += 1
            CAr[k] = 2+c3
            CAc[k] = 2+c4
            k += 1
            CAr[k] = 0+c4
            CAc[k] = 0+c1
            k += 1
            CAr[k] = 0+c4
            CAc[k] = 1+c1
            k += 1
            CAr[k] = 0+c4
            CAc[k] = 2+c1
            k += 1
            CAr[k] = 0+c4
            CAc[k] = 0+c2
            k += 1
            CAr[k] = 0+c4
            CAc[k] = 1+c2
            k += 1
            CAr[k] = 0+c4
            CAc[k] = 2+c2
            k += 1
            CAr[k] = 0+c4
            CAc[k] = 0+c3
            k += 1
            CAr[k] = 0+c4
            CAc[k] = 1+c3
            k += 1
            CAr[k] = 0+c4
            CAc[k] = 2+c3
            k += 1
            CAr[k] = 0+c4
            CAc[k] = 0+c4
            k += 1
            CAr[k] = 0+c4
            CAc[k] = 1+c4
            k += 1
            CAr[k] = 0+c4
            CAc[k] = 2+c4
            k += 1
            CAr[k] = 1+c4
            CAc[k] = 0+c1
            k += 1
            CAr[k] = 1+c4
            CAc[k] = 1+c1
            k += 1
            CAr[k] = 1+c4
            CAc[k] = 2+c1
            k += 1
            CAr[k] = 1+c4
            CAc[k] = 0+c2
            k += 1
            CAr[k] = 1+c4
            CAc[k] = 1+c2
            k += 1
            CAr[k] = 1+c4
            CAc[k] = 2+c2
            k += 1
            CAr[k] = 1+c4
            CAc[k] = 0+c3
            k += 1
            CAr[k] = 1+c4
            CAc[k] = 1+c3
            k += 1
            CAr[k] = 1+c4
            CAc[k] = 2+c3
            k += 1
            CAr[k] = 1+c4
            CAc[k] = 0+c4
            k += 1
            CAr[k] = 1+c4
            CAc[k] = 1+c4
            k += 1
            CAr[k] = 1+c4
            CAc[k] = 2+c4
            k += 1
            CAr[k] = 2+c4
            CAc[k] = 0+c1
            k += 1
            CAr[k] = 2+c4
            CAc[k] = 1+c1
            k += 1
            CAr[k] = 2+c4
            CAc[k] = 2+c1
            k += 1
            CAr[k] = 2+c4
            CAc[k] = 0+c2
            k += 1
            CAr[k] = 2+c4
            CAc[k] = 1+c2
            k += 1
            CAr[k] = 2+c4
            CAc[k] = 2+c2
            k += 1
            CAr[k] = 2+c4
            CAc[k] = 0+c3
            k += 1
            CAr[k] = 2+c4
            CAc[k] = 1+c3
            k += 1
            CAr[k] = 2+c4
            CAc[k] = 2+c3
            k += 1
            CAr[k] = 2+c4
            CAc[k] = 0+c4
            k += 1
            CAr[k] = 2+c4
            CAc[k] = 1+c4
            k += 1
            CAr[k] = 2+c4
            CAc[k] = 2+c4

            # NOTE full integration for KG with two-point Gauss-Legendre quadrature
            wij = 1.
            points[0] = -0.5773502691896257645092
            points[1] = +0.5773502691896257645092

            for i in range(2):
                xi = points[i]
                for j in range(2):
                    eta = points[j]

                    detJ = -0.125*eta*x1*y2 + 0.125*eta*x1*y3 + 0.125*eta*x2*y1 - 0.125*eta*x2*y4 - 0.125*eta*x3*y1 + 0.125*eta*x3*y4 + 0.125*eta*x4*y2 - 0.125*eta*x4*y3 - 0.125*x1*xi*y3 + 0.125*x1*xi*y4 + 0.125*x1*y2 - 0.125*x1*y4 + 0.125*x2*xi*y3 - 0.125*x2*xi*y4 - 0.125*x2*y1 + 0.125*x2*y3 + 0.125*x3*xi*y1 - 0.125*x3*xi*y2 - 0.125*x3*y2 + 0.125*x3*y4 - 0.125*x4*xi*y1 + 0.125*x4*xi*y2 + 0.125*x4*y1 - 0.125*x4*y3

                    S1 = 0.25*eta*xi - 0.25*eta - 0.25*xi + 0.25
                    S2 = -0.25*eta*xi - 0.25*eta + 0.25*xi + 0.25
                    S3 = 0.25*eta*xi + 0.25*eta + 0.25*xi + 0.25
                    S4 = -0.25*eta*xi + 0.25*eta - 0.25*xi + 0.25

                    k = self.init_k_CA
                    CAv[k] += -S1**2*detJ*r13**2*wij
                    k += 1
                    CAv[k] += -S1**2*detJ*r13*r23*wij
                    k += 1
                    CAv[k] += -S1**2*detJ*r13*r33*wij
                    k += 1
                    CAv[k] += -S1*S2*detJ*r13**2*wij
                    k += 1
                    CAv[k] += -S1*S2*detJ*r13*r23*wij
                    k += 1
                    CAv[k] += -S1*S2*detJ*r13*r33*wij
                    k += 1
                    CAv[k] += -S1*S3*detJ*r13**2*wij
                    k += 1
                    CAv[k] += -S1*S3*detJ*r13*r23*wij
                    k += 1
                    CAv[k] += -S1*S3*detJ*r13*r33*wij
                    k += 1
                    CAv[k] += -S1*S4*detJ*r13**2*wij
                    k += 1
                    CAv[k] += -S1*S4*detJ*r13*r23*wij
                    k += 1
                    CAv[k] += -S1*S4*detJ*r13*r33*wij
                    k += 1
                    CAv[k] += -S1**2*detJ*r13*r23*wij
                    k += 1
                    CAv[k] += -S1**2*detJ*r23**2*wij
                    k += 1
                    CAv[k] += -S1**2*detJ*r23*r33*wij
                    k += 1
                    CAv[k] += -S1*S2*detJ*r13*r23*wij
                    k += 1
                    CAv[k] += -S1*S2*detJ*r23**2*wij
                    k += 1
                    CAv[k] += -S1*S2*detJ*r23*r33*wij
                    k += 1
                    CAv[k] += -S1*S3*detJ*r13*r23*wij
                    k += 1
                    CAv[k] += -S1*S3*detJ*r23**2*wij
                    k += 1
                    CAv[k] += -S1*S3*detJ*r23*r33*wij
                    k += 1
                    CAv[k] += -S1*S4*detJ*r13*r23*wij
                    k += 1
                    CAv[k] += -S1*S4*detJ*r23**2*wij
                    k += 1
                    CAv[k] += -S1*S4*detJ*r23*r33*wij
                    k += 1
                    CAv[k] += -S1**2*detJ*r13*r33*wij
                    k += 1
                    CAv[k] += -S1**2*detJ*r23*r33*wij
                    k += 1
                    CAv[k] += -S1**2*detJ*r33**2*wij
                    k += 1
                    CAv[k] += -S1*S2*detJ*r13*r33*wij
                    k += 1
                    CAv[k] += -S1*S2*detJ*r23*r33*wij
                    k += 1
                    CAv[k] += -S1*S2*detJ*r33**2*wij
                    k += 1
                    CAv[k] += -S1*S3*detJ*r13*r33*wij
                    k += 1
                    CAv[k] += -S1*S3*detJ*r23*r33*wij
                    k += 1
                    CAv[k] += -S1*S3*detJ*r33**2*wij
                    k += 1
                    CAv[k] += -S1*S4*detJ*r13*r33*wij
                    k += 1
                    CAv[k] += -S1*S4*detJ*r23*r33*wij
                    k += 1
                    CAv[k] += -S1*S4*detJ*r33**2*wij
                    k += 1
                    CAv[k] += -S1*S2*detJ*r13**2*wij
                    k += 1
                    CAv[k] += -S1*S2*detJ*r13*r23*wij
                    k += 1
                    CAv[k] += -S1*S2*detJ*r13*r33*wij
                    k += 1
                    CAv[k] += -S2**2*detJ*r13**2*wij
                    k += 1
                    CAv[k] += -S2**2*detJ*r13*r23*wij
                    k += 1
                    CAv[k] += -S2**2*detJ*r13*r33*wij
                    k += 1
                    CAv[k] += -S2*S3*detJ*r13**2*wij
                    k += 1
                    CAv[k] += -S2*S3*detJ*r13*r23*wij
                    k += 1
                    CAv[k] += -S2*S3*detJ*r13*r33*wij
                    k += 1
                    CAv[k] += -S2*S4*detJ*r13**2*wij
                    k += 1
                    CAv[k] += -S2*S4*detJ*r13*r23*wij
                    k += 1
                    CAv[k] += -S2*S4*detJ*r13*r33*wij
                    k += 1
                    CAv[k] += -S1*S2*detJ*r13*r23*wij
                    k += 1
                    CAv[k] += -S1*S2*detJ*r23**2*wij
                    k += 1
                    CAv[k] += -S1*S2*detJ*r23*r33*wij
                    k += 1
                    CAv[k] += -S2**2*detJ*r13*r23*wij
                    k += 1
                    CAv[k] += -S2**2*detJ*r23**2*wij
                    k += 1
                    CAv[k] += -S2**2*detJ*r23*r33*wij
                    k += 1
                    CAv[k] += -S2*S3*detJ*r13*r23*wij
                    k += 1
                    CAv[k] += -S2*S3*detJ*r23**2*wij
                    k += 1
                    CAv[k] += -S2*S3*detJ*r23*r33*wij
                    k += 1
                    CAv[k] += -S2*S4*detJ*r13*r23*wij
                    k += 1
                    CAv[k] += -S2*S4*detJ*r23**2*wij
                    k += 1
                    CAv[k] += -S2*S4*detJ*r23*r33*wij
                    k += 1
                    CAv[k] += -S1*S2*detJ*r13*r33*wij
                    k += 1
                    CAv[k] += -S1*S2*detJ*r23*r33*wij
                    k += 1
                    CAv[k] += -S1*S2*detJ*r33**2*wij
                    k += 1
                    CAv[k] += -S2**2*detJ*r13*r33*wij
                    k += 1
                    CAv[k] += -S2**2*detJ*r23*r33*wij
                    k += 1
                    CAv[k] += -S2**2*detJ*r33**2*wij
                    k += 1
                    CAv[k] += -S2*S3*detJ*r13*r33*wij
                    k += 1
                    CAv[k] += -S2*S3*detJ*r23*r33*wij
                    k += 1
                    CAv[k] += -S2*S3*detJ*r33**2*wij
                    k += 1
                    CAv[k] += -S2*S4*detJ*r13*r33*wij
                    k += 1
                    CAv[k] += -S2*S4*detJ*r23*r33*wij
                    k += 1
                    CAv[k] += -S2*S4*detJ*r33**2*wij
                    k += 1
                    CAv[k] += -S1*S3*detJ*r13**2*wij
                    k += 1
                    CAv[k] += -S1*S3*detJ*r13*r23*wij
                    k += 1
                    CAv[k] += -S1*S3*detJ*r13*r33*wij
                    k += 1
                    CAv[k] += -S2*S3*detJ*r13**2*wij
                    k += 1
                    CAv[k] += -S2*S3*detJ*r13*r23*wij
                    k += 1
                    CAv[k] += -S2*S3*detJ*r13*r33*wij
                    k += 1
                    CAv[k] += -S3**2*detJ*r13**2*wij
                    k += 1
                    CAv[k] += -S3**2*detJ*r13*r23*wij
                    k += 1
                    CAv[k] += -S3**2*detJ*r13*r33*wij
                    k += 1
                    CAv[k] += -S3*S4*detJ*r13**2*wij
                    k += 1
                    CAv[k] += -S3*S4*detJ*r13*r23*wij
                    k += 1
                    CAv[k] += -S3*S4*detJ*r13*r33*wij
                    k += 1
                    CAv[k] += -S1*S3*detJ*r13*r23*wij
                    k += 1
                    CAv[k] += -S1*S3*detJ*r23**2*wij
                    k += 1
                    CAv[k] += -S1*S3*detJ*r23*r33*wij
                    k += 1
                    CAv[k] += -S2*S3*detJ*r13*r23*wij
                    k += 1
                    CAv[k] += -S2*S3*detJ*r23**2*wij
                    k += 1
                    CAv[k] += -S2*S3*detJ*r23*r33*wij
                    k += 1
                    CAv[k] += -S3**2*detJ*r13*r23*wij
                    k += 1
                    CAv[k] += -S3**2*detJ*r23**2*wij
                    k += 1
                    CAv[k] += -S3**2*detJ*r23*r33*wij
                    k += 1
                    CAv[k] += -S3*S4*detJ*r13*r23*wij
                    k += 1
                    CAv[k] += -S3*S4*detJ*r23**2*wij
                    k += 1
                    CAv[k] += -S3*S4*detJ*r23*r33*wij
                    k += 1
                    CAv[k] += -S1*S3*detJ*r13*r33*wij
                    k += 1
                    CAv[k] += -S1*S3*detJ*r23*r33*wij
                    k += 1
                    CAv[k] += -S1*S3*detJ*r33**2*wij
                    k += 1
                    CAv[k] += -S2*S3*detJ*r13*r33*wij
                    k += 1
                    CAv[k] += -S2*S3*detJ*r23*r33*wij
                    k += 1
                    CAv[k] += -S2*S3*detJ*r33**2*wij
                    k += 1
                    CAv[k] += -S3**2*detJ*r13*r33*wij
                    k += 1
                    CAv[k] += -S3**2*detJ*r23*r33*wij
                    k += 1
                    CAv[k] += -S3**2*detJ*r33**2*wij
                    k += 1
                    CAv[k] += -S3*S4*detJ*r13*r33*wij
                    k += 1
                    CAv[k] += -S3*S4*detJ*r23*r33*wij
                    k += 1
                    CAv[k] += -S3*S4*detJ*r33**2*wij
                    k += 1
                    CAv[k] += -S1*S4*detJ*r13**2*wij
                    k += 1
                    CAv[k] += -S1*S4*detJ*r13*r23*wij
                    k += 1
                    CAv[k] += -S1*S4*detJ*r13*r33*wij
                    k += 1
                    CAv[k] += -S2*S4*detJ*r13**2*wij
                    k += 1
                    CAv[k] += -S2*S4*detJ*r13*r23*wij
                    k += 1
                    CAv[k] += -S2*S4*detJ*r13*r33*wij
                    k += 1
                    CAv[k] += -S3*S4*detJ*r13**2*wij
                    k += 1
                    CAv[k] += -S3*S4*detJ*r13*r23*wij
                    k += 1
                    CAv[k] += -S3*S4*detJ*r13*r33*wij
                    k += 1
                    CAv[k] += -S4**2*detJ*r13**2*wij
                    k += 1
                    CAv[k] += -S4**2*detJ*r13*r23*wij
                    k += 1
                    CAv[k] += -S4**2*detJ*r13*r33*wij
                    k += 1
                    CAv[k] += -S1*S4*detJ*r13*r23*wij
                    k += 1
                    CAv[k] += -S1*S4*detJ*r23**2*wij
                    k += 1
                    CAv[k] += -S1*S4*detJ*r23*r33*wij
                    k += 1
                    CAv[k] += -S2*S4*detJ*r13*r23*wij
                    k += 1
                    CAv[k] += -S2*S4*detJ*r23**2*wij
                    k += 1
                    CAv[k] += -S2*S4*detJ*r23*r33*wij
                    k += 1
                    CAv[k] += -S3*S4*detJ*r13*r23*wij
                    k += 1
                    CAv[k] += -S3*S4*detJ*r23**2*wij
                    k += 1
                    CAv[k] += -S3*S4*detJ*r23*r33*wij
                    k += 1
                    CAv[k] += -S4**2*detJ*r13*r23*wij
                    k += 1
                    CAv[k] += -S4**2*detJ*r23**2*wij
                    k += 1
                    CAv[k] += -S4**2*detJ*r23*r33*wij
                    k += 1
                    CAv[k] += -S1*S4*detJ*r13*r33*wij
                    k += 1
                    CAv[k] += -S1*S4*detJ*r23*r33*wij
                    k += 1
                    CAv[k] += -S1*S4*detJ*r33**2*wij
                    k += 1
                    CAv[k] += -S2*S4*detJ*r13*r33*wij
                    k += 1
                    CAv[k] += -S2*S4*detJ*r23*r33*wij
                    k += 1
                    CAv[k] += -S2*S4*detJ*r33**2*wij
                    k += 1
                    CAv[k] += -S3*S4*detJ*r13*r33*wij
                    k += 1
                    CAv[k] += -S3*S4*detJ*r23*r33*wij
                    k += 1
                    CAv[k] += -S3*S4*detJ*r33**2*wij
                    k += 1
                    CAv[k] += -S4**2*detJ*r13*r33*wij
                    k += 1
                    CAv[k] += -S4**2*detJ*r23*r33*wij
                    k += 1
                    CAv[k] += -S4**2*detJ*r33**2*wij

