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
:class:`.Quad4` element. An hourglass control adds an artificial
stiffness to `u`, `v`, `w`, `r_x` and `r_y`, whose magnitude can be scaled
with the ``hgfactor_*`` parameters of :meth:`.Quad4R.update_KC0`,
:meth:`.Quad4R.update_probe_finte` and :meth:`.Quad4R.update_fint`.

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
(`\xi=\eta=0`) and weight `w_{ij}=4`, preventing shear locking.
The hourglass control is used according to Brockman 1987 [1]_, with the
orthogonalised hourglass operator of his Eq. (14), which he quotes from
Belytschko and Tsay 1983 [2]_. Brockman considers only isotropic plates,
and his generalized stiffnesses contain the Young's modulus `E`, times `t`
for the in-plane translations `u`, `v` and times `t^3` for `w` and the
rotations. For a laminate, `E` is replaced by equivalent moduli in the
element coordinate system, see :meth:`.Quad4R.update_KC0`: the membrane
moduli `E_{1eq}` and `E_{2eq}` of the extensional compliance `\boldsymbol{A}^{-1}` for
`u` and `v`, and the bending moduli `E_{1b}` and `E_{2b}` of the bending
compliance, the lower-right block of `[[A, B], [B, D]]^{-1}`, for `w`,
`r_x` and `r_y`. Each is the modulus of the homogeneous plate that has the
compliance of the laminate in that direction, so both pairs reduce to `E`
for an isotropic plate, and they differ for a laminate whose bending
stiffness is not `t^2/12` times its extensional stiffness, e.g. a cross-ply
or a sandwich. Brockman calibrated the coefficients of his Eq. (16) on
isotropic plates, and this generalisation does not recalibrate them for
laminates; :class:`.Quad4`, with its mixed quadrature, needs no hourglass
control. With the bending moduli, the artificial stiffness of the rotations
is the same fraction of the bending stiffness of the plate for every
laminate as for an isotropic plate. On simply supported plates under
uniform pressure, with 8 by 8 elements, the centre deflection of an
unsymmetric `[0/90]` cross-ply, whose `\boldsymbol{B}` gives `E_{1b} = 0.36
E_{1eq}`, is 15 % smaller than the Navier solution of the FSDT, against
32 % with the membrane moduli for the rotations, used before 0.11.0, which
over-stabilised it; that of a sandwich with thin faces, `E_{1b} = 2.7
E_{1eq}`, is 14 % smaller, against 6 %, the membrane moduli having
under-stabilised it; see ``test_laminated_plates_against_navier`` in
``test_quad4r_hourglass_control.py``.

.. rubric:: Dimensional homogeneity of the hourglass stiffnesses

The generalized stiffnesses of Brockman's Eqs. (16a) and (16b),

.. math::

    E^{(h)}_u = E^{(h)}_v = \frac{0.10 E t}{1 + 1/A}
    \qquad
    E^{(h)}_w = E^{(h)}_{\theta_x} = E^{(h)}_{\theta_y}
              = \frac{0.10 E t^3}{1 + 1/A}

are not dimensionally homogeneous, `1/A` being an inverse area, so the
artificial stiffness depends on the unit of length the model is written in.
The hourglass term enters the element matrix as

.. math::

    K_{ij} = A E^{(h)} \gamma_i \gamma_j

with `\gamma = \partial^2 \boldsymbol{S}/\partial x \partial y` evaluated at the
centroid, of dimension `1/L^2`. The hourglass amplitude of a translation is
`\gamma^\top u`, of dimension `1/L`, and of a rotation is `\gamma^\top \theta`, of
dimension `1/L^2`. Requiring `K_{ij}` to be a force per unit displacement
for the translations and a moment per unit rotation for the rotations gives

.. math::

    \boldsymbol{E}^{(h)} = F L \quad \text{for } u, v, w
    \qquad
    \boldsymbol{E}^{(h)} = F L^3 \quad \text{for } \theta_x, \theta_y

Since `1/(1 + 1/A) \rightarrow A` for `A \ll 1` and `\rightarrow 1` for
`A \gg 1`, Brockman's factor supplies the area that Eq. (16a) is missing
only in a unit of length that makes the element areas small, and stops
supplying it in a unit that makes them large. The switch happens at
`A = 1` in whatever unit is used. He introduces the factor knowingly, as
"motivated by locking problems observed in elements with extremely small
dimensions", and reports good behaviour "over a range of six orders of
magnitude in the planform dimension".

This element replaces the factor by the area itself wherever doing so costs
nothing, which is four of the five coefficients:

.. math::

    E^{(h)}_u = 0.10 E_{1eq} t A
    \qquad
    E^{(h)}_v = 0.10 E_{2eq} t A

.. math::

    E^{(h)}_{\theta_x} = 0.10 E_{2b} t^3 A
    \qquad
    E^{(h)}_{\theta_y} = 0.10 E_{1b} t^3 A

Each of these scales under a change of length unit exactly as the physical
stiffness it stabilises, `Et` for the in-plane translations and `Et^3` for
the rotations, so the ratio of artificial to physical stiffness is the same
number in every unit. In the small-area limit they coincide with
Eqs. (16a) and (16b), so no benchmark of this element moves: the change is
a reinterpretation of Brockman's factor as the area it tends to, not a
different magnitude of stabilisation. Cook's in-plane bending problem,
whose response legitimately contains hourglass components, is reproduced to
twelve significant figures over nine orders of magnitude of length unit,
where before it drifted at the metre and collapsed at the kilometre.

The transverse coefficient `E^{(h)}_w` keeps Brockman's factor, and this is
a deliberate limitation rather than an oversight. The hourglass operator
cannot distinguish the spurious pattern `w = xy` with
`\theta_x = \theta_y = 0` from a legitimate twist curvature, since both
give `\gamma^\top w = \partial^2 w/\partial x \partial y`. The stabilisation
therefore also stiffens real twist, by the ratio
`3 E^{(h)}_w/(E t^3)`, and the amount that is acceptable is a property of
the problem rather than of the element. Measured on this element's own
benchmarks, as a multiple of `E t^3`: the simply supported plate of
``test_quad4r_linear_buckling_plate_given_stress.py``, `b/t = 250`,
tolerates up to about `10^{-3}`, which changes its buckling load by 0.2
to 0.3 %, whereas `1.67 \times 10^{-2}` stiffens it by about 4.5 %, with
30x14 and with 92x44 elements; the one-element-wide torsion strip of
MacNeal and Harder (1985) [7]_, for which the twist *is* the hourglass
pattern, needs about `10^{-1}`, which gives 0.90 times the reference
twist, and is 1.75, 4.77 and 34 times too flexible with `5 \times
10^{-2}`, `1.67 \times 10^{-2}` and `10^{-3}`. Those are two orders of
magnitude apart and no single dimensionless constant serves both.
Brockman's factor grows with the element area, and in SI units it gives
the thin plates `2.4 \times 10^{-4}` (30x14) and `2.5 \times 10^{-5}`
(92x44), but the strip, whose elements have the area 0.2 in the unit of
its model, only `1.67 \times 10^{-2}`: with one element across its width
the strip remains 4.77 times too flexible, grade F of MacNeal and Harder,
and needs at least two elements across, see
``test_transverse_coefficient_has_no_unit_invariant_constant``.

So the residual unit dependence of this element's hourglass control is
confined to `E^{(h)}_w`, it is documented, and it is measured in
``test_quad4r_hourglass_control.py``. Removing it properly needs an
operator that separates the spurious transverse pattern from genuine twist,
which is what an assumed-strain formulation such as MITC4 provides, rather
than a better choice of coefficient.

The hinged cylindrical roof of Sze et al. (2004) [8]_ shows this dependence
on the problem. Its elements have areas of 16 to 1008 in the unit of length
of the model, the default `E^{(h)}_w` is then `0.94` to `1.0` times
`0.1 E t^3`, and the linear deflection under the central point load is
6 to 7 % smaller than with :class:`.Tria3DSG`, which has no hourglass
control, and, with the transverse shear stiffness multiplied by 100, than
with a shallow-shell Kirchhoff solution. Smaller values of ``hgfactor_w``
approach both.
These measurements show the sensitivity of one problem and do not recommend
another value: the default follows the references above, and a different
default would need further studies.

The coefficients and the operator, as computed, are given in the Notes of
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
  modulus of the laminate and not an adjustable parameter;
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

"""
from libc.math cimport fabs

import numpy as np

from .shellprop cimport ShellProp

cdef int DOF = 6
cdef int NUM_NODES = 4


cdef inline void _hourglass_bending_moduli(double h, double *Ae, double *Be,
        double *De, double *E1b, double *E2b) noexcept nogil:
    r"""Equivalent bending moduli of the hourglass control

    ``E1b = 12/(h**3 d11)`` and ``E2b = 12/(h**3 d22)``, with `[d] = (D -
    B A^{-1} B)^{-1}` the lower-right block of `[[A, B], [B, D]]^{-1}`, from
    the row-major 3x3 matrices ``Ae``, ``Be`` and ``De`` in the element
    coordinate system. See the Notes of :meth:`.Quad4R.update_KC0`.

    """
    cdef int i, j, k
    cdef double detA, detDs
    cdef double Ai[9]
    cdef double BAi[9]
    cdef double Ds[9]
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
    # NOTE bending stiffness with zero membrane forces, D - B A^-1 B, which
    #      is D itself when B is zero and does not depend on the reference
    #      surface set by ShellProp.offset
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
    detDs = (Ds[0]*(Ds[4]*Ds[8] - Ds[5]*Ds[7])
             - Ds[1]*(Ds[3]*Ds[8] - Ds[5]*Ds[6])
             + Ds[2]*(Ds[3]*Ds[7] - Ds[4]*Ds[6]))
    # NOTE 1/d11 and 1/d22 from the cofactors of D - B A^-1 B
    E1b[0] = 12.*detDs/(h**3*(Ds[4]*Ds[8] - Ds[5]*Ds[7]))
    E2b[0] = 12.*detDs/(h**3*(Ds[0]*Ds[8] - Ds[2]*Ds[6]))


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
        ``KG_SPARSE_SIZE = 144``

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
        self.KG_SPARSE_SIZE = 144
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
        in the following order `{u_e}_1, {v_e}_1, {w_e}_1, {{r_x}_e}_1,
        {{r_y}_e}_1, {{r_z}_e}_1`, `{u_e}_2, {v_e}_2, {w_e}_2, {{r_x}_e}_2,
        {{r_y}_e}_2, {{r_z}_e}_2`, `{u_e}_3, {v_e}_3, {w_e}_3, {{r_x}_e}_3,
        {{r_y}_e}_3, {{r_z}_e}_3`, `{u_e}_4, {v_e}_4, {w_e}_4, {{r_x}_e}_4,
        {{r_y}_e}_4, {{r_z}_e}_4`.
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

    References
    ----------
    .. [1] Hughes, T. J. R., and Brezzi, F., On drilling degrees of freedom,
       Computer Methods in Applied Mechanics and Engineering, 72(1), 105-121,
       1989. https://doi.org/10.1016/0045-7825(89)90124-2
    .. [2] Allman, D. J., A compatible triangular element including vertex
       rotations for plane elasticity analysis, Computers & Structures,
       19(1-2), 1-8, 1984. https://doi.org/10.1016/0045-7949(84)90197-4

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
                           double hgfactor_u = 1.,
                           double hgfactor_v = 1.,
                           double hgfactor_w = 1.,
                           double hgfactor_rx = 1.,
                           double hgfactor_ry = 1.,
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
        hgfactor_u, hgfactor_v, hgfactor_w, hgfactor_rx, hgfactor_ry : double
            These offer the possibility to change the default hourglass
            stiffnesses for each degree-of-freedom.
            Dimensionless multipliers of the hourglass stiffnesses of ``u``,
            ``v``, ``w``, ``r_x`` and ``r_y``. Default ``1.``.
        nonlinear : int, optional
            The default ``0`` gives the linear internal forces, ``KC0*u``. Any other
            value adds the geometrically nonlinear terms of the von Karman strains,
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

        """
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
        cdef double E1eq, E2eq, E1b, E2b
        cdef double K6ROT
        cdef double Eu, Ev, Erx, Ery, Ew
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
            # NOTE equivalent bending moduli of the hourglass control of w,
            #      rx and ry, from the bending compliance of the laminate in
            #      the element coordinate system, see the Notes of update_KC0
            _hourglass_bending_moduli(prop.h, Ae, Be, De, &E1b, &E2b)


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

            # NOTE Brockman's hourglass control is derived for isotropic
            #      plates, and applied to laminates with the membrane moduli
            #      E1eq and E2eq for u and v, and with the bending moduli
            #      E1b and E2b for w, rx and ry, see the module documentation
            # NOTE Brockman's 1/(1 + 1/A) of Eq. (16) is replaced by the
            #      dimensionally homogeneous area factor, see the section
            #      "Dimensional homogeneity of the hourglass stiffnesses"
            #      of the module documentation. In short, the term enters
            #      as K_ij = A*E^(h)*gamma_i*gamma_j with gamma of
            #      dimension 1/L**2, so E^(h) must be F*L for the
            #      translations u, v and w and F*L**3 for the rotations
            #      rx and ry. Brockman's factor tends to A only in the
            #      limit A << 1, so it supplies the missing area in some
            #      units and not in others
            Eu = hgfactor_u*0.1*E1eq*prop.h*self.area
            Ev = hgfactor_v*0.1*E2eq*prop.h*self.area
            Erx = hgfactor_rx*0.1*E2b*prop.h**3*self.area
            Ery = hgfactor_ry*0.1*E1b*prop.h**3*self.area
            # NOTE w alone keeps Brockman's 1/(1 + 1/A), so this is the
            #      one coefficient of the scheme that remains unit
            #      dependent, and that is deliberate. The operator cannot
            #      separate the spurious pattern w = x*y from a legitimate
            #      twist curvature, both giving gamma'w = d2w/dxdy, so the
            #      amount of stabilisation wanted is a property of the
            #      problem and not of the element, and no dimensionless
            #      constant reproduces it: the thin plates want about
            #      2e-4 of E*h**3 and the one-element-wide torsion strip of
            #      MacNeal and Harder wants about 2e-2, two orders of
            #      magnitude apart. Brockman's factor happens to supply
            #      both, because it grows with the element area, which in
            #      SI units correlates with what each of those problems
            #      needs. See the module documentation, and
            #      test_quad4r_hourglass_control.py for the measurements.
            #      The mean of the two bending moduli generalises
            #      Eq. (16b) to laminates and reduces to it for an
            #      isotropic plate. It is built from the moduli and not
            #      from Erx and Ery, which already carry hgfactor_rx and
            #      hgfactor_ry, so that hgfactor_w scales this term and
            #      nothing else
            Ew = (hgfactor_w*0.05*(E1b + E2b)*prop.h**3
                  /(1.0 + 1.0/self.area))

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
            KC0e0202 = detJ*wij*(Ew*gamma1**2 + S1_x*(A45*S1_y + A55*S1_x) + S1_y*(A44*S1_y + A45*S1_x))
            KC0e0203 = -S1*detJ*wij*(A44*S1_y + A45*S1_x)
            KC0e0204 = S1*detJ*wij*(A45*S1_y + A55*S1_x)
            KC0e0208 = detJ*wij*(Ew*gamma1*gamma2 + S2_x*(A45*S1_y + A55*S1_x) + S2_y*(A44*S1_y + A45*S1_x))
            KC0e0209 = -S2*detJ*wij*(A44*S1_y + A45*S1_x)
            KC0e0210 = S2*detJ*wij*(A45*S1_y + A55*S1_x)
            KC0e0214 = detJ*wij*(Ew*gamma1*gamma3 + S3_x*(A45*S1_y + A55*S1_x) + S3_y*(A44*S1_y + A45*S1_x))
            KC0e0215 = -S3*detJ*wij*(A44*S1_y + A45*S1_x)
            KC0e0216 = S3*detJ*wij*(A45*S1_y + A55*S1_x)
            KC0e0220 = detJ*wij*(Ew*gamma1*gamma4 + S4_x*(A45*S1_y + A55*S1_x) + S4_y*(A44*S1_y + A45*S1_x))
            KC0e0221 = -S4*detJ*wij*(A44*S1_y + A45*S1_x)
            KC0e0222 = S4*detJ*wij*(A45*S1_y + A55*S1_x)
            KC0e0303 = detJ*wij*(A44*S1**2 + Erx*gamma1**2 + S1_x*(D26*S1_y + D66*S1_x) + S1_y*(D22*S1_y + D26*S1_x))
            KC0e0304 = -detJ*wij*(A45*S1**2 + S1_x*(D12*S1_y + D16*S1_x) + S1_y*(D26*S1_y + D66*S1_x))
            KC0e0306 = -detJ*wij*(S2_x*(B12*S1_y + B16*S1_x) + S2_y*(B26*S1_y + B66*S1_x))
            KC0e0307 = -detJ*wij*(S2_x*(B26*S1_y + B66*S1_x) + S2_y*(B22*S1_y + B26*S1_x))
            KC0e0308 = -S1*detJ*wij*(A44*S2_y + A45*S2_x)
            KC0e0309 = detJ*wij*(A44*S1*S2 + Erx*gamma1*gamma2 + S2_x*(D26*S1_y + D66*S1_x) + S2_y*(D22*S1_y + D26*S1_x))
            KC0e0310 = -detJ*wij*(A45*S1*S2 + S2_x*(D12*S1_y + D16*S1_x) + S2_y*(D26*S1_y + D66*S1_x))
            KC0e0312 = -detJ*wij*(S3_x*(B12*S1_y + B16*S1_x) + S3_y*(B26*S1_y + B66*S1_x))
            KC0e0313 = -detJ*wij*(S3_x*(B26*S1_y + B66*S1_x) + S3_y*(B22*S1_y + B26*S1_x))
            KC0e0314 = -S1*detJ*wij*(A44*S3_y + A45*S3_x)
            KC0e0315 = detJ*wij*(A44*S1*S3 + Erx*gamma1*gamma3 + S3_x*(D26*S1_y + D66*S1_x) + S3_y*(D22*S1_y + D26*S1_x))
            KC0e0316 = -detJ*wij*(A45*S1*S3 + S3_x*(D12*S1_y + D16*S1_x) + S3_y*(D26*S1_y + D66*S1_x))
            KC0e0318 = -detJ*wij*(S4_x*(B12*S1_y + B16*S1_x) + S4_y*(B26*S1_y + B66*S1_x))
            KC0e0319 = -detJ*wij*(S4_x*(B26*S1_y + B66*S1_x) + S4_y*(B22*S1_y + B26*S1_x))
            KC0e0320 = -S1*detJ*wij*(A44*S4_y + A45*S4_x)
            KC0e0321 = detJ*wij*(A44*S1*S4 + Erx*gamma1*gamma4 + S4_x*(D26*S1_y + D66*S1_x) + S4_y*(D22*S1_y + D26*S1_x))
            KC0e0322 = -detJ*wij*(A45*S1*S4 + S4_x*(D12*S1_y + D16*S1_x) + S4_y*(D26*S1_y + D66*S1_x))
            KC0e0404 = detJ*wij*(A55*S1**2 + Ery*gamma1**2 + S1_x*(D11*S1_x + D16*S1_y) + S1_y*(D16*S1_x + D66*S1_y))
            KC0e0406 = detJ*wij*(S2_x*(B11*S1_x + B16*S1_y) + S2_y*(B16*S1_x + B66*S1_y))
            KC0e0407 = detJ*wij*(S2_x*(B16*S1_x + B66*S1_y) + S2_y*(B12*S1_x + B26*S1_y))
            KC0e0408 = S1*detJ*wij*(A45*S2_y + A55*S2_x)
            KC0e0409 = -detJ*wij*(A45*S1*S2 + S2_x*(D16*S1_x + D66*S1_y) + S2_y*(D12*S1_x + D26*S1_y))
            KC0e0410 = detJ*wij*(A55*S1*S2 + Ery*gamma1*gamma2 + S2_x*(D11*S1_x + D16*S1_y) + S2_y*(D16*S1_x + D66*S1_y))
            KC0e0412 = detJ*wij*(S3_x*(B11*S1_x + B16*S1_y) + S3_y*(B16*S1_x + B66*S1_y))
            KC0e0413 = detJ*wij*(S3_x*(B16*S1_x + B66*S1_y) + S3_y*(B12*S1_x + B26*S1_y))
            KC0e0414 = S1*detJ*wij*(A45*S3_y + A55*S3_x)
            KC0e0415 = -detJ*wij*(A45*S1*S3 + S3_x*(D16*S1_x + D66*S1_y) + S3_y*(D12*S1_x + D26*S1_y))
            KC0e0416 = detJ*wij*(A55*S1*S3 + Ery*gamma1*gamma3 + S3_x*(D11*S1_x + D16*S1_y) + S3_y*(D16*S1_x + D66*S1_y))
            KC0e0418 = detJ*wij*(S4_x*(B11*S1_x + B16*S1_y) + S4_y*(B16*S1_x + B66*S1_y))
            KC0e0419 = detJ*wij*(S4_x*(B16*S1_x + B66*S1_y) + S4_y*(B12*S1_x + B26*S1_y))
            KC0e0420 = S1*detJ*wij*(A45*S4_y + A55*S4_x)
            KC0e0421 = -detJ*wij*(A45*S1*S4 + S4_x*(D16*S1_x + D66*S1_y) + S4_y*(D12*S1_x + D26*S1_y))
            KC0e0422 = detJ*wij*(A55*S1*S4 + Ery*gamma1*gamma4 + S4_x*(D11*S1_x + D16*S1_y) + S4_y*(D16*S1_x + D66*S1_y))
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
            KC0e0808 = detJ*wij*(Ew*gamma2**2 + S2_x*(A45*S2_y + A55*S2_x) + S2_y*(A44*S2_y + A45*S2_x))
            KC0e0809 = -S2*detJ*wij*(A44*S2_y + A45*S2_x)
            KC0e0810 = S2*detJ*wij*(A45*S2_y + A55*S2_x)
            KC0e0814 = detJ*wij*(Ew*gamma2*gamma3 + S3_x*(A45*S2_y + A55*S2_x) + S3_y*(A44*S2_y + A45*S2_x))
            KC0e0815 = -S3*detJ*wij*(A44*S2_y + A45*S2_x)
            KC0e0816 = S3*detJ*wij*(A45*S2_y + A55*S2_x)
            KC0e0820 = detJ*wij*(Ew*gamma2*gamma4 + S4_x*(A45*S2_y + A55*S2_x) + S4_y*(A44*S2_y + A45*S2_x))
            KC0e0821 = -S4*detJ*wij*(A44*S2_y + A45*S2_x)
            KC0e0822 = S4*detJ*wij*(A45*S2_y + A55*S2_x)
            KC0e0909 = detJ*wij*(A44*S2**2 + Erx*gamma2**2 + S2_x*(D26*S2_y + D66*S2_x) + S2_y*(D22*S2_y + D26*S2_x))
            KC0e0910 = -detJ*wij*(A45*S2**2 + S2_x*(D12*S2_y + D16*S2_x) + S2_y*(D26*S2_y + D66*S2_x))
            KC0e0912 = -detJ*wij*(S3_x*(B12*S2_y + B16*S2_x) + S3_y*(B26*S2_y + B66*S2_x))
            KC0e0913 = -detJ*wij*(S3_x*(B26*S2_y + B66*S2_x) + S3_y*(B22*S2_y + B26*S2_x))
            KC0e0914 = -S2*detJ*wij*(A44*S3_y + A45*S3_x)
            KC0e0915 = detJ*wij*(A44*S2*S3 + Erx*gamma2*gamma3 + S3_x*(D26*S2_y + D66*S2_x) + S3_y*(D22*S2_y + D26*S2_x))
            KC0e0916 = -detJ*wij*(A45*S2*S3 + S3_x*(D12*S2_y + D16*S2_x) + S3_y*(D26*S2_y + D66*S2_x))
            KC0e0918 = -detJ*wij*(S4_x*(B12*S2_y + B16*S2_x) + S4_y*(B26*S2_y + B66*S2_x))
            KC0e0919 = -detJ*wij*(S4_x*(B26*S2_y + B66*S2_x) + S4_y*(B22*S2_y + B26*S2_x))
            KC0e0920 = -S2*detJ*wij*(A44*S4_y + A45*S4_x)
            KC0e0921 = detJ*wij*(A44*S2*S4 + Erx*gamma2*gamma4 + S4_x*(D26*S2_y + D66*S2_x) + S4_y*(D22*S2_y + D26*S2_x))
            KC0e0922 = -detJ*wij*(A45*S2*S4 + S4_x*(D12*S2_y + D16*S2_x) + S4_y*(D26*S2_y + D66*S2_x))
            KC0e1010 = detJ*wij*(A55*S2**2 + Ery*gamma2**2 + S2_x*(D11*S2_x + D16*S2_y) + S2_y*(D16*S2_x + D66*S2_y))
            KC0e1012 = detJ*wij*(S3_x*(B11*S2_x + B16*S2_y) + S3_y*(B16*S2_x + B66*S2_y))
            KC0e1013 = detJ*wij*(S3_x*(B16*S2_x + B66*S2_y) + S3_y*(B12*S2_x + B26*S2_y))
            KC0e1014 = S2*detJ*wij*(A45*S3_y + A55*S3_x)
            KC0e1015 = -detJ*wij*(A45*S2*S3 + S3_x*(D16*S2_x + D66*S2_y) + S3_y*(D12*S2_x + D26*S2_y))
            KC0e1016 = detJ*wij*(A55*S2*S3 + Ery*gamma2*gamma3 + S3_x*(D11*S2_x + D16*S2_y) + S3_y*(D16*S2_x + D66*S2_y))
            KC0e1018 = detJ*wij*(S4_x*(B11*S2_x + B16*S2_y) + S4_y*(B16*S2_x + B66*S2_y))
            KC0e1019 = detJ*wij*(S4_x*(B16*S2_x + B66*S2_y) + S4_y*(B12*S2_x + B26*S2_y))
            KC0e1020 = S2*detJ*wij*(A45*S4_y + A55*S4_x)
            KC0e1021 = -detJ*wij*(A45*S2*S4 + S4_x*(D16*S2_x + D66*S2_y) + S4_y*(D12*S2_x + D26*S2_y))
            KC0e1022 = detJ*wij*(A55*S2*S4 + Ery*gamma2*gamma4 + S4_x*(D11*S2_x + D16*S2_y) + S4_y*(D16*S2_x + D66*S2_y))
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
            KC0e1414 = detJ*wij*(Ew*gamma3**2 + S3_x*(A45*S3_y + A55*S3_x) + S3_y*(A44*S3_y + A45*S3_x))
            KC0e1415 = -S3*detJ*wij*(A44*S3_y + A45*S3_x)
            KC0e1416 = S3*detJ*wij*(A45*S3_y + A55*S3_x)
            KC0e1420 = detJ*wij*(Ew*gamma3*gamma4 + S4_x*(A45*S3_y + A55*S3_x) + S4_y*(A44*S3_y + A45*S3_x))
            KC0e1421 = -S4*detJ*wij*(A44*S3_y + A45*S3_x)
            KC0e1422 = S4*detJ*wij*(A45*S3_y + A55*S3_x)
            KC0e1515 = detJ*wij*(A44*S3**2 + Erx*gamma3**2 + S3_x*(D26*S3_y + D66*S3_x) + S3_y*(D22*S3_y + D26*S3_x))
            KC0e1516 = -detJ*wij*(A45*S3**2 + S3_x*(D12*S3_y + D16*S3_x) + S3_y*(D26*S3_y + D66*S3_x))
            KC0e1518 = -detJ*wij*(S4_x*(B12*S3_y + B16*S3_x) + S4_y*(B26*S3_y + B66*S3_x))
            KC0e1519 = -detJ*wij*(S4_x*(B26*S3_y + B66*S3_x) + S4_y*(B22*S3_y + B26*S3_x))
            KC0e1520 = -S3*detJ*wij*(A44*S4_y + A45*S4_x)
            KC0e1521 = detJ*wij*(A44*S3*S4 + Erx*gamma3*gamma4 + S4_x*(D26*S3_y + D66*S3_x) + S4_y*(D22*S3_y + D26*S3_x))
            KC0e1522 = -detJ*wij*(A45*S3*S4 + S4_x*(D12*S3_y + D16*S3_x) + S4_y*(D26*S3_y + D66*S3_x))
            KC0e1616 = detJ*wij*(A55*S3**2 + Ery*gamma3**2 + S3_x*(D11*S3_x + D16*S3_y) + S3_y*(D16*S3_x + D66*S3_y))
            KC0e1618 = detJ*wij*(S4_x*(B11*S3_x + B16*S3_y) + S4_y*(B16*S3_x + B66*S3_y))
            KC0e1619 = detJ*wij*(S4_x*(B16*S3_x + B66*S3_y) + S4_y*(B12*S3_x + B26*S3_y))
            KC0e1620 = S3*detJ*wij*(A45*S4_y + A55*S4_x)
            KC0e1621 = -detJ*wij*(A45*S3*S4 + S4_x*(D16*S3_x + D66*S3_y) + S4_y*(D12*S3_x + D26*S3_y))
            KC0e1622 = detJ*wij*(A55*S3*S4 + Ery*gamma3*gamma4 + S4_x*(D11*S3_x + D16*S3_y) + S4_y*(D16*S3_x + D66*S3_y))
            KC0e1818 = detJ*wij*(Eu*gamma4**2 + S4_x*(A11*S4_x + A16*S4_y) + S4_y*(A16*S4_x + A66*S4_y))
            KC0e1819 = detJ*wij*(S4_x*(A16*S4_x + A66*S4_y) + S4_y*(A12*S4_x + A26*S4_y))
            KC0e1821 = -detJ*wij*(S4_x*(B16*S4_x + B66*S4_y) + S4_y*(B12*S4_x + B26*S4_y))
            KC0e1822 = detJ*wij*(S4_x*(B11*S4_x + B16*S4_y) + S4_y*(B16*S4_x + B66*S4_y))
            KC0e1919 = detJ*wij*(Ev*gamma4**2 + S4_x*(A26*S4_y + A66*S4_x) + S4_y*(A22*S4_y + A26*S4_x))
            KC0e1921 = -detJ*wij*(S4_x*(B26*S4_y + B66*S4_x) + S4_y*(B22*S4_y + B26*S4_x))
            KC0e1922 = detJ*wij*(S4_x*(B12*S4_y + B16*S4_x) + S4_y*(B26*S4_y + B66*S4_x))
            KC0e2020 = detJ*wij*(Ew*gamma4**2 + S4_x*(A45*S4_y + A55*S4_x) + S4_y*(A44*S4_y + A45*S4_x))
            KC0e2021 = -S4*detJ*wij*(A44*S4_y + A45*S4_x)
            KC0e2022 = S4*detJ*wij*(A45*S4_y + A55*S4_x)
            KC0e2121 = detJ*wij*(A44*S4**2 + Erx*gamma4**2 + S4_x*(D26*S4_y + D66*S4_x) + S4_y*(D22*S4_y + D26*S4_x))
            KC0e2122 = -detJ*wij*(A45*S4**2 + S4_x*(D12*S4_y + D16*S4_x) + S4_y*(D26*S4_y + D66*S4_x))
            KC0e2222 = detJ*wij*(A55*S4**2 + Ery*gamma4**2 + S4_x*(D11*S4_x + D16*S4_y) + S4_y*(D16*S4_x + D66*S4_y))

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

            if nonlinear:
                self._update_probe_finte_nonlinear(prop)


    cpdef void update_KC0(Quad4R self,
                          long [::1] KC0r,
                          long [::1] KC0c,
                          double [::1] KC0v,
                          ShellProp prop,
                          int update_KC0v_only=0,
                          double hgfactor_u = 1.,
                          double hgfactor_v = 1.,
                          double hgfactor_w = 1.,
                          double hgfactor_rx = 1.,
                          double hgfactor_ry = 1.,
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
        hgfactor_u, hgfactor_v, hgfactor_w, hgfactor_rx, hgfactor_ry : double
            These offer the possibility to change the default hourglass
            stiffnesses for each degree-of-freedom.
            Dimensionless multipliers of the hourglass stiffnesses of ``u``,
            ``v``, ``w``, ``r_x`` and ``r_y``. Default ``1.``.

        Notes
        -----
        The element matrix in element coordinates is the Hessian of the strain
        energy of the Notes of :mod:`pyfe3d.quad4r`. The membrane, coupling,
        bending and transverse shear terms, `\boldsymbol{B}^\top \boldsymbol{C} \boldsymbol{B}` with the
        constitutive matrices `\boldsymbol{C}` of
        :meth:`pyfe3d.shellprop.ShellProp.get_constitutive_element` rotated
        with ``m11, m12, m21, m22``, are evaluated at the centroid,
        `\xi = \eta = 0`, with weight `w_{ij} = 4`, `4 \det J` being the
        element area `A`. The hourglass term adds, for each of `u`, `v`, `w`,
        `r_x` and `r_y`,

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

            E^{(h)}_u = 0.1 \, f_u E_{1eq} t A, \qquad
            E^{(h)}_v = 0.1 \, f_v E_{2eq} t A

        .. math::

            E^{(h)}_{r_x} = 0.1 \, f_{r_x} E_{2b} t^3 A, \qquad
            E^{(h)}_{r_y} = 0.1 \, f_{r_y} E_{1b} t^3 A

        .. math::

            E^{(h)}_w = \frac{0.05 \, f_w (E_{1b} + E_{2b}) t^3}{1 + 1/A}

        where `f_u, f_v, f_w, f_{r_x}, f_{r_y}` are the ``hgfactor_*``
        parameters. The reasons for the area factors are given in the Notes
        of :mod:`pyfe3d.quad4r`. The equivalent moduli, all in the element
        coordinate system, are

        .. math::

            E_{1eq} = \frac{1}{t a_{11}}, \quad
            E_{2eq} = \frac{1}{t a_{22}}, \quad
            E_{1b} = \frac{12}{t^3 d_{11}}, \quad
            E_{2b} = \frac{12}{t^3 d_{22}}

        with `[a] = \boldsymbol{A}^{-1}`, the coupling matrix `\boldsymbol{B}` being neglected,
        and `[d] = (\boldsymbol{D} - \boldsymbol{B} \boldsymbol{A}^{-1} \boldsymbol{B})^{-1}` the lower-right block of
        `[[A, B], [B, D]]^{-1}`. A homogeneous plate of modulus `E` has
        `a_{11} = 1/(E t)` and `d_{11} = 12/(E t^3)`, so `E_{1eq}` is the
        modulus of the homogeneous plate that stretches as the laminate under
        `N_{xx}` alone with the curvatures restrained, and `E_{1b}` that of
        the homogeneous plate that bends as the laminate under `M_{xx}`
        alone with the membrane forces zero; analogously in `y`. Both pairs
        reduce to `E` for an isotropic plate, as in Eq. (16) of Brockman,
        whereas `12 D_{11}/t^3` would give `E/(1 - \nu^2)`. `\boldsymbol{D} -
        \boldsymbol{B} \boldsymbol{A}^{-1} \boldsymbol{B}` and `\boldsymbol{A}` do not change when the reference surface
        is moved with ``prop.offset``, while `\boldsymbol{D}` does, so neither pair
        depends on the offset. `r_x` enters the curvature `\kappa_{yy} =
        -r_{x,y}` and `r_y` the curvature `\kappa_{xx} = r_{y,x}`, hence
        `E_{2b}` in `E^{(h)}_{r_x}` and `E_{1b}` in `E^{(h)}_{r_y}`. The
        hourglass pattern of `w`, `w = xy` on a rectangle, is a twist, and
        the mean of the two bending moduli in `E^{(h)}_w` generalises
        Eq. (16b) of Brockman to laminates and reduces to it for an
        isotropic plate.

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

        References
        ----------
        .. [1] Brockman, R. A., Dynamics of the Bilinear Mindlin Plate Element,
           International Journal for Numerical Methods in Engineering, 24(12),
           2343-2356, 1987. https://doi.org/10.1002/nme.1620241208

        """
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
        cdef double E1eq, E2eq, E1b, E2b
        cdef double K6ROT
        cdef double Eu, Ev, Erx, Ery, Ew
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
            # NOTE equivalent bending moduli of the hourglass control of w,
            #      rx and ry, from the bending compliance of the laminate in
            #      the element coordinate system, see the Notes of update_KC0
            _hourglass_bending_moduli(prop.h, Ae, Be, De, &E1b, &E2b)


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

            # NOTE Brockman's hourglass control is derived for isotropic
            #      plates, and applied to laminates with the membrane moduli
            #      E1eq and E2eq for u and v, and with the bending moduli
            #      E1b and E2b for w, rx and ry, see the module documentation
            # NOTE Brockman's 1/(1 + 1/A) of Eq. (16) is replaced by the
            #      dimensionally homogeneous area factor, see the section
            #      "Dimensional homogeneity of the hourglass stiffnesses"
            #      of the module documentation. In short, the term enters
            #      as K_ij = A*E^(h)*gamma_i*gamma_j with gamma of
            #      dimension 1/L**2, so E^(h) must be F*L for the
            #      translations u, v and w and F*L**3 for the rotations
            #      rx and ry. Brockman's factor tends to A only in the
            #      limit A << 1, so it supplies the missing area in some
            #      units and not in others
            Eu = hgfactor_u*0.1*E1eq*prop.h*self.area
            Ev = hgfactor_v*0.1*E2eq*prop.h*self.area
            Erx = hgfactor_rx*0.1*E2b*prop.h**3*self.area
            Ery = hgfactor_ry*0.1*E1b*prop.h**3*self.area
            # NOTE w alone keeps Brockman's 1/(1 + 1/A), so this is the
            #      one coefficient of the scheme that remains unit
            #      dependent, and that is deliberate. The operator cannot
            #      separate the spurious pattern w = x*y from a legitimate
            #      twist curvature, both giving gamma'w = d2w/dxdy, so the
            #      amount of stabilisation wanted is a property of the
            #      problem and not of the element, and no dimensionless
            #      constant reproduces it: the thin plates want about
            #      2e-4 of E*h**3 and the one-element-wide torsion strip of
            #      MacNeal and Harder wants about 2e-2, two orders of
            #      magnitude apart. Brockman's factor happens to supply
            #      both, because it grows with the element area, which in
            #      SI units correlates with what each of those problems
            #      needs. See the module documentation, and
            #      test_quad4r_hourglass_control.py for the measurements.
            #      The mean of the two bending moduli generalises
            #      Eq. (16b) to laminates and reduces to it for an
            #      isotropic plate. It is built from the moduli and not
            #      from Erx and Ery, which already carry hgfactor_rx and
            #      hgfactor_ry, so that hgfactor_w scales this term and
            #      nothing else
            Ew = (hgfactor_w*0.05*(E1b + E2b)*prop.h**3
                  /(1.0 + 1.0/self.area))

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
            KC0e0202 = detJ*wij*(Ew*gamma1**2 + S1_x*(A45*S1_y + A55*S1_x) + S1_y*(A44*S1_y + A45*S1_x))
            KC0e0203 = -S1*detJ*wij*(A44*S1_y + A45*S1_x)
            KC0e0204 = S1*detJ*wij*(A45*S1_y + A55*S1_x)
            KC0e0208 = detJ*wij*(Ew*gamma1*gamma2 + S2_x*(A45*S1_y + A55*S1_x) + S2_y*(A44*S1_y + A45*S1_x))
            KC0e0209 = -S2*detJ*wij*(A44*S1_y + A45*S1_x)
            KC0e0210 = S2*detJ*wij*(A45*S1_y + A55*S1_x)
            KC0e0214 = detJ*wij*(Ew*gamma1*gamma3 + S3_x*(A45*S1_y + A55*S1_x) + S3_y*(A44*S1_y + A45*S1_x))
            KC0e0215 = -S3*detJ*wij*(A44*S1_y + A45*S1_x)
            KC0e0216 = S3*detJ*wij*(A45*S1_y + A55*S1_x)
            KC0e0220 = detJ*wij*(Ew*gamma1*gamma4 + S4_x*(A45*S1_y + A55*S1_x) + S4_y*(A44*S1_y + A45*S1_x))
            KC0e0221 = -S4*detJ*wij*(A44*S1_y + A45*S1_x)
            KC0e0222 = S4*detJ*wij*(A45*S1_y + A55*S1_x)
            KC0e0303 = detJ*wij*(A44*S1**2 + Erx*gamma1**2 + S1_x*(D26*S1_y + D66*S1_x) + S1_y*(D22*S1_y + D26*S1_x))
            KC0e0304 = -detJ*wij*(A45*S1**2 + S1_x*(D12*S1_y + D16*S1_x) + S1_y*(D26*S1_y + D66*S1_x))
            KC0e0306 = -detJ*wij*(S2_x*(B12*S1_y + B16*S1_x) + S2_y*(B26*S1_y + B66*S1_x))
            KC0e0307 = -detJ*wij*(S2_x*(B26*S1_y + B66*S1_x) + S2_y*(B22*S1_y + B26*S1_x))
            KC0e0308 = -S1*detJ*wij*(A44*S2_y + A45*S2_x)
            KC0e0309 = detJ*wij*(A44*S1*S2 + Erx*gamma1*gamma2 + S2_x*(D26*S1_y + D66*S1_x) + S2_y*(D22*S1_y + D26*S1_x))
            KC0e0310 = -detJ*wij*(A45*S1*S2 + S2_x*(D12*S1_y + D16*S1_x) + S2_y*(D26*S1_y + D66*S1_x))
            KC0e0312 = -detJ*wij*(S3_x*(B12*S1_y + B16*S1_x) + S3_y*(B26*S1_y + B66*S1_x))
            KC0e0313 = -detJ*wij*(S3_x*(B26*S1_y + B66*S1_x) + S3_y*(B22*S1_y + B26*S1_x))
            KC0e0314 = -S1*detJ*wij*(A44*S3_y + A45*S3_x)
            KC0e0315 = detJ*wij*(A44*S1*S3 + Erx*gamma1*gamma3 + S3_x*(D26*S1_y + D66*S1_x) + S3_y*(D22*S1_y + D26*S1_x))
            KC0e0316 = -detJ*wij*(A45*S1*S3 + S3_x*(D12*S1_y + D16*S1_x) + S3_y*(D26*S1_y + D66*S1_x))
            KC0e0318 = -detJ*wij*(S4_x*(B12*S1_y + B16*S1_x) + S4_y*(B26*S1_y + B66*S1_x))
            KC0e0319 = -detJ*wij*(S4_x*(B26*S1_y + B66*S1_x) + S4_y*(B22*S1_y + B26*S1_x))
            KC0e0320 = -S1*detJ*wij*(A44*S4_y + A45*S4_x)
            KC0e0321 = detJ*wij*(A44*S1*S4 + Erx*gamma1*gamma4 + S4_x*(D26*S1_y + D66*S1_x) + S4_y*(D22*S1_y + D26*S1_x))
            KC0e0322 = -detJ*wij*(A45*S1*S4 + S4_x*(D12*S1_y + D16*S1_x) + S4_y*(D26*S1_y + D66*S1_x))
            KC0e0404 = detJ*wij*(A55*S1**2 + Ery*gamma1**2 + S1_x*(D11*S1_x + D16*S1_y) + S1_y*(D16*S1_x + D66*S1_y))
            KC0e0406 = detJ*wij*(S2_x*(B11*S1_x + B16*S1_y) + S2_y*(B16*S1_x + B66*S1_y))
            KC0e0407 = detJ*wij*(S2_x*(B16*S1_x + B66*S1_y) + S2_y*(B12*S1_x + B26*S1_y))
            KC0e0408 = S1*detJ*wij*(A45*S2_y + A55*S2_x)
            KC0e0409 = -detJ*wij*(A45*S1*S2 + S2_x*(D16*S1_x + D66*S1_y) + S2_y*(D12*S1_x + D26*S1_y))
            KC0e0410 = detJ*wij*(A55*S1*S2 + Ery*gamma1*gamma2 + S2_x*(D11*S1_x + D16*S1_y) + S2_y*(D16*S1_x + D66*S1_y))
            KC0e0412 = detJ*wij*(S3_x*(B11*S1_x + B16*S1_y) + S3_y*(B16*S1_x + B66*S1_y))
            KC0e0413 = detJ*wij*(S3_x*(B16*S1_x + B66*S1_y) + S3_y*(B12*S1_x + B26*S1_y))
            KC0e0414 = S1*detJ*wij*(A45*S3_y + A55*S3_x)
            KC0e0415 = -detJ*wij*(A45*S1*S3 + S3_x*(D16*S1_x + D66*S1_y) + S3_y*(D12*S1_x + D26*S1_y))
            KC0e0416 = detJ*wij*(A55*S1*S3 + Ery*gamma1*gamma3 + S3_x*(D11*S1_x + D16*S1_y) + S3_y*(D16*S1_x + D66*S1_y))
            KC0e0418 = detJ*wij*(S4_x*(B11*S1_x + B16*S1_y) + S4_y*(B16*S1_x + B66*S1_y))
            KC0e0419 = detJ*wij*(S4_x*(B16*S1_x + B66*S1_y) + S4_y*(B12*S1_x + B26*S1_y))
            KC0e0420 = S1*detJ*wij*(A45*S4_y + A55*S4_x)
            KC0e0421 = -detJ*wij*(A45*S1*S4 + S4_x*(D16*S1_x + D66*S1_y) + S4_y*(D12*S1_x + D26*S1_y))
            KC0e0422 = detJ*wij*(A55*S1*S4 + Ery*gamma1*gamma4 + S4_x*(D11*S1_x + D16*S1_y) + S4_y*(D16*S1_x + D66*S1_y))
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
            KC0e0808 = detJ*wij*(Ew*gamma2**2 + S2_x*(A45*S2_y + A55*S2_x) + S2_y*(A44*S2_y + A45*S2_x))
            KC0e0809 = -S2*detJ*wij*(A44*S2_y + A45*S2_x)
            KC0e0810 = S2*detJ*wij*(A45*S2_y + A55*S2_x)
            KC0e0814 = detJ*wij*(Ew*gamma2*gamma3 + S3_x*(A45*S2_y + A55*S2_x) + S3_y*(A44*S2_y + A45*S2_x))
            KC0e0815 = -S3*detJ*wij*(A44*S2_y + A45*S2_x)
            KC0e0816 = S3*detJ*wij*(A45*S2_y + A55*S2_x)
            KC0e0820 = detJ*wij*(Ew*gamma2*gamma4 + S4_x*(A45*S2_y + A55*S2_x) + S4_y*(A44*S2_y + A45*S2_x))
            KC0e0821 = -S4*detJ*wij*(A44*S2_y + A45*S2_x)
            KC0e0822 = S4*detJ*wij*(A45*S2_y + A55*S2_x)
            KC0e0909 = detJ*wij*(A44*S2**2 + Erx*gamma2**2 + S2_x*(D26*S2_y + D66*S2_x) + S2_y*(D22*S2_y + D26*S2_x))
            KC0e0910 = -detJ*wij*(A45*S2**2 + S2_x*(D12*S2_y + D16*S2_x) + S2_y*(D26*S2_y + D66*S2_x))
            KC0e0912 = -detJ*wij*(S3_x*(B12*S2_y + B16*S2_x) + S3_y*(B26*S2_y + B66*S2_x))
            KC0e0913 = -detJ*wij*(S3_x*(B26*S2_y + B66*S2_x) + S3_y*(B22*S2_y + B26*S2_x))
            KC0e0914 = -S2*detJ*wij*(A44*S3_y + A45*S3_x)
            KC0e0915 = detJ*wij*(A44*S2*S3 + Erx*gamma2*gamma3 + S3_x*(D26*S2_y + D66*S2_x) + S3_y*(D22*S2_y + D26*S2_x))
            KC0e0916 = -detJ*wij*(A45*S2*S3 + S3_x*(D12*S2_y + D16*S2_x) + S3_y*(D26*S2_y + D66*S2_x))
            KC0e0918 = -detJ*wij*(S4_x*(B12*S2_y + B16*S2_x) + S4_y*(B26*S2_y + B66*S2_x))
            KC0e0919 = -detJ*wij*(S4_x*(B26*S2_y + B66*S2_x) + S4_y*(B22*S2_y + B26*S2_x))
            KC0e0920 = -S2*detJ*wij*(A44*S4_y + A45*S4_x)
            KC0e0921 = detJ*wij*(A44*S2*S4 + Erx*gamma2*gamma4 + S4_x*(D26*S2_y + D66*S2_x) + S4_y*(D22*S2_y + D26*S2_x))
            KC0e0922 = -detJ*wij*(A45*S2*S4 + S4_x*(D12*S2_y + D16*S2_x) + S4_y*(D26*S2_y + D66*S2_x))
            KC0e1010 = detJ*wij*(A55*S2**2 + Ery*gamma2**2 + S2_x*(D11*S2_x + D16*S2_y) + S2_y*(D16*S2_x + D66*S2_y))
            KC0e1012 = detJ*wij*(S3_x*(B11*S2_x + B16*S2_y) + S3_y*(B16*S2_x + B66*S2_y))
            KC0e1013 = detJ*wij*(S3_x*(B16*S2_x + B66*S2_y) + S3_y*(B12*S2_x + B26*S2_y))
            KC0e1014 = S2*detJ*wij*(A45*S3_y + A55*S3_x)
            KC0e1015 = -detJ*wij*(A45*S2*S3 + S3_x*(D16*S2_x + D66*S2_y) + S3_y*(D12*S2_x + D26*S2_y))
            KC0e1016 = detJ*wij*(A55*S2*S3 + Ery*gamma2*gamma3 + S3_x*(D11*S2_x + D16*S2_y) + S3_y*(D16*S2_x + D66*S2_y))
            KC0e1018 = detJ*wij*(S4_x*(B11*S2_x + B16*S2_y) + S4_y*(B16*S2_x + B66*S2_y))
            KC0e1019 = detJ*wij*(S4_x*(B16*S2_x + B66*S2_y) + S4_y*(B12*S2_x + B26*S2_y))
            KC0e1020 = S2*detJ*wij*(A45*S4_y + A55*S4_x)
            KC0e1021 = -detJ*wij*(A45*S2*S4 + S4_x*(D16*S2_x + D66*S2_y) + S4_y*(D12*S2_x + D26*S2_y))
            KC0e1022 = detJ*wij*(A55*S2*S4 + Ery*gamma2*gamma4 + S4_x*(D11*S2_x + D16*S2_y) + S4_y*(D16*S2_x + D66*S2_y))
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
            KC0e1414 = detJ*wij*(Ew*gamma3**2 + S3_x*(A45*S3_y + A55*S3_x) + S3_y*(A44*S3_y + A45*S3_x))
            KC0e1415 = -S3*detJ*wij*(A44*S3_y + A45*S3_x)
            KC0e1416 = S3*detJ*wij*(A45*S3_y + A55*S3_x)
            KC0e1420 = detJ*wij*(Ew*gamma3*gamma4 + S4_x*(A45*S3_y + A55*S3_x) + S4_y*(A44*S3_y + A45*S3_x))
            KC0e1421 = -S4*detJ*wij*(A44*S3_y + A45*S3_x)
            KC0e1422 = S4*detJ*wij*(A45*S3_y + A55*S3_x)
            KC0e1515 = detJ*wij*(A44*S3**2 + Erx*gamma3**2 + S3_x*(D26*S3_y + D66*S3_x) + S3_y*(D22*S3_y + D26*S3_x))
            KC0e1516 = -detJ*wij*(A45*S3**2 + S3_x*(D12*S3_y + D16*S3_x) + S3_y*(D26*S3_y + D66*S3_x))
            KC0e1518 = -detJ*wij*(S4_x*(B12*S3_y + B16*S3_x) + S4_y*(B26*S3_y + B66*S3_x))
            KC0e1519 = -detJ*wij*(S4_x*(B26*S3_y + B66*S3_x) + S4_y*(B22*S3_y + B26*S3_x))
            KC0e1520 = -S3*detJ*wij*(A44*S4_y + A45*S4_x)
            KC0e1521 = detJ*wij*(A44*S3*S4 + Erx*gamma3*gamma4 + S4_x*(D26*S3_y + D66*S3_x) + S4_y*(D22*S3_y + D26*S3_x))
            KC0e1522 = -detJ*wij*(A45*S3*S4 + S4_x*(D12*S3_y + D16*S3_x) + S4_y*(D26*S3_y + D66*S3_x))
            KC0e1616 = detJ*wij*(A55*S3**2 + Ery*gamma3**2 + S3_x*(D11*S3_x + D16*S3_y) + S3_y*(D16*S3_x + D66*S3_y))
            KC0e1618 = detJ*wij*(S4_x*(B11*S3_x + B16*S3_y) + S4_y*(B16*S3_x + B66*S3_y))
            KC0e1619 = detJ*wij*(S4_x*(B16*S3_x + B66*S3_y) + S4_y*(B12*S3_x + B26*S3_y))
            KC0e1620 = S3*detJ*wij*(A45*S4_y + A55*S4_x)
            KC0e1621 = -detJ*wij*(A45*S3*S4 + S4_x*(D16*S3_x + D66*S3_y) + S4_y*(D12*S3_x + D26*S3_y))
            KC0e1622 = detJ*wij*(A55*S3*S4 + Ery*gamma3*gamma4 + S4_x*(D11*S3_x + D16*S3_y) + S4_y*(D16*S3_x + D66*S3_y))
            KC0e1818 = detJ*wij*(Eu*gamma4**2 + S4_x*(A11*S4_x + A16*S4_y) + S4_y*(A16*S4_x + A66*S4_y))
            KC0e1819 = detJ*wij*(S4_x*(A16*S4_x + A66*S4_y) + S4_y*(A12*S4_x + A26*S4_y))
            KC0e1821 = -detJ*wij*(S4_x*(B16*S4_x + B66*S4_y) + S4_y*(B12*S4_x + B26*S4_y))
            KC0e1822 = detJ*wij*(S4_x*(B11*S4_x + B16*S4_y) + S4_y*(B16*S4_x + B66*S4_y))
            KC0e1919 = detJ*wij*(Ev*gamma4**2 + S4_x*(A26*S4_y + A66*S4_x) + S4_y*(A22*S4_y + A26*S4_x))
            KC0e1921 = -detJ*wij*(S4_x*(B26*S4_y + B66*S4_x) + S4_y*(B22*S4_y + B26*S4_x))
            KC0e1922 = detJ*wij*(S4_x*(B12*S4_y + B16*S4_x) + S4_y*(B26*S4_y + B66*S4_x))
            KC0e2020 = detJ*wij*(Ew*gamma4**2 + S4_x*(A45*S4_y + A55*S4_x) + S4_y*(A44*S4_y + A45*S4_x))
            KC0e2021 = -S4*detJ*wij*(A44*S4_y + A45*S4_x)
            KC0e2022 = S4*detJ*wij*(A45*S4_y + A55*S4_x)
            KC0e2121 = detJ*wij*(A44*S4**2 + Erx*gamma4**2 + S4_x*(D26*S4_y + D66*S4_x) + S4_y*(D22*S4_y + D26*S4_x))
            KC0e2122 = -detJ*wij*(A45*S4**2 + S4_x*(D12*S4_y + D16*S4_x) + S4_y*(D26*S4_y + D66*S4_x))
            KC0e2222 = detJ*wij*(A55*S4**2 + Ery*gamma4**2 + S4_x*(D11*S4_x + D16*S4_y) + S4_y*(D16*S4_x + D66*S4_y))

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


    cpdef void update_fint(Quad4R self,
                           double [::1] fint,
                           ShellProp prop,
                           double hgfactor_u = 1.,
                           double hgfactor_v = 1.,
                           double hgfactor_w = 1.,
                           double hgfactor_rx = 1.,
                           double hgfactor_ry = 1.,
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
        hgfactor_u, hgfactor_v, hgfactor_w, hgfactor_rx, hgfactor_ry : double
            These offer the possibility to change the default hourglass
            stiffnesses for each degree-of-freedom.
            Dimensionless multipliers of the hourglass stiffnesses of ``u``,
            ``v``, ``w``, ``r_x`` and ``r_y``. Default ``1.``.
        nonlinear : int, optional
            The default ``0`` gives the linear internal forces, ``KC0*u``. Any other
            value adds the geometrically nonlinear terms of the von Karman strains,
            whose tangent stiffness matrix is ``KC0 + KCNL + KG``, see
            :meth:`.update_KCNL`.

        Notes
        -----
        The internal forces of each node, forces and moments, are rotated
        from the element to the global coordinates with the rotation matrix
        ``r11, ..., r33``. With ``nonlinear``, ``KC0 + KCNL + KG`` is the
        exact Jacobian of the assembled internal forces, see
        :meth:`.update_probe_finte` and :meth:`.update_KCNL`.

        """
        cdef double *finte

        self.update_probe_finte(prop, hgfactor_u, hgfactor_v, hgfactor_w,
                                hgfactor_rx, hgfactor_ry, nonlinear)

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


    cdef void _update_probe_KCNLve(Quad4R self, ShellProp prop) noexcept nogil:
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


    cdef void _update_probe_finte_nonlinear(Quad4R self,
                                            ShellProp prop) noexcept nogil:
        r"""Add the geometrically nonlinear terms to the probe internal forces

        The attribute ``finte`` of the :class:`.Quad4RProbe` receives the terms
        of the nonlinear part of the von Karman membrane strains, evaluated at
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
                    finte[i] += wij*detJ*(
                        # Bm.T*NNL + Bb.T*MNL
                          BLexx[i]*NNL[0] + BLeyy[i]*NNL[1] + BLgxy[i]*NNL[2]
                        + BLkxx[i]*MNL[0] + BLkyy[i]*MNL[1] + BLkxy[i]*MNL[2]
                        # BNL.T*(N + NNL)
                        + w_x*Sw_x[i]*(N[0] + NNL[0])
                        + w_y*Sw_y[i]*(N[1] + NNL[1])
                        + (w_x*Sw_y[i] + w_y*Sw_x[i])*(N[2] + NNL[2])
                    )


    cpdef void update_KCNL(Quad4R self,
                           long [::1] KCNLr,
                           long [::1] KCNLc,
                           double [::1] KCNLv,
                           ShellProp prop,
                           int update_KCNLv_only=0
                           ):
        r"""Update sparse vectors for the nonlinear stiffness matrix KCNL

        KCNL collects the terms of the tangent stiffness matrix of the
        geometrically nonlinear analysis with the von Karman membrane strains
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

        Notes
        -----
        Assuming that KCNL = KC0L + KCL0 + KCLL + KGNL, that is `\boldsymbol{K}_{CNL} + \boldsymbol{K}_{GNL}`, the book's nonlinear constitutive stiffness matrix `\boldsymbol{K}_{CNL}` of the terms of `\boldsymbol{B}_{NL}` plus the geometric `\boldsymbol{K}_{GNL}`, built from the von Karman
        membrane strains

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

        """
        cdef int i, j, node_i, node_j, k, ke, m, n
        cdef int c[4]
        cdef double r[6][6]

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

            self._update_probe_KCNLve(prop)

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
                         int update_KGv_only=0
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
        stiffness in KC0 besides the artificial one of `E^{(h)}_w`, see
        :meth:`.update_KC0`. With the slopes at the two-point quadrature
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

        """
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

        """
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

        References
        ----------
        .. [1] Brockman, R. A., Dynamics of the Bilinear Mindlin Plate Element,
           International Journal for Numerical Methods in Engineering, 24(12),
           2343-2356, 1987. https://doi.org/10.1002/nme.1620241208

        """
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
                Mv[k] += h11*intrhoz*r11*r22 - h11*intrhoz*r12*r21
                k += 1
                Mv[k] += h11*intrhoz*r11*r32 - h11*intrhoz*r12*r31
                k += 1
                Mv[k] += h12*intrho*r11**2 + h12*intrho*r12**2 + h12*intrho*r13**2
                k += 1
                Mv[k] += h12*intrho*r11*r21 + h12*intrho*r12*r22 + h12*intrho*r13*r23
                k += 1
                Mv[k] += h12*intrho*r11*r31 + h12*intrho*r12*r32 + h12*intrho*r13*r33
                k += 1
                Mv[k] += h12*intrhoz*r11*r22 - h12*intrhoz*r12*r21
                k += 1
                Mv[k] += h12*intrhoz*r11*r32 - h12*intrhoz*r12*r31
                k += 1
                Mv[k] += h13*intrho*r11**2 + h13*intrho*r12**2 + h13*intrho*r13**2
                k += 1
                Mv[k] += h13*intrho*r11*r21 + h13*intrho*r12*r22 + h13*intrho*r13*r23
                k += 1
                Mv[k] += h13*intrho*r11*r31 + h13*intrho*r12*r32 + h13*intrho*r13*r33
                k += 1
                Mv[k] += h13*intrhoz*r11*r22 - h13*intrhoz*r12*r21
                k += 1
                Mv[k] += h13*intrhoz*r11*r32 - h13*intrhoz*r12*r31
                k += 1
                Mv[k] += h14*intrho*r11**2 + h14*intrho*r12**2 + h14*intrho*r13**2
                k += 1
                Mv[k] += h14*intrho*r11*r21 + h14*intrho*r12*r22 + h14*intrho*r13*r23
                k += 1
                Mv[k] += h14*intrho*r11*r31 + h14*intrho*r12*r32 + h14*intrho*r13*r33
                k += 1
                Mv[k] += h14*intrhoz*r11*r22 - h14*intrhoz*r12*r21
                k += 1
                Mv[k] += h14*intrhoz*r11*r32 - h14*intrhoz*r12*r31
                k += 1
                Mv[k] += h11*intrho*r11*r21 + h11*intrho*r12*r22 + h11*intrho*r13*r23
                k += 1
                Mv[k] += h11*intrho*r21**2 + h11*intrho*r22**2 + h11*intrho*r23**2
                k += 1
                Mv[k] += h11*intrho*r21*r31 + h11*intrho*r22*r32 + h11*intrho*r23*r33
                k += 1
                Mv[k] += -h11*intrhoz*r11*r22 + h11*intrhoz*r12*r21
                k += 1
                Mv[k] += h11*intrhoz*r21*r32 - h11*intrhoz*r22*r31
                k += 1
                Mv[k] += h12*intrho*r11*r21 + h12*intrho*r12*r22 + h12*intrho*r13*r23
                k += 1
                Mv[k] += h12*intrho*r21**2 + h12*intrho*r22**2 + h12*intrho*r23**2
                k += 1
                Mv[k] += h12*intrho*r21*r31 + h12*intrho*r22*r32 + h12*intrho*r23*r33
                k += 1
                Mv[k] += -h12*intrhoz*r11*r22 + h12*intrhoz*r12*r21
                k += 1
                Mv[k] += h12*intrhoz*r21*r32 - h12*intrhoz*r22*r31
                k += 1
                Mv[k] += h13*intrho*r11*r21 + h13*intrho*r12*r22 + h13*intrho*r13*r23
                k += 1
                Mv[k] += h13*intrho*r21**2 + h13*intrho*r22**2 + h13*intrho*r23**2
                k += 1
                Mv[k] += h13*intrho*r21*r31 + h13*intrho*r22*r32 + h13*intrho*r23*r33
                k += 1
                Mv[k] += -h13*intrhoz*r11*r22 + h13*intrhoz*r12*r21
                k += 1
                Mv[k] += h13*intrhoz*r21*r32 - h13*intrhoz*r22*r31
                k += 1
                Mv[k] += h14*intrho*r11*r21 + h14*intrho*r12*r22 + h14*intrho*r13*r23
                k += 1
                Mv[k] += h14*intrho*r21**2 + h14*intrho*r22**2 + h14*intrho*r23**2
                k += 1
                Mv[k] += h14*intrho*r21*r31 + h14*intrho*r22*r32 + h14*intrho*r23*r33
                k += 1
                Mv[k] += -h14*intrhoz*r11*r22 + h14*intrhoz*r12*r21
                k += 1
                Mv[k] += h14*intrhoz*r21*r32 - h14*intrhoz*r22*r31
                k += 1
                Mv[k] += h11*intrho*r11*r31 + h11*intrho*r12*r32 + h11*intrho*r13*r33
                k += 1
                Mv[k] += h11*intrho*r21*r31 + h11*intrho*r22*r32 + h11*intrho*r23*r33
                k += 1
                Mv[k] += h11*intrho*r31**2 + h11*intrho*r32**2 + h11*intrho*r33**2
                k += 1
                Mv[k] += -h11*intrhoz*r11*r32 + h11*intrhoz*r12*r31
                k += 1
                Mv[k] += -h11*intrhoz*r21*r32 + h11*intrhoz*r22*r31
                k += 1
                Mv[k] += h12*intrho*r11*r31 + h12*intrho*r12*r32 + h12*intrho*r13*r33
                k += 1
                Mv[k] += h12*intrho*r21*r31 + h12*intrho*r22*r32 + h12*intrho*r23*r33
                k += 1
                Mv[k] += h12*intrho*r31**2 + h12*intrho*r32**2 + h12*intrho*r33**2
                k += 1
                Mv[k] += -h12*intrhoz*r11*r32 + h12*intrhoz*r12*r31
                k += 1
                Mv[k] += -h12*intrhoz*r21*r32 + h12*intrhoz*r22*r31
                k += 1
                Mv[k] += h13*intrho*r11*r31 + h13*intrho*r12*r32 + h13*intrho*r13*r33
                k += 1
                Mv[k] += h13*intrho*r21*r31 + h13*intrho*r22*r32 + h13*intrho*r23*r33
                k += 1
                Mv[k] += h13*intrho*r31**2 + h13*intrho*r32**2 + h13*intrho*r33**2
                k += 1
                Mv[k] += -h13*intrhoz*r11*r32 + h13*intrhoz*r12*r31
                k += 1
                Mv[k] += -h13*intrhoz*r21*r32 + h13*intrhoz*r22*r31
                k += 1
                Mv[k] += h14*intrho*r11*r31 + h14*intrho*r12*r32 + h14*intrho*r13*r33
                k += 1
                Mv[k] += h14*intrho*r21*r31 + h14*intrho*r22*r32 + h14*intrho*r23*r33
                k += 1
                Mv[k] += h14*intrho*r31**2 + h14*intrho*r32**2 + h14*intrho*r33**2
                k += 1
                Mv[k] += -h14*intrhoz*r11*r32 + h14*intrhoz*r12*r31
                k += 1
                Mv[k] += -h14*intrhoz*r21*r32 + h14*intrhoz*r22*r31
                k += 1
                Mv[k] += -h11*intrhoz*r11*r22 + h11*intrhoz*r12*r21
                k += 1
                Mv[k] += -h11*intrhoz*r11*r32 + h11*intrhoz*r12*r31
                k += 1
                Mv[k] += h11*intrhoz2*r11**2 + h11*intrhoz2*r12**2
                k += 1
                Mv[k] += h11*intrhoz2*r11*r21 + h11*intrhoz2*r12*r22
                k += 1
                Mv[k] += h11*intrhoz2*r11*r31 + h11*intrhoz2*r12*r32
                k += 1
                Mv[k] += -h12*intrhoz*r11*r22 + h12*intrhoz*r12*r21
                k += 1
                Mv[k] += -h12*intrhoz*r11*r32 + h12*intrhoz*r12*r31
                k += 1
                Mv[k] += h12*intrhoz2*r11**2 + h12*intrhoz2*r12**2
                k += 1
                Mv[k] += h12*intrhoz2*r11*r21 + h12*intrhoz2*r12*r22
                k += 1
                Mv[k] += h12*intrhoz2*r11*r31 + h12*intrhoz2*r12*r32
                k += 1
                Mv[k] += -h13*intrhoz*r11*r22 + h13*intrhoz*r12*r21
                k += 1
                Mv[k] += -h13*intrhoz*r11*r32 + h13*intrhoz*r12*r31
                k += 1
                Mv[k] += h13*intrhoz2*r11**2 + h13*intrhoz2*r12**2
                k += 1
                Mv[k] += h13*intrhoz2*r11*r21 + h13*intrhoz2*r12*r22
                k += 1
                Mv[k] += h13*intrhoz2*r11*r31 + h13*intrhoz2*r12*r32
                k += 1
                Mv[k] += -h14*intrhoz*r11*r22 + h14*intrhoz*r12*r21
                k += 1
                Mv[k] += -h14*intrhoz*r11*r32 + h14*intrhoz*r12*r31
                k += 1
                Mv[k] += h14*intrhoz2*r11**2 + h14*intrhoz2*r12**2
                k += 1
                Mv[k] += h14*intrhoz2*r11*r21 + h14*intrhoz2*r12*r22
                k += 1
                Mv[k] += h14*intrhoz2*r11*r31 + h14*intrhoz2*r12*r32
                k += 1
                Mv[k] += h11*intrhoz*r11*r22 - h11*intrhoz*r12*r21
                k += 1
                Mv[k] += -h11*intrhoz*r21*r32 + h11*intrhoz*r22*r31
                k += 1
                Mv[k] += h11*intrhoz2*r11*r21 + h11*intrhoz2*r12*r22
                k += 1
                Mv[k] += h11*intrhoz2*r21**2 + h11*intrhoz2*r22**2
                k += 1
                Mv[k] += h11*intrhoz2*r21*r31 + h11*intrhoz2*r22*r32
                k += 1
                Mv[k] += h12*intrhoz*r11*r22 - h12*intrhoz*r12*r21
                k += 1
                Mv[k] += -h12*intrhoz*r21*r32 + h12*intrhoz*r22*r31
                k += 1
                Mv[k] += h12*intrhoz2*r11*r21 + h12*intrhoz2*r12*r22
                k += 1
                Mv[k] += h12*intrhoz2*r21**2 + h12*intrhoz2*r22**2
                k += 1
                Mv[k] += h12*intrhoz2*r21*r31 + h12*intrhoz2*r22*r32
                k += 1
                Mv[k] += h13*intrhoz*r11*r22 - h13*intrhoz*r12*r21
                k += 1
                Mv[k] += -h13*intrhoz*r21*r32 + h13*intrhoz*r22*r31
                k += 1
                Mv[k] += h13*intrhoz2*r11*r21 + h13*intrhoz2*r12*r22
                k += 1
                Mv[k] += h13*intrhoz2*r21**2 + h13*intrhoz2*r22**2
                k += 1
                Mv[k] += h13*intrhoz2*r21*r31 + h13*intrhoz2*r22*r32
                k += 1
                Mv[k] += h14*intrhoz*r11*r22 - h14*intrhoz*r12*r21
                k += 1
                Mv[k] += -h14*intrhoz*r21*r32 + h14*intrhoz*r22*r31
                k += 1
                Mv[k] += h14*intrhoz2*r11*r21 + h14*intrhoz2*r12*r22
                k += 1
                Mv[k] += h14*intrhoz2*r21**2 + h14*intrhoz2*r22**2
                k += 1
                Mv[k] += h14*intrhoz2*r21*r31 + h14*intrhoz2*r22*r32
                k += 1
                Mv[k] += h11*intrhoz*r11*r32 - h11*intrhoz*r12*r31
                k += 1
                Mv[k] += h11*intrhoz*r21*r32 - h11*intrhoz*r22*r31
                k += 1
                Mv[k] += h11*intrhoz2*r11*r31 + h11*intrhoz2*r12*r32
                k += 1
                Mv[k] += h11*intrhoz2*r21*r31 + h11*intrhoz2*r22*r32
                k += 1
                Mv[k] += h11*intrhoz2*r31**2 + h11*intrhoz2*r32**2
                k += 1
                Mv[k] += h12*intrhoz*r11*r32 - h12*intrhoz*r12*r31
                k += 1
                Mv[k] += h12*intrhoz*r21*r32 - h12*intrhoz*r22*r31
                k += 1
                Mv[k] += h12*intrhoz2*r11*r31 + h12*intrhoz2*r12*r32
                k += 1
                Mv[k] += h12*intrhoz2*r21*r31 + h12*intrhoz2*r22*r32
                k += 1
                Mv[k] += h12*intrhoz2*r31**2 + h12*intrhoz2*r32**2
                k += 1
                Mv[k] += h13*intrhoz*r11*r32 - h13*intrhoz*r12*r31
                k += 1
                Mv[k] += h13*intrhoz*r21*r32 - h13*intrhoz*r22*r31
                k += 1
                Mv[k] += h13*intrhoz2*r11*r31 + h13*intrhoz2*r12*r32
                k += 1
                Mv[k] += h13*intrhoz2*r21*r31 + h13*intrhoz2*r22*r32
                k += 1
                Mv[k] += h13*intrhoz2*r31**2 + h13*intrhoz2*r32**2
                k += 1
                Mv[k] += h14*intrhoz*r11*r32 - h14*intrhoz*r12*r31
                k += 1
                Mv[k] += h14*intrhoz*r21*r32 - h14*intrhoz*r22*r31
                k += 1
                Mv[k] += h14*intrhoz2*r11*r31 + h14*intrhoz2*r12*r32
                k += 1
                Mv[k] += h14*intrhoz2*r21*r31 + h14*intrhoz2*r22*r32
                k += 1
                Mv[k] += h14*intrhoz2*r31**2 + h14*intrhoz2*r32**2
                k += 1
                Mv[k] += h12*intrho*r11**2 + h12*intrho*r12**2 + h12*intrho*r13**2
                k += 1
                Mv[k] += h12*intrho*r11*r21 + h12*intrho*r12*r22 + h12*intrho*r13*r23
                k += 1
                Mv[k] += h12*intrho*r11*r31 + h12*intrho*r12*r32 + h12*intrho*r13*r33
                k += 1
                Mv[k] += h12*intrhoz*r11*r22 - h12*intrhoz*r12*r21
                k += 1
                Mv[k] += h12*intrhoz*r11*r32 - h12*intrhoz*r12*r31
                k += 1
                Mv[k] += h22*intrho*r11**2 + h22*intrho*r12**2 + h22*intrho*r13**2
                k += 1
                Mv[k] += h22*intrho*r11*r21 + h22*intrho*r12*r22 + h22*intrho*r13*r23
                k += 1
                Mv[k] += h22*intrho*r11*r31 + h22*intrho*r12*r32 + h22*intrho*r13*r33
                k += 1
                Mv[k] += h22*intrhoz*r11*r22 - h22*intrhoz*r12*r21
                k += 1
                Mv[k] += h22*intrhoz*r11*r32 - h22*intrhoz*r12*r31
                k += 1
                Mv[k] += h23*intrho*r11**2 + h23*intrho*r12**2 + h23*intrho*r13**2
                k += 1
                Mv[k] += h23*intrho*r11*r21 + h23*intrho*r12*r22 + h23*intrho*r13*r23
                k += 1
                Mv[k] += h23*intrho*r11*r31 + h23*intrho*r12*r32 + h23*intrho*r13*r33
                k += 1
                Mv[k] += h23*intrhoz*r11*r22 - h23*intrhoz*r12*r21
                k += 1
                Mv[k] += h23*intrhoz*r11*r32 - h23*intrhoz*r12*r31
                k += 1
                Mv[k] += h24*intrho*r11**2 + h24*intrho*r12**2 + h24*intrho*r13**2
                k += 1
                Mv[k] += h24*intrho*r11*r21 + h24*intrho*r12*r22 + h24*intrho*r13*r23
                k += 1
                Mv[k] += h24*intrho*r11*r31 + h24*intrho*r12*r32 + h24*intrho*r13*r33
                k += 1
                Mv[k] += h24*intrhoz*r11*r22 - h24*intrhoz*r12*r21
                k += 1
                Mv[k] += h24*intrhoz*r11*r32 - h24*intrhoz*r12*r31
                k += 1
                Mv[k] += h12*intrho*r11*r21 + h12*intrho*r12*r22 + h12*intrho*r13*r23
                k += 1
                Mv[k] += h12*intrho*r21**2 + h12*intrho*r22**2 + h12*intrho*r23**2
                k += 1
                Mv[k] += h12*intrho*r21*r31 + h12*intrho*r22*r32 + h12*intrho*r23*r33
                k += 1
                Mv[k] += -h12*intrhoz*r11*r22 + h12*intrhoz*r12*r21
                k += 1
                Mv[k] += h12*intrhoz*r21*r32 - h12*intrhoz*r22*r31
                k += 1
                Mv[k] += h22*intrho*r11*r21 + h22*intrho*r12*r22 + h22*intrho*r13*r23
                k += 1
                Mv[k] += h22*intrho*r21**2 + h22*intrho*r22**2 + h22*intrho*r23**2
                k += 1
                Mv[k] += h22*intrho*r21*r31 + h22*intrho*r22*r32 + h22*intrho*r23*r33
                k += 1
                Mv[k] += -h22*intrhoz*r11*r22 + h22*intrhoz*r12*r21
                k += 1
                Mv[k] += h22*intrhoz*r21*r32 - h22*intrhoz*r22*r31
                k += 1
                Mv[k] += h23*intrho*r11*r21 + h23*intrho*r12*r22 + h23*intrho*r13*r23
                k += 1
                Mv[k] += h23*intrho*r21**2 + h23*intrho*r22**2 + h23*intrho*r23**2
                k += 1
                Mv[k] += h23*intrho*r21*r31 + h23*intrho*r22*r32 + h23*intrho*r23*r33
                k += 1
                Mv[k] += -h23*intrhoz*r11*r22 + h23*intrhoz*r12*r21
                k += 1
                Mv[k] += h23*intrhoz*r21*r32 - h23*intrhoz*r22*r31
                k += 1
                Mv[k] += h24*intrho*r11*r21 + h24*intrho*r12*r22 + h24*intrho*r13*r23
                k += 1
                Mv[k] += h24*intrho*r21**2 + h24*intrho*r22**2 + h24*intrho*r23**2
                k += 1
                Mv[k] += h24*intrho*r21*r31 + h24*intrho*r22*r32 + h24*intrho*r23*r33
                k += 1
                Mv[k] += -h24*intrhoz*r11*r22 + h24*intrhoz*r12*r21
                k += 1
                Mv[k] += h24*intrhoz*r21*r32 - h24*intrhoz*r22*r31
                k += 1
                Mv[k] += h12*intrho*r11*r31 + h12*intrho*r12*r32 + h12*intrho*r13*r33
                k += 1
                Mv[k] += h12*intrho*r21*r31 + h12*intrho*r22*r32 + h12*intrho*r23*r33
                k += 1
                Mv[k] += h12*intrho*r31**2 + h12*intrho*r32**2 + h12*intrho*r33**2
                k += 1
                Mv[k] += -h12*intrhoz*r11*r32 + h12*intrhoz*r12*r31
                k += 1
                Mv[k] += -h12*intrhoz*r21*r32 + h12*intrhoz*r22*r31
                k += 1
                Mv[k] += h22*intrho*r11*r31 + h22*intrho*r12*r32 + h22*intrho*r13*r33
                k += 1
                Mv[k] += h22*intrho*r21*r31 + h22*intrho*r22*r32 + h22*intrho*r23*r33
                k += 1
                Mv[k] += h22*intrho*r31**2 + h22*intrho*r32**2 + h22*intrho*r33**2
                k += 1
                Mv[k] += -h22*intrhoz*r11*r32 + h22*intrhoz*r12*r31
                k += 1
                Mv[k] += -h22*intrhoz*r21*r32 + h22*intrhoz*r22*r31
                k += 1
                Mv[k] += h23*intrho*r11*r31 + h23*intrho*r12*r32 + h23*intrho*r13*r33
                k += 1
                Mv[k] += h23*intrho*r21*r31 + h23*intrho*r22*r32 + h23*intrho*r23*r33
                k += 1
                Mv[k] += h23*intrho*r31**2 + h23*intrho*r32**2 + h23*intrho*r33**2
                k += 1
                Mv[k] += -h23*intrhoz*r11*r32 + h23*intrhoz*r12*r31
                k += 1
                Mv[k] += -h23*intrhoz*r21*r32 + h23*intrhoz*r22*r31
                k += 1
                Mv[k] += h24*intrho*r11*r31 + h24*intrho*r12*r32 + h24*intrho*r13*r33
                k += 1
                Mv[k] += h24*intrho*r21*r31 + h24*intrho*r22*r32 + h24*intrho*r23*r33
                k += 1
                Mv[k] += h24*intrho*r31**2 + h24*intrho*r32**2 + h24*intrho*r33**2
                k += 1
                Mv[k] += -h24*intrhoz*r11*r32 + h24*intrhoz*r12*r31
                k += 1
                Mv[k] += -h24*intrhoz*r21*r32 + h24*intrhoz*r22*r31
                k += 1
                Mv[k] += -h12*intrhoz*r11*r22 + h12*intrhoz*r12*r21
                k += 1
                Mv[k] += -h12*intrhoz*r11*r32 + h12*intrhoz*r12*r31
                k += 1
                Mv[k] += h12*intrhoz2*r11**2 + h12*intrhoz2*r12**2
                k += 1
                Mv[k] += h12*intrhoz2*r11*r21 + h12*intrhoz2*r12*r22
                k += 1
                Mv[k] += h12*intrhoz2*r11*r31 + h12*intrhoz2*r12*r32
                k += 1
                Mv[k] += -h22*intrhoz*r11*r22 + h22*intrhoz*r12*r21
                k += 1
                Mv[k] += -h22*intrhoz*r11*r32 + h22*intrhoz*r12*r31
                k += 1
                Mv[k] += h22*intrhoz2*r11**2 + h22*intrhoz2*r12**2
                k += 1
                Mv[k] += h22*intrhoz2*r11*r21 + h22*intrhoz2*r12*r22
                k += 1
                Mv[k] += h22*intrhoz2*r11*r31 + h22*intrhoz2*r12*r32
                k += 1
                Mv[k] += -h23*intrhoz*r11*r22 + h23*intrhoz*r12*r21
                k += 1
                Mv[k] += -h23*intrhoz*r11*r32 + h23*intrhoz*r12*r31
                k += 1
                Mv[k] += h23*intrhoz2*r11**2 + h23*intrhoz2*r12**2
                k += 1
                Mv[k] += h23*intrhoz2*r11*r21 + h23*intrhoz2*r12*r22
                k += 1
                Mv[k] += h23*intrhoz2*r11*r31 + h23*intrhoz2*r12*r32
                k += 1
                Mv[k] += -h24*intrhoz*r11*r22 + h24*intrhoz*r12*r21
                k += 1
                Mv[k] += -h24*intrhoz*r11*r32 + h24*intrhoz*r12*r31
                k += 1
                Mv[k] += h24*intrhoz2*r11**2 + h24*intrhoz2*r12**2
                k += 1
                Mv[k] += h24*intrhoz2*r11*r21 + h24*intrhoz2*r12*r22
                k += 1
                Mv[k] += h24*intrhoz2*r11*r31 + h24*intrhoz2*r12*r32
                k += 1
                Mv[k] += h12*intrhoz*r11*r22 - h12*intrhoz*r12*r21
                k += 1
                Mv[k] += -h12*intrhoz*r21*r32 + h12*intrhoz*r22*r31
                k += 1
                Mv[k] += h12*intrhoz2*r11*r21 + h12*intrhoz2*r12*r22
                k += 1
                Mv[k] += h12*intrhoz2*r21**2 + h12*intrhoz2*r22**2
                k += 1
                Mv[k] += h12*intrhoz2*r21*r31 + h12*intrhoz2*r22*r32
                k += 1
                Mv[k] += h22*intrhoz*r11*r22 - h22*intrhoz*r12*r21
                k += 1
                Mv[k] += -h22*intrhoz*r21*r32 + h22*intrhoz*r22*r31
                k += 1
                Mv[k] += h22*intrhoz2*r11*r21 + h22*intrhoz2*r12*r22
                k += 1
                Mv[k] += h22*intrhoz2*r21**2 + h22*intrhoz2*r22**2
                k += 1
                Mv[k] += h22*intrhoz2*r21*r31 + h22*intrhoz2*r22*r32
                k += 1
                Mv[k] += h23*intrhoz*r11*r22 - h23*intrhoz*r12*r21
                k += 1
                Mv[k] += -h23*intrhoz*r21*r32 + h23*intrhoz*r22*r31
                k += 1
                Mv[k] += h23*intrhoz2*r11*r21 + h23*intrhoz2*r12*r22
                k += 1
                Mv[k] += h23*intrhoz2*r21**2 + h23*intrhoz2*r22**2
                k += 1
                Mv[k] += h23*intrhoz2*r21*r31 + h23*intrhoz2*r22*r32
                k += 1
                Mv[k] += h24*intrhoz*r11*r22 - h24*intrhoz*r12*r21
                k += 1
                Mv[k] += -h24*intrhoz*r21*r32 + h24*intrhoz*r22*r31
                k += 1
                Mv[k] += h24*intrhoz2*r11*r21 + h24*intrhoz2*r12*r22
                k += 1
                Mv[k] += h24*intrhoz2*r21**2 + h24*intrhoz2*r22**2
                k += 1
                Mv[k] += h24*intrhoz2*r21*r31 + h24*intrhoz2*r22*r32
                k += 1
                Mv[k] += h12*intrhoz*r11*r32 - h12*intrhoz*r12*r31
                k += 1
                Mv[k] += h12*intrhoz*r21*r32 - h12*intrhoz*r22*r31
                k += 1
                Mv[k] += h12*intrhoz2*r11*r31 + h12*intrhoz2*r12*r32
                k += 1
                Mv[k] += h12*intrhoz2*r21*r31 + h12*intrhoz2*r22*r32
                k += 1
                Mv[k] += h12*intrhoz2*r31**2 + h12*intrhoz2*r32**2
                k += 1
                Mv[k] += h22*intrhoz*r11*r32 - h22*intrhoz*r12*r31
                k += 1
                Mv[k] += h22*intrhoz*r21*r32 - h22*intrhoz*r22*r31
                k += 1
                Mv[k] += h22*intrhoz2*r11*r31 + h22*intrhoz2*r12*r32
                k += 1
                Mv[k] += h22*intrhoz2*r21*r31 + h22*intrhoz2*r22*r32
                k += 1
                Mv[k] += h22*intrhoz2*r31**2 + h22*intrhoz2*r32**2
                k += 1
                Mv[k] += h23*intrhoz*r11*r32 - h23*intrhoz*r12*r31
                k += 1
                Mv[k] += h23*intrhoz*r21*r32 - h23*intrhoz*r22*r31
                k += 1
                Mv[k] += h23*intrhoz2*r11*r31 + h23*intrhoz2*r12*r32
                k += 1
                Mv[k] += h23*intrhoz2*r21*r31 + h23*intrhoz2*r22*r32
                k += 1
                Mv[k] += h23*intrhoz2*r31**2 + h23*intrhoz2*r32**2
                k += 1
                Mv[k] += h24*intrhoz*r11*r32 - h24*intrhoz*r12*r31
                k += 1
                Mv[k] += h24*intrhoz*r21*r32 - h24*intrhoz*r22*r31
                k += 1
                Mv[k] += h24*intrhoz2*r11*r31 + h24*intrhoz2*r12*r32
                k += 1
                Mv[k] += h24*intrhoz2*r21*r31 + h24*intrhoz2*r22*r32
                k += 1
                Mv[k] += h24*intrhoz2*r31**2 + h24*intrhoz2*r32**2
                k += 1
                Mv[k] += h13*intrho*r11**2 + h13*intrho*r12**2 + h13*intrho*r13**2
                k += 1
                Mv[k] += h13*intrho*r11*r21 + h13*intrho*r12*r22 + h13*intrho*r13*r23
                k += 1
                Mv[k] += h13*intrho*r11*r31 + h13*intrho*r12*r32 + h13*intrho*r13*r33
                k += 1
                Mv[k] += h13*intrhoz*r11*r22 - h13*intrhoz*r12*r21
                k += 1
                Mv[k] += h13*intrhoz*r11*r32 - h13*intrhoz*r12*r31
                k += 1
                Mv[k] += h23*intrho*r11**2 + h23*intrho*r12**2 + h23*intrho*r13**2
                k += 1
                Mv[k] += h23*intrho*r11*r21 + h23*intrho*r12*r22 + h23*intrho*r13*r23
                k += 1
                Mv[k] += h23*intrho*r11*r31 + h23*intrho*r12*r32 + h23*intrho*r13*r33
                k += 1
                Mv[k] += h23*intrhoz*r11*r22 - h23*intrhoz*r12*r21
                k += 1
                Mv[k] += h23*intrhoz*r11*r32 - h23*intrhoz*r12*r31
                k += 1
                Mv[k] += h33*intrho*r11**2 + h33*intrho*r12**2 + h33*intrho*r13**2
                k += 1
                Mv[k] += h33*intrho*r11*r21 + h33*intrho*r12*r22 + h33*intrho*r13*r23
                k += 1
                Mv[k] += h33*intrho*r11*r31 + h33*intrho*r12*r32 + h33*intrho*r13*r33
                k += 1
                Mv[k] += h33*intrhoz*r11*r22 - h33*intrhoz*r12*r21
                k += 1
                Mv[k] += h33*intrhoz*r11*r32 - h33*intrhoz*r12*r31
                k += 1
                Mv[k] += h34*intrho*r11**2 + h34*intrho*r12**2 + h34*intrho*r13**2
                k += 1
                Mv[k] += h34*intrho*r11*r21 + h34*intrho*r12*r22 + h34*intrho*r13*r23
                k += 1
                Mv[k] += h34*intrho*r11*r31 + h34*intrho*r12*r32 + h34*intrho*r13*r33
                k += 1
                Mv[k] += h34*intrhoz*r11*r22 - h34*intrhoz*r12*r21
                k += 1
                Mv[k] += h34*intrhoz*r11*r32 - h34*intrhoz*r12*r31
                k += 1
                Mv[k] += h13*intrho*r11*r21 + h13*intrho*r12*r22 + h13*intrho*r13*r23
                k += 1
                Mv[k] += h13*intrho*r21**2 + h13*intrho*r22**2 + h13*intrho*r23**2
                k += 1
                Mv[k] += h13*intrho*r21*r31 + h13*intrho*r22*r32 + h13*intrho*r23*r33
                k += 1
                Mv[k] += -h13*intrhoz*r11*r22 + h13*intrhoz*r12*r21
                k += 1
                Mv[k] += h13*intrhoz*r21*r32 - h13*intrhoz*r22*r31
                k += 1
                Mv[k] += h23*intrho*r11*r21 + h23*intrho*r12*r22 + h23*intrho*r13*r23
                k += 1
                Mv[k] += h23*intrho*r21**2 + h23*intrho*r22**2 + h23*intrho*r23**2
                k += 1
                Mv[k] += h23*intrho*r21*r31 + h23*intrho*r22*r32 + h23*intrho*r23*r33
                k += 1
                Mv[k] += -h23*intrhoz*r11*r22 + h23*intrhoz*r12*r21
                k += 1
                Mv[k] += h23*intrhoz*r21*r32 - h23*intrhoz*r22*r31
                k += 1
                Mv[k] += h33*intrho*r11*r21 + h33*intrho*r12*r22 + h33*intrho*r13*r23
                k += 1
                Mv[k] += h33*intrho*r21**2 + h33*intrho*r22**2 + h33*intrho*r23**2
                k += 1
                Mv[k] += h33*intrho*r21*r31 + h33*intrho*r22*r32 + h33*intrho*r23*r33
                k += 1
                Mv[k] += -h33*intrhoz*r11*r22 + h33*intrhoz*r12*r21
                k += 1
                Mv[k] += h33*intrhoz*r21*r32 - h33*intrhoz*r22*r31
                k += 1
                Mv[k] += h34*intrho*r11*r21 + h34*intrho*r12*r22 + h34*intrho*r13*r23
                k += 1
                Mv[k] += h34*intrho*r21**2 + h34*intrho*r22**2 + h34*intrho*r23**2
                k += 1
                Mv[k] += h34*intrho*r21*r31 + h34*intrho*r22*r32 + h34*intrho*r23*r33
                k += 1
                Mv[k] += -h34*intrhoz*r11*r22 + h34*intrhoz*r12*r21
                k += 1
                Mv[k] += h34*intrhoz*r21*r32 - h34*intrhoz*r22*r31
                k += 1
                Mv[k] += h13*intrho*r11*r31 + h13*intrho*r12*r32 + h13*intrho*r13*r33
                k += 1
                Mv[k] += h13*intrho*r21*r31 + h13*intrho*r22*r32 + h13*intrho*r23*r33
                k += 1
                Mv[k] += h13*intrho*r31**2 + h13*intrho*r32**2 + h13*intrho*r33**2
                k += 1
                Mv[k] += -h13*intrhoz*r11*r32 + h13*intrhoz*r12*r31
                k += 1
                Mv[k] += -h13*intrhoz*r21*r32 + h13*intrhoz*r22*r31
                k += 1
                Mv[k] += h23*intrho*r11*r31 + h23*intrho*r12*r32 + h23*intrho*r13*r33
                k += 1
                Mv[k] += h23*intrho*r21*r31 + h23*intrho*r22*r32 + h23*intrho*r23*r33
                k += 1
                Mv[k] += h23*intrho*r31**2 + h23*intrho*r32**2 + h23*intrho*r33**2
                k += 1
                Mv[k] += -h23*intrhoz*r11*r32 + h23*intrhoz*r12*r31
                k += 1
                Mv[k] += -h23*intrhoz*r21*r32 + h23*intrhoz*r22*r31
                k += 1
                Mv[k] += h33*intrho*r11*r31 + h33*intrho*r12*r32 + h33*intrho*r13*r33
                k += 1
                Mv[k] += h33*intrho*r21*r31 + h33*intrho*r22*r32 + h33*intrho*r23*r33
                k += 1
                Mv[k] += h33*intrho*r31**2 + h33*intrho*r32**2 + h33*intrho*r33**2
                k += 1
                Mv[k] += -h33*intrhoz*r11*r32 + h33*intrhoz*r12*r31
                k += 1
                Mv[k] += -h33*intrhoz*r21*r32 + h33*intrhoz*r22*r31
                k += 1
                Mv[k] += h34*intrho*r11*r31 + h34*intrho*r12*r32 + h34*intrho*r13*r33
                k += 1
                Mv[k] += h34*intrho*r21*r31 + h34*intrho*r22*r32 + h34*intrho*r23*r33
                k += 1
                Mv[k] += h34*intrho*r31**2 + h34*intrho*r32**2 + h34*intrho*r33**2
                k += 1
                Mv[k] += -h34*intrhoz*r11*r32 + h34*intrhoz*r12*r31
                k += 1
                Mv[k] += -h34*intrhoz*r21*r32 + h34*intrhoz*r22*r31
                k += 1
                Mv[k] += -h13*intrhoz*r11*r22 + h13*intrhoz*r12*r21
                k += 1
                Mv[k] += -h13*intrhoz*r11*r32 + h13*intrhoz*r12*r31
                k += 1
                Mv[k] += h13*intrhoz2*r11**2 + h13*intrhoz2*r12**2
                k += 1
                Mv[k] += h13*intrhoz2*r11*r21 + h13*intrhoz2*r12*r22
                k += 1
                Mv[k] += h13*intrhoz2*r11*r31 + h13*intrhoz2*r12*r32
                k += 1
                Mv[k] += -h23*intrhoz*r11*r22 + h23*intrhoz*r12*r21
                k += 1
                Mv[k] += -h23*intrhoz*r11*r32 + h23*intrhoz*r12*r31
                k += 1
                Mv[k] += h23*intrhoz2*r11**2 + h23*intrhoz2*r12**2
                k += 1
                Mv[k] += h23*intrhoz2*r11*r21 + h23*intrhoz2*r12*r22
                k += 1
                Mv[k] += h23*intrhoz2*r11*r31 + h23*intrhoz2*r12*r32
                k += 1
                Mv[k] += -h33*intrhoz*r11*r22 + h33*intrhoz*r12*r21
                k += 1
                Mv[k] += -h33*intrhoz*r11*r32 + h33*intrhoz*r12*r31
                k += 1
                Mv[k] += h33*intrhoz2*r11**2 + h33*intrhoz2*r12**2
                k += 1
                Mv[k] += h33*intrhoz2*r11*r21 + h33*intrhoz2*r12*r22
                k += 1
                Mv[k] += h33*intrhoz2*r11*r31 + h33*intrhoz2*r12*r32
                k += 1
                Mv[k] += -h34*intrhoz*r11*r22 + h34*intrhoz*r12*r21
                k += 1
                Mv[k] += -h34*intrhoz*r11*r32 + h34*intrhoz*r12*r31
                k += 1
                Mv[k] += h34*intrhoz2*r11**2 + h34*intrhoz2*r12**2
                k += 1
                Mv[k] += h34*intrhoz2*r11*r21 + h34*intrhoz2*r12*r22
                k += 1
                Mv[k] += h34*intrhoz2*r11*r31 + h34*intrhoz2*r12*r32
                k += 1
                Mv[k] += h13*intrhoz*r11*r22 - h13*intrhoz*r12*r21
                k += 1
                Mv[k] += -h13*intrhoz*r21*r32 + h13*intrhoz*r22*r31
                k += 1
                Mv[k] += h13*intrhoz2*r11*r21 + h13*intrhoz2*r12*r22
                k += 1
                Mv[k] += h13*intrhoz2*r21**2 + h13*intrhoz2*r22**2
                k += 1
                Mv[k] += h13*intrhoz2*r21*r31 + h13*intrhoz2*r22*r32
                k += 1
                Mv[k] += h23*intrhoz*r11*r22 - h23*intrhoz*r12*r21
                k += 1
                Mv[k] += -h23*intrhoz*r21*r32 + h23*intrhoz*r22*r31
                k += 1
                Mv[k] += h23*intrhoz2*r11*r21 + h23*intrhoz2*r12*r22
                k += 1
                Mv[k] += h23*intrhoz2*r21**2 + h23*intrhoz2*r22**2
                k += 1
                Mv[k] += h23*intrhoz2*r21*r31 + h23*intrhoz2*r22*r32
                k += 1
                Mv[k] += h33*intrhoz*r11*r22 - h33*intrhoz*r12*r21
                k += 1
                Mv[k] += -h33*intrhoz*r21*r32 + h33*intrhoz*r22*r31
                k += 1
                Mv[k] += h33*intrhoz2*r11*r21 + h33*intrhoz2*r12*r22
                k += 1
                Mv[k] += h33*intrhoz2*r21**2 + h33*intrhoz2*r22**2
                k += 1
                Mv[k] += h33*intrhoz2*r21*r31 + h33*intrhoz2*r22*r32
                k += 1
                Mv[k] += h34*intrhoz*r11*r22 - h34*intrhoz*r12*r21
                k += 1
                Mv[k] += -h34*intrhoz*r21*r32 + h34*intrhoz*r22*r31
                k += 1
                Mv[k] += h34*intrhoz2*r11*r21 + h34*intrhoz2*r12*r22
                k += 1
                Mv[k] += h34*intrhoz2*r21**2 + h34*intrhoz2*r22**2
                k += 1
                Mv[k] += h34*intrhoz2*r21*r31 + h34*intrhoz2*r22*r32
                k += 1
                Mv[k] += h13*intrhoz*r11*r32 - h13*intrhoz*r12*r31
                k += 1
                Mv[k] += h13*intrhoz*r21*r32 - h13*intrhoz*r22*r31
                k += 1
                Mv[k] += h13*intrhoz2*r11*r31 + h13*intrhoz2*r12*r32
                k += 1
                Mv[k] += h13*intrhoz2*r21*r31 + h13*intrhoz2*r22*r32
                k += 1
                Mv[k] += h13*intrhoz2*r31**2 + h13*intrhoz2*r32**2
                k += 1
                Mv[k] += h23*intrhoz*r11*r32 - h23*intrhoz*r12*r31
                k += 1
                Mv[k] += h23*intrhoz*r21*r32 - h23*intrhoz*r22*r31
                k += 1
                Mv[k] += h23*intrhoz2*r11*r31 + h23*intrhoz2*r12*r32
                k += 1
                Mv[k] += h23*intrhoz2*r21*r31 + h23*intrhoz2*r22*r32
                k += 1
                Mv[k] += h23*intrhoz2*r31**2 + h23*intrhoz2*r32**2
                k += 1
                Mv[k] += h33*intrhoz*r11*r32 - h33*intrhoz*r12*r31
                k += 1
                Mv[k] += h33*intrhoz*r21*r32 - h33*intrhoz*r22*r31
                k += 1
                Mv[k] += h33*intrhoz2*r11*r31 + h33*intrhoz2*r12*r32
                k += 1
                Mv[k] += h33*intrhoz2*r21*r31 + h33*intrhoz2*r22*r32
                k += 1
                Mv[k] += h33*intrhoz2*r31**2 + h33*intrhoz2*r32**2
                k += 1
                Mv[k] += h34*intrhoz*r11*r32 - h34*intrhoz*r12*r31
                k += 1
                Mv[k] += h34*intrhoz*r21*r32 - h34*intrhoz*r22*r31
                k += 1
                Mv[k] += h34*intrhoz2*r11*r31 + h34*intrhoz2*r12*r32
                k += 1
                Mv[k] += h34*intrhoz2*r21*r31 + h34*intrhoz2*r22*r32
                k += 1
                Mv[k] += h34*intrhoz2*r31**2 + h34*intrhoz2*r32**2
                k += 1
                Mv[k] += h14*intrho*r11**2 + h14*intrho*r12**2 + h14*intrho*r13**2
                k += 1
                Mv[k] += h14*intrho*r11*r21 + h14*intrho*r12*r22 + h14*intrho*r13*r23
                k += 1
                Mv[k] += h14*intrho*r11*r31 + h14*intrho*r12*r32 + h14*intrho*r13*r33
                k += 1
                Mv[k] += h14*intrhoz*r11*r22 - h14*intrhoz*r12*r21
                k += 1
                Mv[k] += h14*intrhoz*r11*r32 - h14*intrhoz*r12*r31
                k += 1
                Mv[k] += h24*intrho*r11**2 + h24*intrho*r12**2 + h24*intrho*r13**2
                k += 1
                Mv[k] += h24*intrho*r11*r21 + h24*intrho*r12*r22 + h24*intrho*r13*r23
                k += 1
                Mv[k] += h24*intrho*r11*r31 + h24*intrho*r12*r32 + h24*intrho*r13*r33
                k += 1
                Mv[k] += h24*intrhoz*r11*r22 - h24*intrhoz*r12*r21
                k += 1
                Mv[k] += h24*intrhoz*r11*r32 - h24*intrhoz*r12*r31
                k += 1
                Mv[k] += h34*intrho*r11**2 + h34*intrho*r12**2 + h34*intrho*r13**2
                k += 1
                Mv[k] += h34*intrho*r11*r21 + h34*intrho*r12*r22 + h34*intrho*r13*r23
                k += 1
                Mv[k] += h34*intrho*r11*r31 + h34*intrho*r12*r32 + h34*intrho*r13*r33
                k += 1
                Mv[k] += h34*intrhoz*r11*r22 - h34*intrhoz*r12*r21
                k += 1
                Mv[k] += h34*intrhoz*r11*r32 - h34*intrhoz*r12*r31
                k += 1
                Mv[k] += h44*intrho*r11**2 + h44*intrho*r12**2 + h44*intrho*r13**2
                k += 1
                Mv[k] += h44*intrho*r11*r21 + h44*intrho*r12*r22 + h44*intrho*r13*r23
                k += 1
                Mv[k] += h44*intrho*r11*r31 + h44*intrho*r12*r32 + h44*intrho*r13*r33
                k += 1
                Mv[k] += h44*intrhoz*r11*r22 - h44*intrhoz*r12*r21
                k += 1
                Mv[k] += h44*intrhoz*r11*r32 - h44*intrhoz*r12*r31
                k += 1
                Mv[k] += h14*intrho*r11*r21 + h14*intrho*r12*r22 + h14*intrho*r13*r23
                k += 1
                Mv[k] += h14*intrho*r21**2 + h14*intrho*r22**2 + h14*intrho*r23**2
                k += 1
                Mv[k] += h14*intrho*r21*r31 + h14*intrho*r22*r32 + h14*intrho*r23*r33
                k += 1
                Mv[k] += -h14*intrhoz*r11*r22 + h14*intrhoz*r12*r21
                k += 1
                Mv[k] += h14*intrhoz*r21*r32 - h14*intrhoz*r22*r31
                k += 1
                Mv[k] += h24*intrho*r11*r21 + h24*intrho*r12*r22 + h24*intrho*r13*r23
                k += 1
                Mv[k] += h24*intrho*r21**2 + h24*intrho*r22**2 + h24*intrho*r23**2
                k += 1
                Mv[k] += h24*intrho*r21*r31 + h24*intrho*r22*r32 + h24*intrho*r23*r33
                k += 1
                Mv[k] += -h24*intrhoz*r11*r22 + h24*intrhoz*r12*r21
                k += 1
                Mv[k] += h24*intrhoz*r21*r32 - h24*intrhoz*r22*r31
                k += 1
                Mv[k] += h34*intrho*r11*r21 + h34*intrho*r12*r22 + h34*intrho*r13*r23
                k += 1
                Mv[k] += h34*intrho*r21**2 + h34*intrho*r22**2 + h34*intrho*r23**2
                k += 1
                Mv[k] += h34*intrho*r21*r31 + h34*intrho*r22*r32 + h34*intrho*r23*r33
                k += 1
                Mv[k] += -h34*intrhoz*r11*r22 + h34*intrhoz*r12*r21
                k += 1
                Mv[k] += h34*intrhoz*r21*r32 - h34*intrhoz*r22*r31
                k += 1
                Mv[k] += h44*intrho*r11*r21 + h44*intrho*r12*r22 + h44*intrho*r13*r23
                k += 1
                Mv[k] += h44*intrho*r21**2 + h44*intrho*r22**2 + h44*intrho*r23**2
                k += 1
                Mv[k] += h44*intrho*r21*r31 + h44*intrho*r22*r32 + h44*intrho*r23*r33
                k += 1
                Mv[k] += -h44*intrhoz*r11*r22 + h44*intrhoz*r12*r21
                k += 1
                Mv[k] += h44*intrhoz*r21*r32 - h44*intrhoz*r22*r31
                k += 1
                Mv[k] += h14*intrho*r11*r31 + h14*intrho*r12*r32 + h14*intrho*r13*r33
                k += 1
                Mv[k] += h14*intrho*r21*r31 + h14*intrho*r22*r32 + h14*intrho*r23*r33
                k += 1
                Mv[k] += h14*intrho*r31**2 + h14*intrho*r32**2 + h14*intrho*r33**2
                k += 1
                Mv[k] += -h14*intrhoz*r11*r32 + h14*intrhoz*r12*r31
                k += 1
                Mv[k] += -h14*intrhoz*r21*r32 + h14*intrhoz*r22*r31
                k += 1
                Mv[k] += h24*intrho*r11*r31 + h24*intrho*r12*r32 + h24*intrho*r13*r33
                k += 1
                Mv[k] += h24*intrho*r21*r31 + h24*intrho*r22*r32 + h24*intrho*r23*r33
                k += 1
                Mv[k] += h24*intrho*r31**2 + h24*intrho*r32**2 + h24*intrho*r33**2
                k += 1
                Mv[k] += -h24*intrhoz*r11*r32 + h24*intrhoz*r12*r31
                k += 1
                Mv[k] += -h24*intrhoz*r21*r32 + h24*intrhoz*r22*r31
                k += 1
                Mv[k] += h34*intrho*r11*r31 + h34*intrho*r12*r32 + h34*intrho*r13*r33
                k += 1
                Mv[k] += h34*intrho*r21*r31 + h34*intrho*r22*r32 + h34*intrho*r23*r33
                k += 1
                Mv[k] += h34*intrho*r31**2 + h34*intrho*r32**2 + h34*intrho*r33**2
                k += 1
                Mv[k] += -h34*intrhoz*r11*r32 + h34*intrhoz*r12*r31
                k += 1
                Mv[k] += -h34*intrhoz*r21*r32 + h34*intrhoz*r22*r31
                k += 1
                Mv[k] += h44*intrho*r11*r31 + h44*intrho*r12*r32 + h44*intrho*r13*r33
                k += 1
                Mv[k] += h44*intrho*r21*r31 + h44*intrho*r22*r32 + h44*intrho*r23*r33
                k += 1
                Mv[k] += h44*intrho*r31**2 + h44*intrho*r32**2 + h44*intrho*r33**2
                k += 1
                Mv[k] += -h44*intrhoz*r11*r32 + h44*intrhoz*r12*r31
                k += 1
                Mv[k] += -h44*intrhoz*r21*r32 + h44*intrhoz*r22*r31
                k += 1
                Mv[k] += -h14*intrhoz*r11*r22 + h14*intrhoz*r12*r21
                k += 1
                Mv[k] += -h14*intrhoz*r11*r32 + h14*intrhoz*r12*r31
                k += 1
                Mv[k] += h14*intrhoz2*r11**2 + h14*intrhoz2*r12**2
                k += 1
                Mv[k] += h14*intrhoz2*r11*r21 + h14*intrhoz2*r12*r22
                k += 1
                Mv[k] += h14*intrhoz2*r11*r31 + h14*intrhoz2*r12*r32
                k += 1
                Mv[k] += -h24*intrhoz*r11*r22 + h24*intrhoz*r12*r21
                k += 1
                Mv[k] += -h24*intrhoz*r11*r32 + h24*intrhoz*r12*r31
                k += 1
                Mv[k] += h24*intrhoz2*r11**2 + h24*intrhoz2*r12**2
                k += 1
                Mv[k] += h24*intrhoz2*r11*r21 + h24*intrhoz2*r12*r22
                k += 1
                Mv[k] += h24*intrhoz2*r11*r31 + h24*intrhoz2*r12*r32
                k += 1
                Mv[k] += -h34*intrhoz*r11*r22 + h34*intrhoz*r12*r21
                k += 1
                Mv[k] += -h34*intrhoz*r11*r32 + h34*intrhoz*r12*r31
                k += 1
                Mv[k] += h34*intrhoz2*r11**2 + h34*intrhoz2*r12**2
                k += 1
                Mv[k] += h34*intrhoz2*r11*r21 + h34*intrhoz2*r12*r22
                k += 1
                Mv[k] += h34*intrhoz2*r11*r31 + h34*intrhoz2*r12*r32
                k += 1
                Mv[k] += -h44*intrhoz*r11*r22 + h44*intrhoz*r12*r21
                k += 1
                Mv[k] += -h44*intrhoz*r11*r32 + h44*intrhoz*r12*r31
                k += 1
                Mv[k] += h44*intrhoz2*r11**2 + h44*intrhoz2*r12**2
                k += 1
                Mv[k] += h44*intrhoz2*r11*r21 + h44*intrhoz2*r12*r22
                k += 1
                Mv[k] += h44*intrhoz2*r11*r31 + h44*intrhoz2*r12*r32
                k += 1
                Mv[k] += h14*intrhoz*r11*r22 - h14*intrhoz*r12*r21
                k += 1
                Mv[k] += -h14*intrhoz*r21*r32 + h14*intrhoz*r22*r31
                k += 1
                Mv[k] += h14*intrhoz2*r11*r21 + h14*intrhoz2*r12*r22
                k += 1
                Mv[k] += h14*intrhoz2*r21**2 + h14*intrhoz2*r22**2
                k += 1
                Mv[k] += h14*intrhoz2*r21*r31 + h14*intrhoz2*r22*r32
                k += 1
                Mv[k] += h24*intrhoz*r11*r22 - h24*intrhoz*r12*r21
                k += 1
                Mv[k] += -h24*intrhoz*r21*r32 + h24*intrhoz*r22*r31
                k += 1
                Mv[k] += h24*intrhoz2*r11*r21 + h24*intrhoz2*r12*r22
                k += 1
                Mv[k] += h24*intrhoz2*r21**2 + h24*intrhoz2*r22**2
                k += 1
                Mv[k] += h24*intrhoz2*r21*r31 + h24*intrhoz2*r22*r32
                k += 1
                Mv[k] += h34*intrhoz*r11*r22 - h34*intrhoz*r12*r21
                k += 1
                Mv[k] += -h34*intrhoz*r21*r32 + h34*intrhoz*r22*r31
                k += 1
                Mv[k] += h34*intrhoz2*r11*r21 + h34*intrhoz2*r12*r22
                k += 1
                Mv[k] += h34*intrhoz2*r21**2 + h34*intrhoz2*r22**2
                k += 1
                Mv[k] += h34*intrhoz2*r21*r31 + h34*intrhoz2*r22*r32
                k += 1
                Mv[k] += h44*intrhoz*r11*r22 - h44*intrhoz*r12*r21
                k += 1
                Mv[k] += -h44*intrhoz*r21*r32 + h44*intrhoz*r22*r31
                k += 1
                Mv[k] += h44*intrhoz2*r11*r21 + h44*intrhoz2*r12*r22
                k += 1
                Mv[k] += h44*intrhoz2*r21**2 + h44*intrhoz2*r22**2
                k += 1
                Mv[k] += h44*intrhoz2*r21*r31 + h44*intrhoz2*r22*r32
                k += 1
                Mv[k] += h14*intrhoz*r11*r32 - h14*intrhoz*r12*r31
                k += 1
                Mv[k] += h14*intrhoz*r21*r32 - h14*intrhoz*r22*r31
                k += 1
                Mv[k] += h14*intrhoz2*r11*r31 + h14*intrhoz2*r12*r32
                k += 1
                Mv[k] += h14*intrhoz2*r21*r31 + h14*intrhoz2*r22*r32
                k += 1
                Mv[k] += h14*intrhoz2*r31**2 + h14*intrhoz2*r32**2
                k += 1
                Mv[k] += h24*intrhoz*r11*r32 - h24*intrhoz*r12*r31
                k += 1
                Mv[k] += h24*intrhoz*r21*r32 - h24*intrhoz*r22*r31
                k += 1
                Mv[k] += h24*intrhoz2*r11*r31 + h24*intrhoz2*r12*r32
                k += 1
                Mv[k] += h24*intrhoz2*r21*r31 + h24*intrhoz2*r22*r32
                k += 1
                Mv[k] += h24*intrhoz2*r31**2 + h24*intrhoz2*r32**2
                k += 1
                Mv[k] += h34*intrhoz*r11*r32 - h34*intrhoz*r12*r31
                k += 1
                Mv[k] += h34*intrhoz*r21*r32 - h34*intrhoz*r22*r31
                k += 1
                Mv[k] += h34*intrhoz2*r11*r31 + h34*intrhoz2*r12*r32
                k += 1
                Mv[k] += h34*intrhoz2*r21*r31 + h34*intrhoz2*r22*r32
                k += 1
                Mv[k] += h34*intrhoz2*r31**2 + h34*intrhoz2*r32**2
                k += 1
                Mv[k] += h44*intrhoz*r11*r32 - h44*intrhoz*r12*r31
                k += 1
                Mv[k] += h44*intrhoz*r21*r32 - h44*intrhoz*r22*r31
                k += 1
                Mv[k] += h44*intrhoz2*r11*r31 + h44*intrhoz2*r12*r32
                k += 1
                Mv[k] += h44*intrhoz2*r21*r31 + h44*intrhoz2*r22*r32
                k += 1
                Mv[k] += h44*intrhoz2*r31**2 + h44*intrhoz2*r32**2

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
                Mv[k] += intrhoz*r11*r22*valH1 - intrhoz*r12*r21*valH1
                k += 1
                Mv[k] += intrhoz*r11*r32*valH1 - intrhoz*r12*r31*valH1
                k += 1
                Mv[k] += intrho*r11**2*valH1 + intrho*r12**2*valH1 + intrho*r13**2*valH1
                k += 1
                Mv[k] += intrho*r11*r21*valH1 + intrho*r12*r22*valH1 + intrho*r13*r23*valH1
                k += 1
                Mv[k] += intrho*r11*r31*valH1 + intrho*r12*r32*valH1 + intrho*r13*r33*valH1
                k += 1
                Mv[k] += intrhoz*r11*r22*valH1 - intrhoz*r12*r21*valH1
                k += 1
                Mv[k] += intrhoz*r11*r32*valH1 - intrhoz*r12*r31*valH1
                k += 1
                Mv[k] += intrho*r11**2*valH1 + intrho*r12**2*valH1 + intrho*r13**2*valH1
                k += 1
                Mv[k] += intrho*r11*r21*valH1 + intrho*r12*r22*valH1 + intrho*r13*r23*valH1
                k += 1
                Mv[k] += intrho*r11*r31*valH1 + intrho*r12*r32*valH1 + intrho*r13*r33*valH1
                k += 1
                Mv[k] += intrhoz*r11*r22*valH1 - intrhoz*r12*r21*valH1
                k += 1
                Mv[k] += intrhoz*r11*r32*valH1 - intrhoz*r12*r31*valH1
                k += 1
                Mv[k] += intrho*r11**2*valH1 + intrho*r12**2*valH1 + intrho*r13**2*valH1
                k += 1
                Mv[k] += intrho*r11*r21*valH1 + intrho*r12*r22*valH1 + intrho*r13*r23*valH1
                k += 1
                Mv[k] += intrho*r11*r31*valH1 + intrho*r12*r32*valH1 + intrho*r13*r33*valH1
                k += 1
                Mv[k] += intrhoz*r11*r22*valH1 - intrhoz*r12*r21*valH1
                k += 1
                Mv[k] += intrhoz*r11*r32*valH1 - intrhoz*r12*r31*valH1
                k += 1
                Mv[k] += intrho*r11*r21*valH1 + intrho*r12*r22*valH1 + intrho*r13*r23*valH1
                k += 1
                Mv[k] += intrho*r21**2*valH1 + intrho*r22**2*valH1 + intrho*r23**2*valH1
                k += 1
                Mv[k] += intrho*r21*r31*valH1 + intrho*r22*r32*valH1 + intrho*r23*r33*valH1
                k += 1
                Mv[k] += -intrhoz*r11*r22*valH1 + intrhoz*r12*r21*valH1
                k += 1
                Mv[k] += intrhoz*r21*r32*valH1 - intrhoz*r22*r31*valH1
                k += 1
                Mv[k] += intrho*r11*r21*valH1 + intrho*r12*r22*valH1 + intrho*r13*r23*valH1
                k += 1
                Mv[k] += intrho*r21**2*valH1 + intrho*r22**2*valH1 + intrho*r23**2*valH1
                k += 1
                Mv[k] += intrho*r21*r31*valH1 + intrho*r22*r32*valH1 + intrho*r23*r33*valH1
                k += 1
                Mv[k] += -intrhoz*r11*r22*valH1 + intrhoz*r12*r21*valH1
                k += 1
                Mv[k] += intrhoz*r21*r32*valH1 - intrhoz*r22*r31*valH1
                k += 1
                Mv[k] += intrho*r11*r21*valH1 + intrho*r12*r22*valH1 + intrho*r13*r23*valH1
                k += 1
                Mv[k] += intrho*r21**2*valH1 + intrho*r22**2*valH1 + intrho*r23**2*valH1
                k += 1
                Mv[k] += intrho*r21*r31*valH1 + intrho*r22*r32*valH1 + intrho*r23*r33*valH1
                k += 1
                Mv[k] += -intrhoz*r11*r22*valH1 + intrhoz*r12*r21*valH1
                k += 1
                Mv[k] += intrhoz*r21*r32*valH1 - intrhoz*r22*r31*valH1
                k += 1
                Mv[k] += intrho*r11*r21*valH1 + intrho*r12*r22*valH1 + intrho*r13*r23*valH1
                k += 1
                Mv[k] += intrho*r21**2*valH1 + intrho*r22**2*valH1 + intrho*r23**2*valH1
                k += 1
                Mv[k] += intrho*r21*r31*valH1 + intrho*r22*r32*valH1 + intrho*r23*r33*valH1
                k += 1
                Mv[k] += -intrhoz*r11*r22*valH1 + intrhoz*r12*r21*valH1
                k += 1
                Mv[k] += intrhoz*r21*r32*valH1 - intrhoz*r22*r31*valH1
                k += 1
                Mv[k] += intrho*r11*r31*valH1 + intrho*r12*r32*valH1 + intrho*r13*r33*valH1
                k += 1
                Mv[k] += intrho*r21*r31*valH1 + intrho*r22*r32*valH1 + intrho*r23*r33*valH1
                k += 1
                Mv[k] += intrho*r31**2*valH1 + intrho*r32**2*valH1 + intrho*r33**2*valH1
                k += 1
                Mv[k] += -intrhoz*r11*r32*valH1 + intrhoz*r12*r31*valH1
                k += 1
                Mv[k] += -intrhoz*r21*r32*valH1 + intrhoz*r22*r31*valH1
                k += 1
                Mv[k] += intrho*r11*r31*valH1 + intrho*r12*r32*valH1 + intrho*r13*r33*valH1
                k += 1
                Mv[k] += intrho*r21*r31*valH1 + intrho*r22*r32*valH1 + intrho*r23*r33*valH1
                k += 1
                Mv[k] += intrho*r31**2*valH1 + intrho*r32**2*valH1 + intrho*r33**2*valH1
                k += 1
                Mv[k] += -intrhoz*r11*r32*valH1 + intrhoz*r12*r31*valH1
                k += 1
                Mv[k] += -intrhoz*r21*r32*valH1 + intrhoz*r22*r31*valH1
                k += 1
                Mv[k] += intrho*r11*r31*valH1 + intrho*r12*r32*valH1 + intrho*r13*r33*valH1
                k += 1
                Mv[k] += intrho*r21*r31*valH1 + intrho*r22*r32*valH1 + intrho*r23*r33*valH1
                k += 1
                Mv[k] += intrho*r31**2*valH1 + intrho*r32**2*valH1 + intrho*r33**2*valH1
                k += 1
                Mv[k] += -intrhoz*r11*r32*valH1 + intrhoz*r12*r31*valH1
                k += 1
                Mv[k] += -intrhoz*r21*r32*valH1 + intrhoz*r22*r31*valH1
                k += 1
                Mv[k] += intrho*r11*r31*valH1 + intrho*r12*r32*valH1 + intrho*r13*r33*valH1
                k += 1
                Mv[k] += intrho*r21*r31*valH1 + intrho*r22*r32*valH1 + intrho*r23*r33*valH1
                k += 1
                Mv[k] += intrho*r31**2*valH1 + intrho*r32**2*valH1 + intrho*r33**2*valH1
                k += 1
                Mv[k] += -intrhoz*r11*r32*valH1 + intrhoz*r12*r31*valH1
                k += 1
                Mv[k] += -intrhoz*r21*r32*valH1 + intrhoz*r22*r31*valH1
                k += 1
                Mv[k] += -intrhoz*r11*r22*valH1 + intrhoz*r12*r21*valH1
                k += 1
                Mv[k] += -intrhoz*r11*r32*valH1 + intrhoz*r12*r31*valH1
                k += 1
                Mv[k] += intrhoz2*r11**2*valH1 + intrhoz2*r12**2*valH1
                k += 1
                Mv[k] += intrhoz2*r11*r21*valH1 + intrhoz2*r12*r22*valH1
                k += 1
                Mv[k] += intrhoz2*r11*r31*valH1 + intrhoz2*r12*r32*valH1
                k += 1
                Mv[k] += -intrhoz*r11*r22*valH1 + intrhoz*r12*r21*valH1
                k += 1
                Mv[k] += -intrhoz*r11*r32*valH1 + intrhoz*r12*r31*valH1
                k += 1
                Mv[k] += intrhoz2*r11**2*valH1 + intrhoz2*r12**2*valH1
                k += 1
                Mv[k] += intrhoz2*r11*r21*valH1 + intrhoz2*r12*r22*valH1
                k += 1
                Mv[k] += intrhoz2*r11*r31*valH1 + intrhoz2*r12*r32*valH1
                k += 1
                Mv[k] += -intrhoz*r11*r22*valH1 + intrhoz*r12*r21*valH1
                k += 1
                Mv[k] += -intrhoz*r11*r32*valH1 + intrhoz*r12*r31*valH1
                k += 1
                Mv[k] += intrhoz2*r11**2*valH1 + intrhoz2*r12**2*valH1
                k += 1
                Mv[k] += intrhoz2*r11*r21*valH1 + intrhoz2*r12*r22*valH1
                k += 1
                Mv[k] += intrhoz2*r11*r31*valH1 + intrhoz2*r12*r32*valH1
                k += 1
                Mv[k] += -intrhoz*r11*r22*valH1 + intrhoz*r12*r21*valH1
                k += 1
                Mv[k] += -intrhoz*r11*r32*valH1 + intrhoz*r12*r31*valH1
                k += 1
                Mv[k] += intrhoz2*r11**2*valH1 + intrhoz2*r12**2*valH1
                k += 1
                Mv[k] += intrhoz2*r11*r21*valH1 + intrhoz2*r12*r22*valH1
                k += 1
                Mv[k] += intrhoz2*r11*r31*valH1 + intrhoz2*r12*r32*valH1
                k += 1
                Mv[k] += intrhoz*r11*r22*valH1 - intrhoz*r12*r21*valH1
                k += 1
                Mv[k] += -intrhoz*r21*r32*valH1 + intrhoz*r22*r31*valH1
                k += 1
                Mv[k] += intrhoz2*r11*r21*valH1 + intrhoz2*r12*r22*valH1
                k += 1
                Mv[k] += intrhoz2*r21**2*valH1 + intrhoz2*r22**2*valH1
                k += 1
                Mv[k] += intrhoz2*r21*r31*valH1 + intrhoz2*r22*r32*valH1
                k += 1
                Mv[k] += intrhoz*r11*r22*valH1 - intrhoz*r12*r21*valH1
                k += 1
                Mv[k] += -intrhoz*r21*r32*valH1 + intrhoz*r22*r31*valH1
                k += 1
                Mv[k] += intrhoz2*r11*r21*valH1 + intrhoz2*r12*r22*valH1
                k += 1
                Mv[k] += intrhoz2*r21**2*valH1 + intrhoz2*r22**2*valH1
                k += 1
                Mv[k] += intrhoz2*r21*r31*valH1 + intrhoz2*r22*r32*valH1
                k += 1
                Mv[k] += intrhoz*r11*r22*valH1 - intrhoz*r12*r21*valH1
                k += 1
                Mv[k] += -intrhoz*r21*r32*valH1 + intrhoz*r22*r31*valH1
                k += 1
                Mv[k] += intrhoz2*r11*r21*valH1 + intrhoz2*r12*r22*valH1
                k += 1
                Mv[k] += intrhoz2*r21**2*valH1 + intrhoz2*r22**2*valH1
                k += 1
                Mv[k] += intrhoz2*r21*r31*valH1 + intrhoz2*r22*r32*valH1
                k += 1
                Mv[k] += intrhoz*r11*r22*valH1 - intrhoz*r12*r21*valH1
                k += 1
                Mv[k] += -intrhoz*r21*r32*valH1 + intrhoz*r22*r31*valH1
                k += 1
                Mv[k] += intrhoz2*r11*r21*valH1 + intrhoz2*r12*r22*valH1
                k += 1
                Mv[k] += intrhoz2*r21**2*valH1 + intrhoz2*r22**2*valH1
                k += 1
                Mv[k] += intrhoz2*r21*r31*valH1 + intrhoz2*r22*r32*valH1
                k += 1
                Mv[k] += intrhoz*r11*r32*valH1 - intrhoz*r12*r31*valH1
                k += 1
                Mv[k] += intrhoz*r21*r32*valH1 - intrhoz*r22*r31*valH1
                k += 1
                Mv[k] += intrhoz2*r11*r31*valH1 + intrhoz2*r12*r32*valH1
                k += 1
                Mv[k] += intrhoz2*r21*r31*valH1 + intrhoz2*r22*r32*valH1
                k += 1
                Mv[k] += intrhoz2*r31**2*valH1 + intrhoz2*r32**2*valH1
                k += 1
                Mv[k] += intrhoz*r11*r32*valH1 - intrhoz*r12*r31*valH1
                k += 1
                Mv[k] += intrhoz*r21*r32*valH1 - intrhoz*r22*r31*valH1
                k += 1
                Mv[k] += intrhoz2*r11*r31*valH1 + intrhoz2*r12*r32*valH1
                k += 1
                Mv[k] += intrhoz2*r21*r31*valH1 + intrhoz2*r22*r32*valH1
                k += 1
                Mv[k] += intrhoz2*r31**2*valH1 + intrhoz2*r32**2*valH1
                k += 1
                Mv[k] += intrhoz*r11*r32*valH1 - intrhoz*r12*r31*valH1
                k += 1
                Mv[k] += intrhoz*r21*r32*valH1 - intrhoz*r22*r31*valH1
                k += 1
                Mv[k] += intrhoz2*r11*r31*valH1 + intrhoz2*r12*r32*valH1
                k += 1
                Mv[k] += intrhoz2*r21*r31*valH1 + intrhoz2*r22*r32*valH1
                k += 1
                Mv[k] += intrhoz2*r31**2*valH1 + intrhoz2*r32**2*valH1
                k += 1
                Mv[k] += intrhoz*r11*r32*valH1 - intrhoz*r12*r31*valH1
                k += 1
                Mv[k] += intrhoz*r21*r32*valH1 - intrhoz*r22*r31*valH1
                k += 1
                Mv[k] += intrhoz2*r11*r31*valH1 + intrhoz2*r12*r32*valH1
                k += 1
                Mv[k] += intrhoz2*r21*r31*valH1 + intrhoz2*r22*r32*valH1
                k += 1
                Mv[k] += intrhoz2*r31**2*valH1 + intrhoz2*r32**2*valH1
                k += 1
                Mv[k] += intrho*r11**2*valH1 + intrho*r12**2*valH1 + intrho*r13**2*valH1
                k += 1
                Mv[k] += intrho*r11*r21*valH1 + intrho*r12*r22*valH1 + intrho*r13*r23*valH1
                k += 1
                Mv[k] += intrho*r11*r31*valH1 + intrho*r12*r32*valH1 + intrho*r13*r33*valH1
                k += 1
                Mv[k] += intrhoz*r11*r22*valH1 - intrhoz*r12*r21*valH1
                k += 1
                Mv[k] += intrhoz*r11*r32*valH1 - intrhoz*r12*r31*valH1
                k += 1
                Mv[k] += intrho*r11**2*valH1 + intrho*r12**2*valH1 + intrho*r13**2*valH1
                k += 1
                Mv[k] += intrho*r11*r21*valH1 + intrho*r12*r22*valH1 + intrho*r13*r23*valH1
                k += 1
                Mv[k] += intrho*r11*r31*valH1 + intrho*r12*r32*valH1 + intrho*r13*r33*valH1
                k += 1
                Mv[k] += intrhoz*r11*r22*valH1 - intrhoz*r12*r21*valH1
                k += 1
                Mv[k] += intrhoz*r11*r32*valH1 - intrhoz*r12*r31*valH1
                k += 1
                Mv[k] += intrho*r11**2*valH1 + intrho*r12**2*valH1 + intrho*r13**2*valH1
                k += 1
                Mv[k] += intrho*r11*r21*valH1 + intrho*r12*r22*valH1 + intrho*r13*r23*valH1
                k += 1
                Mv[k] += intrho*r11*r31*valH1 + intrho*r12*r32*valH1 + intrho*r13*r33*valH1
                k += 1
                Mv[k] += intrhoz*r11*r22*valH1 - intrhoz*r12*r21*valH1
                k += 1
                Mv[k] += intrhoz*r11*r32*valH1 - intrhoz*r12*r31*valH1
                k += 1
                Mv[k] += intrho*r11**2*valH1 + intrho*r12**2*valH1 + intrho*r13**2*valH1
                k += 1
                Mv[k] += intrho*r11*r21*valH1 + intrho*r12*r22*valH1 + intrho*r13*r23*valH1
                k += 1
                Mv[k] += intrho*r11*r31*valH1 + intrho*r12*r32*valH1 + intrho*r13*r33*valH1
                k += 1
                Mv[k] += intrhoz*r11*r22*valH1 - intrhoz*r12*r21*valH1
                k += 1
                Mv[k] += intrhoz*r11*r32*valH1 - intrhoz*r12*r31*valH1
                k += 1
                Mv[k] += intrho*r11*r21*valH1 + intrho*r12*r22*valH1 + intrho*r13*r23*valH1
                k += 1
                Mv[k] += intrho*r21**2*valH1 + intrho*r22**2*valH1 + intrho*r23**2*valH1
                k += 1
                Mv[k] += intrho*r21*r31*valH1 + intrho*r22*r32*valH1 + intrho*r23*r33*valH1
                k += 1
                Mv[k] += -intrhoz*r11*r22*valH1 + intrhoz*r12*r21*valH1
                k += 1
                Mv[k] += intrhoz*r21*r32*valH1 - intrhoz*r22*r31*valH1
                k += 1
                Mv[k] += intrho*r11*r21*valH1 + intrho*r12*r22*valH1 + intrho*r13*r23*valH1
                k += 1
                Mv[k] += intrho*r21**2*valH1 + intrho*r22**2*valH1 + intrho*r23**2*valH1
                k += 1
                Mv[k] += intrho*r21*r31*valH1 + intrho*r22*r32*valH1 + intrho*r23*r33*valH1
                k += 1
                Mv[k] += -intrhoz*r11*r22*valH1 + intrhoz*r12*r21*valH1
                k += 1
                Mv[k] += intrhoz*r21*r32*valH1 - intrhoz*r22*r31*valH1
                k += 1
                Mv[k] += intrho*r11*r21*valH1 + intrho*r12*r22*valH1 + intrho*r13*r23*valH1
                k += 1
                Mv[k] += intrho*r21**2*valH1 + intrho*r22**2*valH1 + intrho*r23**2*valH1
                k += 1
                Mv[k] += intrho*r21*r31*valH1 + intrho*r22*r32*valH1 + intrho*r23*r33*valH1
                k += 1
                Mv[k] += -intrhoz*r11*r22*valH1 + intrhoz*r12*r21*valH1
                k += 1
                Mv[k] += intrhoz*r21*r32*valH1 - intrhoz*r22*r31*valH1
                k += 1
                Mv[k] += intrho*r11*r21*valH1 + intrho*r12*r22*valH1 + intrho*r13*r23*valH1
                k += 1
                Mv[k] += intrho*r21**2*valH1 + intrho*r22**2*valH1 + intrho*r23**2*valH1
                k += 1
                Mv[k] += intrho*r21*r31*valH1 + intrho*r22*r32*valH1 + intrho*r23*r33*valH1
                k += 1
                Mv[k] += -intrhoz*r11*r22*valH1 + intrhoz*r12*r21*valH1
                k += 1
                Mv[k] += intrhoz*r21*r32*valH1 - intrhoz*r22*r31*valH1
                k += 1
                Mv[k] += intrho*r11*r31*valH1 + intrho*r12*r32*valH1 + intrho*r13*r33*valH1
                k += 1
                Mv[k] += intrho*r21*r31*valH1 + intrho*r22*r32*valH1 + intrho*r23*r33*valH1
                k += 1
                Mv[k] += intrho*r31**2*valH1 + intrho*r32**2*valH1 + intrho*r33**2*valH1
                k += 1
                Mv[k] += -intrhoz*r11*r32*valH1 + intrhoz*r12*r31*valH1
                k += 1
                Mv[k] += -intrhoz*r21*r32*valH1 + intrhoz*r22*r31*valH1
                k += 1
                Mv[k] += intrho*r11*r31*valH1 + intrho*r12*r32*valH1 + intrho*r13*r33*valH1
                k += 1
                Mv[k] += intrho*r21*r31*valH1 + intrho*r22*r32*valH1 + intrho*r23*r33*valH1
                k += 1
                Mv[k] += intrho*r31**2*valH1 + intrho*r32**2*valH1 + intrho*r33**2*valH1
                k += 1
                Mv[k] += -intrhoz*r11*r32*valH1 + intrhoz*r12*r31*valH1
                k += 1
                Mv[k] += -intrhoz*r21*r32*valH1 + intrhoz*r22*r31*valH1
                k += 1
                Mv[k] += intrho*r11*r31*valH1 + intrho*r12*r32*valH1 + intrho*r13*r33*valH1
                k += 1
                Mv[k] += intrho*r21*r31*valH1 + intrho*r22*r32*valH1 + intrho*r23*r33*valH1
                k += 1
                Mv[k] += intrho*r31**2*valH1 + intrho*r32**2*valH1 + intrho*r33**2*valH1
                k += 1
                Mv[k] += -intrhoz*r11*r32*valH1 + intrhoz*r12*r31*valH1
                k += 1
                Mv[k] += -intrhoz*r21*r32*valH1 + intrhoz*r22*r31*valH1
                k += 1
                Mv[k] += intrho*r11*r31*valH1 + intrho*r12*r32*valH1 + intrho*r13*r33*valH1
                k += 1
                Mv[k] += intrho*r21*r31*valH1 + intrho*r22*r32*valH1 + intrho*r23*r33*valH1
                k += 1
                Mv[k] += intrho*r31**2*valH1 + intrho*r32**2*valH1 + intrho*r33**2*valH1
                k += 1
                Mv[k] += -intrhoz*r11*r32*valH1 + intrhoz*r12*r31*valH1
                k += 1
                Mv[k] += -intrhoz*r21*r32*valH1 + intrhoz*r22*r31*valH1
                k += 1
                Mv[k] += -intrhoz*r11*r22*valH1 + intrhoz*r12*r21*valH1
                k += 1
                Mv[k] += -intrhoz*r11*r32*valH1 + intrhoz*r12*r31*valH1
                k += 1
                Mv[k] += intrhoz2*r11**2*valH1 + intrhoz2*r12**2*valH1
                k += 1
                Mv[k] += intrhoz2*r11*r21*valH1 + intrhoz2*r12*r22*valH1
                k += 1
                Mv[k] += intrhoz2*r11*r31*valH1 + intrhoz2*r12*r32*valH1
                k += 1
                Mv[k] += -intrhoz*r11*r22*valH1 + intrhoz*r12*r21*valH1
                k += 1
                Mv[k] += -intrhoz*r11*r32*valH1 + intrhoz*r12*r31*valH1
                k += 1
                Mv[k] += intrhoz2*r11**2*valH1 + intrhoz2*r12**2*valH1
                k += 1
                Mv[k] += intrhoz2*r11*r21*valH1 + intrhoz2*r12*r22*valH1
                k += 1
                Mv[k] += intrhoz2*r11*r31*valH1 + intrhoz2*r12*r32*valH1
                k += 1
                Mv[k] += -intrhoz*r11*r22*valH1 + intrhoz*r12*r21*valH1
                k += 1
                Mv[k] += -intrhoz*r11*r32*valH1 + intrhoz*r12*r31*valH1
                k += 1
                Mv[k] += intrhoz2*r11**2*valH1 + intrhoz2*r12**2*valH1
                k += 1
                Mv[k] += intrhoz2*r11*r21*valH1 + intrhoz2*r12*r22*valH1
                k += 1
                Mv[k] += intrhoz2*r11*r31*valH1 + intrhoz2*r12*r32*valH1
                k += 1
                Mv[k] += -intrhoz*r11*r22*valH1 + intrhoz*r12*r21*valH1
                k += 1
                Mv[k] += -intrhoz*r11*r32*valH1 + intrhoz*r12*r31*valH1
                k += 1
                Mv[k] += intrhoz2*r11**2*valH1 + intrhoz2*r12**2*valH1
                k += 1
                Mv[k] += intrhoz2*r11*r21*valH1 + intrhoz2*r12*r22*valH1
                k += 1
                Mv[k] += intrhoz2*r11*r31*valH1 + intrhoz2*r12*r32*valH1
                k += 1
                Mv[k] += intrhoz*r11*r22*valH1 - intrhoz*r12*r21*valH1
                k += 1
                Mv[k] += -intrhoz*r21*r32*valH1 + intrhoz*r22*r31*valH1
                k += 1
                Mv[k] += intrhoz2*r11*r21*valH1 + intrhoz2*r12*r22*valH1
                k += 1
                Mv[k] += intrhoz2*r21**2*valH1 + intrhoz2*r22**2*valH1
                k += 1
                Mv[k] += intrhoz2*r21*r31*valH1 + intrhoz2*r22*r32*valH1
                k += 1
                Mv[k] += intrhoz*r11*r22*valH1 - intrhoz*r12*r21*valH1
                k += 1
                Mv[k] += -intrhoz*r21*r32*valH1 + intrhoz*r22*r31*valH1
                k += 1
                Mv[k] += intrhoz2*r11*r21*valH1 + intrhoz2*r12*r22*valH1
                k += 1
                Mv[k] += intrhoz2*r21**2*valH1 + intrhoz2*r22**2*valH1
                k += 1
                Mv[k] += intrhoz2*r21*r31*valH1 + intrhoz2*r22*r32*valH1
                k += 1
                Mv[k] += intrhoz*r11*r22*valH1 - intrhoz*r12*r21*valH1
                k += 1
                Mv[k] += -intrhoz*r21*r32*valH1 + intrhoz*r22*r31*valH1
                k += 1
                Mv[k] += intrhoz2*r11*r21*valH1 + intrhoz2*r12*r22*valH1
                k += 1
                Mv[k] += intrhoz2*r21**2*valH1 + intrhoz2*r22**2*valH1
                k += 1
                Mv[k] += intrhoz2*r21*r31*valH1 + intrhoz2*r22*r32*valH1
                k += 1
                Mv[k] += intrhoz*r11*r22*valH1 - intrhoz*r12*r21*valH1
                k += 1
                Mv[k] += -intrhoz*r21*r32*valH1 + intrhoz*r22*r31*valH1
                k += 1
                Mv[k] += intrhoz2*r11*r21*valH1 + intrhoz2*r12*r22*valH1
                k += 1
                Mv[k] += intrhoz2*r21**2*valH1 + intrhoz2*r22**2*valH1
                k += 1
                Mv[k] += intrhoz2*r21*r31*valH1 + intrhoz2*r22*r32*valH1
                k += 1
                Mv[k] += intrhoz*r11*r32*valH1 - intrhoz*r12*r31*valH1
                k += 1
                Mv[k] += intrhoz*r21*r32*valH1 - intrhoz*r22*r31*valH1
                k += 1
                Mv[k] += intrhoz2*r11*r31*valH1 + intrhoz2*r12*r32*valH1
                k += 1
                Mv[k] += intrhoz2*r21*r31*valH1 + intrhoz2*r22*r32*valH1
                k += 1
                Mv[k] += intrhoz2*r31**2*valH1 + intrhoz2*r32**2*valH1
                k += 1
                Mv[k] += intrhoz*r11*r32*valH1 - intrhoz*r12*r31*valH1
                k += 1
                Mv[k] += intrhoz*r21*r32*valH1 - intrhoz*r22*r31*valH1
                k += 1
                Mv[k] += intrhoz2*r11*r31*valH1 + intrhoz2*r12*r32*valH1
                k += 1
                Mv[k] += intrhoz2*r21*r31*valH1 + intrhoz2*r22*r32*valH1
                k += 1
                Mv[k] += intrhoz2*r31**2*valH1 + intrhoz2*r32**2*valH1
                k += 1
                Mv[k] += intrhoz*r11*r32*valH1 - intrhoz*r12*r31*valH1
                k += 1
                Mv[k] += intrhoz*r21*r32*valH1 - intrhoz*r22*r31*valH1
                k += 1
                Mv[k] += intrhoz2*r11*r31*valH1 + intrhoz2*r12*r32*valH1
                k += 1
                Mv[k] += intrhoz2*r21*r31*valH1 + intrhoz2*r22*r32*valH1
                k += 1
                Mv[k] += intrhoz2*r31**2*valH1 + intrhoz2*r32**2*valH1
                k += 1
                Mv[k] += intrhoz*r11*r32*valH1 - intrhoz*r12*r31*valH1
                k += 1
                Mv[k] += intrhoz*r21*r32*valH1 - intrhoz*r22*r31*valH1
                k += 1
                Mv[k] += intrhoz2*r11*r31*valH1 + intrhoz2*r12*r32*valH1
                k += 1
                Mv[k] += intrhoz2*r21*r31*valH1 + intrhoz2*r22*r32*valH1
                k += 1
                Mv[k] += intrhoz2*r31**2*valH1 + intrhoz2*r32**2*valH1
                k += 1
                Mv[k] += intrho*r11**2*valH1 + intrho*r12**2*valH1 + intrho*r13**2*valH1
                k += 1
                Mv[k] += intrho*r11*r21*valH1 + intrho*r12*r22*valH1 + intrho*r13*r23*valH1
                k += 1
                Mv[k] += intrho*r11*r31*valH1 + intrho*r12*r32*valH1 + intrho*r13*r33*valH1
                k += 1
                Mv[k] += intrhoz*r11*r22*valH1 - intrhoz*r12*r21*valH1
                k += 1
                Mv[k] += intrhoz*r11*r32*valH1 - intrhoz*r12*r31*valH1
                k += 1
                Mv[k] += intrho*r11**2*valH1 + intrho*r12**2*valH1 + intrho*r13**2*valH1
                k += 1
                Mv[k] += intrho*r11*r21*valH1 + intrho*r12*r22*valH1 + intrho*r13*r23*valH1
                k += 1
                Mv[k] += intrho*r11*r31*valH1 + intrho*r12*r32*valH1 + intrho*r13*r33*valH1
                k += 1
                Mv[k] += intrhoz*r11*r22*valH1 - intrhoz*r12*r21*valH1
                k += 1
                Mv[k] += intrhoz*r11*r32*valH1 - intrhoz*r12*r31*valH1
                k += 1
                Mv[k] += intrho*r11**2*valH1 + intrho*r12**2*valH1 + intrho*r13**2*valH1
                k += 1
                Mv[k] += intrho*r11*r21*valH1 + intrho*r12*r22*valH1 + intrho*r13*r23*valH1
                k += 1
                Mv[k] += intrho*r11*r31*valH1 + intrho*r12*r32*valH1 + intrho*r13*r33*valH1
                k += 1
                Mv[k] += intrhoz*r11*r22*valH1 - intrhoz*r12*r21*valH1
                k += 1
                Mv[k] += intrhoz*r11*r32*valH1 - intrhoz*r12*r31*valH1
                k += 1
                Mv[k] += intrho*r11**2*valH1 + intrho*r12**2*valH1 + intrho*r13**2*valH1
                k += 1
                Mv[k] += intrho*r11*r21*valH1 + intrho*r12*r22*valH1 + intrho*r13*r23*valH1
                k += 1
                Mv[k] += intrho*r11*r31*valH1 + intrho*r12*r32*valH1 + intrho*r13*r33*valH1
                k += 1
                Mv[k] += intrhoz*r11*r22*valH1 - intrhoz*r12*r21*valH1
                k += 1
                Mv[k] += intrhoz*r11*r32*valH1 - intrhoz*r12*r31*valH1
                k += 1
                Mv[k] += intrho*r11*r21*valH1 + intrho*r12*r22*valH1 + intrho*r13*r23*valH1
                k += 1
                Mv[k] += intrho*r21**2*valH1 + intrho*r22**2*valH1 + intrho*r23**2*valH1
                k += 1
                Mv[k] += intrho*r21*r31*valH1 + intrho*r22*r32*valH1 + intrho*r23*r33*valH1
                k += 1
                Mv[k] += -intrhoz*r11*r22*valH1 + intrhoz*r12*r21*valH1
                k += 1
                Mv[k] += intrhoz*r21*r32*valH1 - intrhoz*r22*r31*valH1
                k += 1
                Mv[k] += intrho*r11*r21*valH1 + intrho*r12*r22*valH1 + intrho*r13*r23*valH1
                k += 1
                Mv[k] += intrho*r21**2*valH1 + intrho*r22**2*valH1 + intrho*r23**2*valH1
                k += 1
                Mv[k] += intrho*r21*r31*valH1 + intrho*r22*r32*valH1 + intrho*r23*r33*valH1
                k += 1
                Mv[k] += -intrhoz*r11*r22*valH1 + intrhoz*r12*r21*valH1
                k += 1
                Mv[k] += intrhoz*r21*r32*valH1 - intrhoz*r22*r31*valH1
                k += 1
                Mv[k] += intrho*r11*r21*valH1 + intrho*r12*r22*valH1 + intrho*r13*r23*valH1
                k += 1
                Mv[k] += intrho*r21**2*valH1 + intrho*r22**2*valH1 + intrho*r23**2*valH1
                k += 1
                Mv[k] += intrho*r21*r31*valH1 + intrho*r22*r32*valH1 + intrho*r23*r33*valH1
                k += 1
                Mv[k] += -intrhoz*r11*r22*valH1 + intrhoz*r12*r21*valH1
                k += 1
                Mv[k] += intrhoz*r21*r32*valH1 - intrhoz*r22*r31*valH1
                k += 1
                Mv[k] += intrho*r11*r21*valH1 + intrho*r12*r22*valH1 + intrho*r13*r23*valH1
                k += 1
                Mv[k] += intrho*r21**2*valH1 + intrho*r22**2*valH1 + intrho*r23**2*valH1
                k += 1
                Mv[k] += intrho*r21*r31*valH1 + intrho*r22*r32*valH1 + intrho*r23*r33*valH1
                k += 1
                Mv[k] += -intrhoz*r11*r22*valH1 + intrhoz*r12*r21*valH1
                k += 1
                Mv[k] += intrhoz*r21*r32*valH1 - intrhoz*r22*r31*valH1
                k += 1
                Mv[k] += intrho*r11*r31*valH1 + intrho*r12*r32*valH1 + intrho*r13*r33*valH1
                k += 1
                Mv[k] += intrho*r21*r31*valH1 + intrho*r22*r32*valH1 + intrho*r23*r33*valH1
                k += 1
                Mv[k] += intrho*r31**2*valH1 + intrho*r32**2*valH1 + intrho*r33**2*valH1
                k += 1
                Mv[k] += -intrhoz*r11*r32*valH1 + intrhoz*r12*r31*valH1
                k += 1
                Mv[k] += -intrhoz*r21*r32*valH1 + intrhoz*r22*r31*valH1
                k += 1
                Mv[k] += intrho*r11*r31*valH1 + intrho*r12*r32*valH1 + intrho*r13*r33*valH1
                k += 1
                Mv[k] += intrho*r21*r31*valH1 + intrho*r22*r32*valH1 + intrho*r23*r33*valH1
                k += 1
                Mv[k] += intrho*r31**2*valH1 + intrho*r32**2*valH1 + intrho*r33**2*valH1
                k += 1
                Mv[k] += -intrhoz*r11*r32*valH1 + intrhoz*r12*r31*valH1
                k += 1
                Mv[k] += -intrhoz*r21*r32*valH1 + intrhoz*r22*r31*valH1
                k += 1
                Mv[k] += intrho*r11*r31*valH1 + intrho*r12*r32*valH1 + intrho*r13*r33*valH1
                k += 1
                Mv[k] += intrho*r21*r31*valH1 + intrho*r22*r32*valH1 + intrho*r23*r33*valH1
                k += 1
                Mv[k] += intrho*r31**2*valH1 + intrho*r32**2*valH1 + intrho*r33**2*valH1
                k += 1
                Mv[k] += -intrhoz*r11*r32*valH1 + intrhoz*r12*r31*valH1
                k += 1
                Mv[k] += -intrhoz*r21*r32*valH1 + intrhoz*r22*r31*valH1
                k += 1
                Mv[k] += intrho*r11*r31*valH1 + intrho*r12*r32*valH1 + intrho*r13*r33*valH1
                k += 1
                Mv[k] += intrho*r21*r31*valH1 + intrho*r22*r32*valH1 + intrho*r23*r33*valH1
                k += 1
                Mv[k] += intrho*r31**2*valH1 + intrho*r32**2*valH1 + intrho*r33**2*valH1
                k += 1
                Mv[k] += -intrhoz*r11*r32*valH1 + intrhoz*r12*r31*valH1
                k += 1
                Mv[k] += -intrhoz*r21*r32*valH1 + intrhoz*r22*r31*valH1
                k += 1
                Mv[k] += -intrhoz*r11*r22*valH1 + intrhoz*r12*r21*valH1
                k += 1
                Mv[k] += -intrhoz*r11*r32*valH1 + intrhoz*r12*r31*valH1
                k += 1
                Mv[k] += intrhoz2*r11**2*valH1 + intrhoz2*r12**2*valH1
                k += 1
                Mv[k] += intrhoz2*r11*r21*valH1 + intrhoz2*r12*r22*valH1
                k += 1
                Mv[k] += intrhoz2*r11*r31*valH1 + intrhoz2*r12*r32*valH1
                k += 1
                Mv[k] += -intrhoz*r11*r22*valH1 + intrhoz*r12*r21*valH1
                k += 1
                Mv[k] += -intrhoz*r11*r32*valH1 + intrhoz*r12*r31*valH1
                k += 1
                Mv[k] += intrhoz2*r11**2*valH1 + intrhoz2*r12**2*valH1
                k += 1
                Mv[k] += intrhoz2*r11*r21*valH1 + intrhoz2*r12*r22*valH1
                k += 1
                Mv[k] += intrhoz2*r11*r31*valH1 + intrhoz2*r12*r32*valH1
                k += 1
                Mv[k] += -intrhoz*r11*r22*valH1 + intrhoz*r12*r21*valH1
                k += 1
                Mv[k] += -intrhoz*r11*r32*valH1 + intrhoz*r12*r31*valH1
                k += 1
                Mv[k] += intrhoz2*r11**2*valH1 + intrhoz2*r12**2*valH1
                k += 1
                Mv[k] += intrhoz2*r11*r21*valH1 + intrhoz2*r12*r22*valH1
                k += 1
                Mv[k] += intrhoz2*r11*r31*valH1 + intrhoz2*r12*r32*valH1
                k += 1
                Mv[k] += -intrhoz*r11*r22*valH1 + intrhoz*r12*r21*valH1
                k += 1
                Mv[k] += -intrhoz*r11*r32*valH1 + intrhoz*r12*r31*valH1
                k += 1
                Mv[k] += intrhoz2*r11**2*valH1 + intrhoz2*r12**2*valH1
                k += 1
                Mv[k] += intrhoz2*r11*r21*valH1 + intrhoz2*r12*r22*valH1
                k += 1
                Mv[k] += intrhoz2*r11*r31*valH1 + intrhoz2*r12*r32*valH1
                k += 1
                Mv[k] += intrhoz*r11*r22*valH1 - intrhoz*r12*r21*valH1
                k += 1
                Mv[k] += -intrhoz*r21*r32*valH1 + intrhoz*r22*r31*valH1
                k += 1
                Mv[k] += intrhoz2*r11*r21*valH1 + intrhoz2*r12*r22*valH1
                k += 1
                Mv[k] += intrhoz2*r21**2*valH1 + intrhoz2*r22**2*valH1
                k += 1
                Mv[k] += intrhoz2*r21*r31*valH1 + intrhoz2*r22*r32*valH1
                k += 1
                Mv[k] += intrhoz*r11*r22*valH1 - intrhoz*r12*r21*valH1
                k += 1
                Mv[k] += -intrhoz*r21*r32*valH1 + intrhoz*r22*r31*valH1
                k += 1
                Mv[k] += intrhoz2*r11*r21*valH1 + intrhoz2*r12*r22*valH1
                k += 1
                Mv[k] += intrhoz2*r21**2*valH1 + intrhoz2*r22**2*valH1
                k += 1
                Mv[k] += intrhoz2*r21*r31*valH1 + intrhoz2*r22*r32*valH1
                k += 1
                Mv[k] += intrhoz*r11*r22*valH1 - intrhoz*r12*r21*valH1
                k += 1
                Mv[k] += -intrhoz*r21*r32*valH1 + intrhoz*r22*r31*valH1
                k += 1
                Mv[k] += intrhoz2*r11*r21*valH1 + intrhoz2*r12*r22*valH1
                k += 1
                Mv[k] += intrhoz2*r21**2*valH1 + intrhoz2*r22**2*valH1
                k += 1
                Mv[k] += intrhoz2*r21*r31*valH1 + intrhoz2*r22*r32*valH1
                k += 1
                Mv[k] += intrhoz*r11*r22*valH1 - intrhoz*r12*r21*valH1
                k += 1
                Mv[k] += -intrhoz*r21*r32*valH1 + intrhoz*r22*r31*valH1
                k += 1
                Mv[k] += intrhoz2*r11*r21*valH1 + intrhoz2*r12*r22*valH1
                k += 1
                Mv[k] += intrhoz2*r21**2*valH1 + intrhoz2*r22**2*valH1
                k += 1
                Mv[k] += intrhoz2*r21*r31*valH1 + intrhoz2*r22*r32*valH1
                k += 1
                Mv[k] += intrhoz*r11*r32*valH1 - intrhoz*r12*r31*valH1
                k += 1
                Mv[k] += intrhoz*r21*r32*valH1 - intrhoz*r22*r31*valH1
                k += 1
                Mv[k] += intrhoz2*r11*r31*valH1 + intrhoz2*r12*r32*valH1
                k += 1
                Mv[k] += intrhoz2*r21*r31*valH1 + intrhoz2*r22*r32*valH1
                k += 1
                Mv[k] += intrhoz2*r31**2*valH1 + intrhoz2*r32**2*valH1
                k += 1
                Mv[k] += intrhoz*r11*r32*valH1 - intrhoz*r12*r31*valH1
                k += 1
                Mv[k] += intrhoz*r21*r32*valH1 - intrhoz*r22*r31*valH1
                k += 1
                Mv[k] += intrhoz2*r11*r31*valH1 + intrhoz2*r12*r32*valH1
                k += 1
                Mv[k] += intrhoz2*r21*r31*valH1 + intrhoz2*r22*r32*valH1
                k += 1
                Mv[k] += intrhoz2*r31**2*valH1 + intrhoz2*r32**2*valH1
                k += 1
                Mv[k] += intrhoz*r11*r32*valH1 - intrhoz*r12*r31*valH1
                k += 1
                Mv[k] += intrhoz*r21*r32*valH1 - intrhoz*r22*r31*valH1
                k += 1
                Mv[k] += intrhoz2*r11*r31*valH1 + intrhoz2*r12*r32*valH1
                k += 1
                Mv[k] += intrhoz2*r21*r31*valH1 + intrhoz2*r22*r32*valH1
                k += 1
                Mv[k] += intrhoz2*r31**2*valH1 + intrhoz2*r32**2*valH1
                k += 1
                Mv[k] += intrhoz*r11*r32*valH1 - intrhoz*r12*r31*valH1
                k += 1
                Mv[k] += intrhoz*r21*r32*valH1 - intrhoz*r22*r31*valH1
                k += 1
                Mv[k] += intrhoz2*r11*r31*valH1 + intrhoz2*r12*r32*valH1
                k += 1
                Mv[k] += intrhoz2*r21*r31*valH1 + intrhoz2*r22*r32*valH1
                k += 1
                Mv[k] += intrhoz2*r31**2*valH1 + intrhoz2*r32**2*valH1
                k += 1
                Mv[k] += intrho*r11**2*valH1 + intrho*r12**2*valH1 + intrho*r13**2*valH1
                k += 1
                Mv[k] += intrho*r11*r21*valH1 + intrho*r12*r22*valH1 + intrho*r13*r23*valH1
                k += 1
                Mv[k] += intrho*r11*r31*valH1 + intrho*r12*r32*valH1 + intrho*r13*r33*valH1
                k += 1
                Mv[k] += intrhoz*r11*r22*valH1 - intrhoz*r12*r21*valH1
                k += 1
                Mv[k] += intrhoz*r11*r32*valH1 - intrhoz*r12*r31*valH1
                k += 1
                Mv[k] += intrho*r11**2*valH1 + intrho*r12**2*valH1 + intrho*r13**2*valH1
                k += 1
                Mv[k] += intrho*r11*r21*valH1 + intrho*r12*r22*valH1 + intrho*r13*r23*valH1
                k += 1
                Mv[k] += intrho*r11*r31*valH1 + intrho*r12*r32*valH1 + intrho*r13*r33*valH1
                k += 1
                Mv[k] += intrhoz*r11*r22*valH1 - intrhoz*r12*r21*valH1
                k += 1
                Mv[k] += intrhoz*r11*r32*valH1 - intrhoz*r12*r31*valH1
                k += 1
                Mv[k] += intrho*r11**2*valH1 + intrho*r12**2*valH1 + intrho*r13**2*valH1
                k += 1
                Mv[k] += intrho*r11*r21*valH1 + intrho*r12*r22*valH1 + intrho*r13*r23*valH1
                k += 1
                Mv[k] += intrho*r11*r31*valH1 + intrho*r12*r32*valH1 + intrho*r13*r33*valH1
                k += 1
                Mv[k] += intrhoz*r11*r22*valH1 - intrhoz*r12*r21*valH1
                k += 1
                Mv[k] += intrhoz*r11*r32*valH1 - intrhoz*r12*r31*valH1
                k += 1
                Mv[k] += intrho*r11**2*valH1 + intrho*r12**2*valH1 + intrho*r13**2*valH1
                k += 1
                Mv[k] += intrho*r11*r21*valH1 + intrho*r12*r22*valH1 + intrho*r13*r23*valH1
                k += 1
                Mv[k] += intrho*r11*r31*valH1 + intrho*r12*r32*valH1 + intrho*r13*r33*valH1
                k += 1
                Mv[k] += intrhoz*r11*r22*valH1 - intrhoz*r12*r21*valH1
                k += 1
                Mv[k] += intrhoz*r11*r32*valH1 - intrhoz*r12*r31*valH1
                k += 1
                Mv[k] += intrho*r11*r21*valH1 + intrho*r12*r22*valH1 + intrho*r13*r23*valH1
                k += 1
                Mv[k] += intrho*r21**2*valH1 + intrho*r22**2*valH1 + intrho*r23**2*valH1
                k += 1
                Mv[k] += intrho*r21*r31*valH1 + intrho*r22*r32*valH1 + intrho*r23*r33*valH1
                k += 1
                Mv[k] += -intrhoz*r11*r22*valH1 + intrhoz*r12*r21*valH1
                k += 1
                Mv[k] += intrhoz*r21*r32*valH1 - intrhoz*r22*r31*valH1
                k += 1
                Mv[k] += intrho*r11*r21*valH1 + intrho*r12*r22*valH1 + intrho*r13*r23*valH1
                k += 1
                Mv[k] += intrho*r21**2*valH1 + intrho*r22**2*valH1 + intrho*r23**2*valH1
                k += 1
                Mv[k] += intrho*r21*r31*valH1 + intrho*r22*r32*valH1 + intrho*r23*r33*valH1
                k += 1
                Mv[k] += -intrhoz*r11*r22*valH1 + intrhoz*r12*r21*valH1
                k += 1
                Mv[k] += intrhoz*r21*r32*valH1 - intrhoz*r22*r31*valH1
                k += 1
                Mv[k] += intrho*r11*r21*valH1 + intrho*r12*r22*valH1 + intrho*r13*r23*valH1
                k += 1
                Mv[k] += intrho*r21**2*valH1 + intrho*r22**2*valH1 + intrho*r23**2*valH1
                k += 1
                Mv[k] += intrho*r21*r31*valH1 + intrho*r22*r32*valH1 + intrho*r23*r33*valH1
                k += 1
                Mv[k] += -intrhoz*r11*r22*valH1 + intrhoz*r12*r21*valH1
                k += 1
                Mv[k] += intrhoz*r21*r32*valH1 - intrhoz*r22*r31*valH1
                k += 1
                Mv[k] += intrho*r11*r21*valH1 + intrho*r12*r22*valH1 + intrho*r13*r23*valH1
                k += 1
                Mv[k] += intrho*r21**2*valH1 + intrho*r22**2*valH1 + intrho*r23**2*valH1
                k += 1
                Mv[k] += intrho*r21*r31*valH1 + intrho*r22*r32*valH1 + intrho*r23*r33*valH1
                k += 1
                Mv[k] += -intrhoz*r11*r22*valH1 + intrhoz*r12*r21*valH1
                k += 1
                Mv[k] += intrhoz*r21*r32*valH1 - intrhoz*r22*r31*valH1
                k += 1
                Mv[k] += intrho*r11*r31*valH1 + intrho*r12*r32*valH1 + intrho*r13*r33*valH1
                k += 1
                Mv[k] += intrho*r21*r31*valH1 + intrho*r22*r32*valH1 + intrho*r23*r33*valH1
                k += 1
                Mv[k] += intrho*r31**2*valH1 + intrho*r32**2*valH1 + intrho*r33**2*valH1
                k += 1
                Mv[k] += -intrhoz*r11*r32*valH1 + intrhoz*r12*r31*valH1
                k += 1
                Mv[k] += -intrhoz*r21*r32*valH1 + intrhoz*r22*r31*valH1
                k += 1
                Mv[k] += intrho*r11*r31*valH1 + intrho*r12*r32*valH1 + intrho*r13*r33*valH1
                k += 1
                Mv[k] += intrho*r21*r31*valH1 + intrho*r22*r32*valH1 + intrho*r23*r33*valH1
                k += 1
                Mv[k] += intrho*r31**2*valH1 + intrho*r32**2*valH1 + intrho*r33**2*valH1
                k += 1
                Mv[k] += -intrhoz*r11*r32*valH1 + intrhoz*r12*r31*valH1
                k += 1
                Mv[k] += -intrhoz*r21*r32*valH1 + intrhoz*r22*r31*valH1
                k += 1
                Mv[k] += intrho*r11*r31*valH1 + intrho*r12*r32*valH1 + intrho*r13*r33*valH1
                k += 1
                Mv[k] += intrho*r21*r31*valH1 + intrho*r22*r32*valH1 + intrho*r23*r33*valH1
                k += 1
                Mv[k] += intrho*r31**2*valH1 + intrho*r32**2*valH1 + intrho*r33**2*valH1
                k += 1
                Mv[k] += -intrhoz*r11*r32*valH1 + intrhoz*r12*r31*valH1
                k += 1
                Mv[k] += -intrhoz*r21*r32*valH1 + intrhoz*r22*r31*valH1
                k += 1
                Mv[k] += intrho*r11*r31*valH1 + intrho*r12*r32*valH1 + intrho*r13*r33*valH1
                k += 1
                Mv[k] += intrho*r21*r31*valH1 + intrho*r22*r32*valH1 + intrho*r23*r33*valH1
                k += 1
                Mv[k] += intrho*r31**2*valH1 + intrho*r32**2*valH1 + intrho*r33**2*valH1
                k += 1
                Mv[k] += -intrhoz*r11*r32*valH1 + intrhoz*r12*r31*valH1
                k += 1
                Mv[k] += -intrhoz*r21*r32*valH1 + intrhoz*r22*r31*valH1
                k += 1
                Mv[k] += -intrhoz*r11*r22*valH1 + intrhoz*r12*r21*valH1
                k += 1
                Mv[k] += -intrhoz*r11*r32*valH1 + intrhoz*r12*r31*valH1
                k += 1
                Mv[k] += intrhoz2*r11**2*valH1 + intrhoz2*r12**2*valH1
                k += 1
                Mv[k] += intrhoz2*r11*r21*valH1 + intrhoz2*r12*r22*valH1
                k += 1
                Mv[k] += intrhoz2*r11*r31*valH1 + intrhoz2*r12*r32*valH1
                k += 1
                Mv[k] += -intrhoz*r11*r22*valH1 + intrhoz*r12*r21*valH1
                k += 1
                Mv[k] += -intrhoz*r11*r32*valH1 + intrhoz*r12*r31*valH1
                k += 1
                Mv[k] += intrhoz2*r11**2*valH1 + intrhoz2*r12**2*valH1
                k += 1
                Mv[k] += intrhoz2*r11*r21*valH1 + intrhoz2*r12*r22*valH1
                k += 1
                Mv[k] += intrhoz2*r11*r31*valH1 + intrhoz2*r12*r32*valH1
                k += 1
                Mv[k] += -intrhoz*r11*r22*valH1 + intrhoz*r12*r21*valH1
                k += 1
                Mv[k] += -intrhoz*r11*r32*valH1 + intrhoz*r12*r31*valH1
                k += 1
                Mv[k] += intrhoz2*r11**2*valH1 + intrhoz2*r12**2*valH1
                k += 1
                Mv[k] += intrhoz2*r11*r21*valH1 + intrhoz2*r12*r22*valH1
                k += 1
                Mv[k] += intrhoz2*r11*r31*valH1 + intrhoz2*r12*r32*valH1
                k += 1
                Mv[k] += -intrhoz*r11*r22*valH1 + intrhoz*r12*r21*valH1
                k += 1
                Mv[k] += -intrhoz*r11*r32*valH1 + intrhoz*r12*r31*valH1
                k += 1
                Mv[k] += intrhoz2*r11**2*valH1 + intrhoz2*r12**2*valH1
                k += 1
                Mv[k] += intrhoz2*r11*r21*valH1 + intrhoz2*r12*r22*valH1
                k += 1
                Mv[k] += intrhoz2*r11*r31*valH1 + intrhoz2*r12*r32*valH1
                k += 1
                Mv[k] += intrhoz*r11*r22*valH1 - intrhoz*r12*r21*valH1
                k += 1
                Mv[k] += -intrhoz*r21*r32*valH1 + intrhoz*r22*r31*valH1
                k += 1
                Mv[k] += intrhoz2*r11*r21*valH1 + intrhoz2*r12*r22*valH1
                k += 1
                Mv[k] += intrhoz2*r21**2*valH1 + intrhoz2*r22**2*valH1
                k += 1
                Mv[k] += intrhoz2*r21*r31*valH1 + intrhoz2*r22*r32*valH1
                k += 1
                Mv[k] += intrhoz*r11*r22*valH1 - intrhoz*r12*r21*valH1
                k += 1
                Mv[k] += -intrhoz*r21*r32*valH1 + intrhoz*r22*r31*valH1
                k += 1
                Mv[k] += intrhoz2*r11*r21*valH1 + intrhoz2*r12*r22*valH1
                k += 1
                Mv[k] += intrhoz2*r21**2*valH1 + intrhoz2*r22**2*valH1
                k += 1
                Mv[k] += intrhoz2*r21*r31*valH1 + intrhoz2*r22*r32*valH1
                k += 1
                Mv[k] += intrhoz*r11*r22*valH1 - intrhoz*r12*r21*valH1
                k += 1
                Mv[k] += -intrhoz*r21*r32*valH1 + intrhoz*r22*r31*valH1
                k += 1
                Mv[k] += intrhoz2*r11*r21*valH1 + intrhoz2*r12*r22*valH1
                k += 1
                Mv[k] += intrhoz2*r21**2*valH1 + intrhoz2*r22**2*valH1
                k += 1
                Mv[k] += intrhoz2*r21*r31*valH1 + intrhoz2*r22*r32*valH1
                k += 1
                Mv[k] += intrhoz*r11*r22*valH1 - intrhoz*r12*r21*valH1
                k += 1
                Mv[k] += -intrhoz*r21*r32*valH1 + intrhoz*r22*r31*valH1
                k += 1
                Mv[k] += intrhoz2*r11*r21*valH1 + intrhoz2*r12*r22*valH1
                k += 1
                Mv[k] += intrhoz2*r21**2*valH1 + intrhoz2*r22**2*valH1
                k += 1
                Mv[k] += intrhoz2*r21*r31*valH1 + intrhoz2*r22*r32*valH1
                k += 1
                Mv[k] += intrhoz*r11*r32*valH1 - intrhoz*r12*r31*valH1
                k += 1
                Mv[k] += intrhoz*r21*r32*valH1 - intrhoz*r22*r31*valH1
                k += 1
                Mv[k] += intrhoz2*r11*r31*valH1 + intrhoz2*r12*r32*valH1
                k += 1
                Mv[k] += intrhoz2*r21*r31*valH1 + intrhoz2*r22*r32*valH1
                k += 1
                Mv[k] += intrhoz2*r31**2*valH1 + intrhoz2*r32**2*valH1
                k += 1
                Mv[k] += intrhoz*r11*r32*valH1 - intrhoz*r12*r31*valH1
                k += 1
                Mv[k] += intrhoz*r21*r32*valH1 - intrhoz*r22*r31*valH1
                k += 1
                Mv[k] += intrhoz2*r11*r31*valH1 + intrhoz2*r12*r32*valH1
                k += 1
                Mv[k] += intrhoz2*r21*r31*valH1 + intrhoz2*r22*r32*valH1
                k += 1
                Mv[k] += intrhoz2*r31**2*valH1 + intrhoz2*r32**2*valH1
                k += 1
                Mv[k] += intrhoz*r11*r32*valH1 - intrhoz*r12*r31*valH1
                k += 1
                Mv[k] += intrhoz*r21*r32*valH1 - intrhoz*r22*r31*valH1
                k += 1
                Mv[k] += intrhoz2*r11*r31*valH1 + intrhoz2*r12*r32*valH1
                k += 1
                Mv[k] += intrhoz2*r21*r31*valH1 + intrhoz2*r22*r32*valH1
                k += 1
                Mv[k] += intrhoz2*r31**2*valH1 + intrhoz2*r32**2*valH1
                k += 1
                Mv[k] += intrhoz*r11*r32*valH1 - intrhoz*r12*r31*valH1
                k += 1
                Mv[k] += intrhoz*r21*r32*valH1 - intrhoz*r22*r31*valH1
                k += 1
                Mv[k] += intrhoz2*r11*r31*valH1 + intrhoz2*r12*r32*valH1
                k += 1
                Mv[k] += intrhoz2*r21*r31*valH1 + intrhoz2*r22*r32*valH1
                k += 1
                Mv[k] += intrhoz2*r31**2*valH1 + intrhoz2*r32**2*valH1

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
                Mv[k] += h11*intrhoz*r11*r22 - h11*intrhoz*r12*r21
                k += 1
                Mv[k] += h11*intrhoz*r11*r32 - h11*intrhoz*r12*r31
                k += 1
                Mv[k] += h11*intrho*r11*r21 + h11*intrho*r12*r22 + h11*intrho*r13*r23
                k += 1
                Mv[k] += h11*intrho*r21**2 + h11*intrho*r22**2 + h11*intrho*r23**2
                k += 1
                Mv[k] += h11*intrho*r21*r31 + h11*intrho*r22*r32 + h11*intrho*r23*r33
                k += 1
                Mv[k] += -h11*intrhoz*r11*r22 + h11*intrhoz*r12*r21
                k += 1
                Mv[k] += h11*intrhoz*r21*r32 - h11*intrhoz*r22*r31
                k += 1
                Mv[k] += h11*intrho*r11*r31 + h11*intrho*r12*r32 + h11*intrho*r13*r33
                k += 1
                Mv[k] += h11*intrho*r21*r31 + h11*intrho*r22*r32 + h11*intrho*r23*r33
                k += 1
                Mv[k] += h11*intrho*r31**2 + h11*intrho*r32**2 + h11*intrho*r33**2
                k += 1
                Mv[k] += -h11*intrhoz*r11*r32 + h11*intrhoz*r12*r31
                k += 1
                Mv[k] += -h11*intrhoz*r21*r32 + h11*intrhoz*r22*r31
                k += 1
                Mv[k] += -h11*intrhoz*r11*r22 + h11*intrhoz*r12*r21
                k += 1
                Mv[k] += -h11*intrhoz*r11*r32 + h11*intrhoz*r12*r31
                k += 1
                Mv[k] += h11*intrhoz2*r11**2 + h11*intrhoz2*r12**2
                k += 1
                Mv[k] += h11*intrhoz2*r11*r21 + h11*intrhoz2*r12*r22
                k += 1
                Mv[k] += h11*intrhoz2*r11*r31 + h11*intrhoz2*r12*r32
                k += 1
                Mv[k] += h11*intrhoz*r11*r22 - h11*intrhoz*r12*r21
                k += 1
                Mv[k] += -h11*intrhoz*r21*r32 + h11*intrhoz*r22*r31
                k += 1
                Mv[k] += h11*intrhoz2*r11*r21 + h11*intrhoz2*r12*r22
                k += 1
                Mv[k] += h11*intrhoz2*r21**2 + h11*intrhoz2*r22**2
                k += 1
                Mv[k] += h11*intrhoz2*r21*r31 + h11*intrhoz2*r22*r32
                k += 1
                Mv[k] += h11*intrhoz*r11*r32 - h11*intrhoz*r12*r31
                k += 1
                Mv[k] += h11*intrhoz*r21*r32 - h11*intrhoz*r22*r31
                k += 1
                Mv[k] += h11*intrhoz2*r11*r31 + h11*intrhoz2*r12*r32
                k += 1
                Mv[k] += h11*intrhoz2*r21*r31 + h11*intrhoz2*r22*r32
                k += 1
                Mv[k] += h11*intrhoz2*r31**2 + h11*intrhoz2*r32**2
                k += 1
                Mv[k] += h22*intrho*r11**2 + h22*intrho*r12**2 + h22*intrho*r13**2
                k += 1
                Mv[k] += h22*intrho*r11*r21 + h22*intrho*r12*r22 + h22*intrho*r13*r23
                k += 1
                Mv[k] += h22*intrho*r11*r31 + h22*intrho*r12*r32 + h22*intrho*r13*r33
                k += 1
                Mv[k] += h22*intrhoz*r11*r22 - h22*intrhoz*r12*r21
                k += 1
                Mv[k] += h22*intrhoz*r11*r32 - h22*intrhoz*r12*r31
                k += 1
                Mv[k] += h22*intrho*r11*r21 + h22*intrho*r12*r22 + h22*intrho*r13*r23
                k += 1
                Mv[k] += h22*intrho*r21**2 + h22*intrho*r22**2 + h22*intrho*r23**2
                k += 1
                Mv[k] += h22*intrho*r21*r31 + h22*intrho*r22*r32 + h22*intrho*r23*r33
                k += 1
                Mv[k] += -h22*intrhoz*r11*r22 + h22*intrhoz*r12*r21
                k += 1
                Mv[k] += h22*intrhoz*r21*r32 - h22*intrhoz*r22*r31
                k += 1
                Mv[k] += h22*intrho*r11*r31 + h22*intrho*r12*r32 + h22*intrho*r13*r33
                k += 1
                Mv[k] += h22*intrho*r21*r31 + h22*intrho*r22*r32 + h22*intrho*r23*r33
                k += 1
                Mv[k] += h22*intrho*r31**2 + h22*intrho*r32**2 + h22*intrho*r33**2
                k += 1
                Mv[k] += -h22*intrhoz*r11*r32 + h22*intrhoz*r12*r31
                k += 1
                Mv[k] += -h22*intrhoz*r21*r32 + h22*intrhoz*r22*r31
                k += 1
                Mv[k] += -h22*intrhoz*r11*r22 + h22*intrhoz*r12*r21
                k += 1
                Mv[k] += -h22*intrhoz*r11*r32 + h22*intrhoz*r12*r31
                k += 1
                Mv[k] += h22*intrhoz2*r11**2 + h22*intrhoz2*r12**2
                k += 1
                Mv[k] += h22*intrhoz2*r11*r21 + h22*intrhoz2*r12*r22
                k += 1
                Mv[k] += h22*intrhoz2*r11*r31 + h22*intrhoz2*r12*r32
                k += 1
                Mv[k] += h22*intrhoz*r11*r22 - h22*intrhoz*r12*r21
                k += 1
                Mv[k] += -h22*intrhoz*r21*r32 + h22*intrhoz*r22*r31
                k += 1
                Mv[k] += h22*intrhoz2*r11*r21 + h22*intrhoz2*r12*r22
                k += 1
                Mv[k] += h22*intrhoz2*r21**2 + h22*intrhoz2*r22**2
                k += 1
                Mv[k] += h22*intrhoz2*r21*r31 + h22*intrhoz2*r22*r32
                k += 1
                Mv[k] += h22*intrhoz*r11*r32 - h22*intrhoz*r12*r31
                k += 1
                Mv[k] += h22*intrhoz*r21*r32 - h22*intrhoz*r22*r31
                k += 1
                Mv[k] += h22*intrhoz2*r11*r31 + h22*intrhoz2*r12*r32
                k += 1
                Mv[k] += h22*intrhoz2*r21*r31 + h22*intrhoz2*r22*r32
                k += 1
                Mv[k] += h22*intrhoz2*r31**2 + h22*intrhoz2*r32**2
                k += 1
                Mv[k] += h33*intrho*r11**2 + h33*intrho*r12**2 + h33*intrho*r13**2
                k += 1
                Mv[k] += h33*intrho*r11*r21 + h33*intrho*r12*r22 + h33*intrho*r13*r23
                k += 1
                Mv[k] += h33*intrho*r11*r31 + h33*intrho*r12*r32 + h33*intrho*r13*r33
                k += 1
                Mv[k] += h33*intrhoz*r11*r22 - h33*intrhoz*r12*r21
                k += 1
                Mv[k] += h33*intrhoz*r11*r32 - h33*intrhoz*r12*r31
                k += 1
                Mv[k] += h33*intrho*r11*r21 + h33*intrho*r12*r22 + h33*intrho*r13*r23
                k += 1
                Mv[k] += h33*intrho*r21**2 + h33*intrho*r22**2 + h33*intrho*r23**2
                k += 1
                Mv[k] += h33*intrho*r21*r31 + h33*intrho*r22*r32 + h33*intrho*r23*r33
                k += 1
                Mv[k] += -h33*intrhoz*r11*r22 + h33*intrhoz*r12*r21
                k += 1
                Mv[k] += h33*intrhoz*r21*r32 - h33*intrhoz*r22*r31
                k += 1
                Mv[k] += h33*intrho*r11*r31 + h33*intrho*r12*r32 + h33*intrho*r13*r33
                k += 1
                Mv[k] += h33*intrho*r21*r31 + h33*intrho*r22*r32 + h33*intrho*r23*r33
                k += 1
                Mv[k] += h33*intrho*r31**2 + h33*intrho*r32**2 + h33*intrho*r33**2
                k += 1
                Mv[k] += -h33*intrhoz*r11*r32 + h33*intrhoz*r12*r31
                k += 1
                Mv[k] += -h33*intrhoz*r21*r32 + h33*intrhoz*r22*r31
                k += 1
                Mv[k] += -h33*intrhoz*r11*r22 + h33*intrhoz*r12*r21
                k += 1
                Mv[k] += -h33*intrhoz*r11*r32 + h33*intrhoz*r12*r31
                k += 1
                Mv[k] += h33*intrhoz2*r11**2 + h33*intrhoz2*r12**2
                k += 1
                Mv[k] += h33*intrhoz2*r11*r21 + h33*intrhoz2*r12*r22
                k += 1
                Mv[k] += h33*intrhoz2*r11*r31 + h33*intrhoz2*r12*r32
                k += 1
                Mv[k] += h33*intrhoz*r11*r22 - h33*intrhoz*r12*r21
                k += 1
                Mv[k] += -h33*intrhoz*r21*r32 + h33*intrhoz*r22*r31
                k += 1
                Mv[k] += h33*intrhoz2*r11*r21 + h33*intrhoz2*r12*r22
                k += 1
                Mv[k] += h33*intrhoz2*r21**2 + h33*intrhoz2*r22**2
                k += 1
                Mv[k] += h33*intrhoz2*r21*r31 + h33*intrhoz2*r22*r32
                k += 1
                Mv[k] += h33*intrhoz*r11*r32 - h33*intrhoz*r12*r31
                k += 1
                Mv[k] += h33*intrhoz*r21*r32 - h33*intrhoz*r22*r31
                k += 1
                Mv[k] += h33*intrhoz2*r11*r31 + h33*intrhoz2*r12*r32
                k += 1
                Mv[k] += h33*intrhoz2*r21*r31 + h33*intrhoz2*r22*r32
                k += 1
                Mv[k] += h33*intrhoz2*r31**2 + h33*intrhoz2*r32**2
                k += 1
                Mv[k] += h44*intrho*r11**2 + h44*intrho*r12**2 + h44*intrho*r13**2
                k += 1
                Mv[k] += h44*intrho*r11*r21 + h44*intrho*r12*r22 + h44*intrho*r13*r23
                k += 1
                Mv[k] += h44*intrho*r11*r31 + h44*intrho*r12*r32 + h44*intrho*r13*r33
                k += 1
                Mv[k] += h44*intrhoz*r11*r22 - h44*intrhoz*r12*r21
                k += 1
                Mv[k] += h44*intrhoz*r11*r32 - h44*intrhoz*r12*r31
                k += 1
                Mv[k] += h44*intrho*r11*r21 + h44*intrho*r12*r22 + h44*intrho*r13*r23
                k += 1
                Mv[k] += h44*intrho*r21**2 + h44*intrho*r22**2 + h44*intrho*r23**2
                k += 1
                Mv[k] += h44*intrho*r21*r31 + h44*intrho*r22*r32 + h44*intrho*r23*r33
                k += 1
                Mv[k] += -h44*intrhoz*r11*r22 + h44*intrhoz*r12*r21
                k += 1
                Mv[k] += h44*intrhoz*r21*r32 - h44*intrhoz*r22*r31
                k += 1
                Mv[k] += h44*intrho*r11*r31 + h44*intrho*r12*r32 + h44*intrho*r13*r33
                k += 1
                Mv[k] += h44*intrho*r21*r31 + h44*intrho*r22*r32 + h44*intrho*r23*r33
                k += 1
                Mv[k] += h44*intrho*r31**2 + h44*intrho*r32**2 + h44*intrho*r33**2
                k += 1
                Mv[k] += -h44*intrhoz*r11*r32 + h44*intrhoz*r12*r31
                k += 1
                Mv[k] += -h44*intrhoz*r21*r32 + h44*intrhoz*r22*r31
                k += 1
                Mv[k] += -h44*intrhoz*r11*r22 + h44*intrhoz*r12*r21
                k += 1
                Mv[k] += -h44*intrhoz*r11*r32 + h44*intrhoz*r12*r31
                k += 1
                Mv[k] += h44*intrhoz2*r11**2 + h44*intrhoz2*r12**2
                k += 1
                Mv[k] += h44*intrhoz2*r11*r21 + h44*intrhoz2*r12*r22
                k += 1
                Mv[k] += h44*intrhoz2*r11*r31 + h44*intrhoz2*r12*r32
                k += 1
                Mv[k] += h44*intrhoz*r11*r22 - h44*intrhoz*r12*r21
                k += 1
                Mv[k] += -h44*intrhoz*r21*r32 + h44*intrhoz*r22*r31
                k += 1
                Mv[k] += h44*intrhoz2*r11*r21 + h44*intrhoz2*r12*r22
                k += 1
                Mv[k] += h44*intrhoz2*r21**2 + h44*intrhoz2*r22**2
                k += 1
                Mv[k] += h44*intrhoz2*r21*r31 + h44*intrhoz2*r22*r32
                k += 1
                Mv[k] += h44*intrhoz*r11*r32 - h44*intrhoz*r12*r31
                k += 1
                Mv[k] += h44*intrhoz*r21*r32 - h44*intrhoz*r22*r31
                k += 1
                Mv[k] += h44*intrhoz2*r11*r31 + h44*intrhoz2*r12*r32
                k += 1
                Mv[k] += h44*intrhoz2*r21*r31 + h44*intrhoz2*r22*r32
                k += 1
                Mv[k] += h44*intrhoz2*r31**2 + h44*intrhoz2*r32**2


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

