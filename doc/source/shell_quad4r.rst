Quad4R: one-point integration and hourglass control
===================================================

:class:`.Quad4R`, module :mod:`pyfe3d.quad4r`, has the kinematics,
interpolation and constitutive relation of :class:`.Quad4`, see
:doc:`shell_formulation`, but integrates the membrane, coupling, bending
and transverse shear terms with a single point at the centroid, `\xi =
\eta = 0` with the weight 4, `4 \det \boldsymbol{J}_0` being the area `A`
of a parallelogram. One point makes the internal forces and the stiffness
matrices cheaper than those of :class:`.Quad4`, and it leaves hourglass
modes, deformations with no strain at the centroid, which must be given
stiffness. The strain energy is that of :eq:`sh-strain-energy` plus the
hourglass energy `U_{hg}`:

- the hourglass modes of the bending and of the transverse shear, of `w`,
  `r_x` and `r_y`, have the **physical stabilisation** of Belytschko and
  Leviathan (1994), without parameter;
- the hourglass modes of the membrane, of `u` and `v`, have the
  **artificial stiffness** of Brockman (1987).

The arguments ``hgfactor_physical`` and ``hgfactor_membrane`` of
:meth:`.Quad4R.update_KC0`, :meth:`.Quad4R.update_probe_finte` and
:meth:`.Quad4R.update_fint` multiply the two; ``1.``, the default, is the
formulation described here, other values serve verification and
sensitivity studies.


Decomposition of the bilinear field
-----------------------------------

Belytschko and Leviathan (1994), Eq. (3.23), write the bilinear shape
functions as a linear field plus an hourglass term,

.. math::
    :label: sh-bl94-shape

    S_I = \Delta_I + x \, b_{xI} + y \, b_{yI} + h \, \gamma_I, \qquad
    h = \xi \eta, \qquad
    \gamma_I = \frac{1}{4} \left( h_I - (\boldsymbol{h}^\top \boldsymbol{x})
    b_{xI} - (\boldsymbol{h}^\top \boldsymbol{y}) b_{yI} \right)

with `b_{xI}`, `b_{yI}` the derivatives at the centroid, `\boldsymbol{h} =
\{1, -1, 1, -1\}` and `\boldsymbol{x}`, `\boldsymbol{y}` the nodal
coordinates, such that `S_{I,x} = b_{xI} + h_{,x} \gamma_I`, Eq. (3.25),
with `\{h_{,x}, h_{,y}\}^\top = \boldsymbol{J}^{-1} \{\eta,
\xi\}^\top`. The vector `\gamma_I` is orthogonal to the linear fields:
`\boldsymbol{\gamma}^\top` applied to the nodal values of any rigid motion
or linear field is zero, on any quadrilateral. The constant part of the
strains is integrated at the centroid, and the stabilisation is the energy
of their non-constant part.


Physical stabilisation of the transverse shear
----------------------------------------------

The transverse shear strains are the assumed covariant strains of Dvorkin
and Bathe (1984), :eq:`sh-mitc4-covariant`, tied at the mid-points of the
edges, here with the labels of Belytschko and Leviathan, `B` `(0, -1)`,
`D` `(0, 1)`, `A` `(-1, 0)` and `C` `(1, 0)`, and split into a constant
and a non-constant part, Eqs. (3.35)-(3.39):

.. math::
    :label: sh-bl94-shear

    e_\xi = \frac{e_B + e_D}{2} + \eta \frac{e_D - e_B}{2}, \qquad
    e_\eta = \frac{e_A + e_C}{2} + \xi \frac{e_C - e_A}{2}

both mapped to the element axes with the Jacobian `\boldsymbol{J}_0` at the
centroid. The constant part replaces the displacement-based shear strains
at the centroid, Eq. (4.12), and the non-constant part gives the
stabilisation of Eq. (4.13), integrated exactly with `\int \xi^2 dA =
\int \eta^2 dA = A/3` and `\int \xi \eta \, dA = 0`:

.. math::
    :label: sh-bl94-shear-stab

    \boldsymbol{K}^s_{stab} = f_p \frac{A}{3} \left( \boldsymbol{D}_1^\top
    \boldsymbol{A}_s \boldsymbol{D}_1 + \boldsymbol{D}_2^\top
    \boldsymbol{A}_s \boldsymbol{D}_2 \right), \qquad
    \boldsymbol{D}_1 = \boldsymbol{J}_0^{-1}
    \begin{Bmatrix} (e_D - e_B)/2 \\ 0 \end{Bmatrix}, \quad
    \boldsymbol{D}_2 = \boldsymbol{J}_0^{-1}
    \begin{Bmatrix} 0 \\ (e_C - e_A)/2 \end{Bmatrix}

with `f_p` = ``hgfactor_physical`` and `\boldsymbol{A}_s` the matrix of
`A_{55}`, `A_{45}` and `A_{44}` for `\gamma_{xz}` and `\gamma_{yz}`; the
coupling of the constant and non-constant parts is omitted, as in the
paper. On a parallelogram this is the transverse shear of :class:`.Quad4`
with two by two points, see :doc:`shell_quad4`.


Physical stabilisation of the bending
-------------------------------------

The non-constant part of the curvatures, Eq. (3.28), is

.. math::
    :label: sh-bl94-bending

    \kappa_{xx} = h_{,x} q_y, \qquad \kappa_{yy} = -h_{,y} q_x, \qquad
    \kappa_{xy} = h_{,y} q_y - h_{,x} q_x, \qquad
    q_x = \boldsymbol{\gamma}^\top \boldsymbol{r}_x, \quad
    q_y = \boldsymbol{\gamma}^\top \boldsymbol{r}_y

with `\boldsymbol{r}_x`, `\boldsymbol{r}_y` the nodal rotations, and its
energy gives the stiffness of Eqs. (4.13), (B.17) and (B.18),

.. math::
    :label: sh-bl94-bending-stab

    K^b_{r_{xI} r_{xJ}} = f_p Q_{xx} \gamma_I \gamma_J, \quad
    K^b_{r_{yI} r_{yJ}} = f_p Q_{yy} \gamma_I \gamma_J, \quad
    K^b_{r_{xI} r_{yJ}} = f_p Q_{xy} \gamma_I \gamma_J

.. math::

    Q_{xx} = H_{xx} D_{66} + H_{yy} D_{22} + 2 H_{xy} D_{26}, \quad
    Q_{yy} = H_{xx} D_{11} + H_{yy} D_{66} + 2 H_{xy} D_{16}, \quad
    Q_{xy} = -H_{xx} D_{16} - H_{yy} D_{26} - H_{xy} (D_{12} + D_{66})

with `H_{xx}`, `H_{yy}`, `H_{xy}` the integrals of `h_{,x}^2`, `h_{,y}^2`
and `h_{,x} h_{,y}` over the element, Eq. (4.14), evaluated with two by two
points, and `D_{ij}` the entries of the bending stiffness with zero
membrane forces, `\boldsymbol{D} - \boldsymbol{B} \boldsymbol{A}^{-1}
\boldsymbol{B}`, which is `\boldsymbol{D}` for a symmetric laminate and
does not depend on the ``offset`` of the :class:`.ShellProp`. On a
parallelogram the one-point bending plus this stabilisation is the bending
integrated with two by two points.

Neither stabilisation has a parameter. The rigid-body motions and every
state of constant strain, of constant curvature, including the uniform
twist `w = kxy`, `r_x = kx`, `r_y = -ky`, and of constant transverse shear
have no non-constant strain, on any quadrilateral, and the assumed shear
strains of a constant curvature vanish too: the element passes the
constant-twist patch test and the thin-plate bending patch test of
Belytschko and Leviathan (1994), Section 6.1, for any thickness and unit
of length, and a single element has only the six rigid-body modes as
zero-energy modes. Both stiffnesses scale with the unit of length as the
physical ones do.


Hourglass control of the membrane
---------------------------------

The hourglass modes of `u` and `v` have the artificial stiffness of
Brockman (1987), with the orthogonalised operator of his Eq. (14), quoted
from Belytschko and Tsay (1983), normalised as in his Eq. (15):

.. math::
    :label: sh-brockman

    K_{ij} = A \, E^{(h)} \gamma_i \gamma_j, \qquad
    \gamma_i = S_{1,xy} \left( h_i - h_x S_{i,x} - h_y S_{i,y} \right),
    \qquad
    h_x = x_1 - x_2 + x_3 - x_4, \quad h_y = y_1 - y_2 + y_3 - y_4

for each of `u` and `v`, with the derivatives at the centroid, the
correction terms in `h_x`, `h_y` vanishing for a parallelogram. The
generalised stiffnesses of Brockman, `0.10 E t/(1 + 1/A)`, are not
dimensionally homogeneous, and tend to the area `A` only for `A \ll 1` in
the unit of length of the model; ``pyfe3d`` uses the area itself and,
for a laminate, the membrane moduli of the extensional compliance
`\boldsymbol{A}^{-1} = [a_{ij}]` in the element axes,

.. math::
    :label: sh-brockman-moduli

    E^{(h)}_u = 0.10 \, f_m E_{1eq} t A, \qquad
    E^{(h)}_v = 0.10 \, f_m E_{2eq} t A, \qquad
    E_{1eq} = \frac{1}{t a_{11}}, \quad E_{2eq} = \frac{1}{t a_{22}}

with `f_m` = ``hgfactor_membrane`` and `t` the thickness, which reduce to
`E` for an isotropic plate and make the response independent of the unit
of length. The physical stabilisation of the membrane of Belytschko and
Leviathan (1994), Eqs. (3.27) and (B.13), which on a flat element is the
bilinear membrane with two by two points, was measured stiffer still in
in-plane bending, and is not used. Neither form is free of locking in the
in-plane bending of strips one element deep, e.g. 0.119, 0.033 and 0.064
of the reference tip displacement of the straight cantilever of MacNeal
and Harder (1985), 6 by 1 elements, on the regular, trapezoidal and
parallelogram meshes; :class:`.Quad4` does not lock there.


Drilling stiffness without enrichment
-------------------------------------

The drilling rotation is tied to the in-plane rotation of the membrane by
the Hughes-Brezzi term, integrated with two by two points, with
`\gamma_{r_z} = A_{66}` for ``drilling_model = 0`` and `K6ROT \cdot
10^{-6} A_{66}` otherwise, see :doc:`shell_drilling`. Unlike
:class:`.Quad4` and :class:`.Tria3R`, the membrane is not enriched with
the drilling rotations: the enrichment removes the excessive in-plane
bending stiffness of the bilinear field, which the one-point integration
already removes. On the skew membrane of Cook, reference 23.96, the tip
displacement is 30.14, 24.66, 24.08 and 23.97 with 2x2 to 16x16
elements, against 20.75, 22.83, 23.54 and 23.81 for the enriched
:class:`.Quad4`. An enrichment sampled at one point would leave the
spurious mode `u \propto x`, `v \propto -y`, `r_z \propto \boldsymbol{h}`,
whose membrane strain at the centroid is cancelled by the enrichment, see
the Notes of :mod:`pyfe3d.quad4r`.


Geometric stiffness
-------------------

The membrane forces of :eq:`sh-KG` are integrated with two by two points,
and the slopes of `w` are taken at the centroid, the single point of the
curvatures and of the transverse shear in ``KC0``, see
:meth:`.Quad4R.update_KG`. The hourglass pattern `w = \xi \eta` then has no
geometric stiffness, as it has no stiffness in ``KC0`` besides the
stabilisation of the transverse shear; with the slopes at the two by two
points, a linear buckling analysis could return a spurious mode of `w`
alternating in sign from node to node, e.g. 41 % below the Mindlin
solution of a simply supported square plate with `b/h = 10` and 16 by 16
elements. With a warped element, ``update_KG`` and ``update_KCNL`` take the
``hgfactor_*`` of ``update_KC0`` for the nodal forces of the eccentricity
term of the rigid offsets, see :doc:`element_choice`.


What the element computes
-------------------------

.. list-table::
    :header-rows: 1

    * - Method
      - Theory
    * - :meth:`.Quad4R.update_KC0`
      - one-point :eq:`sh-KC0`, :eq:`sh-bl94-shear-stab`,
        :eq:`sh-bl94-bending-stab`, :eq:`sh-brockman` and the drilling term
    * - :meth:`.Quad4R.update_fint`, :meth:`.Quad4R.update_probe_finte`
      - gradient of the strain energy with the hourglass terms; with
        ``nonlinear=1`` the von Kármán terms of :eq:`sh-von-karman`
    * - :meth:`.Quad4R.update_KCNL`, :meth:`.Quad4R.update_KG`,
        :meth:`.Quad4R.update_KG_given_stress`
      - :eq:`sh-tangent` and :eq:`sh-KG`, slopes at the centroid
    * - :meth:`.Quad4R.update_M`
      - mass matrices, see :doc:`shell_mass`
    * - :meth:`.Quad4R.update_KA_beta`, :meth:`.Quad4R.update_KA_gamma`,
        :meth:`.Quad4R.update_CA`
      - piston theory, :eq:`sh-piston-matrices`


References
----------

- Belytschko, T. and Leviathan, I., Physical stabilization of the 4-node
  shell element with one point quadrature, Computer Methods in Applied
  Mechanics and Engineering, 113(3-4), 321-350, 1994.
  https://doi.org/10.1016/0045-7825(94)90052-3
- Belytschko, T. and Tsay, C. S., A stabilization procedure for the
  quadrilateral plate element with one-point quadrature, International
  Journal for Numerical Methods in Engineering, 19(3), 405-419, 1983.
  https://doi.org/10.1002/nme.1620190308
- Brockman, R. A., Dynamics of the bilinear Mindlin plate element,
  International Journal for Numerical Methods in Engineering, 24(12),
  2343-2356, 1987. https://doi.org/10.1002/nme.1620241208
- Dvorkin, E. N. and Bathe, K. J., A continuum mechanics based four-node
  shell element for general non-linear analysis, Engineering Computations,
  1(1), 77-88, 1984. https://doi.org/10.1108/eb023562
- MacNeal, R. H. and Harder, R. L., A proposed standard set of problems to
  test finite element accuracy, Finite Elements in Analysis and Design,
  1(1), 3-20, 1985. https://doi.org/10.1016/0168-874X(85)90003-4
