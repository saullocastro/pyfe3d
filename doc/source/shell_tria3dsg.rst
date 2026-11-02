Tria3DSG: the discrete shear gap triangle
=========================================

:class:`.Tria3DSG`, module :mod:`pyfe3d.tria3dsg`, is a flat three-node
triangle with the FSDT kinematics of :doc:`shell_formulation`, linear
interpolation of the six degrees of freedom and transverse shear strains
obtained with the discrete shear gap (DSG) method of Bletzinger, Bischoff
and Ramm (2000). Its membrane is that of the constant-strain triangle,
without drilling enrichment. The same transverse shear field is the
default of :class:`.Tria3R`, which adds the Allman membrane, see
:doc:`shell_tria3r`.


Constant strains of the linear triangle
---------------------------------------

With the area coordinates `S_1`, `S_2`, `S_3` and `\det \boldsymbol{J} =
x_1(y_2 - y_3) + x_2(y_3 - y_1) + x_3(y_1 - y_2)`, twice the signed area,
the derivatives `S_{1,x} = (y_2 - y_3)/\det \boldsymbol{J}`, `S_{1,y} = (x_3
- x_2)/\det \boldsymbol{J}` and their cyclic permutations are constant,
and so are the membrane strains and the curvatures of :eq:`sh-strains`.
One point integrates them exactly, and the triangle has no hourglass
modes. The signed area makes the operators correct for both orderings of
the nodes. The transverse shear strains of the same interpolation are not
usable: `\nabla w` is a single constant vector and the rotations are
linear, such that no nodal values give `\boldsymbol{\gamma}^{(0)} =
\boldsymbol{0}` in a state of constant curvature, and the element locks,
see :doc:`shell_formulation`.


The discrete shear gap
----------------------

With `\boldsymbol{\phi} = (-r_y, r_x)`, the transverse shear strains of
:eq:`sh-strains` are `\{\gamma_{xz}, \gamma_{yz}\} = \nabla w -
\boldsymbol{\phi}`. The shear gap is the part of the increase of the
deflection between two points that is due to shear rather than to bending
(Bletzinger et al., 2000, Sec. 2, Eq. (3)). On the triangle, its nodal
values are obtained by integrating the shear strains along the straight
edges that join a starting node `p` to the other two nodes `q`, with the
rotations interpolated linearly along the edge, such that their integral
is the mean of the two nodal values times the edge vector:

.. math::
    :label: sh-dsg-gap

    \Delta w_p = 0, \qquad
    \Delta w_q = (w_q - w_p) + \frac{1}{2} (r_{y,p} + r_{y,q}) \, \Delta x
    - \frac{1}{2} (r_{x,p} + r_{x,q}) \, \Delta y, \qquad
    \Delta x = x_q - x_p, \quad \Delta y = y_q - y_p

The assumed transverse shear strain is the gradient of the interpolated
gap,

.. math::
    :label: sh-dsg-strain

    \begin{Bmatrix} \gamma_{xz} \\ \gamma_{yz} \end{Bmatrix} = \nabla
    \left( \sum_i S_i \Delta w_i \right) = \sum_{q \ne p} \Delta w_q
    \begin{Bmatrix} S_{q,x} \\ S_{q,y} \end{Bmatrix}

which is constant over the element and a linear combination of the nodal
values only. The following properties follow from :eq:`sh-dsg-gap` and
:eq:`sh-dsg-strain`:

- the rigid-body modes that involve `w`, a translation and the two
  rotations about the in-plane axes, give `\Delta w_q = 0` and no shear;
- a state of constant transverse shear is reproduced exactly;
- the nodal values of a Kirchhoff state of constant curvature, `w`
  quadratic and `r_y = -w_{,x}`, `r_x = w_{,y}`, give `\Delta w_q = 0`:
  the element stores no parasitic shear energy in pure bending, for any
  thickness, which is the locking-free property;
- the operator is formed once per element, without quadrature.

The transverse shear stiffness `\boldsymbol{A}_{ts}` of the
:class:`.ShellProp` is therefore used as the laminate gives it, and the
shear correction keeps its meaning.

**Averaging over the starting node.** The gap measured from one node makes
the operator depend on which node that is, a dependence inherent to the
method, whose influence decreases with refinement (Bletzinger et al.,
2000, Sec. 4.1.1): on an irregular triangle the element matrix changes by
14 and 17 % under the two cyclic relabellings of its nodes. ``pyfe3d``
uses the mean of the three operators,

.. math::
    :label: sh-dsg-average

    \begin{Bmatrix} \gamma_{xz} \\ \gamma_{yz} \end{Bmatrix} = \frac{1}{3}
    \sum_{p=1}^{3} \sum_{q \ne p} \Delta w_q^{(p)}
    \begin{Bmatrix} S_{q,x} \\ S_{q,y} \end{Bmatrix}

which inherits the four properties, since each operator has them and they
are linear in the operator, and which does not depend on the numbering
of the nodes, see the Notes of :mod:`pyfe3d.tria3dsg`.

The transverse shear strains have no von Kármán terms, such that the
DSG operator appears only in ``KC0``. Every operator of the element is
constant over the triangle, and ``KCNL``, ``KG`` and the non-linear
internal forces of :doc:`shell_formulation` are integrated exactly by a
multiplication by the area.


Optional rescaling of the transverse shear stiffness
----------------------------------------------------

The attribute ``alpha_shear_locking`` `= \alpha`, ``0.`` by default,
divides the transverse shear stiffnesses in the element axes by

.. math::
    :label: sh-alpha

    1 + \alpha \frac{\ell^2}{h^2}

with `\ell` the longest edge and `h` the thickness, the factor of Lyly,
Stenberg and Vihinen (1993), Eq. (3.12), called Stenberg's method by
Bischoff and Bletzinger (2004), Sec. 3.1, in the form of Castro et al.
(2019), Eq. (26). In these references the factor sits on top of an
assumed-strain or DSG field that is already free of locking, with values
of `\alpha` near 0.1. On this element it is a mild coarse-mesh
stabilisation: `\alpha = 0.07`, the lower bound of the range of Castro et
al., takes the first natural frequency of a thin plate with a `5 \times 7`
mesh from 12.06 to 7.93 % above the analytical value, and its buckling
load from +1.71 to +0.11 %. The factor is unbounded in `\ell/h`, and a
non-zero `\alpha` removes nearly all the transverse shear stiffness of a
practical shell mesh: the energy of a constant transverse shear state
drops to `1/(1 + \alpha \ell^2/h^2)` of the correct one, and on a plate
with `a/h = 5`, where the transverse shear carries 19 % of the deflection,
the error grows monotonically with `\alpha`. ``alpha_shear_locking`` is
therefore zero by default, and ``0.07`` is the value recorded for coarse
meshes of thin plates only, see the Notes of :class:`.Tria3DSG`.


Membrane and drilling
---------------------

The membrane is the constant-strain triangle, stiff in in-plane bending:
on the skew membrane of Cook with `n \times n` cells split along a
diagonal, the tip displacement is 0.457, 0.713, 0.896 and 0.970 of the
reference for `n` = 4, 8, 16 and 32, against 0.864, 0.948, 0.982 and 0.994
for :class:`.Tria3R`. The drilling rotation is tied to the in-plane
rotation `\theta_z = (v_{,x} - u_{,y})/2` by the Hughes-Brezzi term with
`\gamma_{r_z} = A_{66}`, or with the penalty `K6ROT \cdot 10^{-6} A_{66}`
for ``drilling_model`` different from zero, integrated with the
three-point rule: one point would constrain one combination of the three
`r_z` only, and an assembled patch would have eight zero eigenvalues
instead of six, see :doc:`shell_drilling` and the Notes of
:meth:`.Tria3DSG.update_KC0`.


What the element computes
-------------------------

.. list-table::
    :header-rows: 1

    * - Method
      - Theory
    * - :meth:`.Tria3DSG.update_KC0`
      - :eq:`sh-KC0` with the shear of :eq:`sh-dsg-average`, scaled by
        :eq:`sh-alpha` when ``alpha_shear_locking`` is not zero
    * - :meth:`.Tria3DSG.update_fint`
      - gradient of the strain energy, with ``nonlinear=1`` the von Kármán
        terms
    * - :meth:`.Tria3DSG.update_KCNL`, :meth:`.Tria3DSG.update_KG`,
        :meth:`.Tria3DSG.update_KG_given_stress`
      - :eq:`sh-tangent` and :eq:`sh-KG`, constant over the element
    * - :meth:`.Tria3DSG.update_M`
      - consistent and lumped mass matrices, see :doc:`shell_mass`


References
----------

- Bischoff, M. and Bletzinger, K.-U., Improving stability and accuracy of
  Reissner-Mindlin plate finite elements via algebraic subgrid scale
  stabilization, Computer Methods in Applied Mechanics and Engineering,
  193(15-16), 1517-1528, 2004. https://doi.org/10.1016/j.cma.2003.12.036
- Bletzinger, K.-U., Bischoff, M. and Ramm, E., A unified approach for
  shear-locking-free triangular and rectangular shell finite elements,
  Computers & Structures, 75(3), 321-334, 2000.
  https://doi.org/10.1016/S0045-7949(99)00140-6
- Castro, S. G. P., Donadon, M. V. and Guimaraes, T. A. M., ES-PIM applied
  to buckling of variable angle tow laminates, Composite Structures, 209,
  67-78, 2019. https://doi.org/10.1016/j.compstruct.2018.10.058
- Lyly, M., Stenberg, R. and Vihinen, T., A stable bilinear element for
  the Reissner-Mindlin plate model, Computer Methods in Applied Mechanics
  and Engineering, 110(3-4), 343-357, 1993.
  https://doi.org/10.1016/0045-7825(93)90214-I
