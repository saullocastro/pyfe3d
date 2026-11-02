Quad4: assumed transverse shear strains (MITC4)
===============================================

:class:`.Quad4`, module :mod:`pyfe3d.quad4`, is the recommended
quadrilateral of ``pyfe3d``: a flat four-node element with the FSDT
kinematics of :doc:`shell_formulation`, bilinear interpolation of the six
degrees of freedom, the assumed transverse shear strains of the MITC4
element of Dvorkin and Bathe (1984) and, by default, the drilling
enrichment of the membrane of Allman (1984) with the regularisation of
Hughes and Brezzi (1989), see :doc:`shell_drilling`. Its name in the code,
"mixed integration", refers to the different quadratures of its terms.


Quadratures
-----------

The terms of the stiffness matrix :eq:`sh-KC0` are integrated with
Gauss-Legendre points:

- bending, `\boldsymbol{B}_b^\top \boldsymbol{D} \boldsymbol{B}_b`: two by
  two points, with every drilling model, such that the drilling model does
  not change the bending response;
- membrane and membrane-bending coupling: three by three points with the
  default ``drilling_model = 0``, whose edge modes make the membrane
  strains vary linearly, and two by two points with the penalty
  ``drilling_model = 1``, see :doc:`shell_drilling`;
- transverse shear: the assumed strains below, with two by two points,
  with the default ``shear_model = 1``;
- the geometric stiffness, ``KCNL`` and the non-linear internal forces:
  two by two points for the membrane forces, with the slopes of `w` of
  `Geometric stiffness`_ below.


Assumed transverse shear strains
--------------------------------

The covariant transverse shear strains of the bilinear field, with the
interpolated `w`, `r_x`, `r_y` and coordinates,

.. math::
    :label: sh-mitc4-covariant

    \gamma_{\xi z} = x_{,\xi} \gamma_{xz} + y_{,\xi} \gamma_{yz} =
    w_{,\xi} + x_{,\xi} r_y - y_{,\xi} r_x, \qquad
    \gamma_{\eta z} = w_{,\eta} + x_{,\eta} r_y - y_{,\eta} r_x

are evaluated at the mid-points of the edges, the tying points `A` `(0,
-1)` and `C` `(0, 1)` for `\gamma_{\xi z}` and `D` `(-1, 0)` and `B` `(1,
0)` for `\gamma_{\eta z}`, and interpolated linearly across the element,

.. math::
    :label: sh-mitc4

    \gamma_{\xi z} = \frac{1 - \eta}{2} \gamma_{\xi z}^{A} + \frac{1 +
    \eta}{2} \gamma_{\xi z}^{C}, \qquad
    \gamma_{\eta z} = \frac{1 - \xi}{2} \gamma_{\eta z}^{D} + \frac{1 +
    \xi}{2} \gamma_{\eta z}^{B}

such that `\gamma_{\xi z}` is constant along `\xi` and `\gamma_{\eta z}`
along `\eta`. At each integration point they are brought to the element
axes with the inverse Jacobian of that point, `\{\gamma_{xz},
\gamma_{yz}\}^\top = \boldsymbol{J}^{-1} \{\gamma_{\xi z}, \gamma_{\eta
z}\}^\top`, and the shear energy of :eq:`sh-KC0` is integrated with two by
two points. The strains are exact for a pure twist and for a pure bending of a rectangle, the
element passes the patch test and it has no zero-energy mode besides the
six rigid-body modes (Bathe and Dvorkin, 1985, pp. 372-373).

On a parallelogram, whose Jacobian is constant, the assumed strain at the
centroid is the displacement-based one, and the two by two integral of
the assumed shear energy is the one-point shear term of Hughes et al.
(1977) plus the energy of the parts of the assumed strains linear in
`\eta`, for `\gamma_{\xi z}`, and in `\xi`, for `\gamma_{\eta z}`: a
matrix of rank two, the physical stabilisation of Belytschko and Leviathan
(1994), Eq. (4.13), which :class:`.Quad4R` uses and which vanishes for
every state of constant transverse shear and of constant curvature, see
:doc:`shell_quad4r`. The element is then that of Hughes et al. with the
energy of the transverse shear that varies linearly over the element
added, the energy that a one-point rule discards.


The one-point transverse shear of ``shear_model = 0``
-----------------------------------------------------

With ``shear_model = 0`` the transverse shear is integrated with one point
at the centroid when the thickness of the property and the length `\ell =
\sqrt{A}` of the element give `h/\ell < 1`, the selective scheme of Hughes
et al. (1977), and with the assumed strains above otherwise. It is kept to
reproduce the results of the versions up to 0.11. The one-point rule has
two zero-energy modes besides the rigid-body modes (Hughes et al., 1977,
p. 1542; Hughes, Cohen and Haroun, 1978, Sec. 3.6): the hourglass mode `w
= (x - x_c)(y - y_c)` with zero rotations and the in-plane twist `w = 0`,
`r_x = x - x_c`, `r_y = y - y_c`, `(x_c, y_c)` being the centroid, whose
curvatures vanish and whose shear strains vanish at the centroid. A single
element then has eight zero eigenvalues instead of six. The hourglass mode
survives assembly. It is suppressed by any support that constrains `w`
along a line, but a strip one element wide in torsion, whose twist is the
hourglass pattern, is far too flexible: 377 times the reference twist of
the straight cantilever of MacNeal and Harder (1985), against 0.941 with
the assumed strains, the value of their QUAD4 element.

On other quadrilaterals the displacement-based strain at the centroid does
not vanish under a constant curvature, and the one-point rule fails the
thin-plate bending patch test: 0.87 to 1.22 times the exact deflection of
the interior nodes on the patch of Belytschko and Leviathan (1994), and an
energy error of 580 % on the patch of MacNeal and Harder (1985), which the
assumed strains pass to round-off. For very thick elements, `h/\ell \ge
1`, Hughes et al. propose a "modified one-point shear" with two by two
points for the products of the slopes of `w`; mixing two quadratures in
one energy stores the spurious shear energy `A_{55} \ell^4 \kappa^2/12` in
the pure twist `w = \kappa x y`, `r_x = \kappa x`, `r_y = -\kappa y` of a
square element, `(5/12)(\ell/h)^2` times its twist energy for an
isotropic plate, which is why the assumed strains are used there with
both values of ``shear_model``. The Notes of :mod:`pyfe3d.quad4` give the
results of both rules on the problems of MacNeal and Harder (1985).


Geometric stiffness
-------------------

The membrane forces of :eq:`sh-KG` are integrated with two by two points.
The slopes `w_{,x}` and `w_{,y}` are taken at the same points with the
default ``shear_model = 1``, and at the centroid with ``shear_model = 0``
or ``centroid_slopes = 1``, see :meth:`.Quad4.update_KG`. The reason is
the hourglass pattern `w = \xi \eta` with zero rotations, which has no
slope at the centroid. The one-point shear gives it no stiffness in
``KC0``. With the slopes at the two by two points it has the geometric
stiffness of a full quadrature: in a supported plate, the pattern
modulated by a smooth envelope costs only the shear energy of the
envelope, independent of the element size `\ell`, whereas its geometric
work grows as `1/\ell^2`, and a linear buckling analysis can return a
spurious mode of `w` alternating in sign from node to node below the
physical buckling load once the transverse shear stiffness is low compared
with the bending stiffness, as in a sandwich plate. Taken at the centroid,
the slopes give the pattern no geometric stiffness, consistently with
``KC0``. The assumed shear strains of the default give the pattern its
shear stiffness at every element size, and the slopes at the two by two
points can be used: they integrate the geometric stiffness of a smooth
mode more accurately, while the slopes at the centroid give higher
buckling loads on coarse meshes. Measured on a simply supported plate with
5 elements per half-wave along the load, the buckling load is 4.31 %
above the Kirchhoff solution with the default, 6.81 % with
``centroid_slopes = 1`` and 4.67 % with ``shear_model = 0``.

The same slopes enter ``update_KG``, ``update_KCNL`` and the non-linear
internal forces, so that ``KC0 + KCNL + KG`` remains the exact Jacobian
for every rule. In the co-rotational frame of :class:`.CorotationalShell`,
with ``local='von_karman'``, the transverse translations of a warped
element are the hourglass pattern of its warp, not a deflection, and the
wrapper sets ``centroid_slopes = 1`` while it calls the element.


Membrane and drilling
---------------------

With the default ``drilling_model = 0`` the in-plane displacements are
enriched with the edge modes of Allman, the mid-side functions of the
eight-node serendipity element, whose amplitudes are differences of the
drilling rotations of the nodes, see :doc:`shell_drilling`. The enrichment
removes the excessive in-plane bending stiffness of the bilinear membrane:
on the skew membrane of Cook with two by two elements the tip displacement
is 20.8 against the reference 23.9, where the penalty model gives 11.8,
and the element does not lock in the in-plane bending of strips one
element deep, unlike :class:`.Quad4R`. The curvature and transverse shear
strains are not enriched; with `\boldsymbol{B} \ne \boldsymbol{0}` the
enriched membrane strains couple `r_z` to the curvatures. The
enrichment enters the membrane forces of the geometric stiffness as well,
see :meth:`.Quad4.update_KG`, and the strains recovered with
:meth:`.Quad4Probe.update_BL` must be given the ``drilling_model`` of the
element.


Warped elements
---------------

The matrices are those of the flat projection of the element, linked to
the warped nodes by rigid offsets with the default ``warp_offsets = 1``,
see :doc:`element_choice`, section *Warped quadrilaterals*, and the Notes
of :class:`.Quad4`.


What the element computes
-------------------------

.. list-table::
    :header-rows: 1

    * - Method
      - Theory
    * - :meth:`.Quad4.update_KC0`
      - :eq:`sh-KC0` with the quadratures above
    * - :meth:`.Quad4.update_fint`
      - gradient of the strain energy, with ``nonlinear=1`` the von Kármán
        terms of :eq:`sh-von-karman`
    * - :meth:`.Quad4.update_KCNL`, :meth:`.Quad4.update_KG`,
        :meth:`.Quad4.update_KG_given_stress`
      - :eq:`sh-tangent` and :eq:`sh-KG`
    * - :meth:`.Quad4.update_M`
      - mass matrices, see :doc:`shell_mass`
    * - :meth:`.Quad4.update_KA_beta`, :meth:`.Quad4.update_KA_gamma`,
        :meth:`.Quad4.update_CA`
      - piston theory, :eq:`sh-piston-matrices`
    * - :meth:`.Quad4Probe.update_BL`
      - rows of the strains :eq:`sh-strains` at a point, for the recovery
        of strains


References
----------

- Allman, D. J., A compatible triangular element including vertex
  rotations for plane elasticity analysis, Computers & Structures,
  19(1-2), 1-8, 1984. https://doi.org/10.1016/0045-7949(84)90197-4
- Bathe, K. J. and Dvorkin, E. N., A four-node plate bending element based
  on Mindlin/Reissner plate theory and a mixed interpolation, International
  Journal for Numerical Methods in Engineering, 21(2), 367-383, 1985.
  https://doi.org/10.1002/nme.1620210213
- Belytschko, T. and Leviathan, I., Physical stabilization of the 4-node
  shell element with one point quadrature, Computer Methods in Applied
  Mechanics and Engineering, 113(3-4), 321-350, 1994.
  https://doi.org/10.1016/0045-7825(94)90052-3
- Dvorkin, E. N. and Bathe, K. J., A continuum mechanics based four-node
  shell element for general non-linear analysis, Engineering Computations,
  1(1), 77-88, 1984. https://doi.org/10.1108/eb023562
- Hughes, T. J. R. and Brezzi, F., On drilling degrees of freedom, Computer
  Methods in Applied Mechanics and Engineering, 72(1), 105-121, 1989.
  https://doi.org/10.1016/0045-7825(89)90124-2
- Hughes, T. J. R., Cohen, M. and Haroun, M., Reduced and selective
  integration techniques in the finite element analysis of plates, Nuclear
  Engineering and Design, 46(1), 203-222, 1978.
  https://doi.org/10.1016/0029-5493(78)90184-X
- Hughes, T. J. R., Taylor, R. L. and Kanoknukulchai, W., A simple and
  efficient finite element for plate bending, International Journal for
  Numerical Methods in Engineering, 11(10), 1529-1543, 1977.
  https://doi.org/10.1002/nme.1620111005
- MacNeal, R. H. and Harder, R. L., A proposed standard set of problems to
  test finite element accuracy, Finite Elements in Analysis and Design,
  1(1), 3-20, 1985. https://doi.org/10.1016/0168-874X(85)90003-4
