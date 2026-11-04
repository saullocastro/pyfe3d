Tria3R: discrete shear gap with the Allman membrane
===================================================

:class:`.Tria3R`, module :mod:`pyfe3d.tria3r`, is the recommended triangle
of ``pyfe3d``. It has the kinematics and the constant strains of the linear
triangle, see :doc:`shell_formulation`, and, with the default
``shear_model = 1``, the transverse shear field of :class:`.Tria3DSG`, the
discrete shear gap averaged over the three starting nodes, see
:doc:`shell_tria3dsg`. What distinguishes it from :class:`.Tria3DSG` is
the membrane, enriched by default with the drilling rotations of Allman
(1984), with the regularisation of Hughes and Brezzi (1989), see
:doc:`shell_drilling`.


Quadratures
-----------

The membrane, bending and extension-bending coupling terms, constant over
the triangle, are integrated with one point at the centroid, `S_1 = S_2 =
S_3 = 1/3`, with the weight `1/2` times `\det \boldsymbol{J} = 2 A_e`. The
transverse shear of the DSG field is added through the same
`18 \times 18` block as the drilling terms, see the Notes of :mod:`pyfe3d.tria3r`.
With the default ``drilling_model = 0``, the terms of the Allman
enrichment, whose strains are linear over the element, are integrated with
the three-point rule of Cowper (1973), with the area coordinates `(2/3,
1/6, 1/6)` and their permutations and the weight `A_e/3`, see
:meth:`.Tria3R.update_KC0`; a single point would give the element five zero
eigenvalues over its nine in-plane degrees of freedom instead of three. The
Hughes-Brezzi term is integrated with one point at the centroid, which
makes the element insensitive to `\gamma_{r_z}`. With ``drilling_model =
1`` the penalty is integrated with the three-point rule.


The Allman membrane
-------------------

The edge `k`, joining the nodes `i` and `j`, carries the hierarchical
bubble `\hat{S}_k = 4 S_i S_j` with the amplitude `a_k = \ell_k ({r_z}_i -
{r_z}_j)/8` along its normal, such that the in-plane field is enriched
without a new degree of freedom, see :doc:`shell_drilling`. The edge
modes act on the in-plane translations only: the curvatures and the
transverse shear strains are not enriched, and on a flat plate in bending,
buckling or vibration, where the membrane is decoupled from the bending,
:class:`.Tria3R` and :class:`.Tria3DSG` give the same results to
round-off. With `\boldsymbol{B} \ne \boldsymbol{0}`, for an offset or an
unsymmetric laminate, the enriched membrane strains couple `r_z` to the
curvatures. The enrichment enters the membrane forces of the geometric
stiffness and the non-linear terms, evaluated at the centroid, which
integrates its linear strains exactly with one point, see
:meth:`.Tria3R.update_KG`.

The enrichment makes the element much better in in-plane bending and
better on shells whose response mixes membrane and bending. Measured with
quadrilateral meshes split along the diagonal 1-3, normalised by the
reference, with :class:`.Tria3DSG` in parentheses:

- skew membrane of Cook, reference 23.956 of Felippa (2003), `n \times n`
  cells: 0.864, 0.948, 0.982, 0.994
  for `n` = 4, 8, 16, 32 (0.457, 0.713, 0.896, 0.970);
- Scordelis-Lo roof of Belytschko et al. (1985), Table 3, `N \times N` on
  the quarter: 0.975, 0.989, 0.994,
  0.996 for `N` = 8, 16, 32, 64 (0.853, 0.952, 0.984, 0.994);
- twisted beam of MacNeal and Harder (1985), out-of-plane load: 0.933,
  0.988, 0.997 with 12x2, 48x4 and 192x8 cells (0.817, 0.954, 0.990);
- pinched hemisphere with an 18 degree hole of Belytschko et al. (1985): 0.288, 0.862, 0.998, 1.011
  for `N` = 8, 16, 32, 64 (0.451, 0.926, 0.998, 1.009).

On the coarse meshes of the doubly curved hemisphere, the drilling
rotation of a facet is made of the rotations that bend its neighbours, and
the edge modes turn them into membrane strain, which stiffens an
inextensional deformation. The penalty model, ``drilling_model = 1``, which
has no Allman field, gives 0.929 and 0.988 at `N` = 8 and 16, whereas
reducing ``gamma_rz`` hundredfold does not change the stiffening: it is in
the enrichment, not in the Hughes-Brezzi coefficient. It is gone at `N =
32`. This is why :class:`.Tria3DSG` is the better triangle on a coarse
mesh of a doubly curved shell. The assembly of
:meth:`.Tria3R.update_KC0` takes about half the time of that of
:class:`.Tria3DSG`.


The centroid-sampled shear of ``shear_model = 0``
-------------------------------------------------

With ``shear_model = 0`` the transverse shear strains are sampled at the
centroid, and the transverse shear stiffnesses are divided by :eq:`sh-alpha`
with ``alpha_shear_locking``, e.g. ``0.7``. Here the factor does the
unlocking, since the centroid sample still locks, and its value is about
seven times the values near 0.1 of the references, where the factor sits
on top of a field that is already free of locking (Lyly, Stenberg and
Vihinen, 1993; Bischoff and Bletzinger, 2004; Castro et al., 2019). The
factor tends to zero only when `\ell` falls well below `h`, and on thin
plates and shells the error stalls under refinement at a value set by
`\alpha` and by the mesh pattern: on a simply supported square plate with
`a/h = 1000` under a uniform pressure, the centre deflection is 14.1, 14.4
and 14.3 % too small with 32x32, 64x64 and 128x128 squares cut into
triangles, the buckling load stalls at +16.4 % and the first frequency at
+8.0 %, whereas the DSG field converges to +0.02 %. Lyly, Stenberg and
Vihinen (1993), p. 348, state that the stabilisation was introduced for a
piecewise quadratic deflection and that a linear or bilinear deflection
needs a reduction operator of the MITC type with it; the centroid sample
is neither. A larger `\alpha` moves the stall at the price of a too
flexible coarse mesh, and on a thin cylinder in buckling it can make a
mode at two elements per wavelength the critical one. The measurements are
in the Notes of :mod:`pyfe3d.tria3r`.


What the element computes
-------------------------

.. list-table::
    :header-rows: 1

    * - Method
      - Theory
    * - :meth:`.Tria3R.update_KC0`
      - :eq:`sh-KC0` with the DSG shear of :doc:`shell_tria3dsg` and the
        Allman and Hughes-Brezzi terms of :doc:`shell_drilling`
    * - :meth:`.Tria3R.update_fint`, :meth:`.Tria3R.update_probe_finte`
      - gradient of the strain energy, with ``nonlinear=1`` the von Kármán
        terms
    * - :meth:`.Tria3R.update_KCNL`, :meth:`.Tria3R.update_KG`,
        :meth:`.Tria3R.update_KG_given_stress`
      - :eq:`sh-tangent` and :eq:`sh-KG`, constant over the element
    * - :meth:`.Tria3R.update_M`
      - consistent, lumped and one-point mass matrices, see
        :doc:`shell_mass`


References
----------

- Allman, D. J., A compatible triangular element including vertex
  rotations for plane elasticity analysis, Computers & Structures,
  19(1-2), 1-8, 1984. https://doi.org/10.1016/0045-7949(84)90197-4
- Bischoff, M. and Bletzinger, K.-U., Improving stability and accuracy of
  Reissner-Mindlin plate finite elements via algebraic subgrid scale
  stabilization, Computer Methods in Applied Mechanics and Engineering,
  193(15-16), 1517-1528, 2004. https://doi.org/10.1016/j.cma.2003.12.036
- Belytschko, T., Stolarski, H., Liu, W. K., Carpenter, N. and Ong, J. S.-J.,
  Stress projection for membrane and shear locking in shell finite
  elements, Computer Methods in Applied Mechanics and Engineering, 51(1-3),
  221-258, 1985. https://doi.org/10.1016/0045-7825(85)90035-0
- Castro, S. G. P., Donadon, M. V. and Guimaraes, T. A. M., ES-PIM applied
  to buckling of variable angle tow laminates, Composite Structures, 209,
  67-78, 2019. https://doi.org/10.1016/j.compstruct.2018.10.058
- Cowper, G. R., Gaussian quadrature formulas for triangles, International
  Journal for Numerical Methods in Engineering, 7(3), 405-408, 1973.
  https://doi.org/10.1002/nme.1620070316
- Felippa, C. A., A study of optimal membrane triangles with drilling
  freedoms, Computer Methods in Applied Mechanics and Engineering,
  192(16-18), 2125-2168, 2003.
  https://doi.org/10.1016/S0045-7825(03)00253-6
- Hughes, T. J. R. and Brezzi, F., On drilling degrees of freedom, Computer
  Methods in Applied Mechanics and Engineering, 72(1), 105-121, 1989.
  https://doi.org/10.1016/0045-7825(89)90124-2
- Lyly, M., Stenberg, R. and Vihinen, T., A stable bilinear element for
  the Reissner-Mindlin plate model, Computer Methods in Applied Mechanics
  and Engineering, 110(3-4), 343-357, 1993.
  https://doi.org/10.1016/0045-7825(93)90214-I
- MacNeal, R. H. and Harder, R. L., A proposed standard set of problems to
  test finite element accuracy, Finite Elements in Analysis and Design,
  1(1), 3-20, 1985. https://doi.org/10.1016/0168-874X(85)90003-4
