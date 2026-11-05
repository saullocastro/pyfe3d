Drilling rotations
==================

The rotation `r_z` about the normal of a shell element, the drilling
rotation, appears in none of the strains of the FSDT, see
:eq:`sh-strains`. Without a term of its own, its rows and columns in the
element matrix are empty, and a mesh of coplanar elements gives a singular
global stiffness matrix; when the elements are not coplanar, the drilling
rotation of one element combines with the bending rotations of its
neighbours. Shells connected to beams also need the drilling rotation to
transmit in-plane moments. Every shell element of ``pyfe3d`` ties `r_z` to
the in-plane rotation of the membrane field,

.. math::
    :label: sh-thetaz

    \theta_z = \frac{1}{2} \left( v_{,x} - u_{,y} \right)

with the operator

.. math::
    :label: sh-Brz

    \boldsymbol{B}_{r_z} = \boldsymbol{S}^{r_z} + \frac{1}{2}
    \boldsymbol{S}^u_{,y} - \frac{1}{2} \boldsymbol{S}^v_{,x}, \qquad
    r_z - \theta_z = \boldsymbol{B}_{r_z} \boldsymbol{u}_e

the row ``BLdrilling`` of the probes, see :meth:`.Quad4Probe.update_BL`.
Being built from `r_z - \theta_z`, and not added on the diagonal, the term
does not stiffen a rigid rotation of the element about its normal, such
that all rigid-body motions and constant-strain states are represented
exactly. Two models, selected with the attribute ``drilling_model`` of each
element, give it a coefficient.


Fictitious penalty, ``drilling_model = 1``
------------------------------------------

The approach of MSC Nastran and Autodesk Nastran, through their parameter
``K6ROT``, adds the penalty energy

.. math::
    :label: sh-k6rot

    U_{r_z} = \frac{1}{2} K6ROT \cdot 10^{-6} \int_A A_{66} \left( r_z -
    \theta_z \right)^2 dA

whose only purpose is to remove the singularity; the forces associated
with it are spurious, and a moment about the normal recovered from the
internal forces is meaningless. The quick reference guide of Autodesk
Nastran recommends ``K6ROT = 100`` for static analyses and `10^4` for
modal analyses, whereas that of MSC Nastran states that values above 100
should not be used. The approach is similar to Eq. (2.20) of Adam et al.
(2013).


Physics-based stiffness, ``drilling_model = 0``
-----------------------------------------------

The default replaces the penalty coefficient by the regularisation of
Hughes and Brezzi (1989), with the coefficient `\gamma_{r_z} = A_{66}` of
the laminate, `G h` for an isotropic plate, the value that they identify
as natural for the isotropic problem. In :class:`.Quad4` and
:class:`.Tria3R` the in-plane field is, in addition, enriched with the
drilling rotations after Allman (1984), in the combination of
Ibrahimbegovic, Taylor and Wilson (1990) for the quadrilateral, such that
the drilling rotation produces membrane strain energy. Each edge `k`,
joining the nodes `i` and `j`, of length `\ell_k` and unit normal
`\boldsymbol{n}_k`, carries a quadratic normal displacement whose end
slopes are identified with the drilling rotations of its nodes:

.. math::
    :label: sh-allman

    \begin{Bmatrix} u \\ v \end{Bmatrix} = \sum_i S_i
    \begin{Bmatrix} u_i \\ v_i \end{Bmatrix} + \sum_k \hat{S}_k \,
    \frac{\ell_k}{8} \left( {r_z}_i - {r_z}_j \right) \boldsymbol{n}_k

The enrichment introduces no new degree of freedom: its amplitudes are the
differences of the drilling rotations at the ends of each edge. The
hierarchical bubble `\hat{S}_k` is 1 at the mid-point of the edge and 0
at every corner:

.. math::
    :label: sh-allman-bubbles

    \text{quadrilateral:} \quad
    \hat{S}_k = \frac{1}{2} (1 - \xi^2)(1 + \eta_k \eta) \;\;
    (\eta = \eta_k = \pm 1), \quad
    \hat{S}_k = \frac{1}{2} (1 + \xi_k \xi)(1 - \eta^2) \;\;
    (\xi = \xi_k = \pm 1);
    \qquad
    \text{triangle:} \quad \hat{S}_k = 4 S_i S_j

the mid-side functions of the eight-node serendipity element for the
quadrilateral, see the Notes of :mod:`pyfe3d.quad4` and
:mod:`pyfe3d.tria3r`. The enrichment fills the drilling
columns of the membrane rows, `\tilde{\boldsymbol{B}}_m`, and of the
drilling row, `\tilde{\boldsymbol{B}}_{r_z} = \boldsymbol{S}^{r_z} +
\frac{1}{2} \tilde{\boldsymbol{S}}^u_{,y} - \frac{1}{2}
\tilde{\boldsymbol{S}}^v_{,x}`, and adds to the stiffness matrix

.. math::
    :label: sh-hughes-brezzi

    \boldsymbol{K}_{C0_e}^{r_z} = \gamma_{r_z} \int_A
    \tilde{\boldsymbol{B}}_{r_z}^\top \tilde{\boldsymbol{B}}_{r_z} \, dA,
    \qquad \gamma_{r_z} = A_{66}

The curvature and transverse shear rows are not enriched. Allman's
enrichment alone is rank-deficient by one: `u_i = v_i = 0` with the same
`{r_z}_i = \omega_0` at every node makes every amplitude vanish and
produces no strain, although it is not a rigid motion; the term
:eq:`sh-hughes-brezzi` gives it the energy `\frac{1}{2} \gamma_{r_z} A_e
\omega_0^2` and restores the rank of the element.

Two quadratures matter, following Ibrahimbegovic et al. (1990). The edge
modes make the membrane strains vary linearly, and the membrane is
integrated with three by three points in :class:`.Quad4` and the Allman
terms with three points in :class:`.Tria3R`: with two by two points the
alternating pattern of the drilling rotations of the quadrilateral
produces no membrane strain at any of the four points, a zero-energy mode
that no value of `\gamma_{r_z}` removes. The term :eq:`sh-hughes-brezzi`
is integrated with one point at the centroid: fully integrated, it
over-constrains `r_z = \theta_z` and locks the membrane response as
`\gamma_{r_z}` grows, whereas with one point the response reaches an
asymptote, and one constraint per element is what is needed to remove the
uniform drilling mode. The element response then has a broad plateau of
insensitivity to `\gamma_{r_z}`, and the attribute ``gamma_rz``, negative
by default to select `A_{66}`, serves the sensitivity study that the
literature recommends rather than normal use.

Unlike the penalty, the term is consistent: stationarity with respect to
`r_z` gives `\gamma_{r_z} (r_z - \theta_z) = 0` pointwise, such that the
exact solution has `r_z = \theta_z` and the term contributes no energy for
any positive `\gamma_{r_z}`. The nodal moments about the normal recovered
from the internal forces are physical, and in-plane moments are
transmitted between shells and beams through the shared drilling degree of
freedom. With `\boldsymbol{B} \ne \boldsymbol{0}`, for an offset or an
unsymmetric laminate, the enriched membrane strains couple `r_z` to the
curvatures. The enrichment also enters the membrane forces of the
geometric stiffness and the non-linear terms, such that ``KC0 + KCNL +
KG`` remains the exact Jacobian of the internal forces.


The drilling term of each element
---------------------------------

.. list-table::
    :header-rows: 1

    * - Element
      - ``drilling_model = 0``
      - ``drilling_model = 1``
    * - :class:`.Quad4`
      - Allman enrichment, membrane with 3x3 points, term
        :eq:`sh-hughes-brezzi` at the centroid
      - :eq:`sh-k6rot` with 2x2 points, membrane with 2x2 points
    * - :class:`.Quad4R`
      - no enrichment, :eq:`sh-Brz` with `\gamma_{r_z} = A_{66}`, 2x2
        points
      - same operator and quadrature, `\gamma_{r_z} = K6ROT \cdot 10^{-6}
        A_{66}`
    * - :class:`.Tria3DSG`
      - no enrichment, :eq:`sh-Brz` with `\gamma_{r_z} = A_{66}`, 3 points
      - same operator and quadrature, `\gamma_{r_z} = K6ROT \cdot 10^{-6}
        A_{66}`
    * - :class:`.Tria3R`
      - Allman enrichment with 3 points, term :eq:`sh-hughes-brezzi` at
        the centroid
      - :eq:`sh-k6rot` with 3 points

The default ``K6ROT`` is ``100.`` for :class:`.Quad4`, :class:`.Quad4R` and
:class:`.Tria3R` and ``1.`` for :class:`.Tria3DSG`, see the attributes of
each class. :class:`.Quad4R` has no enrichment because its one-point
integration already removes the excessive in-plane bending stiffness that
the enrichment removes in :class:`.Quad4`, see :doc:`shell_quad4r`, and
:class:`.Tria3DSG` because it is aimed at the transverse shear, see
:doc:`shell_tria3dsg`. The three-point rule of :class:`.Tria3DSG` is
needed, since one point would constrain a single combination of its three
drilling rotations.


Choosing a drilling model
-------------------------

The physics-based model is the default because it is the correct one when
the drilling moment is part of the load path, when shells are connected to
beams or stiffeners that transmit in-plane moments, or when the mesh is too
coarse for the unenriched membrane to be trusted. It is much more accurate
in in-plane bending: on the skew membrane of Cook, reference 23.9, a two
by two mesh of :class:`.Quad4` gives 20.8 against 11.8 with the penalty,
and a four by four mesh of split quadrilaterals of :class:`.Tria3R` gives
20.7 against 11.3. On coarse meshes of doubly curved shells the
enrichment stiffens :class:`.Tria3R`, see :doc:`shell_tria3r`.


References
----------

- Adam, F. M., Mohamed, A. E. and Hassaballa, A. E., Degenerated four
  nodes shell element with drilling degree of freedom, IOSR Journal of
  Engineering, 3(8), 10-20, 2013.
- Allman, D. J., A compatible triangular element including vertex
  rotations for plane elasticity analysis, Computers & Structures,
  19(1-2), 1-8, 1984. https://doi.org/10.1016/0045-7949(84)90197-4
- Hughes, T. J. R. and Brezzi, F., On drilling degrees of freedom, Computer
  Methods in Applied Mechanics and Engineering, 72(1), 105-121, 1989.
  https://doi.org/10.1016/0045-7825(89)90124-2
- Ibrahimbegovic, A., Taylor, R. L. and Wilson, E. L., A robust
  quadrilateral membrane finite element with drilling degrees of freedom,
  International Journal for Numerical Methods in Engineering, 30(3),
  445-457, 1990. https://doi.org/10.1002/nme.1620300305
