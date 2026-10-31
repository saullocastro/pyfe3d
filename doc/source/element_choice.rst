Choosing an element
===================

All the elements have six dofs per node and can be combined in one model,
e.g. shells stiffened by beams, or beams connected by springs, since they
share the dofs of the nodes, see :doc:`conventions`.


Shells
------

The shell elements are flat, with first-order shear deformation theory
(FSDT) kinematics, the constitutive matrices of a :class:`.ShellProp`, see
:doc:`shell_properties`, and the von Kármán membrane strains in the
geometrically non-linear analyses.

.. list-table::
    :header-rows: 1

    * - Element
      - Nodes
      - Use
    * - :class:`.Quad4`
      - 4
      - The recommended quadrilateral. Assumed transverse shear strains of
        the MITC4 element against shear locking, two-by-two bending, and
        the drilling enrichment of the in-plane displacement field, see
        :mod:`.quad4`; only the six rigid-body modes have zero energy and
        the element passes the membrane and bending patch tests. Up to
        0.11, and with ``shear_model = 0``, the transverse shear of the
        elements with `h/\ell < 1` was integrated with one point, after
        Hughes et al. (1977), whose two zero-energy modes made a strip one
        element wide too flexible in torsion, see the zero-energy modes in
        the Notes of :mod:`.quad4`; the geometric stiffness takes the
        slopes of `w` at the centroid, such that the hourglass mode of `w`
        is not a spurious buckling mode, see :meth:`.Quad4.update_KG`.
    * - :class:`.Quad4R`
      - 4
      - One integration point, cheaper than :class:`.Quad4` for the
        internal forces and the stiffness matrices; no drilling enrichment,
        see :mod:`.quad4r`. The bending and the transverse shear, the
        assumed strains of Dvorkin and Bathe, have the physical
        stabilisation of Belytschko and Leviathan (1994), without parameter,
        which controls the hourglass modes of `w`, `r_x` and `r_y` and
        passes the constant-twist and the thin-plate bending patch tests;
        the hourglass modes of `u` and `v` have the artificial stiffness of
        Brockman (1987), with the membrane moduli of a laminate, and the
        one-element-deep strips of MacNeal and Harder lock in in-plane
        bending. ``hgfactor_physical`` and ``hgfactor_membrane`` of
        :meth:`.Quad4R.update_KC0` multiply the two. Up to 0.11 the
        hourglass modes of `w`, `r_x` and `r_y` had Brockman's artificial
        stiffnesses, set with ``hgfactor_w``, ``hgfactor_rx`` and
        ``hgfactor_ry``, which also stiffened a uniform twist, by up to 78 %
        of its energy whatever the mesh where the element areas were large
        in the unit of length.
    * - :class:`.Tria3DSG`
      - 3
      - Discrete shear gap transverse shear field, free of shear locking
        by construction and without a tuning parameter, see
        :mod:`.tria3dsg`. Constant-strain membrane without drilling
        enrichment, stiff in in-plane bending: Cook's membrane is 54 % too
        stiff with 4x4 cells split in two, against 14 % for
        :class:`.Tria3R`. Same results as :class:`.Tria3R` on flat plates
        in bending, buckling and vibration. Preferable on coarse meshes of
        doubly curved shells, where it is the less stiff of the two
        triangles: 0.45 against 0.29 of the reference on the pinched
        hemisphere with 8x8 cells, the two agreeing from 32x32, see
        :mod:`.tria3r`.
    * - :class:`.Tria3R`
      - 3
      - The recommended triangle. Since 0.12.0 the same discrete shear gap
        transverse shear field as :class:`.Tria3DSG`, ``shear_model = 1``,
        free of shear locking without a tuning parameter, with the membrane
        enriched by the drilling rotations of Allman: identical to
        :class:`.Tria3DSG` on flat plates in bending, buckling and
        vibration, better in in-plane bending, on the Scordelis-Lo roof and
        on the twisted beam, stiffer on the coarse meshes of the pinched
        hemisphere, and assembled in half the time, see :mod:`.tria3r`.
        ``shear_model = 0`` with ``alpha_shear_locking = 0.7`` reproduces
        the centroid-sampled shear of the versions up to 0.11, which does not
        converge on thin plates and shells: -14 % on the deflection of a
        simply supported plate with `a/h = 1000`.

Practical rules:

- Prefer quadrilaterals in bending-dominated regions. A mesh of linear
  triangles is stiffer than the quadrilateral mesh it is split from, and its
  response depends on the direction of the diagonals; where triangles are
  needed, alternating the diagonals reduces this bias. :class:`.Tria3R` is
  the recommended triangle, :class:`.Tria3DSG` the alternative for coarse
  meshes of doubly curved shells.
- The four nodes of a quadrilateral in a curved surface are, in general,
  not coplanar, and the element is flat: keep the warping small, see
  `Warped quadrilaterals`_ below. The triangles are always flat.
- The distorted-mesh variants of the benchmarks of MacNeal and Harder
  (1985), `doi:10.1016/0168-874X(85)90003-4
  <https://doi.org/10.1016/0168-874X(85)90003-4>`_, show how much
  distortion an element tolerates.
- The drilling rotation `r_z` of all the shells has, by default, a
  physics-based stiffness, ``drilling_model = 0``, without user parameter.
  ``drilling_model = 1`` selects the fictitious penalty of Nastran,
  controlled by ``K6ROT``, which reproduces the results of versions older
  than 0.10.0.
- The aerodynamic matrices of the piston theory, ``update_KA_beta``,
  ``update_KA_gamma`` and ``update_CA``, exist for :class:`.Quad4` and
  :class:`.Quad4R` only.
- For large rotations, every element can be added to a
  :class:`.CorotationalModel`, see :doc:`nonlinear_analyses`.


Warped quadrilaterals
---------------------

The matrices of :class:`.Quad4` and :class:`.Quad4R` are those of the
projected facet: the flat element in the mean plane of the four nodes,
the plane of the element coordinates, parallel to both diagonals. The
nodes of a warped element are at the offsets `\tilde{z}_1 = \tilde{z}_3 =
-\tilde{z}_2 = -\tilde{z}_4` from the mean plane, and the warping ratio
`d_w/\ell = \max_a|\tilde{z}_a|/\sqrt{A}`, with `A` the area of the
projected facet, measures it; it is the attribute ``warp_ratio`` of
:class:`.CorotationalShell`. A rigid rotation of the warped nodes about an
in-plane axis moves the nodes of the facet in an alternating in-plane
pattern, which strains it, and the facet alone is too stiff in warped
meshes.

Since version 0.12.0 the attribute ``warp_offsets`` of both elements,
``1`` by default, links the facet to the nodes by rigid offsets along its
normal, `u^f_a = u_a - \tilde{z}_a r_{y,a}` and `v^f_a = v_a + \tilde{z}_a
r_{x,a}` in element coordinates, `\boldsymbol{u}^f = \boldsymbol{J}
\boldsymbol{u}_e`: ``KC0``, ``M``, ``KG``, ``KCNL`` and the internal
forces become `\boldsymbol{J}^\top \boldsymbol{K} \boldsymbol{J}` and
`\boldsymbol{J}^\top \boldsymbol{f}` of the facet evaluated at
`\boldsymbol{J} \boldsymbol{u}_e`, the geometric stiffness adds the
eccentricity term `\boldsymbol{K}_E` of the offsets on the rotations of
each node, which ``KCNL`` subtracts, ``KCNL`` being `\boldsymbol{J}^\top
\boldsymbol{K}^f \boldsymbol{J} - \boldsymbol{K}_E` with
`\boldsymbol{K}^f` the ``KCNL`` of the facet, so that ``KC0 +
KCNL + KG`` remains the exact Jacobian of the von Kármán internal forces,
and ``KC0`` annihilates the six rigid motions of the warped nodes, see the
Notes of :class:`.Quad4`, :meth:`.Quad4.update_KG` and
:meth:`.Quad4.update_KCNL`. Flat elements
give the same results, bit for bit, with ``warp_offsets`` ``0`` and ``1``,
and ``0`` reproduces the versions up to 0.11. The correction is derived in
pyfe3d; MacNeal (1978), p. 180, and Cook (1994), p. 552, correct warped
quadrilaterals differently, balancing the moment of the offset membrane
forces with normal nodal forces, and Belytschko, Wong and Chiang (1992),
Section 3.3, with warping terms in the curvatures.

The twisted beam of MacNeal and Harder (1985), Fig. 6, with 12x2 elements
and a warping ratio of 0.024, gives these tip displacements in plane and
out of plane, normalised by those of their Table 3:

.. list-table::
    :header-rows: 1

    * - Element
      - ``warp_offsets = 1``
      - ``warp_offsets = 0``
    * - :class:`.Quad4`
      - 0.996, 0.997
      - 0.25, 0.33
    * - :class:`.Quad4R`
      - 0.991, 0.930
      - 0.24, 0.32

The remaining error of :class:`.Quad4R` is that of its in-plane bending
with one element across the width. With 6x1 :class:`.Quad4` elements,
warping ratio 0.049, the tip displacements are 0.6 % from those of a 96x16
mesh, against 74 to 91 % without the offsets. On the same beam, the offsets
also bring the linear buckling load under an axial tip load and the first
natural frequency of coarse meshes to those of fine meshes, see
:doc:`eigenvalue_analyses`. ``tests/test_quad_warp_offsets.py`` checks the
correction and the twisted beam.

Practical rules for warped quadrilaterals:

- Keep the default ``warp_offsets = 1``; ``0`` serves only to reproduce
  earlier results. The offsets were measured up to the warping ratio of
  0.049 above, and :class:`.CorotationalShell` warns above 0.07, the
  ratio up to which its exact offsets were verified: beyond it, refine the
  mesh or use triangles.
- :meth:`.Quad4Probe.update_BL` takes the ``warp_offsets`` of the element,
  such that the strains recovered with ``ue`` are those of the facet.
- ``update_KG`` and ``update_KCNL`` of :class:`.Quad4R` take the
  ``hgfactor_*`` arguments for the nodal forces of the eccentricity term
  of a warped element: pass them the values of ``update_KC0``, and the
  same values to both for the exact tangent stiffness matrix of the von
  Kármán internal forces.
- The lumped mass matrix of a warped element, `\boldsymbol{J}^\top
  \boldsymbol{M} \boldsymbol{J}`, couples the translations `u`, `v` with
  the rotations `r_y`, `r_x` of the same node, and remains block-diagonal
  per node.


Beams, truss and spring
-----------------------

.. list-table::
    :header-rows: 1

    * - Element
      - Use
    * - :class:`.BeamC`
      - The recommended beam. Timoshenko kinematics with the consistent
        shape functions of Luo (2008), which solve the static equilibrium
        equations of the beam with an uncoupled section, such that one
        element gives the exact tip deflection of a cantilever with shear
        deformation, and the nodal displacements of a frame under nodal
        loads, see :doc:`ex_beam_frame`. With couplings in the sectional
        stiffness matrix, other than extension-twist, e.g. an axis off the
        centroid or the bending-shear couplings of a composite section, it
        converges as the square of the element length, see the Notes of
        :class:`.BeamC`. Since version 0.12.0 its von Kármán axial
        strain takes the element mean of `(v_{,x}^2 + w_{,x}^2)/2`, after
        Crisfield (1991), Section 7.1.3, which removes its membrane
        locking: the central deflection of a beam clamped at both ends,
        with a linear deflection three times its height, is 0.60 % below
        the converged one with 4 elements, against 8.1 % before, and the
        lower limit load of the arch of Fung and Kaplan (1952) with 40
        elements is within 0.38 % of theirs, against 5.8 %, see the Notes
        of :class:`.BeamC` and
        ``tests/test_solver_arc_length_fung_kaplan.py``.
    * - :class:`.BeamLR`
      - Timoshenko kinematics with linear interpolation and reduced
        integration; it converges to the same solutions with mesh
        refinement, quadratically, and needs many more elements, see
        ``tests/test_timoshenko_beam_literature.py``. Since version 0.12.0
        its buckling load includes the reduction due to shear deformation,
        see :doc:`ex_linear_buckling_column`, and its constant slopes leave
        it free of membrane locking in the geometrically non-linear
        cantilever under an end moment, as :class:`.BeamC` with its
        averaged axial strain, see :doc:`tut-nonlinear-benchmarks`. Its
        interpolation does not depend on the sectional stiffness matrix,
        and it converges as fast with strong bending-shear couplings, e.g.
        of composite box beams, as without them, see
        :doc:`cross_sectional_analysis`.
    * - :class:`.Truss`
      - Axial and torsional stiffness only, linear analyses and mass matrix,
        without geometric stiffness matrix, hence no linear buckling. The
        documentation of :mod:`.truss` recommends :class:`.BeamC` instead.
    * - :class:`.Spring`
      - Six constant stiffnesses, ``kxe, kye, kze, krxe, krye, krze``,
        between the same dofs of two nodes, in the element coordinate
        system, e.g. for elastic supports and connections; no mass and no
        geometric stiffness matrix.

The cross section of the beams and of the truss is given by a
:class:`.BeamProp`, whose integrals are taken in the element coordinate
system, see :doc:`shell_properties`. Its orientation is defined by the
vector ``vxyi, vxyj, vxyk`` of ``update_rotation_matrix``, see
:doc:`conventions`. The beams have a transverse shear modulus for each
plane of bending, ``G12`` for the element `xy` plane and ``G13`` for the
`xz` plane, and the beams and the truss take the torsional stiffness
``GJ``; the three follow ``G`` unless they are set. The beams read the
stiffness through the `6 \times 6` sectional stiffness matrix ``C`` of
:class:`.BeamProp`, built from these attributes unless it is set, e.g. by
:func:`.beamprop_from_vabs` with the 21 entries of a composite section;
the truss reads only its entries `E A` and `G J`.


Methods of each element
-----------------------

.. list-table::
    :header-rows: 1

    * - Method
      - Quad4, Quad4R
      - Tria3DSG, Tria3R
      - BeamC, BeamLR
      - Truss
      - Spring
    * - ``update_KC0``
      - yes
      - yes
      - yes
      - yes
      - yes
    * - ``update_fint``, ``nonlinear=1``
      - yes
      - yes
      - yes
      - linear only
      - linear only
    * - ``update_KCNL``
      - yes
      - yes
      - yes
      -
      -
    * - ``update_KG``
      - yes
      - yes
      - yes
      -
      -
    * - ``update_KG_given_stress``
      - yes
      - yes
      -
      -
      -
    * - ``update_M``
      - yes
      - yes
      - yes
      - yes
      -
    * - ``update_KA_beta``, ``update_KA_gamma``, ``update_CA``
      - yes
      -
      -
      -
      -

The arguments of each method are documented in the API pages of the
elements, :doc:`fe_overview`.


Mass matrices
-------------

The argument ``mtype`` of ``update_M`` selects how the kinetic energy of
the element is integrated, with the same values for every element:

.. list-table::
    :header-rows: 1

    * - ``mtype``
      - Mass matrix
      - Elements
    * - ``0``
      - consistent
      - Quad4, Quad4R, Tria3R, Tria3DSG, BeamC, BeamLR, Truss
    * - ``1``
      - lumped
      - Quad4, Quad4R, Tria3R, Tria3DSG, BeamC, BeamLR, Truss
    * - ``2``
      - reduced integration, one point
      - Quad4, Quad4R, Tria3R

Any other value, and ``2`` for an element without a reduced-integration
mass matrix, raises a ``ValueError``. Each matrix is the integral over
the element of the inertia of the section between the interpolated
velocities: for the shells, `I_\rho`, `I_{\rho z}` and `I_{\rho z^2}`, the
``intrho``, ``intrhoz`` and ``intrhoz2`` of the :class:`.ShellProp`, which
include the offset, see :doc:`shell_properties`; for the beams, the
integrals of the :class:`.BeamProp`. The consistent matrices integrate it
exactly. The lumped matrices integrate it with the points at the nodes,
where each shape function is 1 at its own node and 0 at the others, such
that, for the shells, node `i` carries the weight `w_i` times

.. math::

    \begin{bmatrix}
        I_\rho & 0 & 0 & 0 & I_{\rho z} & 0 \\
        0 & I_\rho & 0 & -I_{\rho z} & 0 & 0 \\
        0 & 0 & I_\rho & 0 & 0 & 0 \\
        0 & -I_{\rho z} & 0 & I_{\rho z^2} & 0 & 0 \\
        I_{\rho z} & 0 & 0 & 0 & I_{\rho z^2} & 0 \\
        0 & 0 & 0 & 0 & 0 & 0
    \end{bmatrix}

in element coordinates, for `u, v, w, r_x, r_y, r_z`, with `w_i = A/3` for
the triangles and the Jacobian determinant at the node for the
quadrilaterals, and no coupling between nodes. A lumped shell matrix is
diagonal only without offset, when `I_{\rho z} = 0`; with an offset, the
coupling keeps the centre of mass of the element at the offset. A warped
quadrilateral adds the coupling of its rigid offsets, see `Warped
quadrilaterals`_. The mass,
first moment and rotary inertia given by each scheme are discussed in the
Notes of ``update_M``, e.g. :meth:`.Tria3R.update_M` and
:meth:`.Quad4R.update_M`. The lumped beam matrices are diagonal, without
the first moments ``intrhoy``, ``intrhoz`` and the product ``intrhoyz`` of
the section, see :meth:`.BeamC.update_M`.


References
----------

- Belytschko, T., Wong, B. L. and Chiang, H.-Y., Advances in one-point
  quadrature shell elements, Computer Methods in Applied Mechanics and
  Engineering, 96(1), 93-107, 1992.
  https://doi.org/10.1016/0045-7825(92)90100-X
- Cook, R. D., Four-node 'flat' shell element: drilling degrees of
  freedom, membrane-bending coupling, warped geometry, and behavior,
  Computers & Structures, 50(4), 549-555, 1994.
  https://doi.org/10.1016/0045-7949(94)90025-6
- Crisfield, M. A., Non-linear Finite Element Analysis of Solids and
  Structures, Volume 1: Essentials, John Wiley & Sons, Chichester, 1991.
- Fung, Y. C. and Kaplan, A., Buckling of low arches or curved beams of
  small curvature, NACA Technical Note 2840, National Advisory Committee
  for Aeronautics, Washington, 1952.
  https://ntrs.nasa.gov/citations/19930083579
- MacNeal, R. H., A simple quadrilateral shell element, Computers &
  Structures, 8(2), 175-183, 1978.
  https://doi.org/10.1016/0045-7949(78)90020-2
- MacNeal, R. H. and Harder, R. L., A proposed standard set of problems to
  test finite element accuracy, Finite Elements in Analysis and Design,
  1(1), 3-20, 1985. https://doi.org/10.1016/0168-874X(85)90003-4
