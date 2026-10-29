Choosing an element
===================

All the elements have six dofs per node and can be combined in one model,
e.g. shells stiffened by beams, or beams connected by springs, since they
share the dofs of the nodes, see :doc:`conventions`.


Shells
------

The shell elements are flat, with first-order shear deformation theory
(FSDT) kinematics, the constitutive matrices of a :class:`.ShellProp`, see
:doc:`shell_properties`, and the von Karman membrane strains in the
geometrically non-linear analyses.

.. list-table::
    :header-rows: 1

    * - Element
      - Nodes
      - Use
    * - :class:`.Quad4`
      - 4
      - The recommended quadrilateral. Mixed quadrature of the transverse
        shear terms against shear locking, and the drilling enrichment of
        the in-plane displacement field, see :mod:`.quad4`. A strip one
        element wide is too flexible in torsion, see the zero-energy modes
        in the Notes of :mod:`.quad4`; the geometric stiffness takes the
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
  not coplanar, and the element is flat: keep the warping, the
  out-of-plane deviation of the nodes over the in-plane size, small. The
  triangles are always flat.
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


Beams, truss and spring
-----------------------

.. list-table::
    :header-rows: 1

    * - Element
      - Use
    * - :class:`.BeamC`
      - The recommended beam. Timoshenko kinematics with the consistent
        shape functions of Luo (2008), which solve the static equilibrium
        equations of the beam, such that one element gives the exact tip
        deflection of a cantilever with shear deformation, and the nodal
        displacements of a frame under nodal loads, see
        :doc:`ex_beam_frame`.
    * - :class:`.BeamLR`
      - Timoshenko kinematics with linear interpolation and reduced
        integration; it converges to the same solutions with mesh
        refinement, quadratically, and needs many more elements, see
        ``tests/test_timoshenko_beam_literature.py``.
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
``GJ``; the three follow ``G`` unless they are set.


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
coupling keeps the centre of mass of the element at the offset. The mass,
first moment and rotary inertia given by each scheme are discussed in the
Notes of ``update_M``, e.g. :meth:`.Tria3R.update_M` and
:meth:`.Quad4R.update_M`. The lumped beam matrices are diagonal, without
the first moments ``intrhoy``, ``intrhoz`` and the product ``intrhoyz`` of
the section, see :meth:`.BeamC.update_M`.
