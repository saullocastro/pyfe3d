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
        in the Notes of :mod:`.quad4`.
    * - :class:`.Quad4R`
      - 4
      - One integration point, cheaper than :class:`.Quad4` for the
        internal forces and the stiffness matrices, with an hourglass
        control that adds an artificial stiffness, scaled by the
        ``hgfactor_*`` arguments of :meth:`.Quad4R.update_KC0`; no drilling
        enrichment, see :mod:`.quad4r`. The transverse hourglass stiffness
        also stiffens the twist, by an amount that depends on the problem,
        e.g. 6 to 7 % on the linear deflection of the hinged roof of the
        :ref:`non-linear tutorial
        </tut-nonlinear-benchmarks.ipynb#sensitivity-to-the-transverse-hourglass-stiffness-of-quad4r>`.
    * - :class:`.Tria3DSG`
      - 3
      - The recommended triangle. Discrete shear gap transverse shear field,
        free of shear locking by construction and without a tuning
        parameter, see :mod:`.tria3dsg`. No drilling enrichment: for
        problems dominated by in-plane bending, :class:`.Quad4` and
        :class:`.Tria3R` have one.
    * - :class:`.Tria3R`
      - 3
      - Kept for continuity with the results obtained before
        :class:`.Tria3DSG` existed. Its transverse shear, taken at the
        centroid, locks and is reduced by the factor of
        ``alpha_shear_locking``, ``0.7`` by default, a value that must be
        verified per problem class, see :mod:`.tria3r`.

Practical rules:

- Prefer quadrilaterals in bending-dominated regions. A mesh of linear
  triangles is stiffer than the quadrilateral mesh it is split from, and its
  response depends on the direction of the diagonals; where triangles are
  needed, alternating the diagonals reduces this bias.
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
:doc:`conventions`.


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

The arguments of each method, e.g. the ``mtype`` of the consistent or
lumped mass matrices, are documented in the API pages of the elements,
:doc:`fe_overview`.
