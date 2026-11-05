Beam properties
===============

A :class:`.BeamProp` holds everything of the cross section that the elements
:class:`.BeamC`, :class:`.BeamLR` and :class:`.Truss` read: either the
integrals of a rigid, homogeneous cross section with its moduli, or a full
sectional stiffness matrix, and the mass integrals. This chapter states what
each attribute means in the beam model of :doc:`beam_theory`, how the
sectional stiffness matrix is built from them, and when it must be given
instead.


Cross-section integrals
-----------------------

The integrals are taken over the cross section in the element axes `y`,
`z`, measured from the beam axis that joins the two nodes:

.. list-table::
    :header-rows: 1

    * - Attribute
      - Integral
      - Role in the strain energy
    * - ``A``
      - `\int dA`
      - axial and shear stiffness, `E A`, `G_{12} A`, `G_{13} A`
    * - ``Ay``, ``Az``
      - `\int y \, dA`, `\int z \, dA`
      - coupling of the axial strain with the curvatures and of the shear
        strains with the twist; zero for a centroidal axis
    * - ``Izz``, ``Iyy``
      - `\int y^2 dA`, `\int z^2 dA`
      - bending in the `x y` plane (`v`, `r_z`) and in the `x z` plane
        (`w`, `r_y`)
    * - ``Iyz``
      - `\int y z \, dA`
      - coupling of the two planes of bending; zero for principal axes
    * - ``J``
      - torsion constant
      - torsion, through ``GJ`` when ``GJ`` is not set

The first moments are named after the coordinate they integrate, ``Ay`` `=
\int y \, dA`, which some texts call `Q_z`. The moduli are ``E``, the Young
modulus along the axis, and ``G``, the shear modulus, with three derived
stiffnesses:

- ``G12``, the transverse shear modulus of the element plane `x y`, and
  ``G13``, that of the plane `x z`; both read ``G`` until they are set;
- ``GJ``, the torsional stiffness; it reads ``G*J`` until it is set.

Once set, each keeps its value independently of ``G`` and ``J``, until it
is deleted, e.g. ``del prop.G12``. A property given with ``G`` and ``J``
alone keeps the meaning it had before these attributes existed.


Shear correction and torsion
----------------------------

With the rigid section of :doc:`beam_theory`, the shear strain is uniform
over the section and the strain energy of the shear stresses, per unit
length, is `\frac{1}{2} \boldsymbol{\gamma}^\top \boldsymbol{D}_s
\boldsymbol{\gamma}` with `\boldsymbol{\gamma} = \{\bar{\gamma}_y,
\bar{\gamma}_z, \kappa_x\}^\top` and

.. math::
    :label: bp-Ds

    \boldsymbol{D}_s = \begin{bmatrix}
        G_{12} A & 0 & -G_{12} A_z \\
        0 & G_{13} A & G_{13} A_y \\
        -G_{12} A_z & G_{13} A_y & G_{12} I_{yy} + G_{13} I_{zz}
    \end{bmatrix}

The two corrections that a real section needs enter here:

- the shear stiffnesses of the actual, non-uniform, shear stress
  distribution are `k_y G A` and `k_z G A`, with shear correction factors
  that depend on the shape of the section, the material and, for laminates,
  the stacking sequence; the elements apply none, and the user sets
  ``G12`` `= k_y G` and ``G13`` `= k_z G`, e.g. `k_y = k_z = 5/6` for a
  rectangle;
- the torsional entry of the rigid section is the polar moment of the
  stiffness, exact only for circular sections and far too stiff for open
  thin-walled ones; it is replaced by ``GJ``, e.g. `G J` with the
  Saint-Venant torsion constant `J`, in which the section warps freely.

Applying a correction factor to ``G`` instead of ``G12`` and ``G13`` also
scales ``GJ`` when ``GJ`` is not set. For an arbitrary section, the shear
stiffnesses, the torsional stiffness and the couplings follow from the
cross-sectional analysis of :mod:`pyfe3d.vabs` without assumed factors, see
:doc:`cross_sectional_analysis`.


Sectional stiffness matrix
--------------------------

The elements read the stiffness only through the `6 \times 6` sectional
stiffness matrix ``C`` of the strains `\boldsymbol{\epsilon} =
\{\bar{\varepsilon}, \bar{\gamma}_y, \bar{\gamma}_z, \kappa_x, \kappa_y,
\kappa_z\}^\top` of :doc:`beam_theory`, Eq. :eq:`beam-generalized-strains`.
Until it is set, reading ``C`` returns the matrix of the 11 stiffness
attributes, Eq. :eq:`beam-C-rigid`, i.e. the integral of the rigid section
with `G J` = ``GJ`` in place of the torsional entry of Eq. :eq:`bp-Ds`. Once
set, ``C`` keeps its own 21 independent entries, and the elements no longer
read the 11 attributes, until ``del prop.C``; the mass integrals are
independent of ``C``. The pattern of Eq. :eq:`beam-C-rigid` cannot hold the
couplings of a composite or otherwise anisotropic section, which arise from
the interaction of the anisotropy with the warping of the section:

.. list-table::
    :header-rows: 1

    * - Coupling
      - Entries of ``C``
      - Rigid section, Eq. :eq:`beam-C-rigid`
    * - extension-shear
      - ``C[0, 1]``, ``C[0, 2]``
      - zero
    * - extension-twist
      - ``C[0, 3]``
      - zero
    * - extension-bending
      - ``C[0, 4]``, ``C[0, 5]``
      - `E A_y`, `E A_z`
    * - shear-shear
      - ``C[1, 2]``
      - zero
    * - shear-twist
      - ``C[1, 3]``, ``C[2, 3]``
      - `-G_{12} A_z`, `G_{13} A_y`
    * - bending-shear
      - ``C[1:3, 4:6]``
      - zero
    * - bending-twist
      - ``C[3, 4]``, ``C[3, 5]``
      - zero
    * - bending-bending
      - ``C[4, 5]``
      - `E I_{yz}`

The units of an entry are force times length to the number of its indices
that are 3, 4 or 5, e.g. force for ``C[0, 0]`` and force times length squared
for ``C[4, 4]``.

A set ``C`` must be symmetric to ``1e-10`` times its largest entry, its
symmetric part being stored, and, set or built, it must be positive
semi-definite, since the strain energy per unit length cannot be negative.
:class:`.BeamC` and :class:`.BeamLR` check it each time they read it, with a
Cholesky factorization with diagonal pivoting of the matrix scaled to a unit
diagonal, to a tolerance of ``1e-10``, and raise ``ValueError`` otherwise;
zero rows, e.g. ``GJ = 0`` in a planar model whose out-of-plane degrees of
freedom are constrained, are allowed. With couplings other than ``C[0, 3]``,
:class:`.BeamC` inverts ``C`` to build its shape functions and needs it
positive definite, see :doc:`beam_theory`. :class:`.Truss` reads only
``C[0, 0]`` and ``C[3, 3]`` and does not check ``C``.


From the cross-sectional analysis to the property
-------------------------------------------------

:mod:`pyfe3d.vabs` orders the strains as `\{\gamma_{11}, 2\gamma_{12},
2\gamma_{13}, \kappa_1, \kappa_2, \kappa_3\}`, with `x_2 = y` and `x_3 = z`,
and its curvatures are the components of the curvature vector, `\kappa_2 =
\kappa_z` and `\kappa_3 = -\kappa_y`. Hence the VABS indices ``0, 1, 2, 3, 4,
5`` become ``0, 1, 2, 3, 5, 4`` and the entries that couple `\kappa_3` with
another strain change sign, which :func:`.yu_to_pyfe3d` does.
:func:`.beamprop_from_vabs` creates a :class:`.BeamProp` whose ``C`` is a
matrix in the order of pyfe3d, e.g. ``yu_to_pyfe3d(sec.Kt)``, fills its 11
attributes from the entries of ``C`` for information, modulus-weighted with
a reference Young modulus, and, given the :class:`.CrossSection`, copies the
mass integrals of :meth:`.CrossSection.calc_mass`.
After ``del prop.C`` the 11 attributes alone represent the section: the shear
centre is then the tension centre and the couplings are lost. See
:doc:`cross_sectional_analysis`.


Mass integrals
--------------

The kinetic energy of the beam elements, Eq. :eq:`beam-kinetic`, uses six
integrals of the density `\rho(y, z)` over the section: ``intrho`` `= \int
\rho \, dA`, the mass per unit length; ``intrhoy`` `= \int \rho y \, dA` and
``intrhoz`` `= \int \rho z \, dA`, which couple the inertia of the
translations and of the rotations when the axis is not through the centre of
mass; ``intrhoy2`` `= \int \rho y^2 dA`, ``intrhoz2`` `= \int \rho z^2 dA`
and ``intrhoyz`` `= \int \rho y z \, dA`, the rotary inertia, with
``intrhoy2 + intrhoz2`` the polar inertia of the twist. For a homogeneous
section of density `\rho`, ``intrho`` `= \rho A`, ``intrhoy2`` `= \rho
I_{zz}`, ``intrhoz2`` `= \rho I_{yy}` and ``intrhoyz`` `= \rho I_{yz}`. The
lumped mass matrices drop ``intrhoy``, ``intrhoz`` and ``intrhoyz``, see
:doc:`beam_theory`.
