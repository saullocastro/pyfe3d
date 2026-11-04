Cross-sectional analysis of beams
=================================

The beam elements :class:`.BeamC` and :class:`.BeamLR` take the cross
section through the sectional stiffness matrix ``C`` of a
:class:`.BeamProp`, which by default is built from the integrals of a
rigid cross section, with the torsional stiffness ``GJ`` and the shear
moduli ``G12`` and ``G13``, and their correction factors, given by the
user, see :doc:`beam_properties`. The module :mod:`pyfe3d.vabs`
computes these stiffnesses, and the couplings between them, for an
arbitrary cross section made of anisotropic and non-homogeneous materials,
with the variational asymptotic beam sectional analysis (VABS) of Cesnik
and Hodges (1997) and Yu et al. (2002, 2012): the three-dimensional
elasticity problem of a slender prismatic beam is split into a
two-dimensional problem over the cross section, solved here with finite
elements, and a one-dimensional beam problem, governed by the resulting
sectional stiffness matrix.

The rigid cross section of :doc:`beam_theory` condenses the section into the
integrals of :doc:`beam_properties`, and cannot give the torsional stiffness
of a non-circular section, the actual distribution of the transverse shear
stresses, nor the extension-twist, bending-twist and bending-shear couplings
that the anisotropy of a composite section produces through its warping. The
cross-sectional analysis lets the section warp freely, in and out of its
plane, and finds the warping that minimizes the strain energy for given
strains of the beam.


Variational asymptotic formulation
----------------------------------

**Strains of a warping section.** The beam axis is `x_1` and the cross
section lies in the plane `x_2 x_3`. The displacement of a point of the
section is the rigid motion of the section plus a warping `\boldsymbol{w} =
\{w_1, w_2, w_3\}`, discretized as `\boldsymbol{w} = \boldsymbol{S}
\boldsymbol{V}` with the shape functions `\boldsymbol{S}` of the
two-dimensional elements and three warping degrees of freedom per node,
`\boldsymbol{V}`. For a prismatic beam the three-dimensional strains
`\boldsymbol{\Gamma} = \{\Gamma_{11}, 2\Gamma_{12}, 2\Gamma_{13},
\Gamma_{22}, 2\Gamma_{23}, \Gamma_{33}\}` are, Eq. (34) of Yu et al. (2012),

.. math::
    :label: vabs-strain

    \boldsymbol{\Gamma} = \boldsymbol{\Gamma}_h \boldsymbol{S}
    \boldsymbol{V} + \boldsymbol{\Gamma}_\epsilon \bar{\boldsymbol{\epsilon}}
    + \boldsymbol{\Gamma}_l \boldsymbol{S} \boldsymbol{V}'

with the classical strain measures of the beam `\bar{\boldsymbol{\epsilon}} =
\{\gamma_{11}, \kappa_1, \kappa_2, \kappa_3\}`, the prime the derivative
along `x_1`, and the operators

.. math::
    :label: vabs-operators

    \boldsymbol{\Gamma}_h = \begin{bmatrix}
        0 & 0 & 0 \\ \partial_2 & 0 & 0 \\ \partial_3 & 0 & 0 \\
        0 & \partial_2 & 0 \\ 0 & \partial_3 & \partial_2 \\
        0 & 0 & \partial_3 \end{bmatrix}
    \qquad
    \boldsymbol{\Gamma}_\epsilon = \begin{bmatrix}
        1 & 0 & x_3 & -x_2 \\ 0 & -x_3 & 0 & 0 \\ 0 & x_2 & 0 & 0 \\
        0 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 \end{bmatrix}
    \qquad
    \boldsymbol{\Gamma}_l = \begin{bmatrix}
        \boldsymbol{I}_3 \\ \boldsymbol{0}_3 \end{bmatrix}

`\partial_\alpha = \partial/\partial x_\alpha`. The first row of
`\boldsymbol{\Gamma}_\epsilon` gives the axial strain `\gamma_{11} + x_3
\kappa_2 - x_2 \kappa_3` of a section without warping, which fixes the signs
of the curvatures, see `Two conventions`_ below. With the material matrix
`\boldsymbol{D}` of each element in the beam axes and `\langle \cdot
\rangle` the integral over the section, the energy per unit length,
`\frac{1}{2} \langle \boldsymbol{\Gamma}^\top \boldsymbol{D}
\boldsymbol{\Gamma} \rangle`, is a quadratic form of `\boldsymbol{V}`,
`\boldsymbol{V}'` and `\bar{\boldsymbol{\epsilon}}` with the matrices

.. math::
    :label: vabs-D

    \boldsymbol{D}_{hh} = \langle (\boldsymbol{\Gamma}_h \boldsymbol{S})^\top
    \boldsymbol{D} \boldsymbol{\Gamma}_h \boldsymbol{S} \rangle, \quad
    \boldsymbol{D}_{h\epsilon} = \langle (\boldsymbol{\Gamma}_h
    \boldsymbol{S})^\top \boldsymbol{D} \boldsymbol{\Gamma}_\epsilon \rangle,
    \quad
    \boldsymbol{D}_{\epsilon\epsilon} = \langle
    \boldsymbol{\Gamma}_\epsilon^\top \boldsymbol{D}
    \boldsymbol{\Gamma}_\epsilon \rangle, \quad
    \boldsymbol{D}_{hl}, \; \boldsymbol{D}_{ll}, \; \boldsymbol{D}_{l\epsilon}

the last three defined in the same way with `\boldsymbol{\Gamma}_l
\boldsymbol{S}`, Eqs. (76) and (89) of Yu et al. (2012). They are
integrated with Gauss rules over 3- and 6-node triangles and 4-, 8- and
9-node quadrilaterals and assembled by the kernel ``_integrate`` of
:mod:`pyfe3d.vabs`.

**Asymptotic ordering.** The variational asymptotic method of Cesnik and
Hodges (1997) orders the terms of the energy by the small parameter `h/l`,
the ratio of the dimension of the section to the length over which the
deformation of the beam varies: the terms in `\boldsymbol{V}'` and in the
derivatives of the strains are one order smaller than those in
`\boldsymbol{V}` and `\bar{\boldsymbol{\epsilon}}`. The leading order gives
the classical model, the next one its refinement to the generalized
Timoshenko model.

**Classical model.** Without the terms in `\boldsymbol{V}'`, Eq. (75) of Yu
et al. (2012),

.. math::
    :label: vabs-U0

    2 \mathcal{U}_0 = \boldsymbol{V}^\top \boldsymbol{D}_{hh} \boldsymbol{V}
    + 2 \boldsymbol{V}^\top \boldsymbol{D}_{h\epsilon}
    \bar{\boldsymbol{\epsilon}} + \bar{\boldsymbol{\epsilon}}^\top
    \boldsymbol{D}_{\epsilon\epsilon} \bar{\boldsymbol{\epsilon}}

`\boldsymbol{D}_{hh}` is singular, its kernel `\boldsymbol{\Psi}` being the
four rigid motions of the section that the beam strains already describe,
and four constraints `\boldsymbol{D}_c^\top \boldsymbol{V} = \boldsymbol{0}`
make the warping unique: zero mean warping and zero mean rotation of the
in-plane warping (``constraint='yu2012'``, Yu et al., 2012, Eqs. (15),
(17)), or zero mean warping and zero mean of `x_2 w_3 - x_3 w_2`
(``'yu2002'``, Yu et al., 2002). The minimum of Eq. :eq:`vabs-U0` is
`\boldsymbol{V}_0 = \hat{\boldsymbol{V}}_0 \bar{\boldsymbol{\epsilon}}`,
from

.. math::
    :label: vabs-V0

    \begin{bmatrix} \boldsymbol{D}_{hh} & \boldsymbol{D}_c \\
    \boldsymbol{D}_c^\top & \boldsymbol{0} \end{bmatrix}
    \begin{Bmatrix} \hat{\boldsymbol{V}}_0 \\ \boldsymbol{\Lambda}
    \end{Bmatrix} = \begin{Bmatrix} -\boldsymbol{D}_{h\epsilon} \\
    \boldsymbol{0} \end{Bmatrix}, \qquad
    \boldsymbol{A} = \hat{\boldsymbol{V}}_0^\top \boldsymbol{D}_{h\epsilon}
    + \boldsymbol{D}_{\epsilon\epsilon}

and `\boldsymbol{A}` is the `4 \times 4` classical stiffness ``Kc``, the
generalized Euler-Bernoulli model. The right-hand side is orthogonal to
`\boldsymbol{\Psi}`, the multipliers `\boldsymbol{\Lambda}` vanish, and
``Kc`` does not depend on the constraints. The warping softens the rigid
section, gives the Saint-Venant torsional stiffness, and couples the four
strains of an anisotropic section.

**Refinement.** With `\boldsymbol{V} = \boldsymbol{V}_0 + \boldsymbol{V}_1`
and the terms in `\boldsymbol{V}'`, the first-order warping of a prismatic
beam is `\boldsymbol{V}_1 = \hat{\boldsymbol{V}}_{1S}
\bar{\boldsymbol{\epsilon}}'`, solved from the same system with
`-\boldsymbol{D}_S` on the right-hand side, `\boldsymbol{D}_S =
(\boldsymbol{D}_{hl} - \boldsymbol{D}_{hl}^\top) \hat{\boldsymbol{V}}_0 -
\boldsymbol{D}_{l\epsilon}`, Eqs. (96)-(101) of Yu et al. (2012), and the
energy to second order is

.. math::
    :label: vabs-U2

    2 \mathcal{U} = \bar{\boldsymbol{\epsilon}}^\top \boldsymbol{A}
    \bar{\boldsymbol{\epsilon}} + 2 \bar{\boldsymbol{\epsilon}}^\top
    \boldsymbol{B} \bar{\boldsymbol{\epsilon}}' +
    \bar{\boldsymbol{\epsilon}}'^\top \boldsymbol{C}
    \bar{\boldsymbol{\epsilon}}'

with `\boldsymbol{B}` and `\boldsymbol{C}` given in the Notes of
:meth:`.CrossSection.calc_stiffness`; the term in
`\bar{\boldsymbol{\epsilon}}''` does not enter the stiffness of a prismatic
beam. The derivatives of the strains are not degrees of freedom of a beam
element. The kinematic relation between the classical and the Timoshenko
strains and the one-dimensional equilibrium equations, Eqs. (27)-(28) of Yu
et al. (2012), turn Eq. :eq:`vabs-U2` into the generalized Timoshenko energy
`\boldsymbol{\epsilon}^\top \boldsymbol{X} \boldsymbol{\epsilon} + 2
\boldsymbol{\epsilon}^\top \boldsymbol{Y} \boldsymbol{\gamma}_s +
\boldsymbol{\gamma}_s^\top \boldsymbol{G} \boldsymbol{\gamma}_s`, with the
classical strains `\boldsymbol{\epsilon}` of the Timoshenko model and the
shear strains `\boldsymbol{\gamma}_s = \{2\gamma_{12}, 2\gamma_{13}\}`, in
closed form, Eqs. (105)-(107):

.. math::
    :label: vabs-timoshenko

    \boldsymbol{G} = \left[ \boldsymbol{Q}^\top \boldsymbol{A}^{-1}
    (\boldsymbol{C} - \boldsymbol{B}^\top \boldsymbol{A}^{-1} \boldsymbol{B})
    \boldsymbol{A}^{-1} \boldsymbol{Q} \right]^{-1}, \qquad
    \boldsymbol{Y} = \boldsymbol{B}^\top \boldsymbol{A}^{-1} \boldsymbol{Q}
    \boldsymbol{G}, \qquad
    \boldsymbol{X} = \boldsymbol{A} + \boldsymbol{Y} \boldsymbol{G}^{-1}
    \boldsymbol{Y}^\top

with `\boldsymbol{Q}` of rows `\{0, 0\}, \{0, 0\}, \{0, -1\}, \{1, 0\}`. The
`6 \times 6` stiffness ``Kt`` is `[\boldsymbol{X}, \boldsymbol{Y};
\boldsymbol{Y}^\top, \boldsymbol{G}]` reordered to the strains
`\{\gamma_{11}, 2\gamma_{12}, 2\gamma_{13}, \kappa_1, \kappa_2, \kappa_3\}`.
Its shear entries contain the shear correction factors and the offset of
the shear centre, which are thus computed, not assumed. By construction, the
classical stiffness recovered from ``Kt`` by a condensation of the
flexibility, deleting the shear rows and columns of ``Ft`` `=
\boldsymbol{K}_t^{-1}` and inverting, is ``Kc``, see
:func:`.classical_from_timoshenko`; deleting the rows and columns of the
stiffness instead would impose `\gamma_{12} = \gamma_{13} = 0`, which gives
another matrix whenever the shear and the classical strains are coupled.

**Centres.** From the flexibilities, the shear centre is `e_2 =
-P_{34}/P_{44}`, `e_3 = P_{24}/P_{44}`, with `P_{ij}` the entries of ``Ft``
numbered from 1, Eq. (66) of Yu et al. (2002), and the tension centre `(a_2,
a_3)` is the point where an axial force gives `\kappa_2 = \kappa_3 = 0`
through ``Fc``.


Steps of an analysis
--------------------

1. **Mesh.** Node coordinates `x_2`, `x_3`, the `y` and `z` of the beam
   elements, and the connectivity of 3- or 6-node triangles and 4-, 8- or
   9-node quadrilaterals, numbered counter-clockwise, see
   :class:`.CrossSection`. The origin of `x_2`, `x_3` is the reference axis
   of the beam: all results refer to it, and it need not be the centroid.
2. **Materials.** One `6 \times 6` material matrix per material, in its
   material axes, e.g. from :func:`.isotropic` or :func:`.orthotropic`, and
   per element the material index and two angles: `\theta_1`, which
   orients the local axes of the element, e.g. along a wall, and the ply
   angle `\theta_3`, see :func:`.rotate_material`.
3. **Classical model.** :meth:`.CrossSection.calc_stiffness` solves the
   warping of the cross section for the four classical strain measures,
   extension, twist and two bending curvatures, and condenses it into the
   `4 \times 4` stiffness ``Kc``. This is the generalized Euler-Bernoulli
   model: the warping softens the rigid section, gives the Saint-Venant
   torsional stiffness, and populates the couplings of anisotropic
   sections.
4. **Generalized Timoshenko model.** The same factorization gives the
   first correction of the warping, driven by the derivatives of the
   classical strains along the beam, and the second-order energy is
   transformed into the `6 \times 6` stiffness ``Kt``, which adds the two
   transverse shear strains. Its shear entries contain the shear
   correction factors and the location of the shear centre, which are not
   assumed. The theory of these two steps is in the notes of
   :meth:`.CrossSection.calc_stiffness`, and the discretization in those
   of the kernel ``_integrate`` of :mod:`pyfe3d.vabs`.
5. **Derived quantities.** The flexibilities ``Fc`` and ``Ft``, the tension
   and shear centres, and, with :meth:`.CrossSection.calc_mass`, the mass
   matrix and the mass integrals of :class:`.BeamProp`.
6. **Use in the beam elements.** :func:`.yu_to_pyfe3d` converts the
   matrices to the convention of pyfe3d, and :func:`.beamprop_from_vabs`
   creates the :class:`.BeamProp` with the full `6 \times 6` matrix, see
   below.
7. **Generalized Vlasov model.** For thin-walled open sections,
   :meth:`.CrossSection.calc_vlasov` gives the `5 \times 5` stiffness
   ``Kv``, which adds the restrained warping, see below.

The analysis is linear and per station: a beam whose section varies along
its length needs one analysis per station.


Axes, angles and centres
------------------------

.. figure:: ../figures/vabs_axes.svg

   (a) Flared channel of constant thickness, made of one isotropic
   material, with the tension and shear centres computed by
   :meth:`.CrossSection.calc_stiffness`, and an element of the upper
   flange with its local axes; (b) that element seen from the tip of
   `y_3`, with its material axes.

The beam axis `x_1` points to the viewer of the cross section and `x_2`,
`x_3` are the `y`, `z` of :class:`.BeamC` and :class:`.BeamLR`, with the
origin at the reference axis. Each element has two angles, see
:func:`.rotate_material`:

- `\theta_1` (``theta1``) rotates the local axes `y_2`, `y_3` of the
  element about `y_1 = x_1`, positive from `x_2` towards `x_3`; for a
  wall, `y_2` is usually taken along the wall and `y_3` across it;
- `\theta_3` (``theta3``) rotates the material axes `m_1`, `m_2` about
  `m_3 = y_3`, positive from `y_1` towards `y_2`; for a laminated wall,
  it is the ply angle measured from the beam axis.

The tension centre `(a_2, a_3)` is the point where an axial force `F_1`
gives no bending curvature, `\kappa_2 = \kappa_3 = 0`, and the shear
centre `(e_2, e_3)` the point where the transverse forces `F_2`, `F_3`
give no twist, `\kappa_1 = 0`. Both are measured from the reference axis.
For the homogeneous channel of the figure, the tension centre is the
centroid and the shear centre lies outside the section, on the side of
the web opposite to the flanges.

This convention of the angles reproduces all the signs of the box beam
CUS computed with VABS 3.2 by Yu et al. (2012). With it, two other
published examples give three couplings each with the published magnitude
and the opposite sign, which is attributed to the labelling of their axes
and ply angles: the box beam CAS1 of Yu et al. (2002) and the pipe of Yu
and Hodges (2005), see the notes of :func:`.rotate_material` and
``tests/test_vabs_composite.py``.


Two conventions
---------------

The matrices of :class:`.CrossSection` are written in the convention of
the VABS literature, which orders the strains as `\{\gamma_{11},
2\gamma_{12}, 2\gamma_{13}, \kappa_1, \kappa_2, \kappa_3\}` and defines
the curvatures as the components of the curvature vector, i.e. the rates
of rotation about `x_1`, `x_2` and `x_3`. The beam elements of pyfe3d use
`\{\bar{\varepsilon}, \bar{\gamma}_y, \bar{\gamma}_z, \kappa_x, \kappa_y,
\kappa_z\}`, with `\kappa_y` and `\kappa_z` named after the coordinate
they multiply in the axial strain, see the notes of :class:`.BeamC`. The
two agree on the extension, the shear strains and the twist, while
`\kappa_2 = \kappa_z` and `\kappa_3 = -\kappa_y`; the moments are `M_2 =
M_y` and `M_3 = -M_z`. Hence, when the reference axis is offset from the
centroid by `d_2` along `x_2`, the coupling of the extension with
`\kappa_3` is `-E A d_2` in ``Kt`` and the coupling with `\kappa_y` is `E
A d_2 = E Q_z` after :func:`.yu_to_pyfe3d`, as in the rigid section of
:class:`.BeamC`; this is checked in ``tests/test_vabs_isotropic.py``.

The ordering of the classical model is the same without the shear
strains, and the classical model is recovered from the Timoshenko one by
a condensation of the flexibility, :func:`.classical_from_timoshenko`,
not by deleting rows and columns of the stiffness.


From the sectional matrix to the beam elements
----------------------------------------------

The sectional stiffness matrix ``C`` of :class:`.BeamProp` has the 21
independent entries of a symmetric `6 \times 6` matrix, in the order of
pyfe3d, i.e. ``C = yu_to_pyfe3d(sec.Kt)``:

.. list-table::
   :header-rows: 1

   * - Index of ``C``
     - 0
     - 1
     - 2
     - 3
     - 4
     - 5
   * - Strain of pyfe3d
     - `\bar{\varepsilon}`
     - `\bar{\gamma}_y`
     - `\bar{\gamma}_z`
     - `\kappa_x`
     - `\kappa_y`
     - `\kappa_z`
   * - Strain of ``Kt``, index
     - `\gamma_{11}`, 0
     - `2\gamma_{12}`, 1
     - `2\gamma_{13}`, 2
     - `\kappa_1`, 3
     - `-\kappa_3`, 5
     - `\kappa_2`, 4

:func:`.beamprop_from_vabs` stores it in ``C``, with the extension-shear,
extension-twist, shear-shear, bending-shear and bending-twist couplings of
a composite section, and fills the 11 parameters of the rigid section,
``A``, ``E``, ``G12``, ``G13``, ``GJ``, ``Ay``, ``Az``, ``Iyy``, ``Izz``
and ``Iyz``, from the entries of ``C`` for information; once ``C`` is set,
the elements read only ``C``, which must be positive semi-definite, see
:class:`.BeamProp`. With these 11 parameters alone, i.e. after
``del prop.C``, the shear centre is the tension centre and the couplings
above are lost, see the notes of :class:`.BeamProp`. For isotropic
sections, the computed ``G12``, ``G13`` and ``GJ`` replace the shear
correction factors and the torsion constant that the user would otherwise
supply.

With couplings, :class:`.BeamLR`, whose linear interpolation does not
depend on ``C``, converges as the square of the element length as for an
uncoupled section. :class:`.BeamC` derives its shape functions from the
equilibrium equations of the beam with the full ``C``, see "Shape
functions with couplings" in the notes of :class:`.BeamC`, and is exact
for nodal loads with any ``C``, which must then be positive definite.
For the box beam CUS of Yu et al. (2012), whose bending-shear couplings
are strong, the tip displacements of a cantilever of length 10 in. under
tip loads or moments are exact to 5.4e-16 with one element of
:class:`.BeamC`, and have errors of 2.3e-4 with 32 elements of
:class:`.BeamLR`, which is exact under the tip moments; the twist due to
an axial load is exact with both elements and any mesh, and the natural
frequencies of BeamC converge to those of BeamLR, see
``tests/test_beam_sectional_stiffness.py``.


Generalized Vlasov model
------------------------

In a thin-walled open section, a twist rate that varies along the beam
produces axial stresses through the warping, with a stiffness that can
exceed the Saint-Venant ``GJ`` by far, which neither the classical nor the
Timoshenko model represents. Following Yu et al. (2005), Section 2,
:meth:`.CrossSection.calc_vlasov` moves the origin of the cross section
to the shear centre with :meth:`.CrossSection.translated`, analyses the
translated section, and keeps, in its second-order energy, only the
derivative `\kappa_1'` of the twist rate among the derivatives of the
classical strains. The result is the `5 \times 5` stiffness ``Kv`` of the
strains `\{\gamma_{11}, \kappa_1, \kappa_2, \kappa_3, \kappa_1'\}`
referred to the shear centre, work-conjugate to `\{F_1, M_1, M_2, M_3,
M_\omega\}`, with the bimoment `M_\omega`; its last diagonal entry is the
warping rigidity, `E\Gamma` for an isotropic thin-walled section::

    sec.calc_vlasov()
    EGamma = sec.Kv[4, 4]
    w1 = sec.vlasov_section.warping([0., 1., 0., 0.])[:, 0]

The first four rows and columns are the classical stiffness about the
shear centre. The model is computed by default with the warping
constraints of Yu et al. (2002), ``constraint='yu2002'``, which reproduce
the VABS results of Yu et al. (2005) and, for a strip, their asymptotic
thin-walled warping rigidity. The constraints of Yu et al. (2012),
``constraint='yu2012'``, the default of
:meth:`.CrossSection.calc_stiffness`, differ in the fourth constraint, on
the rotation of the in-plane warping. They give the same first four rows
and columns of ``Kv`` and, in the tested cases, the same shear centre, but
a different warping rigidity ``Kv[4, 4]``, lower for strips and nearly
equal for I- and channel sections. For the cases of ``tests/test_vabs_vlasov.py``, with
`E = 10^7` and `\nu = 0.3` for the isotropic ones:

.. list-table::
   :header-rows: 1

   * - Case
     - ``'yu2002'``
     - ``'yu2012'``
     - Reference
   * - Isotropic strip, Table 1 of Yu et al. (2005)
     - 1.78634
     - 1.61715
     - VABS 1.78635; `E h^3 b^3/[144(1 - \nu^2)]` = 1.78336;
       `E h^3 b^3/144` = 1.62286
   * - Orthotropic strip, Table 3
     - 3.39252
     - 3.32682
     - VABS 3.39251
   * - Anisotropic strip, 15 degrees, Table 4
     - 1.82501
     - 1.59940
     - VABS 1.82499
   * - Isotropic I-beam, Table 5, mesh of the paper
     - 3518.92
     - 3515.17
     - VABS 3553.43; thin-walled `E t b_f^3 h^2/24` = 3526.67
   * - Channel, `t = 0.01`
     - 912.037
     - 911.853
     - thin-walled `E \Gamma` = 911.458

The torsional warping `w_1` per unit twist rate is the same for both. No
beam element of pyfe3d has a
warping degree-of-freedom: ``Kv`` is an output for one-dimensional
Vlasov beam analyses outside pyfe3d, such as that of Section 3 of Yu et
al. (2005).


Warping of the cross section
----------------------------

After :meth:`.CrossSection.calc_stiffness`, :meth:`.CrossSection.warping`
returns the warping displacements `w_1, w_2, w_3` of the nodes for given
strains of the beam, and :meth:`.CrossSection.triangles` a triangulation of
the mesh through all its nodes, from which the warping can be drawn with
any plotting library; pyfe3d depends on none. The warping satisfies
the constraints chosen in :meth:`.CrossSection.calc_stiffness`, among them
zero average, and it is the warping relative to the rigid motion of the
section about the reference axis. In the tested cases, the two sets of
constraints, ``'yu2012'`` and ``'yu2002'``, give the same out-of-plane
warping per unit classical strain, but they can differ in the in-plane
warping and in the
term of the derivatives of the strains; :meth:`.CrossSection.translated`
does not depend on them.

The torsional warping is therefore shown, as in the literature, about the
shear centre: the section translated to the shear centre with
:meth:`.CrossSection.translated`, as in :meth:`.CrossSection.calc_vlasov`,
gives the warping relative to the rotation about the shear centre, with
zero average. For a thin-walled open section it is the principal
sectorial coordinate `\omega` of Vlasov (1961, Eqs. (3.15) and (7.1), pp.
14-16 and 40-41), `\omega = \int \rho \, ds`, `\rho` being the distance from the pole to the tangent of the
mid-line, normalised with the pole at the shear centre such that
`\omega` has zero average and is orthogonal to `x_2` and `x_3`: with
`\rho` positive when the radius from the pole turns from `x_2` towards
`x_3`, the absence of shear strain on the mid-line gives `w_1 = -\omega
\kappa_1`. This is checked for a channel in
``tests/test_vabs_vlasov.py``.

The figure shows, for a thin-walled channel, the out-of-plane warping
`w_1` per unit twist rate `\kappa_1` about the shear centre (cross), with
the tension centre (circle), and the in-plane warping per unit curvature
`\kappa_2`, i.e. the anticlastic deformation due to the Poisson effect,
drawn to scale on the undeformed mesh: the flange in tension, `x_3 > 0`,
contracts, and the flange in compression expands. The torsional warping
is linear along each wall, `-\omega`, and changes sign on the web, at the
centre of the section, and along the flanges.

.. plot:: _scripts/vabs_warping_2d.py
   :include-source: false

The three-dimensional view below shows the same channel as a beam of
length 3 under a uniform twist rate:
the sections rotate about the shear centre and warp along the beam by
`w_1 = -\omega \kappa_1`, the warping being exaggerated and coloured by
`w_1`. The figure can be rotated, panned and zoomed:

.. raw:: html
   :file: _generated/vabs_warping_3d.html


Validation
----------

The tests compare the results with elasticity solutions and with
published VABS results:

- ``tests/test_vabs_isotropic.py``: rectangles, circles and tubes, with the
  axial and bending stiffnesses, the Saint-Venant torsion constant and the
  shear correction factors of the elasticity solutions, Eq. (25) of Yu and
  Hodges (2005) for the rectangle; the isotropic square of Yu et al.
  (2002), the T-section of Yu and Hodges (2005), and the couplings of an
  offset reference axis;
- ``tests/test_vabs_composite.py``: the box beam CUS of Yu et al. (2012),
  reproduced to the four digits of their VABS 3.2 results, the
  anisotropic pipe of Yu and Hodges (2005), with its mass properties, and
  the box beam CAS1 of Yu et al. (2002), with the opposite signs of three
  couplings of the last two examples asserted explicitly, see above;
- ``tests/test_vabs_elements.py``: integration rules, shape functions,
  material rotation and conventions.
- ``tests/test_beam_sectional_stiffness.py``: cantilevers of
  :class:`.BeamC` and :class:`.BeamLR` with a coupled sectional stiffness
  matrix, among them the box beam CUS of Yu et al. (2012), against the
  closed form obtained by integrating the strains `\boldsymbol{C}^{-1}
  \boldsymbol{F}(x)`, and the tangent stiffness matrix against the
  derivative of the internal forces.
- ``tests/test_vabs_warping.py``: the torsional warping of an ellipse
  against the Saint-Venant solution, and the triangulation of every
  element type;
- ``tests/test_vabs_vlasov.py``: the generalized Vlasov model against the
  VABS results of Yu et al. (2005), Tables 1, 3 and 4, isotropic,
  orthotropic and anisotropic strips, reproduced to five digits or more,
  and Table 5, isotropic I-beam, within 1 per cent for the torsional
  rigidity and, for the warping rigidity, about 1 per cent below, a
  converged difference, within 2 per cent, the grade A of MacNeal and
  Harder (1985); the anisotropic
  I-beam of Table 6, with the lay-up reconstructed from Volovoi et al.
  (1999), p. 1038 and Fig. 8, within 1.1 per cent for the effective
  torsional rigidity and 4 per cent for the warping rigidity; the
  composite I-beam of Hodges and Peters (2001), p. 1601, against their
  asymptotic thin-walled constants, with tolerances built from the
  neglected terms of the thin-walled theory, the free-edge factor `1 -
  0.630 t/b` of Timoshenko and Goodier (1951), Art. 95, Eq. (161), of
  the flanges for the torsional rigidity, 3.45 and 2.90 per cent at 0
  and 45 degrees, and `(t/b)^2` for the warping rigidity, 1.31 per cent,
  plus the rounding of the printed values and data; with the walls made
  thinner, the differences from the thin-walled values decrease as `t/b`
  for the torsional and as `(t/b)^2` for the warping rigidity; and the
  torsional warping
  about the shear centre of a thin-walled channel against `-\omega`,
  with its warping rigidity, all with the constraints ``'yu2002'`` and
  ``'yu2012'``.


Limitations
-----------

- Prismatic beams only: initially twisted and curved beams, whose theory
  is in Yu et al. (2002, 2012), are not implemented, and non-zero ``k1``,
  ``k2``, ``k3`` of :class:`.CrossSection` raise ``NotImplementedError``.
- No applied loads in the sectional analysis, Yu et al. (2012), Section
  5.3, and no recovery of the three-dimensional strains and stresses,
  although the warping is available, see :meth:`.CrossSection.warping`.
- The generalized Vlasov model is not used by any beam element of
  pyfe3d.


References
----------

Cesnik, C. E. S., and Hodges, D. H., "VABS: A New Concept for Composite
Rotor Blade Cross-Sectional Modeling", Journal of the American Helicopter
Society, 42(1), pp. 27-38, 1997. https://doi.org/10.4050/JAHS.42.27

Yu, W., Hodges, D. H., Volovoi, V. V., and Cesnik, C. E. S., "On
Timoshenko-like modeling of initially curved and twisted composite beams",
International Journal of Solids and Structures, 39(19), pp. 5101-5121,
2002. https://doi.org/10.1016/S0020-7683(02)00399-2

Yu, W., and Hodges, D. H., "Generalized Timoshenko Theory of the
Variational Asymptotic Beam Sectional Analysis", Journal of the American
Helicopter Society, 50(1), pp. 46-55, 2005.
https://doi.org/10.4050/1.3092842

Yu, W., Hodges, D. H., Volovoi, V. V., and Fuchs, E. D., "A generalized
Vlasov theory for composite beams", Thin-Walled Structures, 43(9), pp.
1493-1511, 2005. https://doi.org/10.1016/j.tws.2005.02.003

Vlasov, V. Z., Thin-Walled Elastic Beams, 2nd ed., Israel Program for
Scientific Translations, Jerusalem, 1961. Translated by Y. Schechtman.

Volovoi, V. V., Hodges, D. H., Berdichevsky, V. L., and Sutyrin, V. G.,
"Asymptotic theory for static behavior of elastic anisotropic I-beams",
International Journal of Solids and Structures, 36(7), pp. 1017-1043,
1999. https://doi.org/10.1016/S0020-7683(97)00341-7

Hodges, D. H., and Peters, D. A., "Lateral-torsional buckling of
cantilevered elastically coupled composite strip- and I-beams",
International Journal of Solids and Structures, 38(9), pp. 1585-1603,
2001. https://doi.org/10.1016/S0020-7683(00)00111-6

Timoshenko, S. P., and Goodier, J. N., Theory of Elasticity, 2nd ed.,
McGraw-Hill, New York, 1951.

MacNeal, R. H., and Harder, R. L., "A proposed standard set of problems
to test finite element accuracy", Finite Elements in Analysis and Design,
1(1), pp. 3-20, 1985. https://doi.org/10.1016/0168-874X(85)90003-4

Yu, W., Hodges, D. H., and Ho, J. C., "Variational asymptotic beam
sectional analysis -- An updated version", International Journal of
Engineering Science, 59, pp. 40-64, 2012.
https://doi.org/10.1016/j.ijengsci.2012.03.006
