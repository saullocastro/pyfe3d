Beam theory and the beam elements
=================================

A beam model replaces the three-dimensional analysis of a slender member by a
one-dimensional problem along a reference axis, in which the cross section
enters only through a few sectional properties. The reduction holds when the
dimensions of the cross section are small compared with the length of the
member and with the length over which the deformation varies along it; it
does not hold near clamped ends, concentrated loads and abrupt changes of
section, nor when the walls of the cross section deform or buckle locally,
where a shell model is needed.

pyfe3d has two Timoshenko beam elements, :class:`.BeamC` and :class:`.BeamLR`,
which share the kinematics, the sectional stiffness matrix of a
:class:`.BeamProp` and the degrees of freedom, and differ in the interpolation
along the element. This chapter derives the beam model they implement, the
two discretizations, their mass, geometric and non-linear stiffness matrices,
and closes with the choice between them. The cross-section properties are
detailed in :doc:`beam_properties`, the truss and the spring in
:doc:`truss_spring`, the computation of the sectional stiffness matrix of an
arbitrary cross section in :doc:`cross_sectional_analysis`, and the large
rotations of beams in :doc:`corotational_formulation`.


Element coordinates and degrees of freedom
------------------------------------------

The element coordinate system has its `x` axis from node 1 to node 2, the
beam axis, and its `y` axis in the plane of `x` and of the vector ``vxyi,
vxyj, vxyk`` given to ``update_rotation_matrix``; the cross section lies in
the `y z` plane, see :doc:`conventions`. Each of the two nodes has the six
degrees of freedom of every node of pyfe3d, and the element displacements, in
element coordinates, are ordered as

.. math::
    :label: beam-ue

    \boldsymbol{u}_e = \{u_1, v_1, w_1, {r_x}_1, {r_y}_1, {r_z}_1,
    u_2, v_2, w_2, {r_x}_2, {r_y}_2, {r_z}_2\}^\top

the attribute ``ue`` of :class:`.BeamCProbe` and :class:`.BeamLRProbe`. The
element length `L` is the attribute ``length``, and `\xi = x/L` is the
coordinate along the element.


Kinematics of the rigid cross section
-------------------------------------

The cross section translates and rotates as a rigid body. With `u, v, w` the
translations and `r_x, r_y, r_z` the rotations of the beam axis, right-handed
about the element axes, the displacements of the point `(y, z)` of the cross
section are

.. math::
    :label: beam-kinematics

    u_x = u - y r_z + z r_y, \qquad u_y = v - z r_x, \qquad u_z = w + y r_x

Linear strains of the three-dimensional body, with the in-plane strains of
the cross section zero by assumption, give

.. math::
    :label: beam-strains-point

    \epsilon_{xx} = u_{,x} + y \kappa_y + z \kappa_z, \qquad
    \gamma_{xy} = \gamma_{xy}^0 - z \kappa_x, \qquad
    \gamma_{xz} = \gamma_{xz}^0 + y \kappa_x

with the six generalized strains of the beam axis

.. math::
    :label: beam-generalized-strains

    \boldsymbol{\epsilon} =
    \begin{Bmatrix} \bar{\varepsilon} \\ \bar{\gamma}_y \\ \bar{\gamma}_z \\
    \kappa_x \\ \kappa_y \\ \kappa_z \end{Bmatrix} =
    \begin{Bmatrix} u_{,x} \\ v_{,x} - r_z \\ w_{,x} + r_y \\ {r_x}_{,x} \\
    -{r_z}_{,x} \\ {r_y}_{,x} \end{Bmatrix}

the axial strain, the transverse shear strains `\gamma_{xy}^0 =
\bar{\gamma}_y` and `\gamma_{xz}^0 = \bar{\gamma}_z` of the planes `x y` and
`x z`, the twist rate and the two bending curvatures. The curvatures are
named after the coordinate they multiply in the axial strain, as in Luo
(2008), not after the axis of the rotation they come from: `\kappa_y` is the
curvature of the bending in the `x y` plane, governed by `v` and `r_z`, and
`\kappa_z` that of the bending in the `x z` plane, governed by `w` and `r_y`.
A positive `\kappa_y` stretches the fibres at `y > 0`. The torsion is that of
Saint-Venant, uniform and without restrained warping; the warping of the cross
section enters only through the torsional stiffness, see
:doc:`beam_properties`.


Sectional stiffness matrix and strain energy
--------------------------------------------

The strain energy of the element is the integral along the axis of a
quadratic form of the generalized strains,

.. math::
    :label: beam-energy

    U = \frac{1}{2} \int_0^L \boldsymbol{\epsilon}^\top \boldsymbol{C}
    \boldsymbol{\epsilon} \, dx, \qquad
    \boldsymbol{F} = \boldsymbol{C} \boldsymbol{\epsilon} =
    \{N, V_y, V_z, M_x, M_z, M_y\}^\top

where `\boldsymbol{C}` is the `6 \times 6` sectional stiffness matrix, the
attribute ``C`` of :class:`.BeamProp`, symmetric with 21 independent entries
`C_{ij}`, `i, j = 0, \ldots, 5`, in the order of
`\boldsymbol{\epsilon}`, and `\boldsymbol{F}` the work-conjugate sectional
forces: the axial force, the two shear forces, the twisting moment and the
bending moments `M_z = \int \sigma_{xx} y \, dA` and `M_y = \int \sigma_{xx}
z \, dA`, both positive when they stretch the fibres at positive `y` or `z`.
Integrating the strain energy density of the rigid cross section, with
Eq. :eq:`beam-strains-point`, gives the matrix of the 11 cross-section
attributes of :class:`.BeamProp` (see :doc:`beam_properties`):

.. math::
    :label: beam-C-rigid

    \boldsymbol{C} = \begin{bmatrix}
        E A & 0 & 0 & 0 & E A_y & E A_z \\
        0 & G_{12} A & 0 & -G_{12} A_z & 0 & 0 \\
        0 & 0 & G_{13} A & G_{13} A_y & 0 & 0 \\
        0 & -G_{12} A_z & G_{13} A_y & G J & 0 & 0 \\
        E A_y & 0 & 0 & 0 & E I_{zz} & E I_{yz} \\
        E A_z & 0 & 0 & 0 & E I_{yz} & E I_{yy}
    \end{bmatrix}

with `A_y = \int y \, dA` and `A_z = \int z \, dA` zero when the axis passes
through the centroid, and `I_{yz} = 0` for principal axes. The rigid section
gives `G_{12} I_{yy} + G_{13} I_{zz}` for the torsional entry, which is exact
only for circular sections; it is replaced by the torsional stiffness `G J`
(``GJ``) given by the user. Equation :eq:`beam-C-rigid` has 11 parameters:
the extension-shear, extension-twist, shear-shear, bending-shear and
bending-twist couplings of an anisotropic or non-homogeneous section are
given by setting ``C`` itself, e.g. with :func:`.beamprop_from_vabs`.

The elements apply no shear correction. In the rigid section the transverse
shear stress is uniform, and the shear stiffnesses are `G A`, against the
`k_y G A` and `k_z G A` of the actual, non-uniform, distributions, with the
shear correction factors `k_y`, `k_z`, e.g. `5/6` for a rectangle. The user
applies them to the moduli, `G_{12} = k_y G` and `G_{13} = k_z G`, which
leaves ``GJ`` unchanged; for an arbitrary section the cross-sectional
analysis of :mod:`pyfe3d.vabs` gives the shear entries, the torsional
stiffness and the couplings without assumed factors.


Equilibrium of an element loaded at its nodes
---------------------------------------------

Stationarity of the total potential energy with Eq. :eq:`beam-energy`, for a
prismatic element without distributed loads, gives the equilibrium equations
of the sectional forces,

.. math::
    :label: beam-equilibrium

    N_{,x} = {V_y}_{,x} = {V_z}_{,x} = {M_x}_{,x} = 0, \qquad
    {M_z}_{,x} = V_y, \qquad {M_y}_{,x} = V_z

i.e. constant axial force, shear forces and twisting moment, and linear
bending moments. For a diagonal `\boldsymbol{C}`, the bending in the `x y`
plane reduces to `(C_{11} \bar{\gamma}_y)_{,x} = 0` and `C_{11}
\bar{\gamma}_y = C_{44} {\kappa_y}_{,x}`, i.e. `G_{12} A \bar{\gamma}_y = E
I_{zz} {\kappa_y}_{,x}`, and likewise for the `x z` plane with `C_{22}` and
`C_{55}`. An interpolation that solves these equations for any nodal values
makes one element exact for a prismatic member loaded at its nodes; this is
the principle of :class:`.BeamC`.

The ratio of the shear to the bending flexibility of an element of length
`L` is measured by

.. math::
    :label: beam-alpha

    \alpha_y = -\frac{12 C_{44}}{C_{11} L^2}, \qquad
    \alpha_z = -\frac{12 C_{55}}{C_{22} L^2}

negative, i.e. `-12 E I_{zz}/(G_{12} A L^2)` and `-12 E I_{yy}/(G_{13} A
L^2)` for Eq. :eq:`beam-C-rigid`. With the radii of gyration `i_z =
\sqrt{I_{zz}/A}` and `i_y = \sqrt{I_{yy}/A}`, `|\alpha_y| = 12 (E/G_{12})
(i_z/L)^2`: the shear deformation decays with the square of the slenderness,
and the Euler-Bernoulli beam, `\bar{\gamma}_y = \bar{\gamma}_z = 0`, is its
limit `\alpha_y = \alpha_z = 0`. For a rectangle of height `h` along `z` and
an isotropic material, `|\alpha_z| = 2 (1 + \nu) (h/L)^2`, about 0.026 for `L/h
= 10` and `\nu = 0.3`. The shear deformation matters for short members,
sandwich and composite sections with low transverse shear moduli, and the
higher vibration modes, whose half-wave length takes the place of `L`.


BeamC: consistent shape functions
---------------------------------

**Uncoupled section.** When the only coupling of `\boldsymbol{C}` is the
extension-twist entry `C_{03}`, e.g. without ``C`` and with ``Ay``, ``Az`` and
``Iyz`` zero, :class:`.BeamC` uses the consistent shape functions of Luo
(2008): `u` and `r_x` linear, and, with `\beta_y = 1/(1 - \alpha_y)`,

.. math::
    :label: beamc-v

    v = \left[1 + \beta_y (2\xi^3 - 3\xi^2 + \alpha_y \xi)\right] v_1
        + L \beta_y \left(\xi^3 + \frac{\alpha_y - 4}{2} \xi^2
        + \left(1 - \frac{\alpha_y}{2}\right) \xi \right) {r_z}_1
        + \beta_y (-2\xi^3 + 3\xi^2 - \alpha_y \xi) v_2
        + L \beta_y \left(\xi^3 - \frac{\alpha_y + 2}{2} \xi^2
        + \frac{\alpha_y}{2} \xi \right) {r_z}_2

.. math::
    :label: beamc-rz

    r_z = \frac{6 \beta_y}{L} (\xi^2 - \xi) (v_1 - v_2)
        + \left[1 + \beta_y (3\xi^2 + (\alpha_y - 4) \xi)\right] {r_z}_1
        + \beta_y (3\xi^2 - (\alpha_y + 2) \xi) {r_z}_2

and the same functions for `w` and `r_y`, with `v` replaced by `w`, `r_z` by
`-r_y`, `\alpha_y` by `\alpha_z` and `\beta_y` by `\beta_z = 1/(1 -
\alpha_z)`. In the code they are ``alphay``, ``alphaz``, ``betay`` and
``betaz``. Cubic `v` and quadratic `r_z` share the same four nodal values,
which is what makes the interpolation consistent: the shear strain is
constant along the element,

.. math::
    :label: beamc-shear

    \bar{\gamma}_y = \alpha_y \beta_y \left( \frac{v_1 - v_2}{L} +
    \frac{{r_z}_1 + {r_z}_2}{2} \right)

and vanishes for `\alpha_y = 0`, where Eq. :eq:`beamc-v` becomes the cubic
Hermite interpolation of the Euler-Bernoulli beam and `r_z = v_{,x}`. The
element is therefore free of shear locking with exact integration. These
functions solve Eq. :eq:`beam-equilibrium` with a diagonal `\boldsymbol{C}`
only with the negative sign of Eq. :eq:`beam-alpha`; the same solution is
Eq. (17) of Luo (2008) and Eq. (3.236) of Petyt (2010), whereas Eq. (21) of
Luo (2008) prints `\alpha_y`, `\alpha_z` with the positive sign, and his Eq.
(20) the functions of `w`, `r_y` with the signs of those of `v`, `r_z`, see
the Notes of :class:`.BeamC`. One element gives the tip deflection `P L^3/(3
E I_{zz}) + P L/(G_{12} A)` of a cantilever under a tip force `P`, Eq. (1.2)
of Bathe (2014), and the stiffness matrix ``KC0`` of :meth:`.BeamC.update_KC0`
is the exact integral of Eq. :eq:`beam-energy` with these functions,
evaluated in closed form.

**Coupled section.** With any other coupling, including those of ``Ay``,
``Az`` and ``Iyz``, Eqs. :eq:`beamc-v` and :eq:`beamc-rz` no longer solve the
equilibrium equations, and :class:`.BeamC` derives its shape functions from
Eq. :eq:`beam-equilibrium` with the full `\boldsymbol{C}`, which must then be
positive definite. Integrating the equilibrium equations from node 1, the
sectional forces are

.. math::
    :label: beamc-q

    \boldsymbol{C} \boldsymbol{\epsilon} = (\boldsymbol{I} + x
    \boldsymbol{Z}) \boldsymbol{q}, \qquad
    \boldsymbol{\epsilon} = \boldsymbol{S} (\boldsymbol{I} + x
    \boldsymbol{Z}) \boldsymbol{q}, \qquad
    \boldsymbol{q} = \{N, V_y, V_z, M_x, M_z(0), M_y(0)\}^\top

with `\boldsymbol{S} = \boldsymbol{C}^{-1}` and the only non-zero entries of
`\boldsymbol{Z}` (indices from 0) `Z_{41} = Z_{52} = 1`. Integrating the
kinematic relations of Eq. :eq:`beam-generalized-strains` from node 1,

.. math::
    :label: beamc-integration

    u = \int \epsilon_0 \, dx, \quad r_x = \int \epsilon_3 \, dx, \quad
    r_z = -\int \epsilon_4 \, dx, \quad r_y = \int \epsilon_5 \, dx, \quad
    v = \int (\epsilon_1 + r_z) \, dx, \quad w = \int (\epsilon_2 - r_y) \, dx

and adding the rigid-body motion `\boldsymbol{R}(x)` of node 1, the field
`\boldsymbol{d} = \{u, v, w, r_x, r_y, r_z\}^\top` is `\boldsymbol{d} =
\boldsymbol{R}(x) \boldsymbol{u}_1 + \boldsymbol{P}(x) \boldsymbol{q}`,
quadratic in `x` for `u` and the rotations and cubic for `v` and `w`, with
coefficients that depend on all the entries of `\boldsymbol{S}`. The statics
of the element clamped at node 1 give `\boldsymbol{q} = \boldsymbol{T}
\boldsymbol{P}_2` for the loads `\boldsymbol{P}_2` at node 2, and
`\boldsymbol{F}_c = \boldsymbol{P}(L) \boldsymbol{T}` is the flexibility of
this cantilever. With `\boldsymbol{G} = [-\boldsymbol{R}(L),
\boldsymbol{I}]`, which takes the rigid-body motion of node 1 out of the
displacements of node 2,

.. math::
    :label: beamc-coupled

    \boldsymbol{q} = \boldsymbol{H}_f \boldsymbol{u}_e, \qquad
    \boldsymbol{H}_f = \boldsymbol{T} \boldsymbol{F}_c^{-1} \boldsymbol{G},
    \qquad
    \boldsymbol{d} = \left( [\boldsymbol{R}(x), \boldsymbol{0}] +
    \boldsymbol{P}(x) \boldsymbol{H}_f \right) \boldsymbol{u}_e, \qquad
    \boldsymbol{K}_{C0_e} = \boldsymbol{G}^\top \boldsymbol{F}_c^{-1}
    \boldsymbol{G}

`\boldsymbol{K}_{C0_e}` is the Hessian of Eq. :eq:`beam-energy` with these
shape functions, and the mass, geometric and non-linear matrices are
integrated analytically with the same polynomials. Without couplings other
than `C_{03}` they reduce to Eqs. :eq:`beamc-v` and :eq:`beamc-rz`, whose
closed-form expressions are then used. Otherwise `\boldsymbol{S}`,
`\boldsymbol{F}_c^{-1}`, `\boldsymbol{H}_f` and the coefficients of the shape
functions are evaluated numerically for each element and call, in the C
kernel ``_coupled_fields`` of :mod:`pyfe3d.beamc`, at a cost of a few
microseconds per element. The derivation is that of pyfe3d, given step by
step in the Notes of :class:`.BeamC`.

One element of :class:`.BeamC` is thus exact under nodal loads for any
positive definite `\boldsymbol{C}`, and the axial force `N = q_0` is uniform
along the element. The functions of Luo (2008) evaluated with the diagonal of
a coupled `\boldsymbol{C}` would not be: for single couplings of a rectangular
section, and for the bending-shear couplings of a composite box beam, their
tip displacements err by up to 8.9e-2 and 0.42, see the Notes of
:class:`.BeamC` and ``tests/test_beam_sectional_stiffness.py``.


BeamLR: linear interpolation and reduced integration
----------------------------------------------------

:class:`.BeamLR` interpolates the six displacements of the axis linearly,
`d = (1 - \xi) d_1 + \xi d_2`. The axial strain, the twist and the
curvatures are then constant in each element, and the shear strains
`\bar{\gamma}_y = v_{,x} - r_z` and `\bar{\gamma}_z = w_{,x} + r_y` linear.
With exact integration, the shear terms of the linear interpolation would
lock in slender elements: a constant curvature, i.e. a linear `r_z`, cannot be
combined with a linear `v` into a shear strain that vanishes along the
element. :class:`.BeamLR` evaluates the strain energy at the centre of the element,

.. math::
    :label: beamlr-KC0

    \boldsymbol{K}_{C0_e} = L \boldsymbol{B}_L^\top \boldsymbol{C}
    \boldsymbol{B}_L, \qquad \boldsymbol{\epsilon} = \boldsymbol{B}_L
    \boldsymbol{u}_e \text{ at } \xi = 1/2

which is exact for the constant strains and imposes the shear strains only at
the centre, see :meth:`.BeamLR.update_KC0`. The interpolation does not
depend on `\boldsymbol{C}`, so any `\boldsymbol{C}`, positive semi-definite,
is used as it is, and every coupling is represented with the same
convergence as an uncoupled section. The price is a constant curvature in
each element: the error decreases as the square of the element length, e.g.
8.3e-2, 5.2e-3 and 3.3e-4 with 1, 4 and 16 elements for the coupling
`C_{15}` under a transverse tip load, against one element of
:class:`.BeamC`, see the Notes of :class:`.BeamC`. Since :class:`.BeamLR`
does not use `\alpha_y`, `\alpha_z`, it is an independent check of their
sign: refined meshes converge to the same tip deflections and frequencies as
:class:`.BeamC`, ``tests/test_timoshenko_beam_literature.py``.


Mass matrices
-------------

The kinetic energy of the displacement field of Eq. :eq:`beam-kinematics`,
integrated over the cross section with the mass integrals of
:class:`.BeamProp`, is

.. math::
    :label: beam-kinetic

    T = \frac{1}{2} \int_0^L \Big[ m (\dot{u}^2 + \dot{v}^2 + \dot{w}^2)
        + m_{yy} \dot{r}_z^2 + m_{zz} \dot{r}_y^2
        - 2 m_{yz} \dot{r}_y \dot{r}_z
        - 2 m_y \dot{u} \dot{r}_z + 2 m_z \dot{u} \dot{r}_y
        - 2 m_z \dot{v} \dot{r}_x + 2 m_y \dot{w} \dot{r}_x
        + (m_{yy} + m_{zz}) \dot{r}_x^2 \Big] dx

with `m` = ``intrho``, `m_y` = ``intrhoy``, `m_z` = ``intrhoz``, `m_{yy}` =
``intrhoy2``, `m_{zz}` = ``intrhoz2`` and `m_{yz}` = ``intrhoyz``. The
consistent mass matrix, ``mtype=0`` of :meth:`.BeamC.update_M` and
:meth:`.BeamLR.update_M`, is the Hessian of `T` with respect to the nodal
velocities, with the shape functions of each element, integrated exactly and
with the rotary inertia: for :class:`.BeamC` it is the matrix of Petyt
(2010), Sections 3.3, 3.4 and 3.9, in each plane of bending, and with
couplings it uses the shape functions of Eq. :eq:`beamc-coupled`; for
:class:`.BeamLR`, e.g. `L m/6` times `[2, 1; 1, 2]` for each translation.
The lumped mass matrix, ``mtype=1``, is the two-point Gauss-Lobatto
quadrature of `T`, whose points are the nodes, after Brockman (1987):
diagonal, with `L m/2` for each translation, `L (m_{yy} + m_{zz})/2` for
`r_x`, `L m_{zz}/2` for `r_y` and `L m_{yy}/2` for `r_z` at each node, and
without the first moments `m_y`, `m_z` and the product `m_{yz}`, which would
make it non-diagonal.


Geometric stiffness and geometrically non-linear analysis
---------------------------------------------------------

The geometric non-linearity of both elements is that of the von Kármán axial
strain of the beam axis,

.. math::
    :label: beam-vonkarman

    \epsilon_{xx} = u_{,x} + \frac{1}{2} \left( v_{,x}^2 + w_{,x}^2 \right)
    + y \kappa_y + z \kappa_z

valid for small strains and moderate rotations in the fixed element axes; the
curvatures and the shear strains stay linear. The internal forces of
``update_fint(..., nonlinear=1)`` are the gradient of the strain energy with
Eq. :eq:`beam-vonkarman`, and ``KC0 + KCNL + KG`` is its exact, symmetric
Jacobian: ``KG`` is the geometric stiffness of the axial force of the linear
strains, homogeneous of degree one in the displacements, and ``KCNL`` holds
the constitutive terms of the non-linear part of the strain together with the
geometric term of its axial force, see :meth:`.BeamC.update_KCNL`. The
splitting keeps ``KG`` suitable for the linear buckling problem
`(\boldsymbol{K}_{C0} + \lambda \boldsymbol{K}_G) \boldsymbol{u} =
\boldsymbol{0}`, see :doc:`eigenvalue_analyses`.

**BeamC: element mean of the non-linear strain.** With a linear `u` and cubic
`v`, `w`, the pointwise `(v_{,x}^2 + w_{,x}^2)/2` is of degree four in `x`,
which the constant `u_{,x}` cannot balance: the axial strain cannot vanish in
inextensional bending and coarse meshes are too stiff, the membrane locking
of Crisfield (1991), Section 7.1.3. :class:`.BeamC` replaces the non-linear
part by its element mean,

.. math::
    :label: beamc-mean

    \epsilon_{xx} = u_{,x} + \bar{\epsilon}_{NL} + y \kappa_y + z \kappa_z,
    \qquad
    \bar{\epsilon}_{NL} = \frac{1}{L} \int_0^L \frac{1}{2} \left( v_{,x}^2 +
    w_{,x}^2 \right) dx = \frac{1}{2} \boldsymbol{u}_e^\top \boldsymbol{H}_m
    \boldsymbol{u}_e

Crisfield (1991), Eqs. (7.31)-(7.32), and (1997), Eqs. (17.56a)-(17.56b), with
`\boldsymbol{H}_m` the mean of the products of the slope rows of the
consistent shape functions, which include the shear deformation. The
geometric stiffness carries the element mean `N_m` of the axial force,

.. math::
    :label: beamc-KG

    \boldsymbol{K}_G = L N_m \boldsymbol{H}_m

and, since the shape functions solve the equilibrium equations, `N` is
uniform and `N_m = N` for any `\boldsymbol{C}`; e.g. `N_m = E A (u_2 -
u_1)/L` for a centroidal uncoupled section, see :meth:`.BeamC.update_KG`.

**BeamLR: slopes of the linear interpolation.** :class:`.BeamLR` takes the
slopes `v_{,x}` and `w_{,x}` from the linear interpolations of `v` and `w`,
constant in each element, so that its geometric stiffness matrix is that of a
string, `N/L` times `[1, -1; -1, 1]` for `v` and for `w` of the two nodes,
see :meth:`.BeamLR.update_KG`. The axial strain can vanish in each element,
and the element is free of membrane locking without averaging. Because the
slopes, not the rotations, enter the axial strain, the buckling load includes
the shear deformation and converges, as the square of the element length, to
`P_e/(1 + P_e/(k G A))`, Eq. (2-57) of Timoshenko and Gere (1961), instead of
the Euler load `P_e`, see the Notes of :class:`.BeamLR`; :class:`.BeamC`,
whose slopes include the shear strains of Eq. :eq:`beamc-shear`, reproduces
Eq. (2-57) to 4.2e-7 with 20 elements for a clamped-free column,
``tests/test_beam_linear_buckling_3D_column.py``.


Transformation to global coordinates
------------------------------------

All the matrices and internal forces are computed in element coordinates and
transformed with the block-diagonal matrix `\boldsymbol{T}` of four copies of
`\boldsymbol{T}_{3\times 3}`, whose columns are the element axes,
`\boldsymbol{K} = \boldsymbol{T} \boldsymbol{K}_e \boldsymbol{T}^\top` and
`\boldsymbol{F} = \boldsymbol{T} \boldsymbol{F}_e`, see :doc:`conventions`.
The element axes are not updated with the displacements: for large rotations
the elements are wrapped by the co-rotational layer, see
:doc:`corotational_formulation`.


Limits of validity
------------------

- Rigid cross section: no in-plane deformation of the section, no shear lag,
  no local buckling of the walls; the warping enters only through ``GJ`` or
  through a ``C`` computed by :mod:`pyfe3d.vabs`.
- Saint-Venant torsion: no warping degree of freedom, hence no restrained
  warping, whose stiffness can dominate the torsion of thin-walled open
  sections; the generalized Vlasov stiffness of
  :meth:`.CrossSection.calc_vlasov` is not used by the elements.
- Prismatic elements: one constant ``C`` per element; a tapered or curved
  member is modelled by several straight elements.
- Linear elasticity and small strains; with ``nonlinear=1``, moderate
  rotations in the fixed element axes. The shear correction factors are the
  user's choice unless ``C`` comes from a cross-sectional analysis.


Choosing between BeamC and BeamLR
---------------------------------

.. list-table::
    :header-rows: 1

    * - Aspect
      - :class:`.BeamC`
      - :class:`.BeamLR`
    * - Interpolation
      - solves the equilibrium equations of the element for its ``C``
      - linear, one-point integration of the strain energy
    * - Nodal loads, prismatic member
      - exact with one element per member, for any ``C``
      - converges as the square of the element length
    * - Coupled ``C`` (e.g. from :mod:`pyfe3d.vabs`)
      - positive definite required; exact, a few microseconds more per
        element and call
      - positive semi-definite suffices; same convergence as uncoupled
    * - Zero stiffnesses in ``C``, e.g. ``GJ = 0`` or ``Iyy = 0`` in a
        planar model
      - allowed when the only coupling is `C_{03}`
      - allowed
    * - Membrane locking (``nonlinear=1``)
      - removed by the element mean of the non-linear axial strain
      - absent, constant slopes
    * - Linear buckling with shear deformation, Eq. (2-57) of Timoshenko
        and Gere (1961)
      - 4.2e-7 with 20 elements for a clamped-free column
      - 1.0e-3 with 20 elements, converging as the square of the element
        length

:class:`.BeamC` is the recommended beam element: it needs the fewest
elements, and with a coupled sectional stiffness matrix it remains exact
under nodal loads. :class:`.BeamLR` is the choice when ``C`` is only positive
semi-definite with couplings, as an independent check of a :class:`.BeamC`
model, or where a simple element with constant strains is preferred. Both
can be mixed with the shells, the truss and the springs in one model, and
both are wrapped by :class:`.CorotationalBeam` for large rotations, see
:doc:`corotational_formulation` and :doc:`element_choice`.


References
----------

- Bathe, K. J., Finite Element Procedures, 2nd ed., K. J. Bathe, Watertown,
  MA, 2014, ISBN 978-0-9790049-5-7.
- Brockman, R. A., Dynamics of the bilinear Mindlin plate element,
  International Journal for Numerical Methods in Engineering, 24(12),
  2343-2356, 1987. https://doi.org/10.1002/nme.1620241208
- Crisfield, M. A., Non-linear Finite Element Analysis of Solids and
  Structures, Volume 1: Essentials, John Wiley & Sons, Chichester, 1991.
- Crisfield, M. A., Non-linear Finite Element Analysis of Solids and
  Structures, Volume 2: Advanced Topics, John Wiley & Sons, Chichester, 1997.
- Luo, Y., An efficient 3D Timoshenko beam element with consistent shape
  functions, Advances in Theoretical and Applied Mechanics, 1(3), 95-106,
  2008.
- Petyt, M., Introduction to Finite Element Vibration Analysis, 2nd ed.,
  Cambridge University Press, Cambridge, 2010.
  https://doi.org/10.1017/CBO9780511761195
- Timoshenko, S. P. and Gere, J. M., Theory of Elastic Stability, 2nd ed.,
  McGraw-Hill, New York, 1961.
