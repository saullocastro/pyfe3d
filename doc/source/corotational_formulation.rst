Co-rotational formulation for large rotations
=============================================

The elements of pyfe3d are formulated in element axes fixed in the initial
configuration, with linear strains or the von Kármán strains of moderate
rotations, see :doc:`beam_theory`. Large displacements and rotations with
small strains are obtained without changing them, with the
element-independent co-rotational (EICR) formulation of Felippa and Haugen
(2005) and Nour-Omid and Rankin (1991), implemented in
:mod:`pyfe3d.corotational`: the motion of each element is split into a rigid
motion, followed by a co-rotational (CR) frame, and a small deformational
motion relative to it, to which the linear element responds. This chapter
presents the finite rotations, the kinematics of the split, the internal
force and the consistent tangent stiffness matrix, and the frames of the
beams, the truss and the spring; how a model is assembled and solved is in
:doc:`nonlinear_analyses`.


Finite rotations
----------------

**Rotation vector and rotation matrix.** The rotational degrees of freedom
of a node of a :class:`.CorotationalModel` are the components of its total
rotation vector `\boldsymbol{\psi}`, of angle `\psi = |\boldsymbol{\psi}|`
about the axis of `\boldsymbol{\psi}`, added by the solvers like the
translations. With `\mathrm{Spin}(\boldsymbol{a}) \boldsymbol{b} =
\boldsymbol{a} \times \boldsymbol{b}` (:func:`.spin`, and its inverse
:func:`.axial`), the nodal rotation matrix is the exponential map, Rodrigues'
formula (:func:`.expso3`):

.. math::
    :label: cr-exp

    \boldsymbol{R} = \exp(\mathrm{Spin}(\boldsymbol{\psi})) =
    \boldsymbol{I} + \frac{\sin \psi}{\psi} \mathrm{Spin}(\boldsymbol{\psi})
    + \frac{1 - \cos \psi}{\psi^2} \mathrm{Spin}(\boldsymbol{\psi})^2

Its inverse for angles up to `\pi`, :func:`.logso3`, extracts the quaternion
of `\boldsymbol{R}` with Spurrier's algorithm, accurate at every angle,
including 0 and `\pi`. Below the angles :data:`.SERIES_ANGLE` and
:data:`.SERIES_ANGLE_ETA_MU`, the coefficients of these matrices are
evaluated with their Taylor series, which avoid the cancellation of the
closed forms.

**Variations.** A variation of the rotation of a node is a spatial spin
`\delta \boldsymbol{\omega}`, `\delta \boldsymbol{R} \boldsymbol{R}^\top =
\mathrm{Spin}(\delta \boldsymbol{\omega})`, related to the variation of the
rotation vector by (:func:`.Lambda`)

.. math::
    :label: cr-Lambda

    \delta \boldsymbol{\omega} = \boldsymbol{\Lambda}(\boldsymbol{\psi})
    \delta \boldsymbol{\psi}, \qquad
    \boldsymbol{\Lambda} = \boldsymbol{I} + \frac{1 - \cos \psi}{\psi^2}
    \mathrm{Spin}(\boldsymbol{\psi}) + \frac{\psi - \sin \psi}{\psi^3}
    \mathrm{Spin}(\boldsymbol{\psi})^2

the `\boldsymbol{H}(\boldsymbol{\theta})` of Crisfield (1997), Eq. (16.90),
and the `\boldsymbol{H}^{-1}` of Felippa and Haugen (2005), Eq. (100). Its
inverse, for the small deformational rotations `\boldsymbol{\theta}` in the
CR frame, is (:func:`.Hmat`, Felippa and Haugen, Eq. (101))

.. math::
    :label: cr-H

    \delta \boldsymbol{\theta} = \boldsymbol{H}(\boldsymbol{\theta}) \delta
    \boldsymbol{\omega}, \qquad
    \boldsymbol{H} = \boldsymbol{I} - \frac{1}{2}
    \mathrm{Spin}(\boldsymbol{\theta}) + \eta \,
    \mathrm{Spin}(\boldsymbol{\theta})^2, \qquad
    \eta = \frac{1 - (\theta/2) \cot(\theta/2)}{\theta^2}

`\boldsymbol{\Lambda}` is singular at `\psi = 2 k \pi`, `k \ge 1`: a node
must turn by less than one full revolution, `|\boldsymbol{\psi}| < 2 \pi`.
Since the rotation vectors are total, no state is kept between the calls,
and the solvers of :class:`.Analysis` are used unchanged. The derivatives of
the transposes of `\boldsymbol{\Lambda}` and `\boldsymbol{H}` times a fixed
moment, which enter the tangent stiffness matrix, are :func:`.dLambdaT_m`
and :func:`.Lmat`.


Base and co-rotational frames
-----------------------------

The frames are written as `3 \times 3` matrices whose columns are the element
axes, the convention of the attributes ``r11, ..., r33`` of the elements
(Felippa and Haugen write their transposes): `\boldsymbol{T}_0`, the base
frame, which is that of ``update_rotation_matrix`` of the element, and
`\boldsymbol{T}_R`, the CR frame of the current configuration. With the base
local coordinates `\tilde{\boldsymbol{x}}_a = \boldsymbol{T}_0^\top
(\boldsymbol{X}_a - \boldsymbol{X}_C)` of node `a`, `\boldsymbol{X}_C` being
the mean of the nodal coordinates, the deformed positions
`\boldsymbol{x}^D_a` and the nodal rotation matrices `\boldsymbol{R}_a`, the
deformational displacements of node `a` in the CR frame are

.. math::
    :label: cr-deformational

    \bar{\boldsymbol{u}}_{d,a} = \boldsymbol{T}_R^\top (\boldsymbol{x}^D_a -
    \boldsymbol{x}_C) - \tilde{\boldsymbol{x}}_a, \qquad
    \bar{\boldsymbol{\theta}}_{d,a} = \mathrm{axial}\left(\log\left(
    \boldsymbol{T}_R^\top \boldsymbol{R}_a \boldsymbol{T}_0 \right)\right)

with `\boldsymbol{x}_C` the mean of `\boldsymbol{x}^D_a` (Felippa and Haugen,
Eq. (22); Nour-Omid and Rankin, Eq. (14)). In this chapter the overbar marks
components in the CR frame, as in Felippa and Haugen. The positions are
computed relative to the base centroid, `(\boldsymbol{X}_a -
\boldsymbol{X}_C) + \boldsymbol{u}_a`, which keeps their round-off
independent of the origin of the coordinates. A rigid motion of the element
moves `\boldsymbol{T}_R` with it and leaves the deformational displacements
`\boldsymbol{d}` zero; the element is then evaluated with its local linear
stiffness matrix `\boldsymbol{K}_e`, computed once in the base configuration
by its own ``update_KC0`` with the rotation matrix set to the identity.

The CR frame is defined by a frame rule of each element type,
:meth:`.CorotationalModel.add` choosing the wrapper:

.. list-table::
    :header-rows: 1

    * - Wrapper, elements
      - CR frame
    * - :class:`.CorotationalShell`, triangles
      - `x` along the side 1-2 and `z` normal to the plane of the nodes;
        the solution depends on the numbering of the nodes
    * - :class:`.CorotationalShell`, quadrilaterals
      - `z` along the cross product of the diagonals and `x` along their
        sum, invariant to a cyclic renumbering; a warped element is linked to
        its flat projection by rigid offsets, see the Notes of
        :class:`.CorotationalShell` and :doc:`element_choice`
    * - :class:`.CorotationalBeam`, :class:`.BeamC` and :class:`.BeamLR`
      - mean nodal triad, below
    * - :class:`.CorotationalTruss`
      - ``'chord'``: the axis alone, Crisfield's co-rotational truss;
        ``'mean_triad'``: as the beams, with the torsion
    * - :class:`.CorotationalSpring`
      - ``'axis'``: as the beams, for an axial spring along its nodes;
        ``'triad'``: mean of the rotated element axes, also for coincident
        nodes

**Mean nodal triad of the beams.** The `x` axis of the CR frame follows the
chord, and the `y` axis lies in the plane of the chord and of the mean of the
base `y` axis `\boldsymbol{q}_0 = \boldsymbol{T}_0 \boldsymbol{e}_2` rotated
by the two nodal rotations
(Crisfield, 1997, Eqs. (17.11)-(17.14); the midpoint interpolation of
Nour-Omid and Rankin, Section 5):

.. math::
    :label: cr-beam-frame

    \boldsymbol{e}_1 = \frac{\boldsymbol{x}^D_2 - \boldsymbol{x}^D_1}{l},
    \qquad \boldsymbol{q} = \frac{1}{2} (\boldsymbol{R}_1 + \boldsymbol{R}_2)
    \boldsymbol{q}_0, \qquad \boldsymbol{e}_3 = \frac{\boldsymbol{e}_1 \times
    \boldsymbol{q}}{|\boldsymbol{e}_1 \times \boldsymbol{q}|}, \qquad
    \boldsymbol{e}_2 = \boldsymbol{e}_3 \times \boldsymbol{e}_1

with `l` the current length. The frame is undefined when `\boldsymbol{q}`
becomes parallel to the chord; below the component `\bar{q}_2` of
``q2_min`` the wrapper returns NaN forces, so that the solvers reduce the
increment, and warns. Nour-Omid and Rankin, Appendix B.3, redefine
`\boldsymbol{q}` in that case, which a formulation without state cannot do
without a jump of the forces.


Internal force and consistent tangent stiffness
-----------------------------------------------

**Projector.** The variation of the CR frame follows from the variations of
the nodal displacements and spins in the CR frame, `\delta
\bar{\boldsymbol{v}}`, through the spin-fitter matrix `\boldsymbol{G}`,
`\delta \boldsymbol{\omega}_R = \boldsymbol{G} \delta \bar{\boldsymbol{v}}`,
derived in pyfe3d for each frame rule and given in the Notes of the wrappers;
for the mean nodal triad

.. math::
    :label: cr-G-beam

    \omega_1 = \frac{1}{2 q_2} \sum_i (\bar{q}_{i2} \omega_{i,x} -
    \bar{q}_{i1} \omega_{i,y}) - \frac{q_1}{q_2} \frac{w_2 - w_1}{l},
    \qquad \omega_2 = -\frac{w_2 - w_1}{l}, \qquad \omega_3 = \frac{v_2
    - v_1}{l}

with `\bar{\boldsymbol{q}} = \boldsymbol{T}_R^\top \boldsymbol{q} = (q_1, q_2,
0)`, `\bar{\boldsymbol{q}}_i = \boldsymbol{T}_R^\top \boldsymbol{R}_i
\boldsymbol{q}_0`, and `v_a`, `w_a`, `\omega_{i,x}`, `\omega_{i,y}` the
local translations and spins of the nodes. With the matrix `\boldsymbol{S}`
of the rigid rotations of the nodes about the centroid, `\boldsymbol{G}
\boldsymbol{S} = \boldsymbol{I}` (Felippa and Haugen, Eq. (17)), and the
projector

.. math::
    :label: cr-projector

    \boldsymbol{P} = \boldsymbol{P}_u - \boldsymbol{S} \boldsymbol{G}

removes the rigid motions from the variations of the element, `\boldsymbol{P}_u`
removing the rigid translations (Felippa and Haugen, Eqs. (11), (12),
(31)-(34)).

**Internal force.** With the local force `\boldsymbol{K}_e \boldsymbol{d}`
transformed from the deformational rotations to the spins by
`\bar{\boldsymbol{H}} = \mathrm{diag}(\boldsymbol{I},
\boldsymbol{H}(\bar{\boldsymbol{\theta}}_{d,a}))`, the internal force in the
CR frame is that of the consistent symmetrizable equilibrated (CSE)
formulation of Felippa and Haugen (2005), Section 6.3, Eq. (53):

.. math::
    :label: cr-force

    \bar{\boldsymbol{p}} = \boldsymbol{P}^\top \bar{\boldsymbol{H}}^\top
    \boldsymbol{K}_e \boldsymbol{d}

**Tangent stiffness.** Its variation gives the tangent stiffness matrix in
the CR frame,

.. math::
    :label: cr-tangent

    \boldsymbol{K}_R = \underbrace{\boldsymbol{P}^\top \bar{\boldsymbol{H}}^\top
    \boldsymbol{K}_e \bar{\boldsymbol{H}} \boldsymbol{P}}_{\text{material}}
    + \underbrace{\boldsymbol{P}^\top \bar{\boldsymbol{L}} \boldsymbol{P} -
    \boldsymbol{F}_{nm} \boldsymbol{G} - \boldsymbol{G}^\top
    \boldsymbol{F}_n^\top \boldsymbol{P}}_{\text{geometric}}

with `\bar{\boldsymbol{L}} = \mathrm{diag}(\boldsymbol{0},
\boldsymbol{L}(\bar{\boldsymbol{\theta}}_{d,a}, \bar{\boldsymbol{m}}_a))` of
the local nodal moments (:func:`.Lmat`), the moment-correction stiffness of
Felippa and Haugen, Eqs. (46) and (102), `-\boldsymbol{F}_{nm}
\boldsymbol{G}` the rotational geometric stiffness, Eqs. (44)-(45), and
`-\boldsymbol{G}^\top \boldsymbol{F}_n^\top \boldsymbol{P}` the
equilibrium-projection geometric stiffness, Eq. (47), with
`\boldsymbol{F}_n` and `\boldsymbol{F}_{nm}` built with the spins of the
forces and moments of `\bar{\boldsymbol{p}}`. Equation (47) is exact when
`\boldsymbol{G}` does not depend on the nodal rotations, their condition
(37), which holds for the frames of the shells. The mean nodal triad
depends on them, and :class:`.CorotationalBeam` uses, with ``exact_KGP=True``
(the default), the exact variation of `\boldsymbol{P}^\top \boldsymbol{p}`,

.. math::
    :label: cr-KGP

    \boldsymbol{K}_{GP} = -\frac{\partial (\boldsymbol{G}^\top
    \boldsymbol{z})}{\partial \bar{\boldsymbol{v}}} - \boldsymbol{G}^\top
    \boldsymbol{F}_n(\boldsymbol{p})^\top \boldsymbol{P}, \qquad
    \boldsymbol{z} = \boldsymbol{S}^\top \boldsymbol{p}, \qquad
    \boldsymbol{p} = \bar{\boldsymbol{H}}^\top \boldsymbol{K}_e \boldsymbol{d}

derived in pyfe3d, whose first term Eq. (47) misses, a term of the order of
the unbalanced force (Felippa and Haugen, Eq. (48)).

**Additive rotation vectors.** The force and the stiffness are transformed
to global components, `\boldsymbol{\mathcal{T}} =
\mathrm{diag}(\boldsymbol{T}_R)`, and from the spins to the variations of
the rotation vectors, `\hat{\boldsymbol{\Lambda}} =
\mathrm{diag}(\boldsymbol{I}, \boldsymbol{\Lambda}(\boldsymbol{\psi}_a))`
(Crisfield, 1997, Section 17.4.2, Eqs. (17.176)-(17.177)):

.. math::
    :label: cr-global

    \boldsymbol{f} = \hat{\boldsymbol{\Lambda}}^\top
    \boldsymbol{\mathcal{T}} \bar{\boldsymbol{p}}, \qquad
    \boldsymbol{K} = \hat{\boldsymbol{\Lambda}}^\top \boldsymbol{\mathcal{T}}
    \boldsymbol{K}_R \boldsymbol{\mathcal{T}}^\top \hat{\boldsymbol{\Lambda}}
    + \mathrm{diag}\left(\boldsymbol{0}, \frac{\partial
    (\boldsymbol{\Lambda}^\top \boldsymbol{m}_a)}{\partial
    \boldsymbol{\psi}_a}\right)

with `\boldsymbol{m}_a` the global moments of node `a` (:func:`.dLambdaT_m`).
`\boldsymbol{K}` is the Jacobian of `\boldsymbol{f}`, exact for the frames of
the shells, the chord of the truss and, with ``exact_KGP=True``, the nodal
triads of the beams and the springs, and unsymmetric away from equilibrium;
its material part enters ``calc_kC`` and its geometric part ``calc_kG`` of
the model. The steps are
those of the kernel ``_force_tangent_c`` of :mod:`pyfe3d.corotational`, and
each tangent is checked against central finite differences in
``tests/test_corotational_core.py``.


Truss and spring
----------------

With ``frame='chord'``, :class:`.CorotationalTruss` is the co-rotational truss
of Crisfield (1991), Section 3.6, written for the space truss of Section 3.7:
with `l_0` and `l` the initial and current lengths, `\boldsymbol{e}` the unit
vector from node 1 to node 2 and the axial force `N = (E A/l_0)(l - l_0)`,

.. math::
    :label: cr-truss

    \boldsymbol{f} = N \begin{Bmatrix} -\boldsymbol{e} \\ \boldsymbol{e}
    \end{Bmatrix}, \qquad
    \boldsymbol{K}_t = \frac{E A}{l_0} \boldsymbol{e} \boldsymbol{e}^\top +
    \frac{N}{l} (\boldsymbol{I} - \boldsymbol{e} \boldsymbol{e}^\top)

on the translations, `\boldsymbol{K}_t` and `-\boldsymbol{K}_t` being the
blocks of the two nodes, material and geometric. The rotations receive no
force, as in the linear element; with a torsional stiffness, ``'auto'``
chooses ``'mean_triad'``, the frame of the beams.

The linear :class:`.Spring` does not balance the moment of its transverse
forces between distinct nodes, see :doc:`truss_spring`; the projector of
:class:`.CorotationalSpring` adds it, so that its tangent at the base
configuration differs from ``KC0``. Its ``'triad'`` frame replaces the chord
of Eq. :eq:`cr-beam-frame` by the mean of the rotated base `x` axes, which
depends on the nodal rotations only and is defined for coincident nodes, up
to a relative rotation of the nodes of 180 degrees.


Von Kármán strains in the co-rotational frame
---------------------------------------------

With ``kinematics='corotational_von_karman'`` of :meth:`.CorotationalModel.add`
(``local='von_karman'`` of the wrappers), the local response is that of the
von Kármán strains of the element, step 8 of the procedure of Crisfield
(1997), Section 17.1.5: the element's own ``update_probe_finte(prop,
nonlinear=1)``, ``update_KCNL`` and ``update_KG`` are evaluated at
`\boldsymbol{d}`, and the local force `\bar{\boldsymbol{f}}(\boldsymbol{d})`
and the local tangent `\boldsymbol{K}_e + \boldsymbol{K}_{CNL}(\boldsymbol{d})
+ \boldsymbol{K}_G(\boldsymbol{d})` replace `\boldsymbol{K}_e
\boldsymbol{d}` and `\boldsymbol{K}_e` in Eqs. :eq:`cr-force` and
:eq:`cr-tangent`. In the CR frame of a beam the deformational translations
are along the chord, `\bar{v} = \bar{w} = 0`:

- :class:`.BeamLR` interpolates `v` and `w` linearly, so its von Kármán terms
  vanish and the result is that of ``'corotational'`` to round-off; the same
  holds for the shells, whose transverse translations in the CR frame are
  zero or the hourglass pattern of the warp;
- :class:`.BeamC` interpolates them with the nodal rotations, and its
  element-mean axial strain of :doc:`beam_theory`, Eq. :eq:`beamc-mean`,
  adds the local higher-order terms of Crisfield (1997), Section 17.1.5,
  Eqs. (17.56a)-(17.56b). On the cantilever under an end force of Sze, Liu
  and Lo (2004), Section 3.1, 16 elements give tip errors of -0.02 % and
  +0.05 % with ``'corotational'`` and +0.002 % and +0.007 % with
  ``'corotational_von_karman'``, see :class:`.CorotationalBeam`.

The truss and the spring, without von Kármán strains of their own, take
``'corotational_von_karman'`` as ``'corotational'``.


Limits of validity
------------------

- Small strains: the local response is linear, or von Kármán, in the CR
  frame; large strains are not represented.
- Each nodal rotation vector below one revolution, `|\boldsymbol{\psi}_a| <
  2 \pi`.
- The mean nodal triad is undefined when the mean of the rotated `y` axes
  turns parallel to the chord, ``q2_min``; for a truss without torsion,
  ``'chord'`` avoids it.
- A node shared with an element of another kinematics is valid only while its
  rotations remain small, since that element reads the rotation vector as a
  small rotation; a warning is issued.
- Moments in the constant external load vector ``fext`` are conjugate to the
  rotation vectors; moments about fixed axes and follower moments depend on
  the configuration and are added with :meth:`.CorotationalModel.add_moment`,
  see :doc:`nonlinear_analyses`.


References
----------

- Crisfield, M. A., Non-linear Finite Element Analysis of Solids and
  Structures, Volume 1: Essentials, John Wiley & Sons, Chichester, 1991.
- Crisfield, M. A., Non-linear Finite Element Analysis of Solids and
  Structures, Volume 2: Advanced Topics, John Wiley & Sons, Chichester, 1997.
- Felippa, C. A. and Haugen, B., A unified formulation of small-strain
  corotational finite elements: I. Theory, Computer Methods in Applied
  Mechanics and Engineering, 194(21-24), 2285-2335, 2005.
  https://doi.org/10.1016/j.cma.2004.07.035
- Nour-Omid, B. and Rankin, C. C., Finite rotation analysis and consistent
  linearization using projectors, Computer Methods in Applied Mechanics and
  Engineering, 93(3), 353-384, 1991.
  https://doi.org/10.1016/0045-7825(91)90248-5
- Sze, K. Y., Liu, X. H. and Lo, S. H., Popular benchmark problems for
  geometric nonlinear analysis of shells, Finite Elements in Analysis and
  Design, 40(11), 1551-1569, 2004. https://doi.org/10.1016/j.finel.2003.11.001
