Shell properties
================

The shell elements read their constitutive matrices and mass integrals
from a :class:`.ShellProp`, module :mod:`pyfe3d.shellprop`, passed to every
``update_*`` method that needs it; the functions of
:mod:`pyfe3d.shellprop_utils` create it from the ply data. This chapter
gives the equivalent single-layer theory behind these classes: the
stiffnesses of a ply, their integration through the thickness, the offset
of the reference surface, the transverse shear stiffness with its
correction, the rotation to the element axes and the lamination
parameters. The use of these matrices by the elements is described in
:doc:`shell_formulation`; the beam properties in :doc:`beam_properties`.


Stiffness of a ply
------------------

A ply is described by a :class:`.MatLamina`, usually created by
:func:`.read_laminaprop` from ``laminaprop = (E11, E22, nu12, G12, G13,
G23)``, ``(E, nu)`` for an isotropic material, for which `G = E/(2(1 +
\nu))`, or nine entries with ``E33``, ``nu13`` and ``nu23``, and by a
:class:`.Lamina` with its thickness and angle. The shell elements use the
plane-stress reduced stiffnesses, `\sigma_{33} = 0`, in the material axes
1, 2 of the ply (Reddy, 2003, Eq. (2.4.4); Jones, 1999, Eq. (2.66)),

.. math::
    :label: sh-ply-Q

    Q_{11} = \frac{E_{11}}{1 - \nu_{12} \nu_{21}}, \quad
    Q_{12} = \frac{\nu_{12} E_{22}}{1 - \nu_{12} \nu_{21}}, \quad
    Q_{22} = \frac{E_{22}}{1 - \nu_{12} \nu_{21}}, \quad
    Q_{66} = G_{12}, \quad Q_{44} = G_{23}, \quad Q_{55} = G_{13}

with `\nu_{21} = \nu_{12} E_{22}/E_{11}`, which depend on ``E11``,
``E22``, ``nu12``, ``G12``, ``G13`` and ``G23`` only, also when
``laminaprop`` gives ``E33``, ``nu13`` and ``nu23``, see
:meth:`.Lamina.rebuild` and :meth:`.MatLamina.rebuild`. With the ply angle
`\theta` measured from the laminate `x` axis to the fibre direction 1, `c
= \cos\theta` and `s = \sin\theta`, the stiffnesses in the laminate axes
are (Reddy, 2003)

.. math::
    :label: sh-ply-rotation

    \begin{aligned}
    q_{11L} &= Q_{11} c^4 + 2 (Q_{12} + 2 Q_{66}) s^2 c^2 + Q_{22} s^4,
    &\quad
    q_{22L} &= Q_{11} s^4 + 2 (Q_{12} + 2 Q_{66}) s^2 c^2 + Q_{22} c^4, \\
    q_{12L} &= (Q_{11} + Q_{22} - 4 Q_{66}) s^2 c^2 + Q_{12} (s^4 + c^4),
    &\quad
    q_{66L} &= (Q_{11} + Q_{22} - 2 Q_{12} - 2 Q_{66}) s^2 c^2 + Q_{66}
    (s^4 + c^4), \\
    q_{16L} &= (Q_{11} - Q_{12} - 2 Q_{66}) s c^3 + (Q_{12} - Q_{22} + 2
    Q_{66}) s^3 c,
    &\quad
    q_{26L} &= (Q_{11} - Q_{12} - 2 Q_{66}) s^3 c + (Q_{12} - Q_{22} + 2
    Q_{66}) s c^3, \\
    q_{44L} &= Q_{44} c^2 + Q_{55} s^2, \quad q_{45L} = (Q_{55} - Q_{44})
    s c,
    &\quad
    q_{55L} &= Q_{55} c^2 + Q_{44} s^2
    \end{aligned}

The same stiffnesses are linear in `\cos 2\theta`, `\sin 2\theta`, `\cos
4\theta` and `\sin 4\theta`, with the invariants of Tsai and Pagano, as
given by Jones (1999), Eq. (2.99),

.. math::
    :label: sh-invariants

    u_1 = \frac{3 Q_{11} + 3 Q_{22} + 2 Q_{12} + 4 Q_{66}}{8}, \quad
    u_2 = \frac{Q_{11} - Q_{22}}{2}, \quad
    u_3 = \frac{Q_{11} + Q_{22} - 2 Q_{12} - 4 Q_{66}}{8}, \quad
    u_4 = \frac{Q_{11} + Q_{22} + 6 Q_{12} - 4 Q_{66}}{8}

with `u_5 = (u_1 - u_4)/2`, and the transverse shear invariants `u_6 =
(Q_{44} + Q_{55})/2`, `u_7 = (Q_{44} - Q_{55})/2`, the attributes ``u1``
to ``u7`` of :class:`.MatLamina`, which give the stiffnesses of the
lamination parameters below. ``MatLamina.rebuild(plane_stress=False)``
gives the three-dimensional stiffnesses and invariants instead, e.g. for
continuum shell elements of other codes; the shell elements of ``pyfe3d``
do not use them. Thermal expansion is not modelled. A zero modulus is
admitted as the limit of a positive-definite material, e.g. the antiplane
core ``laminaprop = (0, 0, nu12, 0, G13, G23)``, which adds only
transverse shear stiffness to a laminate.


Stiffness of the laminate
-------------------------

With the plies of a :class:`.ShellProp` stacked from the bottom, ply `k`
occupying `z_k \le z \le z_{k+1}`, `z_1 = -h/2 + e` and `z_{N+1} = h/2 +
e`, `h` the total thickness and `e` = ``offset``, the stiffnesses of
:eq:`sh-abd` and the mass integrals are, see
:meth:`.ShellProp.calc_constitutive_matrix`,

.. math::
    :label: sh-laminate

    (A_{ij}, B_{ij}, D_{ij}) = \sum_k \int_{z_k}^{z_{k+1}} q_{ijL}^{(k)}
    (1, z, z^2) \, dz, \qquad
    (I_\rho, I_{\rho z}, I_{\rho z^2}) = \sum_k \int_{z_k}^{z_{k+1}}
    \rho^{(k)} (1, z, z^2) \, dz

for `ij = 11, 12, 16, 22, 26, 66`, the attributes ``A11`` to ``D66``,
``intrho``, ``intrhoz`` and ``intrhoz2``. A property is created by:

- :func:`.isotropic_plate`, from ``thickness``, ``E``, ``nu`` and,
  optionally, ``rho``, ``offset`` and ``shear_correction``;
- :func:`.laminated_plate`, from the stacking sequence ``stack``, the ply
  angles in degrees from the bottom to the top ply, the ply thickness
  ``plyt`` or ``plyts``, and ``laminaprop`` or ``laminaprops``;
- :func:`.shellprop_from_LaminationParameters`, from the thickness, a
  :class:`.MatLamina` and the :class:`.LaminationParameters`, see below.

The equivalent moduli ``e1``, ``e2``, ``g12``, ``nu12``, ``nu21`` of
:meth:`.ShellProp.calc_equivalent_properties` are those of the membrane
compliance with zero moments, e.g. `e_1 = 1/(h a_{11})` with `\boldsymbol{a}
= [[\boldsymbol{A}, \boldsymbol{B}], [\boldsymbol{B},
\boldsymbol{D}]]^{-1}`.


Offset of the reference surface
-------------------------------

The nodes of the shell elements are on the reference surface, and `z` is
measured from it along the element normal. With ``offset`` `= e`, the
mid-surface of the laminate is at the distance `e` from the nodes. For a
single isotropic layer, :eq:`sh-laminate` gives

.. math::
    :label: sh-offset-isotropic

    A_{11} = A_{22} = \frac{E h}{1 - \nu^2}, \quad A_{12} = \nu A_{11},
    \quad A_{66} = G h, \qquad
    \boldsymbol{B} = e \boldsymbol{A}, \qquad
    \boldsymbol{D} = \left( \frac{h^2}{12} + e^2 \right) \boldsymbol{A},
    \qquad I_{\rho z} = \rho h e, \qquad
    I_{\rho z^2} = \rho \left( \frac{h^3}{12} + h e^2 \right)

and, for any laminate, with `\boldsymbol{B}^0` and `\boldsymbol{D}^0` about
the mid-surface (Vasiliev and Morozov, 2001, Eq. (5.28)),

.. math::
    :label: sh-offset-shift

    \boldsymbol{B} = \boldsymbol{B}^0 + e \boldsymbol{A}, \qquad
    \boldsymbol{D} = \boldsymbol{D}^0 + 2 e \boldsymbol{B}^0 + e^2
    \boldsymbol{A}

while `\boldsymbol{A}` and the transverse shear stiffnesses do not change.
An offset couples the membrane and the bending responses as an
unsymmetric laminate does, which models e.g. a skin whose nodes are on
its inner surface, shared with the beams of a stiffener; it also moves the
centre of mass in the mass matrices, see :doc:`shell_mass`.


Transverse shear stiffness and shear correction
-----------------------------------------------

The transverse shear strains of the FSDT are constant through the
thickness, whereas the transverse shear stresses of a bent plate vanish at
its faces, and the stiffness `\bar{\boldsymbol{A}}_{ts} = \sum_k
\boldsymbol{C}_s^{(k)} h_k`, with `\boldsymbol{C}_s^{(k)} = [[q_{44L},
q_{45L}], [q_{45L}, q_{55L}]]`, overestimates the shear stiffness. The
attributes ``A44``, ``A45`` and ``A55`` of a :class:`.ShellProp` are the
stiffnesses used by the elements **with the correction already applied**,
and the elements apply no other factor. The correction is selected with
``shear_correction``, see :meth:`.ShellProp.calc_transverse_shear_stiffness`.

**Equilibrium approach of Rohwer (1988), the default** ``'rohwer'``. The
transverse shear stresses are obtained from the equilibrium of two
cylindrical bending states with constant membrane forces, one with the
resultants depending on `x` only, `M_{xx,x} = Q_x` and `M_{xy,x} = Q_y`,
and one with the resultants depending on `y` only, by integrating
`\tau_{xz,z} = -\sigma_{xx,x}` and `\tau_{yz,z} = -\sigma_{yy,y}` from the
bottom face, where the tractions vanish, with the in-plane strains of the
inverse of the ABD matrix. The result is a distribution `\{\tau_{yz},
\tau_{xz}\} = \boldsymbol{f}^{(k)}(z) \{Q_y, Q_x\}`, quadratic in each
ply, continuous at the interfaces and zero at both faces, and the stiffness
follows from the complementary energy:

.. math::
    :label: sh-rohwer

    \boldsymbol{A}_{ts} = \left[ \sum_k \int_{z_k}^{z_{k+1}}
    \boldsymbol{f}^{(k)\top} \left( \boldsymbol{C}_s^{(k)} \right)^{-1}
    \boldsymbol{f}^{(k)} \, dz \right]^{-1}

integrated exactly with three Gauss-Legendre points per ply. It is valid
for arbitrary anisotropic and unsymmetric laminates, gives the full `2
\times 2` matrix and does not depend on ``offset``. It is not invariant to
a rotation of the axes, because the two bending states are tied to `x` and
`y`: the elements use it re-evaluated with the plies rotated to their own
axes, see :meth:`.ShellProp.calc_Ats_element`. The compliance in a frame
rotated by `\theta` is a trigonometric polynomial of degree 10 with a
period of 180 degrees,

.. math::
    :label: sh-rohwer-fourier

    \boldsymbol{A}_{ts}^{-1}(\theta) = \boldsymbol{S}_0 + \sum_{n=1}^{5}
    \left( \boldsymbol{S}_{cn} \cos 2n\theta + \boldsymbol{S}_{sn} \sin
    2n\theta \right)

whose coefficients are computed exactly, once per laminate, from 16
rotated evaluations, such that the cost per element does not depend on the
number of plies.

**Scalar factors.** ``'vlachoutsis'`` applies the factors of Vlachoutsis
(1992) to the constant-strain stiffness, ``A55 = k13*Abar55``, ``A44 =
k23*Abar44`` and ``A45 = (k13 + k23)/2*Abar45``; for the direction 1, with
`C_{11} = q_{11L}` and `C_{55} = q_{55L}` of each ply,

.. math::
    :label: sh-vlachoutsis

    z_n = \frac{\int C_{11} z \, dz}{\int C_{11} \, dz}, \quad
    R = \int C_{11} (z - z_n)^2 dz, \quad
    g(z) = -\int_{z_1}^{z} C_{11} (\bar{z} - z_n) \, d\bar{z}, \quad
    k_{13} = \frac{R^2}{\bar{A}_{55} \int g^2/C_{55} \, dz}

i.e. the energy of the stress `\tau_{xz} = Q_x g(z)/R` of the
one-dimensional equilibrium equals that of the FSDT, and analogously
`k_{23}` with `C_{22}`, `C_{44}`. The other options are applied in the
same way:

- ``'whitney'``: Whitney (1973), identical to ``'vlachoutsis'``;
- ``'chow'``: Chow (1971), for laminates symmetric about the mid-surface,
  for which it is identical to ``'vlachoutsis'``;
- ``'birman_bert'``: the average shear strain factors of Birman and Bert
  (2002);
- ``'thickness_shear'``: the factors that match the first thickness-shear
  frequency of the laminate, after Yang, Norris and Stavsky (1966), as
  described by Sun and Whitney (1973), extending Mindlin (1951); it
  requires the ply densities;
- ``'constant'``: `k = 5/6`;
- ``None``: no correction, `\boldsymbol{A}_{ts} = \bar{\boldsymbol{A}}_{ts}`.

For an isotropic plate ``'rohwer'``, ``'vlachoutsis'``, ``'whitney'``,
``'chow'`` and ``'constant'`` give `5/6 \, G h`, ``'birman_bert'`` gives
`G h` and ``'thickness_shear'`` gives `\pi^2/12 \, G h`. Except for
``'rohwer'``, the elements rotate `\boldsymbol{A}_{ts}` to their axes as a
tensor. The constant-strain ``Abar44``, ``Abar45``, ``Abar55`` and the
constant-stress `\bar{\bar{\boldsymbol{A}}}_{ts} = h^2 [\sum_k
(\boldsymbol{C}_s^{(k)})^{-1} h_k]^{-1}`, ``Abarbar44``, ``Abarbar45``,
``Abarbar55``, are kept for comparison, and ``scf_k13``, ``scf_k23`` are
the resulting ratios, informative only. A :class:`.ShellProp` created from
lamination parameters has no plies and no correction; ``A44``, ``A45``,
``A55`` can then be set directly.

**Sandwich sections.** For stiff faces on a soft core the factor `5/6` is
badly wrong: ``'rohwer'`` and ``'vlachoutsis'`` reproduce the factors
tabulated by Vlachoutsis (1992), Tables I and II, down to `K = 0.01964`,
for which `5/6` overestimates the shear stiffness 42 times, and, for an
antiplane core, the shear stiffness `G_c d^2/c` of Allen (1969), Eq.
(2.15b), with `d` the distance between the centroids of the faces, within
the bounds given by the local bending and the shear strains of the faces
that Allen neglects. The sandwich panel of Rohwer (1988), Sec. 4.3, is not
used as a reference, because its printed stiffnesses are not consistent
with the stated lay-up, see the docstring of
``tests/test_shellprop_transverse_shear.py``.

**Transverse shear stresses.** The stresses of the element strains,
`\boldsymbol{C}_s \boldsymbol{\gamma}^{(0)}`, are constant in each ply and
non-zero at the faces. The stresses for e.g. a failure criterion are
recovered from the shear forces with the distribution of Rohwer,
:meth:`.ShellProp.calc_transverse_shear_stress`, which for a homogeneous
plate is the parabola `\tau_{xz} = 3 Q_x/(2h) (1 - 4 \bar{z}^2/h^2)`, or
from the derivatives of the generalised strains of a solution, by
integrating the three-dimensional equilibrium equations through the
thickness, :meth:`.ShellProp.calc_equilibrium_transverse_shear`, as in the
a posteriori approach of Noor and Peters (1989), whose energy is given by
:meth:`.ShellProp.calc_aposteriori_energy`.


Constitutive matrices in the element axes
-----------------------------------------

The ply angles are measured from the material direction of each element,
see :doc:`conventions`, and the elements rotate the matrices from the
material direction to their own axes with ``m11, m12, m21, m22``, `m_{11}
= m_{22} = \cos\theta`, `m_{21} = -m_{12} = \sin\theta`, `\theta` being
the angle from the element `x` axis to the material direction:

.. math::
    :label: sh-rotation-element

    \boldsymbol{A}_e = \boldsymbol{T}_\sigma \boldsymbol{A}
    \boldsymbol{T}_\sigma^\top, \quad
    \boldsymbol{T}_\sigma = \begin{bmatrix} m_{11}^2 & m_{12}^2 & 2 m_{11}
    m_{12} \\ m_{21}^2 & m_{22}^2 & 2 m_{21} m_{22} \\ m_{11} m_{21} &
    m_{12} m_{22} & m_{11} m_{22} + m_{12} m_{21} \end{bmatrix}, \qquad
    \boldsymbol{A}_{ts,e} = \boldsymbol{T}_\gamma^\top \boldsymbol{A}_{ts}
    \boldsymbol{T}_\gamma, \quad
    \boldsymbol{T}_\gamma = \begin{bmatrix} m_{22} & m_{12} \\ m_{21} &
    m_{11} \end{bmatrix}

and the same for `\boldsymbol{B}` and `\boldsymbol{D}`; with
``'rohwer'`` and plies, `\boldsymbol{A}_{ts,e}` is :eq:`sh-rohwer-fourier`
instead. The rotated matrices are given by
:meth:`.ShellProp.calc_constitutive_element` and
:meth:`.ShellProp.calc_Ats_element`: a ply at 0 degrees seen in an element
whose material direction is at `\theta` from its `x` axis has the
stiffnesses of a ply at `\theta`.


Lamination parameters
---------------------

With `\bar{z} = z/h` measured from the reference surface and the ply angle
`\theta(\bar{z})`, the lamination parameters of
:meth:`.ShellProp.calc_lamination_parameters` are

.. math::
    :label: sh-lamination-parameters

    \{\xi_{A1}, \xi_{A2}, \xi_{A3}, \xi_{A4}\} = \int \{\cos 2\theta,
    \sin 2\theta, \cos 4\theta, \sin 4\theta\} \, d\bar{z}, \qquad
    \xi_{Bi} = 4 \int (\cdot) \, \bar{z} \, d\bar{z}, \qquad
    \xi_{Di} = 12 \int (\cdot) \, \bar{z}^2 d\bar{z}

with the same four functions, and `\{\xi_{Ats,1}, \xi_{Ats,2}\} = \int
\{\cos 2\theta, \sin 2\theta\} d\bar{z}`; the `\xi_{Ai}` and `\xi_{Di}`
are those of Krupa et al. (2018) with ``offset = 0``. For a laminate of
one material the stiffnesses are linear in them, with the invariants
:eq:`sh-invariants`, e.g.

.. math::
    :label: sh-lp-stiffness

    A_{11} = h (u_1 + u_2 \xi_{A1} + u_3 \xi_{A3}), \quad
    A_{16} = h \left( \frac{u_2}{2} \xi_{A2} + u_3 \xi_{A4} \right), \quad
    \bar{A}_{44} = h (u_6 + u_7 \xi_{Ats,1}), \quad
    D_{11} = \frac{h^3}{12} \left( (1 + 12 \bar{e}^2) u_1 + u_2 \xi_{D1} +
    u_3 \xi_{D3} \right)

with `\bar{e} = e/h`, see :func:`.shellprop_from_LaminationParameters`,
whose Notes give all the terms, and whose transverse shear stiffness is the
uncorrected one, the distribution of the plies being unknown. The
lamination parameters include the offset, and
:func:`.shellprop_from_LaminationParameters` needs the same ``offset`` to
give back the stiffnesses. :class:`.GradABD` gives the derivatives of the
stiffnesses with respect to the thickness and the lamination parameters,
e.g. for the optimisation of the stiffness distribution.

**Idealised laminates.** After :meth:`.ShellProp.calc_constitutive_matrix`,
:meth:`.ShellProp.make_balanced`, :meth:`.ShellProp.make_orthotropic` and
:meth:`.ShellProp.make_symmetric` set to zero the stiffnesses that vanish
for a balanced laminate, `A_{16}`, `A_{26}` and `\bar{A}_{45}`, for an
orthotropic one, all the terms 16, 26 and 45, or for a symmetric one,
`\boldsymbol{B}`, and :meth:`.ShellProp.make_smeared` replaces
`\boldsymbol{D}` by `h^2 \boldsymbol{A}/12` and `\boldsymbol{B}` by zero;
they require ``offset = 0``. In a balanced laminate `B_{16}`, `B_{26}`,
`D_{16}` and `D_{26}` depend on the positions of the plies and do not
vanish in general, and with ``'rohwer'`` ``A45`` neither. The functions
:func:`.make_balanced_LP`, :func:`.make_orthotropic_LP` and
:func:`.make_symmetric_LP` do the same with the lamination parameters.



Limits of validity
------------------

- Equivalent single layer: the laminate is represented by the stiffnesses
  of its reference surface, with the plane-stress stiffnesses of each ply
  and an inextensible normal; the transverse shear stresses are recovered
  a posteriori.
- The shear correction of ``'rohwer'`` rests on two cylindrical bending
  states with constant membrane forces; the scalar factors on the
  one-dimensional equilibrium along each direction, exact only for
  specially orthotropic plies, and their ``A45`` is an average that cannot
  represent the coupling of angle-ply laminates with ``Abar45 = 0``.
- Linear elastic plies, without thermal strains; the strengths of
  :class:`.MatLamina` are stored and not used.


References
----------

- Allen, H. G., Analysis and Design of Structural Sandwich Panels,
  Pergamon Press, 1969, Chap. 2.
  https://doi.org/10.1016/B978-0-08-012870-2.50006-7
- Birman, V. and Bert, C. W., On the choice of shear correction factor in
  sandwich structures, Journal of Sandwich Structures and Materials, 4,
  83-95, 2002.
- Chow, T. S., On the propagation of flexural waves in an orthotropic
  laminated plate and its response to an impulsive load, Journal of
  Composite Materials, 5, 306-319, 1971.
- Jones, R. M., Mechanics of Composite Materials, Second Edition, Taylor &
  Francis, Philadelphia, 1999.
- Krupa, E. P., Cooper, J. E., Pirrera, A. and Nangia, R., Improved
  aerostructural performance via aeroservoelastic tailoring of a composite
  wing, The Aeronautical Journal, 122(1255), 1442-1474, 2018.
  https://doi.org/10.1017/aer.2018.66
- Mindlin, R. D., Influence of rotatory inertia and shear on flexural
  motions of isotropic, elastic plates, Journal of Applied Mechanics, 18,
  31-38, 1951.
- Noor, A. K. and Peters, J. M., A posteriori estimates for the shear
  correction factors in multi-layered composite cylinders, Journal of
  Engineering Mechanics, 115, 1225-1244, 1989.
- Reddy, J. N., Mechanics of Laminated Composite Plates and Shells: Theory
  and Analysis, Second Edition, CRC Press, 2003.
  https://doi.org/10.1201/b12409
- Rohwer, K., Improved transverse shear stiffness for layered finite
  elements, DFVLR-FB 88-32, 1988.
- Sun, C. T. and Whitney, J. M., Theories for the dynamic response of
  laminated plates, AIAA Journal, 11, 178-183, 1973.
- Vasiliev, V. V. and Morozov, E. V., Mechanics and Analysis of Composite
  Materials, Elsevier, Oxford, 2001.
- Vlachoutsis, S., Shear correction factors for plates and shells,
  International Journal for Numerical Methods in Engineering, 33(7),
  1537-1552, 1992. https://doi.org/10.1002/nme.1620330712
- Whitney, J. M., Shear correction factors for orthotropic laminates under
  static load, Journal of Applied Mechanics, 40, 302-304, 1973.
- Yang, P. C., Norris, C. H. and Stavsky, Y., Elastic wave propagation in
  heterogeneous plates, International Journal of Solids and Structures, 2,
  665-684, 1966.
