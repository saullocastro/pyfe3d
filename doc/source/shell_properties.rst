Shell and beam properties
=========================

Shell properties and laminates
------------------------------

The shell elements read their constitutive matrices and mass integrals
from a :class:`.ShellProp`, passed to every ``update_*`` method that needs
it. The stiffnesses of the first-order shear deformation theory are

.. math::

    \begin{Bmatrix} N \\ M \end{Bmatrix} =
    \begin{bmatrix} A & B \\ B & D \end{bmatrix}
    \begin{Bmatrix} \varepsilon^{(0)} \\ \varepsilon^{(1)} \end{Bmatrix},
    \qquad
    \begin{Bmatrix} Q_y \\ Q_x \end{Bmatrix} = \boldsymbol{A}_{ts}
    \begin{Bmatrix} \gamma_{yz} \\ \gamma_{xz} \end{Bmatrix}

with `A_{ij}`, `B_{ij}`, `D_{ij}` the thickness integrals of the ply
stiffnesses times `1`, `z` and `z^2`, and `\boldsymbol{A}_{ts}` = ``[[A44, A45],
[A45, A55]]``, see :class:`.ShellProp`. A property is created by:

- :func:`.isotropic_plate`, from ``thickness``, ``E``, ``nu`` and,
  optionally, ``rho``, ``offset`` and ``shear_correction``;
- :func:`.laminated_plate`, from the stacking sequence ``stack``, the ply
  angles in degrees from the bottom to the top ply, the ply thickness
  ``plyt`` or ``plyts``, and the ply material ``laminaprop = (E11, E22,
  nu12, G12, G13, G23)`` or ``laminaprops``, see :func:`.read_laminaprop`;
- :func:`.shellprop_from_LaminationParameters`, from the thickness, a
  :class:`.MatLamina` and the :class:`.LaminationParameters`, whose
  gradients are given by :class:`.GradABD`, e.g. for the optimisation of
  the stiffness distribution, see the tutorials.

The ply stiffnesses are those of plane stress, `\sigma_{33} = 0`, see
:meth:`.Lamina.rebuild`, which depend on ``E11``, ``E22``, ``nu12``,
``G12``, ``G13`` and ``G23`` only, also when ``laminaprop`` gives ``E33``,
``nu13`` and ``nu23``. The invariants of :class:`.MatLamina`, used by
:func:`.shellprop_from_LaminationParameters` and :class:`.GradABD`, are by
default those of the same plane-stress stiffnesses, see
:meth:`.MatLamina.rebuild`, such that the lamination parameters of
:meth:`.ShellProp.calc_lamination_parameters` give back the stiffnesses of
:func:`.laminated_plate`, for any ``laminaprop``:

.. literalinclude:: ../../tests/test_shellprop.py
    :pyobject: test_lamination_parameters_three_dimensional_constants

``MatLamina.rebuild(plane_stress=False)`` gives the three-dimensional
stiffnesses and invariants instead, e.g. for continuum shell elements of
other codes; the shell elements of pyfe3d do not use them.
Plane strain and thermal expansion are not modelled: the thermal expansion
coefficients and the reference temperature of :class:`.MatLamina` are not
used.

The ply angles are measured from the material direction of each element,
the vector ``xmati, xmatj, xmatk`` of ``update_rotation_matrix`` projected
onto the element, or from the element `x` axis when it is not given, see
:doc:`conventions`. The elements rotate the matrices from the material
direction to the element axes with ``m11, m12, m21, m22``;
:meth:`.ShellProp.calc_constitutive_element` gives the rotated matrices.


Offset of the reference surface
-------------------------------

The nodes of the shell elements are on the reference surface, and `z` is
measured from it. With ``offset`` `= e`, the laminate occupies `-h/2 + e
\le z \le h/2 + e`, i.e. its mid-surface is at the distance `e` from the
nodes, along the element `z` axis. For a single isotropic layer the
integrals give

.. math::

    \boldsymbol{B} = e \boldsymbol{A}, \qquad \boldsymbol{D} = \boldsymbol{D}_{e=0} + e^2 \boldsymbol{A}, \qquad
    \int \rho z \, dz = \rho h e, \qquad
    \int \rho z^2 \, dz = \rho \left(\frac{h^3}{12} + h e^2\right)

such that an offset couples the membrane and bending responses. It
models e.g. a skin whose nodes are on its inner surface, shared with the
beams of a stiffener. The mass integrals ``intrho``, ``intrhoz`` and
``intrhoz2`` enter the mass matrices of the shells.

For any laminate, `\boldsymbol{B} = \boldsymbol{B}_{e=0} + e
\boldsymbol{A}` and `\boldsymbol{D} = \boldsymbol{D}_{e=0} + 2 e
\boldsymbol{B}_{e=0} + e^2 \boldsymbol{A}`, the shift of the stiffness
integrals to another reference plane (Vasiliev and Morozov, Mechanics and
Analysis of Composite Materials, Elsevier, 2001, Eq. (5.28), p. 235), while
`\boldsymbol{A}` and the transverse shear stiffnesses do not change. The
lamination parameters of :meth:`.ShellProp.calc_lamination_parameters` are
integrated about the reference surface, i.e. they include the offset, and
:func:`.shellprop_from_LaminationParameters` needs the same ``offset`` to
give back the stiffnesses:

.. literalinclude:: ../../tests/test_shellprop.py
    :pyobject: test_lamination_parameters_offset


Transverse shear correction
---------------------------

The stiffnesses ``A44``, ``A45`` and ``A55`` of a :class:`.ShellProp` are
the ones used by the elements, with the shear correction already applied,
and the elements apply no other correction factor. The correction is
selected with ``shear_correction``, see
:meth:`.ShellProp.calc_transverse_shear_stiffness`:

- ``'rohwer'``, the default: the equilibrium approach of Rohwer (1988),
  giving the full matrix `\boldsymbol{A}_{ts}`, re-evaluated in the element axes by
  :meth:`.ShellProp.calc_Ats_element` when the material direction differs
  from the element `x` axis;
- ``'vlachoutsis'``: the two factors of Vlachoutsis (1992),
  `doi:10.1002/nme.1620330712 <https://doi.org/10.1002/nme.1620330712>`_,
  identical to ``'whitney'`` (Whitney, 1973) and, for laminates symmetric
  about the mid-surface, to ``'chow'`` (Chow, 1971);
- ``'birman_bert'``: the average shear strain factors of Birman and Bert
  (2002);
- ``'thickness_shear'``: the factors that match the first thickness-shear
  frequency of the laminate, which require the ply densities;
- ``'constant'``: `k = 5/6`;
- ``None``: no correction, `\boldsymbol{A}_{ts} = \sum_k \boldsymbol{C}_s^{(k)} h_k`.

For an isotropic plate ``'rohwer'``, ``'vlachoutsis'``, ``'whitney'``,
``'chow'`` and ``'constant'`` give `5/6 \, G h`, ``'birman_bert'`` gives
`G h` and ``'thickness_shear'`` gives `\pi^2/12 \, G h`. Except for
``'rohwer'``, the elements rotate `\boldsymbol{A}_{ts}` to the element axes as a
tensor. None of the corrections depends on ``offset``. The uncorrected
stiffnesses ``Abar44``, ``Abar45``, ``Abar55`` and the constant-stress ones
``Abarbar44``, ``Abarbar45``, ``Abarbar55`` are kept for comparison, and
``scf_k13``, ``scf_k23`` are the resulting ratios, informative only. A
:class:`.ShellProp` created from lamination parameters has no plies and no
correction; ``A44``, ``A45``, ``A55`` can then be set directly.

The transverse shear stresses through the thickness are recovered with
:meth:`.ShellProp.calc_transverse_shear_stress`, from the shear forces and
the distribution of Rohwer (1988), or with
:meth:`.ShellProp.calc_equilibrium_transverse_shear`, from the derivatives of
the generalized strains of a solution, whose strain energy is given by
:meth:`.ShellProp.calc_aposteriori_energy`.


Idealized laminates
-------------------

After :meth:`.ShellProp.calc_constitutive_matrix`, the methods
:meth:`.ShellProp.make_balanced`, :meth:`.ShellProp.make_orthotropic`,
:meth:`.ShellProp.make_symmetric` and :meth:`.ShellProp.make_smeared` set to
zero the stiffnesses that vanish for a balanced, orthotropic or symmetric
laminate, or replace `\boldsymbol{D}` by `h^2 \boldsymbol{A}/12`, which requires ``offset = 0``.
:func:`.make_balanced_LP`, :func:`.make_orthotropic_LP` and
:func:`.make_symmetric_LP` do the same with the lamination parameters, such
that :func:`.shellprop_from_LaminationParameters` gives the same
stiffnesses:

.. literalinclude:: ../../tests/test_shellprop.py
    :pyobject: test_make_orthotropic_LP_transverse_shear


Beam properties
---------------

:class:`.BeamProp` holds the cross-section integrals, in the element axes
`y` and `z`, of :class:`.BeamC`, :class:`.BeamLR` and :class:`.Truss`:
``A``, ``Iyy`` `= \int z^2 dA`, ``Izz`` `= \int y^2 dA`, ``Iyz``, the
torsion constant ``J``, ``Ay`` and ``Az``, which are null when the beam
axis passes through the centroid, the moduli ``E`` and ``G``, and the mass
integrals ``intrho``, ``intrhoy``, ``intrhoz``, ``intrhoy2``, ``intrhoz2``
and ``intrhoyz``. Bending in the element `xy` plane, i.e. the displacement
`v`, is governed by ``Izz`` and by the transverse shear modulus ``G12``,
bending in the `xz` plane by ``Iyy`` and ``G13``, and the torsion by the
torsional stiffness ``GJ``. These three attributes follow ``G`` until they
are set, ``G12 = G13 = G`` and ``GJ = G*J``, such that a property given
with ``G`` and ``J`` only keeps its meaning:

.. literalinclude:: ../../tests/test_beam_shear_moduli.py
    :pyobject: test_defaults_follow_G

No correction is applied by the elements: the shear correction factors of
the two planes, e.g. `5/6` for a rectangular section, are applied by the
user to ``G12`` and ``G13``. Applied to ``G``, as in some of the examples,
they also scale ``GJ``, unless ``GJ`` is set.

The elements read these attributes through the sectional stiffness
matrix ``C`` of :class:`.BeamProp`, of the strains `\{\bar{\varepsilon},
\bar{\gamma}_y, \bar{\gamma}_z, \kappa_x, \kappa_y, \kappa_z\}`,
which is built from them until it is set. Setting ``C`` gives the 21
independent entries of a general section, e.g. the extension-twist and
bending-shear couplings of a composite box beam, which the 11 stiffness
attributes cannot represent; the elements then read only ``C``, and the
mass integrals remain those of the attributes. The torsional and shear
stiffnesses and the couplings of an arbitrary, possibly composite, cross
section are computed by :mod:`pyfe3d.vabs`, and
:func:`.beamprop_from_vabs` sets ``C``, see
:doc:`cross_sectional_analysis`.


Examples
--------

The constitutive matrices of an isotropic plate, and of the same plate
with an offset:

.. literalinclude:: ../../tests/test_shellprop.py
    :pyobject: test_isotropic_plate

.. literalinclude:: ../../tests/test_shellprop.py
    :pyobject: test_offset_isotropic_plate

The shear correction of an isotropic plate, and the stiffnesses of a
cross-ply laminate given by Rohwer (1988):

.. literalinclude:: ../../tests/test_shellprop_transverse_shear.py
    :pyobject: test_isotropic_five_sixths

.. literalinclude:: ../../tests/test_shellprop_transverse_shear.py
    :pyobject: test_rohwer_cross_ply_reference_values

Rohwer, K., Improved transverse shear stiffness for layered finite elements,
DFVLR-FB 88-32, 1988.

For a sandwich, stiff faces on a soft core, the factor `5/6` is badly wrong:
``'rohwer'`` and ``'vlachoutsis'`` reproduce the factors tabulated by
Vlachoutsis (1992), Tables I and II, down to `K = 0.01964`, for which `5/6`
overestimates the shear stiffness 42 times, and, for an antiplane core, the
shear stiffness `G_c d^2/c` of Allen (1969), Eq. (2.15b), with `d` the
distance between the centroids of the faces, within the bounds given by the
local bending and the shear strains of the faces that Allen neglects. The
antiplane core, with zero moduli in the plane of the faces and finite
transverse shear moduli (Allen, 1969, p. 14), is given by ``laminaprop =
(0, 0, nu12, 0, G13, G23)``, or with nine entries ``(0, 0, nu12, 0, G13,
G23, E33, 0, 0)``, see :func:`.read_laminaprop`; it adds only transverse
shear stiffness:

.. literalinclude:: ../../tests/test_shellprop_transverse_shear.py
    :pyobject: test_sandwich_antiplane_core_converges_to_allen_1969

Rohwer (1988), Sec. 4.3, also gives the stiffnesses of a sandwich panel,
Table 4, p. 23, which are not used as a test: its transverse shear
stiffnesses are reproduced only with a foam shear modulus of 0.04 kN/mm\ :sup:`2`,
i.e. `\nu = 0.25` instead of the stated 0.3, and its bending stiffnesses
contradict the stated lay-up, since the ratio `(D_{11} - D_{22})/D_{16}`,
which depends on the positions of the layers alone, is 4.216 for the lay-up
of his Fig. 9 against 4.71 from the printed values. The details are in the
comments of ``tests/test_shellprop_transverse_shear.py``.

Allen, H. G., Analysis and Design of Structural Sandwich Panels, Pergamon
Press, 1969, Chap. 2,
`doi:10.1016/B978-0-08-012870-2.50006-7
<https://doi.org/10.1016/B978-0-08-012870-2.50006-7>`_.
