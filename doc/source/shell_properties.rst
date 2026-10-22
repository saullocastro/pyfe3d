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
    \begin{Bmatrix} Q_y \\ Q_x \end{Bmatrix} = [A_{ts}]
    \begin{Bmatrix} \gamma_{yz} \\ \gamma_{xz} \end{Bmatrix}

with `A_{ij}`, `B_{ij}`, `D_{ij}` the thickness integrals of the ply
stiffnesses times `1`, `z` and `z^2`, and `[A_{ts}]` = ``[[A44, A45],
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

    [B] = e [A], \qquad [D] = [D]_{e=0} + e^2 [A], \qquad
    \int \rho z \, dz = \rho h e, \qquad
    \int \rho z^2 \, dz = \rho \left(\frac{h^3}{12} + h e^2\right)

such that an offset couples the membrane and bending responses. It
models e.g. a skin whose nodes are on its inner surface, shared with the
beams of a stiffener. The mass integrals ``intrho``, ``intrhoz`` and
``intrhoz2`` enter the mass matrices of the shells.


Transverse shear correction
---------------------------

The stiffnesses ``A44``, ``A45`` and ``A55`` of a :class:`.ShellProp` are
the ones used by the elements, with the shear correction already applied,
and the elements apply no other correction factor. The correction is
selected with ``shear_correction``, see
:meth:`.ShellProp.calc_transverse_shear_stiffness`:

- ``'rohwer'``, the default: the equilibrium approach of Rohwer (1988),
  giving the full matrix `[A_{ts}]`, re-evaluated in the element axes by
  :meth:`.ShellProp.calc_Ats_element` when the material direction differs
  from the element `x` axis;
- ``'vlachoutsis'``: the two factors of Vlachoutsis (1992),
  `doi:10.1002/nme.1620330712 <https://doi.org/10.1002/nme.1620330712>`_;
- ``'constant'``: `k = 5/6`;
- ``None``: no correction, `[A_{ts}] = \sum_k [C_s^{(k)}] h_k`.

For an isotropic plate the first three give `5/6 \, G h`. The uncorrected
stiffnesses ``Abar44``, ``Abar45``, ``Abar55`` and the constant-stress ones
``Abarbar44``, ``Abarbar45``, ``Abarbar55`` are kept for comparison, and
``scf_k13``, ``scf_k23`` are the resulting ratios, informative only. A
:class:`.ShellProp` created from lamination parameters has no plies and no
correction; ``A44``, ``A45``, ``A55`` can then be set directly.


Beam properties
---------------

:class:`.BeamProp` holds the cross-section integrals, in the element axes
`y` and `z`, of :class:`.BeamC`, :class:`.BeamLR` and :class:`.Truss`:
``A``, ``Iyy`` `= \int z^2 dA`, ``Izz`` `= \int y^2 dA`, ``Iyz``, the
torsion constant ``J``, ``Ay`` and ``Az``, which are null when the beam
axis passes through the centroid, the moduli ``E`` and ``G``, and the mass
integrals ``intrho``, ``intrhoy``, ``intrhoz``, ``intrhoy2``, ``intrhoz2``
and ``intrhoyz``. No correction is applied by the elements: the shear
correction factor, e.g. `5/6` for a rectangular section, is applied by the
user to ``G``, as in the examples. Bending in the element `xy` plane, i.e.
the displacement `v`, is governed by ``Izz`` and bending in the `xz` plane
by ``Iyy``.


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
