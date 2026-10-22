Linear static analysis of beams and frames
==========================================

An L-shaped frame of :class:`.BeamC` elements, clamped at the base of the
column and loaded at the free end of the beam, in its plane and out of its
plane. The orientation vector of ``update_rotation_matrix`` puts the
element `z` axis of both members along the global `Z` axis, such that the
in-plane bending is governed by ``Izz`` and the out-of-plane bending by
``Iyy``, see :doc:`conventions`. The displacements of the free end, with
the bending, shear, axial and torsional deformations, are exact with one
element per member, and the internal forces at the base are the support
reactions:

.. literalinclude:: ../../tests/test_beamc_frame.py
    :pyobject: frame

.. literalinclude:: ../../tests/test_beamc_frame.py
    :pyobject: test_l_frame_tip_displacements

The tip deflection of a cantilever with shear deformation, Eq. (1.2) of
Bathe (2014), *Finite Element Procedures*, 2nd ed., exact with one
:class:`.BeamC` element, whereas :class:`.BeamLR` converges with mesh
refinement:

.. literalinclude:: ../../tests/test_timoshenko_beam_literature.py
    :pyobject: cantilever_tip

.. literalinclude:: ../../tests/test_timoshenko_beam_literature.py
    :pyobject: test_bathe_cantilever_shear_deflection
