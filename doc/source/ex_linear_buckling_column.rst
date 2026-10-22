Linear buckling analysis of a column
====================================

A clamped-free column of :class:`.BeamC` or :class:`.BeamLR` elements,
along an arbitrary direction in space, under a compressive load at its free
end. Each bending plane buckles at its own Euler load, `\pi^2 E I/(4
L^2)`. The geometric stiffness matrix is assembled with ``update_KG`` from
the linear static solution of a unit load, and the eigenvalue problem is
solved here with :func:`scipy.linalg.eigh`, see :doc:`eigenvalue_analyses`
for :func:`.linear_buckling`:

.. literalinclude:: ../../tests/test_beam_linear_buckling_3D_column.py
    :pyobject: buckling_loads

.. literalinclude:: ../../tests/test_beam_linear_buckling_3D_column.py
    :pyobject: test_column_buckles_at_the_euler_load
