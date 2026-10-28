Linear buckling analysis of a cylinder under torsion
====================================================

The cylinders Z11 and Z33 of Castro et al. (2014),
`doi:10.1016/j.compstruct.2014.07.037
<https://doi.org/10.1016/j.compstruct.2014.07.037>`_, two unsymmetric
laminates of :class:`.Quad4` elements, under torsion, with the boundary
conditions CC2 of that paper. The figure shows the first buckling mode of
Z33, plotted with PyVista using ``refinement=2``:

.. image:: ./ex_linear_buckling_cylinder_torsion.jpg

The laminate of each element is oriented by a material direction along the
axis of the cylinder, see :doc:`conventions`. The torques are compared
with the pure-torsion intercepts of Fig. 13 of the paper, to which the
element converges within 0.7 %; the docstring of the test gives the
measured convergence, and a second test compares an isotropic cylinder with
the classical torsional buckling stress of Batdorf, Stein and Schildcrout,
NACA TN 1344, 1947:

.. literalinclude:: ../../tests/test_quad4_linear_buckling_cylinder_Nxy.py
    :pyobject: test_linear_buckling_cylinder_Nxy
