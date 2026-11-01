Natural frequency analysis of a cylinder
========================================

An isotropic circular cylindrical shell of :class:`.Quad4` elements,
supported at both ends by shear diaphragms, whose lowest frequencies are
compared with the three-dimensional elasticity solution tabulated by
Leissa, A. W., Vibration of Shells, NASA SP-288, 1973, Table 2.8. The
figure shows the first vibration mode, plotted with PyVista using
``refinement=2``:

.. image:: ./ex_natural_frequency_cylinder.jpg

The mass matrix is assembled with ``update_M`` and the frequencies are
computed with :func:`.natural_frequency`, see :doc:`eigenvalue_analyses`.
The docstring of the test gives the measured convergence to the tabulated
values:

.. literalinclude:: ../../tests/test_quad4_natural_freq_cylinder.py
    :pyobject: test_linear_nat_freq_cylinder
