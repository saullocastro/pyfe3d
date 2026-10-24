Natural frequency analysis of a plate
=====================================

A simply supported plate of :class:`.Quad4` elements, whose lowest natural
frequency is compared with the analytical one, with the matrices of
``plate_matrices``, see :doc:`ex_linear_buckling_plate`. The mass matrix is
assembled with ``update_M`` and the frequencies are computed with
:func:`.natural_frequency`, with and without the equilibration of the
diagonal:

.. literalinclude:: ../../tests/test_solver.py
    :pyobject: test_natural_frequency_plate

The consistent, reduced-integration and lumped mass matrices of
:class:`.Quad4`, selected with the argument ``mtype`` of ``update_M``:

.. literalinclude:: ../../tests/test_quad4_natural_freq.py
    :pyobject: test_nat_freq_plate
