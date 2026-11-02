General-purpose finite element solver - pyfe3d
==============================================

The ``pyfe3d`` module is a general-purpose finite element solver for structural
analysis and optimization based on Python and Cython. The main principles
guiding the development of ``pyfe3d`` are: simplicity, efficiency and
compatibility. The aimed level of compatibility allows one to run this solver
in any platform, including the Google Colab environment.

With ``pyfe3d`` you can build models of:

* Shells, with the quadrilaterals :class:`.Quad4` and :class:`.Quad4R` and
  the triangles :class:`.Tria3DSG` and :class:`.Tria3R`, and laminated or
  isotropic properties, :class:`.ShellProp`

* Beams, with the Timoshenko elements :class:`.BeamC` and :class:`.BeamLR`,
  trusses, :class:`.Truss`, and springs, :class:`.Spring`, with the
  sectional stiffness matrices of composite beams computed by the
  cross-sectional analysis of :mod:`pyfe3d.vabs`, see
  :doc:`cross_sectional_analysis`

all with six degrees of freedom per node, whose element matrices are written
into sparse arrays: the linear constitutive stiffness matrix, the
non-linear constitutive and geometric stiffness matrices, the mass matrix
and, for the quadrilaterals, the aerodynamic matrices of the piston theory.
With the solvers of :mod:`pyfe3d.solver` you can run:

* Linear static analyses, `\boldsymbol{K}_{C0} \boldsymbol{u} = \boldsymbol{f}`, with :func:`.solve` or
  :class:`.Analysis`

* Linear static analyses of unsupported structures with inertia relief,
  balancing the loads with the inertia forces of the rigid-body
  acceleration, with :func:`.apply_inertia_relief`, see :doc:`inertia_relief`

* Linear buckling analyses, `(\boldsymbol{K}_{C0} + \lambda \boldsymbol{K}_G) \boldsymbol{u} = \boldsymbol{0}`, with
  :func:`.linear_buckling`

* Natural frequency analyses, `(\boldsymbol{K}_{C0} - \omega_n^2 \boldsymbol{M}) \boldsymbol{u} = \boldsymbol{0}`,
  with :func:`.natural_frequency`, and complex frequencies of damped
  systems, `(\lambda^2 \boldsymbol{M} + \lambda \boldsymbol{C} + \boldsymbol{K}) \boldsymbol{u} = \boldsymbol{0}`, with
  :func:`.complex_frequency`

* Geometrically non-linear static analyses with :class:`.Analysis`, using
  the Newton-Raphson method or the arc-length methods of Riks and of
  Crisfield, with Crisfield's convergence criterion, configuration-dependent
  loads and prescribed displacements, and with large rotations through the
  element-independent co-rotational formulation of
  :mod:`pyfe3d.corotational`

The eigenvalue solvers verify their results and use a dense solver only as
a last resort, see :doc:`eigenvalue_analyses`.


Code repository
---------------

https://github.com/saullocastro/pyfe3d


User guide
----------

.. toctree::
    :maxdepth: 1

    overview.rst
    conventions.rst
    element_choice.rst
    shell_properties.rst
    cross_sectional_analysis.rst
    linear_analyses.rst
    inertia_relief.rst
    nonlinear_analyses.rst
    eigenvalue_analyses.rst


.. toctree::
    :maxdepth: 1
    :caption: Usage examples

    notebooks/ex_linear_static
    notebooks/ex_beam_frame
    notebooks/ex_linear_buckling_plate
    notebooks/ex_linear_buckling_column
    notebooks/ex_linear_buckling_cylinder_torsion
    notebooks/ex_natural_frequency_plate
    notebooks/ex_natural_frequency_cylinder
    notebooks/ex_nonlinear_static

.. toctree::
    :maxdepth: 1
    :caption: Tutorials

    notebooks/tut-opt-lightweight-smeared-deflection-GA
    notebooks/tut-opt-lightweight-smeared-deflection-SLSQP
    notebooks/tut-opt-topology
    notebooks/tut-timoshenko-beam-benchmarks
    notebooks/tut-shell-lumped-mass
    notebooks/tut-bench-macneal-harder
    notebooks/tut-bench-shell-obstacle-course
    notebooks/tut-nonlinear-benchmarks
    notebooks/tut-corotational-large-rotations
    notebooks/tut-prescribed-displacement-cantilever
    notebooks/tut-prescribed-displacement-dome-riks
    notebooks/tut-nafems-cap-axisymmetric
    notebooks/tut-inertia-relief
    notebooks/tut-inertia-relief-aircraft-ha144a


pyfe3d API
----------

.. toctree::
    :maxdepth: 2

    api.rst


Installing pyfe3d
-----------------

Install the distributed binaries by simply doing::

    python -m pip install pyfe3d

or, from the source code, with a C++ compiler::

    python -m pip install .

See :doc:`installation` for building in place, running the tests and
building this documentation, and :doc:`repository_structure` for the
layout of the repository.

.. toctree::
    :hidden:

    installation.rst
    repository_structure.rst
    changelog.rst


License
-------

.. literalinclude:: ../../LICENSE
    :encoding: latin-1

AUTHORS
-------

.. literalinclude:: ../../AUTHORS
    :encoding: latin-1


Citing this library
-------------------

Saullo G. P. Castro. (2026). General-purpose finite element solver based on Python and Cython (Version 0.12.0). Zenodo. DOI: https://doi.org/10.5281/zenodo.6573489.

The DOI represents all the versions and resolves to the latest one. The
repository also has a ``CITATION.cff`` file with the same data.


Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
