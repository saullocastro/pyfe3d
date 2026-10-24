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
  trusses, :class:`.Truss`, and springs, :class:`.Spring`

all with six degrees of freedom per node, whose element matrices are written
into sparse arrays: the linear constitutive stiffness matrix, the
non-linear constitutive and geometric stiffness matrices, the mass matrix
and, for the quadrilaterals, the aerodynamic matrices of the piston theory.
With the solvers of :mod:`pyfe3d.solver` you can run:

* Linear static analyses, `[K_{C0}]\{u\} = \{f\}`, with :func:`.solve` or
  :class:`.Analysis`

* Linear buckling analyses, `([K_{C0}] + \lambda [K_G])\{u\} = \{0\}`, with
  :func:`.linear_buckling`

* Natural frequency analyses, `([K_{C0}] - \omega_n^2 [M])\{u\} = \{0\}`,
  with :func:`.natural_frequency`, and complex frequencies of damped
  systems, `(\lambda^2 [M] + \lambda [C] + [K])\{u\} = \{0\}`, with
  :func:`.complex_frequency`

* Geometrically non-linear static analyses with :class:`.Analysis`, using
  the Newton-Raphson method or the arc-length methods of Riks and of
  Crisfield, with Crisfield's convergence criterion, configuration-dependent
  loads and prescribed displacements

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
    nonlinear_analyses.rst
    eigenvalue_analyses.rst


Usage examples
--------------

.. toctree::
    :maxdepth: 1

    ex_linear_static.rst
    ex_beam_frame.rst
    ex_linear_buckling_plate.rst
    ex_linear_buckling_column.rst
    ex_linear_buckling_cylinder_torsion.rst
    ex_natural_frequency_plate.rst
    ex_natural_frequency_cylinder.rst
    ex_nonlinear_static.rst


Tutorials
---------

.. toctree::
    :maxdepth: 2

    tutorials.rst


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


Changelog
---------

.. toctree::
    :maxdepth: 1

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

Saullo G. P. Castro. (2026). General-purpose finite element solver based on Python and Cython (Version 0.11.0). Zenodo. DOI: https://doi.org/10.5281/zenodo.6573489.

The DOI represents all the versions and resolves to the latest one. The
repository also has a ``CITATION.cff`` file with the same data.


Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
