Repository structure
--------------------

The ``pyfe3d`` repository provides a general-purpose finite element solver
implemented in Python and Cython. This overview describes the layout of the
repository and where each kind of resource lives.

Overall layout
==============

.. code-block:: text

    pyfe3d/
    ├── README.md
    ├── CITATION.cff
    ├── LICENSE
    ├── AUTHORS
    ├── pyfe3d/                 # library: elements and properties (Cython),
    │   └── solver/             #   solvers (Python)
    ├── tests/                  # tests, verifications against the literature
    ├── benchmarks/             # timing scripts of the element methods
    ├── doc/                    # Sphinx documentation
    ├── pyproject.toml          # package metadata, version, build requirements
    ├── setup.py                # Cython extensions (generates version.py)
    └── requirements*.txt

Key components
==============

* **README and citation** - an introduction to the solver and the citation
  information, also in ``CITATION.cff``.
* **Library modules (Cython)** - the element, property and co-rotational
  modules live in ``pyfe3d`` as ``.pyx`` files compiled with Cython, with
  the declaration files ``beamprop.pxd`` and ``shellprop.pxd``;
  ``setup.py`` collects these extensions and generates
  ``pyfe3d/version.py`` during the build, see :doc:`installation`.
  :mod:`pyfe3d.shellprop_utils` is pure Python.
* **Solvers (Python)** - the package :mod:`pyfe3d.solver`, with the linear,
  non-linear and eigenvalue solvers, see :doc:`nonlinear_analyses` and
  :doc:`eigenvalue_solvers`.
* **Tests** - the files ``tests/test_*.py``, run with pytest, verify the
  elements and the solvers against analytical solutions and published
  results, named in the docstring of each test; ``tests/data`` holds the
  matrices and meshes that some of them read, and the helper modules, e.g.
  ``tests/analytic_problems.py``, the analytic problems of the non-linear
  solvers.
* **Benchmarks** - the scripts of ``benchmarks/`` time the element methods,
  to assess the compiler flags and the cost of the constitutive matrices;
  they are not run by the tests.
* **Documentation** - the reStructuredText pages of ``doc/source``, built
  with Sphinx: the theory of the elements, of the analyses and of the
  solvers, and the API reference rendered from the docstrings, see
  :doc:`api`.
* **Requirements** - ``requirements.txt`` lists the run-time and build
  dependencies, ``requirements_ci.txt`` and ``requirements_doc.txt`` the
  packages of the tests and of the documentation, see :doc:`installation`.

Where to start
==============

1. **Build the library** in place, see :doc:`installation`.
2. **Read the overview** of the objects of a model and of the steps of an
   analysis, :doc:`overview`, and the conventions, :doc:`conventions`.
3. **Choose the elements**, see :doc:`element_choice`, and the analysis,
   see the chapters on the analyses and the solvers listed in
   :doc:`overview`.
4. **Read the tests** of the elements and analyses of interest: each one
   builds a complete model and states the reference it is verified against.
