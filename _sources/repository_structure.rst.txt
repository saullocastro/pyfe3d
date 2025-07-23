Repository structure
--------------------

The ``pyfe3d`` repository provides a general-purpose finite element solver implemented
in Python and Cython.  The following overview describes the main layout of the
project and points newcomers to helpful resources.

Overall layout
==============

.. code-block:: text

    pyfe3d/
    ├── README.md
    ├── CITATION.cff
    ├── LICENSE
    ├── pyfe3d/                 # library code (Cython)
    ├── doc/                    # Sphinx documentation
    ├── tests/                  # test cases and usage examples
    ├── setup.py                # build script (generates version.py)
    ├── requirements*.txt

Key components
==============

* **README and Citation** – provide an introduction to the solver and citation
  information.
* **Library modules (Cython)** – element and property classes live in ``pyfe3d``
  as ``.pyx`` files that are compiled with Cython.  ``setup.py`` collects these
  extensions and generates ``pyfe3d/version.py`` during the build.
* **Tests as examples** – scripts in ``tests/`` demonstrate how to build models
  and call the solver.  They act as both regression tests and usage examples.
* **Documentation** – reStructuredText files under ``doc/`` are built with
  Sphinx.  A topology optimisation tutorial is provided as a Jupyter notebook.
* **Requirements** – ``requirements.txt`` lists runtime dependencies while other
  files specify packages for CI and documentation builds.

Suggested next steps
====================

1. **Build the library** using ``python setup.py build_ext --inplace`` to
   compile the Cython extensions.
2. **Explore the tests** in ``tests/`` as practical examples of typical
   analyses.
3. **Browse the documentation** in ``doc/`` for tutorials and API references.
4. **Create your own models** by adapting the test scripts.  Start with simple
   elements such as ``Quad4`` or ``BeamC``.
5. **Check the citation** file if you use ``pyfe3d`` in research.
