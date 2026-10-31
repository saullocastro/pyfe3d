Installation and building
=========================

Installing
----------

Install the distributed wheels with::

    python -m pip install pyfe3d

or, from a clone of the `repository <https://github.com/saullocastro/pyfe3d>`_,
compiling the Cython modules::

    python -m pip install .

The run-time dependencies are NumPy and SciPy; Cython and a C++ compiler
are needed to build from the source code, see ``requirements.txt`` and
``pyproject.toml``, which also holds the package metadata and the version. When no compiler is found, this page gives some
information on how to set up one on different operating systems:
https://cython2.readthedocs.io/en/latest/src/quickstart/install.html


Building in place
-----------------

For development, the extensions are compiled next to their sources with::

    python -m pip install -r requirements.txt
    python setup.py build_ext --inplace

which also writes ``pyfe3d/version.py`` with the version of
``pyproject.toml``. Each element and property, and the co-rotational
layer :mod:`pyfe3d.corotational`, is one extension module,
``pyfe3d/*.pyx``, compiled as C++ with the optimization
flags of ``setup.py``, ``/O2`` with MSVC and ``-O3 -fno-math-errno`` with
GCC and Clang, without flags that change the floating-point results.
:mod:`pyfe3d.solver` and :mod:`pyfe3d.shellprop_utils` are pure Python.

The property classes have Cython declaration files, ``beamprop.pxd`` and
``shellprop.pxd``, distributed with the package, such that other Cython
modules can ``cimport`` them.


Running the tests
-----------------

The tests are in ``tests/`` and run with pytest, as in the continuous
integration of the repository::

    python -m pip install -r requirements_ci.txt
    python -m pytest tests

Several tests can also be run as scripts, plotting their results, e.g.
``python test_quad4_static_point_load.py``.

The tests marked ``slow``, ``@pytest.mark.slow``, registered in
``pyproject.toml``, are those that are slow under the line tracing of the
Cython code and cover no line of the ``.pyx`` files that the other tests
do not cover. Every test runs by default; the coverage workflow,
``.github/workflows/coverage.yml``, deselects them with ``-m "not slow"``
in its run with the line-traced build only, and runs them in its run with
the optimized build, which measures the Python code. A new test that is
slow under line tracing can be marked once the coverage of the other tests,
measured with ``pytest --cov --cov-context=test``, is shown to include
every line of the ``.pyx`` files that it covers.


Building the documentation
--------------------------

The documentation is built with Sphinx, with the packages of
``requirements_doc.txt``::

    python -m pip install -r requirements_doc.txt
    python -m sphinx -b html doc/source doc/build/html
