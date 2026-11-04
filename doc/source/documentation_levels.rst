How the documentation is organised
==================================

The documentation of ``pyfe3d`` is written in four cumulative levels, each
with its own place, so that a reader, or a tool, can go from what a
function does to why it does it without the levels repeating each other.

.. list-table::
    :header-rows: 1

    * - Level
      - Where
      - Content
    * - 1, functional
      - docstrings of the modules, classes, functions and methods, except
        their ``Notes`` and ``References`` sections; comments in the code;
        docstrings and comments of the tests, except their ``Notes`` and
        ``References``; the ``README.md``
      - what each resource does: a one-line summary, an extended summary
        (what, not why), ``Parameters`` with type, unit, shape and
        default, ``Returns``, ``Raises`` and ``See Also``
    * - 2, theory
      - the ``Notes`` and ``References`` sections of the docstrings of the
        library and of the tests, used only for theory
      - the theory that each resource implements, local to it: the
        governing equations in the notation of the code, the assumptions
        and limits of validity, the meaning of each parameter in that
        context, the algorithm, and the published sources
    * - 3, theory in a larger context
      - the pages of this documentation, under "User guide" and "Analyses
        and solvers"
      - the theory written like a book: how the pieces of level 2 fit
        together, the workflow of an analysis, the conventions shared by
        the modules, the choice of element, with links to the API instead
        of repeating level 2
    * - 4, worked examples
      - the Jupyter notebooks in ``notebooks/`` at the root of the
        repository, rendered on this site under "Worked examples" and
        "Tutorials"
      - complete analyses, mostly benchmarks reproduced from the
        literature, with stored outputs and the results compared with
        their reference values

The API pages render levels 1 and 2 together. To find the theory of a
resource, read the ``Notes`` of its docstring; to find how a method fits in
an analysis, read the pages of level 3, starting from :doc:`overview`.


The tests
---------

The docstrings and comments of the tests in ``tests/`` are part of level 1,
their ``Notes`` and ``References`` of level 2; the tests are also usage
examples, see :doc:`repository_structure`. Each test states in its
docstring what it verifies and against what:

* a published value, a table or an equation of the literature, cited with
  its location, e.g. the buckling load of a simply supported plate of
  Leissa (1969), Eq. (10.46);
* a closed-form solution written in the test, with its source;
* a property that holds for any input, e.g. the invariance of a buckling
  load under a rigid rotation of the model, the rank of an element matrix,
  or a patch test.

When a finite element result is compared with the literature, the
tolerance is that of the discretisation, measured by refining the mesh
and stated in the docstring, or one of the grades of MacNeal and Harder
(1985), and not a value fitted to the output of the code. When the
literature is matched only on a mesh too expensive for a test, the
docstring records the runs of the same test with a larger ``refinement``
that match it, and the test compares its coarse mesh with the value that
the same test gave on that mesh, with a tolerance for round-off only,
keeping a check against the literature within a grade.
