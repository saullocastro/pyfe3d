Overview and workflow
=====================

``pyfe3d`` gives finite elements, not a pre-processor: the model is built
in Python, with NumPy arrays for the nodes and the boundary conditions,
and each element writes its contributions into sparse arrays that SciPy
turns into the structural matrices. The same arrays serve linear static,
linear buckling, natural frequency and geometrically non-linear analyses,
solved with :mod:`pyfe3d.solver` or with any SciPy solver.


The pieces of a model
---------------------

.. list-table::
    :header-rows: 1

    * - Object
      - Role
    * - nodal coordinates
      - 1-D array ``x`` of size ``3*M``, `X_1, Y_1, Z_1, X_2, ...`, see
        :doc:`conventions`
    * - property
      - :class:`.ShellProp` for the shells, created by
        :func:`.isotropic_plate` or :func:`.laminated_plate`, see
        :doc:`shell_properties`; :class:`.BeamProp` for the beams and the
        truss; the stiffnesses of a :class:`.Spring` are attributes of the
        element
    * - ``*Data``
      - e.g. :class:`.Quad4Data`: the number of sparse entries that one
        element writes for each matrix, ``KC0_SPARSE_SIZE``,
        ``KCNL_SPARSE_SIZE``, ``KG_SPARSE_SIZE``, ``M_SPARSE_SIZE``, ...
    * - ``*Probe``
      - e.g. :class:`.Quad4Probe`: the buffers of one element in element
        coordinates, ``xe`` (coordinates), ``ue`` (displacements), ``finte``
        (internal forces), shared by all the elements of a type and
        overwritten by each ``update_probe_*`` call
    * - element
      - e.g. :class:`.Quad4`: the connectivity ``n1, n2, ...``, the
        positions ``c1, c2, ...`` of its nodes in the global vectors, the
        positions ``init_k_KC0, init_k_KG, ...`` of its entries in the sparse
        arrays, the rotation matrix ``r11, ..., r33``, and the ``update_*``
        methods

The elements are Cython extension types with compiled ``update_*``
methods, such that a loop over the elements in Python costs one call per
element and matrix.


Assembly of the structural matrices
-----------------------------------

The structural matrices are assembled in the coordinate (COO) format: one
array of row indices, one of column indices and one of values, each with
``*_SPARSE_SIZE`` entries per element, of the types ``pyfe3d.INT`` and
``pyfe3d.DOUBLE``. An element writes its entries at the positions
``init_k_KC0`` to ``init_k_KC0 + KC0_SPARSE_SIZE - 1``, so consecutive
elements get ``init_k_KC0`` incremented by ``KC0_SPARSE_SIZE``. The entries
of different elements at the same row and column are summed when the
arrays are converted with :class:`scipy.sparse.coo_matrix`, which is the
assembly

.. math::

    [K] = \sum_e [L_e]^T [T_e] [K_e] [T_e]^T [L_e]

with `[K_e]` the element matrix in element coordinates, `[T_e]` the
transformation matrix of the element, block diagonal with the matrix
`[T_{3\times 3}]` of the element, whose entries are ``r11, ..., r33``, one
block per node and per translations or rotations, see :doc:`conventions`, and
`[L_e]` the Boolean matrix that extracts the dofs of the element, given by
``c1, c2, ...``. The internal force vector is assembled in place, by
``update_fint``, into a dense vector of size ``N``.

.. list-table:: Entries written per element (``*Data`` attributes)
    :header-rows: 1

    * - Element
      - ``KC0``
      - ``KCNL``
      - ``KG``
      - ``M``
      - Others
    * - :class:`.Quad4`, :class:`.Quad4R`
      - 576
      - 576
      - 144
      - 480
      - ``KA_BETA``, ``KA_GAMMA``, ``CA``: 144
    * - :class:`.Tria3DSG`
      - 324
      - 324
      - 81
      - 324
      -
    * - :class:`.Tria3R`
      - 324
      - 324
      - 81
      - 270
      -
    * - :class:`.BeamC`
      - 144
      - 144
      - 144
      - 144
      -
    * - :class:`.BeamLR`
      - 144
      - 144
      - 36
      - 144
      -
    * - :class:`.Truss`
      - 72
      -
      -
      - 144
      -
    * - :class:`.Spring`
      - 72
      -
      - 0
      - 0
      -

The matrices are:

- ``KC0``, the linear constitutive stiffness matrix `[K_{C0}]`;
- ``KCNL``, the non-linear part of the constitutive stiffness matrix,
  `[K_{CNL}(u)]`, such that `[K_{C0}] + [K_{CNL}] + [K_G]` is the tangent
  stiffness matrix of the geometrically non-linear analyses;
- ``KG``, the geometric stiffness matrix `[K_G(u)]`, from the stress
  resultants of the displacements in the probe, or, for the shells, of
  given membrane stress resultants with ``update_KG_given_stress``;
- ``M``, the mass matrix, consistent, lumped or, for some shells, with
  reduced integration, ``mtype`` 0, 1 or 2 of ``update_M`` for every
  element, see :doc:`element_choice`;
- ``KA_beta``, ``KA_gamma`` and ``CA``, the aerodynamic stiffness and
  damping matrices of the piston theory, for the quadrilaterals.

The methods of each element are listed in :doc:`element_choice`.


Steps of an analysis
--------------------

The tests of the repository, e.g. :doc:`ex_linear_static`, follow these
steps:

1. Nodes: the array ``x`` of the coordinates and a map from the node
   identification numbers to their positions ``pos``.
2. Properties: :func:`.isotropic_plate`, :func:`.laminated_plate` or a
   :class:`.BeamProp`.
3. Elements: one probe and one ``*Data`` per element type; for each
   element, set ``n1, n2, ...``, ``c1 = DOF*pos1, ...`` and the
   ``init_k_*`` of the matrices needed, then call, in this order,
   ``update_rotation_matrix``, ``update_probe_xe`` and the ``update_K*``
   or ``update_M`` methods. The order matters: each method reads the
   attributes updated by the previous ones, see the class documentation
   of each element.
4. Assembly: ``KC0 = coo_matrix((KC0v, (KC0r, KC0c)), shape=(N, N)).tocsc()``.
5. Boundary conditions: a Boolean array ``bk`` of the constrained dofs,
   e.g. ``bk[2::DOF] = edge`` for `w = 0` at the nodes of an edge, and
   ``bu = ~bk`` for the free ones. The matrices are partitioned,
   ``KC0uu = KC0[bu, :][:, bu]``, and the solution is expanded,
   ``u[bu] = uu``. With :class:`.Analysis`, ``bu`` is passed instead and the
   partition is done by the solver, see :doc:`nonlinear_analyses`. The
   solvers of :mod:`pyfe3d.solver` do not remove null rows and columns: a
   dof left without stiffness, e.g. an unattached node, an unstiffened
   drilling rotation or a missing boundary condition, raises a
   :class:`numpy.linalg.LinAlgError` naming the dofs, see
   :func:`.check_null_rows_cols`.
6. Solution: the linear static system `[K_{C0}]_{uu}\{u\}_u = \{f\}_u`, with
   :func:`.solve` or :func:`scipy.sparse.linalg.spsolve`; the eigenvalue
   problems with :func:`.linear_buckling`, :func:`.natural_frequency` and
   :func:`.complex_frequency`,
   see :doc:`eigenvalue_analyses`; the non-linear analyses with
   :class:`.Analysis`.
7. Post-processing: ``update_probe_ue(u)`` brings the displacements of an
   element to element coordinates, in ``probe.ue``; ``update_fint`` adds
   its internal forces, in global coordinates, to a global vector, whose
   values at the constrained dofs are the reactions, `\{F_{int}\}_k -
   \{f\}_k`; and the strains of :class:`.Quad4` are recovered with
   :meth:`.Quad4Probe.update_BL`.

Since the probe is shared, ``update_probe_xe`` and ``update_probe_ue``
must be called for an element right before the methods that read the
probe, e.g. ``update_KG``, ``update_KCNL`` and ``update_fint``, as in the
loops of the examples.


Linear and non-linear problems
------------------------------

The linear static problem, the eigenvalue problems and the non-linear
static problem use the same element matrices:

.. math::

    [K_{C0}]\{u\} = \{f\}

.. math::

    ([K_{C0}] + \lambda [K_G(u_0)])\{u\} = \{0\}, \qquad
    ([K_{C0}] - \omega_n^2 [M])\{u\} = \{0\}

.. math::

    \{R(u, \lambda)\} = \lambda \{F_{ext}\} - \{F_{int}(u)\} = \{0\},
    \qquad [K_T] = -\frac{\partial R}{\partial u} = [K_{C0}] + [K_{CNL}(u)]
    + [K_G(u)]

where `\{u_0\}` is the linear static solution of the reference load of the
buckling analysis and `\{F_{int}(u)\}` is the vector of ``update_fint(...,
nonlinear=1)``, whose exact derivative is `[K_T]` for every element with
``update_KCNL`` (verified in ``tests/test_tangent_consistency.py``). The
linear internal force vector, ``nonlinear=0``, is `[K_{C0}]\{u\}`.
