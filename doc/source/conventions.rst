Conventions
===========

Degrees of freedom
------------------

Every node has ``DOF = 6`` degrees of freedom (dofs), in the order

.. math::

    u, \; v, \; w, \; r_x, \; r_y, \; r_z

the three translations and the three rotations about the axes `X`, `Y`,
`Z` of the global coordinate system. The vectors and matrices of a model
with `M` nodes have size ``N = DOF*M``, and the dofs of the node at position
``pos`` of the model are ``DOF*pos`` to ``DOF*pos + 5``. The elements read
this position from their attributes ``c1``, ``c2``, ... (e.g. ``quad.c1 =
DOF*pos1``), and slices with step ``DOF`` select one dof of all the nodes,
e.g. ``u[2::DOF]`` for `w`. The nodal coordinates are in a separate 1-D
array of size ``3*M``, `X_1, Y_1, Z_1, X_2, ...`, where the elements read
the coordinates of a node at the position ``c1//2``, i.e. ``3*pos``.

The integer arrays of the sparse row and column indices must have the
type ``pyfe3d.INT``, the NumPy type of the C ``long`` of the platform, and
the arrays of values the type ``pyfe3d.DOUBLE``, see :mod:`pyfe3d`.


Coordinate systems
------------------

The element matrices are computed in an element coordinate system `x`,
`y`, `z` and rotated to the global one by the ``update_*`` methods, with the
rotation matrix ``r11, ..., r33`` of each element. Its columns are the
global components of the element axes:

.. math::

    [R] = \begin{bmatrix}
        r_{11} & r_{12} & r_{13} \\
        r_{21} & r_{22} & r_{23} \\
        r_{31} & r_{32} & r_{33}
    \end{bmatrix}
    = \begin{bmatrix}
        x_X & y_X & z_X \\
        x_Y & y_Y & z_Y \\
        x_Z & y_Z & z_Z
    \end{bmatrix}, \qquad
    \begin{Bmatrix} u \\ v \\ w \end{Bmatrix}_{global}
    = [R] \begin{Bmatrix} u \\ v \\ w \end{Bmatrix}_{element}

such that the translations and the rotations of every node transform with
`[R]`, and `[R]^T` brings the global quantities to the element system. The
element axes are defined by ``update_rotation_matrix``:

.. list-table::
    :header-rows: 1

    * - Element
      - `x` axis
      - `z` axis
    * - :class:`.BeamC`, :class:`.BeamLR`
      - from node 1 to node 2
      - `x \times v_{xy}`, with the vector ``vxyi, vxyj, vxyk`` in the `xy`
        plane of the element, which must not be parallel to `x`
    * - :class:`.Spring`
      - the vector ``xi, xj, xk``
      - `x \times v_{xy}`, as for the beams; the identity matrix by default
    * - :class:`.Truss`
      - from node 1 to node 2
      - normal to `x` and to the cyclic permutation of its components
    * - :class:`.Quad4`, :class:`.Quad4R`
      - along the sum of the diagonals `r_2 - r_4` and `r_3 - r_1`
      - along `(r_2 - r_4) \times (r_3 - r_1)`, with `r_i` the position of
        node `i`
    * - :class:`.Tria3R`, :class:`.Tria3DSG`
      - from node 1 to node 2
      - the normal `(r_2 - r_1) \times (r_3 - r_1)`

and `y = z \times x` completes the right-handed system. The normal of a
shell element, hence the side towards which its ``offset`` is positive,
follows the node numbering: counterclockwise when
seen from the tip of `z`, see the figures of :class:`.Quad4` and
:class:`.Tria3R`. The meshes of the examples number the nodes such that the
normals of all the elements point to the same side.

The laminate of a shell element is oriented by a material direction, the
vector ``xmati, xmatj, xmatk`` of ``update_rotation_matrix``, projected onto
the element plane; the angles of the plies are measured from it. The
default, the null vector, uses the element `x` axis, which changes from
element to element in a distorted or curved mesh. The rotation from the
material direction to the element axes is stored in ``m11, m12, m21,
m22``, see :doc:`shell_properties`.


Sign conventions
----------------

- The rotations follow the right-hand rule about the axes. In the element
  system of the shells, the displacements at a distance `z` from the
  reference surface are `u + z r_y` and `v - z r_x`, see :mod:`.quad4`,
  and in the element system of the beams the displacements of the point
  `(y, z)` of the cross section are `u - y r_z + z r_y`, `v - z r_x` and `w
  + y r_x`, see :class:`.BeamC`.
- Forces and moments are positive along the positive axes, the external
  force vector and the internal force vector of ``update_fint`` alike. A
  pressure is a set of nodal forces given by the user.
- The residual of :class:`.Analysis` is `\{R\} = \lambda \{F_{ext}\} -
  \{F_{int}\}`, with the tangent stiffness matrix `[K_T] = -\partial
  R/\partial u`, see :doc:`nonlinear_analyses`.
- The linear buckling problem is `([K_{C0}] + \lambda [K_G])\{u\} = \{0\}`
  and the frequency problem `([K_{C0}] - \omega_n^2 [M])\{u\} = \{0\}`, see
  :doc:`eigenvalue_analyses`.


Units
-----

``pyfe3d`` has no units. The coordinates, the properties and the loads
must use a consistent system of units, e.g. N, m, kg and s, or N, mm, t and
s. The natural frequencies of :func:`.natural_frequency` are then circular
frequencies, in rad/s, and the rotations are in radians. The dimension of
each argument is given in the API documentation.
