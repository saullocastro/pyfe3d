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
`3 \times 3` transformation matrix `\boldsymbol{T}_{3\times 3}` of each element, whose
entries `T_{ij}` are stored in the attributes ``r11, ..., r33`` (`T_{12}` in
``r12`` etc.). Its columns are the global components of the element axes:

.. math::

    \boldsymbol{T}_{3\times 3} = \begin{bmatrix}
        T_{11} & T_{12} & T_{13} \\
        T_{21} & T_{22} & T_{23} \\
        T_{31} & T_{32} & T_{33}
    \end{bmatrix}
    = \begin{bmatrix}
        x_X & y_X & z_X \\
        x_Y & y_Y & z_Y \\
        x_Z & y_Z & z_Z
    \end{bmatrix}, \qquad
    \begin{Bmatrix} u \\ v \\ w \end{Bmatrix}_{global}
    = \boldsymbol{T}_{3\times 3} \begin{Bmatrix} u \\ v \\ w \end{Bmatrix}_{element}

such that the translations and the rotations of every node are brought to
the global system with `\boldsymbol{T}_{3\times 3}`, and to the element system with
`\boldsymbol{T}_{3\times 3}^\top`. The transformation matrix `\boldsymbol{T}` of the element is block
diagonal, with one block `\boldsymbol{T}_{3\times 3}` for the translations and one for
the rotations of each node, e.g. `12 \times 12` for the two-node elements,
such that

.. math::

    \boldsymbol{u} = \boldsymbol{T} \boldsymbol{u}_e, \qquad \boldsymbol{u}_e = \boldsymbol{T}^\top \boldsymbol{u}, \qquad
    \boldsymbol{K} = \boldsymbol{T} \boldsymbol{K}_e \boldsymbol{T}^\top, \qquad \boldsymbol{F} = \boldsymbol{T} \boldsymbol{F}_e

with `\boldsymbol{u}`, `\boldsymbol{K}` and `\boldsymbol{F}` the element dofs, matrices and forces in
global coordinates, and `\boldsymbol{u}_e`, `\boldsymbol{K}_e` and `\boldsymbol{F}_e` in element
coordinates. The element axes are defined by ``update_rotation_matrix``:

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
- The residual of :class:`.Analysis` is `\boldsymbol{R} = \lambda \boldsymbol{F}_{ext} -
  \boldsymbol{F}_{int}`, with the tangent stiffness matrix `\boldsymbol{K}_T = -\partial
  \boldsymbol{R}/\partial \boldsymbol{u}`, see :doc:`nonlinear_analyses`.
- The linear buckling problem is `(\boldsymbol{K}_{C0} + \lambda \boldsymbol{K}_G) \boldsymbol{u} = \boldsymbol{0}`
  and the frequency problem `(\boldsymbol{K}_{C0} - \omega_n^2 \boldsymbol{M}) \boldsymbol{u} = \boldsymbol{0}`, see
  :doc:`eigenvalue_analyses`.


Units
-----

``pyfe3d`` has no units. The coordinates, the properties and the loads
must use a consistent system of units, e.g. N, m, kg and s, or N, mm, t and
s. The natural frequencies of :func:`.natural_frequency` are then circular
frequencies, in rad/s, and the rotations are in radians. The dimension of
each argument is given in the API documentation.
