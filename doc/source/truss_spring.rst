Truss and spring elements
=========================

Besides the beams of :doc:`beam_theory`, pyfe3d has two simpler two-node
elements with the six degrees of freedom per node of every element:
:class:`.Truss`, a bar with axial and torsional stiffness, and
:class:`.Spring`, six uncoupled springs between two nodes. Both share the
nodes of the other elements and the transformation `\boldsymbol{K} =
\boldsymbol{T} \boldsymbol{K}_e \boldsymbol{T}^\top`, `\boldsymbol{F} =
\boldsymbol{T} \boldsymbol{F}_e` of :doc:`conventions`.


Truss
-----

**Kinematics and energy.** The element `x` axis goes from node 1 to node 2;
the `z` axis is normal to `x` and to the cyclic permutation of its
components, and `y` completes the right-handed system, see
:meth:`.Truss.update_rotation_matrix`. The axial displacement `u` and the
twist `r_x` are interpolated linearly, and the strain energy is

.. math::
    :label: truss-energy

    U = \frac{1}{2} \int_0^L \left( E A \, u_{,x}^2 + G J \, {r_x}_{,x}^2
    \right) dx

with `E A` and `G J` the entries ``C[0, 0]`` and ``C[3, 3]`` of the
sectional stiffness matrix of the :class:`.BeamProp`, i.e. ``E*A`` and
``GJ`` when ``C`` is not set; the other entries, couplings included, are not
read. In element coordinates the stiffness matrix has the entries `\pm E
A/L` between the axial translations and `\pm G J/L` between the twist
rotations of the two nodes, and the internal forces are, e.g., `E A (u_2 -
u_1)/L` at node 2, see :meth:`.Truss.update_KC0` and
:meth:`.Truss.update_probe_finte`. The transverse translations `v`, `w` and
the rotations `r_y`, `r_z` have no stiffness: a node connected only to
trusses must have them supported, by other elements or by boundary
conditions.

**Mass.** The rigid cross section has the displacements `u_x = u`, `u_y = v
- z r_x` and `u_z = w + y r_x`, with the three translations and `r_x`
linear, whose kinetic energy is

.. math::
    :label: truss-kinetic

    T = \frac{1}{2} \int_0^L \Big[ m (\dot{u}^2 + \dot{v}^2 + \dot{w}^2)
    - 2 m_z \dot{v} \dot{r}_x + 2 m_y \dot{w} \dot{r}_x
    + (m_{yy} + m_{zz}) \dot{r}_x^2 \Big] dx

with the mass integrals of :doc:`beam_properties`. The consistent mass
matrix, ``mtype=0`` of :meth:`.Truss.update_M`, is its exact Hessian, `L
m/6` times `[2, 1; 1, 2]` for each translation; the lumped one, ``mtype=1``,
the two-point Gauss-Lobatto quadrature, diagonal, without the couplings of
`m_y` and `m_z`. The rotations `r_y` and `r_z` have no inertia.

**Limits.** The element has no geometric stiffness matrix, hence no linear
buckling analysis, and its internal forces are linear; for large
displacements it is wrapped by :class:`.CorotationalTruss`, which gives the
co-rotational truss of Crisfield (1991), see
:doc:`corotational_formulation`. The module :mod:`pyfe3d.truss` recommends
:class:`.BeamC` for its better physical representation; the beam also has
geometric stiffness and non-linear internal forces.


Spring
------

**Energy.** :class:`.Spring` connects the same degree of freedom of two
nodes with the six stiffnesses ``kxe``, ``kye``, ``kze``, ``krxe``,
``krye``, ``krze``, `k_1, \ldots, k_6`, in the element coordinate system:

.. math::
    :label: spring-energy

    U = \frac{1}{2} \sum_{i=1}^{6} k_i \left( d_{2,i} - d_{1,i} \right)^2,
    \qquad
    \boldsymbol{K}_e = \begin{bmatrix} \boldsymbol{k} & -\boldsymbol{k} \\
    -\boldsymbol{k} & \boldsymbol{k} \end{bmatrix}, \qquad
    \boldsymbol{k} = \mathrm{diag}(k_1, \ldots, k_6)

with `d_{a,i}` the six displacements `u, v, w, r_x, r_y, r_z` of node `a`
in element coordinates, see :meth:`.Spring.update_KC0`. There is no coupling
between translations and rotations, no mass and no geometric stiffness
matrix. The element axes are not taken from the nodes, which may coincide:
the `x` axis is the vector ``xi, xj, xk`` and the `z` axis is `x \times
v_{xy}`, with the vector ``vxyi, vxyj, vxyk`` of
:meth:`.Spring.update_rotation_matrix`; by default they are the global axes.

**Rigid motions.** A rigid translation and equal rotations of the two nodes
give no internal forces. Since the nodal coordinates are not used, a rigid
rotation `\boldsymbol{\theta}` of two nodes at different positions
`\boldsymbol{p}_1`, `\boldsymbol{p}_2` produces the relative translation
`\boldsymbol{\theta} \times (\boldsymbol{p}_2 - \boldsymbol{p}_1)`, normal to
the line of the nodes, which loads the translational springs in that
direction, ``kye`` and ``kze`` when the `x` axis is along the line of the
nodes: the linear element does not balance the moment of its transverse
forces. A spring is therefore invariant to rigid rotations only when its
nodes coincide, e.g. elastic supports and connections, or when it is an
axial spring, ``kye = kze = 0``, with its `x` axis along the line of its
nodes. The co-rotational wrapper :class:`.CorotationalSpring`
adds the missing moment through its projector, see
:doc:`corotational_formulation`.


References
----------

- Crisfield, M. A., Non-linear Finite Element Analysis of Solids and
  Structures, Volume 1: Essentials, John Wiley & Sons, Chichester, 1991.
