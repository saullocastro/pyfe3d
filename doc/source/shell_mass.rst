Mass matrices of the shells
===========================

The mass matrices of the shell elements, computed by ``update_M``, e.g.
:meth:`.Quad4.update_M`, follow from the kinetic energy of the FSDT
displacement field :eq:`sh-displacement-field`, integrated through the
thickness, Brockman (1987), Eq. (20):

.. math::
    :label: sh-kinetic-energy

    T = \frac{1}{2} \int_A \left[ I_\rho \left( \dot{u}^2 + \dot{v}^2 +
    \dot{w}^2 \right) + 2 I_{\rho z} \left( \dot{u} \dot{r}_y - \dot{v}
    \dot{r}_x \right) + I_{\rho z^2} \left( \dot{r}_x^2 + \dot{r}_y^2
    \right) \right] dA

with `I_\rho`, `I_{\rho z}` and `I_{\rho z^2}` the thickness integrals of
the density times `1`, `z` and `z^2`, the attributes ``intrho``,
``intrhoz`` and ``intrhoz2`` of the :class:`.ShellProp`, measured from the
reference surface and therefore including the ``offset``, see
:doc:`shell_properties`. With all the fields interpolated by the same
shape functions `S_i`, the block of the nodes `i` and `j` is, in element
coordinates and for `u, v, w, r_x, r_y, r_z`, as in Eq. (21) of Brockman
(1987),

.. math::
    :label: sh-mass-block

    \boldsymbol{M}_{ij} = h_{ij} \begin{bmatrix}
        I_\rho & 0 & 0 & 0 & I_{\rho z} & 0 \\
        0 & I_\rho & 0 & -I_{\rho z} & 0 & 0 \\
        0 & 0 & I_\rho & 0 & 0 & 0 \\
        0 & -I_{\rho z} & 0 & I_{\rho z^2} & 0 & 0 \\
        I_{\rho z} & 0 & 0 & 0 & I_{\rho z^2} & 0 \\
        0 & 0 & 0 & 0 & 0 & 0
    \end{bmatrix}, \qquad h_{ij} = \int_A S_i S_j \, dA

rotated to global coordinates with :eq:`sh-transformation`. The drilling
rotation carries no inertia. The signs of `I_{\rho z}` follow from `u + z
r_y` and `v - z r_x`, see :doc:`conventions`.


Integration schemes
-------------------

The argument ``mtype`` of ``update_M`` selects how `h_{ij}` is integrated:

.. list-table::
    :header-rows: 1

    * - ``mtype``
      - Quadrilaterals, :class:`.Quad4`, :class:`.Quad4R`
      - Triangles, :class:`.Tria3R`, :class:`.Tria3DSG`
    * - ``0``, consistent
      - two by two Gauss-Legendre points, exact
      - three points of Cowper (1973), exact: `h_{ii} = A_e/6`, `h_{ij} =
        A_e/12`
    * - ``1``, lumped
      - two by two Gauss-Lobatto points, the nodes: `h_{ii}` the Jacobian
        determinant at node `i`, `h_{ij} = 0`
      - the vertices: `h_{ii} = A_e/3`, `h_{ij} = 0`
    * - ``2``, one point
      - centroid, `h_{ij} = A/16`
      - centroid, `h_{ij} = A_e/9`, :class:`.Tria3R` only

Any other value, and ``2`` for :class:`.Tria3DSG`, raises a
``ValueError``. With ``mtype = 2`` the mass of a quadrilateral is placed at
the average of its node positions, the centroid of its area only for a
parallelogram.


Lumped mass matrices
--------------------

The lumped matrices integrate the whole kinetic energy :eq:`sh-kinetic-energy`
with a nodal quadrature, in which `S_i` is 1 at node `i` and 0 at the other
nodes: the Gauss-Lobatto points of the lumped matrix of Brockman (1987) for
the quadrilaterals and the vertex rule for the triangles. They therefore
couple no different nodes, and node `i` carries the block
:eq:`sh-mass-block` with `h_{ii}`, the coupling through `I_{\rho z}`
included: the matrix is block-diagonal per node, and diagonal only when
`I_{\rho z} = 0`, as Brockman writes, e.g. for a homogeneous plate without
offset.
The weights sum to the area, such that the lumped matrix has the exact
mass and the exact first moment of the mass through the thickness,
`I_{\rho z} A`, which keeps the centre of mass at the offset from the
reference surface. In the plane of the element the first moment of the
mass is exact for the triangles, since `\sum_i (A_e/3) \boldsymbol{x}_i =
\int_A \boldsymbol{x} \, dA`, and for a parallelogram, where `h_{ii} =
A/4 = \int_A S_i \, dA`, but not for a general quadrilateral, whose
Jacobian determinant varies. Of the rotary inertia, the term `I_{\rho z^2}
A` about the axes in the plane of the element is exact, whereas the part
due to the distribution of `I_\rho` over the area, about any axis, is
concentrated at the nodes: 3 times the exact value about the centroid of a
parallelogram and 4 times that of a triangle, an error that decreases with
the size of the elements.

The mass matrix of a warped quadrilateral with ``warp_offsets = 1`` is
`\boldsymbol{J}^\top \boldsymbol{M} \boldsymbol{J}`, with the rigid
offsets of :doc:`element_choice`; it keeps the form :eq:`sh-mass-block`
with `I_{\rho z}` and `I_{\rho z^2}` replaced by the moments of the mass
about the warped nodes, see :meth:`.Quad4.update_M`, and the lumped matrix
remains block-diagonal per node.


Choosing a mass matrix
----------------------

The consistent matrices integrate the kinetic energy exactly and are the
default. The lumped matrices give the exact mass and, for the triangles
and the parallelograms, the exact centre of mass, with no coupling between
nodes, and overestimate the rotary inertia of the in-plane distribution of
mass. The one-point matrices concentrate the mass at the centroid, or at
the average of the nodes of a quadrilateral, and couple all nodes equally.
The natural frequency analyses of :doc:`eigenvalue_analyses` and the
inertia relief of :doc:`inertia_relief` take the assembled matrix of any
scheme.


References
----------

- Brockman, R. A., Dynamics of the bilinear Mindlin plate element,
  International Journal for Numerical Methods in Engineering, 24(12),
  2343-2356, 1987. https://doi.org/10.1002/nme.1620241208
- Cowper, G. R., Gaussian quadrature formulas for triangles, International
  Journal for Numerical Methods in Engineering, 7(3), 405-408, 1973.
  https://doi.org/10.1002/nme.1620070316
