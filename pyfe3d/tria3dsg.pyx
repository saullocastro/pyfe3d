#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: nonecheck=False
#cython: overflowcheck=False
#cython: initializedcheck=False
#cython: embedsignature=True
#cython: infer_types=False
r"""
==============================================================================
Tria3DSG - Triangular element with discrete shear gap (:mod:`pyfe3d.tria3dsg`)
==============================================================================

.. currentmodule:: pyfe3d.tria3dsg

The module has three classes: :class:`.Tria3DSGData`, with the sizes of the
sparse arrays that one element fills; :class:`.Tria3DSGProbe`, with the
buffers in element coordinates that one or more elements share; and
:class:`.Tria3DSG`, the element. The ``update_*`` methods of
:class:`.Tria3DSG` write the element contributions in sparse row, column and
value arrays, from the positions given by its ``init_k_*`` attributes, or add
them to a global internal force vector.

Three-node triangular shell element with 6 degrees-of-freedom per node, `u`,
`v`, `w`, `r_x`, `r_y`, `r_z`, first-order shear deformation theory, and a
transverse shear field obtained with the discrete shear gap (DSG) method.

The membrane, bending and transverse shear operators of the element are
formed once per element, without a quadrature loop. The transverse shear
stiffnesses of the :class:`pyfe3d.shellprop.ShellProp` are used as given,
unless the ``alpha_shear_locking`` attribute of :class:`.Tria3DSG`, ``0.`` by
default, is set. The drilling rotation `r_z` is tied to the in-plane rotation
of the membrane field with a coefficient selected by the ``drilling_model``
attribute: by default the in-plane shear stiffness ``A66`` of the laminate,
otherwise a small fictitious penalty scaled by ``K6ROT``. The in-plane
displacement field has no drilling enrichment; for problems dominated by
in-plane bending, :class:`pyfe3d.tria3r.Tria3R` and :class:`pyfe3d.quad4.Quad4`
carry one. Geometrically nonlinear analyses with the von Karman membrane
strains are supported by :meth:`.Tria3DSG.update_fint` with ``nonlinear=1``,
:meth:`.Tria3DSG.update_KCNL` and :meth:`.Tria3DSG.update_KG`.

Notes
-----
The transverse shear field is obtained with the discrete shear gap (DSG)
method of Bletzinger, Bischoff and Ramm (2000) [1]_.

.. rubric:: Why this element exists

:class:`pyfe3d.tria3r.Tria3R` takes its transverse shear from a single point
at the centroid, which still locks, and cures the locking by dividing the
transverse shear stiffnesses by `1 + \alpha \ell^2/h^2` with
`\alpha = 0.7`, following Lyly, Stenberg and Vihinen (1993) [2]_. That works for
plate bending, where the error it introduces is `O(1/n^2)` in the number of
elements per span, but it leaves the element unable to reproduce a state of
constant transverse shear: measured on an 8 by 8 patch, its stiffness
against a transverse displacement field with all rotations at zero is
`3 \times 10^{-1}`, `2 \times 10^{-2}` and `1 \times 10^{-3}` of the
:class:`pyfe3d.quad4.Quad4` value at `\ell/h` of 1.25, 6.25 and 25. A
Donnell geometric stiffness matrix works on `\partial w/\partial x`, which
is exactly that subspace, so thin-shell buckling suffers badly. See
``tests/test_transverse_shear_stiffness.py``.

The DSG removes the locking in the operator instead of in the constitutive
matrix, so no stabilisation parameter is needed and the transverse shear
stiffness is used as the laminate gives it.

The algebraic stabilisation of Lyly, Stenberg and Vihinen (1993) [2]_, called
Stenberg's method by Bischoff and Bletzinger (2004) [3]_ (Sec. 3.1), is
nevertheless available through the ``alpha_shear_locking`` attribute, in the
form of Castro et al. (2019) [4]_ (Eq. (26)), which is ``0.`` by default and
then has no effect whatsoever. It is there because Castro et al. apply the
factor on top of a discrete shear gap field, as this one, and Lyly et al. on
top of the assumed-strain field of the MITC4 quadrilateral, so setting it
reproduces the stabilisation they describe, and because on a coarse mesh of a
thin plate it does improve
the answer: `\alpha = 0.07`, the lower bound of the range recommended by
Castro et al., takes the first natural frequency of a `5 \times 7` mesh from
`12.06` to `7.93` per cent above the analytical value and the buckling load
of a thin plate from `+1.71` to `+0.11` per cent. That is the opposite of
its role in :class:`pyfe3d.tria3r.Tria3R`, where the same factor is applied
to a centroid-sampled shear field and has to do the unlocking itself, which
is why that element defaults to `\alpha = 0.7`.

The default here is zero because the factor is unbounded in `\ell/h`, so any
non-zero `\alpha` removes nearly all of the transverse shear stiffness on a
practical shell mesh and with it the three properties listed above. The
Notes of :class:`.Tria3DSG` carry the measurements and the recommended value
of ``alpha_shear_locking``.

The transverse shear stiffnesses `A_{44}`, `A_{45}` and `A_{55}` are read
from the :class:`pyfe3d.shellprop.ShellProp` object with the shear correction
already applied, see
:meth:`pyfe3d.shellprop.ShellProp.calc_transverse_shear_stiffness`, and
brought to the element coordinate system with
:meth:`pyfe3d.shellprop.ShellProp.calc_Ats_element`, which, with
``shear_correction='rohwer'`` and the plies available, re-evaluates the
equilibrium-based stiffness of Rohwer (1988) [7]_ with the plies rotated to
the element coordinate system. No shear correction factor is applied by the
element.

.. rubric:: The discrete shear gap

With `\boldsymbol{\phi} = (-r_y, r_x)` the transverse shear strains of this
element's convention are

.. math::

    \gamma_{xz} = \frac{\partial w}{\partial x} + r_y
    \qquad
    \gamma_{yz} = \frac{\partial w}{\partial y} - r_x

that is `\boldsymbol{\gamma} = \nabla w - \boldsymbol{\phi}`. The shear gap `\Delta w` is
the part of the increase of the deflection between two points that is due to
shear rather than to bending ([1]_, Sec. 2, Eq. (3)). On the three-node
triangle its nodal values are obtained by integrating the shear strains along
the straight edges that join node 1 to the other two nodes ([1]_, Sec. 4.1.1),
with `\boldsymbol{\phi}` interpolated linearly,

.. math::

    \Delta w_1 = 0
    \qquad
    \Delta w_i = (w_i - w_1)
      - \frac{1}{2}(\boldsymbol{\phi}_1 + \boldsymbol{\phi}_i) \cdot
        (\boldsymbol{r}_i - \boldsymbol{r}_1)

The assumed shear strain is then the gradient of the interpolated gap,

.. math::

    \boldsymbol{\gamma} = \nabla \left( \sum_i S_i \Delta w_i \right)
                 = \sum_i \Delta w_i \nabla S_i

which is constant over the element because the shape functions are linear.
Four properties follow, all of them verified symbolically and then
numerically in ``tests/test_tria3dsg.py``:

* the three rigid-body modes that involve `w` give `\boldsymbol{\gamma} = 0`
  exactly;
* a state of constant transverse shear is reproduced exactly, which is what
  Tria3R cannot do;
* a state of constant curvature gives `\boldsymbol{\gamma} = 0` exactly, so the
  element develops no parasitic shear in pure bending. **This is the
  locking-free property**, and it is the reason no stabilisation parameter
  appears anywhere in this element;
* the operator is a linear functional of the nodal values only and is
  constant over the element, so it is formed once per element.

The gap is integrated away from one node, which makes the plain operator
depend on which node that is, a dependence inherent to the method whose
influence on the solution diminishes with mesh refinement ([1]_,
Sec. 4.1.1): measured on an irregular triangle, the element
matrix moves by 14 and 17 per cent under the two cyclic relabellings, against
`3 \times 10^{-16}` for :class:`pyfe3d.tria3r.Tria3R`, whose shear comes from
a symmetric centroid evaluation. A mesh generator orders the nodes of a
triangle as it pleases, so this element averages the three operators, one per
starting node. The averaging costs nothing: each of the three is separately
exact for all of the properties listed above, and those are linear in the
operator, so the mean inherits them, which the tests confirm. What it buys is
invariance, measured at `6 \times 10^{-16}` in
``tests/test_tria3dsg.py::test_node_numbering_invariance``.

.. rubric:: Membrane, bending and drilling

The in-plane and curvature fields are the constant-strain ones of the linear
triangle, so no locking arises there and one point integrates them exactly.
The drilling rotation `r_z` is given stiffness with the regularisation of
Hughes and Brezzi (1989) [5]_, tying `r_z` to the in-plane rotation
`\theta_z = (v_{,x} - u_{,y})/2`, exactly as in :class:`pyfe3d.quad4.Quad4`
and :class:`pyfe3d.tria3r.Tria3R`.

``drilling_model = 0``, the default, uses the physics-based coefficient
`\gamma_{r_z} = A_{66}`, a modulus of the laminate and not an adjustable
parameter. Any other value uses the fictitious penalty
`\gamma_{r_z} = K6ROT \cdot 10^{-6} A_{66}`. With either coefficient the
term `\gamma_{r_z} \int_A (r_z - \theta_z)^2 dA` is integrated with the
three-point rule, see :meth:`.Tria3DSG._update_probe_KC0ve`, whereas the
physics-based model of :class:`pyfe3d.tria3r.Tria3R`, which adds the Allman
enrichment, integrates it with a single point at the centroid.

.. note:: This element does **not** carry the hierarchical edge-mode
    enrichment of Allman (1984) [6]_ that :class:`pyfe3d.tria3r.Tria3R` and
    :class:`pyfe3d.quad4.Quad4` apply to the in-plane displacement field, so
    its membrane response in in-plane bending is the stiff one of the
    constant-strain triangle. The element is aimed at the transverse shear
    behaviour; for in-plane dominated problems prefer the enriched elements.

.. rubric:: Geometrically nonlinear analysis

The element carries the von Karman membrane strains,

.. math::

    \epsilon_{xx} = u_{,x} + \frac{1}{2} w_{,x}^2, \quad
    \epsilon_{yy} = v_{,y} + \frac{1}{2} w_{,y}^2, \quad
    \gamma_{xy} = u_{,y} + v_{,x} + w_{,x} w_{,y}

with the same split of the tangent stiffness as
:class:`pyfe3d.quad4.Quad4` and :class:`pyfe3d.tria3r.Tria3R`,

.. math::

    \boldsymbol{K}_T = \boldsymbol{K}_{C0} + \boldsymbol{K}_{CNL}(\boldsymbol{u}) + \boldsymbol{K}_G(\boldsymbol{u})

which is the exact Jacobian of the internal forces of
:meth:`.Tria3DSG.update_fint` called with ``nonlinear=1``. `\boldsymbol{K}_G` is left
homogeneous of degree one in the displacements so that a linear buckling
analysis can use it directly, the stress of the nonlinear membrane strain
being collected in `\boldsymbol{K}_{CNL}` instead. See :meth:`.Tria3DSG.update_KCNL`.

Two things are simpler here than in the quadrilateral. First, the transverse
shear strains of first-order shear deformation theory have no von Karman
terms, so the discrete shear gap operator appears in `\boldsymbol{K}_{C0}` alone and
the nonlinear methods never touch it. Second, every operator of this element
is constant over the triangle, so `\boldsymbol{K}_{CNL}`, its internal forces and
`\boldsymbol{K}_G` are all integrated exactly by a single multiplication by the area,
with no quadrature loop and no quadrature to keep consistent between the
matrix and the residual. The drilling term is the only one that needs more
than one point, and it is linear.

References
----------
.. [1] Bletzinger, K.-U., Bischoff, M., and Ramm, E., A unified approach for
   shear-locking-free triangular and rectangular shell finite elements,
   Computers & Structures, 75(3), 321-334, 2000.
   https://doi.org/10.1016/S0045-7949(99)00140-6
.. [2] Lyly, M., Stenberg, R., and Vihinen, T., A stable bilinear element for
   the Reissner-Mindlin plate model, Computer Methods in Applied Mechanics
   and Engineering, 110(3-4), 343-357, 1993.
   https://doi.org/10.1016/0045-7825(93)90214-I
.. [3] Bischoff, M., and Bletzinger, K.-U., Improving stability and accuracy
   of Reissner-Mindlin plate finite elements via algebraic subgrid scale
   stabilization, Computer Methods in Applied Mechanics and Engineering,
   193(15-16), 1517-1528, 2004. https://doi.org/10.1016/j.cma.2003.12.036
.. [4] Castro, S. G. P., Donadon, M. V., and Guimaraes, T. A. M., ES-PIM
   applied to buckling of variable angle tow laminates, Composite
   Structures, 209, 67-78, 2019.
   https://doi.org/10.1016/j.compstruct.2018.10.058
.. [5] Hughes, T. J. R., and Brezzi, F., On drilling degrees of freedom,
   Computer Methods in Applied Mechanics and Engineering, 72(1), 105-121,
   1989. https://doi.org/10.1016/0045-7825(89)90124-2
.. [6] Allman, D. J., A compatible triangular element including vertex
   rotations for plane elasticity analysis, Computers & Structures,
   19(1-2), 1-8, 1984. https://doi.org/10.1016/0045-7949(84)90197-4
.. [7] Rohwer, K., Improved transverse shear stiffness for layered finite
   elements, DFVLR-FB 88-32, 1988.

"""
import numpy as np

from .shellprop cimport ShellProp

cdef int DOF = 6
cdef int NUM_NODES = 3


cdef class Tria3DSGData:
    r"""Sizes needed to allocate the sparse matrices

    Each attribute is the number of entries that one :class:`.Tria3DSG`
    writes in the row, column and value arrays of the corresponding sparse
    matrix, starting at the position given by the matching ``init_k_*``
    attribute of the element. For ``n`` elements the arrays need ``n`` times
    these sizes. The constructor takes no parameters.

    Attributes
    ----------
    KC0_SPARSE_SIZE, : int
        ``KC0_SPARSE_SIZE = 324``, the full 18 by 18 element matrix stored
        row by row.
    KCNL_SPARSE_SIZE, : int
        ``KCNL_SPARSE_SIZE = 324``, also the full element matrix stored row
        by row.
    KG_SPARSE_SIZE, : int
        ``KG_SPARSE_SIZE = 81``, the 9 by 9 block of the translational
        degrees-of-freedom.
    M_SPARSE_SIZE, : int
        ``M_SPARSE_SIZE = 324``

    Notes
    -----
    The nonlinear constitutive terms couple every degree-of-freedom, hence
    the full element matrix for KCNL. The geometric stiffness is the Donnell
    one, involving only the gradients of `w`, hence the translational block
    for KG, see :meth:`.Tria3DSG.update_KG_given_stress`.

    """
    cdef public int KC0_SPARSE_SIZE
    cdef public int KCNL_SPARSE_SIZE
    cdef public int KG_SPARSE_SIZE
    cdef public int M_SPARSE_SIZE

    def __cinit__(Tria3DSGData self):
        self.KC0_SPARSE_SIZE = 324
        self.KCNL_SPARSE_SIZE = 324
        self.KG_SPARSE_SIZE = 81
        self.M_SPARSE_SIZE = 324


cdef class Tria3DSGProbe:
    r"""Probe carrying the element-frame coordinates, displacements and operators

    The idea behind using a probe is to avoid allocating one set of buffers
    per finite element. The buffers belong to the probe, and one probe can be
    shared amongst many elements, the contents being updated and read on
    demand. Mind that the probe always holds the values of the last update.

    The constructor takes no parameters and allocates every array filled with
    zeros.

    Attributes
    ----------
    xe, : array-like
        Element-frame nodal coordinates, ``x1, y1, z1, x2, ..., z3``.
        Dimension of length. Updated by :meth:`.Tria3DSG.update_probe_xe`.
    ue, : array-like
        Element-frame nodal displacements, 18 values, in the order `u_1,
        v_1, w_1, {r_x}_1, {r_y}_1, {r_z}_1, u_2, \ldots, {r_z}_3`.
        Translations with dimension of length, rotations dimensionless.
        Updated by :meth:`.Tria3DSG.update_probe_ue`.
    finte, : array-like
        Element-frame internal forces corresponding to ``ue``, 18 values,
        updated by :meth:`.Tria3DSG.update_probe_finte`.
        Forces, and moments (force times length) for the rotations.
    KC0ve, : array-like
        The 18 by 18 element-frame constitutive stiffness matrix stored row
        by row, 324 values.
        Updated by :meth:`.Tria3DSG.update_KC0` and
        :meth:`.Tria3DSG.update_probe_finte`.
    KCNLve, : array-like
        The 18 by 18 element-frame nonlinear constitutive stiffness matrix
        stored row by row, see :meth:`.Tria3DSG.update_KCNL`.
    BLexx, BLeyy, BLgxy : array-like
        Rows of 18 values giving the membrane strains `\epsilon_{xx},
        \epsilon_{yy}, \gamma_{xy}`.
        Their product with ``ue`` gives the membrane strains, dimensionless.
    BLkxx, BLkyy, BLkxy : array-like
        Rows of 18 values giving the curvatures `\kappa_{xx}, \kappa_{yy},
        \kappa_{xy}`.
        Their product with ``ue`` gives the curvatures, with dimension of one
        over length.
    BLgxz, BLgyz : array-like
        Rows of 18 values giving the transverse shear strains `\gamma_{xz},
        \gamma_{yz}` of the discrete shear gap, symmetrised over the three
        starting nodes.
        Their product with ``ue`` gives the strains, dimensionless.
    BLdrilling : array-like
        Row of 18 values giving the drilling residual `r_z - \theta_z`, with
        `\theta_z = (v_{,x} - u_{,y})/2`. Its `r_z` entries are the
        centroid values `1/3`; the stiffness matrix uses three points
        instead, see :meth:`.Tria3DSG.update_KC0`.
        Its product with ``ue`` is dimensionless.
    Sw_x, Sw_y : array-like
        Rows of 18 values giving `w_{,x}` and `w_{,y}`, used by the
        geometrically nonlinear terms.
        ``Sw_x`` holds `\boldsymbol{G}_x = \partial \boldsymbol{S}^w/\partial x`
        and ``Sw_y`` holds `\boldsymbol{G}_y = \partial \boldsymbol{S}^w/\partial
        y`, the underscore denoting the partial derivative.
        Their products with ``ue`` are dimensionless. The rows above are
        written by the methods of :class:`.Tria3DSG` that compute stiffness
        matrices or internal forces.

    Notes
    -----
    Every one of these operators is constant over the triangle, because all
    three shape functions are linear, so each is formed once per element and
    no quadrature loop appears anywhere in this module except for the
    drilling term. ``BLdrilling`` is the residual of the regularisation of
    Hughes and Brezzi [1]_, whose `r_z` part varies over the element; its
    stiffness is integrated exactly with the three-point rule, see
    :meth:`.Tria3DSG._update_probe_KC0ve`. ``BLgxz`` and ``BLgyz`` are the
    discrete shear gap operators averaged over the three starting nodes, see
    :meth:`.Tria3DSG._update_probe_BL_G`.

    References
    ----------
    .. [1] Hughes, T. J. R., and Brezzi, F., On drilling degrees of freedom,
       Computer Methods in Applied Mechanics and Engineering, 72(1), 105-121,
       1989. https://doi.org/10.1016/0045-7825(89)90124-2

    """
    cdef public double [::1] xe
    cdef public double [::1] ue
    cdef public double [::1] finte
    cdef public double [::1] KC0ve
    cdef public double [::1] KCNLve
    cdef public double [::1] BLexx
    cdef public double [::1] BLeyy
    cdef public double [::1] BLgxy
    cdef public double [::1] BLkxx
    cdef public double [::1] BLkyy
    cdef public double [::1] BLkxy
    cdef public double [::1] BLgxz
    cdef public double [::1] BLgyz
    cdef public double [::1] BLdrilling
    cdef public double [::1] Sw_x
    cdef public double [::1] Sw_y

    def __cinit__(Tria3DSGProbe self):
        self.xe = np.zeros(NUM_NODES*DOF//2, dtype=np.float64)
        self.ue = np.zeros(NUM_NODES*DOF, dtype=np.float64)
        self.finte = np.zeros(NUM_NODES*DOF, dtype=np.float64)
        self.KC0ve = np.zeros((NUM_NODES*DOF)**2, dtype=np.float64)
        self.KCNLve = np.zeros((NUM_NODES*DOF)**2, dtype=np.float64)
        self.BLexx = np.zeros(NUM_NODES*DOF, dtype=np.float64)
        self.BLeyy = np.zeros(NUM_NODES*DOF, dtype=np.float64)
        self.BLgxy = np.zeros(NUM_NODES*DOF, dtype=np.float64)
        self.BLkxx = np.zeros(NUM_NODES*DOF, dtype=np.float64)
        self.BLkyy = np.zeros(NUM_NODES*DOF, dtype=np.float64)
        self.BLkxy = np.zeros(NUM_NODES*DOF, dtype=np.float64)
        self.BLgxz = np.zeros(NUM_NODES*DOF, dtype=np.float64)
        self.BLgyz = np.zeros(NUM_NODES*DOF, dtype=np.float64)
        self.BLdrilling = np.zeros(NUM_NODES*DOF, dtype=np.float64)
        self.Sw_x = np.zeros(NUM_NODES*DOF, dtype=np.float64)
        self.Sw_y = np.zeros(NUM_NODES*DOF, dtype=np.float64)


cdef class Tria3DSG:
    r"""Three-node triangular shell element with a discrete shear gap

    The constructor stores the probe and sets the identification numbers and
    the positions ``c1, c2, c3`` to ``-1``, the ``init_k_*`` positions to
    ``0``, and the rotation matrices ``r11, ..., r33`` and ``m11, m12, m21,
    m22`` to the identity. Before computing any element matrix, the caller
    sets ``c1, c2, c3`` and the ``init_k_*`` positions, and calls
    :meth:`.update_rotation_matrix` and :meth:`.update_probe_xe`.

    Parameters
    ----------
    p : :class:`.Tria3DSGProbe`
        Probe used by the element, stored in ``probe``. One probe can be
        shared by many elements.

    Attributes
    ----------
    eid, : int
        Element identification number.
    pid, : int
        Property identification number.
    area, : double
        Element area.
        Dimension of length squared. Updated by :meth:`.update_area` and by
        the methods that compute stiffness matrices, internal forces or the
        mass matrix.
    n1, n2, n3 : int
        Node identification numbers.
    c1, c2, c3 : int
        Position of each node's first degree-of-freedom in the global
        displacement vector.
        The coordinates of a node are read at the positions ``c1//2``,
        ``c2//2`` and ``c3//2`` of the coordinate array.
    init_k_KC0, init_k_KCNL, init_k_KG, init_k_M : int
        Position of this element's first entry in the sparse vectors.
    drilling_model, : int
        ``0``, the default, uses the in-plane shear stiffness ``A66`` of the
        laminate, or ``gamma_rz``, as the coefficient of the drilling term.
        Any other value uses the fictitious penalty ``K6ROT*1e-6*A66``.
    gamma_rz, : double
        When non-negative and ``drilling_model = 0``, overrides ``A66`` as
        the coefficient of the drilling term. Negative, the default ``-1.``,
        means use ``A66``.
        Dimension of force per unit length, as ``A66``.
    K6ROT, : double
        Coefficient of the fictitious penalty, used when
        ``drilling_model != 0``. Default ``1.``.
        Dimensionless.
    alpha_shear_locking, : double
        Optional reduction of the transverse shear stiffnesses ``A44``,
        ``A45`` and ``A55`` of the element, by a factor that grows with the
        square of the ratio of the longest edge to the total thickness,
        applied after the rotation into the element coordinate system and
        after the shear correction. The default ``0.`` disables it entirely.
        Set it deliberately, per problem class, and never as a constant of
        the element.
        Dimensionless.
    m11, m12, m21, m22 : double
        In-plane rotation from the material to the element coordinate
        system, set by :meth:`.update_rotation_matrix`.
    r11, r12, ..., r33 : double
        Entries `T_{ij}` of the transformation matrix `\boldsymbol{T}_{3\times 3}` from
        element to global coordinates, whose columns are the element axes.
    probe, : :class:`.Tria3DSGProbe`
        Shared probe object.

    Notes
    -----
    ``drilling_model = 0`` uses `\gamma_{r_z} = A_{66}` of Hughes and
    Brezzi [1]_, and any other value `K6ROT \cdot 10^{-6} A_{66}`, see the
    module documentation. Overriding `A_{66}` with ``gamma_rz`` is what makes
    the sensitivity study recommended in the literature possible.

    ``alpha_shear_locking`` is the optional algebraic stabilisation of the
    transverse shear stiffness,
    `\hat{A}_{ts} = A_{ts}/(1 + \alpha \ell^2/h^2)` with `\ell` the longest
    edge and `h` the total thickness. The default ``0.`` is the element as
    described in the module documentation: the discrete shear gap is
    already locking-free, so nothing has to be rescaled.

    A non-zero value applies the factor of Stenberg's method, as Bischoff
    and Bletzinger (2004) [2]_ (Sec. 3.1) call the stabilisation of Lyly,
    Stenberg and Vihinen, on top of a discrete shear gap field, as Castro
    et al. (2019) [3]_ (Eq. (26)) do, which is not how
    :class:`pyfe3d.tria3r.Tria3R` uses the same symbol. There the factor
    is applied to a centroid-sampled shear field and has to do the
    unlocking itself, which is why its default
    is 0.7, seven times the literature value, and why it is load
    bearing there and merely helpful here.

    **Recommended value.** ``0.07``, the lower bound of the range
    recommended by Castro et al. (2019), on a thin plate discretised
    coarsely, and only there. Measured with this element, with the
    remaining columns from ``tests/test_tria3dsg.py`` and the study
    described in the module documentation:

    ::

        alpha                     0      0.07     0.10     0.15
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        omega_11,  5 x 7     +12.06%   +7.93%   +6.73%   +4.86%
        omega_11,  9 x 11     +2.75%   +1.69%   +1.27%   +0.60%
        omega_11, 33 x 41     -0.61%   -0.72%   -0.76%   -0.84%
        Nxx_cr,   l/h = 30    +1.71%   +0.11%   -0.33%   -1.02%
        w_max,    a/h = 5     +9.81%  +10.14%  +10.30%  +10.51%

    the first three rows being the first natural frequency of a
    thin plate under refinement, the fourth the buckling load of a
    thin plate and the last the deflection of a plate thick enough
    for transverse shear to carry 19 per cent of it, every entry a
    relative error against the analytical value.

    The buckling row crosses from the stiff to the soft side between
    `0.07` and `0.1`, which is the behaviour Castro et al. report above
    `\alpha = 0.09`, reproduced here on a discrete shear gap field
    rather than on the ES-PIM of that paper.

    **Why the default is nevertheless zero.** The factor
    `\alpha \ell^2/h^2` is unbounded in `\ell/h`, so a non-zero
    `\alpha` removes essentially all of the transverse shear stiffness
    on any practical shell mesh, and with it three properties this
    element was built to have. Measured at the default `\alpha = 0`
    against `\alpha = 0.07`:

    * the constant transverse shear state of
      ``test_constant_transverse_shear_state_is_exact``, exact to
      `10^{-12}` at every element size, drops to `1/(1 + factor)` of
      the correct energy, which is `0.067` at `L/h = 10` and
      `7 \times 10^{-8}` at `L/h = 10^4` for the right triangles of leg
      `L` of that test, whose longest edge is `\ell = \sqrt{2} L`, i.e.
      `0.125` at `\ell/h = 10` and `1.4 \times 10^{-7}` at
      `\ell/h = 10^4`;
    * the sensitivity of `\boldsymbol{K}_{C0}` to `A_{ts}` falls by about three
      hundred on the geometry of
      ``test_transverse_shear_stays_out_of_the_nonlinear_terms``, that
      is the element uses a fraction of a per cent of the transverse
      shear stiffness the laminate gives it;
    * the last row of the table above: where transverse shear is real
      physics rather than a penalty, `\alpha` can only add error, and
      it does so monotonically.

    So the gain is confined to coarse meshes of thin plates, where the
    element is already accurate, and it is paid for in the regime the
    FSDT exists to describe. Set it deliberately, per problem class, as
    for :class:`pyfe3d.tria3r.Tria3R`, and never as a constant of the
    element.

    References
    ----------
    .. [1] Hughes, T. J. R., and Brezzi, F., On drilling degrees of freedom,
       Computer Methods in Applied Mechanics and Engineering, 72(1), 105-121,
       1989. https://doi.org/10.1016/0045-7825(89)90124-2
    .. [2] Bischoff, M., and Bletzinger, K.-U., Improving stability and
       accuracy of Reissner-Mindlin plate finite elements via algebraic
       subgrid scale stabilization, Computer Methods in Applied Mechanics and
       Engineering, 193(15-16), 1517-1528, 2004.
       https://doi.org/10.1016/j.cma.2003.12.036
    .. [3] Castro, S. G. P., Donadon, M. V., and Guimaraes, T. A. M., ES-PIM
       applied to buckling of variable angle tow laminates, Composite
       Structures, 209, 67-78, 2019.
       https://doi.org/10.1016/j.compstruct.2018.10.058

    """
    cdef public int eid, pid
    cdef public int n1, n2, n3
    cdef public int c1, c2, c3
    cdef public int init_k_KC0, init_k_KCNL, init_k_KG, init_k_M
    cdef public int drilling_model
    cdef public double area
    cdef public double gamma_rz
    cdef public double K6ROT
    cdef public double alpha_shear_locking
    cdef public double m11, m12, m21, m22
    cdef public double r11, r12, r13, r21, r22, r23, r31, r32, r33
    cdef public Tria3DSGProbe probe

    def __cinit__(Tria3DSG self, Tria3DSGProbe p):
        self.probe = p
        self.eid = -1
        self.pid = -1
        self.n1 = -1
        self.n2 = -1
        self.n3 = -1
        self.c1 = -1
        self.c2 = -1
        self.c3 = -1
        self.init_k_KC0 = 0
        self.init_k_KCNL = 0
        self.init_k_KG = 0
        self.init_k_M = 0
        self.area = 0
        self.drilling_model = 0
        self.gamma_rz = -1.
        self.K6ROT = 1.
        self.alpha_shear_locking = 0. # NOTE see the attribute docs
        self.m11 = 1.
        self.m12 = 0.
        self.m21 = 0.
        self.m22 = 1.
        self.r11 = self.r22 = self.r33 = 1.
        self.r12 = self.r13 = self.r21 = 0.
        self.r23 = self.r31 = self.r32 = 0.

    cpdef void update_rotation_matrix(Tria3DSG self, double [::1] x,
            double xmati=0., double xmatj=0., double xmatk=0.):
        r"""Update the rotation matrix of the element

        The element `x` axis runs from node 1 to node 2 and the element `z`
        axis is the triangle normal, which is the same construction used by
        :class:`pyfe3d.tria3r.Tria3R`.

        The attributes ``r11, ..., r33`` are updated. ``m11, m12, m21, m22``
        are first reset to the identity and, when the material direction is
        not the zero vector and not parallel to the element normal, set to the
        rotation from the material direction projected onto the element to
        the element ``x`` axis.

        Parameters
        ----------
        x : array-like
            Global nodal coordinates, ``x_1, y_1, z_1, ..., x_M, y_M, z_M``.
            1-D array of size ``3*M``, dimension of length. The coordinates
            of each node are read from the positions ``c1//2``, ``c2//2``
            and ``c3//2``.
        xmati, xmatj, xmatk : double, optional
            Material direction in global coordinates, projected onto the
            element. Leaving it at zero means the material and element
            directions coincide.
            Only the direction of the vector is used. Default ``0.``.

        """
        cdef double xi, xj, xk, yi, yj, yk, zi, zj, zk
        cdef double x1i, x1j, x1k, x2i, x2j, x2k, x3i, x3j, x3k
        cdef double v12i, v12j, v12k, v13i, v13j, v13k
        cdef double tmp, xmatnorm, ymati, ymatj, ymatk, tol

        with nogil:
            x1i = x[self.c1//2 + 0]
            x1j = x[self.c1//2 + 1]
            x1k = x[self.c1//2 + 2]
            x2i = x[self.c2//2 + 0]
            x2j = x[self.c2//2 + 1]
            x2k = x[self.c2//2 + 2]
            x3i = x[self.c3//2 + 0]
            x3j = x[self.c3//2 + 1]
            x3k = x[self.c3//2 + 2]

            v12i = x2i - x1i
            v12j = x2j - x1j
            v12k = x2k - x1k
            v13i = x3i - x1i
            v13j = x3j - x1j
            v13k = x3k - x1k

            zi = v12j*v13k - v12k*v13j
            zj = v12k*v13i - v12i*v13k
            zk = v12i*v13j - v12j*v13i
            tmp = (zi**2 + zj**2 + zk**2)**0.5
            zi /= tmp
            zj /= tmp
            zk /= tmp
            tol = tmp/1e10

            xi = v12i
            xj = v12j
            xk = v12k
            tmp = (xi**2 + xj**2 + xk**2)**0.5
            xi /= tmp
            xj /= tmp
            xk /= tmp

            yi = zj*xk - zk*xj
            yj = zk*xi - zi*xk
            yk = zi*xj - zj*xi
            tmp = (yi**2 + yj**2 + yk**2)**0.5
            yi /= tmp
            yj /= tmp
            yk /= tmp

            self.r11 = xi
            self.r21 = xj
            self.r31 = xk
            self.r12 = yi
            self.r22 = yj
            self.r32 = yk
            self.r13 = zi
            self.r23 = zj
            self.r33 = zk

            # NOTE same construction as Quad4 and Tria3R, which agree with
            #      each other line by line. Three things in it matter and an
            #      earlier hand-written version of this method got all three
            #      wrong, which showed up as a mesh-independent error in the
            #      strain energy of a laminate with B != 0, invisible for an
            #      isotropic plate:
            #
            #      - the material direction must be *projected* onto the
            #        element plane and renormalised before the angle is
            #        taken. The raw dot product xmat.x_elem is the cosine
            #        scaled by the in-plane part of xmat, so m would not be
            #        a rotation matrix whenever xmat has a component along
            #        the element normal;
            #      - the sign of the sine follows xmat.y_elem > 0 giving a
            #        *negative* m12, the rotation being from the material to
            #        the element coordinate system. The opposite convention
            #        silently mirrors every angle ply;
            #      - when xmat is parallel to the element normal the
            #        projection is empty, and m must then be left at the
            #        identity set in __cinit__, i.e. material and element
            #        directions coincide. This is the case a caller hits by
            #        passing the normal of a flat mesh as the material
            #        direction, and it matters for triangles in particular,
            #        because the two triangles of a diagonally split cell
            #        have element frames 45 degrees apart, so a fallback
            #        that depends on the element frame orients the laminate
            #        differently in each of them.
            self.m11 = 1.
            self.m12 = 0.
            self.m21 = 0.
            self.m22 = 1.
            xmatnorm = (xmati**2 + xmatj**2 + xmatk**2)**0.5
            if xmatnorm > tol:
                xmati /= xmatnorm
                xmatj /= xmatnorm
                xmatk /= xmatnorm
                # project the material direction into the element plane,
                # ymat = z X xmat
                ymati = zj*xmatk - zk*xmatj
                ymatj = zk*xmati - zi*xmatk
                ymatk = zi*xmatj - zj*xmati
                tmp = (ymati**2 + ymatj**2 + ymatk**2)**0.5
                if tmp > tol:
                    ymati /= tmp
                    ymatj /= tmp
                    ymatk /= tmp
                    # xmat_projected = ymat X z, overwriting xmat
                    xmati = ymatj*zk - ymatk*zj
                    xmatj = ymatk*zi - ymati*zk
                    xmatk = ymati*zj - ymatj*zi
                    tmp = (xmati**2 + xmatj**2 + xmatk**2)**0.5
                    xmati /= tmp
                    xmatj /= tmp
                    xmatk /= tmp
                    # angle between xmat_projected and the element x axis,
                    # both already normalised
                    self.m11 = xmati*xi + xmatj*xj + xmatk*xk
                    self.m22 = self.m11
                    if (xmati*yi + xmatj*yj + xmatk*yk) > 0.:
                        self.m12 = -(1. - self.m11**2)**0.5
                    else:
                        self.m12 = (1. - self.m11**2)**0.5
                    self.m21 = -self.m12

    cpdef void update_probe_xe(Tria3DSG self, double [::1] x):
        r"""Update the element-frame nodal coordinates in the probe

        The coordinates of the three nodes, read from ``x`` at the positions
        ``c1//2``, ``c2//2`` and ``c3//2``, are rotated to the element
        coordinate system with the transpose `\boldsymbol{T}_{3\times 3}^\top` of the
        transformation matrix and written to ``probe.xe``.
        :meth:`.update_rotation_matrix` must have been called before. The
        ``area`` attribute is not updated here, see :meth:`.update_area`.

        Parameters
        ----------
        x : array-like
            Global nodal coordinates.
            1-D array of size ``3*M``, for ``M`` nodes, arranged as ``x_1,
            y_1, z_1, ..., x_M, y_M, z_M``, dimension of length.

        """
        cdef int i, j
        cdef int c[3]
        cdef double s1[3]
        cdef double s2[3]
        cdef double s3[3]

        with nogil:
            s1[0] = self.r11
            s1[1] = self.r21
            s1[2] = self.r31
            s2[0] = self.r12
            s2[1] = self.r22
            s2[2] = self.r32
            s3[0] = self.r13
            s3[1] = self.r23
            s3[2] = self.r33

            c[0] = self.c1
            c[1] = self.c2
            c[2] = self.c3

            for j in range(NUM_NODES):
                for i in range(DOF//2):
                    self.probe.xe[j*DOF//2 + i] = 0

            for j in range(NUM_NODES):
                for i in range(DOF//2):
                    self.probe.xe[j*DOF//2 + 0] += s1[i]*x[c[j]//2 + i]
                    self.probe.xe[j*DOF//2 + 1] += s2[i]*x[c[j]//2 + i]
                    self.probe.xe[j*DOF//2 + 2] += s3[i]*x[c[j]//2 + i]

    cpdef void update_probe_ue(Tria3DSG self, double [::1] u):
        r"""Update the element-frame nodal displacements in the probe

        The displacements and rotations of the three nodes, read from ``u`` at
        the positions ``c1``, ``c2`` and ``c3``, are rotated to the element
        coordinate system with the transpose `\boldsymbol{T}_{3\times 3}^\top` of the
        transformation matrix and written to ``probe.ue``.
        :meth:`.update_rotation_matrix` must have been called before.

        Parameters
        ----------
        u : array-like
            Global displacement vector.
            1-D array of size ``DOF*M``, for ``M`` nodes, with the six
            degrees-of-freedom of each node in sequence; translations with
            dimension of length and rotations dimensionless.

        """
        cdef int i, j, k
        cdef int c[3]
        cdef double s1[3]
        cdef double s2[3]
        cdef double s3[3]

        with nogil:
            s1[0] = self.r11
            s1[1] = self.r21
            s1[2] = self.r31
            s2[0] = self.r12
            s2[1] = self.r22
            s2[2] = self.r32
            s3[0] = self.r13
            s3[1] = self.r23
            s3[2] = self.r33

            c[0] = self.c1
            c[1] = self.c2
            c[2] = self.c3

            for j in range(NUM_NODES):
                for i in range(DOF):
                    self.probe.ue[j*DOF + i] = 0

            for j in range(NUM_NODES):
                for k in range(2):
                    for i in range(DOF//2):
                        self.probe.ue[j*DOF + k*3 + 0] += s1[i]*u[c[j] + k*3 + i]
                        self.probe.ue[j*DOF + k*3 + 1] += s2[i]*u[c[j] + k*3 + i]
                        self.probe.ue[j*DOF + k*3 + 2] += s3[i]*u[c[j] + k*3 + i]

    cpdef void update_area(Tria3DSG self):
        r"""Update the element area from the probe coordinates

        The ``area`` attribute is updated with the area of the triangle given
        by the element coordinates ``probe.xe``, ignoring the ``z``
        coordinates. The methods that compute stiffness matrices, internal
        forces or the mass matrix compute the area themselves and also store
        it in ``area``, so they do not depend on this method.

        """
        cdef double x1, x2, x3, y1, y2, y3
        with nogil:
            x1 = self.probe.xe[0]
            y1 = self.probe.xe[1]
            x2 = self.probe.xe[3]
            y2 = self.probe.xe[4]
            x3 = self.probe.xe[6]
            y3 = self.probe.xe[7]
            self.area = 0.5*((x2 - x1)*(y3 - y1) - (x3 - x1)*(y2 - y1))
            if self.area < 0.:
                self.area = -self.area # pragma: no cover

    cdef double _update_probe_BL_G(Tria3DSG self) noexcept nogil:
        r"""Update every strain operator of the probe, in element coordinates

        Fills ``BLexx, BLeyy, BLgxy`` with the membrane strains,
        ``BLkxx, BLkyy, BLkxy`` with the curvatures, ``BLgxz, BLgyz`` with
        the discrete shear gap transverse shear strains, ``BLdrilling`` with
        `r_z - \theta_z`, and ``Sw_x, Sw_y`` with `w_{,x}` and `w_{,y}`. It is
        called once per element, with no quadrature loop. The derivatives
        use the signed area of the probe coordinates, so the operators are
        correct for either node ordering, and the area returned is its
        absolute value.

        Returns
        -------
        area : double
            The element area, which is also stored in ``self.area``.

        Notes
        -----
        All three shape functions of this element are linear, so every one of
        these operators is constant over the triangle. With
        `\det J = x_1(y_2 - y_3) + x_2(y_3 - y_1) + x_3(y_1 - y_2)`, twice the
        signed area, the derivatives of the shape functions are
        `S_{1,x} = (y_2 - y_3)/\det J`, `S_{1,y} = (x_3 - x_2)/\det J` and
        their cyclic permutations, and the rows are

        .. math::

            \epsilon_{xx} = u_{,x}, \quad
            \epsilon_{yy} = v_{,y}, \quad
            \gamma_{xy} = u_{,y} + v_{,x}, \quad
            \kappa_{xx} = {r_y}_{,x}, \quad
            \kappa_{yy} = -{r_x}_{,y}, \quad
            \kappa_{xy} = -{r_x}_{,x} + {r_y}_{,y}

        the same convention as :class:`pyfe3d.quad4.Quad4` and
        :class:`pyfe3d.tria3r.Tria3R`.

        The transverse shear rows are the discrete shear gap of [1]_, see the
        module documentation, averaged over the three choices of starting
        node `p`. With `q` either of the other two nodes, `\Delta x = x_q -
        x_p` and `\Delta y = y_q - y_p`, the gap is

        .. math::

            \Delta w_q = (w_q - w_p) + \frac{1}{2}({r_y}_p + {r_y}_q) \Delta x
                         - \frac{1}{2}({r_x}_p + {r_x}_q) \Delta y

        with `\Delta w_p = 0`, and the rows are

        .. math::

            \{\gamma_{xz}, \gamma_{yz}\} = \frac{1}{3} \sum_{p=1}^{3}
            \sum_{q \neq p} \Delta w_q \{S_{q,x}, S_{q,y}\}

        Each of the three operators is separately exact for the rigid-body
        modes, for a state of constant transverse shear and for every
        Kirchhoff curvature state, and those properties are linear in the
        operator, so the mean inherits them, while it removes the dependence
        on the node numbering of a single operator.

        ``BLdrilling`` is the residual `r_z - \theta_z` of the drilling
        regularisation, with `\theta_z = (v_{,x} - u_{,y})/2`. Only its `r_z`
        part varies over the element; the entries written here are its
        centroid values, `S_i = 1/3`, and
        :meth:`._update_probe_KC0ve` overwrites them at the points of the
        three-point rule while it integrates the drilling stiffness.

        References
        ----------
        .. [1] Bletzinger, K.-U., Bischoff, M., and Ramm, E., A unified
           approach for shear-locking-free triangular and rectangular shell
           finite elements, Computers & Structures, 75(3), 321-334, 2000.
           https://doi.org/10.1016/S0045-7949(99)00140-6

        """
        cdef int i, ip, iq, kk
        cdef double x1, y1, x2, y2, x3, y3, detJ, area
        cdef double dx, dy, cx, cy
        cdef double N1x, N2x, N3x, N1y, N2y, N3y
        cdef double Nxa[3]
        cdef double Nya[3]
        cdef double xa[3]
        cdef double ya[3]
        cdef double *BLexx
        cdef double *BLeyy
        cdef double *BLgxy
        cdef double *BLkxx
        cdef double *BLkyy
        cdef double *BLkxy
        cdef double *BLgxz
        cdef double *BLgyz
        cdef double *BLdrilling
        cdef double *Sw_x
        cdef double *Sw_y

        BLexx = &self.probe.BLexx[0]
        BLeyy = &self.probe.BLeyy[0]
        BLgxy = &self.probe.BLgxy[0]
        BLkxx = &self.probe.BLkxx[0]
        BLkyy = &self.probe.BLkyy[0]
        BLkxy = &self.probe.BLkxy[0]
        BLgxz = &self.probe.BLgxz[0]
        BLgyz = &self.probe.BLgyz[0]
        BLdrilling = &self.probe.BLdrilling[0]
        Sw_x = &self.probe.Sw_x[0]
        Sw_y = &self.probe.Sw_y[0]

        # NOTE ignoring z in local coordinates, the element is flat
        x1 = self.probe.xe[0]
        y1 = self.probe.xe[1]
        x2 = self.probe.xe[3]
        y2 = self.probe.xe[4]
        x3 = self.probe.xe[6]
        y3 = self.probe.xe[7]

        # NOTE signed twice the area, so the operators keep the right sign
        #      whatever the node ordering. The area is taken from it rather
        #      than from self.area, so that the methods calling this one do
        #      not depend on update_area() having been called first;
        #      self.area is refreshed here as a convenience
        detJ = x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2)
        area = 0.5*detJ
        if area < 0.:
            area = -area # pragma: no cover
        self.area = area

        N1x = (y2 - y3)/detJ
        N2x = (y3 - y1)/detJ
        N3x = (y1 - y2)/detJ
        N1y = (x3 - x2)/detJ
        N2y = (x1 - x3)/detJ
        N3y = (x2 - x1)/detJ

        for i in range(18):
            BLexx[i] = 0.
            BLeyy[i] = 0.
            BLgxy[i] = 0.
            BLkxx[i] = 0.
            BLkyy[i] = 0.
            BLkxy[i] = 0.
            BLgxz[i] = 0.
            BLgyz[i] = 0.
            BLdrilling[i] = 0.
            Sw_x[i] = 0.
            Sw_y[i] = 0.

        # membrane, constant strain triangle
        BLexx[0] = N1x
        BLexx[6] = N2x
        BLexx[12] = N3x
        BLeyy[1] = N1y
        BLeyy[7] = N2y
        BLeyy[13] = N3y
        BLgxy[0] = N1y
        BLgxy[6] = N2y
        BLgxy[12] = N3y
        BLgxy[1] = N1x
        BLgxy[7] = N2x
        BLgxy[13] = N3x

        # curvature, same convention as Quad4 and Tria3R:
        #     kxx = dry/dx,  kyy = -drx/dy,  kxy = -drx/dx + dry/dy
        BLkxx[4] = N1x
        BLkxx[10] = N2x
        BLkxx[16] = N3x
        BLkyy[3] = -N1y
        BLkyy[9] = -N2y
        BLkyy[15] = -N3y
        BLkxy[3] = -N1x
        BLkxy[9] = -N2x
        BLkxy[15] = -N3x
        BLkxy[4] = N1y
        BLkxy[10] = N2y
        BLkxy[16] = N3y

        # gradients of w, used by the geometrically nonlinear terms
        Sw_x[2] = N1x
        Sw_x[8] = N2x
        Sw_x[14] = N3x
        Sw_y[2] = N1y
        Sw_y[8] = N2y
        Sw_y[14] = N3y

        # NOTE transverse shear, the discrete shear gap of Bletzinger,
        #      Bischoff and Ramm (2000), symmetrised over the three
        #      choices of starting node.
        #
        #      The gap is obtained by integrating the rotations along
        #      the edges away from one node, which makes the plain
        #      operator depend on which node that is: measured on an
        #      irregular triangle, the element matrix moves by 14 and
        #      17 per cent for the two cyclic relabellings, while
        #      Tria3R, whose shear comes from a symmetric centroid
        #      evaluation, moves by 3e-16. A mesh generator is free to
        #      order the nodes of a triangle any way it likes, so that
        #      is not acceptable, and the three operators are averaged.
        #
        #      Averaging costs nothing in accuracy. Each of the three is
        #      separately exact for the rigid-body modes, for a state of
        #      constant transverse shear and for every Kirchhoff
        #      curvature state, and those properties are linear in the
        #      operator, so the mean inherits all of them. See
        #      tests/test_tria3dsg.py.
        #
        #      With p the starting node and q either of the other two,
        #      the gap is
        #
        #          dw_q = (w_q - w_p) + (ry_p + ry_q) dx/2
        #                             - (rx_p + rx_q) dy/2
        #
        #      and the assumed strain is the gradient of the
        #      interpolated gap, sum_q dw_q grad(N_q), with dw_p = 0.
        Nxa[0] = N1x
        Nxa[1] = N2x
        Nxa[2] = N3x
        Nya[0] = N1y
        Nya[1] = N2y
        Nya[2] = N3y
        xa[0] = x1
        xa[1] = x2
        xa[2] = x3
        ya[0] = y1
        ya[1] = y2
        ya[2] = y3

        for ip in range(3):
            for kk in range(1, 3):
                iq = (ip + kk) % 3
                dx = xa[iq] - xa[ip]
                dy = ya[iq] - ya[ip]
                cx = Nxa[iq]/3.
                cy = Nya[iq]/3.
                BLgxz[DOF*iq + 2] += cx
                BLgxz[DOF*ip + 2] -= cx
                BLgxz[DOF*ip + 4] += cx*0.5*dx
                BLgxz[DOF*iq + 4] += cx*0.5*dx
                BLgxz[DOF*ip + 3] -= cx*0.5*dy
                BLgxz[DOF*iq + 3] -= cx*0.5*dy
                BLgyz[DOF*iq + 2] += cy
                BLgyz[DOF*ip + 2] -= cy
                BLgyz[DOF*ip + 4] += cy*0.5*dx
                BLgyz[DOF*iq + 4] += cy*0.5*dx
                BLgyz[DOF*ip + 3] -= cy*0.5*dy
                BLgyz[DOF*iq + 3] -= cy*0.5*dy

        # NOTE drilling constraint of Hughes and Brezzi, r_z - theta_z with
        #      theta_z = (v_,x - u_,y)/2. Only the r_z part varies over the
        #      element; the entries written here are its centroid values,
        #      S_i = 1/3, which is what a caller reading the operator back
        #      expects. _update_probe_KC0ve() overwrites them with the
        #      three-point rule, which is what the stiffness matrix needs
        BLdrilling[0] = N1y/2.
        BLdrilling[6] = N2y/2.
        BLdrilling[12] = N3y/2.
        BLdrilling[1] = -N1x/2.
        BLdrilling[7] = -N2x/2.
        BLdrilling[13] = -N3x/2.
        BLdrilling[5] = 1./3.
        BLdrilling[11] = 1./3.
        BLdrilling[17] = 1./3.

        return area

    cdef void _update_probe_KC0ve(Tria3DSG self, ShellProp prop) noexcept nogil:
        r"""Update the probe values of the constitutive stiffness matrix

        The attribute ``KC0ve`` of the :class:`.Tria3DSGProbe` is updated
        with the 18 by 18 element-frame matrix, stored row by row, from the
        operators of :meth:`._update_probe_BL_G` and the constitutive
        matrices of ``prop`` in the element coordinate system, including the
        options ``alpha_shear_locking``, ``drilling_model``, ``gamma_rz`` and
        ``K6ROT`` of the element.

        Notes
        -----
        Every constitutive operator is constant over the triangle, so each
        contribution is ``area*B.T @ C @ B`` with no quadrature loop, the
        drilling term being the one exception. The generalized constitutive
        matrix `C` relates `\{\epsilon_{xx}, \epsilon_{yy}, \gamma_{xy},
        \kappa_{xx}, \kappa_{yy}, \kappa_{xy}, \gamma_{yz}, \gamma_{xz}\}` to
        the stress resultants,

        .. math::

            C = \begin{bmatrix} A & B & 0 \\ B & D & 0 \\ 0 & 0 & A_{ts}
            \end{bmatrix}, \qquad
            A_{ts} = \begin{bmatrix} A_{44} & A_{45} \\ A_{45} & A_{55}
            \end{bmatrix}

        with `A_{ts}` divided by `1 + \alpha \ell^2/h^2` when
        ``alpha_shear_locking`` `= \alpha \neq 0`, `\ell` being the longest
        edge and `h` the total thickness ``prop.h``.

        The drilling term `\gamma_{r_z} \int_A B_{r_z}^\top B_{r_z} dA`,
        with `B_{r_z}` the row ``BLdrilling`` and `\gamma_{r_z}` given by
        ``gamma_rz``, `A_{66}` or `K6ROT \cdot 10^{-6} A_{66}` as described
        for the attributes of :class:`.Tria3DSG`, must be integrated with
        three points and not one: a single point supplies a single
        constraint, which leaves two of the three `r_z` degrees-of-freedom
        unconstrained and two spurious modes that survive assembly, an
        assembled patch then having 8 zero eigenvalues instead of 6. The
        integrand is quadratic in the shape functions, so the three-point
        interior rule, with the barycentric coordinates `(2/3, 1/6, 1/6)` and
        their permutations, each with the weight ``area/3``, is exact.

        """
        cdef int i, j, m, n, ip
        cdef double Ae[9]
        cdef double Be[9]
        cdef double De[9]
        cdef double Atse[4]
        cdef double A11, A12, A16, A22, A26, A66
        cdef double B11, B12, B16, B22, B26, B66
        cdef double D11, D12, D16, D22, D26, D66
        cdef double A44, A45, A55
        cdef double area, gamma_drill
        cdef double factor, maxl, l12, l23, l31
        cdef double x1, y1, x2, y2, x3, y3
        cdef double e[8]
        cdef double s[8]
        cdef double C[8][8]
        cdef double *KC0ve
        cdef double *BLexx
        cdef double *BLeyy
        cdef double *BLgxy
        cdef double *BLkxx
        cdef double *BLkyy
        cdef double *BLkxy
        cdef double *BLgxz
        cdef double *BLgyz
        cdef double *BLdrilling

        area = self._update_probe_BL_G()

        KC0ve = &self.probe.KC0ve[0]
        BLexx = &self.probe.BLexx[0]
        BLeyy = &self.probe.BLeyy[0]
        BLgxy = &self.probe.BLgxy[0]
        BLkxx = &self.probe.BLkxx[0]
        BLkyy = &self.probe.BLkyy[0]
        BLkxy = &self.probe.BLkxy[0]
        BLgxz = &self.probe.BLgxz[0]
        BLgyz = &self.probe.BLgyz[0]
        BLdrilling = &self.probe.BLdrilling[0]

        prop.get_constitutive_element(self.m11, self.m12, self.m21,
                                      self.m22, Ae, Be, De, Atse)
        A11 = Ae[0]; A12 = Ae[1]; A16 = Ae[2]
        A22 = Ae[4]; A26 = Ae[5]; A66 = Ae[8]
        B11 = Be[0]; B12 = Be[1]; B16 = Be[2]
        B22 = Be[4]; B26 = Be[5]; B66 = Be[8]
        D11 = De[0]; D12 = De[1]; D16 = De[2]
        D22 = De[4]; D26 = De[5]; D66 = De[8]
        A44 = Atse[0]; A45 = Atse[1]; A55 = Atse[3]

        # NOTE optional algebraic stabilisation of the transverse shear,
        #      off by default. The discrete shear gap above is already
        #      locking-free, so this is here only to reproduce the element
        #      of Bischoff and Bletzinger (2004) and Castro et al. (2019),
        #      where the same factor is applied on top of a DSG field with
        #      alpha near 0.1. See the alpha_shear_locking attribute.
        if self.alpha_shear_locking != 0.:
            x1 = self.probe.xe[0]; y1 = self.probe.xe[1]
            x2 = self.probe.xe[3]; y2 = self.probe.xe[4]
            x3 = self.probe.xe[6]; y3 = self.probe.xe[7]
            l12 = ((x1 - x2)**2 + (y1 - y2)**2)**0.5
            l23 = ((x2 - x3)**2 + (y2 - y3)**2)**0.5
            l31 = ((x3 - x1)**2 + (y3 - y1)**2)**0.5
            maxl = l12
            if l23 > maxl:
                maxl = l23
            if l31 > maxl:
                maxl = l31
            factor = self.alpha_shear_locking*maxl**2/prop.h**2
            A44 = A44/(1. + factor)
            A45 = A45/(1. + factor)
            A55 = A55/(1. + factor)

        if self.drilling_model == 0:
            if self.gamma_rz >= 0.:
                gamma_drill = self.gamma_rz
            else:
                gamma_drill = A66
        else:
            gamma_drill = self.K6ROT*1.e-6*A66

        # generalized constitutive matrix, order
        #     exx, eyy, gxy, kxx, kyy, kxy, gyz, gxz
        for i in range(8):
            for j in range(8):
                C[i][j] = 0.
        C[0][0] = A11; C[0][1] = A12; C[0][2] = A16
        C[1][0] = A12; C[1][1] = A22; C[1][2] = A26
        C[2][0] = A16; C[2][1] = A26; C[2][2] = A66
        C[0][3] = B11; C[0][4] = B12; C[0][5] = B16
        C[1][3] = B12; C[1][4] = B22; C[1][5] = B26
        C[2][3] = B16; C[2][4] = B26; C[2][5] = B66
        C[3][0] = B11; C[3][1] = B12; C[3][2] = B16
        C[4][0] = B12; C[4][1] = B22; C[4][2] = B26
        C[5][0] = B16; C[5][1] = B26; C[5][2] = B66
        C[3][3] = D11; C[3][4] = D12; C[3][5] = D16
        C[4][3] = D12; C[4][4] = D22; C[4][5] = D26
        C[5][3] = D16; C[5][4] = D26; C[5][5] = D66
        C[6][6] = A44; C[6][7] = A45
        C[7][6] = A45; C[7][7] = A55

        for i in range(324):
            KC0ve[i] = 0.

        # every constitutive operator is constant over the triangle
        for i in range(18):
            e[0] = BLexx[i]
            e[1] = BLeyy[i]
            e[2] = BLgxy[i]
            e[3] = BLkxx[i]
            e[4] = BLkyy[i]
            e[5] = BLkxy[i]
            e[6] = BLgyz[i]
            e[7] = BLgxz[i]
            for m in range(8):
                s[m] = 0.
                for n in range(8):
                    s[m] += C[m][n]*e[n]
            for j in range(18):
                KC0ve[18*i + j] += area*(
                    s[0]*BLexx[j] + s[1]*BLeyy[j] + s[2]*BLgxy[j]
                    + s[3]*BLkxx[j] + s[4]*BLkyy[j] + s[5]*BLkxy[j]
                    + s[6]*BLgyz[j] + s[7]*BLgxz[j])

        # NOTE the drilling term must be integrated with three points and
        #      not one: a single point supplies a single constraint, which
        #      leaves two of the three r_z degrees-of-freedom unconstrained
        #      and two spurious modes that survive assembly. Measured
        #      before this was fixed: an assembled patch had 8 zero
        #      eigenvalues instead of 6. Tria3R integrates its drilling
        #      term fully for the same reason. The integrand is quadratic
        #      in the shape functions, so the three-point interior rule
        #      below, barycentric (2/3, 1/6, 1/6) and its permutations each
        #      with weight area/3, is exact
        for ip in range(3):
            for i in range(3):
                BLdrilling[DOF*i + 5] = 2./3. if i == ip else 1./6.
            for i in range(18):
                if BLdrilling[i] == 0.:
                    continue
                for j in range(18):
                    KC0ve[18*i + j] += (area/3.*gamma_drill
                                        * BLdrilling[i]*BLdrilling[j])

        # NOTE leaving the operator with the centroid values of r_z, as
        #      _update_probe_BL_G() documents
        BLdrilling[5] = 1./3.
        BLdrilling[11] = 1./3.
        BLdrilling[17] = 1./3.

    cpdef void update_KC0(Tria3DSG self,
                          long [::1] KC0r,
                          long [::1] KC0c,
                          double [::1] KC0v,
                          ShellProp prop,
                          int update_KC0v_only=0,
                          ):
        r"""Update the sparse vectors of the constitutive stiffness matrix

        The element-frame matrix is built by :meth:`._update_probe_KC0ve`
        into the ``KC0ve`` attribute of the probe and rotated to global
        coordinates here.

        Before this method is called, :meth:`.update_probe_xe` must have
        been used to set the element-frame coordinates of the probe.

        The values in global coordinates are added to ``KC0v`` at the
        positions ``init_k_KC0`` to ``init_k_KC0 + KC0_SPARSE_SIZE - 1``. With
        ``update_KC0v_only=0`` the global row and column indices are written
        to ``KC0r`` and ``KC0c`` at the same positions. The rotation matrix of
        :meth:`.update_rotation_matrix` is used. The ``area`` attribute and
        the strain rows of the probe are updated as well.

        Parameters
        ----------
        KC0r, KC0c : array-like
            Row and column positions of the sparse values.
            1-D arrays of integers (``INT``) of size at least ``init_k_KC0 +
            KC0_SPARSE_SIZE``.
        KC0v : array-like
            Sparse values.
            1-D array of ``DOUBLE``, same size as ``KC0r``. The values are
            added to the existing ones, with dimension of force per unit
            length between translations, of force between a translation and a
            rotation, and of force times length between rotations.
        prop : :class:`pyfe3d.shellprop.ShellProp`
            Shell property.
        update_KC0v_only : int, optional
            ``0``, the default, also fills ``KC0r`` and ``KC0c``.

        Notes
        -----
        The formulation of the element matrix is given in
        :meth:`._update_probe_KC0ve`. With `\boldsymbol{T}` the transformation matrix of
        the element, block diagonal with the matrix `\boldsymbol{T}_{3\times 3}` of the
        entries `T_{ij}` stored in ``r11, ..., r33``, whose columns are the
        element axes, for the translations and for the rotations of each node,
        the global values are `\boldsymbol{K}_g = \boldsymbol{T} \boldsymbol{K}_e \boldsymbol{T}^\top`, i.e. `K_{g,mn} =
        T_{mi} K_{e,ij} T_{nj}`.

        """
        cdef int i, j, m, n, k, ke, node_i, node_j
        cdef int c[3]
        cdef double r[6][6]

        with nogil:
            self._update_probe_KC0ve(prop)

            # local to global transformation
            r[0][0] = self.r11; r[0][1] = self.r12; r[0][2] = self.r13
            r[1][0] = self.r21; r[1][1] = self.r22; r[1][2] = self.r23
            r[2][0] = self.r31; r[2][1] = self.r32; r[2][2] = self.r33
            r[3][3] = self.r11; r[3][4] = self.r12; r[3][5] = self.r13
            r[4][3] = self.r21; r[4][4] = self.r22; r[4][5] = self.r23
            r[5][3] = self.r31; r[5][4] = self.r32; r[5][5] = self.r33
            for i in range(3):
                for j in range(3):
                    r[i][j + 3] = 0.
                    r[i + 3][j] = 0.

            c[0] = self.c1
            c[1] = self.c2
            c[2] = self.c3

            if update_KC0v_only == 0:
                for node_i in range(NUM_NODES):
                    for m in range(DOF):
                        for node_j in range(NUM_NODES):
                            for n in range(DOF):
                                k = (self.init_k_KC0
                                     + 18*(node_i*DOF + m) + node_j*DOF + n)
                                KC0r[k] = c[node_i] + m
                                KC0c[k] = c[node_j] + n

            # NOTE Kg_{mn} = r_{mi} Ke_{ij} r_{nj}
            for node_i in range(NUM_NODES):
                for m in range(DOF):
                    for node_j in range(NUM_NODES):
                        for n in range(DOF):
                            k = (self.init_k_KC0
                                 + 18*(node_i*DOF + m) + node_j*DOF + n)
                            for i in range(DOF):
                                for j in range(DOF):
                                    ke = 18*(node_i*DOF + i) + node_j*DOF + j
                                    KC0v[k] += (r[m][i]*self.probe.KC0ve[ke]
                                                * r[n][j])

    cdef void _update_probe_finte_nonlinear(Tria3DSG self,
                                            ShellProp prop) noexcept nogil:
        r"""Add the geometrically nonlinear terms to the probe internal forces

        The attribute ``finte`` of the :class:`.Tria3DSGProbe` receives the
        terms of the von Karman membrane strains, evaluated at the
        displacements ``ue`` of the probe and added to the linear internal
        forces already in ``finte``. See :meth:`.update_KCNL`.

        Notes
        -----
        With the von Karman membrane strain `\boldsymbol{\epsilon}_{NL} =
        \{w_{,x}^2/2, w_{,y}^2/2, w_{,x} w_{,y}\}^\top`, the linear stress
        resultants `\boldsymbol{N} = \boldsymbol{A} \boldsymbol{\epsilon} + \boldsymbol{B} \boldsymbol{\kappa}`, and
        `\boldsymbol{N}_{NL} = \boldsymbol{A} \boldsymbol{\epsilon}_{NL}`, `\boldsymbol{M}_{NL} =
        \boldsymbol{B} \boldsymbol{\epsilon}_{NL}`, the terms added are

        .. math::

            A_e \left( \boldsymbol{B}_m^\top \boldsymbol{N}_{NL} + \boldsymbol{B}_b^\top \boldsymbol{M}_{NL}
            + \boldsymbol{B}_{NL}^\top (\boldsymbol{N} + \boldsymbol{N}_{NL}) \right)

        with `A_e` the element area and `\boldsymbol{B}_{NL}` the variation of
        `\boldsymbol{\epsilon}_{NL}`, see :meth:`.update_KCNL`, such that ``finte``
        becomes the gradient of the strain energy whose Hessian is KC0 +
        KCNL + KG. Every operator is constant over the triangle and so is the
        integrand, so the area is the whole of the quadrature.

        """
        cdef int i, a
        cdef double area, w_x, w_y
        cdef double exx, eyy, gxy, kxx, kyy, kxy
        cdef double Ae[9]
        cdef double Be[9]
        cdef double N[3]
        cdef double NNL[3]
        cdef double MNL[3]
        cdef double epsNL[3]
        cdef double *ue
        cdef double *finte
        cdef double *BLexx
        cdef double *BLeyy
        cdef double *BLgxy
        cdef double *BLkxx
        cdef double *BLkyy
        cdef double *BLkxy
        cdef double *Sw_x
        cdef double *Sw_y

        area = self._update_probe_BL_G()

        ue = &self.probe.ue[0]
        finte = &self.probe.finte[0]
        BLexx = &self.probe.BLexx[0]
        BLeyy = &self.probe.BLeyy[0]
        BLgxy = &self.probe.BLgxy[0]
        BLkxx = &self.probe.BLkxx[0]
        BLkyy = &self.probe.BLkyy[0]
        BLkxy = &self.probe.BLkxy[0]
        Sw_x = &self.probe.Sw_x[0]
        Sw_y = &self.probe.Sw_y[0]

        prop.get_constitutive_element(self.m11, self.m12, self.m21, self.m22,
                                      Ae, Be, NULL, NULL)

        exx = 0.
        eyy = 0.
        gxy = 0.
        kxx = 0.
        kyy = 0.
        kxy = 0.
        w_x = 0.
        w_y = 0.
        for i in range(18):
            exx += BLexx[i]*ue[i]
            eyy += BLeyy[i]*ue[i]
            gxy += BLgxy[i]*ue[i]
            kxx += BLkxx[i]*ue[i]
            kyy += BLkyy[i]*ue[i]
            kxy += BLkxy[i]*ue[i]
            w_x += Sw_x[i]*ue[i]
            w_y += Sw_y[i]*ue[i]

        epsNL[0] = w_x*w_x/2.
        epsNL[1] = w_y*w_y/2.
        epsNL[2] = w_x*w_y

        for a in range(3):
            # stress resultants of the linear strains, as in update_KG
            N[a] = (Ae[3*a]*exx + Ae[3*a + 1]*eyy + Ae[3*a + 2]*gxy
                  + Be[3*a]*kxx + Be[3*a + 1]*kyy + Be[3*a + 2]*kxy)
            # stress resultants of the nonlinear membrane strain
            NNL[a] = (Ae[3*a]*epsNL[0] + Ae[3*a + 1]*epsNL[1]
                      + Ae[3*a + 2]*epsNL[2])
            MNL[a] = (Be[3*a]*epsNL[0] + Be[3*a + 1]*epsNL[1]
                      + Be[3*a + 2]*epsNL[2])

        # NOTE every operator is constant over the triangle and so is the
        #      integrand, so the area is the whole of the quadrature
        for i in range(18):
            finte[i] += area*(
                # Bm.T*NNL + Bb.T*MNL
                  BLexx[i]*NNL[0] + BLeyy[i]*NNL[1] + BLgxy[i]*NNL[2]
                + BLkxx[i]*MNL[0] + BLkyy[i]*MNL[1] + BLkxy[i]*MNL[2]
                # BNL.T*(N + NNL)
                + w_x*Sw_x[i]*(N[0] + NNL[0])
                + w_y*Sw_y[i]*(N[1] + NNL[1])
                + (w_x*Sw_y[i] + w_y*Sw_x[i])*(N[2] + NNL[2])
            )

    cpdef void update_probe_finte(Tria3DSG self, ShellProp prop,
                                  int nonlinear=0):
        r"""Update the internal force vector of the probe

        The attribute ``finte`` of the :class:`.Tria3DSGProbe` is updated
        with the internal forces in element coordinates. Mind that the probe
        can be shared amongst more than one finite element, so it always
        holds the values of the last update.

        Before this method is called, :meth:`.update_probe_xe` and
        :meth:`.update_probe_ue` must have been used.

        The attribute ``KC0ve`` of the probe is updated first, as in
        :meth:`.update_KC0`, the linear internal forces being ``KC0ve``
        times ``ue``.

        Parameters
        ----------
        prop : :class:`pyfe3d.shellprop.ShellProp`
            Shell property.
        nonlinear : int, optional
            The default ``0`` gives the linear internal forces, ``KC0*u``.
            Any other value adds the geometrically nonlinear terms of the
            von Karman strains, see :meth:`._update_probe_finte_nonlinear`
            and :meth:`.update_KCNL`.

        Notes
        -----
        With ``nonlinear=1`` the exact Jacobian of the internal forces is
        ``KC0 + KCNL + KG``, see :meth:`.update_KCNL`.

        """
        cdef int i, j

        with nogil:
            self._update_probe_KC0ve(prop)
            for i in range(18):
                self.probe.finte[i] = 0.
                for j in range(18):
                    self.probe.finte[i] += (self.probe.KC0ve[18*i + j]
                                            * self.probe.ue[j])

            if nonlinear:
                self._update_probe_finte_nonlinear(prop)

    cpdef void update_fint(Tria3DSG self, double [::1] fint, ShellProp prop,
                           int nonlinear=0):
        r"""Update the internal force vector

        The element internal forces, computed in element coordinates with
        :meth:`.update_probe_finte`, are rotated to global coordinates and
        added to ``fint`` at the positions of the nodes, ``c1``, ``c2`` and
        ``c3``. Before this method is called, :meth:`.update_rotation_matrix`,
        :meth:`.update_probe_xe` and :meth:`.update_probe_ue` must have been
        used.

        Parameters
        ----------
        fint : array-like
            Array updated in place with the internal forces, in global
            coordinates. :meth:`.update_probe_finte` is called to obtain the
            element-frame internal forces, available afterwards as
            ``.probe.finte``.
            1-D array of size ``DOF*M``, with forces for the translations and
            moments (force times length) for the rotations.
        prop : :class:`pyfe3d.shellprop.ShellProp`
            Shell property.
        nonlinear : int, optional
            The default ``0`` gives the linear internal forces, ``KC0*u``.
            Any other value adds the geometrically nonlinear terms of the
            von Karman strains, see :meth:`.update_KCNL`.

        Notes
        -----
        With ``nonlinear=1`` the exact Jacobian of the internal forces is
        ``KC0 + KCNL + KG``, see :meth:`.update_KCNL`.

        """
        cdef int node_i, m, i
        cdef int c[3]
        cdef double r[3][3]
        cdef double *finte

        self.update_probe_finte(prop, nonlinear)

        with nogil:
            finte = &self.probe.finte[0]

            r[0][0] = self.r11; r[0][1] = self.r12; r[0][2] = self.r13
            r[1][0] = self.r21; r[1][1] = self.r22; r[1][2] = self.r23
            r[2][0] = self.r31; r[2][1] = self.r32; r[2][2] = self.r33

            c[0] = self.c1
            c[1] = self.c2
            c[2] = self.c3

            # NOTE the translations and the rotations rotate with the same
            #      matrix, hence the two blocks of three
            for node_i in range(NUM_NODES):
                for m in range(3):
                    for i in range(3):
                        fint[c[node_i] + m] += (
                            r[m][i]*finte[DOF*node_i + i])
                        fint[c[node_i] + 3 + m] += (
                            r[m][i]*finte[DOF*node_i + 3 + i])

    cdef void _update_probe_KCNLve(Tria3DSG self, ShellProp prop) noexcept nogil:
        r"""Update the probe values of the nonlinear constitutive stiffness matrix

        The attribute ``KCNLve`` of the :class:`.Tria3DSGProbe` is updated
        with the nonlinear constitutive stiffness matrix in element
        coordinates, stored row by row and evaluated at the displacements
        ``ue`` of the probe. See :meth:`.update_KCNL`.

        Notes
        -----
        KCNL = KC0L + KCL0 + KCLL + KGNL, with the terms given in
        :meth:`.update_KCNL`. Every operator is constant over the triangle
        and so is the integrand, so the area is the whole of the quadrature.

        """
        cdef int i, j, a
        cdef double area, w_x, w_y
        cdef double Ae[9]
        cdef double Be[9]
        cdef double NNL[3]
        # NOTE products stored row by row, with 3 rows and 18 columns
        cdef double ABL[54]
        cdef double BNL[54]
        cdef double ABNL[54]
        cdef double *ue
        cdef double *KCNLve
        cdef double *BLexx
        cdef double *BLeyy
        cdef double *BLgxy
        cdef double *BLkxx
        cdef double *BLkyy
        cdef double *BLkxy
        cdef double *Sw_x
        cdef double *Sw_y

        area = self._update_probe_BL_G()

        ue = &self.probe.ue[0]
        KCNLve = &self.probe.KCNLve[0]
        BLexx = &self.probe.BLexx[0]
        BLeyy = &self.probe.BLeyy[0]
        BLgxy = &self.probe.BLgxy[0]
        BLkxx = &self.probe.BLkxx[0]
        BLkyy = &self.probe.BLkyy[0]
        BLkxy = &self.probe.BLkxy[0]
        Sw_x = &self.probe.Sw_x[0]
        Sw_y = &self.probe.Sw_y[0]

        prop.get_constitutive_element(self.m11, self.m12, self.m21, self.m22,
                                      Ae, Be, NULL, NULL)

        for i in range(324):
            KCNLve[i] = 0.

        w_x = 0.
        w_y = 0.
        for i in range(18):
            w_x += Sw_x[i]*ue[i]
            w_y += Sw_y[i]*ue[i]

        # stress resultants of the nonlinear membrane strain,
        # epsNL = {w_x**2/2, w_y**2/2, w_x*w_y}
        for a in range(3):
            NNL[a] = (Ae[3*a]*w_x*w_x/2. + Ae[3*a + 1]*w_y*w_y/2.
                      + Ae[3*a + 2]*w_x*w_y)

        # BNL, the variation of the nonlinear membrane strain, and the
        # products A*Bm + B*Bb and A*BNL, all stored row by row with 3 rows
        for i in range(18):
            BNL[i] = w_x*Sw_x[i]
            BNL[18 + i] = w_y*Sw_y[i]
            BNL[36 + i] = w_x*Sw_y[i] + w_y*Sw_x[i]
            for a in range(3):
                ABL[18*a + i] = (Ae[3*a]*BLexx[i] + Ae[3*a + 1]*BLeyy[i]
                                 + Ae[3*a + 2]*BLgxy[i]
                                 + Be[3*a]*BLkxx[i] + Be[3*a + 1]*BLkyy[i]
                                 + Be[3*a + 2]*BLkxy[i])
                ABNL[18*a + i] = (Ae[3*a]*BNL[i] + Ae[3*a + 1]*BNL[18 + i]
                                  + Ae[3*a + 2]*BNL[36 + i])

        # NOTE every operator is constant over the triangle and so is the
        #      integrand, so the area is the whole of the quadrature
        for i in range(18):
            for j in range(18):
                KCNLve[18*i + j] += area*(
                    # KC0L = (Bm.T*A + Bb.T*B)*BNL
                      ABL[i]*BNL[j] + ABL[18 + i]*BNL[18 + j]
                    + ABL[36 + i]*BNL[36 + j]
                    # KCL0 = BNL.T*(A*Bm + B*Bb)
                    + BNL[i]*ABL[j] + BNL[18 + i]*ABL[18 + j]
                    + BNL[36 + i]*ABL[36 + j]
                    # KCLL = BNL.T*A*BNL
                    + BNL[i]*ABNL[j] + BNL[18 + i]*ABNL[18 + j]
                    + BNL[36 + i]*ABNL[36 + j]
                    # KGNL = G.T*[NNL]*G
                    + Sw_x[i]*(NNL[0]*Sw_x[j] + NNL[2]*Sw_y[j])
                    + Sw_y[i]*(NNL[2]*Sw_x[j] + NNL[1]*Sw_y[j])
                )

    cpdef void update_KCNL(Tria3DSG self,
                           long [::1] KCNLr,
                           long [::1] KCNLc,
                           double [::1] KCNLv,
                           ShellProp prop,
                           int update_KCNLv_only=0,
                           ):
        r"""Update the sparse vectors of the nonlinear constitutive stiffness matrix

        The nonlinear constitutive stiffness matrix KCNL is the part of the
        tangent stiffness matrix, besides :meth:`.update_KC0` and
        :meth:`.update_KG`, that depends on the displacements through the von
        Karman membrane strains, evaluated at the displacements ``ue`` of the
        probe. The sum of the three is the tangent stiffness matrix of the
        internal forces of :meth:`.update_fint` with ``nonlinear=1``.

        Before this method is called, :meth:`.update_probe_xe` must have
        been used with the node coordinates and :meth:`.update_probe_ue`
        with the current displacements.

        The element matrix is computed in element coordinates into the probe
        attribute ``KCNLve``, rotated to global coordinates, and its values
        are added to ``KCNLv`` at the positions ``init_k_KCNL`` to
        ``init_k_KCNL + KCNL_SPARSE_SIZE - 1``. With ``update_KCNLv_only=0``
        the global row and column indices are written to ``KCNLr`` and
        ``KCNLc`` at the same positions.

        Parameters
        ----------
        KCNLr, KCNLc : array-like
            Row and column positions of the sparse values.
            1-D arrays of integers (``INT``) of size at least ``init_k_KCNL +
            KCNL_SPARSE_SIZE``.
        KCNLv : array-like
            Sparse values.
            1-D array of ``DOUBLE``, same size as ``KCNLr``. The values are
            added to the existing ones, with dimension of force per unit
            length between translations, of force between a translation and a
            rotation, and of force times length between rotations.
        prop : :class:`pyfe3d.shellprop.ShellProp`
            Shell property.
        update_KCNLv_only : int, optional
            ``0``, the default, also fills ``KCNLr`` and ``KCNLc``.

        Notes
        -----
        Assuming that KCNL = KC0L + KCL0 + KCLL + KGNL, built from the von
        Karman membrane strains

        .. math::
            \epsilon_{xx} = u_{,x} + \frac{1}{2} w_{,x}^2, \quad
            \epsilon_{yy} = v_{,y} + \frac{1}{2} w_{,y}^2, \quad
            \gamma_{xy} = u_{,y} + v_{,x} + w_{,x} w_{,y}

        whose nonlinear part is `\boldsymbol{\epsilon}_{NL} = \frac{1}{2} \boldsymbol{B}_{NL}
        \boldsymbol{u}_e`, with `\boldsymbol{B}_{NL}` its variation. With `\boldsymbol{B}_m` and `\boldsymbol{B}_b`
        the linear membrane and bending strain-displacement matrices, `\boldsymbol{G}_x = \partial \boldsymbol{S}^w / \partial x`
        and `\boldsymbol{G}_y = \partial \boldsymbol{S}^w / \partial y` the gradients of `w`, and `\boldsymbol{A}`, `\boldsymbol{B}` the laminate matrices:

        - KC0L = `\boldsymbol{B}_m^\top \boldsymbol{A} \boldsymbol{B}_{NL} + \boldsymbol{B}_b^\top \boldsymbol{B} \boldsymbol{B}_{NL}`
        - KCL0 = KC0L`^\top`
        - KCLL = `\boldsymbol{B}_{NL}^\top \boldsymbol{A} \boldsymbol{B}_{NL}`
        - KGNL = `N_{NL,xx} \boldsymbol{G}_x^\top \boldsymbol{G}_x + N_{NL,yy}
          \boldsymbol{G}_y^\top \boldsymbol{G}_y + N_{NL,xy} (\boldsymbol{G}_x^\top
          \boldsymbol{G}_y + \boldsymbol{G}_y^\top \boldsymbol{G}_x)`, with
          `\boldsymbol{N}_{NL} = \{N_{NL,xx}, N_{NL,yy}, N_{NL,xy}\}^\top
          = \boldsymbol{A} \boldsymbol{\epsilon}_{NL}`

        The first three groups are the constitutive terms coupling the
        linear and the nonlinear parts of the membrane strain. KGNL is
        geometric, carrying the stress of the nonlinear membrane strain. It
        is collected here so that :meth:`.update_KG` stays homogeneous of
        degree one in the displacements, which is what a linear buckling
        analysis needs. With it here,

        .. math::
            \boldsymbol{K}_T = \boldsymbol{K}_{C0} + \boldsymbol{K}_{CNL}(\boldsymbol{u}) + \boldsymbol{K}_G(\boldsymbol{u})

        is the exact Jacobian of the internal forces of :meth:`.update_fint`
        with ``nonlinear=1``, and a Newton-Raphson iteration built on them
        converges quadratically.

        .. note:: The transverse shear strains of first-order shear
            deformation theory carry no von Karman terms, so the discrete
            shear gap operator of this element enters `\boldsymbol{K}_{C0}` alone and
            needs nothing here. The membrane rows have no `r_z` entries,
            this element having no Allman enrichment, so the drilling
            regularisation is likewise untouched. Both are unlike
            :class:`pyfe3d.quad4.Quad4`, whose enriched membrane rows have
            to be carried into its nonlinear tangent.

        Every operator is constant over the triangle, so each term is
        integrated exactly by a multiplication by the element area, see
        :meth:`._update_probe_KCNLve`.

        """
        cdef int i, j, m, n, k, ke, node_i, node_j
        cdef int c[3]
        cdef double r[6][6]

        with nogil:
            self._update_probe_KCNLve(prop)

            # local to global transformation
            r[0][0] = self.r11; r[0][1] = self.r12; r[0][2] = self.r13
            r[1][0] = self.r21; r[1][1] = self.r22; r[1][2] = self.r23
            r[2][0] = self.r31; r[2][1] = self.r32; r[2][2] = self.r33
            r[3][3] = self.r11; r[3][4] = self.r12; r[3][5] = self.r13
            r[4][3] = self.r21; r[4][4] = self.r22; r[4][5] = self.r23
            r[5][3] = self.r31; r[5][4] = self.r32; r[5][5] = self.r33
            for i in range(3):
                for j in range(3):
                    r[i][j + 3] = 0.
                    r[i + 3][j] = 0.

            c[0] = self.c1
            c[1] = self.c2
            c[2] = self.c3

            if update_KCNLv_only == 0:
                for node_i in range(NUM_NODES):
                    for m in range(DOF):
                        for node_j in range(NUM_NODES):
                            for n in range(DOF):
                                k = (self.init_k_KCNL
                                     + 18*(node_i*DOF + m) + node_j*DOF + n)
                                KCNLr[k] = c[node_i] + m
                                KCNLc[k] = c[node_j] + n

            # NOTE Kg_{mn} = r_{mi} Ke_{ij} r_{nj}
            for node_i in range(NUM_NODES):
                for m in range(DOF):
                    for node_j in range(NUM_NODES):
                        for n in range(DOF):
                            k = (self.init_k_KCNL
                                 + 18*(node_i*DOF + m) + node_j*DOF + n)
                            for i in range(DOF):
                                for j in range(DOF):
                                    ke = 18*(node_i*DOF + i) + node_j*DOF + j
                                    KCNLv[k] += (r[m][i]*self.probe.KCNLve[ke]
                                                 * r[n][j])

    cpdef void update_KG_given_stress(Tria3DSG self,
                                      double Nxx, double Nyy, double Nxy,
                                      long [::1] KGr,
                                      long [::1] KGc,
                                      double [::1] KGv,
                                      int update_KGv_only=0,
                                      ):
        r"""Update the sparse vectors of the geometric stiffness matrix

        A constant membrane stress state `N_{xx}, N_{yy}, N_{xy}` is assumed
        within the element. The operator involves only the gradients of the
        element-frame `w`, which are taken from the ``Sw_x, Sw_y`` rows of the
        probe and turned into global translations with the third row of the
        rotation matrix, so only the translational degrees-of-freedom appear.

        .. warning:: The stress resultants are in the **element** coordinate
            system, and for a triangle that is easy to get wrong. The
            element `x` axis runs from node 1 to node 2, so in a structured
            mesh whose cells are split along a diagonal the first triangle
            of each cell has its frame along the mesh while the second has
            it along the diagonal, some 45 degrees away. Handing the same
            `(N_{xx}, 0, 0)` to both then loads half the elements along the
            diagonal, and the answer comes to depend on how the mesh
            generator happened to order the nodes of each triangle. A
            uniaxial state along a global direction has to be rotated to
            the element coordinate system with the rotation matrix ``r11,
            ..., r33`` of this object.

            :meth:`.update_KG` is free of this, recovering the stress from
            the element's own strains in its own frame.

        The values are added to ``KGv`` at the positions ``init_k_KG`` to
        ``init_k_KG + KG_SPARSE_SIZE - 1``. With ``update_KGv_only=0`` the
        global row and column indices are written to ``KGr`` and ``KGc`` at
        the same positions. Before this method is called,
        :meth:`.update_rotation_matrix` and :meth:`.update_probe_xe` must
        have been used. The ``area`` attribute and the strain rows of the
        probe are updated as well.

        Parameters
        ----------
        Nxx, Nyy, Nxy : double
            Membrane stress resultants in the element coordinate system.
            Force per unit length.
        KGr, KGc : array-like
            Row and column positions of the sparse values.
            1-D arrays of integers (``INT``) of size at least ``init_k_KG +
            KG_SPARSE_SIZE``.
        KGv : array-like
            Sparse values.
            1-D array of ``DOUBLE``, same size as ``KGr``. The values are
            added to the existing ones, with dimension of force per unit
            length.
        update_KGv_only : int, optional
            ``0``, the default, also fills ``KGr`` and ``KGc``.

        Notes
        -----
        The operator is the Donnell one. With `S_{i,x}` and `S_{i,y}` the
        constant derivatives of the shape functions, `A_e` the element area and
        `\{T_{13}, T_{23}, T_{33}\}` the element normal in global coordinates,
        the third column of `\boldsymbol{T}_{3\times 3}`, the entry of the global
        translation `m` of node `i` and `n` of node `j` is

        .. math::

            K_{G} = A_e T_{m3} T_{n3}
            \begin{bmatrix} S_{i,x} & S_{i,y} \end{bmatrix}
            \begin{bmatrix} N_{xx} & N_{xy} \\ N_{xy} & N_{yy} \end{bmatrix}
            \begin{bmatrix} S_{j,x} \\ S_{j,y} \end{bmatrix}

        the integrand being constant over the triangle.

        A uniaxial state `N` along a global direction `\boldsymbol{d}` has to be
        rotated. With `\boldsymbol{T}_{3\times 3}` the transformation matrix of this
        object, whose columns are the element axes, `\boldsymbol{a} = \boldsymbol{T}_{3\times 3}^\top
        \boldsymbol{d}` is that direction in element coordinates and the state `N \boldsymbol{a}
        \boldsymbol{a}^\top` has components

        .. math::
            N^e_{xx} = N a_x^2, \quad
            N^e_{yy} = N a_y^2, \quad
            N^e_{xy} = N a_x a_y

        which reduces to `(N, 0, 0)` when the element `x` axis is along
        `\boldsymbol{d}`, so a quadrilateral mesh aligned with the load needs
        nothing. Measured on the cylinder of
        ``tests/test_quad4_linear_buckling_cylinder_displ.py``, whose
        cells are nearly square so the diagonal sits at 44.3 degrees,
        the three cyclic numberings of one triangle give buckling loads
        spanning a factor of fifteen without this rotation and agree to
        2e-3 with it. This is checked by
        ``test_buckling_load_does_not_depend_on_triangle_node_numbering``
        in ``tests/test_tria3dsg.py``.

        """
        cdef int m, n, k, node_i, node_j
        cdef int c[3]
        cdef double rz[3]
        cdef double Nx[3]
        cdef double Ny[3]
        cdef double area, tmp

        with nogil:
            # NOTE the same operators as every other method of this element,
            #      so the gradients of w cannot drift from the ones the
            #      internal forces of update_fint() are built with
            area = self._update_probe_BL_G()
            for node_i in range(NUM_NODES):
                Nx[node_i] = self.probe.Sw_x[DOF*node_i + 2]
                Ny[node_i] = self.probe.Sw_y[DOF*node_i + 2]

            rz[0] = self.r13
            rz[1] = self.r23
            rz[2] = self.r33

            c[0] = self.c1
            c[1] = self.c2
            c[2] = self.c3

            if update_KGv_only == 0:
                for node_i in range(NUM_NODES):
                    for m in range(3):
                        for node_j in range(NUM_NODES):
                            for n in range(3):
                                k = (self.init_k_KG
                                     + 9*(node_i*3 + m) + node_j*3 + n)
                                KGr[k] = c[node_i] + m
                                KGc[k] = c[node_j] + n

            for node_i in range(NUM_NODES):
                for m in range(3):
                    for node_j in range(NUM_NODES):
                        for n in range(3):
                            k = (self.init_k_KG
                                 + 9*(node_i*3 + m) + node_j*3 + n)
                            tmp = (Nx[node_i]*(Nx[node_j]*Nxx
                                               + Ny[node_j]*Nxy)
                                   + Ny[node_i]*(Nx[node_j]*Nxy
                                                 + Ny[node_j]*Nyy))
                            KGv[k] += rz[m]*rz[n]*tmp*area

    cpdef void update_KG(Tria3DSG self,
                         long [::1] KGr,
                         long [::1] KGc,
                         double [::1] KGv,
                         ShellProp prop,
                         int update_KGv_only=0,
                         ):
        r"""Update the geometric stiffness from the current displacements

        The membrane stress resultants are recovered from the probe
        displacements, which :meth:`.update_probe_ue` must have updated,
        using the same strain operators and the same membrane constitutive
        relation that :meth:`.update_fint` uses, and then handed to
        :meth:`.update_KG_given_stress`. Only the linear strains contribute;
        the stress of the von Karman terms of the membrane strain is carried
        by :meth:`.update_KCNL` instead.

        The values are added to ``KGv`` at the positions ``init_k_KG`` to
        ``init_k_KG + KG_SPARSE_SIZE - 1``. With ``update_KGv_only=0`` the
        global row and column indices are written to ``KGr`` and ``KGc`` at
        the same positions. Before this method is called,
        :meth:`.update_rotation_matrix` and :meth:`.update_probe_xe` must
        also have been used.

        Parameters
        ----------
        KGr, KGc : array-like
            Row and column positions of the sparse values.
            1-D arrays of integers (``INT``) of size at least ``init_k_KG +
            KG_SPARSE_SIZE``.
        KGv : array-like
            Sparse values.
            1-D array of ``DOUBLE``, same size as ``KGr``. The values are
            added to the existing ones, with dimension of force per unit
            length.
        prop : :class:`pyfe3d.shellprop.ShellProp`
            Shell property.
        update_KGv_only : int, optional
            ``0``, the default, also fills ``KGr`` and ``KGc``.

        Notes
        -----
        The stress resultants are

        .. math::

            \{N_{xx}, N_{yy}, N_{xy}\}^\top = \boldsymbol{A}\{\epsilon_{xx}, \epsilon_{yy},
            \gamma_{xy}\}^\top + \boldsymbol{B}\{\kappa_{xx}, \kappa_{yy}, \kappa_{xy}\}^\top

        with the linear strains of :meth:`._update_probe_BL_G` and `\boldsymbol{A}`,
        `\boldsymbol{B}` in the element coordinate system. Both the strains and the
        curvatures are constant over the triangle, so the resulting stress
        state is constant too, which is what
        :meth:`.update_KG_given_stress` assumes.

        Only the linear strains contribute, which leaves this matrix
        homogeneous of degree one in the displacements, as a linear buckling
        analysis needs.

        """
        cdef int i
        cdef double Ae[9]
        cdef double Be[9]
        cdef double exx, eyy, gxy, kxx, kyy, kxy
        cdef double Nxx, Nyy, Nxy
        cdef double *ue

        with nogil:
            prop.get_constitutive_element(self.m11, self.m12, self.m21,
                                          self.m22, Ae, Be, NULL, NULL)
            # NOTE the same operators as every other method of this element
            self._update_probe_BL_G()
            ue = &self.probe.ue[0]

            exx = 0.
            eyy = 0.
            gxy = 0.
            kxx = 0.
            kyy = 0.
            kxy = 0.
            for i in range(18):
                exx += self.probe.BLexx[i]*ue[i]
                eyy += self.probe.BLeyy[i]*ue[i]
                gxy += self.probe.BLgxy[i]*ue[i]
                kxx += self.probe.BLkxx[i]*ue[i]
                kyy += self.probe.BLkyy[i]*ue[i]
                kxy += self.probe.BLkxy[i]*ue[i]

            Nxx = (Ae[0]*exx + Ae[1]*eyy + Ae[2]*gxy
                   + Be[0]*kxx + Be[1]*kyy + Be[2]*kxy)
            Nyy = (Ae[3]*exx + Ae[4]*eyy + Ae[5]*gxy
                   + Be[3]*kxx + Be[4]*kyy + Be[5]*kxy)
            Nxy = (Ae[6]*exx + Ae[7]*eyy + Ae[8]*gxy
                   + Be[6]*kxx + Be[7]*kyy + Be[8]*kxy)

        self.update_KG_given_stress(Nxx, Nyy, Nxy, KGr, KGc, KGv,
                                    update_KGv_only)

    cpdef void update_M(Tria3DSG self,
                        long [::1] Mr,
                        long [::1] Mc,
                        double [::1] Mv,
                        ShellProp prop,
                        int mtype=0,
                        ):
        r"""Update the sparse vectors of the mass matrix

        The element matrix is computed from ``intrho``, ``intrhoz`` and
        ``intrhoz2`` of ``prop`` and from the area of the element coordinates
        ``probe.xe``, which is also stored in ``area``; ``intrhoz`` couples
        the translations ``u`` and ``v`` with the rotations ``r_y`` and
        ``r_x``, and ``intrhoz2`` enters only the rotations ``r_x`` and
        ``r_y``. The matrix is rotated to global coordinates and its values
        are added to ``Mv`` at the positions ``init_k_M`` to ``init_k_M +
        M_SPARSE_SIZE - 1``; the global row and column indices are always
        written to ``Mr`` and ``Mc`` at the same positions. Before this method
        is called, :meth:`.update_rotation_matrix` and
        :meth:`.update_probe_xe` must have been used.

        Parameters
        ----------
        Mr, Mc : array-like
            Row and column positions of the sparse values.
            1-D arrays of integers (``INT``) of size at least ``init_k_M +
            M_SPARSE_SIZE``.
        Mv : array-like
            Sparse values.
            1-D array of ``DOUBLE``, same size as ``Mr``. The values are added
            to the existing ones, with dimension of mass between translations,
            of mass times length between a translation and a rotation, and of
            mass times length squared between rotations.
        prop : :class:`pyfe3d.shellprop.ShellProp`
            Shell property.
            ``intrho``, ``intrhoz`` and ``intrhoz2`` are read from it.
        mtype : int, optional
            ``0`` for the consistent mass matrix and ``1`` for the lumped
            one, which puts a third of the element area on each node.
            Default ``0``. The element has no reduced-integration mass
            matrix, ``2``.

        Raises
        ------
        ValueError
            If ``mtype`` is not 0 or 1.

        Notes
        -----
        With the displacements `u + z r_y`, `v - z r_x` and `w` through the
        thickness, consistent with the curvatures of
        :meth:`._update_probe_BL_G`, the kinetic energy integrated through the
        thickness is that of Eq. (20) of Brockman [1]_,

        .. math::

            T = \frac{1}{2} \int_A \left[ I_\rho (\dot{u}^2 + \dot{v}^2 +
            \dot{w}^2) + 2 I_{\rho z} (\dot{u} \dot{r}_y - \dot{v} \dot{r}_x)
            + I_{\rho z^2} (\dot{r}_x^2 + \dot{r}_y^2) \right] dA

        with `I_\rho`, `I_{\rho z}` and `I_{\rho z^2}` given by ``intrho``,
        ``intrhoz`` and ``intrhoz2``, which include the offset. With `h_{ij}`
        the integral of the products of the linear shape functions over the
        element, the element matrix has `I_{\rho} h_{ij}` between the
        translations `u`, `v`, `w` of nodes `i` and `j`, `I_{\rho z^2} h_{ij}`
        between the rotations `r_x` and between the rotations `r_y`, `+I_{\rho
        z} h_{ij}` between `u` and `r_y`, and `-I_{\rho z} h_{ij}` between `v`
        and `r_x`. The drilling rotation `r_z` has no inertia. For the
        consistent mass matrix, `h_{ij} = \int S_i S_j dA` is `A/6` when `i =
        j` and `A/12` otherwise. The lumped mass matrix integrates `T` with
        the vertex rule, the nodal quadrature with the weight `A/3` at each
        node, where `S_i` is 1 at node `i` and 0 at the other nodes, giving
        `h_{ii} = A/3` and `h_{ij} = 0` for `i \neq j`. The coupling through
        `I_{\rho z}` stays in the block of each node, such that the lumped
        matrix is block-diagonal per node, diagonal only when `I_{\rho z} =
        0`, and has the exact mass and the exact first moment of the mass,
        with the centre of mass at the offset from the reference surface.
        The rotary inertia from the distribution of `I_\rho` over the area is
        concentrated at the nodes, 4 times the exact value about the
        centroid of the element; the term `I_{\rho z^2} A` is exact.

        References
        ----------
        .. [1] Brockman, R. A., Dynamics of the Bilinear Mindlin Plate Element,
           International Journal for Numerical Methods in Engineering, 24(12),
           2343-2356, 1987. https://doi.org/10.1002/nme.1620241208

        """
        cdef int i, j, m, n, k, ke, node_i, node_j
        cdef int c[3]
        cdef double r[6][6]
        cdef double Me[324]
        cdef double intrho, intrhoz, intrhoz2, f, A

        # NOTE checked once per call, before the code without the GIL
        if mtype != 0 and mtype != 1:
            raise ValueError('mtype=%d is not supported by Tria3DSG.update_M, '
                             'use 0 (consistent) or 1 (lumped); Tria3DSG has '
                             'no reduced-integration mass matrix (2)' % mtype)

        with nogil:
            intrho = prop.intrho
            intrhoz = prop.intrhoz
            intrhoz2 = prop.intrhoz2
            # NOTE as in update_KC0, the area comes from the probe so that
            #      this method does not depend on update_area()
            A = 0.5*(self.probe.xe[0]*(self.probe.xe[4] - self.probe.xe[7])
                     + self.probe.xe[3]*(self.probe.xe[7] - self.probe.xe[1])
                     + self.probe.xe[6]*(self.probe.xe[1] - self.probe.xe[4]))
            if A < 0.:
                A = -A # pragma: no cover
            self.area = A

            for i in range(324):
                Me[i] = 0.

            for node_i in range(NUM_NODES):
                for node_j in range(NUM_NODES):
                    if mtype == 1:
                        f = A/3. if node_i == node_j else 0.
                    else:
                        f = A/6. if node_i == node_j else A/12.
                    for i in range(3):
                        ke = (18*(node_i*DOF + i) + node_j*DOF + i)
                        Me[ke] += intrho*f
                    for i in range(3, 5):
                        ke = (18*(node_i*DOF + i) + node_j*DOF + i)
                        Me[ke] += intrhoz2*f
                    # NOTE +intrhoz between u and r_y, -intrhoz between v and
                    #      r_x, from u + z*r_y and v - z*r_x
                    Me[18*(node_i*DOF + 0) + node_j*DOF + 4] += intrhoz*f
                    Me[18*(node_i*DOF + 4) + node_j*DOF + 0] += intrhoz*f
                    Me[18*(node_i*DOF + 1) + node_j*DOF + 3] -= intrhoz*f
                    Me[18*(node_i*DOF + 3) + node_j*DOF + 1] -= intrhoz*f

            r[0][0] = self.r11; r[0][1] = self.r12; r[0][2] = self.r13
            r[1][0] = self.r21; r[1][1] = self.r22; r[1][2] = self.r23
            r[2][0] = self.r31; r[2][1] = self.r32; r[2][2] = self.r33
            r[3][3] = self.r11; r[3][4] = self.r12; r[3][5] = self.r13
            r[4][3] = self.r21; r[4][4] = self.r22; r[4][5] = self.r23
            r[5][3] = self.r31; r[5][4] = self.r32; r[5][5] = self.r33
            for i in range(3):
                for j in range(3):
                    r[i][j + 3] = 0.
                    r[i + 3][j] = 0.

            c[0] = self.c1
            c[1] = self.c2
            c[2] = self.c3

            for node_i in range(NUM_NODES):
                for m in range(DOF):
                    for node_j in range(NUM_NODES):
                        for n in range(DOF):
                            k = (self.init_k_M
                                 + 18*(node_i*DOF + m) + node_j*DOF + n)
                            Mr[k] = c[node_i] + m
                            Mc[k] = c[node_j] + n
                            for i in range(DOF):
                                for j in range(DOF):
                                    ke = 18*(node_i*DOF + i) + node_j*DOF + j
                                    Mv[k] += r[m][i]*Me[ke]*r[n][j]
