#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: nonecheck=False
#cython: overflowcheck=False
#cython: initializedcheck=False
#cython: embedsignature=True
#cython: infer_types=False
r"""
===================================================================
BeamC - Consistent Timoshenko 3D beam element (:mod:`pyfe3d.beamc`)
===================================================================

.. currentmodule:: pyfe3d.beamc

Defines the two-node beam element :class:`.BeamC`, with six
degrees-of-freedom per node, its probe :class:`.BeamCProbe` and
:class:`.BeamCData`, which gives the number of sparse entries per element of
each structural matrix. The cross-section and material properties are given
by a :class:`.BeamProp` object.

"""
import numpy as np

from .beamprop cimport BeamProp

cdef int DOF = 6
cdef int NUM_NODES = 2


cdef class BeamCData:
    r"""
    Used to allocate memory for the sparse matrices.

    Each attribute is the number of entries that one :class:`.BeamC`
    element writes in the arrays of rows, columns and values of a sparse
    matrix, at the consecutive positions that start at its ``init_k_KC0``,
    ``init_k_KCNL``, ``init_k_KG`` or ``init_k_M``. The constructor takes no
    argument and sets the values below.

    Attributes
    ----------
    KC0_SPARSE_SIZE, : int
        ``KC0_SPARSE_SIZE = 144``, written by :meth:`.BeamC.update_KC0`.

    KCNL_SPARSE_SIZE, : int
        ``KCNL_SPARSE_SIZE = 144``, written by :meth:`.BeamC.update_KCNL`.

    KG_SPARSE_SIZE, : int
        ``KG_SPARSE_SIZE = 144``, written by :meth:`.BeamC.update_KG`.

    M_SPARSE_SIZE, : int
        ``M_SPARSE_SIZE = 144``, written by :meth:`.BeamC.update_M` with
        ``mtype=0``; with ``mtype=1`` only the first ``36`` are written.

    """
    cdef public int KC0_SPARSE_SIZE
    cdef public int KCNL_SPARSE_SIZE
    cdef public int KG_SPARSE_SIZE
    cdef public int M_SPARSE_SIZE

    def __cinit__(BeamCData self):
        self.KC0_SPARSE_SIZE = 144
        self.KCNL_SPARSE_SIZE = 144
        self.KG_SPARSE_SIZE = 144
        self.M_SPARSE_SIZE = 144


cdef class BeamCProbe:
    r"""
    Probe used for local coordinates, local displacements, local stresses etc

    The idea behind using a probe is to avoid allocating larger memory buffers
    per finite element. The memory buffers are allocated per probe, and one
    probe can be shared amongst many finite elements, with the information
    being updated and retrieved on demand.

    .. note:: The probe can be shared amongst more than one finite element, 
              depending on how you defined them. Mind that the probe will
              always safe the values from the last udpate.

    The constructor takes no argument and allocates the arrays below, filled
    with zeros.


    Attributes
    ----------
    xe, : array-like
        Array of size ``NUM_NODES*DOF//2=6`` containing the nodal coordinates
        in the element coordinate system, in the following order `{x_e}_1,
        {y_e}_1, {z_e}_1, {x_e}_2, {y_e}_2, {z_e}_2`. Dimension length.
        Updated by :meth:`.BeamC.update_probe_xe`.
    ue, : array-like
        Array of size ``NUM_NODES*DOF=12`` containing the element displacements
        in the following order `u_{e_1}, v_{e_1}, w_{e_1}, {r_x}_{e_1},
        {r_y}_{e_1}, {r_z}_{e_1}, u_{e_2}, v_{e_2}, w_{e_2}, {r_x}_{e_2},
        {r_y}_{e_2}, {r_z}_{e_2}`. Translations of dimension length and
        rotations. Updated by :meth:`.BeamC.update_probe_ue`.
    finte, : array-like
        Array of size ``NUM_NODES*DOF=12`` containing the element internal
        forces corresponding to the degrees-of-freedom described by ``ue``,
        forces and moments. Updated by :meth:`.BeamC.update_probe_finte`.
    KCNLve : array-like
        Array of size ``(NUM_NODES*DOF)**2=144`` with the nonlinear
        constitutive stiffness matrix KCNL in element coordinates, stored row
        by row.

    """
    cdef public double [::1] xe
    cdef public double [::1] ue
    cdef public double [::1] finte
    cdef public double [::1] KCNLve

    def __cinit__(BeamCProbe self):
        self.xe = np.zeros(NUM_NODES*DOF//2, dtype=np.float64)
        self.ue = np.zeros(NUM_NODES*DOF, dtype=np.float64)
        self.finte = np.zeros(NUM_NODES*DOF, dtype=np.float64)
        self.KCNLve = np.zeros((NUM_NODES*DOF)**2, dtype=np.float64)


cdef class BeamC:
    r"""
    Timoshenko 3D beam element with consistent shape functions

    Nodal connectivity for the beam element::

        ^ y axis
        |
        |
        ______   --> x axis
        1    2

    The constructor stores ``p`` in ``probe``, sets ``eid``, ``pid``,
    ``n1``, ``n2``, ``c1`` and ``c2`` to ``-1``, and sets the ``init_k_*``
    positions, ``length``, ``vxyi``, ``vxyj``, ``vxyk`` and the rotation
    matrix to ``0``. The methods use the attributes updated by the others:
    :meth:`.update_rotation_matrix` reads ``c1`` and ``c2``;
    :meth:`.update_probe_xe` reads the rotation matrix and updates
    ``length``; the ``update_K*``, :meth:`.update_M` and
    :meth:`.update_probe_finte` methods read ``length`` and the rotation
    matrix; :meth:`.update_KG`, :meth:`.update_KCNL` and
    :meth:`.update_probe_finte` also read ``probe.ue``, updated by
    :meth:`.update_probe_ue`.

    Parameters
    ----------
    p : :class:`.BeamCProbe` object
        Probe of the element, stored in ``probe``. It can be shared with
        other elements.

    Attributes
    ----------
    eid, : int
        Element identification number.
    pid, : int
        Property identification number.
    length, : double
        Element length, dimension length. Updated by
        :meth:`.update_length`.
    r11, r12, r13, r21, r22, r23, r31, r32, r33 : double
        Entries `T_{ij}` of the transformation matrix `\boldsymbol{T}_{3\times 3}` from
        element to global coordinates, whose columns are the element axes.
        Dimensionless.
    vxyi, vxyj, vxyk : double
        Components of a vector on the `XY` plane of the element coordinate
        system, defined using global coordinates. Only the direction of the
        vector is used.
    c1, c2 : int
        Position of each node in the global stiffness matrix, i.e. of the
        first of its ``6`` degrees-of-freedom. The coordinates of the node
        are read from the position ``c1//2`` or ``c2//2`` of the array of
        nodal coordinates.
    n1, n2 : int
        Node identification number.
    init_k_KC0, init_k_KCNL, init_k_KG, init_k_M : int
        Position in the arrays storing the sparse data for the structural
        matrices, where the entries of this element start, see
        :class:`.BeamCData` for their number.
    probe, : :class:`.BeamCProbe` object
        Pointer to the probe.

    Notes
    -----
    Formulation based on reference [1]_.

    The cross section is rigid. In the element coordinate system, whose `x`
    axis is the beam axis, the displacements of the point `(y, z)` of the
    cross section are

    .. math::
        u_x = u - y r_z + z r_y, \qquad u_y = v - z r_x, \qquad
        u_z = w + y r_x

    where `u, v, w, r_x, r_y, r_z` are the translations and rotations of the
    beam axis, interpolated from the 12 element displacements ``ue``. The
    axial strain, with the curvatures `\kappa_y` and `\kappa_z`, and the
    transverse shear strains of the beam axis are

    .. math::
        \epsilon_{xx} = u_{,x} + y \kappa_y + z \kappa_z, \qquad
        \kappa_y = -{r_z}_{,x}, \qquad \kappa_z = {r_y}_{,x}, \qquad
        \gamma_{xy} = v_{,x} - r_z, \qquad \gamma_{xz} = w_{,x} + r_y

    and, with the twist `\kappa_x = {r_x}_{,x}`, the linear strain energy is

    .. math::
        U = \frac{1}{2} \int_0^L \boldsymbol{\epsilon}^\top \boldsymbol{C}
            \boldsymbol{\epsilon} \, dx, \qquad
        \boldsymbol{\epsilon} = \{u_{,x}, \gamma_{xy}, \gamma_{xz},
            \kappa_x, \kappa_y, \kappa_z\}^\top

    where `L` is ``length`` and `\boldsymbol{C}` is the sectional stiffness
    matrix ``C`` of :class:`.BeamProp`, of entries `C_{ij}`, `i, j = 0,
    \ldots, 5`, in the order of `\boldsymbol{\epsilon}`, symmetric, with 21
    independent entries; see :class:`.BeamProp` for its relation to the
    stiffness matrices of :mod:`pyfe3d.vabs`. When ``C`` is not set, it is
    built from the cross-section integrals of :class:`.BeamProp`, and

    .. math::
        U = \frac{1}{2} \int_0^L \Big[
            E \left( A u_{,x}^2 + I_{zz} \kappa_y^2 + I_{yy} \kappa_z^2
            + 2 A_y u_{,x} \kappa_y + 2 A_z u_{,x} \kappa_z
            + 2 I_{yz} \kappa_y \kappa_z \right) \\
            + G_{12} \left( A \gamma_{xy}^2 - 2 A_z \gamma_{xy} {r_x}_{,x}
            \right)
            + G_{13} \left( A \gamma_{xz}^2 + 2 A_y \gamma_{xz} {r_x}_{,x}
            \right)
            + G J {r_x}_{,x}^2 \Big] dx

    where `G_{12}` = ``G12`` and `G_{13}` = ``G13`` are the transverse
    shear moduli of the planes `xy` and `xz`, and `G J` = ``GJ`` is the
    torsional stiffness, see :class:`.BeamProp`. The torsion is of
    Saint-Venant type, without warping, with ``GJ`` in place of the
    torsional stiffness `G_{12} I_{yy} + G_{13} I_{zz}` of the rigid cross
    section, and no shear correction factor is applied. The axial force is
    the first entry of `\boldsymbol{C} \boldsymbol{\epsilon}`, `N = C_{00}
    u_{,x} + C_{01} \gamma_{xy} + C_{02} \gamma_{xz} + C_{03} \kappa_x +
    C_{04} \kappa_y + C_{05} \kappa_z`, i.e. `N = E (A u_{,x} + A_y
    \kappa_y + A_z \kappa_z)` without ``C``.

    `u` and `r_x` are interpolated linearly between the nodes. `v` and `r_z`
    are interpolated with the consistent shape functions of [1]_, cubic for
    `v` and quadratic for `r_z`, which depend on the parameters

    .. math::
        \alpha_y = -\frac{12 C_{44}}{C_{11} L^2}, \qquad
        \alpha_z = -\frac{12 C_{55}}{C_{22} L^2}, \qquad
        \beta_y = \frac{1}{1 - \alpha_y}, \qquad
        \beta_z = \frac{1}{1 - \alpha_z}

    named ``alphay``, ``alphaz``, ``betay`` and ``betaz`` in the code,
    taken from the diagonal of `\boldsymbol{C}`; without ``C``, `\alpha_y =
    -12 E I_{zz}/(G_{12} A L^2)` and `\alpha_z = -12 E I_{yy}/(G_{13} A
    L^2)`. With `\xi = x/L` and the nodal values `v_1, {r_z}_1, v_2,
    {r_z}_2`:

    .. math::
        v = \left[1 + \beta_y (2\xi^3 - 3\xi^2 + \alpha_y \xi)\right] v_1
            + L \beta_y \left(\xi^3 + \frac{\alpha_y - 4}{2} \xi^2
            + \left(1 - \frac{\alpha_y}{2}\right) \xi \right) {r_z}_1 \\
            + \beta_y (-2\xi^3 + 3\xi^2 - \alpha_y \xi) v_2
            + L \beta_y \left(\xi^3 - \frac{\alpha_y + 2}{2} \xi^2
            + \frac{\alpha_y}{2} \xi \right) {r_z}_2

    .. math::
        r_z = \frac{6 \beta_y}{L} (\xi^2 - \xi) (v_1 - v_2)
            + \left[1 + \beta_y (3\xi^2 + (\alpha_y - 4) \xi)\right] {r_z}_1
            + \beta_y (3\xi^2 - (\alpha_y + 2) \xi) {r_z}_2

    `w` and `r_y` use the same functions, with `v` replaced by `w`, `r_z` by
    `-r_y`, `\alpha_y` by `\alpha_z` and `\beta_y` by `\beta_z`. The
    transverse shear strains are then constant along the element, e.g.
    `\gamma_{xy} = \alpha_y \beta_y \left((v_1 - v_2)/L + ({r_z}_1 +
    {r_z}_2)/2\right)`, and they vanish with `\alpha_y = 0`, i.e. with an
    infinite ``G12``, where the interpolation of `v` becomes the cubic
    Hermite interpolation of the Euler-Bernoulli beam.

    The parameters `\alpha_y` and `\alpha_z` are negative. With `\phi_y =
    12 C_{44}/(C_{11} L^2)`, `\beta_y = 1/(1 + \phi_y)` and the functions
    above are the general solution of the homogeneous equilibrium equations
    of the strain energy above with a diagonal `\boldsymbol{C}`, `(C_{11}
    \gamma_{xy})_{,x} = 0` and `C_{11} \gamma_{xy} = C_{44}
    {\kappa_y}_{,x}`, i.e. `G_{12} A \gamma_{xy} = E I_{zz}
    {\kappa_y}_{,x}` (and likewise for `w`, `r_y`, with `C_{22}` and
    `C_{55}`), which these functions satisfy for any nodal values only with
    this sign. The same solution is Eq. (17) of [1]_ and Eq. (3.236) of
    [2]_, whose denominator `1 + 3\beta`, with `\beta = E I/(\kappa A G
    a^2)`, where `\kappa G` is `G_{12}`, and `a = L/2`, is `1 + \phi_y`;
    Eq. (21) of [1]_ gives `\alpha_y` and `\alpha_z` with the positive
    sign, inconsistent with its Eqs. (17) and (20), and its Eq. (20) prints
    the functions of `w` and `r_y` with the signs of those of `v` and
    `r_z`, which, with `\alpha_z = 0`, give `r_y = w_{,x}` instead of the
    `r_y = -w_{,x}` of the kinematics above. `\beta_y`, `\beta_z` are
    finite for any length. The mass and geometric stiffness matrices use
    the same shape functions.

    **Accuracy with couplings** (derived here). For a prismatic beam loaded
    only at its nodes, equilibrium makes the axial force, the shear forces
    and the torque constant and the bending moments linear along the
    element, and the exact strains are `\boldsymbol{C}^{-1}` times these
    sectional forces. The interpolation above contains this solution for
    any nodal loads only if `u_{,x}`, `\kappa_x`, `\gamma_{xy}` and
    `\gamma_{xz}` are constant and `C_{11} \gamma_{xy} = C_{44}
    {\kappa_y}_{,x}`, `C_{22} \gamma_{xz} = C_{55} {\kappa_z}_{,x}`, which
    holds when the only non-zero off-diagonal entry of `\boldsymbol{C}` is
    the extension-twist coupling `C_{03}`. The element is then exact, e.g.
    one element gives the tip displacements of a cantilever under any tip
    load. Any other coupling, including those of ``Ay``, ``Az`` and ``Iyz``
    when ``C`` is not set, breaks this condition, and the element converges
    as the square of the element length, see
    ``tests/test_beam_sectional_stiffness.py``: for a cantilever 0.5 m long
    of rectangular section, 50 mm by 30 mm, with a single coupling `C_{ij}
    = 0.3 \sqrt{C_{ii} C_{jj}}`, the largest error of the tip displacements
    under unit tip loads, relative to the largest, is with one element
    2.6e-4 and 2.7e-4 for `C_{12}` and `C_{45}`, 7.5e-3 for `C_{04}`,
    `C_{05}`, `C_{34}` and `C_{35}`, and up to 8.9e-2 for the couplings of
    a shear strain with the axial strain, the twist or a curvature,
    `C_{01}`, `C_{02}`, `C_{13}`, `C_{23}`, `C_{14}`, `C_{15}`, `C_{24}`
    and `C_{25}`; for `C_{15}` it is 2.4e-2, 2.0e-3 and 1.3e-4 with 16, 64
    and 256 elements. :class:`.BeamLR`, whose interpolation does not depend
    on `\boldsymbol{C}`, gives 8.3e-2, 3.3e-4 and 1.3e-6 with 1, 16 and 256
    elements for the same cases, and is exact under tip moments. With
    strong couplings of the shear strains, e.g. in composite box beams,
    :class:`.BeamLR` or a fine mesh of BeamC is preferable.

    The sign is verified against the literature in
    ``tests/test_timoshenko_beam_literature.py``: one element reproduces
    the cantilever tip deflection `P L^3/(3 E I_{zz}) + P L/(G A)` of Eq.
    (1.2) of [3]_ (``0.053 cm``) and, for a cantilever as long as deep, the
    same formula, where the shear deflection is larger than the bending
    one; one element reproduces the frequency coefficients ``3.322`` and
    ``26.569`` of Example 3.10 of [2]_, which has the same shape functions
    and consistent mass matrix with rotary inertia, three elements the
    errors of Table 3.6 of [2]_, and refined meshes its exact frequencies.
    With the positive sign these tests fail, e.g. by -44 % for the deep
    cantilever and with ``3.952`` instead of ``3.322``. In
    ``tests/test_beam_shear_moduli.py``, with different ``G12`` and
    ``G13``, the same tip deflection and frequencies are obtained in each
    plane with its own modulus, `G = G_{12}` with `I_{zz}` for `v` and `G =
    G_{13}` with `I_{yy}` for `w`.

    With ``nonlinear=1`` in :meth:`.update_fint`, the geometric
    nonlinearity is that of the von Kármán axial strain, with its nonlinear
    part `(v_{,x}^2 + w_{,x}^2)/2` replaced by its element mean, Crisfield
    [4]_, Section 7.1.3, Eqs. (7.31)-(7.32), and Crisfield [5]_, Section
    17.1.5, Eqs. (17.56a)-(17.56b):

    .. math::
        \epsilon_{xx} = u_{,x} + \bar{\epsilon}_{NL} + y \kappa_y + z
        \kappa_z, \qquad \bar{\epsilon}_{NL} = \frac{1}{L} \int_0^L
        \frac{1}{2} \left( v_{,x}^2 + w_{,x}^2 \right) dx

    since the constant `u_{,x}` of the linear `u` cannot balance the
    pointwise value, of degree four in `x` with the cubic `v` and `w`, which
    gives membrane locking [4]_. The tangent stiffness matrix ``KC0 + KCNL +
    KG`` is the exact Jacobian of the internal forces, see
    :meth:`.update_KCNL` and :meth:`.update_KG`, whose axial force is the
    element mean of `N`.

    The matrices and the internal forces are computed in element coordinates
    and transformed to global coordinates with the transformation matrix of the
    element, e.g. `\boldsymbol{K} = \boldsymbol{T} \boldsymbol{K}_e \boldsymbol{T}^\top`, where `\boldsymbol{T}` is block diagonal with
    four copies of the matrix `\boldsymbol{T}_{3\times 3}` of the entries `T_{ij}` stored
    in ``r11, ..., r33``, whose columns are the element axes, one for the
    translations and one for the rotations of each node. The element coordinate
    system is not updated with the displacements.

    References
    ----------
    .. [1] Luo, Y., 2008, “An Efficient 3D Timoshenko Beam Element with
       Consistent Shape Functions,” Adv. Theor. Appl. Mech., 1(3),
       pp. 95–106.
    .. [2] Petyt, M., 2010, Introduction to Finite Element Vibration
       Analysis, 2nd ed., Cambridge University Press, Section 3.9.
       https://doi.org/10.1017/CBO9780511761195
    .. [3] Bathe, K. J., 2014, Finite Element Procedures, 2nd ed., K. J.
       Bathe, Watertown, MA, ISBN 978-0-9790049-5-7, Section 1.2, p. 5,
       Eq. (1.2), Fig. 1.2(b).
    .. [4] Crisfield, M. A., Non-linear Finite Element Analysis of Solids
       and Structures, Volume 1: Essentials, John Wiley & Sons, Chichester,
       1991.
    .. [5] Crisfield, M. A., Non-linear Finite Element Analysis of Solids
       and Structures, Volume 2: Advanced Topics, John Wiley & Sons,
       Chichester, 1997.

    """
    cdef public int eid, pid
    cdef public int n1, n2
    cdef public int c1, c2
    cdef public int init_k_KC0, init_k_KCNL, init_k_KG, init_k_M
    cdef public double length
    cdef public double vxyi, vxyj, vxyk
    cdef public double r11, r12, r13, r21, r22, r23, r31, r32, r33
    cdef public BeamCProbe probe

    def __cinit__(BeamC self, BeamCProbe p):
        self.probe = p
        self.eid = -1
        self.pid = -1
        self.n1 = -1
        self.n2 = -1
        self.c1 = -1
        self.c2 = -1
        self.init_k_KC0 = 0
        self.init_k_KCNL = 0
        self.init_k_KG = 0
        self.init_k_M = 0
        self.length = 0
        self.vxyi = self.vxyj = self.vxyk = 0.
        self.r11 = self.r12 = self.r13 = 0.
        self.r21 = self.r22 = self.r23 = 0.
        self.r31 = self.r32 = self.r33 = 0.


    cpdef void update_rotation_matrix(BeamC self, double vxyi, double vxyj,
                                      double vxyk, double [::1] x):
        r"""Update the rotation matrix of the element

        Attributes ``r11,r12,r13,r21,r22,r23,r31,r32,r33`` are updated, with
        the entries `T_{ij}` of the transformation matrix `\boldsymbol{T}_{3\times 3}`
        from element to global coordinates, whose columns are the element axes.

        The element attributes ``vxyi``, ``vxyj`` and ``vxyk`` are also updated
        when this function is called.

        The element coordinate system is determined, identifying the `ijk`
        components of each axis: `{x_e}_i, {x_e}_j, {x_e}_k`; `{y_e}_i,
        {y_e}_j, {y_e}_k`; `{z_e}_i, {z_e}_j, {z_e}_k`.

        The element `x` axis goes from node 1 to node 2, whose coordinates are
        read from ``x`` at the positions ``c1//2`` and ``c2//2``. The `z` axis
        is normal to the `x` axis and to the vector ``(vxyi, vxyj, vxyk)``, the
        `y` axis completes the right-handed system, and the three axes are
        normalized.


        Parameters
        ----------
        vxyi, vxyj, vxyk : double
            Components of a vector on the `XY` plane of the element coordinate
            system.
            Only the direction of the vector is used. A vector parallel to the
            element axis gives a null `z` axis and a rotation matrix with
            ``nan`` entries.
        x : array-like
            Array with global nodal coordinates, for a total of `M` nodes in
            the model, this array will be arranged as: `x_1, y_1, z_1, x_2,
            y_2, z_2, ..., x_M, y_M, z_M`. Dimension length, 1-D array of size
            ``3*M``.

        """
        cdef double xi, xj, xk, yi, yj, yk, zi, zj, zk, tmp
        cdef double x1i, x1j, x1k, x2i, x2j, x2k

        with nogil:
            self.vxyi = vxyi
            self.vxyj = vxyj
            self.vxyk = vxyk

            x1i = x[self.c1//2 + 0]
            x1j = x[self.c1//2 + 1]
            x1k = x[self.c1//2 + 2]
            x2i = x[self.c2//2 + 0]
            x2j = x[self.c2//2 + 1]
            x2k = x[self.c2//2 + 2]

            xi = x2i - x1i
            xj = x2j - x1j
            xk = x2k - x1k
            tmp = (xi**2 + xj**2 + xk**2)**0.5
            xi /= tmp
            xj /= tmp
            xk /= tmp

            zi = xj*vxyk - xk*vxyj
            zj = -xi*vxyk + xk*vxyi
            zk = xi*vxyj - xj*vxyi
            tmp = (zi**2 + zj**2 + zk**2)**0.5
            zi /= tmp
            zj /= tmp
            zk /= tmp

            yi = -xj*zk + xk*zj
            yj = xi*zk - xk*zi
            yk = -xi*zj + xj*zi
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


    cpdef void update_probe_ue(BeamC self, double [::1] u):
        r"""Update the local displacement vector of the probe of the element

        .. note:: The ``ue`` attribute of object :class:`.BeamCProbe` is
                  updated, accessible using ``.probe.ue``.

        The translations and rotations of the two nodes, read from ``u`` at the
        positions ``c1`` to ``c1 + 5`` and ``c2`` to ``c2 + 5``, are
        transformed to the element coordinate system with the transpose
        `\boldsymbol{T}_{3\times 3}^\top` of the transformation matrix, which must be updated
        before, see :meth:`.update_rotation_matrix`.

        Parameters
        ----------
        u : array-like
            Array with global displacements, for a total of `M` nodes in
            the model, this array will be arranged as: `u_1, v_1, w_1, {r_x}_1,
            {r_y}_1, {r_z}_1, u_2, v_2, w_2, {r_x}_2, {r_y}_2, {r_z}_2, ...,
            u_M, v_M, w_M, {r_x}_M, {r_y}_M, {r_z}_M`. Translations of
            dimension length and rotations, 1-D array of size ``6*M``.

        """
        cdef int i, j
        cdef int c[2]
        cdef double s1[3]
        cdef double s2[3]
        cdef double s3[3]

        with nogil:
            # positions in the global stiffness matrix
            c[0] = self.c1
            c[1] = self.c2

            # global to local transformation of displacements (T.T)
            s1[0] = self.r11
            s1[1] = self.r21
            s1[2] = self.r31
            s2[0] = self.r12
            s2[1] = self.r22
            s2[2] = self.r32
            s3[0] = self.r13
            s3[1] = self.r23
            s3[2] = self.r33

            for j in range(NUM_NODES):
                for i in range(DOF):
                    self.probe.ue[j*DOF + i] = 0

            for j in range(NUM_NODES):
                for i in range(DOF//2):
                    # transforming translations
                    self.probe.ue[j*DOF + 0] += s1[i]*u[c[j] + 0 + i]
                    self.probe.ue[j*DOF + 1] += s2[i]*u[c[j] + 0 + i]
                    self.probe.ue[j*DOF + 2] += s3[i]*u[c[j] + 0 + i]
                    # transforming rotations
                    self.probe.ue[j*DOF + 3] += s1[i]*u[c[j] + 3 + i]
                    self.probe.ue[j*DOF + 4] += s2[i]*u[c[j] + 3 + i]
                    self.probe.ue[j*DOF + 5] += s3[i]*u[c[j] + 3 + i]


    cpdef void update_probe_xe(BeamC self, double [::1] x):
        r"""Update the 3D coordinates of the probe of the element

        .. note:: The ``xe`` attribute of object :class:`.BeamCProbe` is
                  updated, accessible using ``.probe.xe``.

        The coordinates of the two nodes, read from ``x`` at the positions
        ``c1//2`` and ``c2//2``, are transformed to the element coordinate
        system with the transpose `\boldsymbol{T}_{3\times 3}^\top` of the transformation
        matrix, which must be updated before, see
        :meth:`.update_rotation_matrix`. Then :meth:`.update_length` is called,
        updating ``length``.

        Parameters
        ----------
        x : array-like
            Array with global nodal coordinates, for a total of `M` nodes in
            the model, this array will be arranged as: `x_1, y_1, z_1, x_2,
            y_2, z_2, ..., x_M, y_M, z_M`. Dimension length, 1-D array of size
            ``3*M``.

        """
        cdef int i, j
        cdef int c[2]
        cdef double s1[3]
        cdef double s2[3]
        cdef double s3[3]

        with nogil:
            # positions in the global stiffness matrix
            c[0] = self.c1
            c[1] = self.c2

            # global to local transformation of coordinates (T.T)
            s1[0] = self.r11
            s1[1] = self.r21
            s1[2] = self.r31
            s2[0] = self.r12
            s2[1] = self.r22
            s2[2] = self.r32
            s3[0] = self.r13
            s3[1] = self.r23
            s3[2] = self.r33

            for j in range(NUM_NODES):
                for i in range(DOF//2):
                    self.probe.xe[j*DOF//2 + i] = 0

            for j in range(NUM_NODES):
                for i in range(DOF//2):
                    self.probe.xe[j*DOF//2 + 0] += s1[i]*x[c[j]//2 + i]
                    self.probe.xe[j*DOF//2 + 1] += s2[i]*x[c[j]//2 + i]
                    self.probe.xe[j*DOF//2 + 2] += s3[i]*x[c[j]//2 + i]

        self.update_length()


    cpdef void update_length(BeamC self):
        r"""Update element length

        The distance between the two nodes, computed from the coordinates
        ``probe.xe`` in the element coordinate system, is stored in the
        attribute ``length``. Called by :meth:`.update_probe_xe`.

        """
        cdef double x1, x2, y1, y2, z1, z2
        with nogil:
            # NOTE ignoring z in local coordinates
            x1 = self.probe.xe[0]
            y1 = self.probe.xe[1]
            z1 = self.probe.xe[2]
            x2 = self.probe.xe[3]
            y2 = self.probe.xe[4]
            z2 = self.probe.xe[5]
            self.length = ((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)**0.5

    
    cpdef void update_probe_finte(BeamC self,
                           BeamProp prop,
                           int nonlinear=0):
        r"""Update the internal force vector of the probe

        The attribute ``finte`` is updated with the :class:`.BeamCProbe` the
        internal forces in local coordinates. While using this function, mind
        that the probe can be shared amongst more than one finite element,
        depending how you defined them, meaning that the probe will always safe
        the values from the last udpate.

        .. note:: The ``finte`` attribute of object :class:`.BeamCProbe` is
                  updated, accessible using ``.probe.finte``.

        The 12 values of ``finte`` are overwritten. They are computed from
        ``length`` and from the displacements ``probe.ue``, which must be
        updated before, see :meth:`.update_probe_xe` and
        :meth:`.update_probe_ue`.

        Parameters
        ----------
        prop : :class:`.BeamProp` object
            Beam property object from where the stiffness and mass attributes
            are read from.
        nonlinear : int, optional
            The default ``0`` gives the linear internal forces, ``KC0*u``. Any other
            value adds the geometrically nonlinear terms, for which the tangent
            stiffness matrix of the internal forces is ``KC0 + KCNL + KG``, see
            :meth:`.update_KCNL`.

        Notes
        -----
        The linear internal forces are `\boldsymbol{K}_{C0_e} \boldsymbol{u}_e`, with the matrix
        `\boldsymbol{K}_{C0_e}` of :meth:`.update_KC0` in element coordinates. With
        ``nonlinear`` different from ``0``, the terms of the von Kármán strains
        are added, see :meth:`._update_probe_finte_nonlinear`, such that
        ``finte`` is the gradient of the strain energy with the von Kármán
        axial strain averaged over the element, see :meth:`.update_KCNL`,
        for which the exact Jacobian of the internal forces is
        ``KC0 + KCNL + KG``, see :meth:`.update_KCNL`.

        """
        cdef double *ue
        cdef double *finte
        cdef double L, alphay, alphaz, betay, betaz
        cdef double C[36]
        cdef double C00, C01, C02, C03, C04, C05, C11, C12, C13, C14, C15, C22
        cdef double C23, C24, C25, C33, C34, C35, C44, C45, C55

        cdef double KC0e0000, KC0e0001, KC0e0002, KC0e0003, KC0e0004, KC0e0005
        cdef double KC0e0006, KC0e0007, KC0e0008, KC0e0009, KC0e0010, KC0e0011
        cdef double KC0e0101, KC0e0102, KC0e0103, KC0e0104, KC0e0105, KC0e0106
        cdef double KC0e0107, KC0e0108, KC0e0109, KC0e0110, KC0e0111, KC0e0202
        cdef double KC0e0203, KC0e0204, KC0e0205, KC0e0206, KC0e0207, KC0e0208
        cdef double KC0e0209, KC0e0210, KC0e0211, KC0e0303, KC0e0304, KC0e0305
        cdef double KC0e0306, KC0e0307, KC0e0308, KC0e0309, KC0e0310, KC0e0311
        cdef double KC0e0404, KC0e0405, KC0e0406, KC0e0407, KC0e0408, KC0e0409
        cdef double KC0e0410, KC0e0411, KC0e0505, KC0e0506, KC0e0507, KC0e0508
        cdef double KC0e0509, KC0e0510, KC0e0511, KC0e0606, KC0e0607, KC0e0608
        cdef double KC0e0609, KC0e0610, KC0e0611, KC0e0707, KC0e0708, KC0e0709
        cdef double KC0e0710, KC0e0711, KC0e0808, KC0e0809, KC0e0810, KC0e0811
        cdef double KC0e0909, KC0e0910, KC0e0911, KC0e1010, KC0e1011, KC0e1111

        with nogil:
            L = self.length
            prop.get_C(C)
            C00 = C[0]
            C01 = C[1]
            C02 = C[2]
            C03 = C[3]
            C04 = C[4]
            C05 = C[5]
            C11 = C[7]
            C12 = C[8]
            C13 = C[9]
            C14 = C[10]
            C15 = C[11]
            C22 = C[14]
            C23 = C[15]
            C24 = C[16]
            C25 = C[17]
            C33 = C[21]
            C34 = C[22]
            C35 = C[23]
            C44 = C[28]
            C45 = C[29]
            C55 = C[35]

            # NOTE shear parameters of the shape functions from the diagonal of
            #      the sectional stiffness matrix, see the notes of BeamC
            alphay = -12*C44/(C11*L**2)
            alphaz = -12*C55/(C22*L**2)
            betay = 1/(1. - alphay)
            betaz = 1/(1. - alphaz)

            KC0e0000 = C00/L
            KC0e0001 = -C01*alphay*betay/L
            KC0e0002 = -C02*alphaz*betaz/L
            KC0e0003 = C03/L
            KC0e0004 = betaz*(C02*L*alphaz - 2*C05*alphaz + 2*C05)/(2*L)
            KC0e0005 = betay*(-C01*L*alphay + 2*C04*alphay - 2*C04)/(2*L)
            KC0e0006 = -C00/L
            KC0e0007 = C01*alphay*betay/L
            KC0e0008 = C02*alphaz*betaz/L
            KC0e0009 = -C03/L
            KC0e0010 = betaz*(C02*L*alphaz + 2*C05*alphaz - 2*C05)/(2*L)
            KC0e0011 = betay*(-C01*L*alphay - 2*C04*alphay + 2*C04)/(2*L)
            KC0e0101 = betay**2*(C11*L**2*alphay**2 + 12*C44)/L**3
            KC0e0102 = betay*betaz*(C12*L**2*alphay*alphaz + 12*C45)/L**3
            KC0e0103 = -C13*alphay*betay/L
            KC0e0104 = betay*betaz*(-C12*L**2*alphay*alphaz + 2*C15*L*alphay*alphaz - 2*C15*L*alphay - 12*C45)/(2*L**2)
            KC0e0105 = betay**2*(C11*L**2*alphay**2/2 - C14*L*alphay**2 + C14*L*alphay + 6*C44)/L**2
            KC0e0106 = C01*alphay*betay/L
            KC0e0107 = betay**2*(-C11*L**2*alphay**2 - 12*C44)/L**3
            KC0e0108 = betay*betaz*(-C12*L**2*alphay*alphaz - 12*C45)/L**3
            KC0e0109 = C13*alphay*betay/L
            KC0e0110 = betay*betaz*(-C12*L**2*alphay*alphaz - 2*C15*L*alphay*alphaz + 2*C15*L*alphay - 12*C45)/(2*L**2)
            KC0e0111 = betay**2*(C11*L**2*alphay**2/2 + C14*L*alphay**2 - C14*L*alphay + 6*C44)/L**2
            KC0e0202 = betaz**2*(C22*L**2*alphaz**2 + 12*C55)/L**3
            KC0e0203 = -C23*alphaz*betaz/L
            KC0e0204 = betaz**2*(-C22*L**2*alphaz**2/2 + C25*L*alphaz**2 - C25*L*alphaz - 6*C55)/L**2
            KC0e0205 = betay*betaz*(C12*L**2*alphay*alphaz - 2*C24*L*alphay*alphaz + 2*C24*L*alphaz + 12*C45)/(2*L**2)
            KC0e0206 = C02*alphaz*betaz/L
            KC0e0207 = betay*betaz*(-C12*L**2*alphay*alphaz - 12*C45)/L**3
            KC0e0208 = betaz**2*(-C22*L**2*alphaz**2 - 12*C55)/L**3
            KC0e0209 = C23*alphaz*betaz/L
            KC0e0210 = betaz**2*(-C22*L**2*alphaz**2/2 - C25*L*alphaz**2 + C25*L*alphaz - 6*C55)/L**2
            KC0e0211 = betay*betaz*(C12*L**2*alphay*alphaz + 2*C24*L*alphay*alphaz - 2*C24*L*alphaz + 12*C45)/(2*L**2)
            KC0e0303 = C33/L
            KC0e0304 = betaz*(C23*L*alphaz - 2*C35*alphaz + 2*C35)/(2*L)
            KC0e0305 = betay*(-C13*L*alphay + 2*C34*alphay - 2*C34)/(2*L)
            KC0e0306 = -C03/L
            KC0e0307 = C13*alphay*betay/L
            KC0e0308 = C23*alphaz*betaz/L
            KC0e0309 = -C33/L
            KC0e0310 = betaz*(C23*L*alphaz + 2*C35*alphaz - 2*C35)/(2*L)
            KC0e0311 = betay*(-C13*L*alphay - 2*C34*alphay + 2*C34)/(2*L)
            KC0e0404 = betaz**2*(C22*L**2*alphaz**2/4 - C25*L*alphaz**2 + C25*L*alphaz + C55*alphaz**2 - 2*C55*alphaz + 4*C55)/L
            KC0e0405 = betay*betaz*(-C12*L**2*alphay*alphaz + 2*C15*L*alphay*alphaz - 2*C15*L*alphay + 2*C24*L*alphay*alphaz - 2*C24*L*alphaz - 4*C45*alphay*alphaz + 4*C45*alphay + 4*C45*alphaz - 16*C45)/(4*L)
            KC0e0406 = betaz*(-C02*L*alphaz + 2*C05*alphaz - 2*C05)/(2*L)
            KC0e0407 = betay*betaz*(C12*L**2*alphay*alphaz - 2*C15*L*alphay*alphaz + 2*C15*L*alphay + 12*C45)/(2*L**2)
            KC0e0408 = betaz**2*(C22*L**2*alphaz**2/2 - C25*L*alphaz**2 + C25*L*alphaz + 6*C55)/L**2
            KC0e0409 = betaz*(-C23*L*alphaz + 2*C35*alphaz - 2*C35)/(2*L)
            KC0e0410 = betaz**2*(C22*L**2*alphaz**2/4 - C55*alphaz**2 + 2*C55*alphaz + 2*C55)/L
            KC0e0411 = betay*betaz*(-C12*L**2*alphay*alphaz + 2*C15*L*alphay*alphaz - 2*C15*L*alphay - 2*C24*L*alphay*alphaz + 2*C24*L*alphaz + 4*C45*alphay*alphaz - 4*C45*alphay - 4*C45*alphaz - 8*C45)/(4*L)
            KC0e0505 = betay**2*(C11*L**2*alphay**2/4 - C14*L*alphay**2 + C14*L*alphay + C44*alphay**2 - 2*C44*alphay + 4*C44)/L
            KC0e0506 = betay*(C01*L*alphay - 2*C04*alphay + 2*C04)/(2*L)
            KC0e0507 = betay**2*(-C11*L**2*alphay**2/2 + C14*L*alphay**2 - C14*L*alphay - 6*C44)/L**2
            KC0e0508 = betay*betaz*(-C12*L**2*alphay*alphaz + 2*C24*L*alphay*alphaz - 2*C24*L*alphaz - 12*C45)/(2*L**2)
            KC0e0509 = betay*(C13*L*alphay - 2*C34*alphay + 2*C34)/(2*L)
            KC0e0510 = betay*betaz*(-C12*L**2*alphay*alphaz - 2*C15*L*alphay*alphaz + 2*C15*L*alphay + 2*C24*L*alphay*alphaz - 2*C24*L*alphaz + 4*C45*alphay*alphaz - 4*C45*alphay - 4*C45*alphaz - 8*C45)/(4*L)
            KC0e0511 = betay**2*(C11*L**2*alphay**2/4 - C44*alphay**2 + 2*C44*alphay + 2*C44)/L
            KC0e0606 = C00/L
            KC0e0607 = -C01*alphay*betay/L
            KC0e0608 = -C02*alphaz*betaz/L
            KC0e0609 = C03/L
            KC0e0610 = betaz*(-C02*L*alphaz - 2*C05*alphaz + 2*C05)/(2*L)
            KC0e0611 = betay*(C01*L*alphay + 2*C04*alphay - 2*C04)/(2*L)
            KC0e0707 = betay**2*(C11*L**2*alphay**2 + 12*C44)/L**3
            KC0e0708 = betay*betaz*(C12*L**2*alphay*alphaz + 12*C45)/L**3
            KC0e0709 = -C13*alphay*betay/L
            KC0e0710 = betay*betaz*(C12*L**2*alphay*alphaz + 2*C15*L*alphay*alphaz - 2*C15*L*alphay + 12*C45)/(2*L**2)
            KC0e0711 = betay**2*(-C11*L**2*alphay**2/2 - C14*L*alphay**2 + C14*L*alphay - 6*C44)/L**2
            KC0e0808 = betaz**2*(C22*L**2*alphaz**2 + 12*C55)/L**3
            KC0e0809 = -C23*alphaz*betaz/L
            KC0e0810 = betaz**2*(C22*L**2*alphaz**2/2 + C25*L*alphaz**2 - C25*L*alphaz + 6*C55)/L**2
            KC0e0811 = betay*betaz*(-C12*L**2*alphay*alphaz - 2*C24*L*alphay*alphaz + 2*C24*L*alphaz - 12*C45)/(2*L**2)
            KC0e0909 = C33/L
            KC0e0910 = betaz*(-C23*L*alphaz - 2*C35*alphaz + 2*C35)/(2*L)
            KC0e0911 = betay*(C13*L*alphay + 2*C34*alphay - 2*C34)/(2*L)
            KC0e1010 = betaz**2*(C22*L**2*alphaz**2/4 + C25*L*alphaz**2 - C25*L*alphaz + C55*alphaz**2 - 2*C55*alphaz + 4*C55)/L
            KC0e1011 = betay*betaz*(-C12*L**2*alphay*alphaz - 2*C15*L*alphay*alphaz + 2*C15*L*alphay - 2*C24*L*alphay*alphaz + 2*C24*L*alphaz - 4*C45*alphay*alphaz + 4*C45*alphay + 4*C45*alphaz - 16*C45)/(4*L)
            KC0e1111 = betay**2*(C11*L**2*alphay**2/4 + C14*L*alphay**2 - C14*L*alphay + C44*alphay**2 - 2*C44*alphay + 4*C44)/L

            ue = &self.probe.ue[0]
            finte = &self.probe.finte[0]

            finte[0] = KC0e0000*ue[0] + KC0e0001*ue[1] + KC0e0002*ue[2] + KC0e0003*ue[3] + KC0e0004*ue[4] + KC0e0005*ue[5] + KC0e0006*ue[6] + KC0e0007*ue[7] + KC0e0008*ue[8] + KC0e0009*ue[9] + KC0e0010*ue[10] + KC0e0011*ue[11]
            finte[1] = KC0e0001*ue[0] + KC0e0101*ue[1] + KC0e0102*ue[2] + KC0e0103*ue[3] + KC0e0104*ue[4] + KC0e0105*ue[5] + KC0e0106*ue[6] + KC0e0107*ue[7] + KC0e0108*ue[8] + KC0e0109*ue[9] + KC0e0110*ue[10] + KC0e0111*ue[11]
            finte[2] = KC0e0002*ue[0] + KC0e0102*ue[1] + KC0e0202*ue[2] + KC0e0203*ue[3] + KC0e0204*ue[4] + KC0e0205*ue[5] + KC0e0206*ue[6] + KC0e0207*ue[7] + KC0e0208*ue[8] + KC0e0209*ue[9] + KC0e0210*ue[10] + KC0e0211*ue[11]
            finte[3] = KC0e0003*ue[0] + KC0e0103*ue[1] + KC0e0203*ue[2] + KC0e0303*ue[3] + KC0e0304*ue[4] + KC0e0305*ue[5] + KC0e0306*ue[6] + KC0e0307*ue[7] + KC0e0308*ue[8] + KC0e0309*ue[9] + KC0e0310*ue[10] + KC0e0311*ue[11]
            finte[4] = KC0e0004*ue[0] + KC0e0104*ue[1] + KC0e0204*ue[2] + KC0e0304*ue[3] + KC0e0404*ue[4] + KC0e0405*ue[5] + KC0e0406*ue[6] + KC0e0407*ue[7] + KC0e0408*ue[8] + KC0e0409*ue[9] + KC0e0410*ue[10] + KC0e0411*ue[11]
            finte[5] = KC0e0005*ue[0] + KC0e0105*ue[1] + KC0e0205*ue[2] + KC0e0305*ue[3] + KC0e0405*ue[4] + KC0e0505*ue[5] + KC0e0506*ue[6] + KC0e0507*ue[7] + KC0e0508*ue[8] + KC0e0509*ue[9] + KC0e0510*ue[10] + KC0e0511*ue[11]
            finte[6] = KC0e0006*ue[0] + KC0e0106*ue[1] + KC0e0206*ue[2] + KC0e0306*ue[3] + KC0e0406*ue[4] + KC0e0506*ue[5] + KC0e0606*ue[6] + KC0e0607*ue[7] + KC0e0608*ue[8] + KC0e0609*ue[9] + KC0e0610*ue[10] + KC0e0611*ue[11]
            finte[7] = KC0e0007*ue[0] + KC0e0107*ue[1] + KC0e0207*ue[2] + KC0e0307*ue[3] + KC0e0407*ue[4] + KC0e0507*ue[5] + KC0e0607*ue[6] + KC0e0707*ue[7] + KC0e0708*ue[8] + KC0e0709*ue[9] + KC0e0710*ue[10] + KC0e0711*ue[11]
            finte[8] = KC0e0008*ue[0] + KC0e0108*ue[1] + KC0e0208*ue[2] + KC0e0308*ue[3] + KC0e0408*ue[4] + KC0e0508*ue[5] + KC0e0608*ue[6] + KC0e0708*ue[7] + KC0e0808*ue[8] + KC0e0809*ue[9] + KC0e0810*ue[10] + KC0e0811*ue[11]
            finte[9] = KC0e0009*ue[0] + KC0e0109*ue[1] + KC0e0209*ue[2] + KC0e0309*ue[3] + KC0e0409*ue[4] + KC0e0509*ue[5] + KC0e0609*ue[6] + KC0e0709*ue[7] + KC0e0809*ue[8] + KC0e0909*ue[9] + KC0e0910*ue[10] + KC0e0911*ue[11]
            finte[10] = KC0e0010*ue[0] + KC0e0110*ue[1] + KC0e0210*ue[2] + KC0e0310*ue[3] + KC0e0410*ue[4] + KC0e0510*ue[5] + KC0e0610*ue[6] + KC0e0710*ue[7] + KC0e0810*ue[8] + KC0e0910*ue[9] + KC0e1010*ue[10] + KC0e1011*ue[11]
            finte[11] = KC0e0011*ue[0] + KC0e0111*ue[1] + KC0e0211*ue[2] + KC0e0311*ue[3] + KC0e0411*ue[4] + KC0e0511*ue[5] + KC0e0611*ue[6] + KC0e0711*ue[7] + KC0e0811*ue[8] + KC0e0911*ue[9] + KC0e1011*ue[10] + KC0e1111*ue[11]

            if nonlinear:
                self._update_probe_finte_nonlinear(prop)


    cpdef void update_KC0(BeamC self,
                          long [::1] KC0r,
                          long [::1] KC0c,
                          double [::1] KC0v,
                          BeamProp prop,
                          int update_KC0v_only=0
                          ):
        r"""Update sparse vectors for linear constitutive stiffness matrix KC0

        The 144 entries of the matrix of the element, in global coordinates,
        are written at the positions ``init_k_KC0`` to ``init_k_KC0 + 143``:
        the row and column indices, built from ``c1`` and ``c2``, are stored
        in ``KC0r`` and ``KC0c``, and the values are added to those already
        in ``KC0v``. The matrix is computed from ``length``, the rotation
        matrix and the sectional stiffness matrix ``C`` of ``prop``, see
        :class:`.BeamProp`. ``length`` and the
        rotation matrix must be updated before, see
        :meth:`.update_rotation_matrix` and :meth:`.update_probe_xe`.

        Parameters
        ----------
        KC0r : np.array
            Array to store row positions of sparse values. 1-D array of
            integers of the C type ``long``, of size at least
            ``init_k_KC0 + 144``.
        KC0c : np.array
            Array to store column positions of sparse values. Same type and
            size as ``KC0r``.
        KC0v : np.array
            Array to store sparse values. 1-D array of floats of size at least
            ``init_k_KC0 + 144``.
        prop : :class:`.BeamProp` object
            Beam property object from where the stiffness and mass attributes
            are read from.
        update_KC0v_only : int, optional
            The default ``0`` means that the row and column indices ``KC0r``
            and ``KC0c`` should also be updated. Any other value will only
            update the stiffness matrix values ``KC0v``.

        Notes
        -----
        ``KC0`` is the Hessian, with respect to the element displacements
        ``ue``, of the linear strain energy given in the notes of
        :class:`.BeamC`, integrated analytically along the element with the
        consistent shape functions, and transformed to global coordinates as
        described there.

        """
        cdef int c1, c2, k
        cdef double L
        cdef double r11, r12, r13, r21, r22, r23, r31, r32, r33
        cdef double alphay, alphaz, betay, betaz
        cdef double C[36]
        cdef double C00, C01, C02, C03, C04, C05, C11, C12, C13, C14, C15, C22
        cdef double C23, C24, C25, C33, C34, C35, C44, C45, C55

        cdef double KC0e0000, KC0e0001, KC0e0002, KC0e0003, KC0e0004, KC0e0005
        cdef double KC0e0006, KC0e0007, KC0e0008, KC0e0009, KC0e0010, KC0e0011
        cdef double KC0e0101, KC0e0102, KC0e0103, KC0e0104, KC0e0105, KC0e0106
        cdef double KC0e0107, KC0e0108, KC0e0109, KC0e0110, KC0e0111, KC0e0202
        cdef double KC0e0203, KC0e0204, KC0e0205, KC0e0206, KC0e0207, KC0e0208
        cdef double KC0e0209, KC0e0210, KC0e0211, KC0e0303, KC0e0304, KC0e0305
        cdef double KC0e0306, KC0e0307, KC0e0308, KC0e0309, KC0e0310, KC0e0311
        cdef double KC0e0404, KC0e0405, KC0e0406, KC0e0407, KC0e0408, KC0e0409
        cdef double KC0e0410, KC0e0411, KC0e0505, KC0e0506, KC0e0507, KC0e0508
        cdef double KC0e0509, KC0e0510, KC0e0511, KC0e0606, KC0e0607, KC0e0608
        cdef double KC0e0609, KC0e0610, KC0e0611, KC0e0707, KC0e0708, KC0e0709
        cdef double KC0e0710, KC0e0711, KC0e0808, KC0e0809, KC0e0810, KC0e0811
        cdef double KC0e0909, KC0e0910, KC0e0911, KC0e1010, KC0e1011, KC0e1111

        with nogil:
            L = self.length
            prop.get_C(C)
            C00 = C[0]
            C01 = C[1]
            C02 = C[2]
            C03 = C[3]
            C04 = C[4]
            C05 = C[5]
            C11 = C[7]
            C12 = C[8]
            C13 = C[9]
            C14 = C[10]
            C15 = C[11]
            C22 = C[14]
            C23 = C[15]
            C24 = C[16]
            C25 = C[17]
            C33 = C[21]
            C34 = C[22]
            C35 = C[23]
            C44 = C[28]
            C45 = C[29]
            C55 = C[35]

            # NOTE shear parameters of the shape functions from the diagonal of
            #      the sectional stiffness matrix, see the notes of BeamC
            alphay = -12*C44/(C11*L**2)
            alphaz = -12*C55/(C22*L**2)
            betay = 1/(1. - alphay)
            betaz = 1/(1. - alphaz)

            # local to global transformation
            r11 = self.r11
            r12 = self.r12
            r13 = self.r13
            r21 = self.r21
            r22 = self.r22
            r23 = self.r23
            r31 = self.r31
            r32 = self.r32
            r33 = self.r33

            KC0e0000 = C00/L
            KC0e0001 = -C01*alphay*betay/L
            KC0e0002 = -C02*alphaz*betaz/L
            KC0e0003 = C03/L
            KC0e0004 = betaz*(C02*L*alphaz - 2*C05*alphaz + 2*C05)/(2*L)
            KC0e0005 = betay*(-C01*L*alphay + 2*C04*alphay - 2*C04)/(2*L)
            KC0e0006 = -C00/L
            KC0e0007 = C01*alphay*betay/L
            KC0e0008 = C02*alphaz*betaz/L
            KC0e0009 = -C03/L
            KC0e0010 = betaz*(C02*L*alphaz + 2*C05*alphaz - 2*C05)/(2*L)
            KC0e0011 = betay*(-C01*L*alphay - 2*C04*alphay + 2*C04)/(2*L)
            KC0e0101 = betay**2*(C11*L**2*alphay**2 + 12*C44)/L**3
            KC0e0102 = betay*betaz*(C12*L**2*alphay*alphaz + 12*C45)/L**3
            KC0e0103 = -C13*alphay*betay/L
            KC0e0104 = betay*betaz*(-C12*L**2*alphay*alphaz + 2*C15*L*alphay*alphaz - 2*C15*L*alphay - 12*C45)/(2*L**2)
            KC0e0105 = betay**2*(C11*L**2*alphay**2/2 - C14*L*alphay**2 + C14*L*alphay + 6*C44)/L**2
            KC0e0106 = C01*alphay*betay/L
            KC0e0107 = betay**2*(-C11*L**2*alphay**2 - 12*C44)/L**3
            KC0e0108 = betay*betaz*(-C12*L**2*alphay*alphaz - 12*C45)/L**3
            KC0e0109 = C13*alphay*betay/L
            KC0e0110 = betay*betaz*(-C12*L**2*alphay*alphaz - 2*C15*L*alphay*alphaz + 2*C15*L*alphay - 12*C45)/(2*L**2)
            KC0e0111 = betay**2*(C11*L**2*alphay**2/2 + C14*L*alphay**2 - C14*L*alphay + 6*C44)/L**2
            KC0e0202 = betaz**2*(C22*L**2*alphaz**2 + 12*C55)/L**3
            KC0e0203 = -C23*alphaz*betaz/L
            KC0e0204 = betaz**2*(-C22*L**2*alphaz**2/2 + C25*L*alphaz**2 - C25*L*alphaz - 6*C55)/L**2
            KC0e0205 = betay*betaz*(C12*L**2*alphay*alphaz - 2*C24*L*alphay*alphaz + 2*C24*L*alphaz + 12*C45)/(2*L**2)
            KC0e0206 = C02*alphaz*betaz/L
            KC0e0207 = betay*betaz*(-C12*L**2*alphay*alphaz - 12*C45)/L**3
            KC0e0208 = betaz**2*(-C22*L**2*alphaz**2 - 12*C55)/L**3
            KC0e0209 = C23*alphaz*betaz/L
            KC0e0210 = betaz**2*(-C22*L**2*alphaz**2/2 - C25*L*alphaz**2 + C25*L*alphaz - 6*C55)/L**2
            KC0e0211 = betay*betaz*(C12*L**2*alphay*alphaz + 2*C24*L*alphay*alphaz - 2*C24*L*alphaz + 12*C45)/(2*L**2)
            KC0e0303 = C33/L
            KC0e0304 = betaz*(C23*L*alphaz - 2*C35*alphaz + 2*C35)/(2*L)
            KC0e0305 = betay*(-C13*L*alphay + 2*C34*alphay - 2*C34)/(2*L)
            KC0e0306 = -C03/L
            KC0e0307 = C13*alphay*betay/L
            KC0e0308 = C23*alphaz*betaz/L
            KC0e0309 = -C33/L
            KC0e0310 = betaz*(C23*L*alphaz + 2*C35*alphaz - 2*C35)/(2*L)
            KC0e0311 = betay*(-C13*L*alphay - 2*C34*alphay + 2*C34)/(2*L)
            KC0e0404 = betaz**2*(C22*L**2*alphaz**2/4 - C25*L*alphaz**2 + C25*L*alphaz + C55*alphaz**2 - 2*C55*alphaz + 4*C55)/L
            KC0e0405 = betay*betaz*(-C12*L**2*alphay*alphaz + 2*C15*L*alphay*alphaz - 2*C15*L*alphay + 2*C24*L*alphay*alphaz - 2*C24*L*alphaz - 4*C45*alphay*alphaz + 4*C45*alphay + 4*C45*alphaz - 16*C45)/(4*L)
            KC0e0406 = betaz*(-C02*L*alphaz + 2*C05*alphaz - 2*C05)/(2*L)
            KC0e0407 = betay*betaz*(C12*L**2*alphay*alphaz - 2*C15*L*alphay*alphaz + 2*C15*L*alphay + 12*C45)/(2*L**2)
            KC0e0408 = betaz**2*(C22*L**2*alphaz**2/2 - C25*L*alphaz**2 + C25*L*alphaz + 6*C55)/L**2
            KC0e0409 = betaz*(-C23*L*alphaz + 2*C35*alphaz - 2*C35)/(2*L)
            KC0e0410 = betaz**2*(C22*L**2*alphaz**2/4 - C55*alphaz**2 + 2*C55*alphaz + 2*C55)/L
            KC0e0411 = betay*betaz*(-C12*L**2*alphay*alphaz + 2*C15*L*alphay*alphaz - 2*C15*L*alphay - 2*C24*L*alphay*alphaz + 2*C24*L*alphaz + 4*C45*alphay*alphaz - 4*C45*alphay - 4*C45*alphaz - 8*C45)/(4*L)
            KC0e0505 = betay**2*(C11*L**2*alphay**2/4 - C14*L*alphay**2 + C14*L*alphay + C44*alphay**2 - 2*C44*alphay + 4*C44)/L
            KC0e0506 = betay*(C01*L*alphay - 2*C04*alphay + 2*C04)/(2*L)
            KC0e0507 = betay**2*(-C11*L**2*alphay**2/2 + C14*L*alphay**2 - C14*L*alphay - 6*C44)/L**2
            KC0e0508 = betay*betaz*(-C12*L**2*alphay*alphaz + 2*C24*L*alphay*alphaz - 2*C24*L*alphaz - 12*C45)/(2*L**2)
            KC0e0509 = betay*(C13*L*alphay - 2*C34*alphay + 2*C34)/(2*L)
            KC0e0510 = betay*betaz*(-C12*L**2*alphay*alphaz - 2*C15*L*alphay*alphaz + 2*C15*L*alphay + 2*C24*L*alphay*alphaz - 2*C24*L*alphaz + 4*C45*alphay*alphaz - 4*C45*alphay - 4*C45*alphaz - 8*C45)/(4*L)
            KC0e0511 = betay**2*(C11*L**2*alphay**2/4 - C44*alphay**2 + 2*C44*alphay + 2*C44)/L
            KC0e0606 = C00/L
            KC0e0607 = -C01*alphay*betay/L
            KC0e0608 = -C02*alphaz*betaz/L
            KC0e0609 = C03/L
            KC0e0610 = betaz*(-C02*L*alphaz - 2*C05*alphaz + 2*C05)/(2*L)
            KC0e0611 = betay*(C01*L*alphay + 2*C04*alphay - 2*C04)/(2*L)
            KC0e0707 = betay**2*(C11*L**2*alphay**2 + 12*C44)/L**3
            KC0e0708 = betay*betaz*(C12*L**2*alphay*alphaz + 12*C45)/L**3
            KC0e0709 = -C13*alphay*betay/L
            KC0e0710 = betay*betaz*(C12*L**2*alphay*alphaz + 2*C15*L*alphay*alphaz - 2*C15*L*alphay + 12*C45)/(2*L**2)
            KC0e0711 = betay**2*(-C11*L**2*alphay**2/2 - C14*L*alphay**2 + C14*L*alphay - 6*C44)/L**2
            KC0e0808 = betaz**2*(C22*L**2*alphaz**2 + 12*C55)/L**3
            KC0e0809 = -C23*alphaz*betaz/L
            KC0e0810 = betaz**2*(C22*L**2*alphaz**2/2 + C25*L*alphaz**2 - C25*L*alphaz + 6*C55)/L**2
            KC0e0811 = betay*betaz*(-C12*L**2*alphay*alphaz - 2*C24*L*alphay*alphaz + 2*C24*L*alphaz - 12*C45)/(2*L**2)
            KC0e0909 = C33/L
            KC0e0910 = betaz*(-C23*L*alphaz - 2*C35*alphaz + 2*C35)/(2*L)
            KC0e0911 = betay*(C13*L*alphay + 2*C34*alphay - 2*C34)/(2*L)
            KC0e1010 = betaz**2*(C22*L**2*alphaz**2/4 + C25*L*alphaz**2 - C25*L*alphaz + C55*alphaz**2 - 2*C55*alphaz + 4*C55)/L
            KC0e1011 = betay*betaz*(-C12*L**2*alphay*alphaz - 2*C15*L*alphay*alphaz + 2*C15*L*alphay - 2*C24*L*alphay*alphaz + 2*C24*L*alphaz - 4*C45*alphay*alphaz + 4*C45*alphay + 4*C45*alphaz - 16*C45)/(4*L)
            KC0e1111 = betay**2*(C11*L**2*alphay**2/4 + C14*L*alphay**2 - C14*L*alphay + C44*alphay**2 - 2*C44*alphay + 4*C44)/L

            if update_KC0v_only == 0:
                # positions in the global stiffness matrix
                c1 = self.c1
                c2 = self.c2

                k = self.init_k_KC0
                KC0r[k] = 0+c1
                KC0c[k] = 0+c1
                k += 1
                KC0r[k] = 0+c1
                KC0c[k] = 1+c1
                k += 1
                KC0r[k] = 0+c1
                KC0c[k] = 2+c1
                k += 1
                KC0r[k] = 0+c1
                KC0c[k] = 3+c1
                k += 1
                KC0r[k] = 0+c1
                KC0c[k] = 4+c1
                k += 1
                KC0r[k] = 0+c1
                KC0c[k] = 5+c1
                k += 1
                KC0r[k] = 0+c1
                KC0c[k] = 0+c2
                k += 1
                KC0r[k] = 0+c1
                KC0c[k] = 1+c2
                k += 1
                KC0r[k] = 0+c1
                KC0c[k] = 2+c2
                k += 1
                KC0r[k] = 0+c1
                KC0c[k] = 3+c2
                k += 1
                KC0r[k] = 0+c1
                KC0c[k] = 4+c2
                k += 1
                KC0r[k] = 0+c1
                KC0c[k] = 5+c2
                k += 1
                KC0r[k] = 1+c1
                KC0c[k] = 0+c1
                k += 1
                KC0r[k] = 1+c1
                KC0c[k] = 1+c1
                k += 1
                KC0r[k] = 1+c1
                KC0c[k] = 2+c1
                k += 1
                KC0r[k] = 1+c1
                KC0c[k] = 3+c1
                k += 1
                KC0r[k] = 1+c1
                KC0c[k] = 4+c1
                k += 1
                KC0r[k] = 1+c1
                KC0c[k] = 5+c1
                k += 1
                KC0r[k] = 1+c1
                KC0c[k] = 0+c2
                k += 1
                KC0r[k] = 1+c1
                KC0c[k] = 1+c2
                k += 1
                KC0r[k] = 1+c1
                KC0c[k] = 2+c2
                k += 1
                KC0r[k] = 1+c1
                KC0c[k] = 3+c2
                k += 1
                KC0r[k] = 1+c1
                KC0c[k] = 4+c2
                k += 1
                KC0r[k] = 1+c1
                KC0c[k] = 5+c2
                k += 1
                KC0r[k] = 2+c1
                KC0c[k] = 0+c1
                k += 1
                KC0r[k] = 2+c1
                KC0c[k] = 1+c1
                k += 1
                KC0r[k] = 2+c1
                KC0c[k] = 2+c1
                k += 1
                KC0r[k] = 2+c1
                KC0c[k] = 3+c1
                k += 1
                KC0r[k] = 2+c1
                KC0c[k] = 4+c1
                k += 1
                KC0r[k] = 2+c1
                KC0c[k] = 5+c1
                k += 1
                KC0r[k] = 2+c1
                KC0c[k] = 0+c2
                k += 1
                KC0r[k] = 2+c1
                KC0c[k] = 1+c2
                k += 1
                KC0r[k] = 2+c1
                KC0c[k] = 2+c2
                k += 1
                KC0r[k] = 2+c1
                KC0c[k] = 3+c2
                k += 1
                KC0r[k] = 2+c1
                KC0c[k] = 4+c2
                k += 1
                KC0r[k] = 2+c1
                KC0c[k] = 5+c2
                k += 1
                KC0r[k] = 3+c1
                KC0c[k] = 0+c1
                k += 1
                KC0r[k] = 3+c1
                KC0c[k] = 1+c1
                k += 1
                KC0r[k] = 3+c1
                KC0c[k] = 2+c1
                k += 1
                KC0r[k] = 3+c1
                KC0c[k] = 3+c1
                k += 1
                KC0r[k] = 3+c1
                KC0c[k] = 4+c1
                k += 1
                KC0r[k] = 3+c1
                KC0c[k] = 5+c1
                k += 1
                KC0r[k] = 3+c1
                KC0c[k] = 0+c2
                k += 1
                KC0r[k] = 3+c1
                KC0c[k] = 1+c2
                k += 1
                KC0r[k] = 3+c1
                KC0c[k] = 2+c2
                k += 1
                KC0r[k] = 3+c1
                KC0c[k] = 3+c2
                k += 1
                KC0r[k] = 3+c1
                KC0c[k] = 4+c2
                k += 1
                KC0r[k] = 3+c1
                KC0c[k] = 5+c2
                k += 1
                KC0r[k] = 4+c1
                KC0c[k] = 0+c1
                k += 1
                KC0r[k] = 4+c1
                KC0c[k] = 1+c1
                k += 1
                KC0r[k] = 4+c1
                KC0c[k] = 2+c1
                k += 1
                KC0r[k] = 4+c1
                KC0c[k] = 3+c1
                k += 1
                KC0r[k] = 4+c1
                KC0c[k] = 4+c1
                k += 1
                KC0r[k] = 4+c1
                KC0c[k] = 5+c1
                k += 1
                KC0r[k] = 4+c1
                KC0c[k] = 0+c2
                k += 1
                KC0r[k] = 4+c1
                KC0c[k] = 1+c2
                k += 1
                KC0r[k] = 4+c1
                KC0c[k] = 2+c2
                k += 1
                KC0r[k] = 4+c1
                KC0c[k] = 3+c2
                k += 1
                KC0r[k] = 4+c1
                KC0c[k] = 4+c2
                k += 1
                KC0r[k] = 4+c1
                KC0c[k] = 5+c2
                k += 1
                KC0r[k] = 5+c1
                KC0c[k] = 0+c1
                k += 1
                KC0r[k] = 5+c1
                KC0c[k] = 1+c1
                k += 1
                KC0r[k] = 5+c1
                KC0c[k] = 2+c1
                k += 1
                KC0r[k] = 5+c1
                KC0c[k] = 3+c1
                k += 1
                KC0r[k] = 5+c1
                KC0c[k] = 4+c1
                k += 1
                KC0r[k] = 5+c1
                KC0c[k] = 5+c1
                k += 1
                KC0r[k] = 5+c1
                KC0c[k] = 0+c2
                k += 1
                KC0r[k] = 5+c1
                KC0c[k] = 1+c2
                k += 1
                KC0r[k] = 5+c1
                KC0c[k] = 2+c2
                k += 1
                KC0r[k] = 5+c1
                KC0c[k] = 3+c2
                k += 1
                KC0r[k] = 5+c1
                KC0c[k] = 4+c2
                k += 1
                KC0r[k] = 5+c1
                KC0c[k] = 5+c2
                k += 1
                KC0r[k] = 0+c2
                KC0c[k] = 0+c1
                k += 1
                KC0r[k] = 0+c2
                KC0c[k] = 1+c1
                k += 1
                KC0r[k] = 0+c2
                KC0c[k] = 2+c1
                k += 1
                KC0r[k] = 0+c2
                KC0c[k] = 3+c1
                k += 1
                KC0r[k] = 0+c2
                KC0c[k] = 4+c1
                k += 1
                KC0r[k] = 0+c2
                KC0c[k] = 5+c1
                k += 1
                KC0r[k] = 0+c2
                KC0c[k] = 0+c2
                k += 1
                KC0r[k] = 0+c2
                KC0c[k] = 1+c2
                k += 1
                KC0r[k] = 0+c2
                KC0c[k] = 2+c2
                k += 1
                KC0r[k] = 0+c2
                KC0c[k] = 3+c2
                k += 1
                KC0r[k] = 0+c2
                KC0c[k] = 4+c2
                k += 1
                KC0r[k] = 0+c2
                KC0c[k] = 5+c2
                k += 1
                KC0r[k] = 1+c2
                KC0c[k] = 0+c1
                k += 1
                KC0r[k] = 1+c2
                KC0c[k] = 1+c1
                k += 1
                KC0r[k] = 1+c2
                KC0c[k] = 2+c1
                k += 1
                KC0r[k] = 1+c2
                KC0c[k] = 3+c1
                k += 1
                KC0r[k] = 1+c2
                KC0c[k] = 4+c1
                k += 1
                KC0r[k] = 1+c2
                KC0c[k] = 5+c1
                k += 1
                KC0r[k] = 1+c2
                KC0c[k] = 0+c2
                k += 1
                KC0r[k] = 1+c2
                KC0c[k] = 1+c2
                k += 1
                KC0r[k] = 1+c2
                KC0c[k] = 2+c2
                k += 1
                KC0r[k] = 1+c2
                KC0c[k] = 3+c2
                k += 1
                KC0r[k] = 1+c2
                KC0c[k] = 4+c2
                k += 1
                KC0r[k] = 1+c2
                KC0c[k] = 5+c2
                k += 1
                KC0r[k] = 2+c2
                KC0c[k] = 0+c1
                k += 1
                KC0r[k] = 2+c2
                KC0c[k] = 1+c1
                k += 1
                KC0r[k] = 2+c2
                KC0c[k] = 2+c1
                k += 1
                KC0r[k] = 2+c2
                KC0c[k] = 3+c1
                k += 1
                KC0r[k] = 2+c2
                KC0c[k] = 4+c1
                k += 1
                KC0r[k] = 2+c2
                KC0c[k] = 5+c1
                k += 1
                KC0r[k] = 2+c2
                KC0c[k] = 0+c2
                k += 1
                KC0r[k] = 2+c2
                KC0c[k] = 1+c2
                k += 1
                KC0r[k] = 2+c2
                KC0c[k] = 2+c2
                k += 1
                KC0r[k] = 2+c2
                KC0c[k] = 3+c2
                k += 1
                KC0r[k] = 2+c2
                KC0c[k] = 4+c2
                k += 1
                KC0r[k] = 2+c2
                KC0c[k] = 5+c2
                k += 1
                KC0r[k] = 3+c2
                KC0c[k] = 0+c1
                k += 1
                KC0r[k] = 3+c2
                KC0c[k] = 1+c1
                k += 1
                KC0r[k] = 3+c2
                KC0c[k] = 2+c1
                k += 1
                KC0r[k] = 3+c2
                KC0c[k] = 3+c1
                k += 1
                KC0r[k] = 3+c2
                KC0c[k] = 4+c1
                k += 1
                KC0r[k] = 3+c2
                KC0c[k] = 5+c1
                k += 1
                KC0r[k] = 3+c2
                KC0c[k] = 0+c2
                k += 1
                KC0r[k] = 3+c2
                KC0c[k] = 1+c2
                k += 1
                KC0r[k] = 3+c2
                KC0c[k] = 2+c2
                k += 1
                KC0r[k] = 3+c2
                KC0c[k] = 3+c2
                k += 1
                KC0r[k] = 3+c2
                KC0c[k] = 4+c2
                k += 1
                KC0r[k] = 3+c2
                KC0c[k] = 5+c2
                k += 1
                KC0r[k] = 4+c2
                KC0c[k] = 0+c1
                k += 1
                KC0r[k] = 4+c2
                KC0c[k] = 1+c1
                k += 1
                KC0r[k] = 4+c2
                KC0c[k] = 2+c1
                k += 1
                KC0r[k] = 4+c2
                KC0c[k] = 3+c1
                k += 1
                KC0r[k] = 4+c2
                KC0c[k] = 4+c1
                k += 1
                KC0r[k] = 4+c2
                KC0c[k] = 5+c1
                k += 1
                KC0r[k] = 4+c2
                KC0c[k] = 0+c2
                k += 1
                KC0r[k] = 4+c2
                KC0c[k] = 1+c2
                k += 1
                KC0r[k] = 4+c2
                KC0c[k] = 2+c2
                k += 1
                KC0r[k] = 4+c2
                KC0c[k] = 3+c2
                k += 1
                KC0r[k] = 4+c2
                KC0c[k] = 4+c2
                k += 1
                KC0r[k] = 4+c2
                KC0c[k] = 5+c2
                k += 1
                KC0r[k] = 5+c2
                KC0c[k] = 0+c1
                k += 1
                KC0r[k] = 5+c2
                KC0c[k] = 1+c1
                k += 1
                KC0r[k] = 5+c2
                KC0c[k] = 2+c1
                k += 1
                KC0r[k] = 5+c2
                KC0c[k] = 3+c1
                k += 1
                KC0r[k] = 5+c2
                KC0c[k] = 4+c1
                k += 1
                KC0r[k] = 5+c2
                KC0c[k] = 5+c1
                k += 1
                KC0r[k] = 5+c2
                KC0c[k] = 0+c2
                k += 1
                KC0r[k] = 5+c2
                KC0c[k] = 1+c2
                k += 1
                KC0r[k] = 5+c2
                KC0c[k] = 2+c2
                k += 1
                KC0r[k] = 5+c2
                KC0c[k] = 3+c2
                k += 1
                KC0r[k] = 5+c2
                KC0c[k] = 4+c2
                k += 1
                KC0r[k] = 5+c2
                KC0c[k] = 5+c2

            k = self.init_k_KC0
            KC0v[k] += r11*(KC0e0000*r11 + KC0e0001*r12 + KC0e0002*r13) + r12*(KC0e0001*r11 + KC0e0101*r12 + KC0e0102*r13) + r13*(KC0e0002*r11 + KC0e0102*r12 + KC0e0202*r13)
            k += 1
            KC0v[k] += r21*(KC0e0000*r11 + KC0e0001*r12 + KC0e0002*r13) + r22*(KC0e0001*r11 + KC0e0101*r12 + KC0e0102*r13) + r23*(KC0e0002*r11 + KC0e0102*r12 + KC0e0202*r13)
            k += 1
            KC0v[k] += r31*(KC0e0000*r11 + KC0e0001*r12 + KC0e0002*r13) + r32*(KC0e0001*r11 + KC0e0101*r12 + KC0e0102*r13) + r33*(KC0e0002*r11 + KC0e0102*r12 + KC0e0202*r13)
            k += 1
            KC0v[k] += r11*(KC0e0003*r11 + KC0e0103*r12 + KC0e0203*r13) + r12*(KC0e0004*r11 + KC0e0104*r12 + KC0e0204*r13) + r13*(KC0e0005*r11 + KC0e0105*r12 + KC0e0205*r13)
            k += 1
            KC0v[k] += r21*(KC0e0003*r11 + KC0e0103*r12 + KC0e0203*r13) + r22*(KC0e0004*r11 + KC0e0104*r12 + KC0e0204*r13) + r23*(KC0e0005*r11 + KC0e0105*r12 + KC0e0205*r13)
            k += 1
            KC0v[k] += r31*(KC0e0003*r11 + KC0e0103*r12 + KC0e0203*r13) + r32*(KC0e0004*r11 + KC0e0104*r12 + KC0e0204*r13) + r33*(KC0e0005*r11 + KC0e0105*r12 + KC0e0205*r13)
            k += 1
            KC0v[k] += r11*(KC0e0006*r11 + KC0e0106*r12 + KC0e0206*r13) + r12*(KC0e0007*r11 + KC0e0107*r12 + KC0e0207*r13) + r13*(KC0e0008*r11 + KC0e0108*r12 + KC0e0208*r13)
            k += 1
            KC0v[k] += r21*(KC0e0006*r11 + KC0e0106*r12 + KC0e0206*r13) + r22*(KC0e0007*r11 + KC0e0107*r12 + KC0e0207*r13) + r23*(KC0e0008*r11 + KC0e0108*r12 + KC0e0208*r13)
            k += 1
            KC0v[k] += r31*(KC0e0006*r11 + KC0e0106*r12 + KC0e0206*r13) + r32*(KC0e0007*r11 + KC0e0107*r12 + KC0e0207*r13) + r33*(KC0e0008*r11 + KC0e0108*r12 + KC0e0208*r13)
            k += 1
            KC0v[k] += r11*(KC0e0009*r11 + KC0e0109*r12 + KC0e0209*r13) + r12*(KC0e0010*r11 + KC0e0110*r12 + KC0e0210*r13) + r13*(KC0e0011*r11 + KC0e0111*r12 + KC0e0211*r13)
            k += 1
            KC0v[k] += r21*(KC0e0009*r11 + KC0e0109*r12 + KC0e0209*r13) + r22*(KC0e0010*r11 + KC0e0110*r12 + KC0e0210*r13) + r23*(KC0e0011*r11 + KC0e0111*r12 + KC0e0211*r13)
            k += 1
            KC0v[k] += r31*(KC0e0009*r11 + KC0e0109*r12 + KC0e0209*r13) + r32*(KC0e0010*r11 + KC0e0110*r12 + KC0e0210*r13) + r33*(KC0e0011*r11 + KC0e0111*r12 + KC0e0211*r13)
            k += 1
            KC0v[k] += r11*(KC0e0000*r21 + KC0e0001*r22 + KC0e0002*r23) + r12*(KC0e0001*r21 + KC0e0101*r22 + KC0e0102*r23) + r13*(KC0e0002*r21 + KC0e0102*r22 + KC0e0202*r23)
            k += 1
            KC0v[k] += r21*(KC0e0000*r21 + KC0e0001*r22 + KC0e0002*r23) + r22*(KC0e0001*r21 + KC0e0101*r22 + KC0e0102*r23) + r23*(KC0e0002*r21 + KC0e0102*r22 + KC0e0202*r23)
            k += 1
            KC0v[k] += r31*(KC0e0000*r21 + KC0e0001*r22 + KC0e0002*r23) + r32*(KC0e0001*r21 + KC0e0101*r22 + KC0e0102*r23) + r33*(KC0e0002*r21 + KC0e0102*r22 + KC0e0202*r23)
            k += 1
            KC0v[k] += r11*(KC0e0003*r21 + KC0e0103*r22 + KC0e0203*r23) + r12*(KC0e0004*r21 + KC0e0104*r22 + KC0e0204*r23) + r13*(KC0e0005*r21 + KC0e0105*r22 + KC0e0205*r23)
            k += 1
            KC0v[k] += r21*(KC0e0003*r21 + KC0e0103*r22 + KC0e0203*r23) + r22*(KC0e0004*r21 + KC0e0104*r22 + KC0e0204*r23) + r23*(KC0e0005*r21 + KC0e0105*r22 + KC0e0205*r23)
            k += 1
            KC0v[k] += r31*(KC0e0003*r21 + KC0e0103*r22 + KC0e0203*r23) + r32*(KC0e0004*r21 + KC0e0104*r22 + KC0e0204*r23) + r33*(KC0e0005*r21 + KC0e0105*r22 + KC0e0205*r23)
            k += 1
            KC0v[k] += r11*(KC0e0006*r21 + KC0e0106*r22 + KC0e0206*r23) + r12*(KC0e0007*r21 + KC0e0107*r22 + KC0e0207*r23) + r13*(KC0e0008*r21 + KC0e0108*r22 + KC0e0208*r23)
            k += 1
            KC0v[k] += r21*(KC0e0006*r21 + KC0e0106*r22 + KC0e0206*r23) + r22*(KC0e0007*r21 + KC0e0107*r22 + KC0e0207*r23) + r23*(KC0e0008*r21 + KC0e0108*r22 + KC0e0208*r23)
            k += 1
            KC0v[k] += r31*(KC0e0006*r21 + KC0e0106*r22 + KC0e0206*r23) + r32*(KC0e0007*r21 + KC0e0107*r22 + KC0e0207*r23) + r33*(KC0e0008*r21 + KC0e0108*r22 + KC0e0208*r23)
            k += 1
            KC0v[k] += r11*(KC0e0009*r21 + KC0e0109*r22 + KC0e0209*r23) + r12*(KC0e0010*r21 + KC0e0110*r22 + KC0e0210*r23) + r13*(KC0e0011*r21 + KC0e0111*r22 + KC0e0211*r23)
            k += 1
            KC0v[k] += r21*(KC0e0009*r21 + KC0e0109*r22 + KC0e0209*r23) + r22*(KC0e0010*r21 + KC0e0110*r22 + KC0e0210*r23) + r23*(KC0e0011*r21 + KC0e0111*r22 + KC0e0211*r23)
            k += 1
            KC0v[k] += r31*(KC0e0009*r21 + KC0e0109*r22 + KC0e0209*r23) + r32*(KC0e0010*r21 + KC0e0110*r22 + KC0e0210*r23) + r33*(KC0e0011*r21 + KC0e0111*r22 + KC0e0211*r23)
            k += 1
            KC0v[k] += r11*(KC0e0000*r31 + KC0e0001*r32 + KC0e0002*r33) + r12*(KC0e0001*r31 + KC0e0101*r32 + KC0e0102*r33) + r13*(KC0e0002*r31 + KC0e0102*r32 + KC0e0202*r33)
            k += 1
            KC0v[k] += r21*(KC0e0000*r31 + KC0e0001*r32 + KC0e0002*r33) + r22*(KC0e0001*r31 + KC0e0101*r32 + KC0e0102*r33) + r23*(KC0e0002*r31 + KC0e0102*r32 + KC0e0202*r33)
            k += 1
            KC0v[k] += r31*(KC0e0000*r31 + KC0e0001*r32 + KC0e0002*r33) + r32*(KC0e0001*r31 + KC0e0101*r32 + KC0e0102*r33) + r33*(KC0e0002*r31 + KC0e0102*r32 + KC0e0202*r33)
            k += 1
            KC0v[k] += r11*(KC0e0003*r31 + KC0e0103*r32 + KC0e0203*r33) + r12*(KC0e0004*r31 + KC0e0104*r32 + KC0e0204*r33) + r13*(KC0e0005*r31 + KC0e0105*r32 + KC0e0205*r33)
            k += 1
            KC0v[k] += r21*(KC0e0003*r31 + KC0e0103*r32 + KC0e0203*r33) + r22*(KC0e0004*r31 + KC0e0104*r32 + KC0e0204*r33) + r23*(KC0e0005*r31 + KC0e0105*r32 + KC0e0205*r33)
            k += 1
            KC0v[k] += r31*(KC0e0003*r31 + KC0e0103*r32 + KC0e0203*r33) + r32*(KC0e0004*r31 + KC0e0104*r32 + KC0e0204*r33) + r33*(KC0e0005*r31 + KC0e0105*r32 + KC0e0205*r33)
            k += 1
            KC0v[k] += r11*(KC0e0006*r31 + KC0e0106*r32 + KC0e0206*r33) + r12*(KC0e0007*r31 + KC0e0107*r32 + KC0e0207*r33) + r13*(KC0e0008*r31 + KC0e0108*r32 + KC0e0208*r33)
            k += 1
            KC0v[k] += r21*(KC0e0006*r31 + KC0e0106*r32 + KC0e0206*r33) + r22*(KC0e0007*r31 + KC0e0107*r32 + KC0e0207*r33) + r23*(KC0e0008*r31 + KC0e0108*r32 + KC0e0208*r33)
            k += 1
            KC0v[k] += r31*(KC0e0006*r31 + KC0e0106*r32 + KC0e0206*r33) + r32*(KC0e0007*r31 + KC0e0107*r32 + KC0e0207*r33) + r33*(KC0e0008*r31 + KC0e0108*r32 + KC0e0208*r33)
            k += 1
            KC0v[k] += r11*(KC0e0009*r31 + KC0e0109*r32 + KC0e0209*r33) + r12*(KC0e0010*r31 + KC0e0110*r32 + KC0e0210*r33) + r13*(KC0e0011*r31 + KC0e0111*r32 + KC0e0211*r33)
            k += 1
            KC0v[k] += r21*(KC0e0009*r31 + KC0e0109*r32 + KC0e0209*r33) + r22*(KC0e0010*r31 + KC0e0110*r32 + KC0e0210*r33) + r23*(KC0e0011*r31 + KC0e0111*r32 + KC0e0211*r33)
            k += 1
            KC0v[k] += r31*(KC0e0009*r31 + KC0e0109*r32 + KC0e0209*r33) + r32*(KC0e0010*r31 + KC0e0110*r32 + KC0e0210*r33) + r33*(KC0e0011*r31 + KC0e0111*r32 + KC0e0211*r33)
            k += 1
            KC0v[k] += r11*(KC0e0003*r11 + KC0e0004*r12 + KC0e0005*r13) + r12*(KC0e0103*r11 + KC0e0104*r12 + KC0e0105*r13) + r13*(KC0e0203*r11 + KC0e0204*r12 + KC0e0205*r13)
            k += 1
            KC0v[k] += r21*(KC0e0003*r11 + KC0e0004*r12 + KC0e0005*r13) + r22*(KC0e0103*r11 + KC0e0104*r12 + KC0e0105*r13) + r23*(KC0e0203*r11 + KC0e0204*r12 + KC0e0205*r13)
            k += 1
            KC0v[k] += r31*(KC0e0003*r11 + KC0e0004*r12 + KC0e0005*r13) + r32*(KC0e0103*r11 + KC0e0104*r12 + KC0e0105*r13) + r33*(KC0e0203*r11 + KC0e0204*r12 + KC0e0205*r13)
            k += 1
            KC0v[k] += r11*(KC0e0303*r11 + KC0e0304*r12 + KC0e0305*r13) + r12*(KC0e0304*r11 + KC0e0404*r12 + KC0e0405*r13) + r13*(KC0e0305*r11 + KC0e0405*r12 + KC0e0505*r13)
            k += 1
            KC0v[k] += r21*(KC0e0303*r11 + KC0e0304*r12 + KC0e0305*r13) + r22*(KC0e0304*r11 + KC0e0404*r12 + KC0e0405*r13) + r23*(KC0e0305*r11 + KC0e0405*r12 + KC0e0505*r13)
            k += 1
            KC0v[k] += r31*(KC0e0303*r11 + KC0e0304*r12 + KC0e0305*r13) + r32*(KC0e0304*r11 + KC0e0404*r12 + KC0e0405*r13) + r33*(KC0e0305*r11 + KC0e0405*r12 + KC0e0505*r13)
            k += 1
            KC0v[k] += r11*(KC0e0306*r11 + KC0e0406*r12 + KC0e0506*r13) + r12*(KC0e0307*r11 + KC0e0407*r12 + KC0e0507*r13) + r13*(KC0e0308*r11 + KC0e0408*r12 + KC0e0508*r13)
            k += 1
            KC0v[k] += r21*(KC0e0306*r11 + KC0e0406*r12 + KC0e0506*r13) + r22*(KC0e0307*r11 + KC0e0407*r12 + KC0e0507*r13) + r23*(KC0e0308*r11 + KC0e0408*r12 + KC0e0508*r13)
            k += 1
            KC0v[k] += r31*(KC0e0306*r11 + KC0e0406*r12 + KC0e0506*r13) + r32*(KC0e0307*r11 + KC0e0407*r12 + KC0e0507*r13) + r33*(KC0e0308*r11 + KC0e0408*r12 + KC0e0508*r13)
            k += 1
            KC0v[k] += r11*(KC0e0309*r11 + KC0e0409*r12 + KC0e0509*r13) + r12*(KC0e0310*r11 + KC0e0410*r12 + KC0e0510*r13) + r13*(KC0e0311*r11 + KC0e0411*r12 + KC0e0511*r13)
            k += 1
            KC0v[k] += r21*(KC0e0309*r11 + KC0e0409*r12 + KC0e0509*r13) + r22*(KC0e0310*r11 + KC0e0410*r12 + KC0e0510*r13) + r23*(KC0e0311*r11 + KC0e0411*r12 + KC0e0511*r13)
            k += 1
            KC0v[k] += r31*(KC0e0309*r11 + KC0e0409*r12 + KC0e0509*r13) + r32*(KC0e0310*r11 + KC0e0410*r12 + KC0e0510*r13) + r33*(KC0e0311*r11 + KC0e0411*r12 + KC0e0511*r13)
            k += 1
            KC0v[k] += r11*(KC0e0003*r21 + KC0e0004*r22 + KC0e0005*r23) + r12*(KC0e0103*r21 + KC0e0104*r22 + KC0e0105*r23) + r13*(KC0e0203*r21 + KC0e0204*r22 + KC0e0205*r23)
            k += 1
            KC0v[k] += r21*(KC0e0003*r21 + KC0e0004*r22 + KC0e0005*r23) + r22*(KC0e0103*r21 + KC0e0104*r22 + KC0e0105*r23) + r23*(KC0e0203*r21 + KC0e0204*r22 + KC0e0205*r23)
            k += 1
            KC0v[k] += r31*(KC0e0003*r21 + KC0e0004*r22 + KC0e0005*r23) + r32*(KC0e0103*r21 + KC0e0104*r22 + KC0e0105*r23) + r33*(KC0e0203*r21 + KC0e0204*r22 + KC0e0205*r23)
            k += 1
            KC0v[k] += r11*(KC0e0303*r21 + KC0e0304*r22 + KC0e0305*r23) + r12*(KC0e0304*r21 + KC0e0404*r22 + KC0e0405*r23) + r13*(KC0e0305*r21 + KC0e0405*r22 + KC0e0505*r23)
            k += 1
            KC0v[k] += r21*(KC0e0303*r21 + KC0e0304*r22 + KC0e0305*r23) + r22*(KC0e0304*r21 + KC0e0404*r22 + KC0e0405*r23) + r23*(KC0e0305*r21 + KC0e0405*r22 + KC0e0505*r23)
            k += 1
            KC0v[k] += r31*(KC0e0303*r21 + KC0e0304*r22 + KC0e0305*r23) + r32*(KC0e0304*r21 + KC0e0404*r22 + KC0e0405*r23) + r33*(KC0e0305*r21 + KC0e0405*r22 + KC0e0505*r23)
            k += 1
            KC0v[k] += r11*(KC0e0306*r21 + KC0e0406*r22 + KC0e0506*r23) + r12*(KC0e0307*r21 + KC0e0407*r22 + KC0e0507*r23) + r13*(KC0e0308*r21 + KC0e0408*r22 + KC0e0508*r23)
            k += 1
            KC0v[k] += r21*(KC0e0306*r21 + KC0e0406*r22 + KC0e0506*r23) + r22*(KC0e0307*r21 + KC0e0407*r22 + KC0e0507*r23) + r23*(KC0e0308*r21 + KC0e0408*r22 + KC0e0508*r23)
            k += 1
            KC0v[k] += r31*(KC0e0306*r21 + KC0e0406*r22 + KC0e0506*r23) + r32*(KC0e0307*r21 + KC0e0407*r22 + KC0e0507*r23) + r33*(KC0e0308*r21 + KC0e0408*r22 + KC0e0508*r23)
            k += 1
            KC0v[k] += r11*(KC0e0309*r21 + KC0e0409*r22 + KC0e0509*r23) + r12*(KC0e0310*r21 + KC0e0410*r22 + KC0e0510*r23) + r13*(KC0e0311*r21 + KC0e0411*r22 + KC0e0511*r23)
            k += 1
            KC0v[k] += r21*(KC0e0309*r21 + KC0e0409*r22 + KC0e0509*r23) + r22*(KC0e0310*r21 + KC0e0410*r22 + KC0e0510*r23) + r23*(KC0e0311*r21 + KC0e0411*r22 + KC0e0511*r23)
            k += 1
            KC0v[k] += r31*(KC0e0309*r21 + KC0e0409*r22 + KC0e0509*r23) + r32*(KC0e0310*r21 + KC0e0410*r22 + KC0e0510*r23) + r33*(KC0e0311*r21 + KC0e0411*r22 + KC0e0511*r23)
            k += 1
            KC0v[k] += r11*(KC0e0003*r31 + KC0e0004*r32 + KC0e0005*r33) + r12*(KC0e0103*r31 + KC0e0104*r32 + KC0e0105*r33) + r13*(KC0e0203*r31 + KC0e0204*r32 + KC0e0205*r33)
            k += 1
            KC0v[k] += r21*(KC0e0003*r31 + KC0e0004*r32 + KC0e0005*r33) + r22*(KC0e0103*r31 + KC0e0104*r32 + KC0e0105*r33) + r23*(KC0e0203*r31 + KC0e0204*r32 + KC0e0205*r33)
            k += 1
            KC0v[k] += r31*(KC0e0003*r31 + KC0e0004*r32 + KC0e0005*r33) + r32*(KC0e0103*r31 + KC0e0104*r32 + KC0e0105*r33) + r33*(KC0e0203*r31 + KC0e0204*r32 + KC0e0205*r33)
            k += 1
            KC0v[k] += r11*(KC0e0303*r31 + KC0e0304*r32 + KC0e0305*r33) + r12*(KC0e0304*r31 + KC0e0404*r32 + KC0e0405*r33) + r13*(KC0e0305*r31 + KC0e0405*r32 + KC0e0505*r33)
            k += 1
            KC0v[k] += r21*(KC0e0303*r31 + KC0e0304*r32 + KC0e0305*r33) + r22*(KC0e0304*r31 + KC0e0404*r32 + KC0e0405*r33) + r23*(KC0e0305*r31 + KC0e0405*r32 + KC0e0505*r33)
            k += 1
            KC0v[k] += r31*(KC0e0303*r31 + KC0e0304*r32 + KC0e0305*r33) + r32*(KC0e0304*r31 + KC0e0404*r32 + KC0e0405*r33) + r33*(KC0e0305*r31 + KC0e0405*r32 + KC0e0505*r33)
            k += 1
            KC0v[k] += r11*(KC0e0306*r31 + KC0e0406*r32 + KC0e0506*r33) + r12*(KC0e0307*r31 + KC0e0407*r32 + KC0e0507*r33) + r13*(KC0e0308*r31 + KC0e0408*r32 + KC0e0508*r33)
            k += 1
            KC0v[k] += r21*(KC0e0306*r31 + KC0e0406*r32 + KC0e0506*r33) + r22*(KC0e0307*r31 + KC0e0407*r32 + KC0e0507*r33) + r23*(KC0e0308*r31 + KC0e0408*r32 + KC0e0508*r33)
            k += 1
            KC0v[k] += r31*(KC0e0306*r31 + KC0e0406*r32 + KC0e0506*r33) + r32*(KC0e0307*r31 + KC0e0407*r32 + KC0e0507*r33) + r33*(KC0e0308*r31 + KC0e0408*r32 + KC0e0508*r33)
            k += 1
            KC0v[k] += r11*(KC0e0309*r31 + KC0e0409*r32 + KC0e0509*r33) + r12*(KC0e0310*r31 + KC0e0410*r32 + KC0e0510*r33) + r13*(KC0e0311*r31 + KC0e0411*r32 + KC0e0511*r33)
            k += 1
            KC0v[k] += r21*(KC0e0309*r31 + KC0e0409*r32 + KC0e0509*r33) + r22*(KC0e0310*r31 + KC0e0410*r32 + KC0e0510*r33) + r23*(KC0e0311*r31 + KC0e0411*r32 + KC0e0511*r33)
            k += 1
            KC0v[k] += r31*(KC0e0309*r31 + KC0e0409*r32 + KC0e0509*r33) + r32*(KC0e0310*r31 + KC0e0410*r32 + KC0e0510*r33) + r33*(KC0e0311*r31 + KC0e0411*r32 + KC0e0511*r33)
            k += 1
            KC0v[k] += r11*(KC0e0006*r11 + KC0e0007*r12 + KC0e0008*r13) + r12*(KC0e0106*r11 + KC0e0107*r12 + KC0e0108*r13) + r13*(KC0e0206*r11 + KC0e0207*r12 + KC0e0208*r13)
            k += 1
            KC0v[k] += r21*(KC0e0006*r11 + KC0e0007*r12 + KC0e0008*r13) + r22*(KC0e0106*r11 + KC0e0107*r12 + KC0e0108*r13) + r23*(KC0e0206*r11 + KC0e0207*r12 + KC0e0208*r13)
            k += 1
            KC0v[k] += r31*(KC0e0006*r11 + KC0e0007*r12 + KC0e0008*r13) + r32*(KC0e0106*r11 + KC0e0107*r12 + KC0e0108*r13) + r33*(KC0e0206*r11 + KC0e0207*r12 + KC0e0208*r13)
            k += 1
            KC0v[k] += r11*(KC0e0306*r11 + KC0e0307*r12 + KC0e0308*r13) + r12*(KC0e0406*r11 + KC0e0407*r12 + KC0e0408*r13) + r13*(KC0e0506*r11 + KC0e0507*r12 + KC0e0508*r13)
            k += 1
            KC0v[k] += r21*(KC0e0306*r11 + KC0e0307*r12 + KC0e0308*r13) + r22*(KC0e0406*r11 + KC0e0407*r12 + KC0e0408*r13) + r23*(KC0e0506*r11 + KC0e0507*r12 + KC0e0508*r13)
            k += 1
            KC0v[k] += r31*(KC0e0306*r11 + KC0e0307*r12 + KC0e0308*r13) + r32*(KC0e0406*r11 + KC0e0407*r12 + KC0e0408*r13) + r33*(KC0e0506*r11 + KC0e0507*r12 + KC0e0508*r13)
            k += 1
            KC0v[k] += r11*(KC0e0606*r11 + KC0e0607*r12 + KC0e0608*r13) + r12*(KC0e0607*r11 + KC0e0707*r12 + KC0e0708*r13) + r13*(KC0e0608*r11 + KC0e0708*r12 + KC0e0808*r13)
            k += 1
            KC0v[k] += r21*(KC0e0606*r11 + KC0e0607*r12 + KC0e0608*r13) + r22*(KC0e0607*r11 + KC0e0707*r12 + KC0e0708*r13) + r23*(KC0e0608*r11 + KC0e0708*r12 + KC0e0808*r13)
            k += 1
            KC0v[k] += r31*(KC0e0606*r11 + KC0e0607*r12 + KC0e0608*r13) + r32*(KC0e0607*r11 + KC0e0707*r12 + KC0e0708*r13) + r33*(KC0e0608*r11 + KC0e0708*r12 + KC0e0808*r13)
            k += 1
            KC0v[k] += r11*(KC0e0609*r11 + KC0e0709*r12 + KC0e0809*r13) + r12*(KC0e0610*r11 + KC0e0710*r12 + KC0e0810*r13) + r13*(KC0e0611*r11 + KC0e0711*r12 + KC0e0811*r13)
            k += 1
            KC0v[k] += r21*(KC0e0609*r11 + KC0e0709*r12 + KC0e0809*r13) + r22*(KC0e0610*r11 + KC0e0710*r12 + KC0e0810*r13) + r23*(KC0e0611*r11 + KC0e0711*r12 + KC0e0811*r13)
            k += 1
            KC0v[k] += r31*(KC0e0609*r11 + KC0e0709*r12 + KC0e0809*r13) + r32*(KC0e0610*r11 + KC0e0710*r12 + KC0e0810*r13) + r33*(KC0e0611*r11 + KC0e0711*r12 + KC0e0811*r13)
            k += 1
            KC0v[k] += r11*(KC0e0006*r21 + KC0e0007*r22 + KC0e0008*r23) + r12*(KC0e0106*r21 + KC0e0107*r22 + KC0e0108*r23) + r13*(KC0e0206*r21 + KC0e0207*r22 + KC0e0208*r23)
            k += 1
            KC0v[k] += r21*(KC0e0006*r21 + KC0e0007*r22 + KC0e0008*r23) + r22*(KC0e0106*r21 + KC0e0107*r22 + KC0e0108*r23) + r23*(KC0e0206*r21 + KC0e0207*r22 + KC0e0208*r23)
            k += 1
            KC0v[k] += r31*(KC0e0006*r21 + KC0e0007*r22 + KC0e0008*r23) + r32*(KC0e0106*r21 + KC0e0107*r22 + KC0e0108*r23) + r33*(KC0e0206*r21 + KC0e0207*r22 + KC0e0208*r23)
            k += 1
            KC0v[k] += r11*(KC0e0306*r21 + KC0e0307*r22 + KC0e0308*r23) + r12*(KC0e0406*r21 + KC0e0407*r22 + KC0e0408*r23) + r13*(KC0e0506*r21 + KC0e0507*r22 + KC0e0508*r23)
            k += 1
            KC0v[k] += r21*(KC0e0306*r21 + KC0e0307*r22 + KC0e0308*r23) + r22*(KC0e0406*r21 + KC0e0407*r22 + KC0e0408*r23) + r23*(KC0e0506*r21 + KC0e0507*r22 + KC0e0508*r23)
            k += 1
            KC0v[k] += r31*(KC0e0306*r21 + KC0e0307*r22 + KC0e0308*r23) + r32*(KC0e0406*r21 + KC0e0407*r22 + KC0e0408*r23) + r33*(KC0e0506*r21 + KC0e0507*r22 + KC0e0508*r23)
            k += 1
            KC0v[k] += r11*(KC0e0606*r21 + KC0e0607*r22 + KC0e0608*r23) + r12*(KC0e0607*r21 + KC0e0707*r22 + KC0e0708*r23) + r13*(KC0e0608*r21 + KC0e0708*r22 + KC0e0808*r23)
            k += 1
            KC0v[k] += r21*(KC0e0606*r21 + KC0e0607*r22 + KC0e0608*r23) + r22*(KC0e0607*r21 + KC0e0707*r22 + KC0e0708*r23) + r23*(KC0e0608*r21 + KC0e0708*r22 + KC0e0808*r23)
            k += 1
            KC0v[k] += r31*(KC0e0606*r21 + KC0e0607*r22 + KC0e0608*r23) + r32*(KC0e0607*r21 + KC0e0707*r22 + KC0e0708*r23) + r33*(KC0e0608*r21 + KC0e0708*r22 + KC0e0808*r23)
            k += 1
            KC0v[k] += r11*(KC0e0609*r21 + KC0e0709*r22 + KC0e0809*r23) + r12*(KC0e0610*r21 + KC0e0710*r22 + KC0e0810*r23) + r13*(KC0e0611*r21 + KC0e0711*r22 + KC0e0811*r23)
            k += 1
            KC0v[k] += r21*(KC0e0609*r21 + KC0e0709*r22 + KC0e0809*r23) + r22*(KC0e0610*r21 + KC0e0710*r22 + KC0e0810*r23) + r23*(KC0e0611*r21 + KC0e0711*r22 + KC0e0811*r23)
            k += 1
            KC0v[k] += r31*(KC0e0609*r21 + KC0e0709*r22 + KC0e0809*r23) + r32*(KC0e0610*r21 + KC0e0710*r22 + KC0e0810*r23) + r33*(KC0e0611*r21 + KC0e0711*r22 + KC0e0811*r23)
            k += 1
            KC0v[k] += r11*(KC0e0006*r31 + KC0e0007*r32 + KC0e0008*r33) + r12*(KC0e0106*r31 + KC0e0107*r32 + KC0e0108*r33) + r13*(KC0e0206*r31 + KC0e0207*r32 + KC0e0208*r33)
            k += 1
            KC0v[k] += r21*(KC0e0006*r31 + KC0e0007*r32 + KC0e0008*r33) + r22*(KC0e0106*r31 + KC0e0107*r32 + KC0e0108*r33) + r23*(KC0e0206*r31 + KC0e0207*r32 + KC0e0208*r33)
            k += 1
            KC0v[k] += r31*(KC0e0006*r31 + KC0e0007*r32 + KC0e0008*r33) + r32*(KC0e0106*r31 + KC0e0107*r32 + KC0e0108*r33) + r33*(KC0e0206*r31 + KC0e0207*r32 + KC0e0208*r33)
            k += 1
            KC0v[k] += r11*(KC0e0306*r31 + KC0e0307*r32 + KC0e0308*r33) + r12*(KC0e0406*r31 + KC0e0407*r32 + KC0e0408*r33) + r13*(KC0e0506*r31 + KC0e0507*r32 + KC0e0508*r33)
            k += 1
            KC0v[k] += r21*(KC0e0306*r31 + KC0e0307*r32 + KC0e0308*r33) + r22*(KC0e0406*r31 + KC0e0407*r32 + KC0e0408*r33) + r23*(KC0e0506*r31 + KC0e0507*r32 + KC0e0508*r33)
            k += 1
            KC0v[k] += r31*(KC0e0306*r31 + KC0e0307*r32 + KC0e0308*r33) + r32*(KC0e0406*r31 + KC0e0407*r32 + KC0e0408*r33) + r33*(KC0e0506*r31 + KC0e0507*r32 + KC0e0508*r33)
            k += 1
            KC0v[k] += r11*(KC0e0606*r31 + KC0e0607*r32 + KC0e0608*r33) + r12*(KC0e0607*r31 + KC0e0707*r32 + KC0e0708*r33) + r13*(KC0e0608*r31 + KC0e0708*r32 + KC0e0808*r33)
            k += 1
            KC0v[k] += r21*(KC0e0606*r31 + KC0e0607*r32 + KC0e0608*r33) + r22*(KC0e0607*r31 + KC0e0707*r32 + KC0e0708*r33) + r23*(KC0e0608*r31 + KC0e0708*r32 + KC0e0808*r33)
            k += 1
            KC0v[k] += r31*(KC0e0606*r31 + KC0e0607*r32 + KC0e0608*r33) + r32*(KC0e0607*r31 + KC0e0707*r32 + KC0e0708*r33) + r33*(KC0e0608*r31 + KC0e0708*r32 + KC0e0808*r33)
            k += 1
            KC0v[k] += r11*(KC0e0609*r31 + KC0e0709*r32 + KC0e0809*r33) + r12*(KC0e0610*r31 + KC0e0710*r32 + KC0e0810*r33) + r13*(KC0e0611*r31 + KC0e0711*r32 + KC0e0811*r33)
            k += 1
            KC0v[k] += r21*(KC0e0609*r31 + KC0e0709*r32 + KC0e0809*r33) + r22*(KC0e0610*r31 + KC0e0710*r32 + KC0e0810*r33) + r23*(KC0e0611*r31 + KC0e0711*r32 + KC0e0811*r33)
            k += 1
            KC0v[k] += r31*(KC0e0609*r31 + KC0e0709*r32 + KC0e0809*r33) + r32*(KC0e0610*r31 + KC0e0710*r32 + KC0e0810*r33) + r33*(KC0e0611*r31 + KC0e0711*r32 + KC0e0811*r33)
            k += 1
            KC0v[k] += r11*(KC0e0009*r11 + KC0e0010*r12 + KC0e0011*r13) + r12*(KC0e0109*r11 + KC0e0110*r12 + KC0e0111*r13) + r13*(KC0e0209*r11 + KC0e0210*r12 + KC0e0211*r13)
            k += 1
            KC0v[k] += r21*(KC0e0009*r11 + KC0e0010*r12 + KC0e0011*r13) + r22*(KC0e0109*r11 + KC0e0110*r12 + KC0e0111*r13) + r23*(KC0e0209*r11 + KC0e0210*r12 + KC0e0211*r13)
            k += 1
            KC0v[k] += r31*(KC0e0009*r11 + KC0e0010*r12 + KC0e0011*r13) + r32*(KC0e0109*r11 + KC0e0110*r12 + KC0e0111*r13) + r33*(KC0e0209*r11 + KC0e0210*r12 + KC0e0211*r13)
            k += 1
            KC0v[k] += r11*(KC0e0309*r11 + KC0e0310*r12 + KC0e0311*r13) + r12*(KC0e0409*r11 + KC0e0410*r12 + KC0e0411*r13) + r13*(KC0e0509*r11 + KC0e0510*r12 + KC0e0511*r13)
            k += 1
            KC0v[k] += r21*(KC0e0309*r11 + KC0e0310*r12 + KC0e0311*r13) + r22*(KC0e0409*r11 + KC0e0410*r12 + KC0e0411*r13) + r23*(KC0e0509*r11 + KC0e0510*r12 + KC0e0511*r13)
            k += 1
            KC0v[k] += r31*(KC0e0309*r11 + KC0e0310*r12 + KC0e0311*r13) + r32*(KC0e0409*r11 + KC0e0410*r12 + KC0e0411*r13) + r33*(KC0e0509*r11 + KC0e0510*r12 + KC0e0511*r13)
            k += 1
            KC0v[k] += r11*(KC0e0609*r11 + KC0e0610*r12 + KC0e0611*r13) + r12*(KC0e0709*r11 + KC0e0710*r12 + KC0e0711*r13) + r13*(KC0e0809*r11 + KC0e0810*r12 + KC0e0811*r13)
            k += 1
            KC0v[k] += r21*(KC0e0609*r11 + KC0e0610*r12 + KC0e0611*r13) + r22*(KC0e0709*r11 + KC0e0710*r12 + KC0e0711*r13) + r23*(KC0e0809*r11 + KC0e0810*r12 + KC0e0811*r13)
            k += 1
            KC0v[k] += r31*(KC0e0609*r11 + KC0e0610*r12 + KC0e0611*r13) + r32*(KC0e0709*r11 + KC0e0710*r12 + KC0e0711*r13) + r33*(KC0e0809*r11 + KC0e0810*r12 + KC0e0811*r13)
            k += 1
            KC0v[k] += r11*(KC0e0909*r11 + KC0e0910*r12 + KC0e0911*r13) + r12*(KC0e0910*r11 + KC0e1010*r12 + KC0e1011*r13) + r13*(KC0e0911*r11 + KC0e1011*r12 + KC0e1111*r13)
            k += 1
            KC0v[k] += r21*(KC0e0909*r11 + KC0e0910*r12 + KC0e0911*r13) + r22*(KC0e0910*r11 + KC0e1010*r12 + KC0e1011*r13) + r23*(KC0e0911*r11 + KC0e1011*r12 + KC0e1111*r13)
            k += 1
            KC0v[k] += r31*(KC0e0909*r11 + KC0e0910*r12 + KC0e0911*r13) + r32*(KC0e0910*r11 + KC0e1010*r12 + KC0e1011*r13) + r33*(KC0e0911*r11 + KC0e1011*r12 + KC0e1111*r13)
            k += 1
            KC0v[k] += r11*(KC0e0009*r21 + KC0e0010*r22 + KC0e0011*r23) + r12*(KC0e0109*r21 + KC0e0110*r22 + KC0e0111*r23) + r13*(KC0e0209*r21 + KC0e0210*r22 + KC0e0211*r23)
            k += 1
            KC0v[k] += r21*(KC0e0009*r21 + KC0e0010*r22 + KC0e0011*r23) + r22*(KC0e0109*r21 + KC0e0110*r22 + KC0e0111*r23) + r23*(KC0e0209*r21 + KC0e0210*r22 + KC0e0211*r23)
            k += 1
            KC0v[k] += r31*(KC0e0009*r21 + KC0e0010*r22 + KC0e0011*r23) + r32*(KC0e0109*r21 + KC0e0110*r22 + KC0e0111*r23) + r33*(KC0e0209*r21 + KC0e0210*r22 + KC0e0211*r23)
            k += 1
            KC0v[k] += r11*(KC0e0309*r21 + KC0e0310*r22 + KC0e0311*r23) + r12*(KC0e0409*r21 + KC0e0410*r22 + KC0e0411*r23) + r13*(KC0e0509*r21 + KC0e0510*r22 + KC0e0511*r23)
            k += 1
            KC0v[k] += r21*(KC0e0309*r21 + KC0e0310*r22 + KC0e0311*r23) + r22*(KC0e0409*r21 + KC0e0410*r22 + KC0e0411*r23) + r23*(KC0e0509*r21 + KC0e0510*r22 + KC0e0511*r23)
            k += 1
            KC0v[k] += r31*(KC0e0309*r21 + KC0e0310*r22 + KC0e0311*r23) + r32*(KC0e0409*r21 + KC0e0410*r22 + KC0e0411*r23) + r33*(KC0e0509*r21 + KC0e0510*r22 + KC0e0511*r23)
            k += 1
            KC0v[k] += r11*(KC0e0609*r21 + KC0e0610*r22 + KC0e0611*r23) + r12*(KC0e0709*r21 + KC0e0710*r22 + KC0e0711*r23) + r13*(KC0e0809*r21 + KC0e0810*r22 + KC0e0811*r23)
            k += 1
            KC0v[k] += r21*(KC0e0609*r21 + KC0e0610*r22 + KC0e0611*r23) + r22*(KC0e0709*r21 + KC0e0710*r22 + KC0e0711*r23) + r23*(KC0e0809*r21 + KC0e0810*r22 + KC0e0811*r23)
            k += 1
            KC0v[k] += r31*(KC0e0609*r21 + KC0e0610*r22 + KC0e0611*r23) + r32*(KC0e0709*r21 + KC0e0710*r22 + KC0e0711*r23) + r33*(KC0e0809*r21 + KC0e0810*r22 + KC0e0811*r23)
            k += 1
            KC0v[k] += r11*(KC0e0909*r21 + KC0e0910*r22 + KC0e0911*r23) + r12*(KC0e0910*r21 + KC0e1010*r22 + KC0e1011*r23) + r13*(KC0e0911*r21 + KC0e1011*r22 + KC0e1111*r23)
            k += 1
            KC0v[k] += r21*(KC0e0909*r21 + KC0e0910*r22 + KC0e0911*r23) + r22*(KC0e0910*r21 + KC0e1010*r22 + KC0e1011*r23) + r23*(KC0e0911*r21 + KC0e1011*r22 + KC0e1111*r23)
            k += 1
            KC0v[k] += r31*(KC0e0909*r21 + KC0e0910*r22 + KC0e0911*r23) + r32*(KC0e0910*r21 + KC0e1010*r22 + KC0e1011*r23) + r33*(KC0e0911*r21 + KC0e1011*r22 + KC0e1111*r23)
            k += 1
            KC0v[k] += r11*(KC0e0009*r31 + KC0e0010*r32 + KC0e0011*r33) + r12*(KC0e0109*r31 + KC0e0110*r32 + KC0e0111*r33) + r13*(KC0e0209*r31 + KC0e0210*r32 + KC0e0211*r33)
            k += 1
            KC0v[k] += r21*(KC0e0009*r31 + KC0e0010*r32 + KC0e0011*r33) + r22*(KC0e0109*r31 + KC0e0110*r32 + KC0e0111*r33) + r23*(KC0e0209*r31 + KC0e0210*r32 + KC0e0211*r33)
            k += 1
            KC0v[k] += r31*(KC0e0009*r31 + KC0e0010*r32 + KC0e0011*r33) + r32*(KC0e0109*r31 + KC0e0110*r32 + KC0e0111*r33) + r33*(KC0e0209*r31 + KC0e0210*r32 + KC0e0211*r33)
            k += 1
            KC0v[k] += r11*(KC0e0309*r31 + KC0e0310*r32 + KC0e0311*r33) + r12*(KC0e0409*r31 + KC0e0410*r32 + KC0e0411*r33) + r13*(KC0e0509*r31 + KC0e0510*r32 + KC0e0511*r33)
            k += 1
            KC0v[k] += r21*(KC0e0309*r31 + KC0e0310*r32 + KC0e0311*r33) + r22*(KC0e0409*r31 + KC0e0410*r32 + KC0e0411*r33) + r23*(KC0e0509*r31 + KC0e0510*r32 + KC0e0511*r33)
            k += 1
            KC0v[k] += r31*(KC0e0309*r31 + KC0e0310*r32 + KC0e0311*r33) + r32*(KC0e0409*r31 + KC0e0410*r32 + KC0e0411*r33) + r33*(KC0e0509*r31 + KC0e0510*r32 + KC0e0511*r33)
            k += 1
            KC0v[k] += r11*(KC0e0609*r31 + KC0e0610*r32 + KC0e0611*r33) + r12*(KC0e0709*r31 + KC0e0710*r32 + KC0e0711*r33) + r13*(KC0e0809*r31 + KC0e0810*r32 + KC0e0811*r33)
            k += 1
            KC0v[k] += r21*(KC0e0609*r31 + KC0e0610*r32 + KC0e0611*r33) + r22*(KC0e0709*r31 + KC0e0710*r32 + KC0e0711*r33) + r23*(KC0e0809*r31 + KC0e0810*r32 + KC0e0811*r33)
            k += 1
            KC0v[k] += r31*(KC0e0609*r31 + KC0e0610*r32 + KC0e0611*r33) + r32*(KC0e0709*r31 + KC0e0710*r32 + KC0e0711*r33) + r33*(KC0e0809*r31 + KC0e0810*r32 + KC0e0811*r33)
            k += 1
            KC0v[k] += r11*(KC0e0909*r31 + KC0e0910*r32 + KC0e0911*r33) + r12*(KC0e0910*r31 + KC0e1010*r32 + KC0e1011*r33) + r13*(KC0e0911*r31 + KC0e1011*r32 + KC0e1111*r33)
            k += 1
            KC0v[k] += r21*(KC0e0909*r31 + KC0e0910*r32 + KC0e0911*r33) + r22*(KC0e0910*r31 + KC0e1010*r32 + KC0e1011*r33) + r23*(KC0e0911*r31 + KC0e1011*r32 + KC0e1111*r33)
            k += 1
            KC0v[k] += r31*(KC0e0909*r31 + KC0e0910*r32 + KC0e0911*r33) + r32*(KC0e0910*r31 + KC0e1010*r32 + KC0e1011*r33) + r33*(KC0e0911*r31 + KC0e1011*r32 + KC0e1111*r33)

    cpdef void update_fint(BeamC self,
                           double [::1] fint,
                           BeamProp prop,
                           int nonlinear=0):
        r"""Update the internal force vector

        Calls ``update_probe_finte(prop, nonlinear)``, transforms the forces
        and moments of ``probe.finte`` to global coordinates with the rotation
        matrix and adds them to ``fint`` at the positions ``c1`` to
        ``c1 + 5`` and ``c2`` to ``c2 + 5``. The displacements ``probe.ue``
        must be updated before, see :meth:`.update_probe_ue`.

        Parameters
        ----------
        fint : np.array
            Array that is updated in place with the internal forces. The
            internal forces stored in ``fint`` are calculated in global
            coordinates. Method :meth:`.update_probe_finte` is called to update
            the parameter ``finte`` of the :class:`.BeamCProbe` with the
            internal forces in local coordinates. 1-D array of size ``6*M``,
            for `M` nodes in the model.
        prop : :class:`.BeamProp` object
            Beam property object from where the stiffness and mass attributes
            are read from.
        nonlinear : int, optional
            The default ``0`` gives the linear internal forces, ``KC0*u``. Any other
            value adds the geometrically nonlinear terms, for which the tangent
            stiffness matrix of the internal forces is ``KC0 + KCNL + KG``, see
            :meth:`.update_KCNL`.

        Notes
        -----
        The nonlinear terms are those of the von Kármán strains, for which the
        exact Jacobian of the internal forces is ``KC0 + KCNL + KG``, see
        :meth:`.update_probe_finte` and :meth:`.update_KCNL`.

        """
        cdef double *finte

        self.update_probe_finte(prop, nonlinear)
        with nogil:
            finte = &self.probe.finte[0]

            fint[0+self.c1] += finte[0]*self.r11 + finte[1]*self.r12 + finte[2]*self.r13
            fint[1+self.c1] += finte[0]*self.r21 + finte[1]*self.r22 + finte[2]*self.r23
            fint[2+self.c1] += finte[0]*self.r31 + finte[1]*self.r32 + finte[2]*self.r33
            fint[3+self.c1] += finte[3]*self.r11 + finte[4]*self.r12 + finte[5]*self.r13
            fint[4+self.c1] += finte[3]*self.r21 + finte[4]*self.r22 + finte[5]*self.r23
            fint[5+self.c1] += finte[3]*self.r31 + finte[4]*self.r32 + finte[5]*self.r33
            fint[0+self.c2] += finte[6]*self.r11 + finte[7]*self.r12 + finte[8]*self.r13
            fint[1+self.c2] += finte[6]*self.r21 + finte[7]*self.r22 + finte[8]*self.r23
            fint[2+self.c2] += finte[6]*self.r31 + finte[7]*self.r32 + finte[8]*self.r33
            fint[3+self.c2] += finte[9]*self.r11 + finte[10]*self.r12 + finte[11]*self.r13
            fint[4+self.c2] += finte[9]*self.r21 + finte[10]*self.r22 + finte[11]*self.r23
            fint[5+self.c2] += finte[9]*self.r31 + finte[10]*self.r32 + finte[11]*self.r33


    cdef void _update_means(BeamC self, BeamProp prop, double *Hm,
                            double *EBLm, double *BNLm, double *eNLm,
                            double *Nm) noexcept nogil:
        r"""Element means of the nonlinear axial strain and of the axial force

        From ``length``, the displacements ``probe.ue`` and the entries
        ``C[0, :]``, ``C[1, 1]``, ``C[2, 2]``, ``C[4, 4]`` and ``C[5, 5]`` of
        the sectional stiffness matrix of ``prop``, see :class:`.BeamProp`,
        in element coordinates, overwrites the ``144`` entries of
        ``Hm``, stored row by row, and the ``12`` entries of ``EBLm`` and
        ``BNLm``, and writes ``eNLm`` and ``Nm``:

        .. math::
            \boldsymbol{H}_m = \frac{1}{L} \int_0^L \left[
            \left(\frac{\partial \boldsymbol{S}^v}{\partial x}\right)^\top
            \frac{\partial \boldsymbol{S}^v}{\partial x}
            + \left(\frac{\partial \boldsymbol{S}^w}{\partial x}\right)^\top
            \frac{\partial \boldsymbol{S}^w}{\partial x} \right] dx, \qquad
            \bar{\epsilon}_{NL} = \frac{1}{2} \boldsymbol{u}_e^\top
            \boldsymbol{H}_m \boldsymbol{u}_e, \qquad
            \bar{\boldsymbol{B}}_{NL} = \boldsymbol{u}_e^\top \boldsymbol{H}_m

        .. math::
            \overline{\boldsymbol{EB}}_L = \frac{1}{L} \int_0^L
            \boldsymbol{C}_{0,:} \boldsymbol{B}_L dx, \qquad
            N_m = \overline{\boldsymbol{EB}}_L \boldsymbol{u}_e

        where `\partial \boldsymbol{S}^v/\partial x` and `\partial
        \boldsymbol{S}^w/\partial x` are the rows giving the slopes `v_{,x}`
        and `w_{,x}` of the consistent shape functions, see the notes of
        :class:`.BeamC`, `\boldsymbol{B}_L` is the matrix whose rows give
        the strains `\{u_{,x}, \gamma_{xy}, \gamma_{xz}, \kappa_x, \kappa_y,
        \kappa_z\}` and `\boldsymbol{C}_{0,:}` the first row of the
        sectional stiffness matrix. `\bar{\epsilon}_{NL}` (``eNLm``) is the
        element mean of `(v_{,x}^2 + w_{,x}^2)/2`,
        `\bar{\boldsymbol{B}}_{NL}` (``BNLm``) its gradient and `N_m`
        (``Nm``) the element mean of the axial force of the linear strains,
        `\boldsymbol{C}_{0,:} \boldsymbol{\epsilon}`, which is `E (A u_{,x}
        + A_y \kappa_y + A_z \kappa_z)` when ``C`` is not set. The shear
        strains, constant along the element, enter through `\alpha_y
        \beta_y` and `\alpha_z \beta_z`, see the notes of :class:`.BeamC`.
        The integrals are
        analytical, printed by
        ``theory/deriving_equations/beamc/derive_beamc_KC0L_KCL0_KCLL_KGLL_sparse.py``.

        """
        cdef int i, j
        cdef double L, alphay, alphaz, betay, betaz
        cdef double C[36]
        cdef double C00, C01, C02, C03, C04, C05, C11, C22, C44, C55
        cdef double *ue

        L = self.length
        prop.get_C(C)
        C00 = C[0]
        C01 = C[1]
        C02 = C[2]
        C03 = C[3]
        C04 = C[4]
        C05 = C[5]
        C11 = C[7]
        C22 = C[14]
        C44 = C[28]
        C55 = C[35]
        # NOTE shear parameters of the shape functions from the diagonal of
        #      the sectional stiffness matrix, see the notes of BeamC
        alphay = -12*C44/(C11*L**2)
        alphaz = -12*C55/(C22*L**2)
        betay = 1/(1. - alphay)
        betaz = 1/(1. - alphaz)

        ue = &self.probe.ue[0]

        for i in range(NUM_NODES*DOF):
            EBLm[i] = 0.
            for j in range(NUM_NODES*DOF):
                Hm[NUM_NODES*DOF*i + j] = 0.

        # Hm = (1/L)*int_0^L (Sv_x.T*Sv_x + Sw_x.T*Sw_x) dx, stored row by row
        Hm[13] = betay**2*(alphay**2 - 2*alphay + 6./5)/L**2
        Hm[17] = betay**2/(10*L)
        Hm[19] = betay**2*(-alphay**2 + 2*alphay - 6./5)/L**2
        Hm[23] = betay**2/(10*L)
        Hm[26] = betaz**2*(alphaz**2 - 2*alphaz + 6./5)/L**2
        Hm[28] = -betaz**2/(10*L)
        Hm[32] = betaz**2*(-alphaz**2 + 2*alphaz - 6./5)/L**2
        Hm[34] = -betaz**2/(10*L)
        Hm[50] = -betaz**2/(10*L)
        Hm[52] = betaz**2*(5*alphaz**2 - 10*alphaz + 8)/60
        Hm[56] = betaz**2/(10*L)
        Hm[58] = betaz**2*(-5*alphaz**2 + 10*alphaz - 2)/60
        Hm[61] = betay**2/(10*L)
        Hm[65] = betay**2*(5*alphay**2 - 10*alphay + 8)/60
        Hm[67] = -betay**2/(10*L)
        Hm[71] = betay**2*(-5*alphay**2 + 10*alphay - 2)/60
        Hm[85] = betay**2*(-alphay**2 + 2*alphay - 6./5)/L**2
        Hm[89] = -betay**2/(10*L)
        Hm[91] = betay**2*(alphay**2 - 2*alphay + 6./5)/L**2
        Hm[95] = -betay**2/(10*L)
        Hm[98] = betaz**2*(-alphaz**2 + 2*alphaz - 6./5)/L**2
        Hm[100] = betaz**2/(10*L)
        Hm[104] = betaz**2*(alphaz**2 - 2*alphaz + 6./5)/L**2
        Hm[106] = betaz**2/(10*L)
        Hm[122] = -betaz**2/(10*L)
        Hm[124] = betaz**2*(-5*alphaz**2 + 10*alphaz - 2)/60
        Hm[128] = betaz**2/(10*L)
        Hm[130] = betaz**2*(5*alphaz**2 - 10*alphaz + 8)/60
        Hm[133] = betay**2/(10*L)
        Hm[137] = betay**2*(-5*alphay**2 + 10*alphay - 2)/60
        Hm[139] = -betay**2/(10*L)
        Hm[143] = betay**2*(5*alphay**2 - 10*alphay + 8)/60

        # EBLm = (1/L)*int_0^L C[0, :]*{BLexx, BLgy, BLgz, BLkx, BLky, BLkz} dx,
        #        the first row of C times the rows of the strains
        EBLm[0] = -C00/L
        EBLm[1] = C01*alphay*betay/L
        EBLm[2] = C02*alphaz*betaz/L
        EBLm[3] = -C03/L
        EBLm[4] = -C02*alphaz*betaz/2 - C05/L
        EBLm[5] = C01*alphay*betay/2 + C04/L
        EBLm[6] = C00/L
        EBLm[7] = -C01*alphay*betay/L
        EBLm[8] = -C02*alphaz*betaz/L
        EBLm[9] = C03/L
        EBLm[10] = -C02*alphaz*betaz/2 + C05/L
        EBLm[11] = C01*alphay*betay/2 - C04/L

        eNLm[0] = 0.
        Nm[0] = 0.
        for i in range(NUM_NODES*DOF):
            BNLm[i] = 0.
            for j in range(NUM_NODES*DOF):
                BNLm[i] += Hm[NUM_NODES*DOF*i + j]*ue[j]
        for i in range(NUM_NODES*DOF):
            eNLm[0] += BNLm[i]*ue[i]/2.
            Nm[0] += EBLm[i]*ue[i]


    cdef void _update_probe_KCNLve(BeamC self, BeamProp prop) noexcept nogil:
        r"""Update the probe values of the nonlinear stiffness matrix

        The attribute ``KCNLve`` of the :class:`.BeamCProbe` is overwritten
        with the nonlinear stiffness matrix KCNL in element coordinates,
        stored row by row and evaluated at the displacements ``ue`` of the
        probe, from ``length`` and the attributes of ``prop`` read by
        :meth:`._update_means`. See :meth:`.update_KCNL`.

        Notes
        -----
        With the element means `\boldsymbol{H}_m`,
        `\overline{\boldsymbol{EB}}_L`, `\bar{\boldsymbol{B}}_{NL}` and
        `\bar{\epsilon}_{NL}` of :meth:`._update_means` and `\bar{N}_{NL} =
        C_{00} \bar{\epsilon}_{NL}`, with the entry `C_{00}` of the
        sectional stiffness matrix of ``prop``, `E A` when ``C`` is not set,
        the entry `(i, j)` is

        .. math::
            L \left( \overline{EB}_{L_i} \bar{B}_{NL_j}
            + \bar{B}_{NL_i} \overline{EB}_{L_j}
            + C_{00} \, \bar{B}_{NL_i} \bar{B}_{NL_j}
            + \bar{N}_{NL} H_{m_{ij}} \right)

        """
        cdef int i, j
        cdef double L, C00, eNLm, Nm, NNLm
        cdef double C[36]
        cdef double Hm[144]
        cdef double EBLm[12]
        cdef double BNLm[12]
        cdef double *KCNLve

        L = self.length
        prop.get_C(C)
        C00 = C[0]
        KCNLve = &self.probe.KCNLve[0]

        self._update_means(prop, Hm, EBLm, BNLm, &eNLm, &Nm)
        # axial force of the mean nonlinear axial strain
        NNLm = C00*eNLm

        for i in range(NUM_NODES*DOF):
            for j in range(NUM_NODES*DOF):
                KCNLve[NUM_NODES*DOF*i + j] = L*(
                    # KC0L = L*EBLm.T*BNLm
                      EBLm[i]*BNLm[j]
                    # KCL0 = L*BNLm.T*EBLm
                    + BNLm[i]*EBLm[j]
                    # KCLL = L*C00*BNLm.T*BNLm
                    + C00*BNLm[i]*BNLm[j]
                    # KGNL = L*NNLm*Hm
                    + NNLm*Hm[NUM_NODES*DOF*i + j]
                )


    cdef void _update_probe_finte_nonlinear(BeamC self,
                                            BeamProp prop) noexcept nogil:
        r"""Add the geometrically nonlinear terms to the probe internal forces

        The nonlinear terms, evaluated at the displacements ``ue`` of the
        probe, from ``length`` and the attributes of ``prop`` read by
        :meth:`._update_means`, are added to the attribute ``finte`` of the
        :class:`.BeamCProbe`, which must already contain the linear internal
        forces, see :meth:`.update_probe_finte`. See :meth:`.update_KCNL`.

        Notes
        -----
        With the element means of :meth:`._update_means`, the added terms
        are

        .. math::
            L \left( \overline{\boldsymbol{EB}}_L^\top \bar{\epsilon}_{NL}
            + \bar{\boldsymbol{B}}_{NL}^\top \left( N_m + C_{00}
            \bar{\epsilon}_{NL} \right) \right)

        the gradient of `L N_m \bar{\epsilon}_{NL} + \frac{1}{2} C_{00} L
        \bar{\epsilon}_{NL}^2`, the strain energy with the axial strain
        `u_{,x} + \bar{\epsilon}_{NL}` minus its linear part, `C_{00}` being
        the first entry of the sectional stiffness matrix of ``prop``,
        such that ``finte`` becomes the gradient of the strain energy whose
        Hessian is KC0 + KCNL + KG (own derivation, asserted in
        ``theory/deriving_equations/beamc/derive_beamc_KC0L_KCL0_KCLL_KGLL_sparse.py``).

        """
        cdef int i
        cdef double L, C00, eNLm, Nm, NNLm
        cdef double C[36]
        cdef double Hm[144]
        cdef double EBLm[12]
        cdef double BNLm[12]
        cdef double *finte

        L = self.length
        prop.get_C(C)
        C00 = C[0]
        finte = &self.probe.finte[0]

        self._update_means(prop, Hm, EBLm, BNLm, &eNLm, &Nm)
        # axial force of the mean nonlinear axial strain
        NNLm = C00*eNLm

        for i in range(NUM_NODES*DOF):
            finte[i] += L*(
                # L*EBLm.T*eNLm
                  EBLm[i]*eNLm
                # L*BNLm.T*(Nm + NNLm)
                + BNLm[i]*(Nm + NNLm)
            )


    cpdef void update_KCNL(BeamC self,
                           long [::1] KCNLr,
                           long [::1] KCNLc,
                           double [::1] KCNLv,
                           BeamProp prop,
                           int update_KCNLv_only=0
                           ):
        r"""Update sparse vectors for the nonlinear stiffness matrix KCNL

        The matrix, evaluated at the displacements ``probe.ue``, is computed
        from ``length``, the rotation matrix and the attributes of ``prop``
        read by :meth:`._update_probe_KCNLve`. The tangent stiffness matrix
        of the internal forces of :meth:`.update_fint` with ``nonlinear=1`` is
        ``KC0 + KCNL + KG``, see :meth:`.update_KC0` and :meth:`.update_KG`.

        Before this function is called, the probe :class:`.BeamCProbe` attribute
        of the :class:`.BeamC` object must be updated using
        :meth:`.update_probe_ue` with the current displacements; and
        :meth:`.update_probe_xe` with the node coordinates.

        The 144 entries of the matrix of the element, in global coordinates,
        are written at the positions ``init_k_KCNL`` to
        ``init_k_KCNL + 143``: the row and column indices, built from ``c1``
        and ``c2``, are stored in ``KCNLr`` and ``KCNLc``, and the values are
        added to those already in ``KCNLv``. The matrix in element coordinates
        is also stored in ``probe.KCNLve``, overwriting it.

        Parameters
        ----------
        KCNLr : np.array
            Array to store row positions of sparse values. 1-D array of
            integers of the C type ``long``, of size at least
            ``init_k_KCNL + 144``.
        KCNLc : np.array
            Array to store column positions of sparse values. Same type and
            size as ``KCNLr``.
        KCNLv : np.array
            Array to store sparse values. 1-D array of floats of size at least
            ``init_k_KCNL + 144``.
        prop : :class:`.BeamProp` object
            Beam property object from where the stiffness and mass attributes are
            read from.
        update_KCNLv_only : int, optional
            The default ``0`` means that the row and column indices ``KCNLr`` and
            ``KCNLc`` should also be updated. Any other value will only update the
            stiffness matrix values ``KCNLv``.

        Notes
        -----
        KCNL = KC0L + KCL0 + KCLL + KGNL, that is `\boldsymbol{K}_{CNL} +
        \boldsymbol{K}_{GNL}`, the nonlinear constitutive stiffness matrix
        `\boldsymbol{K}_{CNL}` of the terms of the nonlinear axial strain
        plus the geometric `\boldsymbol{K}_{GNL}` of the axial force of the
        nonlinear axial strain. Crisfield [1]_, Section 7.1.1, Eq. (7.24),
        keeps that axial force in the initial-stress matrix instead, that is
        in `\boldsymbol{K}_G + \boldsymbol{K}_{GNL}`; the tangent stiffness
        matrix is the same. The axial strain is that of
        von Kármán with its nonlinear part replaced by the element mean,
        Crisfield [1]_, Section 7.1.3, Eqs. (7.31)-(7.32), and Crisfield
        [2]_, Section 17.1.5, Eqs. (17.56a)-(17.56b):

        .. math::
            \epsilon_{xx} = u_{,x} + \bar{\epsilon}_{NL}, \qquad
            \bar{\epsilon}_{NL} = \frac{1}{L} \int_0^L \frac{1}{2} \left(
            v_{,x}^2 + w_{,x}^2 \right) dx = \frac{1}{2}
            \boldsymbol{u}_e^\top \boldsymbol{H}_m \boldsymbol{u}_e

        with `\boldsymbol{H}_m`, the gradient `\bar{\boldsymbol{B}}_{NL} =
        \boldsymbol{u}_e^\top \boldsymbol{H}_m` and the row
        `\overline{\boldsymbol{EB}}_L` of the mean axial force of the linear
        strains given in :meth:`._update_means`. With `\bar{N}_{NL} = C_{00}
        \bar{\epsilon}_{NL}`, `C_{00}` = ``C[0, 0]`` of :class:`.BeamProp`,
        `E A` when ``C`` is not set (own derivation, asserted in
        ``theory/deriving_equations/beamc/derive_beamc_KC0L_KCL0_KCLL_KGLL_sparse.py``):

        - KC0L = `L \, \overline{\boldsymbol{EB}}_L^\top \bar{\boldsymbol{B}}_{NL}`
        - KCL0 = KC0L`^\top`
        - KCLL = `L \, C_{00} \, \bar{\boldsymbol{B}}_{NL}^\top \bar{\boldsymbol{B}}_{NL}`
        - KGNL = `L \, \bar{N}_{NL} \boldsymbol{H}_m`

        The first three groups are the constitutive terms coupling the linear
        and the nonlinear parts of the axial strain. KGNL is geometric,
        carrying the axial force of the nonlinear axial strain. It is
        collected here so that :meth:`.update_KG` stays homogeneous of degree
        one in the displacements, which is what a linear buckling analysis
        needs. With it here,

        .. math::
            \boldsymbol{K}_T = \boldsymbol{K}_{C0} + \boldsymbol{K}_{CNL}(\boldsymbol{u}) + \boldsymbol{K}_{GNL}(\boldsymbol{u}) + \boldsymbol{K}_G(\boldsymbol{u})

        is the exact, symmetric Jacobian of the internal forces of
        :meth:`.update_fint` with ``nonlinear=1``, and a Newton-Raphson
        iteration built on them converges quadratically.

        With `u` linear and `v`, `w` cubic, the pointwise `(v_{,x}^2 +
        w_{,x}^2)/2` is a polynomial of degree four in `x` that the constant
        `u_{,x}` cannot balance, so that the axial strain cannot vanish in
        inextensional bending and coarse meshes are too stiff (membrane
        locking, [1]_, Section 7.1.3). The element mean removes it; [1]_
        justifies it with the Hu-Washizu principle and a constant membrane
        strain. With zero transverse nodal displacements, as in the local
        frame of a co-rotational element, and an infinite transverse shear
        stiffness, `\bar{\epsilon}_{NL}` is the term
        `\boldsymbol{\theta}_l^\top \boldsymbol{X} \boldsymbol{\theta}_l/60`
        of [2]_, Eq. (17.56a). The slopes `v_{,x}` and `w_{,x}` include the
        transverse shear strains, so that `\alpha_y`, `\beta_y`, `\alpha_z`
        and `\beta_z` enter `\boldsymbol{H}_m`. The von Kármán axial strain is
        valid for small strains and moderate rotations: the nonlinear axial
        strain is uniform over the cross section and along the element, the
        curvatures and the transverse shear strains are linear in the
        displacements, and the element coordinate system is not updated with
        the displacements.

        References
        ----------
        .. [1] Crisfield, M. A., Non-linear Finite Element Analysis of Solids
           and Structures, Volume 1: Essentials, John Wiley & Sons,
           Chichester, 1991.
        .. [2] Crisfield, M. A., Non-linear Finite Element Analysis of Solids
           and Structures, Volume 2: Advanced Topics, John Wiley & Sons,
           Chichester, 1997.

        """
        cdef int i, j, node_i, node_j, k, ke, m, n
        cdef int c[2]
        cdef double r[6][6]

        with nogil:
            # local to global transformation
            # translation DOFs
            r[0][0] = self.r11
            r[0][1] = self.r12
            r[0][2] = self.r13
            r[1][0] = self.r21
            r[1][1] = self.r22
            r[1][2] = self.r23
            r[2][0] = self.r31
            r[2][1] = self.r32
            r[2][2] = self.r33
            # rotation DOFs
            r[0+3][0+3] = self.r11
            r[0+3][1+3] = self.r12
            r[0+3][2+3] = self.r13
            r[1+3][0+3] = self.r21
            r[1+3][1+3] = self.r22
            r[1+3][2+3] = self.r23
            r[2+3][0+3] = self.r31
            r[2+3][1+3] = self.r32
            r[2+3][2+3] = self.r33
            # coupled translation-rotation DOFs
            for i in range(3):
                for j in range(3):
                    r[i][j+3] = 0.
                    r[i+3][j] = 0.

            if update_KCNLv_only == 0:
                # positions in the global stiffness matrix
                c[0] = self.c1
                c[1] = self.c2

                for node_i in range(NUM_NODES):
                    for m in range(DOF):
                        for node_j in range(NUM_NODES):
                            for n in range(DOF):
                                k = self.init_k_KCNL + 12*(node_i*DOF + m) + node_j*DOF + n
                                KCNLr[k] = c[node_i] + m
                                KCNLc[k] = c[node_j] + n

            self._update_probe_KCNLve(prop)

            # NOTE from element to global coordinates:
            #
            # Kg = T @ Ke @ T.T
            #
            # in tensor notation:
            #
            # Kg_{mn} = r_{mi} * Ke_{ij} * r_{nj}
            #
            for node_i in range(NUM_NODES):
                for m in range(DOF):
                    for node_j in range(NUM_NODES):
                        for n in range(DOF):
                            k = self.init_k_KCNL + 12*(node_i*DOF + m) + node_j*DOF + n
                            for i in range(DOF):
                                for j in range(DOF):
                                    ke = 12*(node_i*DOF + i) + node_j*DOF + j
                                    KCNLv[k] += r[m][i]*self.probe.KCNLve[ke]*r[n][j]


    cpdef void update_KG(BeamC self,
                         long [::1] KGr,
                         long [::1] KGc,
                         double [::1] KGv,
                         BeamProp prop,
                         int update_KGv_only=0
                         ):
        r"""Update sparse vectors for geometric stiffness matrix KG

        The 144 entries of the matrix of the element, in global coordinates,
        are written at the positions ``init_k_KG`` to ``init_k_KG + 143``:
        the row and column indices, built from ``c1`` and ``c2``, are stored
        in ``KGr`` and ``KGc``, and the values are added to those already in
        ``KGv``. The matrix depends on the axial force computed from the
        displacements ``probe.ue``, from ``length`` and from the entries
        ``C[0, :]``, ``C[1, 1]``, ``C[2, 2]``, ``C[4, 4]`` and ``C[5, 5]`` of
        the sectional stiffness matrix of ``prop``, see :class:`.BeamProp`.
        ``probe.ue``, ``length`` and the rotation matrix must be
        updated before, see :meth:`.update_probe_ue` and
        :meth:`.update_probe_xe`.

        Parameters
        ----------
        KGr : np.array
           Array to store row positions of sparse values. 1-D array of
           integers of the C type ``long``, of size at least
           ``init_k_KG + 144``.
        KGc : np.array
           Array to store column positions of sparse values. Same type and
           size as ``KGr``.
        KGv : np.array
            Array to store sparse values. 1-D array of floats of size at least
            ``init_k_KG + 144``.
        prop : :class:`.BeamProp` object
            Beam property object from where the stiffness and mass attributes
            are read from.
        update_KGv_only : int, optional
            The default ``0`` means that the row and column indices ``KGr``
            and ``KGc`` are also updated. Any other value will only update the
            stiffness matrix values ``KGv``.

        Notes
        -----
        The geometric stiffness matrix is

        .. math::
            \boldsymbol{K}_G = N_m \int_0^L \left[ \left(\frac{\partial \boldsymbol{S}^v}{\partial x}\right)^\top \frac{\partial \boldsymbol{S}^v}{\partial x}
            + \left(\frac{\partial \boldsymbol{S}^w}{\partial x}\right)^\top \frac{\partial \boldsymbol{S}^w}{\partial x} \right] dx
            = L N_m \boldsymbol{H}_m

        the Hessian of `L N_m \bar{\epsilon}_{NL}` at a constant `N_m`, where
        `\partial \boldsymbol{S}^v / \partial x` and `\partial
        \boldsymbol{S}^w / \partial x` give the slopes `v_{,x}` and `w_{,x}`
        with the consistent shape functions, `\bar{\epsilon}_{NL}` and
        `\boldsymbol{H}_m` are those of :meth:`.update_KCNL` and

        .. math::
            N_m = \frac{1}{L} \int_0^L N dx = \frac{1}{L} \left( C_{00} (u_2
            - u_1) + C_{03} ({r_x}_2 - {r_x}_1) + C_{04} ({r_z}_1 - {r_z}_2)
            + C_{05} ({r_y}_2 - {r_y}_1) \right) + C_{01} \gamma_{xy} +
            C_{02} \gamma_{xz}

        is the element mean of the axial force of the linear strains, `N =
        \boldsymbol{C}_{0,:} \boldsymbol{\epsilon}`, at ``probe.ue``, with
        the constant shear strains `\gamma_{xy}` and `\gamma_{xz}` and the
        first row of the sectional stiffness matrix, see the notes of
        :class:`.BeamC`; when ``C`` is not set, `N = E (A u_{,x} + A_y
        \kappa_y + A_z \kappa_z)` and `N_m = E (A (u_2 - u_1) + A_y ({r_z}_1
        - {r_z}_2) + A_z ({r_y}_2 - {r_y}_1))/L`. The mean follows from the
        mean nonlinear
        axial strain of :meth:`.update_KCNL`, with which the energy of the
        axial strain contains `\int_0^L N dx \, \bar{\epsilon}_{NL}`, such
        that ``KC0 + KCNL + KG`` is the exact Jacobian of the internal forces
        of :meth:`.update_fint` with ``nonlinear=1`` (own derivation, asserted
        in
        ``theory/deriving_equations/beamc/derive_beamc_KC0L_KCL0_KCLL_KGLL_sparse.py``);
        the geometric matrix of Crisfield [1]_, Eq. (7.95), also carries an
        axial force constant along the element. `N_m` is the axial force of
        the nodal internal forces, ``-finte[0] = finte[6]``. `N` varies
        linearly along the element only when ``C[0, 4]`` or ``C[0, 5]`` is
        non-zero, e.g. with non-zero ``Ay`` or ``Az``; otherwise it is
        uniform and `N_m = N`.

        `\boldsymbol{K}_G` is homogeneous of degree one in ``probe.ue``. With ``probe.ue``
        from the linear static solution of a reference load, the critical
        load multipliers `\lambda` of the linear buckling analysis solve
        `(\boldsymbol{K}_{C0} + \lambda \boldsymbol{K}_G) \boldsymbol{u} = \boldsymbol{0}`, see
        :func:`.linear_buckling`. The geometric term of the
        nonlinear axial strain is in :meth:`.update_KCNL`.

        References
        ----------
        .. [1] Crisfield, M. A., Non-linear Finite Element Analysis of Solids
           and Structures, Volume 1: Essentials, John Wiley & Sons,
           Chichester, 1991.

        """
        cdef double *ue
        cdef int c1, c2, k
        cdef double L, Nm
        cdef double r12, r13, r22, r23, r32, r33
        cdef double alphay, alphaz, betay, betaz
        cdef double C[36]
        cdef double C00, C01, C02, C03, C04, C05, C11, C22, C44, C55

        with nogil:
            L = self.length
            prop.get_C(C)
            C00 = C[0]
            C01 = C[1]
            C02 = C[2]
            C03 = C[3]
            C04 = C[4]
            C05 = C[5]
            C11 = C[7]
            C22 = C[14]
            C44 = C[28]
            C55 = C[35]

            # NOTE shear parameters of the shape functions from the diagonal of
            #      the sectional stiffness matrix, see the notes of BeamC
            alphay = -12*C44/(C11*L**2)
            alphaz = -12*C55/(C22*L**2)
            betay = 1/(1. - alphay)
            betaz = 1/(1. - alphaz)

            # local to global transformation
            r12 = self.r12
            r13 = self.r13
            r22 = self.r22
            r23 = self.r23
            r32 = self.r32
            r33 = self.r33

            ue = &self.probe.ue[0]

            if update_KGv_only == 0:
                # positions of nodes 1,2,3,4 in the global matrix
                c1 = self.c1
                c2 = self.c2

                k = self.init_k_KG
                KGr[k] = 0+c1
                KGc[k] = 0+c1
                k += 1
                KGr[k] = 0+c1
                KGc[k] = 1+c1
                k += 1
                KGr[k] = 0+c1
                KGc[k] = 2+c1
                k += 1
                KGr[k] = 0+c1
                KGc[k] = 3+c1
                k += 1
                KGr[k] = 0+c1
                KGc[k] = 4+c1
                k += 1
                KGr[k] = 0+c1
                KGc[k] = 5+c1
                k += 1
                KGr[k] = 0+c1
                KGc[k] = 0+c2
                k += 1
                KGr[k] = 0+c1
                KGc[k] = 1+c2
                k += 1
                KGr[k] = 0+c1
                KGc[k] = 2+c2
                k += 1
                KGr[k] = 0+c1
                KGc[k] = 3+c2
                k += 1
                KGr[k] = 0+c1
                KGc[k] = 4+c2
                k += 1
                KGr[k] = 0+c1
                KGc[k] = 5+c2
                k += 1
                KGr[k] = 1+c1
                KGc[k] = 0+c1
                k += 1
                KGr[k] = 1+c1
                KGc[k] = 1+c1
                k += 1
                KGr[k] = 1+c1
                KGc[k] = 2+c1
                k += 1
                KGr[k] = 1+c1
                KGc[k] = 3+c1
                k += 1
                KGr[k] = 1+c1
                KGc[k] = 4+c1
                k += 1
                KGr[k] = 1+c1
                KGc[k] = 5+c1
                k += 1
                KGr[k] = 1+c1
                KGc[k] = 0+c2
                k += 1
                KGr[k] = 1+c1
                KGc[k] = 1+c2
                k += 1
                KGr[k] = 1+c1
                KGc[k] = 2+c2
                k += 1
                KGr[k] = 1+c1
                KGc[k] = 3+c2
                k += 1
                KGr[k] = 1+c1
                KGc[k] = 4+c2
                k += 1
                KGr[k] = 1+c1
                KGc[k] = 5+c2
                k += 1
                KGr[k] = 2+c1
                KGc[k] = 0+c1
                k += 1
                KGr[k] = 2+c1
                KGc[k] = 1+c1
                k += 1
                KGr[k] = 2+c1
                KGc[k] = 2+c1
                k += 1
                KGr[k] = 2+c1
                KGc[k] = 3+c1
                k += 1
                KGr[k] = 2+c1
                KGc[k] = 4+c1
                k += 1
                KGr[k] = 2+c1
                KGc[k] = 5+c1
                k += 1
                KGr[k] = 2+c1
                KGc[k] = 0+c2
                k += 1
                KGr[k] = 2+c1
                KGc[k] = 1+c2
                k += 1
                KGr[k] = 2+c1
                KGc[k] = 2+c2
                k += 1
                KGr[k] = 2+c1
                KGc[k] = 3+c2
                k += 1
                KGr[k] = 2+c1
                KGc[k] = 4+c2
                k += 1
                KGr[k] = 2+c1
                KGc[k] = 5+c2
                k += 1
                KGr[k] = 3+c1
                KGc[k] = 0+c1
                k += 1
                KGr[k] = 3+c1
                KGc[k] = 1+c1
                k += 1
                KGr[k] = 3+c1
                KGc[k] = 2+c1
                k += 1
                KGr[k] = 3+c1
                KGc[k] = 3+c1
                k += 1
                KGr[k] = 3+c1
                KGc[k] = 4+c1
                k += 1
                KGr[k] = 3+c1
                KGc[k] = 5+c1
                k += 1
                KGr[k] = 3+c1
                KGc[k] = 0+c2
                k += 1
                KGr[k] = 3+c1
                KGc[k] = 1+c2
                k += 1
                KGr[k] = 3+c1
                KGc[k] = 2+c2
                k += 1
                KGr[k] = 3+c1
                KGc[k] = 3+c2
                k += 1
                KGr[k] = 3+c1
                KGc[k] = 4+c2
                k += 1
                KGr[k] = 3+c1
                KGc[k] = 5+c2
                k += 1
                KGr[k] = 4+c1
                KGc[k] = 0+c1
                k += 1
                KGr[k] = 4+c1
                KGc[k] = 1+c1
                k += 1
                KGr[k] = 4+c1
                KGc[k] = 2+c1
                k += 1
                KGr[k] = 4+c1
                KGc[k] = 3+c1
                k += 1
                KGr[k] = 4+c1
                KGc[k] = 4+c1
                k += 1
                KGr[k] = 4+c1
                KGc[k] = 5+c1
                k += 1
                KGr[k] = 4+c1
                KGc[k] = 0+c2
                k += 1
                KGr[k] = 4+c1
                KGc[k] = 1+c2
                k += 1
                KGr[k] = 4+c1
                KGc[k] = 2+c2
                k += 1
                KGr[k] = 4+c1
                KGc[k] = 3+c2
                k += 1
                KGr[k] = 4+c1
                KGc[k] = 4+c2
                k += 1
                KGr[k] = 4+c1
                KGc[k] = 5+c2
                k += 1
                KGr[k] = 5+c1
                KGc[k] = 0+c1
                k += 1
                KGr[k] = 5+c1
                KGc[k] = 1+c1
                k += 1
                KGr[k] = 5+c1
                KGc[k] = 2+c1
                k += 1
                KGr[k] = 5+c1
                KGc[k] = 3+c1
                k += 1
                KGr[k] = 5+c1
                KGc[k] = 4+c1
                k += 1
                KGr[k] = 5+c1
                KGc[k] = 5+c1
                k += 1
                KGr[k] = 5+c1
                KGc[k] = 0+c2
                k += 1
                KGr[k] = 5+c1
                KGc[k] = 1+c2
                k += 1
                KGr[k] = 5+c1
                KGc[k] = 2+c2
                k += 1
                KGr[k] = 5+c1
                KGc[k] = 3+c2
                k += 1
                KGr[k] = 5+c1
                KGc[k] = 4+c2
                k += 1
                KGr[k] = 5+c1
                KGc[k] = 5+c2
                k += 1
                KGr[k] = 0+c2
                KGc[k] = 0+c1
                k += 1
                KGr[k] = 0+c2
                KGc[k] = 1+c1
                k += 1
                KGr[k] = 0+c2
                KGc[k] = 2+c1
                k += 1
                KGr[k] = 0+c2
                KGc[k] = 3+c1
                k += 1
                KGr[k] = 0+c2
                KGc[k] = 4+c1
                k += 1
                KGr[k] = 0+c2
                KGc[k] = 5+c1
                k += 1
                KGr[k] = 0+c2
                KGc[k] = 0+c2
                k += 1
                KGr[k] = 0+c2
                KGc[k] = 1+c2
                k += 1
                KGr[k] = 0+c2
                KGc[k] = 2+c2
                k += 1
                KGr[k] = 0+c2
                KGc[k] = 3+c2
                k += 1
                KGr[k] = 0+c2
                KGc[k] = 4+c2
                k += 1
                KGr[k] = 0+c2
                KGc[k] = 5+c2
                k += 1
                KGr[k] = 1+c2
                KGc[k] = 0+c1
                k += 1
                KGr[k] = 1+c2
                KGc[k] = 1+c1
                k += 1
                KGr[k] = 1+c2
                KGc[k] = 2+c1
                k += 1
                KGr[k] = 1+c2
                KGc[k] = 3+c1
                k += 1
                KGr[k] = 1+c2
                KGc[k] = 4+c1
                k += 1
                KGr[k] = 1+c2
                KGc[k] = 5+c1
                k += 1
                KGr[k] = 1+c2
                KGc[k] = 0+c2
                k += 1
                KGr[k] = 1+c2
                KGc[k] = 1+c2
                k += 1
                KGr[k] = 1+c2
                KGc[k] = 2+c2
                k += 1
                KGr[k] = 1+c2
                KGc[k] = 3+c2
                k += 1
                KGr[k] = 1+c2
                KGc[k] = 4+c2
                k += 1
                KGr[k] = 1+c2
                KGc[k] = 5+c2
                k += 1
                KGr[k] = 2+c2
                KGc[k] = 0+c1
                k += 1
                KGr[k] = 2+c2
                KGc[k] = 1+c1
                k += 1
                KGr[k] = 2+c2
                KGc[k] = 2+c1
                k += 1
                KGr[k] = 2+c2
                KGc[k] = 3+c1
                k += 1
                KGr[k] = 2+c2
                KGc[k] = 4+c1
                k += 1
                KGr[k] = 2+c2
                KGc[k] = 5+c1
                k += 1
                KGr[k] = 2+c2
                KGc[k] = 0+c2
                k += 1
                KGr[k] = 2+c2
                KGc[k] = 1+c2
                k += 1
                KGr[k] = 2+c2
                KGc[k] = 2+c2
                k += 1
                KGr[k] = 2+c2
                KGc[k] = 3+c2
                k += 1
                KGr[k] = 2+c2
                KGc[k] = 4+c2
                k += 1
                KGr[k] = 2+c2
                KGc[k] = 5+c2
                k += 1
                KGr[k] = 3+c2
                KGc[k] = 0+c1
                k += 1
                KGr[k] = 3+c2
                KGc[k] = 1+c1
                k += 1
                KGr[k] = 3+c2
                KGc[k] = 2+c1
                k += 1
                KGr[k] = 3+c2
                KGc[k] = 3+c1
                k += 1
                KGr[k] = 3+c2
                KGc[k] = 4+c1
                k += 1
                KGr[k] = 3+c2
                KGc[k] = 5+c1
                k += 1
                KGr[k] = 3+c2
                KGc[k] = 0+c2
                k += 1
                KGr[k] = 3+c2
                KGc[k] = 1+c2
                k += 1
                KGr[k] = 3+c2
                KGc[k] = 2+c2
                k += 1
                KGr[k] = 3+c2
                KGc[k] = 3+c2
                k += 1
                KGr[k] = 3+c2
                KGc[k] = 4+c2
                k += 1
                KGr[k] = 3+c2
                KGc[k] = 5+c2
                k += 1
                KGr[k] = 4+c2
                KGc[k] = 0+c1
                k += 1
                KGr[k] = 4+c2
                KGc[k] = 1+c1
                k += 1
                KGr[k] = 4+c2
                KGc[k] = 2+c1
                k += 1
                KGr[k] = 4+c2
                KGc[k] = 3+c1
                k += 1
                KGr[k] = 4+c2
                KGc[k] = 4+c1
                k += 1
                KGr[k] = 4+c2
                KGc[k] = 5+c1
                k += 1
                KGr[k] = 4+c2
                KGc[k] = 0+c2
                k += 1
                KGr[k] = 4+c2
                KGc[k] = 1+c2
                k += 1
                KGr[k] = 4+c2
                KGc[k] = 2+c2
                k += 1
                KGr[k] = 4+c2
                KGc[k] = 3+c2
                k += 1
                KGr[k] = 4+c2
                KGc[k] = 4+c2
                k += 1
                KGr[k] = 4+c2
                KGc[k] = 5+c2
                k += 1
                KGr[k] = 5+c2
                KGc[k] = 0+c1
                k += 1
                KGr[k] = 5+c2
                KGc[k] = 1+c1
                k += 1
                KGr[k] = 5+c2
                KGc[k] = 2+c1
                k += 1
                KGr[k] = 5+c2
                KGc[k] = 3+c1
                k += 1
                KGr[k] = 5+c2
                KGc[k] = 4+c1
                k += 1
                KGr[k] = 5+c2
                KGc[k] = 5+c1
                k += 1
                KGr[k] = 5+c2
                KGc[k] = 0+c2
                k += 1
                KGr[k] = 5+c2
                KGc[k] = 1+c2
                k += 1
                KGr[k] = 5+c2
                KGc[k] = 2+c2
                k += 1
                KGr[k] = 5+c2
                KGc[k] = 3+c2
                k += 1
                KGr[k] = 5+c2
                KGc[k] = 4+c2
                k += 1
                KGr[k] = 5+c2
                KGc[k] = 5+c2

            # NOTE element mean of the axial force of the linear strains,
            #      see the notes
            # NOTE first row of C times the element means of the strains
            Nm = C01*alphay*betay*(ue[11]/2 + ue[5]/2 + (ue[1] - ue[7])/L) + C02*alphaz*betaz*(-ue[10]/2 - ue[4]/2 + (ue[2] - ue[8])/L) + (C00*(-ue[0] + ue[6]) + C03*(-ue[3] + ue[9]) + C04*(-ue[11] + ue[5]) + C05*(ue[10] - ue[4]))/L

            k = self.init_k_KG
            KGv[k] += Nm*betay**2*r12**2*(5*alphay**2 - 10*alphay + 6)/(5*L) + Nm*betaz**2*r13**2*(5*alphaz**2 - 10*alphaz + 6)/(5*L)
            k += 1
            KGv[k] += Nm*betay**2*r12*r22*(5*alphay**2 - 10*alphay + 6)/(5*L) + Nm*betaz**2*r13*r23*(5*alphaz**2 - 10*alphaz + 6)/(5*L)
            k += 1
            KGv[k] += Nm*betay**2*r12*r32*(5*alphay**2 - 10*alphay + 6)/(5*L) + Nm*betaz**2*r13*r33*(5*alphaz**2 - 10*alphaz + 6)/(5*L)
            k += 1
            KGv[k] += Nm*betay**2*r12*r13/10 - Nm*betaz**2*r12*r13/10
            k += 1
            KGv[k] += Nm*betay**2*r12*r23/10 - Nm*betaz**2*r13*r22/10
            k += 1
            KGv[k] += Nm*betay**2*r12*r33/10 - Nm*betaz**2*r13*r32/10
            k += 1
            KGv[k] += Nm*betay**2*r12**2*(-5*alphay**2 + 10*alphay - 6)/(5*L) + Nm*betaz**2*r13**2*(-5*alphaz**2 + 10*alphaz - 6)/(5*L)
            k += 1
            KGv[k] += Nm*betay**2*r12*r22*(-5*alphay**2 + 10*alphay - 6)/(5*L) + Nm*betaz**2*r13*r23*(-5*alphaz**2 + 10*alphaz - 6)/(5*L)
            k += 1
            KGv[k] += Nm*betay**2*r12*r32*(-5*alphay**2 + 10*alphay - 6)/(5*L) + Nm*betaz**2*r13*r33*(-5*alphaz**2 + 10*alphaz - 6)/(5*L)
            k += 1
            KGv[k] += Nm*betay**2*r12*r13/10 - Nm*betaz**2*r12*r13/10
            k += 1
            KGv[k] += Nm*betay**2*r12*r23/10 - Nm*betaz**2*r13*r22/10
            k += 1
            KGv[k] += Nm*betay**2*r12*r33/10 - Nm*betaz**2*r13*r32/10
            k += 1
            KGv[k] += Nm*betay**2*r12*r22*(5*alphay**2 - 10*alphay + 6)/(5*L) + Nm*betaz**2*r13*r23*(5*alphaz**2 - 10*alphaz + 6)/(5*L)
            k += 1
            KGv[k] += Nm*betay**2*r22**2*(5*alphay**2 - 10*alphay + 6)/(5*L) + Nm*betaz**2*r23**2*(5*alphaz**2 - 10*alphaz + 6)/(5*L)
            k += 1
            KGv[k] += Nm*betay**2*r22*r32*(5*alphay**2 - 10*alphay + 6)/(5*L) + Nm*betaz**2*r23*r33*(5*alphaz**2 - 10*alphaz + 6)/(5*L)
            k += 1
            KGv[k] += Nm*betay**2*r13*r22/10 - Nm*betaz**2*r12*r23/10
            k += 1
            KGv[k] += Nm*betay**2*r22*r23/10 - Nm*betaz**2*r22*r23/10
            k += 1
            KGv[k] += Nm*betay**2*r22*r33/10 - Nm*betaz**2*r23*r32/10
            k += 1
            KGv[k] += Nm*betay**2*r12*r22*(-5*alphay**2 + 10*alphay - 6)/(5*L) + Nm*betaz**2*r13*r23*(-5*alphaz**2 + 10*alphaz - 6)/(5*L)
            k += 1
            KGv[k] += Nm*betay**2*r22**2*(-5*alphay**2 + 10*alphay - 6)/(5*L) + Nm*betaz**2*r23**2*(-5*alphaz**2 + 10*alphaz - 6)/(5*L)
            k += 1
            KGv[k] += Nm*betay**2*r22*r32*(-5*alphay**2 + 10*alphay - 6)/(5*L) + Nm*betaz**2*r23*r33*(-5*alphaz**2 + 10*alphaz - 6)/(5*L)
            k += 1
            KGv[k] += Nm*betay**2*r13*r22/10 - Nm*betaz**2*r12*r23/10
            k += 1
            KGv[k] += Nm*betay**2*r22*r23/10 - Nm*betaz**2*r22*r23/10
            k += 1
            KGv[k] += Nm*betay**2*r22*r33/10 - Nm*betaz**2*r23*r32/10
            k += 1
            KGv[k] += Nm*betay**2*r12*r32*(5*alphay**2 - 10*alphay + 6)/(5*L) + Nm*betaz**2*r13*r33*(5*alphaz**2 - 10*alphaz + 6)/(5*L)
            k += 1
            KGv[k] += Nm*betay**2*r22*r32*(5*alphay**2 - 10*alphay + 6)/(5*L) + Nm*betaz**2*r23*r33*(5*alphaz**2 - 10*alphaz + 6)/(5*L)
            k += 1
            KGv[k] += Nm*betay**2*r32**2*(5*alphay**2 - 10*alphay + 6)/(5*L) + Nm*betaz**2*r33**2*(5*alphaz**2 - 10*alphaz + 6)/(5*L)
            k += 1
            KGv[k] += Nm*betay**2*r13*r32/10 - Nm*betaz**2*r12*r33/10
            k += 1
            KGv[k] += Nm*betay**2*r23*r32/10 - Nm*betaz**2*r22*r33/10
            k += 1
            KGv[k] += Nm*betay**2*r32*r33/10 - Nm*betaz**2*r32*r33/10
            k += 1
            KGv[k] += Nm*betay**2*r12*r32*(-5*alphay**2 + 10*alphay - 6)/(5*L) + Nm*betaz**2*r13*r33*(-5*alphaz**2 + 10*alphaz - 6)/(5*L)
            k += 1
            KGv[k] += Nm*betay**2*r22*r32*(-5*alphay**2 + 10*alphay - 6)/(5*L) + Nm*betaz**2*r23*r33*(-5*alphaz**2 + 10*alphaz - 6)/(5*L)
            k += 1
            KGv[k] += Nm*betay**2*r32**2*(-5*alphay**2 + 10*alphay - 6)/(5*L) + Nm*betaz**2*r33**2*(-5*alphaz**2 + 10*alphaz - 6)/(5*L)
            k += 1
            KGv[k] += Nm*betay**2*r13*r32/10 - Nm*betaz**2*r12*r33/10
            k += 1
            KGv[k] += Nm*betay**2*r23*r32/10 - Nm*betaz**2*r22*r33/10
            k += 1
            KGv[k] += Nm*betay**2*r32*r33/10 - Nm*betaz**2*r32*r33/10
            k += 1
            KGv[k] += Nm*betay**2*r12*r13/10 - Nm*betaz**2*r12*r13/10
            k += 1
            KGv[k] += Nm*betay**2*r13*r22/10 - Nm*betaz**2*r12*r23/10
            k += 1
            KGv[k] += Nm*betay**2*r13*r32/10 - Nm*betaz**2*r12*r33/10
            k += 1
            KGv[k] += L*Nm*betay**2*r13**2*(5*alphay**2 - 10*alphay + 8)/60 + L*Nm*betaz**2*r12**2*(5*alphaz**2 - 10*alphaz + 8)/60
            k += 1
            KGv[k] += L*Nm*betay**2*r13*r23*(5*alphay**2 - 10*alphay + 8)/60 + L*Nm*betaz**2*r12*r22*(5*alphaz**2 - 10*alphaz + 8)/60
            k += 1
            KGv[k] += L*Nm*betay**2*r13*r33*(5*alphay**2 - 10*alphay + 8)/60 + L*Nm*betaz**2*r12*r32*(5*alphaz**2 - 10*alphaz + 8)/60
            k += 1
            KGv[k] += -Nm*betay**2*r12*r13/10 + Nm*betaz**2*r12*r13/10
            k += 1
            KGv[k] += -Nm*betay**2*r13*r22/10 + Nm*betaz**2*r12*r23/10
            k += 1
            KGv[k] += -Nm*betay**2*r13*r32/10 + Nm*betaz**2*r12*r33/10
            k += 1
            KGv[k] += L*Nm*betay**2*r13**2*(-5*alphay**2 + 10*alphay - 2)/60 + L*Nm*betaz**2*r12**2*(-5*alphaz**2 + 10*alphaz - 2)/60
            k += 1
            KGv[k] += L*Nm*betay**2*r13*r23*(-5*alphay**2 + 10*alphay - 2)/60 + L*Nm*betaz**2*r12*r22*(-5*alphaz**2 + 10*alphaz - 2)/60
            k += 1
            KGv[k] += L*Nm*betay**2*r13*r33*(-5*alphay**2 + 10*alphay - 2)/60 + L*Nm*betaz**2*r12*r32*(-5*alphaz**2 + 10*alphaz - 2)/60
            k += 1
            KGv[k] += Nm*betay**2*r12*r23/10 - Nm*betaz**2*r13*r22/10
            k += 1
            KGv[k] += Nm*betay**2*r22*r23/10 - Nm*betaz**2*r22*r23/10
            k += 1
            KGv[k] += Nm*betay**2*r23*r32/10 - Nm*betaz**2*r22*r33/10
            k += 1
            KGv[k] += L*Nm*betay**2*r13*r23*(5*alphay**2 - 10*alphay + 8)/60 + L*Nm*betaz**2*r12*r22*(5*alphaz**2 - 10*alphaz + 8)/60
            k += 1
            KGv[k] += L*Nm*betay**2*r23**2*(5*alphay**2 - 10*alphay + 8)/60 + L*Nm*betaz**2*r22**2*(5*alphaz**2 - 10*alphaz + 8)/60
            k += 1
            KGv[k] += L*Nm*betay**2*r23*r33*(5*alphay**2 - 10*alphay + 8)/60 + L*Nm*betaz**2*r22*r32*(5*alphaz**2 - 10*alphaz + 8)/60
            k += 1
            KGv[k] += -Nm*betay**2*r12*r23/10 + Nm*betaz**2*r13*r22/10
            k += 1
            KGv[k] += -Nm*betay**2*r22*r23/10 + Nm*betaz**2*r22*r23/10
            k += 1
            KGv[k] += -Nm*betay**2*r23*r32/10 + Nm*betaz**2*r22*r33/10
            k += 1
            KGv[k] += L*Nm*betay**2*r13*r23*(-5*alphay**2 + 10*alphay - 2)/60 + L*Nm*betaz**2*r12*r22*(-5*alphaz**2 + 10*alphaz - 2)/60
            k += 1
            KGv[k] += L*Nm*betay**2*r23**2*(-5*alphay**2 + 10*alphay - 2)/60 + L*Nm*betaz**2*r22**2*(-5*alphaz**2 + 10*alphaz - 2)/60
            k += 1
            KGv[k] += L*Nm*betay**2*r23*r33*(-5*alphay**2 + 10*alphay - 2)/60 + L*Nm*betaz**2*r22*r32*(-5*alphaz**2 + 10*alphaz - 2)/60
            k += 1
            KGv[k] += Nm*betay**2*r12*r33/10 - Nm*betaz**2*r13*r32/10
            k += 1
            KGv[k] += Nm*betay**2*r22*r33/10 - Nm*betaz**2*r23*r32/10
            k += 1
            KGv[k] += Nm*betay**2*r32*r33/10 - Nm*betaz**2*r32*r33/10
            k += 1
            KGv[k] += L*Nm*betay**2*r13*r33*(5*alphay**2 - 10*alphay + 8)/60 + L*Nm*betaz**2*r12*r32*(5*alphaz**2 - 10*alphaz + 8)/60
            k += 1
            KGv[k] += L*Nm*betay**2*r23*r33*(5*alphay**2 - 10*alphay + 8)/60 + L*Nm*betaz**2*r22*r32*(5*alphaz**2 - 10*alphaz + 8)/60
            k += 1
            KGv[k] += L*Nm*betay**2*r33**2*(5*alphay**2 - 10*alphay + 8)/60 + L*Nm*betaz**2*r32**2*(5*alphaz**2 - 10*alphaz + 8)/60
            k += 1
            KGv[k] += -Nm*betay**2*r12*r33/10 + Nm*betaz**2*r13*r32/10
            k += 1
            KGv[k] += -Nm*betay**2*r22*r33/10 + Nm*betaz**2*r23*r32/10
            k += 1
            KGv[k] += -Nm*betay**2*r32*r33/10 + Nm*betaz**2*r32*r33/10
            k += 1
            KGv[k] += L*Nm*betay**2*r13*r33*(-5*alphay**2 + 10*alphay - 2)/60 + L*Nm*betaz**2*r12*r32*(-5*alphaz**2 + 10*alphaz - 2)/60
            k += 1
            KGv[k] += L*Nm*betay**2*r23*r33*(-5*alphay**2 + 10*alphay - 2)/60 + L*Nm*betaz**2*r22*r32*(-5*alphaz**2 + 10*alphaz - 2)/60
            k += 1
            KGv[k] += L*Nm*betay**2*r33**2*(-5*alphay**2 + 10*alphay - 2)/60 + L*Nm*betaz**2*r32**2*(-5*alphaz**2 + 10*alphaz - 2)/60
            k += 1
            KGv[k] += Nm*betay**2*r12**2*(-5*alphay**2 + 10*alphay - 6)/(5*L) + Nm*betaz**2*r13**2*(-5*alphaz**2 + 10*alphaz - 6)/(5*L)
            k += 1
            KGv[k] += Nm*betay**2*r12*r22*(-5*alphay**2 + 10*alphay - 6)/(5*L) + Nm*betaz**2*r13*r23*(-5*alphaz**2 + 10*alphaz - 6)/(5*L)
            k += 1
            KGv[k] += Nm*betay**2*r12*r32*(-5*alphay**2 + 10*alphay - 6)/(5*L) + Nm*betaz**2*r13*r33*(-5*alphaz**2 + 10*alphaz - 6)/(5*L)
            k += 1
            KGv[k] += -Nm*betay**2*r12*r13/10 + Nm*betaz**2*r12*r13/10
            k += 1
            KGv[k] += -Nm*betay**2*r12*r23/10 + Nm*betaz**2*r13*r22/10
            k += 1
            KGv[k] += -Nm*betay**2*r12*r33/10 + Nm*betaz**2*r13*r32/10
            k += 1
            KGv[k] += Nm*betay**2*r12**2*(5*alphay**2 - 10*alphay + 6)/(5*L) + Nm*betaz**2*r13**2*(5*alphaz**2 - 10*alphaz + 6)/(5*L)
            k += 1
            KGv[k] += Nm*betay**2*r12*r22*(5*alphay**2 - 10*alphay + 6)/(5*L) + Nm*betaz**2*r13*r23*(5*alphaz**2 - 10*alphaz + 6)/(5*L)
            k += 1
            KGv[k] += Nm*betay**2*r12*r32*(5*alphay**2 - 10*alphay + 6)/(5*L) + Nm*betaz**2*r13*r33*(5*alphaz**2 - 10*alphaz + 6)/(5*L)
            k += 1
            KGv[k] += -Nm*betay**2*r12*r13/10 + Nm*betaz**2*r12*r13/10
            k += 1
            KGv[k] += -Nm*betay**2*r12*r23/10 + Nm*betaz**2*r13*r22/10
            k += 1
            KGv[k] += -Nm*betay**2*r12*r33/10 + Nm*betaz**2*r13*r32/10
            k += 1
            KGv[k] += Nm*betay**2*r12*r22*(-5*alphay**2 + 10*alphay - 6)/(5*L) + Nm*betaz**2*r13*r23*(-5*alphaz**2 + 10*alphaz - 6)/(5*L)
            k += 1
            KGv[k] += Nm*betay**2*r22**2*(-5*alphay**2 + 10*alphay - 6)/(5*L) + Nm*betaz**2*r23**2*(-5*alphaz**2 + 10*alphaz - 6)/(5*L)
            k += 1
            KGv[k] += Nm*betay**2*r22*r32*(-5*alphay**2 + 10*alphay - 6)/(5*L) + Nm*betaz**2*r23*r33*(-5*alphaz**2 + 10*alphaz - 6)/(5*L)
            k += 1
            KGv[k] += -Nm*betay**2*r13*r22/10 + Nm*betaz**2*r12*r23/10
            k += 1
            KGv[k] += -Nm*betay**2*r22*r23/10 + Nm*betaz**2*r22*r23/10
            k += 1
            KGv[k] += -Nm*betay**2*r22*r33/10 + Nm*betaz**2*r23*r32/10
            k += 1
            KGv[k] += Nm*betay**2*r12*r22*(5*alphay**2 - 10*alphay + 6)/(5*L) + Nm*betaz**2*r13*r23*(5*alphaz**2 - 10*alphaz + 6)/(5*L)
            k += 1
            KGv[k] += Nm*betay**2*r22**2*(5*alphay**2 - 10*alphay + 6)/(5*L) + Nm*betaz**2*r23**2*(5*alphaz**2 - 10*alphaz + 6)/(5*L)
            k += 1
            KGv[k] += Nm*betay**2*r22*r32*(5*alphay**2 - 10*alphay + 6)/(5*L) + Nm*betaz**2*r23*r33*(5*alphaz**2 - 10*alphaz + 6)/(5*L)
            k += 1
            KGv[k] += -Nm*betay**2*r13*r22/10 + Nm*betaz**2*r12*r23/10
            k += 1
            KGv[k] += -Nm*betay**2*r22*r23/10 + Nm*betaz**2*r22*r23/10
            k += 1
            KGv[k] += -Nm*betay**2*r22*r33/10 + Nm*betaz**2*r23*r32/10
            k += 1
            KGv[k] += Nm*betay**2*r12*r32*(-5*alphay**2 + 10*alphay - 6)/(5*L) + Nm*betaz**2*r13*r33*(-5*alphaz**2 + 10*alphaz - 6)/(5*L)
            k += 1
            KGv[k] += Nm*betay**2*r22*r32*(-5*alphay**2 + 10*alphay - 6)/(5*L) + Nm*betaz**2*r23*r33*(-5*alphaz**2 + 10*alphaz - 6)/(5*L)
            k += 1
            KGv[k] += Nm*betay**2*r32**2*(-5*alphay**2 + 10*alphay - 6)/(5*L) + Nm*betaz**2*r33**2*(-5*alphaz**2 + 10*alphaz - 6)/(5*L)
            k += 1
            KGv[k] += -Nm*betay**2*r13*r32/10 + Nm*betaz**2*r12*r33/10
            k += 1
            KGv[k] += -Nm*betay**2*r23*r32/10 + Nm*betaz**2*r22*r33/10
            k += 1
            KGv[k] += -Nm*betay**2*r32*r33/10 + Nm*betaz**2*r32*r33/10
            k += 1
            KGv[k] += Nm*betay**2*r12*r32*(5*alphay**2 - 10*alphay + 6)/(5*L) + Nm*betaz**2*r13*r33*(5*alphaz**2 - 10*alphaz + 6)/(5*L)
            k += 1
            KGv[k] += Nm*betay**2*r22*r32*(5*alphay**2 - 10*alphay + 6)/(5*L) + Nm*betaz**2*r23*r33*(5*alphaz**2 - 10*alphaz + 6)/(5*L)
            k += 1
            KGv[k] += Nm*betay**2*r32**2*(5*alphay**2 - 10*alphay + 6)/(5*L) + Nm*betaz**2*r33**2*(5*alphaz**2 - 10*alphaz + 6)/(5*L)
            k += 1
            KGv[k] += -Nm*betay**2*r13*r32/10 + Nm*betaz**2*r12*r33/10
            k += 1
            KGv[k] += -Nm*betay**2*r23*r32/10 + Nm*betaz**2*r22*r33/10
            k += 1
            KGv[k] += -Nm*betay**2*r32*r33/10 + Nm*betaz**2*r32*r33/10
            k += 1
            KGv[k] += Nm*betay**2*r12*r13/10 - Nm*betaz**2*r12*r13/10
            k += 1
            KGv[k] += Nm*betay**2*r13*r22/10 - Nm*betaz**2*r12*r23/10
            k += 1
            KGv[k] += Nm*betay**2*r13*r32/10 - Nm*betaz**2*r12*r33/10
            k += 1
            KGv[k] += L*Nm*betay**2*r13**2*(-5*alphay**2 + 10*alphay - 2)/60 + L*Nm*betaz**2*r12**2*(-5*alphaz**2 + 10*alphaz - 2)/60
            k += 1
            KGv[k] += L*Nm*betay**2*r13*r23*(-5*alphay**2 + 10*alphay - 2)/60 + L*Nm*betaz**2*r12*r22*(-5*alphaz**2 + 10*alphaz - 2)/60
            k += 1
            KGv[k] += L*Nm*betay**2*r13*r33*(-5*alphay**2 + 10*alphay - 2)/60 + L*Nm*betaz**2*r12*r32*(-5*alphaz**2 + 10*alphaz - 2)/60
            k += 1
            KGv[k] += -Nm*betay**2*r12*r13/10 + Nm*betaz**2*r12*r13/10
            k += 1
            KGv[k] += -Nm*betay**2*r13*r22/10 + Nm*betaz**2*r12*r23/10
            k += 1
            KGv[k] += -Nm*betay**2*r13*r32/10 + Nm*betaz**2*r12*r33/10
            k += 1
            KGv[k] += L*Nm*betay**2*r13**2*(5*alphay**2 - 10*alphay + 8)/60 + L*Nm*betaz**2*r12**2*(5*alphaz**2 - 10*alphaz + 8)/60
            k += 1
            KGv[k] += L*Nm*betay**2*r13*r23*(5*alphay**2 - 10*alphay + 8)/60 + L*Nm*betaz**2*r12*r22*(5*alphaz**2 - 10*alphaz + 8)/60
            k += 1
            KGv[k] += L*Nm*betay**2*r13*r33*(5*alphay**2 - 10*alphay + 8)/60 + L*Nm*betaz**2*r12*r32*(5*alphaz**2 - 10*alphaz + 8)/60
            k += 1
            KGv[k] += Nm*betay**2*r12*r23/10 - Nm*betaz**2*r13*r22/10
            k += 1
            KGv[k] += Nm*betay**2*r22*r23/10 - Nm*betaz**2*r22*r23/10
            k += 1
            KGv[k] += Nm*betay**2*r23*r32/10 - Nm*betaz**2*r22*r33/10
            k += 1
            KGv[k] += L*Nm*betay**2*r13*r23*(-5*alphay**2 + 10*alphay - 2)/60 + L*Nm*betaz**2*r12*r22*(-5*alphaz**2 + 10*alphaz - 2)/60
            k += 1
            KGv[k] += L*Nm*betay**2*r23**2*(-5*alphay**2 + 10*alphay - 2)/60 + L*Nm*betaz**2*r22**2*(-5*alphaz**2 + 10*alphaz - 2)/60
            k += 1
            KGv[k] += L*Nm*betay**2*r23*r33*(-5*alphay**2 + 10*alphay - 2)/60 + L*Nm*betaz**2*r22*r32*(-5*alphaz**2 + 10*alphaz - 2)/60
            k += 1
            KGv[k] += -Nm*betay**2*r12*r23/10 + Nm*betaz**2*r13*r22/10
            k += 1
            KGv[k] += -Nm*betay**2*r22*r23/10 + Nm*betaz**2*r22*r23/10
            k += 1
            KGv[k] += -Nm*betay**2*r23*r32/10 + Nm*betaz**2*r22*r33/10
            k += 1
            KGv[k] += L*Nm*betay**2*r13*r23*(5*alphay**2 - 10*alphay + 8)/60 + L*Nm*betaz**2*r12*r22*(5*alphaz**2 - 10*alphaz + 8)/60
            k += 1
            KGv[k] += L*Nm*betay**2*r23**2*(5*alphay**2 - 10*alphay + 8)/60 + L*Nm*betaz**2*r22**2*(5*alphaz**2 - 10*alphaz + 8)/60
            k += 1
            KGv[k] += L*Nm*betay**2*r23*r33*(5*alphay**2 - 10*alphay + 8)/60 + L*Nm*betaz**2*r22*r32*(5*alphaz**2 - 10*alphaz + 8)/60
            k += 1
            KGv[k] += Nm*betay**2*r12*r33/10 - Nm*betaz**2*r13*r32/10
            k += 1
            KGv[k] += Nm*betay**2*r22*r33/10 - Nm*betaz**2*r23*r32/10
            k += 1
            KGv[k] += Nm*betay**2*r32*r33/10 - Nm*betaz**2*r32*r33/10
            k += 1
            KGv[k] += L*Nm*betay**2*r13*r33*(-5*alphay**2 + 10*alphay - 2)/60 + L*Nm*betaz**2*r12*r32*(-5*alphaz**2 + 10*alphaz - 2)/60
            k += 1
            KGv[k] += L*Nm*betay**2*r23*r33*(-5*alphay**2 + 10*alphay - 2)/60 + L*Nm*betaz**2*r22*r32*(-5*alphaz**2 + 10*alphaz - 2)/60
            k += 1
            KGv[k] += L*Nm*betay**2*r33**2*(-5*alphay**2 + 10*alphay - 2)/60 + L*Nm*betaz**2*r32**2*(-5*alphaz**2 + 10*alphaz - 2)/60
            k += 1
            KGv[k] += -Nm*betay**2*r12*r33/10 + Nm*betaz**2*r13*r32/10
            k += 1
            KGv[k] += -Nm*betay**2*r22*r33/10 + Nm*betaz**2*r23*r32/10
            k += 1
            KGv[k] += -Nm*betay**2*r32*r33/10 + Nm*betaz**2*r32*r33/10
            k += 1
            KGv[k] += L*Nm*betay**2*r13*r33*(5*alphay**2 - 10*alphay + 8)/60 + L*Nm*betaz**2*r12*r32*(5*alphaz**2 - 10*alphaz + 8)/60
            k += 1
            KGv[k] += L*Nm*betay**2*r23*r33*(5*alphay**2 - 10*alphay + 8)/60 + L*Nm*betaz**2*r22*r32*(5*alphaz**2 - 10*alphaz + 8)/60
            k += 1
            KGv[k] += L*Nm*betay**2*r33**2*(5*alphay**2 - 10*alphay + 8)/60 + L*Nm*betaz**2*r32**2*(5*alphaz**2 - 10*alphaz + 8)/60


    cpdef void update_M(BeamC self,
                        long [::1] Mr,
                        long [::1] Mc,
                        double [::1] Mv,
                        BeamProp prop,
                        int mtype=0,
                        ):
        r"""Update sparse vectors for mass matrix M

        The entries of the matrix of the element, in global coordinates, are
        written from the position ``init_k_M``: 144 entries with
        ``mtype=0`` and 36 entries with ``mtype=1``. The row and column
        indices, built from ``c1`` and ``c2``, are stored in ``Mr`` and
        ``Mc``, and the values are added to those already in ``Mv``. The
        matrix is computed from ``length``, the rotation matrix and the attributes
        ``intrho``, ``intrhoy``, ``intrhoz``, ``intrhoy2``, ``intrhoz2`` and
        ``intrhoyz``, and the diagonal entries ``C[1, 1]``, ``C[2, 2]``,
        ``C[4, 4]`` and ``C[5, 5]`` of the sectional stiffness matrix of
        ``prop``. ``length`` and the rotation matrix must be updated before, see :meth:`.update_rotation_matrix` and
        :meth:`.update_probe_xe`.

        Parameters
        ----------
        Mr : np.array
            Array to store row positions of sparse values. 1-D array of
            integers of the C type ``long``, of size at least
            ``init_k_M + 144`` (``init_k_M + 36`` with ``mtype=1``).
        Mc : np.array
            Array to store column positions of sparse values. Same type and
            size as ``Mr``.
        Mv : np.array
            Array to store sparse values. 1-D array of floats, of the same
            size as ``Mr``.
        prop : :class:`.BeamProp` object
            Beam property object from where the inertial attributes are read.
        mtype : int, optional
            0 for consistent mass matrix, 1 for lumped mass matrix. Default
            ``0``. The element has no reduced-integration mass matrix,
            ``2``.

        Raises
        ------
        ValueError
            If ``mtype`` is not 0 or 1.

        Notes
        -----
        The mass matrices come from the kinetic energy of the displacement
        field of the notes of :class:`.BeamC`, integrated over the cross
        section with the inertial attributes of :class:`.BeamProp`:

        .. math::
            T = \frac{1}{2} \int_0^L \Big[ m (\dot{u}^2 + \dot{v}^2 + \dot{w}^2)
                + m_{yy} \dot{r}_z^2 + m_{zz} \dot{r}_y^2
                - 2 m_{yz} \dot{r}_y \dot{r}_z
                - 2 m_y \dot{u} \dot{r}_z + 2 m_z \dot{u} \dot{r}_y
                - 2 m_z \dot{v} \dot{r}_x + 2 m_y \dot{w} \dot{r}_x
                + (m_{yy} + m_{zz}) \dot{r}_x^2 \Big] dx

        with `m` = ``intrho``, `m_y` = ``intrhoy``, `m_z` = ``intrhoz``,
        `m_{yy}` = ``intrhoy2``, `m_{zz}` = ``intrhoz2`` and `m_{yz}` =
        ``intrhoyz``.

        ``mtype=0`` gives the consistent mass matrix, the Hessian of `T` with
        respect to the nodal velocities, interpolated with the consistent
        shape functions of :class:`.BeamC`, which depend on ``alphay`` and
        ``alphaz`` and hence on ``C[1, 1]``, ``C[2, 2]``, ``C[4, 4]`` and
        ``C[5, 5]``, i.e. on ``E``, ``G12``, ``G13``, ``A``, ``Iyy`` and
        ``Izz`` when ``C`` is not set. The
        integral along the element is computed analytically (exactly), and
        the rotary inertia is included.

        ``mtype=1`` gives the lumped mass matrix, diagonal in element
        coordinates: at each node, `L m/2` for the three translations, `L
        (m_{yy} + m_{zz})/2` for `r_x`, `L m_{zz}/2` for `r_y` and `L
        m_{yy}/2` for `r_z`. It corresponds to the two-point Gauss-Lobatto
        quadrature of `T`, whose points are the nodes, without the coupling
        terms of `m_y`, `m_z` and `m_{yz}`, which are not included.

        The lumped mass matrix follows the lumped mass matrix by Lobatto
        integration of Brockman (1987) [1]_, p. 2348, adapted to the beam:
        the two-point Gauss-Lobatto quadrature of `T`, whose points are the
        nodes, gives a diagonal matrix only without the first moments of the
        density, which are therefore dropped. The consistent mass matrix is
        the classical one, see Petyt (2010) [2]_: Sec. 3.3, Eq. (3.59), for
        the axial terms, Sec. 3.4, Eq. (3.102), with `m_{yy} + m_{zz}` in
        place of `\rho I_x`, for the torsional terms, and Sec. 3.9, Eqs.
        (3.234) and (3.237), for the bending terms with the shape functions
        of static equilibrium and the rotary inertia, which
        :class:`.BeamC` reproduces in each plane of bending when `m_y`, `m_z`
        and `m_{yz}` are zero.

        References
        ----------
        .. [1] Brockman, R. A., 1987, “Dynamics of the Bilinear Mindlin Plate
           Element,” Int. J. Numer. Methods Eng., 24(12), pp. 2343–2356.
           https://onlinelibrary.wiley.com/doi/pdf/10.1002/nme.1620241208
        .. [2] Petyt, M., 2010, Introduction to Finite Element Vibration
           Analysis, 2nd ed., Cambridge University Press, Cambridge.
           https://doi.org/10.1017/CBO9780511761195

        """
        cdef int c1, c2, k
        cdef double intrho, intrhoy, intrhoz, intrhoy2, intrhoz2, intrhoyz
        cdef double r11, r12, r13, r21, r22, r23, r31, r32, r33
        cdef double L
        cdef double alphay, alphaz, betay, betaz
        cdef double C[36]
        cdef double C11, C22, C44, C55

        # NOTE checked once per call, before the code without the GIL
        if mtype != 0 and mtype != 1:
            raise ValueError('mtype=%d is not supported by BeamC.update_M, '
                             'use 0 (consistent) or 1 (lumped); BeamC has '
                             'no reduced-integration mass matrix (2)'
                             % mtype)

        with nogil:
            L = self.length
            intrho = prop.intrho
            intrhoy = prop.intrhoy
            intrhoz = prop.intrhoz
            intrhoy2 = prop.intrhoy2
            intrhoz2 = prop.intrhoz2
            intrhoyz = prop.intrhoyz
            prop.get_C(C)
            C11 = C[7]
            C22 = C[14]
            C44 = C[28]
            C55 = C[35]

            # NOTE shear parameters of the shape functions from the diagonal of
            #      the sectional stiffness matrix, see the notes of BeamC
            alphay = -12*C44/(C11*L**2)
            alphaz = -12*C55/(C22*L**2)
            betay = 1/(1. - alphay)
            betaz = 1/(1. - alphaz)

            # local to global transformation
            r11 = self.r11
            r12 = self.r12
            r13 = self.r13
            r21 = self.r21
            r22 = self.r22
            r23 = self.r23
            r31 = self.r31
            r32 = self.r32
            r33 = self.r33

            # positions the global matrices
            c1 = self.c1
            c2 = self.c2

            if mtype == 0: # M_cons consistent mass matrix, using two-point Gauss-Legendre quadrature
                k = self.init_k_M
                Mr[k] = 0+c1
                Mc[k] = 0+c1
                k += 1
                Mr[k] = 0+c1
                Mc[k] = 1+c1
                k += 1
                Mr[k] = 0+c1
                Mc[k] = 2+c1
                k += 1
                Mr[k] = 0+c1
                Mc[k] = 3+c1
                k += 1
                Mr[k] = 0+c1
                Mc[k] = 4+c1
                k += 1
                Mr[k] = 0+c1
                Mc[k] = 5+c1
                k += 1
                Mr[k] = 0+c1
                Mc[k] = 0+c2
                k += 1
                Mr[k] = 0+c1
                Mc[k] = 1+c2
                k += 1
                Mr[k] = 0+c1
                Mc[k] = 2+c2
                k += 1
                Mr[k] = 0+c1
                Mc[k] = 3+c2
                k += 1
                Mr[k] = 0+c1
                Mc[k] = 4+c2
                k += 1
                Mr[k] = 0+c1
                Mc[k] = 5+c2
                k += 1
                Mr[k] = 1+c1
                Mc[k] = 0+c1
                k += 1
                Mr[k] = 1+c1
                Mc[k] = 1+c1
                k += 1
                Mr[k] = 1+c1
                Mc[k] = 2+c1
                k += 1
                Mr[k] = 1+c1
                Mc[k] = 3+c1
                k += 1
                Mr[k] = 1+c1
                Mc[k] = 4+c1
                k += 1
                Mr[k] = 1+c1
                Mc[k] = 5+c1
                k += 1
                Mr[k] = 1+c1
                Mc[k] = 0+c2
                k += 1
                Mr[k] = 1+c1
                Mc[k] = 1+c2
                k += 1
                Mr[k] = 1+c1
                Mc[k] = 2+c2
                k += 1
                Mr[k] = 1+c1
                Mc[k] = 3+c2
                k += 1
                Mr[k] = 1+c1
                Mc[k] = 4+c2
                k += 1
                Mr[k] = 1+c1
                Mc[k] = 5+c2
                k += 1
                Mr[k] = 2+c1
                Mc[k] = 0+c1
                k += 1
                Mr[k] = 2+c1
                Mc[k] = 1+c1
                k += 1
                Mr[k] = 2+c1
                Mc[k] = 2+c1
                k += 1
                Mr[k] = 2+c1
                Mc[k] = 3+c1
                k += 1
                Mr[k] = 2+c1
                Mc[k] = 4+c1
                k += 1
                Mr[k] = 2+c1
                Mc[k] = 5+c1
                k += 1
                Mr[k] = 2+c1
                Mc[k] = 0+c2
                k += 1
                Mr[k] = 2+c1
                Mc[k] = 1+c2
                k += 1
                Mr[k] = 2+c1
                Mc[k] = 2+c2
                k += 1
                Mr[k] = 2+c1
                Mc[k] = 3+c2
                k += 1
                Mr[k] = 2+c1
                Mc[k] = 4+c2
                k += 1
                Mr[k] = 2+c1
                Mc[k] = 5+c2
                k += 1
                Mr[k] = 3+c1
                Mc[k] = 0+c1
                k += 1
                Mr[k] = 3+c1
                Mc[k] = 1+c1
                k += 1
                Mr[k] = 3+c1
                Mc[k] = 2+c1
                k += 1
                Mr[k] = 3+c1
                Mc[k] = 3+c1
                k += 1
                Mr[k] = 3+c1
                Mc[k] = 4+c1
                k += 1
                Mr[k] = 3+c1
                Mc[k] = 5+c1
                k += 1
                Mr[k] = 3+c1
                Mc[k] = 0+c2
                k += 1
                Mr[k] = 3+c1
                Mc[k] = 1+c2
                k += 1
                Mr[k] = 3+c1
                Mc[k] = 2+c2
                k += 1
                Mr[k] = 3+c1
                Mc[k] = 3+c2
                k += 1
                Mr[k] = 3+c1
                Mc[k] = 4+c2
                k += 1
                Mr[k] = 3+c1
                Mc[k] = 5+c2
                k += 1
                Mr[k] = 4+c1
                Mc[k] = 0+c1
                k += 1
                Mr[k] = 4+c1
                Mc[k] = 1+c1
                k += 1
                Mr[k] = 4+c1
                Mc[k] = 2+c1
                k += 1
                Mr[k] = 4+c1
                Mc[k] = 3+c1
                k += 1
                Mr[k] = 4+c1
                Mc[k] = 4+c1
                k += 1
                Mr[k] = 4+c1
                Mc[k] = 5+c1
                k += 1
                Mr[k] = 4+c1
                Mc[k] = 0+c2
                k += 1
                Mr[k] = 4+c1
                Mc[k] = 1+c2
                k += 1
                Mr[k] = 4+c1
                Mc[k] = 2+c2
                k += 1
                Mr[k] = 4+c1
                Mc[k] = 3+c2
                k += 1
                Mr[k] = 4+c1
                Mc[k] = 4+c2
                k += 1
                Mr[k] = 4+c1
                Mc[k] = 5+c2
                k += 1
                Mr[k] = 5+c1
                Mc[k] = 0+c1
                k += 1
                Mr[k] = 5+c1
                Mc[k] = 1+c1
                k += 1
                Mr[k] = 5+c1
                Mc[k] = 2+c1
                k += 1
                Mr[k] = 5+c1
                Mc[k] = 3+c1
                k += 1
                Mr[k] = 5+c1
                Mc[k] = 4+c1
                k += 1
                Mr[k] = 5+c1
                Mc[k] = 5+c1
                k += 1
                Mr[k] = 5+c1
                Mc[k] = 0+c2
                k += 1
                Mr[k] = 5+c1
                Mc[k] = 1+c2
                k += 1
                Mr[k] = 5+c1
                Mc[k] = 2+c2
                k += 1
                Mr[k] = 5+c1
                Mc[k] = 3+c2
                k += 1
                Mr[k] = 5+c1
                Mc[k] = 4+c2
                k += 1
                Mr[k] = 5+c1
                Mc[k] = 5+c2
                k += 1
                Mr[k] = 0+c2
                Mc[k] = 0+c1
                k += 1
                Mr[k] = 0+c2
                Mc[k] = 1+c1
                k += 1
                Mr[k] = 0+c2
                Mc[k] = 2+c1
                k += 1
                Mr[k] = 0+c2
                Mc[k] = 3+c1
                k += 1
                Mr[k] = 0+c2
                Mc[k] = 4+c1
                k += 1
                Mr[k] = 0+c2
                Mc[k] = 5+c1
                k += 1
                Mr[k] = 0+c2
                Mc[k] = 0+c2
                k += 1
                Mr[k] = 0+c2
                Mc[k] = 1+c2
                k += 1
                Mr[k] = 0+c2
                Mc[k] = 2+c2
                k += 1
                Mr[k] = 0+c2
                Mc[k] = 3+c2
                k += 1
                Mr[k] = 0+c2
                Mc[k] = 4+c2
                k += 1
                Mr[k] = 0+c2
                Mc[k] = 5+c2
                k += 1
                Mr[k] = 1+c2
                Mc[k] = 0+c1
                k += 1
                Mr[k] = 1+c2
                Mc[k] = 1+c1
                k += 1
                Mr[k] = 1+c2
                Mc[k] = 2+c1
                k += 1
                Mr[k] = 1+c2
                Mc[k] = 3+c1
                k += 1
                Mr[k] = 1+c2
                Mc[k] = 4+c1
                k += 1
                Mr[k] = 1+c2
                Mc[k] = 5+c1
                k += 1
                Mr[k] = 1+c2
                Mc[k] = 0+c2
                k += 1
                Mr[k] = 1+c2
                Mc[k] = 1+c2
                k += 1
                Mr[k] = 1+c2
                Mc[k] = 2+c2
                k += 1
                Mr[k] = 1+c2
                Mc[k] = 3+c2
                k += 1
                Mr[k] = 1+c2
                Mc[k] = 4+c2
                k += 1
                Mr[k] = 1+c2
                Mc[k] = 5+c2
                k += 1
                Mr[k] = 2+c2
                Mc[k] = 0+c1
                k += 1
                Mr[k] = 2+c2
                Mc[k] = 1+c1
                k += 1
                Mr[k] = 2+c2
                Mc[k] = 2+c1
                k += 1
                Mr[k] = 2+c2
                Mc[k] = 3+c1
                k += 1
                Mr[k] = 2+c2
                Mc[k] = 4+c1
                k += 1
                Mr[k] = 2+c2
                Mc[k] = 5+c1
                k += 1
                Mr[k] = 2+c2
                Mc[k] = 0+c2
                k += 1
                Mr[k] = 2+c2
                Mc[k] = 1+c2
                k += 1
                Mr[k] = 2+c2
                Mc[k] = 2+c2
                k += 1
                Mr[k] = 2+c2
                Mc[k] = 3+c2
                k += 1
                Mr[k] = 2+c2
                Mc[k] = 4+c2
                k += 1
                Mr[k] = 2+c2
                Mc[k] = 5+c2
                k += 1
                Mr[k] = 3+c2
                Mc[k] = 0+c1
                k += 1
                Mr[k] = 3+c2
                Mc[k] = 1+c1
                k += 1
                Mr[k] = 3+c2
                Mc[k] = 2+c1
                k += 1
                Mr[k] = 3+c2
                Mc[k] = 3+c1
                k += 1
                Mr[k] = 3+c2
                Mc[k] = 4+c1
                k += 1
                Mr[k] = 3+c2
                Mc[k] = 5+c1
                k += 1
                Mr[k] = 3+c2
                Mc[k] = 0+c2
                k += 1
                Mr[k] = 3+c2
                Mc[k] = 1+c2
                k += 1
                Mr[k] = 3+c2
                Mc[k] = 2+c2
                k += 1
                Mr[k] = 3+c2
                Mc[k] = 3+c2
                k += 1
                Mr[k] = 3+c2
                Mc[k] = 4+c2
                k += 1
                Mr[k] = 3+c2
                Mc[k] = 5+c2
                k += 1
                Mr[k] = 4+c2
                Mc[k] = 0+c1
                k += 1
                Mr[k] = 4+c2
                Mc[k] = 1+c1
                k += 1
                Mr[k] = 4+c2
                Mc[k] = 2+c1
                k += 1
                Mr[k] = 4+c2
                Mc[k] = 3+c1
                k += 1
                Mr[k] = 4+c2
                Mc[k] = 4+c1
                k += 1
                Mr[k] = 4+c2
                Mc[k] = 5+c1
                k += 1
                Mr[k] = 4+c2
                Mc[k] = 0+c2
                k += 1
                Mr[k] = 4+c2
                Mc[k] = 1+c2
                k += 1
                Mr[k] = 4+c2
                Mc[k] = 2+c2
                k += 1
                Mr[k] = 4+c2
                Mc[k] = 3+c2
                k += 1
                Mr[k] = 4+c2
                Mc[k] = 4+c2
                k += 1
                Mr[k] = 4+c2
                Mc[k] = 5+c2
                k += 1
                Mr[k] = 5+c2
                Mc[k] = 0+c1
                k += 1
                Mr[k] = 5+c2
                Mc[k] = 1+c1
                k += 1
                Mr[k] = 5+c2
                Mc[k] = 2+c1
                k += 1
                Mr[k] = 5+c2
                Mc[k] = 3+c1
                k += 1
                Mr[k] = 5+c2
                Mc[k] = 4+c1
                k += 1
                Mr[k] = 5+c2
                Mc[k] = 5+c1
                k += 1
                Mr[k] = 5+c2
                Mc[k] = 0+c2
                k += 1
                Mr[k] = 5+c2
                Mc[k] = 1+c2
                k += 1
                Mr[k] = 5+c2
                Mc[k] = 2+c2
                k += 1
                Mr[k] = 5+c2
                Mc[k] = 3+c2
                k += 1
                Mr[k] = 5+c2
                Mc[k] = 4+c2
                k += 1
                Mr[k] = 5+c2
                Mc[k] = 5+c2

                k = self.init_k_M
                Mv[k] += r11*(L*intrho*r11/3 + betay*intrhoy*r12/2 + betaz*intrhoz*r13/2) + r12*(betay*intrhoy*r11/2 + betay**2*r12*(70*L**2*alphay**2*intrho - 147*L**2*alphay*intrho + 78*L**2*intrho + 252*intrhoy2)/(210*L) + 6*betay*betaz*intrhoyz*r13/(5*L)) + r13*(betaz*intrhoz*r11/2 + 6*betay*betaz*intrhoyz*r12/(5*L) + betaz**2*r13*(70*L**2*alphaz**2*intrho - 147*L**2*alphaz*intrho + 78*L**2*intrho + 252*intrhoz2)/(210*L))
                k += 1
                Mv[k] += r21*(L*intrho*r11/3 + betay*intrhoy*r12/2 + betaz*intrhoz*r13/2) + r22*(betay*intrhoy*r11/2 + betay**2*r12*(70*L**2*alphay**2*intrho - 147*L**2*alphay*intrho + 78*L**2*intrho + 252*intrhoy2)/(210*L) + 6*betay*betaz*intrhoyz*r13/(5*L)) + r23*(betaz*intrhoz*r11/2 + 6*betay*betaz*intrhoyz*r12/(5*L) + betaz**2*r13*(70*L**2*alphaz**2*intrho - 147*L**2*alphaz*intrho + 78*L**2*intrho + 252*intrhoz2)/(210*L))
                k += 1
                Mv[k] += r31*(L*intrho*r11/3 + betay*intrhoy*r12/2 + betaz*intrhoz*r13/2) + r32*(betay*intrhoy*r11/2 + betay**2*r12*(70*L**2*alphay**2*intrho - 147*L**2*alphay*intrho + 78*L**2*intrho + 252*intrhoy2)/(210*L) + 6*betay*betaz*intrhoyz*r13/(5*L)) + r33*(betaz*intrhoz*r11/2 + 6*betay*betaz*intrhoyz*r12/(5*L) + betaz**2*r13*(70*L**2*alphaz**2*intrho - 147*L**2*alphaz*intrho + 78*L**2*intrho + 252*intrhoz2)/(210*L))
                k += 1
                Mv[k] += r11*(L*betay*intrhoz*r12*(20*alphay - 21)/60 + L*betaz*intrhoy*r13*(21 - 20*alphaz)/60) + r12*(L*betaz*intrhoz*r11*(1 - 4*alphaz)/12 + betay*betaz*intrhoyz*r12*(-5*alphaz - 1)/10 + betaz**2*r13*(-35*L**2*alphaz**2*intrho + 77*L**2*alphaz*intrho - 44*L**2*intrho - 420*alphaz*intrhoz2 - 84*intrhoz2)/840) + r13*(L*betay*intrhoy*r11*(4*alphay - 1)/12 + betay**2*r12*(35*L**2*alphay**2*intrho - 77*L**2*alphay*intrho + 44*L**2*intrho + 420*alphay*intrhoy2 + 84*intrhoy2)/840 + betay*betaz*intrhoyz*r13*(5*alphay + 1)/10)
                k += 1
                Mv[k] += r21*(L*betay*intrhoz*r12*(20*alphay - 21)/60 + L*betaz*intrhoy*r13*(21 - 20*alphaz)/60) + r22*(L*betaz*intrhoz*r11*(1 - 4*alphaz)/12 + betay*betaz*intrhoyz*r12*(-5*alphaz - 1)/10 + betaz**2*r13*(-35*L**2*alphaz**2*intrho + 77*L**2*alphaz*intrho - 44*L**2*intrho - 420*alphaz*intrhoz2 - 84*intrhoz2)/840) + r23*(L*betay*intrhoy*r11*(4*alphay - 1)/12 + betay**2*r12*(35*L**2*alphay**2*intrho - 77*L**2*alphay*intrho + 44*L**2*intrho + 420*alphay*intrhoy2 + 84*intrhoy2)/840 + betay*betaz*intrhoyz*r13*(5*alphay + 1)/10)
                k += 1
                Mv[k] += r31*(L*betay*intrhoz*r12*(20*alphay - 21)/60 + L*betaz*intrhoy*r13*(21 - 20*alphaz)/60) + r32*(L*betaz*intrhoz*r11*(1 - 4*alphaz)/12 + betay*betaz*intrhoyz*r12*(-5*alphaz - 1)/10 + betaz**2*r13*(-35*L**2*alphaz**2*intrho + 77*L**2*alphaz*intrho - 44*L**2*intrho - 420*alphaz*intrhoz2 - 84*intrhoz2)/840) + r33*(L*betay*intrhoy*r11*(4*alphay - 1)/12 + betay**2*r12*(35*L**2*alphay**2*intrho - 77*L**2*alphay*intrho + 44*L**2*intrho + 420*alphay*intrhoy2 + 84*intrhoy2)/840 + betay*betaz*intrhoyz*r13*(5*alphay + 1)/10)
                k += 1
                Mv[k] += r11*(L*intrho*r11/6 + betay*intrhoy*r12/2 + betaz*intrhoz*r13/2) + r12*(-betay*intrhoy*r11/2 + betay**2*r12*(35*L**2*alphay**2*intrho - 63*L**2*alphay*intrho + 27*L**2*intrho - 252*intrhoy2)/(210*L) - 6*betay*betaz*intrhoyz*r13/(5*L)) + r13*(-betaz*intrhoz*r11/2 - 6*betay*betaz*intrhoyz*r12/(5*L) + betaz**2*r13*(35*L**2*alphaz**2*intrho - 63*L**2*alphaz*intrho + 27*L**2*intrho - 252*intrhoz2)/(210*L))
                k += 1
                Mv[k] += r21*(L*intrho*r11/6 + betay*intrhoy*r12/2 + betaz*intrhoz*r13/2) + r22*(-betay*intrhoy*r11/2 + betay**2*r12*(35*L**2*alphay**2*intrho - 63*L**2*alphay*intrho + 27*L**2*intrho - 252*intrhoy2)/(210*L) - 6*betay*betaz*intrhoyz*r13/(5*L)) + r23*(-betaz*intrhoz*r11/2 - 6*betay*betaz*intrhoyz*r12/(5*L) + betaz**2*r13*(35*L**2*alphaz**2*intrho - 63*L**2*alphaz*intrho + 27*L**2*intrho - 252*intrhoz2)/(210*L))
                k += 1
                Mv[k] += r31*(L*intrho*r11/6 + betay*intrhoy*r12/2 + betaz*intrhoz*r13/2) + r32*(-betay*intrhoy*r11/2 + betay**2*r12*(35*L**2*alphay**2*intrho - 63*L**2*alphay*intrho + 27*L**2*intrho - 252*intrhoy2)/(210*L) - 6*betay*betaz*intrhoyz*r13/(5*L)) + r33*(-betaz*intrhoz*r11/2 - 6*betay*betaz*intrhoyz*r12/(5*L) + betaz**2*r13*(35*L**2*alphaz**2*intrho - 63*L**2*alphaz*intrho + 27*L**2*intrho - 252*intrhoz2)/(210*L))
                k += 1
                Mv[k] += r11*(L*betay*intrhoz*r12*(10*alphay - 9)/60 + L*betaz*intrhoy*r13*(9 - 10*alphaz)/60) + r12*(-L*betaz*intrhoz*r11*(2*alphaz + 1)/12 + betay*betaz*intrhoyz*r12*(-5*alphaz - 1)/10 + betaz**2*r13*(35*L**2*alphaz**2*intrho - 63*L**2*alphaz*intrho + 26*L**2*intrho - 420*alphaz*intrhoz2 - 84*intrhoz2)/840) + r13*(L*betay*intrhoy*r11*(2*alphay + 1)/12 + betay**2*r12*(-35*L**2*alphay**2*intrho + 63*L**2*alphay*intrho - 26*L**2*intrho + 420*alphay*intrhoy2 + 84*intrhoy2)/840 + betay*betaz*intrhoyz*r13*(5*alphay + 1)/10)
                k += 1
                Mv[k] += r21*(L*betay*intrhoz*r12*(10*alphay - 9)/60 + L*betaz*intrhoy*r13*(9 - 10*alphaz)/60) + r22*(-L*betaz*intrhoz*r11*(2*alphaz + 1)/12 + betay*betaz*intrhoyz*r12*(-5*alphaz - 1)/10 + betaz**2*r13*(35*L**2*alphaz**2*intrho - 63*L**2*alphaz*intrho + 26*L**2*intrho - 420*alphaz*intrhoz2 - 84*intrhoz2)/840) + r23*(L*betay*intrhoy*r11*(2*alphay + 1)/12 + betay**2*r12*(-35*L**2*alphay**2*intrho + 63*L**2*alphay*intrho - 26*L**2*intrho + 420*alphay*intrhoy2 + 84*intrhoy2)/840 + betay*betaz*intrhoyz*r13*(5*alphay + 1)/10)
                k += 1
                Mv[k] += r31*(L*betay*intrhoz*r12*(10*alphay - 9)/60 + L*betaz*intrhoy*r13*(9 - 10*alphaz)/60) + r32*(-L*betaz*intrhoz*r11*(2*alphaz + 1)/12 + betay*betaz*intrhoyz*r12*(-5*alphaz - 1)/10 + betaz**2*r13*(35*L**2*alphaz**2*intrho - 63*L**2*alphaz*intrho + 26*L**2*intrho - 420*alphaz*intrhoz2 - 84*intrhoz2)/840) + r33*(L*betay*intrhoy*r11*(2*alphay + 1)/12 + betay**2*r12*(-35*L**2*alphay**2*intrho + 63*L**2*alphay*intrho - 26*L**2*intrho + 420*alphay*intrhoy2 + 84*intrhoy2)/840 + betay*betaz*intrhoyz*r13*(5*alphay + 1)/10)
                k += 1
                Mv[k] += r11*(L*intrho*r21/3 + betay*intrhoy*r22/2 + betaz*intrhoz*r23/2) + r12*(betay*intrhoy*r21/2 + betay**2*r22*(70*L**2*alphay**2*intrho - 147*L**2*alphay*intrho + 78*L**2*intrho + 252*intrhoy2)/(210*L) + 6*betay*betaz*intrhoyz*r23/(5*L)) + r13*(betaz*intrhoz*r21/2 + 6*betay*betaz*intrhoyz*r22/(5*L) + betaz**2*r23*(70*L**2*alphaz**2*intrho - 147*L**2*alphaz*intrho + 78*L**2*intrho + 252*intrhoz2)/(210*L))
                k += 1
                Mv[k] += r21*(L*intrho*r21/3 + betay*intrhoy*r22/2 + betaz*intrhoz*r23/2) + r22*(betay*intrhoy*r21/2 + betay**2*r22*(70*L**2*alphay**2*intrho - 147*L**2*alphay*intrho + 78*L**2*intrho + 252*intrhoy2)/(210*L) + 6*betay*betaz*intrhoyz*r23/(5*L)) + r23*(betaz*intrhoz*r21/2 + 6*betay*betaz*intrhoyz*r22/(5*L) + betaz**2*r23*(70*L**2*alphaz**2*intrho - 147*L**2*alphaz*intrho + 78*L**2*intrho + 252*intrhoz2)/(210*L))
                k += 1
                Mv[k] += r31*(L*intrho*r21/3 + betay*intrhoy*r22/2 + betaz*intrhoz*r23/2) + r32*(betay*intrhoy*r21/2 + betay**2*r22*(70*L**2*alphay**2*intrho - 147*L**2*alphay*intrho + 78*L**2*intrho + 252*intrhoy2)/(210*L) + 6*betay*betaz*intrhoyz*r23/(5*L)) + r33*(betaz*intrhoz*r21/2 + 6*betay*betaz*intrhoyz*r22/(5*L) + betaz**2*r23*(70*L**2*alphaz**2*intrho - 147*L**2*alphaz*intrho + 78*L**2*intrho + 252*intrhoz2)/(210*L))
                k += 1
                Mv[k] += r11*(L*betay*intrhoz*r22*(20*alphay - 21)/60 + L*betaz*intrhoy*r23*(21 - 20*alphaz)/60) + r12*(L*betaz*intrhoz*r21*(1 - 4*alphaz)/12 + betay*betaz*intrhoyz*r22*(-5*alphaz - 1)/10 + betaz**2*r23*(-35*L**2*alphaz**2*intrho + 77*L**2*alphaz*intrho - 44*L**2*intrho - 420*alphaz*intrhoz2 - 84*intrhoz2)/840) + r13*(L*betay*intrhoy*r21*(4*alphay - 1)/12 + betay**2*r22*(35*L**2*alphay**2*intrho - 77*L**2*alphay*intrho + 44*L**2*intrho + 420*alphay*intrhoy2 + 84*intrhoy2)/840 + betay*betaz*intrhoyz*r23*(5*alphay + 1)/10)
                k += 1
                Mv[k] += r21*(L*betay*intrhoz*r22*(20*alphay - 21)/60 + L*betaz*intrhoy*r23*(21 - 20*alphaz)/60) + r22*(L*betaz*intrhoz*r21*(1 - 4*alphaz)/12 + betay*betaz*intrhoyz*r22*(-5*alphaz - 1)/10 + betaz**2*r23*(-35*L**2*alphaz**2*intrho + 77*L**2*alphaz*intrho - 44*L**2*intrho - 420*alphaz*intrhoz2 - 84*intrhoz2)/840) + r23*(L*betay*intrhoy*r21*(4*alphay - 1)/12 + betay**2*r22*(35*L**2*alphay**2*intrho - 77*L**2*alphay*intrho + 44*L**2*intrho + 420*alphay*intrhoy2 + 84*intrhoy2)/840 + betay*betaz*intrhoyz*r23*(5*alphay + 1)/10)
                k += 1
                Mv[k] += r31*(L*betay*intrhoz*r22*(20*alphay - 21)/60 + L*betaz*intrhoy*r23*(21 - 20*alphaz)/60) + r32*(L*betaz*intrhoz*r21*(1 - 4*alphaz)/12 + betay*betaz*intrhoyz*r22*(-5*alphaz - 1)/10 + betaz**2*r23*(-35*L**2*alphaz**2*intrho + 77*L**2*alphaz*intrho - 44*L**2*intrho - 420*alphaz*intrhoz2 - 84*intrhoz2)/840) + r33*(L*betay*intrhoy*r21*(4*alphay - 1)/12 + betay**2*r22*(35*L**2*alphay**2*intrho - 77*L**2*alphay*intrho + 44*L**2*intrho + 420*alphay*intrhoy2 + 84*intrhoy2)/840 + betay*betaz*intrhoyz*r23*(5*alphay + 1)/10)
                k += 1
                Mv[k] += r11*(L*intrho*r21/6 + betay*intrhoy*r22/2 + betaz*intrhoz*r23/2) + r12*(-betay*intrhoy*r21/2 + betay**2*r22*(35*L**2*alphay**2*intrho - 63*L**2*alphay*intrho + 27*L**2*intrho - 252*intrhoy2)/(210*L) - 6*betay*betaz*intrhoyz*r23/(5*L)) + r13*(-betaz*intrhoz*r21/2 - 6*betay*betaz*intrhoyz*r22/(5*L) + betaz**2*r23*(35*L**2*alphaz**2*intrho - 63*L**2*alphaz*intrho + 27*L**2*intrho - 252*intrhoz2)/(210*L))
                k += 1
                Mv[k] += r21*(L*intrho*r21/6 + betay*intrhoy*r22/2 + betaz*intrhoz*r23/2) + r22*(-betay*intrhoy*r21/2 + betay**2*r22*(35*L**2*alphay**2*intrho - 63*L**2*alphay*intrho + 27*L**2*intrho - 252*intrhoy2)/(210*L) - 6*betay*betaz*intrhoyz*r23/(5*L)) + r23*(-betaz*intrhoz*r21/2 - 6*betay*betaz*intrhoyz*r22/(5*L) + betaz**2*r23*(35*L**2*alphaz**2*intrho - 63*L**2*alphaz*intrho + 27*L**2*intrho - 252*intrhoz2)/(210*L))
                k += 1
                Mv[k] += r31*(L*intrho*r21/6 + betay*intrhoy*r22/2 + betaz*intrhoz*r23/2) + r32*(-betay*intrhoy*r21/2 + betay**2*r22*(35*L**2*alphay**2*intrho - 63*L**2*alphay*intrho + 27*L**2*intrho - 252*intrhoy2)/(210*L) - 6*betay*betaz*intrhoyz*r23/(5*L)) + r33*(-betaz*intrhoz*r21/2 - 6*betay*betaz*intrhoyz*r22/(5*L) + betaz**2*r23*(35*L**2*alphaz**2*intrho - 63*L**2*alphaz*intrho + 27*L**2*intrho - 252*intrhoz2)/(210*L))
                k += 1
                Mv[k] += r11*(L*betay*intrhoz*r22*(10*alphay - 9)/60 + L*betaz*intrhoy*r23*(9 - 10*alphaz)/60) + r12*(-L*betaz*intrhoz*r21*(2*alphaz + 1)/12 + betay*betaz*intrhoyz*r22*(-5*alphaz - 1)/10 + betaz**2*r23*(35*L**2*alphaz**2*intrho - 63*L**2*alphaz*intrho + 26*L**2*intrho - 420*alphaz*intrhoz2 - 84*intrhoz2)/840) + r13*(L*betay*intrhoy*r21*(2*alphay + 1)/12 + betay**2*r22*(-35*L**2*alphay**2*intrho + 63*L**2*alphay*intrho - 26*L**2*intrho + 420*alphay*intrhoy2 + 84*intrhoy2)/840 + betay*betaz*intrhoyz*r23*(5*alphay + 1)/10)
                k += 1
                Mv[k] += r21*(L*betay*intrhoz*r22*(10*alphay - 9)/60 + L*betaz*intrhoy*r23*(9 - 10*alphaz)/60) + r22*(-L*betaz*intrhoz*r21*(2*alphaz + 1)/12 + betay*betaz*intrhoyz*r22*(-5*alphaz - 1)/10 + betaz**2*r23*(35*L**2*alphaz**2*intrho - 63*L**2*alphaz*intrho + 26*L**2*intrho - 420*alphaz*intrhoz2 - 84*intrhoz2)/840) + r23*(L*betay*intrhoy*r21*(2*alphay + 1)/12 + betay**2*r22*(-35*L**2*alphay**2*intrho + 63*L**2*alphay*intrho - 26*L**2*intrho + 420*alphay*intrhoy2 + 84*intrhoy2)/840 + betay*betaz*intrhoyz*r23*(5*alphay + 1)/10)
                k += 1
                Mv[k] += r31*(L*betay*intrhoz*r22*(10*alphay - 9)/60 + L*betaz*intrhoy*r23*(9 - 10*alphaz)/60) + r32*(-L*betaz*intrhoz*r21*(2*alphaz + 1)/12 + betay*betaz*intrhoyz*r22*(-5*alphaz - 1)/10 + betaz**2*r23*(35*L**2*alphaz**2*intrho - 63*L**2*alphaz*intrho + 26*L**2*intrho - 420*alphaz*intrhoz2 - 84*intrhoz2)/840) + r33*(L*betay*intrhoy*r21*(2*alphay + 1)/12 + betay**2*r22*(-35*L**2*alphay**2*intrho + 63*L**2*alphay*intrho - 26*L**2*intrho + 420*alphay*intrhoy2 + 84*intrhoy2)/840 + betay*betaz*intrhoyz*r23*(5*alphay + 1)/10)
                k += 1
                Mv[k] += r11*(L*intrho*r31/3 + betay*intrhoy*r32/2 + betaz*intrhoz*r33/2) + r12*(betay*intrhoy*r31/2 + betay**2*r32*(70*L**2*alphay**2*intrho - 147*L**2*alphay*intrho + 78*L**2*intrho + 252*intrhoy2)/(210*L) + 6*betay*betaz*intrhoyz*r33/(5*L)) + r13*(betaz*intrhoz*r31/2 + 6*betay*betaz*intrhoyz*r32/(5*L) + betaz**2*r33*(70*L**2*alphaz**2*intrho - 147*L**2*alphaz*intrho + 78*L**2*intrho + 252*intrhoz2)/(210*L))
                k += 1
                Mv[k] += r21*(L*intrho*r31/3 + betay*intrhoy*r32/2 + betaz*intrhoz*r33/2) + r22*(betay*intrhoy*r31/2 + betay**2*r32*(70*L**2*alphay**2*intrho - 147*L**2*alphay*intrho + 78*L**2*intrho + 252*intrhoy2)/(210*L) + 6*betay*betaz*intrhoyz*r33/(5*L)) + r23*(betaz*intrhoz*r31/2 + 6*betay*betaz*intrhoyz*r32/(5*L) + betaz**2*r33*(70*L**2*alphaz**2*intrho - 147*L**2*alphaz*intrho + 78*L**2*intrho + 252*intrhoz2)/(210*L))
                k += 1
                Mv[k] += r31*(L*intrho*r31/3 + betay*intrhoy*r32/2 + betaz*intrhoz*r33/2) + r32*(betay*intrhoy*r31/2 + betay**2*r32*(70*L**2*alphay**2*intrho - 147*L**2*alphay*intrho + 78*L**2*intrho + 252*intrhoy2)/(210*L) + 6*betay*betaz*intrhoyz*r33/(5*L)) + r33*(betaz*intrhoz*r31/2 + 6*betay*betaz*intrhoyz*r32/(5*L) + betaz**2*r33*(70*L**2*alphaz**2*intrho - 147*L**2*alphaz*intrho + 78*L**2*intrho + 252*intrhoz2)/(210*L))
                k += 1
                Mv[k] += r11*(L*betay*intrhoz*r32*(20*alphay - 21)/60 + L*betaz*intrhoy*r33*(21 - 20*alphaz)/60) + r12*(L*betaz*intrhoz*r31*(1 - 4*alphaz)/12 + betay*betaz*intrhoyz*r32*(-5*alphaz - 1)/10 + betaz**2*r33*(-35*L**2*alphaz**2*intrho + 77*L**2*alphaz*intrho - 44*L**2*intrho - 420*alphaz*intrhoz2 - 84*intrhoz2)/840) + r13*(L*betay*intrhoy*r31*(4*alphay - 1)/12 + betay**2*r32*(35*L**2*alphay**2*intrho - 77*L**2*alphay*intrho + 44*L**2*intrho + 420*alphay*intrhoy2 + 84*intrhoy2)/840 + betay*betaz*intrhoyz*r33*(5*alphay + 1)/10)
                k += 1
                Mv[k] += r21*(L*betay*intrhoz*r32*(20*alphay - 21)/60 + L*betaz*intrhoy*r33*(21 - 20*alphaz)/60) + r22*(L*betaz*intrhoz*r31*(1 - 4*alphaz)/12 + betay*betaz*intrhoyz*r32*(-5*alphaz - 1)/10 + betaz**2*r33*(-35*L**2*alphaz**2*intrho + 77*L**2*alphaz*intrho - 44*L**2*intrho - 420*alphaz*intrhoz2 - 84*intrhoz2)/840) + r23*(L*betay*intrhoy*r31*(4*alphay - 1)/12 + betay**2*r32*(35*L**2*alphay**2*intrho - 77*L**2*alphay*intrho + 44*L**2*intrho + 420*alphay*intrhoy2 + 84*intrhoy2)/840 + betay*betaz*intrhoyz*r33*(5*alphay + 1)/10)
                k += 1
                Mv[k] += r31*(L*betay*intrhoz*r32*(20*alphay - 21)/60 + L*betaz*intrhoy*r33*(21 - 20*alphaz)/60) + r32*(L*betaz*intrhoz*r31*(1 - 4*alphaz)/12 + betay*betaz*intrhoyz*r32*(-5*alphaz - 1)/10 + betaz**2*r33*(-35*L**2*alphaz**2*intrho + 77*L**2*alphaz*intrho - 44*L**2*intrho - 420*alphaz*intrhoz2 - 84*intrhoz2)/840) + r33*(L*betay*intrhoy*r31*(4*alphay - 1)/12 + betay**2*r32*(35*L**2*alphay**2*intrho - 77*L**2*alphay*intrho + 44*L**2*intrho + 420*alphay*intrhoy2 + 84*intrhoy2)/840 + betay*betaz*intrhoyz*r33*(5*alphay + 1)/10)
                k += 1
                Mv[k] += r11*(L*intrho*r31/6 + betay*intrhoy*r32/2 + betaz*intrhoz*r33/2) + r12*(-betay*intrhoy*r31/2 + betay**2*r32*(35*L**2*alphay**2*intrho - 63*L**2*alphay*intrho + 27*L**2*intrho - 252*intrhoy2)/(210*L) - 6*betay*betaz*intrhoyz*r33/(5*L)) + r13*(-betaz*intrhoz*r31/2 - 6*betay*betaz*intrhoyz*r32/(5*L) + betaz**2*r33*(35*L**2*alphaz**2*intrho - 63*L**2*alphaz*intrho + 27*L**2*intrho - 252*intrhoz2)/(210*L))
                k += 1
                Mv[k] += r21*(L*intrho*r31/6 + betay*intrhoy*r32/2 + betaz*intrhoz*r33/2) + r22*(-betay*intrhoy*r31/2 + betay**2*r32*(35*L**2*alphay**2*intrho - 63*L**2*alphay*intrho + 27*L**2*intrho - 252*intrhoy2)/(210*L) - 6*betay*betaz*intrhoyz*r33/(5*L)) + r23*(-betaz*intrhoz*r31/2 - 6*betay*betaz*intrhoyz*r32/(5*L) + betaz**2*r33*(35*L**2*alphaz**2*intrho - 63*L**2*alphaz*intrho + 27*L**2*intrho - 252*intrhoz2)/(210*L))
                k += 1
                Mv[k] += r31*(L*intrho*r31/6 + betay*intrhoy*r32/2 + betaz*intrhoz*r33/2) + r32*(-betay*intrhoy*r31/2 + betay**2*r32*(35*L**2*alphay**2*intrho - 63*L**2*alphay*intrho + 27*L**2*intrho - 252*intrhoy2)/(210*L) - 6*betay*betaz*intrhoyz*r33/(5*L)) + r33*(-betaz*intrhoz*r31/2 - 6*betay*betaz*intrhoyz*r32/(5*L) + betaz**2*r33*(35*L**2*alphaz**2*intrho - 63*L**2*alphaz*intrho + 27*L**2*intrho - 252*intrhoz2)/(210*L))
                k += 1
                Mv[k] += r11*(L*betay*intrhoz*r32*(10*alphay - 9)/60 + L*betaz*intrhoy*r33*(9 - 10*alphaz)/60) + r12*(-L*betaz*intrhoz*r31*(2*alphaz + 1)/12 + betay*betaz*intrhoyz*r32*(-5*alphaz - 1)/10 + betaz**2*r33*(35*L**2*alphaz**2*intrho - 63*L**2*alphaz*intrho + 26*L**2*intrho - 420*alphaz*intrhoz2 - 84*intrhoz2)/840) + r13*(L*betay*intrhoy*r31*(2*alphay + 1)/12 + betay**2*r32*(-35*L**2*alphay**2*intrho + 63*L**2*alphay*intrho - 26*L**2*intrho + 420*alphay*intrhoy2 + 84*intrhoy2)/840 + betay*betaz*intrhoyz*r33*(5*alphay + 1)/10)
                k += 1
                Mv[k] += r21*(L*betay*intrhoz*r32*(10*alphay - 9)/60 + L*betaz*intrhoy*r33*(9 - 10*alphaz)/60) + r22*(-L*betaz*intrhoz*r31*(2*alphaz + 1)/12 + betay*betaz*intrhoyz*r32*(-5*alphaz - 1)/10 + betaz**2*r33*(35*L**2*alphaz**2*intrho - 63*L**2*alphaz*intrho + 26*L**2*intrho - 420*alphaz*intrhoz2 - 84*intrhoz2)/840) + r23*(L*betay*intrhoy*r31*(2*alphay + 1)/12 + betay**2*r32*(-35*L**2*alphay**2*intrho + 63*L**2*alphay*intrho - 26*L**2*intrho + 420*alphay*intrhoy2 + 84*intrhoy2)/840 + betay*betaz*intrhoyz*r33*(5*alphay + 1)/10)
                k += 1
                Mv[k] += r31*(L*betay*intrhoz*r32*(10*alphay - 9)/60 + L*betaz*intrhoy*r33*(9 - 10*alphaz)/60) + r32*(-L*betaz*intrhoz*r31*(2*alphaz + 1)/12 + betay*betaz*intrhoyz*r32*(-5*alphaz - 1)/10 + betaz**2*r33*(35*L**2*alphaz**2*intrho - 63*L**2*alphaz*intrho + 26*L**2*intrho - 420*alphaz*intrhoz2 - 84*intrhoz2)/840) + r33*(L*betay*intrhoy*r31*(2*alphay + 1)/12 + betay**2*r32*(-35*L**2*alphay**2*intrho + 63*L**2*alphay*intrho - 26*L**2*intrho + 420*alphay*intrhoy2 + 84*intrhoy2)/840 + betay*betaz*intrhoyz*r33*(5*alphay + 1)/10)
                k += 1
                Mv[k] += r11*(L*betay*intrhoy*r13*(4*alphay - 1)/12 + L*betaz*intrhoz*r12*(1 - 4*alphaz)/12) + r12*(L*betay*intrhoz*r11*(20*alphay - 21)/60 + betay**2*r13*(35*L**2*alphay**2*intrho - 77*L**2*alphay*intrho + 44*L**2*intrho + 420*alphay*intrhoy2 + 84*intrhoy2)/840 + betay*betaz*intrhoyz*r12*(-5*alphaz - 1)/10) + r13*(L*betaz*intrhoy*r11*(21 - 20*alphaz)/60 + betay*betaz*intrhoyz*r13*(5*alphay + 1)/10 + betaz**2*r12*(-35*L**2*alphaz**2*intrho + 77*L**2*alphaz*intrho - 44*L**2*intrho - 420*alphaz*intrhoz2 - 84*intrhoz2)/840)
                k += 1
                Mv[k] += r21*(L*betay*intrhoy*r13*(4*alphay - 1)/12 + L*betaz*intrhoz*r12*(1 - 4*alphaz)/12) + r22*(L*betay*intrhoz*r11*(20*alphay - 21)/60 + betay**2*r13*(35*L**2*alphay**2*intrho - 77*L**2*alphay*intrho + 44*L**2*intrho + 420*alphay*intrhoy2 + 84*intrhoy2)/840 + betay*betaz*intrhoyz*r12*(-5*alphaz - 1)/10) + r23*(L*betaz*intrhoy*r11*(21 - 20*alphaz)/60 + betay*betaz*intrhoyz*r13*(5*alphay + 1)/10 + betaz**2*r12*(-35*L**2*alphaz**2*intrho + 77*L**2*alphaz*intrho - 44*L**2*intrho - 420*alphaz*intrhoz2 - 84*intrhoz2)/840)
                k += 1
                Mv[k] += r31*(L*betay*intrhoy*r13*(4*alphay - 1)/12 + L*betaz*intrhoz*r12*(1 - 4*alphaz)/12) + r32*(L*betay*intrhoz*r11*(20*alphay - 21)/60 + betay**2*r13*(35*L**2*alphay**2*intrho - 77*L**2*alphay*intrho + 44*L**2*intrho + 420*alphay*intrhoy2 + 84*intrhoy2)/840 + betay*betaz*intrhoyz*r12*(-5*alphaz - 1)/10) + r33*(L*betaz*intrhoy*r11*(21 - 20*alphaz)/60 + betay*betaz*intrhoyz*r13*(5*alphay + 1)/10 + betaz**2*r12*(-35*L**2*alphaz**2*intrho + 77*L**2*alphaz*intrho - 44*L**2*intrho - 420*alphaz*intrhoz2 - 84*intrhoz2)/840)
                k += 1
                Mv[k] += r11*(L**2*betay*intrhoz*r13*(5*alphay - 6)/120 + L**2*betaz*intrhoy*r12*(5*alphaz - 6)/120 + L*r11*(intrhoy2 + intrhoz2)/3) + r12*(L**2*betaz*intrhoy*r11*(5*alphaz - 6)/120 + L*betay*betaz*intrhoyz*r13*(-20*alphay*alphaz + 5*alphay + 5*alphaz - 8)/60 + L*betaz**2*r12*(7*L**2*alphaz**2*intrho - 14*L**2*alphaz*intrho + 8*L**2*intrho + 280*alphaz**2*intrhoz2 - 140*alphaz*intrhoz2 + 112*intrhoz2)/840) + r13*(L**2*betay*intrhoz*r11*(5*alphay - 6)/120 + L*betay**2*r13*(7*L**2*alphay**2*intrho - 14*L**2*alphay*intrho + 8*L**2*intrho + 280*alphay**2*intrhoy2 - 140*alphay*intrhoy2 + 112*intrhoy2)/840 + L*betay*betaz*intrhoyz*r12*(-20*alphay*alphaz + 5*alphay + 5*alphaz - 8)/60)
                k += 1
                Mv[k] += r21*(L**2*betay*intrhoz*r13*(5*alphay - 6)/120 + L**2*betaz*intrhoy*r12*(5*alphaz - 6)/120 + L*r11*(intrhoy2 + intrhoz2)/3) + r22*(L**2*betaz*intrhoy*r11*(5*alphaz - 6)/120 + L*betay*betaz*intrhoyz*r13*(-20*alphay*alphaz + 5*alphay + 5*alphaz - 8)/60 + L*betaz**2*r12*(7*L**2*alphaz**2*intrho - 14*L**2*alphaz*intrho + 8*L**2*intrho + 280*alphaz**2*intrhoz2 - 140*alphaz*intrhoz2 + 112*intrhoz2)/840) + r23*(L**2*betay*intrhoz*r11*(5*alphay - 6)/120 + L*betay**2*r13*(7*L**2*alphay**2*intrho - 14*L**2*alphay*intrho + 8*L**2*intrho + 280*alphay**2*intrhoy2 - 140*alphay*intrhoy2 + 112*intrhoy2)/840 + L*betay*betaz*intrhoyz*r12*(-20*alphay*alphaz + 5*alphay + 5*alphaz - 8)/60)
                k += 1
                Mv[k] += r31*(L**2*betay*intrhoz*r13*(5*alphay - 6)/120 + L**2*betaz*intrhoy*r12*(5*alphaz - 6)/120 + L*r11*(intrhoy2 + intrhoz2)/3) + r32*(L**2*betaz*intrhoy*r11*(5*alphaz - 6)/120 + L*betay*betaz*intrhoyz*r13*(-20*alphay*alphaz + 5*alphay + 5*alphaz - 8)/60 + L*betaz**2*r12*(7*L**2*alphaz**2*intrho - 14*L**2*alphaz*intrho + 8*L**2*intrho + 280*alphaz**2*intrhoz2 - 140*alphaz*intrhoz2 + 112*intrhoz2)/840) + r33*(L**2*betay*intrhoz*r11*(5*alphay - 6)/120 + L*betay**2*r13*(7*L**2*alphay**2*intrho - 14*L**2*alphay*intrho + 8*L**2*intrho + 280*alphay**2*intrhoy2 - 140*alphay*intrhoy2 + 112*intrhoy2)/840 + L*betay*betaz*intrhoyz*r12*(-20*alphay*alphaz + 5*alphay + 5*alphaz - 8)/60)
                k += 1
                Mv[k] += r11*(L*betay*intrhoy*r13*(2*alphay + 1)/12 - L*betaz*intrhoz*r12*(2*alphaz + 1)/12) + r12*(L*betay*intrhoz*r11*(10*alphay - 9)/60 + betay**2*r13*(35*L**2*alphay**2*intrho - 63*L**2*alphay*intrho + 26*L**2*intrho - 420*alphay*intrhoy2 - 84*intrhoy2)/840 + betay*betaz*intrhoyz*r12*(5*alphaz + 1)/10) + r13*(L*betaz*intrhoy*r11*(9 - 10*alphaz)/60 + betay*betaz*intrhoyz*r13*(-5*alphay - 1)/10 + betaz**2*r12*(-35*L**2*alphaz**2*intrho + 63*L**2*alphaz*intrho - 26*L**2*intrho + 420*alphaz*intrhoz2 + 84*intrhoz2)/840)
                k += 1
                Mv[k] += r21*(L*betay*intrhoy*r13*(2*alphay + 1)/12 - L*betaz*intrhoz*r12*(2*alphaz + 1)/12) + r22*(L*betay*intrhoz*r11*(10*alphay - 9)/60 + betay**2*r13*(35*L**2*alphay**2*intrho - 63*L**2*alphay*intrho + 26*L**2*intrho - 420*alphay*intrhoy2 - 84*intrhoy2)/840 + betay*betaz*intrhoyz*r12*(5*alphaz + 1)/10) + r23*(L*betaz*intrhoy*r11*(9 - 10*alphaz)/60 + betay*betaz*intrhoyz*r13*(-5*alphay - 1)/10 + betaz**2*r12*(-35*L**2*alphaz**2*intrho + 63*L**2*alphaz*intrho - 26*L**2*intrho + 420*alphaz*intrhoz2 + 84*intrhoz2)/840)
                k += 1
                Mv[k] += r31*(L*betay*intrhoy*r13*(2*alphay + 1)/12 - L*betaz*intrhoz*r12*(2*alphaz + 1)/12) + r32*(L*betay*intrhoz*r11*(10*alphay - 9)/60 + betay**2*r13*(35*L**2*alphay**2*intrho - 63*L**2*alphay*intrho + 26*L**2*intrho - 420*alphay*intrhoy2 - 84*intrhoy2)/840 + betay*betaz*intrhoyz*r12*(5*alphaz + 1)/10) + r33*(L*betaz*intrhoy*r11*(9 - 10*alphaz)/60 + betay*betaz*intrhoyz*r13*(-5*alphay - 1)/10 + betaz**2*r12*(-35*L**2*alphaz**2*intrho + 63*L**2*alphaz*intrho - 26*L**2*intrho + 420*alphaz*intrhoz2 + 84*intrhoz2)/840)
                k += 1
                Mv[k] += r11*(L**2*betay*intrhoz*r13*(5*alphay - 4)/120 + L**2*betaz*intrhoy*r12*(5*alphaz - 4)/120 + L*r11*(intrhoy2 + intrhoz2)/6) + r12*(L**2*betaz*intrhoy*r11*(4 - 5*alphaz)/120 + L*betay*betaz*intrhoyz*r13*(-10*alphay*alphaz - 5*alphay - 5*alphaz + 2)/60 + L*betaz**2*r12*(-7*L**2*alphaz**2*intrho + 14*L**2*alphaz*intrho - 6*L**2*intrho + 140*alphaz**2*intrhoz2 + 140*alphaz*intrhoz2 - 28*intrhoz2)/840) + r13*(L**2*betay*intrhoz*r11*(4 - 5*alphay)/120 + L*betay**2*r13*(-7*L**2*alphay**2*intrho + 14*L**2*alphay*intrho - 6*L**2*intrho + 140*alphay**2*intrhoy2 + 140*alphay*intrhoy2 - 28*intrhoy2)/840 + L*betay*betaz*intrhoyz*r12*(-10*alphay*alphaz - 5*alphay - 5*alphaz + 2)/60)
                k += 1
                Mv[k] += r21*(L**2*betay*intrhoz*r13*(5*alphay - 4)/120 + L**2*betaz*intrhoy*r12*(5*alphaz - 4)/120 + L*r11*(intrhoy2 + intrhoz2)/6) + r22*(L**2*betaz*intrhoy*r11*(4 - 5*alphaz)/120 + L*betay*betaz*intrhoyz*r13*(-10*alphay*alphaz - 5*alphay - 5*alphaz + 2)/60 + L*betaz**2*r12*(-7*L**2*alphaz**2*intrho + 14*L**2*alphaz*intrho - 6*L**2*intrho + 140*alphaz**2*intrhoz2 + 140*alphaz*intrhoz2 - 28*intrhoz2)/840) + r23*(L**2*betay*intrhoz*r11*(4 - 5*alphay)/120 + L*betay**2*r13*(-7*L**2*alphay**2*intrho + 14*L**2*alphay*intrho - 6*L**2*intrho + 140*alphay**2*intrhoy2 + 140*alphay*intrhoy2 - 28*intrhoy2)/840 + L*betay*betaz*intrhoyz*r12*(-10*alphay*alphaz - 5*alphay - 5*alphaz + 2)/60)
                k += 1
                Mv[k] += r31*(L**2*betay*intrhoz*r13*(5*alphay - 4)/120 + L**2*betaz*intrhoy*r12*(5*alphaz - 4)/120 + L*r11*(intrhoy2 + intrhoz2)/6) + r32*(L**2*betaz*intrhoy*r11*(4 - 5*alphaz)/120 + L*betay*betaz*intrhoyz*r13*(-10*alphay*alphaz - 5*alphay - 5*alphaz + 2)/60 + L*betaz**2*r12*(-7*L**2*alphaz**2*intrho + 14*L**2*alphaz*intrho - 6*L**2*intrho + 140*alphaz**2*intrhoz2 + 140*alphaz*intrhoz2 - 28*intrhoz2)/840) + r33*(L**2*betay*intrhoz*r11*(4 - 5*alphay)/120 + L*betay**2*r13*(-7*L**2*alphay**2*intrho + 14*L**2*alphay*intrho - 6*L**2*intrho + 140*alphay**2*intrhoy2 + 140*alphay*intrhoy2 - 28*intrhoy2)/840 + L*betay*betaz*intrhoyz*r12*(-10*alphay*alphaz - 5*alphay - 5*alphaz + 2)/60)
                k += 1
                Mv[k] += r11*(L*betay*intrhoy*r23*(4*alphay - 1)/12 + L*betaz*intrhoz*r22*(1 - 4*alphaz)/12) + r12*(L*betay*intrhoz*r21*(20*alphay - 21)/60 + betay**2*r23*(35*L**2*alphay**2*intrho - 77*L**2*alphay*intrho + 44*L**2*intrho + 420*alphay*intrhoy2 + 84*intrhoy2)/840 + betay*betaz*intrhoyz*r22*(-5*alphaz - 1)/10) + r13*(L*betaz*intrhoy*r21*(21 - 20*alphaz)/60 + betay*betaz*intrhoyz*r23*(5*alphay + 1)/10 + betaz**2*r22*(-35*L**2*alphaz**2*intrho + 77*L**2*alphaz*intrho - 44*L**2*intrho - 420*alphaz*intrhoz2 - 84*intrhoz2)/840)
                k += 1
                Mv[k] += r21*(L*betay*intrhoy*r23*(4*alphay - 1)/12 + L*betaz*intrhoz*r22*(1 - 4*alphaz)/12) + r22*(L*betay*intrhoz*r21*(20*alphay - 21)/60 + betay**2*r23*(35*L**2*alphay**2*intrho - 77*L**2*alphay*intrho + 44*L**2*intrho + 420*alphay*intrhoy2 + 84*intrhoy2)/840 + betay*betaz*intrhoyz*r22*(-5*alphaz - 1)/10) + r23*(L*betaz*intrhoy*r21*(21 - 20*alphaz)/60 + betay*betaz*intrhoyz*r23*(5*alphay + 1)/10 + betaz**2*r22*(-35*L**2*alphaz**2*intrho + 77*L**2*alphaz*intrho - 44*L**2*intrho - 420*alphaz*intrhoz2 - 84*intrhoz2)/840)
                k += 1
                Mv[k] += r31*(L*betay*intrhoy*r23*(4*alphay - 1)/12 + L*betaz*intrhoz*r22*(1 - 4*alphaz)/12) + r32*(L*betay*intrhoz*r21*(20*alphay - 21)/60 + betay**2*r23*(35*L**2*alphay**2*intrho - 77*L**2*alphay*intrho + 44*L**2*intrho + 420*alphay*intrhoy2 + 84*intrhoy2)/840 + betay*betaz*intrhoyz*r22*(-5*alphaz - 1)/10) + r33*(L*betaz*intrhoy*r21*(21 - 20*alphaz)/60 + betay*betaz*intrhoyz*r23*(5*alphay + 1)/10 + betaz**2*r22*(-35*L**2*alphaz**2*intrho + 77*L**2*alphaz*intrho - 44*L**2*intrho - 420*alphaz*intrhoz2 - 84*intrhoz2)/840)
                k += 1
                Mv[k] += r11*(L**2*betay*intrhoz*r23*(5*alphay - 6)/120 + L**2*betaz*intrhoy*r22*(5*alphaz - 6)/120 + L*r21*(intrhoy2 + intrhoz2)/3) + r12*(L**2*betaz*intrhoy*r21*(5*alphaz - 6)/120 + L*betay*betaz*intrhoyz*r23*(-20*alphay*alphaz + 5*alphay + 5*alphaz - 8)/60 + L*betaz**2*r22*(7*L**2*alphaz**2*intrho - 14*L**2*alphaz*intrho + 8*L**2*intrho + 280*alphaz**2*intrhoz2 - 140*alphaz*intrhoz2 + 112*intrhoz2)/840) + r13*(L**2*betay*intrhoz*r21*(5*alphay - 6)/120 + L*betay**2*r23*(7*L**2*alphay**2*intrho - 14*L**2*alphay*intrho + 8*L**2*intrho + 280*alphay**2*intrhoy2 - 140*alphay*intrhoy2 + 112*intrhoy2)/840 + L*betay*betaz*intrhoyz*r22*(-20*alphay*alphaz + 5*alphay + 5*alphaz - 8)/60)
                k += 1
                Mv[k] += r21*(L**2*betay*intrhoz*r23*(5*alphay - 6)/120 + L**2*betaz*intrhoy*r22*(5*alphaz - 6)/120 + L*r21*(intrhoy2 + intrhoz2)/3) + r22*(L**2*betaz*intrhoy*r21*(5*alphaz - 6)/120 + L*betay*betaz*intrhoyz*r23*(-20*alphay*alphaz + 5*alphay + 5*alphaz - 8)/60 + L*betaz**2*r22*(7*L**2*alphaz**2*intrho - 14*L**2*alphaz*intrho + 8*L**2*intrho + 280*alphaz**2*intrhoz2 - 140*alphaz*intrhoz2 + 112*intrhoz2)/840) + r23*(L**2*betay*intrhoz*r21*(5*alphay - 6)/120 + L*betay**2*r23*(7*L**2*alphay**2*intrho - 14*L**2*alphay*intrho + 8*L**2*intrho + 280*alphay**2*intrhoy2 - 140*alphay*intrhoy2 + 112*intrhoy2)/840 + L*betay*betaz*intrhoyz*r22*(-20*alphay*alphaz + 5*alphay + 5*alphaz - 8)/60)
                k += 1
                Mv[k] += r31*(L**2*betay*intrhoz*r23*(5*alphay - 6)/120 + L**2*betaz*intrhoy*r22*(5*alphaz - 6)/120 + L*r21*(intrhoy2 + intrhoz2)/3) + r32*(L**2*betaz*intrhoy*r21*(5*alphaz - 6)/120 + L*betay*betaz*intrhoyz*r23*(-20*alphay*alphaz + 5*alphay + 5*alphaz - 8)/60 + L*betaz**2*r22*(7*L**2*alphaz**2*intrho - 14*L**2*alphaz*intrho + 8*L**2*intrho + 280*alphaz**2*intrhoz2 - 140*alphaz*intrhoz2 + 112*intrhoz2)/840) + r33*(L**2*betay*intrhoz*r21*(5*alphay - 6)/120 + L*betay**2*r23*(7*L**2*alphay**2*intrho - 14*L**2*alphay*intrho + 8*L**2*intrho + 280*alphay**2*intrhoy2 - 140*alphay*intrhoy2 + 112*intrhoy2)/840 + L*betay*betaz*intrhoyz*r22*(-20*alphay*alphaz + 5*alphay + 5*alphaz - 8)/60)
                k += 1
                Mv[k] += r11*(L*betay*intrhoy*r23*(2*alphay + 1)/12 - L*betaz*intrhoz*r22*(2*alphaz + 1)/12) + r12*(L*betay*intrhoz*r21*(10*alphay - 9)/60 + betay**2*r23*(35*L**2*alphay**2*intrho - 63*L**2*alphay*intrho + 26*L**2*intrho - 420*alphay*intrhoy2 - 84*intrhoy2)/840 + betay*betaz*intrhoyz*r22*(5*alphaz + 1)/10) + r13*(L*betaz*intrhoy*r21*(9 - 10*alphaz)/60 + betay*betaz*intrhoyz*r23*(-5*alphay - 1)/10 + betaz**2*r22*(-35*L**2*alphaz**2*intrho + 63*L**2*alphaz*intrho - 26*L**2*intrho + 420*alphaz*intrhoz2 + 84*intrhoz2)/840)
                k += 1
                Mv[k] += r21*(L*betay*intrhoy*r23*(2*alphay + 1)/12 - L*betaz*intrhoz*r22*(2*alphaz + 1)/12) + r22*(L*betay*intrhoz*r21*(10*alphay - 9)/60 + betay**2*r23*(35*L**2*alphay**2*intrho - 63*L**2*alphay*intrho + 26*L**2*intrho - 420*alphay*intrhoy2 - 84*intrhoy2)/840 + betay*betaz*intrhoyz*r22*(5*alphaz + 1)/10) + r23*(L*betaz*intrhoy*r21*(9 - 10*alphaz)/60 + betay*betaz*intrhoyz*r23*(-5*alphay - 1)/10 + betaz**2*r22*(-35*L**2*alphaz**2*intrho + 63*L**2*alphaz*intrho - 26*L**2*intrho + 420*alphaz*intrhoz2 + 84*intrhoz2)/840)
                k += 1
                Mv[k] += r31*(L*betay*intrhoy*r23*(2*alphay + 1)/12 - L*betaz*intrhoz*r22*(2*alphaz + 1)/12) + r32*(L*betay*intrhoz*r21*(10*alphay - 9)/60 + betay**2*r23*(35*L**2*alphay**2*intrho - 63*L**2*alphay*intrho + 26*L**2*intrho - 420*alphay*intrhoy2 - 84*intrhoy2)/840 + betay*betaz*intrhoyz*r22*(5*alphaz + 1)/10) + r33*(L*betaz*intrhoy*r21*(9 - 10*alphaz)/60 + betay*betaz*intrhoyz*r23*(-5*alphay - 1)/10 + betaz**2*r22*(-35*L**2*alphaz**2*intrho + 63*L**2*alphaz*intrho - 26*L**2*intrho + 420*alphaz*intrhoz2 + 84*intrhoz2)/840)
                k += 1
                Mv[k] += r11*(L**2*betay*intrhoz*r23*(5*alphay - 4)/120 + L**2*betaz*intrhoy*r22*(5*alphaz - 4)/120 + L*r21*(intrhoy2 + intrhoz2)/6) + r12*(L**2*betaz*intrhoy*r21*(4 - 5*alphaz)/120 + L*betay*betaz*intrhoyz*r23*(-10*alphay*alphaz - 5*alphay - 5*alphaz + 2)/60 + L*betaz**2*r22*(-7*L**2*alphaz**2*intrho + 14*L**2*alphaz*intrho - 6*L**2*intrho + 140*alphaz**2*intrhoz2 + 140*alphaz*intrhoz2 - 28*intrhoz2)/840) + r13*(L**2*betay*intrhoz*r21*(4 - 5*alphay)/120 + L*betay**2*r23*(-7*L**2*alphay**2*intrho + 14*L**2*alphay*intrho - 6*L**2*intrho + 140*alphay**2*intrhoy2 + 140*alphay*intrhoy2 - 28*intrhoy2)/840 + L*betay*betaz*intrhoyz*r22*(-10*alphay*alphaz - 5*alphay - 5*alphaz + 2)/60)
                k += 1
                Mv[k] += r21*(L**2*betay*intrhoz*r23*(5*alphay - 4)/120 + L**2*betaz*intrhoy*r22*(5*alphaz - 4)/120 + L*r21*(intrhoy2 + intrhoz2)/6) + r22*(L**2*betaz*intrhoy*r21*(4 - 5*alphaz)/120 + L*betay*betaz*intrhoyz*r23*(-10*alphay*alphaz - 5*alphay - 5*alphaz + 2)/60 + L*betaz**2*r22*(-7*L**2*alphaz**2*intrho + 14*L**2*alphaz*intrho - 6*L**2*intrho + 140*alphaz**2*intrhoz2 + 140*alphaz*intrhoz2 - 28*intrhoz2)/840) + r23*(L**2*betay*intrhoz*r21*(4 - 5*alphay)/120 + L*betay**2*r23*(-7*L**2*alphay**2*intrho + 14*L**2*alphay*intrho - 6*L**2*intrho + 140*alphay**2*intrhoy2 + 140*alphay*intrhoy2 - 28*intrhoy2)/840 + L*betay*betaz*intrhoyz*r22*(-10*alphay*alphaz - 5*alphay - 5*alphaz + 2)/60)
                k += 1
                Mv[k] += r31*(L**2*betay*intrhoz*r23*(5*alphay - 4)/120 + L**2*betaz*intrhoy*r22*(5*alphaz - 4)/120 + L*r21*(intrhoy2 + intrhoz2)/6) + r32*(L**2*betaz*intrhoy*r21*(4 - 5*alphaz)/120 + L*betay*betaz*intrhoyz*r23*(-10*alphay*alphaz - 5*alphay - 5*alphaz + 2)/60 + L*betaz**2*r22*(-7*L**2*alphaz**2*intrho + 14*L**2*alphaz*intrho - 6*L**2*intrho + 140*alphaz**2*intrhoz2 + 140*alphaz*intrhoz2 - 28*intrhoz2)/840) + r33*(L**2*betay*intrhoz*r21*(4 - 5*alphay)/120 + L*betay**2*r23*(-7*L**2*alphay**2*intrho + 14*L**2*alphay*intrho - 6*L**2*intrho + 140*alphay**2*intrhoy2 + 140*alphay*intrhoy2 - 28*intrhoy2)/840 + L*betay*betaz*intrhoyz*r22*(-10*alphay*alphaz - 5*alphay - 5*alphaz + 2)/60)
                k += 1
                Mv[k] += r11*(L*betay*intrhoy*r33*(4*alphay - 1)/12 + L*betaz*intrhoz*r32*(1 - 4*alphaz)/12) + r12*(L*betay*intrhoz*r31*(20*alphay - 21)/60 + betay**2*r33*(35*L**2*alphay**2*intrho - 77*L**2*alphay*intrho + 44*L**2*intrho + 420*alphay*intrhoy2 + 84*intrhoy2)/840 + betay*betaz*intrhoyz*r32*(-5*alphaz - 1)/10) + r13*(L*betaz*intrhoy*r31*(21 - 20*alphaz)/60 + betay*betaz*intrhoyz*r33*(5*alphay + 1)/10 + betaz**2*r32*(-35*L**2*alphaz**2*intrho + 77*L**2*alphaz*intrho - 44*L**2*intrho - 420*alphaz*intrhoz2 - 84*intrhoz2)/840)
                k += 1
                Mv[k] += r21*(L*betay*intrhoy*r33*(4*alphay - 1)/12 + L*betaz*intrhoz*r32*(1 - 4*alphaz)/12) + r22*(L*betay*intrhoz*r31*(20*alphay - 21)/60 + betay**2*r33*(35*L**2*alphay**2*intrho - 77*L**2*alphay*intrho + 44*L**2*intrho + 420*alphay*intrhoy2 + 84*intrhoy2)/840 + betay*betaz*intrhoyz*r32*(-5*alphaz - 1)/10) + r23*(L*betaz*intrhoy*r31*(21 - 20*alphaz)/60 + betay*betaz*intrhoyz*r33*(5*alphay + 1)/10 + betaz**2*r32*(-35*L**2*alphaz**2*intrho + 77*L**2*alphaz*intrho - 44*L**2*intrho - 420*alphaz*intrhoz2 - 84*intrhoz2)/840)
                k += 1
                Mv[k] += r31*(L*betay*intrhoy*r33*(4*alphay - 1)/12 + L*betaz*intrhoz*r32*(1 - 4*alphaz)/12) + r32*(L*betay*intrhoz*r31*(20*alphay - 21)/60 + betay**2*r33*(35*L**2*alphay**2*intrho - 77*L**2*alphay*intrho + 44*L**2*intrho + 420*alphay*intrhoy2 + 84*intrhoy2)/840 + betay*betaz*intrhoyz*r32*(-5*alphaz - 1)/10) + r33*(L*betaz*intrhoy*r31*(21 - 20*alphaz)/60 + betay*betaz*intrhoyz*r33*(5*alphay + 1)/10 + betaz**2*r32*(-35*L**2*alphaz**2*intrho + 77*L**2*alphaz*intrho - 44*L**2*intrho - 420*alphaz*intrhoz2 - 84*intrhoz2)/840)
                k += 1
                Mv[k] += r11*(L**2*betay*intrhoz*r33*(5*alphay - 6)/120 + L**2*betaz*intrhoy*r32*(5*alphaz - 6)/120 + L*r31*(intrhoy2 + intrhoz2)/3) + r12*(L**2*betaz*intrhoy*r31*(5*alphaz - 6)/120 + L*betay*betaz*intrhoyz*r33*(-20*alphay*alphaz + 5*alphay + 5*alphaz - 8)/60 + L*betaz**2*r32*(7*L**2*alphaz**2*intrho - 14*L**2*alphaz*intrho + 8*L**2*intrho + 280*alphaz**2*intrhoz2 - 140*alphaz*intrhoz2 + 112*intrhoz2)/840) + r13*(L**2*betay*intrhoz*r31*(5*alphay - 6)/120 + L*betay**2*r33*(7*L**2*alphay**2*intrho - 14*L**2*alphay*intrho + 8*L**2*intrho + 280*alphay**2*intrhoy2 - 140*alphay*intrhoy2 + 112*intrhoy2)/840 + L*betay*betaz*intrhoyz*r32*(-20*alphay*alphaz + 5*alphay + 5*alphaz - 8)/60)
                k += 1
                Mv[k] += r21*(L**2*betay*intrhoz*r33*(5*alphay - 6)/120 + L**2*betaz*intrhoy*r32*(5*alphaz - 6)/120 + L*r31*(intrhoy2 + intrhoz2)/3) + r22*(L**2*betaz*intrhoy*r31*(5*alphaz - 6)/120 + L*betay*betaz*intrhoyz*r33*(-20*alphay*alphaz + 5*alphay + 5*alphaz - 8)/60 + L*betaz**2*r32*(7*L**2*alphaz**2*intrho - 14*L**2*alphaz*intrho + 8*L**2*intrho + 280*alphaz**2*intrhoz2 - 140*alphaz*intrhoz2 + 112*intrhoz2)/840) + r23*(L**2*betay*intrhoz*r31*(5*alphay - 6)/120 + L*betay**2*r33*(7*L**2*alphay**2*intrho - 14*L**2*alphay*intrho + 8*L**2*intrho + 280*alphay**2*intrhoy2 - 140*alphay*intrhoy2 + 112*intrhoy2)/840 + L*betay*betaz*intrhoyz*r32*(-20*alphay*alphaz + 5*alphay + 5*alphaz - 8)/60)
                k += 1
                Mv[k] += r31*(L**2*betay*intrhoz*r33*(5*alphay - 6)/120 + L**2*betaz*intrhoy*r32*(5*alphaz - 6)/120 + L*r31*(intrhoy2 + intrhoz2)/3) + r32*(L**2*betaz*intrhoy*r31*(5*alphaz - 6)/120 + L*betay*betaz*intrhoyz*r33*(-20*alphay*alphaz + 5*alphay + 5*alphaz - 8)/60 + L*betaz**2*r32*(7*L**2*alphaz**2*intrho - 14*L**2*alphaz*intrho + 8*L**2*intrho + 280*alphaz**2*intrhoz2 - 140*alphaz*intrhoz2 + 112*intrhoz2)/840) + r33*(L**2*betay*intrhoz*r31*(5*alphay - 6)/120 + L*betay**2*r33*(7*L**2*alphay**2*intrho - 14*L**2*alphay*intrho + 8*L**2*intrho + 280*alphay**2*intrhoy2 - 140*alphay*intrhoy2 + 112*intrhoy2)/840 + L*betay*betaz*intrhoyz*r32*(-20*alphay*alphaz + 5*alphay + 5*alphaz - 8)/60)
                k += 1
                Mv[k] += r11*(L*betay*intrhoy*r33*(2*alphay + 1)/12 - L*betaz*intrhoz*r32*(2*alphaz + 1)/12) + r12*(L*betay*intrhoz*r31*(10*alphay - 9)/60 + betay**2*r33*(35*L**2*alphay**2*intrho - 63*L**2*alphay*intrho + 26*L**2*intrho - 420*alphay*intrhoy2 - 84*intrhoy2)/840 + betay*betaz*intrhoyz*r32*(5*alphaz + 1)/10) + r13*(L*betaz*intrhoy*r31*(9 - 10*alphaz)/60 + betay*betaz*intrhoyz*r33*(-5*alphay - 1)/10 + betaz**2*r32*(-35*L**2*alphaz**2*intrho + 63*L**2*alphaz*intrho - 26*L**2*intrho + 420*alphaz*intrhoz2 + 84*intrhoz2)/840)
                k += 1
                Mv[k] += r21*(L*betay*intrhoy*r33*(2*alphay + 1)/12 - L*betaz*intrhoz*r32*(2*alphaz + 1)/12) + r22*(L*betay*intrhoz*r31*(10*alphay - 9)/60 + betay**2*r33*(35*L**2*alphay**2*intrho - 63*L**2*alphay*intrho + 26*L**2*intrho - 420*alphay*intrhoy2 - 84*intrhoy2)/840 + betay*betaz*intrhoyz*r32*(5*alphaz + 1)/10) + r23*(L*betaz*intrhoy*r31*(9 - 10*alphaz)/60 + betay*betaz*intrhoyz*r33*(-5*alphay - 1)/10 + betaz**2*r32*(-35*L**2*alphaz**2*intrho + 63*L**2*alphaz*intrho - 26*L**2*intrho + 420*alphaz*intrhoz2 + 84*intrhoz2)/840)
                k += 1
                Mv[k] += r31*(L*betay*intrhoy*r33*(2*alphay + 1)/12 - L*betaz*intrhoz*r32*(2*alphaz + 1)/12) + r32*(L*betay*intrhoz*r31*(10*alphay - 9)/60 + betay**2*r33*(35*L**2*alphay**2*intrho - 63*L**2*alphay*intrho + 26*L**2*intrho - 420*alphay*intrhoy2 - 84*intrhoy2)/840 + betay*betaz*intrhoyz*r32*(5*alphaz + 1)/10) + r33*(L*betaz*intrhoy*r31*(9 - 10*alphaz)/60 + betay*betaz*intrhoyz*r33*(-5*alphay - 1)/10 + betaz**2*r32*(-35*L**2*alphaz**2*intrho + 63*L**2*alphaz*intrho - 26*L**2*intrho + 420*alphaz*intrhoz2 + 84*intrhoz2)/840)
                k += 1
                Mv[k] += r11*(L**2*betay*intrhoz*r33*(5*alphay - 4)/120 + L**2*betaz*intrhoy*r32*(5*alphaz - 4)/120 + L*r31*(intrhoy2 + intrhoz2)/6) + r12*(L**2*betaz*intrhoy*r31*(4 - 5*alphaz)/120 + L*betay*betaz*intrhoyz*r33*(-10*alphay*alphaz - 5*alphay - 5*alphaz + 2)/60 + L*betaz**2*r32*(-7*L**2*alphaz**2*intrho + 14*L**2*alphaz*intrho - 6*L**2*intrho + 140*alphaz**2*intrhoz2 + 140*alphaz*intrhoz2 - 28*intrhoz2)/840) + r13*(L**2*betay*intrhoz*r31*(4 - 5*alphay)/120 + L*betay**2*r33*(-7*L**2*alphay**2*intrho + 14*L**2*alphay*intrho - 6*L**2*intrho + 140*alphay**2*intrhoy2 + 140*alphay*intrhoy2 - 28*intrhoy2)/840 + L*betay*betaz*intrhoyz*r32*(-10*alphay*alphaz - 5*alphay - 5*alphaz + 2)/60)
                k += 1
                Mv[k] += r21*(L**2*betay*intrhoz*r33*(5*alphay - 4)/120 + L**2*betaz*intrhoy*r32*(5*alphaz - 4)/120 + L*r31*(intrhoy2 + intrhoz2)/6) + r22*(L**2*betaz*intrhoy*r31*(4 - 5*alphaz)/120 + L*betay*betaz*intrhoyz*r33*(-10*alphay*alphaz - 5*alphay - 5*alphaz + 2)/60 + L*betaz**2*r32*(-7*L**2*alphaz**2*intrho + 14*L**2*alphaz*intrho - 6*L**2*intrho + 140*alphaz**2*intrhoz2 + 140*alphaz*intrhoz2 - 28*intrhoz2)/840) + r23*(L**2*betay*intrhoz*r31*(4 - 5*alphay)/120 + L*betay**2*r33*(-7*L**2*alphay**2*intrho + 14*L**2*alphay*intrho - 6*L**2*intrho + 140*alphay**2*intrhoy2 + 140*alphay*intrhoy2 - 28*intrhoy2)/840 + L*betay*betaz*intrhoyz*r32*(-10*alphay*alphaz - 5*alphay - 5*alphaz + 2)/60)
                k += 1
                Mv[k] += r31*(L**2*betay*intrhoz*r33*(5*alphay - 4)/120 + L**2*betaz*intrhoy*r32*(5*alphaz - 4)/120 + L*r31*(intrhoy2 + intrhoz2)/6) + r32*(L**2*betaz*intrhoy*r31*(4 - 5*alphaz)/120 + L*betay*betaz*intrhoyz*r33*(-10*alphay*alphaz - 5*alphay - 5*alphaz + 2)/60 + L*betaz**2*r32*(-7*L**2*alphaz**2*intrho + 14*L**2*alphaz*intrho - 6*L**2*intrho + 140*alphaz**2*intrhoz2 + 140*alphaz*intrhoz2 - 28*intrhoz2)/840) + r33*(L**2*betay*intrhoz*r31*(4 - 5*alphay)/120 + L*betay**2*r33*(-7*L**2*alphay**2*intrho + 14*L**2*alphay*intrho - 6*L**2*intrho + 140*alphay**2*intrhoy2 + 140*alphay*intrhoy2 - 28*intrhoy2)/840 + L*betay*betaz*intrhoyz*r32*(-10*alphay*alphaz - 5*alphay - 5*alphaz + 2)/60)
                k += 1
                Mv[k] += r11*(L*intrho*r11/6 - betay*intrhoy*r12/2 - betaz*intrhoz*r13/2) + r12*(betay*intrhoy*r11/2 + betay**2*r12*(35*L**2*alphay**2*intrho - 63*L**2*alphay*intrho + 27*L**2*intrho - 252*intrhoy2)/(210*L) - 6*betay*betaz*intrhoyz*r13/(5*L)) + r13*(betaz*intrhoz*r11/2 - 6*betay*betaz*intrhoyz*r12/(5*L) + betaz**2*r13*(35*L**2*alphaz**2*intrho - 63*L**2*alphaz*intrho + 27*L**2*intrho - 252*intrhoz2)/(210*L))
                k += 1
                Mv[k] += r21*(L*intrho*r11/6 - betay*intrhoy*r12/2 - betaz*intrhoz*r13/2) + r22*(betay*intrhoy*r11/2 + betay**2*r12*(35*L**2*alphay**2*intrho - 63*L**2*alphay*intrho + 27*L**2*intrho - 252*intrhoy2)/(210*L) - 6*betay*betaz*intrhoyz*r13/(5*L)) + r23*(betaz*intrhoz*r11/2 - 6*betay*betaz*intrhoyz*r12/(5*L) + betaz**2*r13*(35*L**2*alphaz**2*intrho - 63*L**2*alphaz*intrho + 27*L**2*intrho - 252*intrhoz2)/(210*L))
                k += 1
                Mv[k] += r31*(L*intrho*r11/6 - betay*intrhoy*r12/2 - betaz*intrhoz*r13/2) + r32*(betay*intrhoy*r11/2 + betay**2*r12*(35*L**2*alphay**2*intrho - 63*L**2*alphay*intrho + 27*L**2*intrho - 252*intrhoy2)/(210*L) - 6*betay*betaz*intrhoyz*r13/(5*L)) + r33*(betaz*intrhoz*r11/2 - 6*betay*betaz*intrhoyz*r12/(5*L) + betaz**2*r13*(35*L**2*alphaz**2*intrho - 63*L**2*alphaz*intrho + 27*L**2*intrho - 252*intrhoz2)/(210*L))
                k += 1
                Mv[k] += r11*(L*betay*intrhoz*r12*(10*alphay - 9)/60 + L*betaz*intrhoy*r13*(9 - 10*alphaz)/60) + r12*(-L*betaz*intrhoz*r11*(2*alphaz + 1)/12 + betay*betaz*intrhoyz*r12*(5*alphaz + 1)/10 + betaz**2*r13*(-35*L**2*alphaz**2*intrho + 63*L**2*alphaz*intrho - 26*L**2*intrho + 420*alphaz*intrhoz2 + 84*intrhoz2)/840) + r13*(L*betay*intrhoy*r11*(2*alphay + 1)/12 + betay**2*r12*(35*L**2*alphay**2*intrho - 63*L**2*alphay*intrho + 26*L**2*intrho - 420*alphay*intrhoy2 - 84*intrhoy2)/840 + betay*betaz*intrhoyz*r13*(-5*alphay - 1)/10)
                k += 1
                Mv[k] += r21*(L*betay*intrhoz*r12*(10*alphay - 9)/60 + L*betaz*intrhoy*r13*(9 - 10*alphaz)/60) + r22*(-L*betaz*intrhoz*r11*(2*alphaz + 1)/12 + betay*betaz*intrhoyz*r12*(5*alphaz + 1)/10 + betaz**2*r13*(-35*L**2*alphaz**2*intrho + 63*L**2*alphaz*intrho - 26*L**2*intrho + 420*alphaz*intrhoz2 + 84*intrhoz2)/840) + r23*(L*betay*intrhoy*r11*(2*alphay + 1)/12 + betay**2*r12*(35*L**2*alphay**2*intrho - 63*L**2*alphay*intrho + 26*L**2*intrho - 420*alphay*intrhoy2 - 84*intrhoy2)/840 + betay*betaz*intrhoyz*r13*(-5*alphay - 1)/10)
                k += 1
                Mv[k] += r31*(L*betay*intrhoz*r12*(10*alphay - 9)/60 + L*betaz*intrhoy*r13*(9 - 10*alphaz)/60) + r32*(-L*betaz*intrhoz*r11*(2*alphaz + 1)/12 + betay*betaz*intrhoyz*r12*(5*alphaz + 1)/10 + betaz**2*r13*(-35*L**2*alphaz**2*intrho + 63*L**2*alphaz*intrho - 26*L**2*intrho + 420*alphaz*intrhoz2 + 84*intrhoz2)/840) + r33*(L*betay*intrhoy*r11*(2*alphay + 1)/12 + betay**2*r12*(35*L**2*alphay**2*intrho - 63*L**2*alphay*intrho + 26*L**2*intrho - 420*alphay*intrhoy2 - 84*intrhoy2)/840 + betay*betaz*intrhoyz*r13*(-5*alphay - 1)/10)
                k += 1
                Mv[k] += r11*(L*intrho*r11/3 - betay*intrhoy*r12/2 - betaz*intrhoz*r13/2) + r12*(-betay*intrhoy*r11/2 + betay**2*r12*(70*L**2*alphay**2*intrho - 147*L**2*alphay*intrho + 78*L**2*intrho + 252*intrhoy2)/(210*L) + 6*betay*betaz*intrhoyz*r13/(5*L)) + r13*(-betaz*intrhoz*r11/2 + 6*betay*betaz*intrhoyz*r12/(5*L) + betaz**2*r13*(70*L**2*alphaz**2*intrho - 147*L**2*alphaz*intrho + 78*L**2*intrho + 252*intrhoz2)/(210*L))
                k += 1
                Mv[k] += r21*(L*intrho*r11/3 - betay*intrhoy*r12/2 - betaz*intrhoz*r13/2) + r22*(-betay*intrhoy*r11/2 + betay**2*r12*(70*L**2*alphay**2*intrho - 147*L**2*alphay*intrho + 78*L**2*intrho + 252*intrhoy2)/(210*L) + 6*betay*betaz*intrhoyz*r13/(5*L)) + r23*(-betaz*intrhoz*r11/2 + 6*betay*betaz*intrhoyz*r12/(5*L) + betaz**2*r13*(70*L**2*alphaz**2*intrho - 147*L**2*alphaz*intrho + 78*L**2*intrho + 252*intrhoz2)/(210*L))
                k += 1
                Mv[k] += r31*(L*intrho*r11/3 - betay*intrhoy*r12/2 - betaz*intrhoz*r13/2) + r32*(-betay*intrhoy*r11/2 + betay**2*r12*(70*L**2*alphay**2*intrho - 147*L**2*alphay*intrho + 78*L**2*intrho + 252*intrhoy2)/(210*L) + 6*betay*betaz*intrhoyz*r13/(5*L)) + r33*(-betaz*intrhoz*r11/2 + 6*betay*betaz*intrhoyz*r12/(5*L) + betaz**2*r13*(70*L**2*alphaz**2*intrho - 147*L**2*alphaz*intrho + 78*L**2*intrho + 252*intrhoz2)/(210*L))
                k += 1
                Mv[k] += r11*(L*betay*intrhoz*r12*(20*alphay - 21)/60 + L*betaz*intrhoy*r13*(21 - 20*alphaz)/60) + r12*(L*betaz*intrhoz*r11*(1 - 4*alphaz)/12 + betay*betaz*intrhoyz*r12*(5*alphaz + 1)/10 + betaz**2*r13*(35*L**2*alphaz**2*intrho - 77*L**2*alphaz*intrho + 44*L**2*intrho + 420*alphaz*intrhoz2 + 84*intrhoz2)/840) + r13*(L*betay*intrhoy*r11*(4*alphay - 1)/12 + betay**2*r12*(-35*L**2*alphay**2*intrho + 77*L**2*alphay*intrho - 44*L**2*intrho - 420*alphay*intrhoy2 - 84*intrhoy2)/840 + betay*betaz*intrhoyz*r13*(-5*alphay - 1)/10)
                k += 1
                Mv[k] += r21*(L*betay*intrhoz*r12*(20*alphay - 21)/60 + L*betaz*intrhoy*r13*(21 - 20*alphaz)/60) + r22*(L*betaz*intrhoz*r11*(1 - 4*alphaz)/12 + betay*betaz*intrhoyz*r12*(5*alphaz + 1)/10 + betaz**2*r13*(35*L**2*alphaz**2*intrho - 77*L**2*alphaz*intrho + 44*L**2*intrho + 420*alphaz*intrhoz2 + 84*intrhoz2)/840) + r23*(L*betay*intrhoy*r11*(4*alphay - 1)/12 + betay**2*r12*(-35*L**2*alphay**2*intrho + 77*L**2*alphay*intrho - 44*L**2*intrho - 420*alphay*intrhoy2 - 84*intrhoy2)/840 + betay*betaz*intrhoyz*r13*(-5*alphay - 1)/10)
                k += 1
                Mv[k] += r31*(L*betay*intrhoz*r12*(20*alphay - 21)/60 + L*betaz*intrhoy*r13*(21 - 20*alphaz)/60) + r32*(L*betaz*intrhoz*r11*(1 - 4*alphaz)/12 + betay*betaz*intrhoyz*r12*(5*alphaz + 1)/10 + betaz**2*r13*(35*L**2*alphaz**2*intrho - 77*L**2*alphaz*intrho + 44*L**2*intrho + 420*alphaz*intrhoz2 + 84*intrhoz2)/840) + r33*(L*betay*intrhoy*r11*(4*alphay - 1)/12 + betay**2*r12*(-35*L**2*alphay**2*intrho + 77*L**2*alphay*intrho - 44*L**2*intrho - 420*alphay*intrhoy2 - 84*intrhoy2)/840 + betay*betaz*intrhoyz*r13*(-5*alphay - 1)/10)
                k += 1
                Mv[k] += r11*(L*intrho*r21/6 - betay*intrhoy*r22/2 - betaz*intrhoz*r23/2) + r12*(betay*intrhoy*r21/2 + betay**2*r22*(35*L**2*alphay**2*intrho - 63*L**2*alphay*intrho + 27*L**2*intrho - 252*intrhoy2)/(210*L) - 6*betay*betaz*intrhoyz*r23/(5*L)) + r13*(betaz*intrhoz*r21/2 - 6*betay*betaz*intrhoyz*r22/(5*L) + betaz**2*r23*(35*L**2*alphaz**2*intrho - 63*L**2*alphaz*intrho + 27*L**2*intrho - 252*intrhoz2)/(210*L))
                k += 1
                Mv[k] += r21*(L*intrho*r21/6 - betay*intrhoy*r22/2 - betaz*intrhoz*r23/2) + r22*(betay*intrhoy*r21/2 + betay**2*r22*(35*L**2*alphay**2*intrho - 63*L**2*alphay*intrho + 27*L**2*intrho - 252*intrhoy2)/(210*L) - 6*betay*betaz*intrhoyz*r23/(5*L)) + r23*(betaz*intrhoz*r21/2 - 6*betay*betaz*intrhoyz*r22/(5*L) + betaz**2*r23*(35*L**2*alphaz**2*intrho - 63*L**2*alphaz*intrho + 27*L**2*intrho - 252*intrhoz2)/(210*L))
                k += 1
                Mv[k] += r31*(L*intrho*r21/6 - betay*intrhoy*r22/2 - betaz*intrhoz*r23/2) + r32*(betay*intrhoy*r21/2 + betay**2*r22*(35*L**2*alphay**2*intrho - 63*L**2*alphay*intrho + 27*L**2*intrho - 252*intrhoy2)/(210*L) - 6*betay*betaz*intrhoyz*r23/(5*L)) + r33*(betaz*intrhoz*r21/2 - 6*betay*betaz*intrhoyz*r22/(5*L) + betaz**2*r23*(35*L**2*alphaz**2*intrho - 63*L**2*alphaz*intrho + 27*L**2*intrho - 252*intrhoz2)/(210*L))
                k += 1
                Mv[k] += r11*(L*betay*intrhoz*r22*(10*alphay - 9)/60 + L*betaz*intrhoy*r23*(9 - 10*alphaz)/60) + r12*(-L*betaz*intrhoz*r21*(2*alphaz + 1)/12 + betay*betaz*intrhoyz*r22*(5*alphaz + 1)/10 + betaz**2*r23*(-35*L**2*alphaz**2*intrho + 63*L**2*alphaz*intrho - 26*L**2*intrho + 420*alphaz*intrhoz2 + 84*intrhoz2)/840) + r13*(L*betay*intrhoy*r21*(2*alphay + 1)/12 + betay**2*r22*(35*L**2*alphay**2*intrho - 63*L**2*alphay*intrho + 26*L**2*intrho - 420*alphay*intrhoy2 - 84*intrhoy2)/840 + betay*betaz*intrhoyz*r23*(-5*alphay - 1)/10)
                k += 1
                Mv[k] += r21*(L*betay*intrhoz*r22*(10*alphay - 9)/60 + L*betaz*intrhoy*r23*(9 - 10*alphaz)/60) + r22*(-L*betaz*intrhoz*r21*(2*alphaz + 1)/12 + betay*betaz*intrhoyz*r22*(5*alphaz + 1)/10 + betaz**2*r23*(-35*L**2*alphaz**2*intrho + 63*L**2*alphaz*intrho - 26*L**2*intrho + 420*alphaz*intrhoz2 + 84*intrhoz2)/840) + r23*(L*betay*intrhoy*r21*(2*alphay + 1)/12 + betay**2*r22*(35*L**2*alphay**2*intrho - 63*L**2*alphay*intrho + 26*L**2*intrho - 420*alphay*intrhoy2 - 84*intrhoy2)/840 + betay*betaz*intrhoyz*r23*(-5*alphay - 1)/10)
                k += 1
                Mv[k] += r31*(L*betay*intrhoz*r22*(10*alphay - 9)/60 + L*betaz*intrhoy*r23*(9 - 10*alphaz)/60) + r32*(-L*betaz*intrhoz*r21*(2*alphaz + 1)/12 + betay*betaz*intrhoyz*r22*(5*alphaz + 1)/10 + betaz**2*r23*(-35*L**2*alphaz**2*intrho + 63*L**2*alphaz*intrho - 26*L**2*intrho + 420*alphaz*intrhoz2 + 84*intrhoz2)/840) + r33*(L*betay*intrhoy*r21*(2*alphay + 1)/12 + betay**2*r22*(35*L**2*alphay**2*intrho - 63*L**2*alphay*intrho + 26*L**2*intrho - 420*alphay*intrhoy2 - 84*intrhoy2)/840 + betay*betaz*intrhoyz*r23*(-5*alphay - 1)/10)
                k += 1
                Mv[k] += r11*(L*intrho*r21/3 - betay*intrhoy*r22/2 - betaz*intrhoz*r23/2) + r12*(-betay*intrhoy*r21/2 + betay**2*r22*(70*L**2*alphay**2*intrho - 147*L**2*alphay*intrho + 78*L**2*intrho + 252*intrhoy2)/(210*L) + 6*betay*betaz*intrhoyz*r23/(5*L)) + r13*(-betaz*intrhoz*r21/2 + 6*betay*betaz*intrhoyz*r22/(5*L) + betaz**2*r23*(70*L**2*alphaz**2*intrho - 147*L**2*alphaz*intrho + 78*L**2*intrho + 252*intrhoz2)/(210*L))
                k += 1
                Mv[k] += r21*(L*intrho*r21/3 - betay*intrhoy*r22/2 - betaz*intrhoz*r23/2) + r22*(-betay*intrhoy*r21/2 + betay**2*r22*(70*L**2*alphay**2*intrho - 147*L**2*alphay*intrho + 78*L**2*intrho + 252*intrhoy2)/(210*L) + 6*betay*betaz*intrhoyz*r23/(5*L)) + r23*(-betaz*intrhoz*r21/2 + 6*betay*betaz*intrhoyz*r22/(5*L) + betaz**2*r23*(70*L**2*alphaz**2*intrho - 147*L**2*alphaz*intrho + 78*L**2*intrho + 252*intrhoz2)/(210*L))
                k += 1
                Mv[k] += r31*(L*intrho*r21/3 - betay*intrhoy*r22/2 - betaz*intrhoz*r23/2) + r32*(-betay*intrhoy*r21/2 + betay**2*r22*(70*L**2*alphay**2*intrho - 147*L**2*alphay*intrho + 78*L**2*intrho + 252*intrhoy2)/(210*L) + 6*betay*betaz*intrhoyz*r23/(5*L)) + r33*(-betaz*intrhoz*r21/2 + 6*betay*betaz*intrhoyz*r22/(5*L) + betaz**2*r23*(70*L**2*alphaz**2*intrho - 147*L**2*alphaz*intrho + 78*L**2*intrho + 252*intrhoz2)/(210*L))
                k += 1
                Mv[k] += r11*(L*betay*intrhoz*r22*(20*alphay - 21)/60 + L*betaz*intrhoy*r23*(21 - 20*alphaz)/60) + r12*(L*betaz*intrhoz*r21*(1 - 4*alphaz)/12 + betay*betaz*intrhoyz*r22*(5*alphaz + 1)/10 + betaz**2*r23*(35*L**2*alphaz**2*intrho - 77*L**2*alphaz*intrho + 44*L**2*intrho + 420*alphaz*intrhoz2 + 84*intrhoz2)/840) + r13*(L*betay*intrhoy*r21*(4*alphay - 1)/12 + betay**2*r22*(-35*L**2*alphay**2*intrho + 77*L**2*alphay*intrho - 44*L**2*intrho - 420*alphay*intrhoy2 - 84*intrhoy2)/840 + betay*betaz*intrhoyz*r23*(-5*alphay - 1)/10)
                k += 1
                Mv[k] += r21*(L*betay*intrhoz*r22*(20*alphay - 21)/60 + L*betaz*intrhoy*r23*(21 - 20*alphaz)/60) + r22*(L*betaz*intrhoz*r21*(1 - 4*alphaz)/12 + betay*betaz*intrhoyz*r22*(5*alphaz + 1)/10 + betaz**2*r23*(35*L**2*alphaz**2*intrho - 77*L**2*alphaz*intrho + 44*L**2*intrho + 420*alphaz*intrhoz2 + 84*intrhoz2)/840) + r23*(L*betay*intrhoy*r21*(4*alphay - 1)/12 + betay**2*r22*(-35*L**2*alphay**2*intrho + 77*L**2*alphay*intrho - 44*L**2*intrho - 420*alphay*intrhoy2 - 84*intrhoy2)/840 + betay*betaz*intrhoyz*r23*(-5*alphay - 1)/10)
                k += 1
                Mv[k] += r31*(L*betay*intrhoz*r22*(20*alphay - 21)/60 + L*betaz*intrhoy*r23*(21 - 20*alphaz)/60) + r32*(L*betaz*intrhoz*r21*(1 - 4*alphaz)/12 + betay*betaz*intrhoyz*r22*(5*alphaz + 1)/10 + betaz**2*r23*(35*L**2*alphaz**2*intrho - 77*L**2*alphaz*intrho + 44*L**2*intrho + 420*alphaz*intrhoz2 + 84*intrhoz2)/840) + r33*(L*betay*intrhoy*r21*(4*alphay - 1)/12 + betay**2*r22*(-35*L**2*alphay**2*intrho + 77*L**2*alphay*intrho - 44*L**2*intrho - 420*alphay*intrhoy2 - 84*intrhoy2)/840 + betay*betaz*intrhoyz*r23*(-5*alphay - 1)/10)
                k += 1
                Mv[k] += r11*(L*intrho*r31/6 - betay*intrhoy*r32/2 - betaz*intrhoz*r33/2) + r12*(betay*intrhoy*r31/2 + betay**2*r32*(35*L**2*alphay**2*intrho - 63*L**2*alphay*intrho + 27*L**2*intrho - 252*intrhoy2)/(210*L) - 6*betay*betaz*intrhoyz*r33/(5*L)) + r13*(betaz*intrhoz*r31/2 - 6*betay*betaz*intrhoyz*r32/(5*L) + betaz**2*r33*(35*L**2*alphaz**2*intrho - 63*L**2*alphaz*intrho + 27*L**2*intrho - 252*intrhoz2)/(210*L))
                k += 1
                Mv[k] += r21*(L*intrho*r31/6 - betay*intrhoy*r32/2 - betaz*intrhoz*r33/2) + r22*(betay*intrhoy*r31/2 + betay**2*r32*(35*L**2*alphay**2*intrho - 63*L**2*alphay*intrho + 27*L**2*intrho - 252*intrhoy2)/(210*L) - 6*betay*betaz*intrhoyz*r33/(5*L)) + r23*(betaz*intrhoz*r31/2 - 6*betay*betaz*intrhoyz*r32/(5*L) + betaz**2*r33*(35*L**2*alphaz**2*intrho - 63*L**2*alphaz*intrho + 27*L**2*intrho - 252*intrhoz2)/(210*L))
                k += 1
                Mv[k] += r31*(L*intrho*r31/6 - betay*intrhoy*r32/2 - betaz*intrhoz*r33/2) + r32*(betay*intrhoy*r31/2 + betay**2*r32*(35*L**2*alphay**2*intrho - 63*L**2*alphay*intrho + 27*L**2*intrho - 252*intrhoy2)/(210*L) - 6*betay*betaz*intrhoyz*r33/(5*L)) + r33*(betaz*intrhoz*r31/2 - 6*betay*betaz*intrhoyz*r32/(5*L) + betaz**2*r33*(35*L**2*alphaz**2*intrho - 63*L**2*alphaz*intrho + 27*L**2*intrho - 252*intrhoz2)/(210*L))
                k += 1
                Mv[k] += r11*(L*betay*intrhoz*r32*(10*alphay - 9)/60 + L*betaz*intrhoy*r33*(9 - 10*alphaz)/60) + r12*(-L*betaz*intrhoz*r31*(2*alphaz + 1)/12 + betay*betaz*intrhoyz*r32*(5*alphaz + 1)/10 + betaz**2*r33*(-35*L**2*alphaz**2*intrho + 63*L**2*alphaz*intrho - 26*L**2*intrho + 420*alphaz*intrhoz2 + 84*intrhoz2)/840) + r13*(L*betay*intrhoy*r31*(2*alphay + 1)/12 + betay**2*r32*(35*L**2*alphay**2*intrho - 63*L**2*alphay*intrho + 26*L**2*intrho - 420*alphay*intrhoy2 - 84*intrhoy2)/840 + betay*betaz*intrhoyz*r33*(-5*alphay - 1)/10)
                k += 1
                Mv[k] += r21*(L*betay*intrhoz*r32*(10*alphay - 9)/60 + L*betaz*intrhoy*r33*(9 - 10*alphaz)/60) + r22*(-L*betaz*intrhoz*r31*(2*alphaz + 1)/12 + betay*betaz*intrhoyz*r32*(5*alphaz + 1)/10 + betaz**2*r33*(-35*L**2*alphaz**2*intrho + 63*L**2*alphaz*intrho - 26*L**2*intrho + 420*alphaz*intrhoz2 + 84*intrhoz2)/840) + r23*(L*betay*intrhoy*r31*(2*alphay + 1)/12 + betay**2*r32*(35*L**2*alphay**2*intrho - 63*L**2*alphay*intrho + 26*L**2*intrho - 420*alphay*intrhoy2 - 84*intrhoy2)/840 + betay*betaz*intrhoyz*r33*(-5*alphay - 1)/10)
                k += 1
                Mv[k] += r31*(L*betay*intrhoz*r32*(10*alphay - 9)/60 + L*betaz*intrhoy*r33*(9 - 10*alphaz)/60) + r32*(-L*betaz*intrhoz*r31*(2*alphaz + 1)/12 + betay*betaz*intrhoyz*r32*(5*alphaz + 1)/10 + betaz**2*r33*(-35*L**2*alphaz**2*intrho + 63*L**2*alphaz*intrho - 26*L**2*intrho + 420*alphaz*intrhoz2 + 84*intrhoz2)/840) + r33*(L*betay*intrhoy*r31*(2*alphay + 1)/12 + betay**2*r32*(35*L**2*alphay**2*intrho - 63*L**2*alphay*intrho + 26*L**2*intrho - 420*alphay*intrhoy2 - 84*intrhoy2)/840 + betay*betaz*intrhoyz*r33*(-5*alphay - 1)/10)
                k += 1
                Mv[k] += r11*(L*intrho*r31/3 - betay*intrhoy*r32/2 - betaz*intrhoz*r33/2) + r12*(-betay*intrhoy*r31/2 + betay**2*r32*(70*L**2*alphay**2*intrho - 147*L**2*alphay*intrho + 78*L**2*intrho + 252*intrhoy2)/(210*L) + 6*betay*betaz*intrhoyz*r33/(5*L)) + r13*(-betaz*intrhoz*r31/2 + 6*betay*betaz*intrhoyz*r32/(5*L) + betaz**2*r33*(70*L**2*alphaz**2*intrho - 147*L**2*alphaz*intrho + 78*L**2*intrho + 252*intrhoz2)/(210*L))
                k += 1
                Mv[k] += r21*(L*intrho*r31/3 - betay*intrhoy*r32/2 - betaz*intrhoz*r33/2) + r22*(-betay*intrhoy*r31/2 + betay**2*r32*(70*L**2*alphay**2*intrho - 147*L**2*alphay*intrho + 78*L**2*intrho + 252*intrhoy2)/(210*L) + 6*betay*betaz*intrhoyz*r33/(5*L)) + r23*(-betaz*intrhoz*r31/2 + 6*betay*betaz*intrhoyz*r32/(5*L) + betaz**2*r33*(70*L**2*alphaz**2*intrho - 147*L**2*alphaz*intrho + 78*L**2*intrho + 252*intrhoz2)/(210*L))
                k += 1
                Mv[k] += r31*(L*intrho*r31/3 - betay*intrhoy*r32/2 - betaz*intrhoz*r33/2) + r32*(-betay*intrhoy*r31/2 + betay**2*r32*(70*L**2*alphay**2*intrho - 147*L**2*alphay*intrho + 78*L**2*intrho + 252*intrhoy2)/(210*L) + 6*betay*betaz*intrhoyz*r33/(5*L)) + r33*(-betaz*intrhoz*r31/2 + 6*betay*betaz*intrhoyz*r32/(5*L) + betaz**2*r33*(70*L**2*alphaz**2*intrho - 147*L**2*alphaz*intrho + 78*L**2*intrho + 252*intrhoz2)/(210*L))
                k += 1
                Mv[k] += r11*(L*betay*intrhoz*r32*(20*alphay - 21)/60 + L*betaz*intrhoy*r33*(21 - 20*alphaz)/60) + r12*(L*betaz*intrhoz*r31*(1 - 4*alphaz)/12 + betay*betaz*intrhoyz*r32*(5*alphaz + 1)/10 + betaz**2*r33*(35*L**2*alphaz**2*intrho - 77*L**2*alphaz*intrho + 44*L**2*intrho + 420*alphaz*intrhoz2 + 84*intrhoz2)/840) + r13*(L*betay*intrhoy*r31*(4*alphay - 1)/12 + betay**2*r32*(-35*L**2*alphay**2*intrho + 77*L**2*alphay*intrho - 44*L**2*intrho - 420*alphay*intrhoy2 - 84*intrhoy2)/840 + betay*betaz*intrhoyz*r33*(-5*alphay - 1)/10)
                k += 1
                Mv[k] += r21*(L*betay*intrhoz*r32*(20*alphay - 21)/60 + L*betaz*intrhoy*r33*(21 - 20*alphaz)/60) + r22*(L*betaz*intrhoz*r31*(1 - 4*alphaz)/12 + betay*betaz*intrhoyz*r32*(5*alphaz + 1)/10 + betaz**2*r33*(35*L**2*alphaz**2*intrho - 77*L**2*alphaz*intrho + 44*L**2*intrho + 420*alphaz*intrhoz2 + 84*intrhoz2)/840) + r23*(L*betay*intrhoy*r31*(4*alphay - 1)/12 + betay**2*r32*(-35*L**2*alphay**2*intrho + 77*L**2*alphay*intrho - 44*L**2*intrho - 420*alphay*intrhoy2 - 84*intrhoy2)/840 + betay*betaz*intrhoyz*r33*(-5*alphay - 1)/10)
                k += 1
                Mv[k] += r31*(L*betay*intrhoz*r32*(20*alphay - 21)/60 + L*betaz*intrhoy*r33*(21 - 20*alphaz)/60) + r32*(L*betaz*intrhoz*r31*(1 - 4*alphaz)/12 + betay*betaz*intrhoyz*r32*(5*alphaz + 1)/10 + betaz**2*r33*(35*L**2*alphaz**2*intrho - 77*L**2*alphaz*intrho + 44*L**2*intrho + 420*alphaz*intrhoz2 + 84*intrhoz2)/840) + r33*(L*betay*intrhoy*r31*(4*alphay - 1)/12 + betay**2*r32*(-35*L**2*alphay**2*intrho + 77*L**2*alphay*intrho - 44*L**2*intrho - 420*alphay*intrhoy2 - 84*intrhoy2)/840 + betay*betaz*intrhoyz*r33*(-5*alphay - 1)/10)
                k += 1
                Mv[k] += r11*(L*betay*intrhoy*r13*(2*alphay + 1)/12 - L*betaz*intrhoz*r12*(2*alphaz + 1)/12) + r12*(L*betay*intrhoz*r11*(10*alphay - 9)/60 + betay**2*r13*(-35*L**2*alphay**2*intrho + 63*L**2*alphay*intrho - 26*L**2*intrho + 420*alphay*intrhoy2 + 84*intrhoy2)/840 + betay*betaz*intrhoyz*r12*(-5*alphaz - 1)/10) + r13*(L*betaz*intrhoy*r11*(9 - 10*alphaz)/60 + betay*betaz*intrhoyz*r13*(5*alphay + 1)/10 + betaz**2*r12*(35*L**2*alphaz**2*intrho - 63*L**2*alphaz*intrho + 26*L**2*intrho - 420*alphaz*intrhoz2 - 84*intrhoz2)/840)
                k += 1
                Mv[k] += r21*(L*betay*intrhoy*r13*(2*alphay + 1)/12 - L*betaz*intrhoz*r12*(2*alphaz + 1)/12) + r22*(L*betay*intrhoz*r11*(10*alphay - 9)/60 + betay**2*r13*(-35*L**2*alphay**2*intrho + 63*L**2*alphay*intrho - 26*L**2*intrho + 420*alphay*intrhoy2 + 84*intrhoy2)/840 + betay*betaz*intrhoyz*r12*(-5*alphaz - 1)/10) + r23*(L*betaz*intrhoy*r11*(9 - 10*alphaz)/60 + betay*betaz*intrhoyz*r13*(5*alphay + 1)/10 + betaz**2*r12*(35*L**2*alphaz**2*intrho - 63*L**2*alphaz*intrho + 26*L**2*intrho - 420*alphaz*intrhoz2 - 84*intrhoz2)/840)
                k += 1
                Mv[k] += r31*(L*betay*intrhoy*r13*(2*alphay + 1)/12 - L*betaz*intrhoz*r12*(2*alphaz + 1)/12) + r32*(L*betay*intrhoz*r11*(10*alphay - 9)/60 + betay**2*r13*(-35*L**2*alphay**2*intrho + 63*L**2*alphay*intrho - 26*L**2*intrho + 420*alphay*intrhoy2 + 84*intrhoy2)/840 + betay*betaz*intrhoyz*r12*(-5*alphaz - 1)/10) + r33*(L*betaz*intrhoy*r11*(9 - 10*alphaz)/60 + betay*betaz*intrhoyz*r13*(5*alphay + 1)/10 + betaz**2*r12*(35*L**2*alphaz**2*intrho - 63*L**2*alphaz*intrho + 26*L**2*intrho - 420*alphaz*intrhoz2 - 84*intrhoz2)/840)
                k += 1
                Mv[k] += r11*(L**2*betay*intrhoz*r13*(4 - 5*alphay)/120 + L**2*betaz*intrhoy*r12*(4 - 5*alphaz)/120 + L*r11*(intrhoy2 + intrhoz2)/6) + r12*(L**2*betaz*intrhoy*r11*(5*alphaz - 4)/120 + L*betay*betaz*intrhoyz*r13*(-10*alphay*alphaz - 5*alphay - 5*alphaz + 2)/60 + L*betaz**2*r12*(-7*L**2*alphaz**2*intrho + 14*L**2*alphaz*intrho - 6*L**2*intrho + 140*alphaz**2*intrhoz2 + 140*alphaz*intrhoz2 - 28*intrhoz2)/840) + r13*(L**2*betay*intrhoz*r11*(5*alphay - 4)/120 + L*betay**2*r13*(-7*L**2*alphay**2*intrho + 14*L**2*alphay*intrho - 6*L**2*intrho + 140*alphay**2*intrhoy2 + 140*alphay*intrhoy2 - 28*intrhoy2)/840 + L*betay*betaz*intrhoyz*r12*(-10*alphay*alphaz - 5*alphay - 5*alphaz + 2)/60)
                k += 1
                Mv[k] += r21*(L**2*betay*intrhoz*r13*(4 - 5*alphay)/120 + L**2*betaz*intrhoy*r12*(4 - 5*alphaz)/120 + L*r11*(intrhoy2 + intrhoz2)/6) + r22*(L**2*betaz*intrhoy*r11*(5*alphaz - 4)/120 + L*betay*betaz*intrhoyz*r13*(-10*alphay*alphaz - 5*alphay - 5*alphaz + 2)/60 + L*betaz**2*r12*(-7*L**2*alphaz**2*intrho + 14*L**2*alphaz*intrho - 6*L**2*intrho + 140*alphaz**2*intrhoz2 + 140*alphaz*intrhoz2 - 28*intrhoz2)/840) + r23*(L**2*betay*intrhoz*r11*(5*alphay - 4)/120 + L*betay**2*r13*(-7*L**2*alphay**2*intrho + 14*L**2*alphay*intrho - 6*L**2*intrho + 140*alphay**2*intrhoy2 + 140*alphay*intrhoy2 - 28*intrhoy2)/840 + L*betay*betaz*intrhoyz*r12*(-10*alphay*alphaz - 5*alphay - 5*alphaz + 2)/60)
                k += 1
                Mv[k] += r31*(L**2*betay*intrhoz*r13*(4 - 5*alphay)/120 + L**2*betaz*intrhoy*r12*(4 - 5*alphaz)/120 + L*r11*(intrhoy2 + intrhoz2)/6) + r32*(L**2*betaz*intrhoy*r11*(5*alphaz - 4)/120 + L*betay*betaz*intrhoyz*r13*(-10*alphay*alphaz - 5*alphay - 5*alphaz + 2)/60 + L*betaz**2*r12*(-7*L**2*alphaz**2*intrho + 14*L**2*alphaz*intrho - 6*L**2*intrho + 140*alphaz**2*intrhoz2 + 140*alphaz*intrhoz2 - 28*intrhoz2)/840) + r33*(L**2*betay*intrhoz*r11*(5*alphay - 4)/120 + L*betay**2*r13*(-7*L**2*alphay**2*intrho + 14*L**2*alphay*intrho - 6*L**2*intrho + 140*alphay**2*intrhoy2 + 140*alphay*intrhoy2 - 28*intrhoy2)/840 + L*betay*betaz*intrhoyz*r12*(-10*alphay*alphaz - 5*alphay - 5*alphaz + 2)/60)
                k += 1
                Mv[k] += r11*(L*betay*intrhoy*r13*(4*alphay - 1)/12 + L*betaz*intrhoz*r12*(1 - 4*alphaz)/12) + r12*(L*betay*intrhoz*r11*(20*alphay - 21)/60 + betay**2*r13*(-35*L**2*alphay**2*intrho + 77*L**2*alphay*intrho - 44*L**2*intrho - 420*alphay*intrhoy2 - 84*intrhoy2)/840 + betay*betaz*intrhoyz*r12*(5*alphaz + 1)/10) + r13*(L*betaz*intrhoy*r11*(21 - 20*alphaz)/60 + betay*betaz*intrhoyz*r13*(-5*alphay - 1)/10 + betaz**2*r12*(35*L**2*alphaz**2*intrho - 77*L**2*alphaz*intrho + 44*L**2*intrho + 420*alphaz*intrhoz2 + 84*intrhoz2)/840)
                k += 1
                Mv[k] += r21*(L*betay*intrhoy*r13*(4*alphay - 1)/12 + L*betaz*intrhoz*r12*(1 - 4*alphaz)/12) + r22*(L*betay*intrhoz*r11*(20*alphay - 21)/60 + betay**2*r13*(-35*L**2*alphay**2*intrho + 77*L**2*alphay*intrho - 44*L**2*intrho - 420*alphay*intrhoy2 - 84*intrhoy2)/840 + betay*betaz*intrhoyz*r12*(5*alphaz + 1)/10) + r23*(L*betaz*intrhoy*r11*(21 - 20*alphaz)/60 + betay*betaz*intrhoyz*r13*(-5*alphay - 1)/10 + betaz**2*r12*(35*L**2*alphaz**2*intrho - 77*L**2*alphaz*intrho + 44*L**2*intrho + 420*alphaz*intrhoz2 + 84*intrhoz2)/840)
                k += 1
                Mv[k] += r31*(L*betay*intrhoy*r13*(4*alphay - 1)/12 + L*betaz*intrhoz*r12*(1 - 4*alphaz)/12) + r32*(L*betay*intrhoz*r11*(20*alphay - 21)/60 + betay**2*r13*(-35*L**2*alphay**2*intrho + 77*L**2*alphay*intrho - 44*L**2*intrho - 420*alphay*intrhoy2 - 84*intrhoy2)/840 + betay*betaz*intrhoyz*r12*(5*alphaz + 1)/10) + r33*(L*betaz*intrhoy*r11*(21 - 20*alphaz)/60 + betay*betaz*intrhoyz*r13*(-5*alphay - 1)/10 + betaz**2*r12*(35*L**2*alphaz**2*intrho - 77*L**2*alphaz*intrho + 44*L**2*intrho + 420*alphaz*intrhoz2 + 84*intrhoz2)/840)
                k += 1
                Mv[k] += r11*(L**2*betay*intrhoz*r13*(6 - 5*alphay)/120 + L**2*betaz*intrhoy*r12*(6 - 5*alphaz)/120 + L*r11*(intrhoy2 + intrhoz2)/3) + r12*(L**2*betaz*intrhoy*r11*(6 - 5*alphaz)/120 + L*betay*betaz*intrhoyz*r13*(-20*alphay*alphaz + 5*alphay + 5*alphaz - 8)/60 + L*betaz**2*r12*(7*L**2*alphaz**2*intrho - 14*L**2*alphaz*intrho + 8*L**2*intrho + 280*alphaz**2*intrhoz2 - 140*alphaz*intrhoz2 + 112*intrhoz2)/840) + r13*(L**2*betay*intrhoz*r11*(6 - 5*alphay)/120 + L*betay**2*r13*(7*L**2*alphay**2*intrho - 14*L**2*alphay*intrho + 8*L**2*intrho + 280*alphay**2*intrhoy2 - 140*alphay*intrhoy2 + 112*intrhoy2)/840 + L*betay*betaz*intrhoyz*r12*(-20*alphay*alphaz + 5*alphay + 5*alphaz - 8)/60)
                k += 1
                Mv[k] += r21*(L**2*betay*intrhoz*r13*(6 - 5*alphay)/120 + L**2*betaz*intrhoy*r12*(6 - 5*alphaz)/120 + L*r11*(intrhoy2 + intrhoz2)/3) + r22*(L**2*betaz*intrhoy*r11*(6 - 5*alphaz)/120 + L*betay*betaz*intrhoyz*r13*(-20*alphay*alphaz + 5*alphay + 5*alphaz - 8)/60 + L*betaz**2*r12*(7*L**2*alphaz**2*intrho - 14*L**2*alphaz*intrho + 8*L**2*intrho + 280*alphaz**2*intrhoz2 - 140*alphaz*intrhoz2 + 112*intrhoz2)/840) + r23*(L**2*betay*intrhoz*r11*(6 - 5*alphay)/120 + L*betay**2*r13*(7*L**2*alphay**2*intrho - 14*L**2*alphay*intrho + 8*L**2*intrho + 280*alphay**2*intrhoy2 - 140*alphay*intrhoy2 + 112*intrhoy2)/840 + L*betay*betaz*intrhoyz*r12*(-20*alphay*alphaz + 5*alphay + 5*alphaz - 8)/60)
                k += 1
                Mv[k] += r31*(L**2*betay*intrhoz*r13*(6 - 5*alphay)/120 + L**2*betaz*intrhoy*r12*(6 - 5*alphaz)/120 + L*r11*(intrhoy2 + intrhoz2)/3) + r32*(L**2*betaz*intrhoy*r11*(6 - 5*alphaz)/120 + L*betay*betaz*intrhoyz*r13*(-20*alphay*alphaz + 5*alphay + 5*alphaz - 8)/60 + L*betaz**2*r12*(7*L**2*alphaz**2*intrho - 14*L**2*alphaz*intrho + 8*L**2*intrho + 280*alphaz**2*intrhoz2 - 140*alphaz*intrhoz2 + 112*intrhoz2)/840) + r33*(L**2*betay*intrhoz*r11*(6 - 5*alphay)/120 + L*betay**2*r13*(7*L**2*alphay**2*intrho - 14*L**2*alphay*intrho + 8*L**2*intrho + 280*alphay**2*intrhoy2 - 140*alphay*intrhoy2 + 112*intrhoy2)/840 + L*betay*betaz*intrhoyz*r12*(-20*alphay*alphaz + 5*alphay + 5*alphaz - 8)/60)
                k += 1
                Mv[k] += r11*(L*betay*intrhoy*r23*(2*alphay + 1)/12 - L*betaz*intrhoz*r22*(2*alphaz + 1)/12) + r12*(L*betay*intrhoz*r21*(10*alphay - 9)/60 + betay**2*r23*(-35*L**2*alphay**2*intrho + 63*L**2*alphay*intrho - 26*L**2*intrho + 420*alphay*intrhoy2 + 84*intrhoy2)/840 + betay*betaz*intrhoyz*r22*(-5*alphaz - 1)/10) + r13*(L*betaz*intrhoy*r21*(9 - 10*alphaz)/60 + betay*betaz*intrhoyz*r23*(5*alphay + 1)/10 + betaz**2*r22*(35*L**2*alphaz**2*intrho - 63*L**2*alphaz*intrho + 26*L**2*intrho - 420*alphaz*intrhoz2 - 84*intrhoz2)/840)
                k += 1
                Mv[k] += r21*(L*betay*intrhoy*r23*(2*alphay + 1)/12 - L*betaz*intrhoz*r22*(2*alphaz + 1)/12) + r22*(L*betay*intrhoz*r21*(10*alphay - 9)/60 + betay**2*r23*(-35*L**2*alphay**2*intrho + 63*L**2*alphay*intrho - 26*L**2*intrho + 420*alphay*intrhoy2 + 84*intrhoy2)/840 + betay*betaz*intrhoyz*r22*(-5*alphaz - 1)/10) + r23*(L*betaz*intrhoy*r21*(9 - 10*alphaz)/60 + betay*betaz*intrhoyz*r23*(5*alphay + 1)/10 + betaz**2*r22*(35*L**2*alphaz**2*intrho - 63*L**2*alphaz*intrho + 26*L**2*intrho - 420*alphaz*intrhoz2 - 84*intrhoz2)/840)
                k += 1
                Mv[k] += r31*(L*betay*intrhoy*r23*(2*alphay + 1)/12 - L*betaz*intrhoz*r22*(2*alphaz + 1)/12) + r32*(L*betay*intrhoz*r21*(10*alphay - 9)/60 + betay**2*r23*(-35*L**2*alphay**2*intrho + 63*L**2*alphay*intrho - 26*L**2*intrho + 420*alphay*intrhoy2 + 84*intrhoy2)/840 + betay*betaz*intrhoyz*r22*(-5*alphaz - 1)/10) + r33*(L*betaz*intrhoy*r21*(9 - 10*alphaz)/60 + betay*betaz*intrhoyz*r23*(5*alphay + 1)/10 + betaz**2*r22*(35*L**2*alphaz**2*intrho - 63*L**2*alphaz*intrho + 26*L**2*intrho - 420*alphaz*intrhoz2 - 84*intrhoz2)/840)
                k += 1
                Mv[k] += r11*(L**2*betay*intrhoz*r23*(4 - 5*alphay)/120 + L**2*betaz*intrhoy*r22*(4 - 5*alphaz)/120 + L*r21*(intrhoy2 + intrhoz2)/6) + r12*(L**2*betaz*intrhoy*r21*(5*alphaz - 4)/120 + L*betay*betaz*intrhoyz*r23*(-10*alphay*alphaz - 5*alphay - 5*alphaz + 2)/60 + L*betaz**2*r22*(-7*L**2*alphaz**2*intrho + 14*L**2*alphaz*intrho - 6*L**2*intrho + 140*alphaz**2*intrhoz2 + 140*alphaz*intrhoz2 - 28*intrhoz2)/840) + r13*(L**2*betay*intrhoz*r21*(5*alphay - 4)/120 + L*betay**2*r23*(-7*L**2*alphay**2*intrho + 14*L**2*alphay*intrho - 6*L**2*intrho + 140*alphay**2*intrhoy2 + 140*alphay*intrhoy2 - 28*intrhoy2)/840 + L*betay*betaz*intrhoyz*r22*(-10*alphay*alphaz - 5*alphay - 5*alphaz + 2)/60)
                k += 1
                Mv[k] += r21*(L**2*betay*intrhoz*r23*(4 - 5*alphay)/120 + L**2*betaz*intrhoy*r22*(4 - 5*alphaz)/120 + L*r21*(intrhoy2 + intrhoz2)/6) + r22*(L**2*betaz*intrhoy*r21*(5*alphaz - 4)/120 + L*betay*betaz*intrhoyz*r23*(-10*alphay*alphaz - 5*alphay - 5*alphaz + 2)/60 + L*betaz**2*r22*(-7*L**2*alphaz**2*intrho + 14*L**2*alphaz*intrho - 6*L**2*intrho + 140*alphaz**2*intrhoz2 + 140*alphaz*intrhoz2 - 28*intrhoz2)/840) + r23*(L**2*betay*intrhoz*r21*(5*alphay - 4)/120 + L*betay**2*r23*(-7*L**2*alphay**2*intrho + 14*L**2*alphay*intrho - 6*L**2*intrho + 140*alphay**2*intrhoy2 + 140*alphay*intrhoy2 - 28*intrhoy2)/840 + L*betay*betaz*intrhoyz*r22*(-10*alphay*alphaz - 5*alphay - 5*alphaz + 2)/60)
                k += 1
                Mv[k] += r31*(L**2*betay*intrhoz*r23*(4 - 5*alphay)/120 + L**2*betaz*intrhoy*r22*(4 - 5*alphaz)/120 + L*r21*(intrhoy2 + intrhoz2)/6) + r32*(L**2*betaz*intrhoy*r21*(5*alphaz - 4)/120 + L*betay*betaz*intrhoyz*r23*(-10*alphay*alphaz - 5*alphay - 5*alphaz + 2)/60 + L*betaz**2*r22*(-7*L**2*alphaz**2*intrho + 14*L**2*alphaz*intrho - 6*L**2*intrho + 140*alphaz**2*intrhoz2 + 140*alphaz*intrhoz2 - 28*intrhoz2)/840) + r33*(L**2*betay*intrhoz*r21*(5*alphay - 4)/120 + L*betay**2*r23*(-7*L**2*alphay**2*intrho + 14*L**2*alphay*intrho - 6*L**2*intrho + 140*alphay**2*intrhoy2 + 140*alphay*intrhoy2 - 28*intrhoy2)/840 + L*betay*betaz*intrhoyz*r22*(-10*alphay*alphaz - 5*alphay - 5*alphaz + 2)/60)
                k += 1
                Mv[k] += r11*(L*betay*intrhoy*r23*(4*alphay - 1)/12 + L*betaz*intrhoz*r22*(1 - 4*alphaz)/12) + r12*(L*betay*intrhoz*r21*(20*alphay - 21)/60 + betay**2*r23*(-35*L**2*alphay**2*intrho + 77*L**2*alphay*intrho - 44*L**2*intrho - 420*alphay*intrhoy2 - 84*intrhoy2)/840 + betay*betaz*intrhoyz*r22*(5*alphaz + 1)/10) + r13*(L*betaz*intrhoy*r21*(21 - 20*alphaz)/60 + betay*betaz*intrhoyz*r23*(-5*alphay - 1)/10 + betaz**2*r22*(35*L**2*alphaz**2*intrho - 77*L**2*alphaz*intrho + 44*L**2*intrho + 420*alphaz*intrhoz2 + 84*intrhoz2)/840)
                k += 1
                Mv[k] += r21*(L*betay*intrhoy*r23*(4*alphay - 1)/12 + L*betaz*intrhoz*r22*(1 - 4*alphaz)/12) + r22*(L*betay*intrhoz*r21*(20*alphay - 21)/60 + betay**2*r23*(-35*L**2*alphay**2*intrho + 77*L**2*alphay*intrho - 44*L**2*intrho - 420*alphay*intrhoy2 - 84*intrhoy2)/840 + betay*betaz*intrhoyz*r22*(5*alphaz + 1)/10) + r23*(L*betaz*intrhoy*r21*(21 - 20*alphaz)/60 + betay*betaz*intrhoyz*r23*(-5*alphay - 1)/10 + betaz**2*r22*(35*L**2*alphaz**2*intrho - 77*L**2*alphaz*intrho + 44*L**2*intrho + 420*alphaz*intrhoz2 + 84*intrhoz2)/840)
                k += 1
                Mv[k] += r31*(L*betay*intrhoy*r23*(4*alphay - 1)/12 + L*betaz*intrhoz*r22*(1 - 4*alphaz)/12) + r32*(L*betay*intrhoz*r21*(20*alphay - 21)/60 + betay**2*r23*(-35*L**2*alphay**2*intrho + 77*L**2*alphay*intrho - 44*L**2*intrho - 420*alphay*intrhoy2 - 84*intrhoy2)/840 + betay*betaz*intrhoyz*r22*(5*alphaz + 1)/10) + r33*(L*betaz*intrhoy*r21*(21 - 20*alphaz)/60 + betay*betaz*intrhoyz*r23*(-5*alphay - 1)/10 + betaz**2*r22*(35*L**2*alphaz**2*intrho - 77*L**2*alphaz*intrho + 44*L**2*intrho + 420*alphaz*intrhoz2 + 84*intrhoz2)/840)
                k += 1
                Mv[k] += r11*(L**2*betay*intrhoz*r23*(6 - 5*alphay)/120 + L**2*betaz*intrhoy*r22*(6 - 5*alphaz)/120 + L*r21*(intrhoy2 + intrhoz2)/3) + r12*(L**2*betaz*intrhoy*r21*(6 - 5*alphaz)/120 + L*betay*betaz*intrhoyz*r23*(-20*alphay*alphaz + 5*alphay + 5*alphaz - 8)/60 + L*betaz**2*r22*(7*L**2*alphaz**2*intrho - 14*L**2*alphaz*intrho + 8*L**2*intrho + 280*alphaz**2*intrhoz2 - 140*alphaz*intrhoz2 + 112*intrhoz2)/840) + r13*(L**2*betay*intrhoz*r21*(6 - 5*alphay)/120 + L*betay**2*r23*(7*L**2*alphay**2*intrho - 14*L**2*alphay*intrho + 8*L**2*intrho + 280*alphay**2*intrhoy2 - 140*alphay*intrhoy2 + 112*intrhoy2)/840 + L*betay*betaz*intrhoyz*r22*(-20*alphay*alphaz + 5*alphay + 5*alphaz - 8)/60)
                k += 1
                Mv[k] += r21*(L**2*betay*intrhoz*r23*(6 - 5*alphay)/120 + L**2*betaz*intrhoy*r22*(6 - 5*alphaz)/120 + L*r21*(intrhoy2 + intrhoz2)/3) + r22*(L**2*betaz*intrhoy*r21*(6 - 5*alphaz)/120 + L*betay*betaz*intrhoyz*r23*(-20*alphay*alphaz + 5*alphay + 5*alphaz - 8)/60 + L*betaz**2*r22*(7*L**2*alphaz**2*intrho - 14*L**2*alphaz*intrho + 8*L**2*intrho + 280*alphaz**2*intrhoz2 - 140*alphaz*intrhoz2 + 112*intrhoz2)/840) + r23*(L**2*betay*intrhoz*r21*(6 - 5*alphay)/120 + L*betay**2*r23*(7*L**2*alphay**2*intrho - 14*L**2*alphay*intrho + 8*L**2*intrho + 280*alphay**2*intrhoy2 - 140*alphay*intrhoy2 + 112*intrhoy2)/840 + L*betay*betaz*intrhoyz*r22*(-20*alphay*alphaz + 5*alphay + 5*alphaz - 8)/60)
                k += 1
                Mv[k] += r31*(L**2*betay*intrhoz*r23*(6 - 5*alphay)/120 + L**2*betaz*intrhoy*r22*(6 - 5*alphaz)/120 + L*r21*(intrhoy2 + intrhoz2)/3) + r32*(L**2*betaz*intrhoy*r21*(6 - 5*alphaz)/120 + L*betay*betaz*intrhoyz*r23*(-20*alphay*alphaz + 5*alphay + 5*alphaz - 8)/60 + L*betaz**2*r22*(7*L**2*alphaz**2*intrho - 14*L**2*alphaz*intrho + 8*L**2*intrho + 280*alphaz**2*intrhoz2 - 140*alphaz*intrhoz2 + 112*intrhoz2)/840) + r33*(L**2*betay*intrhoz*r21*(6 - 5*alphay)/120 + L*betay**2*r23*(7*L**2*alphay**2*intrho - 14*L**2*alphay*intrho + 8*L**2*intrho + 280*alphay**2*intrhoy2 - 140*alphay*intrhoy2 + 112*intrhoy2)/840 + L*betay*betaz*intrhoyz*r22*(-20*alphay*alphaz + 5*alphay + 5*alphaz - 8)/60)
                k += 1
                Mv[k] += r11*(L*betay*intrhoy*r33*(2*alphay + 1)/12 - L*betaz*intrhoz*r32*(2*alphaz + 1)/12) + r12*(L*betay*intrhoz*r31*(10*alphay - 9)/60 + betay**2*r33*(-35*L**2*alphay**2*intrho + 63*L**2*alphay*intrho - 26*L**2*intrho + 420*alphay*intrhoy2 + 84*intrhoy2)/840 + betay*betaz*intrhoyz*r32*(-5*alphaz - 1)/10) + r13*(L*betaz*intrhoy*r31*(9 - 10*alphaz)/60 + betay*betaz*intrhoyz*r33*(5*alphay + 1)/10 + betaz**2*r32*(35*L**2*alphaz**2*intrho - 63*L**2*alphaz*intrho + 26*L**2*intrho - 420*alphaz*intrhoz2 - 84*intrhoz2)/840)
                k += 1
                Mv[k] += r21*(L*betay*intrhoy*r33*(2*alphay + 1)/12 - L*betaz*intrhoz*r32*(2*alphaz + 1)/12) + r22*(L*betay*intrhoz*r31*(10*alphay - 9)/60 + betay**2*r33*(-35*L**2*alphay**2*intrho + 63*L**2*alphay*intrho - 26*L**2*intrho + 420*alphay*intrhoy2 + 84*intrhoy2)/840 + betay*betaz*intrhoyz*r32*(-5*alphaz - 1)/10) + r23*(L*betaz*intrhoy*r31*(9 - 10*alphaz)/60 + betay*betaz*intrhoyz*r33*(5*alphay + 1)/10 + betaz**2*r32*(35*L**2*alphaz**2*intrho - 63*L**2*alphaz*intrho + 26*L**2*intrho - 420*alphaz*intrhoz2 - 84*intrhoz2)/840)
                k += 1
                Mv[k] += r31*(L*betay*intrhoy*r33*(2*alphay + 1)/12 - L*betaz*intrhoz*r32*(2*alphaz + 1)/12) + r32*(L*betay*intrhoz*r31*(10*alphay - 9)/60 + betay**2*r33*(-35*L**2*alphay**2*intrho + 63*L**2*alphay*intrho - 26*L**2*intrho + 420*alphay*intrhoy2 + 84*intrhoy2)/840 + betay*betaz*intrhoyz*r32*(-5*alphaz - 1)/10) + r33*(L*betaz*intrhoy*r31*(9 - 10*alphaz)/60 + betay*betaz*intrhoyz*r33*(5*alphay + 1)/10 + betaz**2*r32*(35*L**2*alphaz**2*intrho - 63*L**2*alphaz*intrho + 26*L**2*intrho - 420*alphaz*intrhoz2 - 84*intrhoz2)/840)
                k += 1
                Mv[k] += r11*(L**2*betay*intrhoz*r33*(4 - 5*alphay)/120 + L**2*betaz*intrhoy*r32*(4 - 5*alphaz)/120 + L*r31*(intrhoy2 + intrhoz2)/6) + r12*(L**2*betaz*intrhoy*r31*(5*alphaz - 4)/120 + L*betay*betaz*intrhoyz*r33*(-10*alphay*alphaz - 5*alphay - 5*alphaz + 2)/60 + L*betaz**2*r32*(-7*L**2*alphaz**2*intrho + 14*L**2*alphaz*intrho - 6*L**2*intrho + 140*alphaz**2*intrhoz2 + 140*alphaz*intrhoz2 - 28*intrhoz2)/840) + r13*(L**2*betay*intrhoz*r31*(5*alphay - 4)/120 + L*betay**2*r33*(-7*L**2*alphay**2*intrho + 14*L**2*alphay*intrho - 6*L**2*intrho + 140*alphay**2*intrhoy2 + 140*alphay*intrhoy2 - 28*intrhoy2)/840 + L*betay*betaz*intrhoyz*r32*(-10*alphay*alphaz - 5*alphay - 5*alphaz + 2)/60)
                k += 1
                Mv[k] += r21*(L**2*betay*intrhoz*r33*(4 - 5*alphay)/120 + L**2*betaz*intrhoy*r32*(4 - 5*alphaz)/120 + L*r31*(intrhoy2 + intrhoz2)/6) + r22*(L**2*betaz*intrhoy*r31*(5*alphaz - 4)/120 + L*betay*betaz*intrhoyz*r33*(-10*alphay*alphaz - 5*alphay - 5*alphaz + 2)/60 + L*betaz**2*r32*(-7*L**2*alphaz**2*intrho + 14*L**2*alphaz*intrho - 6*L**2*intrho + 140*alphaz**2*intrhoz2 + 140*alphaz*intrhoz2 - 28*intrhoz2)/840) + r23*(L**2*betay*intrhoz*r31*(5*alphay - 4)/120 + L*betay**2*r33*(-7*L**2*alphay**2*intrho + 14*L**2*alphay*intrho - 6*L**2*intrho + 140*alphay**2*intrhoy2 + 140*alphay*intrhoy2 - 28*intrhoy2)/840 + L*betay*betaz*intrhoyz*r32*(-10*alphay*alphaz - 5*alphay - 5*alphaz + 2)/60)
                k += 1
                Mv[k] += r31*(L**2*betay*intrhoz*r33*(4 - 5*alphay)/120 + L**2*betaz*intrhoy*r32*(4 - 5*alphaz)/120 + L*r31*(intrhoy2 + intrhoz2)/6) + r32*(L**2*betaz*intrhoy*r31*(5*alphaz - 4)/120 + L*betay*betaz*intrhoyz*r33*(-10*alphay*alphaz - 5*alphay - 5*alphaz + 2)/60 + L*betaz**2*r32*(-7*L**2*alphaz**2*intrho + 14*L**2*alphaz*intrho - 6*L**2*intrho + 140*alphaz**2*intrhoz2 + 140*alphaz*intrhoz2 - 28*intrhoz2)/840) + r33*(L**2*betay*intrhoz*r31*(5*alphay - 4)/120 + L*betay**2*r33*(-7*L**2*alphay**2*intrho + 14*L**2*alphay*intrho - 6*L**2*intrho + 140*alphay**2*intrhoy2 + 140*alphay*intrhoy2 - 28*intrhoy2)/840 + L*betay*betaz*intrhoyz*r32*(-10*alphay*alphaz - 5*alphay - 5*alphaz + 2)/60)
                k += 1
                Mv[k] += r11*(L*betay*intrhoy*r33*(4*alphay - 1)/12 + L*betaz*intrhoz*r32*(1 - 4*alphaz)/12) + r12*(L*betay*intrhoz*r31*(20*alphay - 21)/60 + betay**2*r33*(-35*L**2*alphay**2*intrho + 77*L**2*alphay*intrho - 44*L**2*intrho - 420*alphay*intrhoy2 - 84*intrhoy2)/840 + betay*betaz*intrhoyz*r32*(5*alphaz + 1)/10) + r13*(L*betaz*intrhoy*r31*(21 - 20*alphaz)/60 + betay*betaz*intrhoyz*r33*(-5*alphay - 1)/10 + betaz**2*r32*(35*L**2*alphaz**2*intrho - 77*L**2*alphaz*intrho + 44*L**2*intrho + 420*alphaz*intrhoz2 + 84*intrhoz2)/840)
                k += 1
                Mv[k] += r21*(L*betay*intrhoy*r33*(4*alphay - 1)/12 + L*betaz*intrhoz*r32*(1 - 4*alphaz)/12) + r22*(L*betay*intrhoz*r31*(20*alphay - 21)/60 + betay**2*r33*(-35*L**2*alphay**2*intrho + 77*L**2*alphay*intrho - 44*L**2*intrho - 420*alphay*intrhoy2 - 84*intrhoy2)/840 + betay*betaz*intrhoyz*r32*(5*alphaz + 1)/10) + r23*(L*betaz*intrhoy*r31*(21 - 20*alphaz)/60 + betay*betaz*intrhoyz*r33*(-5*alphay - 1)/10 + betaz**2*r32*(35*L**2*alphaz**2*intrho - 77*L**2*alphaz*intrho + 44*L**2*intrho + 420*alphaz*intrhoz2 + 84*intrhoz2)/840)
                k += 1
                Mv[k] += r31*(L*betay*intrhoy*r33*(4*alphay - 1)/12 + L*betaz*intrhoz*r32*(1 - 4*alphaz)/12) + r32*(L*betay*intrhoz*r31*(20*alphay - 21)/60 + betay**2*r33*(-35*L**2*alphay**2*intrho + 77*L**2*alphay*intrho - 44*L**2*intrho - 420*alphay*intrhoy2 - 84*intrhoy2)/840 + betay*betaz*intrhoyz*r32*(5*alphaz + 1)/10) + r33*(L*betaz*intrhoy*r31*(21 - 20*alphaz)/60 + betay*betaz*intrhoyz*r33*(-5*alphay - 1)/10 + betaz**2*r32*(35*L**2*alphaz**2*intrho - 77*L**2*alphaz*intrho + 44*L**2*intrho + 420*alphaz*intrhoz2 + 84*intrhoz2)/840)
                k += 1
                Mv[k] += r11*(L**2*betay*intrhoz*r33*(6 - 5*alphay)/120 + L**2*betaz*intrhoy*r32*(6 - 5*alphaz)/120 + L*r31*(intrhoy2 + intrhoz2)/3) + r12*(L**2*betaz*intrhoy*r31*(6 - 5*alphaz)/120 + L*betay*betaz*intrhoyz*r33*(-20*alphay*alphaz + 5*alphay + 5*alphaz - 8)/60 + L*betaz**2*r32*(7*L**2*alphaz**2*intrho - 14*L**2*alphaz*intrho + 8*L**2*intrho + 280*alphaz**2*intrhoz2 - 140*alphaz*intrhoz2 + 112*intrhoz2)/840) + r13*(L**2*betay*intrhoz*r31*(6 - 5*alphay)/120 + L*betay**2*r33*(7*L**2*alphay**2*intrho - 14*L**2*alphay*intrho + 8*L**2*intrho + 280*alphay**2*intrhoy2 - 140*alphay*intrhoy2 + 112*intrhoy2)/840 + L*betay*betaz*intrhoyz*r32*(-20*alphay*alphaz + 5*alphay + 5*alphaz - 8)/60)
                k += 1
                Mv[k] += r21*(L**2*betay*intrhoz*r33*(6 - 5*alphay)/120 + L**2*betaz*intrhoy*r32*(6 - 5*alphaz)/120 + L*r31*(intrhoy2 + intrhoz2)/3) + r22*(L**2*betaz*intrhoy*r31*(6 - 5*alphaz)/120 + L*betay*betaz*intrhoyz*r33*(-20*alphay*alphaz + 5*alphay + 5*alphaz - 8)/60 + L*betaz**2*r32*(7*L**2*alphaz**2*intrho - 14*L**2*alphaz*intrho + 8*L**2*intrho + 280*alphaz**2*intrhoz2 - 140*alphaz*intrhoz2 + 112*intrhoz2)/840) + r23*(L**2*betay*intrhoz*r31*(6 - 5*alphay)/120 + L*betay**2*r33*(7*L**2*alphay**2*intrho - 14*L**2*alphay*intrho + 8*L**2*intrho + 280*alphay**2*intrhoy2 - 140*alphay*intrhoy2 + 112*intrhoy2)/840 + L*betay*betaz*intrhoyz*r32*(-20*alphay*alphaz + 5*alphay + 5*alphaz - 8)/60)
                k += 1
                Mv[k] += r31*(L**2*betay*intrhoz*r33*(6 - 5*alphay)/120 + L**2*betaz*intrhoy*r32*(6 - 5*alphaz)/120 + L*r31*(intrhoy2 + intrhoz2)/3) + r32*(L**2*betaz*intrhoy*r31*(6 - 5*alphaz)/120 + L*betay*betaz*intrhoyz*r33*(-20*alphay*alphaz + 5*alphay + 5*alphaz - 8)/60 + L*betaz**2*r32*(7*L**2*alphaz**2*intrho - 14*L**2*alphaz*intrho + 8*L**2*intrho + 280*alphaz**2*intrhoz2 - 140*alphaz*intrhoz2 + 112*intrhoz2)/840) + r33*(L**2*betay*intrhoz*r31*(6 - 5*alphay)/120 + L*betay**2*r33*(7*L**2*alphay**2*intrho - 14*L**2*alphay*intrho + 8*L**2*intrho + 280*alphay**2*intrhoy2 - 140*alphay*intrhoy2 + 112*intrhoy2)/840 + L*betay*betaz*intrhoyz*r32*(-20*alphay*alphaz + 5*alphay + 5*alphaz - 8)/60)

            elif mtype == 1: # M_lump lumped mass matrix using two-point Gauss-Lobatto quadrature
                k = self.init_k_M
                Mr[k] = 0+c1
                Mc[k] = 0+c1
                k += 1
                Mr[k] = 0+c1
                Mc[k] = 1+c1
                k += 1
                Mr[k] = 0+c1
                Mc[k] = 2+c1
                k += 1
                Mr[k] = 1+c1
                Mc[k] = 0+c1
                k += 1
                Mr[k] = 1+c1
                Mc[k] = 1+c1
                k += 1
                Mr[k] = 1+c1
                Mc[k] = 2+c1
                k += 1
                Mr[k] = 2+c1
                Mc[k] = 0+c1
                k += 1
                Mr[k] = 2+c1
                Mc[k] = 1+c1
                k += 1
                Mr[k] = 2+c1
                Mc[k] = 2+c1
                k += 1
                Mr[k] = 3+c1
                Mc[k] = 3+c1
                k += 1
                Mr[k] = 3+c1
                Mc[k] = 4+c1
                k += 1
                Mr[k] = 3+c1
                Mc[k] = 5+c1
                k += 1
                Mr[k] = 4+c1
                Mc[k] = 3+c1
                k += 1
                Mr[k] = 4+c1
                Mc[k] = 4+c1
                k += 1
                Mr[k] = 4+c1
                Mc[k] = 5+c1
                k += 1
                Mr[k] = 5+c1
                Mc[k] = 3+c1
                k += 1
                Mr[k] = 5+c1
                Mc[k] = 4+c1
                k += 1
                Mr[k] = 5+c1
                Mc[k] = 5+c1
                k += 1
                Mr[k] = 0+c2
                Mc[k] = 0+c2
                k += 1
                Mr[k] = 0+c2
                Mc[k] = 1+c2
                k += 1
                Mr[k] = 0+c2
                Mc[k] = 2+c2
                k += 1
                Mr[k] = 1+c2
                Mc[k] = 0+c2
                k += 1
                Mr[k] = 1+c2
                Mc[k] = 1+c2
                k += 1
                Mr[k] = 1+c2
                Mc[k] = 2+c2
                k += 1
                Mr[k] = 2+c2
                Mc[k] = 0+c2
                k += 1
                Mr[k] = 2+c2
                Mc[k] = 1+c2
                k += 1
                Mr[k] = 2+c2
                Mc[k] = 2+c2
                k += 1
                Mr[k] = 3+c2
                Mc[k] = 3+c2
                k += 1
                Mr[k] = 3+c2
                Mc[k] = 4+c2
                k += 1
                Mr[k] = 3+c2
                Mc[k] = 5+c2
                k += 1
                Mr[k] = 4+c2
                Mc[k] = 3+c2
                k += 1
                Mr[k] = 4+c2
                Mc[k] = 4+c2
                k += 1
                Mr[k] = 4+c2
                Mc[k] = 5+c2
                k += 1
                Mr[k] = 5+c2
                Mc[k] = 3+c2
                k += 1
                Mr[k] = 5+c2
                Mc[k] = 4+c2
                k += 1
                Mr[k] = 5+c2
                Mc[k] = 5+c2

                # NOTE two-point Gauss-Lobatto quadrature

                k = self.init_k_M
                Mv[k] += L*betay**2*intrho*r12**2*(alphay**2 - 2*alphay + 1)/2 + L*betaz**2*intrho*r13**2*(alphaz**2 - 2*alphaz + 1)/2 + L*intrho*r11**2/2
                k += 1
                Mv[k] += L*betay**2*intrho*r12*r22*(alphay**2 - 2*alphay + 1)/2 + L*betaz**2*intrho*r13*r23*(alphaz**2 - 2*alphaz + 1)/2 + L*intrho*r11*r21/2
                k += 1
                Mv[k] += L*betay**2*intrho*r12*r32*(alphay**2 - 2*alphay + 1)/2 + L*betaz**2*intrho*r13*r33*(alphaz**2 - 2*alphaz + 1)/2 + L*intrho*r11*r31/2
                k += 1
                Mv[k] += L*betay**2*intrho*r12*r22*(alphay**2 - 2*alphay + 1)/2 + L*betaz**2*intrho*r13*r23*(alphaz**2 - 2*alphaz + 1)/2 + L*intrho*r11*r21/2
                k += 1
                Mv[k] += L*betay**2*intrho*r22**2*(alphay**2 - 2*alphay + 1)/2 + L*betaz**2*intrho*r23**2*(alphaz**2 - 2*alphaz + 1)/2 + L*intrho*r21**2/2
                k += 1
                Mv[k] += L*betay**2*intrho*r22*r32*(alphay**2 - 2*alphay + 1)/2 + L*betaz**2*intrho*r23*r33*(alphaz**2 - 2*alphaz + 1)/2 + L*intrho*r21*r31/2
                k += 1
                Mv[k] += L*betay**2*intrho*r12*r32*(alphay**2 - 2*alphay + 1)/2 + L*betaz**2*intrho*r13*r33*(alphaz**2 - 2*alphaz + 1)/2 + L*intrho*r11*r31/2
                k += 1
                Mv[k] += L*betay**2*intrho*r22*r32*(alphay**2 - 2*alphay + 1)/2 + L*betaz**2*intrho*r23*r33*(alphaz**2 - 2*alphaz + 1)/2 + L*intrho*r21*r31/2
                k += 1
                Mv[k] += L*betay**2*intrho*r32**2*(alphay**2 - 2*alphay + 1)/2 + L*betaz**2*intrho*r33**2*(alphaz**2 - 2*alphaz + 1)/2 + L*intrho*r31**2/2
                k += 1
                Mv[k] += L*betay**2*intrhoy2*r13**2*(alphay**2 - 2*alphay + 1)/2 + L*betaz**2*intrhoz2*r12**2*(alphaz**2 - 2*alphaz + 1)/2 + L*r11**2*(intrhoy2 + intrhoz2)/2
                k += 1
                Mv[k] += L*betay**2*intrhoy2*r13*r23*(alphay**2 - 2*alphay + 1)/2 + L*betaz**2*intrhoz2*r12*r22*(alphaz**2 - 2*alphaz + 1)/2 + L*r11*r21*(intrhoy2 + intrhoz2)/2
                k += 1
                Mv[k] += L*betay**2*intrhoy2*r13*r33*(alphay**2 - 2*alphay + 1)/2 + L*betaz**2*intrhoz2*r12*r32*(alphaz**2 - 2*alphaz + 1)/2 + L*r11*r31*(intrhoy2 + intrhoz2)/2
                k += 1
                Mv[k] += L*betay**2*intrhoy2*r13*r23*(alphay**2 - 2*alphay + 1)/2 + L*betaz**2*intrhoz2*r12*r22*(alphaz**2 - 2*alphaz + 1)/2 + L*r11*r21*(intrhoy2 + intrhoz2)/2
                k += 1
                Mv[k] += L*betay**2*intrhoy2*r23**2*(alphay**2 - 2*alphay + 1)/2 + L*betaz**2*intrhoz2*r22**2*(alphaz**2 - 2*alphaz + 1)/2 + L*r21**2*(intrhoy2 + intrhoz2)/2
                k += 1
                Mv[k] += L*betay**2*intrhoy2*r23*r33*(alphay**2 - 2*alphay + 1)/2 + L*betaz**2*intrhoz2*r22*r32*(alphaz**2 - 2*alphaz + 1)/2 + L*r21*r31*(intrhoy2 + intrhoz2)/2
                k += 1
                Mv[k] += L*betay**2*intrhoy2*r13*r33*(alphay**2 - 2*alphay + 1)/2 + L*betaz**2*intrhoz2*r12*r32*(alphaz**2 - 2*alphaz + 1)/2 + L*r11*r31*(intrhoy2 + intrhoz2)/2
                k += 1
                Mv[k] += L*betay**2*intrhoy2*r23*r33*(alphay**2 - 2*alphay + 1)/2 + L*betaz**2*intrhoz2*r22*r32*(alphaz**2 - 2*alphaz + 1)/2 + L*r21*r31*(intrhoy2 + intrhoz2)/2
                k += 1
                Mv[k] += L*betay**2*intrhoy2*r33**2*(alphay**2 - 2*alphay + 1)/2 + L*betaz**2*intrhoz2*r32**2*(alphaz**2 - 2*alphaz + 1)/2 + L*r31**2*(intrhoy2 + intrhoz2)/2
                k += 1
                Mv[k] += L*betay**2*intrho*r12**2*(alphay**2 - 2*alphay + 1)/2 + L*betaz**2*intrho*r13**2*(alphaz**2 - 2*alphaz + 1)/2 + L*intrho*r11**2/2
                k += 1
                Mv[k] += L*betay**2*intrho*r12*r22*(alphay**2 - 2*alphay + 1)/2 + L*betaz**2*intrho*r13*r23*(alphaz**2 - 2*alphaz + 1)/2 + L*intrho*r11*r21/2
                k += 1
                Mv[k] += L*betay**2*intrho*r12*r32*(alphay**2 - 2*alphay + 1)/2 + L*betaz**2*intrho*r13*r33*(alphaz**2 - 2*alphaz + 1)/2 + L*intrho*r11*r31/2
                k += 1
                Mv[k] += L*betay**2*intrho*r12*r22*(alphay**2 - 2*alphay + 1)/2 + L*betaz**2*intrho*r13*r23*(alphaz**2 - 2*alphaz + 1)/2 + L*intrho*r11*r21/2
                k += 1
                Mv[k] += L*betay**2*intrho*r22**2*(alphay**2 - 2*alphay + 1)/2 + L*betaz**2*intrho*r23**2*(alphaz**2 - 2*alphaz + 1)/2 + L*intrho*r21**2/2
                k += 1
                Mv[k] += L*betay**2*intrho*r22*r32*(alphay**2 - 2*alphay + 1)/2 + L*betaz**2*intrho*r23*r33*(alphaz**2 - 2*alphaz + 1)/2 + L*intrho*r21*r31/2
                k += 1
                Mv[k] += L*betay**2*intrho*r12*r32*(alphay**2 - 2*alphay + 1)/2 + L*betaz**2*intrho*r13*r33*(alphaz**2 - 2*alphaz + 1)/2 + L*intrho*r11*r31/2
                k += 1
                Mv[k] += L*betay**2*intrho*r22*r32*(alphay**2 - 2*alphay + 1)/2 + L*betaz**2*intrho*r23*r33*(alphaz**2 - 2*alphaz + 1)/2 + L*intrho*r21*r31/2
                k += 1
                Mv[k] += L*betay**2*intrho*r32**2*(alphay**2 - 2*alphay + 1)/2 + L*betaz**2*intrho*r33**2*(alphaz**2 - 2*alphaz + 1)/2 + L*intrho*r31**2/2
                k += 1
                Mv[k] += L*betay**2*intrhoy2*r13**2*(alphay**2 - 2*alphay + 1)/2 + L*betaz**2*intrhoz2*r12**2*(alphaz**2 - 2*alphaz + 1)/2 + L*r11**2*(intrhoy2 + intrhoz2)/2
                k += 1
                Mv[k] += L*betay**2*intrhoy2*r13*r23*(alphay**2 - 2*alphay + 1)/2 + L*betaz**2*intrhoz2*r12*r22*(alphaz**2 - 2*alphaz + 1)/2 + L*r11*r21*(intrhoy2 + intrhoz2)/2
                k += 1
                Mv[k] += L*betay**2*intrhoy2*r13*r33*(alphay**2 - 2*alphay + 1)/2 + L*betaz**2*intrhoz2*r12*r32*(alphaz**2 - 2*alphaz + 1)/2 + L*r11*r31*(intrhoy2 + intrhoz2)/2
                k += 1
                Mv[k] += L*betay**2*intrhoy2*r13*r23*(alphay**2 - 2*alphay + 1)/2 + L*betaz**2*intrhoz2*r12*r22*(alphaz**2 - 2*alphaz + 1)/2 + L*r11*r21*(intrhoy2 + intrhoz2)/2
                k += 1
                Mv[k] += L*betay**2*intrhoy2*r23**2*(alphay**2 - 2*alphay + 1)/2 + L*betaz**2*intrhoz2*r22**2*(alphaz**2 - 2*alphaz + 1)/2 + L*r21**2*(intrhoy2 + intrhoz2)/2
                k += 1
                Mv[k] += L*betay**2*intrhoy2*r23*r33*(alphay**2 - 2*alphay + 1)/2 + L*betaz**2*intrhoz2*r22*r32*(alphaz**2 - 2*alphaz + 1)/2 + L*r21*r31*(intrhoy2 + intrhoz2)/2
                k += 1
                Mv[k] += L*betay**2*intrhoy2*r13*r33*(alphay**2 - 2*alphay + 1)/2 + L*betaz**2*intrhoz2*r12*r32*(alphaz**2 - 2*alphaz + 1)/2 + L*r11*r31*(intrhoy2 + intrhoz2)/2
                k += 1
                Mv[k] += L*betay**2*intrhoy2*r23*r33*(alphay**2 - 2*alphay + 1)/2 + L*betaz**2*intrhoz2*r22*r32*(alphaz**2 - 2*alphaz + 1)/2 + L*r21*r31*(intrhoy2 + intrhoz2)/2
                k += 1
                Mv[k] += L*betay**2*intrhoy2*r33**2*(alphay**2 - 2*alphay + 1)/2 + L*betaz**2*intrhoz2*r32**2*(alphaz**2 - 2*alphaz + 1)/2 + L*r31**2*(intrhoy2 + intrhoz2)/2
