#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: nonecheck=False
#cython: overflowcheck=False
#cython: initializedcheck=False
#cython: embedsignature=True
#cython: infer_types=False
r"""
==========================================================================================
BeamLR - Linear Timoshenko 3D beam element with reduced integration (:mod:`pyfe3d.beamlr`)
==========================================================================================

.. currentmodule:: pyfe3d.beamlr

Defines the two-node beam element :class:`.BeamLR`, with six
degrees-of-freedom per node, its probe :class:`.BeamLRProbe` and
:class:`.BeamLRData`, which gives the number of sparse entries per element
of each structural matrix. The cross-section and material properties are
given by a :class:`.BeamProp` object.

"""
import numpy as np

from .beamprop cimport BeamProp

cdef int DOF = 6
cdef int NUM_NODES = 2


cdef class BeamLRData:
    r"""
    Used to allocate memory for the sparse matrices.

    Each attribute is the number of entries that one :class:`.BeamLR`
    element writes in the arrays of rows, columns and values of a sparse
    matrix, at the consecutive positions that start at its ``init_k_KC0``,
    ``init_k_KCNL``, ``init_k_KG`` or ``init_k_M``. The constructor takes no
    argument and sets the values below.

    Attributes
    ----------
    KC0_SPARSE_SIZE, : int
        ``KC0_SPARSE_SIZE = 144``, written by :meth:`.BeamLR.update_KC0`.

    KCNL_SPARSE_SIZE, : int
        ``KCNL_SPARSE_SIZE = 144``, written by :meth:`.BeamLR.update_KCNL`.

    KG_SPARSE_SIZE, : int
        ``KG_SPARSE_SIZE = 36``, written by :meth:`.BeamLR.update_KG`.

    M_SPARSE_SIZE, : int
        ``M_SPARSE_SIZE = 144``, written by :meth:`.BeamLR.update_M` with
        ``mtype=0``; with ``mtype=1`` only the first ``36`` are written.

    """
    cdef public int KC0_SPARSE_SIZE
    cdef public int KCNL_SPARSE_SIZE
    cdef public int KG_SPARSE_SIZE
    cdef public int M_SPARSE_SIZE

    def __cinit__(BeamLRData self):
        self.KC0_SPARSE_SIZE = 144
        self.KCNL_SPARSE_SIZE = 144
        self.KG_SPARSE_SIZE = 36
        self.M_SPARSE_SIZE = 144


cdef class BeamLRProbe:
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
        Updated by :meth:`.BeamLR.update_probe_xe`.
    ue, : array-like
        Array of size ``NUM_NODES*DOF=12`` containing the element displacements
        in the following order `u_{e_1}, v_{e_1}, w_{e_1}, {r_x}_{e_1},
        {r_y}_{e_1}, {r_z}_{e_1}, u_{e_2}, v_{e_2}, w_{e_2}, {r_x}_{e_2},
        {r_y}_{e_2}, {r_z}_{e_2}`. Translations of dimension length and
        rotations. Updated by :meth:`.BeamLR.update_probe_ue`.
    finte, : array-like
        Array of size ``NUM_NODES*DOF=12`` containing the element internal
        forces corresponding to the degrees-of-freedom described by ``ue``,
        forces and moments. Updated by :meth:`.BeamLR.update_probe_finte`.
    BLexx, BLky, BLkz : array-like
        Arrays of size ``NUM_NODES*DOF=12`` with the rows of the linear
        strain-displacement matrix for the axial strain and the two
        curvatures, at the last evaluated integration point.
    BLgammay, BLgammaz, BLkx : array-like
        Arrays of size ``NUM_NODES*DOF=12`` with the rows of the linear
        strain-displacement matrix for the two transverse shear strains and
        the twist, at the last evaluated integration point, read with the
        three rows above for the axial force of a sectional stiffness matrix
        with couplings, see :meth:`.BeamLR._update_probe_BL_G`.
    Sv_x, Sw_x : array-like
        Arrays of size ``NUM_NODES*DOF=12`` with the rows giving the slopes
        `v_{,x}` and `w_{,x}` of the linear interpolations of `v` and `w`,
        constant along the element, as in :class:`.BeamCProbe`. Up to
        version 0.11 they held the interpolations of the rotations, `r_z`
        and `-r_y`, see the notes of :class:`.BeamLR`.
    KCNLve : array-like
        Array of size ``(NUM_NODES*DOF)**2=144`` with the nonlinear
        constitutive stiffness matrix KCNL in element coordinates, stored row
        by row.

    """
    cdef public double [::1] xe
    cdef public double [::1] ue
    cdef public double [::1] finte
    cdef public double [::1] BLexx
    cdef public double [::1] BLky
    cdef public double [::1] BLkz
    cdef public double [::1] BLgammay
    cdef public double [::1] BLgammaz
    cdef public double [::1] BLkx
    cdef public double [::1] Sv_x
    cdef public double [::1] Sw_x
    cdef public double [::1] KCNLve

    def __cinit__(BeamLRProbe self):
        self.xe = np.zeros(NUM_NODES*DOF//2, dtype=np.float64)
        self.ue = np.zeros(NUM_NODES*DOF, dtype=np.float64)
        self.finte = np.zeros(NUM_NODES*DOF, dtype=np.float64)
        self.BLexx = np.zeros(NUM_NODES*DOF, dtype=np.float64)
        self.BLky = np.zeros(NUM_NODES*DOF, dtype=np.float64)
        self.BLkz = np.zeros(NUM_NODES*DOF, dtype=np.float64)
        self.BLgammay = np.zeros(NUM_NODES*DOF, dtype=np.float64)
        self.BLgammaz = np.zeros(NUM_NODES*DOF, dtype=np.float64)
        self.BLkx = np.zeros(NUM_NODES*DOF, dtype=np.float64)
        self.Sv_x = np.zeros(NUM_NODES*DOF, dtype=np.float64)
        self.Sw_x = np.zeros(NUM_NODES*DOF, dtype=np.float64)
        self.KCNLve = np.zeros((NUM_NODES*DOF)**2, dtype=np.float64)


cdef class BeamLR:
    r"""
    Timoshenko 3D beam element with linear shape functions

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
    p : :class:`.BeamLRProbe` object
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
        :class:`.BeamLRData` for their number.
    probe : :class:`.BeamLRProbe` object
        Pointer to the probe.

    Notes
    -----
    Formulation based on reference [1]_, replacing the consistent shape
    functions by linear functions and performing numerical integration with
    just 1 point at the beam center.

    The displacements of the cross section, the strains and the linear
    strain energy are those given in the notes of :class:`.BeamC`, with the
    sectional stiffness matrix ``C`` of :class:`.BeamProp`, which is built
    from the transverse shear moduli ``G12`` and ``G13`` of the planes `xy`
    and `xz`, the torsional stiffness ``GJ`` and the other attributes when
    it is not set. The six displacements `u, v, w, r_x, r_y, r_z` of the
    beam axis are interpolated linearly between the two nodes, such that
    the axial strain `u_{,x}`, the curvatures `\kappa_y = -{r_z}_{,x}`,
    `\kappa_z = {r_y}_{,x}` and the twist `{r_x}_{,x}` are constant along
    the element, while the transverse shear strains `\gamma_{xy} = v_{,x} -
    r_z` and `\gamma_{xz} = w_{,x} + r_y` are linear. The strain energy is
    evaluated at the center of the element, i.e. with a reduced one-point
    integration, `\boldsymbol{K}_{C0} = L \boldsymbol{B}_L^\top
    \boldsymbol{C} \boldsymbol{B}_L` with the strain-displacement matrix
    `\boldsymbol{B}_L` at the center, which is exact for the terms of the
    constant strains and avoids the shear locking of the linear
    interpolation in slender elements. The interpolation does not depend on
    ``C``, and every coupling of ``C``, e.g. the bending-shear couplings of
    a composite section, is represented with the same convergence as the
    square of the element length as an uncoupled section, see the notes of
    :class:`.BeamC` and ``tests/test_beam_sectional_stiffness.py``. Since
    the curvature is constant in each element, more elements are needed
    than with :class:`.BeamC` for the same accuracy in bending.

    The element does not use the shear parameters `\alpha_y`, `\alpha_z` of
    :class:`.BeamC`, and is therefore an independent check of their sign:
    in ``tests/test_timoshenko_beam_literature.py`` refined meshes of
    :class:`.BeamLR` converge to the same cantilever tip deflection `P
    L^3/(3 E I_{zz}) + P L/(G A)`, Eq. (1.2) of [2]_, as one element of
    :class:`.BeamC`, and to the same exact frequencies of a deep cantilever,
    Table 3.6 of [3]_; in ``tests/test_beam_shear_moduli.py`` they do so in
    each plane with its own modulus when ``G12`` and ``G13`` differ.

    The geometric stiffness matrix of :meth:`.update_KG`, the nonlinear
    matrix of :meth:`.update_KCNL` and the nonlinear internal forces of
    :meth:`.update_probe_finte` come from the von Kármán axial strain
    `\epsilon_{xx} = u_{,x} + (v_{,x}^2 + w_{,x}^2)/2`, whose slopes
    `v_{,x}` and `w_{,x}` are taken, since version 0.12.0, from the linear
    interpolations of `v` and `w`, as the axial strain requires and as in
    :class:`.BeamC`. They are constant along the element, and the geometric
    stiffness matrix is that of a string, `N/L` times ``[[1, -1], [-1,
    1]]`` for `v` and for `w`. Up to version 0.11 the slopes were taken from
    the rotations, `v_{,x} = r_z` and `w_{,x} = -r_y`, which holds only
    without shear deformation: the buckling mode could then deflect with
    vanishing shear strains, and the buckling load converged to the Euler
    load `P_e` whatever the shear stiffness `\kappa G A`, instead of `P_e/(1
    + P_e/(\kappa G A))`, Eq. (2-57) of [4]_. For a clamped-free column with
    `P_e/(\kappa G A)` = 0.0002, 0.018, 0.18 and 0.59, `P_{cr}/P_e` was
    1.0000 with 100 and with 800 elements, against 0.9998, 0.9825, 0.8488
    and 0.6274 from Eq. (2-57), i.e. 0.02 % to 59 % unconservative. It is
    now 0.9998, 0.9825, 0.8488 and 0.6274 with 160 elements, converging to
    Eq. (2-57) as the square of the element length, see
    ``tests/test_beam_linear_buckling_3D_column.py``. For slender columns
    the discretization error is about twice that of the former matrix: for
    `P_e/(\kappa G A) = 0.0002`, the load is 4.1e-3 and 1.0e-3 above Eq.
    (2-57) with 10 and 20 elements, where it was 2.1e-3 and 5.1e-4 above
    `P_e`. With constant slopes the axial strain `u_{,x} + w_{,x}^2/2` can
    vanish in each element, so BeamLR is free of membrane locking; BeamC
    avoids it with the element-averaged axial strain described in its
    notes, see ``tests/test_nonlinear_benchmarks_sze2004.py``.

    The matrices and the internal forces are computed in element
    coordinates and transformed to global coordinates with the rotation
    matrix of the element, as described in the notes of :class:`.BeamC`.

    References
    ----------
    .. [1] Luo, Y., 2008, “An Efficient 3D Timoshenko Beam Element with
       Consistent Shape Functions,” Adv. Theor. Appl. Mech., 1(3),
       pp. 95–106.
    .. [2] Bathe, K. J., 2014, Finite Element Procedures, 2nd ed., K. J.
       Bathe, Watertown, MA, ISBN 978-0-9790049-5-7, Section 1.2, p. 5,
       Eq. (1.2), Fig. 1.2(b).
    .. [3] Petyt, M., 2010, Introduction to Finite Element Vibration
       Analysis, 2nd ed., Cambridge University Press, Section 3.9.
       https://doi.org/10.1017/CBO9780511761195
    .. [4] Timoshenko, S. P., Gere, J. M., 1961, Theory of Elastic
       Stability, 2nd ed., McGraw-Hill, Section 2.17, p. 133, Eq. (2-57).

    """
    cdef public int eid, pid
    cdef public int n1, n2
    cdef public int c1, c2
    cdef public int init_k_KC0, init_k_KCNL, init_k_KG, init_k_M
    cdef public double length
    cdef public double vxyi, vxyj, vxyk
    cdef public double r11, r12, r13, r21, r22, r23, r31, r32, r33
    cdef public BeamLRProbe probe

    def __cinit__(BeamLR self, BeamLRProbe p):
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


    cpdef void update_rotation_matrix(BeamLR self, double vxyi, double vxyj,
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
            system, defined using global coordinates.
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


    cpdef void update_probe_ue(BeamLR self, double [::1] u):
        r"""Update the local displacement vector of the probe of the element

        .. note:: The ``ue`` attribute of object :class:`.BeamLRProbe` is
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


    cpdef void update_probe_xe(BeamLR self, double [::1] x):
        r"""Update the 3D coordinates of the probe of the element

        .. note:: The ``xe`` attribute of object :class:`.BeamLRProbe` is
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


    cpdef void update_length(BeamLR self):
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


    cpdef void update_probe_finte(BeamLR self,
                           BeamProp prop,
                           int nonlinear=0):
        r"""Update the internal force vector of the probe

        The attribute ``finte`` is updated with the :class:`.BeamLRProbe` the
        internal forces in local coordinates. While using this function, mind
        that the probe can be shared amongst more than one finite element,
        depending how you defined them, meaning that the probe will always safe
        the values from the last udpate.

        .. note:: The ``finte`` attribute of object :class:`.BeamLRProbe` is
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
        axial strain, for which the exact Jacobian of the internal forces is
        ``KC0 + KCNL + KG``, see :meth:`.update_KCNL`.

        """
        cdef double *ue
        cdef double *finte
        cdef double L
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

            KC0e0000 = C00/L
            KC0e0001 = C01/L
            KC0e0002 = C02/L
            KC0e0003 = C03/L
            KC0e0004 = -C02/2 + C05/L
            KC0e0005 = C01/2 - C04/L
            KC0e0006 = -C00/L
            KC0e0007 = -C01/L
            KC0e0008 = -C02/L
            KC0e0009 = -C03/L
            KC0e0010 = -C02/2 - C05/L
            KC0e0011 = C01/2 + C04/L
            KC0e0101 = C11/L
            KC0e0102 = C12/L
            KC0e0103 = C13/L
            KC0e0104 = -C12/2 + C15/L
            KC0e0105 = C11/2 - C14/L
            KC0e0106 = -C01/L
            KC0e0107 = -C11/L
            KC0e0108 = -C12/L
            KC0e0109 = -C13/L
            KC0e0110 = -C12/2 - C15/L
            KC0e0111 = C11/2 + C14/L
            KC0e0202 = C22/L
            KC0e0203 = C23/L
            KC0e0204 = -C22/2 + C25/L
            KC0e0205 = C12/2 - C24/L
            KC0e0206 = -C02/L
            KC0e0207 = -C12/L
            KC0e0208 = -C22/L
            KC0e0209 = -C23/L
            KC0e0210 = -C22/2 - C25/L
            KC0e0211 = C12/2 + C24/L
            KC0e0303 = C33/L
            KC0e0304 = -C23/2 + C35/L
            KC0e0305 = C13/2 - C34/L
            KC0e0306 = -C03/L
            KC0e0307 = -C13/L
            KC0e0308 = -C23/L
            KC0e0309 = -C33/L
            KC0e0310 = -C23/2 - C35/L
            KC0e0311 = C13/2 + C34/L
            KC0e0404 = C22*L/4 - C25 + C55/L
            KC0e0405 = -C12*L/4 + C15/2 + C24/2 - C45/L
            KC0e0406 = C02/2 - C05/L
            KC0e0407 = C12/2 - C15/L
            KC0e0408 = C22/2 - C25/L
            KC0e0409 = C23/2 - C35/L
            KC0e0410 = C22*L/4 - C55/L
            KC0e0411 = -C12*L/4 + C15/2 - C24/2 + C45/L
            KC0e0505 = C11*L/4 - C14 + C44/L
            KC0e0506 = -C01/2 + C04/L
            KC0e0507 = -C11/2 + C14/L
            KC0e0508 = -C12/2 + C24/L
            KC0e0509 = -C13/2 + C34/L
            KC0e0510 = -C12*L/4 - C15/2 + C24/2 + C45/L
            KC0e0511 = C11*L/4 - C44/L
            KC0e0606 = C00/L
            KC0e0607 = C01/L
            KC0e0608 = C02/L
            KC0e0609 = C03/L
            KC0e0610 = C02/2 + C05/L
            KC0e0611 = -C01/2 - C04/L
            KC0e0707 = C11/L
            KC0e0708 = C12/L
            KC0e0709 = C13/L
            KC0e0710 = C12/2 + C15/L
            KC0e0711 = -C11/2 - C14/L
            KC0e0808 = C22/L
            KC0e0809 = C23/L
            KC0e0810 = C22/2 + C25/L
            KC0e0811 = -C12/2 - C24/L
            KC0e0909 = C33/L
            KC0e0910 = C23/2 + C35/L
            KC0e0911 = -C13/2 - C34/L
            KC0e1010 = C22*L/4 + C25 + C55/L
            KC0e1011 = -C12*L/4 - C15/2 - C24/2 - C45/L
            KC0e1111 = C11*L/4 + C14 + C44/L

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


    cpdef void update_KC0(BeamLR self,
                          long [::1] KC0r,
                          long [::1] KC0c,
                          double [::1] KC0v,
                          BeamProp prop,
                          int update_KC0v_only=0,
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
        :class:`.BeamC`, with the linear shape functions and the integration
        described in the notes of :class:`.BeamLR`, transformed to global
        coordinates with the rotation matrix of the element.

        """
        cdef int c1, c2, k
        cdef double L
        cdef double r11, r12, r13, r21, r22, r23, r31, r32, r33
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
            KC0e0001 = C01/L
            KC0e0002 = C02/L
            KC0e0003 = C03/L
            KC0e0004 = -C02/2 + C05/L
            KC0e0005 = C01/2 - C04/L
            KC0e0006 = -C00/L
            KC0e0007 = -C01/L
            KC0e0008 = -C02/L
            KC0e0009 = -C03/L
            KC0e0010 = -C02/2 - C05/L
            KC0e0011 = C01/2 + C04/L
            KC0e0101 = C11/L
            KC0e0102 = C12/L
            KC0e0103 = C13/L
            KC0e0104 = -C12/2 + C15/L
            KC0e0105 = C11/2 - C14/L
            KC0e0106 = -C01/L
            KC0e0107 = -C11/L
            KC0e0108 = -C12/L
            KC0e0109 = -C13/L
            KC0e0110 = -C12/2 - C15/L
            KC0e0111 = C11/2 + C14/L
            KC0e0202 = C22/L
            KC0e0203 = C23/L
            KC0e0204 = -C22/2 + C25/L
            KC0e0205 = C12/2 - C24/L
            KC0e0206 = -C02/L
            KC0e0207 = -C12/L
            KC0e0208 = -C22/L
            KC0e0209 = -C23/L
            KC0e0210 = -C22/2 - C25/L
            KC0e0211 = C12/2 + C24/L
            KC0e0303 = C33/L
            KC0e0304 = -C23/2 + C35/L
            KC0e0305 = C13/2 - C34/L
            KC0e0306 = -C03/L
            KC0e0307 = -C13/L
            KC0e0308 = -C23/L
            KC0e0309 = -C33/L
            KC0e0310 = -C23/2 - C35/L
            KC0e0311 = C13/2 + C34/L
            KC0e0404 = C22*L/4 - C25 + C55/L
            KC0e0405 = -C12*L/4 + C15/2 + C24/2 - C45/L
            KC0e0406 = C02/2 - C05/L
            KC0e0407 = C12/2 - C15/L
            KC0e0408 = C22/2 - C25/L
            KC0e0409 = C23/2 - C35/L
            KC0e0410 = C22*L/4 - C55/L
            KC0e0411 = -C12*L/4 + C15/2 - C24/2 + C45/L
            KC0e0505 = C11*L/4 - C14 + C44/L
            KC0e0506 = -C01/2 + C04/L
            KC0e0507 = -C11/2 + C14/L
            KC0e0508 = -C12/2 + C24/L
            KC0e0509 = -C13/2 + C34/L
            KC0e0510 = -C12*L/4 - C15/2 + C24/2 + C45/L
            KC0e0511 = C11*L/4 - C44/L
            KC0e0606 = C00/L
            KC0e0607 = C01/L
            KC0e0608 = C02/L
            KC0e0609 = C03/L
            KC0e0610 = C02/2 + C05/L
            KC0e0611 = -C01/2 - C04/L
            KC0e0707 = C11/L
            KC0e0708 = C12/L
            KC0e0709 = C13/L
            KC0e0710 = C12/2 + C15/L
            KC0e0711 = -C11/2 - C14/L
            KC0e0808 = C22/L
            KC0e0809 = C23/L
            KC0e0810 = C22/2 + C25/L
            KC0e0811 = -C12/2 - C24/L
            KC0e0909 = C33/L
            KC0e0910 = C23/2 + C35/L
            KC0e0911 = -C13/2 - C34/L
            KC0e1010 = C22*L/4 + C25 + C55/L
            KC0e1011 = -C12*L/4 - C15/2 - C24/2 - C45/L
            KC0e1111 = C11*L/4 + C14 + C44/L

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

    cpdef void update_fint(BeamLR self,
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
            the parameter ``finte`` of the :class:`.BeamLRProbe` with the
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


    cdef void _update_probe_BL_G(BeamLR self, double xi) noexcept nogil:
        r"""Update the probe rows of the strains and slopes

        The arrays ``BLexx``, ``BLky``, ``BLkz``, ``BLgammay``, ``BLgammaz``,
        ``BLkx``, ``Sv_x`` and ``Sw_x`` of the :class:`.BeamLRProbe` are
        overwritten with the rows that, multiplied by the element
        displacements ``ue``, give in element coordinates the axial strain
        of the beam axis, the two curvatures, the two transverse shear
        strains, the twist and the two slopes of the axis, at the natural
        coordinate ``xi``, from ``-1`` at node 1 to ``+1`` at node 2. With
        linear shape functions, only the rows of the shear strains depend on
        ``xi``, through the rotations; the element calls this method with
        ``xi = 0``, the center.

        Notes
        -----
        The rows are obtained with linear shape functions:

        - ``BLexx``: `u_{,x}`
        - ``BLky``: `\kappa_y = -{r_z}_{,x}`
        - ``BLkz``: `\kappa_z = {r_y}_{,x}`
        - ``BLgammay``: `\gamma_{xy} = v_{,x} - r_z`
        - ``BLgammaz``: `\gamma_{xz} = w_{,x} + r_y`
        - ``BLkx``: `\kappa_x = {r_x}_{,x}`
        - ``Sv_x``, ``Sw_x``: the slopes `v_{,x}` and `w_{,x}` of the
          interpolations of `v` and `w`, as in :meth:`.update_KG`; up to
          version 0.11 they were taken from the rotations, `v_{,x} = r_z`
          and `w_{,x} = -r_y`, see the notes of :class:`.BeamLR`

        """
        cdef int i
        cdef double L
        cdef double *BLexx
        cdef double *BLky
        cdef double *BLkz
        cdef double *BLgammay
        cdef double *BLgammaz
        cdef double *BLkx
        cdef double *Sv_x
        cdef double *Sw_x

        BLexx = &self.probe.BLexx[0]
        BLky = &self.probe.BLky[0]
        BLkz = &self.probe.BLkz[0]
        BLgammay = &self.probe.BLgammay[0]
        BLgammaz = &self.probe.BLgammaz[0]
        BLkx = &self.probe.BLkx[0]
        Sv_x = &self.probe.Sv_x[0]
        Sw_x = &self.probe.Sw_x[0]

        L = self.length

        for i in range(NUM_NODES*DOF):
            BLexx[i] = 0.
            BLky[i] = 0.
            BLkz[i] = 0.
            BLgammay[i] = 0.
            BLgammaz[i] = 0.
            BLkx[i] = 0.
            Sv_x[i] = 0.
            Sw_x[i] = 0.

        # u,x
        BLexx[0] = -1/L
        BLexx[6] = 1/L

        # ky = -rz,x
        BLky[5] = 1/L
        BLky[11] = -1/L

        # kz = ry,x
        BLkz[4] = -1/L
        BLkz[10] = 1/L

        # gammay = v,x - rz
        BLgammay[1] = -1/L
        BLgammay[5] = xi/2 - 1./2
        BLgammay[7] = 1/L
        BLgammay[11] = -xi/2 - 1./2

        # gammaz = w,x + ry
        BLgammaz[2] = -1/L
        BLgammaz[4] = 1./2 - xi/2
        BLgammaz[8] = 1/L
        BLgammaz[10] = xi/2 + 1./2

        # kx = rx,x
        BLkx[3] = -1/L
        BLkx[9] = 1/L

        # v,x, from the interpolation of v
        Sv_x[1] = -1/L
        Sv_x[7] = 1/L

        # w,x, from the interpolation of w
        Sw_x[2] = -1/L
        Sw_x[8] = 1/L


    cdef void _update_probe_KCNLve(BeamLR self, BeamProp prop) noexcept nogil:
        r"""Update the probe values of the nonlinear stiffness matrix

        The attribute ``KCNLve`` of the :class:`.BeamLRProbe` is overwritten
        with the nonlinear stiffness matrix KCNL in element
        coordinates, stored row by row and evaluated at the displacements
        ``ue`` of the probe, from ``length`` and the first row ``C[0, :]`` of
        the sectional stiffness matrix of ``prop``. The arrays of the rows of
        :meth:`._update_probe_BL_G` in the probe are overwritten too. See
        :meth:`.update_KCNL`.

        Notes
        -----
        KCNL = KC0L + KCL0 + KCLL + KGNL, see :meth:`.update_KCNL`, has the
        terms given in :meth:`.BeamC._update_probe_KCNLve`, with the rows of
        :meth:`._update_probe_BL_G` at the center, i.e. the integrand at the
        center times ``length``, the one-point integration of
        :meth:`.update_KC0`. The integrand is constant along the element,
        or linear through the shear strains when ``C[0, 1]`` or ``C[0, 2]``
        is non-zero, so this integration is exact.

        """
        cdef int i, j
        cdef double L, weight, v_x, w_x, NNL
        cdef double C[36]
        cdef double C00, C01, C02, C03, C04, C05
        cdef double EBL[12]
        cdef double BNL[12]
        cdef double *ue
        cdef double *KCNLve
        cdef double *BLexx
        cdef double *BLky
        cdef double *BLkz
        cdef double *BLgammay
        cdef double *BLgammaz
        cdef double *BLkx
        cdef double *Sv_x
        cdef double *Sw_x

        L = self.length
        prop.get_C(C)
        C00 = C[0]
        C01 = C[1]
        C02 = C[2]
        C03 = C[3]
        C04 = C[4]
        C05 = C[5]

        ue = &self.probe.ue[0]
        KCNLve = &self.probe.KCNLve[0]
        BLexx = &self.probe.BLexx[0]
        BLky = &self.probe.BLky[0]
        BLkz = &self.probe.BLkz[0]
        BLgammay = &self.probe.BLgammay[0]
        BLgammaz = &self.probe.BLgammaz[0]
        BLkx = &self.probe.BLkx[0]
        Sv_x = &self.probe.Sv_x[0]
        Sw_x = &self.probe.Sw_x[0]

        for i in range(12*12):
            KCNLve[i] = 0.

        # NOTE the integrands are constant along the element, as the rows of
        #      _update_probe_BL_G, hence the integral is the integrand at
        #      the center times L
        weight = L
        self._update_probe_BL_G(0.)

        v_x = 0.
        w_x = 0.
        for i in range(NUM_NODES*DOF):
            v_x += Sv_x[i]*ue[i]
            w_x += Sw_x[i]*ue[i]

        # axial force of the nonlinear axial strain, eNL = (v_x**2 + w_x**2)/2
        NNL = C00*(v_x*v_x + w_x*w_x)/2.

        for i in range(NUM_NODES*DOF):
            # row of C*BL giving the axial force N, the first row of C times
            # the strains at the centre
            EBL[i] = (C00*BLexx[i] + C04*BLky[i] + C05*BLkz[i]
                      + C01*BLgammay[i] + C02*BLgammaz[i] + C03*BLkx[i])
            # BNL, the variation of the nonlinear axial strain
            BNL[i] = v_x*Sv_x[i] + w_x*Sw_x[i]

        for i in range(NUM_NODES*DOF):
            for j in range(NUM_NODES*DOF):
                KCNLve[12*i + j] += weight*(
                    # KC0L = BL.T*D*BNL
                      EBL[i]*BNL[j]
                    # KCL0 = BNL.T*D*BL
                    + BNL[i]*EBL[j]
                    # KCLL = BNL.T*D*BNL
                    + C00*BNL[i]*BNL[j]
                    # KGNL = NNL*(Sv_x.T*Sv_x + Sw_x.T*Sw_x)
                    + NNL*(Sv_x[i]*Sv_x[j] + Sw_x[i]*Sw_x[j])
                )


    cdef void _update_probe_finte_nonlinear(BeamLR self,
                                            BeamProp prop) noexcept nogil:
        r"""Add the geometrically nonlinear terms to the probe internal forces

        The nonlinear terms, evaluated at the displacements ``ue`` of the
        probe, from ``length`` and the first row ``C[0, :]`` of the sectional
        stiffness matrix of ``prop``, are added to the attribute ``finte`` of
        the :class:`.BeamLRProbe`, which must already contain the linear
        internal forces, see :meth:`.update_probe_finte`. The arrays of the
        rows of :meth:`._update_probe_BL_G` in the probe are overwritten.
        See :meth:`.update_KCNL`.

        Notes
        -----
        The added terms are those of the von Kármán axial strain
        `\epsilon_{NL} = (v_{,x}^2 + w_{,x}^2)/2`, with the slopes of the
        interpolations of `v` and `w`, such that ``finte`` becomes the
        gradient of the strain energy whose Hessian is KC0 + KCNL + KG. They
        are the terms given in :meth:`.BeamC._update_probe_finte_nonlinear`,
        integrated, as the strain energy of :meth:`.update_KC0`, as their
        value at the center times ``length``.

        """
        cdef int i
        cdef double L, weight, v_x, w_x, eNL, N, NNL
        cdef double C[36]
        cdef double C00, C01, C02, C03, C04, C05
        cdef double EBL[12]
        cdef double *ue
        cdef double *finte
        cdef double *BLexx
        cdef double *BLky
        cdef double *BLkz
        cdef double *BLgammay
        cdef double *BLgammaz
        cdef double *BLkx
        cdef double *Sv_x
        cdef double *Sw_x

        L = self.length
        prop.get_C(C)
        C00 = C[0]
        C01 = C[1]
        C02 = C[2]
        C03 = C[3]
        C04 = C[4]
        C05 = C[5]

        ue = &self.probe.ue[0]
        finte = &self.probe.finte[0]
        BLexx = &self.probe.BLexx[0]
        BLky = &self.probe.BLky[0]
        BLkz = &self.probe.BLkz[0]
        BLgammay = &self.probe.BLgammay[0]
        BLgammaz = &self.probe.BLgammaz[0]
        BLkx = &self.probe.BLkx[0]
        Sv_x = &self.probe.Sv_x[0]
        Sw_x = &self.probe.Sw_x[0]

        # NOTE the integrands are constant along the element, as the rows of
        #      _update_probe_BL_G, hence the integral is the integrand at
        #      the center times L
        weight = L
        self._update_probe_BL_G(0.)

        v_x = 0.
        w_x = 0.
        N = 0.
        for i in range(NUM_NODES*DOF):
            v_x += Sv_x[i]*ue[i]
            w_x += Sw_x[i]*ue[i]
            # row of C*BL giving the axial force N, the first row of C times
            # the strains at the centre
            EBL[i] = (C00*BLexx[i] + C04*BLky[i] + C05*BLkz[i]
                      + C01*BLgammay[i] + C02*BLgammaz[i] + C03*BLkx[i])
            N += EBL[i]*ue[i]

        # nonlinear axial strain and its axial force
        eNL = (v_x*v_x + w_x*w_x)/2.
        NNL = C00*eNL

        for i in range(NUM_NODES*DOF):
            finte[i] += weight*(
                # BL.T*D*{eNL, 0, 0, 0, 0, 0}
                  EBL[i]*eNL
                # BNL.T*(N + NNL)
                + (v_x*Sv_x[i] + w_x*Sw_x[i])*(N + NNL)
            )


    cpdef void update_KCNL(BeamLR self,
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

        Before this function is called, the probe :class:`.BeamLRProbe` attribute
        of the :class:`.BeamLR` object must be updated using
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
        \boldsymbol{K}_{GNL}`: the nonlinear constitutive stiffness matrix
        `\boldsymbol{K}_{CNL}`, the terms of `\int_0^L (\boldsymbol{B}_L +
        \boldsymbol{B}_{NL})^\top \boldsymbol{C}_{sec} (\boldsymbol{B}_L +
        \boldsymbol{B}_{NL}) dx` that contain `\boldsymbol{B}_{NL}`, plus the
        geometric stiffness matrix `\boldsymbol{K}_{GNL}` of the axial force
        of the nonlinear axial strain. Crisfield [1]_, Section 7.1.1, Eq.
        (7.24), keeps that axial force in the initial-stress matrix instead,
        whose axial force is the total one of Eq. (7.18), that is
        `\boldsymbol{K}_G + \boldsymbol{K}_{GNL}`; the tangent stiffness
        matrix is the same. The matrices are built from the von Kármán
        axial strain

        .. math::
            \epsilon_{xx} = u_{,x} + \frac{1}{2} \left( v_{,x}^2 + w_{,x}^2 \right)

        whose nonlinear part is `\epsilon_{NL} = \frac{1}{2} \boldsymbol{B}_{NL} \boldsymbol{u}_e`, with
        `\boldsymbol{B}_{NL} = v_{,x} \frac{\partial \boldsymbol{S}^v}{\partial x} + w_{,x} \frac{\partial \boldsymbol{S}^w}{\partial x}` its variation, `\boldsymbol{S}^v` and `\boldsymbol{S}^w` being the linear interpolations of `v` and `w`, whose derivatives, the rows ``Sv_x`` and ``Sw_x`` of :meth:`._update_probe_BL_G`, are constant along the element. Up to version 0.11 the slopes were taken from the rotations, `v_{,x} = r_z` and `w_{,x} = -r_y`, see the notes of :class:`.BeamLR`. With
        `\boldsymbol{B}_L` the linear strain-displacement matrix, `\boldsymbol{C}_{sec}` the sectional
        stiffness matrix ``C`` of :class:`.BeamProp`, whose first row couples the axial strain with
        the other strains, e.g. through `E A_y` and `E A_z`, and `N_{NL} = C_{00} \epsilon_{NL}`, `C_{00} = E A` when ``C`` is not set:

        - KC0L = `\boldsymbol{B}_L^\top \boldsymbol{C}_{sec} \boldsymbol{B}_{NL}`
        - KCL0 = KC0L`^\top`
        - KCLL = `\boldsymbol{B}_{NL}^\top \boldsymbol{C}_{sec} \boldsymbol{B}_{NL}`
        - KGNL = `N_{NL} \left( \frac{\partial \boldsymbol{S}^v}{\partial x}^\top \frac{\partial \boldsymbol{S}^v}{\partial x} + \frac{\partial \boldsymbol{S}^w}{\partial x}^\top \frac{\partial \boldsymbol{S}^w}{\partial x} \right)`

        The first three groups are the constitutive terms coupling the linear and
        the nonlinear parts of the axial strain. KGNL is geometric, carrying the
        axial force of the nonlinear axial strain. It is collected here so that
        :meth:`.update_KG` stays homogeneous of degree one in the displacements,
        which is what a linear buckling analysis needs. With it here,

        .. math::
            \boldsymbol{K}_T = \boldsymbol{K}_{C0} + \boldsymbol{K}_{CNL}(\boldsymbol{u}) + \boldsymbol{K}_{GNL}(\boldsymbol{u}) + \boldsymbol{K}_G(\boldsymbol{u})

        is the exact Jacobian of the internal forces of :meth:`.update_fint` with
        ``nonlinear=1``, and a Newton-Raphson iteration built on them converges
        quadratically.

        The integration is made by :meth:`._update_probe_KCNLve`, exactly,
        since the integrands are constant along the element. The von Kármán axial strain is valid for small strains and
        moderate rotations: the nonlinear axial strain `\epsilon_{NL}` is
        uniform over the cross section, the curvatures and the transverse
        shear strains are linear in the displacements, and the element
        coordinate system is not updated with the displacements.

        References
        ----------
        .. [1] Crisfield, M. A., Non-linear Finite Element Analysis of Solids
           and Structures, Volume 1: Essentials, John Wiley & Sons,
           Chichester, 1991.

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


    cpdef void update_KG(BeamLR self,
                         long [::1] KGr,
                         long [::1] KGc,
                         double [::1] KGv,
                         BeamProp prop,
                         int update_KGv_only=0,
                         ):
        r"""Update sparse vectors for geometric stiffness matrix KG

        The 36 entries of the matrix of the element, in global coordinates,
        are written at the positions ``init_k_KG`` to ``init_k_KG + 35``:
        the row and column indices, built from ``c1`` and ``c2``, are stored
        in ``KGr`` and ``KGc``, and the values are added to those already in
        ``KGv``. The matrix depends on the axial force computed from the
        displacements ``probe.ue``, from ``length`` and from the first row
        ``C[0, :]`` of the sectional stiffness matrix of ``prop``.
        ``probe.ue``, ``length`` and the rotation matrix must be updated
        before, see
        :meth:`.update_probe_ue` and :meth:`.update_probe_xe`. The rows and
        columns of the entries are those of the translations of the two
        nodes; up to version 0.11 they were those of the rotations.

        Parameters
        ----------
        KGr : np.array
           Array to store row positions of sparse values. 1-D array of
           integers of the C type ``long``, of size at least
           ``init_k_KG + 36``.
        KGc : np.array
           Array to store column positions of sparse values. Same type and
           size as ``KGr``.
        KGv : np.array
            Array to store sparse values. 1-D array of floats of size at least
            ``init_k_KG + 36``.
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
            \boldsymbol{K}_G = \int_0^L N \left( \frac{\partial \boldsymbol{S}^v}{\partial x}^\top \frac{\partial \boldsymbol{S}^v}{\partial x} + \frac{\partial \boldsymbol{S}^w}{\partial x}^\top \frac{\partial \boldsymbol{S}^w}{\partial x} \right) dx

        the Hessian of `\frac{1}{2} \int_0^L N (v_{,x}^2 + w_{,x}^2) dx` at a
        constant axial force `N`, where the slopes are those of the linear
        interpolations `\boldsymbol{S}^v` and `\boldsymbol{S}^w` of `v` and
        `w`, see :meth:`._update_probe_BL_G`, and `N =
        \boldsymbol{C}_{0,:} \boldsymbol{\epsilon}` is the axial force of the
        linear strains at ``probe.ue``, at the center of the element, where
        :meth:`.update_KC0` evaluates the strain energy, see the notes of
        :class:`.BeamC`; when ``C`` is not set, `N = E (A u_{,x} + A_y
        \kappa_y + A_z \kappa_z)`, constant along the element. `N` is also
        the axial force of the nodal internal forces, ``-finte[0] =
        finte[6]``. In element coordinates the matrix is `N/L` times
        ``[[1, -1], [-1, 1]]`` for the `v` and for the `w` of the two nodes.

        The slopes are those that enter the axial strain: up to version 0.11
        they were taken from the rotations, `v_{,x} = r_z` and `w_{,x} =
        -r_y`, which neglects the shear deformation in buckling, so that the
        buckling load converged to the Euler load instead of Eq. (2-57) of
        Timoshenko and Gere (1961), see the notes of :class:`.BeamLR`.

        `\boldsymbol{K}_G` is homogeneous of degree one in ``probe.ue``. With ``probe.ue``
        from the linear static solution of a reference load, the critical
        load multipliers `\lambda` of the linear buckling analysis solve
        `(\boldsymbol{K}_{C0} + \lambda \boldsymbol{K}_G) \boldsymbol{u} = \boldsymbol{0}`, see
        :func:`.linear_buckling`. The geometric term of the
        nonlinear axial strain is in :meth:`.update_KCNL`.

        """
        cdef double *ue
        cdef int k, node_i, node_j, m, n
        cdef int c[2]
        cdef double L, N, sign
        cdef double r[3][3]
        cdef double C[36]
        cdef double C00, C01, C02, C03, C04, C05

        with nogil:
            L = self.length
            prop.get_C(C)
            C00 = C[0]
            C01 = C[1]
            C02 = C[2]
            C03 = C[3]
            C04 = C[4]
            C05 = C[5]

            # local to global transformation
            r[0][0] = self.r11
            r[0][1] = self.r12
            r[0][2] = self.r13
            r[1][0] = self.r21
            r[1][1] = self.r22
            r[1][2] = self.r23
            r[2][0] = self.r31
            r[2][1] = self.r32
            r[2][2] = self.r33

            ue = &self.probe.ue[0]

            if update_KGv_only == 0:
                # positions in the global stiffness matrix
                c[0] = self.c1
                c[1] = self.c2

                # NOTE rows and columns of the translations of the two nodes
                for node_i in range(NUM_NODES):
                    for m in range(3):
                        for node_j in range(NUM_NODES):
                            for n in range(3):
                                k = self.init_k_KG + 6*(3*node_i + m) + 3*node_j + n
                                KGr[k] = c[node_i] + m
                                KGc[k] = c[node_j] + n

            # NOTE first row of C times the strains at the centre
            N = C01*(-ue[11]/2 - ue[5]/2 + (-ue[1] + ue[7])/L) + C02*(ue[10]/2 + ue[4]/2 + (-ue[2] + ue[8])/L) + (C00*(-ue[0] + ue[6]) + C03*(-ue[3] + ue[9]) + C04*(-ue[11] + ue[5]) + C05*(ue[10] - ue[4]))/L

            # NOTE in element coordinates KG = N/L*[[1, -1], [-1, 1]] for v
            #      and for w, i.e. N*L*(Sv_x.T*Sv_x + Sw_x.T*Sw_x) with the
            #      constant rows of _update_probe_BL_G, transformed to global
            #      coordinates as KG_mn = r_m1*N/L*r_n1 + r_m2*N/L*r_n2 for
            #      each pair of nodes, with a minus sign between two nodes
            for node_i in range(NUM_NODES):
                for m in range(3):
                    for node_j in range(NUM_NODES):
                        sign = 1. if node_i == node_j else -1.
                        for n in range(3):
                            k = self.init_k_KG + 6*(3*node_i + m) + 3*node_j + n
                            KGv[k] += sign*N/L*(r[m][1]*r[n][1] + r[m][2]*r[n][2])


    cpdef void update_M(BeamLR self,
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
        ``intrhoyz`` of ``prop``. ``length`` and the rotation matrix must be
        updated before, see :meth:`.update_rotation_matrix` and
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
        respect to the nodal velocities, with the six displacements of the
        beam axis interpolated linearly, as in :class:`.BeamLR`. The integral
        along the element is computed exactly, e.g. with the two-point
        Gauss-Legendre quadrature, and the rotary inertia is included.

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
        the integral of Petyt (2010) [2]_, Sec. 3.3, Eqs. (3.58) and (3.59),
        for each linearly interpolated field, e.g. `L m/6` times `[2, 1; 1,
        2]` for each translation, and Sec. 3.4, Eq. (3.102), with `m_{yy} +
        m_{zz}` in place of `\rho I_x`, for the twist `r_x`.

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

        # NOTE checked once per call, before the code without the GIL
        if mtype != 0 and mtype != 1:
            raise ValueError('mtype=%d is not supported by BeamLR.update_M, '
                             'use 0 (consistent) or 1 (lumped); BeamLR has '
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
                Mv[k] += 0.333333333333333*L*intrho*r11**2 + 0.333333333333333*L*intrho*r12**2 + 0.333333333333333*L*intrho*r13**2
                k += 1
                Mv[k] += 0.333333333333333*L*intrho*r11*r21 + 0.333333333333333*L*intrho*r12*r22 + 0.333333333333333*L*intrho*r13*r23
                k += 1
                Mv[k] += 0.333333333333333*L*intrho*r11*r31 + 0.333333333333333*L*intrho*r12*r32 + 0.333333333333333*L*intrho*r13*r33
                k += 1
                Mv[k] += -0.333333333333333*L*intrhoy*r11*r13 + 0.333333333333333*L*intrhoz*r11*r12 + r11*(0.333333333333333*L*intrhoy*r13 - 0.333333333333333*L*intrhoz*r12)
                k += 1
                Mv[k] += -0.333333333333333*L*intrhoy*r11*r23 + 0.333333333333333*L*intrhoz*r11*r22 + r21*(0.333333333333333*L*intrhoy*r13 - 0.333333333333333*L*intrhoz*r12)
                k += 1
                Mv[k] += -0.333333333333333*L*intrhoy*r11*r33 + 0.333333333333333*L*intrhoz*r11*r32 + r31*(0.333333333333333*L*intrhoy*r13 - 0.333333333333333*L*intrhoz*r12)
                k += 1
                Mv[k] += 0.166666666666667*L*intrho*r11**2 + 0.166666666666667*L*intrho*r12**2 + 0.166666666666667*L*intrho*r13**2
                k += 1
                Mv[k] += 0.166666666666667*L*intrho*r11*r21 + 0.166666666666667*L*intrho*r12*r22 + 0.166666666666667*L*intrho*r13*r23
                k += 1
                Mv[k] += 0.166666666666667*L*intrho*r11*r31 + 0.166666666666667*L*intrho*r12*r32 + 0.166666666666667*L*intrho*r13*r33
                k += 1
                Mv[k] += -0.166666666666667*L*intrhoy*r11*r13 + 0.166666666666667*L*intrhoz*r11*r12 + r11*(0.166666666666667*L*intrhoy*r13 - 0.166666666666667*L*intrhoz*r12)
                k += 1
                Mv[k] += -0.166666666666667*L*intrhoy*r11*r23 + 0.166666666666667*L*intrhoz*r11*r22 + r21*(0.166666666666667*L*intrhoy*r13 - 0.166666666666667*L*intrhoz*r12)
                k += 1
                Mv[k] += -0.166666666666667*L*intrhoy*r11*r33 + 0.166666666666667*L*intrhoz*r11*r32 + r31*(0.166666666666667*L*intrhoy*r13 - 0.166666666666667*L*intrhoz*r12)
                k += 1
                Mv[k] += 0.333333333333333*L*intrho*r11*r21 + 0.333333333333333*L*intrho*r12*r22 + 0.333333333333333*L*intrho*r13*r23
                k += 1
                Mv[k] += 0.333333333333333*L*intrho*r21**2 + 0.333333333333333*L*intrho*r22**2 + 0.333333333333333*L*intrho*r23**2
                k += 1
                Mv[k] += 0.333333333333333*L*intrho*r21*r31 + 0.333333333333333*L*intrho*r22*r32 + 0.333333333333333*L*intrho*r23*r33
                k += 1
                Mv[k] += -0.333333333333333*L*intrhoy*r13*r21 + 0.333333333333333*L*intrhoz*r12*r21 + r11*(0.333333333333333*L*intrhoy*r23 - 0.333333333333333*L*intrhoz*r22)
                k += 1
                Mv[k] += -0.333333333333333*L*intrhoy*r21*r23 + 0.333333333333333*L*intrhoz*r21*r22 + r21*(0.333333333333333*L*intrhoy*r23 - 0.333333333333333*L*intrhoz*r22)
                k += 1
                Mv[k] += -0.333333333333333*L*intrhoy*r21*r33 + 0.333333333333333*L*intrhoz*r21*r32 + r31*(0.333333333333333*L*intrhoy*r23 - 0.333333333333333*L*intrhoz*r22)
                k += 1
                Mv[k] += 0.166666666666667*L*intrho*r11*r21 + 0.166666666666667*L*intrho*r12*r22 + 0.166666666666667*L*intrho*r13*r23
                k += 1
                Mv[k] += 0.166666666666667*L*intrho*r21**2 + 0.166666666666667*L*intrho*r22**2 + 0.166666666666667*L*intrho*r23**2
                k += 1
                Mv[k] += 0.166666666666667*L*intrho*r21*r31 + 0.166666666666667*L*intrho*r22*r32 + 0.166666666666667*L*intrho*r23*r33
                k += 1
                Mv[k] += -0.166666666666667*L*intrhoy*r13*r21 + 0.166666666666667*L*intrhoz*r12*r21 + r11*(0.166666666666667*L*intrhoy*r23 - 0.166666666666667*L*intrhoz*r22)
                k += 1
                Mv[k] += -0.166666666666667*L*intrhoy*r21*r23 + 0.166666666666667*L*intrhoz*r21*r22 + r21*(0.166666666666667*L*intrhoy*r23 - 0.166666666666667*L*intrhoz*r22)
                k += 1
                Mv[k] += -0.166666666666667*L*intrhoy*r21*r33 + 0.166666666666667*L*intrhoz*r21*r32 + r31*(0.166666666666667*L*intrhoy*r23 - 0.166666666666667*L*intrhoz*r22)
                k += 1
                Mv[k] += 0.333333333333333*L*intrho*r11*r31 + 0.333333333333333*L*intrho*r12*r32 + 0.333333333333333*L*intrho*r13*r33
                k += 1
                Mv[k] += 0.333333333333333*L*intrho*r21*r31 + 0.333333333333333*L*intrho*r22*r32 + 0.333333333333333*L*intrho*r23*r33
                k += 1
                Mv[k] += 0.333333333333333*L*intrho*r31**2 + 0.333333333333333*L*intrho*r32**2 + 0.333333333333333*L*intrho*r33**2
                k += 1
                Mv[k] += -0.333333333333333*L*intrhoy*r13*r31 + 0.333333333333333*L*intrhoz*r12*r31 + r11*(0.333333333333333*L*intrhoy*r33 - 0.333333333333333*L*intrhoz*r32)
                k += 1
                Mv[k] += -0.333333333333333*L*intrhoy*r23*r31 + 0.333333333333333*L*intrhoz*r22*r31 + r21*(0.333333333333333*L*intrhoy*r33 - 0.333333333333333*L*intrhoz*r32)
                k += 1
                Mv[k] += -0.333333333333333*L*intrhoy*r31*r33 + 0.333333333333333*L*intrhoz*r31*r32 + r31*(0.333333333333333*L*intrhoy*r33 - 0.333333333333333*L*intrhoz*r32)
                k += 1
                Mv[k] += 0.166666666666667*L*intrho*r11*r31 + 0.166666666666667*L*intrho*r12*r32 + 0.166666666666667*L*intrho*r13*r33
                k += 1
                Mv[k] += 0.166666666666667*L*intrho*r21*r31 + 0.166666666666667*L*intrho*r22*r32 + 0.166666666666667*L*intrho*r23*r33
                k += 1
                Mv[k] += 0.166666666666667*L*intrho*r31**2 + 0.166666666666667*L*intrho*r32**2 + 0.166666666666667*L*intrho*r33**2
                k += 1
                Mv[k] += -0.166666666666667*L*intrhoy*r13*r31 + 0.166666666666667*L*intrhoz*r12*r31 + r11*(0.166666666666667*L*intrhoy*r33 - 0.166666666666667*L*intrhoz*r32)
                k += 1
                Mv[k] += -0.166666666666667*L*intrhoy*r23*r31 + 0.166666666666667*L*intrhoz*r22*r31 + r21*(0.166666666666667*L*intrhoy*r33 - 0.166666666666667*L*intrhoz*r32)
                k += 1
                Mv[k] += -0.166666666666667*L*intrhoy*r31*r33 + 0.166666666666667*L*intrhoz*r31*r32 + r31*(0.166666666666667*L*intrhoy*r33 - 0.166666666666667*L*intrhoz*r32)
                k += 1
                Mv[k] += 0.333333333333333*L*intrhoy*r11*r13 - 0.333333333333333*L*intrhoz*r11*r12 + r11*(-0.333333333333333*L*intrhoy*r13 + 0.333333333333333*L*intrhoz*r12)
                k += 1
                Mv[k] += 0.333333333333333*L*intrhoy*r11*r23 - 0.333333333333333*L*intrhoz*r11*r22 + r21*(-0.333333333333333*L*intrhoy*r13 + 0.333333333333333*L*intrhoz*r12)
                k += 1
                Mv[k] += 0.333333333333333*L*intrhoy*r11*r33 - 0.333333333333333*L*intrhoz*r11*r32 + r31*(-0.333333333333333*L*intrhoy*r13 + 0.333333333333333*L*intrhoz*r12)
                k += 1
                Mv[k] += 0.333333333333333*L*r11**2*(intrhoy2 + intrhoz2) + r12*(-0.333333333333333*L*intrhoyz*r13 + 0.333333333333333*L*intrhoz2*r12) + r13*(0.333333333333333*L*intrhoy2*r13 - 0.333333333333333*L*intrhoyz*r12)
                k += 1
                Mv[k] += 0.333333333333333*L*r11*r21*(intrhoy2 + intrhoz2) + r22*(-0.333333333333333*L*intrhoyz*r13 + 0.333333333333333*L*intrhoz2*r12) + r23*(0.333333333333333*L*intrhoy2*r13 - 0.333333333333333*L*intrhoyz*r12)
                k += 1
                Mv[k] += 0.333333333333333*L*r11*r31*(intrhoy2 + intrhoz2) + r32*(-0.333333333333333*L*intrhoyz*r13 + 0.333333333333333*L*intrhoz2*r12) + r33*(0.333333333333333*L*intrhoy2*r13 - 0.333333333333333*L*intrhoyz*r12)
                k += 1
                Mv[k] += 0.166666666666667*L*intrhoy*r11*r13 - 0.166666666666667*L*intrhoz*r11*r12 + r11*(-0.166666666666667*L*intrhoy*r13 + 0.166666666666667*L*intrhoz*r12)
                k += 1
                Mv[k] += 0.166666666666667*L*intrhoy*r11*r23 - 0.166666666666667*L*intrhoz*r11*r22 + r21*(-0.166666666666667*L*intrhoy*r13 + 0.166666666666667*L*intrhoz*r12)
                k += 1
                Mv[k] += 0.166666666666667*L*intrhoy*r11*r33 - 0.166666666666667*L*intrhoz*r11*r32 + r31*(-0.166666666666667*L*intrhoy*r13 + 0.166666666666667*L*intrhoz*r12)
                k += 1
                Mv[k] += 0.166666666666667*L*r11**2*(intrhoy2 + intrhoz2) + r12*(-0.166666666666667*L*intrhoyz*r13 + 0.166666666666667*L*intrhoz2*r12) + r13*(0.166666666666667*L*intrhoy2*r13 - 0.166666666666667*L*intrhoyz*r12)
                k += 1
                Mv[k] += 0.166666666666667*L*r11*r21*(intrhoy2 + intrhoz2) + r22*(-0.166666666666667*L*intrhoyz*r13 + 0.166666666666667*L*intrhoz2*r12) + r23*(0.166666666666667*L*intrhoy2*r13 - 0.166666666666667*L*intrhoyz*r12)
                k += 1
                Mv[k] += 0.166666666666667*L*r11*r31*(intrhoy2 + intrhoz2) + r32*(-0.166666666666667*L*intrhoyz*r13 + 0.166666666666667*L*intrhoz2*r12) + r33*(0.166666666666667*L*intrhoy2*r13 - 0.166666666666667*L*intrhoyz*r12)
                k += 1
                Mv[k] += 0.333333333333333*L*intrhoy*r13*r21 - 0.333333333333333*L*intrhoz*r12*r21 + r11*(-0.333333333333333*L*intrhoy*r23 + 0.333333333333333*L*intrhoz*r22)
                k += 1
                Mv[k] += 0.333333333333333*L*intrhoy*r21*r23 - 0.333333333333333*L*intrhoz*r21*r22 + r21*(-0.333333333333333*L*intrhoy*r23 + 0.333333333333333*L*intrhoz*r22)
                k += 1
                Mv[k] += 0.333333333333333*L*intrhoy*r21*r33 - 0.333333333333333*L*intrhoz*r21*r32 + r31*(-0.333333333333333*L*intrhoy*r23 + 0.333333333333333*L*intrhoz*r22)
                k += 1
                Mv[k] += 0.333333333333333*L*r11*r21*(intrhoy2 + intrhoz2) + r12*(-0.333333333333333*L*intrhoyz*r23 + 0.333333333333333*L*intrhoz2*r22) + r13*(0.333333333333333*L*intrhoy2*r23 - 0.333333333333333*L*intrhoyz*r22)
                k += 1
                Mv[k] += 0.333333333333333*L*r21**2*(intrhoy2 + intrhoz2) + r22*(-0.333333333333333*L*intrhoyz*r23 + 0.333333333333333*L*intrhoz2*r22) + r23*(0.333333333333333*L*intrhoy2*r23 - 0.333333333333333*L*intrhoyz*r22)
                k += 1
                Mv[k] += 0.333333333333333*L*r21*r31*(intrhoy2 + intrhoz2) + r32*(-0.333333333333333*L*intrhoyz*r23 + 0.333333333333333*L*intrhoz2*r22) + r33*(0.333333333333333*L*intrhoy2*r23 - 0.333333333333333*L*intrhoyz*r22)
                k += 1
                Mv[k] += 0.166666666666667*L*intrhoy*r13*r21 - 0.166666666666667*L*intrhoz*r12*r21 + r11*(-0.166666666666667*L*intrhoy*r23 + 0.166666666666667*L*intrhoz*r22)
                k += 1
                Mv[k] += 0.166666666666667*L*intrhoy*r21*r23 - 0.166666666666667*L*intrhoz*r21*r22 + r21*(-0.166666666666667*L*intrhoy*r23 + 0.166666666666667*L*intrhoz*r22)
                k += 1
                Mv[k] += 0.166666666666667*L*intrhoy*r21*r33 - 0.166666666666667*L*intrhoz*r21*r32 + r31*(-0.166666666666667*L*intrhoy*r23 + 0.166666666666667*L*intrhoz*r22)
                k += 1
                Mv[k] += 0.166666666666667*L*r11*r21*(intrhoy2 + intrhoz2) + r12*(-0.166666666666667*L*intrhoyz*r23 + 0.166666666666667*L*intrhoz2*r22) + r13*(0.166666666666667*L*intrhoy2*r23 - 0.166666666666667*L*intrhoyz*r22)
                k += 1
                Mv[k] += 0.166666666666667*L*r21**2*(intrhoy2 + intrhoz2) + r22*(-0.166666666666667*L*intrhoyz*r23 + 0.166666666666667*L*intrhoz2*r22) + r23*(0.166666666666667*L*intrhoy2*r23 - 0.166666666666667*L*intrhoyz*r22)
                k += 1
                Mv[k] += 0.166666666666667*L*r21*r31*(intrhoy2 + intrhoz2) + r32*(-0.166666666666667*L*intrhoyz*r23 + 0.166666666666667*L*intrhoz2*r22) + r33*(0.166666666666667*L*intrhoy2*r23 - 0.166666666666667*L*intrhoyz*r22)
                k += 1
                Mv[k] += 0.333333333333333*L*intrhoy*r13*r31 - 0.333333333333333*L*intrhoz*r12*r31 + r11*(-0.333333333333333*L*intrhoy*r33 + 0.333333333333333*L*intrhoz*r32)
                k += 1
                Mv[k] += 0.333333333333333*L*intrhoy*r23*r31 - 0.333333333333333*L*intrhoz*r22*r31 + r21*(-0.333333333333333*L*intrhoy*r33 + 0.333333333333333*L*intrhoz*r32)
                k += 1
                Mv[k] += 0.333333333333333*L*intrhoy*r31*r33 - 0.333333333333333*L*intrhoz*r31*r32 + r31*(-0.333333333333333*L*intrhoy*r33 + 0.333333333333333*L*intrhoz*r32)
                k += 1
                Mv[k] += 0.333333333333333*L*r11*r31*(intrhoy2 + intrhoz2) + r12*(-0.333333333333333*L*intrhoyz*r33 + 0.333333333333333*L*intrhoz2*r32) + r13*(0.333333333333333*L*intrhoy2*r33 - 0.333333333333333*L*intrhoyz*r32)
                k += 1
                Mv[k] += 0.333333333333333*L*r21*r31*(intrhoy2 + intrhoz2) + r22*(-0.333333333333333*L*intrhoyz*r33 + 0.333333333333333*L*intrhoz2*r32) + r23*(0.333333333333333*L*intrhoy2*r33 - 0.333333333333333*L*intrhoyz*r32)
                k += 1
                Mv[k] += 0.333333333333333*L*r31**2*(intrhoy2 + intrhoz2) + r32*(-0.333333333333333*L*intrhoyz*r33 + 0.333333333333333*L*intrhoz2*r32) + r33*(0.333333333333333*L*intrhoy2*r33 - 0.333333333333333*L*intrhoyz*r32)
                k += 1
                Mv[k] += 0.166666666666667*L*intrhoy*r13*r31 - 0.166666666666667*L*intrhoz*r12*r31 + r11*(-0.166666666666667*L*intrhoy*r33 + 0.166666666666667*L*intrhoz*r32)
                k += 1
                Mv[k] += 0.166666666666667*L*intrhoy*r23*r31 - 0.166666666666667*L*intrhoz*r22*r31 + r21*(-0.166666666666667*L*intrhoy*r33 + 0.166666666666667*L*intrhoz*r32)
                k += 1
                Mv[k] += 0.166666666666667*L*intrhoy*r31*r33 - 0.166666666666667*L*intrhoz*r31*r32 + r31*(-0.166666666666667*L*intrhoy*r33 + 0.166666666666667*L*intrhoz*r32)
                k += 1
                Mv[k] += 0.166666666666667*L*r11*r31*(intrhoy2 + intrhoz2) + r12*(-0.166666666666667*L*intrhoyz*r33 + 0.166666666666667*L*intrhoz2*r32) + r13*(0.166666666666667*L*intrhoy2*r33 - 0.166666666666667*L*intrhoyz*r32)
                k += 1
                Mv[k] += 0.166666666666667*L*r21*r31*(intrhoy2 + intrhoz2) + r22*(-0.166666666666667*L*intrhoyz*r33 + 0.166666666666667*L*intrhoz2*r32) + r23*(0.166666666666667*L*intrhoy2*r33 - 0.166666666666667*L*intrhoyz*r32)
                k += 1
                Mv[k] += 0.166666666666667*L*r31**2*(intrhoy2 + intrhoz2) + r32*(-0.166666666666667*L*intrhoyz*r33 + 0.166666666666667*L*intrhoz2*r32) + r33*(0.166666666666667*L*intrhoy2*r33 - 0.166666666666667*L*intrhoyz*r32)
                k += 1
                Mv[k] += 0.166666666666667*L*intrho*r11**2 + 0.166666666666667*L*intrho*r12**2 + 0.166666666666667*L*intrho*r13**2
                k += 1
                Mv[k] += 0.166666666666667*L*intrho*r11*r21 + 0.166666666666667*L*intrho*r12*r22 + 0.166666666666667*L*intrho*r13*r23
                k += 1
                Mv[k] += 0.166666666666667*L*intrho*r11*r31 + 0.166666666666667*L*intrho*r12*r32 + 0.166666666666667*L*intrho*r13*r33
                k += 1
                Mv[k] += -0.166666666666667*L*intrhoy*r11*r13 + 0.166666666666667*L*intrhoz*r11*r12 + r11*(0.166666666666667*L*intrhoy*r13 - 0.166666666666667*L*intrhoz*r12)
                k += 1
                Mv[k] += -0.166666666666667*L*intrhoy*r11*r23 + 0.166666666666667*L*intrhoz*r11*r22 + r21*(0.166666666666667*L*intrhoy*r13 - 0.166666666666667*L*intrhoz*r12)
                k += 1
                Mv[k] += -0.166666666666667*L*intrhoy*r11*r33 + 0.166666666666667*L*intrhoz*r11*r32 + r31*(0.166666666666667*L*intrhoy*r13 - 0.166666666666667*L*intrhoz*r12)
                k += 1
                Mv[k] += 0.333333333333333*L*intrho*r11**2 + 0.333333333333333*L*intrho*r12**2 + 0.333333333333333*L*intrho*r13**2
                k += 1
                Mv[k] += 0.333333333333333*L*intrho*r11*r21 + 0.333333333333333*L*intrho*r12*r22 + 0.333333333333333*L*intrho*r13*r23
                k += 1
                Mv[k] += 0.333333333333333*L*intrho*r11*r31 + 0.333333333333333*L*intrho*r12*r32 + 0.333333333333333*L*intrho*r13*r33
                k += 1
                Mv[k] += -0.333333333333333*L*intrhoy*r11*r13 + 0.333333333333333*L*intrhoz*r11*r12 + r11*(0.333333333333333*L*intrhoy*r13 - 0.333333333333333*L*intrhoz*r12)
                k += 1
                Mv[k] += -0.333333333333333*L*intrhoy*r11*r23 + 0.333333333333333*L*intrhoz*r11*r22 + r21*(0.333333333333333*L*intrhoy*r13 - 0.333333333333333*L*intrhoz*r12)
                k += 1
                Mv[k] += -0.333333333333333*L*intrhoy*r11*r33 + 0.333333333333333*L*intrhoz*r11*r32 + r31*(0.333333333333333*L*intrhoy*r13 - 0.333333333333333*L*intrhoz*r12)
                k += 1
                Mv[k] += 0.166666666666667*L*intrho*r11*r21 + 0.166666666666667*L*intrho*r12*r22 + 0.166666666666667*L*intrho*r13*r23
                k += 1
                Mv[k] += 0.166666666666667*L*intrho*r21**2 + 0.166666666666667*L*intrho*r22**2 + 0.166666666666667*L*intrho*r23**2
                k += 1
                Mv[k] += 0.166666666666667*L*intrho*r21*r31 + 0.166666666666667*L*intrho*r22*r32 + 0.166666666666667*L*intrho*r23*r33
                k += 1
                Mv[k] += -0.166666666666667*L*intrhoy*r13*r21 + 0.166666666666667*L*intrhoz*r12*r21 + r11*(0.166666666666667*L*intrhoy*r23 - 0.166666666666667*L*intrhoz*r22)
                k += 1
                Mv[k] += -0.166666666666667*L*intrhoy*r21*r23 + 0.166666666666667*L*intrhoz*r21*r22 + r21*(0.166666666666667*L*intrhoy*r23 - 0.166666666666667*L*intrhoz*r22)
                k += 1
                Mv[k] += -0.166666666666667*L*intrhoy*r21*r33 + 0.166666666666667*L*intrhoz*r21*r32 + r31*(0.166666666666667*L*intrhoy*r23 - 0.166666666666667*L*intrhoz*r22)
                k += 1
                Mv[k] += 0.333333333333333*L*intrho*r11*r21 + 0.333333333333333*L*intrho*r12*r22 + 0.333333333333333*L*intrho*r13*r23
                k += 1
                Mv[k] += 0.333333333333333*L*intrho*r21**2 + 0.333333333333333*L*intrho*r22**2 + 0.333333333333333*L*intrho*r23**2
                k += 1
                Mv[k] += 0.333333333333333*L*intrho*r21*r31 + 0.333333333333333*L*intrho*r22*r32 + 0.333333333333333*L*intrho*r23*r33
                k += 1
                Mv[k] += -0.333333333333333*L*intrhoy*r13*r21 + 0.333333333333333*L*intrhoz*r12*r21 + r11*(0.333333333333333*L*intrhoy*r23 - 0.333333333333333*L*intrhoz*r22)
                k += 1
                Mv[k] += -0.333333333333333*L*intrhoy*r21*r23 + 0.333333333333333*L*intrhoz*r21*r22 + r21*(0.333333333333333*L*intrhoy*r23 - 0.333333333333333*L*intrhoz*r22)
                k += 1
                Mv[k] += -0.333333333333333*L*intrhoy*r21*r33 + 0.333333333333333*L*intrhoz*r21*r32 + r31*(0.333333333333333*L*intrhoy*r23 - 0.333333333333333*L*intrhoz*r22)
                k += 1
                Mv[k] += 0.166666666666667*L*intrho*r11*r31 + 0.166666666666667*L*intrho*r12*r32 + 0.166666666666667*L*intrho*r13*r33
                k += 1
                Mv[k] += 0.166666666666667*L*intrho*r21*r31 + 0.166666666666667*L*intrho*r22*r32 + 0.166666666666667*L*intrho*r23*r33
                k += 1
                Mv[k] += 0.166666666666667*L*intrho*r31**2 + 0.166666666666667*L*intrho*r32**2 + 0.166666666666667*L*intrho*r33**2
                k += 1
                Mv[k] += -0.166666666666667*L*intrhoy*r13*r31 + 0.166666666666667*L*intrhoz*r12*r31 + r11*(0.166666666666667*L*intrhoy*r33 - 0.166666666666667*L*intrhoz*r32)
                k += 1
                Mv[k] += -0.166666666666667*L*intrhoy*r23*r31 + 0.166666666666667*L*intrhoz*r22*r31 + r21*(0.166666666666667*L*intrhoy*r33 - 0.166666666666667*L*intrhoz*r32)
                k += 1
                Mv[k] += -0.166666666666667*L*intrhoy*r31*r33 + 0.166666666666667*L*intrhoz*r31*r32 + r31*(0.166666666666667*L*intrhoy*r33 - 0.166666666666667*L*intrhoz*r32)
                k += 1
                Mv[k] += 0.333333333333333*L*intrho*r11*r31 + 0.333333333333333*L*intrho*r12*r32 + 0.333333333333333*L*intrho*r13*r33
                k += 1
                Mv[k] += 0.333333333333333*L*intrho*r21*r31 + 0.333333333333333*L*intrho*r22*r32 + 0.333333333333333*L*intrho*r23*r33
                k += 1
                Mv[k] += 0.333333333333333*L*intrho*r31**2 + 0.333333333333333*L*intrho*r32**2 + 0.333333333333333*L*intrho*r33**2
                k += 1
                Mv[k] += -0.333333333333333*L*intrhoy*r13*r31 + 0.333333333333333*L*intrhoz*r12*r31 + r11*(0.333333333333333*L*intrhoy*r33 - 0.333333333333333*L*intrhoz*r32)
                k += 1
                Mv[k] += -0.333333333333333*L*intrhoy*r23*r31 + 0.333333333333333*L*intrhoz*r22*r31 + r21*(0.333333333333333*L*intrhoy*r33 - 0.333333333333333*L*intrhoz*r32)
                k += 1
                Mv[k] += -0.333333333333333*L*intrhoy*r31*r33 + 0.333333333333333*L*intrhoz*r31*r32 + r31*(0.333333333333333*L*intrhoy*r33 - 0.333333333333333*L*intrhoz*r32)
                k += 1
                Mv[k] += 0.166666666666667*L*intrhoy*r11*r13 - 0.166666666666667*L*intrhoz*r11*r12 + r11*(-0.166666666666667*L*intrhoy*r13 + 0.166666666666667*L*intrhoz*r12)
                k += 1
                Mv[k] += 0.166666666666667*L*intrhoy*r11*r23 - 0.166666666666667*L*intrhoz*r11*r22 + r21*(-0.166666666666667*L*intrhoy*r13 + 0.166666666666667*L*intrhoz*r12)
                k += 1
                Mv[k] += 0.166666666666667*L*intrhoy*r11*r33 - 0.166666666666667*L*intrhoz*r11*r32 + r31*(-0.166666666666667*L*intrhoy*r13 + 0.166666666666667*L*intrhoz*r12)
                k += 1
                Mv[k] += 0.166666666666667*L*r11**2*(intrhoy2 + intrhoz2) + r12*(-0.166666666666667*L*intrhoyz*r13 + 0.166666666666667*L*intrhoz2*r12) + r13*(0.166666666666667*L*intrhoy2*r13 - 0.166666666666667*L*intrhoyz*r12)
                k += 1
                Mv[k] += 0.166666666666667*L*r11*r21*(intrhoy2 + intrhoz2) + r22*(-0.166666666666667*L*intrhoyz*r13 + 0.166666666666667*L*intrhoz2*r12) + r23*(0.166666666666667*L*intrhoy2*r13 - 0.166666666666667*L*intrhoyz*r12)
                k += 1
                Mv[k] += 0.166666666666667*L*r11*r31*(intrhoy2 + intrhoz2) + r32*(-0.166666666666667*L*intrhoyz*r13 + 0.166666666666667*L*intrhoz2*r12) + r33*(0.166666666666667*L*intrhoy2*r13 - 0.166666666666667*L*intrhoyz*r12)
                k += 1
                Mv[k] += 0.333333333333333*L*intrhoy*r11*r13 - 0.333333333333333*L*intrhoz*r11*r12 + r11*(-0.333333333333333*L*intrhoy*r13 + 0.333333333333333*L*intrhoz*r12)
                k += 1
                Mv[k] += 0.333333333333333*L*intrhoy*r11*r23 - 0.333333333333333*L*intrhoz*r11*r22 + r21*(-0.333333333333333*L*intrhoy*r13 + 0.333333333333333*L*intrhoz*r12)
                k += 1
                Mv[k] += 0.333333333333333*L*intrhoy*r11*r33 - 0.333333333333333*L*intrhoz*r11*r32 + r31*(-0.333333333333333*L*intrhoy*r13 + 0.333333333333333*L*intrhoz*r12)
                k += 1
                Mv[k] += 0.333333333333333*L*r11**2*(intrhoy2 + intrhoz2) + r12*(-0.333333333333333*L*intrhoyz*r13 + 0.333333333333333*L*intrhoz2*r12) + r13*(0.333333333333333*L*intrhoy2*r13 - 0.333333333333333*L*intrhoyz*r12)
                k += 1
                Mv[k] += 0.333333333333333*L*r11*r21*(intrhoy2 + intrhoz2) + r22*(-0.333333333333333*L*intrhoyz*r13 + 0.333333333333333*L*intrhoz2*r12) + r23*(0.333333333333333*L*intrhoy2*r13 - 0.333333333333333*L*intrhoyz*r12)
                k += 1
                Mv[k] += 0.333333333333333*L*r11*r31*(intrhoy2 + intrhoz2) + r32*(-0.333333333333333*L*intrhoyz*r13 + 0.333333333333333*L*intrhoz2*r12) + r33*(0.333333333333333*L*intrhoy2*r13 - 0.333333333333333*L*intrhoyz*r12)
                k += 1
                Mv[k] += 0.166666666666667*L*intrhoy*r13*r21 - 0.166666666666667*L*intrhoz*r12*r21 + r11*(-0.166666666666667*L*intrhoy*r23 + 0.166666666666667*L*intrhoz*r22)
                k += 1
                Mv[k] += 0.166666666666667*L*intrhoy*r21*r23 - 0.166666666666667*L*intrhoz*r21*r22 + r21*(-0.166666666666667*L*intrhoy*r23 + 0.166666666666667*L*intrhoz*r22)
                k += 1
                Mv[k] += 0.166666666666667*L*intrhoy*r21*r33 - 0.166666666666667*L*intrhoz*r21*r32 + r31*(-0.166666666666667*L*intrhoy*r23 + 0.166666666666667*L*intrhoz*r22)
                k += 1
                Mv[k] += 0.166666666666667*L*r11*r21*(intrhoy2 + intrhoz2) + r12*(-0.166666666666667*L*intrhoyz*r23 + 0.166666666666667*L*intrhoz2*r22) + r13*(0.166666666666667*L*intrhoy2*r23 - 0.166666666666667*L*intrhoyz*r22)
                k += 1
                Mv[k] += 0.166666666666667*L*r21**2*(intrhoy2 + intrhoz2) + r22*(-0.166666666666667*L*intrhoyz*r23 + 0.166666666666667*L*intrhoz2*r22) + r23*(0.166666666666667*L*intrhoy2*r23 - 0.166666666666667*L*intrhoyz*r22)
                k += 1
                Mv[k] += 0.166666666666667*L*r21*r31*(intrhoy2 + intrhoz2) + r32*(-0.166666666666667*L*intrhoyz*r23 + 0.166666666666667*L*intrhoz2*r22) + r33*(0.166666666666667*L*intrhoy2*r23 - 0.166666666666667*L*intrhoyz*r22)
                k += 1
                Mv[k] += 0.333333333333333*L*intrhoy*r13*r21 - 0.333333333333333*L*intrhoz*r12*r21 + r11*(-0.333333333333333*L*intrhoy*r23 + 0.333333333333333*L*intrhoz*r22)
                k += 1
                Mv[k] += 0.333333333333333*L*intrhoy*r21*r23 - 0.333333333333333*L*intrhoz*r21*r22 + r21*(-0.333333333333333*L*intrhoy*r23 + 0.333333333333333*L*intrhoz*r22)
                k += 1
                Mv[k] += 0.333333333333333*L*intrhoy*r21*r33 - 0.333333333333333*L*intrhoz*r21*r32 + r31*(-0.333333333333333*L*intrhoy*r23 + 0.333333333333333*L*intrhoz*r22)
                k += 1
                Mv[k] += 0.333333333333333*L*r11*r21*(intrhoy2 + intrhoz2) + r12*(-0.333333333333333*L*intrhoyz*r23 + 0.333333333333333*L*intrhoz2*r22) + r13*(0.333333333333333*L*intrhoy2*r23 - 0.333333333333333*L*intrhoyz*r22)
                k += 1
                Mv[k] += 0.333333333333333*L*r21**2*(intrhoy2 + intrhoz2) + r22*(-0.333333333333333*L*intrhoyz*r23 + 0.333333333333333*L*intrhoz2*r22) + r23*(0.333333333333333*L*intrhoy2*r23 - 0.333333333333333*L*intrhoyz*r22)
                k += 1
                Mv[k] += 0.333333333333333*L*r21*r31*(intrhoy2 + intrhoz2) + r32*(-0.333333333333333*L*intrhoyz*r23 + 0.333333333333333*L*intrhoz2*r22) + r33*(0.333333333333333*L*intrhoy2*r23 - 0.333333333333333*L*intrhoyz*r22)
                k += 1
                Mv[k] += 0.166666666666667*L*intrhoy*r13*r31 - 0.166666666666667*L*intrhoz*r12*r31 + r11*(-0.166666666666667*L*intrhoy*r33 + 0.166666666666667*L*intrhoz*r32)
                k += 1
                Mv[k] += 0.166666666666667*L*intrhoy*r23*r31 - 0.166666666666667*L*intrhoz*r22*r31 + r21*(-0.166666666666667*L*intrhoy*r33 + 0.166666666666667*L*intrhoz*r32)
                k += 1
                Mv[k] += 0.166666666666667*L*intrhoy*r31*r33 - 0.166666666666667*L*intrhoz*r31*r32 + r31*(-0.166666666666667*L*intrhoy*r33 + 0.166666666666667*L*intrhoz*r32)
                k += 1
                Mv[k] += 0.166666666666667*L*r11*r31*(intrhoy2 + intrhoz2) + r12*(-0.166666666666667*L*intrhoyz*r33 + 0.166666666666667*L*intrhoz2*r32) + r13*(0.166666666666667*L*intrhoy2*r33 - 0.166666666666667*L*intrhoyz*r32)
                k += 1
                Mv[k] += 0.166666666666667*L*r21*r31*(intrhoy2 + intrhoz2) + r22*(-0.166666666666667*L*intrhoyz*r33 + 0.166666666666667*L*intrhoz2*r32) + r23*(0.166666666666667*L*intrhoy2*r33 - 0.166666666666667*L*intrhoyz*r32)
                k += 1
                Mv[k] += 0.166666666666667*L*r31**2*(intrhoy2 + intrhoz2) + r32*(-0.166666666666667*L*intrhoyz*r33 + 0.166666666666667*L*intrhoz2*r32) + r33*(0.166666666666667*L*intrhoy2*r33 - 0.166666666666667*L*intrhoyz*r32)
                k += 1
                Mv[k] += 0.333333333333333*L*intrhoy*r13*r31 - 0.333333333333333*L*intrhoz*r12*r31 + r11*(-0.333333333333333*L*intrhoy*r33 + 0.333333333333333*L*intrhoz*r32)
                k += 1
                Mv[k] += 0.333333333333333*L*intrhoy*r23*r31 - 0.333333333333333*L*intrhoz*r22*r31 + r21*(-0.333333333333333*L*intrhoy*r33 + 0.333333333333333*L*intrhoz*r32)
                k += 1
                Mv[k] += 0.333333333333333*L*intrhoy*r31*r33 - 0.333333333333333*L*intrhoz*r31*r32 + r31*(-0.333333333333333*L*intrhoy*r33 + 0.333333333333333*L*intrhoz*r32)
                k += 1
                Mv[k] += 0.333333333333333*L*r11*r31*(intrhoy2 + intrhoz2) + r12*(-0.333333333333333*L*intrhoyz*r33 + 0.333333333333333*L*intrhoz2*r32) + r13*(0.333333333333333*L*intrhoy2*r33 - 0.333333333333333*L*intrhoyz*r32)
                k += 1
                Mv[k] += 0.333333333333333*L*r21*r31*(intrhoy2 + intrhoz2) + r22*(-0.333333333333333*L*intrhoyz*r33 + 0.333333333333333*L*intrhoz2*r32) + r23*(0.333333333333333*L*intrhoy2*r33 - 0.333333333333333*L*intrhoyz*r32)
                k += 1
                Mv[k] += 0.333333333333333*L*r31**2*(intrhoy2 + intrhoz2) + r32*(-0.333333333333333*L*intrhoyz*r33 + 0.333333333333333*L*intrhoz2*r32) + r33*(0.333333333333333*L*intrhoy2*r33 - 0.333333333333333*L*intrhoyz*r32)

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

                # NOTE obtained with two-point Gauss-Lobatto quadrature

                k = self.init_k_M
                Mv[k] += L*intrho*r11**2/2 + L*intrho*r12**2/2 + L*intrho*r13**2/2
                k += 1
                Mv[k] += L*intrho*r11*r21/2 + L*intrho*r12*r22/2 + L*intrho*r13*r23/2
                k += 1
                Mv[k] += L*intrho*r11*r31/2 + L*intrho*r12*r32/2 + L*intrho*r13*r33/2
                k += 1
                Mv[k] += L*intrho*r11*r21/2 + L*intrho*r12*r22/2 + L*intrho*r13*r23/2
                k += 1
                Mv[k] += L*intrho*r21**2/2 + L*intrho*r22**2/2 + L*intrho*r23**2/2
                k += 1
                Mv[k] += L*intrho*r21*r31/2 + L*intrho*r22*r32/2 + L*intrho*r23*r33/2
                k += 1
                Mv[k] += L*intrho*r11*r31/2 + L*intrho*r12*r32/2 + L*intrho*r13*r33/2
                k += 1
                Mv[k] += L*intrho*r21*r31/2 + L*intrho*r22*r32/2 + L*intrho*r23*r33/2
                k += 1
                Mv[k] += L*intrho*r31**2/2 + L*intrho*r32**2/2 + L*intrho*r33**2/2
                k += 1
                Mv[k] += L*intrhoy2*r13**2/2 + L*intrhoz2*r12**2/2 + L*r11**2*(intrhoy2 + intrhoz2)/2
                k += 1
                Mv[k] += L*intrhoy2*r13*r23/2 + L*intrhoz2*r12*r22/2 + L*r11*r21*(intrhoy2 + intrhoz2)/2
                k += 1
                Mv[k] += L*intrhoy2*r13*r33/2 + L*intrhoz2*r12*r32/2 + L*r11*r31*(intrhoy2 + intrhoz2)/2
                k += 1
                Mv[k] += L*intrhoy2*r13*r23/2 + L*intrhoz2*r12*r22/2 + L*r11*r21*(intrhoy2 + intrhoz2)/2
                k += 1
                Mv[k] += L*intrhoy2*r23**2/2 + L*intrhoz2*r22**2/2 + L*r21**2*(intrhoy2 + intrhoz2)/2
                k += 1
                Mv[k] += L*intrhoy2*r23*r33/2 + L*intrhoz2*r22*r32/2 + L*r21*r31*(intrhoy2 + intrhoz2)/2
                k += 1
                Mv[k] += L*intrhoy2*r13*r33/2 + L*intrhoz2*r12*r32/2 + L*r11*r31*(intrhoy2 + intrhoz2)/2
                k += 1
                Mv[k] += L*intrhoy2*r23*r33/2 + L*intrhoz2*r22*r32/2 + L*r21*r31*(intrhoy2 + intrhoz2)/2
                k += 1
                Mv[k] += L*intrhoy2*r33**2/2 + L*intrhoz2*r32**2/2 + L*r31**2*(intrhoy2 + intrhoz2)/2
                k += 1
                Mv[k] += L*intrho*r11**2/2 + L*intrho*r12**2/2 + L*intrho*r13**2/2
                k += 1
                Mv[k] += L*intrho*r11*r21/2 + L*intrho*r12*r22/2 + L*intrho*r13*r23/2
                k += 1
                Mv[k] += L*intrho*r11*r31/2 + L*intrho*r12*r32/2 + L*intrho*r13*r33/2
                k += 1
                Mv[k] += L*intrho*r11*r21/2 + L*intrho*r12*r22/2 + L*intrho*r13*r23/2
                k += 1
                Mv[k] += L*intrho*r21**2/2 + L*intrho*r22**2/2 + L*intrho*r23**2/2
                k += 1
                Mv[k] += L*intrho*r21*r31/2 + L*intrho*r22*r32/2 + L*intrho*r23*r33/2
                k += 1
                Mv[k] += L*intrho*r11*r31/2 + L*intrho*r12*r32/2 + L*intrho*r13*r33/2
                k += 1
                Mv[k] += L*intrho*r21*r31/2 + L*intrho*r22*r32/2 + L*intrho*r23*r33/2
                k += 1
                Mv[k] += L*intrho*r31**2/2 + L*intrho*r32**2/2 + L*intrho*r33**2/2
                k += 1
                Mv[k] += L*intrhoy2*r13**2/2 + L*intrhoz2*r12**2/2 + L*r11**2*(intrhoy2 + intrhoz2)/2
                k += 1
                Mv[k] += L*intrhoy2*r13*r23/2 + L*intrhoz2*r12*r22/2 + L*r11*r21*(intrhoy2 + intrhoz2)/2
                k += 1
                Mv[k] += L*intrhoy2*r13*r33/2 + L*intrhoz2*r12*r32/2 + L*r11*r31*(intrhoy2 + intrhoz2)/2
                k += 1
                Mv[k] += L*intrhoy2*r13*r23/2 + L*intrhoz2*r12*r22/2 + L*r11*r21*(intrhoy2 + intrhoz2)/2
                k += 1
                Mv[k] += L*intrhoy2*r23**2/2 + L*intrhoz2*r22**2/2 + L*r21**2*(intrhoy2 + intrhoz2)/2
                k += 1
                Mv[k] += L*intrhoy2*r23*r33/2 + L*intrhoz2*r22*r32/2 + L*r21*r31*(intrhoy2 + intrhoz2)/2
                k += 1
                Mv[k] += L*intrhoy2*r13*r33/2 + L*intrhoz2*r12*r32/2 + L*r11*r31*(intrhoy2 + intrhoz2)/2
                k += 1
                Mv[k] += L*intrhoy2*r23*r33/2 + L*intrhoz2*r22*r32/2 + L*r21*r31*(intrhoy2 + intrhoz2)/2
                k += 1
                Mv[k] += L*intrhoy2*r33**2/2 + L*intrhoz2*r32**2/2 + L*r31**2*(intrhoy2 + intrhoz2)/2
