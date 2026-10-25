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
        in the following order `{u_e}_1, {v_e}_1, {w_e}_1, {{r_x}_e}_1,
        {{r_y}_e}_1, {{r_z}_e}_1, {u_e}_2, {v_e}_2, {w_e}_2, {{r_x}_e}_2,
        {{r_y}_e}_2, {{r_z}_e}_2`. Translations of dimension length and
        rotations. Updated by :meth:`.BeamLR.update_probe_ue`.
    finte, : array-like
        Array of size ``NUM_NODES*DOF=12`` containing the element internal
        forces corresponding to the degrees-of-freedom described by ``ue``,
        forces and moments. Updated by :meth:`.BeamLR.update_probe_finte`.
    BLexx, BLky, BLkz : array-like
        Arrays of size ``NUM_NODES*DOF=12`` with the rows of the linear
        strain-displacement matrix for the axial strain and the two
        curvatures, at the last evaluated integration point.
    Gvx, Gwx : array-like
        Arrays of size ``NUM_NODES*DOF=12`` with the rows giving the slopes
        `v_{,x}` and `w_{,x}`, at the last evaluated integration point.
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
    cdef public double [::1] Gvx
    cdef public double [::1] Gwx
    cdef public double [::1] KCNLve

    def __cinit__(BeamLRProbe self):
        self.xe = np.zeros(NUM_NODES*DOF//2, dtype=np.float64)
        self.ue = np.zeros(NUM_NODES*DOF, dtype=np.float64)
        self.finte = np.zeros(NUM_NODES*DOF, dtype=np.float64)
        self.BLexx = np.zeros(NUM_NODES*DOF, dtype=np.float64)
        self.BLky = np.zeros(NUM_NODES*DOF, dtype=np.float64)
        self.BLkz = np.zeros(NUM_NODES*DOF, dtype=np.float64)
        self.Gvx = np.zeros(NUM_NODES*DOF, dtype=np.float64)
        self.Gwx = np.zeros(NUM_NODES*DOF, dtype=np.float64)
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
        Rotation matrix from local to global coordinates. Dimensionless.
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
    transverse shear moduli ``G12`` and ``G13`` of the planes `xy` and `xz`
    and the torsional stiffness ``GJ`` of :class:`.BeamProp`. The six
    displacements `u, v, w, r_x, r_y, r_z` of the beam axis are interpolated
    linearly between the two nodes, such that the axial strain `u_{,x}`, the
    curvatures `\kappa_y = -{r_z}_{,x}`, `\kappa_z = {r_y}_{,x}` and the
    twist `{r_x}_{,x}` are constant along the element and integrated
    exactly. The terms of the transverse shear strains `\gamma_{xy} = v_{,x}
    - r_z` and `\gamma_{xz} = w_{,x} + r_y`, including their coupling with
    the twist through ``Ay`` and ``Az``, are evaluated at the center of the
    element, i.e. with a reduced one-point integration, which avoids the
    shear locking of the linear interpolation in slender elements. Since the
    curvature is constant in each element, more elements are needed than
    with :class:`.BeamC` for the same accuracy in bending.

    The element does not use the shear parameters `\alpha_y`, `\alpha_z` of
    :class:`.BeamC`, and is therefore an independent check of their sign:
    in ``tests/test_timoshenko_beam_literature.py`` refined meshes of
    :class:`.BeamLR` converge to the same cantilever tip deflection `P
    L^3/(3 E I_{zz}) + P L/(G A)`, Eq. (1.2) of [2]_, as one element of
    :class:`.BeamC`, and to the same exact frequencies of a deep cantilever,
    Table 3.6 of [3]_; in ``tests/test_beam_shear_moduli.py`` they do so in
    each plane with its own modulus when ``G12`` and ``G13`` differ.

    The matrices and the internal forces are computed in element
    coordinates and transformed to global coordinates with the rotation
    matrix of the element, as described in the notes of :class:`.BeamC`.

    References
    ----------
    .. [1] Luo, Y., 2008, “An Efficient 3D Timoshenko Beam Element with
       Consistent Shape Functions,” Adv. Theor. Appl. Mech., 1(3),
       pp. 95–106.
    .. [2] Bathe, K. J., 1996, Finite Element Procedures, Prentice Hall,
       Englewood Cliffs, NJ, Section 1.2, p. 5.
    .. [3] Petyt, M., 2010, Introduction to Finite Element Vibration
       Analysis, 2nd ed., Cambridge University Press, Section 3.9.
       https://doi.org/10.1017/CBO9780511761195

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

        Attributes ``r11,r12,r13,r21,r22,r23,r31,r32,r33`` are updated,
        corresponding to the rotation matrix from local to global coordinates.

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

        The translations and rotations of the two nodes, read from ``u`` at
        the positions ``c1`` to ``c1 + 5`` and ``c2`` to ``c2 + 5``, are
        transformed to the element coordinate system with the transpose of the
        rotation matrix, which must be updated before, see
        :meth:`.update_rotation_matrix`.

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

            # global to local transformation of displacements (T)
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
        system with the transpose of the rotation matrix, which must be updated
        before, see :meth:`.update_rotation_matrix`. Then
        :meth:`.update_length` is called, updating ``length``.

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

            # global to local transformation of coordinates (r)
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
        The linear internal forces are `[K_{C0_e}] \{u_e\}`, with the matrix
        `[K_{C0_e}]` of :meth:`.update_KC0` in element coordinates. With
        ``nonlinear`` different from ``0``, the terms of the von Karman strains
        are added, see :meth:`._update_probe_finte_nonlinear`, such that
        ``finte`` is the gradient of the strain energy with the von Karman
        axial strain, for which the exact Jacobian of the internal forces is
        ``KC0 + KCNL + KG``, see :meth:`.update_KCNL`.

        """
        cdef double *ue
        cdef double *finte
        cdef double L, A, E, G12, G13, GJ, Ay, Az, Iyy, Izz, Iyz

        with nogil:
            L = self.length
            A = prop.A
            E = prop.E
            G12 = prop.get_G12()
            G13 = prop.get_G13()
            GJ = prop.get_GJ()
            Ay = prop.Ay
            Az = prop.Az
            Iyy = prop.Iyy
            Izz = prop.Izz
            Iyz = prop.Iyz

            ue = &self.probe.ue[0]
            finte = &self.probe.finte[0]

            finte[0] = 1.0*E*(A*ue[0] - A*ue[6] + Ay*ue[11] - Ay*ue[5] - Az*ue[10] + Az*ue[4])/L
            finte[1] = 0.5*G12*(A*L*(ue[11] + ue[5]) + 2*A*ue[1] - 2*A*ue[7] - 2*Az*ue[3] + 2*Az*ue[9])/L
            finte[2] = -0.5*G13*(A*L*(ue[10] + ue[4]) - 2*A*ue[2] + 2*A*ue[8] - 2*Ay*ue[3] + 2*Ay*ue[9])/L
            finte[3] = (1.0*Ay*G13*ue[2] - 1.0*Ay*G13*ue[8] - 1.0*Az*G12*ue[1] + 1.0*Az*G12*ue[7] + 1.0*GJ*ue[3] - 1.0*GJ*ue[9] - 0.5*L*(Ay*G13*ue[10] + Ay*G13*ue[4] + Az*G12*ue[11] + Az*G12*ue[5]))/L
            finte[4] = (0.25*A*G13*L**2*(ue[10] + ue[4]) + 1.0*E*(Az*ue[0] - Az*ue[6] - Iyy*ue[10] + Iyy*ue[4] + Iyz*ue[11] - Iyz*ue[5]) - 0.5*G13*L*(A*ue[2] - A*ue[8] + Ay*ue[3] - Ay*ue[9]))/L
            finte[5] = (0.25*A*G12*L**2*(ue[11] + ue[5]) - 1.0*E*(Ay*ue[0] - Ay*ue[6] - Iyz*ue[10] + Iyz*ue[4] + Izz*ue[11] - Izz*ue[5]) + 0.5*G12*L*(A*ue[1] - A*ue[7] - Az*ue[3] + Az*ue[9]))/L
            finte[6] = 1.0*E*(-A*ue[0] + A*ue[6] - Ay*ue[11] + Ay*ue[5] + Az*ue[10] - Az*ue[4])/L
            finte[7] = -0.5*G12*(A*L*(ue[11] + ue[5]) + 2*A*ue[1] - 2*A*ue[7] - 2*Az*ue[3] + 2*Az*ue[9])/L
            finte[8] = 0.5*G13*(A*L*(ue[10] + ue[4]) - 2*A*ue[2] + 2*A*ue[8] - 2*Ay*ue[3] + 2*Ay*ue[9])/L
            finte[9] = (-1.0*Ay*G13*ue[2] + 1.0*Ay*G13*ue[8] + 1.0*Az*G12*ue[1] - 1.0*Az*G12*ue[7] - 1.0*GJ*ue[3] + 1.0*GJ*ue[9] + 0.5*L*(Ay*G13*ue[10] + Ay*G13*ue[4] + Az*G12*ue[11] + Az*G12*ue[5]))/L
            finte[10] = (0.25*A*G13*L**2*(ue[10] + ue[4]) - 1.0*E*(Az*ue[0] - Az*ue[6] - Iyy*ue[10] + Iyy*ue[4] + Iyz*ue[11] - Iyz*ue[5]) - 0.5*G13*L*(A*ue[2] - A*ue[8] + Ay*ue[3] - Ay*ue[9]))/L
            finte[11] = (0.25*A*G12*L**2*(ue[11] + ue[5]) + 1.0*E*(Ay*ue[0] - Ay*ue[6] - Iyz*ue[10] + Iyz*ue[4] + Izz*ue[11] - Izz*ue[5]) + 0.5*G12*L*(A*ue[1] - A*ue[7] - Az*ue[3] + Az*ue[9]))/L

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
        matrix and the attributes ``A``, ``E``, ``G12``, ``G13``, ``GJ``,
        ``Ay``, ``Az``, ``Iyy``, ``Izz`` and ``Iyz`` of ``prop``. ``length``
        and the
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
        cdef double L, A, E, G12, G13, GJ, Ay, Az, Iyy, Izz, Iyz
        cdef double r11, r12, r13, r21, r22, r23, r31, r32, r33

        with nogil:
            L = self.length
            A = prop.A
            E = prop.E
            G12 = prop.get_G12()
            G13 = prop.get_G13()
            GJ = prop.get_GJ()
            Ay = prop.Ay
            Az = prop.Az
            Iyy = prop.Iyy
            Izz = prop.Izz
            Iyz = prop.Iyz

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
            KC0v[k] += 1.0*A*E*r11**2/L + 1.0*A*G12*r12**2/L + 1.0*A*G13*r13**2/L
            k += 1
            KC0v[k] += 1.0*A*E*r11*r21/L + 1.0*A*G12*r12*r22/L + 1.0*A*G13*r13*r23/L
            k += 1
            KC0v[k] += 1.0*A*E*r11*r31/L + 1.0*A*G12*r12*r32/L + 1.0*A*G13*r13*r33/L
            k += 1
            KC0v[k] += r11*(1.0*Ay*G13*r13/L - 1.0*Az*G12*r12/L) + r12*(-0.5*A*G13*r13 + 1.0*Az*E*r11/L) + r13*(0.5*A*G12*r12 - 1.0*Ay*E*r11/L)
            k += 1
            KC0v[k] += r21*(1.0*Ay*G13*r13/L - 1.0*Az*G12*r12/L) + r22*(-0.5*A*G13*r13 + 1.0*Az*E*r11/L) + r23*(0.5*A*G12*r12 - 1.0*Ay*E*r11/L)
            k += 1
            KC0v[k] += r31*(1.0*Ay*G13*r13/L - 1.0*Az*G12*r12/L) + r32*(-0.5*A*G13*r13 + 1.0*Az*E*r11/L) + r33*(0.5*A*G12*r12 - 1.0*Ay*E*r11/L)
            k += 1
            KC0v[k] += -1.0*A*E*r11**2/L - 1.0*A*G12*r12**2/L - 1.0*A*G13*r13**2/L
            k += 1
            KC0v[k] += -1.0*A*E*r11*r21/L - 1.0*A*G12*r12*r22/L - 1.0*A*G13*r13*r23/L
            k += 1
            KC0v[k] += -1.0*A*E*r11*r31/L - 1.0*A*G12*r12*r32/L - 1.0*A*G13*r13*r33/L
            k += 1
            KC0v[k] += r11*(-1.0*Ay*G13*r13/L + 1.0*Az*G12*r12/L) + r12*(-0.5*A*G13*r13 - 1.0*Az*E*r11/L) + r13*(0.5*A*G12*r12 + 1.0*Ay*E*r11/L)
            k += 1
            KC0v[k] += r21*(-1.0*Ay*G13*r13/L + 1.0*Az*G12*r12/L) + r22*(-0.5*A*G13*r13 - 1.0*Az*E*r11/L) + r23*(0.5*A*G12*r12 + 1.0*Ay*E*r11/L)
            k += 1
            KC0v[k] += r31*(-1.0*Ay*G13*r13/L + 1.0*Az*G12*r12/L) + r32*(-0.5*A*G13*r13 - 1.0*Az*E*r11/L) + r33*(0.5*A*G12*r12 + 1.0*Ay*E*r11/L)
            k += 1
            KC0v[k] += 1.0*A*E*r11*r21/L + 1.0*A*G12*r12*r22/L + 1.0*A*G13*r13*r23/L
            k += 1
            KC0v[k] += 1.0*A*E*r21**2/L + 1.0*A*G12*r22**2/L + 1.0*A*G13*r23**2/L
            k += 1
            KC0v[k] += 1.0*A*E*r21*r31/L + 1.0*A*G12*r22*r32/L + 1.0*A*G13*r23*r33/L
            k += 1
            KC0v[k] += r11*(1.0*Ay*G13*r23/L - 1.0*Az*G12*r22/L) + r12*(-0.5*A*G13*r23 + 1.0*Az*E*r21/L) + r13*(0.5*A*G12*r22 - 1.0*Ay*E*r21/L)
            k += 1
            KC0v[k] += r21*(1.0*Ay*G13*r23/L - 1.0*Az*G12*r22/L) + r22*(-0.5*A*G13*r23 + 1.0*Az*E*r21/L) + r23*(0.5*A*G12*r22 - 1.0*Ay*E*r21/L)
            k += 1
            KC0v[k] += r31*(1.0*Ay*G13*r23/L - 1.0*Az*G12*r22/L) + r32*(-0.5*A*G13*r23 + 1.0*Az*E*r21/L) + r33*(0.5*A*G12*r22 - 1.0*Ay*E*r21/L)
            k += 1
            KC0v[k] += -1.0*A*E*r11*r21/L - 1.0*A*G12*r12*r22/L - 1.0*A*G13*r13*r23/L
            k += 1
            KC0v[k] += -1.0*A*E*r21**2/L - 1.0*A*G12*r22**2/L - 1.0*A*G13*r23**2/L
            k += 1
            KC0v[k] += -1.0*A*E*r21*r31/L - 1.0*A*G12*r22*r32/L - 1.0*A*G13*r23*r33/L
            k += 1
            KC0v[k] += r11*(-1.0*Ay*G13*r23/L + 1.0*Az*G12*r22/L) + r12*(-0.5*A*G13*r23 - 1.0*Az*E*r21/L) + r13*(0.5*A*G12*r22 + 1.0*Ay*E*r21/L)
            k += 1
            KC0v[k] += r21*(-1.0*Ay*G13*r23/L + 1.0*Az*G12*r22/L) + r22*(-0.5*A*G13*r23 - 1.0*Az*E*r21/L) + r23*(0.5*A*G12*r22 + 1.0*Ay*E*r21/L)
            k += 1
            KC0v[k] += r31*(-1.0*Ay*G13*r23/L + 1.0*Az*G12*r22/L) + r32*(-0.5*A*G13*r23 - 1.0*Az*E*r21/L) + r33*(0.5*A*G12*r22 + 1.0*Ay*E*r21/L)
            k += 1
            KC0v[k] += 1.0*A*E*r11*r31/L + 1.0*A*G12*r12*r32/L + 1.0*A*G13*r13*r33/L
            k += 1
            KC0v[k] += 1.0*A*E*r21*r31/L + 1.0*A*G12*r22*r32/L + 1.0*A*G13*r23*r33/L
            k += 1
            KC0v[k] += 1.0*A*E*r31**2/L + 1.0*A*G12*r32**2/L + 1.0*A*G13*r33**2/L
            k += 1
            KC0v[k] += r11*(1.0*Ay*G13*r33/L - 1.0*Az*G12*r32/L) + r12*(-0.5*A*G13*r33 + 1.0*Az*E*r31/L) + r13*(0.5*A*G12*r32 - 1.0*Ay*E*r31/L)
            k += 1
            KC0v[k] += r21*(1.0*Ay*G13*r33/L - 1.0*Az*G12*r32/L) + r22*(-0.5*A*G13*r33 + 1.0*Az*E*r31/L) + r23*(0.5*A*G12*r32 - 1.0*Ay*E*r31/L)
            k += 1
            KC0v[k] += r31*(1.0*Ay*G13*r33/L - 1.0*Az*G12*r32/L) + r32*(-0.5*A*G13*r33 + 1.0*Az*E*r31/L) + r33*(0.5*A*G12*r32 - 1.0*Ay*E*r31/L)
            k += 1
            KC0v[k] += -1.0*A*E*r11*r31/L - 1.0*A*G12*r12*r32/L - 1.0*A*G13*r13*r33/L
            k += 1
            KC0v[k] += -1.0*A*E*r21*r31/L - 1.0*A*G12*r22*r32/L - 1.0*A*G13*r23*r33/L
            k += 1
            KC0v[k] += -1.0*A*E*r31**2/L - 1.0*A*G12*r32**2/L - 1.0*A*G13*r33**2/L
            k += 1
            KC0v[k] += r11*(-1.0*Ay*G13*r33/L + 1.0*Az*G12*r32/L) + r12*(-0.5*A*G13*r33 - 1.0*Az*E*r31/L) + r13*(0.5*A*G12*r32 + 1.0*Ay*E*r31/L)
            k += 1
            KC0v[k] += r21*(-1.0*Ay*G13*r33/L + 1.0*Az*G12*r32/L) + r22*(-0.5*A*G13*r33 - 1.0*Az*E*r31/L) + r23*(0.5*A*G12*r32 + 1.0*Ay*E*r31/L)
            k += 1
            KC0v[k] += r31*(-1.0*Ay*G13*r33/L + 1.0*Az*G12*r32/L) + r32*(-0.5*A*G13*r33 - 1.0*Az*E*r31/L) + r33*(0.5*A*G12*r32 + 1.0*Ay*E*r31/L)
            k += 1
            KC0v[k] += r11*(-1.0*Ay*E*r13/L + 1.0*Az*E*r12/L) + r12*(0.5*A*G12*r13 - 1.0*Az*G12*r11/L) + r13*(-0.5*A*G13*r12 + 1.0*Ay*G13*r11/L)
            k += 1
            KC0v[k] += r21*(-1.0*Ay*E*r13/L + 1.0*Az*E*r12/L) + r22*(0.5*A*G12*r13 - 1.0*Az*G12*r11/L) + r23*(-0.5*A*G13*r12 + 1.0*Ay*G13*r11/L)
            k += 1
            KC0v[k] += r31*(-1.0*Ay*E*r13/L + 1.0*Az*E*r12/L) + r32*(0.5*A*G12*r13 - 1.0*Az*G12*r11/L) + r33*(-0.5*A*G13*r12 + 1.0*Ay*G13*r11/L)
            k += 1
            KC0v[k] += r11*(-0.5*Ay*G13*r12 - 0.5*Az*G12*r13 + 1.0*GJ*r11/L) + r12*(-0.5*Ay*G13*r11 - 1.0*E*Iyz*r13/L + 1.0*L*r12*(A*G13/4 + E*Iyy/L**2)) + r13*(-0.5*Az*G12*r11 - 1.0*E*Iyz*r12/L + 1.0*L*r13*(A*G12/4 + E*Izz/L**2))
            k += 1
            KC0v[k] += r21*(-0.5*Ay*G13*r12 - 0.5*Az*G12*r13 + 1.0*GJ*r11/L) + r22*(-0.5*Ay*G13*r11 - 1.0*E*Iyz*r13/L + 1.0*L*r12*(A*G13/4 + E*Iyy/L**2)) + r23*(-0.5*Az*G12*r11 - 1.0*E*Iyz*r12/L + 1.0*L*r13*(A*G12/4 + E*Izz/L**2))
            k += 1
            KC0v[k] += r31*(-0.5*Ay*G13*r12 - 0.5*Az*G12*r13 + 1.0*GJ*r11/L) + r32*(-0.5*Ay*G13*r11 - 1.0*E*Iyz*r13/L + 1.0*L*r12*(A*G13/4 + E*Iyy/L**2)) + r33*(-0.5*Az*G12*r11 - 1.0*E*Iyz*r12/L + 1.0*L*r13*(A*G12/4 + E*Izz/L**2))
            k += 1
            KC0v[k] += r11*(1.0*Ay*E*r13/L - 1.0*Az*E*r12/L) + r12*(-0.5*A*G12*r13 + 1.0*Az*G12*r11/L) + r13*(0.5*A*G13*r12 - 1.0*Ay*G13*r11/L)
            k += 1
            KC0v[k] += r21*(1.0*Ay*E*r13/L - 1.0*Az*E*r12/L) + r22*(-0.5*A*G12*r13 + 1.0*Az*G12*r11/L) + r23*(0.5*A*G13*r12 - 1.0*Ay*G13*r11/L)
            k += 1
            KC0v[k] += r31*(1.0*Ay*E*r13/L - 1.0*Az*E*r12/L) + r32*(-0.5*A*G12*r13 + 1.0*Az*G12*r11/L) + r33*(0.5*A*G13*r12 - 1.0*Ay*G13*r11/L)
            k += 1
            KC0v[k] += r11*(0.5*Ay*G13*r12 + 0.5*Az*G12*r13 - 1.0*GJ*r11/L) + r12*(-0.5*Ay*G13*r11 + 1.0*E*Iyz*r13/L + 1.0*L*r12*(A*G13/4 - E*Iyy/L**2)) + r13*(-0.5*Az*G12*r11 + 1.0*E*Iyz*r12/L + 1.0*L*r13*(A*G12/4 - E*Izz/L**2))
            k += 1
            KC0v[k] += r21*(0.5*Ay*G13*r12 + 0.5*Az*G12*r13 - 1.0*GJ*r11/L) + r22*(-0.5*Ay*G13*r11 + 1.0*E*Iyz*r13/L + 1.0*L*r12*(A*G13/4 - E*Iyy/L**2)) + r23*(-0.5*Az*G12*r11 + 1.0*E*Iyz*r12/L + 1.0*L*r13*(A*G12/4 - E*Izz/L**2))
            k += 1
            KC0v[k] += r31*(0.5*Ay*G13*r12 + 0.5*Az*G12*r13 - 1.0*GJ*r11/L) + r32*(-0.5*Ay*G13*r11 + 1.0*E*Iyz*r13/L + 1.0*L*r12*(A*G13/4 - E*Iyy/L**2)) + r33*(-0.5*Az*G12*r11 + 1.0*E*Iyz*r12/L + 1.0*L*r13*(A*G12/4 - E*Izz/L**2))
            k += 1
            KC0v[k] += r11*(-1.0*Ay*E*r23/L + 1.0*Az*E*r22/L) + r12*(0.5*A*G12*r23 - 1.0*Az*G12*r21/L) + r13*(-0.5*A*G13*r22 + 1.0*Ay*G13*r21/L)
            k += 1
            KC0v[k] += r21*(-1.0*Ay*E*r23/L + 1.0*Az*E*r22/L) + r22*(0.5*A*G12*r23 - 1.0*Az*G12*r21/L) + r23*(-0.5*A*G13*r22 + 1.0*Ay*G13*r21/L)
            k += 1
            KC0v[k] += r31*(-1.0*Ay*E*r23/L + 1.0*Az*E*r22/L) + r32*(0.5*A*G12*r23 - 1.0*Az*G12*r21/L) + r33*(-0.5*A*G13*r22 + 1.0*Ay*G13*r21/L)
            k += 1
            KC0v[k] += r11*(-0.5*Ay*G13*r22 - 0.5*Az*G12*r23 + 1.0*GJ*r21/L) + r12*(-0.5*Ay*G13*r21 - 1.0*E*Iyz*r23/L + 1.0*L*r22*(A*G13/4 + E*Iyy/L**2)) + r13*(-0.5*Az*G12*r21 - 1.0*E*Iyz*r22/L + 1.0*L*r23*(A*G12/4 + E*Izz/L**2))
            k += 1
            KC0v[k] += r21*(-0.5*Ay*G13*r22 - 0.5*Az*G12*r23 + 1.0*GJ*r21/L) + r22*(-0.5*Ay*G13*r21 - 1.0*E*Iyz*r23/L + 1.0*L*r22*(A*G13/4 + E*Iyy/L**2)) + r23*(-0.5*Az*G12*r21 - 1.0*E*Iyz*r22/L + 1.0*L*r23*(A*G12/4 + E*Izz/L**2))
            k += 1
            KC0v[k] += r31*(-0.5*Ay*G13*r22 - 0.5*Az*G12*r23 + 1.0*GJ*r21/L) + r32*(-0.5*Ay*G13*r21 - 1.0*E*Iyz*r23/L + 1.0*L*r22*(A*G13/4 + E*Iyy/L**2)) + r33*(-0.5*Az*G12*r21 - 1.0*E*Iyz*r22/L + 1.0*L*r23*(A*G12/4 + E*Izz/L**2))
            k += 1
            KC0v[k] += r11*(1.0*Ay*E*r23/L - 1.0*Az*E*r22/L) + r12*(-0.5*A*G12*r23 + 1.0*Az*G12*r21/L) + r13*(0.5*A*G13*r22 - 1.0*Ay*G13*r21/L)
            k += 1
            KC0v[k] += r21*(1.0*Ay*E*r23/L - 1.0*Az*E*r22/L) + r22*(-0.5*A*G12*r23 + 1.0*Az*G12*r21/L) + r23*(0.5*A*G13*r22 - 1.0*Ay*G13*r21/L)
            k += 1
            KC0v[k] += r31*(1.0*Ay*E*r23/L - 1.0*Az*E*r22/L) + r32*(-0.5*A*G12*r23 + 1.0*Az*G12*r21/L) + r33*(0.5*A*G13*r22 - 1.0*Ay*G13*r21/L)
            k += 1
            KC0v[k] += r11*(0.5*Ay*G13*r22 + 0.5*Az*G12*r23 - 1.0*GJ*r21/L) + r12*(-0.5*Ay*G13*r21 + 1.0*E*Iyz*r23/L + 1.0*L*r22*(A*G13/4 - E*Iyy/L**2)) + r13*(-0.5*Az*G12*r21 + 1.0*E*Iyz*r22/L + 1.0*L*r23*(A*G12/4 - E*Izz/L**2))
            k += 1
            KC0v[k] += r21*(0.5*Ay*G13*r22 + 0.5*Az*G12*r23 - 1.0*GJ*r21/L) + r22*(-0.5*Ay*G13*r21 + 1.0*E*Iyz*r23/L + 1.0*L*r22*(A*G13/4 - E*Iyy/L**2)) + r23*(-0.5*Az*G12*r21 + 1.0*E*Iyz*r22/L + 1.0*L*r23*(A*G12/4 - E*Izz/L**2))
            k += 1
            KC0v[k] += r31*(0.5*Ay*G13*r22 + 0.5*Az*G12*r23 - 1.0*GJ*r21/L) + r32*(-0.5*Ay*G13*r21 + 1.0*E*Iyz*r23/L + 1.0*L*r22*(A*G13/4 - E*Iyy/L**2)) + r33*(-0.5*Az*G12*r21 + 1.0*E*Iyz*r22/L + 1.0*L*r23*(A*G12/4 - E*Izz/L**2))
            k += 1
            KC0v[k] += r11*(-1.0*Ay*E*r33/L + 1.0*Az*E*r32/L) + r12*(0.5*A*G12*r33 - 1.0*Az*G12*r31/L) + r13*(-0.5*A*G13*r32 + 1.0*Ay*G13*r31/L)
            k += 1
            KC0v[k] += r21*(-1.0*Ay*E*r33/L + 1.0*Az*E*r32/L) + r22*(0.5*A*G12*r33 - 1.0*Az*G12*r31/L) + r23*(-0.5*A*G13*r32 + 1.0*Ay*G13*r31/L)
            k += 1
            KC0v[k] += r31*(-1.0*Ay*E*r33/L + 1.0*Az*E*r32/L) + r32*(0.5*A*G12*r33 - 1.0*Az*G12*r31/L) + r33*(-0.5*A*G13*r32 + 1.0*Ay*G13*r31/L)
            k += 1
            KC0v[k] += r11*(-0.5*Ay*G13*r32 - 0.5*Az*G12*r33 + 1.0*GJ*r31/L) + r12*(-0.5*Ay*G13*r31 - 1.0*E*Iyz*r33/L + 1.0*L*r32*(A*G13/4 + E*Iyy/L**2)) + r13*(-0.5*Az*G12*r31 - 1.0*E*Iyz*r32/L + 1.0*L*r33*(A*G12/4 + E*Izz/L**2))
            k += 1
            KC0v[k] += r21*(-0.5*Ay*G13*r32 - 0.5*Az*G12*r33 + 1.0*GJ*r31/L) + r22*(-0.5*Ay*G13*r31 - 1.0*E*Iyz*r33/L + 1.0*L*r32*(A*G13/4 + E*Iyy/L**2)) + r23*(-0.5*Az*G12*r31 - 1.0*E*Iyz*r32/L + 1.0*L*r33*(A*G12/4 + E*Izz/L**2))
            k += 1
            KC0v[k] += r31*(-0.5*Ay*G13*r32 - 0.5*Az*G12*r33 + 1.0*GJ*r31/L) + r32*(-0.5*Ay*G13*r31 - 1.0*E*Iyz*r33/L + 1.0*L*r32*(A*G13/4 + E*Iyy/L**2)) + r33*(-0.5*Az*G12*r31 - 1.0*E*Iyz*r32/L + 1.0*L*r33*(A*G12/4 + E*Izz/L**2))
            k += 1
            KC0v[k] += r11*(1.0*Ay*E*r33/L - 1.0*Az*E*r32/L) + r12*(-0.5*A*G12*r33 + 1.0*Az*G12*r31/L) + r13*(0.5*A*G13*r32 - 1.0*Ay*G13*r31/L)
            k += 1
            KC0v[k] += r21*(1.0*Ay*E*r33/L - 1.0*Az*E*r32/L) + r22*(-0.5*A*G12*r33 + 1.0*Az*G12*r31/L) + r23*(0.5*A*G13*r32 - 1.0*Ay*G13*r31/L)
            k += 1
            KC0v[k] += r31*(1.0*Ay*E*r33/L - 1.0*Az*E*r32/L) + r32*(-0.5*A*G12*r33 + 1.0*Az*G12*r31/L) + r33*(0.5*A*G13*r32 - 1.0*Ay*G13*r31/L)
            k += 1
            KC0v[k] += r11*(0.5*Ay*G13*r32 + 0.5*Az*G12*r33 - 1.0*GJ*r31/L) + r12*(-0.5*Ay*G13*r31 + 1.0*E*Iyz*r33/L + 1.0*L*r32*(A*G13/4 - E*Iyy/L**2)) + r13*(-0.5*Az*G12*r31 + 1.0*E*Iyz*r32/L + 1.0*L*r33*(A*G12/4 - E*Izz/L**2))
            k += 1
            KC0v[k] += r21*(0.5*Ay*G13*r32 + 0.5*Az*G12*r33 - 1.0*GJ*r31/L) + r22*(-0.5*Ay*G13*r31 + 1.0*E*Iyz*r33/L + 1.0*L*r32*(A*G13/4 - E*Iyy/L**2)) + r23*(-0.5*Az*G12*r31 + 1.0*E*Iyz*r32/L + 1.0*L*r33*(A*G12/4 - E*Izz/L**2))
            k += 1
            KC0v[k] += r31*(0.5*Ay*G13*r32 + 0.5*Az*G12*r33 - 1.0*GJ*r31/L) + r32*(-0.5*Ay*G13*r31 + 1.0*E*Iyz*r33/L + 1.0*L*r32*(A*G13/4 - E*Iyy/L**2)) + r33*(-0.5*Az*G12*r31 + 1.0*E*Iyz*r32/L + 1.0*L*r33*(A*G12/4 - E*Izz/L**2))
            k += 1
            KC0v[k] += -1.0*A*E*r11**2/L - 1.0*A*G12*r12**2/L - 1.0*A*G13*r13**2/L
            k += 1
            KC0v[k] += -1.0*A*E*r11*r21/L - 1.0*A*G12*r12*r22/L - 1.0*A*G13*r13*r23/L
            k += 1
            KC0v[k] += -1.0*A*E*r11*r31/L - 1.0*A*G12*r12*r32/L - 1.0*A*G13*r13*r33/L
            k += 1
            KC0v[k] += r11*(-1.0*Ay*G13*r13/L + 1.0*Az*G12*r12/L) + r12*(0.5*A*G13*r13 - 1.0*Az*E*r11/L) + r13*(-0.5*A*G12*r12 + 1.0*Ay*E*r11/L)
            k += 1
            KC0v[k] += r21*(-1.0*Ay*G13*r13/L + 1.0*Az*G12*r12/L) + r22*(0.5*A*G13*r13 - 1.0*Az*E*r11/L) + r23*(-0.5*A*G12*r12 + 1.0*Ay*E*r11/L)
            k += 1
            KC0v[k] += r31*(-1.0*Ay*G13*r13/L + 1.0*Az*G12*r12/L) + r32*(0.5*A*G13*r13 - 1.0*Az*E*r11/L) + r33*(-0.5*A*G12*r12 + 1.0*Ay*E*r11/L)
            k += 1
            KC0v[k] += 1.0*A*E*r11**2/L + 1.0*A*G12*r12**2/L + 1.0*A*G13*r13**2/L
            k += 1
            KC0v[k] += 1.0*A*E*r11*r21/L + 1.0*A*G12*r12*r22/L + 1.0*A*G13*r13*r23/L
            k += 1
            KC0v[k] += 1.0*A*E*r11*r31/L + 1.0*A*G12*r12*r32/L + 1.0*A*G13*r13*r33/L
            k += 1
            KC0v[k] += r11*(1.0*Ay*G13*r13/L - 1.0*Az*G12*r12/L) + r12*(0.5*A*G13*r13 + 1.0*Az*E*r11/L) + r13*(-0.5*A*G12*r12 - 1.0*Ay*E*r11/L)
            k += 1
            KC0v[k] += r21*(1.0*Ay*G13*r13/L - 1.0*Az*G12*r12/L) + r22*(0.5*A*G13*r13 + 1.0*Az*E*r11/L) + r23*(-0.5*A*G12*r12 - 1.0*Ay*E*r11/L)
            k += 1
            KC0v[k] += r31*(1.0*Ay*G13*r13/L - 1.0*Az*G12*r12/L) + r32*(0.5*A*G13*r13 + 1.0*Az*E*r11/L) + r33*(-0.5*A*G12*r12 - 1.0*Ay*E*r11/L)
            k += 1
            KC0v[k] += -1.0*A*E*r11*r21/L - 1.0*A*G12*r12*r22/L - 1.0*A*G13*r13*r23/L
            k += 1
            KC0v[k] += -1.0*A*E*r21**2/L - 1.0*A*G12*r22**2/L - 1.0*A*G13*r23**2/L
            k += 1
            KC0v[k] += -1.0*A*E*r21*r31/L - 1.0*A*G12*r22*r32/L - 1.0*A*G13*r23*r33/L
            k += 1
            KC0v[k] += r11*(-1.0*Ay*G13*r23/L + 1.0*Az*G12*r22/L) + r12*(0.5*A*G13*r23 - 1.0*Az*E*r21/L) + r13*(-0.5*A*G12*r22 + 1.0*Ay*E*r21/L)
            k += 1
            KC0v[k] += r21*(-1.0*Ay*G13*r23/L + 1.0*Az*G12*r22/L) + r22*(0.5*A*G13*r23 - 1.0*Az*E*r21/L) + r23*(-0.5*A*G12*r22 + 1.0*Ay*E*r21/L)
            k += 1
            KC0v[k] += r31*(-1.0*Ay*G13*r23/L + 1.0*Az*G12*r22/L) + r32*(0.5*A*G13*r23 - 1.0*Az*E*r21/L) + r33*(-0.5*A*G12*r22 + 1.0*Ay*E*r21/L)
            k += 1
            KC0v[k] += 1.0*A*E*r11*r21/L + 1.0*A*G12*r12*r22/L + 1.0*A*G13*r13*r23/L
            k += 1
            KC0v[k] += 1.0*A*E*r21**2/L + 1.0*A*G12*r22**2/L + 1.0*A*G13*r23**2/L
            k += 1
            KC0v[k] += 1.0*A*E*r21*r31/L + 1.0*A*G12*r22*r32/L + 1.0*A*G13*r23*r33/L
            k += 1
            KC0v[k] += r11*(1.0*Ay*G13*r23/L - 1.0*Az*G12*r22/L) + r12*(0.5*A*G13*r23 + 1.0*Az*E*r21/L) + r13*(-0.5*A*G12*r22 - 1.0*Ay*E*r21/L)
            k += 1
            KC0v[k] += r21*(1.0*Ay*G13*r23/L - 1.0*Az*G12*r22/L) + r22*(0.5*A*G13*r23 + 1.0*Az*E*r21/L) + r23*(-0.5*A*G12*r22 - 1.0*Ay*E*r21/L)
            k += 1
            KC0v[k] += r31*(1.0*Ay*G13*r23/L - 1.0*Az*G12*r22/L) + r32*(0.5*A*G13*r23 + 1.0*Az*E*r21/L) + r33*(-0.5*A*G12*r22 - 1.0*Ay*E*r21/L)
            k += 1
            KC0v[k] += -1.0*A*E*r11*r31/L - 1.0*A*G12*r12*r32/L - 1.0*A*G13*r13*r33/L
            k += 1
            KC0v[k] += -1.0*A*E*r21*r31/L - 1.0*A*G12*r22*r32/L - 1.0*A*G13*r23*r33/L
            k += 1
            KC0v[k] += -1.0*A*E*r31**2/L - 1.0*A*G12*r32**2/L - 1.0*A*G13*r33**2/L
            k += 1
            KC0v[k] += r11*(-1.0*Ay*G13*r33/L + 1.0*Az*G12*r32/L) + r12*(0.5*A*G13*r33 - 1.0*Az*E*r31/L) + r13*(-0.5*A*G12*r32 + 1.0*Ay*E*r31/L)
            k += 1
            KC0v[k] += r21*(-1.0*Ay*G13*r33/L + 1.0*Az*G12*r32/L) + r22*(0.5*A*G13*r33 - 1.0*Az*E*r31/L) + r23*(-0.5*A*G12*r32 + 1.0*Ay*E*r31/L)
            k += 1
            KC0v[k] += r31*(-1.0*Ay*G13*r33/L + 1.0*Az*G12*r32/L) + r32*(0.5*A*G13*r33 - 1.0*Az*E*r31/L) + r33*(-0.5*A*G12*r32 + 1.0*Ay*E*r31/L)
            k += 1
            KC0v[k] += 1.0*A*E*r11*r31/L + 1.0*A*G12*r12*r32/L + 1.0*A*G13*r13*r33/L
            k += 1
            KC0v[k] += 1.0*A*E*r21*r31/L + 1.0*A*G12*r22*r32/L + 1.0*A*G13*r23*r33/L
            k += 1
            KC0v[k] += 1.0*A*E*r31**2/L + 1.0*A*G12*r32**2/L + 1.0*A*G13*r33**2/L
            k += 1
            KC0v[k] += r11*(1.0*Ay*G13*r33/L - 1.0*Az*G12*r32/L) + r12*(0.5*A*G13*r33 + 1.0*Az*E*r31/L) + r13*(-0.5*A*G12*r32 - 1.0*Ay*E*r31/L)
            k += 1
            KC0v[k] += r21*(1.0*Ay*G13*r33/L - 1.0*Az*G12*r32/L) + r22*(0.5*A*G13*r33 + 1.0*Az*E*r31/L) + r23*(-0.5*A*G12*r32 - 1.0*Ay*E*r31/L)
            k += 1
            KC0v[k] += r31*(1.0*Ay*G13*r33/L - 1.0*Az*G12*r32/L) + r32*(0.5*A*G13*r33 + 1.0*Az*E*r31/L) + r33*(-0.5*A*G12*r32 - 1.0*Ay*E*r31/L)
            k += 1
            KC0v[k] += r11*(1.0*Ay*E*r13/L - 1.0*Az*E*r12/L) + r12*(0.5*A*G12*r13 + 1.0*Az*G12*r11/L) + r13*(-0.5*A*G13*r12 - 1.0*Ay*G13*r11/L)
            k += 1
            KC0v[k] += r21*(1.0*Ay*E*r13/L - 1.0*Az*E*r12/L) + r22*(0.5*A*G12*r13 + 1.0*Az*G12*r11/L) + r23*(-0.5*A*G13*r12 - 1.0*Ay*G13*r11/L)
            k += 1
            KC0v[k] += r31*(1.0*Ay*E*r13/L - 1.0*Az*E*r12/L) + r32*(0.5*A*G12*r13 + 1.0*Az*G12*r11/L) + r33*(-0.5*A*G13*r12 - 1.0*Ay*G13*r11/L)
            k += 1
            KC0v[k] += r11*(-0.5*Ay*G13*r12 - 0.5*Az*G12*r13 - 1.0*GJ*r11/L) + r12*(0.5*Ay*G13*r11 + 1.0*E*Iyz*r13/L + 1.0*L*r12*(A*G13/4 - E*Iyy/L**2)) + r13*(0.5*Az*G12*r11 + 1.0*E*Iyz*r12/L + 1.0*L*r13*(A*G12/4 - E*Izz/L**2))
            k += 1
            KC0v[k] += r21*(-0.5*Ay*G13*r12 - 0.5*Az*G12*r13 - 1.0*GJ*r11/L) + r22*(0.5*Ay*G13*r11 + 1.0*E*Iyz*r13/L + 1.0*L*r12*(A*G13/4 - E*Iyy/L**2)) + r23*(0.5*Az*G12*r11 + 1.0*E*Iyz*r12/L + 1.0*L*r13*(A*G12/4 - E*Izz/L**2))
            k += 1
            KC0v[k] += r31*(-0.5*Ay*G13*r12 - 0.5*Az*G12*r13 - 1.0*GJ*r11/L) + r32*(0.5*Ay*G13*r11 + 1.0*E*Iyz*r13/L + 1.0*L*r12*(A*G13/4 - E*Iyy/L**2)) + r33*(0.5*Az*G12*r11 + 1.0*E*Iyz*r12/L + 1.0*L*r13*(A*G12/4 - E*Izz/L**2))
            k += 1
            KC0v[k] += r11*(-1.0*Ay*E*r13/L + 1.0*Az*E*r12/L) + r12*(-0.5*A*G12*r13 - 1.0*Az*G12*r11/L) + r13*(0.5*A*G13*r12 + 1.0*Ay*G13*r11/L)
            k += 1
            KC0v[k] += r21*(-1.0*Ay*E*r13/L + 1.0*Az*E*r12/L) + r22*(-0.5*A*G12*r13 - 1.0*Az*G12*r11/L) + r23*(0.5*A*G13*r12 + 1.0*Ay*G13*r11/L)
            k += 1
            KC0v[k] += r31*(-1.0*Ay*E*r13/L + 1.0*Az*E*r12/L) + r32*(-0.5*A*G12*r13 - 1.0*Az*G12*r11/L) + r33*(0.5*A*G13*r12 + 1.0*Ay*G13*r11/L)
            k += 1
            KC0v[k] += r11*(0.5*Ay*G13*r12 + 0.5*Az*G12*r13 + 1.0*GJ*r11/L) + r12*(0.5*Ay*G13*r11 - 1.0*E*Iyz*r13/L + 1.0*L*r12*(A*G13/4 + E*Iyy/L**2)) + r13*(0.5*Az*G12*r11 - 1.0*E*Iyz*r12/L + 1.0*L*r13*(A*G12/4 + E*Izz/L**2))
            k += 1
            KC0v[k] += r21*(0.5*Ay*G13*r12 + 0.5*Az*G12*r13 + 1.0*GJ*r11/L) + r22*(0.5*Ay*G13*r11 - 1.0*E*Iyz*r13/L + 1.0*L*r12*(A*G13/4 + E*Iyy/L**2)) + r23*(0.5*Az*G12*r11 - 1.0*E*Iyz*r12/L + 1.0*L*r13*(A*G12/4 + E*Izz/L**2))
            k += 1
            KC0v[k] += r31*(0.5*Ay*G13*r12 + 0.5*Az*G12*r13 + 1.0*GJ*r11/L) + r32*(0.5*Ay*G13*r11 - 1.0*E*Iyz*r13/L + 1.0*L*r12*(A*G13/4 + E*Iyy/L**2)) + r33*(0.5*Az*G12*r11 - 1.0*E*Iyz*r12/L + 1.0*L*r13*(A*G12/4 + E*Izz/L**2))
            k += 1
            KC0v[k] += r11*(1.0*Ay*E*r23/L - 1.0*Az*E*r22/L) + r12*(0.5*A*G12*r23 + 1.0*Az*G12*r21/L) + r13*(-0.5*A*G13*r22 - 1.0*Ay*G13*r21/L)
            k += 1
            KC0v[k] += r21*(1.0*Ay*E*r23/L - 1.0*Az*E*r22/L) + r22*(0.5*A*G12*r23 + 1.0*Az*G12*r21/L) + r23*(-0.5*A*G13*r22 - 1.0*Ay*G13*r21/L)
            k += 1
            KC0v[k] += r31*(1.0*Ay*E*r23/L - 1.0*Az*E*r22/L) + r32*(0.5*A*G12*r23 + 1.0*Az*G12*r21/L) + r33*(-0.5*A*G13*r22 - 1.0*Ay*G13*r21/L)
            k += 1
            KC0v[k] += r11*(-0.5*Ay*G13*r22 - 0.5*Az*G12*r23 - 1.0*GJ*r21/L) + r12*(0.5*Ay*G13*r21 + 1.0*E*Iyz*r23/L + 1.0*L*r22*(A*G13/4 - E*Iyy/L**2)) + r13*(0.5*Az*G12*r21 + 1.0*E*Iyz*r22/L + 1.0*L*r23*(A*G12/4 - E*Izz/L**2))
            k += 1
            KC0v[k] += r21*(-0.5*Ay*G13*r22 - 0.5*Az*G12*r23 - 1.0*GJ*r21/L) + r22*(0.5*Ay*G13*r21 + 1.0*E*Iyz*r23/L + 1.0*L*r22*(A*G13/4 - E*Iyy/L**2)) + r23*(0.5*Az*G12*r21 + 1.0*E*Iyz*r22/L + 1.0*L*r23*(A*G12/4 - E*Izz/L**2))
            k += 1
            KC0v[k] += r31*(-0.5*Ay*G13*r22 - 0.5*Az*G12*r23 - 1.0*GJ*r21/L) + r32*(0.5*Ay*G13*r21 + 1.0*E*Iyz*r23/L + 1.0*L*r22*(A*G13/4 - E*Iyy/L**2)) + r33*(0.5*Az*G12*r21 + 1.0*E*Iyz*r22/L + 1.0*L*r23*(A*G12/4 - E*Izz/L**2))
            k += 1
            KC0v[k] += r11*(-1.0*Ay*E*r23/L + 1.0*Az*E*r22/L) + r12*(-0.5*A*G12*r23 - 1.0*Az*G12*r21/L) + r13*(0.5*A*G13*r22 + 1.0*Ay*G13*r21/L)
            k += 1
            KC0v[k] += r21*(-1.0*Ay*E*r23/L + 1.0*Az*E*r22/L) + r22*(-0.5*A*G12*r23 - 1.0*Az*G12*r21/L) + r23*(0.5*A*G13*r22 + 1.0*Ay*G13*r21/L)
            k += 1
            KC0v[k] += r31*(-1.0*Ay*E*r23/L + 1.0*Az*E*r22/L) + r32*(-0.5*A*G12*r23 - 1.0*Az*G12*r21/L) + r33*(0.5*A*G13*r22 + 1.0*Ay*G13*r21/L)
            k += 1
            KC0v[k] += r11*(0.5*Ay*G13*r22 + 0.5*Az*G12*r23 + 1.0*GJ*r21/L) + r12*(0.5*Ay*G13*r21 - 1.0*E*Iyz*r23/L + 1.0*L*r22*(A*G13/4 + E*Iyy/L**2)) + r13*(0.5*Az*G12*r21 - 1.0*E*Iyz*r22/L + 1.0*L*r23*(A*G12/4 + E*Izz/L**2))
            k += 1
            KC0v[k] += r21*(0.5*Ay*G13*r22 + 0.5*Az*G12*r23 + 1.0*GJ*r21/L) + r22*(0.5*Ay*G13*r21 - 1.0*E*Iyz*r23/L + 1.0*L*r22*(A*G13/4 + E*Iyy/L**2)) + r23*(0.5*Az*G12*r21 - 1.0*E*Iyz*r22/L + 1.0*L*r23*(A*G12/4 + E*Izz/L**2))
            k += 1
            KC0v[k] += r31*(0.5*Ay*G13*r22 + 0.5*Az*G12*r23 + 1.0*GJ*r21/L) + r32*(0.5*Ay*G13*r21 - 1.0*E*Iyz*r23/L + 1.0*L*r22*(A*G13/4 + E*Iyy/L**2)) + r33*(0.5*Az*G12*r21 - 1.0*E*Iyz*r22/L + 1.0*L*r23*(A*G12/4 + E*Izz/L**2))
            k += 1
            KC0v[k] += r11*(1.0*Ay*E*r33/L - 1.0*Az*E*r32/L) + r12*(0.5*A*G12*r33 + 1.0*Az*G12*r31/L) + r13*(-0.5*A*G13*r32 - 1.0*Ay*G13*r31/L)
            k += 1
            KC0v[k] += r21*(1.0*Ay*E*r33/L - 1.0*Az*E*r32/L) + r22*(0.5*A*G12*r33 + 1.0*Az*G12*r31/L) + r23*(-0.5*A*G13*r32 - 1.0*Ay*G13*r31/L)
            k += 1
            KC0v[k] += r31*(1.0*Ay*E*r33/L - 1.0*Az*E*r32/L) + r32*(0.5*A*G12*r33 + 1.0*Az*G12*r31/L) + r33*(-0.5*A*G13*r32 - 1.0*Ay*G13*r31/L)
            k += 1
            KC0v[k] += r11*(-0.5*Ay*G13*r32 - 0.5*Az*G12*r33 - 1.0*GJ*r31/L) + r12*(0.5*Ay*G13*r31 + 1.0*E*Iyz*r33/L + 1.0*L*r32*(A*G13/4 - E*Iyy/L**2)) + r13*(0.5*Az*G12*r31 + 1.0*E*Iyz*r32/L + 1.0*L*r33*(A*G12/4 - E*Izz/L**2))
            k += 1
            KC0v[k] += r21*(-0.5*Ay*G13*r32 - 0.5*Az*G12*r33 - 1.0*GJ*r31/L) + r22*(0.5*Ay*G13*r31 + 1.0*E*Iyz*r33/L + 1.0*L*r32*(A*G13/4 - E*Iyy/L**2)) + r23*(0.5*Az*G12*r31 + 1.0*E*Iyz*r32/L + 1.0*L*r33*(A*G12/4 - E*Izz/L**2))
            k += 1
            KC0v[k] += r31*(-0.5*Ay*G13*r32 - 0.5*Az*G12*r33 - 1.0*GJ*r31/L) + r32*(0.5*Ay*G13*r31 + 1.0*E*Iyz*r33/L + 1.0*L*r32*(A*G13/4 - E*Iyy/L**2)) + r33*(0.5*Az*G12*r31 + 1.0*E*Iyz*r32/L + 1.0*L*r33*(A*G12/4 - E*Izz/L**2))
            k += 1
            KC0v[k] += r11*(-1.0*Ay*E*r33/L + 1.0*Az*E*r32/L) + r12*(-0.5*A*G12*r33 - 1.0*Az*G12*r31/L) + r13*(0.5*A*G13*r32 + 1.0*Ay*G13*r31/L)
            k += 1
            KC0v[k] += r21*(-1.0*Ay*E*r33/L + 1.0*Az*E*r32/L) + r22*(-0.5*A*G12*r33 - 1.0*Az*G12*r31/L) + r23*(0.5*A*G13*r32 + 1.0*Ay*G13*r31/L)
            k += 1
            KC0v[k] += r31*(-1.0*Ay*E*r33/L + 1.0*Az*E*r32/L) + r32*(-0.5*A*G12*r33 - 1.0*Az*G12*r31/L) + r33*(0.5*A*G13*r32 + 1.0*Ay*G13*r31/L)
            k += 1
            KC0v[k] += r11*(0.5*Ay*G13*r32 + 0.5*Az*G12*r33 + 1.0*GJ*r31/L) + r12*(0.5*Ay*G13*r31 - 1.0*E*Iyz*r33/L + 1.0*L*r32*(A*G13/4 + E*Iyy/L**2)) + r13*(0.5*Az*G12*r31 - 1.0*E*Iyz*r32/L + 1.0*L*r33*(A*G12/4 + E*Izz/L**2))
            k += 1
            KC0v[k] += r21*(0.5*Ay*G13*r32 + 0.5*Az*G12*r33 + 1.0*GJ*r31/L) + r22*(0.5*Ay*G13*r31 - 1.0*E*Iyz*r33/L + 1.0*L*r32*(A*G13/4 + E*Iyy/L**2)) + r23*(0.5*Az*G12*r31 - 1.0*E*Iyz*r32/L + 1.0*L*r33*(A*G12/4 + E*Izz/L**2))
            k += 1
            KC0v[k] += r31*(0.5*Ay*G13*r32 + 0.5*Az*G12*r33 + 1.0*GJ*r31/L) + r32*(0.5*Ay*G13*r31 - 1.0*E*Iyz*r33/L + 1.0*L*r32*(A*G13/4 + E*Iyy/L**2)) + r33*(0.5*Az*G12*r31 - 1.0*E*Iyz*r32/L + 1.0*L*r33*(A*G12/4 + E*Izz/L**2))


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
        The nonlinear terms are those of the von Karman strains, for which the
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
        r"""Update the probe rows of the axial strain, curvatures and slopes

        The arrays ``BLexx``, ``BLky``, ``BLkz``, ``Gvx`` and ``Gwx`` of the
        :class:`.BeamLRProbe` are overwritten with the rows that, multiplied by
        the element displacements ``ue``, give in element coordinates the axial
        strain of the beam axis, the two curvatures and the two slopes of the
        axis, evaluated at the natural coordinate ``xi``, from ``-1`` at node 1
        to ``+1`` at node 2. They depend on ``length``.

        Notes
        -----
        The rows are obtained with linear shape functions:

        - ``BLexx``: `u_{,x}`
        - ``BLky``: `\kappa_y = -{r_z}_{,x}`
        - ``BLkz``: `\kappa_z = {r_y}_{,x}`
        - ``Gvx``, ``Gwx``: the slopes `v_{,x}` and `w_{,x}`, taken from the
          rotations as `v_{,x} = r_z` and `w_{,x} = -r_y`, as in
          :meth:`.update_KG`

        """
        cdef int i
        cdef double L
        cdef double *BLexx
        cdef double *BLky
        cdef double *BLkz
        cdef double *Gvx
        cdef double *Gwx

        BLexx = &self.probe.BLexx[0]
        BLky = &self.probe.BLky[0]
        BLkz = &self.probe.BLkz[0]
        Gvx = &self.probe.Gvx[0]
        Gwx = &self.probe.Gwx[0]

        L = self.length

        for i in range(NUM_NODES*DOF):
            BLexx[i] = 0.
            BLky[i] = 0.
            BLkz[i] = 0.
            Gvx[i] = 0.
            Gwx[i] = 0.

        # u,x
        BLexx[0] = -1/L
        BLexx[6] = 1/L

        # ky = -rz,x
        BLky[5] = 1/L
        BLky[11] = -1/L

        # kz = ry,x
        BLkz[4] = -1/L
        BLkz[10] = 1/L

        # v,x = rz
        Gvx[5] = (1 - xi)/2
        Gvx[11] = (1 + xi)/2

        # w,x = -ry
        Gwx[4] = -(1 - xi)/2
        Gwx[10] = -(1 + xi)/2


    cdef void _update_probe_KCNLve(BeamLR self, BeamProp prop) noexcept nogil:
        r"""Update the probe values of the nonlinear constitutive stiffness matrix

        The attribute ``KCNLve`` of the :class:`.BeamLRProbe` is overwritten
        with the nonlinear constitutive stiffness matrix KCNL in element
        coordinates, stored row by row and evaluated at the displacements
        ``ue`` of the probe, from ``length`` and the attributes ``E``, ``A``,
        ``Ay`` and ``Az`` of ``prop``. The arrays ``BLexx``, ``BLky``,
        ``BLkz``, ``Gvx`` and ``Gwx`` of the probe are overwritten too. See
        :meth:`.update_KCNL`.

        Notes
        -----
        KCNL = KC0L + KCL0 + KCLL + KGNL, see :meth:`.update_KCNL`, is
        integrated along the element with the 3-point Gauss-Legendre
        quadrature, exact for these polynomials of degree up to 4 in ``xi``.
        At each point the entries receive the terms given in
        :meth:`.BeamC._update_probe_KCNLve`, with the rows of
        :meth:`._update_probe_BL_G`.

        """
        cdef int i, j, pt
        cdef double L, E, A, Ay, Az, xi, weight, v_x, w_x, NNL
        cdef double points[3]
        cdef double weights[3]
        cdef double EBL[12]
        cdef double BNL[12]
        cdef double *ue
        cdef double *KCNLve
        cdef double *BLexx
        cdef double *BLky
        cdef double *BLkz
        cdef double *Gvx
        cdef double *Gwx

        L = self.length
        E = prop.E
        A = prop.A
        Ay = prop.Ay
        Az = prop.Az

        ue = &self.probe.ue[0]
        KCNLve = &self.probe.KCNLve[0]
        BLexx = &self.probe.BLexx[0]
        BLky = &self.probe.BLky[0]
        BLkz = &self.probe.BLkz[0]
        Gvx = &self.probe.Gvx[0]
        Gwx = &self.probe.Gwx[0]

        for i in range(12*12):
            KCNLve[i] = 0.

        # NOTE 3-point Gauss-Legendre quadrature, exact for these polynomials
        points[0] = -0.774596669241483377035853079956
        weights[0] = 0.555555555555555555555555555556
        points[1] = 0.
        weights[1] = 0.888888888888888888888888888889
        points[2] = +0.774596669241483377035853079956
        weights[2] = 0.555555555555555555555555555556

        for pt in range(3):
            xi = points[pt]
            weight = weights[pt]*L/2.
            self._update_probe_BL_G(xi)

            v_x = 0.
            w_x = 0.
            for i in range(NUM_NODES*DOF):
                v_x += Gvx[i]*ue[i]
                w_x += Gwx[i]*ue[i]

            # axial force of the nonlinear axial strain, eNL = (v_x**2 + w_x**2)/2
            NNL = E*A*(v_x*v_x + w_x*w_x)/2.

            for i in range(NUM_NODES*DOF):
                # row of D*BL giving the axial force N = E*(A*u,x + Ay*ky + Az*kz)
                EBL[i] = E*(A*BLexx[i] + Ay*BLky[i] + Az*BLkz[i])
                # BNL, the variation of the nonlinear axial strain
                BNL[i] = v_x*Gvx[i] + w_x*Gwx[i]

            for i in range(NUM_NODES*DOF):
                for j in range(NUM_NODES*DOF):
                    KCNLve[12*i + j] += weight*(
                        # KC0L = BL.T*D*BNL
                          EBL[i]*BNL[j]
                        # KCL0 = BNL.T*D*BL
                        + BNL[i]*EBL[j]
                        # KCLL = BNL.T*D*BNL
                        + E*A*BNL[i]*BNL[j]
                        # KGNL = NNL*(Gvx.T*Gvx + Gwx.T*Gwx)
                        + NNL*(Gvx[i]*Gvx[j] + Gwx[i]*Gwx[j])
                    )


    cdef void _update_probe_finte_nonlinear(BeamLR self,
                                            BeamProp prop) noexcept nogil:
        r"""Add the geometrically nonlinear terms to the probe internal forces

        The nonlinear terms, evaluated at the displacements ``ue`` of the
        probe, from ``length`` and the attributes ``E``, ``A``, ``Ay`` and
        ``Az`` of ``prop``, are added to the attribute ``finte`` of the
        :class:`.BeamLRProbe`, which must already contain the linear internal
        forces, see :meth:`.update_probe_finte`. The arrays ``BLexx``,
        ``BLky``, ``BLkz``, ``Gvx`` and ``Gwx`` of the probe are overwritten.
        See :meth:`.update_KCNL`.

        Notes
        -----
        The added terms are those of the von Karman axial strain
        `\epsilon_{NL} = (v_{,x}^2 + w_{,x}^2)/2`, with the slopes taken from
        the rotations, such that ``finte`` becomes the gradient of the strain
        energy whose Hessian is KC0 + KCNL + KG. They are integrated with the
        3-point Gauss-Legendre quadrature, each point adding the terms given
        in :meth:`.BeamC._update_probe_finte_nonlinear`.

        """
        cdef int i, pt
        cdef double L, E, A, Ay, Az, xi, weight, v_x, w_x, eNL, N, NNL
        cdef double points[3]
        cdef double weights[3]
        cdef double EBL[12]
        cdef double *ue
        cdef double *finte
        cdef double *BLexx
        cdef double *BLky
        cdef double *BLkz
        cdef double *Gvx
        cdef double *Gwx

        L = self.length
        E = prop.E
        A = prop.A
        Ay = prop.Ay
        Az = prop.Az

        ue = &self.probe.ue[0]
        finte = &self.probe.finte[0]
        BLexx = &self.probe.BLexx[0]
        BLky = &self.probe.BLky[0]
        BLkz = &self.probe.BLkz[0]
        Gvx = &self.probe.Gvx[0]
        Gwx = &self.probe.Gwx[0]

        # NOTE 3-point Gauss-Legendre quadrature, exact for these polynomials
        points[0] = -0.774596669241483377035853079956
        weights[0] = 0.555555555555555555555555555556
        points[1] = 0.
        weights[1] = 0.888888888888888888888888888889
        points[2] = +0.774596669241483377035853079956
        weights[2] = 0.555555555555555555555555555556

        for pt in range(3):
            xi = points[pt]
            weight = weights[pt]*L/2.
            self._update_probe_BL_G(xi)

            v_x = 0.
            w_x = 0.
            N = 0.
            for i in range(NUM_NODES*DOF):
                v_x += Gvx[i]*ue[i]
                w_x += Gwx[i]*ue[i]
                # row of D*BL giving the axial force N = E*(A*u,x + Ay*ky + Az*kz)
                EBL[i] = E*(A*BLexx[i] + Ay*BLky[i] + Az*BLkz[i])
                N += EBL[i]*ue[i]

            # nonlinear axial strain and its axial force
            eNL = (v_x*v_x + w_x*w_x)/2.
            NNL = E*A*eNL

            for i in range(NUM_NODES*DOF):
                finte[i] += weight*(
                    # BL.T*D*{eNL, 0, 0, 0, 0, 0}
                      EBL[i]*eNL
                    # BNL.T*(N + NNL)
                    + (v_x*Gvx[i] + w_x*Gwx[i])*(N + NNL)
                )


    cpdef void update_KCNL(BeamLR self,
                           long [::1] KCNLr,
                           long [::1] KCNLc,
                           double [::1] KCNLv,
                           BeamProp prop,
                           int update_KCNLv_only=0
                           ):
        r"""Update sparse vectors for the nonlinear constitutive stiffness matrix KCNL

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
        Assuming that KCNL = KC0L + KCL0 + KCLL + KGNL, built from the von Karman
        axial strain

        .. math::
            \epsilon_{xx} = u_{,x} + \frac{1}{2} \left( v_{,x}^2 + w_{,x}^2 \right)

        whose nonlinear part is `\epsilon_{NL} = \frac{1}{2} [B_{NL}] \{u_e\}`, with
        `[B_{NL}] = v_{,x} [G_v] + w_{,x} [G_w]` its variation, where the slopes are taken from the rotations as `v_{,x} = r_z` and `w_{,x} = -r_y`. With
        `[B_L]` the linear strain-displacement matrix, `[D]` the constitutive
        matrix, whose first column couples the axial strain with `E A`, `E A_y`
        and `E A_z`, and `N_{NL} = E A \epsilon_{NL}`:

        - KC0L = `[B_L]^T [D] [B_{NL}]`
        - KCL0 = KC0L`^T`
        - KCLL = `[B_{NL}]^T [D] [B_{NL}]`
        - KGNL = `N_{NL} ([G_v]^T [G_v] + [G_w]^T [G_w])`

        The first three groups are the constitutive terms coupling the linear and
        the nonlinear parts of the axial strain. KGNL is geometric, carrying the
        axial force of the nonlinear axial strain. It is collected here so that
        :meth:`.update_KG` stays homogeneous of degree one in the displacements,
        which is what a linear buckling analysis needs. With it here,

        .. math::
            K_T = K_{C0} + K_{CNL}(u) + K_G(u)

        is the exact Jacobian of the internal forces of :meth:`.update_fint` with
        ``nonlinear=1``, and a Newton-Raphson iteration built on them converges
        quadratically.

        The integration is made by :meth:`._update_probe_KCNLve` with the
        3-point Gauss-Legendre quadrature, exact for the linear shape
        functions. The von Karman axial strain is valid for small strains and
        moderate rotations: the nonlinear axial strain `\epsilon_{NL}` is
        uniform over the cross section, the curvatures and the transverse
        shear strains are linear in the displacements, and the element
        coordinate system is not updated with the displacements.

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
            # Kg = T.T @ Ke @ T
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
        displacements ``probe.ue``, from ``length`` and from the attributes
        ``A``, ``E``, ``Ay`` and ``Az`` of ``prop``. ``probe.ue``,
        ``length`` and the rotation matrix must be updated before, see
        :meth:`.update_probe_ue` and :meth:`.update_probe_xe`. The rows and
        columns of the entries are those of the rotations of the two nodes.

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
            [K_G] = \int_0^L N \left( [G_v]^T [G_v] + [G_w]^T [G_w] \right) dx

        the Hessian of `\frac{1}{2} \int_0^L N (v_{,x}^2 + w_{,x}^2) dx` at a
        constant axial force `N`, where the slopes are taken from the
        rotations, `v_{,x} = r_z` and `w_{,x} = -r_y`, interpolated linearly,
        see :meth:`._update_probe_BL_G`, and `N = E (A u_{,x} + A_y \kappa_y
        + A_z \kappa_z)` is the axial force of the linear strains at
        ``probe.ue``, constant along the element, see the notes of
        :class:`.BeamC`. The integral is computed exactly, hence the matrix
        involves only the rotations.

        `[K_G]` is homogeneous of degree one in ``probe.ue``. With ``probe.ue``
        from the linear static solution of a reference load, the critical
        load multipliers `\lambda` of the linear buckling analysis solve
        `([K_{C0}] + \lambda [K_G]) \{u\} = \{0\}`, see
        :func:`.linear_buckling`. The geometric term of the
        nonlinear axial strain is in :meth:`.update_KCNL`.

        """
        cdef double *ue
        cdef int c1, c2, k
        cdef double L, A, E, Ay, Az, N
        cdef double r12, r13, r22, r23, r32, r33

        with nogil:
            L = self.length
            A = prop.A
            E = prop.E
            Ay = prop.Ay
            Az = prop.Az

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
                KGc[k] = 3+c2
                k += 1
                KGr[k] = 3+c1
                KGc[k] = 4+c2
                k += 1
                KGr[k] = 3+c1
                KGc[k] = 5+c2
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
                KGc[k] = 3+c2
                k += 1
                KGr[k] = 4+c1
                KGc[k] = 4+c2
                k += 1
                KGr[k] = 4+c1
                KGc[k] = 5+c2
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
                KGc[k] = 3+c2
                k += 1
                KGr[k] = 5+c1
                KGc[k] = 4+c2
                k += 1
                KGr[k] = 5+c1
                KGc[k] = 5+c2
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
                KGc[k] = 3+c2
                k += 1
                KGr[k] = 3+c2
                KGc[k] = 4+c2
                k += 1
                KGr[k] = 3+c2
                KGc[k] = 5+c2
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
                KGc[k] = 3+c2
                k += 1
                KGr[k] = 4+c2
                KGc[k] = 4+c2
                k += 1
                KGr[k] = 4+c2
                KGc[k] = 5+c2
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
                KGc[k] = 3+c2
                k += 1
                KGr[k] = 5+c2
                KGc[k] = 4+c2
                k += 1
                KGr[k] = 5+c2
                KGc[k] = 5+c2

            N = E*(-A*ue[0] + A*ue[6] - Ay*ue[11] + Ay*ue[5] + Az*ue[10] - Az*ue[4])/L

            k = self.init_k_KG
            KGv[k] += 0.333333333333333*L*N*r12**2 + 0.333333333333333*L*N*r13**2
            k += 1
            KGv[k] += 0.333333333333333*L*N*r12*r22 + 0.333333333333333*L*N*r13*r23
            k += 1
            KGv[k] += 0.333333333333333*L*N*r12*r32 + 0.333333333333333*L*N*r13*r33
            k += 1
            KGv[k] += 0.166666666666667*L*N*r12**2 + 0.166666666666667*L*N*r13**2
            k += 1
            KGv[k] += 0.166666666666667*L*N*r12*r22 + 0.166666666666667*L*N*r13*r23
            k += 1
            KGv[k] += 0.166666666666667*L*N*r12*r32 + 0.166666666666667*L*N*r13*r33
            k += 1
            KGv[k] += 0.333333333333333*L*N*r12*r22 + 0.333333333333333*L*N*r13*r23
            k += 1
            KGv[k] += 0.333333333333333*L*N*r22**2 + 0.333333333333333*L*N*r23**2
            k += 1
            KGv[k] += 0.333333333333333*L*N*r22*r32 + 0.333333333333333*L*N*r23*r33
            k += 1
            KGv[k] += 0.166666666666667*L*N*r12*r22 + 0.166666666666667*L*N*r13*r23
            k += 1
            KGv[k] += 0.166666666666667*L*N*r22**2 + 0.166666666666667*L*N*r23**2
            k += 1
            KGv[k] += 0.166666666666667*L*N*r22*r32 + 0.166666666666667*L*N*r23*r33
            k += 1
            KGv[k] += 0.333333333333333*L*N*r12*r32 + 0.333333333333333*L*N*r13*r33
            k += 1
            KGv[k] += 0.333333333333333*L*N*r22*r32 + 0.333333333333333*L*N*r23*r33
            k += 1
            KGv[k] += 0.333333333333333*L*N*r32**2 + 0.333333333333333*L*N*r33**2
            k += 1
            KGv[k] += 0.166666666666667*L*N*r12*r32 + 0.166666666666667*L*N*r13*r33
            k += 1
            KGv[k] += 0.166666666666667*L*N*r22*r32 + 0.166666666666667*L*N*r23*r33
            k += 1
            KGv[k] += 0.166666666666667*L*N*r32**2 + 0.166666666666667*L*N*r33**2
            k += 1
            KGv[k] += 0.166666666666667*L*N*r12**2 + 0.166666666666667*L*N*r13**2
            k += 1
            KGv[k] += 0.166666666666667*L*N*r12*r22 + 0.166666666666667*L*N*r13*r23
            k += 1
            KGv[k] += 0.166666666666667*L*N*r12*r32 + 0.166666666666667*L*N*r13*r33
            k += 1
            KGv[k] += 0.333333333333333*L*N*r12**2 + 0.333333333333333*L*N*r13**2
            k += 1
            KGv[k] += 0.333333333333333*L*N*r12*r22 + 0.333333333333333*L*N*r13*r23
            k += 1
            KGv[k] += 0.333333333333333*L*N*r12*r32 + 0.333333333333333*L*N*r13*r33
            k += 1
            KGv[k] += 0.166666666666667*L*N*r12*r22 + 0.166666666666667*L*N*r13*r23
            k += 1
            KGv[k] += 0.166666666666667*L*N*r22**2 + 0.166666666666667*L*N*r23**2
            k += 1
            KGv[k] += 0.166666666666667*L*N*r22*r32 + 0.166666666666667*L*N*r23*r33
            k += 1
            KGv[k] += 0.333333333333333*L*N*r12*r22 + 0.333333333333333*L*N*r13*r23
            k += 1
            KGv[k] += 0.333333333333333*L*N*r22**2 + 0.333333333333333*L*N*r23**2
            k += 1
            KGv[k] += 0.333333333333333*L*N*r22*r32 + 0.333333333333333*L*N*r23*r33
            k += 1
            KGv[k] += 0.166666666666667*L*N*r12*r32 + 0.166666666666667*L*N*r13*r33
            k += 1
            KGv[k] += 0.166666666666667*L*N*r22*r32 + 0.166666666666667*L*N*r23*r33
            k += 1
            KGv[k] += 0.166666666666667*L*N*r32**2 + 0.166666666666667*L*N*r33**2
            k += 1
            KGv[k] += 0.333333333333333*L*N*r12*r32 + 0.333333333333333*L*N*r13*r33
            k += 1
            KGv[k] += 0.333333333333333*L*N*r22*r32 + 0.333333333333333*L*N*r23*r33
            k += 1
            KGv[k] += 0.333333333333333*L*N*r32**2 + 0.333333333333333*L*N*r33**2


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
