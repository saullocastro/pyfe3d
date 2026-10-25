#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: nonecheck=False
#cython: overflowcheck=False
#cython: initializedcheck=False
#cython: embedsignature=True
#cython: infer_types=False
r"""
===========================================================================
Spring - 3D spring element with constant stiffnesses (:mod:`pyfe3d.spring`)
===========================================================================

.. currentmodule:: pyfe3d.spring

Defines the two-node element :class:`.Spring`, with six degrees-of-freedom
per node and a constant stiffness for each of them, its probe
:class:`.SpringProbe` and :class:`.SpringData`, which gives the number of
sparse entries per element of each structural matrix.

"""
import numpy as np

cdef int DOF = 6
cdef int NUM_NODES = 2


cdef class SpringData:
    r"""
    Used to allocate memory for the sparse matrices.

    Each attribute is the number of entries that one :class:`.Spring`
    element writes in the arrays of rows, columns and values of a sparse
    matrix, at the consecutive positions that start at its ``init_k_KC0``,
    ``init_k_KG`` or ``init_k_M``. The constructor takes no argument and
    sets the values below.

    Attributes
    ----------
    KC0_SPARSE_SIZE, : int
        ``KC0_SPARSE_SIZE = 72``, written by :meth:`.Spring.update_KC0`.

    KG_SPARSE_SIZE, : int
        ``KG_SPARSE_SIZE = 0``, the element has no geometric stiffness
        matrix.

    M_SPARSE_SIZE, : int
        ``M_SPARSE_SIZE = 0``, the element has no mass matrix.

    """
    cdef public int KC0_SPARSE_SIZE
    cdef public int KG_SPARSE_SIZE
    cdef public int M_SPARSE_SIZE
    def __cinit__(SpringData self):
        self.KC0_SPARSE_SIZE = 72
        self.KG_SPARSE_SIZE = 0
        self.M_SPARSE_SIZE = 0


cdef class SpringProbe:
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
        Not updated by the methods of :class:`.Spring`.
    ue, : array-like
        Array of size ``NUM_NODES*DOF=12`` containing the element displacements
        in the following order `{u_e}_1, {v_e}_1, {w_e}_1, {{r_x}_e}_1,
        {{r_y}_e}_1, {{r_z}_e}_1, {u_e}_2, {v_e}_2, {w_e}_2, {{r_x}_e}_2,
        {{r_y}_e}_2, {{r_z}_e}_2`. Translations of dimension length and
        rotations. Updated by :meth:`.Spring.update_probe_ue`.
    finte, : array-like
        Array of size ``NUM_NODES*DOF=12`` containing the element internal
        forces corresponding to the degrees-of-freedom described by ``ue``,
        forces and moments. Updated by :meth:`.Spring.update_probe_finte`.

    """
    cdef public double [::1] xe
    cdef public double [::1] ue
    cdef public double [::1] finte

    def __cinit__(SpringProbe self):
        self.xe = np.zeros(NUM_NODES*DOF//2, dtype=np.float64)
        self.ue = np.zeros(NUM_NODES*DOF, dtype=np.float64)
        self.finte = np.zeros(NUM_NODES*DOF, dtype=np.float64)


cdef class Spring:
    r"""
    Spring 3D beam element with constant stiffnesses

    .. note:: The default behaviour assumes that the local coordinate system is
              aligned with the global coordinate system

    The constructor stores ``p`` in ``probe``, sets ``eid``, ``n1``, ``n2``,
    ``c1`` and ``c2`` to ``-1``, sets the ``init_k_*`` positions, the
    stiffnesses and ``vxyi``, ``vxyj``, ``vxyk`` to ``0``, and sets the
    rotation matrix to the identity. :meth:`.update_KC0` and
    :meth:`.update_probe_finte` read the stiffnesses and the rotation matrix;
    :meth:`.update_probe_finte` also reads ``probe.ue``, updated by
    :meth:`.update_probe_ue`. The element has no length, and the nodal
    coordinates are not used.

    Parameters
    ----------
    p : :class:`.SpringProbe` object
        Probe of the element, stored in ``probe``. It can be shared with
        other elements.

    Attributes
    ----------
    eid, : int
        Element identification number.
    kxe, kye, kze: double
        Translational stiffnesses in the element coordinate system, force per
        unit length.
    krxe, krye, krze : double
        Rotational stiffnesses in the element coordinate system, moment per
        unit rotation.
    r11, r12, r13, r21, r22, r23, r31, r32, r33 : double
        Entries `T_{ij}` of the transformation matrix `[T_{3\times 3}]` from
        element to global coordinates, whose columns are the element axes. By
        default it assumes that the element is aligned with the global
        coordinate system. Dimensionless.
    vxyi, vxyj, vxyk : double
        Components of a vector on the `XY` plane of the element coordinate
        system, defined using global coordinates. Only the direction of the
        vector is used.
    c1, c2 : int
        Position of each node in the global stiffness matrix, i.e. of the
        first of its ``6`` degrees-of-freedom.
    n1, n2 : int
        Node identification number.
    init_k_KC0, init_k_KG, init_k_M : int
        Position in the arrays storing the sparse data for the structural
        matrices, where the entries of this element start, see
        :class:`.SpringData` for their number.
    probe : :class:`.SpringProbe` object
        Pointer to the probe.

    Notes
    -----
    The strain energy of the element is

    .. math::
        U = \frac{1}{2} \sum_{i=1}^{6} k_i \left( {u_e}_{i+6} - {u_e}_{i}
            \right)^2

    where `k_1, ..., k_6` are ``kxe``, ``kye``, ``kze``, ``krxe``, ``krye``
    and ``krze``, and ``ue`` are the displacements in the element coordinate
    system, in the order of :class:`.SpringProbe`. Each stiffness connects
    only the same degree-of-freedom of the two nodes, without coupling
    between translations and rotations, such that the internal forces are
    ``finte = [K_e] ue`` with

    .. math::
        [K_e] = \begin{bmatrix} [k] & -[k] \\ -[k] & [k] \end{bmatrix},
        \qquad [k] = \mathrm{diag}(k_1, ..., k_6)

    transformed to global coordinates with the transformation matrix of the
    element, `[K] = [T] [K_e] [T]^T`, where `[T]` is block diagonal with four
    copies of the matrix `[T_{3\times 3}]` of the entries `T_{ij}` stored in
    ``r11, ..., r33``, whose columns are the element axes.

    A rigid translation and equal rotations of the two nodes give no
    internal forces. Since the nodal coordinates are not used, a rigid
    rotation `\{\theta\}` of two nodes at different positions `\{p_1\}`
    and `\{p_2\}` gives the relative translation `\{\theta\} \times
    (\{p_2\} - \{p_1\})`, normal to the line of the nodes, which loads
    the translational stiffnesses in that direction: with the element `x`
    axis along the line of the nodes, ``kye`` and ``kze``. These
    properties are verified in ``tests/test_spring.py``.

    """
    cdef public int eid
    cdef public int n1, n2
    cdef public int c1, c2
    cdef public int init_k_KC0, init_k_KG, init_k_M
    cdef public double kxe, kye, kze
    cdef public double krxe, krye, krze
    cdef public double vxyi, vxyj, vxyk
    cdef public double r11, r12, r13, r21, r22, r23, r31, r32, r33
    cdef public SpringProbe probe

    def __cinit__(Spring self, SpringProbe p):
        self.probe = p
        self.eid = -1
        self.n1 = -1
        self.n2 = -1
        self.c1 = -1
        self.c2 = -1
        self.init_k_KC0 = 0
        self.init_k_KG = 0
        self.init_k_M = 0
        self.kxe = self.kye = self.kze = 0
        self.krxe = self.krye = self.krze = 0
        self.vxyi = self.vxyj = self.vxyk = 0.
        self.r11 = 1
        self.r22 = 1
        self.r33 = 1
        self.r12 = self.r13 = 0.
        self.r21 = self.r23 = 0.
        self.r31 = self.r32 = 0.


    cpdef void update_rotation_matrix(Spring self, double xi, double xj, double xk,
            double vxyi, double vxyj, double vxyk):
        r"""Update the rotation matrix of the element

        Attributes ``r11,r12,r13,r21,r22,r23,r31,r32,r33`` are updated, with
        the entries `T_{ij}` of the transformation matrix `[T_{3\times 3}]`
        from element to global coordinates, whose columns are the element axes.

        The element coordinate system is determined, identifying the `ijk`
        components of each axis: `{x_e}_i, {x_e}_j, {x_e}_k`; `{y_e}_i,
        {y_e}_j, {y_e}_k`; `{z_e}_i, {z_e}_j, {z_e}_k`.

        The element `x` axis is ``(xi, xj, xk)``, the `z` axis is normal to the
        `x` axis and to the vector ``(vxyi, vxyj, vxyk)``, the `y` axis
        completes the right-handed system, and the three axes are normalized.
        The element attributes ``vxyi``, ``vxyj`` and ``vxyk`` are also
        updated.


        Parameters
        ----------
        xi, xj, xk : double
            Components, in global coordinates, of the element x-axis. Only the
            direction of the vector is used.
        vxyi, vxyj, vxyk : double
            Components, in global coordinates, of a vector on the `XY` plane of
            the element coordinate system. Only the direction of the vector is
            used. A vector parallel to the element x-axis gives a null `z` axis
            and a rotation matrix with ``nan`` entries.

        """
        cdef double yi, yj, yk, zi, zj, zk, tmp

        with nogil:
            self.vxyi = vxyi
            self.vxyj = vxyj
            self.vxyk = vxyk

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


    cpdef void update_probe_ue(Spring self, double [::1] u):
        r"""Update the local displacement vector of the probe of the element

        .. note:: The ``ue`` attribute of object :class:`.SpringProbe` is
                  updated, accessible using ``.probe.ue``.

        The translations and rotations of the two nodes, read from ``u`` at the
        positions ``c1`` to ``c1 + 5`` and ``c2`` to ``c2 + 5``, are
        transformed to the element coordinate system with the transpose
        `[T_{3\times 3}]^T` of the transformation matrix, which must be updated
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

        # NOTE verified against [T]^T u in tests/test_spring.py
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


    cpdef void update_probe_finte(Spring self):
        r"""Update the internal force vector of the probe

        The attribute ``finte`` is updated with the :class:`.SpringProbe` the
        internal forces in local coordinates. While using this function, mind
        that the probe can be shared amongst more than one finite element,
        depending how you defined them, meaning that the probe will always safe
        the values from the last udpate.

        .. note:: The ``finte`` attribute of object :class:`.SpringProbe` is
                  updated, accessible using ``.probe.finte``.

        The 12 values of ``finte`` are overwritten. They are computed from the
        stiffnesses ``kxe``, ``kye``, ``kze``, ``krxe``, ``krye`` and ``krze``
        and from the displacements ``probe.ue``, which must be updated before,
        see :meth:`.update_probe_ue`.

        """
        cdef double *ue
        cdef double *finte

        with nogil:
            ue = &self.probe.ue[0]
            finte = &self.probe.finte[0]
            finte[0] = self.kxe*(ue[0] - ue[6])
            finte[1] = self.kye*(ue[1] - ue[7])
            finte[2] = self.kze*(ue[2] - ue[8])
            finte[3] = self.krxe*(ue[3] - ue[9])
            finte[4] = self.krye*(-ue[10] + ue[4])
            finte[5] = self.krze*(-ue[11] + ue[5])
            finte[6] = self.kxe*(-ue[0] + ue[6])
            finte[7] = self.kye*(-ue[1] + ue[7])
            finte[8] = self.kze*(-ue[2] + ue[8])
            finte[9] = self.krxe*(-ue[3] + ue[9])
            finte[10] = self.krye*(ue[10] - ue[4])
            finte[11] = self.krze*(ue[11] - ue[5])


    cpdef void update_KC0(Spring self,
                          long [::1] KC0r,
                          long [::1] KC0c,
                          double [::1] KC0v,
                          int update_KC0v_only=0,
                          ):
        r"""Update sparse vectors for linear constitutive stiffness matrix KC0

        The 72 entries of the matrix of the element, in global coordinates,
        are written at the positions ``init_k_KC0`` to ``init_k_KC0 + 71``:
        the row and column indices, built from ``c1`` and ``c2``, are stored
        in ``KC0r`` and ``KC0c``, and the values are added to those already
        in ``KC0v``. The matrix is computed from the rotation matrix and the
        stiffnesses ``kxe``, ``kye``, ``kze``, ``krxe``, ``krye`` and
        ``krze``. The rotation matrix must be updated before, when the
        element is not aligned with the global coordinate system, see
        :meth:`.update_rotation_matrix`.

        Parameters
        ----------
        KC0r : np.array
            Array to store row positions of sparse values. 1-D array of
            integers of the C type ``long``, of size at least
            ``init_k_KC0 + 72``.
        KC0c : np.array
            Array to store column positions of sparse values. Same type and
            size as ``KC0r``.
        KC0v : np.array
            Array to store sparse values. 1-D array of floats of size at least
            ``init_k_KC0 + 72``.
        update_KC0v_only : int, optional
            The default ``0`` means that the row and column indices ``KC0r``
            and ``KC0c`` should also be updated. Any other value will only
            update the stiffness matrix values ``KC0v``.

        """
        cdef int c1, c2, k
        cdef double kxe, kye, kze, krxe, krye, krze
        cdef double r11, r12, r13, r21, r22, r23, r31, r32, r33

        with nogil:
            kxe = self.kxe
            kye = self.kye
            kze = self.kze
            krxe = self.krxe
            krye = self.krye
            krze = self.krze

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
                KC0c[k] = 0+c2
                k += 1
                KC0r[k] = 0+c1
                KC0c[k] = 1+c2
                k += 1
                KC0r[k] = 0+c1
                KC0c[k] = 2+c2
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
                KC0c[k] = 0+c2
                k += 1
                KC0r[k] = 1+c1
                KC0c[k] = 1+c2
                k += 1
                KC0r[k] = 1+c1
                KC0c[k] = 2+c2
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
                KC0c[k] = 0+c2
                k += 1
                KC0r[k] = 2+c1
                KC0c[k] = 1+c2
                k += 1
                KC0r[k] = 2+c1
                KC0c[k] = 2+c2
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
                KC0c[k] = 3+c2
                k += 1
                KC0r[k] = 3+c1
                KC0c[k] = 4+c2
                k += 1
                KC0r[k] = 3+c1
                KC0c[k] = 5+c2
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
                KC0c[k] = 3+c2
                k += 1
                KC0r[k] = 4+c1
                KC0c[k] = 4+c2
                k += 1
                KC0r[k] = 4+c1
                KC0c[k] = 5+c2
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
                KC0c[k] = 0+c2
                k += 1
                KC0r[k] = 0+c2
                KC0c[k] = 1+c2
                k += 1
                KC0r[k] = 0+c2
                KC0c[k] = 2+c2
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
                KC0c[k] = 0+c2
                k += 1
                KC0r[k] = 1+c2
                KC0c[k] = 1+c2
                k += 1
                KC0r[k] = 1+c2
                KC0c[k] = 2+c2
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
                KC0c[k] = 0+c2
                k += 1
                KC0r[k] = 2+c2
                KC0c[k] = 1+c2
                k += 1
                KC0r[k] = 2+c2
                KC0c[k] = 2+c2
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
                KC0c[k] = 3+c2
                k += 1
                KC0r[k] = 3+c2
                KC0c[k] = 4+c2
                k += 1
                KC0r[k] = 3+c2
                KC0c[k] = 5+c2
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
                KC0c[k] = 3+c2
                k += 1
                KC0r[k] = 4+c2
                KC0c[k] = 4+c2
                k += 1
                KC0r[k] = 4+c2
                KC0c[k] = 5+c2
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
                KC0c[k] = 3+c2
                k += 1
                KC0r[k] = 5+c2
                KC0c[k] = 4+c2
                k += 1
                KC0r[k] = 5+c2
                KC0c[k] = 5+c2

            k = self.init_k_KC0
            KC0v[k] += kxe*r11**2 + kye*r12**2 + kze*r13**2
            k += 1
            KC0v[k] += kxe*r11*r21 + kye*r12*r22 + kze*r13*r23
            k += 1
            KC0v[k] += kxe*r11*r31 + kye*r12*r32 + kze*r13*r33
            k += 1
            KC0v[k] += -kxe*r11**2 - kye*r12**2 - kze*r13**2
            k += 1
            KC0v[k] += -kxe*r11*r21 - kye*r12*r22 - kze*r13*r23
            k += 1
            KC0v[k] += -kxe*r11*r31 - kye*r12*r32 - kze*r13*r33
            k += 1
            KC0v[k] += kxe*r11*r21 + kye*r12*r22 + kze*r13*r23
            k += 1
            KC0v[k] += kxe*r21**2 + kye*r22**2 + kze*r23**2
            k += 1
            KC0v[k] += kxe*r21*r31 + kye*r22*r32 + kze*r23*r33
            k += 1
            KC0v[k] += -kxe*r11*r21 - kye*r12*r22 - kze*r13*r23
            k += 1
            KC0v[k] += -kxe*r21**2 - kye*r22**2 - kze*r23**2
            k += 1
            KC0v[k] += -kxe*r21*r31 - kye*r22*r32 - kze*r23*r33
            k += 1
            KC0v[k] += kxe*r11*r31 + kye*r12*r32 + kze*r13*r33
            k += 1
            KC0v[k] += kxe*r21*r31 + kye*r22*r32 + kze*r23*r33
            k += 1
            KC0v[k] += kxe*r31**2 + kye*r32**2 + kze*r33**2
            k += 1
            KC0v[k] += -kxe*r11*r31 - kye*r12*r32 - kze*r13*r33
            k += 1
            KC0v[k] += -kxe*r21*r31 - kye*r22*r32 - kze*r23*r33
            k += 1
            KC0v[k] += -kxe*r31**2 - kye*r32**2 - kze*r33**2
            k += 1
            KC0v[k] += krxe*r11**2 + krye*r12**2 + krze*r13**2
            k += 1
            KC0v[k] += krxe*r11*r21 + krye*r12*r22 + krze*r13*r23
            k += 1
            KC0v[k] += krxe*r11*r31 + krye*r12*r32 + krze*r13*r33
            k += 1
            KC0v[k] += -krxe*r11**2 - krye*r12**2 - krze*r13**2
            k += 1
            KC0v[k] += -krxe*r11*r21 - krye*r12*r22 - krze*r13*r23
            k += 1
            KC0v[k] += -krxe*r11*r31 - krye*r12*r32 - krze*r13*r33
            k += 1
            KC0v[k] += krxe*r11*r21 + krye*r12*r22 + krze*r13*r23
            k += 1
            KC0v[k] += krxe*r21**2 + krye*r22**2 + krze*r23**2
            k += 1
            KC0v[k] += krxe*r21*r31 + krye*r22*r32 + krze*r23*r33
            k += 1
            KC0v[k] += -krxe*r11*r21 - krye*r12*r22 - krze*r13*r23
            k += 1
            KC0v[k] += -krxe*r21**2 - krye*r22**2 - krze*r23**2
            k += 1
            KC0v[k] += -krxe*r21*r31 - krye*r22*r32 - krze*r23*r33
            k += 1
            KC0v[k] += krxe*r11*r31 + krye*r12*r32 + krze*r13*r33
            k += 1
            KC0v[k] += krxe*r21*r31 + krye*r22*r32 + krze*r23*r33
            k += 1
            KC0v[k] += krxe*r31**2 + krye*r32**2 + krze*r33**2
            k += 1
            KC0v[k] += -krxe*r11*r31 - krye*r12*r32 - krze*r13*r33
            k += 1
            KC0v[k] += -krxe*r21*r31 - krye*r22*r32 - krze*r23*r33
            k += 1
            KC0v[k] += -krxe*r31**2 - krye*r32**2 - krze*r33**2
            k += 1
            KC0v[k] += -kxe*r11**2 - kye*r12**2 - kze*r13**2
            k += 1
            KC0v[k] += -kxe*r11*r21 - kye*r12*r22 - kze*r13*r23
            k += 1
            KC0v[k] += -kxe*r11*r31 - kye*r12*r32 - kze*r13*r33
            k += 1
            KC0v[k] += kxe*r11**2 + kye*r12**2 + kze*r13**2
            k += 1
            KC0v[k] += kxe*r11*r21 + kye*r12*r22 + kze*r13*r23
            k += 1
            KC0v[k] += kxe*r11*r31 + kye*r12*r32 + kze*r13*r33
            k += 1
            KC0v[k] += -kxe*r11*r21 - kye*r12*r22 - kze*r13*r23
            k += 1
            KC0v[k] += -kxe*r21**2 - kye*r22**2 - kze*r23**2
            k += 1
            KC0v[k] += -kxe*r21*r31 - kye*r22*r32 - kze*r23*r33
            k += 1
            KC0v[k] += kxe*r11*r21 + kye*r12*r22 + kze*r13*r23
            k += 1
            KC0v[k] += kxe*r21**2 + kye*r22**2 + kze*r23**2
            k += 1
            KC0v[k] += kxe*r21*r31 + kye*r22*r32 + kze*r23*r33
            k += 1
            KC0v[k] += -kxe*r11*r31 - kye*r12*r32 - kze*r13*r33
            k += 1
            KC0v[k] += -kxe*r21*r31 - kye*r22*r32 - kze*r23*r33
            k += 1
            KC0v[k] += -kxe*r31**2 - kye*r32**2 - kze*r33**2
            k += 1
            KC0v[k] += kxe*r11*r31 + kye*r12*r32 + kze*r13*r33
            k += 1
            KC0v[k] += kxe*r21*r31 + kye*r22*r32 + kze*r23*r33
            k += 1
            KC0v[k] += kxe*r31**2 + kye*r32**2 + kze*r33**2
            k += 1
            KC0v[k] += -krxe*r11**2 - krye*r12**2 - krze*r13**2
            k += 1
            KC0v[k] += -krxe*r11*r21 - krye*r12*r22 - krze*r13*r23
            k += 1
            KC0v[k] += -krxe*r11*r31 - krye*r12*r32 - krze*r13*r33
            k += 1
            KC0v[k] += krxe*r11**2 + krye*r12**2 + krze*r13**2
            k += 1
            KC0v[k] += krxe*r11*r21 + krye*r12*r22 + krze*r13*r23
            k += 1
            KC0v[k] += krxe*r11*r31 + krye*r12*r32 + krze*r13*r33
            k += 1
            KC0v[k] += -krxe*r11*r21 - krye*r12*r22 - krze*r13*r23
            k += 1
            KC0v[k] += -krxe*r21**2 - krye*r22**2 - krze*r23**2
            k += 1
            KC0v[k] += -krxe*r21*r31 - krye*r22*r32 - krze*r23*r33
            k += 1
            KC0v[k] += krxe*r11*r21 + krye*r12*r22 + krze*r13*r23
            k += 1
            KC0v[k] += krxe*r21**2 + krye*r22**2 + krze*r23**2
            k += 1
            KC0v[k] += krxe*r21*r31 + krye*r22*r32 + krze*r23*r33
            k += 1
            KC0v[k] += -krxe*r11*r31 - krye*r12*r32 - krze*r13*r33
            k += 1
            KC0v[k] += -krxe*r21*r31 - krye*r22*r32 - krze*r23*r33
            k += 1
            KC0v[k] += -krxe*r31**2 - krye*r32**2 - krze*r33**2
            k += 1
            KC0v[k] += krxe*r11*r31 + krye*r12*r32 + krze*r13*r33
            k += 1
            KC0v[k] += krxe*r21*r31 + krye*r22*r32 + krze*r23*r33
            k += 1
            KC0v[k] += krxe*r31**2 + krye*r32**2 + krze*r33**2


    cpdef void update_fint(Spring self, double [::1] fint):
        r"""Update the internal force vector

        Calls ``update_probe_finte()``, transforms the forces and moments of
        ``probe.finte`` to global coordinates with the rotation matrix and
        adds them to ``fint`` at the positions ``c1`` to ``c1 + 5`` and ``c2``
        to ``c2 + 5``. The displacements ``probe.ue`` must be updated before,
        see :meth:`.update_probe_ue`.

        Parameters
        ----------
        fint : np.array
            Array that is updated in place with the internal forces. The
            internal forces stored in ``fint`` are calculated in global
            coordinates. Method :meth:`.update_probe_finte` is called to update
            the parameter ``finte`` of the :class:`.SpringProbe` with the
            internal forces in local coordinates. 1-D array of size ``6*M``,
            for `M` nodes in the model.

        """
        cdef double *finte

        self.update_probe_finte()
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

