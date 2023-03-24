#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: nonecheck=False
#cython: overflowcheck=False
#cython: embedsignature=True
#cython: infer_types=False
r"""
Spring - 3D spring element with constant stiffnesses (:mod:`pyfe3d.spring`)
=========================================================================

.. currentmodule:: pyfe3d.spring

"""
import numpy as np

cdef int DOF = 6
cdef int NUM_NODES = 2


cdef class SpringData:
    r"""
    Used to allocate memory for the sparse matrices.

    Attributes
    ----------
    KC0_SPARSE_SIZE, : int
        ``KC0_SPARSE_SIZE = 72``

    KG_SPARSE_SIZE, : int
        ``KG_SPARSE_SIZE = 0``

    M_SPARSE_SIZE, : int
        ``M_SPARSE_SIZE = 0``

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

    Attributes
    ----------
    xe, : array-like
        Array of size ``NUM_NODES*DOF//2=6`` containing the nodal coordinates
        in the element coordinate system, in the following order `{x_e}_1,
        {y_e}_1, {z_e}_1, {x_e}_2, {y_e}_2, {z_e}_2`.
    ue, : array-like
        Array of size ``NUM_NODES*DOF=12`` containing the element displacements
        in the following order `{u_e}_1, {v_e}_1, {w_e}_1, {{r_x}_e}_1,
        {{r_y}_e}_1, {{r_z}_e}_1, {u_e}_2, {v_e}_2, {w_e}_2, {{r_x}_e}_2,
        {{r_y}_e}_2, {{r_z}_e}_2`.

    """
    cdef public double [::1] xe
    cdef public double [::1] ue
    def __cinit__(SpringProbe self):
        self.xe = np.zeros(NUM_NODES*DOF//2, dtype=np.float64)
        self.ue = np.zeros(NUM_NODES*DOF, dtype=np.float64)

cdef class Spring:
    r"""
    Spring 3D beam element with constant stiffnesses

    .. note:: The default behaviour assumes that the local coordinate system is
              aligned with the global coordinate system

    Attributes
    ----------
    eid, : int
        Element identification number.
    kxe, kye, kze: double
        Translational stiffnesses in the element coordinate system.
    krxe, krye, krze : double
        Rotational stiffnesses in the element coordinate system.
    r11, r12, r13, r21, r22, r23, r31, r32, r33 : double
        Rotation matrix to the global coordinate system. By default it assumes
        that the element is aligned with the global coordinate system.
    c1, c2 : int
        Position of each node in the global stiffness matrix.
    n1, n2 : int
        Node identification number.
    init_k_KC0, init_k_KG, init_k_M : int
        Position in the arrays storing the sparse data for the structural
        matrices.

    """
    cdef public int eid
    cdef public int n1, n2
    cdef public int c1, c2
    cdef public int init_k_KC0, init_k_KG, init_k_M
    cdef public double kxe, kye, kze
    cdef public double krxe, krye, krze
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
        self.r11 = 1
        self.r22 = 1
        self.r33 = 1
        self.r12 = self.r13 = 0.
        self.r21 = self.r23 = 0.
        self.r31 = self.r32 = 0.


    cpdef void update_rotation_matrix(Spring self, double xi, double xj, double xk,
            double vxyi, double vxyj, double vxyk):
        r"""Update the rotation matrix of the element

        Attributes ``r11,r12,r13,r21,r22,r23,r31,r32,r33`` are updated,
        corresponding to the rotation matrix from local to global coordinates.

        The element coordinate system is determined, identifying the `ijk`
        components of each axis: `{x_e}_i, {x_e}_j, {x_e}_k`; `{y_e}_i,
        {y_e}_j, {y_e}_k`; `{z_e}_i, {z_e}_j, {z_e}_k`.

        The rotation matrix terms are calculated after solving 9 equations.

        Parameters
        ----------
        xi, xj, xk : double
            Components, in global coordinates, of the element x-axis.
        vxyi, vxyj, vxyk : double
            Components, in global coordinates, of a vector on the `XY` plane of
            the element coordinate system.

        """
        cdef double yi, yj, yk, zi, zj, zk, tmp
        cdef double x1i, x1j, x1k, x2i, x2j, x2k, x3i, x3j, x3k, x4i, x4j, x4k

        with nogil:
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

        .. note:: The ``probe`` attribute object :class:`.SpringProbe` is
                  updated, not the element object.

        Parameters
        ----------
        u : array-like
            Array with global displacements, for a total of `M` nodes in
            the model, this array will be arranged as: `u_1, v_1, w_1, {r_x}_1,
            {r_y}_1, {r_z}_1, u_2, v_2, w_2, {r_x}_2, {r_y}_2, {r_z}_2, ...,
            u_M, v_M, w_M, {r_x}_M, {r_y}_M, {r_z}_M`.

        """
        cdef int i, j
        cdef int c[2]
        cdef double s1[3]
        cdef double s2[3]
        cdef double s3[3]

        # TODO double check all this part
        with nogil:
            # positions in the global stiffness matrix
            c[0] = self.c1
            c[1] = self.c2

            # global to local transformation of displacements
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


    cpdef void update_KC0(Spring self,
                          long [::1] KC0r,
                          long [::1] KC0c,
                          double [::1] KC0v,
                          int update_KC0v_only=0,
                          ):
        r"""Update sparse vectors for linear constitutive stiffness matrix KC0

        Properties
        ----------
        KC0r : np.array
            Array to store row positions of sparse values
        KC0c : np.array
            Array to store column positions of sparse values
        KC0v : np.array
            Array to store sparse values
        update_KC0v_only : int
            The default `0` means that only `KC0v` is updated. Any other value will
            lead to `KC0r` and `KC0c` also being updated.

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

            # Local to global transformation
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


