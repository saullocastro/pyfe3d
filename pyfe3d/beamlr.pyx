#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: nonecheck=False
#cython: overflowcheck=False
#cython: embedsignature=True
#cython: infer_types=False
r"""
BeamLR - Linear Timoshenko 3D beam element with reduced integration (:mod:`pyfe3d.beamlr`)
==========================================================================================

.. currentmodule:: pyfe3d.beamlr

"""
import numpy as np

from .beamprop cimport BeamProp

cdef int DOF = 6
cdef int NUM_NODES = 2


cdef class BeamLRData:
    r"""
    Used to allocate memory for the sparse matrices.

    Attributes
    ----------
    KC0_SPARSE_SIZE, : int
        ``KC0_SPARSE_SIZE = 144``

    KG_SPARSE_SIZE, : int
        ``KG_SPARSE_SIZE = 36``

    M_SPARSE_SIZE, : int
        ``M_SPARSE_SIZE = 144``

    """
    cdef public int KC0_SPARSE_SIZE
    cdef public int KG_SPARSE_SIZE
    cdef public int M_SPARSE_SIZE
    def __cinit__(BeamLRData self):
        self.KC0_SPARSE_SIZE = 144
        self.KG_SPARSE_SIZE = 36
        self.M_SPARSE_SIZE = 144


cdef class BeamLRProbe:
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
    def __cinit__(BeamLRProbe self):
        self.xe = np.zeros(NUM_NODES*DOF//2, dtype=np.float64)
        self.ue = np.zeros(NUM_NODES*DOF, dtype=np.float64)


cdef class BeamLR:
    r"""
    Timoshenko 3D beam element with linear shape functions

    Formulation based on reference, replacing the consistent shape functions by
    linear functions and performing numerical integration with just 1 point at
    the beam center:

        Luo, Y., 2008, “An Efficient 3D Timoshenko Beam Element with Consistent
        Shape Functions,” Adv. Theor. Appl. Mech., 1(3), pp. 95–106.

    Nodal connectivity for the beam element::

        ^ y axis
        |
        |
        ______   --> x axis
        1    2

    Attributes
    ----------
    eid, : int
        Element identification number.
    length, : double
        Element length.
    r11, r12, r13, r21, r22, r23, r31, r32, r33 : double
        Rotation matrix to the global coordinate system.
    c1, c2 : int
        Position of each node in the global stiffness matrix.
    n1, n2 : int
        Node identification number.
    init_k_KC0, init_k_KG, init_k_M : int
        Position in the arrays storing the sparse data for the structural
        matrices.
    probe : :class:`.BeamLRProbe` object
        Pointer to the probe.

    """
    cdef public int eid
    cdef public int n1, n2
    cdef public int c1, c2
    cdef public int init_k_KC0, init_k_KG, init_k_M
    cdef public double length
    cdef public double r11, r12, r13, r21, r22, r23, r31, r32, r33
    cdef public BeamLRProbe probe

    def __cinit__(BeamLR self, BeamLRProbe p):
        self.probe = p
        self.eid = -1
        self.n1 = -1
        self.n2 = -1
        self.c1 = -1
        self.c2 = -1
        self.init_k_KC0 = 0
        # self.init_k_KCNL = 0
        self.init_k_KG = 0
        self.init_k_M = 0
        self.length = 0
        self.r11 = self.r12 = self.r13 = 0.
        self.r21 = self.r22 = self.r23 = 0.
        self.r31 = self.r32 = self.r33 = 0.


    cpdef void update_rotation_matrix(BeamLR self, double vxyi, double vxyj,
                                      double vxyk, double [::1] x):
        r"""Update the rotation matrix of the element

        Attributes ``r11,r12,r13,r21,r22,r23,r31,r32,r33`` are updated,
        corresponding to the rotation matrix from local to global coordinates.

        The element coordinate system is determined, identifying the `ijk`
        components of each axis: `{x_e}_i, {x_e}_j, {x_e}_k`; `{y_e}_i,
        {y_e}_j, {y_e}_k`; `{z_e}_i, {z_e}_j, {z_e}_k`.

        The rotation matrix terms are calculated after solving 9 equations.

        Parameters
        ----------
        vxyi, vxyj, vxyk : double
            Components of a vector on the `XY` plane of the element coordinate
            system.
        x : array-like
            Array with global nodal coordinates, for a total of `M` nodes in
            the model, this array will be arranged as: `x_1, y_1, z_1, x_2,
            y_2, z_2, ..., x_M, y_M, z_M`.

        """
        cdef double xi, xj, xk, yi, yj, yk, zi, zj, zk, tmp
        cdef double x1i, x1j, x1k, x2i, x2j, x2k, x3i, x3j, x3k, x4i, x4j, x4k

        with nogil:
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

        .. note:: The ``probe`` attribute object :class:`.BeamLRProbe` is
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


    cpdef void update_probe_xe(BeamLR self, double [::1] x):
        r"""Update the 3D coordinates of the probe of the element

        .. note:: The ``probe`` attribute object :class:`.BeamLRProbe` is
                  updated, not the element object.

        Parameters
        ----------
        x : array-like
            Array with global nodal coordinates, for a total of `M` nodes in
            the model, this array will be arranged as: `x_1, y_1, z_1, x_2,
            y_2, z_2, ..., x_M, y_M, z_M`.

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


    cpdef void update_KC0(BeamLR self,
                          long [::1] KC0r,
                          long [::1] KC0c,
                          double [::1] KC0v,
                          BeamProp prop,
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
        prop : :class:`BeamProp` object
            Beam property object from where the stiffness and mass attributes
            are read from.
        update_KC0v_only : int
            The default `0` means that only `KC0v` is updated. Any other value will
            lead to `KC0r` and `KC0c` also being updated.

        """
        cdef int c1, c2, k
        cdef double L, A, E, G, Ay, Az, Iyy, Izz, Iyz, J
        cdef double r11, r12, r13, r21, r22, r23, r31, r32, r33

        with nogil:
            L = self.length
            A = prop.A
            E = prop.E
            G = prop.G
            Ay = prop.Ay
            Az = prop.Az
            Iyy = prop.Iyy
            Izz = prop.Izz
            Iyz = prop.Iyz
            J = prop.J

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
            KC0v[k] += 1.0*A*E*r11**2/L + 1.0*A*G*r12**2/L + 1.0*A*G*r13**2/L
            k += 1
            KC0v[k] += 1.0*A*E*r11*r21/L + 1.0*A*G*r12*r22/L + 1.0*A*G*r13*r23/L
            k += 1
            KC0v[k] += 1.0*A*E*r11*r31/L + 1.0*A*G*r12*r32/L + 1.0*A*G*r13*r33/L
            k += 1
            KC0v[k] += r11*(1.0*Ay*G*r13/L - 1.0*Az*G*r12/L) + r12*(-0.5*A*G*r13 + 1.0*Az*E*r11/L) + r13*(0.5*A*G*r12 - 1.0*Ay*E*r11/L)
            k += 1
            KC0v[k] += r21*(1.0*Ay*G*r13/L - 1.0*Az*G*r12/L) + r22*(-0.5*A*G*r13 + 1.0*Az*E*r11/L) + r23*(0.5*A*G*r12 - 1.0*Ay*E*r11/L)
            k += 1
            KC0v[k] += r31*(1.0*Ay*G*r13/L - 1.0*Az*G*r12/L) + r32*(-0.5*A*G*r13 + 1.0*Az*E*r11/L) + r33*(0.5*A*G*r12 - 1.0*Ay*E*r11/L)
            k += 1
            KC0v[k] += -1.0*A*E*r11**2/L - 1.0*A*G*r12**2/L - 1.0*A*G*r13**2/L
            k += 1
            KC0v[k] += -1.0*A*E*r11*r21/L - 1.0*A*G*r12*r22/L - 1.0*A*G*r13*r23/L
            k += 1
            KC0v[k] += -1.0*A*E*r11*r31/L - 1.0*A*G*r12*r32/L - 1.0*A*G*r13*r33/L
            k += 1
            KC0v[k] += r11*(-1.0*Ay*G*r13/L + 1.0*Az*G*r12/L) + r12*(-0.5*A*G*r13 - 1.0*Az*E*r11/L) + r13*(0.5*A*G*r12 + 1.0*Ay*E*r11/L)
            k += 1
            KC0v[k] += r21*(-1.0*Ay*G*r13/L + 1.0*Az*G*r12/L) + r22*(-0.5*A*G*r13 - 1.0*Az*E*r11/L) + r23*(0.5*A*G*r12 + 1.0*Ay*E*r11/L)
            k += 1
            KC0v[k] += r31*(-1.0*Ay*G*r13/L + 1.0*Az*G*r12/L) + r32*(-0.5*A*G*r13 - 1.0*Az*E*r11/L) + r33*(0.5*A*G*r12 + 1.0*Ay*E*r11/L)
            k += 1
            KC0v[k] += 1.0*A*E*r11*r21/L + 1.0*A*G*r12*r22/L + 1.0*A*G*r13*r23/L
            k += 1
            KC0v[k] += 1.0*A*E*r21**2/L + 1.0*A*G*r22**2/L + 1.0*A*G*r23**2/L
            k += 1
            KC0v[k] += 1.0*A*E*r21*r31/L + 1.0*A*G*r22*r32/L + 1.0*A*G*r23*r33/L
            k += 1
            KC0v[k] += r11*(1.0*Ay*G*r23/L - 1.0*Az*G*r22/L) + r12*(-0.5*A*G*r23 + 1.0*Az*E*r21/L) + r13*(0.5*A*G*r22 - 1.0*Ay*E*r21/L)
            k += 1
            KC0v[k] += r21*(1.0*Ay*G*r23/L - 1.0*Az*G*r22/L) + r22*(-0.5*A*G*r23 + 1.0*Az*E*r21/L) + r23*(0.5*A*G*r22 - 1.0*Ay*E*r21/L)
            k += 1
            KC0v[k] += r31*(1.0*Ay*G*r23/L - 1.0*Az*G*r22/L) + r32*(-0.5*A*G*r23 + 1.0*Az*E*r21/L) + r33*(0.5*A*G*r22 - 1.0*Ay*E*r21/L)
            k += 1
            KC0v[k] += -1.0*A*E*r11*r21/L - 1.0*A*G*r12*r22/L - 1.0*A*G*r13*r23/L
            k += 1
            KC0v[k] += -1.0*A*E*r21**2/L - 1.0*A*G*r22**2/L - 1.0*A*G*r23**2/L
            k += 1
            KC0v[k] += -1.0*A*E*r21*r31/L - 1.0*A*G*r22*r32/L - 1.0*A*G*r23*r33/L
            k += 1
            KC0v[k] += r11*(-1.0*Ay*G*r23/L + 1.0*Az*G*r22/L) + r12*(-0.5*A*G*r23 - 1.0*Az*E*r21/L) + r13*(0.5*A*G*r22 + 1.0*Ay*E*r21/L)
            k += 1
            KC0v[k] += r21*(-1.0*Ay*G*r23/L + 1.0*Az*G*r22/L) + r22*(-0.5*A*G*r23 - 1.0*Az*E*r21/L) + r23*(0.5*A*G*r22 + 1.0*Ay*E*r21/L)
            k += 1
            KC0v[k] += r31*(-1.0*Ay*G*r23/L + 1.0*Az*G*r22/L) + r32*(-0.5*A*G*r23 - 1.0*Az*E*r21/L) + r33*(0.5*A*G*r22 + 1.0*Ay*E*r21/L)
            k += 1
            KC0v[k] += 1.0*A*E*r11*r31/L + 1.0*A*G*r12*r32/L + 1.0*A*G*r13*r33/L
            k += 1
            KC0v[k] += 1.0*A*E*r21*r31/L + 1.0*A*G*r22*r32/L + 1.0*A*G*r23*r33/L
            k += 1
            KC0v[k] += 1.0*A*E*r31**2/L + 1.0*A*G*r32**2/L + 1.0*A*G*r33**2/L
            k += 1
            KC0v[k] += r11*(1.0*Ay*G*r33/L - 1.0*Az*G*r32/L) + r12*(-0.5*A*G*r33 + 1.0*Az*E*r31/L) + r13*(0.5*A*G*r32 - 1.0*Ay*E*r31/L)
            k += 1
            KC0v[k] += r21*(1.0*Ay*G*r33/L - 1.0*Az*G*r32/L) + r22*(-0.5*A*G*r33 + 1.0*Az*E*r31/L) + r23*(0.5*A*G*r32 - 1.0*Ay*E*r31/L)
            k += 1
            KC0v[k] += r31*(1.0*Ay*G*r33/L - 1.0*Az*G*r32/L) + r32*(-0.5*A*G*r33 + 1.0*Az*E*r31/L) + r33*(0.5*A*G*r32 - 1.0*Ay*E*r31/L)
            k += 1
            KC0v[k] += -1.0*A*E*r11*r31/L - 1.0*A*G*r12*r32/L - 1.0*A*G*r13*r33/L
            k += 1
            KC0v[k] += -1.0*A*E*r21*r31/L - 1.0*A*G*r22*r32/L - 1.0*A*G*r23*r33/L
            k += 1
            KC0v[k] += -1.0*A*E*r31**2/L - 1.0*A*G*r32**2/L - 1.0*A*G*r33**2/L
            k += 1
            KC0v[k] += r11*(-1.0*Ay*G*r33/L + 1.0*Az*G*r32/L) + r12*(-0.5*A*G*r33 - 1.0*Az*E*r31/L) + r13*(0.5*A*G*r32 + 1.0*Ay*E*r31/L)
            k += 1
            KC0v[k] += r21*(-1.0*Ay*G*r33/L + 1.0*Az*G*r32/L) + r22*(-0.5*A*G*r33 - 1.0*Az*E*r31/L) + r23*(0.5*A*G*r32 + 1.0*Ay*E*r31/L)
            k += 1
            KC0v[k] += r31*(-1.0*Ay*G*r33/L + 1.0*Az*G*r32/L) + r32*(-0.5*A*G*r33 - 1.0*Az*E*r31/L) + r33*(0.5*A*G*r32 + 1.0*Ay*E*r31/L)
            k += 1
            KC0v[k] += r11*(-1.0*Ay*E*r13/L + 1.0*Az*E*r12/L) + r12*(0.5*A*G*r13 - 1.0*Az*G*r11/L) + r13*(-0.5*A*G*r12 + 1.0*Ay*G*r11/L)
            k += 1
            KC0v[k] += r21*(-1.0*Ay*E*r13/L + 1.0*Az*E*r12/L) + r22*(0.5*A*G*r13 - 1.0*Az*G*r11/L) + r23*(-0.5*A*G*r12 + 1.0*Ay*G*r11/L)
            k += 1
            KC0v[k] += r31*(-1.0*Ay*E*r13/L + 1.0*Az*E*r12/L) + r32*(0.5*A*G*r13 - 1.0*Az*G*r11/L) + r33*(-0.5*A*G*r12 + 1.0*Ay*G*r11/L)
            k += 1
            KC0v[k] += r11*(-0.5*Ay*G*r12 - 0.5*Az*G*r13 + 1.0*G*J*r11/L) + r12*(-0.5*Ay*G*r11 - 1.0*E*Iyz*r13/L + 1.0*L*r12*(A*G/4 + E*Iyy/L**2)) + r13*(-0.5*Az*G*r11 - 1.0*E*Iyz*r12/L + 1.0*L*r13*(A*G/4 + E*Izz/L**2))
            k += 1
            KC0v[k] += r21*(-0.5*Ay*G*r12 - 0.5*Az*G*r13 + 1.0*G*J*r11/L) + r22*(-0.5*Ay*G*r11 - 1.0*E*Iyz*r13/L + 1.0*L*r12*(A*G/4 + E*Iyy/L**2)) + r23*(-0.5*Az*G*r11 - 1.0*E*Iyz*r12/L + 1.0*L*r13*(A*G/4 + E*Izz/L**2))
            k += 1
            KC0v[k] += r31*(-0.5*Ay*G*r12 - 0.5*Az*G*r13 + 1.0*G*J*r11/L) + r32*(-0.5*Ay*G*r11 - 1.0*E*Iyz*r13/L + 1.0*L*r12*(A*G/4 + E*Iyy/L**2)) + r33*(-0.5*Az*G*r11 - 1.0*E*Iyz*r12/L + 1.0*L*r13*(A*G/4 + E*Izz/L**2))
            k += 1
            KC0v[k] += r11*(1.0*Ay*E*r13/L - 1.0*Az*E*r12/L) + r12*(-0.5*A*G*r13 + 1.0*Az*G*r11/L) + r13*(0.5*A*G*r12 - 1.0*Ay*G*r11/L)
            k += 1
            KC0v[k] += r21*(1.0*Ay*E*r13/L - 1.0*Az*E*r12/L) + r22*(-0.5*A*G*r13 + 1.0*Az*G*r11/L) + r23*(0.5*A*G*r12 - 1.0*Ay*G*r11/L)
            k += 1
            KC0v[k] += r31*(1.0*Ay*E*r13/L - 1.0*Az*E*r12/L) + r32*(-0.5*A*G*r13 + 1.0*Az*G*r11/L) + r33*(0.5*A*G*r12 - 1.0*Ay*G*r11/L)
            k += 1
            KC0v[k] += r11*(0.5*Ay*G*r12 + 0.5*Az*G*r13 - 1.0*G*J*r11/L) + r12*(-0.5*Ay*G*r11 + 1.0*E*Iyz*r13/L + 1.0*L*r12*(A*G/4 - E*Iyy/L**2)) + r13*(-0.5*Az*G*r11 + 1.0*E*Iyz*r12/L + 1.0*L*r13*(A*G/4 - E*Izz/L**2))
            k += 1
            KC0v[k] += r21*(0.5*Ay*G*r12 + 0.5*Az*G*r13 - 1.0*G*J*r11/L) + r22*(-0.5*Ay*G*r11 + 1.0*E*Iyz*r13/L + 1.0*L*r12*(A*G/4 - E*Iyy/L**2)) + r23*(-0.5*Az*G*r11 + 1.0*E*Iyz*r12/L + 1.0*L*r13*(A*G/4 - E*Izz/L**2))
            k += 1
            KC0v[k] += r31*(0.5*Ay*G*r12 + 0.5*Az*G*r13 - 1.0*G*J*r11/L) + r32*(-0.5*Ay*G*r11 + 1.0*E*Iyz*r13/L + 1.0*L*r12*(A*G/4 - E*Iyy/L**2)) + r33*(-0.5*Az*G*r11 + 1.0*E*Iyz*r12/L + 1.0*L*r13*(A*G/4 - E*Izz/L**2))
            k += 1
            KC0v[k] += r11*(-1.0*Ay*E*r23/L + 1.0*Az*E*r22/L) + r12*(0.5*A*G*r23 - 1.0*Az*G*r21/L) + r13*(-0.5*A*G*r22 + 1.0*Ay*G*r21/L)
            k += 1
            KC0v[k] += r21*(-1.0*Ay*E*r23/L + 1.0*Az*E*r22/L) + r22*(0.5*A*G*r23 - 1.0*Az*G*r21/L) + r23*(-0.5*A*G*r22 + 1.0*Ay*G*r21/L)
            k += 1
            KC0v[k] += r31*(-1.0*Ay*E*r23/L + 1.0*Az*E*r22/L) + r32*(0.5*A*G*r23 - 1.0*Az*G*r21/L) + r33*(-0.5*A*G*r22 + 1.0*Ay*G*r21/L)
            k += 1
            KC0v[k] += r11*(-0.5*Ay*G*r22 - 0.5*Az*G*r23 + 1.0*G*J*r21/L) + r12*(-0.5*Ay*G*r21 - 1.0*E*Iyz*r23/L + 1.0*L*r22*(A*G/4 + E*Iyy/L**2)) + r13*(-0.5*Az*G*r21 - 1.0*E*Iyz*r22/L + 1.0*L*r23*(A*G/4 + E*Izz/L**2))
            k += 1
            KC0v[k] += r21*(-0.5*Ay*G*r22 - 0.5*Az*G*r23 + 1.0*G*J*r21/L) + r22*(-0.5*Ay*G*r21 - 1.0*E*Iyz*r23/L + 1.0*L*r22*(A*G/4 + E*Iyy/L**2)) + r23*(-0.5*Az*G*r21 - 1.0*E*Iyz*r22/L + 1.0*L*r23*(A*G/4 + E*Izz/L**2))
            k += 1
            KC0v[k] += r31*(-0.5*Ay*G*r22 - 0.5*Az*G*r23 + 1.0*G*J*r21/L) + r32*(-0.5*Ay*G*r21 - 1.0*E*Iyz*r23/L + 1.0*L*r22*(A*G/4 + E*Iyy/L**2)) + r33*(-0.5*Az*G*r21 - 1.0*E*Iyz*r22/L + 1.0*L*r23*(A*G/4 + E*Izz/L**2))
            k += 1
            KC0v[k] += r11*(1.0*Ay*E*r23/L - 1.0*Az*E*r22/L) + r12*(-0.5*A*G*r23 + 1.0*Az*G*r21/L) + r13*(0.5*A*G*r22 - 1.0*Ay*G*r21/L)
            k += 1
            KC0v[k] += r21*(1.0*Ay*E*r23/L - 1.0*Az*E*r22/L) + r22*(-0.5*A*G*r23 + 1.0*Az*G*r21/L) + r23*(0.5*A*G*r22 - 1.0*Ay*G*r21/L)
            k += 1
            KC0v[k] += r31*(1.0*Ay*E*r23/L - 1.0*Az*E*r22/L) + r32*(-0.5*A*G*r23 + 1.0*Az*G*r21/L) + r33*(0.5*A*G*r22 - 1.0*Ay*G*r21/L)
            k += 1
            KC0v[k] += r11*(0.5*Ay*G*r22 + 0.5*Az*G*r23 - 1.0*G*J*r21/L) + r12*(-0.5*Ay*G*r21 + 1.0*E*Iyz*r23/L + 1.0*L*r22*(A*G/4 - E*Iyy/L**2)) + r13*(-0.5*Az*G*r21 + 1.0*E*Iyz*r22/L + 1.0*L*r23*(A*G/4 - E*Izz/L**2))
            k += 1
            KC0v[k] += r21*(0.5*Ay*G*r22 + 0.5*Az*G*r23 - 1.0*G*J*r21/L) + r22*(-0.5*Ay*G*r21 + 1.0*E*Iyz*r23/L + 1.0*L*r22*(A*G/4 - E*Iyy/L**2)) + r23*(-0.5*Az*G*r21 + 1.0*E*Iyz*r22/L + 1.0*L*r23*(A*G/4 - E*Izz/L**2))
            k += 1
            KC0v[k] += r31*(0.5*Ay*G*r22 + 0.5*Az*G*r23 - 1.0*G*J*r21/L) + r32*(-0.5*Ay*G*r21 + 1.0*E*Iyz*r23/L + 1.0*L*r22*(A*G/4 - E*Iyy/L**2)) + r33*(-0.5*Az*G*r21 + 1.0*E*Iyz*r22/L + 1.0*L*r23*(A*G/4 - E*Izz/L**2))
            k += 1
            KC0v[k] += r11*(-1.0*Ay*E*r33/L + 1.0*Az*E*r32/L) + r12*(0.5*A*G*r33 - 1.0*Az*G*r31/L) + r13*(-0.5*A*G*r32 + 1.0*Ay*G*r31/L)
            k += 1
            KC0v[k] += r21*(-1.0*Ay*E*r33/L + 1.0*Az*E*r32/L) + r22*(0.5*A*G*r33 - 1.0*Az*G*r31/L) + r23*(-0.5*A*G*r32 + 1.0*Ay*G*r31/L)
            k += 1
            KC0v[k] += r31*(-1.0*Ay*E*r33/L + 1.0*Az*E*r32/L) + r32*(0.5*A*G*r33 - 1.0*Az*G*r31/L) + r33*(-0.5*A*G*r32 + 1.0*Ay*G*r31/L)
            k += 1
            KC0v[k] += r11*(-0.5*Ay*G*r32 - 0.5*Az*G*r33 + 1.0*G*J*r31/L) + r12*(-0.5*Ay*G*r31 - 1.0*E*Iyz*r33/L + 1.0*L*r32*(A*G/4 + E*Iyy/L**2)) + r13*(-0.5*Az*G*r31 - 1.0*E*Iyz*r32/L + 1.0*L*r33*(A*G/4 + E*Izz/L**2))
            k += 1
            KC0v[k] += r21*(-0.5*Ay*G*r32 - 0.5*Az*G*r33 + 1.0*G*J*r31/L) + r22*(-0.5*Ay*G*r31 - 1.0*E*Iyz*r33/L + 1.0*L*r32*(A*G/4 + E*Iyy/L**2)) + r23*(-0.5*Az*G*r31 - 1.0*E*Iyz*r32/L + 1.0*L*r33*(A*G/4 + E*Izz/L**2))
            k += 1
            KC0v[k] += r31*(-0.5*Ay*G*r32 - 0.5*Az*G*r33 + 1.0*G*J*r31/L) + r32*(-0.5*Ay*G*r31 - 1.0*E*Iyz*r33/L + 1.0*L*r32*(A*G/4 + E*Iyy/L**2)) + r33*(-0.5*Az*G*r31 - 1.0*E*Iyz*r32/L + 1.0*L*r33*(A*G/4 + E*Izz/L**2))
            k += 1
            KC0v[k] += r11*(1.0*Ay*E*r33/L - 1.0*Az*E*r32/L) + r12*(-0.5*A*G*r33 + 1.0*Az*G*r31/L) + r13*(0.5*A*G*r32 - 1.0*Ay*G*r31/L)
            k += 1
            KC0v[k] += r21*(1.0*Ay*E*r33/L - 1.0*Az*E*r32/L) + r22*(-0.5*A*G*r33 + 1.0*Az*G*r31/L) + r23*(0.5*A*G*r32 - 1.0*Ay*G*r31/L)
            k += 1
            KC0v[k] += r31*(1.0*Ay*E*r33/L - 1.0*Az*E*r32/L) + r32*(-0.5*A*G*r33 + 1.0*Az*G*r31/L) + r33*(0.5*A*G*r32 - 1.0*Ay*G*r31/L)
            k += 1
            KC0v[k] += r11*(0.5*Ay*G*r32 + 0.5*Az*G*r33 - 1.0*G*J*r31/L) + r12*(-0.5*Ay*G*r31 + 1.0*E*Iyz*r33/L + 1.0*L*r32*(A*G/4 - E*Iyy/L**2)) + r13*(-0.5*Az*G*r31 + 1.0*E*Iyz*r32/L + 1.0*L*r33*(A*G/4 - E*Izz/L**2))
            k += 1
            KC0v[k] += r21*(0.5*Ay*G*r32 + 0.5*Az*G*r33 - 1.0*G*J*r31/L) + r22*(-0.5*Ay*G*r31 + 1.0*E*Iyz*r33/L + 1.0*L*r32*(A*G/4 - E*Iyy/L**2)) + r23*(-0.5*Az*G*r31 + 1.0*E*Iyz*r32/L + 1.0*L*r33*(A*G/4 - E*Izz/L**2))
            k += 1
            KC0v[k] += r31*(0.5*Ay*G*r32 + 0.5*Az*G*r33 - 1.0*G*J*r31/L) + r32*(-0.5*Ay*G*r31 + 1.0*E*Iyz*r33/L + 1.0*L*r32*(A*G/4 - E*Iyy/L**2)) + r33*(-0.5*Az*G*r31 + 1.0*E*Iyz*r32/L + 1.0*L*r33*(A*G/4 - E*Izz/L**2))
            k += 1
            KC0v[k] += -1.0*A*E*r11**2/L - 1.0*A*G*r12**2/L - 1.0*A*G*r13**2/L
            k += 1
            KC0v[k] += -1.0*A*E*r11*r21/L - 1.0*A*G*r12*r22/L - 1.0*A*G*r13*r23/L
            k += 1
            KC0v[k] += -1.0*A*E*r11*r31/L - 1.0*A*G*r12*r32/L - 1.0*A*G*r13*r33/L
            k += 1
            KC0v[k] += r11*(-1.0*Ay*G*r13/L + 1.0*Az*G*r12/L) + r12*(0.5*A*G*r13 - 1.0*Az*E*r11/L) + r13*(-0.5*A*G*r12 + 1.0*Ay*E*r11/L)
            k += 1
            KC0v[k] += r21*(-1.0*Ay*G*r13/L + 1.0*Az*G*r12/L) + r22*(0.5*A*G*r13 - 1.0*Az*E*r11/L) + r23*(-0.5*A*G*r12 + 1.0*Ay*E*r11/L)
            k += 1
            KC0v[k] += r31*(-1.0*Ay*G*r13/L + 1.0*Az*G*r12/L) + r32*(0.5*A*G*r13 - 1.0*Az*E*r11/L) + r33*(-0.5*A*G*r12 + 1.0*Ay*E*r11/L)
            k += 1
            KC0v[k] += 1.0*A*E*r11**2/L + 1.0*A*G*r12**2/L + 1.0*A*G*r13**2/L
            k += 1
            KC0v[k] += 1.0*A*E*r11*r21/L + 1.0*A*G*r12*r22/L + 1.0*A*G*r13*r23/L
            k += 1
            KC0v[k] += 1.0*A*E*r11*r31/L + 1.0*A*G*r12*r32/L + 1.0*A*G*r13*r33/L
            k += 1
            KC0v[k] += r11*(1.0*Ay*G*r13/L - 1.0*Az*G*r12/L) + r12*(0.5*A*G*r13 + 1.0*Az*E*r11/L) + r13*(-0.5*A*G*r12 - 1.0*Ay*E*r11/L)
            k += 1
            KC0v[k] += r21*(1.0*Ay*G*r13/L - 1.0*Az*G*r12/L) + r22*(0.5*A*G*r13 + 1.0*Az*E*r11/L) + r23*(-0.5*A*G*r12 - 1.0*Ay*E*r11/L)
            k += 1
            KC0v[k] += r31*(1.0*Ay*G*r13/L - 1.0*Az*G*r12/L) + r32*(0.5*A*G*r13 + 1.0*Az*E*r11/L) + r33*(-0.5*A*G*r12 - 1.0*Ay*E*r11/L)
            k += 1
            KC0v[k] += -1.0*A*E*r11*r21/L - 1.0*A*G*r12*r22/L - 1.0*A*G*r13*r23/L
            k += 1
            KC0v[k] += -1.0*A*E*r21**2/L - 1.0*A*G*r22**2/L - 1.0*A*G*r23**2/L
            k += 1
            KC0v[k] += -1.0*A*E*r21*r31/L - 1.0*A*G*r22*r32/L - 1.0*A*G*r23*r33/L
            k += 1
            KC0v[k] += r11*(-1.0*Ay*G*r23/L + 1.0*Az*G*r22/L) + r12*(0.5*A*G*r23 - 1.0*Az*E*r21/L) + r13*(-0.5*A*G*r22 + 1.0*Ay*E*r21/L)
            k += 1
            KC0v[k] += r21*(-1.0*Ay*G*r23/L + 1.0*Az*G*r22/L) + r22*(0.5*A*G*r23 - 1.0*Az*E*r21/L) + r23*(-0.5*A*G*r22 + 1.0*Ay*E*r21/L)
            k += 1
            KC0v[k] += r31*(-1.0*Ay*G*r23/L + 1.0*Az*G*r22/L) + r32*(0.5*A*G*r23 - 1.0*Az*E*r21/L) + r33*(-0.5*A*G*r22 + 1.0*Ay*E*r21/L)
            k += 1
            KC0v[k] += 1.0*A*E*r11*r21/L + 1.0*A*G*r12*r22/L + 1.0*A*G*r13*r23/L
            k += 1
            KC0v[k] += 1.0*A*E*r21**2/L + 1.0*A*G*r22**2/L + 1.0*A*G*r23**2/L
            k += 1
            KC0v[k] += 1.0*A*E*r21*r31/L + 1.0*A*G*r22*r32/L + 1.0*A*G*r23*r33/L
            k += 1
            KC0v[k] += r11*(1.0*Ay*G*r23/L - 1.0*Az*G*r22/L) + r12*(0.5*A*G*r23 + 1.0*Az*E*r21/L) + r13*(-0.5*A*G*r22 - 1.0*Ay*E*r21/L)
            k += 1
            KC0v[k] += r21*(1.0*Ay*G*r23/L - 1.0*Az*G*r22/L) + r22*(0.5*A*G*r23 + 1.0*Az*E*r21/L) + r23*(-0.5*A*G*r22 - 1.0*Ay*E*r21/L)
            k += 1
            KC0v[k] += r31*(1.0*Ay*G*r23/L - 1.0*Az*G*r22/L) + r32*(0.5*A*G*r23 + 1.0*Az*E*r21/L) + r33*(-0.5*A*G*r22 - 1.0*Ay*E*r21/L)
            k += 1
            KC0v[k] += -1.0*A*E*r11*r31/L - 1.0*A*G*r12*r32/L - 1.0*A*G*r13*r33/L
            k += 1
            KC0v[k] += -1.0*A*E*r21*r31/L - 1.0*A*G*r22*r32/L - 1.0*A*G*r23*r33/L
            k += 1
            KC0v[k] += -1.0*A*E*r31**2/L - 1.0*A*G*r32**2/L - 1.0*A*G*r33**2/L
            k += 1
            KC0v[k] += r11*(-1.0*Ay*G*r33/L + 1.0*Az*G*r32/L) + r12*(0.5*A*G*r33 - 1.0*Az*E*r31/L) + r13*(-0.5*A*G*r32 + 1.0*Ay*E*r31/L)
            k += 1
            KC0v[k] += r21*(-1.0*Ay*G*r33/L + 1.0*Az*G*r32/L) + r22*(0.5*A*G*r33 - 1.0*Az*E*r31/L) + r23*(-0.5*A*G*r32 + 1.0*Ay*E*r31/L)
            k += 1
            KC0v[k] += r31*(-1.0*Ay*G*r33/L + 1.0*Az*G*r32/L) + r32*(0.5*A*G*r33 - 1.0*Az*E*r31/L) + r33*(-0.5*A*G*r32 + 1.0*Ay*E*r31/L)
            k += 1
            KC0v[k] += 1.0*A*E*r11*r31/L + 1.0*A*G*r12*r32/L + 1.0*A*G*r13*r33/L
            k += 1
            KC0v[k] += 1.0*A*E*r21*r31/L + 1.0*A*G*r22*r32/L + 1.0*A*G*r23*r33/L
            k += 1
            KC0v[k] += 1.0*A*E*r31**2/L + 1.0*A*G*r32**2/L + 1.0*A*G*r33**2/L
            k += 1
            KC0v[k] += r11*(1.0*Ay*G*r33/L - 1.0*Az*G*r32/L) + r12*(0.5*A*G*r33 + 1.0*Az*E*r31/L) + r13*(-0.5*A*G*r32 - 1.0*Ay*E*r31/L)
            k += 1
            KC0v[k] += r21*(1.0*Ay*G*r33/L - 1.0*Az*G*r32/L) + r22*(0.5*A*G*r33 + 1.0*Az*E*r31/L) + r23*(-0.5*A*G*r32 - 1.0*Ay*E*r31/L)
            k += 1
            KC0v[k] += r31*(1.0*Ay*G*r33/L - 1.0*Az*G*r32/L) + r32*(0.5*A*G*r33 + 1.0*Az*E*r31/L) + r33*(-0.5*A*G*r32 - 1.0*Ay*E*r31/L)
            k += 1
            KC0v[k] += r11*(1.0*Ay*E*r13/L - 1.0*Az*E*r12/L) + r12*(0.5*A*G*r13 + 1.0*Az*G*r11/L) + r13*(-0.5*A*G*r12 - 1.0*Ay*G*r11/L)
            k += 1
            KC0v[k] += r21*(1.0*Ay*E*r13/L - 1.0*Az*E*r12/L) + r22*(0.5*A*G*r13 + 1.0*Az*G*r11/L) + r23*(-0.5*A*G*r12 - 1.0*Ay*G*r11/L)
            k += 1
            KC0v[k] += r31*(1.0*Ay*E*r13/L - 1.0*Az*E*r12/L) + r32*(0.5*A*G*r13 + 1.0*Az*G*r11/L) + r33*(-0.5*A*G*r12 - 1.0*Ay*G*r11/L)
            k += 1
            KC0v[k] += r11*(-0.5*Ay*G*r12 - 0.5*Az*G*r13 - 1.0*G*J*r11/L) + r12*(0.5*Ay*G*r11 + 1.0*E*Iyz*r13/L + 1.0*L*r12*(A*G/4 - E*Iyy/L**2)) + r13*(0.5*Az*G*r11 + 1.0*E*Iyz*r12/L + 1.0*L*r13*(A*G/4 - E*Izz/L**2))
            k += 1
            KC0v[k] += r21*(-0.5*Ay*G*r12 - 0.5*Az*G*r13 - 1.0*G*J*r11/L) + r22*(0.5*Ay*G*r11 + 1.0*E*Iyz*r13/L + 1.0*L*r12*(A*G/4 - E*Iyy/L**2)) + r23*(0.5*Az*G*r11 + 1.0*E*Iyz*r12/L + 1.0*L*r13*(A*G/4 - E*Izz/L**2))
            k += 1
            KC0v[k] += r31*(-0.5*Ay*G*r12 - 0.5*Az*G*r13 - 1.0*G*J*r11/L) + r32*(0.5*Ay*G*r11 + 1.0*E*Iyz*r13/L + 1.0*L*r12*(A*G/4 - E*Iyy/L**2)) + r33*(0.5*Az*G*r11 + 1.0*E*Iyz*r12/L + 1.0*L*r13*(A*G/4 - E*Izz/L**2))
            k += 1
            KC0v[k] += r11*(-1.0*Ay*E*r13/L + 1.0*Az*E*r12/L) + r12*(-0.5*A*G*r13 - 1.0*Az*G*r11/L) + r13*(0.5*A*G*r12 + 1.0*Ay*G*r11/L)
            k += 1
            KC0v[k] += r21*(-1.0*Ay*E*r13/L + 1.0*Az*E*r12/L) + r22*(-0.5*A*G*r13 - 1.0*Az*G*r11/L) + r23*(0.5*A*G*r12 + 1.0*Ay*G*r11/L)
            k += 1
            KC0v[k] += r31*(-1.0*Ay*E*r13/L + 1.0*Az*E*r12/L) + r32*(-0.5*A*G*r13 - 1.0*Az*G*r11/L) + r33*(0.5*A*G*r12 + 1.0*Ay*G*r11/L)
            k += 1
            KC0v[k] += r11*(0.5*Ay*G*r12 + 0.5*Az*G*r13 + 1.0*G*J*r11/L) + r12*(0.5*Ay*G*r11 - 1.0*E*Iyz*r13/L + 1.0*L*r12*(A*G/4 + E*Iyy/L**2)) + r13*(0.5*Az*G*r11 - 1.0*E*Iyz*r12/L + 1.0*L*r13*(A*G/4 + E*Izz/L**2))
            k += 1
            KC0v[k] += r21*(0.5*Ay*G*r12 + 0.5*Az*G*r13 + 1.0*G*J*r11/L) + r22*(0.5*Ay*G*r11 - 1.0*E*Iyz*r13/L + 1.0*L*r12*(A*G/4 + E*Iyy/L**2)) + r23*(0.5*Az*G*r11 - 1.0*E*Iyz*r12/L + 1.0*L*r13*(A*G/4 + E*Izz/L**2))
            k += 1
            KC0v[k] += r31*(0.5*Ay*G*r12 + 0.5*Az*G*r13 + 1.0*G*J*r11/L) + r32*(0.5*Ay*G*r11 - 1.0*E*Iyz*r13/L + 1.0*L*r12*(A*G/4 + E*Iyy/L**2)) + r33*(0.5*Az*G*r11 - 1.0*E*Iyz*r12/L + 1.0*L*r13*(A*G/4 + E*Izz/L**2))
            k += 1
            KC0v[k] += r11*(1.0*Ay*E*r23/L - 1.0*Az*E*r22/L) + r12*(0.5*A*G*r23 + 1.0*Az*G*r21/L) + r13*(-0.5*A*G*r22 - 1.0*Ay*G*r21/L)
            k += 1
            KC0v[k] += r21*(1.0*Ay*E*r23/L - 1.0*Az*E*r22/L) + r22*(0.5*A*G*r23 + 1.0*Az*G*r21/L) + r23*(-0.5*A*G*r22 - 1.0*Ay*G*r21/L)
            k += 1
            KC0v[k] += r31*(1.0*Ay*E*r23/L - 1.0*Az*E*r22/L) + r32*(0.5*A*G*r23 + 1.0*Az*G*r21/L) + r33*(-0.5*A*G*r22 - 1.0*Ay*G*r21/L)
            k += 1
            KC0v[k] += r11*(-0.5*Ay*G*r22 - 0.5*Az*G*r23 - 1.0*G*J*r21/L) + r12*(0.5*Ay*G*r21 + 1.0*E*Iyz*r23/L + 1.0*L*r22*(A*G/4 - E*Iyy/L**2)) + r13*(0.5*Az*G*r21 + 1.0*E*Iyz*r22/L + 1.0*L*r23*(A*G/4 - E*Izz/L**2))
            k += 1
            KC0v[k] += r21*(-0.5*Ay*G*r22 - 0.5*Az*G*r23 - 1.0*G*J*r21/L) + r22*(0.5*Ay*G*r21 + 1.0*E*Iyz*r23/L + 1.0*L*r22*(A*G/4 - E*Iyy/L**2)) + r23*(0.5*Az*G*r21 + 1.0*E*Iyz*r22/L + 1.0*L*r23*(A*G/4 - E*Izz/L**2))
            k += 1
            KC0v[k] += r31*(-0.5*Ay*G*r22 - 0.5*Az*G*r23 - 1.0*G*J*r21/L) + r32*(0.5*Ay*G*r21 + 1.0*E*Iyz*r23/L + 1.0*L*r22*(A*G/4 - E*Iyy/L**2)) + r33*(0.5*Az*G*r21 + 1.0*E*Iyz*r22/L + 1.0*L*r23*(A*G/4 - E*Izz/L**2))
            k += 1
            KC0v[k] += r11*(-1.0*Ay*E*r23/L + 1.0*Az*E*r22/L) + r12*(-0.5*A*G*r23 - 1.0*Az*G*r21/L) + r13*(0.5*A*G*r22 + 1.0*Ay*G*r21/L)
            k += 1
            KC0v[k] += r21*(-1.0*Ay*E*r23/L + 1.0*Az*E*r22/L) + r22*(-0.5*A*G*r23 - 1.0*Az*G*r21/L) + r23*(0.5*A*G*r22 + 1.0*Ay*G*r21/L)
            k += 1
            KC0v[k] += r31*(-1.0*Ay*E*r23/L + 1.0*Az*E*r22/L) + r32*(-0.5*A*G*r23 - 1.0*Az*G*r21/L) + r33*(0.5*A*G*r22 + 1.0*Ay*G*r21/L)
            k += 1
            KC0v[k] += r11*(0.5*Ay*G*r22 + 0.5*Az*G*r23 + 1.0*G*J*r21/L) + r12*(0.5*Ay*G*r21 - 1.0*E*Iyz*r23/L + 1.0*L*r22*(A*G/4 + E*Iyy/L**2)) + r13*(0.5*Az*G*r21 - 1.0*E*Iyz*r22/L + 1.0*L*r23*(A*G/4 + E*Izz/L**2))
            k += 1
            KC0v[k] += r21*(0.5*Ay*G*r22 + 0.5*Az*G*r23 + 1.0*G*J*r21/L) + r22*(0.5*Ay*G*r21 - 1.0*E*Iyz*r23/L + 1.0*L*r22*(A*G/4 + E*Iyy/L**2)) + r23*(0.5*Az*G*r21 - 1.0*E*Iyz*r22/L + 1.0*L*r23*(A*G/4 + E*Izz/L**2))
            k += 1
            KC0v[k] += r31*(0.5*Ay*G*r22 + 0.5*Az*G*r23 + 1.0*G*J*r21/L) + r32*(0.5*Ay*G*r21 - 1.0*E*Iyz*r23/L + 1.0*L*r22*(A*G/4 + E*Iyy/L**2)) + r33*(0.5*Az*G*r21 - 1.0*E*Iyz*r22/L + 1.0*L*r23*(A*G/4 + E*Izz/L**2))
            k += 1
            KC0v[k] += r11*(1.0*Ay*E*r33/L - 1.0*Az*E*r32/L) + r12*(0.5*A*G*r33 + 1.0*Az*G*r31/L) + r13*(-0.5*A*G*r32 - 1.0*Ay*G*r31/L)
            k += 1
            KC0v[k] += r21*(1.0*Ay*E*r33/L - 1.0*Az*E*r32/L) + r22*(0.5*A*G*r33 + 1.0*Az*G*r31/L) + r23*(-0.5*A*G*r32 - 1.0*Ay*G*r31/L)
            k += 1
            KC0v[k] += r31*(1.0*Ay*E*r33/L - 1.0*Az*E*r32/L) + r32*(0.5*A*G*r33 + 1.0*Az*G*r31/L) + r33*(-0.5*A*G*r32 - 1.0*Ay*G*r31/L)
            k += 1
            KC0v[k] += r11*(-0.5*Ay*G*r32 - 0.5*Az*G*r33 - 1.0*G*J*r31/L) + r12*(0.5*Ay*G*r31 + 1.0*E*Iyz*r33/L + 1.0*L*r32*(A*G/4 - E*Iyy/L**2)) + r13*(0.5*Az*G*r31 + 1.0*E*Iyz*r32/L + 1.0*L*r33*(A*G/4 - E*Izz/L**2))
            k += 1
            KC0v[k] += r21*(-0.5*Ay*G*r32 - 0.5*Az*G*r33 - 1.0*G*J*r31/L) + r22*(0.5*Ay*G*r31 + 1.0*E*Iyz*r33/L + 1.0*L*r32*(A*G/4 - E*Iyy/L**2)) + r23*(0.5*Az*G*r31 + 1.0*E*Iyz*r32/L + 1.0*L*r33*(A*G/4 - E*Izz/L**2))
            k += 1
            KC0v[k] += r31*(-0.5*Ay*G*r32 - 0.5*Az*G*r33 - 1.0*G*J*r31/L) + r32*(0.5*Ay*G*r31 + 1.0*E*Iyz*r33/L + 1.0*L*r32*(A*G/4 - E*Iyy/L**2)) + r33*(0.5*Az*G*r31 + 1.0*E*Iyz*r32/L + 1.0*L*r33*(A*G/4 - E*Izz/L**2))
            k += 1
            KC0v[k] += r11*(-1.0*Ay*E*r33/L + 1.0*Az*E*r32/L) + r12*(-0.5*A*G*r33 - 1.0*Az*G*r31/L) + r13*(0.5*A*G*r32 + 1.0*Ay*G*r31/L)
            k += 1
            KC0v[k] += r21*(-1.0*Ay*E*r33/L + 1.0*Az*E*r32/L) + r22*(-0.5*A*G*r33 - 1.0*Az*G*r31/L) + r23*(0.5*A*G*r32 + 1.0*Ay*G*r31/L)
            k += 1
            KC0v[k] += r31*(-1.0*Ay*E*r33/L + 1.0*Az*E*r32/L) + r32*(-0.5*A*G*r33 - 1.0*Az*G*r31/L) + r33*(0.5*A*G*r32 + 1.0*Ay*G*r31/L)
            k += 1
            KC0v[k] += r11*(0.5*Ay*G*r32 + 0.5*Az*G*r33 + 1.0*G*J*r31/L) + r12*(0.5*Ay*G*r31 - 1.0*E*Iyz*r33/L + 1.0*L*r32*(A*G/4 + E*Iyy/L**2)) + r13*(0.5*Az*G*r31 - 1.0*E*Iyz*r32/L + 1.0*L*r33*(A*G/4 + E*Izz/L**2))
            k += 1
            KC0v[k] += r21*(0.5*Ay*G*r32 + 0.5*Az*G*r33 + 1.0*G*J*r31/L) + r22*(0.5*Ay*G*r31 - 1.0*E*Iyz*r33/L + 1.0*L*r32*(A*G/4 + E*Iyy/L**2)) + r23*(0.5*Az*G*r31 - 1.0*E*Iyz*r32/L + 1.0*L*r33*(A*G/4 + E*Izz/L**2))
            k += 1
            KC0v[k] += r31*(0.5*Ay*G*r32 + 0.5*Az*G*r33 + 1.0*G*J*r31/L) + r32*(0.5*Ay*G*r31 - 1.0*E*Iyz*r33/L + 1.0*L*r32*(A*G/4 + E*Iyy/L**2)) + r33*(0.5*Az*G*r31 - 1.0*E*Iyz*r32/L + 1.0*L*r33*(A*G/4 + E*Izz/L**2))


    cpdef void update_KG(BeamLR self,
                         long [::1] KGr,
                         long [::1] KGc,
                         double [::1] KGv,
                         BeamProp prop,
                         int update_KGv_only=0,
                         ):
        r"""Update sparse vectors for geometric stiffness matrix KG

        Properties
        ----------
        KGr : np.array
           Array to store row positions of sparse values
        KGc : np.array
           Array to store column positions of sparse values
        KGv : np.array
            Array to store sparse values
        prop : :class:`BeamProp` object
            Beam property object from where the stiffness and mass attributes
            are read from.
        update_KGv_only : int
            The default `0` means that only `KGv` is updated. Any other value will
            lead to `KGr` and `KGc` also being updated.

        """
        cdef double *ue
        cdef int c1, c2, k
        cdef double L, A, E, N
        cdef double r11, r12, r13, r21, r22, r23, r31, r32, r33

        with nogil:
            L = self.length
            A = prop.A
            E = prop.E

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

            N = A*E*(-ue[0] + ue[6])/L

            k = self.init_k_KG
            KGv[k] += r12*(0.333333333333333*L*N*r12 - 0.333333333333333*L*N*r13) + r13*(-0.333333333333333*L*N*r12 + 0.333333333333333*L*N*r13)
            k += 1
            KGv[k] += r22*(0.333333333333333*L*N*r12 - 0.333333333333333*L*N*r13) + r23*(-0.333333333333333*L*N*r12 + 0.333333333333333*L*N*r13)
            k += 1
            KGv[k] += r32*(0.333333333333333*L*N*r12 - 0.333333333333333*L*N*r13) + r33*(-0.333333333333333*L*N*r12 + 0.333333333333333*L*N*r13)
            k += 1
            KGv[k] += r12*(0.166666666666667*L*N*r12 - 0.166666666666667*L*N*r13) + r13*(-0.166666666666667*L*N*r12 + 0.166666666666667*L*N*r13)
            k += 1
            KGv[k] += r22*(0.166666666666667*L*N*r12 - 0.166666666666667*L*N*r13) + r23*(-0.166666666666667*L*N*r12 + 0.166666666666667*L*N*r13)
            k += 1
            KGv[k] += r32*(0.166666666666667*L*N*r12 - 0.166666666666667*L*N*r13) + r33*(-0.166666666666667*L*N*r12 + 0.166666666666667*L*N*r13)
            k += 1
            KGv[k] += r12*(0.333333333333333*L*N*r22 - 0.333333333333333*L*N*r23) + r13*(-0.333333333333333*L*N*r22 + 0.333333333333333*L*N*r23)
            k += 1
            KGv[k] += r22*(0.333333333333333*L*N*r22 - 0.333333333333333*L*N*r23) + r23*(-0.333333333333333*L*N*r22 + 0.333333333333333*L*N*r23)
            k += 1
            KGv[k] += r32*(0.333333333333333*L*N*r22 - 0.333333333333333*L*N*r23) + r33*(-0.333333333333333*L*N*r22 + 0.333333333333333*L*N*r23)
            k += 1
            KGv[k] += r12*(0.166666666666667*L*N*r22 - 0.166666666666667*L*N*r23) + r13*(-0.166666666666667*L*N*r22 + 0.166666666666667*L*N*r23)
            k += 1
            KGv[k] += r22*(0.166666666666667*L*N*r22 - 0.166666666666667*L*N*r23) + r23*(-0.166666666666667*L*N*r22 + 0.166666666666667*L*N*r23)
            k += 1
            KGv[k] += r32*(0.166666666666667*L*N*r22 - 0.166666666666667*L*N*r23) + r33*(-0.166666666666667*L*N*r22 + 0.166666666666667*L*N*r23)
            k += 1
            KGv[k] += r12*(0.333333333333333*L*N*r32 - 0.333333333333333*L*N*r33) + r13*(-0.333333333333333*L*N*r32 + 0.333333333333333*L*N*r33)
            k += 1
            KGv[k] += r22*(0.333333333333333*L*N*r32 - 0.333333333333333*L*N*r33) + r23*(-0.333333333333333*L*N*r32 + 0.333333333333333*L*N*r33)
            k += 1
            KGv[k] += r32*(0.333333333333333*L*N*r32 - 0.333333333333333*L*N*r33) + r33*(-0.333333333333333*L*N*r32 + 0.333333333333333*L*N*r33)
            k += 1
            KGv[k] += r12*(0.166666666666667*L*N*r32 - 0.166666666666667*L*N*r33) + r13*(-0.166666666666667*L*N*r32 + 0.166666666666667*L*N*r33)
            k += 1
            KGv[k] += r22*(0.166666666666667*L*N*r32 - 0.166666666666667*L*N*r33) + r23*(-0.166666666666667*L*N*r32 + 0.166666666666667*L*N*r33)
            k += 1
            KGv[k] += r32*(0.166666666666667*L*N*r32 - 0.166666666666667*L*N*r33) + r33*(-0.166666666666667*L*N*r32 + 0.166666666666667*L*N*r33)
            k += 1
            KGv[k] += r12*(0.166666666666667*L*N*r12 - 0.166666666666667*L*N*r13) + r13*(-0.166666666666667*L*N*r12 + 0.166666666666667*L*N*r13)
            k += 1
            KGv[k] += r22*(0.166666666666667*L*N*r12 - 0.166666666666667*L*N*r13) + r23*(-0.166666666666667*L*N*r12 + 0.166666666666667*L*N*r13)
            k += 1
            KGv[k] += r32*(0.166666666666667*L*N*r12 - 0.166666666666667*L*N*r13) + r33*(-0.166666666666667*L*N*r12 + 0.166666666666667*L*N*r13)
            k += 1
            KGv[k] += r12*(0.333333333333333*L*N*r12 - 0.333333333333333*L*N*r13) + r13*(-0.333333333333333*L*N*r12 + 0.333333333333333*L*N*r13)
            k += 1
            KGv[k] += r22*(0.333333333333333*L*N*r12 - 0.333333333333333*L*N*r13) + r23*(-0.333333333333333*L*N*r12 + 0.333333333333333*L*N*r13)
            k += 1
            KGv[k] += r32*(0.333333333333333*L*N*r12 - 0.333333333333333*L*N*r13) + r33*(-0.333333333333333*L*N*r12 + 0.333333333333333*L*N*r13)
            k += 1
            KGv[k] += r12*(0.166666666666667*L*N*r22 - 0.166666666666667*L*N*r23) + r13*(-0.166666666666667*L*N*r22 + 0.166666666666667*L*N*r23)
            k += 1
            KGv[k] += r22*(0.166666666666667*L*N*r22 - 0.166666666666667*L*N*r23) + r23*(-0.166666666666667*L*N*r22 + 0.166666666666667*L*N*r23)
            k += 1
            KGv[k] += r32*(0.166666666666667*L*N*r22 - 0.166666666666667*L*N*r23) + r33*(-0.166666666666667*L*N*r22 + 0.166666666666667*L*N*r23)
            k += 1
            KGv[k] += r12*(0.333333333333333*L*N*r22 - 0.333333333333333*L*N*r23) + r13*(-0.333333333333333*L*N*r22 + 0.333333333333333*L*N*r23)
            k += 1
            KGv[k] += r22*(0.333333333333333*L*N*r22 - 0.333333333333333*L*N*r23) + r23*(-0.333333333333333*L*N*r22 + 0.333333333333333*L*N*r23)
            k += 1
            KGv[k] += r32*(0.333333333333333*L*N*r22 - 0.333333333333333*L*N*r23) + r33*(-0.333333333333333*L*N*r22 + 0.333333333333333*L*N*r23)
            k += 1
            KGv[k] += r12*(0.166666666666667*L*N*r32 - 0.166666666666667*L*N*r33) + r13*(-0.166666666666667*L*N*r32 + 0.166666666666667*L*N*r33)
            k += 1
            KGv[k] += r22*(0.166666666666667*L*N*r32 - 0.166666666666667*L*N*r33) + r23*(-0.166666666666667*L*N*r32 + 0.166666666666667*L*N*r33)
            k += 1
            KGv[k] += r32*(0.166666666666667*L*N*r32 - 0.166666666666667*L*N*r33) + r33*(-0.166666666666667*L*N*r32 + 0.166666666666667*L*N*r33)
            k += 1
            KGv[k] += r12*(0.333333333333333*L*N*r32 - 0.333333333333333*L*N*r33) + r13*(-0.333333333333333*L*N*r32 + 0.333333333333333*L*N*r33)
            k += 1
            KGv[k] += r22*(0.333333333333333*L*N*r32 - 0.333333333333333*L*N*r33) + r23*(-0.333333333333333*L*N*r32 + 0.333333333333333*L*N*r33)
            k += 1
            KGv[k] += r32*(0.333333333333333*L*N*r32 - 0.333333333333333*L*N*r33) + r33*(-0.333333333333333*L*N*r32 + 0.333333333333333*L*N*r33)


    cpdef void update_M(BeamLR self,
                        long [::1] Mr,
                        long [::1] Mc,
                        double [::1] Mv,
                        BeamProp prop,
                        int mtype=0,
                        ):
        r"""Update sparse vectors for mass matrix M

        Properties
        ----------
        Mr : np.array
            Array to store row positions of sparse values
        Mc : np.array
            Array to store column positions of sparse values
        Mv : np.array
            Array to store sparse values
        mtype : int, optional
            0 for consistent mass matrix using method from Brockman 1987
            1 for reduced integration mass matrix using method from Brockman 1987
            2 for lumped mass matrix using method from Brockman 1987

        """
        cdef int c1, c2, k
        cdef double intrho, intrhoy, intrhoz, intrhoy2, intrhoz2, intrhoyz
        cdef double r11, r12, r13, r21, r22, r23, r31, r32, r33
        cdef double L, A, E

        with nogil:
            L = self.length
            intrho = prop.intrho
            intrhoy = prop.intrhoy
            intrhoz = prop.intrhoz
            intrhoy2 = prop.intrhoy2
            intrhoz2 = prop.intrhoz2
            intrhoyz = prop.intrhoyz
            A = prop.A
            E = prop.E

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
