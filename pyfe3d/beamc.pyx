#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: nonecheck=False
#cython: overflowcheck=False
#cython: embedsignature=True
#cython: infer_types=False
r"""
BeamC - Consistent Timoshenko 3D beam element (:mod:`pyfe3d.beamc`)
===================================================================

.. currentmodule:: pyfe3d.beamc

"""
import numpy as np

from .beamprop cimport BeamProp

cdef int DOF = 6
cdef int NUM_NODES = 2


cdef class BeamCData:
    r"""
    Used to allocate memory for the sparse matrices.

    Attributes
    ----------
    KC0_SPARSE_SIZE, : int
        ``KC0_SPARSE_SIZE = 144``

    KG_SPARSE_SIZE, : int
        ``KG_SPARSE_SIZE = 144``

    M_SPARSE_SIZE, : int
        ``M_SPARSE_SIZE = 144``

    """
    cdef public int KC0_SPARSE_SIZE
    cdef public int KG_SPARSE_SIZE
    cdef public int M_SPARSE_SIZE
    def __cinit__(BeamCData self):
        self.KC0_SPARSE_SIZE = 144
        self.KG_SPARSE_SIZE = 144
        self.M_SPARSE_SIZE = 144


cdef class BeamCProbe:
    r"""
    Probe used for local coordinates, local displacements, local stresses etc

    .. note:: The probe can be shared amongst more than one finite element, 
              depending how you defined them. Mind that the probe will always
              safe the values from the last udpate.


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
    finte, : array-like
        Array of size ``NUM_NODES*DOF=12`` containing the element internal
        forces corresponding to the degrees-of-freedom described by ``ue``.

    """
    cdef public double [::1] xe
    cdef public double [::1] ue
    cdef public double [::1] finte
    def __cinit__(BeamCProbe self):
        self.xe = np.zeros(NUM_NODES*DOF//2, dtype=np.float64)
        self.ue = np.zeros(NUM_NODES*DOF, dtype=np.float64)
        self.finte = np.zeros(NUM_NODES*DOF, dtype=np.float64)


cdef class BeamC:
    r"""
    Timoshenko 3D beam element with consistent shape functions

    Formulation based on reference:

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
    pid, : int
        Property identification number.
    length, : double
        Element length.
    r11, r12, r13, r21, r22, r23, r31, r32, r33 : double
        Rotation matrix from local to global coordinates.
    vxyi, vxyj, vxyk : double
        Components of a vector on the `XY` plane of the element coordinate
        system, defined using global coordinates.
    c1, c2 : int
        Position of each node in the global stiffness matrix.
    n1, n2 : int
        Node identification number.
    init_k_KC0, init_k_KG, init_k_M : int
        Position in the arrays storing the sparse data for the structural
        matrices.
    probe, : :class:`.BeamCProbe` object
        Pointer to the probe.

    """
    cdef public int eid, pid
    cdef public int n1, n2
    cdef public int c1, c2
    cdef public int init_k_KC0, init_k_KG, init_k_M
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
        # self.init_k_KCNL = 0
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

        Attributes ``r11,r12,r13,r21,r22,r23,r31,r32,r33`` are updated,
        corresponding to the rotation matrix from local to global coordinates.

        The element attributes ``vxyi``, ``vxyj`` and ``vxyk`` are also updated
        when this function is called.

        The element coordinate system is determined, identifying the `ijk`
        components of each axis: `{x_e}_i, {x_e}_j, {x_e}_k`; `{y_e}_i,
        {y_e}_j, {y_e}_k`; `{z_e}_i, {z_e}_j, {z_e}_k`.


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

        .. note:: The ``probe`` attribute object :class:`.BeamCProbe` is
                  updated, not the finite element.

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

        with nogil:
            # positions in the global stiffness matrix
            c[0] = self.c1
            c[1] = self.c2

            # global to local transformation of displacements (R.T)
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

        .. note:: The ``probe`` attribute object :class:`.BeamCProbe` is
                  updated, not the finite element.

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

            # global to local transformation of displacements (R.T)
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
                           BeamProp prop):
        r"""Update the internal force vector of the probe

        The attribute ``finte`` is updated with the :class:`.BeamCProbe` the
        internal forces in local coordinates. While using this function, mind
        that the probe can be shared amongst more than one finite element,
        depending how you defined them, meaning that the probe will always safe
        the values from the last udpate.

        .. note:: The ``probe`` attribute object :class:`.BeamCProbe` is
                  updated, not the finite element.

        Parameters
        ----------
        prop : :class:`.BeamProp` object
            Beam property object from where the stiffness and mass attributes
            are read from.

        """
        cdef double *ue
        cdef double *finte
        cdef double L, A, E, G, Ay, Az, Iyy, Izz, Iyz, J, alphay, alphaz, betay, betaz

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

            alphay = 12*E*Izz/(G*A*L**2)
            alphaz = 12*E*Iyy/(G*A*L**2)
            betay = 1/(1. - alphay)
            betaz = 1/(1. - alphaz)

            ue = &self.probe.ue[0]
            finte = &self.probe.finte[0]

            finte[0] = E*(A*ue[0] - A*ue[6] - Ay*alphay*betay*ue[11] + Ay*alphay*betay*ue[5] + Ay*betay*ue[11] - Ay*betay*ue[5] + Az*alphaz*betaz*ue[10] - Az*alphaz*betaz*ue[4] - Az*betaz*ue[10] + Az*betaz*ue[4])/L
            finte[1] = betay*(A*G*L**3*alphay**2*betay*ue[11] + A*G*L**3*alphay**2*betay*ue[5] + 2*A*G*L**2*alphay**2*betay*ue[1] - 2*A*G*L**2*alphay**2*betay*ue[7] + 2*Az*G*L**2*alphay*ue[3] - 2*Az*G*L**2*alphay*ue[9] - 12*E*Iyz*L*betaz*ue[10] - 12*E*Iyz*L*betaz*ue[4] + 24*E*Iyz*betaz*ue[2] - 24*E*Iyz*betaz*ue[8] + 12*E*Izz*L*betay*ue[11] + 12*E*Izz*L*betay*ue[5] + 24*E*Izz*betay*ue[1] - 24*E*Izz*betay*ue[7])/(2*L**3)
            finte[2] = betaz*(-A*G*L**3*alphaz**2*betaz*ue[10] - A*G*L**3*alphaz**2*betaz*ue[4] + 2*A*G*L**2*alphaz**2*betaz*ue[2] - 2*A*G*L**2*alphaz**2*betaz*ue[8] - 2*Ay*G*L**2*alphaz*ue[3] + 2*Ay*G*L**2*alphaz*ue[9] - 12*E*Iyy*L*betaz*ue[10] - 12*E*Iyy*L*betaz*ue[4] + 24*E*Iyy*betaz*ue[2] - 24*E*Iyy*betaz*ue[8] + 12*E*Iyz*L*betay*ue[11] + 12*E*Iyz*L*betay*ue[5] + 24*E*Iyz*betay*ue[1] - 24*E*Iyz*betay*ue[7])/(2*L**3)
            finte[3] = G*(Ay*L*alphaz*betaz*ue[10] + Ay*L*alphaz*betaz*ue[4] - 2*Ay*alphaz*betaz*ue[2] + 2*Ay*alphaz*betaz*ue[8] + Az*L*alphay*betay*ue[11] + Az*L*alphay*betay*ue[5] + 2*Az*alphay*betay*ue[1] - 2*Az*alphay*betay*ue[7] + 2*J*ue[3] - 2*J*ue[9])/(2*L)
            finte[4] = betaz*(A*G*L**3*alphaz**2*betaz*ue[10] + A*G*L**3*alphaz**2*betaz*ue[4] - 2*A*G*L**2*alphaz**2*betaz*ue[2] + 2*A*G*L**2*alphaz**2*betaz*ue[8] + 2*Ay*G*L**2*alphaz*ue[3] - 2*Ay*G*L**2*alphaz*ue[9] - 4*Az*E*L*alphaz*ue[0] + 4*Az*E*L*alphaz*ue[6] + 4*Az*E*L*ue[0] - 4*Az*E*L*ue[6] - 4*E*Iyy*L*alphaz**2*betaz*ue[10] + 4*E*Iyy*L*alphaz**2*betaz*ue[4] + 8*E*Iyy*L*alphaz*betaz*ue[10] - 8*E*Iyy*L*alphaz*betaz*ue[4] + 8*E*Iyy*L*betaz*ue[10] + 16*E*Iyy*L*betaz*ue[4] - 24*E*Iyy*betaz*ue[2] + 24*E*Iyy*betaz*ue[8] + 4*E*Iyz*L*alphay*alphaz*betay*ue[11] - 4*E*Iyz*L*alphay*alphaz*betay*ue[5] - 4*E*Iyz*L*alphay*betay*ue[11] + 4*E*Iyz*L*alphay*betay*ue[5] - 4*E*Iyz*L*alphaz*betay*ue[11] + 4*E*Iyz*L*alphaz*betay*ue[5] - 8*E*Iyz*L*betay*ue[11] - 16*E*Iyz*L*betay*ue[5] - 24*E*Iyz*betay*ue[1] + 24*E*Iyz*betay*ue[7])/(4*L**2)
            finte[5] = betay*(A*G*L**3*alphay**2*betay*ue[11] + A*G*L**3*alphay**2*betay*ue[5] + 2*A*G*L**2*alphay**2*betay*ue[1] - 2*A*G*L**2*alphay**2*betay*ue[7] + 4*Ay*E*L*alphay*ue[0] - 4*Ay*E*L*alphay*ue[6] - 4*Ay*E*L*ue[0] + 4*Ay*E*L*ue[6] + 2*Az*G*L**2*alphay*ue[3] - 2*Az*G*L**2*alphay*ue[9] + 4*E*Iyz*L*alphay*alphaz*betaz*ue[10] - 4*E*Iyz*L*alphay*alphaz*betaz*ue[4] - 4*E*Iyz*L*alphay*betaz*ue[10] + 4*E*Iyz*L*alphay*betaz*ue[4] - 4*E*Iyz*L*alphaz*betaz*ue[10] + 4*E*Iyz*L*alphaz*betaz*ue[4] - 8*E*Iyz*L*betaz*ue[10] - 16*E*Iyz*L*betaz*ue[4] + 24*E*Iyz*betaz*ue[2] - 24*E*Iyz*betaz*ue[8] - 4*E*Izz*L*alphay**2*betay*ue[11] + 4*E*Izz*L*alphay**2*betay*ue[5] + 8*E*Izz*L*alphay*betay*ue[11] - 8*E*Izz*L*alphay*betay*ue[5] + 8*E*Izz*L*betay*ue[11] + 16*E*Izz*L*betay*ue[5] + 24*E*Izz*betay*ue[1] - 24*E*Izz*betay*ue[7])/(4*L**2)
            finte[6] = E*(-A*ue[0] + A*ue[6] + Ay*alphay*betay*ue[11] - Ay*alphay*betay*ue[5] - Ay*betay*ue[11] + Ay*betay*ue[5] - Az*alphaz*betaz*ue[10] + Az*alphaz*betaz*ue[4] + Az*betaz*ue[10] - Az*betaz*ue[4])/L
            finte[7] = betay*(-A*G*L**3*alphay**2*betay*ue[11] - A*G*L**3*alphay**2*betay*ue[5] - 2*A*G*L**2*alphay**2*betay*ue[1] + 2*A*G*L**2*alphay**2*betay*ue[7] - 2*Az*G*L**2*alphay*ue[3] + 2*Az*G*L**2*alphay*ue[9] + 12*E*Iyz*L*betaz*ue[10] + 12*E*Iyz*L*betaz*ue[4] - 24*E*Iyz*betaz*ue[2] + 24*E*Iyz*betaz*ue[8] - 12*E*Izz*L*betay*ue[11] - 12*E*Izz*L*betay*ue[5] - 24*E*Izz*betay*ue[1] + 24*E*Izz*betay*ue[7])/(2*L**3)
            finte[8] = betaz*(A*G*L**3*alphaz**2*betaz*ue[10] + A*G*L**3*alphaz**2*betaz*ue[4] - 2*A*G*L**2*alphaz**2*betaz*ue[2] + 2*A*G*L**2*alphaz**2*betaz*ue[8] + 2*Ay*G*L**2*alphaz*ue[3] - 2*Ay*G*L**2*alphaz*ue[9] + 12*E*Iyy*L*betaz*ue[10] + 12*E*Iyy*L*betaz*ue[4] - 24*E*Iyy*betaz*ue[2] + 24*E*Iyy*betaz*ue[8] - 12*E*Iyz*L*betay*ue[11] - 12*E*Iyz*L*betay*ue[5] - 24*E*Iyz*betay*ue[1] + 24*E*Iyz*betay*ue[7])/(2*L**3)
            finte[9] = G*(-Ay*L*alphaz*betaz*ue[10] - Ay*L*alphaz*betaz*ue[4] + 2*Ay*alphaz*betaz*ue[2] - 2*Ay*alphaz*betaz*ue[8] - Az*L*alphay*betay*ue[11] - Az*L*alphay*betay*ue[5] - 2*Az*alphay*betay*ue[1] + 2*Az*alphay*betay*ue[7] - 2*J*ue[3] + 2*J*ue[9])/(2*L)
            finte[10] = betaz*(A*G*L**3*alphaz**2*betaz*ue[10] + A*G*L**3*alphaz**2*betaz*ue[4] - 2*A*G*L**2*alphaz**2*betaz*ue[2] + 2*A*G*L**2*alphaz**2*betaz*ue[8] + 2*Ay*G*L**2*alphaz*ue[3] - 2*Ay*G*L**2*alphaz*ue[9] + 4*Az*E*L*alphaz*ue[0] - 4*Az*E*L*alphaz*ue[6] - 4*Az*E*L*ue[0] + 4*Az*E*L*ue[6] + 4*E*Iyy*L*alphaz**2*betaz*ue[10] - 4*E*Iyy*L*alphaz**2*betaz*ue[4] - 8*E*Iyy*L*alphaz*betaz*ue[10] + 8*E*Iyy*L*alphaz*betaz*ue[4] + 16*E*Iyy*L*betaz*ue[10] + 8*E*Iyy*L*betaz*ue[4] - 24*E*Iyy*betaz*ue[2] + 24*E*Iyy*betaz*ue[8] - 4*E*Iyz*L*alphay*alphaz*betay*ue[11] + 4*E*Iyz*L*alphay*alphaz*betay*ue[5] + 4*E*Iyz*L*alphay*betay*ue[11] - 4*E*Iyz*L*alphay*betay*ue[5] + 4*E*Iyz*L*alphaz*betay*ue[11] - 4*E*Iyz*L*alphaz*betay*ue[5] - 16*E*Iyz*L*betay*ue[11] - 8*E*Iyz*L*betay*ue[5] - 24*E*Iyz*betay*ue[1] + 24*E*Iyz*betay*ue[7])/(4*L**2)
            finte[11] = betay*(A*G*L**3*alphay**2*betay*ue[11] + A*G*L**3*alphay**2*betay*ue[5] + 2*A*G*L**2*alphay**2*betay*ue[1] - 2*A*G*L**2*alphay**2*betay*ue[7] - 4*Ay*E*L*alphay*ue[0] + 4*Ay*E*L*alphay*ue[6] + 4*Ay*E*L*ue[0] - 4*Ay*E*L*ue[6] + 2*Az*G*L**2*alphay*ue[3] - 2*Az*G*L**2*alphay*ue[9] - 4*E*Iyz*L*alphay*alphaz*betaz*ue[10] + 4*E*Iyz*L*alphay*alphaz*betaz*ue[4] + 4*E*Iyz*L*alphay*betaz*ue[10] - 4*E*Iyz*L*alphay*betaz*ue[4] + 4*E*Iyz*L*alphaz*betaz*ue[10] - 4*E*Iyz*L*alphaz*betaz*ue[4] - 16*E*Iyz*L*betaz*ue[10] - 8*E*Iyz*L*betaz*ue[4] + 24*E*Iyz*betaz*ue[2] - 24*E*Iyz*betaz*ue[8] + 4*E*Izz*L*alphay**2*betay*ue[11] - 4*E*Izz*L*alphay**2*betay*ue[5] - 8*E*Izz*L*alphay*betay*ue[11] + 8*E*Izz*L*alphay*betay*ue[5] + 16*E*Izz*L*betay*ue[11] + 8*E*Izz*L*betay*ue[5] + 24*E*Izz*betay*ue[1] - 24*E*Izz*betay*ue[7])/(4*L**2)


    cpdef void update_KC0(BeamC self,
                          long [::1] KC0r,
                          long [::1] KC0c,
                          double [::1] KC0v,
                          BeamProp prop,
                          int update_KC0v_only=0
                          ):
        r"""Update sparse vectors for linear constitutive stiffness matrix KC0

        Parameters
        ----------
        KC0r : np.array
            Array to store row positions of sparse values
        KC0c : np.array
            Array to store column positions of sparse values
        KC0v : np.array
            Array to store sparse values
        prop : :class:`.BeamProp` object
            Beam property object from where the stiffness and mass attributes
            are read from.
        update_KC0v_only : int
            The default ``0`` means that the row and column indices ``KC0r``
            and ``KC0c`` should also be updated. Any other value will only
            update the stiffness matrix values ``KC0v``.

        """
        cdef int c1, c2, k
        cdef double L, A, E, G, Ay, Az, Iyy, Izz, Iyz, J
        cdef double r11, r12, r13, r21, r22, r23, r31, r32, r33
        cdef double alphay, alphaz, betay, betaz
        cdef double KC0e0000, KC0e0004, KC0e0005, KC0e0006, KC0e0010, KC0e0011
        cdef double KC0e0101, KC0e0102, KC0e0103, KC0e0104, KC0e0105, KC0e0107, KC0e0108, KC0e0109, KC0e0110, KC0e0111
        cdef double KC0e0202, KC0e0203, KC0e0204, KC0e0205, KC0e0207, KC0e0208, KC0e0209, KC0e0210, KC0e0211
        cdef double KC0e0303, KC0e0304, KC0e0305, KC0e0307, KC0e0308, KC0e0309, KC0e0310, KC0e0311
        cdef double KC0e0404, KC0e0405, KC0e0406, KC0e0407, KC0e0408, KC0e0409, KC0e0410, KC0e0411
        cdef double KC0e0505, KC0e0506, KC0e0507, KC0e0508, KC0e0509, KC0e0510, KC0e0511
        cdef double KC0e0606, KC0e0610, KC0e0611
        cdef double KC0e0707, KC0e0708, KC0e0709, KC0e0710, KC0e0711
        cdef double KC0e0808, KC0e0809, KC0e0810, KC0e0811
        cdef double KC0e0909, KC0e0910, KC0e0911
        cdef double KC0e1010, KC0e1011, KC0e1111

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

            alphay = 12*E*Izz/(G*A*L**2)
            alphaz = 12*E*Iyy/(G*A*L**2)
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

            KC0e0000 = A*E/L
            KC0e0004 = Az*E*betaz*(1 - alphaz)/L
            KC0e0005 = Ay*E*betay*(alphay - 1)/L
            KC0e0006 = -A*E/L
            KC0e0010 = Az*E*betaz*(alphaz - 1)/L
            KC0e0011 = Ay*E*betay*(1 - alphay)/L
            KC0e0101 = betay**2*(A*G*L**2*alphay**2 + 12*E*Izz)/L**3
            KC0e0102 = 12*E*Iyz*betay*betaz/L**3
            KC0e0103 = Az*G*alphay*betay/L
            KC0e0104 = -6*E*Iyz*betay*betaz/L**2
            KC0e0105 = betay**2*(A*G*L**2*alphay**2 + 12*E*Izz)/(2*L**2)
            KC0e0107 = betay**2*(-A*G*L**2*alphay**2 - 12*E*Izz)/L**3
            KC0e0108 = -12*E*Iyz*betay*betaz/L**3
            KC0e0109 = -Az*G*alphay*betay/L
            KC0e0110 = -6*E*Iyz*betay*betaz/L**2
            KC0e0111 = betay**2*(A*G*L**2*alphay**2 + 12*E*Izz)/(2*L**2)
            KC0e0202 = betaz**2*(A*G*L**2*alphaz**2 + 12*E*Iyy)/L**3
            KC0e0203 = -Ay*G*alphaz*betaz/L
            KC0e0204 = betaz**2*(-A*G*L**2*alphaz**2 - 12*E*Iyy)/(2*L**2)
            KC0e0205 = 6*E*Iyz*betay*betaz/L**2
            KC0e0207 = -12*E*Iyz*betay*betaz/L**3
            KC0e0208 = betaz**2*(-A*G*L**2*alphaz**2 - 12*E*Iyy)/L**3
            KC0e0209 = Ay*G*alphaz*betaz/L
            KC0e0210 = betaz**2*(-A*G*L**2*alphaz**2 - 12*E*Iyy)/(2*L**2)
            KC0e0211 = 6*E*Iyz*betay*betaz/L**2
            KC0e0303 = G*J/L
            KC0e0304 = Ay*G*alphaz*betaz/2
            KC0e0305 = Az*G*alphay*betay/2
            KC0e0307 = -Az*G*alphay*betay/L
            KC0e0308 = Ay*G*alphaz*betaz/L
            KC0e0309 = -G*J/L
            KC0e0310 = Ay*G*alphaz*betaz/2
            KC0e0311 = Az*G*alphay*betay/2
            KC0e0404 = betaz**2*(A*G*L**2*alphaz**2/4 + E*Iyy*alphaz**2 - 2*E*Iyy*alphaz + 4*E*Iyy)/L
            KC0e0405 = E*Iyz*betay*betaz*(-alphay*alphaz + alphay + alphaz - 4)/L
            KC0e0406 = Az*E*betaz*(alphaz - 1)/L
            KC0e0407 = 6*E*Iyz*betay*betaz/L**2
            KC0e0408 = betaz**2*(A*G*L**2*alphaz**2 + 12*E*Iyy)/(2*L**2)
            KC0e0409 = -Ay*G*alphaz*betaz/2
            KC0e0410 = betaz**2*(A*G*L**2*alphaz**2/4 - E*Iyy*alphaz**2 + 2*E*Iyy*alphaz + 2*E*Iyy)/L
            KC0e0411 = E*Iyz*betay*betaz*(alphay*alphaz - alphay - alphaz - 2)/L
            KC0e0505 = betay**2*(A*G*L**2*alphay**2/4 + E*Izz*alphay**2 - 2*E*Izz*alphay + 4*E*Izz)/L
            KC0e0506 = Ay*E*betay*(1 - alphay)/L
            KC0e0507 = betay**2*(-A*G*L**2*alphay**2 - 12*E*Izz)/(2*L**2)
            KC0e0508 = -6*E*Iyz*betay*betaz/L**2
            KC0e0509 = -Az*G*alphay*betay/2
            KC0e0510 = E*Iyz*betay*betaz*(alphay*alphaz - alphay - alphaz - 2)/L
            KC0e0511 = betay**2*(A*G*L**2*alphay**2/4 - E*Izz*alphay**2 + 2*E*Izz*alphay + 2*E*Izz)/L
            KC0e0606 = A*E/L
            KC0e0610 = Az*E*betaz*(1 - alphaz)/L
            KC0e0611 = Ay*E*betay*(alphay - 1)/L
            KC0e0707 = betay**2*(A*G*L**2*alphay**2 + 12*E*Izz)/L**3
            KC0e0708 = 12*E*Iyz*betay*betaz/L**3
            KC0e0709 = Az*G*alphay*betay/L
            KC0e0710 = 6*E*Iyz*betay*betaz/L**2
            KC0e0711 = betay**2*(-A*G*L**2*alphay**2 - 12*E*Izz)/(2*L**2)
            KC0e0808 = betaz**2*(A*G*L**2*alphaz**2 + 12*E*Iyy)/L**3
            KC0e0809 = -Ay*G*alphaz*betaz/L
            KC0e0810 = betaz**2*(A*G*L**2*alphaz**2 + 12*E*Iyy)/(2*L**2)
            KC0e0811 = -6*E*Iyz*betay*betaz/L**2
            KC0e0909 = G*J/L
            KC0e0910 = -Ay*G*alphaz*betaz/2
            KC0e0911 = -Az*G*alphay*betay/2
            KC0e1010 = betaz**2*(A*G*L**2*alphaz**2/4 + E*Iyy*alphaz**2 - 2*E*Iyy*alphaz + 4*E*Iyy)/L
            KC0e1011 = E*Iyz*betay*betaz*(-alphay*alphaz + alphay + alphaz - 4)/L
            KC0e1111 = betay**2*(A*G*L**2*alphay**2/4 + E*Izz*alphay**2 - 2*E*Izz*alphay + 4*E*Izz)/L

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
            KC0v[k] += KC0e0000*r11**2 + r12*(KC0e0101*r12 + KC0e0102*r13) + r13*(KC0e0102*r12 + KC0e0202*r13)
            k += 1
            KC0v[k] += KC0e0000*r11*r21 + r22*(KC0e0101*r12 + KC0e0102*r13) + r23*(KC0e0102*r12 + KC0e0202*r13)
            k += 1
            KC0v[k] += KC0e0000*r11*r31 + r32*(KC0e0101*r12 + KC0e0102*r13) + r33*(KC0e0102*r12 + KC0e0202*r13)
            k += 1
            KC0v[k] += r11*(KC0e0103*r12 + KC0e0203*r13) + r12*(KC0e0004*r11 + KC0e0104*r12 + KC0e0204*r13) + r13*(KC0e0005*r11 + KC0e0105*r12 + KC0e0205*r13)
            k += 1
            KC0v[k] += r21*(KC0e0103*r12 + KC0e0203*r13) + r22*(KC0e0004*r11 + KC0e0104*r12 + KC0e0204*r13) + r23*(KC0e0005*r11 + KC0e0105*r12 + KC0e0205*r13)
            k += 1
            KC0v[k] += r31*(KC0e0103*r12 + KC0e0203*r13) + r32*(KC0e0004*r11 + KC0e0104*r12 + KC0e0204*r13) + r33*(KC0e0005*r11 + KC0e0105*r12 + KC0e0205*r13)
            k += 1
            KC0v[k] += KC0e0006*r11**2 + r12*(KC0e0107*r12 + KC0e0207*r13) + r13*(KC0e0108*r12 + KC0e0208*r13)
            k += 1
            KC0v[k] += KC0e0006*r11*r21 + r22*(KC0e0107*r12 + KC0e0207*r13) + r23*(KC0e0108*r12 + KC0e0208*r13)
            k += 1
            KC0v[k] += KC0e0006*r11*r31 + r32*(KC0e0107*r12 + KC0e0207*r13) + r33*(KC0e0108*r12 + KC0e0208*r13)
            k += 1
            KC0v[k] += r11*(KC0e0109*r12 + KC0e0209*r13) + r12*(KC0e0010*r11 + KC0e0110*r12 + KC0e0210*r13) + r13*(KC0e0011*r11 + KC0e0111*r12 + KC0e0211*r13)
            k += 1
            KC0v[k] += r21*(KC0e0109*r12 + KC0e0209*r13) + r22*(KC0e0010*r11 + KC0e0110*r12 + KC0e0210*r13) + r23*(KC0e0011*r11 + KC0e0111*r12 + KC0e0211*r13)
            k += 1
            KC0v[k] += r31*(KC0e0109*r12 + KC0e0209*r13) + r32*(KC0e0010*r11 + KC0e0110*r12 + KC0e0210*r13) + r33*(KC0e0011*r11 + KC0e0111*r12 + KC0e0211*r13)
            k += 1
            KC0v[k] += KC0e0000*r11*r21 + r12*(KC0e0101*r22 + KC0e0102*r23) + r13*(KC0e0102*r22 + KC0e0202*r23)
            k += 1
            KC0v[k] += KC0e0000*r21**2 + r22*(KC0e0101*r22 + KC0e0102*r23) + r23*(KC0e0102*r22 + KC0e0202*r23)
            k += 1
            KC0v[k] += KC0e0000*r21*r31 + r32*(KC0e0101*r22 + KC0e0102*r23) + r33*(KC0e0102*r22 + KC0e0202*r23)
            k += 1
            KC0v[k] += r11*(KC0e0103*r22 + KC0e0203*r23) + r12*(KC0e0004*r21 + KC0e0104*r22 + KC0e0204*r23) + r13*(KC0e0005*r21 + KC0e0105*r22 + KC0e0205*r23)
            k += 1
            KC0v[k] += r21*(KC0e0103*r22 + KC0e0203*r23) + r22*(KC0e0004*r21 + KC0e0104*r22 + KC0e0204*r23) + r23*(KC0e0005*r21 + KC0e0105*r22 + KC0e0205*r23)
            k += 1
            KC0v[k] += r31*(KC0e0103*r22 + KC0e0203*r23) + r32*(KC0e0004*r21 + KC0e0104*r22 + KC0e0204*r23) + r33*(KC0e0005*r21 + KC0e0105*r22 + KC0e0205*r23)
            k += 1
            KC0v[k] += KC0e0006*r11*r21 + r12*(KC0e0107*r22 + KC0e0207*r23) + r13*(KC0e0108*r22 + KC0e0208*r23)
            k += 1
            KC0v[k] += KC0e0006*r21**2 + r22*(KC0e0107*r22 + KC0e0207*r23) + r23*(KC0e0108*r22 + KC0e0208*r23)
            k += 1
            KC0v[k] += KC0e0006*r21*r31 + r32*(KC0e0107*r22 + KC0e0207*r23) + r33*(KC0e0108*r22 + KC0e0208*r23)
            k += 1
            KC0v[k] += r11*(KC0e0109*r22 + KC0e0209*r23) + r12*(KC0e0010*r21 + KC0e0110*r22 + KC0e0210*r23) + r13*(KC0e0011*r21 + KC0e0111*r22 + KC0e0211*r23)
            k += 1
            KC0v[k] += r21*(KC0e0109*r22 + KC0e0209*r23) + r22*(KC0e0010*r21 + KC0e0110*r22 + KC0e0210*r23) + r23*(KC0e0011*r21 + KC0e0111*r22 + KC0e0211*r23)
            k += 1
            KC0v[k] += r31*(KC0e0109*r22 + KC0e0209*r23) + r32*(KC0e0010*r21 + KC0e0110*r22 + KC0e0210*r23) + r33*(KC0e0011*r21 + KC0e0111*r22 + KC0e0211*r23)
            k += 1
            KC0v[k] += KC0e0000*r11*r31 + r12*(KC0e0101*r32 + KC0e0102*r33) + r13*(KC0e0102*r32 + KC0e0202*r33)
            k += 1
            KC0v[k] += KC0e0000*r21*r31 + r22*(KC0e0101*r32 + KC0e0102*r33) + r23*(KC0e0102*r32 + KC0e0202*r33)
            k += 1
            KC0v[k] += KC0e0000*r31**2 + r32*(KC0e0101*r32 + KC0e0102*r33) + r33*(KC0e0102*r32 + KC0e0202*r33)
            k += 1
            KC0v[k] += r11*(KC0e0103*r32 + KC0e0203*r33) + r12*(KC0e0004*r31 + KC0e0104*r32 + KC0e0204*r33) + r13*(KC0e0005*r31 + KC0e0105*r32 + KC0e0205*r33)
            k += 1
            KC0v[k] += r21*(KC0e0103*r32 + KC0e0203*r33) + r22*(KC0e0004*r31 + KC0e0104*r32 + KC0e0204*r33) + r23*(KC0e0005*r31 + KC0e0105*r32 + KC0e0205*r33)
            k += 1
            KC0v[k] += r31*(KC0e0103*r32 + KC0e0203*r33) + r32*(KC0e0004*r31 + KC0e0104*r32 + KC0e0204*r33) + r33*(KC0e0005*r31 + KC0e0105*r32 + KC0e0205*r33)
            k += 1
            KC0v[k] += KC0e0006*r11*r31 + r12*(KC0e0107*r32 + KC0e0207*r33) + r13*(KC0e0108*r32 + KC0e0208*r33)
            k += 1
            KC0v[k] += KC0e0006*r21*r31 + r22*(KC0e0107*r32 + KC0e0207*r33) + r23*(KC0e0108*r32 + KC0e0208*r33)
            k += 1
            KC0v[k] += KC0e0006*r31**2 + r32*(KC0e0107*r32 + KC0e0207*r33) + r33*(KC0e0108*r32 + KC0e0208*r33)
            k += 1
            KC0v[k] += r11*(KC0e0109*r32 + KC0e0209*r33) + r12*(KC0e0010*r31 + KC0e0110*r32 + KC0e0210*r33) + r13*(KC0e0011*r31 + KC0e0111*r32 + KC0e0211*r33)
            k += 1
            KC0v[k] += r21*(KC0e0109*r32 + KC0e0209*r33) + r22*(KC0e0010*r31 + KC0e0110*r32 + KC0e0210*r33) + r23*(KC0e0011*r31 + KC0e0111*r32 + KC0e0211*r33)
            k += 1
            KC0v[k] += r31*(KC0e0109*r32 + KC0e0209*r33) + r32*(KC0e0010*r31 + KC0e0110*r32 + KC0e0210*r33) + r33*(KC0e0011*r31 + KC0e0111*r32 + KC0e0211*r33)
            k += 1
            KC0v[k] += r11*(KC0e0004*r12 + KC0e0005*r13) + r12*(KC0e0103*r11 + KC0e0104*r12 + KC0e0105*r13) + r13*(KC0e0203*r11 + KC0e0204*r12 + KC0e0205*r13)
            k += 1
            KC0v[k] += r21*(KC0e0004*r12 + KC0e0005*r13) + r22*(KC0e0103*r11 + KC0e0104*r12 + KC0e0105*r13) + r23*(KC0e0203*r11 + KC0e0204*r12 + KC0e0205*r13)
            k += 1
            KC0v[k] += r31*(KC0e0004*r12 + KC0e0005*r13) + r32*(KC0e0103*r11 + KC0e0104*r12 + KC0e0105*r13) + r33*(KC0e0203*r11 + KC0e0204*r12 + KC0e0205*r13)
            k += 1
            KC0v[k] += r11*(KC0e0303*r11 + KC0e0304*r12 + KC0e0305*r13) + r12*(KC0e0304*r11 + KC0e0404*r12 + KC0e0405*r13) + r13*(KC0e0305*r11 + KC0e0405*r12 + KC0e0505*r13)
            k += 1
            KC0v[k] += r21*(KC0e0303*r11 + KC0e0304*r12 + KC0e0305*r13) + r22*(KC0e0304*r11 + KC0e0404*r12 + KC0e0405*r13) + r23*(KC0e0305*r11 + KC0e0405*r12 + KC0e0505*r13)
            k += 1
            KC0v[k] += r31*(KC0e0303*r11 + KC0e0304*r12 + KC0e0305*r13) + r32*(KC0e0304*r11 + KC0e0404*r12 + KC0e0405*r13) + r33*(KC0e0305*r11 + KC0e0405*r12 + KC0e0505*r13)
            k += 1
            KC0v[k] += r11*(KC0e0406*r12 + KC0e0506*r13) + r12*(KC0e0307*r11 + KC0e0407*r12 + KC0e0507*r13) + r13*(KC0e0308*r11 + KC0e0408*r12 + KC0e0508*r13)
            k += 1
            KC0v[k] += r21*(KC0e0406*r12 + KC0e0506*r13) + r22*(KC0e0307*r11 + KC0e0407*r12 + KC0e0507*r13) + r23*(KC0e0308*r11 + KC0e0408*r12 + KC0e0508*r13)
            k += 1
            KC0v[k] += r31*(KC0e0406*r12 + KC0e0506*r13) + r32*(KC0e0307*r11 + KC0e0407*r12 + KC0e0507*r13) + r33*(KC0e0308*r11 + KC0e0408*r12 + KC0e0508*r13)
            k += 1
            KC0v[k] += r11*(KC0e0309*r11 + KC0e0409*r12 + KC0e0509*r13) + r12*(KC0e0310*r11 + KC0e0410*r12 + KC0e0510*r13) + r13*(KC0e0311*r11 + KC0e0411*r12 + KC0e0511*r13)
            k += 1
            KC0v[k] += r21*(KC0e0309*r11 + KC0e0409*r12 + KC0e0509*r13) + r22*(KC0e0310*r11 + KC0e0410*r12 + KC0e0510*r13) + r23*(KC0e0311*r11 + KC0e0411*r12 + KC0e0511*r13)
            k += 1
            KC0v[k] += r31*(KC0e0309*r11 + KC0e0409*r12 + KC0e0509*r13) + r32*(KC0e0310*r11 + KC0e0410*r12 + KC0e0510*r13) + r33*(KC0e0311*r11 + KC0e0411*r12 + KC0e0511*r13)
            k += 1
            KC0v[k] += r11*(KC0e0004*r22 + KC0e0005*r23) + r12*(KC0e0103*r21 + KC0e0104*r22 + KC0e0105*r23) + r13*(KC0e0203*r21 + KC0e0204*r22 + KC0e0205*r23)
            k += 1
            KC0v[k] += r21*(KC0e0004*r22 + KC0e0005*r23) + r22*(KC0e0103*r21 + KC0e0104*r22 + KC0e0105*r23) + r23*(KC0e0203*r21 + KC0e0204*r22 + KC0e0205*r23)
            k += 1
            KC0v[k] += r31*(KC0e0004*r22 + KC0e0005*r23) + r32*(KC0e0103*r21 + KC0e0104*r22 + KC0e0105*r23) + r33*(KC0e0203*r21 + KC0e0204*r22 + KC0e0205*r23)
            k += 1
            KC0v[k] += r11*(KC0e0303*r21 + KC0e0304*r22 + KC0e0305*r23) + r12*(KC0e0304*r21 + KC0e0404*r22 + KC0e0405*r23) + r13*(KC0e0305*r21 + KC0e0405*r22 + KC0e0505*r23)
            k += 1
            KC0v[k] += r21*(KC0e0303*r21 + KC0e0304*r22 + KC0e0305*r23) + r22*(KC0e0304*r21 + KC0e0404*r22 + KC0e0405*r23) + r23*(KC0e0305*r21 + KC0e0405*r22 + KC0e0505*r23)
            k += 1
            KC0v[k] += r31*(KC0e0303*r21 + KC0e0304*r22 + KC0e0305*r23) + r32*(KC0e0304*r21 + KC0e0404*r22 + KC0e0405*r23) + r33*(KC0e0305*r21 + KC0e0405*r22 + KC0e0505*r23)
            k += 1
            KC0v[k] += r11*(KC0e0406*r22 + KC0e0506*r23) + r12*(KC0e0307*r21 + KC0e0407*r22 + KC0e0507*r23) + r13*(KC0e0308*r21 + KC0e0408*r22 + KC0e0508*r23)
            k += 1
            KC0v[k] += r21*(KC0e0406*r22 + KC0e0506*r23) + r22*(KC0e0307*r21 + KC0e0407*r22 + KC0e0507*r23) + r23*(KC0e0308*r21 + KC0e0408*r22 + KC0e0508*r23)
            k += 1
            KC0v[k] += r31*(KC0e0406*r22 + KC0e0506*r23) + r32*(KC0e0307*r21 + KC0e0407*r22 + KC0e0507*r23) + r33*(KC0e0308*r21 + KC0e0408*r22 + KC0e0508*r23)
            k += 1
            KC0v[k] += r11*(KC0e0309*r21 + KC0e0409*r22 + KC0e0509*r23) + r12*(KC0e0310*r21 + KC0e0410*r22 + KC0e0510*r23) + r13*(KC0e0311*r21 + KC0e0411*r22 + KC0e0511*r23)
            k += 1
            KC0v[k] += r21*(KC0e0309*r21 + KC0e0409*r22 + KC0e0509*r23) + r22*(KC0e0310*r21 + KC0e0410*r22 + KC0e0510*r23) + r23*(KC0e0311*r21 + KC0e0411*r22 + KC0e0511*r23)
            k += 1
            KC0v[k] += r31*(KC0e0309*r21 + KC0e0409*r22 + KC0e0509*r23) + r32*(KC0e0310*r21 + KC0e0410*r22 + KC0e0510*r23) + r33*(KC0e0311*r21 + KC0e0411*r22 + KC0e0511*r23)
            k += 1
            KC0v[k] += r11*(KC0e0004*r32 + KC0e0005*r33) + r12*(KC0e0103*r31 + KC0e0104*r32 + KC0e0105*r33) + r13*(KC0e0203*r31 + KC0e0204*r32 + KC0e0205*r33)
            k += 1
            KC0v[k] += r21*(KC0e0004*r32 + KC0e0005*r33) + r22*(KC0e0103*r31 + KC0e0104*r32 + KC0e0105*r33) + r23*(KC0e0203*r31 + KC0e0204*r32 + KC0e0205*r33)
            k += 1
            KC0v[k] += r31*(KC0e0004*r32 + KC0e0005*r33) + r32*(KC0e0103*r31 + KC0e0104*r32 + KC0e0105*r33) + r33*(KC0e0203*r31 + KC0e0204*r32 + KC0e0205*r33)
            k += 1
            KC0v[k] += r11*(KC0e0303*r31 + KC0e0304*r32 + KC0e0305*r33) + r12*(KC0e0304*r31 + KC0e0404*r32 + KC0e0405*r33) + r13*(KC0e0305*r31 + KC0e0405*r32 + KC0e0505*r33)
            k += 1
            KC0v[k] += r21*(KC0e0303*r31 + KC0e0304*r32 + KC0e0305*r33) + r22*(KC0e0304*r31 + KC0e0404*r32 + KC0e0405*r33) + r23*(KC0e0305*r31 + KC0e0405*r32 + KC0e0505*r33)
            k += 1
            KC0v[k] += r31*(KC0e0303*r31 + KC0e0304*r32 + KC0e0305*r33) + r32*(KC0e0304*r31 + KC0e0404*r32 + KC0e0405*r33) + r33*(KC0e0305*r31 + KC0e0405*r32 + KC0e0505*r33)
            k += 1
            KC0v[k] += r11*(KC0e0406*r32 + KC0e0506*r33) + r12*(KC0e0307*r31 + KC0e0407*r32 + KC0e0507*r33) + r13*(KC0e0308*r31 + KC0e0408*r32 + KC0e0508*r33)
            k += 1
            KC0v[k] += r21*(KC0e0406*r32 + KC0e0506*r33) + r22*(KC0e0307*r31 + KC0e0407*r32 + KC0e0507*r33) + r23*(KC0e0308*r31 + KC0e0408*r32 + KC0e0508*r33)
            k += 1
            KC0v[k] += r31*(KC0e0406*r32 + KC0e0506*r33) + r32*(KC0e0307*r31 + KC0e0407*r32 + KC0e0507*r33) + r33*(KC0e0308*r31 + KC0e0408*r32 + KC0e0508*r33)
            k += 1
            KC0v[k] += r11*(KC0e0309*r31 + KC0e0409*r32 + KC0e0509*r33) + r12*(KC0e0310*r31 + KC0e0410*r32 + KC0e0510*r33) + r13*(KC0e0311*r31 + KC0e0411*r32 + KC0e0511*r33)
            k += 1
            KC0v[k] += r21*(KC0e0309*r31 + KC0e0409*r32 + KC0e0509*r33) + r22*(KC0e0310*r31 + KC0e0410*r32 + KC0e0510*r33) + r23*(KC0e0311*r31 + KC0e0411*r32 + KC0e0511*r33)
            k += 1
            KC0v[k] += r31*(KC0e0309*r31 + KC0e0409*r32 + KC0e0509*r33) + r32*(KC0e0310*r31 + KC0e0410*r32 + KC0e0510*r33) + r33*(KC0e0311*r31 + KC0e0411*r32 + KC0e0511*r33)
            k += 1
            KC0v[k] += KC0e0006*r11**2 + r12*(KC0e0107*r12 + KC0e0108*r13) + r13*(KC0e0207*r12 + KC0e0208*r13)
            k += 1
            KC0v[k] += KC0e0006*r11*r21 + r22*(KC0e0107*r12 + KC0e0108*r13) + r23*(KC0e0207*r12 + KC0e0208*r13)
            k += 1
            KC0v[k] += KC0e0006*r11*r31 + r32*(KC0e0107*r12 + KC0e0108*r13) + r33*(KC0e0207*r12 + KC0e0208*r13)
            k += 1
            KC0v[k] += r11*(KC0e0307*r12 + KC0e0308*r13) + r12*(KC0e0406*r11 + KC0e0407*r12 + KC0e0408*r13) + r13*(KC0e0506*r11 + KC0e0507*r12 + KC0e0508*r13)
            k += 1
            KC0v[k] += r21*(KC0e0307*r12 + KC0e0308*r13) + r22*(KC0e0406*r11 + KC0e0407*r12 + KC0e0408*r13) + r23*(KC0e0506*r11 + KC0e0507*r12 + KC0e0508*r13)
            k += 1
            KC0v[k] += r31*(KC0e0307*r12 + KC0e0308*r13) + r32*(KC0e0406*r11 + KC0e0407*r12 + KC0e0408*r13) + r33*(KC0e0506*r11 + KC0e0507*r12 + KC0e0508*r13)
            k += 1
            KC0v[k] += KC0e0606*r11**2 + r12*(KC0e0707*r12 + KC0e0708*r13) + r13*(KC0e0708*r12 + KC0e0808*r13)
            k += 1
            KC0v[k] += KC0e0606*r11*r21 + r22*(KC0e0707*r12 + KC0e0708*r13) + r23*(KC0e0708*r12 + KC0e0808*r13)
            k += 1
            KC0v[k] += KC0e0606*r11*r31 + r32*(KC0e0707*r12 + KC0e0708*r13) + r33*(KC0e0708*r12 + KC0e0808*r13)
            k += 1
            KC0v[k] += r11*(KC0e0709*r12 + KC0e0809*r13) + r12*(KC0e0610*r11 + KC0e0710*r12 + KC0e0810*r13) + r13*(KC0e0611*r11 + KC0e0711*r12 + KC0e0811*r13)
            k += 1
            KC0v[k] += r21*(KC0e0709*r12 + KC0e0809*r13) + r22*(KC0e0610*r11 + KC0e0710*r12 + KC0e0810*r13) + r23*(KC0e0611*r11 + KC0e0711*r12 + KC0e0811*r13)
            k += 1
            KC0v[k] += r31*(KC0e0709*r12 + KC0e0809*r13) + r32*(KC0e0610*r11 + KC0e0710*r12 + KC0e0810*r13) + r33*(KC0e0611*r11 + KC0e0711*r12 + KC0e0811*r13)
            k += 1
            KC0v[k] += KC0e0006*r11*r21 + r12*(KC0e0107*r22 + KC0e0108*r23) + r13*(KC0e0207*r22 + KC0e0208*r23)
            k += 1
            KC0v[k] += KC0e0006*r21**2 + r22*(KC0e0107*r22 + KC0e0108*r23) + r23*(KC0e0207*r22 + KC0e0208*r23)
            k += 1
            KC0v[k] += KC0e0006*r21*r31 + r32*(KC0e0107*r22 + KC0e0108*r23) + r33*(KC0e0207*r22 + KC0e0208*r23)
            k += 1
            KC0v[k] += r11*(KC0e0307*r22 + KC0e0308*r23) + r12*(KC0e0406*r21 + KC0e0407*r22 + KC0e0408*r23) + r13*(KC0e0506*r21 + KC0e0507*r22 + KC0e0508*r23)
            k += 1
            KC0v[k] += r21*(KC0e0307*r22 + KC0e0308*r23) + r22*(KC0e0406*r21 + KC0e0407*r22 + KC0e0408*r23) + r23*(KC0e0506*r21 + KC0e0507*r22 + KC0e0508*r23)
            k += 1
            KC0v[k] += r31*(KC0e0307*r22 + KC0e0308*r23) + r32*(KC0e0406*r21 + KC0e0407*r22 + KC0e0408*r23) + r33*(KC0e0506*r21 + KC0e0507*r22 + KC0e0508*r23)
            k += 1
            KC0v[k] += KC0e0606*r11*r21 + r12*(KC0e0707*r22 + KC0e0708*r23) + r13*(KC0e0708*r22 + KC0e0808*r23)
            k += 1
            KC0v[k] += KC0e0606*r21**2 + r22*(KC0e0707*r22 + KC0e0708*r23) + r23*(KC0e0708*r22 + KC0e0808*r23)
            k += 1
            KC0v[k] += KC0e0606*r21*r31 + r32*(KC0e0707*r22 + KC0e0708*r23) + r33*(KC0e0708*r22 + KC0e0808*r23)
            k += 1
            KC0v[k] += r11*(KC0e0709*r22 + KC0e0809*r23) + r12*(KC0e0610*r21 + KC0e0710*r22 + KC0e0810*r23) + r13*(KC0e0611*r21 + KC0e0711*r22 + KC0e0811*r23)
            k += 1
            KC0v[k] += r21*(KC0e0709*r22 + KC0e0809*r23) + r22*(KC0e0610*r21 + KC0e0710*r22 + KC0e0810*r23) + r23*(KC0e0611*r21 + KC0e0711*r22 + KC0e0811*r23)
            k += 1
            KC0v[k] += r31*(KC0e0709*r22 + KC0e0809*r23) + r32*(KC0e0610*r21 + KC0e0710*r22 + KC0e0810*r23) + r33*(KC0e0611*r21 + KC0e0711*r22 + KC0e0811*r23)
            k += 1
            KC0v[k] += KC0e0006*r11*r31 + r12*(KC0e0107*r32 + KC0e0108*r33) + r13*(KC0e0207*r32 + KC0e0208*r33)
            k += 1
            KC0v[k] += KC0e0006*r21*r31 + r22*(KC0e0107*r32 + KC0e0108*r33) + r23*(KC0e0207*r32 + KC0e0208*r33)
            k += 1
            KC0v[k] += KC0e0006*r31**2 + r32*(KC0e0107*r32 + KC0e0108*r33) + r33*(KC0e0207*r32 + KC0e0208*r33)
            k += 1
            KC0v[k] += r11*(KC0e0307*r32 + KC0e0308*r33) + r12*(KC0e0406*r31 + KC0e0407*r32 + KC0e0408*r33) + r13*(KC0e0506*r31 + KC0e0507*r32 + KC0e0508*r33)
            k += 1
            KC0v[k] += r21*(KC0e0307*r32 + KC0e0308*r33) + r22*(KC0e0406*r31 + KC0e0407*r32 + KC0e0408*r33) + r23*(KC0e0506*r31 + KC0e0507*r32 + KC0e0508*r33)
            k += 1
            KC0v[k] += r31*(KC0e0307*r32 + KC0e0308*r33) + r32*(KC0e0406*r31 + KC0e0407*r32 + KC0e0408*r33) + r33*(KC0e0506*r31 + KC0e0507*r32 + KC0e0508*r33)
            k += 1
            KC0v[k] += KC0e0606*r11*r31 + r12*(KC0e0707*r32 + KC0e0708*r33) + r13*(KC0e0708*r32 + KC0e0808*r33)
            k += 1
            KC0v[k] += KC0e0606*r21*r31 + r22*(KC0e0707*r32 + KC0e0708*r33) + r23*(KC0e0708*r32 + KC0e0808*r33)
            k += 1
            KC0v[k] += KC0e0606*r31**2 + r32*(KC0e0707*r32 + KC0e0708*r33) + r33*(KC0e0708*r32 + KC0e0808*r33)
            k += 1
            KC0v[k] += r11*(KC0e0709*r32 + KC0e0809*r33) + r12*(KC0e0610*r31 + KC0e0710*r32 + KC0e0810*r33) + r13*(KC0e0611*r31 + KC0e0711*r32 + KC0e0811*r33)
            k += 1
            KC0v[k] += r21*(KC0e0709*r32 + KC0e0809*r33) + r22*(KC0e0610*r31 + KC0e0710*r32 + KC0e0810*r33) + r23*(KC0e0611*r31 + KC0e0711*r32 + KC0e0811*r33)
            k += 1
            KC0v[k] += r31*(KC0e0709*r32 + KC0e0809*r33) + r32*(KC0e0610*r31 + KC0e0710*r32 + KC0e0810*r33) + r33*(KC0e0611*r31 + KC0e0711*r32 + KC0e0811*r33)
            k += 1
            KC0v[k] += r11*(KC0e0010*r12 + KC0e0011*r13) + r12*(KC0e0109*r11 + KC0e0110*r12 + KC0e0111*r13) + r13*(KC0e0209*r11 + KC0e0210*r12 + KC0e0211*r13)
            k += 1
            KC0v[k] += r21*(KC0e0010*r12 + KC0e0011*r13) + r22*(KC0e0109*r11 + KC0e0110*r12 + KC0e0111*r13) + r23*(KC0e0209*r11 + KC0e0210*r12 + KC0e0211*r13)
            k += 1
            KC0v[k] += r31*(KC0e0010*r12 + KC0e0011*r13) + r32*(KC0e0109*r11 + KC0e0110*r12 + KC0e0111*r13) + r33*(KC0e0209*r11 + KC0e0210*r12 + KC0e0211*r13)
            k += 1
            KC0v[k] += r11*(KC0e0309*r11 + KC0e0310*r12 + KC0e0311*r13) + r12*(KC0e0409*r11 + KC0e0410*r12 + KC0e0411*r13) + r13*(KC0e0509*r11 + KC0e0510*r12 + KC0e0511*r13)
            k += 1
            KC0v[k] += r21*(KC0e0309*r11 + KC0e0310*r12 + KC0e0311*r13) + r22*(KC0e0409*r11 + KC0e0410*r12 + KC0e0411*r13) + r23*(KC0e0509*r11 + KC0e0510*r12 + KC0e0511*r13)
            k += 1
            KC0v[k] += r31*(KC0e0309*r11 + KC0e0310*r12 + KC0e0311*r13) + r32*(KC0e0409*r11 + KC0e0410*r12 + KC0e0411*r13) + r33*(KC0e0509*r11 + KC0e0510*r12 + KC0e0511*r13)
            k += 1
            KC0v[k] += r11*(KC0e0610*r12 + KC0e0611*r13) + r12*(KC0e0709*r11 + KC0e0710*r12 + KC0e0711*r13) + r13*(KC0e0809*r11 + KC0e0810*r12 + KC0e0811*r13)
            k += 1
            KC0v[k] += r21*(KC0e0610*r12 + KC0e0611*r13) + r22*(KC0e0709*r11 + KC0e0710*r12 + KC0e0711*r13) + r23*(KC0e0809*r11 + KC0e0810*r12 + KC0e0811*r13)
            k += 1
            KC0v[k] += r31*(KC0e0610*r12 + KC0e0611*r13) + r32*(KC0e0709*r11 + KC0e0710*r12 + KC0e0711*r13) + r33*(KC0e0809*r11 + KC0e0810*r12 + KC0e0811*r13)
            k += 1
            KC0v[k] += r11*(KC0e0909*r11 + KC0e0910*r12 + KC0e0911*r13) + r12*(KC0e0910*r11 + KC0e1010*r12 + KC0e1011*r13) + r13*(KC0e0911*r11 + KC0e1011*r12 + KC0e1111*r13)
            k += 1
            KC0v[k] += r21*(KC0e0909*r11 + KC0e0910*r12 + KC0e0911*r13) + r22*(KC0e0910*r11 + KC0e1010*r12 + KC0e1011*r13) + r23*(KC0e0911*r11 + KC0e1011*r12 + KC0e1111*r13)
            k += 1
            KC0v[k] += r31*(KC0e0909*r11 + KC0e0910*r12 + KC0e0911*r13) + r32*(KC0e0910*r11 + KC0e1010*r12 + KC0e1011*r13) + r33*(KC0e0911*r11 + KC0e1011*r12 + KC0e1111*r13)
            k += 1
            KC0v[k] += r11*(KC0e0010*r22 + KC0e0011*r23) + r12*(KC0e0109*r21 + KC0e0110*r22 + KC0e0111*r23) + r13*(KC0e0209*r21 + KC0e0210*r22 + KC0e0211*r23)
            k += 1
            KC0v[k] += r21*(KC0e0010*r22 + KC0e0011*r23) + r22*(KC0e0109*r21 + KC0e0110*r22 + KC0e0111*r23) + r23*(KC0e0209*r21 + KC0e0210*r22 + KC0e0211*r23)
            k += 1
            KC0v[k] += r31*(KC0e0010*r22 + KC0e0011*r23) + r32*(KC0e0109*r21 + KC0e0110*r22 + KC0e0111*r23) + r33*(KC0e0209*r21 + KC0e0210*r22 + KC0e0211*r23)
            k += 1
            KC0v[k] += r11*(KC0e0309*r21 + KC0e0310*r22 + KC0e0311*r23) + r12*(KC0e0409*r21 + KC0e0410*r22 + KC0e0411*r23) + r13*(KC0e0509*r21 + KC0e0510*r22 + KC0e0511*r23)
            k += 1
            KC0v[k] += r21*(KC0e0309*r21 + KC0e0310*r22 + KC0e0311*r23) + r22*(KC0e0409*r21 + KC0e0410*r22 + KC0e0411*r23) + r23*(KC0e0509*r21 + KC0e0510*r22 + KC0e0511*r23)
            k += 1
            KC0v[k] += r31*(KC0e0309*r21 + KC0e0310*r22 + KC0e0311*r23) + r32*(KC0e0409*r21 + KC0e0410*r22 + KC0e0411*r23) + r33*(KC0e0509*r21 + KC0e0510*r22 + KC0e0511*r23)
            k += 1
            KC0v[k] += r11*(KC0e0610*r22 + KC0e0611*r23) + r12*(KC0e0709*r21 + KC0e0710*r22 + KC0e0711*r23) + r13*(KC0e0809*r21 + KC0e0810*r22 + KC0e0811*r23)
            k += 1
            KC0v[k] += r21*(KC0e0610*r22 + KC0e0611*r23) + r22*(KC0e0709*r21 + KC0e0710*r22 + KC0e0711*r23) + r23*(KC0e0809*r21 + KC0e0810*r22 + KC0e0811*r23)
            k += 1
            KC0v[k] += r31*(KC0e0610*r22 + KC0e0611*r23) + r32*(KC0e0709*r21 + KC0e0710*r22 + KC0e0711*r23) + r33*(KC0e0809*r21 + KC0e0810*r22 + KC0e0811*r23)
            k += 1
            KC0v[k] += r11*(KC0e0909*r21 + KC0e0910*r22 + KC0e0911*r23) + r12*(KC0e0910*r21 + KC0e1010*r22 + KC0e1011*r23) + r13*(KC0e0911*r21 + KC0e1011*r22 + KC0e1111*r23)
            k += 1
            KC0v[k] += r21*(KC0e0909*r21 + KC0e0910*r22 + KC0e0911*r23) + r22*(KC0e0910*r21 + KC0e1010*r22 + KC0e1011*r23) + r23*(KC0e0911*r21 + KC0e1011*r22 + KC0e1111*r23)
            k += 1
            KC0v[k] += r31*(KC0e0909*r21 + KC0e0910*r22 + KC0e0911*r23) + r32*(KC0e0910*r21 + KC0e1010*r22 + KC0e1011*r23) + r33*(KC0e0911*r21 + KC0e1011*r22 + KC0e1111*r23)
            k += 1
            KC0v[k] += r11*(KC0e0010*r32 + KC0e0011*r33) + r12*(KC0e0109*r31 + KC0e0110*r32 + KC0e0111*r33) + r13*(KC0e0209*r31 + KC0e0210*r32 + KC0e0211*r33)
            k += 1
            KC0v[k] += r21*(KC0e0010*r32 + KC0e0011*r33) + r22*(KC0e0109*r31 + KC0e0110*r32 + KC0e0111*r33) + r23*(KC0e0209*r31 + KC0e0210*r32 + KC0e0211*r33)
            k += 1
            KC0v[k] += r31*(KC0e0010*r32 + KC0e0011*r33) + r32*(KC0e0109*r31 + KC0e0110*r32 + KC0e0111*r33) + r33*(KC0e0209*r31 + KC0e0210*r32 + KC0e0211*r33)
            k += 1
            KC0v[k] += r11*(KC0e0309*r31 + KC0e0310*r32 + KC0e0311*r33) + r12*(KC0e0409*r31 + KC0e0410*r32 + KC0e0411*r33) + r13*(KC0e0509*r31 + KC0e0510*r32 + KC0e0511*r33)
            k += 1
            KC0v[k] += r21*(KC0e0309*r31 + KC0e0310*r32 + KC0e0311*r33) + r22*(KC0e0409*r31 + KC0e0410*r32 + KC0e0411*r33) + r23*(KC0e0509*r31 + KC0e0510*r32 + KC0e0511*r33)
            k += 1
            KC0v[k] += r31*(KC0e0309*r31 + KC0e0310*r32 + KC0e0311*r33) + r32*(KC0e0409*r31 + KC0e0410*r32 + KC0e0411*r33) + r33*(KC0e0509*r31 + KC0e0510*r32 + KC0e0511*r33)
            k += 1
            KC0v[k] += r11*(KC0e0610*r32 + KC0e0611*r33) + r12*(KC0e0709*r31 + KC0e0710*r32 + KC0e0711*r33) + r13*(KC0e0809*r31 + KC0e0810*r32 + KC0e0811*r33)
            k += 1
            KC0v[k] += r21*(KC0e0610*r32 + KC0e0611*r33) + r22*(KC0e0709*r31 + KC0e0710*r32 + KC0e0711*r33) + r23*(KC0e0809*r31 + KC0e0810*r32 + KC0e0811*r33)
            k += 1
            KC0v[k] += r31*(KC0e0610*r32 + KC0e0611*r33) + r32*(KC0e0709*r31 + KC0e0710*r32 + KC0e0711*r33) + r33*(KC0e0809*r31 + KC0e0810*r32 + KC0e0811*r33)
            k += 1
            KC0v[k] += r11*(KC0e0909*r31 + KC0e0910*r32 + KC0e0911*r33) + r12*(KC0e0910*r31 + KC0e1010*r32 + KC0e1011*r33) + r13*(KC0e0911*r31 + KC0e1011*r32 + KC0e1111*r33)
            k += 1
            KC0v[k] += r21*(KC0e0909*r31 + KC0e0910*r32 + KC0e0911*r33) + r22*(KC0e0910*r31 + KC0e1010*r32 + KC0e1011*r33) + r23*(KC0e0911*r31 + KC0e1011*r32 + KC0e1111*r33)
            k += 1
            KC0v[k] += r31*(KC0e0909*r31 + KC0e0910*r32 + KC0e0911*r33) + r32*(KC0e0910*r31 + KC0e1010*r32 + KC0e1011*r33) + r33*(KC0e0911*r31 + KC0e1011*r32 + KC0e1111*r33)


    cpdef void update_fint(BeamC self,
                           double [::1] fint,
                           BeamProp prop):
        r"""Update the internal force vector

        Parameters
        ----------
        fint : np.array
            Array that is updated in place with the internal forces. The
            internal forces stored in ``fint`` are calculated in global
            coordinates. Method :meth:`.update_probe_finte` is called to update
            the parameter ``finte`` of the :class:`.BeamCProbe` with the
            internal forces in local coordinates.
        prop : :class:`.BeamProp` object
            Beam property object from where the stiffness and mass attributes
            are read from.

        """
        cdef double *finte

        self.update_probe_finte(prop)
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


    cpdef void update_KG(BeamC self,
                         long [::1] KGr,
                         long [::1] KGc,
                         double [::1] KGv,
                         BeamProp prop,
                         int update_KGv_only=0
                         ):
        r"""Update sparse vectors for geometric stiffness matrix KG

        Parameters
        ----------
        KGr : np.array
           Array to store row positions of sparse values
        KGc : np.array
           Array to store column positions of sparse values
        KGv : np.array
            Array to store sparse values
        prop : :class:`.BeamProp` object
            Beam property object from where the stiffness and mass attributes
            are read from.
        update_KGv_only : int
            The default `0` means that only `KGv` is updated. Any other value will
            lead to `KGr` and `KGc` also being updated.

        """
        cdef double *ue
        cdef int c1, c2, k
        cdef double L, A, E, G, Iyy, Izz, Iyz, J, N
        cdef double r11, r12, r13, r21, r22, r23, r31, r32, r33
        cdef double alphay, alphaz, betay, betaz

        with nogil:
            L = self.length
            A = prop.A
            E = prop.E
            G = prop.G
            Iyy = prop.Iyy
            Izz = prop.Izz
            Iyz = prop.Iyz
            J = prop.J

            alphay = 12*E*Izz/(G*A*L**2)
            alphaz = 12*E*Iyy/(G*A*L**2)
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

            N = A*E*(-ue[0] + ue[6])/L

            k = self.init_k_KG
            KGv[k] += r12*(N*betay**2*r12*(5*alphay**2 - 10*alphay + 6)/(5*L) + N*betay*betaz*r13*(5*alphay*alphaz - 5*alphay - 5*alphaz + 6)/(5*L)) + r13*(N*betay*betaz*r12*(5*alphay*alphaz - 5*alphay - 5*alphaz + 6)/(5*L) + N*betaz**2*r13*(5*alphaz**2 - 10*alphaz + 6)/(5*L))
            k += 1
            KGv[k] += r22*(N*betay**2*r12*(5*alphay**2 - 10*alphay + 6)/(5*L) + N*betay*betaz*r13*(5*alphay*alphaz - 5*alphay - 5*alphaz + 6)/(5*L)) + r23*(N*betay*betaz*r12*(5*alphay*alphaz - 5*alphay - 5*alphaz + 6)/(5*L) + N*betaz**2*r13*(5*alphaz**2 - 10*alphaz + 6)/(5*L))
            k += 1
            KGv[k] += r32*(N*betay**2*r12*(5*alphay**2 - 10*alphay + 6)/(5*L) + N*betay*betaz*r13*(5*alphay*alphaz - 5*alphay - 5*alphaz + 6)/(5*L)) + r33*(N*betay*betaz*r12*(5*alphay*alphaz - 5*alphay - 5*alphaz + 6)/(5*L) + N*betaz**2*r13*(5*alphaz**2 - 10*alphaz + 6)/(5*L))
            k += 1
            KGv[k] += r12*(-N*betay*betaz*r12/10 - N*betaz**2*r13/10) + r13*(N*betay**2*r12/10 + N*betay*betaz*r13/10)
            k += 1
            KGv[k] += r22*(-N*betay*betaz*r12/10 - N*betaz**2*r13/10) + r23*(N*betay**2*r12/10 + N*betay*betaz*r13/10)
            k += 1
            KGv[k] += r32*(-N*betay*betaz*r12/10 - N*betaz**2*r13/10) + r33*(N*betay**2*r12/10 + N*betay*betaz*r13/10)
            k += 1
            KGv[k] += r12*(N*betay**2*r12*(-5*alphay**2 + 10*alphay - 6)/(5*L) + N*betay*betaz*r13*(-5*alphay*alphaz + 5*alphay + 5*alphaz - 6)/(5*L)) + r13*(N*betay*betaz*r12*(-5*alphay*alphaz + 5*alphay + 5*alphaz - 6)/(5*L) + N*betaz**2*r13*(-5*alphaz**2 + 10*alphaz - 6)/(5*L))
            k += 1
            KGv[k] += r22*(N*betay**2*r12*(-5*alphay**2 + 10*alphay - 6)/(5*L) + N*betay*betaz*r13*(-5*alphay*alphaz + 5*alphay + 5*alphaz - 6)/(5*L)) + r23*(N*betay*betaz*r12*(-5*alphay*alphaz + 5*alphay + 5*alphaz - 6)/(5*L) + N*betaz**2*r13*(-5*alphaz**2 + 10*alphaz - 6)/(5*L))
            k += 1
            KGv[k] += r32*(N*betay**2*r12*(-5*alphay**2 + 10*alphay - 6)/(5*L) + N*betay*betaz*r13*(-5*alphay*alphaz + 5*alphay + 5*alphaz - 6)/(5*L)) + r33*(N*betay*betaz*r12*(-5*alphay*alphaz + 5*alphay + 5*alphaz - 6)/(5*L) + N*betaz**2*r13*(-5*alphaz**2 + 10*alphaz - 6)/(5*L))
            k += 1
            KGv[k] += r12*(-N*betay*betaz*r12/10 - N*betaz**2*r13/10) + r13*(N*betay**2*r12/10 + N*betay*betaz*r13/10)
            k += 1
            KGv[k] += r22*(-N*betay*betaz*r12/10 - N*betaz**2*r13/10) + r23*(N*betay**2*r12/10 + N*betay*betaz*r13/10)
            k += 1
            KGv[k] += r32*(-N*betay*betaz*r12/10 - N*betaz**2*r13/10) + r33*(N*betay**2*r12/10 + N*betay*betaz*r13/10)
            k += 1
            KGv[k] += r12*(N*betay**2*r22*(5*alphay**2 - 10*alphay + 6)/(5*L) + N*betay*betaz*r23*(5*alphay*alphaz - 5*alphay - 5*alphaz + 6)/(5*L)) + r13*(N*betay*betaz*r22*(5*alphay*alphaz - 5*alphay - 5*alphaz + 6)/(5*L) + N*betaz**2*r23*(5*alphaz**2 - 10*alphaz + 6)/(5*L))
            k += 1
            KGv[k] += r22*(N*betay**2*r22*(5*alphay**2 - 10*alphay + 6)/(5*L) + N*betay*betaz*r23*(5*alphay*alphaz - 5*alphay - 5*alphaz + 6)/(5*L)) + r23*(N*betay*betaz*r22*(5*alphay*alphaz - 5*alphay - 5*alphaz + 6)/(5*L) + N*betaz**2*r23*(5*alphaz**2 - 10*alphaz + 6)/(5*L))
            k += 1
            KGv[k] += r32*(N*betay**2*r22*(5*alphay**2 - 10*alphay + 6)/(5*L) + N*betay*betaz*r23*(5*alphay*alphaz - 5*alphay - 5*alphaz + 6)/(5*L)) + r33*(N*betay*betaz*r22*(5*alphay*alphaz - 5*alphay - 5*alphaz + 6)/(5*L) + N*betaz**2*r23*(5*alphaz**2 - 10*alphaz + 6)/(5*L))
            k += 1
            KGv[k] += r12*(-N*betay*betaz*r22/10 - N*betaz**2*r23/10) + r13*(N*betay**2*r22/10 + N*betay*betaz*r23/10)
            k += 1
            KGv[k] += r22*(-N*betay*betaz*r22/10 - N*betaz**2*r23/10) + r23*(N*betay**2*r22/10 + N*betay*betaz*r23/10)
            k += 1
            KGv[k] += r32*(-N*betay*betaz*r22/10 - N*betaz**2*r23/10) + r33*(N*betay**2*r22/10 + N*betay*betaz*r23/10)
            k += 1
            KGv[k] += r12*(N*betay**2*r22*(-5*alphay**2 + 10*alphay - 6)/(5*L) + N*betay*betaz*r23*(-5*alphay*alphaz + 5*alphay + 5*alphaz - 6)/(5*L)) + r13*(N*betay*betaz*r22*(-5*alphay*alphaz + 5*alphay + 5*alphaz - 6)/(5*L) + N*betaz**2*r23*(-5*alphaz**2 + 10*alphaz - 6)/(5*L))
            k += 1
            KGv[k] += r22*(N*betay**2*r22*(-5*alphay**2 + 10*alphay - 6)/(5*L) + N*betay*betaz*r23*(-5*alphay*alphaz + 5*alphay + 5*alphaz - 6)/(5*L)) + r23*(N*betay*betaz*r22*(-5*alphay*alphaz + 5*alphay + 5*alphaz - 6)/(5*L) + N*betaz**2*r23*(-5*alphaz**2 + 10*alphaz - 6)/(5*L))
            k += 1
            KGv[k] += r32*(N*betay**2*r22*(-5*alphay**2 + 10*alphay - 6)/(5*L) + N*betay*betaz*r23*(-5*alphay*alphaz + 5*alphay + 5*alphaz - 6)/(5*L)) + r33*(N*betay*betaz*r22*(-5*alphay*alphaz + 5*alphay + 5*alphaz - 6)/(5*L) + N*betaz**2*r23*(-5*alphaz**2 + 10*alphaz - 6)/(5*L))
            k += 1
            KGv[k] += r12*(-N*betay*betaz*r22/10 - N*betaz**2*r23/10) + r13*(N*betay**2*r22/10 + N*betay*betaz*r23/10)
            k += 1
            KGv[k] += r22*(-N*betay*betaz*r22/10 - N*betaz**2*r23/10) + r23*(N*betay**2*r22/10 + N*betay*betaz*r23/10)
            k += 1
            KGv[k] += r32*(-N*betay*betaz*r22/10 - N*betaz**2*r23/10) + r33*(N*betay**2*r22/10 + N*betay*betaz*r23/10)
            k += 1
            KGv[k] += r12*(N*betay**2*r32*(5*alphay**2 - 10*alphay + 6)/(5*L) + N*betay*betaz*r33*(5*alphay*alphaz - 5*alphay - 5*alphaz + 6)/(5*L)) + r13*(N*betay*betaz*r32*(5*alphay*alphaz - 5*alphay - 5*alphaz + 6)/(5*L) + N*betaz**2*r33*(5*alphaz**2 - 10*alphaz + 6)/(5*L))
            k += 1
            KGv[k] += r22*(N*betay**2*r32*(5*alphay**2 - 10*alphay + 6)/(5*L) + N*betay*betaz*r33*(5*alphay*alphaz - 5*alphay - 5*alphaz + 6)/(5*L)) + r23*(N*betay*betaz*r32*(5*alphay*alphaz - 5*alphay - 5*alphaz + 6)/(5*L) + N*betaz**2*r33*(5*alphaz**2 - 10*alphaz + 6)/(5*L))
            k += 1
            KGv[k] += r32*(N*betay**2*r32*(5*alphay**2 - 10*alphay + 6)/(5*L) + N*betay*betaz*r33*(5*alphay*alphaz - 5*alphay - 5*alphaz + 6)/(5*L)) + r33*(N*betay*betaz*r32*(5*alphay*alphaz - 5*alphay - 5*alphaz + 6)/(5*L) + N*betaz**2*r33*(5*alphaz**2 - 10*alphaz + 6)/(5*L))
            k += 1
            KGv[k] += r12*(-N*betay*betaz*r32/10 - N*betaz**2*r33/10) + r13*(N*betay**2*r32/10 + N*betay*betaz*r33/10)
            k += 1
            KGv[k] += r22*(-N*betay*betaz*r32/10 - N*betaz**2*r33/10) + r23*(N*betay**2*r32/10 + N*betay*betaz*r33/10)
            k += 1
            KGv[k] += r32*(-N*betay*betaz*r32/10 - N*betaz**2*r33/10) + r33*(N*betay**2*r32/10 + N*betay*betaz*r33/10)
            k += 1
            KGv[k] += r12*(N*betay**2*r32*(-5*alphay**2 + 10*alphay - 6)/(5*L) + N*betay*betaz*r33*(-5*alphay*alphaz + 5*alphay + 5*alphaz - 6)/(5*L)) + r13*(N*betay*betaz*r32*(-5*alphay*alphaz + 5*alphay + 5*alphaz - 6)/(5*L) + N*betaz**2*r33*(-5*alphaz**2 + 10*alphaz - 6)/(5*L))
            k += 1
            KGv[k] += r22*(N*betay**2*r32*(-5*alphay**2 + 10*alphay - 6)/(5*L) + N*betay*betaz*r33*(-5*alphay*alphaz + 5*alphay + 5*alphaz - 6)/(5*L)) + r23*(N*betay*betaz*r32*(-5*alphay*alphaz + 5*alphay + 5*alphaz - 6)/(5*L) + N*betaz**2*r33*(-5*alphaz**2 + 10*alphaz - 6)/(5*L))
            k += 1
            KGv[k] += r32*(N*betay**2*r32*(-5*alphay**2 + 10*alphay - 6)/(5*L) + N*betay*betaz*r33*(-5*alphay*alphaz + 5*alphay + 5*alphaz - 6)/(5*L)) + r33*(N*betay*betaz*r32*(-5*alphay*alphaz + 5*alphay + 5*alphaz - 6)/(5*L) + N*betaz**2*r33*(-5*alphaz**2 + 10*alphaz - 6)/(5*L))
            k += 1
            KGv[k] += r12*(-N*betay*betaz*r32/10 - N*betaz**2*r33/10) + r13*(N*betay**2*r32/10 + N*betay*betaz*r33/10)
            k += 1
            KGv[k] += r22*(-N*betay*betaz*r32/10 - N*betaz**2*r33/10) + r23*(N*betay**2*r32/10 + N*betay*betaz*r33/10)
            k += 1
            KGv[k] += r32*(-N*betay*betaz*r32/10 - N*betaz**2*r33/10) + r33*(N*betay**2*r32/10 + N*betay*betaz*r33/10)
            k += 1
            KGv[k] += r12*(N*betay**2*r13/10 - N*betay*betaz*r12/10) + r13*(N*betay*betaz*r13/10 - N*betaz**2*r12/10)
            k += 1
            KGv[k] += r22*(N*betay**2*r13/10 - N*betay*betaz*r12/10) + r23*(N*betay*betaz*r13/10 - N*betaz**2*r12/10)
            k += 1
            KGv[k] += r32*(N*betay**2*r13/10 - N*betay*betaz*r12/10) + r33*(N*betay*betaz*r13/10 - N*betaz**2*r12/10)
            k += 1
            KGv[k] += r12*(L*N*betay*betaz*r13*(-5*alphay*alphaz + 5*alphay + 5*alphaz - 8)/60 + L*N*betaz**2*r12*(5*alphaz**2 - 10*alphaz + 8)/60) + r13*(L*N*betay**2*r13*(5*alphay**2 - 10*alphay + 8)/60 + L*N*betay*betaz*r12*(-5*alphay*alphaz + 5*alphay + 5*alphaz - 8)/60)
            k += 1
            KGv[k] += r22*(L*N*betay*betaz*r13*(-5*alphay*alphaz + 5*alphay + 5*alphaz - 8)/60 + L*N*betaz**2*r12*(5*alphaz**2 - 10*alphaz + 8)/60) + r23*(L*N*betay**2*r13*(5*alphay**2 - 10*alphay + 8)/60 + L*N*betay*betaz*r12*(-5*alphay*alphaz + 5*alphay + 5*alphaz - 8)/60)
            k += 1
            KGv[k] += r32*(L*N*betay*betaz*r13*(-5*alphay*alphaz + 5*alphay + 5*alphaz - 8)/60 + L*N*betaz**2*r12*(5*alphaz**2 - 10*alphaz + 8)/60) + r33*(L*N*betay**2*r13*(5*alphay**2 - 10*alphay + 8)/60 + L*N*betay*betaz*r12*(-5*alphay*alphaz + 5*alphay + 5*alphaz - 8)/60)
            k += 1
            KGv[k] += r12*(-N*betay**2*r13/10 + N*betay*betaz*r12/10) + r13*(-N*betay*betaz*r13/10 + N*betaz**2*r12/10)
            k += 1
            KGv[k] += r22*(-N*betay**2*r13/10 + N*betay*betaz*r12/10) + r23*(-N*betay*betaz*r13/10 + N*betaz**2*r12/10)
            k += 1
            KGv[k] += r32*(-N*betay**2*r13/10 + N*betay*betaz*r12/10) + r33*(-N*betay*betaz*r13/10 + N*betaz**2*r12/10)
            k += 1
            KGv[k] += r12*(L*N*betay*betaz*r13*(5*alphay*alphaz - 5*alphay - 5*alphaz + 2)/60 + L*N*betaz**2*r12*(-5*alphaz**2 + 10*alphaz - 2)/60) + r13*(L*N*betay**2*r13*(-5*alphay**2 + 10*alphay - 2)/60 + L*N*betay*betaz*r12*(5*alphay*alphaz - 5*alphay - 5*alphaz + 2)/60)
            k += 1
            KGv[k] += r22*(L*N*betay*betaz*r13*(5*alphay*alphaz - 5*alphay - 5*alphaz + 2)/60 + L*N*betaz**2*r12*(-5*alphaz**2 + 10*alphaz - 2)/60) + r23*(L*N*betay**2*r13*(-5*alphay**2 + 10*alphay - 2)/60 + L*N*betay*betaz*r12*(5*alphay*alphaz - 5*alphay - 5*alphaz + 2)/60)
            k += 1
            KGv[k] += r32*(L*N*betay*betaz*r13*(5*alphay*alphaz - 5*alphay - 5*alphaz + 2)/60 + L*N*betaz**2*r12*(-5*alphaz**2 + 10*alphaz - 2)/60) + r33*(L*N*betay**2*r13*(-5*alphay**2 + 10*alphay - 2)/60 + L*N*betay*betaz*r12*(5*alphay*alphaz - 5*alphay - 5*alphaz + 2)/60)
            k += 1
            KGv[k] += r12*(N*betay**2*r23/10 - N*betay*betaz*r22/10) + r13*(N*betay*betaz*r23/10 - N*betaz**2*r22/10)
            k += 1
            KGv[k] += r22*(N*betay**2*r23/10 - N*betay*betaz*r22/10) + r23*(N*betay*betaz*r23/10 - N*betaz**2*r22/10)
            k += 1
            KGv[k] += r32*(N*betay**2*r23/10 - N*betay*betaz*r22/10) + r33*(N*betay*betaz*r23/10 - N*betaz**2*r22/10)
            k += 1
            KGv[k] += r12*(L*N*betay*betaz*r23*(-5*alphay*alphaz + 5*alphay + 5*alphaz - 8)/60 + L*N*betaz**2*r22*(5*alphaz**2 - 10*alphaz + 8)/60) + r13*(L*N*betay**2*r23*(5*alphay**2 - 10*alphay + 8)/60 + L*N*betay*betaz*r22*(-5*alphay*alphaz + 5*alphay + 5*alphaz - 8)/60)
            k += 1
            KGv[k] += r22*(L*N*betay*betaz*r23*(-5*alphay*alphaz + 5*alphay + 5*alphaz - 8)/60 + L*N*betaz**2*r22*(5*alphaz**2 - 10*alphaz + 8)/60) + r23*(L*N*betay**2*r23*(5*alphay**2 - 10*alphay + 8)/60 + L*N*betay*betaz*r22*(-5*alphay*alphaz + 5*alphay + 5*alphaz - 8)/60)
            k += 1
            KGv[k] += r32*(L*N*betay*betaz*r23*(-5*alphay*alphaz + 5*alphay + 5*alphaz - 8)/60 + L*N*betaz**2*r22*(5*alphaz**2 - 10*alphaz + 8)/60) + r33*(L*N*betay**2*r23*(5*alphay**2 - 10*alphay + 8)/60 + L*N*betay*betaz*r22*(-5*alphay*alphaz + 5*alphay + 5*alphaz - 8)/60)
            k += 1
            KGv[k] += r12*(-N*betay**2*r23/10 + N*betay*betaz*r22/10) + r13*(-N*betay*betaz*r23/10 + N*betaz**2*r22/10)
            k += 1
            KGv[k] += r22*(-N*betay**2*r23/10 + N*betay*betaz*r22/10) + r23*(-N*betay*betaz*r23/10 + N*betaz**2*r22/10)
            k += 1
            KGv[k] += r32*(-N*betay**2*r23/10 + N*betay*betaz*r22/10) + r33*(-N*betay*betaz*r23/10 + N*betaz**2*r22/10)
            k += 1
            KGv[k] += r12*(L*N*betay*betaz*r23*(5*alphay*alphaz - 5*alphay - 5*alphaz + 2)/60 + L*N*betaz**2*r22*(-5*alphaz**2 + 10*alphaz - 2)/60) + r13*(L*N*betay**2*r23*(-5*alphay**2 + 10*alphay - 2)/60 + L*N*betay*betaz*r22*(5*alphay*alphaz - 5*alphay - 5*alphaz + 2)/60)
            k += 1
            KGv[k] += r22*(L*N*betay*betaz*r23*(5*alphay*alphaz - 5*alphay - 5*alphaz + 2)/60 + L*N*betaz**2*r22*(-5*alphaz**2 + 10*alphaz - 2)/60) + r23*(L*N*betay**2*r23*(-5*alphay**2 + 10*alphay - 2)/60 + L*N*betay*betaz*r22*(5*alphay*alphaz - 5*alphay - 5*alphaz + 2)/60)
            k += 1
            KGv[k] += r32*(L*N*betay*betaz*r23*(5*alphay*alphaz - 5*alphay - 5*alphaz + 2)/60 + L*N*betaz**2*r22*(-5*alphaz**2 + 10*alphaz - 2)/60) + r33*(L*N*betay**2*r23*(-5*alphay**2 + 10*alphay - 2)/60 + L*N*betay*betaz*r22*(5*alphay*alphaz - 5*alphay - 5*alphaz + 2)/60)
            k += 1
            KGv[k] += r12*(N*betay**2*r33/10 - N*betay*betaz*r32/10) + r13*(N*betay*betaz*r33/10 - N*betaz**2*r32/10)
            k += 1
            KGv[k] += r22*(N*betay**2*r33/10 - N*betay*betaz*r32/10) + r23*(N*betay*betaz*r33/10 - N*betaz**2*r32/10)
            k += 1
            KGv[k] += r32*(N*betay**2*r33/10 - N*betay*betaz*r32/10) + r33*(N*betay*betaz*r33/10 - N*betaz**2*r32/10)
            k += 1
            KGv[k] += r12*(L*N*betay*betaz*r33*(-5*alphay*alphaz + 5*alphay + 5*alphaz - 8)/60 + L*N*betaz**2*r32*(5*alphaz**2 - 10*alphaz + 8)/60) + r13*(L*N*betay**2*r33*(5*alphay**2 - 10*alphay + 8)/60 + L*N*betay*betaz*r32*(-5*alphay*alphaz + 5*alphay + 5*alphaz - 8)/60)
            k += 1
            KGv[k] += r22*(L*N*betay*betaz*r33*(-5*alphay*alphaz + 5*alphay + 5*alphaz - 8)/60 + L*N*betaz**2*r32*(5*alphaz**2 - 10*alphaz + 8)/60) + r23*(L*N*betay**2*r33*(5*alphay**2 - 10*alphay + 8)/60 + L*N*betay*betaz*r32*(-5*alphay*alphaz + 5*alphay + 5*alphaz - 8)/60)
            k += 1
            KGv[k] += r32*(L*N*betay*betaz*r33*(-5*alphay*alphaz + 5*alphay + 5*alphaz - 8)/60 + L*N*betaz**2*r32*(5*alphaz**2 - 10*alphaz + 8)/60) + r33*(L*N*betay**2*r33*(5*alphay**2 - 10*alphay + 8)/60 + L*N*betay*betaz*r32*(-5*alphay*alphaz + 5*alphay + 5*alphaz - 8)/60)
            k += 1
            KGv[k] += r12*(-N*betay**2*r33/10 + N*betay*betaz*r32/10) + r13*(-N*betay*betaz*r33/10 + N*betaz**2*r32/10)
            k += 1
            KGv[k] += r22*(-N*betay**2*r33/10 + N*betay*betaz*r32/10) + r23*(-N*betay*betaz*r33/10 + N*betaz**2*r32/10)
            k += 1
            KGv[k] += r32*(-N*betay**2*r33/10 + N*betay*betaz*r32/10) + r33*(-N*betay*betaz*r33/10 + N*betaz**2*r32/10)
            k += 1
            KGv[k] += r12*(L*N*betay*betaz*r33*(5*alphay*alphaz - 5*alphay - 5*alphaz + 2)/60 + L*N*betaz**2*r32*(-5*alphaz**2 + 10*alphaz - 2)/60) + r13*(L*N*betay**2*r33*(-5*alphay**2 + 10*alphay - 2)/60 + L*N*betay*betaz*r32*(5*alphay*alphaz - 5*alphay - 5*alphaz + 2)/60)
            k += 1
            KGv[k] += r22*(L*N*betay*betaz*r33*(5*alphay*alphaz - 5*alphay - 5*alphaz + 2)/60 + L*N*betaz**2*r32*(-5*alphaz**2 + 10*alphaz - 2)/60) + r23*(L*N*betay**2*r33*(-5*alphay**2 + 10*alphay - 2)/60 + L*N*betay*betaz*r32*(5*alphay*alphaz - 5*alphay - 5*alphaz + 2)/60)
            k += 1
            KGv[k] += r32*(L*N*betay*betaz*r33*(5*alphay*alphaz - 5*alphay - 5*alphaz + 2)/60 + L*N*betaz**2*r32*(-5*alphaz**2 + 10*alphaz - 2)/60) + r33*(L*N*betay**2*r33*(-5*alphay**2 + 10*alphay - 2)/60 + L*N*betay*betaz*r32*(5*alphay*alphaz - 5*alphay - 5*alphaz + 2)/60)
            k += 1
            KGv[k] += r12*(N*betay**2*r12*(-5*alphay**2 + 10*alphay - 6)/(5*L) + N*betay*betaz*r13*(-5*alphay*alphaz + 5*alphay + 5*alphaz - 6)/(5*L)) + r13*(N*betay*betaz*r12*(-5*alphay*alphaz + 5*alphay + 5*alphaz - 6)/(5*L) + N*betaz**2*r13*(-5*alphaz**2 + 10*alphaz - 6)/(5*L))
            k += 1
            KGv[k] += r22*(N*betay**2*r12*(-5*alphay**2 + 10*alphay - 6)/(5*L) + N*betay*betaz*r13*(-5*alphay*alphaz + 5*alphay + 5*alphaz - 6)/(5*L)) + r23*(N*betay*betaz*r12*(-5*alphay*alphaz + 5*alphay + 5*alphaz - 6)/(5*L) + N*betaz**2*r13*(-5*alphaz**2 + 10*alphaz - 6)/(5*L))
            k += 1
            KGv[k] += r32*(N*betay**2*r12*(-5*alphay**2 + 10*alphay - 6)/(5*L) + N*betay*betaz*r13*(-5*alphay*alphaz + 5*alphay + 5*alphaz - 6)/(5*L)) + r33*(N*betay*betaz*r12*(-5*alphay*alphaz + 5*alphay + 5*alphaz - 6)/(5*L) + N*betaz**2*r13*(-5*alphaz**2 + 10*alphaz - 6)/(5*L))
            k += 1
            KGv[k] += r12*(N*betay*betaz*r12/10 + N*betaz**2*r13/10) + r13*(-N*betay**2*r12/10 - N*betay*betaz*r13/10)
            k += 1
            KGv[k] += r22*(N*betay*betaz*r12/10 + N*betaz**2*r13/10) + r23*(-N*betay**2*r12/10 - N*betay*betaz*r13/10)
            k += 1
            KGv[k] += r32*(N*betay*betaz*r12/10 + N*betaz**2*r13/10) + r33*(-N*betay**2*r12/10 - N*betay*betaz*r13/10)
            k += 1
            KGv[k] += r12*(N*betay**2*r12*(5*alphay**2 - 10*alphay + 6)/(5*L) + N*betay*betaz*r13*(5*alphay*alphaz - 5*alphay - 5*alphaz + 6)/(5*L)) + r13*(N*betay*betaz*r12*(5*alphay*alphaz - 5*alphay - 5*alphaz + 6)/(5*L) + N*betaz**2*r13*(5*alphaz**2 - 10*alphaz + 6)/(5*L))
            k += 1
            KGv[k] += r22*(N*betay**2*r12*(5*alphay**2 - 10*alphay + 6)/(5*L) + N*betay*betaz*r13*(5*alphay*alphaz - 5*alphay - 5*alphaz + 6)/(5*L)) + r23*(N*betay*betaz*r12*(5*alphay*alphaz - 5*alphay - 5*alphaz + 6)/(5*L) + N*betaz**2*r13*(5*alphaz**2 - 10*alphaz + 6)/(5*L))
            k += 1
            KGv[k] += r32*(N*betay**2*r12*(5*alphay**2 - 10*alphay + 6)/(5*L) + N*betay*betaz*r13*(5*alphay*alphaz - 5*alphay - 5*alphaz + 6)/(5*L)) + r33*(N*betay*betaz*r12*(5*alphay*alphaz - 5*alphay - 5*alphaz + 6)/(5*L) + N*betaz**2*r13*(5*alphaz**2 - 10*alphaz + 6)/(5*L))
            k += 1
            KGv[k] += r12*(N*betay*betaz*r12/10 + N*betaz**2*r13/10) + r13*(-N*betay**2*r12/10 - N*betay*betaz*r13/10)
            k += 1
            KGv[k] += r22*(N*betay*betaz*r12/10 + N*betaz**2*r13/10) + r23*(-N*betay**2*r12/10 - N*betay*betaz*r13/10)
            k += 1
            KGv[k] += r32*(N*betay*betaz*r12/10 + N*betaz**2*r13/10) + r33*(-N*betay**2*r12/10 - N*betay*betaz*r13/10)
            k += 1
            KGv[k] += r12*(N*betay**2*r22*(-5*alphay**2 + 10*alphay - 6)/(5*L) + N*betay*betaz*r23*(-5*alphay*alphaz + 5*alphay + 5*alphaz - 6)/(5*L)) + r13*(N*betay*betaz*r22*(-5*alphay*alphaz + 5*alphay + 5*alphaz - 6)/(5*L) + N*betaz**2*r23*(-5*alphaz**2 + 10*alphaz - 6)/(5*L))
            k += 1
            KGv[k] += r22*(N*betay**2*r22*(-5*alphay**2 + 10*alphay - 6)/(5*L) + N*betay*betaz*r23*(-5*alphay*alphaz + 5*alphay + 5*alphaz - 6)/(5*L)) + r23*(N*betay*betaz*r22*(-5*alphay*alphaz + 5*alphay + 5*alphaz - 6)/(5*L) + N*betaz**2*r23*(-5*alphaz**2 + 10*alphaz - 6)/(5*L))
            k += 1
            KGv[k] += r32*(N*betay**2*r22*(-5*alphay**2 + 10*alphay - 6)/(5*L) + N*betay*betaz*r23*(-5*alphay*alphaz + 5*alphay + 5*alphaz - 6)/(5*L)) + r33*(N*betay*betaz*r22*(-5*alphay*alphaz + 5*alphay + 5*alphaz - 6)/(5*L) + N*betaz**2*r23*(-5*alphaz**2 + 10*alphaz - 6)/(5*L))
            k += 1
            KGv[k] += r12*(N*betay*betaz*r22/10 + N*betaz**2*r23/10) + r13*(-N*betay**2*r22/10 - N*betay*betaz*r23/10)
            k += 1
            KGv[k] += r22*(N*betay*betaz*r22/10 + N*betaz**2*r23/10) + r23*(-N*betay**2*r22/10 - N*betay*betaz*r23/10)
            k += 1
            KGv[k] += r32*(N*betay*betaz*r22/10 + N*betaz**2*r23/10) + r33*(-N*betay**2*r22/10 - N*betay*betaz*r23/10)
            k += 1
            KGv[k] += r12*(N*betay**2*r22*(5*alphay**2 - 10*alphay + 6)/(5*L) + N*betay*betaz*r23*(5*alphay*alphaz - 5*alphay - 5*alphaz + 6)/(5*L)) + r13*(N*betay*betaz*r22*(5*alphay*alphaz - 5*alphay - 5*alphaz + 6)/(5*L) + N*betaz**2*r23*(5*alphaz**2 - 10*alphaz + 6)/(5*L))
            k += 1
            KGv[k] += r22*(N*betay**2*r22*(5*alphay**2 - 10*alphay + 6)/(5*L) + N*betay*betaz*r23*(5*alphay*alphaz - 5*alphay - 5*alphaz + 6)/(5*L)) + r23*(N*betay*betaz*r22*(5*alphay*alphaz - 5*alphay - 5*alphaz + 6)/(5*L) + N*betaz**2*r23*(5*alphaz**2 - 10*alphaz + 6)/(5*L))
            k += 1
            KGv[k] += r32*(N*betay**2*r22*(5*alphay**2 - 10*alphay + 6)/(5*L) + N*betay*betaz*r23*(5*alphay*alphaz - 5*alphay - 5*alphaz + 6)/(5*L)) + r33*(N*betay*betaz*r22*(5*alphay*alphaz - 5*alphay - 5*alphaz + 6)/(5*L) + N*betaz**2*r23*(5*alphaz**2 - 10*alphaz + 6)/(5*L))
            k += 1
            KGv[k] += r12*(N*betay*betaz*r22/10 + N*betaz**2*r23/10) + r13*(-N*betay**2*r22/10 - N*betay*betaz*r23/10)
            k += 1
            KGv[k] += r22*(N*betay*betaz*r22/10 + N*betaz**2*r23/10) + r23*(-N*betay**2*r22/10 - N*betay*betaz*r23/10)
            k += 1
            KGv[k] += r32*(N*betay*betaz*r22/10 + N*betaz**2*r23/10) + r33*(-N*betay**2*r22/10 - N*betay*betaz*r23/10)
            k += 1
            KGv[k] += r12*(N*betay**2*r32*(-5*alphay**2 + 10*alphay - 6)/(5*L) + N*betay*betaz*r33*(-5*alphay*alphaz + 5*alphay + 5*alphaz - 6)/(5*L)) + r13*(N*betay*betaz*r32*(-5*alphay*alphaz + 5*alphay + 5*alphaz - 6)/(5*L) + N*betaz**2*r33*(-5*alphaz**2 + 10*alphaz - 6)/(5*L))
            k += 1
            KGv[k] += r22*(N*betay**2*r32*(-5*alphay**2 + 10*alphay - 6)/(5*L) + N*betay*betaz*r33*(-5*alphay*alphaz + 5*alphay + 5*alphaz - 6)/(5*L)) + r23*(N*betay*betaz*r32*(-5*alphay*alphaz + 5*alphay + 5*alphaz - 6)/(5*L) + N*betaz**2*r33*(-5*alphaz**2 + 10*alphaz - 6)/(5*L))
            k += 1
            KGv[k] += r32*(N*betay**2*r32*(-5*alphay**2 + 10*alphay - 6)/(5*L) + N*betay*betaz*r33*(-5*alphay*alphaz + 5*alphay + 5*alphaz - 6)/(5*L)) + r33*(N*betay*betaz*r32*(-5*alphay*alphaz + 5*alphay + 5*alphaz - 6)/(5*L) + N*betaz**2*r33*(-5*alphaz**2 + 10*alphaz - 6)/(5*L))
            k += 1
            KGv[k] += r12*(N*betay*betaz*r32/10 + N*betaz**2*r33/10) + r13*(-N*betay**2*r32/10 - N*betay*betaz*r33/10)
            k += 1
            KGv[k] += r22*(N*betay*betaz*r32/10 + N*betaz**2*r33/10) + r23*(-N*betay**2*r32/10 - N*betay*betaz*r33/10)
            k += 1
            KGv[k] += r32*(N*betay*betaz*r32/10 + N*betaz**2*r33/10) + r33*(-N*betay**2*r32/10 - N*betay*betaz*r33/10)
            k += 1
            KGv[k] += r12*(N*betay**2*r32*(5*alphay**2 - 10*alphay + 6)/(5*L) + N*betay*betaz*r33*(5*alphay*alphaz - 5*alphay - 5*alphaz + 6)/(5*L)) + r13*(N*betay*betaz*r32*(5*alphay*alphaz - 5*alphay - 5*alphaz + 6)/(5*L) + N*betaz**2*r33*(5*alphaz**2 - 10*alphaz + 6)/(5*L))
            k += 1
            KGv[k] += r22*(N*betay**2*r32*(5*alphay**2 - 10*alphay + 6)/(5*L) + N*betay*betaz*r33*(5*alphay*alphaz - 5*alphay - 5*alphaz + 6)/(5*L)) + r23*(N*betay*betaz*r32*(5*alphay*alphaz - 5*alphay - 5*alphaz + 6)/(5*L) + N*betaz**2*r33*(5*alphaz**2 - 10*alphaz + 6)/(5*L))
            k += 1
            KGv[k] += r32*(N*betay**2*r32*(5*alphay**2 - 10*alphay + 6)/(5*L) + N*betay*betaz*r33*(5*alphay*alphaz - 5*alphay - 5*alphaz + 6)/(5*L)) + r33*(N*betay*betaz*r32*(5*alphay*alphaz - 5*alphay - 5*alphaz + 6)/(5*L) + N*betaz**2*r33*(5*alphaz**2 - 10*alphaz + 6)/(5*L))
            k += 1
            KGv[k] += r12*(N*betay*betaz*r32/10 + N*betaz**2*r33/10) + r13*(-N*betay**2*r32/10 - N*betay*betaz*r33/10)
            k += 1
            KGv[k] += r22*(N*betay*betaz*r32/10 + N*betaz**2*r33/10) + r23*(-N*betay**2*r32/10 - N*betay*betaz*r33/10)
            k += 1
            KGv[k] += r32*(N*betay*betaz*r32/10 + N*betaz**2*r33/10) + r33*(-N*betay**2*r32/10 - N*betay*betaz*r33/10)
            k += 1
            KGv[k] += r12*(N*betay**2*r13/10 - N*betay*betaz*r12/10) + r13*(N*betay*betaz*r13/10 - N*betaz**2*r12/10)
            k += 1
            KGv[k] += r22*(N*betay**2*r13/10 - N*betay*betaz*r12/10) + r23*(N*betay*betaz*r13/10 - N*betaz**2*r12/10)
            k += 1
            KGv[k] += r32*(N*betay**2*r13/10 - N*betay*betaz*r12/10) + r33*(N*betay*betaz*r13/10 - N*betaz**2*r12/10)
            k += 1
            KGv[k] += r12*(L*N*betay*betaz*r13*(5*alphay*alphaz - 5*alphay - 5*alphaz + 2)/60 + L*N*betaz**2*r12*(-5*alphaz**2 + 10*alphaz - 2)/60) + r13*(L*N*betay**2*r13*(-5*alphay**2 + 10*alphay - 2)/60 + L*N*betay*betaz*r12*(5*alphay*alphaz - 5*alphay - 5*alphaz + 2)/60)
            k += 1
            KGv[k] += r22*(L*N*betay*betaz*r13*(5*alphay*alphaz - 5*alphay - 5*alphaz + 2)/60 + L*N*betaz**2*r12*(-5*alphaz**2 + 10*alphaz - 2)/60) + r23*(L*N*betay**2*r13*(-5*alphay**2 + 10*alphay - 2)/60 + L*N*betay*betaz*r12*(5*alphay*alphaz - 5*alphay - 5*alphaz + 2)/60)
            k += 1
            KGv[k] += r32*(L*N*betay*betaz*r13*(5*alphay*alphaz - 5*alphay - 5*alphaz + 2)/60 + L*N*betaz**2*r12*(-5*alphaz**2 + 10*alphaz - 2)/60) + r33*(L*N*betay**2*r13*(-5*alphay**2 + 10*alphay - 2)/60 + L*N*betay*betaz*r12*(5*alphay*alphaz - 5*alphay - 5*alphaz + 2)/60)
            k += 1
            KGv[k] += r12*(-N*betay**2*r13/10 + N*betay*betaz*r12/10) + r13*(-N*betay*betaz*r13/10 + N*betaz**2*r12/10)
            k += 1
            KGv[k] += r22*(-N*betay**2*r13/10 + N*betay*betaz*r12/10) + r23*(-N*betay*betaz*r13/10 + N*betaz**2*r12/10)
            k += 1
            KGv[k] += r32*(-N*betay**2*r13/10 + N*betay*betaz*r12/10) + r33*(-N*betay*betaz*r13/10 + N*betaz**2*r12/10)
            k += 1
            KGv[k] += r12*(L*N*betay*betaz*r13*(-5*alphay*alphaz + 5*alphay + 5*alphaz - 8)/60 + L*N*betaz**2*r12*(5*alphaz**2 - 10*alphaz + 8)/60) + r13*(L*N*betay**2*r13*(5*alphay**2 - 10*alphay + 8)/60 + L*N*betay*betaz*r12*(-5*alphay*alphaz + 5*alphay + 5*alphaz - 8)/60)
            k += 1
            KGv[k] += r22*(L*N*betay*betaz*r13*(-5*alphay*alphaz + 5*alphay + 5*alphaz - 8)/60 + L*N*betaz**2*r12*(5*alphaz**2 - 10*alphaz + 8)/60) + r23*(L*N*betay**2*r13*(5*alphay**2 - 10*alphay + 8)/60 + L*N*betay*betaz*r12*(-5*alphay*alphaz + 5*alphay + 5*alphaz - 8)/60)
            k += 1
            KGv[k] += r32*(L*N*betay*betaz*r13*(-5*alphay*alphaz + 5*alphay + 5*alphaz - 8)/60 + L*N*betaz**2*r12*(5*alphaz**2 - 10*alphaz + 8)/60) + r33*(L*N*betay**2*r13*(5*alphay**2 - 10*alphay + 8)/60 + L*N*betay*betaz*r12*(-5*alphay*alphaz + 5*alphay + 5*alphaz - 8)/60)
            k += 1
            KGv[k] += r12*(N*betay**2*r23/10 - N*betay*betaz*r22/10) + r13*(N*betay*betaz*r23/10 - N*betaz**2*r22/10)
            k += 1
            KGv[k] += r22*(N*betay**2*r23/10 - N*betay*betaz*r22/10) + r23*(N*betay*betaz*r23/10 - N*betaz**2*r22/10)
            k += 1
            KGv[k] += r32*(N*betay**2*r23/10 - N*betay*betaz*r22/10) + r33*(N*betay*betaz*r23/10 - N*betaz**2*r22/10)
            k += 1
            KGv[k] += r12*(L*N*betay*betaz*r23*(5*alphay*alphaz - 5*alphay - 5*alphaz + 2)/60 + L*N*betaz**2*r22*(-5*alphaz**2 + 10*alphaz - 2)/60) + r13*(L*N*betay**2*r23*(-5*alphay**2 + 10*alphay - 2)/60 + L*N*betay*betaz*r22*(5*alphay*alphaz - 5*alphay - 5*alphaz + 2)/60)
            k += 1
            KGv[k] += r22*(L*N*betay*betaz*r23*(5*alphay*alphaz - 5*alphay - 5*alphaz + 2)/60 + L*N*betaz**2*r22*(-5*alphaz**2 + 10*alphaz - 2)/60) + r23*(L*N*betay**2*r23*(-5*alphay**2 + 10*alphay - 2)/60 + L*N*betay*betaz*r22*(5*alphay*alphaz - 5*alphay - 5*alphaz + 2)/60)
            k += 1
            KGv[k] += r32*(L*N*betay*betaz*r23*(5*alphay*alphaz - 5*alphay - 5*alphaz + 2)/60 + L*N*betaz**2*r22*(-5*alphaz**2 + 10*alphaz - 2)/60) + r33*(L*N*betay**2*r23*(-5*alphay**2 + 10*alphay - 2)/60 + L*N*betay*betaz*r22*(5*alphay*alphaz - 5*alphay - 5*alphaz + 2)/60)
            k += 1
            KGv[k] += r12*(-N*betay**2*r23/10 + N*betay*betaz*r22/10) + r13*(-N*betay*betaz*r23/10 + N*betaz**2*r22/10)
            k += 1
            KGv[k] += r22*(-N*betay**2*r23/10 + N*betay*betaz*r22/10) + r23*(-N*betay*betaz*r23/10 + N*betaz**2*r22/10)
            k += 1
            KGv[k] += r32*(-N*betay**2*r23/10 + N*betay*betaz*r22/10) + r33*(-N*betay*betaz*r23/10 + N*betaz**2*r22/10)
            k += 1
            KGv[k] += r12*(L*N*betay*betaz*r23*(-5*alphay*alphaz + 5*alphay + 5*alphaz - 8)/60 + L*N*betaz**2*r22*(5*alphaz**2 - 10*alphaz + 8)/60) + r13*(L*N*betay**2*r23*(5*alphay**2 - 10*alphay + 8)/60 + L*N*betay*betaz*r22*(-5*alphay*alphaz + 5*alphay + 5*alphaz - 8)/60)
            k += 1
            KGv[k] += r22*(L*N*betay*betaz*r23*(-5*alphay*alphaz + 5*alphay + 5*alphaz - 8)/60 + L*N*betaz**2*r22*(5*alphaz**2 - 10*alphaz + 8)/60) + r23*(L*N*betay**2*r23*(5*alphay**2 - 10*alphay + 8)/60 + L*N*betay*betaz*r22*(-5*alphay*alphaz + 5*alphay + 5*alphaz - 8)/60)
            k += 1
            KGv[k] += r32*(L*N*betay*betaz*r23*(-5*alphay*alphaz + 5*alphay + 5*alphaz - 8)/60 + L*N*betaz**2*r22*(5*alphaz**2 - 10*alphaz + 8)/60) + r33*(L*N*betay**2*r23*(5*alphay**2 - 10*alphay + 8)/60 + L*N*betay*betaz*r22*(-5*alphay*alphaz + 5*alphay + 5*alphaz - 8)/60)
            k += 1
            KGv[k] += r12*(N*betay**2*r33/10 - N*betay*betaz*r32/10) + r13*(N*betay*betaz*r33/10 - N*betaz**2*r32/10)
            k += 1
            KGv[k] += r22*(N*betay**2*r33/10 - N*betay*betaz*r32/10) + r23*(N*betay*betaz*r33/10 - N*betaz**2*r32/10)
            k += 1
            KGv[k] += r32*(N*betay**2*r33/10 - N*betay*betaz*r32/10) + r33*(N*betay*betaz*r33/10 - N*betaz**2*r32/10)
            k += 1
            KGv[k] += r12*(L*N*betay*betaz*r33*(5*alphay*alphaz - 5*alphay - 5*alphaz + 2)/60 + L*N*betaz**2*r32*(-5*alphaz**2 + 10*alphaz - 2)/60) + r13*(L*N*betay**2*r33*(-5*alphay**2 + 10*alphay - 2)/60 + L*N*betay*betaz*r32*(5*alphay*alphaz - 5*alphay - 5*alphaz + 2)/60)
            k += 1
            KGv[k] += r22*(L*N*betay*betaz*r33*(5*alphay*alphaz - 5*alphay - 5*alphaz + 2)/60 + L*N*betaz**2*r32*(-5*alphaz**2 + 10*alphaz - 2)/60) + r23*(L*N*betay**2*r33*(-5*alphay**2 + 10*alphay - 2)/60 + L*N*betay*betaz*r32*(5*alphay*alphaz - 5*alphay - 5*alphaz + 2)/60)
            k += 1
            KGv[k] += r32*(L*N*betay*betaz*r33*(5*alphay*alphaz - 5*alphay - 5*alphaz + 2)/60 + L*N*betaz**2*r32*(-5*alphaz**2 + 10*alphaz - 2)/60) + r33*(L*N*betay**2*r33*(-5*alphay**2 + 10*alphay - 2)/60 + L*N*betay*betaz*r32*(5*alphay*alphaz - 5*alphay - 5*alphaz + 2)/60)
            k += 1
            KGv[k] += r12*(-N*betay**2*r33/10 + N*betay*betaz*r32/10) + r13*(-N*betay*betaz*r33/10 + N*betaz**2*r32/10)
            k += 1
            KGv[k] += r22*(-N*betay**2*r33/10 + N*betay*betaz*r32/10) + r23*(-N*betay*betaz*r33/10 + N*betaz**2*r32/10)
            k += 1
            KGv[k] += r32*(-N*betay**2*r33/10 + N*betay*betaz*r32/10) + r33*(-N*betay*betaz*r33/10 + N*betaz**2*r32/10)
            k += 1
            KGv[k] += r12*(L*N*betay*betaz*r33*(-5*alphay*alphaz + 5*alphay + 5*alphaz - 8)/60 + L*N*betaz**2*r32*(5*alphaz**2 - 10*alphaz + 8)/60) + r13*(L*N*betay**2*r33*(5*alphay**2 - 10*alphay + 8)/60 + L*N*betay*betaz*r32*(-5*alphay*alphaz + 5*alphay + 5*alphaz - 8)/60)
            k += 1
            KGv[k] += r22*(L*N*betay*betaz*r33*(-5*alphay*alphaz + 5*alphay + 5*alphaz - 8)/60 + L*N*betaz**2*r32*(5*alphaz**2 - 10*alphaz + 8)/60) + r23*(L*N*betay**2*r33*(5*alphay**2 - 10*alphay + 8)/60 + L*N*betay*betaz*r32*(-5*alphay*alphaz + 5*alphay + 5*alphaz - 8)/60)
            k += 1
            KGv[k] += r32*(L*N*betay*betaz*r33*(-5*alphay*alphaz + 5*alphay + 5*alphaz - 8)/60 + L*N*betaz**2*r32*(5*alphaz**2 - 10*alphaz + 8)/60) + r33*(L*N*betay**2*r33*(5*alphay**2 - 10*alphay + 8)/60 + L*N*betay*betaz*r32*(-5*alphay*alphaz + 5*alphay + 5*alphaz - 8)/60)


    cpdef void update_M(BeamC self,
                        long [::1] Mr,
                        long [::1] Mc,
                        double [::1] Mv,
                        BeamProp prop,
                        int mtype=0,
                        ):
        r"""Update sparse vectors for mass matrix M

        Parameters
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
        cdef double L, A, E, G, Iyy, Izz, Iyz, J
        cdef double alphay, alphaz, betay, betaz

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
            G = prop.G
            Iyy = prop.Iyy
            Izz = prop.Izz
            Iyz = prop.Iyz
            J = prop.J

            alphay = 12*E*Izz/(G*A*L**2)
            alphaz = 12*E*Iyy/(G*A*L**2)
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
