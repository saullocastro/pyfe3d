#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: nonecheck=False
#cython: infer_types=False
"""
BeamC - Consistent Timoshenko 3D beam element
---------------------------------------------

"""
import numpy as np
cimport numpy as np

from .beamprop cimport BeamProp

ctypedef np.int64_t cINT
INT = np.int64
ctypedef np.double_t cDOUBLE
DOUBLE = np.float64
cdef cINT DOF = 6
cdef cINT NUM_NODES = 2

cdef class BeamCData:
    r"""
    Used to allocate memory for the sparse matrices.

    Attributes
    ----------
    KC0_SPARSE_SIZE : int
        ``KC0_SPARSE_SIZE = 144``

    KG_SPARSE_SIZE : int
        ``KG_SPARSE_SIZE = 144``

    M_SPARSE_SIZE : int
        ``M_SPARSE_SIZE = 144``

    """
    cdef public cINT KC0_SPARSE_SIZE
    cdef public cINT KG_SPARSE_SIZE
    cdef public cINT M_SPARSE_SIZE
    def __cinit__(BeamCData self):
        self.KC0_SPARSE_SIZE = 144
        self.KG_SPARSE_SIZE = 144
        self.M_SPARSE_SIZE = 144

cdef class BeamCProbe:
    r"""
    Probe used for local coordinates, local displacements, local stresses etc

    Attributes
    ----------
    xe : array-like
        Array of size ``NUM_NODES*DOF//2=6`` containing the nodal coordinates
        in the element coordinate system, in the following order `{x_e}_1,
        {y_e}_1, {z_e}_1, {x_e}_2, {y_e}_2, {z_e}_2`.
    ue : array-like
        Array of size ``NUM_NODES*DOF=12`` containing the element displacements
        in the following order `{u_e}_1, {v_e}_1, {w_e}_1, {{r_x}_e}_1,
        {{r_y}_e}_1, {{r_z}_e}_1, {u_e}_2, {v_e}_2, {w_e}_2, {{r_x}_e}_2,
        {{r_y}_e}_2, {{r_z}_e}_2`.

    """
    cdef public cDOUBLE[:] xe
    cdef public cDOUBLE[:] ue
    def __cinit__(BeamCProbe self):
        self.xe = np.zeros(NUM_NODES*DOF//2, dtype=DOUBLE)
        self.ue = np.zeros(NUM_NODES*DOF, dtype=DOUBLE)

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
    eid : int
        Element identification number.
    length : double
        Element length.
    cosa, cosb, cosg : double
        Cossine of rotation angles that define the 3D position of the element.
    c1, c2 : int
        Position of each node in the global stiffness matrix.
    n1, n2 : int
        Node identification number.
    init_k_KC0, init_k_KG, init_k_M : int
        Position in the arrays storing the sparse data for the structural
        matrices.
    _p : :class:`.BeamCProbe` object
        Pointer to the probe.

    """
    cdef public cINT eid
    cdef public cINT n1, n2
    cdef public cINT c1, c2
    cdef public cINT init_k_KC0, init_k_KG, init_k_M
    cdef public double length
    cdef public double cosa, cosb, cosg
    cdef BeamCProbe _p

    def __cinit__(BeamC self, BeamCProbe p):
        self._p = p
        self.eid = -1
        self.n1 = -1
        self.n2 = -1
        self.c1 = -1
        self.c2 = -1
        self.init_k_KC0 = 0
        #self.init_k_KCNL = 0
        self.init_k_KG = 0
        self.init_k_M = 0
        self.length = 0
        self.cosa = 1.
        self.cosb = 1.
        self.cosg = 1.

    cpdef void update_ue(BeamC self, np.ndarray[cDOUBLE, ndim=1] u):
       r"""Update the local displacement vector of the element

        Parameters
        ----------
        u : array-like
            Array with global displacements, for a total of `M` nodes in
            the model, this array will be arranged as: `u_1, v_1, w_1, {r_x}_1,
            {r_y}_1, {r_z}_1, u_2, v_2, w_2, {r_x}_2, {r_y}_2, {r_z}_2, ...,
            u_M, v_M, w_M, {r_x}_M, {r_y}_M, {r_z}_M`.

        """
        cdef int i, j
        cdef cINT c[2]
        cdef double su[3]
        cdef double sv[3]
        cdef double sw[3]
        cdef double sina, sinb, sing

        #FIXME double check all this part
        with nogil:
            # positions in the global stiffness matrix
            c[0] = self.c1
            c[1] = self.c2

            # global to local transformation of displacements
            sina = (1. - self.cosa**2)**0.5
            sinb = (1. - self.cosb**2)**0.5
            sing = (1. - self.cosg**2)**0.5

            su[0] = self.cosb*self.cosg
            su[1] = self.cosb*sing
            su[2] = -sinb
            sv[0] = -self.cosa*sing + self.cosg*sina*sinb
            sv[1] = self.cosa*self.cosg + sina*sinb*sing
            sv[2] = self.cosb*sina
            sw[0] = self.cosa*self.cosg*sinb + sina*sing
            sw[1] = self.cosa*sinb*sing - self.cosg*sina
            sw[2] = self.cosa*self.cosb

            for j in range(NUM_NODES):
                for i in range(DOF):
                    self._p.ue[j*DOF + i] = 0

            for j in range(NUM_NODES):
                for i in range(DOF//2):
                    #transforming translations
                    self._p.ue[j*DOF + 0] += su[i]*u[c[j] + 0 + i]
                    self._p.ue[j*DOF + 1] += sv[i]*u[c[j] + 0 + i]
                    self._p.ue[j*DOF + 2] += sw[i]*u[c[j] + 0 + i]
                    #transforming rotations
                    self._p.ue[j*DOF + 3] += su[i]*u[c[j] + 3 + i]
                    self._p.ue[j*DOF + 4] += sv[i]*u[c[j] + 3 + i]
                    self._p.ue[j*DOF + 5] += sw[i]*u[c[j] + 3 + i]

    cpdef void update_xe(BeamC self, np.ndarray[cDOUBLE, ndim=1] x):
       r"""Update the 3D coordinates of the element

        Parameters
        ----------
        x : array-like
            Array with global nodal coordinates, for a total of `M` nodes in
            the model, this array will be arranged as: `x_1, y_1, z_1, x_2,
            y_2, z_2, ..., x_M, y_M, z_M`.

        """
        cdef int i, j
        cdef cINT c[2]
        cdef double su[3]
        cdef double sv[3]
        cdef double sw[3]
        cdef double sina, sinb, sing

        with nogil:
            # positions in the global stiffness matrix
            c[0] = self.c1
            c[1] = self.c2

            # global to local transformation of displacements
            sina = (1. - self.cosa**2)**0.5
            sinb = (1. - self.cosb**2)**0.5
            sing = (1. - self.cosg**2)**0.5

            su[0] = self.cosb*self.cosg
            su[1] = self.cosb*sing
            su[2] = -sinb
            sv[0] = -self.cosa*sing + self.cosg*sina*sinb
            sv[1] = self.cosa*self.cosg + sina*sinb*sing
            sv[2] = self.cosb*sina
            sw[0] = self.cosa*self.cosg*sinb + sina*sing
            sw[1] = self.cosa*sinb*sing - self.cosg*sina
            sw[2] = self.cosa*self.cosb

            for j in range(NUM_NODES):
                for i in range(DOF//2):
                    self._p.xe[j*DOF//2 + i] = 0

            for j in range(NUM_NODES):
                for i in range(DOF//2):
                    self._p.xe[j*DOF//2 + 0] += su[i]*x[c[j]//2 + i]
                    self._p.xe[j*DOF//2 + 1] += sv[i]*x[c[j]//2 + i]
                    self._p.xe[j*DOF//2 + 2] += sw[i]*x[c[j]//2 + i]

        self.update_length()

    cpdef void update_length(BeamC self):
       r"""Update element length

        """
        cdef double x1, x2, y1, y2, z1, z2
        with nogil:
            #NOTE ignoring z in local coordinates
            x1 = self._p.xe[0]
            y1 = self._p.xe[1]
            z1 = self._p.xe[2]
            x2 = self._p.xe[3]
            y2 = self._p.xe[4]
            z2 = self._p.xe[5]
            self.length = ((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)**0.5

    cpdef void update_KC0(BeamC self,
            np.ndarray[cINT, ndim=1] KC0r,
            np.ndarray[cINT, ndim=1] KC0c,
            np.ndarray[cDOUBLE, ndim=1] KC0v,
            BeamProp prop,
            int update_KC0v_only=0
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
        cdef cINT c1, c2, k
        cdef double L, A, E, G, scf, J, Ay, Az, Iyy, Izz
        cdef double r11, r12, r13, r21, r22, r23, r31, r32, r33
        cdef double sina, sinb, sing

        with nogil:
            L = self.length
            A = prop.A
            E = prop.E
            G = prop.G
            scf = prop.scf
            J = prop.J
            Ay = prop.Ay
            Az = prop.Az
            Iyy = prop.Iyy
            Izz = prop.Izz

            #Local to global transformation
            sina = (1. - self.cosa**2)**0.5
            sinb = (1. - self.cosb**2)**0.5
            sing = (1. - self.cosg**2)**0.5
            r11 = self.cosb*self.cosg
            r12 = -self.cosa*sing + self.cosg*sina*sinb
            r13 = self.cosa*self.cosg*sinb + sina*sing
            r21 = self.cosb*sing
            r22 = self.cosa*self.cosg + sina*sinb*sing
            r23 = self.cosa*sinb*sing - self.cosg*sina
            r31 = -sinb
            r32 = self.cosb*sina
            r33 = self.cosa*self.cosb

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
            KC0v[k] += A*E*r11**2/L + r12*(-12*A**2*E*G**2*J*L*r13*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) + 12*A*E*G*Izz*r12*scf*(A*G*L**2*scf + 12*E*Izz)/(L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2))) + r13*(-12*A**2*E*G**2*J*L*r12*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) + 12*A*G*r13*scf*(2*A**2*G**2*L**4*scf**2 - 15*A*E*G*Iyy*L**2*scf + 60*E**2*Iyy**2)/(5*L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)))
            k += 1
            KC0v[k] += A*E*r11*r21/L + r22*(-12*A**2*E*G**2*J*L*r13*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) + 12*A*E*G*Izz*r12*scf*(A*G*L**2*scf + 12*E*Izz)/(L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2))) + r23*(-12*A**2*E*G**2*J*L*r12*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) + 12*A*G*r13*scf*(2*A**2*G**2*L**4*scf**2 - 15*A*E*G*Iyy*L**2*scf + 60*E**2*Iyy**2)/(5*L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)))
            k += 1
            KC0v[k] += A*E*r11*r31/L + r32*(-12*A**2*E*G**2*J*L*r13*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) + 12*A*E*G*Izz*r12*scf*(A*G*L**2*scf + 12*E*Izz)/(L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2))) + r33*(-12*A**2*E*G**2*J*L*r12*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) + 12*A*G*r13*scf*(2*A**2*G**2*L**4*scf**2 - 15*A*E*G*Iyy*L**2*scf + 60*E**2*Iyy**2)/(5*L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)))
            k += 1
            KC0v[k] += r11*(2*Ay*G*r13*scf*(-A*G*L**2*scf + 6*E*Iyy)/(L*(A*G*L**2*scf - 12*E*Iyy)) + 12*Az*E*G*Izz*r12*scf/(L*(A*G*L**2*scf - 12*E*Izz))) + r12*(-6*A**2*E*G**2*J*L**2*r12*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) + 2*A*G*r13*scf*(A**2*G**2*L**4*scf**2 + 45*A*E*G*Iyy*L**2*scf - 180*E**2*Iyy**2)/(5*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)) + Az*E*r11/L) + r13*(-6*A**2*E*G**2*J*L**2*r13*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) + 6*A*E*G*Izz*r12*scf*(A*G*L**2*scf + 12*E*Izz)/(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2) - Ay*E*r11/L)
            k += 1
            KC0v[k] += r21*(2*Ay*G*r13*scf*(-A*G*L**2*scf + 6*E*Iyy)/(L*(A*G*L**2*scf - 12*E*Iyy)) + 12*Az*E*G*Izz*r12*scf/(L*(A*G*L**2*scf - 12*E*Izz))) + r22*(-6*A**2*E*G**2*J*L**2*r12*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) + 2*A*G*r13*scf*(A**2*G**2*L**4*scf**2 + 45*A*E*G*Iyy*L**2*scf - 180*E**2*Iyy**2)/(5*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)) + Az*E*r11/L) + r23*(-6*A**2*E*G**2*J*L**2*r13*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) + 6*A*E*G*Izz*r12*scf*(A*G*L**2*scf + 12*E*Izz)/(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2) - Ay*E*r11/L)
            k += 1
            KC0v[k] += r31*(2*Ay*G*r13*scf*(-A*G*L**2*scf + 6*E*Iyy)/(L*(A*G*L**2*scf - 12*E*Iyy)) + 12*Az*E*G*Izz*r12*scf/(L*(A*G*L**2*scf - 12*E*Izz))) + r32*(-6*A**2*E*G**2*J*L**2*r12*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) + 2*A*G*r13*scf*(A**2*G**2*L**4*scf**2 + 45*A*E*G*Iyy*L**2*scf - 180*E**2*Iyy**2)/(5*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)) + Az*E*r11/L) + r33*(-6*A**2*E*G**2*J*L**2*r13*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) + 6*A*E*G*Izz*r12*scf*(A*G*L**2*scf + 12*E*Izz)/(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2) - Ay*E*r11/L)
            k += 1
            KC0v[k] += -A*E*r11**2/L + r12*(12*A**2*E*G**2*J*L*r13*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) - 12*A*E*G*Izz*r12*scf*(A*G*L**2*scf + 12*E*Izz)/(L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2))) + r13*(12*A**2*E*G**2*J*L*r12*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) + 12*A*G*r13*scf*(-2*A**2*G**2*L**4*scf**2 + 15*A*E*G*Iyy*L**2*scf - 60*E**2*Iyy**2)/(5*L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)))
            k += 1
            KC0v[k] += -A*E*r11*r21/L + r22*(12*A**2*E*G**2*J*L*r13*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) - 12*A*E*G*Izz*r12*scf*(A*G*L**2*scf + 12*E*Izz)/(L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2))) + r23*(12*A**2*E*G**2*J*L*r12*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) + 12*A*G*r13*scf*(-2*A**2*G**2*L**4*scf**2 + 15*A*E*G*Iyy*L**2*scf - 60*E**2*Iyy**2)/(5*L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)))
            k += 1
            KC0v[k] += -A*E*r11*r31/L + r32*(12*A**2*E*G**2*J*L*r13*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) - 12*A*E*G*Izz*r12*scf*(A*G*L**2*scf + 12*E*Izz)/(L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2))) + r33*(12*A**2*E*G**2*J*L*r12*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) + 12*A*G*r13*scf*(-2*A**2*G**2*L**4*scf**2 + 15*A*E*G*Iyy*L**2*scf - 60*E**2*Iyy**2)/(5*L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)))
            k += 1
            KC0v[k] += r11*(2*Ay*G*r13*scf*(A*G*L**2*scf - 6*E*Iyy)/(L*(A*G*L**2*scf - 12*E*Iyy)) - 12*Az*E*G*Izz*r12*scf/(L*(A*G*L**2*scf - 12*E*Izz))) + r12*(-6*A**2*E*G**2*J*L**2*r12*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) + 2*A*G*r13*scf*(A**2*G**2*L**4*scf**2 + 45*A*E*G*Iyy*L**2*scf - 180*E**2*Iyy**2)/(5*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)) - Az*E*r11/L) + r13*(-6*A**2*E*G**2*J*L**2*r13*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) + 6*A*E*G*Izz*r12*scf*(A*G*L**2*scf + 12*E*Izz)/(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2) + Ay*E*r11/L)
            k += 1
            KC0v[k] += r21*(2*Ay*G*r13*scf*(A*G*L**2*scf - 6*E*Iyy)/(L*(A*G*L**2*scf - 12*E*Iyy)) - 12*Az*E*G*Izz*r12*scf/(L*(A*G*L**2*scf - 12*E*Izz))) + r22*(-6*A**2*E*G**2*J*L**2*r12*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) + 2*A*G*r13*scf*(A**2*G**2*L**4*scf**2 + 45*A*E*G*Iyy*L**2*scf - 180*E**2*Iyy**2)/(5*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)) - Az*E*r11/L) + r23*(-6*A**2*E*G**2*J*L**2*r13*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) + 6*A*E*G*Izz*r12*scf*(A*G*L**2*scf + 12*E*Izz)/(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2) + Ay*E*r11/L)
            k += 1
            KC0v[k] += r31*(2*Ay*G*r13*scf*(A*G*L**2*scf - 6*E*Iyy)/(L*(A*G*L**2*scf - 12*E*Iyy)) - 12*Az*E*G*Izz*r12*scf/(L*(A*G*L**2*scf - 12*E*Izz))) + r32*(-6*A**2*E*G**2*J*L**2*r12*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) + 2*A*G*r13*scf*(A**2*G**2*L**4*scf**2 + 45*A*E*G*Iyy*L**2*scf - 180*E**2*Iyy**2)/(5*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)) - Az*E*r11/L) + r33*(-6*A**2*E*G**2*J*L**2*r13*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) + 6*A*E*G*Izz*r12*scf*(A*G*L**2*scf + 12*E*Izz)/(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2) + Ay*E*r11/L)
            k += 1
            KC0v[k] += A*E*r11*r21/L + r12*(-12*A**2*E*G**2*J*L*r23*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) + 12*A*E*G*Izz*r22*scf*(A*G*L**2*scf + 12*E*Izz)/(L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2))) + r13*(-12*A**2*E*G**2*J*L*r22*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) + 12*A*G*r23*scf*(2*A**2*G**2*L**4*scf**2 - 15*A*E*G*Iyy*L**2*scf + 60*E**2*Iyy**2)/(5*L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)))
            k += 1
            KC0v[k] += A*E*r21**2/L + r22*(-12*A**2*E*G**2*J*L*r23*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) + 12*A*E*G*Izz*r22*scf*(A*G*L**2*scf + 12*E*Izz)/(L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2))) + r23*(-12*A**2*E*G**2*J*L*r22*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) + 12*A*G*r23*scf*(2*A**2*G**2*L**4*scf**2 - 15*A*E*G*Iyy*L**2*scf + 60*E**2*Iyy**2)/(5*L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)))
            k += 1
            KC0v[k] += A*E*r21*r31/L + r32*(-12*A**2*E*G**2*J*L*r23*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) + 12*A*E*G*Izz*r22*scf*(A*G*L**2*scf + 12*E*Izz)/(L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2))) + r33*(-12*A**2*E*G**2*J*L*r22*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) + 12*A*G*r23*scf*(2*A**2*G**2*L**4*scf**2 - 15*A*E*G*Iyy*L**2*scf + 60*E**2*Iyy**2)/(5*L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)))
            k += 1
            KC0v[k] += r11*(2*Ay*G*r23*scf*(-A*G*L**2*scf + 6*E*Iyy)/(L*(A*G*L**2*scf - 12*E*Iyy)) + 12*Az*E*G*Izz*r22*scf/(L*(A*G*L**2*scf - 12*E*Izz))) + r12*(-6*A**2*E*G**2*J*L**2*r22*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) + 2*A*G*r23*scf*(A**2*G**2*L**4*scf**2 + 45*A*E*G*Iyy*L**2*scf - 180*E**2*Iyy**2)/(5*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)) + Az*E*r21/L) + r13*(-6*A**2*E*G**2*J*L**2*r23*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) + 6*A*E*G*Izz*r22*scf*(A*G*L**2*scf + 12*E*Izz)/(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2) - Ay*E*r21/L)
            k += 1
            KC0v[k] += r21*(2*Ay*G*r23*scf*(-A*G*L**2*scf + 6*E*Iyy)/(L*(A*G*L**2*scf - 12*E*Iyy)) + 12*Az*E*G*Izz*r22*scf/(L*(A*G*L**2*scf - 12*E*Izz))) + r22*(-6*A**2*E*G**2*J*L**2*r22*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) + 2*A*G*r23*scf*(A**2*G**2*L**4*scf**2 + 45*A*E*G*Iyy*L**2*scf - 180*E**2*Iyy**2)/(5*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)) + Az*E*r21/L) + r23*(-6*A**2*E*G**2*J*L**2*r23*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) + 6*A*E*G*Izz*r22*scf*(A*G*L**2*scf + 12*E*Izz)/(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2) - Ay*E*r21/L)
            k += 1
            KC0v[k] += r31*(2*Ay*G*r23*scf*(-A*G*L**2*scf + 6*E*Iyy)/(L*(A*G*L**2*scf - 12*E*Iyy)) + 12*Az*E*G*Izz*r22*scf/(L*(A*G*L**2*scf - 12*E*Izz))) + r32*(-6*A**2*E*G**2*J*L**2*r22*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) + 2*A*G*r23*scf*(A**2*G**2*L**4*scf**2 + 45*A*E*G*Iyy*L**2*scf - 180*E**2*Iyy**2)/(5*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)) + Az*E*r21/L) + r33*(-6*A**2*E*G**2*J*L**2*r23*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) + 6*A*E*G*Izz*r22*scf*(A*G*L**2*scf + 12*E*Izz)/(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2) - Ay*E*r21/L)
            k += 1
            KC0v[k] += -A*E*r11*r21/L + r12*(12*A**2*E*G**2*J*L*r23*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) - 12*A*E*G*Izz*r22*scf*(A*G*L**2*scf + 12*E*Izz)/(L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2))) + r13*(12*A**2*E*G**2*J*L*r22*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) + 12*A*G*r23*scf*(-2*A**2*G**2*L**4*scf**2 + 15*A*E*G*Iyy*L**2*scf - 60*E**2*Iyy**2)/(5*L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)))
            k += 1
            KC0v[k] += -A*E*r21**2/L + r22*(12*A**2*E*G**2*J*L*r23*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) - 12*A*E*G*Izz*r22*scf*(A*G*L**2*scf + 12*E*Izz)/(L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2))) + r23*(12*A**2*E*G**2*J*L*r22*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) + 12*A*G*r23*scf*(-2*A**2*G**2*L**4*scf**2 + 15*A*E*G*Iyy*L**2*scf - 60*E**2*Iyy**2)/(5*L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)))
            k += 1
            KC0v[k] += -A*E*r21*r31/L + r32*(12*A**2*E*G**2*J*L*r23*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) - 12*A*E*G*Izz*r22*scf*(A*G*L**2*scf + 12*E*Izz)/(L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2))) + r33*(12*A**2*E*G**2*J*L*r22*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) + 12*A*G*r23*scf*(-2*A**2*G**2*L**4*scf**2 + 15*A*E*G*Iyy*L**2*scf - 60*E**2*Iyy**2)/(5*L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)))
            k += 1
            KC0v[k] += r11*(2*Ay*G*r23*scf*(A*G*L**2*scf - 6*E*Iyy)/(L*(A*G*L**2*scf - 12*E*Iyy)) - 12*Az*E*G*Izz*r22*scf/(L*(A*G*L**2*scf - 12*E*Izz))) + r12*(-6*A**2*E*G**2*J*L**2*r22*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) + 2*A*G*r23*scf*(A**2*G**2*L**4*scf**2 + 45*A*E*G*Iyy*L**2*scf - 180*E**2*Iyy**2)/(5*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)) - Az*E*r21/L) + r13*(-6*A**2*E*G**2*J*L**2*r23*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) + 6*A*E*G*Izz*r22*scf*(A*G*L**2*scf + 12*E*Izz)/(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2) + Ay*E*r21/L)
            k += 1
            KC0v[k] += r21*(2*Ay*G*r23*scf*(A*G*L**2*scf - 6*E*Iyy)/(L*(A*G*L**2*scf - 12*E*Iyy)) - 12*Az*E*G*Izz*r22*scf/(L*(A*G*L**2*scf - 12*E*Izz))) + r22*(-6*A**2*E*G**2*J*L**2*r22*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) + 2*A*G*r23*scf*(A**2*G**2*L**4*scf**2 + 45*A*E*G*Iyy*L**2*scf - 180*E**2*Iyy**2)/(5*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)) - Az*E*r21/L) + r23*(-6*A**2*E*G**2*J*L**2*r23*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) + 6*A*E*G*Izz*r22*scf*(A*G*L**2*scf + 12*E*Izz)/(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2) + Ay*E*r21/L)
            k += 1
            KC0v[k] += r31*(2*Ay*G*r23*scf*(A*G*L**2*scf - 6*E*Iyy)/(L*(A*G*L**2*scf - 12*E*Iyy)) - 12*Az*E*G*Izz*r22*scf/(L*(A*G*L**2*scf - 12*E*Izz))) + r32*(-6*A**2*E*G**2*J*L**2*r22*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) + 2*A*G*r23*scf*(A**2*G**2*L**4*scf**2 + 45*A*E*G*Iyy*L**2*scf - 180*E**2*Iyy**2)/(5*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)) - Az*E*r21/L) + r33*(-6*A**2*E*G**2*J*L**2*r23*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) + 6*A*E*G*Izz*r22*scf*(A*G*L**2*scf + 12*E*Izz)/(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2) + Ay*E*r21/L)
            k += 1
            KC0v[k] += A*E*r11*r31/L + r12*(-12*A**2*E*G**2*J*L*r33*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) + 12*A*E*G*Izz*r32*scf*(A*G*L**2*scf + 12*E*Izz)/(L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2))) + r13*(-12*A**2*E*G**2*J*L*r32*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) + 12*A*G*r33*scf*(2*A**2*G**2*L**4*scf**2 - 15*A*E*G*Iyy*L**2*scf + 60*E**2*Iyy**2)/(5*L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)))
            k += 1
            KC0v[k] += A*E*r21*r31/L + r22*(-12*A**2*E*G**2*J*L*r33*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) + 12*A*E*G*Izz*r32*scf*(A*G*L**2*scf + 12*E*Izz)/(L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2))) + r23*(-12*A**2*E*G**2*J*L*r32*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) + 12*A*G*r33*scf*(2*A**2*G**2*L**4*scf**2 - 15*A*E*G*Iyy*L**2*scf + 60*E**2*Iyy**2)/(5*L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)))
            k += 1
            KC0v[k] += A*E*r31**2/L + r32*(-12*A**2*E*G**2*J*L*r33*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) + 12*A*E*G*Izz*r32*scf*(A*G*L**2*scf + 12*E*Izz)/(L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2))) + r33*(-12*A**2*E*G**2*J*L*r32*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) + 12*A*G*r33*scf*(2*A**2*G**2*L**4*scf**2 - 15*A*E*G*Iyy*L**2*scf + 60*E**2*Iyy**2)/(5*L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)))
            k += 1
            KC0v[k] += r11*(2*Ay*G*r33*scf*(-A*G*L**2*scf + 6*E*Iyy)/(L*(A*G*L**2*scf - 12*E*Iyy)) + 12*Az*E*G*Izz*r32*scf/(L*(A*G*L**2*scf - 12*E*Izz))) + r12*(-6*A**2*E*G**2*J*L**2*r32*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) + 2*A*G*r33*scf*(A**2*G**2*L**4*scf**2 + 45*A*E*G*Iyy*L**2*scf - 180*E**2*Iyy**2)/(5*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)) + Az*E*r31/L) + r13*(-6*A**2*E*G**2*J*L**2*r33*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) + 6*A*E*G*Izz*r32*scf*(A*G*L**2*scf + 12*E*Izz)/(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2) - Ay*E*r31/L)
            k += 1
            KC0v[k] += r21*(2*Ay*G*r33*scf*(-A*G*L**2*scf + 6*E*Iyy)/(L*(A*G*L**2*scf - 12*E*Iyy)) + 12*Az*E*G*Izz*r32*scf/(L*(A*G*L**2*scf - 12*E*Izz))) + r22*(-6*A**2*E*G**2*J*L**2*r32*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) + 2*A*G*r33*scf*(A**2*G**2*L**4*scf**2 + 45*A*E*G*Iyy*L**2*scf - 180*E**2*Iyy**2)/(5*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)) + Az*E*r31/L) + r23*(-6*A**2*E*G**2*J*L**2*r33*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) + 6*A*E*G*Izz*r32*scf*(A*G*L**2*scf + 12*E*Izz)/(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2) - Ay*E*r31/L)
            k += 1
            KC0v[k] += r31*(2*Ay*G*r33*scf*(-A*G*L**2*scf + 6*E*Iyy)/(L*(A*G*L**2*scf - 12*E*Iyy)) + 12*Az*E*G*Izz*r32*scf/(L*(A*G*L**2*scf - 12*E*Izz))) + r32*(-6*A**2*E*G**2*J*L**2*r32*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) + 2*A*G*r33*scf*(A**2*G**2*L**4*scf**2 + 45*A*E*G*Iyy*L**2*scf - 180*E**2*Iyy**2)/(5*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)) + Az*E*r31/L) + r33*(-6*A**2*E*G**2*J*L**2*r33*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) + 6*A*E*G*Izz*r32*scf*(A*G*L**2*scf + 12*E*Izz)/(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2) - Ay*E*r31/L)
            k += 1
            KC0v[k] += -A*E*r11*r31/L + r12*(12*A**2*E*G**2*J*L*r33*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) - 12*A*E*G*Izz*r32*scf*(A*G*L**2*scf + 12*E*Izz)/(L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2))) + r13*(12*A**2*E*G**2*J*L*r32*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) + 12*A*G*r33*scf*(-2*A**2*G**2*L**4*scf**2 + 15*A*E*G*Iyy*L**2*scf - 60*E**2*Iyy**2)/(5*L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)))
            k += 1
            KC0v[k] += -A*E*r21*r31/L + r22*(12*A**2*E*G**2*J*L*r33*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) - 12*A*E*G*Izz*r32*scf*(A*G*L**2*scf + 12*E*Izz)/(L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2))) + r23*(12*A**2*E*G**2*J*L*r32*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) + 12*A*G*r33*scf*(-2*A**2*G**2*L**4*scf**2 + 15*A*E*G*Iyy*L**2*scf - 60*E**2*Iyy**2)/(5*L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)))
            k += 1
            KC0v[k] += -A*E*r31**2/L + r32*(12*A**2*E*G**2*J*L*r33*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) - 12*A*E*G*Izz*r32*scf*(A*G*L**2*scf + 12*E*Izz)/(L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2))) + r33*(12*A**2*E*G**2*J*L*r32*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) + 12*A*G*r33*scf*(-2*A**2*G**2*L**4*scf**2 + 15*A*E*G*Iyy*L**2*scf - 60*E**2*Iyy**2)/(5*L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)))
            k += 1
            KC0v[k] += r11*(2*Ay*G*r33*scf*(A*G*L**2*scf - 6*E*Iyy)/(L*(A*G*L**2*scf - 12*E*Iyy)) - 12*Az*E*G*Izz*r32*scf/(L*(A*G*L**2*scf - 12*E*Izz))) + r12*(-6*A**2*E*G**2*J*L**2*r32*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) + 2*A*G*r33*scf*(A**2*G**2*L**4*scf**2 + 45*A*E*G*Iyy*L**2*scf - 180*E**2*Iyy**2)/(5*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)) - Az*E*r31/L) + r13*(-6*A**2*E*G**2*J*L**2*r33*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) + 6*A*E*G*Izz*r32*scf*(A*G*L**2*scf + 12*E*Izz)/(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2) + Ay*E*r31/L)
            k += 1
            KC0v[k] += r21*(2*Ay*G*r33*scf*(A*G*L**2*scf - 6*E*Iyy)/(L*(A*G*L**2*scf - 12*E*Iyy)) - 12*Az*E*G*Izz*r32*scf/(L*(A*G*L**2*scf - 12*E*Izz))) + r22*(-6*A**2*E*G**2*J*L**2*r32*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) + 2*A*G*r33*scf*(A**2*G**2*L**4*scf**2 + 45*A*E*G*Iyy*L**2*scf - 180*E**2*Iyy**2)/(5*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)) - Az*E*r31/L) + r23*(-6*A**2*E*G**2*J*L**2*r33*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) + 6*A*E*G*Izz*r32*scf*(A*G*L**2*scf + 12*E*Izz)/(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2) + Ay*E*r31/L)
            k += 1
            KC0v[k] += r31*(2*Ay*G*r33*scf*(A*G*L**2*scf - 6*E*Iyy)/(L*(A*G*L**2*scf - 12*E*Iyy)) - 12*Az*E*G*Izz*r32*scf/(L*(A*G*L**2*scf - 12*E*Izz))) + r32*(-6*A**2*E*G**2*J*L**2*r32*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) + 2*A*G*r33*scf*(A**2*G**2*L**4*scf**2 + 45*A*E*G*Iyy*L**2*scf - 180*E**2*Iyy**2)/(5*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)) - Az*E*r31/L) + r33*(-6*A**2*E*G**2*J*L**2*r33*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) + 6*A*E*G*Izz*r32*scf*(A*G*L**2*scf + 12*E*Izz)/(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2) + Ay*E*r31/L)
            k += 1
            KC0v[k] += r11*(-Ay*E*r13/L + Az*E*r12/L) + r12*(-6*A**2*E*G**2*J*L**2*r12*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) + 6*A*E*G*Izz*r13*scf*(A*G*L**2*scf + 12*E*Izz)/(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2) + 12*Az*E*G*Izz*r11*scf/(L*(A*G*L**2*scf - 12*E*Izz))) + r13*(-6*A**2*E*G**2*J*L**2*r13*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) + 2*A*G*r12*scf*(A**2*G**2*L**4*scf**2 + 45*A*E*G*Iyy*L**2*scf - 180*E**2*Iyy**2)/(5*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)) + 2*Ay*G*r11*scf*(A*G*L**2*scf - 6*E*Iyy)/(L*(A*G*L**2*scf - 12*E*Iyy)))
            k += 1
            KC0v[k] += r21*(-Ay*E*r13/L + Az*E*r12/L) + r22*(-6*A**2*E*G**2*J*L**2*r12*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) + 6*A*E*G*Izz*r13*scf*(A*G*L**2*scf + 12*E*Izz)/(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2) + 12*Az*E*G*Izz*r11*scf/(L*(A*G*L**2*scf - 12*E*Izz))) + r23*(-6*A**2*E*G**2*J*L**2*r13*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) + 2*A*G*r12*scf*(A**2*G**2*L**4*scf**2 + 45*A*E*G*Iyy*L**2*scf - 180*E**2*Iyy**2)/(5*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)) + 2*Ay*G*r11*scf*(A*G*L**2*scf - 6*E*Iyy)/(L*(A*G*L**2*scf - 12*E*Iyy)))
            k += 1
            KC0v[k] += r31*(-Ay*E*r13/L + Az*E*r12/L) + r32*(-6*A**2*E*G**2*J*L**2*r12*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) + 6*A*E*G*Izz*r13*scf*(A*G*L**2*scf + 12*E*Izz)/(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2) + 12*Az*E*G*Izz*r11*scf/(L*(A*G*L**2*scf - 12*E*Izz))) + r33*(-6*A**2*E*G**2*J*L**2*r13*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) + 2*A*G*r12*scf*(A**2*G**2*L**4*scf**2 + 45*A*E*G*Iyy*L**2*scf - 180*E**2*Iyy**2)/(5*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)) + 2*Ay*G*r11*scf*(A*G*L**2*scf - 6*E*Iyy)/(L*(A*G*L**2*scf - 12*E*Iyy)))
            k += 1
            KC0v[k] += r11*(-6*Ay*E*G*Iyy*r12*scf/(A*G*L**2*scf - 12*E*Iyy) + 6*Az*E*G*Izz*r13*scf/(A*G*L**2*scf - 12*E*Izz) - G*r11*scf*(Iyy + Izz)/L) + r12*(6*Ay*E*G*Iyy*r11*scf/(A*G*L**2*scf - 12*E*Iyy) + 4*E*J*r13*(-A**2*G**2*L**4*scf**2 + 3*A*E*G*Iyy*L**2*scf + 3*A*E*G*Izz*L**2*scf - 36*E**2*Iyy*Izz)/(L*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + 4*r12*(2*A**3*G**3*L**6*scf**3 - 15*A**2*E*G**2*Iyy*L**4*scf**2 + 225*A*E**2*G*Iyy**2*L**2*scf + 540*E**3*Iyy**3)/(15*L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2))) + r13*(6*Az*E*G*Izz*r11*scf/(A*G*L**2*scf - 12*E*Izz) + 4*E*Izz*r13*(A**2*G**2*L**4*scf**2 + 3*A*E*G*Izz*L**2*scf + 36*E**2*Izz**2)/(L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)) + 4*E*J*r12*(-A**2*G**2*L**4*scf**2 + 3*A*E*G*Iyy*L**2*scf + 3*A*E*G*Izz*L**2*scf - 36*E**2*Iyy*Izz)/(L*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)))
            k += 1
            KC0v[k] += r21*(-6*Ay*E*G*Iyy*r12*scf/(A*G*L**2*scf - 12*E*Iyy) + 6*Az*E*G*Izz*r13*scf/(A*G*L**2*scf - 12*E*Izz) - G*r11*scf*(Iyy + Izz)/L) + r22*(6*Ay*E*G*Iyy*r11*scf/(A*G*L**2*scf - 12*E*Iyy) + 4*E*J*r13*(-A**2*G**2*L**4*scf**2 + 3*A*E*G*Iyy*L**2*scf + 3*A*E*G*Izz*L**2*scf - 36*E**2*Iyy*Izz)/(L*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + 4*r12*(2*A**3*G**3*L**6*scf**3 - 15*A**2*E*G**2*Iyy*L**4*scf**2 + 225*A*E**2*G*Iyy**2*L**2*scf + 540*E**3*Iyy**3)/(15*L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2))) + r23*(6*Az*E*G*Izz*r11*scf/(A*G*L**2*scf - 12*E*Izz) + 4*E*Izz*r13*(A**2*G**2*L**4*scf**2 + 3*A*E*G*Izz*L**2*scf + 36*E**2*Izz**2)/(L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)) + 4*E*J*r12*(-A**2*G**2*L**4*scf**2 + 3*A*E*G*Iyy*L**2*scf + 3*A*E*G*Izz*L**2*scf - 36*E**2*Iyy*Izz)/(L*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)))
            k += 1
            KC0v[k] += r31*(-6*Ay*E*G*Iyy*r12*scf/(A*G*L**2*scf - 12*E*Iyy) + 6*Az*E*G*Izz*r13*scf/(A*G*L**2*scf - 12*E*Izz) - G*r11*scf*(Iyy + Izz)/L) + r32*(6*Ay*E*G*Iyy*r11*scf/(A*G*L**2*scf - 12*E*Iyy) + 4*E*J*r13*(-A**2*G**2*L**4*scf**2 + 3*A*E*G*Iyy*L**2*scf + 3*A*E*G*Izz*L**2*scf - 36*E**2*Iyy*Izz)/(L*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + 4*r12*(2*A**3*G**3*L**6*scf**3 - 15*A**2*E*G**2*Iyy*L**4*scf**2 + 225*A*E**2*G*Iyy**2*L**2*scf + 540*E**3*Iyy**3)/(15*L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2))) + r33*(6*Az*E*G*Izz*r11*scf/(A*G*L**2*scf - 12*E*Izz) + 4*E*Izz*r13*(A**2*G**2*L**4*scf**2 + 3*A*E*G*Izz*L**2*scf + 36*E**2*Izz**2)/(L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)) + 4*E*J*r12*(-A**2*G**2*L**4*scf**2 + 3*A*E*G*Iyy*L**2*scf + 3*A*E*G*Izz*L**2*scf - 36*E**2*Iyy*Izz)/(L*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)))
            k += 1
            KC0v[k] += r11*(Ay*E*r13/L - Az*E*r12/L) + r12*(6*A**2*E*G**2*J*L**2*r12*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) - 6*A*E*G*Izz*r13*scf*(A*G*L**2*scf + 12*E*Izz)/(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2) - 12*Az*E*G*Izz*r11*scf/(L*(A*G*L**2*scf - 12*E*Izz))) + r13*(6*A**2*E*G**2*J*L**2*r13*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) + 2*A*G*r12*scf*(-A**2*G**2*L**4*scf**2 - 45*A*E*G*Iyy*L**2*scf + 180*E**2*Iyy**2)/(5*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)) + 2*Ay*G*r11*scf*(-A*G*L**2*scf + 6*E*Iyy)/(L*(A*G*L**2*scf - 12*E*Iyy)))
            k += 1
            KC0v[k] += r21*(Ay*E*r13/L - Az*E*r12/L) + r22*(6*A**2*E*G**2*J*L**2*r12*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) - 6*A*E*G*Izz*r13*scf*(A*G*L**2*scf + 12*E*Izz)/(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2) - 12*Az*E*G*Izz*r11*scf/(L*(A*G*L**2*scf - 12*E*Izz))) + r23*(6*A**2*E*G**2*J*L**2*r13*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) + 2*A*G*r12*scf*(-A**2*G**2*L**4*scf**2 - 45*A*E*G*Iyy*L**2*scf + 180*E**2*Iyy**2)/(5*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)) + 2*Ay*G*r11*scf*(-A*G*L**2*scf + 6*E*Iyy)/(L*(A*G*L**2*scf - 12*E*Iyy)))
            k += 1
            KC0v[k] += r31*(Ay*E*r13/L - Az*E*r12/L) + r32*(6*A**2*E*G**2*J*L**2*r12*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) - 6*A*E*G*Izz*r13*scf*(A*G*L**2*scf + 12*E*Izz)/(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2) - 12*Az*E*G*Izz*r11*scf/(L*(A*G*L**2*scf - 12*E*Izz))) + r33*(6*A**2*E*G**2*J*L**2*r13*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) + 2*A*G*r12*scf*(-A**2*G**2*L**4*scf**2 - 45*A*E*G*Iyy*L**2*scf + 180*E**2*Iyy**2)/(5*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)) + 2*Ay*G*r11*scf*(-A*G*L**2*scf + 6*E*Iyy)/(L*(A*G*L**2*scf - 12*E*Iyy)))
            k += 1
            KC0v[k] += r11*(6*Ay*E*G*Iyy*r12*scf/(A*G*L**2*scf - 12*E*Iyy) - 6*Az*E*G*Izz*r13*scf/(A*G*L**2*scf - 12*E*Izz) + G*r11*scf*(Iyy + Izz)/L) + r12*(6*Ay*E*G*Iyy*r11*scf/(A*G*L**2*scf - 12*E*Iyy) + 2*E*J*r13*(-A**2*G**2*L**4*scf**2 - 6*A*E*G*Iyy*L**2*scf - 6*A*E*G*Izz*L**2*scf + 72*E**2*Iyy*Izz)/(L*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + 2*r12*(-A**3*G**3*L**6*scf**3 + 75*A**2*E*G**2*Iyy*L**4*scf**2 + 90*A*E**2*G*Iyy**2*L**2*scf - 1080*E**3*Iyy**3)/(15*L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2))) + r13*(6*Az*E*G*Izz*r11*scf/(A*G*L**2*scf - 12*E*Izz) + 2*E*Izz*r13*(A**2*G**2*L**4*scf**2 + 30*A*E*G*Izz*L**2*scf - 72*E**2*Izz**2)/(L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)) + 2*E*J*r12*(-A**2*G**2*L**4*scf**2 - 6*A*E*G*Iyy*L**2*scf - 6*A*E*G*Izz*L**2*scf + 72*E**2*Iyy*Izz)/(L*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)))
            k += 1
            KC0v[k] += r21*(6*Ay*E*G*Iyy*r12*scf/(A*G*L**2*scf - 12*E*Iyy) - 6*Az*E*G*Izz*r13*scf/(A*G*L**2*scf - 12*E*Izz) + G*r11*scf*(Iyy + Izz)/L) + r22*(6*Ay*E*G*Iyy*r11*scf/(A*G*L**2*scf - 12*E*Iyy) + 2*E*J*r13*(-A**2*G**2*L**4*scf**2 - 6*A*E*G*Iyy*L**2*scf - 6*A*E*G*Izz*L**2*scf + 72*E**2*Iyy*Izz)/(L*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + 2*r12*(-A**3*G**3*L**6*scf**3 + 75*A**2*E*G**2*Iyy*L**4*scf**2 + 90*A*E**2*G*Iyy**2*L**2*scf - 1080*E**3*Iyy**3)/(15*L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2))) + r23*(6*Az*E*G*Izz*r11*scf/(A*G*L**2*scf - 12*E*Izz) + 2*E*Izz*r13*(A**2*G**2*L**4*scf**2 + 30*A*E*G*Izz*L**2*scf - 72*E**2*Izz**2)/(L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)) + 2*E*J*r12*(-A**2*G**2*L**4*scf**2 - 6*A*E*G*Iyy*L**2*scf - 6*A*E*G*Izz*L**2*scf + 72*E**2*Iyy*Izz)/(L*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)))
            k += 1
            KC0v[k] += r31*(6*Ay*E*G*Iyy*r12*scf/(A*G*L**2*scf - 12*E*Iyy) - 6*Az*E*G*Izz*r13*scf/(A*G*L**2*scf - 12*E*Izz) + G*r11*scf*(Iyy + Izz)/L) + r32*(6*Ay*E*G*Iyy*r11*scf/(A*G*L**2*scf - 12*E*Iyy) + 2*E*J*r13*(-A**2*G**2*L**4*scf**2 - 6*A*E*G*Iyy*L**2*scf - 6*A*E*G*Izz*L**2*scf + 72*E**2*Iyy*Izz)/(L*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + 2*r12*(-A**3*G**3*L**6*scf**3 + 75*A**2*E*G**2*Iyy*L**4*scf**2 + 90*A*E**2*G*Iyy**2*L**2*scf - 1080*E**3*Iyy**3)/(15*L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2))) + r33*(6*Az*E*G*Izz*r11*scf/(A*G*L**2*scf - 12*E*Izz) + 2*E*Izz*r13*(A**2*G**2*L**4*scf**2 + 30*A*E*G*Izz*L**2*scf - 72*E**2*Izz**2)/(L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)) + 2*E*J*r12*(-A**2*G**2*L**4*scf**2 - 6*A*E*G*Iyy*L**2*scf - 6*A*E*G*Izz*L**2*scf + 72*E**2*Iyy*Izz)/(L*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)))
            k += 1
            KC0v[k] += r11*(-Ay*E*r23/L + Az*E*r22/L) + r12*(-6*A**2*E*G**2*J*L**2*r22*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) + 6*A*E*G*Izz*r23*scf*(A*G*L**2*scf + 12*E*Izz)/(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2) + 12*Az*E*G*Izz*r21*scf/(L*(A*G*L**2*scf - 12*E*Izz))) + r13*(-6*A**2*E*G**2*J*L**2*r23*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) + 2*A*G*r22*scf*(A**2*G**2*L**4*scf**2 + 45*A*E*G*Iyy*L**2*scf - 180*E**2*Iyy**2)/(5*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)) + 2*Ay*G*r21*scf*(A*G*L**2*scf - 6*E*Iyy)/(L*(A*G*L**2*scf - 12*E*Iyy)))
            k += 1
            KC0v[k] += r21*(-Ay*E*r23/L + Az*E*r22/L) + r22*(-6*A**2*E*G**2*J*L**2*r22*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) + 6*A*E*G*Izz*r23*scf*(A*G*L**2*scf + 12*E*Izz)/(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2) + 12*Az*E*G*Izz*r21*scf/(L*(A*G*L**2*scf - 12*E*Izz))) + r23*(-6*A**2*E*G**2*J*L**2*r23*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) + 2*A*G*r22*scf*(A**2*G**2*L**4*scf**2 + 45*A*E*G*Iyy*L**2*scf - 180*E**2*Iyy**2)/(5*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)) + 2*Ay*G*r21*scf*(A*G*L**2*scf - 6*E*Iyy)/(L*(A*G*L**2*scf - 12*E*Iyy)))
            k += 1
            KC0v[k] += r31*(-Ay*E*r23/L + Az*E*r22/L) + r32*(-6*A**2*E*G**2*J*L**2*r22*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) + 6*A*E*G*Izz*r23*scf*(A*G*L**2*scf + 12*E*Izz)/(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2) + 12*Az*E*G*Izz*r21*scf/(L*(A*G*L**2*scf - 12*E*Izz))) + r33*(-6*A**2*E*G**2*J*L**2*r23*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) + 2*A*G*r22*scf*(A**2*G**2*L**4*scf**2 + 45*A*E*G*Iyy*L**2*scf - 180*E**2*Iyy**2)/(5*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)) + 2*Ay*G*r21*scf*(A*G*L**2*scf - 6*E*Iyy)/(L*(A*G*L**2*scf - 12*E*Iyy)))
            k += 1
            KC0v[k] += r11*(-6*Ay*E*G*Iyy*r22*scf/(A*G*L**2*scf - 12*E*Iyy) + 6*Az*E*G*Izz*r23*scf/(A*G*L**2*scf - 12*E*Izz) - G*r21*scf*(Iyy + Izz)/L) + r12*(6*Ay*E*G*Iyy*r21*scf/(A*G*L**2*scf - 12*E*Iyy) + 4*E*J*r23*(-A**2*G**2*L**4*scf**2 + 3*A*E*G*Iyy*L**2*scf + 3*A*E*G*Izz*L**2*scf - 36*E**2*Iyy*Izz)/(L*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + 4*r22*(2*A**3*G**3*L**6*scf**3 - 15*A**2*E*G**2*Iyy*L**4*scf**2 + 225*A*E**2*G*Iyy**2*L**2*scf + 540*E**3*Iyy**3)/(15*L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2))) + r13*(6*Az*E*G*Izz*r21*scf/(A*G*L**2*scf - 12*E*Izz) + 4*E*Izz*r23*(A**2*G**2*L**4*scf**2 + 3*A*E*G*Izz*L**2*scf + 36*E**2*Izz**2)/(L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)) + 4*E*J*r22*(-A**2*G**2*L**4*scf**2 + 3*A*E*G*Iyy*L**2*scf + 3*A*E*G*Izz*L**2*scf - 36*E**2*Iyy*Izz)/(L*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)))
            k += 1
            KC0v[k] += r21*(-6*Ay*E*G*Iyy*r22*scf/(A*G*L**2*scf - 12*E*Iyy) + 6*Az*E*G*Izz*r23*scf/(A*G*L**2*scf - 12*E*Izz) - G*r21*scf*(Iyy + Izz)/L) + r22*(6*Ay*E*G*Iyy*r21*scf/(A*G*L**2*scf - 12*E*Iyy) + 4*E*J*r23*(-A**2*G**2*L**4*scf**2 + 3*A*E*G*Iyy*L**2*scf + 3*A*E*G*Izz*L**2*scf - 36*E**2*Iyy*Izz)/(L*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + 4*r22*(2*A**3*G**3*L**6*scf**3 - 15*A**2*E*G**2*Iyy*L**4*scf**2 + 225*A*E**2*G*Iyy**2*L**2*scf + 540*E**3*Iyy**3)/(15*L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2))) + r23*(6*Az*E*G*Izz*r21*scf/(A*G*L**2*scf - 12*E*Izz) + 4*E*Izz*r23*(A**2*G**2*L**4*scf**2 + 3*A*E*G*Izz*L**2*scf + 36*E**2*Izz**2)/(L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)) + 4*E*J*r22*(-A**2*G**2*L**4*scf**2 + 3*A*E*G*Iyy*L**2*scf + 3*A*E*G*Izz*L**2*scf - 36*E**2*Iyy*Izz)/(L*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)))
            k += 1
            KC0v[k] += r31*(-6*Ay*E*G*Iyy*r22*scf/(A*G*L**2*scf - 12*E*Iyy) + 6*Az*E*G*Izz*r23*scf/(A*G*L**2*scf - 12*E*Izz) - G*r21*scf*(Iyy + Izz)/L) + r32*(6*Ay*E*G*Iyy*r21*scf/(A*G*L**2*scf - 12*E*Iyy) + 4*E*J*r23*(-A**2*G**2*L**4*scf**2 + 3*A*E*G*Iyy*L**2*scf + 3*A*E*G*Izz*L**2*scf - 36*E**2*Iyy*Izz)/(L*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + 4*r22*(2*A**3*G**3*L**6*scf**3 - 15*A**2*E*G**2*Iyy*L**4*scf**2 + 225*A*E**2*G*Iyy**2*L**2*scf + 540*E**3*Iyy**3)/(15*L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2))) + r33*(6*Az*E*G*Izz*r21*scf/(A*G*L**2*scf - 12*E*Izz) + 4*E*Izz*r23*(A**2*G**2*L**4*scf**2 + 3*A*E*G*Izz*L**2*scf + 36*E**2*Izz**2)/(L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)) + 4*E*J*r22*(-A**2*G**2*L**4*scf**2 + 3*A*E*G*Iyy*L**2*scf + 3*A*E*G*Izz*L**2*scf - 36*E**2*Iyy*Izz)/(L*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)))
            k += 1
            KC0v[k] += r11*(Ay*E*r23/L - Az*E*r22/L) + r12*(6*A**2*E*G**2*J*L**2*r22*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) - 6*A*E*G*Izz*r23*scf*(A*G*L**2*scf + 12*E*Izz)/(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2) - 12*Az*E*G*Izz*r21*scf/(L*(A*G*L**2*scf - 12*E*Izz))) + r13*(6*A**2*E*G**2*J*L**2*r23*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) + 2*A*G*r22*scf*(-A**2*G**2*L**4*scf**2 - 45*A*E*G*Iyy*L**2*scf + 180*E**2*Iyy**2)/(5*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)) + 2*Ay*G*r21*scf*(-A*G*L**2*scf + 6*E*Iyy)/(L*(A*G*L**2*scf - 12*E*Iyy)))
            k += 1
            KC0v[k] += r21*(Ay*E*r23/L - Az*E*r22/L) + r22*(6*A**2*E*G**2*J*L**2*r22*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) - 6*A*E*G*Izz*r23*scf*(A*G*L**2*scf + 12*E*Izz)/(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2) - 12*Az*E*G*Izz*r21*scf/(L*(A*G*L**2*scf - 12*E*Izz))) + r23*(6*A**2*E*G**2*J*L**2*r23*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) + 2*A*G*r22*scf*(-A**2*G**2*L**4*scf**2 - 45*A*E*G*Iyy*L**2*scf + 180*E**2*Iyy**2)/(5*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)) + 2*Ay*G*r21*scf*(-A*G*L**2*scf + 6*E*Iyy)/(L*(A*G*L**2*scf - 12*E*Iyy)))
            k += 1
            KC0v[k] += r31*(Ay*E*r23/L - Az*E*r22/L) + r32*(6*A**2*E*G**2*J*L**2*r22*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) - 6*A*E*G*Izz*r23*scf*(A*G*L**2*scf + 12*E*Izz)/(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2) - 12*Az*E*G*Izz*r21*scf/(L*(A*G*L**2*scf - 12*E*Izz))) + r33*(6*A**2*E*G**2*J*L**2*r23*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) + 2*A*G*r22*scf*(-A**2*G**2*L**4*scf**2 - 45*A*E*G*Iyy*L**2*scf + 180*E**2*Iyy**2)/(5*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)) + 2*Ay*G*r21*scf*(-A*G*L**2*scf + 6*E*Iyy)/(L*(A*G*L**2*scf - 12*E*Iyy)))
            k += 1
            KC0v[k] += r11*(6*Ay*E*G*Iyy*r22*scf/(A*G*L**2*scf - 12*E*Iyy) - 6*Az*E*G*Izz*r23*scf/(A*G*L**2*scf - 12*E*Izz) + G*r21*scf*(Iyy + Izz)/L) + r12*(6*Ay*E*G*Iyy*r21*scf/(A*G*L**2*scf - 12*E*Iyy) + 2*E*J*r23*(-A**2*G**2*L**4*scf**2 - 6*A*E*G*Iyy*L**2*scf - 6*A*E*G*Izz*L**2*scf + 72*E**2*Iyy*Izz)/(L*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + 2*r22*(-A**3*G**3*L**6*scf**3 + 75*A**2*E*G**2*Iyy*L**4*scf**2 + 90*A*E**2*G*Iyy**2*L**2*scf - 1080*E**3*Iyy**3)/(15*L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2))) + r13*(6*Az*E*G*Izz*r21*scf/(A*G*L**2*scf - 12*E*Izz) + 2*E*Izz*r23*(A**2*G**2*L**4*scf**2 + 30*A*E*G*Izz*L**2*scf - 72*E**2*Izz**2)/(L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)) + 2*E*J*r22*(-A**2*G**2*L**4*scf**2 - 6*A*E*G*Iyy*L**2*scf - 6*A*E*G*Izz*L**2*scf + 72*E**2*Iyy*Izz)/(L*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)))
            k += 1
            KC0v[k] += r21*(6*Ay*E*G*Iyy*r22*scf/(A*G*L**2*scf - 12*E*Iyy) - 6*Az*E*G*Izz*r23*scf/(A*G*L**2*scf - 12*E*Izz) + G*r21*scf*(Iyy + Izz)/L) + r22*(6*Ay*E*G*Iyy*r21*scf/(A*G*L**2*scf - 12*E*Iyy) + 2*E*J*r23*(-A**2*G**2*L**4*scf**2 - 6*A*E*G*Iyy*L**2*scf - 6*A*E*G*Izz*L**2*scf + 72*E**2*Iyy*Izz)/(L*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + 2*r22*(-A**3*G**3*L**6*scf**3 + 75*A**2*E*G**2*Iyy*L**4*scf**2 + 90*A*E**2*G*Iyy**2*L**2*scf - 1080*E**3*Iyy**3)/(15*L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2))) + r23*(6*Az*E*G*Izz*r21*scf/(A*G*L**2*scf - 12*E*Izz) + 2*E*Izz*r23*(A**2*G**2*L**4*scf**2 + 30*A*E*G*Izz*L**2*scf - 72*E**2*Izz**2)/(L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)) + 2*E*J*r22*(-A**2*G**2*L**4*scf**2 - 6*A*E*G*Iyy*L**2*scf - 6*A*E*G*Izz*L**2*scf + 72*E**2*Iyy*Izz)/(L*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)))
            k += 1
            KC0v[k] += r31*(6*Ay*E*G*Iyy*r22*scf/(A*G*L**2*scf - 12*E*Iyy) - 6*Az*E*G*Izz*r23*scf/(A*G*L**2*scf - 12*E*Izz) + G*r21*scf*(Iyy + Izz)/L) + r32*(6*Ay*E*G*Iyy*r21*scf/(A*G*L**2*scf - 12*E*Iyy) + 2*E*J*r23*(-A**2*G**2*L**4*scf**2 - 6*A*E*G*Iyy*L**2*scf - 6*A*E*G*Izz*L**2*scf + 72*E**2*Iyy*Izz)/(L*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + 2*r22*(-A**3*G**3*L**6*scf**3 + 75*A**2*E*G**2*Iyy*L**4*scf**2 + 90*A*E**2*G*Iyy**2*L**2*scf - 1080*E**3*Iyy**3)/(15*L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2))) + r33*(6*Az*E*G*Izz*r21*scf/(A*G*L**2*scf - 12*E*Izz) + 2*E*Izz*r23*(A**2*G**2*L**4*scf**2 + 30*A*E*G*Izz*L**2*scf - 72*E**2*Izz**2)/(L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)) + 2*E*J*r22*(-A**2*G**2*L**4*scf**2 - 6*A*E*G*Iyy*L**2*scf - 6*A*E*G*Izz*L**2*scf + 72*E**2*Iyy*Izz)/(L*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)))
            k += 1
            KC0v[k] += r11*(-Ay*E*r33/L + Az*E*r32/L) + r12*(-6*A**2*E*G**2*J*L**2*r32*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) + 6*A*E*G*Izz*r33*scf*(A*G*L**2*scf + 12*E*Izz)/(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2) + 12*Az*E*G*Izz*r31*scf/(L*(A*G*L**2*scf - 12*E*Izz))) + r13*(-6*A**2*E*G**2*J*L**2*r33*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) + 2*A*G*r32*scf*(A**2*G**2*L**4*scf**2 + 45*A*E*G*Iyy*L**2*scf - 180*E**2*Iyy**2)/(5*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)) + 2*Ay*G*r31*scf*(A*G*L**2*scf - 6*E*Iyy)/(L*(A*G*L**2*scf - 12*E*Iyy)))
            k += 1
            KC0v[k] += r21*(-Ay*E*r33/L + Az*E*r32/L) + r22*(-6*A**2*E*G**2*J*L**2*r32*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) + 6*A*E*G*Izz*r33*scf*(A*G*L**2*scf + 12*E*Izz)/(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2) + 12*Az*E*G*Izz*r31*scf/(L*(A*G*L**2*scf - 12*E*Izz))) + r23*(-6*A**2*E*G**2*J*L**2*r33*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) + 2*A*G*r32*scf*(A**2*G**2*L**4*scf**2 + 45*A*E*G*Iyy*L**2*scf - 180*E**2*Iyy**2)/(5*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)) + 2*Ay*G*r31*scf*(A*G*L**2*scf - 6*E*Iyy)/(L*(A*G*L**2*scf - 12*E*Iyy)))
            k += 1
            KC0v[k] += r31*(-Ay*E*r33/L + Az*E*r32/L) + r32*(-6*A**2*E*G**2*J*L**2*r32*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) + 6*A*E*G*Izz*r33*scf*(A*G*L**2*scf + 12*E*Izz)/(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2) + 12*Az*E*G*Izz*r31*scf/(L*(A*G*L**2*scf - 12*E*Izz))) + r33*(-6*A**2*E*G**2*J*L**2*r33*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) + 2*A*G*r32*scf*(A**2*G**2*L**4*scf**2 + 45*A*E*G*Iyy*L**2*scf - 180*E**2*Iyy**2)/(5*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)) + 2*Ay*G*r31*scf*(A*G*L**2*scf - 6*E*Iyy)/(L*(A*G*L**2*scf - 12*E*Iyy)))
            k += 1
            KC0v[k] += r11*(-6*Ay*E*G*Iyy*r32*scf/(A*G*L**2*scf - 12*E*Iyy) + 6*Az*E*G*Izz*r33*scf/(A*G*L**2*scf - 12*E*Izz) - G*r31*scf*(Iyy + Izz)/L) + r12*(6*Ay*E*G*Iyy*r31*scf/(A*G*L**2*scf - 12*E*Iyy) + 4*E*J*r33*(-A**2*G**2*L**4*scf**2 + 3*A*E*G*Iyy*L**2*scf + 3*A*E*G*Izz*L**2*scf - 36*E**2*Iyy*Izz)/(L*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + 4*r32*(2*A**3*G**3*L**6*scf**3 - 15*A**2*E*G**2*Iyy*L**4*scf**2 + 225*A*E**2*G*Iyy**2*L**2*scf + 540*E**3*Iyy**3)/(15*L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2))) + r13*(6*Az*E*G*Izz*r31*scf/(A*G*L**2*scf - 12*E*Izz) + 4*E*Izz*r33*(A**2*G**2*L**4*scf**2 + 3*A*E*G*Izz*L**2*scf + 36*E**2*Izz**2)/(L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)) + 4*E*J*r32*(-A**2*G**2*L**4*scf**2 + 3*A*E*G*Iyy*L**2*scf + 3*A*E*G*Izz*L**2*scf - 36*E**2*Iyy*Izz)/(L*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)))
            k += 1
            KC0v[k] += r21*(-6*Ay*E*G*Iyy*r32*scf/(A*G*L**2*scf - 12*E*Iyy) + 6*Az*E*G*Izz*r33*scf/(A*G*L**2*scf - 12*E*Izz) - G*r31*scf*(Iyy + Izz)/L) + r22*(6*Ay*E*G*Iyy*r31*scf/(A*G*L**2*scf - 12*E*Iyy) + 4*E*J*r33*(-A**2*G**2*L**4*scf**2 + 3*A*E*G*Iyy*L**2*scf + 3*A*E*G*Izz*L**2*scf - 36*E**2*Iyy*Izz)/(L*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + 4*r32*(2*A**3*G**3*L**6*scf**3 - 15*A**2*E*G**2*Iyy*L**4*scf**2 + 225*A*E**2*G*Iyy**2*L**2*scf + 540*E**3*Iyy**3)/(15*L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2))) + r23*(6*Az*E*G*Izz*r31*scf/(A*G*L**2*scf - 12*E*Izz) + 4*E*Izz*r33*(A**2*G**2*L**4*scf**2 + 3*A*E*G*Izz*L**2*scf + 36*E**2*Izz**2)/(L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)) + 4*E*J*r32*(-A**2*G**2*L**4*scf**2 + 3*A*E*G*Iyy*L**2*scf + 3*A*E*G*Izz*L**2*scf - 36*E**2*Iyy*Izz)/(L*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)))
            k += 1
            KC0v[k] += r31*(-6*Ay*E*G*Iyy*r32*scf/(A*G*L**2*scf - 12*E*Iyy) + 6*Az*E*G*Izz*r33*scf/(A*G*L**2*scf - 12*E*Izz) - G*r31*scf*(Iyy + Izz)/L) + r32*(6*Ay*E*G*Iyy*r31*scf/(A*G*L**2*scf - 12*E*Iyy) + 4*E*J*r33*(-A**2*G**2*L**4*scf**2 + 3*A*E*G*Iyy*L**2*scf + 3*A*E*G*Izz*L**2*scf - 36*E**2*Iyy*Izz)/(L*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + 4*r32*(2*A**3*G**3*L**6*scf**3 - 15*A**2*E*G**2*Iyy*L**4*scf**2 + 225*A*E**2*G*Iyy**2*L**2*scf + 540*E**3*Iyy**3)/(15*L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2))) + r33*(6*Az*E*G*Izz*r31*scf/(A*G*L**2*scf - 12*E*Izz) + 4*E*Izz*r33*(A**2*G**2*L**4*scf**2 + 3*A*E*G*Izz*L**2*scf + 36*E**2*Izz**2)/(L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)) + 4*E*J*r32*(-A**2*G**2*L**4*scf**2 + 3*A*E*G*Iyy*L**2*scf + 3*A*E*G*Izz*L**2*scf - 36*E**2*Iyy*Izz)/(L*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)))
            k += 1
            KC0v[k] += r11*(Ay*E*r33/L - Az*E*r32/L) + r12*(6*A**2*E*G**2*J*L**2*r32*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) - 6*A*E*G*Izz*r33*scf*(A*G*L**2*scf + 12*E*Izz)/(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2) - 12*Az*E*G*Izz*r31*scf/(L*(A*G*L**2*scf - 12*E*Izz))) + r13*(6*A**2*E*G**2*J*L**2*r33*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) + 2*A*G*r32*scf*(-A**2*G**2*L**4*scf**2 - 45*A*E*G*Iyy*L**2*scf + 180*E**2*Iyy**2)/(5*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)) + 2*Ay*G*r31*scf*(-A*G*L**2*scf + 6*E*Iyy)/(L*(A*G*L**2*scf - 12*E*Iyy)))
            k += 1
            KC0v[k] += r21*(Ay*E*r33/L - Az*E*r32/L) + r22*(6*A**2*E*G**2*J*L**2*r32*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) - 6*A*E*G*Izz*r33*scf*(A*G*L**2*scf + 12*E*Izz)/(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2) - 12*Az*E*G*Izz*r31*scf/(L*(A*G*L**2*scf - 12*E*Izz))) + r23*(6*A**2*E*G**2*J*L**2*r33*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) + 2*A*G*r32*scf*(-A**2*G**2*L**4*scf**2 - 45*A*E*G*Iyy*L**2*scf + 180*E**2*Iyy**2)/(5*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)) + 2*Ay*G*r31*scf*(-A*G*L**2*scf + 6*E*Iyy)/(L*(A*G*L**2*scf - 12*E*Iyy)))
            k += 1
            KC0v[k] += r31*(Ay*E*r33/L - Az*E*r32/L) + r32*(6*A**2*E*G**2*J*L**2*r32*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) - 6*A*E*G*Izz*r33*scf*(A*G*L**2*scf + 12*E*Izz)/(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2) - 12*Az*E*G*Izz*r31*scf/(L*(A*G*L**2*scf - 12*E*Izz))) + r33*(6*A**2*E*G**2*J*L**2*r33*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) + 2*A*G*r32*scf*(-A**2*G**2*L**4*scf**2 - 45*A*E*G*Iyy*L**2*scf + 180*E**2*Iyy**2)/(5*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)) + 2*Ay*G*r31*scf*(-A*G*L**2*scf + 6*E*Iyy)/(L*(A*G*L**2*scf - 12*E*Iyy)))
            k += 1
            KC0v[k] += r11*(6*Ay*E*G*Iyy*r32*scf/(A*G*L**2*scf - 12*E*Iyy) - 6*Az*E*G*Izz*r33*scf/(A*G*L**2*scf - 12*E*Izz) + G*r31*scf*(Iyy + Izz)/L) + r12*(6*Ay*E*G*Iyy*r31*scf/(A*G*L**2*scf - 12*E*Iyy) + 2*E*J*r33*(-A**2*G**2*L**4*scf**2 - 6*A*E*G*Iyy*L**2*scf - 6*A*E*G*Izz*L**2*scf + 72*E**2*Iyy*Izz)/(L*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + 2*r32*(-A**3*G**3*L**6*scf**3 + 75*A**2*E*G**2*Iyy*L**4*scf**2 + 90*A*E**2*G*Iyy**2*L**2*scf - 1080*E**3*Iyy**3)/(15*L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2))) + r13*(6*Az*E*G*Izz*r31*scf/(A*G*L**2*scf - 12*E*Izz) + 2*E*Izz*r33*(A**2*G**2*L**4*scf**2 + 30*A*E*G*Izz*L**2*scf - 72*E**2*Izz**2)/(L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)) + 2*E*J*r32*(-A**2*G**2*L**4*scf**2 - 6*A*E*G*Iyy*L**2*scf - 6*A*E*G*Izz*L**2*scf + 72*E**2*Iyy*Izz)/(L*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)))
            k += 1
            KC0v[k] += r21*(6*Ay*E*G*Iyy*r32*scf/(A*G*L**2*scf - 12*E*Iyy) - 6*Az*E*G*Izz*r33*scf/(A*G*L**2*scf - 12*E*Izz) + G*r31*scf*(Iyy + Izz)/L) + r22*(6*Ay*E*G*Iyy*r31*scf/(A*G*L**2*scf - 12*E*Iyy) + 2*E*J*r33*(-A**2*G**2*L**4*scf**2 - 6*A*E*G*Iyy*L**2*scf - 6*A*E*G*Izz*L**2*scf + 72*E**2*Iyy*Izz)/(L*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + 2*r32*(-A**3*G**3*L**6*scf**3 + 75*A**2*E*G**2*Iyy*L**4*scf**2 + 90*A*E**2*G*Iyy**2*L**2*scf - 1080*E**3*Iyy**3)/(15*L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2))) + r23*(6*Az*E*G*Izz*r31*scf/(A*G*L**2*scf - 12*E*Izz) + 2*E*Izz*r33*(A**2*G**2*L**4*scf**2 + 30*A*E*G*Izz*L**2*scf - 72*E**2*Izz**2)/(L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)) + 2*E*J*r32*(-A**2*G**2*L**4*scf**2 - 6*A*E*G*Iyy*L**2*scf - 6*A*E*G*Izz*L**2*scf + 72*E**2*Iyy*Izz)/(L*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)))
            k += 1
            KC0v[k] += r31*(6*Ay*E*G*Iyy*r32*scf/(A*G*L**2*scf - 12*E*Iyy) - 6*Az*E*G*Izz*r33*scf/(A*G*L**2*scf - 12*E*Izz) + G*r31*scf*(Iyy + Izz)/L) + r32*(6*Ay*E*G*Iyy*r31*scf/(A*G*L**2*scf - 12*E*Iyy) + 2*E*J*r33*(-A**2*G**2*L**4*scf**2 - 6*A*E*G*Iyy*L**2*scf - 6*A*E*G*Izz*L**2*scf + 72*E**2*Iyy*Izz)/(L*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + 2*r32*(-A**3*G**3*L**6*scf**3 + 75*A**2*E*G**2*Iyy*L**4*scf**2 + 90*A*E**2*G*Iyy**2*L**2*scf - 1080*E**3*Iyy**3)/(15*L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2))) + r33*(6*Az*E*G*Izz*r31*scf/(A*G*L**2*scf - 12*E*Izz) + 2*E*Izz*r33*(A**2*G**2*L**4*scf**2 + 30*A*E*G*Izz*L**2*scf - 72*E**2*Izz**2)/(L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)) + 2*E*J*r32*(-A**2*G**2*L**4*scf**2 - 6*A*E*G*Iyy*L**2*scf - 6*A*E*G*Izz*L**2*scf + 72*E**2*Iyy*Izz)/(L*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)))
            k += 1
            KC0v[k] += -A*E*r11**2/L + r12*(12*A**2*E*G**2*J*L*r13*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) - 12*A*E*G*Izz*r12*scf*(A*G*L**2*scf + 12*E*Izz)/(L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2))) + r13*(12*A**2*E*G**2*J*L*r12*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) + 12*A*G*r13*scf*(-2*A**2*G**2*L**4*scf**2 + 15*A*E*G*Iyy*L**2*scf - 60*E**2*Iyy**2)/(5*L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)))
            k += 1
            KC0v[k] += -A*E*r11*r21/L + r22*(12*A**2*E*G**2*J*L*r13*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) - 12*A*E*G*Izz*r12*scf*(A*G*L**2*scf + 12*E*Izz)/(L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2))) + r23*(12*A**2*E*G**2*J*L*r12*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) + 12*A*G*r13*scf*(-2*A**2*G**2*L**4*scf**2 + 15*A*E*G*Iyy*L**2*scf - 60*E**2*Iyy**2)/(5*L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)))
            k += 1
            KC0v[k] += -A*E*r11*r31/L + r32*(12*A**2*E*G**2*J*L*r13*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) - 12*A*E*G*Izz*r12*scf*(A*G*L**2*scf + 12*E*Izz)/(L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2))) + r33*(12*A**2*E*G**2*J*L*r12*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) + 12*A*G*r13*scf*(-2*A**2*G**2*L**4*scf**2 + 15*A*E*G*Iyy*L**2*scf - 60*E**2*Iyy**2)/(5*L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)))
            k += 1
            KC0v[k] += r11*(2*Ay*G*r13*scf*(A*G*L**2*scf - 6*E*Iyy)/(L*(A*G*L**2*scf - 12*E*Iyy)) - 12*Az*E*G*Izz*r12*scf/(L*(A*G*L**2*scf - 12*E*Izz))) + r12*(6*A**2*E*G**2*J*L**2*r12*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) + 2*A*G*r13*scf*(-A**2*G**2*L**4*scf**2 - 45*A*E*G*Iyy*L**2*scf + 180*E**2*Iyy**2)/(5*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)) - Az*E*r11/L) + r13*(6*A**2*E*G**2*J*L**2*r13*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) - 6*A*E*G*Izz*r12*scf*(A*G*L**2*scf + 12*E*Izz)/(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2) + Ay*E*r11/L)
            k += 1
            KC0v[k] += r21*(2*Ay*G*r13*scf*(A*G*L**2*scf - 6*E*Iyy)/(L*(A*G*L**2*scf - 12*E*Iyy)) - 12*Az*E*G*Izz*r12*scf/(L*(A*G*L**2*scf - 12*E*Izz))) + r22*(6*A**2*E*G**2*J*L**2*r12*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) + 2*A*G*r13*scf*(-A**2*G**2*L**4*scf**2 - 45*A*E*G*Iyy*L**2*scf + 180*E**2*Iyy**2)/(5*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)) - Az*E*r11/L) + r23*(6*A**2*E*G**2*J*L**2*r13*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) - 6*A*E*G*Izz*r12*scf*(A*G*L**2*scf + 12*E*Izz)/(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2) + Ay*E*r11/L)
            k += 1
            KC0v[k] += r31*(2*Ay*G*r13*scf*(A*G*L**2*scf - 6*E*Iyy)/(L*(A*G*L**2*scf - 12*E*Iyy)) - 12*Az*E*G*Izz*r12*scf/(L*(A*G*L**2*scf - 12*E*Izz))) + r32*(6*A**2*E*G**2*J*L**2*r12*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) + 2*A*G*r13*scf*(-A**2*G**2*L**4*scf**2 - 45*A*E*G*Iyy*L**2*scf + 180*E**2*Iyy**2)/(5*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)) - Az*E*r11/L) + r33*(6*A**2*E*G**2*J*L**2*r13*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) - 6*A*E*G*Izz*r12*scf*(A*G*L**2*scf + 12*E*Izz)/(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2) + Ay*E*r11/L)
            k += 1
            KC0v[k] += A*E*r11**2/L + r12*(-12*A**2*E*G**2*J*L*r13*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) + 12*A*E*G*Izz*r12*scf*(A*G*L**2*scf + 12*E*Izz)/(L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2))) + r13*(-12*A**2*E*G**2*J*L*r12*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) + 12*A*G*r13*scf*(2*A**2*G**2*L**4*scf**2 - 15*A*E*G*Iyy*L**2*scf + 60*E**2*Iyy**2)/(5*L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)))
            k += 1
            KC0v[k] += A*E*r11*r21/L + r22*(-12*A**2*E*G**2*J*L*r13*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) + 12*A*E*G*Izz*r12*scf*(A*G*L**2*scf + 12*E*Izz)/(L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2))) + r23*(-12*A**2*E*G**2*J*L*r12*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) + 12*A*G*r13*scf*(2*A**2*G**2*L**4*scf**2 - 15*A*E*G*Iyy*L**2*scf + 60*E**2*Iyy**2)/(5*L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)))
            k += 1
            KC0v[k] += A*E*r11*r31/L + r32*(-12*A**2*E*G**2*J*L*r13*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) + 12*A*E*G*Izz*r12*scf*(A*G*L**2*scf + 12*E*Izz)/(L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2))) + r33*(-12*A**2*E*G**2*J*L*r12*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) + 12*A*G*r13*scf*(2*A**2*G**2*L**4*scf**2 - 15*A*E*G*Iyy*L**2*scf + 60*E**2*Iyy**2)/(5*L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)))
            k += 1
            KC0v[k] += r11*(2*Ay*G*r13*scf*(-A*G*L**2*scf + 6*E*Iyy)/(L*(A*G*L**2*scf - 12*E*Iyy)) + 12*Az*E*G*Izz*r12*scf/(L*(A*G*L**2*scf - 12*E*Izz))) + r12*(6*A**2*E*G**2*J*L**2*r12*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) + 2*A*G*r13*scf*(-A**2*G**2*L**4*scf**2 - 45*A*E*G*Iyy*L**2*scf + 180*E**2*Iyy**2)/(5*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)) + Az*E*r11/L) + r13*(6*A**2*E*G**2*J*L**2*r13*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) - 6*A*E*G*Izz*r12*scf*(A*G*L**2*scf + 12*E*Izz)/(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2) - Ay*E*r11/L)
            k += 1
            KC0v[k] += r21*(2*Ay*G*r13*scf*(-A*G*L**2*scf + 6*E*Iyy)/(L*(A*G*L**2*scf - 12*E*Iyy)) + 12*Az*E*G*Izz*r12*scf/(L*(A*G*L**2*scf - 12*E*Izz))) + r22*(6*A**2*E*G**2*J*L**2*r12*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) + 2*A*G*r13*scf*(-A**2*G**2*L**4*scf**2 - 45*A*E*G*Iyy*L**2*scf + 180*E**2*Iyy**2)/(5*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)) + Az*E*r11/L) + r23*(6*A**2*E*G**2*J*L**2*r13*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) - 6*A*E*G*Izz*r12*scf*(A*G*L**2*scf + 12*E*Izz)/(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2) - Ay*E*r11/L)
            k += 1
            KC0v[k] += r31*(2*Ay*G*r13*scf*(-A*G*L**2*scf + 6*E*Iyy)/(L*(A*G*L**2*scf - 12*E*Iyy)) + 12*Az*E*G*Izz*r12*scf/(L*(A*G*L**2*scf - 12*E*Izz))) + r32*(6*A**2*E*G**2*J*L**2*r12*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) + 2*A*G*r13*scf*(-A**2*G**2*L**4*scf**2 - 45*A*E*G*Iyy*L**2*scf + 180*E**2*Iyy**2)/(5*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)) + Az*E*r11/L) + r33*(6*A**2*E*G**2*J*L**2*r13*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) - 6*A*E*G*Izz*r12*scf*(A*G*L**2*scf + 12*E*Izz)/(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2) - Ay*E*r11/L)
            k += 1
            KC0v[k] += -A*E*r11*r21/L + r12*(12*A**2*E*G**2*J*L*r23*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) - 12*A*E*G*Izz*r22*scf*(A*G*L**2*scf + 12*E*Izz)/(L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2))) + r13*(12*A**2*E*G**2*J*L*r22*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) + 12*A*G*r23*scf*(-2*A**2*G**2*L**4*scf**2 + 15*A*E*G*Iyy*L**2*scf - 60*E**2*Iyy**2)/(5*L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)))
            k += 1
            KC0v[k] += -A*E*r21**2/L + r22*(12*A**2*E*G**2*J*L*r23*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) - 12*A*E*G*Izz*r22*scf*(A*G*L**2*scf + 12*E*Izz)/(L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2))) + r23*(12*A**2*E*G**2*J*L*r22*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) + 12*A*G*r23*scf*(-2*A**2*G**2*L**4*scf**2 + 15*A*E*G*Iyy*L**2*scf - 60*E**2*Iyy**2)/(5*L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)))
            k += 1
            KC0v[k] += -A*E*r21*r31/L + r32*(12*A**2*E*G**2*J*L*r23*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) - 12*A*E*G*Izz*r22*scf*(A*G*L**2*scf + 12*E*Izz)/(L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2))) + r33*(12*A**2*E*G**2*J*L*r22*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) + 12*A*G*r23*scf*(-2*A**2*G**2*L**4*scf**2 + 15*A*E*G*Iyy*L**2*scf - 60*E**2*Iyy**2)/(5*L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)))
            k += 1
            KC0v[k] += r11*(2*Ay*G*r23*scf*(A*G*L**2*scf - 6*E*Iyy)/(L*(A*G*L**2*scf - 12*E*Iyy)) - 12*Az*E*G*Izz*r22*scf/(L*(A*G*L**2*scf - 12*E*Izz))) + r12*(6*A**2*E*G**2*J*L**2*r22*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) + 2*A*G*r23*scf*(-A**2*G**2*L**4*scf**2 - 45*A*E*G*Iyy*L**2*scf + 180*E**2*Iyy**2)/(5*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)) - Az*E*r21/L) + r13*(6*A**2*E*G**2*J*L**2*r23*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) - 6*A*E*G*Izz*r22*scf*(A*G*L**2*scf + 12*E*Izz)/(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2) + Ay*E*r21/L)
            k += 1
            KC0v[k] += r21*(2*Ay*G*r23*scf*(A*G*L**2*scf - 6*E*Iyy)/(L*(A*G*L**2*scf - 12*E*Iyy)) - 12*Az*E*G*Izz*r22*scf/(L*(A*G*L**2*scf - 12*E*Izz))) + r22*(6*A**2*E*G**2*J*L**2*r22*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) + 2*A*G*r23*scf*(-A**2*G**2*L**4*scf**2 - 45*A*E*G*Iyy*L**2*scf + 180*E**2*Iyy**2)/(5*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)) - Az*E*r21/L) + r23*(6*A**2*E*G**2*J*L**2*r23*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) - 6*A*E*G*Izz*r22*scf*(A*G*L**2*scf + 12*E*Izz)/(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2) + Ay*E*r21/L)
            k += 1
            KC0v[k] += r31*(2*Ay*G*r23*scf*(A*G*L**2*scf - 6*E*Iyy)/(L*(A*G*L**2*scf - 12*E*Iyy)) - 12*Az*E*G*Izz*r22*scf/(L*(A*G*L**2*scf - 12*E*Izz))) + r32*(6*A**2*E*G**2*J*L**2*r22*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) + 2*A*G*r23*scf*(-A**2*G**2*L**4*scf**2 - 45*A*E*G*Iyy*L**2*scf + 180*E**2*Iyy**2)/(5*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)) - Az*E*r21/L) + r33*(6*A**2*E*G**2*J*L**2*r23*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) - 6*A*E*G*Izz*r22*scf*(A*G*L**2*scf + 12*E*Izz)/(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2) + Ay*E*r21/L)
            k += 1
            KC0v[k] += A*E*r11*r21/L + r12*(-12*A**2*E*G**2*J*L*r23*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) + 12*A*E*G*Izz*r22*scf*(A*G*L**2*scf + 12*E*Izz)/(L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2))) + r13*(-12*A**2*E*G**2*J*L*r22*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) + 12*A*G*r23*scf*(2*A**2*G**2*L**4*scf**2 - 15*A*E*G*Iyy*L**2*scf + 60*E**2*Iyy**2)/(5*L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)))
            k += 1
            KC0v[k] += A*E*r21**2/L + r22*(-12*A**2*E*G**2*J*L*r23*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) + 12*A*E*G*Izz*r22*scf*(A*G*L**2*scf + 12*E*Izz)/(L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2))) + r23*(-12*A**2*E*G**2*J*L*r22*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) + 12*A*G*r23*scf*(2*A**2*G**2*L**4*scf**2 - 15*A*E*G*Iyy*L**2*scf + 60*E**2*Iyy**2)/(5*L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)))
            k += 1
            KC0v[k] += A*E*r21*r31/L + r32*(-12*A**2*E*G**2*J*L*r23*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) + 12*A*E*G*Izz*r22*scf*(A*G*L**2*scf + 12*E*Izz)/(L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2))) + r33*(-12*A**2*E*G**2*J*L*r22*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) + 12*A*G*r23*scf*(2*A**2*G**2*L**4*scf**2 - 15*A*E*G*Iyy*L**2*scf + 60*E**2*Iyy**2)/(5*L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)))
            k += 1
            KC0v[k] += r11*(2*Ay*G*r23*scf*(-A*G*L**2*scf + 6*E*Iyy)/(L*(A*G*L**2*scf - 12*E*Iyy)) + 12*Az*E*G*Izz*r22*scf/(L*(A*G*L**2*scf - 12*E*Izz))) + r12*(6*A**2*E*G**2*J*L**2*r22*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) + 2*A*G*r23*scf*(-A**2*G**2*L**4*scf**2 - 45*A*E*G*Iyy*L**2*scf + 180*E**2*Iyy**2)/(5*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)) + Az*E*r21/L) + r13*(6*A**2*E*G**2*J*L**2*r23*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) - 6*A*E*G*Izz*r22*scf*(A*G*L**2*scf + 12*E*Izz)/(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2) - Ay*E*r21/L)
            k += 1
            KC0v[k] += r21*(2*Ay*G*r23*scf*(-A*G*L**2*scf + 6*E*Iyy)/(L*(A*G*L**2*scf - 12*E*Iyy)) + 12*Az*E*G*Izz*r22*scf/(L*(A*G*L**2*scf - 12*E*Izz))) + r22*(6*A**2*E*G**2*J*L**2*r22*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) + 2*A*G*r23*scf*(-A**2*G**2*L**4*scf**2 - 45*A*E*G*Iyy*L**2*scf + 180*E**2*Iyy**2)/(5*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)) + Az*E*r21/L) + r23*(6*A**2*E*G**2*J*L**2*r23*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) - 6*A*E*G*Izz*r22*scf*(A*G*L**2*scf + 12*E*Izz)/(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2) - Ay*E*r21/L)
            k += 1
            KC0v[k] += r31*(2*Ay*G*r23*scf*(-A*G*L**2*scf + 6*E*Iyy)/(L*(A*G*L**2*scf - 12*E*Iyy)) + 12*Az*E*G*Izz*r22*scf/(L*(A*G*L**2*scf - 12*E*Izz))) + r32*(6*A**2*E*G**2*J*L**2*r22*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) + 2*A*G*r23*scf*(-A**2*G**2*L**4*scf**2 - 45*A*E*G*Iyy*L**2*scf + 180*E**2*Iyy**2)/(5*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)) + Az*E*r21/L) + r33*(6*A**2*E*G**2*J*L**2*r23*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) - 6*A*E*G*Izz*r22*scf*(A*G*L**2*scf + 12*E*Izz)/(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2) - Ay*E*r21/L)
            k += 1
            KC0v[k] += -A*E*r11*r31/L + r12*(12*A**2*E*G**2*J*L*r33*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) - 12*A*E*G*Izz*r32*scf*(A*G*L**2*scf + 12*E*Izz)/(L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2))) + r13*(12*A**2*E*G**2*J*L*r32*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) + 12*A*G*r33*scf*(-2*A**2*G**2*L**4*scf**2 + 15*A*E*G*Iyy*L**2*scf - 60*E**2*Iyy**2)/(5*L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)))
            k += 1
            KC0v[k] += -A*E*r21*r31/L + r22*(12*A**2*E*G**2*J*L*r33*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) - 12*A*E*G*Izz*r32*scf*(A*G*L**2*scf + 12*E*Izz)/(L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2))) + r23*(12*A**2*E*G**2*J*L*r32*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) + 12*A*G*r33*scf*(-2*A**2*G**2*L**4*scf**2 + 15*A*E*G*Iyy*L**2*scf - 60*E**2*Iyy**2)/(5*L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)))
            k += 1
            KC0v[k] += -A*E*r31**2/L + r32*(12*A**2*E*G**2*J*L*r33*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) - 12*A*E*G*Izz*r32*scf*(A*G*L**2*scf + 12*E*Izz)/(L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2))) + r33*(12*A**2*E*G**2*J*L*r32*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) + 12*A*G*r33*scf*(-2*A**2*G**2*L**4*scf**2 + 15*A*E*G*Iyy*L**2*scf - 60*E**2*Iyy**2)/(5*L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)))
            k += 1
            KC0v[k] += r11*(2*Ay*G*r33*scf*(A*G*L**2*scf - 6*E*Iyy)/(L*(A*G*L**2*scf - 12*E*Iyy)) - 12*Az*E*G*Izz*r32*scf/(L*(A*G*L**2*scf - 12*E*Izz))) + r12*(6*A**2*E*G**2*J*L**2*r32*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) + 2*A*G*r33*scf*(-A**2*G**2*L**4*scf**2 - 45*A*E*G*Iyy*L**2*scf + 180*E**2*Iyy**2)/(5*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)) - Az*E*r31/L) + r13*(6*A**2*E*G**2*J*L**2*r33*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) - 6*A*E*G*Izz*r32*scf*(A*G*L**2*scf + 12*E*Izz)/(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2) + Ay*E*r31/L)
            k += 1
            KC0v[k] += r21*(2*Ay*G*r33*scf*(A*G*L**2*scf - 6*E*Iyy)/(L*(A*G*L**2*scf - 12*E*Iyy)) - 12*Az*E*G*Izz*r32*scf/(L*(A*G*L**2*scf - 12*E*Izz))) + r22*(6*A**2*E*G**2*J*L**2*r32*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) + 2*A*G*r33*scf*(-A**2*G**2*L**4*scf**2 - 45*A*E*G*Iyy*L**2*scf + 180*E**2*Iyy**2)/(5*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)) - Az*E*r31/L) + r23*(6*A**2*E*G**2*J*L**2*r33*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) - 6*A*E*G*Izz*r32*scf*(A*G*L**2*scf + 12*E*Izz)/(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2) + Ay*E*r31/L)
            k += 1
            KC0v[k] += r31*(2*Ay*G*r33*scf*(A*G*L**2*scf - 6*E*Iyy)/(L*(A*G*L**2*scf - 12*E*Iyy)) - 12*Az*E*G*Izz*r32*scf/(L*(A*G*L**2*scf - 12*E*Izz))) + r32*(6*A**2*E*G**2*J*L**2*r32*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) + 2*A*G*r33*scf*(-A**2*G**2*L**4*scf**2 - 45*A*E*G*Iyy*L**2*scf + 180*E**2*Iyy**2)/(5*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)) - Az*E*r31/L) + r33*(6*A**2*E*G**2*J*L**2*r33*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) - 6*A*E*G*Izz*r32*scf*(A*G*L**2*scf + 12*E*Izz)/(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2) + Ay*E*r31/L)
            k += 1
            KC0v[k] += A*E*r11*r31/L + r12*(-12*A**2*E*G**2*J*L*r33*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) + 12*A*E*G*Izz*r32*scf*(A*G*L**2*scf + 12*E*Izz)/(L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2))) + r13*(-12*A**2*E*G**2*J*L*r32*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) + 12*A*G*r33*scf*(2*A**2*G**2*L**4*scf**2 - 15*A*E*G*Iyy*L**2*scf + 60*E**2*Iyy**2)/(5*L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)))
            k += 1
            KC0v[k] += A*E*r21*r31/L + r22*(-12*A**2*E*G**2*J*L*r33*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) + 12*A*E*G*Izz*r32*scf*(A*G*L**2*scf + 12*E*Izz)/(L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2))) + r23*(-12*A**2*E*G**2*J*L*r32*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) + 12*A*G*r33*scf*(2*A**2*G**2*L**4*scf**2 - 15*A*E*G*Iyy*L**2*scf + 60*E**2*Iyy**2)/(5*L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)))
            k += 1
            KC0v[k] += A*E*r31**2/L + r32*(-12*A**2*E*G**2*J*L*r33*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) + 12*A*E*G*Izz*r32*scf*(A*G*L**2*scf + 12*E*Izz)/(L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2))) + r33*(-12*A**2*E*G**2*J*L*r32*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) + 12*A*G*r33*scf*(2*A**2*G**2*L**4*scf**2 - 15*A*E*G*Iyy*L**2*scf + 60*E**2*Iyy**2)/(5*L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)))
            k += 1
            KC0v[k] += r11*(2*Ay*G*r33*scf*(-A*G*L**2*scf + 6*E*Iyy)/(L*(A*G*L**2*scf - 12*E*Iyy)) + 12*Az*E*G*Izz*r32*scf/(L*(A*G*L**2*scf - 12*E*Izz))) + r12*(6*A**2*E*G**2*J*L**2*r32*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) + 2*A*G*r33*scf*(-A**2*G**2*L**4*scf**2 - 45*A*E*G*Iyy*L**2*scf + 180*E**2*Iyy**2)/(5*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)) + Az*E*r31/L) + r13*(6*A**2*E*G**2*J*L**2*r33*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) - 6*A*E*G*Izz*r32*scf*(A*G*L**2*scf + 12*E*Izz)/(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2) - Ay*E*r31/L)
            k += 1
            KC0v[k] += r21*(2*Ay*G*r33*scf*(-A*G*L**2*scf + 6*E*Iyy)/(L*(A*G*L**2*scf - 12*E*Iyy)) + 12*Az*E*G*Izz*r32*scf/(L*(A*G*L**2*scf - 12*E*Izz))) + r22*(6*A**2*E*G**2*J*L**2*r32*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) + 2*A*G*r33*scf*(-A**2*G**2*L**4*scf**2 - 45*A*E*G*Iyy*L**2*scf + 180*E**2*Iyy**2)/(5*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)) + Az*E*r31/L) + r23*(6*A**2*E*G**2*J*L**2*r33*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) - 6*A*E*G*Izz*r32*scf*(A*G*L**2*scf + 12*E*Izz)/(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2) - Ay*E*r31/L)
            k += 1
            KC0v[k] += r31*(2*Ay*G*r33*scf*(-A*G*L**2*scf + 6*E*Iyy)/(L*(A*G*L**2*scf - 12*E*Iyy)) + 12*Az*E*G*Izz*r32*scf/(L*(A*G*L**2*scf - 12*E*Izz))) + r32*(6*A**2*E*G**2*J*L**2*r32*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) + 2*A*G*r33*scf*(-A**2*G**2*L**4*scf**2 - 45*A*E*G*Iyy*L**2*scf + 180*E**2*Iyy**2)/(5*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)) + Az*E*r31/L) + r33*(6*A**2*E*G**2*J*L**2*r33*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) - 6*A*E*G*Izz*r32*scf*(A*G*L**2*scf + 12*E*Izz)/(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2) - Ay*E*r31/L)
            k += 1
            KC0v[k] += r11*(Ay*E*r13/L - Az*E*r12/L) + r12*(-6*A**2*E*G**2*J*L**2*r12*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) + 6*A*E*G*Izz*r13*scf*(A*G*L**2*scf + 12*E*Izz)/(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2) - 12*Az*E*G*Izz*r11*scf/(L*(A*G*L**2*scf - 12*E*Izz))) + r13*(-6*A**2*E*G**2*J*L**2*r13*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) + 2*A*G*r12*scf*(A**2*G**2*L**4*scf**2 + 45*A*E*G*Iyy*L**2*scf - 180*E**2*Iyy**2)/(5*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)) + 2*Ay*G*r11*scf*(-A*G*L**2*scf + 6*E*Iyy)/(L*(A*G*L**2*scf - 12*E*Iyy)))
            k += 1
            KC0v[k] += r21*(Ay*E*r13/L - Az*E*r12/L) + r22*(-6*A**2*E*G**2*J*L**2*r12*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) + 6*A*E*G*Izz*r13*scf*(A*G*L**2*scf + 12*E*Izz)/(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2) - 12*Az*E*G*Izz*r11*scf/(L*(A*G*L**2*scf - 12*E*Izz))) + r23*(-6*A**2*E*G**2*J*L**2*r13*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) + 2*A*G*r12*scf*(A**2*G**2*L**4*scf**2 + 45*A*E*G*Iyy*L**2*scf - 180*E**2*Iyy**2)/(5*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)) + 2*Ay*G*r11*scf*(-A*G*L**2*scf + 6*E*Iyy)/(L*(A*G*L**2*scf - 12*E*Iyy)))
            k += 1
            KC0v[k] += r31*(Ay*E*r13/L - Az*E*r12/L) + r32*(-6*A**2*E*G**2*J*L**2*r12*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) + 6*A*E*G*Izz*r13*scf*(A*G*L**2*scf + 12*E*Izz)/(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2) - 12*Az*E*G*Izz*r11*scf/(L*(A*G*L**2*scf - 12*E*Izz))) + r33*(-6*A**2*E*G**2*J*L**2*r13*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) + 2*A*G*r12*scf*(A**2*G**2*L**4*scf**2 + 45*A*E*G*Iyy*L**2*scf - 180*E**2*Iyy**2)/(5*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)) + 2*Ay*G*r11*scf*(-A*G*L**2*scf + 6*E*Iyy)/(L*(A*G*L**2*scf - 12*E*Iyy)))
            k += 1
            KC0v[k] += r11*(-6*Ay*E*G*Iyy*r12*scf/(A*G*L**2*scf - 12*E*Iyy) + 6*Az*E*G*Izz*r13*scf/(A*G*L**2*scf - 12*E*Izz) + G*r11*scf*(Iyy + Izz)/L) + r12*(-6*Ay*E*G*Iyy*r11*scf/(A*G*L**2*scf - 12*E*Iyy) + 2*E*J*r13*(-A**2*G**2*L**4*scf**2 - 6*A*E*G*Iyy*L**2*scf - 6*A*E*G*Izz*L**2*scf + 72*E**2*Iyy*Izz)/(L*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + 2*r12*(-A**3*G**3*L**6*scf**3 + 75*A**2*E*G**2*Iyy*L**4*scf**2 + 90*A*E**2*G*Iyy**2*L**2*scf - 1080*E**3*Iyy**3)/(15*L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2))) + r13*(-6*Az*E*G*Izz*r11*scf/(A*G*L**2*scf - 12*E*Izz) + 2*E*Izz*r13*(A**2*G**2*L**4*scf**2 + 30*A*E*G*Izz*L**2*scf - 72*E**2*Izz**2)/(L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)) + 2*E*J*r12*(-A**2*G**2*L**4*scf**2 - 6*A*E*G*Iyy*L**2*scf - 6*A*E*G*Izz*L**2*scf + 72*E**2*Iyy*Izz)/(L*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)))
            k += 1
            KC0v[k] += r21*(-6*Ay*E*G*Iyy*r12*scf/(A*G*L**2*scf - 12*E*Iyy) + 6*Az*E*G*Izz*r13*scf/(A*G*L**2*scf - 12*E*Izz) + G*r11*scf*(Iyy + Izz)/L) + r22*(-6*Ay*E*G*Iyy*r11*scf/(A*G*L**2*scf - 12*E*Iyy) + 2*E*J*r13*(-A**2*G**2*L**4*scf**2 - 6*A*E*G*Iyy*L**2*scf - 6*A*E*G*Izz*L**2*scf + 72*E**2*Iyy*Izz)/(L*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + 2*r12*(-A**3*G**3*L**6*scf**3 + 75*A**2*E*G**2*Iyy*L**4*scf**2 + 90*A*E**2*G*Iyy**2*L**2*scf - 1080*E**3*Iyy**3)/(15*L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2))) + r23*(-6*Az*E*G*Izz*r11*scf/(A*G*L**2*scf - 12*E*Izz) + 2*E*Izz*r13*(A**2*G**2*L**4*scf**2 + 30*A*E*G*Izz*L**2*scf - 72*E**2*Izz**2)/(L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)) + 2*E*J*r12*(-A**2*G**2*L**4*scf**2 - 6*A*E*G*Iyy*L**2*scf - 6*A*E*G*Izz*L**2*scf + 72*E**2*Iyy*Izz)/(L*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)))
            k += 1
            KC0v[k] += r31*(-6*Ay*E*G*Iyy*r12*scf/(A*G*L**2*scf - 12*E*Iyy) + 6*Az*E*G*Izz*r13*scf/(A*G*L**2*scf - 12*E*Izz) + G*r11*scf*(Iyy + Izz)/L) + r32*(-6*Ay*E*G*Iyy*r11*scf/(A*G*L**2*scf - 12*E*Iyy) + 2*E*J*r13*(-A**2*G**2*L**4*scf**2 - 6*A*E*G*Iyy*L**2*scf - 6*A*E*G*Izz*L**2*scf + 72*E**2*Iyy*Izz)/(L*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + 2*r12*(-A**3*G**3*L**6*scf**3 + 75*A**2*E*G**2*Iyy*L**4*scf**2 + 90*A*E**2*G*Iyy**2*L**2*scf - 1080*E**3*Iyy**3)/(15*L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2))) + r33*(-6*Az*E*G*Izz*r11*scf/(A*G*L**2*scf - 12*E*Izz) + 2*E*Izz*r13*(A**2*G**2*L**4*scf**2 + 30*A*E*G*Izz*L**2*scf - 72*E**2*Izz**2)/(L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)) + 2*E*J*r12*(-A**2*G**2*L**4*scf**2 - 6*A*E*G*Iyy*L**2*scf - 6*A*E*G*Izz*L**2*scf + 72*E**2*Iyy*Izz)/(L*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)))
            k += 1
            KC0v[k] += r11*(-Ay*E*r13/L + Az*E*r12/L) + r12*(6*A**2*E*G**2*J*L**2*r12*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) - 6*A*E*G*Izz*r13*scf*(A*G*L**2*scf + 12*E*Izz)/(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2) + 12*Az*E*G*Izz*r11*scf/(L*(A*G*L**2*scf - 12*E*Izz))) + r13*(6*A**2*E*G**2*J*L**2*r13*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) + 2*A*G*r12*scf*(-A**2*G**2*L**4*scf**2 - 45*A*E*G*Iyy*L**2*scf + 180*E**2*Iyy**2)/(5*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)) + 2*Ay*G*r11*scf*(A*G*L**2*scf - 6*E*Iyy)/(L*(A*G*L**2*scf - 12*E*Iyy)))
            k += 1
            KC0v[k] += r21*(-Ay*E*r13/L + Az*E*r12/L) + r22*(6*A**2*E*G**2*J*L**2*r12*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) - 6*A*E*G*Izz*r13*scf*(A*G*L**2*scf + 12*E*Izz)/(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2) + 12*Az*E*G*Izz*r11*scf/(L*(A*G*L**2*scf - 12*E*Izz))) + r23*(6*A**2*E*G**2*J*L**2*r13*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) + 2*A*G*r12*scf*(-A**2*G**2*L**4*scf**2 - 45*A*E*G*Iyy*L**2*scf + 180*E**2*Iyy**2)/(5*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)) + 2*Ay*G*r11*scf*(A*G*L**2*scf - 6*E*Iyy)/(L*(A*G*L**2*scf - 12*E*Iyy)))
            k += 1
            KC0v[k] += r31*(-Ay*E*r13/L + Az*E*r12/L) + r32*(6*A**2*E*G**2*J*L**2*r12*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) - 6*A*E*G*Izz*r13*scf*(A*G*L**2*scf + 12*E*Izz)/(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2) + 12*Az*E*G*Izz*r11*scf/(L*(A*G*L**2*scf - 12*E*Izz))) + r33*(6*A**2*E*G**2*J*L**2*r13*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) + 2*A*G*r12*scf*(-A**2*G**2*L**4*scf**2 - 45*A*E*G*Iyy*L**2*scf + 180*E**2*Iyy**2)/(5*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)) + 2*Ay*G*r11*scf*(A*G*L**2*scf - 6*E*Iyy)/(L*(A*G*L**2*scf - 12*E*Iyy)))
            k += 1
            KC0v[k] += r11*(6*Ay*E*G*Iyy*r12*scf/(A*G*L**2*scf - 12*E*Iyy) - 6*Az*E*G*Izz*r13*scf/(A*G*L**2*scf - 12*E*Izz) - G*r11*scf*(Iyy + Izz)/L) + r12*(-6*Ay*E*G*Iyy*r11*scf/(A*G*L**2*scf - 12*E*Iyy) + 4*E*J*r13*(-A**2*G**2*L**4*scf**2 + 3*A*E*G*Iyy*L**2*scf + 3*A*E*G*Izz*L**2*scf - 36*E**2*Iyy*Izz)/(L*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + 4*r12*(2*A**3*G**3*L**6*scf**3 - 15*A**2*E*G**2*Iyy*L**4*scf**2 + 225*A*E**2*G*Iyy**2*L**2*scf + 540*E**3*Iyy**3)/(15*L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2))) + r13*(-6*Az*E*G*Izz*r11*scf/(A*G*L**2*scf - 12*E*Izz) + 4*E*Izz*r13*(A**2*G**2*L**4*scf**2 + 3*A*E*G*Izz*L**2*scf + 36*E**2*Izz**2)/(L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)) + 4*E*J*r12*(-A**2*G**2*L**4*scf**2 + 3*A*E*G*Iyy*L**2*scf + 3*A*E*G*Izz*L**2*scf - 36*E**2*Iyy*Izz)/(L*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)))
            k += 1
            KC0v[k] += r21*(6*Ay*E*G*Iyy*r12*scf/(A*G*L**2*scf - 12*E*Iyy) - 6*Az*E*G*Izz*r13*scf/(A*G*L**2*scf - 12*E*Izz) - G*r11*scf*(Iyy + Izz)/L) + r22*(-6*Ay*E*G*Iyy*r11*scf/(A*G*L**2*scf - 12*E*Iyy) + 4*E*J*r13*(-A**2*G**2*L**4*scf**2 + 3*A*E*G*Iyy*L**2*scf + 3*A*E*G*Izz*L**2*scf - 36*E**2*Iyy*Izz)/(L*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + 4*r12*(2*A**3*G**3*L**6*scf**3 - 15*A**2*E*G**2*Iyy*L**4*scf**2 + 225*A*E**2*G*Iyy**2*L**2*scf + 540*E**3*Iyy**3)/(15*L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2))) + r23*(-6*Az*E*G*Izz*r11*scf/(A*G*L**2*scf - 12*E*Izz) + 4*E*Izz*r13*(A**2*G**2*L**4*scf**2 + 3*A*E*G*Izz*L**2*scf + 36*E**2*Izz**2)/(L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)) + 4*E*J*r12*(-A**2*G**2*L**4*scf**2 + 3*A*E*G*Iyy*L**2*scf + 3*A*E*G*Izz*L**2*scf - 36*E**2*Iyy*Izz)/(L*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)))
            k += 1
            KC0v[k] += r31*(6*Ay*E*G*Iyy*r12*scf/(A*G*L**2*scf - 12*E*Iyy) - 6*Az*E*G*Izz*r13*scf/(A*G*L**2*scf - 12*E*Izz) - G*r11*scf*(Iyy + Izz)/L) + r32*(-6*Ay*E*G*Iyy*r11*scf/(A*G*L**2*scf - 12*E*Iyy) + 4*E*J*r13*(-A**2*G**2*L**4*scf**2 + 3*A*E*G*Iyy*L**2*scf + 3*A*E*G*Izz*L**2*scf - 36*E**2*Iyy*Izz)/(L*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + 4*r12*(2*A**3*G**3*L**6*scf**3 - 15*A**2*E*G**2*Iyy*L**4*scf**2 + 225*A*E**2*G*Iyy**2*L**2*scf + 540*E**3*Iyy**3)/(15*L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2))) + r33*(-6*Az*E*G*Izz*r11*scf/(A*G*L**2*scf - 12*E*Izz) + 4*E*Izz*r13*(A**2*G**2*L**4*scf**2 + 3*A*E*G*Izz*L**2*scf + 36*E**2*Izz**2)/(L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)) + 4*E*J*r12*(-A**2*G**2*L**4*scf**2 + 3*A*E*G*Iyy*L**2*scf + 3*A*E*G*Izz*L**2*scf - 36*E**2*Iyy*Izz)/(L*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)))
            k += 1
            KC0v[k] += r11*(Ay*E*r23/L - Az*E*r22/L) + r12*(-6*A**2*E*G**2*J*L**2*r22*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) + 6*A*E*G*Izz*r23*scf*(A*G*L**2*scf + 12*E*Izz)/(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2) - 12*Az*E*G*Izz*r21*scf/(L*(A*G*L**2*scf - 12*E*Izz))) + r13*(-6*A**2*E*G**2*J*L**2*r23*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) + 2*A*G*r22*scf*(A**2*G**2*L**4*scf**2 + 45*A*E*G*Iyy*L**2*scf - 180*E**2*Iyy**2)/(5*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)) + 2*Ay*G*r21*scf*(-A*G*L**2*scf + 6*E*Iyy)/(L*(A*G*L**2*scf - 12*E*Iyy)))
            k += 1
            KC0v[k] += r21*(Ay*E*r23/L - Az*E*r22/L) + r22*(-6*A**2*E*G**2*J*L**2*r22*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) + 6*A*E*G*Izz*r23*scf*(A*G*L**2*scf + 12*E*Izz)/(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2) - 12*Az*E*G*Izz*r21*scf/(L*(A*G*L**2*scf - 12*E*Izz))) + r23*(-6*A**2*E*G**2*J*L**2*r23*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) + 2*A*G*r22*scf*(A**2*G**2*L**4*scf**2 + 45*A*E*G*Iyy*L**2*scf - 180*E**2*Iyy**2)/(5*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)) + 2*Ay*G*r21*scf*(-A*G*L**2*scf + 6*E*Iyy)/(L*(A*G*L**2*scf - 12*E*Iyy)))
            k += 1
            KC0v[k] += r31*(Ay*E*r23/L - Az*E*r22/L) + r32*(-6*A**2*E*G**2*J*L**2*r22*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) + 6*A*E*G*Izz*r23*scf*(A*G*L**2*scf + 12*E*Izz)/(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2) - 12*Az*E*G*Izz*r21*scf/(L*(A*G*L**2*scf - 12*E*Izz))) + r33*(-6*A**2*E*G**2*J*L**2*r23*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) + 2*A*G*r22*scf*(A**2*G**2*L**4*scf**2 + 45*A*E*G*Iyy*L**2*scf - 180*E**2*Iyy**2)/(5*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)) + 2*Ay*G*r21*scf*(-A*G*L**2*scf + 6*E*Iyy)/(L*(A*G*L**2*scf - 12*E*Iyy)))
            k += 1
            KC0v[k] += r11*(-6*Ay*E*G*Iyy*r22*scf/(A*G*L**2*scf - 12*E*Iyy) + 6*Az*E*G*Izz*r23*scf/(A*G*L**2*scf - 12*E*Izz) + G*r21*scf*(Iyy + Izz)/L) + r12*(-6*Ay*E*G*Iyy*r21*scf/(A*G*L**2*scf - 12*E*Iyy) + 2*E*J*r23*(-A**2*G**2*L**4*scf**2 - 6*A*E*G*Iyy*L**2*scf - 6*A*E*G*Izz*L**2*scf + 72*E**2*Iyy*Izz)/(L*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + 2*r22*(-A**3*G**3*L**6*scf**3 + 75*A**2*E*G**2*Iyy*L**4*scf**2 + 90*A*E**2*G*Iyy**2*L**2*scf - 1080*E**3*Iyy**3)/(15*L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2))) + r13*(-6*Az*E*G*Izz*r21*scf/(A*G*L**2*scf - 12*E*Izz) + 2*E*Izz*r23*(A**2*G**2*L**4*scf**2 + 30*A*E*G*Izz*L**2*scf - 72*E**2*Izz**2)/(L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)) + 2*E*J*r22*(-A**2*G**2*L**4*scf**2 - 6*A*E*G*Iyy*L**2*scf - 6*A*E*G*Izz*L**2*scf + 72*E**2*Iyy*Izz)/(L*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)))
            k += 1
            KC0v[k] += r21*(-6*Ay*E*G*Iyy*r22*scf/(A*G*L**2*scf - 12*E*Iyy) + 6*Az*E*G*Izz*r23*scf/(A*G*L**2*scf - 12*E*Izz) + G*r21*scf*(Iyy + Izz)/L) + r22*(-6*Ay*E*G*Iyy*r21*scf/(A*G*L**2*scf - 12*E*Iyy) + 2*E*J*r23*(-A**2*G**2*L**4*scf**2 - 6*A*E*G*Iyy*L**2*scf - 6*A*E*G*Izz*L**2*scf + 72*E**2*Iyy*Izz)/(L*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + 2*r22*(-A**3*G**3*L**6*scf**3 + 75*A**2*E*G**2*Iyy*L**4*scf**2 + 90*A*E**2*G*Iyy**2*L**2*scf - 1080*E**3*Iyy**3)/(15*L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2))) + r23*(-6*Az*E*G*Izz*r21*scf/(A*G*L**2*scf - 12*E*Izz) + 2*E*Izz*r23*(A**2*G**2*L**4*scf**2 + 30*A*E*G*Izz*L**2*scf - 72*E**2*Izz**2)/(L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)) + 2*E*J*r22*(-A**2*G**2*L**4*scf**2 - 6*A*E*G*Iyy*L**2*scf - 6*A*E*G*Izz*L**2*scf + 72*E**2*Iyy*Izz)/(L*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)))
            k += 1
            KC0v[k] += r31*(-6*Ay*E*G*Iyy*r22*scf/(A*G*L**2*scf - 12*E*Iyy) + 6*Az*E*G*Izz*r23*scf/(A*G*L**2*scf - 12*E*Izz) + G*r21*scf*(Iyy + Izz)/L) + r32*(-6*Ay*E*G*Iyy*r21*scf/(A*G*L**2*scf - 12*E*Iyy) + 2*E*J*r23*(-A**2*G**2*L**4*scf**2 - 6*A*E*G*Iyy*L**2*scf - 6*A*E*G*Izz*L**2*scf + 72*E**2*Iyy*Izz)/(L*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + 2*r22*(-A**3*G**3*L**6*scf**3 + 75*A**2*E*G**2*Iyy*L**4*scf**2 + 90*A*E**2*G*Iyy**2*L**2*scf - 1080*E**3*Iyy**3)/(15*L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2))) + r33*(-6*Az*E*G*Izz*r21*scf/(A*G*L**2*scf - 12*E*Izz) + 2*E*Izz*r23*(A**2*G**2*L**4*scf**2 + 30*A*E*G*Izz*L**2*scf - 72*E**2*Izz**2)/(L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)) + 2*E*J*r22*(-A**2*G**2*L**4*scf**2 - 6*A*E*G*Iyy*L**2*scf - 6*A*E*G*Izz*L**2*scf + 72*E**2*Iyy*Izz)/(L*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)))
            k += 1
            KC0v[k] += r11*(-Ay*E*r23/L + Az*E*r22/L) + r12*(6*A**2*E*G**2*J*L**2*r22*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) - 6*A*E*G*Izz*r23*scf*(A*G*L**2*scf + 12*E*Izz)/(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2) + 12*Az*E*G*Izz*r21*scf/(L*(A*G*L**2*scf - 12*E*Izz))) + r13*(6*A**2*E*G**2*J*L**2*r23*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) + 2*A*G*r22*scf*(-A**2*G**2*L**4*scf**2 - 45*A*E*G*Iyy*L**2*scf + 180*E**2*Iyy**2)/(5*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)) + 2*Ay*G*r21*scf*(A*G*L**2*scf - 6*E*Iyy)/(L*(A*G*L**2*scf - 12*E*Iyy)))
            k += 1
            KC0v[k] += r21*(-Ay*E*r23/L + Az*E*r22/L) + r22*(6*A**2*E*G**2*J*L**2*r22*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) - 6*A*E*G*Izz*r23*scf*(A*G*L**2*scf + 12*E*Izz)/(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2) + 12*Az*E*G*Izz*r21*scf/(L*(A*G*L**2*scf - 12*E*Izz))) + r23*(6*A**2*E*G**2*J*L**2*r23*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) + 2*A*G*r22*scf*(-A**2*G**2*L**4*scf**2 - 45*A*E*G*Iyy*L**2*scf + 180*E**2*Iyy**2)/(5*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)) + 2*Ay*G*r21*scf*(A*G*L**2*scf - 6*E*Iyy)/(L*(A*G*L**2*scf - 12*E*Iyy)))
            k += 1
            KC0v[k] += r31*(-Ay*E*r23/L + Az*E*r22/L) + r32*(6*A**2*E*G**2*J*L**2*r22*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) - 6*A*E*G*Izz*r23*scf*(A*G*L**2*scf + 12*E*Izz)/(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2) + 12*Az*E*G*Izz*r21*scf/(L*(A*G*L**2*scf - 12*E*Izz))) + r33*(6*A**2*E*G**2*J*L**2*r23*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) + 2*A*G*r22*scf*(-A**2*G**2*L**4*scf**2 - 45*A*E*G*Iyy*L**2*scf + 180*E**2*Iyy**2)/(5*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)) + 2*Ay*G*r21*scf*(A*G*L**2*scf - 6*E*Iyy)/(L*(A*G*L**2*scf - 12*E*Iyy)))
            k += 1
            KC0v[k] += r11*(6*Ay*E*G*Iyy*r22*scf/(A*G*L**2*scf - 12*E*Iyy) - 6*Az*E*G*Izz*r23*scf/(A*G*L**2*scf - 12*E*Izz) - G*r21*scf*(Iyy + Izz)/L) + r12*(-6*Ay*E*G*Iyy*r21*scf/(A*G*L**2*scf - 12*E*Iyy) + 4*E*J*r23*(-A**2*G**2*L**4*scf**2 + 3*A*E*G*Iyy*L**2*scf + 3*A*E*G*Izz*L**2*scf - 36*E**2*Iyy*Izz)/(L*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + 4*r22*(2*A**3*G**3*L**6*scf**3 - 15*A**2*E*G**2*Iyy*L**4*scf**2 + 225*A*E**2*G*Iyy**2*L**2*scf + 540*E**3*Iyy**3)/(15*L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2))) + r13*(-6*Az*E*G*Izz*r21*scf/(A*G*L**2*scf - 12*E*Izz) + 4*E*Izz*r23*(A**2*G**2*L**4*scf**2 + 3*A*E*G*Izz*L**2*scf + 36*E**2*Izz**2)/(L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)) + 4*E*J*r22*(-A**2*G**2*L**4*scf**2 + 3*A*E*G*Iyy*L**2*scf + 3*A*E*G*Izz*L**2*scf - 36*E**2*Iyy*Izz)/(L*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)))
            k += 1
            KC0v[k] += r21*(6*Ay*E*G*Iyy*r22*scf/(A*G*L**2*scf - 12*E*Iyy) - 6*Az*E*G*Izz*r23*scf/(A*G*L**2*scf - 12*E*Izz) - G*r21*scf*(Iyy + Izz)/L) + r22*(-6*Ay*E*G*Iyy*r21*scf/(A*G*L**2*scf - 12*E*Iyy) + 4*E*J*r23*(-A**2*G**2*L**4*scf**2 + 3*A*E*G*Iyy*L**2*scf + 3*A*E*G*Izz*L**2*scf - 36*E**2*Iyy*Izz)/(L*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + 4*r22*(2*A**3*G**3*L**6*scf**3 - 15*A**2*E*G**2*Iyy*L**4*scf**2 + 225*A*E**2*G*Iyy**2*L**2*scf + 540*E**3*Iyy**3)/(15*L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2))) + r23*(-6*Az*E*G*Izz*r21*scf/(A*G*L**2*scf - 12*E*Izz) + 4*E*Izz*r23*(A**2*G**2*L**4*scf**2 + 3*A*E*G*Izz*L**2*scf + 36*E**2*Izz**2)/(L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)) + 4*E*J*r22*(-A**2*G**2*L**4*scf**2 + 3*A*E*G*Iyy*L**2*scf + 3*A*E*G*Izz*L**2*scf - 36*E**2*Iyy*Izz)/(L*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)))
            k += 1
            KC0v[k] += r31*(6*Ay*E*G*Iyy*r22*scf/(A*G*L**2*scf - 12*E*Iyy) - 6*Az*E*G*Izz*r23*scf/(A*G*L**2*scf - 12*E*Izz) - G*r21*scf*(Iyy + Izz)/L) + r32*(-6*Ay*E*G*Iyy*r21*scf/(A*G*L**2*scf - 12*E*Iyy) + 4*E*J*r23*(-A**2*G**2*L**4*scf**2 + 3*A*E*G*Iyy*L**2*scf + 3*A*E*G*Izz*L**2*scf - 36*E**2*Iyy*Izz)/(L*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + 4*r22*(2*A**3*G**3*L**6*scf**3 - 15*A**2*E*G**2*Iyy*L**4*scf**2 + 225*A*E**2*G*Iyy**2*L**2*scf + 540*E**3*Iyy**3)/(15*L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2))) + r33*(-6*Az*E*G*Izz*r21*scf/(A*G*L**2*scf - 12*E*Izz) + 4*E*Izz*r23*(A**2*G**2*L**4*scf**2 + 3*A*E*G*Izz*L**2*scf + 36*E**2*Izz**2)/(L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)) + 4*E*J*r22*(-A**2*G**2*L**4*scf**2 + 3*A*E*G*Iyy*L**2*scf + 3*A*E*G*Izz*L**2*scf - 36*E**2*Iyy*Izz)/(L*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)))
            k += 1
            KC0v[k] += r11*(Ay*E*r33/L - Az*E*r32/L) + r12*(-6*A**2*E*G**2*J*L**2*r32*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) + 6*A*E*G*Izz*r33*scf*(A*G*L**2*scf + 12*E*Izz)/(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2) - 12*Az*E*G*Izz*r31*scf/(L*(A*G*L**2*scf - 12*E*Izz))) + r13*(-6*A**2*E*G**2*J*L**2*r33*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) + 2*A*G*r32*scf*(A**2*G**2*L**4*scf**2 + 45*A*E*G*Iyy*L**2*scf - 180*E**2*Iyy**2)/(5*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)) + 2*Ay*G*r31*scf*(-A*G*L**2*scf + 6*E*Iyy)/(L*(A*G*L**2*scf - 12*E*Iyy)))
            k += 1
            KC0v[k] += r21*(Ay*E*r33/L - Az*E*r32/L) + r22*(-6*A**2*E*G**2*J*L**2*r32*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) + 6*A*E*G*Izz*r33*scf*(A*G*L**2*scf + 12*E*Izz)/(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2) - 12*Az*E*G*Izz*r31*scf/(L*(A*G*L**2*scf - 12*E*Izz))) + r23*(-6*A**2*E*G**2*J*L**2*r33*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) + 2*A*G*r32*scf*(A**2*G**2*L**4*scf**2 + 45*A*E*G*Iyy*L**2*scf - 180*E**2*Iyy**2)/(5*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)) + 2*Ay*G*r31*scf*(-A*G*L**2*scf + 6*E*Iyy)/(L*(A*G*L**2*scf - 12*E*Iyy)))
            k += 1
            KC0v[k] += r31*(Ay*E*r33/L - Az*E*r32/L) + r32*(-6*A**2*E*G**2*J*L**2*r32*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) + 6*A*E*G*Izz*r33*scf*(A*G*L**2*scf + 12*E*Izz)/(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2) - 12*Az*E*G*Izz*r31*scf/(L*(A*G*L**2*scf - 12*E*Izz))) + r33*(-6*A**2*E*G**2*J*L**2*r33*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) + 2*A*G*r32*scf*(A**2*G**2*L**4*scf**2 + 45*A*E*G*Iyy*L**2*scf - 180*E**2*Iyy**2)/(5*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)) + 2*Ay*G*r31*scf*(-A*G*L**2*scf + 6*E*Iyy)/(L*(A*G*L**2*scf - 12*E*Iyy)))
            k += 1
            KC0v[k] += r11*(-6*Ay*E*G*Iyy*r32*scf/(A*G*L**2*scf - 12*E*Iyy) + 6*Az*E*G*Izz*r33*scf/(A*G*L**2*scf - 12*E*Izz) + G*r31*scf*(Iyy + Izz)/L) + r12*(-6*Ay*E*G*Iyy*r31*scf/(A*G*L**2*scf - 12*E*Iyy) + 2*E*J*r33*(-A**2*G**2*L**4*scf**2 - 6*A*E*G*Iyy*L**2*scf - 6*A*E*G*Izz*L**2*scf + 72*E**2*Iyy*Izz)/(L*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + 2*r32*(-A**3*G**3*L**6*scf**3 + 75*A**2*E*G**2*Iyy*L**4*scf**2 + 90*A*E**2*G*Iyy**2*L**2*scf - 1080*E**3*Iyy**3)/(15*L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2))) + r13*(-6*Az*E*G*Izz*r31*scf/(A*G*L**2*scf - 12*E*Izz) + 2*E*Izz*r33*(A**2*G**2*L**4*scf**2 + 30*A*E*G*Izz*L**2*scf - 72*E**2*Izz**2)/(L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)) + 2*E*J*r32*(-A**2*G**2*L**4*scf**2 - 6*A*E*G*Iyy*L**2*scf - 6*A*E*G*Izz*L**2*scf + 72*E**2*Iyy*Izz)/(L*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)))
            k += 1
            KC0v[k] += r21*(-6*Ay*E*G*Iyy*r32*scf/(A*G*L**2*scf - 12*E*Iyy) + 6*Az*E*G*Izz*r33*scf/(A*G*L**2*scf - 12*E*Izz) + G*r31*scf*(Iyy + Izz)/L) + r22*(-6*Ay*E*G*Iyy*r31*scf/(A*G*L**2*scf - 12*E*Iyy) + 2*E*J*r33*(-A**2*G**2*L**4*scf**2 - 6*A*E*G*Iyy*L**2*scf - 6*A*E*G*Izz*L**2*scf + 72*E**2*Iyy*Izz)/(L*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + 2*r32*(-A**3*G**3*L**6*scf**3 + 75*A**2*E*G**2*Iyy*L**4*scf**2 + 90*A*E**2*G*Iyy**2*L**2*scf - 1080*E**3*Iyy**3)/(15*L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2))) + r23*(-6*Az*E*G*Izz*r31*scf/(A*G*L**2*scf - 12*E*Izz) + 2*E*Izz*r33*(A**2*G**2*L**4*scf**2 + 30*A*E*G*Izz*L**2*scf - 72*E**2*Izz**2)/(L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)) + 2*E*J*r32*(-A**2*G**2*L**4*scf**2 - 6*A*E*G*Iyy*L**2*scf - 6*A*E*G*Izz*L**2*scf + 72*E**2*Iyy*Izz)/(L*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)))
            k += 1
            KC0v[k] += r31*(-6*Ay*E*G*Iyy*r32*scf/(A*G*L**2*scf - 12*E*Iyy) + 6*Az*E*G*Izz*r33*scf/(A*G*L**2*scf - 12*E*Izz) + G*r31*scf*(Iyy + Izz)/L) + r32*(-6*Ay*E*G*Iyy*r31*scf/(A*G*L**2*scf - 12*E*Iyy) + 2*E*J*r33*(-A**2*G**2*L**4*scf**2 - 6*A*E*G*Iyy*L**2*scf - 6*A*E*G*Izz*L**2*scf + 72*E**2*Iyy*Izz)/(L*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + 2*r32*(-A**3*G**3*L**6*scf**3 + 75*A**2*E*G**2*Iyy*L**4*scf**2 + 90*A*E**2*G*Iyy**2*L**2*scf - 1080*E**3*Iyy**3)/(15*L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2))) + r33*(-6*Az*E*G*Izz*r31*scf/(A*G*L**2*scf - 12*E*Izz) + 2*E*Izz*r33*(A**2*G**2*L**4*scf**2 + 30*A*E*G*Izz*L**2*scf - 72*E**2*Izz**2)/(L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)) + 2*E*J*r32*(-A**2*G**2*L**4*scf**2 - 6*A*E*G*Iyy*L**2*scf - 6*A*E*G*Izz*L**2*scf + 72*E**2*Iyy*Izz)/(L*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)))
            k += 1
            KC0v[k] += r11*(-Ay*E*r33/L + Az*E*r32/L) + r12*(6*A**2*E*G**2*J*L**2*r32*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) - 6*A*E*G*Izz*r33*scf*(A*G*L**2*scf + 12*E*Izz)/(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2) + 12*Az*E*G*Izz*r31*scf/(L*(A*G*L**2*scf - 12*E*Izz))) + r13*(6*A**2*E*G**2*J*L**2*r33*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) + 2*A*G*r32*scf*(-A**2*G**2*L**4*scf**2 - 45*A*E*G*Iyy*L**2*scf + 180*E**2*Iyy**2)/(5*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)) + 2*Ay*G*r31*scf*(A*G*L**2*scf - 6*E*Iyy)/(L*(A*G*L**2*scf - 12*E*Iyy)))
            k += 1
            KC0v[k] += r21*(-Ay*E*r33/L + Az*E*r32/L) + r22*(6*A**2*E*G**2*J*L**2*r32*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) - 6*A*E*G*Izz*r33*scf*(A*G*L**2*scf + 12*E*Izz)/(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2) + 12*Az*E*G*Izz*r31*scf/(L*(A*G*L**2*scf - 12*E*Izz))) + r23*(6*A**2*E*G**2*J*L**2*r33*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) + 2*A*G*r32*scf*(-A**2*G**2*L**4*scf**2 - 45*A*E*G*Iyy*L**2*scf + 180*E**2*Iyy**2)/(5*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)) + 2*Ay*G*r31*scf*(A*G*L**2*scf - 6*E*Iyy)/(L*(A*G*L**2*scf - 12*E*Iyy)))
            k += 1
            KC0v[k] += r31*(-Ay*E*r33/L + Az*E*r32/L) + r32*(6*A**2*E*G**2*J*L**2*r32*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) - 6*A*E*G*Izz*r33*scf*(A*G*L**2*scf + 12*E*Izz)/(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2) + 12*Az*E*G*Izz*r31*scf/(L*(A*G*L**2*scf - 12*E*Izz))) + r33*(6*A**2*E*G**2*J*L**2*r33*scf**2/(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz) + 2*A*G*r32*scf*(-A**2*G**2*L**4*scf**2 - 45*A*E*G*Iyy*L**2*scf + 180*E**2*Iyy**2)/(5*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)) + 2*Ay*G*r31*scf*(A*G*L**2*scf - 6*E*Iyy)/(L*(A*G*L**2*scf - 12*E*Iyy)))
            k += 1
            KC0v[k] += r11*(6*Ay*E*G*Iyy*r32*scf/(A*G*L**2*scf - 12*E*Iyy) - 6*Az*E*G*Izz*r33*scf/(A*G*L**2*scf - 12*E*Izz) - G*r31*scf*(Iyy + Izz)/L) + r12*(-6*Ay*E*G*Iyy*r31*scf/(A*G*L**2*scf - 12*E*Iyy) + 4*E*J*r33*(-A**2*G**2*L**4*scf**2 + 3*A*E*G*Iyy*L**2*scf + 3*A*E*G*Izz*L**2*scf - 36*E**2*Iyy*Izz)/(L*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + 4*r32*(2*A**3*G**3*L**6*scf**3 - 15*A**2*E*G**2*Iyy*L**4*scf**2 + 225*A*E**2*G*Iyy**2*L**2*scf + 540*E**3*Iyy**3)/(15*L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2))) + r13*(-6*Az*E*G*Izz*r31*scf/(A*G*L**2*scf - 12*E*Izz) + 4*E*Izz*r33*(A**2*G**2*L**4*scf**2 + 3*A*E*G*Izz*L**2*scf + 36*E**2*Izz**2)/(L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)) + 4*E*J*r32*(-A**2*G**2*L**4*scf**2 + 3*A*E*G*Iyy*L**2*scf + 3*A*E*G*Izz*L**2*scf - 36*E**2*Iyy*Izz)/(L*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)))
            k += 1
            KC0v[k] += r21*(6*Ay*E*G*Iyy*r32*scf/(A*G*L**2*scf - 12*E*Iyy) - 6*Az*E*G*Izz*r33*scf/(A*G*L**2*scf - 12*E*Izz) - G*r31*scf*(Iyy + Izz)/L) + r22*(-6*Ay*E*G*Iyy*r31*scf/(A*G*L**2*scf - 12*E*Iyy) + 4*E*J*r33*(-A**2*G**2*L**4*scf**2 + 3*A*E*G*Iyy*L**2*scf + 3*A*E*G*Izz*L**2*scf - 36*E**2*Iyy*Izz)/(L*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + 4*r32*(2*A**3*G**3*L**6*scf**3 - 15*A**2*E*G**2*Iyy*L**4*scf**2 + 225*A*E**2*G*Iyy**2*L**2*scf + 540*E**3*Iyy**3)/(15*L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2))) + r23*(-6*Az*E*G*Izz*r31*scf/(A*G*L**2*scf - 12*E*Izz) + 4*E*Izz*r33*(A**2*G**2*L**4*scf**2 + 3*A*E*G*Izz*L**2*scf + 36*E**2*Izz**2)/(L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)) + 4*E*J*r32*(-A**2*G**2*L**4*scf**2 + 3*A*E*G*Iyy*L**2*scf + 3*A*E*G*Izz*L**2*scf - 36*E**2*Iyy*Izz)/(L*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)))
            k += 1
            KC0v[k] += r31*(6*Ay*E*G*Iyy*r32*scf/(A*G*L**2*scf - 12*E*Iyy) - 6*Az*E*G*Izz*r33*scf/(A*G*L**2*scf - 12*E*Izz) - G*r31*scf*(Iyy + Izz)/L) + r32*(-6*Ay*E*G*Iyy*r31*scf/(A*G*L**2*scf - 12*E*Iyy) + 4*E*J*r33*(-A**2*G**2*L**4*scf**2 + 3*A*E*G*Iyy*L**2*scf + 3*A*E*G*Izz*L**2*scf - 36*E**2*Iyy*Izz)/(L*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + 4*r32*(2*A**3*G**3*L**6*scf**3 - 15*A**2*E*G**2*Iyy*L**4*scf**2 + 225*A*E**2*G*Iyy**2*L**2*scf + 540*E**3*Iyy**3)/(15*L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2))) + r33*(-6*Az*E*G*Izz*r31*scf/(A*G*L**2*scf - 12*E*Izz) + 4*E*Izz*r33*(A**2*G**2*L**4*scf**2 + 3*A*E*G*Izz*L**2*scf + 36*E**2*Izz**2)/(L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)) + 4*E*J*r32*(-A**2*G**2*L**4*scf**2 + 3*A*E*G*Iyy*L**2*scf + 3*A*E*G*Izz*L**2*scf - 36*E**2*Iyy*Izz)/(L*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)))


    cpdef void update_KG(BeamC self,
            np.ndarray[cINT, ndim=1] KGr,
            np.ndarray[cINT, ndim=1] KGc,
            np.ndarray[cDOUBLE, ndim=1] KGv,
            BeamProp prop,
            int update_KGv_only=0
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
        cdef cINT c1, c2, k
        cdef double L, A, E, G, scf, Iyy, Izz, N
        cdef double r11, r12, r13, r21, r22, r23, r31, r32, r33
        cdef double sina, sinb, sing

        with nogil:
            L = self.length
            A = prop.A
            E = prop.E
            G = prop.G
            scf = prop.scf
            Iyy = prop.Iyy
            Izz = prop.Izz

            #Local to global transformation
            sina = (1. - self.cosa**2)**0.5
            sinb = (1. - self.cosb**2)**0.5
            sing = (1. - self.cosg**2)**0.5
            r11 = self.cosb*self.cosg
            r12 = -self.cosa*sing + self.cosg*sina*sinb
            r13 = self.cosa*self.cosg*sinb + sina*sing
            r21 = self.cosb*sing
            r22 = self.cosa*self.cosg + sina*sinb*sing
            r23 = self.cosa*sinb*sing - self.cosg*sina
            r31 = -sinb
            r32 = self.cosb*sina
            r33 = self.cosa*self.cosb

            ue = &self._p.ue[0]

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
            KGv[k] += r12*(6*N*r12*(A**2*G**2*L**4*scf**2 - 20*A*E*G*Izz*L**2*scf + 120*E**2*Izz**2)/(5*L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)) + 6*N*r13*(A**2*G**2*L**4*scf**2 - 10*A*E*G*Iyy*L**2*scf - 10*A*E*G*Izz*L**2*scf + 120*E**2*Iyy*Izz)/(5*L*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz))) + r13*(6*N*r12*(A**2*G**2*L**4*scf**2 - 10*A*E*G*Iyy*L**2*scf - 10*A*E*G*Izz*L**2*scf + 120*E**2*Iyy*Izz)/(5*L*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + 6*N*r13*(A**2*G**2*L**4*scf**2 - 20*A*E*G*Iyy*L**2*scf + 120*E**2*Iyy**2)/(5*L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)))
            k += 1
            KGv[k] += r22*(6*N*r12*(A**2*G**2*L**4*scf**2 - 20*A*E*G*Izz*L**2*scf + 120*E**2*Izz**2)/(5*L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)) + 6*N*r13*(A**2*G**2*L**4*scf**2 - 10*A*E*G*Iyy*L**2*scf - 10*A*E*G*Izz*L**2*scf + 120*E**2*Iyy*Izz)/(5*L*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz))) + r23*(6*N*r12*(A**2*G**2*L**4*scf**2 - 10*A*E*G*Iyy*L**2*scf - 10*A*E*G*Izz*L**2*scf + 120*E**2*Iyy*Izz)/(5*L*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + 6*N*r13*(A**2*G**2*L**4*scf**2 - 20*A*E*G*Iyy*L**2*scf + 120*E**2*Iyy**2)/(5*L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)))
            k += 1
            KGv[k] += r32*(6*N*r12*(A**2*G**2*L**4*scf**2 - 20*A*E*G*Izz*L**2*scf + 120*E**2*Izz**2)/(5*L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)) + 6*N*r13*(A**2*G**2*L**4*scf**2 - 10*A*E*G*Iyy*L**2*scf - 10*A*E*G*Izz*L**2*scf + 120*E**2*Iyy*Izz)/(5*L*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz))) + r33*(6*N*r12*(A**2*G**2*L**4*scf**2 - 10*A*E*G*Iyy*L**2*scf - 10*A*E*G*Izz*L**2*scf + 120*E**2*Iyy*Izz)/(5*L*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + 6*N*r13*(A**2*G**2*L**4*scf**2 - 20*A*E*G*Iyy*L**2*scf + 120*E**2*Iyy**2)/(5*L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)))
            k += 1
            KGv[k] += r12*(A**2*G**2*L**4*N*r12*scf**2/(10*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + A**2*G**2*L**4*N*r13*scf**2/(10*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2))) + r13*(A**2*G**2*L**4*N*r12*scf**2/(10*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)) + A**2*G**2*L**4*N*r13*scf**2/(10*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)))
            k += 1
            KGv[k] += r22*(A**2*G**2*L**4*N*r12*scf**2/(10*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + A**2*G**2*L**4*N*r13*scf**2/(10*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2))) + r23*(A**2*G**2*L**4*N*r12*scf**2/(10*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)) + A**2*G**2*L**4*N*r13*scf**2/(10*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)))
            k += 1
            KGv[k] += r32*(A**2*G**2*L**4*N*r12*scf**2/(10*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + A**2*G**2*L**4*N*r13*scf**2/(10*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2))) + r33*(A**2*G**2*L**4*N*r12*scf**2/(10*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)) + A**2*G**2*L**4*N*r13*scf**2/(10*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)))
            k += 1
            KGv[k] += r12*(-6*N*r12*(A**2*G**2*L**4*scf**2 - 20*A*E*G*Izz*L**2*scf + 120*E**2*Izz**2)/(5*L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)) - 6*N*r13*(A**2*G**2*L**4*scf**2 - 10*A*E*G*Iyy*L**2*scf - 10*A*E*G*Izz*L**2*scf + 120*E**2*Iyy*Izz)/(5*L*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz))) + r13*(-6*N*r12*(A**2*G**2*L**4*scf**2 - 10*A*E*G*Iyy*L**2*scf - 10*A*E*G*Izz*L**2*scf + 120*E**2*Iyy*Izz)/(5*L*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) - 6*N*r13*(A**2*G**2*L**4*scf**2 - 20*A*E*G*Iyy*L**2*scf + 120*E**2*Iyy**2)/(5*L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)))
            k += 1
            KGv[k] += r22*(-6*N*r12*(A**2*G**2*L**4*scf**2 - 20*A*E*G*Izz*L**2*scf + 120*E**2*Izz**2)/(5*L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)) - 6*N*r13*(A**2*G**2*L**4*scf**2 - 10*A*E*G*Iyy*L**2*scf - 10*A*E*G*Izz*L**2*scf + 120*E**2*Iyy*Izz)/(5*L*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz))) + r23*(-6*N*r12*(A**2*G**2*L**4*scf**2 - 10*A*E*G*Iyy*L**2*scf - 10*A*E*G*Izz*L**2*scf + 120*E**2*Iyy*Izz)/(5*L*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) - 6*N*r13*(A**2*G**2*L**4*scf**2 - 20*A*E*G*Iyy*L**2*scf + 120*E**2*Iyy**2)/(5*L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)))
            k += 1
            KGv[k] += r32*(-6*N*r12*(A**2*G**2*L**4*scf**2 - 20*A*E*G*Izz*L**2*scf + 120*E**2*Izz**2)/(5*L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)) - 6*N*r13*(A**2*G**2*L**4*scf**2 - 10*A*E*G*Iyy*L**2*scf - 10*A*E*G*Izz*L**2*scf + 120*E**2*Iyy*Izz)/(5*L*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz))) + r33*(-6*N*r12*(A**2*G**2*L**4*scf**2 - 10*A*E*G*Iyy*L**2*scf - 10*A*E*G*Izz*L**2*scf + 120*E**2*Iyy*Izz)/(5*L*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) - 6*N*r13*(A**2*G**2*L**4*scf**2 - 20*A*E*G*Iyy*L**2*scf + 120*E**2*Iyy**2)/(5*L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)))
            k += 1
            KGv[k] += r12*(A**2*G**2*L**4*N*r12*scf**2/(10*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + A**2*G**2*L**4*N*r13*scf**2/(10*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2))) + r13*(A**2*G**2*L**4*N*r12*scf**2/(10*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)) + A**2*G**2*L**4*N*r13*scf**2/(10*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)))
            k += 1
            KGv[k] += r22*(A**2*G**2*L**4*N*r12*scf**2/(10*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + A**2*G**2*L**4*N*r13*scf**2/(10*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2))) + r23*(A**2*G**2*L**4*N*r12*scf**2/(10*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)) + A**2*G**2*L**4*N*r13*scf**2/(10*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)))
            k += 1
            KGv[k] += r32*(A**2*G**2*L**4*N*r12*scf**2/(10*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + A**2*G**2*L**4*N*r13*scf**2/(10*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2))) + r33*(A**2*G**2*L**4*N*r12*scf**2/(10*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)) + A**2*G**2*L**4*N*r13*scf**2/(10*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)))
            k += 1
            KGv[k] += r12*(6*N*r22*(A**2*G**2*L**4*scf**2 - 20*A*E*G*Izz*L**2*scf + 120*E**2*Izz**2)/(5*L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)) + 6*N*r23*(A**2*G**2*L**4*scf**2 - 10*A*E*G*Iyy*L**2*scf - 10*A*E*G*Izz*L**2*scf + 120*E**2*Iyy*Izz)/(5*L*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz))) + r13*(6*N*r22*(A**2*G**2*L**4*scf**2 - 10*A*E*G*Iyy*L**2*scf - 10*A*E*G*Izz*L**2*scf + 120*E**2*Iyy*Izz)/(5*L*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + 6*N*r23*(A**2*G**2*L**4*scf**2 - 20*A*E*G*Iyy*L**2*scf + 120*E**2*Iyy**2)/(5*L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)))
            k += 1
            KGv[k] += r22*(6*N*r22*(A**2*G**2*L**4*scf**2 - 20*A*E*G*Izz*L**2*scf + 120*E**2*Izz**2)/(5*L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)) + 6*N*r23*(A**2*G**2*L**4*scf**2 - 10*A*E*G*Iyy*L**2*scf - 10*A*E*G*Izz*L**2*scf + 120*E**2*Iyy*Izz)/(5*L*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz))) + r23*(6*N*r22*(A**2*G**2*L**4*scf**2 - 10*A*E*G*Iyy*L**2*scf - 10*A*E*G*Izz*L**2*scf + 120*E**2*Iyy*Izz)/(5*L*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + 6*N*r23*(A**2*G**2*L**4*scf**2 - 20*A*E*G*Iyy*L**2*scf + 120*E**2*Iyy**2)/(5*L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)))
            k += 1
            KGv[k] += r32*(6*N*r22*(A**2*G**2*L**4*scf**2 - 20*A*E*G*Izz*L**2*scf + 120*E**2*Izz**2)/(5*L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)) + 6*N*r23*(A**2*G**2*L**4*scf**2 - 10*A*E*G*Iyy*L**2*scf - 10*A*E*G*Izz*L**2*scf + 120*E**2*Iyy*Izz)/(5*L*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz))) + r33*(6*N*r22*(A**2*G**2*L**4*scf**2 - 10*A*E*G*Iyy*L**2*scf - 10*A*E*G*Izz*L**2*scf + 120*E**2*Iyy*Izz)/(5*L*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + 6*N*r23*(A**2*G**2*L**4*scf**2 - 20*A*E*G*Iyy*L**2*scf + 120*E**2*Iyy**2)/(5*L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)))
            k += 1
            KGv[k] += r12*(A**2*G**2*L**4*N*r22*scf**2/(10*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + A**2*G**2*L**4*N*r23*scf**2/(10*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2))) + r13*(A**2*G**2*L**4*N*r22*scf**2/(10*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)) + A**2*G**2*L**4*N*r23*scf**2/(10*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)))
            k += 1
            KGv[k] += r22*(A**2*G**2*L**4*N*r22*scf**2/(10*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + A**2*G**2*L**4*N*r23*scf**2/(10*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2))) + r23*(A**2*G**2*L**4*N*r22*scf**2/(10*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)) + A**2*G**2*L**4*N*r23*scf**2/(10*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)))
            k += 1
            KGv[k] += r32*(A**2*G**2*L**4*N*r22*scf**2/(10*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + A**2*G**2*L**4*N*r23*scf**2/(10*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2))) + r33*(A**2*G**2*L**4*N*r22*scf**2/(10*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)) + A**2*G**2*L**4*N*r23*scf**2/(10*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)))
            k += 1
            KGv[k] += r12*(-6*N*r22*(A**2*G**2*L**4*scf**2 - 20*A*E*G*Izz*L**2*scf + 120*E**2*Izz**2)/(5*L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)) - 6*N*r23*(A**2*G**2*L**4*scf**2 - 10*A*E*G*Iyy*L**2*scf - 10*A*E*G*Izz*L**2*scf + 120*E**2*Iyy*Izz)/(5*L*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz))) + r13*(-6*N*r22*(A**2*G**2*L**4*scf**2 - 10*A*E*G*Iyy*L**2*scf - 10*A*E*G*Izz*L**2*scf + 120*E**2*Iyy*Izz)/(5*L*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) - 6*N*r23*(A**2*G**2*L**4*scf**2 - 20*A*E*G*Iyy*L**2*scf + 120*E**2*Iyy**2)/(5*L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)))
            k += 1
            KGv[k] += r22*(-6*N*r22*(A**2*G**2*L**4*scf**2 - 20*A*E*G*Izz*L**2*scf + 120*E**2*Izz**2)/(5*L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)) - 6*N*r23*(A**2*G**2*L**4*scf**2 - 10*A*E*G*Iyy*L**2*scf - 10*A*E*G*Izz*L**2*scf + 120*E**2*Iyy*Izz)/(5*L*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz))) + r23*(-6*N*r22*(A**2*G**2*L**4*scf**2 - 10*A*E*G*Iyy*L**2*scf - 10*A*E*G*Izz*L**2*scf + 120*E**2*Iyy*Izz)/(5*L*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) - 6*N*r23*(A**2*G**2*L**4*scf**2 - 20*A*E*G*Iyy*L**2*scf + 120*E**2*Iyy**2)/(5*L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)))
            k += 1
            KGv[k] += r32*(-6*N*r22*(A**2*G**2*L**4*scf**2 - 20*A*E*G*Izz*L**2*scf + 120*E**2*Izz**2)/(5*L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)) - 6*N*r23*(A**2*G**2*L**4*scf**2 - 10*A*E*G*Iyy*L**2*scf - 10*A*E*G*Izz*L**2*scf + 120*E**2*Iyy*Izz)/(5*L*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz))) + r33*(-6*N*r22*(A**2*G**2*L**4*scf**2 - 10*A*E*G*Iyy*L**2*scf - 10*A*E*G*Izz*L**2*scf + 120*E**2*Iyy*Izz)/(5*L*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) - 6*N*r23*(A**2*G**2*L**4*scf**2 - 20*A*E*G*Iyy*L**2*scf + 120*E**2*Iyy**2)/(5*L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)))
            k += 1
            KGv[k] += r12*(A**2*G**2*L**4*N*r22*scf**2/(10*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + A**2*G**2*L**4*N*r23*scf**2/(10*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2))) + r13*(A**2*G**2*L**4*N*r22*scf**2/(10*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)) + A**2*G**2*L**4*N*r23*scf**2/(10*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)))
            k += 1
            KGv[k] += r22*(A**2*G**2*L**4*N*r22*scf**2/(10*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + A**2*G**2*L**4*N*r23*scf**2/(10*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2))) + r23*(A**2*G**2*L**4*N*r22*scf**2/(10*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)) + A**2*G**2*L**4*N*r23*scf**2/(10*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)))
            k += 1
            KGv[k] += r32*(A**2*G**2*L**4*N*r22*scf**2/(10*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + A**2*G**2*L**4*N*r23*scf**2/(10*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2))) + r33*(A**2*G**2*L**4*N*r22*scf**2/(10*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)) + A**2*G**2*L**4*N*r23*scf**2/(10*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)))
            k += 1
            KGv[k] += r12*(6*N*r32*(A**2*G**2*L**4*scf**2 - 20*A*E*G*Izz*L**2*scf + 120*E**2*Izz**2)/(5*L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)) + 6*N*r33*(A**2*G**2*L**4*scf**2 - 10*A*E*G*Iyy*L**2*scf - 10*A*E*G*Izz*L**2*scf + 120*E**2*Iyy*Izz)/(5*L*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz))) + r13*(6*N*r32*(A**2*G**2*L**4*scf**2 - 10*A*E*G*Iyy*L**2*scf - 10*A*E*G*Izz*L**2*scf + 120*E**2*Iyy*Izz)/(5*L*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + 6*N*r33*(A**2*G**2*L**4*scf**2 - 20*A*E*G*Iyy*L**2*scf + 120*E**2*Iyy**2)/(5*L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)))
            k += 1
            KGv[k] += r22*(6*N*r32*(A**2*G**2*L**4*scf**2 - 20*A*E*G*Izz*L**2*scf + 120*E**2*Izz**2)/(5*L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)) + 6*N*r33*(A**2*G**2*L**4*scf**2 - 10*A*E*G*Iyy*L**2*scf - 10*A*E*G*Izz*L**2*scf + 120*E**2*Iyy*Izz)/(5*L*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz))) + r23*(6*N*r32*(A**2*G**2*L**4*scf**2 - 10*A*E*G*Iyy*L**2*scf - 10*A*E*G*Izz*L**2*scf + 120*E**2*Iyy*Izz)/(5*L*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + 6*N*r33*(A**2*G**2*L**4*scf**2 - 20*A*E*G*Iyy*L**2*scf + 120*E**2*Iyy**2)/(5*L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)))
            k += 1
            KGv[k] += r32*(6*N*r32*(A**2*G**2*L**4*scf**2 - 20*A*E*G*Izz*L**2*scf + 120*E**2*Izz**2)/(5*L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)) + 6*N*r33*(A**2*G**2*L**4*scf**2 - 10*A*E*G*Iyy*L**2*scf - 10*A*E*G*Izz*L**2*scf + 120*E**2*Iyy*Izz)/(5*L*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz))) + r33*(6*N*r32*(A**2*G**2*L**4*scf**2 - 10*A*E*G*Iyy*L**2*scf - 10*A*E*G*Izz*L**2*scf + 120*E**2*Iyy*Izz)/(5*L*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + 6*N*r33*(A**2*G**2*L**4*scf**2 - 20*A*E*G*Iyy*L**2*scf + 120*E**2*Iyy**2)/(5*L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)))
            k += 1
            KGv[k] += r12*(A**2*G**2*L**4*N*r32*scf**2/(10*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + A**2*G**2*L**4*N*r33*scf**2/(10*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2))) + r13*(A**2*G**2*L**4*N*r32*scf**2/(10*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)) + A**2*G**2*L**4*N*r33*scf**2/(10*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)))
            k += 1
            KGv[k] += r22*(A**2*G**2*L**4*N*r32*scf**2/(10*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + A**2*G**2*L**4*N*r33*scf**2/(10*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2))) + r23*(A**2*G**2*L**4*N*r32*scf**2/(10*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)) + A**2*G**2*L**4*N*r33*scf**2/(10*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)))
            k += 1
            KGv[k] += r32*(A**2*G**2*L**4*N*r32*scf**2/(10*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + A**2*G**2*L**4*N*r33*scf**2/(10*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2))) + r33*(A**2*G**2*L**4*N*r32*scf**2/(10*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)) + A**2*G**2*L**4*N*r33*scf**2/(10*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)))
            k += 1
            KGv[k] += r12*(-6*N*r32*(A**2*G**2*L**4*scf**2 - 20*A*E*G*Izz*L**2*scf + 120*E**2*Izz**2)/(5*L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)) - 6*N*r33*(A**2*G**2*L**4*scf**2 - 10*A*E*G*Iyy*L**2*scf - 10*A*E*G*Izz*L**2*scf + 120*E**2*Iyy*Izz)/(5*L*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz))) + r13*(-6*N*r32*(A**2*G**2*L**4*scf**2 - 10*A*E*G*Iyy*L**2*scf - 10*A*E*G*Izz*L**2*scf + 120*E**2*Iyy*Izz)/(5*L*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) - 6*N*r33*(A**2*G**2*L**4*scf**2 - 20*A*E*G*Iyy*L**2*scf + 120*E**2*Iyy**2)/(5*L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)))
            k += 1
            KGv[k] += r22*(-6*N*r32*(A**2*G**2*L**4*scf**2 - 20*A*E*G*Izz*L**2*scf + 120*E**2*Izz**2)/(5*L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)) - 6*N*r33*(A**2*G**2*L**4*scf**2 - 10*A*E*G*Iyy*L**2*scf - 10*A*E*G*Izz*L**2*scf + 120*E**2*Iyy*Izz)/(5*L*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz))) + r23*(-6*N*r32*(A**2*G**2*L**4*scf**2 - 10*A*E*G*Iyy*L**2*scf - 10*A*E*G*Izz*L**2*scf + 120*E**2*Iyy*Izz)/(5*L*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) - 6*N*r33*(A**2*G**2*L**4*scf**2 - 20*A*E*G*Iyy*L**2*scf + 120*E**2*Iyy**2)/(5*L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)))
            k += 1
            KGv[k] += r32*(-6*N*r32*(A**2*G**2*L**4*scf**2 - 20*A*E*G*Izz*L**2*scf + 120*E**2*Izz**2)/(5*L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)) - 6*N*r33*(A**2*G**2*L**4*scf**2 - 10*A*E*G*Iyy*L**2*scf - 10*A*E*G*Izz*L**2*scf + 120*E**2*Iyy*Izz)/(5*L*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz))) + r33*(-6*N*r32*(A**2*G**2*L**4*scf**2 - 10*A*E*G*Iyy*L**2*scf - 10*A*E*G*Izz*L**2*scf + 120*E**2*Iyy*Izz)/(5*L*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) - 6*N*r33*(A**2*G**2*L**4*scf**2 - 20*A*E*G*Iyy*L**2*scf + 120*E**2*Iyy**2)/(5*L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)))
            k += 1
            KGv[k] += r12*(A**2*G**2*L**4*N*r32*scf**2/(10*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + A**2*G**2*L**4*N*r33*scf**2/(10*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2))) + r13*(A**2*G**2*L**4*N*r32*scf**2/(10*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)) + A**2*G**2*L**4*N*r33*scf**2/(10*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)))
            k += 1
            KGv[k] += r22*(A**2*G**2*L**4*N*r32*scf**2/(10*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + A**2*G**2*L**4*N*r33*scf**2/(10*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2))) + r23*(A**2*G**2*L**4*N*r32*scf**2/(10*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)) + A**2*G**2*L**4*N*r33*scf**2/(10*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)))
            k += 1
            KGv[k] += r32*(A**2*G**2*L**4*N*r32*scf**2/(10*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + A**2*G**2*L**4*N*r33*scf**2/(10*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2))) + r33*(A**2*G**2*L**4*N*r32*scf**2/(10*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)) + A**2*G**2*L**4*N*r33*scf**2/(10*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)))
            k += 1
            KGv[k] += r12*(A**2*G**2*L**4*N*r12*scf**2/(10*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + A**2*G**2*L**4*N*r13*scf**2/(10*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2))) + r13*(A**2*G**2*L**4*N*r12*scf**2/(10*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)) + A**2*G**2*L**4*N*r13*scf**2/(10*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)))
            k += 1
            KGv[k] += r22*(A**2*G**2*L**4*N*r12*scf**2/(10*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + A**2*G**2*L**4*N*r13*scf**2/(10*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2))) + r23*(A**2*G**2*L**4*N*r12*scf**2/(10*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)) + A**2*G**2*L**4*N*r13*scf**2/(10*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)))
            k += 1
            KGv[k] += r32*(A**2*G**2*L**4*N*r12*scf**2/(10*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + A**2*G**2*L**4*N*r13*scf**2/(10*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2))) + r33*(A**2*G**2*L**4*N*r12*scf**2/(10*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)) + A**2*G**2*L**4*N*r13*scf**2/(10*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)))
            k += 1
            KGv[k] += r12*(2*L*N*r12*(A**2*G**2*L**4*scf**2 - 15*A*E*G*Iyy*L**2*scf + 90*E**2*Iyy**2)/(15*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)) + L*N*r13*(2*A**2*G**2*L**4*scf**2 - 15*A*E*G*Iyy*L**2*scf - 15*A*E*G*Izz*L**2*scf + 180*E**2*Iyy*Izz)/(15*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz))) + r13*(L*N*r12*(2*A**2*G**2*L**4*scf**2 - 15*A*E*G*Iyy*L**2*scf - 15*A*E*G*Izz*L**2*scf + 180*E**2*Iyy*Izz)/(15*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + 2*L*N*r13*(A**2*G**2*L**4*scf**2 - 15*A*E*G*Izz*L**2*scf + 90*E**2*Izz**2)/(15*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)))
            k += 1
            KGv[k] += r22*(2*L*N*r12*(A**2*G**2*L**4*scf**2 - 15*A*E*G*Iyy*L**2*scf + 90*E**2*Iyy**2)/(15*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)) + L*N*r13*(2*A**2*G**2*L**4*scf**2 - 15*A*E*G*Iyy*L**2*scf - 15*A*E*G*Izz*L**2*scf + 180*E**2*Iyy*Izz)/(15*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz))) + r23*(L*N*r12*(2*A**2*G**2*L**4*scf**2 - 15*A*E*G*Iyy*L**2*scf - 15*A*E*G*Izz*L**2*scf + 180*E**2*Iyy*Izz)/(15*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + 2*L*N*r13*(A**2*G**2*L**4*scf**2 - 15*A*E*G*Izz*L**2*scf + 90*E**2*Izz**2)/(15*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)))
            k += 1
            KGv[k] += r32*(2*L*N*r12*(A**2*G**2*L**4*scf**2 - 15*A*E*G*Iyy*L**2*scf + 90*E**2*Iyy**2)/(15*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)) + L*N*r13*(2*A**2*G**2*L**4*scf**2 - 15*A*E*G*Iyy*L**2*scf - 15*A*E*G*Izz*L**2*scf + 180*E**2*Iyy*Izz)/(15*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz))) + r33*(L*N*r12*(2*A**2*G**2*L**4*scf**2 - 15*A*E*G*Iyy*L**2*scf - 15*A*E*G*Izz*L**2*scf + 180*E**2*Iyy*Izz)/(15*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + 2*L*N*r13*(A**2*G**2*L**4*scf**2 - 15*A*E*G*Izz*L**2*scf + 90*E**2*Izz**2)/(15*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)))
            k += 1
            KGv[k] += r12*(-A**2*G**2*L**4*N*r12*scf**2/(10*A**2*G**2*L**4*scf**2 - 120*A*E*G*Iyy*L**2*scf - 120*A*E*G*Izz*L**2*scf + 1440*E**2*Iyy*Izz) - A**2*G**2*L**4*N*r13*scf**2/(10*A**2*G**2*L**4*scf**2 - 240*A*E*G*Izz*L**2*scf + 1440*E**2*Izz**2)) + r13*(-A**2*G**2*L**4*N*r12*scf**2/(10*A**2*G**2*L**4*scf**2 - 240*A*E*G*Iyy*L**2*scf + 1440*E**2*Iyy**2) - A**2*G**2*L**4*N*r13*scf**2/(10*A**2*G**2*L**4*scf**2 - 120*A*E*G*Iyy*L**2*scf - 120*A*E*G*Izz*L**2*scf + 1440*E**2*Iyy*Izz))
            k += 1
            KGv[k] += r22*(-A**2*G**2*L**4*N*r12*scf**2/(10*A**2*G**2*L**4*scf**2 - 120*A*E*G*Iyy*L**2*scf - 120*A*E*G*Izz*L**2*scf + 1440*E**2*Iyy*Izz) - A**2*G**2*L**4*N*r13*scf**2/(10*A**2*G**2*L**4*scf**2 - 240*A*E*G*Izz*L**2*scf + 1440*E**2*Izz**2)) + r23*(-A**2*G**2*L**4*N*r12*scf**2/(10*A**2*G**2*L**4*scf**2 - 240*A*E*G*Iyy*L**2*scf + 1440*E**2*Iyy**2) - A**2*G**2*L**4*N*r13*scf**2/(10*A**2*G**2*L**4*scf**2 - 120*A*E*G*Iyy*L**2*scf - 120*A*E*G*Izz*L**2*scf + 1440*E**2*Iyy*Izz))
            k += 1
            KGv[k] += r32*(-A**2*G**2*L**4*N*r12*scf**2/(10*A**2*G**2*L**4*scf**2 - 120*A*E*G*Iyy*L**2*scf - 120*A*E*G*Izz*L**2*scf + 1440*E**2*Iyy*Izz) - A**2*G**2*L**4*N*r13*scf**2/(10*A**2*G**2*L**4*scf**2 - 240*A*E*G*Izz*L**2*scf + 1440*E**2*Izz**2)) + r33*(-A**2*G**2*L**4*N*r12*scf**2/(10*A**2*G**2*L**4*scf**2 - 240*A*E*G*Iyy*L**2*scf + 1440*E**2*Iyy**2) - A**2*G**2*L**4*N*r13*scf**2/(10*A**2*G**2*L**4*scf**2 - 120*A*E*G*Iyy*L**2*scf - 120*A*E*G*Izz*L**2*scf + 1440*E**2*Iyy*Izz))
            k += 1
            KGv[k] += r12*(-L*N*r12*(A**2*G**2*L**4*scf**2 - 60*A*E*G*Iyy*L**2*scf + 360*E**2*Iyy**2)/(30*A**2*G**2*L**4*scf**2 - 720*A*E*G*Iyy*L**2*scf + 4320*E**2*Iyy**2) - L*N*r13*(A**2*G**2*L**4*scf**2 - 30*A*E*G*Iyy*L**2*scf - 30*A*E*G*Izz*L**2*scf + 360*E**2*Iyy*Izz)/(30*A**2*G**2*L**4*scf**2 - 360*A*E*G*Iyy*L**2*scf - 360*A*E*G*Izz*L**2*scf + 4320*E**2*Iyy*Izz)) + r13*(-L*N*r12*(A**2*G**2*L**4*scf**2 - 30*A*E*G*Iyy*L**2*scf - 30*A*E*G*Izz*L**2*scf + 360*E**2*Iyy*Izz)/(30*A**2*G**2*L**4*scf**2 - 360*A*E*G*Iyy*L**2*scf - 360*A*E*G*Izz*L**2*scf + 4320*E**2*Iyy*Izz) - L*N*r13*(A**2*G**2*L**4*scf**2 - 60*A*E*G*Izz*L**2*scf + 360*E**2*Izz**2)/(30*A**2*G**2*L**4*scf**2 - 720*A*E*G*Izz*L**2*scf + 4320*E**2*Izz**2))
            k += 1
            KGv[k] += r22*(-L*N*r12*(A**2*G**2*L**4*scf**2 - 60*A*E*G*Iyy*L**2*scf + 360*E**2*Iyy**2)/(30*A**2*G**2*L**4*scf**2 - 720*A*E*G*Iyy*L**2*scf + 4320*E**2*Iyy**2) - L*N*r13*(A**2*G**2*L**4*scf**2 - 30*A*E*G*Iyy*L**2*scf - 30*A*E*G*Izz*L**2*scf + 360*E**2*Iyy*Izz)/(30*A**2*G**2*L**4*scf**2 - 360*A*E*G*Iyy*L**2*scf - 360*A*E*G*Izz*L**2*scf + 4320*E**2*Iyy*Izz)) + r23*(-L*N*r12*(A**2*G**2*L**4*scf**2 - 30*A*E*G*Iyy*L**2*scf - 30*A*E*G*Izz*L**2*scf + 360*E**2*Iyy*Izz)/(30*A**2*G**2*L**4*scf**2 - 360*A*E*G*Iyy*L**2*scf - 360*A*E*G*Izz*L**2*scf + 4320*E**2*Iyy*Izz) - L*N*r13*(A**2*G**2*L**4*scf**2 - 60*A*E*G*Izz*L**2*scf + 360*E**2*Izz**2)/(30*A**2*G**2*L**4*scf**2 - 720*A*E*G*Izz*L**2*scf + 4320*E**2*Izz**2))
            k += 1
            KGv[k] += r32*(-L*N*r12*(A**2*G**2*L**4*scf**2 - 60*A*E*G*Iyy*L**2*scf + 360*E**2*Iyy**2)/(30*A**2*G**2*L**4*scf**2 - 720*A*E*G*Iyy*L**2*scf + 4320*E**2*Iyy**2) - L*N*r13*(A**2*G**2*L**4*scf**2 - 30*A*E*G*Iyy*L**2*scf - 30*A*E*G*Izz*L**2*scf + 360*E**2*Iyy*Izz)/(30*A**2*G**2*L**4*scf**2 - 360*A*E*G*Iyy*L**2*scf - 360*A*E*G*Izz*L**2*scf + 4320*E**2*Iyy*Izz)) + r33*(-L*N*r12*(A**2*G**2*L**4*scf**2 - 30*A*E*G*Iyy*L**2*scf - 30*A*E*G*Izz*L**2*scf + 360*E**2*Iyy*Izz)/(30*A**2*G**2*L**4*scf**2 - 360*A*E*G*Iyy*L**2*scf - 360*A*E*G*Izz*L**2*scf + 4320*E**2*Iyy*Izz) - L*N*r13*(A**2*G**2*L**4*scf**2 - 60*A*E*G*Izz*L**2*scf + 360*E**2*Izz**2)/(30*A**2*G**2*L**4*scf**2 - 720*A*E*G*Izz*L**2*scf + 4320*E**2*Izz**2))
            k += 1
            KGv[k] += r12*(A**2*G**2*L**4*N*r22*scf**2/(10*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + A**2*G**2*L**4*N*r23*scf**2/(10*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2))) + r13*(A**2*G**2*L**4*N*r22*scf**2/(10*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)) + A**2*G**2*L**4*N*r23*scf**2/(10*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)))
            k += 1
            KGv[k] += r22*(A**2*G**2*L**4*N*r22*scf**2/(10*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + A**2*G**2*L**4*N*r23*scf**2/(10*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2))) + r23*(A**2*G**2*L**4*N*r22*scf**2/(10*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)) + A**2*G**2*L**4*N*r23*scf**2/(10*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)))
            k += 1
            KGv[k] += r32*(A**2*G**2*L**4*N*r22*scf**2/(10*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + A**2*G**2*L**4*N*r23*scf**2/(10*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2))) + r33*(A**2*G**2*L**4*N*r22*scf**2/(10*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)) + A**2*G**2*L**4*N*r23*scf**2/(10*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)))
            k += 1
            KGv[k] += r12*(2*L*N*r22*(A**2*G**2*L**4*scf**2 - 15*A*E*G*Iyy*L**2*scf + 90*E**2*Iyy**2)/(15*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)) + L*N*r23*(2*A**2*G**2*L**4*scf**2 - 15*A*E*G*Iyy*L**2*scf - 15*A*E*G*Izz*L**2*scf + 180*E**2*Iyy*Izz)/(15*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz))) + r13*(L*N*r22*(2*A**2*G**2*L**4*scf**2 - 15*A*E*G*Iyy*L**2*scf - 15*A*E*G*Izz*L**2*scf + 180*E**2*Iyy*Izz)/(15*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + 2*L*N*r23*(A**2*G**2*L**4*scf**2 - 15*A*E*G*Izz*L**2*scf + 90*E**2*Izz**2)/(15*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)))
            k += 1
            KGv[k] += r22*(2*L*N*r22*(A**2*G**2*L**4*scf**2 - 15*A*E*G*Iyy*L**2*scf + 90*E**2*Iyy**2)/(15*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)) + L*N*r23*(2*A**2*G**2*L**4*scf**2 - 15*A*E*G*Iyy*L**2*scf - 15*A*E*G*Izz*L**2*scf + 180*E**2*Iyy*Izz)/(15*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz))) + r23*(L*N*r22*(2*A**2*G**2*L**4*scf**2 - 15*A*E*G*Iyy*L**2*scf - 15*A*E*G*Izz*L**2*scf + 180*E**2*Iyy*Izz)/(15*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + 2*L*N*r23*(A**2*G**2*L**4*scf**2 - 15*A*E*G*Izz*L**2*scf + 90*E**2*Izz**2)/(15*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)))
            k += 1
            KGv[k] += r32*(2*L*N*r22*(A**2*G**2*L**4*scf**2 - 15*A*E*G*Iyy*L**2*scf + 90*E**2*Iyy**2)/(15*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)) + L*N*r23*(2*A**2*G**2*L**4*scf**2 - 15*A*E*G*Iyy*L**2*scf - 15*A*E*G*Izz*L**2*scf + 180*E**2*Iyy*Izz)/(15*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz))) + r33*(L*N*r22*(2*A**2*G**2*L**4*scf**2 - 15*A*E*G*Iyy*L**2*scf - 15*A*E*G*Izz*L**2*scf + 180*E**2*Iyy*Izz)/(15*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + 2*L*N*r23*(A**2*G**2*L**4*scf**2 - 15*A*E*G*Izz*L**2*scf + 90*E**2*Izz**2)/(15*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)))
            k += 1
            KGv[k] += r12*(-A**2*G**2*L**4*N*r22*scf**2/(10*A**2*G**2*L**4*scf**2 - 120*A*E*G*Iyy*L**2*scf - 120*A*E*G*Izz*L**2*scf + 1440*E**2*Iyy*Izz) - A**2*G**2*L**4*N*r23*scf**2/(10*A**2*G**2*L**4*scf**2 - 240*A*E*G*Izz*L**2*scf + 1440*E**2*Izz**2)) + r13*(-A**2*G**2*L**4*N*r22*scf**2/(10*A**2*G**2*L**4*scf**2 - 240*A*E*G*Iyy*L**2*scf + 1440*E**2*Iyy**2) - A**2*G**2*L**4*N*r23*scf**2/(10*A**2*G**2*L**4*scf**2 - 120*A*E*G*Iyy*L**2*scf - 120*A*E*G*Izz*L**2*scf + 1440*E**2*Iyy*Izz))
            k += 1
            KGv[k] += r22*(-A**2*G**2*L**4*N*r22*scf**2/(10*A**2*G**2*L**4*scf**2 - 120*A*E*G*Iyy*L**2*scf - 120*A*E*G*Izz*L**2*scf + 1440*E**2*Iyy*Izz) - A**2*G**2*L**4*N*r23*scf**2/(10*A**2*G**2*L**4*scf**2 - 240*A*E*G*Izz*L**2*scf + 1440*E**2*Izz**2)) + r23*(-A**2*G**2*L**4*N*r22*scf**2/(10*A**2*G**2*L**4*scf**2 - 240*A*E*G*Iyy*L**2*scf + 1440*E**2*Iyy**2) - A**2*G**2*L**4*N*r23*scf**2/(10*A**2*G**2*L**4*scf**2 - 120*A*E*G*Iyy*L**2*scf - 120*A*E*G*Izz*L**2*scf + 1440*E**2*Iyy*Izz))
            k += 1
            KGv[k] += r32*(-A**2*G**2*L**4*N*r22*scf**2/(10*A**2*G**2*L**4*scf**2 - 120*A*E*G*Iyy*L**2*scf - 120*A*E*G*Izz*L**2*scf + 1440*E**2*Iyy*Izz) - A**2*G**2*L**4*N*r23*scf**2/(10*A**2*G**2*L**4*scf**2 - 240*A*E*G*Izz*L**2*scf + 1440*E**2*Izz**2)) + r33*(-A**2*G**2*L**4*N*r22*scf**2/(10*A**2*G**2*L**4*scf**2 - 240*A*E*G*Iyy*L**2*scf + 1440*E**2*Iyy**2) - A**2*G**2*L**4*N*r23*scf**2/(10*A**2*G**2*L**4*scf**2 - 120*A*E*G*Iyy*L**2*scf - 120*A*E*G*Izz*L**2*scf + 1440*E**2*Iyy*Izz))
            k += 1
            KGv[k] += r12*(-L*N*r22*(A**2*G**2*L**4*scf**2 - 60*A*E*G*Iyy*L**2*scf + 360*E**2*Iyy**2)/(30*A**2*G**2*L**4*scf**2 - 720*A*E*G*Iyy*L**2*scf + 4320*E**2*Iyy**2) - L*N*r23*(A**2*G**2*L**4*scf**2 - 30*A*E*G*Iyy*L**2*scf - 30*A*E*G*Izz*L**2*scf + 360*E**2*Iyy*Izz)/(30*A**2*G**2*L**4*scf**2 - 360*A*E*G*Iyy*L**2*scf - 360*A*E*G*Izz*L**2*scf + 4320*E**2*Iyy*Izz)) + r13*(-L*N*r22*(A**2*G**2*L**4*scf**2 - 30*A*E*G*Iyy*L**2*scf - 30*A*E*G*Izz*L**2*scf + 360*E**2*Iyy*Izz)/(30*A**2*G**2*L**4*scf**2 - 360*A*E*G*Iyy*L**2*scf - 360*A*E*G*Izz*L**2*scf + 4320*E**2*Iyy*Izz) - L*N*r23*(A**2*G**2*L**4*scf**2 - 60*A*E*G*Izz*L**2*scf + 360*E**2*Izz**2)/(30*A**2*G**2*L**4*scf**2 - 720*A*E*G*Izz*L**2*scf + 4320*E**2*Izz**2))
            k += 1
            KGv[k] += r22*(-L*N*r22*(A**2*G**2*L**4*scf**2 - 60*A*E*G*Iyy*L**2*scf + 360*E**2*Iyy**2)/(30*A**2*G**2*L**4*scf**2 - 720*A*E*G*Iyy*L**2*scf + 4320*E**2*Iyy**2) - L*N*r23*(A**2*G**2*L**4*scf**2 - 30*A*E*G*Iyy*L**2*scf - 30*A*E*G*Izz*L**2*scf + 360*E**2*Iyy*Izz)/(30*A**2*G**2*L**4*scf**2 - 360*A*E*G*Iyy*L**2*scf - 360*A*E*G*Izz*L**2*scf + 4320*E**2*Iyy*Izz)) + r23*(-L*N*r22*(A**2*G**2*L**4*scf**2 - 30*A*E*G*Iyy*L**2*scf - 30*A*E*G*Izz*L**2*scf + 360*E**2*Iyy*Izz)/(30*A**2*G**2*L**4*scf**2 - 360*A*E*G*Iyy*L**2*scf - 360*A*E*G*Izz*L**2*scf + 4320*E**2*Iyy*Izz) - L*N*r23*(A**2*G**2*L**4*scf**2 - 60*A*E*G*Izz*L**2*scf + 360*E**2*Izz**2)/(30*A**2*G**2*L**4*scf**2 - 720*A*E*G*Izz*L**2*scf + 4320*E**2*Izz**2))
            k += 1
            KGv[k] += r32*(-L*N*r22*(A**2*G**2*L**4*scf**2 - 60*A*E*G*Iyy*L**2*scf + 360*E**2*Iyy**2)/(30*A**2*G**2*L**4*scf**2 - 720*A*E*G*Iyy*L**2*scf + 4320*E**2*Iyy**2) - L*N*r23*(A**2*G**2*L**4*scf**2 - 30*A*E*G*Iyy*L**2*scf - 30*A*E*G*Izz*L**2*scf + 360*E**2*Iyy*Izz)/(30*A**2*G**2*L**4*scf**2 - 360*A*E*G*Iyy*L**2*scf - 360*A*E*G*Izz*L**2*scf + 4320*E**2*Iyy*Izz)) + r33*(-L*N*r22*(A**2*G**2*L**4*scf**2 - 30*A*E*G*Iyy*L**2*scf - 30*A*E*G*Izz*L**2*scf + 360*E**2*Iyy*Izz)/(30*A**2*G**2*L**4*scf**2 - 360*A*E*G*Iyy*L**2*scf - 360*A*E*G*Izz*L**2*scf + 4320*E**2*Iyy*Izz) - L*N*r23*(A**2*G**2*L**4*scf**2 - 60*A*E*G*Izz*L**2*scf + 360*E**2*Izz**2)/(30*A**2*G**2*L**4*scf**2 - 720*A*E*G*Izz*L**2*scf + 4320*E**2*Izz**2))
            k += 1
            KGv[k] += r12*(A**2*G**2*L**4*N*r32*scf**2/(10*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + A**2*G**2*L**4*N*r33*scf**2/(10*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2))) + r13*(A**2*G**2*L**4*N*r32*scf**2/(10*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)) + A**2*G**2*L**4*N*r33*scf**2/(10*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)))
            k += 1
            KGv[k] += r22*(A**2*G**2*L**4*N*r32*scf**2/(10*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + A**2*G**2*L**4*N*r33*scf**2/(10*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2))) + r23*(A**2*G**2*L**4*N*r32*scf**2/(10*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)) + A**2*G**2*L**4*N*r33*scf**2/(10*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)))
            k += 1
            KGv[k] += r32*(A**2*G**2*L**4*N*r32*scf**2/(10*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + A**2*G**2*L**4*N*r33*scf**2/(10*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2))) + r33*(A**2*G**2*L**4*N*r32*scf**2/(10*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)) + A**2*G**2*L**4*N*r33*scf**2/(10*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)))
            k += 1
            KGv[k] += r12*(2*L*N*r32*(A**2*G**2*L**4*scf**2 - 15*A*E*G*Iyy*L**2*scf + 90*E**2*Iyy**2)/(15*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)) + L*N*r33*(2*A**2*G**2*L**4*scf**2 - 15*A*E*G*Iyy*L**2*scf - 15*A*E*G*Izz*L**2*scf + 180*E**2*Iyy*Izz)/(15*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz))) + r13*(L*N*r32*(2*A**2*G**2*L**4*scf**2 - 15*A*E*G*Iyy*L**2*scf - 15*A*E*G*Izz*L**2*scf + 180*E**2*Iyy*Izz)/(15*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + 2*L*N*r33*(A**2*G**2*L**4*scf**2 - 15*A*E*G*Izz*L**2*scf + 90*E**2*Izz**2)/(15*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)))
            k += 1
            KGv[k] += r22*(2*L*N*r32*(A**2*G**2*L**4*scf**2 - 15*A*E*G*Iyy*L**2*scf + 90*E**2*Iyy**2)/(15*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)) + L*N*r33*(2*A**2*G**2*L**4*scf**2 - 15*A*E*G*Iyy*L**2*scf - 15*A*E*G*Izz*L**2*scf + 180*E**2*Iyy*Izz)/(15*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz))) + r23*(L*N*r32*(2*A**2*G**2*L**4*scf**2 - 15*A*E*G*Iyy*L**2*scf - 15*A*E*G*Izz*L**2*scf + 180*E**2*Iyy*Izz)/(15*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + 2*L*N*r33*(A**2*G**2*L**4*scf**2 - 15*A*E*G*Izz*L**2*scf + 90*E**2*Izz**2)/(15*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)))
            k += 1
            KGv[k] += r32*(2*L*N*r32*(A**2*G**2*L**4*scf**2 - 15*A*E*G*Iyy*L**2*scf + 90*E**2*Iyy**2)/(15*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)) + L*N*r33*(2*A**2*G**2*L**4*scf**2 - 15*A*E*G*Iyy*L**2*scf - 15*A*E*G*Izz*L**2*scf + 180*E**2*Iyy*Izz)/(15*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz))) + r33*(L*N*r32*(2*A**2*G**2*L**4*scf**2 - 15*A*E*G*Iyy*L**2*scf - 15*A*E*G*Izz*L**2*scf + 180*E**2*Iyy*Izz)/(15*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + 2*L*N*r33*(A**2*G**2*L**4*scf**2 - 15*A*E*G*Izz*L**2*scf + 90*E**2*Izz**2)/(15*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)))
            k += 1
            KGv[k] += r12*(-A**2*G**2*L**4*N*r32*scf**2/(10*A**2*G**2*L**4*scf**2 - 120*A*E*G*Iyy*L**2*scf - 120*A*E*G*Izz*L**2*scf + 1440*E**2*Iyy*Izz) - A**2*G**2*L**4*N*r33*scf**2/(10*A**2*G**2*L**4*scf**2 - 240*A*E*G*Izz*L**2*scf + 1440*E**2*Izz**2)) + r13*(-A**2*G**2*L**4*N*r32*scf**2/(10*A**2*G**2*L**4*scf**2 - 240*A*E*G*Iyy*L**2*scf + 1440*E**2*Iyy**2) - A**2*G**2*L**4*N*r33*scf**2/(10*A**2*G**2*L**4*scf**2 - 120*A*E*G*Iyy*L**2*scf - 120*A*E*G*Izz*L**2*scf + 1440*E**2*Iyy*Izz))
            k += 1
            KGv[k] += r22*(-A**2*G**2*L**4*N*r32*scf**2/(10*A**2*G**2*L**4*scf**2 - 120*A*E*G*Iyy*L**2*scf - 120*A*E*G*Izz*L**2*scf + 1440*E**2*Iyy*Izz) - A**2*G**2*L**4*N*r33*scf**2/(10*A**2*G**2*L**4*scf**2 - 240*A*E*G*Izz*L**2*scf + 1440*E**2*Izz**2)) + r23*(-A**2*G**2*L**4*N*r32*scf**2/(10*A**2*G**2*L**4*scf**2 - 240*A*E*G*Iyy*L**2*scf + 1440*E**2*Iyy**2) - A**2*G**2*L**4*N*r33*scf**2/(10*A**2*G**2*L**4*scf**2 - 120*A*E*G*Iyy*L**2*scf - 120*A*E*G*Izz*L**2*scf + 1440*E**2*Iyy*Izz))
            k += 1
            KGv[k] += r32*(-A**2*G**2*L**4*N*r32*scf**2/(10*A**2*G**2*L**4*scf**2 - 120*A*E*G*Iyy*L**2*scf - 120*A*E*G*Izz*L**2*scf + 1440*E**2*Iyy*Izz) - A**2*G**2*L**4*N*r33*scf**2/(10*A**2*G**2*L**4*scf**2 - 240*A*E*G*Izz*L**2*scf + 1440*E**2*Izz**2)) + r33*(-A**2*G**2*L**4*N*r32*scf**2/(10*A**2*G**2*L**4*scf**2 - 240*A*E*G*Iyy*L**2*scf + 1440*E**2*Iyy**2) - A**2*G**2*L**4*N*r33*scf**2/(10*A**2*G**2*L**4*scf**2 - 120*A*E*G*Iyy*L**2*scf - 120*A*E*G*Izz*L**2*scf + 1440*E**2*Iyy*Izz))
            k += 1
            KGv[k] += r12*(-L*N*r32*(A**2*G**2*L**4*scf**2 - 60*A*E*G*Iyy*L**2*scf + 360*E**2*Iyy**2)/(30*A**2*G**2*L**4*scf**2 - 720*A*E*G*Iyy*L**2*scf + 4320*E**2*Iyy**2) - L*N*r33*(A**2*G**2*L**4*scf**2 - 30*A*E*G*Iyy*L**2*scf - 30*A*E*G*Izz*L**2*scf + 360*E**2*Iyy*Izz)/(30*A**2*G**2*L**4*scf**2 - 360*A*E*G*Iyy*L**2*scf - 360*A*E*G*Izz*L**2*scf + 4320*E**2*Iyy*Izz)) + r13*(-L*N*r32*(A**2*G**2*L**4*scf**2 - 30*A*E*G*Iyy*L**2*scf - 30*A*E*G*Izz*L**2*scf + 360*E**2*Iyy*Izz)/(30*A**2*G**2*L**4*scf**2 - 360*A*E*G*Iyy*L**2*scf - 360*A*E*G*Izz*L**2*scf + 4320*E**2*Iyy*Izz) - L*N*r33*(A**2*G**2*L**4*scf**2 - 60*A*E*G*Izz*L**2*scf + 360*E**2*Izz**2)/(30*A**2*G**2*L**4*scf**2 - 720*A*E*G*Izz*L**2*scf + 4320*E**2*Izz**2))
            k += 1
            KGv[k] += r22*(-L*N*r32*(A**2*G**2*L**4*scf**2 - 60*A*E*G*Iyy*L**2*scf + 360*E**2*Iyy**2)/(30*A**2*G**2*L**4*scf**2 - 720*A*E*G*Iyy*L**2*scf + 4320*E**2*Iyy**2) - L*N*r33*(A**2*G**2*L**4*scf**2 - 30*A*E*G*Iyy*L**2*scf - 30*A*E*G*Izz*L**2*scf + 360*E**2*Iyy*Izz)/(30*A**2*G**2*L**4*scf**2 - 360*A*E*G*Iyy*L**2*scf - 360*A*E*G*Izz*L**2*scf + 4320*E**2*Iyy*Izz)) + r23*(-L*N*r32*(A**2*G**2*L**4*scf**2 - 30*A*E*G*Iyy*L**2*scf - 30*A*E*G*Izz*L**2*scf + 360*E**2*Iyy*Izz)/(30*A**2*G**2*L**4*scf**2 - 360*A*E*G*Iyy*L**2*scf - 360*A*E*G*Izz*L**2*scf + 4320*E**2*Iyy*Izz) - L*N*r33*(A**2*G**2*L**4*scf**2 - 60*A*E*G*Izz*L**2*scf + 360*E**2*Izz**2)/(30*A**2*G**2*L**4*scf**2 - 720*A*E*G*Izz*L**2*scf + 4320*E**2*Izz**2))
            k += 1
            KGv[k] += r32*(-L*N*r32*(A**2*G**2*L**4*scf**2 - 60*A*E*G*Iyy*L**2*scf + 360*E**2*Iyy**2)/(30*A**2*G**2*L**4*scf**2 - 720*A*E*G*Iyy*L**2*scf + 4320*E**2*Iyy**2) - L*N*r33*(A**2*G**2*L**4*scf**2 - 30*A*E*G*Iyy*L**2*scf - 30*A*E*G*Izz*L**2*scf + 360*E**2*Iyy*Izz)/(30*A**2*G**2*L**4*scf**2 - 360*A*E*G*Iyy*L**2*scf - 360*A*E*G*Izz*L**2*scf + 4320*E**2*Iyy*Izz)) + r33*(-L*N*r32*(A**2*G**2*L**4*scf**2 - 30*A*E*G*Iyy*L**2*scf - 30*A*E*G*Izz*L**2*scf + 360*E**2*Iyy*Izz)/(30*A**2*G**2*L**4*scf**2 - 360*A*E*G*Iyy*L**2*scf - 360*A*E*G*Izz*L**2*scf + 4320*E**2*Iyy*Izz) - L*N*r33*(A**2*G**2*L**4*scf**2 - 60*A*E*G*Izz*L**2*scf + 360*E**2*Izz**2)/(30*A**2*G**2*L**4*scf**2 - 720*A*E*G*Izz*L**2*scf + 4320*E**2*Izz**2))
            k += 1
            KGv[k] += r12*(-6*N*r12*(A**2*G**2*L**4*scf**2 - 20*A*E*G*Izz*L**2*scf + 120*E**2*Izz**2)/(5*L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)) - 6*N*r13*(A**2*G**2*L**4*scf**2 - 10*A*E*G*Iyy*L**2*scf - 10*A*E*G*Izz*L**2*scf + 120*E**2*Iyy*Izz)/(5*L*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz))) + r13*(-6*N*r12*(A**2*G**2*L**4*scf**2 - 10*A*E*G*Iyy*L**2*scf - 10*A*E*G*Izz*L**2*scf + 120*E**2*Iyy*Izz)/(5*L*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) - 6*N*r13*(A**2*G**2*L**4*scf**2 - 20*A*E*G*Iyy*L**2*scf + 120*E**2*Iyy**2)/(5*L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)))
            k += 1
            KGv[k] += r22*(-6*N*r12*(A**2*G**2*L**4*scf**2 - 20*A*E*G*Izz*L**2*scf + 120*E**2*Izz**2)/(5*L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)) - 6*N*r13*(A**2*G**2*L**4*scf**2 - 10*A*E*G*Iyy*L**2*scf - 10*A*E*G*Izz*L**2*scf + 120*E**2*Iyy*Izz)/(5*L*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz))) + r23*(-6*N*r12*(A**2*G**2*L**4*scf**2 - 10*A*E*G*Iyy*L**2*scf - 10*A*E*G*Izz*L**2*scf + 120*E**2*Iyy*Izz)/(5*L*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) - 6*N*r13*(A**2*G**2*L**4*scf**2 - 20*A*E*G*Iyy*L**2*scf + 120*E**2*Iyy**2)/(5*L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)))
            k += 1
            KGv[k] += r32*(-6*N*r12*(A**2*G**2*L**4*scf**2 - 20*A*E*G*Izz*L**2*scf + 120*E**2*Izz**2)/(5*L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)) - 6*N*r13*(A**2*G**2*L**4*scf**2 - 10*A*E*G*Iyy*L**2*scf - 10*A*E*G*Izz*L**2*scf + 120*E**2*Iyy*Izz)/(5*L*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz))) + r33*(-6*N*r12*(A**2*G**2*L**4*scf**2 - 10*A*E*G*Iyy*L**2*scf - 10*A*E*G*Izz*L**2*scf + 120*E**2*Iyy*Izz)/(5*L*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) - 6*N*r13*(A**2*G**2*L**4*scf**2 - 20*A*E*G*Iyy*L**2*scf + 120*E**2*Iyy**2)/(5*L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)))
            k += 1
            KGv[k] += r12*(-A**2*G**2*L**4*N*r12*scf**2/(10*A**2*G**2*L**4*scf**2 - 120*A*E*G*Iyy*L**2*scf - 120*A*E*G*Izz*L**2*scf + 1440*E**2*Iyy*Izz) - A**2*G**2*L**4*N*r13*scf**2/(10*A**2*G**2*L**4*scf**2 - 240*A*E*G*Iyy*L**2*scf + 1440*E**2*Iyy**2)) + r13*(-A**2*G**2*L**4*N*r12*scf**2/(10*A**2*G**2*L**4*scf**2 - 240*A*E*G*Izz*L**2*scf + 1440*E**2*Izz**2) - A**2*G**2*L**4*N*r13*scf**2/(10*A**2*G**2*L**4*scf**2 - 120*A*E*G*Iyy*L**2*scf - 120*A*E*G*Izz*L**2*scf + 1440*E**2*Iyy*Izz))
            k += 1
            KGv[k] += r22*(-A**2*G**2*L**4*N*r12*scf**2/(10*A**2*G**2*L**4*scf**2 - 120*A*E*G*Iyy*L**2*scf - 120*A*E*G*Izz*L**2*scf + 1440*E**2*Iyy*Izz) - A**2*G**2*L**4*N*r13*scf**2/(10*A**2*G**2*L**4*scf**2 - 240*A*E*G*Iyy*L**2*scf + 1440*E**2*Iyy**2)) + r23*(-A**2*G**2*L**4*N*r12*scf**2/(10*A**2*G**2*L**4*scf**2 - 240*A*E*G*Izz*L**2*scf + 1440*E**2*Izz**2) - A**2*G**2*L**4*N*r13*scf**2/(10*A**2*G**2*L**4*scf**2 - 120*A*E*G*Iyy*L**2*scf - 120*A*E*G*Izz*L**2*scf + 1440*E**2*Iyy*Izz))
            k += 1
            KGv[k] += r32*(-A**2*G**2*L**4*N*r12*scf**2/(10*A**2*G**2*L**4*scf**2 - 120*A*E*G*Iyy*L**2*scf - 120*A*E*G*Izz*L**2*scf + 1440*E**2*Iyy*Izz) - A**2*G**2*L**4*N*r13*scf**2/(10*A**2*G**2*L**4*scf**2 - 240*A*E*G*Iyy*L**2*scf + 1440*E**2*Iyy**2)) + r33*(-A**2*G**2*L**4*N*r12*scf**2/(10*A**2*G**2*L**4*scf**2 - 240*A*E*G*Izz*L**2*scf + 1440*E**2*Izz**2) - A**2*G**2*L**4*N*r13*scf**2/(10*A**2*G**2*L**4*scf**2 - 120*A*E*G*Iyy*L**2*scf - 120*A*E*G*Izz*L**2*scf + 1440*E**2*Iyy*Izz))
            k += 1
            KGv[k] += r12*(6*N*r12*(A**2*G**2*L**4*scf**2 - 20*A*E*G*Izz*L**2*scf + 120*E**2*Izz**2)/(5*L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)) + 6*N*r13*(A**2*G**2*L**4*scf**2 - 10*A*E*G*Iyy*L**2*scf - 10*A*E*G*Izz*L**2*scf + 120*E**2*Iyy*Izz)/(5*L*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz))) + r13*(6*N*r12*(A**2*G**2*L**4*scf**2 - 10*A*E*G*Iyy*L**2*scf - 10*A*E*G*Izz*L**2*scf + 120*E**2*Iyy*Izz)/(5*L*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + 6*N*r13*(A**2*G**2*L**4*scf**2 - 20*A*E*G*Iyy*L**2*scf + 120*E**2*Iyy**2)/(5*L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)))
            k += 1
            KGv[k] += r22*(6*N*r12*(A**2*G**2*L**4*scf**2 - 20*A*E*G*Izz*L**2*scf + 120*E**2*Izz**2)/(5*L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)) + 6*N*r13*(A**2*G**2*L**4*scf**2 - 10*A*E*G*Iyy*L**2*scf - 10*A*E*G*Izz*L**2*scf + 120*E**2*Iyy*Izz)/(5*L*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz))) + r23*(6*N*r12*(A**2*G**2*L**4*scf**2 - 10*A*E*G*Iyy*L**2*scf - 10*A*E*G*Izz*L**2*scf + 120*E**2*Iyy*Izz)/(5*L*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + 6*N*r13*(A**2*G**2*L**4*scf**2 - 20*A*E*G*Iyy*L**2*scf + 120*E**2*Iyy**2)/(5*L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)))
            k += 1
            KGv[k] += r32*(6*N*r12*(A**2*G**2*L**4*scf**2 - 20*A*E*G*Izz*L**2*scf + 120*E**2*Izz**2)/(5*L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)) + 6*N*r13*(A**2*G**2*L**4*scf**2 - 10*A*E*G*Iyy*L**2*scf - 10*A*E*G*Izz*L**2*scf + 120*E**2*Iyy*Izz)/(5*L*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz))) + r33*(6*N*r12*(A**2*G**2*L**4*scf**2 - 10*A*E*G*Iyy*L**2*scf - 10*A*E*G*Izz*L**2*scf + 120*E**2*Iyy*Izz)/(5*L*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + 6*N*r13*(A**2*G**2*L**4*scf**2 - 20*A*E*G*Iyy*L**2*scf + 120*E**2*Iyy**2)/(5*L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)))
            k += 1
            KGv[k] += r12*(-A**2*G**2*L**4*N*r12*scf**2/(10*A**2*G**2*L**4*scf**2 - 120*A*E*G*Iyy*L**2*scf - 120*A*E*G*Izz*L**2*scf + 1440*E**2*Iyy*Izz) - A**2*G**2*L**4*N*r13*scf**2/(10*A**2*G**2*L**4*scf**2 - 240*A*E*G*Iyy*L**2*scf + 1440*E**2*Iyy**2)) + r13*(-A**2*G**2*L**4*N*r12*scf**2/(10*A**2*G**2*L**4*scf**2 - 240*A*E*G*Izz*L**2*scf + 1440*E**2*Izz**2) - A**2*G**2*L**4*N*r13*scf**2/(10*A**2*G**2*L**4*scf**2 - 120*A*E*G*Iyy*L**2*scf - 120*A*E*G*Izz*L**2*scf + 1440*E**2*Iyy*Izz))
            k += 1
            KGv[k] += r22*(-A**2*G**2*L**4*N*r12*scf**2/(10*A**2*G**2*L**4*scf**2 - 120*A*E*G*Iyy*L**2*scf - 120*A*E*G*Izz*L**2*scf + 1440*E**2*Iyy*Izz) - A**2*G**2*L**4*N*r13*scf**2/(10*A**2*G**2*L**4*scf**2 - 240*A*E*G*Iyy*L**2*scf + 1440*E**2*Iyy**2)) + r23*(-A**2*G**2*L**4*N*r12*scf**2/(10*A**2*G**2*L**4*scf**2 - 240*A*E*G*Izz*L**2*scf + 1440*E**2*Izz**2) - A**2*G**2*L**4*N*r13*scf**2/(10*A**2*G**2*L**4*scf**2 - 120*A*E*G*Iyy*L**2*scf - 120*A*E*G*Izz*L**2*scf + 1440*E**2*Iyy*Izz))
            k += 1
            KGv[k] += r32*(-A**2*G**2*L**4*N*r12*scf**2/(10*A**2*G**2*L**4*scf**2 - 120*A*E*G*Iyy*L**2*scf - 120*A*E*G*Izz*L**2*scf + 1440*E**2*Iyy*Izz) - A**2*G**2*L**4*N*r13*scf**2/(10*A**2*G**2*L**4*scf**2 - 240*A*E*G*Iyy*L**2*scf + 1440*E**2*Iyy**2)) + r33*(-A**2*G**2*L**4*N*r12*scf**2/(10*A**2*G**2*L**4*scf**2 - 240*A*E*G*Izz*L**2*scf + 1440*E**2*Izz**2) - A**2*G**2*L**4*N*r13*scf**2/(10*A**2*G**2*L**4*scf**2 - 120*A*E*G*Iyy*L**2*scf - 120*A*E*G*Izz*L**2*scf + 1440*E**2*Iyy*Izz))
            k += 1
            KGv[k] += r12*(-6*N*r22*(A**2*G**2*L**4*scf**2 - 20*A*E*G*Izz*L**2*scf + 120*E**2*Izz**2)/(5*L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)) - 6*N*r23*(A**2*G**2*L**4*scf**2 - 10*A*E*G*Iyy*L**2*scf - 10*A*E*G*Izz*L**2*scf + 120*E**2*Iyy*Izz)/(5*L*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz))) + r13*(-6*N*r22*(A**2*G**2*L**4*scf**2 - 10*A*E*G*Iyy*L**2*scf - 10*A*E*G*Izz*L**2*scf + 120*E**2*Iyy*Izz)/(5*L*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) - 6*N*r23*(A**2*G**2*L**4*scf**2 - 20*A*E*G*Iyy*L**2*scf + 120*E**2*Iyy**2)/(5*L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)))
            k += 1
            KGv[k] += r22*(-6*N*r22*(A**2*G**2*L**4*scf**2 - 20*A*E*G*Izz*L**2*scf + 120*E**2*Izz**2)/(5*L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)) - 6*N*r23*(A**2*G**2*L**4*scf**2 - 10*A*E*G*Iyy*L**2*scf - 10*A*E*G*Izz*L**2*scf + 120*E**2*Iyy*Izz)/(5*L*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz))) + r23*(-6*N*r22*(A**2*G**2*L**4*scf**2 - 10*A*E*G*Iyy*L**2*scf - 10*A*E*G*Izz*L**2*scf + 120*E**2*Iyy*Izz)/(5*L*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) - 6*N*r23*(A**2*G**2*L**4*scf**2 - 20*A*E*G*Iyy*L**2*scf + 120*E**2*Iyy**2)/(5*L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)))
            k += 1
            KGv[k] += r32*(-6*N*r22*(A**2*G**2*L**4*scf**2 - 20*A*E*G*Izz*L**2*scf + 120*E**2*Izz**2)/(5*L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)) - 6*N*r23*(A**2*G**2*L**4*scf**2 - 10*A*E*G*Iyy*L**2*scf - 10*A*E*G*Izz*L**2*scf + 120*E**2*Iyy*Izz)/(5*L*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz))) + r33*(-6*N*r22*(A**2*G**2*L**4*scf**2 - 10*A*E*G*Iyy*L**2*scf - 10*A*E*G*Izz*L**2*scf + 120*E**2*Iyy*Izz)/(5*L*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) - 6*N*r23*(A**2*G**2*L**4*scf**2 - 20*A*E*G*Iyy*L**2*scf + 120*E**2*Iyy**2)/(5*L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)))
            k += 1
            KGv[k] += r12*(-A**2*G**2*L**4*N*r22*scf**2/(10*A**2*G**2*L**4*scf**2 - 120*A*E*G*Iyy*L**2*scf - 120*A*E*G*Izz*L**2*scf + 1440*E**2*Iyy*Izz) - A**2*G**2*L**4*N*r23*scf**2/(10*A**2*G**2*L**4*scf**2 - 240*A*E*G*Iyy*L**2*scf + 1440*E**2*Iyy**2)) + r13*(-A**2*G**2*L**4*N*r22*scf**2/(10*A**2*G**2*L**4*scf**2 - 240*A*E*G*Izz*L**2*scf + 1440*E**2*Izz**2) - A**2*G**2*L**4*N*r23*scf**2/(10*A**2*G**2*L**4*scf**2 - 120*A*E*G*Iyy*L**2*scf - 120*A*E*G*Izz*L**2*scf + 1440*E**2*Iyy*Izz))
            k += 1
            KGv[k] += r22*(-A**2*G**2*L**4*N*r22*scf**2/(10*A**2*G**2*L**4*scf**2 - 120*A*E*G*Iyy*L**2*scf - 120*A*E*G*Izz*L**2*scf + 1440*E**2*Iyy*Izz) - A**2*G**2*L**4*N*r23*scf**2/(10*A**2*G**2*L**4*scf**2 - 240*A*E*G*Iyy*L**2*scf + 1440*E**2*Iyy**2)) + r23*(-A**2*G**2*L**4*N*r22*scf**2/(10*A**2*G**2*L**4*scf**2 - 240*A*E*G*Izz*L**2*scf + 1440*E**2*Izz**2) - A**2*G**2*L**4*N*r23*scf**2/(10*A**2*G**2*L**4*scf**2 - 120*A*E*G*Iyy*L**2*scf - 120*A*E*G*Izz*L**2*scf + 1440*E**2*Iyy*Izz))
            k += 1
            KGv[k] += r32*(-A**2*G**2*L**4*N*r22*scf**2/(10*A**2*G**2*L**4*scf**2 - 120*A*E*G*Iyy*L**2*scf - 120*A*E*G*Izz*L**2*scf + 1440*E**2*Iyy*Izz) - A**2*G**2*L**4*N*r23*scf**2/(10*A**2*G**2*L**4*scf**2 - 240*A*E*G*Iyy*L**2*scf + 1440*E**2*Iyy**2)) + r33*(-A**2*G**2*L**4*N*r22*scf**2/(10*A**2*G**2*L**4*scf**2 - 240*A*E*G*Izz*L**2*scf + 1440*E**2*Izz**2) - A**2*G**2*L**4*N*r23*scf**2/(10*A**2*G**2*L**4*scf**2 - 120*A*E*G*Iyy*L**2*scf - 120*A*E*G*Izz*L**2*scf + 1440*E**2*Iyy*Izz))
            k += 1
            KGv[k] += r12*(6*N*r22*(A**2*G**2*L**4*scf**2 - 20*A*E*G*Izz*L**2*scf + 120*E**2*Izz**2)/(5*L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)) + 6*N*r23*(A**2*G**2*L**4*scf**2 - 10*A*E*G*Iyy*L**2*scf - 10*A*E*G*Izz*L**2*scf + 120*E**2*Iyy*Izz)/(5*L*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz))) + r13*(6*N*r22*(A**2*G**2*L**4*scf**2 - 10*A*E*G*Iyy*L**2*scf - 10*A*E*G*Izz*L**2*scf + 120*E**2*Iyy*Izz)/(5*L*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + 6*N*r23*(A**2*G**2*L**4*scf**2 - 20*A*E*G*Iyy*L**2*scf + 120*E**2*Iyy**2)/(5*L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)))
            k += 1
            KGv[k] += r22*(6*N*r22*(A**2*G**2*L**4*scf**2 - 20*A*E*G*Izz*L**2*scf + 120*E**2*Izz**2)/(5*L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)) + 6*N*r23*(A**2*G**2*L**4*scf**2 - 10*A*E*G*Iyy*L**2*scf - 10*A*E*G*Izz*L**2*scf + 120*E**2*Iyy*Izz)/(5*L*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz))) + r23*(6*N*r22*(A**2*G**2*L**4*scf**2 - 10*A*E*G*Iyy*L**2*scf - 10*A*E*G*Izz*L**2*scf + 120*E**2*Iyy*Izz)/(5*L*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + 6*N*r23*(A**2*G**2*L**4*scf**2 - 20*A*E*G*Iyy*L**2*scf + 120*E**2*Iyy**2)/(5*L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)))
            k += 1
            KGv[k] += r32*(6*N*r22*(A**2*G**2*L**4*scf**2 - 20*A*E*G*Izz*L**2*scf + 120*E**2*Izz**2)/(5*L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)) + 6*N*r23*(A**2*G**2*L**4*scf**2 - 10*A*E*G*Iyy*L**2*scf - 10*A*E*G*Izz*L**2*scf + 120*E**2*Iyy*Izz)/(5*L*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz))) + r33*(6*N*r22*(A**2*G**2*L**4*scf**2 - 10*A*E*G*Iyy*L**2*scf - 10*A*E*G*Izz*L**2*scf + 120*E**2*Iyy*Izz)/(5*L*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + 6*N*r23*(A**2*G**2*L**4*scf**2 - 20*A*E*G*Iyy*L**2*scf + 120*E**2*Iyy**2)/(5*L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)))
            k += 1
            KGv[k] += r12*(-A**2*G**2*L**4*N*r22*scf**2/(10*A**2*G**2*L**4*scf**2 - 120*A*E*G*Iyy*L**2*scf - 120*A*E*G*Izz*L**2*scf + 1440*E**2*Iyy*Izz) - A**2*G**2*L**4*N*r23*scf**2/(10*A**2*G**2*L**4*scf**2 - 240*A*E*G*Iyy*L**2*scf + 1440*E**2*Iyy**2)) + r13*(-A**2*G**2*L**4*N*r22*scf**2/(10*A**2*G**2*L**4*scf**2 - 240*A*E*G*Izz*L**2*scf + 1440*E**2*Izz**2) - A**2*G**2*L**4*N*r23*scf**2/(10*A**2*G**2*L**4*scf**2 - 120*A*E*G*Iyy*L**2*scf - 120*A*E*G*Izz*L**2*scf + 1440*E**2*Iyy*Izz))
            k += 1
            KGv[k] += r22*(-A**2*G**2*L**4*N*r22*scf**2/(10*A**2*G**2*L**4*scf**2 - 120*A*E*G*Iyy*L**2*scf - 120*A*E*G*Izz*L**2*scf + 1440*E**2*Iyy*Izz) - A**2*G**2*L**4*N*r23*scf**2/(10*A**2*G**2*L**4*scf**2 - 240*A*E*G*Iyy*L**2*scf + 1440*E**2*Iyy**2)) + r23*(-A**2*G**2*L**4*N*r22*scf**2/(10*A**2*G**2*L**4*scf**2 - 240*A*E*G*Izz*L**2*scf + 1440*E**2*Izz**2) - A**2*G**2*L**4*N*r23*scf**2/(10*A**2*G**2*L**4*scf**2 - 120*A*E*G*Iyy*L**2*scf - 120*A*E*G*Izz*L**2*scf + 1440*E**2*Iyy*Izz))
            k += 1
            KGv[k] += r32*(-A**2*G**2*L**4*N*r22*scf**2/(10*A**2*G**2*L**4*scf**2 - 120*A*E*G*Iyy*L**2*scf - 120*A*E*G*Izz*L**2*scf + 1440*E**2*Iyy*Izz) - A**2*G**2*L**4*N*r23*scf**2/(10*A**2*G**2*L**4*scf**2 - 240*A*E*G*Iyy*L**2*scf + 1440*E**2*Iyy**2)) + r33*(-A**2*G**2*L**4*N*r22*scf**2/(10*A**2*G**2*L**4*scf**2 - 240*A*E*G*Izz*L**2*scf + 1440*E**2*Izz**2) - A**2*G**2*L**4*N*r23*scf**2/(10*A**2*G**2*L**4*scf**2 - 120*A*E*G*Iyy*L**2*scf - 120*A*E*G*Izz*L**2*scf + 1440*E**2*Iyy*Izz))
            k += 1
            KGv[k] += r12*(-6*N*r32*(A**2*G**2*L**4*scf**2 - 20*A*E*G*Izz*L**2*scf + 120*E**2*Izz**2)/(5*L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)) - 6*N*r33*(A**2*G**2*L**4*scf**2 - 10*A*E*G*Iyy*L**2*scf - 10*A*E*G*Izz*L**2*scf + 120*E**2*Iyy*Izz)/(5*L*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz))) + r13*(-6*N*r32*(A**2*G**2*L**4*scf**2 - 10*A*E*G*Iyy*L**2*scf - 10*A*E*G*Izz*L**2*scf + 120*E**2*Iyy*Izz)/(5*L*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) - 6*N*r33*(A**2*G**2*L**4*scf**2 - 20*A*E*G*Iyy*L**2*scf + 120*E**2*Iyy**2)/(5*L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)))
            k += 1
            KGv[k] += r22*(-6*N*r32*(A**2*G**2*L**4*scf**2 - 20*A*E*G*Izz*L**2*scf + 120*E**2*Izz**2)/(5*L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)) - 6*N*r33*(A**2*G**2*L**4*scf**2 - 10*A*E*G*Iyy*L**2*scf - 10*A*E*G*Izz*L**2*scf + 120*E**2*Iyy*Izz)/(5*L*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz))) + r23*(-6*N*r32*(A**2*G**2*L**4*scf**2 - 10*A*E*G*Iyy*L**2*scf - 10*A*E*G*Izz*L**2*scf + 120*E**2*Iyy*Izz)/(5*L*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) - 6*N*r33*(A**2*G**2*L**4*scf**2 - 20*A*E*G*Iyy*L**2*scf + 120*E**2*Iyy**2)/(5*L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)))
            k += 1
            KGv[k] += r32*(-6*N*r32*(A**2*G**2*L**4*scf**2 - 20*A*E*G*Izz*L**2*scf + 120*E**2*Izz**2)/(5*L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)) - 6*N*r33*(A**2*G**2*L**4*scf**2 - 10*A*E*G*Iyy*L**2*scf - 10*A*E*G*Izz*L**2*scf + 120*E**2*Iyy*Izz)/(5*L*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz))) + r33*(-6*N*r32*(A**2*G**2*L**4*scf**2 - 10*A*E*G*Iyy*L**2*scf - 10*A*E*G*Izz*L**2*scf + 120*E**2*Iyy*Izz)/(5*L*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) - 6*N*r33*(A**2*G**2*L**4*scf**2 - 20*A*E*G*Iyy*L**2*scf + 120*E**2*Iyy**2)/(5*L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)))
            k += 1
            KGv[k] += r12*(-A**2*G**2*L**4*N*r32*scf**2/(10*A**2*G**2*L**4*scf**2 - 120*A*E*G*Iyy*L**2*scf - 120*A*E*G*Izz*L**2*scf + 1440*E**2*Iyy*Izz) - A**2*G**2*L**4*N*r33*scf**2/(10*A**2*G**2*L**4*scf**2 - 240*A*E*G*Iyy*L**2*scf + 1440*E**2*Iyy**2)) + r13*(-A**2*G**2*L**4*N*r32*scf**2/(10*A**2*G**2*L**4*scf**2 - 240*A*E*G*Izz*L**2*scf + 1440*E**2*Izz**2) - A**2*G**2*L**4*N*r33*scf**2/(10*A**2*G**2*L**4*scf**2 - 120*A*E*G*Iyy*L**2*scf - 120*A*E*G*Izz*L**2*scf + 1440*E**2*Iyy*Izz))
            k += 1
            KGv[k] += r22*(-A**2*G**2*L**4*N*r32*scf**2/(10*A**2*G**2*L**4*scf**2 - 120*A*E*G*Iyy*L**2*scf - 120*A*E*G*Izz*L**2*scf + 1440*E**2*Iyy*Izz) - A**2*G**2*L**4*N*r33*scf**2/(10*A**2*G**2*L**4*scf**2 - 240*A*E*G*Iyy*L**2*scf + 1440*E**2*Iyy**2)) + r23*(-A**2*G**2*L**4*N*r32*scf**2/(10*A**2*G**2*L**4*scf**2 - 240*A*E*G*Izz*L**2*scf + 1440*E**2*Izz**2) - A**2*G**2*L**4*N*r33*scf**2/(10*A**2*G**2*L**4*scf**2 - 120*A*E*G*Iyy*L**2*scf - 120*A*E*G*Izz*L**2*scf + 1440*E**2*Iyy*Izz))
            k += 1
            KGv[k] += r32*(-A**2*G**2*L**4*N*r32*scf**2/(10*A**2*G**2*L**4*scf**2 - 120*A*E*G*Iyy*L**2*scf - 120*A*E*G*Izz*L**2*scf + 1440*E**2*Iyy*Izz) - A**2*G**2*L**4*N*r33*scf**2/(10*A**2*G**2*L**4*scf**2 - 240*A*E*G*Iyy*L**2*scf + 1440*E**2*Iyy**2)) + r33*(-A**2*G**2*L**4*N*r32*scf**2/(10*A**2*G**2*L**4*scf**2 - 240*A*E*G*Izz*L**2*scf + 1440*E**2*Izz**2) - A**2*G**2*L**4*N*r33*scf**2/(10*A**2*G**2*L**4*scf**2 - 120*A*E*G*Iyy*L**2*scf - 120*A*E*G*Izz*L**2*scf + 1440*E**2*Iyy*Izz))
            k += 1
            KGv[k] += r12*(6*N*r32*(A**2*G**2*L**4*scf**2 - 20*A*E*G*Izz*L**2*scf + 120*E**2*Izz**2)/(5*L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)) + 6*N*r33*(A**2*G**2*L**4*scf**2 - 10*A*E*G*Iyy*L**2*scf - 10*A*E*G*Izz*L**2*scf + 120*E**2*Iyy*Izz)/(5*L*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz))) + r13*(6*N*r32*(A**2*G**2*L**4*scf**2 - 10*A*E*G*Iyy*L**2*scf - 10*A*E*G*Izz*L**2*scf + 120*E**2*Iyy*Izz)/(5*L*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + 6*N*r33*(A**2*G**2*L**4*scf**2 - 20*A*E*G*Iyy*L**2*scf + 120*E**2*Iyy**2)/(5*L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)))
            k += 1
            KGv[k] += r22*(6*N*r32*(A**2*G**2*L**4*scf**2 - 20*A*E*G*Izz*L**2*scf + 120*E**2*Izz**2)/(5*L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)) + 6*N*r33*(A**2*G**2*L**4*scf**2 - 10*A*E*G*Iyy*L**2*scf - 10*A*E*G*Izz*L**2*scf + 120*E**2*Iyy*Izz)/(5*L*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz))) + r23*(6*N*r32*(A**2*G**2*L**4*scf**2 - 10*A*E*G*Iyy*L**2*scf - 10*A*E*G*Izz*L**2*scf + 120*E**2*Iyy*Izz)/(5*L*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + 6*N*r33*(A**2*G**2*L**4*scf**2 - 20*A*E*G*Iyy*L**2*scf + 120*E**2*Iyy**2)/(5*L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)))
            k += 1
            KGv[k] += r32*(6*N*r32*(A**2*G**2*L**4*scf**2 - 20*A*E*G*Izz*L**2*scf + 120*E**2*Izz**2)/(5*L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)) + 6*N*r33*(A**2*G**2*L**4*scf**2 - 10*A*E*G*Iyy*L**2*scf - 10*A*E*G*Izz*L**2*scf + 120*E**2*Iyy*Izz)/(5*L*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz))) + r33*(6*N*r32*(A**2*G**2*L**4*scf**2 - 10*A*E*G*Iyy*L**2*scf - 10*A*E*G*Izz*L**2*scf + 120*E**2*Iyy*Izz)/(5*L*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + 6*N*r33*(A**2*G**2*L**4*scf**2 - 20*A*E*G*Iyy*L**2*scf + 120*E**2*Iyy**2)/(5*L*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)))
            k += 1
            KGv[k] += r12*(-A**2*G**2*L**4*N*r32*scf**2/(10*A**2*G**2*L**4*scf**2 - 120*A*E*G*Iyy*L**2*scf - 120*A*E*G*Izz*L**2*scf + 1440*E**2*Iyy*Izz) - A**2*G**2*L**4*N*r33*scf**2/(10*A**2*G**2*L**4*scf**2 - 240*A*E*G*Iyy*L**2*scf + 1440*E**2*Iyy**2)) + r13*(-A**2*G**2*L**4*N*r32*scf**2/(10*A**2*G**2*L**4*scf**2 - 240*A*E*G*Izz*L**2*scf + 1440*E**2*Izz**2) - A**2*G**2*L**4*N*r33*scf**2/(10*A**2*G**2*L**4*scf**2 - 120*A*E*G*Iyy*L**2*scf - 120*A*E*G*Izz*L**2*scf + 1440*E**2*Iyy*Izz))
            k += 1
            KGv[k] += r22*(-A**2*G**2*L**4*N*r32*scf**2/(10*A**2*G**2*L**4*scf**2 - 120*A*E*G*Iyy*L**2*scf - 120*A*E*G*Izz*L**2*scf + 1440*E**2*Iyy*Izz) - A**2*G**2*L**4*N*r33*scf**2/(10*A**2*G**2*L**4*scf**2 - 240*A*E*G*Iyy*L**2*scf + 1440*E**2*Iyy**2)) + r23*(-A**2*G**2*L**4*N*r32*scf**2/(10*A**2*G**2*L**4*scf**2 - 240*A*E*G*Izz*L**2*scf + 1440*E**2*Izz**2) - A**2*G**2*L**4*N*r33*scf**2/(10*A**2*G**2*L**4*scf**2 - 120*A*E*G*Iyy*L**2*scf - 120*A*E*G*Izz*L**2*scf + 1440*E**2*Iyy*Izz))
            k += 1
            KGv[k] += r32*(-A**2*G**2*L**4*N*r32*scf**2/(10*A**2*G**2*L**4*scf**2 - 120*A*E*G*Iyy*L**2*scf - 120*A*E*G*Izz*L**2*scf + 1440*E**2*Iyy*Izz) - A**2*G**2*L**4*N*r33*scf**2/(10*A**2*G**2*L**4*scf**2 - 240*A*E*G*Iyy*L**2*scf + 1440*E**2*Iyy**2)) + r33*(-A**2*G**2*L**4*N*r32*scf**2/(10*A**2*G**2*L**4*scf**2 - 240*A*E*G*Izz*L**2*scf + 1440*E**2*Izz**2) - A**2*G**2*L**4*N*r33*scf**2/(10*A**2*G**2*L**4*scf**2 - 120*A*E*G*Iyy*L**2*scf - 120*A*E*G*Izz*L**2*scf + 1440*E**2*Iyy*Izz))
            k += 1
            KGv[k] += r12*(A**2*G**2*L**4*N*r12*scf**2/(10*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + A**2*G**2*L**4*N*r13*scf**2/(10*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2))) + r13*(A**2*G**2*L**4*N*r12*scf**2/(10*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)) + A**2*G**2*L**4*N*r13*scf**2/(10*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)))
            k += 1
            KGv[k] += r22*(A**2*G**2*L**4*N*r12*scf**2/(10*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + A**2*G**2*L**4*N*r13*scf**2/(10*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2))) + r23*(A**2*G**2*L**4*N*r12*scf**2/(10*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)) + A**2*G**2*L**4*N*r13*scf**2/(10*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)))
            k += 1
            KGv[k] += r32*(A**2*G**2*L**4*N*r12*scf**2/(10*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + A**2*G**2*L**4*N*r13*scf**2/(10*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2))) + r33*(A**2*G**2*L**4*N*r12*scf**2/(10*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)) + A**2*G**2*L**4*N*r13*scf**2/(10*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)))
            k += 1
            KGv[k] += r12*(-L*N*r12*(A**2*G**2*L**4*scf**2 - 60*A*E*G*Iyy*L**2*scf + 360*E**2*Iyy**2)/(30*A**2*G**2*L**4*scf**2 - 720*A*E*G*Iyy*L**2*scf + 4320*E**2*Iyy**2) - L*N*r13*(A**2*G**2*L**4*scf**2 - 30*A*E*G*Iyy*L**2*scf - 30*A*E*G*Izz*L**2*scf + 360*E**2*Iyy*Izz)/(30*A**2*G**2*L**4*scf**2 - 360*A*E*G*Iyy*L**2*scf - 360*A*E*G*Izz*L**2*scf + 4320*E**2*Iyy*Izz)) + r13*(-L*N*r12*(A**2*G**2*L**4*scf**2 - 30*A*E*G*Iyy*L**2*scf - 30*A*E*G*Izz*L**2*scf + 360*E**2*Iyy*Izz)/(30*A**2*G**2*L**4*scf**2 - 360*A*E*G*Iyy*L**2*scf - 360*A*E*G*Izz*L**2*scf + 4320*E**2*Iyy*Izz) - L*N*r13*(A**2*G**2*L**4*scf**2 - 60*A*E*G*Izz*L**2*scf + 360*E**2*Izz**2)/(30*A**2*G**2*L**4*scf**2 - 720*A*E*G*Izz*L**2*scf + 4320*E**2*Izz**2))
            k += 1
            KGv[k] += r22*(-L*N*r12*(A**2*G**2*L**4*scf**2 - 60*A*E*G*Iyy*L**2*scf + 360*E**2*Iyy**2)/(30*A**2*G**2*L**4*scf**2 - 720*A*E*G*Iyy*L**2*scf + 4320*E**2*Iyy**2) - L*N*r13*(A**2*G**2*L**4*scf**2 - 30*A*E*G*Iyy*L**2*scf - 30*A*E*G*Izz*L**2*scf + 360*E**2*Iyy*Izz)/(30*A**2*G**2*L**4*scf**2 - 360*A*E*G*Iyy*L**2*scf - 360*A*E*G*Izz*L**2*scf + 4320*E**2*Iyy*Izz)) + r23*(-L*N*r12*(A**2*G**2*L**4*scf**2 - 30*A*E*G*Iyy*L**2*scf - 30*A*E*G*Izz*L**2*scf + 360*E**2*Iyy*Izz)/(30*A**2*G**2*L**4*scf**2 - 360*A*E*G*Iyy*L**2*scf - 360*A*E*G*Izz*L**2*scf + 4320*E**2*Iyy*Izz) - L*N*r13*(A**2*G**2*L**4*scf**2 - 60*A*E*G*Izz*L**2*scf + 360*E**2*Izz**2)/(30*A**2*G**2*L**4*scf**2 - 720*A*E*G*Izz*L**2*scf + 4320*E**2*Izz**2))
            k += 1
            KGv[k] += r32*(-L*N*r12*(A**2*G**2*L**4*scf**2 - 60*A*E*G*Iyy*L**2*scf + 360*E**2*Iyy**2)/(30*A**2*G**2*L**4*scf**2 - 720*A*E*G*Iyy*L**2*scf + 4320*E**2*Iyy**2) - L*N*r13*(A**2*G**2*L**4*scf**2 - 30*A*E*G*Iyy*L**2*scf - 30*A*E*G*Izz*L**2*scf + 360*E**2*Iyy*Izz)/(30*A**2*G**2*L**4*scf**2 - 360*A*E*G*Iyy*L**2*scf - 360*A*E*G*Izz*L**2*scf + 4320*E**2*Iyy*Izz)) + r33*(-L*N*r12*(A**2*G**2*L**4*scf**2 - 30*A*E*G*Iyy*L**2*scf - 30*A*E*G*Izz*L**2*scf + 360*E**2*Iyy*Izz)/(30*A**2*G**2*L**4*scf**2 - 360*A*E*G*Iyy*L**2*scf - 360*A*E*G*Izz*L**2*scf + 4320*E**2*Iyy*Izz) - L*N*r13*(A**2*G**2*L**4*scf**2 - 60*A*E*G*Izz*L**2*scf + 360*E**2*Izz**2)/(30*A**2*G**2*L**4*scf**2 - 720*A*E*G*Izz*L**2*scf + 4320*E**2*Izz**2))
            k += 1
            KGv[k] += r12*(-A**2*G**2*L**4*N*r12*scf**2/(10*A**2*G**2*L**4*scf**2 - 120*A*E*G*Iyy*L**2*scf - 120*A*E*G*Izz*L**2*scf + 1440*E**2*Iyy*Izz) - A**2*G**2*L**4*N*r13*scf**2/(10*A**2*G**2*L**4*scf**2 - 240*A*E*G*Izz*L**2*scf + 1440*E**2*Izz**2)) + r13*(-A**2*G**2*L**4*N*r12*scf**2/(10*A**2*G**2*L**4*scf**2 - 240*A*E*G*Iyy*L**2*scf + 1440*E**2*Iyy**2) - A**2*G**2*L**4*N*r13*scf**2/(10*A**2*G**2*L**4*scf**2 - 120*A*E*G*Iyy*L**2*scf - 120*A*E*G*Izz*L**2*scf + 1440*E**2*Iyy*Izz))
            k += 1
            KGv[k] += r22*(-A**2*G**2*L**4*N*r12*scf**2/(10*A**2*G**2*L**4*scf**2 - 120*A*E*G*Iyy*L**2*scf - 120*A*E*G*Izz*L**2*scf + 1440*E**2*Iyy*Izz) - A**2*G**2*L**4*N*r13*scf**2/(10*A**2*G**2*L**4*scf**2 - 240*A*E*G*Izz*L**2*scf + 1440*E**2*Izz**2)) + r23*(-A**2*G**2*L**4*N*r12*scf**2/(10*A**2*G**2*L**4*scf**2 - 240*A*E*G*Iyy*L**2*scf + 1440*E**2*Iyy**2) - A**2*G**2*L**4*N*r13*scf**2/(10*A**2*G**2*L**4*scf**2 - 120*A*E*G*Iyy*L**2*scf - 120*A*E*G*Izz*L**2*scf + 1440*E**2*Iyy*Izz))
            k += 1
            KGv[k] += r32*(-A**2*G**2*L**4*N*r12*scf**2/(10*A**2*G**2*L**4*scf**2 - 120*A*E*G*Iyy*L**2*scf - 120*A*E*G*Izz*L**2*scf + 1440*E**2*Iyy*Izz) - A**2*G**2*L**4*N*r13*scf**2/(10*A**2*G**2*L**4*scf**2 - 240*A*E*G*Izz*L**2*scf + 1440*E**2*Izz**2)) + r33*(-A**2*G**2*L**4*N*r12*scf**2/(10*A**2*G**2*L**4*scf**2 - 240*A*E*G*Iyy*L**2*scf + 1440*E**2*Iyy**2) - A**2*G**2*L**4*N*r13*scf**2/(10*A**2*G**2*L**4*scf**2 - 120*A*E*G*Iyy*L**2*scf - 120*A*E*G*Izz*L**2*scf + 1440*E**2*Iyy*Izz))
            k += 1
            KGv[k] += r12*(2*L*N*r12*(A**2*G**2*L**4*scf**2 - 15*A*E*G*Iyy*L**2*scf + 90*E**2*Iyy**2)/(15*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)) + L*N*r13*(2*A**2*G**2*L**4*scf**2 - 15*A*E*G*Iyy*L**2*scf - 15*A*E*G*Izz*L**2*scf + 180*E**2*Iyy*Izz)/(15*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz))) + r13*(L*N*r12*(2*A**2*G**2*L**4*scf**2 - 15*A*E*G*Iyy*L**2*scf - 15*A*E*G*Izz*L**2*scf + 180*E**2*Iyy*Izz)/(15*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + 2*L*N*r13*(A**2*G**2*L**4*scf**2 - 15*A*E*G*Izz*L**2*scf + 90*E**2*Izz**2)/(15*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)))
            k += 1
            KGv[k] += r22*(2*L*N*r12*(A**2*G**2*L**4*scf**2 - 15*A*E*G*Iyy*L**2*scf + 90*E**2*Iyy**2)/(15*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)) + L*N*r13*(2*A**2*G**2*L**4*scf**2 - 15*A*E*G*Iyy*L**2*scf - 15*A*E*G*Izz*L**2*scf + 180*E**2*Iyy*Izz)/(15*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz))) + r23*(L*N*r12*(2*A**2*G**2*L**4*scf**2 - 15*A*E*G*Iyy*L**2*scf - 15*A*E*G*Izz*L**2*scf + 180*E**2*Iyy*Izz)/(15*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + 2*L*N*r13*(A**2*G**2*L**4*scf**2 - 15*A*E*G*Izz*L**2*scf + 90*E**2*Izz**2)/(15*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)))
            k += 1
            KGv[k] += r32*(2*L*N*r12*(A**2*G**2*L**4*scf**2 - 15*A*E*G*Iyy*L**2*scf + 90*E**2*Iyy**2)/(15*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)) + L*N*r13*(2*A**2*G**2*L**4*scf**2 - 15*A*E*G*Iyy*L**2*scf - 15*A*E*G*Izz*L**2*scf + 180*E**2*Iyy*Izz)/(15*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz))) + r33*(L*N*r12*(2*A**2*G**2*L**4*scf**2 - 15*A*E*G*Iyy*L**2*scf - 15*A*E*G*Izz*L**2*scf + 180*E**2*Iyy*Izz)/(15*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + 2*L*N*r13*(A**2*G**2*L**4*scf**2 - 15*A*E*G*Izz*L**2*scf + 90*E**2*Izz**2)/(15*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)))
            k += 1
            KGv[k] += r12*(A**2*G**2*L**4*N*r22*scf**2/(10*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + A**2*G**2*L**4*N*r23*scf**2/(10*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2))) + r13*(A**2*G**2*L**4*N*r22*scf**2/(10*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)) + A**2*G**2*L**4*N*r23*scf**2/(10*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)))
            k += 1
            KGv[k] += r22*(A**2*G**2*L**4*N*r22*scf**2/(10*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + A**2*G**2*L**4*N*r23*scf**2/(10*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2))) + r23*(A**2*G**2*L**4*N*r22*scf**2/(10*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)) + A**2*G**2*L**4*N*r23*scf**2/(10*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)))
            k += 1
            KGv[k] += r32*(A**2*G**2*L**4*N*r22*scf**2/(10*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + A**2*G**2*L**4*N*r23*scf**2/(10*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2))) + r33*(A**2*G**2*L**4*N*r22*scf**2/(10*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)) + A**2*G**2*L**4*N*r23*scf**2/(10*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)))
            k += 1
            KGv[k] += r12*(-L*N*r22*(A**2*G**2*L**4*scf**2 - 60*A*E*G*Iyy*L**2*scf + 360*E**2*Iyy**2)/(30*A**2*G**2*L**4*scf**2 - 720*A*E*G*Iyy*L**2*scf + 4320*E**2*Iyy**2) - L*N*r23*(A**2*G**2*L**4*scf**2 - 30*A*E*G*Iyy*L**2*scf - 30*A*E*G*Izz*L**2*scf + 360*E**2*Iyy*Izz)/(30*A**2*G**2*L**4*scf**2 - 360*A*E*G*Iyy*L**2*scf - 360*A*E*G*Izz*L**2*scf + 4320*E**2*Iyy*Izz)) + r13*(-L*N*r22*(A**2*G**2*L**4*scf**2 - 30*A*E*G*Iyy*L**2*scf - 30*A*E*G*Izz*L**2*scf + 360*E**2*Iyy*Izz)/(30*A**2*G**2*L**4*scf**2 - 360*A*E*G*Iyy*L**2*scf - 360*A*E*G*Izz*L**2*scf + 4320*E**2*Iyy*Izz) - L*N*r23*(A**2*G**2*L**4*scf**2 - 60*A*E*G*Izz*L**2*scf + 360*E**2*Izz**2)/(30*A**2*G**2*L**4*scf**2 - 720*A*E*G*Izz*L**2*scf + 4320*E**2*Izz**2))
            k += 1
            KGv[k] += r22*(-L*N*r22*(A**2*G**2*L**4*scf**2 - 60*A*E*G*Iyy*L**2*scf + 360*E**2*Iyy**2)/(30*A**2*G**2*L**4*scf**2 - 720*A*E*G*Iyy*L**2*scf + 4320*E**2*Iyy**2) - L*N*r23*(A**2*G**2*L**4*scf**2 - 30*A*E*G*Iyy*L**2*scf - 30*A*E*G*Izz*L**2*scf + 360*E**2*Iyy*Izz)/(30*A**2*G**2*L**4*scf**2 - 360*A*E*G*Iyy*L**2*scf - 360*A*E*G*Izz*L**2*scf + 4320*E**2*Iyy*Izz)) + r23*(-L*N*r22*(A**2*G**2*L**4*scf**2 - 30*A*E*G*Iyy*L**2*scf - 30*A*E*G*Izz*L**2*scf + 360*E**2*Iyy*Izz)/(30*A**2*G**2*L**4*scf**2 - 360*A*E*G*Iyy*L**2*scf - 360*A*E*G*Izz*L**2*scf + 4320*E**2*Iyy*Izz) - L*N*r23*(A**2*G**2*L**4*scf**2 - 60*A*E*G*Izz*L**2*scf + 360*E**2*Izz**2)/(30*A**2*G**2*L**4*scf**2 - 720*A*E*G*Izz*L**2*scf + 4320*E**2*Izz**2))
            k += 1
            KGv[k] += r32*(-L*N*r22*(A**2*G**2*L**4*scf**2 - 60*A*E*G*Iyy*L**2*scf + 360*E**2*Iyy**2)/(30*A**2*G**2*L**4*scf**2 - 720*A*E*G*Iyy*L**2*scf + 4320*E**2*Iyy**2) - L*N*r23*(A**2*G**2*L**4*scf**2 - 30*A*E*G*Iyy*L**2*scf - 30*A*E*G*Izz*L**2*scf + 360*E**2*Iyy*Izz)/(30*A**2*G**2*L**4*scf**2 - 360*A*E*G*Iyy*L**2*scf - 360*A*E*G*Izz*L**2*scf + 4320*E**2*Iyy*Izz)) + r33*(-L*N*r22*(A**2*G**2*L**4*scf**2 - 30*A*E*G*Iyy*L**2*scf - 30*A*E*G*Izz*L**2*scf + 360*E**2*Iyy*Izz)/(30*A**2*G**2*L**4*scf**2 - 360*A*E*G*Iyy*L**2*scf - 360*A*E*G*Izz*L**2*scf + 4320*E**2*Iyy*Izz) - L*N*r23*(A**2*G**2*L**4*scf**2 - 60*A*E*G*Izz*L**2*scf + 360*E**2*Izz**2)/(30*A**2*G**2*L**4*scf**2 - 720*A*E*G*Izz*L**2*scf + 4320*E**2*Izz**2))
            k += 1
            KGv[k] += r12*(-A**2*G**2*L**4*N*r22*scf**2/(10*A**2*G**2*L**4*scf**2 - 120*A*E*G*Iyy*L**2*scf - 120*A*E*G*Izz*L**2*scf + 1440*E**2*Iyy*Izz) - A**2*G**2*L**4*N*r23*scf**2/(10*A**2*G**2*L**4*scf**2 - 240*A*E*G*Izz*L**2*scf + 1440*E**2*Izz**2)) + r13*(-A**2*G**2*L**4*N*r22*scf**2/(10*A**2*G**2*L**4*scf**2 - 240*A*E*G*Iyy*L**2*scf + 1440*E**2*Iyy**2) - A**2*G**2*L**4*N*r23*scf**2/(10*A**2*G**2*L**4*scf**2 - 120*A*E*G*Iyy*L**2*scf - 120*A*E*G*Izz*L**2*scf + 1440*E**2*Iyy*Izz))
            k += 1
            KGv[k] += r22*(-A**2*G**2*L**4*N*r22*scf**2/(10*A**2*G**2*L**4*scf**2 - 120*A*E*G*Iyy*L**2*scf - 120*A*E*G*Izz*L**2*scf + 1440*E**2*Iyy*Izz) - A**2*G**2*L**4*N*r23*scf**2/(10*A**2*G**2*L**4*scf**2 - 240*A*E*G*Izz*L**2*scf + 1440*E**2*Izz**2)) + r23*(-A**2*G**2*L**4*N*r22*scf**2/(10*A**2*G**2*L**4*scf**2 - 240*A*E*G*Iyy*L**2*scf + 1440*E**2*Iyy**2) - A**2*G**2*L**4*N*r23*scf**2/(10*A**2*G**2*L**4*scf**2 - 120*A*E*G*Iyy*L**2*scf - 120*A*E*G*Izz*L**2*scf + 1440*E**2*Iyy*Izz))
            k += 1
            KGv[k] += r32*(-A**2*G**2*L**4*N*r22*scf**2/(10*A**2*G**2*L**4*scf**2 - 120*A*E*G*Iyy*L**2*scf - 120*A*E*G*Izz*L**2*scf + 1440*E**2*Iyy*Izz) - A**2*G**2*L**4*N*r23*scf**2/(10*A**2*G**2*L**4*scf**2 - 240*A*E*G*Izz*L**2*scf + 1440*E**2*Izz**2)) + r33*(-A**2*G**2*L**4*N*r22*scf**2/(10*A**2*G**2*L**4*scf**2 - 240*A*E*G*Iyy*L**2*scf + 1440*E**2*Iyy**2) - A**2*G**2*L**4*N*r23*scf**2/(10*A**2*G**2*L**4*scf**2 - 120*A*E*G*Iyy*L**2*scf - 120*A*E*G*Izz*L**2*scf + 1440*E**2*Iyy*Izz))
            k += 1
            KGv[k] += r12*(2*L*N*r22*(A**2*G**2*L**4*scf**2 - 15*A*E*G*Iyy*L**2*scf + 90*E**2*Iyy**2)/(15*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)) + L*N*r23*(2*A**2*G**2*L**4*scf**2 - 15*A*E*G*Iyy*L**2*scf - 15*A*E*G*Izz*L**2*scf + 180*E**2*Iyy*Izz)/(15*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz))) + r13*(L*N*r22*(2*A**2*G**2*L**4*scf**2 - 15*A*E*G*Iyy*L**2*scf - 15*A*E*G*Izz*L**2*scf + 180*E**2*Iyy*Izz)/(15*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + 2*L*N*r23*(A**2*G**2*L**4*scf**2 - 15*A*E*G*Izz*L**2*scf + 90*E**2*Izz**2)/(15*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)))
            k += 1
            KGv[k] += r22*(2*L*N*r22*(A**2*G**2*L**4*scf**2 - 15*A*E*G*Iyy*L**2*scf + 90*E**2*Iyy**2)/(15*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)) + L*N*r23*(2*A**2*G**2*L**4*scf**2 - 15*A*E*G*Iyy*L**2*scf - 15*A*E*G*Izz*L**2*scf + 180*E**2*Iyy*Izz)/(15*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz))) + r23*(L*N*r22*(2*A**2*G**2*L**4*scf**2 - 15*A*E*G*Iyy*L**2*scf - 15*A*E*G*Izz*L**2*scf + 180*E**2*Iyy*Izz)/(15*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + 2*L*N*r23*(A**2*G**2*L**4*scf**2 - 15*A*E*G*Izz*L**2*scf + 90*E**2*Izz**2)/(15*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)))
            k += 1
            KGv[k] += r32*(2*L*N*r22*(A**2*G**2*L**4*scf**2 - 15*A*E*G*Iyy*L**2*scf + 90*E**2*Iyy**2)/(15*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)) + L*N*r23*(2*A**2*G**2*L**4*scf**2 - 15*A*E*G*Iyy*L**2*scf - 15*A*E*G*Izz*L**2*scf + 180*E**2*Iyy*Izz)/(15*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz))) + r33*(L*N*r22*(2*A**2*G**2*L**4*scf**2 - 15*A*E*G*Iyy*L**2*scf - 15*A*E*G*Izz*L**2*scf + 180*E**2*Iyy*Izz)/(15*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + 2*L*N*r23*(A**2*G**2*L**4*scf**2 - 15*A*E*G*Izz*L**2*scf + 90*E**2*Izz**2)/(15*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)))
            k += 1
            KGv[k] += r12*(A**2*G**2*L**4*N*r32*scf**2/(10*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + A**2*G**2*L**4*N*r33*scf**2/(10*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2))) + r13*(A**2*G**2*L**4*N*r32*scf**2/(10*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)) + A**2*G**2*L**4*N*r33*scf**2/(10*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)))
            k += 1
            KGv[k] += r22*(A**2*G**2*L**4*N*r32*scf**2/(10*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + A**2*G**2*L**4*N*r33*scf**2/(10*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2))) + r23*(A**2*G**2*L**4*N*r32*scf**2/(10*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)) + A**2*G**2*L**4*N*r33*scf**2/(10*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)))
            k += 1
            KGv[k] += r32*(A**2*G**2*L**4*N*r32*scf**2/(10*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + A**2*G**2*L**4*N*r33*scf**2/(10*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2))) + r33*(A**2*G**2*L**4*N*r32*scf**2/(10*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)) + A**2*G**2*L**4*N*r33*scf**2/(10*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)))
            k += 1
            KGv[k] += r12*(-L*N*r32*(A**2*G**2*L**4*scf**2 - 60*A*E*G*Iyy*L**2*scf + 360*E**2*Iyy**2)/(30*A**2*G**2*L**4*scf**2 - 720*A*E*G*Iyy*L**2*scf + 4320*E**2*Iyy**2) - L*N*r33*(A**2*G**2*L**4*scf**2 - 30*A*E*G*Iyy*L**2*scf - 30*A*E*G*Izz*L**2*scf + 360*E**2*Iyy*Izz)/(30*A**2*G**2*L**4*scf**2 - 360*A*E*G*Iyy*L**2*scf - 360*A*E*G*Izz*L**2*scf + 4320*E**2*Iyy*Izz)) + r13*(-L*N*r32*(A**2*G**2*L**4*scf**2 - 30*A*E*G*Iyy*L**2*scf - 30*A*E*G*Izz*L**2*scf + 360*E**2*Iyy*Izz)/(30*A**2*G**2*L**4*scf**2 - 360*A*E*G*Iyy*L**2*scf - 360*A*E*G*Izz*L**2*scf + 4320*E**2*Iyy*Izz) - L*N*r33*(A**2*G**2*L**4*scf**2 - 60*A*E*G*Izz*L**2*scf + 360*E**2*Izz**2)/(30*A**2*G**2*L**4*scf**2 - 720*A*E*G*Izz*L**2*scf + 4320*E**2*Izz**2))
            k += 1
            KGv[k] += r22*(-L*N*r32*(A**2*G**2*L**4*scf**2 - 60*A*E*G*Iyy*L**2*scf + 360*E**2*Iyy**2)/(30*A**2*G**2*L**4*scf**2 - 720*A*E*G*Iyy*L**2*scf + 4320*E**2*Iyy**2) - L*N*r33*(A**2*G**2*L**4*scf**2 - 30*A*E*G*Iyy*L**2*scf - 30*A*E*G*Izz*L**2*scf + 360*E**2*Iyy*Izz)/(30*A**2*G**2*L**4*scf**2 - 360*A*E*G*Iyy*L**2*scf - 360*A*E*G*Izz*L**2*scf + 4320*E**2*Iyy*Izz)) + r23*(-L*N*r32*(A**2*G**2*L**4*scf**2 - 30*A*E*G*Iyy*L**2*scf - 30*A*E*G*Izz*L**2*scf + 360*E**2*Iyy*Izz)/(30*A**2*G**2*L**4*scf**2 - 360*A*E*G*Iyy*L**2*scf - 360*A*E*G*Izz*L**2*scf + 4320*E**2*Iyy*Izz) - L*N*r33*(A**2*G**2*L**4*scf**2 - 60*A*E*G*Izz*L**2*scf + 360*E**2*Izz**2)/(30*A**2*G**2*L**4*scf**2 - 720*A*E*G*Izz*L**2*scf + 4320*E**2*Izz**2))
            k += 1
            KGv[k] += r32*(-L*N*r32*(A**2*G**2*L**4*scf**2 - 60*A*E*G*Iyy*L**2*scf + 360*E**2*Iyy**2)/(30*A**2*G**2*L**4*scf**2 - 720*A*E*G*Iyy*L**2*scf + 4320*E**2*Iyy**2) - L*N*r33*(A**2*G**2*L**4*scf**2 - 30*A*E*G*Iyy*L**2*scf - 30*A*E*G*Izz*L**2*scf + 360*E**2*Iyy*Izz)/(30*A**2*G**2*L**4*scf**2 - 360*A*E*G*Iyy*L**2*scf - 360*A*E*G*Izz*L**2*scf + 4320*E**2*Iyy*Izz)) + r33*(-L*N*r32*(A**2*G**2*L**4*scf**2 - 30*A*E*G*Iyy*L**2*scf - 30*A*E*G*Izz*L**2*scf + 360*E**2*Iyy*Izz)/(30*A**2*G**2*L**4*scf**2 - 360*A*E*G*Iyy*L**2*scf - 360*A*E*G*Izz*L**2*scf + 4320*E**2*Iyy*Izz) - L*N*r33*(A**2*G**2*L**4*scf**2 - 60*A*E*G*Izz*L**2*scf + 360*E**2*Izz**2)/(30*A**2*G**2*L**4*scf**2 - 720*A*E*G*Izz*L**2*scf + 4320*E**2*Izz**2))
            k += 1
            KGv[k] += r12*(-A**2*G**2*L**4*N*r32*scf**2/(10*A**2*G**2*L**4*scf**2 - 120*A*E*G*Iyy*L**2*scf - 120*A*E*G*Izz*L**2*scf + 1440*E**2*Iyy*Izz) - A**2*G**2*L**4*N*r33*scf**2/(10*A**2*G**2*L**4*scf**2 - 240*A*E*G*Izz*L**2*scf + 1440*E**2*Izz**2)) + r13*(-A**2*G**2*L**4*N*r32*scf**2/(10*A**2*G**2*L**4*scf**2 - 240*A*E*G*Iyy*L**2*scf + 1440*E**2*Iyy**2) - A**2*G**2*L**4*N*r33*scf**2/(10*A**2*G**2*L**4*scf**2 - 120*A*E*G*Iyy*L**2*scf - 120*A*E*G*Izz*L**2*scf + 1440*E**2*Iyy*Izz))
            k += 1
            KGv[k] += r22*(-A**2*G**2*L**4*N*r32*scf**2/(10*A**2*G**2*L**4*scf**2 - 120*A*E*G*Iyy*L**2*scf - 120*A*E*G*Izz*L**2*scf + 1440*E**2*Iyy*Izz) - A**2*G**2*L**4*N*r33*scf**2/(10*A**2*G**2*L**4*scf**2 - 240*A*E*G*Izz*L**2*scf + 1440*E**2*Izz**2)) + r23*(-A**2*G**2*L**4*N*r32*scf**2/(10*A**2*G**2*L**4*scf**2 - 240*A*E*G*Iyy*L**2*scf + 1440*E**2*Iyy**2) - A**2*G**2*L**4*N*r33*scf**2/(10*A**2*G**2*L**4*scf**2 - 120*A*E*G*Iyy*L**2*scf - 120*A*E*G*Izz*L**2*scf + 1440*E**2*Iyy*Izz))
            k += 1
            KGv[k] += r32*(-A**2*G**2*L**4*N*r32*scf**2/(10*A**2*G**2*L**4*scf**2 - 120*A*E*G*Iyy*L**2*scf - 120*A*E*G*Izz*L**2*scf + 1440*E**2*Iyy*Izz) - A**2*G**2*L**4*N*r33*scf**2/(10*A**2*G**2*L**4*scf**2 - 240*A*E*G*Izz*L**2*scf + 1440*E**2*Izz**2)) + r33*(-A**2*G**2*L**4*N*r32*scf**2/(10*A**2*G**2*L**4*scf**2 - 240*A*E*G*Iyy*L**2*scf + 1440*E**2*Iyy**2) - A**2*G**2*L**4*N*r33*scf**2/(10*A**2*G**2*L**4*scf**2 - 120*A*E*G*Iyy*L**2*scf - 120*A*E*G*Izz*L**2*scf + 1440*E**2*Iyy*Izz))
            k += 1
            KGv[k] += r12*(2*L*N*r32*(A**2*G**2*L**4*scf**2 - 15*A*E*G*Iyy*L**2*scf + 90*E**2*Iyy**2)/(15*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)) + L*N*r33*(2*A**2*G**2*L**4*scf**2 - 15*A*E*G*Iyy*L**2*scf - 15*A*E*G*Izz*L**2*scf + 180*E**2*Iyy*Izz)/(15*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz))) + r13*(L*N*r32*(2*A**2*G**2*L**4*scf**2 - 15*A*E*G*Iyy*L**2*scf - 15*A*E*G*Izz*L**2*scf + 180*E**2*Iyy*Izz)/(15*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + 2*L*N*r33*(A**2*G**2*L**4*scf**2 - 15*A*E*G*Izz*L**2*scf + 90*E**2*Izz**2)/(15*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)))
            k += 1
            KGv[k] += r22*(2*L*N*r32*(A**2*G**2*L**4*scf**2 - 15*A*E*G*Iyy*L**2*scf + 90*E**2*Iyy**2)/(15*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)) + L*N*r33*(2*A**2*G**2*L**4*scf**2 - 15*A*E*G*Iyy*L**2*scf - 15*A*E*G*Izz*L**2*scf + 180*E**2*Iyy*Izz)/(15*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz))) + r23*(L*N*r32*(2*A**2*G**2*L**4*scf**2 - 15*A*E*G*Iyy*L**2*scf - 15*A*E*G*Izz*L**2*scf + 180*E**2*Iyy*Izz)/(15*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + 2*L*N*r33*(A**2*G**2*L**4*scf**2 - 15*A*E*G*Izz*L**2*scf + 90*E**2*Izz**2)/(15*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)))
            k += 1
            KGv[k] += r32*(2*L*N*r32*(A**2*G**2*L**4*scf**2 - 15*A*E*G*Iyy*L**2*scf + 90*E**2*Iyy**2)/(15*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)) + L*N*r33*(2*A**2*G**2*L**4*scf**2 - 15*A*E*G*Iyy*L**2*scf - 15*A*E*G*Izz*L**2*scf + 180*E**2*Iyy*Izz)/(15*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz))) + r33*(L*N*r32*(2*A**2*G**2*L**4*scf**2 - 15*A*E*G*Iyy*L**2*scf - 15*A*E*G*Izz*L**2*scf + 180*E**2*Iyy*Izz)/(15*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + 2*L*N*r33*(A**2*G**2*L**4*scf**2 - 15*A*E*G*Izz*L**2*scf + 90*E**2*Izz**2)/(15*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)))


    cpdef void update_M(BeamC self,
            np.ndarray[cINT, ndim=1] Mr,
            np.ndarray[cINT, ndim=1] Mc,
            np.ndarray[cDOUBLE, ndim=1] Mv,
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
        cdef cINT c1, c2, k
        cdef double intrho, intrhoy, intrhoz, intrhoy2, intrhoz2, intrhoyz
        cdef double r11, r12, r13, r21, r22, r23, r31, r32, r33
        cdef double L, A, E, G, scf, Iyy, Izz
        cdef double sina, sinb, sing

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
            scf = prop.scf
            Iyy = prop.Iyy
            Izz = prop.Izz

            #Local to global transformation
            sina = (1. - self.cosa**2)**0.5
            sinb = (1. - self.cosb**2)**0.5
            sing = (1. - self.cosg**2)**0.5
            r11 = self.cosb*self.cosg
            r12 = -self.cosa*sing + self.cosg*sina*sinb
            r13 = self.cosa*self.cosg*sinb + sina*sing
            r21 = self.cosb*sing
            r22 = self.cosa*self.cosg + sina*sinb*sing
            r23 = self.cosa*sinb*sing - self.cosg*sina
            r31 = -sinb
            r32 = self.cosb*sina
            r33 = self.cosa*self.cosb

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
                Mv[k] += r11*(A*G*L**2*intrhoy*r12*scf/(2*(A*G*L**2*scf - 12*E*Izz)) - A*G*L**2*intrhoz*r13*scf/(2*A*G*L**2*scf - 24*E*Iyy) + L*intrho*r11/3) + r12*(-6*A**2*G**2*L**3*intrhoyz*r13*scf**2/(5*A**2*G**2*L**4*scf**2 - 60*A*E*G*Iyy*L**2*scf - 60*A*E*G*Izz*L**2*scf + 720*E**2*Iyy*Izz) + A*G*L**2*intrhoy*r11*scf/(2*(A*G*L**2*scf - 12*E*Izz)) + L*r12*(13*A**2*G**2*L**4*intrho*scf**2 + 42*A**2*G**2*L**2*intrhoy2*scf**2 - 294*A*E*G*Izz*L**2*intrho*scf + 1680*E**2*Izz**2*intrho)/(35*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2))) + r13*(-6*A**2*G**2*L**3*intrhoyz*r12*scf**2/(5*A**2*G**2*L**4*scf**2 - 60*A*E*G*Iyy*L**2*scf - 60*A*E*G*Izz*L**2*scf + 720*E**2*Iyy*Izz) - A*G*L**2*intrhoz*r11*scf/(2*A*G*L**2*scf - 24*E*Iyy) + L*r13*(13*A**2*G**2*L**4*intrho*scf**2 + 42*A**2*G**2*L**2*intrhoz2*scf**2 - 294*A*E*G*Iyy*L**2*intrho*scf + 1680*E**2*Iyy**2*intrho)/(35*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)))
                k += 1
                Mv[k] += r21*(A*G*L**2*intrhoy*r12*scf/(2*(A*G*L**2*scf - 12*E*Izz)) - A*G*L**2*intrhoz*r13*scf/(2*A*G*L**2*scf - 24*E*Iyy) + L*intrho*r11/3) + r22*(-6*A**2*G**2*L**3*intrhoyz*r13*scf**2/(5*A**2*G**2*L**4*scf**2 - 60*A*E*G*Iyy*L**2*scf - 60*A*E*G*Izz*L**2*scf + 720*E**2*Iyy*Izz) + A*G*L**2*intrhoy*r11*scf/(2*(A*G*L**2*scf - 12*E*Izz)) + L*r12*(13*A**2*G**2*L**4*intrho*scf**2 + 42*A**2*G**2*L**2*intrhoy2*scf**2 - 294*A*E*G*Izz*L**2*intrho*scf + 1680*E**2*Izz**2*intrho)/(35*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2))) + r23*(-6*A**2*G**2*L**3*intrhoyz*r12*scf**2/(5*A**2*G**2*L**4*scf**2 - 60*A*E*G*Iyy*L**2*scf - 60*A*E*G*Izz*L**2*scf + 720*E**2*Iyy*Izz) - A*G*L**2*intrhoz*r11*scf/(2*A*G*L**2*scf - 24*E*Iyy) + L*r13*(13*A**2*G**2*L**4*intrho*scf**2 + 42*A**2*G**2*L**2*intrhoz2*scf**2 - 294*A*E*G*Iyy*L**2*intrho*scf + 1680*E**2*Iyy**2*intrho)/(35*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)))
                k += 1
                Mv[k] += r31*(A*G*L**2*intrhoy*r12*scf/(2*(A*G*L**2*scf - 12*E*Izz)) - A*G*L**2*intrhoz*r13*scf/(2*A*G*L**2*scf - 24*E*Iyy) + L*intrho*r11/3) + r32*(-6*A**2*G**2*L**3*intrhoyz*r13*scf**2/(5*A**2*G**2*L**4*scf**2 - 60*A*E*G*Iyy*L**2*scf - 60*A*E*G*Izz*L**2*scf + 720*E**2*Iyy*Izz) + A*G*L**2*intrhoy*r11*scf/(2*(A*G*L**2*scf - 12*E*Izz)) + L*r12*(13*A**2*G**2*L**4*intrho*scf**2 + 42*A**2*G**2*L**2*intrhoy2*scf**2 - 294*A*E*G*Izz*L**2*intrho*scf + 1680*E**2*Izz**2*intrho)/(35*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2))) + r33*(-6*A**2*G**2*L**3*intrhoyz*r12*scf**2/(5*A**2*G**2*L**4*scf**2 - 60*A*E*G*Iyy*L**2*scf - 60*A*E*G*Izz*L**2*scf + 720*E**2*Iyy*Izz) - A*G*L**2*intrhoz*r11*scf/(2*A*G*L**2*scf - 24*E*Iyy) + L*r13*(13*A**2*G**2*L**4*intrho*scf**2 + 42*A**2*G**2*L**2*intrhoz2*scf**2 - 294*A*E*G*Iyy*L**2*intrho*scf + 1680*E**2*Iyy**2*intrho)/(35*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)))
                k += 1
                Mv[k] += r11*(L*intrhoy*r13*(7*A*G*L**2*scf - 80*E*Iyy)/(20*(A*G*L**2*scf - 12*E*Iyy)) + L*intrhoz*r12*(-7*A*G*L**2*scf + 80*E*Izz)/(20*(A*G*L**2*scf - 12*E*Izz))) + r12*(-A*G*L**2*intrhoyz*r12*scf*(A*G*L**2*scf + 60*E*Iyy)/(10*A**2*G**2*L**4*scf**2 - 120*A*E*G*Iyy*L**2*scf - 120*A*E*G*Izz*L**2*scf + 1440*E**2*Iyy*Izz) + L**2*r13*(11*A**2*G**2*L**4*intrho*scf**2 + 21*A**2*G**2*L**2*intrhoz2*scf**2 - 231*A*E*G*Iyy*L**2*intrho*scf + 1260*A*E*G*Iyy*intrhoz2*scf + 1260*E**2*Iyy**2*intrho)/(210*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)) + L*intrhoz*r11*(A*G*L**2*scf - 48*E*Iyy)/(12*(A*G*L**2*scf - 12*E*Iyy))) + r13*(-A*G*L**2*intrhoyz*r13*scf*(A*G*L**2*scf + 60*E*Izz)/(10*A**2*G**2*L**4*scf**2 - 120*A*E*G*Iyy*L**2*scf - 120*A*E*G*Izz*L**2*scf + 1440*E**2*Iyy*Izz) + L**2*r12*(11*A**2*G**2*L**4*intrho*scf**2 + 21*A**2*G**2*L**2*intrhoy2*scf**2 - 231*A*E*G*Izz*L**2*intrho*scf + 1260*A*E*G*Izz*intrhoy2*scf + 1260*E**2*Izz**2*intrho)/(210*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)) + L*intrhoy*r11*(-A*G*L**2*scf + 48*E*Izz)/(12*(A*G*L**2*scf - 12*E*Izz)))
                k += 1
                Mv[k] += r21*(L*intrhoy*r13*(7*A*G*L**2*scf - 80*E*Iyy)/(20*(A*G*L**2*scf - 12*E*Iyy)) + L*intrhoz*r12*(-7*A*G*L**2*scf + 80*E*Izz)/(20*(A*G*L**2*scf - 12*E*Izz))) + r22*(-A*G*L**2*intrhoyz*r12*scf*(A*G*L**2*scf + 60*E*Iyy)/(10*A**2*G**2*L**4*scf**2 - 120*A*E*G*Iyy*L**2*scf - 120*A*E*G*Izz*L**2*scf + 1440*E**2*Iyy*Izz) + L**2*r13*(11*A**2*G**2*L**4*intrho*scf**2 + 21*A**2*G**2*L**2*intrhoz2*scf**2 - 231*A*E*G*Iyy*L**2*intrho*scf + 1260*A*E*G*Iyy*intrhoz2*scf + 1260*E**2*Iyy**2*intrho)/(210*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)) + L*intrhoz*r11*(A*G*L**2*scf - 48*E*Iyy)/(12*(A*G*L**2*scf - 12*E*Iyy))) + r23*(-A*G*L**2*intrhoyz*r13*scf*(A*G*L**2*scf + 60*E*Izz)/(10*A**2*G**2*L**4*scf**2 - 120*A*E*G*Iyy*L**2*scf - 120*A*E*G*Izz*L**2*scf + 1440*E**2*Iyy*Izz) + L**2*r12*(11*A**2*G**2*L**4*intrho*scf**2 + 21*A**2*G**2*L**2*intrhoy2*scf**2 - 231*A*E*G*Izz*L**2*intrho*scf + 1260*A*E*G*Izz*intrhoy2*scf + 1260*E**2*Izz**2*intrho)/(210*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)) + L*intrhoy*r11*(-A*G*L**2*scf + 48*E*Izz)/(12*(A*G*L**2*scf - 12*E*Izz)))
                k += 1
                Mv[k] += r31*(L*intrhoy*r13*(7*A*G*L**2*scf - 80*E*Iyy)/(20*(A*G*L**2*scf - 12*E*Iyy)) + L*intrhoz*r12*(-7*A*G*L**2*scf + 80*E*Izz)/(20*(A*G*L**2*scf - 12*E*Izz))) + r32*(-A*G*L**2*intrhoyz*r12*scf*(A*G*L**2*scf + 60*E*Iyy)/(10*A**2*G**2*L**4*scf**2 - 120*A*E*G*Iyy*L**2*scf - 120*A*E*G*Izz*L**2*scf + 1440*E**2*Iyy*Izz) + L**2*r13*(11*A**2*G**2*L**4*intrho*scf**2 + 21*A**2*G**2*L**2*intrhoz2*scf**2 - 231*A*E*G*Iyy*L**2*intrho*scf + 1260*A*E*G*Iyy*intrhoz2*scf + 1260*E**2*Iyy**2*intrho)/(210*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)) + L*intrhoz*r11*(A*G*L**2*scf - 48*E*Iyy)/(12*(A*G*L**2*scf - 12*E*Iyy))) + r33*(-A*G*L**2*intrhoyz*r13*scf*(A*G*L**2*scf + 60*E*Izz)/(10*A**2*G**2*L**4*scf**2 - 120*A*E*G*Iyy*L**2*scf - 120*A*E*G*Izz*L**2*scf + 1440*E**2*Iyy*Izz) + L**2*r12*(11*A**2*G**2*L**4*intrho*scf**2 + 21*A**2*G**2*L**2*intrhoy2*scf**2 - 231*A*E*G*Izz*L**2*intrho*scf + 1260*A*E*G*Izz*intrhoy2*scf + 1260*E**2*Izz**2*intrho)/(210*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)) + L*intrhoy*r11*(-A*G*L**2*scf + 48*E*Izz)/(12*(A*G*L**2*scf - 12*E*Izz)))
                k += 1
                Mv[k] += r11*(A*G*L**2*intrhoy*r12*scf/(2*(A*G*L**2*scf - 12*E*Izz)) - A*G*L**2*intrhoz*r13*scf/(2*A*G*L**2*scf - 24*E*Iyy) + L*intrho*r11/6) + r12*(6*A**2*G**2*L**3*intrhoyz*r13*scf**2/(5*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) - A*G*L**2*intrhoy*r11*scf/(2*A*G*L**2*scf - 24*E*Izz) + r12*(9*A**2*G**2*L**5*intrho*scf**2 - 84*A**2*G**2*L**3*intrhoy2*scf**2 - 252*A*E*G*Izz*L**3*intrho*scf + 1680*E**2*Izz**2*L*intrho)/(70*A**2*G**2*L**4*scf**2 - 1680*A*E*G*Izz*L**2*scf + 10080*E**2*Izz**2)) + r13*(6*A**2*G**2*L**3*intrhoyz*r12*scf**2/(5*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + A*G*L**2*intrhoz*r11*scf/(2*(A*G*L**2*scf - 12*E*Iyy)) + r13*(9*A**2*G**2*L**5*intrho*scf**2 - 84*A**2*G**2*L**3*intrhoz2*scf**2 - 252*A*E*G*Iyy*L**3*intrho*scf + 1680*E**2*Iyy**2*L*intrho)/(70*A**2*G**2*L**4*scf**2 - 1680*A*E*G*Iyy*L**2*scf + 10080*E**2*Iyy**2))
                k += 1
                Mv[k] += r21*(A*G*L**2*intrhoy*r12*scf/(2*(A*G*L**2*scf - 12*E*Izz)) - A*G*L**2*intrhoz*r13*scf/(2*A*G*L**2*scf - 24*E*Iyy) + L*intrho*r11/6) + r22*(6*A**2*G**2*L**3*intrhoyz*r13*scf**2/(5*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) - A*G*L**2*intrhoy*r11*scf/(2*A*G*L**2*scf - 24*E*Izz) + r12*(9*A**2*G**2*L**5*intrho*scf**2 - 84*A**2*G**2*L**3*intrhoy2*scf**2 - 252*A*E*G*Izz*L**3*intrho*scf + 1680*E**2*Izz**2*L*intrho)/(70*A**2*G**2*L**4*scf**2 - 1680*A*E*G*Izz*L**2*scf + 10080*E**2*Izz**2)) + r23*(6*A**2*G**2*L**3*intrhoyz*r12*scf**2/(5*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + A*G*L**2*intrhoz*r11*scf/(2*(A*G*L**2*scf - 12*E*Iyy)) + r13*(9*A**2*G**2*L**5*intrho*scf**2 - 84*A**2*G**2*L**3*intrhoz2*scf**2 - 252*A*E*G*Iyy*L**3*intrho*scf + 1680*E**2*Iyy**2*L*intrho)/(70*A**2*G**2*L**4*scf**2 - 1680*A*E*G*Iyy*L**2*scf + 10080*E**2*Iyy**2))
                k += 1
                Mv[k] += r31*(A*G*L**2*intrhoy*r12*scf/(2*(A*G*L**2*scf - 12*E*Izz)) - A*G*L**2*intrhoz*r13*scf/(2*A*G*L**2*scf - 24*E*Iyy) + L*intrho*r11/6) + r32*(6*A**2*G**2*L**3*intrhoyz*r13*scf**2/(5*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) - A*G*L**2*intrhoy*r11*scf/(2*A*G*L**2*scf - 24*E*Izz) + r12*(9*A**2*G**2*L**5*intrho*scf**2 - 84*A**2*G**2*L**3*intrhoy2*scf**2 - 252*A*E*G*Izz*L**3*intrho*scf + 1680*E**2*Izz**2*L*intrho)/(70*A**2*G**2*L**4*scf**2 - 1680*A*E*G*Izz*L**2*scf + 10080*E**2*Izz**2)) + r33*(6*A**2*G**2*L**3*intrhoyz*r12*scf**2/(5*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + A*G*L**2*intrhoz*r11*scf/(2*(A*G*L**2*scf - 12*E*Iyy)) + r13*(9*A**2*G**2*L**5*intrho*scf**2 - 84*A**2*G**2*L**3*intrhoz2*scf**2 - 252*A*E*G*Iyy*L**3*intrho*scf + 1680*E**2*Iyy**2*L*intrho)/(70*A**2*G**2*L**4*scf**2 - 1680*A*E*G*Iyy*L**2*scf + 10080*E**2*Iyy**2))
                k += 1
                Mv[k] += r11*(L*intrhoy*r13*(3*A*G*L**2*scf - 40*E*Iyy)/(20*(A*G*L**2*scf - 12*E*Iyy)) + L*intrhoz*r12*(-3*A*G*L**2*scf + 40*E*Izz)/(20*(A*G*L**2*scf - 12*E*Izz))) + r12*(-A*G*L**2*intrhoyz*r12*scf*(A*G*L**2*scf + 60*E*Iyy)/(10*A**2*G**2*L**4*scf**2 - 120*A*E*G*Iyy*L**2*scf - 120*A*E*G*Izz*L**2*scf + 1440*E**2*Iyy*Izz) + L**2*r13*(-13*A**2*G**2*L**4*intrho*scf**2 + 42*A**2*G**2*L**2*intrhoz2*scf**2 + 378*A*E*G*Iyy*L**2*intrho*scf + 2520*A*E*G*Iyy*intrhoz2*scf - 2520*E**2*Iyy**2*intrho)/(420*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)) - L*intrhoz*r11*(A*G*L**2*scf + 24*E*Iyy)/(12*A*G*L**2*scf - 144*E*Iyy)) + r13*(-A*G*L**2*intrhoyz*r13*scf*(A*G*L**2*scf + 60*E*Izz)/(10*A**2*G**2*L**4*scf**2 - 120*A*E*G*Iyy*L**2*scf - 120*A*E*G*Izz*L**2*scf + 1440*E**2*Iyy*Izz) + L**2*r12*(-13*A**2*G**2*L**4*intrho*scf**2 + 42*A**2*G**2*L**2*intrhoy2*scf**2 + 378*A*E*G*Izz*L**2*intrho*scf + 2520*A*E*G*Izz*intrhoy2*scf - 2520*E**2*Izz**2*intrho)/(420*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)) + L*intrhoy*r11*(A*G*L**2*scf + 24*E*Izz)/(12*(A*G*L**2*scf - 12*E*Izz)))
                k += 1
                Mv[k] += r21*(L*intrhoy*r13*(3*A*G*L**2*scf - 40*E*Iyy)/(20*(A*G*L**2*scf - 12*E*Iyy)) + L*intrhoz*r12*(-3*A*G*L**2*scf + 40*E*Izz)/(20*(A*G*L**2*scf - 12*E*Izz))) + r22*(-A*G*L**2*intrhoyz*r12*scf*(A*G*L**2*scf + 60*E*Iyy)/(10*A**2*G**2*L**4*scf**2 - 120*A*E*G*Iyy*L**2*scf - 120*A*E*G*Izz*L**2*scf + 1440*E**2*Iyy*Izz) + L**2*r13*(-13*A**2*G**2*L**4*intrho*scf**2 + 42*A**2*G**2*L**2*intrhoz2*scf**2 + 378*A*E*G*Iyy*L**2*intrho*scf + 2520*A*E*G*Iyy*intrhoz2*scf - 2520*E**2*Iyy**2*intrho)/(420*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)) - L*intrhoz*r11*(A*G*L**2*scf + 24*E*Iyy)/(12*A*G*L**2*scf - 144*E*Iyy)) + r23*(-A*G*L**2*intrhoyz*r13*scf*(A*G*L**2*scf + 60*E*Izz)/(10*A**2*G**2*L**4*scf**2 - 120*A*E*G*Iyy*L**2*scf - 120*A*E*G*Izz*L**2*scf + 1440*E**2*Iyy*Izz) + L**2*r12*(-13*A**2*G**2*L**4*intrho*scf**2 + 42*A**2*G**2*L**2*intrhoy2*scf**2 + 378*A*E*G*Izz*L**2*intrho*scf + 2520*A*E*G*Izz*intrhoy2*scf - 2520*E**2*Izz**2*intrho)/(420*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)) + L*intrhoy*r11*(A*G*L**2*scf + 24*E*Izz)/(12*(A*G*L**2*scf - 12*E*Izz)))
                k += 1
                Mv[k] += r31*(L*intrhoy*r13*(3*A*G*L**2*scf - 40*E*Iyy)/(20*(A*G*L**2*scf - 12*E*Iyy)) + L*intrhoz*r12*(-3*A*G*L**2*scf + 40*E*Izz)/(20*(A*G*L**2*scf - 12*E*Izz))) + r32*(-A*G*L**2*intrhoyz*r12*scf*(A*G*L**2*scf + 60*E*Iyy)/(10*A**2*G**2*L**4*scf**2 - 120*A*E*G*Iyy*L**2*scf - 120*A*E*G*Izz*L**2*scf + 1440*E**2*Iyy*Izz) + L**2*r13*(-13*A**2*G**2*L**4*intrho*scf**2 + 42*A**2*G**2*L**2*intrhoz2*scf**2 + 378*A*E*G*Iyy*L**2*intrho*scf + 2520*A*E*G*Iyy*intrhoz2*scf - 2520*E**2*Iyy**2*intrho)/(420*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)) - L*intrhoz*r11*(A*G*L**2*scf + 24*E*Iyy)/(12*A*G*L**2*scf - 144*E*Iyy)) + r33*(-A*G*L**2*intrhoyz*r13*scf*(A*G*L**2*scf + 60*E*Izz)/(10*A**2*G**2*L**4*scf**2 - 120*A*E*G*Iyy*L**2*scf - 120*A*E*G*Izz*L**2*scf + 1440*E**2*Iyy*Izz) + L**2*r12*(-13*A**2*G**2*L**4*intrho*scf**2 + 42*A**2*G**2*L**2*intrhoy2*scf**2 + 378*A*E*G*Izz*L**2*intrho*scf + 2520*A*E*G*Izz*intrhoy2*scf - 2520*E**2*Izz**2*intrho)/(420*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)) + L*intrhoy*r11*(A*G*L**2*scf + 24*E*Izz)/(12*(A*G*L**2*scf - 12*E*Izz)))
                k += 1
                Mv[k] += r11*(A*G*L**2*intrhoy*r22*scf/(2*(A*G*L**2*scf - 12*E*Izz)) - A*G*L**2*intrhoz*r23*scf/(2*A*G*L**2*scf - 24*E*Iyy) + L*intrho*r21/3) + r12*(-6*A**2*G**2*L**3*intrhoyz*r23*scf**2/(5*A**2*G**2*L**4*scf**2 - 60*A*E*G*Iyy*L**2*scf - 60*A*E*G*Izz*L**2*scf + 720*E**2*Iyy*Izz) + A*G*L**2*intrhoy*r21*scf/(2*(A*G*L**2*scf - 12*E*Izz)) + L*r22*(13*A**2*G**2*L**4*intrho*scf**2 + 42*A**2*G**2*L**2*intrhoy2*scf**2 - 294*A*E*G*Izz*L**2*intrho*scf + 1680*E**2*Izz**2*intrho)/(35*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2))) + r13*(-6*A**2*G**2*L**3*intrhoyz*r22*scf**2/(5*A**2*G**2*L**4*scf**2 - 60*A*E*G*Iyy*L**2*scf - 60*A*E*G*Izz*L**2*scf + 720*E**2*Iyy*Izz) - A*G*L**2*intrhoz*r21*scf/(2*A*G*L**2*scf - 24*E*Iyy) + L*r23*(13*A**2*G**2*L**4*intrho*scf**2 + 42*A**2*G**2*L**2*intrhoz2*scf**2 - 294*A*E*G*Iyy*L**2*intrho*scf + 1680*E**2*Iyy**2*intrho)/(35*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)))
                k += 1
                Mv[k] += r21*(A*G*L**2*intrhoy*r22*scf/(2*(A*G*L**2*scf - 12*E*Izz)) - A*G*L**2*intrhoz*r23*scf/(2*A*G*L**2*scf - 24*E*Iyy) + L*intrho*r21/3) + r22*(-6*A**2*G**2*L**3*intrhoyz*r23*scf**2/(5*A**2*G**2*L**4*scf**2 - 60*A*E*G*Iyy*L**2*scf - 60*A*E*G*Izz*L**2*scf + 720*E**2*Iyy*Izz) + A*G*L**2*intrhoy*r21*scf/(2*(A*G*L**2*scf - 12*E*Izz)) + L*r22*(13*A**2*G**2*L**4*intrho*scf**2 + 42*A**2*G**2*L**2*intrhoy2*scf**2 - 294*A*E*G*Izz*L**2*intrho*scf + 1680*E**2*Izz**2*intrho)/(35*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2))) + r23*(-6*A**2*G**2*L**3*intrhoyz*r22*scf**2/(5*A**2*G**2*L**4*scf**2 - 60*A*E*G*Iyy*L**2*scf - 60*A*E*G*Izz*L**2*scf + 720*E**2*Iyy*Izz) - A*G*L**2*intrhoz*r21*scf/(2*A*G*L**2*scf - 24*E*Iyy) + L*r23*(13*A**2*G**2*L**4*intrho*scf**2 + 42*A**2*G**2*L**2*intrhoz2*scf**2 - 294*A*E*G*Iyy*L**2*intrho*scf + 1680*E**2*Iyy**2*intrho)/(35*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)))
                k += 1
                Mv[k] += r31*(A*G*L**2*intrhoy*r22*scf/(2*(A*G*L**2*scf - 12*E*Izz)) - A*G*L**2*intrhoz*r23*scf/(2*A*G*L**2*scf - 24*E*Iyy) + L*intrho*r21/3) + r32*(-6*A**2*G**2*L**3*intrhoyz*r23*scf**2/(5*A**2*G**2*L**4*scf**2 - 60*A*E*G*Iyy*L**2*scf - 60*A*E*G*Izz*L**2*scf + 720*E**2*Iyy*Izz) + A*G*L**2*intrhoy*r21*scf/(2*(A*G*L**2*scf - 12*E*Izz)) + L*r22*(13*A**2*G**2*L**4*intrho*scf**2 + 42*A**2*G**2*L**2*intrhoy2*scf**2 - 294*A*E*G*Izz*L**2*intrho*scf + 1680*E**2*Izz**2*intrho)/(35*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2))) + r33*(-6*A**2*G**2*L**3*intrhoyz*r22*scf**2/(5*A**2*G**2*L**4*scf**2 - 60*A*E*G*Iyy*L**2*scf - 60*A*E*G*Izz*L**2*scf + 720*E**2*Iyy*Izz) - A*G*L**2*intrhoz*r21*scf/(2*A*G*L**2*scf - 24*E*Iyy) + L*r23*(13*A**2*G**2*L**4*intrho*scf**2 + 42*A**2*G**2*L**2*intrhoz2*scf**2 - 294*A*E*G*Iyy*L**2*intrho*scf + 1680*E**2*Iyy**2*intrho)/(35*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)))
                k += 1
                Mv[k] += r11*(L*intrhoy*r23*(7*A*G*L**2*scf - 80*E*Iyy)/(20*(A*G*L**2*scf - 12*E*Iyy)) + L*intrhoz*r22*(-7*A*G*L**2*scf + 80*E*Izz)/(20*(A*G*L**2*scf - 12*E*Izz))) + r12*(-A*G*L**2*intrhoyz*r22*scf*(A*G*L**2*scf + 60*E*Iyy)/(10*A**2*G**2*L**4*scf**2 - 120*A*E*G*Iyy*L**2*scf - 120*A*E*G*Izz*L**2*scf + 1440*E**2*Iyy*Izz) + L**2*r23*(11*A**2*G**2*L**4*intrho*scf**2 + 21*A**2*G**2*L**2*intrhoz2*scf**2 - 231*A*E*G*Iyy*L**2*intrho*scf + 1260*A*E*G*Iyy*intrhoz2*scf + 1260*E**2*Iyy**2*intrho)/(210*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)) + L*intrhoz*r21*(A*G*L**2*scf - 48*E*Iyy)/(12*(A*G*L**2*scf - 12*E*Iyy))) + r13*(-A*G*L**2*intrhoyz*r23*scf*(A*G*L**2*scf + 60*E*Izz)/(10*A**2*G**2*L**4*scf**2 - 120*A*E*G*Iyy*L**2*scf - 120*A*E*G*Izz*L**2*scf + 1440*E**2*Iyy*Izz) + L**2*r22*(11*A**2*G**2*L**4*intrho*scf**2 + 21*A**2*G**2*L**2*intrhoy2*scf**2 - 231*A*E*G*Izz*L**2*intrho*scf + 1260*A*E*G*Izz*intrhoy2*scf + 1260*E**2*Izz**2*intrho)/(210*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)) + L*intrhoy*r21*(-A*G*L**2*scf + 48*E*Izz)/(12*(A*G*L**2*scf - 12*E*Izz)))
                k += 1
                Mv[k] += r21*(L*intrhoy*r23*(7*A*G*L**2*scf - 80*E*Iyy)/(20*(A*G*L**2*scf - 12*E*Iyy)) + L*intrhoz*r22*(-7*A*G*L**2*scf + 80*E*Izz)/(20*(A*G*L**2*scf - 12*E*Izz))) + r22*(-A*G*L**2*intrhoyz*r22*scf*(A*G*L**2*scf + 60*E*Iyy)/(10*A**2*G**2*L**4*scf**2 - 120*A*E*G*Iyy*L**2*scf - 120*A*E*G*Izz*L**2*scf + 1440*E**2*Iyy*Izz) + L**2*r23*(11*A**2*G**2*L**4*intrho*scf**2 + 21*A**2*G**2*L**2*intrhoz2*scf**2 - 231*A*E*G*Iyy*L**2*intrho*scf + 1260*A*E*G*Iyy*intrhoz2*scf + 1260*E**2*Iyy**2*intrho)/(210*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)) + L*intrhoz*r21*(A*G*L**2*scf - 48*E*Iyy)/(12*(A*G*L**2*scf - 12*E*Iyy))) + r23*(-A*G*L**2*intrhoyz*r23*scf*(A*G*L**2*scf + 60*E*Izz)/(10*A**2*G**2*L**4*scf**2 - 120*A*E*G*Iyy*L**2*scf - 120*A*E*G*Izz*L**2*scf + 1440*E**2*Iyy*Izz) + L**2*r22*(11*A**2*G**2*L**4*intrho*scf**2 + 21*A**2*G**2*L**2*intrhoy2*scf**2 - 231*A*E*G*Izz*L**2*intrho*scf + 1260*A*E*G*Izz*intrhoy2*scf + 1260*E**2*Izz**2*intrho)/(210*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)) + L*intrhoy*r21*(-A*G*L**2*scf + 48*E*Izz)/(12*(A*G*L**2*scf - 12*E*Izz)))
                k += 1
                Mv[k] += r31*(L*intrhoy*r23*(7*A*G*L**2*scf - 80*E*Iyy)/(20*(A*G*L**2*scf - 12*E*Iyy)) + L*intrhoz*r22*(-7*A*G*L**2*scf + 80*E*Izz)/(20*(A*G*L**2*scf - 12*E*Izz))) + r32*(-A*G*L**2*intrhoyz*r22*scf*(A*G*L**2*scf + 60*E*Iyy)/(10*A**2*G**2*L**4*scf**2 - 120*A*E*G*Iyy*L**2*scf - 120*A*E*G*Izz*L**2*scf + 1440*E**2*Iyy*Izz) + L**2*r23*(11*A**2*G**2*L**4*intrho*scf**2 + 21*A**2*G**2*L**2*intrhoz2*scf**2 - 231*A*E*G*Iyy*L**2*intrho*scf + 1260*A*E*G*Iyy*intrhoz2*scf + 1260*E**2*Iyy**2*intrho)/(210*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)) + L*intrhoz*r21*(A*G*L**2*scf - 48*E*Iyy)/(12*(A*G*L**2*scf - 12*E*Iyy))) + r33*(-A*G*L**2*intrhoyz*r23*scf*(A*G*L**2*scf + 60*E*Izz)/(10*A**2*G**2*L**4*scf**2 - 120*A*E*G*Iyy*L**2*scf - 120*A*E*G*Izz*L**2*scf + 1440*E**2*Iyy*Izz) + L**2*r22*(11*A**2*G**2*L**4*intrho*scf**2 + 21*A**2*G**2*L**2*intrhoy2*scf**2 - 231*A*E*G*Izz*L**2*intrho*scf + 1260*A*E*G*Izz*intrhoy2*scf + 1260*E**2*Izz**2*intrho)/(210*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)) + L*intrhoy*r21*(-A*G*L**2*scf + 48*E*Izz)/(12*(A*G*L**2*scf - 12*E*Izz)))
                k += 1
                Mv[k] += r11*(A*G*L**2*intrhoy*r22*scf/(2*(A*G*L**2*scf - 12*E*Izz)) - A*G*L**2*intrhoz*r23*scf/(2*A*G*L**2*scf - 24*E*Iyy) + L*intrho*r21/6) + r12*(6*A**2*G**2*L**3*intrhoyz*r23*scf**2/(5*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) - A*G*L**2*intrhoy*r21*scf/(2*A*G*L**2*scf - 24*E*Izz) + r22*(9*A**2*G**2*L**5*intrho*scf**2 - 84*A**2*G**2*L**3*intrhoy2*scf**2 - 252*A*E*G*Izz*L**3*intrho*scf + 1680*E**2*Izz**2*L*intrho)/(70*A**2*G**2*L**4*scf**2 - 1680*A*E*G*Izz*L**2*scf + 10080*E**2*Izz**2)) + r13*(6*A**2*G**2*L**3*intrhoyz*r22*scf**2/(5*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + A*G*L**2*intrhoz*r21*scf/(2*(A*G*L**2*scf - 12*E*Iyy)) + r23*(9*A**2*G**2*L**5*intrho*scf**2 - 84*A**2*G**2*L**3*intrhoz2*scf**2 - 252*A*E*G*Iyy*L**3*intrho*scf + 1680*E**2*Iyy**2*L*intrho)/(70*A**2*G**2*L**4*scf**2 - 1680*A*E*G*Iyy*L**2*scf + 10080*E**2*Iyy**2))
                k += 1
                Mv[k] += r21*(A*G*L**2*intrhoy*r22*scf/(2*(A*G*L**2*scf - 12*E*Izz)) - A*G*L**2*intrhoz*r23*scf/(2*A*G*L**2*scf - 24*E*Iyy) + L*intrho*r21/6) + r22*(6*A**2*G**2*L**3*intrhoyz*r23*scf**2/(5*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) - A*G*L**2*intrhoy*r21*scf/(2*A*G*L**2*scf - 24*E*Izz) + r22*(9*A**2*G**2*L**5*intrho*scf**2 - 84*A**2*G**2*L**3*intrhoy2*scf**2 - 252*A*E*G*Izz*L**3*intrho*scf + 1680*E**2*Izz**2*L*intrho)/(70*A**2*G**2*L**4*scf**2 - 1680*A*E*G*Izz*L**2*scf + 10080*E**2*Izz**2)) + r23*(6*A**2*G**2*L**3*intrhoyz*r22*scf**2/(5*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + A*G*L**2*intrhoz*r21*scf/(2*(A*G*L**2*scf - 12*E*Iyy)) + r23*(9*A**2*G**2*L**5*intrho*scf**2 - 84*A**2*G**2*L**3*intrhoz2*scf**2 - 252*A*E*G*Iyy*L**3*intrho*scf + 1680*E**2*Iyy**2*L*intrho)/(70*A**2*G**2*L**4*scf**2 - 1680*A*E*G*Iyy*L**2*scf + 10080*E**2*Iyy**2))
                k += 1
                Mv[k] += r31*(A*G*L**2*intrhoy*r22*scf/(2*(A*G*L**2*scf - 12*E*Izz)) - A*G*L**2*intrhoz*r23*scf/(2*A*G*L**2*scf - 24*E*Iyy) + L*intrho*r21/6) + r32*(6*A**2*G**2*L**3*intrhoyz*r23*scf**2/(5*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) - A*G*L**2*intrhoy*r21*scf/(2*A*G*L**2*scf - 24*E*Izz) + r22*(9*A**2*G**2*L**5*intrho*scf**2 - 84*A**2*G**2*L**3*intrhoy2*scf**2 - 252*A*E*G*Izz*L**3*intrho*scf + 1680*E**2*Izz**2*L*intrho)/(70*A**2*G**2*L**4*scf**2 - 1680*A*E*G*Izz*L**2*scf + 10080*E**2*Izz**2)) + r33*(6*A**2*G**2*L**3*intrhoyz*r22*scf**2/(5*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + A*G*L**2*intrhoz*r21*scf/(2*(A*G*L**2*scf - 12*E*Iyy)) + r23*(9*A**2*G**2*L**5*intrho*scf**2 - 84*A**2*G**2*L**3*intrhoz2*scf**2 - 252*A*E*G*Iyy*L**3*intrho*scf + 1680*E**2*Iyy**2*L*intrho)/(70*A**2*G**2*L**4*scf**2 - 1680*A*E*G*Iyy*L**2*scf + 10080*E**2*Iyy**2))
                k += 1
                Mv[k] += r11*(L*intrhoy*r23*(3*A*G*L**2*scf - 40*E*Iyy)/(20*(A*G*L**2*scf - 12*E*Iyy)) + L*intrhoz*r22*(-3*A*G*L**2*scf + 40*E*Izz)/(20*(A*G*L**2*scf - 12*E*Izz))) + r12*(-A*G*L**2*intrhoyz*r22*scf*(A*G*L**2*scf + 60*E*Iyy)/(10*A**2*G**2*L**4*scf**2 - 120*A*E*G*Iyy*L**2*scf - 120*A*E*G*Izz*L**2*scf + 1440*E**2*Iyy*Izz) + L**2*r23*(-13*A**2*G**2*L**4*intrho*scf**2 + 42*A**2*G**2*L**2*intrhoz2*scf**2 + 378*A*E*G*Iyy*L**2*intrho*scf + 2520*A*E*G*Iyy*intrhoz2*scf - 2520*E**2*Iyy**2*intrho)/(420*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)) - L*intrhoz*r21*(A*G*L**2*scf + 24*E*Iyy)/(12*A*G*L**2*scf - 144*E*Iyy)) + r13*(-A*G*L**2*intrhoyz*r23*scf*(A*G*L**2*scf + 60*E*Izz)/(10*A**2*G**2*L**4*scf**2 - 120*A*E*G*Iyy*L**2*scf - 120*A*E*G*Izz*L**2*scf + 1440*E**2*Iyy*Izz) + L**2*r22*(-13*A**2*G**2*L**4*intrho*scf**2 + 42*A**2*G**2*L**2*intrhoy2*scf**2 + 378*A*E*G*Izz*L**2*intrho*scf + 2520*A*E*G*Izz*intrhoy2*scf - 2520*E**2*Izz**2*intrho)/(420*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)) + L*intrhoy*r21*(A*G*L**2*scf + 24*E*Izz)/(12*(A*G*L**2*scf - 12*E*Izz)))
                k += 1
                Mv[k] += r21*(L*intrhoy*r23*(3*A*G*L**2*scf - 40*E*Iyy)/(20*(A*G*L**2*scf - 12*E*Iyy)) + L*intrhoz*r22*(-3*A*G*L**2*scf + 40*E*Izz)/(20*(A*G*L**2*scf - 12*E*Izz))) + r22*(-A*G*L**2*intrhoyz*r22*scf*(A*G*L**2*scf + 60*E*Iyy)/(10*A**2*G**2*L**4*scf**2 - 120*A*E*G*Iyy*L**2*scf - 120*A*E*G*Izz*L**2*scf + 1440*E**2*Iyy*Izz) + L**2*r23*(-13*A**2*G**2*L**4*intrho*scf**2 + 42*A**2*G**2*L**2*intrhoz2*scf**2 + 378*A*E*G*Iyy*L**2*intrho*scf + 2520*A*E*G*Iyy*intrhoz2*scf - 2520*E**2*Iyy**2*intrho)/(420*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)) - L*intrhoz*r21*(A*G*L**2*scf + 24*E*Iyy)/(12*A*G*L**2*scf - 144*E*Iyy)) + r23*(-A*G*L**2*intrhoyz*r23*scf*(A*G*L**2*scf + 60*E*Izz)/(10*A**2*G**2*L**4*scf**2 - 120*A*E*G*Iyy*L**2*scf - 120*A*E*G*Izz*L**2*scf + 1440*E**2*Iyy*Izz) + L**2*r22*(-13*A**2*G**2*L**4*intrho*scf**2 + 42*A**2*G**2*L**2*intrhoy2*scf**2 + 378*A*E*G*Izz*L**2*intrho*scf + 2520*A*E*G*Izz*intrhoy2*scf - 2520*E**2*Izz**2*intrho)/(420*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)) + L*intrhoy*r21*(A*G*L**2*scf + 24*E*Izz)/(12*(A*G*L**2*scf - 12*E*Izz)))
                k += 1
                Mv[k] += r31*(L*intrhoy*r23*(3*A*G*L**2*scf - 40*E*Iyy)/(20*(A*G*L**2*scf - 12*E*Iyy)) + L*intrhoz*r22*(-3*A*G*L**2*scf + 40*E*Izz)/(20*(A*G*L**2*scf - 12*E*Izz))) + r32*(-A*G*L**2*intrhoyz*r22*scf*(A*G*L**2*scf + 60*E*Iyy)/(10*A**2*G**2*L**4*scf**2 - 120*A*E*G*Iyy*L**2*scf - 120*A*E*G*Izz*L**2*scf + 1440*E**2*Iyy*Izz) + L**2*r23*(-13*A**2*G**2*L**4*intrho*scf**2 + 42*A**2*G**2*L**2*intrhoz2*scf**2 + 378*A*E*G*Iyy*L**2*intrho*scf + 2520*A*E*G*Iyy*intrhoz2*scf - 2520*E**2*Iyy**2*intrho)/(420*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)) - L*intrhoz*r21*(A*G*L**2*scf + 24*E*Iyy)/(12*A*G*L**2*scf - 144*E*Iyy)) + r33*(-A*G*L**2*intrhoyz*r23*scf*(A*G*L**2*scf + 60*E*Izz)/(10*A**2*G**2*L**4*scf**2 - 120*A*E*G*Iyy*L**2*scf - 120*A*E*G*Izz*L**2*scf + 1440*E**2*Iyy*Izz) + L**2*r22*(-13*A**2*G**2*L**4*intrho*scf**2 + 42*A**2*G**2*L**2*intrhoy2*scf**2 + 378*A*E*G*Izz*L**2*intrho*scf + 2520*A*E*G*Izz*intrhoy2*scf - 2520*E**2*Izz**2*intrho)/(420*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)) + L*intrhoy*r21*(A*G*L**2*scf + 24*E*Izz)/(12*(A*G*L**2*scf - 12*E*Izz)))
                k += 1
                Mv[k] += r11*(A*G*L**2*intrhoy*r32*scf/(2*(A*G*L**2*scf - 12*E*Izz)) - A*G*L**2*intrhoz*r33*scf/(2*A*G*L**2*scf - 24*E*Iyy) + L*intrho*r31/3) + r12*(-6*A**2*G**2*L**3*intrhoyz*r33*scf**2/(5*A**2*G**2*L**4*scf**2 - 60*A*E*G*Iyy*L**2*scf - 60*A*E*G*Izz*L**2*scf + 720*E**2*Iyy*Izz) + A*G*L**2*intrhoy*r31*scf/(2*(A*G*L**2*scf - 12*E*Izz)) + L*r32*(13*A**2*G**2*L**4*intrho*scf**2 + 42*A**2*G**2*L**2*intrhoy2*scf**2 - 294*A*E*G*Izz*L**2*intrho*scf + 1680*E**2*Izz**2*intrho)/(35*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2))) + r13*(-6*A**2*G**2*L**3*intrhoyz*r32*scf**2/(5*A**2*G**2*L**4*scf**2 - 60*A*E*G*Iyy*L**2*scf - 60*A*E*G*Izz*L**2*scf + 720*E**2*Iyy*Izz) - A*G*L**2*intrhoz*r31*scf/(2*A*G*L**2*scf - 24*E*Iyy) + L*r33*(13*A**2*G**2*L**4*intrho*scf**2 + 42*A**2*G**2*L**2*intrhoz2*scf**2 - 294*A*E*G*Iyy*L**2*intrho*scf + 1680*E**2*Iyy**2*intrho)/(35*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)))
                k += 1
                Mv[k] += r21*(A*G*L**2*intrhoy*r32*scf/(2*(A*G*L**2*scf - 12*E*Izz)) - A*G*L**2*intrhoz*r33*scf/(2*A*G*L**2*scf - 24*E*Iyy) + L*intrho*r31/3) + r22*(-6*A**2*G**2*L**3*intrhoyz*r33*scf**2/(5*A**2*G**2*L**4*scf**2 - 60*A*E*G*Iyy*L**2*scf - 60*A*E*G*Izz*L**2*scf + 720*E**2*Iyy*Izz) + A*G*L**2*intrhoy*r31*scf/(2*(A*G*L**2*scf - 12*E*Izz)) + L*r32*(13*A**2*G**2*L**4*intrho*scf**2 + 42*A**2*G**2*L**2*intrhoy2*scf**2 - 294*A*E*G*Izz*L**2*intrho*scf + 1680*E**2*Izz**2*intrho)/(35*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2))) + r23*(-6*A**2*G**2*L**3*intrhoyz*r32*scf**2/(5*A**2*G**2*L**4*scf**2 - 60*A*E*G*Iyy*L**2*scf - 60*A*E*G*Izz*L**2*scf + 720*E**2*Iyy*Izz) - A*G*L**2*intrhoz*r31*scf/(2*A*G*L**2*scf - 24*E*Iyy) + L*r33*(13*A**2*G**2*L**4*intrho*scf**2 + 42*A**2*G**2*L**2*intrhoz2*scf**2 - 294*A*E*G*Iyy*L**2*intrho*scf + 1680*E**2*Iyy**2*intrho)/(35*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)))
                k += 1
                Mv[k] += r31*(A*G*L**2*intrhoy*r32*scf/(2*(A*G*L**2*scf - 12*E*Izz)) - A*G*L**2*intrhoz*r33*scf/(2*A*G*L**2*scf - 24*E*Iyy) + L*intrho*r31/3) + r32*(-6*A**2*G**2*L**3*intrhoyz*r33*scf**2/(5*A**2*G**2*L**4*scf**2 - 60*A*E*G*Iyy*L**2*scf - 60*A*E*G*Izz*L**2*scf + 720*E**2*Iyy*Izz) + A*G*L**2*intrhoy*r31*scf/(2*(A*G*L**2*scf - 12*E*Izz)) + L*r32*(13*A**2*G**2*L**4*intrho*scf**2 + 42*A**2*G**2*L**2*intrhoy2*scf**2 - 294*A*E*G*Izz*L**2*intrho*scf + 1680*E**2*Izz**2*intrho)/(35*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2))) + r33*(-6*A**2*G**2*L**3*intrhoyz*r32*scf**2/(5*A**2*G**2*L**4*scf**2 - 60*A*E*G*Iyy*L**2*scf - 60*A*E*G*Izz*L**2*scf + 720*E**2*Iyy*Izz) - A*G*L**2*intrhoz*r31*scf/(2*A*G*L**2*scf - 24*E*Iyy) + L*r33*(13*A**2*G**2*L**4*intrho*scf**2 + 42*A**2*G**2*L**2*intrhoz2*scf**2 - 294*A*E*G*Iyy*L**2*intrho*scf + 1680*E**2*Iyy**2*intrho)/(35*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)))
                k += 1
                Mv[k] += r11*(L*intrhoy*r33*(7*A*G*L**2*scf - 80*E*Iyy)/(20*(A*G*L**2*scf - 12*E*Iyy)) + L*intrhoz*r32*(-7*A*G*L**2*scf + 80*E*Izz)/(20*(A*G*L**2*scf - 12*E*Izz))) + r12*(-A*G*L**2*intrhoyz*r32*scf*(A*G*L**2*scf + 60*E*Iyy)/(10*A**2*G**2*L**4*scf**2 - 120*A*E*G*Iyy*L**2*scf - 120*A*E*G*Izz*L**2*scf + 1440*E**2*Iyy*Izz) + L**2*r33*(11*A**2*G**2*L**4*intrho*scf**2 + 21*A**2*G**2*L**2*intrhoz2*scf**2 - 231*A*E*G*Iyy*L**2*intrho*scf + 1260*A*E*G*Iyy*intrhoz2*scf + 1260*E**2*Iyy**2*intrho)/(210*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)) + L*intrhoz*r31*(A*G*L**2*scf - 48*E*Iyy)/(12*(A*G*L**2*scf - 12*E*Iyy))) + r13*(-A*G*L**2*intrhoyz*r33*scf*(A*G*L**2*scf + 60*E*Izz)/(10*A**2*G**2*L**4*scf**2 - 120*A*E*G*Iyy*L**2*scf - 120*A*E*G*Izz*L**2*scf + 1440*E**2*Iyy*Izz) + L**2*r32*(11*A**2*G**2*L**4*intrho*scf**2 + 21*A**2*G**2*L**2*intrhoy2*scf**2 - 231*A*E*G*Izz*L**2*intrho*scf + 1260*A*E*G*Izz*intrhoy2*scf + 1260*E**2*Izz**2*intrho)/(210*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)) + L*intrhoy*r31*(-A*G*L**2*scf + 48*E*Izz)/(12*(A*G*L**2*scf - 12*E*Izz)))
                k += 1
                Mv[k] += r21*(L*intrhoy*r33*(7*A*G*L**2*scf - 80*E*Iyy)/(20*(A*G*L**2*scf - 12*E*Iyy)) + L*intrhoz*r32*(-7*A*G*L**2*scf + 80*E*Izz)/(20*(A*G*L**2*scf - 12*E*Izz))) + r22*(-A*G*L**2*intrhoyz*r32*scf*(A*G*L**2*scf + 60*E*Iyy)/(10*A**2*G**2*L**4*scf**2 - 120*A*E*G*Iyy*L**2*scf - 120*A*E*G*Izz*L**2*scf + 1440*E**2*Iyy*Izz) + L**2*r33*(11*A**2*G**2*L**4*intrho*scf**2 + 21*A**2*G**2*L**2*intrhoz2*scf**2 - 231*A*E*G*Iyy*L**2*intrho*scf + 1260*A*E*G*Iyy*intrhoz2*scf + 1260*E**2*Iyy**2*intrho)/(210*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)) + L*intrhoz*r31*(A*G*L**2*scf - 48*E*Iyy)/(12*(A*G*L**2*scf - 12*E*Iyy))) + r23*(-A*G*L**2*intrhoyz*r33*scf*(A*G*L**2*scf + 60*E*Izz)/(10*A**2*G**2*L**4*scf**2 - 120*A*E*G*Iyy*L**2*scf - 120*A*E*G*Izz*L**2*scf + 1440*E**2*Iyy*Izz) + L**2*r32*(11*A**2*G**2*L**4*intrho*scf**2 + 21*A**2*G**2*L**2*intrhoy2*scf**2 - 231*A*E*G*Izz*L**2*intrho*scf + 1260*A*E*G*Izz*intrhoy2*scf + 1260*E**2*Izz**2*intrho)/(210*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)) + L*intrhoy*r31*(-A*G*L**2*scf + 48*E*Izz)/(12*(A*G*L**2*scf - 12*E*Izz)))
                k += 1
                Mv[k] += r31*(L*intrhoy*r33*(7*A*G*L**2*scf - 80*E*Iyy)/(20*(A*G*L**2*scf - 12*E*Iyy)) + L*intrhoz*r32*(-7*A*G*L**2*scf + 80*E*Izz)/(20*(A*G*L**2*scf - 12*E*Izz))) + r32*(-A*G*L**2*intrhoyz*r32*scf*(A*G*L**2*scf + 60*E*Iyy)/(10*A**2*G**2*L**4*scf**2 - 120*A*E*G*Iyy*L**2*scf - 120*A*E*G*Izz*L**2*scf + 1440*E**2*Iyy*Izz) + L**2*r33*(11*A**2*G**2*L**4*intrho*scf**2 + 21*A**2*G**2*L**2*intrhoz2*scf**2 - 231*A*E*G*Iyy*L**2*intrho*scf + 1260*A*E*G*Iyy*intrhoz2*scf + 1260*E**2*Iyy**2*intrho)/(210*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)) + L*intrhoz*r31*(A*G*L**2*scf - 48*E*Iyy)/(12*(A*G*L**2*scf - 12*E*Iyy))) + r33*(-A*G*L**2*intrhoyz*r33*scf*(A*G*L**2*scf + 60*E*Izz)/(10*A**2*G**2*L**4*scf**2 - 120*A*E*G*Iyy*L**2*scf - 120*A*E*G*Izz*L**2*scf + 1440*E**2*Iyy*Izz) + L**2*r32*(11*A**2*G**2*L**4*intrho*scf**2 + 21*A**2*G**2*L**2*intrhoy2*scf**2 - 231*A*E*G*Izz*L**2*intrho*scf + 1260*A*E*G*Izz*intrhoy2*scf + 1260*E**2*Izz**2*intrho)/(210*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)) + L*intrhoy*r31*(-A*G*L**2*scf + 48*E*Izz)/(12*(A*G*L**2*scf - 12*E*Izz)))
                k += 1
                Mv[k] += r11*(A*G*L**2*intrhoy*r32*scf/(2*(A*G*L**2*scf - 12*E*Izz)) - A*G*L**2*intrhoz*r33*scf/(2*A*G*L**2*scf - 24*E*Iyy) + L*intrho*r31/6) + r12*(6*A**2*G**2*L**3*intrhoyz*r33*scf**2/(5*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) - A*G*L**2*intrhoy*r31*scf/(2*A*G*L**2*scf - 24*E*Izz) + r32*(9*A**2*G**2*L**5*intrho*scf**2 - 84*A**2*G**2*L**3*intrhoy2*scf**2 - 252*A*E*G*Izz*L**3*intrho*scf + 1680*E**2*Izz**2*L*intrho)/(70*A**2*G**2*L**4*scf**2 - 1680*A*E*G*Izz*L**2*scf + 10080*E**2*Izz**2)) + r13*(6*A**2*G**2*L**3*intrhoyz*r32*scf**2/(5*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + A*G*L**2*intrhoz*r31*scf/(2*(A*G*L**2*scf - 12*E*Iyy)) + r33*(9*A**2*G**2*L**5*intrho*scf**2 - 84*A**2*G**2*L**3*intrhoz2*scf**2 - 252*A*E*G*Iyy*L**3*intrho*scf + 1680*E**2*Iyy**2*L*intrho)/(70*A**2*G**2*L**4*scf**2 - 1680*A*E*G*Iyy*L**2*scf + 10080*E**2*Iyy**2))
                k += 1
                Mv[k] += r21*(A*G*L**2*intrhoy*r32*scf/(2*(A*G*L**2*scf - 12*E*Izz)) - A*G*L**2*intrhoz*r33*scf/(2*A*G*L**2*scf - 24*E*Iyy) + L*intrho*r31/6) + r22*(6*A**2*G**2*L**3*intrhoyz*r33*scf**2/(5*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) - A*G*L**2*intrhoy*r31*scf/(2*A*G*L**2*scf - 24*E*Izz) + r32*(9*A**2*G**2*L**5*intrho*scf**2 - 84*A**2*G**2*L**3*intrhoy2*scf**2 - 252*A*E*G*Izz*L**3*intrho*scf + 1680*E**2*Izz**2*L*intrho)/(70*A**2*G**2*L**4*scf**2 - 1680*A*E*G*Izz*L**2*scf + 10080*E**2*Izz**2)) + r23*(6*A**2*G**2*L**3*intrhoyz*r32*scf**2/(5*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + A*G*L**2*intrhoz*r31*scf/(2*(A*G*L**2*scf - 12*E*Iyy)) + r33*(9*A**2*G**2*L**5*intrho*scf**2 - 84*A**2*G**2*L**3*intrhoz2*scf**2 - 252*A*E*G*Iyy*L**3*intrho*scf + 1680*E**2*Iyy**2*L*intrho)/(70*A**2*G**2*L**4*scf**2 - 1680*A*E*G*Iyy*L**2*scf + 10080*E**2*Iyy**2))
                k += 1
                Mv[k] += r31*(A*G*L**2*intrhoy*r32*scf/(2*(A*G*L**2*scf - 12*E*Izz)) - A*G*L**2*intrhoz*r33*scf/(2*A*G*L**2*scf - 24*E*Iyy) + L*intrho*r31/6) + r32*(6*A**2*G**2*L**3*intrhoyz*r33*scf**2/(5*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) - A*G*L**2*intrhoy*r31*scf/(2*A*G*L**2*scf - 24*E*Izz) + r32*(9*A**2*G**2*L**5*intrho*scf**2 - 84*A**2*G**2*L**3*intrhoy2*scf**2 - 252*A*E*G*Izz*L**3*intrho*scf + 1680*E**2*Izz**2*L*intrho)/(70*A**2*G**2*L**4*scf**2 - 1680*A*E*G*Izz*L**2*scf + 10080*E**2*Izz**2)) + r33*(6*A**2*G**2*L**3*intrhoyz*r32*scf**2/(5*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + A*G*L**2*intrhoz*r31*scf/(2*(A*G*L**2*scf - 12*E*Iyy)) + r33*(9*A**2*G**2*L**5*intrho*scf**2 - 84*A**2*G**2*L**3*intrhoz2*scf**2 - 252*A*E*G*Iyy*L**3*intrho*scf + 1680*E**2*Iyy**2*L*intrho)/(70*A**2*G**2*L**4*scf**2 - 1680*A*E*G*Iyy*L**2*scf + 10080*E**2*Iyy**2))
                k += 1
                Mv[k] += r11*(L*intrhoy*r33*(3*A*G*L**2*scf - 40*E*Iyy)/(20*(A*G*L**2*scf - 12*E*Iyy)) + L*intrhoz*r32*(-3*A*G*L**2*scf + 40*E*Izz)/(20*(A*G*L**2*scf - 12*E*Izz))) + r12*(-A*G*L**2*intrhoyz*r32*scf*(A*G*L**2*scf + 60*E*Iyy)/(10*A**2*G**2*L**4*scf**2 - 120*A*E*G*Iyy*L**2*scf - 120*A*E*G*Izz*L**2*scf + 1440*E**2*Iyy*Izz) + L**2*r33*(-13*A**2*G**2*L**4*intrho*scf**2 + 42*A**2*G**2*L**2*intrhoz2*scf**2 + 378*A*E*G*Iyy*L**2*intrho*scf + 2520*A*E*G*Iyy*intrhoz2*scf - 2520*E**2*Iyy**2*intrho)/(420*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)) - L*intrhoz*r31*(A*G*L**2*scf + 24*E*Iyy)/(12*A*G*L**2*scf - 144*E*Iyy)) + r13*(-A*G*L**2*intrhoyz*r33*scf*(A*G*L**2*scf + 60*E*Izz)/(10*A**2*G**2*L**4*scf**2 - 120*A*E*G*Iyy*L**2*scf - 120*A*E*G*Izz*L**2*scf + 1440*E**2*Iyy*Izz) + L**2*r32*(-13*A**2*G**2*L**4*intrho*scf**2 + 42*A**2*G**2*L**2*intrhoy2*scf**2 + 378*A*E*G*Izz*L**2*intrho*scf + 2520*A*E*G*Izz*intrhoy2*scf - 2520*E**2*Izz**2*intrho)/(420*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)) + L*intrhoy*r31*(A*G*L**2*scf + 24*E*Izz)/(12*(A*G*L**2*scf - 12*E*Izz)))
                k += 1
                Mv[k] += r21*(L*intrhoy*r33*(3*A*G*L**2*scf - 40*E*Iyy)/(20*(A*G*L**2*scf - 12*E*Iyy)) + L*intrhoz*r32*(-3*A*G*L**2*scf + 40*E*Izz)/(20*(A*G*L**2*scf - 12*E*Izz))) + r22*(-A*G*L**2*intrhoyz*r32*scf*(A*G*L**2*scf + 60*E*Iyy)/(10*A**2*G**2*L**4*scf**2 - 120*A*E*G*Iyy*L**2*scf - 120*A*E*G*Izz*L**2*scf + 1440*E**2*Iyy*Izz) + L**2*r33*(-13*A**2*G**2*L**4*intrho*scf**2 + 42*A**2*G**2*L**2*intrhoz2*scf**2 + 378*A*E*G*Iyy*L**2*intrho*scf + 2520*A*E*G*Iyy*intrhoz2*scf - 2520*E**2*Iyy**2*intrho)/(420*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)) - L*intrhoz*r31*(A*G*L**2*scf + 24*E*Iyy)/(12*A*G*L**2*scf - 144*E*Iyy)) + r23*(-A*G*L**2*intrhoyz*r33*scf*(A*G*L**2*scf + 60*E*Izz)/(10*A**2*G**2*L**4*scf**2 - 120*A*E*G*Iyy*L**2*scf - 120*A*E*G*Izz*L**2*scf + 1440*E**2*Iyy*Izz) + L**2*r32*(-13*A**2*G**2*L**4*intrho*scf**2 + 42*A**2*G**2*L**2*intrhoy2*scf**2 + 378*A*E*G*Izz*L**2*intrho*scf + 2520*A*E*G*Izz*intrhoy2*scf - 2520*E**2*Izz**2*intrho)/(420*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)) + L*intrhoy*r31*(A*G*L**2*scf + 24*E*Izz)/(12*(A*G*L**2*scf - 12*E*Izz)))
                k += 1
                Mv[k] += r31*(L*intrhoy*r33*(3*A*G*L**2*scf - 40*E*Iyy)/(20*(A*G*L**2*scf - 12*E*Iyy)) + L*intrhoz*r32*(-3*A*G*L**2*scf + 40*E*Izz)/(20*(A*G*L**2*scf - 12*E*Izz))) + r32*(-A*G*L**2*intrhoyz*r32*scf*(A*G*L**2*scf + 60*E*Iyy)/(10*A**2*G**2*L**4*scf**2 - 120*A*E*G*Iyy*L**2*scf - 120*A*E*G*Izz*L**2*scf + 1440*E**2*Iyy*Izz) + L**2*r33*(-13*A**2*G**2*L**4*intrho*scf**2 + 42*A**2*G**2*L**2*intrhoz2*scf**2 + 378*A*E*G*Iyy*L**2*intrho*scf + 2520*A*E*G*Iyy*intrhoz2*scf - 2520*E**2*Iyy**2*intrho)/(420*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)) - L*intrhoz*r31*(A*G*L**2*scf + 24*E*Iyy)/(12*A*G*L**2*scf - 144*E*Iyy)) + r33*(-A*G*L**2*intrhoyz*r33*scf*(A*G*L**2*scf + 60*E*Izz)/(10*A**2*G**2*L**4*scf**2 - 120*A*E*G*Iyy*L**2*scf - 120*A*E*G*Izz*L**2*scf + 1440*E**2*Iyy*Izz) + L**2*r32*(-13*A**2*G**2*L**4*intrho*scf**2 + 42*A**2*G**2*L**2*intrhoy2*scf**2 + 378*A*E*G*Izz*L**2*intrho*scf + 2520*A*E*G*Izz*intrhoy2*scf - 2520*E**2*Izz**2*intrho)/(420*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)) + L*intrhoy*r31*(A*G*L**2*scf + 24*E*Izz)/(12*(A*G*L**2*scf - 12*E*Izz)))
                k += 1
                Mv[k] += r11*(L*intrhoy*r13*(-A*G*L**2*scf + 48*E*Izz)/(12*(A*G*L**2*scf - 12*E*Izz)) + L*intrhoz*r12*(A*G*L**2*scf - 48*E*Iyy)/(12*(A*G*L**2*scf - 12*E*Iyy))) + r12*(-A*G*L**2*intrhoyz*r12*scf*(A*G*L**2*scf + 60*E*Iyy)/(10*A**2*G**2*L**4*scf**2 - 120*A*E*G*Iyy*L**2*scf - 120*A*E*G*Izz*L**2*scf + 1440*E**2*Iyy*Izz) + L**2*r13*(11*A**2*G**2*L**4*intrho*scf**2 + 21*A**2*G**2*L**2*intrhoy2*scf**2 - 231*A*E*G*Izz*L**2*intrho*scf + 1260*A*E*G*Izz*intrhoy2*scf + 1260*E**2*Izz**2*intrho)/(210*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)) + L*intrhoz*r11*(-7*A*G*L**2*scf + 80*E*Izz)/(20*(A*G*L**2*scf - 12*E*Izz))) + r13*(-A*G*L**2*intrhoyz*r13*scf*(A*G*L**2*scf + 60*E*Izz)/(10*A**2*G**2*L**4*scf**2 - 120*A*E*G*Iyy*L**2*scf - 120*A*E*G*Izz*L**2*scf + 1440*E**2*Iyy*Izz) + L**2*r12*(11*A**2*G**2*L**4*intrho*scf**2 + 21*A**2*G**2*L**2*intrhoz2*scf**2 - 231*A*E*G*Iyy*L**2*intrho*scf + 1260*A*E*G*Iyy*intrhoz2*scf + 1260*E**2*Iyy**2*intrho)/(210*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)) + L*intrhoy*r11*(7*A*G*L**2*scf - 80*E*Iyy)/(20*(A*G*L**2*scf - 12*E*Iyy)))
                k += 1
                Mv[k] += r21*(L*intrhoy*r13*(-A*G*L**2*scf + 48*E*Izz)/(12*(A*G*L**2*scf - 12*E*Izz)) + L*intrhoz*r12*(A*G*L**2*scf - 48*E*Iyy)/(12*(A*G*L**2*scf - 12*E*Iyy))) + r22*(-A*G*L**2*intrhoyz*r12*scf*(A*G*L**2*scf + 60*E*Iyy)/(10*A**2*G**2*L**4*scf**2 - 120*A*E*G*Iyy*L**2*scf - 120*A*E*G*Izz*L**2*scf + 1440*E**2*Iyy*Izz) + L**2*r13*(11*A**2*G**2*L**4*intrho*scf**2 + 21*A**2*G**2*L**2*intrhoy2*scf**2 - 231*A*E*G*Izz*L**2*intrho*scf + 1260*A*E*G*Izz*intrhoy2*scf + 1260*E**2*Izz**2*intrho)/(210*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)) + L*intrhoz*r11*(-7*A*G*L**2*scf + 80*E*Izz)/(20*(A*G*L**2*scf - 12*E*Izz))) + r23*(-A*G*L**2*intrhoyz*r13*scf*(A*G*L**2*scf + 60*E*Izz)/(10*A**2*G**2*L**4*scf**2 - 120*A*E*G*Iyy*L**2*scf - 120*A*E*G*Izz*L**2*scf + 1440*E**2*Iyy*Izz) + L**2*r12*(11*A**2*G**2*L**4*intrho*scf**2 + 21*A**2*G**2*L**2*intrhoz2*scf**2 - 231*A*E*G*Iyy*L**2*intrho*scf + 1260*A*E*G*Iyy*intrhoz2*scf + 1260*E**2*Iyy**2*intrho)/(210*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)) + L*intrhoy*r11*(7*A*G*L**2*scf - 80*E*Iyy)/(20*(A*G*L**2*scf - 12*E*Iyy)))
                k += 1
                Mv[k] += r31*(L*intrhoy*r13*(-A*G*L**2*scf + 48*E*Izz)/(12*(A*G*L**2*scf - 12*E*Izz)) + L*intrhoz*r12*(A*G*L**2*scf - 48*E*Iyy)/(12*(A*G*L**2*scf - 12*E*Iyy))) + r32*(-A*G*L**2*intrhoyz*r12*scf*(A*G*L**2*scf + 60*E*Iyy)/(10*A**2*G**2*L**4*scf**2 - 120*A*E*G*Iyy*L**2*scf - 120*A*E*G*Izz*L**2*scf + 1440*E**2*Iyy*Izz) + L**2*r13*(11*A**2*G**2*L**4*intrho*scf**2 + 21*A**2*G**2*L**2*intrhoy2*scf**2 - 231*A*E*G*Izz*L**2*intrho*scf + 1260*A*E*G*Izz*intrhoy2*scf + 1260*E**2*Izz**2*intrho)/(210*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)) + L*intrhoz*r11*(-7*A*G*L**2*scf + 80*E*Izz)/(20*(A*G*L**2*scf - 12*E*Izz))) + r33*(-A*G*L**2*intrhoyz*r13*scf*(A*G*L**2*scf + 60*E*Izz)/(10*A**2*G**2*L**4*scf**2 - 120*A*E*G*Iyy*L**2*scf - 120*A*E*G*Izz*L**2*scf + 1440*E**2*Iyy*Izz) + L**2*r12*(11*A**2*G**2*L**4*intrho*scf**2 + 21*A**2*G**2*L**2*intrhoz2*scf**2 - 231*A*E*G*Iyy*L**2*intrho*scf + 1260*A*E*G*Iyy*intrhoz2*scf + 1260*E**2*Iyy**2*intrho)/(210*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)) + L*intrhoy*r11*(7*A*G*L**2*scf - 80*E*Iyy)/(20*(A*G*L**2*scf - 12*E*Iyy)))
                k += 1
                Mv[k] += r11*(L**2*intrhoy*r12*(A*G*L**2*scf - 10*E*Iyy)/(20*(A*G*L**2*scf - 12*E*Iyy)) + L**2*intrhoz*r13*(-A*G*L**2*scf + 10*E*Izz)/(20*(A*G*L**2*scf - 12*E*Izz)) + L*r11*(intrhoy2 + intrhoz2)/3) + r12*(L**2*intrhoy*r11*(A*G*L**2*scf - 10*E*Iyy)/(20*(A*G*L**2*scf - 12*E*Iyy)) + L*intrhoyz*r13*(-2*A**2*G**2*L**4*scf**2 + 15*A*E*G*Iyy*L**2*scf + 15*A*E*G*Izz*L**2*scf - 720*E**2*Iyy*Izz)/(15*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + L*r12*(A**2*G**2*L**6*intrho*scf**2 + 14*A**2*G**2*L**4*intrhoz2*scf**2 - 21*A*E*G*Iyy*L**4*intrho*scf - 210*A*E*G*Iyy*L**2*intrhoz2*scf + 126*E**2*Iyy**2*L**2*intrho + 5040*E**2*Iyy**2*intrhoz2)/(105*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2))) + r13*(L**2*intrhoz*r11*(-A*G*L**2*scf + 10*E*Izz)/(20*(A*G*L**2*scf - 12*E*Izz)) + L*intrhoyz*r12*(-2*A**2*G**2*L**4*scf**2 + 15*A*E*G*Iyy*L**2*scf + 15*A*E*G*Izz*L**2*scf - 720*E**2*Iyy*Izz)/(15*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + L*r13*(A**2*G**2*L**6*intrho*scf**2 + 14*A**2*G**2*L**4*intrhoy2*scf**2 - 21*A*E*G*Izz*L**4*intrho*scf - 210*A*E*G*Izz*L**2*intrhoy2*scf + 126*E**2*Izz**2*L**2*intrho + 5040*E**2*Izz**2*intrhoy2)/(105*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)))
                k += 1
                Mv[k] += r21*(L**2*intrhoy*r12*(A*G*L**2*scf - 10*E*Iyy)/(20*(A*G*L**2*scf - 12*E*Iyy)) + L**2*intrhoz*r13*(-A*G*L**2*scf + 10*E*Izz)/(20*(A*G*L**2*scf - 12*E*Izz)) + L*r11*(intrhoy2 + intrhoz2)/3) + r22*(L**2*intrhoy*r11*(A*G*L**2*scf - 10*E*Iyy)/(20*(A*G*L**2*scf - 12*E*Iyy)) + L*intrhoyz*r13*(-2*A**2*G**2*L**4*scf**2 + 15*A*E*G*Iyy*L**2*scf + 15*A*E*G*Izz*L**2*scf - 720*E**2*Iyy*Izz)/(15*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + L*r12*(A**2*G**2*L**6*intrho*scf**2 + 14*A**2*G**2*L**4*intrhoz2*scf**2 - 21*A*E*G*Iyy*L**4*intrho*scf - 210*A*E*G*Iyy*L**2*intrhoz2*scf + 126*E**2*Iyy**2*L**2*intrho + 5040*E**2*Iyy**2*intrhoz2)/(105*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2))) + r23*(L**2*intrhoz*r11*(-A*G*L**2*scf + 10*E*Izz)/(20*(A*G*L**2*scf - 12*E*Izz)) + L*intrhoyz*r12*(-2*A**2*G**2*L**4*scf**2 + 15*A*E*G*Iyy*L**2*scf + 15*A*E*G*Izz*L**2*scf - 720*E**2*Iyy*Izz)/(15*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + L*r13*(A**2*G**2*L**6*intrho*scf**2 + 14*A**2*G**2*L**4*intrhoy2*scf**2 - 21*A*E*G*Izz*L**4*intrho*scf - 210*A*E*G*Izz*L**2*intrhoy2*scf + 126*E**2*Izz**2*L**2*intrho + 5040*E**2*Izz**2*intrhoy2)/(105*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)))
                k += 1
                Mv[k] += r31*(L**2*intrhoy*r12*(A*G*L**2*scf - 10*E*Iyy)/(20*(A*G*L**2*scf - 12*E*Iyy)) + L**2*intrhoz*r13*(-A*G*L**2*scf + 10*E*Izz)/(20*(A*G*L**2*scf - 12*E*Izz)) + L*r11*(intrhoy2 + intrhoz2)/3) + r32*(L**2*intrhoy*r11*(A*G*L**2*scf - 10*E*Iyy)/(20*(A*G*L**2*scf - 12*E*Iyy)) + L*intrhoyz*r13*(-2*A**2*G**2*L**4*scf**2 + 15*A*E*G*Iyy*L**2*scf + 15*A*E*G*Izz*L**2*scf - 720*E**2*Iyy*Izz)/(15*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + L*r12*(A**2*G**2*L**6*intrho*scf**2 + 14*A**2*G**2*L**4*intrhoz2*scf**2 - 21*A*E*G*Iyy*L**4*intrho*scf - 210*A*E*G*Iyy*L**2*intrhoz2*scf + 126*E**2*Iyy**2*L**2*intrho + 5040*E**2*Iyy**2*intrhoz2)/(105*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2))) + r33*(L**2*intrhoz*r11*(-A*G*L**2*scf + 10*E*Izz)/(20*(A*G*L**2*scf - 12*E*Izz)) + L*intrhoyz*r12*(-2*A**2*G**2*L**4*scf**2 + 15*A*E*G*Iyy*L**2*scf + 15*A*E*G*Izz*L**2*scf - 720*E**2*Iyy*Izz)/(15*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + L*r13*(A**2*G**2*L**6*intrho*scf**2 + 14*A**2*G**2*L**4*intrhoy2*scf**2 - 21*A*E*G*Izz*L**4*intrho*scf - 210*A*E*G*Izz*L**2*intrhoy2*scf + 126*E**2*Izz**2*L**2*intrho + 5040*E**2*Izz**2*intrhoy2)/(105*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)))
                k += 1
                Mv[k] += r11*(L*intrhoy*r13*(A*G*L**2*scf + 24*E*Izz)/(12*(A*G*L**2*scf - 12*E*Izz)) - L*intrhoz*r12*(A*G*L**2*scf + 24*E*Iyy)/(12*A*G*L**2*scf - 144*E*Iyy)) + r12*(A*G*L**2*intrhoyz*r12*scf*(A*G*L**2*scf + 60*E*Iyy)/(10*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + L**2*r13*(13*A**2*G**2*L**4*intrho*scf**2 - 42*A**2*G**2*L**2*intrhoy2*scf**2 - 378*A*E*G*Izz*L**2*intrho*scf - 2520*A*E*G*Izz*intrhoy2*scf + 2520*E**2*Izz**2*intrho)/(420*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)) + L*intrhoz*r11*(-3*A*G*L**2*scf + 40*E*Izz)/(20*(A*G*L**2*scf - 12*E*Izz))) + r13*(A*G*L**2*intrhoyz*r13*scf*(A*G*L**2*scf + 60*E*Izz)/(10*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + L**2*r12*(13*A**2*G**2*L**4*intrho*scf**2 - 42*A**2*G**2*L**2*intrhoz2*scf**2 - 378*A*E*G*Iyy*L**2*intrho*scf - 2520*A*E*G*Iyy*intrhoz2*scf + 2520*E**2*Iyy**2*intrho)/(420*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)) + L*intrhoy*r11*(3*A*G*L**2*scf - 40*E*Iyy)/(20*(A*G*L**2*scf - 12*E*Iyy)))
                k += 1
                Mv[k] += r21*(L*intrhoy*r13*(A*G*L**2*scf + 24*E*Izz)/(12*(A*G*L**2*scf - 12*E*Izz)) - L*intrhoz*r12*(A*G*L**2*scf + 24*E*Iyy)/(12*A*G*L**2*scf - 144*E*Iyy)) + r22*(A*G*L**2*intrhoyz*r12*scf*(A*G*L**2*scf + 60*E*Iyy)/(10*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + L**2*r13*(13*A**2*G**2*L**4*intrho*scf**2 - 42*A**2*G**2*L**2*intrhoy2*scf**2 - 378*A*E*G*Izz*L**2*intrho*scf - 2520*A*E*G*Izz*intrhoy2*scf + 2520*E**2*Izz**2*intrho)/(420*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)) + L*intrhoz*r11*(-3*A*G*L**2*scf + 40*E*Izz)/(20*(A*G*L**2*scf - 12*E*Izz))) + r23*(A*G*L**2*intrhoyz*r13*scf*(A*G*L**2*scf + 60*E*Izz)/(10*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + L**2*r12*(13*A**2*G**2*L**4*intrho*scf**2 - 42*A**2*G**2*L**2*intrhoz2*scf**2 - 378*A*E*G*Iyy*L**2*intrho*scf - 2520*A*E*G*Iyy*intrhoz2*scf + 2520*E**2*Iyy**2*intrho)/(420*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)) + L*intrhoy*r11*(3*A*G*L**2*scf - 40*E*Iyy)/(20*(A*G*L**2*scf - 12*E*Iyy)))
                k += 1
                Mv[k] += r31*(L*intrhoy*r13*(A*G*L**2*scf + 24*E*Izz)/(12*(A*G*L**2*scf - 12*E*Izz)) - L*intrhoz*r12*(A*G*L**2*scf + 24*E*Iyy)/(12*A*G*L**2*scf - 144*E*Iyy)) + r32*(A*G*L**2*intrhoyz*r12*scf*(A*G*L**2*scf + 60*E*Iyy)/(10*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + L**2*r13*(13*A**2*G**2*L**4*intrho*scf**2 - 42*A**2*G**2*L**2*intrhoy2*scf**2 - 378*A*E*G*Izz*L**2*intrho*scf - 2520*A*E*G*Izz*intrhoy2*scf + 2520*E**2*Izz**2*intrho)/(420*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)) + L*intrhoz*r11*(-3*A*G*L**2*scf + 40*E*Izz)/(20*(A*G*L**2*scf - 12*E*Izz))) + r33*(A*G*L**2*intrhoyz*r13*scf*(A*G*L**2*scf + 60*E*Izz)/(10*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + L**2*r12*(13*A**2*G**2*L**4*intrho*scf**2 - 42*A**2*G**2*L**2*intrhoz2*scf**2 - 378*A*E*G*Iyy*L**2*intrho*scf - 2520*A*E*G*Iyy*intrhoz2*scf + 2520*E**2*Iyy**2*intrho)/(420*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)) + L*intrhoy*r11*(3*A*G*L**2*scf - 40*E*Iyy)/(20*(A*G*L**2*scf - 12*E*Iyy)))
                k += 1
                Mv[k] += r11*(L**2*intrhoy*r12*(A*G*L**2*scf - 15*E*Iyy)/(30*(A*G*L**2*scf - 12*E*Iyy)) + L**2*intrhoz*r13*(-A*G*L**2*scf + 15*E*Izz)/(30*(A*G*L**2*scf - 12*E*Izz)) + L*r11*(intrhoy2 + intrhoz2)/6) + r12*(L**2*intrhoy*r11*(-A*G*L**2*scf + 15*E*Iyy)/(30*(A*G*L**2*scf - 12*E*Iyy)) + L*intrhoyz*r13*(A**2*G**2*L**4*scf**2 - 30*A*E*G*Iyy*L**2*scf - 30*A*E*G*Izz*L**2*scf - 720*E**2*Iyy*Izz)/(30*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + L*r12*(-3*A**2*G**2*L**6*intrho*scf**2 - 14*A**2*G**2*L**4*intrhoz2*scf**2 + 84*A*E*G*Iyy*L**4*intrho*scf + 840*A*E*G*Iyy*L**2*intrhoz2*scf - 504*E**2*Iyy**2*L**2*intrho + 10080*E**2*Iyy**2*intrhoz2)/(420*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2))) + r13*(L**2*intrhoz*r11*(A*G*L**2*scf - 15*E*Izz)/(30*(A*G*L**2*scf - 12*E*Izz)) + L*intrhoyz*r12*(A**2*G**2*L**4*scf**2 - 30*A*E*G*Iyy*L**2*scf - 30*A*E*G*Izz*L**2*scf - 720*E**2*Iyy*Izz)/(30*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + L*r13*(-3*A**2*G**2*L**6*intrho*scf**2 - 14*A**2*G**2*L**4*intrhoy2*scf**2 + 84*A*E*G*Izz*L**4*intrho*scf + 840*A*E*G*Izz*L**2*intrhoy2*scf - 504*E**2*Izz**2*L**2*intrho + 10080*E**2*Izz**2*intrhoy2)/(420*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)))
                k += 1
                Mv[k] += r21*(L**2*intrhoy*r12*(A*G*L**2*scf - 15*E*Iyy)/(30*(A*G*L**2*scf - 12*E*Iyy)) + L**2*intrhoz*r13*(-A*G*L**2*scf + 15*E*Izz)/(30*(A*G*L**2*scf - 12*E*Izz)) + L*r11*(intrhoy2 + intrhoz2)/6) + r22*(L**2*intrhoy*r11*(-A*G*L**2*scf + 15*E*Iyy)/(30*(A*G*L**2*scf - 12*E*Iyy)) + L*intrhoyz*r13*(A**2*G**2*L**4*scf**2 - 30*A*E*G*Iyy*L**2*scf - 30*A*E*G*Izz*L**2*scf - 720*E**2*Iyy*Izz)/(30*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + L*r12*(-3*A**2*G**2*L**6*intrho*scf**2 - 14*A**2*G**2*L**4*intrhoz2*scf**2 + 84*A*E*G*Iyy*L**4*intrho*scf + 840*A*E*G*Iyy*L**2*intrhoz2*scf - 504*E**2*Iyy**2*L**2*intrho + 10080*E**2*Iyy**2*intrhoz2)/(420*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2))) + r23*(L**2*intrhoz*r11*(A*G*L**2*scf - 15*E*Izz)/(30*(A*G*L**2*scf - 12*E*Izz)) + L*intrhoyz*r12*(A**2*G**2*L**4*scf**2 - 30*A*E*G*Iyy*L**2*scf - 30*A*E*G*Izz*L**2*scf - 720*E**2*Iyy*Izz)/(30*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + L*r13*(-3*A**2*G**2*L**6*intrho*scf**2 - 14*A**2*G**2*L**4*intrhoy2*scf**2 + 84*A*E*G*Izz*L**4*intrho*scf + 840*A*E*G*Izz*L**2*intrhoy2*scf - 504*E**2*Izz**2*L**2*intrho + 10080*E**2*Izz**2*intrhoy2)/(420*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)))
                k += 1
                Mv[k] += r31*(L**2*intrhoy*r12*(A*G*L**2*scf - 15*E*Iyy)/(30*(A*G*L**2*scf - 12*E*Iyy)) + L**2*intrhoz*r13*(-A*G*L**2*scf + 15*E*Izz)/(30*(A*G*L**2*scf - 12*E*Izz)) + L*r11*(intrhoy2 + intrhoz2)/6) + r32*(L**2*intrhoy*r11*(-A*G*L**2*scf + 15*E*Iyy)/(30*(A*G*L**2*scf - 12*E*Iyy)) + L*intrhoyz*r13*(A**2*G**2*L**4*scf**2 - 30*A*E*G*Iyy*L**2*scf - 30*A*E*G*Izz*L**2*scf - 720*E**2*Iyy*Izz)/(30*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + L*r12*(-3*A**2*G**2*L**6*intrho*scf**2 - 14*A**2*G**2*L**4*intrhoz2*scf**2 + 84*A*E*G*Iyy*L**4*intrho*scf + 840*A*E*G*Iyy*L**2*intrhoz2*scf - 504*E**2*Iyy**2*L**2*intrho + 10080*E**2*Iyy**2*intrhoz2)/(420*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2))) + r33*(L**2*intrhoz*r11*(A*G*L**2*scf - 15*E*Izz)/(30*(A*G*L**2*scf - 12*E*Izz)) + L*intrhoyz*r12*(A**2*G**2*L**4*scf**2 - 30*A*E*G*Iyy*L**2*scf - 30*A*E*G*Izz*L**2*scf - 720*E**2*Iyy*Izz)/(30*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + L*r13*(-3*A**2*G**2*L**6*intrho*scf**2 - 14*A**2*G**2*L**4*intrhoy2*scf**2 + 84*A*E*G*Izz*L**4*intrho*scf + 840*A*E*G*Izz*L**2*intrhoy2*scf - 504*E**2*Izz**2*L**2*intrho + 10080*E**2*Izz**2*intrhoy2)/(420*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)))
                k += 1
                Mv[k] += r11*(L*intrhoy*r23*(-A*G*L**2*scf + 48*E*Izz)/(12*(A*G*L**2*scf - 12*E*Izz)) + L*intrhoz*r22*(A*G*L**2*scf - 48*E*Iyy)/(12*(A*G*L**2*scf - 12*E*Iyy))) + r12*(-A*G*L**2*intrhoyz*r22*scf*(A*G*L**2*scf + 60*E*Iyy)/(10*A**2*G**2*L**4*scf**2 - 120*A*E*G*Iyy*L**2*scf - 120*A*E*G*Izz*L**2*scf + 1440*E**2*Iyy*Izz) + L**2*r23*(11*A**2*G**2*L**4*intrho*scf**2 + 21*A**2*G**2*L**2*intrhoy2*scf**2 - 231*A*E*G*Izz*L**2*intrho*scf + 1260*A*E*G*Izz*intrhoy2*scf + 1260*E**2*Izz**2*intrho)/(210*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)) + L*intrhoz*r21*(-7*A*G*L**2*scf + 80*E*Izz)/(20*(A*G*L**2*scf - 12*E*Izz))) + r13*(-A*G*L**2*intrhoyz*r23*scf*(A*G*L**2*scf + 60*E*Izz)/(10*A**2*G**2*L**4*scf**2 - 120*A*E*G*Iyy*L**2*scf - 120*A*E*G*Izz*L**2*scf + 1440*E**2*Iyy*Izz) + L**2*r22*(11*A**2*G**2*L**4*intrho*scf**2 + 21*A**2*G**2*L**2*intrhoz2*scf**2 - 231*A*E*G*Iyy*L**2*intrho*scf + 1260*A*E*G*Iyy*intrhoz2*scf + 1260*E**2*Iyy**2*intrho)/(210*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)) + L*intrhoy*r21*(7*A*G*L**2*scf - 80*E*Iyy)/(20*(A*G*L**2*scf - 12*E*Iyy)))
                k += 1
                Mv[k] += r21*(L*intrhoy*r23*(-A*G*L**2*scf + 48*E*Izz)/(12*(A*G*L**2*scf - 12*E*Izz)) + L*intrhoz*r22*(A*G*L**2*scf - 48*E*Iyy)/(12*(A*G*L**2*scf - 12*E*Iyy))) + r22*(-A*G*L**2*intrhoyz*r22*scf*(A*G*L**2*scf + 60*E*Iyy)/(10*A**2*G**2*L**4*scf**2 - 120*A*E*G*Iyy*L**2*scf - 120*A*E*G*Izz*L**2*scf + 1440*E**2*Iyy*Izz) + L**2*r23*(11*A**2*G**2*L**4*intrho*scf**2 + 21*A**2*G**2*L**2*intrhoy2*scf**2 - 231*A*E*G*Izz*L**2*intrho*scf + 1260*A*E*G*Izz*intrhoy2*scf + 1260*E**2*Izz**2*intrho)/(210*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)) + L*intrhoz*r21*(-7*A*G*L**2*scf + 80*E*Izz)/(20*(A*G*L**2*scf - 12*E*Izz))) + r23*(-A*G*L**2*intrhoyz*r23*scf*(A*G*L**2*scf + 60*E*Izz)/(10*A**2*G**2*L**4*scf**2 - 120*A*E*G*Iyy*L**2*scf - 120*A*E*G*Izz*L**2*scf + 1440*E**2*Iyy*Izz) + L**2*r22*(11*A**2*G**2*L**4*intrho*scf**2 + 21*A**2*G**2*L**2*intrhoz2*scf**2 - 231*A*E*G*Iyy*L**2*intrho*scf + 1260*A*E*G*Iyy*intrhoz2*scf + 1260*E**2*Iyy**2*intrho)/(210*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)) + L*intrhoy*r21*(7*A*G*L**2*scf - 80*E*Iyy)/(20*(A*G*L**2*scf - 12*E*Iyy)))
                k += 1
                Mv[k] += r31*(L*intrhoy*r23*(-A*G*L**2*scf + 48*E*Izz)/(12*(A*G*L**2*scf - 12*E*Izz)) + L*intrhoz*r22*(A*G*L**2*scf - 48*E*Iyy)/(12*(A*G*L**2*scf - 12*E*Iyy))) + r32*(-A*G*L**2*intrhoyz*r22*scf*(A*G*L**2*scf + 60*E*Iyy)/(10*A**2*G**2*L**4*scf**2 - 120*A*E*G*Iyy*L**2*scf - 120*A*E*G*Izz*L**2*scf + 1440*E**2*Iyy*Izz) + L**2*r23*(11*A**2*G**2*L**4*intrho*scf**2 + 21*A**2*G**2*L**2*intrhoy2*scf**2 - 231*A*E*G*Izz*L**2*intrho*scf + 1260*A*E*G*Izz*intrhoy2*scf + 1260*E**2*Izz**2*intrho)/(210*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)) + L*intrhoz*r21*(-7*A*G*L**2*scf + 80*E*Izz)/(20*(A*G*L**2*scf - 12*E*Izz))) + r33*(-A*G*L**2*intrhoyz*r23*scf*(A*G*L**2*scf + 60*E*Izz)/(10*A**2*G**2*L**4*scf**2 - 120*A*E*G*Iyy*L**2*scf - 120*A*E*G*Izz*L**2*scf + 1440*E**2*Iyy*Izz) + L**2*r22*(11*A**2*G**2*L**4*intrho*scf**2 + 21*A**2*G**2*L**2*intrhoz2*scf**2 - 231*A*E*G*Iyy*L**2*intrho*scf + 1260*A*E*G*Iyy*intrhoz2*scf + 1260*E**2*Iyy**2*intrho)/(210*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)) + L*intrhoy*r21*(7*A*G*L**2*scf - 80*E*Iyy)/(20*(A*G*L**2*scf - 12*E*Iyy)))
                k += 1
                Mv[k] += r11*(L**2*intrhoy*r22*(A*G*L**2*scf - 10*E*Iyy)/(20*(A*G*L**2*scf - 12*E*Iyy)) + L**2*intrhoz*r23*(-A*G*L**2*scf + 10*E*Izz)/(20*(A*G*L**2*scf - 12*E*Izz)) + L*r21*(intrhoy2 + intrhoz2)/3) + r12*(L**2*intrhoy*r21*(A*G*L**2*scf - 10*E*Iyy)/(20*(A*G*L**2*scf - 12*E*Iyy)) + L*intrhoyz*r23*(-2*A**2*G**2*L**4*scf**2 + 15*A*E*G*Iyy*L**2*scf + 15*A*E*G*Izz*L**2*scf - 720*E**2*Iyy*Izz)/(15*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + L*r22*(A**2*G**2*L**6*intrho*scf**2 + 14*A**2*G**2*L**4*intrhoz2*scf**2 - 21*A*E*G*Iyy*L**4*intrho*scf - 210*A*E*G*Iyy*L**2*intrhoz2*scf + 126*E**2*Iyy**2*L**2*intrho + 5040*E**2*Iyy**2*intrhoz2)/(105*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2))) + r13*(L**2*intrhoz*r21*(-A*G*L**2*scf + 10*E*Izz)/(20*(A*G*L**2*scf - 12*E*Izz)) + L*intrhoyz*r22*(-2*A**2*G**2*L**4*scf**2 + 15*A*E*G*Iyy*L**2*scf + 15*A*E*G*Izz*L**2*scf - 720*E**2*Iyy*Izz)/(15*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + L*r23*(A**2*G**2*L**6*intrho*scf**2 + 14*A**2*G**2*L**4*intrhoy2*scf**2 - 21*A*E*G*Izz*L**4*intrho*scf - 210*A*E*G*Izz*L**2*intrhoy2*scf + 126*E**2*Izz**2*L**2*intrho + 5040*E**2*Izz**2*intrhoy2)/(105*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)))
                k += 1
                Mv[k] += r21*(L**2*intrhoy*r22*(A*G*L**2*scf - 10*E*Iyy)/(20*(A*G*L**2*scf - 12*E*Iyy)) + L**2*intrhoz*r23*(-A*G*L**2*scf + 10*E*Izz)/(20*(A*G*L**2*scf - 12*E*Izz)) + L*r21*(intrhoy2 + intrhoz2)/3) + r22*(L**2*intrhoy*r21*(A*G*L**2*scf - 10*E*Iyy)/(20*(A*G*L**2*scf - 12*E*Iyy)) + L*intrhoyz*r23*(-2*A**2*G**2*L**4*scf**2 + 15*A*E*G*Iyy*L**2*scf + 15*A*E*G*Izz*L**2*scf - 720*E**2*Iyy*Izz)/(15*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + L*r22*(A**2*G**2*L**6*intrho*scf**2 + 14*A**2*G**2*L**4*intrhoz2*scf**2 - 21*A*E*G*Iyy*L**4*intrho*scf - 210*A*E*G*Iyy*L**2*intrhoz2*scf + 126*E**2*Iyy**2*L**2*intrho + 5040*E**2*Iyy**2*intrhoz2)/(105*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2))) + r23*(L**2*intrhoz*r21*(-A*G*L**2*scf + 10*E*Izz)/(20*(A*G*L**2*scf - 12*E*Izz)) + L*intrhoyz*r22*(-2*A**2*G**2*L**4*scf**2 + 15*A*E*G*Iyy*L**2*scf + 15*A*E*G*Izz*L**2*scf - 720*E**2*Iyy*Izz)/(15*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + L*r23*(A**2*G**2*L**6*intrho*scf**2 + 14*A**2*G**2*L**4*intrhoy2*scf**2 - 21*A*E*G*Izz*L**4*intrho*scf - 210*A*E*G*Izz*L**2*intrhoy2*scf + 126*E**2*Izz**2*L**2*intrho + 5040*E**2*Izz**2*intrhoy2)/(105*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)))
                k += 1
                Mv[k] += r31*(L**2*intrhoy*r22*(A*G*L**2*scf - 10*E*Iyy)/(20*(A*G*L**2*scf - 12*E*Iyy)) + L**2*intrhoz*r23*(-A*G*L**2*scf + 10*E*Izz)/(20*(A*G*L**2*scf - 12*E*Izz)) + L*r21*(intrhoy2 + intrhoz2)/3) + r32*(L**2*intrhoy*r21*(A*G*L**2*scf - 10*E*Iyy)/(20*(A*G*L**2*scf - 12*E*Iyy)) + L*intrhoyz*r23*(-2*A**2*G**2*L**4*scf**2 + 15*A*E*G*Iyy*L**2*scf + 15*A*E*G*Izz*L**2*scf - 720*E**2*Iyy*Izz)/(15*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + L*r22*(A**2*G**2*L**6*intrho*scf**2 + 14*A**2*G**2*L**4*intrhoz2*scf**2 - 21*A*E*G*Iyy*L**4*intrho*scf - 210*A*E*G*Iyy*L**2*intrhoz2*scf + 126*E**2*Iyy**2*L**2*intrho + 5040*E**2*Iyy**2*intrhoz2)/(105*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2))) + r33*(L**2*intrhoz*r21*(-A*G*L**2*scf + 10*E*Izz)/(20*(A*G*L**2*scf - 12*E*Izz)) + L*intrhoyz*r22*(-2*A**2*G**2*L**4*scf**2 + 15*A*E*G*Iyy*L**2*scf + 15*A*E*G*Izz*L**2*scf - 720*E**2*Iyy*Izz)/(15*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + L*r23*(A**2*G**2*L**6*intrho*scf**2 + 14*A**2*G**2*L**4*intrhoy2*scf**2 - 21*A*E*G*Izz*L**4*intrho*scf - 210*A*E*G*Izz*L**2*intrhoy2*scf + 126*E**2*Izz**2*L**2*intrho + 5040*E**2*Izz**2*intrhoy2)/(105*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)))
                k += 1
                Mv[k] += r11*(L*intrhoy*r23*(A*G*L**2*scf + 24*E*Izz)/(12*(A*G*L**2*scf - 12*E*Izz)) - L*intrhoz*r22*(A*G*L**2*scf + 24*E*Iyy)/(12*A*G*L**2*scf - 144*E*Iyy)) + r12*(A*G*L**2*intrhoyz*r22*scf*(A*G*L**2*scf + 60*E*Iyy)/(10*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + L**2*r23*(13*A**2*G**2*L**4*intrho*scf**2 - 42*A**2*G**2*L**2*intrhoy2*scf**2 - 378*A*E*G*Izz*L**2*intrho*scf - 2520*A*E*G*Izz*intrhoy2*scf + 2520*E**2*Izz**2*intrho)/(420*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)) + L*intrhoz*r21*(-3*A*G*L**2*scf + 40*E*Izz)/(20*(A*G*L**2*scf - 12*E*Izz))) + r13*(A*G*L**2*intrhoyz*r23*scf*(A*G*L**2*scf + 60*E*Izz)/(10*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + L**2*r22*(13*A**2*G**2*L**4*intrho*scf**2 - 42*A**2*G**2*L**2*intrhoz2*scf**2 - 378*A*E*G*Iyy*L**2*intrho*scf - 2520*A*E*G*Iyy*intrhoz2*scf + 2520*E**2*Iyy**2*intrho)/(420*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)) + L*intrhoy*r21*(3*A*G*L**2*scf - 40*E*Iyy)/(20*(A*G*L**2*scf - 12*E*Iyy)))
                k += 1
                Mv[k] += r21*(L*intrhoy*r23*(A*G*L**2*scf + 24*E*Izz)/(12*(A*G*L**2*scf - 12*E*Izz)) - L*intrhoz*r22*(A*G*L**2*scf + 24*E*Iyy)/(12*A*G*L**2*scf - 144*E*Iyy)) + r22*(A*G*L**2*intrhoyz*r22*scf*(A*G*L**2*scf + 60*E*Iyy)/(10*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + L**2*r23*(13*A**2*G**2*L**4*intrho*scf**2 - 42*A**2*G**2*L**2*intrhoy2*scf**2 - 378*A*E*G*Izz*L**2*intrho*scf - 2520*A*E*G*Izz*intrhoy2*scf + 2520*E**2*Izz**2*intrho)/(420*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)) + L*intrhoz*r21*(-3*A*G*L**2*scf + 40*E*Izz)/(20*(A*G*L**2*scf - 12*E*Izz))) + r23*(A*G*L**2*intrhoyz*r23*scf*(A*G*L**2*scf + 60*E*Izz)/(10*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + L**2*r22*(13*A**2*G**2*L**4*intrho*scf**2 - 42*A**2*G**2*L**2*intrhoz2*scf**2 - 378*A*E*G*Iyy*L**2*intrho*scf - 2520*A*E*G*Iyy*intrhoz2*scf + 2520*E**2*Iyy**2*intrho)/(420*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)) + L*intrhoy*r21*(3*A*G*L**2*scf - 40*E*Iyy)/(20*(A*G*L**2*scf - 12*E*Iyy)))
                k += 1
                Mv[k] += r31*(L*intrhoy*r23*(A*G*L**2*scf + 24*E*Izz)/(12*(A*G*L**2*scf - 12*E*Izz)) - L*intrhoz*r22*(A*G*L**2*scf + 24*E*Iyy)/(12*A*G*L**2*scf - 144*E*Iyy)) + r32*(A*G*L**2*intrhoyz*r22*scf*(A*G*L**2*scf + 60*E*Iyy)/(10*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + L**2*r23*(13*A**2*G**2*L**4*intrho*scf**2 - 42*A**2*G**2*L**2*intrhoy2*scf**2 - 378*A*E*G*Izz*L**2*intrho*scf - 2520*A*E*G*Izz*intrhoy2*scf + 2520*E**2*Izz**2*intrho)/(420*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)) + L*intrhoz*r21*(-3*A*G*L**2*scf + 40*E*Izz)/(20*(A*G*L**2*scf - 12*E*Izz))) + r33*(A*G*L**2*intrhoyz*r23*scf*(A*G*L**2*scf + 60*E*Izz)/(10*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + L**2*r22*(13*A**2*G**2*L**4*intrho*scf**2 - 42*A**2*G**2*L**2*intrhoz2*scf**2 - 378*A*E*G*Iyy*L**2*intrho*scf - 2520*A*E*G*Iyy*intrhoz2*scf + 2520*E**2*Iyy**2*intrho)/(420*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)) + L*intrhoy*r21*(3*A*G*L**2*scf - 40*E*Iyy)/(20*(A*G*L**2*scf - 12*E*Iyy)))
                k += 1
                Mv[k] += r11*(L**2*intrhoy*r22*(A*G*L**2*scf - 15*E*Iyy)/(30*(A*G*L**2*scf - 12*E*Iyy)) + L**2*intrhoz*r23*(-A*G*L**2*scf + 15*E*Izz)/(30*(A*G*L**2*scf - 12*E*Izz)) + L*r21*(intrhoy2 + intrhoz2)/6) + r12*(L**2*intrhoy*r21*(-A*G*L**2*scf + 15*E*Iyy)/(30*(A*G*L**2*scf - 12*E*Iyy)) + L*intrhoyz*r23*(A**2*G**2*L**4*scf**2 - 30*A*E*G*Iyy*L**2*scf - 30*A*E*G*Izz*L**2*scf - 720*E**2*Iyy*Izz)/(30*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + L*r22*(-3*A**2*G**2*L**6*intrho*scf**2 - 14*A**2*G**2*L**4*intrhoz2*scf**2 + 84*A*E*G*Iyy*L**4*intrho*scf + 840*A*E*G*Iyy*L**2*intrhoz2*scf - 504*E**2*Iyy**2*L**2*intrho + 10080*E**2*Iyy**2*intrhoz2)/(420*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2))) + r13*(L**2*intrhoz*r21*(A*G*L**2*scf - 15*E*Izz)/(30*(A*G*L**2*scf - 12*E*Izz)) + L*intrhoyz*r22*(A**2*G**2*L**4*scf**2 - 30*A*E*G*Iyy*L**2*scf - 30*A*E*G*Izz*L**2*scf - 720*E**2*Iyy*Izz)/(30*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + L*r23*(-3*A**2*G**2*L**6*intrho*scf**2 - 14*A**2*G**2*L**4*intrhoy2*scf**2 + 84*A*E*G*Izz*L**4*intrho*scf + 840*A*E*G*Izz*L**2*intrhoy2*scf - 504*E**2*Izz**2*L**2*intrho + 10080*E**2*Izz**2*intrhoy2)/(420*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)))
                k += 1
                Mv[k] += r21*(L**2*intrhoy*r22*(A*G*L**2*scf - 15*E*Iyy)/(30*(A*G*L**2*scf - 12*E*Iyy)) + L**2*intrhoz*r23*(-A*G*L**2*scf + 15*E*Izz)/(30*(A*G*L**2*scf - 12*E*Izz)) + L*r21*(intrhoy2 + intrhoz2)/6) + r22*(L**2*intrhoy*r21*(-A*G*L**2*scf + 15*E*Iyy)/(30*(A*G*L**2*scf - 12*E*Iyy)) + L*intrhoyz*r23*(A**2*G**2*L**4*scf**2 - 30*A*E*G*Iyy*L**2*scf - 30*A*E*G*Izz*L**2*scf - 720*E**2*Iyy*Izz)/(30*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + L*r22*(-3*A**2*G**2*L**6*intrho*scf**2 - 14*A**2*G**2*L**4*intrhoz2*scf**2 + 84*A*E*G*Iyy*L**4*intrho*scf + 840*A*E*G*Iyy*L**2*intrhoz2*scf - 504*E**2*Iyy**2*L**2*intrho + 10080*E**2*Iyy**2*intrhoz2)/(420*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2))) + r23*(L**2*intrhoz*r21*(A*G*L**2*scf - 15*E*Izz)/(30*(A*G*L**2*scf - 12*E*Izz)) + L*intrhoyz*r22*(A**2*G**2*L**4*scf**2 - 30*A*E*G*Iyy*L**2*scf - 30*A*E*G*Izz*L**2*scf - 720*E**2*Iyy*Izz)/(30*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + L*r23*(-3*A**2*G**2*L**6*intrho*scf**2 - 14*A**2*G**2*L**4*intrhoy2*scf**2 + 84*A*E*G*Izz*L**4*intrho*scf + 840*A*E*G*Izz*L**2*intrhoy2*scf - 504*E**2*Izz**2*L**2*intrho + 10080*E**2*Izz**2*intrhoy2)/(420*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)))
                k += 1
                Mv[k] += r31*(L**2*intrhoy*r22*(A*G*L**2*scf - 15*E*Iyy)/(30*(A*G*L**2*scf - 12*E*Iyy)) + L**2*intrhoz*r23*(-A*G*L**2*scf + 15*E*Izz)/(30*(A*G*L**2*scf - 12*E*Izz)) + L*r21*(intrhoy2 + intrhoz2)/6) + r32*(L**2*intrhoy*r21*(-A*G*L**2*scf + 15*E*Iyy)/(30*(A*G*L**2*scf - 12*E*Iyy)) + L*intrhoyz*r23*(A**2*G**2*L**4*scf**2 - 30*A*E*G*Iyy*L**2*scf - 30*A*E*G*Izz*L**2*scf - 720*E**2*Iyy*Izz)/(30*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + L*r22*(-3*A**2*G**2*L**6*intrho*scf**2 - 14*A**2*G**2*L**4*intrhoz2*scf**2 + 84*A*E*G*Iyy*L**4*intrho*scf + 840*A*E*G*Iyy*L**2*intrhoz2*scf - 504*E**2*Iyy**2*L**2*intrho + 10080*E**2*Iyy**2*intrhoz2)/(420*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2))) + r33*(L**2*intrhoz*r21*(A*G*L**2*scf - 15*E*Izz)/(30*(A*G*L**2*scf - 12*E*Izz)) + L*intrhoyz*r22*(A**2*G**2*L**4*scf**2 - 30*A*E*G*Iyy*L**2*scf - 30*A*E*G*Izz*L**2*scf - 720*E**2*Iyy*Izz)/(30*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + L*r23*(-3*A**2*G**2*L**6*intrho*scf**2 - 14*A**2*G**2*L**4*intrhoy2*scf**2 + 84*A*E*G*Izz*L**4*intrho*scf + 840*A*E*G*Izz*L**2*intrhoy2*scf - 504*E**2*Izz**2*L**2*intrho + 10080*E**2*Izz**2*intrhoy2)/(420*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)))
                k += 1
                Mv[k] += r11*(L*intrhoy*r33*(-A*G*L**2*scf + 48*E*Izz)/(12*(A*G*L**2*scf - 12*E*Izz)) + L*intrhoz*r32*(A*G*L**2*scf - 48*E*Iyy)/(12*(A*G*L**2*scf - 12*E*Iyy))) + r12*(-A*G*L**2*intrhoyz*r32*scf*(A*G*L**2*scf + 60*E*Iyy)/(10*A**2*G**2*L**4*scf**2 - 120*A*E*G*Iyy*L**2*scf - 120*A*E*G*Izz*L**2*scf + 1440*E**2*Iyy*Izz) + L**2*r33*(11*A**2*G**2*L**4*intrho*scf**2 + 21*A**2*G**2*L**2*intrhoy2*scf**2 - 231*A*E*G*Izz*L**2*intrho*scf + 1260*A*E*G*Izz*intrhoy2*scf + 1260*E**2*Izz**2*intrho)/(210*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)) + L*intrhoz*r31*(-7*A*G*L**2*scf + 80*E*Izz)/(20*(A*G*L**2*scf - 12*E*Izz))) + r13*(-A*G*L**2*intrhoyz*r33*scf*(A*G*L**2*scf + 60*E*Izz)/(10*A**2*G**2*L**4*scf**2 - 120*A*E*G*Iyy*L**2*scf - 120*A*E*G*Izz*L**2*scf + 1440*E**2*Iyy*Izz) + L**2*r32*(11*A**2*G**2*L**4*intrho*scf**2 + 21*A**2*G**2*L**2*intrhoz2*scf**2 - 231*A*E*G*Iyy*L**2*intrho*scf + 1260*A*E*G*Iyy*intrhoz2*scf + 1260*E**2*Iyy**2*intrho)/(210*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)) + L*intrhoy*r31*(7*A*G*L**2*scf - 80*E*Iyy)/(20*(A*G*L**2*scf - 12*E*Iyy)))
                k += 1
                Mv[k] += r21*(L*intrhoy*r33*(-A*G*L**2*scf + 48*E*Izz)/(12*(A*G*L**2*scf - 12*E*Izz)) + L*intrhoz*r32*(A*G*L**2*scf - 48*E*Iyy)/(12*(A*G*L**2*scf - 12*E*Iyy))) + r22*(-A*G*L**2*intrhoyz*r32*scf*(A*G*L**2*scf + 60*E*Iyy)/(10*A**2*G**2*L**4*scf**2 - 120*A*E*G*Iyy*L**2*scf - 120*A*E*G*Izz*L**2*scf + 1440*E**2*Iyy*Izz) + L**2*r33*(11*A**2*G**2*L**4*intrho*scf**2 + 21*A**2*G**2*L**2*intrhoy2*scf**2 - 231*A*E*G*Izz*L**2*intrho*scf + 1260*A*E*G*Izz*intrhoy2*scf + 1260*E**2*Izz**2*intrho)/(210*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)) + L*intrhoz*r31*(-7*A*G*L**2*scf + 80*E*Izz)/(20*(A*G*L**2*scf - 12*E*Izz))) + r23*(-A*G*L**2*intrhoyz*r33*scf*(A*G*L**2*scf + 60*E*Izz)/(10*A**2*G**2*L**4*scf**2 - 120*A*E*G*Iyy*L**2*scf - 120*A*E*G*Izz*L**2*scf + 1440*E**2*Iyy*Izz) + L**2*r32*(11*A**2*G**2*L**4*intrho*scf**2 + 21*A**2*G**2*L**2*intrhoz2*scf**2 - 231*A*E*G*Iyy*L**2*intrho*scf + 1260*A*E*G*Iyy*intrhoz2*scf + 1260*E**2*Iyy**2*intrho)/(210*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)) + L*intrhoy*r31*(7*A*G*L**2*scf - 80*E*Iyy)/(20*(A*G*L**2*scf - 12*E*Iyy)))
                k += 1
                Mv[k] += r31*(L*intrhoy*r33*(-A*G*L**2*scf + 48*E*Izz)/(12*(A*G*L**2*scf - 12*E*Izz)) + L*intrhoz*r32*(A*G*L**2*scf - 48*E*Iyy)/(12*(A*G*L**2*scf - 12*E*Iyy))) + r32*(-A*G*L**2*intrhoyz*r32*scf*(A*G*L**2*scf + 60*E*Iyy)/(10*A**2*G**2*L**4*scf**2 - 120*A*E*G*Iyy*L**2*scf - 120*A*E*G*Izz*L**2*scf + 1440*E**2*Iyy*Izz) + L**2*r33*(11*A**2*G**2*L**4*intrho*scf**2 + 21*A**2*G**2*L**2*intrhoy2*scf**2 - 231*A*E*G*Izz*L**2*intrho*scf + 1260*A*E*G*Izz*intrhoy2*scf + 1260*E**2*Izz**2*intrho)/(210*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)) + L*intrhoz*r31*(-7*A*G*L**2*scf + 80*E*Izz)/(20*(A*G*L**2*scf - 12*E*Izz))) + r33*(-A*G*L**2*intrhoyz*r33*scf*(A*G*L**2*scf + 60*E*Izz)/(10*A**2*G**2*L**4*scf**2 - 120*A*E*G*Iyy*L**2*scf - 120*A*E*G*Izz*L**2*scf + 1440*E**2*Iyy*Izz) + L**2*r32*(11*A**2*G**2*L**4*intrho*scf**2 + 21*A**2*G**2*L**2*intrhoz2*scf**2 - 231*A*E*G*Iyy*L**2*intrho*scf + 1260*A*E*G*Iyy*intrhoz2*scf + 1260*E**2*Iyy**2*intrho)/(210*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)) + L*intrhoy*r31*(7*A*G*L**2*scf - 80*E*Iyy)/(20*(A*G*L**2*scf - 12*E*Iyy)))
                k += 1
                Mv[k] += r11*(L**2*intrhoy*r32*(A*G*L**2*scf - 10*E*Iyy)/(20*(A*G*L**2*scf - 12*E*Iyy)) + L**2*intrhoz*r33*(-A*G*L**2*scf + 10*E*Izz)/(20*(A*G*L**2*scf - 12*E*Izz)) + L*r31*(intrhoy2 + intrhoz2)/3) + r12*(L**2*intrhoy*r31*(A*G*L**2*scf - 10*E*Iyy)/(20*(A*G*L**2*scf - 12*E*Iyy)) + L*intrhoyz*r33*(-2*A**2*G**2*L**4*scf**2 + 15*A*E*G*Iyy*L**2*scf + 15*A*E*G*Izz*L**2*scf - 720*E**2*Iyy*Izz)/(15*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + L*r32*(A**2*G**2*L**6*intrho*scf**2 + 14*A**2*G**2*L**4*intrhoz2*scf**2 - 21*A*E*G*Iyy*L**4*intrho*scf - 210*A*E*G*Iyy*L**2*intrhoz2*scf + 126*E**2*Iyy**2*L**2*intrho + 5040*E**2*Iyy**2*intrhoz2)/(105*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2))) + r13*(L**2*intrhoz*r31*(-A*G*L**2*scf + 10*E*Izz)/(20*(A*G*L**2*scf - 12*E*Izz)) + L*intrhoyz*r32*(-2*A**2*G**2*L**4*scf**2 + 15*A*E*G*Iyy*L**2*scf + 15*A*E*G*Izz*L**2*scf - 720*E**2*Iyy*Izz)/(15*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + L*r33*(A**2*G**2*L**6*intrho*scf**2 + 14*A**2*G**2*L**4*intrhoy2*scf**2 - 21*A*E*G*Izz*L**4*intrho*scf - 210*A*E*G*Izz*L**2*intrhoy2*scf + 126*E**2*Izz**2*L**2*intrho + 5040*E**2*Izz**2*intrhoy2)/(105*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)))
                k += 1
                Mv[k] += r21*(L**2*intrhoy*r32*(A*G*L**2*scf - 10*E*Iyy)/(20*(A*G*L**2*scf - 12*E*Iyy)) + L**2*intrhoz*r33*(-A*G*L**2*scf + 10*E*Izz)/(20*(A*G*L**2*scf - 12*E*Izz)) + L*r31*(intrhoy2 + intrhoz2)/3) + r22*(L**2*intrhoy*r31*(A*G*L**2*scf - 10*E*Iyy)/(20*(A*G*L**2*scf - 12*E*Iyy)) + L*intrhoyz*r33*(-2*A**2*G**2*L**4*scf**2 + 15*A*E*G*Iyy*L**2*scf + 15*A*E*G*Izz*L**2*scf - 720*E**2*Iyy*Izz)/(15*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + L*r32*(A**2*G**2*L**6*intrho*scf**2 + 14*A**2*G**2*L**4*intrhoz2*scf**2 - 21*A*E*G*Iyy*L**4*intrho*scf - 210*A*E*G*Iyy*L**2*intrhoz2*scf + 126*E**2*Iyy**2*L**2*intrho + 5040*E**2*Iyy**2*intrhoz2)/(105*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2))) + r23*(L**2*intrhoz*r31*(-A*G*L**2*scf + 10*E*Izz)/(20*(A*G*L**2*scf - 12*E*Izz)) + L*intrhoyz*r32*(-2*A**2*G**2*L**4*scf**2 + 15*A*E*G*Iyy*L**2*scf + 15*A*E*G*Izz*L**2*scf - 720*E**2*Iyy*Izz)/(15*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + L*r33*(A**2*G**2*L**6*intrho*scf**2 + 14*A**2*G**2*L**4*intrhoy2*scf**2 - 21*A*E*G*Izz*L**4*intrho*scf - 210*A*E*G*Izz*L**2*intrhoy2*scf + 126*E**2*Izz**2*L**2*intrho + 5040*E**2*Izz**2*intrhoy2)/(105*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)))
                k += 1
                Mv[k] += r31*(L**2*intrhoy*r32*(A*G*L**2*scf - 10*E*Iyy)/(20*(A*G*L**2*scf - 12*E*Iyy)) + L**2*intrhoz*r33*(-A*G*L**2*scf + 10*E*Izz)/(20*(A*G*L**2*scf - 12*E*Izz)) + L*r31*(intrhoy2 + intrhoz2)/3) + r32*(L**2*intrhoy*r31*(A*G*L**2*scf - 10*E*Iyy)/(20*(A*G*L**2*scf - 12*E*Iyy)) + L*intrhoyz*r33*(-2*A**2*G**2*L**4*scf**2 + 15*A*E*G*Iyy*L**2*scf + 15*A*E*G*Izz*L**2*scf - 720*E**2*Iyy*Izz)/(15*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + L*r32*(A**2*G**2*L**6*intrho*scf**2 + 14*A**2*G**2*L**4*intrhoz2*scf**2 - 21*A*E*G*Iyy*L**4*intrho*scf - 210*A*E*G*Iyy*L**2*intrhoz2*scf + 126*E**2*Iyy**2*L**2*intrho + 5040*E**2*Iyy**2*intrhoz2)/(105*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2))) + r33*(L**2*intrhoz*r31*(-A*G*L**2*scf + 10*E*Izz)/(20*(A*G*L**2*scf - 12*E*Izz)) + L*intrhoyz*r32*(-2*A**2*G**2*L**4*scf**2 + 15*A*E*G*Iyy*L**2*scf + 15*A*E*G*Izz*L**2*scf - 720*E**2*Iyy*Izz)/(15*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + L*r33*(A**2*G**2*L**6*intrho*scf**2 + 14*A**2*G**2*L**4*intrhoy2*scf**2 - 21*A*E*G*Izz*L**4*intrho*scf - 210*A*E*G*Izz*L**2*intrhoy2*scf + 126*E**2*Izz**2*L**2*intrho + 5040*E**2*Izz**2*intrhoy2)/(105*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)))
                k += 1
                Mv[k] += r11*(L*intrhoy*r33*(A*G*L**2*scf + 24*E*Izz)/(12*(A*G*L**2*scf - 12*E*Izz)) - L*intrhoz*r32*(A*G*L**2*scf + 24*E*Iyy)/(12*A*G*L**2*scf - 144*E*Iyy)) + r12*(A*G*L**2*intrhoyz*r32*scf*(A*G*L**2*scf + 60*E*Iyy)/(10*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + L**2*r33*(13*A**2*G**2*L**4*intrho*scf**2 - 42*A**2*G**2*L**2*intrhoy2*scf**2 - 378*A*E*G*Izz*L**2*intrho*scf - 2520*A*E*G*Izz*intrhoy2*scf + 2520*E**2*Izz**2*intrho)/(420*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)) + L*intrhoz*r31*(-3*A*G*L**2*scf + 40*E*Izz)/(20*(A*G*L**2*scf - 12*E*Izz))) + r13*(A*G*L**2*intrhoyz*r33*scf*(A*G*L**2*scf + 60*E*Izz)/(10*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + L**2*r32*(13*A**2*G**2*L**4*intrho*scf**2 - 42*A**2*G**2*L**2*intrhoz2*scf**2 - 378*A*E*G*Iyy*L**2*intrho*scf - 2520*A*E*G*Iyy*intrhoz2*scf + 2520*E**2*Iyy**2*intrho)/(420*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)) + L*intrhoy*r31*(3*A*G*L**2*scf - 40*E*Iyy)/(20*(A*G*L**2*scf - 12*E*Iyy)))
                k += 1
                Mv[k] += r21*(L*intrhoy*r33*(A*G*L**2*scf + 24*E*Izz)/(12*(A*G*L**2*scf - 12*E*Izz)) - L*intrhoz*r32*(A*G*L**2*scf + 24*E*Iyy)/(12*A*G*L**2*scf - 144*E*Iyy)) + r22*(A*G*L**2*intrhoyz*r32*scf*(A*G*L**2*scf + 60*E*Iyy)/(10*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + L**2*r33*(13*A**2*G**2*L**4*intrho*scf**2 - 42*A**2*G**2*L**2*intrhoy2*scf**2 - 378*A*E*G*Izz*L**2*intrho*scf - 2520*A*E*G*Izz*intrhoy2*scf + 2520*E**2*Izz**2*intrho)/(420*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)) + L*intrhoz*r31*(-3*A*G*L**2*scf + 40*E*Izz)/(20*(A*G*L**2*scf - 12*E*Izz))) + r23*(A*G*L**2*intrhoyz*r33*scf*(A*G*L**2*scf + 60*E*Izz)/(10*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + L**2*r32*(13*A**2*G**2*L**4*intrho*scf**2 - 42*A**2*G**2*L**2*intrhoz2*scf**2 - 378*A*E*G*Iyy*L**2*intrho*scf - 2520*A*E*G*Iyy*intrhoz2*scf + 2520*E**2*Iyy**2*intrho)/(420*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)) + L*intrhoy*r31*(3*A*G*L**2*scf - 40*E*Iyy)/(20*(A*G*L**2*scf - 12*E*Iyy)))
                k += 1
                Mv[k] += r31*(L*intrhoy*r33*(A*G*L**2*scf + 24*E*Izz)/(12*(A*G*L**2*scf - 12*E*Izz)) - L*intrhoz*r32*(A*G*L**2*scf + 24*E*Iyy)/(12*A*G*L**2*scf - 144*E*Iyy)) + r32*(A*G*L**2*intrhoyz*r32*scf*(A*G*L**2*scf + 60*E*Iyy)/(10*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + L**2*r33*(13*A**2*G**2*L**4*intrho*scf**2 - 42*A**2*G**2*L**2*intrhoy2*scf**2 - 378*A*E*G*Izz*L**2*intrho*scf - 2520*A*E*G*Izz*intrhoy2*scf + 2520*E**2*Izz**2*intrho)/(420*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)) + L*intrhoz*r31*(-3*A*G*L**2*scf + 40*E*Izz)/(20*(A*G*L**2*scf - 12*E*Izz))) + r33*(A*G*L**2*intrhoyz*r33*scf*(A*G*L**2*scf + 60*E*Izz)/(10*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + L**2*r32*(13*A**2*G**2*L**4*intrho*scf**2 - 42*A**2*G**2*L**2*intrhoz2*scf**2 - 378*A*E*G*Iyy*L**2*intrho*scf - 2520*A*E*G*Iyy*intrhoz2*scf + 2520*E**2*Iyy**2*intrho)/(420*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)) + L*intrhoy*r31*(3*A*G*L**2*scf - 40*E*Iyy)/(20*(A*G*L**2*scf - 12*E*Iyy)))
                k += 1
                Mv[k] += r11*(L**2*intrhoy*r32*(A*G*L**2*scf - 15*E*Iyy)/(30*(A*G*L**2*scf - 12*E*Iyy)) + L**2*intrhoz*r33*(-A*G*L**2*scf + 15*E*Izz)/(30*(A*G*L**2*scf - 12*E*Izz)) + L*r31*(intrhoy2 + intrhoz2)/6) + r12*(L**2*intrhoy*r31*(-A*G*L**2*scf + 15*E*Iyy)/(30*(A*G*L**2*scf - 12*E*Iyy)) + L*intrhoyz*r33*(A**2*G**2*L**4*scf**2 - 30*A*E*G*Iyy*L**2*scf - 30*A*E*G*Izz*L**2*scf - 720*E**2*Iyy*Izz)/(30*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + L*r32*(-3*A**2*G**2*L**6*intrho*scf**2 - 14*A**2*G**2*L**4*intrhoz2*scf**2 + 84*A*E*G*Iyy*L**4*intrho*scf + 840*A*E*G*Iyy*L**2*intrhoz2*scf - 504*E**2*Iyy**2*L**2*intrho + 10080*E**2*Iyy**2*intrhoz2)/(420*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2))) + r13*(L**2*intrhoz*r31*(A*G*L**2*scf - 15*E*Izz)/(30*(A*G*L**2*scf - 12*E*Izz)) + L*intrhoyz*r32*(A**2*G**2*L**4*scf**2 - 30*A*E*G*Iyy*L**2*scf - 30*A*E*G*Izz*L**2*scf - 720*E**2*Iyy*Izz)/(30*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + L*r33*(-3*A**2*G**2*L**6*intrho*scf**2 - 14*A**2*G**2*L**4*intrhoy2*scf**2 + 84*A*E*G*Izz*L**4*intrho*scf + 840*A*E*G*Izz*L**2*intrhoy2*scf - 504*E**2*Izz**2*L**2*intrho + 10080*E**2*Izz**2*intrhoy2)/(420*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)))
                k += 1
                Mv[k] += r21*(L**2*intrhoy*r32*(A*G*L**2*scf - 15*E*Iyy)/(30*(A*G*L**2*scf - 12*E*Iyy)) + L**2*intrhoz*r33*(-A*G*L**2*scf + 15*E*Izz)/(30*(A*G*L**2*scf - 12*E*Izz)) + L*r31*(intrhoy2 + intrhoz2)/6) + r22*(L**2*intrhoy*r31*(-A*G*L**2*scf + 15*E*Iyy)/(30*(A*G*L**2*scf - 12*E*Iyy)) + L*intrhoyz*r33*(A**2*G**2*L**4*scf**2 - 30*A*E*G*Iyy*L**2*scf - 30*A*E*G*Izz*L**2*scf - 720*E**2*Iyy*Izz)/(30*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + L*r32*(-3*A**2*G**2*L**6*intrho*scf**2 - 14*A**2*G**2*L**4*intrhoz2*scf**2 + 84*A*E*G*Iyy*L**4*intrho*scf + 840*A*E*G*Iyy*L**2*intrhoz2*scf - 504*E**2*Iyy**2*L**2*intrho + 10080*E**2*Iyy**2*intrhoz2)/(420*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2))) + r23*(L**2*intrhoz*r31*(A*G*L**2*scf - 15*E*Izz)/(30*(A*G*L**2*scf - 12*E*Izz)) + L*intrhoyz*r32*(A**2*G**2*L**4*scf**2 - 30*A*E*G*Iyy*L**2*scf - 30*A*E*G*Izz*L**2*scf - 720*E**2*Iyy*Izz)/(30*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + L*r33*(-3*A**2*G**2*L**6*intrho*scf**2 - 14*A**2*G**2*L**4*intrhoy2*scf**2 + 84*A*E*G*Izz*L**4*intrho*scf + 840*A*E*G*Izz*L**2*intrhoy2*scf - 504*E**2*Izz**2*L**2*intrho + 10080*E**2*Izz**2*intrhoy2)/(420*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)))
                k += 1
                Mv[k] += r31*(L**2*intrhoy*r32*(A*G*L**2*scf - 15*E*Iyy)/(30*(A*G*L**2*scf - 12*E*Iyy)) + L**2*intrhoz*r33*(-A*G*L**2*scf + 15*E*Izz)/(30*(A*G*L**2*scf - 12*E*Izz)) + L*r31*(intrhoy2 + intrhoz2)/6) + r32*(L**2*intrhoy*r31*(-A*G*L**2*scf + 15*E*Iyy)/(30*(A*G*L**2*scf - 12*E*Iyy)) + L*intrhoyz*r33*(A**2*G**2*L**4*scf**2 - 30*A*E*G*Iyy*L**2*scf - 30*A*E*G*Izz*L**2*scf - 720*E**2*Iyy*Izz)/(30*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + L*r32*(-3*A**2*G**2*L**6*intrho*scf**2 - 14*A**2*G**2*L**4*intrhoz2*scf**2 + 84*A*E*G*Iyy*L**4*intrho*scf + 840*A*E*G*Iyy*L**2*intrhoz2*scf - 504*E**2*Iyy**2*L**2*intrho + 10080*E**2*Iyy**2*intrhoz2)/(420*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2))) + r33*(L**2*intrhoz*r31*(A*G*L**2*scf - 15*E*Izz)/(30*(A*G*L**2*scf - 12*E*Izz)) + L*intrhoyz*r32*(A**2*G**2*L**4*scf**2 - 30*A*E*G*Iyy*L**2*scf - 30*A*E*G*Izz*L**2*scf - 720*E**2*Iyy*Izz)/(30*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + L*r33*(-3*A**2*G**2*L**6*intrho*scf**2 - 14*A**2*G**2*L**4*intrhoy2*scf**2 + 84*A*E*G*Izz*L**4*intrho*scf + 840*A*E*G*Izz*L**2*intrhoy2*scf - 504*E**2*Izz**2*L**2*intrho + 10080*E**2*Izz**2*intrhoy2)/(420*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)))
                k += 1
                Mv[k] += r11*(-A*G*L**2*intrhoy*r12*scf/(2*A*G*L**2*scf - 24*E*Izz) + A*G*L**2*intrhoz*r13*scf/(2*(A*G*L**2*scf - 12*E*Iyy)) + L*intrho*r11/6) + r12*(6*A**2*G**2*L**3*intrhoyz*r13*scf**2/(5*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + A*G*L**2*intrhoy*r11*scf/(2*(A*G*L**2*scf - 12*E*Izz)) + r12*(9*A**2*G**2*L**5*intrho*scf**2 - 84*A**2*G**2*L**3*intrhoy2*scf**2 - 252*A*E*G*Izz*L**3*intrho*scf + 1680*E**2*Izz**2*L*intrho)/(70*A**2*G**2*L**4*scf**2 - 1680*A*E*G*Izz*L**2*scf + 10080*E**2*Izz**2)) + r13*(6*A**2*G**2*L**3*intrhoyz*r12*scf**2/(5*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) - A*G*L**2*intrhoz*r11*scf/(2*A*G*L**2*scf - 24*E*Iyy) + r13*(9*A**2*G**2*L**5*intrho*scf**2 - 84*A**2*G**2*L**3*intrhoz2*scf**2 - 252*A*E*G*Iyy*L**3*intrho*scf + 1680*E**2*Iyy**2*L*intrho)/(70*A**2*G**2*L**4*scf**2 - 1680*A*E*G*Iyy*L**2*scf + 10080*E**2*Iyy**2))
                k += 1
                Mv[k] += r21*(-A*G*L**2*intrhoy*r12*scf/(2*A*G*L**2*scf - 24*E*Izz) + A*G*L**2*intrhoz*r13*scf/(2*(A*G*L**2*scf - 12*E*Iyy)) + L*intrho*r11/6) + r22*(6*A**2*G**2*L**3*intrhoyz*r13*scf**2/(5*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + A*G*L**2*intrhoy*r11*scf/(2*(A*G*L**2*scf - 12*E*Izz)) + r12*(9*A**2*G**2*L**5*intrho*scf**2 - 84*A**2*G**2*L**3*intrhoy2*scf**2 - 252*A*E*G*Izz*L**3*intrho*scf + 1680*E**2*Izz**2*L*intrho)/(70*A**2*G**2*L**4*scf**2 - 1680*A*E*G*Izz*L**2*scf + 10080*E**2*Izz**2)) + r23*(6*A**2*G**2*L**3*intrhoyz*r12*scf**2/(5*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) - A*G*L**2*intrhoz*r11*scf/(2*A*G*L**2*scf - 24*E*Iyy) + r13*(9*A**2*G**2*L**5*intrho*scf**2 - 84*A**2*G**2*L**3*intrhoz2*scf**2 - 252*A*E*G*Iyy*L**3*intrho*scf + 1680*E**2*Iyy**2*L*intrho)/(70*A**2*G**2*L**4*scf**2 - 1680*A*E*G*Iyy*L**2*scf + 10080*E**2*Iyy**2))
                k += 1
                Mv[k] += r31*(-A*G*L**2*intrhoy*r12*scf/(2*A*G*L**2*scf - 24*E*Izz) + A*G*L**2*intrhoz*r13*scf/(2*(A*G*L**2*scf - 12*E*Iyy)) + L*intrho*r11/6) + r32*(6*A**2*G**2*L**3*intrhoyz*r13*scf**2/(5*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + A*G*L**2*intrhoy*r11*scf/(2*(A*G*L**2*scf - 12*E*Izz)) + r12*(9*A**2*G**2*L**5*intrho*scf**2 - 84*A**2*G**2*L**3*intrhoy2*scf**2 - 252*A*E*G*Izz*L**3*intrho*scf + 1680*E**2*Izz**2*L*intrho)/(70*A**2*G**2*L**4*scf**2 - 1680*A*E*G*Izz*L**2*scf + 10080*E**2*Izz**2)) + r33*(6*A**2*G**2*L**3*intrhoyz*r12*scf**2/(5*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) - A*G*L**2*intrhoz*r11*scf/(2*A*G*L**2*scf - 24*E*Iyy) + r13*(9*A**2*G**2*L**5*intrho*scf**2 - 84*A**2*G**2*L**3*intrhoz2*scf**2 - 252*A*E*G*Iyy*L**3*intrho*scf + 1680*E**2*Iyy**2*L*intrho)/(70*A**2*G**2*L**4*scf**2 - 1680*A*E*G*Iyy*L**2*scf + 10080*E**2*Iyy**2))
                k += 1
                Mv[k] += r11*(L*intrhoy*r13*(3*A*G*L**2*scf - 40*E*Iyy)/(20*(A*G*L**2*scf - 12*E*Iyy)) + L*intrhoz*r12*(-3*A*G*L**2*scf + 40*E*Izz)/(20*(A*G*L**2*scf - 12*E*Izz))) + r12*(A*G*L**2*intrhoyz*r12*scf*(A*G*L**2*scf + 60*E*Iyy)/(10*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + L**2*r13*(13*A**2*G**2*L**4*intrho*scf**2 - 42*A**2*G**2*L**2*intrhoz2*scf**2 - 378*A*E*G*Iyy*L**2*intrho*scf - 2520*A*E*G*Iyy*intrhoz2*scf + 2520*E**2*Iyy**2*intrho)/(420*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)) - L*intrhoz*r11*(A*G*L**2*scf + 24*E*Iyy)/(12*A*G*L**2*scf - 144*E*Iyy)) + r13*(A*G*L**2*intrhoyz*r13*scf*(A*G*L**2*scf + 60*E*Izz)/(10*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + L**2*r12*(13*A**2*G**2*L**4*intrho*scf**2 - 42*A**2*G**2*L**2*intrhoy2*scf**2 - 378*A*E*G*Izz*L**2*intrho*scf - 2520*A*E*G*Izz*intrhoy2*scf + 2520*E**2*Izz**2*intrho)/(420*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)) + L*intrhoy*r11*(A*G*L**2*scf + 24*E*Izz)/(12*(A*G*L**2*scf - 12*E*Izz)))
                k += 1
                Mv[k] += r21*(L*intrhoy*r13*(3*A*G*L**2*scf - 40*E*Iyy)/(20*(A*G*L**2*scf - 12*E*Iyy)) + L*intrhoz*r12*(-3*A*G*L**2*scf + 40*E*Izz)/(20*(A*G*L**2*scf - 12*E*Izz))) + r22*(A*G*L**2*intrhoyz*r12*scf*(A*G*L**2*scf + 60*E*Iyy)/(10*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + L**2*r13*(13*A**2*G**2*L**4*intrho*scf**2 - 42*A**2*G**2*L**2*intrhoz2*scf**2 - 378*A*E*G*Iyy*L**2*intrho*scf - 2520*A*E*G*Iyy*intrhoz2*scf + 2520*E**2*Iyy**2*intrho)/(420*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)) - L*intrhoz*r11*(A*G*L**2*scf + 24*E*Iyy)/(12*A*G*L**2*scf - 144*E*Iyy)) + r23*(A*G*L**2*intrhoyz*r13*scf*(A*G*L**2*scf + 60*E*Izz)/(10*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + L**2*r12*(13*A**2*G**2*L**4*intrho*scf**2 - 42*A**2*G**2*L**2*intrhoy2*scf**2 - 378*A*E*G*Izz*L**2*intrho*scf - 2520*A*E*G*Izz*intrhoy2*scf + 2520*E**2*Izz**2*intrho)/(420*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)) + L*intrhoy*r11*(A*G*L**2*scf + 24*E*Izz)/(12*(A*G*L**2*scf - 12*E*Izz)))
                k += 1
                Mv[k] += r31*(L*intrhoy*r13*(3*A*G*L**2*scf - 40*E*Iyy)/(20*(A*G*L**2*scf - 12*E*Iyy)) + L*intrhoz*r12*(-3*A*G*L**2*scf + 40*E*Izz)/(20*(A*G*L**2*scf - 12*E*Izz))) + r32*(A*G*L**2*intrhoyz*r12*scf*(A*G*L**2*scf + 60*E*Iyy)/(10*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + L**2*r13*(13*A**2*G**2*L**4*intrho*scf**2 - 42*A**2*G**2*L**2*intrhoz2*scf**2 - 378*A*E*G*Iyy*L**2*intrho*scf - 2520*A*E*G*Iyy*intrhoz2*scf + 2520*E**2*Iyy**2*intrho)/(420*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)) - L*intrhoz*r11*(A*G*L**2*scf + 24*E*Iyy)/(12*A*G*L**2*scf - 144*E*Iyy)) + r33*(A*G*L**2*intrhoyz*r13*scf*(A*G*L**2*scf + 60*E*Izz)/(10*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + L**2*r12*(13*A**2*G**2*L**4*intrho*scf**2 - 42*A**2*G**2*L**2*intrhoy2*scf**2 - 378*A*E*G*Izz*L**2*intrho*scf - 2520*A*E*G*Izz*intrhoy2*scf + 2520*E**2*Izz**2*intrho)/(420*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)) + L*intrhoy*r11*(A*G*L**2*scf + 24*E*Izz)/(12*(A*G*L**2*scf - 12*E*Izz)))
                k += 1
                Mv[k] += r11*(-A*G*L**2*intrhoy*r12*scf/(2*A*G*L**2*scf - 24*E*Izz) + A*G*L**2*intrhoz*r13*scf/(2*(A*G*L**2*scf - 12*E*Iyy)) + L*intrho*r11/3) + r12*(-6*A**2*G**2*L**3*intrhoyz*r13*scf**2/(5*A**2*G**2*L**4*scf**2 - 60*A*E*G*Iyy*L**2*scf - 60*A*E*G*Izz*L**2*scf + 720*E**2*Iyy*Izz) - A*G*L**2*intrhoy*r11*scf/(2*A*G*L**2*scf - 24*E*Izz) + L*r12*(13*A**2*G**2*L**4*intrho*scf**2 + 42*A**2*G**2*L**2*intrhoy2*scf**2 - 294*A*E*G*Izz*L**2*intrho*scf + 1680*E**2*Izz**2*intrho)/(35*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2))) + r13*(-6*A**2*G**2*L**3*intrhoyz*r12*scf**2/(5*A**2*G**2*L**4*scf**2 - 60*A*E*G*Iyy*L**2*scf - 60*A*E*G*Izz*L**2*scf + 720*E**2*Iyy*Izz) + A*G*L**2*intrhoz*r11*scf/(2*(A*G*L**2*scf - 12*E*Iyy)) + L*r13*(13*A**2*G**2*L**4*intrho*scf**2 + 42*A**2*G**2*L**2*intrhoz2*scf**2 - 294*A*E*G*Iyy*L**2*intrho*scf + 1680*E**2*Iyy**2*intrho)/(35*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)))
                k += 1
                Mv[k] += r21*(-A*G*L**2*intrhoy*r12*scf/(2*A*G*L**2*scf - 24*E*Izz) + A*G*L**2*intrhoz*r13*scf/(2*(A*G*L**2*scf - 12*E*Iyy)) + L*intrho*r11/3) + r22*(-6*A**2*G**2*L**3*intrhoyz*r13*scf**2/(5*A**2*G**2*L**4*scf**2 - 60*A*E*G*Iyy*L**2*scf - 60*A*E*G*Izz*L**2*scf + 720*E**2*Iyy*Izz) - A*G*L**2*intrhoy*r11*scf/(2*A*G*L**2*scf - 24*E*Izz) + L*r12*(13*A**2*G**2*L**4*intrho*scf**2 + 42*A**2*G**2*L**2*intrhoy2*scf**2 - 294*A*E*G*Izz*L**2*intrho*scf + 1680*E**2*Izz**2*intrho)/(35*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2))) + r23*(-6*A**2*G**2*L**3*intrhoyz*r12*scf**2/(5*A**2*G**2*L**4*scf**2 - 60*A*E*G*Iyy*L**2*scf - 60*A*E*G*Izz*L**2*scf + 720*E**2*Iyy*Izz) + A*G*L**2*intrhoz*r11*scf/(2*(A*G*L**2*scf - 12*E*Iyy)) + L*r13*(13*A**2*G**2*L**4*intrho*scf**2 + 42*A**2*G**2*L**2*intrhoz2*scf**2 - 294*A*E*G*Iyy*L**2*intrho*scf + 1680*E**2*Iyy**2*intrho)/(35*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)))
                k += 1
                Mv[k] += r31*(-A*G*L**2*intrhoy*r12*scf/(2*A*G*L**2*scf - 24*E*Izz) + A*G*L**2*intrhoz*r13*scf/(2*(A*G*L**2*scf - 12*E*Iyy)) + L*intrho*r11/3) + r32*(-6*A**2*G**2*L**3*intrhoyz*r13*scf**2/(5*A**2*G**2*L**4*scf**2 - 60*A*E*G*Iyy*L**2*scf - 60*A*E*G*Izz*L**2*scf + 720*E**2*Iyy*Izz) - A*G*L**2*intrhoy*r11*scf/(2*A*G*L**2*scf - 24*E*Izz) + L*r12*(13*A**2*G**2*L**4*intrho*scf**2 + 42*A**2*G**2*L**2*intrhoy2*scf**2 - 294*A*E*G*Izz*L**2*intrho*scf + 1680*E**2*Izz**2*intrho)/(35*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2))) + r33*(-6*A**2*G**2*L**3*intrhoyz*r12*scf**2/(5*A**2*G**2*L**4*scf**2 - 60*A*E*G*Iyy*L**2*scf - 60*A*E*G*Izz*L**2*scf + 720*E**2*Iyy*Izz) + A*G*L**2*intrhoz*r11*scf/(2*(A*G*L**2*scf - 12*E*Iyy)) + L*r13*(13*A**2*G**2*L**4*intrho*scf**2 + 42*A**2*G**2*L**2*intrhoz2*scf**2 - 294*A*E*G*Iyy*L**2*intrho*scf + 1680*E**2*Iyy**2*intrho)/(35*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)))
                k += 1
                Mv[k] += r11*(L*intrhoy*r13*(7*A*G*L**2*scf - 80*E*Iyy)/(20*(A*G*L**2*scf - 12*E*Iyy)) + L*intrhoz*r12*(-7*A*G*L**2*scf + 80*E*Izz)/(20*(A*G*L**2*scf - 12*E*Izz))) + r12*(A*G*L**2*intrhoyz*r12*scf*(A*G*L**2*scf + 60*E*Iyy)/(10*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + L**2*r13*(-11*A**2*G**2*L**4*intrho*scf**2 - 21*A**2*G**2*L**2*intrhoz2*scf**2 + 231*A*E*G*Iyy*L**2*intrho*scf - 1260*A*E*G*Iyy*intrhoz2*scf - 1260*E**2*Iyy**2*intrho)/(210*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)) + L*intrhoz*r11*(A*G*L**2*scf - 48*E*Iyy)/(12*(A*G*L**2*scf - 12*E*Iyy))) + r13*(A*G*L**2*intrhoyz*r13*scf*(A*G*L**2*scf + 60*E*Izz)/(10*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + L**2*r12*(-11*A**2*G**2*L**4*intrho*scf**2 - 21*A**2*G**2*L**2*intrhoy2*scf**2 + 231*A*E*G*Izz*L**2*intrho*scf - 1260*A*E*G*Izz*intrhoy2*scf - 1260*E**2*Izz**2*intrho)/(210*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)) + L*intrhoy*r11*(-A*G*L**2*scf + 48*E*Izz)/(12*(A*G*L**2*scf - 12*E*Izz)))
                k += 1
                Mv[k] += r21*(L*intrhoy*r13*(7*A*G*L**2*scf - 80*E*Iyy)/(20*(A*G*L**2*scf - 12*E*Iyy)) + L*intrhoz*r12*(-7*A*G*L**2*scf + 80*E*Izz)/(20*(A*G*L**2*scf - 12*E*Izz))) + r22*(A*G*L**2*intrhoyz*r12*scf*(A*G*L**2*scf + 60*E*Iyy)/(10*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + L**2*r13*(-11*A**2*G**2*L**4*intrho*scf**2 - 21*A**2*G**2*L**2*intrhoz2*scf**2 + 231*A*E*G*Iyy*L**2*intrho*scf - 1260*A*E*G*Iyy*intrhoz2*scf - 1260*E**2*Iyy**2*intrho)/(210*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)) + L*intrhoz*r11*(A*G*L**2*scf - 48*E*Iyy)/(12*(A*G*L**2*scf - 12*E*Iyy))) + r23*(A*G*L**2*intrhoyz*r13*scf*(A*G*L**2*scf + 60*E*Izz)/(10*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + L**2*r12*(-11*A**2*G**2*L**4*intrho*scf**2 - 21*A**2*G**2*L**2*intrhoy2*scf**2 + 231*A*E*G*Izz*L**2*intrho*scf - 1260*A*E*G*Izz*intrhoy2*scf - 1260*E**2*Izz**2*intrho)/(210*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)) + L*intrhoy*r11*(-A*G*L**2*scf + 48*E*Izz)/(12*(A*G*L**2*scf - 12*E*Izz)))
                k += 1
                Mv[k] += r31*(L*intrhoy*r13*(7*A*G*L**2*scf - 80*E*Iyy)/(20*(A*G*L**2*scf - 12*E*Iyy)) + L*intrhoz*r12*(-7*A*G*L**2*scf + 80*E*Izz)/(20*(A*G*L**2*scf - 12*E*Izz))) + r32*(A*G*L**2*intrhoyz*r12*scf*(A*G*L**2*scf + 60*E*Iyy)/(10*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + L**2*r13*(-11*A**2*G**2*L**4*intrho*scf**2 - 21*A**2*G**2*L**2*intrhoz2*scf**2 + 231*A*E*G*Iyy*L**2*intrho*scf - 1260*A*E*G*Iyy*intrhoz2*scf - 1260*E**2*Iyy**2*intrho)/(210*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)) + L*intrhoz*r11*(A*G*L**2*scf - 48*E*Iyy)/(12*(A*G*L**2*scf - 12*E*Iyy))) + r33*(A*G*L**2*intrhoyz*r13*scf*(A*G*L**2*scf + 60*E*Izz)/(10*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + L**2*r12*(-11*A**2*G**2*L**4*intrho*scf**2 - 21*A**2*G**2*L**2*intrhoy2*scf**2 + 231*A*E*G*Izz*L**2*intrho*scf - 1260*A*E*G*Izz*intrhoy2*scf - 1260*E**2*Izz**2*intrho)/(210*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)) + L*intrhoy*r11*(-A*G*L**2*scf + 48*E*Izz)/(12*(A*G*L**2*scf - 12*E*Izz)))
                k += 1
                Mv[k] += r11*(-A*G*L**2*intrhoy*r22*scf/(2*A*G*L**2*scf - 24*E*Izz) + A*G*L**2*intrhoz*r23*scf/(2*(A*G*L**2*scf - 12*E*Iyy)) + L*intrho*r21/6) + r12*(6*A**2*G**2*L**3*intrhoyz*r23*scf**2/(5*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + A*G*L**2*intrhoy*r21*scf/(2*(A*G*L**2*scf - 12*E*Izz)) + r22*(9*A**2*G**2*L**5*intrho*scf**2 - 84*A**2*G**2*L**3*intrhoy2*scf**2 - 252*A*E*G*Izz*L**3*intrho*scf + 1680*E**2*Izz**2*L*intrho)/(70*A**2*G**2*L**4*scf**2 - 1680*A*E*G*Izz*L**2*scf + 10080*E**2*Izz**2)) + r13*(6*A**2*G**2*L**3*intrhoyz*r22*scf**2/(5*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) - A*G*L**2*intrhoz*r21*scf/(2*A*G*L**2*scf - 24*E*Iyy) + r23*(9*A**2*G**2*L**5*intrho*scf**2 - 84*A**2*G**2*L**3*intrhoz2*scf**2 - 252*A*E*G*Iyy*L**3*intrho*scf + 1680*E**2*Iyy**2*L*intrho)/(70*A**2*G**2*L**4*scf**2 - 1680*A*E*G*Iyy*L**2*scf + 10080*E**2*Iyy**2))
                k += 1
                Mv[k] += r21*(-A*G*L**2*intrhoy*r22*scf/(2*A*G*L**2*scf - 24*E*Izz) + A*G*L**2*intrhoz*r23*scf/(2*(A*G*L**2*scf - 12*E*Iyy)) + L*intrho*r21/6) + r22*(6*A**2*G**2*L**3*intrhoyz*r23*scf**2/(5*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + A*G*L**2*intrhoy*r21*scf/(2*(A*G*L**2*scf - 12*E*Izz)) + r22*(9*A**2*G**2*L**5*intrho*scf**2 - 84*A**2*G**2*L**3*intrhoy2*scf**2 - 252*A*E*G*Izz*L**3*intrho*scf + 1680*E**2*Izz**2*L*intrho)/(70*A**2*G**2*L**4*scf**2 - 1680*A*E*G*Izz*L**2*scf + 10080*E**2*Izz**2)) + r23*(6*A**2*G**2*L**3*intrhoyz*r22*scf**2/(5*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) - A*G*L**2*intrhoz*r21*scf/(2*A*G*L**2*scf - 24*E*Iyy) + r23*(9*A**2*G**2*L**5*intrho*scf**2 - 84*A**2*G**2*L**3*intrhoz2*scf**2 - 252*A*E*G*Iyy*L**3*intrho*scf + 1680*E**2*Iyy**2*L*intrho)/(70*A**2*G**2*L**4*scf**2 - 1680*A*E*G*Iyy*L**2*scf + 10080*E**2*Iyy**2))
                k += 1
                Mv[k] += r31*(-A*G*L**2*intrhoy*r22*scf/(2*A*G*L**2*scf - 24*E*Izz) + A*G*L**2*intrhoz*r23*scf/(2*(A*G*L**2*scf - 12*E*Iyy)) + L*intrho*r21/6) + r32*(6*A**2*G**2*L**3*intrhoyz*r23*scf**2/(5*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + A*G*L**2*intrhoy*r21*scf/(2*(A*G*L**2*scf - 12*E*Izz)) + r22*(9*A**2*G**2*L**5*intrho*scf**2 - 84*A**2*G**2*L**3*intrhoy2*scf**2 - 252*A*E*G*Izz*L**3*intrho*scf + 1680*E**2*Izz**2*L*intrho)/(70*A**2*G**2*L**4*scf**2 - 1680*A*E*G*Izz*L**2*scf + 10080*E**2*Izz**2)) + r33*(6*A**2*G**2*L**3*intrhoyz*r22*scf**2/(5*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) - A*G*L**2*intrhoz*r21*scf/(2*A*G*L**2*scf - 24*E*Iyy) + r23*(9*A**2*G**2*L**5*intrho*scf**2 - 84*A**2*G**2*L**3*intrhoz2*scf**2 - 252*A*E*G*Iyy*L**3*intrho*scf + 1680*E**2*Iyy**2*L*intrho)/(70*A**2*G**2*L**4*scf**2 - 1680*A*E*G*Iyy*L**2*scf + 10080*E**2*Iyy**2))
                k += 1
                Mv[k] += r11*(L*intrhoy*r23*(3*A*G*L**2*scf - 40*E*Iyy)/(20*(A*G*L**2*scf - 12*E*Iyy)) + L*intrhoz*r22*(-3*A*G*L**2*scf + 40*E*Izz)/(20*(A*G*L**2*scf - 12*E*Izz))) + r12*(A*G*L**2*intrhoyz*r22*scf*(A*G*L**2*scf + 60*E*Iyy)/(10*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + L**2*r23*(13*A**2*G**2*L**4*intrho*scf**2 - 42*A**2*G**2*L**2*intrhoz2*scf**2 - 378*A*E*G*Iyy*L**2*intrho*scf - 2520*A*E*G*Iyy*intrhoz2*scf + 2520*E**2*Iyy**2*intrho)/(420*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)) - L*intrhoz*r21*(A*G*L**2*scf + 24*E*Iyy)/(12*A*G*L**2*scf - 144*E*Iyy)) + r13*(A*G*L**2*intrhoyz*r23*scf*(A*G*L**2*scf + 60*E*Izz)/(10*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + L**2*r22*(13*A**2*G**2*L**4*intrho*scf**2 - 42*A**2*G**2*L**2*intrhoy2*scf**2 - 378*A*E*G*Izz*L**2*intrho*scf - 2520*A*E*G*Izz*intrhoy2*scf + 2520*E**2*Izz**2*intrho)/(420*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)) + L*intrhoy*r21*(A*G*L**2*scf + 24*E*Izz)/(12*(A*G*L**2*scf - 12*E*Izz)))
                k += 1
                Mv[k] += r21*(L*intrhoy*r23*(3*A*G*L**2*scf - 40*E*Iyy)/(20*(A*G*L**2*scf - 12*E*Iyy)) + L*intrhoz*r22*(-3*A*G*L**2*scf + 40*E*Izz)/(20*(A*G*L**2*scf - 12*E*Izz))) + r22*(A*G*L**2*intrhoyz*r22*scf*(A*G*L**2*scf + 60*E*Iyy)/(10*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + L**2*r23*(13*A**2*G**2*L**4*intrho*scf**2 - 42*A**2*G**2*L**2*intrhoz2*scf**2 - 378*A*E*G*Iyy*L**2*intrho*scf - 2520*A*E*G*Iyy*intrhoz2*scf + 2520*E**2*Iyy**2*intrho)/(420*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)) - L*intrhoz*r21*(A*G*L**2*scf + 24*E*Iyy)/(12*A*G*L**2*scf - 144*E*Iyy)) + r23*(A*G*L**2*intrhoyz*r23*scf*(A*G*L**2*scf + 60*E*Izz)/(10*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + L**2*r22*(13*A**2*G**2*L**4*intrho*scf**2 - 42*A**2*G**2*L**2*intrhoy2*scf**2 - 378*A*E*G*Izz*L**2*intrho*scf - 2520*A*E*G*Izz*intrhoy2*scf + 2520*E**2*Izz**2*intrho)/(420*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)) + L*intrhoy*r21*(A*G*L**2*scf + 24*E*Izz)/(12*(A*G*L**2*scf - 12*E*Izz)))
                k += 1
                Mv[k] += r31*(L*intrhoy*r23*(3*A*G*L**2*scf - 40*E*Iyy)/(20*(A*G*L**2*scf - 12*E*Iyy)) + L*intrhoz*r22*(-3*A*G*L**2*scf + 40*E*Izz)/(20*(A*G*L**2*scf - 12*E*Izz))) + r32*(A*G*L**2*intrhoyz*r22*scf*(A*G*L**2*scf + 60*E*Iyy)/(10*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + L**2*r23*(13*A**2*G**2*L**4*intrho*scf**2 - 42*A**2*G**2*L**2*intrhoz2*scf**2 - 378*A*E*G*Iyy*L**2*intrho*scf - 2520*A*E*G*Iyy*intrhoz2*scf + 2520*E**2*Iyy**2*intrho)/(420*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)) - L*intrhoz*r21*(A*G*L**2*scf + 24*E*Iyy)/(12*A*G*L**2*scf - 144*E*Iyy)) + r33*(A*G*L**2*intrhoyz*r23*scf*(A*G*L**2*scf + 60*E*Izz)/(10*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + L**2*r22*(13*A**2*G**2*L**4*intrho*scf**2 - 42*A**2*G**2*L**2*intrhoy2*scf**2 - 378*A*E*G*Izz*L**2*intrho*scf - 2520*A*E*G*Izz*intrhoy2*scf + 2520*E**2*Izz**2*intrho)/(420*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)) + L*intrhoy*r21*(A*G*L**2*scf + 24*E*Izz)/(12*(A*G*L**2*scf - 12*E*Izz)))
                k += 1
                Mv[k] += r11*(-A*G*L**2*intrhoy*r22*scf/(2*A*G*L**2*scf - 24*E*Izz) + A*G*L**2*intrhoz*r23*scf/(2*(A*G*L**2*scf - 12*E*Iyy)) + L*intrho*r21/3) + r12*(-6*A**2*G**2*L**3*intrhoyz*r23*scf**2/(5*A**2*G**2*L**4*scf**2 - 60*A*E*G*Iyy*L**2*scf - 60*A*E*G*Izz*L**2*scf + 720*E**2*Iyy*Izz) - A*G*L**2*intrhoy*r21*scf/(2*A*G*L**2*scf - 24*E*Izz) + L*r22*(13*A**2*G**2*L**4*intrho*scf**2 + 42*A**2*G**2*L**2*intrhoy2*scf**2 - 294*A*E*G*Izz*L**2*intrho*scf + 1680*E**2*Izz**2*intrho)/(35*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2))) + r13*(-6*A**2*G**2*L**3*intrhoyz*r22*scf**2/(5*A**2*G**2*L**4*scf**2 - 60*A*E*G*Iyy*L**2*scf - 60*A*E*G*Izz*L**2*scf + 720*E**2*Iyy*Izz) + A*G*L**2*intrhoz*r21*scf/(2*(A*G*L**2*scf - 12*E*Iyy)) + L*r23*(13*A**2*G**2*L**4*intrho*scf**2 + 42*A**2*G**2*L**2*intrhoz2*scf**2 - 294*A*E*G*Iyy*L**2*intrho*scf + 1680*E**2*Iyy**2*intrho)/(35*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)))
                k += 1
                Mv[k] += r21*(-A*G*L**2*intrhoy*r22*scf/(2*A*G*L**2*scf - 24*E*Izz) + A*G*L**2*intrhoz*r23*scf/(2*(A*G*L**2*scf - 12*E*Iyy)) + L*intrho*r21/3) + r22*(-6*A**2*G**2*L**3*intrhoyz*r23*scf**2/(5*A**2*G**2*L**4*scf**2 - 60*A*E*G*Iyy*L**2*scf - 60*A*E*G*Izz*L**2*scf + 720*E**2*Iyy*Izz) - A*G*L**2*intrhoy*r21*scf/(2*A*G*L**2*scf - 24*E*Izz) + L*r22*(13*A**2*G**2*L**4*intrho*scf**2 + 42*A**2*G**2*L**2*intrhoy2*scf**2 - 294*A*E*G*Izz*L**2*intrho*scf + 1680*E**2*Izz**2*intrho)/(35*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2))) + r23*(-6*A**2*G**2*L**3*intrhoyz*r22*scf**2/(5*A**2*G**2*L**4*scf**2 - 60*A*E*G*Iyy*L**2*scf - 60*A*E*G*Izz*L**2*scf + 720*E**2*Iyy*Izz) + A*G*L**2*intrhoz*r21*scf/(2*(A*G*L**2*scf - 12*E*Iyy)) + L*r23*(13*A**2*G**2*L**4*intrho*scf**2 + 42*A**2*G**2*L**2*intrhoz2*scf**2 - 294*A*E*G*Iyy*L**2*intrho*scf + 1680*E**2*Iyy**2*intrho)/(35*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)))
                k += 1
                Mv[k] += r31*(-A*G*L**2*intrhoy*r22*scf/(2*A*G*L**2*scf - 24*E*Izz) + A*G*L**2*intrhoz*r23*scf/(2*(A*G*L**2*scf - 12*E*Iyy)) + L*intrho*r21/3) + r32*(-6*A**2*G**2*L**3*intrhoyz*r23*scf**2/(5*A**2*G**2*L**4*scf**2 - 60*A*E*G*Iyy*L**2*scf - 60*A*E*G*Izz*L**2*scf + 720*E**2*Iyy*Izz) - A*G*L**2*intrhoy*r21*scf/(2*A*G*L**2*scf - 24*E*Izz) + L*r22*(13*A**2*G**2*L**4*intrho*scf**2 + 42*A**2*G**2*L**2*intrhoy2*scf**2 - 294*A*E*G*Izz*L**2*intrho*scf + 1680*E**2*Izz**2*intrho)/(35*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2))) + r33*(-6*A**2*G**2*L**3*intrhoyz*r22*scf**2/(5*A**2*G**2*L**4*scf**2 - 60*A*E*G*Iyy*L**2*scf - 60*A*E*G*Izz*L**2*scf + 720*E**2*Iyy*Izz) + A*G*L**2*intrhoz*r21*scf/(2*(A*G*L**2*scf - 12*E*Iyy)) + L*r23*(13*A**2*G**2*L**4*intrho*scf**2 + 42*A**2*G**2*L**2*intrhoz2*scf**2 - 294*A*E*G*Iyy*L**2*intrho*scf + 1680*E**2*Iyy**2*intrho)/(35*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)))
                k += 1
                Mv[k] += r11*(L*intrhoy*r23*(7*A*G*L**2*scf - 80*E*Iyy)/(20*(A*G*L**2*scf - 12*E*Iyy)) + L*intrhoz*r22*(-7*A*G*L**2*scf + 80*E*Izz)/(20*(A*G*L**2*scf - 12*E*Izz))) + r12*(A*G*L**2*intrhoyz*r22*scf*(A*G*L**2*scf + 60*E*Iyy)/(10*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + L**2*r23*(-11*A**2*G**2*L**4*intrho*scf**2 - 21*A**2*G**2*L**2*intrhoz2*scf**2 + 231*A*E*G*Iyy*L**2*intrho*scf - 1260*A*E*G*Iyy*intrhoz2*scf - 1260*E**2*Iyy**2*intrho)/(210*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)) + L*intrhoz*r21*(A*G*L**2*scf - 48*E*Iyy)/(12*(A*G*L**2*scf - 12*E*Iyy))) + r13*(A*G*L**2*intrhoyz*r23*scf*(A*G*L**2*scf + 60*E*Izz)/(10*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + L**2*r22*(-11*A**2*G**2*L**4*intrho*scf**2 - 21*A**2*G**2*L**2*intrhoy2*scf**2 + 231*A*E*G*Izz*L**2*intrho*scf - 1260*A*E*G*Izz*intrhoy2*scf - 1260*E**2*Izz**2*intrho)/(210*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)) + L*intrhoy*r21*(-A*G*L**2*scf + 48*E*Izz)/(12*(A*G*L**2*scf - 12*E*Izz)))
                k += 1
                Mv[k] += r21*(L*intrhoy*r23*(7*A*G*L**2*scf - 80*E*Iyy)/(20*(A*G*L**2*scf - 12*E*Iyy)) + L*intrhoz*r22*(-7*A*G*L**2*scf + 80*E*Izz)/(20*(A*G*L**2*scf - 12*E*Izz))) + r22*(A*G*L**2*intrhoyz*r22*scf*(A*G*L**2*scf + 60*E*Iyy)/(10*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + L**2*r23*(-11*A**2*G**2*L**4*intrho*scf**2 - 21*A**2*G**2*L**2*intrhoz2*scf**2 + 231*A*E*G*Iyy*L**2*intrho*scf - 1260*A*E*G*Iyy*intrhoz2*scf - 1260*E**2*Iyy**2*intrho)/(210*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)) + L*intrhoz*r21*(A*G*L**2*scf - 48*E*Iyy)/(12*(A*G*L**2*scf - 12*E*Iyy))) + r23*(A*G*L**2*intrhoyz*r23*scf*(A*G*L**2*scf + 60*E*Izz)/(10*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + L**2*r22*(-11*A**2*G**2*L**4*intrho*scf**2 - 21*A**2*G**2*L**2*intrhoy2*scf**2 + 231*A*E*G*Izz*L**2*intrho*scf - 1260*A*E*G*Izz*intrhoy2*scf - 1260*E**2*Izz**2*intrho)/(210*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)) + L*intrhoy*r21*(-A*G*L**2*scf + 48*E*Izz)/(12*(A*G*L**2*scf - 12*E*Izz)))
                k += 1
                Mv[k] += r31*(L*intrhoy*r23*(7*A*G*L**2*scf - 80*E*Iyy)/(20*(A*G*L**2*scf - 12*E*Iyy)) + L*intrhoz*r22*(-7*A*G*L**2*scf + 80*E*Izz)/(20*(A*G*L**2*scf - 12*E*Izz))) + r32*(A*G*L**2*intrhoyz*r22*scf*(A*G*L**2*scf + 60*E*Iyy)/(10*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + L**2*r23*(-11*A**2*G**2*L**4*intrho*scf**2 - 21*A**2*G**2*L**2*intrhoz2*scf**2 + 231*A*E*G*Iyy*L**2*intrho*scf - 1260*A*E*G*Iyy*intrhoz2*scf - 1260*E**2*Iyy**2*intrho)/(210*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)) + L*intrhoz*r21*(A*G*L**2*scf - 48*E*Iyy)/(12*(A*G*L**2*scf - 12*E*Iyy))) + r33*(A*G*L**2*intrhoyz*r23*scf*(A*G*L**2*scf + 60*E*Izz)/(10*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + L**2*r22*(-11*A**2*G**2*L**4*intrho*scf**2 - 21*A**2*G**2*L**2*intrhoy2*scf**2 + 231*A*E*G*Izz*L**2*intrho*scf - 1260*A*E*G*Izz*intrhoy2*scf - 1260*E**2*Izz**2*intrho)/(210*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)) + L*intrhoy*r21*(-A*G*L**2*scf + 48*E*Izz)/(12*(A*G*L**2*scf - 12*E*Izz)))
                k += 1
                Mv[k] += r11*(-A*G*L**2*intrhoy*r32*scf/(2*A*G*L**2*scf - 24*E*Izz) + A*G*L**2*intrhoz*r33*scf/(2*(A*G*L**2*scf - 12*E*Iyy)) + L*intrho*r31/6) + r12*(6*A**2*G**2*L**3*intrhoyz*r33*scf**2/(5*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + A*G*L**2*intrhoy*r31*scf/(2*(A*G*L**2*scf - 12*E*Izz)) + r32*(9*A**2*G**2*L**5*intrho*scf**2 - 84*A**2*G**2*L**3*intrhoy2*scf**2 - 252*A*E*G*Izz*L**3*intrho*scf + 1680*E**2*Izz**2*L*intrho)/(70*A**2*G**2*L**4*scf**2 - 1680*A*E*G*Izz*L**2*scf + 10080*E**2*Izz**2)) + r13*(6*A**2*G**2*L**3*intrhoyz*r32*scf**2/(5*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) - A*G*L**2*intrhoz*r31*scf/(2*A*G*L**2*scf - 24*E*Iyy) + r33*(9*A**2*G**2*L**5*intrho*scf**2 - 84*A**2*G**2*L**3*intrhoz2*scf**2 - 252*A*E*G*Iyy*L**3*intrho*scf + 1680*E**2*Iyy**2*L*intrho)/(70*A**2*G**2*L**4*scf**2 - 1680*A*E*G*Iyy*L**2*scf + 10080*E**2*Iyy**2))
                k += 1
                Mv[k] += r21*(-A*G*L**2*intrhoy*r32*scf/(2*A*G*L**2*scf - 24*E*Izz) + A*G*L**2*intrhoz*r33*scf/(2*(A*G*L**2*scf - 12*E*Iyy)) + L*intrho*r31/6) + r22*(6*A**2*G**2*L**3*intrhoyz*r33*scf**2/(5*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + A*G*L**2*intrhoy*r31*scf/(2*(A*G*L**2*scf - 12*E*Izz)) + r32*(9*A**2*G**2*L**5*intrho*scf**2 - 84*A**2*G**2*L**3*intrhoy2*scf**2 - 252*A*E*G*Izz*L**3*intrho*scf + 1680*E**2*Izz**2*L*intrho)/(70*A**2*G**2*L**4*scf**2 - 1680*A*E*G*Izz*L**2*scf + 10080*E**2*Izz**2)) + r23*(6*A**2*G**2*L**3*intrhoyz*r32*scf**2/(5*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) - A*G*L**2*intrhoz*r31*scf/(2*A*G*L**2*scf - 24*E*Iyy) + r33*(9*A**2*G**2*L**5*intrho*scf**2 - 84*A**2*G**2*L**3*intrhoz2*scf**2 - 252*A*E*G*Iyy*L**3*intrho*scf + 1680*E**2*Iyy**2*L*intrho)/(70*A**2*G**2*L**4*scf**2 - 1680*A*E*G*Iyy*L**2*scf + 10080*E**2*Iyy**2))
                k += 1
                Mv[k] += r31*(-A*G*L**2*intrhoy*r32*scf/(2*A*G*L**2*scf - 24*E*Izz) + A*G*L**2*intrhoz*r33*scf/(2*(A*G*L**2*scf - 12*E*Iyy)) + L*intrho*r31/6) + r32*(6*A**2*G**2*L**3*intrhoyz*r33*scf**2/(5*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + A*G*L**2*intrhoy*r31*scf/(2*(A*G*L**2*scf - 12*E*Izz)) + r32*(9*A**2*G**2*L**5*intrho*scf**2 - 84*A**2*G**2*L**3*intrhoy2*scf**2 - 252*A*E*G*Izz*L**3*intrho*scf + 1680*E**2*Izz**2*L*intrho)/(70*A**2*G**2*L**4*scf**2 - 1680*A*E*G*Izz*L**2*scf + 10080*E**2*Izz**2)) + r33*(6*A**2*G**2*L**3*intrhoyz*r32*scf**2/(5*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) - A*G*L**2*intrhoz*r31*scf/(2*A*G*L**2*scf - 24*E*Iyy) + r33*(9*A**2*G**2*L**5*intrho*scf**2 - 84*A**2*G**2*L**3*intrhoz2*scf**2 - 252*A*E*G*Iyy*L**3*intrho*scf + 1680*E**2*Iyy**2*L*intrho)/(70*A**2*G**2*L**4*scf**2 - 1680*A*E*G*Iyy*L**2*scf + 10080*E**2*Iyy**2))
                k += 1
                Mv[k] += r11*(L*intrhoy*r33*(3*A*G*L**2*scf - 40*E*Iyy)/(20*(A*G*L**2*scf - 12*E*Iyy)) + L*intrhoz*r32*(-3*A*G*L**2*scf + 40*E*Izz)/(20*(A*G*L**2*scf - 12*E*Izz))) + r12*(A*G*L**2*intrhoyz*r32*scf*(A*G*L**2*scf + 60*E*Iyy)/(10*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + L**2*r33*(13*A**2*G**2*L**4*intrho*scf**2 - 42*A**2*G**2*L**2*intrhoz2*scf**2 - 378*A*E*G*Iyy*L**2*intrho*scf - 2520*A*E*G*Iyy*intrhoz2*scf + 2520*E**2*Iyy**2*intrho)/(420*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)) - L*intrhoz*r31*(A*G*L**2*scf + 24*E*Iyy)/(12*A*G*L**2*scf - 144*E*Iyy)) + r13*(A*G*L**2*intrhoyz*r33*scf*(A*G*L**2*scf + 60*E*Izz)/(10*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + L**2*r32*(13*A**2*G**2*L**4*intrho*scf**2 - 42*A**2*G**2*L**2*intrhoy2*scf**2 - 378*A*E*G*Izz*L**2*intrho*scf - 2520*A*E*G*Izz*intrhoy2*scf + 2520*E**2*Izz**2*intrho)/(420*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)) + L*intrhoy*r31*(A*G*L**2*scf + 24*E*Izz)/(12*(A*G*L**2*scf - 12*E*Izz)))
                k += 1
                Mv[k] += r21*(L*intrhoy*r33*(3*A*G*L**2*scf - 40*E*Iyy)/(20*(A*G*L**2*scf - 12*E*Iyy)) + L*intrhoz*r32*(-3*A*G*L**2*scf + 40*E*Izz)/(20*(A*G*L**2*scf - 12*E*Izz))) + r22*(A*G*L**2*intrhoyz*r32*scf*(A*G*L**2*scf + 60*E*Iyy)/(10*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + L**2*r33*(13*A**2*G**2*L**4*intrho*scf**2 - 42*A**2*G**2*L**2*intrhoz2*scf**2 - 378*A*E*G*Iyy*L**2*intrho*scf - 2520*A*E*G*Iyy*intrhoz2*scf + 2520*E**2*Iyy**2*intrho)/(420*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)) - L*intrhoz*r31*(A*G*L**2*scf + 24*E*Iyy)/(12*A*G*L**2*scf - 144*E*Iyy)) + r23*(A*G*L**2*intrhoyz*r33*scf*(A*G*L**2*scf + 60*E*Izz)/(10*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + L**2*r32*(13*A**2*G**2*L**4*intrho*scf**2 - 42*A**2*G**2*L**2*intrhoy2*scf**2 - 378*A*E*G*Izz*L**2*intrho*scf - 2520*A*E*G*Izz*intrhoy2*scf + 2520*E**2*Izz**2*intrho)/(420*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)) + L*intrhoy*r31*(A*G*L**2*scf + 24*E*Izz)/(12*(A*G*L**2*scf - 12*E*Izz)))
                k += 1
                Mv[k] += r31*(L*intrhoy*r33*(3*A*G*L**2*scf - 40*E*Iyy)/(20*(A*G*L**2*scf - 12*E*Iyy)) + L*intrhoz*r32*(-3*A*G*L**2*scf + 40*E*Izz)/(20*(A*G*L**2*scf - 12*E*Izz))) + r32*(A*G*L**2*intrhoyz*r32*scf*(A*G*L**2*scf + 60*E*Iyy)/(10*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + L**2*r33*(13*A**2*G**2*L**4*intrho*scf**2 - 42*A**2*G**2*L**2*intrhoz2*scf**2 - 378*A*E*G*Iyy*L**2*intrho*scf - 2520*A*E*G*Iyy*intrhoz2*scf + 2520*E**2*Iyy**2*intrho)/(420*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)) - L*intrhoz*r31*(A*G*L**2*scf + 24*E*Iyy)/(12*A*G*L**2*scf - 144*E*Iyy)) + r33*(A*G*L**2*intrhoyz*r33*scf*(A*G*L**2*scf + 60*E*Izz)/(10*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + L**2*r32*(13*A**2*G**2*L**4*intrho*scf**2 - 42*A**2*G**2*L**2*intrhoy2*scf**2 - 378*A*E*G*Izz*L**2*intrho*scf - 2520*A*E*G*Izz*intrhoy2*scf + 2520*E**2*Izz**2*intrho)/(420*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)) + L*intrhoy*r31*(A*G*L**2*scf + 24*E*Izz)/(12*(A*G*L**2*scf - 12*E*Izz)))
                k += 1
                Mv[k] += r11*(-A*G*L**2*intrhoy*r32*scf/(2*A*G*L**2*scf - 24*E*Izz) + A*G*L**2*intrhoz*r33*scf/(2*(A*G*L**2*scf - 12*E*Iyy)) + L*intrho*r31/3) + r12*(-6*A**2*G**2*L**3*intrhoyz*r33*scf**2/(5*A**2*G**2*L**4*scf**2 - 60*A*E*G*Iyy*L**2*scf - 60*A*E*G*Izz*L**2*scf + 720*E**2*Iyy*Izz) - A*G*L**2*intrhoy*r31*scf/(2*A*G*L**2*scf - 24*E*Izz) + L*r32*(13*A**2*G**2*L**4*intrho*scf**2 + 42*A**2*G**2*L**2*intrhoy2*scf**2 - 294*A*E*G*Izz*L**2*intrho*scf + 1680*E**2*Izz**2*intrho)/(35*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2))) + r13*(-6*A**2*G**2*L**3*intrhoyz*r32*scf**2/(5*A**2*G**2*L**4*scf**2 - 60*A*E*G*Iyy*L**2*scf - 60*A*E*G*Izz*L**2*scf + 720*E**2*Iyy*Izz) + A*G*L**2*intrhoz*r31*scf/(2*(A*G*L**2*scf - 12*E*Iyy)) + L*r33*(13*A**2*G**2*L**4*intrho*scf**2 + 42*A**2*G**2*L**2*intrhoz2*scf**2 - 294*A*E*G*Iyy*L**2*intrho*scf + 1680*E**2*Iyy**2*intrho)/(35*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)))
                k += 1
                Mv[k] += r21*(-A*G*L**2*intrhoy*r32*scf/(2*A*G*L**2*scf - 24*E*Izz) + A*G*L**2*intrhoz*r33*scf/(2*(A*G*L**2*scf - 12*E*Iyy)) + L*intrho*r31/3) + r22*(-6*A**2*G**2*L**3*intrhoyz*r33*scf**2/(5*A**2*G**2*L**4*scf**2 - 60*A*E*G*Iyy*L**2*scf - 60*A*E*G*Izz*L**2*scf + 720*E**2*Iyy*Izz) - A*G*L**2*intrhoy*r31*scf/(2*A*G*L**2*scf - 24*E*Izz) + L*r32*(13*A**2*G**2*L**4*intrho*scf**2 + 42*A**2*G**2*L**2*intrhoy2*scf**2 - 294*A*E*G*Izz*L**2*intrho*scf + 1680*E**2*Izz**2*intrho)/(35*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2))) + r23*(-6*A**2*G**2*L**3*intrhoyz*r32*scf**2/(5*A**2*G**2*L**4*scf**2 - 60*A*E*G*Iyy*L**2*scf - 60*A*E*G*Izz*L**2*scf + 720*E**2*Iyy*Izz) + A*G*L**2*intrhoz*r31*scf/(2*(A*G*L**2*scf - 12*E*Iyy)) + L*r33*(13*A**2*G**2*L**4*intrho*scf**2 + 42*A**2*G**2*L**2*intrhoz2*scf**2 - 294*A*E*G*Iyy*L**2*intrho*scf + 1680*E**2*Iyy**2*intrho)/(35*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)))
                k += 1
                Mv[k] += r31*(-A*G*L**2*intrhoy*r32*scf/(2*A*G*L**2*scf - 24*E*Izz) + A*G*L**2*intrhoz*r33*scf/(2*(A*G*L**2*scf - 12*E*Iyy)) + L*intrho*r31/3) + r32*(-6*A**2*G**2*L**3*intrhoyz*r33*scf**2/(5*A**2*G**2*L**4*scf**2 - 60*A*E*G*Iyy*L**2*scf - 60*A*E*G*Izz*L**2*scf + 720*E**2*Iyy*Izz) - A*G*L**2*intrhoy*r31*scf/(2*A*G*L**2*scf - 24*E*Izz) + L*r32*(13*A**2*G**2*L**4*intrho*scf**2 + 42*A**2*G**2*L**2*intrhoy2*scf**2 - 294*A*E*G*Izz*L**2*intrho*scf + 1680*E**2*Izz**2*intrho)/(35*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2))) + r33*(-6*A**2*G**2*L**3*intrhoyz*r32*scf**2/(5*A**2*G**2*L**4*scf**2 - 60*A*E*G*Iyy*L**2*scf - 60*A*E*G*Izz*L**2*scf + 720*E**2*Iyy*Izz) + A*G*L**2*intrhoz*r31*scf/(2*(A*G*L**2*scf - 12*E*Iyy)) + L*r33*(13*A**2*G**2*L**4*intrho*scf**2 + 42*A**2*G**2*L**2*intrhoz2*scf**2 - 294*A*E*G*Iyy*L**2*intrho*scf + 1680*E**2*Iyy**2*intrho)/(35*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)))
                k += 1
                Mv[k] += r11*(L*intrhoy*r33*(7*A*G*L**2*scf - 80*E*Iyy)/(20*(A*G*L**2*scf - 12*E*Iyy)) + L*intrhoz*r32*(-7*A*G*L**2*scf + 80*E*Izz)/(20*(A*G*L**2*scf - 12*E*Izz))) + r12*(A*G*L**2*intrhoyz*r32*scf*(A*G*L**2*scf + 60*E*Iyy)/(10*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + L**2*r33*(-11*A**2*G**2*L**4*intrho*scf**2 - 21*A**2*G**2*L**2*intrhoz2*scf**2 + 231*A*E*G*Iyy*L**2*intrho*scf - 1260*A*E*G*Iyy*intrhoz2*scf - 1260*E**2*Iyy**2*intrho)/(210*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)) + L*intrhoz*r31*(A*G*L**2*scf - 48*E*Iyy)/(12*(A*G*L**2*scf - 12*E*Iyy))) + r13*(A*G*L**2*intrhoyz*r33*scf*(A*G*L**2*scf + 60*E*Izz)/(10*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + L**2*r32*(-11*A**2*G**2*L**4*intrho*scf**2 - 21*A**2*G**2*L**2*intrhoy2*scf**2 + 231*A*E*G*Izz*L**2*intrho*scf - 1260*A*E*G*Izz*intrhoy2*scf - 1260*E**2*Izz**2*intrho)/(210*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)) + L*intrhoy*r31*(-A*G*L**2*scf + 48*E*Izz)/(12*(A*G*L**2*scf - 12*E*Izz)))
                k += 1
                Mv[k] += r21*(L*intrhoy*r33*(7*A*G*L**2*scf - 80*E*Iyy)/(20*(A*G*L**2*scf - 12*E*Iyy)) + L*intrhoz*r32*(-7*A*G*L**2*scf + 80*E*Izz)/(20*(A*G*L**2*scf - 12*E*Izz))) + r22*(A*G*L**2*intrhoyz*r32*scf*(A*G*L**2*scf + 60*E*Iyy)/(10*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + L**2*r33*(-11*A**2*G**2*L**4*intrho*scf**2 - 21*A**2*G**2*L**2*intrhoz2*scf**2 + 231*A*E*G*Iyy*L**2*intrho*scf - 1260*A*E*G*Iyy*intrhoz2*scf - 1260*E**2*Iyy**2*intrho)/(210*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)) + L*intrhoz*r31*(A*G*L**2*scf - 48*E*Iyy)/(12*(A*G*L**2*scf - 12*E*Iyy))) + r23*(A*G*L**2*intrhoyz*r33*scf*(A*G*L**2*scf + 60*E*Izz)/(10*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + L**2*r32*(-11*A**2*G**2*L**4*intrho*scf**2 - 21*A**2*G**2*L**2*intrhoy2*scf**2 + 231*A*E*G*Izz*L**2*intrho*scf - 1260*A*E*G*Izz*intrhoy2*scf - 1260*E**2*Izz**2*intrho)/(210*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)) + L*intrhoy*r31*(-A*G*L**2*scf + 48*E*Izz)/(12*(A*G*L**2*scf - 12*E*Izz)))
                k += 1
                Mv[k] += r31*(L*intrhoy*r33*(7*A*G*L**2*scf - 80*E*Iyy)/(20*(A*G*L**2*scf - 12*E*Iyy)) + L*intrhoz*r32*(-7*A*G*L**2*scf + 80*E*Izz)/(20*(A*G*L**2*scf - 12*E*Izz))) + r32*(A*G*L**2*intrhoyz*r32*scf*(A*G*L**2*scf + 60*E*Iyy)/(10*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + L**2*r33*(-11*A**2*G**2*L**4*intrho*scf**2 - 21*A**2*G**2*L**2*intrhoz2*scf**2 + 231*A*E*G*Iyy*L**2*intrho*scf - 1260*A*E*G*Iyy*intrhoz2*scf - 1260*E**2*Iyy**2*intrho)/(210*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)) + L*intrhoz*r31*(A*G*L**2*scf - 48*E*Iyy)/(12*(A*G*L**2*scf - 12*E*Iyy))) + r33*(A*G*L**2*intrhoyz*r33*scf*(A*G*L**2*scf + 60*E*Izz)/(10*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + L**2*r32*(-11*A**2*G**2*L**4*intrho*scf**2 - 21*A**2*G**2*L**2*intrhoy2*scf**2 + 231*A*E*G*Izz*L**2*intrho*scf - 1260*A*E*G*Izz*intrhoy2*scf - 1260*E**2*Izz**2*intrho)/(210*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)) + L*intrhoy*r31*(-A*G*L**2*scf + 48*E*Izz)/(12*(A*G*L**2*scf - 12*E*Izz)))
                k += 1
                Mv[k] += r11*(L*intrhoy*r13*(A*G*L**2*scf + 24*E*Izz)/(12*(A*G*L**2*scf - 12*E*Izz)) - L*intrhoz*r12*(A*G*L**2*scf + 24*E*Iyy)/(12*A*G*L**2*scf - 144*E*Iyy)) + r12*(-A*G*L**2*intrhoyz*r12*scf*(A*G*L**2*scf + 60*E*Iyy)/(10*A**2*G**2*L**4*scf**2 - 120*A*E*G*Iyy*L**2*scf - 120*A*E*G*Izz*L**2*scf + 1440*E**2*Iyy*Izz) + L**2*r13*(-13*A**2*G**2*L**4*intrho*scf**2 + 42*A**2*G**2*L**2*intrhoy2*scf**2 + 378*A*E*G*Izz*L**2*intrho*scf + 2520*A*E*G*Izz*intrhoy2*scf - 2520*E**2*Izz**2*intrho)/(420*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)) + L*intrhoz*r11*(-3*A*G*L**2*scf + 40*E*Izz)/(20*(A*G*L**2*scf - 12*E*Izz))) + r13*(-A*G*L**2*intrhoyz*r13*scf*(A*G*L**2*scf + 60*E*Izz)/(10*A**2*G**2*L**4*scf**2 - 120*A*E*G*Iyy*L**2*scf - 120*A*E*G*Izz*L**2*scf + 1440*E**2*Iyy*Izz) + L**2*r12*(-13*A**2*G**2*L**4*intrho*scf**2 + 42*A**2*G**2*L**2*intrhoz2*scf**2 + 378*A*E*G*Iyy*L**2*intrho*scf + 2520*A*E*G*Iyy*intrhoz2*scf - 2520*E**2*Iyy**2*intrho)/(420*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)) + L*intrhoy*r11*(3*A*G*L**2*scf - 40*E*Iyy)/(20*(A*G*L**2*scf - 12*E*Iyy)))
                k += 1
                Mv[k] += r21*(L*intrhoy*r13*(A*G*L**2*scf + 24*E*Izz)/(12*(A*G*L**2*scf - 12*E*Izz)) - L*intrhoz*r12*(A*G*L**2*scf + 24*E*Iyy)/(12*A*G*L**2*scf - 144*E*Iyy)) + r22*(-A*G*L**2*intrhoyz*r12*scf*(A*G*L**2*scf + 60*E*Iyy)/(10*A**2*G**2*L**4*scf**2 - 120*A*E*G*Iyy*L**2*scf - 120*A*E*G*Izz*L**2*scf + 1440*E**2*Iyy*Izz) + L**2*r13*(-13*A**2*G**2*L**4*intrho*scf**2 + 42*A**2*G**2*L**2*intrhoy2*scf**2 + 378*A*E*G*Izz*L**2*intrho*scf + 2520*A*E*G*Izz*intrhoy2*scf - 2520*E**2*Izz**2*intrho)/(420*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)) + L*intrhoz*r11*(-3*A*G*L**2*scf + 40*E*Izz)/(20*(A*G*L**2*scf - 12*E*Izz))) + r23*(-A*G*L**2*intrhoyz*r13*scf*(A*G*L**2*scf + 60*E*Izz)/(10*A**2*G**2*L**4*scf**2 - 120*A*E*G*Iyy*L**2*scf - 120*A*E*G*Izz*L**2*scf + 1440*E**2*Iyy*Izz) + L**2*r12*(-13*A**2*G**2*L**4*intrho*scf**2 + 42*A**2*G**2*L**2*intrhoz2*scf**2 + 378*A*E*G*Iyy*L**2*intrho*scf + 2520*A*E*G*Iyy*intrhoz2*scf - 2520*E**2*Iyy**2*intrho)/(420*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)) + L*intrhoy*r11*(3*A*G*L**2*scf - 40*E*Iyy)/(20*(A*G*L**2*scf - 12*E*Iyy)))
                k += 1
                Mv[k] += r31*(L*intrhoy*r13*(A*G*L**2*scf + 24*E*Izz)/(12*(A*G*L**2*scf - 12*E*Izz)) - L*intrhoz*r12*(A*G*L**2*scf + 24*E*Iyy)/(12*A*G*L**2*scf - 144*E*Iyy)) + r32*(-A*G*L**2*intrhoyz*r12*scf*(A*G*L**2*scf + 60*E*Iyy)/(10*A**2*G**2*L**4*scf**2 - 120*A*E*G*Iyy*L**2*scf - 120*A*E*G*Izz*L**2*scf + 1440*E**2*Iyy*Izz) + L**2*r13*(-13*A**2*G**2*L**4*intrho*scf**2 + 42*A**2*G**2*L**2*intrhoy2*scf**2 + 378*A*E*G*Izz*L**2*intrho*scf + 2520*A*E*G*Izz*intrhoy2*scf - 2520*E**2*Izz**2*intrho)/(420*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)) + L*intrhoz*r11*(-3*A*G*L**2*scf + 40*E*Izz)/(20*(A*G*L**2*scf - 12*E*Izz))) + r33*(-A*G*L**2*intrhoyz*r13*scf*(A*G*L**2*scf + 60*E*Izz)/(10*A**2*G**2*L**4*scf**2 - 120*A*E*G*Iyy*L**2*scf - 120*A*E*G*Izz*L**2*scf + 1440*E**2*Iyy*Izz) + L**2*r12*(-13*A**2*G**2*L**4*intrho*scf**2 + 42*A**2*G**2*L**2*intrhoz2*scf**2 + 378*A*E*G*Iyy*L**2*intrho*scf + 2520*A*E*G*Iyy*intrhoz2*scf - 2520*E**2*Iyy**2*intrho)/(420*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)) + L*intrhoy*r11*(3*A*G*L**2*scf - 40*E*Iyy)/(20*(A*G*L**2*scf - 12*E*Iyy)))
                k += 1
                Mv[k] += r11*(L**2*intrhoy*r12*(-A*G*L**2*scf + 15*E*Iyy)/(30*(A*G*L**2*scf - 12*E*Iyy)) + L**2*intrhoz*r13*(A*G*L**2*scf - 15*E*Izz)/(30*(A*G*L**2*scf - 12*E*Izz)) + L*r11*(intrhoy2 + intrhoz2)/6) + r12*(L**2*intrhoy*r11*(A*G*L**2*scf - 15*E*Iyy)/(30*(A*G*L**2*scf - 12*E*Iyy)) + L*intrhoyz*r13*(A**2*G**2*L**4*scf**2 - 30*A*E*G*Iyy*L**2*scf - 30*A*E*G*Izz*L**2*scf - 720*E**2*Iyy*Izz)/(30*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + L*r12*(-3*A**2*G**2*L**6*intrho*scf**2 - 14*A**2*G**2*L**4*intrhoz2*scf**2 + 84*A*E*G*Iyy*L**4*intrho*scf + 840*A*E*G*Iyy*L**2*intrhoz2*scf - 504*E**2*Iyy**2*L**2*intrho + 10080*E**2*Iyy**2*intrhoz2)/(420*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2))) + r13*(L**2*intrhoz*r11*(-A*G*L**2*scf + 15*E*Izz)/(30*(A*G*L**2*scf - 12*E*Izz)) + L*intrhoyz*r12*(A**2*G**2*L**4*scf**2 - 30*A*E*G*Iyy*L**2*scf - 30*A*E*G*Izz*L**2*scf - 720*E**2*Iyy*Izz)/(30*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + L*r13*(-3*A**2*G**2*L**6*intrho*scf**2 - 14*A**2*G**2*L**4*intrhoy2*scf**2 + 84*A*E*G*Izz*L**4*intrho*scf + 840*A*E*G*Izz*L**2*intrhoy2*scf - 504*E**2*Izz**2*L**2*intrho + 10080*E**2*Izz**2*intrhoy2)/(420*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)))
                k += 1
                Mv[k] += r21*(L**2*intrhoy*r12*(-A*G*L**2*scf + 15*E*Iyy)/(30*(A*G*L**2*scf - 12*E*Iyy)) + L**2*intrhoz*r13*(A*G*L**2*scf - 15*E*Izz)/(30*(A*G*L**2*scf - 12*E*Izz)) + L*r11*(intrhoy2 + intrhoz2)/6) + r22*(L**2*intrhoy*r11*(A*G*L**2*scf - 15*E*Iyy)/(30*(A*G*L**2*scf - 12*E*Iyy)) + L*intrhoyz*r13*(A**2*G**2*L**4*scf**2 - 30*A*E*G*Iyy*L**2*scf - 30*A*E*G*Izz*L**2*scf - 720*E**2*Iyy*Izz)/(30*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + L*r12*(-3*A**2*G**2*L**6*intrho*scf**2 - 14*A**2*G**2*L**4*intrhoz2*scf**2 + 84*A*E*G*Iyy*L**4*intrho*scf + 840*A*E*G*Iyy*L**2*intrhoz2*scf - 504*E**2*Iyy**2*L**2*intrho + 10080*E**2*Iyy**2*intrhoz2)/(420*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2))) + r23*(L**2*intrhoz*r11*(-A*G*L**2*scf + 15*E*Izz)/(30*(A*G*L**2*scf - 12*E*Izz)) + L*intrhoyz*r12*(A**2*G**2*L**4*scf**2 - 30*A*E*G*Iyy*L**2*scf - 30*A*E*G*Izz*L**2*scf - 720*E**2*Iyy*Izz)/(30*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + L*r13*(-3*A**2*G**2*L**6*intrho*scf**2 - 14*A**2*G**2*L**4*intrhoy2*scf**2 + 84*A*E*G*Izz*L**4*intrho*scf + 840*A*E*G*Izz*L**2*intrhoy2*scf - 504*E**2*Izz**2*L**2*intrho + 10080*E**2*Izz**2*intrhoy2)/(420*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)))
                k += 1
                Mv[k] += r31*(L**2*intrhoy*r12*(-A*G*L**2*scf + 15*E*Iyy)/(30*(A*G*L**2*scf - 12*E*Iyy)) + L**2*intrhoz*r13*(A*G*L**2*scf - 15*E*Izz)/(30*(A*G*L**2*scf - 12*E*Izz)) + L*r11*(intrhoy2 + intrhoz2)/6) + r32*(L**2*intrhoy*r11*(A*G*L**2*scf - 15*E*Iyy)/(30*(A*G*L**2*scf - 12*E*Iyy)) + L*intrhoyz*r13*(A**2*G**2*L**4*scf**2 - 30*A*E*G*Iyy*L**2*scf - 30*A*E*G*Izz*L**2*scf - 720*E**2*Iyy*Izz)/(30*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + L*r12*(-3*A**2*G**2*L**6*intrho*scf**2 - 14*A**2*G**2*L**4*intrhoz2*scf**2 + 84*A*E*G*Iyy*L**4*intrho*scf + 840*A*E*G*Iyy*L**2*intrhoz2*scf - 504*E**2*Iyy**2*L**2*intrho + 10080*E**2*Iyy**2*intrhoz2)/(420*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2))) + r33*(L**2*intrhoz*r11*(-A*G*L**2*scf + 15*E*Izz)/(30*(A*G*L**2*scf - 12*E*Izz)) + L*intrhoyz*r12*(A**2*G**2*L**4*scf**2 - 30*A*E*G*Iyy*L**2*scf - 30*A*E*G*Izz*L**2*scf - 720*E**2*Iyy*Izz)/(30*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + L*r13*(-3*A**2*G**2*L**6*intrho*scf**2 - 14*A**2*G**2*L**4*intrhoy2*scf**2 + 84*A*E*G*Izz*L**4*intrho*scf + 840*A*E*G*Izz*L**2*intrhoy2*scf - 504*E**2*Izz**2*L**2*intrho + 10080*E**2*Izz**2*intrhoy2)/(420*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)))
                k += 1
                Mv[k] += r11*(L*intrhoy*r13*(-A*G*L**2*scf + 48*E*Izz)/(12*(A*G*L**2*scf - 12*E*Izz)) + L*intrhoz*r12*(A*G*L**2*scf - 48*E*Iyy)/(12*(A*G*L**2*scf - 12*E*Iyy))) + r12*(A*G*L**2*intrhoyz*r12*scf*(A*G*L**2*scf + 60*E*Iyy)/(10*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + L**2*r13*(-11*A**2*G**2*L**4*intrho*scf**2 - 21*A**2*G**2*L**2*intrhoy2*scf**2 + 231*A*E*G*Izz*L**2*intrho*scf - 1260*A*E*G*Izz*intrhoy2*scf - 1260*E**2*Izz**2*intrho)/(210*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)) + L*intrhoz*r11*(-7*A*G*L**2*scf + 80*E*Izz)/(20*(A*G*L**2*scf - 12*E*Izz))) + r13*(A*G*L**2*intrhoyz*r13*scf*(A*G*L**2*scf + 60*E*Izz)/(10*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + L**2*r12*(-11*A**2*G**2*L**4*intrho*scf**2 - 21*A**2*G**2*L**2*intrhoz2*scf**2 + 231*A*E*G*Iyy*L**2*intrho*scf - 1260*A*E*G*Iyy*intrhoz2*scf - 1260*E**2*Iyy**2*intrho)/(210*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)) + L*intrhoy*r11*(7*A*G*L**2*scf - 80*E*Iyy)/(20*(A*G*L**2*scf - 12*E*Iyy)))
                k += 1
                Mv[k] += r21*(L*intrhoy*r13*(-A*G*L**2*scf + 48*E*Izz)/(12*(A*G*L**2*scf - 12*E*Izz)) + L*intrhoz*r12*(A*G*L**2*scf - 48*E*Iyy)/(12*(A*G*L**2*scf - 12*E*Iyy))) + r22*(A*G*L**2*intrhoyz*r12*scf*(A*G*L**2*scf + 60*E*Iyy)/(10*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + L**2*r13*(-11*A**2*G**2*L**4*intrho*scf**2 - 21*A**2*G**2*L**2*intrhoy2*scf**2 + 231*A*E*G*Izz*L**2*intrho*scf - 1260*A*E*G*Izz*intrhoy2*scf - 1260*E**2*Izz**2*intrho)/(210*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)) + L*intrhoz*r11*(-7*A*G*L**2*scf + 80*E*Izz)/(20*(A*G*L**2*scf - 12*E*Izz))) + r23*(A*G*L**2*intrhoyz*r13*scf*(A*G*L**2*scf + 60*E*Izz)/(10*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + L**2*r12*(-11*A**2*G**2*L**4*intrho*scf**2 - 21*A**2*G**2*L**2*intrhoz2*scf**2 + 231*A*E*G*Iyy*L**2*intrho*scf - 1260*A*E*G*Iyy*intrhoz2*scf - 1260*E**2*Iyy**2*intrho)/(210*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)) + L*intrhoy*r11*(7*A*G*L**2*scf - 80*E*Iyy)/(20*(A*G*L**2*scf - 12*E*Iyy)))
                k += 1
                Mv[k] += r31*(L*intrhoy*r13*(-A*G*L**2*scf + 48*E*Izz)/(12*(A*G*L**2*scf - 12*E*Izz)) + L*intrhoz*r12*(A*G*L**2*scf - 48*E*Iyy)/(12*(A*G*L**2*scf - 12*E*Iyy))) + r32*(A*G*L**2*intrhoyz*r12*scf*(A*G*L**2*scf + 60*E*Iyy)/(10*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + L**2*r13*(-11*A**2*G**2*L**4*intrho*scf**2 - 21*A**2*G**2*L**2*intrhoy2*scf**2 + 231*A*E*G*Izz*L**2*intrho*scf - 1260*A*E*G*Izz*intrhoy2*scf - 1260*E**2*Izz**2*intrho)/(210*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)) + L*intrhoz*r11*(-7*A*G*L**2*scf + 80*E*Izz)/(20*(A*G*L**2*scf - 12*E*Izz))) + r33*(A*G*L**2*intrhoyz*r13*scf*(A*G*L**2*scf + 60*E*Izz)/(10*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + L**2*r12*(-11*A**2*G**2*L**4*intrho*scf**2 - 21*A**2*G**2*L**2*intrhoz2*scf**2 + 231*A*E*G*Iyy*L**2*intrho*scf - 1260*A*E*G*Iyy*intrhoz2*scf - 1260*E**2*Iyy**2*intrho)/(210*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)) + L*intrhoy*r11*(7*A*G*L**2*scf - 80*E*Iyy)/(20*(A*G*L**2*scf - 12*E*Iyy)))
                k += 1
                Mv[k] += r11*(L**2*intrhoy*r12*(-A*G*L**2*scf + 10*E*Iyy)/(20*(A*G*L**2*scf - 12*E*Iyy)) + L**2*intrhoz*r13*(A*G*L**2*scf - 10*E*Izz)/(20*(A*G*L**2*scf - 12*E*Izz)) + L*r11*(intrhoy2 + intrhoz2)/3) + r12*(L**2*intrhoy*r11*(-A*G*L**2*scf + 10*E*Iyy)/(20*(A*G*L**2*scf - 12*E*Iyy)) + L*intrhoyz*r13*(-2*A**2*G**2*L**4*scf**2 + 15*A*E*G*Iyy*L**2*scf + 15*A*E*G*Izz*L**2*scf - 720*E**2*Iyy*Izz)/(15*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + L*r12*(A**2*G**2*L**6*intrho*scf**2 + 14*A**2*G**2*L**4*intrhoz2*scf**2 - 21*A*E*G*Iyy*L**4*intrho*scf - 210*A*E*G*Iyy*L**2*intrhoz2*scf + 126*E**2*Iyy**2*L**2*intrho + 5040*E**2*Iyy**2*intrhoz2)/(105*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2))) + r13*(L**2*intrhoz*r11*(A*G*L**2*scf - 10*E*Izz)/(20*(A*G*L**2*scf - 12*E*Izz)) + L*intrhoyz*r12*(-2*A**2*G**2*L**4*scf**2 + 15*A*E*G*Iyy*L**2*scf + 15*A*E*G*Izz*L**2*scf - 720*E**2*Iyy*Izz)/(15*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + L*r13*(A**2*G**2*L**6*intrho*scf**2 + 14*A**2*G**2*L**4*intrhoy2*scf**2 - 21*A*E*G*Izz*L**4*intrho*scf - 210*A*E*G*Izz*L**2*intrhoy2*scf + 126*E**2*Izz**2*L**2*intrho + 5040*E**2*Izz**2*intrhoy2)/(105*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)))
                k += 1
                Mv[k] += r21*(L**2*intrhoy*r12*(-A*G*L**2*scf + 10*E*Iyy)/(20*(A*G*L**2*scf - 12*E*Iyy)) + L**2*intrhoz*r13*(A*G*L**2*scf - 10*E*Izz)/(20*(A*G*L**2*scf - 12*E*Izz)) + L*r11*(intrhoy2 + intrhoz2)/3) + r22*(L**2*intrhoy*r11*(-A*G*L**2*scf + 10*E*Iyy)/(20*(A*G*L**2*scf - 12*E*Iyy)) + L*intrhoyz*r13*(-2*A**2*G**2*L**4*scf**2 + 15*A*E*G*Iyy*L**2*scf + 15*A*E*G*Izz*L**2*scf - 720*E**2*Iyy*Izz)/(15*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + L*r12*(A**2*G**2*L**6*intrho*scf**2 + 14*A**2*G**2*L**4*intrhoz2*scf**2 - 21*A*E*G*Iyy*L**4*intrho*scf - 210*A*E*G*Iyy*L**2*intrhoz2*scf + 126*E**2*Iyy**2*L**2*intrho + 5040*E**2*Iyy**2*intrhoz2)/(105*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2))) + r23*(L**2*intrhoz*r11*(A*G*L**2*scf - 10*E*Izz)/(20*(A*G*L**2*scf - 12*E*Izz)) + L*intrhoyz*r12*(-2*A**2*G**2*L**4*scf**2 + 15*A*E*G*Iyy*L**2*scf + 15*A*E*G*Izz*L**2*scf - 720*E**2*Iyy*Izz)/(15*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + L*r13*(A**2*G**2*L**6*intrho*scf**2 + 14*A**2*G**2*L**4*intrhoy2*scf**2 - 21*A*E*G*Izz*L**4*intrho*scf - 210*A*E*G*Izz*L**2*intrhoy2*scf + 126*E**2*Izz**2*L**2*intrho + 5040*E**2*Izz**2*intrhoy2)/(105*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)))
                k += 1
                Mv[k] += r31*(L**2*intrhoy*r12*(-A*G*L**2*scf + 10*E*Iyy)/(20*(A*G*L**2*scf - 12*E*Iyy)) + L**2*intrhoz*r13*(A*G*L**2*scf - 10*E*Izz)/(20*(A*G*L**2*scf - 12*E*Izz)) + L*r11*(intrhoy2 + intrhoz2)/3) + r32*(L**2*intrhoy*r11*(-A*G*L**2*scf + 10*E*Iyy)/(20*(A*G*L**2*scf - 12*E*Iyy)) + L*intrhoyz*r13*(-2*A**2*G**2*L**4*scf**2 + 15*A*E*G*Iyy*L**2*scf + 15*A*E*G*Izz*L**2*scf - 720*E**2*Iyy*Izz)/(15*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + L*r12*(A**2*G**2*L**6*intrho*scf**2 + 14*A**2*G**2*L**4*intrhoz2*scf**2 - 21*A*E*G*Iyy*L**4*intrho*scf - 210*A*E*G*Iyy*L**2*intrhoz2*scf + 126*E**2*Iyy**2*L**2*intrho + 5040*E**2*Iyy**2*intrhoz2)/(105*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2))) + r33*(L**2*intrhoz*r11*(A*G*L**2*scf - 10*E*Izz)/(20*(A*G*L**2*scf - 12*E*Izz)) + L*intrhoyz*r12*(-2*A**2*G**2*L**4*scf**2 + 15*A*E*G*Iyy*L**2*scf + 15*A*E*G*Izz*L**2*scf - 720*E**2*Iyy*Izz)/(15*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + L*r13*(A**2*G**2*L**6*intrho*scf**2 + 14*A**2*G**2*L**4*intrhoy2*scf**2 - 21*A*E*G*Izz*L**4*intrho*scf - 210*A*E*G*Izz*L**2*intrhoy2*scf + 126*E**2*Izz**2*L**2*intrho + 5040*E**2*Izz**2*intrhoy2)/(105*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)))
                k += 1
                Mv[k] += r11*(L*intrhoy*r23*(A*G*L**2*scf + 24*E*Izz)/(12*(A*G*L**2*scf - 12*E*Izz)) - L*intrhoz*r22*(A*G*L**2*scf + 24*E*Iyy)/(12*A*G*L**2*scf - 144*E*Iyy)) + r12*(-A*G*L**2*intrhoyz*r22*scf*(A*G*L**2*scf + 60*E*Iyy)/(10*A**2*G**2*L**4*scf**2 - 120*A*E*G*Iyy*L**2*scf - 120*A*E*G*Izz*L**2*scf + 1440*E**2*Iyy*Izz) + L**2*r23*(-13*A**2*G**2*L**4*intrho*scf**2 + 42*A**2*G**2*L**2*intrhoy2*scf**2 + 378*A*E*G*Izz*L**2*intrho*scf + 2520*A*E*G*Izz*intrhoy2*scf - 2520*E**2*Izz**2*intrho)/(420*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)) + L*intrhoz*r21*(-3*A*G*L**2*scf + 40*E*Izz)/(20*(A*G*L**2*scf - 12*E*Izz))) + r13*(-A*G*L**2*intrhoyz*r23*scf*(A*G*L**2*scf + 60*E*Izz)/(10*A**2*G**2*L**4*scf**2 - 120*A*E*G*Iyy*L**2*scf - 120*A*E*G*Izz*L**2*scf + 1440*E**2*Iyy*Izz) + L**2*r22*(-13*A**2*G**2*L**4*intrho*scf**2 + 42*A**2*G**2*L**2*intrhoz2*scf**2 + 378*A*E*G*Iyy*L**2*intrho*scf + 2520*A*E*G*Iyy*intrhoz2*scf - 2520*E**2*Iyy**2*intrho)/(420*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)) + L*intrhoy*r21*(3*A*G*L**2*scf - 40*E*Iyy)/(20*(A*G*L**2*scf - 12*E*Iyy)))
                k += 1
                Mv[k] += r21*(L*intrhoy*r23*(A*G*L**2*scf + 24*E*Izz)/(12*(A*G*L**2*scf - 12*E*Izz)) - L*intrhoz*r22*(A*G*L**2*scf + 24*E*Iyy)/(12*A*G*L**2*scf - 144*E*Iyy)) + r22*(-A*G*L**2*intrhoyz*r22*scf*(A*G*L**2*scf + 60*E*Iyy)/(10*A**2*G**2*L**4*scf**2 - 120*A*E*G*Iyy*L**2*scf - 120*A*E*G*Izz*L**2*scf + 1440*E**2*Iyy*Izz) + L**2*r23*(-13*A**2*G**2*L**4*intrho*scf**2 + 42*A**2*G**2*L**2*intrhoy2*scf**2 + 378*A*E*G*Izz*L**2*intrho*scf + 2520*A*E*G*Izz*intrhoy2*scf - 2520*E**2*Izz**2*intrho)/(420*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)) + L*intrhoz*r21*(-3*A*G*L**2*scf + 40*E*Izz)/(20*(A*G*L**2*scf - 12*E*Izz))) + r23*(-A*G*L**2*intrhoyz*r23*scf*(A*G*L**2*scf + 60*E*Izz)/(10*A**2*G**2*L**4*scf**2 - 120*A*E*G*Iyy*L**2*scf - 120*A*E*G*Izz*L**2*scf + 1440*E**2*Iyy*Izz) + L**2*r22*(-13*A**2*G**2*L**4*intrho*scf**2 + 42*A**2*G**2*L**2*intrhoz2*scf**2 + 378*A*E*G*Iyy*L**2*intrho*scf + 2520*A*E*G*Iyy*intrhoz2*scf - 2520*E**2*Iyy**2*intrho)/(420*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)) + L*intrhoy*r21*(3*A*G*L**2*scf - 40*E*Iyy)/(20*(A*G*L**2*scf - 12*E*Iyy)))
                k += 1
                Mv[k] += r31*(L*intrhoy*r23*(A*G*L**2*scf + 24*E*Izz)/(12*(A*G*L**2*scf - 12*E*Izz)) - L*intrhoz*r22*(A*G*L**2*scf + 24*E*Iyy)/(12*A*G*L**2*scf - 144*E*Iyy)) + r32*(-A*G*L**2*intrhoyz*r22*scf*(A*G*L**2*scf + 60*E*Iyy)/(10*A**2*G**2*L**4*scf**2 - 120*A*E*G*Iyy*L**2*scf - 120*A*E*G*Izz*L**2*scf + 1440*E**2*Iyy*Izz) + L**2*r23*(-13*A**2*G**2*L**4*intrho*scf**2 + 42*A**2*G**2*L**2*intrhoy2*scf**2 + 378*A*E*G*Izz*L**2*intrho*scf + 2520*A*E*G*Izz*intrhoy2*scf - 2520*E**2*Izz**2*intrho)/(420*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)) + L*intrhoz*r21*(-3*A*G*L**2*scf + 40*E*Izz)/(20*(A*G*L**2*scf - 12*E*Izz))) + r33*(-A*G*L**2*intrhoyz*r23*scf*(A*G*L**2*scf + 60*E*Izz)/(10*A**2*G**2*L**4*scf**2 - 120*A*E*G*Iyy*L**2*scf - 120*A*E*G*Izz*L**2*scf + 1440*E**2*Iyy*Izz) + L**2*r22*(-13*A**2*G**2*L**4*intrho*scf**2 + 42*A**2*G**2*L**2*intrhoz2*scf**2 + 378*A*E*G*Iyy*L**2*intrho*scf + 2520*A*E*G*Iyy*intrhoz2*scf - 2520*E**2*Iyy**2*intrho)/(420*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)) + L*intrhoy*r21*(3*A*G*L**2*scf - 40*E*Iyy)/(20*(A*G*L**2*scf - 12*E*Iyy)))
                k += 1
                Mv[k] += r11*(L**2*intrhoy*r22*(-A*G*L**2*scf + 15*E*Iyy)/(30*(A*G*L**2*scf - 12*E*Iyy)) + L**2*intrhoz*r23*(A*G*L**2*scf - 15*E*Izz)/(30*(A*G*L**2*scf - 12*E*Izz)) + L*r21*(intrhoy2 + intrhoz2)/6) + r12*(L**2*intrhoy*r21*(A*G*L**2*scf - 15*E*Iyy)/(30*(A*G*L**2*scf - 12*E*Iyy)) + L*intrhoyz*r23*(A**2*G**2*L**4*scf**2 - 30*A*E*G*Iyy*L**2*scf - 30*A*E*G*Izz*L**2*scf - 720*E**2*Iyy*Izz)/(30*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + L*r22*(-3*A**2*G**2*L**6*intrho*scf**2 - 14*A**2*G**2*L**4*intrhoz2*scf**2 + 84*A*E*G*Iyy*L**4*intrho*scf + 840*A*E*G*Iyy*L**2*intrhoz2*scf - 504*E**2*Iyy**2*L**2*intrho + 10080*E**2*Iyy**2*intrhoz2)/(420*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2))) + r13*(L**2*intrhoz*r21*(-A*G*L**2*scf + 15*E*Izz)/(30*(A*G*L**2*scf - 12*E*Izz)) + L*intrhoyz*r22*(A**2*G**2*L**4*scf**2 - 30*A*E*G*Iyy*L**2*scf - 30*A*E*G*Izz*L**2*scf - 720*E**2*Iyy*Izz)/(30*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + L*r23*(-3*A**2*G**2*L**6*intrho*scf**2 - 14*A**2*G**2*L**4*intrhoy2*scf**2 + 84*A*E*G*Izz*L**4*intrho*scf + 840*A*E*G*Izz*L**2*intrhoy2*scf - 504*E**2*Izz**2*L**2*intrho + 10080*E**2*Izz**2*intrhoy2)/(420*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)))
                k += 1
                Mv[k] += r21*(L**2*intrhoy*r22*(-A*G*L**2*scf + 15*E*Iyy)/(30*(A*G*L**2*scf - 12*E*Iyy)) + L**2*intrhoz*r23*(A*G*L**2*scf - 15*E*Izz)/(30*(A*G*L**2*scf - 12*E*Izz)) + L*r21*(intrhoy2 + intrhoz2)/6) + r22*(L**2*intrhoy*r21*(A*G*L**2*scf - 15*E*Iyy)/(30*(A*G*L**2*scf - 12*E*Iyy)) + L*intrhoyz*r23*(A**2*G**2*L**4*scf**2 - 30*A*E*G*Iyy*L**2*scf - 30*A*E*G*Izz*L**2*scf - 720*E**2*Iyy*Izz)/(30*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + L*r22*(-3*A**2*G**2*L**6*intrho*scf**2 - 14*A**2*G**2*L**4*intrhoz2*scf**2 + 84*A*E*G*Iyy*L**4*intrho*scf + 840*A*E*G*Iyy*L**2*intrhoz2*scf - 504*E**2*Iyy**2*L**2*intrho + 10080*E**2*Iyy**2*intrhoz2)/(420*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2))) + r23*(L**2*intrhoz*r21*(-A*G*L**2*scf + 15*E*Izz)/(30*(A*G*L**2*scf - 12*E*Izz)) + L*intrhoyz*r22*(A**2*G**2*L**4*scf**2 - 30*A*E*G*Iyy*L**2*scf - 30*A*E*G*Izz*L**2*scf - 720*E**2*Iyy*Izz)/(30*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + L*r23*(-3*A**2*G**2*L**6*intrho*scf**2 - 14*A**2*G**2*L**4*intrhoy2*scf**2 + 84*A*E*G*Izz*L**4*intrho*scf + 840*A*E*G*Izz*L**2*intrhoy2*scf - 504*E**2*Izz**2*L**2*intrho + 10080*E**2*Izz**2*intrhoy2)/(420*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)))
                k += 1
                Mv[k] += r31*(L**2*intrhoy*r22*(-A*G*L**2*scf + 15*E*Iyy)/(30*(A*G*L**2*scf - 12*E*Iyy)) + L**2*intrhoz*r23*(A*G*L**2*scf - 15*E*Izz)/(30*(A*G*L**2*scf - 12*E*Izz)) + L*r21*(intrhoy2 + intrhoz2)/6) + r32*(L**2*intrhoy*r21*(A*G*L**2*scf - 15*E*Iyy)/(30*(A*G*L**2*scf - 12*E*Iyy)) + L*intrhoyz*r23*(A**2*G**2*L**4*scf**2 - 30*A*E*G*Iyy*L**2*scf - 30*A*E*G*Izz*L**2*scf - 720*E**2*Iyy*Izz)/(30*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + L*r22*(-3*A**2*G**2*L**6*intrho*scf**2 - 14*A**2*G**2*L**4*intrhoz2*scf**2 + 84*A*E*G*Iyy*L**4*intrho*scf + 840*A*E*G*Iyy*L**2*intrhoz2*scf - 504*E**2*Iyy**2*L**2*intrho + 10080*E**2*Iyy**2*intrhoz2)/(420*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2))) + r33*(L**2*intrhoz*r21*(-A*G*L**2*scf + 15*E*Izz)/(30*(A*G*L**2*scf - 12*E*Izz)) + L*intrhoyz*r22*(A**2*G**2*L**4*scf**2 - 30*A*E*G*Iyy*L**2*scf - 30*A*E*G*Izz*L**2*scf - 720*E**2*Iyy*Izz)/(30*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + L*r23*(-3*A**2*G**2*L**6*intrho*scf**2 - 14*A**2*G**2*L**4*intrhoy2*scf**2 + 84*A*E*G*Izz*L**4*intrho*scf + 840*A*E*G*Izz*L**2*intrhoy2*scf - 504*E**2*Izz**2*L**2*intrho + 10080*E**2*Izz**2*intrhoy2)/(420*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)))
                k += 1
                Mv[k] += r11*(L*intrhoy*r23*(-A*G*L**2*scf + 48*E*Izz)/(12*(A*G*L**2*scf - 12*E*Izz)) + L*intrhoz*r22*(A*G*L**2*scf - 48*E*Iyy)/(12*(A*G*L**2*scf - 12*E*Iyy))) + r12*(A*G*L**2*intrhoyz*r22*scf*(A*G*L**2*scf + 60*E*Iyy)/(10*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + L**2*r23*(-11*A**2*G**2*L**4*intrho*scf**2 - 21*A**2*G**2*L**2*intrhoy2*scf**2 + 231*A*E*G*Izz*L**2*intrho*scf - 1260*A*E*G*Izz*intrhoy2*scf - 1260*E**2*Izz**2*intrho)/(210*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)) + L*intrhoz*r21*(-7*A*G*L**2*scf + 80*E*Izz)/(20*(A*G*L**2*scf - 12*E*Izz))) + r13*(A*G*L**2*intrhoyz*r23*scf*(A*G*L**2*scf + 60*E*Izz)/(10*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + L**2*r22*(-11*A**2*G**2*L**4*intrho*scf**2 - 21*A**2*G**2*L**2*intrhoz2*scf**2 + 231*A*E*G*Iyy*L**2*intrho*scf - 1260*A*E*G*Iyy*intrhoz2*scf - 1260*E**2*Iyy**2*intrho)/(210*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)) + L*intrhoy*r21*(7*A*G*L**2*scf - 80*E*Iyy)/(20*(A*G*L**2*scf - 12*E*Iyy)))
                k += 1
                Mv[k] += r21*(L*intrhoy*r23*(-A*G*L**2*scf + 48*E*Izz)/(12*(A*G*L**2*scf - 12*E*Izz)) + L*intrhoz*r22*(A*G*L**2*scf - 48*E*Iyy)/(12*(A*G*L**2*scf - 12*E*Iyy))) + r22*(A*G*L**2*intrhoyz*r22*scf*(A*G*L**2*scf + 60*E*Iyy)/(10*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + L**2*r23*(-11*A**2*G**2*L**4*intrho*scf**2 - 21*A**2*G**2*L**2*intrhoy2*scf**2 + 231*A*E*G*Izz*L**2*intrho*scf - 1260*A*E*G*Izz*intrhoy2*scf - 1260*E**2*Izz**2*intrho)/(210*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)) + L*intrhoz*r21*(-7*A*G*L**2*scf + 80*E*Izz)/(20*(A*G*L**2*scf - 12*E*Izz))) + r23*(A*G*L**2*intrhoyz*r23*scf*(A*G*L**2*scf + 60*E*Izz)/(10*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + L**2*r22*(-11*A**2*G**2*L**4*intrho*scf**2 - 21*A**2*G**2*L**2*intrhoz2*scf**2 + 231*A*E*G*Iyy*L**2*intrho*scf - 1260*A*E*G*Iyy*intrhoz2*scf - 1260*E**2*Iyy**2*intrho)/(210*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)) + L*intrhoy*r21*(7*A*G*L**2*scf - 80*E*Iyy)/(20*(A*G*L**2*scf - 12*E*Iyy)))
                k += 1
                Mv[k] += r31*(L*intrhoy*r23*(-A*G*L**2*scf + 48*E*Izz)/(12*(A*G*L**2*scf - 12*E*Izz)) + L*intrhoz*r22*(A*G*L**2*scf - 48*E*Iyy)/(12*(A*G*L**2*scf - 12*E*Iyy))) + r32*(A*G*L**2*intrhoyz*r22*scf*(A*G*L**2*scf + 60*E*Iyy)/(10*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + L**2*r23*(-11*A**2*G**2*L**4*intrho*scf**2 - 21*A**2*G**2*L**2*intrhoy2*scf**2 + 231*A*E*G*Izz*L**2*intrho*scf - 1260*A*E*G*Izz*intrhoy2*scf - 1260*E**2*Izz**2*intrho)/(210*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)) + L*intrhoz*r21*(-7*A*G*L**2*scf + 80*E*Izz)/(20*(A*G*L**2*scf - 12*E*Izz))) + r33*(A*G*L**2*intrhoyz*r23*scf*(A*G*L**2*scf + 60*E*Izz)/(10*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + L**2*r22*(-11*A**2*G**2*L**4*intrho*scf**2 - 21*A**2*G**2*L**2*intrhoz2*scf**2 + 231*A*E*G*Iyy*L**2*intrho*scf - 1260*A*E*G*Iyy*intrhoz2*scf - 1260*E**2*Iyy**2*intrho)/(210*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)) + L*intrhoy*r21*(7*A*G*L**2*scf - 80*E*Iyy)/(20*(A*G*L**2*scf - 12*E*Iyy)))
                k += 1
                Mv[k] += r11*(L**2*intrhoy*r22*(-A*G*L**2*scf + 10*E*Iyy)/(20*(A*G*L**2*scf - 12*E*Iyy)) + L**2*intrhoz*r23*(A*G*L**2*scf - 10*E*Izz)/(20*(A*G*L**2*scf - 12*E*Izz)) + L*r21*(intrhoy2 + intrhoz2)/3) + r12*(L**2*intrhoy*r21*(-A*G*L**2*scf + 10*E*Iyy)/(20*(A*G*L**2*scf - 12*E*Iyy)) + L*intrhoyz*r23*(-2*A**2*G**2*L**4*scf**2 + 15*A*E*G*Iyy*L**2*scf + 15*A*E*G*Izz*L**2*scf - 720*E**2*Iyy*Izz)/(15*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + L*r22*(A**2*G**2*L**6*intrho*scf**2 + 14*A**2*G**2*L**4*intrhoz2*scf**2 - 21*A*E*G*Iyy*L**4*intrho*scf - 210*A*E*G*Iyy*L**2*intrhoz2*scf + 126*E**2*Iyy**2*L**2*intrho + 5040*E**2*Iyy**2*intrhoz2)/(105*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2))) + r13*(L**2*intrhoz*r21*(A*G*L**2*scf - 10*E*Izz)/(20*(A*G*L**2*scf - 12*E*Izz)) + L*intrhoyz*r22*(-2*A**2*G**2*L**4*scf**2 + 15*A*E*G*Iyy*L**2*scf + 15*A*E*G*Izz*L**2*scf - 720*E**2*Iyy*Izz)/(15*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + L*r23*(A**2*G**2*L**6*intrho*scf**2 + 14*A**2*G**2*L**4*intrhoy2*scf**2 - 21*A*E*G*Izz*L**4*intrho*scf - 210*A*E*G*Izz*L**2*intrhoy2*scf + 126*E**2*Izz**2*L**2*intrho + 5040*E**2*Izz**2*intrhoy2)/(105*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)))
                k += 1
                Mv[k] += r21*(L**2*intrhoy*r22*(-A*G*L**2*scf + 10*E*Iyy)/(20*(A*G*L**2*scf - 12*E*Iyy)) + L**2*intrhoz*r23*(A*G*L**2*scf - 10*E*Izz)/(20*(A*G*L**2*scf - 12*E*Izz)) + L*r21*(intrhoy2 + intrhoz2)/3) + r22*(L**2*intrhoy*r21*(-A*G*L**2*scf + 10*E*Iyy)/(20*(A*G*L**2*scf - 12*E*Iyy)) + L*intrhoyz*r23*(-2*A**2*G**2*L**4*scf**2 + 15*A*E*G*Iyy*L**2*scf + 15*A*E*G*Izz*L**2*scf - 720*E**2*Iyy*Izz)/(15*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + L*r22*(A**2*G**2*L**6*intrho*scf**2 + 14*A**2*G**2*L**4*intrhoz2*scf**2 - 21*A*E*G*Iyy*L**4*intrho*scf - 210*A*E*G*Iyy*L**2*intrhoz2*scf + 126*E**2*Iyy**2*L**2*intrho + 5040*E**2*Iyy**2*intrhoz2)/(105*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2))) + r23*(L**2*intrhoz*r21*(A*G*L**2*scf - 10*E*Izz)/(20*(A*G*L**2*scf - 12*E*Izz)) + L*intrhoyz*r22*(-2*A**2*G**2*L**4*scf**2 + 15*A*E*G*Iyy*L**2*scf + 15*A*E*G*Izz*L**2*scf - 720*E**2*Iyy*Izz)/(15*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + L*r23*(A**2*G**2*L**6*intrho*scf**2 + 14*A**2*G**2*L**4*intrhoy2*scf**2 - 21*A*E*G*Izz*L**4*intrho*scf - 210*A*E*G*Izz*L**2*intrhoy2*scf + 126*E**2*Izz**2*L**2*intrho + 5040*E**2*Izz**2*intrhoy2)/(105*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)))
                k += 1
                Mv[k] += r31*(L**2*intrhoy*r22*(-A*G*L**2*scf + 10*E*Iyy)/(20*(A*G*L**2*scf - 12*E*Iyy)) + L**2*intrhoz*r23*(A*G*L**2*scf - 10*E*Izz)/(20*(A*G*L**2*scf - 12*E*Izz)) + L*r21*(intrhoy2 + intrhoz2)/3) + r32*(L**2*intrhoy*r21*(-A*G*L**2*scf + 10*E*Iyy)/(20*(A*G*L**2*scf - 12*E*Iyy)) + L*intrhoyz*r23*(-2*A**2*G**2*L**4*scf**2 + 15*A*E*G*Iyy*L**2*scf + 15*A*E*G*Izz*L**2*scf - 720*E**2*Iyy*Izz)/(15*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + L*r22*(A**2*G**2*L**6*intrho*scf**2 + 14*A**2*G**2*L**4*intrhoz2*scf**2 - 21*A*E*G*Iyy*L**4*intrho*scf - 210*A*E*G*Iyy*L**2*intrhoz2*scf + 126*E**2*Iyy**2*L**2*intrho + 5040*E**2*Iyy**2*intrhoz2)/(105*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2))) + r33*(L**2*intrhoz*r21*(A*G*L**2*scf - 10*E*Izz)/(20*(A*G*L**2*scf - 12*E*Izz)) + L*intrhoyz*r22*(-2*A**2*G**2*L**4*scf**2 + 15*A*E*G*Iyy*L**2*scf + 15*A*E*G*Izz*L**2*scf - 720*E**2*Iyy*Izz)/(15*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + L*r23*(A**2*G**2*L**6*intrho*scf**2 + 14*A**2*G**2*L**4*intrhoy2*scf**2 - 21*A*E*G*Izz*L**4*intrho*scf - 210*A*E*G*Izz*L**2*intrhoy2*scf + 126*E**2*Izz**2*L**2*intrho + 5040*E**2*Izz**2*intrhoy2)/(105*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)))
                k += 1
                Mv[k] += r11*(L*intrhoy*r33*(A*G*L**2*scf + 24*E*Izz)/(12*(A*G*L**2*scf - 12*E*Izz)) - L*intrhoz*r32*(A*G*L**2*scf + 24*E*Iyy)/(12*A*G*L**2*scf - 144*E*Iyy)) + r12*(-A*G*L**2*intrhoyz*r32*scf*(A*G*L**2*scf + 60*E*Iyy)/(10*A**2*G**2*L**4*scf**2 - 120*A*E*G*Iyy*L**2*scf - 120*A*E*G*Izz*L**2*scf + 1440*E**2*Iyy*Izz) + L**2*r33*(-13*A**2*G**2*L**4*intrho*scf**2 + 42*A**2*G**2*L**2*intrhoy2*scf**2 + 378*A*E*G*Izz*L**2*intrho*scf + 2520*A*E*G*Izz*intrhoy2*scf - 2520*E**2*Izz**2*intrho)/(420*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)) + L*intrhoz*r31*(-3*A*G*L**2*scf + 40*E*Izz)/(20*(A*G*L**2*scf - 12*E*Izz))) + r13*(-A*G*L**2*intrhoyz*r33*scf*(A*G*L**2*scf + 60*E*Izz)/(10*A**2*G**2*L**4*scf**2 - 120*A*E*G*Iyy*L**2*scf - 120*A*E*G*Izz*L**2*scf + 1440*E**2*Iyy*Izz) + L**2*r32*(-13*A**2*G**2*L**4*intrho*scf**2 + 42*A**2*G**2*L**2*intrhoz2*scf**2 + 378*A*E*G*Iyy*L**2*intrho*scf + 2520*A*E*G*Iyy*intrhoz2*scf - 2520*E**2*Iyy**2*intrho)/(420*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)) + L*intrhoy*r31*(3*A*G*L**2*scf - 40*E*Iyy)/(20*(A*G*L**2*scf - 12*E*Iyy)))
                k += 1
                Mv[k] += r21*(L*intrhoy*r33*(A*G*L**2*scf + 24*E*Izz)/(12*(A*G*L**2*scf - 12*E*Izz)) - L*intrhoz*r32*(A*G*L**2*scf + 24*E*Iyy)/(12*A*G*L**2*scf - 144*E*Iyy)) + r22*(-A*G*L**2*intrhoyz*r32*scf*(A*G*L**2*scf + 60*E*Iyy)/(10*A**2*G**2*L**4*scf**2 - 120*A*E*G*Iyy*L**2*scf - 120*A*E*G*Izz*L**2*scf + 1440*E**2*Iyy*Izz) + L**2*r33*(-13*A**2*G**2*L**4*intrho*scf**2 + 42*A**2*G**2*L**2*intrhoy2*scf**2 + 378*A*E*G*Izz*L**2*intrho*scf + 2520*A*E*G*Izz*intrhoy2*scf - 2520*E**2*Izz**2*intrho)/(420*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)) + L*intrhoz*r31*(-3*A*G*L**2*scf + 40*E*Izz)/(20*(A*G*L**2*scf - 12*E*Izz))) + r23*(-A*G*L**2*intrhoyz*r33*scf*(A*G*L**2*scf + 60*E*Izz)/(10*A**2*G**2*L**4*scf**2 - 120*A*E*G*Iyy*L**2*scf - 120*A*E*G*Izz*L**2*scf + 1440*E**2*Iyy*Izz) + L**2*r32*(-13*A**2*G**2*L**4*intrho*scf**2 + 42*A**2*G**2*L**2*intrhoz2*scf**2 + 378*A*E*G*Iyy*L**2*intrho*scf + 2520*A*E*G*Iyy*intrhoz2*scf - 2520*E**2*Iyy**2*intrho)/(420*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)) + L*intrhoy*r31*(3*A*G*L**2*scf - 40*E*Iyy)/(20*(A*G*L**2*scf - 12*E*Iyy)))
                k += 1
                Mv[k] += r31*(L*intrhoy*r33*(A*G*L**2*scf + 24*E*Izz)/(12*(A*G*L**2*scf - 12*E*Izz)) - L*intrhoz*r32*(A*G*L**2*scf + 24*E*Iyy)/(12*A*G*L**2*scf - 144*E*Iyy)) + r32*(-A*G*L**2*intrhoyz*r32*scf*(A*G*L**2*scf + 60*E*Iyy)/(10*A**2*G**2*L**4*scf**2 - 120*A*E*G*Iyy*L**2*scf - 120*A*E*G*Izz*L**2*scf + 1440*E**2*Iyy*Izz) + L**2*r33*(-13*A**2*G**2*L**4*intrho*scf**2 + 42*A**2*G**2*L**2*intrhoy2*scf**2 + 378*A*E*G*Izz*L**2*intrho*scf + 2520*A*E*G*Izz*intrhoy2*scf - 2520*E**2*Izz**2*intrho)/(420*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)) + L*intrhoz*r31*(-3*A*G*L**2*scf + 40*E*Izz)/(20*(A*G*L**2*scf - 12*E*Izz))) + r33*(-A*G*L**2*intrhoyz*r33*scf*(A*G*L**2*scf + 60*E*Izz)/(10*A**2*G**2*L**4*scf**2 - 120*A*E*G*Iyy*L**2*scf - 120*A*E*G*Izz*L**2*scf + 1440*E**2*Iyy*Izz) + L**2*r32*(-13*A**2*G**2*L**4*intrho*scf**2 + 42*A**2*G**2*L**2*intrhoz2*scf**2 + 378*A*E*G*Iyy*L**2*intrho*scf + 2520*A*E*G*Iyy*intrhoz2*scf - 2520*E**2*Iyy**2*intrho)/(420*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)) + L*intrhoy*r31*(3*A*G*L**2*scf - 40*E*Iyy)/(20*(A*G*L**2*scf - 12*E*Iyy)))
                k += 1
                Mv[k] += r11*(L**2*intrhoy*r32*(-A*G*L**2*scf + 15*E*Iyy)/(30*(A*G*L**2*scf - 12*E*Iyy)) + L**2*intrhoz*r33*(A*G*L**2*scf - 15*E*Izz)/(30*(A*G*L**2*scf - 12*E*Izz)) + L*r31*(intrhoy2 + intrhoz2)/6) + r12*(L**2*intrhoy*r31*(A*G*L**2*scf - 15*E*Iyy)/(30*(A*G*L**2*scf - 12*E*Iyy)) + L*intrhoyz*r33*(A**2*G**2*L**4*scf**2 - 30*A*E*G*Iyy*L**2*scf - 30*A*E*G*Izz*L**2*scf - 720*E**2*Iyy*Izz)/(30*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + L*r32*(-3*A**2*G**2*L**6*intrho*scf**2 - 14*A**2*G**2*L**4*intrhoz2*scf**2 + 84*A*E*G*Iyy*L**4*intrho*scf + 840*A*E*G*Iyy*L**2*intrhoz2*scf - 504*E**2*Iyy**2*L**2*intrho + 10080*E**2*Iyy**2*intrhoz2)/(420*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2))) + r13*(L**2*intrhoz*r31*(-A*G*L**2*scf + 15*E*Izz)/(30*(A*G*L**2*scf - 12*E*Izz)) + L*intrhoyz*r32*(A**2*G**2*L**4*scf**2 - 30*A*E*G*Iyy*L**2*scf - 30*A*E*G*Izz*L**2*scf - 720*E**2*Iyy*Izz)/(30*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + L*r33*(-3*A**2*G**2*L**6*intrho*scf**2 - 14*A**2*G**2*L**4*intrhoy2*scf**2 + 84*A*E*G*Izz*L**4*intrho*scf + 840*A*E*G*Izz*L**2*intrhoy2*scf - 504*E**2*Izz**2*L**2*intrho + 10080*E**2*Izz**2*intrhoy2)/(420*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)))
                k += 1
                Mv[k] += r21*(L**2*intrhoy*r32*(-A*G*L**2*scf + 15*E*Iyy)/(30*(A*G*L**2*scf - 12*E*Iyy)) + L**2*intrhoz*r33*(A*G*L**2*scf - 15*E*Izz)/(30*(A*G*L**2*scf - 12*E*Izz)) + L*r31*(intrhoy2 + intrhoz2)/6) + r22*(L**2*intrhoy*r31*(A*G*L**2*scf - 15*E*Iyy)/(30*(A*G*L**2*scf - 12*E*Iyy)) + L*intrhoyz*r33*(A**2*G**2*L**4*scf**2 - 30*A*E*G*Iyy*L**2*scf - 30*A*E*G*Izz*L**2*scf - 720*E**2*Iyy*Izz)/(30*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + L*r32*(-3*A**2*G**2*L**6*intrho*scf**2 - 14*A**2*G**2*L**4*intrhoz2*scf**2 + 84*A*E*G*Iyy*L**4*intrho*scf + 840*A*E*G*Iyy*L**2*intrhoz2*scf - 504*E**2*Iyy**2*L**2*intrho + 10080*E**2*Iyy**2*intrhoz2)/(420*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2))) + r23*(L**2*intrhoz*r31*(-A*G*L**2*scf + 15*E*Izz)/(30*(A*G*L**2*scf - 12*E*Izz)) + L*intrhoyz*r32*(A**2*G**2*L**4*scf**2 - 30*A*E*G*Iyy*L**2*scf - 30*A*E*G*Izz*L**2*scf - 720*E**2*Iyy*Izz)/(30*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + L*r33*(-3*A**2*G**2*L**6*intrho*scf**2 - 14*A**2*G**2*L**4*intrhoy2*scf**2 + 84*A*E*G*Izz*L**4*intrho*scf + 840*A*E*G*Izz*L**2*intrhoy2*scf - 504*E**2*Izz**2*L**2*intrho + 10080*E**2*Izz**2*intrhoy2)/(420*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)))
                k += 1
                Mv[k] += r31*(L**2*intrhoy*r32*(-A*G*L**2*scf + 15*E*Iyy)/(30*(A*G*L**2*scf - 12*E*Iyy)) + L**2*intrhoz*r33*(A*G*L**2*scf - 15*E*Izz)/(30*(A*G*L**2*scf - 12*E*Izz)) + L*r31*(intrhoy2 + intrhoz2)/6) + r32*(L**2*intrhoy*r31*(A*G*L**2*scf - 15*E*Iyy)/(30*(A*G*L**2*scf - 12*E*Iyy)) + L*intrhoyz*r33*(A**2*G**2*L**4*scf**2 - 30*A*E*G*Iyy*L**2*scf - 30*A*E*G*Izz*L**2*scf - 720*E**2*Iyy*Izz)/(30*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + L*r32*(-3*A**2*G**2*L**6*intrho*scf**2 - 14*A**2*G**2*L**4*intrhoz2*scf**2 + 84*A*E*G*Iyy*L**4*intrho*scf + 840*A*E*G*Iyy*L**2*intrhoz2*scf - 504*E**2*Iyy**2*L**2*intrho + 10080*E**2*Iyy**2*intrhoz2)/(420*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2))) + r33*(L**2*intrhoz*r31*(-A*G*L**2*scf + 15*E*Izz)/(30*(A*G*L**2*scf - 12*E*Izz)) + L*intrhoyz*r32*(A**2*G**2*L**4*scf**2 - 30*A*E*G*Iyy*L**2*scf - 30*A*E*G*Izz*L**2*scf - 720*E**2*Iyy*Izz)/(30*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + L*r33*(-3*A**2*G**2*L**6*intrho*scf**2 - 14*A**2*G**2*L**4*intrhoy2*scf**2 + 84*A*E*G*Izz*L**4*intrho*scf + 840*A*E*G*Izz*L**2*intrhoy2*scf - 504*E**2*Izz**2*L**2*intrho + 10080*E**2*Izz**2*intrhoy2)/(420*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)))
                k += 1
                Mv[k] += r11*(L*intrhoy*r33*(-A*G*L**2*scf + 48*E*Izz)/(12*(A*G*L**2*scf - 12*E*Izz)) + L*intrhoz*r32*(A*G*L**2*scf - 48*E*Iyy)/(12*(A*G*L**2*scf - 12*E*Iyy))) + r12*(A*G*L**2*intrhoyz*r32*scf*(A*G*L**2*scf + 60*E*Iyy)/(10*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + L**2*r33*(-11*A**2*G**2*L**4*intrho*scf**2 - 21*A**2*G**2*L**2*intrhoy2*scf**2 + 231*A*E*G*Izz*L**2*intrho*scf - 1260*A*E*G*Izz*intrhoy2*scf - 1260*E**2*Izz**2*intrho)/(210*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)) + L*intrhoz*r31*(-7*A*G*L**2*scf + 80*E*Izz)/(20*(A*G*L**2*scf - 12*E*Izz))) + r13*(A*G*L**2*intrhoyz*r33*scf*(A*G*L**2*scf + 60*E*Izz)/(10*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + L**2*r32*(-11*A**2*G**2*L**4*intrho*scf**2 - 21*A**2*G**2*L**2*intrhoz2*scf**2 + 231*A*E*G*Iyy*L**2*intrho*scf - 1260*A*E*G*Iyy*intrhoz2*scf - 1260*E**2*Iyy**2*intrho)/(210*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)) + L*intrhoy*r31*(7*A*G*L**2*scf - 80*E*Iyy)/(20*(A*G*L**2*scf - 12*E*Iyy)))
                k += 1
                Mv[k] += r21*(L*intrhoy*r33*(-A*G*L**2*scf + 48*E*Izz)/(12*(A*G*L**2*scf - 12*E*Izz)) + L*intrhoz*r32*(A*G*L**2*scf - 48*E*Iyy)/(12*(A*G*L**2*scf - 12*E*Iyy))) + r22*(A*G*L**2*intrhoyz*r32*scf*(A*G*L**2*scf + 60*E*Iyy)/(10*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + L**2*r33*(-11*A**2*G**2*L**4*intrho*scf**2 - 21*A**2*G**2*L**2*intrhoy2*scf**2 + 231*A*E*G*Izz*L**2*intrho*scf - 1260*A*E*G*Izz*intrhoy2*scf - 1260*E**2*Izz**2*intrho)/(210*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)) + L*intrhoz*r31*(-7*A*G*L**2*scf + 80*E*Izz)/(20*(A*G*L**2*scf - 12*E*Izz))) + r23*(A*G*L**2*intrhoyz*r33*scf*(A*G*L**2*scf + 60*E*Izz)/(10*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + L**2*r32*(-11*A**2*G**2*L**4*intrho*scf**2 - 21*A**2*G**2*L**2*intrhoz2*scf**2 + 231*A*E*G*Iyy*L**2*intrho*scf - 1260*A*E*G*Iyy*intrhoz2*scf - 1260*E**2*Iyy**2*intrho)/(210*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)) + L*intrhoy*r31*(7*A*G*L**2*scf - 80*E*Iyy)/(20*(A*G*L**2*scf - 12*E*Iyy)))
                k += 1
                Mv[k] += r31*(L*intrhoy*r33*(-A*G*L**2*scf + 48*E*Izz)/(12*(A*G*L**2*scf - 12*E*Izz)) + L*intrhoz*r32*(A*G*L**2*scf - 48*E*Iyy)/(12*(A*G*L**2*scf - 12*E*Iyy))) + r32*(A*G*L**2*intrhoyz*r32*scf*(A*G*L**2*scf + 60*E*Iyy)/(10*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + L**2*r33*(-11*A**2*G**2*L**4*intrho*scf**2 - 21*A**2*G**2*L**2*intrhoy2*scf**2 + 231*A*E*G*Izz*L**2*intrho*scf - 1260*A*E*G*Izz*intrhoy2*scf - 1260*E**2*Izz**2*intrho)/(210*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)) + L*intrhoz*r31*(-7*A*G*L**2*scf + 80*E*Izz)/(20*(A*G*L**2*scf - 12*E*Izz))) + r33*(A*G*L**2*intrhoyz*r33*scf*(A*G*L**2*scf + 60*E*Izz)/(10*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + L**2*r32*(-11*A**2*G**2*L**4*intrho*scf**2 - 21*A**2*G**2*L**2*intrhoz2*scf**2 + 231*A*E*G*Iyy*L**2*intrho*scf - 1260*A*E*G*Iyy*intrhoz2*scf - 1260*E**2*Iyy**2*intrho)/(210*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2)) + L*intrhoy*r31*(7*A*G*L**2*scf - 80*E*Iyy)/(20*(A*G*L**2*scf - 12*E*Iyy)))
                k += 1
                Mv[k] += r11*(L**2*intrhoy*r32*(-A*G*L**2*scf + 10*E*Iyy)/(20*(A*G*L**2*scf - 12*E*Iyy)) + L**2*intrhoz*r33*(A*G*L**2*scf - 10*E*Izz)/(20*(A*G*L**2*scf - 12*E*Izz)) + L*r31*(intrhoy2 + intrhoz2)/3) + r12*(L**2*intrhoy*r31*(-A*G*L**2*scf + 10*E*Iyy)/(20*(A*G*L**2*scf - 12*E*Iyy)) + L*intrhoyz*r33*(-2*A**2*G**2*L**4*scf**2 + 15*A*E*G*Iyy*L**2*scf + 15*A*E*G*Izz*L**2*scf - 720*E**2*Iyy*Izz)/(15*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + L*r32*(A**2*G**2*L**6*intrho*scf**2 + 14*A**2*G**2*L**4*intrhoz2*scf**2 - 21*A*E*G*Iyy*L**4*intrho*scf - 210*A*E*G*Iyy*L**2*intrhoz2*scf + 126*E**2*Iyy**2*L**2*intrho + 5040*E**2*Iyy**2*intrhoz2)/(105*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2))) + r13*(L**2*intrhoz*r31*(A*G*L**2*scf - 10*E*Izz)/(20*(A*G*L**2*scf - 12*E*Izz)) + L*intrhoyz*r32*(-2*A**2*G**2*L**4*scf**2 + 15*A*E*G*Iyy*L**2*scf + 15*A*E*G*Izz*L**2*scf - 720*E**2*Iyy*Izz)/(15*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + L*r33*(A**2*G**2*L**6*intrho*scf**2 + 14*A**2*G**2*L**4*intrhoy2*scf**2 - 21*A*E*G*Izz*L**4*intrho*scf - 210*A*E*G*Izz*L**2*intrhoy2*scf + 126*E**2*Izz**2*L**2*intrho + 5040*E**2*Izz**2*intrhoy2)/(105*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)))
                k += 1
                Mv[k] += r21*(L**2*intrhoy*r32*(-A*G*L**2*scf + 10*E*Iyy)/(20*(A*G*L**2*scf - 12*E*Iyy)) + L**2*intrhoz*r33*(A*G*L**2*scf - 10*E*Izz)/(20*(A*G*L**2*scf - 12*E*Izz)) + L*r31*(intrhoy2 + intrhoz2)/3) + r22*(L**2*intrhoy*r31*(-A*G*L**2*scf + 10*E*Iyy)/(20*(A*G*L**2*scf - 12*E*Iyy)) + L*intrhoyz*r33*(-2*A**2*G**2*L**4*scf**2 + 15*A*E*G*Iyy*L**2*scf + 15*A*E*G*Izz*L**2*scf - 720*E**2*Iyy*Izz)/(15*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + L*r32*(A**2*G**2*L**6*intrho*scf**2 + 14*A**2*G**2*L**4*intrhoz2*scf**2 - 21*A*E*G*Iyy*L**4*intrho*scf - 210*A*E*G*Iyy*L**2*intrhoz2*scf + 126*E**2*Iyy**2*L**2*intrho + 5040*E**2*Iyy**2*intrhoz2)/(105*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2))) + r23*(L**2*intrhoz*r31*(A*G*L**2*scf - 10*E*Izz)/(20*(A*G*L**2*scf - 12*E*Izz)) + L*intrhoyz*r32*(-2*A**2*G**2*L**4*scf**2 + 15*A*E*G*Iyy*L**2*scf + 15*A*E*G*Izz*L**2*scf - 720*E**2*Iyy*Izz)/(15*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + L*r33*(A**2*G**2*L**6*intrho*scf**2 + 14*A**2*G**2*L**4*intrhoy2*scf**2 - 21*A*E*G*Izz*L**4*intrho*scf - 210*A*E*G*Izz*L**2*intrhoy2*scf + 126*E**2*Izz**2*L**2*intrho + 5040*E**2*Izz**2*intrhoy2)/(105*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)))
                k += 1
                Mv[k] += r31*(L**2*intrhoy*r32*(-A*G*L**2*scf + 10*E*Iyy)/(20*(A*G*L**2*scf - 12*E*Iyy)) + L**2*intrhoz*r33*(A*G*L**2*scf - 10*E*Izz)/(20*(A*G*L**2*scf - 12*E*Izz)) + L*r31*(intrhoy2 + intrhoz2)/3) + r32*(L**2*intrhoy*r31*(-A*G*L**2*scf + 10*E*Iyy)/(20*(A*G*L**2*scf - 12*E*Iyy)) + L*intrhoyz*r33*(-2*A**2*G**2*L**4*scf**2 + 15*A*E*G*Iyy*L**2*scf + 15*A*E*G*Izz*L**2*scf - 720*E**2*Iyy*Izz)/(15*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + L*r32*(A**2*G**2*L**6*intrho*scf**2 + 14*A**2*G**2*L**4*intrhoz2*scf**2 - 21*A*E*G*Iyy*L**4*intrho*scf - 210*A*E*G*Iyy*L**2*intrhoz2*scf + 126*E**2*Iyy**2*L**2*intrho + 5040*E**2*Iyy**2*intrhoz2)/(105*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Iyy*L**2*scf + 144*E**2*Iyy**2))) + r33*(L**2*intrhoz*r31*(A*G*L**2*scf - 10*E*Izz)/(20*(A*G*L**2*scf - 12*E*Izz)) + L*intrhoyz*r32*(-2*A**2*G**2*L**4*scf**2 + 15*A*E*G*Iyy*L**2*scf + 15*A*E*G*Izz*L**2*scf - 720*E**2*Iyy*Izz)/(15*(A**2*G**2*L**4*scf**2 - 12*A*E*G*Iyy*L**2*scf - 12*A*E*G*Izz*L**2*scf + 144*E**2*Iyy*Izz)) + L*r33*(A**2*G**2*L**6*intrho*scf**2 + 14*A**2*G**2*L**4*intrhoy2*scf**2 - 21*A*E*G*Izz*L**4*intrho*scf - 210*A*E*G*Izz*L**2*intrhoy2*scf + 126*E**2*Izz**2*L**2*intrho + 5040*E**2*Izz**2*intrhoy2)/(105*(A**2*G**2*L**4*scf**2 - 24*A*E*G*Izz*L**2*scf + 144*E**2*Izz**2)))

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

                #NOTE two-point Gauss-Lobatto quadrature

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
