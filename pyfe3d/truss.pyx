#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: nonecheck=False
#cython: infer_types=False
"""
Truss - Linear truss 3D element with analytical integration
-----------------------------------------------------------

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

cdef class TrussData:
    cdef public cINT KC0_SPARSE_SIZE
    cdef public cINT KG_SPARSE_SIZE
    cdef public cINT M_SPARSE_SIZE
    def __cinit__(TrussData self):
        self.KC0_SPARSE_SIZE = 72
        self.KG_SPARSE_SIZE = 36
        self.M_SPARSE_SIZE = 144

cdef class TrussProbe:
    """
    Probe used for local coordinates, local displacements, local stresses etc
    """
    cdef public cDOUBLE[:] xe
    cdef public cDOUBLE[:] ue
    def __cinit__(TrussProbe self):
        self.xe = np.zeros(NUM_NODES*DOF//2, dtype=DOUBLE)
        self.ue = np.zeros(NUM_NODES*DOF, dtype=DOUBLE)

cdef class Truss:
    """
       ^ y axis
       |
       |
       ______   --> x axis
       1    2
       Assumed 2D plane for all derivations

       Timoshenko 3D beam element with consistent shape functions from:
       Luo, Y., 2008, “An Efficient 3D Timoshenko Beam Element with Consistent Shape Functions,” Adv. Theor. Appl. Mech., 1(3), pp. 95–106.


    """
    cdef public cINT eid
    cdef public cINT n1, n2
    cdef public cINT c1, c2
    cdef public cINT init_k_KC0, init_k_KG, init_k_M
    cdef public double length
    cdef public double cosa, cosb, cosg
    cdef TrussProbe _p

    def __cinit__(Truss self, TrussProbe p):
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

    cpdef void update_ue(Truss self, np.ndarray[cDOUBLE, ndim=1] u):
        """Update the local displacement vector of the element

        Parameters
        ----------
        u : array-like
            Global displacement vector

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

    cpdef void update_xe(Truss self, np.ndarray[cDOUBLE, ndim=1] x):
        """Update the 3D coordinates of the element

        Parameters
        ----------
        x : array-like
            Array with global nodal coordinates x1, y1, z1, x2, y2, z2, ...

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

    cpdef void update_length(Truss self):
        """Update element length

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

    cpdef void update_KC0(Truss self,
            np.ndarray[cINT, ndim=1] KC0r,
            np.ndarray[cINT, ndim=1] KC0c,
            np.ndarray[cDOUBLE, ndim=1] KC0v,
            BeamProp prop,
            int update_KC0v_only=0
            ):
        """Update sparse vectors for linear constitutive stiffness matrix KC0

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
        cdef double L, A, E, G, scf, Iyy, Izz
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
            KC0v[k] += 1.0*A*E*r11**2/L
            k += 1
            KC0v[k] += 1.0*A*E*r11*r21/L
            k += 1
            KC0v[k] += 1.0*A*E*r11*r31/L
            k += 1
            KC0v[k] += -1.0*A*E*r11**2/L
            k += 1
            KC0v[k] += -1.0*A*E*r11*r21/L
            k += 1
            KC0v[k] += -1.0*A*E*r11*r31/L
            k += 1
            KC0v[k] += 1.0*A*E*r11*r21/L
            k += 1
            KC0v[k] += 1.0*A*E*r21**2/L
            k += 1
            KC0v[k] += 1.0*A*E*r21*r31/L
            k += 1
            KC0v[k] += -1.0*A*E*r11*r21/L
            k += 1
            KC0v[k] += -1.0*A*E*r21**2/L
            k += 1
            KC0v[k] += -1.0*A*E*r21*r31/L
            k += 1
            KC0v[k] += 1.0*A*E*r11*r31/L
            k += 1
            KC0v[k] += 1.0*A*E*r21*r31/L
            k += 1
            KC0v[k] += 1.0*A*E*r31**2/L
            k += 1
            KC0v[k] += -1.0*A*E*r11*r31/L
            k += 1
            KC0v[k] += -1.0*A*E*r21*r31/L
            k += 1
            KC0v[k] += -1.0*A*E*r31**2/L
            k += 1
            KC0v[k] += -1.0*G*r11**2*scf*(Iyy + Izz)/L
            k += 1
            KC0v[k] += -1.0*G*r11*r21*scf*(Iyy + Izz)/L
            k += 1
            KC0v[k] += -1.0*G*r11*r31*scf*(Iyy + Izz)/L
            k += 1
            KC0v[k] += 1.0*G*r11**2*scf*(Iyy + Izz)/L
            k += 1
            KC0v[k] += 1.0*G*r11*r21*scf*(Iyy + Izz)/L
            k += 1
            KC0v[k] += 1.0*G*r11*r31*scf*(Iyy + Izz)/L
            k += 1
            KC0v[k] += -1.0*G*r11*r21*scf*(Iyy + Izz)/L
            k += 1
            KC0v[k] += -1.0*G*r21**2*scf*(Iyy + Izz)/L
            k += 1
            KC0v[k] += -1.0*G*r21*r31*scf*(Iyy + Izz)/L
            k += 1
            KC0v[k] += 1.0*G*r11*r21*scf*(Iyy + Izz)/L
            k += 1
            KC0v[k] += 1.0*G*r21**2*scf*(Iyy + Izz)/L
            k += 1
            KC0v[k] += 1.0*G*r21*r31*scf*(Iyy + Izz)/L
            k += 1
            KC0v[k] += -1.0*G*r11*r31*scf*(Iyy + Izz)/L
            k += 1
            KC0v[k] += -1.0*G*r21*r31*scf*(Iyy + Izz)/L
            k += 1
            KC0v[k] += -1.0*G*r31**2*scf*(Iyy + Izz)/L
            k += 1
            KC0v[k] += 1.0*G*r11*r31*scf*(Iyy + Izz)/L
            k += 1
            KC0v[k] += 1.0*G*r21*r31*scf*(Iyy + Izz)/L
            k += 1
            KC0v[k] += 1.0*G*r31**2*scf*(Iyy + Izz)/L
            k += 1
            KC0v[k] += -1.0*A*E*r11**2/L
            k += 1
            KC0v[k] += -1.0*A*E*r11*r21/L
            k += 1
            KC0v[k] += -1.0*A*E*r11*r31/L
            k += 1
            KC0v[k] += 1.0*A*E*r11**2/L
            k += 1
            KC0v[k] += 1.0*A*E*r11*r21/L
            k += 1
            KC0v[k] += 1.0*A*E*r11*r31/L
            k += 1
            KC0v[k] += -1.0*A*E*r11*r21/L
            k += 1
            KC0v[k] += -1.0*A*E*r21**2/L
            k += 1
            KC0v[k] += -1.0*A*E*r21*r31/L
            k += 1
            KC0v[k] += 1.0*A*E*r11*r21/L
            k += 1
            KC0v[k] += 1.0*A*E*r21**2/L
            k += 1
            KC0v[k] += 1.0*A*E*r21*r31/L
            k += 1
            KC0v[k] += -1.0*A*E*r11*r31/L
            k += 1
            KC0v[k] += -1.0*A*E*r21*r31/L
            k += 1
            KC0v[k] += -1.0*A*E*r31**2/L
            k += 1
            KC0v[k] += 1.0*A*E*r11*r31/L
            k += 1
            KC0v[k] += 1.0*A*E*r21*r31/L
            k += 1
            KC0v[k] += 1.0*A*E*r31**2/L
            k += 1
            KC0v[k] += 1.0*G*r11**2*scf*(Iyy + Izz)/L
            k += 1
            KC0v[k] += 1.0*G*r11*r21*scf*(Iyy + Izz)/L
            k += 1
            KC0v[k] += 1.0*G*r11*r31*scf*(Iyy + Izz)/L
            k += 1
            KC0v[k] += -1.0*G*r11**2*scf*(Iyy + Izz)/L
            k += 1
            KC0v[k] += -1.0*G*r11*r21*scf*(Iyy + Izz)/L
            k += 1
            KC0v[k] += -1.0*G*r11*r31*scf*(Iyy + Izz)/L
            k += 1
            KC0v[k] += 1.0*G*r11*r21*scf*(Iyy + Izz)/L
            k += 1
            KC0v[k] += 1.0*G*r21**2*scf*(Iyy + Izz)/L
            k += 1
            KC0v[k] += 1.0*G*r21*r31*scf*(Iyy + Izz)/L
            k += 1
            KC0v[k] += -1.0*G*r11*r21*scf*(Iyy + Izz)/L
            k += 1
            KC0v[k] += -1.0*G*r21**2*scf*(Iyy + Izz)/L
            k += 1
            KC0v[k] += -1.0*G*r21*r31*scf*(Iyy + Izz)/L
            k += 1
            KC0v[k] += 1.0*G*r11*r31*scf*(Iyy + Izz)/L
            k += 1
            KC0v[k] += 1.0*G*r21*r31*scf*(Iyy + Izz)/L
            k += 1
            KC0v[k] += 1.0*G*r31**2*scf*(Iyy + Izz)/L
            k += 1
            KC0v[k] += -1.0*G*r11*r31*scf*(Iyy + Izz)/L
            k += 1
            KC0v[k] += -1.0*G*r21*r31*scf*(Iyy + Izz)/L
            k += 1
            KC0v[k] += -1.0*G*r31**2*scf*(Iyy + Izz)/L


    cpdef void update_KG(Truss self,
            np.ndarray[cINT, ndim=1] KGr,
            np.ndarray[cINT, ndim=1] KGc,
            np.ndarray[cDOUBLE, ndim=1] KGv,
            BeamProp prop,
            int update_KGv_only=0
            ):
        """Update sparse vectors for geometric stiffness matrix KG

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
        cdef double L, A, E, N
        cdef double r11, r12, r13, r21, r22, r23, r31, r32, r33
        cdef double sina, sinb, sing

        with nogil:
            L = self.length
            A = prop.A
            E = prop.E

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


    cpdef void update_M(Truss self,
            np.ndarray[cINT, ndim=1] Mr,
            np.ndarray[cINT, ndim=1] Mc,
            np.ndarray[cDOUBLE, ndim=1] Mv,
            BeamProp prop,
            int mtype=0,
            ):
        """Update sparse vectors for mass matrix M

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
        cdef double L, A, E
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

                #NOTE obtained with two-point Gauss-Lobatto quadrature

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
