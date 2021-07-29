#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: nonecheck=False
#cython: infer_types=False
"""
Truss - Linear truss 3D element with analytical integration
-----------------------------------------------------------

.. note:: The :class:`.BeamLR` element is recommended because of the better
          physical representation.

.. note:: The :class:`.Truss` element does not support linear buckling
          analysis.

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
    r"""
    Used to allocate memory for the sparse matrices.

    Attributes
    ----------
    KC0_SPARSE_SIZE : int
        ``KC0_SPARSE_SIZE = 72``

    M_SPARSE_SIZE : int
        ``M_SPARSE_SIZE = 72``

    """
    cdef public cINT KC0_SPARSE_SIZE
    cdef public cINT M_SPARSE_SIZE
    def __cinit__(TrussData self):
        self.KC0_SPARSE_SIZE = 72
        self.M_SPARSE_SIZE = 72

cdef class TrussProbe:
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
    def __cinit__(TrussProbe self):
        self.xe = np.zeros(NUM_NODES*DOF//2, dtype=DOUBLE)
        self.ue = np.zeros(NUM_NODES*DOF, dtype=DOUBLE)

cdef class Truss:
    r"""
    Truss 3D element for axial- and torsion-only behavior

    Nodal connectivity for the truss element::

        ______   --> u  ->>- rx
        1    2


    .. note:: The :class:`.BeamLR` is recommended because of the better
              physical representation.

    """
    cdef public cINT eid
    cdef public cINT n1, n2
    cdef public cINT c1, c2
    cdef public cINT init_k_KC0, init_k_M
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
        self.init_k_M = 0
        self.length = 0
        self.cosa = 1.
        self.cosb = 1.
        self.cosg = 1.

    cpdef void update_ue(Truss self, np.ndarray[cDOUBLE, ndim=1] u):
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

    cpdef void update_xe(Truss self, np.ndarray[cDOUBLE, ndim=1] x):
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

    cpdef void update_length(Truss self):
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

    cpdef void update_KC0(Truss self,
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


    cpdef void update_M(Truss self,
            np.ndarray[cINT, ndim=1] Mr,
            np.ndarray[cINT, ndim=1] Mc,
            np.ndarray[cDOUBLE, ndim=1] Mv,
            BeamProp prop,
            int mtype=0,
            ):
        r"""Update sparse vectors for mass matrix M

        For the :class:`.Truss` element, only the inertial terms ``intrho``,
        ``intrhoy2`` and ``intrhoz2`` of the beam property are important.

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
        cdef double intrho, intrhoy2, intrhoz2
        cdef double r11, r12, r13, r21, r22, r23, r31, r32, r33
        cdef double L, A, E
        cdef double sina, sinb, sing

        with nogil:
            L = self.length
            intrho = prop.intrho
            intrhoy2 = prop.intrhoy2
            intrhoz2 = prop.intrhoz2
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
                Mc[k] = 0+c2
                k += 1
                Mr[k] = 0+c1
                Mc[k] = 1+c2
                k += 1
                Mr[k] = 0+c1
                Mc[k] = 2+c2
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
                Mc[k] = 0+c2
                k += 1
                Mr[k] = 1+c1
                Mc[k] = 1+c2
                k += 1
                Mr[k] = 1+c1
                Mc[k] = 2+c2
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
                Mc[k] = 0+c2
                k += 1
                Mr[k] = 2+c1
                Mc[k] = 1+c2
                k += 1
                Mr[k] = 2+c1
                Mc[k] = 2+c2
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
                Mc[k] = 3+c2
                k += 1
                Mr[k] = 3+c1
                Mc[k] = 4+c2
                k += 1
                Mr[k] = 3+c1
                Mc[k] = 5+c2
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
                Mc[k] = 3+c2
                k += 1
                Mr[k] = 4+c1
                Mc[k] = 4+c2
                k += 1
                Mr[k] = 4+c1
                Mc[k] = 5+c2
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
                Mc[k] = 0+c2
                k += 1
                Mr[k] = 0+c2
                Mc[k] = 1+c2
                k += 1
                Mr[k] = 0+c2
                Mc[k] = 2+c2
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
                Mc[k] = 0+c2
                k += 1
                Mr[k] = 1+c2
                Mc[k] = 1+c2
                k += 1
                Mr[k] = 1+c2
                Mc[k] = 2+c2
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
                Mc[k] = 0+c2
                k += 1
                Mr[k] = 2+c2
                Mc[k] = 1+c2
                k += 1
                Mr[k] = 2+c2
                Mc[k] = 2+c2
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
                Mc[k] = 3+c2
                k += 1
                Mr[k] = 3+c2
                Mc[k] = 4+c2
                k += 1
                Mr[k] = 3+c2
                Mc[k] = 5+c2
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
                Mc[k] = 3+c2
                k += 1
                Mr[k] = 4+c2
                Mc[k] = 4+c2
                k += 1
                Mr[k] = 4+c2
                Mc[k] = 5+c2
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
                Mc[k] = 3+c2
                k += 1
                Mr[k] = 5+c2
                Mc[k] = 4+c2
                k += 1
                Mr[k] = 5+c2
                Mc[k] = 5+c2

                k = self.init_k_M
                Mv[k] += 0.333333333333333*L*intrho*r11**2
                k += 1
                Mv[k] += 0.333333333333333*L*intrho*r11*r21
                k += 1
                Mv[k] += 0.333333333333333*L*intrho*r11*r31
                k += 1
                Mv[k] += 0.166666666666667*L*intrho*r11**2
                k += 1
                Mv[k] += 0.166666666666667*L*intrho*r11*r21
                k += 1
                Mv[k] += 0.166666666666667*L*intrho*r11*r31
                k += 1
                Mv[k] += 0.333333333333333*L*intrho*r11*r21
                k += 1
                Mv[k] += 0.333333333333333*L*intrho*r21**2
                k += 1
                Mv[k] += 0.333333333333333*L*intrho*r21*r31
                k += 1
                Mv[k] += 0.166666666666667*L*intrho*r11*r21
                k += 1
                Mv[k] += 0.166666666666667*L*intrho*r21**2
                k += 1
                Mv[k] += 0.166666666666667*L*intrho*r21*r31
                k += 1
                Mv[k] += 0.333333333333333*L*intrho*r11*r31
                k += 1
                Mv[k] += 0.333333333333333*L*intrho*r21*r31
                k += 1
                Mv[k] += 0.333333333333333*L*intrho*r31**2
                k += 1
                Mv[k] += 0.166666666666667*L*intrho*r11*r31
                k += 1
                Mv[k] += 0.166666666666667*L*intrho*r21*r31
                k += 1
                Mv[k] += 0.166666666666667*L*intrho*r31**2
                k += 1
                Mv[k] += 0.333333333333333*L*r11**2*(intrhoy2 + intrhoz2)
                k += 1
                Mv[k] += 0.333333333333333*L*r11*r21*(intrhoy2 + intrhoz2)
                k += 1
                Mv[k] += 0.333333333333333*L*r11*r31*(intrhoy2 + intrhoz2)
                k += 1
                Mv[k] += 0.166666666666667*L*r11**2*(intrhoy2 + intrhoz2)
                k += 1
                Mv[k] += 0.166666666666667*L*r11*r21*(intrhoy2 + intrhoz2)
                k += 1
                Mv[k] += 0.166666666666667*L*r11*r31*(intrhoy2 + intrhoz2)
                k += 1
                Mv[k] += 0.333333333333333*L*r11*r21*(intrhoy2 + intrhoz2)
                k += 1
                Mv[k] += 0.333333333333333*L*r21**2*(intrhoy2 + intrhoz2)
                k += 1
                Mv[k] += 0.333333333333333*L*r21*r31*(intrhoy2 + intrhoz2)
                k += 1
                Mv[k] += 0.166666666666667*L*r11*r21*(intrhoy2 + intrhoz2)
                k += 1
                Mv[k] += 0.166666666666667*L*r21**2*(intrhoy2 + intrhoz2)
                k += 1
                Mv[k] += 0.166666666666667*L*r21*r31*(intrhoy2 + intrhoz2)
                k += 1
                Mv[k] += 0.333333333333333*L*r11*r31*(intrhoy2 + intrhoz2)
                k += 1
                Mv[k] += 0.333333333333333*L*r21*r31*(intrhoy2 + intrhoz2)
                k += 1
                Mv[k] += 0.333333333333333*L*r31**2*(intrhoy2 + intrhoz2)
                k += 1
                Mv[k] += 0.166666666666667*L*r11*r31*(intrhoy2 + intrhoz2)
                k += 1
                Mv[k] += 0.166666666666667*L*r21*r31*(intrhoy2 + intrhoz2)
                k += 1
                Mv[k] += 0.166666666666667*L*r31**2*(intrhoy2 + intrhoz2)
                k += 1
                Mv[k] += 0.166666666666667*L*intrho*r11**2
                k += 1
                Mv[k] += 0.166666666666667*L*intrho*r11*r21
                k += 1
                Mv[k] += 0.166666666666667*L*intrho*r11*r31
                k += 1
                Mv[k] += 0.333333333333333*L*intrho*r11**2
                k += 1
                Mv[k] += 0.333333333333333*L*intrho*r11*r21
                k += 1
                Mv[k] += 0.333333333333333*L*intrho*r11*r31
                k += 1
                Mv[k] += 0.166666666666667*L*intrho*r11*r21
                k += 1
                Mv[k] += 0.166666666666667*L*intrho*r21**2
                k += 1
                Mv[k] += 0.166666666666667*L*intrho*r21*r31
                k += 1
                Mv[k] += 0.333333333333333*L*intrho*r11*r21
                k += 1
                Mv[k] += 0.333333333333333*L*intrho*r21**2
                k += 1
                Mv[k] += 0.333333333333333*L*intrho*r21*r31
                k += 1
                Mv[k] += 0.166666666666667*L*intrho*r11*r31
                k += 1
                Mv[k] += 0.166666666666667*L*intrho*r21*r31
                k += 1
                Mv[k] += 0.166666666666667*L*intrho*r31**2
                k += 1
                Mv[k] += 0.333333333333333*L*intrho*r11*r31
                k += 1
                Mv[k] += 0.333333333333333*L*intrho*r21*r31
                k += 1
                Mv[k] += 0.333333333333333*L*intrho*r31**2
                k += 1
                Mv[k] += 0.166666666666667*L*r11**2*(intrhoy2 + intrhoz2)
                k += 1
                Mv[k] += 0.166666666666667*L*r11*r21*(intrhoy2 + intrhoz2)
                k += 1
                Mv[k] += 0.166666666666667*L*r11*r31*(intrhoy2 + intrhoz2)
                k += 1
                Mv[k] += 0.333333333333333*L*r11**2*(intrhoy2 + intrhoz2)
                k += 1
                Mv[k] += 0.333333333333333*L*r11*r21*(intrhoy2 + intrhoz2)
                k += 1
                Mv[k] += 0.333333333333333*L*r11*r31*(intrhoy2 + intrhoz2)
                k += 1
                Mv[k] += 0.166666666666667*L*r11*r21*(intrhoy2 + intrhoz2)
                k += 1
                Mv[k] += 0.166666666666667*L*r21**2*(intrhoy2 + intrhoz2)
                k += 1
                Mv[k] += 0.166666666666667*L*r21*r31*(intrhoy2 + intrhoz2)
                k += 1
                Mv[k] += 0.333333333333333*L*r11*r21*(intrhoy2 + intrhoz2)
                k += 1
                Mv[k] += 0.333333333333333*L*r21**2*(intrhoy2 + intrhoz2)
                k += 1
                Mv[k] += 0.333333333333333*L*r21*r31*(intrhoy2 + intrhoz2)
                k += 1
                Mv[k] += 0.166666666666667*L*r11*r31*(intrhoy2 + intrhoz2)
                k += 1
                Mv[k] += 0.166666666666667*L*r21*r31*(intrhoy2 + intrhoz2)
                k += 1
                Mv[k] += 0.166666666666667*L*r31**2*(intrhoy2 + intrhoz2)
                k += 1
                Mv[k] += 0.333333333333333*L*r11*r31*(intrhoy2 + intrhoz2)
                k += 1
                Mv[k] += 0.333333333333333*L*r21*r31*(intrhoy2 + intrhoz2)
                k += 1
                Mv[k] += 0.333333333333333*L*r31**2*(intrhoy2 + intrhoz2)

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
                Mv[k] += L*intrho*r11**2/2
                k += 1
                Mv[k] += L*intrho*r11*r21/2
                k += 1
                Mv[k] += L*intrho*r11*r31/2
                k += 1
                Mv[k] += L*intrho*r11*r21/2
                k += 1
                Mv[k] += L*intrho*r21**2/2
                k += 1
                Mv[k] += L*intrho*r21*r31/2
                k += 1
                Mv[k] += L*intrho*r11*r31/2
                k += 1
                Mv[k] += L*intrho*r21*r31/2
                k += 1
                Mv[k] += L*intrho*r31**2/2
                k += 1
                Mv[k] += L*r11**2*(intrhoy2 + intrhoz2)/2
                k += 1
                Mv[k] += L*r11*r21*(intrhoy2 + intrhoz2)/2
                k += 1
                Mv[k] += L*r11*r31*(intrhoy2 + intrhoz2)/2
                k += 1
                Mv[k] += L*r11*r21*(intrhoy2 + intrhoz2)/2
                k += 1
                Mv[k] += L*r21**2*(intrhoy2 + intrhoz2)/2
                k += 1
                Mv[k] += L*r21*r31*(intrhoy2 + intrhoz2)/2
                k += 1
                Mv[k] += L*r11*r31*(intrhoy2 + intrhoz2)/2
                k += 1
                Mv[k] += L*r21*r31*(intrhoy2 + intrhoz2)/2
                k += 1
                Mv[k] += L*r31**2*(intrhoy2 + intrhoz2)/2
                k += 1
                Mv[k] += L*intrho*r11**2/2
                k += 1
                Mv[k] += L*intrho*r11*r21/2
                k += 1
                Mv[k] += L*intrho*r11*r31/2
                k += 1
                Mv[k] += L*intrho*r11*r21/2
                k += 1
                Mv[k] += L*intrho*r21**2/2
                k += 1
                Mv[k] += L*intrho*r21*r31/2
                k += 1
                Mv[k] += L*intrho*r11*r31/2
                k += 1
                Mv[k] += L*intrho*r21*r31/2
                k += 1
                Mv[k] += L*intrho*r31**2/2
                k += 1
                Mv[k] += L*r11**2*(intrhoy2 + intrhoz2)/2
                k += 1
                Mv[k] += L*r11*r21*(intrhoy2 + intrhoz2)/2
                k += 1
                Mv[k] += L*r11*r31*(intrhoy2 + intrhoz2)/2
                k += 1
                Mv[k] += L*r11*r21*(intrhoy2 + intrhoz2)/2
                k += 1
                Mv[k] += L*r21**2*(intrhoy2 + intrhoz2)/2
                k += 1
                Mv[k] += L*r21*r31*(intrhoy2 + intrhoz2)/2
                k += 1
                Mv[k] += L*r11*r31*(intrhoy2 + intrhoz2)/2
                k += 1
                Mv[k] += L*r21*r31*(intrhoy2 + intrhoz2)/2
                k += 1
                Mv[k] += L*r31**2*(intrhoy2 + intrhoz2)/2
