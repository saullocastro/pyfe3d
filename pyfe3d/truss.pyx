#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: nonecheck=False
#cython: infer_types=False
"""
Truss - Linear truss 3D element with analytical integration (:mod:`pyfe3d.truss`)
=================================================================================

.. currentmodule:: pyfe3d.truss

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
    cdef public double r11, r12, r13, r21, r22, r23, r31, r32, r33
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
        self.r11 = self.r12 = self.r13 = 0.
        self.r21 = self.r22 = self.r23 = 0.
        self.r31 = self.r32 = self.r33 = 0.


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

        #FIXME double check all this part
        with nogil:
            # positions in the global stiffness matrix
            c[0] = self.c1
            c[1] = self.c2

            # global to local transformation of displacements
            su[0] = self.r11
            su[1] = self.r21
            su[2] = self.r31
            sv[0] = self.r12
            sv[1] = self.r22
            sv[2] = self.r32
            sw[0] = self.r13
            sw[1] = self.r23
            sw[2] = self.r33

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


    cpdef void update_rotation_matrix(Truss self, np.ndarray[cDOUBLE, ndim=1] x):
        r"""Update the rotation matrix of the element

        Attributes ``r11,r12,r13,r21,r22,r23,r31,r32,r33`` are updated,
        corresponding to the rotation matrix from local to global coordinates.

        The element coordinate system is determined, identifying the `ijk`
        components of each axis: `{x_e}_i, {x_e}_j, {x_e}_k`; `{y_e}_i,
        {y_e}_j, {y_e}_k`; `{z_e}_i, {z_e}_j, {z_e}_k`.

        The rotation matrix terms are calculated after solving 9 equations.

        Parameters
        ----------
        x : array-like
            Array with global nodal coordinates, for a total of `M` nodes in
            the model, this array will be arranged as: `x_1, y_1, z_1, x_2,
            y_2, z_2, ..., x_M, y_M, z_M`.

        """
        cdef double vxyi, vxyj, vxyk
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

            #NOTE arbitrary off-axis vector
            vxyi = xj
            vxyj = xk
            vxyk = xi

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

            tmp = xi*yj*zk - xi*yk*zj - xj*yi*zk + xj*yk*zi + xk*yi*zj - xk*yj*zi
            self.r11 = (yj*zk - yk*zj)/tmp
            self.r12 = (-xj*zk + xk*zj)/tmp
            self.r13 = (xj*yk - xk*yj)/tmp
            self.r21 = (-yi*zk + yk*zi)/tmp
            self.r22 = (xi*zk - xk*zi)/tmp
            self.r23 = (-xi*yk + xk*yi)/tmp
            self.r31 = (yi*zj - yj*zi)/tmp
            self.r32 = (-xi*zj + xj*zi)/tmp
            self.r33 = (xi*yj - xj*yi)/tmp


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

        with nogil:
            # positions in the global stiffness matrix
            c[0] = self.c1
            c[1] = self.c2

            # global to local transformation of displacements
            su[0] = self.r11
            su[1] = self.r21
            su[2] = self.r31
            sv[0] = self.r12
            sv[1] = self.r22
            sv[2] = self.r32
            sw[0] = self.r13
            sw[1] = self.r23
            sw[2] = self.r33

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

        with nogil:
            L = self.length
            A = prop.A
            E = prop.E
            G = prop.G
            scf = prop.scf
            Iyy = prop.Iyy
            Izz = prop.Izz

            #Local to global transformation
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

        with nogil:
            L = self.length
            intrho = prop.intrho
            intrhoy2 = prop.intrhoy2
            intrhoz2 = prop.intrhoz2
            A = prop.A
            E = prop.E

            #Local to global transformation
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
