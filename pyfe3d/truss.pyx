#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: nonecheck=False
#cython: overflowcheck=False
#cython: initializedcheck=False
#cython: embedsignature=True
#cython: infer_types=False
"""
=================================================================================
Truss - Linear truss 3D element with analytical integration (:mod:`pyfe3d.truss`)
=================================================================================

.. currentmodule:: pyfe3d.truss

.. note:: The :class:`.BeamC` element is recommended because of the better
          physical representation.

.. note:: The :class:`.Truss` element does not support linear buckling
          analysis.

Defines the two-node element :class:`.Truss`, with six degrees-of-freedom
per node, its probe :class:`.TrussProbe` and :class:`.TrussData`, which
gives the number of sparse entries per element of each structural matrix.
The cross-section and material properties are given by a
:class:`.BeamProp` object.

"""
import numpy as np

from .beamprop cimport BeamProp

cdef int DOF = 6
cdef int NUM_NODES = 2


cdef class TrussData:
    r"""
    Used to allocate memory for the sparse matrices.

    Each attribute is the number of entries that one :class:`.Truss`
    element writes in the arrays of rows, columns and values of a sparse
    matrix, at the consecutive positions that start at its ``init_k_KC0`` or
    ``init_k_M``. The constructor takes no argument and sets the values
    below.

    Attributes
    ----------
    KC0_SPARSE_SIZE : int
        ``KC0_SPARSE_SIZE = 72``, written by :meth:`.Truss.update_KC0`.

    M_SPARSE_SIZE : int
        ``M_SPARSE_SIZE = 144``, written by :meth:`.Truss.update_M` with
        ``mtype=0``; with ``mtype=1`` only the first ``36`` are written.

    """
    cdef public int KC0_SPARSE_SIZE
    cdef public int M_SPARSE_SIZE

    def __cinit__(TrussData self):
        self.KC0_SPARSE_SIZE = 72
        self.M_SPARSE_SIZE = 144


cdef class TrussProbe:
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
    xe : array-like
        Array of size ``NUM_NODES*DOF//2=6`` containing the nodal coordinates
        in the element coordinate system, in the following order `{x_e}_1,
        {y_e}_1, {z_e}_1, {x_e}_2, {y_e}_2, {z_e}_2`. Dimension length.
        Updated by :meth:`.Truss.update_probe_xe`.
    ue : array-like
        Array of size ``NUM_NODES*DOF=12`` containing the element displacements
        in the following order `u_{e_1}, v_{e_1}, w_{e_1}, {r_x}_{e_1},
        {r_y}_{e_1}, {r_z}_{e_1}, u_{e_2}, v_{e_2}, w_{e_2}, {r_x}_{e_2},
        {r_y}_{e_2}, {r_z}_{e_2}`. Translations of dimension length and
        rotations. Updated by :meth:`.Truss.update_probe_ue`.
    finte, : array-like
        Array of size ``NUM_NODES*DOF=12`` containing the element internal
        forces corresponding to the degrees-of-freedom described by ``ue``,
        forces and moments. Updated by :meth:`.Truss.update_probe_finte`.

    """
    cdef public double [::1] xe
    cdef public double [::1] ue
    cdef public double [::1] finte

    def __cinit__(TrussProbe self):
        self.xe = np.zeros(NUM_NODES*DOF//2, dtype=np.float64)
        self.ue = np.zeros(NUM_NODES*DOF, dtype=np.float64)
        self.finte = np.zeros(NUM_NODES*DOF, dtype=np.float64)


cdef class Truss:
    r"""
    Truss 3D element for axial- and torsion-only behavior

    Nodal connectivity for the truss element::

        ______   --> u  ->>- rx
        1    2


    .. note:: The :class:`.BeamC` is recommended because of the better
              physical representation.

    The constructor stores ``p`` in ``probe``, sets ``eid``, ``pid``,
    ``n1``, ``n2``, ``c1`` and ``c2`` to ``-1``, and sets the ``init_k_*``
    positions, ``length`` and the rotation matrix to ``0``. The methods use
    the attributes updated by the others: :meth:`.update_rotation_matrix`
    reads ``c1`` and ``c2``; :meth:`.update_probe_xe` reads the rotation
    matrix and updates ``length``; :meth:`.update_KC0`, :meth:`.update_M`
    and :meth:`.update_probe_finte` read ``length`` and the rotation matrix;
    :meth:`.update_probe_finte` also reads ``probe.ue``, updated by
    :meth:`.update_probe_ue`.

    Parameters
    ----------
    p : :class:`.TrussProbe` object
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
    c1, c2 : int
        Position of each node in the global stiffness matrix, i.e. of the
        first of its ``6`` degrees-of-freedom. The coordinates of the node
        are read from the position ``c1//2`` or ``c2//2`` of the array of
        nodal coordinates.
    n1, n2 : int
        Node identification number.
    init_k_KC0, init_k_M : int
        Position in the arrays storing the sparse data for the structural
        matrices, where the entries of this element start, see
        :class:`.TrussData` for their number.
    probe, : :class:`.TrussProbe` object
        Pointer to the probe.

    Notes
    -----
    The strain energy of the element is

    .. math::
        U = \frac{1}{2} \int_0^L \left( E A u_{,x}^2 + G J {r_x}_{,x}^2
            \right) dx

    where `E A` and `G J` are the entries ``C[0, 0]`` and ``C[3, 3]`` of
    the sectional stiffness matrix ``C`` of :class:`.BeamProp`, i.e. ``E*A``
    and the torsional stiffness ``GJ`` when ``C`` is not set; the other
    entries of ``C``, e.g. the extension-twist coupling ``C[0, 3]``, are
    not used. The axial displacement `u` and the twist rotation `r_x` are
    interpolated linearly between the nodes, in the element coordinate
    system, and `L` = ``length``. The transverse translations `v`, `w` and
    the rotations `r_y`, `r_z` have no stiffness and must be supported by
    other elements or by boundary conditions; the element has no geometric
    stiffness matrix. The
    matrices and the internal forces are computed in element coordinates and
    transformed to global coordinates with the transformation matrix of the
    element, `\boldsymbol{K} = \boldsymbol{T} \boldsymbol{K}_e \boldsymbol{T}^\top`, where `\boldsymbol{T}` is block diagonal with four
    copies of the matrix `\boldsymbol{T}_{3\times 3}` of the entries `T_{ij}` stored in
    ``r11, ..., r33``, whose columns are the element axes.

    """
    cdef public int eid, pid
    cdef public int n1, n2
    cdef public int c1, c2
    cdef public int init_k_KC0, init_k_M
    cdef public double length
    cdef public double r11, r12, r13, r21, r22, r23, r31, r32, r33
    cdef public TrussProbe probe

    def __cinit__(Truss self, TrussProbe p):
        self.probe = p
        self.eid = -1
        self.pid = -1
        self.n1 = -1
        self.n2 = -1
        self.c1 = -1
        self.c2 = -1
        self.init_k_KC0 = 0
        # self.init_k_KCNL = 0
        self.init_k_M = 0
        self.length = 0
        self.r11 = self.r12 = self.r13 = 0.
        self.r21 = self.r22 = self.r23 = 0.
        self.r31 = self.r32 = self.r33 = 0.


    cpdef void update_probe_ue(Truss self, double [::1] u):
        r"""Update the local displacement vector of the probe of the element

        .. note:: The ``ue`` attribute of object :class:`.TrussProbe` is
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


    cpdef void update_rotation_matrix(Truss self, double [::1] x):
        r"""Update the rotation matrix of the element

        Attributes ``r11,r12,r13,r21,r22,r23,r31,r32,r33`` are updated, with
        the entries `T_{ij}` of the transformation matrix `\boldsymbol{T}_{3\times 3}`
        from element to global coordinates, whose columns are the element axes.

        The element coordinate system is determined, identifying the `ijk`
        components of each axis: `{x_e}_i, {x_e}_j, {x_e}_k`; `{y_e}_i,
        {y_e}_j, {y_e}_k`; `{z_e}_i, {z_e}_j, {z_e}_k`.

        The element `x` axis goes from node 1 to node 2, whose coordinates are
        read from ``x`` at the positions ``c1//2`` and ``c2//2``. The `z` axis
        is normal to the `x` axis and to the vector obtained by a cyclic
        permutation of the components of the `x` axis, ``(xj, xk, xi)``, the
        `y` axis completes the right-handed system, and the three axes are
        normalized.


        Parameters
        ----------
        x : array-like
            Array with global nodal coordinates, for a total of `M` nodes in
            the model, this array will be arranged as: `x_1, y_1, z_1, x_2,
            y_2, z_2, ..., x_M, y_M, z_M`. Dimension length, 1-D array of size
            ``3*M``.

        """
        cdef double vxyi, vxyj, vxyk
        cdef double xi, xj, xk, yi, yj, yk, zi, zj, zk, tmp
        cdef double x1i, x1j, x1k, x2i, x2j, x2k

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

            # NOTE arbitrary off-axis vector
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

            self.r11 = xi
            self.r21 = xj
            self.r31 = xk
            self.r12 = yi
            self.r22 = yj
            self.r32 = yk
            self.r13 = zi
            self.r23 = zj
            self.r33 = zk


    cpdef void update_probe_xe(Truss self, double [::1] x):
        r"""Update the 3D coordinates of the probe of the element

        .. note:: The ``xe`` attribute of object :class:`.TrussProbe` is
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


    cpdef void update_length(Truss self):
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


    cpdef void update_probe_finte(Truss self, BeamProp prop):
        r"""Update the internal force vector of the probe

        The attribute ``finte`` is updated with the :class:`.TrussProbe` the
        internal forces in local coordinates. While using this function, mind
        that the probe can be shared amongst more than one finite element,
        depending how you defined them, meaning that the probe will always safe
        the values from the last udpate.

        .. note:: The ``finte`` attribute of object :class:`.TrussProbe` is
                  updated, accessible using ``.probe.finte``.

        The 12 values of ``finte`` are overwritten. They are computed from
        ``length``, from ``C[0, 0]`` and ``C[3, 3]`` of the sectional
        stiffness matrix of ``prop``, see the notes of :class:`.Truss`, and
        from the displacements ``probe.ue``, which must be
        updated before, see :meth:`.update_probe_xe` and
        :meth:`.update_probe_ue`. Only the axial forces and the torsion
        moments are not zero.

        Parameters
        ----------
        prop : :class:`.BeamProp` object
            Beam property object from where the stiffness and mass attributes
            are read from.

        """
        cdef double *ue
        cdef double *finte
        cdef double L, EA, GJ
        cdef double C[36]

        with nogil:
            L = self.length
            # NOTE only EA and GJ of the sectional stiffness matrix of the
            #      property, see the notes of Truss
            prop.get_C(C)
            EA = C[0]
            GJ = C[21]

            ue = &self.probe.ue[0]
            finte = &self.probe.finte[0]

            finte[0] = 1.0*EA*(ue[0] - ue[6])/L
            finte[1] = 0.
            finte[2] = 0.
            finte[3] = 1.0*GJ*(ue[3] - ue[9])/L
            finte[4] = 0.
            finte[5] = 0.
            finte[6] = 1.0*EA*(-ue[0] + ue[6])/L
            finte[7] = 0.
            finte[8] = 0.
            finte[9] = 1.0*GJ*(-ue[3] + ue[9])/L
            finte[10] = 0.
            finte[11] = 0.


    cpdef void update_KC0(Truss self,
                          long [::1] KC0r,
                          long [::1] KC0c,
                          double [::1] KC0v,
                          BeamProp prop,
                          int update_KC0v_only=0,
                          ):
        r"""Update sparse vectors for linear constitutive stiffness matrix KC0

        The 72 entries of the matrix of the element, in global coordinates,
        are written at the positions ``init_k_KC0`` to ``init_k_KC0 + 71``:
        the row and column indices, built from ``c1`` and ``c2``, are stored
        in ``KC0r`` and ``KC0c``, and the values are added to those already
        in ``KC0v``. The matrix is computed from ``length``, the rotation
        matrix and ``C[0, 0]`` and ``C[3, 3]`` of the sectional stiffness
        matrix of ``prop``, see the notes of :class:`.Truss`. ``length``
        and the rotation matrix must be updated before,
        see :meth:`.update_rotation_matrix` and :meth:`.update_probe_xe`.

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
        prop : :class:`.BeamProp` object
            Beam property object from where the stiffness and mass attributes
            are read from.
        update_KC0v_only : int, optional
            The default ``0`` means that the row and column indices ``KC0r``
            and ``KC0c`` should also be updated. Any other value will only
            update the stiffness matrix values ``KC0v``.

        Notes
        -----
        In element coordinates, ``KC0`` is the Hessian of the strain energy
        given in the notes of :class:`.Truss`, with the entries `\pm E A/L`
        between the axial translations and `\pm G J/L` between the twist
        rotations of the two nodes.

        """
        cdef int c1, c2, k
        cdef double L, EA, GJ
        cdef double C[36]
        cdef double r11, r21, r31

        with nogil:
            L = self.length
            # NOTE only EA and GJ of the sectional stiffness matrix of the
            #      property, see the notes of Truss
            prop.get_C(C)
            EA = C[0]
            GJ = C[21]

            # local to global transformation
            r11 = self.r11
            r21 = self.r21
            r31 = self.r31

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
            KC0v[k] += 1.0*EA*r11**2/L
            k += 1
            KC0v[k] += 1.0*EA*r11*r21/L
            k += 1
            KC0v[k] += 1.0*EA*r11*r31/L
            k += 1
            KC0v[k] += -1.0*EA*r11**2/L
            k += 1
            KC0v[k] += -1.0*EA*r11*r21/L
            k += 1
            KC0v[k] += -1.0*EA*r11*r31/L
            k += 1
            KC0v[k] += 1.0*EA*r11*r21/L
            k += 1
            KC0v[k] += 1.0*EA*r21**2/L
            k += 1
            KC0v[k] += 1.0*EA*r21*r31/L
            k += 1
            KC0v[k] += -1.0*EA*r11*r21/L
            k += 1
            KC0v[k] += -1.0*EA*r21**2/L
            k += 1
            KC0v[k] += -1.0*EA*r21*r31/L
            k += 1
            KC0v[k] += 1.0*EA*r11*r31/L
            k += 1
            KC0v[k] += 1.0*EA*r21*r31/L
            k += 1
            KC0v[k] += 1.0*EA*r31**2/L
            k += 1
            KC0v[k] += -1.0*EA*r11*r31/L
            k += 1
            KC0v[k] += -1.0*EA*r21*r31/L
            k += 1
            KC0v[k] += -1.0*EA*r31**2/L
            k += 1
            KC0v[k] += 1.0*GJ*r11**2/L
            k += 1
            KC0v[k] += 1.0*GJ*r11*r21/L
            k += 1
            KC0v[k] += 1.0*GJ*r11*r31/L
            k += 1
            KC0v[k] += -1.0*GJ*r11**2/L
            k += 1
            KC0v[k] += -1.0*GJ*r11*r21/L
            k += 1
            KC0v[k] += -1.0*GJ*r11*r31/L
            k += 1
            KC0v[k] += 1.0*GJ*r11*r21/L
            k += 1
            KC0v[k] += 1.0*GJ*r21**2/L
            k += 1
            KC0v[k] += 1.0*GJ*r21*r31/L
            k += 1
            KC0v[k] += -1.0*GJ*r11*r21/L
            k += 1
            KC0v[k] += -1.0*GJ*r21**2/L
            k += 1
            KC0v[k] += -1.0*GJ*r21*r31/L
            k += 1
            KC0v[k] += 1.0*GJ*r11*r31/L
            k += 1
            KC0v[k] += 1.0*GJ*r21*r31/L
            k += 1
            KC0v[k] += 1.0*GJ*r31**2/L
            k += 1
            KC0v[k] += -1.0*GJ*r11*r31/L
            k += 1
            KC0v[k] += -1.0*GJ*r21*r31/L
            k += 1
            KC0v[k] += -1.0*GJ*r31**2/L
            k += 1
            KC0v[k] += -1.0*EA*r11**2/L
            k += 1
            KC0v[k] += -1.0*EA*r11*r21/L
            k += 1
            KC0v[k] += -1.0*EA*r11*r31/L
            k += 1
            KC0v[k] += 1.0*EA*r11**2/L
            k += 1
            KC0v[k] += 1.0*EA*r11*r21/L
            k += 1
            KC0v[k] += 1.0*EA*r11*r31/L
            k += 1
            KC0v[k] += -1.0*EA*r11*r21/L
            k += 1
            KC0v[k] += -1.0*EA*r21**2/L
            k += 1
            KC0v[k] += -1.0*EA*r21*r31/L
            k += 1
            KC0v[k] += 1.0*EA*r11*r21/L
            k += 1
            KC0v[k] += 1.0*EA*r21**2/L
            k += 1
            KC0v[k] += 1.0*EA*r21*r31/L
            k += 1
            KC0v[k] += -1.0*EA*r11*r31/L
            k += 1
            KC0v[k] += -1.0*EA*r21*r31/L
            k += 1
            KC0v[k] += -1.0*EA*r31**2/L
            k += 1
            KC0v[k] += 1.0*EA*r11*r31/L
            k += 1
            KC0v[k] += 1.0*EA*r21*r31/L
            k += 1
            KC0v[k] += 1.0*EA*r31**2/L
            k += 1
            KC0v[k] += -1.0*GJ*r11**2/L
            k += 1
            KC0v[k] += -1.0*GJ*r11*r21/L
            k += 1
            KC0v[k] += -1.0*GJ*r11*r31/L
            k += 1
            KC0v[k] += 1.0*GJ*r11**2/L
            k += 1
            KC0v[k] += 1.0*GJ*r11*r21/L
            k += 1
            KC0v[k] += 1.0*GJ*r11*r31/L
            k += 1
            KC0v[k] += -1.0*GJ*r11*r21/L
            k += 1
            KC0v[k] += -1.0*GJ*r21**2/L
            k += 1
            KC0v[k] += -1.0*GJ*r21*r31/L
            k += 1
            KC0v[k] += 1.0*GJ*r11*r21/L
            k += 1
            KC0v[k] += 1.0*GJ*r21**2/L
            k += 1
            KC0v[k] += 1.0*GJ*r21*r31/L
            k += 1
            KC0v[k] += -1.0*GJ*r11*r31/L
            k += 1
            KC0v[k] += -1.0*GJ*r21*r31/L
            k += 1
            KC0v[k] += -1.0*GJ*r31**2/L
            k += 1
            KC0v[k] += 1.0*GJ*r11*r31/L
            k += 1
            KC0v[k] += 1.0*GJ*r21*r31/L
            k += 1
            KC0v[k] += 1.0*GJ*r31**2/L


    cpdef void update_fint(Truss self, double [::1] fint, BeamProp prop):
        r"""Update the internal force vector

        Calls ``update_probe_finte(prop)``, transforms the forces and moments
        of ``probe.finte`` to global coordinates with the rotation matrix and
        adds them to ``fint`` at the positions ``c1`` to ``c1 + 5`` and ``c2``
        to ``c2 + 5``. The displacements ``probe.ue`` must be updated before,
        see :meth:`.update_probe_ue`.

        Parameters
        ----------
        fint : np.array
            Array that is updated in place with the internal forces. The
            internal forces stored in ``fint`` are calculated in global
            coordinates. Method :meth:`.update_probe_finte` is called to update
            the parameter ``finte`` of the :class:`.TrussProbe` with the
            internal forces in local coordinates. 1-D array of size ``6*M``,
            for `M` nodes in the model.
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


    cpdef void update_M(Truss self,
                        long [::1] Mr,
                        long [::1] Mc,
                        double [::1] Mv,
                        BeamProp prop,
                        int mtype=0,
                        ):
        r"""Update sparse vectors for mass matrix M

        For the :class:`.Truss` element, the inertial terms ``intrho``,
        ``intrhoy``, ``intrhoz``, ``intrhoy2`` and ``intrhoz2`` of the beam
        property are important.

        The entries of the matrix of the element, in global coordinates, are
        written from the position ``init_k_M``: 144 entries with
        ``mtype=0`` and 36 entries with ``mtype=1``. The row and column
        indices, built from ``c1`` and ``c2``, are stored in ``Mr`` and
        ``Mc``, and the values are added to those already in ``Mv``. The
        matrix is computed from ``length``, the rotation matrix and the inertial terms above.
        ``length`` and the rotation matrix must be updated before, see
        :meth:`.update_rotation_matrix` and :meth:`.update_probe_xe`.

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
        The mass matrices come from the kinetic energy of the rigid cross
        section, whose point `(y, z)` has the displacements `u_x = u`, `u_y
        = v - z r_x` and `u_z = w + y r_x`, with the three translations and
        the twist rotation `r_x` interpolated linearly:

        .. math::
            T = \frac{1}{2} \int_0^L \Big[ m (\dot{u}^2 + \dot{v}^2 + \dot{w}^2)
                - 2 m_z \dot{v} \dot{r}_x + 2 m_y \dot{w} \dot{r}_x
                + (m_{yy} + m_{zz}) \dot{r}_x^2 \Big] dx

        with `m` = ``intrho``, `m_y` = ``intrhoy``, `m_z` = ``intrhoz``,
        `m_{yy}` = ``intrhoy2`` and `m_{zz}` = ``intrhoz2``. The coupling
        terms, `-m_z` between `v` and `r_x` and `+m_y` between `w` and `r_x`,
        are those of :meth:`.BeamLR.update_M` and :meth:`.BeamC.update_M`.
        The rotations `r_y` and `r_z` have no inertia. ``mtype=0`` gives the
        consistent mass matrix, the Hessian of `T` with respect to the nodal
        velocities, integrated exactly along the element. ``mtype=1`` gives
        the lumped mass matrix, diagonal in element coordinates: at each node
        `L m/2` for the three translations and `L (m_{yy} + m_{zz})/2` for
        `r_x`, without the coupling terms of `m_y` and `m_z`.

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
        cdef double intrho, intrhoy, intrhoz, intrhoy2, intrhoz2
        cdef double r11, r12, r13, r21, r22, r23, r31, r32, r33
        cdef double L

        # NOTE checked once per call, before the code without the GIL
        if mtype != 0 and mtype != 1:
            raise ValueError('mtype=%d is not supported by Truss.update_M, '
                             'use 0 (consistent) or 1 (lumped); Truss has '
                             'no reduced-integration mass matrix (2)'
                             % mtype)

        with nogil:
            L = self.length
            intrho = prop.intrho
            intrhoy = prop.intrhoy
            intrhoz = prop.intrhoz
            intrhoy2 = prop.intrhoy2
            intrhoz2 = prop.intrhoz2

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
                Mv[k] += r11*(0.333333333333333*L*intrhoy*r13 - 0.333333333333333*L*intrhoz*r12)
                k += 1
                Mv[k] += r21*(0.333333333333333*L*intrhoy*r13 - 0.333333333333333*L*intrhoz*r12)
                k += 1
                Mv[k] += r31*(0.333333333333333*L*intrhoy*r13 - 0.333333333333333*L*intrhoz*r12)
                k += 1
                Mv[k] += 0.166666666666667*L*intrho*r11**2 + 0.166666666666667*L*intrho*r12**2 + 0.166666666666667*L*intrho*r13**2
                k += 1
                Mv[k] += 0.166666666666667*L*intrho*r11*r21 + 0.166666666666667*L*intrho*r12*r22 + 0.166666666666667*L*intrho*r13*r23
                k += 1
                Mv[k] += 0.166666666666667*L*intrho*r11*r31 + 0.166666666666667*L*intrho*r12*r32 + 0.166666666666667*L*intrho*r13*r33
                k += 1
                Mv[k] += r11*(0.166666666666667*L*intrhoy*r13 - 0.166666666666667*L*intrhoz*r12)
                k += 1
                Mv[k] += r21*(0.166666666666667*L*intrhoy*r13 - 0.166666666666667*L*intrhoz*r12)
                k += 1
                Mv[k] += r31*(0.166666666666667*L*intrhoy*r13 - 0.166666666666667*L*intrhoz*r12)
                k += 1
                Mv[k] += 0.333333333333333*L*intrho*r11*r21 + 0.333333333333333*L*intrho*r12*r22 + 0.333333333333333*L*intrho*r13*r23
                k += 1
                Mv[k] += 0.333333333333333*L*intrho*r21**2 + 0.333333333333333*L*intrho*r22**2 + 0.333333333333333*L*intrho*r23**2
                k += 1
                Mv[k] += 0.333333333333333*L*intrho*r21*r31 + 0.333333333333333*L*intrho*r22*r32 + 0.333333333333333*L*intrho*r23*r33
                k += 1
                Mv[k] += r11*(0.333333333333333*L*intrhoy*r23 - 0.333333333333333*L*intrhoz*r22)
                k += 1
                Mv[k] += r21*(0.333333333333333*L*intrhoy*r23 - 0.333333333333333*L*intrhoz*r22)
                k += 1
                Mv[k] += r31*(0.333333333333333*L*intrhoy*r23 - 0.333333333333333*L*intrhoz*r22)
                k += 1
                Mv[k] += 0.166666666666667*L*intrho*r11*r21 + 0.166666666666667*L*intrho*r12*r22 + 0.166666666666667*L*intrho*r13*r23
                k += 1
                Mv[k] += 0.166666666666667*L*intrho*r21**2 + 0.166666666666667*L*intrho*r22**2 + 0.166666666666667*L*intrho*r23**2
                k += 1
                Mv[k] += 0.166666666666667*L*intrho*r21*r31 + 0.166666666666667*L*intrho*r22*r32 + 0.166666666666667*L*intrho*r23*r33
                k += 1
                Mv[k] += r11*(0.166666666666667*L*intrhoy*r23 - 0.166666666666667*L*intrhoz*r22)
                k += 1
                Mv[k] += r21*(0.166666666666667*L*intrhoy*r23 - 0.166666666666667*L*intrhoz*r22)
                k += 1
                Mv[k] += r31*(0.166666666666667*L*intrhoy*r23 - 0.166666666666667*L*intrhoz*r22)
                k += 1
                Mv[k] += 0.333333333333333*L*intrho*r11*r31 + 0.333333333333333*L*intrho*r12*r32 + 0.333333333333333*L*intrho*r13*r33
                k += 1
                Mv[k] += 0.333333333333333*L*intrho*r21*r31 + 0.333333333333333*L*intrho*r22*r32 + 0.333333333333333*L*intrho*r23*r33
                k += 1
                Mv[k] += 0.333333333333333*L*intrho*r31**2 + 0.333333333333333*L*intrho*r32**2 + 0.333333333333333*L*intrho*r33**2
                k += 1
                Mv[k] += r11*(0.333333333333333*L*intrhoy*r33 - 0.333333333333333*L*intrhoz*r32)
                k += 1
                Mv[k] += r21*(0.333333333333333*L*intrhoy*r33 - 0.333333333333333*L*intrhoz*r32)
                k += 1
                Mv[k] += r31*(0.333333333333333*L*intrhoy*r33 - 0.333333333333333*L*intrhoz*r32)
                k += 1
                Mv[k] += 0.166666666666667*L*intrho*r11*r31 + 0.166666666666667*L*intrho*r12*r32 + 0.166666666666667*L*intrho*r13*r33
                k += 1
                Mv[k] += 0.166666666666667*L*intrho*r21*r31 + 0.166666666666667*L*intrho*r22*r32 + 0.166666666666667*L*intrho*r23*r33
                k += 1
                Mv[k] += 0.166666666666667*L*intrho*r31**2 + 0.166666666666667*L*intrho*r32**2 + 0.166666666666667*L*intrho*r33**2
                k += 1
                Mv[k] += r11*(0.166666666666667*L*intrhoy*r33 - 0.166666666666667*L*intrhoz*r32)
                k += 1
                Mv[k] += r21*(0.166666666666667*L*intrhoy*r33 - 0.166666666666667*L*intrhoz*r32)
                k += 1
                Mv[k] += r31*(0.166666666666667*L*intrhoy*r33 - 0.166666666666667*L*intrhoz*r32)
                k += 1
                Mv[k] += 0.333333333333333*L*intrhoy*r11*r13 - 0.333333333333333*L*intrhoz*r11*r12
                k += 1
                Mv[k] += 0.333333333333333*L*intrhoy*r11*r23 - 0.333333333333333*L*intrhoz*r11*r22
                k += 1
                Mv[k] += 0.333333333333333*L*intrhoy*r11*r33 - 0.333333333333333*L*intrhoz*r11*r32
                k += 1
                Mv[k] += 0.333333333333333*L*r11**2*(intrhoy2 + intrhoz2)
                k += 1
                Mv[k] += 0.333333333333333*L*r11*r21*(intrhoy2 + intrhoz2)
                k += 1
                Mv[k] += 0.333333333333333*L*r11*r31*(intrhoy2 + intrhoz2)
                k += 1
                Mv[k] += 0.166666666666667*L*intrhoy*r11*r13 - 0.166666666666667*L*intrhoz*r11*r12
                k += 1
                Mv[k] += 0.166666666666667*L*intrhoy*r11*r23 - 0.166666666666667*L*intrhoz*r11*r22
                k += 1
                Mv[k] += 0.166666666666667*L*intrhoy*r11*r33 - 0.166666666666667*L*intrhoz*r11*r32
                k += 1
                Mv[k] += 0.166666666666667*L*r11**2*(intrhoy2 + intrhoz2)
                k += 1
                Mv[k] += 0.166666666666667*L*r11*r21*(intrhoy2 + intrhoz2)
                k += 1
                Mv[k] += 0.166666666666667*L*r11*r31*(intrhoy2 + intrhoz2)
                k += 1
                Mv[k] += 0.333333333333333*L*intrhoy*r13*r21 - 0.333333333333333*L*intrhoz*r12*r21
                k += 1
                Mv[k] += 0.333333333333333*L*intrhoy*r21*r23 - 0.333333333333333*L*intrhoz*r21*r22
                k += 1
                Mv[k] += 0.333333333333333*L*intrhoy*r21*r33 - 0.333333333333333*L*intrhoz*r21*r32
                k += 1
                Mv[k] += 0.333333333333333*L*r11*r21*(intrhoy2 + intrhoz2)
                k += 1
                Mv[k] += 0.333333333333333*L*r21**2*(intrhoy2 + intrhoz2)
                k += 1
                Mv[k] += 0.333333333333333*L*r21*r31*(intrhoy2 + intrhoz2)
                k += 1
                Mv[k] += 0.166666666666667*L*intrhoy*r13*r21 - 0.166666666666667*L*intrhoz*r12*r21
                k += 1
                Mv[k] += 0.166666666666667*L*intrhoy*r21*r23 - 0.166666666666667*L*intrhoz*r21*r22
                k += 1
                Mv[k] += 0.166666666666667*L*intrhoy*r21*r33 - 0.166666666666667*L*intrhoz*r21*r32
                k += 1
                Mv[k] += 0.166666666666667*L*r11*r21*(intrhoy2 + intrhoz2)
                k += 1
                Mv[k] += 0.166666666666667*L*r21**2*(intrhoy2 + intrhoz2)
                k += 1
                Mv[k] += 0.166666666666667*L*r21*r31*(intrhoy2 + intrhoz2)
                k += 1
                Mv[k] += 0.333333333333333*L*intrhoy*r13*r31 - 0.333333333333333*L*intrhoz*r12*r31
                k += 1
                Mv[k] += 0.333333333333333*L*intrhoy*r23*r31 - 0.333333333333333*L*intrhoz*r22*r31
                k += 1
                Mv[k] += 0.333333333333333*L*intrhoy*r31*r33 - 0.333333333333333*L*intrhoz*r31*r32
                k += 1
                Mv[k] += 0.333333333333333*L*r11*r31*(intrhoy2 + intrhoz2)
                k += 1
                Mv[k] += 0.333333333333333*L*r21*r31*(intrhoy2 + intrhoz2)
                k += 1
                Mv[k] += 0.333333333333333*L*r31**2*(intrhoy2 + intrhoz2)
                k += 1
                Mv[k] += 0.166666666666667*L*intrhoy*r13*r31 - 0.166666666666667*L*intrhoz*r12*r31
                k += 1
                Mv[k] += 0.166666666666667*L*intrhoy*r23*r31 - 0.166666666666667*L*intrhoz*r22*r31
                k += 1
                Mv[k] += 0.166666666666667*L*intrhoy*r31*r33 - 0.166666666666667*L*intrhoz*r31*r32
                k += 1
                Mv[k] += 0.166666666666667*L*r11*r31*(intrhoy2 + intrhoz2)
                k += 1
                Mv[k] += 0.166666666666667*L*r21*r31*(intrhoy2 + intrhoz2)
                k += 1
                Mv[k] += 0.166666666666667*L*r31**2*(intrhoy2 + intrhoz2)
                k += 1
                Mv[k] += 0.166666666666667*L*intrho*r11**2 + 0.166666666666667*L*intrho*r12**2 + 0.166666666666667*L*intrho*r13**2
                k += 1
                Mv[k] += 0.166666666666667*L*intrho*r11*r21 + 0.166666666666667*L*intrho*r12*r22 + 0.166666666666667*L*intrho*r13*r23
                k += 1
                Mv[k] += 0.166666666666667*L*intrho*r11*r31 + 0.166666666666667*L*intrho*r12*r32 + 0.166666666666667*L*intrho*r13*r33
                k += 1
                Mv[k] += r11*(0.166666666666667*L*intrhoy*r13 - 0.166666666666667*L*intrhoz*r12)
                k += 1
                Mv[k] += r21*(0.166666666666667*L*intrhoy*r13 - 0.166666666666667*L*intrhoz*r12)
                k += 1
                Mv[k] += r31*(0.166666666666667*L*intrhoy*r13 - 0.166666666666667*L*intrhoz*r12)
                k += 1
                Mv[k] += 0.333333333333333*L*intrho*r11**2 + 0.333333333333333*L*intrho*r12**2 + 0.333333333333333*L*intrho*r13**2
                k += 1
                Mv[k] += 0.333333333333333*L*intrho*r11*r21 + 0.333333333333333*L*intrho*r12*r22 + 0.333333333333333*L*intrho*r13*r23
                k += 1
                Mv[k] += 0.333333333333333*L*intrho*r11*r31 + 0.333333333333333*L*intrho*r12*r32 + 0.333333333333333*L*intrho*r13*r33
                k += 1
                Mv[k] += r11*(0.333333333333333*L*intrhoy*r13 - 0.333333333333333*L*intrhoz*r12)
                k += 1
                Mv[k] += r21*(0.333333333333333*L*intrhoy*r13 - 0.333333333333333*L*intrhoz*r12)
                k += 1
                Mv[k] += r31*(0.333333333333333*L*intrhoy*r13 - 0.333333333333333*L*intrhoz*r12)
                k += 1
                Mv[k] += 0.166666666666667*L*intrho*r11*r21 + 0.166666666666667*L*intrho*r12*r22 + 0.166666666666667*L*intrho*r13*r23
                k += 1
                Mv[k] += 0.166666666666667*L*intrho*r21**2 + 0.166666666666667*L*intrho*r22**2 + 0.166666666666667*L*intrho*r23**2
                k += 1
                Mv[k] += 0.166666666666667*L*intrho*r21*r31 + 0.166666666666667*L*intrho*r22*r32 + 0.166666666666667*L*intrho*r23*r33
                k += 1
                Mv[k] += r11*(0.166666666666667*L*intrhoy*r23 - 0.166666666666667*L*intrhoz*r22)
                k += 1
                Mv[k] += r21*(0.166666666666667*L*intrhoy*r23 - 0.166666666666667*L*intrhoz*r22)
                k += 1
                Mv[k] += r31*(0.166666666666667*L*intrhoy*r23 - 0.166666666666667*L*intrhoz*r22)
                k += 1
                Mv[k] += 0.333333333333333*L*intrho*r11*r21 + 0.333333333333333*L*intrho*r12*r22 + 0.333333333333333*L*intrho*r13*r23
                k += 1
                Mv[k] += 0.333333333333333*L*intrho*r21**2 + 0.333333333333333*L*intrho*r22**2 + 0.333333333333333*L*intrho*r23**2
                k += 1
                Mv[k] += 0.333333333333333*L*intrho*r21*r31 + 0.333333333333333*L*intrho*r22*r32 + 0.333333333333333*L*intrho*r23*r33
                k += 1
                Mv[k] += r11*(0.333333333333333*L*intrhoy*r23 - 0.333333333333333*L*intrhoz*r22)
                k += 1
                Mv[k] += r21*(0.333333333333333*L*intrhoy*r23 - 0.333333333333333*L*intrhoz*r22)
                k += 1
                Mv[k] += r31*(0.333333333333333*L*intrhoy*r23 - 0.333333333333333*L*intrhoz*r22)
                k += 1
                Mv[k] += 0.166666666666667*L*intrho*r11*r31 + 0.166666666666667*L*intrho*r12*r32 + 0.166666666666667*L*intrho*r13*r33
                k += 1
                Mv[k] += 0.166666666666667*L*intrho*r21*r31 + 0.166666666666667*L*intrho*r22*r32 + 0.166666666666667*L*intrho*r23*r33
                k += 1
                Mv[k] += 0.166666666666667*L*intrho*r31**2 + 0.166666666666667*L*intrho*r32**2 + 0.166666666666667*L*intrho*r33**2
                k += 1
                Mv[k] += r11*(0.166666666666667*L*intrhoy*r33 - 0.166666666666667*L*intrhoz*r32)
                k += 1
                Mv[k] += r21*(0.166666666666667*L*intrhoy*r33 - 0.166666666666667*L*intrhoz*r32)
                k += 1
                Mv[k] += r31*(0.166666666666667*L*intrhoy*r33 - 0.166666666666667*L*intrhoz*r32)
                k += 1
                Mv[k] += 0.333333333333333*L*intrho*r11*r31 + 0.333333333333333*L*intrho*r12*r32 + 0.333333333333333*L*intrho*r13*r33
                k += 1
                Mv[k] += 0.333333333333333*L*intrho*r21*r31 + 0.333333333333333*L*intrho*r22*r32 + 0.333333333333333*L*intrho*r23*r33
                k += 1
                Mv[k] += 0.333333333333333*L*intrho*r31**2 + 0.333333333333333*L*intrho*r32**2 + 0.333333333333333*L*intrho*r33**2
                k += 1
                Mv[k] += r11*(0.333333333333333*L*intrhoy*r33 - 0.333333333333333*L*intrhoz*r32)
                k += 1
                Mv[k] += r21*(0.333333333333333*L*intrhoy*r33 - 0.333333333333333*L*intrhoz*r32)
                k += 1
                Mv[k] += r31*(0.333333333333333*L*intrhoy*r33 - 0.333333333333333*L*intrhoz*r32)
                k += 1
                Mv[k] += 0.166666666666667*L*intrhoy*r11*r13 - 0.166666666666667*L*intrhoz*r11*r12
                k += 1
                Mv[k] += 0.166666666666667*L*intrhoy*r11*r23 - 0.166666666666667*L*intrhoz*r11*r22
                k += 1
                Mv[k] += 0.166666666666667*L*intrhoy*r11*r33 - 0.166666666666667*L*intrhoz*r11*r32
                k += 1
                Mv[k] += 0.166666666666667*L*r11**2*(intrhoy2 + intrhoz2)
                k += 1
                Mv[k] += 0.166666666666667*L*r11*r21*(intrhoy2 + intrhoz2)
                k += 1
                Mv[k] += 0.166666666666667*L*r11*r31*(intrhoy2 + intrhoz2)
                k += 1
                Mv[k] += 0.333333333333333*L*intrhoy*r11*r13 - 0.333333333333333*L*intrhoz*r11*r12
                k += 1
                Mv[k] += 0.333333333333333*L*intrhoy*r11*r23 - 0.333333333333333*L*intrhoz*r11*r22
                k += 1
                Mv[k] += 0.333333333333333*L*intrhoy*r11*r33 - 0.333333333333333*L*intrhoz*r11*r32
                k += 1
                Mv[k] += 0.333333333333333*L*r11**2*(intrhoy2 + intrhoz2)
                k += 1
                Mv[k] += 0.333333333333333*L*r11*r21*(intrhoy2 + intrhoz2)
                k += 1
                Mv[k] += 0.333333333333333*L*r11*r31*(intrhoy2 + intrhoz2)
                k += 1
                Mv[k] += 0.166666666666667*L*intrhoy*r13*r21 - 0.166666666666667*L*intrhoz*r12*r21
                k += 1
                Mv[k] += 0.166666666666667*L*intrhoy*r21*r23 - 0.166666666666667*L*intrhoz*r21*r22
                k += 1
                Mv[k] += 0.166666666666667*L*intrhoy*r21*r33 - 0.166666666666667*L*intrhoz*r21*r32
                k += 1
                Mv[k] += 0.166666666666667*L*r11*r21*(intrhoy2 + intrhoz2)
                k += 1
                Mv[k] += 0.166666666666667*L*r21**2*(intrhoy2 + intrhoz2)
                k += 1
                Mv[k] += 0.166666666666667*L*r21*r31*(intrhoy2 + intrhoz2)
                k += 1
                Mv[k] += 0.333333333333333*L*intrhoy*r13*r21 - 0.333333333333333*L*intrhoz*r12*r21
                k += 1
                Mv[k] += 0.333333333333333*L*intrhoy*r21*r23 - 0.333333333333333*L*intrhoz*r21*r22
                k += 1
                Mv[k] += 0.333333333333333*L*intrhoy*r21*r33 - 0.333333333333333*L*intrhoz*r21*r32
                k += 1
                Mv[k] += 0.333333333333333*L*r11*r21*(intrhoy2 + intrhoz2)
                k += 1
                Mv[k] += 0.333333333333333*L*r21**2*(intrhoy2 + intrhoz2)
                k += 1
                Mv[k] += 0.333333333333333*L*r21*r31*(intrhoy2 + intrhoz2)
                k += 1
                Mv[k] += 0.166666666666667*L*intrhoy*r13*r31 - 0.166666666666667*L*intrhoz*r12*r31
                k += 1
                Mv[k] += 0.166666666666667*L*intrhoy*r23*r31 - 0.166666666666667*L*intrhoz*r22*r31
                k += 1
                Mv[k] += 0.166666666666667*L*intrhoy*r31*r33 - 0.166666666666667*L*intrhoz*r31*r32
                k += 1
                Mv[k] += 0.166666666666667*L*r11*r31*(intrhoy2 + intrhoz2)
                k += 1
                Mv[k] += 0.166666666666667*L*r21*r31*(intrhoy2 + intrhoz2)
                k += 1
                Mv[k] += 0.166666666666667*L*r31**2*(intrhoy2 + intrhoz2)
                k += 1
                Mv[k] += 0.333333333333333*L*intrhoy*r13*r31 - 0.333333333333333*L*intrhoz*r12*r31
                k += 1
                Mv[k] += 0.333333333333333*L*intrhoy*r23*r31 - 0.333333333333333*L*intrhoz*r22*r31
                k += 1
                Mv[k] += 0.333333333333333*L*intrhoy*r31*r33 - 0.333333333333333*L*intrhoz*r31*r32
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
