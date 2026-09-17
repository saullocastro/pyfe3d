#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: nonecheck=False
#cython: overflowcheck=False
#cython: embedsignature=True
#cython: infer_types=False
r"""
Shell property module (:mod:`pyfe3d.shellprop`)
==============================================================

Highly based on the `composites <https://saullocastro.github.io/composites/>`_. module.

.. currentmodule:: pyfe3d.shellprop

"""
from libc.math cimport fabs, sqrt

import numpy as np

DOUBLE = np.float64

cdef class LaminationParameters:
    r"""Lamination parameters

    Attributes
    ----------
    xiA1, xiA2, xiA3, xiA4 : float
        Lamination parameters `\xi_{Ai}` (in-plane)
    xiB1, xiB2, xiB3, xiB4 : float
        Lamination parameters `\xi_{Bi}` (in-plane coupling with bending)
    xiD1, xiD2, xiD3, xiD4 : float
        Lamination parameters `\xi_{Di}` (bending)
    xiAts1, xiAts2 : float
        Lamination parameters `\xi_{Ats,i}` (transverse shear)

    """
    def __init__(LaminationParameters self):
        self.xiA1=0; self.xiA2=0; self.xiA3=0; self.xiA4=0
        self.xiB1=0; self.xiB2=0; self.xiB3=0; self.xiB4=0
        self.xiD1=0; self.xiD2=0; self.xiD3=0; self.xiD4=0
        self.xiAts1=0; self.xiAts2=0


cdef class MatLamina:
    r"""
    Orthotropic material lamina

    Attributes
    ----------

    e1 : float
        Young Modulus in direction 1
    e2 : float
        Young Modulus in direction 2
    g12 : float
        in-plane shear modulus
    g13 : float
        transverse shear modulus for plane 1-Z
    g23 : float
        transverse shear modulus for plane 2-Z
    nu12 :
        Poisson's ratio 12
    nu13 :
        Poisson's ratio 13
    nu23 :
        Poisson's ratio 23
    nu21 :
        Poisson's ratio 21: use formula nu12/e1 = nu21/e2
    nu31 :
        Poisson's ratio 31: use formula nu31/e3 = nu13/e1
    nu32 :
        Poisson's ratio 32: use formula nu23/e2 = nu32/e3
    rho :
        especific mass (mass / volume)
    a1 :
        thermal expansion coeffiecient in direction 1
    a2 :
        thermal expansion coeffiecient in direction 2
    a3 :
        thermal expansion coeffiecient in direction 3
    tref :
        reference temperature
    st1,st2 :
        allowable tensile stresses for directions 1 and 2
    sc1,sc2 :
        allowable compressive stresses for directions 1 and 2
    ss12 :
        allowable in-plane stress for shear
    q11 :
        lamina constitutive constant 11
    q12 :
        lamina constitutive constant 12
    q13 :
        lamina constitutive constant 13
    q21 :
        lamina constitutive constant 21
    q22 :
        lamina constitutive constant 22
    q23 :
        lamina constitutive constant 23
    q31 :
        lamina constitutive constant 31
    q32 :
        lamina constitutive constant 32
    q33 :
        lamina constitutive constant 33
    q44 :
        lamina constitutive constant 44
    q55 :
        lamina constitutive constant 55
    q66 :
        lamina constitutive constant 66
    ci :
        lamina stiffness constants
    ui :
        lamina material invariants

    Notes
    -----
    For isotropic materials when the user defines `\nu` and `E`, `G` will be
    recaculated based on equation: `G = E/(2 \times (1+\nu))`; in a lower
    priority if the user defines `\nu` and `G`, `E` will be recaculated based
    on equation: `E = 2 \times (1+\nu) \times G`.

    """
    def __init__(MatLamina self):
        pass

    cpdef void rebuild(MatLamina self):
        r"""Update constitutive and invariant terms

        Reference:

            Reddy, J. N., Mechanics of laminated composite plates and shells.
            Theory and analysis. Second Edition. CRC Press, 2004.

        """
        cdef double e1, e2, e3, nu12, nu21, nu13, nu31, nu23, nu32, delta, den
        e1 = self.e1
        e2 = self.e2
        e3 = self.e3
        nu12 = self.nu12
        nu21 = self.nu21
        nu13 = self.nu13
        nu31 = self.nu31
        nu23 = self.nu23
        nu32 = self.nu32
        delta = (1-nu12*nu21-nu23*nu32-nu31*nu13-2*nu21*nu32*nu13)/(e1*e2)
        self.c11 = (1    - nu23*nu23)/(delta*e2)
        self.c12 = (nu21 + nu31*nu23)/(delta*e2)
        self.c13 = (nu31 + nu21*nu32)/(delta*e2)
        self.c22 = (1    - nu13*nu31)/(delta*e1)
        self.c23 = (nu32 + nu12*nu31)/(delta*e1)
        self.c33 = e3*(1    - nu12*nu21)/(delta*e1*e2)
        self.c44 = self.g23
        self.c55 = self.g13
        self.c66 = self.g12

        # from references:
        #   hansen_hvejsen_2007 page 43
        #
        #   Guerdal Z., R. T. Haftka and P. Hajela (1999), Design and
        #   Optimization of Laminated Composite Materials, Wiley-Interscience.
        den = (1 - self.nu12 * self.nu21
                 - self.nu13 * self.nu31
                 - self.nu23 * self.nu32
                 - self.nu12 * self.nu23 * self.nu31
                 - self.nu13 * self.nu21 * self.nu32)
        self.q11 = self.e1*(1         - self.nu23 * self.nu32) / den
        self.q12 = self.e1*(self.nu21 + self.nu23 * self.nu31) / den
        self.q13 = self.e1*(self.nu31 + self.nu21 * self.nu32) / den
        self.q21 = self.e2*(self.nu12 + self.nu13 * self.nu32) / den
        self.q22 = self.e2*(1         - self.nu13 * self.nu31) / den
        self.q23 = self.e2*(self.nu32 + self.nu12 * self.nu31) / den
        self.q31 = self.e3*(self.nu13 + self.nu12 * self.nu32) / den
        self.q32 = self.e3*(self.nu23 + self.nu13 * self.nu21) / den
        self.q33 = self.e3*(1         - self.nu12 * self.nu21) / den
        self.q66 = self.g12
        self.q44 = self.g23
        self.q55 = self.g13
        #
        # from reference:
        #   Jones R. M. (1999), Mechanics of Composite Materials, second edn,
        #   Taylor & Francis, Inc., 325 Chestnut Street, Philadelphia,
        #   PA 19106. ISBN 1-56032-712-X
        # slightly changed to include the transverse shear terms u6 and u7,
        #   taken from ABAQUS Example Problems Manual, vol1, example 1.2.2
        #   Laminated composite shell: buckling of a
        #   cylindrical panel with a circular hole
        #
        self.u1 = (3*self.q11 + 3*self.q22 + 2*self.q12 + 4*self.q66) / 8.
        self.u2 = (self.q11 - self.q22) / 2.
        self.u3 = (self.q11 + self.q22 - 2*self.q12 - 4*self.q66) / 8.
        self.u4 = (self.q11 + self.q22 + 6*self.q12 - 4*self.q66) / 8.
        self.u5 = (self.u1 - self.u4) / 2.
        self.u6 = (self.q44 + self.q55) / 2.
        self.u7 = (self.q44 - self.q55) / 2.

    cpdef void trace_normalize_plane_stress(MatLamina self):
        r"""Trace-normalize the lamina properties for plane stress

        Modify the original :class:`.MatLamina` object with a
        trace-normalization performed after calculating the trace according to
        Eq. 1 of reference:

            Melo, J. D. D., Bi, J., and Tsai, S. W., 2017, “A Novel
            Invariant-Based Design Approach to Carbon Fiber Reinforced
            Laminates,” Compos. Struct., 159, pp. 44–52.

        The trace calculated as `tr = Q_{11} + Q_{22} + 2Q_{66}`.  The
        universal in-plane stress stiffness components
        `Q_{11},Q_{12},Q_{22},Q_{44},Q_{55},Q_{66}` are divided by `tr`, and
        the invariants `U_1,U_2,U_3,U_4,U_5,U_6,U_7` are calculated with the
        normalized stiffnesses, such they also become trace-normalized
        invariants. These can be accessed using the ``u1,u2,u3,u4,u5,u6,u7``
        attributes.

        """
        cdef double tr
        tr = self.q11 + self.q22 + 2*self.q66
        self.q11 /= tr
        self.q12 /= tr
        self.q22 /= tr
        self.q44 /= tr
        self.q55 /= tr
        self.q66 /= tr
        self.u1 /= tr
        self.u2 /= tr
        self.u3 /= tr
        self.u4 /= tr
        self.u5 /= tr
        self.u6 /= tr
        self.u7 /= tr

    cpdef double [:, ::1] get_constitutive_matrix(MatLamina self):
        r"""Return the constitutive matrix
        """
        return np.array(
            [[self.c11, self.c12, self.c13,   0,   0,   0],
             [self.c12, self.c22, self.c23,   0,   0,   0],
             [self.c13, self.c23, self.c33,   0,   0,   0],
             [  0,   0,   0, self.c44,   0,   0],
             [  0,   0,   0,   0, self.c55,   0],
             [  0,   0,   0,   0,   0, self.c66]], dtype=DOUBLE)

    cpdef double [:, ::1] get_invariant_matrix(MatLamina self):
        r"""Return the invariant matrix
        """
        return np.array(
            [[self.u1,  self.u2,    0,  self.u3,   0],            # q11
             [self.u1, -self.u2,    0,  self.u3,   0],            # q22
             [self.u4,   0,    0, -self.u3,   0],                 # q12
             [self.u5,   0,    0, -self.u3,   0],                 # q66
             [ 0,   0, self.u2/2.,   0,  self.u3],                # q16
             [ 0,   0, self.u2/2.,   0, -self.u3],                # q26
             [self.u6,  self.u7,    0,   0,   0],                 # q44
             [ 0,   0, -self.u7,    0,   0],                      # q45
             [self.u6, -self.u7,    0,   0,   0]], dtype=DOUBLE)  # q55


cdef class Lamina:
    r"""
    Attributes
    ----------

    plyid : int
        Identificaiton of the composite lamina
    matlamina : :class:`.MatLamina` object
        A :class:`.MatLamina` object
    h : float
        Ply thickness
    thetadeg : float
        Ply angle in degrees

    """
    def __init__(Lamina self):
        pass

    cpdef void rebuild(Lamina self):
        r"""Update constitutive matrices

        Reference:

            Reddy, J. N., Mechanics of Laminated Composite Plates and
            Shells - Theory and Analysys. Second Edition. CRC PRESS, 2004.
        """
        cdef double thetarad, e1, e2, nu12, nu21, g12, g13, g23
        cdef double q11, q12, q22, q44, q55, q16, q26, q66
        cdef double cos2, cos3, cos4, sin2, sin3, sin4, sincos
        thetarad = deg2rad(self.thetadeg)
        self.cost = cos(thetarad)
        self.cos2t = cos(2*thetarad)
        self.cos4t = cos(4*thetarad)
        self.sint = sin(thetarad)
        self.sin2t = sin(2*thetarad)
        self.sin4t = sin(4*thetarad)
        cos2 = self.cost**2
        cos3 = self.cost**3
        cos4 = self.cost**4
        sin2 = self.sint**2
        sin3 = self.sint**3
        sin4 = self.sint**4
        sincos = self.sint*self.cost
        # STRAINS
        # different from stress due to:
        #     2*e12 = e6    2*e13 = e5    2*e23 = e4
        # to laminate
        # self.Rstrain = np.transpose(self.Tstress)
        # to lamina
        # self.Tstrain = np.transpose(self.Rstress)
        e1   = self.matlamina.e1
        e2   = self.matlamina.e2
        nu12 = self.matlamina.nu12
        nu21 = self.matlamina.nu21
        g12  = self.matlamina.g12
        g13  = self.matlamina.g13
        g23  = self.matlamina.g23

        # plane stress
        #TODO plane strain
        q11  = e1/(1-nu12*nu21)
        q12  = nu12*e2/(1-nu12*nu21)
        q22  = e2/(1-nu12*nu21)
        q44  = g23
        q55  = g13
        q16 = 0
        q26 = 0
        q66  = g12

        self.q11L = q11*cos4 + 2*(q12 + 2*q66)*sin2*cos2 + q22*sin4
        self.q12L = (q11 + q22 - 4*q66)*sin2*cos2 + q12*(sin4 + cos4)
        self.q22L = q11*sin4 + 2*(q12 + 2*q66)*sin2*cos2 + q22*cos4
        self.q16L = (q11 - q12 - 2*q66)*self.sint*cos3 + (q12 - q22 + 2*q66)*sin3*self.cost
        self.q26L = (q11 - q12 - 2*q66)*sin3*self.cost + (q12 - q22 + 2*q66)*self.sint*cos3
        self.q66L = (q11 + q22 - 2*q12 - 2*q66)*sin2*cos2 + q66*(sin4 + cos4)
        self.q44L = q44*cos2 + q55*sin2
        self.q45L = (q55 - q44)*sincos
        self.q55L = q55*cos2 + q44*sin2

        #TODO add the thermal coeficient terms when calculating the
        #     stresses... to take into account eventual thermal expansions or
        #     contractions

    cpdef double [:, ::1] get_transf_matrix_displ_to_laminate(Lamina self):
        r"""Return displacement transformation matrix from lamina to laminate"""
        return np.array([[ self.cost, self.sint, 0],
                         [-self.sint, self.cost, 0],
                         [   0,     0, 1]], dtype=DOUBLE)

    cpdef double [:, ::1] get_constitutive_matrix(Lamina self):
        r"""Return the constitutive matrix"""
        return np.array([[self.q11L, self.q12L, self.q16L,    0,    0],
                         [self.q12L, self.q22L, self.q26L,    0,    0],
                         [self.q16L, self.q26L, self.q66L,    0,    0],
                         [   0,    0,    0, self.q44L, self.q45L],
                         [   0,    0,    0, self.q45L, self.q55L]], dtype=DOUBLE)

    cpdef double [:, ::1] get_transf_matrix_stress_to_lamina(Lamina self):
        r"""Return stress transformation matrix from laminate to lamina"""
        cdef double cos2, sin2, sincos
        cos2 = self.cost**2
        sin2 = self.sint**2
        sincos = self.sint*self.cost
        return np.array(
            [[ cos2, sin2, 0, 0, 0, self.sin2t],
             [ sin2, cos2, 0, 0, 0, -self.sin2t],
             [ 0, 0, 1, 0, 0, 0],
             [ 0, 0, 0, self.cost, -self.sint, 0],
             [ 0, 0, 0, self.sint,  self.cost, 0],
             [-sincos, sincos, 0, 0, 0, cos2-sin2]], dtype=DOUBLE)

    cpdef double [:, ::1] get_transf_matrix_stress_to_laminate(Lamina self):
        r"""Return stress transformation matrix from lamina to laminate"""
        cdef double cos2, sin2, sincos
        cos2 = self.cost**2
        sin2 = self.sint**2
        sincos = self.sint*self.cost
        return np.array(
            [[ cos2, sin2, 0, 0,   0, -self.sin2t],
             [ sin2, cos2, 0, 0,   0, self.sin2t],
             [ 0, 0, 1, 0, 0, 0],
             [ 0, 0, 0,  self.cost, self.sint, 0],
             [ 0, 0, 0, -self.sint, self.cost, 0],
             [sincos, -sincos, 0, 0, 0, cos2-sin2]], dtype=DOUBLE)


cdef int PLYDATA_SIZE = 10


cdef void _rotate_ply(double *pd, double m11, double m12, double m21,
        double m22, double *q) noexcept nogil:
    r"""Ply stiffnesses rotated from the material to the element frame

    ``pd`` contains ``h, q11L, q12L, q16L, q22L, q26L, q66L, q44L, q45L, q55L``
    in the material frame. The output ``q`` contains ``q11, q12, q16, q22,
    q26, q66, q44, q45, q55`` in the element frame, with `C^e = T_\sigma C
    T_\sigma^T` and `C_s^e = T_s C_s T_s^T`, `T_s = [[m_{22}, m_{21}],
    [m_{12}, m_{11}]]`.

    """
    cdef int i, j, k, l
    cdef double C[9]
    cdef double T[9]
    cdef double Ce[9]
    cdef double s44, s45, s55
    if m12 == 0:
        for i in range(9):
            q[i] = pd[i+1]
        return
    C[0] = pd[1]; C[1] = pd[2]; C[2] = pd[3]
    C[3] = pd[2]; C[4] = pd[4]; C[5] = pd[5]
    C[6] = pd[3]; C[7] = pd[5]; C[8] = pd[6]
    T[0] = m11*m11; T[1] = m12*m12; T[2] = 2*m11*m12
    T[3] = m21*m21; T[4] = m22*m22; T[5] = 2*m21*m22
    T[6] = m11*m21; T[7] = m12*m22; T[8] = m11*m22 + m12*m21
    for i in range(3):
        for j in range(3):
            Ce[3*i+j] = 0.
            for k in range(3):
                for l in range(3):
                    Ce[3*i+j] += T[3*i+k]*C[3*k+l]*T[3*j+l]
    q[0] = Ce[0]; q[1] = Ce[1]; q[2] = Ce[2]
    q[3] = Ce[4]; q[4] = Ce[5]; q[5] = Ce[8]
    s44 = pd[7]; s45 = pd[8]; s55 = pd[9]
    q[6] = m22*m22*s44 + 2*m22*m21*s45 + m21*m21*s55
    q[7] = m22*m12*s44 + (m22*m11 + m21*m12)*s45 + m21*m11*s55
    q[8] = m12*m12*s44 + 2*m12*m11*s45 + m11*m11*s55


cdef int _rohwer(int N, double *plydata, double offset, double m11,
        double m12, double m21, double m22, double *z, double *fcoef,
        double *S) noexcept nogil:
    r"""Equilibrium transverse shear distribution and compliance (Rohwer, 1988)

    The plies are rotated from the material to the element frame using
    `m_{11}`, `m_{12}`, `m_{21}`, `m_{22}`, such that the two cylindrical
    bending states are posed along the element axes.

    Parameters
    ----------
    N : int
        Number of plies.
    plydata : pointer
        ``N*PLYDATA_SIZE`` values, see :func:`_rotate_ply`.
    offset : double
        Offset of the laminate mid-surface with respect to the reference
        surface.
    m11, m12, m21, m22 : double
        In-plane rotation from the material to the element frame.
    z : pointer
        Output with the ``N + 1`` ply interfaces, or ``NULL``.
    fcoef : pointer
        Output with ``N*3*2*2`` coefficients, such that `f^{(k)}(z) = F_0 + z
        F_1 + z^2 F_2`, or ``NULL``.
    S : pointer
        Output with the transverse shear compliance ``S00, S01, S11``, or
        ``NULL`` to skip its computation.

    Returns
    -------
    status : int
        ``0`` if successful, ``1`` for a singular ply `C_s^{(k)}`, ``2`` for a
        singular ABD matrix.

    """
    cdef int i, j, k, p, ig
    cdef double h, za, zb, zk, dz1, dz2, dz3, zm, dz, zg, zg2, wdz
    cdef double tmp, fac, det, i44, i45, i55
    cdef double f00, f01, f10, f11, g00, g01, g10, g11
    cdef double F[12]
    cdef double q[9]
    cdef double c1[3]
    cdef double c2[3]
    cdef double c3[3]
    cdef double ABD[36]
    cdef double Hs[36]
    cdef double scale[6]
    cdef double p1[6]
    cdef double q1[6]
    cdef double p2[6]
    cdef double q2[6]
    cdef double p1_prev[6]
    cdef double q1_prev[6]
    cdef double p2_prev[6]
    cdef double q2_prev[6]
    cdef double acc_x[6]
    cdef double acc_y[6]
    cdef double gp[3]
    cdef double gw[3]

    # NOTE 3-point Gauss-Legendre, exact for polynomials up to degree 5
    gp[0] = -0.7745966692414834; gp[1] = 0.; gp[2] = 0.7745966692414834
    gw[0] = 5./9.; gw[1] = 8./9.; gw[2] = 5./9.

    h = 0.
    for k in range(N):
        h += plydata[PLYDATA_SIZE*k]

    # ABD = [[A, B], [B, D]] from the rotated plies
    for i in range(36):
        ABD[i] = 0.
    za = -h/2. + offset
    for k in range(N):
        zb = za + plydata[PLYDATA_SIZE*k]
        _rotate_ply(&plydata[PLYDATA_SIZE*k], m11, m12, m21, m22, q)
        dz1 = zb - za
        dz2 = (zb*zb - za*za)/2.
        dz3 = (zb*zb*zb - za*za*za)/3.
        c1[0] = q[0]; c1[1] = q[1]; c1[2] = q[2]
        c2[0] = q[1]; c2[1] = q[3]; c2[2] = q[4]
        c3[0] = q[2]; c3[1] = q[4]; c3[2] = q[5]
        for j in range(3):
            ABD[0*6 + j] += c1[j]*dz1
            ABD[1*6 + j] += c2[j]*dz1
            ABD[2*6 + j] += c3[j]*dz1
            ABD[0*6 + j+3] += c1[j]*dz2
            ABD[1*6 + j+3] += c2[j]*dz2
            ABD[2*6 + j+3] += c3[j]*dz2
            ABD[3*6 + j] += c1[j]*dz2
            ABD[4*6 + j] += c2[j]*dz2
            ABD[5*6 + j] += c3[j]*dz2
            ABD[3*6 + j+3] += c1[j]*dz3
            ABD[4*6 + j+3] += c2[j]*dz3
            ABD[5*6 + j+3] += c3[j]*dz3
        za = zb

    # Hstar = inv(ABD) by Gauss-Jordan elimination with partial pivoting,
    # after a symmetric diagonal scaling such that the singularity check does
    # not depend on the units
    for i in range(6):
        if not ABD[7*i] > 0:
            return 2
        scale[i] = 1./sqrt(ABD[7*i])
    for i in range(6):
        for j in range(6):
            ABD[6*i+j] *= scale[i]*scale[j]
            Hs[6*i+j] = 1. if i == j else 0.
    for k in range(6):
        p = k
        for i in range(k+1, 6):
            if fabs(ABD[6*i+k]) > fabs(ABD[6*p+k]):
                p = i
        if not fabs(ABD[6*p+k]) > 1e-12:
            return 2
        if p != k:
            for j in range(6):
                tmp = ABD[6*k+j]; ABD[6*k+j] = ABD[6*p+j]; ABD[6*p+j] = tmp
                tmp = Hs[6*k+j]; Hs[6*k+j] = Hs[6*p+j]; Hs[6*p+j] = tmp
        tmp = ABD[6*k+k]
        for j in range(6):
            ABD[6*k+j] /= tmp
            Hs[6*k+j] /= tmp
        for i in range(6):
            if i == k:
                continue
            fac = ABD[6*i+k]
            if fac == 0:
                continue
            for j in range(6):
                ABD[6*i+j] -= fac*ABD[6*k+j]
                Hs[6*i+j] -= fac*Hs[6*k+j]
    for i in range(6):
        for j in range(6):
            Hs[6*i+j] *= scale[i]*scale[j]

    for j in range(6):
        acc_x[j] = 0.; acc_y[j] = 0.
        p1_prev[j] = 0.; q1_prev[j] = 0.; p2_prev[j] = 0.; q2_prev[j] = 0.
    if S != NULL:
        S[0] = 0.; S[1] = 0.; S[2] = 0.

    za = -h/2. + offset
    if z != NULL:
        z[0] = za
    for k in range(N):
        zb = za + plydata[PLYDATA_SIZE*k]
        if z != NULL:
            z[k+1] = zb
        _rotate_ply(&plydata[PLYDATA_SIZE*k], m11, m12, m21, m22, q)
        c1[0] = q[0]; c1[1] = q[1]; c1[2] = q[2]
        c2[0] = q[1]; c2[1] = q[3]; c2[2] = q[4]
        zk = za
        for j in range(6):
            # c Pi(z) = z c Hstar[:3, :] + z^2/2 c Hstar[3:, :]
            p1[j] = c1[0]*Hs[0*6+j] + c1[1]*Hs[1*6+j] + c1[2]*Hs[2*6+j]
            q1[j] = c1[0]*Hs[3*6+j] + c1[1]*Hs[4*6+j] + c1[2]*Hs[5*6+j]
            p2[j] = c2[0]*Hs[0*6+j] + c2[1]*Hs[1*6+j] + c2[2]*Hs[2*6+j]
            q2[j] = c2[0]*Hs[3*6+j] + c2[1]*Hs[4*6+j] + c2[2]*Hs[5*6+j]
            # a^(k) = sum_{i=1..k} (c^(i) - c^(i-1)) Pi(z_i)
            acc_x[j] += zk*(p1[j] - p1_prev[j]) + zk*zk/2.*(q1[j] - q1_prev[j])
            acc_y[j] += zk*(p2[j] - p2_prev[j]) + zk*zk/2.*(q2[j] - q2_prev[j])
            p1_prev[j] = p1[j]
            q1_prev[j] = q1[j]
            p2_prev[j] = p2[j]
            q2_prev[j] = q2[j]
        # NOTE F[4*power + 2*row + col], with row 0 for tau_yz, row 1 for
        #      tau_xz, col 0 for Q_y and col 1 for Q_x
        # tau_yz row: (a_y - c_2 Pi(z)) Ly, picking components 4 (Q_y) and 5 (Q_x)
        F[0] = acc_y[4]; F[4] = -p2[4]; F[8] = -q2[4]/2.
        F[1] = acc_y[5]; F[5] = -p2[5]; F[9] = -q2[5]/2.
        # tau_xz row: (a_x - c_1 Pi(z)) Lx, picking components 5 (Q_y) and 3 (Q_x)
        F[2] = acc_x[5]; F[6] = -p1[5]; F[10] = -q1[5]/2.
        F[3] = acc_x[3]; F[7] = -p1[3]; F[11] = -q1[3]/2.
        if fcoef != NULL:
            for i in range(12):
                fcoef[12*k + i] = F[i]

        if S != NULL:
            det = q[6]*q[8] - q[7]*q[7]
            if q[6] <= 0 or q[8] <= 0 or det <= 1e-12*q[6]*q[8]:
                return 1
            i44 = q[8]/det
            i45 = -q[7]/det
            i55 = q[6]/det
            zm = (za + zb)/2.
            dz = (zb - za)/2.
            for ig in range(3):
                zg = zm + dz*gp[ig]
                zg2 = zg*zg
                wdz = gw[ig]*dz
                f00 = F[0] + zg*F[4] + zg2*F[8]
                f01 = F[1] + zg*F[5] + zg2*F[9]
                f10 = F[2] + zg*F[6] + zg2*F[10]
                f11 = F[3] + zg*F[7] + zg2*F[11]
                # g = inv(Cs) f
                g00 = i44*f00 + i45*f10
                g01 = i44*f01 + i45*f11
                g10 = i45*f00 + i55*f10
                g11 = i45*f01 + i55*f11
                # S += f^T inv(Cs) f
                S[0] += wdz*(f00*g00 + f10*g10)
                S[1] += wdz*(f00*g01 + f10*g11)
                S[2] += wdz*(f01*g01 + f11*g11)
        za = zb

    return 0


_SHELLPROP_STATE = (
    'A11', 'A12', 'A16', 'A22', 'A26', 'A66',
    'B11', 'B12', 'B16', 'B22', 'B26', 'B66',
    'D11', 'D12', 'D16', 'D22', 'D26', 'D66',
    'A44', 'A45', 'A55',
    'Abar44', 'Abar45', 'Abar55',
    'Abarbar44', 'Abarbar45', 'Abarbar55',
    'e1', 'e2', 'g12', 'nu12', 'nu21',
    'scf_k13', 'scf_k23', 'h', 'offset', 'intrho', 'intrhoz', 'intrhoz2',
    'plies', 'stack', 'shear_correction')


def _singular_Cs_error(int k, Lamina ply):
    return ValueError('Ply %d (plyid=%d, thetadeg=%g) has a singular '
            'transverse shear constitutive matrix Cs = [[q44L, q45L], '
            '[q45L, q55L]] = [[%g, %g], [%g, %g]]; g13 and g23 must be '
            'positive' % (k, ply.plyid, ply.thetadeg, ply.q44L, ply.q45L,
                ply.q45L, ply.q55L))


cdef class ShellProp:
    r"""
    Attributes
    ----------

    plies : list
        List of plies
    stack : list
        List of angles for each ply
    h : float
        Total thickness of the laminate
    offset : float
        Offset at the normal direction
    e1, e2 : float
        Equivalent laminate moduli in directions 1 and 2
    g12 : float
        Equivalent laminate shear modulus in the 12 direction
    nu12, nu21 : float
        Equivalent laminate Poisson ratios in the 12 and 21 directions
    A44, A45, A55 : float
        Transverse shear stiffnesses of the first-order shear deformation
        theory (FSDT) in the material coordinate system, **with the shear
        correction already applied** according to ``shear_correction``. The
        elements use them directly, without any shear correction factor. See
        :meth:`.calc_transverse_shear_stiffness` and
        :meth:`.calc_Ats_element` for how they are brought to the element
        coordinate system.
    Abar44, Abar45, Abar55 : float
        Constant-strain (uncorrected) transverse shear stiffnesses
        `\bar{A}_{ts} = \sum_k C_s^{(k)} h_k`.
    Abarbar44, Abarbar45, Abarbar55 : float
        Constant-stress transverse shear stiffnesses `\bar{\bar{A}}_{ts} = h^2
        [\sum_k (C_s^{(k)})^{-1} h_k]^{-1}`, for comparison only. Equal to
        ``nan`` when a ply has a singular `C_s^{(k)}`.
    shear_correction : str or None
        Method used to obtain ``A44``, ``A45``, ``A55`` from the ply data, see
        :meth:`.calc_transverse_shear_stiffness`. Default is ``'rohwer'``.
    scf_k13, scf_k23 : float
        Reported shear correction ratios ``A55/Abar55`` and ``A44/Abar44``.
        They are informative only, the correction is already inside ``A44``,
        ``A45``, ``A55``.
    intrho : float
        Integral `\int_{-h/2+offset}^{+h/2+offset} \rho(z) dz`, used in
        equivalent single layer finite element mass matrices
    intrhoz : float
        Integral `\int_{-h/2+offset}^{+h/2+offset} \rho(z)z dz`, used in
        equivalent single layer finite element mass matrices
    intrhoz2 : float
        Integral `\int_{-h/2+offset}^{+h/2+offset} \rho(z)z^2 dz`, used in
        equivalent single layer finite element mass matrices

    """
    def __init__(ShellProp self):
        self.h = 0.
        self.e1 = 0.
        self.e2 = 0.
        self.g12 = 0.
        self.nu12 = 0.
        self.nu21 = 0.
        self.offset = 0.
        self.scf_k13 = 1.
        self.scf_k23 = 1.
        self.intrho = 0.
        self.intrhoz = 0.
        self.intrhoz2 = 0.
        self.plies = []
        self.stack = []
        self.shear_correction = 'rohwer'
        self._ts_element_frame = False
        self._ts_ready = False
        self._ts_nplies = 0

    def __reduce__(ShellProp self):
        state = {name: getattr(self, name) for name in _SHELLPROP_STATE}
        state['_ts_element_frame'] = self._ts_element_frame
        state['_ts_nplies'] = self._ts_nplies
        state['_ts_offset'] = self._ts_offset
        if self._ts_element_frame:
            state['_ts_plydata'] = np.array(self._ts_plydata, dtype=DOUBLE)
        return (ShellProp, (), state)

    def __setstate__(ShellProp self, state):
        for name in _SHELLPROP_STATE:
            setattr(self, name, state[name])
        self._ts_ready = False
        self._ts_element_frame = state['_ts_element_frame']
        self._ts_nplies = state['_ts_nplies']
        self._ts_offset = state['_ts_offset']
        if self._ts_element_frame:
            self._ts_plydata = np.ascontiguousarray(state['_ts_plydata'],
                                                    dtype=DOUBLE)

    cdef double [:, ::1] get_A(ShellProp self):
        return np.array([[self.A11, self.A12, self.A16],
                         [self.A12, self.A22, self.A26],
                         [self.A16, self.A26, self.A66]], dtype=DOUBLE)
    cdef double [:, ::1] get_B(ShellProp self):
        return np.array([[self.B11, self.B12, self.B16],
                         [self.B12, self.B22, self.B26],
                         [self.B16, self.B26, self.B66]], dtype=DOUBLE)
    cdef double [:, ::1] get_D(ShellProp self):
        return np.array([[self.D11, self.D12, self.D16],
                         [self.D12, self.D22, self.D26],
                         [self.D16, self.D26, self.D66]], dtype=DOUBLE)
    cdef double [:, ::1] get_Ats(ShellProp self):
        return np.array([[self.A44, self.A45],
                         [self.A45, self.A55]], dtype=DOUBLE)
    cdef double [:, ::1] get_Abar_ts(ShellProp self):
        return np.array([[self.Abar44, self.Abar45],
                         [self.Abar45, self.Abar55]], dtype=DOUBLE)
    cdef double [:, ::1] get_Abarbar_ts(ShellProp self):
        return np.array([[self.Abarbar44, self.Abarbar45],
                         [self.Abarbar45, self.Abarbar55]], dtype=DOUBLE)
    cdef double [:, ::1] get_ABD(ShellProp self):
        return np.array([[self.A11, self.A12, self.A16, self.B11, self.B12, self.B16],
                         [self.A12, self.A22, self.A26, self.B12, self.B22, self.B26],
                         [self.A16, self.A26, self.A66, self.B16, self.B26, self.B66],
                         [self.B11, self.B12, self.B16, self.D11, self.D12, self.D16],
                         [self.B12, self.B22, self.B26, self.D12, self.D22, self.D26],
                         [self.B16, self.B26, self.B66, self.D16, self.D26, self.D66]], dtype=DOUBLE)
    @property
    def A(self):
        return np.asarray(self.get_A())
    @property
    def B(self):
        return np.asarray(self.get_B())
    @property
    def D(self):
        return np.asarray(self.get_D())
    @property
    def Ats(self):
        r"""Transverse shear stiffness matrix ``[[A44, A45], [A45, A55]]``

        Index 4 corresponds to `yz` and index 5 to `xz`, such that `\{Q_y,
        Q_x\}^T = A_{ts} \{\gamma_{yz}, \gamma_{xz}\}^T`. The shear correction
        is already applied, see :meth:`.calc_transverse_shear_stiffness`.

        """
        return np.asarray(self.get_Ats())
    @property
    def Abar_ts(self):
        r"""Constant-strain ``[[Abar44, Abar45], [Abar45, Abar55]]``"""
        return np.asarray(self.get_Abar_ts())
    @property
    def Abarbar_ts(self):
        r"""Constant-stress ``[[Abarbar44, Abarbar45], [Abarbar45, Abarbar55]]``"""
        return np.asarray(self.get_Abarbar_ts())
    @property
    def ABD(self):
        return np.asarray(self.get_ABD())


    cdef void get_Ats_element(ShellProp self, double m11, double m12, double
            m21, double m22, double *A44, double *A45, double *A55) noexcept nogil:
        r"""Transverse shear stiffness in the element coordinate system

        Used by the shell elements, with `m_{11}`, `m_{12}`, `m_{21}`,
        `m_{22}` being the in-plane rotation from the material to the element
        coordinate system. See :meth:`.calc_Ats_element` for details.

        """
        cdef int status
        cdef double S[3]
        cdef double detS, t44, t45, t55
        if m12 == 0:
            A44[0] = self.A44
            A45[0] = self.A45
            A55[0] = self.A55
            return
        if self._ts_element_frame:
            status = _rohwer(self._ts_nplies, &self._ts_plydata[0, 0],
                             self._ts_offset, m11, m12, m21, m22, NULL, NULL,
                             S)
            detS = S[0]*S[2] - S[1]*S[1]
            if status == 0 and detS > 0:
                A44[0] = S[2]/detS
                A45[0] = -S[1]/detS
                A55[0] = S[0]/detS
                return
        # NOTE tensor rotation A_e = T_s A T_s^T, T_s = [[m22, m21], [m12, m11]]
        t44 = self.A44
        t45 = self.A45
        t55 = self.A55
        A44[0] = m22*m22*t44 + 2*m22*m21*t45 + m21*m21*t55
        A45[0] = m22*m12*t44 + (m22*m11 + m21*m12)*t45 + m21*m11*t55
        A55[0] = m12*m12*t44 + 2*m12*m11*t45 + m11*m11*t55


    def calc_Ats_element(ShellProp self, double thetadeg):
        r"""Transverse shear stiffness in an element coordinate system

        The element coordinate system is such that the material direction
        makes an angle `\theta` with the element `x` axis, measured towards
        the element `y` axis. This is the same rotation used by the shell
        elements, where `m_{11} = \cos\theta`, `m_{12} = -\sin\theta`, `m_{21}
        = \sin\theta` and `m_{22} = \cos\theta`.

        The transverse shear stiffness obtained with the equilibrium approach
        of Rohwer (1988) is not invariant to a rotation of the reference
        frame, because the two cylindrical bending states are tied to the
        `x` and `y` axes. Therefore, when ``shear_correction='rohwer'`` and
        the plies are available, the plies are rotated to the element frame,
        i.e. all ply angles are shifted by `\theta`, and the stiffness is
        re-evaluated in that frame, such that the assumed static state and the
        element kinematics refer to the same pair of directions.

        In all other cases, i.e. ``shear_correction`` ``'constant'``,
        ``'vlachoutsis'`` or ``None``, or when no plies exist, e.g. for a
        property created from lamination parameters or with ``A44``, ``A45``,
        ``A55`` given directly, the stiffness is rotated as a second-order
        tensor:

        .. math::

            A_{ts}^e = T_s A_{ts} T_s^T \qquad T_s = \begin{bmatrix}
            \cos\theta & \sin\theta \\ -\sin\theta & \cos\theta \end{bmatrix}

        which is exact for the constant-strain stiffness.

        .. note:: Changes to ``A44``, ``A45``, ``A55`` made after
                  :meth:`.calc_constitutive_matrix` are only used for
                  elements with `m_{12} = 0` when the plies are available
                  and ``shear_correction='rohwer'``, because the stiffness
                  is otherwise re-evaluated from the plies.

        Parameters
        ----------
        thetadeg : float
            Angle `\theta` in degrees.

        Returns
        -------
        Ats : np.ndarray
            Matrix ``[[A44, A45], [A45, A55]]`` in the element coordinate
            system.

        """
        cdef double A44, A45, A55, thetarad, c, s
        thetarad = deg2rad(thetadeg)
        c = cos(thetarad)
        s = sin(thetarad)
        self.get_Ats_element(c, -s, s, c, &A44, &A45, &A55)
        return np.array([[A44, A45], [A45, A55]], dtype=DOUBLE)


    cdef void _store_ply_data(ShellProp self) except *:
        cdef int k, N
        cdef Lamina ply
        N = <int>len(self.plies)
        plydata = np.zeros((max(N, 1), PLYDATA_SIZE), dtype=DOUBLE)
        for k in range(N):
            ply = self.plies[k]
            plydata[k, 0] = ply.h
            plydata[k, 1] = ply.q11L
            plydata[k, 2] = ply.q12L
            plydata[k, 3] = ply.q16L
            plydata[k, 4] = ply.q22L
            plydata[k, 5] = ply.q26L
            plydata[k, 6] = ply.q66L
            plydata[k, 7] = ply.q44L
            plydata[k, 8] = ply.q45L
            plydata[k, 9] = ply.q55L
        self._ts_plydata = plydata
        self._ts_nplies = N
        self._ts_offset = self.offset


    cpdef void calc_transverse_shear_stiffness(ShellProp self) except *:
        r"""Update the transverse shear stiffnesses ``A44``, ``A45``, ``A55``

        Called at the end of :meth:`.calc_constitutive_matrix`. The attributes
        ``A44``, ``A45``, ``A55`` are the transverse shear stiffnesses of the
        first-order shear deformation theory (FSDT) **with the shear
        correction already applied**, and no shear correction factor is
        applied to them by the elements.

        Conventions: `\{\tau_{yz}, \tau_{xz}\}^T`, `\{Q_y, Q_x\}^T`,
        `A_{ts} = [[A_{44}, A_{45}], [A_{45}, A_{55}]]`, index 4 corresponds
        to `yz` and index 5 to `xz`. The coordinate `z` is measured from the
        reference surface, with the plies running from `z_1 = -h/2 +
        offset` to `z_{N+1} = +h/2 + offset`, and `C_s^{(k)} = [[q_{44L},
        q_{45L}], [q_{45L}, q_{55L}]]`, which is in general a full matrix.

        The method is selected by the attribute ``shear_correction``:

        - ``'rohwer'`` (default): equilibrium approach of Rohwer (1988). The
          transverse shear stresses are obtained from the equilibrium of two
          cylindrical bending states, with zero tractions at the bottom and
          top faces and continuity at every interface, `\{\tau_{yz},
          \tau_{xz}\}^T = f^{(k)}(z) \{Q_y, Q_x\}^T`. The 2x2 stiffness is
          obtained from the complementary energy:

          .. math::

              A_{ts} = \left[ \sum_k \int_{z_k}^{z_{k+1}} f^{(k)T}
              \left(C_s^{(k)}\right)^{-1} f^{(k)} dz \right]^{-1}

          The integrand is a polynomial of degree 4 in `z` within each ply,
          such that the 3-point Gauss-Legendre rule used per ply is exact.
          The method is valid for arbitrary anisotropic and unsymmetric
          laminates, and the result does not depend on ``offset``. The result
          is not invariant to a rotation of the reference frame, because the
          two cylindrical bending states are tied to the `x` and `y` axes.
          For this reason, the shell elements re-evaluate it in the element
          coordinate system, see :meth:`.calc_Ats_element`.

        - ``'vlachoutsis'``: the scalar factors `k_{13}`, `k_{23}` of
          Vlachoutsis (1992) are applied to the constant-strain stiffness,
          ``A55 = k13*Abar55``, ``A44 = k23*Abar44``, and the ad-hoc ``A45 =
          (k13 + k23)/2*Abar45``, which cannot represent the coupling of
          angle-ply laminates with ``Abar45 = 0``. The factors use `C_{11}`
          and `C_{22}` of each ply and the direction-wise neutral surfaces,
          being exact only for specially orthotropic plies.

        - ``'constant'``: `k = 5/6`, i.e. ``A44 = 5/6*Abar44``, ``A45 =
          5/6*Abar45``, ``A55 = 5/6*Abar55``.

        - ``None``: no correction, ``A44 = Abar44``, ``A45 = Abar45``, ``A55 =
          Abar55``.

        The following attributes are also updated: ``Abar44``, ``Abar45``,
        ``Abar55`` (constant strain), ``Abarbar44``, ``Abarbar45``,
        ``Abarbar55`` (constant stress, ``nan`` if a ply has a singular
        `C_s^{(k)}`), and the ratios ``scf_k13 = A55/Abar55`` and ``scf_k23 =
        A44/Abar44``, which are informative only.

        References:

            Rohwer, K. "Improved transverse shear stiffness for layered
            finite elements", DFVLR-FB 88-32, 1988.

            Vlachoutsis, S. "Shear correction factors for plates and shells",
            Int. Journal for Numerical Methods in Engineering, Vol. 33,
            1537-1552, 1992.

        Raises
        ------
        ValueError
            If ``shear_correction`` is not recognized; if a ply has a
            singular transverse shear constitutive matrix `C_s^{(k)}`, e.g.
            ``g13 = 0`` or ``g23 = 0`` (``'rohwer'`` and ``'vlachoutsis'``);
            or if the ABD matrix of the laminate is singular (``'rohwer'``).

        """
        cdef int k, ig, N, alpha, singular_ply, status
        cdef double h, det, za, zb, zm, dz, zg
        cdef double S[3]
        cdef double Sbb44, Sbb45, Sbb55, detS
        cdef double Dk, Gk, num, den, zn, R, d, I, gz, gza, kappa
        cdef double k13, k23
        cdef double [::1] z
        cdef double [:, :, :, ::1] fcoef
        cdef double gp[3]
        cdef double gw[3]
        cdef Lamina ply

        # 3-point Gauss-Legendre, exact for polynomials up to degree 5
        gp[0] = -0.7745966692414834; gp[1] = 0.; gp[2] = 0.7745966692414834
        gw[0] = 5./9.; gw[1] = 8./9.; gw[2] = 5./9.

        mode = self.shear_correction
        if not (mode is None or mode in ('rohwer', 'vlachoutsis', 'constant')):
            raise ValueError("shear_correction must be 'rohwer', "
                             "'vlachoutsis', 'constant' or None, got %r"
                             % (mode,))

        self._ts_ready = False
        self._ts_element_frame = False
        self.A44 = 0; self.A45 = 0; self.A55 = 0
        self.Abar44 = 0; self.Abar45 = 0; self.Abar55 = 0
        self.Abarbar44 = 0; self.Abarbar45 = 0; self.Abarbar55 = 0
        N = <int>len(self.plies)
        if N == 0:
            return
        self._store_ply_data()

        # constant-strain and constant-stress stiffnesses
        h = 0.
        singular_ply = -1
        Sbb44 = 0; Sbb45 = 0; Sbb55 = 0
        z = np.zeros(N + 1, dtype=DOUBLE)
        for k in range(N):
            ply = self.plies[k]
            h += ply.h
            z[k+1] = h
            self.Abar44 += ply.q44L*ply.h
            self.Abar45 += ply.q45L*ply.h
            self.Abar55 += ply.q55L*ply.h
            det = ply.q44L*ply.q55L - ply.q45L*ply.q45L
            if (ply.q44L <= 0 or ply.q55L <= 0
                    or det <= 1e-12*ply.q44L*ply.q55L):
                if singular_ply < 0:
                    singular_ply = k
            else:
                Sbb44 += ply.q55L/det*ply.h
                Sbb45 += -ply.q45L/det*ply.h
                Sbb55 += ply.q44L/det*ply.h
        for k in range(N + 1):
            z[k] += -h/2. + self.offset
        if singular_ply < 0:
            detS = Sbb44*Sbb55 - Sbb45*Sbb45
            self.Abarbar44 = h*h*Sbb55/detS
            self.Abarbar45 = -h*h*Sbb45/detS
            self.Abarbar55 = h*h*Sbb44/detS
        else:
            self.Abarbar44 = np.nan
            self.Abarbar45 = np.nan
            self.Abarbar55 = np.nan

        if mode is None:
            self.A44 = self.Abar44
            self.A45 = self.Abar45
            self.A55 = self.Abar55

        elif mode == 'constant':
            self.A44 = 5/6.*self.Abar44
            self.A45 = 5/6.*self.Abar45
            self.A55 = 5/6.*self.Abar55

        elif mode == 'rohwer':
            if singular_ply >= 0:
                raise _singular_Cs_error(singular_ply, self.plies[singular_ply])
            z = np.zeros(N + 1, dtype=DOUBLE)
            fcoef = np.zeros((N, 3, 2, 2), dtype=DOUBLE)
            status = _rohwer(N, &self._ts_plydata[0, 0], self._ts_offset,
                             1., 0., 0., 1., &z[0], &fcoef[0, 0, 0, 0], S)
            if status == 2:
                raise ValueError('The ABD matrix of the laminate is singular '
                                 'or ill-conditioned, the transverse shear '
                                 'stiffness cannot be computed')
            detS = S[0]*S[2] - S[1]*S[1]
            if status != 0 or not detS > 0:
                raise ValueError('Singular transverse shear compliance, the '
                                 'transverse shear stiffness cannot be '
                                 'computed')
            self.A44 = S[2]/detS
            self.A45 = -S[1]/detS
            self.A55 = S[0]/detS
            self._ts_z = z
            self._ts_fcoef = fcoef
            self._ts_ready = True
            self._ts_element_frame = True

        elif mode == 'vlachoutsis':
            k13 = 0; k23 = 0
            for alpha in range(2):
                # alpha = 0: direction 1 (x), uses C11 and G13 = q55L
                # alpha = 1: direction 2 (y), uses C22 and G23 = q44L
                num = 0; den = 0; d = 0
                for k in range(N):
                    ply = self.plies[k]
                    Gk = ply.q55L if alpha == 0 else ply.q44L
                    if not Gk > 0:
                        raise _singular_Cs_error(k, ply)
                    Dk = ply.q11L if alpha == 0 else ply.q22L
                    za = z[k]
                    zb = z[k+1]
                    num += Dk*(zb*zb - za*za)/2.
                    den += Dk*(zb - za)
                    d += Gk*ply.h
                if not den > 0:
                    raise ValueError('Vlachoutsis shear correction factors '
                                     'require positive in-plane stiffnesses')
                zn = num/den
                R = 0; I = 0; gza = 0
                for k in range(N):
                    ply = self.plies[k]
                    Gk = ply.q55L if alpha == 0 else ply.q44L
                    Dk = ply.q11L if alpha == 0 else ply.q22L
                    za = z[k]
                    zb = z[k+1]
                    R += Dk*((zb - zn)**3 - (za - zn)**3)/3.
                    zm = (za + zb)/2.
                    dz = (zb - za)/2.
                    for ig in range(3):
                        zg = zm + dz*gp[ig]
                        gz = gza - Dk*(0.5*(zg*zg - za*za) - zn*(zg - za))
                        I += gw[ig]*dz*gz*gz/Gk
                    # g at the top interface of this ply
                    gza = gza - Dk*(0.5*(zb*zb - za*za) - zn*(zb - za))
                kappa = R*R/(d*I)
                if alpha == 0:
                    k13 = kappa
                else:
                    k23 = kappa
            self.A44 = k23*self.Abar44
            self.A45 = (k13 + k23)/2.*self.Abar45
            self.A55 = k13*self.Abar55

        self.scf_k13 = self.A55/self.Abar55 if self.Abar55 != 0 else np.nan
        self.scf_k23 = self.A44/self.Abar44 if self.Abar44 != 0 else np.nan


    cpdef tuple calc_transverse_shear_stress(ShellProp self, double z,
            double Qy, double Qx):
        r"""Transverse shear stresses at a given height

        Evaluates the equilibrium distribution of Rohwer (1988), in the
        material coordinate system:

        .. math::

            \begin{Bmatrix} \tau_{yz} \\ \tau_{xz} \end{Bmatrix} =
            f^{(k)}(z) \begin{Bmatrix} Q_y \\ Q_x \end{Bmatrix}

        where `f^{(k)}(z)` is quadratic within each ply, vanishes at the bottom
        and top faces and is continuous at the ply interfaces. This is the
        consistent way to recover `\tau_{xz}` and `\tau_{yz}`, e.g. for
        failure criteria, since `C_s \gamma` is constant within each ply and
        non-zero at the free surfaces. For a homogeneous plate it gives the
        parabola `\tau_{xz} = 3 Q_x/(2h) (1 - 4 \bar{z}^2/h^2)`, with `\bar{z}`
        measured from the mid-surface.

        The distribution only depends on the in-plane stiffnesses of the plies
        and is the same for every ``shear_correction``. It is computed by
        :meth:`.calc_transverse_shear_stiffness` when ``shear_correction`` is
        ``'rohwer'``, or on the first call otherwise, and it is reset by
        :meth:`.calc_constitutive_matrix`, which must be called again if the
        plies are modified.

        Parameters
        ----------
        z : float
            Height measured from the reference surface, within `[-h/2 +
            offset, +h/2 + offset]`. At a ply interface both plies give the
            same result.
        Qy, Qx : float
            Transverse shear forces per unit length, `Q_y` and `Q_x`, e.g.
            ``{Qy, Qx} = Ats @ {gamma_yz, gamma_xz}``.

        Returns
        -------
        tau_yz, tau_xz : tuple of float
            Transverse shear stresses.

        Raises
        ------
        ValueError
            If ``z`` is outside the laminate, if the laminate has no plies, or
            if the ABD matrix of the laminate is singular.

        """
        cdef int k, lo, hi, mid, N, status
        cdef double tol
        cdef double [::1] zi
        cdef double [:, :, :, ::1] fc

        if not self._ts_ready:
            N = <int>len(self.plies)
            if N == 0:
                raise ValueError('ShellProp with 0 plies!')
            self._store_ply_data()
            zi = np.zeros(N + 1, dtype=DOUBLE)
            fc = np.zeros((N, 3, 2, 2), dtype=DOUBLE)
            status = _rohwer(N, &self._ts_plydata[0, 0], self._ts_offset,
                             1., 0., 0., 1., &zi[0], &fc[0, 0, 0, 0], NULL)
            if status != 0:
                raise ValueError('The ABD matrix of the laminate is singular '
                                 'or ill-conditioned, the transverse shear '
                                 'distribution cannot be computed')
            self._ts_z = zi
            self._ts_fcoef = fc
            self._ts_ready = True
        zi = self._ts_z
        fc = self._ts_fcoef
        N = <int>zi.shape[0] - 1
        tol = 1e-12*(zi[N] - zi[0])
        if not (zi[0] - tol <= z <= zi[N] + tol):
            raise ValueError('z=%g is outside the laminate, [%g, %g]'
                             % (z, zi[0], zi[N]))
        # last ply k with zi[k] <= z
        lo = 0
        hi = N - 1
        while lo < hi:
            mid = (lo + hi + 1)//2
            if zi[mid] <= z:
                lo = mid
            else:
                hi = mid - 1
        k = lo
        return ((fc[k, 0, 0, 0] + z*fc[k, 1, 0, 0] + z*z*fc[k, 2, 0, 0])*Qy
              + (fc[k, 0, 0, 1] + z*fc[k, 1, 0, 1] + z*z*fc[k, 2, 0, 1])*Qx,
                (fc[k, 0, 1, 0] + z*fc[k, 1, 1, 0] + z*z*fc[k, 2, 1, 0])*Qy
              + (fc[k, 0, 1, 1] + z*fc[k, 1, 1, 1] + z*z*fc[k, 2, 1, 1])*Qx)


    cpdef void calc_equivalent_properties(ShellProp self):
        r"""Calculate the equivalent laminate properties

        The following attributes are updated:

            ``e1``, ``e2``, ``g12``, ```u12``, ``nu21``

        """
        AI = np.linalg.inv(self.get_ABD())
        a11, a12, a22, a33 = AI[0,0], AI[0,1], AI[1,1], AI[2,2]
        self.e1 = 1./(self.h*a11)
        self.e2 = 1./(self.h*a22)
        self.g12 = 1./(self.h*a33)
        self.nu12 = - a12 / a11
        self.nu21 = - a12 / a22


    cpdef void calc_constitutive_matrix(ShellProp self) except *:
        r"""Calculate the laminate constitutive terms

        This is the commonly called ``ABD`` matrix with ``shape=(6, 6)``. When
        the first-order shear deformation theory is used, the transverse shear
        stiffnesses ``A44``, ``A45``, ``A55`` are also required, which are
        calculated at the end by :meth:`.calc_transverse_shear_stiffness`,
        with the shear correction selected by ``shear_correction`` already
        applied.

        """
        cdef double h0, hk_1, hk
        self.h = 0.
        self.intrho = 0.
        self.intrhoz = 0.
        self.intrhoz2 = 0.
        for ply in self.plies:
            self.h += ply.h
        h0 = -self.h/2. + self.offset
        self.A11 = 0; self.A12 = 0; self.A16 = 0; self.A22 = 0; self.A26 = 0; self.A66 = 0
        self.B11 = 0; self.B12 = 0; self.B16 = 0; self.B22 = 0; self.B26 = 0; self.B66 = 0
        self.D11 = 0; self.D12 = 0; self.D16 = 0; self.D22 = 0; self.D26 = 0; self.D66 = 0
        for ply in self.plies:
            hk_1 = h0
            h0 += ply.h
            hk = h0

            self.intrho += ply.matlamina.rho*(hk - hk_1)
            self.intrhoz += ply.matlamina.rho*(hk*hk/2. - hk_1*hk_1/2.)
            self.intrhoz2 += ply.matlamina.rho*(hk*hk*hk/3. - hk_1*hk_1*hk_1/3.)

            self.A11 += ply.q11L*(hk - hk_1)
            self.A12 += ply.q12L*(hk - hk_1)
            self.A16 += ply.q16L*(hk - hk_1)
            self.A22 += ply.q22L*(hk - hk_1)
            self.A26 += ply.q26L*(hk - hk_1)
            self.A66 += ply.q66L*(hk - hk_1)

            self.B11 += 1/2.*ply.q11L*(hk*hk - hk_1*hk_1)
            self.B12 += 1/2.*ply.q12L*(hk*hk - hk_1*hk_1)
            self.B16 += 1/2.*ply.q16L*(hk*hk - hk_1*hk_1)
            self.B22 += 1/2.*ply.q22L*(hk*hk - hk_1*hk_1)
            self.B26 += 1/2.*ply.q26L*(hk*hk - hk_1*hk_1)
            self.B66 += 1/2.*ply.q66L*(hk*hk - hk_1*hk_1)

            self.D11 += 1/3.*ply.q11L*(hk*hk*hk - hk_1*hk_1*hk_1)
            self.D12 += 1/3.*ply.q12L*(hk*hk*hk - hk_1*hk_1*hk_1)
            self.D16 += 1/3.*ply.q16L*(hk*hk*hk - hk_1*hk_1*hk_1)
            self.D22 += 1/3.*ply.q22L*(hk*hk*hk - hk_1*hk_1*hk_1)
            self.D26 += 1/3.*ply.q26L*(hk*hk*hk - hk_1*hk_1*hk_1)
            self.D66 += 1/3.*ply.q66L*(hk*hk*hk - hk_1*hk_1*hk_1)

        self.calc_transverse_shear_stiffness()

    cpdef void force_balanced(ShellProp self):
        r"""Force a balanced laminate

        The attributes `A_{16}`, `A_{26}`, `B_{16}`, `B_{26}` are set to zero
        to force a balanced laminate.

        """
        if self.offset != 0.:
            raise RuntimeError('Laminates with offset cannot be forced balanced!')
        self.A16 = 0.
        self.A26 = 0.
        self.B16 = 0.
        self.B26 = 0.

    cpdef void force_orthotropic(ShellProp self):
        r"""Force an orthotropic laminate

        The attributes `A_{16}`, `A_{26}`, `B_{16}`, `B_{26}`, `D_{16}`,
        `D_{26}` are set to zero to force an orthotropic laminate.

        """
        if self.offset != 0.:
            raise RuntimeError('Laminates with offset cannot be forced orthotropic!')
        self.A16 = 0.
        self.A26 = 0.
        self.B16 = 0.
        self.B26 = 0.
        self.D16 = 0.
        self.D26 = 0.

    cpdef void force_symmetric(ShellProp self):
        r"""Force a symmetric laminate

        The `B_{ij}` terms of the constitutive matrix are set to zero.

        """
        if self.offset != 0.:
            raise RuntimeError(
                    'Laminates with offset cannot be forced symmetric!')
        self.B11 = 0
        self.B12 = 0
        self.B16 = 0
        self.B22 = 0
        self.B26 = 0
        self.B66 = 0

    cpdef LaminationParameters calc_lamination_parameters(ShellProp self):
        r"""Calculate the lamination parameters.

        The following attributes are calculated:

            ``xiA``, ``xiB``, ``xiD``, ``xiAts``

        """
        cdef double h0, hk, hk_1, h, zbar1, zbar2, Afac, Bfac, Dfac, Atsfac
        cdef LaminationParameters lp = LaminationParameters()

        if len(self.plies) == 0:
            raise ValueError('ShellProp with 0 plies!')

        h = 0.
        for ply in self.plies:
            h += ply.h

        h0 = -h/2. + self.offset
        for ply in self.plies:
            ply.rebuild()
            hk_1 = h0
            h0 += ply.h
            hk = h0
            zbar2 = hk/h
            zbar1 = hk_1/h

            Afac = zbar2 - zbar1
            Bfac = 2*(zbar2*zbar2 - zbar1*zbar1)
            Dfac = 4*(zbar2*zbar2*zbar2 - zbar1*zbar1*zbar1)
            Atsfac = zbar2 - zbar1

            lp.xiA1 += Afac * ply.cos2t
            lp.xiA2 += Afac * ply.sin2t
            lp.xiA3 += Afac * ply.cos4t
            lp.xiA4 += Afac * ply.sin4t

            lp.xiB1 += Bfac * ply.cos2t
            lp.xiB2 += Bfac * ply.sin2t
            lp.xiB3 += Bfac * ply.cos4t
            lp.xiB4 += Bfac * ply.sin4t

            lp.xiD1 += Dfac * ply.cos2t
            lp.xiD2 += Dfac * ply.sin2t
            lp.xiD3 += Dfac * ply.cos4t
            lp.xiD4 += Dfac * ply.sin4t

            lp.xiAts1 += Atsfac * ply.cos2t
            lp.xiAts2 += Atsfac * ply.sin2t

        return lp


cpdef LaminationParameters force_balanced_LP(LaminationParameters lp):
    r"""Force balanced lamination parameters

    The lamination parameters `\xi_{A2}` and `\xi_{A4}` are set to null to
    force a balanced laminate.

    """
    lp.xiA2 = 0
    lp.xiA4 = 0
    return lp


cpdef LaminationParameters force_symmetric_LP(LaminationParameters lp):
    r"""Force symmetric lamination parameters

    The lamination parameters `\xi_{Bi}` are set to null to force a symmetric
    laminate.

    """
    lp.xiB1 = 0
    lp.xiB2 = 0
    lp.xiB3 = 0
    lp.xiB4 = 0
    return lp


cpdef LaminationParameters force_orthotropic_LP(LaminationParameters lp):
    r"""Force orthotropic lamination parameters

    The lamination parameters `\xi_{A2}`, `\xi_{A4}`, `\xi_{B2}`, `\xi_{B4}`,
    `\xi_{D2}` and `\xi_{D4}` are set to null to force an orthotropic laminate.
    The `\xi_{D2}` and `\xi_{D4}` are related to the bend-twist coupling and
    become often very small for balanced laminates with a large amount of
    plies.

    """
    lp.xiA2 = 0
    lp.xiA4 = 0
    lp.xiB2 = 0
    lp.xiB4 = 0
    lp.xiD2 = 0
    lp.xiD4 = 0
    return lp


cpdef ShellProp shellprop_from_LaminationParameters(double thickness, MatLamina
        mat, LaminationParameters lp):
    r"""Return a :class:`.ShellProp` object based in the thickness, material and
    lamination parameters

    Parameters
    ----------
    thickness : float
        The total thickness of the laminate
    mat : :class:`.MatLamina` object
        Material object
    lp : :class:`.LaminationParameters` object
        The container class with all lamination parameters already defined

    Returns
    -------
    lam : :class:`.ShellProp`
        laminate with the ABD and Ats matrices already calculated

    Notes
    -----
    Since the through-thickness distribution of the plies is not known from
    the lamination parameters, no shear correction can be computed. The
    transverse shear stiffnesses ``A44``, ``A45``, ``A55`` are therefore
    equal to the constant-strain ``Abar44``, ``Abar45``, ``Abar55``,
    ``shear_correction`` is ``None``, ``scf_k13 = scf_k23 = 1`` and the
    constant-stress ``Abarbar44``, ``Abarbar45``, ``Abarbar55`` are ``nan``.
    If a shear correction is desired, ``A44``, ``A45``, ``A55`` can be
    modified directly, and the elements will rotate them as a tensor.

    """
    lam = ShellProp()
    lam.h = thickness

    lam.A11 = lam.h*(mat.u1 + mat.u2*lp.xiA1 + 0*lp.xiA2 + mat.u3*lp.xiA3 + 0*lp.xiA4)
    lam.A12 = lam.h*(mat.u4 + 0*lp.xiA1 + 0*lp.xiA2 + (-1)*mat.u3*lp.xiA3 + 0*lp.xiA4)
    lam.A22 = lam.h*(mat.u1 + (-1)*mat.u2*lp.xiA1 + 0*lp.xiA2 + mat.u3*lp.xiA3 + 0*lp.xiA4)
    lam.A16 = lam.h*(0 + 0*lp.xiA1 + mat.u2/2.*lp.xiA2 + 0*lp.xiA3 + mat.u3*lp.xiA4)
    lam.A26 = lam.h*(0 + 0*lp.xiA1 + mat.u2/2.*lp.xiA2 + 0*lp.xiA3 + (-1)*mat.u3*lp.xiA4)
    lam.A66 = lam.h*(mat.u5 + 0*lp.xiA1 + 0*lp.xiA2 + (-1)*mat.u3*lp.xiA3 + 0*lp.xiA4)

    lam.B11 = lam.h*lam.h/4.*(mat.u2*lp.xiB1 + 0*lp.xiB2 + mat.u3*lp.xiB3 + 0*lp.xiB4)
    lam.B12 = lam.h*lam.h/4.*(0*lp.xiB1 + 0*lp.xiB2 + (-1)*mat.u3*lp.xiB3 + 0*lp.xiB4)
    lam.B22 = lam.h*lam.h/4.*((-1)*mat.u2*lp.xiB1 + 0*lp.xiB2 + mat.u3*lp.xiB3 + 0*lp.xiB4)
    lam.B16 = lam.h*lam.h/4.*(0*lp.xiB1 + mat.u2/2.*lp.xiB2 + 0*lp.xiB3 + mat.u3*lp.xiB4)
    lam.B26 = lam.h*lam.h/4.*(0*lp.xiB1 + mat.u2/2.*lp.xiB2 + 0*lp.xiB3 + (-1)*mat.u3*lp.xiB4)
    lam.B66 = lam.h*lam.h/4.*(0*lp.xiB1 + 0*lp.xiB2 + (-1)*mat.u3*lp.xiB3 + 0*lp.xiB4)

    lam.D11 = lam.h*lam.h*lam.h/12.*(mat.u1 + mat.u2*lp.xiD1 + 0*lp.xiD2 + mat.u3*lp.xiD3 + 0*lp.xiD4)
    lam.D12 = lam.h*lam.h*lam.h/12.*(mat.u4 + 0*lp.xiD1 + 0*lp.xiD2 + (-1)*mat.u3*lp.xiD3 + 0*lp.xiD4)
    lam.D22 = lam.h*lam.h*lam.h/12.*(mat.u1 + (-1)*mat.u2*lp.xiD1 + 0*lp.xiD2 + mat.u3*lp.xiD3 + 0*lp.xiD4)
    lam.D16 = lam.h*lam.h*lam.h/12.*(0 + 0*lp.xiD1 + mat.u2/2.*lp.xiD2 + 0*lp.xiD3 + mat.u3*lp.xiD4)
    lam.D26 = lam.h*lam.h*lam.h/12.*(0 + 0*lp.xiD1 + mat.u2/2.*lp.xiD2 + 0*lp.xiD3 + (-1)*mat.u3*lp.xiD4)
    lam.D66 = lam.h*lam.h*lam.h/12.*(mat.u5 + 0*lp.xiD1 + 0*lp.xiD2 + (-1)*mat.u3*lp.xiD3 + 0*lp.xiD4)

    lam.Abar44 = lam.h*(mat.u6 + mat.u7*lp.xiAts1 + 0*lp.xiAts2)
    lam.Abar45 = lam.h*(0 + 0*lp.xiAts1 + (-1)*mat.u7*lp.xiAts2)
    lam.Abar55 = lam.h*(mat.u6 + (-1)*mat.u7*lp.xiAts1 + 0*lp.xiAts2)
    # NOTE the through-thickness ply distribution is not known, such that no
    #      shear correction can be computed
    lam.shear_correction = None
    lam.A44 = lam.Abar44
    lam.A45 = lam.Abar45
    lam.A55 = lam.Abar55
    lam.Abarbar44 = np.nan
    lam.Abarbar45 = np.nan
    lam.Abarbar55 = np.nan
    lam.scf_k13 = 1.
    lam.scf_k23 = 1.

    return lam


cpdef ShellProp shellprop_from_lamination_parameters(double thickness, MatLamina
        matlamina, double xiA1, double xiA2, double xiA3, double xiA4,
        double xiB1, double xiB2, double xiB3, double xiB4,
        double xiD1, double xiD2, double xiD3, double xiD4,
        double xiAts1=0, double xiAts2=0):
    r"""Return a :class:`.ShellProp` object based in the thickness, material and
    lamination parameters

    Note that `\xi_{Ats,1}` and `\xi_{Ats,2}` are optional and usually equal
    to zero, becoming important only when the transverse shear modulus is
    different in the two directions, i.e.  when `G_{13} \ne G{23}`.

    Parameters
    ----------
    thickness : float
        The total thickness of the plate
    matlamina : :class:`.MatLamina` object
        Material object
    xiAj, xiBj, xiDj, xiAtsj : float
        The 14 lamination parameters according to the first-order shear
        deformation theory: `\xi_{A1} \cdots \xi_{A4}`, `\xi_{B1} \cdots
        \xi_{B4}`, `\xi_{D1} \cdots \xi_{D4}`, `\xi_{Ats,1}` and
        `\xi_{Ats,2}`


    Returns
    -------
    lam : :class:`.ShellProp`
        Shell property with the ABD and Ats matrices already calculated. See
        :func:`.shellprop_from_LaminationParameters` for the transverse shear
        stiffnesses.

    """
    lp = LaminationParameters()
    lp.xiA1 = xiA1
    lp.xiA2 = xiA2
    lp.xiA3 = xiA3
    lp.xiA4 = xiA4
    lp.xiB1 = xiB1
    lp.xiB2 = xiB2
    lp.xiB3 = xiB3
    lp.xiB4 = xiB4
    lp.xiD1 = xiD1
    lp.xiD2 = xiD2
    lp.xiD3 = xiD3
    lp.xiD4 = xiD4
    lp.xiAts1 = xiAts1
    lp.xiAts2 = xiAts2
    return shellprop_from_LaminationParameters(thickness, matlamina, lp)


cdef class GradABD:
    r"""Container to store the gradients of the ABD matrix and of the
    constant-strain transverse shear stiffnesses with respect to the
    lamination parameters

    Attributes
    ==========

    gradAij, gradBij, gradDij, gradAtsij : tuple of 2D np.array objects
        The shapes of these gradient matrices are:

            gradAij: (6, 5)
            gradBij: (6, 5)
            gradDij: (6, 5)
            gradAtsij: (3, 3)

        They contain the gradients of each laminate stiffness with respect to
        the thickness and respective lamination parameters. The rows and
        columns correspond to::

            gradAij
            -------

                h xiA1 xiA2 xiA3 xiA4
            A11
            A12
            A16
            A22
            A26
            A66

            gradBij
            -------

                h xiB1 xiB2 xiB3 xiB4
            B11
            B12
            B16
            B22
            B26
            B66

            gradDij
            -------

                h xiD1 xiD2 xiD3 xiD4
            D11
            D12
            D16
            D22
            D26
            D66

            gradAtsij
            ---------

                h xiAts1 xiAts2
            Abar44
            Abar45
            Abar55

    """
    def __init__(GradABD self):
        self.gradAij = np.zeros((6, 5), dtype=DOUBLE)
        self.gradBij = np.zeros((6, 5), dtype=DOUBLE)
        self.gradDij = np.zeros((6, 5), dtype=DOUBLE)
        self.gradAtsij = np.zeros((3, 3), dtype=DOUBLE)

    cpdef void calc_LP_grad(GradABD self, double thickness, MatLamina mat, LaminationParameters lp):
        r"""Gradients of the shell stiffnesses with respect to the thickness and
        lamination parameters

        Parameters
        ----------
        thickness : float
            The total thickness of the laminate
        mat : :class:`.MatLamina` object
            Material object
        lp : :class:`.LaminationParameters` object
            The container class with all lamination parameters already defined

        Returns
        -------
        None
            The attributes of the object are updated.


        """
        cdef int i, j
        cdef double h
        cdef double [:, ::1] gradinv

        gradinv = np.zeros((6, 4), dtype=DOUBLE)
        h = thickness

        gradinv = np.array([[mat.u2, 0, mat.u3, 0],
                            [0, 0, -mat.u3, 0],
                            [0, mat.u2/2., 0, mat.u3],
                            [-mat.u2, 0, mat.u3, 0],
                            [0, mat.u2/2., 0, -mat.u3],
                            [0, 0, -mat.u3, 0]])

        # d(A11 A12 A16 A22 A26 A66) / dh
        self.gradAij[0, 0] = (mat.u1 + mat.u2*lp.xiA1 + 0*lp.xiA2 + mat.u3*lp.xiA3 + 0*lp.xiA4)
        self.gradAij[1, 0] = (mat.u4 + 0*lp.xiA1 + 0*lp.xiA2 + (-1)*mat.u3*lp.xiA3 + 0*lp.xiA4)
        self.gradAij[2, 0] = (0 + 0*lp.xiA1 + mat.u2/2.*lp.xiA2 + 0*lp.xiA3 + mat.u3*lp.xiA4)
        self.gradAij[3, 0] = (mat.u1 + (-1)*mat.u2*lp.xiA1 + 0*lp.xiA2 + mat.u3*lp.xiA3 + 0*lp.xiA4)
        self.gradAij[4, 0] = (0 + 0*lp.xiA1 + mat.u2/2.*lp.xiA2 + 0*lp.xiA3 + (-1)*mat.u3*lp.xiA4)
        self.gradAij[5, 0] = (mat.u5 + 0*lp.xiA1 + 0*lp.xiA2 + (-1)*mat.u3*lp.xiA3 + 0*lp.xiA4)

        # d(A11 A12 A16 A22 A26 A66) / d(xiA1, xiA2, xiA3, xiA4)
        for i in range(5):
            for j in range(4):
                self.gradAij[i, j+1] = h*gradinv[i, j]

        # d(B11 B12 B16 B22 B26 B66) / dh
        self.gradBij[0, 0] = h/2.*(mat.u2*lp.xiB1 + 0*lp.xiB2 + mat.u3*lp.xiB3 + 0*lp.xiB4)
        self.gradBij[1, 0] = h/2.*(0*lp.xiB1 + 0*lp.xiB2 + (-1)*mat.u3*lp.xiB3 + 0*lp.xiB4)
        self.gradBij[2, 0] = h/2.*(0*lp.xiB1 + mat.u2/2.*lp.xiB2 + 0*lp.xiB3 + mat.u3*lp.xiB4)
        self.gradBij[3, 0] = h/2.*((-1)*mat.u2*lp.xiB1 + 0*lp.xiB2 + mat.u3*lp.xiB3 + 0*lp.xiB4)
        self.gradBij[4, 0] = h/2.*(0*lp.xiB1 + mat.u2/2.*lp.xiB2 + 0*lp.xiB3 + (-1)*mat.u3*lp.xiB4)
        self.gradBij[5, 0] = h/2.*(0*lp.xiB1 + 0*lp.xiB2 + (-1)*mat.u3*lp.xiB3 + 0*lp.xiB4)

        # d(B11 B12 B16 B22 B26 B66) / d(xiB1, xiB2, xiB3, xiB4)
        for i in range(5):
            for j in range(4):
                self.gradBij[i, j+1] = h*h/4.*gradinv[i, j]

        # d(D11 D12 D16 D22 D26 D66) / dh
        self.gradDij[0, 0] = h*h/4.*(mat.u1 + mat.u2*lp.xiD1 + 0*lp.xiD2 + mat.u3*lp.xiD3 + 0*lp.xiD4)
        self.gradDij[1, 0] = h*h/4.*(mat.u4 + 0*lp.xiD1 + 0*lp.xiD2 + (-1)*mat.u3*lp.xiD3 + 0*lp.xiD4)
        self.gradDij[2, 0] = h*h/4.*(0 + 0*lp.xiD1 + mat.u2/2.*lp.xiD2 + 0*lp.xiD3 + mat.u3*lp.xiD4)
        self.gradDij[3, 0] = h*h/4.*(mat.u1 + (-1)*mat.u2*lp.xiD1 + 0*lp.xiD2 + mat.u3*lp.xiD3 + 0*lp.xiD4)
        self.gradDij[4, 0] = h*h/4.*(0 + 0*lp.xiD1 + mat.u2/2.*lp.xiD2 + 0*lp.xiD3 + (-1)*mat.u3*lp.xiD4)
        self.gradDij[5, 0] = h*h/4.*(mat.u5 + 0*lp.xiD1 + 0*lp.xiD2 + (-1)*mat.u3*lp.xiD3 + 0*lp.xiD4)

        # d(D11 D12 D16 D22 D26 D66) / d(xiD1, xiD2, xiD3, xiD4)
        for i in range(5):
            for j in range(4):
                self.gradDij[i, j+1] = h*h*h/12.*gradinv[i, j]

        # d(Abar44 Abar45 Abar55) / dh
        self.gradAtsij[0, 0] = (mat.u6 + mat.u7*lp.xiAts1 + 0*lp.xiAts2)
        self.gradAtsij[1, 0] = (0 + 0*lp.xiAts1 + (-1)*mat.u7*lp.xiAts2)
        self.gradAtsij[2, 0] = (mat.u6 + (-1)*mat.u7*lp.xiAts1 + 0*lp.xiAts2)

        # d(Abar44 Abar45 Abar55) / d(xiAts1, xiAts2)
        self.gradAtsij[0, 1] = h*mat.u7
        self.gradAtsij[1, 2] = h*(-mat.u7)
        self.gradAtsij[2, 1] = h*(-mat.u7)
