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
    xiE1, xiE2 : float
        Lamination parameters `\xi_{Ei}` (transverse shear)

    """
    def __init__(LaminationParameters self):
        self.xiA1=0; self.xiA2=0; self.xiA3=0; self.xiA4=0
        self.xiB1=0; self.xiB2=0; self.xiB3=0; self.xiB4=0
        self.xiD1=0; self.xiD2=0; self.xiD3=0; self.xiD4=0
        self.xiE1=0; self.xiE2=0


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


cdef class ShellProp:
    r"""
    Attributes
    ----------

    plies : list
        List of plies
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
    scf_k13, scf_k23 : float
        Shear correction factor in the 13 and 23 directions
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
        self.scf_k13 = 5/6.
        self.scf_k23 = 5/6.
        self.intrho = 0.
        self.intrhoz = 0.
        self.intrhoz2 = 0.
        self.plies = []
        self.stack = []

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
    cdef double [:, ::1] get_E(ShellProp self):
        return np.array([[self.E44, self.E45],
                         [self.E45, self.E55]], dtype=DOUBLE)
    cdef double [:, ::1] get_ABD(ShellProp self):
        return np.array([[self.A11, self.A12, self.A16, self.B11, self.B12, self.B16],
                         [self.A12, self.A22, self.A26, self.B12, self.B22, self.B26],
                         [self.A16, self.A26, self.A66, self.B16, self.B26, self.B66],
                         [self.B11, self.B12, self.B16, self.D11, self.D12, self.D16],
                         [self.B12, self.B22, self.B26, self.D12, self.D22, self.D26],
                         [self.B16, self.B26, self.B66, self.D16, self.D26, self.D66]], dtype=DOUBLE)
    cdef double [:, ::1] get_ABDE(ShellProp self):
        return np.array([[self.A11, self.A12, self.A16, self.B11, self.B12, self.B16, 0, 0],
                         [self.A12, self.A22, self.A26, self.B12, self.B22, self.B26, 0, 0],
                         [self.A16, self.A26, self.A66, self.B16, self.B26, self.B66, 0, 0],
                         [self.B11, self.B12, self.B16, self.D11, self.D12, self.D16, 0, 0],
                         [self.B12, self.B22, self.B26, self.D12, self.D22, self.D26, 0, 0],
                         [self.B16, self.B26, self.B66, self.D16, self.D26, self.D66, 0, 0],
                         [0, 0, 0, 0, 0, 0, self.E44, self.E45],
                         [0, 0, 0, 0, 0, 0, self.E45, self.E55]], dtype=DOUBLE)
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
    def E(self):
        return np.asarray(self.get_E())
    @property
    def ABD(self):
        return np.asarray(self.get_ABD())
    @property
    def ABDE(self):
        return np.asarray(self.get_ABDE())


    cpdef void calc_scf(ShellProp self):
        r"""Update shear correction factors of the :class:`.ShellProp` object

        Reference:

            Vlachoutsis, S. "Shear correction factors for plates and shells",
            Int. Journal for Numerical Methods in Engineering, Vol. 33,
            1537-1552, 1992.

            http://onlinelibrary.wiley.com/doi/10.1002/nme.1620330712/full


        Using "one shear correction factor" (see reference), assuming:

        - constant G13, G23, E1, E2, nu12, nu21 within each ply
        - g1 calculated using z at the middle of each ply
        - zn1 = :class:`.ShellProp` ``offset`` attribute

        Returns
        -------

        k13, k23 : tuple
            Shear correction factors. Also updates attributes: ``scf_k13`` and
            ``scf_k23``.

        """
        cdef double D1, R1, den1, D2, R2, den2, offset, zbot, z1, z2, thetarad
        cdef double e1, e2, nu12, nu21
        D1 = 0
        R1 = 0
        den1 = 0

        D2 = 0
        R2 = 0
        den2 = 0

        offset = self.offset
        zbot = -self.h/2. + offset
        z1 = zbot

        for ply in self.plies:
            z2 = z1 + ply.h
            thetarad = deg2rad(ply.thetadeg)
            e1 = (ply.matlamina.e1 * np.cos(thetarad) +
                  ply.matlamina.e2 * np.sin(thetarad))
            e2 = (ply.matlamina.e2 * np.cos(thetarad) +
                  ply.matlamina.e1 * np.sin(thetarad))
            nu12 = (ply.matlamina.nu12 * np.cos(thetarad) +
                  ply.matlamina.nu21 * np.sin(thetarad))
            nu21 = (ply.matlamina.nu21 * np.cos(thetarad) +
              ply.matlamina.nu12 * np.sin(thetarad))

            D1 += e1 / (1 - nu12*nu21)
            R1 += D1*((z2 - offset)**3/3. - (z1 - offset)**3/3.)
            den1 += ply.matlamina.g13 * ply.h * (self.h / ply.h) * D1**2*(15*offset*z1**4 + 30*offset*z1**2*zbot*(2*offset - zbot) - 15*offset*z2**4 + 30*offset*z2**2*zbot*(-2*offset + zbot) - 3*z1**5 + 10*z1**3*(-2*offset**2 - 2*offset*zbot + zbot**2) - 15*z1*zbot**2*(4*offset**2 - 4*offset*zbot + zbot**2) + 3*z2**5 + 10*z2**3*(2*offset**2 + 2*offset*zbot - zbot**2) + 15*z2*zbot**2*(4*offset**2 - 4*offset*zbot + zbot**2))/(60*ply.matlamina.g13)

            D2 += e2 / (1 - nu12*nu21)
            R2 += D2*((z2 - self.offset)**3/3. - (z1 - self.offset)**3/3.)
            den2 += ply.matlamina.g23 * ply.h * (self.h / ply.h) * D2**2*(15*offset*z1**4 + 30*offset*z1**2*zbot*(2*offset - zbot) - 15*offset*z2**4 + 30*offset*z2**2*zbot*(-2*offset + zbot) - 3*z1**5 + 10*z1**3*(-2*offset**2 - 2*offset*zbot + zbot**2) - 15*z1*zbot**2*(4*offset**2 - 4*offset*zbot + zbot**2) + 3*z2**5 + 10*z2**3*(2*offset**2 + 2*offset*zbot - zbot**2) + 15*z2*zbot**2*(4*offset**2 - 4*offset*zbot + zbot**2))/(60*ply.matlamina.g23)

            z1 = z2

        self.scf_k13 = R1**2 / den1
        self.scf_k23 = R2**2 / den2


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


    cpdef void calc_constitutive_matrix(ShellProp self):
        r"""Calculate the laminate constitutive terms

        This is the commonly called ``ABD`` matrix with ``shape=(6, 6)`` when
        the classical laminated plate theory is used, or the ``ABDE`` matrix
        when the first-order shear deformation theory is used, containing the
        transverse shear terms.

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
        self.E44 = 0; self.E45 = 0; self.E55 = 0
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

            self.E44 += ply.q44L*(hk - hk_1)
            self.E45 += ply.q45L*(hk - hk_1)
            self.E55 += ply.q55L*(hk - hk_1)

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

            ``xiA``, ``xiB``, ``xiD``, ``xiE``

        """
        cdef double h0, hk, hk_1, h, zbar1, zbar2, Afac, Bfac, Dfac, Efac
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
            Efac = zbar2 - zbar1

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

            lp.xiE1 += Efac * ply.cos2t
            lp.xiE2 += Efac * ply.sin2t

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
        laminate with the ABD and ABDE matrices already calculated

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

    lam.E44 = lam.h*(mat.u6 + mat.u7*lp.xiE1 + 0*lp.xiE2)
    lam.E45 = lam.h*(0 + 0*lp.xiE1 + (-1)*mat.u7*lp.xiE2)
    lam.E55 = lam.h*(mat.u6 + (-1)*mat.u7*lp.xiE1 + 0*lp.xiE2)

    return lam


cpdef ShellProp shellprop_from_lamination_parameters(double thickness, MatLamina
        matlamina, double xiA1, double xiA2, double xiA3, double xiA4,
        double xiB1, double xiB2, double xiB3, double xiB4,
        double xiD1, double xiD2, double xiD3, double xiD4,
        double xiE1=0, double xiE2=0):
    r"""Return a :class:`.ShellProp` object based in the thickness, material and
    lamination parameters

    Note that `\xi_{E1}` and `\xi_{E2}` are optional and usually equal to zero,
    becoming important only when the transverse shear modulus is different in
    the two directions, i.e.  when `G_{13} \ne G{23}`.

    Parameters
    ----------
    thickness : float
        The total thickness of the plate
    matlamina : :class:`.MatLamina` object
        Material object
    xiAj, xiBj, xiDj, xiEj : float
        The 14 lamination parameters according to the first-order shear
        deformation theory: `\xi_{A1} \cdots \xi_{A4}`, `\xi_{B1} \cdots
        \xi_{B4}`, `\xi_{D1} \cdots \xi_{D4}`, `\xi_{E1}` and `\xi_{E2}`


    Returns
    -------
    lam : :class:`.ShellProp`
        Shell property with the ABD and ABDE matrices already calculated

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
    lp.xiE1 = xiE1
    lp.xiE2 = xiE2
    return shellprop_from_LaminationParameters(thickness, matlamina, lp)


cdef class GradABDE:
    r"""Container to store the gradients of the ABDE matrices with respect to
    the lamination parameters

    Attributes
    ==========

    gradAij, gradBij, gradDij, gradEij : tuple of 2D np.array objects
        The shapes of these gradient matrices are:

            gradAij: (6, 5)
            gradBij: (6, 5)
            gradDij: (6, 5)
            gradEij: (3, 3)

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

            gradEij
            -------

                h xiE1 xiE2
            E44
            E45
            E55

    """
    def __init__(GradABDE self):
        self.gradAij = np.zeros((6, 5), dtype=DOUBLE)
        self.gradBij = np.zeros((6, 5), dtype=DOUBLE)
        self.gradDij = np.zeros((6, 5), dtype=DOUBLE)
        self.gradEij = np.zeros((3, 3), dtype=DOUBLE)

    cpdef void calc_LP_grad(GradABDE self, double thickness, MatLamina mat, LaminationParameters lp):
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

        # d(E11 E12 E16 E22 E26 E66) / dh
        self.gradEij[0, 0] = (mat.u6 + mat.u7*lp.xiE1 + 0*lp.xiE2)
        self.gradEij[1, 0] = (0 + 0*lp.xiE1 + (-1)*mat.u7*lp.xiE2)
        self.gradEij[2, 0] = (mat.u6 + (-1)*mat.u7*lp.xiE1 + 0*lp.xiE2)

        # d(E11 E12 E16 E22 E26 E66) / d(xiE1, xiE2)
        self.gradEij[0, 1] = h*mat.u7
        self.gradEij[1, 2] = h*(-mat.u7)
        self.gradEij[2, 1] = h*(-mat.u7)
