#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: nonecheck=False
#cython: infer_types=False
"""
Shell property module (:mod:`pyfe3d.shellprop`)
==============================================================

Highly based on the `composites <https://saullocastro.github.io/composites/>`_. module.

.. currentmodule:: pyfe3d.shellprop

"""
import numpy as np

INT = np.int64
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
    xiE1, xiE2, xiE3, xiE4 : float
        Lamination parameters `\xi_{Ei}` (transverse shear)

    """
    def __init__(LaminationParameters self):
        self.xiA1=0; self.xiA2=0; self.xiA3=0; self.xiA4=0
        self.xiB1=0; self.xiB2=0; self.xiB3=0; self.xiB4=0
        self.xiD1=0; self.xiD2=0; self.xiD3=0; self.xiD4=0
        self.xiE1=0; self.xiE2=0; self.xiE3=0; self.xiE4=0

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
        """Update constitutive and invariant terms

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
        self.q44 = self.g12
        self.q55 = self.g23
        self.q66 = self.g13
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
        self.u1 = (3*self.q11 + 3*self.q22 + 2*self.q12 + 4*self.q44) / 8.
        self.u2 = (self.q11 - self.q22) / 2.
        self.u3 = (self.q11 + self.q22 - 2*self.q12 - 4*self.q44) / 8.
        self.u4 = (self.q11 + self.q22 + 6*self.q12 - 4*self.q44) / 8.
        self.u5 = (self.u1-self.u4) / 2.
        self.u6 = (self.q55 + self.q66) / 2.
        self.u7 = (self.q55 - self.q66) / 2.

    cpdef cDOUBLE[:, :] get_constitutive_matrix(MatLamina self):
        """Return the constitutive matrix
        """
        return np.array(
            [[self.c11, self.c12, self.c13,   0,   0,   0],
             [self.c12, self.c22, self.c23,   0,   0,   0],
             [self.c13, self.c23, self.c33,   0,   0,   0],
             [  0,   0,   0, self.c44,   0,   0],
             [  0,   0,   0,   0, self.c55,   0],
             [  0,   0,   0,   0,   0, self.c66]], dtype=DOUBLE)

    cpdef cDOUBLE[:, :] get_invariant_matrix(MatLamina self):
        """Return the invariant matrix
        """
        return np.array(
            [[self.u1,  self.u2,    0,  self.u3,   0],            # q11
             [self.u1, -self.u2,    0,  self.u3,   0],            # q22
             [self.u4,   0,    0, -self.u3,   0],                 # q12
             [self.u6,  self.u7,    0,   0,   0],                 # q55
             [self.u6, -self.u7,    0,   0,   0],                 # q66
             [ 0,   0, -self.u7,    0,   0],                      # q56
             [self.u5,   0,    0, -self.u3,   0],                 # q44
             [ 0,   0, self.u2/2.,   0,  self.u3],                # q14
             [ 0,   0, self.u2/2.,   0, -self.u3]], dtype=DOUBLE) # q24


cdef class Lamina:
    """
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
        """Update constitutive matrices

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

    cpdef cDOUBLE[:, :] get_transf_matrix_displ_to_laminate(Lamina self):
        """Return displacement transformation matrix from lamina to laminate"""
        return np.array([[ self.cost, self.sint, 0],
                         [-self.sint, self.cost, 0],
                         [   0,     0, 1]], dtype=DOUBLE)

    cpdef cDOUBLE[:, :] get_constitutive_matrix(Lamina self):
        """Return the constitutive matrix"""
        return np.array([[self.q11L, self.q12L, self.q16L,    0,    0],
                         [self.q12L, self.q22L, self.q26L,    0,    0],
                         [self.q16L, self.q26L, self.q66L,    0,    0],
                         [   0,    0,    0, self.q44L, self.q45L],
                         [   0,    0,    0, self.q45L, self.q55L]], dtype=DOUBLE)

    cpdef cDOUBLE[:, :] get_transf_matrix_stress_to_lamina(Lamina self):
        """Return stress transformation matrix from laminate to lamina"""
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

    cpdef cDOUBLE[:, :] get_transf_matrix_stress_to_laminate(Lamina self):
        """Return stress transformation matrix from lamina to laminate"""
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
    nu12, n21 : float
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
        self.offset = 0.
        self.scf_k13 = 5/6.
        self.scf_k23 = 5/6.
        self.plies = []
        self.stack = []

    cdef cDOUBLE[:, :] get_A(ShellProp self):
        return np.array([[self.A11, self.A12, self.A16],
                         [self.A12, self.A22, self.A26],
                         [self.A16, self.A26, self.A66]], dtype=DOUBLE)
    cdef cDOUBLE[:, :] get_B(ShellProp self):
        return np.array([[self.B11, self.B12, self.B16],
                         [self.B12, self.B22, self.B26],
                         [self.B16, self.B26, self.B66]], dtype=DOUBLE)
    cdef cDOUBLE[:, :] get_D(ShellProp self):
        return np.array([[self.D11, self.D12, self.D16],
                         [self.D12, self.D22, self.D26],
                         [self.D16, self.D26, self.D66]], dtype=DOUBLE)
    cdef cDOUBLE[:, :] get_E(ShellProp self):
        return np.array([[self.E44, self.E45],
                         [self.E45, self.E55]], dtype=DOUBLE)
    cdef cDOUBLE[:, :] get_ABD(ShellProp self):
        return np.array([[self.A11, self.A12, self.A16, self.B11, self.B12, self.B16],
                         [self.A12, self.A22, self.A26, self.B12, self.B22, self.B26],
                         [self.A16, self.A26, self.A66, self.B16, self.B26, self.B66],
                         [self.B11, self.B12, self.B16, self.D11, self.D12, self.D16],
                         [self.B12, self.B22, self.B26, self.D12, self.D22, self.D26],
                         [self.B16, self.B26, self.B66, self.D16, self.D26, self.D66]], dtype=DOUBLE)
    cdef cDOUBLE[:, :] get_ABDE(ShellProp self):
        return np.array([[self.A11, self.A12, self.A16, self.B11, self.B12, self.B16, 0, 0],
                         [self.A12, self.A22, self.A26, self.B12, self.B22, self.B26, 0, 0],
                         [self.A16, self.A26, self.A66, self.B16, self.B26, self.B66, 0, 0],
                         [self.B11, self.B12, self.B16, self.D11, self.D12, self.D16, 0, 0],
                         [self.B12, self.B22, self.B26, self.D12, self.D22, self.D26, 0, 0],
                         [self.B16, self.B26, self.B66, self.D16, self.D26, self.D66, 0, 0],
                         [0, 0, 0, 0, 0, 0, self.E44, self.E45],
                         [0, 0, 0, 0, 0, 0, self.E44, self.E55]], dtype=DOUBLE)
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

    cpdef void rebuild(ShellProp self):
        """Update thickness and density"""
        cdef double rhoh
        self.h = 0.
        rhoh = 0.
        for ply in self.plies:
            self.h += ply.h
        for ply in self.plies:
            ply.rebuild()
            self.h += ply.h
            rhoh += ply.matlamina.rho * ply.h
        self.rho = rhoh / self.h

    cpdef void calc_scf(ShellProp self):
        """Update shear correction factors of the :class:`.ShellProp` object

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
            Shear correction factors. Also updates attributes: ``scf_k13``
            and ``scf_k23``.

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
        """Calculate the equivalent laminate properties

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
        """Calculate the laminate constitutive terms

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
            self.E44 += ply.q44L*(hk - hk_1)
            self.E45 += ply.q45L*(hk - hk_1)
            self.E55 += ply.q55L*(hk - hk_1)

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

    cpdef void force_orthotropic(ShellProp self):
        r"""Force an orthotropic laminate

        The attributes

        `A_{16}`, `A_{26}`, `B_{16}`, `B_{26}`, `D_{16}`, `D_{26}`

        are set to zero to force an orthotropic laminate.

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
        """Force a symmetric laminate

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
        """Calculate the lamination parameters.

        The following attributes are calculated:

            ``xiA``, ``xiB``, ``xiD``, ``xiE``

        """
        cdef double h0, hk, hk_1, h, Afac, Bfac, Dfac, Efac
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

            Afac = ply.h / h
            Bfac = (2. / h**2) * (hk**2 - hk_1**2)
            Dfac = (4. / h**3) * (hk**3 - hk_1**3)
            Efac = (1. / h) * (hk - hk_1)

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
            lp.xiE3 += Efac * ply.cos4t
            lp.xiE4 += Efac * ply.sin4t

        return lp


cpdef LaminationParameters force_balanced_LP(LaminationParameters lp):
    r"""Force balanced lamination parameters

    The lamination parameters `\xi_{A2}` and `\xi_{A4}` are set to null
    to force a balanced laminate.

    """
    lp.xiA2 = 0
    lp.xiA4 = 0
    return lp

cpdef LaminationParameters force_symmetric_LP(LaminationParameters lp):
    r"""Force symmetric lamination parameters

    The lamination parameters `\xi_{Bi}` are set to null
    to force a symmetric laminate.

    """
    lp.xiB1 = 0
    lp.xiB2 = 0
    lp.xiB3 = 0
    lp.xiB4 = 0
    return lp


cpdef ShellProp laminate_from_lamination_parameters(double thickness, MatLamina
        matlamina, LaminationParameters lp):
    r"""Return a :class:`.ShellProp` object based in the thickness, material and
    lamination parameters

    Parameters
    ----------
    thickness : float
        The total thickness of the laminate
    matlamina : :class:`.MatLamina` object
        Material object
    lp : :class:`.LaminationParameters` object
        Lamination parameters

    Returns
    -------
    lam : :class:`.ShellProp`
        laminate with the ABD and ABDE matrices already calculated

    """
    lam = ShellProp()
    lam.h = thickness
    u = matlamina.get_invariant_matrix()
    # dummies used to unpack vector results
    du1, du2, du3, du4, du5, du6 = 0, 0, 0, 0, 0, 0
    # A matrix terms
    lam.A11, lam.A22, lam.A12, du1,du2,du3, lam.A66, lam.A16, lam.A26 =\
        (lam.h       ) * np.dot(u, np.array([1, lp.xiA1, lp.xiA2, lp.xiA3, lp.xiA4]))
    # B matrix terms
    lam.B11, lam.B22, lam.B12, du1,du2,du3, lam.B66, lam.B16, lam.B26 =\
        (lam.h**2/4. ) * np.dot(u, np.array([0, lp.xiB1, lp.xiB2, lp.xiB3, lp.xiB4]))
    # D matrix terms
    lam.D11, lam.D22, lam.D12, du1,du2,du3, lam.D66, lam.D16, lam.D26 =\
        (lam.h**3/12.) * np.dot(u, np.array([1, lp.xiD1, lp.xiD2, lp.xiD3, lp.xiD4]))
    # E matrix terms
    du1,du2,du3, lam.E44, lam.E55, lam.E45, du4,du5,du6 =\
        (lam.h       ) * np.dot(u, np.array([1, lp.xiE1, lp.xiE2, lp.xiE3, lp.xiE4]))
    return lam


cpdef ShellProp laminate_from_lamination_parameters2(double thickness, MatLamina
        matlamina, double xiA1, double xiA2, double xiA3, double xiA4,
        double xiB1, double xiB2, double xiB3, double xiB4,
        double xiD1, double xiD2, double xiD3, double xiD4,
        double xiE1, double xiE2, double xiE3, double xiE4):
    r"""Return a :class:`.ShellProp` object based in the thickness, material and
    lamination parameters

    Parameters
    ----------
    thickness : float
        The total thickness of the laminate
    matlamina : :class:`.MatLamina` object
        Material object
    xiA1 to xiD4 : float
        The 16 lamination parameters `\xi_{A1} \cdots \xi_{A4}`,  `\xi_{B1}
        \cdots \xi_{B4}`, `\xi_{C1} \cdots \xi_{C4}`,  `\xi_{D1} \cdots
        \xi_{D4}`, `\xi_{E1} \cdots \xi_{E4}`


    Returns
    -------
    lam : :class:`.ShellProp`
        laminate with the ABD and ABDE matrices already calculated

    """
    lp = LaminationParameters()
    return laminate_from_lamination_parameters(thickness, matlamina, lp)


def read_laminaprop(laminaprop, rho=0):
    """Returns a :class:`.MatLamina` object based on an input ``laminaprop`` tuple

    Parameters
    ----------
    laminaprop : list or tuple
        Tuple containing the folliwing entries:

            (e1, e2, nu12, g12, g13, g23, e3, nu13, nu23)

        for othotropic materials the user can only supply:

            (e1, e2, nu12, g12, g13, g23)

        for isotropic materials the user can only supply:

            (e, nu) # new

            (e1, e2, nu12) # legacy, kept for compatibility with old codes

        ======  ==============================
        symbol  value
        ======  ==============================
        e1      Young Module in direction 1
        e2      Young Module in direction 2
        nu12    12 Poisson's ratio
        g12     12 Shear Modulus
        g13     13 Shear Modulus
        g23     13 Shear Modulus
        e3      Young Module in direction 3
        nu13    13 Poisson's ratio
        nu23    23 Poisson's ratio
        ======  ==============================


    rho : float, optional
        Material density


    Returns
    -------
    matlam : MatLamina
        A :class:`.MatLamina` object.

    """
    matlam = MatLamina()

    #laminaProp = (e1, e2, nu12, g12, g13, g23, e3, nu13, nu23)
    if laminaprop == None:
        raise ValueError('laminaprop must be a tuple')
    if len(laminaprop) == 3: #ISOTROPIC legacy
        e = laminaprop[0]
        nu = laminaprop[2]
        g = e/(2*(1+nu))
        laminaprop = (e, e, nu, g, g, g, e, nu, nu)
    if len(laminaprop) == 2: #ISOTROPIC new
        e = laminaprop[0]
        nu = laminaprop[1]
        g = e/(2*(1+nu))
        laminaprop = (e, e, nu, g, g, g, e, nu, nu)
    nu12 = laminaprop[2]
    if len(laminaprop) < 9:
        e2 = laminaprop[1]
        laminaprop = tuple(list(laminaprop)[:6] + [e2, nu12, nu12])
    matlam.e1 = laminaprop[0]
    matlam.e2 = laminaprop[1]
    matlam.e3 = laminaprop[6]
    matlam.nu12 = laminaprop[2]
    matlam.nu13 = laminaprop[7]
    matlam.nu23 = laminaprop[8]
    matlam.nu21 = matlam.nu12 * matlam.e2 / matlam.e1
    matlam.nu31 = matlam.nu13 * matlam.e3 / matlam.e1
    matlam.nu32 = matlam.nu23 * matlam.e3 / matlam.e2
    matlam.g12 = laminaprop[3]
    matlam.g13 = laminaprop[4]
    matlam.g23 = laminaprop[5]
    matlam.rho = rho
    matlam.rebuild()

    return matlam


def laminated_plate(stack, plyt=None, laminaprop=None, rho=0., plyts=None, laminaprops=None,
        rhos=None, offset=0., calc_scf=True):
    """Read a laminate stacking sequence data.

    :class:`.ShellProp` object is returned based on the inputs given.

    Parameters
    ----------
    stack : list
        Angles of the stacking sequence in degrees.
    plyt : float, optional
        When all plies have the same thickness, ``plyt`` can be supplied.
    laminaprop : tuple, optional
        When all plies have the same material properties, ``laminaprop``
        can be supplied.
    rho : float, optional
        Uniform material density to be used for all plies.
    plyts : list, optional
        A list of floats with the thickness of each ply.
    laminaprops : list, optional
        A list of tuples with a laminaprop for each ply.
    rhos : list, optional
        A list of floats with the material density of each ply.
    offset : float, optional
        Offset along the normal axis about the mid-surface, which influences
        the laminate properties.
    calc_scf : bool, optional
        If True, use :func:`.ShellProp.calc_scf` to compute shear correction
        factors, otherwise the default value of 5/6 is used

    Notes
    -----
    ``plyt`` or ``plyts`` must be supplied
    ``laminaprop`` or ``laminaprops`` must be supplied

    For orthotropic plies, the ``laminaprop`` should be::

        laminaprop = (E11, E22, nu12, G12, G13, G23)

    For isotropic plies, the ``laminaprop`` should be::

        laminaprop = (E, nu)

    """
    lam = ShellProp()
    lam.offset = offset
    lam.stack = list(stack)

    if plyts is None:
        if plyt is None:
            raise ValueError('plyt or plyts must be supplied')
        else:
            plyts = [plyt for i in stack]

    if laminaprops is None:
        if laminaprop is None:
            raise ValueError('laminaprop or laminaprops must be supplied')
        else:
            laminaprops = [laminaprop for i in stack]

    if rhos is None:
        rhos = [rho for i in stack]

    plies = []
    for plyt, laminaprop, thetadeg, rho in zip(plyts, laminaprops, stack, rhos):
        laminaprop = laminaprop
        ply = Lamina()
        ply.thetadeg = float(thetadeg)
        ply.h = plyt
        ply.matlamina = read_laminaprop(laminaprop, rho)
        plies.append(ply)
    lam.plies = plies

    lam.rebuild()
    lam.calc_constitutive_matrix()
    if calc_scf:
        lam.calc_scf()

    return lam


def isotropic_plate(thickness, E, nu, offset=0., calc_scf=True, rho=0.):
    """Read data for an isotropic plate

    :class:`.ShellProp` object is returned based on the inputs given.

    Parameters
    ----------
    thickness : float
        Plate thickness.
    E : float
        Young modulus.
    nu : float, optional
        Poisson's ratio.
    rho : float, optional
        Material density
    offset : float, optional
        Offset along the normal axis about the mid-surface, which influences
        the extension-bending coupling (B matrix).
    calc_scf : bool, optional
        If True, use :func:`.ShellProp.calc_scf` to compute shear correction
        factors, otherwise the default value of 5/6 is used.

    """
    return laminated_plate(plyt=thickness, stack=[0], laminaprop=(E, nu),
            rho=rho, offset=offset, calc_scf=calc_scf)

