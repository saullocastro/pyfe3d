#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: nonecheck=False
#cython: overflowcheck=False
#cython: initializedcheck=False
#cython: embedsignature=True
#cython: infer_types=False
r"""
===============================================
Shell property module (:mod:`pyfe3d.shellprop`)
===============================================

Highly based on the `composites <https://saullocastro.github.io/composites/>`_. module.

.. currentmodule:: pyfe3d.shellprop

Classes of the material (:class:`.MatLamina`), of the ply (:class:`.Lamina`)
and of the shell property (:class:`.ShellProp`), of the lamination parameters
(:class:`.LaminationParameters`) and of their gradients (:class:`.GradABD`),
and functions that create a :class:`.ShellProp` from lamination parameters.
The shell elements of pyfe3d take their constitutive matrices from a
:class:`.ShellProp`. The functions of :mod:`pyfe3d.shellprop_utils` create
these objects from the ply data.

"""
from libc.math cimport fabs, sqrt

import numpy as np

DOUBLE = np.float64

cdef class LaminationParameters:
    r"""Lamination parameters

    Container of the 14 lamination parameters, all dimensionless and
    initialized to zero. It is returned by
    :meth:`.ShellProp.calc_lamination_parameters` and used by
    :func:`.shellprop_from_LaminationParameters` and
    :meth:`.GradABD.calc_LP_grad`.

    Attributes
    ----------
    xiA1, xiA2, xiA3, xiA4 : float
        Lamination parameters `\xi_{Ai}` (in-plane), dimensionless
    xiB1, xiB2, xiB3, xiB4 : float
        Lamination parameters `\xi_{Bi}` (in-plane coupling with bending),
        dimensionless
    xiD1, xiD2, xiD3, xiD4 : float
        Lamination parameters `\xi_{Di}` (bending), dimensionless
    xiAts1, xiAts2 : float
        Lamination parameters `\xi_{Ats,i}` (transverse shear),
        dimensionless

    Notes
    -----
    The lamination parameters are defined in
    :meth:`.ShellProp.calc_lamination_parameters`, and the stiffnesses that
    they give in :func:`.shellprop_from_LaminationParameters`.

    """
    def __init__(LaminationParameters self):
        self.xiA1=0; self.xiA2=0; self.xiA3=0; self.xiA4=0
        self.xiB1=0; self.xiB2=0; self.xiB3=0; self.xiB4=0
        self.xiD1=0; self.xiD2=0; self.xiD3=0; self.xiD4=0
        self.xiAts1=0; self.xiAts2=0


cdef class MatLamina:
    r"""
    Orthotropic material lamina

    Container of the elastic, thermal, density and strength properties of a
    material, created with all attributes equal to zero. After the elastic
    properties are set, :meth:`rebuild` computes the constitutive constants
    ``q11`` to ``q66``, the stiffness constants ``c11`` to ``c66`` and the
    invariants ``u1`` to ``u7``, all with the dimension of force per unit
    area; :meth:`trace_normalize_plane_stress` divides some of them by the
    trace, such that they become dimensionless. Usually created by
    :func:`.read_laminaprop`.

    Attributes
    ----------

    e1 : float
        Young Modulus in direction 1, force per unit area
    e2 : float
        Young Modulus in direction 2, force per unit area
    e3 : float
        Young Modulus in direction 3, force per unit area
    g12 : float
        in-plane shear modulus, force per unit area
    g13 : float
        transverse shear modulus for plane 1-Z, force per unit area
    g23 : float
        transverse shear modulus for plane 2-Z, force per unit area
    nu12 : float
        Poisson's ratio 12, dimensionless
    nu13 : float
        Poisson's ratio 13, dimensionless
    nu23 : float
        Poisson's ratio 23, dimensionless
    nu21 : float
        Poisson's ratio 21: use formula nu12/e1 = nu21/e2, dimensionless
    nu31 : float
        Poisson's ratio 31: use formula nu31/e3 = nu13/e1, dimensionless
    nu32 : float
        Poisson's ratio 32: use formula nu23/e2 = nu32/e3, dimensionless
    rho : float
        especific mass (mass / volume)
    a1 : float
        thermal expansion coeffiecient in direction 1, per unit temperature
    a2 : float
        thermal expansion coeffiecient in direction 2, per unit temperature
    a3 : float
        thermal expansion coeffiecient in direction 3, per unit temperature
    tref : float
        reference temperature
    st1,st2 : float
        allowable tensile stresses for directions 1 and 2, force per unit
        area
    sc1,sc2 : float
        allowable compressive stresses for directions 1 and 2, force per unit
        area
    ss12 : float
        allowable in-plane stress for shear, force per unit area
    q11 : float
        lamina constitutive constant 11
    q12 : float
        lamina constitutive constant 12
    q13 : float
        lamina constitutive constant 13
    q21 : float
        lamina constitutive constant 21
    q22 : float
        lamina constitutive constant 22
    q23 : float
        lamina constitutive constant 23
    q31 : float
        lamina constitutive constant 31
    q32 : float
        lamina constitutive constant 32
    q33 : float
        lamina constitutive constant 33
    q44 : float
        lamina constitutive constant 44
    q55 : float
        lamina constitutive constant 55
    q66 : float
        lamina constitutive constant 66
    c11, c12, c13, c22, c23, c33, c44, c55, c66 : float
        lamina stiffness constants
    u1, u2, u3, u4, u5, u6, u7 : float
        lamina material invariants

    Notes
    -----
    For isotropic materials when the user defines `\nu` and `E`, `G` will be
    recaculated based on equation: `G = E/(2 \times (1+\nu))`; in a lower
    priority if the user defines `\nu` and `G`, `E` will be recaculated based
    on equation: `E = 2 \times (1+\nu) \times G`.

    TODO(author): :class:`.MatLamina` recomputes neither `G` nor `E`; only
    :func:`.read_laminaprop` computes ``g12 = g13 = g23 = e/(2*(1+nu))``, for
    ``laminaprop = (e, nu)``.

    """
    def __init__(MatLamina self):
        pass

    cpdef void rebuild(MatLamina self):
        r"""Update constitutive and invariant terms

        Computes, from the moduli ``e1``, ``e2``, ``e3``, ``g12``, ``g13``,
        ``g23`` and the Poisson's ratios ``nu12``, ``nu21``, ``nu13``,
        ``nu31``, ``nu23``, ``nu32``, the stiffness constants ``c11`` to
        ``c66``, the constitutive constants ``q11`` to ``q66`` and the
        invariants ``u1`` to ``u7``, with the dimension of force per unit
        area. ``e1`` and ``e2`` must be non-zero.

        Notes
        -----
        With `\Delta = 1 - \nu_{12}\nu_{21} - \nu_{13}\nu_{31} -
        \nu_{23}\nu_{32} - \nu_{12}\nu_{23}\nu_{31} -
        \nu_{13}\nu_{21}\nu_{32}`, the constitutive constants are the
        stiffnesses of the orthotropic material in its principal axes [2]_:

        .. math::

            q_{11} = e_1 (1 - \nu_{23}\nu_{32})/\Delta, \quad
            q_{12} = e_1 (\nu_{21} + \nu_{23}\nu_{31})/\Delta, \quad
            q_{13} = e_1 (\nu_{31} + \nu_{21}\nu_{32})/\Delta \\
            q_{21} = e_2 (\nu_{12} + \nu_{13}\nu_{32})/\Delta, \quad
            q_{22} = e_2 (1 - \nu_{13}\nu_{31})/\Delta, \quad
            q_{23} = e_2 (\nu_{32} + \nu_{12}\nu_{31})/\Delta \\
            q_{31} = e_3 (\nu_{13} + \nu_{12}\nu_{23})/\Delta, \quad
            q_{32} = e_3 (\nu_{23} + \nu_{13}\nu_{21})/\Delta, \quad
            q_{33} = e_3 (1 - \nu_{12}\nu_{21})/\Delta \\
            q_{44} = g_{23}, \quad q_{55} = g_{13}, \quad q_{66} = g_{12}

        which are symmetric, `q_{ij} = q_{ji}`, when the Poisson's ratios
        satisfy the reciprocity `\nu_{ij}/e_i = \nu_{ji}/e_j`, see
        :func:`.read_laminaprop`. With `e_3 = 0`, as given by
        :func:`.read_laminaprop` for 2 or 6 entries, `\nu_{31} = \nu_{32} =
        0` and `q_{11}`, `q_{12}`, `q_{22}` are the plane-stress reduced
        stiffnesses `e_1/(1 - \nu_{12}\nu_{21})`, `e_1 \nu_{21}/(1 -
        \nu_{12}\nu_{21})` and `e_2/(1 - \nu_{12}\nu_{21})` used by
        :meth:`.Lamina.rebuild`.

        The stiffness constants are computed with `\delta = (1 -
        \nu_{12}\nu_{21} - \nu_{23}\nu_{32} - \nu_{31}\nu_{13} -
        2\nu_{21}\nu_{32}\nu_{13})/(e_1 e_2)` [1]_:

        .. math::

            c_{11} = (1 - \nu_{23}\nu_{32})/(\delta e_2), \quad
            c_{12} = (\nu_{21} + \nu_{31}\nu_{23})/(\delta e_2), \quad
            c_{13} = (\nu_{31} + \nu_{21}\nu_{32})/(\delta e_2) \\
            c_{22} = (1 - \nu_{13}\nu_{31})/(\delta e_1), \quad
            c_{23} = (\nu_{32} + \nu_{12}\nu_{31})/(\delta e_1), \quad
            c_{33} = e_3 (1 - \nu_{12}\nu_{21})/(\delta e_1 e_2) \\
            c_{44} = g_{23}, \quad c_{55} = g_{13}, \quad c_{66} = g_{12}

        With the reciprocity, `\delta e_1 e_2 = \Delta` and `c_{ij} =
        q_{ij}`; both are the entries of the inverse of the compliance matrix
        of the orthotropic material, with `S_{11} = 1/e_1`, `S_{12} =
        -\nu_{12}/e_1`, `S_{13} = -\nu_{13}/e_1`, `S_{22} = 1/e_2`, `S_{23} =
        -\nu_{23}/e_2`, `S_{33} = 1/e_3`, `S_{44} = 1/g_{23}`, `S_{55} =
        1/g_{13}` and `S_{66} = 1/g_{12}`.

        The invariants `u_1` to `u_5` are those of Jones [3]_, and `u_6`,
        `u_7` the transverse shear invariants of the ABAQUS manual [4]_:

        .. math::

            u_1 = (3 q_{11} + 3 q_{22} + 2 q_{12} + 4 q_{66})/8, \quad
            u_2 = (q_{11} - q_{22})/2 \\
            u_3 = (q_{11} + q_{22} - 2 q_{12} - 4 q_{66})/8, \quad
            u_4 = (q_{11} + q_{22} + 6 q_{12} - 4 q_{66})/8 \\
            u_5 = (u_1 - u_4)/2, \quad
            u_6 = (q_{44} + q_{55})/2, \quad
            u_7 = (q_{44} - q_{55})/2

        They give the stiffnesses of a ply at any angle, see
        :meth:`get_invariant_matrix`.

        TODO(author): with `e_3 \ne 0` the invariants are computed from the
        three-dimensional `q_{11}`, `q_{12}`, `q_{22}`, not from the
        plane-stress reduced stiffnesses used by :meth:`.Lamina.rebuild`; is
        this intended?

        References
        ----------
        .. [1] Reddy, J. N., Mechanics of Laminated Composite Plates and
           Shells: Theory and Analysis, Second Edition, CRC Press, 2004.
        .. [2] Gürdal, Z., Haftka, R. T. and Hajela, P., Design and
           Optimization of Laminated Composite Materials, Wiley-Interscience,
           1999.
        .. [3] Jones, R. M., Mechanics of Composite Materials, Second
           Edition, Taylor & Francis, Philadelphia, 1999.
        .. [4] ABAQUS Example Problems Manual, Vol. 1, Example 1.2.2,
           Laminated composite shell: buckling of a cylindrical panel with a
           circular hole.

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
        self.c11 = (1    - nu23*nu32)/(delta*e2)
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
        self.q31 = self.e3*(self.nu13 + self.nu12 * self.nu23) / den
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
        trace-normalization: the constitutive constants ``q11``, ``q12``,
        ``q22``, ``q44``, ``q55``, ``q66`` and the invariants ``u1`` to
        ``u7`` are divided by the trace ``tr = q11 + q22 + 2*q66``, such that
        they become dimensionless trace-normalized values. These can be
        accessed using the same attributes. The other attributes are not
        modified.

        Notes
        -----
        The trace is calculated according to Eq. 1 of Melo et al. [1]_, `tr
        = Q_{11} + Q_{22} + 2Q_{66}`.  The universal in-plane stress
        stiffness components `Q_{11},Q_{12},Q_{22},Q_{44},Q_{55},Q_{66}` are
        divided by `tr`, and the invariants `U_1,U_2,U_3,U_4,U_5,U_6,U_7` are
        calculated with the normalized stiffnesses, such they also become
        trace-normalized invariants; since the invariants are linear in the
        stiffnesses, they are divided by `tr` as well. The trace does not
        depend on the ply angle, since `q_{11L} + q_{22L} + 2 q_{66L} = 2(u_1
        + u_5)`, see :meth:`get_invariant_matrix`.

        References
        ----------
        .. [1] Melo, J. D. D., Bi, J., and Tsai, S. W., A Novel
           Invariant-Based Design Approach to Carbon Fiber Reinforced
           Laminates, Composite Structures, 159, 44-52, 2017.

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

        Builds the symmetric matrix of the stiffness constants ``c11`` to
        ``c66`` computed by :meth:`rebuild`, with the rows and columns ordered
        as 11, 22, 33, 23, 13, 12 (``c44 = g23``, ``c55 = g13``, ``c66 =
        g12``).

        Returns
        -------
        C : double[:, ::1]
            Memoryview of shape ``(6, 6)``, force per unit area.

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

        Builds the matrix of the invariants ``u1`` to ``u7`` computed by
        :meth:`rebuild`. The rows correspond to ``q11``, ``q22``, ``q12``,
        ``q66``, ``q16``, ``q26``, ``q44``, ``q45``, ``q55`` and the columns to
        the factors of ``1`` and of the lamination parameters, in the same
        combinations used by :func:`.shellprop_from_LaminationParameters`.

        Returns
        -------
        U : double[:, ::1]
            Memoryview of shape ``(9, 5)``, force per unit area.

        Notes
        -----
        The stiffnesses of a ply at the angle `\theta` in the laminate axes,
        see :meth:`.Lamina.rebuild`, are linear in `\cos 2\theta`, `\sin
        2\theta`, `\cos 4\theta` and `\sin 4\theta`, with the invariants of
        :meth:`rebuild` as coefficients [1]_:

        .. math::

            \{q_{11L}, q_{22L}, q_{12L}, q_{66L}, q_{16L}, q_{26L}, q_{44L},
            q_{45L}, q_{55L}\}^T = U \{1, \cos 2\theta, \sin 2\theta, \cos
            4\theta, \sin 4\theta\}^T

        which holds when ``u1`` to ``u5`` are computed from the plane-stress
        reduced stiffnesses, i.e. with ``e3 = 0``. Integrated through the
        thickness, the trigonometric terms give the lamination parameters,
        see :func:`.shellprop_from_LaminationParameters`.

        References
        ----------
        .. [1] Jones, R. M., Mechanics of Composite Materials, Second
           Edition, Taylor & Francis, Philadelphia, 1999.

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
    r"""Ply of a laminate

    Created with all attributes equal to zero and ``matlamina`` equal to
    ``None``. After ``matlamina``, ``h`` and ``thetadeg`` are set,
    :meth:`rebuild` computes the trigonometric terms of the ply angle and the
    ply stiffnesses ``q11L`` to ``q55L`` in the laminate axes. Usually created
    by :func:`.laminated_plate`.

    Attributes
    ----------

    plyid : int
        Identificaiton of the composite lamina
    matlamina : :class:`.MatLamina` object
        A :class:`.MatLamina` object
    h : float
        Ply thickness, length
    thetadeg : float
        Ply angle in degrees
    cost, sint : float
        Cosine and sine of the ply angle, dimensionless, computed by
        :meth:`rebuild`
    cos2t, sin2t, cos4t, sin4t : float
        Cosine and sine of two and four times the ply angle, dimensionless,
        computed by :meth:`rebuild`
    q11L, q12L, q16L, q22L, q26L, q66L : float
        Plane-stress stiffnesses of the ply in the laminate axes, force per
        unit area, computed by :meth:`rebuild`
    q44L, q45L, q55L : float
        Transverse shear stiffnesses of the ply in the laminate axes, force
        per unit area, computed by :meth:`rebuild`

    """
    def __init__(Lamina self):
        pass

    cpdef void rebuild(Lamina self):
        r"""Update constitutive matrices

        Computes ``cost``, ``sint``, ``cos2t``, ``sin2t``, ``cos4t`` and
        ``sin4t`` from ``thetadeg``, and the ply stiffnesses ``q11L`` to
        ``q55L`` in the laminate axes from the plane-stress stiffnesses of
        ``matlamina`` in the material axes, which are computed from its
        ``e1``, ``e2``, ``nu12``, ``nu21``, ``g12``, ``g13`` and ``g23``.

        Notes
        -----
        The plane-stress reduced stiffnesses in the material axes are

        .. math::

            Q_{11} = \frac{e_1}{1 - \nu_{12}\nu_{21}}, \quad
            Q_{12} = \frac{\nu_{12} e_2}{1 - \nu_{12}\nu_{21}}, \quad
            Q_{22} = \frac{e_2}{1 - \nu_{12}\nu_{21}}, \quad
            Q_{66} = g_{12}, \quad Q_{44} = g_{23}, \quad Q_{55} = g_{13}

        such that ``e3``, ``nu13`` and ``nu23`` are not used, and thermal
        expansion is not considered. With `c = \cos\theta` and `s =
        \sin\theta`, `\theta` being the ply angle, the stiffnesses in the
        laminate axes are [1]_:

        .. math::

            q_{11L} = Q_{11} c^4 + 2 (Q_{12} + 2 Q_{66}) s^2 c^2 + Q_{22} s^4 \\
            q_{12L} = (Q_{11} + Q_{22} - 4 Q_{66}) s^2 c^2 + Q_{12} (s^4 + c^4) \\
            q_{22L} = Q_{11} s^4 + 2 (Q_{12} + 2 Q_{66}) s^2 c^2 + Q_{22} c^4 \\
            q_{16L} = (Q_{11} - Q_{12} - 2 Q_{66}) s c^3 + (Q_{12} - Q_{22} +
            2 Q_{66}) s^3 c \\
            q_{26L} = (Q_{11} - Q_{12} - 2 Q_{66}) s^3 c + (Q_{12} - Q_{22} +
            2 Q_{66}) s c^3 \\
            q_{66L} = (Q_{11} + Q_{22} - 2 Q_{12} - 2 Q_{66}) s^2 c^2 + Q_{66}
            (s^4 + c^4) \\
            q_{44L} = Q_{44} c^2 + Q_{55} s^2, \quad
            q_{45L} = (Q_{55} - Q_{44}) s c, \quad
            q_{55L} = Q_{55} c^2 + Q_{44} s^2

        References
        ----------
        .. [1] Reddy, J. N., Mechanics of Laminated Composite Plates and
           Shells: Theory and Analysis, Second Edition, CRC Press, 2004.

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
        r"""Return displacement transformation matrix from lamina to laminate

        Builds ``[[cost, sint, 0], [-sint, cost, 0], [0, 0, 1]]`` from the
        terms computed by :meth:`rebuild`.

        Returns
        -------
        T : double[:, ::1]
            Memoryview of shape ``(3, 3)``, dimensionless.

        Notes
        -----
        With the ply angle `\theta` measured from the laminate `x` axis to
        the material direction 1, as in the stiffnesses of :meth:`rebuild`,
        the material direction 1 is `\{\cos\theta, \sin\theta, 0\}` in the
        laminate axes, such that `\{u_1, u_2, u_3\}^T = T \{u_x, u_y,
        u_z\}^T`.

        TODO(author): this maps the laminate components to the lamina
        components, while the name and the summary say from lamina to
        laminate.

        """
        return np.array([[ self.cost, self.sint, 0],
                         [-self.sint, self.cost, 0],
                         [   0,     0, 1]], dtype=DOUBLE)

    cpdef double [:, ::1] get_constitutive_matrix(Lamina self):
        r"""Return the constitutive matrix

        Builds the symmetric matrix of the ply stiffnesses in the laminate axes
        computed by :meth:`rebuild`, ``[[q11L, q12L, q16L, 0, 0], [q12L, q22L,
        q26L, 0, 0], [q16L, q26L, q66L, 0, 0], [0, 0, 0, q44L, q45L], [0, 0,
        0, q45L, q55L]]``.

        Returns
        -------
        C : double[:, ::1]
            Memoryview of shape ``(5, 5)``, force per unit area.

        """
        return np.array([[self.q11L, self.q12L, self.q16L,    0,    0],
                         [self.q12L, self.q22L, self.q26L,    0,    0],
                         [self.q16L, self.q26L, self.q66L,    0,    0],
                         [   0,    0,    0, self.q44L, self.q45L],
                         [   0,    0,    0, self.q45L, self.q55L]], dtype=DOUBLE)

    cpdef double [:, ::1] get_transf_matrix_stress_to_lamina(Lamina self):
        r"""Return stress transformation matrix from laminate to lamina

        Builds the matrix from the terms ``cost``, ``sint`` and ``sin2t``
        computed by :meth:`rebuild`. TODO(author): order of the six stress
        components.

        Returns
        -------
        T : double[:, ::1]
            Memoryview of shape ``(6, 6)``, dimensionless.

        See Also
        --------
        get_transf_matrix_stress_to_laminate

        Notes
        -----
        With the stress components ordered as `\{\sigma_{xx}, \sigma_{yy},
        \sigma_{zz}, \tau_{yz}, \tau_{xz}, \tau_{xy}\}`, the order of
        :meth:`.MatLamina.get_constitutive_matrix`, the matrix is the
        transformation of the stresses from the laminate axes to the material
        axes for a rotation by the ply angle `\theta` about `z`, e.g.
        `\sigma_{11} = \cos^2\theta \sigma_{xx} + \sin^2\theta \sigma_{yy} +
        \sin 2\theta \tau_{xy}` [1]_. Its inverse,
        :meth:`get_transf_matrix_stress_to_laminate`, is the same matrix for
        `-\theta`.

        References
        ----------
        .. [1] Reddy, J. N., Mechanics of Laminated Composite Plates and
           Shells: Theory and Analysis, Second Edition, CRC Press, 2004.

        """
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
        r"""Return stress transformation matrix from lamina to laminate

        Builds the matrix from the terms ``cost``, ``sint`` and ``sin2t``
        computed by :meth:`rebuild`, with the opposite sign of all the terms
        in ``sint`` and ``sin2t`` with respect to
        :meth:`get_transf_matrix_stress_to_lamina`. TODO(author): order of the
        six stress components.

        Returns
        -------
        T : double[:, ::1]
            Memoryview of shape ``(6, 6)``, dimensionless.

        See Also
        --------
        get_transf_matrix_stress_to_lamina

        Notes
        -----
        It is the inverse of :meth:`get_transf_matrix_stress_to_lamina`, i.e.
        the transformation of the stresses from the material axes to the
        laminate axes, obtained by replacing the ply angle `\theta` by
        `-\theta`.

        """
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


cdef inline void _congruence3(double *T, double *M, double *out) noexcept nogil:
    r"""Symmetric 3x3 congruence `out = T M T^T`, all stored row by row"""
    cdef int i, j, k
    cdef double TM[9]
    for i in range(3):
        for j in range(3):
            TM[3*i+j] = T[3*i]*M[j] + T[3*i+1]*M[3+j] + T[3*i+2]*M[6+j]
    for i in range(3):
        for j in range(i, 3):
            out[3*i+j] = TM[3*i]*T[3*j] + TM[3*i+1]*T[3*j+1] + TM[3*i+2]*T[3*j+2]
            out[3*j+i] = out[3*i+j]


cdef void _rotate_ply(double *pd, double m11, double m12, double m21,
        double m22, double *q) noexcept nogil:
    r"""Ply stiffnesses rotated from the material to the element frame

    ``pd`` contains ``h, q11L, q12L, q16L, q22L, q26L, q66L, q44L, q45L, q55L``
    in the material frame. The output ``q`` contains ``q11, q12, q16, q22,
    q26, q66, q44, q45, q55`` in the element frame, rotated with `m_{11}`,
    `m_{12}`, `m_{21}`, `m_{22}`, the in-plane rotation from the material to
    the element frame. With `m_{12} = 0` the stiffnesses are copied
    unchanged.

    Notes
    -----
    With `\{v\}_e = [[m_{11}, m_{12}], [m_{21}, m_{22}]] \{v\}_m` for the
    in-plane vectors, the plane-stress stiffness `C = [[q_{11L}, q_{12L},
    q_{16L}], [q_{12L}, q_{22L}, q_{26L}], [q_{16L}, q_{26L}, q_{66L}]]` and
    the transverse shear stiffness `C_s = [[q_{44L}, q_{45L}], [q_{45L},
    q_{55L}]]` are rotated as `C^e = T_\sigma C T_\sigma^T` and `C_s^e = T_s
    C_s T_s^T`, with the transformations of `\{\sigma_{xx}, \sigma_{yy},
    \tau_{xy}\}` and of `\{\tau_{yz}, \tau_{xz}\}`:

    .. math::

        T_\sigma = \begin{bmatrix} m_{11}^2 & m_{12}^2 & 2 m_{11} m_{12} \\
        m_{21}^2 & m_{22}^2 & 2 m_{21} m_{22} \\
        m_{11} m_{21} & m_{12} m_{22} & m_{11} m_{22} + m_{12} m_{21}
        \end{bmatrix}, \qquad
        T_s = \begin{bmatrix} m_{22} & m_{21} \\ m_{12} & m_{11}
        \end{bmatrix}

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
    r"""Equilibrium transverse shear distribution and compliance of ``'rohwer'``

    The plies are rotated from the material to the element frame using
    `m_{11}`, `m_{12}`, `m_{21}`, `m_{22}`, see :func:`_rotate_ply`, and the
    distribution and the compliance are computed in the element frame, with
    `z` measured from the reference surface. The ABD matrix of the rotated
    plies is inverted by Gauss-Jordan elimination with partial pivoting,
    after a symmetric diagonal scaling.

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

    Notes
    -----
    The plies are rotated to the element frame such that the two cylindrical
    bending states are posed along the element axes. The diagonal scaling
    makes the singularity check of the pivots, ``1e-12``, independent of the
    units.

    The equilibrium approach of Rohwer (1988) [1]_ assumes two cylindrical
    bending states with constant membrane forces: one in which the resultants
    depend on `x` only, with `M_{xx,x} = Q_x` and `M_{xy,x} = Q_y`, and one
    in which they depend on `y` only, with `M_{yy,y} = Q_y` and `M_{xy,y} =
    Q_x`. `\tau_{xz}` is obtained from the first state and `\tau_{yz}` from
    the second one, integrating the equilibrium equations `\tau_{xz,z} =
    -\sigma_{xx,x}` and `\tau_{yz,z} = -\sigma_{yy,y}` from the bottom face
    `z_1`, where the tractions vanish. With `H^* = [[A, B], [B, D]]^{-1}`,
    the in-plane strains of a state are `(H^*_{1:3} + z H^*_{4:6}) \{N,
    M\}`, where `H^*_{1:3}` and `H^*_{4:6}` are the first and the last three
    rows of `H^*` and `\{N, M\} = \{N_{xx}, N_{yy}, N_{xy}, M_{xx}, M_{yy},
    M_{xy}\}`. With `c_1^{(k)} = \{q_{11}, q_{12}, q_{16}\}` and `c_2^{(k)} =
    \{q_{12}, q_{22}, q_{26}\}` the rows of ply `k` that give `\sigma_{xx}`
    and `\sigma_{yy}`, and `\Pi(z) = z H^*_{1:3} + z^2/2 H^*_{4:6}`:

    .. math::

        \tau_{xz}(z) = \left( a_1^{(k)} - c_1^{(k)} \Pi(z) \right) L_x
        \begin{Bmatrix} Q_y \\ Q_x \end{Bmatrix}, \qquad
        \tau_{yz}(z) = \left( a_2^{(k)} - c_2^{(k)} \Pi(z) \right) L_y
        \begin{Bmatrix} Q_y \\ Q_x \end{Bmatrix} \\
        a_\alpha^{(k)} = \sum_{i=1}^{k} \left( c_\alpha^{(i)} -
        c_\alpha^{(i-1)} \right) \Pi(z_i), \qquad c_\alpha^{(0)} = 0

    for `z_k \le z \le z_{k+1}`, where `L_x` places `Q_y` and `Q_x` at the
    components `M_{xy}` and `M_{xx}` of `\{N, M\}_{,x}`, and `L_y` places
    them at `M_{yy}` and `M_{xy}` of `\{N, M\}_{,y}`. The sum
    `a_\alpha^{(k)}` makes the stresses continuous at the interfaces. Within
    each ply `f^{(k)}` is quadratic in `z`, and it vanishes at the top face,
    since `\int c_1 (H^*_{1:3} + z H^*_{4:6}) dz` through the thickness is the
    row of the identity of `N_{xx}`, analogously for `c_2` and `N_{yy}`,
    which `L_x` and `L_y` do not select.

    The compliance is `S = \sum_k \int_{z_k}^{z_{k+1}} f^{(k)T}
    (C_s^{(k)})^{-1} f^{(k)} dz`, computed with three Gauss-Legendre points
    per ply, exact for the integrand of degree 4 in `z`, see
    :meth:`.ShellProp.calc_transverse_shear_stiffness`.

    References
    ----------
    .. [1] Rohwer, K., 1988, Improved transverse shear stiffness for layered
       finite elements, DFVLR-FB 88-32.

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
            # NOTE same frame-invariant criterion as the ply-wise check in
            #      calc_transverse_shear_stiffness, which rejects these plies
            if (q[6] <= 0 or q[8] <= 0
                    or det <= 0.25e-12*(q[6] + q[8])*(q[6] + q[8])):
                return 1 # pragma: no cover
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
    r"""Shell property, with its stiffnesses and mass integrals

    Created with ``offset = 0``, ``plies = []``, ``stack = []``,
    ``shear_correction = 'rohwer'``, ``scf_k13 = scf_k23 = 1`` and all other
    attributes equal to zero. After ``plies``, a list of :class:`.Lamina`
    objects from the bottom to the top face, and ``offset`` are set,
    :meth:`calc_constitutive_matrix` computes ``h``, the stiffnesses and the
    mass integrals, and :meth:`calc_equivalent_properties` the equivalent
    moduli. The stiffnesses are thickness integrals of the ply stiffnesses
    ``qijL`` of :class:`.Lamina`, with `z` measured from the reference
    surface. Usually created by :func:`.laminated_plate`,
    :func:`.isotropic_plate` or :func:`.shellprop_from_LaminationParameters`.
    The matrices in an element coordinate system are given by
    :meth:`calc_constitutive_element` and :meth:`calc_Ats_element`.

    The object can be pickled: the public attributes and the coefficients
    used for the transverse shear stiffness in the element coordinate system
    are saved, and the through-thickness transverse shear distribution used
    by :meth:`calc_transverse_shear_stress` is recomputed when needed.

    Attributes
    ----------

    plies : list
        List of plies
    stack : list
        List of angles for each ply
    h : float
        Total thickness of the laminate, length
    offset : float
        Offset at the normal direction, length
    e1, e2 : float
        Equivalent laminate moduli in directions 1 and 2, force per unit area
    g12 : float
        Equivalent laminate shear modulus in the 12 direction, force per unit
        area
    nu12, nu21 : float
        Equivalent laminate Poisson ratios in the 12 and 21 directions,
        dimensionless
    A11, A12, A16, A22, A26, A66 : float
        Membrane stiffnesses `\int q_{ijL} dz`, force per unit length
    B11, B12, B16, B22, B26, B66 : float
        Membrane-bending coupling stiffnesses `\int q_{ijL} z dz`, force
    D11, D12, D16, D22, D26, D66 : float
        Bending stiffnesses `\int q_{ijL} z^2 dz`, force times length
    A44, A45, A55 : float
        Transverse shear stiffnesses of the first-order shear deformation
        theory (FSDT) in the material coordinate system, **with the shear
        correction already applied** according to ``shear_correction``. The
        elements use them directly, without any shear correction factor. See
        :meth:`.calc_transverse_shear_stiffness` and
        :meth:`.calc_Ats_element` for how they are brought to the element
        coordinate system. Force per unit length.
    Abar44, Abar45, Abar55 : float
        Constant-strain (uncorrected) transverse shear stiffnesses
        `\bar{A}_{ts} = \sum_k C_s^{(k)} h_k`. Force per unit length.
    Abarbar44, Abarbar45, Abarbar55 : float
        Constant-stress transverse shear stiffnesses `\bar{\bar{A}}_{ts} = h^2
        [\sum_k (C_s^{(k)})^{-1} h_k]^{-1}`, for comparison only. Equal to
        ``nan`` when a ply has a singular `C_s^{(k)}`. Force per unit length.
    shear_correction : str or None
        Method used to obtain ``A44``, ``A45``, ``A55`` from the ply data, see
        :meth:`.calc_transverse_shear_stiffness`. Default is ``'rohwer'``.
    scf_k13, scf_k23 : float
        Reported shear correction ratios ``A55/Abar55`` and ``A44/Abar44``.
        They are informative only, the correction is already inside ``A44``,
        ``A45``, ``A55``. Dimensionless.
    intrho : float
        Integral `\int_{-h/2+offset}^{+h/2+offset} \rho(z) dz`, used in
        equivalent single layer finite element mass matrices, mass per unit
        area
    intrhoz : float
        Integral `\int_{-h/2+offset}^{+h/2+offset} \rho(z)z dz`, used in
        equivalent single layer finite element mass matrices, mass per unit
        area times length
    intrhoz2 : float
        Integral `\int_{-h/2+offset}^{+h/2+offset} \rho(z)z^2 dz`, used in
        equivalent single layer finite element mass matrices, mass per unit
        area times length squared

    Notes
    -----
    In the notation of Reddy [1]_, the stiffnesses relate the stress
    resultants to the generalized strains of the reference surface,
    `\varepsilon^{(0)} = \{\varepsilon_{xx}^{(0)}, \varepsilon_{yy}^{(0)},
    \gamma_{xy}^{(0)}\}` and `\varepsilon^{(1)} = \{\varepsilon_{xx}^{(1)},
    \varepsilon_{yy}^{(1)}, \gamma_{xy}^{(1)}\}`, the in-plane strains being
    `\varepsilon^{(0)} + z \varepsilon^{(1)}`. In the first-order shear
    deformation theory (FSDT) of the shell elements:

    .. math::

        \begin{Bmatrix} N \\ M \end{Bmatrix} =
        \begin{bmatrix} A & B \\ B & D \end{bmatrix}
        \begin{Bmatrix} \varepsilon^{(0)} \\ \varepsilon^{(1)}
        \end{Bmatrix}, \qquad
        \begin{Bmatrix} Q_y \\ Q_x \end{Bmatrix} = A_{ts}
        \begin{Bmatrix} \gamma_{yz} \\ \gamma_{xz} \end{Bmatrix}

    with `N = \{N_{xx}, N_{yy}, N_{xy}\}` and `M = \{M_{xx}, M_{yy},
    M_{xy}\}`, where the constant transverse shear strains through the
    thickness require the shear correction contained in ``A44``, ``A45``,
    ``A55``. The stiffnesses are computed by :meth:`calc_constitutive_matrix`.
    With ``offset`` different than zero, the reference surface is not the
    mid-surface, and ``B`` is not zero even for an isotropic plate.

    References
    ----------
    .. [1] Reddy, J. N., Mechanics of Laminated Composite Plates and Shells:
       Theory and Analysis, Second Edition, CRC Press, 2004.

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
        state['_ts_fourier'] = [self._ts_fourier[i] for i in range(33)]
        return (ShellProp, (), state)

    def __setstate__(ShellProp self, state):
        for name in _SHELLPROP_STATE:
            setattr(self, name, state[name])
        self._ts_ready = False
        self._ts_element_frame = state['_ts_element_frame']
        self._ts_nplies = state['_ts_nplies']
        self._ts_offset = state['_ts_offset']
        for i, value in enumerate(state['_ts_fourier']):
            self._ts_fourier[i] = value

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
        r"""Membrane stiffness matrix

        Built from the attributes ``A11`` to ``A66``; a new array is returned
        at each access, such that changing it does not change the attributes.

        Returns
        -------
        A : ndarray
            Array of shape ``(3, 3)``, force per unit length.

        """
        return np.asarray(self.get_A())
    @property
    def B(self):
        r"""Membrane-bending coupling stiffness matrix

        Built from the attributes ``B11`` to ``B66``; a new array is returned
        at each access, such that changing it does not change the attributes.

        Returns
        -------
        B : ndarray
            Array of shape ``(3, 3)``, force.

        """
        return np.asarray(self.get_B())
    @property
    def D(self):
        r"""Bending stiffness matrix

        Built from the attributes ``D11`` to ``D66``; a new array is returned
        at each access, such that changing it does not change the attributes.

        Returns
        -------
        D : ndarray
            Array of shape ``(3, 3)``, force times length.

        """
        return np.asarray(self.get_D())
    @property
    def Ats(self):
        r"""Transverse shear stiffness matrix ``[[A44, A45], [A45, A55]]``

        Index 4 corresponds to `yz` and index 5 to `xz`, such that `\{Q_y,
        Q_x\}^T = A_{ts} \{\gamma_{yz}, \gamma_{xz}\}^T`. The shear correction
        is already applied, see :meth:`.calc_transverse_shear_stiffness`.
        A new array is returned at each access, such that changing it does not
        change the attributes.

        Returns
        -------
        Ats : ndarray
            Array of shape ``(2, 2)``, force per unit length.

        """
        return np.asarray(self.get_Ats())
    @property
    def Abar_ts(self):
        r"""Constant-strain ``[[Abar44, Abar45], [Abar45, Abar55]]``

        Uncorrected transverse shear stiffness, built from the attributes
        ``Abar44``, ``Abar45``, ``Abar55``.
        A new array is returned at each access, such that changing it does not
        change the attributes.

        Returns
        -------
        Abar_ts : ndarray
            Array of shape ``(2, 2)``, force per unit length.

        """
        return np.asarray(self.get_Abar_ts())
    @property
    def Abarbar_ts(self):
        r"""Constant-stress ``[[Abarbar44, Abarbar45], [Abarbar45, Abarbar55]]``

        Built from the attributes ``Abarbar44``, ``Abarbar45``, ``Abarbar55``.
        A new array is returned at each access, such that changing it does not
        change the attributes.

        Returns
        -------
        Abarbar_ts : ndarray
            Array of shape ``(2, 2)``, force per unit length.

        """
        return np.asarray(self.get_Abarbar_ts())
    @property
    def ABD(self):
        r"""Stiffness matrix ``[[A, B], [B, D]]``

        Built from the attributes ``A11`` to ``D66``; a new array is returned
        at each access, such that changing it does not change the attributes.

        Returns
        -------
        ABD : ndarray
            Array of shape ``(6, 6)``, with the dimensions of ``A``, ``B`` and
            ``D``.

        """
        return np.asarray(self.get_ABD())


    cdef void get_constitutive_element(ShellProp self, double m11, double m12,
            double m21, double m22, double *A, double *B, double *D,
            double *Ats) noexcept nogil:
        r"""Constitutive matrices in the element coordinate system

        This is the single function used by all shell elements to obtain the
        constitutive matrices, with `m_{11}`, `m_{12}`, `m_{21}`, `m_{22}`
        being the in-plane rotation from the material to the element
        coordinate system. When `m_{12} = 0` the matrices are used as stored.

        The 3x3 matrices ``A``, ``B`` and ``D`` are stored row by row and
        rotated as plane-stress stiffnesses. The 2x2 matrix ``Ats = [[A44,
        A45], [A45, A55]]`` is stored row by row, see
        :meth:`.calc_Ats_element` for its rotation. Any of the pointers can be
        ``NULL``, in which case the corresponding matrix is skipped.

        Notes
        -----
        ``A``, ``B`` and ``D`` are rotated with `[A]_e = [T_\sigma] [A]
        [T_\sigma]^T`, with `T_\sigma` of :func:`_rotate_ply`. Without the
        Fourier coefficients of :meth:`.calc_Ats_element`, ``Ats`` is rotated
        with `T_s A_{ts} T_s^T`, `T_s = [[m_{22}, m_{21}], [m_{12},
        m_{11}]]`. With them, the compliance `S = A_{ts}^{-1}` is evaluated
        from its Fourier series, with `\cos 2\theta = m_{11}^2 - m_{21}^2`,
        `\sin 2\theta = 2 m_{11} m_{21}` and the higher harmonics obtained by
        the recurrences `\cos 2(n+1)\theta = 2 \cos 2\theta \cos 2n\theta -
        \cos 2(n-1)\theta`, `\sin 2(n+1)\theta = 2 \cos 2\theta \sin
        2n\theta - \sin 2(n-1)\theta`, and then inverted.

        """
        cdef int i, n
        cdef double T[9]
        cdef double Amat[9]
        cdef double Bmat[9]
        cdef double Dmat[9]
        cdef double S[3]
        cdef double c2, s2, cn, cn_1, sn, sn_1, tmp, detS, t44, t45, t55

        Amat[0] = self.A11; Amat[1] = self.A12; Amat[2] = self.A16
        Amat[3] = self.A12; Amat[4] = self.A22; Amat[5] = self.A26
        Amat[6] = self.A16; Amat[7] = self.A26; Amat[8] = self.A66
        Bmat[0] = self.B11; Bmat[1] = self.B12; Bmat[2] = self.B16
        Bmat[3] = self.B12; Bmat[4] = self.B22; Bmat[5] = self.B26
        Bmat[6] = self.B16; Bmat[7] = self.B26; Bmat[8] = self.B66
        Dmat[0] = self.D11; Dmat[1] = self.D12; Dmat[2] = self.D16
        Dmat[3] = self.D12; Dmat[4] = self.D22; Dmat[5] = self.D26
        Dmat[6] = self.D16; Dmat[7] = self.D26; Dmat[8] = self.D66

        # NOTE using m12 as a criterion to check if material coordinates were
        #      defined
        if m12 == 0:
            for i in range(9):
                if A != NULL:
                    A[i] = Amat[i]
                if B != NULL:
                    B[i] = Bmat[i]
                if D != NULL:
                    D[i] = Dmat[i]
            if Ats != NULL:
                Ats[0] = self.A44
                Ats[1] = self.A45
                Ats[2] = self.A45
                Ats[3] = self.A55
            return

        T[0] = m11*m11; T[1] = m12*m12; T[2] = 2*m11*m12
        T[3] = m21*m21; T[4] = m22*m22; T[5] = 2*m21*m22
        T[6] = m11*m21; T[7] = m12*m22; T[8] = m11*m22 + m12*m21
        if A != NULL:
            _congruence3(T, Amat, A)
        if B != NULL:
            _congruence3(T, Bmat, B)
        if D != NULL:
            _congruence3(T, Dmat, D)

        if Ats == NULL:
            return

        if self._ts_element_frame:
            # NOTE the compliance S = inv(Ats) in the element frame is a
            #      trigonometric polynomial with harmonics of 2*theta up to the
            #      5th, which is evaluated exactly from its Fourier coefficients,
            #      with cos(2*theta) = m11**2 - m21**2 and
            #      sin(2*theta) = 2*m11*m21
            c2 = m11*m11 - m21*m21
            s2 = 2*m11*m21
            for i in range(3):
                S[i] = self._ts_fourier[11*i]
            cn_1 = 1.
            sn_1 = 0.
            cn = c2
            sn = s2
            for n in range(1, 6):
                for i in range(3):
                    S[i] += self._ts_fourier[11*i + 2*n - 1]*cn + self._ts_fourier[11*i + 2*n]*sn
                tmp = cn
                cn = 2*c2*cn - cn_1
                cn_1 = tmp
                tmp = sn
                sn = 2*c2*sn - sn_1
                sn_1 = tmp
            detS = S[0]*S[2] - S[1]*S[1]
            Ats[0] = S[2]/detS
            Ats[1] = -S[1]/detS
            Ats[2] = Ats[1]
            Ats[3] = S[0]/detS
            return

        # NOTE tensor rotation A_e = T_s A T_s^T, T_s = [[m22, m21], [m12, m11]]
        t44 = self.A44
        t45 = self.A45
        t55 = self.A55
        Ats[0] = m22*m22*t44 + 2*m22*m21*t45 + m21*m21*t55
        Ats[1] = m22*m12*t44 + (m22*m11 + m21*m12)*t45 + m21*m11*t55
        Ats[2] = Ats[1]
        Ats[3] = m12*m12*t44 + 2*m12*m11*t45 + m11*m11*t55


    def calc_constitutive_element(ShellProp self, double thetadeg):
        r"""Constitutive matrices in an element coordinate system

        The element coordinate system is such that the material direction
        makes an angle `\theta` with the element `x` axis, measured towards
        the element `y` axis, as used by the shell elements, with `m_{11} =
        \cos\theta`, `m_{12} = -\sin\theta`, `m_{21} = \sin\theta` and `m_{22}
        = \cos\theta`.

        Parameters
        ----------
        thetadeg : float
            Angle `\theta` in degrees.

        Returns
        -------
        A, B, D, Ats : tuple of np.ndarray
            The 3x3 matrices ``A``, ``B``, ``D`` and the 2x2 matrix ``Ats`` in
            the element coordinate system. See :meth:`.calc_Ats_element`.

        """
        cdef double thetarad, c, s
        cdef double [:, ::1] A, B, D, Ats
        A = np.zeros((3, 3), dtype=DOUBLE)
        B = np.zeros((3, 3), dtype=DOUBLE)
        D = np.zeros((3, 3), dtype=DOUBLE)
        Ats = np.zeros((2, 2), dtype=DOUBLE)
        thetarad = deg2rad(thetadeg)
        c = cos(thetarad)
        s = sin(thetarad)
        self.get_constitutive_element(c, -s, s, c, &A[0, 0], &B[0, 0],
                                      &D[0, 0], &Ats[0, 0])
        return np.asarray(A), np.asarray(B), np.asarray(D), np.asarray(Ats)


    def calc_Ats_element(ShellProp self, double thetadeg):
        r"""Transverse shear stiffness in an element coordinate system

        The element coordinate system is such that the material direction
        makes an angle `\theta` with the element `x` axis, measured towards
        the element `y` axis. This is the same rotation used by the shell
        elements, where `m_{11} = \cos\theta`, `m_{12} = -\sin\theta`, `m_{21}
        = \sin\theta` and `m_{22} = \cos\theta`.

        When ``shear_correction='rohwer'`` and the plies are available, the
        stiffness corresponds to the plies rotated to the element frame, i.e.
        all ply angles shifted by `\theta`, and it is evaluated from Fourier
        coefficients of its inverse calculated once per laminate by
        :meth:`.calc_transverse_shear_stiffness`, such that the cost per
        element does not depend on the number of plies.

        In all other cases, i.e. ``shear_correction`` ``'constant'``,
        ``'vlachoutsis'`` or ``None``, or when no plies exist, e.g. for a
        property created from lamination parameters or with ``A44``, ``A45``,
        ``A55`` given directly, ``A44``, ``A45``, ``A55`` are rotated as the
        components of a second-order tensor.

        .. note:: Changes to ``A44``, ``A45``, ``A55`` made after
                  :meth:`.calc_constitutive_matrix` are only used for
                  elements with `m_{12} = 0` when the plies are available
                  and ``shear_correction='rohwer'``, because the stiffness
                  is otherwise obtained from the Fourier coefficients
                  calculated from the plies.

        Parameters
        ----------
        thetadeg : float
            Angle `\theta` in degrees.

        Returns
        -------
        Ats : np.ndarray
            Matrix ``[[A44, A45], [A45, A55]]`` in the element coordinate
            system.

        Notes
        -----
        The transverse shear stiffness obtained with the equilibrium approach
        of Rohwer (1988) [1]_ is not invariant to a rotation of the reference
        frame, because the two cylindrical bending states are tied to the
        `x` and `y` axes. Therefore, when ``shear_correction='rohwer'`` and
        the plies are available, the stiffness corresponds to the plies rotated
        to the element frame, i.e. all ply angles shifted by `\theta`, such
        that the assumed static state and the element kinematics refer to the
        same pair of directions.

        Re-evaluating the method of Rohwer for each element has a cost
        proportional to the number of plies. Instead, the compliance `S =
        A_{ts}^{-1}` in the element frame is represented exactly by its
        Fourier series:

        .. math::

            S(\theta) = S_0 + \sum_{n=1}^{5} \left( S_{cn} \cos 2n\theta +
            S_{sn} \sin 2n\theta \right)

        because the equilibrium distribution `f^{(k)}(z)` is a polynomial of
        degree 4 in `\cos\theta`, `\sin\theta` and `(C_s^{(k)})^{-1}` of degree
        2, such that `S` is a trigonometric polynomial of degree 10 with a
        period of 180 degrees. The coefficients are calculated once per
        laminate by :meth:`.calc_transverse_shear_stiffness`, from 16
        evaluations in rotated frames, and the cost per element becomes
        independent of the number of plies. With `M = 16` equally spaced
        samples `\theta_j = j \pi/M`, the discrete Fourier coefficients of the
        harmonics `2n\theta`, `n \le 5`, are exact, since the aliased
        harmonics would be `M - n \ge 11`.

        In all other cases the stiffness is rotated as a second-order
        tensor:

        .. math::

            A_{ts}^e = T_s A_{ts} T_s^T \qquad T_s = \begin{bmatrix}
            \cos\theta & \sin\theta \\ -\sin\theta & \cos\theta \end{bmatrix}

        which is exact for the constant-strain stiffness.

        References
        ----------
        .. [1] Rohwer, K., 1988, Improved transverse shear stiffness for
           layered finite elements, DFVLR-FB 88-32.

        """
        cdef double thetarad, c, s
        cdef double Ats[4]
        thetarad = deg2rad(thetadeg)
        c = cos(thetarad)
        s = sin(thetarad)
        self.get_constitutive_element(c, -s, s, c, NULL, NULL, NULL, Ats)
        return np.array([[Ats[0], Ats[1]], [Ats[2], Ats[3]]], dtype=DOUBLE)


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

        - ``'rohwer'`` (default): equilibrium approach, giving the full 2x2
          matrix. The shell elements use it evaluated in the element
          coordinate system, see :meth:`.calc_Ats_element`, whose
          coefficients are also computed here, together with the
          through-thickness distribution used by
          :meth:`.calc_transverse_shear_stress`.

        - ``'vlachoutsis'``: the scalar factors ``k13``, ``k23`` are applied
          to the constant-strain stiffness, ``A55 = k13*Abar55``, ``A44 =
          k23*Abar44``, and ``A45 = (k13 + k23)/2*Abar45``.

        - ``'constant'``: `k = 5/6`, i.e. ``A44 = 5/6*Abar44``, ``A45 =
          5/6*Abar45``, ``A55 = 5/6*Abar55``.

        - ``None``: no correction, ``A44 = Abar44``, ``A45 = Abar45``, ``A55 =
          Abar55``.

        The following attributes are also updated: ``Abar44``, ``Abar45``,
        ``Abar55`` (constant strain), ``Abarbar44``, ``Abarbar45``,
        ``Abarbar55`` (constant stress, ``nan`` if a ply has a singular
        `C_s^{(k)}`), and the ratios ``scf_k13 = A55/Abar55`` and ``scf_k23 =
        A44/Abar44``, which are informative only.

        Without plies, ``A44``, ``A45``, ``A55``, ``Abar44``, ``Abar45``,
        ``Abar55``, ``Abarbar44``, ``Abarbar45`` and ``Abarbar55`` are set to
        zero and nothing else is updated.

        Raises
        ------
        ValueError
            If ``shear_correction`` is not recognized; if a ply has a
            singular transverse shear constitutive matrix `C_s^{(k)}`, e.g.
            ``g13 = 0`` or ``g23 = 0`` (``'rohwer'`` and ``'vlachoutsis'``);
            or if the ABD matrix of the laminate is singular (``'rohwer'``).
            Also if the transverse shear compliance obtained with
            ``'rohwer'`` is singular, in the material or in a rotated
            coordinate system, or if the sum of ``q11L`` (direction 1) or
            ``q22L`` (direction 2) times the ply thicknesses is not positive
            (``'vlachoutsis'``).

        Notes
        -----
        The constant-strain stiffness `\bar{A}_{ts} = \sum_k C_s^{(k)} h_k`
        corresponds to transverse shear strains constant through the
        thickness, and the constant-stress stiffness `\bar{\bar{A}}_{ts} = h^2
        [\sum_k (C_s^{(k)})^{-1} h_k]^{-1}` to the constant stresses
        `\{\tau_{yz}, \tau_{xz}\} = \{Q_y, Q_x\}/h`, with the same
        complementary energy.

        - ``'rohwer'``: equilibrium approach of Rohwer (1988) [1]_. The
          transverse shear stresses are obtained from the equilibrium of two
          cylindrical bending states, with zero tractions at the bottom and
          top faces and continuity at every interface, `\{\tau_{yz},
          \tau_{xz}\}^T = f^{(k)}(z) \{Q_y, Q_x\}^T`, see :func:`._rohwer`.
          The 2x2 stiffness is obtained from the complementary energy:

          .. math::

              A_{ts} = \left[ \sum_k \int_{z_k}^{z_{k+1}} f^{(k)T}
              \left(C_s^{(k)}\right)^{-1} f^{(k)} dz \right]^{-1}

          The integrand is a polynomial of degree 4 in `z` within each ply,
          such that the 3-point Gauss-Legendre rule used per ply is exact.
          The method is valid for arbitrary anisotropic and unsymmetric
          laminates, and the result does not depend on ``offset``. The result
          is not invariant to a rotation of the reference frame, because the
          two cylindrical bending states are tied to the `x` and `y` axes.
          For this reason, the shell elements use it evaluated in the element
          coordinate system, see :meth:`.calc_Ats_element`, from the Fourier
          coefficients of the compliance computed here with 16 rotations of
          the plies.

        - ``'vlachoutsis'``: the scalar factors `k_{13}`, `k_{23}` of
          Vlachoutsis (1992) [2]_ are applied to the constant-strain
          stiffness, ``A55 = k13*Abar55``, ``A44 = k23*Abar44``, and the
          ad-hoc ``A45 = (k13 + k23)/2*Abar45``, which cannot represent the
          coupling of angle-ply laminates with ``Abar45 = 0``. The factors use
          `C_{11}` and `C_{22}` of each ply and the direction-wise neutral
          surfaces, being exact only for specially orthotropic plies. For the
          direction 1, with `C_{11} = q_{11L}` and `C_{55} = q_{55L}` of each
          ply:

          .. math::

              z_n = \frac{\int C_{11} z dz}{\int C_{11} dz}, \quad
              R = \int C_{11} (z - z_n)^2 dz, \quad
              g(z) = -\int_{z_1}^{z} C_{11} (\bar{z} - z_n) d\bar{z}, \quad
              k_{13} = \frac{R^2}{\bar{A}_{55} \int g^2/C_{55} dz}

          i.e. the energy of the transverse shear stress `\tau_{xz} = Q_x
          g(z)/R` of the one-dimensional equilibrium `\tau_{xz,z} =
          -\sigma_{xx,x}` is equated with that of the FSDT, `Q_x^2/(k_{13}
          \bar{A}_{55})`. Analogously for `k_{23}` with `C_{22} = q_{22L}`,
          `C_{44} = q_{44L}` and `\bar{A}_{44}`. The integrals of `g` are
          evaluated with three Gauss-Legendre points per ply, which is exact,
          `g^2` being a polynomial of degree 4 within each ply.

        - ``'constant'``: `k = 5/6`, the factor that ``'rohwer'`` and
          ``'vlachoutsis'`` give for a homogeneous plate.

        References
        ----------
        .. [1] Rohwer, K., 1988, Improved transverse shear stiffness for
           layered finite elements, DFVLR-FB 88-32.
        .. [2] Vlachoutsis, S., 1992, Shear correction factors for plates and
           shells, International Journal for Numerical Methods in
           Engineering, 33(7), pp. 1537-1552.
           https://doi.org/10.1002/nme.1620330712

        """
        cdef int i, k, n, M, ig, N, alpha, singular_ply, status
        cdef double h, det, za, zb, zm, dz, zg, theta, c, s
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
            # NOTE relative to the trace, such that the check does not
            #      depend on the frame
            if (ply.q44L <= 0 or ply.q55L <= 0
                    or det <= 0.25e-12*(ply.q44L + ply.q55L)**2):
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
                raise ValueError('Singular transverse shear compliance, the ' # pragma: no cover
                                 'transverse shear stiffness cannot be '
                                 'computed')
            self.A44 = S[2]/detS
            self.A45 = -S[1]/detS
            self.A55 = S[0]/detS
            self._ts_z = z
            self._ts_fcoef = fcoef
            self._ts_ready = True

            # NOTE Fourier coefficients of the compliance S(theta) = inv(Ats)
            #      in the element frame. Since f^(k) is a polynomial of degree
            #      4 in cos(theta), sin(theta), and inv(Cs) of degree 2, S is a
            #      polynomial of degree 10 with period 180 degrees, i.e. a
            #      trigonometric polynomial with harmonics of 2*theta up to the
            #      5th. With M = 16 samples its Fourier coefficients are exact
            M = 16
            for i in range(33):
                self._ts_fourier[i] = 0.
            for k in range(M):
                theta = k*np.pi/M
                c = cos(theta)
                s = sin(theta)
                status = _rohwer(N, &self._ts_plydata[0, 0], self._ts_offset,
                                 c, -s, s, c, NULL, NULL, S)
                if status != 0:
                    raise ValueError('Singular transverse shear compliance in ' # pragma: no cover
                                     'a rotated frame')
                for i in range(3):
                    self._ts_fourier[11*i] += S[i]/M
                    for n in range(1, 6):
                        self._ts_fourier[11*i + 2*n - 1] += 2./M*S[i]*cos(2*n*theta)
                        self._ts_fourier[11*i + 2*n] += 2./M*S[i]*sin(2*n*theta)
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
                    raise ValueError('Vlachoutsis shear correction factors ' # pragma: no cover
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

        Evaluates, for the transverse shear forces ``Qy`` and ``Qx``, the
        through-thickness distribution of the transverse shear stresses of the
        equilibrium approach used by ``shear_correction='rohwer'``, see
        :meth:`.calc_transverse_shear_stiffness`, in the material coordinate
        system, which is quadratic within each ply, vanishes at the bottom and
        top faces and is continuous at the ply interfaces, e.g. for failure
        criteria.

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

        Notes
        -----
        Evaluates the equilibrium distribution of Rohwer (1988) [1]_, in the
        material coordinate system, see :func:`._rohwer`:

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

        References
        ----------
        .. [1] Rohwer, K., 1988, Improved transverse shear stiffness for
           layered finite elements, DFVLR-FB 88-32.

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
                raise ValueError('The ABD matrix of the laminate is singular ' # pragma: no cover
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

            ``e1``, ``e2``, ``g12``, ``nu12``, ``nu21``

        They are computed from the inverse of the 6x6 ``ABD`` matrix and from
        ``h``.

        Notes
        -----
        With `a = [[A, B], [B, D]]^{-1}`, whose upper-left 3x3 block is the
        membrane compliance of the laminate with zero moments and free
        curvatures:

        .. math::

            e_1 = \frac{1}{h a_{11}}, \quad e_2 = \frac{1}{h a_{22}}, \quad
            g_{12} = \frac{1}{h a_{66}}, \quad
            \nu_{12} = -\frac{a_{12}}{a_{11}}, \quad
            \nu_{21} = -\frac{a_{12}}{a_{22}}

        where the index 6 is the third row and column of `a`.

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

        The attributes ``h``, ``intrho``, ``intrhoz``, ``intrhoz2`` and
        ``A11`` to ``D66`` are computed from ``plies`` and ``offset``, with the
        ply stiffnesses of the last call to :meth:`.Lamina.rebuild` and the
        density ``rho`` of the material of each ply.

        Notes
        -----
        With the plies running from `z_1 = -h/2 + offset` to `z_{N+1} = +h/2
        + offset`, the stiffnesses in the notation of Reddy [1]_ and the mass
        integrals are, integrated exactly ply by ply:

        .. math::

            (A_{ij}, B_{ij}, D_{ij}) = \sum_k \int_{z_k}^{z_{k+1}}
            q_{ijL}^{(k)} (1, z, z^2) dz, \quad ij = 11, 12, 16, 22, 26, 66 \\
            (intrho, intrhoz, intrhoz2) = \sum_k \int_{z_k}^{z_{k+1}}
            \rho^{(k)} (1, z, z^2) dz

        and ``A44``, ``A45``, ``A55`` follow from
        :meth:`.calc_transverse_shear_stiffness`.

        References
        ----------
        .. [1] Reddy, J. N., Mechanics of Laminated Composite Plates and
           Shells: Theory and Analysis, Second Edition, CRC Press, 2004.

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

        Raises
        ------
        RuntimeError
            If ``offset`` is not zero.

        Notes
        -----
        `q_{16L}` and `q_{26L}` are odd functions of the ply angle, see
        :meth:`.Lamina.rebuild`, such that `A_{16} = A_{26} = 0` in a
        balanced laminate, i.e. with a ply at `-\theta` for every ply at
        `+\theta`, of the same material and thickness.

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

        Raises
        ------
        RuntimeError
            If ``offset`` is not zero.

        Notes
        -----
        These terms vanish in a laminate whose plies are all at 0 or 90
        degrees, for which `q_{16L} = q_{26L} = 0`, see
        :meth:`.Lamina.rebuild`.

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

        Raises
        ------
        RuntimeError
            If ``offset`` is not zero.

        Notes
        -----
        In a laminate symmetric about its mid-surface the integrand `q_{ijL}
        z` of `B_{ij}` is an odd function of `z`, see
        :meth:`.calc_constitutive_matrix`, such that `B_{ij} = 0`.

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

        They are the attributes of the returned object, computed from the ply
        angles and from the ply heights measured from the reference surface,
        i.e. including ``offset``, divided by the total thickness. The
        object is not modified, except that :meth:`.Lamina.rebuild` is called
        for each ply.

        Returns
        -------
        lp : :class:`.LaminationParameters`
            New object with the lamination parameters, dimensionless.

        Raises
        ------
        ValueError
            If there are no plies.

        Notes
        -----
        With `\bar{z} = z/h`, `z` measured from the reference surface, and
        `\theta(\bar{z})` the ply angle:

        .. math::

            \{\xi_{A1}, \xi_{A2}, \xi_{A3}, \xi_{A4}\} = \int \{\cos 2\theta,
            \sin 2\theta, \cos 4\theta, \sin 4\theta\} d\bar{z} \\
            \{\xi_{B1}, \xi_{B2}, \xi_{B3}, \xi_{B4}\} = 4 \int \{\cos
            2\theta, \sin 2\theta, \cos 4\theta, \sin 4\theta\} \bar{z}
            d\bar{z} \\
            \{\xi_{D1}, \xi_{D2}, \xi_{D3}, \xi_{D4}\} = 12 \int \{\cos
            2\theta, \sin 2\theta, \cos 4\theta, \sin 4\theta\} \bar{z}^2
            d\bar{z} \\
            \{\xi_{Ats,1}, \xi_{Ats,2}\} = \int \{\cos 2\theta, \sin
            2\theta\} d\bar{z}

        integrated exactly ply by ply, such that ``xiAts1 = xiA1`` and
        ``xiAts2 = xiA2``. With these definitions, the stiffnesses of the
        laminate are linear in the lamination parameters, see
        :func:`.shellprop_from_LaminationParameters`.

        TODO(author): source of these definitions, e.g. Gürdal, Haftka and
        Hajela (1999), cited in the comments of :meth:`.MatLamina.rebuild`?

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

    The object ``lp`` is modified in place and returned.

    Parameters
    ----------
    lp : :class:`.LaminationParameters`
        Lamination parameters, modified in place.

    Returns
    -------
    lp : :class:`.LaminationParameters`
        The same object given as input.

    See Also
    --------
    force_symmetric_LP, force_orthotropic_LP

    Notes
    -----
    In a balanced laminate, with a ply at `-\theta` for every ply at
    `+\theta` of the same thickness, the integrals of `\sin 2\theta` and
    `\sin 4\theta` through the thickness vanish, see
    :meth:`.ShellProp.calc_lamination_parameters`, and so do `A_{16}` and
    `A_{26}`.

    """
    lp.xiA2 = 0
    lp.xiA4 = 0
    return lp


cpdef LaminationParameters force_symmetric_LP(LaminationParameters lp):
    r"""Force symmetric lamination parameters

    The lamination parameters `\xi_{Bi}` are set to null to force a symmetric
    laminate.

    The object ``lp`` is modified in place and returned.

    Parameters
    ----------
    lp : :class:`.LaminationParameters`
        Lamination parameters, modified in place.

    Returns
    -------
    lp : :class:`.LaminationParameters`
        The same object given as input.

    See Also
    --------
    force_balanced_LP, force_orthotropic_LP

    Notes
    -----
    In a laminate symmetric about its mid-surface, the integrands of
    `\xi_{Bi}`, linear in `\bar{z}`, are odd functions of `\bar{z}`, see
    :meth:`.ShellProp.calc_lamination_parameters`, such that `\xi_{Bi} = 0`
    and the coupling stiffnesses `B_{ij}` vanish.

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

    The object ``lp`` is modified in place and returned.

    Parameters
    ----------
    lp : :class:`.LaminationParameters`
        Lamination parameters, modified in place.

    Returns
    -------
    lp : :class:`.LaminationParameters`
        The same object given as input.

    See Also
    --------
    force_balanced_LP, force_symmetric_LP

    Notes
    -----
    The lamination parameters of the sine terms give the stiffnesses
    `A_{16}`, `A_{26}`, `B_{16}`, `B_{26}`, `D_{16}` and `D_{26}`, see
    :func:`.shellprop_from_LaminationParameters`. The `\xi_{D2}` and
    `\xi_{D4}` are related to the bend-twist coupling and become often very
    small for balanced laminates with a large amount of plies.

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

    In a new :class:`.ShellProp` object, the stiffnesses ``A11`` to ``D66`` and
    ``Abar44``, ``Abar45``, ``Abar55`` are computed from ``thickness``, from
    the invariants ``u1`` to ``u7`` of ``mat`` and from the lamination
    parameters of ``lp``, with the combinations of
    :meth:`.MatLamina.get_invariant_matrix`. No plies are created, ``offset``
    is zero, and the mass integrals ``intrho``, ``intrhoz``, ``intrhoz2`` and
    the equivalent properties ``e1``, ``e2``, ``g12``, ``nu12``, ``nu21`` are
    not computed, remaining zero.
    The transverse shear stiffnesses ``A44``, ``A45``, ``A55`` are equal to
    the constant-strain ``Abar44``, ``Abar45``, ``Abar55``,
    ``shear_correction`` is ``None``, ``scf_k13 = scf_k23 = 1`` and the
    constant-stress ``Abarbar44``, ``Abarbar45``, ``Abarbar55`` are ``nan``.
    If a shear correction is desired, ``A44``, ``A45``, ``A55`` can be
    modified directly, and the elements will rotate them as a tensor.

    Parameters
    ----------
    thickness : float
        The total thickness of the laminate, length
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
    The stiffnesses of a ply are linear in the trigonometric functions of
    its angle, with the invariants of ``mat`` as coefficients, see
    :meth:`.MatLamina.get_invariant_matrix`, and the lamination parameters
    are their thickness integrals, see
    :meth:`.ShellProp.calc_lamination_parameters`. For a laminate of a
    single material, with `h` = ``thickness``, `\int dz = h \int d\bar{z}`,
    `\int z dz = h^2/4 \cdot 4\int \bar{z} d\bar{z}` and `\int z^2 dz =
    h^3/12 \cdot 12 \int \bar{z}^2 d\bar{z}` give:

    .. math::

        A_{11} = h (u_1 + u_2 \xi_{A1} + u_3 \xi_{A3}), \quad
        A_{12} = h (u_4 - u_3 \xi_{A3}), \quad
        A_{22} = h (u_1 - u_2 \xi_{A1} + u_3 \xi_{A3}) \\
        A_{16} = h (u_2/2 \, \xi_{A2} + u_3 \xi_{A4}), \quad
        A_{26} = h (u_2/2 \, \xi_{A2} - u_3 \xi_{A4}), \quad
        A_{66} = h (u_5 - u_3 \xi_{A3}) \\
        B_{11} = \frac{h^2}{4} (u_2 \xi_{B1} + u_3 \xi_{B3}), \quad
        B_{12} = -\frac{h^2}{4} u_3 \xi_{B3}, \quad
        B_{22} = \frac{h^2}{4} (-u_2 \xi_{B1} + u_3 \xi_{B3}) \\
        B_{16} = \frac{h^2}{4} (u_2/2 \, \xi_{B2} + u_3 \xi_{B4}), \quad
        B_{26} = \frac{h^2}{4} (u_2/2 \, \xi_{B2} - u_3 \xi_{B4}), \quad
        B_{66} = -\frac{h^2}{4} u_3 \xi_{B3} \\
        \bar{A}_{44} = h (u_6 + u_7 \xi_{Ats,1}), \quad
        \bar{A}_{45} = -h u_7 \xi_{Ats,2}, \quad
        \bar{A}_{55} = h (u_6 - u_7 \xi_{Ats,1})

    and `D_{ij}` as `A_{ij}`, with the factor `h^3/12` instead of `h` and
    `\xi_{Di}` instead of `\xi_{Ai}`. Since the through-thickness
    distribution of the plies is not known from the lamination parameters,
    no shear correction can be computed, hence the uncorrected transverse
    shear stiffnesses. Without plies, the elements rotate them as a tensor,
    see :meth:`.ShellProp.calc_Ats_element`.

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

    The 14 lamination parameters according to the first-order shear
    deformation theory are `\xi_{A1} \cdots \xi_{A4}`, `\xi_{B1} \cdots
    \xi_{B4}`, `\xi_{D1} \cdots \xi_{D4}`, `\xi_{Ats,1}` and
    `\xi_{Ats,2}`. They are stored in a new
    :class:`.LaminationParameters` object, passed to
    :func:`.shellprop_from_LaminationParameters`.

    Parameters
    ----------
    thickness : float
        The total thickness of the plate, length
    matlamina : :class:`.MatLamina` object
        Material object
    xiA1, xiA2, xiA3, xiA4 : float
        Lamination parameters `\xi_{A1} \cdots \xi_{A4}`, dimensionless.
    xiB1, xiB2, xiB3, xiB4 : float
        Lamination parameters `\xi_{B1} \cdots \xi_{B4}`, dimensionless.
    xiD1, xiD2, xiD3, xiD4 : float
        Lamination parameters `\xi_{D1} \cdots \xi_{D4}`, dimensionless.
    xiAts1, xiAts2 : float, optional
        Lamination parameters `\xi_{Ats,1}` and `\xi_{Ats,2}`,
        dimensionless. Default ``0``.


    Returns
    -------
    lam : :class:`.ShellProp`
        Shell property with the ABD and Ats matrices already calculated. See
        :func:`.shellprop_from_LaminationParameters` for the transverse shear
        stiffnesses.

    Notes
    -----
    Note that `\xi_{Ats,1}` and `\xi_{Ats,2}` are optional and usually equal
    to zero, becoming important only when the transverse shear modulus is
    different in the two directions, i.e.  when `G_{13} \ne G{23}`, since
    they multiply the invariant `u_7 = (q_{44} - q_{55})/2 = (g_{23} -
    g_{13})/2`, see :func:`.shellprop_from_LaminationParameters`.

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

    Created with all gradients equal to zero, which are updated by
    :meth:`calc_LP_grad`.

    Attributes
    ==========

    gradAij, gradBij, gradDij, gradAtsij : double[:, ::1]
        The shapes of these gradient matrices are:

            gradAij: (6, 5)
            gradBij: (6, 5)
            gradDij: (6, 5)
            gradAtsij: (3, 3)

        They contain the gradients of each laminate stiffness with respect to
        the thickness and respective lamination parameters. Column 0 has the
        dimension of the respective stiffness divided by a length and the
        other columns the dimension of the stiffness. The rows and
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

        Updates ``gradAij``, ``gradBij``, ``gradDij`` and ``gradAtsij`` with
        the derivatives of the stiffnesses computed by
        :func:`.shellprop_from_LaminationParameters` from ``thickness``,
        from the invariants ``u1`` to ``u7`` of ``mat`` and from the
        lamination parameters of ``lp``; column 0 contains the derivatives
        with respect to the thickness. The entries ``[0, 2]``, ``[1, 1]`` and
        ``[2, 2]`` of ``gradAtsij``, which do not depend on the lamination
        parameters, are not updated. All six rows of ``gradAij``, ``gradBij``
        and ``gradDij`` are updated, including row 5 (``A66``, ``B66``,
        ``D66``), whose derivatives with respect to `\xi_{A3}`, `\xi_{B3}`
        and `\xi_{D3}` are `-h u_3`, `-h^2/4 \, u_3` and `-h^3/12 \, u_3`.

        Parameters
        ----------
        thickness : float
            The total thickness of the laminate, length
        mat : :class:`.MatLamina` object
            Material object
        lp : :class:`.LaminationParameters` object
            The container class with all lamination parameters already defined

        Returns
        -------
        None
            The attributes of the object are updated.

        Notes
        -----
        The stiffnesses of :func:`.shellprop_from_LaminationParameters` are
        `A = h \, U \{1, \xi_A\}`, `B = h^2/4 \, U \{0, \xi_B\}`, `D = h^3/12
        \, U \{1, \xi_D\}` and `\bar{A}_{ts} = h \, U_{ts} \{1,
        \xi_{Ats}\}`, with `U` the rows of the in-plane terms of
        :meth:`.MatLamina.get_invariant_matrix` and `U_{ts}` those of the
        transverse shear terms. At constant lamination parameters,
        `\partial A/\partial h = U \{1, \xi_A\}`, `\partial B/\partial h = h/2
        \, U \{0, \xi_B\}`, `\partial D/\partial h = h^2/4 \, U \{1, \xi_D\}`
        and `\partial \bar{A}_{ts}/\partial h = U_{ts} \{1, \xi_{Ats}\}`, and
        at constant thickness the derivatives with respect to the lamination
        parameters are the columns of `U` multiplied by `h`, `h^2/4` and
        `h^3/12`, respectively, and those of `U_{ts}` by `h`.

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
        for i in range(6):
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
        for i in range(6):
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
        for i in range(6):
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
