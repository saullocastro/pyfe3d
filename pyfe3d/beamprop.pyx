#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: nonecheck=False
#cython: overflowcheck=False
#cython: initializedcheck=False
#cython: embedsignature=True
#cython: infer_types=False
"""
=============================================
Beam property module (:mod:`pyfe3d.beamprop`)
=============================================

.. currentmodule:: pyfe3d.beamprop

Defines :class:`.BeamProp`, the cross-section and material properties read
by the elements :class:`.BeamC`, :class:`.BeamLR` and :class:`.Truss`.

"""
import numpy as np

cdef class BeamProp:
    r"""
    General class to represent beam properties.

    No shear correction factors or geometric correction factors are applied
    internally during the calculation of the structural matrices; any
    correction must be applied by the user to the attributes.

    The constructor takes no argument and sets all attributes to ``0``. The
    attributes are read, each time it is called, by the ``update_*`` methods
    of :class:`.BeamC`, :class:`.BeamLR` and :class:`.Truss` that receive
    the property.

    The integrals over the cross section are taken with the coordinates `y`
    and `z` of the element coordinate system, measured from the beam axis
    that joins the two nodes of the element.

    ``G12``, ``G13`` and ``GJ`` follow ``G`` until they are set: reading
    ``G12`` or ``G13`` returns ``G``, and reading ``GJ`` returns ``G*J``,
    with the values of ``G`` and ``J`` at the time of reading. Once set, the
    attribute keeps its own value, independently of ``G`` and ``J``, until
    it is deleted, e.g. ``del prop.G12``, which makes it follow ``G`` again.
    A property that sets only ``G`` and ``J`` gives the same matrices as in
    the versions of pyfe3d without ``G12``, ``G13`` and ``GJ``.

    The elements read the stiffness of the cross section through the
    sectional stiffness matrix ``C``, see the Notes. ``C`` follows the 11
    stiffness attributes ``A``, ``E``, ``G12``, ``G13``, ``GJ``, ``Ay``,
    ``Az``, ``Iyy``, ``Izz`` and ``Iyz`` until it is set: reading ``C``
    returns the matrix built from them. Once set, ``C`` keeps its own 21
    independent entries and the elements no longer read the 11 attributes,
    until it is deleted, ``del prop.C``. A property that does not set ``C``
    gives the same matrices as the versions of pyfe3d without ``C``, to
    round-off.


    Attributes
    ----------
    A, : float
        Area of the cross section, dimension length squared.
    E, : float
        Young modulus, force per unit area.
    G, : float
        Shear modulus, force per unit area. It is the value of ``G12`` and
        ``G13`` when they are not set, and it multiplies ``J`` in ``GJ`` when
        ``GJ`` is not set.
    G12, : float
        Transverse shear modulus of the element plane `xy`, of the bending
        with the displacement `v`, force per unit area. Returns ``G`` when
        not set.
    G13, : float
        Transverse shear modulus of the element plane `xz`, of the bending
        with the displacement `w`, force per unit area. Returns ``G`` when
        not set.
    GJ, : float
        Torsional stiffness, dimension force times length squared. Returns
        ``G*J`` when not set.
    Iyy, : float
        Second moment of area about the y axis, the integral of the square of
        `z` over the cross section, dimension length to the fourth.
    Izz, : float
        Second moment of area about the z axis, the integral of the square of
        `y` over the cross section, dimension length to the fourth.
    Iyz, : float
        Product moment of area, the integral of `y` times `z` over the cross
        section, dimension length to the fourth.
    J, : float
        Torsion constant, dimension length to the fourth. The elements use it
        only through ``GJ``, when ``GJ`` is not set.
    Ay, : float
        Integral of `y` over the cross section, dimension length cubed.
    Az, : float
        Integral of `z` over the cross section, dimension length cubed.
    intrho, : float
        Integral of the density over the cross section, dimension mass per
        unit length.
    intrhoy, : float
        Integral of `y` times the density over the cross section, dimension
        mass.
    intrhoz, : float
        Integral of `z` times the density over the cross section, dimension
        mass.
    intrhoy2, : float
        Integral of the square of `y` times the density over the cross
        section, dimension mass times length.
    intrhoz2, : float
        Integral of the square of `z` times the density over the cross
        section, dimension mass times length.
    intrhoyz, : float
        Integral of `y` times `z` times the density over the cross section,
        dimension mass times length.
    C, : np.ndarray
        Sectional stiffness matrix, shape ``(6, 6)``, symmetric, of the
        strains `\{\bar{\varepsilon}, \bar{\gamma}_y, \bar{\gamma}_z,
        \kappa_x, \kappa_y, \kappa_z\}`, see the Notes; dimension force
        times length to the number of its indices ``i``, ``j`` that are
        ``3``, ``4`` or ``5``, e.g. force for ``C[0, 0]`` and force times
        length squared for ``C[4, 4]``. Reading it returns a copy, which is
        the matrix of the 11 stiffness attributes when ``C`` is not set. It
        can be set with any array-like of shape ``(6, 6)``, e.g.
        ``yu_to_pyfe3d(sec.Kt)`` of :mod:`pyfe3d.vabs`, whose asymmetry, if
        any, must not exceed ``1e-10`` times its largest entry; the
        symmetric part is stored.

    Notes
    -----
    The attributes are the following integrals over the cross section, where
    `\rho` Is the density:

    - ``Iyy``: `\int_y \int_z z^2 dy dz`
    - ``Izz``: `\int_y \int_z y^2 dy dz`
    - ``Iyz``: `\int_y \int_z y z dy dz`
    - ``Ay``: `\int_y \int_z y dy dz`
    - ``Az``: `\int_y \int_z z dy dz`
    - ``intrho``: `\int_{y_e} \int_{z_e} \rho(y, z) dy dz`
    - ``intrhoy``: `\int_y \int_z y \rho(y, z) dy dz`
    - ``intrhoz``: `\int_y \int_z z \rho(y, z) dy dz`
    - ``intrhoy2``: `\int_y \int_z y^2 \rho(y, z) dy dz`
    - ``intrhoz2``: `\int_y \int_z z^2 \rho(y, z) dy dz`
    - ``intrhoyz``: `\int_y \int_z y z \rho(y, z) dy dz`

    In the beam elements, see :class:`.BeamC`, the transverse shear strains
    of the rigid cross section at the point `(y, z)` are `\gamma_{xy} - z
    {r_x}_{,x}` and `\gamma_{xz} + y {r_x}_{,x}`, where `\gamma_{xy}` and
    `\gamma_{xz}` are those of the beam axis and `{r_x}_{,x}` is the twist.
    With `G_{12}` = ``G12`` multiplying the first and `G_{13}` = ``G13``
    the second, their integral over the cross section gives the shear part
    of the strain energy per unit length, `\frac{1}{2} \boldsymbol{\gamma}^\top \boldsymbol{D}_s
    \boldsymbol{\gamma}` with `\boldsymbol{\gamma} = \{\gamma_{xy}, \gamma_{xz},
    {r_x}_{,x}\}^\top` and

    .. math::
        \boldsymbol{D}_s = \begin{bmatrix}
            G_{12} A & 0 & -G_{12} A_z \\
            0 & G_{13} A & G_{13} A_y \\
            -G_{12} A_z & G_{13} A_y & G_{12} I_{yy} + G_{13} I_{zz}
        \end{bmatrix}

    The elements replace the last entry, the torsional stiffness of the
    rigid cross section without warping, by ``GJ``, given by the user, e.g.
    `G J` with the torsion constant `J` [1]_ of the Saint-Venant torsion,
    in which the cross section warps freely. ``G12`` and ``G13`` also enter
    the shear parameters `\alpha_y` and `\alpha_z` of the shape functions of
    :class:`.BeamC`. With non-zero ``Ay`` and ``Az``, i.e. when the beam axis
    is not the centroidal axis, the axial strain is coupled with the bending
    curvatures and the transverse shear strains with the twist.
    ``intrhoy``, ``intrhoz`` and ``intrhoyz`` couple the inertia of the
    translations and of the rotations, see :meth:`.BeamC.update_M`.

    **Sectional stiffness matrix.** :class:`.BeamC` and :class:`.BeamLR`
    take the strain energy per unit length as `\frac{1}{2}
    \boldsymbol{\epsilon}^\top \boldsymbol{C} \boldsymbol{\epsilon}`,
    with `\boldsymbol{C}` = ``C`` and the generalized strains of the beam
    axis

    .. math::
        \boldsymbol{\epsilon} = \{\bar{\varepsilon}, \bar{\gamma}_y,
        \bar{\gamma}_z, \kappa_x, \kappa_y, \kappa_z\}^\top
        = \{u_{,x}, v_{,x} - r_z, w_{,x} + r_y, {r_x}_{,x}, -{r_z}_{,x},
        {r_y}_{,x}\}^\top

    i.e. the index ``0`` is the axial strain, ``1`` and ``2`` are the
    transverse shear strains of the planes `xy` and `xz`, ``3`` is the
    twist, and ``4`` and ``5`` are the curvatures that give the axial strain
    `\bar{\varepsilon} + y \kappa_y + z \kappa_z` at the point `(y, z)`
    of the rigid cross section. The work-conjugate sectional forces
    `\boldsymbol{C} \boldsymbol{\epsilon}` are `\{N, V_y, V_z, M_x, M_z,
    M_y\}`, with `M_z = \int \sigma_{xx} y \, dA` and `M_y = \int
    \sigma_{xx} z \, dA`. This is the order of :func:`.yu_to_pyfe3d`,
    which converts the generalized Timoshenko stiffness matrix of the
    strains `\{\gamma_{11}, 2\gamma_{12}, 2\gamma_{13}, \kappa_1,
    \kappa_2, \kappa_3\}` of VABS [2]_, e.g. ``CrossSection.Kt`` of
    :mod:`pyfe3d.vabs`: with `x_2 = y` and `x_3 = z`, `\kappa_1 =
    \kappa_x`, `\kappa_2 = \kappa_z` and `\kappa_3 = -\kappa_y`, so
    the VABS indices ``0, 1, 2, 3, 4, 5`` become ``0, 1, 2, 3, 5, 4``, and
    the entries that couple `\kappa_3` with another strain change sign.

    When ``C`` is not set, it is built from the 11 stiffness attributes, as
    the integral over the rigid cross section given below:

    .. math::
        \boldsymbol{C} = \begin{bmatrix}
            E A & 0 & 0 & 0 & E A_y & E A_z \\
            0 & G_{12} A & 0 & -G_{12} A_z & 0 & 0 \\
            0 & 0 & G_{13} A & G_{13} A_y & 0 & 0 \\
            0 & -G_{12} A_z & G_{13} A_y & G J & 0 & 0 \\
            E A_y & 0 & 0 & 0 & E I_{zz} & E I_{yz} \\
            E A_z & 0 & 0 & 0 & E I_{yz} & E I_{yy}
        \end{bmatrix}

    with `G J` = ``GJ``. This pattern has 11 parameters, against the 21
    independent entries of a general symmetric matrix: the extension-shear
    (``C[0, 1]``, ``C[0, 2]``), extension-twist (``C[0, 3]``), shear-shear
    (``C[1, 2]``), bending-shear (``C[1:3, 4:6]``) and bending-twist
    (``C[3, 4]``, ``C[3, 5]``) couplings of a composite section, e.g. of
    the circumferentially uniform stiffness (CUS) box of [2]_, can only be
    given through ``C``. The shear parameters `\alpha_y` and `\alpha_z` of
    the shape functions of :class:`.BeamC` are taken from the diagonal of
    ``C``, `\alpha_y = -12 C_{44}/(C_{11} L^2)` and `\alpha_z = -12
    C_{55}/(C_{22} L^2)`, which with the matrix above are `-12 E
    I_{zz}/(G_{12} A L^2)` and `-12 E I_{yy}/(G_{13} A L^2)`, see the notes
    of :class:`.BeamC` for the accuracy with couplings. The axial force of
    the geometric stiffness is the first row of `\boldsymbol{C}
    \boldsymbol{\epsilon}`. :class:`.Truss` reads only ``C[0, 0]``, its
    `E A`, and ``C[3, 3]``, its `G J`. The mass integrals ``intrho``, etc.,
    are independent of ``C``.

    The shear correction factors `k_y` and `k_z` of the two planes are
    applied to the moduli, ``G12`` `= k_y G` and ``G13`` `= k_z G`, which
    leaves the torsional stiffness ``G*J`` unchanged. It is possible to have
    more complex scenarios where the different cross section parameters
    ``Ay``, ``Az`` and ``GJ`` have different corrections applied due to
    geometric properties. It is the user's responsability to apply these
    correction factors properly to achieve a general representation of the
    beam element behaviour by means of this property class.

    For beams with homogeneous material along the cross section some of the
    quantities above can be calculated as follows, assuming that ``rho`` is the
    material density:

        ``intrho = A*rho``
        ``intrhoy2 = Izz*rho``
        ``intrhoz2 = Iyy*rho``
        ``intrhoyz = Iyz*rho``

    References
    ----------
    .. [1] Torsion constant, Wikipedia,
       https://en.wikipedia.org/wiki/Torsion_constant
    .. [2] Yu, W., Hodges, D. H., and Ho, J. C., "Variational asymptotic
       beam sectional analysis -- An updated version", International
       Journal of Engineering Science, 59, pp. 40--64, 2012.
       https://doi.org/10.1016/j.ijengsci.2012.03.006

    """
    def __init__(BeamProp self):
        cdef int i
        self.A = 0
        self.E = 0
        self.G = 0
        self.Iyy = 0
        self.Izz = 0
        self.Iyz = 0
        self.J = 0
        self.Ay = 0
        self.Az = 0
        self.intrho = 0
        self.intrhoy = 0
        self.intrhoz = 0
        self.intrhoy2 = 0
        self.intrhoz2 = 0
        self.intrhoyz = 0
        self._G12 = 0
        self._G13 = 0
        self._GJ = 0
        self._G12_given = False
        self._G13_given = False
        self._GJ_given = False
        for i in range(36):
            self._C[i] = 0
        self._C_given = False

    cdef double get_G12(BeamProp self) noexcept nogil:
        # NOTE value read by the elements, see the attribute G12
        if self._G12_given:
            return self._G12
        return self.G

    cdef double get_G13(BeamProp self) noexcept nogil:
        # NOTE value read by the elements, see the attribute G13
        if self._G13_given:
            return self._G13
        return self.G

    cdef double get_GJ(BeamProp self) noexcept nogil:
        # NOTE value read by the elements, see the attribute GJ
        if self._GJ_given:
            return self._GJ
        return self.G*self.J

    cdef void get_C(BeamProp self, double *C) noexcept nogil:
        # NOTE matrix read by the elements, see the attribute C, written to
        #      the 36 entries of C, row by row, in the order {eps, gammay,
        #      gammaz, kx, ky, kz}
        cdef int i
        cdef double G12, G13
        if self._C_given:
            for i in range(36):
                C[i] = self._C[i]
            return
        for i in range(36):
            C[i] = 0
        G12 = self.get_G12()
        G13 = self.get_G13()
        C[0*6 + 0] = self.E*self.A
        C[0*6 + 4] = self.E*self.Ay
        C[4*6 + 0] = C[0*6 + 4]
        C[0*6 + 5] = self.E*self.Az
        C[5*6 + 0] = C[0*6 + 5]
        C[1*6 + 1] = G12*self.A
        C[1*6 + 3] = -G12*self.Az
        C[3*6 + 1] = C[1*6 + 3]
        C[2*6 + 2] = G13*self.A
        C[2*6 + 3] = G13*self.Ay
        C[3*6 + 2] = C[2*6 + 3]
        C[3*6 + 3] = self.get_GJ()
        C[4*6 + 4] = self.E*self.Izz
        C[4*6 + 5] = self.E*self.Iyz
        C[5*6 + 4] = C[4*6 + 5]
        C[5*6 + 5] = self.E*self.Iyy

    # NOTE the properties have no docstring, they are documented in the
    #      Attributes of the class, as the public attributes

    @property
    def G12(BeamProp self):
        return self.get_G12()

    @G12.setter
    def G12(BeamProp self, double value):
        self._G12 = value
        self._G12_given = True

    @G12.deleter
    def G12(BeamProp self):
        self._G12 = 0
        self._G12_given = False

    @property
    def G13(BeamProp self):
        return self.get_G13()

    @G13.setter
    def G13(BeamProp self, double value):
        self._G13 = value
        self._G13_given = True

    @G13.deleter
    def G13(BeamProp self):
        self._G13 = 0
        self._G13_given = False

    @property
    def GJ(BeamProp self):
        return self.get_GJ()

    @GJ.setter
    def GJ(BeamProp self, double value):
        self._GJ = value
        self._GJ_given = True

    @GJ.deleter
    def GJ(BeamProp self):
        self._GJ = 0
        self._GJ_given = False

    @property
    def C(BeamProp self):
        cdef double Cv[36]
        cdef int i
        self.get_C(Cv)
        out = np.empty(36, dtype=np.float64)
        for i in range(36):
            out[i] = Cv[i]
        return out.reshape(6, 6)

    @C.setter
    def C(BeamProp self, value):
        cdef int i, j
        Cin = np.array(value, dtype=np.float64)
        if Cin.shape != (6, 6):
            raise ValueError('C must have shape (6, 6)')
        if not np.all(np.isfinite(Cin)):
            raise ValueError('C must have finite entries')
        if np.abs(Cin - Cin.T).max() > 1.e-10*np.abs(Cin).max():
            raise ValueError('C must be symmetric')
        Cin = (Cin + Cin.T)/2
        for i in range(6):
            for j in range(6):
                self._C[6*i + j] = Cin[i, j]
        self._C_given = True

    @C.deleter
    def C(BeamProp self):
        cdef int i
        for i in range(36):
            self._C[i] = 0
        self._C_given = False
