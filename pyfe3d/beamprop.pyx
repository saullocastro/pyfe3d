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


    Attributes
    ----------
    A, : float
        Area of the cross section, dimension length squared.
    E, : float
        Young modulus, force per unit area.
    G, : float
        Shear modulus, force per unit area.
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
        Torsion stiffness or torsion constant. The elements multiply it by
        ``G``, such that it has the dimension length to the fourth.
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

    ``J`` is the torsion constant [1]_. In the strain energy of the beam
    elements, see :class:`.BeamC`, ``G*J`` multiplies the square of the
    twist `{r_x}_{,x}`, in place of the polar moment `\int_y \int_z (y^2 +
    z^2) dy dz` of a cross section without warping. With non-zero ``Ay`` and
    ``Az``, i.e. when the beam axis is not the centroidal axis, the axial
    strain is coupled with the bending curvatures and the transverse shear
    strains with the twist. ``intrhoy``, ``intrhoz`` and ``intrhoyz`` couple
    the inertia of the translations and of the rotations, see
    :meth:`.BeamC.update_M`.

    About shear correction factors, usually one should only apply the shear
    correction factor to the shear modulus ``G``. It is, however, possible to
    have more complex scenarios where the different cross section parameters
    ``Ay``, ``Az``, ``J`` have different corrections applied due to geometric
    properties. It is the user's responsability to apply these correction
    factors properly to achieve a general representation of the beam element
    behaviour by means of this property class.

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

    """
    def __init__(BeamProp self):
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

