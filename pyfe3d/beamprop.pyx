#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: nonecheck=False
#cython: overflowcheck=False
#cython: embedsignature=True
#cython: infer_types=False
"""
Beam property module (:mod:`pyfe3d.beamprop`)
=============================================

.. currentmodule:: pyfe3d.beamprop

"""
cdef class BeamProp:
    r"""
    General class to represent beam properties.

    About shear correction factors, usually one should only apply the shear
    correction factor to the shear modulus ``G``. It is, however, possible to
    have more complex scenarios where the different cross section parameters
    ``Ay``, ``Az``, ``J`` have different corrections applied due to geometric
    properties. It is the user's responsability to apply these correction
    factors properly to achieve a general representation of the beam element
    behaviour by means of this property class. No shear correction factors or
    geometric correction factors are applied internally during the calculation
    of the structural matrices.


    Attributes
    ----------
    A, : float
        Area of the cross section
    E, : float
        Young modulus
    G, : float
        Shear modulus
    Iyy, : float
        Second moment of area about the y axis `\int_y \int_z z^2 dy dz`
    Izz, : float
        Second moment of area about the z axis `\int_y \int_z y^2 dy dz`
    Iyz, : float
        Product moment of area `\int_y \int_z y z dy dz`
    J, : float
        Torsion stiffness or torsion constant, see
        https://en.wikipedia.org/wiki/Torsion_constant.
    Ay, : float
        Integral `\int_y \int_z y dy dz`
    Az, : float
        Integral `\int_y \int_z z dy dz`
    intrho, : float
        Integral `\int_{y_e} \int_{z_e} \rho(y, z) dy dz`, where `\rho` Is the
        density
    intrhoy, : float
        Integral `\int_y \int_z y \rho(y, z) dy dz`
    intrhoz, : float
        Integral `\int_y \int_z z \rho(y, z) dy dz`
    intrhoy2, : float
        Integral `\int_y \int_z y^2 \rho(y, z) dy dz`
    intrhoz2, : float
        Integral `\int_y \int_z z^2 \rho(y, z) dy dz`
    intrhoyz, : float
        Integral `\int_y \int_z y z \rho(y, z) dy dz`

    Notes
    -----
    For beams with homogeneous material along the cross section some of the
    quantities above can be calculated as follows, assuming that ``rho`` is the
    material density:

        ``intrho = A*rho``
        ``intrhoy2 = Izz*rho``
        ``intrhoz2 = Iyy*rho``
        ``intrhoyz = Iyz*rho``

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

