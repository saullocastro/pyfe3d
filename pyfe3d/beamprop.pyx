#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: nonecheck=False
#cython: infer_types=False
"""
Beam property module (:mod:`pyfe3d.beamprop`)
=============================================

.. currentmodule:: pyfe3d.beamprop

"""
import numpy as np

INT = np.int64
DOUBLE = np.float64

cdef class BeamProp:
    r"""
    Attributes
    ----------
    A : float
        Area of the cross section
    E : float
        Young modulus
    G : float
        Shear modulus
    scf : float
        Shear correction factor
    Iyy : float
        Second moment of area about the y axis `\int_y \int_z y^2 dy dz`
    Izz : float
        Second moment of area about the z axis `\int_y \int_z z^2 dy dz`
    J : float
        Integral `\int_y \int_z y z dy dz`
    Ay : float
        Integral `\int_y \int_z y dy dz`
    Az : float
        Integral `\int_y \int_z z dy dz`
    intrho : float
        Integral `\int_{y_e} \int_{z_e} \rho(y, z) dy dz`, where `\rho` Is the
        density
    intrhoy : float
        Integral `\int_y \int_z y \rho(y, z) dy dz`
    intrhoz : float
        Integral `\int_y \int_z z \rho(y, z) dy dz`
    intrhoy2 : float
        Integral `\int_y \int_z y^2 \rho(y, z) dy dz`
    intrhoz2 : float
        Integral `\int_y \int_z z^2 \rho(y, z) dy dz`
    intrhoyz : float
        Integral `\int_y \int_z y z \rho(y, z) dy dz`

    """
    def __cinit__(BeamProp self):
        self.scf = 5/6.

