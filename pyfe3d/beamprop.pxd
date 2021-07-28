cimport numpy as np

ctypedef np.int64_t cINT
ctypedef np.double_t cDOUBLE

cdef extern from "math.h":
    double cos(double t) nogil
    double sin(double t) nogil
    double atan(double t) nogil

cdef inline double dummy(double d) nogil:
    return d

cdef class BeamProp:
    cdef public double A, E, G, scf, Iyy, Izz, J, Ay, Az
    cdef public double intrho, intrhoy, intrhoz, intrhoy2, intrhoz2, intrhoyz
    cpdef void dummy(BeamProp)
