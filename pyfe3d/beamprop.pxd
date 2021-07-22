cimport numpy as np

ctypedef np.int64_t cINT
ctypedef np.double_t cDOUBLE

cdef inline double dummy() nogil:
    return 0.

cdef class BeamProp:
    cdef public double A, E, G, scf, Iyy, Izz, J, Ay, Az
    cdef public double intrho, intrhoy, intrhoz, intrhoy2, intrhoz2, intrhoyz
