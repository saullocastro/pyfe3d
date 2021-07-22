cdef inline double dummy() nogil:
    return 0.

cdef class BeamProp:
    cdef public double A, E, G, scf, Iyy, Izz, J, Ay, Az
    cdef public double intrho, intrhoy, intrhoz, intrhoy2, intrhoz2, intrhoyz
