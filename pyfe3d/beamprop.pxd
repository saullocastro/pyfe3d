cdef class BeamProp:
    cdef public double A, E, G, Iyy, Izz, Iyz, J, Ay, Az
    cdef public double intrho, intrhoy, intrhoz, intrhoy2, intrhoz2, intrhoyz
    cdef double _G12, _G13, _GJ
    cdef bint _G12_given, _G13_given, _GJ_given
    cdef double get_G12(BeamProp self) noexcept nogil
    cdef double get_G13(BeamProp self) noexcept nogil
    cdef double get_GJ(BeamProp self) noexcept nogil
