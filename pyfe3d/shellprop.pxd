cimport numpy as np

ctypedef np.int64_t cINT
ctypedef np.double_t cDOUBLE

cdef extern from "math.h":
    double cos(double t) nogil
    double sin(double t) nogil
    double atan(double t) nogil

cdef inline double deg2rad(double thetadeg) nogil:
    return thetadeg*4*atan(1.)/180.

cdef class LaminationParameters:
    cdef public double xiA1, xiA2, xiA3, xiA4
    cdef public double xiB1, xiB2, xiB3, xiB4
    cdef public double xiD1, xiD2, xiD3, xiD4
    cdef public double xiE1, xiE2, xiE3, xiE4

cdef class MatLamina:
    cdef public double e1, e2, e3, g12, g13, g23, nu12, nu21, nu13, nu31, nu23, nu32
    cdef public double rho, a1, a2, a3, tref
    cdef public double st1, st2, sc1, sc2, ss12
    cdef public double q11, q12, q13, q21, q22, q23, q31, q32, q33, q44, q55, q66
    cdef public double c11, c12, c13, c22, c23, c33, c44, c55, c66
    cdef public double u1, u2, u3, u4, u5, u6, u7
    cpdef void rebuild(MatLamina)
    cpdef cDOUBLE[:, :] get_constitutive_matrix(MatLamina)
    cpdef cDOUBLE[:, :] get_invariant_matrix(MatLamina)


cdef class Lamina:
    cdef public cINT plyid
    cdef public double h, thetadeg, cost, cos2t, cos4t, sint, sin2t, sin4t
    cdef public double q11L, q12L, q22L, q16L, q26L, q66L, q44L, q45L, q55L
    cdef public MatLamina matlamina
    cpdef void rebuild(Lamina)
    cpdef cDOUBLE[:, :] get_transf_matrix_displ_to_laminate(Lamina)
    cpdef cDOUBLE[:, :] get_constitutive_matrix(Lamina)
    cpdef cDOUBLE[:, :] get_transf_matrix_stress_to_lamina(Lamina)
    cpdef cDOUBLE[:, :] get_transf_matrix_stress_to_laminate(Lamina)

cdef class ShellProp:
    cdef public double A11, A12, A16, A22, A26, A66
    cdef public double B11, B12, B16, B22, B26, B66
    cdef public double D11, D12, D16, D22, D26, D66
    cdef public double E44, E45, E55
    cdef public double e1, e2, g12, nu12, nu21
    cdef public double scf_k13, scf_k23, h, offset, rho, intrho, intrhoz, intrhoz2
    cdef public list plies
    cdef public list stack
    cdef cDOUBLE[:, :] get_A(ShellProp)
    cdef cDOUBLE[:, :] get_B(ShellProp)
    cdef cDOUBLE[:, :] get_D(ShellProp)
    cdef cDOUBLE[:, :] get_E(ShellProp)
    cdef cDOUBLE[:, :] get_ABD(ShellProp)
    cdef cDOUBLE[:, :] get_ABDE(ShellProp)
    cpdef void rebuild(ShellProp)
    cpdef void calc_scf(ShellProp)
    cpdef void calc_equivalent_properties(ShellProp)
    cpdef void calc_constitutive_matrix(ShellProp)
    cpdef void force_orthotropic(ShellProp)
    cpdef void force_symmetric(ShellProp)
    cpdef LaminationParameters calc_lamination_parameters(ShellProp)

