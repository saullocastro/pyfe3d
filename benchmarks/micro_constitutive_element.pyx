# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
r"""Cost of ShellProp.get_constitutive_element without Python overhead

Compiled and run by ``bench_constitutive_element.py``.

"""
from libc.math cimport cos, sin
import time

from pyfe3d.shellprop cimport ShellProp


def bench(ShellProp prop, long n, double thetadeg, int full):
    r"""Time per call in ns

    With ``full=0`` only the transverse shear stiffness is computed, with
    ``full=1`` also the A, B and D matrices.

    """
    cdef long i
    cdef double c = cos(thetadeg*3.141592653589793/180.)
    cdef double s = sin(thetadeg*3.141592653589793/180.)
    cdef double A[9]
    cdef double B[9]
    cdef double D[9]
    cdef double Ats[4]
    t0 = time.perf_counter()
    with nogil:
        for i in range(n):
            if full:
                prop.get_constitutive_element(c, -s, s, c, A, B, D, Ats)
            else:
                prop.get_constitutive_element(c, -s, s, c, NULL, NULL, NULL,
                                              Ats)
    return (time.perf_counter() - t0)/n*1e9
