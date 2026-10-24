"""Approximation functions and plate properties of the Ritz models of the tests

The Legendre-based approximation functions of Castro and Donadon (2017) and
the bending and transverse shear stiffnesses of an isotropic plate, used by
the Ritz models of the tests of the solvers. They depend only on NumPy.

Reference: Saullo G. P. Castro, Maurício V. Donadon. "Assembly of
Semi-Analytical models to Address Linear Buckling and Vibration of Stiffened
Composite Panels with Debonding Defect". Composite Structures, 2017.
https://doi.org/10.1016/j.compstruct.2016.10.026

"""
from functools import lru_cache
from types import SimpleNamespace

import numpy as np
from numpy.polynomial import legendre


#: Power series coefficients of the cubic Hermite functions, which control
#: the displacement and the rotation at each extremity
_HERMITE = np.array([[0.5, -0.75, 0., 0.25],
                     [0.125, -0.125, -0.125, 0.125],
                     [0.5, 0.75, 0., -0.25],
                     [-0.125, -0.125, 0.125, 0.125]])


@lru_cache(maxsize=None)
def _coefficients(n, d1t, d1r, d2t, d2r, der):
    """Legendre series coefficients, one column per function"""
    C = np.zeros((max(n, 4), max(n, 4)))
    for i, flag in enumerate((d1t, d1r, d2t, d2r)):
        C[:4, i] = flag*legendre.poly2leg(_HERMITE[i])
    for i in range(4, n):
        Pi_2 = np.zeros(i - 1)
        Pi_2[-1] = 1.
        C[:i+1, i] = legendre.legint(Pi_2, m=2, lbnd=-1)
    C = legendre.legder(C[:, :n], m=der, axis=0) if der else C[:, :n]
    C.flags.writeable = False
    return C


def vecf(n, xi, d1t, d1r, d2t, d2r):
    r"""Hierarchical functions of Bardell at the natural coordinate ``xi``

    The first four are cubic Hermite functions, whose displacement and
    rotation at `\xi=-1` and `\xi=+1` are controlled by the flags ``d1t``,
    ``d1r``, ``d2t`` and ``d2r``. The following ones are the Legendre
    polynomials `P_{i-2}` integrated twice from `-1`, which vanish with
    their first derivative at both extremities.

    """
    return legendre.legval(xi, _coefficients(n, d1t, d1r, d2t, d2r, 0))


def vecfxi(n, xi, d1t, d1r, d2t, d2r):
    """First derivative of :func:`vecf` with respect to ``xi``"""
    return legendre.legval(xi, _coefficients(n, d1t, d1r, d2t, d2r, 1))


def vecfxixi(n, xi, d1t, d1r, d2t, d2r):
    """Second derivative of :func:`vecf` with respect to ``xi``"""
    return legendre.legval(xi, _coefficients(n, d1t, d1r, d2t, d2r, 2))


def isotropic_plate(thickness, E, nu):
    """Bending and transverse shear stiffnesses of an isotropic plate

    The transverse shear stiffnesses use the shear correction factor 5/6.

    """
    D = E*thickness**3/(12*(1 - nu**2))
    G = E/(2*(1 + nu))
    return SimpleNamespace(
        D11=D, D12=nu*D, D16=0., D22=D, D26=0., D66=(1 - nu)*D/2,
        A44=5/6*G*thickness, A45=0., A55=5/6*G*thickness)
