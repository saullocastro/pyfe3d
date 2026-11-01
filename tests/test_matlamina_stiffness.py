"""Stiffness constants of MatLamina against the inverse of the compliance

For an orthotropic material in its principal axes, the stiffness matrix is
the inverse of the compliance matrix, with ``S11 = 1/E1``, ``S12 =
-nu12/E1``, ``S13 = -nu13/E1``, ``S22 = 1/E2``, ``S23 = -nu23/E2``, ``S33 =
1/E3``, ``S44 = 1/G23``, ``S55 = 1/G13`` and ``S66 = 1/G12``. A material with
``E2 != E3`` and a non-zero ``nu23`` distinguishes ``nu23`` from ``nu32``.

The constitutive constants ``q11`` to ``q33`` are by default those of plane
stress, ``sigma33 = 0``: the inverse of the reduced compliance with ``S11 =
1/E1``, ``S12 = -nu12/E1``, ``S22 = 1/E2`` (Jones, R. M., Mechanics of
Composite Materials, 2nd ed., 1999, Eqs. (2.61) and (2.62), p. 71), equal
to the condensed ``Cij - Ci3 Cj3/C33``, Eq. (2.64), p. 72; after
``rebuild(plane_stress=False)`` they are the three-dimensional ``Cij``.
"""
import sys
sys.path.append('..')

import numpy as np

from pyfe3d.shellprop_utils import read_laminaprop

E1, E2, E3 = 150.e9, 8.7e9, 10.1e9
NU12, NU13, NU23 = 0.28, 0.3, 0.45
G12, G13, G23 = 5.0e9, 5.5e9, 3.1e9


def get_mat_and_C():
    mat = read_laminaprop((E1, E2, NU12, G12, G13, G23, E3, NU13, NU23))
    S = np.zeros((6, 6))
    S[0, 0] = 1/E1
    S[1, 1] = 1/E2
    S[2, 2] = 1/E3
    S[0, 1] = S[1, 0] = -NU12/E1
    S[0, 2] = S[2, 0] = -NU13/E1
    S[1, 2] = S[2, 1] = -NU23/E2
    S[3, 3] = 1/G23
    S[4, 4] = 1/G13
    S[5, 5] = 1/G12
    return mat, np.linalg.inv(S)


def test_c_against_inverse_compliance():
    mat, C = get_mat_and_C()
    for name in ['c11', 'c12', 'c13', 'c22', 'c23', 'c33', 'c44', 'c55',
                 'c66']:
        i, j = int(name[1]) - 1, int(name[2]) - 1
        print(name, getattr(mat, name), C[i, j])
        assert np.isclose(getattr(mat, name), C[i, j], rtol=1e-12), name


def test_q_against_inverse_compliance():
    mat, C = get_mat_and_C()
    # plane stress, the default
    assert mat.plane_stress
    Sr = np.array([[1/E1, -NU12/E1], [-NU12/E1, 1/E2]])
    Q = np.linalg.inv(Sr)
    Qc = C[:2, :2] - np.outer(C[:2, 2], C[:2, 2])/C[2, 2]
    assert np.allclose(Q, Qc, rtol=1e-12, atol=0)
    for i in range(3):
        for j in range(3):
            name = 'q%d%d' % (i + 1, j + 1)
            ref = Q[i, j] if i < 2 and j < 2 else 0.
            assert np.isclose(getattr(mat, name), ref, rtol=1e-12,
                              atol=0), name
    for name in ['q44', 'q55', 'q66']:
        i = int(name[1]) - 1
        assert np.isclose(getattr(mat, name), C[i, i], rtol=1e-12), name
    # three-dimensional option
    mat.rebuild(plane_stress=False)
    assert not mat.plane_stress
    for i in range(3):
        for j in range(3):
            name = 'q%d%d' % (i + 1, j + 1)
            print(name, getattr(mat, name), C[i, j])
            assert np.isclose(getattr(mat, name), C[i, j], rtol=1e-12), name
    for name in ['q44', 'q55', 'q66']:
        i = int(name[1]) - 1
        assert np.isclose(getattr(mat, name), C[i, i], rtol=1e-12), name


if __name__ == '__main__':
    test_c_against_inverse_compliance()
    test_q_against_inverse_compliance()
