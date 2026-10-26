"""Gradients of GradABD against central finite differences

The stiffnesses of shellprop_from_lamination_parameters are linear in the
lamination parameters and polynomial in the thickness, so central finite
differences reproduce their derivatives to round-off. An orthotropic ply is
needed: with an isotropic ply ``u2 = u3 = 0`` and all the derivatives with
respect to the lamination parameters vanish.
"""
import sys
sys.path.append('..')

import numpy as np

from pyfe3d.shellprop import (shellprop_from_lamination_parameters,
                              LaminationParameters, GradABD)
from pyfe3d.shellprop_utils import read_laminaprop

ROWS = ['11', '12', '16', '22', '26', '66']


def get_props(h, xi):
    mat = read_laminaprop((142.e9, 7.72e9, 0.34, 3.80e9, 3.80e9, 3.0e9))
    prop = shellprop_from_lamination_parameters(h, mat, *xi)
    return mat, prop


def stiffnesses(prop, letter):
    return np.array([getattr(prop, letter + r) for r in ROWS])


def test_calc_LP_grad_finite_differences():
    h = 2.3e-3
    xi = [0.5, 0.4, -0.3, -0.6,
          0.2, -0.1, 0.35, 0.15,
          -0.25, 0.3, 0.45, -0.2]
    mat, prop = get_props(h, xi)
    lp = LaminationParameters()
    names = ['xiA1', 'xiA2', 'xiA3', 'xiA4', 'xiB1', 'xiB2', 'xiB3', 'xiB4',
             'xiD1', 'xiD2', 'xiD3', 'xiD4']
    for name, value in zip(names, xi):
        setattr(lp, name, value)
    grad = GradABD()
    grad.calc_LP_grad(h, mat, lp)
    for letter, gradij, offset in [('A', grad.gradAij, 0),
                                   ('B', grad.gradBij, 4),
                                   ('D', grad.gradDij, 8)]:
        gradij = np.asarray(gradij)
        # thickness
        dh = 1e-6*h
        fd = (stiffnesses(get_props(h + dh, xi)[1], letter)
              - stiffnesses(get_props(h - dh, xi)[1], letter))/(2*dh)
        scale = abs(fd).max()
        assert np.allclose(gradij[:, 0], fd, rtol=1e-6, atol=1e-8*scale)
        # lamination parameters
        for j in range(4):
            d = 1e-4
            xip = list(xi)
            xim = list(xi)
            xip[offset + j] += d
            xim[offset + j] -= d
            fd = (stiffnesses(get_props(h, xip)[1], letter)
                  - stiffnesses(get_props(h, xim)[1], letter))/(2*d)
            scale = abs(fd).max()
            print(letter, j + 1, gradij[:, j + 1], fd)
            assert np.allclose(gradij[:, j + 1], fd, rtol=1e-6,
                               atol=1e-8*scale), (letter, j + 1)


if __name__ == '__main__':
    test_calc_LP_grad_finite_differences()
