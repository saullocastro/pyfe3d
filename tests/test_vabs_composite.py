r"""Cross-sectional analysis of composite sections

Verification of :class:`pyfe3d.vabs.CrossSection` against published VABS
results:

- thin-walled box beam CUS, Table 1 and Table 2 (VABS 3.2 column) of Yu et
  al. (2012), p. 56-57: all nine non-negligible entries, signs included;
- anisotropic pipe of Fig. 4 and Tables 2-3 of Yu and Hodges (2005),
  p. 51-52, and its mass properties, Eq. (26);
- thin-walled box beam CAS1, Tables 1-2 of Yu et al. (2002), p. 5112, the
  classical stiffness of the column VABS_R.

The plies are oriented with :func:`pyfe3d.vabs.rotate_material`: the local
axis `y_2` of each wall is tangent to the wall, counter-clockwise around
the section, `y_3` points inwards, and the ply angle is measured from the
beam axis towards `y_2`, see the figure of :class:`pyfe3d.vabs.CrossSection`.

This convention of pyfe3d reproduces all signs of the CUS box. With it, the
pipe and CAS1 give three couplings each with the published magnitude and
the opposite sign: ``S14``, ``S25``, ``S36`` of the pipe and ``S12``,
``S45``, ``S46`` of CAS1. These differences are attributed to the
labelling of the axes and ply angles in those papers, as explained in the
docstrings of the two tests, which check the magnitudes and assert the
opposite signs explicitly.

References: Yu, W., Hodges, D. H., and Ho, J. C., International Journal of
Engineering Science, 59, pp. 40-64, 2012; Yu, W., and Hodges, D. H.,
Journal of the American Helicopter Society, 50(1), pp. 46-55, 2005; Yu, W.,
Hodges, D. H., Volovoi, V., and Cesnik, C. E. S., International Journal of
Solids and Structures, 39(19), pp. 5101-5121, 2002.
"""
import sys
import os

import numpy as np
import pytest

from pyfe3d.vabs import (CrossSection, orthotropic, classical_from_timoshenko,
                         yu_to_pyfe3d, pyfe3d_to_yu)

sys.path.append(os.path.dirname(__file__))
import vabs_meshes as vm


def pipe_mesh(ns, na, nt, etype, L=2., Ri=0.3, Ro=0.5):
    r"""Pipe of Fig. 4 of Yu and Hodges (2005): two straight walls of length
    ``L`` and two half circles of inner radius ``Ri`` and outer radius
    ``Ro``; walls numbered bottom, right, top, left"""
    def straight(y_out, y_in, x0, x1):
        def P(u, v):
            return x0 + (x1 - x0)*u, (1 - v)*y_out + v*y_in
        return P

    def arc(xc, phi0):
        def P(u, v):
            phi = phi0 + np.pi*u
            r = Ro - (Ro - Ri)*v
            return xc + r*np.cos(phi), r*np.sin(phi)
        return P
    h = L/2
    return vm.walls([(straight(-Ro, -Ri, -h, h), ns), (arc(h, -np.pi/2), na),
                     (straight(Ro, Ri, h, -h), ns), (arc(-h, np.pi/2), na)],
                    nt, etype)


@pytest.mark.slow
def test_CUS_Yu2012_Table2():
    r"""Box beam CUS: outer width ``0.953 in.``, height ``0.53 in.``,
    thickness ``0.03 in.``, six plies at 15 degrees in every wall, Table 1
    of Yu et al. (2012). The mesh has 15 and 10 elements along the width and
    the height and 6 across the walls, the same numbers as the VABS mesh
    described on p. 5112 of Yu et al. (2002), to which Yu et al. (2012),
    p. 56, refer, here with 8-node instead of 6-node quadrilaterals. All
    nine entries agree with the VABS 3.2 column of Table 2 to its four
    digits. With a finer mesh, ``S22`` and ``S33`` decrease by up to
    0.3 %."""
    D = orthotropic(E1=20.59e6, E2=1.42e6, E3=1.42e6, G12=0.87e6,
                    G13=0.87e6, G23=0.696e6, nu12=0.30, nu13=0.30, nu23=0.34)
    x2, x3, els, wall, layer, theta1 = vm.box(0.953, 0.53, 0.03, 15, 10, 6, 8)
    sec = CrossSection(x2, x3, els, [D], [0]*len(els), theta1=theta1,
                       theta3=15.*np.ones(len(els)))
    sec.calc_stiffness()
    Kt = sec.Kt
    ref = {(0, 0): 1.437e6, (0, 3): 1.074e5, (1, 1): 9.027e4,
           (1, 4): -5.201e4, (2, 2): 3.943e4, (2, 5): -5.635e4,
           (3, 3): 1.679e4, (4, 4): 6.621e4, (5, 5): 1.725e5}
    for (i, j), v in ref.items():
        assert '%.3e' % Kt[i, j] == '%.3e' % v, (i, j, Kt[i, j], v)
    # NOTE "the other components are negligible", Yu et al. (2012), p. 56
    for i in range(6):
        for j in range(i, 6):
            if (i, j) not in ref:
                assert abs(Kt[i, j]) < 1e-10*np.sqrt(Kt[i, i]*Kt[j, j])
    # NOTE the classical stiffness is the condensed Timoshenko stiffness
    assert np.allclose(classical_from_timoshenko(Kt), sec.Kc, rtol=1e-9,
                       atol=1e-9*np.abs(sec.Kc).max())
    assert np.allclose(pyfe3d_to_yu(yu_to_pyfe3d(Kt)), Kt)
    # NOTE same classical stiffness with the constraints of Yu et al. (2002)
    Kc = sec.Kc.copy()
    sec.calc_stiffness('yu2002')
    assert np.allclose(sec.Kc, Kc, rtol=1e-10, atol=1e-10*np.abs(Kc).max())


@pytest.mark.slow
def test_pipe_Yu2005_Table3():
    r"""Anisotropic pipe of Fig. 4 and Table 2 of Yu and Hodges (2005): two
    layers of thickness ``0.1 in.``, 0 degrees outside and 90 degrees
    inside in the straight walls, -45 degrees outside and 45 degrees inside
    in the half circles, ``El = 20.59e6 psi``, ``Et = 1.42e6 psi``, ``Glt =
    Gtn = 8.7e5 psi``, ``nu_lt = nu_tn = 0.42``, ``rho = 0.057 lb
    sec^2/in.^4``.

    The mass properties of Eq. (26) are reproduced to their six digits. The
    stiffnesses agree with the VABS column of Table 3 within 0.5 % and with
    the SVBT column within 0.3 %. The magnitudes of the extension-twist and
    shear-bending couplings ``S14``, ``S25`` and ``S36`` agree, but with the
    ply angles read from Fig. 4 their signs are opposite to those of
    Table 3; the signs of Table 3 are obtained with the opposite ply angles
    in the half circles. Yu and Hodges (2005) do not state the convention
    of the ply angles of Fig. 4. With the convention of pyfe3d,
    :func:`pyfe3d.vabs.rotate_material`, which reproduces all signs of the
    CUS box of Yu et al. (2012), the outer -45 degree layer dominates and
    gives a negative ``S14``; the difference is attributed to the labelling
    of the ply angles of Fig. 4, and the convention of pyfe3d is kept.
    """
    D = orthotropic(20.59e6, 1.42e6, 1.42e6, 8.7e5, 8.7e5, 8.7e5,
                    0.42, 0.42, 0.42)
    k = 4
    x2, x3, els, wall, layer, theta1 = pipe_mesh(40, 40, 2*k, 8)
    outer = layer < k
    straight = (wall % 2) == 0
    theta3 = np.zeros(len(els))
    theta3[straight & ~outer] = 90.
    theta3[~straight & outer] = -45.
    theta3[~straight & ~outer] = 45.
    sec = CrossSection(x2, x3, els, [D], [0]*len(els), theta1=theta1,
                       theta3=theta3, densities=[0.057])
    sec.calc_mass()
    assert '%.5e' % sec.mass_per_length == '7.42513e-02'
    assert '%.5e' % sec.mass_matrix[4, 4] == '9.88336e-03'
    assert '%.5e' % sec.mass_matrix[5, 5] == '6.11827e-02'
    assert np.allclose(sec.mass_centre, 0, atol=1e-12)
    sec.calc_stiffness()
    Kt = sec.Kt
    vabs = {(0, 0): 1.03890e7, (1, 1): 7.84299e5, (2, 2): 3.29002e5,
            (0, 3): 9.82878e4, (1, 4): -8.18782e3, (2, 5): -5.18541e4,
            (3, 3): 6.86973e5, (4, 4): 1.88236e6, (5, 5): 5.38972e6}
    svbt = {(0, 0): 1.03892e7, (1, 1): 7.85310e5, (2, 2): 3.29279e5,
            (0, 3): 9.84575e4, (1, 4): -8.21805e3, (2, 5): -5.20981e4,
            (3, 3): 6.87275e5, (4, 4): 1.88238e6, (5, 5): 5.38987e6}
    for (i, j), v in vabs.items():
        assert np.isclose(abs(Kt[i, j]), abs(v), rtol=5e-3)
        assert np.isclose(abs(Kt[i, j]), abs(svbt[(i, j)]), rtol=3e-3)
    # NOTE S14, S25 and S36 have the opposite signs of Table 3, see the
    #      docstring; all other entries have the signs of Table 3
    flipped = [(0, 3), (1, 4), (2, 5)]
    for (i, j), v in vabs.items():
        s = -1 if (i, j) in flipped else 1
        assert np.sign(Kt[i, j]) == s*np.sign(v), (i, j, Kt[i, j], v)
    # NOTE the signs of Table 3 with the opposite ply angles in the half
    #      circles
    sec2 = CrossSection(x2, x3, els, [D], [0]*len(els), theta1=theta1,
                        theta3=np.where(straight, theta3, -theta3))
    sec2.calc_stiffness()
    for (i, j), v in vabs.items():
        assert np.isclose(sec2.Kt[i, j], v, rtol=5e-3), (i, j, sec2.Kt[i, j])
    for i in range(6):
        for j in range(i, 6):
            if (i, j) not in vabs:
                assert abs(Kt[i, j]) < 1e-10*np.sqrt(Kt[i, i]*Kt[j, j])


@pytest.mark.slow
def test_CAS1_Yu2002_Table2():
    r"""Box beam CAS1 of Table 1 of Yu et al. (2002), same geometry as the
    CUS box, ``nu_lt = nu_tn = 0.42``, ``Glt = 8.7e5 psi``, ``Gtn = 6.96e5
    psi``; right wall ``[15/-15]_3``, left wall ``[-15/15]_3``, upper wall
    ``[-15]_6``, lower wall ``[15]_6``, read from the inner surface, as the
    layers of Table 2 of Yu and Hodges (2005). The classical stiffness is
    compared with the column VABS_R of Table 2, which is the inverse of the
    4 x 4 classical block of the flexibility, i.e. the classical stiffness.

    The magnitudes of the five main entries agree to the three digits of
    the table, those of the small couplings ``S46`` and ``S56`` within
    0.5 %. The signs of ``S45``, ``S46`` and of the extension-shear
    coupling ``S12`` of the Timoshenko matrix are opposite to those of
    Table 2; they are those of Table 2 if the walls named upper and right
    are those at negative `x_3` and `x_2`, i.e. for the section rotated by
    180 degrees. Yu et al. (2002) show no sketch of the axes of CAS1; the
    difference is attributed to the labelling of the walls, and the
    convention of pyfe3d, which reproduces all signs of the CUS box, is
    kept.
    """
    D = orthotropic(20.59e6, 1.42e6, 1.42e6, 8.7e5, 8.7e5, 6.96e5,
                    0.42, 0.42, 0.42)
    x2, x3, els, wall, layer, theta1 = vm.box(0.953, 0.53, 0.03, 15, 10, 6, 8)
    # NOTE position of the ply in the stack, 0 for the innermost
    pos = 5 - layer
    theta3 = np.zeros(len(els))
    theta3[wall == 0] = 15.
    theta3[wall == 2] = -15.
    theta3[wall == 1] = np.where(pos[wall == 1] % 2 == 0, 15., -15.)
    theta3[wall == 3] = np.where(pos[wall == 3] % 2 == 0, -15., 15.)
    sec = CrossSection(x2, x3, els, [D], [0]*len(els), theta1=theta1,
                       theta3=theta3)
    sec.calc_stiffness()
    Kc = sec.Kc
    # NOTE (i, j) of Kc, {gamma11, kappa1, kappa2, kappa3}
    main = {(0, 0): 0.985e6, (1, 1): 0.170e5, (1, 2): 0.176e5,
            (2, 2): 0.591e5, (3, 3): 0.141e6}
    for (i, j), v in main.items():
        assert '%.2e' % abs(Kc[i, j]) == '%.2e' % v, (i, j, Kc[i, j], v)
    small = {(1, 3): -0.351e3, (2, 3): -0.372e3}
    for (i, j), v in small.items():
        assert np.isclose(abs(Kc[i, j]), abs(v), rtol=5e-3)
    assert abs(Kc[0, 1]) < 1e-8*np.sqrt(Kc[0, 0]*Kc[1, 1])
    # NOTE S45 and S46 have the opposite signs of Table 2, see the
    #      docstring; S11, S44, S55, S66 and S56 have the signs of Table 2
    table = dict(main)
    table.update(small)
    flipped = [(1, 2), (1, 3)]
    for (i, j), v in table.items():
        s = -1 if (i, j) in flipped else 1
        assert np.sign(Kc[i, j]) == s*np.sign(v), (i, j, Kc[i, j], v)
    # NOTE ... and so has S12 of the Timoshenko matrix, negative in Table 2
    assert sec.Kt[0, 1] > 0
    # NOTE section rotated by 180 degrees: all signs of Table 2
    sec = CrossSection(-x2, -x3, els, [D], [0]*len(els), theta1=theta1 + 180,
                       theta3=theta3)
    sec.calc_stiffness()
    Kt = sec.Kt
    signs = {(0, 1): -1, (0, 2): -1, (1, 2): 1, (3, 4): 1, (3, 5): -1,
             (4, 5): -1}
    for (i, j), s in signs.items():
        assert np.sign(Kt[i, j]) == s
    assert np.allclose(sec.Kc, Kc*np.outer([1, 1, -1, -1], [1, 1, -1, -1]),
                       rtol=1e-8, atol=1e-8*np.abs(Kc).max())


if __name__ == '__main__':
    test_CUS_Yu2012_Table2()
    test_pipe_Yu2005_Table3()
    test_CAS1_Yu2002_Table2()
