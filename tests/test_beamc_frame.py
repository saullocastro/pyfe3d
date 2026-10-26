r"""Linear static analysis of an L-shaped frame of BeamC elements

A column along `Y`, clamped at its base, of height `H`, and a beam along `X`
of length `L`, rigidly connected at the top of the column, loaded at the
free end of the beam. With the shear deformation of the Timoshenko theory,
the axial deformation and the Saint-Venant torsion, the displacements of the
free end follow from the principle of virtual work (unit-load method). For
the in-plane load `F_Y = -P`:

.. math::

    v = -P \left(\frac{L^3}{3 E I_{zz}} + \frac{L}{G A}
        + \frac{L^2 H}{E I_{zz}} + \frac{H}{E A}\right), \qquad
    u = \frac{P L H^2}{2 E I_{zz}}, \qquad
    r_z = -P \left(\frac{L H}{E I_{zz}} + \frac{L^2}{2 E I_{zz}}\right)

and for the out-of-plane load `F_Z = P`, which bends both members about
their weak axis and twists the column with the torque `P L`:

.. math::

    w = P \left(\frac{L^3 + H^3}{3 E I_{yy}} + \frac{L + H}{G A}
        + \frac{L^2 H}{G J}\right)

The consistent shape functions of BeamC solve the equilibrium equations of
the Timoshenko beam, such that these nodal displacements are exact with any
number of elements per member.
"""
import sys
sys.path.append('..')

import numpy as np
import pytest
from scipy.sparse import coo_matrix

from pyfe3d.beamprop import BeamProp
from pyfe3d.solver import solve
from pyfe3d import BeamC, BeamCData, BeamCProbe, DOF, INT, DOUBLE


def frame(num_elements, H=3., L=2.):
    r"""Nodes, elements, property and ``KC0`` of the L-shaped frame, with
    ``num_elements`` elements per member"""
    E = 200.e9
    nu = 0.3
    b = 0.05 # width along the element z axis
    h = 0.10 # height along the element y axis
    prop = BeamProp()
    prop.E = E
    # NOTE no shear correction, A is the shear area
    prop.G = E/(2*(1 + nu))
    prop.A = b*h
    prop.Izz = b*h**3/12
    prop.Iyy = h*b**3/12
    prop.J = 1.5e-6

    # NOTE column from (0, 0, 0) to (0, H, 0), then beam to (L, H, 0)
    s = np.linspace(0, 1, num_elements + 1)
    column = np.column_stack((0*s, H*s, 0*s))
    beam = np.column_stack((L*s[1:], H + 0*s[1:], 0*s[1:]))
    ncoords = np.vstack((column, beam))
    x = ncoords.flatten()
    num_nodes = ncoords.shape[0]
    N = DOF*num_nodes

    data = BeamCData()
    probe = BeamCProbe()
    num_beams = num_nodes - 1
    KC0r = np.zeros(data.KC0_SPARSE_SIZE*num_beams, dtype=INT)
    KC0c = np.zeros(data.KC0_SPARSE_SIZE*num_beams, dtype=INT)
    KC0v = np.zeros(data.KC0_SPARSE_SIZE*num_beams, dtype=DOUBLE)
    elements = []
    init_k_KC0 = 0
    for pos1 in range(num_beams):
        pos2 = pos1 + 1
        elem = BeamC(probe)
        elem.n1 = pos1 + 1
        elem.n2 = pos2 + 1
        elem.c1 = DOF*pos1
        elem.c2 = DOF*pos2
        elem.init_k_KC0 = init_k_KC0
        # NOTE the element z axis is Z for both members: in-plane bending
        #      with Izz, out-of-plane bending with Iyy
        if pos1 < num_elements:
            elem.update_rotation_matrix(-1., 0., 0., x) # column
        else:
            elem.update_rotation_matrix(0., 1., 0., x) # beam
        elem.update_probe_xe(x)
        elem.update_KC0(KC0r, KC0c, KC0v, prop)
        elements.append(elem)
        init_k_KC0 += data.KC0_SPARSE_SIZE
    KC0 = coo_matrix((KC0v, (KC0r, KC0c)), shape=(N, N)).tocsc()
    return elements, prop, x, N, KC0


@pytest.mark.parametrize('num_elements', [1, 4])
def test_l_frame_tip_displacements(num_elements):
    H = 3.
    L = 2.
    P = 1.e3
    elements, prop, x, N, KC0 = frame(num_elements, H=H, L=L)
    E, G, A = prop.E, prop.G, prop.A
    Izz, Iyy, J = prop.Izz, prop.Iyy, prop.J

    # NOTE clamped base, the first node
    bk = np.zeros(N, dtype=bool)
    bk[:DOF] = True
    bu = ~bk
    KC0uu = KC0[bu, :][:, bu]
    tip = N - DOF

    # in-plane load at the free end
    f = np.zeros(N)
    f[tip + 1] = -P
    u = np.zeros(N)
    u[bu] = solve(KC0uu, f[bu])
    v_ref = -P*(L**3/(3*E*Izz) + L/(G*A) + L**2*H/(E*Izz) + H/(E*A))
    u_ref = P*L*H**2/(2*E*Izz)
    rz_ref = -P*(L*H/(E*Izz) + L**2/(2*E*Izz))
    print('v', u[tip + 1], v_ref, 'u', u[tip + 0], u_ref)
    assert np.isclose(u[tip + 1], v_ref, rtol=1e-8)
    assert np.isclose(u[tip + 0], u_ref, rtol=1e-8)
    assert np.isclose(u[tip + 5], rz_ref, rtol=1e-8)

    # NOTE the internal forces at the base are the support reactions,
    #      balancing the load and its moment P*L about Z
    fint = np.zeros(N)
    for elem in elements:
        elem.update_probe_ue(u)
        elem.update_fint(fint, prop)
    assert np.allclose(fint[:DOF], [0, P, 0, 0, 0, P*L], atol=1e-6*P*L)
    assert np.allclose(fint[bu], f[bu], atol=1e-6*P)

    # out-of-plane load at the free end
    f = np.zeros(N)
    f[tip + 2] = P
    u = np.zeros(N)
    u[bu] = solve(KC0uu, f[bu])
    w_ref = P*((L**3 + H**3)/(3*E*Iyy) + (L + H)/(G*A) + L**2*H/(G*J))
    print('w', u[tip + 2], w_ref)
    assert np.isclose(u[tip + 2], w_ref, rtol=1e-8)


if __name__ == '__main__':
    test_l_frame_tip_displacements(4)
