"""Geometrically nonlinear static analyses with a full Newton-Raphson method

Since KT = KC0 + KCNL(u) + KG(u) is the exact Jacobian of the internal forces
given by ``update_fint(..., nonlinear=1)``, the Newton-Raphson iteration
converges quadratically. Two classical problems where the membrane action
stiffens the structure are verified:

- simply supported square plate with immovable edges under uniform pressure,
  with the Quad4, Quad4R and Tria3R elements
- beam clamped at both ends under a central transverse load, arbitrarily
  oriented in space, with the BeamC and BeamLR elements
"""
import sys
sys.path.append('..')

import numpy as np
import pytest
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

from pyfe3d.beamprop import BeamProp
from pyfe3d.shellprop_utils import isotropic_plate
from pyfe3d import (Quad4, Quad4Data, Quad4Probe, Quad4R, Quad4RData,
                    Quad4RProbe, Tria3R, Tria3RData, Tria3RProbe, BeamC,
                    BeamCData, BeamCProbe, BeamLR, BeamLRData, BeamLRProbe,
                    DOF, INT, DOUBLE)

SHELLS = {
    'Quad4': (Quad4, Quad4Probe, Quad4Data),
    'Quad4R': (Quad4R, Quad4RProbe, Quad4RData),
    'Tria3R': (Tria3R, Tria3RProbe, Tria3RData),
}
BEAMS = {
    'BeamC': (BeamC, BeamCProbe, BeamCData, 20),
    'BeamLR': (BeamLR, BeamLRProbe, BeamLRData, 80),
}


def sparse(r, c, v, N):
    return coo_matrix((v, (r, c)), shape=(N, N)).tocsc()


def solve(elements, data, prop, x, N, bu, fext, max_iter=20, tol=1.e-10):
    """Linear solution and full Newton-Raphson solution

    Returns the linear displacements, the nonlinear displacements and the
    history of the residual norms relative to the external forces.
    """
    num_elements = len(elements)
    KC0r = np.zeros(data.KC0_SPARSE_SIZE*num_elements, dtype=INT)
    KC0c = np.zeros(data.KC0_SPARSE_SIZE*num_elements, dtype=INT)
    KC0v = np.zeros(data.KC0_SPARSE_SIZE*num_elements, dtype=DOUBLE)
    KCNLr = np.zeros(data.KCNL_SPARSE_SIZE*num_elements, dtype=INT)
    KCNLc = np.zeros(data.KCNL_SPARSE_SIZE*num_elements, dtype=INT)
    KCNLv = np.zeros(data.KCNL_SPARSE_SIZE*num_elements, dtype=DOUBLE)
    KGr = np.zeros(data.KG_SPARSE_SIZE*num_elements, dtype=INT)
    KGc = np.zeros(data.KG_SPARSE_SIZE*num_elements, dtype=INT)
    KGv = np.zeros(data.KG_SPARSE_SIZE*num_elements, dtype=DOUBLE)
    for elem in elements:
        elem.update_probe_xe(x)
        elem.update_KC0(KC0r, KC0c, KC0v, prop)
    KC0uu = sparse(KC0r, KC0c, KC0v, N)[bu, :][:, bu]

    u_linear = np.zeros(N)
    u_linear[bu] = spsolve(KC0uu, fext[bu])

    u = np.zeros(N)
    residuals = []
    for iteration in range(max_iter):
        fint = np.zeros(N)
        KCNLv[:] = 0
        KGv[:] = 0
        for elem in elements:
            # NOTE the probe is shared amongst the elements
            elem.update_probe_xe(x)
            elem.update_probe_ue(u)
            elem.update_fint(fint, prop, nonlinear=1)
            elem.update_KCNL(KCNLr, KCNLc, KCNLv, prop)
            elem.update_KG(KGr, KGc, KGv, prop)
        residual = fext[bu] - fint[bu]
        residuals.append(np.linalg.norm(residual)/np.linalg.norm(fext[bu]))
        if residuals[-1] < tol:
            break
        KT = sparse(KC0r, KC0c, KC0v, N) + sparse(KCNLr, KCNLc, KCNLv, N) + sparse(KGr, KGc, KGv, N)
        u[bu] += spsolve(KT[bu, :][:, bu], residual)
    return u_linear, u, residuals


def plate(name, nx=11, ny=11):
    """Simply supported square plate with immovable edges, uniform pressure

    The pressure gives a linear deflection of about three times the thickness.
    """
    cls, probecls, datacls = SHELLS[name]
    a = b = 1.
    h = 0.01
    E = 70.e9
    nu = 0.3
    D = E*h**3/(12*(1 - nu**2))
    q = 3*h*D/(0.00406*a**4)
    prop = isotropic_plate(thickness=h, E=E, nu=nu)

    xtmp = np.linspace(0, a, nx)
    ytmp = np.linspace(0, b, ny)
    xmesh, ymesh = np.meshgrid(xtmp, ytmp)
    ncoords = np.vstack((xmesh.T.flatten(), ymesh.T.flatten(),
                         np.zeros(nx*ny))).T
    x = ncoords.flatten()
    nids = np.arange(nx*ny).reshape(nx, ny)
    n1s = nids[:-1, :-1].flatten()
    n2s = nids[1:, :-1].flatten()
    n3s = nids[1:, 1:].flatten()
    n4s = nids[:-1, 1:].flatten()
    if name == 'Tria3R':
        connectivity = ([(n1, n2, n3) for n1, n2, n3 in zip(n1s, n2s, n3s)]
                        + [(n1, n3, n4) for n1, n3, n4 in zip(n1s, n3s, n4s)])
    else:
        connectivity = list(zip(n1s, n2s, n3s, n4s))

    data = datacls()
    probe = probecls()
    elements = []
    for k, nodes in enumerate(connectivity):
        elem = cls(probe)
        for i, n in enumerate(nodes):
            setattr(elem, 'n%d' % (i + 1), n)
            setattr(elem, 'c%d' % (i + 1), DOF*n)
        elem.init_k_KC0 = k*data.KC0_SPARSE_SIZE
        elem.init_k_KCNL = k*data.KCNL_SPARSE_SIZE
        elem.init_k_KG = k*data.KG_SPARSE_SIZE
        elem.update_rotation_matrix(x)
        elem.update_probe_xe(x)
        elements.append(elem)

    N = DOF*nx*ny
    X = ncoords[:, 0]
    Y = ncoords[:, 1]
    edge = (np.isclose(X, 0) | np.isclose(X, a) | np.isclose(Y, 0)
            | np.isclose(Y, b))
    bk = np.zeros(N, dtype=bool)
    bk[0::DOF] = edge
    bk[1::DOF] = edge
    bk[2::DOF] = edge
    # NOTE drilling rotations do not take part in the response of a flat plate
    bk[5::DOF] = True
    bu = ~bk

    # uniform pressure lumped with the tributary areas of the nodes
    wx = np.full(nx, a/(nx - 1))
    wx[[0, -1]] /= 2
    wy = np.full(ny, b/(ny - 1))
    wy[[0, -1]] /= 2
    fext = np.zeros(N)
    fext[2::DOF] = q*np.outer(wx, wy).flatten()

    center = nids[nx//2, ny//2]
    return elements, data, prop, x, N, bu, fext, lambda u: u[DOF*center + 2]/h


def beam(name):
    """Beam clamped at both ends with a central transverse load

    The beam axis and the load are not aligned with the global axes. The load
    gives a linear deflection of about three times the height of the cross
    section.
    """
    cls, probecls, datacls, num_elements = BEAMS[name]
    L = 1.
    b = h = 0.02
    E = 70.e9
    prop = BeamProp()
    prop.A = b*h
    prop.E = E
    prop.G = 5/6*E/2/1.3
    prop.Izz = b*h**3/12
    prop.Iyy = b**3*h/12
    prop.J = prop.Izz + prop.Iyy
    P = 3*h*192*E*prop.Izz/L**3

    axis = np.array([1., 2., 3.])/np.sqrt(14.)
    vxy = np.array([0., 0., 1.])
    load = np.cross(axis, vxy + [0.3, 0., 0.])
    load /= np.linalg.norm(load)
    s = np.linspace(0, L, num_elements + 1)
    x = (s[:, None]*axis[None, :]).flatten()
    N = DOF*(num_elements + 1)

    data = datacls()
    probe = probecls()
    elements = []
    for k in range(num_elements):
        elem = cls(probe)
        elem.n1, elem.n2 = k, k + 1
        elem.c1, elem.c2 = DOF*k, DOF*(k + 1)
        elem.init_k_KC0 = k*data.KC0_SPARSE_SIZE
        elem.init_k_KCNL = k*data.KCNL_SPARSE_SIZE
        elem.init_k_KG = k*data.KG_SPARSE_SIZE
        elem.update_rotation_matrix(vxy[0], vxy[1], vxy[2], x)
        elem.update_probe_xe(x)
        elements.append(elem)

    bk = np.zeros(N, dtype=bool)
    bk[:DOF] = True
    bk[-DOF:] = True
    bu = ~bk
    center = num_elements//2
    fext = np.zeros(N)
    fext[DOF*center:DOF*center + 3] = P*load
    return elements, data, prop, x, N, bu, fext, lambda u: (u[DOF*center:DOF*center + 3] @ load)/h


def check_quadratic_convergence(residuals):
    assert residuals[-1] < 1.e-10, 'not converged: %s' % residuals
    assert len(residuals) <= 12, 'too many iterations: %s' % residuals
    # in the asymptotic range the number of correct digits doubles at each
    # iteration, as opposed to growing by a constant amount, until the
    # residual reaches the round-off floor
    asymptotic = [(r0, r1) for r0, r1 in zip(residuals[:-1], residuals[1:])
                  if r0 < 1.e-2]
    assert asymptotic, residuals
    for r0, r1 in asymptotic:
        assert r1 < max(r0**1.6, 1.e-11), (
            'residual %.3e -> %.3e is not quadratic convergence' % (r0, r1))


@pytest.mark.parametrize('name', sorted(SHELLS))
def test_plate_newton_raphson(name):
    *model, deflection = plate(name)
    u_linear, u, residuals = solve(*model)
    check_quadratic_convergence(residuals)
    # the membrane action stiffens the plate
    assert deflection(u) < 0.7*deflection(u_linear)


@pytest.mark.parametrize('name', sorted(BEAMS))
def test_beam_newton_raphson(name):
    *model, deflection = beam(name)
    u_linear, u, residuals = solve(*model)
    check_quadratic_convergence(residuals)
    # the membrane action stiffens the beam
    assert deflection(u) < 0.7*deflection(u_linear)


def test_elements_agree():
    """Different elements must give similar nonlinear deflections"""
    plates = {}
    for name in SHELLS:
        *model, deflection = plate(name)
        plates[name] = deflection(solve(*model)[1])
    values = np.array(list(plates.values()))
    assert values.max()/values.min() < 1.05, plates

    beams = {}
    for name in BEAMS:
        *model, deflection = beam(name)
        beams[name] = deflection(solve(*model)[1])
    values = np.array(list(beams.values()))
    assert values.max()/values.min() < 1.05, beams


if __name__ == '__main__':
    for name in sorted(SHELLS):
        *model, deflection = plate(name)
        u_linear, u, residuals = solve(*model)
        print('%-7s w/h linear %.4f  nonlinear %.4f  residuals %s' % (
            name, deflection(u_linear), deflection(u),
            ' '.join('%.1e' % r for r in residuals)))
    for name in sorted(BEAMS):
        *model, deflection = beam(name)
        u_linear, u, residuals = solve(*model)
        print('%-7s w/h linear %.4f  nonlinear %.4f  residuals %s' % (
            name, deflection(u_linear), deflection(u),
            ' '.join('%.1e' % r for r in residuals)))
