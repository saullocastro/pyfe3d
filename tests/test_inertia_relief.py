r"""Static analysis with inertia relief of unsupported structures

pyfe3d has no inertia-relief function; the procedure is built here from the
public API, the assembled ``KC0`` and ``M``, as in the notebook
``doc/source/tut-inertia-relief.ipynb``. With the reference (statically
determinate) degrees of freedom ``bref``, whose values are set to zero and
which are part of ``bk``, the rigid-body modes are, Barnett et al. (1995),
p. 6, Eq. (19),

.. math::

    \boldsymbol{\Phi} = \begin{bmatrix} \boldsymbol{\Phi}_u \\ \boldsymbol{\Phi}_k \end{bmatrix}
    = \begin{bmatrix} -\boldsymbol{K}_{{C0}_{uu}}^{-1} \boldsymbol{K}_{{C0}_{uk}} \\ \boldsymbol{I}
    \end{bmatrix}

the rigid-body accelerations are, Eqs. (21) and (22),
`\boldsymbol{\Phi}^\top \boldsymbol{M} \boldsymbol{\Phi} \ddot{\boldsymbol{\alpha}} =
\boldsymbol{\Phi}^\top \boldsymbol{F}_{ext}`, and the displacements relative to the
reference support are, p. 7, Eqs. (23) and (24),
`\boldsymbol{K}_{{C0}_{uu}} \boldsymbol{u}_u = \boldsymbol{F}_{{ext}_u} - \boldsymbol{M}_u
\boldsymbol{\Phi} \ddot{\boldsymbol{\alpha}}`.

Barnett, A. R., Widrick, T. W., Ludwiczak, D. R., 1995, Closed-Form Static
Analysis with Inertia Relief and Displacement-Dependent Loads Using a
MSC/NASTRAN DMAP Alter, NASA TM-106836, https://ntrs.nasa.gov/citations/19950013233

The tests check, against closed-form solutions derived in the notebook:
a spring between two point masses; a chain of springs and point masses; a
free-free BeamC beam under a balanced load set, its own weight and axial
loads, for which the rigid-body accelerations and the reactions of the
reference support vanish; and a free-free beam and bar under an end force,
for which the deflection relative to the reference support is the
Euler-Bernoulli solution of `E I w'''' = q`, with the inertia load `q(x)`
linear in `x`.
"""
import sys
sys.path.append('..')

import numpy as np
from scipy.sparse import coo_matrix, csc_matrix
from scipy.sparse.linalg import spsolve

from pyfe3d.beamprop import BeamProp
from pyfe3d import (Spring, SpringData, SpringProbe, BeamC, BeamCData,
                    BeamCProbe, DOF, INT, DOUBLE)


def inertia_relief(KC0, M, fext, bref, bfix=None):
    r"""Static analysis with inertia relief

    Parameters
    ----------
    KC0, M : sparse matrices
        Stiffness and mass matrices of the unsupported model, shape (N, N).
    fext : array
        External forces, shape (N,).
    bref : boolean array
        Reference degrees of freedom, a statically determinate support,
        set to zero.
    bfix : boolean array, optional
        Degrees of freedom that are constrained in the model itself, e.g.
        the out-of-plane ones of a model in one dimension, also set to zero.

    Returns
    -------
    out : dict
        ``Phi`` (N, nr), ``acc`` the rigid-body accelerations
        `\ddot{\boldsymbol{\alpha}}` (nr,), ``finertia`` the d'Alembert forces
        `-\boldsymbol{M} \boldsymbol{\Phi} \ddot{\boldsymbol{\alpha}}`, ``ftot``,
        ``u`` the displacements relative to the reference support, ``R``
        the residual `\boldsymbol{K}_{C0} \boldsymbol{u} - \boldsymbol{F}_{tot}`,
        whose values at ``bref`` are the reactions of the reference
        support, and ``bu``.
    """
    N = KC0.shape[0]
    bfix = np.zeros(N, dtype=bool) if bfix is None else bfix
    bk = bref | bfix
    bu = ~bk
    KC0 = csc_matrix(KC0)
    Kuu = KC0[bu, :][:, bu]
    Kuref = KC0[bu, :][:, bref].toarray()
    nr = bref.sum()
    Phi = np.zeros((N, nr))
    Phi[bref] = np.eye(nr)
    Phiu = spsolve(Kuu, Kuref)
    Phi[bu] = -Phiu.reshape(bu.sum(), nr)
    Mrr = Phi.T @ (M @ Phi)
    acc = np.linalg.solve(Mrr, Phi.T @ fext)
    finertia = -(M @ (Phi @ acc))
    ftot = fext + finertia
    u = np.zeros(N)
    u[bu] = spsolve(Kuu, ftot[bu])
    R = KC0 @ u - ftot
    return dict(Phi=Phi, acc=acc, finertia=finertia, ftot=ftot, u=u, R=R,
                bu=bu)


def spring_chain(ks, x=None):
    r"""Springs along `x` between consecutive nodes, stiffness ``ks``"""
    n = len(ks) + 1
    N = DOF*n
    data = SpringData()
    probe = SpringProbe()
    KC0r = np.zeros(data.KC0_SPARSE_SIZE*len(ks), dtype=INT)
    KC0c = np.zeros(data.KC0_SPARSE_SIZE*len(ks), dtype=INT)
    KC0v = np.zeros(data.KC0_SPARSE_SIZE*len(ks), dtype=DOUBLE)
    springs = []
    for i, k in enumerate(ks):
        s = Spring(probe)
        s.init_k_KC0 = i*data.KC0_SPARSE_SIZE
        s.n1, s.n2 = i, i + 1
        s.c1, s.c2 = DOF*i, DOF*(i + 1)
        s.kxe = k
        s.update_rotation_matrix(1., 0., 0., 1., 1., 0.)
        s.update_KC0(KC0r, KC0c, KC0v)
        springs.append(s)
    KC0 = coo_matrix((KC0v, (KC0r, KC0c)), shape=(N, N)).tocsc()
    # NOTE only the displacement along x is free, the other dofs are
    #      constrained in the model itself, as by the rollers
    bfix = np.ones(N, dtype=bool)
    bfix[0::DOF] = False
    return KC0, bfix, springs


def point_masses(ms):
    N = DOF*len(ms)
    Md = np.zeros(N)
    Md[0::DOF] = ms
    return coo_matrix((Md, (np.arange(N), np.arange(N))), shape=(N, N)).tocsc()


def spring_forces(springs, u):
    r"""Axial force of each spring, positive in tension"""
    out = []
    for s in springs:
        s.update_probe_ue(u)
        s.update_probe_finte()
        out.append(s.probe.finte[6])
    return np.array(out)


def beam_prop(E=210e9, rho=7850., b=0.02, h=0.05, G=None,
              rotary_xz=True):
    r"""Rectangular section, width ``b`` along `y`, height ``h`` along `z`

    ``rotary_xz=False`` drops the rotary inertia of the bending in the
    plane `xz`, ``intrhoz2``, as in the Euler-Bernoulli closed form.
    """
    p = BeamProp()
    p.A = b*h
    p.E = E
    p.G = E/2.6 if G is None else G
    p.Iyy = b*h**3/12
    p.Izz = h*b**3/12
    p.J = p.Iyy + p.Izz
    p.intrho = rho*p.A
    p.intrhoy2 = rho*p.Izz
    p.intrhoz2 = rho*p.Iyy if rotary_xz else 0.
    return p


def beam_model(L, nel, prop, mtype=0):
    r"""BeamC elements along the global `x` axis, element axes = global"""
    n = nel + 1
    N = DOF*n
    xs = np.linspace(0, L, n)
    ncoords = np.column_stack((xs, np.zeros(n), np.zeros(n))).flatten()
    data = BeamCData()
    probe = BeamCProbe()
    size_M = data.M_SPARSE_SIZE if mtype == 0 else 36
    KC0r = np.zeros(data.KC0_SPARSE_SIZE*nel, dtype=INT)
    KC0c = np.zeros(data.KC0_SPARSE_SIZE*nel, dtype=INT)
    KC0v = np.zeros(data.KC0_SPARSE_SIZE*nel, dtype=DOUBLE)
    Mr = np.zeros(size_M*nel, dtype=INT)
    Mc = np.zeros(size_M*nel, dtype=INT)
    Mv = np.zeros(size_M*nel, dtype=DOUBLE)
    beams = []
    for i in range(nel):
        e = BeamC(probe)
        e.init_k_KC0 = i*data.KC0_SPARSE_SIZE
        e.init_k_M = i*size_M
        e.n1, e.n2 = i, i + 1
        e.c1, e.c2 = DOF*i, DOF*(i + 1)
        e.update_rotation_matrix(0., 1., 0., ncoords)
        e.update_probe_xe(ncoords)
        e.update_KC0(KC0r, KC0c, KC0v, prop)
        e.update_M(Mr, Mc, Mv, prop, mtype=mtype)
        beams.append(e)
    KC0 = coo_matrix((KC0v, (KC0r, KC0c)), shape=(N, N)).tocsc()
    M = coo_matrix((Mv, (Mr, Mc)), shape=(N, N)).tocsc()
    return xs, KC0, M, beams


def geometric_rigid_modes(xs):
    r"""Translations and rotations about the origin of nodes on the x axis"""
    n = len(xs)
    Phi = np.zeros((DOF*n, 6))
    for i, x in enumerate(xs):
        Phi[DOF*i:DOF*i + 3, :3] = np.eye(3)
        Phi[DOF*i + 3:DOF*i + 6, 3:] = np.eye(3)
        # rotation r_y moves the point (x, 0, 0) by w = -x r_y
        # rotation r_z moves it by v = x r_z
        Phi[DOF*i + 2, 4] = -x
        Phi[DOF*i + 1, 5] = x
    return Phi


def beam_end_forces(beams, prop, u, M, acc_field):
    r"""Section forces at the ends of each element, in element coordinates

    ``finte`` minus the element's own share of the distributed load,
    ``M_e @ acc_field``, here the inertia (and gravity) load of the
    element; ``acc_field`` is the global field whose product with ``M``
    gives the distributed load. Returns an array (nel, 12).
    """
    out = []
    for e in beams:
        e.update_probe_ue(u)
        e.update_probe_finte(prop)
        fe = np.array(e.probe.finte)
        c1, c2 = e.c1, e.c2
        idx = np.r_[c1:c1 + DOF, c2:c2 + DOF]
        Me = element_M(e, prop)
        fq = Me @ acc_field[idx]
        out.append(fe - fq)
    return np.array(out)


def element_M(e, prop, mtype=0):
    r"""Mass matrix of one element, global = element coordinates here"""
    data = BeamCData()
    Mr = np.zeros(data.M_SPARSE_SIZE, dtype=INT)
    Mc = np.zeros(data.M_SPARSE_SIZE, dtype=INT)
    Mv = np.zeros(data.M_SPARSE_SIZE, dtype=DOUBLE)
    c1, c2, k = e.c1, e.c2, e.init_k_M
    e.c1, e.c2, e.init_k_M = 0, DOF, 0
    e.update_M(Mr, Mc, Mv, prop, mtype=mtype)
    e.c1, e.c2, e.init_k_M = c1, c2, k
    return coo_matrix((Mv, (Mr, Mc)), shape=(2*DOF, 2*DOF)).toarray()


def test_two_node_spring():
    r"""Two point masses joined by a spring, free to move only along `x`

    Rigid-body acceleration `\ddot{\alpha} = (F_1 + F_2)/(m_1 + m_2)` and,
    with the reference support `\bar{u}_1 = 0`, `u_2 = (m_1 F_2 - m_2
    F_1)/(k (m_1 + m_2))`, derived in the notebook from the equations of
    motion `m_1 \ddot{u}_1 + k (u_1 - u_2) = F_1` and `m_2 \ddot{u}_2 + k
    (u_2 - u_1) = F_2` with `\ddot{u}_1 = \ddot{u}_2 = \ddot{\alpha}`.
    """
    k, F, m1, m2 = 1000., 10., 2., 3.
    KC0, bfix, springs = spring_chain([k])
    M = point_masses([m1, m2])
    bref = np.zeros(KC0.shape[0], dtype=bool)
    bref[0] = True
    for F1, F2 in [(-F, F), (0., F), (4., F)]:
        fext = np.zeros(KC0.shape[0])
        fext[0], fext[DOF] = F1, F2
        out = inertia_relief(KC0, M, fext, bref, bfix)
        assert np.allclose(out['Phi'][[0, DOF], 0], [1., 1.])
        acc = (F1 + F2)/(m1 + m2)
        assert np.isclose(out['acc'][0], acc, rtol=1e-14)
        u2 = (m1*F2 - m2*F1)/(k*(m1 + m2))
        assert np.isclose(out['u'][DOF], u2, rtol=1e-14)
        assert abs(out['Phi'].T @ out['ftot']).max() < 1e-13*F
        # reaction of the reference support
        assert abs(out['R'][0]) < 1e-13*F
        assert np.isclose(spring_forces(springs, out['u'])[0], k*u2,
                          rtol=1e-14)


def chain_data():
    ks = np.array([1000., 2000., 1500., 3000.])
    ms = np.array([1., 2., 1.5, 0.5, 3.])
    F = np.array([0., 5., -2., 0., 12.])
    return ks, ms, F


def test_spring_chain():
    r"""Chain of point masses and springs under unbalanced loads

    `\ddot{\alpha} = \sum F_i / \sum m_i`; the force of the spring between
    the nodes `j` and `j + 1`, positive in tension, is that of the masses and
    loads to its right, `N_j = \sum_{i > j} (F_i - m_i \ddot{\alpha})`; the
    displacements relative to the reference node `r` follow from
    `u_{j+1} - u_j = N_j/k_j` and `u_r = 0`. The spring forces do not depend
    on the reference node, the displacements differ by a rigid translation.
    """
    ks, ms, F = chain_data()
    KC0, bfix, springs = spring_chain(ks)
    M = point_masses(ms)
    N = KC0.shape[0]
    fext = np.zeros(N)
    fext[0::DOF] = F
    acc = F.sum()/ms.sum()
    Nj = np.array([(F[j+1:] - ms[j+1:]*acc).sum() for j in range(len(ks))])
    us = []
    for r in (0, 3):
        bref = np.zeros(N, dtype=bool)
        bref[DOF*r] = True
        out = inertia_relief(KC0, M, fext, bref, bfix)
        assert np.isclose(out['acc'][0], acc, rtol=1e-14)
        assert abs(out['R'][DOF*r]) < 1e-13*abs(F).max()
        assert np.allclose(spring_forces(springs, out['u']), Nj, rtol=1e-12)
        u = np.concatenate(([0.], np.cumsum(Nj/ks)))
        u -= u[r]
        assert np.allclose(out['u'][0::DOF], u, rtol=1e-12, atol=1e-16)
        us.append(out['u'][0::DOF])
    diff = us[1] - us[0]
    assert np.allclose(diff, diff[0], rtol=1e-12)


def reference_dofs(nnodes, choice):
    r"""``choice='clamp'``: the six dofs of the first node;
    ``choice='pin-roller'``: `u, v, w, r_x` of the first node and `v, w` of
    the last node"""
    N = DOF*nnodes
    bref = np.zeros(N, dtype=bool)
    if choice == 'clamp':
        bref[:DOF] = True
    else:
        bref[[0, 1, 2, 3]] = True
        bref[DOF*(nnodes - 1) + 1] = True
        bref[DOF*(nnodes - 1) + 2] = True
    return bref


def balanced_beam(L=2., nel=10, g=9.81, Paxial=1000., a=0.4):
    r"""Own weight from ``M`` times the uniform acceleration `-g` along `z`,
    upward point loads `W/2` at `x = a` and `x = L - a`, axial end forces
    `-P` at `x = 0` and `+P` at `x = L`"""
    prop = beam_prop()
    xs, KC0, M, beams = beam_model(L, nel, prop)
    N = KC0.shape[0]
    agrav = np.zeros(N)
    agrav[2::DOF] = -g
    fgrav = M @ agrav
    W = -fgrav[2::DOF].sum()
    fext = fgrav.copy()
    for xa in (a, L - a):
        fext[DOF*np.argmin(abs(xs - xa)) + 2] += W/2
    fext[0] -= Paxial
    fext[DOF*nel] += Paxial
    return prop, xs, KC0, M, beams, agrav, fgrav, W, fext


def test_beam_balanced_loads():
    r"""Free-free BeamC beam under its own weight, two upward point loads
    `W/2` at `x = a` and `x = L - a` and opposite axial end forces

    The load set is balanced in force and moment: the rigid-body
    accelerations and the reactions of the reference support vanish to
    round-off, for two statically determinate supports, and the bending
    moment equals the statics of the beam, `M(x) = -m g x^2/2 + (W/2)
    \langle x - a \rangle + (W/2) \langle x - L + a \rangle`, independently
    of the support.
    """
    L, g, Paxial, a = 2., 9.81, 1000., 0.4
    prop, xs, KC0, M, beams, agrav, fgrav, W, fext = balanced_beam(
        L=L, g=g, Paxial=Paxial, a=a)
    m = prop.intrho
    assert np.isclose(W, m*L*g, rtol=1e-13)
    Phig = geometric_rigid_modes(xs)
    assert abs(Phig.T @ fext).max() < 1e-12*W
    Mcf = (-m*g*xs**2/2 + W/2*np.clip(xs - a, 0, None)
           + W/2*np.clip(xs - (L - a), 0, None))
    Ms = []
    for choice in ('clamp', 'pin-roller'):
        bref = reference_dofs(len(xs), choice)
        out = inertia_relief(KC0, M, fext, bref)
        assert abs(KC0 @ out['Phi']).max() < 1e-6*abs(KC0).max()
        assert abs(out['acc'][:3]).max() < 1e-12*g
        assert abs(out['R'][bref]).max() < 1e-10*W
        S = beam_end_forces(beams, prop, out['u'], M, agrav)
        Mfe = np.append(S[:, 4], -S[-1, 10])
        assert np.allclose(Mfe, Mcf, atol=1e-10*abs(Mcf).max())
        # axial force
        assert np.allclose(S[:, 6], Paxial, rtol=1e-12)
        Ms.append(Mfe)
    assert np.allclose(Ms[0], Ms[1], atol=1e-12*abs(Ms[0]).max())


def free_free_closed_form(xs, L, P, EI, GA, choice):
    r"""Deflection relative to the reference support of a free-free beam
    under the end force `P` at `x = L`, derived in the notebook:
    `M(x) = P x^2 (L - x)/L^2` and `w = P (L x^4/12 - x^5/20)/(E I L^2) -
    M/(G A)`, for the clamp at `x = 0`; the pin-roller subtracts the chord
    through `w(0) = w(L) = 0`."""
    Mx = P*xs**2*(L - xs)/L**2
    w = P*(L*xs**4/12 - xs**5/20)/(EI*L**2) - Mx/GA
    if choice == 'pin-roller':
        w = w - xs/L*(P*L**3/(30*EI))
    return w, Mx


def test_free_free_beam_end_force():
    r"""Free-free BeamC beam under a transverse end force `P`

    Newton-Euler: `\ddot{w}_0 = -2 P/(m L)` and `\ddot{r}_{y0} = -6 P/(m
    L^2)` at the first node; the deflection relative to the reference
    support is that of `E I w'''' = q(x)`, `q = (P/L)(2 - 6 x/L)`, plus the
    shear deflection `-M/(G A)`, see :func:`free_free_closed_form`. Without
    the rotary inertia of the plane `xz`, as in the closed form, the
    consistent mass matrix gives the nodal deflections and the bending
    moments to round-off; the lumped mass matrix converges as `1/n^2`,
    with a relative error of the deflection, measured with 10, 20 and 40
    elements, of 0.75/n^2 to 0.80/n^2 for the clamp and 1.10/n^2 for the
    pin-roller, asserted below 1.5/n^2.
    """
    L, P = 2., 100.
    prop = beam_prop(rotary_xz=False)
    prop.G = 5/6*prop.E/2.6
    m, EI, GA = prop.intrho, prop.E*prop.Iyy, prop.G*prop.A
    for mtype, nels in ((0, (10,)), (1, (10, 20, 40))):
        for nel in nels:
            xs, KC0, M, beams = beam_model(L, nel, prop, mtype=mtype)
            fext = np.zeros(KC0.shape[0])
            fext[DOF*nel + 2] = P
            for choice in ('clamp', 'pin-roller'):
                bref = reference_dofs(len(xs), choice)
                out = inertia_relief(KC0, M, fext, bref)
                wcf, Mx = free_free_closed_form(xs, L, P, EI, GA, choice)
                err = abs(out['u'][2::DOF] - wcf).max()/abs(wcf).max()
                if mtype == 0:
                    assert err < 1e-9
                    if choice == 'clamp':
                        assert np.isclose(out['acc'][2], -2*P/(m*L))
                        assert np.isclose(out['acc'][4], -6*P/(m*L**2))
                    S = beam_end_forces(beams, prop, out['u'], M,
                                        -(out['Phi'] @ out['acc']))
                    Mfe = np.append(S[:, 4], -S[-1, 10])
                    assert np.allclose(Mfe, Mx, atol=1e-10*abs(Mx).max())
                else:
                    assert err < 1.5/nel**2


def test_free_free_bar_end_force():
    r"""Free-free bar under an axial end force `P` at `x = L`

    `\ddot{u} = P/(m L)`, `N(x) = P x/L` and, relative to `u(0) = 0`, `u =
    P x^2/(2 E A L)`; with the consistent mass matrix, the nodal
    displacements are exact, and so is the constant axial force of each
    element, equal to `N` at its midpoint since `u` is quadratic.
    """
    L, P, nel = 2., 100., 10
    prop = beam_prop()
    xs, KC0, M, beams = beam_model(L, nel, prop)
    fext = np.zeros(KC0.shape[0])
    fext[DOF*nel] = P
    out = inertia_relief(KC0, M, fext, reference_dofs(len(xs), 'clamp'))
    m, EA = prop.intrho, prop.E*prop.A
    assert np.isclose(out['acc'][0], P/(m*L), rtol=1e-13)
    ucf = P*xs**2/(2*EA*L)
    assert np.allclose(out['u'][0::DOF], ucf, atol=1e-12*ucf.max())
    Ne = (out['u'][DOF::DOF] - out['u'][0:-DOF:DOF])*EA/(L/nel)
    xm = (xs[1:] + xs[:-1])/2
    assert np.allclose(Ne, P*xm/L, rtol=1e-11)
    S = beam_end_forces(beams, prop, out['u'], M, -(out['Phi'] @ out['acc']))
    assert np.allclose(S[:, 6], P*xs[1:]/L, rtol=1e-11)


if __name__ == '__main__':
    test_two_node_spring()
    test_spring_chain()
    test_beam_balanced_loads()
    test_free_free_beam_end_force()
    test_free_free_bar_end_force()
