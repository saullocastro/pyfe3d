r"""Static analysis with inertia relief of unsupported structures

The function :func:`pyfe3d.solver.apply_inertia_relief` is checked with the
models of the notebook ``doc/source/tut-inertia-relief.ipynb``, built here.
With the reference (statically determinate) degrees of freedom ``bref``,
whose values are set to zero and which are part of ``bk``, the rigid-body
modes are, Barnett et al. (1995), p. 6, Eq. (19),

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

The first tests check the function itself: the errors of its inputs, the
rigid-body modes, zero accelerations for balanced loads and the invariance
of the strains under the choice of the reference support. The others
check, against closed-form solutions derived in the notebook: three point
masses joined by two springs, with balanced and unbalanced loads, a
fictitious support and inertia relief; a chain of springs and point masses; a
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
import pytest
from numpy.linalg import LinAlgError
from scipy.sparse import coo_matrix, csc_matrix

from pyfe3d.beamprop import BeamProp
from pyfe3d.solver import apply_inertia_relief
from pyfe3d import (Spring, SpringData, SpringProbe, BeamC, BeamCData,
                    BeamCProbe, DOF, INT, DOUBLE)


def run_inertia_relief(KC0, M, fext, bref, bfix=None):
    r"""Results of :func:`pyfe3d.solver.apply_inertia_relief` collected in a
    dict, with the d'Alembert forces

    Returns
    -------
    out : dict
        ``u``, ``acc``, ``Phi`` and ``R`` of
        :func:`pyfe3d.solver.apply_inertia_relief` with
        ``return_reactions=True``; ``finertia``, the d'Alembert forces
        `-\boldsymbol{M} \boldsymbol{\Phi} \ddot{\boldsymbol{\alpha}}`;
        ``ftot = fext + finertia``; and ``bu``, the free dofs.
    """
    u, acc, Phi, R = apply_inertia_relief(KC0, M, fext, bref, bfix,
                                          return_reactions=True)
    finertia = -(M @ (Phi @ acc))
    bk = bref if bfix is None else bref | bfix
    return dict(Phi=Phi, acc=acc, finertia=finertia, ftot=fext + finertia,
                u=u, R=R, bu=~bk)


def _free_beam(nel=6, **kwargs):
    prop = beam_prop(**kwargs)
    xs, KC0, M, beams = beam_model(2., nel, prop)
    return prop, xs, KC0, M, beams


def test_function_input_errors():
    r"""Errors of :func:`pyfe3d.solver.apply_inertia_relief` for invalid inputs

    Shapes, Boolean arrays, an empty ``bref``, ``bref`` overlapping ``bfix``
    and an unsymmetric ``KC0`` raise ``ValueError``; a reference support
    that leaves a rigid-body mode free, five of the six dofs of a node of a
    free beam, each of the six left out, raises a ``LinAlgError`` for a
    ``KC0uu`` that is not positive definite or singular up to the round-off;
    a statically indeterminate one, seven dofs, raises a
    ``LinAlgError`` that says so; and a rigid-body mode without mass, the
    rotation about the axis of a beam without rotary inertia, raises a
    ``LinAlgError``.
    """
    prop, xs, KC0, M, beams = _free_beam()
    N = KC0.shape[0]
    f = np.zeros(N)
    bref = reference_dofs(len(xs), 'clamp')
    with pytest.raises(ValueError, match='M must have the shape'):
        apply_inertia_relief(KC0, M[:-1, :-1], f, bref)
    with pytest.raises(ValueError, match='fext must be'):
        apply_inertia_relief(KC0, M, f[:-1], bref)
    with pytest.raises(ValueError, match='bref must be a Boolean'):
        apply_inertia_relief(KC0, M, f, np.flatnonzero(bref))
    with pytest.raises(ValueError, match='bref has no'):
        apply_inertia_relief(KC0, M, f, np.zeros(N, dtype=bool))
    with pytest.raises(ValueError, match='overlap'):
        apply_inertia_relief(KC0, M, f, bref, bfix=bref.copy())
    Kuns = KC0.tolil()
    Kuns[0, 1] += 1.
    with pytest.raises(ValueError, match='symmetric'):
        apply_inertia_relief(Kuns.tocsc(), M, f, bref)
    for i in range(6):
        too_few = bref.copy()
        too_few[i] = False
        with pytest.raises(LinAlgError, match='must suppress every rigid-body'):
            apply_inertia_relief(KC0, M, f, too_few)
    too_many = bref.copy()
    too_many[DOF*(len(xs) - 1) + 2] = True
    with pytest.raises(LinAlgError, match='statically indeterminate'):
        apply_inertia_relief(KC0, M, f, too_many)
    prop0 = beam_prop(rotary_xz=False)
    prop0.intrhoy2 = 0.
    _, KC00, M0, _ = beam_model(2., 6, prop0)
    with pytest.raises(LinAlgError, match='no mass'):
        apply_inertia_relief(KC00, M0, f, bref)


def test_function_balanced_loads():
    r"""Balanced loads give zero rigid-body accelerations and reactions

    Loads ``f = KC0 @ v``, for any ``v``, are balanced, since
    `\boldsymbol{\Phi}^\top \boldsymbol{K}_{C0} = \boldsymbol{0}`: the
    rigid-body accelerations and the reactions vanish, and the displacements
    are ``v`` minus the rigid-body motion ``Phi @ v[bref]`` that brings the
    reference dofs to zero.
    """
    prop, xs, KC0, M, beams = _free_beam()
    N = KC0.shape[0]
    rng = np.random.default_rng(1)
    v = rng.standard_normal(N)*1e-3
    f = KC0 @ v
    for choice in ('clamp', 'pin-roller'):
        bref = reference_dofs(len(xs), choice)
        u, acc, Phi, R = apply_inertia_relief(KC0, M, f, bref,
                                              return_reactions=True)
        assert np.allclose(Phi[bref], np.eye(6))
        assert abs(KC0 @ Phi).max() < 1e-8*abs(KC0).max()
        assert abs(acc).max() < 1e-10*abs(f).max()
        assert abs(R).max() < 1e-8*abs(f).max()
        assert np.allclose(u, v - Phi @ v[bref], rtol=0,
                           atol=1e-10*abs(v).max())
        # the default return
        out = apply_inertia_relief(KC0, M, f, bref)
        assert len(out) == 3 and np.allclose(out[0], u)


def test_function_strains_independent_of_reference():
    r"""The internal forces do not depend on the reference support

    A free beam under an unbalanced load, an end force and an end moment in
    both planes and an axial force: with three statically determinate
    reference supports, the clamp at either end and the pin-roller, the
    rigid-body accelerations of the centre of mass and the element
    internal forces are the same, and the displacements differ by a
    rigid-body motion.
    """
    prop, xs, KC0, M, beams = _free_beam()
    N = KC0.shape[0]
    n = len(xs)
    f = np.zeros(N)
    f[DOF*(n - 1):DOF*n] = [30., -20., 100., 5., -15., 8.]
    choices = {'clamp': reference_dofs(n, 'clamp'),
               'pin-roller': reference_dofs(n, 'pin-roller')}
    end = np.zeros(N, dtype=bool)
    end[DOF*(n - 1):] = True
    choices['clamp at x = L'] = end
    Phig = geometric_rigid_modes(xs)
    results = {}
    for name, bref in choices.items():
        u, acc, Phi, R = apply_inertia_relief(KC0, M, f, bref,
                                              return_reactions=True)
        assert abs(R[bref]).max() < 1e-10*abs(f).max()
        # accelerations of the 6 geometric rigid-body coordinates
        C = np.linalg.lstsq(Phig, Phi, rcond=None)[0]
        finte = []
        for e in beams:
            e.update_probe_ue(u)
            e.update_probe_finte(prop)
            finte.append(np.array(e.probe.finte))
        results[name] = (u, C @ acc, np.array(finte))
    u0, a0, fi0 = results['clamp']
    for name, (u, a, fi) in results.items():
        assert np.allclose(a, a0, rtol=1e-10)
        assert np.allclose(fi, fi0, rtol=0, atol=1e-9*abs(fi0).max())
        du = u - u0
        c = np.linalg.lstsq(Phig, du, rcond=None)[0]
        assert abs(Phig @ c - du).max() < 1e-9*max(abs(u0).max(), 1e-30)


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


def test_three_node_springs():
    r"""Three point masses joined by two springs, `k_1` between the nodes 1
    and 2 and `k_2` between the nodes 2 and 3, free to move only along `x`

    The stiffness matrix has rank 2 and the null space `\{1, 1, 1\}`. For
    balanced loads, `F_1 + F_2 + F_3 = 0`, the static solution is `u = \{0,
    -F_1/k_1, -F_1/k_1 + F_3/k_2\} + \alpha \{1, 1, 1\}`. The fictitious
    support `\bar{u}_1 = 0` alone, without inertia forces, gives `u_2 = (F_2
    + F_3)/k_1`, `u_3 = u_2 + F_3/k_2` and the reaction `R_1 = -(F_1 + F_2 +
    F_3)`, zero for balanced loads. With inertia relief and
    the reference support `\bar{u}_1 = 0`, `\boldsymbol{K}_{{C0}_{uu}} = [[k_1 +
    k_2, -k_2], [-k_2, k_2]]`, `\boldsymbol{K}_{{C0}_{uk}} = \{-k_1, 0\}`,
    `\boldsymbol{\Phi}_u = \{1, 1\}`, `\ddot{\alpha} = (F_1 + F_2 + F_3)/(m_1
    + m_2 + m_3)`, `u_2 = (F_2 + F_3 - (m_2 + m_3) \ddot{\alpha})/k_1` and
    `u_3 = u_2 + (F_3 - m_3 \ddot{\alpha})/k_2`, derived and checked with
    SymPy in the notebook. With the reference support at node 3, the
    displacements change by `-u_3` and the spring forces do not.
    """
    k1, k2 = 2000., 500.
    ms = np.array([1., 2., 3.])
    KC0, bfix, springs = spring_chain([k1, k2])
    M = point_masses(ms)
    N = KC0.shape[0]
    Kx = KC0.toarray()[0::DOF, 0::DOF]
    assert np.linalg.matrix_rank(Kx) == 2
    assert np.allclose(Kx.sum(axis=1), 0.)
    Kuu = Kx[1:, 1:]
    assert np.allclose(Kuu, [[k1 + k2, -k2], [-k2, k2]])
    assert np.isclose(np.linalg.det(Kuu), k1*k2)
    assert np.allclose(Kx[1:, 0], [-k1, 0.])
    # fictitious support u1 = 0 alone, without inertia forces
    bref1 = np.zeros(N, dtype=bool)
    bref1[0] = True
    bu1 = ~(bref1 | bfix)
    for Fs in ([-5., 1., 4.], [0., 0., 12.], [3., -4., 10.]):
        Fs = np.array(Fs)
        fext = np.zeros(N)
        fext[0::DOF] = Fs
        u = np.zeros(N)
        u[bu1] = np.linalg.solve(KC0[bu1, :][:, bu1].toarray(), fext[bu1])
        u2 = (Fs[1] + Fs[2])/k1
        assert np.allclose(u[0::DOF], [0., u2, u2 + Fs[2]/k2], rtol=1e-12)
        R1 = (KC0 @ u - fext)[0]
        assert np.isclose(R1, -Fs.sum(), rtol=1e-12, atol=1e-13*abs(Fs).max())
        if Fs.sum() == 0:
            # the balanced static solution, with alpha = 0, and R1 = 0
            ubal = np.array([0., -Fs[0]/k1, -Fs[0]/k1 + Fs[2]/k2])
            assert np.allclose(Kx @ (ubal + 0.7), Fs, atol=1e-12)
            assert np.allclose(u[0::DOF], ubal, rtol=1e-12)
            assert abs(R1) < 1e-13*abs(Fs).max()
            # inertia relief: no acceleration, the same solution
            out = run_inertia_relief(KC0, M, fext, bref1, bfix)
            assert abs(out['acc'][0]) < 1e-14*abs(Fs).max()
            assert np.allclose(out['u'], u, rtol=1e-12, atol=1e-16)
    for Fs in ([0., 0., 12.], [3., -4., 10.]):
        Fs = np.array(Fs)
        fext = np.zeros(N)
        fext[0::DOF] = Fs
        acc = Fs.sum()/ms.sum()
        u2 = (Fs[1] + Fs[2] - (ms[1] + ms[2])*acc)/k1
        u3 = u2 + (Fs[2] - ms[2]*acc)/k2
        for r in (0, 2):
            bref = np.zeros(N, dtype=bool)
            bref[DOF*r] = True
            out = run_inertia_relief(KC0, M, fext, bref, bfix)
            assert np.allclose(out['Phi'][0::DOF, 0], 1., rtol=1e-14)
            assert np.isclose(out['acc'][0], acc, rtol=1e-14)
            assert abs(out['R'][DOF*r]) < 1e-13*abs(Fs).max()
            uref = np.array([0., u2, u3]) - (u3 if r == 2 else 0.)
            assert np.allclose(out['u'][0::DOF], uref, rtol=1e-12, atol=1e-16)
            assert np.allclose(spring_forces(springs, out['u']),
                               [k1*u2, k2*(u3 - u2)], rtol=1e-12)
        if Fs[0] == Fs[1] == 0:
            # 1 mm and 13 mm; the springs accelerate the masses on their left
            assert np.isclose(u2, 1e-3) and np.isclose(u3, 13e-3)
            assert np.isclose(k1*u2, ms[0]*acc)
            assert np.isclose(k2*(u3 - u2), (ms[0] + ms[1])*acc)


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
        out = run_inertia_relief(KC0, M, fext, bref, bfix)
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
        out = run_inertia_relief(KC0, M, fext, bref)
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
                out = run_inertia_relief(KC0, M, fext, bref)
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
    out = run_inertia_relief(KC0, M, fext, reference_dofs(len(xs), 'clamp'))
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
    test_three_node_springs()
    test_spring_chain()
    test_beam_balanced_loads()
    test_free_free_beam_end_force()
    test_free_free_bar_end_force()
