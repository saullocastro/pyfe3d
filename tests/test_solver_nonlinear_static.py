r"""Non-linear static analyses of pyfe3d models with :class:`pyfe3d.solver.Analysis`

The models of ``test_nonlinear_static_newton_raphson.py``, a simply
supported plate with immovable edges under uniform pressure and a beam
clamped at both ends under a central load, are solved with the solvers of
:mod:`pyfe3d.solver` instead of the hand-written Newton-Raphson loop of that
test, through callables of all the degrees-of-freedom and the mask ``bu`` of
the free ones:

- the Newton-Raphson method reproduces the solution of the hand-written
  loop, with quadratic convergence of Crisfield's `\zeta`;
- Crisfield's criterion does not depend on the units of the rotations:
  scaling the rotational degrees-of-freedom changes neither the number of
  iterations nor `\zeta`, whereas it changes the relative criterion; and its
  norms are over the free degrees-of-freedom only;
- a prescribed displacement imposed exactly, with ``uk``, gives the reaction
  that equals the load of the load-controlled analysis with the same
  deflection;
- the piston-theory aerodynamic load of Quad4R, a configuration-dependent
  load with an unsymmetric load stiffness, converges quadratically with the
  unsymmetric tangent stiffness matrix.
"""
import sys
sys.path.append('..')

import numpy as np
import pytest
from scipy.sparse import coo_matrix, csr_matrix

from pyfe3d import DOF, INT, DOUBLE
from pyfe3d.solver import Analysis, crisfield_weights, crisfield_zeta, solve

from test_nonlinear_static_newton_raphson import plate, beam
from test_nonlinear_static_newton_raphson import solve as solve_by_hand


class Model(object):
    r"""Callables of :class:`pyfe3d.solver.Analysis` for a list of pyfe3d
    elements, with the vectors and matrices of all the degrees-of-freedom

    ``calc_kC`` gives ``KC0`` when ``NLgeom=False`` and ``KC0 + KCNL(u)``
    otherwise, ``calc_kG`` gives ``KG(u)`` and ``calc_fint`` the internal
    force vector of ``update_fint(..., nonlinear=1)``, whose exact Jacobian
    is ``KC0 + KCNL + KG``. ``S`` scales the degrees-of-freedom, the
    callables then working with ``c = S u``, the forces ``S^{-1} f`` and the
    matrices ``S^{-1} K S^{-1}``.
    """
    def __init__(self, elements, data, prop, x, N, fext, S=None):
        self.elements = elements
        self.prop = prop
        self.x = x
        self.N = N
        self.fext = fext
        self.S = np.ones(N) if S is None else S
        num_elements = len(elements)
        self.data = data
        self.KC0r = np.zeros(data.KC0_SPARSE_SIZE*num_elements, dtype=INT)
        self.KC0c = np.zeros(data.KC0_SPARSE_SIZE*num_elements, dtype=INT)
        KC0v = np.zeros(data.KC0_SPARSE_SIZE*num_elements, dtype=DOUBLE)
        self.KCNLr = np.zeros(data.KCNL_SPARSE_SIZE*num_elements, dtype=INT)
        self.KCNLc = np.zeros(data.KCNL_SPARSE_SIZE*num_elements, dtype=INT)
        self.KGr = np.zeros(data.KG_SPARSE_SIZE*num_elements, dtype=INT)
        self.KGc = np.zeros(data.KG_SPARSE_SIZE*num_elements, dtype=INT)
        for elem in elements:
            elem.update_probe_xe(x)
            elem.update_KC0(self.KC0r, self.KC0c, KC0v, prop)
        self.KC0 = self.sparse(self.KC0r, self.KC0c, KC0v)
        self.fint_calls = []

    def sparse(self, r, c, v):
        return coo_matrix((v, (r, c)), shape=(self.N, self.N)).tocsr()

    def scaled(self, A):
        Sinv = 1./self.S
        return csr_matrix(A.multiply(Sinv[:, None]).multiply(Sinv[None, :]))

    def internal_forces(self, u):
        fint = np.zeros(self.N)
        for elem in self.elements:
            # NOTE the probe is shared amongst the elements
            elem.update_probe_xe(self.x)
            elem.update_probe_ue(u)
            elem.update_fint(fint, self.prop, nonlinear=1)
        return fint

    def KCNL(self, u):
        v = np.zeros(self.data.KCNL_SPARSE_SIZE*len(self.elements))
        for elem in self.elements:
            elem.update_probe_xe(self.x)
            elem.update_probe_ue(u)
            elem.update_KCNL(self.KCNLr, self.KCNLc, v, self.prop)
        return self.sparse(self.KCNLr, self.KCNLc, v)

    def KG(self, u):
        v = np.zeros(self.data.KG_SPARSE_SIZE*len(self.elements))
        for elem in self.elements:
            elem.update_probe_xe(self.x)
            elem.update_probe_ue(u)
            elem.update_KG(self.KGr, self.KGc, v, self.prop)
        return self.sparse(self.KGr, self.KGc, v)

    # NOTE callables of pyfe3d.solver.Analysis
    def calc_fext(self, inc=1., silent=True):
        return inc*self.fext/self.S

    def calc_fint(self, c, silent=True):
        self.fint_calls.append(c.copy())
        return self.internal_forces(c/self.S)/self.S

    def calc_kC(self, c=None, NLgeom=False, silent=True):
        if c is None or not NLgeom:
            return self.scaled(self.KC0)
        return self.scaled(self.KC0 + self.KCNL(c/self.S))

    def calc_kG(self, c=None, NLgeom=False, silent=True):
        return self.scaled(self.KG(c/self.S))

    def analysis(self, bu, uk=None, **kwargs):
        an = Analysis(self.calc_fext, self.calc_fint, self.calc_kC,
                      self.calc_kG, bu=bu, uk=uk)
        for key, value in kwargs.items():
            setattr(an, key, value)
        return an

    def zetas(self, bu):
        r"""Crisfield's `\zeta` of every call of ``calc_fint``, at the load
        factor 1, over the free degrees-of-freedom"""
        w = crisfield_weights(self.scaled(self.KC0)[bu, :][:, bu])
        out = []
        for c in self.fint_calls:
            fint = (self.internal_forces(c/self.S)/self.S)[bu]
            fext = (self.fext/self.S)[bu]
            out.append(crisfield_zeta(fext - fint, fext, fint, w))
        return out


#: Factors of the units of the rotations, the last one changes the
#: iterations of the relative criterion
SCALES = (1., 1.e3, 1.e-6)


def rotation_scaling(N, s):
    r"""``S`` that multiplies the rotational degrees-of-freedom by ``s``"""
    S = np.ones(N)
    for i in range(3, DOF):
        S[i::DOF] = s
    return S


@pytest.mark.parametrize('kind, name', [('plate', 'Quad4R'),
                                        ('plate', 'Tria3DSG'),
                                        ('beam', 'BeamC')])
def test_newton_raphson_reproduces_the_hand_written_loop(kind, name):
    r"""A single increment at the full load, as the hand-written loop, gives
    the same solution, with the quadratic convergence of `\zeta`"""
    elements, data, prop, x, N, bu, fext, deflection = (
        plate(name) if kind == 'plate' else beam(name))
    u_linear, u_ref, residuals = solve_by_hand(elements, data, prop, x, N,
                                               bu, fext)
    model = Model(elements, data, prop, x, N, fext)
    an = model.analysis(bu, initialInc=1., relTOL=1.e-9)
    increments, cs = an.static(NLgeom=True, silent=True)
    assert increments == [1.]
    u = cs[-1]
    assert u.shape == (N,)
    assert np.all(u[~bu] == 0)
    print(name, 'w/h', deflection(u), 'hand-written', deflection(u_ref))
    assert np.isclose(deflection(u), deflection(u_ref), rtol=1.e-7)
    np.testing.assert_allclose(u, u_ref, rtol=1.e-6,
                               atol=1.e-8*np.abs(u_ref).max())
    zetas = model.zetas(bu)
    print('zeta', ' '.join('%.1e' % z for z in zetas))
    assert zetas[-1] <= 1.e-9
    assert len(zetas) <= 8
    # NOTE quadratic decay of zeta in the asymptotic range
    for z0, z1 in zip(zetas[:-1], zetas[1:]):
        if z0 < 1.e-2:
            assert z1 < max(10*z0**1.8, 1.e-11), zetas


@pytest.mark.parametrize('kind, name', [('plate', 'Quad4R'),
                                        ('beam', 'BeamC')])
def test_crisfield_criterion_units_invariance(kind, name):
    r"""Measuring the rotations in other units, `\theta' = s \theta`, i.e.
    the moments `M' = M/s` and the stiffnesses `K' = S^{-1} K S^{-1}`, does
    not change Crisfield's `\zeta`, hence neither the number of iterations
    nor the increments, whereas it changes the relative criterion"""
    elements, data, prop, x, N, bu, fext, deflection = (
        plate(name) if kind == 'plate' else beam(name))
    results = {}
    for convergence in ('crisfield', 'relative'):
        for s in SCALES:
            S = rotation_scaling(N, s)
            model = Model(elements, data, prop, x, N, fext, S=S)
            an = model.analysis(bu, initialInc=0.25, relTOL=1.e-8,
                                convergence=convergence)
            increments, cs = an.static(NLgeom=True, silent=True)
            u = cs[-1]/S if cs else None
            results[convergence, s] = (increments, len(model.fint_calls),
                                       model.zetas(bu), deflection(u)
                                       if cs else None)
            print(convergence, s, 'increments', len(increments),
                  'fint calls', len(model.fint_calls), 'w/h',
                  results[convergence, s][3])
    inc0, calls0, zetas0, w0 = results['crisfield', 1.]
    assert inc0[-1] == 1.
    for s in SCALES[1:]:
        inc, calls, zetas, w = results['crisfield', s]
        assert calls == calls0
        np.testing.assert_allclose(inc, inc0, rtol=1.e-12)
        np.testing.assert_allclose(zetas, zetas0, rtol=1.e-5, atol=1.e-13)
        assert np.isclose(w, w0, rtol=1.e-7)
    # NOTE the Euclidean norm of the residual mixes forces and moments: with
    #      the rotations in micro-units the moments dominate it, and the
    #      relative criterion needs another number of iterations, or does not
    #      converge at all
    calls = [results['relative', s][1] for s in SCALES]
    assert results['relative', 1.][0][-1] == 1.
    assert calls[-1] != calls[0], calls


def test_crisfield_criterion_free_dofs_only():
    r"""The residual at the constrained degrees-of-freedom is the support
    reaction: included in the norms it keeps `\zeta` of the order of one,
    over the free degrees-of-freedom it is below the tolerance. The same
    model with the constrained rows and columns zeroed instead of ``bu``
    converges with the same iterations"""
    elements, data, prop, x, N, bu, fext, deflection = plate('Quad4R')
    model = Model(elements, data, prop, x, N, fext)
    an = model.analysis(bu, initialInc=1., relTOL=1.e-9)
    increments, cs = an.static(NLgeom=True, silent=True)
    u = cs[-1]
    fint = model.internal_forces(u)
    R = fext - fint
    w_free = crisfield_weights(model.KC0[bu, :][:, bu])
    zeta_free = crisfield_zeta(R[bu], fext[bu], fint[bu], w_free)
    # NOTE the drilling rotations of the flat plate have a null diagonal, all
    #      the other degrees-of-freedom enter the norms
    d = np.abs(model.KC0.diagonal())
    w_all = np.zeros(N)
    w_all[d > 0] = 1./d[d > 0]
    zeta_all = crisfield_zeta(R, fext, fint, w_all)
    print('zeta free dofs %.2e, all dofs %.2e' % (zeta_free, zeta_all))
    assert zeta_free <= 1.e-9
    assert zeta_all > 0.1

    # NOTE constrained degrees-of-freedom as null rows and columns
    keep = bu.astype(float)
    P = csr_matrix(coo_matrix((keep, (np.arange(N), np.arange(N))),
                              shape=(N, N)))
    calls = []

    def calc_fint(c, silent=True):
        calls.append(c.copy())
        return keep*model.internal_forces(c)

    an_null = Analysis(lambda inc=1., silent=True: inc*keep*fext, calc_fint,
                       lambda c=None, NLgeom=False, silent=True:
                       P @ model.calc_kC(c, NLgeom) @ P,
                       lambda c=None, NLgeom=False, silent=True:
                       P @ model.calc_kG(c, NLgeom) @ P)
    an_null.initialInc = 1.
    an_null.relTOL = 1.e-9
    increments_null, cs_null = an_null.static(NLgeom=True, silent=True)
    assert len(calls) == len(model.fint_calls)
    np.testing.assert_allclose(cs_null[-1], u, rtol=1.e-9,
                               atol=1.e-12*np.abs(u).max())


def point_loaded_plate():
    r"""The plate of ``plate('Quad4R')`` with a central point load whose
    linear deflection is three times the thickness, instead of the
    pressure"""
    elements, data, prop, x, N, bu, fext, deflection = plate('Quad4R')
    h = 0.01
    center = int(np.argmin((x[0::3] - 0.5)**2 + (x[1::3] - 0.5)**2))
    dof = DOF*center + 2
    f = np.zeros(N)
    f[dof] = 1.
    model = Model(elements, data, prop, x, N, f)
    w1 = solve(model.KC0[bu, :][:, bu], f[bu])[np.flatnonzero(bu) == dof][0]
    P = 3*h/w1
    model.fext = P*f
    return model, bu, dof, P


@pytest.mark.parametrize('NL_method', ['NR', 'arc_length_riks'])
def test_prescribed_displacement_and_reaction(NL_method):
    r"""Displacement control: the central deflection of the load-controlled
    analysis, prescribed with ``uk``, gives the same displacement field and
    a reaction equal to the load"""
    model, bu, dof, P = point_loaded_plate()
    an = model.analysis(bu, initialInc=0.2, relTOL=1.e-10)
    increments, cs = an.static(NLgeom=True, silent=True)
    u_load = cs[-1]
    w_c = u_load[dof]
    print('load control: w/h = %.6f' % (w_c/0.01))

    bu_d = bu.copy()
    bu_d[dof] = False
    uk = np.zeros(model.N)
    uk[dof] = w_c
    model.fext = np.zeros(model.N)
    an_d = model.analysis(bu_d, uk=uk, initialInc=0.2, relTOL=1.e-10,
                          NL_method=NL_method)
    increments_d, cs_d = an_d.static(NLgeom=True, silent=True)
    assert np.isclose(increments_d[-1], 1.)
    for lbd, u in zip(increments_d, cs_d):
        assert np.isclose(u[dof], lbd*w_c, rtol=1.e-14)
    u_disp = cs_d[-1]
    np.testing.assert_allclose(u_disp, u_load, rtol=1.e-6,
                               atol=1.e-8*np.abs(u_load).max())
    reaction = model.internal_forces(u_disp)[dof]
    print('reaction %.8e, load %.8e' % (reaction, P))
    assert np.isclose(reaction, P, rtol=1.e-6)
    # NOTE the reaction grows faster than the deflection: membrane
    #      stiffening
    reactions = [model.internal_forces(u)[dof]/lbd
                 for lbd, u in zip(increments_d, cs_d)]
    assert np.all(np.diff(reactions) > 0)


def piston_theory_plate():
    r"""Quad4R plate under pressure with the piston-theory aerodynamic load
    `\{F_a\} = \beta [K_{A_\beta}]\{u\}`, scaled to 20 % of the pressure
    load in the linear solution"""
    elements, data, prop, x, N, bu, fext, deflection = plate('Quad4R')
    r = np.zeros(data.KA_BETA_SPARSE_SIZE*len(elements), dtype=INT)
    c = np.zeros(data.KA_BETA_SPARSE_SIZE*len(elements), dtype=INT)
    v = np.zeros(data.KA_BETA_SPARSE_SIZE*len(elements), dtype=DOUBLE)
    for k, elem in enumerate(elements):
        elem.init_k_KA_beta = k*data.KA_BETA_SPARSE_SIZE
        elem.update_probe_xe(x)
        elem.update_KA_beta(r, c, v)
    KA = coo_matrix((v, (r, c)), shape=(N, N)).tocsr()
    model = Model(elements, data, prop, x, N, fext)
    u_lin = np.zeros(N)
    u_lin[bu] = solve(model.KC0[bu, :][:, bu], fext[bu])
    beta = 0.2*np.linalg.norm(fext[bu])/np.linalg.norm((KA @ u_lin)[bu])
    return model, bu, beta*KA, deflection


def test_piston_theory_load_unsymmetric_tangent():
    r"""The aerodynamic load is configuration-dependent, `\{F_a(u)\} =
    \lambda \beta [K_{A_\beta}]\{u\}` with the load factor `\lambda`: it
    enters ``calc_fint`` with ``inc`` and its load stiffness `-\lambda \beta
    [K_{A_\beta}]`, unsymmetric, enters ``calc_kC``. Newton-Raphson
    converges quadratically, the residual of the full model vanishes, and
    the linear static analysis with ``calc_fext(c=...)`` gives the solution
    of `([K_{C_0}] - \beta [K_{A_\beta}])\{u\} = \{F\}`"""
    model, bu, BKA, deflection = piston_theory_plate()
    from pyfe3d.solver import is_symmetric
    assert not is_symmetric((model.KC0 - BKA)[bu, :][:, bu])

    zetas = []

    def calc_fint(c, inc=1., silent=True):
        f = model.internal_forces(c) - inc*(BKA @ c)
        R = (inc*model.fext - f)[bu]
        w = crisfield_weights(model.KC0[bu, :][:, bu])
        zetas.append((inc, crisfield_zeta(R, inc*model.fext[bu], f[bu], w)))
        return f

    def calc_kC(c=None, NLgeom=False, inc=1., silent=True):
        if c is None or not NLgeom:
            return model.KC0
        return model.KC0 + model.KCNL(c) - inc*BKA

    def calc_fext(inc=1., silent=True, c=None):
        if c is None:
            return inc*model.fext
        return inc*(model.fext + BKA @ c)

    an = Analysis(calc_fext, calc_fint, calc_kC, model.calc_kG, bu=bu)
    an.initialInc = 1.
    an.relTOL = 1.e-10
    increments, cs = an.static(NLgeom=True, silent=True)
    assert increments == [1.]
    u = cs[-1]
    R = (model.fext + BKA @ u - model.internal_forces(u))[bu]
    assert np.linalg.norm(R) <= 1.e-8*np.linalg.norm(model.fext[bu])
    z = [zeta for inc, zeta in zetas]
    print('zeta', ' '.join('%.1e' % zi for zi in z))
    assert len(z) <= 8
    for z0, z1 in zip(z[:-1], z[1:]):
        if z0 < 1.e-2:
            assert z1 < max(10*z0**1.8, 1.e-11), z

    # NOTE the aerodynamic load changes the solution
    model_dead = Model(model.elements, model.data, model.prop, model.x,
                       model.N, model.fext)
    an_dead = model_dead.analysis(bu, initialInc=1., relTOL=1.e-10)
    u_dead = an_dead.static(NLgeom=True, silent=True)[1][-1]
    assert abs(deflection(u) - deflection(u_dead)) > 1.e-3*deflection(u_dead)

    # NOTE geometrically linear: the load correction iterations of the linear
    #      static analysis
    an_lin = Analysis(calc_fext, None, calc_kC, None, bu=bu)
    an_lin.relTOL = 1.e-12
    an_lin.maxNumIter = 100
    u_lin = an_lin.static(silent=True)[1][-1]
    ref = np.zeros(model.N)
    ref[bu] = solve((model.KC0 - BKA)[bu, :][:, bu], model.fext[bu])
    np.testing.assert_allclose(u_lin, ref, rtol=1.e-8,
                               atol=1.e-10*np.abs(ref).max())


if __name__ == '__main__':
    test_crisfield_criterion_units_invariance('beam', 'BeamC')
