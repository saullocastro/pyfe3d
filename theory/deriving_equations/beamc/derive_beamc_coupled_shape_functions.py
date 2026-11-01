r"""
Shape functions of BeamC from the coupled sectional equations (derived here)

   ^ y axis
   |
   |
   ______   --> x axis
   1    2
   Reference 2D plane for all derivations

Strain energy per unit length (eps^T C eps)/2, with the generalized strains
eps = {u_x, v_x - rz, w_x + ry, rx_x, -rz_x, ry_x} and any symmetric
positive definite 6x6 C, see the notes of pyfe3d.beamc.BeamC.

1. Equilibrium of a prismatic element loaded only at its nodes: with the
   virtual work, the sectional forces F = C eps = {N, Vy, Vz, Mx, M4, M5}
   satisfy N' = Vy' = Vz' = Mx' = 0, M4' = Vy, M5' = Vz. Hence
   F = M(x) q, q = {N, Vy, Vz, Mx, M4(0), M5(0)}, M(x) = I + x Z, and the
   strains are eps = S M(x) q, S = C^-1, linear in x.
2. Integrating the kinematic relations from x = 0 gives the 12-parameter
   field d(x) = R(x) d0 + P(x) q, d = {u, v, w, rx, ry, rz}, with the
   rigid-body motion R(x) d0 of node 1. The six "force" parameters q give
   displacements cubic in x for v and w and quadratic for u, rx, ry, rz.
3. The nodal values: d(0) = d0 = u1, d(L) = R(L) u1 + P(L) q = u2. With the
   statics of the element clamped at node 1, q = T P2 for the loads P2 at
   node 2, and the flexibility Fc = P(L) T, symmetric positive definite:
   q = T Fc^-1 (u2 - R(L) u1) = Hf ue, and the shape functions are
   N(x) = [R(x), 0] + P(x) Hf.
4. KC0 = int (B^T C B) dx = Hf^T (int M^T S M dx) Hf = G^T Fc^-1 G,
   G = [-R(L), I], since T^T (int M^T S M dx) T = Fc.

pyfe3d.beamc.BeamC evaluates steps 1-4 numerically, at run time, from S
(``_coupled_fields`` in pyfe3d/beamc.pyx) when C has couplings other than
C[0, 3]; otherwise it uses the closed-form expressions of the shape
functions of Luo (2008), printed by the other scripts of this folder.

This script checks:
- symbolically, for a general S of 21 entries, that the field of step 2
  gives the strains S M(x) q of step 1, i.e. it is the exact solution;
- symbolically, that for an uncoupled section, i.e. S with only the
  extension-twist coupling S03, the shape functions of step 3 are those of
  Luo (2008), Eqs. (19)-(20), with alpha = -12 C44/(C11 L**2) < 0, as in
  derive_beamc_KC0_sparse.py;
- numerically, for a C with all 21 entries non-zero, that one element gives
  the tip displacements of the cantilever of step 1 under the six tip
  loads, that KC0 of step 4 is the Hessian of the strain energy of the
  shape functions of step 3, and that T^T (int M^T S M dx) T = Fc.

Reference: Luo, Y., 2008, "An Efficient 3D Timoshenko Beam Element with
Consistent Shape Functions," Adv. Theor. Appl. Mech., 1(3), pp. 95-106.

"""
import numpy as np
import sympy
from sympy import Matrix, symbols, integrate, simplify, zeros, eye

x, L = symbols('x, L', positive=True)
U, V, W, RX, RY, RZ = range(6)

# general symmetric compliance S = C^-1, 21 entries
Ssym = Matrix(6, 6, lambda i, j: symbols('S%d%d' % (min(i, j), max(i, j))))
q = Matrix(symbols('N, Vy, Vz, Mx, M40, M50'))
d0 = Matrix(symbols('u0, v0, w0, rx0, ry0, rz0'))

# step 1: sectional forces and strains
Z = zeros(6, 6)
Z[4, 1] = 1
Z[5, 2] = 1


def field(S):
    """Field d(x) = R(x) d0 + P(x) q of step 2, as a 6x1 Matrix"""
    eps = S*(eye(6) + x*Z)*q
    t = symbols('t', positive=True)
    I = lambda f: integrate(f.subs(x, t), (t, 0, x))
    u = I(eps[0])
    rx = I(eps[3])
    rz = I(-eps[4])
    ry = I(eps[5])
    v = I(eps[1] + rz)
    w = I(eps[2] - ry)
    d = Matrix([u, v, w, rx, ry, rz])
    R = eye(6)
    R[V, RZ] = x
    R[W, RY] = -x
    return R*d0 + d, R


def strains(d):
    u, v, w, rx, ry, rz = d
    return Matrix([u.diff(x), v.diff(x) - rz, w.diff(x) + ry, rx.diff(x),
                   -rz.diff(x), ry.diff(x)])


d, R = field(Ssym)
print('step 2: field of a general S')
for name, di in zip(['u', 'v', 'w', 'rx', 'ry', 'rz'], d):
    print('   ', name, '=', sympy.collect(sympy.expand(di), x))
check = (strains(d) - Ssym*(eye(6) + x*Z)*q).expand()
assert check == zeros(6, 1)
print('step 2: the field gives the strains S M(x) q of step 1 - OK')


def shape_functions(S):
    """N(x) (6x12) of step 3"""
    d, R = field(S)
    P = d.jacobian(q)
    PL = P.subs(x, L)
    RL = R.subs(x, L)
    T = eye(6)
    T[4, :] = zeros(1, 6)
    T[5, :] = zeros(1, 6)
    T[4, RZ] = -1
    T[4, V] = -L
    T[5, RY] = 1
    T[5, W] = -L
    Fc = PL*T
    Hf = T*Fc.inv()*(-RL).row_join(eye(6))
    N = R.row_join(zeros(6, 6)) + P*Hf
    return N, Fc, T, P


# step 3, uncoupled section: Luo (2008)
S00, S03, S33, S11, S22, S44, S55 = symbols('S00, S03, S33, S11, S22, S44, '
                                            'S55', positive=True)
Sunc = zeros(6, 6)
Sunc[0, 0] = S00
Sunc[0, 3] = Sunc[3, 0] = S03
Sunc[3, 3] = S33
Sunc[1, 1] = S11
Sunc[2, 2] = S22
Sunc[4, 4] = S44
Sunc[5, 5] = S55
Nunc, _, _, _ = shape_functions(Sunc)

xi = x/L
# alphay = -12*C44/(C11*L**2) with C44 = 1/S44 and C11 = 1/S11
alphay = -12*S11/(S44*L**2)
alphaz = -12*S22/(S55*L**2)
betay = 1/(1 - alphay)
betaz = 1/(1 - alphaz)
S1 = 1 - xi
S2 = xi
Hv1 = betay*( 2*xi**3 - 3*xi**2 + alphay*xi + 1 - alphay)
Hv2 = betay*(-2*xi**3 + 3*xi**2 - alphay*xi)
Hw1 = betaz*( 2*xi**3 - 3*xi**2 + alphaz*xi + 1 - alphaz)
Hw2 = betaz*(-2*xi**3 + 3*xi**2 - alphaz*xi)
Hrz1 = L*betay*(xi**3 + (alphay/2 - 2)*xi**2 + (1 - alphay/2)*xi)
Hrz2 = L*betay*(xi**3 - (1 + alphay/2)*xi**2 + (alphay/2)*xi)
Hry1 = -L*betaz*(xi**3 + (alphaz/2 - 2)*xi**2 + (1 - alphaz/2)*xi)
Hry2 = -L*betaz*(xi**3 - (1 + alphaz/2)*xi**2 + (alphaz/2)*xi)
Gv1 = 6*betay/L*( xi**2 - xi)
Gv2 = 6*betay/L*(-xi**2 + xi)
Gw1 = -6*betaz/L*( xi**2 - xi)
Gw2 = -6*betaz/L*(-xi**2 + xi)
Grz1 = betay*(3*xi**2 + (alphay - 4)*xi + 1 - alphay)
Grz2 = betay*(3*xi**2 - (alphay + 2)*xi)
Gry1 = betaz*(3*xi**2 + (alphaz - 4)*xi + 1 - alphaz)
Gry2 = betaz*(3*xi**2 - (alphaz + 2)*xi)
# the functions of derive_beamc_KC0_sparse.py
Nluo = Matrix([
    [S1, 0, 0, 0, 0, 0, S2, 0, 0, 0, 0, 0],
    [0, Hv1, 0, 0, 0, Hrz1, 0, Hv2, 0, 0, 0, Hrz2],
    [0, 0, Hw1, 0, Hry1, 0, 0, 0, Hw2, 0, Hry2, 0],
    [0, 0, 0, S1, 0, 0, 0, 0, 0, S2, 0, 0],
    [0, 0, Gw1, 0, Gry1, 0, 0, 0, Gw2, 0, Gry2, 0],
    [0, Gv1, 0, 0, 0, Grz1, 0, Gv2, 0, 0, 0, Grz2],
    ])
diff = (Nunc - Nluo).applyfunc(lambda e: simplify(sympy.together(e)))
assert diff == zeros(6, 12)
print('step 3: uncoupled S (S03 only) gives the shape functions of Luo (2008)'
      ' - OK')

# numerical checks with a C of 21 non-zero entries
rng = np.random.default_rng(5)
d_ = np.sqrt([1.05e8, 3.25e7, 3.25e7, 2.2e3, 7.9e3, 2.2e4])
Rr = 0.25*rng.uniform(-1, 1, (6, 6))
Rr = (Rr + Rr.T)/2
np.fill_diagonal(Rr, 1.)
Cn = Rr*np.outer(d_, d_)
assert np.all(np.linalg.eigvalsh(Cn) > 0) and np.all(Cn != 0)
Sn = np.linalg.inv(Cn)
Ln = 0.4
Snum = Matrix(6, 6, lambda i, j: sympy.Float((Sn[i, j] + Sn[j, i])/2, 30))
Nnum, Fc, T, P = shape_functions(Snum)
Nnum = Nnum.subs(L, Ln)
Fcn = np.array(Fc.subs(L, Ln).evalf(), dtype=float)
Tn = np.array(T.subs(L, Ln), dtype=float)
K22 = np.linalg.inv(Fcn)
G = np.hstack([-np.array(R.subs(x, Ln), dtype=float), np.eye(6)])
KC0 = G.T @ K22 @ G
# Hessian of the strain energy of the shape functions
B = strains(Nnum*Matrix(symbols('q0:12'))).jacobian(Matrix(symbols('q0:12')))
Hess = (B.T*Matrix(Cn)*B).applyfunc(lambda e: integrate(sympy.expand(e),
                                                       (x, 0, Ln)))
Hess = np.array(Hess.evalf(), dtype=float)
err = np.abs(Hess - KC0).max()/np.abs(KC0).max()
print('step 4: KC0 = G^T Fc^-1 G against the Hessian of the energy, rel. '
      'error %.1e' % err)
assert err < 1e-12
Mn = (eye(6) + x*Z).subs(L, Ln)
Phi = (Mn.T*Snum*Mn).applyfunc(lambda e: integrate(e, (x, 0, Ln)))
Phi = np.array(Phi.evalf(), dtype=float)
err = np.abs(Tn.T @ Phi @ Tn - Fcn).max()/np.abs(Fcn).max()
print('step 4: T^T (int M^T S M dx) T = Fc, rel. error %.1e' % err)
assert err < 1e-12
# one element, clamped at node 1, tip loads: closed form of the cantilever
for i in range(6):
    f = np.zeros(6)
    f[i] = 1.
    tip = np.linalg.solve(KC0[6:, 6:], f)
    # closed form: integrate the strains S M(x) q, q = T f
    qn = Tn @ f
    exact = np.array(P.subs(x, Ln).subs(L, Ln).evalf(), dtype=float) @ qn
    err = np.abs(tip - exact).max()/np.abs(exact).max()
    assert err < 1e-10, (i, err)
print('one element with 21 non-zero entries of C gives the tip displacements'
      ' of the cantilever under the six tip loads - OK')
