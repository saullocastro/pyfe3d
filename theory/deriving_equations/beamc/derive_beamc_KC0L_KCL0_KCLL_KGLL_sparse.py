"""
Nonlinear stiffness matrix KCNL, K_CNL + K_GNL, for the BeamC element

Von Kármán axial strain with the nonlinear part replaced by its element mean

    exx = u,x + eNLm,    eNLm = (1/L)*int_0^L (v,x**2 + w,x**2)/2 dx

With u linear and v, w cubic, the pointwise eNL = (v,x**2 + w,x**2)/2 is a
quartic in x that the constant u,x cannot balance, which gives membrane
locking. Crisfield replaces the membrane strain by its average over the
element, Crisfield (1991) vol. 1, Sec. 7.1.3, Eqs. (7.31) and (7.32), there
justified by the Hu-Washizu principle with a constant membrane strain, and
uses the same average for the local element of a co-rotational beam,
Crisfield (1991) vol. 1, Sec. 7.2.7, Eqs. (7.89) and (7.90), and Crisfield
(1997) vol. 2, Sec. 17.1.5, Eqs. (17.56a) and (17.56b).

NOTE own derivation, checked below: with eNLm in place of eNL, every pointwise
quantity of the previous formulation is replaced by its element mean

- Hm = (1/L)*int_0^L (Sv_x.T*Sv_x + Sw_x.T*Sw_x) dx, such that
  eNLm = ue.T*Hm*ue/2
- BNLm = d(eNLm)/d(ue) = (Hm*ue).T = (1/L)*int_0^L BNL dx
- EBLm = (1/L)*int_0^L C[0, :]*{BLexx, BLgy, BLgz, BLkx, BLky, BLkz} dx,
  the row of the mean linear axial force Nm = EBLm*ue, with the first row
  of the sectional stiffness matrix C of pyfe3d.beamprop.BeamProp, i.e.
  (1/L)*int_0^L E*(A*BLexx + Ay*BLky + Az*BLkz) dx without its attribute C
- NNLm = C00*eNLm

and the strain energy of the axial strain becomes

    U = UL + L*Nm*eNLm + L*C00*eNLm**2/2

where UL is the energy of the linear strains, such that

- finte = KC0*ue + L*(EBLm.T*eNLm + BNLm.T*(Nm + NNLm))
- KC0L = L*EBLm.T*BNLm
- KCL0 = L*BNLm.T*EBLm = KC0L.T
- KCLL = L*C00*BNLm.T*BNLm
- KGLL = L*NNLm*Hm
- KG = L*Nm*Hm, the geometric stiffness of the mean linear axial force, see
  derive_beamc_KG_sparse.py

and KT = KC0 + KCNL(ue) + KG(ue), KCNL = KC0L + KCL0 + KCLL + KGLL, is the exact
Hessian of U, which is asserted below. The 1/60*theta.T*X*theta of
Crisfield (1997) Eq. (17.56a) is eNLm for a straight Euler-Bernoulli beam,
alphay = alphaz = 0, with zero transverse nodal displacements, which is also
asserted below.

The shear flexibility enters only through the slopes v,x and w,x of the
consistent shape functions, i.e. through alphay, betay, alphaz and betaz in
Hm. The averaging does not change the linear part, KC0, nor KG when
C04 = C05 = 0 (Ay = Az = 0 without the attribute C), since the linear axial
force is then uniform.

Timoshenko 3D beam element with consistent shape functions from:
Luo, Y., 2008, “An Efficient 3D Timoshenko Beam Element with Consistent Shape Functions,” Adv. Theor. Appl. Mech., 1(3), pp. 95–106.

Crisfield, M. A., 1991, Non-linear Finite Element Analysis of Solids and
Structures, Vol. 1: Essentials, Wiley.

Crisfield, M. A., 1997, Non-linear Finite Element Analysis of Solids and
Structures, Vol. 2: Advanced Topics, Wiley.

"""
import numpy as np
import sympy
from sympy import simplify, expand, integrate, Matrix, var, symbols, Rational

DOF = 6
NUM_NODES = 2

var('x', real=True)
var('L, E, Iyy, Izz, Iyz, G12, G13, GJ, A, Ay, Az', real=True)

# definitions of Eqs. 20 and 21 of Luo, Y., 2008
xi = x/L
# NOTE in Luo 2008 Iy represents the area moment of inertia in the plane of y
#     or rotating about the z axis. Here we say that Izz = Iy
# NOTE in Luo 2008 Iz represents the area moment of inertia in the plane of z
#     or rotating about the y axis. Here we say that Iyy = Iz
Iy = Izz
Iz = Iyy
var('alphay, alphaz, betay, betaz', real=True)

S1 = 1 - xi
S2 = xi
Hv1 = betay*( 2*xi**3 - 3*xi**2 + alphay*xi + 1 - alphay)
Hv2 = betay*(-2*xi**3 + 3*xi**2 - alphay*xi)
Hw1 = betaz*( 2*xi**3 - 3*xi**2 + alphaz*xi + 1 - alphaz)
Hw2 = betaz*(-2*xi**3 + 3*xi**2 - alphaz*xi)
Hrz1 = Htheta1 = L*betay*(xi**3 + (alphay/2 - 2)*xi**2 + (1 - alphay/2)*xi)
Hrz2 = Htheta2 = L*betay*(xi**3 - (1 + alphay/2)*xi**2 + (alphay/2)*xi)
Hry1 = Hpsi1   = -L*betaz*(xi**3 + (alphaz/2 - 2)*xi**2 + (1 - alphaz/2)*xi)
Hry2 = Hpsi2   = -L*betaz*(xi**3 - (1 + alphaz/2)*xi**2 + (alphaz/2)*xi)
Gv1 = 6*betay/L*( xi**2 - xi)
Gv2 = 6*betay/L*(-xi**2 + xi)
Gw1 = -6*betaz/L*( xi**2 - xi)
Gw2 = -6*betaz/L*(-xi**2 + xi)
Grz1 = Gtheta1 = betay*(3*xi**2 + (alphay - 4)*xi + 1 - alphay)
Grz2 = Gtheta2 = betay*(3*xi**2 - (alphay + 2)*xi)
Gry1 = Gpsi1   = betaz*(3*xi**2 + (alphaz - 4)*xi + 1 - alphaz)
Gry2 = Gpsi2   = betaz*(3*xi**2 - (alphaz + 2)*xi)

# Degrees-of-freedom illustrated in Fig. 1 of Luo, Y., 2008
#              u, v, w, rx, ry, rz (for each node)
# interpolation according to Eq. 19 of Luo, Y. 2008
Nu =  Matrix([[S1, 0, 0, 0, 0, 0,
               S2, 0, 0, 0, 0, 0]])
Nv =  Matrix([[0, Hv1, 0, 0, 0, Hrz1,
               0, Hv2, 0, 0, 0, Hrz2]])
Nw =  Matrix([[0, 0, Hw1, 0, Hry1, 0,
               0, 0, Hw2, 0, Hry2, 0]])
Nrx = Matrix([[0, 0, 0, S1, 0, 0,
               0, 0, 0, S2, 0, 0]])
Nry = Matrix([[0, 0, Gw1, 0, Gry1, 0,
               0, 0, Gw2, 0, Gry2, 0]])
Nrz = Matrix([[0, Gv1, 0, 0, 0, Grz1,
               0, Gv2, 0, 0, 0, Grz2]])

# From Eqs. 12 in Luo, Y. 2008
D11 = Matrix([
    [ E*A,  E*Ay,  E*Az, 0, 0, 0],
    [E*Ay,  E*Iy, E*Iyz, 0, 0, 0],
    [E*Az, E*Iyz,  E*Iz, 0, 0, 0],
    [   0,     0,     0,   G12*A,      0, -G12*Az],
    [   0,     0,     0,       0,  G13*A,  G13*Ay],
    [   0,     0,     0, -G12*Az, G13*Ay,      GJ]])

# NOTE sectional stiffness matrix C of pyfe3d.beamprop.BeamProp, symmetric,
#      with the 21 independent entries Cij, i <= j, in the order of the
#      strains {eps, gammay, gammaz, kx, ky, kz} of pyfe3d, see
#      pyfe3d.vabs.yu_to_pyfe3d. D is C in the order {exx, ky, kz, gammay,
#      gammaz, kx} of the rows of BL below
C = Matrix(6, 6, lambda i, j: var('C%d%d' % (min(i, j), max(i, j)), real=True))
PERM = [0, 4, 5, 1, 2, 3]
D = C.extract(PERM, PERM)
# NOTE without the attribute C, BeamProp builds C from its 11 stiffness
#      attributes, such that D becomes D11 above
C_FROM_11 = {D[i, j]: D11[i, j] for i in range(6) for j in range(i, 6)}
assert D.subs(C_FROM_11) == D11

# From Eq. 8 in Luo, Y. 2008
# {exx, ky, kz, gammay, gammaz, kx}
BLexx = Nu.diff(x)
BLky = -Nrz.diff(x)
BLkz = Nry.diff(x)
BL = Matrix([
    BLexx,
    BLky,
    BLkz,
    Nv.diff(x) - Nrz,
    Nw.diff(x) + Nry,
    Nrx.diff(x)])

# slopes v,x and w,x
Sv_x = Nv.diff(x)
Sw_x = Nw.diff(x)

ue = Matrix([symbols(r'ue[%d]' % i) for i in range(NUM_NODES*DOF)])
v_x = (Sv_x*ue)[0]
w_x = (Sw_x*ue)[0]

# pointwise nonlinear part of the axial strain and its variation
eNL = (v_x**2 + w_x**2)/2
BNL = Matrix([[eNL.diff(ui) for ui in ue]])
assert expand(BNL - (v_x*Sv_x + w_x*Sw_x)) == sympy.zeros(1, NUM_NODES*DOF)

# NOTE own derivation: element means, see the docstring
G = Sv_x.T*Sv_x + Sw_x.T*Sw_x
Hm = (integrate(expand(G), (x, 0, L))/L).applyfunc(simplify)
eNLm = (ue.T*Hm*ue)[0]/2
BNLm = (Hm*ue).T
BLgy = Nv.diff(x) - Nrz
BLgz = Nw.diff(x) + Nry
BLkx = Nrx.diff(x)
EBL = (D*BL)[0, :]
assert expand(EBL - (C00*BLexx + C04*BLky + C05*BLkz + C01*BLgy + C02*BLgz
                     + C03*BLkx)) == sympy.zeros(1, NUM_NODES*DOF)
EBLm = (integrate(EBL, (x, 0, L))/L).applyfunc(simplify)
# NOTE without the attribute C, EBLm is E*(A*BLexx + Ay*BLky + Az*BLkz)
EBLm11 = integrate(E*(A*BLexx + Ay*BLky + Az*BLkz), (x, 0, L))/L
assert expand(EBLm.subs(C_FROM_11) - EBLm11) == sympy.zeros(1, NUM_NODES*DOF)
Nm = (EBLm*ue)[0]
NNLm = C00*eNLm

print('checking the element means', flush=True)
assert expand(eNLm - integrate(expand(eNL), (x, 0, L))/L) == 0
assert expand(BNLm - integrate(expand(BNL), (x, 0, L))/L) == sympy.zeros(1, NUM_NODES*DOF)
assert expand(BNLm - Matrix([[eNLm.diff(ui) for ui in ue]])) == sympy.zeros(1, NUM_NODES*DOF)
# NOTE Hm couples only v with rz and w with ry, as G does
for i in range(NUM_NODES*DOF):
    for j in range(NUM_NODES*DOF):
        if Hm[i, j] != 0:
            assert (i % DOF, j % DOF) in [(a, b) for a in (1, 5) for b in (1, 5)] + [(a, b) for a in (2, 4) for b in (2, 4)]
print('    OK', flush=True)

print('checking Crisfield (1997) Eq. (17.56a): eNLm = theta.T*X*theta/60', flush=True)
# NOTE Euler-Bernoulli, straight, zero transverse nodal displacements; the
#     local rotations of Crisfield are (theta_l1, ..., theta_l6) = (rx1, ry1,
#     rz1, rx2, ry2, rz2), see Crisfield (1997) Sec. 17.1.5, step 6
X = Matrix([
    [0,  0,  0, 0,  0,  0],
    [0,  4,  0, 0, -1,  0],
    [0,  0,  4, 0,  0, -1],
    [0,  0,  0, 0,  0,  0],
    [0, -1,  0, 0,  4,  0],
    [0,  0, -1, 0,  0,  4]])
theta = Matrix([ue[3], ue[4], ue[5], ue[9], ue[10], ue[11]])
EB = {alphay: 0, alphaz: 0, betay: 1, betaz: 1,
      ue[1]: 0, ue[2]: 0, ue[7]: 0, ue[8]: 0}
assert expand(eNLm.subs(EB) - (theta.T*X*theta)[0]/60) == 0
print('    OK', flush=True)

# NOTE the linear part is not changed
eps = BL*ue
KC0e = integrate(expand(BL.T*D*BL), (x, 0, L))
UL = integrate(expand((eps.T*D*eps)[0]/2), (x, 0, L))
U = UL + L*Nm*eNLm + L*C00*eNLm**2/2

# finte, KCNL and KG in element coordinates
finteNL = L*(EBLm.T*eNLm + BNLm.T*(Nm + NNLm))
KC0Le = L*EBLm.T*BNLm
KCL0e = L*BNLm.T*EBLm
KCLLe = L*C00*BNLm.T*BNLm
KGLLe = L*NNLm*Hm
KCNLe = KC0Le + KCL0e + KCLLe + KGLLe
KGe = L*Nm*Hm

print('checking that finte is the gradient of U', flush=True)
grad = Matrix([U.diff(ui) for ui in ue])
assert expand(grad - KC0e*ue - finteNL) == sympy.zeros(NUM_NODES*DOF, 1)
print('    OK', flush=True)

print('checking that KC0 + KCNL + KG is the Hessian of U', flush=True)
for i in range(NUM_NODES*DOF):
    for j in range(i, NUM_NODES*DOF):
        assert expand(grad[i].diff(ue[j]) - (KC0e[i, j] + KCNLe[i, j] + KGe[i, j])) == 0
assert expand(KCNLe - KCNLe.T) == sympy.zeros(NUM_NODES*DOF, NUM_NODES*DOF)
print('    OK', flush=True)

print('checking that KG is homogeneous of degree one in ue', flush=True)
var('s')
assert expand(KGe.subs({ui: s*ui for ui in ue}, simultaneous=True) - s*KGe) == sympy.zeros(NUM_NODES*DOF, NUM_NODES*DOF)
print('    OK', flush=True)


# NOTE beamc.pyx is compiled with cdivision=True, such that the rational
#     numbers are printed with a floating-point numerator
from sympy.printing.str import StrPrinter
class CPrinter(StrPrinter):
    def _print_Rational(self, expr):
        return '%d./%d' % (expr.p, expr.q)
def cstr(expr):
    return CPrinter().doprint(expr)
BETAS = {betay: 1/(1 - alphay), betaz: 1/(1 - alphaz)}

print()
print()
print('_______________________________________')
print()
print('printing code for the element means Hm and EBLm')
print('_______________________________________')
print()
print()
print('        # Hm = (1/L)*int_0^L (Sv_x.T*Sv_x + Sw_x.T*Sw_x) dx, stored row by row')
for ind, val in np.ndenumerate(Hm):
    if val == 0:
        continue
    i, j = ind
    print('        Hm[%d] = %s' % (NUM_NODES*DOF*i + j, cstr(simplify(val))))
print()
print('        # EBLm = (1/L)*int_0^L (C00*BLexx + C04*BLky + C05*BLkz + C01*BLgy + C02*BLgz + C03*BLkx) dx')
# NOTE printed with alphay*betay and alphaz*betaz, using betay*(1 - alphay)
#      = betaz*(1 - alphaz) = 1; the mean shear strains are
#      gammay = alphay*betay*((v1 - v2)/L + (rz1 + rz2)/2) and
#      gammaz = alphaz*betaz*((w1 - w2)/L - (ry1 + ry2)/2)
EBLm_print = [-C00/L, C01*alphay*betay/L, C02*alphaz*betaz/L, -C03/L,
              -C02*alphaz*betaz/2 - C05/L, C01*alphay*betay/2 + C04/L,
              C00/L, -C01*alphay*betay/L, -C02*alphaz*betaz/L, C03/L,
              -C02*alphaz*betaz/2 + C05/L, C01*alphay*betay/2 - C04/L]
for i, val in enumerate(EBLm):
    assert simplify((val - EBLm_print[i]).subs(BETAS)) == 0
    if val == 0:
        continue
    print('        EBLm[%d] = %s' % (i, cstr(EBLm_print[i])))

print()
print()
print('_______________________________________')
print()
print('printing code for _update_probe_finte_nonlinear and _update_probe_KCNLve')
print('_______________________________________')
print()
print()
# same expressions in terms of the means, as they would be implemented in
# beamc.pyx
def row(name, vals):
    return Matrix([[symbols('%s[%d]' % (name, i)) if vals[i] != 0 else 0
                    for i in range(NUM_NODES*DOF)]])
sEBLm = row('EBLm', EBLm)
sHm = Matrix(NUM_NODES*DOF, NUM_NODES*DOF,
             lambda i, j: symbols('Hm[%d]' % (NUM_NODES*DOF*i + j)) if Hm[i, j] != 0 else 0)
sBNLm = row('BNLm', BNLm)
for i, val in enumerate(BNLm):
    if val == 0:
        continue
    print('        BNLm[%d] = %s' % (i, (sHm[i, :]*ue)[0]))
var('eNLm, Nm, NNLm')
print('        eNLm = %s' % (sum(ue[i]*symbols('BNLm[%d]' % i) for i in range(NUM_NODES*DOF) if BNLm[i] != 0)/2))
print('        Nm = %s' % (sEBLm*ue)[0])
print('        NNLm = C00*eNLm')
print()
sfinte = L*(sEBLm.T*eNLm + sBNLm.T*(Nm + NNLm))
for i, val in enumerate(sfinte):
    if val == 0:
        continue
    print('        finte[%d] += %s' % (i, val))
print()
sKCNLe = L*(sEBLm.T*sBNLm + sBNLm.T*sEBLm + C00*sBNLm.T*sBNLm + NNLm*sHm)
for ind, val in np.ndenumerate(sKCNLe):
    if val == 0:
        continue
    i, j = ind
    print('        KCNLve[%d] = %s' % (NUM_NODES*DOF*i + j, val))

print()
print()
print('_______________________________________')
print()
print('printing code for sparse implementation')
print('_______________________________________')
print()
print()
# NOTE KCNL = R*KCNLe*R.T, with R made of 3x3 blocks for the translations
#     and the rotations of each node, such that any non-zero term fills its
#     whole 3x3 block in global coordinates
nonzero = np.array([[expand(KCNLe[i, j]) != 0
                     for j in range(NUM_NODES*DOF)]
                    for i in range(NUM_NODES*DOF)])
blocks = nonzero.reshape(2*NUM_NODES, 3, 2*NUM_NODES, 3).any(axis=(1, 3))
nonzero = np.kron(blocks, np.ones((3, 3), dtype=bool))

def name_ind(i):
    node = i//DOF
    if node >= 0 and node < NUM_NODES:
        return 'c%d' % (node + 1)
    else:
        raise

# NOTE printing only the non-zero terms, such that KCNL_SPARSE_SIZE can be
#     smaller than the (NUM_NODES*DOF)**2 terms stored by update_KCNL in
#     beamc.pyx
KCNL_SPARSE_SIZE = 0
for ind, val in np.ndenumerate(nonzero):
    if not val:
        continue
    KCNL_SPARSE_SIZE += 1
    i, j = ind
    si = name_ind(i)
    sj = name_ind(j)
    print('            k += 1')
    print('            KCNLr[k] = %d+%s' % (i%DOF, si))
    print('            KCNLc[k] = %d+%s' % (j%DOF, sj))
print('KCNL_SPARSE_SIZE', KCNL_SPARSE_SIZE)
