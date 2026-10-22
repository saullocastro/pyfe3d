"""
Nonlinear constitutive stiffness matrix KCNL for the BeamC element

KCNL = KC0L + KCL0 + KCLL + KGLL, built from the von Karman axial strain

    exx = u,x + eNL,    eNL = (v,x**2 + w,x**2)/2

- KC0L = BL.T*D*BNL
- KCL0 = BNL.T*D*BL = KC0L.T
- KCLL = BNL.T*D*BNL
- KGLL = NNL*(Gvx.T*Gvx + Gwx.T*Gwx), with NNL = E*A*eNL

where BNL = d(eNL)/d(ue) = v,x*Gvx + w,x*Gwx acts on the axial strain only,
such that eNL = BNL*ue/2 and not BNL*ue.

KGLL is the geometric stiffness carried by the axial force of the nonlinear
axial strain, called KGNL in pyfe3d/beamc.pyx. update_KG builds the geometric
stiffness from the linear axial force N = E*(A*u,x + Ay*ky + Az*kz) alone, which
keeps KG homogeneous of degree one in ue, as a linear buckling analysis needs.
The missing piece is collected in KCNL instead, such that

    KT = KC0 + KCNL(ue) + KG(ue)

is the exact Hessian of the strain energy, i.e. the exact Jacobian of fint,
which is asserted below.

Timoshenko 3D beam element with consistent shape functions from:
Luo, Y., 2008, “An Efficient 3D Timoshenko Beam Element with Consistent Shape Functions,” Adv. Theor. Appl. Mech., 1(3), pp. 95–106.

"""
import numpy as np
import sympy
from sympy import simplify, expand, Matrix, var, symbols, Poly

DOF = 6
NUM_NODES = 2
# NOTE number of Gauss-Legendre points used in beamc.pyx
NUM_GAUSS_POINTS = 5

var('x', real=True)
var('L, E, Iyy, Izz, Iyz, G, A, Ay, Az, J', real=True)

# definitions of Eqs. 20 and 21 of Luo, Y., 2008
xi = x/L
# NOTE in Luo 2008 Iy represents the area moment of inertia in the plane of y
#     or rotating about the z axis. Here we say that Izz = Iy
# NOTE in Luo 2008 Iz represents the area moment of inertia in the plane of z
#     or rotating about the y axis. Here we say that Iyy = Iz
Iy = Izz
Iz = Iyy
var('alphay, alphaz, betay, betaz', real=True)

N1 = 1 - xi
N2 = xi
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
Nu =  Matrix([[N1, 0, 0, 0, 0, 0,
               N2, 0, 0, 0, 0, 0]])
Nv =  Matrix([[0, Hv1, 0, 0, 0, Hrz1,
               0, Hv2, 0, 0, 0, Hrz2]])
Nw =  Matrix([[0, 0, Hw1, 0, Hry1, 0,
               0, 0, Hw2, 0, Hry2, 0]])
Nrx = Matrix([[0, 0, 0, N1, 0, 0,
               0, 0, 0, N2, 0, 0]])
Nry = Matrix([[0, 0, Gw1, 0, Gry1, 0,
               0, 0, Gw2, 0, Gry2, 0]])
Nrz = Matrix([[0, Gv1, 0, 0, 0, Grz1,
               0, Gv2, 0, 0, 0, Grz2]])

# From Eqs. 12 in Luo, Y. 2008
D = Matrix([
    [ E*A,  E*Ay,  E*Az, 0, 0, 0],
    [E*Ay,  E*Iy, E*Iyz, 0, 0, 0],
    [E*Az, E*Iyz,  E*Iz, 0, 0, 0],
    [   0,     0,     0,   G*A,    0, -G*Az],
    [   0,     0,     0,     0,  G*A,  G*Ay],
    [   0,     0,     0, -G*Az, G*Ay,  G*J]])

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
Gvx = Nv.diff(x)
Gwx = Nw.diff(x)

ue = Matrix([symbols(r'ue[%d]' % i) for i in range(NUM_NODES*DOF)])
v_x = (Gvx*ue)[0]
w_x = (Gwx*ue)[0]

# nonlinear part of the axial strain and its variation
eNL = (v_x**2 + w_x**2)/2
BNL = Matrix([[eNL.diff(ui) for ui in ue]])
assert expand(BNL - (v_x*Gvx + w_x*Gwx)) == sympy.zeros(1, NUM_NODES*DOF)
# NOTE eNL is quadratic in ue, such that BNL*ue counts it twice
assert expand((BNL*ue)[0]/2 - eNL) == 0
# BNL acts on the axial strain only
BNLfull = Matrix([BNL, sympy.zeros(5, NUM_NODES*DOF)])
epsNL = Matrix([eNL, 0, 0, 0, 0, 0])
NNL = (D*epsNL)[0]

# integrands of KCNL in element coordinates
KC0Le = BL.T*D*BNLfull
KCL0e = BNLfull.T*D*BL
KCLLe = BNLfull.T*D*BNLfull
KGLLe = NNL*(Gvx.T*Gvx + Gwx.T*Gwx)
KCNLe = KC0Le + KCL0e + KCLLe + KGLLe

print('checking that KC0 + KCNL + KG is the Hessian of the strain energy', flush=True)
N = (D*BL*ue)[0]
KC0e = BL.T*D*BL
KGe = N*(Gvx.T*Gvx + Gwx.T*Gwx)
eps = BL*ue + epsNL
U = (eps.T*D*eps)[0]/2
grad = [U.diff(ui) for ui in ue]
for i in range(NUM_NODES*DOF):
    for j in range(i, NUM_NODES*DOF):
        assert expand(grad[i].diff(ue[j]) - (KC0e[i, j] + KCNLe[i, j] + KGe[i, j])) == 0
print('    OK', flush=True)

print('checking that %d Gauss-Legendre points integrate KCNL exactly' % NUM_GAUSS_POINTS, flush=True)
max_degree = 0
for val in KCNLe:
    val = expand(val)
    if val == 0:
        continue
    max_degree = max(max_degree, Poly(val, x).degree())
print('    polynomial degree in x:', max_degree)
assert max_degree <= 2*NUM_GAUSS_POINTS - 1
print('    OK', flush=True)

print()
print()
print('_______________________________________')
print()
print('printing code for _update_probe_BL_G')
print('_______________________________________')
print()
print()
var('xi', real=True)
for name, row in [('BLexx', BLexx), ('BLky', BLky), ('BLkz', BLkz),
                  ('Gvx', Gvx), ('Gwx', Gwx)]:
    for i, val in enumerate(row):
        if val == 0:
            continue
        print('        %s[%d] = %s' % (name, i, simplify(val.subs(x, xi*L))))
    print()

print()
print('_______________________________________')
print()
print('printing code for _update_probe_KCNLve')
print('_______________________________________')
print()
print()
# row of D*BL giving the axial force, N = EBL*ue
EBL = (D*BL)[0, :]
assert expand(EBL - E*(A*BLexx + Ay*BLky + Az*BLkz)) == sympy.zeros(1, NUM_NODES*DOF)
print('            # EBL = E*(A*BLexx + Ay*BLky + Az*BLkz)')
print('            # BNL = v_x*Gvx + w_x*Gwx')
print('            # NNL =', NNL.subs(eNL, var('eNL')), 'with eNL = (v_x**2 + w_x**2)/2')
print()

# same integrand in terms of the probe rows, as implemented in beamc.pyx
var('v_x, w_x, NNL, weight')
def row(name, vals):
    return Matrix([[symbols('%s[%d]' % (name, i)) if vals[i] != 0 else 0
                    for i in range(NUM_NODES*DOF)]])
sBLexx = row('BLexx', BLexx)
sBLky = row('BLky', BLky)
sBLkz = row('BLkz', BLkz)
sGvx = row('Gvx', Gvx)
sGwx = row('Gwx', Gwx)
sEBL = E*(A*sBLexx + Ay*sBLky + Az*sBLkz)
sBNL = v_x*sGvx + w_x*sGwx
sKCNLe = weight*(sEBL.T*sBNL + sBNL.T*sEBL + E*A*sBNL.T*sBNL
                 + NNL*(sGvx.T*sGvx + sGwx.T*sGwx))
for ind, val in np.ndenumerate(sKCNLe):
    if val == 0:
        continue
    i, j = ind
    print('                KCNLve[%d] += %s' % (NUM_NODES*DOF*i + j, val))

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
