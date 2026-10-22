import numpy as np
import sympy
from sympy import simplify, integrate, Matrix, var, symbols

r"""

   ^ y axis
   |
   |
   ______   --> x axis
   1    2
   Reference 2D plane for all derivations

   Timoshenko 3D beam element with consistent shape functions from:
   Luo, Y., 2008, “An Efficient 3D Timoshenko Beam Element with Consistent Shape Functions,” Adv. Theor. Appl. Mech., 1(3), pp. 95–106.

"""

DOF = 6
NUM_NODES = 2

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
# TODO replace G by G12 and G13, but how to do for the D matrix?
# alphay = 12*E*Iy/(G*A*L**2)
# alphaz = 12*E*Iz/(G*A*L**2)
# betay = 1/(1 - alphay)
# betaz = 1/(1 - alphaz)
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
#              u, v, w, phi, psi, theta (for each node)
#              u, v, w, rx, ry, rz
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

# u v w  rx  ry  rz  (rows are node 1, node2)

# From Eqs. 8 and 9 in Luo, Y. 2008
# exx = u,x + (-rz,x)*y + (ry,x)*z
# exy = (v.diff(x) - rz*y) - (rx)*z # TODO
# exz = (w.diff(x) + ry*z) + (rx)*y # TODO
# BL = Matrix([
    # Nu.diff(x) + (-Nrz.diff(x))*y + Nry.diff(x)*z,
    #(Nv.diff(x) - Nrz) - Nrx*z,
    #(Nw.diff(x) + Nry) + Nrx*y,
    #])
# dy = dz = 0
# BL = integrate(BL, (y, -hy/2+dy, +hy/2+dy))
# BL = simplify(integrate(BL, (z, -hz/2+dz, +hz/2+dz)))

# From Eqs. 12 in Luo, Y. 2008
D = Matrix([
    [ E*A,  E*Ay,  E*Az, 0, 0, 0],
    [E*Ay,  E*Iy, E*Iyz, 0, 0, 0],
    [E*Az, E*Iyz,  E*Iz, 0, 0, 0],
    [   0,     0,     0,   G*A,    0, -G*Az],
    [   0,     0,     0,     0,  G*A,  G*Ay],
    [   0,     0,     0, -G*Az, G*Ay,  G*J]])
# From Eq. 8 in Luo, Y. 2008
# epsilon = u,x
# kappay = -theta,x = -rz,x
# kappaz = psi,x = ry,x
# gammay = v,x - theta = v,x - rz
# gammaz = w,x + psi = w,x + ry
# kappax = phi,x
# putting in a BL matrix
# BL = derivative of NL

# EXample of Solid element =
# exx = u,x + 1/2(u,x*u,x + v,x*v,x + w,x*w,x)
# exy = u,x + 1/2(v,x*u,y...)
# eyy = v,y + 1/2(u,y*u,y + v,y*v,y + w,y*w,y)
# ezz = w,z + 1/2(u,z*u,z + v,z*v,z + w,z*w,z)

BL = Matrix([
    Nu.diff(x),
    -Nrz.diff(x),
    Nry.diff(x),
    Nv.diff(x) - Nrz,
    Nw.diff(x) + Nry,
    Nrx.diff(x)])

KC0e = BL.T*D*BL
KC0e = integrate(KC0e, (x, 0, L))

nonzero = set()
for ind, val in np.ndenumerate(KC0e):
    if sympy.expand(val) == 0:
        continue
    i, j = ind
    if i > j:
        continue # NOTE ignoring symmetric part
    name = 'KC0e%02d%02d' % (i, j)
    nonzero.add(name)
    print('%s = %s' % (name, simplify(val)))

rows = []
for i in range(NUM_NODES*DOF):
    cols = []
    for j in range(NUM_NODES*DOF):
        if j >= i:
            name = 'KC0e%02d%02d' % (i, j)
        else:
            name = 'KC0e%02d%02d' % (j, i)
        if name in nonzero:
            cols.append(var(name))
        else:
            cols.append(0)
    rows.append(cols)
KC0e = Matrix(rows)

# displacements in global coordinates corresponding to one finite element
ue = Matrix([symbols(r'ue[%d]' % i) for i in range(0, KC0e.shape[1])])

finte = KC0e*ue

print('transformation local to global')
var('r11, r12, r13, r21, r22, r23, r31, r32, r33')
Rlocal2global = Matrix([[r11, r12, r13],
                        [r21, r22, r23],
                        [r31, r32, r33]])
R = sympy.zeros(NUM_NODES*DOF, NUM_NODES*DOF)
for i in range(2*NUM_NODES):
    R[i*DOF//2:(i+1)*DOF//2, i*DOF//2:(i+1)*DOF//2] += Rlocal2global

nonzero = set()
for ind, val in np.ndenumerate(finte):
    if sympy.expand(val) == 0:
        continue
    i, j = ind
    name = 'finte[%d]' % (i)
    nonzero.add(name)
    print('%s = %s' % (name, simplify(val)))

rows = []
for i in range(NUM_NODES*DOF):
    name = 'finte[%d]' % (i)
    if name in nonzero:
        rows.append(var(name))
    else:
        rows.append(0)
finte = Matrix(rows)
fint = R*finte


def name_ind(i):
    if i >= 0*DOF and i < 1*DOF:
        return 'c1'
    elif i >= 1*DOF and i < 2*DOF:
        return 'c2'

        raise

print()
print()
print('_______________________________________')
print()
print('printing fint')
print('_______________________________________')
print()
print()
for ind, val in np.ndenumerate(fint):
    i, j = ind
    si = name_ind(i)
    if sympy.expand(val) == 0:
        continue
    print('            fint[%d+self.%s] +=' % (i%DOF, si), val)
print()
print()
