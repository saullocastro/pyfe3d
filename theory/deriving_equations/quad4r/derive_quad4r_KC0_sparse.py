r"""
Constitutive stiffness matrix (linear) for Quad4R element

    4 ____ 3
     /   /
    /   /   positive normal in CCW
   /___/
   1    2

"""
import numpy as np
import sympy
from sympy import simplify, Matrix, var
from sympy.vector import CoordSys3D


DOF = 6
NUM_NODES = 4

var('h', positive=True, real=True)
var('x1, y1, x2, y2, x3, y3, x4, y4', real=True, positive=True)
var('xi, eta, A, K6ROT')
var('A11, A12, A16, A22, A26, A66')
var('B11, B12, B16, B22, B26, B66')
var('D11, D12, D16, D22, D26, D66')
var('E44, E45, E55')
#NOTE shear correction factor should be applied to E44, E45 and E55
#     in the finite element code

ONE = sympy.Integer(1)

R = CoordSys3D('R')
r1 = x1*R.i + y1*R.j
r2 = x2*R.i + y2*R.j
r3 = x3*R.i + y3*R.j
r4 = x4*R.i + y4*R.j

rbottom = r1 + (r2 - r1)*(xi + 1)/2.
rtop = r4 + (r3 - r4)*(xi + 1)/2.
r = rbottom + (rtop - rbottom)*(eta + 1)/2.
xfunc = r.components[R.i]
yfunc = r.components[R.j]

# Jacobian
# http://kis.tu.kielce.pl/mo/COLORADO_FEM/colorado/IFEM.Ch17.pdf
#NOTE for linear element these derivatives are constant
# xi = xi(x, y)
# eta = eta(x, y)
#J = [dx/dxi  dy/dxi ]
#    [dx/deta dy/deta]
# dx   = J.T dxi
# dy         deta
#
# dxi   = Jinv.T dx
# deta           dy
#
# Jinv:
# d/dx = d/dxi*dxi/dx + d/deta*deta/dx = [dxi/dx   deta/dx] d/dxi  =  [j11  j12] d/dxi
# d/dy   d/dxi*dxi/dy + d/deta*deta/dy   [dxi/dy   deta/dy] d/deta =  [j21  j22] d/deta
#
J = Matrix([[xfunc.diff(xi),  yfunc.diff(xi)],
            [xfunc.diff(eta), yfunc.diff(eta)]])
detJ = J.det().simplify()
detJfunc = detJ
print('detJ =', detJ.simplify())
print('xi=0, eta=0')
print('detJ =', detJ.subs(dict(xi=0, eta=0)).simplify())

j = J.inv()
j11 = j[0, 0].simplify()
j12 = j[0, 1].simplify()
j21 = j[1, 0].simplify()
j22 = j[1, 1].simplify()

print('j11 =', j11.simplify())
print('j12 =', j12.simplify())
print('j21 =', j21.simplify())
print('j22 =', j22.simplify())

print('xi=0, eta=0')
print('j11 =', j11.subs(dict(xi=0, eta=0)).simplify())
print('j12 =', j12.subs(dict(xi=0, eta=0)).simplify())
print('j21 =', j21.subs(dict(xi=0, eta=0)).simplify())
print('j22 =', j22.subs(dict(xi=0, eta=0)).simplify())

j11, j12, j21, j22 = var('j11, j12, j21, j22')

N1 = (eta*xi - eta - xi + 1)/4.
N2 = -(eta*xi + eta - xi - 1)/4.
N3 = (eta*xi + eta + xi + 1)/4.
N4 = -(eta*xi - eta + xi - 1)/4.

S1_xi = N1.diff(xi)
S2_xi = N2.diff(xi)
S3_xi = N3.diff(xi)
S4_xi = N4.diff(xi)

S1_eta = N1.diff(eta)
S2_eta = N2.diff(eta)
S3_eta = N3.diff(eta)
S4_eta = N4.diff(eta)

S1_x = j11*S1_xi + j12*S1_eta
S2_x = j11*S2_xi + j12*S2_eta
S3_x = j11*S3_xi + j12*S3_eta
S4_x = j11*S4_xi + j12*S4_eta

S1_xxi = S1_x.diff(xi)
S1_xeta = S1_x.diff(eta)
S2_xxi = S2_x.diff(xi)
S2_xeta = S2_x.diff(eta)
S3_xxi = S3_x.diff(xi)
S3_xeta = S3_x.diff(eta)
S4_xxi = S4_x.diff(xi)
S4_xeta = S4_x.diff(eta)

S1_xy = j21*S1_xxi + j22*S1_xeta
S2_xy = j21*S2_xxi + j22*S2_xeta
S3_xy = j21*S3_xxi + j22*S3_xeta
S4_xy = j21*S4_xxi + j22*S4_xeta

S1_y = j21*S1_xi + j22*S1_eta
S2_y = j21*S2_xi + j22*S2_eta
S3_y = j21*S3_xi + j22*S3_eta
S4_y = j21*S4_xi + j22*S4_eta

S1_yxi = S1_y.diff(xi)
S1_yeta = S1_y.diff(eta)
S2_yxi = S2_y.diff(xi)
S2_yeta = S2_y.diff(eta)
S3_yxi = S3_y.diff(xi)
S3_yeta = S3_y.diff(eta)
S4_yxi = S4_y.diff(xi)
S4_yeta = S4_y.diff(eta)

S1_yx = j11*S1_yxi + j12*S1_yeta
S2_yx = j11*S2_yxi + j12*S2_yeta
S3_yx = j11*S3_yxi + j12*S3_yeta
S4_yx = j11*S4_yxi + j12*S4_yeta

print('N1 =', N1.simplify())
print('N2 =', N2.simplify())
print('N3 =', N3.simplify())
print('N4 =', N4.simplify())

print('S1_x =', S1_x.simplify())
print('S2_x =', S2_x.simplify())
print('S3_x =', S3_x.simplify())
print('S4_x =', S4_x.simplify())

print('S1_y =', S1_y.simplify())
print('S2_y =', S2_y.simplify())
print('S3_y =', S3_y.simplify())
print('S4_y =', S4_y.simplify())
print('')
print('S1_xy =', S1_xy.simplify())
print('S2_xy =', S2_xy.simplify())
print('S3_xy =', S3_xy.simplify())
print('S4_xy =', S4_xy.simplify())
print('')
print('Si_yx only for checking purposes')
print('')
print('S1_yx =', S1_yx.simplify())
print('S2_yx =', S2_yx.simplify())
print('S3_yx =', S3_yx.simplify())
print('S4_yx =', S4_yx.simplify())
print('')
print('')
print('xi=0, eta=0')
print('S1_yx =', S1_yx.subs(dict(xi=0, eta=0)).simplify())
print('S2_yx =', S2_yx.subs(dict(xi=0, eta=0)).simplify())
print('S3_yx =', S3_yx.subs(dict(xi=0, eta=0)).simplify())
print('S4_yx =', S4_yx.subs(dict(xi=0, eta=0)).simplify())

print('N1 =', N1.subs(dict(xi=0, eta=0)).simplify())
print('N2 =', N2.subs(dict(xi=0, eta=0)).simplify())
print('N3 =', N3.subs(dict(xi=0, eta=0)).simplify())
print('N4 =', N4.subs(dict(xi=0, eta=0)).simplify())

print('S1_x =', S1_x.subs(dict(xi=0, eta=0)).simplify())
print('S2_x =', S2_x.subs(dict(xi=0, eta=0)).simplify())
print('S3_x =', S3_x.subs(dict(xi=0, eta=0)).simplify())
print('S4_x =', S4_x.subs(dict(xi=0, eta=0)).simplify())

print('S1_y =', S1_y.subs(dict(xi=0, eta=0)).simplify())
print('S2_y =', S2_y.subs(dict(xi=0, eta=0)).simplify())
print('S3_y =', S3_y.subs(dict(xi=0, eta=0)).simplify())
print('S4_y =', S4_y.subs(dict(xi=0, eta=0)).simplify())
print('')
print('S1_xy =', S1_xy.subs(dict(xi=0, eta=0)).simplify())
print('S2_xy =', S2_xy.subs(dict(xi=0, eta=0)).simplify())
print('S3_xy =', S3_xy.subs(dict(xi=0, eta=0)).simplify())
print('S4_xy =', S4_xy.subs(dict(xi=0, eta=0)).simplify())
print('')

detJ = var('detJ')
N1, N2, N3, N4 = var('N1, N2, N3, N4')
S1_x, S2_x, S3_x, S4_x = var('S1_x, S2_x, S3_x, S4_x')
S1_y, S2_y, S3_y, S4_y = var('S1_y, S2_y, S3_y, S4_y')

#Nu =  Matrix([[N1, 0, 0, 0, 0, 0, N2, 0, 0, 0, 0, 0, N3, 0, 0, 0, 0, 0, N4, 0, 0, 0, 0, 0]])
#Nv =  Matrix([[0, N1, 0, 0, 0, 0, 0, N2, 0, 0, 0, 0, 0, N3, 0, 0, 0, 0, 0, N4, 0, 0, 0, 0]])
#Nw =  Matrix([[0, 0, N1, 0, 0, 0, 0, 0, N2, 0, 0, 0, 0, 0, N3, 0, 0, 0, 0, 0, N4, 0, 0, 0]])
#Nrx = Matrix([[0, 0, 0, N1, 0, 0, 0, 0, 0, N2, 0, 0, 0, 0, 0, N3, 0, 0, 0, 0, 0, N4, 0, 0]])
#Nry = Matrix([[0, 0, 0, 0, N1, 0, 0, 0, 0, 0, N2, 0, 0, 0, 0, 0, N3, 0, 0, 0, 0, 0, N4, 0]])
#Nrz = Matrix([[0, 0, 0, 0, 0, N1, 0, 0, 0, 0, 0, N2, 0, 0, 0, 0, 0, N3, 0, 0, 0, 0, 0, N4]])

# u v w  rx  ry  rz  (rows are node 1, node2, node3, node4)

#exx = u,x = (dxi/dx)*u,xi + (deta/dx)*u,eta = j11 u,xi + j12 u,eta
BLexx = Matrix([[S1_x, 0, 0, 0, 0, 0,
                 S2_x, 0, 0, 0, 0, 0,
                 S3_x, 0, 0, 0, 0, 0,
                 S4_x, 0, 0, 0, 0, 0]])
#eyy = v,y = (dxi/dy)*v,xi + (deta/dy)*v,eta = j21 v,xi + j22 v,eta
BLeyy = Matrix([[0, S1_y, 0, 0, 0, 0,
                 0, S2_y, 0, 0, 0, 0,
                 0, S3_y, 0, 0, 0, 0,
                 0, S4_y, 0, 0, 0, 0]])
#gxy = u,y + v,x = (dxi/dy)*u,xi + (deta/dy)*u,eta + (dxi/dx)*v,xi + (deta/dy)*v,eta
BLgxy = Matrix([[S1_y, S1_x, 0, 0, 0, 0,
                 S2_y, S2_x, 0, 0, 0, 0,
                 S3_y, S3_x, 0, 0, 0, 0,
                 S4_y, S4_x, 0, 0, 0, 0]])
#kxx = phix,x = (dxi/dx)*phix,xi + (deta/dx)*phix,eta
#kxx = ry,x = (dxi/dx)*ry,xi + (deta/dx)*ry,eta
BLkxx = Matrix([[0, 0, 0, 0, S1_x, 0,
                 0, 0, 0, 0, S2_x, 0,
                 0, 0, 0, 0, S3_x, 0,
                 0, 0, 0, 0, S4_x, 0]])
#kyy = phiy,y = (dxi/dy)*phiy,xi + (deta/dy)*phiy,eta
#kyy = -rx,y = (dxi/dy)*(-rx),xi + (deta/dy)*(-rx),eta
BLkyy = Matrix([[0, 0, 0, -S1_y, 0, 0,
                 0, 0, 0, -S2_y, 0, 0,
                 0, 0, 0, -S3_y, 0, 0,
                 0, 0, 0, -S4_y, 0, 0]])
#kxy = phix,y + phiy,x = (dxi/dy)*phix,xi + (deta/dy)*phix,eta
#                       +(dxi/dx)*phiy,xi + (deta/dx)*phiy,eta
#kxy = ry,y + (-rx),x = (dxi/dy)*ry,xi + (deta/dy)*ry,eta
#                       +(dxi/dx)*(-rx),xi + (deta/dx)*(-rx),eta
BLkxy = Matrix([[0, 0, 0, -S1_x, S1_y, 0,
                 0, 0, 0, -S2_x, S2_y, 0,
                 0, 0, 0, -S3_x, S3_y, 0,
                 0, 0, 0, -S4_x, S4_y, 0]])
#gyz = phiy + w,y
#    = -rx + w,y
BLgyz = Matrix([[0, 0, S1_y, -N1, 0, 0,
                 0, 0, S2_y, -N2, 0, 0,
                 0, 0, S3_y, -N3, 0, 0,
                 0, 0, S4_y, -N4, 0, 0]])
#gxz = phix + w,x
#    = ry + w,x
BLgxz = Matrix([[0, 0, S1_x, 0, N1, 0,
                 0, 0, S2_x, 0, N2, 0,
                 0, 0, S3_x, 0, N3, 0,
                 0, 0, S4_x, 0, N4, 0]])
# for drilling stiffness
#   see Eq. 2.20 in F.M. Adam, A.E. Mohamed, A.E. Hassaballa, Degenerated Four Nodes Shell Element with Drilling Degree of Freedom, IOSR J. Eng. 3 (2013) 10–20. www.iosrjen.org (accessed April 20, 2020).
BLdrilling = Matrix([[S1_y/2., -S1_x/2., 0, 0, 0, N1,
                      S2_y/2., -S2_x/2., 0, 0, 0, N2,
                      S3_y/2., -S3_x/2., 0, 0, 0, N3,
                      S4_y/2., -S4_x/2., 0, 0, 0, N4]])

BL = Matrix([BLexx, BLeyy, BLgxy, BLkxx, BLkyy, BLkxy, BLgyz, BLgxz])

# hourglass control as per Brockman 1987
# https://onlinelibrary.wiley.com/doi/pdf/10.1002/nme.1620241208
# Eqs. 16a and 16b
# adapted to composites replacing
#     E*h by E1eq*h, with E1eq = 1/(h*A^(-1)[0,0]) in u direction
#     E*h by E2eq*h, with E2eq = 1/(h*A^(-1)[1,1]) in v direction
#     E*h**3 by E1b*h**3 in ry direction, E1b = 12/(h**3*d[0,0])
#     E*h**3 by E2b*h**3 in rx direction, E2b = 12/(h**3*d[1,1])
#     E*h**3 by (E1b + E2b)/2*h**3 in w direction
# with [d] = (D - B A^(-1) B)^(-1) the lower-right block of [[A, B], [B, D]]^(-1),
# all in the element coordinate system. E1eq is the modulus of the
# homogeneous plate with the laminate's membrane compliance a11 = eps_xx/N_xx
# (other membrane forces zero, B neglected), and E1b is the modulus of the
# homogeneous plate with the laminate's bending compliance
# d11 = kappa_xx/M_xx (other moments and all membrane forces zero), since a
# homogeneous plate has a11 = 1/(E*h) and d11 = 12/(E*h**3). Both reduce to
# E for an isotropic plate, and d does not depend on the reference surface.
# The moduli are computed by _hourglass_bending_moduli() in quad4r.pyx
#
# and with Brockman's 1/(1 + 1/A) replaced by the dimensionally homogeneous
# area factor. The hourglass term enters the element matrix as
#
#     K_ij = A*E^(h)*gamma_i*gamma_j
#
# with gamma = d2N/dxdy at the centroid, of dimension 1/L**2. The hourglass
# amplitude of a translation is gamma'u, of dimension 1/L, and of a rotation
# is gamma'theta, of dimension 1/L**2, so equating A*E^(h)*gamma**2 to a
# force per unit displacement for the translations and to a moment per unit
# rotation for the rotations gives
#
#     [E^(h)] = F*L    for u, v and w        [E^(h)] = F*L**3   for rx, ry
#
# Hence E*h*A for the in-plane translations and E*h**3*A for the rotations.
# Brockman's 1/(1 + 1/A) tends to A for small areas and to 1 for large ones,
# so it supplies the missing area only in whatever unit makes A << 1, which
# is the source of its unit dependence. In the small-area limit the forms
# below coincide with Eqs. (16a) and (16b), so no benchmark moves.
#
# Only four of the five coefficients are changed. E^(h)_w keeps Brockman's
# factor, because the operator cannot separate the spurious pattern w = x*y
# from a legitimate twist curvature, both giving gamma'w = d2w/dxdy, so the
# stabilisation also stiffens real twist and the acceptable amount is a
# property of the problem: the thin plates want about 2e-4 of E*h**3 and the
# one-element-wide torsion strip of MacNeal and Harder wants about 2e-2. No
# dimensionless constant serves both; Brockman's factor does, because it
# grows with the element area. See the Quad4R module documentation.
var('E1eq, E2eq, E1b, E2b')
print('gamma1 = S1_xy')
print('gamma2 = S2_xy')
print('gamma3 = S3_xy')
print('gamma4 = S4_xy')
Eu = 0.10*E1eq*h*A
Ev = 0.10*E2eq*h*A
Erx = 0.10*E2b*h**3*A
Ery = 0.10*E1b*h**3*A
Ew = 0.10*(E1b + E2b)/2.*h**3/(1. + 1./A)
Erz = 0
print('Eu =', Eu)
print('Ev =', Ev)
print('Ew =', Ew)
print('Erx =', Erx)
print('Ery =', Ery)
print('Erz =', Erz)
gamma1, gamma2, gamma3, gamma4 = var('gamma1, gamma2, gamma3, gamma4')
Eu, Ev, Ew, Erx, Ery = var('Eu, Ev, Ew, Erx, Ery')
Bhourglass = Matrix([[gamma1, 0, 0, 0, 0, 0,
                      gamma2, 0, 0, 0, 0, 0,
                      gamma3, 0, 0, 0, 0, 0,
                      gamma4, 0, 0, 0, 0, 0],
                     [0, gamma1, 0, 0, 0, 0,
                      0, gamma2, 0, 0, 0, 0,
                      0, gamma3, 0, 0, 0, 0,
                      0, gamma4, 0, 0, 0, 0],
                     [0, 0, gamma1, 0, 0, 0,
                      0, 0, gamma2, 0, 0, 0,
                      0, 0, gamma3, 0, 0, 0,
                      0, 0, gamma4, 0, 0, 0],
                     [0, 0, 0, gamma1, 0, 0,
                      0, 0, 0, gamma2, 0, 0,
                      0, 0, 0, gamma3, 0, 0,
                      0, 0, 0, gamma4, 0, 0],
                     [0, 0, 0, 0, gamma1, 0,
                      0, 0, 0, 0, gamma2, 0,
                      0, 0, 0, 0, gamma3, 0,
                      0, 0, 0, 0, gamma4, 0],
                     [0, 0, 0, 0, 0, gamma1,
                      0, 0, 0, 0, 0, gamma2,
                      0, 0, 0, 0, 0, gamma3,
                      0, 0, 0, 0, 0, gamma4]])
Egamma = Matrix([[Eu, 0, 0, 0, 0, 0],
                 [0, Ev, 0, 0, 0, 0],
                 [0, 0, Ew, 0, 0, 0],
                 [0, 0, 0, Erx, 0, 0],
                 [0, 0, 0, 0, Ery, 0],
                 [0, 0, 0, 0, 0, Erz]
                 ])

ABDE = Matrix(
        [[A11, A12, A16, B11, B12, B16,   0,   0],
         [A12, A22, A26, B12, B22, B26,   0,   0],
         [A16, A26, A66, B16, B26, B66,   0,   0],
         [B11, B12, B16, D11, D12, D16,   0,   0],
         [B12, B22, B26, D12, D22, D26,   0,   0],
         [B16, B26, B66, D16, D26, D66,   0,   0],
         [  0,   0,   0,   0,   0,   0, E44, E45],
         [  0,   0,   0,   0,   0,   0, E45, E55]])

var('wij')

# Constitutive linear stiffness matrix

KC0e_full = wij*detJ*(BL.T*ABDE*BL
        + Bhourglass.T*Egamma*Bhourglass
        + K6ROT*(1e-6)*A66*BLdrilling.T*BLdrilling)

print()
print('printing cdef double entries')
print()
names = []
split_points = []
count = -1
rows = set()
rows.add(0)
nonzero = set()
for ind, val in np.ndenumerate(KC0e_full):
    if sympy.expand(val) == 0:
        continue
    i, j = ind
    if i > j:
        continue # NOTE ignoring symmetric part
    count += 1
    name = 'KC0e%02d%02d' % (i, j)
    nonzero.add(name)
    names.append(name)
    if i not in rows:
        rows.add(i)
        split_points.append(count)

to_print = np.array_split(names, split_points)
for group in to_print:
    print('cdef double %s' % (', '.join(group)))

print()

# NOTE first without drilling and reduced integration
print()
print('KC0e terms for REDUCED INTEGRATION')
print()

#subs(xi=0, eta=0) in many places above was used
KC0e_partial = wij*detJ*(BL.T*ABDE*BL
        + Bhourglass.T*Egamma*Bhourglass
        + 0*K6ROT*(1e-6)*A66*BLdrilling.T*BLdrilling)

# KC0 represents the global linear stiffness matrix
print()
print('transformation local to global')
var('r11, r12, r13, r21, r22, r23, r31, r32, r33')
Rlocal2global = Matrix([[r11, r12, r13],
                        [r21, r22, r23],
                        [r31, r32, r33]])
R = sympy.zeros(NUM_NODES*DOF, NUM_NODES*DOF)
for i in range(2*NUM_NODES):
    R[i*DOF//2:(i+1)*DOF//2, i*DOF//2:(i+1)*DOF//2] += Rlocal2global

#NOTE line below to visually check the Rmatrix
#np.savetxt('Rmatrix.txt', R, fmt='% 3s')

for ind, val in np.ndenumerate(KC0e_partial):
    if sympy.expand(val) == 0:
        continue
    i, j = ind
    if i > j:
        continue # NOTE ignoring symmetric part
    name = 'KC0e%02d%02d' % (i, j)
    print('%s = %s' % (name, simplify(val)))

# NOTE second with drilling and full integration

print()
print('KC0e terms for FULL integration, drilling only')
print()

KC0e_drilling = wij*detJ*(0*BL.T*ABDE*BL
        + 0*Bhourglass.T*Egamma*Bhourglass
        + K6ROT*(1e-6)*A66*BLdrilling.T*BLdrilling)

for ind, val in np.ndenumerate(KC0e_drilling):
    if sympy.expand(val) == 0:
        continue
    i, j = ind
    if i > j:
        continue # NOTE ignoring symmetric part
    name = 'KC0e%02d%02d' % (i, j)
    print('%s += %s' % (name, simplify(val)))

# NOTE defining symbols that will be used for KC0e

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

KC0 = R*KC0e*R.T

def name_ind(i):
    if i >= 0*DOF and i < 1*DOF:
        return 'c1'
    elif i >= 1*DOF and i < 2*DOF:
        return 'c2'
    elif i >= 2*DOF and i < 3*DOF:
        return 'c3'
    elif i >= 3*DOF and i < 4*DOF:
        return 'c4'
    else:
        raise

print()
print()
print('_______________________________________')
print()
print('printing code for sparse implementation')
print('_______________________________________')
print()
print()
KC0_SPARSE_SIZE = 0
for ind, val in np.ndenumerate(KC0):
    if sympy.expand(val) == 0:
        continue
    KC0_SPARSE_SIZE += 1
    i, j = ind
    si = name_ind(i)
    sj = name_ind(j)
    print('        k += 1')
    print('        KC0r[k] = %d+%s' % (i%DOF, si))
    print('        KC0c[k] = %d+%s' % (j%DOF, sj))
print('KC0_SPARSE_SIZE', KC0_SPARSE_SIZE)
print()
print()
for ind, val in np.ndenumerate(KC0):
    if sympy.expand(val) == 0:
        continue
    print('        k += 1')
    print('        KC0v[k] +=', val)
print()
print()
