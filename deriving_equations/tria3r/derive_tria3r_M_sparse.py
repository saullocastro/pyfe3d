import numpy as np
import sympy
from sympy import simplify, integrate, Matrix, var
from sympy.vector import CoordSys3D, cross

r"""

   3
   |\
   | \    positive normal in CCW
   |  \
   |___\
   1    2

"""

DOF = 6
num_nodes = 3

var('h', positive=True, real=True)
var('x1, y1, x2, y2, x3, y3', real=True, positive=True)
var('rho, xi, eta, A')
#NOTE shear correction factor should be applied to E44, E45 and E55
#     in the finite element code

ONE = sympy.Integer(1)

detJ = 2*A # it can be easily shown
N1, N2, N3 = var('N1, N2, N3')
N1x, N2x, N3x = var('N1x, N2x, N3x')
N1y, N2y, N3y = var('N1y, N2y, N3y')

Nu =  Matrix([[N1, 0, 0, 0, 0, 0, N2, 0, 0, 0, 0, 0, N3, 0, 0, 0, 0, 0]])
Nv =  Matrix([[0, N1, 0, 0, 0, 0, 0, N2, 0, 0, 0, 0, 0, N3, 0, 0, 0, 0]])
Nw =  Matrix([[0, 0, N1, 0, 0, 0, 0, 0, N2, 0, 0, 0, 0, 0, N3, 0, 0, 0]])
Nrx = Matrix([[0, 0, 0, N1, 0, 0, 0, 0, 0, N2, 0, 0, 0, 0, 0, N3, 0, 0]])
Nry = Matrix([[0, 0, 0, 0, N1, 0, 0, 0, 0, 0, N2, 0, 0, 0, 0, 0, N3, 0]])
Nrz = Matrix([[0, 0, 0, 0, 0, N1, 0, 0, 0, 0, 0, N2, 0, 0, 0, 0, 0, N3]])

var('wij, intrho, intrhoz, intrhoz2')

#NOTE reduced integration of stiffness to remove shear locking
#subs(xi=0, eta=0) in many places above was used

#NOTE intrho and intrhoz2 already consider offset
# https://saullocastro.github.io/composites/#composites.core.Laminate
#intrho     integral `\int_{-h/2+offset}^{+h/2+offset} \rho(z) dz`,
#           used in equivalent single layer finite element mass
#           matrices
#intrhoz    integral `\int_{-h/2+offset}^{+h/2+offset} \rho(z)z dz`,
#           used in equivalent single layer finite element mass
#           matrices
#intrhoz2   integral `\int_{-h/2+offset}^{+h/2+offset} \rho(z)z^2 dz`,
#           used in equivalent single layer finite element mass
#           matrices
#
# For drilling contribution in the mass matrix, see Eq. 3.34 in https://scholar.colorado.edu/concern/graduate_thesis_or_dissertations/nv935321z
alpha=0 # drilling penalty factor for mass matrix, I found that only alpha=0 works
#
# Fully integrated mass matrix according to Brockman 1987
# See Eq. 20 for kinetic energy, with R1=intrho, R2=intrhoz, R3=intrhoz2, as per Eq. 18
Me = wij*detJ*(intrho*Nu.T*Nu + intrho*Nv.T*Nv + intrho*Nw.T*Nw
    + intrhoz*Nu.T*Nry + intrhoz*Nry.T*Nu
    - intrhoz*Nv.T*Nrx - intrhoz*Nrx.T*Nv
    + intrhoz2*Nrx.T*Nrx + intrhoz2*Nry.T*Nry + intrho*Nrz.T*Nrz)

# Mass matrix by a mix of reduced integration and projection method by Brockman 1987 Eq. 21
# as suggested by Brockman 1987, only the out-of-plane and rotations rx and ry
# are projected
# whereas u, v and here rz will be calculated using single-point quadrature

# Brockman 1987, Eq. 8a
x = Matrix([[x1, x2, x3]]).T
# Brockman 1987, Eq. 8b
y = Matrix([[y1, y2, y3]]).T

# Brockman 1987, Eq. 5a
s = Matrix([[1, 1, 1]]).T

N = Matrix([[N1, N2, N3]]).T
print()
print('H matrix exactly integrated')
# Brockman 1987, Eq. 19
H = wij*detJ*N*N.T
for i in range(num_nodes):
    for j in range(num_nodes):
        print('h%d%d +=' % (i+1, j+1), H[i, j])
print()
print('verifying symmetry of H matrix')
print((H[1, 0] - H[0, 1]).expand())
print((H[2, 0] - H[0, 2]).expand())
print((H[2, 1] - H[1, 2]).expand())
print()
var('h11, h12, h13, h22, h23, h33')
H = Matrix([[h11, h12, h13],
            [h12, h22, h23],
            [h13, h23, h33]])

# Brockman 1987, Eq. 22, assumed a constant Jacobian determinant
#NOTE for trias distorted elements also have a constant Jacobian determinant
#NOTE adapted from Brockman for triangular elements
#H = A/6*Matrix([[2, 1, 1],
                #[1, 2, 1],
                #[1, 1, 2]])
# Brockman 1987, Eq. 28a-c
cxx = simplify(x.T*H*x)
cyy = simplify(y.T*H*y)
cxy = simplify(x.T*H*y)
assert cxx.shape == (1, 1)
assert cyy.shape == (1, 1)
assert cxy.shape == (1, 1)

print('cxx =', cxx[0, 0])
print('cyy =', cyy[0, 0])
print('cxy =', cxy[0, 0])
print()

cxx, cyy, cxy = var('cxx, cyy, cxy')

# Brockman 1987, Eq. 6a and 6b
#TODO trying this adaptation from quadilateral elements
#     Nx and Ny are evaluated at xi=eta=0
b1 = Matrix([[N1x, N2x, N3x]]).T
b2 = Matrix([[N1y, N2y, N3y]]).T
print('b1, b2 at center point xi=eta=0')

# Brockman 1987, Eq. 31, H matrix with 1-point reduced integration
#TODO
valH1 = 2*A/9.
print('valH1 =', valH1)

valH1 = var('valH1')
H1 = sympy.ones(3, 3)*valH1

# Consistent mass matrix assuming constant Jacobian determinant by Brockman 1987 Eq. 21 using H per Eq. 22
#NOTE distorted elements do not have a constant Jacobian determinant
Z = sympy.zeros(3, 3)
# Me_cons is originally in the order u1,u2,u3,v1,v2,v3, ... , rz1,rz2,rz3
Me_cons = Matrix([
    [ intrho*H,          Z,        Z,          Z,  intrhoz*H,        Z],
    [        Z,   intrho*H,        Z, -intrhoz*H,          Z,        Z],
    [        Z,          Z, intrho*H,          Z,          Z,        Z],
    [        Z, -intrhoz*H,        Z, intrhoz2*H,          Z,        Z],
    [intrhoz*H,          Z,        Z,          Z, intrhoz2*H,        Z],
    [        Z,          Z,        Z,          Z,          Z, alpha*intrho*H]])
# Making Me_cons in the order u1,v1,w1,rx1,ry1,rz1, ..., u3,v3,w3,rx3,ry3,rz3
tmp = np.array(Me_cons, dtype=object)
order = [0, 3, 6, 9, 12, 15,   1, 4, 7, 10, 13, 16,    2, 5, 8, 11, 14, 17]
tmp = tmp[order]
tmp = tmp[:, order]
Me_cons = Matrix(tmp)

# Mass matrix by reduced integration by Brockman 1987 Eq. 21 using H1 per Eq. 31
# Me_red is originally in the order u1,u2,u3,v1,v2,v3, ... , rz1,rz2,rz3
Me_red = Matrix([
    [ intrho*H1,           Z,         Z,           Z,  intrhoz*H1,         Z],
    [         Z,   intrho*H1,         Z, -intrhoz*H1,           Z,         Z],
    [         Z,           Z, intrho*H1,           Z,           Z,         Z],
    [         Z, -intrhoz*H1,         Z, intrhoz2*H1,           Z,         Z],
    [intrhoz*H1,           Z,         Z,           Z, intrhoz2*H1,         Z],
    [         Z,           Z,         Z,           Z,           Z, alpha*intrho*H1]])
# Making Me_red in the order u1,v1,w1,rx1,ry1,rz1, ..., u4,v4,w4,rx4,ry4,rz4
tmp = np.array(Me_red, dtype=object)
tmp = tmp[order]
tmp = tmp[:, order]
Me_red = Matrix(tmp)

# Lumped mass matrix using Lobatto integration, where integration points are placed at nodes
# Brockman 1987 as reference, and A. Ralston, A First Course in Nurnericul Analysis, McGraw-Hill. Ncw York, 196
# Forcing intrhoz=0 to end up with a diagonal matrix
# Me_lump is originally in the order u1,u2,u3,u4,v1,v2,v3,v4, ... , rz1,rz2,rz3,rz4
Me_lump = Matrix([
    [ intrho*H,          Z,        Z,          Z, 0*intrhoz*H,        Z],
    [        Z,   intrho*H,        Z, -0*intrhoz*H,          Z,        Z],
    [        Z,          Z, intrho*H,          Z,          Z,        Z],
    [        Z, -0*intrhoz*H,        Z, intrhoz2*H,          Z,        Z],
    [0*intrhoz*H,          Z,        Z,          Z, intrhoz2*H,        Z],
    [        Z,          Z,        Z,          Z,          Z, alpha*intrho*H]])
# Making Me_lump in the order u1,v1,w1,rx1,ry1,rz1, ..., u4,v4,w4,rx4,ry4,rz4
tmp = np.array(Me_lump, dtype=object)
tmp = tmp[order]
tmp = tmp[:, order]
Me_lump = Matrix(tmp)

# KC0 represents the global linear stiffness matrix
# see mapy https://github.com/saullocastro/mapy/blob/master/mapy/model/coords.py#L284
var('cosa, cosb, cosg, sina, sinb, sing')
R2local = Matrix([
           [ cosb*cosg               ,  cosb*sing ,                  -sinb ],
           [-cosa*sing+cosg*sina*sinb,  cosa*cosg+sina*sinb*sing, cosb*sina],
           [ sina*sing+cosa*cosg*sinb, -cosg*sina+cosa*sinb*sing, cosa*cosb]])
print()
print('transformation global to local')
print('s11 =', R2local[0, 0])
print('s12 =', R2local[0, 1])
print('s13 =', R2local[0, 2])
print('s21 =', R2local[1, 0])
print('s22 =', R2local[1, 1])
print('s23 =', R2local[1, 2])
print('s31 =', R2local[2, 0])
print('s32 =', R2local[2, 1])
print('s33 =', R2local[2, 2])
print()
R2global = R2local.T
print()
print('transformation local to global')
print('r11 =', R2global[0, 0])
print('r12 =', R2global[0, 1])
print('r13 =', R2global[0, 2])
print('r21 =', R2global[1, 0])
print('r22 =', R2global[1, 1])
print('r23 =', R2global[1, 2])
print('r31 =', R2global[2, 0])
print('r32 =', R2global[2, 1])
print('r33 =', R2global[2, 2])
print()
var('r11, r12, r13, r21, r22, r23, r31, r32, r33')
R2global = Matrix([[r11, r12, r13],
                   [r21, r22, r23],
                   [r31, r32, r33]])
R = sympy.zeros(num_nodes*DOF, num_nodes*DOF)
for i in range(2*num_nodes):
    R[i*DOF//2:(i+1)*DOF//2, i*DOF//2:(i+1)*DOF//2] += R2global

M = R*Me*R.T
M_cons = R*Me_cons*R.T
M_red = R*Me_red*R.T
M_lump = R*Me_lump*R.T

def name_ind(i):
    if i >= 0*DOF and i < 1*DOF:
        return 'c1'
    elif i >= 1*DOF and i < 2*DOF:
        return 'c2'
    elif i >= 2*DOF and i < 3*DOF:
        return 'c3'
    else:
        raise

print()
print()
print('_______________________________________')
print('')
print('printing code for sparse implementation')
print('_______________________________________')
print()
print('consistent mass matrix according to Saullo G. P. Castro, arbitrary integration points')
print()
print()
M_SPARSE_SIZE = 0
for ind, val in np.ndenumerate(M):
    if val == 0:
        continue
    M_SPARSE_SIZE += 1
    i, j = ind
    si = name_ind(i)
    sj = name_ind(j)
    print('            k += 1')
    print('            Mr[k] = %d+%s' % (i%DOF, si))
    print('            Mc[k] = %d+%s' % (j%DOF, sj))
print('M_SPARSE_SIZE', M_SPARSE_SIZE)
print()
print()
print()
for ind, val in np.ndenumerate(M):
    if val == 0:
        continue
    print('            k += 1')
    print('            Mv[k] +=', M[ind])
print()
print()
print()
print('consistent mass matrix according to Brockman')
print('M_cons')
print()
print()
M_SPARSE_SIZE = 0
for ind, val in np.ndenumerate(M_cons):
    if val == 0:
        continue
    M_SPARSE_SIZE += 1
    i, j = ind
    si = name_ind(i)
    sj = name_ind(j)
    print('            k += 1')
    print('            Mr[k] = %d+%s' % (i%DOF, si))
    print('            Mc[k] = %d+%s' % (j%DOF, sj))
print('M_SPARSE_SIZE', M_SPARSE_SIZE)
print()
print()
print()
for ind, val in np.ndenumerate(M_cons):
    if val == 0:
        continue
    print('            k += 1')
    print('            Mv[k] +=', M_cons[ind])
print()
print()
print()
print('fully produced by reduced integration mass matrix')
print('M_red')
print()
print()
M_SPARSE_SIZE = 0
for ind, val in np.ndenumerate(M_red):
    if val == 0:
        continue
    M_SPARSE_SIZE += 1
    i, j = ind
    si = name_ind(i)
    sj = name_ind(j)
    print('            k += 1')
    print('            Mr[k] = %d+%s' % (i%DOF, si))
    print('            Mc[k] = %d+%s' % (j%DOF, sj))
print('M_SPARSE_SIZE', M_SPARSE_SIZE)
print()
print()
print()
for ind, val in np.ndenumerate(M_red):
    if val == 0:
        continue
    print('            k += 1')
    print('            Mv[k] +=', M_red[ind])
print()
print()
print()
print('lumped mass matrix using Lobatto method, integration points at four nodes')
print('M_lump')
print()
print()
M_SPARSE_SIZE = 0
for ind, val in np.ndenumerate(M_lump):
    if val == 0:
        continue
    M_SPARSE_SIZE += 1
    i, j = ind
    si = name_ind(i)
    sj = name_ind(j)
    print('            k += 1')
    print('            Mr[k] = %d+%s' % (i%DOF, si))
    print('            Mc[k] = %d+%s' % (j%DOF, sj))
print('M_SPARSE_SIZE', M_SPARSE_SIZE)
for ind, val in np.ndenumerate(M_lump):
    if val == 0:
        continue
    print('            k += 1')
    print('            Mv[k] +=', M_lump[ind])
print()
print()
print()
