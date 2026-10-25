"""
Geometric stiffness matrix for Quad4R element
"""
import numpy as np
import sympy
from sympy import var, symbols, Matrix, simplify

num_nodes = 4
cpu_count = 6
DOF = 6

var('wij, detJ')
var('A11, A12, A16, A22, A26, A66')
var('B11, B12, B16, B22, B26, B66')
var('D11, D12, D16, D22, D26, D66')
var('S1_x, S2_x, S3_x, S4_x')
var('S1_y, S2_y, S3_y, S4_y')

# u v w  rx  ry  rz  (rows are node 1, node2, node3, node4)

A = Matrix([
    [A11, A12, A16],
    [A12, A22, A26],
    [A16, A26, A66]])
B = Matrix([
    [B11, B12, B16],
    [B12, B22, B26],
    [B16, B26, B66]])
D = Matrix([
    [D11, D12, D16],
    [D12, D22, D26],
    [D16, D26, D66]])


detJ = var('detJ')
S1_x, S2_x, S3_x, S4_x = var('S1_x, S2_x, S3_x, S4_x')
S1_y, S2_y, S3_y, S4_y = var('S1_y, S2_y, S3_y, S4_y')

# u v w  rx  ry  rz  (node 1, node2, node3, node4)

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

# membrane
Bm = Matrix([BLexx, BLeyy, BLgxy])

# bending
Bb = Matrix([BLkxx, BLkyy, BLkxy])


print()
print()
print()

# Geometric stiffness matrix using Donnell's type of geometric nonlinearity
# (or van Karman shell nonlinear terms)
# displacements in global coordinates corresponding to one finite element
ue = Matrix([symbols(r'ue[%d]' % i) for i in range(0, Bb.shape[1])])
Nmembrane = A*Bm*ue + B*Bb*ue

print('Nxx =', simplify(Nmembrane[0, 0]))
print('Nyy =', simplify(Nmembrane[1, 0]))
print('Nxy =', simplify(Nmembrane[2, 0]))

var('Nxx, Nyy, Nxy')
# G is [[Nxx, Nxy], [Nxy, Nyy]].T (see Eq. B.1, for Donnell's equations, in https://www.sciencedirect.com/science/article/pii/S0263822314003602)
Nmatrix = Matrix([[Nxx, Nxy],
                  [Nxy, Nyy]])

#dw/dx
Sw_x = Matrix([[0, 0, S1_x, 0, 0, 0,
               0, 0, S2_x, 0, 0, 0,
               0, 0, S3_x, 0, 0, 0,
               0, 0, S4_x, 0, 0, 0]])
#dw/dy
Sw_y = Matrix([[0, 0, S1_y, 0, 0, 0,
               0, 0, S2_y, 0, 0, 0,
               0, 0, S3_y, 0, 0, 0,
               0, 0, S4_y, 0, 0, 0]])

# G is [[dw/dx, dw/dy]].T (see Eq. A.10, for Donnell's equations, in https://www.sciencedirect.com/science/article/pii/S0263822314003602)
G = Matrix([
    Sw_x,
    Sw_y
    ])

KGe = wij*detJ*G.T*Nmatrix*G

print('transformation local to global')
var('r11, r12, r13, r21, r22, r23, r31, r32, r33')
Rlocal2global = Matrix([[r11, r12, r13],
                        [r21, r22, r23],
                        [r31, r32, r33]])
R = sympy.zeros(num_nodes*DOF, num_nodes*DOF)
for i in range(2*num_nodes):
    R[i*DOF//2:(i+1)*DOF//2, i*DOF//2:(i+1)*DOF//2] += Rlocal2global

KG = R*KGe*R.T

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
KG_SPARSE_SIZE = 0
for ind, val in np.ndenumerate(KG):
    if sympy.expand(val) == 0:
        continue
    KG_SPARSE_SIZE += 1
    i, j = ind
    si = name_ind(i)
    sj = name_ind(j)
    print('        k += 1')
    print('        KGr[k] = %d+%s' % (i%DOF, si))
    print('        KGc[k] = %d+%s' % (j%DOF, sj))
print('KG_SPARSE_SIZE', KG_SPARSE_SIZE)
print()
print()
for ind, val in np.ndenumerate(KG):
    if sympy.expand(val) == 0:
        continue
    print('        k += 1')
    print('        KGv[k] +=', KG[ind])
print()
print()
