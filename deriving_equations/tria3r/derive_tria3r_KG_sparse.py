"""
Geometric stiffness matrix for BFS cylinder
"""
import numpy as np
import sympy
from sympy import var, symbols, Matrix, simplify

num_nodes = 3
DOF = 6

var('wij, detJ')
var('A11, A12, A16, A22, A26, A66')
var('B11, B12, B16, B22, B26, B66')
var('D11, D12, D16, D22, D26, D66')
var('N1x, N2x, N3x')
var('N1y, N2y, N3y')

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
N1x, N2x, N3x = var('N1x, N2x, N3x')
N1y, N2y, N3y = var('N1y, N2y, N3y')

# u v w  rx  ry  rz  (node 1, node2, node3, node4)

#exx = u,x
BLexx = Matrix([[N1x, 0, 0, 0, 0, 0,
                 N2x, 0, 0, 0, 0, 0,
                 N3x, 0, 0, 0, 0, 0]])
#eyy = v,y
BLeyy = Matrix([[0, N1y, 0, 0, 0, 0,
                 0, N2y, 0, 0, 0, 0,
                 0, N3y, 0, 0, 0, 0]])
#gxy = u,y + v,x
BLgxy = Matrix([[N1y, N1x, 0, 0, 0, 0,
                 N2y, N2x, 0, 0, 0, 0,
                 N3y, N3x, 0, 0, 0, 0]])
#kxx = phix,x
#kxx = ry,x
BLkxx = Matrix([[0, 0, 0, 0, N1x, 0,
                 0, 0, 0, 0, N2x, 0,
                 0, 0, 0, 0, N3x, 0]])
#kyy = phiy,y
#kyy = -rx,y
BLkyy = Matrix([[0, 0, 0, -N1y, 0, 0,
                 0, 0, 0, -N2y, 0, 0,
                 0, 0, 0, -N3y, 0, 0]])
#kxy = phix,y + phiy,x
#kxy = ry,y + (-rx),x
BLkxy = Matrix([[0, 0, 0, -N1x, N1y, 0,
                 0, 0, 0, -N2x, N2y, 0,
                 0, 0, 0, -N3x, N3y, 0]])

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
Nwx = Matrix([[0, 0, N1x, 0, 0, 0,
               0, 0, N2x, 0, 0, 0,
               0, 0, N3x, 0, 0, 0]])
#dw/dy
Nwy = Matrix([[0, 0, N1y, 0, 0, 0,
               0, 0, N2y, 0, 0, 0,
               0, 0, N3y, 0, 0, 0]])

# G is [[dw/dx, dw/dy]].T (see Eq. A.10, for Donnell's equations, in https://www.sciencedirect.com/science/article/pii/S0263822314003602)
G = Matrix([
    Nwx,
    Nwy
    ])

KGe = wij*detJ*G.T*Nmatrix*G

# KG represents the global linear stiffness matrix
# see mapy https://github.com/saullocastro/mapy/blob/master/mapy/model/coords.py#L284
var('cosa, cosb, cosg, sina, sinb, sing')
R2local = Matrix([
           [ cosb*cosg               ,  cosb*sing ,                  -sinb ],
           [-cosa*sing+cosg*sina*sinb,  cosa*cosg+sina*sinb*sing, cosb*sina],
           [ sina*sing+cosa*cosg*sinb, -cosg*sina+cosa*sinb*sing, cosa*cosb]])
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

KG = R*KGe*R.T

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
