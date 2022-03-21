import numpy as np
import sympy
from sympy import simplify, integrate, Matrix, var, symbols

r"""

   ^ y axis
   |
   |
   ______   --> x axis
   1    2

   Truss 3D element with linear shape functions and analytical integration

   Adapted from the beam constitutive matrix of
   Luo, Y., 2008, “An Efficient 3D Timoshenko Beam Element with Consistent Shape Functions,” Adv. Theor. Appl. Mech., 1(3), pp. 95–106.

"""
#NOTE not making sense to have KG for the truss element, because in theory we
#     have that Izz=Iyy=0
#     I would rather not have Truss and only work with BeamLR, which has
#     physical meaning

DOF = 6
num_nodes = 2

var('xi', real=True)
sympy.var('hy, hz, dy, dz, L, E, Iyy, Izz, J, G, A', real=True, positive=True)
Iy = Izz
Iz = Iyy

N1 = (1-xi)/2
N2 = (1+xi)/2

N1x = -1/L
N2x = +1/L

# Degrees-of-freedom illustrated in Fig. 1 of Luo, Y., 2008
#              u, v, w, phi, psi, theta (for each node)
#              u, v, w, rx, ry, rz
# linear interpolation for all field variables
Nu =  Matrix([[N1, 0, 0, 0, 0, 0,
               N2, 0, 0, 0, 0, 0]])
Nrx =  Matrix([[0, 0, 0, N1, 0, 0,
                0, 0, 0, N2, 0, 0]])

Nvx =  Matrix([[0, N1x, 0, 0, 0, 0,
                0, N2x, 0, 0, 0, 0]])
Nwx =  Matrix([[0, 0, N1x, 0, 0, 0,
                0, 0, N2x, 0, 0, 0]])

# u v w  rx  ry  rz  (rows are node 1, node2, node3, node4)

#From Eqs. 12 in Luo, Y. 2008
#NOTE assuming Ay=Az=0 to have Nmembrane constant
D = Matrix([
    [ E*A,  0],
    [   0, G*J]])
#From Eq. 8 in Luo, Y. 2008, keeping only terms pertaining the truss element
#epsilon = u,x
#kappax = phi,x
# putting in a BL matrix
BL = Matrix([
    2/L*Nu.diff(xi),
    2/L*Nrx.diff(xi)])

# Geometric stiffness matrix using Donnell's type of geometric nonlinearity
# (or van Karman nonlinear terms)
# exx = u,x + 1/2 w,x^2 + 1/2 v,x^2 - z d2w/dx2 - y d2v/dx2
# remembering that
# ry = -dw/dx
# rz = dv/dx
# then:
# exx = u,x + 1/2 ry^2 + 1/2 rz^2 + z ry,x - y rz,x , compare with Eq. 7 in Luo, Y. 2008

# displacements in global coordinates corresponding to one finite element
ue = Matrix([symbols(r'ue[%d]' % i) for i in range(0, BL.shape[1])])
Nmembrane = D*BL*ue

N = simplify(Nmembrane[0])
print('N =', N, flush=True)
#NOTE for constant properties, N will be constant along x
N = var('N', real=True)

# G is dv/dx + dw/dx
Gmatrix = Nvx + Nwx

KGe = L/2.*simplify(integrate((Gmatrix.T*Gmatrix)*N, (xi, -1, +1)))

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
