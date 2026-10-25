"""
Geometric stiffness matrix for BFS cylinder
"""
import numpy as np
import sympy
from sympy import var, symbols, Matrix, simplify

num_nodes = 4
cpu_count = 6
DOF = 6

detJ = var('detJ')
var('wij, r11, r12')

# u v w  rx  ry  rz  (rows are node 1, node2, node3, node4)
N1, N2, N3, N4 = var('N1, N2, N3, N4')
S1_x, S2_x, S3_x, S4_x = var('S1_x, S2_x, S3_x, S4_x')
S1_y, S2_y, S3_y, S4_y = var('S1_y, S2_y, S3_y, S4_y')

# u v w  rx  ry  rz  (node 1, node2, node3, node4)

#w
Nw = Matrix([[0, 0, N1, 0, 0, 0,
              0, 0, N2, 0, 0, 0,
              0, 0, N3, 0, 0, 0,
              0, 0, N4, 0, 0, 0]])
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

Sw_X = Sw_x * r11 + Sw_y * r12 # NOTE assuming airflow parallel to global X;
# dx/dX = r11 and dy/dX = r12, since Rlocal2global has the element axes as columns

KAe_beta = -wij*detJ*(Nw.T*Sw_X)
KAe_gamma = wij*detJ*(Nw.T*Nw)
CAe = -wij*detJ*(Nw.T*Nw)

# KA_beta represents the global aerodynamic matrix using the piston theory
var('r11, r12, r13, r21, r22, r23, r31, r32, r33')
Rlocal2global = Matrix([[r11, r12, r13],
                        [r21, r22, r23],
                        [r31, r32, r33]])
R = sympy.zeros(num_nodes*DOF, num_nodes*DOF)
for i in range(2*num_nodes):
    R[i*DOF//2:(i+1)*DOF//2, i*DOF//2:(i+1)*DOF//2] += Rlocal2global

KA_beta = R*KAe_beta*R.T
KA_gamma = R*KAe_gamma*R.T
CA = R*CAe*R.T

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
KA_BETA_SPARSE_SIZE = 0
for ind, val in np.ndenumerate(KA_beta):
    if sympy.expand(val) == 0:
        continue
    KA_BETA_SPARSE_SIZE += 1
    i, j = ind
    si = name_ind(i)
    sj = name_ind(j)
    print('        k += 1')
    print('        KA_betar[k] = %d+%s' % (i%DOF, si))
    print('        KA_betac[k] = %d+%s' % (j%DOF, sj))
print('KA_BETA_SPARSE_SIZE', KA_BETA_SPARSE_SIZE)
print()
print()
for ind, val in np.ndenumerate(KA_beta):
    if sympy.expand(val) == 0:
        continue
    print('        k += 1')
    print('        KA_betav[k] +=', KA_beta[ind])
print()
print()
KA_GAMMA_SPARSE_SIZE = 0
for ind, val in np.ndenumerate(KA_gamma):
    if sympy.expand(val) == 0:
        continue
    KA_GAMMA_SPARSE_SIZE += 1
    i, j = ind
    si = name_ind(i)
    sj = name_ind(j)
    print('        k += 1')
    print('        KA_gammar[k] = %d+%s' % (i%DOF, si))
    print('        KA_gammac[k] = %d+%s' % (j%DOF, sj))
print('KA_GAMMA_SPARSE_SIZE', KA_GAMMA_SPARSE_SIZE)
print()
print()
for ind, val in np.ndenumerate(KA_beta):
    if sympy.expand(val) == 0:
        continue
    print('        k += 1')
    print('        KA_gammav[k] +=', KA_gamma[ind])
print()
print()
CA_SPARSE_SIZE = 0
for ind, val in np.ndenumerate(CA):
    if sympy.expand(val) == 0:
        continue
    CA_SPARSE_SIZE += 1
    i, j = ind
    si = name_ind(i)
    sj = name_ind(j)
    print('        k += 1')
    print('        CAr[k] = %d+%s' % (i%DOF, si))
    print('        CAc[k] = %d+%s' % (j%DOF, sj))
print('CA_SPARSE_SIZE', CA_SPARSE_SIZE)
print()
print()
for ind, val in np.ndenumerate(CA):
    if sympy.expand(val) == 0:
        continue
    print('        k += 1')
    print('        CAv[k] +=', CA[ind])
print()
print()
