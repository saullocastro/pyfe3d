import numpy as np
import sympy
from sympy import simplify, integrate, Matrix, var
from sympy.vector import CoordSys3D, cross

r"""

   ^ y axis
   |
   |
   ______   --> x axis
   1    2

   Spring 3D beam element with 6 constant stiffnesses, defined in the element
   coordinate system

"""

DOF = 6
num_nodes = 2

var('xi', real=True)
sympy.var('kxe, kye, kze, krxe, krye, krze', real=True, positive=True)

KC0e = Matrix([
               [ kxe,   0,   0,    0,    0,    0,-kxe,   0,   0,    0,    0,     0],
               [   0, kye,   0,    0,    0,    0,   0,-kye,   0,    0,    0,     0],
               [   0,   0, kze,    0,    0,    0,   0,   0,-kze,    0,    0,     0],
               [   0,   0,   0, krxe,    0,    0,   0,   0,   0,-krxe,    0,     0],
               [   0,   0,   0,    0, krye,    0,   0,   0,   0,    0,-krye,     0],
               [   0,   0,   0,    0,    0, krze,   0,   0,   0,    0,    0, -krze],
               [-kxe,   0,   0,    0,    0,    0, kxe,   0,   0,    0,    0,     0],
               [   0,-kye,   0,    0,    0,    0,   0, kye,   0,    0,    0,     0],
               [   0,   0,-kze,    0,    0,    0,   0,   0, kze,    0,    0,     0],
               [   0,   0,   0,-krxe,    0,    0,   0,   0,   0, krxe,    0,     0],
               [   0,   0,   0,    0,-krye,    0,   0,   0,   0,    0, krye,     0],
               [   0,   0,   0,    0,    0,-krze,   0,   0,   0,    0,    0,  krze],
               ])

# KC0 represents the global linear stiffness matrix
var('r11, r12, r13, r21, r22, r23, r31, r32, r33')
R2global = Matrix([[r11, r12, r13],
                   [r21, r22, r23],
                   [r31, r32, r33]])
R = sympy.zeros(num_nodes*DOF, num_nodes*DOF)
for i in range(2*num_nodes):
    R[i*DOF//2:(i+1)*DOF//2, i*DOF//2:(i+1)*DOF//2] += R2global

KC0 = R*KC0e*R.T

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
KC0_SPARSE_SIZE = 0
for ind, val in np.ndenumerate(KC0):
    if sympy.expand(val) == 0:
        continue
    KC0_SPARSE_SIZE += 1
    i, j = ind
    si = name_ind(i)
    sj = name_ind(j)
    print('            k += 1')
    print('            KC0r[k] = %d+%s' % (i%DOF, si))
    print('            KC0c[k] = %d+%s' % (j%DOF, sj))
print('KC0_SPARSE_SIZE', KC0_SPARSE_SIZE)
print()
print()
for ind, val in np.ndenumerate(KC0):
    if sympy.expand(val) == 0:
        continue
    print('            k += 1')
    print('            KC0v[k] +=', val)
print()
print()
