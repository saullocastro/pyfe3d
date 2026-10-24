import numpy as np
import sympy
from sympy import simplify, Matrix, var, symbols

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
NUM_NODES = 2

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
Rlocal2global = Matrix([[r11, r12, r13],
                        [r21, r22, r23],
                        [r31, r32, r33]])
R = sympy.zeros(NUM_NODES*DOF, NUM_NODES*DOF)
for i in range(2*NUM_NODES):
    R[i*DOF//2:(i+1)*DOF//2, i*DOF//2:(i+1)*DOF//2] += Rlocal2global

KC0 = R*KC0e*R.T

# displacements in global coordinates corresponding to one finite element
ue = Matrix([symbols(r'ue[%d]' % i) for i in range(0, KC0e.shape[1])])

finte = KC0e*ue

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
