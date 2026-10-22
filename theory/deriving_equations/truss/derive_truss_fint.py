import numpy as np
import sympy
from sympy import simplify, Matrix, var, symbols, integrate

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
num_nodes = 2

var('xi', real=True)
sympy.var('L, E, J, G, A', real=True, positive=True)

N1 = (1-xi)/2
N2 = (1+xi)/2

# Degrees-of-freedom illustrated in Fig. 1 of Luo, Y., 2008
#              u, v, w, phi, psi, theta (for each node)
#              u, v, w, rx, ry, rz
# linear interpolation for all field variables
Nu =  Matrix([[N1, 0, 0, 0, 0, 0,
               N2, 0, 0, 0, 0, 0]])
Nrx =  Matrix([[0, 0, 0, N1, 0, 0,
                0, 0, 0, N2, 0, 0]])

# u v w  rx  ry  rz  (rows are node 1, node2, node3, node4)

#From Eqs. 12 in Luo, Y. 2008
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

# displacements in global coordinates corresponding to one finite element
ue = Matrix([symbols(r'ue[%d]' % i) for i in range(0, BL.shape[1])])

finte = BL.T*D*BL*ue
# NOTE exact integration
finte = L/2.*integrate(finte, (xi, -1, +1))

print('transformation local to global')
var('r11, r12, r13, r21, r22, r23, r31, r32, r33')
Rlocal2global = Matrix([[r11, r12, r13],
                        [r21, r22, r23],
                        [r31, r32, r33]])
R = sympy.zeros(num_nodes*DOF, num_nodes*DOF)
for i in range(2*num_nodes):
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
for i in range(num_nodes*DOF):
    name = 'finte%02d' % (i)
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
    print('            fint[%d+%s] +=' % (i%DOF, si), val)
print()
print()
