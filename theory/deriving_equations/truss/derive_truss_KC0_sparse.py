import numpy as np
import sympy
from sympy import simplify, integrate, Matrix, var

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

KC0e = BL.T*D*BL
KC0e = L/2.*simplify(integrate(KC0e, (xi, -1, +1)))

print('transformation local to global')
var('r11, r12, r13, r21, r22, r23, r31, r32, r33')
Rlocal2global = Matrix([[r11, r12, r13],
                        [r21, r22, r23],
                        [r31, r32, r33]])
R = sympy.zeros(num_nodes*DOF, num_nodes*DOF)
for i in range(2*num_nodes):
    R[i*DOF//2:(i+1)*DOF//2, i*DOF//2:(i+1)*DOF//2] += Rlocal2global

#NOTE line below to visually check the Rmatrix
#np.savetxt('Rmatrix.txt', R, fmt='% 3s')

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
