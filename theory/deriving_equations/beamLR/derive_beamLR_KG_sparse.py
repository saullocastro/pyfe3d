import numpy as np
import sympy
from sympy import simplify, integrate, Matrix, var, symbols

r"""

   ^ y axis
   |
   |
   ______   --> x axis
   1    2
   Assumed 2D plane for all derivations

   Timoshenko 3D beam element with consistent shape functions from:
   Luo, Y., 2008, “An Efficient 3D Timoshenko Beam Element with Consistent Shape Functions,” Adv. Theor. Appl. Mech., 1(3), pp. 95–106.

"""

DOF = 6
num_nodes = 2

var('xi', real=True)
sympy.var('hy, hz, dy, dz, L, E, Iyy, Izz, Iyz, J, G, A, Ay, Az', real=True, positive=True)

# definitions of Eqs. 20 and 21 of Luo, Y., 2008
#NOTE in Luo 2008 Iy represents the area moment of inertia in the plane of y
#     or rotating about the z axis. Here we say that Izz = Iy
#NOTE in Luo 2008 Iz represents the area moment of inertia in the plane of z
#     or rotating about the y axis. Here we say that Iyy = Iz
Iy = Izz
Iz = Iyy

N1 = (1-xi)/2
N2 = (1+xi)/2

# Degrees-of-freedom illustrated in Fig. 1 of Luo, Y., 2008
#              u, v, w, phi, psi, theta (for each node)
#              u, v, w, rx, ry, rz
# linear interpolation for all field variables
Nu =  Matrix([[N1, 0, 0, 0, 0, 0,
               N2, 0, 0, 0, 0, 0]])
Nv =  Matrix([[0, N1, 0, 0, 0, 0,
               0, N2, 0, 0, 0, 0]])
Nw =  Matrix([[0, 0, N1, 0, 0, 0,
               0, 0, N2, 0, 0, 0]])
Nrx = Matrix([[0, 0, 0, N1, 0, 0,
               0, 0, 0, N2, 0, 0]])
Nry = Matrix([[0, 0, 0, 0, N1, 0,
               0, 0, 0, 0, N2, 0]])
Nrz = Matrix([[0, 0, 0, 0, 0, N1,
               0, 0, 0, 0, 0, N2]])

# u v w  rx  ry  rz  (rows are node 1, node2, node3, node4)

#From Eqs. 8 and 9 in Luo, Y. 2008
#exx = u,x + (-rz,x)*y + (ry,x)*z
#exy = (v.diff(x) - rz) - (rx)*z
#exz = (w.diff(x) + ry) + (rx)*y
#BL = Matrix([
    #Nu.diff(x) + (-Nrz.diff(x))*y + Nry.diff(x)*z,
    #(Nv.diff(x) - Nrz) - Nrx*z,
    #(Nw.diff(x) + Nry) + Nrx*y,
    #])
#dy = dz = 0
#BL = integrate(BL, (y, -hy/2+dy, +hy/2+dy))
#BL = simplify(integrate(BL, (z, -hz/2+dz, +hz/2+dz)))

#From Eqs. 12 in Luo, Y. 2008
# p = D rho
# p = [N My Mz Qy Qz Mx]
# p = [e ky kz gammay gammaz kx]
D = Matrix([
    [ E*A, E*Ay, E*Az, 0, 0, 0],
    [E*Ay, E*Iy,  E*Iyz, 0, 0, 0],
    [E*Az,  E*Iyz, E*Iz, 0, 0, 0],
    [   0,    0,    0,   G*A, 0, -G*Az],
    [   0,    0,    0,  0,  G*A, G*Ay],
    [   0,    0,    0, -G*Az, G*Ay, G*J]])
#From Eq. 8 in Luo, Y. 2008
#epsilon = u,x
#kappay = -theta,x = -rz,x
#kappaz = psi,x = ry,x
#gammay = v,x - theta = v,x - rz
#gammaz = w,x + psi = w,x + ry
#kappax = phi,x
# putting in a BL matrix
BL = Matrix([
    2/L*Nu.diff(xi),
    -2/L*Nrz.diff(xi),
    2/L*Nry.diff(xi),
    2/L*Nv.diff(xi) - Nrz,
    2/L*Nw.diff(xi) + Nry,
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
assert N.diff(xi) == 0
print('N =', N, flush=True)
#NOTE N = E*(A*u,x - Ay*rz,x + Az*ry,x) is constant along x, because u,x,
#     rz,x and ry,x are constant for linear shape functions
N = var('N', real=True)

# the slopes are taken from the rotations, v,x = rz and w,x = -ry
#NOTE, I tried Nvx and Nwx here and it does not work, leading to the same KG
#      obtained for the truss element. Thus, I recommend keeping Nrz and Nry,
#      which might become more inconsistent when the beam becomes thicker, i.e.
#      when L/sqrt(A) becomes lower.
#NOTE the slopes enter the von Karman strain as two separate squares,
#     exx = u,x + (v,x**2 + w,x**2)/2, so the geometric stiffness is
#     N*(Nrz.T*Nrz + Nry.T*Nry). Using the square of the sum,
#     N*(Nrz - Nry).T*(Nrz - Nry), couples the two bending planes and halves
#     the buckling load of a column with a square cross section

KGe = L/2.*simplify(integrate((Nrz.T*Nrz + Nry.T*Nry)*N, (xi, -1, +1)))

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
