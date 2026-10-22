import numpy as np
import sympy
from sympy import simplify, integrate, Matrix, var

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
sympy.var('hy, hz, dy, dz, L, E, Iyy, Izz, Iyz, J, Rxy, G, A, Ay, Az', real=True, positive=True)

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

var('intrho, intrhoy, intrhoz, intrhoy2, intrhoz2, intrhoyz')

#NOTE while performing any of the integrations below,
#     offsets should be considered
#   intrho     integral ``\int_{y_e} \int_{z_e} \rho(y, z) dy dz``, where ``\rho``
#              is the density
#   intrhoy    integral ``\int_y \int_z y \rho(y, z) dy dz``
#   intrhoz    integral ``\int_y \int_z z \rho(y, z) dy dz``
#   intrhoy2   integral ``\int_y \int_z y^2 \rho(y, z) dy dz``
#   intrhoz2   integral ``\int_y \int_z z^2 \rho(y, z) dy dz``
#   intrhoyz   integral ``\int_y \int_z y z \rho(y, z) dy dz``
#
# Fully integrated mass matrix using kinematics from Luo 2008
Me = (
    # contributions from u displacement, u(xe, ye, ze) = uc - rz*ye + ry*ze
    intrho*Nu.T*Nu
    - intrhoy*(Nu.T*Nrz + Nrz.T*Nu) + intrhoz*(Nu.T*Nry + Nry.T*Nu)
    - intrhoyz*(Nrz.T*Nry + Nry.T*Nrz) + intrhoy2*Nrz.T*Nrz + intrhoz2*Nry.T*Nry
    # contributions from v displacement, v(xe, ye, ze) = vc - rx*ze
    + intrho*Nv.T*Nv
    - intrhoz*(Nv.T*Nrx + Nrx.T*Nv) + intrhoz2*Nrx.T*Nrx
    # contributions from w displacement, w(xe, ye, ze) = wc + rx*ye
    + intrho*Nw.T*Nw
    + intrhoy*(Nw.T*Nrx + Nrx.T*Nw) + intrhoy2*Nrx.T*Nrx
    )
print('finished calculating Me', flush=True)
Me = simplify(Me)
print('finished simplifying Me', flush=True)
Me_cons = L/2.*integrate(Me, (xi, -1, 1))
print('finished integrating Me', flush=True)
Me_cons = simplify(Me_cons)
print('finished simplifying Me_cos', flush=True)

# Lumped mass matrix using Lobatto integration, where integration points are placed at the nodes
# Brockman 1987 as reference, and A. Ralston, A First Course in Nurnericul Analysis, McGraw-Hill. Ncw York, 196
# Forcing intrhoz=intrhoy=intrhoxy=0 to end up with a diagonal matrix
#NOTE two-point Gauss-Lobatto quadrature
wi = 1
# points[0] = -1., here it would be x=0
# points[1] = +1., here it would be x=L
Me_lump = L/2*(
        + wi*Me.expand().subs({xi: -1, intrhoy: 0, intrhoz: 0, intrhoyz: 0})
        + wi*Me.expand().subs({xi: +1, intrhoy: 0, intrhoz: 0, intrhoyz: 0})
            )
print('finished calculating Me_lump', flush=True)
Me_lump = simplify(Me_lump)
print('finished simplifying Me_lump', flush=True)

# M represents the global matrix, Me the element matrix
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

M_cons = R*Me_cons*R.T
M_lump = R*Me_lump*R.T

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
print('')
print('printing code for sparse implementation')
print('_______________________________________')
print()
print('consistent mass matrix')
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
print('lumped mass matrix using Lobatto method, integration points at 2 nodes')
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

