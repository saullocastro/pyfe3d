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

var('x, xi, eta', real=True)
var('L, E, Iyy, Izz, Iyz, J, G, A, Ay, Az', real=True, positive=True)

# definitions of Eqs. 20 and 21 of Luo, Y., 2008
# xi = x/L
# NOTE in Luo 2008 Iy represents the area moment of inertia in the plane of y
#     or rotating about the z axis. Here we say that Izz = Iy
# NOTE in Luo 2008 Iz represents the area moment of inertia in the plane of z
#     or rotating about the y axis. Here we say that Iyy = Iz
Iy = Izz
Iz = Iyy
# TODO replace G by G12 and G13, but how to do for the D matrix?
# alphay = 12*E*Iy/(G*A*L**2)
# alphaz = 12*E*Iz/(G*A*L**2)
# betay = 1/(1 - alphay)
# betaz = 1/(1 - alphaz)
var('alphay, alphaz, betay, betaz', real=True)

xi = (eta + 1)/2
N1 = 1 - xi
N2 = xi
Hv1 = betay*(2*xi**3 - 3*xi**2 + alphay*xi + 1 - alphay)
Hv2 = betay*(-2*xi**3 + 3*xi**2 - alphay*xi)
Hw1 = betaz*(2*xi**3 - 3*xi**2 + alphaz*xi + 1 - alphaz)
Hw2 = betaz*(-2*xi**3 + 3*xi**2 - alphaz*xi)
Hrz1 = Htheta1 = L*betay*(xi**3 + (alphay/2 - 2)*xi**2 + (1 - alphay/2)*xi)
Hrz2 = Htheta2 = L*betay*(xi**3 - (1 + alphay/2)*xi**2 + (alphay/2)*xi)
Hry1 = Hpsi1   = -L*betaz*(xi**3 + (alphaz/2 - 2)*xi**2 + (1 - alphaz/2)*xi)
Hry2 = Hpsi2   = -L*betaz*(xi**3 - (1 + alphaz/2)*xi**2 + (alphaz/2)*xi)
Gv1 = 6*betay/L*(xi**2 - xi)
Gv2 = 6*betay/L*(-xi**2 + xi)
Gw1 = -6*betaz/L*(xi**2 - xi)
Gw2 = -6*betaz/L*(-xi**2 + xi)
Grz1 = Gtheta1 = betay*(3*xi**2 + (alphay - 4)*xi + 1 - alphay)
Grz2 = Gtheta2 = betay*(3*xi**2 - (alphay + 2)*xi)
Gry1 = Gpsi1 = betaz*(3*xi**2 + (alphaz - 4)*xi + 1 - alphaz)
Gry2 = Gpsi2 = betaz*(3*xi**2 - (alphaz + 2)*xi)

# Degrees-of-freedom illustrated in Fig. 1 of Luo, Y., 2008
#              u, v, w, phi, psi, theta (for each node)
#              u, v, w, rx, ry, rz
# interpolation according to Eq. 19 of Luo, Y. 2008
Nu =  Matrix([[N1, 0, 0, 0, 0, 0,
               N2, 0, 0, 0, 0, 0]])
Nv =  Matrix([[0, Hv1, 0, 0, 0, Hrz1,
               0, Hv2, 0, 0, 0, Hrz2]])
Nw =  Matrix([[0, 0, Hw1, 0, Hry1, 0,
               0, 0, Hw2, 0, Hry2, 0]])
Nrx = Matrix([[0, 0, 0, N1, 0, 0,
               0, 0, 0, N2, 0, 0]])
Nry = Matrix([[0, 0, Gw1, 0, Gry1, 0,
               0, 0, Gw2, 0, Gry2, 0]])
Nrz = Matrix([[0, Gv1, 0, 0, 0, Grz1,
               0, Gv2, 0, 0, 0, Grz2]])

var('intrho, intrhoy, intrhoz, intrhoy2, intrhoz2, intrhoyz')

# NOTE while performing any of the integrations below,
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
Me_cons = L/2*integrate(Me, (eta, -1, 1))
print('finished integrating Me', flush=True)
Me_cons = simplify(Me_cons)
print('finished simplifying Me_cos', flush=True)

# Lumped mass matrix using Lobatto integration, where integration points are placed at the nodes
# Brockman 1987 as reference, and A. Ralston, A First Course in Nurnericul Analysis, McGraw-Hill. Ncw York, 196
# Forcing intrhoz=intrhoy=intrhoxy=0 to end up with a diagonal matrix
# NOTE two-point Gauss-Lobatto quadrature
wi = 1
# points[0] = -1., here it would be x=0
# points[1] = +1., here it would be x=L
Me_lump = L/2*(
        + wi*Me.expand().subs({eta: -1, intrhoy: 0, intrhoz: 0, intrhoyz: 0})
        + wi*Me.expand().subs({eta: +1, intrhoy: 0, intrhoz: 0, intrhoyz: 0})
            )
print('finished calculating Me_lump', flush=True)
Me_lump = simplify(Me_lump)
print('finished simplifying Me_lump', flush=True)

print('transformation local to global')
var('r11, r12, r13, r21, r22, r23, r31, r32, r33')
Rlocal2global = Matrix([[r11, r12, r13],
                        [r21, r22, r23],
                        [r31, r32, r33]])
R = sympy.zeros(num_nodes*DOF, num_nodes*DOF)
for i in range(2*num_nodes):
    R[i*DOF//2:(i+1)*DOF//2, i*DOF//2:(i+1)*DOF//2] += Rlocal2global

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

