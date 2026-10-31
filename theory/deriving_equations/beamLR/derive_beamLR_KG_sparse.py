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
sympy.var('hy, hz, dy, dz, L, E, Iyy, Izz, Iyz, G12, G13, GJ, A, Ay, Az', real=True, positive=True)

# definitions of Eqs. 20 and 21 of Luo, Y., 2008
#NOTE in Luo 2008 Iy represents the area moment of inertia in the plane of y
#     or rotating about the z axis. Here we say that Izz = Iy
#NOTE in Luo 2008 Iz represents the area moment of inertia in the plane of z
#     or rotating about the y axis. Here we say that Iyy = Iz
Iy = Izz
Iz = Iyy

S1 = (1-xi)/2
S2 = (1+xi)/2

# Degrees-of-freedom illustrated in Fig. 1 of Luo, Y., 2008
#              u, v, w, phi, psi, theta (for each node)
#              u, v, w, rx, ry, rz
# linear interpolation for all field variables
Nu =  Matrix([[S1, 0, 0, 0, 0, 0,
               S2, 0, 0, 0, 0, 0]])
Nv =  Matrix([[0, S1, 0, 0, 0, 0,
               0, S2, 0, 0, 0, 0]])
Nw =  Matrix([[0, 0, S1, 0, 0, 0,
               0, 0, S2, 0, 0, 0]])
Nrx = Matrix([[0, 0, 0, S1, 0, 0,
               0, 0, 0, S2, 0, 0]])
Nry = Matrix([[0, 0, 0, 0, S1, 0,
               0, 0, 0, 0, S2, 0]])
Nrz = Matrix([[0, 0, 0, 0, 0, S1,
               0, 0, 0, 0, 0, S2]])

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

#From Eqs. 11 and 12 in Luo, Y. 2008
# p = D rho
# p = [N Mz My Qy Qz Mx]
# NOTE Mz = int(sigma_xx*y) and My = int(sigma_xx*z), Eq. 11 of Luo, Y. 2008,
#      whose Eq. 12 lists My before Mz
# p = [e ky kz gammay gammaz kx]
# NOTE two transverse shear moduli, as the attributes of
#      pyfe3d.beamprop.BeamProp: G12 for gammay of the plane xy (v, rz) and
#      G13 for gammaz of the plane xz (w, ry), and the torsional stiffness
#      GJ given directly; the shear block of D is checked in
#      derive_beamLR_KC0_sparse.py
D11 = Matrix([
    [ E*A, E*Ay, E*Az, 0, 0, 0],
    [E*Ay, E*Iy,  E*Iyz, 0, 0, 0],
    [E*Az,  E*Iyz, E*Iz, 0, 0, 0],
    [   0,     0,     0,   G12*A,      0, -G12*Az],
    [   0,     0,     0,       0,  G13*A,  G13*Ay],
    [   0,     0,     0, -G12*Az, G13*Ay,      GJ]])

# NOTE sectional stiffness matrix C of pyfe3d.beamprop.BeamProp, symmetric,
#      with the 21 independent entries Cij, i <= j, in the order of the
#      strains {eps, gammay, gammaz, kx, ky, kz} of pyfe3d, see
#      pyfe3d.vabs.yu_to_pyfe3d. D is C in the order {exx, ky, kz, gammay,
#      gammaz, kx} of the rows of BL below
C = Matrix(6, 6, lambda i, j: var('C%d%d' % (min(i, j), max(i, j)), real=True))
PERM = [0, 4, 5, 1, 2, 3]
D = C.extract(PERM, PERM)
# NOTE without the attribute C, BeamProp builds C from its 11 stiffness
#      attributes, such that D becomes D11 above
C_FROM_11 = {D[i, j]: D11[i, j] for i in range(6) for j in range(i, 6)}
assert D.subs(C_FROM_11) == D11
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
# (or von Kármán nonlinear terms)
# exx = u,x + 1/2 w,x^2 + 1/2 v,x^2 - z d2w/dx2 - y d2v/dx2
# where the slopes v,x and w,x are those of the interpolations of v and w;
# with shear deformation they differ from the rotations, v,x = rz + gammay
# and w,x = -ry + gammaz

# displacements in global coordinates corresponding to one finite element
ue = Matrix([symbols(r'ue[%d]' % i) for i in range(0, BL.shape[1])])
Nmembrane = D*BL*ue

#NOTE the axial force N = C00*u,x + C01*gammay + C02*gammaz + C03*rx,x
#     + C04*ky + C05*kz is linear in xi when C01 != 0 or C02 != 0, through
#     the rotations in gammay = v,x - rz and gammaz = w,x + ry, the other
#     strains being constant for linear shape functions. The one-point
#     integration of derive_beamLR_KC0_sparse.py evaluates the strain
#     energy at the centre, xi = 0, hence the axial force of the geometric
#     stiffness is N at xi = 0, which is also its element mean, such that
#     KC0 + KCNL + KG is the exact Jacobian of fint, see
#     derive_beamLR_KC0L_KCL0_KCLL_KGLL_sparse.py
assert sympy.expand(Nmembrane[0].diff(xi, 2)) == 0
N = simplify(Nmembrane[0].subs(xi, 0))
assert sympy.expand(N - sympy.integrate(Nmembrane[0], (xi, -1, 1))/2) == 0
# NOTE printed as the first row of C times the strains at the centre
N_print = ((C00*(ue[6] - ue[0]) + C03*(ue[9] - ue[3]) + C04*(ue[5] - ue[11])
            + C05*(ue[10] - ue[4]))/L
           + C01*((ue[7] - ue[1])/L - (ue[5] + ue[11])/2)
           + C02*((ue[8] - ue[2])/L + (ue[4] + ue[10])/2))
assert sympy.expand(N - N_print) == 0
print('N =', N_print, flush=True)
#NOTE without the attribute C of BeamProp, N = E*(A*u,x - Ay*rz,x
#     + Az*ry,x), constant along x
assert sympy.expand(N.subs(C_FROM_11) - (D11*BL*ue)[0]) == 0
N = var('N', real=True)

# the slopes are taken from the interpolations of v and w
Sv_x = 2/L*Nv.diff(xi)
Sw_x = 2/L*Nw.diff(xi)
#NOTE up to pyfe3d 0.11 the slopes were taken from the rotations, v,x = rz
#     and w,x = -ry, i.e. Nrz and -Nry here, which neglects the shear
#     deformation: the buckling load of a clamped-free column converged to
#     the Euler load P_e instead of P_e/(1 + P_e/(kappa*G*A)), Eq. (2-57) of
#     Timoshenko and Gere (1961), 59 % above it for P_e/(kappa*G*A) = 0.59.
#     With Sv_x and Sw_x the matrix is that of a string, N/L*[[1, -1], [-1,
#     1]] for v and for w, and the buckling load converges to Eq. (2-57) as
#     the square of the element length; for slender columns its
#     discretization error is about twice that of the former matrix
#NOTE the slopes enter the von Kármán strain as two separate squares,
#     exx = u,x + (v,x**2 + w,x**2)/2, so the geometric stiffness is
#     N*(Sv_x.T*Sv_x + Sw_x.T*Sw_x). Using the square of the sum,
#     N*(Sv_x + Sw_x).T*(Sv_x + Sw_x), couples the two bending planes and
#     halves the buckling load of a column with a square cross section

KGe = L/2.*simplify(integrate((Sv_x.T*Sv_x + Sw_x.T*Sw_x)*N, (xi, -1, +1)))
assert simplify(KGe[1, 1] - N/L) == 0 and simplify(KGe[1, 7] + N/L) == 0

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
