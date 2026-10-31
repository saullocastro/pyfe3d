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

var('x, xi', real=True)
sympy.var('hy, hz, dy, dz, L, E, Iyy, Izz, Iyz, G12, G13, GJ, A, Ay, Az', real=True, positive=True)

# definitions of Eqs. 20 and 21 of Luo, Y., 2008
xi = x/L
# NOTE in Luo 2008 Iy represents the area moment of inertia in the plane of y
#     or rotating about the z axis. Here we say that Izz = Iy
# NOTE in Luo 2008 Iz represents the area moment of inertia in the plane of z
#     or rotating about the y axis. Here we say that Iyy = Iz
Iy = Izz
Iz = Iyy
# NOTE two transverse shear moduli, as the attributes of
#      pyfe3d.beamprop.BeamProp: G12 for the shear strain exy of the plane
#      xy (v, rz) and G13 for exz of the plane xz (w, ry), and the torsional
#      stiffness GJ given directly; see the shear block of D, checked in
#      derive_beamc_KC0_sparse.py
# alphay = -12*C44/(C11*L**2), negative: see the Notes of pyfe3d.beamc.BeamC
#     (Luo 2008 Eq. (17); Eq. (21) prints the sign wrongly), from the
#     diagonal of the sectional stiffness matrix C defined below, which is
#     -12*E*Iy/(G12*A*L**2) without the attribute C of BeamProp
# alphaz = -12*C55/(C22*L**2), or -12*E*Iz/(G13*A*L**2) without C
# betay = 1/(1 - alphay)
# betaz = 1/(1 - alphaz)
var('alphay, alphaz, betay, betaz', real=True)

S1 = 1 - xi
S2 = xi
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
Nu =  Matrix([[S1, 0, 0, 0, 0, 0,
               S2, 0, 0, 0, 0, 0]])
Nv =  Matrix([[0, Hv1, 0, 0, 0, Hrz1,
               0, Hv2, 0, 0, 0, Hrz2]])
Nw =  Matrix([[0, 0, Hw1, 0, Hry1, 0,
               0, 0, Hw2, 0, Hry2, 0]])
Nrx = Matrix([[0, 0, 0, S1, 0, 0,
               0, 0, 0, S2, 0, 0]])
Nry = Matrix([[0, 0, Gw1, 0, Gry1, 0,
               0, 0, Gw2, 0, Gry2, 0]])
Nrz = Matrix([[0, Gv1, 0, 0, 0, Grz1,
               0, Gv2, 0, 0, 0, Grz2]])

Sv_x = simplify(Nv.diff(x))
Sw_x = simplify(Nw.diff(x))

# u v w  rx  ry  rz  (rows are node 1, node2, node3, node4)

# From Eqs. 8 and 9 in Luo, Y. 2008
# exx = u,x + (-rz,x)*y + (ry,x)*z
# exy = (v.diff(x) - rz) - (rx)*z
# exz = (w.diff(x) + ry) + (rx)*y
# BL = Matrix([
    # Nu.diff(x) + (-Nrz.diff(x))*y + Nry.diff(x)*z,
    #(Nv.diff(x) - Nrz) - Nrx*z,
    #(Nw.diff(x) + Nry) + Nrx*y,
    #])
# dy = dz = 0
# BL = integrate(BL, (y, -hy/2+dy, +hy/2+dy))
# BL = simplify(integrate(BL, (z, -hz/2+dz, +hz/2+dz)))

# From Eqs. 12 in Luo, Y. 2008
D11 = Matrix([
    [ E*A,  E*Ay,  E*Az, 0, 0, 0],
    [E*Ay,  E*Iy, E*Iyz, 0, 0, 0],
    [E*Az, E*Iyz,  E*Iz, 0, 0, 0],
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
# From Eq. 8 in Luo, Y. 2008
# epsilon = u,x
# kappay = -theta,x = -rz,x
# kappaz = psi,x = ry,x
# gammay = v,x - theta = v,x - rz
# gammaz = w,x + psi = w,x + ry
# kappax = phi,x
# putting in a BL matrix
BL = Matrix([
    Nu.diff(x),
    -Nrz.diff(x),
    Nry.diff(x),
    Nv.diff(x) - Nrz,
    Nw.diff(x) + Nry,
    Nrx.diff(x)])


# Geometric stiffness matrix using Donnell's type of geometric nonlinearity
# (or von Kármán nonlinear terms)
# exx = u,x + 1/2 w,x^2 + 1/2 v,x^2 - z d2w/dx2 - y d2v/dx2
# remembering that
# ry = -dw/dx
# rz = dv/dx
# then:
# exx = u,x + 1/2 ry^2 + 1/2 rz^2 + z ry,x - y rz,x , compare with Eq. 7 in Luo, Y. 2008

# displacements in global coordinates corresponding to one finite element
ue = Matrix([symbols(r'ue[%d]' % i) for i in range(0, BL.shape[1])])
Nmembrane = D*BL*ue

# NOTE the axial force N = C00*u,x + C01*gammay + C02*gammaz + C03*rx,x
#      + C04*ky + C05*kz, with ky = -rz,x and kz = ry,x, which is
#      E*(A*u,x - Ay*rz,x + Az*ry,x) without the attribute C, varies
#      linearly along x when C04 != 0 or C05 != 0, because rz,x and ry,x
#      do, the other strains being constant. With the
#      nonlinear axial strain replaced by its element mean, see
#      derive_beamc_KC0L_KCL0_KCLL_KGLL_sparse.py and Crisfield (1991)
#      vol. 1, Sec. 7.1.3, Eq. (7.31), the strain energy contains
#      int_0^L N dx*eNLm = L*Nm*eNLm, such that the geometric stiffness
#      consistent with update_fint(..., nonlinear=1) carries the element
#      mean Nm = (Na + Nb)/2 of N (own derivation, asserted in
#      derive_beamc_KC0L_KCL0_KCLL_KGLL_sparse.py). Nm is also the axial
#      force of the nodal internal forces, -finte[0] = finte[6] = Nm. When
#      C04 = C05 = 0, N is uniform and Nm = N
N = Nmembrane[0]
assert simplify(N.diff(x, 2)) == 0
Na = simplify(N.subs(x, 0))
Nb = simplify(N.subs(x, L))
Nm = simplify(integrate(N, (x, 0, L))/L)
assert simplify(Nm - (Na + Nb)/2) == 0
assert simplify(Nm.subs({C04: 0, C05: 0}) - N.subs({C04: 0, C05: 0})) == 0
Nm11 = integrate((D11*BL*ue)[0], (x, 0, L))/L
assert simplify(Nm.subs(C_FROM_11) - Nm11) == 0
# NOTE printed as the first row of C times the mean strains, using
#      betay*(1 - alphay) = betaz*(1 - alphaz) = 1
Nm_print = ((C00*(ue[6] - ue[0]) + C03*(ue[9] - ue[3]) + C04*(ue[5] - ue[11])
             + C05*(ue[10] - ue[4]))/L
            + C01*alphay*betay*((ue[1] - ue[7])/L + (ue[5] + ue[11])/2)
            + C02*alphaz*betaz*((ue[2] - ue[8])/L - (ue[4] + ue[10])/2))
assert simplify((Nm - Nm_print).subs({betay: 1/(1 - alphay), betaz: 1/(1 - alphaz)})) == 0
print('Nm =', Nm_print, flush=True)
Nm = var('Nm', real=True)
N = Nm

# NOTE v,x and w,x enter the von Kármán strain as two separate squares,
#      exx = u,x + (v,x**2 + w,x**2)/2, so the geometric stiffness is
#      N*(Sv_x.T*Sv_x + Sw_x.T*Sw_x). Using the square of the sum,
#      N*(Sv_x + Sw_x).T*(Sv_x + Sw_x), couples the two bending planes and
#      halves the buckling load of a column with a square cross section
KGe = simplify(integrate((Sv_x.T*Sv_x + Sw_x.T*Sw_x)*N, (x, 0, L)))

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
