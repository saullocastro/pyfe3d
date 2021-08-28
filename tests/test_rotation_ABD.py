import sys
sys.path.append('..')

from numpy import sin, cos
import numpy as np

from pyfe3d.shellprop_utils import laminated_plate

def test_rotation_ABD():
        # material
        E1 = 200e9
        E2 = 50e9
        nu12 = 0.3
        G12 = 8e9
        h = 1e-3

        # creating elements and populating global stiffness

        thetadeg = 30
        theta = np.deg2rad(thetadeg)

        prop_ref = laminated_plate(stack=[0], laminaprop=(E1, E2, nu12, G12, G12, G12), plyt=h)
        prop_rot = laminated_plate(stack=[thetadeg], laminaprop=(E1, E2, nu12, G12, G12, G12), plyt=h)

        r11 = np.cos(-theta)
        r12 = np.sin(-theta)
        r21 = -np.sin(-theta)
        r22 = np.cos(-theta)

        Anew = np.zeros((3, 3))
        A11 = prop_ref.A11
        A12 = prop_ref.A12
        A16 = prop_ref.A16
        A22 = prop_ref.A22
        A26 = prop_ref.A26
        A66 = prop_ref.A66
        Bnew = np.zeros((3, 3))
        B11 = prop_ref.B11
        B12 = prop_ref.B12
        B16 = prop_ref.B16
        B22 = prop_ref.B22
        B26 = prop_ref.B26
        B66 = prop_ref.B66
        Dnew = np.zeros((3, 3))
        D11 = prop_ref.D11
        D12 = prop_ref.D12
        D16 = prop_ref.D16
        D22 = prop_ref.D22
        D26 = prop_ref.D26
        D66 = prop_ref.D66

        Anew[0, 0] = r11**2*(A11*r11**2 + A12*r12**2 + 2*A16*r11*r12) + 2*r11*r12*(A16*r11**2 + A26*r12**2 + 2*A66*r11*r12) + r12**2*(A12*r11**2 + A22*r12**2 + 2*A26*r11*r12)
        Anew[0, 1] = r21**2*(A11*r11**2 + A12*r12**2 + 2*A16*r11*r12) + 2*r21*r22*(A16*r11**2 + A26*r12**2 + 2*A66*r11*r12) + r22**2*(A12*r11**2 + A22*r12**2 + 2*A26*r11*r12)
        Anew[0, 2] = r11*r21*(A11*r11**2 + A12*r12**2 + 2*A16*r11*r12) + r12*r22*(A12*r11**2 + A22*r12**2 + 2*A26*r11*r12) + (r11*r22 + r12*r21)*(A16*r11**2 + A26*r12**2 + 2*A66*r11*r12)
        Anew[1, 0] = r11**2*(A11*r21**2 + A12*r22**2 + 2*A16*r21*r22) + 2*r11*r12*(A16*r21**2 + A26*r22**2 + 2*A66*r21*r22) + r12**2*(A12*r21**2 + A22*r22**2 + 2*A26*r21*r22)
        Anew[1, 1] = r21**2*(A11*r21**2 + A12*r22**2 + 2*A16*r21*r22) + 2*r21*r22*(A16*r21**2 + A26*r22**2 + 2*A66*r21*r22) + r22**2*(A12*r21**2 + A22*r22**2 + 2*A26*r21*r22)
        Anew[1, 2] = r11*r21*(A11*r21**2 + A12*r22**2 + 2*A16*r21*r22) + r12*r22*(A12*r21**2 + A22*r22**2 + 2*A26*r21*r22) + (r11*r22 + r12*r21)*(A16*r21**2 + A26*r22**2 + 2*A66*r21*r22)
        Anew[2, 0] = r11**2*(A11*r11*r21 + A12*r12*r22 + A16*(r11*r22 + r12*r21)) + 2*r11*r12*(A16*r11*r21 + A26*r12*r22 + A66*(r11*r22 + r12*r21)) + r12**2*(A12*r11*r21 + A22*r12*r22 + A26*(r11*r22 + r12*r21))
        Anew[2, 1] = r21**2*(A11*r11*r21 + A12*r12*r22 + A16*(r11*r22 + r12*r21)) + 2*r21*r22*(A16*r11*r21 + A26*r12*r22 + A66*(r11*r22 + r12*r21)) + r22**2*(A12*r11*r21 + A22*r12*r22 + A26*(r11*r22 + r12*r21))
        Anew[2, 2] = r11*r21*(A11*r11*r21 + A12*r12*r22 + A16*(r11*r22 + r12*r21)) + r12*r22*(A12*r11*r21 + A22*r12*r22 + A26*(r11*r22 + r12*r21)) + (r11*r22 + r12*r21)*(A16*r11*r21 + A26*r12*r22 + A66*(r11*r22 + r12*r21))

        Bnew[0, 0] = r11**2*(B11*r11**2 + B12*r12**2 + 2*B16*r11*r12) + 2*r11*r12*(B16*r11**2 + B26*r12**2 + 2*B66*r11*r12) + r12**2*(B12*r11**2 + B22*r12**2 + 2*B26*r11*r12)
        Bnew[0, 1] = r21**2*(B11*r11**2 + B12*r12**2 + 2*B16*r11*r12) + 2*r21*r22*(B16*r11**2 + B26*r12**2 + 2*B66*r11*r12) + r22**2*(B12*r11**2 + B22*r12**2 + 2*B26*r11*r12)
        Bnew[0, 2] = r11*r21*(B11*r11**2 + B12*r12**2 + 2*B16*r11*r12) + r12*r22*(B12*r11**2 + B22*r12**2 + 2*B26*r11*r12) + (r11*r22 + r12*r21)*(B16*r11**2 + B26*r12**2 + 2*B66*r11*r12)
        Bnew[1, 0] = r11**2*(B11*r21**2 + B12*r22**2 + 2*B16*r21*r22) + 2*r11*r12*(B16*r21**2 + B26*r22**2 + 2*B66*r21*r22) + r12**2*(B12*r21**2 + B22*r22**2 + 2*B26*r21*r22)
        Bnew[1, 1] = r21**2*(B11*r21**2 + B12*r22**2 + 2*B16*r21*r22) + 2*r21*r22*(B16*r21**2 + B26*r22**2 + 2*B66*r21*r22) + r22**2*(B12*r21**2 + B22*r22**2 + 2*B26*r21*r22)
        Bnew[1, 2] = r11*r21*(B11*r21**2 + B12*r22**2 + 2*B16*r21*r22) + r12*r22*(B12*r21**2 + B22*r22**2 + 2*B26*r21*r22) + (r11*r22 + r12*r21)*(B16*r21**2 + B26*r22**2 + 2*B66*r21*r22)
        Bnew[2, 0] = r11**2*(B11*r11*r21 + B12*r12*r22 + B16*(r11*r22 + r12*r21)) + 2*r11*r12*(B16*r11*r21 + B26*r12*r22 + B66*(r11*r22 + r12*r21)) + r12**2*(B12*r11*r21 + B22*r12*r22 + B26*(r11*r22 + r12*r21))
        Bnew[2, 1] = r21**2*(B11*r11*r21 + B12*r12*r22 + B16*(r11*r22 + r12*r21)) + 2*r21*r22*(B16*r11*r21 + B26*r12*r22 + B66*(r11*r22 + r12*r21)) + r22**2*(B12*r11*r21 + B22*r12*r22 + B26*(r11*r22 + r12*r21))
        Bnew[2, 2] = r11*r21*(B11*r11*r21 + B12*r12*r22 + B16*(r11*r22 + r12*r21)) + r12*r22*(B12*r11*r21 + B22*r12*r22 + B26*(r11*r22 + r12*r21)) + (r11*r22 + r12*r21)*(B16*r11*r21 + B26*r12*r22 + B66*(r11*r22 + r12*r21))

        Dnew[0, 0] = r11**2*(D11*r11**2 + D12*r12**2 + 2*D16*r11*r12) + 2*r11*r12*(D16*r11**2 + D26*r12**2 + 2*D66*r11*r12) + r12**2*(D12*r11**2 + D22*r12**2 + 2*D26*r11*r12)
        Dnew[0, 1] = r21**2*(D11*r11**2 + D12*r12**2 + 2*D16*r11*r12) + 2*r21*r22*(D16*r11**2 + D26*r12**2 + 2*D66*r11*r12) + r22**2*(D12*r11**2 + D22*r12**2 + 2*D26*r11*r12)
        Dnew[0, 2] = r11*r21*(D11*r11**2 + D12*r12**2 + 2*D16*r11*r12) + r12*r22*(D12*r11**2 + D22*r12**2 + 2*D26*r11*r12) + (r11*r22 + r12*r21)*(D16*r11**2 + D26*r12**2 + 2*D66*r11*r12)
        Dnew[1, 0] = r11**2*(D11*r21**2 + D12*r22**2 + 2*D16*r21*r22) + 2*r11*r12*(D16*r21**2 + D26*r22**2 + 2*D66*r21*r22) + r12**2*(D12*r21**2 + D22*r22**2 + 2*D26*r21*r22)
        Dnew[1, 1] = r21**2*(D11*r21**2 + D12*r22**2 + 2*D16*r21*r22) + 2*r21*r22*(D16*r21**2 + D26*r22**2 + 2*D66*r21*r22) + r22**2*(D12*r21**2 + D22*r22**2 + 2*D26*r21*r22)
        Dnew[1, 2] = r11*r21*(D11*r21**2 + D12*r22**2 + 2*D16*r21*r22) + r12*r22*(D12*r21**2 + D22*r22**2 + 2*D26*r21*r22) + (r11*r22 + r12*r21)*(D16*r21**2 + D26*r22**2 + 2*D66*r21*r22)
        Dnew[2, 0] = r11**2*(D11*r11*r21 + D12*r12*r22 + D16*(r11*r22 + r12*r21)) + 2*r11*r12*(D16*r11*r21 + D26*r12*r22 + D66*(r11*r22 + r12*r21)) + r12**2*(D12*r11*r21 + D22*r12*r22 + D26*(r11*r22 + r12*r21))
        Dnew[2, 1] = r21**2*(D11*r11*r21 + D12*r12*r22 + D16*(r11*r22 + r12*r21)) + 2*r21*r22*(D16*r11*r21 + D26*r12*r22 + D66*(r11*r22 + r12*r21)) + r22**2*(D12*r11*r21 + D22*r12*r22 + D26*(r11*r22 + r12*r21))
        Dnew[2, 2] = r11*r21*(D11*r11*r21 + D12*r12*r22 + D16*(r11*r22 + r12*r21)) + r12*r22*(D12*r11*r21 + D22*r12*r22 + D26*(r11*r22 + r12*r21)) + (r11*r22 + r12*r21)*(D16*r11*r21 + D26*r12*r22 + D66*(r11*r22 + r12*r21))

        print('A reference')
        print(prop_rot.A)
        print('A calculated')
        print(Anew)
        print()
        assert np.allclose(prop_rot.A, Anew)
        assert np.allclose(prop_rot.B, Bnew)
        assert np.allclose(prop_rot.D, Dnew)

if __name__ == "__main__":
    test_rotation_ABD()
