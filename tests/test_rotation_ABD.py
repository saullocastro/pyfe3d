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

        m11 = np.cos(-theta)
        m12 = np.sin(-theta)
        m21 = -np.sin(-theta)
        m22 = np.cos(-theta)

        Arot = np.zeros((3, 3))
        A11 = prop_ref.A11
        A12 = prop_ref.A12
        A16 = prop_ref.A16
        A22 = prop_ref.A22
        A26 = prop_ref.A26
        A66 = prop_ref.A66
        Brot = np.zeros((3, 3))
        B11 = prop_ref.B11
        B12 = prop_ref.B12
        B16 = prop_ref.B16
        B22 = prop_ref.B22
        B26 = prop_ref.B26
        B66 = prop_ref.B66
        Drot = np.zeros((3, 3))
        D11 = prop_ref.D11
        D12 = prop_ref.D12
        D16 = prop_ref.D16
        D22 = prop_ref.D22
        D26 = prop_ref.D26
        D66 = prop_ref.D66

        Arot[0, 0] = m11**2*(A11*m11**2 + A12*m12**2 + 2*A16*m11*m12) + 2*m11*m12*(A16*m11**2 + A26*m12**2 + 2*A66*m11*m12) + m12**2*(A12*m11**2 + A22*m12**2 + 2*A26*m11*m12)
        Arot[0, 1] = m21**2*(A11*m11**2 + A12*m12**2 + 2*A16*m11*m12) + 2*m21*m22*(A16*m11**2 + A26*m12**2 + 2*A66*m11*m12) + m22**2*(A12*m11**2 + A22*m12**2 + 2*A26*m11*m12)
        Arot[0, 2] = m11*m21*(A11*m11**2 + A12*m12**2 + 2*A16*m11*m12) + m12*m22*(A12*m11**2 + A22*m12**2 + 2*A26*m11*m12) + (m11*m22 + m12*m21)*(A16*m11**2 + A26*m12**2 + 2*A66*m11*m12)
        Arot[1, 0] = m11**2*(A11*m21**2 + A12*m22**2 + 2*A16*m21*m22) + 2*m11*m12*(A16*m21**2 + A26*m22**2 + 2*A66*m21*m22) + m12**2*(A12*m21**2 + A22*m22**2 + 2*A26*m21*m22)
        Arot[1, 1] = m21**2*(A11*m21**2 + A12*m22**2 + 2*A16*m21*m22) + 2*m21*m22*(A16*m21**2 + A26*m22**2 + 2*A66*m21*m22) + m22**2*(A12*m21**2 + A22*m22**2 + 2*A26*m21*m22)
        Arot[1, 2] = m11*m21*(A11*m21**2 + A12*m22**2 + 2*A16*m21*m22) + m12*m22*(A12*m21**2 + A22*m22**2 + 2*A26*m21*m22) + (m11*m22 + m12*m21)*(A16*m21**2 + A26*m22**2 + 2*A66*m21*m22)
        Arot[2, 0] = m11**2*(A11*m11*m21 + A12*m12*m22 + A16*(m11*m22 + m12*m21)) + 2*m11*m12*(A16*m11*m21 + A26*m12*m22 + A66*(m11*m22 + m12*m21)) + m12**2*(A12*m11*m21 + A22*m12*m22 + A26*(m11*m22 + m12*m21))
        Arot[2, 1] = m21**2*(A11*m11*m21 + A12*m12*m22 + A16*(m11*m22 + m12*m21)) + 2*m21*m22*(A16*m11*m21 + A26*m12*m22 + A66*(m11*m22 + m12*m21)) + m22**2*(A12*m11*m21 + A22*m12*m22 + A26*(m11*m22 + m12*m21))
        Arot[2, 2] = m11*m21*(A11*m11*m21 + A12*m12*m22 + A16*(m11*m22 + m12*m21)) + m12*m22*(A12*m11*m21 + A22*m12*m22 + A26*(m11*m22 + m12*m21)) + (m11*m22 + m12*m21)*(A16*m11*m21 + A26*m12*m22 + A66*(m11*m22 + m12*m21))

        Brot[0, 0] = m11**2*(B11*m11**2 + B12*m12**2 + 2*B16*m11*m12) + 2*m11*m12*(B16*m11**2 + B26*m12**2 + 2*B66*m11*m12) + m12**2*(B12*m11**2 + B22*m12**2 + 2*B26*m11*m12)
        Brot[0, 1] = m21**2*(B11*m11**2 + B12*m12**2 + 2*B16*m11*m12) + 2*m21*m22*(B16*m11**2 + B26*m12**2 + 2*B66*m11*m12) + m22**2*(B12*m11**2 + B22*m12**2 + 2*B26*m11*m12)
        Brot[0, 2] = m11*m21*(B11*m11**2 + B12*m12**2 + 2*B16*m11*m12) + m12*m22*(B12*m11**2 + B22*m12**2 + 2*B26*m11*m12) + (m11*m22 + m12*m21)*(B16*m11**2 + B26*m12**2 + 2*B66*m11*m12)
        Brot[1, 0] = m11**2*(B11*m21**2 + B12*m22**2 + 2*B16*m21*m22) + 2*m11*m12*(B16*m21**2 + B26*m22**2 + 2*B66*m21*m22) + m12**2*(B12*m21**2 + B22*m22**2 + 2*B26*m21*m22)
        Brot[1, 1] = m21**2*(B11*m21**2 + B12*m22**2 + 2*B16*m21*m22) + 2*m21*m22*(B16*m21**2 + B26*m22**2 + 2*B66*m21*m22) + m22**2*(B12*m21**2 + B22*m22**2 + 2*B26*m21*m22)
        Brot[1, 2] = m11*m21*(B11*m21**2 + B12*m22**2 + 2*B16*m21*m22) + m12*m22*(B12*m21**2 + B22*m22**2 + 2*B26*m21*m22) + (m11*m22 + m12*m21)*(B16*m21**2 + B26*m22**2 + 2*B66*m21*m22)
        Brot[2, 0] = m11**2*(B11*m11*m21 + B12*m12*m22 + B16*(m11*m22 + m12*m21)) + 2*m11*m12*(B16*m11*m21 + B26*m12*m22 + B66*(m11*m22 + m12*m21)) + m12**2*(B12*m11*m21 + B22*m12*m22 + B26*(m11*m22 + m12*m21))
        Brot[2, 1] = m21**2*(B11*m11*m21 + B12*m12*m22 + B16*(m11*m22 + m12*m21)) + 2*m21*m22*(B16*m11*m21 + B26*m12*m22 + B66*(m11*m22 + m12*m21)) + m22**2*(B12*m11*m21 + B22*m12*m22 + B26*(m11*m22 + m12*m21))
        Brot[2, 2] = m11*m21*(B11*m11*m21 + B12*m12*m22 + B16*(m11*m22 + m12*m21)) + m12*m22*(B12*m11*m21 + B22*m12*m22 + B26*(m11*m22 + m12*m21)) + (m11*m22 + m12*m21)*(B16*m11*m21 + B26*m12*m22 + B66*(m11*m22 + m12*m21))

        Drot[0, 0] = m11**2*(D11*m11**2 + D12*m12**2 + 2*D16*m11*m12) + 2*m11*m12*(D16*m11**2 + D26*m12**2 + 2*D66*m11*m12) + m12**2*(D12*m11**2 + D22*m12**2 + 2*D26*m11*m12)
        Drot[0, 1] = m21**2*(D11*m11**2 + D12*m12**2 + 2*D16*m11*m12) + 2*m21*m22*(D16*m11**2 + D26*m12**2 + 2*D66*m11*m12) + m22**2*(D12*m11**2 + D22*m12**2 + 2*D26*m11*m12)
        Drot[0, 2] = m11*m21*(D11*m11**2 + D12*m12**2 + 2*D16*m11*m12) + m12*m22*(D12*m11**2 + D22*m12**2 + 2*D26*m11*m12) + (m11*m22 + m12*m21)*(D16*m11**2 + D26*m12**2 + 2*D66*m11*m12)
        Drot[1, 0] = m11**2*(D11*m21**2 + D12*m22**2 + 2*D16*m21*m22) + 2*m11*m12*(D16*m21**2 + D26*m22**2 + 2*D66*m21*m22) + m12**2*(D12*m21**2 + D22*m22**2 + 2*D26*m21*m22)
        Drot[1, 1] = m21**2*(D11*m21**2 + D12*m22**2 + 2*D16*m21*m22) + 2*m21*m22*(D16*m21**2 + D26*m22**2 + 2*D66*m21*m22) + m22**2*(D12*m21**2 + D22*m22**2 + 2*D26*m21*m22)
        Drot[1, 2] = m11*m21*(D11*m21**2 + D12*m22**2 + 2*D16*m21*m22) + m12*m22*(D12*m21**2 + D22*m22**2 + 2*D26*m21*m22) + (m11*m22 + m12*m21)*(D16*m21**2 + D26*m22**2 + 2*D66*m21*m22)
        Drot[2, 0] = m11**2*(D11*m11*m21 + D12*m12*m22 + D16*(m11*m22 + m12*m21)) + 2*m11*m12*(D16*m11*m21 + D26*m12*m22 + D66*(m11*m22 + m12*m21)) + m12**2*(D12*m11*m21 + D22*m12*m22 + D26*(m11*m22 + m12*m21))
        Drot[2, 1] = m21**2*(D11*m11*m21 + D12*m12*m22 + D16*(m11*m22 + m12*m21)) + 2*m21*m22*(D16*m11*m21 + D26*m12*m22 + D66*(m11*m22 + m12*m21)) + m22**2*(D12*m11*m21 + D22*m12*m22 + D26*(m11*m22 + m12*m21))
        Drot[2, 2] = m11*m21*(D11*m11*m21 + D12*m12*m22 + D16*(m11*m22 + m12*m21)) + m12*m22*(D12*m11*m21 + D22*m12*m22 + D26*(m11*m22 + m12*m21)) + (m11*m22 + m12*m21)*(D16*m11*m21 + D26*m12*m22 + D66*(m11*m22 + m12*m21))

        print('A reference')
        print(prop_rot.A)
        print('A calculated')
        print(Arot)
        print()
        assert np.allclose(prop_rot.A, Arot)
        assert np.allclose(prop_rot.B, Brot)
        assert np.allclose(prop_rot.D, Drot)

        A11 = Arot[0, 0]
        A12 = Arot[0, 1]
        A16 = Arot[0, 2]
        A22 = Arot[1, 1]
        A26 = Arot[1, 2]
        A66 = Arot[2, 2]
        a11 = (-A22*A66 + A26**2)/(-A11*A22*A66 + A11*A26**2 + A12**2*A66 - 2*A12*A16*A26 + A16**2*A22)
        a22 = (-A11*A66 + A16**2)/(-A11*A22*A66 + A11*A26**2 + A12**2*A66 - 2*A12*A16*A26 + A16**2*A22)
        E1eqrot = 1./(h*a11)
        E2eqrot = 1./(h*a22)
        assert np.isclose(E1eqrot, prop_rot.e1)
        assert np.isclose(E2eqrot, prop_rot.e2)
        #assert np.isclose(E2eqrot, prop_rot.e2)


if __name__ == "__main__":
    test_rotation_ABD()
