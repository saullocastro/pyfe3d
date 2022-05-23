import sys
sys.path.append('..')

import numpy as np
from scipy.sparse.linalg import eigsh
from scipy.sparse import coo_matrix

from pyfe3d.beamprop import BeamProp
from pyfe3d import BeamC, BeamCData, BeamCProbe, DOF, INT, DOUBLE

def test_nat_freq_curved_beam(refinement=1, mtypes=range(2)):
    for rotation_rad in [np.pi/2, 3*np.pi/2]:
        for mtype in mtypes:
            print('mtype', mtype)
            n = 50*refinement
            # comparing with:
            # https://www.sciencedirect.com/science/article/pii/S0168874X06000916
            #NOTE 2D problem, using only Izz, ignoring any out of XZ plane displacement
            #                 ignoring torsion (rx) as well
            # see section 5.5
            E = 206.8e9 # Pa
            #nu = 0.3
            G = 77.9e9 # Pa
            rho = 7855 # kg/m3
            A = 4.071e-3
            Izz = 6.456e-6

            thetabeam = np.deg2rad(97)
            r = 2.438
            thetas = np.linspace(thetabeam, 0, n)
            x = r*np.cos(thetas)
            print('x', x)
            normal = r*np.sin(thetas)
            y = normal*np.cos(rotation_rad)
            print('y', y)
            z = normal*np.sin(rotation_rad)
            print('z', z)

            ncoords = np.vstack((x, y, z)).T
            nids = 1 + np.arange(ncoords.shape[0])
            nid_pos = dict(zip(nids, np.arange(len(nids))))

            n1s = nids[0:-1]
            n2s = nids[1:]

            num_elements = len(n1s)
            print('num_elements', num_elements)

            p = BeamCProbe()
            data = BeamCData()

            KC0r = np.zeros(data.KC0_SPARSE_SIZE*num_elements, dtype=INT)
            KC0c = np.zeros(data.KC0_SPARSE_SIZE*num_elements, dtype=INT)
            KC0v = np.zeros(data.KC0_SPARSE_SIZE*num_elements, dtype=DOUBLE)
            Mr = np.zeros(data.M_SPARSE_SIZE*num_elements, dtype=INT)
            Mc = np.zeros(data.M_SPARSE_SIZE*num_elements, dtype=INT)
            Mv = np.zeros(data.M_SPARSE_SIZE*num_elements, dtype=DOUBLE)
            N = DOF*n
            print('num_DOF', N)

            prop = BeamProp()
            prop.A = A
            prop.E = E
            scf = 5/6.
            prop.G = scf*G
            prop.Izz = Izz
            prop.intrho = rho*A
            prop.intrhoy2 = rho*Izz

            ncoords_flatten = ncoords.flatten()

            beams = []
            init_k_KC0 = 0
            init_k_M = 0
            for n1, n2 in zip(n1s, n2s):
                pos1 = nid_pos[n1]
                pos2 = nid_pos[n2]
                beam = BeamC(p)
                beam.init_k_KC0 = init_k_KC0
                beam.init_k_M = init_k_M
                beam.n1 = n1
                beam.n2 = n2
                beam.c1 = DOF*pos1
                beam.c2 = DOF*pos2
                vxy = ncoords[pos1]
                beam.update_rotation_matrix(vxy[0], vxy[1], vxy[2], ncoords_flatten)
                print(beam.r11, beam.r12, beam.r13)
                print(beam.r21, beam.r22, beam.r23)
                print(beam.r31, beam.r32, beam.r33)
                print()
                beam.update_probe_xe(ncoords_flatten)
                beam.update_KC0(KC0r, KC0c, KC0v, prop)
                beam.update_M(Mr, Mc, Mv, prop, mtype=mtype)
                beams.append(beam)
                init_k_KC0 += data.KC0_SPARSE_SIZE
                init_k_M += data.M_SPARSE_SIZE

            print('elements created')

            KC0 = coo_matrix((KC0v, (KC0r, KC0c)), shape=(N, N)).tocsc()
            M = coo_matrix((Mv, (Mr, Mc)), shape=(N, N)).tocsc()

            print('sparse KC0 and M created')

            bk = np.zeros(N, dtype=bool)
            check = np.isclose(x, x.min()) | np.isclose(x, x.max())
            bk[0::DOF] = check # u
            bk[2::DOF] = check # w
            bk[1::DOF] = True
            bk[3::DOF] = True
            bk[5::DOF] = True
            bu = ~bk

            Kuu = KC0[bu, :][:, bu]
            Muu = M[bu, :][:, bu]

            num_eigenvalues = 3
            eigvals, eigvecsu = eigsh(A=Kuu, M=Muu, sigma=-1., which='LM',
                    k=num_eigenvalues, tol=1e-4)
            omegan = eigvals**0.5
            omega123_from_paper = [396.98, 931.22, 1797.31]
            omega123_expected_here = [395.50396255, 923.75280464, 1773.32657347]
            print('Reference omega123_from_paper', omega123_from_paper)
            print('Reference omega123_expected_here', omega123_expected_here)
            print('Numerical omega123', omegan)
            print()
            assert np.allclose(omega123_expected_here, omegan, rtol=1e-3)

if __name__ == '__main__':
    test_nat_freq_curved_beam(refinement=1)
