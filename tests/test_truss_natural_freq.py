import sys
sys.path.append('..')

import numpy as np
from scipy.sparse.linalg import eigsh
from scipy.sparse import coo_matrix

from pyfe3d.beamprop import BeamProp
from pyfe3d import Truss, TrussData, TrussProbe, DOF, INT, DOUBLE

def test_truss_natural_freq(refinement=1, mtypes=range(2)):
    for mtype in mtypes:
        print('mtype', mtype)
        n = 101*refinement
        L = 3

        E = 203.e9 # Pa
        rho = 7.83e3 # kg/m3

        x = np.linspace(0, L, n)
        y = np.ones_like(x)
        b = 0.05 # m
        h = 0.05 # m
        A = h*b

        ncoords = np.vstack((x, y, np.zeros_like(x))).T
        nids = 1 + np.arange(ncoords.shape[0])
        nid_pos = dict(zip(nids, np.arange(len(nids))))

        n1s = nids[0:-1]
        n2s = nids[1:]

        num_elements = len(n1s)
        print('num_elements', num_elements)

        p = TrussProbe()
        data = TrussData()

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
        prop.G = scf*E/2/(1+0.3)
        prop.intrho = rho*A
        prop.intrhoy2 = 0 # used to calculate torsional constant
        prop.intrhoz2 = 0 # used to calculate torsional constant

        ncoords_flatten = ncoords.flatten()

        trusses = []
        init_k_KC0 = 0
        init_k_M = 0
        for n1, n2 in zip(n1s, n2s):
            pos1 = nid_pos[n1]
            pos2 = nid_pos[n2]
            truss = Truss(p)
            truss.init_k_KC0 = init_k_KC0
            truss.init_k_M = init_k_M
            truss.n1 = n1
            truss.n2 = n2
            truss.c1 = DOF*pos1
            truss.c2 = DOF*pos2
            truss.update_rotation_matrix(ncoords_flatten)
            truss.update_probe_xe(ncoords_flatten)
            truss.update_KC0(KC0r, KC0c, KC0v, prop)
            truss.update_M(Mr, Mc, Mv, prop, mtype=mtype)
            trusses.append(truss)
            init_k_KC0 += data.KC0_SPARSE_SIZE
            init_k_M += data.M_SPARSE_SIZE

        print('elements created')

        KC0 = coo_matrix((KC0v, (KC0r, KC0c)), shape=(N, N)).tocsc()
        M = coo_matrix((Mv, (Mr, Mc)), shape=(N, N)).tocsc()

        print('sparse KC0 and M created')

        # applying boundary conditions
        bk = np.zeros(N, dtype=bool)
        # simply-supported with right-end free to move horizontally
        check = np.isclose(x, 0.)
        bk[0::DOF] = check # u
        check = np.isclose(x, L)
        # removing out-of-axis translations
        bk[1::DOF] = True
        bk[2::DOF] = True
        # removing rotations
        bk[3::DOF] = True
        bk[4::DOF] = True
        bk[5::DOF] = True
        bu = ~bk

        Kuu = KC0[bu, :][:, bu]
        Muu = M[bu, :][:, bu]

        num_eigenvalues = 5
        eigvals, eigvecsu = eigsh(A=Kuu, M=Muu, sigma=-1., which='LM',
                k=num_eigenvalues, tol=1e-3)
        omegan = eigvals**0.5

        omegan_theoretical = [(2*k-1)*np.pi/L/2*(E/rho)**0.5 for k in range(1,
            num_eigenvalues+1)]
        print('Theoretical omegan', omegan_theoretical)
        print('Numerical omegan', omegan)
        print()
        assert np.allclose(omegan_theoretical, omegan, rtol=0.01)

if __name__ == '__main__':
    test_truss_natural_freq(refinement=1)
