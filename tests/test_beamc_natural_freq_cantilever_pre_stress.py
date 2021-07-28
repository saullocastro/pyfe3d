import sys
sys.path.append('..')

import numpy as np
from scipy.sparse.linalg import eigsh, spsolve
from scipy.sparse import coo_matrix

from pyfe3d.beamprop import BeamProp
from pyfe3d import BeamC, BeamCData, BeamCProbe, DOF, INT, DOUBLE

def test_nat_freq_cantilever(refinement=1, mtypes=range(2)):
    for mtype in mtypes:
        print('mtype', mtype)
        n = 50*refinement
        L = 3 # total size of the beam along x

        # Material Lastrobe Lescalloy
        E = 203.e9 # Pa
        rho = 7.83e3 # kg/m3

        x = np.linspace(0, L, n)
        # path
        y = np.ones_like(x)
        # tapered properties
        b = 0.05 # m
        h = 0.05 # m
        A = h*b
        Izz = b*h**3/12

        # getting nodes
        ncoords = np.vstack((x, y, np.zeros_like(x))).T
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
        KGr = np.zeros(data.KG_SPARSE_SIZE*num_elements, dtype=INT)
        KGc = np.zeros(data.KG_SPARSE_SIZE*num_elements, dtype=INT)
        KGv = np.zeros(data.KG_SPARSE_SIZE*num_elements, dtype=DOUBLE)
        Mr = np.zeros(data.M_SPARSE_SIZE*num_elements, dtype=INT)
        Mc = np.zeros(data.M_SPARSE_SIZE*num_elements, dtype=INT)
        Mv = np.zeros(data.M_SPARSE_SIZE*num_elements, dtype=DOUBLE)
        N = DOF*n
        print('num_DOF', N)

        prop = BeamProp()
        prop.A = A
        prop.E = E
        prop.G = E/2/(1+0.3)
        prop.scf = 5/6.
        prop.Izz = Izz
        prop.intrho = rho*A
        prop.intrhoy2 = rho*Izz

        beams = []
        init_k_KC0 = 0
        init_k_KG = 0
        init_k_M = 0
        for n1, n2 in zip(n1s, n2s):
            pos1 = nid_pos[n1]
            pos2 = nid_pos[n2]
            x1, y1, z1 = ncoords[pos1]
            x2, y2, z2 = ncoords[pos2]
            beam = BeamC(p)
            beam.init_k_KC0 = init_k_KC0
            beam.init_k_KG = init_k_KG
            beam.init_k_M = init_k_M
            beam.n1 = n1
            beam.n2 = n2
            beam.c1 = DOF*pos1
            beam.c2 = DOF*pos2
            beam.cosa = 1
            beam.cosb = 1
            beam.cosg = np.cos(np.arctan2(y2 - y1, x2 - x1))
            beam.update_xe(ncoords.flatten())
            beam.update_KC0(KC0r, KC0c, KC0v, prop)
            beam.update_M(Mr, Mc, Mv, prop, mtype=mtype)
            beams.append(beam)
            init_k_KC0 += data.KC0_SPARSE_SIZE
            init_k_KG += data.KG_SPARSE_SIZE
            init_k_M += data.M_SPARSE_SIZE

        print('elements created')

        KC0 = coo_matrix((KC0v, (KC0r, KC0c)), shape=(N, N)).tocsc()
        M = coo_matrix((Mv, (Mr, Mc)), shape=(N, N)).tocsc()

        print('sparse KC0 and M created')

        # applying boundary conditions
        bk = np.zeros(N, dtype=bool) #array to store known DOFs
        check = np.isclose(x, 0.)
        # clamping
        bk[0::DOF] = check # u
        bk[1::DOF] = check # v
        bk[2::DOF] = check # w
        bk[5::DOF] = check # rz
        # removing out of XY plane displacements
        bk[3::DOF] = True # rx
        bk[4::DOF] = True # ry
        bu = ~bk # same as np.logical_not, defining unknown DOFs

        # defining external force vector
        # applying load along u at x=L
        fext = np.zeros(N)
        load = -28900
        check = np.isclose(x, L)
        fext[0::DOF][check] = load

        # sub-matrices corresponding to unknown DOFs
        Kuu = KC0[bu, :][:, bu]
        Muu = M[bu, :][:, bu]
        fextu = fext[bu]

        # static solver
        uu = spsolve(Kuu, fextu)
        u = np.zeros(N)
        u[bu] = uu

        # geometric stiffness
        for beam in beams:
            beam.update_ue(u)
            beam.update_KG(KGr, KGc, KGv, prop)
        KG = coo_matrix((KGv, (KGr, KGc)), shape=(N, N)).tocsc()
        KGuu = KG[bu, :][:, bu]
        print('sparse KG created')

        # linear buckling check
        eigvals, eigvecsu = eigsh(A=Kuu, k=1, which='SM', M=KGuu,
                tol=1e-9, sigma=1., mode='buckling')
        load_mult = -eigvals
        print('linear buckling Pcr=', load_mult*load)

        # natural frequency
        num_eigenvalues = 3
        eigvals, eigvecsu = eigsh(A=Kuu + KGuu, M=Muu, sigma=-1., which='LM',
                k=num_eigenvalues, tol=1e-3)
        omegan = eigvals**0.5

        alpha123 = np.array([1.875, 4.694, 7.885])
        omega123 = alpha123**2*np.sqrt(E*Izz/(rho*A*L**4))
        omega123_expected = [1.63753891, 164.43505923, 491.08289778]
        print('Theoretical omega123', omega123)
        print('Expected omega123 with pre-stress', omega123_expected)
        print('Numerical omega123', omegan)
        print()
        assert np.allclose(omega123_expected, omegan, rtol=1e-2)

if __name__ == '__main__':
    test_nat_freq_cantilever(refinement=1)
