import sys
sys.path.append('..')

import numpy as np
from scipy.sparse import coo_matrix

from pyfe3d.beamprop import BeamProp
from pyfe3d import BeamC, BeamCData, BeamCProbe, DOF, INT, DOUBLE
from pyfe3d.solver import natural_frequency


def test_nat_freq_curved_beam(refinement=1, mtypes=range(2)):
    r"""First three in-plane frequencies of a hinged circular arch, in the
    XZ plane

    Same arch as in ``test_beamc_natural_freq_curved.py``, rotated about the
    X axis to lie in the XZ plane, with the element orientation vector
    pointing to the center of the arch.

    The reference is Table 6, p. 1186, of Yoon, K.-Y., Park, N.-H., Choi,
    Y.-J., Kang, Y.-J., Natural frequencies of thin-walled curved beams,
    Finite Elements in Analysis and Design, 42(13), 2006, pp. 1176-1186,
    https://doi.org/10.1016/j.finel.2006.05.002: 396.98, 931.22 and 1797.31
    rad/s for the hinged-hinged arch of Sec. 5.5 and Fig. 8, p. 1185, with
    ``R = 2.438 m``, an angle of 97 degrees, ``A = 4.071e-3 m**2``, ``I_z =
    6.456e-6 m**4``, ``E = 206.8 GPa``, ``G = 77.9 GPa`` and ``rho = 7855
    kg/m**3``. These values come from the curved finite element of the
    paper, whose mesh is not given; they differ from those of Chaudhuri in
    the same table by 2.1, 0.7 and 7.6 %. The element follows the
    Euler-Bernoulli hypothesis, Sec. 2.1, assumption (3), p. 1177, i.e. it
    has no shear deformation, and its kinetic energy, Eq. (5), is that of
    the displacements (1) of the whole cross section, with rotary inertia.

    The arch is modelled twice:

    - with shear deformation, shear correction factor 5/6, the frequencies
      converge to 394.42, 918.51 and 1756.26 rad/s (800 elements), 0.64,
      1.36 and 2.28 % below the reference; with 49 elements they are 0.61,
      1.31 and 2.24 to 2.26 % below, with 24 elements 0.49 to 2.17 %. The
      tolerance of 3 % covers this difference of model;
    - without shear deformation, with ``G`` multiplied by ``1e4`` (the
      frequencies change by less than 1e-5 between ``1e3`` and ``1e6``),
      the frequencies converge to 396.71, 929.86 and 1795.51 rad/s, 0.07,
      0.15 and 0.10 % below the reference; with 49 elements they are 0.03 to
      0.09 % below, with 24 elements 0.04 to 0.11 % above.

    The frequencies of the rotated arch equal those of the arch in the XY
    plane to 1e-8.
    """
    # NOTE these rotation_rad values will keep the problem 2D on the XZ plane
    for rotation_rad in [np.pi/2, 3*np.pi/2]:
        for G_factor, rtol in [(1., 0.03), (1.e4, 0.003)]:
            for mtype in mtypes:
                print('mtype', mtype, 'G_factor', G_factor)
                n = 50*refinement
                #NOTE 2D problem, using only Izz, ignoring any out of XZ plane displacement
                #                 ignoring torsion (rx) as well
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
                normal = r*np.sin(thetas)
                y = normal*np.cos(rotation_rad)
                z = normal*np.sin(rotation_rad)

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

                prop = BeamProp()
                prop.A = A
                prop.E = E
                scf = 5/6.
                prop.G = scf*G*G_factor
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
                check = ((np.isclose(x, x[0]) & np.isclose(z, z[0]))
                         | (np.isclose(x, x[-1]) & np.isclose(z, z[-1])))
                bk[0::DOF] = check # u (axial displacement)
                bk[2::DOF] = check # w (deflection)
                bk[1::DOF] = True # out of XZ plane displacement
                bk[3::DOF] = True # torsion rx
                bk[5::DOF] = True # out-of-plane deflection
                bu = ~bk

                Kuu = KC0[bu, :][:, bu]
                Muu = M[bu, :][:, bu]

                num_eigenvalues = 3
                # NOTE pyfe3d.solver.natural_frequency equilibrates the diagonal
                #      before calling the eigensolver and returns the circular
                #      frequencies already sorted
                omegan, eigvecsu = natural_frequency(Kuu, Muu, num_eigvalues=num_eigenvalues, tol=1e-5)
                # NOTE Table 6 of Yoon et al. (2006)
                omega123_from_paper = [396.98, 931.22, 1797.31]
                print('Reference omega123_from_paper', omega123_from_paper)
                print('Numerical omega123', omegan)
                print()
                assert np.allclose(omega123_from_paper, omegan, rtol=rtol)

if __name__ == '__main__':
    test_nat_freq_curved_beam(refinement=1)
