import sys
sys.path.append('..')

import numpy as np
from scipy.sparse import coo_matrix

from pyfe3d.beamprop import BeamProp
from pyfe3d import BeamLR, BeamLRData, BeamLRProbe, DOF, INT, DOUBLE
from pyfe3d.solver import natural_frequency


def test_nat_freq_curved_beam(refinement=2, mtypes=range(2)):
    r"""First three in-plane frequencies of a hinged circular arch, in the
    XZ plane

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
      1.36 and 2.28 % below the reference; with 99 elements they are 0.60
      to 2.22 % below, with 49 elements 0.44 to 2.01 %. The tolerance of 3 %
      covers this difference of model;
    - without shear deformation, with ``G`` multiplied by ``1e4`` (the
      frequencies change by less than 1e-5 between ``1e3`` and ``1e6``),
      the frequencies converge to 396.71, 929.86 and 1795.51 rad/s, 0.07,
      0.15 and 0.10 % below the reference; with 99 elements they are 0.10
      % below to 0.10 % above, with 49 elements up to 0.71 % above (0.21 %
      with the lumped mass matrix), hence the 99 elements of the default
      ``refinement=2``.

    Following the policy of the tests whose reference is not reached on the
    mesh used, the frequencies are compared with the values pre-determined by
    this same test with the default ``refinement`` on 2026-10-29, with
    ``rtol = 1e-4`` for the round-off of the eigenvalue solver, called with
    ``tol=1e-9``; the frequencies with shear deformation, whose difference
    to the reference is one of the model, are compared with the reference
    with grade B of MacNeal, R. H. and Harder, R. L., A proposed standard set
    of problems to test finite element accuracy, Finite Elements in Analysis
    and Design, 1(1), 3-20, 1985,
    https://doi.org/10.1016/0168-874X(85)90003-4, p. 14 (at most 10 %), as a
    bound; until 0.12.0 with 3 %, close to the 2.3 % of the model.
    """
    # NOTE frequencies pre-determined with refinement=2, see the
    #      docstring, keyed by G_factor and mtype
    predetermined = {
        (1., 0): [394.6182542119208,
                   919.4785639487773, 1759.6161855612868],
        (1., 1): [394.52409125689485,
                   918.9628379446241, 1757.4773827807282],
        (1.e4, 0): [396.9055203071102,
                   930.8507448485416, 1799.0468388188258],
        (1.e4, 1): [396.8108157503955,
                   930.3299176932983, 1796.8617682785157]}
    for rotation_rad in [np.pi/2, 3*np.pi/2]:
        for G_factor, rtol in [(1., 0.10), (1.e4, 0.003)]:
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

                arc_deg = 97
                thetabeam = np.deg2rad(arc_deg)
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

                p = BeamLRProbe()
                data = BeamLRData()

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
                    beam = BeamLR(p)
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

                # boundary conditions
                bk = np.zeros(N, dtype=bool)
                # NOTE the end nodes are the first and the last of the arc; selecting
                #      them with np.isclose(x, x.max()) also caught the neighbour of
                #      the end at theta = 0 with fine meshes, where x varies as theta**2
                check = np.zeros(n, dtype=bool)
                check[[0, -1]] = True
                bk[0::DOF] = check
                bk[2::DOF] = check
                bk[1::DOF] = True
                bk[3::DOF] = True
                bk[5::DOF] = True
                bu = ~bk

                Kuu = KC0[bu, :][:, bu]
                Muu = M[bu, :][:, bu]

                num_eigenvalues = 3
                # NOTE pyfe3d.solver.natural_frequency equilibrates the diagonal
                #      before calling the eigensolver and returns the circular
                #      frequencies already sorted
                omegan, eigvecsu = natural_frequency(Kuu, Muu, num_eigvalues=num_eigenvalues, tol=1e-9)
                # NOTE Table 6 of Yoon et al. (2006)
                omega123_from_paper = [396.98, 931.22, 1797.31]
                print('Reference omega123_from_paper', omega123_from_paper)
                print('Numerical omega123', omegan)
                print()
                assert np.allclose(omega123_from_paper, omegan, rtol=rtol)
                if refinement == 2:
                    assert np.allclose(omegan, predetermined[G_factor, mtype],
                                       rtol=1e-4)


if __name__ == '__main__':
    test_nat_freq_curved_beam()
