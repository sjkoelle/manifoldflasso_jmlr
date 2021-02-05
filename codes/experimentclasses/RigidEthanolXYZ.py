from codes.experimentclasses.AtomicRegression import AtomicRegression
from codes.geometer.RiemannianManifold import RiemannianManifold
from codes.otherfunctions.data_stream import data_stream
from codes.geometer.ShapeSpace import compute3angles
import numpy as np
import torch
import math
from sklearn.decomposition import TruncatedSVD
from pathos.multiprocessing import ProcessingPool as Pool


class RigidEthanolXYZ(AtomicRegression):
    """
    This class estimates
    Parameters
    ----------
    filename : string,
        Data file to load
    ii : np.array(dtype = int),
        List of adjacencies
    jj : np.array,
        List of adjacencies part 2
    d : int,
        dimension over which to evaluate the radii (smaller usually better)
    rmin : float,
        smallest radius ( = rad_bw_ratio * bandwidth) to consider
    rmax : float,
        largest radius ( = rad_bw_ratio * bandwidth) to consider
    ntry : int,
        number of radii between rmax and rmin to try
    run_parallel : bool,
        whether to run the analysis in parallel over radii
    search_space : str,
        either 'linspace' or 'logspace', choose to search in log or linear space
    rad_bw_ratio : float,
        the ratio of radius and kernel bandwidth, default to be 3 (radius = 3*h)
    Methods
    -------
    load_data :
        Loads filename as AtomicRegression.data
    get_atoms_4 :
    	Gets atomic tetrahedra based off of ii and jj
    get_atoms_3 :
    	Gets triples of atoms

    """

    # AtomicRegression(dim, ii, jj, filename)
    def __init__(self, dim, cor, xvar, cores, noise, custom_bonds=None ):
        # def __init__(self, r, R, p,n,d, selectedpoints, dim):
        # self.ii = ii
        # self.jj = jj
        self.n = 10000
        self.natoms = 9
        self.cor = cor
        self.cores = cores
        self.noise = noise
        # n_atoms = len(np.unique(ii))
        self.xvar = xvar
        #self.atoms4, self.p = self.get_atoms_4()
        self.d = 27
        # self.atoms3, self.d = self.get_atoms_3(natoms)
        # self.selectedpoints = selectedpoints
        self.dim = dim
        if custom_bonds.any() != None:
            self.atoms4 = custom_bonds
            self.p = custom_bonds.shape[0]
        # self.n = n
    # def __init__(self, dim, cor, xvar, ii, jj, cores, noise, custom_bonds=None):
    #     natoms = 9
    #     n = 10000
    #     self.cor = cor
    #     self.xvar = xvar
    #     self.cores = cores
    #     self.noise = noise
    #     AtomicRegression.__init__(self, dim, n, ii, jj, natoms, cores)


        # FlassoManifold.__init__(self, data, jacg, selectedpoints, dim, diffusion_time)
        # self.tangent_bases = np.zeros((self.n,self.d,self.dim))

    #     def load_data(self, angles = False):
    #         #filename = 'tolueneangles.npz'
    #         n = self.n
    #         d = self.d
    #         dim = self.dim
    #         cor = self.cor
    #         natoms = self.natoms

    #         positions = np.zeros((n,natoms,3))
    #         positions[0,0,:] = np.asarray([0.,0.,0.])
    #         positions[0,1,:] = np.asarray([-1.,0.,0.])
    #         positions[0,2,:] = np.asarray([0.,1.,0.])
    #         positions[0,8,:] = np.asarray([.5,1.,0.])
    #         positions[0,3,:] = np.asarray([0.,.25 * np.cos(2/3 * np.pi),.25 * np.sin(2/3 * np.pi)])
    #         positions[0,4,:] = np.asarray([0.,.25 * np.cos(2/3 * np.pi),.25 * np.sin(4/3 * np.pi)])
    #         positions[0,5,:] = np.asarray([-1.,.3,0.])
    #         positions[0,6,:] = np.asarray([-1.,.3 * np.cos(2/3 * np.pi),.3 * np.sin(2/3 * np.pi)])
    #         positions[0,7,:] = np.asarray([-1.,.3 * np.cos(2/3 * np.pi),.3 * np.sin(4/3 * np.pi)])
    #         angles1 = np.tile(np.linspace(start = 0.,stop = 2*math.pi, num = int(np.sqrt(n)), endpoint = False), int(np.sqrt(n)))
    #         angles2 = np.repeat(np.linspace(start = 0.,stop = 2*math.pi, num = int(np.sqrt(n)), endpoint = False), int(np.sqrt(n)))
    #         for i in range(1,n):
    #             rotationmatrix1 = np.zeros((3,3))
    #             rotationmatrix1[0,0] = 1
    #             rotationmatrix1[1,1] = np.cos(angles1[i])
    #             rotationmatrix1[1,2] = -np.sin(angles1[i])
    #             rotationmatrix1[2,2] = np.cos(angles1[i])
    #             rotationmatrix1[2,1] = np.sin(angles1[i])

    #             rotationmatrix2 = np.zeros((3,3))
    #             rotationmatrix2[0,0] = 1
    #             rotationmatrix2[1,1] = np.cos(angles2[i])
    #             rotationmatrix2[1,2] = -np.sin(angles2[i])
    #             rotationmatrix2[2,2] = np.cos(angles2[i])
    #             rotationmatrix2[2,1] = np.sin(angles2[i])
    #             positions[i,np.asarray([3,4]),:] = positions[0,np.asarray([3,4]),:]
    #             positions[i,np.asarray([2,8]),:] = np.matmul(rotationmatrix1, positions[0,np.asarray([2,8]),:].transpose()).transpose()
    #             positions[i,np.asarray([1,5,6,7]),:] = np.matmul(rotationmatrix2, positions[0,np.asarray([1,5,6,7]),:].transpose()).transpose()

    #         covariance = np.identity(natoms)
    #         for i in range(natoms):
    #             for j in range(natoms):
    #                 if i != j:
    #                     covariance[i,j] = cor

    #         for i in range(n):
    #             for j in range(3):
    #                 positions[i,:,j] = positions[i,:,j] + np.random.multivariate_normal(positions[i,:,j], covariance,size = 1)

    #         if angles == True:
    #             p = Pool(cores)
    #             results = p.map(lambda i: compute3angles(position = positions[i[0],atoms3[i[1]],:]),data_stream(10000,84))
    #             data = np.reshape(results, (n,(d)))
    #         else:
    #             data = np.reshape(positions,(n,d))
    #         return(RiemannianManifold(data, dim))

    def load_data(self, angles=False):
        # filename = 'tolueneangles.npz'
        n = self.n
        d = self.d
        dim = self.dim
        cor = self.cor
        natoms = self.natoms
        xvar = self.xvar
        cores = self.cores

        positions = np.zeros((n, natoms, 3))
        positions[0, 0, :] = np.asarray([0., 0., 0.])
        positions[0, 1, :] = np.asarray([-10., 0., 0.])
        positions[0, 2, :] = np.asarray([0., 10., 0.])
        # positions[0,8,:] = np.asarray([1.,1.,0.])
        positions[0, 8, :] = np.asarray([1., 10., 0.])
        positions[0, 3, :] = np.asarray([0., np.cos(2 / 3 * np.pi), np.sin(2 / 3 * np.pi)])
        positions[0, 4, :] = np.asarray([0., np.cos(2 / 3 * np.pi), np.sin(4 / 3 * np.pi)])
        positions[0, 5, :] = np.asarray([-10., 1., 0.])
        positions[0, 6, :] = np.asarray([-10., np.cos(2 / 3 * np.pi), np.sin(2 / 3 * np.pi)])
        positions[0, 7, :] = np.asarray([-10., np.cos(2 / 3 * np.pi), np.sin(4 / 3 * np.pi)])
        angles1 = np.tile(np.linspace(start=0., stop=2 * math.pi, num=int(np.sqrt(n)), endpoint=False), int(np.sqrt(n)))
        angles2 = np.repeat(np.linspace(start=0., stop=2 * math.pi, num=int(np.sqrt(n)), endpoint=False),
                            int(np.sqrt(n)))
        for i in range(1, n):
            #     rotationmatrix1 = np.zeros((3,3))
            #     rotationmatrix1[0,0] = 1
            #     rotationmatrix1[1,1] = np.cos(angles1[i])
            #     rotationmatrix1[1,2] = -np.sin(angles1[i])
            #     rotationmatrix1[2,2] = np.cos(angles1[i])
            #     rotationmatrix1[2,1] = np.sin(angles1[i])

            rotationmatrix1 = np.zeros((3, 3))
            rotationmatrix1[1, 1] = 1
            rotationmatrix1[0, 0] = np.cos(angles1[i])
            rotationmatrix1[0, 2] = -np.sin(angles1[i])
            rotationmatrix1[2, 2] = np.cos(angles1[i])
            rotationmatrix1[2, 0] = np.sin(angles1[i])

            rotationmatrix2 = np.zeros((3, 3))
            rotationmatrix2[0, 0] = 1
            rotationmatrix2[1, 1] = np.cos(angles2[i])
            rotationmatrix2[1, 2] = -np.sin(angles2[i])
            rotationmatrix2[2, 2] = np.cos(angles2[i])
            rotationmatrix2[2, 1] = np.sin(angles2[i])
            positions[i, np.asarray([3, 4]), :] = positions[0, np.asarray([3, 4]), :]
            positions[i, np.asarray([2, 8]), :] = np.matmul(rotationmatrix1,
                                                            positions[0, np.asarray([2, 8]), :].transpose()).transpose()
            positions[i, np.asarray([1, 5, 6, 7]), :] = np.matmul(rotationmatrix2,
                                                                  positions[0, np.asarray([1, 5, 6, 7]),
                                                                  :].transpose()).transpose()

        print(positions[0])
        covariance = np.identity(natoms)
        for i in range(natoms):
            for j in range(natoms):
                if i != j:
                    covariance[i, j] = cor
        # covariance = xvar * covariance
        # for i in range(n):
        #    for j in range(3):
        #        positions[i,:,j] = np.random.multivariate_normal(positions[i,:,j], covariance,size = 1)
        self.positions = positions

        print(positions[0])
        if angles == True:
            p = Pool(cores)
            results = p.map(lambda i: compute3angles(position=positions[i[0], atoms3[i[1]], :]), data_stream(10000, 84))
            data = np.reshape(results, (n, (d)))
        else:
            data = np.reshape(positions, (n, d))
        return (RiemannianManifold(data, dim))

    # def get_atoms_4(self):
    #     atoms4 = np.asarray([[6, 1, 0, 4], [5, 1, 0, 4], [4, 0, 2, 8], [7, 6, 5, 1], [3, 0, 2, 4]], dtype=int)
    #     return (atoms4, atoms4.shape[0])

    #     def get_atoms_4(self, ii, jj):
    #         #ii = self.ii
    #         #jj = self.jj
    #         natoms = self.natoms
    #         adj = np.asarray([ii,jj])
    #         adj2 = adj.copy()
    #         adj2[0,:] = adj[1,:]
    #         adj2[1,:] = adj[0,:]
    #         adjacencymatrix = np.concatenate([adj, adj2], axis = 1)
    #         ii = adjacencymatrix[0,:]
    #         jj = adjacencymatrix[1,:]
    #         #ii = ii - 1
    #         #jj = jj - 1
    #         #known adjacencies
    #         molecularadjacencymatrix = sparse.csr_matrix((np.ones(len(ii)), (ii, jj)))
    #         #compute atomic tetrahedra with central atoms in middle two coordinates
    #         atoms4 = np.ones((1,4))
    #         for i in range(natoms):
    #             nebs = molecularadjacencymatrix[i].indices
    #             #nnebs = len(molecularadjacencymatrix[i].indices)
    #             for j in nebs:
    #                 if j > i:
    #                     i1s = np.setdiff1d(molecularadjacencymatrix[i].indices,j)
    #                     j1s = np.setdiff1d(molecularadjacencymatrix[j].indices,i)
    #                     for j1 in j1s:
    #                         for i1 in i1s:
    #                             atom4 = np.reshape(np.asarray([ i1, i, j, j1, ]), (1,4))
    #                             atoms4 = np.concatenate((atoms4, atom4), axis = 0)
    #         atoms4 = atoms4[1:atoms4.shape[0],:]
    #         atoms4 = np.asarray(atoms4, dtype = int)
    #         return(atoms4, atoms4.shape[0])

    #     def get_atoms_3(self, natoms):
    #         atoms3 = np.asarray(list(itertools.combinations(range(natoms), 3)))
    #         return(atoms3, atoms3.shape[0]*3)

    # def get_central_point(self, nsel, fitmodel, data):
    #     centralpoint = np.zeros(nsel)
    #     for i in range(nsel):
    #         # print(i)
    #         distomean = np.zeros(len(fitmodel.adjacency_matrix[i].indices))
    #         meanpoint = data[fitmodel.adjacency_matrix[i].indices, :].mean(axis=0)
    #         for j in range(len(fitmodel.adjacency_matrix[i].indices)):
    #             distomean[j] = np.linalg.norm(data[fitmodel.adjacency_matrix[i].indices[j], :] - meanpoint)
    #         centralpoint[i] = fitmodel.adjacency_matrix[i].indices[distomean.argmin()]
    #
    #     centralpoint = np.asarray(centralpoint, dtype=int)
    #     return (centralpoint)

    #     def g(self, x):
    #         atoms4 = self.atoms4
    #         atoms3 = self.atoms3

    #         #angles9 = np.reshape(x, (atoms3.shape[0],3))
    #         output = []
    #         combos = np.asarray([[0,1,2],[1,2,3],[0,2,3],[0,1,3]])
    #         for atom4 in atoms4:
    #             angles4 = []
    #             actived = np.zeros(4)
    #             for i in range(4):
    #                 actived[i] = np.where([set(item).issubset(atom4[combos[i,:]]) for item in atoms3])[0][0]
    #             actived = np.asarray(actived, dtype =int)
    #             naive = np.reshape(x, (atoms3.shape[0],3))[actived,:]
    #             for i in range(4):
    #                 a = atoms3[actived[i]]
    #                 b = atom4[np.in1d(atom4, atoms3[actived[i]])]
    #                 for j in range(3):
    #                     angles4.append(naive[i,np.where(a == b[j])[0]])
    #             a4 = np.reshape(angles4, (4,3))
    #             #print(a4)
    #             output.append(g4(a4))
    #         return(np.asarray(output))

    def get_dx_g_full(self, data):

        n = data.shape[0]
        d = self.d
        p = self.p
        output = np.zeros((n, p, d))
        for i in range(n):
            # if angles == True:
            #     pass
            # else:
            output[i, :, :] = self.get_dx_g_xyz(data[i]).transpose()
        return (output)

    #     def get_dx_g_pos_pytorch_full(self,data):
    #         n = data.shape[0]
    #         d = self.d
    #         p = self.p
    #         output = np.zeros((n,p,d))
    #         for i in range(n):
    #             output[i,:,:] = self.get_dx_g_pos_pytorch(data[i]).transpose()
    #         return(output)
    def get_g(self, x):
        atoms4 = self.atoms4
        # atoms3 = self.atoms3
        # p = len(atoms4)
        p = self.p
        # combos = np.asarray([[0,1,2],[1,2,3],[0,2,3],[0,1,3]])
        # d = atoms3.shape[0] * 3
        d = self.d
        output = np.zeros((p))
        # loop over tetrehedra
        for k in range(p):
            atom4 = atoms4[k, :]
            x4 = np.zeros((4, 3))
            for i in range(4):
                x4[i, :] = x[(atom4[i] * 3):(atom4[i] * 3 + 3)]
            x4 = np.asarray(x4)
            fitin = self.g4(x4, True)
            output[k] = fitin[0]
        return (output)

    def get_g_full(self, data):
        n = data.shape[0]
        p = self.p
        output = np.zeros((n, p))
        for i in range(n):
            # if angles == True:
            #     pass
            # else:
            output[i, :] = self.get_g(data[i]).transpose()
        return (output)

    def get_dx_g_xyz(self, x):
        atoms4 = self.atoms4
        # atoms3 = self.atoms3
        # p = len(atoms4)
        p = self.p
        # combos = np.asarray([[0,1,2],[1,2,3],[0,2,3],[0,1,3]])
        # d = atoms3.shape[0] * 3
        d = self.d
        output = np.zeros((d, p))
        # loop over tetrehedra
        for k in range(p):
            atom4 = atoms4[k, :]
            x4 = np.zeros((4, 3))
            for i in range(4):
                x4[i, :] = x[(atom4[i] * 3):(atom4[i] * 3 + 3)]
            x4 = np.asarray(x4)
            fitin = self.g4(x4, True)
            for i in range(4):
                # a = atoms3[actived[i]]
                for j in range(3):
                    # plus the lowest index first
                    output[3 * atom4[i] + j, k] = fitin[1][i, j]
        return (output)

    # need to check range of arccos
    def positions_to_torsion(self, positions4):
        positions4 = torch.tensor(positions4, requires_grad=True)
        d1 = positions4[0]
        c1 = positions4[1]
        c2 = positions4[2]
        d2 = positions4[3]
        cc = c2 - c1
        ip = torch.sum((d1 - c1) * (c2 - c1)) / (torch.sum((c2 - c1) ** 2))
        tilded1 = [d1[0] - ip * cc[0], d1[1] - ip * cc[1], d1[2] - ip * cc[2]]
        iq = torch.sum((d2 - c2) * (c1 - c2)) / (torch.sum((c1 - c2) ** 2))
        cc2 = (c1 - c2)
        tilded2 = [d2[0] - iq * cc2[0], d2[1] - iq * cc2[1], d2[2] - iq * cc2[2]]
        tilded2star = [tilded2[0] + cc2[0], tilded2[1] + cc2[1], tilded2[2] + cc2[2]]
        ab = torch.sqrt((tilded2star[0] - c1[0]) ** 2 + (tilded2star[1] - c1[1]) ** 2 + (tilded2star[2] - c1[2]) ** 2)
        bc = torch.sqrt((tilded2star[0] - tilded1[0]) ** 2 + (tilded2star[1] - tilded1[1]) ** 2 + (
                    tilded2star[2] - tilded1[2]) ** 2)
        ca = torch.sqrt((tilded1[0] - c1[0]) ** 2 + (tilded1[1] - c1[1]) ** 2 + (tilded1[2] - c1[2]) ** 2)
        output = torch.acos((ab ** 2 - bc ** 2 + ca ** 2) / (2 * ab * ca))
        return (output)

    def g4(self, positions4, grad=True):
        positions4 = torch.tensor(positions4, requires_grad=True)
        torsion = self.positions_to_torsion(positions4)
        torsion.backward(retain_graph=True)
        return (torsion, positions4.grad)

    # def get_embedding3(self, geom, n_components, diffusion_time, d):
    #
    #     Y0, eigenvalues, eigenvectors = spectral_embedding.spectral_embedding(geom=geom, eigen_solver='amg',
    #                                                                           random_state=6, diffusion_maps=True,
    #                                                                           diffusion_time=diffusion_time,
    #                                                                           n_components=n_components)
    #     # geom.rmetric = RiemannMetric(Y0,geom.laplacian_matrix,n_dim=n_components)
    #     geom.rmetric = RiemannMetric(Y0, geom.laplacian_matrix, n_dim=d)
    #     geom.rmetric.get_rmetric()
    #     output = RiemannianManifold(Y0, d)
    #     output.geom = geom
    #     return (output)
    #
    # def get_embedding3notfixed(self, geom, n_components, diffusion_time, d):
    #
    #     Y0, eigenvalues, eigenvectors = spectral_embedding.spectral_embedding(geom=geom, eigen_solver='amg',
    #                                                                           random_state=6, diffusion_maps=True,
    #                                                                           diffusion_time=diffusion_time,
    #                                                                           n_components=n_components)
    #     geom.rmetric = RiemannMetric(Y0, geom.laplacian_matrix, n_dim=n_components)
    #     # dim = self.dim
    #     # geom.rmetric = RiemannMetric(Y0,geom.laplacian_matrix,n_dim=dim)
    #     geom.rmetric.get_rmetric()
    #     output = RiemannianManifold(Y0, d)
    #     output.geom = geom
    #     return (output)
    #
    # def get_dF_js_idM(self, M, N, M_tangent_bundle_sub, N_tangent_bundle, selectedpoints):
    #
    #     # manifold_experiment.df_M = manifold_experiment.estimate_df_js_sel(manifold_experiment.M, manifold_experiment.N, manifold_experiment.M.induced_estimated_tb, manifold_experiment.N.induced_estimated_tb, sample_pts)
    #     # n = self.n
    #     q = self.q
    #     d = self.d
    #     # M = self.M
    #     # N = self.N
    #     dim = self.dim
    #
    #     affinity_matrix = M.geom.affinity_matrix
    #
    #     nsel = len(selectedpoints)
    #     dF = np.zeros((nsel, dim, q))
    #
    #     for i in range(nsel):
    #         pt = selectedpoints[i]
    #         neighborspt = affinity_matrix[selectedpoints[i]].indices
    #         deltap0 = M.data[neighborspt, :] - M.data[pt, :]
    #         deltaq0 = N.data[neighborspt, :] - N.data[pt, :]
    #         projected_M = np.matmul(M_tangent_bundle_sub.tangent_bases[i, :, :].transpose(),
    #                                 deltap0.transpose()).transpose()
    #         # projected_rescaled_M = np.matmul(np.diag(M_tangent_bundle_sub.rmetric.Gsvals[selectedpoints[i]]),projected_M.transpose())
    #         projected_rescaled_M = projected_M.transpose()
    #         b = np.linalg.pinv(projected_rescaled_M)
    #         a = np.zeros((len(neighborspt), q))
    #         rescaled_basis = np.matmul(N_tangent_bundle.tangent_bases[selectedpoints[i], :, :][:, :],
    #                                    np.diag(N.geom.rmetric.Gsvals[selectedpoints[i]]))
    #         projected_N = np.dot(rescaled_basis.transpose(), deltaq0.transpose())
    #         projected_N_expanded = np.matmul(N_tangent_bundle.tangent_bases[selectedpoints[i], :, :][:, :], projected_N)
    #         a = projected_N_expanded
    #         dF[i, :, :][:, :] = np.matmul(a, b).transpose()
    #     return (dF)
