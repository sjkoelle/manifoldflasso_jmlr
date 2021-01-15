import numpy as np
from scipy import sparse
import itertools

def get_atoms_4(natoms, ii, jj):
    # ii = self.ii
    # jj = self.jj
    natoms = natoms
    adj = np.asarray([ii, jj])
    adj2 = adj.copy()
    adj2[0, :] = adj[1, :]
    adj2[1, :] = adj[0, :]
    adjacencymatrix = np.concatenate([adj, adj2], axis=1)
    ii = adjacencymatrix[0, :]
    jj = adjacencymatrix[1, :]
    # ii = ii - 1
    # jj = jj - 1
    # known adjacencies
    molecularadjacencymatrix = sparse.csr_matrix((np.ones(len(ii)), (ii, jj)))
    # compute atomic tetrahedra with central atoms in middle two coordinates
    atoms4 = np.ones((1, 4))
    for i in range(natoms):
        nebs = molecularadjacencymatrix[i].indices
        # nnebs = len(molecularadjacencymatrix[i].indices)
        for j in nebs:
            if j > i:
                i1s = np.setdiff1d(molecularadjacencymatrix[i].indices, j)
                j1s = np.setdiff1d(molecularadjacencymatrix[j].indices, i)
                for j1 in j1s:
                    for i1 in i1s:
                        atom4 = np.reshape(np.asarray([i1, i, j, j1, ]), (1, 4))
                        atoms4 = np.concatenate((atoms4, atom4), axis=0)
    atoms4 = atoms4[1:atoms4.shape[0], :]
    atoms4 = np.asarray(atoms4, dtype=int)
    return (atoms4, atoms4.shape[0])


def get_all_atoms_4(natoms):
    combos = np.asarray(list(itertools.combinations(list(range(natoms)), 4)))
    nc = combos.shape[0]
    tor_mat = np.zeros((nc, 6, 4), dtype=int)
    for c in range(nc):
        tor_mat[c] = np.asarray([combos[c][[0, 1, 2, 3]],
                                 combos[c][[1, 0, 2, 3]],
                                 # combos[c][[0,2,1,3]],
                                 combos[c][[3, 1, 0, 2]],
                                 combos[c][[2, 1, 3, 0]],
                                 # combos[0][[1,2,3,0]],
                                 combos[c][[0, 3, 2, 1]],
                                 combos[c][[1, 0, 3, 2]]])
    output = np.reshape(tor_mat, (nc * 6, 4))
    return (output, output.shape[0])