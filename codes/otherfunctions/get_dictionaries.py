import numpy as np
from scipy import sparse

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