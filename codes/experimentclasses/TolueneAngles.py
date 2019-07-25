from codes.experimentclasses.AtomicRegression import AtomicRegression
from codes.geometer.RiemannianManifold import RiemannianManifold
import numpy as np
import scipy
import os
workingdirectory = os.popen('git rev-parse --show-toplevel').read()[:-1]

class TolueneAngles(AtomicRegression):
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
    def __init__(self, dim, n, ii, jj,cores, custom_bonds = None):
        # def __init__(self, r, R, p,n,d, selectedpoints, dim):
        self.ii = ii
        self.jj = jj
        self.n = n
        natoms = 15
        self.natoms = natoms
        # self.cor = cor
        # n_atoms = len(np.unique(ii))
        # self.xvar = xvar
        self.atoms4, self.p = self.get_atoms_4(ii, jj)
        self.d = 27
        self.atoms3, self.d = self.get_atoms_3()
        # self.selectedpoints = selectedpoints
        self.dim = dim
        AtomicRegression.__init__(self, dim, n, ii, jj, natoms, cores)
        # self.n = n
        print(custom_bonds)
        if custom_bonds.any() != None:
            self.atoms4 = custom_bonds
            self.p = custom_bonds.shape[0]


    def load_data(self, angles=False):
        n = self.n
        d = self.d
        dim = self.dim
        natoms = self.natoms
        atoms3 = self.atoms3

        filename_xyz = workingdirectory + '/untracked_data/chemistry_data/toluene.mat'
        filename_angle_indices = workingdirectory + '/untracked_data/chemistry_data/tolueneindices020619.npy'
        data_xyz_loaded = scipy.io.loadmat(filename_xyz)
        angle_indices = np.load(filename_angle_indices)
        positions = data_xyz_loaded['R'][angle_indices]
        self.positions = positions
        filename_angles = workingdirectory + '/untracked_data/chemistry_data/tolueneangles020619.npy'
        data = np.reshape(np.load(filename_angles), (50000, 3 * len(atoms3)))
        data = np.arccos(data)
        return (RiemannianManifold(data, dim))