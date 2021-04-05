import matplotlib
matplotlib.use('Agg')
import numpy as np
from codes.flasso.FlassoManifold import FlassoManifold
from codes.geometer.RiemannianManifold import RiemannianManifold
from scipy.stats import special_ortho_group

def get_grad(t):
    output = np.zeros((49,2))
    output[0,0] = ((np.cos(t) - t*np.sin(t)) / (np.sin(t) + t*np.cos(t)))
    output[2,0] = 1.
    output[1,1] = 1.
    output = output / np.linalg.norm(output, axis = 0)
    return(output)

class SwissRoll49(FlassoManifold):
    """
    Parameters
    ----------
    cor : string,
        Data file to load
    xvar : np.array(dtype = int),
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
    generate_data :
        Simulates data
    get_atoms_4 :
    	Gets atomic tetrahedra based off of ii and jj
    get_atoms_3 :
    	Gets triples of atoms

    """

    # AtomicRegression(dim, ii, jj, filename)
    def __init__(self, xvar, cores, noise):
        natoms = 9
        self.xvar = xvar
        self.cores = cores
        self.noise = noise
        self.dim = 2
        self.d = 49
        self.p = 51

    def generate_data(self, n, theta):
        self.n = n
        d = self.d
        xvar = self.xvar
        dim = self.dim
        noise = self.noise
        self.theta = theta
        ts = .75 * np.pi * (1 + 2 * np.random.uniform(low=0.0, high=1.0, size=n)) + 15
        # ts = np.pi * (1 + 2 * np.random.uniform(low=0.0, high=1.0, size=n)) + 15
        # ts = 1.5*np.pi * (1 + 2 * np.random.uniform(low=0.0, high=1.0, size=n)) + 15
        x = ts * np.cos(ts)
        y = 101 * np.random.uniform(low=0.0, high=1.0, size=n)
        self.ys = y
        z = ts * np.sin(ts)
        self.ts = ts
        X = np.vstack((x, y, z))
        # X += noise * generator.randn(3, n_samples)
        X = X.T
        unrotated_data = np.zeros((n, 49))
        unrotated_data[:, :3] = X
        rotator = special_ortho_group.rvs(49)
        # rotator = np.identity(49)
        self.rotator = rotator
        data = np.matmul(unrotated_data, rotator)
        if xvar != 0.:
            data += np.random.multivariate_normal(np.zeros(49), np.identity(49) * xvar, n)
        # for i in range(n):
        #    #print(i)
        #    data[i] += np.random.multivariate_normal(np.zeros(49),np.identity(49)*xvar)
        # data = rotator(X,theta)
        # data = dup_cols(data, indx=0, num_dups=46)
        # data = np.reshape(results, (n, (d)))
        return (RiemannianManifold(data, dim))

    def get_dx_g_full(self, data):
        d = self.d
        p = self.p
        # n = data.shape[0]
        n = len(self.selected_points)
        ts = self.ts[self.selected_points]
        grads = np.asarray([get_grad(t) for t in ts])
        rotator = self.rotator
        # grads2 = rotator(grads, theta)
        fullgrads = np.zeros((n, d, p))
        for i in range(n):
            fullgrads[i, :, 0:2] = np.matmul(rotator.transpose(), grads[i])
            fullgrads[i, :, 2:] = np.identity(49)
        output = np.swapaxes(fullgrads, 1, 2)
        return (output)

    def get_dx_g_full_2(self, selected_points):
        d = self.d
        p = self.p
        # n = data.shape[0]
        n = len(selected_points)
        ts = self.ts[selected_points]
        grads = np.asarray([get_grad(t) for t in ts])
        rotator = self.rotator
        # grads2 = rotator(grads, theta)
        fullgrads = np.zeros((n, d, p))
        for i in range(n):
            # 040220 added transpose... then removed
            fullgrads[i, :, 0:2] = np.matmul(rotator.transpose(), grads[i])
            fullgrads[i, :, 2:] = np.identity(49)
        output = np.swapaxes(fullgrads, 1, 2)
        return (output)
