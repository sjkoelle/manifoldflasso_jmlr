
import os
import datetime
import numpy as np
import dill as pickle
import random
import sys
from matplotlib import rcParams

rcParams['figure.figsize'] = 25, 10

from sklearn.decomposition import TruncatedSVD
import scipy

np.random.seed(0)
random.seed(0)
now = datetime.datetime.now().strftime("%B_%d_%Y_%H_%M_%S")
workingdirectory = os.popen('git rev-parse --show-toplevel').read()[:-1]
sys.path.append(workingdirectory)
os.chdir(workingdirectory)

from codes.experimentclasses.AtomicRegression2 import AtomicRegression
from codes.otherfunctions.get_dictionaries import get_atoms_4
from codes.flasso.Replicate import Replicate
from codes.geometer.RiemannianManifold import RiemannianManifold
from codes.flasso.GradientGroupLasso import batch_stream, get_sr_lambda_sam_parallel
from codes.otherfunctions.get_grads import get_grads3


n = 50000  # number of data points to simulate
nsel = 100  # number of points to analyze with lasso
n_neighbors = 1000  # number of neighbors in megaman
m = 3  # number of embedding dimensions (diffusion maps)
diffusion_time = 1.  # embedding radius
dim = 1  # manifold dimension
dimnoise = 1  # manifold dimension (not in mflasso paper)
cores = 3
nreps = 25
natoms = 15
ii = np.asarray([0, 0, 0, 0, 1, 6, 5, 6, 5, 4, 4, 3, 3, 2, 2])
jj = np.asarray([8, 9, 7, 1, 6, 14, 13, 5, 4, 12, 3, 11, 2, 10, 1])

savename = 'toluene_021521'
savefolder = 'toluene'
data_wd = '/Users/samsonkoelle/Downloads/manigrad-100818/mani-samk-gradients/'

folder = workingdirectory + '/Figures/toluene/' + now + 'n' + str(n) + 'nsel' + str(nsel) + 'nreps' + str(nreps)
os.mkdir(folder)
experiment = AtomicRegression(natoms)
experiment.m = m
experiment.dim = dim
experiment.dnoise = dim
experiment.nreps = nreps
experiment.nsel = nsel
experiment.atoms3, experiment.da = experiment.get_atoms_3()
experiment.atoms4, experiment.p = get_atoms_4(natoms,ii,jj)

data_wd = '/Users/samsonkoelle/Downloads/manigrad-100818/mani-samk-gradients/'
data_xyz_loaded = scipy.io.loadmat(data_wd + '/untracked_data/chemistry_data/toluene.mat')
angle_indices = np.load(data_wd + '/untracked_data/chemistry_data/tolueneindices020619.npy')
experiment.positions = data_xyz_loaded['R'][angle_indices]
filename_angles = data_wd + '/untracked_data/chemistry_data/tolueneangles020619.npy'
data = np.arccos(np.reshape(np.load(filename_angles), (50000, experiment.da)))

experiment.M = RiemannianManifold(data, dim)  # experiment.load_data(workingdirectory = data_wd)
experiment.svd = TruncatedSVD(n_components=50)
experiment.Mpca = RiemannianManifold(experiment.svd.fit_transform(experiment.M.data), dim)
experiment.Mpca.geom = experiment.Mpca.compute_geom(diffusion_time, n_neighbors)
experiment.N = experiment.Mpca.get_embedding3(experiment.Mpca.geom, m, diffusion_time, dim)

print('pre-gradient acquisition', datetime.datetime.now())
replicates = {}
for i in range(nreps):
    replicates[i] = Replicate(nsel=nsel, n=experiment.M.data.shape[0])
    replicates[i].df_M, replicates[i].dg_M, replicates[i].dg_w, replicates[i].dg_w_pca, replicates[i].dgw_norm, \
    replicates[i].tangent_bases = get_grads3(experiment, experiment.Mpca, experiment.M, experiment.N,
                                             replicates[i].selected_points, experiment.svd)

gl_itermax = 500
r = 0
max_search = 30
reg_l2 = 0.
tol = 1e-14
learning_rate = 100
results = {}
for r in range(nreps):
    ul = np.linalg.norm(np.einsum('n m d, n p d -> n p m ', replicates[r].df_M, replicates[r].dg_M),
                        axis=tuple([0, 2])).max()
    lambdas_start = [0., ul]
    # to fix...
    replicates[r].dg_M = np.swapaxes(replicates[r].dg_M, 1, 2)
    replicates[r].df_M = np.swapaxes(replicates[r].df_M, 1, 2)
    replicates[r].results = get_sr_lambda_sam_parallel(replicates[r], gl_itermax, lambdas_start, reg_l2, max_search,
                                                       dim, tol, learning_rate)



with open(workingdirectory + '/untracked_data/embeddings/' + savefolder + '/' + savename + 'replicates_small.pkl',
          'wb') as output:
    pickle.dump(results, output, pickle.HIGHEST_PROTOCOL)