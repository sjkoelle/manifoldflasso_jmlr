#Samson Koelle
#Meila group
#021419


#rootdirectory = '/Users/samsonkoelle/Downloads/manigrad-100818/mani-samk-gradients'
#f = open(rootdirectory + '/code/source/packagecontrol.py')
#source = f.read()
#exec(source)
#f = open(rootdirectory + '/code/source/sourcecontrol.py')
#source = f.read()
#exec(source)
#f = open(rootdirectory + '/code/source/RigidEthanol.py')
#source = f.read()
#exec(source)
import matplotlib
from shutil import copyfile
matplotlib.use('Agg')
import os
import datetime
import numpy as np
import dill as pickle
import random
import sys
np.random.seed(0)
random.seed(0)
now = datetime.datetime.now().strftime("%B_%d_%Y_%H_%M_%S")
workingdirectory = os.popen('git rev-parse --show-toplevel').read()[:-1]
sys.path.append(workingdirectory)
os.chdir(workingdirectory)
#print(os.getcwd())
from codes.experimentclasses.EthanolAngles import EthanolAngles
from codes.otherfunctions.multiplot import plot_reg_path_ax_lambdasearch_tangent
from codes.otherfunctions.get_dictionaries import get_atoms_4
from codes.flasso.Replicate import Replicate
from codes.otherfunctions.get_grads_tangent import get_grads_tangent
from codes.otherfunctions.multirun import get_support_recovery_lambda
from codes.otherfunctions.multirun import get_lower_interesting_lambda
import matplotlib.pyplot as plt
from codes.otherfunctions.multirun import get_coeffs_and_lambdas
from codes.geometer.RiemannianManifold import RiemannianManifold
import matplotlib.pyplot as plt

#set parameters
n = 50000 #number of data points to simulate
nsel = 100 #number of points to analyze with lasso
itermax = 1000 #maximum iterations per lasso run
tol = 1e-10 #convergence criteria for lasso
#lambdas = np.asarray([5,10,15,20,25,50,75,100], dtype = np.float16)#lambda values for lasso
#lambdas = np.asarray([0,2.95339658e-06, 5.90679317e-06, 8.86018975e-06, 1.18135863e-05,
#       1.47669829e-05, 2.95339658e-05, 4.43009487e-05, 5.90679317e-05])
#lambdas = np.asarray([0,.001,.01,.1,1,10], dtype = np.float16)
lambdas = np.asarray(np.hstack([np.asarray([0]),np.logspace(-3,-1,11)]), dtype = np.float16)
#lambdas = np.asarray([0,1,2,3,4,5,6,7,8,9,10], dtype = np.float16)
n_neighbors = 100
n_components = 4#number of embedding dimensions (diffusion maps)
diffusion_time = .50 #diffusion time controls gaussian kernel radius per gradients paper
dim = 4 #manifold dimension
dimnoise = 4 #noise dimension
cores = 16 #number of cores for parallel processing
ii = np.asarray([0,0,0,0,1,1,1,2]) # atom adjacencies for dihedral angle computation
jj = np.asarray([1,2,3,4,5,6,7,8])
#run experiment
#atoms4 = np.asarray([[9,0,1,2],[0,1,2,3],[1,2,3,4],[2,3,4,5],[3,4,5,6],[4,5,6,1],[5,6,1,0]],dtype = int)
atoms4 = np.asarray([[6,1,0,4],[4,0,2,8],[7,6,5,1],[3,0,2,4]],dtype = int)
folder = workingdirectory + '/Figures/ethanol/dim4' + now
os.mkdir(folder)


new_MN = True
src = workingdirectory + '/codes/experiments/ethanolpca_multirun_tangent_2.py'
filenamescript = folder + '/script.py'
copyfile(src, filenamescript)

new_grad = True
savename = 'ethanol_052820_dim4'
savefolder = 'ethanol'
loadfolder = 'ethanol'
loadname = 'ethanol_052820_dim4'
nreps = 25
atoms4,p = get_atoms_4(9,ii,jj)
if new_MN == True:
    experiment = EthanolAngles(dim,  ii, jj,cores,atoms4)
    projector  = np.load(workingdirectory + '/untracked_data/chemistry_data/ethanolangles022119_pca50_components.npy')
    experiment.projector = projector
    experiment.M = experiment.load_data()  # if noise == False then noise parameters are overriden
    experiment.Mpca = RiemannianManifold(np.load(workingdirectory + '/untracked_data/chemistry_data/ethanolangles022119_pca50.npy'), dim)
    experiment.q = n_components
    experiment.dimnoise = dimnoise
    n = experiment.n
    experiment.Mpca.geom = experiment.Mpca.compute_geom(diffusion_time, n_neighbors)
    # experiment.N = experiment.Mpca.get_embedding3(experiment.Mpca.geom, n_components, diffusion_time, dim)
    # with open(workingdirectory + '/untracked_data/embeddings/' + savefolder + '/' + savename + '.pkl' ,
    #          'wb') as output:
    #      pickle.dump(experiment, output, pickle.HIGHEST_PROTOCOL)

lambda_max = 1
max_search = 30

experiment.p = p
experiment.atoms4 = atoms4
experiment.itermax = itermax
experiment.tol = tol
experiment.dnoise = dim
experiment.nreps = nreps
experiment.nsel = nsel
experiment.folder = folder
experiment.m = dim
replicates = {}
selected_points_save = np.zeros((nreps,nsel))
for i in range(nreps):
    selected_points = np.random.choice(list(range(n)),nsel,replace = False)
    selected_points_save[i] = selected_points
    replicates[i] = Replicate()
    replicates[i].nsel = nsel
    replicates[i].selected_points = selected_points
    replicates[i].df_M,replicates[i].dg_M,replicates[i].dg_w ,replicates[i].dg_w_pca ,replicates[i].dgw_norm  = get_grads_tangent(experiment, experiment.Mpca, experiment.M, selected_points, False)
    replicates[i].xtrain, replicates[i].groups = experiment.construct_X(replicates[i].dg_M)
    replicates[i].ytrain = experiment.construct_Y(replicates[i].df_M)
    replicates[i].coeff_dict = {}
    replicates[i].coeff_dict[0] = experiment.get_betas_spam2(replicates[i].xtrain, replicates[i].ytrain, replicates[i].groups, np.asarray([0]), nsel, experiment.dim, itermax, tol)
    replicates[i].combined_norms = {}
    replicates[i].combined_norms[0] = np.linalg.norm(np.linalg.norm(replicates[i].coeff_dict[0][:, :, :, :], axis=2), axis=1)[0,:]
    replicates[i].higher_lambda,replicates[i].coeff_dict,replicates[i].combined_norms = get_support_recovery_lambda(experiment, replicates[i],  lambda_max, max_search,2) #dim 2... should use dim dnoise but too busy
    replicates[i].lower_lambda,replicates[i].coeff_dict,replicates[i].combined_norms = get_lower_interesting_lambda(experiment, replicates[i],  lambda_max, max_search)
    #= experiment.get_betas_spam2(replicates[i].xtrain, replicates[i].ytrain, replicates[i].groups, lambdas, len(selected_points), n_embedding_coordinates, itermax, tol)

m  = experiment.q
#fig, axes_all = plt.subplots(nreps, 1,figsize=(15 * m, 15*nreps))
#fig.suptitle('Regularization paths')
for i in range(nreps):
    replicates[i].coeffs, replicates[i].lambdas_plot = get_coeffs_and_lambdas(replicates[i].coeff_dict, replicates[i].lower_lambda, replicates[i].higher_lambda)
    #plot_reg_path_ax_lambdasearch_tangent(axes_all[i], replicates[i].coeffs, replicates[i].lambdas_plot * np.sqrt(m * nsel), fig)
#fig.savefig(folder + '/beta_paths')

with open(folder + '/replicates' + savename + '.pkl','wb') as output:
    pickle.dump(replicates, output, pickle.HIGHEST_PROTOCOL)
#



