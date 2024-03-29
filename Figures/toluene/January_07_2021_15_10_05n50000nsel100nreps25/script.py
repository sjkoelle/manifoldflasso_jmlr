import matplotlib
matplotlib.use('Agg')
import os
import datetime
import numpy as np
import dill as pickle
import random
import sys
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
from collections import OrderedDict
import math
from matplotlib.lines import Line2D
from pylab import rcParams
from collections import Counter
from itertools import combinations
from shutil import copyfile
rcParams['figure.figsize'] = 25, 10


np.random.seed(0)
random.seed(0)
now = datetime.datetime.now().strftime("%B_%d_%Y_%H_%M_%S")
workingdirectory = os.popen('git rev-parse --show-toplevel').read()[:-1]
sys.path.append(workingdirectory)
os.chdir(workingdirectory)
from codes.otherfunctions.multiplot import highlight_cell
from codes.experimentclasses.TolueneAngles import TolueneAngles
#from codes.otherfunctions.multirun import get_coeffs_reps
#from codes.otherfunctions.multirun import get_grads_reps_pca2
from codes.otherfunctions.multiplot import plot_betas, plot_betas2reorder
from codes.geometer.RiemannianManifold import RiemannianManifold
from codes.otherfunctions.get_dictionaries import get_atoms_4
from codes.otherfunctions.get_grads import get_grads
from codes.otherfunctions.multirun import get_support_recovery_lambda
from codes.otherfunctions.multirun import get_lower_interesting_lambda
from codes.otherfunctions.multirun import get_coeffs_and_lambdas
from codes.otherfunctions.multirun import get_support
from codes.otherfunctions.multiplot import plot_support_1d
from codes.otherfunctions.multiplot import plot_reg_path_ax_lambdasearch
from codes.otherfunctions.multiplot import plot_gs_v_dgnorm
from codes.otherfunctions.multiplot import plot_dot_distributions
from codes.otherfunctions.multirun import get_olsnorm_and_supportsbrute_1d
from codes.otherfunctions.multirun import get_cosines
from codes.geometer.ShapeSpace import ShapeSpace
from codes.geometer.TangentBundle import TangentBundle
from codes.flasso.Replicate import Replicate

#set parameters
n = 50000 #number of data points to simulate
nsel = 100 #number of points to analyze with lasso
nreps = 25
itermax = 1000 #maximum iterations per lasso run
tol = 1e-10 #convergence criteria for lasso
#lambdas = np.asarray([0,.01,.1,1,10,100], dtype = np.float16)#lambda values for lasso
lambdas = np.asarray(np.hstack([np.asarray([0]),np.logspace(-3,1,11)]), dtype = np.float16)
n_neighbors = 1000 #number of neighbors in megaman
m = 2 #number of embedding dimensions (diffusion maps)
#diffusion_time = 1. #diffusion time controls gaussian kernel radius per gradients paper
diffusion_time = 1. #(yuchia suggestion)
dim = 1 #manifold dimension
dimnoise = 1
cores = 3 #number of cores for parallel processing
cor = 0.0 #correlation for noise
var = 0.00001 #variance scaler for noise
ii = np.asarray([0, 0, 0, 0, 1, 6, 5, 6, 5, 4, 4, 3, 3, 2, 2])
jj = np.asarray([8, 9, 7, 1, 6, 14, 13, 5, 4, 12, 3, 11, 2, 10, 1])

#these are just for loading... probably not necessary
atoms4 = np.asarray([[9,0,1,2],[0,1,2,3],[1,2,3,4],[2,3,4,5],[3,4,5,6],[4,5,6,1],[5,6,1,0],[0,1,3,11],[10,2,4,12],[11,3,5,13],[12,4,6,14],[10,2,6,13],[0,1,5,13],[11,3,6,14],[12,4,1,0],[10,2,5,13]],dtype = int)
lambda_max = 1
max_search = 30
new_MN = True
savename = 'toluene_110120'
savefolder = 'toluene'
loadfolder = 'toluene'
loadname = 'toluene_110120'

folder = workingdirectory + '/Figures/toluene/' + now + 'n' + str(n) + 'nsel' + str(nsel) + 'nreps' + str(nreps)
os.mkdir(folder)
src = workingdirectory + '/codes/experiments/toluene_110120_nsel100_nreps25.py'
filenamescript = folder + '/script.py'
copyfile(src, filenamescript)

if new_MN == True:
    experiment = TolueneAngles(dim, n,ii, jj, cores, atoms4)
    projector = np.load(workingdirectory + '/untracked_data/chemistry_data/tolueneangles020619_pca50_components.npy')
    experiment.M = experiment.load_data()  # if noise == False then noise parameters are overriden
    experiment.Mpca = RiemannianManifold(
        np.load(workingdirectory + '/untracked_data/chemistry_data/tolueneangles020619_pca50.npy'), dim)
    experiment.q = m
    experiment.m = m
    experiment.dimnoise = dimnoise
    experiment.Mpca.geom = experiment.Mpca.compute_geom(diffusion_time, n_neighbors)
    experiment.N = experiment.Mpca.get_embedding3(experiment.Mpca.geom, m, diffusion_time, dim)
#     with open(workingdirectory +
#             '/untracked_data/embeddings/' + savefolder + '/' + savename + '.pkl' ,
#             'wb') as output:
#         pickle.dump(experiment, output, pickle.HIGHEST_PROTOCOL)

atoms4,p = get_atoms_4(15,ii,jj)
experiment.p = p
experiment.atoms4 = atoms4
experiment.itermax = itermax
experiment.tol = tol
experiment.projector = projector
experiment.dnoise = dim
experiment.nreps = nreps
experiment.nsel = nsel
experiment.folder = folder

replicates = {}
selected_points_save = np.zeros((nreps,nsel))
for i in range(nreps):
    selected_points = np.random.choice(list(range(n)),nsel,replace = False)
    selected_points_save[i] = selected_points
    replicates[i] = Replicate()
    replicates[i].nsel = nsel
    replicates[i].selected_points = selected_points
    replicates[i].df_M,replicates[i].dg_M,replicates[i].dg_w ,replicates[i].dg_w_pca ,replicates[i].dgw_norm  = get_grads(experiment, experiment.Mpca, experiment.M, experiment.N, selected_points)
    replicates[i].xtrain, replicates[i].groups = experiment.construct_X_js(replicates[i].dg_M)
    replicates[i].ytrain = experiment.construct_Y_js(replicates[i].df_M,dimnoise)
    replicates[i].coeff_dict = {}
    replicates[i].coeff_dict[0] = experiment.get_betas_spam2(replicates[i].xtrain, replicates[i].ytrain, replicates[i].groups, np.asarray([0]), nsel, experiment.m, itermax, tol)
    replicates[i].combined_norms = {}
    replicates[i].combined_norms[0] = np.linalg.norm(np.linalg.norm(replicates[i].coeff_dict[0][:, :, :, :], axis=2), axis=1)[0,:]
    replicates[i].higher_lambda,replicates[i].coeff_dict,replicates[i].combined_norms = get_support_recovery_lambda(experiment, replicates[i],  lambda_max, max_search,dim)
    replicates[i].lower_lambda,replicates[i].coeff_dict,replicates[i].combined_norms = get_lower_interesting_lambda(experiment, replicates[i],  lambda_max, max_search)
    #= experiment.get_betas_spam2(replicates[i].xtrain, replicates[i].ytrain, replicates[i].groups, lambdas, len(selected_points), n_embedding_coordinates, itermax, tol)


fig, axes_all = plt.subplots(nreps, m + 1,figsize=(15 * m, 15*nreps))
fig.suptitle('Regularization paths')
for i in range(nreps):
    replicates[i].coeffs, replicates[i].lambdas_plot = get_coeffs_and_lambdas(replicates[i].coeff_dict, replicates[i].lower_lambda, replicates[i].higher_lambda)
    plot_reg_path_ax_lambdasearch(axes_all[i], replicates[i].coeffs, replicates[i].lambdas_plot * np.sqrt(m * nsel), fig)
fig.savefig(folder + '/beta_paths')

print(workingdirectory +
        '/untracked_data/embeddings/' + savefolder + '/' + savename + 'replicates.pkl')
with open(workingdirectory +
        '/untracked_data/embeddings/' + savefolder + '/' + savename + 'replicates.pkl' ,
        'wb') as output:
    pickle.dump(replicates, output, pickle.HIGHEST_PROTOCOL)

print(2+2)
#supports = {}
#for i in range(nreps):
#    supports[i] = get_support(replicates[i].coeffs, dim)

#fig, ax = plt.subplots(figsize=(15 , 15 ))
#fig, ax = plt.figure(figsize=(15 , 15 ))
#plot_support_1d(supports, experiment.p)
fig.savefig(folder + '/flasso_support')


ols_norm, supports_brute = get_olsnorm_and_supportsbrute_1d(experiment,replicates)



fig, axes_all = plt.subplots(nreps,figsize=(15*nreps,15))
fig.suptitle('GL norm for different OLS solutions')
for r in range(nreps):
    axes_all[r].imshow(np.log(np.expand_dims(ols_norm[r],1)))
    highlight_cell(0,supports_brute[r][0],color="limegreen", linewidth=3,ax=axes_all[r])
fig.savefig(folder + '/olsnorms')

fig, ax = plt.subplots(figsize=(15 , 15 ))
#fig, ax = plt.figure(figsize=(15 , 15 ))
plot_support_1d(supports_brute, experiment.p)
fig.savefig(folder + '/ols_supports')

fig, axes = plt.subplots(nreps, p, figsize=(15 * p, 15 * nreps))
plot_gs_v_dgnorm(experiment,replicates, axes)
fig.savefig(folder + '/gs_v_dgnorm.png')

plot_dot_distributions(experiment,replicates)