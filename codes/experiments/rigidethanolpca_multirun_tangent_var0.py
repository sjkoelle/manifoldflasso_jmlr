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
from codes.experimentclasses.RigidEthanolPCA import RigidEthanolPCA
from codes.otherfunctions.multirun import get_coeffs_reps_tangent
from codes.otherfunctions.multirun import get_grads_reps_pca2_tangent
from codes.otherfunctions.multiplot import plot_reg_path_ax_lambdasearch_tangent
from codes.otherfunctions.get_dictionaries import get_atoms_4
from codes.flasso.Replicate import Replicate
from codes.otherfunctions.get_grads_tangent import get_grads_tangent
from codes.otherfunctions.multirun import get_support_recovery_lambda
from codes.otherfunctions.multirun import get_lower_interesting_lambda
import matplotlib.pyplot as plt
from codes.otherfunctions.multirun import get_coeffs_and_lambdas
from codes.geometer.RiemannianManifold import RiemannianManifold

#set parameters
n = 10000 #number of data points to simulate
nsel = 50 #number of points to analyze with lasso
itermax = 1000 #maximum iterations per lasso run
tol = 1e-10 #convergence criteria for lasso
#lambdas = np.asarray([0,.01,.1,1,10,100], dtype = np.float16)#lambda values for lasso
#lambdas = np.asarray(np.hstack([np.asarray([0]),np.logspace(-3,1,11)]), dtype = np.float16)
#lambdas = np.asarray(np.hstack([np.asarray([0]),np.logspace(-3,0,7), np.logspace(0,2,5),np.logspace(2,3,2)]), dtype = np.float16)
lambdas = np.asarray(np.hstack([np.asarray([0]),np.logspace(-2,1,15)]), dtype = np.float16)
n_neighbors = 1000 #number of neighbors in megaman
n_components = 3 #number of embedding dimensions (diffusion maps)
#diffusion_time = 1. #diffusion time controls gaussian kernel radius per gradients paper
#diffusion_time =.05 #(yuchia suggestion)
diffusion_time =.25 #(yuchia suggestion)
dim = 4 #manifold dimension
dimnoise = 4
cores = 3 #number of cores for parallel processing
cor = 0.0 #correlation for noise
var = 0.00001 #variance scaler for noise
ii = np.asarray([0,0,0,0,1,1,1,2]) # atom adjacencies for dihedral angle computation
jj = np.asarray([1,2,3,4,5,6,7,8])

#run experiment
atoms4 = np.asarray([[6,1,0,4],[4,0,2,8],[7,6,5,1],[3,0,2,4]],dtype = int)

m = 3
new_MN = True
new_grad = True
savename = 'rigidethanol_032520'
savefolder = 'rigidethanol'
loadfolder = 'rigidethanol'
loadname = 'rigidethanol_032520'
nreps = 25
atoms4,p = get_atoms_4(9,ii,jj)
folder = workingdirectory + '/Figures/rigidethanol/' + now + 'n' + str(n) + 'nsel' + str(nsel) + 'nreps' + str(nreps)
os.mkdir(folder)

if new_MN == True:
    experiment = RigidEthanolPCA(dim, cor,var,ii,jj, cores, False, atoms4)
    #projector  = np.load(workingdirectory + '/untracked_data/chemistry_data/ethanolangles022119_pca50_components.npy')
    #experiment.M = experiment.load_data()  # if noise == False then noise parameters are overriden
	#experiment.Mpca = RiemannianManifold(np.load(workingdirectory + '/untracked_data/chemistry_data/ethanolangles022119_pca50.npy'), dim)
    experiment.M, experiment.Mpca, projector = experiment.generate_data(noise=False)
    experiment.q = m
    experiment.m = m
    experiment.dimnoise = dimnoise
    experiment.projector = projector
    experiment.Mpca.geom = experiment.Mpca.compute_geom(diffusion_time, n_neighbors)
    experiment.N = experiment.Mpca.get_embedding3(experiment.Mpca.geom, m, diffusion_time, dim)
    with open(workingdirectory + '/untracked_data/embeddings/' + savefolder + '/' + savename + '.pkl' ,
             'wb') as output:
         pickle.dump(experiment, output, pickle.HIGHEST_PROTOCOL)

lambda_max = 1
max_search = 30

experiment.p = p# + experiment.d
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
    replicates[i].ytrain = experiment.construct_Y(replicates[i].df_M,list(range(nsel)))
    replicates[i].coeff_dict = {}
    replicates[i].coeff_dict[0] = experiment.get_betas_spam2(replicates[i].xtrain, replicates[i].ytrain, replicates[i].groups, np.asarray([0]), nsel, experiment.dim, itermax, tol)
    replicates[i].combined_norms = {}
    replicates[i].combined_norms[0] = np.linalg.norm(np.linalg.norm(replicates[i].coeff_dict[0][:, :, :, :], axis=2), axis=1)[0,:]
    replicates[i].higher_lambda,replicates[i].coeff_dict,replicates[i].combined_norms = get_support_recovery_lambda(experiment, replicates[i],  lambda_max, max_search,dim)
    replicates[i].lower_lambda,replicates[i].coeff_dict,replicates[i].combined_norms = get_lower_interesting_lambda(experiment, replicates[i],  lambda_max, max_search)
    #= experiment.get_betas_spam2(replicates[i].xtrain, replicates[i].ytrain, replicates[i].groups, lambdas, len(selected_points), n_embedding_coordinates, itermax, tol)

#fig, axes_all = plt.subplots(nreps, 1,figsize=(15 * m, 15*nreps))
#fig.suptitle('Regularization paths')
for i in range(nreps):
    replicates[i].coeffs, replicates[i].lambdas_plot = get_coeffs_and_lambdas(replicates[i].coeff_dict, replicates[i].lower_lambda, replicates[i].higher_lambda)
#    plot_reg_path_ax_lambdasearch_tangent(axes_all[i], replicates[i].coeffs, replicates[i].lambdas_plot * np.sqrt(m * nsel), fig)
#fig.savefig(folder + '/beta_paths')

with open(folder + '/replicates' + savename + '.pkl','wb') as output:
    pickle.dump(replicates, output, pickle.HIGHEST_PROTOCOL)
#experiment = RigidEthanolPCA(dim, cor,var,ii,jj, cores, False, atoms4)
#experiment.M, experiment.Mpca,projector  = experiment.generate_data(noise = False)
#experiment.q = n_components
#experiment.dimnoise = dimnoise
#experiment.Mpca.geom = experiment.Mpca.compute_geom(diffusion_time, n_neighbors)
#experiment.N = experiment.Mpca.get_embedding3(experiment.Mpca.geom, n_components, diffusion_time, dim)
#experiment.g0 = experiment.get_g_full_sub(experiment.M.data,experiment.atoms4[0])
#experiment.g1 = experiment.get_g_full_sub(experiment.M.data,experiment.atoms4[1])
#folder = workingdirectory + '/Figures/rigidethanol/' + now
#os.mkdir(folder)
#experiment.N.plot([0,1,2], list(range(n)),experiment.g0,.1,.1, folder + '/g1')
#experiment.N.plot([0,1,2], list(range(n)),experiment.g1,.1,.1, folder + '/g2')

#experiment.M.selected_points = np.random.choice(list(range(n)),nsel,replace = False)
#import pickle
#with open('ethanolsavegeom1.pkl', 'wb') as output:
#    pickle.dump(experiment.N, output, pickle.HIGHEST_PROTOCOL)
# print('pregrad',datetime.datetime.now().strftime("%B_%d_%Y_%H_%M_%S"))
# experiments = get_grads_reps_pca2_tangent(experiment, nreps, nsel,cores,projector)
# #experiments.df_M
# #with open('tolueneexperiments0306_3custom_1000.pkl', 'wb') as output:
# #    pickle.dump(experiments, output, pickle.HIGHEST_PROTOCOL)
# print('precoeff',datetime.datetime.now().strftime("%B_%d_%Y_%H_%M_%S"))
# experiments = get_coeffs_reps_tangent(experiments, nreps, lambdas, itermax,nsel,tol)
# xaxis = np.asarray(lambdas, dtype = np.float64) * np.sqrt(n * n_components)
# title ='RigidEthanolTangent'
# gnames = np.asarray([r"$\displaystyle g_1$",
# 	r"$\displaystyle g_2$",
# 	r"$\displaystyle g_3$",
# 	r"$\displaystyle g_4$"])
# #gnames = np.asarray(list(range(experiment.p)), dtype = str)
# #folder = workingdirectory + '/Figures/rigidethanol/' + now
# #os.mkdir(folder)
# filename = folder + '/betas'
# print('preplot',datetime.datetime.now().strftime("%B_%d_%Y_%H_%M_%S"))
# plot_betas2reorder_tangent(experiments, xaxis, title,filename, gnames,nsel)
# for j in range(nreps):
#     np.save(folder+'/coeffs' + 'rep'+ str(j) + 'var'+str(var), experiments[j].coeffs)
# filenamescript = folder + '/script.py'
# from shutil import copyfile
# src = workingdirectory + '/codes/experiments/rigidethanolpca_multirun_tangent_var0.py'
# copyfile(src, filenamescript)
