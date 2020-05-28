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
from codes.experimentclasses.TolueneAngles import TolueneAngles
from codes.otherfunctions.multirun import get_coeffs_reps_tangent
from codes.otherfunctions.multirun import get_grads_reps_pca2_tangent
from codes.otherfunctions.multiplot import plot_betas, plot_betas2reorder_tangent
from codes.geometer.RiemannianManifold import RiemannianManifold

#set parameters
n = 50000 #number of data points to simulate
nsel = 50 #number of points to analyze with lasso
itermax = 100000 #maximum iterations per lasso run
tol = 1e-10 #convergence criteria for lasso
#lambdas = np.asarray([5,10,15,20,25,50,75,100], dtype = np.float16)#lambda values for lasso
#lambdas = np.asarray([0,2.95339658e-06, 5.90679317e-06, 8.86018975e-06, 1.18135863e-05,
#       1.47669829e-05, 2.95339658e-05, 4.43009487e-05, 5.90679317e-05])
#lambdas = np.asarray([0,.001,.01,.1,1,10], dtype = np.float16)
lambdas = np.asarray(np.hstack([np.asarray([0]),np.logspace(-3,-1,11)]), dtype = np.float16)
#lambdas = np.asarray([0,1,2,3,4,5,6,7,8,9,10], dtype = np.float16)
n_neighbors = 1000 #number of neighbors in megaman
n_components = 2 #number of embedding dimensions (diffusion maps)
diffusion_time = 2.23 #diffusion time controls gaussian kernel radius per gradients paper
dim = 1 #slow manifold dimension
dimnoise = 1 #noise dimension
cores = 16 #number of cores for parallel processing
ii = np.asarray([0, 0, 0, 0, 1, 6, 5, 6, 5, 4, 4, 3, 3, 2, 2])
jj = np.asarray([8, 9, 7, 1, 6, 14, 13, 5, 4, 12, 3, 11, 2, 10, 1])

#run experiment
#atoms4 = np.asarray([[9,0,1,2],[0,1,2,3],[1,2,3,4],[2,3,4,5],[3,4,5,6],[4,5,6,1],[5,6,1,0]],dtype = int)
atoms4 = np.asarray([[9,0,1,2],[0,1,2,3],[1,2,3,4],[2,3,4,5],[3,4,5,6],[4,5,6,1],[5,6,1,0],[0,1,3,11],[10,2,4,12],[11,3,5,13],[12,4,6,14],[10,2,6,13],[0,1,5,13],[11,3,6,14],[12,4,1,0],[10,2,5,13]],dtype = int)
nreps = 5

folder = workingdirectory + '/Figures/toluene/' + now
os.mkdir(folder)

load_M = False
load_N = False
load_exp = False

if load_exp == True:
    # experiment = EthanolAngles(dim, ii,jj,cores, atoms4)
    file = open(workingdirectory + '/untracked_data/embeddings/toluene/toluene_012420.pkl',
        'rb')
    experiment = pickle.load(file)
    file.close()

new_MN = False
savename = 'toluene_013020'
savefolder = 'toluene'

loadname = 'toluene_013020'
loadfolder = 'toluene'
if new_MN == True:
    experiment = TolueneAngles(dim, n,ii, jj, cores, atoms4)
    projector = np.load(workingdirectory + '/untracked_data/chemistry_data/tolueneangles020619_pca50_components.npy')
    experiment.M = experiment.load_data()  # if noise == False then noise parameters are overriden
    experiment.Mpca = RiemannianManifold(
        np.load(workingdirectory + '/untracked_data/chemistry_data/tolueneangles020619_pca50.npy'), dim)
    experiment.q = n_components
    experiment.dimnoise = dimnoise
    experiment.Mpca.geom = experiment.Mpca.compute_geom(diffusion_time, n_neighbors)
    experiment.N = experiment.Mpca.get_embedding3(experiment.Mpca.geom, n_components, diffusion_time, dim)
    with open(workingdirectory + 
            '/untracked_data/embeddings/' + savefolder + '/' + savename + '.pkl' ,
            'wb') as output:
        pickle.dump(experiment, output, pickle.HIGHEST_PROTOCOL)

if new_MN == False:
    file = open(workingdirectory + '/untracked_data/embeddings/' + loadfolder + '/' + loadname + '.pkl', 'rb')
    experiment = pickle.load(file)
    projector = np.load(workingdirectory + '/untracked_data/chemistry_data/tolueneangles020619_pca50_components.npy')
    file.close()

print('pregrad',datetime.datetime.now().strftime("%B_%d_%Y_%H_%M_%S"))
new_grad = True
if new_grad == True:
    experiments = get_grads_reps_pca2_tangent(experiment, nreps, nsel,cores,projector)
    experiments = get_coeffs_reps_tangent(experiments, nreps, lambdas, itermax, nsel, tol)
    with open(workingdirectory + '/untracked_data/embeddings/' + savefolder + '/' + savename + 's.pkl' , 'wb') as output:
        pickle.dump(experiments, output, pickle.HIGHEST_PROTOCOL)

if new_grad == False:
    file = open(workingdirectory + '/untracked_data/embeddings/' + loadfolder + '/' + loadname + 's.pkl', 'rb')
    experiments = pickle.load(file)
    projector = np.load(workingdirectory + '/untracked_data/chemistry_data/tolueneangles020619_pca50_components.npy')
    file.close()

xaxis = np.asarray(lambdas, dtype = np.float64) * np.sqrt(n * dim)
title ='Toluene'
#gnames = np.asarray(list(range(experiment.p)), dtype = str)
gnames = np.asarray([r"$\displaystyle g_1$",
	r"$\displaystyle g_2$",
	r"$\displaystyle g_3$",
	r"$\displaystyle g_4$",
	r"$\displaystyle g_5$",
	r"$\displaystyle g_6$",
	r"$\displaystyle g_7$",
	r"$\displaystyle g_8$",
	r"$\displaystyle g_9$",
	r"$\displaystyle g_{10}$",
	r"$\displaystyle g_{11}$",
	r"$\displaystyle g_{12}$",
	r"$\displaystyle g_{13}$",
	r"$\displaystyle g_{14}$",
	r"$\displaystyle g_{15}$",
	r"$\displaystyle g_{16}$"])
filename = folder + '/betas_tangent'
print('preplot',datetime.datetime.now().strftime("%B_%d_%Y_%H_%M_%S"))
plot_betas2reorder_tangent(experiments, xaxis, title,filename, gnames,nsel)
filenamescript = folder + '/script.py'
experiment.gs = experiment.get_g_full(experiment.M.data)
np.save(folder+'/gs', experiment.gs)

experiment.N.plot([0,1], list(range(n)),experiment.gs[:,0],.1,.1, folder + '/g1')

for i in range(nreps):
    np.save(folder + '/coeffs' + str(i), experiments[i].coeffs)

np.save(folder + '/embedding_data', experiment.N.data)
from shutil import copyfile
src = workingdirectory + '/codes/experiments/toluenepca_multirun_tangent_2.py'
copyfile(src, filenamescript)
