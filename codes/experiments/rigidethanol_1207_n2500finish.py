
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
#from datetime import datetime

from shutil import copyfile
rcParams['figure.figsize'] = 25, 10

np.random.seed(0)
random.seed(0)
now = datetime.datetime.now().strftime("%B_%d_%Y_%H_%M_%S")
workingdirectory = os.popen('git rev-parse --show-toplevel').read()[:-1]
sys.path.append(workingdirectory)
os.chdir(workingdirectory)
from codes.experimentclasses.RigidEthanolPCA2 import RigidEthanolPCA2
from codes.otherfunctions.get_dictionaries import get_all_atoms_4
from codes.otherfunctions.get_grads import get_grads
from codes.otherfunctions.multirun import get_support_recovery_lambda
from codes.otherfunctions.multirun import get_lower_interesting_lambda
from codes.otherfunctions.multirun import get_coeffs_and_lambdas
from codes.otherfunctions.multirun import get_support
from codes.otherfunctions.multiplot import plot_support_2d
from codes.otherfunctions.multiplot import plot_reg_path_ax_lambdasearch
from codes.otherfunctions.multiplot import plot_gs_v_dgnorm
from codes.otherfunctions.multiplot import plot_dot_distributions
from codes.otherfunctions.multirun import get_cosines
from codes.flasso.Replicate import Replicate
from codes.otherfunctions.multirun import get_olsnorm_and_supportsbrute
from codes.otherfunctions.multiplot import highlight_cell

with open('/homes/sjkoelle/manifoldflasso_jmlr/untracked_data/embeddings/rigidethanol/rigidethanol_120220_samgl_n2500_pall_nrep5replicates.pkl' ,
         'rb') as loader:
     replicates = pickle.load(loader)

#selected_points_save = np.asarray(selected_points_save, dtype = int)
gl_itermax = 500
p =756
nsel = 2500
lambdas_start = [0.,.0005 * np.sqrt(nsel * p)]
max_search = 15
reg_l2 = 0.
card = 2
tol = 1e-14
learning_rate = 100

from pathos.multiprocessing import ProcessingPool as Pool
from codes.flasso.GradientGroupLasso import batch_stream, get_sr_lambda_sam_parallel

print('pre-gradient descent')
print(datetime.datetime.now())
cores = 16
pcor = Pool(cores)
result = get_sr_lambda_sam_parallel(replicates[0], gl_itermax, lambdas_start,reg_l2, max_search, card, tol,learning_rate)

with open('/homes/sjkoelle/manifoldflasso_jmlr/untracked_data/embeddings/rigidethanol/1207_2500result_finish.pkl' ,
         'wb') as output:
     pickle.dump(result, output, pickle.HIGHEST_PROTOCOL)

print('done')
print(datetime.datetime.now())

     