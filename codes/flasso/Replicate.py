import numpy as np
from einops import rearrange

def get_detected_values2d(subset, supports,nreps):
    
    detected_values = np.zeros((subset.shape[0],subset.shape[0]))
    nreps = len(list(supports.keys()))
    for r in range(nreps):
        i1 = np.where(subset == int(supports[r][0]))[0]
        i2 = np.where(subset == int(supports[r][1]))[0]
        detected_values[i1,i2] += 1
        detected_values[i2,i1] += 1
    
    detected_values = np.asarray(np.where(detected_values > 0))
    return(detected_values)

def get_support_indices2d(sup_set, sup_sel,nreps):
    
    #toplot = np.zeros(np.repeat(sup_set.shape[0], d ))
    toplot = np.zeros((sup_set.shape[0],sup_set.shape[0]))
    for r in range(nreps):
        i1 = np.where(sup_set == sup_sel[r][0])[0]
        i2 = np.where(sup_set == sup_sel[r][1])[0]
        toplot[i1,i2] += 1
        toplot[i2,i1] += 1   
    return(toplot)

class Replicate():

    def __init__(self, nsel = None, n = None, selected_points = None):

        self.nsel = nsel
        if selected_points is not None:
            self.selected_points = selected_points
        else:
            self.selected_points = np.random.choice(list(range(n)), nsel, replace=False)


    def get_ordered_axes(self):
        replicate = self
        cs = rearrange(np.asarray(list(replicate.results[1].values())), 'l n p m -> l m n p')
        xaxis = np.asarray(np.asarray(list(replicate.results[1].keys())))
        xaxis_reorder = xaxis[xaxis.argsort()]
        cs_reorder = cs[xaxis.argsort()]
        xaxis_reorder = xaxis[xaxis.argsort()]
        replicate.cs = cs
        replicate.cs_reorder = cs_reorder
        replicate.xaxis_reorder = xaxis_reorder
        replicate.xaxis = xaxis

    def get_selection_lambda(self):

        replicate = self
        lambdas = np.asarray(list(replicate.results[1].keys()))
        lambdas.sort()
        sel_l = np.where(lambdas ==  replicate.results[0])[0][0]
        return(sel_l)
