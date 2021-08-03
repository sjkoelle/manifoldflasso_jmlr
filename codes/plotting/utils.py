import numpy as np
from einops import rearrange
import matplotlib.pyplot as plt

def get_ordered_axes(replicates):

    nreps = len(list(replicates.keys()))
    for r in range(nreps):
        cs = rearrange(np.asarray(list(replicates[r].results[1].values())), 'l n p m -> l m n p')
        xaxis = np.asarray(np.asarray(list(replicates[r].results[1].keys())))
        xaxis_reorder = xaxis[xaxis.argsort()]
        cs_reorder = cs[xaxis.argsort()]
        xaxis_reorder = xaxis[xaxis.argsort()]
        replicates[r].cs = cs
        replicates[r].cs_reorder = cs_reorder
        replicates[r].xaxis_reorder = xaxis_reorder
        replicates[r].xaxis = xaxis
        
    return(replicates)

def get_names(subset):
    
    names = np.zeros(len(subset), dtype = object)
    for s in range(len(subset)):
        names[s] = r"$g_{{{}}}$".format(subset[s])
    return(names)

def get_color_subset(color_superset, superset, subset):
    
    colors_subset = np.zeros((len(subset),4 ))
    print(subset.shape,colors_subset.shape)
    for s in range(len(subset)):
        colors_subset[s] = color_superset[np.where(superset == subset[s])[0]] 
    return(colors_subset)



def get_cmap(subset):
    
    cmap = plt.get_cmap('rainbow',len(subset))
    colors_subset = np.zeros((len(subset),4))
    for s in range(len(subset)):
        colors_subset[s] = cmap(s)   
        
    return(colors_subset)
