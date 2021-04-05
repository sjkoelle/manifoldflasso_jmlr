
import autograd.numpy as np
from codes.flasso.FlassoExperiment2 import FlassoExperiment
from sklearn.linear_model import LinearRegression

class FlassoManifold(FlassoExperiment):
    """
    This class provides methods for estimating Df and projecting into tangent spaces with Riemannian metric where f is a data reparameterization
    Parameters
    ----------
    n : int,
        Number of samples
    selectedpoints : np.array(dtype = int),
        Which points to pass to function regression
    dim : int,
        dimension of manifold
    data:
    embedding:
    Dg : function,
        the differential of g
    Optional Parameters
    ----------
    tangent_bundle: np.array((n, d, dim), dtype = float)
        Bases for tangent bundle
    Methods
    -------
    get_flat_tangent_bundle(self),
        get tangent bundle with flat bases
    estimate_Df(self, tangent_bases),
        estimates Df in given tangent bases
    get_rmetric(self, embedding, tangent_bases)
        estimates rmetric in given embedding and tangent_bases
    """

    #A FlassoManifold experiment explains the results of a manifold embedding by a FlassoExperiment
    #The key method is the embedding gradient estimation method (gets dF)
    def __init__(self):
        2+2

    def project(self, tangent_bases, vectors):
        # n = vectors.shape[0]
        # p = vectors.shape[1]
        # dim = tangent_bases.shape[2]

        # dg_M = np.zeros((n,p,dim))
        #
        # for i in range(n):
        #     dg_M[i] = np.matmul(tangent_bases[i].transpose(), vectors[i].transpose()).transpose()

        dg_M = np.einsum('n b d, n p b -> n p d', tangent_bases, vectors)

        return(dg_M)


    def get_dF_js_idM(self, M, N, M_tangent_bundle_sub, N_tangent_bundle, selectedpoints, dim = None):

        if dim == None:
            dim = self.dim
        m = self.m
        affinity_matrix = M.geom.affinity_matrix

        nsel = len(selectedpoints)
        dF = np.zeros((nsel, m, dim))

        for i in range(nsel):
            pt = selectedpoints[i]
            neighborspt = affinity_matrix[selectedpoints[i]].indices
            deltap0 = M.data[neighborspt, :] - M.data[pt, :]
            deltaq0 = N.data[neighborspt, :] - N.data[pt, :]
            projected_M = np.matmul(M_tangent_bundle_sub.tangent_bases[i, :, :].transpose(),
                                    deltap0.transpose()).transpose()
            tan_proj = np.dot(N_tangent_bundle.tangent_bases[selectedpoints[i]],N_tangent_bundle.tangent_bases[selectedpoints[i]].transpose())
            projected_N = np.dot(deltaq0, tan_proj)

            lr = LinearRegression()
            weights = affinity_matrix[selectedpoints[i]].data
            lr.fit(projected_M, projected_N, weights)
            dF[i, :, :][:, :] = lr.coef_#np.linalg.lstsq(projected_M, deltaq0)[0]#np.matmul(a, b).transpose()
        return (dF)


    def get_central_point(self, nsel, fitmodel, data):
        centralpoint = np.zeros(nsel)
        for i in range(nsel):
            # print(i)
            distomean = np.zeros(len(fitmodel.adjacency_matrix[i].indices))
            meanpoint = data[fitmodel.adjacency_matrix[i].indices, :].mean(axis=0)
            for j in range(len(fitmodel.adjacency_matrix[i].indices)):
                distomean[j] = np.linalg.norm(data[fitmodel.adjacency_matrix[i].indices[j], :] - meanpoint)
            centralpoint[i] = fitmodel.adjacency_matrix[i].indices[distomean.argmin()]
        centralpoint = np.asarray(centralpoint, dtype=int)
        return (centralpoint)
