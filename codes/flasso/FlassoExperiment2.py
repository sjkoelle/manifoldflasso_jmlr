# import matplotlib
# matplotlib.use('Agg')
import sklearn
import numpy as np
import matplotlib.pyplot as plt
import scipy
import matplotlib.colors
import spams
from collections import OrderedDict
from pylab import rcParams
from codes.flasso.GLMaccelerated import GLM
import scipy.sparse as ssp
rcParams['figure.figsize'] = 25, 10


def cosine_similarity(a, b):
    output = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return (output)


class FlassoExperiment:
    """
    FlassoExperiment
    """

    def __init__(self):
        2 + 2


    def _flatten_coefficient(self, coeff):
        n = coeff.shape[1]
        p = coeff.shape[2]
        q = coeff.shape[0]

        output = np.zeros((n * p * q))
        for k in range(q):
            for i in range(n):
                output[((k * n * p) + (i * p)):((k * n * p) + (i + 1) * p)] = coeff[k, i, :]
        return (output)

    def get_l2loss(self, coeffs, ys, xs):

        n = coeffs.shape[2]
        nlam = coeffs.shape[0]
        output = np.zeros(nlam)
        for i in range(nlam):
            coeffvec = self._flatten_coefficient(coeffs[i])
            output[i] = np.sum((ys - np.dot(coeffvec, xs.transpose())) ** 2)
        output = output / n
        return (output)

    def normalize(self, differential):

        n = differential.shape[0]
        gammas = ((1 / n) * np.linalg.norm(differential, axis=tuple([0, 2])))
        normed = np.einsum('n p d, p -> n p d', differential, gammas ** (-1))
        return(normed)


    def compute_penalty2(self, coeffs):
        n = coeffs.shape[2]
        nlam = coeffs.shape[0]
        q = coeffs.shape[1]
        p = coeffs.shape[3]

        # p = self.p
        pen = np.zeros(nlam)
        for l in range(nlam):
            norm2 = np.zeros(p)
            for j in range(p):
                norm2[j] = np.linalg.norm(coeffs[l, :, :, j])
            pen[l] = np.sum(norm2)
        pen = pen / n
        return (pen)

    def get_cosines(self, dg):
        n = dg.shape[0]
        p = dg.shape[1]
        d = dg.shape[2]

        coses = np.zeros((n, p, p))
        for i in range(n):
            for j in range(p):
                for k in range(p):
                    coses[i, j, k] = cosine_similarity(dg[i, j, :], dg[i, k,
                                                                    :])  # sklearn.metrics.pairwise.cosine_similarity(X = np.reshape(dg[:,i,:], (1,d*n)),Y = np.reshape(dg[:,j,:], (1,d*n)))[0][0]
        cos_summary = np.abs(coses).sum(axis=0) / n
        return (cos_summary)
