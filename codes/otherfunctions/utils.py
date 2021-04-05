import numpy as np

def cosine_similarity(a, b):
    output = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return (output)

# def get_cosines(self, dg):
def get_cosines(dg):
    n = dg.shape[0]
    p = dg.shape[1]
    d = dg.shape[2]

    coses = np.zeros((n, p, p))
    for i in range(n):
        for j in range(p):
            for k in range(p):
                coses[i, j, k] = cosine_similarity(dg[i, j, :], dg[i, k,:])  # sklearn.metrics.pairwise.cosine_similarity(X = np.reshape(dg[:,i,:], (1,d*n)),Y = np.reshape(dg[:,j,:], (1,d*n)))[0][0]
    # cos_summary = np.abs(coses).sum(axis = 0) / n
    #cos_summary = np.sum(coses ** 2, axis=0) / n
    return (coses)

#def get_mu(dg_M):

def cosine_similarity(a, b):
    output = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return (output)


def get_kappa_s(dg_M):
    n = dg_M.shape[0]
    d = dg_M.shape[1]
    p = dg_M.shape[2]
    maxes = np.zeros(n)
    # coses_zerodiag = coses.copy()
    kappa_ij = np.zeros((n, p))
    for i in range(n):
        kappa_ij[i] = np.linalg.norm(dg_M[i], axis=0)
    kappa_i = np.max(kappa_ij, axis=1) / np.min(kappa_ij, axis=1)
    kappa = np.max(kappa_i)
    return (kappa)


def get_coses_full_ind(dg, ind):
    n = dg.shape[0]
    p = dg.shape[1]
    d = dg.shape[2]
    l = len(ind)
    coses = np.zeros((n, l, p))

    for i in range(n):
        # print(i)
        for j in range(l):
            for k in range(p):
                if ind[j] != k:
                    coses[i, j, k] = cosine_similarity(dg[i, ind[j], :], dg[i, k,
                                                                         :])  # sklearn.metrics.pairwise.cosine_similarity(X = np.reshape(dg[:,i,:], (1,d*n)),Y = np.reshape(dg[:,j,:], (1,d*n)))[0][0]
    # cos_summary = np.abs(coses).sum(axis = 0) / n
    # cos_summary = np.sum(coses ** 2, axis=0) / n
    return (coses)


def get_mu_full_ind(dg, ind):
    n = dg.shape[0]
    p = dg.shape[1]
    d = dg.shape[2]
    l = len(ind)
    coses = np.zeros((n, l, p))
    for i in range(n):
        # print(i)
        for j in range(l):
            for k in range(p):
                if ind[j] != k:
                    coses[i, j, k] = cosine_similarity(dg[i, ind[j], :], dg[i, k,
                                                                         :])  # sklearn.metrics.pairwise.cosine_similarity(X = np.reshape(dg[:,i,:], (1,d*n)),Y = np.reshape(dg[:,j,:], (1,d*n)))[0][0]
    # cos_summary = np.abs(coses).sum(axis = 0) / n
    # cos_summary = np.sum(coses ** 2, axis=0) / n
    return (coses.max())


def get_min_min(dg_M):
    n = dg_M.shape[0]
    d = dg_M.shape[1]
    p = dg_M.shape[2]
    maxes = np.zeros(n)
    # coses_zerodiag = coses.copy()
    kappa_ij = np.zeros((n, p))
    for i in range(n):
        kappa_ij[i] = np.linalg.norm(dg_M[i], axis=0)
    min_min = np.min(kappa_ij)
    return (min_min)


def get_gamma_max(dg_M):
    output = np.sum(np.sum(dg_M ** 2, axis=1), axis=0).max()
    return (output)

#def get_gamma_max():

