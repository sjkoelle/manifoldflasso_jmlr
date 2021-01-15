from codes.geometer.RiemannianManifold import RiemannianManifold
from codes.geometer.ShapeSpace import ShapeSpace
from codes.geometer.TangentBundle import TangentBundle
import numpy as np
import copy
from collections import Counter
from itertools import combinations
#
# def get_grads_reps(experiment, nreps, nsel,cores):
#
#     experiments = {}
#     dim = experiment.dim
#
#     for i in range(nreps):
#         experiments[i] = copy.copy(experiment)
#         experiments[i].M.selected_points = np.random.choice(list(range(experiment.n)), nsel, replace=False)
#         tangent_bases = experiments[i].M.get_wlpca_tangent_sel(experiments[i].M, experiments[i].M.selected_points)
#         subM = RiemannianManifold(experiments[i].M.data[experiments[i].M.selected_points], dim)
#         subM.tb = TangentBundle(subM, tangent_bases)
#         experiments[i].N.tangent_bundle = TangentBundle(experiments[i].N, experiments[i].N.geom.rmetric.Hvv)
#         experiments[i].df_M = experiments[i].get_dF_js_idM(experiments[i].M, experiments[i].N, subM.tb, experiments[i].N.tangent_bundle,
#                                                    experiments[i].M.selected_points)
#         #experiments[i].df_M = experiments[i].df_M / np.linalg.norm(experiments[i].df_M, axis=1).sum(axis=0)
#         experiments[i].df_M2 = experiments[i].df_M / np.linalg.norm(experiments[i].df_M) ** 2
#         experiments[i].dg_x = experiments[i].get_dx_g_full(experiments[i].M.data[experiments[i].M.selected_points])
#         experiments[i].W = ShapeSpace(experiments[i].positions, experiments[i].M.data)
#         experiments[i].dw = experiments[i].W.get_dw(cores, experiments[i].atoms3, experiments[i].natoms, experiments[i].M.selected_points)
#         experiments[i].dg_w = experiments[i].project(experiments[i].dw, experiments[i].dg_x)
#         tb_w_tangent_bases = experiments[i].project(experiments[i].dw, np.swapaxes(subM.tb.tangent_bases, 1, 2))
#         experiments[i].dw_norm = experiments[i].normalize(experiments[i].dg_w)
#         experiments[i].dg_M = experiments[i].project(np.swapaxes(tb_w_tangent_bases, 1, 2), experiments[i].dw_norm)
#
#         #experiments[i].coeffs = experiments[i].get_betas_spam2(experiments[i].xtrain, experiments[i].ytrain, experiments[i].groups, lambdas,
#         #                                              nsel, experiments[i].q, itermax, tol)
#     return(experiments)
#
# def get_grads_reps_pca(experiment, nreps, nsel,cores, projector):
#
#     experiments = {}
#     dim = experiment.dim
#     dimnoise = experiment.dimnoise
#     for i in range(nreps):
#         experiments[i] = copy.copy(experiment)
#         experiments[i].M.selected_points = np.random.choice(list(range(experiment.n)), nsel, replace=False)
#         tangent_bases = experiments[i].M.get_wlpca_tangent_sel(experiments[i].Mpca, experiments[i].M.selected_points,dimnoise)
#         subM = RiemannianManifold(experiments[i].Mpca.data[experiments[i].M.selected_points], dim)
#         subM.tb = TangentBundle(subM, tangent_bases)
#         experiments[i].N.tangent_bundle = TangentBundle(experiments[i].N, experiments[i].N.geom.rmetric.Hvv)
#         experiments[i].df_M = experiments[i].get_dF_js_idM(experiments[i].Mpca, experiments[i].N, subM.tb, experiments[i].N.tangent_bundle,
#                                                    experiments[i].M.selected_points,dimnoise)
#         #experiments[i].df_M = experiments[i].df_M / np.linalg.norm(experiments[i].df_M, axis=1).sum(axis=0)
#         experiments[i].df_M2 = experiments[i].df_M / np.linalg.norm(experiments[i].df_M) ** 2
#         experiments[i].dg_x = experiments[i].get_dx_g_full(experiments[i].M.data[experiments[i].M.selected_points])
#         experiments[i].dg_x_pca = np.asarray([np.matmul(projector, experiments[i].dg_x[j].transpose()) for j in range(nsel)])
#         experiments[i].W = ShapeSpace(experiments[i].positions, experiments[i].M.data)
#         experiments[i].dw = experiments[i].W.get_dw(cores, experiments[i].atoms3, experiments[i].natoms, experiments[i].M.selected_points)
#         experiments[i].dw_pca = np.asarray([np.matmul(projector, experiments[i].dw[j]) for j in range(nsel)])
#         experiments[i].dg_w = experiments[i].project(experiments[i].dw_pca, np.swapaxes(experiments[i].dg_x_pca,1,2))
#         tb_w_tangent_bases = experiments[i].project(experiments[i].dw_pca, np.swapaxes(subM.tb.tangent_bases, 1, 2))
#         experiments[i].dw_norm = experiments[i].normalize(experiments[i].dg_w)
#         experiments[i].dg_M = experiments[i].project(np.swapaxes(tb_w_tangent_bases, 1, 2), experiments[i].dw_norm)
#
#         #experiments[i].coeffs = experiments[i].get_betas_spam2(experiments[i].xtrain, experiments[i].ytrain, experiments[i].groups, lambdas,
#         #                                              nsel, experiments[i].q, itermax, tol)
#     return(experiments)

# def get_grads_reps_pca2(experiment, nreps, nsel,cores, projector):
#
#     experiments = {}
#     dim = experiment.dim
#     dimnoise = experiment.dimnoise
#     for i in range(nreps):
#         experiments[i] = copy.copy(experiment)
#         experiments[i].M.selected_points = np.random.choice(list(range(experiment.n)), nsel, replace=False)
#         tangent_bases = experiments[i].M.get_wlpca_tangent_sel(experiments[i].Mpca, experiments[i].M.selected_points,dimnoise)
#         subM = RiemannianManifold(experiments[i].Mpca.data[experiments[i].M.selected_points], dim)
#         subM.tb = TangentBundle(subM, tangent_bases)
#         #experiments[i].N.tangent_bundle = TangentBundle(experiments[i].N, np.swapaxes(experiments[i].N.geom.rmetric.Hvvexperiment.N.geom.rmetric.Hvv[0,:2,:],1,2))
#         experiments[i].N.tangent_bundle = TangentBundle(experiments[i].N, np.swapaxes(experiments[i].N.geom.rmetric.Hvv[:,:dim,:],1,2))
#         experiments[i].df_M = experiments[i].get_dF_js_idM(experiments[i].Mpca, experiments[i].N, subM.tb, experiments[i].N.tangent_bundle,
#                                                    experiments[i].M.selected_points,dimnoise)
#         #experiments[i].df_M = experiments[i].df_M / np.linalg.norm(experiments[i].df_M, axis=1).sum(axis=0)
#         experiments[i].df_M2 = experiments[i].df_M / np.linalg.norm(experiments[i].df_M) ** 2
#         experiments[i].dg_x = experiments[i].get_dx_g_full(experiments[i].M.data[experiments[i].M.selected_points])
#         experiments[i].W = ShapeSpace(experiments[i].positions, experiments[i].M.data)
#         experiments[i].dw = experiments[i].W.get_dw(cores, experiments[i].atoms3, experiments[i].natoms, experiments[i].M.selected_points)
#         experiments[i].dg_w = experiments[i].project(np.swapaxes(experiments[i].dw, 1, 2),
#                                              experiments[i].project(experiments[i].dw, experiments[i].dg_x))
#         experiments[i].dg_w_pca = np.asarray([np.matmul(projector, experiments[i].dg_w[j].transpose()).transpose() for j in range(nsel)])
#         experiments[i].dgw_norm = experiments[i].normalize(experiments[i].dg_w_pca)
#         # tb_w_tangent_bases = experiment.project(experiment.dw_pca, np.swapaxes(subM.tb.tangent_bases,1,2))
#         # experiment.dgw_norm = experiment.normalize(experiment.dg_w)
#         experiments[i].dg_M = experiments[i].project(subM.tb.tangent_bases, experiments[i].dgw_norm)
#
#         #experiments[i].dw_pca = np.asarray([np.matmul(projector, experiments[i].dw[j]) for j in range(nsel)])
#         #experiments[i].dg_w = experiments[i].project(experiments[i].dw_pca, np.swapaxes(experiments[i].dg_x_pca,1,2))
#         #tb_w_tangent_bases = experiments[i].project(experiments[i].dw_pca, np.swapaxes(subM.tb.tangent_bases, 1, 2))
#         #experiments[i].dw_norm = experiments[i].normalize(experiments[i].dg_w)
#         #experiments[i].dg_M = experiments[i].project(np.swapaxes(tb_w_tangent_bases, 1, 2), experiments[i].dw_norm)
#
#         #experiments[i].coeffs = experiments[i].get_betas_spam2(experiments[i].xtrain, experiments[i].ytrain, experiments[i].groups, lambdas,
#         #                                              nsel, experiments[i].q, itermax, tol)
#     return(experiments)

#
# def get_grads_reps_pca2_tangent(experiment, nreps, nsel,cores, projector):
#
#     experiments = {}
#     dim = experiment.dim
#     dimnoise = experiment.dimnoise
#     for i in range(nreps):
#         experiments[i] = copy.copy(experiment)
#         experiments[i].M.selected_points = np.random.choice(list(range(experiment.n)), nsel, replace=False)
#         tangent_bases = experiments[i].M.get_wlpca_tangent_sel(experiments[i].Mpca, experiments[i].M.selected_points,dimnoise)
#         subM = RiemannianManifold(experiments[i].Mpca.data[experiments[i].M.selected_points], dim)
#         subM.tb = TangentBundle(subM, tangent_bases)
#         experiments[i].N.tangent_bundle = TangentBundle(experiments[i].N, experiments[i].N.geom.rmetric.Hvv)
#         #experiments[i].df_M = experiments[i].get_dF_js_idM(experiments[i].Mpca, experiments[i].N, subM.tb, experiments[i].N.tangent_bundle,
#         #                                           experiments[i].M.selected_points,dimnoise)
#         #experiments[i].df_M = experiments[i].df_M / np.linalg.norm(experiments[i].df_M, axis=1).sum(axis=0)
#         #experiments[i].df_M2 = experiments[i].df_M / np.linalg.norm(experiments[i].df_M) ** 2
#         experiments[i].df_M = np.asarray([np.identity(dim) for i in range(nsel)])
#         experiments[i].dg_x = experiments[i].get_dx_g_full(experiments[i].M.data[experiments[i].M.selected_points])
#         experiments[i].W = ShapeSpace(experiments[i].positions, experiments[i].M.data)
#         experiments[i].dw = experiments[i].W.get_dw(cores, experiments[i].atoms3, experiments[i].natoms, experiments[i].M.selected_points)
#         experiments[i].dg_w = experiments[i].project(np.swapaxes(experiments[i].dw, 1, 2),
#                                              experiments[i].project(experiments[i].dw, experiments[i].dg_x))
#         experiments[i].dg_w_pca = np.asarray([np.matmul(projector, experiments[i].dg_w[j].transpose()).transpose() for j in range(nsel)])
#         experiments[i].dgw_norm = experiments[i].normalize(experiments[i].dg_w_pca)
#         # tb_w_tangent_bases = experiment.project(experiment.dw_pca, np.swapaxes(subM.tb.tangent_bases,1,2))
#         # experiment.dgw_norm = experiment.normalize(experiment.dg_w)
#         #experiment[i].dg_M = experiment.project(subM.tb.tangent_bases, experiment.dgw_norm)
#         experiments[i].dg_M = experiments[i].project(subM.tb.tangent_bases, experiments[i].dgw_norm)
#
#         #experiments[i].dw_pca = np.asarray([np.matmul(projector, experiments[i].dw[j]) for j in range(nsel)])
#         #experiments[i].dg_w = experiments[i].project(experiments[i].dw_pca, np.swapaxes(experiments[i].dg_x_pca,1,2))
#         #tb_w_tangent_bases = experiments[i].project(experiments[i].dw_pca, np.swapaxes(subM.tb.tangent_bases, 1, 2))
#         #experiments[i].dw_norm = experiments[i].normalize(experiments[i].dg_w)
#         #experiments[i].dg_M = experiments[i].project(np.swapaxes(tb_w_tangent_bases, 1, 2), experiments[i].dw_norm)
#
#         #experiments[i].coeffs = experiments[i].get_betas_spam2(experiments[i].xtrain, experiments[i].ytrain, experiments[i].groups, lambdas,
#         #                                              nsel, experiments[i].q, itermax, tol)
#     return(experiments)

#
# def get_grads_reps_noshape(experiment, nreps, nsel,cores):
#
#     experiments = {}
#     dim = experiment.dim
#
#     for i in range(nreps):
#         experiments[i] = copy.copy(experiment)
#         experiments[i].M.selected_points = np.random.choice(list(range(experiment.n)), nsel, replace=False)
#         experiments[i].selected_points = experiments[i].M.selected_points
#         tangent_bases = experiments[i].M.get_wlpca_tangent_sel(experiments[i].M, experiments[i].M.selected_points)
#         subM = RiemannianManifold(experiments[i].M.data[experiments[i].M.selected_points], dim)
#         subM.tb = TangentBundle(subM, tangent_bases)
#         experiments[i].N.tangent_bundle = TangentBundle(experiments[i].N, experiments[i].N.geom.rmetric.Hvv)
#         experiments[i].df_M = experiments[i].get_dF_js_idM(experiments[i].M, experiments[i].N, subM.tb, experiments[i].N.tangent_bundle,
#                                                    experiments[i].M.selected_points)
#         #experiments[i].df_M = experiments[i].df_M / np.linalg.norm(experiments[i].df_M, axis=1).sum(axis=0)
#         experiments[i].dg_x = experiments[i].get_dx_g_full(experiments[i].M.data[experiments[i].M.selected_points])
#         #experiments[i].W = ShapeSpace(experiments[i].positions, experiments[i].M.data)
#         #experiments[i].dw = experiments[i].W.get_dw(cores, experiments[i].atoms3, experiments[i].natoms, experiments[i].M.selected_points)
#         #experiments[i].dg_w = experiments[i].project(experiments[i].dw, experiments[i].dg_x)
#         #tb_w_tangent_bases = experiments[i].project(experiments[i].dw, np.swapaxes(subM.tb.tangent_bases, 1, 2))
#         experiments[i].dg_x_norm = experiments[i].normalize(experiments[i].dg_x)
#         experiments[i].dg_M = experiments[i].project(tangent_bases, experiments[i].dg_x_norm)
#
#         #experiments[i].coeffs = experiments[i].get_betas_spam2(experiments[i].xtrain, experiments[i].ytrain, experiments[i].groups, lambdas,
#         #                                              nsel, experiments[i].q, itermax, tol)
#     return(experiments)
#
# #
# def get_grads_reps_noshape_tangent(experiment, nreps, nsel, cores):
#
#     experiments = {}
#     dim = experiment.dim
#
#     for i in range(nreps):
#         experiments[i] = copy.copy(experiment)
#         experiments[i].M.selected_points = np.random.choice(list(range(experiment.n)), nsel, replace=False)
#         experiments[i].selected_points = experiments[i].M.selected_points
#         tangent_bases = experiments[i].M.get_wlpca_tangent_sel(experiments[i].M, experiments[i].M.selected_points)
#         subM = RiemannianManifold(experiments[i].M.data[experiments[i].M.selected_points], dim)
#         subM.tb = TangentBundle(subM, tangent_bases)
#         #experiments[i].N.tangent_bundle = TangentBundle(experiments[i].N, experiments[i].N.geom.rmetric.embedding_eigenvectors)
#         #experiments[i].df_M = experiments[i].get_dF_js_idM(experiments[i].M, experiments[i].N, subM.tb, experiments[i].N.tangent_bundle,
#         #                                           experiments[i].M.selected_points)
#         experiments[i].df_M = np.asarray([np.identity(dim) for i in range(nsel)])
#         #experiments[i].df_M = experiments[i].df_M / np.linalg.norm(experiments[i].df_M, axis=1).sum(axis=0)
#         experiments[i].dg_x = experiments[i].get_dx_g_full(experiments[i].M.data[experiments[i].M.selected_points])
#         #experiments[i].W = ShapeSpace(experiments[i].positions, experiments[i].M.data)
#         #experiments[i].dw = experiments[i].W.get_dw(cores, experiments[i].atoms3, experiments[i].natoms, experiments[i].M.selected_points)
#         #experiments[i].dg_w = experiments[i].project(experiments[i].dw, experiments[i].dg_x)
#         #tb_w_tangent_bases = experiments[i].project(experiments[i].dw, np.swapaxes(subM.tb.tangent_bases, 1, 2))
#         experiments[i].dg_x_norm = experiments[i].normalize(experiments[i].dg_x)
#         experiments[i].dg_M = experiments[i].project(tangent_bases, experiments[i].dg_x_norm)
#
#         #experiments[i].coeffs = experiments[i].get_betas_spam2(experiments[i].xtrain, experiments[i].ytrain, experiments[i].groups, lambdas,
#         #                                              nsel, experiments[i].q, itermax, tol)
#     return(experiments)

def get_coeffs_reps(experiments, nreps, lambdas, itermax,nsel,tol):

    for i in range(nreps):
        dimnoise = experiments[i].dimnoise
        experiments[i].xtrain, experiments[i].groups = experiments[i].construct_X_js(experiments[i].dg_M)
        experiments[i].ytrain = experiments[i].construct_Y_js(experiments[i].df_M,dimnoise)
        experiments[i].coeffs = experiments[i].get_betas_spam2(experiments[i].xtrain, experiments[i].ytrain, experiments[i].groups, lambdas,
                                                       nsel, experiments[i].q, itermax, tol)
    return(experiments)

def get_coeffs_reps_tangent(experiments, nreps, lambdas, itermax,nsel,tol):
    dim = experiments[0].dim
    for i in range(nreps):
        dimnoise = experiments[i].dimnoise
        experiments[i].xtrain, experiments[i].groups = experiments[i].construct_X(experiments[i].dg_M)
        experiments[i].ytrain = experiments[i].construct_Y(experiments[i].df_M, list(range(nsel)))
        experiments[i].coeffs = experiments[i].get_betas_spam2(experiments[i].xtrain, experiments[i].ytrain, experiments[i].groups, lambdas,
                                                       nsel, dim, itermax, tol)
    return(experiments)

def get_losses_reps(experiments, lambdas, nreps):

    losses = np.zeros((nreps, lambdas))
    for i in range(nreps):
        losses[i] = experiments[i].get_l2loss(experiments[i].coeffs, experiments[i].ytrain, experiments[i].xtrain)
    return(losses)

def get_penalty_reps(experiments, lambdas, nreps):

    penalties = np.zeros((nreps, lambdas))
    for i in range(nreps):
        penalties[i] = experiments[i].compute_penalty2(experiments[i].coeffs)
    return(penalties)

#
# def get_grads(experiment, Mpca, Mangles, N, selected_points):
#     dimnoise = experiment.dimnoise
#     dim = experiment.dim
#     cores = experiment.cores
#
#     tangent_bases = Mpca.get_wlpca_tangent_sel(Mpca, selected_points, dimnoise)
#     subM = RiemannianManifold(Mpca.data[selected_points], dim)
#     subM.tb = TangentBundle(subM, tangent_bases)
#     N.tangent_bundle = TangentBundle(N, np.swapaxes(N.geom.rmetric.Hvv[:,:dim,:],1,2))
#
#     df_M = experiment.get_dF_js_idM(Mpca, N, subM.tb, N.tangent_bundle, selected_points, dimnoise)
#     df_M2 = df_M / np.sum(np.linalg.norm(df_M, axis=1) ** 2, axis=0)
#     dg_x = experiment.get_dx_g_full(Mangles.data[selected_points])
#
#     W = ShapeSpace(experiment.positions, Mangles.data)
#     dw = W.get_dw(cores, experiment.atoms3, experiment.natoms, selected_points)
#     dg_w = experiment.project(np.swapaxes(dw, 1, 2),
#                               experiment.project(dw, dg_x))
#
#     dg_w_pca = np.asarray([np.matmul(experiment.projector, dg_w[j].transpose()).transpose() for j in range(len(selected_points))])
#     dgw_norm = experiment.normalize(dg_w_pca)
#     dg_M = experiment.project(subM.tb.tangent_bases, dgw_norm)
#     return (df_M2, dg_M, dg_w, dg_w_pca, dgw_norm)

def get_grads_noshape_swiss(experiment, M, N, selected_points):

    dim = experiment.dim
    dimnoise = experiment.dimnoise

    tangent_bases = M.get_wlpca_tangent_sel(M, selected_points, dimnoise)
    subM = RiemannianManifold(M.data[selected_points], dim)
    subM.tb = TangentBundle(subM, tangent_bases)
    N.tangent_bundle = TangentBundle(N, np.swapaxes(N.geom.rmetric.Hvv[:,:dim,:],1,2))
    df_M = experiment.get_dF_js_idM(M, N, subM.tb, N.tangent_bundle,
                                               selected_points)
    df_M2 = np.swapaxes(experiment.normalize(np.swapaxes(df_M,1,2)),1,2)
    dg_x = experiment.get_dx_g_full_2(selected_points)
    dg_x_norm = experiment.normalize(dg_x)
    dg_M = experiment.project(tangent_bases, dg_x_norm)

    return(df_M2, dg_M, dg_x, dg_x_norm)



def get_penalty(coeffs):
    pen = np.sum(np.linalg.norm(coeffs, axis=1))
    return (pen)


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
                coses[i, j, k] = cosine_similarity(dg[i, j, :], dg[i, k,
                                                                :])  # sklearn.metrics.pairwise.cosine_similarity(X = np.reshape(dg[:,i,:], (1,d*n)),Y = np.reshape(dg[:,j,:], (1,d*n)))[0][0]
    # cos_summary = np.abs(coses).sum(axis = 0) / n
    cos_summary = np.sum(coses ** 2, axis=0) / n
    return (cos_summary)


def get_dot_distribution(dg):
    n = dg.shape[0]
    p = dg.shape[1]
    d = dg.shape[2]

    coses = np.zeros((n, p, p))
    for i in range(n):
        for j in range(p):
            for k in range(p):
                coses[i, j, k] = np.dot(dg[i, j, :], dg[i, k,
                                                     :])  # sklearn.metrics.pairwise.cosine_similarity(X = np.reshape(dg[:,i,:], (1,d*n)),Y = np.reshape(dg[:,j,:], (1,d*n)))[0][0]
    return (coses)


def get_lower_interesting_lambda(experiment, replicate, lambda_max, max_search):
    p = experiment.p
    coeffs = replicate.coeff_dict
    combined_norms = replicate.combined_norms
    nsel = replicate.nsel
    tol = experiment.tol
    itermax = experiment.itermax

    low_probe = lambda_max

    for i in range(max_search):
        print(i, low_probe)
        if not np.isin(low_probe, list(combined_norms.keys())):
            coeffs[low_probe] = experiment.get_betas_spam2(replicate.xtrain, replicate.ytrain, replicate.groups,
                                                           np.asarray([low_probe]), nsel, experiment.m, itermax, tol)
            combined_norms[low_probe] = np.linalg.norm(np.linalg.norm(coeffs[low_probe][:, :, :, :], axis=2), axis=1)[0,
                                        :]
        low_enough = np.where((combined_norms[low_probe] > .85 * combined_norms[0]))[0]
        high_enough = np.where((combined_norms[low_probe] < .95 * combined_norms[0]))[0]
        if (any(np.intersect1d(high_enough, low_enough))) and (len(low_enough) == p):
            low_int = low_probe
            print('we did it')
            return (low_int, coeffs, combined_norms)
        else:
            if len(low_enough) < p:
                low_probe = low_probe / 2
            else:
                low_probe = low_probe * 1.5
    # return(lambda_max, coeffs, combined_norms)
    return (low_probe, coeffs, combined_norms)


def get_support_recovery_lambda(experiment, replicate, lambda_max, max_search, dim):
    coeffs = replicate.coeff_dict
    combined_norms = replicate.combined_norms
    nsel = replicate.nsel
    high_probe = lambda_max

    tol = experiment.tol
    itermax = experiment.itermax

    for i in range(max_search):
        print(i, high_probe)
        if not np.isin(high_probe, list(combined_norms.keys())):
            coeffs[high_probe] = experiment.get_betas_spam2(replicate.xtrain, replicate.ytrain, replicate.groups,
                                                            np.asarray([high_probe]), nsel, experiment.m, itermax, tol)
            combined_norms[high_probe] = np.linalg.norm(np.linalg.norm(coeffs[high_probe][:, :, :, :], axis=2), axis=1)[
                                         0, :]
        n_comp = len(np.where(combined_norms[high_probe] != 0)[0])
        if n_comp == dim:
            high_int = high_probe
            print('we did it')
            return (high_int, coeffs, combined_norms)
        else:
            if n_comp < dim:
                high_probe = high_probe / 2
            if n_comp > dim:
                high_probe = high_probe * 1.5
    # return(lambda_max, coeffs,combined_norms)
    return (high_probe, coeffs, combined_norms)




def get_support_recovery_lambda_warmstart(experiment, replicate, lambda_max, max_search, dim):
    coeffs = replicate.coeff_dict
    combined_norms = replicate.combined_norms
    nsel = replicate.nsel
    probe = 0.

    tol = experiment.tol
    itermax = experiment.itermax
    max_lam = lambda_max

    for i in range(max_search):
        print(i, probe)
        if not np.isin(probe, list(combined_norms.keys())):
            print('probe',probe)
            coeffs[probe] = experiment.get_betas_spam2(replicate.xtrain, replicate.ytrain, replicate.groups,
                                                            np.asarray([probe]), nsel, experiment.m, itermax, tol)

            combined_norms[probe] = np.linalg.norm(np.linalg.norm(coeffs[probe][:, :, :, :], axis=2), axis=1)[
                                         0, :]
        n_comp = len(np.where(combined_norms[probe] != 0)[0])
        if n_comp == dim:
            high_int = probe
            print('we did it')
            return (high_int, coeffs, combined_norms)
        else:
            if n_comp < dim:
                max_lam = np.min(max_lam, probe)
                #high_probe = high_probe / 2
            if n_comp > dim:
                min_lam = np.max(min_lam, probe)
                if len(max_lam) == 0:
                    probe = probe * 2
                else:
                    probe = (min_lam + max_lam)/2
                #high_probe = high_probe * 1.5
    # return(lambda_max, coeffs,combined_norms)
    return (probe, coeffs, combined_norms)
# def get_coeffs_and_lambdas(coeff_dict, lower_lambda, higher_lambda):
#     lambdas = np.asarray(list(coeff_dict.keys()))
#     lambdas.sort()
#     lambdas_relevant = np.asarray([x for x in lambdas if lower_lambda <= x <= higher_lambda])
#     print(lambdas_relevant.shape)
#     coeffs = np.zeros(np.hstack([len(lambdas_relevant), coeff_dict[0].shape[1:]]))
#     for i in range(len(lambdas_relevant)):
#         coeffs[i] = coeff_dict[lambdas_relevant[i]][0]
#     return (coeffs, lambdas_relevant)


def get_coeffs_and_lambdas(coeff_dict, lower_lambda, higher_lambda):
    lambdas = np.asarray(list(coeff_dict.keys()))
    lambdas.sort()
    lambdas_relevant = np.asarray([x for x in lambdas if lower_lambda <= x <= higher_lambda])
    print(lambdas_relevant.shape)
    coeffs = np.zeros(np.hstack([len(lambdas_relevant), coeff_dict[0].shape[1:]]))
    for i in range(len(lambdas_relevant)):
        coeffs[i] = coeff_dict[lambdas_relevant[i]][0]
    return (coeffs, lambdas_relevant)


# def get_support(coeffs, dim):
#     selected_functions = np.asarray(np.where(np.sum(np.sum(coeffs ** 2, axis=1), axis=1) > 0))

#     selection_lambda = np.min(np.where(np.asarray(list(Counter(selected_functions[0]).values())) == dim)[0])

#     selected_functions_at_selection_lambda = selected_functions[1][
#         np.where(selected_functions[0] == selection_lambda)[0]]

#     return (selected_functions_at_selection_lambda)


def get_support(coeffs, dim):
    selected_functions = np.asarray(np.where(np.sum(np.sum(coeffs ** 2, axis=1), axis=1) > 0))

    sls = np.where(np.asarray(list(Counter(selected_functions[0]).values())) == dim)[0]
    if len(sls) > 0:
        selection_lambda = np.min(sls)
        selected_functions_at_selection_lambda = selected_functions[1][
            np.where(selected_functions[0] == selection_lambda)[0]]

        return (selected_functions_at_selection_lambda)
    else:
        return(np.nan)


def get_olsnorm_and_supportsbrute(experiment, replicates):
    dim = experiment.dim
    dnoise = experiment.dnoise
    nreps = experiment.nreps
    nsel = experiment.nsel
    p = experiment.p

    parameterizations_possible = np.asarray(list(combinations(range(experiment.p), dnoise)))
    nparameterizations_possible = parameterizations_possible.shape[0]
    supports_brute = {}
    penalties = np.zeros((nreps, nparameterizations_possible))
    ols_norm = np.zeros((nreps, p, p))
    for r in range(nreps):
        brute_coeffs = np.zeros((nsel, nparameterizations_possible, experiment.dim, experiment.q))
        orthogonality = np.zeros((nsel, nparameterizations_possible))
        for i in range(nsel):
            for j in range(nparameterizations_possible):
                brute_coeffs[i, j] = np.linalg.lstsq(replicates[r].dg_M[i, parameterizations_possible[j], :], replicates[r].df_M[i])[0]
                # orthogonality[i,j] = get_penalty(brute_coeffs[i,j])
        for j in range(nparameterizations_possible):
            penalties[r, j] = experiment.compute_penalty2(np.expand_dims(brute_coeffs[:, j], 0))
        for j in range(nparameterizations_possible):
            ols_norm[r, parameterizations_possible[j][0], parameterizations_possible[j][1]] = penalties[r, j]
            ols_norm[r, parameterizations_possible[j][1], parameterizations_possible[j][0]] = penalties[r, j]
        supports_brute[r] = parameterizations_possible[penalties[r, :].argmin()]
    return (ols_norm, supports_brute)


def get_olsnorm_and_supportsbrute_1d(experiment, replicates):
    dim = experiment.dim
    dnoise = experiment.dnoise
    nreps = experiment.nreps
    nsel = experiment.nsel
    p = experiment.p
    parameterizations_possible = np.asarray(list(combinations(range(experiment.p), dnoise)))
    nparameterizations_possible = parameterizations_possible.shape[0]
    supports_brute = {}
    penalties = np.zeros((nreps, nparameterizations_possible))
    ols_norm = np.zeros((nreps, p))
    for r in range(nreps):
        brute_coeffs = np.zeros((nsel, nparameterizations_possible, experiment.dim, experiment.q))
        orthogonality = np.zeros((nsel, nparameterizations_possible))
        for i in range(nsel):
            for j in range(nparameterizations_possible):
                brute_coeffs[i, j] = np.linalg.lstsq(replicates[r].dg_M[i, parameterizations_possible[j], :], replicates[r].df_M[i])[0]
                # orthogonality[i,j] = get_penalty(brute_coeffs[i,j])
        for j in range(nparameterizations_possible):
            penalties[r, j] = experiment.compute_penalty2(np.expand_dims(brute_coeffs[:, j], 0))
        for j in range(nparameterizations_possible):
            ols_norm[r, parameterizations_possible[j][0]] = penalties[r, j]
        supports_brute[r] = parameterizations_possible[penalties[r, :].argmin()]
    return (ols_norm, supports_brute)


def get_grads_reps_pca2_tangent(experiment, nreps, nsel,cores, projector):

    experiments = {}
    dim = experiment.dim
    dimnoise = experiment.dimnoise
    for i in range(nreps):
        experiments[i] = copy.copy(experiment)
        experiments[i].M.selected_points = np.random.choice(list(range(experiment.n)), nsel, replace=False)
        tangent_bases = experiments[i].M.get_wlpca_tangent_sel(experiments[i].Mpca, experiments[i].M.selected_points,dimnoise)
        subM = RiemannianManifold(experiments[i].Mpca.data[experiments[i].M.selected_points], dim)
        subM.tb = TangentBundle(subM, tangent_bases)
        experiments[i].N.tangent_bundle = TangentBundle(experiments[i].N, experiments[i].N.geom.rmetric.embedding_eigenvectors)
        #experiments[i].df_M = experiments[i].get_dF_js_idM(experiments[i].Mpca, experiments[i].N, subM.tb, experiments[i].N.tangent_bundle,
        #                                           experiments[i].M.selected_points,dimnoise)
        #experiments[i].df_M = experiments[i].df_M / np.linalg.norm(experiments[i].df_M, axis=1).sum(axis=0)
        #experiments[i].df_M2 = experiments[i].df_M / np.linalg.norm(experiments[i].df_M) ** 2
        experiments[i].df_M = np.asarray([np.identity(dim) for i in range(nsel)])
        experiments[i].dg_x = experiments[i].get_dx_g_full(experiments[i].M.data[experiments[i].M.selected_points])
        experiments[i].W = ShapeSpace(experiments[i].positions, experiments[i].M.data)
        experiments[i].dw = experiments[i].W.get_dw(cores, experiments[i].atoms3, experiments[i].natoms, experiments[i].M.selected_points)
        experiments[i].dg_w = experiments[i].project(np.swapaxes(experiments[i].dw, 1, 2),
                                             experiments[i].project(experiments[i].dw, experiments[i].dg_x))
        experiments[i].dg_w_pca = np.asarray([np.matmul(projector, experiments[i].dg_w[j].transpose()).transpose() for j in range(nsel)])
        experiments[i].dgw_norm = experiments[i].normalize(experiments[i].dg_w_pca)
        # tb_w_tangent_bases = experiment.project(experiment.dw_pca, np.swapaxes(subM.tb.tangent_bases,1,2))
        # experiment.dgw_norm = experiment.normalize(experiment.dg_w)
        #experiment[i].dg_M = experiment.project(subM.tb.tangent_bases, experiment.dgw_norm)
        experiments[i].dg_M = experiments[i].project(subM.tb.tangent_bases, experiments[i].dgw_norm)

        #experiments[i].dw_pca = np.asarray([np.matmul(projector, experiments[i].dw[j]) for j in range(nsel)])
        #experiments[i].dg_w = experiments[i].project(experiments[i].dw_pca, np.swapaxes(experiments[i].dg_x_pca,1,2))
        #tb_w_tangent_bases = experiments[i].project(experiments[i].dw_pca, np.swapaxes(subM.tb.tangent_bases, 1, 2))
        #experiments[i].dw_norm = experiments[i].normalize(experiments[i].dg_w)
        #experiments[i].dg_M = experiments[i].project(np.swapaxes(tb_w_tangent_bases, 1, 2), experiments[i].dw_norm)

        #experiments[i].coeffs = experiments[i].get_betas_spam2(experiments[i].xtrain, experiments[i].ytrain, experiments[i].groups, lambdas,
        #                                              nsel, experiments[i].q, itermax, tol)
    return(experiments)
