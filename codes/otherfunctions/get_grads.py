from codes.geometer.RiemannianManifold import RiemannianManifold
from codes.geometer.ShapeSpace import ShapeSpace
from codes.geometer.TangentBundle import TangentBundle
import numpy as np

# def get_grads(experiment, Mpca, Mangles, N, selected_points):
#     dimnoise = experiment.dimnoise
#     dim = experiment.dim
#     cores = experiment.cores

#     tangent_bases = Mpca.get_wlpca_tangent_sel(Mpca, selected_points, dimnoise)
#     subM = RiemannianManifold(Mpca.data[selected_points], dim)
#     subM.tb = TangentBundle(subM, tangent_bases)
#     N.tangent_bundle = TangentBundle(N, N.geom.rmetric.embedding_eigenvectors)

#     df_M = experiment.get_dF_js_idM(Mpca, N, subM.tb, N.tangent_bundle, selected_points, dimnoise)
#     df_M2 = df_M / np.sum(np.linalg.norm(df_M, axis=1) ** 2, axis=0)
#     dg_x = experiment.get_dx_g_full(Mangles.data[selected_points])

#     W = ShapeSpace(experiment.positions, Mangles.data)
#     dw = W.get_dw(cores, experiment.atoms3, experiment.natoms, selected_points)
#     dg_w = experiment.project(np.swapaxes(dw, 1, 2),
#                               experiment.project(dw, dg_x))

#     dg_w_pca = np.asarray([np.matmul(experiment.projector, dg_w[j].transpose()).transpose() for j in range(len(selected_points))])
#     dgw_norm = experiment.normalize(dg_w_pca)
#     dg_M = experiment.project(subM.tb.tangent_bases, dgw_norm)
#     return (df_M2, dg_M, dg_w, dg_w_pca, dgw_norm)


def get_grads(experiment, Mpca, Mangles, N, selected_points):
    dimnoise = experiment.dimnoise
    dim = experiment.dim
    cores = experiment.cores

    tangent_bases = Mpca.get_wlpca_tangent_sel(Mpca, selected_points, dimnoise)
    subM = RiemannianManifold(Mpca.data[selected_points], dim)
    subM.tb = TangentBundle(subM, tangent_bases)
    N.tangent_bundle = TangentBundle(N, np.swapaxes(N.geom.rmetric.Hvv[:,:dim,:],1,2))

    df_M = experiment.get_dF_js_idM(Mpca, N, subM.tb, N.tangent_bundle, selected_points, dimnoise)
    df_M2 = df_M / np.sum(np.linalg.norm(df_M, axis=1) ** 2, axis=0)
    dg_x = experiment.get_dx_g_full(Mangles.data[selected_points])

    W = ShapeSpace(experiment.positions, Mangles.data)
    dw = W.get_dw(cores, experiment.atoms3, experiment.natoms, selected_points)
    dg_w = experiment.project(np.swapaxes(dw, 1, 2),
                              experiment.project(dw, dg_x))

    dg_w_pca = np.asarray([np.matmul(experiment.projector, dg_w[j].transpose()).transpose() for j in range(len(selected_points))])
    dgw_norm = experiment.normalize(dg_w_pca)
    dg_M = experiment.project(subM.tb.tangent_bases, dgw_norm)
    return (df_M2, dg_M, dg_w, dg_w_pca, dgw_norm)
    
