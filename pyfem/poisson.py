
import numpy as np

def poisson_Kloc(basis, jacb_det, jacb_inv):

    topo  = basis.topo
    order = basis.order
    cub_points, cub_weights = topo.get_quadrature(order+1)
    Kloc = np.zeros((basis.n_dofs, basis.n_dofs),
                    dtype=np.double)
    cub_vals = basis.eval_ref(np.eye(basis.n_dofs),
                              cub_points, d=1)

    for i in range(basis.n_dofs):
        for j in range(basis.n_dofs):
            d1 = jacb_inv.T.dot(cub_vals[i].T)
            d2 = jacb_inv.T.dot(cub_vals[j].T)
            p = d1*d2
            Kloc[i,j] = np.sum(p, axis=0).dot(cub_weights)
    Kloc = jacb_det*Kloc

    return Kloc

def poisson_Mloc(basis, jacb_det):

    topo  = basis.topo
    order = basis.order
    cub_points, cub_weights = topo.get_quadrature(order+1)
    Mloc = np.zeros((basis.n_dofs, basis.n_dofs),
                    dtype=np.double)
    cub_vals = basis.eval_ref(np.eye(basis.n_dofs),
                              cub_points, d=0)

    for i in range(basis.n_dofs):
        for j in range(basis.n_dofs):
            d1 = cub_vals[i].T
            d2 = cub_vals[j].T
            p = d1*d2
            Mloc[i,j] = np.sum(p*cub_weights)
    Mloc = jacb_det*Mloc

    return Mloc
