
import numpy as np
import scipy.sparse as sps

def simple_assembly(mesh, Kloc):

    n_dofs = Kloc.shape[0]
    
    sparse_guess = 8*mesh.n_elems*n_dofs*2
    rows = np.zeros(sparse_guess, dtype=np.double)
    cols = np.zeros(sparse_guess, dtype=np.double)
    vals = np.zeros(sparse_guess, dtype=np.double)

    ind = 0
    for ielem in range(mesh.n_elems):
        Kelem = Kloc*mesh.jacb_inv_det[ielem]
        for i in range(n_dofs):
            for j in range(n_dofs):
                id1 = mesh.elem_to_dof[ielem, i]
                id2 = mesh.elem_to_dof[ielem, j]
                if not ((id1 in mesh.boundary_dofs) or \
                       (id2 in mesh.boundary_dofs)):
                    rows[ind] = id1
                    cols[ind] = id2
                    vals[ind] = Kelem[i,j]
                    ind += 1

    for idof in mesh.boundary_dofs:
        rows[ind] = idof
        cols[ind] = idof
        vals[ind] = 1.0
        ind += 1

    K = sps.coo_matrix((vals[:ind],(rows[:ind],cols[:ind])))
    
    return K.tocsr()

def simple_build_rhs(topo, basis, mesh, f):
    
    rhs = np.zeros(mesh.n_dofs, dtype=np.double)
    
    cub_points, cub_weights = topo.get_quadrature(basis.order+1)
    quad_points = topo.ref_to_phys(mesh.vertices[mesh.elem_to_vertex],
                                   cub_points)
    f_quad = f(quad_points)
    cub_vals = basis.eval_ref(np.eye(basis.n_dofs),
                              cub_points, d=0)
    a = f_quad.reshape((f_quad.shape[0],1,-1))*cub_vals
    a = a.dot(cub_weights)
    a = a*mesh.jacb_det.reshape((-1,1))

    for ielem in range(mesh.n_elems):
        rhs[mesh.elem_to_dof[ielem]] += a[ielem]
    rhs = -rhs
    rhs[mesh.boundary_dofs] = 0.0
    
    return rhs
