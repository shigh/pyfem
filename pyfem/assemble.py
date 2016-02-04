
import numpy as np
import scipy.sparse as sps

def simple_assembly(mesh, Kloc):

    n_dofs = Kloc.shape[0]
    
    arr_size = 2**12
    rows = np.zeros(arr_size, dtype=np.double)
    cols = np.zeros(arr_size, dtype=np.double)
    vals = np.zeros(arr_size, dtype=np.double)
    rows_all = [rows]
    cols_all = [cols]
    vals_all = [vals]

    ind  = 0
    iloc = 0
    for ielem in range(mesh.n_elems):
        for i in range(n_dofs):
            for j in range(n_dofs):
                id1 = mesh.elem_to_dof[ielem, i]
                id2 = mesh.elem_to_dof[ielem, j]
                if not ((id1 in mesh.boundary_dofs) or \
                       (id2 in mesh.boundary_dofs)):
                    rows[iloc] = id1
                    cols[iloc] = id2
                    vals[iloc] = Kloc[i,j]
                    iloc += 1
                    ind  += 1

                    if iloc==arr_size:
                        rows = np.zeros(arr_size, dtype=np.double)
                        cols = np.zeros(arr_size, dtype=np.double)
                        vals = np.zeros(arr_size, dtype=np.double)
                        rows_all.append(rows)
                        cols_all.append(cols)
                        vals_all.append(vals)
                        iloc = 0

    for idof in mesh.boundary_dofs:
        rows[iloc] = idof
        cols[iloc] = idof
        vals[iloc] = 1.0
        iloc += 1
        ind  += 1

        if iloc==arr_size:
            rows = np.zeros(arr_size, dtype=np.double)
            cols = np.zeros(arr_size, dtype=np.double)
            vals = np.zeros(arr_size, dtype=np.double)
            rows_all.append(rows)
            cols_all.append(cols)
            vals_all.append(vals)
            iloc = 0

    rows = np.hstack(rows_all)
    cols = np.hstack(cols_all)
    vals = np.hstack(vals_all)
    K = sps.coo_matrix((vals[:ind],(rows[:ind],cols[:ind])))
    
    return K.tocsr()
    
def simple_build_rhs(topo, basis, mesh, f):
    
    rhs = np.zeros(mesh.n_dofs, dtype=np.double)
    
    cub_points, cub_weights = topo.get_quadrature(basis.order+1)
    etv = mesh.vertices[mesh.elem_to_vertex]
    quad_points = topo.ref_to_phys(etv, cub_points)
    f_quad = f(quad_points)
    cub_vals = basis.eval_ref(np.eye(basis.n_dofs),
                              cub_points, d=0)
    jacb     = topo.calc_jacb(etv)
    jacb_det = topo.calc_jacb_det(jacb)

    a = f_quad.reshape((f_quad.shape[0],1,-1))*cub_vals
    a = a.dot(cub_weights)
    a = a*jacb_det.reshape((-1,1))

    for ielem in range(mesh.n_elems):
        rhs[mesh.elem_to_dof[ielem]] += a[ielem]
    rhs[mesh.boundary_dofs] = 0.0
    
    return rhs
