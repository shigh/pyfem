import numpy as np
cimport numpy as np
from numpy import pi
from scipy.sparse import coo_matrix
import scipy.sparse as sps

ctypedef np.float64_t FLOAT

def simple_assembly(mesh, FLOAT[:,:] Kloc):

    cdef int n_dofs = Kloc.shape[0]
    
    cdef int arr_size = 2**12
    cdef np.ndarray[dtype=FLOAT,ndim=1] rows = np.zeros(arr_size,
                                                        dtype=np.double)
    cdef np.ndarray[dtype=FLOAT,ndim=1] cols = np.zeros(arr_size,
                                                        dtype=np.double)
    cdef np.ndarray[dtype=FLOAT,ndim=1] vals = np.zeros(arr_size,
                                                        dtype=np.double)
    rows_all = []
    cols_all = []
    vals_all = []

    cdef long[:,:] elem_to_dof   = mesh.elem_to_dof
    cdef long[:]   boundary_dofs = mesh.boundary_dofs
    
    s = set(mesh.boundary_dofs)
    on_bndy = lambda i:i in s
    
    cdef int n_elems = mesh.n_elems
    cdef int ind  = 0
    cdef int iloc = 0
    cdef int ielem, i, j, id1, id2
    for ielem in range(n_elems):
        for i in range(n_dofs):
            for j in range(n_dofs):
                id1 = elem_to_dof[ielem, i]
                id2 = elem_to_dof[ielem, j]
                if not (on_bndy(id1) or on_bndy(id2)):
                    
                    rows[iloc] = id1
                    cols[iloc] = id2
                    vals[iloc] = Kloc[i,j]
                    iloc += 1
                    ind  += 1

                    if iloc==arr_size:
                        rows_all.append(rows.copy())
                        cols_all.append(cols.copy())
                        vals_all.append(vals.copy())
                        iloc = 0

    cdef int idof
    for idof in boundary_dofs:
        rows[iloc] = idof
        cols[iloc] = idof
        vals[iloc] = 1.0
        iloc += 1
        ind  += 1

        if iloc==arr_size:
            rows_all.append(rows.copy())
            cols_all.append(cols.copy())
            vals_all.append(vals.copy())
            iloc = 0

    rows_all.append(rows.copy())
    cols_all.append(cols.copy())
    vals_all.append(vals.copy())
            
    rows = np.hstack(rows_all)
    cols = np.hstack(cols_all)
    vals = np.hstack(vals_all)
    K = sps.coo_matrix((vals[:ind],(rows[:ind],cols[:ind])))
    
    return K.tocsr()
    
