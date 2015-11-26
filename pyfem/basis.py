
import numpy as np

class Basis1D(object):
    
    def eval_ref(self, coeffs, ref, d=0):
        
        do_ravel = coeffs.ndim==1
        if do_ravel:
            coeffs = coeffs.reshape((1,-1))
        
        res = np.zeros((coeffs.shape[0], 
                        len(ref)))
        
        polys = self.basis_polys[d]    
        for i in range(self.q):
            y = polys[i](ref)
            res += coeffs[:,i].reshape((-1,1))*y
            
        if do_ravel: return res.ravel()
        return res

class LagrangeBasis1D(Basis1D):
    
    def __init__(self, order):
        self.order = order
        self.q = order+1
        self.n_dofs = order+1

        roots = np.linspace(-1, 1, order+1)
        basis_polys = {}
        bp = []
        bpd1 = []
        flags = np.ones(order+1).astype(np.bool)
        for i in range(order+1):
            flags[:] = True
            flags[i] = False
            r = roots[flags]
            c = np.prod(roots[i]-r)
            p = np.poly1d(r, True)/c
            bp.append(p)
            bpd1.append(p.deriv())
            
        basis_polys[0] = bp
        basis_polys[1] = bpd1
        self.basis_polys = basis_polys
        
        ids = np.arange(order+1, dtype=np.int)
        self.vertex_dofs = np.array([ids[0],ids[-1]],
                                    dtype=np.int)
        self.center_dofs = ids[1:-1]
        self.edge_dofs   = np.array([], dtype=np.int)
        self.face_dofs   = np.array([], dtype=np.int)
        self.n_dof_per_vertex  = 1
        self.n_dof_per_center  = len(self.center_dofs)
        self.n_vertex_per_elem = 2
        self.n_edge_per_elem   = 0
        self.n_face_per_elem   = 0
