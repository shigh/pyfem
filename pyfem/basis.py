
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
    
    def __init__(self, topo, order):
        self.topo  = topo
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
        self.bubble_dofs = ids[1:-1]
        self.edge_dofs   = np.array([], dtype=np.int)
        self.face_dofs   = np.array([], dtype=np.int)
        self.n_dof_per_vertex  = 1
        self.n_dof_per_bubble  = len(self.bubble_dofs)
        self.n_vertex_per_elem = 2
        self.n_edge_per_elem   = 0
        self.n_face_per_elem   = 0

def legendre_list(order):
    
    k_max = order+1
    L0 = np.poly1d([1.0])
    L1 = np.poly1d([1.0, 0.0])
    x  = np.poly1d([1.0, 0.0])

    Lp = [L0, L1]
    for k in range(2, k_max+1):
        Lp.append((2.0*k-1.0)/k*x*Lp[k-1]-(k-1.0)/k*Lp[k-2])
    
    return Lp

def lobatto_list(order):
    
    k_max = order+1
    x  = np.poly1d([1.0, 0.0])
    l0 = (1.0-x)/2.0
    l1 = (x+1.0)/2.0

    lp = [l0, l1]
    Lp = legendre_list(k_max)
    for k in range(2, k_max+1):
        L2 = np.sqrt(2.0/(2.0*k-1.0))
        lp.append(Lp[k-1].integ()/L2)
        lp[-1] -= lp[-1](-1.0)
        
    return lp

class LobattoBasis1D(Basis1D):
    
    def __init__(self, topo, order):
        self.topo  = topo
        self.order = order
        self.q = order+1
        self.n_dofs = order+1

        basis_polys = {}            
        bp   = lobatto_list(order)
        bpd1 = [b.deriv() for b in bp]
        basis_polys[0] = bp
        basis_polys[1] = bpd1
        self.basis_polys = basis_polys
        
        ids = np.arange(order+1, dtype=np.int)
        self.vertex_dofs = np.array([ids[0],ids[1]],
                                    dtype=np.int)
        self.bubble_dofs = ids[2:]
        self.edge_dofs   = np.array([], dtype=np.int)
        self.face_dofs   = np.array([], dtype=np.int)
        self.n_dof_per_vertex  = 1
        self.n_dof_per_bubble  = len(self.bubble_dofs)
        self.n_vertex_per_elem = 2
        self.n_edge_per_elem   = 0
        self.n_face_per_elem   = 0
