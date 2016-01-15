
import numpy as np
from poly import *

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

class Basis2D(object):
    
    def eval_ref(self, coeffs, ref, d=0):
        
        do_ravel = coeffs.ndim==1
        if do_ravel:
            coeffs = coeffs.reshape((1,-1))
        
        assert ref.ndim==2
        assert ref.shape[1]==2
        
        if d==0:
            res = self._eval_ref_d0(coeffs, ref)
            if do_ravel:
                return res.ravel()
        elif d==1:
            res = self._eval_ref_d1(coeffs, ref)
            if do_ravel:
                return res.reshape((res.shape[1],
                                    res.shape[2]))
            
        if do_ravel: return res.ravel()
        return res

    def _eval_ref_d0(self, coeffs, ref):
        
        res = np.zeros((coeffs.shape[0], 
                        ref.shape[0]))
        
        x_ref = ref[:,0]
        y_ref = ref[:,1]
        polys = self.basis_polys[0]    
        for i in range(self.n_dofs):
            y = polys[i](x_ref, y_ref)
            res += coeffs[:,i].reshape((-1,1))*y

        return res

    def _eval_ref_d1(self, coeffs, ref):
        
        res = np.zeros((coeffs.shape[0], 
                        2,
                        ref.shape[0]))
        
        x_ref = ref[:,0]
        y_ref = ref[:,1]
        polys = self.basis_polys[1]
        for i in range(self.n_dofs):
            dx = polys[i][0](x_ref, y_ref)
            dy = polys[i][1](x_ref, y_ref)
            c = coeffs[:,i].reshape((-1,1))
            res[:,0,:] += c*dx
            res[:,1,:] += c*dy

        return res
        

class LagrangeBasisQuad(Basis2D):

    is_nodal = True

    def __init__(self, topo, order):
        self.topo  = topo
        self.order = order
        self.q = q = order+1
        self.n_dofs = (order+1)**2

        roots = np.linspace(-1, 1, order+1)
        la   = lagrange_list(order)
        dla  = [l.deriv() for l in la]
        bp   = []
        bpd1 = []
                
        vertex_to_dof = [[] for _ in 
                         range(topo.n_vertices)]
        edge_to_dof   = [[] for _ in 
                         range(topo.n_edges)]
        bubble_to_dof = []
        dof_ref = []

        ind = 0
        for iy in range(order+1):
            for ix in range(order+1):
                lx = la[ix]
                ly = la[iy]
                dlx = dla[ix]
                dly = dla[iy]
                f  = lambda x,y,lx=lx,ly=ly:   lx(x)*ly(y)
                dx = lambda x,y,dlx=dlx,ly=ly: dlx(x)*ly(y)
                dy = lambda x,y,lx=lx,dly=dly: lx(x)*dly(y)
                bp.append(f)
                bpd1.append([dx, dy])
                dof_ref.append((roots[ix], roots[iy]))

                if (iy==0):
                    if (ix==0):
                        vertex_to_dof[0].append(ind)
                    elif (ix==order):
                        vertex_to_dof[1].append(ind)
                    else:
                        edge_to_dof[0].append(ind)
                elif (iy==order):
                    if (ix==0):
                        vertex_to_dof[3].append(ind)
                    elif (ix==order):
                        vertex_to_dof[2].append(ind)
                    else:
                        edge_to_dof[2].append(ind)
                elif (ix==0):
                    edge_to_dof[3].append(ind)
                elif (ix==order):
                    edge_to_dof[1].append(ind)
                else:
                    bubble_to_dof.append(ind)
                    
                ind +=1
        
        basis_polys = {}
        basis_polys[0] = bp
        basis_polys[1] = bpd1
        self.basis_polys = basis_polys
        
        self.n_dof_per_vertex = 1
        self.n_dof_per_edge   = order-1
        self.n_dof_per_bubble = (order-1)**2
        
        for e in edge_to_dof:
            assert len(e)==self.n_dof_per_edge
        for e in vertex_to_dof:
            assert len(e)==self.n_dof_per_vertex
        assert len(bubble_to_dof)== self.n_dof_per_bubble
        assert ind==self.n_dofs
        
        self.edge_to_dof   = np.array(edge_to_dof, dtype=np.int)
        self.vertex_to_dof = np.array(vertex_to_dof, dtype=np.int)
        self.bubble_to_dof = np.array(bubble_to_dof, dtype=np.int)
        self.dof_ref       = np.array(dof_ref, dtype=np.double)
