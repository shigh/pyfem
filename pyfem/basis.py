
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

class LagrangeBasisInterval(Basis1D):
    
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
        self.dof_ref = roots

class LobattoBasisInterval(Basis1D):
    
    def __init__(self, topo, order):
        self.topo  = topo
        self.order = order
        self.q = order+1
        self.n_dofs = order+1

        basis_polys = {}            
        basis_polys[0] = lobatto_list(order)
        basis_polys[1] = lobatto_list_d1(order)
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

    dim = 2
    n_dof_per_face = 0

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

        poly = self.basis_polys[0]
        bp_inds = self.basis_poly_inds
        refT = ref.T
        pref = np.zeros((len(poly), 2, len(ref)))
        for i in range(len(poly)):
            pref[i,:,:] = poly[i](refT)

        for i in range(self.n_dofs):
            ix, iy = bp_inds[i]
            y = pref[ix,0,:]*pref[iy,1,:]
            res += coeffs[:,i].reshape((-1,1))*y

        return res

    def _eval_ref_d1(self, coeffs, ref):
        
        res = np.zeros((coeffs.shape[0], 
                        ref.shape[0], 2))

        poly  = self.basis_polys[0]
        dpoly = self.basis_polys[1]
        bp_inds = self.basis_poly_inds
        refT  = ref.T
        pref  = np.zeros((len(poly), 2, len(ref)))
        dpref = np.zeros((len(dpoly), 2, len(ref)))
        for i in range(len(poly)):
            pref[i,:,:]  = poly[i](refT)
            dpref[i,:,:] = dpoly[i](refT)
        
        for i in range(self.n_dofs):
            ix, iy = bp_inds[i]
            dx = dpref[ix,0,:]*pref[iy,1,:]
            dy = pref[ix,0,:]*dpref[iy,1,:]

            c = coeffs[:,i].reshape((-1,1))
            res[:,:,0] += c*dx
            res[:,:,1] += c*dy

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
        basis_poly_inds = []

        ind = 0
        for iy in range(order+1):
            for ix in range(order+1):
                dof_ref.append((roots[ix], roots[iy]))
                basis_poly_inds += [(ix, iy)]

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
        basis_polys[0] = la
        basis_polys[1] = dla
        self.basis_polys = basis_polys
        self.basis_poly_inds = basis_poly_inds
        
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


class LobattoBasisQuad(Basis2D):

    is_nodal = False

    def __init__(self, topo, order):
        self.topo  = topo
        self.order = order

        polys = lobatto_list(order)
        l0 = polys[0]
        l1 = polys[1]
        lp = polys[2:]

        n = len(lp)
        n_dof_per_vertex = 1
        n_dof_per_edge   = n
        n_dof_per_bubble = n*n
        n_dofs = n_dof_per_vertex*topo.n_vertices+\
                 n_dof_per_edge*topo.n_edges+\
                 n_dof_per_bubble

        basis_poly_inds = []

        n_vertex_dofs = n_dof_per_vertex*topo.n_vertices
        vertex_to_dof = np.arange(n_vertex_dofs, dtype=np.int)
        vertex_to_dof = vertex_to_dof.reshape((topo.n_vertices,
                                               n_dof_per_vertex))

        basis_poly_inds += [(0,0),
                            (1,0),
                            (1,1),
                            (0,1)]

        n_edge_dofs = n_dof_per_edge*topo.n_edges
        edge_to_dof = np.arange(n_edge_dofs, dtype=np.int)
        edge_to_dof += n_vertex_dofs
        edge_to_dof = edge_to_dof.reshape((n_dof_per_edge,
                                           topo.n_edges)).T
        for i in range(n):

            basis_poly_inds += [(i+2,0),
                                (1,i+2),
                                (i+2,1),
                                (0,i+2)]

        bubble_to_dof = np.arange(n_dof_per_bubble,
                                  dtype=np.int)
        bubble_to_dof += n_vertex_dofs+n_edge_dofs
        for i in range(n):
            for j in range(n):
                basis_poly_inds += [(j+2,i+2)]

        basis_polys = {}
        basis_polys[0] = lobatto_list(order)
        basis_polys[1] = lobatto_list_d1(order)
        self.basis_polys = basis_polys
        self.basis_poly_inds = basis_poly_inds

        if order>1:
            assert bubble_to_dof[-1]==n_dofs-1

        self.n_dof_per_vertex = n_dof_per_vertex
        self.n_dof_per_edge   = n_dof_per_edge
        self.n_dof_per_bubble = n_dof_per_bubble
        self.n_dofs           = n_dofs

        self.edge_to_dof   = edge_to_dof
        self.vertex_to_dof = vertex_to_dof
        self.bubble_to_dof = bubble_to_dof


class Basis3D(object):

    dim = 3

    def eval_ref(self, coeffs, ref, d=0):
        
        do_ravel = coeffs.ndim==1
        if do_ravel:
            coeffs = coeffs.reshape((1,-1))
        
        assert ref.ndim==2
        assert ref.shape[1]==3
        
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

        poly = self.basis_polys[0]
        bp_inds = self.basis_poly_inds
        refT = ref.T
        pref = np.zeros((len(poly), 3, len(ref)))
        for i in range(len(poly)):
            pref[i,:,:] = poly[i](refT)

        for i in range(self.n_dofs):
            ix, iy, iz = bp_inds[i]
            y = pref[ix,0,:]*pref[iy,1,:]*pref[iz,2,:]
            res += coeffs[:,i].reshape((-1,1))*y

        return res

    def _eval_ref_d1(self, coeffs, ref):
        
        res = np.zeros((coeffs.shape[0], 
                        ref.shape[0], 3))

        poly  = self.basis_polys[0]
        dpoly = self.basis_polys[1]
        bp_inds = self.basis_poly_inds
        refT  = ref.T
        pref  = np.zeros((len(poly), 3, len(ref)))
        dpref = np.zeros((len(dpoly), 3, len(ref)))
        for i in range(len(poly)):
            pref[i,:,:]  = poly[i](refT)
            dpref[i,:,:] = dpoly[i](refT)
        
        for i in range(self.n_dofs):
            ix, iy, iz = bp_inds[i]
            dx = dpref[ix,0,:]*pref[iy,1,:]*pref[iz,2,:]
            dy = pref[ix,0,:]*dpref[iy,1,:]*pref[iz,2,:]
            dz = pref[ix,0,:]*pref[iy,1,:]*dpref[iz,2,:]

            c = coeffs[:,i].reshape((-1,1))
            res[:,:,0] += c*dx
            res[:,:,1] += c*dy
            res[:,:,2] += c*dz

        return res

    def _eval_d1(self, ref):
        
        res = np.zeros((self.n_dofs,
                        ref.shape[0], 3))

        poly  = self.basis_polys[0]
        dpoly = self.basis_polys[1]
        bp_inds = self.basis_poly_inds
        refT  = ref.T
        pref  = np.zeros((len(poly), 3, len(ref)))
        dpref = np.zeros((len(dpoly), 3, len(ref)))
        for i in range(len(poly)):
            pref[i,:,:]  = poly[i](refT)
            dpref[i,:,:] = dpoly[i](refT)
        
        for i in range(self.n_dofs):
            ix, iy, iz = bp_inds[i]
            res[i,:,0] = dpref[ix,0,:]*pref[iy,1,:]*pref[iz,2,:]
            res[i,:,1] = pref[ix,0,:]*dpref[iy,1,:]*pref[iz,2,:]
            res[i,:,2] = pref[ix,0,:]*pref[iy,1,:]*dpref[iz,2,:]

        return res

    def _eval_d0(self, ref):
        
        res = np.zeros((self.n_dofs,
                        ref.shape[0]))

        poly  = self.basis_polys[0]
        bp_inds = self.basis_poly_inds
        refT  = ref.T
        pref  = np.zeros((len(poly), 3, len(ref)))
        for i in range(len(poly)):
            pref[i,:,:]  = poly[i](refT)
        
        for i in range(self.n_dofs):
            ix, iy, iz = bp_inds[i]
            res[i,:] = pref[ix,0,:]*pref[iy,1,:]*pref[iz,2,:]

        return res
        

class LagrangeBasisHex(Basis3D):

    is_nodal = True

    def __init__(self, topo, order):
        self.topo  = topo
        self.order = order
        self.q = q = order+1
        self.n_dofs = (order+1)**3

        roots = np.linspace(-1, 1, order+1)
        la   = lagrange_list(order)
        dla  = [l.deriv() for l in la]

        n_dof_per_vertex = 1
        n_dof_per_edge   = order-1
        n_dof_per_face   = (order-1)**2
        n_dof_per_bubble = (order-1)**3

        assert (n_dof_per_vertex*topo.n_vertices+\
                n_dof_per_edge*topo.n_edges+\
                n_dof_per_face*topo.n_faces+\
                n_dof_per_bubble)==self.n_dofs

        dof_ref = []
        basis_poly_inds = []
        ind = 0
        for iz in range(order+1):
            for iy in range(order+1):
                for ix in range(order+1):
                    dof_ref.append((roots[ix], roots[iy], roots[iz]))
                    basis_poly_inds.append((ix, iy, iz))

        vertex_to_dof = np.zeros((topo.n_vertices, n_dof_per_vertex),
                                 dtype=np.int)
        edge_to_dof   = np.zeros((topo.n_edges, n_dof_per_edge),
                                 dtype=np.int)
        face_to_dof   = np.zeros((topo.n_faces, n_dof_per_face),
                                 dtype=np.int)
        bubble_to_dof = np.zeros(n_dof_per_bubble,
                                 dtype=np.int)

        # Assign DOF mappings
        nd = order+1
        dofs = np.arange(nd**3, dtype=np.int).reshape((nd,nd,nd))

        vertex_to_dof[0,0] = dofs[0,0,0]
        vertex_to_dof[1,0] = dofs[0,0,-1]
        vertex_to_dof[2,0] = dofs[0,-1,-1]
        vertex_to_dof[3,0] = dofs[0,-1,0]
        vertex_to_dof[4,0] = dofs[-1,0,0]
        vertex_to_dof[5,0] = dofs[-1,0,-1]
        vertex_to_dof[6,0] = dofs[-1,-1,-1]
        vertex_to_dof[7,0] = dofs[-1,-1,0]

        if n_dof_per_edge>0:
            edge_to_dof[0,:]  = dofs[0,0,1:-1]
            edge_to_dof[1,:]  = dofs[0,1:-1,-1]
            edge_to_dof[2,:]  = dofs[0,-1,1:-1]
            edge_to_dof[3,:]  = dofs[0,1:-1,0]
            edge_to_dof[4,:]  = dofs[1:-1,0,0]
            edge_to_dof[5,:]  = dofs[1:-1,0,-1]
            edge_to_dof[6,:]  = dofs[1:-1,-1,-1]
            edge_to_dof[7,:]  = dofs[1:-1,-1,0]
            edge_to_dof[8,:]  = dofs[-1,0,1:-1]
            edge_to_dof[9,:]  = dofs[-1,1:-1,-1]
            edge_to_dof[10,:] = dofs[-1,-1,1:-1]
            edge_to_dof[11,:] = dofs[-1,1:-1,0]

        if n_dof_per_face>0:
            face_to_dof[0,:] = dofs[1:-1,1:-1,0].ravel()
            face_to_dof[1,:] = dofs[1:-1,1:-1,-1].ravel()
            face_to_dof[2,:] = dofs[1:-1,0,1:-1].ravel()
            face_to_dof[3,:] = dofs[1:-1,-1,1:-1].ravel()
            face_to_dof[4,:] = dofs[0,1:-1,1:-1].ravel()
            face_to_dof[5,:] = dofs[-1,1:-1,1:-1].ravel()

        if n_dof_per_bubble>0:
            bubble_to_dof[:] = dofs[1:-1,1:-1,1:-1].ravel()

        self.vertex_to_dof = vertex_to_dof
        self.edge_to_dof   = edge_to_dof
        self.face_to_dof   = face_to_dof
        self.bubble_to_dof = bubble_to_dof

        dof_check = np.hstack([vertex_to_dof.ravel(),
                               edge_to_dof.ravel(),
                               face_to_dof.ravel(),
                               bubble_to_dof.ravel()])
        dofs = dofs.ravel()
        assert len(dof_check)==len(dofs)
        assert np.all(np.sort(dof_check)==dofs)

        basis_polys = {}
        basis_polys[0] = la
        basis_polys[1] = dla
        self.basis_polys = basis_polys
        self.basis_poly_inds = basis_poly_inds

        self.n_dof_per_vertex = n_dof_per_vertex
        self.n_dof_per_edge   = n_dof_per_edge
        self.n_dof_per_face   = n_dof_per_face
        self.n_dof_per_bubble = n_dof_per_bubble
        self.dof_ref = np.array(dof_ref, dtype=np.double)


class LobattoBasisHex(Basis3D):

    is_nodal = False

    def __init__(self, topo, order):
        self.topo  = topo
        self.order = order

        polys = lobatto_list(order)
        l0 = polys[0]
        l1 = polys[1]
        lp = polys[2:]

        n = len(lp)
        n_dof_per_vertex = 1
        n_dof_per_edge   = n
        n_dof_per_face   = n*n
        n_dof_per_bubble = n*n*n
        n_dofs = n_dof_per_vertex*topo.n_vertices+\
                 n_dof_per_edge*topo.n_edges+\
                 n_dof_per_face*topo.n_faces+\
                 n_dof_per_bubble

        basis_poly_inds = []

        n_vertex_dofs = n_dof_per_vertex*topo.n_vertices
        vertex_to_dof = np.arange(n_vertex_dofs, dtype=np.int)
        vertex_to_dof = vertex_to_dof.reshape((topo.n_vertices,
                                               n_dof_per_vertex))

        basis_poly_inds += [(0,0,0),
                            (1,0,0),
                            (1,1,0),
                            (0,1,0),
                            (0,0,1),
                            (1,0,1),
                            (1,1,1),
                            (0,1,1)]

        n_edge_dofs = n_dof_per_edge*topo.n_edges
        edge_to_dof = np.arange(n_edge_dofs, dtype=np.int)
        edge_to_dof += n_vertex_dofs
        edge_to_dof = edge_to_dof.reshape((n_dof_per_edge,
                                           topo.n_edges)).T
        for i in range(n):
            
            basis_poly_inds += [(i+2,0,0),
                                (1,i+2,0),
                                (i+2,1,0),
                                (0,i+2,0),
                                (0,0,i+2),
                                (1,0,i+2),
                                (1,1,i+2),
                                (0,1,i+2),
                                (i+2,0,1),
                                (1,i+2,1),
                                (i+2,1,1),
                                (0,i+2,1)]

        n_face_dofs = n_dof_per_face*topo.n_faces
        face_to_dof = np.arange(n_face_dofs, dtype=np.int)
        face_to_dof += n_edge_dofs+n_vertex_dofs
        face_to_dof = face_to_dof.reshape((n_dof_per_face,
                                           topo.n_faces)).T
        for i in range(n):
            for j in range(n):

                basis_poly_inds += [(0,j+2,i+2),
                                    (1,j+2,i+2),
                                    (j+2,0,i+2),
                                    (j+2,1,i+2),
                                    (j+2,i+2,0),
                                    (j+2,i+2,1)]
                
        bubble_to_dof = np.arange(n_dof_per_bubble,
                                  dtype=np.int)
        bubble_to_dof += n_vertex_dofs+n_edge_dofs+n_face_dofs
        for iz in range(n):
            for iy in range(n):
                for ix in range(n):

                    basis_poly_inds += [(ix+2,iy+2,iz+2)]

        basis_polys = {}
        basis_polys[0] = lobatto_list(order)
        basis_polys[1] = lobatto_list_d1(order)
        self.basis_polys = basis_polys
        self.basis_poly_inds = basis_poly_inds

        if n_dof_per_bubble>0:
            assert bubble_to_dof[-1]==n_dofs-1
        else:
            assert vertex_to_dof[-1]==n_dofs-1

        self.n_dof_per_vertex = n_dof_per_vertex
        self.n_dof_per_edge   = n_dof_per_edge
        self.n_dof_per_face   = n_dof_per_face
        self.n_dof_per_bubble = n_dof_per_bubble
        self.n_dofs           = n_dofs

        self.edge_to_dof   = edge_to_dof
        self.face_to_dof   = face_to_dof
        self.vertex_to_dof = vertex_to_dof
        self.bubble_to_dof = bubble_to_dof
