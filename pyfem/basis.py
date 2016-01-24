
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


class LobattoBasisQuad(Basis2D):

    is_nodal = False

    def __init__(self, topo, order):
        self.topo  = topo
        self.order = order

        polys = lobatto_list(order)
        l0 = polys[0]
        l1 = polys[1]
        lp = polys[2:]

        dl0 = l0.deriv()
        dl1 = l1.deriv()
        dlp = [l.deriv() for l in lp]

        n = len(lp)
        n_dof_per_vertex = 1
        n_dof_per_edge   = n
        n_dof_per_bubble = n*n
        n_dofs = n_dof_per_vertex*topo.n_vertices+\
                 n_dof_per_edge*topo.n_edges+\
                 n_dof_per_bubble

        bp   = []
        bpd1 = []

        n_vertex_dofs = n_dof_per_vertex*topo.n_vertices
        vertex_to_dof = np.arange(n_vertex_dofs, dtype=np.int)
        vertex_to_dof = vertex_to_dof.reshape((topo.n_vertices,
                                               n_dof_per_vertex))

        bp.append(lambda x,y:l0(x)*l0(y))
        bp.append(lambda x,y:l1(x)*l0(y))
        bp.append(lambda x,y:l1(x)*l1(y))
        bp.append(lambda x,y:l0(x)*l1(y))

        bpd1.append([lambda x,y:dl0(x)*l0(y),
                     lambda x,y:l0(x)*dl0(y)])
        bpd1.append([lambda x,y:dl1(x)*l0(y),
                     lambda x,y:l1(x)*dl0(y)])
        bpd1.append([lambda x,y:dl1(x)*l1(y),
                     lambda x,y:l1(x)*dl1(y)])
        bpd1.append([lambda x,y:dl0(x)*l1(y),
                     lambda x,y:l0(x)*dl1(y)])

        n_edge_dofs = n_dof_per_edge*topo.n_edges
        edge_to_dof = np.arange(n_edge_dofs, dtype=np.int)
        edge_to_dof += n_vertex_dofs
        edge_to_dof = edge_to_dof.reshape((n_dof_per_edge,
                                           topo.n_edges)).T
        for i in range(n):
            lk  = lp[i]
            dlk = dlp[i]

            bp.append(lambda x,y,lk=lk:lk(x)*l0(y))
            bp.append(lambda x,y,lk=lk:l1(x)*lk(y))
            bp.append(lambda x,y,lk=lk:lk(x)*l1(y))
            bp.append(lambda x,y,lk=lk:l0(x)*lk(y))

            bpd1.append([lambda x,y,dlk=dlk:dlk(x)*l0(y),
                         lambda x,y,lk=lk:lk(x)*dl0(y)])
            bpd1.append([lambda x,y,lk=lk:dl1(x)*lk(y),
                         lambda x,y,dlk=dlk:l1(x)*dlk(y)])
            bpd1.append([lambda x,y,dlk=dlk:dlk(x)*l1(y),
                         lambda x,y,lk=lk:lk(x)*dl1(y)])
            bpd1.append([lambda x,y,lk=lk:dl0(x)*lk(y),
                         lambda x,y,dlk=dlk:l0(x)*dlk(y)])

        bubble_to_dof = np.arange(n_dof_per_bubble,
                                  dtype=np.int)
        bubble_to_dof += n_vertex_dofs+n_edge_dofs
        for i in range(n):
            for j in range(n):
                lx = lp[j]
                ly = lp[i]
                f = lambda x,y,lx=lx,ly=ly:lx(x)*ly(y)
                bp.append(f)

                dlx = dlp[j]
                dly = dlp[i]
                df1 = lambda x,y,dlx=dlx,ly=ly:dlx(x)*ly(y)
                df2 = lambda x,y,lx=lx,dly=dly:lx(x)*dly(y)
                bpd1.append([df1, df2])

        basis_polys = {}
        basis_polys[0] = bp
        basis_polys[1] = bpd1
        self.basis_polys = basis_polys

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
        
        x_ref = ref[:,0]
        y_ref = ref[:,1]
        z_ref = ref[:,2]
        polys = self.basis_polys[0]    
        for i in range(self.n_dofs):
            y = polys[i](x_ref, y_ref, z_ref)
            res += coeffs[:,i].reshape((-1,1))*y

        return res

    def _eval_ref_d1(self, coeffs, ref):
        
        res = np.zeros((coeffs.shape[0], 
                        3,
                        ref.shape[0]))
        
        x_ref = ref[:,0]
        y_ref = ref[:,1]
        z_ref = ref[:,2]
        polys = self.basis_polys[1]
        for i in range(self.n_dofs):
            dx = polys[i][0](x_ref, y_ref, z_ref)
            dy = polys[i][1](x_ref, y_ref, z_ref)
            dz = polys[i][2](x_ref, y_ref, z_ref)
            c = coeffs[:,i].reshape((-1,1))
            res[:,0,:] += c*dx
            res[:,1,:] += c*dy
            res[:,2,:] += c*dz

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
        bp   = []
        bpd1 = []

        n_dof_per_vertex = 1
        n_dof_per_edge   = order-1
        n_dof_per_face   = (order-1)**2
        n_dof_per_bubble = (order-1)**3

        assert n_dof_per_vertex*topo.n_vertices+\
               n_dof_per_edge*topo.n_edges+\
               n_dof_per_face*topo.n_faces+\
               n_dof_per_bubble==self.n_dofs

        dof_ref = []

        ind = 0
        for iz in range(order+1):
            for iy in range(order+1):
                for ix in range(order+1):
                    lx = la[ix]
                    ly = la[iy]
                    lz = la[iz]
                    dlx = dla[ix]
                    dly = dla[iy]
                    dlz = dla[iz]
                    f  = lambda x,y,z,lx=lx,ly=ly,lz=lz:lx(x)*ly(y)*lz(z)
                    dx = lambda x,y,z,dlx=dlx,ly=ly,lz=lz: dlx(x)*ly(y)*lz(z)
                    dy = lambda x,y,z,lx=lx,dly=dly,lz=lz: lx(x)*dly(y)*lz(z)
                    dz = lambda x,y,z,lx=lx,ly=ly,dlz=dlz: lx(x)*ly(y)*dlz(z)
                    bp.append(f)
                    bpd1.append([dx, dy, dz])
                    dof_ref.append((roots[ix], roots[iy], roots[iz]))

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
        basis_polys[0] = bp
        basis_polys[1] = bpd1
        self.basis_polys = basis_polys
        
        self.n_dof_per_vertex = n_dof_per_vertex
        self.n_dof_per_edge   = n_dof_per_edge
        self.n_dof_per_face   = n_dof_per_face
        self.n_dof_per_bubble = n_dof_per_bubble
        self.dof_ref = np.array(dof_ref, dtype=np.double)
