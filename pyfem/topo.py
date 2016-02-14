
import numpy as np
from scipy.special.orthogonal import p_roots

class Interval(object):
    
    interval = (-1.0, 1.0)
    h        = 2.0
    
    def calc_jacb(self, nodes):
        do_ravel = nodes.ndim==1
        if do_ravel:
            nodes = nodes.reshape((1,-1))
        
        jacb = (nodes[:,1]-nodes[:,0])/self.h
        assert np.all(jacb!=0.0)
        
        if do_ravel: return jacb.ravel()
        return jacb
    
    def calc_jacb_det(self, jacb):
        return jacb
    
    def calc_jacb_inv(self, jacb):
        return 1.0/jacb
    
    def calc_jacb_inv_det(self, jacb):
        return 1.0/jacb
    
    def get_quadrature(self, n):
        return p_roots(n)
    
    def ref_to_phys(self, nodes, ref):
        do_ravel = nodes.ndim==1
        if do_ravel:
            nodes = nodes.reshape((1,-1))
        
        a = nodes[:,0].reshape((-1,1))
        b = (nodes[:,1]-nodes[:,0])
        b = b.reshape((-1,1))
        phys = a+b*(ref+1)/self.h
        
        if do_ravel: return phys.ravel()
        return phys
    
    def phys_to_ref(self, nodes, phys):
        pass


class SQuad(object):
    
    interval = (-1.0, 1.0)
    h        = 2.0
    
    vertices = np.array([[-1,-1],
                         [1,-1],
                         [1,1],
                         [-1,1]], dtype=np.int)
    edge_to_vertex = np.array([[0,1],
                               [1,2],
                               [3,2],
                               [0,3]], dtype=np.int)

    face_to_vertex = np.array([], dtype=np.int)
    face_to_edge = np.array([], dtype=np.int)
    
    n_vertices = 4
    n_edges    = 4
    n_vertex_per_edge = 2
    n_faces           = 0
    n_edge_per_face   = 0
    n_vertex_per_face = 0

    
    def calc_jacb(self, nodes):
        assert nodes.ndim==3
        
        dx = nodes[:,1,0]-nodes[:,0,0]
        dy = nodes[:,3,1]-nodes[:,0,1]
        
        jacb = np.zeros((nodes.shape[0], 2, 2))
        jacb[:,0,0] = dx/self.h
        jacb[:,1,1] = dy/self.h
        
        return jacb
        
    def calc_jacb_det(self, jacb):
        return np.linalg.det(jacb)
    
    def calc_jacb_inv(self, jacb):
        det = np.linalg.det(jacb)
        det = det.reshape((jacb.shape[0],1,1))
        inv = np.zeros_like(jacb)
        inv[:,0,0] =  jacb[:,1,1]
        inv[:,1,1] =  jacb[:,0,0]
        inv[:,0,1] = -jacb[:,0,1]
        inv[:,1,0] = -jacb[:,1,0]
        return inv/det
    
    def calc_jacb_inv_det(self, jacb_inv):
        return np.linalg.det(jacb_inv)
    
    def get_quadrature(self, n):
        xg, wg = p_roots(n)
        x = np.zeros((n**2, 2),
                     dtype=np.double)
        w = np.zeros(n**2,
                     dtype=np.double)
        
        for i in range(n):
            for j in range(n):
                p = i*n+j
                x[p, 0] = xg[j]
                x[p, 1] = xg[i]
                w[p]    = wg[i]*wg[j]
                
        return x, w
    
    def ref_to_phys(self, nodes, ref, jacb=None):

        if jacb is None:
            jacb = self.calc_jacb(nodes)

        b = nodes[:, 0, :]
        phys = np.zeros((nodes.shape[0],
                         ref.shape[0], 2),
                        dtype=np.double)
        for i in range(nodes.shape[0]):
            phys[i] = np.dot(jacb[i], ref.T+1).T+b[i]
        
        return phys
    
    def phys_to_ref(self, nodes, jacb_inv, phys):
        pass

class SHex(object):
    
    interval = (-1.0, 1.0)
    h        = 2.0
    
    vertices = np.array([[-1,-1,-1],
                         [1,-1,-1],
                         [1,1,-1],
                         [-1,1,-1],
                         [-1,-1,1],
                         [1,-1,1],
                         [1,1,1],
                         [-1,1,1]],
                        dtype=np.int)

    # Type up with 1 based index to make comparison with book easier,
    # then subtract 1 to get 0 based index
    edge_to_vertex = np.array([[1,2],
                               [2,3],
                               [4,3],
                               [1,4],
                               [1,5],
                               [2,6],
                               [3,7],
                               [4,8],
                               [5,6],
                               [6,7],
                               [8,7],
                               [5,8]],
                              dtype=np.int)-1

    face_to_vertex = np.array([[1,5,8,4],
                               [2,3,7,6],
                               [1,2,6,5],
                               [3,7,8,4],
                               [1,2,3,4],
                               [5,6,7,8]],
                              dtype=np.int)-1
                               
    face_to_edge = np.array([[5,12,8,4],
                             [2,7,10,6],
                             [1,6,9,5],
                             [7,11,8,3],
                             [1,2,3,4],
                             [9,10,11,12]],
                            dtype=np.int)-1

    n_vertices        = 8
    n_edges           = 12
    n_vertex_per_edge = 2
    n_faces           = 6
    n_edge_per_face   = 4
    n_vertex_per_face = 4
    
    def calc_jacb(self, nodes):
        assert nodes.ndim==3
        
        dx = nodes[:,1,0]-nodes[:,0,0]
        dy = nodes[:,3,1]-nodes[:,0,1]
        dz = nodes[:,4,2]-nodes[:,0,2]
        
        jacb = np.zeros((nodes.shape[0], 3, 3))
        jacb[:,0,0] = dx/self.h
        jacb[:,1,1] = dy/self.h
        jacb[:,2,2] = dz/self.h
        
        return jacb
        
    def calc_jacb_det(self, jacb):
        return np.linalg.det(jacb)
    
    def calc_jacb_inv(self, jacb):
        
        inv = np.zeros_like(jacb)
        inv[:,0,0] =  1.0/jacb[:,0,0]
        inv[:,1,1] =  1.0/jacb[:,1,1]
        inv[:,2,2] =  1.0/jacb[:,2,2]
        
        return inv
    
    def calc_jacb_inv_det(self, jacb_inv):
        return np.linalg.det(jacb_inv)
    
    def get_quadrature(self, n):
        xg, wg = p_roots(n)
        x = np.zeros((n**3, 3),
                     dtype=np.double)
        w = np.zeros(n**3,
                     dtype=np.double)
        
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    p = i*n*n+j*n+k
                    x[p, 0] = xg[k]
                    x[p, 1] = xg[j]
                    x[p, 2] = xg[i]
                    w[p]    = wg[i]*wg[j]*wg[k]

        return x, w
    
    def ref_to_phys(self, nodes, ref, jacb=None):

        if jacb is None:
            jacb = self.calc_jacb(nodes)

        b = nodes[:, 0, :]
        phys = np.zeros((nodes.shape[0],
                         ref.shape[0], 3),
                        dtype=np.double)
        for i in range(nodes.shape[0]):
            phys[i] = np.dot(jacb[i], ref.T+1).T+b[i]
        
        return phys
    
    def phys_to_ref(self, nodes, jacb_inv, phys):
        assert False
        
