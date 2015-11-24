
import numpy as np
from scipy.special.orthogonal import p_roots

class IntervalTopo1D(object):
    
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
