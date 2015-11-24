
import numpy as np

class Mesh1D(object):
    
    def __init__(self, topo, basis):
        self.topo = topo
        self.basis = basis
        
    def build_mesh(self, nodes, elem_to_node, boundary_nodes=[]):
        self.nodes = nodes
        self.elem_to_node = elem_to_node
        
        self.n_elems  = elem_to_node.shape[0]
        self.n_nodes  = len(nodes)
        self.n_dofs   = (self.basis.n_dofs-1)*self.n_elems+1
        self.jacb     = self.topo.calc_jacb(nodes[elem_to_node])
        self.jacb_det = self.topo.calc_jacb_det(self.jacb)
        self.jacb_inv = self.topo.calc_jacb_inv(self.jacb)
        self.jacb_inv_det = self.topo.calc_jacb_inv_det(self.jacb)
        
        elem_to_dof = np.zeros((self.n_elems, self.basis.n_dofs),
                               dtype=np.int)
        
        basis = self.basis
        assert basis.n_dofs_per_node==1
        node_to_dof = np.arange(self.n_nodes)
        node_dofs = basis.node_dofs.ravel()
        elem_to_dof[:,node_dofs] = node_to_dof[elem_to_node]
        
        n_center_dofs = self.n_elems*basis.n_dofs_per_center
        center_dofs = np.arange(n_center_dofs)+len(node_to_dof)
        center_dofs = center_dofs.reshape((self.n_elems,
                                           basis.n_dofs_per_center))
        elem_to_dof[:,basis.center_dofs] = center_dofs
                                          
        self.boundary_dofs = node_to_dof[boundary_nodes]
        self.elem_to_dof = elem_to_dof
