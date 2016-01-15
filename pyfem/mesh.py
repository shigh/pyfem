
import numpy as np

class Mesh1D(object):
    
    def __init__(self, topo, basis):
        self.topo = topo
        self.basis = basis
        
    def build_mesh(self, vertices, elem_to_vertex,
                   boundary_vertices=[]):
        self.vertices = vertices
        self.elem_to_vertex = elem_to_vertex
        
        self.n_elems  = elem_to_vertex.shape[0]
        self.n_vertices = len(vertices)
        self.n_dofs   = (self.basis.n_dofs-1)*self.n_elems+1
        self.jacb     = self.topo.calc_jacb(vertices[elem_to_vertex])
        self.jacb_det = self.topo.calc_jacb_det(self.jacb)
        self.jacb_inv = self.topo.calc_jacb_inv(self.jacb)
        self.jacb_inv_det = self.topo.calc_jacb_inv_det(self.jacb)
        
        elem_to_dof = np.zeros((self.n_elems, self.basis.n_dofs),
                               dtype=np.int)
        
        basis = self.basis
        assert basis.n_dof_per_vertex==1
        vertex_to_dof = np.arange(self.n_vertices)
        vertex_dofs = basis.vertex_dofs.ravel()
        elem_to_dof[:,vertex_dofs] = vertex_to_dof[elem_to_vertex]
        
        n_bubble_dofs = self.n_elems*basis.n_dof_per_bubble
        bubble_dofs = np.arange(n_bubble_dofs)+len(vertex_to_dof)
        bubble_dofs = bubble_dofs.reshape((self.n_elems,
                                           basis.n_dof_per_bubble))
        elem_to_dof[:,basis.bubble_dofs] = bubble_dofs
                                          
        self.boundary_dofs = vertex_to_dof[boundary_vertices]
        self.elem_to_dof   = elem_to_dof


class Mesh2D(object):
    
    def __init__(self, topo, basis):
        self.topo = topo
        self.basis = basis
        
    def build_mesh(self, vertices, elem_to_vertex,
                   boundary_vertices=[]):
        self.vertices = vertices
        self.elem_to_vertex = elem_to_vertex
        self.boundary_vertices = boundary_vertices
        basis = self.basis
        topo  = self.topo
        
        # Build set of edges and edge maps
        elem_to_edge = np.zeros((len(elem_to_vertex),
                                topo.n_edges), dtype=np.int)
        edge_id = {}
        eid = 0
        for ielem in range(len(elem_to_vertex)):
            etv = elem_to_vertex[ielem]
            elem_edges = etv[topo.edge_to_vertex]
            for iedge in range(topo.n_edges):
                edge = elem_edges[iedge]
                edge.sort()
                t = tuple(edge)
                if not t in edge_id:
                    edge_id[t] = eid
                    eid += 1
                elem_to_edge[ielem, iedge] = edge_id[t]
        
        assert len(edge_id)==eid
        edge_to_vertex = np.zeros((len(edge_id), topo.n_vertex_per_edge),
                                  dtype=np.int)
        for k, v in edge_id.iteritems():
            edge_to_vertex[v, :] = k
        assert np.all(edge_to_vertex[:,0]<edge_to_vertex[:,1])
        self.elem_to_edge = elem_to_edge
        self.edge_to_vertex = edge_to_vertex
        self.edge_id = edge_id
        
        # Component and DOF counts
        self.n_elems    = len(elem_to_vertex)
        self.n_vertices = len(vertices)
        self.n_edges    = len(edge_id)
        
        self.n_dofs   = self.n_vertices*basis.n_dof_per_vertex+\
                        self.n_edges*basis.n_dof_per_edge+\
                        self.n_elems*basis.n_dof_per_bubble
        
        elem_to_dof = np.zeros((self.n_elems, basis.n_dofs),
                               dtype=np.int)
        
        # Vertex DOFs
        n_vertex_dofs = self.n_vertices*basis.n_dof_per_vertex
        vertex_to_dof = np.arange(n_vertex_dofs)
        bvtd = basis.vertex_to_dof.ravel()
        elem_to_dof[:,bvtd] = vertex_to_dof[elem_to_vertex]
        self.vertex_to_dof = vertex_to_dof.reshape((self.n_vertices,-1))
        
        # Edge DOFs
        n_edge_dofs = self.n_edges*basis.n_dof_per_edge
        edge_dofs   = np.arange(n_edge_dofs)+n_vertex_dofs
        edge_to_dof = edge_dofs.reshape((self.n_edges, -1))
        betd = basis.edge_to_dof.ravel()
        elem_to_dof[:,betd] = edge_to_dof[elem_to_edge].reshape((self.n_elems, -1))
        self.edge_to_dof = edge_to_dof
        
        # Bubble DOFs
        n_bubble_dofs = self.n_elems*basis.n_dof_per_bubble
        bubble_dofs = np.arange(n_bubble_dofs)+n_vertex_dofs\
                                              +n_edge_dofs
        bubble_to_dof = bubble_dofs.reshape((self.n_elems, -1))
        bbtd = basis.bubble_to_dof.ravel()
        elem_to_dof[:,bbtd] = bubble_to_dof
        self.bubble_to_dof = bubble_to_dof

        # Find edges on the boundary
        boundary_edges = []
        for iedge in range(self.n_edges):
            edge = self.edge_to_vertex[iedge]
            if (edge[0] in boundary_vertices) and\
               (edge[1] in boundary_vertices):
                boundary_edges.append(iedge)
        boundary_edges = np.array(boundary_edges, dtype=np.int)
        self.boundary_dofs = np.hstack([vertex_to_dof[boundary_vertices].ravel(),
                                        edge_to_dof[boundary_edges].ravel()])

        assert np.max(elem_to_dof)==(self.n_dofs-1)
        self.elem_to_dof   = elem_to_dof

    def get_dof_phys(self):

        topo, basis = self.topo, self.basis
        nodes = self.vertices[self.elem_to_vertex]
        phys  = topo.ref_to_phys(nodes, basis.dof_ref)

        dof_phys = np.zeros((self.n_dofs, 2))
        for ielem in range(self.n_elems):
            for idof in range(self.basis.n_dofs):
                dof = self.elem_to_dof[ielem, idof]
                dof_phys[dof] = phys[ielem, idof]

        return dof_phys

def uniform_nodes_2d(n_elems, x_max, y_max):
    
    x_vals = np.linspace(0, x_max, n_elems+1)
    y_vals = np.linspace(0, y_max, n_elems+1)

    vertices = np.zeros(((n_elems+1)**2, 2), dtype=np.double)
    elem_to_vertex = np.zeros((n_elems**2, 4), dtype=np.int)

    for i in range(n_elems):
        for j in range(n_elems):
            elem = i*n_elems+j
            elem_to_vertex[elem,0] = i*(n_elems+1)+j
            elem_to_vertex[elem,1] = i*(n_elems+1)+j+1
            elem_to_vertex[elem,2] = (i+1)*(n_elems+1)+j+1
            elem_to_vertex[elem,3] = (i+1)*(n_elems+1)+j

    boundary_vertices = []
    for i in range(n_elems+1):
        for j in range(n_elems+1):
            v = i*(n_elems+1)+j
            vertices[v,0] = x_vals[j]
            vertices[v,1] = y_vals[i]
            if (i==0) or (j==0) or\
               (i==n_elems) or (j==n_elems):
                boundary_vertices.append(v)
                
    return (vertices, elem_to_vertex, boundary_vertices)
        
