
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


class Mesh(object):
    
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

        has_faces = topo.n_faces>0

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

        # Build set of faces and face maps
        elem_to_face = np.zeros((len(elem_to_vertex),
                                 topo.n_faces), dtype=np.int)
        face_id = {}
        fid = 0
        for ielem in range(len(elem_to_vertex)):
            etv = elem_to_vertex[ielem]
            elem_faces = etv[topo.face_to_vertex]
            for iface in range(topo.n_faces):
                face = elem_faces[iface]
                face.sort()
                t = tuple(face)
                if not t in face_id:
                    face_id[t] = fid
                    fid += 1
                elem_to_face[ielem, iface] = face_id[t]
        
        assert len(face_id)==fid
        face_to_vertex = np.zeros((len(face_id), topo.n_vertex_per_face),
                                  dtype=np.int)
        for k, v in face_id.iteritems():
            face_to_vertex[v, :] = k
        if has_faces:
            assert np.all(face_to_vertex[:,0]<face_to_vertex[:,1])
            assert np.all(face_to_vertex[:,1]<face_to_vertex[:,2])
            assert np.all(face_to_vertex[:,2]<face_to_vertex[:,3])
        self.elem_to_face = elem_to_face
        self.face_to_vertex = face_to_vertex
        self.face_id = face_id
        
        # Component and DOF counts
        self.n_elems    = len(elem_to_vertex)
        self.n_vertices = len(vertices)
        self.n_edges    = len(edge_id)
        self.n_faces    = len(face_id)
        
        self.n_dofs   = self.n_vertices*basis.n_dof_per_vertex+\
                        self.n_edges*basis.n_dof_per_edge+\
                        self.n_faces*basis.n_dof_per_face+\
                        self.n_elems*basis.n_dof_per_bubble
        
        elem_to_dof = np.zeros((self.n_elems, basis.n_dofs),
                               dtype=np.int)-1
        
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

        # Face DOFs
        n_face_dofs = self.n_faces*basis.n_dof_per_face
        if has_faces:
            face_dofs   = np.arange(n_face_dofs)+n_vertex_dofs+n_edge_dofs
            face_to_dof = face_dofs.reshape((self.n_faces, -1))
            bftd = basis.face_to_dof.ravel()
            elem_to_dof[:,bftd] = face_to_dof[elem_to_face].reshape((self.n_elems, -1))
            self.face_to_dof = face_to_dof
        else:
            self.face_to_dof = np.array([], np.int)
        
        # Bubble DOFs
        n_bubble_dofs = self.n_elems*basis.n_dof_per_bubble
        bubble_dofs = np.arange(n_bubble_dofs)+n_vertex_dofs\
                      +n_edge_dofs+n_face_dofs
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

        # Find faces on the boundary
        boundary_faces = []
        for iface in range(self.n_faces):
            face = self.face_to_vertex[iface]
            if (face[0] in boundary_vertices) and\
               (face[1] in boundary_vertices) and\
               (face[2] in boundary_vertices) and\
               (face[3] in boundary_vertices):
                boundary_faces.append(iface)
        boundary_faces = np.array(boundary_faces, dtype=np.int)

        bdofs = [vertex_to_dof[boundary_vertices].ravel(),
                 edge_to_dof[boundary_edges].ravel()]
        if has_faces:
            bdofs.append(face_to_dof[boundary_faces].ravel())
        self.boundary_dofs = np.hstack(bdofs)

        assert np.all(elem_to_dof>=0)
        assert np.max(elem_to_dof)==(self.n_dofs-1)
        self.elem_to_dof = elem_to_dof

    def get_dof_phys(self):

        topo, basis = self.topo, self.basis
        nodes = self.vertices[self.elem_to_vertex]
        phys  = topo.ref_to_phys(nodes, basis.dof_ref)

        dof_phys = np.zeros((self.n_dofs, basis.dim))
        for ielem in range(self.n_elems):
            for idof in range(self.basis.n_dofs):
                dof = self.elem_to_dof[ielem, idof]
                dof_phys[dof] = phys[ielem, idof]

        return dof_phys

    def eval_ref(self, b, ref, d=0):

        assert (b.ndim==1) and (len(b)==self.n_dofs)

        coeffs = b[self.elem_to_dof]
        r = self.basis.eval_ref(coeffs, ref, d=d)

        return r

    def eval_elem_ref(self, b, elem, ref):

        assert (b.ndim==1) and (len(b)==self.n_dofs)
        assert len(elem)==len(ref)
        assert (ref.ndim==2) and (ref.shape[1]==self.basis.dim)

        res = np.zeros_like(elem, dtype=np.double)
        for e in np.unique(elem):
            coeffs = b[self.elem_to_dof[e]]
            is_elem = elem==e
            eref = ref[is_elem]
            r = self.basis.eval_ref(coeffs, eref)
            res[is_elem] = r

        return res


def uniform_nodes_2d(n_elems, x_max, y_max,
                     get_elem_ref=False,
                     periodic=False):
    
    x_vals = np.linspace(0, x_max, n_elems+1)
    y_vals = np.linspace(0, y_max, n_elems+1)

    vertex_map = {}
    edge_map   = {}

    vertices = np.zeros(((n_elems+1)**2, 2), dtype=np.double)
    elem_to_vertex = np.zeros((n_elems**2, 4), dtype=np.int)

    for i in range(n_elems):
        for j in range(n_elems):
            elem = i*n_elems+j
            elem_to_vertex[elem,0] = i*(n_elems+1)+j
            elem_to_vertex[elem,1] = i*(n_elems+1)+j+1
            elem_to_vertex[elem,2] = (i+1)*(n_elems+1)+j+1
            elem_to_vertex[elem,3] = (i+1)*(n_elems+1)+j

    if periodic:
        i = n_elems-1
        for j in range(n_elems+1):
            v  = (i+1)*(n_elems+1)+j
            mv = j
            vertex_map[v] = mv

        j = n_elems-1
        for i in range(n_elems+1):
            v =  i*(n_elems+1)+j+1
            mv = i*(n_elems+1)
            if mv in vertex_map:
                vertex_map[v] = vertex_map[mv]
            else:
                vertex_map[v] = mv

    boundary_vertices = []
    for i in range(n_elems+1):
        for j in range(n_elems+1):
            v = i*(n_elems+1)+j
            vertices[v,0] = x_vals[j]
            vertices[v,1] = y_vals[i]
            if (i==0) or (j==0) or\
               (i==n_elems) or (j==n_elems):
                boundary_vertices.append(v)

    # Return element and ref location in that element for a given
    # set of points
    def _get_elem_ref(phys):

        Lx, Ly = x_max, y_max
        nelx, nely = n_elems, n_elems
        hx, hy = [a[1]-a[0] for a in
                  (x_vals, y_vals)]

        x = np.array(phys[:,0], ndmin=1)
        y = np.array(phys[:,1], ndmin=1)
        assert np.all(x>=0) and np.all(x<=Lx)
        assert np.all(y>=0) and np.all(y<=Ly)

        yelem = np.floor(y/hy).astype(np.int)
        xelem = np.floor(x/hx).astype(np.int)
        xelem[xelem==nelx] = nelx-1
        yelem[yelem==nely] = nely-1
        elem = yelem*nelx+xelem

        assert np.max(xelem)<nelx
        assert np.max(yelem)<nely

        xref = (x-xelem*hx)*2.0/hx-1.0
        yref = (y-yelem*hy)*2.0/hy-1.0

        ref = np.zeros_like(phys)
        ref[:,0] = xref
        ref[:,1] = yref
        return (elem, ref)

    ret = [vertices, elem_to_vertex,
           np.array(boundary_vertices, dtype=np.int)]
    if get_elem_ref:
        ret.append(_get_elem_ref)
    if periodic:
        ret.append(vertex_map)

    return ret

def uniform_nodes_3d(n_elems, x_max, y_max, z_max, get_elem_ref=False):
    
    x_vals = np.linspace(0, x_max, n_elems+1)
    y_vals = np.linspace(0, y_max, n_elems+1)
    z_vals = np.linspace(0, z_max, n_elems+1)

    vertices = np.zeros(((n_elems+1)**3, 3), dtype=np.double)
    elem_to_vertex = np.zeros((n_elems**3, 8), dtype=np.int)

    nv = n_elems+1
    for iz in range(n_elems):
        for iy in range(n_elems):
            for ix in range(n_elems):
                elem = (iz*n_elems+iy)*n_elems+ix
                c = iz*nv*nv
                elem_to_vertex[elem,0] = c+iy*nv+ix
                elem_to_vertex[elem,1] = c+iy*nv+ix+1
                elem_to_vertex[elem,2] = c+(iy+1)*nv+ix+1
                elem_to_vertex[elem,3] = c+(iy+1)*nv+ix
                c = (iz+1)*nv*nv
                elem_to_vertex[elem,4] = c+iy*nv+ix
                elem_to_vertex[elem,5] = c+iy*nv+ix+1
                elem_to_vertex[elem,6] = c+(iy+1)*nv+ix+1
                elem_to_vertex[elem,7] = c+(iy+1)*nv+ix

    boundary_vertices = []
    for iz in range(nv):
        for iy in range(nv):
            for ix in range(nv):
                v = (iz*nv+iy)*nv+ix
                vertices[v,0] = x_vals[ix]
                vertices[v,1] = y_vals[iy]
                vertices[v,2] = z_vals[iz]
                if (ix==0) or (iy==0) or (iz==0) or\
                   (ix==n_elems) or (iy==n_elems) or (iz==n_elems):
                    boundary_vertices.append(v)

        # Return element and ref location in that element for a given
        # set of points
        def _get_elem_ref(phys):

            Lx, Ly, Lz = x_max, y_max, z_max
            nelx, nely, nelz = n_elems, n_elems, n_elems
            hx, hy, hz = [a[1]-a[0] for a in
                          (x_vals, y_vals, z_vals)]

            x = np.array(phys[:,0], ndmin=1)
            y = np.array(phys[:,1], ndmin=1)
            z = np.array(phys[:,2], ndmin=1)
            assert np.all(x>=0) and np.all(x<=Lx)
            assert np.all(y>=0) and np.all(y<=Ly)
            assert np.all(z>=0) and np.all(z<=Lz)

            zelem = np.floor(z/hz).astype(np.int)
            yelem = np.floor(y/hy).astype(np.int)
            xelem = np.floor(x/hx).astype(np.int)
            xelem[xelem==nelx] = nelx-1
            yelem[yelem==nely] = nely-1
            zelem[zelem==nelz] = nelz-1
            elem = zelem*nely*nelx+yelem*nelx+xelem

            assert np.max(xelem)<nelx
            assert np.max(yelem)<nely
            assert np.max(zelem)<nelz

            xref = (x-xelem*hx)*2.0/hx-1.0
            yref = (y-yelem*hy)*2.0/hy-1.0
            zref = (z-zelem*hz)*2.0/hz-1.0

            ref = np.zeros_like(phys)
            ref[:,0] = xref
            ref[:,1] = yref
            ref[:,2] = zref
            return (elem, ref)
                    

    if get_elem_ref:
            return (vertices, elem_to_vertex,
                    np.array(boundary_vertices, dtype=np.int),
                    _get_elem_ref)
    else:
        return (vertices, elem_to_vertex,
                np.array(boundary_vertices, dtype=np.int))
