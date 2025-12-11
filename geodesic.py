from collections import defaultdict
import numpy as np
import trimesh

EDGE_TO_FACES = defaultdict(set)
VERTEX_TO_FACES = defaultdict(set)
FACE_TO_FACES = defaultdict(set)

def build_adjacency(mesh):
    global EDGE_TO_FACES, VERTEX_TO_FACES, FACE_TO_FACES

    # Initialize vertex/edge adjacency
    for f_idx, face in enumerate(mesh.faces):
        for loc_edge in range(3):
            v1 = face[loc_edge]
            v2 = face[(loc_edge + 1) % 3]
            edge = (v1, v2)
            EDGE_TO_FACES[tuple(sorted(edge))].add(f_idx)
            VERTEX_TO_FACES[v1].add(f_idx)
            VERTEX_TO_FACES[v2].add(f_idx)

    # Initialize face adjacency
    for edge in EDGE_TO_FACES.keys():
        if len(EDGE_TO_FACES[tuple(sorted(edge))]) == 2:
            f1, f2 = list(EDGE_TO_FACES[tuple(sorted(edge))])
            FACE_TO_FACES[f1].add(f2)
            FACE_TO_FACES[f2].add(f1)  
            
def b_to_w(mesh, tri_idx, b):
    # Convert barycentric coordinate to world coordinate
    ts =  mesh.vertices[mesh.faces[[tri_idx]]]
    bs = b[np.newaxis, :]
    w = trimesh.triangles.barycentric_to_points(ts, bs)[0]
    return w

def w_to_b(mesh, tri_idx, w):
    # Convert world coordiante to barycentric coordinate
    ts =  mesh.vertices[mesh.faces[[tri_idx]]]
    ws = w[np.newaxis, :]
    b = trimesh.triangles.points_to_barycentric(ts, ws)[0]
    return b

def basis_gradients(A, B, C):
    e1 = B - A
    e2 = C - A

    n = np.cross(e1, e2)
    norm2 = np.dot(n, n)

    grad_phi_A = np.cross(n, C - B) / norm2
    grad_phi_B = np.cross(n, A - C) / norm2
    grad_phi_C = np.cross(n, B - A) / norm2

    return grad_phi_A, grad_phi_B, grad_phi_C

def grad_T_w(mesh, distances, tri_idx, b):
    grad_vs = {}

    # For each vertex of the triangle compute direction
    # by gradients by taking average over all surounding
    # vertices
    for i, v in enumerate(mesh.faces[tri_idx]):
        grad_sum = np.zeros(3)
        for f in VERTEX_TO_FACES[v]:
            fvs_idx = mesh.faces[f]
            A = mesh.vertices[fvs_idx[0]]
            B = mesh.vertices[fvs_idx[1]]
            C = mesh.vertices[fvs_idx[2]]
            grad_phi_A, grad_phi_B, grad_phi_C = basis_gradients(A, B, C)
            fTs = [distances[v] for v in fvs_idx]
            grad_T_v = (fTs[0] * grad_phi_A +
                        fTs[1] * grad_phi_B +
                        fTs[2] * grad_phi_C)
            grad_sum += grad_T_v / len(VERTEX_TO_FACES[v])
        grad_vs[i] = grad_sum

    # Compute ∇T via barycentric interpolation
    grad_T = (b[0] * grad_vs[0] +
              b[1] * grad_vs[1] +
              b[2] * grad_vs[2])

    return grad_T

def project_to_plane(mesh, tri_idx, w):
    A, B, C = [mesh.vertices[v] for v in mesh.faces[tri_idx]]
    n = np.cross(B - A, C - A)
    n /= np.linalg.norm(n)
    w = w - np.dot(w - A, n) * n
    return w

def trace_geodesic(mesh, distances, source_idx, target_idx, h=1e-2):

    # Check if adjacency list has been built
    if len(FACE_TO_FACES.keys()) == 0:
        build_adjacency(mesh)

    # Initialize 
    f = list(VERTEX_TO_FACES[target_idx])[0]

    # We move slightly off the vertex to avoid hit degeneracies
    local_idx = np.where(mesh.faces[f] == target_idx)[0][0]
    b = np.zeros(3)
    b[local_idx] = 0.99
    b[(local_idx + 1) % 3] = 0.005
    b[(local_idx + 2) % 3] = 0.005
    
    p = b_to_w(mesh, f, b)
    
    path = [p.copy()]
    
    while True:
        # Compute gradient at p
        grad_T = grad_T_w(mesh, distances, f, b)

        # We are in triangle adjacent to source
        if source_idx in mesh.faces[f]:
            break

        norm = np.linalg.norm(grad_T)

        direction = -grad_T
        if norm > 0:
            direction /= norm
        move = h * direction

        # Euler step
        p_new = p + move
        b_new = w_to_b(mesh, f, p_new)

        # New point crosses an edge
        if any(b_new < 0):
            
            # Compute crossed edge
            v1_idx = (np.argmin(b_new) + 1)%3
            v2_idx = (v1_idx+1)%3
            v1 = mesh.faces[f][v1_idx]
            v2 = mesh.faces[f][v2_idx]
            edge = (v2, v1)

            # Get neighboring facet
            nbrs = list(EDGE_TO_FACES[tuple(sorted((v1, v2)))])
            if (len(nbrs) != 2):
                raise ValueError("Edge was found not shared")
            nbr = nbrs[0] if nbrs[0] != f else nbrs[1]

            # Clip to crossed edge
            cross_idx = np.argmin(b_new)
            t = b[cross_idx] / (b[cross_idx] - b_new[cross_idx])
            p_edge = p + t * (p_new - p)

            # Project back onto plane
            p_edge = project_to_plane(mesh, f, p_edge)

            # New path node            
            path.append(p_edge.copy())

            # Update 
            p = p_edge
            f = nbr
            b = w_to_b(mesh, f, p)

        # New point remains in facet
        else:
            
            # Project back onto plane
            p_new  = project_to_plane(mesh, f, p_new)

            # New path node
            path.append(p_new.copy())

            # Update
            p = p_new
            b = w_to_b(mesh, f, p)


    # Snap to source at the end 
    path.append(mesh.vertices[source_idx])

    # Convert to (N, 2, 3) shape necessary for viser
    path = np.stack([path[:-1], path[1:]], axis=1)
    
    return np.array(path)

def visualize_gradient(mesh, distances, samples = 5000, length=0.002):
    # Helper function which places gradient vectors
    # evenly spaced over the mesh
    
    # Check if adjacency list has been built
    if len(FACE_TO_FACES.keys()) == 0:
        build_adjacency(mesh)
        
    arrows_start = []
    arrows_end   = []

    # Sample surface of the mesh
    ps, fs = trimesh.sample.sample_surface_even(mesh, samples)
    
    for p, f in zip(ps, fs):
        # Get gradient at this point
        grad_w       = grad_T_w(mesh, distances, f, w_to_b(mesh, f, p))

        # Compute an arrow of length `length' in the direction of the gradient
        direction = -grad_w

        norm = np.linalg.norm(direction)
        if norm < 1e-8:
            continue
        direction /= norm

        end_point = p + length * direction
        arrows_start.append(p)
        arrows_end.append(end_point)

    arrows_start = np.array(arrows_start)
    arrows_end   = np.array(arrows_end)

    # Convert to (N, 2, 3) shape necessary for viser
    segments = np.stack([arrows_start, arrows_end], axis=1)
    
    return segments


def trace_contours(mesh, distances, num_levels=20, spline_points=100):
    # Compute the num_level level sets of T using splines for smooth interpolation
    
    V = mesh.vertices
    F = mesh.faces
    min_d, max_d = distances.min(), distances.max()
    delta = (max_d - min_d) / num_levels

    line_segments = []

    for f in F:
        verts = V[f]
        dists = distances[f]

        for level in np.arange(min_d, max_d, delta):
            edge_points = []
            for i, j in [(0,1),(1,2),(2,0)]:
                d1, d2 = dists[i], dists[j]
                if (d1 - level)*(d2 - level) < 0:
                    t = (level - d1) / (d2 - d1)
                    point = verts[i] + t*(verts[j] - verts[i])
                    edge_points.append(point)

            if len(edge_points) == 2:
                pts = np.array(edge_points)
                t = np.linspace(0, 1, spline_points)
                smooth_pts = (1-t)[:,None]*pts[0] + t[:,None]*pts[1]
                line_segments.append(smooth_pts)

    # Convert to (N, 2, 3) shape necessary for viser
    all_pairs = []
    for seg in line_segments:
        for j in range(len(seg) - 1):
            all_pairs.append([seg[j], seg[j + 1]])

    return np.array(all_pairs)
