from collections import defaultdict
import numpy as np
import trimesh

EDGE_TO_FACES = defaultdict(set)
VERTEX_TO_FACES = defaultdict(set)
FACE_TO_FACES = defaultdict(set)
NEW_FACE_DELTA = 1e-3

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

    for edge in EDGE_TO_FACES.keys():
        if len(EDGE_TO_FACES[tuple(sorted(edge))]) == 2:
            f1, f2 = list(EDGE_TO_FACES[tuple(sorted(edge))])
            FACE_TO_FACES[f1].add(f2)
            FACE_TO_FACES[f2].add(f1)  
            
def b_to_w(mesh, tri_idx, b):
    ts =  mesh.vertices[mesh.faces[[tri_idx]]]
    bs = b[np.newaxis, :]
    w = trimesh.triangles.barycentric_to_points(ts, bs)[0]
    return w

def w_to_b(mesh, tri_idx, w):
    ts =  mesh.vertices[mesh.faces[[tri_idx]]]
    ws = w[np.newaxis, :]
    b = trimesh.triangles.points_to_barycentric(ts, ws)[0]
    return b

def get_edge_intersection(mesh, face_idx, b, b_new):
    if np.all(b_new >= -1e-8):
        return None, None

    face = mesh.faces[face_idx]
    intersection = None

    edge_idx = None
    for i in range(3):
        edge_idx = i
        intersection = segment_intersect(bary_to_world(mesh, face_idx, b), \
                                                      bary_to_world(mesh, face_idx, b_new), \
                                                      mesh.vertices[face[i]], \
                                                      mesh.vertices[face[(i+1)%3]])
        if intersection is not None:
            break

    if intersection is None:
        return None, None
    
    edge = frozenset({face[edge_idx], face[(edge_idx+1)%3]})
    return intersection, edge

def basis_gradients(A, B, C):
    e1 = B - A
    e2 = C - A

    n = np.cross(e1, e2)
    norm2 = np.dot(n, n)  # |n|^2

    grad_phi_A = np.cross(n, C - B) / norm2
    grad_phi_B = np.cross(n, A - C) / norm2
    grad_phi_C = np.cross(n, B - A) / norm2

    return grad_phi_A, grad_phi_B, grad_phi_C

def grad_T_w(mesh, distances, tri_idx, b):
    grad_vs = {}
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

    grad_T = (b[0] * grad_vs[0] +
              b[1] * grad_vs[1] +
              b[2] * grad_vs[2])

    return grad_T

def get_bary(p, a, b, c):
    v0 = b - a
    v1 = c - a
    v2 = p - a

    d00 = np.dot(v0, v0)
    d01 = np.dot(v0, v1)
    d11 = np.dot(v1, v1)
    d20 = np.dot(v2, v0)
    d21 = np.dot(v2, v1)

    denom = d00 * d11 - d01 * d01

    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w

    return np.array([u, v, w])

def get_tri_and_coords(mesh, point, previous_face_id=None):
    # We move one face at a time so only surrounding faces
    # need to be checked
    if previous_face_id is not None:
        candidates = [previous_face_id]
        for n in FACE_ADJACENCY[previous_face_id]:
            if n != -1:
                candidates.append(n)
    else:
        candidates = np.arange(len(mesh.faces))

    triangles =  mesh.vertices[mesh.faces[candidates]]
    points = np.tile(point, (len(candidates), 1))

    bary = trimesh.triangles.points_to_barycentric(
        triangles, points, method='cramer')

    valid = np.all(bary >= -1e-8, axis=1)

    if np.any(valid):
        return candidates[np.argmax(valid)], bary[np.argmax(valid)]
    else:
        raise ValueError("Not located in a triangle!")


def solve_grad_at_point(mesh, point, distances, previous_face_id=None):
    # Locate point in triangle
    tri_idx, bary = get_tri_and_coords(mesh, point, previous_face_id)
    triangle = mesh.faces[tri_idx]

    A = mesh.vertices[triangle[0]]
    B = mesh.vertices[triangle[1]]
    C = mesh.vertices[triangle[2]]
    helper_idxs = [triangle[0], triangle[1], triangle[2]]

    for trip_idx in FACE_ADJACENCY[tri_idx]:
        for v in mesh.faces[trip_idx]:
            if v not in helper_idxs:
                helper_idxs.append(v)

    bs = [get_bary(mesh.vertices[p], A, B, C) for p in helper_idxs]
    if (len(bs) != 6):
        return None, None, None
    
    bary_coords = np.array([
       [bs[0][0], bs[0][1], bs[0][2]],
       [bs[1][0], bs[1][1], bs[1][2]],
       [bs[2][0], bs[2][1], bs[2][2]],
       [bs[3][0], bs[3][1], bs[3][2]],
       [bs[4][0], bs[4][1], bs[4][2]],
       [bs[5][0], bs[5][1], bs[5][2]],
    ])

    T = np.array([distances[p] for p in helper_idxs])

    A = np.zeros((6,6))
    for i, (a,b,g) in enumerate(bary_coords):
        A[i] = [a**2, b**2, g**2, a*b, a*g, b*g]

    alpha, beta, gamma = bary
    c = np.linalg.lstsq(A, T, rcond=None)[0]
    c1, c2, c3, c4, c5, c6 = c
    dT_dalpha = 2*c1*alpha + c4*beta + c5*gamma
    dT_dbeta  = 2*c2*beta + c4*alpha + c6*gamma
    dT_dgamma = 2*c3*gamma + c5*alpha + c6*beta
    grad_T =  np.array([dT_dalpha, dT_dbeta, dT_dgamma])
    return grad_T, tri_idx, bary

def trace_geodesic(mesh, distances, source_idx, target_idx, step_size=1e-3):

    # Check if adjacency list has been build
    if len(FACE_TO_FACES.keys()) == 0:
        build_adjacency(mesh)

    f = list(VERTEX_TO_FACES[target_idx])[0]

    # We move slightly off the vertex to avoid hit degeneracies
    local_idx = np.where(mesh.faces[f] == target_idx)[0][0]
    b = np.zeros(3)
    b[local_idx] = 0.99
    b[(local_idx + 1) % 3] = 0.005
    b[(local_idx + 2) % 3] = 0.005
    p = b_to_w(mesh, f, b)
    
    path = [p.copy()]

    h = step_size
    steps = 0
    
    while True:
        steps += 1

        grad_T = grad_T_w(mesh, distances, f, b)


        if source_idx in mesh.faces[f]:
            break

        norm = np.linalg.norm(grad_T)

        direction = -grad_T
        if norm > 0:
            direction /= norm
        move = h * direction
        # # Sonic point
        # if norm < 1e-7 or np.isnan(norm):
        #     # fall back to tiny Euler step in the direction of steepest descent
        #     direction = -grad_bary
        #     if norm > 0:
        #         direction /= norm
        #     move = 1e-6 * direction
        # else:
        #     unit_grad = -grad_bary / norm

        #     # Heun predictor
        #     bary_mid = bary + h * unit_grad
        #     bary_mid = np.clip(bary_mid, 0, None)
        #     bary_mid /= bary_mid.sum()

        #     point_mid = (bary_mid[0] * mesh.vertices[mesh.faces[current_face, 0]] +
        #                  bary_mid[1] * mesh.vertices[mesh.faces[current_face, 1]] +
        #                  bary_mid[2] * mesh.vertices[mesh.faces[current_face, 2]])

        #     # gradient at midpoint
        #     grad_mid, _, _ = solve_grad_at_point(mesh, point_mid, distances, current_face)
        #     norm_mid = np.linalg.norm(grad_mid)
        #     unit_mid = -grad_mid / (norm_mid + 1e-12)

        #     # Heun corrector
        #     move = 0.5 * h * (unit_grad + unit_mid)
            

        p_new = p + move
        b_new = w_to_b(mesh, f, p_new)
        if any(b_new < 0):

            v1_idx = (np.argmin(b_new) + 1)%3
            v2_idx = (v1_idx+1)%3
            v1 = mesh.faces[f][v1_idx]
            v2 = mesh.faces[f][v2_idx]
            edge = (v2, v1)
            nbrs = list(EDGE_TO_FACES[tuple(sorted((v1, v2)))])
            if (len(nbrs) != 2):
                raise ValueError("Edge was found not shared")
            nbr = nbrs[0] if nbrs[0] != f else nbrs[1]
            cross_idx = np.argmin(b_new)
            t = b[cross_idx] / (b[cross_idx] - b_new[cross_idx])
            p_edge = p + t * (p_new - p)

            # Project back onto plane
            A, B, C = [mesh.vertices[v] for v in mesh.faces[f]]
            n = np.cross(B - A, C - A)
            n /= np.linalg.norm(n)
            p_edge = p_edge - np.dot(p_edge - A, n) * n
            
            path.append(p_edge.copy())
            
            p = p_edge
            #print(grad_T)
            #print(f"Crossing {f} to {nbr}")
            f = nbr
            b = w_to_b(mesh, f, p)
        else:
            p = p_new
            b = w_to_b(mesh, f, p)

            # Project back onto plane
            A, B, C = [mesh.vertices[v] for v in mesh.faces[f]]
            n = np.cross(B - A, C - A)
            n /= np.linalg.norm(n)
            p = p - np.dot(p - A, n) * n
            
            path.append(p.copy())


    # Final snap to source vertex (optional, looks nicer)
    path.append(mesh.vertices[source_idx])
    return np.array(path)

def visualize_gradient(mesh, distances, length=0.002):
    if FACE_TO_FACES is None:
        build_adjacency(mesh)
    arrows_start = []
    arrows_end   = []

    ps, fs = trimesh.sample.sample_surface_even(mesh, 5000)
    for p, f in zip(ps, fs):
        # Get gradient at this point
        grad_w       = grad_T_w(mesh, distances, f, w_to_b(mesh, f, p))

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

    # Stack into (N, 2, 3) 
    segments = np.stack([arrows_start, arrows_end], axis=1)

    return segments


def compute_contour_segments(mesh, distances, num_levels=20, spline_points=10):
    V = mesh.vertices
    F = mesh.faces
    min_d, max_d = distances.min(), distances.max()
    delta = (max_d - min_d) / num_levels

    segments_smooth = []

    for f in F:
        verts = V[f]
        dists = distances[f]

        for level in np.arange(min_d, max_d, delta):
            edge_points = []
            # Check each edge of the triangle
            for i, j in [(0,1),(1,2),(2,0)]:
                d1, d2 = dists[i], dists[j]
                if (d1 - level)*(d2 - level) < 0:  # contour crosses this edge
                    t = (level - d1) / (d2 - d1)
                    point = verts[i] + t*(verts[j] - verts[i])
                    edge_points.append(point)

            if len(edge_points) == 2:
                # Fit a small spline (linear will suffice for a triangle)
                pts = np.array(edge_points)
                # Optionally resample points along line
                t = np.linspace(0, 1, spline_points)
                smooth_pts = (1-t)[:,None]*pts[0] + t[:,None]*pts[1]
                segments_smooth.append(smooth_pts)

    if len(segments_smooth) == 0:
        return np.zeros((0, spline_points, 3))

    return np.array(segments_smooth)

def smooth_segments_to_line_segments(smooth_segments):
    """
    Convert smooth contour segments (N, P, 3) into Viser-compatible (M, 2, 3)
    line segments by connecting consecutive points along the spline.
    """
    line_segments = []
    for seg in smooth_segments:
        # seg.shape = (P, 3)
        for i in range(len(seg)-1):
            line_segments.append([seg[i], seg[i+1]])
    return np.array(line_segments, dtype=np.float32)
import numpy as np
from scipy.sparse import csr_matrix

def exact_geodesic_path(V, F, distances, source_idx, target_idx=None, target_face=None, target_bary=None, step_size=1e-6):
    """
    Trace one exact geodesic from target → source using Surazhsky 2005 method.
    V, F            (nv,3)
    F,                (nf,3)
    distances,        (nv,)      precomputed exact geodesic distances from source_idx
    source_idx,       int
    target_idx OR (target_face + target_bary) defines the endpoint
    """
    nv = V.shape[0]

    # Precompute 1-ring
    rows = np.hstack((F[:,0], F[:,1], F[:,2], F[:,1], F[:,2], F[:,0]))
    cols = np.hstack((F[:,1], F[:,2], F[:,0], F[:,0], F[:,1], F[:,2]))
    adj = csr_matrix((np.ones_like(rows), (rows, cols)), shape=(nv,nv))
    vv = [adj[i].indices for i in range(nv)]

    path = []

    # Starting position (the endpoint we trace back from)
    if target_idx is not None:
        # start from a vertex
        current_pos_3d = V[target_idx].copy()
        current_vertex = target_idx
        current_face   = None
        current_bary   = None
    else:
        # start from a point inside a face
        current_face = target_face
        current_bary = np.asarray(target_bary)
        i,j,k = F[current_face]
        current_pos_3d = current_bary[0]*V[i] + current_bary[1]*V[j] + current_bary[2]*V[k]
        current_vertex = None

    path.append(current_pos_3d)

    for _ in range(20000):
        if np.linalg.norm(current_pos_3d - V[source_idx]) < 1e-8:
            path.append(V[source_idx])
            break

        if current_face is None:
            # on vertex → discrete step to best neighbor
            neigh = vv[current_vertex]
            next_v = min(neigh, key=lambda u: distances[u])
            path.append(V[next_v])
            current_vertex = next_v
            current_face = None
            current_pos_3d = V[next_v]
            continue

        # Inside a face → compute quadratic gradient
        f = current_face
        i, j, k = F[f]
        Ti, Tj, Tk = distances[i], distances[j], distances[k]

        # Find opposite vertices D (opp to jk), E (opp to ik), F (opp to ij)
        opp_D = opp_E = opp_F = -1
        for n in vv[j]:
            if n != i and n != k and np.any(np.all(np.sort(F, axis=1) == np.sort([j,k,n]), axis=1)):
                opp_D = n; break
        for n in vv[i]:
            if n != j and n != k and np.any(np.all(np.sort(F, axis=1) == np.sort([i,k,n]), axis=1)):
                opp_E = n; break
        for n in vv[i]:
            if n != j and n != k and np.any(np.all(np.sort(F, axis=1) == np.sort([i,j,n]), axis=1)):
                opp_F = n; break

        if -1 in (opp_D, opp_E, opp_F):
            # fallback: treat as vertex
            current_face = None
            current_vertex = i if current_bary[0] > 0.9 else j if current_bary[1] > 0.9 else k
            continue

        # Six Bézier control values
        t200 = Ti
        t020 = Tj
        t002 = Tk
        t110 = 2*distances[opp_D] - Tj - Tk   # midpoint of j–k
        t101 = 2*distances[opp_E] - Ti - Tk   # midpoint of i–k
        t011 = 2*distances[opp_F] - Ti - Tj   # midpoint of i–j

        α, β, γ = current_bary

        # Gradient of quadratic Bézier patch (Surazhsky Eq. A.7)
        dT_dα = 2*(t200*α + t110*β + t101*γ) - (t110 + t101)
        dT_dβ = 2*(t020*β + t110*α + t011*γ) - (t110 + t011)
        dT_dγ = -(dT_dα + dT_dβ)   # since α+β+γ=1

        # Convert to 3D gradient
        grad = dT_dα * (V[j] - V[k]) + dT_dβ * (V[k] - V[i]) + dT_dγ * (V[i] - V[j])
        grad = grad / (np.linalg.norm(grad) + 1e-12)

        # Take one Heun/Euler step backward along the geodesic
        new_pos = current_pos_3d - step_size * grad * np.linalg.norm(V[i]-V[j])   # adaptive size

        # Project back onto the mesh (closest point)
        _, sqdist, face_id, bc = igl.point_mesh_squared_distance(new_pos[None,:], V, F)
        new_face = face_id[0]
        new_bary  = bc[0]

        current_pos_3d = new_pos
        current_face   = new_face
        current_bary   = new_bary

        path.append(current_pos_3d)

    return np.array(path)

def segment_intersect(p1, p2, q1, q2, eps=1e-10):
    for p in (p1, p2):
        for q in (q1, q2):
            if np.linalg.norm(p - q) < eps:
                #print("SHARED POINT")
                return p 
    d1 = p2 - p1
    d2 = q2 - q1
    d = q1 - p1
    
    denom = np.dot(d1, d1) * np.dot(d2, d2) - np.dot(d1, d2)**2    
    
    t = (np.dot(d, d1) * np.dot(d2, d2) - np.dot(d, d2) * np.dot(d1, d2)) / denom
    s = (np.dot(d, d2) * np.dot(d1, d1) - np.dot(d, d1) * np.dot(d1, d2)) / denom

    #print(t)
    #print(s)
    
    if 0 <= t <= 1 and 0 <= s <= 1:
        # Optional: return intersection point
        intersection = p1 + t * d1
        return intersection #, intersection
    
    return None
