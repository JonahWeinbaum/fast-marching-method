import trimesh
import heapq
import numpy as np
from collections import defaultdict
from geodesic import *

def eikonal_update_triangle(A, B, C, TA, TB):
    # Given two known eikonal solutions, find the third
    # by approximating the front as a plane
    
    a = np.linalg.norm(B - C)
    b = np.linalg.norm(A - C)
    AC = A - C
    BC = B - C
    u = TB - TA
    
    cos_theta = np.dot(AC, BC) / (a * b)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    sin2_theta = 1 - cos_theta**2

    fallback = min(TA + b, TB + a)

    # At degeneracy just use Dijkstra
    if cos_theta <= 0.0:
        return fallback

    coeff_A = a**2 + b**2 + 2*a*b*cos_theta
    coeff_B = 2*b*u*(a*cos_theta-b)
    coeff_C = b**2 * (u**2 - a**2*sin2_theta)

    disc = coeff_B**2 - 4*coeff_A*coeff_C
    
    # At degeneracy just use Dijkstra
    if disc < 0.0:
        return fallback

    sqrt_d = np.sqrt(disc)

    # Solutions to quadratic 
    t1 = (-coeff_B + sqrt_d) / (2*coeff_A)
    t2 = (-coeff_B - sqrt_d) / (2*coeff_A)    

    valid = [t for t in (t1, t2) if u < t]

    # At degeneracy just use Dijkstra
    if valid == []:
        return fallback

    # Compute TC using best t value
    t = min(valid)
    if a*cos_theta < (b*(t-u)/t) and \
       (b*(t-u)/t) < a / cos_theta:
       return t + TA
    else:
       return np.min([b+TA, a + TB])
       
def fast_marching(mesh, source_idx):
    V = mesh.vertices
    F = mesh.faces
    
    N = len(V)

    # Initialize infinite distance for all nodes
    # except source
    distance = np.full(N, np.inf)
    distance[source_idx] = 0.0

    # Initialize status of each node along the front
    status = np.zeros(N, dtype=np.int8)  # 0=FAR, 1=TRIAL, 2=KNOWN
    status[source_idx] = 2

    heap = []

    # Add vertices adjacent to source node
    for v in mesh.vertex_neighbors[source_idx]:
        d = np.linalg.norm(V[v] - V[source_idx])
        distance[v] = d
        status[v] = 1
        heapq.heappush(heap, (d, v))

    while heap:

        # Update front status
        d_min, v = heapq.heappop(heap)
        if status[v] == 2:
            continue
        status[v] = 2

        for f_idx in VERTEX_TO_FACES[v]:
            tri = F[f_idx]
            known = [u for u in tri if status[u] == 2]
            unknown = [u for u in tri if status[u] != 2]

            # We require at least two knowns to compute an unknown
            if len(unknown) != 1 or len(known) < 2:
                continue
            
            u = unknown[0]
            A, B = V[known[0]], V[known[1]]
            C = V[u]
            dA, dB = distance[known[0]], distance[known[1]]

            # Compute distance to third vertex 
            d_new = eikonal_update_triangle(A, B, C, dA, dB)

            # Accept smaller distance
            if d_new < distance[u]:
                distance[u] = d_new
                heapq.heappush(heap, (d_new, u))
                status[u] = 1

    return distance

