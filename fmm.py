import time
import trimesh
import viser
import heapq
import math
import trimesh
import numpy as np
import viser.transforms as tf
from scipy.interpolate import splprep, splev

def quadratic_solver(a, b, c):
    if (np.square(b) - 4*a*c) < 0.0:
        return [np.inf, np.inf]
    else:
        s1 = (-b + np.sqrt(np.square(b) - 4*a*c)) / (2*a)
        s2 = (-b - np.sqrt(np.square(b) - 4*a*c)) / (2*a)
        return [s1, s2]

def eikonal_update_triangle(A, B, C, TA, TB):
    a = np.linalg.norm(B - C)
    b = np.linalg.norm(A - C)
    AC = A - C
    BC = B - C
    u = TB - TA
    
    cos_theta = np.dot(AC, BC) / (a * b)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    sin2_theta = 1 - cos_theta**2

    fallback = min(TA + b, TB + a)
    
    if cos_theta <= 0.0:
        return fallback

    coeff_A = a**2 + b**2 + 2*a*b*cos_theta
    coeff_B = 2*b*u*(a*cos_theta-b)
    coeff_C = b**2 * (u**2 - a**2*sin2_theta)

    disc = coeff_B**2 - 4*coeff_A*coeff_C

    if disc < 0.0:
        return fallback

    sqrt_d = np.sqrt(disc)
    
    t1 = (-coeff_B + sqrt_d) / (2*coeff_A)
    t2 = (-coeff_B - sqrt_d) / (2*coeff_A)    

    valid = [t for t in (t1, t2) if u < t]
    if valid == []:
        return fallback
    
    t = min(valid)
    if a*cos_theta < (b*(t-u)/t) and \
       (b*(t-u)/t) < a / cos_theta:
       return t + TA
    else:
       return np.min([b+TA, a + TB])
       
def fast_marching(mesh: trimesh.Trimesh, source_idx: int):
    V = mesh.vertices
    F = mesh.faces
    N = len(V)

    distance = np.full(N, np.inf)
    distance[source_idx] = 0.0
    status = np.zeros(N, dtype=np.int8)  # 0=FAR, 1=TRIAL, 2=KNOWN

    heap = []
    status[source_idx] = 2

    for v in mesh.vertex_neighbors[source_idx]:
        d = np.linalg.norm(V[v] - V[source_idx])
        distance[v] = d
        status[v] = 1
        heapq.heappush(heap, (d, v))

    vertex_faces = [[] for _ in range(N)]
    for i, f in enumerate(F):
        for v in f:
            vertex_faces[v].append(i)

    while heap:
        d_min, v = heapq.heappop(heap)
        if status[v] == 2:
            continue
        status[v] = 2

        for f_idx in vertex_faces[v]:
            tri = F[f_idx]
            known = [u for u in tri if status[u] == 2]
            unknown = [u for u in tri if status[u] != 2]

            if len(unknown) != 1 or len(known) < 2:
                continue
            
            u = unknown[0]
            A, B = V[known[0]], V[known[1]]
            C = V[u]
            dA, dB = distance[known[0]], distance[known[1]]

            d_new = eikonal_update_triangle(A, B, C, dA, dB)
            
            if d_new < distance[u]:
                distance[u] = d_new
                heapq.heappush(heap, (d_new, u))
                status[u] = 1

    return distance
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
def geodesic_path(mesh, distances, source_idx, target_idx):
    """
    Compute a geodesic path along the mesh from target_idx to source_idx
    by following the steepest descent of the distance field.
    Returns a list of vertex positions.
    """
    path = [mesh.vertices[target_idx]]
    current = target_idx
    visited = set()
    visited.add(current)

    while current != source_idx:
        neighbors = mesh.vertex_neighbors[current]
        neighbor_distances = distances[neighbors]
        min_idx = neighbors[np.argmin(neighbor_distances)]

        if min_idx in visited:  # safeguard against loops
            break
        visited.add(min_idx)
        path.append(mesh.vertices[min_idx])
        current = min_idx

    return np.array(path)

def path_to_segments(path):
    """
    Convert a path (Nx3 points) to segments (N-1,2,3)
    """
    segments = np.stack([path[:-1], path[1:]], axis=1)
    return segments

def distance_to_colors(distances):
    distances[np.isinf(distances)] = np.nanmax(distances[np.isfinite(distances)])
    dnorm = (distances - distances.min()) / (distances.max() - distances.min())
    colors = np.zeros((len(distances), 4), dtype=np.uint8)
    colors[:,0] = (255 * (1.0 - dnorm)).astype(np.uint8)  # red
    colors[:,2] = (255 * dnorm).astype(np.uint8)          # blue
    colors[:,3] = 255                                     # alpha
    return colors

# TODO ASSERT THAT ITS WATERTIGHT
def main():
    mesh = trimesh.load("bunny.obj")
    components = mesh.split(only_watertight=False)
    mesh = components[0]
    mesh.apply_scale(20)
    wxyz = tf.SO3.from_x_radians(np.pi/2)
    mesh_wxyz = wxyz.wxyz
    R = wxyz.as_matrix()
    mesh_extent = mesh.bounding_box.extents
    mesh_size = np.linalg.norm(mesh_extent)
    sphere_radius = mesh_size * 0.005
    
    # Start Viser server
    server = viser.ViserServer()
    source_input = server.gui.add_number(
    "/source_index",
    min=0,
    max=len(mesh.vertices) - 1,
    step=1,
    initial_value=0,
    )
    
    source_button = server.gui.add_button(
        "/source_index",
    )


    sink_input = server.gui.add_number(
    "/sink_index",
    min=0,
    max=len(mesh.vertices) - 1,
    step=1,
    initial_value=200,
    )
    
    sink_button = server.gui.add_button(
        "/sink_index",
    )


    source_idx = 0
    source_pos_world = R @ mesh.vertices[source_idx]

    sink_idx = 200
    sink_pos_world = R @ mesh.vertices[sink_idx]
    # Create a small red sphere at source position
    source_sphere = server.scene.add_icosphere(
        name="/source_marker",
        radius=sphere_radius,                   # adjust size
        color=(0, 150, 0),            
        position=tuple(source_pos_world),    # initial position
    )

    sink_sphere = server.scene.add_icosphere(
        name="/sink_marker",
        radius=sphere_radius,                   # adjust size
        color=(0, 150, 0),            
        position=tuple(sink_pos_world),    # initial position
    )
    distances = fast_marching(mesh, source_idx)
    mesh.visual.vertex_colors = distance_to_colors(distances)

    # Remove old mesh and add new
    server.scene.remove_by_name("/mesh")
    server.scene.add_mesh_trimesh(
        "/mesh",
        mesh=mesh,
        wxyz=mesh_wxyz,
        position=(0,0,0)
    )

    segments = compute_contour_segments(mesh, distances, num_levels=20)
    segments_viser = smooth_segments_to_line_segments(segments)
    segments_world = np.einsum('ij,nkj->nki', R, segments_viser)
    color = (0, 0, 0)  # black lines
    server.add_line_segments(
        name="/distance_contours",
        points=segments_world,
        colors=color,
        line_width=1.0
    )

    path = geodesic_path(mesh, distances, source_idx, 200)
    path_rotated = np.einsum('ij,nkj->nki', R, path_to_segments(path))
    server.add_line_segments(
    name="/geodesic_path",
    points=path_rotated,
    colors=(255,255,0),  # bright yellow
    line_width=2.0
    )
    def update_model(event):
      new_source = int(source_input.value)
      new_sink = int(sink_input.value)
      
      print("Recomputing FMM with source:", new_source)

      distances = fast_marching(mesh, new_source)
      mesh.visual.vertex_colors = distance_to_colors(distances)

      server.scene.remove_by_name("/mesh")
      server.scene.add_mesh_trimesh(
          "/mesh",
          mesh=mesh,
          wxyz=mesh_wxyz,
          position=(0, 0, 0),
      )
      source_pos_world = R @ mesh.vertices[new_source]
      sink_pos_world = R @ mesh.vertices[new_sink]
      server.scene.remove_by_name("/source_marker")
      server.scene.add_icosphere(
        name="/source_marker",
        radius=sphere_radius,                   # adjust size
        color=(0, 150, 0),            
        position=tuple(source_pos_world),    # initial position
      )
      server.scene.remove_by_name("/sink_marker")
      server.scene.add_icosphere(
        name="/sink_marker",
        radius=sphere_radius,                   # adjust size
        color=(0, 150, 0),
        position=tuple(sink_pos_world),    # initial position
      )
      segments = compute_contour_segments(mesh, distances, num_levels=20)
      segments_world = np.einsum('ij,nkj->nki', R, segments)

      # remove old contours and add new
      server.scene.remove_by_name("/distance_contours")
      server.add_line_segments(
          name="/distance_contours",
          points=segments_world,
          colors=(0,0,0),
          line_width=1.0
      )
      server.scene.remove_by_name("/geodesic_path")
      path = geodesic_path(mesh, distances, new_source, new_sink)
      path_rotated = np.einsum('ij,nkj->nki', R, path_to_segments(path))
      server.add_line_segments(
      name="/geodesic_path",
      points=path_rotated,
      colors=(255,255,0),  # bright yellow
      line_width=2.0
      )
    source_button.on_click(update_model)
    sink_button.on_click(update_model)

    # Keep server alive
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("Server stopped")

if __name__ == "__main__":
    main()

