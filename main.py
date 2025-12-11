from fmm import *
from utils import *
from geodesic import *
import time
import trimesh
import viser
import viser.transforms as tf
import numpy as np

# Mesh setup
MESH_FOLDER = "."
MESH_NAME = "bunny.obj"
MESH_SCALE_FACTOR = 20

SERVER = None
MESH = None
DISTANCES = None

# Size of source and sink point
POINT_SCALE_FACTOR = 0.005

# Rotation matrices
T_WXYZ = tf.SO3.from_x_radians(np.pi/2).wxyz
T_MATRIX = tf.SO3.from_x_radians(np.pi/2).as_matrix()

# General GUI information
class GUI:
    def __init__(self):
        self.source_num = None
        self.source_idx = None
        self.source_btn = None
        self.sink_num = None
        self.sink_idx = None
        self.sink_btn = None

GUI = GUI()

# Create Viser scene
def init_scene():
    global SERVER, DISTANCES

    # Initialize to first and last vertex for source and sink
    GUI.source_idx = source_idx = 0
    source_pos_world = T_MATRIX @ MESH.vertices[source_idx]

    GUI.sink_idx = sink_idx = len(MESH.vertices) - 1
    sink_pos_world = T_MATRIX @ MESH.vertices[sink_idx]


    # Compute distance field
    DISTANCES = fast_marching(MESH, source_idx)

    # Compute color according to distance field
    MESH.visual.vertex_colors = distance_to_colors(DISTANCES)

    # Add mesh
    SERVER.scene.add_mesh_trimesh(
        "/mesh",
        mesh = MESH,
        wxyz=T_WXYZ,
        position=(0,0,0),
    )

    # Add level sets
    contours = trace_contours(MESH, DISTANCES)
    contours = contours @ T_MATRIX.T
    SERVER.scene.add_line_segments(
        "/contours",
        points = contours,
        colors = (0, 0, 0),
    )

    # Add geodesic path
    path = trace_geodesic(MESH, DISTANCES, source_idx, sink_idx)
    path = path @ T_MATRIX.T
    SERVER.scene.add_line_segments(
        "/geodesic",
        points = path,
        colors = (0, 150, 0),
        line_width = 3.0
    )

    # Add points at source and sink
    SERVER.scene.add_icosphere(
        name="/source_marker",
        radius=np.linalg.norm(MESH.bounding_box.extents)*POINT_SCALE_FACTOR,
        color=(0, 150, 0),
        position=tuple(source_pos_world),
    )

    SERVER.scene.add_icosphere(
        name="/sink_marker",
        radius=np.linalg.norm(MESH.bounding_box.extents)*POINT_SCALE_FACTOR,
        color=(0, 150, 0),
        position=tuple(sink_pos_world), 
    )

# Update scene on Viser event
def update(event):
    global SERVER, DISTANCES

    source_not_updated = GUI.source_idx == GUI.source_num.value

    # Update source and sink
    GUI.source_idx = source_idx = int(GUI.source_num.value)
    source_pos_world = T_MATRIX @ MESH.vertices[source_idx]

    GUI.sink_idx = sink_idx = int(GUI.sink_num.value)
    sink_pos_world = T_MATRIX @ MESH.vertices[sink_idx]
    
    # Check if only sink was updated
    # if so, we only need to move the geodesic
    if (source_not_updated):
        path = trace_geodesic(MESH, DISTANCES, source_idx, sink_idx)
        path = path @ T_MATRIX.T

        # Remove old elements
        SERVER.scene.remove_by_name("/geodesic")
        SERVER.scene.remove_by_name("/sink_marker")
        
        # Add new elements
        SERVER.scene.add_line_segments(
            "/geodesic",
            points = path,
            colors = (0, 150, 0),
            line_width = 3.0
        )

        SERVER.scene.add_icosphere(
            name="/sink_marker",
            radius=np.linalg.norm(MESH.bounding_box.extents)*POINT_SCALE_FACTOR,
            color=(0, 150, 0),
            position=tuple(sink_pos_world), 
        )
    else: 
        # Recalculate all fields
        DISTANCES = fast_marching(MESH, source_idx)
        contours = trace_contours(MESH, DISTANCES)
        contours = contours @ T_MATRIX.T
        path = trace_geodesic(MESH, DISTANCES, source_idx, sink_idx)
        path = path @ T_MATRIX.T

        # Compute color according to distance field
        MESH.visual.vertex_colors = distance_to_colors(DISTANCES)

        # Remove old elements
        SERVER.scene.remove_by_name("/mesh")
        SERVER.scene.remove_by_name("/contours")
        SERVER.scene.remove_by_name("/geodesic")
        SERVER.scene.remove_by_name("/source_marker")
        SERVER.scene.remove_by_name("/sink_marker")

        # Add in new elements
        SERVER.scene.add_mesh_trimesh(
            "/mesh",
            mesh = MESH,
            wxyz=T_WXYZ,
            position=(0,0,0),
        )

        SERVER.scene.add_line_segments(
            "/contours",
            points = contours,
            colors = (0, 0, 0),
        )

        SERVER.scene.add_line_segments(
            "/geodesic",
            points = path,
            colors = (0, 150, 0),
            line_width = 3.0
        )

        SERVER.scene.add_icosphere(
            name="/source_marker",
            radius=np.linalg.norm(MESH.bounding_box.extents)*POINT_SCALE_FACTOR,
            color=(0, 150, 0),
            position=tuple(source_pos_world),
        )

        SERVER.scene.add_icosphere(
            name="/sink_marker",
            radius=np.linalg.norm(MESH.bounding_box.extents)*POINT_SCALE_FACTOR,
            color=(0, 150, 0),
            position=tuple(sink_pos_world), 
        )


# Create Viser GUI
def init_gui():
    global GUI

    # Create field for changing source vertex
    GUI.source_num = SERVER.gui.add_number(
        "Source Index",
        min=0,
        max=len(MESH.vertices) - 1,
        step=1,
        initial_value=0,
    )

    #Create button to instantiate change
    GUI.source_btn = SERVER.gui.add_button(
        "Update",
    )

    # Create field for changing sink vertex    
    GUI.sink_num = SERVER.gui.add_number(
        "Destination Index",
        min=0,
        max=len(MESH.vertices) - 1,
        step=1,
        initial_value=len(MESH.vertices) - 1,
    )
    #Create button to instantiate change
    GUI.sink_btn = SERVER.gui.add_button(
        "Update",
    )

    GUI.source_btn.on_click(update)
    GUI.sink_btn.on_click(update)

# Create Viser server
def init():
    global SERVER, MESH

    # Load in first mesh in MESHES
    # ensuring we only consider a connected component
    MESH = trimesh.load(MESH_FOLDER + "/" +  MESH_NAME).split(only_watertight=False)[0]

    # Translate and rotate mesh so it is upright
    # and centroid is at origin
    centroid = MESH.centroid
    MESH.vertices -= centroid
    MESH.apply_scale(MESH_SCALE_FACTOR)

    # Precompute adjacencies
    build_adjacency(MESH)

    # Create server
    SERVER = viser.ViserServer()
    SERVER.scene.enable_default_lights(enabled=False)

    # Initialize GUI and scene
    init_gui()
    init_scene()

def main():

    # Create server
    init()

    # Keep server alive
    try:
        while True:
            continue
    except KeyboardInterrupt:
        print("Server stopped")

if __name__ == "__main__":
    main()
