from fmm import *
from utils import *
from geodesic import *
import time
import trimesh
import viser
import viser.transforms as tf
import numpy as np

MESH_FOLDER = "meshes"
MESHES = ["bunny", "happy"]
MESH_SCALE_FACTOR = 2

SERVER = None
MESH = None

POINT_SCALE_FACTOR = 0.005

T_WXYZ = tf.SO3.from_x_radians(np.pi/2).wxyz
T_MATRIX = tf.SO3.from_x_radians(np.pi/2).as_matrix()

class GUI:
    def __init__(self):
        self.source_num = None
        self.sink_num = None
        self.source_btn = None
        self.sink_btn = None

GUI = GUI()

def init_scene():
    global SERVER

    source_idx = 0
    source_pos_world = T_MATRIX @ MESH.vertices[source_idx]

    sink_idx = len(MESH.vertices) - 1
    sink_pos_world = T_MATRIX @ MESH.vertices[sink_idx]
    # Create a small red sphere at source position
    source_sphere = SERVER.scene.add_icosphere(
        name="/source_marker",
        radius=np.linalg.norm(MESH.bounding_box.extents)*POINT_SCALE_FACTOR,
        color=(0, 150, 0),
        position=tuple(source_pos_world),    # initial position
    )

    sink_sphere = SERVER.scene.add_icosphere(
        name="/sink_marker",
        radius=np.linalg.norm(MESH.bounding_box.extents)*POINT_SCALE_FACTOR,
        color=(0, 150, 0),
        position=tuple(sink_pos_world),    # initial position
    )

    distances = fast_marching(MESH, source_idx)
    MESH.visual.vertex_colors = distance_to_colors(distances)

    SERVER.scene.add_mesh_trimesh(
        "/mesh",
        mesh = MESH,
        wxyz=T_WXYZ,
        position=(0,0,0),
    )

    path = trace_geodesic(MESH, distances, source_idx, sink_idx)
    path = path @ T_MATRIX.T
    path = np.stack([path[:-1], path[1:]], axis=1)
    SERVER.scene.add_line_segments(
        "/gradient",
        points = path,
        colors = (0, 150, 0),
        line_width = 3.0
    )


    
def update_scene(event):
    global SERVER

    source_idx = int(GUI.source_num.value)
    source_pos_world = T_MATRIX @ MESH.vertices[source_idx]

    sink_idx = int(GUI.sink_num.value)
    sink_pos_world = T_MATRIX @ MESH.vertices[sink_idx]
    # Create a small red sphere at source position
    source_sphere = SERVER.scene.add_icosphere(
        name="/source_marker",
        radius=np.linalg.norm(MESH.bounding_box.extents)*POINT_SCALE_FACTOR,
        color=(0, 150, 0),
        position=tuple(source_pos_world),    # initial position
    )

    sink_sphere = SERVER.scene.add_icosphere(
        name="/sink_marker",
        radius=np.linalg.norm(MESH.bounding_box.extents)*POINT_SCALE_FACTOR,
        color=(0, 150, 0),
        position=tuple(sink_pos_world),    # initial position
    )
    distances = fast_marching(MESH, source_idx)
    MESH.visual.vertex_colors = distance_to_colors(distances)

    SERVER.scene.add_mesh_trimesh(
        "/mesh",
        mesh=MESH,
        wxyz=T_WXYZ,
        position=(0,0,0),
        cast_shadow = False,
        receive_shadow = False,
        flat_shading=False,
        material=viser.MeshUnlitMaterial()
    )

def init_gui():
    global GUI
    GUI.source_num = SERVER.gui.add_number(
    "Source Index",
    min=0,
    max=len(MESH.vertices) - 1,
    step=1,
    initial_value=0,
    )


    GUI.source_btn = SERVER.gui.add_button(
        "Update",
    )

    GUI.sink_num = SERVER.gui.add_number(
    "Destination Index",
    min=0,
    max=len(MESH.vertices) - 1,
    step=1,
    initial_value=len(MESH.vertices) - 1,
    )

    GUI.sink_btn = SERVER.gui.add_button(
        "Update",
    )

    GUI.source_btn.on_click(update)
    GUI.sink_btn.on_click(update)

def init():
    global SERVER, MESH

    MESH = trimesh.load(MESH_FOLDER + "/" +  MESHES[0] + ".obj").split(only_watertight=False)[0]
    MESH.apply_scale(MESH_SCALE_FACTOR)
    build_adjacency(MESH)
    
    SERVER = viser.ViserServer()
    SERVER.scene.enable_default_lights(enabled=False)
    init_gui()
    init_scene()

def update(event):
    update_scene(event)

def main():
    MESH = trimesh.load(MESH_FOLDER + "/" +  MESHES[0] + ".obj").split(only_watertight=False)[0]
    init()

    # Keep server alive
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("Server stopped")

if __name__ == "__main__":
    main()
