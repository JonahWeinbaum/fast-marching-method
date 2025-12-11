======================================================================
Fast Marching Method (FMM) for Geodesic Distances on Meshes
======================================================================

Description:
------------
Python implementation of the Fast Marching Method (FMM) to compute:
  - Geodesic distance fields
  - Level-set contours
  - Shortest paths between points
Includes an interactive Viser visualizer at http://localhost:8080 where the mesh,
distance field, level sets, and geodesic path can be visualized and changed dynamically.

Citations:
---------
Kimmel, R. & Sethian, J.A. (1998). Computing Geodesic Paths on Manifolds.
Sethian, J.A. (1996). A Fast Marching Level Set Method for Monotonically Advancing Fronts.

File Structure:
------
.
├── main.py             # Main entry point which launches server
├── fmm.py              # Methods for computing distance field via FMM
├── geodesic.py         # Methods for tracing geodesic paths & contours
├── utils.py            # Utility functions for mesh visualization
├── meshes/             # Folder containing mesh files
│   └── bunny.obj       # Stanford bunny as default mesh
│   └── <mesh files>
├── requirements.txt    # Dependencies (trimesh, viser, etc.)


Requirements:
-------------
Python 3.9+
Viser 1.0.16+
Trimesh 4.10.1+

All of which can be installed with
```
pip install -r requirements.txt
```

(Halligan / Remote Server):
----------------------------------
If using Halligan or a remove server, make
sure to forward port `8080` which is the default
port for the web visualizer. This can be done via

```
ssh -L 8080:localhost:8080 <user>@<host>
```

Further, if using Halligan servers, requirments must
be installed in a virtual environment as such

```
python3 -m venv .venv
source .venv/bin/activate  
pip install -r requirements.txt
```

Running:
--------
python3 main.py

Then open http://localhost:8080 in your local browser.

Changing the mesh:
------------------
1. Place mesh file in meshes/
2. Set MESH_NAME variable in main.py
3. Run python3 main.py

Visualizer Features:
-------------------
- Source and target selection
- Geodesic distance field
- Geodesic path rendering
- Geodesic level-sets
