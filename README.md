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
```
.
├── main.py             # Main entry point which launches server
├── fmm.py              # Methods for computing distance field via FMM
├── geodesic.py         # Methods for tracing geodesic paths & contours
├── utils.py            # Utility functions for mesh visualization
├── bunny.obj            # Stanford bunny mesh as default mesh
├── requirements.txt    # Dependencies (trimesh, viser, etc.)
```

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
1. Place mesh file in root directory
2. Set MESH_NAME variable in main.py
3. Run python3 main.py

Visualizer Features:
-------------------
- Source and target selection
- Geodesic distance field
- Geodesic path rendering
- Geodesic level-sets

Learned Material:
----------------
In implementing the actual FMM method for computing
geodesics I learned about numerical methods to approximate solutions to
differential equations and how they can be viewed as fundamentally
geometric problems.

The FMM method itself is very similar to Dijkstra's
algorithm on graphs so it was much easier implement than I had originally
thought. What was not easy, however, was implementing backtracking
along the gradient to trace out geodesic paths. This required
much more handling of degeneracies in order to get sub-facet segments
included in my path. The original paper regarding FMM for solving
the eikonal equation on meshes is actually fairly vague about
how they handled a lot of these degeneracies, so this was certainly
an exercise in thinking though all the possible cases that can
occur as we move along the manifold, such that we don't accidentally
leave the surface and such that the gradient is always well-defined.
This part taught me the most as it required me to consider many different
possible ways of approximating solutions to differential equations on meshes
and many of them did not end up working. I finally landed on a relatively
simplified strategy which, at it's most basic level, is just Euler's method
for solving ODEs. The paper uses this as well but only at points where the
gradient is not stable (which they call 'sonic points'). I chose to use it everywhere,
and I think the results are more than decent enough from a visual standpoint.

At some point I would like to fully implement their use of Huen's method but it uses many control
points from adjacent triangles and I could not figure out how to handle degeneracies
at the boundary with this method. Over all, this was the most I've ever
had to actually reason about ODEs especially from a geometric point and that
taught me a great deal. 