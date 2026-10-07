# Hybrid MPI-CUDA Misinformation Propagation Simulator

The NetLogo model is the existing sequential work and scientific reference. This
repository contains only the proposed hybrid MPI+CUDA system and its React UI.

## Application flow

1. Enter graph size, mean degree, SBFC probabilities, ticks, and random seed.
2. Generate the complete sparse social network, exported by the native engine.
3. Rotate, zoom, pan, and drag nodes in the 3D WebGL view. Click a node or enter
   its ID to assign it as the only initial Believer.
4. Start the hybrid simulation directly from the React interface.
5. Watch all node colors and complete-network S/B/F counts update live. The view
   separates into C1...Cn according to the actual MPI rank assignments.
6. Review reach, peak Believers, peak tick, graph time, simulation time, and total time.

The browser renders every configured node using WebGL 2 buffers. Below 20,000
nodes, spring layout and spatial-cell repulsion run in a background worker.
Larger graphs use a GPU-resident approximate spring layout with at most eight
sampled neighbours per node, plus a radial spreading force. Position updates stay
on the browser GPU; only a dragged/focused node is read back. This layout sample
changes the picture, never the simulation graph or MPI messages. There is
no 180/400-node preview cap. Use Hide parameters to expand the workspace, or
Fullscreen for the network alone; progress and analytics remain below the view.
Shift-drag or right-drag pans the camera. Pause motion freezes the layout without
stopping the simulation. Hide connections reduces visual clutter and GPU work.

100,000 nodes is a useful large-network starting point. Larger networks depend on
browser memory and the local machine's GPU: visualization runs on the browser's
GPU, while simulation runs on Colab's GPU. All connections are loaded. The
overview draws at most 100,000 evenly sampled connections and explicitly shows
that count; **Draw all connections (slower)** restores the complete edge display.
Large graphs use one-pixel nodes, a capped canvas pixel ratio, a 30-frame-per-second
draw schedule, and GPU picking. Layout pauses during camera/node gestures.
Increase Record every N ticks to reduce
full-network state traffic for large runs. Only the latest node-state snapshot is
retained alongside population-count history.

MPI partitions use the engine's contiguous node-ID ranges. Rank 0 owns C1, rank 1
owns C2, and so on; cross-partition edges remain visible. Each rank holds only its
own adjacency rows and a compact buffer of required remote neighbours (ghost
nodes). The graph appears as one network until the engine finishes partitioning.

MPI ranks are selected automatically. The heuristic targets 25,000 nodes per
rank, normally uses at least two ranks, and caps at eight and twice the available
CPU count. A conservative GPU-memory estimate can lower the count further.
For a typical two-CPU Colab runtime with enough GPU memory, 10,000 nodes selects
two ranks and 100,000 selects four. These are workload partitions, not additional
GPUs; several processes may share one GPU. More ranks do not guarantee speedup.

To compare 2, 4, 8, and 10 ranks on the actual Colab GPU, run this in a new
notebook cell after building the engine (avoid running it during a simulation):

```python
!python3 /content/sbfc-app/engine/tests/benchmark_ranks.py --nodes 1000000 --mean-degree 6 --ticks 100 --repeats 2
```

This can take several minutes because each trial also generates its graph.
It chooses the lowest median simulation time and stores a runtime-local profile.
Regenerate a graph with the same node count and mean degree; automatic selection
uses that measured rank count, including ten if it wins and fits GPU memory.
The profile is tied to GPU UUIDs and rejected on a different runtime or insufficient
free memory. The probe uses default SBFC probabilities and sparse visualization
sampling; different parameters and frame rates can change the fastest choice.
Without a matching profile the conservative heuristic above remains active.

The launcher uses `--oversubscribe --bind-to none` on Linux, allowing multiple
partitions even when Open MPI advertises only one CPU slot. Rank count is
recomputed at run launch, and the UI uses the engine's confirmed assignments.
Per tick, CUDA computes probabilities and state transitions for owned nodes.
`MPI_Alltoallv` exchanges only requested boundary states, `MPI_Reduce` combines
counts, and `MPI_Gatherv` gathers full state only for sampled visualization frames.
All transitions use the previous tick's states, including cross-partition links.

## Architecture

```text
Colab notebook iframe
        |
        v
React dashboard <---- Server-Sent Events ---- Python API
        |                                          |
        +-------------- run request ---------------+
                                                   |
                                                   v
                                          mpirun + CUDA engine
```

## Source layout

```text
engine/src/sbfc_hybrid.cu        Hybrid MPI+CUDA simulation
frontend/                        React and TypeScript application
server/server.py                 Process API and live event stream
colab/Hybrid_MPI_CUDA_SBFC.ipynb Complete Colab deployment notebook
```

## Run the complete application in Colab

Push the project to GitHub, open `colab/Hybrid_MPI_CUDA_SBFC.ipynb` in Colab,
select a T4 GPU runtime, check the repository URL/branch, and run all cells. The
notebook installs Node 22 and compiles for the allocated GPU architecture. The
application cell embeds the React application with fullscreen permission. It remains
available while the Colab runtime is connected. Rebuild the engine and frontend
together after updating this version; the stream now carries all node states.

The build cell runs a small MPI/CUDA equivalence check using 1, 2, 3, and 4 ranks.
It compares exact states and counts and verifies boundary-node metadata. The
single-rank run is a validation reference; the dashboard selects ranks automatically.
A multi-GPU scaling experiment still requires multiple CUDA devices.

## Local UI development

```powershell
cd frontend
npm install
npm run build
cd ..
python server/server.py
```

Open <http://127.0.0.1:8000>. The hybrid engine can only run locally when CUDA
and MPI are installed and `engine/bin/sbfc-hybrid` has been compiled.

## Validation

`python -m unittest discover -s server -v` checks the graph-export protocol,
configuration validation, source IDs outside the old preview, and bounded
snapshot retention. After compiling the engine, run
`python engine/tests/check_graph_export.py` to verify graph integrity and
reproducibility through 100,000 nodes. Graph export does not run CUDA kernels.

Compile `engine/tests/check_partitions.cpp` with a C++14 compiler to verify local
adjacency remapping and synchronous state equivalence for 1–8 partitions,
including uneven node counts and a 100,000-node network. This CPU check exercises
the same partition code and state-update function used by CUDA. On the GPU
runtime, `python engine/tests/check_mpi_equivalence.py` checks the actual MPI/CUDA
exchange path and exact seeded results with different process counts.
