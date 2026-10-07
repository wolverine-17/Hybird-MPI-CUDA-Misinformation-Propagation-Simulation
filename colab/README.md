# Run the complete application in Colab

1. Push this project to a public GitHub repository.
2. Open `Hybrid_MPI_CUDA_SBFC.ipynb` in Google Colab.
3. Select **Runtime > Change runtime type > T4 GPU**.
4. Check `REPO_URL` and `REPO_BRANCH` in the first code cell. They must point to
   the repository and branch containing the latest changes.
5. Run every cell in order.
6. Use the React application embedded under **Open the application**.

The notebook installs Open MPI and Node.js 22, compiles `sbfc-hybrid` for the
allocated GPU architecture, builds the
React application, starts the Python API, and exposes port 8000 through Colab's
authenticated notebook iframe. No simulation config or result file is moved
between the browser and Colab.

The app is session-bound. It stops when the runtime disconnects, and the iframe
is intended for the user who is signed in to the active notebook. A standard T4
runtime has one GPU. Multiple automatically selected MPI ranks share that GPU;
this demonstrates graph partitioning and message passing on one machine. It does
not create additional GPUs or guarantee a speedup. The launcher permits CPU
oversubscription to avoid Colab's "not enough slots" error.

Generate network loads every node and connection into an interactive 3D WebGL 2
view. Drag the background to rotate, scroll to zoom, Shift-drag to pan, or drag an
individual node. Click a node or enter its ID to select the source. When a run
starts, labels C1...Cn show the engine's actual MPI node ranges. The count is
selected from graph size, CPU availability, and a conservative GPU-memory budget;
there is no manual MPI-rank input. Each process stores its own adjacency rows and
exchanges only required remote-neighbour states with `MPI_Alltoallv`. All links
between subgraphs are retained. Hide parameters
expands the workspace; progress and analytics are underneath. The notebook iframe
requests fullscreen permission; parent browser restrictions may still apply.

Rendering uses your browser's GPU. Start with 10,000 nodes, then try 100,000.
At 20,000 nodes and above, approximate spring layout also runs on the browser
GPU, with positions kept on-device. Every node is drawn. The overview shows at
most 100,000 connections; use **Draw all connections (slower)** for every edge.
All original edges remain in the simulation. Larger/dense networks depend on
browser memory and graphics hardware. Hide
connections to reduce clutter or increase the sampling interval to reduce live
state traffic. There is no capped or sampled node preview.

Generation uses the CPU and sends a compact binary edge buffer to the browser.
CUDA computes simulation ticks after the graph and partitions have been prepared.
The flowing browser layout uses your local graphics hardware, so it does not
appear in Colab's GPU memory panel. That panel reports memory, not compute
utilization. To inspect an actively progressing simulation, run a notebook cell:

```python
!nvidia-smi
!nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv
```

Look for `sbfc-hybrid` compute processes. Short kernel bursts can yield a low
utilization sample, and memory is released when the native run finishes.

For large simulations the live viewer displays the latest snapshot rather than
replaying every older color frame. Node colors use a separate byte buffer,
polled up to four times per second; progress/count history remains in the SSE
stream. At 100,000 nodes or above the engine also caps visual snapshot generation
at four per second, including initial/final states regardless of the cap.
This changes visualization frequency, not CUDA ticks or the requested analytics
samples. Final colors are fetched before the UI reports completion.

After updating, start a fresh Colab runtime and run all cells so the server,
frontend, and native binary use the same version. The build cell checks real
MPI/CUDA results with 1/2/3/4 ranks before the dashboard opens. If this validation
fails, its error output identifies the failing launch or state comparison.

For measured rank selection, add and run a cell after the engine has built:

```python
!python3 /content/sbfc-app/engine/tests/benchmark_ranks.py --nodes 1000000 --mean-degree 6 --ticks 100 --repeats 2
```

It compares 2/4/8/10 ranks, skipping candidates outside the GPU-memory budget.
Allow several minutes and run it while no simulation is active. Regenerate the
same graph size/degree in the app to use the fastest median simulation time.
The result applies only to this GPU runtime; changed probabilities or visualization
sampling can affect performance. The normal conservative selection remains when
there is no matching benchmark. Change `--nodes` to benchmark a different size.
