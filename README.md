# Hybrid MPI-CUDA Misinformation Propagation Simulator

The NetLogo model is the existing sequential work and scientific reference. This
repository contains only the proposed hybrid MPI+CUDA system and its React UI.

## Application flow

1. Enter graph size, mean degree, SBFC probabilities, ticks, and random seed.
2. Generate a bounded preview of the sparse social network.
3. Click one visible node to assign it as the only initial Believer.
4. Start the hybrid simulation directly from the React interface.
5. Watch the preview-node colors and complete-network S/B/F counts update live.
6. Review reach, peak Believers, peak tick, graph time, simulation time, and total time.

For million-node experiments, the browser displays at most 400 selectable nodes.
The native engine still simulates all nodes. Rendering the complete graph would
recreate NetLogo's visualization bottleneck.

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
select a T4 GPU runtime, replace the repository URL, and run all cells. The final
cell embeds the complete React application. It remains available while the Colab
runtime is connected.

Use one MPI rank on a standard one-GPU Colab runtime. A real multi-GPU scaling
experiment requires a runtime or cluster with one CUDA device per MPI rank.

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
