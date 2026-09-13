# Run the complete application in Colab

1. Push this project to a public GitHub repository.
2. Open `Hybrid_MPI_CUDA_SBFC.ipynb` in Google Colab.
3. Select **Runtime > Change runtime type > T4 GPU**.
4. Replace `REPO_URL` in the first code cell with the repository clone URL.
5. Run every cell in order.
6. Use the React application embedded under **Open the application**.

The notebook installs Open MPI and Node.js, compiles `sbfc-hybrid`, builds the
React application, starts the Python API, and exposes port 8000 through Colab's
authenticated notebook iframe. No simulation config or result file is moved
between the browser and Colab.

The app is session-bound. It stops when the runtime disconnects, and the iframe
is intended for the user who is signed in to the active notebook. A standard T4
runtime has one GPU, so use one MPI rank. More ranks only demonstrate real
multi-GPU distribution when the runtime provides multiple CUDA devices.
