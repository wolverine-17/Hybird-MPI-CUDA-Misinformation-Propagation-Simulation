from __future__ import annotations

import json
import math
import os
import subprocess
import tempfile
import threading
import uuid
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
FRONTEND_DIST = ROOT / "frontend" / "dist"
ENGINE_DIR = ROOT / "engine" / "bin"

RUNS: dict[str, dict] = {}
RUNS_LOCK = threading.Lock()

EXECUTABLES = {
    "hybrid": "sbfc-hybrid.exe" if os.name == "nt" else "sbfc-hybrid",
}


def validate_config(raw: dict) -> dict:
    config = {
        "mode": "hybrid",
        "nodes": int(raw.get("nodes", 10000)),
        "meanDegree": int(raw.get("meanDegree", 6)),
        "ticks": int(raw.get("ticks", 300)),
        "sampleEvery": int(raw.get("sampleEvery", 1)),
        "sourceNode": int(raw.get("sourceNode", -1)),
        "previewNodes": int(raw.get("nodes", 10000)),
        "alpha": float(raw.get("alpha", 0.30)),
        "beta": float(raw.get("beta", 0.50)),
        "verify": float(raw.get("verify", 0.05)),
        "forget": float(raw.get("forget", 0.10)),
        "seed": int(raw.get("seed", 42)),
    }
    if config["mode"] not in EXECUTABLES:
        raise ValueError("Unknown execution mode")
    if not 4 <= config["nodes"] <= 10_000_000:
        raise ValueError("Nodes must be between 4 and 10,000,000")
    if not 2 <= config["meanDegree"] < config["nodes"]:
        raise ValueError("Mean degree must be at least 2 and smaller than nodes")
    if not 1 <= config["ticks"] <= 100_000:
        raise ValueError("Ticks must be between 1 and 100,000")
    if not 1 <= config["sampleEvery"] <= config["ticks"]:
        raise ValueError("Sample interval must be between 1 and ticks")
    for key in ("beta", "verify", "forget"):
        if not 0 <= config[key] <= 1:
            raise ValueError(f"{key} must be between 0 and 1")
    if not 0 <= config["alpha"] < 1:
        raise ValueError("alpha must be at least 0 and below 1")
    if not 0 <= config["sourceNode"] < config["nodes"]:
        raise ValueError("Select a valid source node from the graph preview")
    if not 0 <= config["seed"] < 2**64:
        raise ValueError("Seed must be an unsigned 64-bit integer")
    if any(not math.isfinite(config[key]) for key in ("alpha", "beta", "verify", "forget")):
        raise ValueError("Probabilities must be finite")
    return config


def rank_fits_memory(nodes: int, mean_degree: int, ranks: int, gpu_free_mib: list[int]) -> bool:
    if not gpu_free_mib:
        return False
    budget = min(gpu_free_mib) * 1024**2 * 0.70
    processes = math.ceil(ranks / len(gpu_free_mib))
    estimated = nodes * (mean_degree * 4 + 10) + processes * (384 * 1024**2 + nodes)
    return estimated <= budget


def measured_rank_choice(profile: dict, nodes: int, mean_degree: int, gpu_ids: list[str], gpu_free_mib: list[int]) -> int | None:
    if profile.get("nodes") != nodes or profile.get("meanDegree") != mean_degree or profile.get("gpuIds") != gpu_ids or not gpu_ids:
        return None
    ranks = profile.get("mpiRanks")
    if type(ranks) is not int or not 2 <= ranks <= min(10, nodes):
        return None
    return ranks if rank_fits_memory(nodes, mean_degree, ranks, gpu_free_mib) else None


def choose_execution_plan(nodes: int, mean_degree: int, cpu_slots: int, gpu_free_mib: list[int]) -> dict:
    """A bounded heuristic, not a claim that more ranks always run faster."""
    target = 25_000
    desired = max(2, math.ceil(nodes / target))
    ranks = min(nodes, desired, 8, max(2, cpu_slots * 2))
    gpu_count = len(gpu_free_mib)
    if gpu_free_mib:
        # Leave 30% free. Include context overhead and a conservative bound for
        # local CSR plus ghost-state copies on a shared GPU.
        budget = min(gpu_free_mib) * 1024**2 * 0.70
        while ranks > 1:
            per_gpu_processes = math.ceil(ranks / gpu_count)
            estimated = nodes * (mean_degree * 4 + 10) + per_gpu_processes * (384 * 1024**2 + nodes)
            if estimated <= budget:
                break
            ranks -= 1
    return {"mpiRanks": ranks, "targetNodesPerRank": target, "cpuSlots": cpu_slots,
            "gpuCount": gpu_count, "sharedGpu": gpu_count > 0 and ranks > gpu_count,
            "automatic": True}


def automatic_execution_plan(nodes: int, mean_degree: int) -> dict:
    try:
        cpu_slots = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        cpu_slots = os.cpu_count() or 1
    gpu_free_mib = []
    try:
        probe = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                               capture_output=True, text=True, check=True, timeout=5)
        gpu_free_mib = [int(line.strip()) for line in probe.stdout.splitlines() if line.strip()]
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    plan = choose_execution_plan(nodes, mean_degree, cpu_slots, gpu_free_mib)
    try:
        profile = json.loads((ENGINE_DIR / "rank-profile.json").read_text(encoding="utf-8"))
        probe = subprocess.run(["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
                               capture_output=True, text=True, check=True, timeout=5)
        ranks = measured_rank_choice(profile, nodes, mean_degree, probe.stdout.splitlines(), gpu_free_mib)
        if ranks is not None:
            plan.update(mpiRanks=ranks, sharedGpu=ranks > len(gpu_free_mib), benchmarked=True)
    except (OSError, ValueError, TypeError, AttributeError, subprocess.SubprocessError):
        pass
    return plan


def graph_preview(nodes: int, mean_degree: int, seed: int) -> dict:
    """Export the complete graph using the simulation's own generator."""
    validate_config({"nodes": nodes, "meanDegree": mean_degree, "seed": seed, "sourceNode": 0})
    executable = ENGINE_DIR / EXECUTABLES["hybrid"]
    if not executable.exists():
        raise RuntimeError("Build the native engine before generating a network (run the Colab build cell).")
    result = subprocess.run([
        str(executable), "--nodes", str(nodes), "--mean-degree", str(mean_degree),
        "--seed", str(seed), "--graph-only", "1",
    ], cwd=ROOT, capture_output=True, text=True, check=True, timeout=300)
    graph = json.loads(result.stdout)
    graph["executionPlan"] = automatic_execution_plan(nodes, mean_degree)
    return graph


def append_event(run_id: str, event: dict) -> None:
    with RUNS_LOCK:
        run = RUNS[run_id]
        # Keep analytics history, but only retain the latest full-network snapshot.
        # Older events remain immutable for any client currently sending them.
        previous = run.get("state_index")
        if "nodeStates" in event:
            if previous is not None:
                run["events"][previous] = {
                    k: v for k, v in run["events"][previous].items() if k != "nodeStates"
                }
            run["state_index"] = len(run["events"])
        run["events"].append(event)
        if event.get("kind") in {"summary", "error"}:
            run["done"] = True
        run["condition"].notify_all()


def simulation_command(config: dict) -> list[str]:
    executable = ENGINE_DIR / EXECUTABLES[config["mode"]]
    mpi_launcher = os.environ.get("MPIEXEC", "mpiexec" if os.name == "nt" else "mpirun")
    command = [mpi_launcher]
    if os.name != "nt":
        # Colab can expose fewer Open MPI slots than the logical partitions.
        command += ["--oversubscribe", "--bind-to", "none"]
        if os.geteuid() == 0:
            command.append("--allow-run-as-root")
    command += [
        "-np", str(config["mpiRanks"]), str(executable),
        "--nodes", str(config["nodes"]),
        "--mean-degree", str(config["meanDegree"]),
        "--ticks", str(config["ticks"]),
        "--sample-every", str(config["sampleEvery"]),
        "--source-node", str(config["sourceNode"]),
        "--preview-nodes", str(config["previewNodes"]),
        "--alpha", str(config["alpha"]),
        "--beta", str(config["beta"]),
        "--verify", str(config["verify"]),
        "--forget", str(config["forget"]),
        "--seed", str(config["seed"]),
    ]
    return command


def execute_run(run_id: str, config: dict) -> None:
    executable = ENGINE_DIR / EXECUTABLES[config["mode"]]
    if not executable.exists():
        append_event(run_id, {"kind": "error", "message": f"Engine is not built yet: {executable.name}"})
        return
    command = simulation_command(config)

    stderr_file = None
    try:
        # A file avoids a full stderr pipe blocking the engine while stdout is read.
        stderr_file = tempfile.TemporaryFile(mode="w+", encoding="utf-8")
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=stderr_file,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            line = line.strip()
            if line:
                append_event(run_id, json.loads(line))
        return_code = process.wait()
        stderr_file.seek(0)
        stderr = stderr_file.read().strip()
        stderr_file.close()
        if return_code and not RUNS[run_id]["done"]:
            append_event(run_id, {
                "kind": "error",
                "message": stderr or f"Engine exited with status {return_code}",
            })
    except Exception as exc:
        append_event(run_id, {"kind": "error", "message": str(exc)})
    finally:
        if stderr_file is not None:
            stderr_file.close()


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(FRONTEND_DIST), **kwargs)

    def send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/simulations":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            raw = json.loads(self.rfile.read(length) or b"{}")
            config = validate_config(raw)
            plan = automatic_execution_plan(config["nodes"], config["meanDegree"])
            config["mpiRanks"] = plan["mpiRanks"]
            run_id = uuid.uuid4().hex
            condition = threading.Condition(RUNS_LOCK)
            with RUNS_LOCK:
                RUNS[run_id] = {
                    "events": [], "done": False, "condition": condition, "config": config
                }
            threading.Thread(target=execute_run, args=(run_id, config), daemon=True).start()
            self.send_json(HTTPStatus.ACCEPTED, {"id": run_id, "executionPlan": plan})
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            self.send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path in {"/api/graph", "/api/graph-preview"}:
            from urllib.parse import parse_qs
            try:
                query = parse_qs(urlparse(self.path).query)
                payload = graph_preview(
                    int(query.get("nodes", ["10000"])[0]),
                    int(query.get("meanDegree", ["6"])[0]),
                    int(query.get("seed", ["42"])[0]),
                )
                self.send_json(HTTPStatus.OK, payload)
            except (ValueError, TypeError) as exc:
                self.send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            except (RuntimeError, OSError, subprocess.SubprocessError) as exc:
                message = getattr(exc, "stderr", None) or str(exc)
                self.send_json(HTTPStatus.SERVICE_UNAVAILABLE, {"error": message})
            return
        parts = path.strip("/").split("/")
        if len(parts) == 4 and parts[:2] == ["api", "simulations"] and parts[3] == "events":
            self.stream_events(parts[2])
            return
        if path.startswith("/api/"):
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        if path != "/" and not (FRONTEND_DIST / path.lstrip("/")).exists():
            self.path = "/index.html"
        super().do_GET()

    def stream_events(self, run_id: str) -> None:
        with RUNS_LOCK:
            if run_id not in RUNS:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()

        cursor = 0
        try:
            while True:
                with RUNS_LOCK:
                    run = RUNS[run_id]
                    while cursor >= len(run["events"]) and not run["done"]:
                        run["condition"].wait(timeout=15)
                        if cursor >= len(run["events"]) and not run["done"]:
                            self.wfile.write(b": keep-alive\n\n")
                            self.wfile.flush()
                    events = run["events"][cursor:]
                    done = run["done"]
                    cursor = len(run["events"])
                for event in events:
                    payload = json.dumps(event, separators=(",", ":"))
                    self.wfile.write(f"data: {payload}\n\n".encode("utf-8"))
                    self.wfile.flush()
                if done:
                    return
        except (BrokenPipeError, ConnectionResetError):
            return


if __name__ == "__main__":
    if not FRONTEND_DIST.exists():
        print("frontend/dist does not exist; run 'npm run build' in frontend first.")
    host = os.environ.get("SBFC_HOST", "127.0.0.1")
    port = int(os.environ.get("SBFC_PORT", "8000"))
    server = ThreadingHTTPServer((host, port), Handler)
    print(f"SBFC application: http://{host}:{port}")
    server.serve_forever()
