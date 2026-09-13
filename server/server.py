from __future__ import annotations

import json
import os
import subprocess
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
        "previewNodes": int(raw.get("previewNodes", 180)),
        "mpiRanks": int(raw.get("mpiRanks", 1)),
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
    if not 1 <= config["mpiRanks"] <= 64:
        raise ValueError("MPI ranks must be between 1 and 64")
    if config["mpiRanks"] > config["nodes"]:
        raise ValueError("MPI ranks cannot exceed the number of nodes")
    return config


def graph_preview(nodes: int, mean_degree: int, seed: int, limit: int = 180) -> dict:
    """Return the first nodes of a deterministic sparse BA-style graph for selection."""
    shown = max(4, min(nodes, limit, 400))
    m = max(1, min(mean_degree // 2, shown - 1))
    def mix64(value: int) -> int:
        mask = (1 << 64) - 1
        value = (value + 0x9E3779B97F4A7C15) & mask
        value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
        value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
        return (value ^ (value >> 31)) & mask

    rng_state = seed
    edges: list[list[int]] = []
    degree_pool: list[int] = []
    initial = m + 1
    for a in range(initial):
        for b in range(a + 1, initial):
            edges.append([a, b])
            degree_pool.extend((a, b))
    for node in range(initial, shown):
        selected: list[int] = []
        selected_set: set[int] = set()
        while len(selected) < m:
            rng_state = mix64(rng_state)
            candidate = degree_pool[rng_state % len(degree_pool)]
            if candidate not in selected_set:
                selected.append(candidate)
                selected_set.add(candidate)
        for target in selected:
            edges.append([node, target])
            degree_pool.extend((node, target))
    preview_nodes = []
    for node in range(shown):
        angle = node * 2.399963229728653
        radius = 18 + 42 * ((node + 1) / shown) ** 0.55
        preview_nodes.append({
            "id": node,
            "x": 50 + radius * __import__("math").cos(angle),
            "y": 50 + radius * __import__("math").sin(angle),
        })
    return {"nodes": preview_nodes, "edges": edges, "totalNodes": nodes, "shownNodes": shown}


def append_event(run_id: str, event: dict) -> None:
    with RUNS_LOCK:
        run = RUNS[run_id]
        run["events"].append(event)
        if event.get("kind") in {"summary", "error"}:
            run["done"] = True
        run["condition"].notify_all()


def execute_run(run_id: str, config: dict) -> None:
    executable = ENGINE_DIR / EXECUTABLES[config["mode"]]
    if not executable.exists():
        append_event(run_id, {
            "kind": "error",
            "message": f"{config['mode']} engine is not built yet: {executable.name}",
        })
        return

    mpi_launcher = os.environ.get("MPIEXEC", "mpiexec" if os.name == "nt" else "mpirun")
    command = [mpi_launcher]
    if os.name != "nt" and os.geteuid() == 0:
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

    try:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            line = line.strip()
            if line:
                append_event(run_id, json.loads(line))
        stderr = process.stderr.read().strip() if process.stderr else ""
        return_code = process.wait()
        if return_code and not RUNS[run_id]["done"]:
            append_event(run_id, {
                "kind": "error",
                "message": stderr or f"Engine exited with status {return_code}",
            })
    except Exception as exc:
        append_event(run_id, {"kind": "error", "message": str(exc)})


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
            run_id = uuid.uuid4().hex
            condition = threading.Condition(RUNS_LOCK)
            with RUNS_LOCK:
                RUNS[run_id] = {
                    "events": [], "done": False, "condition": condition, "config": config
                }
            threading.Thread(target=execute_run, args=(run_id, config), daemon=True).start()
            self.send_json(HTTPStatus.ACCEPTED, {"id": run_id})
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            self.send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/graph-preview":
            from urllib.parse import parse_qs
            try:
                query = parse_qs(urlparse(self.path).query)
                payload = graph_preview(
                    int(query.get("nodes", ["10000"])[0]),
                    int(query.get("meanDegree", ["6"])[0]),
                    int(query.get("seed", ["42"])[0]),
                    int(query.get("limit", ["180"])[0]),
                )
                self.send_json(HTTPStatus.OK, payload)
            except (ValueError, TypeError) as exc:
                self.send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
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
