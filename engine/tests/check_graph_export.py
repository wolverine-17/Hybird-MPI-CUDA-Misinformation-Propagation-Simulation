"""Check the compiled engine's full graph export without running CUDA kernels.

Run after building: python engine/tests/check_graph_export.py
"""
import argparse
import json
import os
from pathlib import Path
import subprocess


def main():
    default_engine = Path(__file__).resolve().parents[1] / "bin" / (
        "sbfc-hybrid.exe" if os.name == "nt" else "sbfc-hybrid"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", type=Path, default=default_engine)
    args = parser.parse_args()
    for nodes, degree in [(4, 2), (17, 7), (1000, 6), (100000, 6)]:
        command = [str(args.engine.resolve()), "--nodes", str(nodes),
                   "--mean-degree", str(degree), "--seed", "42", "--graph-only", "1"]
        raw = subprocess.check_output(command, text=True, timeout=120)
        graph = json.loads(raw)
        m = degree // 2
        expected_edges = m * (m + 1) // 2 + (nodes - m - 1) * m
        assert graph["totalNodes"] == nodes
        assert graph["edgeCount"] == expected_edges
        flat = graph["edges"]
        assert len(flat) == expected_edges * 2
        pairs = set(zip(flat[::2], flat[1::2]))
        assert len(pairs) == expected_edges, "Duplicate connections"
        assert all(0 <= a < b < nodes for a, b in pairs), "Invalid node ID or self-link"
        assert len(set(flat)) == nodes, "A configured node is missing from the network"
        assert subprocess.check_output(command, text=True, timeout=120) == raw, "Seed is not reproducible"
        print(f"PASS: {nodes:,} nodes, {expected_edges:,} unique connections")


if __name__ == "__main__":
    main()
