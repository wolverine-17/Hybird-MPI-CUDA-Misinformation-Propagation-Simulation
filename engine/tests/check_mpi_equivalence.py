"""Run on the GPU runtime: verify 1/2/3/4-rank results and boundary metadata."""
import json
import os
from pathlib import Path
import subprocess


def main():
    engine = Path(__file__).resolve().parents[1] / "bin" / "sbfc-hybrid"
    graph = json.loads(subprocess.check_output([
        str(engine), "--nodes", "257", "--mean-degree", "6", "--seed", "42", "--graph-only", "1"
    ], text=True, timeout=120))
    baseline = None
    for ranks in (1, 2, 3, 4):
        command = ["mpirun", "--oversubscribe", "--bind-to", "none"]
        if os.geteuid() == 0:
            command.append("--allow-run-as-root")
        command += ["-np", str(ranks), str(engine), "--nodes", "257", "--mean-degree", "6",
                    "--seed", "42", "--source-node", "0", "--ticks", "24", "--sample-every", "3",
                    "--beta", "0.8", "--forget", "0", "--verify", "0.02"]
        result = subprocess.run(command, text=True, capture_output=True, timeout=180)
        if result.returncode:
            raise RuntimeError(f"{ranks}-rank validation failed:\n{result.stderr}")
        events = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
        partition = next(e for e in events if e["kind"] == "partition")
        owner = lambda node: ((node + 1) * ranks - 1) // 257
        flat = graph["edges"]
        expected_cross = sum(owner(a) != owner(b) for a, b in zip(flat[::2], flat[1::2]))
        assert partition["mpiRanks"] == ranks
        assert partition["crossEdges"] == expected_cross
        for p in partition["partitions"]:
            ghosts = set()
            for a, b in zip(flat[::2], flat[1::2]):
                if p["begin"] <= a < p["end"] and not p["begin"] <= b < p["end"]:
                    ghosts.add(b)
                if p["begin"] <= b < p["end"] and not p["begin"] <= a < p["end"]:
                    ghosts.add(a)
            assert p["ghostNodes"] == len(ghosts)
        states = [{k: v for k, v in event.items() if k != "elapsedSeconds"}
                  for event in events if event["kind"] == "progress"]
        assert len(states) == 9 and states[-1]["tick"] == 24
        assert states[-1]["reached"] > 1
        assert all(len(s["nodeStates"]) == 257 for s in states)
        assert events[-1]["kind"] == "summary"
        if baseline is None:
            baseline = states
        else:
            assert states == baseline, f"{ranks}-rank states/counts differ from the one-rank reference"
        print(f"PASS real MPI/CUDA: {ranks} ranks, {expected_cross} cross-partition links, identical states/counts")

    # Exercise the wall-clock-limited visual path with actual MPI collectives.
    large_reference = None
    for ranks in (2, 4):
        command = ["mpirun", "--oversubscribe", "--bind-to", "none"]
        if os.geteuid() == 0:
            command.append("--allow-run-as-root")
        command += ["-np", str(ranks), str(engine), "--nodes", "100000", "--mean-degree", "6",
                    "--seed", "42", "--source-node", "0", "--ticks", "12", "--sample-every", "1"]
        result = subprocess.run(command, text=True, capture_output=True, timeout=180)
        if result.returncode:
            raise RuntimeError(f"Large-graph {ranks}-rank validation failed:\n{result.stderr}")
        events = [json.loads(line) for line in result.stdout.splitlines() if line.startswith('{')]
        progress = [e for e in events if e['kind'] == 'progress']
        assert [e['tick'] for e in progress] == list(range(13))
        assert len(progress[0]['nodeStates']) == 100000 and len(progress[-1]['nodeStates']) == 100000
        counts = [{k: v for k, v in e.items() if k not in ('elapsedSeconds', 'nodeStates')} for e in progress]
        reference = (counts, progress[-1]['nodeStates'])
        if large_reference is None:
            large_reference = reference
        else:
            assert reference == large_reference, 'Visual throttling changed counts or final states across ranks'
        assert events[-1]['kind'] == 'summary'
        print(f'PASS large-graph MPI/CUDA: {ranks} ranks, every count sample and exact final colors')


if __name__ == "__main__":
    main()
