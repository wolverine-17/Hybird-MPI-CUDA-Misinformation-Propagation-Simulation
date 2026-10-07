"""Run in Colab to choose MPI ranks using this runtime and a specified graph size."""
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--nodes', type=int, default=1_000_000)
    parser.add_argument('--mean-degree', type=int, default=6)
    parser.add_argument('--ticks', type=int, default=100)
    parser.add_argument('--repeats', type=int, default=2)
    args = parser.parse_args()
    if not 4 <= args.nodes <= 10_000_000 or args.mean_degree < 2 or args.ticks < 1 or args.repeats < 1:
        parser.error('Use 4–10,000,000 nodes, degree >= 2, and positive ticks/repeats')
    root = Path(__file__).resolve().parents[2]
    import sys
    sys.path.insert(0, str(root / 'server'))
    from server import rank_fits_memory
    gpu_ids = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True).splitlines()
    free = [int(x) for x in subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True).splitlines()]
    if not gpu_ids:
        raise RuntimeError('This benchmark requires a CUDA GPU runtime')
    results = []
    for ranks in (2, 4, 8, 10):
        if ranks > args.nodes or not rank_fits_memory(args.nodes, args.mean_degree, ranks, free):
            print(f'Skipping {ranks} ranks: GPU memory budget', flush=True)
            continue
        trials = []
        for repeat in range(args.repeats):
            command = ['mpirun', '--oversubscribe', '--bind-to', 'none']
            if os.geteuid() == 0:
                command.append('--allow-run-as-root')
            command += ['-np', str(ranks), str(root / 'engine/bin/sbfc-hybrid'), '--nodes', str(args.nodes),
                        '--mean-degree', str(args.mean_degree), '--ticks', str(args.ticks), '--seed', '42',
                        '--source-node', '0', '--sample-every', str(args.ticks)]
            print(f'Benchmarking {ranks} ranks, trial {repeat + 1}/{args.repeats}…', flush=True)
            result = subprocess.run(command, text=True, capture_output=True, timeout=1800)
            if result.returncode:
                print(f'{ranks} ranks failed: {result.stderr[-2000:]}', flush=True)
                break
            summary = next((json.loads(line) for line in reversed(result.stdout.splitlines())
                            if line.startswith('{') and json.loads(line).get('kind') == 'summary'), None)
            if summary is None:
                raise RuntimeError('Engine did not return a summary')
            trials.append(float(summary['simulationSeconds']))
        if len(trials) == args.repeats:
            seconds = statistics.median(trials)
            results.append({'ranks': ranks, 'simulationSeconds': seconds, 'trials': trials})
            print(f'{ranks} ranks: median {seconds:.3f}s for {args.ticks} ticks', flush=True)
    if not results:
        raise RuntimeError('No successful benchmark; existing selection left unchanged')
    winner = min(results, key=lambda r: (r['simulationSeconds'], r['ranks']))
    profile = {'nodes': args.nodes, 'meanDegree': args.mean_degree, 'gpuIds': gpu_ids,
               'ticks': args.ticks, 'mpiRanks': winner['ranks'], 'results': results, 'createdAt': time.time()}
    path = root / 'engine/bin/rank-profile.json'
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(profile, indent=2), encoding='utf-8')
    temporary.replace(path)
    print(f"Selected {winner['ranks']} ranks. Regenerate this size/degree in the app to use the result.")
    print('Timing excludes graph generation; benchmark traffic uses only initial/final state snapshots.')


if __name__ == '__main__':
    main()
