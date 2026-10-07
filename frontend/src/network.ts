export type ExecutionPlan = { mpiRanks: number; cpuSlots: number; gpuCount: number; sharedGpu: boolean; targetNodesPerRank: number; automatic: boolean; benchmarked?: boolean };
export type NetworkGraph = { totalNodes: number; edgeCount: number; edges: number[]; executionPlan?: ExecutionPlan };
export type PartitionInfo = { rank: number; begin: number; end: number; ghostNodes: number; adjacencyEntries: number; gpu: number };
export type PartitionEvent = { kind: 'partition'; mpiRanks: number; crossEdges: number; partitions: PartitionInfo[] };

// These boundaries must match the integer division used by the MPI engine.
export function partitionBounds(nodes: number, ranks: number, rank: number) {
  return [Math.floor(nodes * rank / ranks), Math.floor(nodes * (rank + 1) / ranks)] as const;
}

export function partitionFor(node: number, nodes: number, ranks: number) {
  return Math.min(ranks - 1, Math.ceil((node + 1) * ranks / nodes) - 1);
}

export function clusterCenter(rank: number, ranks: number): [number, number, number] {
  if (ranks === 1) return [0, 0, 0];
  const cols = Math.ceil(Math.sqrt(ranks)), rows = Math.ceil(ranks / cols);
  return [(rank % cols - (cols - 1) / 2) * 230, (Math.floor(rank / cols) - (rows - 1) / 2) * 230, 0];
}
