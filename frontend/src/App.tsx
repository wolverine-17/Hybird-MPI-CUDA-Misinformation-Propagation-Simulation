import { useMemo, useRef, useState } from "react";

type Progress = {
  kind: "progress";
  tick: number;
  susceptible: number;
  believers: number;
  factCheckers: number;
  reached: number;
  maxBelievers: number;
  maxBelieversTick: number;
  elapsedSeconds: number;
  previewStates?: number[];
};
type Summary = {
  kind: "summary";
  nodes: number;
  edges: number;
  ticks: number;
  susceptible: number;
  believers: number;
  factCheckers: number;
  reached: number;
  maxBelievers: number;
  maxBelieversTick: number;
  graphSeconds: number;
  simulationSeconds: number;
  totalSeconds: number;
  mpiRanks?: number;
  gpuCount?: number;
  sourceNode?: number;
  seed?: number;
  meanDegree?: number;
};
type Preview = {
  nodes: { id: number; x: number; y: number }[];
  edges: [number, number][];
  totalNodes: number;
  shownNodes: number;
};
type Config = {
  nodes: number;
  meanDegree: number;
  ticks: number;
  sampleEvery: number;
  alpha: number;
  beta: number;
  verify: number;
  forget: number;
  seed: number;
  sourceNode: number;
  previewNodes: number;
  mpiRanks: number;
};

const initialConfig: Config = {
  nodes: 10000,
  meanDegree: 6,
  ticks: 300,
  sampleEvery: 1,
  alpha: 0.3,
  beta: 0.5,
  verify: 0.05,
  forget: 0.1,
  seed: 42,
  sourceNode: -1,
  previewNodes: 180,
  mpiRanks: 1,
};
const compact = new Intl.NumberFormat("en", {
  notation: "compact",
  maximumFractionDigits: 1,
});

function StateChart({ data, nodes }: { data: Progress[]; nodes: number }) {
  const w = 900,
    h = 330,
    p = { l: 58, r: 22, t: 18, b: 40 },
    mt = Math.max(1, data.at(-1)?.tick ?? 1),
    x = (v: number) => p.l + (v / mt) * (w - p.l - p.r),
    y = (v: number) => p.t + (1 - v / Math.max(1, nodes)) * (h - p.t - p.b),
    path = (k: "susceptible" | "believers" | "factCheckers") =>
      data
        .map(
          (d, i) =>
            `${i ? "L" : "M"}${x(d.tick).toFixed(1)},${y(d[k]).toFixed(1)}`,
        )
        .join(" ");
  return (
    <div className="chart-wrap">
      <svg
        viewBox={`0 0 ${w} ${h}`}
        role="img"
        aria-label="Replayed S, B and F population counts"
      >
        {[0, 0.25, 0.5, 0.75, 1].map((r) => (
          <g key={r}>
            <line
              className="grid"
              x1={p.l}
              x2={w - p.r}
              y1={y(nodes * r)}
              y2={y(nodes * r)}
            />
            <text
              className="axis-label"
              x={p.l - 10}
              y={y(nodes * r) + 4}
              textAnchor="end"
            >
              {compact.format(nodes * r)}
            </text>
          </g>
        ))}
        <line className="axis" x1={p.l} x2={p.l} y1={p.t} y2={h - p.b} />
        <line
          className="axis"
          x1={p.l}
          x2={w - p.r}
          y1={h - p.b}
          y2={h - p.b}
        />
        {data.length > 1 && (
          <>
            <path className="series susceptible" d={path("susceptible")} />
            <path className="series believers" d={path("believers")} />
            <path className="series fact-checkers" d={path("factCheckers")} />
          </>
        )}
        <text className="axis-title" x={w / 2} y={h - 7} textAnchor="middle">
          Simulation tick
        </text>
        <text className="axis-label" x={p.l} y={h - p.b + 20}>
          0
        </text>
        <text
          className="axis-label"
          x={w - p.r}
          y={h - p.b + 20}
          textAnchor="end"
        >
          {mt}
        </text>
      </svg>
    </div>
  );
}

function NetworkPreview({
  graph,
  selected,
  states,
  disabled,
  onSelect,
}: {
  graph: Preview;
  selected: number;
  states?: number[];
  disabled: boolean;
  onSelect: (id: number) => void;
}) {
  const byId = new Map(graph.nodes.map((n) => [n.id, n]));
  return (
    <div className="network-wrap">
      <svg
        viewBox="-15 -15 130 130"
        role="img"
        aria-label={`Selectable preview of ${graph.shownNodes} of ${graph.totalNodes} nodes`}
      >
        <g className="network-edges">
          {graph.edges.map(([a, b], i) => {
            const u = byId.get(a)!,
              v = byId.get(b)!;
            return <line key={i} x1={u.x} y1={u.y} x2={v.x} y2={v.y} />;
          })}
        </g>
        <g>
          {graph.nodes.map((n, i) => {
            const state = states?.[i] ?? 0,
              cls = state === 1 ? "node-b" : state === 2 ? "node-f" : "node-s";
            return (
              <circle
                key={n.id}
                className={`${cls} ${selected === n.id ? "selected" : ""}`}
                cx={n.x}
                cy={n.y}
                r={selected === n.id ? 2.25 : 1.35}
                role="button"
                aria-label={`Select node ${n.id} as source`}
                onClick={() => !disabled && onSelect(n.id)}
              />
            );
          })}
        </g>
      </svg>
      <p>
        {graph.shownNodes.toLocaleString()} selectable nodes shown from a{" "}
        {graph.totalNodes.toLocaleString()}-node network
      </p>
    </div>
  );
}

function NumberField({
  label,
  name,
  value,
  step = 1,
  min,
  max,
  onChange,
}: {
  label: string;
  name: keyof Config;
  value: number;
  step?: number;
  min?: number;
  max?: number;
  onChange: (n: keyof Config, v: number) => void;
}) {
  return (
    <label className="field">
      <span>{label}</span>
      <input
        type="number"
        value={value}
        step={step}
        min={min}
        max={max}
        onChange={(e) => onChange(name, Number(e.target.value))}
      />
    </label>
  );
}

export default function App() {
  const [config, setConfig] = useState(initialConfig),
    [graph, setGraph] = useState<Preview | null>(null),
    [frames, setFrames] = useState<Progress[]>([]),
    [summary, setSummary] = useState<Summary | null>(null),
    [running, setRunning] = useState(false),
    [status, setStatus] = useState("Configure the network"),
    [error, setError] = useState("");
  const eventSource = useRef<EventSource | null>(null);
  const current = frames.at(-1),
    percent = Math.min(
      100,
      ((current?.tick ?? 0) / (summary?.ticks ?? config.ticks)) * 100,
    );
  const metrics = useMemo(
    () =>
      [
        ["Susceptible", current?.susceptible ?? config.nodes, "s"],
        ["Believers", current?.believers ?? 0, "b"],
        ["Fact-checkers", current?.factCheckers ?? 0, "f"],
      ] as const,
    [current, config.nodes],
  );
  const update = (name: keyof Config, value: number) => {
    const reset = ["nodes", "meanDegree", "seed"].includes(name);
    setConfig((old) => ({
      ...old,
      [name]: value,
      sourceNode: reset ? -1 : old.sourceNode,
    }));
    if (reset) setGraph(null);
  };
  async function prepareGraph() {
    setError("");
    setStatus("Generating graph preview");
    try {
      const q = new URLSearchParams({
          nodes: String(config.nodes),
          meanDegree: String(config.meanDegree),
          seed: String(config.seed),
          limit: String(config.previewNodes),
        }),
        r = await fetch(`/api/graph-preview?${q}`),
        data = await r.json();
      if (!r.ok) throw new Error(data.error);
      setGraph(data);
      setFrames([]);
      setSummary(null);
      setConfig((old) => ({ ...old, sourceNode: -1 }));
      setStatus("Select one node as the Believer source");
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setStatus("Preview failed");
    }
  }
  async function runSimulation() {
    if (config.sourceNode < 0 || running) return;
    setError("");
    setFrames([]);
    setSummary(null);
    setRunning(true);
    setStatus("Launching MPI + CUDA engine");
    eventSource.current?.close();
    try {
      const response = await fetch("/api/simulations", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(config),
      });
      const result = await response.json();
      if (!response.ok) throw new Error(result.error ?? "Simulation request failed");

      const source = new EventSource(`/api/simulations/${result.id}/events`);
      eventSource.current = source;
      source.onmessage = (message) => {
        const event = JSON.parse(message.data) as Progress | Summary | { kind: "error"; message: string };
        if (event.kind === "progress") {
          setFrames((old) => [...old, event]);
          setStatus(`Hybrid simulation: tick ${event.tick} / ${config.ticks}`);
        } else if (event.kind === "summary") {
          setSummary(event);
          setRunning(false);
          setStatus("Simulation completed");
          source.close();
        } else {
          setError(event.message);
          setRunning(false);
          setStatus("Simulation failed");
          source.close();
        }
      };
      source.onerror = () => {
        if (source.readyState === EventSource.CLOSED) return;
        setError("Connection to the Colab simulation stream was interrupted.");
        setRunning(false);
        setStatus("Connection interrupted");
        source.close();
      };
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      setRunning(false);
      setStatus("Simulation failed");
    }
  }
  return (
    <main>
      <header>
        <div>
          <p className="eyebrow">MPI + CUDA misinformation laboratory</p>
          <h1>SBFC Propagation Simulator</h1>
          <p className="subtitle">
            Select one source node and run the MPI + CUDA model directly on the
            active Colab GPU.
          </p>
        </div>
        <div
          className={`run-state ${running ? "active" : ""}`}
          aria-live="polite"
        >
          <span />
          {status}
        </div>
      </header>
      <div className="layout">
        <aside className="controls">
          <section>
            <h2>Network and execution</h2>
            <NumberField
              label="Users / nodes"
              name="nodes"
              value={config.nodes}
              min={4}
              max={10000000}
              onChange={update}
            />
            <NumberField
              label="Mean degree"
              name="meanDegree"
              value={config.meanDegree}
              min={2}
              onChange={update}
            />
            <NumberField
              label="MPI ranks (1 for Colab T4)"
              name="mpiRanks"
              value={config.mpiRanks}
              min={1}
              max={64}
              onChange={update}
            />
            <NumberField
              label="Simulation ticks"
              name="ticks"
              value={config.ticks}
              min={1}
              onChange={update}
            />
            <NumberField
              label="Record every N ticks"
              name="sampleEvery"
              value={config.sampleEvery}
              min={1}
              onChange={update}
            />
            <NumberField
              label="Preview nodes"
              name="previewNodes"
              value={config.previewNodes}
              min={20}
              max={400}
              onChange={update}
            />
            <NumberField
              label="Random seed"
              name="seed"
              value={config.seed}
              min={0}
              onChange={update}
            />
            <button type="button" className="secondary" onClick={prepareGraph}>
              1. Generate graph preview
            </button>
          </section>
          <section>
            <h2>SBFC parameters</h2>
            <NumberField
              label="Credibility α"
              name="alpha"
              value={config.alpha}
              min={0}
              max={0.99}
              step={0.01}
              onChange={update}
            />
            <NumberField
              label="Spreading β"
              name="beta"
              value={config.beta}
              min={0}
              max={1}
              step={0.01}
              onChange={update}
            />
            <NumberField
              label="Verification p_v"
              name="verify"
              value={config.verify}
              min={0}
              max={1}
              step={0.01}
              onChange={update}
            />
            <NumberField
              label="Forgetting p_f"
              name="forget"
              value={config.forget}
              min={0}
              max={1}
              step={0.01}
              onChange={update}
            />
          </section>
          <div className="source-readout">
            Source Believer{" "}
            <strong>
              {config.sourceNode < 0
                ? "Not selected"
                : `Node ${config.sourceNode}`}
            </strong>
          </div>
          <button
            className="primary"
            disabled={!graph || config.sourceNode < 0 || running}
            onClick={runSimulation}
          >
            {running ? "Running MPI + CUDA…" : "2. Run hybrid simulation"}
          </button>
        </aside>
        <section className="workspace">
          {graph ? (
            <>
              <div className="chart-heading">
                <div>
                  <h2>Selectable network preview</h2>
                  <p>Click one node to make it the only initial Believer</p>
                </div>
                <div className="legend">
                  <span className="s">S</span>
                  <span className="b">B</span>
                  <span className="f">F</span>
                </div>
              </div>
              <NetworkPreview
                graph={graph}
                selected={config.sourceNode}
                states={current?.previewStates}
                disabled={running}
                onSelect={(id) => {
                  setConfig((old) => ({ ...old, sourceNode: id }));
                  setStatus(`Node ${id} selected as the only Believer`);
                }}
              />
            </>
          ) : (
            <div className="empty-network">
              <strong>No graph prepared</strong>
              <span>
                Set the network inputs and generate a selectable preview.
              </span>
            </div>
          )}
          <div className="metric-row">
            {metrics.map(([label, value, c]) => (
              <article key={label} className={`metric ${c}`}>
                <span>{label}</span>
                <strong>{compact.format(value)}</strong>
              </article>
            ))}
          </div>
          <div
            className="progress-track"
            aria-label={`${percent.toFixed(0)} percent complete`}
          >
            <div style={{ width: `${percent}%` }} />
          </div>
          <div className="chart-heading">
            <div>
              <h2>Population state over time</h2>
              <p>
                {current
                  ? `Live tick ${current.tick}; hybrid time ${current.elapsedSeconds.toFixed(3)} s`
                  : "Run the model to stream live MPI + CUDA results"}
              </p>
            </div>
          </div>
          <StateChart
            data={frames}
            nodes={summary?.nodes ?? config.nodes}
          />
          {error && (
            <p className="error" role="alert">
              {error}
            </p>
          )}
          {summary && (
            <div className="analysis">
              <div>
                <span>Edges</span>
                <strong>{summary.edges.toLocaleString()}</strong>
              </div>
              <div>
                <span>Users reached</span>
                <strong>{summary.reached.toLocaleString()}</strong>
              </div>
              <div>
                <span>Peak believers</span>
                <strong>
                  {summary.maxBelievers.toLocaleString()} at tick{" "}
                  {summary.maxBelieversTick}
                </strong>
              </div>
              <div>
                <span>Graph construction</span>
                <strong>{summary.graphSeconds.toFixed(3)} s</strong>
              </div>
              <div>
                <span>Hybrid simulation</span>
                <strong>{summary.simulationSeconds.toFixed(3)} s</strong>
              </div>
              <div>
                <span>Total execution</span>
                <strong>{summary.totalSeconds.toFixed(3)} s</strong>
              </div>
            </div>
          )}
        </section>
      </div>
    </main>
  );
}
