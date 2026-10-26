import { useEffect, useRef, useState } from "react";
import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { api, DEFAULT_INPUT, FEATURES, PRESETS, pct } from "../api.js";
import { Card, ErrorBox, Gauge, Loading, PageHeader } from "../components/common.jsx";
import ShapBars from "../components/ShapBars.jsx";

const TYPES = [
  { value: "L", label: "Low", hint: "50% of products" },
  { value: "M", label: "Medium", hint: "30% of products" },
  { value: "H", label: "High", hint: "20% of products" },
];

// Debounce a value so slider drags don't flood the API.
function useDebounced(value, delay) {
  const [debounced, setDebounced] = useState(value);
  useEffect(() => {
    const id = setTimeout(() => setDebounced(value), delay);
    return () => clearTimeout(id);
  }, [value, delay]);
  return debounced;
}

function SensorInput({ feature, value, onChange }) {
  return (
    <div className="sensor">
      <div className="sensor-head">
        <label htmlFor={feature.key}>{feature.label}</label>
        <div className="sensor-number">
          <input
            type="number"
            value={value}
            step={feature.step}
            onChange={(e) => onChange(e.target.value === "" ? "" : Number(e.target.value))}
          />
          <span>{feature.unit}</span>
        </div>
      </div>
      <input
        id={feature.key}
        type="range"
        min={feature.min}
        max={feature.max}
        step={feature.step}
        value={value === "" ? feature.min : value}
        onChange={(e) => onChange(Number(e.target.value))}
      />
      <div className="sensor-range">
        <span>{feature.min}</span>
        <span>{feature.max}</span>
      </div>
    </div>
  );
}

function plainLanguage(explanations) {
  if (!explanations?.length) return null;
  const risers = explanations.filter((e) => e.shap_value > 0).slice(0, 2);
  const lowerers = explanations.filter((e) => e.shap_value < 0).slice(0, 2);
  const parts = [];
  if (risers.length) parts.push(`${risers.map((e) => e.feature).join(" and ")} push the risk up`);
  if (lowerers.length) parts.push(`${lowerers.map((e) => e.feature).join(" and ")} pull it down`);
  return parts.join(", while ") + ".";
}

export default function Predict() {
  const [input, setInput] = useState(DEFAULT_INPUT);
  const [prediction, setPrediction] = useState(null);
  const [predError, setPredError] = useState(null);
  const [explanation, setExplanation] = useState(null);
  const [explaining, setExplaining] = useState(false);
  const [autoExplain, setAutoExplain] = useState(true);
  const [sweepFeature, setSweepFeature] = useState("tool_wear");
  const [sweep, setSweep] = useState(null);
  const [sample, setSample] = useState(null);
  const [history, setHistory] = useState([]);
  const requestId = useRef(0);

  const valid = FEATURES.every((f) => input[f.key] !== "" && !Number.isNaN(input[f.key]));
  const debounced = useDebounced(input, 250);
  const debouncedSlow = useDebounced(input, 700);

  // Live prediction
  useEffect(() => {
    if (!valid) return;
    const id = ++requestId.current;
    api
      .predict(debounced)
      .then((res) => {
        if (id !== requestId.current) return;
        setPrediction(res);
        setPredError(null);
      })
      .catch((err) => id === requestId.current && setPredError(err));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [debounced]);

  // SHAP explanation (slower, so it waits longer)
  const runExplain = (payload) => {
    setExplaining(true);
    api
      .explain(payload)
      .then((res) => {
        setExplanation(res);
        setHistory((h) =>
          [{ at: new Date(), input: payload, probability: res.failure_probability, result: res.result, top: res.explanations[0]?.feature }, ...h].slice(0, 8)
        );
      })
      .catch(() => setExplanation(null))
      .finally(() => setExplaining(false));
  };

  useEffect(() => {
    if (autoExplain && valid) runExplain(debouncedSlow);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [debouncedSlow, autoExplain]);

  // What-if sweep
  useEffect(() => {
    if (!valid) return;
    api
      .sweep(debouncedSlow, sweepFeature, 50)
      .then(setSweep)
      .catch(() => setSweep(null));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [debouncedSlow, sweepFeature]);

  const update = (key, value) => {
    setSample(null);
    setInput((prev) => ({ ...prev, [key]: value }));
  };

  const loadSample = async (failure) => {
    try {
      const s = await api.datasetSample(failure);
      setSample(s);
      setInput(s.input);
    } catch (err) {
      setPredError(err);
    }
  };

  const sweepMeta = FEATURES.find((f) => f.key === sweepFeature);
  const danger = prediction?.prediction === 1;

  return (
    <>
      <PageHeader
        title="Predict & Explain"
        subtitle="Adjust the machine's sensor readings. The MLP model re-scores the failure risk live and SHAP explains why."
      />

      <div className="preset-bar">
        <span className="muted">Scenarios:</span>
        {PRESETS.map((p) => (
          <button key={p.name} className="chip" onClick={() => { setSample(null); setInput(p.input); }}>
            {p.name}
          </button>
        ))}
        <span className="divider" />
        <button className="chip outline" onClick={() => loadSample(1)}>🎲 Real failure record</button>
        <button className="chip outline" onClick={() => loadSample(0)}>🎲 Real healthy record</button>
      </div>

      {sample && (
        <div className={`sample-banner ${sample.actual_failure ? "danger" : "safe"}`}>
          Loaded dataset record <strong>#{sample.udi}</strong> ({sample.product_id}). Actual outcome:{" "}
          <strong>{sample.actual_failure ? "Failure" : "No failure"}</strong>
          {sample.failure_modes.length > 0 && <> · modes: {sample.failure_modes.join(", ")}</>}
          {prediction && (
            <>
              {" "}
              → model says <strong>{prediction.result}</strong>{" "}
              {prediction.prediction === sample.actual_failure ? "✅ correct" : "❌ mismatch"}
            </>
          )}
        </div>
      )}

      <div className="grid predict-grid">
        <Card title="Sensor readings" subtitle="Machine configuration and operating conditions">
          <div className="type-picker">
            <span className="field-label">Machine type (product quality)</span>
            <div className="segmented">
              {TYPES.map((t) => (
                <button
                  key={t.value}
                  className={input.type === t.value ? "active" : ""}
                  onClick={() => update("type", t.value)}
                  title={t.hint}
                >
                  <strong>{t.value}</strong> {t.label}
                </button>
              ))}
            </div>
          </div>
          {FEATURES.map((f) => (
            <SensorInput key={f.key} feature={f} value={input[f.key]} onChange={(v) => update(f.key, v)} />
          ))}
          <div className="derived">
            <span>
              ΔT (process − air): <strong>{(input.process_temperature - input.air_temperature).toFixed(1)} K</strong>
            </span>
            <span>
              Power: <strong>{((input.torque * input.rotational_speed * 2 * Math.PI) / 60 / 1000).toFixed(2)} kW</strong>
            </span>
          </div>
        </Card>

        <div className="stack">
          <Card title="Failure risk" className={`result-card ${danger ? "danger" : "safe"}`}>
            {predError && <ErrorBox error={predError} />}
            {!predError && !prediction && <Loading label="Scoring…" />}
            {prediction && !predError && (
              <div className="result">
                <Gauge value={prediction.failure_probability} />
                <div className={`verdict ${danger ? "danger" : "safe"}`}>
                  {danger ? "⚠ Likely Failure" : "✓ No Failure Expected"}
                </div>
                <p className="muted">
                  Model confidence {pct(prediction.confidence)} · decision threshold 50%
                </p>
              </div>
            )}
          </Card>

          <Card
            title="Why? — SHAP explanation"
            subtitle="Contribution of each feature to this prediction"
            actions={
              <div className="row gap">
                <label className="toggle">
                  <input type="checkbox" checked={autoExplain} onChange={(e) => setAutoExplain(e.target.checked)} />
                  Auto
                </label>
                {!autoExplain && (
                  <button className="btn" disabled={explaining || !valid} onClick={() => runExplain(input)}>
                    Explain
                  </button>
                )}
              </div>
            }
          >
            {explaining && !explanation && <Loading label="Computing SHAP values…" />}
            {explanation && (
              <div className={explaining ? "faded" : ""}>
                <p className="insight">{plainLanguage(explanation.explanations)}</p>
                <ShapBars items={explanation.explanations} />
              </div>
            )}
          </Card>
        </div>
      </div>

      <Card
        title="What-if analysis"
        subtitle="How the failure probability changes if one reading varies while the others stay fixed"
        actions={
          <select value={sweepFeature} onChange={(e) => setSweepFeature(e.target.value)}>
            {FEATURES.map((f) => (
              <option key={f.key} value={f.key}>
                {f.label}
              </option>
            ))}
          </select>
        }
      >
        {!sweep ? (
          <Loading />
        ) : (
          <ResponsiveContainer width="100%" height={280}>
            <LineChart data={sweep.points} margin={{ top: 10, right: 20, bottom: 20, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--grid)" />
              <XAxis
                dataKey="value"
                type="number"
                domain={["dataMin", "dataMax"]}
                tickFormatter={(v) => Math.round(v)}
                stroke="var(--muted)"
                label={{ value: `${sweepMeta.label} (${sweepMeta.unit})`, position: "insideBottom", offset: -10, fill: "var(--muted)" }}
              />
              <YAxis domain={[0, 1]} tickFormatter={(v) => `${v * 100}%`} stroke="var(--muted)" />
              <Tooltip
                formatter={(v) => [pct(v), "Failure probability"]}
                labelFormatter={(v) => `${sweepMeta.label}: ${v} ${sweepMeta.unit}`}
                contentStyle={{ background: "var(--surface)", border: "1px solid var(--border)" }}
              />
              <ReferenceLine y={0.5} stroke="var(--danger)" strokeDasharray="4 4" label={{ value: "threshold", fill: "var(--danger)", position: "right" }} />
              <ReferenceLine x={input[sweepFeature]} stroke="var(--accent)" label={{ value: "current", fill: "var(--accent)", position: "top" }} />
              <Line type="monotone" dataKey="failure_probability" stroke="var(--accent)" strokeWidth={2.5} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        )}
      </Card>

      {history.length > 0 && (
        <Card title="Recent predictions" subtitle="Click a row to restore those readings">
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Time</th>
                  <th>Type</th>
                  <th>Air K</th>
                  <th>Proc K</th>
                  <th>RPM</th>
                  <th>Torque</th>
                  <th>Wear</th>
                  <th>Risk</th>
                  <th>Top driver</th>
                </tr>
              </thead>
              <tbody>
                {history.map((h, i) => (
                  <tr key={i} className="clickable" onClick={() => setInput(h.input)}>
                    <td>{h.at.toLocaleTimeString()}</td>
                    <td>{h.input.type}</td>
                    <td>{h.input.air_temperature}</td>
                    <td>{h.input.process_temperature}</td>
                    <td>{h.input.rotational_speed}</td>
                    <td>{h.input.torque}</td>
                    <td>{h.input.tool_wear}</td>
                    <td>
                      <span className={`pill ${h.probability >= 0.5 ? "danger" : "safe"}`}>{pct(h.probability)}</span>
                    </td>
                    <td>{h.top}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}
    </>
  );
}
