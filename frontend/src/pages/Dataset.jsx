import { useState } from "react";
import { Bar, BarChart, CartesianGrid, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api, reportImage } from "../api.js";
import { Card, ErrorBox, Loading, PageHeader, Stat, useAsync } from "../components/common.jsx";

const PLOTS = {
  air_temperature: "air_temperature_k",
  process_temperature: "process_temperature_k",
  rotational_speed: "rotational_speed_rpm",
  torque: "torque_nm",
  tool_wear: "tool_wear_min",
};

export default function Dataset() {
  const { data, loading, error } = useAsync(api.datasetSummary);
  const [feature, setFeature] = useState("tool_wear");
  const [view, setView] = useState("vs_failure");

  if (loading) return <Loading />;
  if (error) return <ErrorBox error={error} />;

  const current = data.features.find((f) => f.key === feature);

  return (
    <>
      <PageHeader
        title="Dataset"
        subtitle="AI4I 2020 Predictive Maintenance Dataset — synthetic milling-machine records reflecting real industrial conditions."
      />

      <div className="stats">
        <Stat label="Records" value={data.rows.toLocaleString()} />
        <Stat label="Columns" value={data.columns} hint="6 used as model inputs" />
        <Stat label="Failures" value={data.failures} tone="danger" hint={`${data.failure_rate}% of records`} />
        <Stat label="Healthy" value={data.healthy.toLocaleString()} tone="safe" />
        <Stat label="Missing values" value={data.missing_values} />
      </div>

      <div className="grid two">
        <Card title="Failure modes" subtitle="A record can have more than one mode (excluded from model inputs)">
          <ResponsiveContainer width="100%" height={240}>
            <BarChart data={data.failure_modes} layout="vertical" margin={{ left: 40, right: 20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--grid)" horizontal={false} />
              <XAxis type="number" stroke="var(--muted)" fontSize={12} />
              <YAxis type="category" dataKey="name" stroke="var(--muted)" fontSize={12} width={150} />
              <Tooltip contentStyle={{ background: "var(--surface)", border: "1px solid var(--border)" }} />
              <Bar dataKey="count" fill="var(--danger)" radius={[0, 4, 4, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </Card>

        <Card title="Failure rate by machine type" subtitle="L = low, M = medium, H = high product quality">
          <ResponsiveContainer width="100%" height={240}>
            <BarChart data={data.types} margin={{ right: 20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--grid)" vertical={false} />
              <XAxis dataKey="type" stroke="var(--muted)" />
              <YAxis unit="%" stroke="var(--muted)" fontSize={12} />
              <Tooltip
                formatter={(v, _n, p) => [`${v}% (${p.payload.failures} of ${p.payload.count})`, "Failure rate"]}
                contentStyle={{ background: "var(--surface)", border: "1px solid var(--border)" }}
              />
              <Bar dataKey="failure_rate" radius={[4, 4, 0, 0]}>
                {data.types.map((t) => (
                  <Cell key={t.type} fill={t.type === "L" ? "#ff318c" : t.type === "M" ? "#7b61ff" : "#087cfa"} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </Card>
      </div>

      <Card
        title="Sensor feature explorer"
        actions={
          <div className="row gap wrap">
            <div className="segmented small">
              {data.features.map((f) => (
                <button key={f.key} className={feature === f.key ? "active" : ""} onClick={() => setFeature(f.key)}>
                  {f.column.replace(/ \[.*\]/, "")}
                </button>
              ))}
            </div>
            <div className="segmented small">
              <button className={view === "vs_failure" ? "active" : ""} onClick={() => setView("vs_failure")}>
                By outcome
              </button>
              <button className={view === "distribution" ? "active" : ""} onClick={() => setView("distribution")}>
                Distribution
              </button>
            </div>
          </div>
        }
      >
        <div className="grid feature-grid">
          <div className="report-image">
            <img src={reportImage(`${PLOTS[feature]}_${view}.png`)} alt={`${current.column} ${view}`} />
          </div>
          <div className="feature-stats">
            <h3>{current.column}</h3>
            <dl>
              <dt>Range</dt>
              <dd>
                {current.min} – {current.max}
              </dd>
              <dt>Mean ± std</dt>
              <dd>
                {current.mean} ± {current.std}
              </dd>
              <dt>Mean (healthy)</dt>
              <dd className="safe-text">{current.mean_healthy}</dd>
              <dt>Mean (failure)</dt>
              <dd className="danger-text">{current.mean_failure}</dd>
            </dl>
            <p className="muted small">
              Difference between failed and healthy machines:{" "}
              <strong>
                {(current.mean_failure - current.mean_healthy > 0 ? "+" : "") + (current.mean_failure - current.mean_healthy).toFixed(2)}
              </strong>
            </p>
          </div>
        </div>
      </Card>

      <div className="grid two">
        <Card title="Correlation matrix">
          <div className="report-image">
            <img src={reportImage("correlation_matrix.png")} alt="Correlation matrix" />
          </div>
        </Card>
        <Card title="Target distribution">
          <div className="report-image">
            <img src={reportImage("target_distribution.png")} alt="Target distribution" />
          </div>
        </Card>
      </div>
    </>
  );
}
