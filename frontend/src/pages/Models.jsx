import { useMemo, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { api, pct, reportImage } from "../api.js";
import { Card, ErrorBox, Loading, PageHeader, useAsync } from "../components/common.jsx";

const COLORS = { MLP: "#7b61ff", LSTM: "#fc801d", "1D-CNN": "#087cfa" };
const METRICS = [
  { key: "accuracy", label: "Accuracy" },
  { key: "precision", label: "Precision" },
  { key: "recall", label: "Recall" },
  { key: "f1", label: "F1 Score" },
  { key: "roc_auc", label: "ROC-AUC" },
];

// Recompute the confusion matrix and metrics for a given decision threshold.
function evaluate(probabilities, labels, threshold) {
  let tp = 0, fp = 0, tn = 0, fn = 0;
  probabilities.forEach((p, i) => {
    const predicted = p >= threshold ? 1 : 0;
    if (predicted === 1 && labels[i] === 1) tp++;
    else if (predicted === 1) fp++;
    else if (labels[i] === 1) fn++;
    else tn++;
  });
  const precision = tp + fp ? tp / (tp + fp) : 0;
  const recall = tp + fn ? tp / (tp + fn) : 0;
  return {
    tp, fp, tn, fn,
    accuracy: (tp + tn) / labels.length,
    precision,
    recall,
    f1: precision + recall ? (2 * precision * recall) / (precision + recall) : 0,
  };
}

function ConfusionMatrix({ m }) {
  const total = m.tp + m.fp + m.tn + m.fn;
  const cell = (value, cls, label) => (
    <div className={`cm-cell ${cls}`} style={{ "--intensity": Math.min(1, 0.15 + value / total * 3) }}>
      <strong>{value}</strong>
      <span>{label}</span>
    </div>
  );
  return (
    <div className="cm">
      <div />
      <div className="cm-head">Pred: OK</div>
      <div className="cm-head">Pred: Fail</div>
      <div className="cm-side">Actual: OK</div>
      {cell(m.tn, "good", "True neg.")}
      {cell(m.fp, "warn", "False alarm")}
      <div className="cm-side">Actual: Fail</div>
      {cell(m.fn, "bad", "Missed")}
      {cell(m.tp, "good", "Caught")}
    </div>
  );
}

export default function Models() {
  const { data, loading, error } = useAsync(api.metrics);
  const [threshold, setThreshold] = useState(0.5);
  const [imageModel, setImageModel] = useState("mlp");
  const [imageKind, setImageKind] = useState("loss_curve");

  const evaluated = useMemo(() => {
    if (!data) return [];
    return data.models.map((m) => ({ ...m, live: evaluate(m.probabilities, data.labels, threshold) }));
  }, [data, threshold]);

  const best = useMemo(() => {
    const out = {};
    METRICS.forEach(({ key }) => {
      out[key] = Math.max(...evaluated.map((m) => m[key]));
    });
    return out;
  }, [evaluated]);

  if (loading) return <Loading />;
  if (error) return <ErrorBox error={error} />;

  const barData = METRICS.map(({ key, label }) => ({
    metric: label,
    ...Object.fromEntries(data.models.map((m) => [m.name, +m[key].toFixed(4)])),
  }));

  const testFailures = data.labels.filter((l) => l === 1).length;

  return (
    <>
      <PageHeader
        title="Model Comparison"
        subtitle={`Three deep learning architectures evaluated on the same held-out test set (${data.labels.length} machines, ${testFailures} failures).`}
      />

      <Card title="Reported test metrics" subtitle="At the default 0.5 decision threshold · best value per column highlighted">
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Model</th>
                {METRICS.map((m) => (
                  <th key={m.key}>{m.label}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {data.models.map((m) => (
                <tr key={m.name}>
                  <td>
                    <span className="swatch" style={{ background: COLORS[m.name] }} /> <strong>{m.name}</strong>
                  </td>
                  {METRICS.map(({ key }) => (
                    <td key={key} className={m[key] === best[key] ? "best" : ""}>
                      {m[key].toFixed(4)}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="note">
          💡 Recall matters most here: a missed failure (false negative) is far costlier than a false alarm. Precision is low because
          failures are only 3.4% of the data and the models were trained with a class weight that favours catching failures.
        </p>
      </Card>

      <div className="grid two">
        <Card title="Metrics side by side">
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={barData} margin={{ top: 10, right: 10, bottom: 0, left: -10 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--grid)" vertical={false} />
              <XAxis dataKey="metric" stroke="var(--muted)" fontSize={12} />
              <YAxis domain={[0, 1]} stroke="var(--muted)" fontSize={12} />
              <Tooltip contentStyle={{ background: "var(--surface)", border: "1px solid var(--border)" }} />
              <Legend />
              {data.models.map((m) => (
                <Bar key={m.name} dataKey={m.name} fill={COLORS[m.name]} radius={[4, 4, 0, 0]} />
              ))}
            </BarChart>
          </ResponsiveContainer>
        </Card>

        <Card title="ROC curves" subtitle="True-positive rate vs false-positive rate across all thresholds">
          <ResponsiveContainer width="100%" height={300}>
            <LineChart margin={{ top: 10, right: 10, bottom: 10, left: -10 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--grid)" />
              <XAxis dataKey="fpr" type="number" domain={[0, 1]} stroke="var(--muted)" fontSize={12} />
              <YAxis dataKey="tpr" type="number" domain={[0, 1]} stroke="var(--muted)" fontSize={12} />
              <Tooltip
                formatter={(v, n) => [Number(v).toFixed(3), n]}
                labelFormatter={(v) => `FPR ${Number(v).toFixed(3)}`}
                contentStyle={{ background: "var(--surface)", border: "1px solid var(--border)" }}
              />
              <Legend />
              <Line data={[{ fpr: 0, tpr: 0 }, { fpr: 1, tpr: 1 }]} dataKey="tpr" name="Random" stroke="var(--muted)" strokeDasharray="4 4" dot={false} />
              {data.models.map((m) => (
                <Line
                  key={m.name}
                  data={m.roc}
                  dataKey="tpr"
                  name={`${m.name} (AUC ${m.roc_auc.toFixed(3)})`}
                  stroke={COLORS[m.name]}
                  strokeWidth={2}
                  dot={false}
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </Card>
      </div>

      <Card
        title="Decision threshold explorer"
        subtitle="Slide the threshold to trade false alarms against missed failures — recomputed live from each model's test-set probabilities"
      >
        <div className="threshold">
          <input type="range" min={0.01} max={0.99} step={0.01} value={threshold} onChange={(e) => setThreshold(Number(e.target.value))} />
          <span className="threshold-value">{threshold.toFixed(2)}</span>
          <button className="chip" onClick={() => setThreshold(0.5)}>Reset</button>
        </div>
        <div className="grid three">
          {evaluated.map((m) => (
            <div key={m.name} className="model-tile">
              <h3>
                <span className="swatch" style={{ background: COLORS[m.name] }} /> {m.name}
              </h3>
              <ConfusionMatrix m={m.live} />
              <div className="mini-metrics">
                <span>Recall <strong>{pct(m.live.recall)}</strong></span>
                <span>Precision <strong>{pct(m.live.precision)}</strong></span>
                <span>F1 <strong>{m.live.f1.toFixed(3)}</strong></span>
                <span>Accuracy <strong>{pct(m.live.accuracy)}</strong></span>
              </div>
            </div>
          ))}
        </div>
      </Card>

      <Card
        title="Training reports"
        subtitle="Plots generated by the training scripts"
        actions={
          <div className="row gap">
            <div className="segmented small">
              {data.models.map((m) => (
                <button key={m.key} className={imageModel === m.key ? "active" : ""} onClick={() => setImageModel(m.key)}>
                  {m.name}
                </button>
              ))}
            </div>
            <div className="segmented small">
              {[
                ["loss_curve", "Loss"],
                ["confusion_matrix", "Confusion"],
                ["roc_curve", "ROC"],
              ].map(([k, l]) => (
                <button key={k} className={imageKind === k ? "active" : ""} onClick={() => setImageKind(k)}>
                  {l}
                </button>
              ))}
            </div>
          </div>
        }
      >
        <div className="report-image">
          <img src={reportImage(`${imageModel}_${imageKind}.png`)} alt={`${imageModel} ${imageKind}`} />
        </div>
      </Card>
    </>
  );
}
