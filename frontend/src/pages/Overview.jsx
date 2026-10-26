import { Link } from "react-router-dom";
import { Cell, Pie, PieChart, ResponsiveContainer, Tooltip } from "recharts";
import { api, pct } from "../api.js";
import { Card, ErrorBox, Loading, PageHeader, Stat, useAsync } from "../components/common.jsx";

const PIPELINE = [
  { title: "Dataset", text: "10,000 AI4I 2020 machine records", icon: "🗄" },
  { title: "EDA", text: "Distributions, correlations, imbalance", icon: "🔍" },
  { title: "Preprocessing", text: "Scaling + one-hot, stratified split", icon: "🧹" },
  { title: "Training", text: "MLP, LSTM and 1D-CNN with class weights", icon: "🧠" },
  { title: "Evaluation", text: "Accuracy, recall, F1, ROC-AUC", icon: "📊" },
  { title: "SHAP", text: "Global + local explanations", icon: "💡" },
  { title: "API", text: "FastAPI predict & explain", icon: "🔌" },
];

export default function Overview() {
  const summary = useAsync(api.datasetSummary);
  const metrics = useAsync(api.metrics);

  if (summary.loading || metrics.loading) return <Loading />;
  if (summary.error || metrics.error) return <ErrorBox error={summary.error || metrics.error} />;

  const s = summary.data;
  const best = [...metrics.data.models].sort((a, b) => b.roc_auc - a.roc_auc)[0];
  const pie = [
    { name: "Healthy", value: s.healthy },
    { name: "Failure", value: s.failures },
  ];

  return (
    <>
      <PageHeader
        title="Explainable IoT Failure Prediction"
        subtitle="Deep learning models predict machine failure from sensor readings, and SHAP explains every decision."
      >
        <Link to="/predict" className="btn primary">
          Try a prediction →
        </Link>
      </PageHeader>

      <div className="stats">
        <Stat label="Machine records" value={s.rows.toLocaleString()} />
        <Stat label="Failure rate" value={`${s.failure_rate}%`} tone="danger" hint={`${s.failures} failures`} />
        <Stat label="Models compared" value={metrics.data.models.length} hint="MLP · LSTM · 1D-CNN" />
        <Stat label={`Best ROC-AUC (${best.name})`} value={best.roc_auc.toFixed(3)} tone="safe" />
        <Stat label={`${best.name} recall`} value={pct(best.recall)} hint="failures caught on test set" />
      </div>

      <Card title="Pipeline" subtitle="From raw sensor data to explained predictions">
        <div className="pipeline">
          {PIPELINE.map((step, i) => (
            <div className="pipeline-step" key={step.title}>
              <span className="pipeline-icon">{step.icon}</span>
              <strong>
                {i + 1}. {step.title}
              </strong>
              <span>{step.text}</span>
            </div>
          ))}
        </div>
      </Card>

      <div className="grid two">
        <Card title="Class imbalance" subtitle="Failures are rare — accuracy alone would be misleading">
          <div className="pie-wrap">
            <ResponsiveContainer width="100%" height={220}>
              <PieChart>
                <Pie data={pie} dataKey="value" innerRadius={60} outerRadius={90} paddingAngle={2}>
                  <Cell fill="var(--safe)" />
                  <Cell fill="var(--danger)" />
                </Pie>
                <Tooltip contentStyle={{ background: "var(--surface)", border: "1px solid var(--border)" }} />
              </PieChart>
            </ResponsiveContainer>
            <div className="legend-list">
              <span>
                <i style={{ background: "var(--safe)" }} /> Healthy · {s.healthy.toLocaleString()}
              </span>
              <span>
                <i style={{ background: "var(--danger)" }} /> Failure · {s.failures}
              </span>
            </div>
          </div>
        </Card>

        <Card title="Explore">
          <div className="explore-links">
            <Link to="/predict">
              <strong>⚡ Predict & Explain</strong>
              <span>Tune sensor sliders and watch the risk and its SHAP drivers update live.</span>
            </Link>
            <Link to="/models">
              <strong>▤ Model Comparison</strong>
              <span>Compare metrics, ROC curves and play with the decision threshold.</span>
            </Link>
            <Link to="/explainability">
              <strong>◐ Explainability</strong>
              <span>See which sensors matter most to the model overall.</span>
            </Link>
            <Link to="/dataset">
              <strong>▦ Dataset</strong>
              <span>Distributions, failure modes and machine types.</span>
            </Link>
          </div>
        </Card>
      </div>
    </>
  );
}
