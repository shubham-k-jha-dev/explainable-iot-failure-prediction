import { useEffect, useState } from "react";

// Runs an async loader once and tracks loading / error state.
export function useAsync(loader, deps = []) {
  const [state, setState] = useState({ loading: true, data: null, error: null });

  useEffect(() => {
    let cancelled = false;
    setState((s) => ({ ...s, loading: true, error: null }));
    loader()
      .then((data) => !cancelled && setState({ loading: false, data, error: null }))
      .catch((error) => !cancelled && setState({ loading: false, data: null, error }));
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);

  return state;
}

export function PageHeader({ title, subtitle, children }) {
  return (
    <header className="page-header">
      <div>
        <h1>{title}</h1>
        {subtitle && <p>{subtitle}</p>}
      </div>
      {children}
    </header>
  );
}

export function Card({ title, subtitle, children, className = "", actions }) {
  return (
    <section className={`card ${className}`}>
      {(title || actions) && (
        <div className="card-head">
          <div>
            {title && <h2>{title}</h2>}
            {subtitle && <p className="muted">{subtitle}</p>}
          </div>
          {actions}
        </div>
      )}
      {children}
    </section>
  );
}

export function Stat({ label, value, hint, tone }) {
  return (
    <div className={`stat ${tone || ""}`}>
      <span className="stat-label">{label}</span>
      <span className="stat-value">{value}</span>
      {hint && <span className="stat-hint">{hint}</span>}
    </div>
  );
}

export function Loading({ label = "Loading…" }) {
  return (
    <div className="loading">
      <span className="spinner" /> {label}
    </div>
  );
}

export function ErrorBox({ error }) {
  return (
    <div className="error-box">
      <strong>Could not reach the backend.</strong>
      <p>{error?.message}</p>
      <p className="muted">
        Start it from the project root with <code>uvicorn api.main:app --reload --port 8000</code>.
      </p>
    </div>
  );
}

// Semi-circular gauge for a probability in [0, 1].
export function Gauge({ value, threshold = 0.5, size = 220 }) {
  const r = size / 2 - 16;
  const cx = size / 2;
  const cy = size / 2 + 4;
  const angle = (v) => Math.PI * (1 - v);
  const point = (v) => [cx + r * Math.cos(angle(v)), cy - r * Math.sin(angle(v))];
  const arc = (from, to) => {
    const [x1, y1] = point(from);
    const [x2, y2] = point(to);
    return `M ${x1} ${y1} A ${r} ${r} 0 0 1 ${x2} ${y2}`;
  };
  const clamped = Math.min(Math.max(value, 0), 0.9999);
  const danger = value >= threshold;
  const tick = (v, radius) => [cx + radius * Math.cos(angle(v)), cy - radius * Math.sin(angle(v))];
  const [tx1, ty1] = tick(threshold, r - 14);
  const [tx2, ty2] = tick(threshold, r + 10);

  return (
    <svg className="gauge" viewBox={`0 0 ${size} ${size / 2 + 30}`} role="img" aria-label={`Failure probability ${(value * 100).toFixed(1)}%`}>
      <path d={arc(0, 0.9999)} className="gauge-track" />
      {clamped > 0.001 && <path d={arc(0, clamped)} className={`gauge-fill ${danger ? "danger" : "safe"}`} />}
      <line x1={tx1} y1={ty1} x2={tx2} y2={ty2} className="gauge-threshold" />
      <text x={cx} y={cy - 14} className="gauge-value">
        {(value * 100).toFixed(1)}%
      </text>
      <text x={cx} y={cy + 12} className="gauge-caption">
        failure probability
      </text>
    </svg>
  );
}
