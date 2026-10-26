import { useEffect, useState } from "react";
import { NavLink, Route, Routes } from "react-router-dom";
import { api } from "./api.js";
import Overview from "./pages/Overview.jsx";
import Predict from "./pages/Predict.jsx";
import Models from "./pages/Models.jsx";
import Explainability from "./pages/Explainability.jsx";
import Dataset from "./pages/Dataset.jsx";

const NAV = [
  { to: "/", label: "Overview", icon: "◎" },
  { to: "/predict", label: "Predict & Explain", icon: "⚡" },
  { to: "/models", label: "Model Comparison", icon: "▤" },
  { to: "/explainability", label: "Explainability", icon: "◐" },
  { to: "/dataset", label: "Dataset", icon: "▦" },
];

function ApiStatus() {
  const [health, setHealth] = useState({ state: "checking" });

  useEffect(() => {
    let cancelled = false;
    const check = () =>
      api
        .health()
        .then((h) => !cancelled && setHealth({ state: "ok", ...h }))
        .catch(() => !cancelled && setHealth({ state: "down" }));
    check();
    const id = setInterval(check, 15000);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, []);

  const label =
    health.state === "ok"
      ? `API online · ${health.model} on ${health.device}`
      : health.state === "down"
        ? "API offline"
        : "Checking API…";

  return (
    <div className={`api-status ${health.state}`}>
      <span className="dot" />
      {label}
    </div>
  );
}

export default function App() {
  const [menuOpen, setMenuOpen] = useState(false);

  return (
    <div className="shell">
      <aside className={`sidebar ${menuOpen ? "open" : ""}`}>
        <div className="brand">
          <span className="brand-icon">⚙</span>
          <div>
            <strong>Predictive Maintenance</strong>
            <small>Explainable IoT failure prediction</small>
          </div>
        </div>
        <nav>
          {NAV.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              end={item.to === "/"}
              onClick={() => setMenuOpen(false)}
              className={({ isActive }) => `nav-link ${isActive ? "active" : ""}`}
            >
              <span className="nav-icon">{item.icon}</span>
              {item.label}
            </NavLink>
          ))}
        </nav>
        <ApiStatus />
      </aside>

      <main className="content">
        <button className="menu-toggle" onClick={() => setMenuOpen((o) => !o)} aria-label="Toggle menu">
          ☰
        </button>
        <Routes>
          <Route path="/" element={<Overview />} />
          <Route path="/predict" element={<Predict />} />
          <Route path="/models" element={<Models />} />
          <Route path="/explainability" element={<Explainability />} />
          <Route path="/dataset" element={<Dataset />} />
        </Routes>
      </main>
    </div>
  );
}
