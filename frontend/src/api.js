// Thin wrapper around the FastAPI backend (proxied through Vite at /api).
const BASE = import.meta.env.VITE_API_URL || "/api";

async function request(path, options = {}) {
  const response = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const body = await response.json();
      detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch {
      /* keep statusText */
    }
    throw new Error(`${response.status}: ${detail}`);
  }
  return response.json();
}

export const api = {
  health: () => request("/health"),
  predict: (input) => request("/predict", { method: "POST", body: JSON.stringify(input) }),
  explain: (input) => request("/explain", { method: "POST", body: JSON.stringify(input) }),
  sweep: (input, feature, steps = 40) =>
    request("/sweep", { method: "POST", body: JSON.stringify({ input, feature, steps }) }),
  metrics: () => request("/metrics"),
  shapGlobal: () => request("/shap/global"),
  datasetSummary: () => request("/dataset/summary"),
  datasetSample: (failure) =>
    request(`/dataset/sample${failure === undefined ? "" : `?failure=${failure}`}`),
};

export const reportImage = (name) => `${BASE}/static/reports/${name}`;

// Input fields used by the model, with sensible slider ranges taken from the
// AI4I 2020 dataset.
export const FEATURES = [
  { key: "air_temperature", label: "Air Temperature", unit: "K", min: 295, max: 305, step: 0.1 },
  { key: "process_temperature", label: "Process Temperature", unit: "K", min: 305, max: 314, step: 0.1 },
  { key: "rotational_speed", label: "Rotational Speed", unit: "rpm", min: 1160, max: 2890, step: 1 },
  { key: "torque", label: "Torque", unit: "Nm", min: 3, max: 77, step: 0.1 },
  { key: "tool_wear", label: "Tool Wear", unit: "min", min: 0, max: 253, step: 1 },
];

export const DEFAULT_INPUT = {
  type: "M",
  air_temperature: 298.5,
  process_temperature: 309.1,
  rotational_speed: 1408,
  torque: 42.8,
  tool_wear: 10,
};

export const PRESETS = [
  { name: "Healthy baseline", input: DEFAULT_INPUT },
  {
    name: "Worn tool",
    input: { type: "L", air_temperature: 300.2, process_temperature: 310.4, rotational_speed: 1450, torque: 48.1, tool_wear: 225 },
  },
  {
    name: "Overheating",
    input: { type: "L", air_temperature: 303.4, process_temperature: 311.6, rotational_speed: 1290, torque: 58.9, tool_wear: 110 },
  },
  {
    name: "Power overload",
    input: { type: "M", air_temperature: 299.0, process_temperature: 309.5, rotational_speed: 1210, torque: 72.0, tool_wear: 80 },
  },
  {
    name: "High-speed, low torque",
    input: { type: "H", air_temperature: 298.0, process_temperature: 308.5, rotational_speed: 2700, torque: 6.5, tool_wear: 40 },
  },
];

export const pct = (v, digits = 1) => `${(v * 100).toFixed(digits)}%`;
