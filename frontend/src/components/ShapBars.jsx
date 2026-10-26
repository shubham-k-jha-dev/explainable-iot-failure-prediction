// Diverging horizontal bar chart for SHAP values: red pushes toward failure,
// green pushes away from it.
export default function ShapBars({ items, valueKey = "shap_value", showDirection = true }) {
  if (!items?.length) return null;
  const max = Math.max(...items.map((i) => Math.abs(i[valueKey]))) || 1;

  return (
    <div className="shap-bars">
      <div className="shap-axis">
        <span>← lowers risk</span>
        <span>raises risk →</span>
      </div>
      {items.map((item) => {
        const v = item[valueKey];
        const width = `${(Math.abs(v) / max) * 50}%`;
        const positive = v > 0;
        return (
          <div className="shap-row" key={item.feature} title={item.direction || ""}>
            <span className="shap-label">{item.feature}</span>
            <div className="shap-track">
              <div className="shap-center" />
              <div
                className={`shap-bar ${positive ? "up" : "down"}`}
                style={positive ? { left: "50%", width } : { right: "50%", width }}
              />
            </div>
            <span className={`shap-value ${positive ? "up" : "down"}`}>
              {positive ? "+" : ""}
              {v.toFixed(4)}
            </span>
            {showDirection && item.direction && <span className="sr-only">{item.direction}</span>}
          </div>
        );
      })}
    </div>
  );
}
