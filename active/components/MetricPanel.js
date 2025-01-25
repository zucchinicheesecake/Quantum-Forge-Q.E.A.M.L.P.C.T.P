import React from 'react';

const MetricPanel = ({ title, value, threshold, unit }) => {
  const isCritical = value < threshold;

  return (
    <div className={`metric-panel ${isCritical ? 'critical' : ''}`}>
      <h3>{title}</h3>
      <p>{value} {unit}</p>
      {isCritical && <span className="alert">Alert: Below threshold!</span>}
    </div>
  );
};

export default MetricPanel;
