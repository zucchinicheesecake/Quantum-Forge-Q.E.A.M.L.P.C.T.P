import React from 'react';
import useMetrics from '../hooks/useMetrics'; // Changed to default import
import MetricPanel from './MetricPanel'; // Confirmed path

const QuantumMetrics = () => {
  const metrics = useMetrics();
  return (
    <div className="quantum-grid">
      <MetricPanel 
        title="Entanglement Fidelity" 
        value={metrics.fidelity} 
        threshold={0.9}
      />
      <MetricPanel
        title="Order Latency"
        value={metrics.latency}
        threshold={150}
        unit="ms"
      />
    </div>
  );
};

export default QuantumMetrics;
