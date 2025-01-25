import { useState, useEffect } from 'react';

const useMetrics = () => {
  const [metrics, setMetrics] = useState({ fidelity: 0.95, latency: 100 });

  useEffect(() => {
    // Simulate fetching metrics
    const interval = setInterval(() => {
      setMetrics(prevMetrics => ({
        fidelity: Math.random(),
        latency: Math.floor(Math.random() * 200)
      }));
    }, 5000);

    return () => clearInterval(interval);
  }, []);

  return metrics;
};

export default useMetrics;
