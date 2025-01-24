import React, { useState, useEffect } from 'react';
import CircuitVisualizer from './CircuitVisualizer';
import DataInputPanel from './DataInputPanel';

const MarketDashboard = () => {
    const [marketData, setMarketData] = useState(null);
    const [quantumState, setQuantumState] = useState(null);

    const fetchMarketRegime = async (data) => {
        try {
            const response = await fetch('/api/market-regime', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data)
            });
            const result = await response.json();
            setQuantumState(result);
        } catch (error) {
            console.error('Error fetching market regime:', error);
        }
    };

    return (
        <div className="dashboard-container">
            <DataInputPanel onDataSubmit={fetchMarketRegime} />
            <CircuitVisualizer quantumState={quantumState} />
        </div>
    );
};

export default MarketDashboard;
