import React from 'react';
import { useQuantumCandles } from './hooks';

const QuantumCandlestick = ({ symbol }) => {
    const { data } = useQuantumCandles(symbol);
    return (
        <PlotlyChart 
            data={[{
                qbits: data.entanglement,
                type: 'qcandlestick',
                increasing: { line: { color: '#00FF88' } },
                quantumLayout: true
            }]}
        />
    );
};

export default QuantumCandlestick;
