import React, { useEffect, useRef } from 'react';
import { createChart } from 'lightweight-charts';
import styled from 'styled-components';

const ChartContainer = styled.div`
  width: 100%;
  height: 600px;
  background: ${props => props.theme.chartBackground};
`;

const TradingChart = ({ data, signals, indicators }) => {
  const chartContainerRef = useRef();
  const chart = useRef();

  useEffect(() => {
    chart.current = createChart(chartContainerRef.current, {
      width: chartContainerRef.current.clientWidth,
      height: 600,
      layout: {
        backgroundColor: '#253248',
        textColor: 'rgba(255, 255, 255, 0.9)',
      },
      grid: {
        vertLines: { color: '#334158' },
        horzLines: { color: '#334158' },
      },
      crosshair: {
        mode: 'normal',
      },
      priceScale: {
        borderColor: '#485c7b',
      },
      timeScale: {
        borderColor: '#485c7b',
      },
    });

    const candleSeries = chart.current.addCandlestickSeries({
      upColor: '#4CAF50',
      downColor: '#FF5252',
      borderVisible: false,
      wickUpColor: '#4CAF50',
      wickDownColor: '#FF5252',
    });

    candleSeries.setData(data);

    // Add volume histogram
    const volumeSeries = chart.current.addHistogramSeries({
      color: '#26a69a',
      priceFormat: {
        type: 'volume',
      },
      priceScaleId: '',
    });

    volumeSeries.setData(data.map(item => ({
      time: item.time,
      value: item.volume,
      color: item.close > item.open ? '#4CAF50' : '#FF5252',
    })));

    // Add indicators
    if (indicators) {
      Object.entries(indicators).forEach(([name, data]) => {
        const lineSeries = chart.current.addLineSeries({
          color: data.color,
          lineWidth: 2,
        });
        lineSeries.setData(data.values);
      });
    }

    // Add trading signals
    if (signals) {
      const markers = signals.map(signal => ({
        time: signal.time,
        position: signal.type === 'buy' ? 'belowBar' : 'aboveBar',
        color: signal.type === 'buy' ? '#4CAF50' : '#FF5252',
        shape: signal.type === 'buy' ? 'arrowUp' : 'arrowDown',
        text: signal.type.toUpperCase(),
      }));
      candleSeries.setMarkers(markers);
    }

    return () => {
      chart.current.remove();
    };
  }, [data, signals, indicators]);

  return <ChartContainer ref={chartContainerRef} />;
};

export default TradingChart;
