import { useState, useEffect } from 'react'
import ChartMemoryStatus from './ChartMemoryStatus'
import CleanupControls from './CleanupControls'
import { ChartMemoryState, TradingChartConfig } from '../types/chart'
import { cleanupChartSeries, cleanupChartAxis, getMemoryUsageMetrics } from '../utils/chartMemoryUtils'

const ChartManager = () => {
  const [memoryState, setMemoryState] = useState<ChartMemoryState>({
    seriesCount: 5,
    axisCount: 2,
    memoryUsage: 256,
    status: 'idle',
    latency: 0,
    throughput: 0,
    quantumState: 'idle'
  })

  const [config] = useState<TradingChartConfig>({
    maxSeries: 100,
    maxAxis: 10,
    memoryLimit: 1024,
    latencyThreshold: 5,
    quantumEnabled: false
  })

  useEffect(() => {
    const interval = setInterval(async () => {
      try {
        const metrics = await getMemoryUsageMetrics()
        setMemoryState(prev => ({
          ...prev,
          ...metrics,
          quantumState: config.quantumEnabled ? 'processing' : 'idle'
        }))
      } catch (error) {
        console.error('Failed to update metrics:', error)
      }
    }, 1000)

    return () => clearInterval(interval)
  }, [config.quantumEnabled])

  const handleSeriesCleanup = async () => {
    setMemoryState(prev => ({ ...prev, status: 'cleaning' }))
    try {
      await cleanupChartSeries(config)
      setMemoryState(prev => ({
        ...prev,
        seriesCount: Math.max(0, prev.seriesCount - 1),
        memoryUsage: Math.max(0, prev.memoryUsage - 48),
        status: 'success'
      }))
    } catch (error) {
      setMemoryState(prev => ({ ...prev, status: 'error' }))
    }
  }

  const handleAxisCleanup = async () => {
    setMemoryState(prev => ({ ...prev, status: 'cleaning' }))
    try {
      await cleanupChartAxis(config)
      setMemoryState(prev => ({
        ...prev,
        axisCount: Math.max(0, prev.axisCount - 1),
        memoryUsage: Math.max(0, prev.memoryUsage - 32),
        status: 'success'
      }))
    } catch (error) {
      setMemoryState(prev => ({ ...prev, status: 'error' }))
    }
  }

  return (
    <div className="bg-gray-800 rounded-lg p-6 shadow-xl">
      <ChartMemoryStatus memoryState={memoryState} config={config} />
      <CleanupControls 
        onSeriesCleanup={handleSeriesCleanup}
        onAxisCleanup={handleAxisCleanup}
        disabled={memoryState.status === 'cleaning'}
        quantumEnabled={config.quantumEnabled}
      />
    </div>
  )
}

export default ChartManager
