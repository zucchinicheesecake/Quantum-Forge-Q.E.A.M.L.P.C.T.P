export type Status = 'idle' | 'cleaning' | 'success' | 'error'

export interface ChartMemoryState {
  seriesCount: number
  axisCount: number
  memoryUsage: number
  status: Status
  latency?: number  // Added for trading system integration
  throughput?: number  // Added for performance monitoring
  quantumState?: 'idle' | 'processing'  // Added for quantum integration
}

export interface TradingChartConfig {
  maxSeries: number
  maxAxis: number
  memoryLimit: number
  latencyThreshold: number
  quantumEnabled: boolean
}
