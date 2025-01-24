import { ChartMemoryState, TradingChartConfig } from '../types/chart'
import StatusIndicator from './StatusIndicator'

interface Props {
  memoryState: ChartMemoryState
  config: TradingChartConfig
}

const ChartMemoryStatus = ({ memoryState, config }: Props) => {
  return (
    <div className="space-y-4 mb-6">
      <div className="grid grid-cols-3 gap-4">
        <div className="bg-gray-700 p-4 rounded-lg">
          <h3 className="text-sm text-gray-400">Series Count</h3>
          <p className="text-2xl font-bold">{memoryState.seriesCount}/{config.maxSeries}</p>
        </div>
        <div className="bg-gray-700 p-4 rounded-lg">
          <h3 className="text-sm text-gray-400">Axis Count</h3>
          <p className="text-2xl font-bold">{memoryState.axisCount}/{config.maxAxis}</p>
        </div>
        <div className="bg-gray-700 p-4 rounded-lg">
          <h3 className="text-sm text-gray-400">Memory Usage</h3>
          <p className="text-2xl font-bold">{memoryState.memoryUsage}MB/{config.memoryLimit}MB</p>
        </div>
      </div>
      <div className="grid grid-cols-3 gap-4">
        <div className="bg-gray-700 p-4 rounded-lg">
          <h3 className="text-sm text-gray-400">Latency</h3>
          <p className="text-2xl font-bold">{memoryState.latency?.toFixed(2)}ms</p>
        </div>
        <div className="bg-gray-700 p-4 rounded-lg">
          <h3 className="text-sm text-gray-400">Throughput</h3>
          <p className="text-2xl font-bold">{memoryState.throughput?.toFixed(2)} ops/s</p>
        </div>
        <div className="bg-gray-700 p-4 rounded-lg">
          <h3 className="text-sm text-gray-400">Quantum State</h3>
          <p className="text-2xl font-bold">{memoryState.quantumState}</p>
        </div>
      </div>
      <StatusIndicator status={memoryState.status} />
    </div>
  )
}

export default ChartMemoryStatus
