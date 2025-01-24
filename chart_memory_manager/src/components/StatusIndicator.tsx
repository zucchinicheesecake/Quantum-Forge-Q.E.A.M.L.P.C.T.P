import { Status } from '../types/chart'

interface Props {
  status: Status
}

const StatusIndicator = ({ status }: Props) => {
  const getStatusColor = () => {
    switch (status) {
      case 'success': return 'bg-green-500'
      case 'error': return 'bg-red-500'
      case 'cleaning': return 'bg-yellow-500'
      default: return 'bg-gray-500'
    }
  }

  return (
    <div className="flex items-center gap-2">
      <div className={`w-3 h-3 rounded-full ${getStatusColor()}`} />
      <span className="capitalize">{status}</span>
    </div>
  )
}

export default StatusIndicator
