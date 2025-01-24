interface Props {
  onSeriesCleanup: () => void
  onAxisCleanup: () => void
  disabled: boolean
  quantumEnabled: boolean
}

const CleanupControls = ({ onSeriesCleanup, onAxisCleanup, disabled, quantumEnabled }: Props) => {
  return (
    <div className="flex gap-4">
      <button
        onClick={onSeriesCleanup}
        disabled={disabled}
        className={`flex-1 ${
          quantumEnabled ? 'bg-teal-600 hover:bg-teal-700' : 'bg-blue-600 hover:bg-blue-700'
        } disabled:bg-blue-800 disabled:cursor-not-allowed px-4 py-2 rounded-lg transition-colors`}
      >
        {quantumEnabled ? 'Quantum Clean Series' : 'Clean Series'}
      </button>
      <button
        onClick={onAxisCleanup}
        disabled={disabled}
        className={`flex-1 ${
          quantumEnabled ? 'bg-pink-600 hover:bg-pink-700' : 'bg-purple-600 hover:bg-purple-700'
        } disabled:bg-purple-800 disabled:cursor-not-allowed px-4 py-2 rounded-lg transition-colors`}
      >
        {quantumEnabled ? 'Quantum Clean Axis' : 'Clean Axis'}
      </button>
    </div>
  )
}

export default CleanupControls
