interface StatusUpdatesProps {
  updates: string[]
}

const StatusUpdates = ({ updates }: StatusUpdatesProps) => {
  return (
    <div className="bg-white p-6 rounded-lg shadow-sm h-[400px] overflow-auto">
      <h2 className="text-xl font-semibold text-gray-800 mb-4">Status Updates</h2>
      <div className="space-y-2">
        {updates.map((update, index) => (
          <div 
            key={index}
            className="p-3 bg-gray-50 rounded-md text-sm text-gray-700"
          >
            {update}
          </div>
        ))}
      </div>
    </div>
  )
}

export default StatusUpdates
