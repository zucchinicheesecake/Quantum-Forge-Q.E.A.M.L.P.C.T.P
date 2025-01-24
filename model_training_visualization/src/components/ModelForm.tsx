import { useState } from 'react'
import { ModelParams } from '../types/types'

interface ModelFormProps {
  onSubmit: (params: ModelParams) => void
}

const ModelForm = ({ onSubmit }: ModelFormProps) => {
  const [params, setParams] = useState<ModelParams>({
    learningRate: 0.001,
    epochs: 10,
    batchSize: 32
  })

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    // Added validation to prevent invalid inputs
    if (params.learningRate <= 0 || params.epochs <= 0 || params.batchSize <= 0) {
      alert('Please enter valid positive values for all parameters')
      return
    }
    onSubmit(params)
  }

  return (
    <form onSubmit={handleSubmit} className="space-y-4 bg-white p-6 rounded-lg shadow-sm">
      <h2 className="text-xl font-semibold text-gray-800 mb-4">Model Parameters</h2>
      
      <div className="space-y-2">
        <label className="block text-sm font-medium text-gray-700">
          Learning Rate
          <input
            type="number"
            step="0.001"
            min="0.0001"
            required
            value={params.learningRate}
            onChange={e => setParams({...params, learningRate: parseFloat(e.target.value)})}
            className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
          />
        </label>
      </div>

      <div className="space-y-2">
        <label className="block text-sm font-medium text-gray-700">
          Epochs
          <input
            type="number"
            min="1"
            required
            value={params.epochs}
            onChange={e => setParams({...params, epochs: parseInt(e.target.value)})}
            className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
          />
        </label>
      </div>

      <div className="space-y-2">
        <label className="block text-sm font-medium text-gray-700">
          Batch Size
          <input
            type="number"
            min="1"
            required
            value={params.batchSize}
            onChange={e => setParams({...params, batchSize: parseInt(e.target.value)})}
            className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
          />
        </label>
      </div>

      <button
        type="submit"
        className="w-full bg-blue-500 text-white py-2 px-4 rounded-md hover:bg-blue-600 transition-colors duration-200"
      >
        Start Training
      </button>
    </form>
  )
}

export default ModelForm
