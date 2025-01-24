import { useState } from 'react'
import ModelForm from './ModelForm'
import LoadingBar from './LoadingBar'
import StatusUpdates from './StatusUpdates'
import { ModelParams } from '../types/types'

const Dashboard = () => {
  const [progress, setProgress] = useState(0)
  const [status, setStatus] = useState<string[]>([])

  const handleStartTraining = (params: ModelParams) => {
    setStatus([...status, 'Training started with new parameters...'])
    simulateTraining()
  }

  const simulateTraining = () => {
    let currentProgress = 0
    const interval = setInterval(() => {
      currentProgress += 1
      setProgress(currentProgress)
      if (currentProgress % 20 === 0) {
        setStatus(prev => [...prev, `Training progress: ${currentProgress}%`])
      }
      if (currentProgress >= 100) {
        clearInterval(interval)
        setStatus(prev => [...prev, 'Training completed!'])
      }
    }, 100)
  }

  return (
    <div className="space-y-8">
      <h1 className="text-3xl font-bold text-gray-800">Model Training Dashboard</h1>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
        <div className="space-y-6">
          <ModelForm onSubmit={handleStartTraining} />
          <LoadingBar progress={progress} />
        </div>
        <StatusUpdates updates={status} />
      </div>
    </div>
  )
}

export default Dashboard
