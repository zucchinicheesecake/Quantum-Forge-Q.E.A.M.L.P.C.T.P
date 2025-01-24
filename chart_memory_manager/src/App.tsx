import ChartManager from './components/ChartManager'

function App() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 to-gray-800 text-white p-8">
      <div className="max-w-6xl mx-auto">
        <h1 className="text-3xl font-bold mb-8 text-center">Chart Memory Manager</h1>
        <ChartManager />
      </div>
    </div>
  )
}

export default App
