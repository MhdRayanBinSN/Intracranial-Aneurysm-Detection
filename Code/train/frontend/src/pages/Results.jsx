import { useLocation, useParams, Link } from 'react-router-dom'
import { motion } from 'framer-motion'
import { 
  CheckCircleIcon, 
  ExclamationCircleIcon,
  ArrowLeftIcon,
  ArrowDownTrayIcon,
  ChartBarIcon,
} from '@heroicons/react/24/outline'
import { Chart as ChartJS, ArcElement, Tooltip, Legend, CategoryScale, LinearScale, BarElement } from 'chart.js'
import { Doughnut, Bar } from 'react-chartjs-2'

ChartJS.register(ArcElement, Tooltip, Legend, CategoryScale, LinearScale, BarElement)

// Location groups for visualization
const locationGroups = {
  'Anterior Circulation': [
    'Left Infraclinoid Internal Carotid Artery',
    'Right Infraclinoid Internal Carotid Artery',
    'Left Supraclinoid Internal Carotid Artery',
    'Right Supraclinoid Internal Carotid Artery',
    'Left Middle Cerebral Artery',
    'Right Middle Cerebral Artery',
    'Left Anterior Cerebral Artery',
    'Right Anterior Cerebral Artery',
  ],
  'Posterior Circulation': [
    'Left Posterior Communicating Artery',
    'Right Posterior Communicating Artery',
    'Basilar Tip',
    'Other Posterior Circulation',
  ],
}

export default function Results() {
  const { id } = useParams()
  const location = useLocation()
  const result = location.state?.result

  if (!result) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center">
          <h2 className="text-2xl font-bold mb-4">Results Not Found</h2>
          <p className="text-surface-400 mb-6">The analysis results could not be loaded.</p>
          <Link to="/analysis" className="btn-primary">
            New Analysis
          </Link>
        </div>
      </div>
    )
  }

  const { predictions, overall_risk, confidence, processing_time } = result

  const detectedLocations = predictions.filter(p => p.detected && p.location !== 'Aneurysm Present')
  const overallPrediction = predictions.find(p => p.location === 'Aneurysm Present')

  // Chart data
  const riskColor = {
    Low: '#10b981',
    Moderate: '#f59e0b',
    High: '#ef4444',
  }

  const doughnutData = {
    labels: ['Confidence', 'Uncertainty'],
    datasets: [{
      data: [confidence * 100, (1 - confidence) * 100],
      backgroundColor: [riskColor[overall_risk], '#1e293b'],
      borderWidth: 0,
    }],
  }

  const barData = {
    labels: predictions.slice(0, -1).map(p => {
      // Shorten labels
      return p.location
        .replace('Internal Carotid Artery', 'ICA')
        .replace('Cerebral Artery', 'CA')
        .replace('Communicating Artery', 'Comm')
        .replace('Left ', 'L ')
        .replace('Right ', 'R ')
    }),
    datasets: [{
      label: 'Probability',
      data: predictions.slice(0, -1).map(p => p.probability * 100),
      backgroundColor: predictions.slice(0, -1).map(p => 
        p.detected ? riskColor.High : '#0ea5e9'
      ),
      borderRadius: 6,
    }],
  }

  const barOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { display: false },
    },
    scales: {
      y: {
        beginAtZero: true,
        max: 100,
        grid: { color: '#334155' },
        ticks: { color: '#94a3b8' },
      },
      x: {
        grid: { display: false },
        ticks: { 
          color: '#94a3b8',
          font: { size: 10 },
          maxRotation: 45,
        },
      },
    },
  }

  return (
    <div className="min-h-screen py-12 px-4">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="flex items-center justify-between mb-8"
        >
          <div>
            <Link 
              to="/analysis" 
              className="inline-flex items-center gap-2 text-surface-400 hover:text-white mb-4"
            >
              <ArrowLeftIcon className="w-4 h-4" />
              Back to Analysis
            </Link>
            <h1 className="text-3xl font-bold">Analysis Results</h1>
            <p className="text-surface-400 mt-1">
              ID: {id} • Processed in {processing_time.toFixed(2)}s
            </p>
          </div>
          <button className="btn-secondary flex items-center gap-2">
            <ArrowDownTrayIcon className="w-5 h-5" />
            Download Report
          </button>
        </motion.div>

        {/* Risk Overview */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="grid md:grid-cols-3 gap-6 mb-8"
        >
          {/* Risk Level */}
          <div className={`card p-6 border-2 ${
            overall_risk === 'High' ? 'border-risk-high/50 bg-risk-high/5' :
            overall_risk === 'Moderate' ? 'border-risk-moderate/50 bg-risk-moderate/5' :
            'border-risk-low/50 bg-risk-low/5'
          }`}>
            <div className="flex items-center gap-4">
              {overall_risk === 'Low' ? (
                <CheckCircleIcon className="w-12 h-12 text-risk-low" />
              ) : (
                <ExclamationCircleIcon className={`w-12 h-12 ${
                  overall_risk === 'High' ? 'text-risk-high' : 'text-risk-moderate'
                }`} />
              )}
              <div>
                <p className="text-surface-400 text-sm">Risk Level</p>
                <p className={`text-2xl font-bold ${
                  overall_risk === 'High' ? 'text-risk-high' :
                  overall_risk === 'Moderate' ? 'text-risk-moderate' :
                  'text-risk-low'
                }`}>{overall_risk}</p>
              </div>
            </div>
          </div>

          {/* Confidence */}
          <div className="card p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-surface-400 text-sm">Confidence</p>
                <p className="text-2xl font-bold">{(confidence * 100).toFixed(1)}%</p>
              </div>
              <div className="w-20 h-20">
                <Doughnut 
                  data={doughnutData} 
                  options={{ 
                    cutout: '70%',
                    plugins: { legend: { display: false } }
                  }} 
                />
              </div>
            </div>
          </div>

          {/* Detections */}
          <div className="card p-6">
            <div className="flex items-center gap-4">
              <ChartBarIcon className="w-12 h-12 text-primary-400" />
              <div>
                <p className="text-surface-400 text-sm">Detections</p>
                <p className="text-2xl font-bold">
                  {detectedLocations.length}
                  <span className="text-surface-500 text-lg font-normal"> / 12</span>
                </p>
              </div>
            </div>
          </div>
        </motion.div>

        {/* Bar Chart */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="card p-6 mb-8"
        >
          <h2 className="text-xl font-semibold mb-6">Probability by Location</h2>
          <div className="h-80">
            <Bar data={barData} options={barOptions} />
          </div>
        </motion.div>

        {/* Detailed Results */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
          className="grid md:grid-cols-2 gap-6"
        >
          {Object.entries(locationGroups).map(([group, locations]) => (
            <div key={group} className="card p-6">
              <h3 className="text-lg font-semibold mb-4">{group}</h3>
              <div className="space-y-3">
                {locations.map((loc) => {
                  const pred = predictions.find(p => p.location === loc)
                  if (!pred) return null
                  
                  return (
                    <div key={loc} className="flex items-center justify-between">
                      <div className="flex items-center gap-3">
                        <div className={`w-2 h-2 rounded-full ${
                          pred.detected ? 'bg-risk-high' : 'bg-surface-600'
                        }`} />
                        <span className={`text-sm ${
                          pred.detected ? 'text-white' : 'text-surface-400'
                        }`}>
                          {loc.replace('Internal Carotid Artery', 'ICA')
                             .replace('Cerebral Artery', 'CA')
                             .replace('Communicating Artery', 'Comm')}
                        </span>
                      </div>
                      <div className="flex items-center gap-2">
                        <div className="w-24 h-2 bg-surface-800 rounded-full overflow-hidden">
                          <div 
                            className={`h-full ${pred.detected ? 'bg-risk-high' : 'bg-primary-500'}`}
                            style={{ width: `${pred.probability * 100}%` }}
                          />
                        </div>
                        <span className="text-xs text-surface-500 w-12 text-right">
                          {(pred.probability * 100).toFixed(1)}%
                        </span>
                      </div>
                    </div>
                  )
                })}
              </div>
            </div>
          ))}
        </motion.div>

        {/* Disclaimer */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.4 }}
          className="mt-8 p-4 bg-surface-900/50 border border-surface-800 rounded-xl"
        >
          <p className="text-xs text-surface-500 text-center">
            <strong>Disclaimer:</strong> This AI analysis is for research and educational purposes only. 
            It is not intended to replace professional medical diagnosis. 
            Always consult with qualified healthcare professionals for medical decisions.
          </p>
        </motion.div>
      </div>
    </div>
  )
}
