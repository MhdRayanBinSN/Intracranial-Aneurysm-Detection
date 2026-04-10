import { useLocation, Link } from 'react-router-dom'
import { motion } from 'framer-motion'
import {
  ArrowLeftIcon,
  ScaleIcon,
  CheckCircleIcon,
  ExclamationTriangleIcon,
  ClockIcon,
  BeakerIcon,
  MapPinIcon,
  ExclamationCircleIcon,
  CpuChipIcon,
  SparklesIcon,
} from '@heroicons/react/24/outline'

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000'
const resolveImageUrl = (url) => {
  if (!url) return null
  if (url.startsWith('data:')) return url
  return `${API_BASE}${url}`
}

function MedGemmaPanel({ data }) {
  if (!data) return <ErrorCard message="No data received from MedGemma" />
  if (data.error && data.available === false) return <ErrorCard message={data.error} />

  const findingsByLocation = data.findings_by_location || {}
  const locationNames = Object.keys(findingsByLocation)
  const totalFindings = data.findings?.length || 0

  return (
    <div className="space-y-4">
      {/* Status */}
      <div className={`p-4 rounded-xl border-l-4 ${data.has_findings ? 'border-l-risk-high bg-risk-high/5' : 'border-l-risk-low bg-risk-low/5'}`}>
        <div className="flex items-center gap-2">
          {data.has_findings
            ? <ExclamationTriangleIcon className="w-5 h-5 text-risk-high" />
            : <CheckCircleIcon className="w-5 h-5 text-risk-low" />}
          <p className={`font-semibold ${data.has_findings ? 'text-risk-high' : 'text-risk-low'}`}>
            {data.has_findings ? 'Potential Findings Detected' : 'No Obvious Abnormalities'}
          </p>
        </div>
        <p className="text-surface-400 text-sm mt-1">
          {data.has_findings
            ? `${totalFindings} slice(s) with findings across ${locationNames.length} location(s)`
            : 'No high-intensity regions matching aneurysm criteria found.'}
        </p>
      </div>

      {/* Stats row */}
      <div className="grid grid-cols-3 gap-3">
        <StatCard label="Slices Analyzed" value={data.slices_analyzed ?? '—'} color="text-primary-400" />
        <StatCard label="Locations" value={data.num_locations ?? locationNames.length} color={data.has_findings ? 'text-risk-high' : 'text-risk-low'} />
        <StatCard label="Status" value={data.has_findings ? 'Detected' : 'Clear'} color={data.has_findings ? 'text-risk-high' : 'text-risk-low'} />
      </div>

      {/* Locations */}
      {locationNames.length > 0 && (
        <div className="card p-4 space-y-2">
          <p className="text-xs text-surface-500 uppercase tracking-wider font-medium flex items-center gap-1.5">
            <MapPinIcon className="w-3.5 h-3.5" /> Locations with findings
          </p>
          {locationNames.map(loc => (
            <div key={loc} className="flex items-center justify-between text-sm">
              <div className="flex items-center gap-2">
                <div className="w-1.5 h-1.5 rounded-full bg-risk-high" />
                <span className="text-surface-300">{loc}</span>
              </div>
              <span className="text-white font-medium">{findingsByLocation[loc].length} slice(s)</span>
            </div>
          ))}
        </div>
      )}

      {/* First finding image if available */}
      {data.findings?.[0]?.image && (
        <div className="card p-3">
          <p className="text-xs text-surface-500 uppercase tracking-wider mb-2">Sample Finding</p>
          <img
            src={resolveImageUrl(data.findings[0].image)}
            alt="Top finding"
            className="w-full rounded-lg border border-surface-700"
          />
        </div>
      )}
    </div>
  )
}

function LegacyPanel({ data }) {
  if (!data) return <ErrorCard message="No data received from Legacy model" />

  if (data.available === false) {
    return (
      <div className="card p-6 border-dashed border-2 border-surface-700 flex flex-col items-center text-center gap-4">
        <div className="w-14 h-14 rounded-xl bg-surface-800 flex items-center justify-center">
          <CpuChipIcon className="w-8 h-8 text-surface-500" />
        </div>
        <div>
          <h3 className="font-semibold text-surface-300 mb-1">Legacy Model Unavailable</h3>
          <p className="text-surface-500 text-sm leading-relaxed">{data.info || data.error}</p>
        </div>
        <div className="bg-surface-800/50 rounded-lg px-4 py-3 text-xs text-surface-500 w-full text-left space-y-1">
          <p className="font-medium text-surface-400">To enable Legacy model:</p>
          <p>1. Run <code className="text-primary-400">python ml/train.py</code></p>
          <p>2. Weights saved to <code className="text-primary-400">ml/checkpoints/best.pt</code></p>
          <p>3. Restart the backend</p>
        </div>
      </div>
    )
  }

  // If legacy model IS available (future)
  const risk = data.overall_risk
  const riskColor = risk === 'High' ? 'text-risk-high' : risk === 'Moderate' ? 'text-amber-400' : 'text-risk-low'

  return (
    <div className="space-y-4">
      <div className={`p-4 rounded-xl border-l-4 ${risk === 'High' ? 'border-l-risk-high bg-risk-high/5' : risk === 'Moderate' ? 'border-l-amber-400 bg-amber-400/5' : 'border-l-risk-low bg-risk-low/5'}`}>
        <p className={`font-semibold text-lg ${riskColor}`}>{risk} Risk</p>
        <p className="text-surface-400 text-sm mt-1">Confidence: {(data.confidence * 100).toFixed(1)}%</p>
      </div>
      <div className="space-y-2">
        {(data.predictions || []).filter(p => p.detected).map((p, i) => (
          <div key={i} className="flex items-center justify-between bg-surface-800/50 rounded-lg px-3 py-2">
            <span className="text-sm text-surface-300">{p.location}</span>
            <span className="text-xs font-bold text-risk-high">{(p.probability * 100).toFixed(0)}%</span>
          </div>
        ))}
      </div>
    </div>
  )
}

function StatCard({ label, value, color }) {
  return (
    <div className="card p-4 text-center">
      <p className={`text-2xl font-bold ${color}`}>{value}</p>
      <p className="text-xs text-surface-500 mt-1">{label}</p>
    </div>
  )
}

function ErrorCard({ message }) {
  return (
    <div className="card p-5 bg-risk-high/10 border border-risk-high/30 flex items-start gap-3">
      <ExclamationCircleIcon className="w-5 h-5 text-risk-high flex-shrink-0 mt-0.5" />
      <p className="text-risk-high text-sm">{message}</p>
    </div>
  )
}

export default function CompareResults() {
  const location = useLocation()
  const result = location.state?.result

  if (!result) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="text-center">
          <ScaleIcon className="w-16 h-16 text-surface-600 mx-auto mb-4" />
          <h2 className="text-2xl font-bold mb-2">No Results</h2>
          <p className="text-surface-400 mb-6">Please run a comparison first.</p>
          <Link to="/compare" className="btn-primary inline-flex items-center gap-2">
            <ArrowLeftIcon className="w-4 h-4" /> Back to Compare
          </Link>
        </motion.div>
      </div>
    )
  }

  const { medgemma, legacy, comparison_time } = result

  return (
    <div className="min-h-screen py-12 px-4">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="mb-8">
          <Link to="/compare" className="inline-flex items-center gap-2 text-surface-400 hover:text-white transition-colors mb-6 group">
            <ArrowLeftIcon className="w-4 h-4 group-hover:-translate-x-1 transition-transform" />
            Run Another Comparison
          </Link>
          <div className="flex items-center gap-3">
            <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-primary-500 to-accent-500 flex items-center justify-center">
              <ScaleIcon className="w-7 h-7 text-white" />
            </div>
            <div>
              <h1 className="text-3xl font-bold"><span className="text-primary-400">Comparison</span> Results</h1>
              <p className="text-surface-400 text-sm mt-1">Side-by-side Model Analysis</p>
            </div>
          </div>
        </motion.div>

        {/* Summary Bar */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="card p-4 mb-8 flex flex-wrap items-center gap-4"
        >
          <div className="flex items-center gap-2 text-sm">
            <ClockIcon className="w-4 h-4 text-blue-400" />
            <span className="text-surface-400">Total time:</span>
            <span className="text-white font-medium">{comparison_time}s</span>
          </div>
          <div className="h-4 w-px bg-surface-700" />
          <div className="flex items-center gap-2 text-sm">
            <SparklesIcon className="w-4 h-4 text-primary-400" />
            <span className="text-surface-400">MedGemma:</span>
            <span className={`font-medium ${medgemma?.has_findings ? 'text-risk-high' : medgemma?.error ? 'text-surface-400' : 'text-risk-low'}`}>
              {medgemma?.error ? 'Error' : medgemma?.has_findings ? 'Findings Detected' : 'Clear'}
            </span>
          </div>
          <div className="h-4 w-px bg-surface-700" />
          <div className="flex items-center gap-2 text-sm">
            <CpuChipIcon className="w-4 h-4 text-surface-500" />
            <span className="text-surface-400">Legacy ML:</span>
            <span className="text-surface-500 font-medium">Unavailable</span>
          </div>
        </motion.div>

        {/* Two-column side by side */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* LEFT: MedGemma */}
          <motion.div initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.2 }}>
            <div className="flex items-center gap-3 mb-4">
              <div className="w-9 h-9 rounded-lg bg-primary-500/20 flex items-center justify-center">
                <SparklesIcon className="w-5 h-5 text-primary-400" />
              </div>
              <div>
                <h2 className="font-bold text-lg">MedGemma</h2>
                <p className="text-xs text-surface-500">LLM + Intensity Region Detection</p>
              </div>
            </div>
            <MedGemmaPanel data={medgemma} />
          </motion.div>

          {/* RIGHT: Legacy */}
          <motion.div initial={{ opacity: 0, x: 20 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.25 }}>
            <div className="flex items-center gap-3 mb-4">
              <div className="w-9 h-9 rounded-lg bg-surface-700 flex items-center justify-center">
                <CpuChipIcon className="w-5 h-5 text-surface-400" />
              </div>
              <div>
                <h2 className="font-bold text-lg">Legacy ML Model</h2>
                <p className="text-xs text-surface-500">ResNet3D-18 · 13-location Classifier</p>
              </div>
            </div>
            <LegacyPanel data={legacy} />
          </motion.div>
        </div>

        {/* Disclaimer */}
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.4 }} className="mt-8 p-4 bg-surface-900/50 border border-surface-800 rounded-xl">
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
