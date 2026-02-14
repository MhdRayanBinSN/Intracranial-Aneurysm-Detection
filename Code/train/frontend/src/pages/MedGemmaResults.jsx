import { useLocation, Link } from 'react-router-dom'
import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { 
  DocumentTextIcon,
  ExclamationTriangleIcon,
  CheckCircleIcon,
  SparklesIcon,
  EyeIcon,
  ArrowLeftIcon,
  ClockIcon,
  BeakerIcon,
  MagnifyingGlassIcon,
  XMarkIcon,
  MapPinIcon,
  ChevronDownIcon,
  ChevronUpIcon,
  CubeTransparentIcon,
  SignalIcon,
  ShieldExclamationIcon,
  CpuChipIcon,
} from '@heroicons/react/24/outline'

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000'

// Resolve image URL — handles both base64 (data:) and API paths (/medgemma/...)
const resolveImageUrl = (url) => {
  if (!url) return null
  if (url.startsWith('data:')) return url // base64 passthrough
  return `${API_BASE}${url}` // prepend API base for relative paths
}

export default function MedGemmaResults() {
  const location = useLocation()
  const result = location.state?.result
  const [selectedSlice, setSelectedSlice] = useState(null)
  const [expandedLocations, setExpandedLocations] = useState({})

  // Toggle a location's expanded state
  const toggleLocation = (loc) => {
    setExpandedLocations(prev => ({
      ...prev,
      [loc]: !prev[loc]
    }))
  }

  // Expand all locations
  const expandAll = () => {
    if (!result?.findings_by_location) return
    const all = {}
    Object.keys(result.findings_by_location).forEach(loc => { all[loc] = true })
    setExpandedLocations(all)
  }

  // Collapse all
  const collapseAll = () => setExpandedLocations({})

  // No results - show fallback
  if (!result) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center"
        >
          <SparklesIcon className="w-16 h-16 text-surface-600 mx-auto mb-4" />
          <h2 className="text-2xl font-bold mb-2">No Results Available</h2>
          <p className="text-surface-400 mb-6">Please upload and analyze scans first.</p>
          <Link to="/medgemma" className="btn-primary inline-flex items-center gap-2">
            <ArrowLeftIcon className="w-4 h-4" />
            Back to Upload
          </Link>
        </motion.div>
      </div>
    )
  }

  const findingsByLocation = result.findings_by_location || {}
  const locationNames = Object.keys(findingsByLocation)
  const totalFindings = result.findings?.length || 0

  return (
    <div className="min-h-screen py-12 px-4">
      <div className="max-w-6xl mx-auto">
        {/* Header with Back Button */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-8"
        >
          <Link 
            to="/medgemma" 
            className="inline-flex items-center gap-2 text-surface-400 hover:text-white transition-colors mb-6 group"
          >
            <ArrowLeftIcon className="w-4 h-4 group-hover:-translate-x-1 transition-transform" />
            Back to Upload
          </Link>
          
          <div className="flex items-center gap-3">
            <div className="w-12 h-12 rounded-xl bg-primary-500/20 flex items-center justify-center">
              <SparklesIcon className="w-7 h-7 text-primary-400" />
            </div>
            <div>
              <h1 className="text-3xl font-bold">
                <span className="text-primary-400">Analysis</span> Results
              </h1>
              <p className="text-surface-400 text-sm mt-1">
                AI-Powered Intracranial Aneurysm Detection Report
              </p>
            </div>
          </div>
        </motion.div>

        <div className="space-y-6">
          {/* Summary Stats */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
            className="grid grid-cols-2 md:grid-cols-5 gap-4"
          >
            <div className="card p-5 text-center group hover:border-primary-500/30 transition-colors">
              <div className="w-10 h-10 rounded-lg bg-primary-500/10 flex items-center justify-center mx-auto mb-3">
                <BeakerIcon className="w-5 h-5 text-primary-400" />
              </div>
              <p className="text-2xl font-bold text-primary-400">{result.slices_analyzed}</p>
              <p className="text-xs text-surface-500 mt-1">Slices Analyzed</p>
            </div>
            
            <div className="card p-5 text-center group hover:border-primary-500/30 transition-colors">
              <div className={`w-10 h-10 rounded-lg flex items-center justify-center mx-auto mb-3 ${
                result.has_findings ? 'bg-risk-high/10' : 'bg-risk-low/10'
              }`}>
                <MapPinIcon className={`w-5 h-5 ${result.has_findings ? 'text-risk-high' : 'text-risk-low'}`} />
              </div>
              <p className={`text-2xl font-bold ${result.has_findings ? 'text-risk-high' : 'text-risk-low'}`}>
                {result.num_locations || 0}
              </p>
              <p className="text-xs text-surface-500 mt-1">Locations</p>
            </div>

            <div className="card p-5 text-center group hover:border-primary-500/30 transition-colors">
              <div className={`w-10 h-10 rounded-lg flex items-center justify-center mx-auto mb-3 ${
                result.has_findings ? 'bg-risk-high/10' : 'bg-risk-low/10'
              }`}>
                {result.has_findings ? (
                  <ExclamationTriangleIcon className="w-5 h-5 text-risk-high" />
                ) : (
                  <CheckCircleIcon className="w-5 h-5 text-risk-low" />
                )}
              </div>
              <p className={`text-2xl font-bold ${result.has_findings ? 'text-risk-high' : 'text-risk-low'}`}>
                {totalFindings}
              </p>
              <p className="text-xs text-surface-500 mt-1">Detected Slices</p>
            </div>
            
            <div className="card p-5 text-center group hover:border-primary-500/30 transition-colors">
              <div className="w-10 h-10 rounded-lg bg-blue-500/10 flex items-center justify-center mx-auto mb-3">
                <ClockIcon className="w-5 h-5 text-blue-400" />
              </div>
              <p className="text-2xl font-bold text-blue-400">{result.processing_time?.toFixed(1)}s</p>
              <p className="text-xs text-surface-500 mt-1">Processing Time</p>
            </div>

            <div className="card p-5 text-center group hover:border-primary-500/30 transition-colors">
              <div className={`w-10 h-10 rounded-lg flex items-center justify-center mx-auto mb-3 ${
                result.has_findings ? 'bg-risk-high/10' : 'bg-risk-low/10'
              }`}>
                <MagnifyingGlassIcon className={`w-5 h-5 ${result.has_findings ? 'text-risk-high' : 'text-risk-low'}`} />
              </div>
              <p className={`text-2xl font-bold ${result.has_findings ? 'text-risk-high' : 'text-risk-low'}`}>
                {result.has_findings ? 'Detected' : 'Clear'}
              </p>
              <p className="text-xs text-surface-500 mt-1">Status</p>
            </div>
          </motion.div>

          {/* Status Badge */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.15 }}
            className={`card p-5 border-l-4 ${
              result.has_findings 
                ? 'border-l-risk-high bg-risk-high/5'
                : 'border-l-risk-low bg-risk-low/5'
            }`}
          >
            <div className="flex items-center gap-3">
              {result.has_findings ? (
                <ExclamationTriangleIcon className="w-6 h-6 text-risk-high flex-shrink-0" />
              ) : (
                <CheckCircleIcon className="w-6 h-6 text-risk-low flex-shrink-0" />
              )}
              <div>
                <p className={`font-semibold text-lg ${result.has_findings ? 'text-risk-high' : 'text-risk-low'}`}>
                  {result.has_findings ? 'Potential Findings Detected' : 'No Obvious Abnormalities'}
                </p>
                <p className="text-surface-400 text-sm mt-1">
                  {result.has_findings 
                    ? `${totalFindings} slice(s) across ${locationNames.length} anatomical location(s). Review each location below.`
                    : 'The AI did not detect any obvious abnormalities in the uploaded scans.'}
                </p>
              </div>
            </div>
          </motion.div>

          {/* AI Report */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
            className="card p-6"
          >
            <div className="flex items-center gap-3 mb-4">
              <div className="w-8 h-8 rounded-lg bg-primary-500/10 flex items-center justify-center">
                <DocumentTextIcon className="w-5 h-5 text-primary-400" />
              </div>
              <h2 className="text-xl font-semibold">AI Analysis Report</h2>
            </div>
            
            <div className="bg-surface-800/50 rounded-xl p-6 border border-surface-700/50 space-y-4">
              {/* Report Title */}
              <div className="flex items-center gap-2 pb-3 border-b border-surface-700/50">
                <ShieldExclamationIcon className="w-5 h-5 text-primary-400" />
                <h3 className="font-semibold text-surface-100">CT Scan Analysis Report</h3>
              </div>
              
              {/* Stats Row */}
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                <div className="flex items-center gap-2.5 bg-surface-900/50 rounded-lg px-4 py-3">
                  <BeakerIcon className="w-4 h-4 text-primary-400 flex-shrink-0" />
                  <span className="text-sm text-surface-300">Analyzed: <span className="text-white font-medium">{result.slices_analyzed} slice(s)</span></span>
                </div>
                {result.has_findings && (
                  <div className="flex items-center gap-2.5 bg-surface-900/50 rounded-lg px-4 py-3">
                    <MagnifyingGlassIcon className="w-4 h-4 text-risk-high flex-shrink-0" />
                    <span className="text-sm text-surface-300">Findings: <span className="text-risk-high font-medium">{totalFindings} slice(s)</span></span>
                  </div>
                )}
                {result.has_findings && (
                  <div className="flex items-center gap-2.5 bg-surface-900/50 rounded-lg px-4 py-3">
                    <MapPinIcon className="w-4 h-4 text-amber-400 flex-shrink-0" />
                    <span className="text-sm text-surface-300">Locations: <span className="text-amber-400 font-medium">{result.num_locations || 0}</span></span>
                  </div>
                )}
              </div>

              {/* Location Breakdown */}
              {result.has_findings && locationNames.length > 0 && (
                <div className="space-y-1.5">
                  <p className="text-xs text-surface-500 font-medium uppercase tracking-wider">Locations with findings</p>
                  {locationNames.map(loc => (
                    <div key={loc} className="flex items-center gap-2 text-sm">
                      <div className="w-1.5 h-1.5 rounded-full bg-risk-high flex-shrink-0" />
                      <span className="text-surface-300">{loc}:</span>
                      <span className="text-white font-medium">{findingsByLocation[loc].length} slice(s)</span>
                    </div>
                  ))}
                </div>
              )}

              {/* Method & Warning */}
              <div className="flex items-center gap-2.5 pt-2 border-t border-surface-700/30">
                <CpuChipIcon className="w-4 h-4 text-surface-500 flex-shrink-0" />
                <span className="text-xs text-surface-500">Detection Method: Intensity-based region segmentation</span>
              </div>
              {result.has_findings && (
                <div className="flex items-start gap-2.5 bg-amber-500/5 border border-amber-500/20 rounded-lg px-4 py-3">
                  <ExclamationTriangleIcon className="w-4 h-4 text-amber-400 flex-shrink-0 mt-0.5" />
                  <span className="text-xs text-amber-300/80">This is automated detection. All findings require verification by a qualified radiologist.</span>
                </div>
              )}
            </div>
          </motion.div>

          {/* ═══════════════════════════════════════════════════ */}
          {/* FINDINGS GROUPED BY LOCATION */}
          {/* ═══════════════════════════════════════════════════ */}
          {locationNames.length > 0 && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.25 }}
              className="space-y-4"
            >
              {/* Section Header */}
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <div className="w-8 h-8 rounded-lg bg-risk-high/10 flex items-center justify-center">
                    <MapPinIcon className="w-5 h-5 text-risk-high" />
                  </div>
                  <h2 className="text-xl font-semibold">Findings by Location</h2>
                  <span className="bg-risk-high/20 text-risk-high text-xs font-medium px-3 py-1 rounded-full">
                    {locationNames.length} location(s)
                  </span>
                </div>
                <div className="flex gap-2">
                  <button 
                    onClick={expandAll}
                    className="text-xs text-primary-400 hover:text-primary-300 transition-colors px-2 py-1 rounded bg-surface-800/50 hover:bg-surface-700/50"
                  >
                    Expand All
                  </button>
                  <button 
                    onClick={collapseAll}
                    className="text-xs text-surface-400 hover:text-surface-300 transition-colors px-2 py-1 rounded bg-surface-800/50 hover:bg-surface-700/50"
                  >
                    Collapse All
                  </button>
                </div>
              </div>

              {/* Location Cards */}
              {locationNames.map((locationName, locIdx) => {
                const slices = findingsByLocation[locationName]
                const isExpanded = expandedLocations[locationName] !== false // default expanded
                
                return (
                  <motion.div
                    key={locationName}
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.3 + locIdx * 0.05 }}
                    className="card overflow-hidden"
                  >
                    {/* Location Header - Clickable */}
                    <button
                      onClick={() => toggleLocation(locationName)}
                      className="w-full flex items-center justify-between p-5 hover:bg-surface-800/30 transition-colors text-left"
                    >
                      <div className="flex items-center gap-3">
                        <div className="w-10 h-10 rounded-lg bg-risk-high/15 flex items-center justify-center">
                          <MapPinIcon className="w-5 h-5 text-risk-high" />
                        </div>
                        <div>
                          <h3 className="font-semibold text-white text-base">{locationName}</h3>
                          <p className="text-surface-400 text-sm mt-0.5">
                            {slices.length} slice{slices.length !== 1 ? 's' : ''} detected
                            {' — '}
                            Slices: {slices.map(s => `#${s.slice_number}`).join(', ')}
                          </p>
                        </div>
                      </div>
                      <div className="flex items-center gap-3">
                        <span className="bg-risk-high/20 text-risk-high text-xs font-bold px-2.5 py-1 rounded-full">
                          {slices.length}
                        </span>
                        {isExpanded ? (
                          <ChevronUpIcon className="w-5 h-5 text-surface-400" />
                        ) : (
                          <ChevronDownIcon className="w-5 h-5 text-surface-400" />
                        )}
                      </div>
                    </button>

                    {/* Expanded: Slice Gallery */}
                    <AnimatePresence>
                      {isExpanded && (
                        <motion.div
                          initial={{ height: 0, opacity: 0 }}
                          animate={{ height: 'auto', opacity: 1 }}
                          exit={{ height: 0, opacity: 0 }}
                          transition={{ duration: 0.2 }}
                          className="overflow-hidden"
                        >
                          <div className="border-t border-surface-800 px-5 pb-5">
                            {/* Slice Grid */}
                            <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-3 mt-4">
                              {slices.map((slice, sliceIdx) => (
                                <motion.div
                                  key={slice.slice_index}
                                  initial={{ opacity: 0, scale: 0.95 }}
                                  animate={{ opacity: 1, scale: 1 }}
                                  transition={{ delay: sliceIdx * 0.03 }}
                                  className="group cursor-pointer"
                                  onClick={() => setSelectedSlice(slice)}
                                >
                                  {/* Thumbnail */}
                                  <div className="relative aspect-square rounded-lg overflow-hidden border-2 border-surface-700 group-hover:border-primary-400 transition-all">
                                    {slice.image ? (
                                      <img 
                                        src={resolveImageUrl(slice.image)} 
                                        alt={`Slice ${slice.slice_number}`}
                                        className="w-full h-full object-cover"
                                        loading="lazy"
                                      />
                                    ) : (
                                      <div className="w-full h-full bg-surface-800 flex items-center justify-center">
                                        <span className="text-surface-500 text-xs">No image</span>
                                      </div>
                                    )}
                                    {/* Hover Overlay */}
                                    <div className="absolute inset-0 bg-black/60 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center">
                                      <EyeIcon className="w-6 h-6 text-white" />
                                    </div>
                                    {/* Slice Number Badge */}
                                    <div className="absolute top-1.5 left-1.5 bg-black/70 text-white text-[10px] font-bold px-1.5 py-0.5 rounded">
                                      #{slice.slice_number}
                                    </div>
                                  </div>
                                  {/* Slice Label */}
                                  <p className="text-xs text-surface-400 mt-1.5 text-center">
                                    Slice {slice.slice_number}
                                  </p>
                                </motion.div>
                              ))}
                            </div>
                          </div>
                        </motion.div>
                      )}
                    </AnimatePresence>
                  </motion.div>
                )
              })}
            </motion.div>
          )}

          {/* Image Viewer Modal */}
          <AnimatePresence>
            {selectedSlice && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="fixed inset-0 bg-black/85 backdrop-blur-sm z-50 flex items-center justify-center p-4"
                onClick={() => setSelectedSlice(null)}
              >
                <motion.div
                  initial={{ opacity: 0, scale: 0.9 }}
                  animate={{ opacity: 1, scale: 1 }}
                  exit={{ opacity: 0, scale: 0.9 }}
                  className="card p-6 max-w-4xl w-full max-h-[90vh] overflow-y-auto"
                  onClick={(e) => e.stopPropagation()}
                >
                  <div className="flex justify-between items-center mb-4">
                    <div>
                      <h3 className="text-lg font-semibold flex items-center gap-2">
                        <EyeIcon className="w-5 h-5 text-primary-400" />
                        Slice {selectedSlice.slice_number} — Enlarged View
                      </h3>
                      {selectedSlice.location && (
                        <p className="text-surface-400 text-sm mt-1 flex items-center gap-1.5">
                          <MapPinIcon className="w-4 h-4" />
                          {selectedSlice.location || findingsByLocation && Object.entries(findingsByLocation).find(([_, slices]) => slices.some(s => s.slice_index === selectedSlice.slice_index))?.[0]}
                        </p>
                      )}
                    </div>
                    <button
                      onClick={() => setSelectedSlice(null)}
                      className="text-surface-400 hover:text-white transition-colors p-1 rounded-lg hover:bg-surface-800"
                    >
                      <XMarkIcon className="w-5 h-5" />
                    </button>
                  </div>
                  <div className="flex justify-center mb-4">
                    <img 
                      src={resolveImageUrl(selectedSlice.image)} 
                      alt={`Slice ${selectedSlice.slice_number}`}
                      className="max-w-full max-h-[60vh] rounded-xl border border-surface-700"
                    />
                  </div>
                  {/* Slice details - structured with icons */}
                  {selectedSlice && (
                    <div className="space-y-3">
                      {/* Location */}
                      {selectedSlice.location && (
                        <div className="flex items-center gap-3 bg-surface-800/50 rounded-lg px-4 py-3 border border-surface-700/50">
                          <MapPinIcon className="w-5 h-5 text-risk-high flex-shrink-0" />
                          <div>
                            <p className="text-xs text-surface-500 font-medium uppercase tracking-wider">Location</p>
                            <p className="text-sm text-white font-medium">{selectedSlice.location}</p>
                          </div>
                        </div>
                      )}

                      {/* Bbox & Intensity Row */}
                      <div className="grid grid-cols-2 gap-3">
                        {selectedSlice.bbox && (
                          <div className="flex items-center gap-3 bg-surface-800/50 rounded-lg px-4 py-3 border border-surface-700/50">
                            <CubeTransparentIcon className="w-5 h-5 text-blue-400 flex-shrink-0" />
                            <div>
                              <p className="text-xs text-surface-500 font-medium uppercase tracking-wider">Bounding Box</p>
                              <p className="text-sm text-surface-200">({selectedSlice.bbox[0]}, {selectedSlice.bbox[1]}) to ({selectedSlice.bbox[2]}, {selectedSlice.bbox[3]})</p>
                            </div>
                          </div>
                        )}
                        {selectedSlice.intensity != null && (
                          <div className="flex items-center gap-3 bg-surface-800/50 rounded-lg px-4 py-3 border border-surface-700/50">
                            <SignalIcon className="w-5 h-5 text-amber-400 flex-shrink-0" />
                            <div>
                              <p className="text-xs text-surface-500 font-medium uppercase tracking-wider">Max Intensity</p>
                              <p className="text-sm text-surface-200">{Number(selectedSlice.intensity).toFixed(0)}</p>
                            </div>
                          </div>
                        )}
                      </div>

                      {/* AI Analysis */}
                      {selectedSlice.response && (
                        <div className="bg-surface-800/50 rounded-lg px-4 py-3 border border-surface-700/50">
                          <div className="flex items-center gap-2 mb-2">
                            <SparklesIcon className="w-4 h-4 text-primary-400" />
                            <p className="text-xs text-surface-500 font-medium uppercase tracking-wider">AI Analysis</p>
                          </div>
                          <p className="text-sm text-surface-300 whitespace-pre-wrap leading-relaxed">
                            {selectedSlice.response}
                          </p>
                        </div>
                      )}

                      {/* Warning */}
                      <div className="flex items-start gap-2.5 bg-amber-500/5 border border-amber-500/20 rounded-lg px-4 py-2.5">
                        <ExclamationTriangleIcon className="w-4 h-4 text-amber-400 flex-shrink-0 mt-0.5" />
                        <span className="text-xs text-amber-300/80">Automated detection — requires radiologist review.</span>
                      </div>
                    </div>
                  )}
                </motion.div>
              </motion.div>
            )}
          </AnimatePresence>

          {/* Actions Bar */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.35 }}
            className="flex items-center justify-between"
          >
            <Link 
              to="/medgemma"
              className="btn-secondary inline-flex items-center gap-2"
            >
              <ArrowLeftIcon className="w-4 h-4" />
              Analyze More Scans
            </Link>
          </motion.div>

          {/* Disclaimer */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.4 }}
            className="p-4 bg-surface-900/50 border border-surface-800 rounded-xl"
          >
            <p className="text-xs text-surface-500 text-center">
              <strong>Disclaimer:</strong> This AI analysis is for research and educational purposes only. 
              It is not intended to replace professional medical diagnosis. 
              Always consult with qualified healthcare professionals for medical decisions.
            </p>
          </motion.div>
        </div>
      </div>
    </div>
  )
}
