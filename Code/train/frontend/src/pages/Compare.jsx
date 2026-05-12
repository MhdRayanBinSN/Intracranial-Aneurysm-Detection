import { useState, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import { motion } from 'framer-motion'
import {
  ScaleIcon,
  CloudArrowUpIcon,
  XMarkIcon,
  BeakerIcon,
} from '@heroicons/react/24/outline'
import toast from 'react-hot-toast'
import { compareModels } from '../api/client'

export default function Compare() {
  const navigate = useNavigate()
  const [files, setFiles] = useState([])
  const [isDragging, setIsDragging] = useState(false)
  const [isRunning, setIsRunning] = useState(false)
  const [error, setError] = useState(null)
  const fileInputRef = useRef(null)

  const handleDrag = (e) => { e.preventDefault(); e.stopPropagation() }
  const handleDragIn = (e) => { e.preventDefault(); e.stopPropagation(); setIsDragging(true) }
  const handleDragOut = (e) => { e.preventDefault(); e.stopPropagation(); setIsDragging(false) }
  const handleDrop = (e) => {
    e.preventDefault(); e.stopPropagation(); setIsDragging(false)
    handleFiles(Array.from(e.dataTransfer.files))
  }

  const handleFiles = (newFiles) => {
    const valid = newFiles.filter(f =>
      f.name.toLowerCase().endsWith('.dcm') ||
      f.name.toLowerCase().endsWith('.dicom') ||
      f.name.toLowerCase().endsWith('.nii') ||
      f.name.toLowerCase().endsWith('.nii.gz') ||
      !f.name.includes('.')
    )
    if (valid.length === 0) {
      toast.error('Please upload DICOM (.dcm) or NIfTI (.nii/.nii.gz) files')
      return
    }
    setFiles(prev => [...prev, ...valid])
    setError(null)
    toast.success(`Added ${valid.length} file(s)`)
  }

  const removeFile = (i) => setFiles(prev => prev.filter((_, idx) => idx !== i))
  const clearFiles = () => { setFiles([]); setError(null) }

  const handleRun = async () => {
    if (files.length === 0) { toast.error('Please upload scan files first'); return }
    setIsRunning(true)
    setError(null)
    try {
      const result = await compareModels(files)
      toast.success('Comparison complete!')
      navigate('/compare/results', { state: { result } })
    } catch (err) {
      const msg = err.response?.data?.detail || err.message || 'Comparison failed'
      setError(msg)
      toast.error(msg)
    } finally {
      setIsRunning(false)
    }
  }

  return (
    <div className="min-h-screen py-12 px-4">
      <div className="max-w-4xl mx-auto">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center mb-12"
        >
          <div className="flex items-center justify-center gap-3 mb-4">
            <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-primary-500 to-accent-500 flex items-center justify-center">
              <ScaleIcon className="w-7 h-7 text-white" />
            </div>
            <h1 className="text-4xl font-bold">
              Model <span className="text-primary-400">Comparison</span>
            </h1>
          </div>
          <p className="text-surface-400 max-w-xl mx-auto">
            Upload your scans and run both the <strong className="text-white">MedGemma</strong> detector
            and the <strong className="text-white">Legacy ML</strong> model simultaneously.
            Results are displayed side by side.
          </p>

          {/* Model badges */}
          <div className="flex items-center justify-center gap-3 mt-6">
            <span className="px-4 py-1.5 rounded-full bg-primary-500/20 text-primary-400 text-sm font-medium border border-primary-500/30">
              🧠 MedGemma (LLM + Region Detection)
            </span>
            <span className="text-surface-600">vs</span>
            <span className="px-4 py-1.5 rounded-full bg-surface-700/50 text-surface-400 text-sm font-medium border border-surface-700">
              🤖 ResNet3D-18 (Legacy ML)
            </span>
          </div>
        </motion.div>

        <div className="space-y-6">
          {/* Upload Zone */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
            className={`card p-8 border-2 border-dashed transition-all cursor-pointer ${
              isDragging
                ? 'border-primary-400 bg-primary-500/10'
                : 'border-surface-700 hover:border-primary-500/50'
            }`}
            onDragEnter={handleDragIn}
            onDragLeave={handleDragOut}
            onDragOver={handleDrag}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
          >
            <input
              ref={fileInputRef}
              type="file"
              multiple
              accept=".dcm,.dicom,.nii,.nii.gz,application/dicom"
              onChange={(e) => handleFiles(Array.from(e.target.files))}
              className="hidden"
            />
            <div className="text-center">
              <CloudArrowUpIcon className={`w-16 h-16 mx-auto mb-4 ${isDragging ? 'text-primary-400' : 'text-surface-500'}`} />
              <h3 className="text-xl font-semibold mb-2">
                {isDragging ? 'Drop files here' : 'Upload Brain Scans'}
              </h3>
              <p className="text-surface-400 text-sm">Drag & drop or click to browse</p>
              <p className="text-surface-500 text-xs mt-2">Supports: .dcm, .nii, .nii.gz</p>
            </div>
          </motion.div>

          {/* File List */}
          {files.length > 0 && (
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="card p-4">
              <div className="flex justify-between items-center mb-3">
                <h4 className="font-medium text-surface-200">{files.length} file(s) selected</h4>
                <button onClick={clearFiles} className="text-sm text-surface-400 hover:text-risk-high transition-colors">
                  Clear all
                </button>
              </div>
              <div className="max-h-40 overflow-y-auto space-y-2">
                {files.map((f, i) => (
                  <div key={i} className="flex items-center justify-between bg-surface-800/50 rounded-lg px-3 py-2">
                    <span className="text-sm text-surface-300 truncate flex-1">{f.name}</span>
                    <span className="text-xs text-surface-500 ml-2">{(f.size / 1024).toFixed(0)} KB</span>
                    <button onClick={(e) => { e.stopPropagation(); removeFile(i) }} className="ml-2 text-surface-500 hover:text-risk-high">
                      <XMarkIcon className="w-4 h-4" />
                    </button>
                  </div>
                ))}
              </div>
            </motion.div>
          )}

          {/* Error */}
          {error && (
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="card p-4 bg-risk-high/10 border border-risk-high/30">
              <p className="text-risk-high text-sm">{error}</p>
            </motion.div>
          )}

          {/* Run Button */}
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.2 }}>
            <motion.button
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              onClick={handleRun}
              disabled={isRunning || files.length === 0}
              className="w-full btn-primary py-5 text-lg disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-3"
            >
              {isRunning ? (
                <>
                  <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin" />
                  Running Both Models...
                </>
              ) : (
                <>
                  <BeakerIcon className="w-6 h-6" />
                  Compare {files.length > 0 ? `${files.length} Scan(s)` : 'Scans'}
                </>
              )}
            </motion.button>
          </motion.div>

          <p className="text-center text-surface-500 text-sm">
            MedGemma analysis may take several minutes for large scan sets.
          </p>
        </div>
      </div>
    </div>
  )
}
