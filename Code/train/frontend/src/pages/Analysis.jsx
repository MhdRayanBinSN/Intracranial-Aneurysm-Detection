import { useState, useCallback } from 'react'
import { useNavigate } from 'react-router-dom'
import { motion } from 'framer-motion'
import { useDropzone } from 'react-dropzone'
import { 
  CloudArrowUpIcon, 
  DocumentIcon, 
  XMarkIcon,
  BeakerIcon,
  ExclamationTriangleIcon 
} from '@heroicons/react/24/outline'
import toast from 'react-hot-toast'
import { analyzeDicom, getDemoPrediction } from '../api/client'

const modalityOptions = [
  { value: 'CTA', label: 'CTA (CT Angiography)' },
  { value: 'MRA', label: 'MRA (MR Angiography)' },
  { value: 'MRI', label: 'MRI (T1/T2)' },
]

export default function Analysis() {
  const navigate = useNavigate()
  const [files, setFiles] = useState([])
  const [modality, setModality] = useState('CTA')
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [progress, setProgress] = useState(0)
  const [error, setError] = useState(null)

  const onDrop = useCallback((acceptedFiles) => {
    const dicomFiles = acceptedFiles.filter(file => 
      file.name.endsWith('.dcm') || 
      file.name.endsWith('.dicom') ||
      !file.name.includes('.')  // DICOM files often have no extension
    )
    
    if (dicomFiles.length === 0) {
      toast.error('Please upload DICOM files (.dcm)')
      return
    }
    
    setFiles(prev => [...prev, ...dicomFiles])
    setError(null)
  }, [])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'application/dicom': ['.dcm', '.dicom'],
    },
    multiple: true,
  })

  const removeFile = (index) => {
    setFiles(prev => prev.filter((_, i) => i !== index))
  }

  const handleAnalyze = async () => {
    if (files.length === 0) {
      toast.error('Please upload at least one DICOM file')
      return
    }

    setIsAnalyzing(true)
    setProgress(0)
    setError(null)

    try {
      const result = await analyzeDicom(files, modality, (p) => {
        setProgress(p)
      })
      
      toast.success('Analysis complete!')
      navigate(`/results/${result.id}`, { state: { result } })
    } catch (err) {
      console.error('Analysis error:', err)
      setError(err.response?.data?.detail || 'Analysis failed. Please try again.')
      toast.error('Analysis failed')
    } finally {
      setIsAnalyzing(false)
    }
  }

  const handleDemo = async () => {
    setIsAnalyzing(true)
    setProgress(50)

    try {
      const result = await getDemoPrediction()
      setProgress(100)
      toast.success('Demo analysis complete!')
      navigate(`/results/${result.id}`, { state: { result } })
    } catch (err) {
      console.error('Demo error:', err)
      toast.error('Demo failed. Is the backend running?')
    } finally {
      setIsAnalyzing(false)
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
          <h1 className="text-4xl font-bold mb-4">
            Upload <span className="text-primary-400">DICOM</span> Files
          </h1>
          <p className="text-surface-400 max-w-xl mx-auto">
            Upload your medical imaging files for AI-powered aneurysm detection. 
            Supports CTA, MRA, and MRI scans.
          </p>
        </motion.div>

        <div className="space-y-6">
          {/* Modality Selection */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
            className="card p-6"
          >
            <label className="block text-sm font-medium text-surface-300 mb-3">
              Imaging Modality
            </label>
            <div className="grid grid-cols-3 gap-3">
              {modalityOptions.map((option) => (
                <button
                  key={option.value}
                  onClick={() => setModality(option.value)}
                  className={`p-4 rounded-xl border-2 transition-all ${
                    modality === option.value
                      ? 'border-primary-500 bg-primary-500/10 text-primary-400'
                      : 'border-surface-700 hover:border-surface-600 text-surface-300'
                  }`}
                >
                  <span className="font-medium">{option.value}</span>
                  <p className="text-xs mt-1 text-surface-500">{option.label.split(' ').slice(1).join(' ')}</p>
                </button>
              ))}
            </div>
          </motion.div>

          {/* Upload Area */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
          >
            <div
              {...getRootProps()}
              className={`card p-12 border-2 border-dashed transition-all cursor-pointer ${
                isDragActive
                  ? 'border-primary-500 bg-primary-500/10'
                  : 'border-surface-700 hover:border-surface-600'
              }`}
            >
              <input {...getInputProps()} />
              <div className="text-center">
                <CloudArrowUpIcon className={`w-16 h-16 mx-auto mb-4 ${
                  isDragActive ? 'text-primary-400' : 'text-surface-500'
                }`} />
                <p className="text-lg font-medium mb-2">
                  {isDragActive ? 'Drop files here...' : 'Drag & drop DICOM files here'}
                </p>
                <p className="text-surface-500 text-sm">
                  or click to browse files
                </p>
              </div>
            </div>
          </motion.div>

          {/* File List */}
          {files.length > 0 && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              className="card p-6"
            >
              <div className="flex items-center justify-between mb-4">
                <h3 className="font-medium">Uploaded Files ({files.length})</h3>
                <button
                  onClick={() => setFiles([])}
                  className="text-sm text-surface-400 hover:text-white"
                >
                  Clear All
                </button>
              </div>
              <div className="space-y-2 max-h-60 overflow-y-auto">
                {files.map((file, index) => (
                  <motion.div
                    key={index}
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: index * 0.05 }}
                    className="flex items-center justify-between p-3 bg-surface-800/50 rounded-lg"
                  >
                    <div className="flex items-center gap-3">
                      <DocumentIcon className="w-5 h-5 text-primary-400" />
                      <span className="text-sm truncate max-w-xs">{file.name}</span>
                      <span className="text-xs text-surface-500">
                        {(file.size / 1024).toFixed(1)} KB
                      </span>
                    </div>
                    <button
                      onClick={() => removeFile(index)}
                      className="p-1 hover:bg-surface-700 rounded"
                    >
                      <XMarkIcon className="w-4 h-4 text-surface-400" />
                    </button>
                  </motion.div>
                ))}
              </div>
            </motion.div>
          )}

          {/* Error Display */}
          {error && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="flex items-center gap-3 p-4 bg-risk-high/10 border border-risk-high/30 rounded-xl text-risk-high"
            >
              <ExclamationTriangleIcon className="w-5 h-5 flex-shrink-0" />
              <span>{error}</span>
            </motion.div>
          )}

          {/* Progress Bar */}
          {isAnalyzing && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="card p-6"
            >
              <div className="flex items-center gap-4 mb-3">
                <BeakerIcon className="w-6 h-6 text-primary-400 animate-pulse" />
                <span className="font-medium">Analyzing...</span>
                <span className="text-surface-400">{progress}%</span>
              </div>
              <div className="h-2 bg-surface-800 rounded-full overflow-hidden">
                <motion.div
                  initial={{ width: 0 }}
                  animate={{ width: `${progress}%` }}
                  className="h-full bg-gradient-to-r from-primary-500 to-accent-500"
                />
              </div>
            </motion.div>
          )}

          {/* Action Buttons */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3 }}
            className="flex flex-col sm:flex-row gap-4"
          >
            <motion.button
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              onClick={handleAnalyze}
              disabled={isAnalyzing || files.length === 0}
              className="flex-1 btn-primary py-4 text-lg disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {isAnalyzing ? 'Analyzing...' : 'Analyze Files'}
            </motion.button>
            <motion.button
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              onClick={handleDemo}
              disabled={isAnalyzing}
              className="btn-secondary py-4 text-lg disabled:opacity-50"
            >
              Try Demo
            </motion.button>
          </motion.div>

          {/* Help Text */}
          <p className="text-center text-surface-500 text-sm">
            Your data is processed locally and never stored. 
            For best results, upload a complete DICOM series.
          </p>
        </div>
      </div>
    </div>
  )
}
