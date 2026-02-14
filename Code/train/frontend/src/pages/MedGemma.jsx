import { useState, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import { motion } from 'framer-motion'
import { 
  BeakerIcon, 
  SparklesIcon,
  CloudArrowUpIcon,
  XMarkIcon
} from '@heroicons/react/24/outline'
import toast from 'react-hot-toast'
import api from '../api/client'

export default function MedGemma() {
  const navigate = useNavigate()
  const [files, setFiles] = useState([])
  const [isDragging, setIsDragging] = useState(false)
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [error, setError] = useState(null)
  const [uploadedImages, setUploadedImages] = useState([]) // For previews
  const fileInputRef = useRef(null)

  // Handle drag events
  const handleDrag = (e) => {
    e.preventDefault()
    e.stopPropagation()
  }

  const handleDragIn = (e) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragging(true)
  }

  const handleDragOut = (e) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragging(false)
  }

  const handleDrop = (e) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragging(false)
    
    const droppedFiles = Array.from(e.dataTransfer.files)
    handleFiles(droppedFiles)
  }

  const handleFileSelect = (e) => {
    const selectedFiles = Array.from(e.target.files)
    handleFiles(selectedFiles)
  }

  const handleFiles = (newFiles) => {
    // Filter for image files and DICOM
    const validFiles = newFiles.filter(file => 
      file.name.toLowerCase().endsWith('.dcm') || 
      file.name.toLowerCase().endsWith('.nii') ||
      file.name.toLowerCase().endsWith('.nii.gz')
    )
    
    if (validFiles.length === 0) {
      toast.error('Please upload DICOM (.dcm) or NIfTI (.nii) files')
      return
    }
    
    setFiles(prev => [...prev, ...validFiles])
    setError(null)
    
    // Create previews for non-DICOM files
    validFiles.forEach(file => {
      if (file.type.startsWith('image/')) {
        const reader = new FileReader()
        reader.onload = (e) => {
          setUploadedImages(prev => [...prev, { name: file.name, src: e.target.result }])
        }
        reader.readAsDataURL(file)
      }
    })
    
    toast.success(`Added ${validFiles.length} file(s)`)
  }

  const removeFile = (index) => {
    setFiles(prev => prev.filter((_, i) => i !== index))
    setUploadedImages(prev => prev.filter((_, i) => i !== index))
  }

  const clearFiles = () => {
    setFiles([])
    setUploadedImages([])
    setError(null)
  }

  const handleAnalyze = async () => {
    if (files.length === 0) {
      toast.error('Please upload CT slices first')
      return
    }

    setIsAnalyzing(true)
    setError(null)

    try {
      // Create FormData with files
      const formData = new FormData()
      files.forEach((file, index) => {
        formData.append('files', file)
      })

      const response = await api.post('/medgemma/analyze-upload', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
        timeout: 120000, // 2 minutes
      })
      
      toast.success('Analysis complete! Redirecting to results...')
      // Navigate to results page with data
      navigate('/medgemma/results', { state: { result: response.data } })
    } catch (err) {
      console.error('Analysis error:', err)
      setError(err.response?.data?.detail || 'Analysis failed')
      toast.error('Analysis failed')
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
          <div className="flex items-center justify-center gap-3 mb-4">
            <SparklesIcon className="w-10 h-10 text-primary-400" />
            <h1 className="text-4xl font-bold">
              <span className="text-primary-400">Intracranial Aneurysm</span> Analysis
            </h1>
          </div>
          <p className="text-surface-400 max-w-xl mx-auto">
            Upload your CTA, MRA, or MRI scans and get instant AI-powered detection and analysis.
          </p>
        </motion.div>

        <div className="space-y-6">
          {/* Upload Area */}
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
              accept=".dcm,.nii,.nii.gz"
              onChange={handleFileSelect}
              className="hidden"
            />
            
            <div className="text-center">
              <CloudArrowUpIcon className={`w-16 h-16 mx-auto mb-4 ${
                isDragging ? 'text-primary-400' : 'text-surface-500'
              }`} />
              <h3 className="text-xl font-semibold mb-2">
                {isDragging ? 'Drop files here' : 'Upload Brain Scans'}
              </h3>
              <p className="text-surface-400 text-sm">
                Drag & drop DICOM or image files, or click to browse
              </p>
              <p className="text-surface-500 text-xs mt-2">
                Supports: .dcm, .nii, .nii.gz (Medical Formats Only)
              </p>
            </div>
          </motion.div>

          {/* File List */}
          {files.length > 0 && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="card p-4"
            >
              <div className="flex justify-between items-center mb-3">
                <h4 className="font-medium text-surface-200">
                  {files.length} file(s) selected
                </h4>
                <button
                  onClick={(e) => { e.stopPropagation(); clearFiles(); }}
                  className="text-sm text-surface-400 hover:text-risk-high transition-colors"
                >
                  Clear all
                </button>
              </div>
              <div className="max-h-40 overflow-y-auto space-y-2">
                {files.map((file, idx) => (
                  <div 
                    key={idx}
                    className="flex items-center justify-between bg-surface-800/50 rounded-lg px-3 py-2"
                  >
                    <span className="text-sm text-surface-300 truncate flex-1">
                      {file.name}
                    </span>
                    <span className="text-xs text-surface-500 ml-2">
                      {(file.size / 1024).toFixed(0)} KB
                    </span>
                    <button
                      onClick={(e) => { e.stopPropagation(); removeFile(idx); }}
                      className="ml-2 text-surface-500 hover:text-risk-high"
                    >
                      <XMarkIcon className="w-4 h-4" />
                    </button>
                  </div>
                ))}
              </div>
            </motion.div>
          )}

          {/* Analyze Button */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
          >
            <motion.button
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              onClick={handleAnalyze}
              disabled={isAnalyzing || files.length === 0}
              className="w-full btn-primary py-5 text-lg disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-3"
            >
              {isAnalyzing ? (
                <>
                  <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin" />
                  Analyzing {files.length} slice(s)...
                </>
              ) : (
                <>
                  <BeakerIcon className="w-6 h-6" />
                  Analyze {files.length > 0 ? `${files.length} Scan(s)` : 'Brain Scan'}
                </>
              )}
            </motion.button>
          </motion.div>

          {/* Error Display */}
          {error && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="card p-4 bg-risk-high/10 border border-risk-high/30"
            >
              <p className="text-risk-high text-sm">{error}</p>
            </motion.div>
          )}



          {/* Info Text */}
          <p className="text-center text-surface-500 text-sm">
            Demo analysis for presentation purposes. Results should be verified by a qualified radiologist.
          </p>
        </div>
      </div>
    </div>
  )
}
