import { useState, useRef } from 'react'
import { motion } from 'framer-motion'
import { 
  BeakerIcon, 
  DocumentTextIcon,
  ExclamationTriangleIcon,
  CheckCircleIcon,
  SparklesIcon,
  EyeIcon,
  CloudArrowUpIcon,
  XMarkIcon
} from '@heroicons/react/24/outline'
import toast from 'react-hot-toast'
import api from '../api/client'

export default function MedGemma() {
  const [files, setFiles] = useState([])
  const [isDragging, setIsDragging] = useState(false)
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [uploadedImages, setUploadedImages] = useState([]) // For previews
  const [selectedSlice, setSelectedSlice] = useState(null)
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
      file.name.endsWith('.dcm') || 
      file.type.startsWith('image/') ||
      file.name.match(/\.(jpg|jpeg|png|bmp|tif|tiff)$/i)
    )
    
    if (validFiles.length === 0) {
      toast.error('Please upload DICOM or image files')
      return
    }
    
    setFiles(prev => [...prev, ...validFiles])
    setResult(null)
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
    setResult(null)
    setError(null)
  }

  const handleAnalyze = async () => {
    if (files.length === 0) {
      toast.error('Please upload CT slices first')
      return
    }

    setIsAnalyzing(true)
    setResult(null)
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
      
      setResult(response.data)
      toast.success('Analysis complete!')
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
              <span className="text-primary-400">CT Scan</span> Analysis
            </h1>
          </div>
          <p className="text-surface-400 max-w-xl mx-auto">
            Upload your CT slices and get instant analysis with visual heatmaps.
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
              accept=".dcm,.jpg,.jpeg,.png,.bmp,.tif,.tiff,image/*"
              onChange={handleFileSelect}
              className="hidden"
            />
            
            <div className="text-center">
              <CloudArrowUpIcon className={`w-16 h-16 mx-auto mb-4 ${
                isDragging ? 'text-primary-400' : 'text-surface-500'
              }`} />
              <h3 className="text-xl font-semibold mb-2">
                {isDragging ? 'Drop files here' : 'Upload CT Slices'}
              </h3>
              <p className="text-surface-400 text-sm">
                Drag & drop DICOM or image files, or click to browse
              </p>
              <p className="text-surface-500 text-xs mt-2">
                Supports: .dcm, .jpg, .png, .bmp, .tif
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
                  Analyze {files.length > 0 ? `${files.length} Slice(s)` : 'CT Scan'}
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

          {/* Results */}
          {result && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="space-y-4"
            >
              {/* Summary Card */}
              <div className="card p-6">
                <div className="flex items-center gap-3 mb-4">
                  <DocumentTextIcon className="w-6 h-6 text-primary-400" />
                  <h2 className="text-xl font-semibold">Analysis Report</h2>
                </div>
                
                <div className="grid grid-cols-3 gap-4 mb-6">
                  <div className="bg-surface-800/50 rounded-xl p-4 text-center">
                    <p className="text-2xl font-bold text-primary-400">{result.slices_analyzed}</p>
                    <p className="text-sm text-surface-400">Slices Analyzed</p>
                  </div>
                  <div className="bg-surface-800/50 rounded-xl p-4 text-center">
                    <p className="text-2xl font-bold text-primary-400">{result.findings?.length || 0}</p>
                    <p className="text-sm text-surface-400">Findings</p>
                  </div>
                  <div className="bg-surface-800/50 rounded-xl p-4 text-center">
                    <p className="text-2xl font-bold text-primary-400">{result.processing_time?.toFixed(1)}s</p>
                    <p className="text-sm text-surface-400">Processing Time</p>
                  </div>
                </div>

                {/* Status Badge */}
                <div className={`inline-flex items-center gap-2 px-4 py-2 rounded-full mb-4 ${
                  result.has_findings 
                    ? 'bg-risk-high/20 text-risk-high'
                    : 'bg-risk-low/20 text-risk-low'
                }`}>
                  {result.has_findings ? (
                    <ExclamationTriangleIcon className="w-5 h-5" />
                  ) : (
                    <CheckCircleIcon className="w-5 h-5" />
                  )}
                  <span className="font-medium">
                    {result.has_findings ? 'Potential Findings Detected' : 'No Obvious Abnormalities'}
                  </span>
                </div>

                {/* Report */}
                <div className="bg-surface-800/50 rounded-xl p-6 mt-4">
                  <h3 className="text-sm font-medium text-surface-300 mb-3">Report</h3>
                  <p className="text-surface-200 whitespace-pre-wrap leading-relaxed">
                    {result.report}
                  </p>
                </div>
              </div>

              {/* Findings with Images */}
              {result.findings && result.findings.length > 0 && (
                <div className="card p-6">
                  <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                    <ExclamationTriangleIcon className="w-5 h-5 text-risk-high" />
                    Detailed Findings
                  </h3>
                  <div className="space-y-4">
                    {result.findings.map((finding, idx) => (
                      <motion.div
                        key={idx}
                        initial={{ opacity: 0, x: -20 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: idx * 0.1 }}
                        className="bg-risk-high/10 border border-risk-high/30 rounded-xl p-4"
                      >
                        <div className="flex gap-4">
                          {/* Image Preview */}
                          {finding.image && (
                            <div 
                              className="w-32 h-32 rounded-lg overflow-hidden flex-shrink-0 cursor-pointer border border-surface-700 hover:border-primary-400 transition-colors"
                              onClick={() => setSelectedSlice({ index: finding.slice_index, image: finding.image })}
                            >
                              <img 
                                src={finding.image} 
                                alt={`Slice ${finding.slice_number}`}
                                className="w-full h-full object-cover"
                              />
                            </div>
                          )}
                          <div className="flex-1">
                            <p className="font-medium text-risk-high mb-1">
                              Slice {finding.slice_number}
                            </p>
                            <p className="text-surface-300 text-sm whitespace-pre-wrap">
                              {finding.response}
                            </p>
                          </div>
                        </div>
                      </motion.div>
                    ))}
                  </div>
                </div>
              )}

              {/* Image Viewer Modal */}
              {selectedSlice && (
                <motion.div
                  initial={{ opacity: 0, scale: 0.9 }}
                  animate={{ opacity: 1, scale: 1 }}
                  className="card p-6"
                >
                  <div className="flex justify-between items-center mb-4">
                    <h3 className="text-lg font-semibold flex items-center gap-2">
                      <EyeIcon className="w-5 h-5 text-primary-400" />
                      Slice {selectedSlice.index + 1} - Enlarged View
                    </h3>
                    <button
                      onClick={() => setSelectedSlice(null)}
                      className="text-surface-400 hover:text-white transition-colors"
                    >
                      ✕ Close
                    </button>
                  </div>
                  <div className="flex justify-center">
                    <img 
                      src={selectedSlice.image} 
                      alt={`Slice ${selectedSlice.index + 1}`}
                      className="max-w-full max-h-[500px] rounded-xl border border-surface-700"
                    />
                  </div>
                </motion.div>
              )}
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
