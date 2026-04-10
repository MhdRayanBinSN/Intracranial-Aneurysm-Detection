/**
 * API client for communicating with the FastAPI backend.
 */

import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

// Create axios instance
const api = axios.create({
    baseURL: API_BASE_URL,
    timeout: 1200000, // 20 minutes for MedGemma analysis
    headers: {
        'Content-Type': 'application/json',
    },
});

// Health check
export const checkHealth = async () => {
    const response = await api.get('/health');
    return response.data;
};

// Analyze DICOM files
export const analyzeDicom = async (files, modality = 'CTA', onProgress) => {
    const formData = new FormData();

    files.forEach((file) => {
        formData.append('files', file);
    });

    const response = await api.post('/analyze', formData, {
        headers: {
            'Content-Type': 'multipart/form-data',
        },
        params: { modality },
        onUploadProgress: (progressEvent) => {
            if (onProgress) {
                const percentCompleted = Math.round(
                    (progressEvent.loaded * 100) / progressEvent.total
                );
                onProgress(percentCompleted);
            }
        },
    });

    return response.data;
};

// Get analysis by ID
export const getAnalysis = async (analysisId) => {
    const response = await api.get(`/analysis/${analysisId}`);
    return response.data;
};

// Get anatomical locations list
export const getLocations = async () => {
    const response = await api.get('/locations');
    return response.data;
};

// Demo prediction (for testing without model)
export const getDemoPrediction = async () => {
    const response = await api.post('/demo/predict');
    return response.data;
};

// ==================== MEDGEMMA API ====================

// Get patients for MedGemma analysis
export const getMedGemmaPatients = async () => {
    const response = await api.get('/medgemma/patients');
    return response.data;
};

// Analyze with MedGemma
export const analyzeMedGemma = async (seriesUid, sampleEvery = 20) => {
    const response = await api.post('/medgemma/analyze', null, {
        params: { series_uid: seriesUid, sample_every: sampleEvery },
        timeout: 300000, // 5 minutes for MedGemma
    });
    return response.data;
};

// Get MedGemma results
export const getMedGemmaResults = async (analysisId) => {
    const response = await api.get(`/medgemma/results/${analysisId}`);
    return response.data;
};

// Get slice image with optional heatmap
export const getSliceImage = async (seriesUid, sliceIndex, heatmap = false) => {
    const response = await api.get(`/medgemma/slice/${seriesUid}/${sliceIndex}`, {
        params: { heatmap }
    });
    return response.data;
};

// Compare both models on same files
export const compareModels = async (files) => {
    const formData = new FormData()
    files.forEach(f => formData.append('files', f))
    const response = await api.post('/compare', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
        timeout: 1200000, // 20 min — MedGemma takes time
    })
    return response.data
}

// Export the API instance
export default api;

