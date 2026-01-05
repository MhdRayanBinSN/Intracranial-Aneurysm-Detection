/**
 * API client for communicating with the FastAPI backend.
 */

import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

// Create axios instance
const api = axios.create({
    baseURL: API_BASE_URL,
    timeout: 120000, // 2 minutes for long inference
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

// Export the API instance
export default api;
