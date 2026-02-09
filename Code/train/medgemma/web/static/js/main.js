/**
 * MedGemma Dashboard - Frontend JavaScript
 */

// API Base URL
const API_BASE = '';

// Current job tracking
let currentJobId = null;
let pollInterval = null;

// ==================== INITIALIZATION ====================
document.addEventListener('DOMContentLoaded', () => {
    initNavigation();
    loadPatients();
    setupEventListeners();
});

// ==================== NAVIGATION ====================
function initNavigation() {
    const navItems = document.querySelectorAll('.nav-item');

    navItems.forEach(item => {
        item.addEventListener('click', (e) => {
            e.preventDefault();

            // Update active state
            navItems.forEach(n => n.classList.remove('active'));
            item.classList.add('active');

            // Show corresponding page
            const pageId = item.dataset.page;
            showPage(pageId);
        });
    });
}

function showPage(pageId) {
    // Hide all pages
    document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));

    // Show target page
    const targetPage = document.getElementById(`page-${pageId}`);
    if (targetPage) {
        targetPage.classList.add('active');
    }

    // Update header
    const titles = {
        'overview': 'Dashboard Overview',
        'analyze': 'Analyze Patient',
        'results': 'Analysis Results',
        'compare': 'Model Comparison'
    };

    document.getElementById('page-title').textContent = titles[pageId] || 'Dashboard';
    document.getElementById('current-page').textContent = titles[pageId]?.split(' ')[0] || pageId;
}

// ==================== EVENT LISTENERS ====================
function setupEventListeners() {
    // New Analysis button
    document.getElementById('new-analysis-btn').addEventListener('click', () => {
        document.querySelector('[data-page="analyze"]').click();
    });

    // Start Analysis button
    document.getElementById('start-analysis-btn').addEventListener('click', startAnalysis);
}

// ==================== LOAD PATIENTS ====================
async function loadPatients() {
    const select = document.getElementById('patient-select');

    try {
        const response = await fetch(`${API_BASE}/api/patients`);
        const data = await response.json();

        if (data.success) {
            select.innerHTML = '<option value="">Select a patient...</option>';

            data.patients.forEach(patient => {
                const option = document.createElement('option');
                option.value = patient.series_uid;
                option.textContent = patient.display_name;
                select.appendChild(option);
            });
        } else {
            select.innerHTML = '<option value="">Error loading patients</option>';
        }
    } catch (error) {
        console.error('Error loading patients:', error);
        select.innerHTML = '<option value="">Error loading patients</option>';
    }
}

// ==================== ANALYSIS ====================
async function startAnalysis() {
    const seriesUid = document.getElementById('patient-select').value;
    const useMedgemma = document.getElementById('use-medgemma').checked;
    const useResnet = document.getElementById('use-resnet').checked;

    if (!seriesUid) {
        alert('Please select a patient first!');
        return;
    }

    if (!useMedgemma && !useResnet) {
        alert('Please select at least one model!');
        return;
    }

    // Determine model type
    let modelType = 'both';
    if (useMedgemma && !useResnet) modelType = 'medgemma';
    if (!useMedgemma && useResnet) modelType = 'resnet';

    // Show progress
    document.querySelector('.patient-selector').style.display = 'none';
    document.getElementById('analysis-progress').style.display = 'block';
    document.getElementById('progress-fill').style.width = '0%';
    document.getElementById('progress-text').textContent = 'Starting analysis...';

    try {
        // Start analysis
        const response = await fetch(`${API_BASE}/api/analyze`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ series_uid: seriesUid, model_type: modelType })
        });

        const data = await response.json();

        if (data.success) {
            currentJobId = data.job_id;
            startPolling();
        } else {
            showError(data.error || 'Failed to start analysis');
        }
    } catch (error) {
        showError('Failed to connect to server');
    }
}

function startPolling() {
    pollInterval = setInterval(async () => {
        try {
            const response = await fetch(`${API_BASE}/api/status/${currentJobId}`);
            const data = await response.json();

            if (data.success) {
                const job = data.job;

                // Update progress
                const progress = job.progress || 0;
                document.getElementById('progress-fill').style.width = `${progress}%`;
                document.getElementById('progress-text').textContent =
                    `Analyzing... ${progress}% (${job.status})`;

                // Check if completed
                if (job.status === 'completed') {
                    clearInterval(pollInterval);
                    onAnalysisComplete();
                } else if (job.status === 'error') {
                    clearInterval(pollInterval);
                    showError(job.error || 'Analysis failed');
                }
            }
        } catch (error) {
            console.error('Polling error:', error);
        }
    }, 2000);
}

async function onAnalysisComplete() {
    document.getElementById('progress-fill').style.width = '100%';
    document.getElementById('progress-text').textContent = 'Analysis complete!';

    // Fetch results
    try {
        const response = await fetch(`${API_BASE}/api/results/${currentJobId}`);
        const data = await response.json();

        if (data.success) {
            displayResults(data);

            // Navigate to results page after delay
            setTimeout(() => {
                document.querySelector('[data-page="results"]').click();
                resetAnalyzeForm();
            }, 1500);
        }
    } catch (error) {
        showError('Failed to fetch results');
    }
}

function displayResults(data) {
    // Update results time
    document.getElementById('results-time').textContent = `Completed at ${new Date().toLocaleTimeString()}`;

    // MedGemma Results
    const medgemmaContainer = document.getElementById('medgemma-results');
    if (data.medgemma) {
        if (data.medgemma.error) {
            medgemmaContainer.innerHTML = `<p class="error">${data.medgemma.error}</p>`;
        } else {
            let html = `
                <div class="result-summary">
                    <p><strong>Slices Analyzed:</strong> ${data.medgemma.analyzed_slices}</p>
                    <p><strong>Findings:</strong> ${data.medgemma.findings?.length || 0}</p>
                </div>
            `;

            if (data.medgemma.has_aneurysm) {
                html += `<div class="finding-item">
                    <div class="finding-title">⚠️ Potential Aneurysm Detected</div>
                </div>`;

                data.medgemma.findings.forEach(finding => {
                    html += `
                        <div class="finding-item">
                            <div class="finding-title">Slice ${finding.slice_number}</div>
                            <div class="finding-text">${finding.response}</div>
                        </div>
                    `;
                });
            } else {
                html += `<div class="finding-item clear">
                    <div class="finding-title">✅ No Aneurysm Detected</div>
                    <div class="finding-text">All analyzed slices appear normal.</div>
                </div>`;
            }

            medgemmaContainer.innerHTML = html;
        }
    } else {
        medgemmaContainer.innerHTML = '<p class="placeholder">MedGemma not used for this analysis.</p>';
    }

    // ResNet Results
    const resnetContainer = document.getElementById('resnet-results');
    if (data.resnet) {
        if (data.resnet.error) {
            resnetContainer.innerHTML = `<p class="error">${data.resnet.error}</p>`;
        } else {
            let html = `
                <div class="result-summary">
                    <p><strong>Prediction:</strong> ${data.resnet.prediction}</p>
                    <p><strong>Confidence:</strong> ${(data.resnet.confidence * 100).toFixed(1)}%</p>
                </div>
            `;

            if (data.resnet.locations && data.resnet.locations.length > 0) {
                html += `<div class="finding-title" style="margin: 16px 0 8px;">Detected Locations:</div>`;

                data.resnet.locations.forEach(loc => {
                    const confidenceClass = loc.confidence > 0.8 ? 'high' : 'medium';
                    html += `
                        <div class="finding-item">
                            <div class="finding-title">📍 ${loc.name}</div>
                            <span class="confidence-badge ${confidenceClass}">${(loc.confidence * 100).toFixed(0)}% confidence</span>
                        </div>
                    `;
                });
            }

            if (data.resnet.note) {
                html += `<p class="note" style="margin-top: 16px; font-size: 13px; color: #6b7280;">${data.resnet.note}</p>`;
            }

            resnetContainer.innerHTML = html;
        }
    } else {
        resnetContainer.innerHTML = '<p class="placeholder">3D ResNet not used for this analysis.</p>';
    }

    // Update compare page
    updateCompareView(data);
}

function updateCompareView(data) {
    const container = document.getElementById('compare-results');

    let html = `
        <div class="compare-table">
            <table>
                <thead>
                    <tr>
                        <th>Aspect</th>
                        <th>MedGemma</th>
                        <th>3D ResNet</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>Detection</td>
                        <td>${data.medgemma?.has_aneurysm ? '⚠️ Found' : '✅ Clear'}</td>
                        <td>${data.resnet?.prediction || 'N/A'}</td>
                    </tr>
                    <tr>
                        <td>Findings Count</td>
                        <td>${data.medgemma?.findings?.length || 0} slices</td>
                        <td>${data.resnet?.locations?.length || 0} locations</td>
                    </tr>
                    <tr>
                        <td>Confidence</td>
                        <td>95% (per slice)</td>
                        <td>${data.resnet ? (data.resnet.confidence * 100).toFixed(1) + '%' : 'N/A'}</td>
                    </tr>
                </tbody>
            </table>
        </div>
    `;

    container.innerHTML = html;
}

function resetAnalyzeForm() {
    document.querySelector('.patient-selector').style.display = 'block';
    document.getElementById('analysis-progress').style.display = 'none';
    document.getElementById('patient-select').value = '';
}

function showError(message) {
    alert(`Error: ${message}`);
    resetAnalyzeForm();
    if (pollInterval) clearInterval(pollInterval);
}
