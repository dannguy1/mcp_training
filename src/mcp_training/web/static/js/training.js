/**
 * Training management JavaScript functionality
 */

console.log('=== TRAINING.JS SCRIPT LOADING ===');
console.log('Training.js script loaded at:', new Date().toISOString());

// Test if dependencies are available
console.log('Bootstrap available:', typeof bootstrap !== 'undefined');
console.log('Utils available:', typeof utils !== 'undefined');

class TrainingManager {
    constructor() {
        this.jobs = [];
        this.filteredJobs = [];
        this.currentJobId = null;
        this.updateInterval = null;
        this.init();
    }
    
    init() {
        this.setupEventListeners();
        this.loadTrainingJobs();
        this.startAutoRefresh();
    }
    
    setupEventListeners() {
        console.log('Setting up event listeners...');
        
        // Search and filter
        const searchInput = document.getElementById('searchInput');
        const statusFilter = document.getElementById('statusFilter');
        
        if (searchInput) {
            searchInput.addEventListener('input', utils.debounce(() => {
                this.filterJobs();
            }, 300));
        }
        
        if (statusFilter) {
            statusFilter.addEventListener('change', () => {
                this.filterJobs();
            });
        }
        
        // Export file selection
        const exportFile = document.getElementById('exportFile');
        if (exportFile) {
            exportFile.addEventListener('change', () => {
                this.handleExportFileSelect();
            });
        }
        
        // Form validation
        const trainingForm = document.getElementById('trainingForm');
        if (trainingForm) {
            trainingForm.addEventListener('submit', (e) => {
                e.preventDefault();
                this.submitTraining();
            });
        }
        
        // New Training Job button
        const newTrainingJobBtn = document.getElementById('newTrainingJobBtn');
        console.log('New Training Job button found:', !!newTrainingJobBtn);
        if (newTrainingJobBtn) {
            newTrainingJobBtn.addEventListener('click', () => {
                console.log('New Training Job button clicked');
                this.openTrainingModal();
            });
        }
        
        // Submit Training button
        const submitTrainingBtn = document.getElementById('submitTrainingBtn');
        console.log('Submit Training button found:', !!submitTrainingBtn);
        if (submitTrainingBtn) {
            submitTrainingBtn.addEventListener('click', () => {
                console.log('Submit Training button clicked');
                this.submitTraining();
            });
        }
        
        // Refresh Jobs button
        const refreshJobsBtn = document.getElementById('refreshJobsBtn');
        console.log('Refresh Jobs button found:', !!refreshJobsBtn);
        if (refreshJobsBtn) {
            refreshJobsBtn.addEventListener('click', () => {
                console.log('Refresh Jobs button clicked');
                this.refresh();
            });
        }
        
        // Export Jobs button
        const exportJobsBtn = document.getElementById('exportJobsBtn');
        console.log('Export Jobs button found:', !!exportJobsBtn);
        if (exportJobsBtn) {
            exportJobsBtn.addEventListener('click', () => {
                console.log('Export Jobs button clicked');
                this.exportJobs();
            });
        }
        
        // Modal events
        const newTrainingModal = document.getElementById('newTrainingModal');
        if (newTrainingModal) {
            newTrainingModal.addEventListener('hidden.bs.modal', () => {
                this.resetForm();
            });
        }
        
        // Confirm delete
        const confirmDeleteBtn = document.getElementById('confirmDeleteBtn');
        if (confirmDeleteBtn) {
            confirmDeleteBtn.addEventListener('click', () => {
                this.confirmDelete();
            });
        }
        
        console.log('Event listeners setup completed');
    }
    
    async loadTrainingJobs() {
        try {
            utils.showLoading();
            const jobs = await utils.apiCall('/training/jobs');
            this.jobs = jobs;
            this.filteredJobs = [...jobs];
            this.updateTrainingTable();
            this.updateJobStatistics();
        } catch (error) {
            utils.showError('Failed to load training jobs', error);
        } finally {
            utils.hideLoading();
        }
    }
    
    updateTrainingTable() {
        const tableBody = document.querySelector('#trainingTable tbody');
        if (!tableBody) return;
        
        if (this.filteredJobs.length === 0) {
            tableBody.innerHTML = `
                <tr>
                    <td colspan="8" class="text-center text-muted py-4">
                        <i class="bi bi-gear fs-1 mb-3"></i>
                        <p>No training jobs found</p>
                    </td>
                </tr>
            `;
            return;
        }
        
        tableBody.innerHTML = this.filteredJobs.map(job => {
            // Determine evaluation status
            let evalStatus = '';
            let evalStatusClass = 'secondary';
            
            if (job.status === 'completed' && job.evaluation_results) {
                const evaluation = job.evaluation_results;
                if (evaluation.threshold_checks) {
                    const allPassed = Object.values(evaluation.threshold_checks).every(check => check.passed);
                    evalStatus = allPassed ? 'PASS' : 'FAIL';
                    evalStatusClass = allPassed ? 'success' : 'danger';
                } else {
                    evalStatus = 'EVAL';
                    evalStatusClass = 'info';
                }
            } else if (job.status === 'completed') {
                evalStatus = 'NO_EVAL';
                evalStatusClass = 'warning';
            } else if (job.status === 'failed') {
                evalStatus = 'ERROR';
                evalStatusClass = 'danger';
            } else {
                evalStatus = 'PENDING';
                evalStatusClass = 'secondary';
            }
            
            // Get best metric for quick reference
            let bestMetric = '';
            if (job.evaluation_results && job.evaluation_results.accuracy) {
                bestMetric = `Acc: ${(job.evaluation_results.accuracy * 100).toFixed(1)}%`;
            }
            
            return `
                <tr data-job-id="${job.id}">
                    <td>${job.id}</td>
                    <td>
                        <div>
                            <strong>${job.model_name || 'Unnamed Job'}</strong>
                            ${job.description ? `<br><small class="text-muted">${job.description.substring(0, 50)}${job.description.length > 50 ? '...' : ''}</small>` : ''}
                        </div>
                    </td>
                    <td>
                        <span class="badge bg-${utils.getStatusColor(job.status)}">
                            ${job.status}
                        </span>
                    </td>
                    <td>
                        <div>
                            <div>${job.export_file || 'N/A'}</div>
                            ${bestMetric ? `<small class="text-success">${bestMetric}</small>` : ''}
                        </div>
                    </td>
                    <td>${utils.formatDateTime(job.created_at)}</td>
                    <td>
                        <div class="progress" style="height: 6px;">
                            <div class="progress-bar bg-${utils.getStatusColor(job.status)}" 
                                 style="width: ${job.progress || 0}%"></div>
                        </div>
                        <small class="text-muted">${job.progress || 0}%</small>
                    </td>
                    <td>
                        <span class="badge bg-${evalStatusClass}">
                            ${evalStatus}
                        </span>
                    </td>
                    <td class="table-actions">
                        <div class="btn-group btn-group-sm">
                            <button class="btn btn-outline-info" onclick="trainingManager.viewJobDetails('${job.id}')" 
                                    title="View Details">
                                <i class="bi bi-eye"></i>
                            </button>
                            ${job.status === 'running' ? `
                                <button class="btn btn-outline-warning" onclick="trainingManager.cancelJob('${job.id}')" 
                                        title="Cancel Job">
                                    <i class="bi bi-stop"></i>
                                </button>
                            ` : ''}
                            <button class="btn btn-outline-danger" onclick="trainingManager.deleteJob('${job.id}')" 
                                    title="Delete Job">
                                <i class="bi bi-trash"></i>
                            </button>
                        </div>
                    </td>
                </tr>
            `;
        }).join('');
    }
    
    updateJobStatistics() {
        const stats = {
            total: this.jobs.length,
            running: this.jobs.filter(job => job.status === 'running').length,
            completed: this.jobs.filter(job => job.status === 'completed').length,
            failed: this.jobs.filter(job => job.status === 'failed').length,
            pending: this.jobs.filter(job => job.status === 'pending').length
        };
        
        // Update statistics display if elements exist
        Object.keys(stats).forEach(key => {
            const element = document.querySelector(`[data-stat="${key}"]`);
            if (element) {
                element.textContent = stats[key];
            }
        });
    }
    
    filterJobs() {
        const searchTerm = document.getElementById('searchInput')?.value.toLowerCase() || '';
        const statusFilter = document.getElementById('statusFilter')?.value || '';
        
        this.filteredJobs = this.jobs.filter(job => {
            const matchesSearch = !searchTerm || 
                job.id.toString().includes(searchTerm) ||
                (job.name && job.name.toLowerCase().includes(searchTerm)) ||
                (job.export_file && job.export_file.toLowerCase().includes(searchTerm));
            
            const matchesStatus = !statusFilter || job.status === statusFilter;
            
            return matchesSearch && matchesStatus;
        });
        
        this.updateTrainingTable();
    }
    
    async submitTraining() {
        try {
            // Validate form first
            if (!this.validateForm()) {
                return;
            }
            
            utils.showLoading();
            
            const exportFileSelect = document.getElementById('exportFile');
            const jobName = document.getElementById('jobName')?.value;
            const modelConfig = document.getElementById('modelConfig')?.value;
            const maxIterations = document.getElementById('maxIterations')?.value;
            const learningRate = document.getElementById('learningRate')?.value;
            const description = document.getElementById('description')?.value;
            
            if (!exportFileSelect?.value) {
                throw new Error('Please select an export file');
            }
            
            const requestData = {
                export_file: exportFileSelect.value,
                model_cfg: {
                    type: modelConfig || "isolation_forest",
                    name: jobName || "Training Job"
                },
                training_config: {
                    max_iterations: parseInt(maxIterations) || 1000,
                    learning_rate: parseFloat(learningRate) || 0.01
                }
            };
            
            if (description) {
                requestData.description = description;
            }
            
            const response = await utils.apiCall('/training/jobs', {
                method: 'POST',
                body: JSON.stringify(requestData)
            });
            
            utils.showSuccess('Training job started successfully');
            
            // Close the modal
            const modal = bootstrap.Modal.getInstance(document.getElementById('newTrainingModal'));
            if (modal) {
                modal.hide();
            }
            
            // Refresh the training jobs list
            this.loadTrainingJobs();
            
        } catch (error) {
            console.error('Error submitting training job:', error);
            utils.showError('Failed to start training job', error);
        } finally {
            utils.hideLoading();
        }
    }
    
    async cancelJob(jobId) {
        try {
            await utils.apiCall(`/training/jobs/${jobId}/cancel`, { method: 'POST' });
            utils.showSuccess('Training job cancelled successfully');
            this.loadTrainingJobs();
        } catch (error) {
            utils.showError('Failed to cancel training job', error);
        }
    }
    
    async deleteJob(jobId) {
        this.currentJobId = jobId;
        const modal = new bootstrap.Modal(document.getElementById('confirmDeleteModal'));
        modal.show();
    }
    
    async confirmDelete() {
        if (!this.currentJobId) return;
        
        try {
            await utils.apiCall(`/training/jobs/${this.currentJobId}`, { method: 'DELETE' });
            utils.showSuccess('Training job deleted successfully');
            
            const modal = bootstrap.Modal.getInstance(document.getElementById('confirmDeleteModal'));
            modal.hide();
            
            this.loadTrainingJobs();
        } catch (error) {
            utils.showError('Failed to delete training job', error);
        } finally {
            this.currentJobId = null;
        }
    }
    
    async viewJobDetails(jobId) {
        try {
            const job = await utils.apiCall(`/training/jobs/${jobId}`);
            this.showJobDetailsModal(job);
        } catch (error) {
            utils.showError('Failed to load job details', error);
        }
    }
    
    showJobDetailsModal(job) {
        const modal = new bootstrap.Modal(document.getElementById('jobDetailsModal'));
        const content = document.getElementById('jobDetailsContent');
        
        // Base job information
        let jobDetailsHtml = `
            <div class="row">
                <div class="col-md-6">
                    <h6>Job Information</h6>
                    <table class="table table-sm">
                        <tr><td>ID:</td><td>${job.id}</td></tr>
                        <tr><td>Name:</td><td>${job.name || 'Unnamed'}</td></tr>
                        <tr><td>Status:</td><td><span class="badge bg-${utils.getStatusColor(job.status)}">${job.status}</span></td></tr>
                        <tr><td>Created:</td><td>${utils.formatDateTime(job.created_at)}</td></tr>
                        <tr><td>Updated:</td><td>${utils.formatDateTime(job.updated_at)}</td></tr>
                    </table>
                </div>
                <div class="col-md-6">
                    <h6>Training Details</h6>
                    <table class="table table-sm">
                        <tr><td>Export File:</td><td>${job.export_file || 'N/A'}</td></tr>
                        <tr><td>Config:</td><td>${job.config || 'default'}</td></tr>
                        <tr><td>Progress:</td><td>${job.progress || 0}%</td></tr>
                        <tr><td>Iterations:</td><td>${job.current_iteration || 0} / ${job.max_iterations || 'N/A'}</td></tr>
                        <tr><td>Duration:</td><td>${this.calculateDuration(job.created_at, job.updated_at)}</td></tr>
                    </table>
                </div>
            </div>
        `;

        // Add description if available
        if (job.description) {
            jobDetailsHtml += `
                <div class="row mt-3">
                    <div class="col-12">
                        <h6>Description</h6>
                        <p>${job.description}</p>
                    </div>
                </div>
            `;
        }

        // Add comprehensive evaluation results if available
        if (job.evaluation_results) {
            const evaluation = job.evaluation_results;
            jobDetailsHtml += `
                <div class="row mt-4">
                    <div class="col-12">
                        <h6><i class="bi bi-graph-up me-2"></i>Model Evaluation Results</h6>
                        <div class="card">
                            <div class="card-body">
                                <div class="row">
                                    <div class="col-md-6">
                                        <h6 class="text-primary">Performance Metrics</h6>
                                        <table class="table table-sm">
                                            <tr><td>Accuracy:</td><td><strong>${(evaluation.accuracy * 100).toFixed(2)}%</strong></td></tr>
                                            <tr><td>Precision:</td><td><strong>${(evaluation.precision * 100).toFixed(2)}%</strong></td></tr>
                                            <tr><td>Recall:</td><td><strong>${(evaluation.recall * 100).toFixed(2)}%</strong></td></tr>
                                            <tr><td>F1 Score:</td><td><strong>${(evaluation.f1_score * 100).toFixed(2)}%</strong></td></tr>
                                            <tr><td>ROC AUC:</td><td><strong>${(evaluation.roc_auc * 100).toFixed(2)}%</strong></td></tr>
                                        </table>
                                    </div>
                                    <div class="col-md-6">
                                        <h6 class="text-primary">Threshold Checks</h6>
                                        <table class="table table-sm">
                                            ${evaluation.threshold_checks ? Object.entries(evaluation.threshold_checks).map(([metric, result]) => `
                                                <tr>
                                                    <td>${metric}:</td>
                                                    <td>
                                                        <span class="badge bg-${result.passed ? 'success' : 'danger'}">
                                                            ${result.passed ? 'PASS' : 'FAIL'}
                                                        </span>
                                                        <small class="text-muted ms-2">(${result.value.toFixed(3)} / ${result.threshold})</small>
                                                    </td>
                                                </tr>
                                            `).join('') : '<tr><td colspan="2">No threshold checks available</td></tr>'}
                                        </table>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            `;
        }

        // Add feature importance if available
        if (job.evaluation_results && job.evaluation_results.feature_importance) {
            const features = job.evaluation_results.feature_importance;
            jobDetailsHtml += `
                <div class="row mt-3">
                    <div class="col-12">
                        <h6><i class="bi bi-bar-chart me-2"></i>Feature Importance</h6>
                        <div class="card">
                            <div class="card-body">
                                <div class="row">
                                    ${Object.entries(features).slice(0, 10).map(([feature, importance]) => `
                                        <div class="col-md-6 mb-2">
                                            <div class="d-flex justify-content-between align-items-center">
                                                <span class="text-truncate" title="${feature}">${feature}</span>
                                                <div class="d-flex align-items-center">
                                                    <div class="progress me-2" style="width: 100px; height: 8px;">
                                                        <div class="progress-bar bg-primary" style="width: ${importance * 100}%"></div>
                                                    </div>
                                                    <small class="text-muted">${(importance * 100).toFixed(1)}%</small>
                                                </div>
                                            </div>
                                        </div>
                                    `).join('')}
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            `;
        }

        // Add recommendations if available
        if (job.evaluation_results && job.evaluation_results.recommendations) {
            const recommendations = job.evaluation_results.recommendations;
            jobDetailsHtml += `
                <div class="row mt-3">
                    <div class="col-12">
                        <h6><i class="bi bi-lightbulb me-2"></i>Recommendations</h6>
                        <div class="card">
                            <div class="card-body">
                                ${recommendations.length > 0 ? `
                                    <ul class="list-group list-group-flush">
                                        ${recommendations.map(rec => `
                                            <li class="list-group-item d-flex align-items-start">
                                                <i class="bi bi-${rec.priority === 'high' ? 'exclamation-triangle text-warning' : 
                                                               rec.priority === 'medium' ? 'info-circle text-info' : 
                                                               'check-circle text-success'} me-2 mt-1"></i>
                                                <div>
                                                    <strong>${rec.category}:</strong> ${rec.message}
                                                    ${rec.suggestion ? `<br><small class="text-muted">Suggestion: ${rec.suggestion}</small>` : ''}
                                                </div>
                                            </li>
                                        `).join('')}
                                    </ul>
                                ` : '<p class="text-muted">No specific recommendations available.</p>'}
                            </div>
                        </div>
                    </div>
                </div>
            `;
        }

        // Add training report if available
        if (job.training_report) {
            const report = job.training_report;
            jobDetailsHtml += `
                <div class="row mt-3">
                    <div class="col-12">
                        <h6><i class="bi bi-file-text me-2"></i>Training Report</h6>
                        <div class="card">
                            <div class="card-body">
                                <div class="row">
                                    <div class="col-md-6">
                                        <h6 class="text-primary">Training Summary</h6>
                                        <table class="table table-sm">
                                            <tr><td>Total Iterations:</td><td>${report.total_iterations || 'N/A'}</td></tr>
                                            <tr><td>Final Loss:</td><td>${report.final_loss ? report.final_loss.toFixed(4) : 'N/A'}</td></tr>
                                            <tr><td>Training Time:</td><td>${report.training_time || 'N/A'}</td></tr>
                                            <tr><td>Model Size:</td><td>${report.model_size || 'N/A'}</td></tr>
                                        </table>
                                    </div>
                                    <div class="col-md-6">
                                        <h6 class="text-primary">Data Summary</h6>
                                        <table class="table table-sm">
                                            <tr><td>Training Samples:</td><td>${report.training_samples || 'N/A'}</td></tr>
                                            <tr><td>Validation Samples:</td><td>${report.validation_samples || 'N/A'}</td></tr>
                                            <tr><td>Test Samples:</td><td>${report.test_samples || 'N/A'}</td></tr>
                                            <tr><td>Features:</td><td>${report.num_features || 'N/A'}</td></tr>
                                        </table>
                                    </div>
                                </div>
                                ${report.convergence_info ? `
                                    <div class="mt-3">
                                        <h6 class="text-primary">Convergence Information</h6>
                                        <p class="mb-2">${report.convergence_info}</p>
                                    </div>
                                ` : ''}
                            </div>
                        </div>
                    </div>
                </div>
            `;
        }

        // Add legacy metrics if available (fallback)
        if (job.metrics && !job.evaluation_results) {
            jobDetailsHtml += `
                <div class="row mt-3">
                    <div class="col-12">
                        <h6>Training Metrics</h6>
                        <pre class="bg-light p-3 rounded">${JSON.stringify(job.metrics, null, 2)}</pre>
                    </div>
                </div>
            `;
        }

        content.innerHTML = jobDetailsHtml;
        
        // Show/hide download button based on job status
        const downloadBtn = document.getElementById('downloadResultsBtn');
        if (downloadBtn) {
            downloadBtn.style.display = job.status === 'completed' ? 'inline-block' : 'none';
            downloadBtn.onclick = () => this.downloadResults(job.id);
        }
        
        modal.show();
    }
    
    async downloadResults(jobId) {
        try {
            const response = await fetch(`${API_BASE}/training/jobs/${jobId}/results`);
            if (response.ok) {
                const blob = await response.blob();
                const url = window.URL.createObjectURL(blob);
                utils.downloadFile(url, `training-results-${jobId}.json`);
            } else {
                throw new Error('Failed to download results');
            }
        } catch (error) {
            utils.showError('Failed to download results', error);
        }
    }
    
    calculateDuration(startTime, endTime) {
        const start = new Date(startTime);
        const end = new Date(endTime || Date.now());
        const duration = Math.floor((end - start) / 1000);
        return utils.formatDuration(duration);
    }
    
    handleExportFileSelect() {
        const exportFile = document.getElementById('exportFile');
        if (exportFile && exportFile.value) {
            console.log('Export file selected:', exportFile.value);
            // Enable submit button when file is selected
            const submitBtn = document.querySelector('#newTrainingModal .btn-primary');
            if (submitBtn) {
                submitBtn.disabled = false;
            }
        }
    }
    
    resetForm() {
        const form = document.getElementById('trainingForm');
        if (form) {
            form.reset();
        }
        
        // Reset export file dropdown
        const exportFile = document.getElementById('exportFile');
        if (exportFile) {
            exportFile.innerHTML = '<option value="">Select an export file...</option>';
        }
        
        // Disable submit button initially
        const submitBtn = document.querySelector('#newTrainingModal .btn-primary');
        if (submitBtn) {
            submitBtn.disabled = true;
        }
    }
    
    startAutoRefresh() {
        // Refresh every 10 seconds for active jobs
        this.updateInterval = setInterval(() => {
            this.loadTrainingJobs();
        }, 10000);
    }
    
    stopAutoRefresh() {
        if (this.updateInterval) {
            clearInterval(this.updateInterval);
            this.updateInterval = null;
        }
    }
    
    // Public methods
    refresh() {
        this.loadTrainingJobs();
    }
    
    exportJobs() {
        const data = this.filteredJobs.map(job => ({
            id: job.id,
            name: job.name,
            status: job.status,
            export_file: job.export_file,
            created_at: job.created_at,
            progress: job.progress
        }));
        
        const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
        const url = window.URL.createObjectURL(blob);
        utils.downloadFile(url, 'training-jobs.json');
    }
    
    async loadExportFiles() {
        console.log('TrainingManager.loadExportFiles called');
        try {
            const exports = await utils.apiCall('/training/exports');
            console.log('Export files loaded:', exports);
            const select = document.getElementById('exportFile');
            if (!select) {
                console.warn('Export file select element not found');
                return;
            }
            select.innerHTML = '<option value="">Select an export file...</option>';
            exports.forEach(exportFile => {
                const option = document.createElement('option');
                option.value = exportFile.path;
                option.textContent = `${exportFile.filename} (${utils.formatFileSize(exportFile.size)})`;
                select.appendChild(option);
            });
            console.log('Export file dropdown updated with', exports.length, 'files');
        } catch (error) {
            console.error('Error loading export files:', error);
            utils.showError('Failed to load export files', error);
        }
    }
    
    openTrainingModal() {
        console.log('TrainingManager.openTrainingModal called');
        
        // Check if Bootstrap is available
        if (typeof bootstrap === 'undefined') {
            console.error('Bootstrap is not loaded!');
            utils.showError('Bootstrap is not loaded. Please refresh the page.');
            return;
        }
        
        // Check if modal element exists
        const modalElement = document.getElementById('newTrainingModal');
        if (!modalElement) {
            console.error('Modal element not found!');
            utils.showError('Modal element not found. Please refresh the page.');
            return;
        }
        
        console.log('Modal element found:', modalElement);
        
        try {
            const modal = new bootstrap.Modal(modalElement);
            console.log('Bootstrap modal created:', modal);
            modal.show();
            console.log('Modal shown successfully');
            
            // Load export files after a short delay to ensure modal is rendered
            setTimeout(() => {
                console.log('Loading export files...');
                this.loadExportFiles();
            }, 100);
        } catch (error) {
            console.error('Error showing modal:', error);
            utils.showError('Failed to show modal: ' + error.message);
        }
    }
    
    validateForm() {
        const jobName = document.getElementById('jobName')?.value;
        const modelConfig = document.getElementById('modelConfig')?.value;
        const exportFile = document.getElementById('exportFile')?.value;
        
        const errors = [];
        
        if (!jobName?.trim()) {
            errors.push('Job name is required');
        }
        
        if (!modelConfig) {
            errors.push('Model configuration is required');
        }
        
        if (!exportFile) {
            errors.push('Export file is required');
        }
        
        if (errors.length > 0) {
            utils.showError('Please fix the following errors:\n' + errors.join('\n'));
            return false;
        }
        
        return true;
    }
    
    destroy() {
        this.stopAutoRefresh();
    }
}

// Initialize training manager when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    console.log('DOM loaded, initializing TrainingManager...');
    
    // Check if we're on the training page
    const trainingTable = document.getElementById('trainingTable');
    if (!trainingTable) {
        console.log('Not on training page, skipping TrainingManager initialization');
        return;
    }
    
    try {
        window.trainingManager = new TrainingManager();
        console.log('TrainingManager initialized successfully:', window.trainingManager);
        
        // Add fallback global functions for backward compatibility
        window.openTrainingModal = () => {
            console.log('Global openTrainingModal called (fallback)');
            if (window.trainingManager) {
                window.trainingManager.openTrainingModal();
            } else {
                console.error('TrainingManager not available');
            }
        };
        
        window.submitTraining = () => {
            console.log('Global submitTraining called (fallback)');
            if (window.trainingManager) {
                window.trainingManager.submitTraining();
            } else {
                console.error('TrainingManager not available');
            }
        };
        
        // Test modal functionality
        console.log('Testing modal functionality...');
        const modalElement = document.getElementById('newTrainingModal');
        if (modalElement) {
            console.log('Modal element found and ready');
        } else {
            console.error('Modal element not found during initialization');
        }
        
        // Test export file loading
        console.log('Testing export file loading...');
        window.trainingManager.loadExportFiles().then(() => {
            console.log('Export files loaded successfully during initialization');
        }).catch(error => {
            console.error('Failed to load export files during initialization:', error);
        });
        
    } catch (error) {
        console.error('Failed to initialize TrainingManager:', error);
        utils.showError('Failed to initialize training manager: ' + error.message);
    }
});

console.log('=== TRAINING.JS SCRIPT COMPLETED ==='); 