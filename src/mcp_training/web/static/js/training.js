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
                    <td colspan="7" class="text-center text-muted py-4">
                        <i class="bi bi-gear fs-1 mb-3"></i>
                        <p>No training jobs found</p>
                    </td>
                </tr>
            `;
            return;
        }
        
        tableBody.innerHTML = this.filteredJobs.map(job => `
            <tr data-job-id="${job.id}">
                <td>${job.id}</td>
                <td>${job.model_name || 'Unnamed Job'}</td>
                <td>
                    <span class="badge bg-${utils.getStatusColor(job.status)}">
                        ${job.status}
                    </span>
                </td>
                <td>${job.export_file || 'N/A'}</td>
                <td>${utils.formatDateTime(job.created_at)}</td>
                <td>
                    <div class="progress" style="height: 6px;">
                        <div class="progress-bar bg-${utils.getStatusColor(job.status)}" 
                             style="width: ${job.progress || 0}%"></div>
                    </div>
                    <small class="text-muted">${job.progress || 0}%</small>
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
        `).join('');
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
        
        content.innerHTML = `
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
            ${job.description ? `
                <div class="row mt-3">
                    <div class="col-12">
                        <h6>Description</h6>
                        <p>${job.description}</p>
                    </div>
                </div>
            ` : ''}
            ${job.metrics ? `
                <div class="row mt-3">
                    <div class="col-12">
                        <h6>Training Metrics</h6>
                        <pre class="bg-light p-3 rounded">${JSON.stringify(job.metrics, null, 2)}</pre>
                    </div>
                </div>
            ` : ''}
        `;
        
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
        const select = document.getElementById('exportFile');
        const submitBtn = document.querySelector('#newTrainingModal .btn-primary');
        
        if (select && submitBtn) {
            submitBtn.disabled = !select.value;
        }
    }
    
    resetForm() {
        const form = document.getElementById('trainingForm');
        if (form) {
            form.reset();
        }
        
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
    
    destroy() {
        this.stopAutoRefresh();
    }
}

// Global functions for use in HTML
window.openTrainingModal = () => {
    console.log('openTrainingModal called');
    const modal = new bootstrap.Modal(document.getElementById('newTrainingModal'));
    modal.show();
    // Load export files after a short delay to ensure modal is rendered
    setTimeout(() => {
        if (window.trainingManager) {
            window.trainingManager.loadExportFiles();
        } else {
            console.warn('trainingManager not available yet');
        }
    }, 100);
};

window.submitTraining = () => {
    if (window.trainingManager) {
        window.trainingManager.submitTraining();
    } else {
        console.warn('trainingManager not available yet');
    }
};

// Function to load export files and populate the dropdown
window.loadExportFiles = async function() {
    console.log('Attempting to load export files...');
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
};

// Initialize training manager when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    console.log('DOM loaded, initializing TrainingManager...');
    window.trainingManager = new TrainingManager();
    console.log('TrainingManager initialized:', window.trainingManager);
});

console.log('=== TRAINING.JS SCRIPT COMPLETED ==='); 