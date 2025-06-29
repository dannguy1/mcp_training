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
        console.log('TrainingManager constructor called');
        this.jobs = [];
        this.filteredJobs = [];
        this.currentJobId = null;
        this.updateInterval = null;
        this.lastRefreshTime = null;
        this.init();
    }
    
    init() {
        console.log('TrainingManager init called');
        this.setupEventListeners();
        this.setupRefreshButton();
        this.loadTrainingJobs();
        // Auto-refresh disabled for training optimization
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
        const exportFiles = document.getElementById('exportFiles');
        if (exportFiles) {
            exportFiles.addEventListener('change', () => {
                this.handleExportFilesSelect();
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
            newTrainingModal.addEventListener('show.bs.modal', () => {
                this.loadExportFiles();
            });
            
            newTrainingModal.addEventListener('hidden.bs.modal', () => {
                this.resetTrainingForm();
            });
        }
        
        const uploadExportModal = document.getElementById('uploadExportModal');
        if (uploadExportModal) {
            uploadExportModal.addEventListener('show.bs.modal', () => {
                console.log('Upload modal opening, re-attaching event listeners...');
                this.attachUploadEventListeners();
            });
            
            uploadExportModal.addEventListener('hidden.bs.modal', () => {
                this.resetUploadForm();
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
            const jobsResponse = await utils.apiCall('/api/training/jobs');
            const jobs = jobsResponse.trainings || jobsResponse;
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
            // Get the job ID - try multiple possible field names
            const jobId = job.id || job.training_id || job.job_id || 'unknown';
            
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
                <tr data-job-id="${jobId}">
                    <td>${jobId}</td>
                    <td>
                        <div>
                            <strong>${job.model_name || job.name || 'Unnamed Job'}</strong>
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
                            <button class="btn btn-outline-info" onclick="trainingManager.viewJobDetails('${jobId}')" 
                                    title="View Details">
                                <i class="bi bi-eye"></i>
                            </button>
                            ${job.status === 'running' ? `
                                <button class="btn btn-outline-warning" onclick="trainingManager.cancelJob('${jobId}')" 
                                        title="Cancel Job">
                                    <i class="bi bi-stop"></i>
                                </button>
                            ` : ''}
                            <button class="btn btn-outline-danger" onclick="trainingManager.deleteJob('${jobId}')" 
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
            // Get job ID safely
            const jobId = job.id || job.training_id || job.job_id || '';
            
            const matchesSearch = !searchTerm || 
                jobId.toString().includes(searchTerm) ||
                (job.name && job.name.toLowerCase().includes(searchTerm)) ||
                (job.model_name && job.model_name.toLowerCase().includes(searchTerm)) ||
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
            
            const exportFilesSelect = document.getElementById('exportFiles');
            const jobName = document.getElementById('jobName')?.value;
            const modelConfig = document.getElementById('modelConfig')?.value;
            const maxIterations = document.getElementById('maxIterations')?.value;
            const learningRate = document.getElementById('learningRate')?.value;
            const description = document.getElementById('description')?.value;
            
            // Get all selected export files
            const selectedFiles = Array.from(exportFilesSelect.selectedOptions).map(option => option.value);
            
            if (selectedFiles.length === 0) {
                throw new Error('Please select at least one export file');
            }
            
            const requestData = {
                export_files: selectedFiles,
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
            
            const response = await utils.apiCall('/api/training/jobs', {
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
            utils.showError('Failed to start training job', error);
        } finally {
            utils.hideLoading();
        }
    }
    
    async cancelJob(jobId) {
        try {
            await utils.apiCall(`/api/training/jobs/${jobId}/cancel`, { method: 'POST' });
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
            await utils.apiCall(`/api/training/jobs/${this.currentJobId}`, { method: 'DELETE' });
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
            console.log('Fetching job details for:', jobId);
            const response = await fetch(`/api/training/jobs/${jobId}`);
            const job = await response.json();
            console.log('Job details response:', job);
            
            if (response.ok) {
                this.showJobDetailsModal(job);
            } else {
                utils.showError('Failed to load job details', job.error || 'Unknown error');
            }
        } catch (error) {
            console.error('Error fetching job details:', error);
            utils.showError('Failed to load job details', error);
        }
    }
    
    showJobDetailsModal(job) {
        console.log('Showing job details for:', job);
        console.log('Comprehensive stats:', job.comprehensive_stats);
        
        const modal = new bootstrap.Modal(document.getElementById('jobDetailsModal'));
        const content = document.getElementById('jobDetailsContent');
        
        // Get job ID safely
        const jobId = job.id || job.training_id || job.job_id || 'unknown';
        const jobName = job.name || job.model_name || 'Unnamed Job';
        
        // Base job information
        let jobDetailsHtml = `
            <div class="row">
                <div class="col-md-6">
                    <h6>Job Information</h6>
                    <table class="table table-sm">
                        <tr><td>ID:</td><td>${jobId}</td></tr>
                        <tr><td>Name:</td><td>${jobName}</td></tr>
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

        // Add comprehensive statistics if available
        if (job.comprehensive_stats) {
            console.log('Processing comprehensive stats:', job.comprehensive_stats);
            const stats = job.comprehensive_stats;
            const trainingInfo = stats.training_info;
            const evaluationSummary = stats.evaluation_summary;
            const performanceMetrics = stats.performance_metrics;
            
            console.log('Training info:', trainingInfo);
            console.log('Evaluation summary:', evaluationSummary);
            console.log('Performance metrics:', performanceMetrics);
            
            jobDetailsHtml += `
                <div class="row mt-4">
                    <div class="col-12">
                        <h6><i class="bi bi-graph-up me-2"></i>Comprehensive Training Statistics</h6>
                        <div class="card">
                            <div class="card-body">
                                <div class="row">
                                    <div class="col-md-6">
                                        <h6 class="text-primary">Training Information</h6>
                                        <table class="table table-sm">
                                            <tr><td>Training Samples:</td><td><strong>${trainingInfo.samples.toLocaleString()}</strong></td></tr>
                                            <tr><td>Features:</td><td><strong>${trainingInfo.features}</strong></td></tr>
                                            <tr><td>Training Duration:</td><td><strong>${trainingInfo.duration_seconds.toFixed(2)}s</strong></td></tr>
                                            <tr><td>Export File Size:</td><td><strong>${trainingInfo.export_size_mb} MB</strong></td></tr>
                                            <tr><td>Model Type:</td><td><strong>${job.model_type}</strong></td></tr>
                                        </table>
                                    </div>
                                    <div class="col-md-6">
                                        <h6 class="text-primary">Model Parameters</h6>
                                        <table class="table table-sm">
                                            ${Object.entries(trainingInfo.model_parameters).map(([param, value]) => `
                                                <tr><td>${param}:</td><td><strong>${value}</strong></td></tr>
                                            `).join('')}
                                        </table>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            `;
            
            // Add performance metrics
            if (performanceMetrics) {
                jobDetailsHtml += `
                    <div class="row mt-3">
                        <div class="col-12">
                            <h6><i class="bi bi-speedometer2 me-2"></i>Performance Metrics</h6>
                            <div class="card">
                                <div class="card-body">
                                    <div class="row">
                                        <div class="col-md-6">
                                            <h6 class="text-primary">Score Statistics</h6>
                                            <table class="table table-sm">
                                                <tr><td>Score Mean:</td><td><strong>${performanceMetrics.score_mean.toFixed(4)}</strong></td></tr>
                                                <tr><td>Score Std:</td><td><strong>${performanceMetrics.score_std.toFixed(4)}</strong></td></tr>
                                                <tr><td>Score Min:</td><td><strong>${performanceMetrics.score_min.toFixed(4)}</strong></td></tr>
                                                <tr><td>Score Max:</td><td><strong>${performanceMetrics.score_max.toFixed(4)}</strong></td></tr>
                                            </table>
                                        </div>
                                        <div class="col-md-6">
                                            <h6 class="text-primary">Anomaly Detection</h6>
                                            <table class="table table-sm">
                                                <tr><td>Anomaly Ratio:</td><td><strong>${(performanceMetrics.anomaly_ratio * 100).toFixed(2)}%</strong></td></tr>
                                                <tr><td>Threshold Value:</td><td><strong>${performanceMetrics.threshold_value.toFixed(4)}</strong></td></tr>
                                            </table>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                `;
            }
            
            // Add evaluation summary
            if (evaluationSummary) {
                jobDetailsHtml += `
                    <div class="row mt-3">
                        <div class="col-12">
                            <h6><i class="bi bi-check-circle me-2"></i>Evaluation Summary</h6>
                            <div class="card">
                                <div class="card-body">
                                    <div class="row">
                                        <div class="col-md-6">
                                            <h6 class="text-primary">Overall Performance</h6>
                                            <table class="table table-sm">
                                                <tr><td>Threshold Pass Rate:</td><td><strong>${(evaluationSummary.overall_performance.threshold_pass_rate * 100).toFixed(0)}%</strong></td></tr>
                                                <tr><td>Passed Thresholds:</td><td><strong>${evaluationSummary.overall_performance.passed_thresholds}/${evaluationSummary.overall_performance.total_thresholds}</strong></td></tr>
                                            </table>
                                        </div>
                                        <div class="col-md-6">
                                            <h6 class="text-primary">Best/Worst Metrics</h6>
                                            <table class="table table-sm">
                                                <tr><td>Best Metric:</td><td><strong>${evaluationSummary.best_metrics.best_metric[0]} (${evaluationSummary.best_metrics.best_metric[1].toFixed(4)})</strong></td></tr>
                                                <tr><td>Worst Metric:</td><td><strong>${evaluationSummary.best_metrics.worst_metric[0]} (${evaluationSummary.best_metrics.worst_metric[1].toFixed(4)})</strong></td></tr>
                                            </table>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                `;
                
                // Add recommendations
                if (evaluationSummary.recommendations && evaluationSummary.recommendations.length > 0) {
                    jobDetailsHtml += `
                        <div class="row mt-3">
                            <div class="col-12">
                                <h6><i class="bi bi-lightbulb me-2"></i>Recommendations</h6>
                                <div class="card">
                                    <div class="card-body">
                                        <ul class="list-group list-group-flush">
                                            ${evaluationSummary.recommendations.map(rec => `
                                                <li class="list-group-item d-flex align-items-start">
                                                    <i class="bi bi-info-circle text-info me-2 mt-1"></i>
                                                    <div>${rec}</div>
                                                </li>
                                            `).join('')}
                                        </ul>
                                    </div>
                                </div>
                            </div>
                        </div>
                    `;
                }
            }
            
            // Add feature names
            if (trainingInfo.feature_names && trainingInfo.feature_names.length > 0) {
                jobDetailsHtml += `
                    <div class="row mt-3">
                        <div class="col-12">
                            <h6><i class="bi bi-list-ul me-2"></i>Feature Names (${trainingInfo.feature_names.length} total)</h6>
                            <div class="card">
                                <div class="card-body">
                                    <div class="row">
                                        ${trainingInfo.feature_names.map((feature, index) => `
                                            <div class="col-md-4 mb-1">
                                                <small class="text-muted">${index + 1}. ${feature}</small>
                                            </div>
                                        `).join('')}
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                `;
            }
        }

        // Display export file information
        if (job.export_files && job.export_files.length > 0) {
            const exportFilesHtml = job.export_files.map(file => {
                const fileName = file.split('/').pop(); // Get just the filename
                return `<li><code>${fileName}</code></li>`;
            }).join('');
            
            jobDetailsHtml += `
                <div class="row mt-3">
                    <div class="col-12">
                        <h6><i class="bi bi-file-earmark-text me-2"></i>Export Files (${job.export_files.length})</h6>
                        <ul class="list-unstyled">
                            ${exportFilesHtml}
                        </ul>
                    </div>
                </div>
            `;
        }

        content.innerHTML = jobDetailsHtml;
        console.log('Final job details HTML length:', jobDetailsHtml.length);
        console.log('Modal content element:', content);
        
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
            const response = await fetch(`${API_BASE}/api/training/jobs/${jobId}/results`);
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
    
    handleExportFilesSelect() {
        const exportFiles = document.getElementById('exportFiles');
        if (exportFiles && exportFiles.selectedOptions.length > 0) {
            const selectedCount = exportFiles.selectedOptions.length;
            console.log(`Export files selected: ${selectedCount} files`);
            // Enable submit button when files are selected
            const submitBtn = document.querySelector('#newTrainingModal .btn-primary');
            if (submitBtn) {
                submitBtn.disabled = false;
            }
        }
    }
    
    resetTrainingForm() {
        // Reset form fields
        const jobName = document.getElementById('jobName');
        const modelConfig = document.getElementById('modelConfig');
        const maxIterations = document.getElementById('maxIterations');
        const learningRate = document.getElementById('learningRate');
        const description = document.getElementById('description');
        
        if (jobName) jobName.value = '';
        if (modelConfig) modelConfig.value = '';
        if (maxIterations) maxIterations.value = '1000';
        if (learningRate) learningRate.value = '0.01';
        if (description) description.value = '';
        
        // Reset export files dropdown
        const exportFiles = document.getElementById('exportFiles');
        if (exportFiles) {
            exportFiles.innerHTML = '<option value="">Select export files...</option>';
        }
        
        // Disable submit button
        const submitBtn = document.querySelector('#newTrainingModal .btn-primary');
        if (submitBtn) {
            submitBtn.disabled = true;
        }
    }
    
    startAutoRefresh() {
        // Disable auto-refresh completely - rely on WebSocket push updates and manual refresh
        // This eliminates wasteful polling and reduces system load during training
        console.log('Auto-refresh disabled for training optimization');
    }
    
    stopAutoRefresh() {
        // No auto-refresh to stop
        if (this.updateInterval) {
            clearInterval(this.updateInterval);
            this.updateInterval = null;
        }
    }
    
    // Manual refresh method for active training jobs
    async refreshActiveJobs() {
        const activeJobs = this.jobs.filter(job => 
            job.status === 'running' || job.status === 'pending'
        );
        
        if (activeJobs.length > 0) {
            console.log(`Manual refresh for ${activeJobs.length} active jobs`);
            await this.loadTrainingJobs();
        }
    }
    
    // Add refresh button functionality
    setupRefreshButton() {
        const refreshBtn = document.getElementById('refreshJobsBtn');
        if (refreshBtn) {
            refreshBtn.addEventListener('click', async () => {
                refreshBtn.disabled = true;
                refreshBtn.innerHTML = '<i class="bi bi-arrow-clockwise spin"></i>';
                
                try {
                    await this.loadTrainingJobs();
                    utils.showSuccess('Training jobs refreshed');
                } catch (error) {
                    utils.showError('Failed to refresh training jobs', error);
                } finally {
                    refreshBtn.disabled = false;
                    refreshBtn.innerHTML = '<i class="bi bi-arrow-clockwise"></i>';
                }
            });
        }
    }
    
    // Public methods
    refresh() {
        this.loadTrainingJobs();
    }
    
    exportJobs() {
        const data = this.filteredJobs.map(job => {
            const jobId = job.id || job.training_id || job.job_id || 'unknown';
            return {
                id: jobId,
                name: job.name || job.model_name || 'Unnamed Job',
                status: job.status,
                export_file: job.export_file,
                created_at: job.created_at,
                progress: job.progress
            };
        });
        
        const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
        const url = window.URL.createObjectURL(blob);
        utils.downloadFile(url, 'training-jobs.json');
    }
    
    async loadExportFiles() {
        console.log('TrainingManager.loadExportFiles called');
        try {
            const exports = await utils.apiCall('/api/training/exports');
            console.log('Export files loaded:', exports);
            const select = document.getElementById('exportFiles');
            if (!select) {
                console.warn('Export files select element not found');
                return;
            }
            select.innerHTML = '<option value="">Select export files...</option>';
            exports.forEach(exportFile => {
                const option = document.createElement('option');
                option.value = exportFile.path;
                option.textContent = `${exportFile.filename} (${utils.formatFileSize(exportFile.size)})`;
                select.appendChild(option);
            });
            console.log('Export files dropdown updated with', exports.length, 'files');
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
        const exportFiles = document.getElementById('exportFiles');
        
        const errors = [];
        
        if (!jobName?.trim()) {
            errors.push('Job name is required');
        }
        
        if (!modelConfig) {
            errors.push('Model configuration is required');
        }
        
        if (!exportFiles || exportFiles.selectedOptions.length === 0) {
            errors.push('At least one export file is required');
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
    
    handleDragOver(e) {
        e.preventDefault();
        e.currentTarget.classList.add('dragover');
    }
    
    handleFileDrop(e) {
        e.preventDefault();
        e.currentTarget.classList.remove('dragover');
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            this.handleFile(files[0]);
        }
    }
    
    handleFileSelect(e) {
        const file = e.target.files[0];
        if (file) {
            this.handleFile(file);
        }
    }
    
    handleFile(file) {
        console.log('handleFile called with:', file.name, file.size);
        
        // Validate file type
        if (!file.name.endsWith('.json')) {
            utils.showError('Please select a valid JSON file');
            return;
        }
        
        // Store the file reference in the class instance
        this.currentUploadFile = file;
        console.log('File stored in instance:', this.currentUploadFile?.name);
        
        // Update file info display
        const uploadArea = document.getElementById('trainingFileUploadArea');
        if (uploadArea) {
            uploadArea.innerHTML = `
                <div class="alert alert-info">
                    <i class="bi bi-file-earmark-text me-2"></i>
                    <strong>${file.name}</strong>
                    <small class="text-muted ms-2">(${utils.formatFileSize(file.size)})</small>
                </div>
            `;
        }
        
        // Enable upload button
        const uploadBtn = document.getElementById('trainingUploadBtn');
        if (uploadBtn) {
            uploadBtn.disabled = false;
        }
        
        // Also try to set the file on the file input element
        const fileInput = document.querySelector('input[type="file"][id="trainingUploadFile"]');
        if (fileInput) {
            try {
                const dataTransfer = new DataTransfer();
                dataTransfer.items.add(file);
                fileInput.files = dataTransfer.files;
                console.log('File set on input element:', fileInput.files[0]?.name);
            } catch (error) {
                console.log('Could not set file on input element:', error.message);
            }
        }
    }
    
    async uploadFile() {
        try {
            console.log('uploadFile method called');
            console.log('utils object available:', typeof utils !== 'undefined');
            console.log('utils.apiCall available:', typeof utils?.apiCall === 'function');
            
            if (typeof utils === 'undefined') {
                throw new Error('Utils object is not available');
            }
            
            // Check if we have a stored file reference
            if (this.currentUploadFile) {
                console.log('Using stored file reference:', this.currentUploadFile.name);
                const file = this.currentUploadFile;
                
                console.log('File to upload:', file.name, file.size);
                
                // Validate file type
                if (!file.name.endsWith('.json')) {
                    utils.showError('Please select a valid JSON file');
                    return;
                }
                
                utils.showLoading();
                
                // Show upload progress
                const progressContainer = document.getElementById('trainingUploadProgress');
                if (progressContainer) {
                    progressContainer.style.display = 'block';
                }
                
                const formData = new FormData();
                formData.append('file', file);
                
                console.log('Making API call to upload file...');
                const response = await utils.apiCall('/api/training/exports/upload', {
                    method: 'POST',
                    body: formData,
                    headers: {} // Let browser set content-type for FormData
                });
                
                console.log('Upload response:', response);
                utils.showSuccess('Export file uploaded successfully');
                
                // Close modal
                const modal = bootstrap.Modal.getInstance(document.getElementById('uploadExportModal'));
                if (modal) {
                    modal.hide();
                }
                
                // Refresh export files list in training modal
                this.loadExportFiles();
                
                // Refresh training jobs to show any new jobs that might use this file
                this.loadTrainingJobs();
                
            } else {
                utils.showError('Please select a file to upload');
            }
            
        } catch (error) {
            console.error('Upload error details:', error);
            console.error('Error stack:', error.stack);
            
            if (typeof utils !== 'undefined') {
                utils.showError('Failed to upload file', error);
            } else {
                // Fallback error handling if utils is not available
                alert(`Failed to upload file: ${error.message}`);
            }
        } finally {
            if (typeof utils !== 'undefined') {
                utils.hideLoading();
            }
            
            // Hide progress
            const progressContainer = document.getElementById('trainingUploadProgress');
            if (progressContainer) {
                progressContainer.style.display = 'none';
            }
        }
    }
    
    resetUploadForm() {
        console.log('resetUploadForm called');
        
        const uploadArea = document.getElementById('trainingFileUploadArea');
        console.log('Upload area found:', !!uploadArea);
        
        if (uploadArea) {
            // Store the file input element
            const fileInput = uploadArea.querySelector('input[type="file"]');
            console.log('Existing file input found:', !!fileInput);
            
            // Update the display content
            uploadArea.innerHTML = `
                <i class="bi bi-cloud-upload fs-1 text-muted mb-3"></i>
                <h6>Drag and drop your export file here</h6>
                <p class="text-muted">or click to browse</p>
            `;
            
            // Re-add the file input element
            if (fileInput) {
                fileInput.value = '';
                uploadArea.appendChild(fileInput);
                console.log('Re-added existing file input');
            } else {
                // Create a new file input if none exists
                const newFileInput = document.createElement('input');
                newFileInput.type = 'file';
                newFileInput.id = 'trainingUploadFile';
                newFileInput.accept = '.json';
                newFileInput.style.display = 'none';
                newFileInput.style.position = 'absolute';
                newFileInput.style.opacity = '0';
                newFileInput.style.pointerEvents = 'none';
                newFileInput.style.zIndex = '-1';
                uploadArea.appendChild(newFileInput);
                console.log('Created new file input element');
            }
            
            // Verify the file input exists after reset
            const verifyFileInput = document.getElementById('trainingUploadFile');
            console.log('File input verification after reset:', !!verifyFileInput);
        }
        
        const uploadBtn = document.getElementById('trainingUploadBtn');
        if (uploadBtn) {
            uploadBtn.disabled = true;
        }
    }
    
    attachUploadEventListeners() {
        console.log('Attaching upload event listeners...');
        
        const uploadArea = document.getElementById('trainingFileUploadArea');
        const uploadFile = document.getElementById('trainingUploadFile');
        const uploadBtn = document.getElementById('trainingUploadBtn');
        
        console.log('Upload elements found:', {
            area: !!uploadArea,
            file: !!uploadFile,
            btn: !!uploadBtn
        });
        
        if (uploadArea && uploadFile) {
            // Remove any existing listeners by cloning and replacing the upload area
            console.log('Removing existing listeners by cloning upload area...');
            
            // Clone upload area to remove all listeners
            const newUploadArea = uploadArea.cloneNode(true);
            uploadArea.parentNode.replaceChild(newUploadArea, uploadArea);
            
            // Get the file input from the cloned area
            const newUploadFile = newUploadArea.querySelector('#trainingUploadFile');
            
            console.log('New elements after cloning:', {
                area: !!newUploadArea,
                file: !!newUploadFile
            });
            
            if (newUploadFile) {
                // Store reference to the file input
                this.currentFileInput = newUploadFile;
                
                // Create simple click handler
                this.uploadAreaClickHandler = (e) => {
                    console.log('Upload area clicked!');
                    e.preventDefault();
                    e.stopPropagation();
                    
                    console.log('Triggering file input click');
                    this.currentFileInput.click();
                };
                
                // Create drag and drop handlers
                this.uploadAreaDragOverHandler = (e) => {
                    e.preventDefault();
                    e.currentTarget.classList.add('dragover');
                };
                
                this.uploadAreaDropHandler = (e) => {
                    e.preventDefault();
                    e.currentTarget.classList.remove('dragover');
                    
                    const files = e.dataTransfer.files;
                    if (files.length > 0) {
                        console.log('File dropped:', files[0].name);
                        this.handleFile(files[0]);
                    }
                };
                
                // Create file change handler
                this.uploadFileChangeHandler = (e) => {
                    console.log('File input change event');
                    const file = e.target.files[0];
                    if (file) {
                        console.log('File selected via input:', file.name);
                        this.handleFile(file);
                    }
                };
                
                // Attach event listeners
                newUploadArea.addEventListener('click', this.uploadAreaClickHandler);
                newUploadArea.addEventListener('dragover', this.uploadAreaDragOverHandler);
                newUploadArea.addEventListener('drop', this.uploadAreaDropHandler);
                newUploadFile.addEventListener('change', this.uploadFileChangeHandler);
                
                console.log('Upload event listeners attached successfully');
            } else {
                console.error('File input not found in cloned upload area');
            }
        } else {
            console.error('Upload area or file input not found during attachment');
        }
        
        if (uploadBtn) {
            if (this.uploadBtnClickHandler) {
                uploadBtn.removeEventListener('click', this.uploadBtnClickHandler);
            }
            this.uploadBtnClickHandler = () => {
                console.log('Upload button clicked!');
                this.uploadFile();
            };
            uploadBtn.addEventListener('click', this.uploadBtnClickHandler);
        }
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
    
    // Test if upload modal elements exist
    console.log('Testing upload modal elements...');
    const uploadModal = document.getElementById('uploadExportModal');
    const uploadArea = document.getElementById('trainingFileUploadArea');
    const uploadFile = document.getElementById('trainingUploadFile');
    const uploadBtn = document.getElementById('trainingUploadBtn');
    
    console.log('Upload modal elements found:', {
        modal: !!uploadModal,
        area: !!uploadArea,
        file: !!uploadFile,
        btn: !!uploadBtn
    });
    
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
        
        // Test upload modal functionality
        console.log('Testing upload modal functionality...');
        const uploadModalElement = document.getElementById('uploadExportModal');
        const uploadButton = document.querySelector('[data-bs-target="#uploadExportModal"]');
        if (uploadModalElement) {
            console.log('Upload modal element found and ready');
        } else {
            console.error('Upload modal element not found during initialization');
        }
        if (uploadButton) {
            console.log('Upload button found:', uploadButton);
            // Test if Bootstrap modal is working
            uploadButton.addEventListener('click', () => {
                console.log('Upload button clicked via event listener');
            });
        } else {
            console.error('Upload button not found during initialization');
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