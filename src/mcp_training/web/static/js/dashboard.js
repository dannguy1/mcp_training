/**
 * Dashboard-specific JavaScript functionality
 */

class DashboardManager {
    constructor() {
        this.charts = {};
        this.updateInterval = null;
        this.init();
    }
    
    init() {
        this.setupEventListeners();
        this.initCharts();
        this.startAutoRefresh();
    }
    
    setupEventListeners() {
        // File upload area
        const fileUploadArea = document.getElementById('fileUploadArea');
        const uploadFile = document.getElementById('uploadFile');
        
        if (fileUploadArea && uploadFile) {
            fileUploadArea.addEventListener('click', () => uploadFile.click());
            fileUploadArea.addEventListener('dragover', this.handleDragOver.bind(this));
            fileUploadArea.addEventListener('drop', this.handleFileDrop.bind(this));
            uploadFile.addEventListener('change', this.handleFileSelect.bind(this));
        }
        
        // Upload button
        const uploadBtn = document.getElementById('uploadBtn');
        if (uploadBtn) {
            uploadBtn.addEventListener('click', () => this.uploadFile());
        }
        
        // Training form
        const exportFile = document.getElementById('exportFile');
        if (exportFile) {
            exportFile.addEventListener('change', this.handleExportFileSelect.bind(this));
        }
        
        // Chart time range buttons
        document.querySelectorAll('[data-chart-range]').forEach(button => {
            button.addEventListener('click', (e) => {
                this.updateChartRange(e.target.getAttribute('data-chart-range'));
            });
        });
    }
    
    initCharts() {
        this.initTrainingChart();
        this.initSystemMetricsChart();
    }
    
    initTrainingChart() {
        const ctx = document.getElementById('trainingChart');
        if (!ctx) return;
        
        // Check if Chart.js is available
        if (typeof Chart === 'undefined') {
            console.warn('Chart.js not loaded, skipping chart initialization');
            return;
        }
        
        // Check if canvas is already being used by another chart
        if (ctx.chart) {
            console.log('Canvas already has a chart, destroying existing chart');
            ctx.chart.destroy();
        }
        
        // Destroy existing chart if it exists
        if (this.charts.training) {
            this.charts.training.destroy();
        }
        
        try {
            this.charts.training = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Training Progress',
                        data: [],
                        borderColor: '#0d6efd',
                        backgroundColor: 'rgba(13, 110, 253, 0.1)',
                        tension: 0.4,
                        fill: true,
                        pointRadius: 4,
                        pointHoverRadius: 6
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            display: false
                        },
                        tooltip: {
                            mode: 'index',
                            intersect: false,
                            callbacks: {
                                label: function(context) {
                                    return `Progress: ${context.parsed.y}%`;
                                }
                            }
                        }
                    },
                    scales: {
                        x: {
                            display: true,
                            title: {
                                display: true,
                                text: 'Time'
                            }
                        },
                        y: {
                            beginAtZero: true,
                            max: 100,
                            title: {
                                display: true,
                                text: 'Progress (%)'
                            },
                            ticks: {
                                callback: function(value) {
                                    return value + '%';
                                }
                            }
                        }
                    },
                    interaction: {
                        mode: 'nearest',
                        axis: 'x',
                        intersect: false
                    }
                }
            });
            
            // Store reference to chart on canvas element
            ctx.chart = this.charts.training;
            
        } catch (error) {
            console.error('Failed to initialize training chart:', error);
        }
    }
    
    initSystemMetricsChart() {
        const ctx = document.getElementById('systemMetricsChart');
        if (!ctx) return;
        
        // Check if Chart.js is available
        if (typeof Chart === 'undefined') {
            console.warn('Chart.js not loaded, skipping chart initialization');
            return;
        }
        
        // Check if canvas is already being used by another chart
        if (ctx.chart) {
            console.log('Canvas already has a chart, destroying existing chart');
            ctx.chart.destroy();
        }
        
        // Destroy existing chart if it exists
        if (this.charts.systemMetrics) {
            this.charts.systemMetrics.destroy();
        }
        
        try {
            this.charts.systemMetrics = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'CPU Usage',
                        data: [],
                        borderColor: '#dc3545',
                        backgroundColor: 'rgba(220, 53, 69, 0.1)',
                        tension: 0.4,
                        fill: false
                    }, {
                        label: 'Memory Usage',
                        data: [],
                        borderColor: '#fd7e14',
                        backgroundColor: 'rgba(253, 126, 20, 0.1)',
                        tension: 0.4,
                        fill: false
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            display: true,
                            position: 'top'
                        }
                    },
                    scales: {
                        x: {
                            display: true,
                            title: {
                                display: true,
                                text: 'Time'
                            }
                        },
                        y: {
                            beginAtZero: true,
                            max: 100,
                            title: {
                                display: true,
                                text: 'Usage (%)'
                            },
                            ticks: {
                                callback: function(value) {
                                    return value + '%';
                                }
                            }
                        }
                    }
                }
            });
            
            // Store reference to chart on canvas element
            ctx.chart = this.charts.systemMetrics;
            
        } catch (error) {
            console.error('Failed to initialize system metrics chart:', error);
        }
    }
    
    async updateDashboard() {
        try {
            const [status, trainingJobsResponse, modelsResponse] = await Promise.all([
                utils.apiCall('/api/health/status'),
                utils.apiCall('/api/training/jobs'),
                utils.apiCall('/api/models')
            ]);
            
            // Extract the trainings array from the new response structure
            const trainingJobs = trainingJobsResponse.trainings || trainingJobsResponse;
            const models = modelsResponse.models || modelsResponse;
            
            this.updateStatusCards(status);
            this.updateTrainingChart(trainingJobs);
            this.updateActivityList(trainingJobs);
            this.updateSystemMetrics(status);
            this.updateEvaluationStatistics(trainingJobs);
            this.updateRecentRecommendations(trainingJobs);
        } catch (error) {
            console.error('Failed to update dashboard:', error);
        }
    }
    
    updateStatusCards(status) {
        // Update status card values
        Object.keys(status).forEach(key => {
            const element = document.querySelector(`[data-status="${key}"]`);
            if (element) {
                element.textContent = status[key];
            }
        });
        
        // Update status indicators
        const statusElements = document.querySelectorAll('[data-status-indicator]');
        statusElements.forEach(element => {
            const statusKey = element.getAttribute('data-status-indicator');
            const statusValue = status[statusKey];
            if (statusValue) {
                element.className = `badge bg-${utils.getStatusColor(statusValue)}`;
                element.textContent = statusValue;
            }
        });
    }
    
    updateTrainingChart(trainingJobs) {
        if (!this.charts.training) return;
        
        const activeJobs = trainingJobs.filter(job => job.status === 'running');
        const now = new Date();
        
        // Update chart data
        this.charts.training.data.labels = activeJobs.map(job => {
            const startTime = new Date(job.created_at);
            const duration = Math.floor((now - startTime) / 1000 / 60); // minutes
            return `${duration}m`;
        });
        
        this.charts.training.data.datasets[0].data = activeJobs.map(job => job.progress || 0);
        this.charts.training.update('none'); // Update without animation for performance
    }
    
    updateSystemMetrics(status) {
        if (!this.charts.systemMetrics) return;
        
        const now = new Date();
        const timeLabel = now.toLocaleTimeString();
        
        // Add new data point
        this.charts.systemMetrics.data.labels.push(timeLabel);
        this.charts.systemMetrics.data.datasets[0].data.push(status.cpu_usage || 0);
        this.charts.systemMetrics.data.datasets[1].data.push(status.memory_usage || 0);
        
        // Keep only last 20 data points
        if (this.charts.systemMetrics.data.labels.length > 20) {
            this.charts.systemMetrics.data.labels.shift();
            this.charts.systemMetrics.data.datasets[0].data.shift();
            this.charts.systemMetrics.data.datasets[1].data.shift();
        }
        
        this.charts.systemMetrics.update('none');
    }
    
    updateActivityList(trainingJobs) {
        const activityList = document.getElementById('activityList');
        if (!activityList) return;
        
        const recentJobs = trainingJobs
            .sort((a, b) => new Date(b.created_at) - new Date(a.created_at))
            .slice(0, 5);
        
        if (recentJobs.length === 0) {
            activityList.innerHTML = `
                <div class="text-center text-muted py-4">
                    <i class="bi bi-activity fs-1 mb-3"></i>
                    <p>No recent activity</p>
                </div>
            `;
            return;
        }
        
        activityList.innerHTML = recentJobs.map(job => `
            <div class="activity-item d-flex align-items-center py-2">
                <div class="activity-icon me-3">
                    <i class="bi bi-${utils.getStatusIcon(job.status)} text-${utils.getStatusColor(job.status)}"></i>
                </div>
                <div class="activity-content flex-grow-1">
                    <div class="activity-title">${job.name || `Training Job ${job.id}`}</div>
                    <div class="activity-time text-muted small">${utils.formatDateTime(job.created_at)}</div>
                </div>
                <div class="activity-status">
                    <span class="badge bg-${utils.getStatusColor(job.status)}">${job.status}</span>
                </div>
            </div>
        `).join('');
    }
    
    updateEvaluationStatistics(trainingJobs) {
        // Filter completed jobs with evaluation results
        const evaluatedJobs = trainingJobs.filter(job => 
            job.status === 'completed' && job.evaluation_results
        );
        
        if (evaluatedJobs.length === 0) {
            // Reset statistics to default values
            this.updateStatElement('eval_pass_rate', '0%');
            this.updateStatElement('eval_fail_rate', '0%');
            this.updateStatElement('avg_accuracy', '0%');
            this.updateStatElement('avg_f1_score', '0%');
            return;
        }
        
        // Calculate pass/fail rates
        const passCount = evaluatedJobs.filter(job => {
            const evaluation = job.evaluation_results;
            if (evaluation.threshold_checks) {
                const allPassed = Object.values(evaluation.threshold_checks).every(check => check.passed);
                return allPassed;
            }
            return false;
        }).length;
        
        const failCount = evaluatedJobs.length - passCount;
        const passRate = ((passCount / evaluatedJobs.length) * 100).toFixed(1);
        const failRate = ((failCount / evaluatedJobs.length) * 100).toFixed(1);
        
        // Calculate average metrics
        const avgAccuracy = evaluatedJobs.reduce((sum, job) => {
            return sum + (job.evaluation_results.accuracy || 0);
        }, 0) / evaluatedJobs.length;
        
        const avgF1Score = evaluatedJobs.reduce((sum, job) => {
            return sum + (job.evaluation_results.f1_score || 0);
        }, 0) / evaluatedJobs.length;
        
        // Update display
        this.updateStatElement('eval_pass_rate', `${passRate}%`);
        this.updateStatElement('eval_fail_rate', `${failRate}%`);
        this.updateStatElement('avg_accuracy', `${(avgAccuracy * 100).toFixed(1)}%`);
        this.updateStatElement('avg_f1_score', `${(avgF1Score * 100).toFixed(1)}%`);
    }
    
    updateRecentRecommendations(trainingJobs) {
        const recommendationsContainer = document.getElementById('recentRecommendations');
        if (!recommendationsContainer) return;
        
        // Collect all recommendations from recent completed jobs
        const allRecommendations = [];
        const recentCompletedJobs = trainingJobs
            .filter(job => job.status === 'completed' && job.evaluation_results)
            .sort((a, b) => new Date(b.updated_at) - new Date(a.updated_at))
            .slice(0, 10);
        
        recentCompletedJobs.forEach(job => {
            if (job.evaluation_results.recommendations) {
                job.evaluation_results.recommendations.forEach(rec => {
                    allRecommendations.push({
                        ...rec,
                        jobId: job.id,
                        jobName: job.name || `Job ${job.id}`,
                        timestamp: job.updated_at
                    });
                });
            }
        });
        
        // Sort by priority and take top 5
        const sortedRecommendations = allRecommendations
            .sort((a, b) => {
                const priorityOrder = { high: 3, medium: 2, low: 1 };
                return priorityOrder[b.priority] - priorityOrder[a.priority];
            })
            .slice(0, 5);
        
        if (sortedRecommendations.length === 0) {
            recommendationsContainer.innerHTML = `
                <div class="text-center text-muted py-3">
                    <i class="bi bi-lightbulb fs-1 mb-2"></i>
                    <p>No recent recommendations</p>
                </div>
            `;
            return;
        }
        
        recommendationsContainer.innerHTML = sortedRecommendations.map(rec => `
            <div class="recommendation-item ${rec.priority}-priority p-2 mb-2 rounded">
                <div class="d-flex align-items-start">
                    <i class="bi bi-${rec.priority === 'high' ? 'exclamation-triangle text-warning' : 
                                     rec.priority === 'medium' ? 'info-circle text-info' : 
                                     'check-circle text-success'} me-2 mt-1"></i>
                    <div class="flex-grow-1">
                        <div class="small fw-semibold">${rec.category}</div>
                        <div class="small text-muted">${rec.message}</div>
                        <div class="small text-muted">
                            <i class="bi bi-gear me-1"></i>${rec.jobName}
                        </div>
                    </div>
                </div>
            </div>
        `).join('');
    }
    
    updateStatElement(statName, value) {
        const element = document.querySelector(`[data-stat="${statName}"]`);
        if (element) {
            element.textContent = value;
        }
    }
    
    updateChartRange(range) {
        // Update active button
        document.querySelectorAll('[data-chart-range]').forEach(btn => {
            btn.classList.remove('active');
        });
        event.target.classList.add('active');
        
        // Update chart data based on range
        this.updateTrainingChartData(range);
    }
    
    async updateTrainingChartData(range) {
        try {
            const params = new URLSearchParams({ range });
            const data = await utils.apiCall(`/api/training/chart-data?${params}`);
            
            if (this.charts.training) {
                this.charts.training.data.labels = data.labels;
                this.charts.training.data.datasets[0].data = data.values;
                this.charts.training.update();
            }
        } catch (error) {
            console.error('Failed to update chart data:', error);
        }
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
    
    handleExportFileSelect(e) {
        const file = e.target.files[0];
        if (file) {
            this.updateFileInfo(file, 'fileName', 'fileSize');
        }
    }
    
    handleFile(file) {
        // Validate file type
        if (!file.name.endsWith('.json')) {
            utils.showError('Please select a valid JSON file');
            return;
        }
        
        // Update file info
        this.updateFileInfo(file, 'uploadFileName', 'uploadFileSize');
        
        // Enable upload button
        const uploadBtn = document.getElementById('uploadBtn');
        if (uploadBtn) {
            uploadBtn.disabled = false;
        }
    }
    
    async uploadFile() {
        try {
            const fileInput = document.getElementById('uploadFile');
            if (!fileInput.files.length) {
                utils.showError('Please select a file to upload');
                return;
            }
            
            const file = fileInput.files[0];
            
            // Validate file type
            if (!file.name.endsWith('.json')) {
                utils.showError('Please select a valid JSON file');
                return;
            }
            
            utils.showLoading();
            
            // Show upload progress
            const progressContainer = document.getElementById('uploadProgress');
            const progressBar = progressContainer?.querySelector('.progress-bar');
            if (progressContainer) {
                progressContainer.style.display = 'block';
            }
            
            const formData = new FormData();
            formData.append('file', file);
            
            const response = await utils.apiCall('/api/training/exports/upload', {
                method: 'POST',
                body: formData,
                headers: {} // Let browser set content-type for FormData
            });
            
            utils.showSuccess('Export file uploaded successfully');
            
            // Close modal and refresh data
            const modal = bootstrap.Modal.getInstance(document.getElementById('uploadModal'));
            if (modal) {
                modal.hide();
            }
            
            // Reset form
            this.resetUploadForm();
            
            // Refresh dashboard to show new file
            this.updateDashboard();
            
        } catch (error) {
            utils.showError('Failed to upload file', error);
        } finally {
            utils.hideLoading();
            
            // Hide progress
            const progressContainer = document.getElementById('uploadProgress');
            if (progressContainer) {
                progressContainer.style.display = 'none';
            }
        }
    }
    
    resetUploadForm() {
        const fileInput = document.getElementById('uploadFile');
        if (fileInput) {
            fileInput.value = '';
        }
        
        const uploadBtn = document.getElementById('uploadBtn');
        if (uploadBtn) {
            uploadBtn.disabled = true;
        }
        
        // Reset file info display
        const fileInfo = document.getElementById('fileInfo');
        if (fileInfo) {
            fileInfo.style.display = 'none';
        }
    }
    
    updateFileInfo(file, nameId, sizeId) {
        const nameElement = document.getElementById(nameId);
        const sizeElement = document.getElementById(sizeId);
        
        if (nameElement) {
            nameElement.textContent = file.name;
        }
        if (sizeElement) {
            sizeElement.textContent = `(${utils.formatFileSize(file.size)})`;
        }
        
        // Show file info
        const fileInfo = document.getElementById('fileInfo');
        if (fileInfo) {
            fileInfo.style.display = 'block';
        }
    }
    
    startAutoRefresh() {
        // Refresh dashboard every 2 minutes - very conservative for training system
        this.updateInterval = setInterval(() => {
            this.updateDashboard();
        }, 120000); // 2 minutes
    }
    
    stopAutoRefresh() {
        if (this.updateInterval) {
            clearInterval(this.updateInterval);
            this.updateInterval = null;
        }
    }
    
    // Public methods for external access
    refresh() {
        this.updateDashboard();
    }
    
    destroy() {
        this.stopAutoRefresh();
        
        // Destroy charts
        Object.values(this.charts).forEach(chart => {
            if (chart && chart.destroy) {
                chart.destroy();
            }
        });
    }
}

// Initialize dashboard when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    // Add a small delay to ensure main app has finished loading
    setTimeout(() => {
        // Prevent multiple initializations
        if (window.dashboard) {
            window.dashboard.destroy();
        }
        window.dashboard = new DashboardManager();
    }, 100);
});

// Global functions for use in HTML
window.refreshDashboard = () => window.dashboard?.refresh();
window.uploadExportFile = () => window.dashboard?.uploadFile(); 