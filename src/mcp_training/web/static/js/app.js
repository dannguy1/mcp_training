/**
 * Main Application JavaScript for MCP Training Service
 */

class MCPTrainingApp {
    constructor() {
        this.currentPage = utils.getCurrentPage();
        this.charts = {};
        this.websocket = null;
        this.autoRefreshInterval = null;
        this.init();
    }
    
    init() {
        this.setupEventListeners();
        this.setActiveNavigation();
        this.initTooltips();
        this.initPopovers();
        this.loadPageData();
        this.setupWebSocket();
        this.startAutoRefresh();
    }
    
    setupEventListeners() {
        // Sidebar toggle using Bootstrap offcanvas
        const sidebarToggle = document.getElementById('sidebarToggle');
        const sidebar = document.getElementById('sidebar');
        
        if (sidebarToggle && sidebar) {
            sidebarToggle.addEventListener('click', () => {
                const offcanvas = new bootstrap.Offcanvas(sidebar);
                offcanvas.show();
            });
        }
        
        // Close sidebar on mobile when navigation links are clicked
        document.querySelectorAll('[data-page]').forEach(link => {
            link.addEventListener('click', () => {
                // Close sidebar on mobile devices
                if (window.innerWidth < 992) {
                    const offcanvas = bootstrap.Offcanvas.getInstance(sidebar);
                    if (offcanvas) {
                        offcanvas.hide();
                    }
                }
            });
        });
        
        // Global error handling
        window.addEventListener('error', (e) => {
            console.error('Global error:', e.error);
            utils.showError('An unexpected error occurred', e.error);
        });
        
        // Handle unhandled promise rejections
        window.addEventListener('unhandledrejection', (e) => {
            console.error('Unhandled promise rejection:', e.reason);
            utils.showError('An unexpected error occurred', e.reason);
        });
    }
    
    toggleSidebar() {
        const sidebar = document.getElementById('sidebar');
        if (sidebar) {
            const offcanvas = new bootstrap.Offcanvas(sidebar);
            offcanvas.show();
        }
    }
    
    setActiveNavigation() {
        utils.setActiveNavigation();
    }
    
    initTooltips() {
        utils.initTooltips();
    }
    
    initPopovers() {
        utils.initPopovers();
    }
    
    async loadPageData() {
        try {
            switch (this.currentPage) {
                case 'dashboard':
                    await this.loadDashboardData();
                    break;
                case 'training':
                    await this.loadTrainingData();
                    break;
                case 'models':
                    await this.loadModelsData();
                    break;
                case 'logs':
                    await this.loadLogsData();
                    break;
                case 'settings':
                    await this.loadSettingsData();
                    break;
            }
        } catch (error) {
            console.error('Failed to load page data:', error);
            utils.showError('Failed to load page data', error);
        }
    }
    
    async loadDashboardData() {
        try {
            const [status, trainingJobs, models] = await Promise.all([
                utils.apiCall('/health/status'),
                utils.apiCall('/training/jobs'),
                utils.apiCall('/models')
            ]);
            
            this.updateDashboard(status, trainingJobs, models);
        } catch (error) {
            utils.showError('Failed to load dashboard data', error);
        }
    }
    
    async loadTrainingData() {
        try {
            const trainingJobs = await utils.apiCall('/training/jobs');
            this.updateTrainingTable(trainingJobs);
        } catch (error) {
            utils.showError('Failed to load training data', error);
        }
    }
    
    async loadModelsData() {
        try {
            const models = await utils.apiCall('/models');
            this.updateModelsTable(models);
        } catch (error) {
            utils.showError('Failed to load models data', error);
        }
    }
    
    async loadLogsData() {
        try {
            const logs = await utils.apiCall('/logs/');
            this.updateLogsTable(logs);
        } catch (error) {
            utils.showError('Failed to load logs data', error);
        }
    }
    
    async loadSettingsData() {
        try {
            const settings = await utils.apiCall('/settings');
            this.updateSettingsForm(settings);
        } catch (error) {
            utils.showError('Failed to load settings data', error);
        }
    }
    
    updateDashboard(status, trainingJobs, models) {
        this.updateStatusCards(status);
        this.updateTrainingChart(trainingJobs);
        this.updateActivityList(trainingJobs);
        this.updateQuickActions();
    }
    
    updateStatusCards(status) {
        // Update system status cards
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
        const chartCanvas = document.getElementById('trainingChart');
        if (!chartCanvas) return;
        
        if (!this.charts.training) {
            this.initTrainingChart(chartCanvas);
        }
        
        const activeJobs = trainingJobs.filter(job => job.status === 'running');
        
        this.charts.training.data.labels = activeJobs.map(job => `Job ${job.id}`);
        this.charts.training.data.datasets[0].data = activeJobs.map(job => job.progress || 0);
        this.charts.training.update();
    }
    
    initTrainingChart(canvas) {
        this.charts.training = new Chart(canvas, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Training Progress',
                    data: [],
                    borderColor: '#0d6efd',
                    backgroundColor: 'rgba(13, 110, 253, 0.1)',
                    tension: 0.4,
                    fill: true
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100,
                        ticks: {
                            callback: function(value) {
                                return value + '%';
                            }
                        }
                    }
                }
            }
        });
    }
    
    updateActivityList(trainingJobs) {
        const activityList = document.getElementById('activityList');
        if (!activityList) return;
        
        const recentJobs = trainingJobs
            .sort((a, b) => new Date(b.created_at) - new Date(a.created_at))
            .slice(0, 5);
        
        activityList.innerHTML = recentJobs.map(job => `
            <div class="activity-item d-flex align-items-center py-2">
                <div class="activity-icon me-3">
                    <i class="bi bi-${utils.getStatusIcon(job.status)} text-${utils.getStatusColor(job.status)}"></i>
                </div>
                <div class="activity-content flex-grow-1">
                    <div class="activity-title">Training Job ${job.id}</div>
                    <div class="activity-time text-muted small">${utils.formatDateTime(job.created_at)}</div>
                </div>
                <div class="activity-status">
                    <span class="badge bg-${utils.getStatusColor(job.status)}">${job.status}</span>
                </div>
            </div>
        `).join('');
    }
    
    updateQuickActions() {
        // Update quick action buttons with current state
        const startTrainingBtn = document.querySelector('[data-action="start-training"]');
        if (startTrainingBtn) {
            startTrainingBtn.addEventListener('click', () => {
                this.showNewTrainingModal();
            });
        }
        
        const uploadExportBtn = document.querySelector('[data-action="upload-export"]');
        if (uploadExportBtn) {
            uploadExportBtn.addEventListener('click', () => {
                this.showUploadModal();
            });
        }
    }
    
    updateTrainingTable(trainingJobs) {
        const tableBody = document.querySelector('#trainingTable tbody');
        if (!tableBody) return;
        
        tableBody.innerHTML = trainingJobs.map(job => `
            <tr data-job-id="${job.id}">
                <td>${job.id}</td>
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
                        <button class="btn btn-outline-info" onclick="app.viewJobDetails('${job.id}')" 
                                title="View Details">
                            <i class="bi bi-eye"></i>
                        </button>
                        ${job.status === 'running' ? `
                            <button class="btn btn-outline-warning" onclick="app.cancelJob('${job.id}')" 
                                    title="Cancel Job">
                                <i class="bi bi-stop"></i>
                            </button>
                        ` : ''}
                        <button class="btn btn-outline-danger" onclick="app.deleteJob('${job.id}')" 
                                title="Delete Job">
                            <i class="bi bi-trash"></i>
                        </button>
                    </div>
                </td>
            </tr>
        `).join('');
    }
    
    updateModelsTable(models) {
        const tableBody = document.querySelector('#modelsTable tbody');
        if (!tableBody) return;
        
        tableBody.innerHTML = models.map(model => `
            <tr>
                <td>${model.id}</td>
                <td>${model.name}</td>
                <td>${model.version}</td>
                <td>
                    <span class="badge bg-${utils.getStatusColor(model.status)}">
                        ${model.status}
                    </span>
                </td>
                <td>${utils.formatDateTime(model.created_at)}</td>
                <td>${utils.formatFileSize(model.size || 0)}</td>
                <td class="table-actions">
                    <div class="btn-group btn-group-sm">
                        <button class="btn btn-outline-info" onclick="app.viewModelDetails('${model.id}')" 
                                title="View Details">
                            <i class="bi bi-eye"></i>
                        </button>
                        <button class="btn btn-outline-success" onclick="app.deployModel('${model.id}')" 
                                title="Deploy Model">
                            <i class="bi bi-rocket"></i>
                        </button>
                        <button class="btn btn-outline-secondary" onclick="app.downloadModel('${model.id}')" 
                                title="Download Model">
                            <i class="bi bi-download"></i>
                        </button>
                        <button class="btn btn-outline-danger" onclick="app.deleteModel('${model.id}')" 
                                title="Delete Model">
                            <i class="bi bi-trash"></i>
                        </button>
                    </div>
                </td>
            </tr>
        `).join('');
    }
    
    updateLogsTable(logs) {
        const tableBody = document.querySelector('#logsTable tbody');
        if (!tableBody) return;
        
        tableBody.innerHTML = logs.map(log => `
            <tr>
                <td>${utils.formatDateTime(log.timestamp)}</td>
                <td>
                    <span class="badge bg-${utils.getStatusColor(log.level)}">
                        ${log.level.toUpperCase()}
                    </span>
                </td>
                <td>${log.service}</td>
                <td class="text-truncate-2">${log.message}</td>
                <td class="table-actions">
                    <button class="btn btn-outline-info btn-sm" onclick="app.viewLogDetails('${log.id}')" 
                            title="View Details">
                        <i class="bi bi-eye"></i>
                    </button>
                </td>
            </tr>
        `).join('');
    }
    
    setupWebSocket() {
        try {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = `${protocol}//${window.location.host}/ws`;
            
            this.websocket = new WebSocket(wsUrl);
            
            this.websocket.onopen = () => {
                console.log('WebSocket connected');
            };
            
            this.websocket.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    this.handleWebSocketMessage(data);
                } catch (error) {
                    console.error('Failed to parse WebSocket message:', error);
                }
            };
            
            this.websocket.onerror = (error) => {
                console.error('WebSocket error:', error);
            };
            
            this.websocket.onclose = () => {
                console.log('WebSocket disconnected');
                // Attempt to reconnect after 5 seconds
                setTimeout(() => {
                    this.setupWebSocket();
                }, 5000);
            };
        } catch (error) {
            console.error('Failed to setup WebSocket:', error);
        }
    }
    
    handleWebSocketMessage(data) {
        switch (data.type) {
            case 'training_update':
                this.handleTrainingUpdate(data.data);
                break;
            case 'model_ready':
                this.handleModelReady(data.data);
                break;
            case 'system_status':
                this.handleSystemStatusUpdate(data.data);
                break;
            case 'error':
                utils.showError('System Error', new Error(data.message));
                break;
        }
    }
    
    handleTrainingUpdate(data) {
        // Update training progress in real-time
        const progressElement = document.querySelector(`[data-job-id="${data.job_id}"] .progress-bar`);
        if (progressElement) {
            progressElement.style.width = `${data.progress}%`;
            progressElement.nextElementSibling.textContent = `${data.progress}%`;
        }
        
        // Update status badge
        const statusElement = document.querySelector(`[data-job-id="${data.job_id}"] .badge`);
        if (statusElement) {
            statusElement.className = `badge bg-${utils.getStatusColor(data.status)}`;
            statusElement.textContent = data.status;
        }
        
        // Show notification for completed jobs
        if (data.status === 'completed') {
            utils.showSuccess(`Training job ${data.job_id} completed successfully`);
        } else if (data.status === 'failed') {
            utils.showError(`Training job ${data.job_id} failed`);
        }
    }
    
    handleModelReady(data) {
        utils.showSuccess(`Model ${data.model_id} is ready for deployment`);
        // Refresh models table
        this.loadModelsData();
    }
    
    handleSystemStatusUpdate(data) {
        this.updateStatusCards(data);
    }
    
    startAutoRefresh() {
        // Refresh data every 30 seconds
        this.autoRefreshInterval = setInterval(() => {
            this.loadPageData();
        }, 30000);
    }
    
    stopAutoRefresh() {
        if (this.autoRefreshInterval) {
            clearInterval(this.autoRefreshInterval);
            this.autoRefreshInterval = null;
        }
    }
    
    // Action methods
    showNewTrainingModal() {
        const modal = new bootstrap.Modal(document.getElementById('newTrainingModal'));
        modal.show();
    }
    
    showUploadModal() {
        const modal = new bootstrap.Modal(document.getElementById('uploadModal'));
        modal.show();
    }
    
    async submitTraining() {
        try {
            utils.showLoading();
            
            const formData = new FormData();
            const fileInput = document.getElementById('exportFile');
            const configSelect = document.getElementById('modelConfig');
            
            if (fileInput.files.length === 0) {
                throw new Error('Please select an export file');
            }
            
            formData.append('export_file', fileInput.files[0]);
            formData.append('config', configSelect.value);
            
            const response = await utils.apiCall('/training/jobs', {
                method: 'POST',
                body: formData,
                headers: {} // Let browser set content-type for FormData
            });
            
            utils.showSuccess('Training job started successfully');
            
            // Close modal and refresh data
            const modal = bootstrap.Modal.getInstance(document.getElementById('newTrainingModal'));
            modal.hide();
            
            this.loadPageData();
            
        } catch (error) {
            utils.showError('Failed to start training job', error);
        } finally {
            utils.hideLoading();
        }
    }
    
    async cancelJob(jobId) {
        if (!confirm('Are you sure you want to cancel this training job?')) {
            return;
        }
        
        try {
            await utils.apiCall(`/training/jobs/${jobId}/cancel`, { method: 'POST' });
            utils.showSuccess('Training job cancelled successfully');
            this.loadPageData();
        } catch (error) {
            utils.showError('Failed to cancel training job', error);
        }
    }
    
    async deleteJob(jobId) {
        if (!confirm('Are you sure you want to delete this training job?')) {
            return;
        }
        
        try {
            await utils.apiCall(`/training/jobs/${jobId}`, { method: 'DELETE' });
            utils.showSuccess('Training job deleted successfully');
            this.loadPageData();
        } catch (error) {
            utils.showError('Failed to delete training job', error);
        }
    }
    
    async deployModel(modelId) {
        try {
            await utils.apiCall(`/models/${modelId}/deploy`, { method: 'POST' });
            utils.showSuccess('Model deployed successfully');
            this.loadPageData();
        } catch (error) {
            utils.showError('Failed to deploy model', error);
        }
    }
    
    async downloadModel(modelId) {
        try {
            const response = await fetch(`${API_BASE}/models/${modelId}/download`);
            if (response.ok) {
                const blob = await response.blob();
                const url = window.URL.createObjectURL(blob);
                utils.downloadFile(url, `model-${modelId}.joblib`);
            } else {
                throw new Error('Failed to download model');
            }
        } catch (error) {
            utils.showError('Failed to download model', error);
        }
    }
    
    async deleteModel(modelId) {
        if (!confirm('Are you sure you want to delete this model?')) {
            return;
        }
        
        try {
            await utils.apiCall(`/models/${modelId}`, { method: 'DELETE' });
            utils.showSuccess('Model deleted successfully');
            this.loadPageData();
        } catch (error) {
            utils.showError('Failed to delete model', error);
        }
    }
    
    viewJobDetails(jobId) {
        // Implement job details view
        utils.showInfo(`Viewing details for job ${jobId}`);
    }
    
    viewModelDetails(modelId) {
        // Implement model details view
        utils.showInfo(`Viewing details for model ${modelId}`);
    }
    
    viewLogDetails(logId) {
        // Implement log details view
        utils.showInfo(`Viewing details for log ${logId}`);
    }
}

// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.app = new MCPTrainingApp();
});

// Global functions for use in HTML
window.startTraining = () => window.app.showNewTrainingModal();
window.uploadExport = () => window.app.showUploadModal();
window.viewModels = () => window.location.href = '/models';
window.submitTraining = () => window.app.submitTraining(); 