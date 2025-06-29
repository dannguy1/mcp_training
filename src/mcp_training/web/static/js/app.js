/**
 * Main Application JavaScript for MCP Training Service
 */

class MCPTrainingApp {
    constructor() {
        this.currentPage = utils.getCurrentPage();
        this.websocket = null;
        this.autoRefreshInterval = null;
        this.init();
    }
    
    init() {
        this.setupEventListeners();
        this.setActiveNavigation();
        this.initTooltips();
        this.initPopovers();
        this.setupWebSocket();
        this.startAutoRefresh();
        this.setupPageUnloadDetection();
        this.setupGlobalErrorHandling();
        this.loadPageData();
    }
    
    setupEventListeners() {
        // Navigation
        document.querySelectorAll('.nav-link').forEach(link => {
            link.addEventListener('click', (e) => {
                e.preventDefault();
                const href = link.getAttribute('href');
                if (href && href !== '#') {
                    window.location.href = href;
                }
            });
        });
        
        // Sidebar toggle
        const sidebarToggle = document.getElementById('sidebarToggle');
        if (sidebarToggle) {
            sidebarToggle.addEventListener('click', () => this.toggleSidebar());
        }
        
        // Close sidebar on mobile when navigation links are clicked
        document.querySelectorAll('[data-page]').forEach(link => {
            link.addEventListener('click', () => {
                // Close sidebar on mobile devices
                if (window.innerWidth < 992) {
                    const sidebar = document.getElementById('sidebar');
                    if (sidebar) {
                        const offcanvas = bootstrap.Offcanvas.getInstance(sidebar);
                        if (offcanvas) {
                            offcanvas.hide();
                        }
                    }
                }
            });
        });
        
        // Quick actions
        this.updateQuickActions();
        
        // Training modal events
        const newTrainingModal = document.getElementById('newTrainingModal');
        if (newTrainingModal) {
            newTrainingModal.addEventListener('show.bs.modal', () => {
                this.loadExportFiles();
            });
            
            newTrainingModal.addEventListener('hidden.bs.modal', () => {
                this.resetTrainingForm();
            });
        }
        
        // Submit training button
        const submitTrainingBtn = document.getElementById('submitTrainingBtn');
        if (submitTrainingBtn) {
            submitTrainingBtn.addEventListener('click', () => {
                this.submitTraining();
            });
        }
        
        // Global error handling
        this.setupGlobalErrorHandling();
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
            const [status, trainingJobsResponse, modelsResponse] = await Promise.all([
                utils.apiCall('/api/health/status'),
                utils.apiCall('/api/training/jobs'),
                utils.apiCall('/api/models')
            ]);
            
            // Extract the trainings array from the new response structure
            const trainingJobs = trainingJobsResponse.trainings || trainingJobsResponse;
            const models = modelsResponse.models || modelsResponse;
            
            this.updateDashboard(status, trainingJobs, models);
        } catch (error) {
            utils.showError('Failed to load dashboard data', error);
        }
    }
    
    async loadTrainingData() {
        try {
            const trainingJobsResponse = await utils.apiCall('/api/training/jobs');
            const trainingJobs = trainingJobsResponse.trainings || trainingJobsResponse;
            this.updateTrainingTable(trainingJobs);
        } catch (error) {
            utils.showError('Failed to load training data', error);
        }
    }
    
    async loadModelsData() {
        try {
            const modelsResponse = await utils.apiCall('/api/models');
            const models = modelsResponse.models || modelsResponse;
            this.updateModelsTable(models);
        } catch (error) {
            utils.showError('Failed to load models data', error);
        }
    }
    
    async loadLogsData() {
        try {
            const response = await utils.apiCall('/api/logs/');
            // Extract logs array from response
            const logs = response.logs || [];
            this.updateLogsTable(logs);
        } catch (error) {
            utils.showError('Failed to load logs data', error);
        }
    }
    
    async loadSettingsData() {
        // Settings are handled by SettingsManager in settings.js
        // No need to load settings data here
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
        // Let DashboardManager handle chart updates if it exists
        if (window.dashboard && window.dashboard.charts && window.dashboard.charts.training) {
            window.dashboard.updateTrainingChart(trainingJobs);
        }
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
            
            // Set connection timeout
            this.connectionTimeout = setTimeout(() => {
                if (this.websocket && this.websocket.readyState === WebSocket.CONNECTING) {
                    console.warn('WebSocket connection timeout');
                    this.websocket.close();
                }
            }, 10000); // 10 second timeout
            
            this.websocket.onopen = () => {
                console.log('WebSocket connected');
                clearTimeout(this.connectionTimeout);
                this.reconnectAttempts = 0;
                this.clearConnectionError();
                
                // Send a ping to verify connection is working
                this.sendPing();
            };
            
            this.websocket.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    
                    // Handle ping/pong for connection health
                    if (data.type === 'pong') {
                        this.lastPong = Date.now();
                        return;
                    }
                    
                    this.handleWebSocketMessage(data);
                } catch (error) {
                    console.error('Failed to parse WebSocket message:', error);
                }
            };
            
            this.websocket.onerror = (error) => {
                console.error('WebSocket error:', error);
                clearTimeout(this.connectionTimeout);
                this.showConnectionError();
            };
            
            this.websocket.onclose = (event) => {
                console.log('WebSocket disconnected', event.code, event.reason);
                clearTimeout(this.connectionTimeout);
                
                // Don't show loading state for normal closures or during page unload
                if (event.code !== 1000 && event.code !== 1001 && !this.isPageUnloading) {
                    this.showConnectionError();
                }
                
                // Implement exponential backoff for reconnection
                const delay = Math.min(1000 * Math.pow(2, this.reconnectAttempts), 30000);
                this.reconnectAttempts++;
                
                setTimeout(() => {
                    if (!this.isPageUnloading && this.reconnectAttempts < 5) {
                        console.log(`Attempting WebSocket reconnection (attempt ${this.reconnectAttempts})`);
                        this.setupWebSocket();
                    } else if (this.reconnectAttempts >= 5) {
                        console.error('Max WebSocket reconnection attempts reached');
                        this.showConnectionError('Connection failed after multiple attempts. Please refresh the page.');
                    }
                }, delay);
            };
        } catch (error) {
            console.error('Failed to setup WebSocket:', error);
            this.showConnectionError();
        }
    }
    
    sendPing() {
        if (this.websocket && this.websocket.readyState === WebSocket.OPEN) {
            try {
                this.websocket.send(JSON.stringify({ type: 'ping' }));
                
                // Check if we get a pong response within 5 seconds
                setTimeout(() => {
                    if (this.lastPong && (Date.now() - this.lastPong) > 5000) {
                        console.warn('WebSocket ping timeout, reconnecting...');
                        this.websocket.close();
                    }
                }, 5000);
            } catch (error) {
                console.error('Failed to send ping:', error);
            }
        }
    }
    
    showConnectionError(message = 'Connection lost, reconnecting...') {
        // Show a subtle connection indicator instead of full loading overlay
        const indicator = document.getElementById('connectionIndicator');
        if (indicator) {
            indicator.style.display = 'block';
            indicator.className = 'connection-indicator connection-error';
            indicator.innerHTML = `<i class="fas fa-exclamation-triangle"></i> ${message}`;
        }
        
        // Also show a toast notification
        if (typeof utils !== 'undefined') {
            utils.showWarning(message);
        }
    }
    
    clearConnectionError() {
        const indicator = document.getElementById('connectionIndicator');
        if (indicator) {
            indicator.style.display = 'none';
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
            
            // Refresh the entire training table to ensure UI is up to date
            if (typeof trainingManager !== 'undefined') {
                trainingManager.loadTrainingJobs();
            }
            
            // Clear any loading states that might be active
            utils.hideLoading();
            
        } else if (data.status === 'failed') {
            utils.showError(`Training job ${data.job_id} failed`);
            
            // Refresh the training table to show the failed state
            if (typeof trainingManager !== 'undefined') {
                trainingManager.loadTrainingJobs();
            }
            
            // Clear any loading states
            utils.hideLoading();
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
        // Disable auto-refresh by default for training-focused system
        // Only enable if explicitly requested by user
        const autoRefreshEnabled = localStorage.getItem('autoRefreshEnabled') === 'true';
        
        if (autoRefreshEnabled) {
            // Very conservative refresh - only every 5 minutes
            this.autoRefreshInterval = setInterval(() => {
                this.loadPageData();
            }, 300000); // 5 minutes
        }
    }
    
    stopAutoRefresh() {
        if (this.autoRefreshInterval) {
            clearInterval(this.autoRefreshInterval);
            this.autoRefreshInterval = null;
        }
    }
    
    // Add page unload detection to prevent reconnection attempts
    setupPageUnloadDetection() {
        this.isPageUnloading = false;
        
        window.addEventListener('beforeunload', () => {
            this.isPageUnloading = true;
            this.stopAutoRefresh();
            if (this.websocket) {
                this.websocket.close(1000, 'Page unload');
            }
        });
        
        // Add keyboard shortcuts for emergency recovery
        document.addEventListener('keydown', (event) => {
            // Ctrl+Shift+R to force hide all loading states and refresh current page
            if (event.ctrlKey && event.shiftKey && event.key === 'R') {
                event.preventDefault();
                console.log('Emergency recovery: Force hiding all loading states and refreshing page');
                if (typeof utils !== 'undefined') {
                    utils.forceHideAllLoading();
                }
                this.clearConnectionError();
                
                // Refresh the current page data
                this.loadPageData();
                utils.showInfo('All loading states cleared and page refreshed. UI should be responsive now.');
            }
            
            // Ctrl+Shift+C to reconnect WebSocket
            if (event.ctrlKey && event.shiftKey && event.key === 'C') {
                event.preventDefault();
                console.log('Emergency recovery: Reconnecting WebSocket');
                if (this.websocket) {
                    this.websocket.close();
                }
                this.reconnectAttempts = 0;
                this.setupWebSocket();
                utils.showInfo('WebSocket reconnection initiated.');
            }
            
            // Ctrl+Shift+F to toggle auto-refresh (for debugging)
            if (event.ctrlKey && event.shiftKey && event.key === 'F') {
                event.preventDefault();
                const current = localStorage.getItem('autoRefreshEnabled') === 'true';
                localStorage.setItem('autoRefreshEnabled', (!current).toString());
                if (current) {
                    this.stopAutoRefresh();
                    utils.showInfo('Auto-refresh disabled');
                } else {
                    this.startAutoRefresh();
                    utils.showInfo('Auto-refresh enabled (5 min intervals)');
                }
            }
            
            // Ctrl+Shift+T to force refresh training page specifically
            if (event.ctrlKey && event.shiftKey && event.key === 'T') {
                event.preventDefault();
                console.log('Emergency recovery: Force refreshing training page');
                if (typeof trainingManager !== 'undefined') {
                    trainingManager.loadTrainingJobs();
                    utils.showInfo('Training page refreshed.');
                } else {
                    utils.showInfo('Training manager not available. Please navigate to Training page first.');
                }
            }
        });
        
        // Remove periodic health checks to reduce overhead
        // Only check WebSocket health when needed
    }
    
    checkUIIHealth() {
        // Simplified health check - only check WebSocket if it exists
        if (this.websocket && this.websocket.readyState === WebSocket.OPEN) {
            this.sendPing();
        }
    }
    
    // Action methods
    showNewTrainingModal() {
        const modal = new bootstrap.Modal(document.getElementById('newTrainingModal'));
        modal.show();
        
        // Load export files when modal opens
        this.loadExportFiles();
    }
    
    async loadExportFiles() {
        try {
            const response = await utils.apiCall('/api/training/exports');
            const exportFiles = response.exports || [];
            
            const exportFilesSelect = document.getElementById('exportFiles');
            if (exportFilesSelect) {
                // Clear existing options
                exportFilesSelect.innerHTML = '<option value="">Select export files...</option>';
                
                // Add export files
                exportFiles.forEach(file => {
                    const option = document.createElement('option');
                    option.value = file.path;
                    option.textContent = file.name;
                    exportFilesSelect.appendChild(option);
                });
                
                // Enable/disable submit button based on selection
                exportFilesSelect.addEventListener('change', () => {
                    const submitBtn = document.getElementById('submitTrainingBtn');
                    if (submitBtn) {
                        submitBtn.disabled = exportFilesSelect.selectedOptions.length === 0;
                    }
                });
            }
        } catch (error) {
            console.error('Failed to load export files:', error);
            utils.showError('Failed to load export files', error);
        }
    }
    
    resetTrainingForm() {
        // Reset form fields
        const jobName = document.getElementById('jobName');
        const modelConfig = document.getElementById('modelConfig');
        const maxIterations = document.getElementById('maxIterations');
        const learningRate = document.getElementById('learningRate');
        const description = document.getElementById('description');
        const exportFiles = document.getElementById('exportFiles');
        
        if (jobName) jobName.value = '';
        if (modelConfig) modelConfig.value = '';
        if (maxIterations) maxIterations.value = '1000';
        if (learningRate) learningRate.value = '0.01';
        if (description) description.value = '';
        if (exportFiles) {
            exportFiles.innerHTML = '<option value="">Select export files...</option>';
        }
        
        // Disable submit button
        const submitBtn = document.getElementById('submitTrainingBtn');
        if (submitBtn) {
            submitBtn.disabled = true;
        }
    }
    
    showUploadModal() {
        const modal = new bootstrap.Modal(document.getElementById('uploadModal'));
        modal.show();
    }
    
    showHelpModal() {
        // Create help modal if it doesn't exist
        let helpModal = document.getElementById('helpModal');
        if (!helpModal) {
            helpModal = document.createElement('div');
            helpModal.id = 'helpModal';
            helpModal.className = 'modal fade';
            helpModal.innerHTML = `
                <div class="modal-dialog modal-lg">
                    <div class="modal-content">
                        <div class="modal-header">
                            <h5 class="modal-title">
                                <i class="bi bi-question-circle me-2"></i>
                                Help & Keyboard Shortcuts
                            </h5>
                            <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                        </div>
                        <div class="modal-body">
                            <div class="row">
                                <div class="col-md-6">
                                    <h6><i class="bi bi-keyboard me-2"></i>Keyboard Shortcuts</h6>
                                    <div class="table-responsive">
                                        <table class="table table-sm">
                                            <tbody>
                                                <tr>
                                                    <td><kbd>Ctrl</kbd> + <kbd>Shift</kbd> + <kbd>R</kbd></td>
                                                    <td>Force hide all loading states</td>
                                                </tr>
                                                <tr>
                                                    <td><kbd>Ctrl</kbd> + <kbd>Shift</kbd> + <kbd>C</kbd></td>
                                                    <td>Reconnect WebSocket</td>
                                                </tr>
                                            </tbody>
                                        </table>
                                    </div>
                                </div>
                                <div class="col-md-6">
                                    <h6><i class="bi bi-tools me-2"></i>Troubleshooting</h6>
                                    <div class="alert alert-info">
                                        <strong>UI appears greyed out?</strong>
                                        <ul class="mb-0 mt-2">
                                            <li>Press <kbd>Ctrl</kbd> + <kbd>Shift</kbd> + <kbd>R</kbd> to force clear</li>
                                            <li>Check the connection indicator in the top-right</li>
                                            <li>Try refreshing the page</li>
                                        </ul>
                                    </div>
                                </div>
                            </div>
                            <div class="row mt-3">
                                <div class="col-12">
                                    <h6><i class="bi bi-info-circle me-2"></i>Connection Status</h6>
                                    <div class="d-flex align-items-center gap-3">
                                        <div class="d-flex align-items-center gap-2">
                                            <i class="bi bi-circle-fill text-success"></i>
                                            <span>Connected</span>
                                        </div>
                                        <div class="d-flex align-items-center gap-2">
                                            <i class="bi bi-circle-fill text-warning"></i>
                                            <span>Reconnecting</span>
                                        </div>
                                        <div class="d-flex align-items-center gap-2">
                                            <i class="bi bi-circle-fill text-danger"></i>
                                            <span>Disconnected</span>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div class="modal-footer">
                            <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Close</button>
                        </div>
                    </div>
                </div>
            `;
            document.body.appendChild(helpModal);
        }
        
        const modal = new bootstrap.Modal(helpModal);
        modal.show();
    }
    
    async submitTraining() {
        try {
            // Validate form first
            const exportFilesSelect = document.getElementById('exportFiles');
            const jobName = document.getElementById('jobName')?.value;
            const modelConfig = document.getElementById('modelConfig')?.value;
            
            if (!exportFilesSelect || exportFilesSelect.selectedOptions.length === 0) {
                throw new Error('Please select at least one export file');
            }
            
            if (!jobName || jobName.trim() === '') {
                throw new Error('Please enter a job name');
            }
            
            if (!modelConfig || modelConfig.trim() === '') {
                throw new Error('Please select a model configuration');
            }
            
            utils.showLoading();
            
            // Get form data
            const maxIterations = document.getElementById('maxIterations')?.value;
            const learningRate = document.getElementById('learningRate')?.value;
            const description = document.getElementById('description')?.value;
            
            // Get all selected export files
            const selectedFiles = Array.from(exportFilesSelect.selectedOptions).map(option => option.value);
            
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
            
            console.log('Submitting training request:', requestData);
            
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
            this.loadPageData();
            
        } catch (error) {
            console.error('Training submission error:', error);
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
            await utils.apiCall(`/api/training/jobs/${jobId}/cancel`, { method: 'POST' });
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
            await utils.apiCall(`/api/training/jobs/${jobId}`, { method: 'DELETE' });
            utils.showSuccess('Training job deleted successfully');
            this.loadPageData();
        } catch (error) {
            utils.showError('Failed to delete training job', error);
        }
    }
    
    async deployModel(modelId) {
        try {
            await utils.apiCall(`/api/models/${modelId}/deploy`, { method: 'POST' });
            utils.showSuccess('Model deployed successfully');
            this.loadPageData();
        } catch (error) {
            utils.showError('Failed to deploy model', error);
        }
    }
    
    async downloadModel(modelId) {
        try {
            const response = await fetch(`${API_BASE}/api/models/${modelId}/download`);
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
            await utils.apiCall(`/api/models/${modelId}`, { method: 'DELETE' });
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
    
    setupGlobalErrorHandling() {
        // Handle unhandled promise rejections
        window.addEventListener('unhandledrejection', (event) => {
            console.error('Unhandled promise rejection:', event.reason);
            event.preventDefault();
            
            // Hide any loading states
            if (typeof utils !== 'undefined') {
                utils.hideLoading();
            }
            
            // Show error notification
            if (typeof utils !== 'undefined') {
                utils.showError('An unexpected error occurred. Please try again.');
            }
        });
        
        // Handle global JavaScript errors
        window.addEventListener('error', (event) => {
            console.error('Global error:', event.error);
            
            // Hide any loading states
            if (typeof utils !== 'undefined') {
                utils.hideLoading();
            }
            
            // Don't show error for network errors (they're handled elsewhere)
            if (!event.message.includes('fetch') && !event.message.includes('NetworkError')) {
                if (typeof utils !== 'undefined') {
                    utils.showError('A JavaScript error occurred. Please refresh the page.');
                }
            }
        });
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