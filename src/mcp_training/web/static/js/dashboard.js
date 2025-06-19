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
    }
    
    initSystemMetricsChart() {
        const ctx = document.getElementById('systemMetricsChart');
        if (!ctx) return;
        
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
    }
    
    async updateDashboard() {
        try {
            const [status, trainingJobs, models] = await Promise.all([
                utils.apiCall('/health/status'),
                utils.apiCall('/training/jobs'),
                utils.apiCall('/models')
            ]);
            
            this.updateStatusCards(status);
            this.updateTrainingChart(trainingJobs);
            this.updateActivityList(trainingJobs);
            this.updateSystemMetrics(status);
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
            const data = await utils.apiCall(`/training/chart-data?${params}`);
            
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
        // Refresh dashboard every 30 seconds
        this.updateInterval = setInterval(() => {
            this.updateDashboard();
        }, 30000);
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
    window.dashboard = new DashboardManager();
});

// Global functions for use in HTML
window.refreshDashboard = () => window.dashboard.refresh(); 