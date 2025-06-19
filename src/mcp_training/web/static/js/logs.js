/**
 * Logs viewer JavaScript functionality
 */

class LogsManager {
    constructor() {
        this.logs = [];
        this.filteredLogs = [];
        this.liveMode = false;
        this.autoScroll = true;
        this.updateInterval = null;
        this.websocket = null;
        this.init();
    }
    
    init() {
        this.setupEventListeners();
        this.loadLogs();
        this.setupWebSocket();
    }
    
    setupEventListeners() {
        // Search and filter
        const searchInput = document.getElementById('searchInput');
        const levelFilter = document.getElementById('levelFilter');
        const serviceFilter = document.getElementById('serviceFilter');
        const startTime = document.getElementById('startTime');
        const endTime = document.getElementById('endTime');
        
        if (searchInput) {
            searchInput.addEventListener('input', utils.debounce(() => {
                this.filterLogs();
            }, 300));
        }
        
        if (levelFilter) {
            levelFilter.addEventListener('change', () => {
                this.filterLogs();
            });
        }
        
        if (serviceFilter) {
            serviceFilter.addEventListener('change', () => {
                this.filterLogs();
            });
        }
        
        if (startTime) {
            startTime.addEventListener('change', () => {
                this.filterLogs();
            });
        }
        
        if (endTime) {
            endTime.addEventListener('change', () => {
                this.filterLogs();
            });
        }
        
        // Control buttons
        const refreshLogsBtn = document.getElementById('refreshLogsBtn');
        const exportLogsBtn = document.getElementById('exportLogsBtn');
        const liveLogsBtn = document.getElementById('liveLogsBtn');
        const clearLogsBtn = document.getElementById('clearLogsBtn');
        const autoScrollBtn = document.getElementById('autoScrollBtn');
        
        if (refreshLogsBtn) {
            refreshLogsBtn.addEventListener('click', () => {
                this.loadLogs();
            });
        }
        
        if (exportLogsBtn) {
            exportLogsBtn.addEventListener('click', () => {
                this.showExportModal();
            });
        }
        
        if (liveLogsBtn) {
            liveLogsBtn.addEventListener('click', () => {
                this.toggleLiveMode();
            });
        }
        
        if (clearLogsBtn) {
            clearLogsBtn.addEventListener('click', () => {
                this.clearLogs();
            });
        }
        
        if (autoScrollBtn) {
            autoScrollBtn.addEventListener('click', () => {
                this.toggleAutoScroll();
            });
        }
        
        // Copy log button
        const copyLogBtn = document.getElementById('copyLogBtn');
        if (copyLogBtn) {
            copyLogBtn.addEventListener('click', () => {
                this.copyLogDetails();
            });
        }
        
        // Table scroll for auto-scroll detection
        const tableContainer = document.querySelector('.table-responsive');
        if (tableContainer) {
            tableContainer.addEventListener('scroll', () => {
                this.checkAutoScroll();
            });
        }
    }
    
    async loadLogs() {
        try {
            utils.showLoading();
            
            const params = new URLSearchParams();
            const startTime = document.getElementById('startTime')?.value;
            const endTime = document.getElementById('endTime')?.value;
            const level = document.getElementById('levelFilter')?.value;
            const service = document.getElementById('serviceFilter')?.value;
            
            if (startTime) params.append('start_time', startTime);
            if (endTime) params.append('end_time', endTime);
            if (level) params.append('level', level);
            if (service) params.append('service', service);
            
            const response = await utils.apiCall(`/logs/?${params}`);
            
            // Extract logs array from response
            this.logs = response.logs || [];
            this.filteredLogs = [...this.logs];
            this.updateLogsTable();
            this.updateLogStatistics();
        } catch (error) {
            utils.showError('Failed to load logs', error);
        } finally {
            utils.hideLoading();
        }
    }
    
    updateLogsTable() {
        const tableBody = document.querySelector('#logsTable tbody');
        if (!tableBody) return;
        
        if (this.filteredLogs.length === 0) {
            tableBody.innerHTML = `
                <tr>
                    <td colspan="5" class="text-center text-muted py-4">
                        <i class="bi bi-file-text fs-1 mb-3"></i>
                        <p>No logs found</p>
                    </td>
                </tr>
            `;
            return;
        }
        
        tableBody.innerHTML = this.filteredLogs.map(log => `
            <tr data-log-id="${log.id}">
                <td>${utils.formatDateTime(log.timestamp)}</td>
                <td>
                    <span class="badge bg-${utils.getStatusColor(log.level)}">
                        ${log.level.toUpperCase()}
                    </span>
                </td>
                <td>${log.service}</td>
                <td class="text-truncate-2">${this.escapeHtml(log.message)}</td>
                <td class="table-actions">
                    <button class="btn btn-outline-info btn-sm" onclick="logsManager.viewLogDetails('${log.id}')" 
                            title="View Details">
                        <i class="bi bi-eye"></i>
                    </button>
                </td>
            </tr>
        `).join('');
        
        // Auto-scroll to bottom if enabled
        if (this.autoScroll) {
            this.scrollToBottom();
        }
    }
    
    updateLogStatistics() {
        const stats = {
            total_logs: this.logs.length,
            info_logs: this.logs.filter(log => log.level === 'INFO').length,
            warning_logs: this.logs.filter(log => log.level === 'WARNING').length,
            error_logs: this.logs.filter(log => log.level === 'ERROR').length,
            debug_logs: this.logs.filter(log => log.level === 'DEBUG').length
        };
        
        // Update statistics display
        Object.keys(stats).forEach(key => {
            const element = document.querySelector(`[data-status="${key}"]`);
            if (element) {
                element.textContent = stats[key];
            }
        });
    }
    
    filterLogs() {
        const searchTerm = document.getElementById('searchInput')?.value.toLowerCase() || '';
        const levelFilter = document.getElementById('levelFilter')?.value || '';
        const serviceFilter = document.getElementById('serviceFilter')?.value || '';
        const startTime = document.getElementById('startTime')?.value;
        const endTime = document.getElementById('endTime')?.value;
        
        this.filteredLogs = this.logs.filter(log => {
            const matchesSearch = !searchTerm || 
                log.message.toLowerCase().includes(searchTerm) ||
                log.service.toLowerCase().includes(searchTerm) ||
                log.level.toLowerCase().includes(searchTerm);
            
            const matchesLevel = !levelFilter || log.level === levelFilter;
            const matchesService = !serviceFilter || log.service === serviceFilter;
            
            let matchesTime = true;
            if (startTime) {
                matchesTime = matchesTime && new Date(log.timestamp) >= new Date(startTime);
            }
            if (endTime) {
                matchesTime = matchesTime && new Date(log.timestamp) <= new Date(endTime);
            }
            
            return matchesSearch && matchesLevel && matchesService && matchesTime;
        });
        
        this.updateLogsTable();
    }
    
    async viewLogDetails(logId) {
        try {
            const log = await utils.apiCall(`/logs/${logId}`);
            this.showLogDetailsModal(log);
        } catch (error) {
            utils.showError('Failed to load log details', error);
        }
    }
    
    showLogDetailsModal(log) {
        const modal = new bootstrap.Modal(document.getElementById('logDetailsModal'));
        const content = document.getElementById('logDetailsContent');
        
        content.innerHTML = `
            <div class="row">
                <div class="col-md-6">
                    <h6>Log Information</h6>
                    <table class="table table-sm">
                        <tr><td>ID:</td><td>${log.id}</td></tr>
                        <tr><td>Timestamp:</td><td>${utils.formatDateTime(log.timestamp)}</td></tr>
                        <tr><td>Level:</td><td><span class="badge bg-${utils.getStatusColor(log.level)}">${log.level.toUpperCase()}</span></td></tr>
                        <tr><td>Service:</td><td>${log.service}</td></tr>
                        <tr><td>Logger:</td><td>${log.logger || 'N/A'}</td></tr>
                    </table>
                </div>
                <div class="col-md-6">
                    <h6>Additional Details</h6>
                    <table class="table table-sm">
                        <tr><td>Request ID:</td><td><code>${log.request_id || 'N/A'}</code></td></tr>
                        <tr><td>User ID:</td><td>${log.user_id || 'N/A'}</td></tr>
                        <tr><td>IP Address:</td><td>${log.ip_address || 'N/A'}</td></tr>
                        <tr><td>User Agent:</td><td>${log.user_agent || 'N/A'}</td></tr>
                        <tr><td>Duration:</td><td>${log.duration ? `${log.duration}ms` : 'N/A'}</td></tr>
                    </table>
                </div>
            </div>
            <div class="row mt-3">
                <div class="col-12">
                    <h6>Message</h6>
                    <pre class="bg-light p-3 rounded">${this.escapeHtml(log.message)}</pre>
                </div>
            </div>
            ${log.exception ? `
                <div class="row mt-3">
                    <div class="col-12">
                        <h6>Exception</h6>
                        <pre class="bg-danger text-white p-3 rounded">${this.escapeHtml(log.exception)}</pre>
                    </div>
                </div>
            ` : ''}
            ${log.extra ? `
                <div class="row mt-3">
                    <div class="col-12">
                        <h6>Additional Data</h6>
                        <pre class="bg-light p-3 rounded">${JSON.stringify(log.extra, null, 2)}</pre>
                    </div>
                </div>
            ` : ''}
        `;
        
        modal.show();
    }
    
    copyLogDetails() {
        const content = document.getElementById('logDetailsContent');
        if (content) {
            const text = content.textContent || content.innerText;
            utils.copyToClipboard(text);
        }
    }
    
    showExportModal() {
        const modal = new bootstrap.Modal(document.getElementById('exportLogsModal'));
        
        // Set default values
        const exportStartTime = document.getElementById('exportStartTime');
        const exportEndTime = document.getElementById('exportEndTime');
        const startTime = document.getElementById('startTime')?.value;
        const endTime = document.getElementById('endTime')?.value;
        
        if (exportStartTime && startTime) {
            exportStartTime.value = startTime;
        }
        if (exportEndTime && endTime) {
            exportEndTime.value = endTime;
        }
        
        modal.show();
    }
    
    async exportLogs() {
        try {
            const format = document.getElementById('exportFormat')?.value || 'json';
            const level = document.getElementById('exportLevel')?.value || '';
            const startTime = document.getElementById('exportStartTime')?.value;
            const endTime = document.getElementById('exportEndTime')?.value;
            const includeMetadata = document.getElementById('includeMetadata')?.checked || false;
            
            const params = new URLSearchParams({
                format,
                include_metadata: includeMetadata
            });
            
            if (level) params.append('level', level);
            if (startTime) params.append('start_time', startTime);
            if (endTime) params.append('end_time', endTime);
            
            const response = await fetch(`${API_BASE}/logs/export?${params}`);
            if (response.ok) {
                const blob = await response.blob();
                const url = window.URL.createObjectURL(blob);
                const filename = `logs-export-${new Date().toISOString().split('T')[0]}.${format}`;
                utils.downloadFile(url, filename);
                
                const modal = bootstrap.Modal.getInstance(document.getElementById('exportLogsModal'));
                modal.hide();
            } else {
                throw new Error('Failed to export logs');
            }
        } catch (error) {
            utils.showError('Failed to export logs', error);
        }
    }
    
    async clearLogs() {
        if (!confirm('Are you sure you want to clear all logs? This action cannot be undone.')) {
            return;
        }
        
        try {
            await utils.apiCall('/logs/', { method: 'DELETE' });
            utils.showSuccess('Logs cleared successfully');
            this.loadLogs();
        } catch (error) {
            utils.showError('Failed to clear logs', error);
        }
    }
    
    toggleLiveMode() {
        this.liveMode = !this.liveMode;
        const liveLogsBtn = document.getElementById('liveLogsBtn');
        const liveStatus = document.getElementById('liveStatus');
        
        if (this.liveMode) {
            liveLogsBtn.innerHTML = '<i class="bi bi-stop me-2"></i>Stop Live';
            liveLogsBtn.className = 'btn btn-danger';
            liveStatus.textContent = 'ON';
            liveStatus.className = 'mb-0 text-success';
            
            // Start live updates
            this.startLiveUpdates();
        } else {
            liveLogsBtn.innerHTML = '<i class="bi bi-play me-2"></i>Live Logs';
            liveLogsBtn.className = 'btn btn-primary';
            liveStatus.textContent = 'OFF';
            liveStatus.className = 'mb-0 text-muted';
            
            // Stop live updates
            this.stopLiveUpdates();
        }
    }
    
    toggleAutoScroll() {
        this.autoScroll = !this.autoScroll;
        const autoScrollBtn = document.getElementById('autoScrollBtn');
        
        if (this.autoScroll) {
            autoScrollBtn.innerHTML = '<i class="bi bi-arrow-down me-1"></i>Auto-scroll';
            autoScrollBtn.className = 'btn btn-outline-success btn-sm';
            this.scrollToBottom();
        } else {
            autoScrollBtn.innerHTML = '<i class="bi bi-arrow-down me-1"></i>Auto-scroll';
            autoScrollBtn.className = 'btn btn-outline-secondary btn-sm';
        }
    }
    
    setupWebSocket() {
        try {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = `${protocol}//${window.location.host}/ws/logs`;
            
            this.websocket = new WebSocket(wsUrl);
            
            this.websocket.onopen = () => {
                console.log('Logs WebSocket connected');
            };
            
            this.websocket.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    this.handleLogMessage(data);
                } catch (error) {
                    console.error('Failed to parse log message:', error);
                }
            };
            
            this.websocket.onerror = (error) => {
                console.error('Logs WebSocket error:', error);
            };
            
            this.websocket.onclose = () => {
                console.log('Logs WebSocket disconnected');
                // Attempt to reconnect after 5 seconds
                setTimeout(() => {
                    this.setupWebSocket();
                }, 5000);
            };
        } catch (error) {
            console.error('Failed to setup logs WebSocket:', error);
        }
    }
    
    handleLogMessage(data) {
        if (data.type === 'log') {
            const log = data.data;
            this.logs.unshift(log); // Add to beginning
            this.filteredLogs.unshift(log);
            
            // Keep only last 1000 logs
            if (this.logs.length > 1000) {
                this.logs.pop();
                this.filteredLogs.pop();
            }
            
            this.updateLogsTable();
            this.updateLogStatistics();
        }
    }
    
    startLiveUpdates() {
        // Refresh logs every 5 seconds in live mode
        this.updateInterval = setInterval(() => {
            this.loadLogs();
        }, 5000);
    }
    
    stopLiveUpdates() {
        if (this.updateInterval) {
            clearInterval(this.updateInterval);
            this.updateInterval = null;
        }
    }
    
    scrollToBottom() {
        const tableContainer = document.querySelector('.table-responsive');
        if (tableContainer) {
            tableContainer.scrollTop = tableContainer.scrollHeight;
        }
    }
    
    checkAutoScroll() {
        const tableContainer = document.querySelector('.table-responsive');
        if (tableContainer) {
            const isAtBottom = tableContainer.scrollTop + tableContainer.clientHeight >= tableContainer.scrollHeight - 10;
            this.autoScroll = isAtBottom;
            
            const autoScrollBtn = document.getElementById('autoScrollBtn');
            if (autoScrollBtn) {
                if (this.autoScroll) {
                    autoScrollBtn.className = 'btn btn-outline-success btn-sm';
                } else {
                    autoScrollBtn.className = 'btn btn-outline-secondary btn-sm';
                }
            }
        }
    }
    
    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
    
    destroy() {
        this.stopLiveUpdates();
        
        if (this.websocket) {
            this.websocket.close();
        }
    }
}

// Initialize logs manager when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.logsManager = new LogsManager();
});

// Global functions for use in HTML
window.exportLogs = () => window.logsManager.exportLogs(); 