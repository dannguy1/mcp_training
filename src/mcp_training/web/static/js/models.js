/**
 * Models management JavaScript functionality
 */

class ModelsManager {
    constructor() {
        this.models = [];
        this.filteredModels = [];
        this.selectedModels = new Set();
        this.currentModelId = null;
        this.updateInterval = null;
        this.init();
    }
    
    init() {
        this.setupEventListeners();
        this.loadModels();
        this.startAutoRefresh();
    }
    
    setupEventListeners() {
        // Search and filter
        const searchInput = document.getElementById('searchInput');
        const statusFilter = document.getElementById('statusFilter');
        const versionFilter = document.getElementById('versionFilter');
        
        if (searchInput) {
            searchInput.addEventListener('input', utils.debounce(() => {
                this.filterModels();
            }, 300));
        }
        
        if (statusFilter) {
            statusFilter.addEventListener('change', () => {
                this.filterModels();
            });
        }
        
        if (versionFilter) {
            versionFilter.addEventListener('change', () => {
                this.filterModels();
            });
        }
        
        // Select all checkbox
        const selectAll = document.getElementById('selectAll');
        if (selectAll) {
            selectAll.addEventListener('change', (e) => {
                this.toggleSelectAll(e.target.checked);
            });
        }
        
        // File upload area
        const modelFileUploadArea = document.getElementById('modelFileUploadArea');
        const modelFile = document.getElementById('modelFile');
        
        if (modelFileUploadArea && modelFile) {
            modelFileUploadArea.addEventListener('click', () => modelFile.click());
            modelFileUploadArea.addEventListener('dragover', this.handleDragOver.bind(this));
            modelFileUploadArea.addEventListener('drop', this.handleFileDrop.bind(this));
            modelFile.addEventListener('change', this.handleFileSelect.bind(this));
        }
        
        // Form validation
        const uploadModelForm = document.getElementById('uploadModelForm');
        if (uploadModelForm) {
            uploadModelForm.addEventListener('submit', (e) => {
                e.preventDefault();
                this.uploadModel();
            });
        }
        
        // Modal events
        const uploadModelModal = document.getElementById('uploadModelModal');
        if (uploadModelModal) {
            uploadModelModal.addEventListener('hidden.bs.modal', () => {
                this.resetUploadForm();
            });
        }
        
        // Confirm delete
        const confirmDeleteBtn = document.getElementById('confirmDeleteBtn');
        if (confirmDeleteBtn) {
            confirmDeleteBtn.addEventListener('click', () => {
                this.confirmBulkDelete();
            });
        }
    }
    
    async loadModels() {
        try {
            utils.showLoading();
            const models = await utils.apiCall('/models');
            this.models = models;
            this.filteredModels = [...models];
            this.updateModelsTable();
            this.updateModelStatistics();
        } catch (error) {
            utils.showError('Failed to load models', error);
        } finally {
            utils.hideLoading();
        }
    }
    
    updateModelsTable() {
        const tableBody = document.querySelector('#modelsTable tbody');
        if (!tableBody) return;
        
        if (this.filteredModels.length === 0) {
            tableBody.innerHTML = `
                <tr>
                    <td colspan="9" class="text-center text-muted py-4">
                        <i class="bi bi-box fs-1 mb-3"></i>
                        <p>No models found</p>
                    </td>
                </tr>
            `;
            return;
        }
        
        tableBody.innerHTML = this.filteredModels.map(model => `
            <tr data-model-id="${model.id}">
                <td>
                    <input type="checkbox" class="form-check-input model-checkbox" 
                           value="${model.id}" onchange="modelsManager.toggleModelSelection('${model.id}')">
                </td>
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
                <td>
                    ${model.performance ? `
                        <span class="badge bg-success">${model.performance.accuracy || 0}%</span>
                    ` : '<span class="text-muted">N/A</span>'}
                </td>
                <td class="table-actions">
                    <div class="btn-group btn-group-sm">
                        <button class="btn btn-outline-info" onclick="modelsManager.viewModelDetails('${model.id}')" 
                                title="View Details">
                            <i class="bi bi-eye"></i>
                        </button>
                        ${model.status === 'ready' ? `
                            <button class="btn btn-outline-success" onclick="modelsManager.deployModel('${model.id}')" 
                                    title="Deploy Model">
                                <i class="bi bi-rocket"></i>
                            </button>
                        ` : ''}
                        <button class="btn btn-outline-secondary" onclick="modelsManager.downloadModel('${model.id}')" 
                                title="Download Model">
                            <i class="bi bi-download"></i>
                        </button>
                        <button class="btn btn-outline-danger" onclick="modelsManager.deleteModel('${model.id}')" 
                                title="Delete Model">
                            <i class="bi bi-trash"></i>
                        </button>
                    </div>
                </td>
            </tr>
        `).join('');
    }
    
    updateModelStatistics() {
        const stats = {
            total_models: this.models.length,
            deployed_models: this.models.filter(model => model.status === 'deployed').length,
            storage_used: this.calculateStorageUsed(),
            latest_version: this.getLatestVersion()
        };
        
        // Update statistics display
        Object.keys(stats).forEach(key => {
            const element = document.querySelector(`[data-status="${key}"]`);
            if (element) {
                element.textContent = stats[key];
            }
        });
    }
    
    calculateStorageUsed() {
        const totalBytes = this.models.reduce((sum, model) => sum + (model.size || 0), 0);
        return utils.formatFileSize(totalBytes);
    }
    
    getLatestVersion() {
        if (this.models.length === 0) return 'v1.0.0';
        
        const versions = this.models.map(model => model.version);
        const sortedVersions = versions.sort((a, b) => {
            const aParts = a.split('.').map(Number);
            const bParts = b.split('.').map(Number);
            
            for (let i = 0; i < Math.max(aParts.length, bParts.length); i++) {
                const aPart = aParts[i] || 0;
                const bPart = bParts[i] || 0;
                if (aPart !== bPart) {
                    return bPart - aPart; // Descending order
                }
            }
            return 0;
        });
        
        return `v${sortedVersions[0]}`;
    }
    
    filterModels() {
        const searchTerm = document.getElementById('searchInput')?.value.toLowerCase() || '';
        const statusFilter = document.getElementById('statusFilter')?.value || '';
        const versionFilter = document.getElementById('versionFilter')?.value || '';
        
        this.filteredModels = this.models.filter(model => {
            const matchesSearch = !searchTerm || 
                model.id.toString().includes(searchTerm) ||
                model.name.toLowerCase().includes(searchTerm) ||
                model.version.toLowerCase().includes(searchTerm) ||
                (model.description && model.description.toLowerCase().includes(searchTerm));
            
            const matchesStatus = !statusFilter || model.status === statusFilter;
            
            let matchesVersion = true;
            if (versionFilter === 'latest') {
                const latestVersion = this.getLatestVersion().substring(1); // Remove 'v' prefix
                matchesVersion = model.version === latestVersion;
            } else if (versionFilter === 'stable') {
                matchesVersion = !model.version.includes('alpha') && !model.version.includes('beta');
            }
            
            return matchesSearch && matchesStatus && matchesVersion;
        });
        
        this.updateModelsTable();
        this.updateSelectAllState();
    }
    
    toggleSelectAll(checked) {
        const checkboxes = document.querySelectorAll('.model-checkbox');
        checkboxes.forEach(checkbox => {
            checkbox.checked = checked;
            this.toggleModelSelection(checkbox.value, checked);
        });
    }
    
    toggleModelSelection(modelId, checked = null) {
        const checkbox = document.querySelector(`input[value="${modelId}"]`);
        const isChecked = checked !== null ? checked : checkbox.checked;
        
        if (isChecked) {
            this.selectedModels.add(modelId);
        } else {
            this.selectedModels.delete(modelId);
        }
        
        this.updateSelectAllState();
    }
    
    updateSelectAllState() {
        const selectAll = document.getElementById('selectAll');
        const checkboxes = document.querySelectorAll('.model-checkbox');
        
        if (selectAll && checkboxes.length > 0) {
            const checkedCount = Array.from(checkboxes).filter(cb => cb.checked).length;
            selectAll.checked = checkedCount === checkboxes.length;
            selectAll.indeterminate = checkedCount > 0 && checkedCount < checkboxes.length;
        }
    }
    
    async uploadModel() {
        try {
            utils.showLoading();
            
            const formData = new FormData();
            const fileInput = document.getElementById('modelFile');
            const modelName = document.getElementById('modelName')?.value;
            const modelVersion = document.getElementById('modelVersion')?.value;
            const modelType = document.getElementById('modelType')?.value;
            const modelFramework = document.getElementById('modelFramework')?.value;
            const modelDescription = document.getElementById('modelDescription')?.value;
            const autoDeploy = document.getElementById('autoDeploy')?.checked;
            
            if (!fileInput.files.length) {
                throw new Error('Please select a model file');
            }
            
            formData.append('model_file', fileInput.files[0]);
            formData.append('name', modelName || 'Unnamed Model');
            formData.append('version', modelVersion || '1.0.0');
            formData.append('type', modelType || 'custom');
            formData.append('framework', modelFramework || 'custom');
            formData.append('description', modelDescription || '');
            formData.append('auto_deploy', autoDeploy || false);
            
            const response = await utils.apiCall('/models', {
                method: 'POST',
                body: formData,
                headers: {} // Let browser set content-type for FormData
            });
            
            utils.showSuccess('Model uploaded successfully');
            
            // Close modal and refresh data
            const modal = bootstrap.Modal.getInstance(document.getElementById('uploadModelModal'));
            modal.hide();
            
            this.loadModels();
            
        } catch (error) {
            utils.showError('Failed to upload model', error);
        } finally {
            utils.hideLoading();
        }
    }
    
    async deployModel(modelId) {
        try {
            const deploymentName = prompt('Enter deployment name:');
            if (!deploymentName) return;
            
            const deploymentConfig = {
                name: deploymentName,
                environment: 'production',
                replicas: 2,
                resources: {
                    cpu: '500m',
                    memory: '1Gi'
                }
            };
            
            await utils.apiCall(`/models/${modelId}/deploy`, {
                method: 'POST',
                body: JSON.stringify(deploymentConfig),
                headers: { 'Content-Type': 'application/json' }
            });
            
            utils.showSuccess('Model deployed successfully');
            this.loadModels();
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
            this.loadModels();
        } catch (error) {
            utils.showError('Failed to delete model', error);
        }
    }
    
    async viewModelDetails(modelId) {
        try {
            const model = await utils.apiCall(`/models/${modelId}`);
            this.showModelDetailsModal(model);
        } catch (error) {
            utils.showError('Failed to load model details', error);
        }
    }
    
    showModelDetailsModal(model) {
        const modal = new bootstrap.Modal(document.getElementById('modelDetailsModal'));
        const content = document.getElementById('modelDetailsContent');
        
        content.innerHTML = `
            <div class="row">
                <div class="col-md-6">
                    <h6>Model Information</h6>
                    <table class="table table-sm">
                        <tr><td>ID:</td><td>${model.id}</td></tr>
                        <tr><td>Name:</td><td>${model.name}</td></tr>
                        <tr><td>Version:</td><td>${model.version}</td></tr>
                        <tr><td>Status:</td><td><span class="badge bg-${utils.getStatusColor(model.status)}">${model.status}</span></td></tr>
                        <tr><td>Type:</td><td>${model.type || 'N/A'}</td></tr>
                        <tr><td>Framework:</td><td>${model.framework || 'N/A'}</td></tr>
                    </table>
                </div>
                <div class="col-md-6">
                    <h6>Technical Details</h6>
                    <table class="table table-sm">
                        <tr><td>Size:</td><td>${utils.formatFileSize(model.size || 0)}</td></tr>
                        <tr><td>Created:</td><td>${utils.formatDateTime(model.created_at)}</td></tr>
                        <tr><td>Updated:</td><td>${utils.formatDateTime(model.updated_at)}</td></tr>
                        <tr><td>File Path:</td><td>${model.file_path || 'N/A'}</td></tr>
                        <tr><td>Checksum:</td><td><code>${model.checksum || 'N/A'}</code></td></tr>
                    </table>
                </div>
            </div>
            ${model.description ? `
                <div class="row mt-3">
                    <div class="col-12">
                        <h6>Description</h6>
                        <p>${model.description}</p>
                    </div>
                </div>
            ` : ''}
            ${model.performance ? `
                <div class="row mt-3">
                    <div class="col-12">
                        <h6>Performance Metrics</h6>
                        <div class="row">
                            <div class="col-md-3">
                                <div class="text-center">
                                    <h4 class="text-success">${model.performance.accuracy || 0}%</h4>
                                    <small class="text-muted">Accuracy</small>
                                </div>
                            </div>
                            <div class="col-md-3">
                                <div class="text-center">
                                    <h4 class="text-info">${model.performance.precision || 0}%</h4>
                                    <small class="text-muted">Precision</small>
                                </div>
                            </div>
                            <div class="col-md-3">
                                <div class="text-center">
                                    <h4 class="text-warning">${model.performance.recall || 0}%</h4>
                                    <small class="text-muted">Recall</small>
                                </div>
                            </div>
                            <div class="col-md-3">
                                <div class="text-center">
                                    <h4 class="text-primary">${model.performance.f1_score || 0}%</h4>
                                    <small class="text-muted">F1 Score</small>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            ` : ''}
            ${model.metadata ? `
                <div class="row mt-3">
                    <div class="col-12">
                        <h6>Model Metadata</h6>
                        <pre class="bg-light p-3 rounded">${JSON.stringify(model.metadata, null, 2)}</pre>
                    </div>
                </div>
            ` : ''}
        `;
        
        // Setup action buttons
        const deployBtn = document.getElementById('deployModelBtn');
        const downloadBtn = document.getElementById('downloadModelBtn');
        
        if (deployBtn) {
            deployBtn.onclick = () => this.deployModel(model.id);
            deployBtn.style.display = model.status === 'ready' ? 'inline-block' : 'none';
        }
        
        if (downloadBtn) {
            downloadBtn.onclick = () => this.downloadModel(model.id);
        }
        
        modal.show();
    }
    
    async bulkDelete() {
        if (this.selectedModels.size === 0) {
            utils.showWarning('Please select models to delete');
            return;
        }
        
        if (!confirm(`Are you sure you want to delete ${this.selectedModels.size} models?`)) {
            return;
        }
        
        try {
            const promises = Array.from(this.selectedModels).map(modelId =>
                utils.apiCall(`/models/${modelId}`, { method: 'DELETE' })
            );
            
            await Promise.all(promises);
            utils.showSuccess(`Successfully deleted ${this.selectedModels.size} models`);
            this.selectedModels.clear();
            this.loadModels();
        } catch (error) {
            utils.showError('Failed to delete some models', error);
        }
    }
    
    async exportModels() {
        const data = this.filteredModels.map(model => ({
            id: model.id,
            name: model.name,
            version: model.version,
            status: model.status,
            type: model.type,
            framework: model.framework,
            size: model.size,
            created_at: model.created_at,
            performance: model.performance
        }));
        
        const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
        const url = window.URL.createObjectURL(blob);
        utils.downloadFile(url, 'models-export.json');
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
        // Validate file type
        const allowedTypes = ['.joblib', '.pkl', '.model'];
        const isValidType = allowedTypes.some(type => file.name.endsWith(type));
        
        if (!isValidType) {
            utils.showError('Please select a valid model file (.joblib, .pkl, .model)');
            return;
        }
        
        // Update file info
        this.updateFileInfo(file);
        
        // Enable form submission
        const submitBtn = document.querySelector('#uploadModelModal .btn-primary');
        if (submitBtn) {
            submitBtn.disabled = false;
        }
    }
    
    updateFileInfo(file) {
        const fileInfo = document.getElementById('modelFileInfo');
        const fileName = document.getElementById('modelFileName');
        const fileSize = document.getElementById('modelFileSize');
        
        if (fileName) fileName.textContent = file.name;
        if (fileSize) fileSize.textContent = `(${utils.formatFileSize(file.size)})`;
        if (fileInfo) fileInfo.style.display = 'block';
    }
    
    resetUploadForm() {
        const form = document.getElementById('uploadModelForm');
        if (form) {
            form.reset();
        }
        
        const fileInfo = document.getElementById('modelFileInfo');
        if (fileInfo) {
            fileInfo.style.display = 'none';
        }
        
        const submitBtn = document.querySelector('#uploadModelModal .btn-primary');
        if (submitBtn) {
            submitBtn.disabled = true;
        }
    }
    
    startAutoRefresh() {
        // Refresh every 30 seconds
        this.updateInterval = setInterval(() => {
            this.loadModels();
        }, 30000);
    }
    
    stopAutoRefresh() {
        if (this.updateInterval) {
            clearInterval(this.updateInterval);
            this.updateInterval = null;
        }
    }
    
    // Public methods
    refresh() {
        this.loadModels();
    }
    
    destroy() {
        this.stopAutoRefresh();
    }
}

// Initialize models manager when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.modelsManager = new ModelsManager();
});

// Global functions for use in HTML
window.uploadModel = () => window.modelsManager.uploadModel(); 