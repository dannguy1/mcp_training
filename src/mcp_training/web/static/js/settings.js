/**
 * Settings management JavaScript functionality
 */

class SettingsManager {
    constructor() {
        this.settings = {};
        this.originalSettings = {};
        this.init();
    }
    
    init() {
        this.setupEventListeners();
        this.loadSettings();
    }
    
    setupEventListeners() {
        // General form submission
        document.getElementById('generalSettingsForm')?.addEventListener('submit', (e) => {
            e.preventDefault();
            this.saveGeneralSettings();
        });
        
        // Training form submission
        document.getElementById('trainingSettingsForm')?.addEventListener('submit', (e) => {
            e.preventDefault();
            this.saveTrainingSettings();
        });
        
        // Storage form submission
        document.getElementById('storageSettingsForm')?.addEventListener('submit', (e) => {
            e.preventDefault();
            this.saveStorageSettings();
        });
        
        // Logging form submission
        document.getElementById('loggingSettingsForm')?.addEventListener('submit', (e) => {
            e.preventDefault();
            this.saveLoggingSettings();
        });
        
        // Security form submission
        document.getElementById('securitySettingsForm')?.addEventListener('submit', (e) => {
            e.preventDefault();
            this.saveSecuritySettings();
        });
        
        // Performance mode change
        document.getElementById('performanceMode')?.addEventListener('change', () => {
            this.applyPerformanceSettings();
        });
        
        // Live updates toggle
        document.getElementById('liveUpdates')?.addEventListener('change', () => {
            this.applyPerformanceSettings();
        });
        
        // Tab switching
        const tabButtons = document.querySelectorAll('[data-bs-toggle="tab"]');
        tabButtons.forEach(button => {
            button.addEventListener('shown.bs.tab', (e) => {
                const target = e.target.getAttribute('data-bs-target');
                this.loadTabData(target);
            });
        });
        
        // Action buttons
        const resetBtn = document.querySelector('[onclick="app.resetSettings()"]');
        const exportBtn = document.querySelector('[onclick="app.exportSettings()"]');
        const saveBtn = document.querySelector('[onclick="app.saveSettings()"]');
        
        if (resetBtn) {
            resetBtn.addEventListener('click', () => {
                this.resetSettings();
            });
        }
        
        if (exportBtn) {
            exportBtn.addEventListener('click', () => {
                this.exportSettings();
            });
        }
        
        if (saveBtn) {
            saveBtn.addEventListener('click', () => {
                this.saveSettings();
            });
        }
        
        // Import settings
        const settingsFile = document.getElementById('settingsFile');
        if (settingsFile) {
            settingsFile.addEventListener('change', (e) => {
                this.handleSettingsFileSelect(e);
            });
        }
        
        // Real-time validation
        this.setupRealTimeValidation();
    }
    
    setupRealTimeValidation() {
        // Validate numeric inputs
        const numericInputs = document.querySelectorAll('input[type="number"]');
        numericInputs.forEach(input => {
            input.addEventListener('input', () => {
                this.validateNumericInput(input);
            });
        });
        
        // Validate JSON inputs
        const jsonInputs = document.querySelectorAll('textarea[id*="Config"]');
        jsonInputs.forEach(input => {
            input.addEventListener('input', () => {
                this.validateJsonInput(input);
            });
        });
        
        // Validate required fields
        const requiredInputs = document.querySelectorAll('input[required], select[required]');
        requiredInputs.forEach(input => {
            input.addEventListener('blur', () => {
                this.validateRequiredField(input);
            });
        });
    }
    
    async loadSettings() {
        try {
            utils.showLoading();
            const settings = await utils.apiCall('/api/settings');
            this.settings = settings;
            this.originalSettings = JSON.parse(JSON.stringify(settings));
            this.populateSettingsForms();
        } catch (error) {
            console.error('Failed to load settings:', error);
            utils.showError('Failed to load settings', error);
            // Set default settings if API fails
            this.settings = this.getDefaultSettings();
            this.originalSettings = JSON.parse(JSON.stringify(this.settings));
            this.populateSettingsForms();
        } finally {
            utils.hideLoading();
        }
    }
    
    populateSettingsForms() {
        // General Settings
        this.setFormValue('serviceName', this.settings.general?.service_name);
        this.setFormValue('serviceVersion', this.settings.general?.version || '1.0.0');
        this.setFormValue('timezone', this.settings.general?.timezone || 'UTC');
        this.setFormValue('dateFormat', this.settings.general?.date_format || 'YYYY-MM-DD');
        this.setFormValue('autoRefresh', this.settings.general?.auto_refresh !== false);
        this.setFormValue('notifications', this.settings.general?.notifications !== false);
        
        // Performance Settings
        this.setFormValue('performanceMode', this.settings.general?.performance_mode || 'training');
        this.setFormValue('liveUpdates', this.settings.general?.live_updates === true);
        
        // Training Settings
        this.setFormValue('maxConcurrentJobs', this.settings.training?.max_concurrent_jobs || 3);
        this.setFormValue('defaultMaxIterations', this.settings.training?.default_max_iterations || 1000);
        this.setFormValue('defaultLearningRate', this.settings.training?.default_learning_rate || 0.01);
        this.setFormValue('jobTimeout', this.settings.training?.job_timeout || 24);
        this.setFormValue('defaultConfig', this.settings.training?.default_config || '{"algorithm": "random_forest", "n_estimators": 100, "max_depth": 10}');
        
        // Storage Settings
        this.setFormValue('modelsDir', this.settings.storage?.models_dir || 'models');
        this.setFormValue('exportsDir', this.settings.storage?.exports_dir || 'exports');
        this.setFormValue('logsDir', this.settings.storage?.logs_dir || 'logs');
        this.setFormValue('maxStorageGB', this.settings.storage?.max_storage_gb || 10);
        this.setFormValue('autoCleanup', this.settings.storage?.auto_cleanup !== false);
        this.setFormValue('retentionDays', this.settings.storage?.retention_days || 30);
        
        // Logging Settings
        this.setFormValue('logLevel', this.settings.logging?.level || 'INFO');
        this.setFormValue('logFormat', this.settings.logging?.format || 'structured');
        this.setFormValue('trainingOnlyLogging', this.settings.logging?.training_only !== false);
        this.setFormValue('logRotation', this.settings.logging?.rotation || 'daily');
        this.setFormValue('logRetention', this.settings.logging?.retention || 30);
        
        // Monitoring Settings
        this.setFormValue('enableMonitoring', this.settings.monitoring?.enabled !== false);
        this.setFormValue('prometheusPort', this.settings.monitoring?.prometheus_port || 9091);
        this.setFormValue('healthCheckInterval', this.settings.monitoring?.health_check_interval || 30);
        this.setFormValue('performanceMonitoring', this.settings.monitoring?.performance_monitoring !== false);
        
        // Security Settings
        this.setFormValue('apiKeyRequired', this.settings.security?.api_key_required === true);
        this.setFormValue('allowedIPs', this.settings.security?.allowed_ips?.join(', ') || '');
        this.setFormValue('rateLimiting', this.settings.security?.rate_limiting !== false);
        this.setFormValue('sslEnabled', this.settings.security?.ssl?.enabled === true);
        this.setFormValue('sslCertFile', this.settings.security?.ssl?.cert_file || '');
        this.setFormValue('sslKeyFile', this.settings.security?.ssl?.key_file || '');
        
        // Integration Settings
        this.setFormValue('mcpServiceEnabled', this.settings.integration?.mcp_service?.enabled !== false);
        this.setFormValue('mcpServiceUrl', this.settings.integration?.mcp_service?.api_url || 'http://localhost:8000');
        this.setFormValue('mcpServiceKey', this.settings.integration?.mcp_service?.api_key || '');
        this.setFormValue('autoDeploy', this.settings.integration?.deployment?.auto_deploy === true);
        this.setFormValue('deploymentDir', this.settings.integration?.deployment?.deployment_dir || '../mcp_service/models');
        this.setFormValue('backupExisting', this.settings.integration?.deployment?.backup_existing !== false);
    }
    
    setFormValue(elementId, value) {
        const element = document.getElementById(elementId);
        if (!element) return;
        
        if (element.type === 'checkbox') {
            element.checked = Boolean(value);
        } else {
            element.value = value;
        }
    }
    
    async saveSettings() {
        try {
            utils.showLoading();
            
            const settings = this.collectSettingsFromForms();
            
            await utils.apiCall('/api/settings', {
                method: 'PUT',
                body: JSON.stringify(settings),
                headers: { 'Content-Type': 'application/json' }
            });
            
            utils.showSuccess('Settings saved successfully');
            this.originalSettings = JSON.parse(JSON.stringify(settings));
            
        } catch (error) {
            utils.showError('Failed to save settings', error);
        } finally {
            utils.hideLoading();
        }
    }
    
    collectSettingsFromForms() {
        const settings = {
            general: {
                service_name: this.getFormValue('serviceName'),
                version: this.getFormValue('serviceVersion'),
                timezone: this.getFormValue('timezone'),
                date_format: this.getFormValue('dateFormat'),
                auto_refresh: this.getFormValue('autoRefresh'),
                notifications: this.getFormValue('notifications'),
                performance_mode: this.getFormValue('performanceMode'),
                live_updates: this.getFormValue('liveUpdates')
            },
            training: {
                max_concurrent_jobs: parseInt(this.getFormValue('maxConcurrentJobs')),
                default_max_iterations: parseInt(this.getFormValue('defaultMaxIterations')),
                default_learning_rate: parseFloat(this.getFormValue('defaultLearningRate')),
                job_timeout: parseInt(this.getFormValue('jobTimeout')),
                default_config: this.parseJsonValue(this.getFormValue('defaultConfig'))
            },
            storage: {
                models_dir: this.getFormValue('modelsDir'),
                exports_dir: this.getFormValue('exportsDir'),
                logs_dir: this.getFormValue('logsDir'),
                max_storage_gb: parseInt(this.getFormValue('maxStorageGB')),
                auto_cleanup: this.getFormValue('autoCleanup'),
                retention_days: parseInt(this.getFormValue('retentionDays'))
            },
            logging: {
                level: this.getFormValue('logLevel'),
                format: this.getFormValue('logFormat'),
                training_only: this.getFormValue('trainingOnlyLogging'),
                rotation: this.getFormValue('logRotation'),
                retention: parseInt(this.getFormValue('logRetention')),
                enable_monitoring: this.getFormValue('enableMonitoring'),
                prometheus_port: parseInt(this.getFormValue('prometheusPort')),
                health_check_interval: parseInt(this.getFormValue('healthCheckInterval')),
                performance_monitoring: this.getFormValue('performanceMonitoring')
            },
            security: {
                api_key_required: this.getFormValue('apiKeyRequired'),
                allowed_ips: this.getFormValue('allowedIPs').split(',').map(ip => ip.trim()),
                rate_limiting: this.getFormValue('rateLimiting'),
                ssl: {
                    enabled: this.getFormValue('sslEnabled'),
                    cert_file: this.getFormValue('sslCertFile'),
                    key_file: this.getFormValue('sslKeyFile')
                }
            },
            monitoring: {
                enabled: this.getFormValue('enableMonitoring'),
                prometheus_port: parseInt(this.getFormValue('prometheusPort')),
                health_check_interval: parseInt(this.getFormValue('healthCheckInterval')),
                performance_monitoring: this.getFormValue('performanceMonitoring')
            },
            integration: {
                mcp_service: {
                    enabled: this.getFormValue('mcpServiceEnabled'),
                    api_url: this.getFormValue('mcpServiceUrl'),
                    api_key: this.getFormValue('mcpServiceKey')
                },
                deployment: {
                    auto_deploy: this.getFormValue('autoDeploy'),
                    deployment_dir: this.getFormValue('deploymentDir'),
                    backup_existing: this.getFormValue('backupExisting')
                }
            }
        };
        
        return settings;
    }
    
    getFormValue(elementId) {
        const element = document.getElementById(elementId);
        if (!element) return null;
        
        if (element.type === 'checkbox') {
            return element.checked;
        } else {
            return element.value;
        }
    }
    
    parseJsonValue(value) {
        try {
            return JSON.parse(value);
        } catch (error) {
            return value;
        }
    }
    
    resetSettings() {
        if (!confirm('Are you sure you want to reset all settings to their default values? This action cannot be undone.')) {
            return;
        }
        
        this.settings = JSON.parse(JSON.stringify(this.originalSettings));
        this.populateSettingsForms();
        utils.showSuccess('Settings reset to original values');
    }
    
    exportSettings() {
        const settings = this.collectSettingsFromForms();
        const blob = new Blob([JSON.stringify(settings, null, 2)], { type: 'application/json' });
        const url = window.URL.createObjectURL(blob);
        utils.downloadFile(url, 'mcp-training-settings.json');
    }
    
    handleSettingsFileSelect(event) {
        const file = event.target.files[0];
        if (!file) return;
        
        if (!file.name.endsWith('.json')) {
            utils.showError('Please select a valid JSON settings file');
            return;
        }
        
        const reader = new FileReader();
        reader.onload = (e) => {
            try {
                const settings = JSON.parse(e.target.result);
                this.importSettings(settings);
            } catch (error) {
                utils.showError('Failed to parse settings file', error);
            }
        };
        reader.readAsText(file);
    }
    
    importSettings(settings) {
        if (!confirm('Are you sure you want to import these settings? This will overwrite your current settings.')) {
            return;
        }
        
        this.settings = settings;
        this.populateSettingsForms();
        utils.showSuccess('Settings imported successfully');
    }
    
    handleTabChange(tabId) {
        // Validate current tab before switching
        const currentTab = document.querySelector('.tab-pane.active');
        if (currentTab) {
            this.validateTab(currentTab.id);
        }
    }
    
    validateTab(tabId) {
        const form = document.querySelector(`#${tabId} form`);
        if (!form) return true;
        
        const inputs = form.querySelectorAll('input, select, textarea');
        let isValid = true;
        
        inputs.forEach(input => {
            if (!this.validateField(input)) {
                isValid = false;
            }
        });
        
        return isValid;
    }
    
    validateField(field) {
        // Remove previous validation classes
        field.classList.remove('is-valid', 'is-invalid');
        
        // Check required fields
        if (field.hasAttribute('required') && !field.value.trim()) {
            field.classList.add('is-invalid');
            return false;
        }
        
        // Validate numeric fields
        if (field.type === 'number') {
            return this.validateNumericInput(field);
        }
        
        // Validate JSON fields
        if (field.id && field.id.includes('Config')) {
            return this.validateJsonInput(field);
        }
        
        field.classList.add('is-valid');
        return true;
    }
    
    validateNumericInput(input) {
        const value = parseFloat(input.value);
        const min = parseFloat(input.min);
        const max = parseFloat(input.max);
        
        if (isNaN(value)) {
            input.classList.add('is-invalid');
            return false;
        }
        
        if (min !== undefined && value < min) {
            input.classList.add('is-invalid');
            return false;
        }
        
        if (max !== undefined && value > max) {
            input.classList.add('is-invalid');
            return false;
        }
        
        input.classList.add('is-valid');
        return true;
    }
    
    validateJsonInput(input) {
        try {
            JSON.parse(input.value);
            input.classList.add('is-valid');
            input.classList.remove('is-invalid');
            return true;
        } catch (error) {
            input.classList.add('is-invalid');
            input.classList.remove('is-valid');
            return false;
        }
    }
    
    validateRequiredField(field) {
        if (field.hasAttribute('required') && !field.value.trim()) {
            field.classList.add('is-invalid');
            return false;
        } else {
            field.classList.remove('is-invalid');
            field.classList.add('is-valid');
            return true;
        }
    }
    
    // Public methods for external access
    refresh() {
        this.loadSettings();
    }
    
    getSetting(path) {
        const keys = path.split('.');
        let value = this.settings;
        
        for (const key of keys) {
            if (value && typeof value === 'object' && key in value) {
                value = value[key];
            } else {
                return undefined;
            }
        }
        
        return value;
    }
    
    setSetting(path, value) {
        const keys = path.split('.');
        let current = this.settings;
        
        for (let i = 0; i < keys.length - 1; i++) {
            const key = keys[i];
            if (!(key in current) || typeof current[key] !== 'object') {
                current[key] = {};
            }
            current = current[key];
        }
        
        current[keys[keys.length - 1]] = value;
    }
    
    getDefaultSettings() {
        return {
            general: {
                service_name: "MCP Training Service",
                version: "1.0.0",
                timezone: "UTC",
                date_format: "YYYY-MM-DD",
                auto_refresh: true,
                notifications: true,
                performance_mode: "training",
                live_updates: false
            },
            training: {
                max_concurrent_jobs: 3,
                default_max_iterations: 1000,
                default_learning_rate: 0.01,
                job_timeout: 24,
                default_config: {"algorithm": "random_forest", "n_estimators": 100, "max_depth": 10}
            },
            storage: {
                models_dir: "models",
                exports_dir: "exports",
                logs_dir: "logs",
                max_storage_gb: 10,
                auto_cleanup: true,
                retention_days: 30
            },
            logging: {
                level: "INFO",
                format: "structured",
                training_only: false,
                rotation: "daily",
                retention: 30,
                enable_monitoring: false,
                prometheus_port: 9091,
                health_check_interval: 30,
                performance_monitoring: false
            },
            security: {
                api_key_required: false,
                allowed_ips: ["http://localhost:3000", "https://example.com"],
                rate_limiting: false,
                ssl: {
                    enabled: false,
                    cert_file: "",
                    key_file: ""
                }
            },
            monitoring: {
                enabled: false,
                prometheus_port: 9091,
                health_check_interval: 30,
                performance_monitoring: false
            },
            integration: {
                mcp_service: {
                    enabled: false,
                    api_url: "http://localhost:8000",
                    api_key: ""
                },
                deployment: {
                    auto_deploy: false,
                    deployment_dir: "../mcp_service/models",
                    backup_existing: false
                }
            }
        };
    }
    
    applyPerformanceSettings() {
        const performanceMode = this.getFormValue('performanceMode');
        const liveUpdates = this.getFormValue('liveUpdates');
        
        // Store settings in localStorage for immediate effect
        localStorage.setItem('performanceMode', performanceMode);
        localStorage.setItem('liveUpdatesEnabled', liveUpdates.toString());
        
        // Apply performance mode settings
        switch (performanceMode) {
            case 'training':
                // Minimal updates - best for training
                localStorage.setItem('autoRefreshEnabled', 'false');
                break;
            case 'balanced':
                // Moderate updates
                localStorage.setItem('autoRefreshEnabled', 'true');
                break;
            case 'responsive':
                // Frequent updates
                localStorage.setItem('autoRefreshEnabled', 'true');
                break;
        }
        
        // Show notification about performance mode change
        if (typeof utils !== 'undefined') {
            utils.showInfo(`Performance mode set to: ${performanceMode}. Changes will take effect on page refresh.`);
        }
    }
}

// Initialize settings manager when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    try {
        window.settingsManager = new SettingsManager();
    } catch (error) {
        console.error('Failed to initialize SettingsManager:', error);
        // Create a minimal settings manager to prevent errors
        window.settingsManager = {
            settings: {},
            loadSettings: () => Promise.resolve(),
            saveSettings: () => Promise.resolve(),
            resetSettings: () => {},
            exportSettings: () => {},
            importSettings: () => {}
        };
    }
});

// Global functions for use in HTML
window.saveSettings = () => window.settingsManager.saveSettings();
window.resetSettings = () => window.settingsManager.resetSettings();
window.exportSettings = () => window.settingsManager.exportSettings();
window.importSettings = () => window.settingsManager.importSettings(); 