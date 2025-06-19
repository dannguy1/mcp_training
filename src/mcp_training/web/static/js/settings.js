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
        // Settings tabs - updated for horizontal tabs
        const settingsTabs = document.querySelectorAll('[data-bs-toggle="tab"]');
        settingsTabs.forEach(tab => {
            tab.addEventListener('shown.bs.tab', (e) => {
                this.handleTabChange(e.target.getAttribute('data-bs-target'));
            });
        });
        
        // Form validation
        const forms = document.querySelectorAll('form[id$="SettingsForm"]');
        forms.forEach(form => {
            form.addEventListener('submit', (e) => {
                e.preventDefault();
                this.saveSettings();
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
            const settings = await utils.apiCall('/settings');
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
        this.setFormValue('logFile', this.settings.logging?.file || 'logs/mcp_training.log');
        this.setFormValue('maxLogSizeMB', this.settings.logging?.max_size_mb || 100);
        this.setFormValue('logToConsole', this.settings.logging?.console !== false);
        this.setFormValue('logToFile', this.settings.logging?.file_enabled !== false);
        
        // Security Settings
        this.setFormValue('authEnabled', this.settings.security?.auth_enabled ? 'true' : 'false');
        this.setFormValue('apiKey', this.settings.security?.api_key || '');
        this.setFormValue('corsOrigins', this.settings.security?.cors_origins || '');
        this.setFormValue('rateLimit', this.settings.security?.rate_limit || 100);
        this.setFormValue('httpsOnly', this.settings.security?.https_only || false);
        this.setFormValue('secureHeaders', this.settings.security?.secure_headers !== false);
        
        // Advanced Settings
        this.setFormValue('debugMode', this.settings.advanced?.debug_mode ? 'true' : 'false');
        this.setFormValue('performanceMonitoring', this.settings.advanced?.performance_monitoring !== false ? 'true' : 'false');
        this.setFormValue('websocketEnabled', this.settings.advanced?.websocket_enabled !== false ? 'true' : 'false');
        this.setFormValue('autoBackup', this.settings.advanced?.auto_backup ? 'true' : 'false');
        this.setFormValue('customConfig', this.settings.advanced?.custom_config || '{"custom_setting": "value"}');
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
            
            await utils.apiCall('/settings', {
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
                notifications: this.getFormValue('notifications')
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
                file: this.getFormValue('logFile'),
                max_size_mb: parseInt(this.getFormValue('maxLogSizeMB')),
                console: this.getFormValue('logToConsole'),
                file_enabled: this.getFormValue('logToFile')
            },
            security: {
                auth_enabled: this.getFormValue('authEnabled') === 'true',
                api_key: this.getFormValue('apiKey'),
                cors_origins: this.getFormValue('corsOrigins'),
                rate_limit: parseInt(this.getFormValue('rateLimit')),
                https_only: this.getFormValue('httpsOnly'),
                secure_headers: this.getFormValue('secureHeaders')
            },
            advanced: {
                debug_mode: this.getFormValue('debugMode') === 'true',
                performance_monitoring: this.getFormValue('performanceMonitoring') === 'true',
                websocket_enabled: this.getFormValue('websocketEnabled') === 'true',
                auto_backup: this.getFormValue('autoBackup') === 'true',
                custom_config: this.parseJsonValue(this.getFormValue('customConfig'))
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
                notifications: true
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
                file: "logs/mcp_training.log",
                max_size_mb: 100,
                console: true,
                file_enabled: true
            },
            security: {
                auth_enabled: false,
                api_key: "",
                cors_origins: "http://localhost:3000,https://example.com",
                rate_limit: 100,
                https_only: false,
                secure_headers: true
            },
            advanced: {
                debug_mode: false,
                performance_monitoring: true,
                websocket_enabled: true,
                auto_backup: false,
                custom_config: {"custom_setting": "value"}
            }
        };
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