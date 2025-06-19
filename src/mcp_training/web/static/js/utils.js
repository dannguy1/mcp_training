/**
 * Utility functions for MCP Training Service UI
 */

// API Base URL
const API_BASE = '/api';

/**
 * Make API calls with error handling
 */
async function apiCall(endpoint, options = {}) {
    const url = `${API_BASE}${endpoint}`;
    const config = {
        headers: {
            'Content-Type': 'application/json',
            ...options.headers
        },
        ...options
    };

    try {
        const response = await fetch(url, config);
        
        if (!response.ok) {
            throw new Error(`API call failed: ${response.status} ${response.statusText}`);
        }
        
        return await response.json();
    } catch (error) {
        console.error('API call error:', error);
        throw error;
    }
}

/**
 * Show notification toast
 */
function showNotification(message, type = 'info', duration = 5000) {
    // Create toast container if it doesn't exist
    let toastContainer = document.getElementById('toastContainer');
    if (!toastContainer) {
        toastContainer = document.createElement('div');
        toastContainer.id = 'toastContainer';
        toastContainer.className = 'toast-container position-fixed top-0 end-0 p-3';
        toastContainer.style.zIndex = '9999';
        document.body.appendChild(toastContainer);
    }

    // Create toast element
    const toastId = 'toast-' + Date.now();
    const toast = document.createElement('div');
    toast.className = `toast align-items-center text-white bg-${type} border-0`;
    toast.id = toastId;
    toast.setAttribute('role', 'alert');
    toast.setAttribute('aria-live', 'assertive');
    toast.setAttribute('aria-atomic', 'true');

    toast.innerHTML = `
        <div class="d-flex">
            <div class="toast-body">
                ${message}
            </div>
            <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast" aria-label="Close"></button>
        </div>
    `;

    toastContainer.appendChild(toast);

    // Show toast
    const bsToast = new bootstrap.Toast(toast, { delay: duration });
    bsToast.show();

    // Remove toast element after it's hidden
    toast.addEventListener('hidden.bs.toast', () => {
        toast.remove();
    });
}

/**
 * Show error notification
 */
function showError(message, error = null) {
    const fullMessage = error ? `${message}: ${error.message}` : message;
    showNotification(fullMessage, 'danger');
}

/**
 * Show success notification
 */
function showSuccess(message) {
    showNotification(message, 'success');
}

/**
 * Show warning notification
 */
function showWarning(message) {
    showNotification(message, 'warning');
}

/**
 * Show info notification
 */
function showInfo(message) {
    showNotification(message, 'info');
}

/**
 * Show/hide loading overlay
 */
function showLoading() {
    const overlay = document.getElementById('loadingOverlay');
    if (overlay) {
        overlay.style.display = 'flex';
    }
}

function hideLoading() {
    const overlay = document.getElementById('loadingOverlay');
    if (overlay) {
        overlay.style.display = 'none';
    }
}

/**
 * Format date/time
 */
function formatDateTime(dateString) {
    const date = new Date(dateString);
    return date.toLocaleString();
}

function formatDate(dateString) {
    const date = new Date(dateString);
    return date.toLocaleDateString();
}

function formatTime(dateString) {
    const date = new Date(dateString);
    return date.toLocaleTimeString();
}

/**
 * Format file size
 */
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

/**
 * Format duration
 */
function formatDuration(seconds) {
    if (seconds < 60) {
        return `${seconds}s`;
    } else if (seconds < 3600) {
        const minutes = Math.floor(seconds / 60);
        const remainingSeconds = seconds % 60;
        return `${minutes}m ${remainingSeconds}s`;
    } else {
        const hours = Math.floor(seconds / 3600);
        const minutes = Math.floor((seconds % 3600) / 60);
        return `${hours}h ${minutes}m`;
    }
}

/**
 * Get status color class
 */
function getStatusColor(status) {
    const statusMap = {
        'running': 'info',
        'completed': 'success',
        'failed': 'danger',
        'pending': 'warning',
        'cancelled': 'secondary',
        'healthy': 'success',
        'unhealthy': 'danger',
        'warning': 'warning'
    };
    return statusMap[status.toLowerCase()] || 'secondary';
}

/**
 * Get status icon class
 */
function getStatusIcon(status) {
    const iconMap = {
        'running': 'play-circle',
        'completed': 'check-circle',
        'failed': 'x-circle',
        'pending': 'clock',
        'cancelled': 'stop-circle',
        'healthy': 'check-circle',
        'unhealthy': 'x-circle',
        'warning': 'exclamation-triangle'
    };
    return iconMap[status.toLowerCase()] || 'question-circle';
}

/**
 * Debounce function
 */
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

/**
 * Throttle function
 */
function throttle(func, limit) {
    let inThrottle;
    return function() {
        const args = arguments;
        const context = this;
        if (!inThrottle) {
            func.apply(context, args);
            inThrottle = true;
            setTimeout(() => inThrottle = false, limit);
        }
    };
}

/**
 * Copy text to clipboard
 */
async function copyToClipboard(text) {
    try {
        await navigator.clipboard.writeText(text);
        showSuccess('Copied to clipboard');
    } catch (err) {
        console.error('Failed to copy text: ', err);
        showError('Failed to copy to clipboard');
    }
}

/**
 * Download file
 */
function downloadFile(url, filename) {
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}

/**
 * Validate email format
 */
function isValidEmail(email) {
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return emailRegex.test(email);
}

/**
 * Validate file type
 */
function isValidFileType(file, allowedTypes) {
    return allowedTypes.includes(file.type);
}

/**
 * Validate file size
 */
function isValidFileSize(file, maxSizeMB) {
    const maxSizeBytes = maxSizeMB * 1024 * 1024;
    return file.size <= maxSizeBytes;
}

/**
 * Get current page name
 */
function getCurrentPage() {
    const path = window.location.pathname;
    if (path === '/') return 'dashboard';
    return path.substring(1);
}

/**
 * Set active navigation item
 */
function setActiveNavigation() {
    const currentPage = getCurrentPage();
    document.querySelectorAll('[data-page]').forEach(link => {
        if (link.getAttribute('data-page') === currentPage) {
            link.classList.add('active');
        } else {
            link.classList.remove('active');
        }
    });
}

/**
 * Initialize tooltips
 */
function initTooltips() {
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
}

/**
 * Initialize popovers
 */
function initPopovers() {
    const popoverTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="popover"]'));
    popoverTriggerList.map(function (popoverTriggerEl) {
        return new bootstrap.Popover(popoverTriggerEl);
    });
}

/**
 * Logout function
 */
function logout() {
    // Clear any stored data
    localStorage.clear();
    sessionStorage.clear();
    
    // Redirect to login or show logout message
    showInfo('Logged out successfully');
    
    // For now, just reload the page
    setTimeout(() => {
        window.location.reload();
    }, 1000);
}

// Export functions for use in other modules
window.utils = {
    apiCall,
    showNotification,
    showError,
    showSuccess,
    showWarning,
    showInfo,
    showLoading,
    hideLoading,
    formatDateTime,
    formatDate,
    formatTime,
    formatFileSize,
    formatDuration,
    getStatusColor,
    getStatusIcon,
    debounce,
    throttle,
    copyToClipboard,
    downloadFile,
    isValidEmail,
    isValidFileType,
    isValidFileSize,
    getCurrentPage,
    setActiveNavigation,
    initTooltips,
    initPopovers,
    logout
}; 