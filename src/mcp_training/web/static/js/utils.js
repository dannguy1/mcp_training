/**
 * Utility functions for MCP Training Service UI
 */

// API Base URL - Point to backend server
const API_BASE = 'http://localhost:8000';

/**
 * Make API calls with error handling and timeout - optimized for training system
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

    // Add timeout to prevent hanging requests - shorter for training system
    const timeout = options.timeout || 15000; // 15 second default timeout (reduced from 30)
    
    try {
        // Create abort controller for timeout
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), timeout);
        
        config.signal = controller.signal;
        
        const response = await fetch(url, config);
        
        clearTimeout(timeoutId);
        
        if (!response.ok) {
            // Handle specific HTTP error codes
            if (response.status === 401) {
                throw new Error('Authentication required. Please log in again.');
            } else if (response.status === 403) {
                throw new Error('Access denied. You do not have permission for this action.');
            } else if (response.status === 404) {
                throw new Error('Resource not found. The requested endpoint does not exist.');
            } else if (response.status === 500) {
                throw new Error('Server error. Please try again later.');
            } else if (response.status === 503) {
                throw new Error('Service temporarily unavailable. Please try again later.');
            } else {
                throw new Error(`API call failed: ${response.status} ${response.statusText}`);
            }
        }
        
        // Check if response is JSON
        const contentType = response.headers.get('content-type');
        if (contentType && contentType.includes('application/json')) {
            return await response.json();
        } else {
            // Handle non-JSON responses
            const text = await response.text();
            return { success: true, data: text };
        }
        
    } catch (error) {
        console.error('API call error:', error);
        
        // Handle specific error types
        if (error.name === 'AbortError') {
            throw new Error('Request timed out. Please try again.');
        } else if (error.name === 'TypeError' && error.message.includes('fetch')) {
            throw new Error('Network error. Please check your connection and try again.');
        }
        
        throw error;
    }
}

/**
 * Lightweight performance monitoring
 */
const performanceMonitor = {
    apiCalls: 0,
    lastReset: Date.now(),
    
    trackApiCall() {
        this.apiCalls++;
        const now = Date.now();
        
        // Log performance every 5 minutes
        if (now - this.lastReset > 300000) {
            console.log(`Performance: ${this.apiCalls} API calls in last 5 minutes`);
            this.apiCalls = 0;
            this.lastReset = now;
        }
    },
    
    getApiCallRate() {
        const elapsed = (Date.now() - this.lastReset) / 1000 / 60; // minutes
        return this.apiCalls / elapsed;
    }
};

// Track API calls for performance monitoring
const originalApiCall = apiCall;
apiCall = async function(endpoint, options = {}) {
    performanceMonitor.trackApiCall();
    return originalApiCall(endpoint, options);
};

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
 * Show/hide loading overlay with improved error handling
 */
function showLoading(message = 'Loading...') {
    const overlay = document.getElementById('loadingOverlay');
    if (overlay) {
        // Update loading message if provided
        const messageElement = overlay.querySelector('.loading-message');
        if (messageElement) {
            messageElement.textContent = message;
        }
        
        overlay.style.display = 'flex';
        console.log('Loading overlay shown:', message);
        
        // Auto-hide after 20 seconds to prevent stuck loading state (increased from 15)
        const timeoutId = setTimeout(() => {
            if (overlay.style.display === 'flex') {
                console.warn('Loading overlay auto-hidden after timeout');
                hideLoading();
                showWarning('Operation timed out. Please try again.');
            }
        }, 20000);
        
        // Store timeout ID for potential cancellation
        overlay.dataset.timeoutId = timeoutId;
    } else {
        console.warn('Loading overlay element not found');
    }
}

function hideLoading() {
    const overlay = document.getElementById('loadingOverlay');
    if (overlay) {
        // Clear any pending timeout
        if (overlay.dataset.timeoutId) {
            clearTimeout(parseInt(overlay.dataset.timeoutId));
            delete overlay.dataset.timeoutId;
        }
        
        overlay.style.display = 'none';
        console.log('Loading overlay hidden');
    } else {
        console.warn('Loading overlay element not found');
    }
}

/**
 * Show loading with progress indicator
 */
function showLoadingWithProgress(message = 'Loading...', progress = 0) {
    const overlay = document.getElementById('loadingOverlay');
    if (overlay) {
        // Update loading message
        const messageElement = overlay.querySelector('.loading-message');
        if (messageElement) {
            messageElement.textContent = message;
        }
        
        // Update progress bar if it exists
        const progressBar = overlay.querySelector('.progress-bar');
        if (progressBar) {
            progressBar.style.width = `${progress}%`;
            progressBar.setAttribute('aria-valuenow', progress);
        }
        
        overlay.style.display = 'flex';
    }
}

/**
 * Force hide all loading states (emergency recovery)
 */
function forceHideAllLoading() {
    hideLoading();
    
    // Hide any other loading indicators
    const loadingElements = document.querySelectorAll('.loading, .spinner, .overlay');
    loadingElements.forEach(element => {
        element.style.display = 'none';
    });
    
    // Enable any disabled buttons
    const disabledButtons = document.querySelectorAll('button:disabled, input:disabled');
    disabledButtons.forEach(button => {
        button.disabled = false;
    });
    
    console.log('All loading states force-hidden');
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

// Global function for debugging loading overlay
window.forceHideLoading = function() {
    console.log('Force hiding loading overlay...');
    hideLoading();
};

// Auto-hide loading overlay on page load to prevent stuck state
document.addEventListener('DOMContentLoaded', function() {
    setTimeout(() => {
        hideLoading();
    }, 1000);
}); 