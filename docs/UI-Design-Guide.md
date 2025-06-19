# UI Design Guide

## Table of Contents
1. [Technology Stack](#technology-stack)
2. [Project Structure](#project-structure)
3. [Layout Structure](#layout-structure)
4. [Navigation](#navigation)
5. [Components](#components)
6. [Styling](#styling)
7. [Responsive Design](#responsive-design)
8. [Implementation Patterns](#implementation-patterns)

## Technology Stack

### Core Dependencies
- **FastAPI** for backend API and server-side rendering
- **Jinja2 Templates** for server-side rendered UI
- **Bootstrap 5.3.6** for styling framework and components
- **Bootstrap Icons** for icon library
- **Chart.js** for data visualization
- **Vanilla JavaScript** for client-side interactions
- **WebSocket** for real-time updates

### Development Tools
- **Python 3.10+** for backend development
- **Pydantic** for data validation and serialization
- **Uvicorn** for ASGI server
- **ESLint** for JavaScript linting (if using)

## Project Structure

```
src/mcp_training/web/
├── templates/           # Jinja2 HTML templates
│   ├── base.html       # Main layout template
│   ├── components/     # Reusable UI components
│   │   ├── navbar.html # Top navigation bar
│   │   ├── sidebar.html # Off-canvas sidebar
│   │   └── loading.html # Loading overlay
│   ├── pages/          # Page-specific templates
│   │   ├── dashboard.html
│   │   ├── training.html
│   │   ├── models.html
│   │   ├── logs.html
│   │   └── settings.html
│   └── partials/       # Template partials
│       ├── head.html   # HTML head section
│       └── scripts.html # JavaScript includes
├── static/             # Static assets
│   ├── css/           # Stylesheets
│   │   ├── main.css   # Global styles
│   │   ├── components.css # Component styles
│   │   └── variables.css # CSS variables
│   └── js/            # JavaScript files
│       ├── app.js     # Main application logic
│       ├── utils.js   # Utility functions
│       ├── dashboard.js # Dashboard functionality
│       ├── training.js # Training page logic
│       ├── models.js  # Models page logic
│       ├── logs.js    # Logs viewer logic
│       └── settings.js # Settings management
└── api/               # FastAPI routes
    ├── app.py         # Main FastAPI application
    ├── routes/        # API route handlers
    │   ├── web.py     # Web page routes
    │   ├── training.py # Training API
    │   ├── models.py  # Models API
    │   ├── logs.py    # Logs API
    │   └── settings.py # Settings API
    └── middleware/    # Middleware components
```

## Layout Structure

### Main Layout Pattern
```html
<!-- base.html - Main wrapper -->
<!DOCTYPE html>
<html lang="en">
<head>
    {% include 'partials/head.html' %}
</head>
<body>
    <div class="d-flex flex-column min-vh-100">
        {% include 'components/navbar.html' %}
        
        <main class="flex-grow-1">
            <div class="container-fluid py-4">
                {% block content %}{% endblock %}
            </div>
        </main>
    </div>
    
    {% include 'components/sidebar.html' %}
    {% include 'components/loading.html' %}
    {% include 'partials/scripts.html' %}
    {% block extra_scripts %}{% endblock %}
</body>
</html>
```

### Page Structure
- Page title as `h2` element with `mb-4` class
- Content organized in Bootstrap Cards
- Consistent spacing using Bootstrap utility classes
- Loading states with overlay component
- Error states with Bootstrap Alert components

## Navigation

### Fixed Top Bar Implementation
```html
<!-- navbar.html -->
<nav class="navbar navbar-expand-lg navbar-dark bg-dark px-3">
    <div class="container-fluid">
        <button class="btn btn-dark me-3" id="sidebarToggle" type="button">
            <i class="bi bi-list"></i>
        </button>
        
        <a class="navbar-brand" href="/">
            <i class="bi bi-cpu me-2"></i>
            MCP Training Service
        </a>
    </div>
</nav>
```

**Key Classes:**
- `navbar-dark bg-dark` - Dark theme navigation
- `px-3` - Horizontal padding
- `sticky-top` - Fixed positioning (handled by CSS)
- `z-index: 1030` - High z-index for overlay

### Off-Canvas Sidebar Implementation
```html
<!-- sidebar.html -->
<div class="offcanvas offcanvas-start bg-dark text-light" id="sidebar" tabindex="-1" aria-labelledby="sidebarLabel" data-bs-backdrop="true" data-bs-scroll="true">
    <div class="offcanvas-header border-bottom border-secondary">
        <h5 class="offcanvas-title" id="sidebarLabel">Navigation</h5>
        <button type="button" class="btn-close btn-close-white" data-bs-dismiss="offcanvas" aria-label="Close"></button>
    </div>
    
    <div class="offcanvas-body">
        <nav class="sidebar-nav">
            <ul class="nav flex-column">
                <li class="nav-item">
                    <a class="nav-link text-light mb-2" href="/" data-page="dashboard">
                        <i class="bi bi-speedometer2 me-2"></i>
                        Dashboard
                    </a>
                </li>
                <!-- Additional navigation items -->
            </ul>
        </nav>
    </div>
</div>
```

**Key Classes:**
- `offcanvas offcanvas-start` - Left-side off-canvas panel
- `bg-dark text-light` - Dark background with light text
- `flex-column` - Vertical navigation layout
- `text-light mb-2` - Light text with bottom margin

### Sidebar Toggle JavaScript
```javascript
// app.js
document.addEventListener('DOMContentLoaded', function() {
    const sidebarToggle = document.getElementById('sidebarToggle');
    const sidebar = new bootstrap.Offcanvas(document.getElementById('sidebar'));
    
    sidebarToggle.addEventListener('click', function() {
        sidebar.show();
    });
});
```

## Components

### Card Implementation
```html
<div class="card mb-4">
    <div class="card-header">
        <h6 class="mb-0">Card Title</h6>
    </div>
    <div class="card-body">
        <!-- Card content -->
    </div>
</div>
```

**Key Classes:**
- `mb-4` - Bottom margin for spacing
- `card-header` - White background with bottom border
- `card-body` - Content padding

### Table Implementation
```html
<div class="card">
    <div class="card-body">
        <div class="table-responsive">
            <table class="table table-hover">
                <thead>
                    <tr>
                        <th>Column Header</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>Data</td>
                    </tr>
                </tbody>
            </table>
        </div>
    </div>
</div>
```

**Key Classes:**
- `table-responsive` - Responsive table behavior
- `table-hover` - Row hover effects
- Wrapped in Card for consistent styling

### Status Badge Implementation
```html
<span class="badge bg-success">Healthy</span>
<span class="badge bg-danger">Error</span>
<span class="badge bg-warning">Warning</span>
<span class="badge bg-info">Info</span>
```

**Status Color Mapping:**
```javascript
// utils.js
function getStatusColor(status) {
    switch (status.toLowerCase()) {
        case 'connected':
        case 'healthy':
        case 'completed':
            return 'success';
        case 'disconnected':
        case 'failed':
            return 'danger';
        case 'warning':
        case 'pending':
            return 'warning';
        case 'running':
            return 'info';
        default:
            return 'secondary';
    }
}
```

### Form Implementation
```html
<form id="settingsForm">
    <div class="row">
        <div class="col-md-6">
            <div class="mb-3">
                <label for="settingName" class="form-label">Setting Name</label>
                <input type="text" class="form-control" id="settingName" placeholder="Enter setting">
                <div class="form-text">Help text for the setting.</div>
            </div>
        </div>
    </div>
    <button type="submit" class="btn btn-primary">
        <i class="bi bi-check me-2"></i>
        Save Settings
    </button>
</form>
```

**Key Classes:**
- `mb-3` - Bottom margin for form groups
- `form-label` - Consistent label styling
- `form-control` - Input styling with focus states
- `form-text` - Help text styling

### Horizontal Tabs Implementation
```html
<!-- For internal page navigation (e.g., Settings page) -->
<ul class="nav nav-tabs" id="pageTabs" role="tablist">
    <li class="nav-item" role="presentation">
        <button class="nav-link active" id="tab1-tab" data-bs-toggle="tab" data-bs-target="#tab1" type="button" role="tab">
            <i class="bi bi-gear me-2"></i>
            Tab 1
        </button>
    </li>
    <li class="nav-item" role="presentation">
        <button class="nav-link" id="tab2-tab" data-bs-toggle="tab" data-bs-target="#tab2" type="button" role="tab">
            <i class="bi bi-cpu me-2"></i>
            Tab 2
        </button>
    </li>
</ul>

<div class="tab-content" id="pageTabsContent">
    <div class="tab-pane fade show active" id="tab1" role="tabpanel">
        <!-- Tab 1 content -->
    </div>
    <div class="tab-pane fade" id="tab2" role="tabpanel">
        <!-- Tab 2 content -->
    </div>
</div>
```

## Styling

### CSS File Organization
1. **variables.css** - CSS custom properties and design tokens
2. **main.css** - Global styles and layout
3. **components.css** - Component-specific styles

### Design Tokens (CSS Variables)
```css
:root {
  /* Color Palette */
  --bs-primary: #0d6efd;
  --bs-success: #198754;
  --bs-warning: #ffc107;
  --bs-danger: #dc3545;
  --bs-info: #0dcaf0;
  --bs-secondary: #6c757d;
  
  /* Typography */
  --font-family: Inter, system-ui, Avenir, Helvetica, Arial, sans-serif;
  --font-weight-normal: 400;
  --font-weight-medium: 500;
  --font-weight-semibold: 600;
  
  /* Spacing */
  --spacing-xs: 0.25rem;
  --spacing-sm: 0.5rem;
  --spacing-md: 1rem;
  --spacing-lg: 1.5rem;
  --spacing-xl: 3rem;
  
  /* Shadows */
  --shadow-sm: 0 2px 4px rgba(0, 0, 0, 0.1);
  --shadow-md: 0 4px 8px rgba(0, 0, 0, 0.15);
  --shadow-lg: 0 8px 16px rgba(0, 0, 0, 0.2);
  
  /* Border Radius */
  --border-radius-sm: 0.25rem;
  --border-radius: 0.375rem;
  --border-radius-lg: 0.5rem;
  
  /* Transitions */
  --transition-fast: 0.15s ease-in-out;
  --transition-normal: 0.2s ease-in-out;
  --transition-slow: 0.3s ease-in-out;
}
```

### Layout Styling
```css
/* Fixed navbar */
.navbar {
    position: sticky;
    top: 0;
    z-index: 1030;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}

/* Main content area */
main {
    flex-grow: 1;
    width: 100%;
    min-height: calc(100vh - 56px); /* Account for navbar height */
}

/* Offcanvas sidebar */
.offcanvas {
    width: 250px;
}

.offcanvas-header {
    padding: 1rem;
}

.offcanvas-body {
    padding: 1rem;
}
```

### Card Styling
```css
.card {
    border: none;
    box-shadow: var(--shadow-sm);
    transition: transform var(--transition-normal), box-shadow var(--transition-normal);
    border-radius: var(--border-radius);
}

.card:hover {
    transform: translateY(-2px);
    box-shadow: var(--shadow-md);
}

.card-header {
    background-color: #fff;
    border-bottom: 1px solid rgba(0, 0, 0, 0.1);
    font-weight: var(--font-weight-semibold);
    border-radius: var(--border-radius) var(--border-radius) 0 0 !important;
}
```

### Button Styling
```css
.btn {
    font-weight: var(--font-weight-medium);
    padding: 0.5rem 1rem;
    border-radius: var(--border-radius);
    transition: all var(--transition-normal);
}

.btn:hover {
    transform: translateY(-1px);
}
```

### Form Control Styling
```css
.form-control:focus,
.form-select:focus {
    border-color: #86b7fe;
    box-shadow: 0 0 0 0.25rem rgba(13, 110, 253, 0.25);
}
```

### Horizontal Tabs Styling
```css
/* Horizontal Tabs Styles */
.nav-tabs .nav-link {
    color: #6c757d !important;
    background-color: transparent !important;
    border: 1px solid transparent !important;
    border-radius: var(--border-radius) var(--border-radius) 0 0 !important;
    padding: 0.75rem 1rem !important;
    font-weight: var(--font-weight-medium) !important;
    transition: all var(--transition-normal) !important;
}

.nav-tabs .nav-link:hover {
    color: var(--bs-primary) !important;
    background-color: rgba(13, 110, 253, 0.05) !important;
    border-color: #e9ecef #e9ecef #dee2e6 !important;
}

.nav-tabs .nav-link.active {
    color: #212529 !important;
    background-color: #fff !important;
    border-color: #dee2e6 #dee2e6 #fff !important;
    border-bottom-color: #fff !important;
    font-weight: var(--font-weight-semibold) !important;
}
```

## Responsive Design

### Breakpoints
```css
/* Mobile: < 768px */
@media (max-width: 768px) {
    .offcanvas {
        width: 100%;
    }
    
    .card-deck {
        flex-direction: column;
    }
    
    .card-deck .card {
        margin-bottom: 1rem;
    }
    
    .table-responsive {
        font-size: 0.875rem;
    }
}

/* Tablet: 768px - 992px */
@media (min-width: 768px) and (max-width: 992px) {
    .offcanvas {
        width: 100%;
        max-width: 300px;
    }
}

/* Desktop: > 992px */
@media (min-width: 992px) {
    .offcanvas {
        width: 250px;
    }
}
```

### Mobile Adaptations
- Off-canvas sidebar with hamburger menu
- Stacked card layouts
- Adjusted button sizes for touch
- Responsive tables with horizontal scroll

### Tablet Adaptations
- Sidebar becomes off-canvas
- Grid adjustments for medium screens
- Maintained readability with proper spacing

### Desktop Optimizations
- Off-canvas sidebar (can be expanded to fixed sidebar)
- Multi-column layouts
- Hover effects
- Optimal spacing and typography

## Implementation Patterns

### Page Template Pattern
```html
{% extends "base.html" %}

{% block title %}Page Title - MCP Training Service{% endblock %}

{% block content %}
<div class="row">
    <div class="col-12">
        <h2 class="mb-4">Page Title</h2>
    </div>
</div>

<div class="row">
    <div class="col-12">
        <div class="card">
            <div class="card-header">
                <h6 class="mb-0">Card Title</h6>
            </div>
            <div class="card-body">
                <!-- Page content -->
            </div>
        </div>
    </div>
</div>
{% endblock %}

{% block extra_scripts %}
<script src="{{ url_for('static', path='/js/page-specific.js') }}"></script>
{% endblock %}
```

### JavaScript Module Pattern
```javascript
// page-specific.js
class PageManager {
    constructor() {
        this.init();
    }
    
    init() {
        this.setupEventListeners();
        this.loadData();
    }
    
    setupEventListeners() {
        // Event listener setup
    }
    
    async loadData() {
        try {
            utils.showLoading();
            const data = await utils.apiCall('/api/endpoint');
            this.updateUI(data);
        } catch (error) {
            utils.showError('Failed to load data', error);
        } finally {
            utils.hideLoading();
        }
    }
    
    updateUI(data) {
        // Update DOM with data
    }
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.pageManager = new PageManager();
});
```

### API Integration Pattern
```javascript
// utils.js
const utils = {
    async apiCall(endpoint, options = {}) {
        const defaultOptions = {
            method: 'GET',
            headers: {
                'Content-Type': 'application/json',
            },
        };
        
        const config = { ...defaultOptions, ...options };
        
        try {
            const response = await fetch(endpoint, config);
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            return await response.json();
        } catch (error) {
            console.error('API call failed:', error);
            throw error;
        }
    },
    
    showLoading() {
        const overlay = document.getElementById('loadingOverlay');
        if (overlay) overlay.style.display = 'flex';
    },
    
    hideLoading() {
        const overlay = document.getElementById('loadingOverlay');
        if (overlay) overlay.style.display = 'none';
    },
    
    showError(message, error) {
        // Show error notification
        console.error(message, error);
    },
    
    showSuccess(message) {
        // Show success notification
        console.log(message);
    }
};
```

### WebSocket Integration Pattern
```javascript
// app.js
class WebSocketManager {
    constructor() {
        this.ws = null;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        this.reconnectDelay = 1000;
    }
    
    connect() {
        try {
            this.ws = new WebSocket('ws://localhost:8000/ws');
            
            this.ws.onopen = () => {
                console.log('WebSocket connected');
                this.reconnectAttempts = 0;
            };
            
            this.ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                this.handleMessage(data);
            };
            
            this.ws.onclose = () => {
                console.log('WebSocket disconnected');
                this.reconnect();
            };
            
            this.ws.onerror = (error) => {
                console.error('WebSocket error:', error);
            };
        } catch (error) {
            console.error('Failed to connect WebSocket:', error);
        }
    }
    
    reconnect() {
        if (this.reconnectAttempts < this.maxReconnectAttempts) {
            this.reconnectAttempts++;
            setTimeout(() => {
                console.log(`Reconnecting... Attempt ${this.reconnectAttempts}`);
                this.connect();
            }, this.reconnectDelay * this.reconnectAttempts);
        }
    }
    
    handleMessage(data) {
        // Handle different message types
        switch (data.type) {
            case 'status_update':
                this.updateStatus(data.payload);
                break;
            case 'training_update':
                this.updateTraining(data.payload);
                break;
            default:
                console.log('Unknown message type:', data.type);
        }
    }
}
```

### Form Handling Pattern
```javascript
// settings.js
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
        // Form submission
        const form = document.getElementById('settingsForm');
        if (form) {
            form.addEventListener('submit', (e) => {
                e.preventDefault();
                this.saveSettings();
            });
        }
        
        // Real-time validation
        this.setupRealTimeValidation();
    }
    
    async loadSettings() {
        try {
            utils.showLoading();
            const settings = await utils.apiCall('/api/settings');
            this.settings = settings;
            this.originalSettings = JSON.parse(JSON.stringify(settings));
            this.populateSettingsForms();
        } catch (error) {
            utils.showError('Failed to load settings', error);
        } finally {
            utils.hideLoading();
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
}
```

### Loading and Error States
```html
<!-- Loading State -->
<div class="loading-overlay" id="loadingOverlay">
    <div class="spinner-border text-primary" role="status">
        <span class="visually-hidden">Loading...</span>
    </div>
</div>

<!-- Error State -->
<div class="alert alert-danger" role="alert">
    <i class="bi bi-exclamation-triangle me-2"></i>
    Error message here
</div>

<!-- Empty State -->
<div class="text-center text-muted py-4">
    <i class="bi bi-file-text fs-1 mb-3"></i>
    <p>No data available</p>
</div>
```

## Best Practices

### Template Organization
1. **Template Inheritance** - Use `{% extends %}` for consistent layouts
2. **Component Reusability** - Create reusable components with `{% include %}`
3. **Block Structure** - Use `{% block %}` for flexible content areas
4. **Conditional Rendering** - Use `{% if %}` for dynamic content

### JavaScript Organization
1. **Module Pattern** - Use classes for page-specific functionality
2. **Event Delegation** - Use event listeners efficiently
3. **Error Handling** - Always handle API errors gracefully
4. **Loading States** - Show loading indicators for async operations

### Styling Guidelines
1. **Bootstrap First** - Use Bootstrap classes before custom CSS
2. **CSS Variables** - Use design tokens for consistency
3. **Responsive Design** - Always consider mobile-first approach
4. **Performance** - Minimize custom CSS and leverage Bootstrap utilities

### State Management
1. **Server State** - Use FastAPI for data management
2. **Client State** - Use JavaScript classes for UI state
3. **Form State** - Use controlled components with proper validation
4. **Real-time Updates** - Use WebSocket for live data updates 