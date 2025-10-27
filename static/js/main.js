// Global state
let selectedFile = null;
let currentLanguage = 'english';

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
    initializeLanguageSelector();
    initializeFileUpload();
    initializeAnimations();
});

// Language Selector
function initializeLanguageSelector() {
    const langButtons = document.querySelectorAll('.lang-btn');
    
    langButtons.forEach(btn => {
        btn.addEventListener('click', function() {
            const lang = this.dataset.lang;
            setLanguage(lang);
        });
    });
    
    // Set active language on page load
    const savedLang = getCurrentLanguage();
    if (savedLang) {
        const activeBtn = document.querySelector(`.lang-btn[data-lang="${savedLang}"]`);
        if (activeBtn) {
            activeBtn.classList.add('active');
        }
    }
}

function setLanguage(lang) {
    currentLanguage = lang;
    
    // Update active state
    document.querySelectorAll('.lang-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    document.querySelector(`.lang-btn[data-lang="${lang}"]`).classList.add('active');
    
    // Send to backend
    fetch(`/set-language/${lang}`)
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                // Reload page to apply translations
                location.reload();
            }
        })
        .catch(error => {
            console.error('Error setting language:', error);
        });
}

function getCurrentLanguage() {
    const activeBtn = document.querySelector('.lang-btn.active');
    return activeBtn ? activeBtn.dataset.lang : 'english';
}

// File Upload Handler
function initializeFileUpload() {
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('fileInput');
    const uploadForm = document.getElementById('uploadForm');
    
    if (!uploadArea || !fileInput) return;
    
    // Click to upload
    uploadArea.addEventListener('click', () => {
        fileInput.click();
    });
    
    // File input change
    fileInput.addEventListener('change', function(e) {
        handleFileSelect(e.target.files[0]);
    });
    
    // Drag and drop
    uploadArea.addEventListener('dragover', function(e) {
        e.preventDefault();
        e.stopPropagation();
        this.classList.add('dragover');
    });
    
    uploadArea.addEventListener('dragleave', function(e) {
        e.preventDefault();
        e.stopPropagation();
        this.classList.remove('dragover');
    });
    
    uploadArea.addEventListener('drop', function(e) {
        e.preventDefault();
        e.stopPropagation();
        this.classList.remove('dragover');
        
        const file = e.dataTransfer.files[0];
        handleFileSelect(file);
    });
    
    // Form submission
    if (uploadForm) {
        uploadForm.addEventListener('submit', handleFormSubmit);
    }
}

function handleFileSelect(file) {
    if (!file) return;
    
    // Validate file type
    const validTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/webp'];
    if (!validTypes.includes(file.type)) {
        showAlert('Please upload a valid image file (JPG, PNG, or WEBP)', 'error');
        return;
    }
    
    // Validate file size (16MB max)
    if (file.size > 16 * 1024 * 1024) {
        showAlert('File size must be less than 16MB', 'error');
        return;
    }
    
    selectedFile = file;
    
    // Show preview
    const reader = new FileReader();
    reader.onload = function(e) {
        displayImagePreview(e.target.result);
    };
    reader.readAsDataURL(file);
}

function displayImagePreview(imageUrl) {
    const previewSection = document.getElementById('imagePreview');
    const previewImg = document.getElementById('previewImg');
    const uploadBtn = document.getElementById('uploadBtn');
    
    if (previewSection && previewImg) {
        previewImg.src = imageUrl;
        previewSection.classList.add('active');
    }
    
    if (uploadBtn) {
        uploadBtn.disabled = false;
    }
}

function removeImage() {
    selectedFile = null;
    const previewSection = document.getElementById('imagePreview');
    const fileInput = document.getElementById('fileInput');
    const uploadBtn = document.getElementById('uploadBtn');
    
    if (previewSection) {
        previewSection.classList.remove('active');
    }
    
    if (fileInput) {
        fileInput.value = '';
    }
    
    if (uploadBtn) {
        uploadBtn.disabled = true;
    }
}

function handleFormSubmit(e) {
    e.preventDefault();
    
    if (!selectedFile) {
        showAlert('Please select an image first', 'error');
        return;
    }
    
    const formData = new FormData();
    formData.append('leaf_image', selectedFile);
    
    // Show loading state
    showLoading(true);
    hideAlert();
    
    // Disable upload button
    const uploadBtn = document.getElementById('uploadBtn');
    if (uploadBtn) {
        uploadBtn.disabled = true;
        uploadBtn.textContent = 'Processing...';
    }
    
    // Submit to backend
    fetch('/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            // Redirect to results
            window.location.href = data.redirect;
        } else {
            showAlert(data.error || 'Upload failed. Please try again.', 'error');
            showLoading(false);
            if (uploadBtn) {
                uploadBtn.disabled = false;
                uploadBtn.textContent = 'Detect Disease';
            }
        }
    })
    .catch(error => {
        console.error('Upload error:', error);
        showAlert('Network error. Please check your connection and try again.', 'error');
        showLoading(false);
        if (uploadBtn) {
            uploadBtn.disabled = false;
            uploadBtn.textContent = 'Detect Disease';
        }
    });
}

// Alert System
function showAlert(message, type = 'info') {
    const alertDiv = document.getElementById('alertMessage');
    if (!alertDiv) return;
    
    alertDiv.textContent = message;
    alertDiv.className = `alert alert-${type} active`;
    
    // Auto-hide after 5 seconds
    setTimeout(() => {
        hideAlert();
    }, 5000);
}

function hideAlert() {
    const alertDiv = document.getElementById('alertMessage');
    if (alertDiv) {
        alertDiv.classList.remove('active');
    }
}

// Loading Indicator
function showLoading(show) {
    const loadingDiv = document.getElementById('loadingIndicator');
    if (loadingDiv) {
        if (show) {
            loadingDiv.classList.add('active');
        } else {
            loadingDiv.classList.remove('active');
        }
    }
}

// Animations
function initializeAnimations() {
    // Fade in elements on scroll
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };
    
    const observer = new IntersectionObserver(function(entries) {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.opacity = '1';
                entry.target.style.transform = 'translateY(0)';
            }
        });
    }, observerOptions);
    
    // Observe feature cards
    const featureCards = document.querySelectorAll('.feature-card');
    featureCards.forEach(card => {
        card.style.opacity = '0';
        card.style.transform = 'translateY(20px)';
        card.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
        observer.observe(card);
    });
    
    // Animate confidence meter on result page
    const meterFill = document.querySelector('.meter-fill');
    if (meterFill) {
        const targetWidth = meterFill.dataset.confidence;
        setTimeout(() => {
            meterFill.style.width = targetWidth + '%';
        }, 500);
    }
}

// Result Page Functions
function scanAnotherLeaf() {
    window.location.href = '/upload';
}

function downloadReport() {
    window.print();
}

function shareResult() {
    if (navigator.share) {
        navigator.share({
            title: 'Crop Disease Detection Result',
            text: 'Check out my crop disease detection result',
            url: window.location.href
        }).catch(error => console.log('Error sharing:', error));
    } else {
        // Fallback: copy link to clipboard
        const url = window.location.href;
        navigator.clipboard.writeText(url).then(() => {
            showAlert('Link copied to clipboard!', 'success');
        });
    }
}

// Utility Functions
function formatConfidence(confidence) {
    return Math.round(confidence) + '%';
}

function getSeverityClass(severity) {
    const severityMap = {
        'healthy': 'severity-healthy',
        'mild': 'severity-mild',
        'moderate': 'severity-moderate',
        'severe': 'severity-severe'
    };
    return severityMap[severity] || 'severity-moderate';
}

// Camera Capture (for mobile devices)
function captureFromCamera() {
    const fileInput = document.getElementById('fileInput');
    if (fileInput) {
        fileInput.setAttribute('capture', 'environment');
        fileInput.click();
    }
}

// Print-specific styles
window.addEventListener('beforeprint', function() {
    document.body.classList.add('printing');
});

window.addEventListener('afterprint', function() {
    document.body.classList.remove('printing');
});

// Service Worker Registration (for future PWA support)
if ('serviceWorker' in navigator) {
    window.addEventListener('load', function() {
        // Uncomment when service worker is ready
        // navigator.serviceWorker.register('/sw.js')
        //     .then(reg => console.log('Service Worker registered'))
        //     .catch(err => console.log('Service Worker registration failed'));
    });
}
