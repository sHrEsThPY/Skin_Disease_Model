document.addEventListener("DOMContentLoaded", () => {
    // Elements
    const dropZone = document.getElementById("drop-zone");
    const fileInput = document.getElementById("file-input");
    const uploadPrompt = document.getElementById("upload-prompt");
    const imagePreview = document.getElementById("image-preview");
    const fileInfo = document.getElementById("file-info");
    const analyseBtn = document.getElementById("analyse-btn");
    const resetBtn = document.getElementById("reset-btn");
    const loadingOverlay = document.getElementById("loading-overlay");
    const resultsPanel = document.getElementById("results-panel");
    const errorToast = document.getElementById("error-toast");
    const errorText = document.getElementById("error-text");
    const qualityWarning = document.getElementById("quality-warning");
    const qualityWarningText = document.getElementById("quality-warning-text");

    // Camera elements
    const openCameraBtn = document.getElementById("open-camera-btn");
    const cameraVideo = document.getElementById("camera-video");
    const cameraControls = document.getElementById("camera-controls");
    const switchCameraBtn = document.getElementById("switch-camera-btn");
    const captureBtn = document.getElementById("capture-btn");
    const capturePreviewContainer = document.getElementById("capture-preview-container");
    const capturePreview = document.getElementById("capture-preview");
    const retakeBtn = document.getElementById("retake-btn");
    const cameraCanvas = document.getElementById("camera-canvas");
    const sourceBadge = document.getElementById("source-badge");

    // State
    let selectedFile = null;
    let stream = null;
    let facingMode = 'environment';
    let currentSource = 'upload'; // 'upload' or 'camera'
    let imageBlob = null;

    // --- Tab Handling ---
    document.querySelectorAll('button[data-bs-toggle="tab"]').forEach(tab => {
        tab.addEventListener('shown.bs.tab', (e) => {
            if (e.target.id === 'camera-tab') {
                currentSource = 'camera';
                clearUploadState();
            } else {
                currentSource = 'upload';
                stopCamera();
                clearCameraState();
            }
            updateAnalyseBtn();
        });
    });

    // --- Upload Logic ---
    dropZone.addEventListener("click", () => fileInput.click());

    fileInput.addEventListener("change", (e) => {
        if (e.target.files.length > 0) handleFile(e.target.files[0]);
    });

    dropZone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropZone.classList.add("dragover");
    });

    dropZone.addEventListener("dragleave", () => {
        dropZone.classList.remove("dragover");
    });

    dropZone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropZone.classList.remove("dragover");
        if (e.dataTransfer.files.length > 0) handleFile(e.dataTransfer.files[0]);
    });

    function handleFile(file) {
        hideError();
        qualityWarning.classList.add('d-none');
        const validTypes = ['image/jpeg', 'image/png', 'image/webp'];
        if (!validTypes.includes(file.type)) {
            showError("Invalid file type. Please upload a JPEG, PNG, or WEBP image.");
            return;
        }

        if (file.size > 5 * 1024 * 1024) {
            showError("File too large. Maximum size is 5MB.");
            return;
        }

        selectedFile = file;
        imageBlob = file;

        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            imagePreview.classList.remove("d-none");
            uploadPrompt.classList.add("d-none");
            fileInfo.textContent = `${file.name} (${(file.size / 1024 / 1024).toFixed(2)} MB)`;
            fileInfo.classList.remove("d-none");
            currentSource = 'upload';
            updateAnalyseBtn();
        };
        reader.readAsDataURL(file);
    }

    // --- Camera Logic ---
    openCameraBtn.addEventListener('click', async () => {
        openCameraBtn.classList.add('d-none');
        cameraVideo.classList.remove('d-none');
        cameraControls.classList.remove('d-none');
        await openCamera();
    });

    async function openCamera() {
        try {
            stream = await navigator.mediaDevices.getUserMedia({
                video: { facingMode, width: { ideal: 1280 }, height: { ideal: 720 } }
            });
            cameraVideo.srcObject = stream;
            
            if (facingMode === 'user') {
                cameraVideo.classList.remove('env-facing');
            } else {
                cameraVideo.classList.add('env-facing');
            }
        } catch (err) {
            showError("Could not access camera. Please check permissions.");
            openCameraBtn.classList.remove('d-none');
        }
    }

    function stopCamera() {
        if (stream) {
            stream.getTracks().forEach(t => t.stop());
            stream = null;
        }
    }

    switchCameraBtn.addEventListener('click', () => {
        facingMode = (facingMode === 'user') ? 'environment' : 'user';
        stopCamera();
        openCamera();
    });

    captureBtn.addEventListener('click', () => {
        const w = cameraVideo.videoWidth;
        const h = cameraVideo.videoHeight;
        cameraCanvas.width = w;
        cameraCanvas.height = h;
        const ctx = cameraCanvas.getContext('2d');
        
        if (facingMode === 'user') {
            ctx.translate(w, 0);
            ctx.scale(-1, 1);
        }
        
        ctx.drawImage(cameraVideo, 0, 0, w, h);
        
        const dataUrl = cameraCanvas.toDataURL('image/png');
        capturePreview.src = dataUrl;
        
        cameraCanvas.toBlob(blob => {
            imageBlob = blob;
            currentSource = 'camera';
            updateAnalyseBtn();
            
            // Image quality check
            checkImageQuality(ctx, w, h);
            
            cameraVideo.classList.add('d-none');
            cameraControls.classList.add('d-none');
            capturePreviewContainer.classList.remove('d-none');
            stopCamera();
        }, 'image/png');
    });

    retakeBtn.addEventListener('click', () => {
        capturePreviewContainer.classList.add('d-none');
        cameraVideo.classList.remove('d-none');
        cameraControls.classList.remove('d-none');
        imageBlob = null;
        updateAnalyseBtn();
        qualityWarning.classList.add('d-none');
        openCamera();
    });

    function checkImageQuality(ctx, w, h) {
        const imageData = ctx.getImageData(0, 0, w, h);
        const data = imageData.data;
        let sumLuminance = 0;
        
        // Fast approx check
        const step = 4 * 10;
        let count = 0;
        for (let i = 0; i < data.length; i += step) {
            const r = data[i];
            const g = data[i+1];
            const b = data[i+2];
            const lum = 0.299 * r + 0.587 * g + 0.114 * b;
            sumLuminance += lum;
            count++;
        }
        
        const avgLum = sumLuminance / count;
        if (avgLum < 40) {
            showWarning("Image too dark — improve lighting");
        } else if (avgLum > 220) {
            showWarning("Image too bright — reduce glare");
        } else {
            qualityWarning.classList.add('d-none');
        }
    }

    // --- Global Controls ---
    function updateAnalyseBtn() {
        analyseBtn.disabled = (imageBlob === null);
    }

    function clearUploadState() {
        selectedFile = null;
        if (currentSource !== 'camera') imageBlob = null;
        imagePreview.src = "";
        imagePreview.classList.add('d-none');
        uploadPrompt.classList.remove('d-none');
        fileInfo.classList.add('d-none');
        fileInput.value = "";
    }

    function clearCameraState() {
        if (currentSource !== 'upload') imageBlob = null;
        capturePreviewContainer.classList.add('d-none');
        openCameraBtn.classList.remove('d-none');
        cameraVideo.classList.add('d-none');
        cameraControls.classList.add('d-none');
    }

    analyseBtn.addEventListener("click", async () => {
        if (!imageBlob) return;

        analyseBtn.disabled = true;
        loadingOverlay.classList.remove("d-none");
        resultsPanel.classList.add("d-none");
        hideError();

        const formData = new FormData();
        formData.append("image", imageBlob, 'capture.png');

        try {
            const resp = await fetch("/predict", {
                method: "POST",
                body: formData
            });

            const data = await resp.json();

            if (!resp.ok) {
                throw new Error(data.error || "An error occurred during analysis.");
            }

            renderResults(data);
        } catch (err) {
            showError(err.message);
            updateAnalyseBtn();
        } finally {
            loadingOverlay.classList.add("d-none");
        }
    });

    resetBtn.addEventListener("click", () => {
        resultsPanel.classList.add("d-none");
        clearUploadState();
        clearCameraState();
        qualityWarning.classList.add('d-none');
        updateAnalyseBtn();
        // Scroll to top
        window.scrollTo({ top: 0, behavior: 'smooth' });
    });

    function renderResults(data) {
        // Top 1
        document.getElementById("top1-class").textContent = data.top1.class;
        document.getElementById("top1-pct").textContent = data.top1.confidence_pct;
        
        const top1Bar = document.getElementById("top1-bar");
        top1Bar.className = "progress-bar progress-bar-animated rounded-pill";
        top1Bar.style.width = "0%";
        
        const badge = document.getElementById("top1-badge");
        badge.textContent = data.top1.confidence_level === 'high' ? '🟢 High Confidence' : 
                           (data.top1.confidence_level === 'medium' ? '🟡 Moderate Confidence' : '🔴 Low Confidence');
        badge.className = `badge bg-${data.top1.color_code} shadow-sm`;

        if (data.top1.color_code === 'green') {
            top1Bar.classList.add('bg-success');
        } else if (data.top1.color_code === 'yellow') {
            top1Bar.classList.add('bg-warning');
        } else {
            top1Bar.classList.add('bg-danger');
        }

        setTimeout(() => {
            top1Bar.style.width = `${data.top1.confidence * 100}%`;
        }, 100);

        // Top 2
        document.getElementById("top2-class").textContent = data.top2.class;
        document.getElementById("top2-pct").textContent = data.top2.confidence_pct;

        // Probabilities
        const container = document.getElementById("all-probs-container");
        container.innerHTML = '';
        for (const [cls, prob] of Object.entries(data.all_probabilities)) {
            const pct = (prob * 100).toFixed(1);
            container.innerHTML += `
                <div class="prob-row">
                    <div class="prob-label">
                        <span>${cls}</span>
                        <span>${pct}%</span>
                    </div>
                    <div class="prob-track">
                        <div class="prob-fill" style="width: 0%"></div>
                    </div>
                </div>
            `;
        }

        setTimeout(() => {
            const fills = container.querySelectorAll('.prob-fill');
            const entries = Object.values(data.all_probabilities);
            fills.forEach((fill, idx) => {
                fill.style.width = `${entries[idx] * 100}%`;
            });
        }, 100);

        const fb = document.getElementById("fallback-warning");
        if (data.fallback_warning) fb.classList.remove("d-none");
        else fb.classList.add("d-none");

        document.getElementById("desc-title").textContent = data.top1.class;
        document.getElementById("condition-desc").textContent = data.condition_description;
        document.getElementById("condition-rec").textContent = data.recommendation;

        if (currentSource === 'camera') {
            sourceBadge.textContent = '📷 Real-time capture';
        } else {
            sourceBadge.textContent = '📁 Uploaded image';
        }
        
        let fileDimText = "";
        if (data.width && data.height) {
            fileDimText = ` • ${data.width}x${data.height}`;
        }
        const timeStr = new Date().toLocaleTimeString();
        sourceBadge.textContent += `${fileDimText} • ${timeStr}`;

        resultsPanel.classList.remove("d-none");
    }

    function showError(msg) {
        errorText.textContent = msg;
        errorToast.classList.remove("d-none");
        setTimeout(() => errorToast.classList.add("d-none"), 5000);
    }
    
    function showWarning(msg) {
        qualityWarningText.textContent = msg;
        qualityWarning.classList.remove('d-none');
    }

    function hideError() {
        errorToast.classList.add("d-none");
    }
});
