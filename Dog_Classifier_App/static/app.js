// CanineVision AI - Frontend Logic
document.addEventListener('DOMContentLoaded', () => {
  // Elements
  const dropzone = document.getElementById('dropzone');
  const fileInput = document.getElementById('fileInput');
  const btnBrowse = document.getElementById('btnBrowse');
  const dropzonePrompt = document.getElementById('dropzonePrompt');
  const previewContainer = document.getElementById('previewContainer');
  const imagePreview = document.getElementById('imagePreview');
  const previewFilename = document.getElementById('previewFilename');
  const btnChangeImage = document.getElementById('btnChangeImage');

  const btnSampleHusky = document.getElementById('btnSampleHusky');
  const btnSampleChihuahua = document.getElementById('btnSampleChihuahua');

  const btnClassify = document.getElementById('btnClassify');
  const btnSpinner = document.getElementById('btnSpinner');
  const btnText = btnClassify.querySelector('.btn-text');

  const resultsIdle = document.getElementById('resultsIdle');
  const resultsActive = document.getElementById('resultsActive');

  const topBreed = document.getElementById('topBreed');
  const topConfidence = document.getElementById('topConfidence');
  const winnerCaption = document.getElementById('winnerCaption');
  const probBarsList = document.getElementById('probBarsList');

  const telModel = document.getElementById('telModel');
  const telLatency = document.getElementById('telLatency');

  let currentFile = null;

  // Initialize model info
  fetch('/api/info')
    .then(r => r.json())
    .then(data => {
      if (data.model_name) {
        document.getElementById('modelBadge').innerHTML = 
          `<span class="status-dot"></span> Model: ${data.model_name}`;
        telModel.textContent = data.model_name;
      }
    })
    .catch(() => {});

  // File selection triggers
  btnBrowse.addEventListener('click', (e) => {
    e.stopPropagation();
    fileInput.click();
  });

  dropzone.addEventListener('click', () => {
    if (!currentFile) {
      fileInput.click();
    }
  });

  fileInput.addEventListener('change', (e) => {
    if (e.target.files && e.target.files[0]) {
      handleSelectedFile(e.target.files[0]);
    }
  });

  // Drag and Drop
  ['dragenter', 'dragover'].forEach(eventName => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.add('drag-active');
    });
  });

  ['dragleave', 'drop'].forEach(eventName => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.remove('drag-active');
    });
  });

  dropzone.addEventListener('drop', (e) => {
    const dt = e.dataTransfer;
    if (dt.files && dt.files[0]) {
      handleSelectedFile(dt.files[0]);
    }
  });

  function handleSelectedFile(file) {
    if (!file.type.startsWith('image/')) {
      alert('Please upload an image file (JPEG, PNG, WEBP).');
      return;
    }

    currentFile = file;

    // Read and preview
    const reader = new FileReader();
    reader.onload = (e) => {
      imagePreview.src = e.target.result;
      previewFilename.textContent = file.name || 'uploaded_image.jpg';
      dropzonePrompt.classList.add('hidden');
      previewContainer.classList.remove('hidden');
      btnClassify.disabled = false;

      // Clear sample buttons active state
      btnSampleHusky.classList.remove('active');
      btnSampleChihuahua.classList.remove('active');
    };
    reader.readAsDataURL(file);
  }

  btnChangeImage.addEventListener('click', (e) => {
    e.stopPropagation();
    currentFile = null;
    fileInput.value = '';
    previewContainer.classList.add('hidden');
    dropzonePrompt.classList.remove('hidden');
    btnClassify.disabled = true;

    // Reset results
    resultsActive.classList.add('hidden');
    resultsIdle.classList.remove('hidden');
    btnSampleHusky.classList.remove('active');
    btnSampleChihuahua.classList.remove('active');
  });

  // Sample Images Loader
  async function loadSampleImage(url, filename, buttonElement) {
    btnSampleHusky.classList.remove('active');
    btnSampleChihuahua.classList.remove('active');
    buttonElement.classList.add('active');

    try {
      const res = await fetch(url);
      const blob = await res.blob();
      const file = new File([blob], filename, { type: 'image/jpeg' });
      handleSelectedFile(file);

      // Auto classify sample immediately for instant demo
      classifyCurrentFile();
    } catch (err) {
      console.error('Failed to load sample image:', err);
    }
  }

  btnSampleHusky.addEventListener('click', () => {
    loadSampleImage('/static/samples/husky_sample.jpg', 'siberian_husky_sample.jpg', btnSampleHusky);
  });

  btnSampleChihuahua.addEventListener('click', () => {
    loadSampleImage('/static/samples/chihuahua_sample.jpg', 'chihuahua_sample.jpg', btnSampleChihuahua);
  });

  // Classification API Call
  async function classifyCurrentFile() {
    if (!currentFile) return;

    btnClassify.disabled = true;
    btnSpinner.classList.remove('hidden');
    btnText.textContent = 'Classifying...';

    const formData = new FormData();
    formData.append('file', currentFile);

    try {
      const response = await fetch('/api/predict', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const err = await response.json();
        throw new Error(err.detail || 'Prediction failed');
      }

      const result = await response.json();
      renderResults(result);
    } catch (error) {
      alert(`Classification error: ${error.message}`);
    } finally {
      btnClassify.disabled = false;
      btnSpinner.classList.add('hidden');
      btnText.textContent = 'Classify Dog Breed';
    }
  }

  btnClassify.addEventListener('click', classifyCurrentFile);

  function renderResults(data) {
    resultsIdle.classList.add('hidden');
    resultsActive.classList.remove('hidden');

    topBreed.textContent = data.predicted_breed;
    topConfidence.textContent = `${data.confidence.toFixed(1)}%`;

    if (data.confidence >= 90) {
      winnerCaption.textContent = 'Exceptionally high confidence match';
    } else if (data.confidence >= 70) {
      winnerCaption.textContent = 'High confidence match';
    } else {
      winnerCaption.textContent = 'Moderate confidence match';
    }

    telLatency.textContent = `⚡ ${data.latency_ms} ms`;
    telModel.textContent = data.model_name;

    // Render probability comparison bars
    probBarsList.innerHTML = '';
    data.breakdown.forEach((item, index) => {
      const isTop = index === 0;
      const probItem = document.createElement('div');
      probItem.className = `prob-item ${isTop ? 'is-top' : ''}`;

      probItem.innerHTML = `
        <div class="prob-header">
          <span>${item.breed === 'Siberian Husky' ? '🐺' : '🐶'} ${item.breed}</span>
          <span class="prob-pct">${item.probability.toFixed(1)}%</span>
        </div>
        <div class="prob-track">
          <div class="prob-fill" style="width: 0%"></div>
        </div>
      `;

      probBarsList.appendChild(probItem);

      // Trigger animated width after insertion
      setTimeout(() => {
        const fill = probItem.querySelector('.prob-fill');
        fill.style.width = `${Math.max(item.probability, 1)}%`;
      }, 50);
    });
  }
});
