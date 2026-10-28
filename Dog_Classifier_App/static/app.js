// CanineVision AI - Frontend Logic: 120-Breed Classifier & Image Explorer
document.addEventListener('DOMContentLoaded', () => {
  // Navigation Tabs & Views
  const tabClassifier = document.getElementById('tabClassifier');
  const tabGallery = document.getElementById('tabGallery');
  const tabAudit = document.getElementById('tabAudit');
  const viewClassifier = document.getElementById('viewClassifier');
  const viewGallery = document.getElementById('viewGallery');
  const viewAudit = document.getElementById('viewAudit');

  // Classifier Elements
  const dropzone = document.getElementById('dropzone');
  const fileInput = document.getElementById('fileInput');
  const btnBrowse = document.getElementById('btnBrowse');
  const dropzonePrompt = document.getElementById('dropzonePrompt');
  const previewContainer = document.getElementById('previewContainer');
  const imagePreview = document.getElementById('imagePreview');
  const previewFilename = document.getElementById('previewFilename');
  const btnChangeImage = document.getElementById('btnChangeImage');
  const sampleButtons = document.querySelectorAll('.sample-btn');

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
  const telClasses = document.getElementById('telClasses');
  const classesBadge = document.getElementById('classesBadge');

  // Gallery Explorer Elements
  const breedSelect = document.getElementById('breedSelect');
  const breedSearchInput = document.getElementById('breedSearchInput');
  const btnClearSearch = document.getElementById('btnClearSearch');
  const searchSuggestions = document.getElementById('searchSuggestions');
  const galleryGrid = document.getElementById('galleryGrid');
  const breedBanner = document.getElementById('breedBanner');
  const breedBannerTitle = document.getElementById('breedBannerTitle');
  const breedSynset = document.getElementById('breedSynset');
  const breedImageCount = document.getElementById('breedImageCount');
  const galleryBreedCountText = document.getElementById('galleryBreedCountText');

  // Audit Studio Elements
  const auditPendingBadge = document.getElementById('auditPendingBadge');
  const auditPairsDeck = document.getElementById('auditPairsDeck');
  const btnBatchRecommended = document.getElementById('btnBatchRecommended');
  const btnGenerateCleanManifest = document.getElementById('btnGenerateCleanManifest');
  const auditFilterPills = document.getElementById('auditFilterPills');
  const auditSearchInput = document.getElementById('auditSearchInput');
  const manifestAlert = document.getElementById('manifestAlert');
  const manifestAlertText = document.getElementById('manifestAlertText');
  const statTotalPairs = document.getElementById('statTotalPairs');
  const statPendingPairs = document.getElementById('statPendingPairs');
  const statPhotosToRemove = document.getElementById('statPhotosToRemove');
  const statPairsKept = document.getElementById('statPairsKept');

  // Lightbox Modal Elements
  const lightboxModal = document.getElementById('lightboxModal');
  const lightboxBackdrop = document.getElementById('lightboxBackdrop');
  const lightboxClose = document.getElementById('lightboxClose');
  const lightboxImg = document.getElementById('lightboxImg');
  const lightboxBreedName = document.getElementById('lightboxBreedName');
  const btnClassifyFromGallery = document.getElementById('btnClassifyFromGallery');

  let currentFile = null;
  let allBreeds = [];
  let allCandidates = [];
  let currentAuditFilter = 'all';
  let currentActiveLightboxUrl = null;

  // =========================================================================
  // 1. Navigation / Mode Switcher
  // =========================================================================
  function switchMode(mode) {
    [tabClassifier, tabGallery, tabAudit].forEach(t => t && t.classList.remove('active'));
    [viewClassifier, viewGallery, viewAudit].forEach(v => v && v.classList.add('hidden'));

    if (mode === 'classifier') {
      tabClassifier.classList.add('active');
      viewClassifier.classList.remove('hidden');
    } else if (mode === 'gallery') {
      tabGallery.classList.add('active');
      viewGallery.classList.remove('hidden');
      if (breedSelect && !breedSelect.value && allBreeds.length > 0) {
        const defaultBreed = allBreeds.find(b => b.name.toLowerCase().includes('golden retriever')) || allBreeds[0];
        breedSelect.value = defaultBreed.id;
        loadBreedGallery(defaultBreed.id);
      }
    } else if (mode === 'audit') {
      tabAudit.classList.add('active');
      viewAudit.classList.remove('hidden');
      loadAuditCandidates();
    }
  }

  tabClassifier.addEventListener('click', () => switchMode('classifier'));
  tabGallery.addEventListener('click', () => switchMode('gallery'));
  if (tabAudit) tabAudit.addEventListener('click', () => switchMode('audit'));

  // =========================================================================
  // 2. Initialize Telemetry & Breeds List
  // =========================================================================
  fetch('/api/info')
    .then(r => r.json())
    .then(data => {
      if (data.model_name) {
        document.getElementById('modelBadge').innerHTML = 
          `<span class="status-dot"></span> Model: ${data.model_name}`;
        if (telModel) telModel.textContent = data.model_name;
      }
      if (data.total_classes) {
        const countStr = `${data.total_classes} Breeds`;
        if (telClasses) telClasses.textContent = countStr;
        if (classesBadge) classesBadge.textContent = `${countStr} Evaluated`;
      }
    })
    .catch(() => {});

  // Fetch 120 Breeds for the Dropdown & Explorer
  fetch('/api/breeds')
    .then(r => r.json())
    .then(breeds => {
      // Ensure allBreeds is strictly sorted alphabetically A-Z by display name
      allBreeds = [...breeds].sort((a, b) => a.name.localeCompare(b.name));
      if (galleryBreedCountText) {
        galleryBreedCountText.textContent = `${allBreeds.length} Breeds Available`;
      }
      populateBreedDropdown(allBreeds);
    })
    .catch(err => {
      console.error('Error fetching breeds:', err);
      if (galleryBreedCountText) galleryBreedCountText.textContent = '120 Breeds';
    });

  /**
   * Sorts the dropdown list alphabetically and removes the photo number from the text.
   * Only displays the clean breed name in the dropdown list.
   */
  function populateBreedDropdown(breeds) {
    const selectedVal = breedSelect.value;
    breedSelect.innerHTML = '<option value="" disabled selected>Select from 120 Dog Breeds (A–Z)...</option>';

    // Strictly sort alphabetically A-Z
    const sorted = [...breeds].sort((a, b) => a.name.localeCompare(b.name));

    sorted.forEach(breed => {
      const opt = document.createElement('option');
      opt.value = breed.id;
      // Requirement: remove the photo number from dropdown list text
      opt.textContent = breed.name;
      breedSelect.appendChild(opt);
    });

    if (selectedVal && sorted.some(b => b.id === selectedVal)) {
      breedSelect.value = selectedVal;
    }
  }

  /**
   * Helper function to check if the position of a phrase is not present in the text,
   * but first check if the first few letters match.
   *
   * @param {string} text - The target breed name (e.g. "Golden Retriever")
   * @param {string} phrase - The search input phrase (e.g. "gol")
   * @returns {Object} Match details including presence, prefix match status, and position index
   */
  function matchPhraseInText(text, phrase) {
    const target = text.toLowerCase();
    const query = phrase.toLowerCase();

    // 1. First check if the first few letters match
    const startsWithMatch = target.startsWith(query);

    // 2. Check if the position of the phrase is not present in the text
    const pos = target.indexOf(query);
    const isPresent = (pos !== -1);

    // Check if any individual word within the text starts with the phrase
    const words = target.split(/[\s-]+/);
    const wordStartMatch = words.some(w => w.startsWith(query));

    return {
      isPresent,          // false if position of phrase is not present in text
      startsWithMatch,    // true if first few letters match
      wordStartMatch,     // true if any word begins with phrase
      position: pos       // 0-based position index in text
    };
  }

  /**
   * Smart Search Ranking Algorithm:
   * Uses matchPhraseInText to filter and return an ordered result:
   * 1. Breeds where the first few letters match (Priority 1)
   * 2. Breeds where any word starts with the query (Priority 2)
   * 3. Breeds where the phrase appears later in the text, ordered by position (Priority 3)
   *
   * @param {Array} breeds - List of breed objects
   * @param {string} query - Search string
   * @returns {Array} Ordered list of matching breeds with match metadata
   */
  function smartSearchBreeds(breeds, query) {
    const q = query.trim();
    if (!q) {
      return [...breeds].sort((a, b) => a.name.localeCompare(b.name)).map(b => ({
        breed: b,
        priority: 0,
        pos: 0,
        matchType: 'all',
      }));
    }

    const matches = [];

    for (const breed of breeds) {
      const match = matchPhraseInText(breed.name, q);

      // Check if the position of phrase is not present in the text: exclude if not present
      if (!match.isPresent) {
        continue;
      }

      // Categorize priority based on whether first few letters matched
      let priority = 3;
      let matchType = 'Contains';

      if (match.startsWithMatch) {
        priority = 1;
        matchType = 'Starts with';
      } else if (match.wordStartMatch) {
        priority = 2;
        matchType = 'Word match';
      }

      matches.push({
        breed,
        priority,
        pos: match.position,
        matchType,
      });
    }

    // Return an ordered result:
    // 1. By priority: first few letters match > word start match > substring match
    // 2. By position of the phrase in the text (earlier occurrence ranks higher)
    // 3. Alphabetically by breed name
    matches.sort((a, b) => {
      if (a.priority !== b.priority) return a.priority - b.priority;
      if (a.pos !== b.pos) return a.pos - b.pos;
      return a.breed.name.localeCompare(b.breed.name);
    });

    return matches;
  }

  // Render smart suggestions dropdown under the search input
  function renderSuggestions(matchedItems, query) {
    if (!searchSuggestions) return;

    if (!query.trim() || matchedItems.length === 0) {
      searchSuggestions.classList.add('hidden');
      searchSuggestions.innerHTML = '';
      return;
    }

    searchSuggestions.innerHTML = '';
    searchSuggestions.classList.remove('hidden');

    matchedItems.slice(0, 8).forEach(({ breed, pos, matchType }) => {
      const item = document.createElement('div');
      item.className = 'suggestion-item';

      // Highlight matched characters in the breed name
      const original = breed.name;
      const qLen = query.trim().length;
      let highlightedHTML = original;
      if (pos >= 0 && pos + qLen <= original.length) {
        const before = original.substring(0, pos);
        const match = original.substring(pos, pos + qLen);
        const after = original.substring(pos + qLen);
        highlightedHTML = `${before}<span class="match-highlight">${match}</span>${after}`;
      }

      item.innerHTML = `
        <span>${highlightedHTML}</span>
        <span class="suggestion-priority-tag">${matchType}</span>
      `;

      item.addEventListener('click', () => {
        selectBreed(breed);
        searchSuggestions.classList.add('hidden');
      });

      searchSuggestions.appendChild(item);
    });
  }

  function selectBreed(breed) {
    breedSelect.value = breed.id;
    breedSearchInput.value = breed.name;
    btnClearSearch.classList.remove('hidden');
    if (searchSuggestions) searchSuggestions.classList.add('hidden');
    loadBreedGallery(breed.id);
  }

  // Search input handler with smart search and suggestions
  breedSearchInput.addEventListener('input', (e) => {
    const query = e.target.value;
    if (query.trim().length > 0) {
      btnClearSearch.classList.remove('hidden');
    } else {
      btnClearSearch.classList.add('hidden');
    }

    const orderedMatches = smartSearchBreeds(allBreeds, query);
    renderSuggestions(orderedMatches, query);

    // Also filter and re-order the dropdown options based on the smart search result
    if (query.trim()) {
      const matchedBreeds = orderedMatches.map(m => m.breed);
      populateBreedDropdown(matchedBreeds);
      if (matchedBreeds.length > 0) {
        breedSelect.value = matchedBreeds[0].id;
      }
    } else {
      populateBreedDropdown(allBreeds);
    }
  });

  // Keyboard navigation on search input
  breedSearchInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
      const query = breedSearchInput.value.trim();
      const orderedMatches = smartSearchBreeds(allBreeds, query);
      if (orderedMatches.length > 0) {
        selectBreed(orderedMatches[0].breed);
      }
    } else if (e.key === 'Escape') {
      if (searchSuggestions) searchSuggestions.classList.add('hidden');
    }
  });

  // Close suggestions when clicking outside
  document.addEventListener('click', (e) => {
    if (!e.target.closest('.search-box') && searchSuggestions) {
      searchSuggestions.classList.add('hidden');
    }
  });

  btnClearSearch.addEventListener('click', () => {
    breedSearchInput.value = '';
    btnClearSearch.classList.add('hidden');
    if (searchSuggestions) searchSuggestions.classList.add('hidden');
    populateBreedDropdown(allBreeds);
  });

  breedSelect.addEventListener('change', () => {
    if (breedSelect.value) {
      const selected = allBreeds.find(b => b.id === breedSelect.value);
      if (selected) {
        breedSearchInput.value = selected.name;
        btnClearSearch.classList.remove('hidden');
      }
      loadBreedGallery(breedSelect.value);
    }
  });

  // =========================================================================
  // 3. Load Breed Gallery
  // =========================================================================
  async function loadBreedGallery(breedId) {
    galleryGrid.innerHTML = `
      <div class="gallery-empty-state">
        <div class="btn-spinner" style="width: 32px; height: 32px; margin: 0 auto 12px auto;"></div>
        <p>Loading all images for ${breedId}...</p>
      </div>
    `;

    try {
      const res = await fetch(`/api/breeds/${encodeURIComponent(breedId)}/images`);
      if (!res.ok) throw new Error('Failed to load breed images');
      const data = await res.json();

      // Update banner
      breedBanner.classList.remove('hidden');
      breedBannerTitle.textContent = data.name;
      breedSynset.textContent = data.breed_id.split('-')[0] || data.breed_id;
      breedImageCount.textContent = `${data.total_images} Images`;

      // Render gallery grid
      galleryGrid.innerHTML = '';
      if (data.images.length === 0) {
        galleryGrid.innerHTML = '<p class="gallery-empty-state">No images found for this breed.</p>';
        return;
      }

      data.images.forEach((imgUrl, idx) => {
        const item = document.createElement('div');
        item.className = 'gallery-item';
        const filename = imgUrl.split('/').pop();

        item.innerHTML = `
          <img src="${imgUrl}" alt="${data.name}" class="gallery-img" loading="lazy">
          <div class="gallery-item-overlay">
            <span class="gallery-filename">${filename}</span>
            <button type="button" class="btn-classify-thumb" data-url="${imgUrl}" data-name="${filename}" title="Classify in model">
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg>
              <span>Classify Photo</span>
            </button>
          </div>
        `;

        // Click anywhere on tile to open Lightbox
        item.addEventListener('click', (e) => {
          if (e.target.closest('.btn-classify-thumb')) return;
          openLightbox(imgUrl, data.name);
        });

        // Click "Classify Photo" on thumbnail
        const classifyBtn = item.querySelector('.btn-classify-thumb');
        classifyBtn.addEventListener('click', (e) => {
          e.stopPropagation();
          classifyImageFromUrl(imgUrl, filename);
        });

        galleryGrid.appendChild(item);
      });
    } catch (err) {
      console.error('Gallery loading error:', err);
      galleryGrid.innerHTML = `<p class="gallery-empty-state">Error loading images: ${err.message}</p>`;
    }
  }

  // =========================================================================
  // 4. Lightbox Modal
  // =========================================================================
  function openLightbox(imgUrl, breedName) {
    currentActiveLightboxUrl = imgUrl;
    lightboxImg.src = imgUrl;
    lightboxBreedName.textContent = breedName;
    lightboxModal.classList.remove('hidden');
  }

  function closeLightbox() {
    lightboxModal.classList.add('hidden');
    currentActiveLightboxUrl = null;
  }

  lightboxClose.addEventListener('click', closeLightbox);
  lightboxBackdrop.addEventListener('click', closeLightbox);

  btnClassifyFromGallery.addEventListener('click', () => {
    if (currentActiveLightboxUrl) {
      const filename = currentActiveLightboxUrl.split('/').pop();
      closeLightbox();
      classifyImageFromUrl(currentActiveLightboxUrl, filename);
    }
  });

  async function classifyImageFromUrl(url, filename) {
    try {
      const res = await fetch(url);
      const blob = await res.blob();
      const file = new File([blob], filename, { type: 'image/jpeg' });
      switchMode('classifier');
      handleSelectedFile(file);
      classifyCurrentFile();
    } catch (err) {
      alert('Could not classify image: ' + err.message);
    }
  }

  // =========================================================================
  // 5. Classifier Upload & Drag-and-Drop
  // =========================================================================
  btnBrowse.addEventListener('click', (e) => {
    e.stopPropagation();
    fileInput.click();
  });

  dropzone.addEventListener('click', () => {
    if (!currentFile) fileInput.click();
  });

  fileInput.addEventListener('change', (e) => {
    if (e.target.files && e.target.files[0]) {
      handleSelectedFile(e.target.files[0]);
    }
  });

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

  function clearActiveSampleButtons() {
    sampleButtons.forEach(btn => btn.classList.remove('active'));
  }

  function handleSelectedFile(file) {
    if (!file.type.startsWith('image/')) {
      alert('Please upload a valid image file (JPEG, PNG, WEBP).');
      return;
    }

    currentFile = file;

    const reader = new FileReader();
    reader.onload = (e) => {
      imagePreview.src = e.target.result;
      previewFilename.textContent = file.name || 'uploaded_dog_image.jpg';
      dropzonePrompt.classList.add('hidden');
      previewContainer.classList.remove('hidden');
      btnClassify.disabled = false;
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
    clearActiveSampleButtons();
  });

  // =========================================================================
  // 6. 1-Click Quick Sample Testing
  // =========================================================================
  sampleButtons.forEach(btn => {
    btn.addEventListener('click', async () => {
      clearActiveSampleButtons();
      btn.classList.add('active');

      const url = btn.dataset.url;
      const name = (btn.dataset.name || 'sample').toLowerCase().replace(/\s+/g, '_') + '.jpg';

      try {
        const res = await fetch(url);
        if (!res.ok) throw new Error('Sample image could not be loaded');
        const blob = await res.blob();
        const file = new File([blob], name, { type: 'image/jpeg' });
        handleSelectedFile(file);
        classifyCurrentFile();
      } catch (err) {
        console.error('Failed to load sample image:', err);
        alert('Failed to load sample image: ' + err.message);
      }
    });
  });

  // =========================================================================
  // 7. Inference API Request (/api/predict)
  // =========================================================================
  async function classifyCurrentFile() {
    if (!currentFile) return;

    btnClassify.disabled = true;
    btnSpinner.classList.remove('hidden');
    btnText.textContent = 'Classifying 120 Breeds...';

    const formData = new FormData();
    formData.append('file', currentFile);

    try {
      const response = await fetch('/api/predict', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const err = await response.json();
        throw new Error(err.detail || 'Classification failed');
      }

      const result = await response.json();
      renderResults(result);
    } catch (error) {
      console.error('Inference error:', error);
      alert(`Classification Error: ${error.message}`);
    } finally {
      btnClassify.disabled = false;
      btnSpinner.classList.add('hidden');
      btnText.textContent = 'Classify Dog Breed';
    }
  }

  btnClassify.addEventListener('click', classifyCurrentFile);

  // =========================================================================
  // 8. Render Top-5 Predictions and Probabilities
  // =========================================================================
  function renderResults(data) {
    resultsIdle.classList.add('hidden');
    resultsActive.classList.remove('hidden');

    // Winner Hero Card
    topBreed.textContent = data.predicted_breed;
    topConfidence.textContent = `${data.confidence.toFixed(1)}%`;

    if (data.confidence >= 80) {
      winnerCaption.textContent = 'High confidence match across 120 breeds';
    } else if (data.confidence >= 50) {
      winnerCaption.textContent = 'Confident prediction';
    } else {
      winnerCaption.textContent = 'Top candidate among fine-grained breeds';
    }

    if (telLatency) telLatency.textContent = `⚡ ${data.latency_ms} ms`;
    if (telModel && data.model_name) telModel.textContent = data.model_name;
    if (telClasses && data.total_classes) telClasses.textContent = `${data.total_classes} Breeds`;

    // Render Top 5 Probability Breakdown Bars
    probBarsList.innerHTML = '';
    const topList = data.top5 || (data.breakdown ? data.breakdown.slice(0, 5) : []);

    topList.forEach((item, index) => {
      const rank = item.rank || (index + 1);
      const isTop = rank === 1;
      const probItem = document.createElement('div');
      probItem.className = `prob-item ${isTop ? 'is-top' : ''}`;

      probItem.innerHTML = `
        <div class="prob-header">
          <div class="prob-header-left">
            <span class="rank-badge">#${rank}</span>
            <span class="prob-breed-name">${item.breed}</span>
          </div>
          <span class="prob-pct">${item.probability.toFixed(1)}%</span>
        </div>
        <div class="prob-track">
          <div class="prob-fill" style="width: 0%"></div>
        </div>
      `;

      probBarsList.appendChild(probItem);

      // Trigger smooth width transition after DOM insertion
      setTimeout(() => {
        const fill = probItem.querySelector('.prob-fill');
        if (fill) {
          fill.style.width = `${Math.max(item.probability, 1.5)}%`;
        }
      }, 50 * (index + 1));
    });
  }

  // =========================================================================
  // 9. Dataset Deduplication & Conflict Audit Studio
  // =========================================================================

  async function loadAuditCandidates() {
    if (!auditPairsDeck) return;
    auditPairsDeck.innerHTML = `
      <div class="gallery-empty-state">
        <div class="btn-spinner" style="width: 32px; height: 32px; margin: 0 auto 12px auto;"></div>
        <p>Loading candidate duplicate pairs from audit engine...</p>
      </div>
    `;

    try {
      const [candidatesRes, summaryRes] = await Promise.all([
        fetch('/api/audit/candidates'),
        fetch('/api/audit/summary')
      ]);

      if (!candidatesRes.ok) throw new Error('Failed to load candidate duplicates');
      allCandidates = await candidatesRes.json();

      if (summaryRes.ok) {
        const summary = await summaryRes.json();
        updateAuditStats(summary);
      }

      renderAuditPairs();
    } catch (err) {
      console.error('Error loading audit candidates:', err);
      auditPairsDeck.innerHTML = `
        <div class="gallery-empty-state">
          <p style="color: #ef4444;">Error loading audit data: ${err.message}</p>
        </div>
      `;
    }
  }

  function updateAuditStats(summary) {
    if (statTotalPairs) statTotalPairs.textContent = summary.total_pairs;
    if (statPendingPairs) statPendingPairs.textContent = summary.pending_pairs;
    if (statPhotosToRemove) statPhotosToRemove.textContent = summary.unique_photos_to_remove;
    if (statPairsKept) statPairsKept.textContent = summary.kept_count;
    if (auditPendingBadge) {
      auditPendingBadge.textContent = summary.pending_pairs;
      auditPendingBadge.style.display = summary.pending_pairs > 0 ? 'inline-block' : 'none';
    }
  }

  function renderAuditPairs() {
    if (!auditPairsDeck) return;
    const searchVal = (auditSearchInput ? auditSearchInput.value : '').trim().toLowerCase();

    const filtered = allCandidates.filter(pair => {
      // Filter by category / status
      if (currentAuditFilter === 'label_conflict' && pair.category !== 'label_conflict') return false;
      if (currentAuditFilter === 'cross_split_leak' && pair.category !== 'cross_split_leak') return false;
      if (currentAuditFilter === 'same_split_duplicate' && pair.category !== 'same_split_duplicate') return false;
      if (currentAuditFilter === 'pending' && pair.status !== 'pending') return false;
      if (currentAuditFilter === 'reviewed' && pair.status === 'pending') return false;

      // Filter by search query
      if (searchVal) {
        const matchA = pair.photo_a.breed_name.toLowerCase().includes(searchVal) || pair.photo_a.key.toLowerCase().includes(searchVal);
        const matchB = pair.photo_b.breed_name.toLowerCase().includes(searchVal) || pair.photo_b.key.toLowerCase().includes(searchVal);
        if (!matchA && !matchB) return false;
      }
      return true;
    });

    if (filtered.length === 0) {
      auditPairsDeck.innerHTML = `
        <div class="gallery-empty-state">
          <p>No candidate pairs match the current filter.</p>
        </div>
      `;
      return;
    }

    auditPairsDeck.innerHTML = '';

    filtered.forEach(pair => {
      const card = document.createElement('div');
      const statusClass = pair.status.replace('approved_', '');
      card.className = `pair-card status-${statusClass}`;
      card.id = `pairCard-${pair.pair_id}`;

      // Status pill label
      let statusTagText = 'Pending Review';
      let statusTagClass = 'status-tag-pending';
      if (pair.status === 'approved_remove_a') {
        statusTagText = 'Removing Photo A';
        statusTagClass = 'status-tag-remove_a';
      } else if (pair.status === 'approved_remove_b') {
        statusTagText = 'Removing Photo B';
        statusTagClass = 'status-tag-remove_b';
      } else if (pair.status === 'approved_remove_both') {
        statusTagText = 'Removing Both Photos';
        statusTagClass = 'status-tag-remove_both';
      } else if (pair.status === 'rejected_keep_both') {
        statusTagText = 'Kept Both (Distinct)';
        statusTagClass = 'status-tag-kept';
      }

      const categoryLabel = pair.category === 'label_conflict' ? 'Label Conflict'
                          : pair.category === 'cross_split_leak' ? 'Cross-Split Leak'
                          : 'Same-Split Duplicate';

      const exactBadge = pair.exact_copy ? '<span class="dist-badge exact-match">Exact Copy</span>' : '';

      card.innerHTML = `
        <div class="pair-header">
          <div style="display: flex; align-items: center; gap: 10px;">
            <span class="pair-id-tag">Pair #${pair.pair_id}</span>
            <span class="pair-category-badge category-${pair.category}">${categoryLabel}</span>
            ${exactBadge}
          </div>
          <span class="pair-status-tag ${statusTagClass}" id="pairStatusTag-${pair.pair_id}">${statusTagText}</span>
        </div>

        <div class="pair-comparison-grid">
          <!-- Photo A Panel -->
          <div class="pair-photo-panel ${pair.status === 'approved_remove_a' || pair.status === 'approved_remove_both' ? 'is-marked-removal' : ''}" id="panelA-${pair.pair_id}">
            <div class="pair-photo-img-wrap">
              <img src="${pair.photo_a.url}" alt="${pair.photo_a.breed_name}" class="pair-photo-img" loading="lazy">
            </div>
            <div class="pair-photo-info">
              <span class="pair-breed-name">${pair.photo_a.breed_name}</span>
              <div class="pair-photo-sub">
                <span class="pile-pill pile-${pair.photo_a.pile}">${pair.photo_a.pile}</span>
                <span class="pair-filename" title="${pair.photo_a.key}">${pair.photo_a.key.split('/').pop()}</span>
              </div>
            </div>
          </div>

          <!-- Match Center Diagnostic Meta -->
          <div class="pair-center-meta">
            <span class="dist-badge">dHash: ${pair.dhash_distance} &bull; pHash: ${pair.phash_distance}</span>
            <p class="pair-reason-desc">${pair.reason}</p>
          </div>

          <!-- Photo B Panel -->
          <div class="pair-photo-panel ${pair.status === 'approved_remove_b' || pair.status === 'approved_remove_both' ? 'is-marked-removal' : ''}" id="panelB-${pair.pair_id}">
            <div class="pair-photo-img-wrap">
              <img src="${pair.photo_b.url}" alt="${pair.photo_b.breed_name}" class="pair-photo-img" loading="lazy">
            </div>
            <div class="pair-photo-info">
              <span class="pair-breed-name">${pair.photo_b.breed_name}</span>
              <div class="pair-photo-sub">
                <span class="pile-pill pile-${pair.photo_b.pile}">${pair.photo_b.pile}</span>
                <span class="pair-filename" title="${pair.photo_b.key}">${pair.photo_b.key.split('/').pop()}</span>
              </div>
            </div>
          </div>
        </div>

        <!-- 1-Click Action Buttons -->
        <div class="pair-actions-row">
          <button type="button" class="btn-action-decision btn-remove-a ${pair.status === 'approved_remove_a' ? 'active-decision' : ''}" data-pair="${pair.pair_id}" data-action="remove_a">
            <span>❌ Remove Photo A</span>
          </button>
          <button type="button" class="btn-action-decision btn-remove-b ${pair.status === 'approved_remove_b' ? 'active-decision' : ''}" data-pair="${pair.pair_id}" data-action="remove_b">
            <span>❌ Remove Photo B</span>
          </button>
          <button type="button" class="btn-action-decision btn-remove-both ${pair.status === 'approved_remove_both' ? 'active-decision' : ''}" data-pair="${pair.pair_id}" data-action="remove_both">
            <span>⚠️ Remove Both</span>
          </button>
          <button type="button" class="btn-action-decision btn-keep-both ${pair.status === 'rejected_keep_both' ? 'active-decision' : ''}" data-pair="${pair.pair_id}" data-action="keep_both">
            <span>✅ Keep Both (Distinct)</span>
          </button>
        </div>
      `;

      // Attach 1-click decision event listeners
      card.querySelectorAll('.btn-action-decision').forEach(btn => {
        btn.addEventListener('click', () => {
          const pairId = parseInt(btn.dataset.pair);
          const action = btn.dataset.action;
          handleAuditDecision(pairId, action);
        });
      });

      auditPairsDeck.appendChild(card);
    });
  }

  async function handleAuditDecision(pairId, action) {
    try {
      const res = await fetch('/api/audit/decision', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ pair_id: pairId, action: action })
      });

      if (!res.ok) throw new Error('Failed to save decision');
      const data = await res.json();

      // Update local object
      const candidate = allCandidates.find(c => c.pair_id === pairId);
      if (candidate) {
        candidate.status = data.new_status;
      }

      // Re-fetch summary stats
      const summaryRes = await fetch('/api/audit/summary');
      if (summaryRes.ok) {
        const summary = await summaryRes.json();
        updateAuditStats(summary);
      }

      // Re-render
      renderAuditPairs();
    } catch (err) {
      console.error('Audit decision error:', err);
      alert('Error updating decision: ' + err.message);
    }
  }

  // Filter Pills Handling
  if (auditFilterPills) {
    auditFilterPills.querySelectorAll('.filter-pill').forEach(pill => {
      pill.addEventListener('click', () => {
        auditFilterPills.querySelectorAll('.filter-pill').forEach(p => p.classList.remove('active'));
        pill.classList.add('active');
        currentAuditFilter = pill.dataset.filter;
        renderAuditPairs();
      });
    });
  }

  // Search input in audit studio
  if (auditSearchInput) {
    auditSearchInput.addEventListener('input', () => {
      renderAuditPairs();
    });
  }

  // Batch action: Apply recommended to all pending
  if (btnBatchRecommended) {
    btnBatchRecommended.addEventListener('click', async () => {
      if (!confirm('Apply all recommended duplicate/conflict actions to all pending pairs?')) return;
      btnBatchRecommended.disabled = true;
      try {
        const res = await fetch('/api/audit/batch', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ action: 'apply_recommended' })
        });
        if (!res.ok) throw new Error('Batch update failed');
        await loadAuditCandidates();
      } catch (err) {
        alert('Batch error: ' + err.message);
      } finally {
        btnBatchRecommended.disabled = false;
      }
    });
  }

  // Generate clean manifest button
  if (btnGenerateCleanManifest) {
    btnGenerateCleanManifest.addEventListener('click', async () => {
      btnGenerateCleanManifest.disabled = true;
      btnGenerateCleanManifest.innerHTML = '<span class="btn-spinner" style="width:14px;height:14px;"></span> Generating...';

      try {
        const res = await fetch('/api/audit/generate_clean_manifest', { method: 'POST' });
        if (!res.ok) {
          const err = await res.json();
          throw new Error(err.detail || 'Failed to generate clean manifest');
        }
        const data = await res.json();
        const s = data.summary;

        if (manifestAlert) {
          manifestAlert.classList.remove('hidden');
          manifestAlertText.textContent = 
            `Clean manifest generated (${s.clean_total} photos). Removed ${s.total_removed} photos: Train -${s.removed_per_split.train}, Val -${s.removed_per_split.val}, Test -${s.removed_per_split.test}.`;
        }
        alert(`Success! split_manifest_clean.json generated with ${s.clean_total} total images (${s.total_removed} duplicates removed).`);
      } catch (err) {
        alert('Clean manifest error: ' + err.message);
      } finally {
        btnGenerateCleanManifest.disabled = false;
        btnGenerateCleanManifest.innerHTML = `
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M19 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11l5 5v11a2 2 0 0 1-2 2z"/><polyline points="17 21 17 13 7 13 7 21"/><polyline points="7 3 7 8 15 8"/></svg>
          <span>Generate Clean Manifest</span>
        `;
      }
    });
  }

  // Initial fetch of audit summary to show badge count on initial page load
  fetch('/api/audit/summary')
    .then(r => r.json())
    .then(summary => {
      if (auditPendingBadge && summary.pending_pairs > 0) {
        auditPendingBadge.textContent = summary.pending_pairs;
        auditPendingBadge.style.display = 'inline-block';
      }
    })
    .catch(() => {});
});

