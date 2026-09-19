
(function () {
  'use strict';

  // Subsystem Configurations
  const SUBSYSTEMS = {
    door: {
      id: 'door',
      name: 'Door Subsystem',
      title: '2. Ingest Door Telemetry Data',
      hint: 'Upload continuous door sensor CSV stream to detect abnormal resistance cycles:',
      dropPrompt: 'Drag & Drop Continuous Telemetry File (.csv) Here',
      dropSubprompt: 'or click to browse your local continuous stream CSV',
      badge: '.CSV',
      accept: '.csv',
      multiple: false,
      btnLabel: '⚡ Run Door Prediction',
      tableTitle: 'door_predictions',
      defaultCsvName: 'door_predictions.csv'
    },
    acv: {
      id: 'acv',
      name: 'ACV (HVAC) Subsystem',
      title: '2. Ingest ACV Telemetry Data',
      hint: 'Upload your Excel (.xlsx) file to run fault localisation:',
      dropPrompt: 'Drag & Drop Telemetry File (.xlsx) Here',
      dropSubprompt: 'or click to browse your local Excel files',
      badge: '.XLSX',
      accept: '.xlsx',
      multiple: false,
      btnLabel: '⚡ Run ACV Prediction',
      tableTitle: 'Prediction Result',
      defaultCsvName: 'predictions.csv'
    },
    corrugation: {
      id: 'corrugation',
      name: 'Rail Corrugation Subsystem',
      disabled: true
    },
    shm: {
      id: 'shm',
      name: 'SHM (Structural Health Monitoring)',
      title: '2. Ingest SHM Dynamic Stress Data',
      hint: 'Upload dynamic stress CSV file(s) or a .zip archive to estimate cumulative fatigue damage:',
      dropPrompt: 'Drag & Drop Stress File(s) (.csv / .zip) Here',
      dropSubprompt: 'or click to select one or multiple test CSVs, or a .zip archive',
      badge: '.CSV, .ZIP',
      accept: '.csv,.zip',
      multiple: true,
      btnLabel: '⚡ Run SHM Prediction',
      tableTitle: 'shm_predictions',
      defaultCsvName: 'shm_predictions.csv'
    }
  };

  // State
  let currentSubsystemId = 'acv';
  let uploadedFiles = []; // array of { filename, content, size }
  let inferenceResult = null;

  // DOM Elements
  const subsystemCards = document.querySelectorAll('.subsystem-card');
  const ingestionTitle = document.getElementById('ingestion-section-title');
  const ingestionHint = document.getElementById('ingestion-section-hint');
  const dropzone = document.getElementById('dropzone');
  const dropzonePrompt = document.getElementById('dropzone-prompt');
  const dropzoneSubprompt = document.getElementById('dropzone-subprompt');
  const fileTypeBadge = document.getElementById('file-type-badge');
  const fileInput = document.getElementById('file-input');

  const fileInspector = document.getElementById('file-inspector');
  const fileNameDisplay = document.getElementById('file-name-display');
  const fileMetaDisplay = document.getElementById('file-meta-display');
  const btnRunInference = document.getElementById('btn-run-inference');

  const resultsContainer = document.getElementById('results-container');
  const verdictBanner = document.getElementById('verdict-banner');
  const verdictIcon = document.getElementById('verdict-icon');
  const verdictLabel = document.getElementById('verdict-label');
  const verdictTitle = document.getElementById('verdict-title');
  const verdictSummary = document.getElementById('verdict-summary');
  const scoreTitle = document.getElementById('score-title');
  const scoreVal = document.getElementById('score-val');

  const predictionTableTitle = document.getElementById('prediction-table-title');
  const officialTableHead = document.getElementById('official-table-head');
  const officialTableBody = document.getElementById('official-table-body');
  const btnExportCsv = document.getElementById('btn-export-csv');

  const secondaryPanel = document.getElementById('secondary-details-panel');
  const secondaryTitle = document.getElementById('secondary-details-title');
  const secondarySub = document.getElementById('secondary-details-sub');
  const secondaryTableHead = document.getElementById('secondary-table-head');
  const secondaryTableBody = document.getElementById('secondary-table-body');

  const btnReset = document.getElementById('btn-reset');
  const modalHelp = document.getElementById('modal-help');
  const btnHelp = document.getElementById('btn-help');
  const btnCloseModal = document.getElementById('btn-close-modal');

  const modalDiagnostic = document.getElementById('modal-diagnostic');
  const btnCloseDiagnostic = document.getElementById('btn-close-diagnostic');
  const btnDismissDiagnostic = document.getElementById('btn-dismiss-diagnostic');
  const btnCopyTraceback = document.getElementById('btn-copy-traceback');
  const diagnosticModule = document.getElementById('diagnostic-module');
  const diagnosticErrorMsg = document.getElementById('diagnostic-error-msg');
  const diagnosticHint = document.getElementById('diagnostic-hint');
  const diagnosticTraceback = document.getElementById('diagnostic-traceback');

  const backendStatusText = document.getElementById('backend-status-text');
  const backendStatusIndicator = document.getElementById('backend-status-indicator');

  // Initialization
  function init() {
    setupEventListeners();
    selectSubsystem('acv');
    checkBackendStatus();
  }

  async function checkBackendStatus() {
    try {
      const res = await fetch('/api/status');
      if (res.ok) {
        const data = await res.json();
        if (backendStatusText) {
          backendStatusText.textContent = 'Models Active: Door, ACV, SHM';
        }
        if (backendStatusIndicator) {
          backendStatusIndicator.title = 'Backend connected: Door, ACV, and SHM ready';
          const dot = backendStatusIndicator.querySelector('.pulse-dot');
          if (dot) dot.style.background = '#10b981';
        }
      }
    } catch (e) {
      if (backendStatusText) {
        backendStatusText.textContent = 'Server Offline (Start python3 app.py)';
      }
      if (backendStatusIndicator) {
        backendStatusIndicator.title = 'Run python3 app.py to connect models';
        const dot = backendStatusIndicator.querySelector('.pulse-dot');
        if (dot) dot.style.background = '#f59e0b';
      }
    }
  }

  function setupEventListeners() {
    // Subsystem Selection
    subsystemCards.forEach(card => {
      card.addEventListener('click', () => {
        const id = card.getAttribute('data-subsystem');
        if (card.classList.contains('disabled') || id === 'corrugation') {
          alert('Model not implemented for Rail Corrugation yet.');
          return;
        }
        selectSubsystem(id);
      });
    });

    // Drag & Drop
    ['dragenter', 'dragover'].forEach(eventName => {
      dropzone.addEventListener(eventName, (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropzone.classList.add('drag-over');
      });
    });

    ['dragleave', 'drop'].forEach(eventName => {
      dropzone.addEventListener(eventName, (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropzone.classList.remove('drag-over');
      });
    });

    dropzone.addEventListener('drop', (e) => {
      const dt = e.dataTransfer;
      const files = dt.files;
      if (files && files.length > 0) {
        handleFiles(Array.from(files));
      }
    });

    dropzone.addEventListener('click', () => {
      fileInput.click();
    });

    fileInput.addEventListener('change', (e) => {
      if (e.target.files && e.target.files.length > 0) {
        handleFiles(Array.from(e.target.files));
      }
    });

    // Run Inference
    btnRunInference.addEventListener('click', () => {
      executeInference();
    });

    // Export CSV
    if (btnExportCsv) {
      btnExportCsv.addEventListener('click', exportResultsCsv);
    }

    // Reset View
    if (btnReset) {
      btnReset.addEventListener('click', resetSession);
    }

    // Help Modal
    if (btnHelp && modalHelp && btnCloseModal) {
      btnHelp.addEventListener('click', () => modalHelp.classList.add('open'));
      btnCloseModal.addEventListener('click', () => modalHelp.classList.remove('open'));
      modalHelp.addEventListener('click', (e) => {
        if (e.target === modalHelp) modalHelp.classList.remove('open');
      });
    }

    // Diagnostic Modal
    if (modalDiagnostic) {
      if (btnCloseDiagnostic) btnCloseDiagnostic.addEventListener('click', () => modalDiagnostic.classList.remove('open'));
      if (btnDismissDiagnostic) btnDismissDiagnostic.addEventListener('click', () => modalDiagnostic.classList.remove('open'));
      modalDiagnostic.addEventListener('click', (e) => {
        if (e.target === modalDiagnostic) modalDiagnostic.classList.remove('open');
      });
    }

    if (btnCopyTraceback) {
      btnCopyTraceback.addEventListener('click', () => {
        if (diagnosticTraceback) {
          navigator.clipboard.writeText(diagnosticTraceback.textContent);
          btnCopyTraceback.textContent = '✓ Copied!';
          setTimeout(() => { btnCopyTraceback.textContent = '📋 Copy Traceback'; }, 2000);
        }
      });
    }
  }

  function selectSubsystem(subsystemId) {
    currentSubsystemId = subsystemId;
    const sub = SUBSYSTEMS[subsystemId];
    if (!sub) return;

    // Update active card
    subsystemCards.forEach(card => {
      const id = card.getAttribute('data-subsystem');
      if (id === subsystemId) {
        card.classList.add('active');
      } else {
        card.classList.remove('active');
      }
    });

    // Update Ingestion Text and Dropzone configuration
    if (ingestionTitle) ingestionTitle.textContent = sub.title;
    if (ingestionHint) ingestionHint.textContent = sub.hint;
    if (dropzonePrompt) dropzonePrompt.textContent = sub.dropPrompt;
    if (dropzoneSubprompt) dropzoneSubprompt.textContent = sub.dropSubprompt;
    if (fileTypeBadge) fileTypeBadge.textContent = sub.badge;

    fileInput.accept = sub.accept;
    fileInput.multiple = Boolean(sub.multiple);

    if (btnRunInference) {
      btnRunInference.textContent = sub.btnLabel;
    }

    resetSession();
  }

  function resetSession() {
    uploadedFiles = [];
    inferenceResult = null;
    fileInspector.style.display = 'none';
    btnRunInference.disabled = true;
    resultsContainer.style.display = 'none';
    fileInput.value = '';
  }

  // File Ingestion
  function readFileAsDataUrl(file) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = (e) => resolve(e.target.result);
      reader.onerror = () => reject(new Error(`Failed to read file: ${file.name}`));
      reader.readAsDataURL(file);
    });
  }

  async function handleFiles(files) {
    if (!files || files.length === 0) return;
    const sub = SUBSYSTEMS[currentSubsystemId];

    // Validate file extensions
    const validFiles = files.filter(f => {
      const ext = '.' + f.name.split('.').pop().toLowerCase();
      if (currentSubsystemId === 'acv') return ext === '.xlsx';
      if (currentSubsystemId === 'door') return ext === '.csv';
      if (currentSubsystemId === 'shm') return ext === '.csv' || ext === '.zip';
      return false;
    });

    if (validFiles.length === 0) {
      alert(`Invalid file type selected. Expected: ${sub.badge} for ${sub.name}.`);
      return;
    }

    uploadedFiles = [];
    fileNameDisplay.textContent = 'Reading file(s)...';
    fileMetaDisplay.textContent = 'Processing upload';
    fileInspector.style.display = 'flex';
    btnRunInference.disabled = true;

    try {
      for (const file of validFiles) {
        const content = await readFileAsDataUrl(file);
        uploadedFiles.push({
          filename: file.name,
          content: content,
          size: file.size
        });
      }

      if (uploadedFiles.length === 1) {
        const f = uploadedFiles[0];
        fileNameDisplay.textContent = f.filename;
        fileMetaDisplay.textContent = `${(f.size / 1024).toFixed(1)} KB • Ready for ${sub.name}`;
      } else {
        const totalSize = uploadedFiles.reduce((acc, f) => acc + f.size, 0);
        fileNameDisplay.textContent = `${uploadedFiles.length} files selected (Batch)`;
        fileMetaDisplay.textContent = `${(totalSize / 1024).toFixed(1)} KB total • Ready for ${sub.name}`;
      }

      btnRunInference.disabled = false;
    } catch (err) {
      alert('Error reading file(s): ' + err.message);
      resetSession();
    }
  }

  // Model Execution
  async function executeInference() {
    if (uploadedFiles.length === 0) return;

    const sub = SUBSYSTEMS[currentSubsystemId];
    btnRunInference.disabled = true;
    btnRunInference.innerHTML = `<span class="spinner"></span> Running ${sub.name} Model...`;

    try {
      let payload;
      if (currentSubsystemId === 'shm' && uploadedFiles.length > 1) {
        payload = {
          subsystem: 'shm',
          filename: 'batch_upload.csv',
          files: uploadedFiles.map(f => ({
            filename: f.filename,
            file_content: f.content
          }))
        };
      } else {
        const f = uploadedFiles[0];
        payload = {
          subsystem: currentSubsystemId,
          filename: f.filename,
          file_content: f.content
        };
      }

      const response = await fetch('/api/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-Subsystem': currentSubsystemId,
          'X-Filename': payload.filename
        },
        body: JSON.stringify(payload)
      });

      const result = await response.json();

      if (response.ok && result.success) {
        inferenceResult = result;
        renderResults(result);
      } else {
        showDiagnosticModal(result);
      }
    } catch (netErr) {
      console.error('Backend connection error:', netErr);
      showDiagnosticModal({
        subsystem: currentSubsystemId,
        error_type: 'ConnectionError',
        error: 'Unable to reach the Python backend server at /api/predict. Please verify that python3 app.py is running.',
        hint: 'Start the server in your terminal with: python3 app.py',
        traceback: netErr.stack || netErr.toString()
      });
    } finally {
      btnRunInference.disabled = false;
      btnRunInference.textContent = sub.btnLabel;
    }
  }

  function showDiagnosticModal(errData) {
    if (!modalDiagnostic) return;
    const subId = errData.subsystem || currentSubsystemId;
    if (diagnosticModule) {
      if (subId === 'door') diagnosticModule.textContent = 'models/Door/door_model.json';
      else if (subId === 'shm') diagnosticModule.textContent = 'models/SHM/shm_model.json';
      else diagnosticModule.textContent = 'Optional_Items/ACV/code/src/pipeline.py';
    }
    if (diagnosticErrorMsg) diagnosticErrorMsg.textContent = `${errData.error_type || 'Error'}: ${errData.error || 'Execution failed'}`;
    if (diagnosticHint) diagnosticHint.textContent = `Ensure the input file is formatted properly for the ${subId.toUpperCase()} model.`;
    if (diagnosticTraceback) diagnosticTraceback.textContent = errData.traceback || 'No traceback provided.';
    modalDiagnostic.classList.add('open');
  }

  // Results Rendering
  function renderResults(result) {
    if (currentSubsystemId === 'acv') {
      renderAcvResults(result);
    } else if (currentSubsystemId === 'door') {
      renderDoorResults(result);
    } else if (currentSubsystemId === 'shm') {
      renderShmResults(result);
    }

    resultsContainer.style.display = 'block';
    resultsContainer.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  // 1. Render Door Results
  function renderDoorResults(data) {
    // Verdict Banner
    verdictBanner.className = 'verdict-banner ' + (data.status === 'warning' ? 'banner-warning' : 'banner-healthy');
    verdictIcon.textContent = '🚪';
    verdictLabel.textContent = 'DOOR CYCLE SEGMENTATION & RESISTANCE';
    verdictTitle.textContent = data.headline;
    verdictSummary.textContent = data.summary;
    scoreTitle.textContent = 'Abnormal Resistance';
    scoreVal.textContent = `${data.abnormal_cycles} / ${data.total_cycles}`;

    // Official Table: start_time, end_time, prediction
    predictionTableTitle.textContent = 'door_predictions';
    officialTableHead.innerHTML = `
      <tr>
        <th style="width: 50px;">#</th>
        <th>start_time</th>
        <th>end_time</th>
        <th>prediction</th>
      </tr>
    `;

    officialTableBody.innerHTML = '';
    const predictions = data.predictions || [];
    predictions.forEach((row, idx) => {
      const isAbnormal = (row.prediction === 'Abnormal resistance');
      const tr = document.createElement('tr');
      if (isAbnormal) {
        tr.style.background = '#fef2f2';
      }
      tr.innerHTML = `
        <td class="row-num">${idx + 1}</td>
        <td><strong>${escapeHtml(row.start_time)}</strong></td>
        <td>${escapeHtml(row.end_time)}</td>
        <td>
          <span class="task-tag" style="${isAbnormal ? 'background: #fee2e2; color: #b91c1c; border-color: #fca5a5;' : 'background: #f0fdf4; color: #15803d; border-color: #bbf7d0;'}">
            ${isAbnormal ? '⚠️ Abnormal resistance' : '✓ Normal'}
          </span>
        </td>
      `;
      officialTableBody.appendChild(tr);
    });

    // Secondary Panel: Cycle details
    if (secondaryPanel) {
      secondaryPanel.style.display = 'block';
      secondaryTitle.textContent = 'Cycle Telemetry & Motor Current Details';
      secondarySub.textContent = 'Segmented cycles with motor current (mA) and operation classification';
      secondaryTableHead.innerHTML = `
        <tr>
          <th>Cycle #</th>
          <th>Operation</th>
          <th>Start Time</th>
          <th>End Time</th>
          <th>Mean Motor Current</th>
          <th>Evaluation</th>
        </tr>
      `;
      secondaryTableBody.innerHTML = '';
      predictions.forEach((row, idx) => {
        const isAbnormal = (row.prediction === 'Abnormal resistance');
        const tr = document.createElement('tr');
        tr.innerHTML = `
          <td><strong>Cycle ${idx + 1}</strong></td>
          <td><span class="task-tag">${row.operation || 'Cycle'}</span></td>
          <td>${escapeHtml(row.start_time)}</td>
          <td>${escapeHtml(row.end_time)}</td>
          <td><strong>${row.current_mean ? row.current_mean + ' mA' : '--'}</strong></td>
          <td style="color: ${isAbnormal ? '#b91c1c' : '#15803d'}; font-weight: 600;">
            ${row.prediction}
          </td>
        `;
        secondaryTableBody.appendChild(tr);
      });
    }
  }

  // 2. Render ACV Results
  function renderAcvResults(data) {
    // Verdict Banner
    verdictBanner.className = 'verdict-banner banner-warning';
    verdictIcon.textContent = '❄️';
    verdictLabel.textContent = 'ACV REFRIGERANT LEAK LOCALISATION';
    verdictTitle.textContent = `Suspected Faulty Car: Car ${data.faulty_car}`;
    verdictSummary.textContent = `Full Ranking: ${data.ranking_string}`;
    scoreTitle.textContent = 'Most Likely Faulty Car';
    scoreVal.textContent = `Car ${data.faulty_car}`;

    // Official Table: file_id, ranked_cars
    predictionTableTitle.textContent = 'Prediction Result';
    officialTableHead.innerHTML = `
      <tr>
        <th style="width: 50px;">#</th>
        <th>file_id</th>
        <th>ranked_cars</th>
      </tr>
    `;

    officialTableBody.innerHTML = `
      <tr>
        <td class="row-num">1</td>
        <td><strong>${escapeHtml(data.file_id)}</strong></td>
        <td style="font-family: monospace; font-weight: 600; color: #1e293b;">${escapeHtml(data.ranking_string)}</td>
      </tr>
    `;

    // Secondary Panel: Detailed Car Scores
    if (secondaryPanel) {
      secondaryPanel.style.display = 'block';
      secondaryTitle.textContent = 'Car Fault Ranking Details';
      secondarySub.textContent = 'Fault scores based on mean cooling error (°C)';
      secondaryTableHead.innerHTML = `
        <tr>
          <th>Rank</th>
          <th>Car ID</th>
          <th>Fault Score (Mean Cooling Error)</th>
          <th>Status</th>
        </tr>
      `;
      secondaryTableBody.innerHTML = '';
      if (Array.isArray(data.scores)) {
        data.scores.forEach(item => {
          const isTopFault = (String(item.car_id) === String(data.faulty_car) || item.rank === 1);
          const tr = document.createElement('tr');
          if (isTopFault) tr.style.background = '#fef2f2';

          const scoreVal = (typeof item.fault_score === 'number') ? item.fault_score.toFixed(4) : item.fault_score;
          tr.innerHTML = `
            <td><strong>#${item.rank}</strong></td>
            <td><strong>Car ${item.car_id}</strong></td>
            <td>${scoreVal} °C</td>
            <td>
              <span class="task-tag" style="${isTopFault ? 'background: #fee2e2; color: #b91c1c; border-color: #fca5a5;' : 'background: #f0fdf4; color: #15803d; border-color: #bbf7d0;'}">
                ${isTopFault ? '⚠️ Suspected Leak' : '✓ Normal'}
              </span>
            </td>
          `;
          secondaryTableBody.appendChild(tr);
        });
      }
    }
  }

  // 3. Render SHM Results
  function renderShmResults(data) {
    // Verdict Banner
    verdictBanner.className = 'verdict-banner ' + (data.status === 'critical' ? 'banner-critical' : (data.status === 'warning' ? 'banner-warning' : 'banner-healthy'));
    verdictIcon.textContent = '🏗️';
    verdictLabel.textContent = 'SHM CUMULATIVE FATIGUE DAMAGE';
    verdictTitle.textContent = data.headline;
    verdictSummary.textContent = data.summary;
    scoreTitle.textContent = 'Peak Fatigue Damage';
    scoreVal.textContent = typeof data.max_damage === 'number' ? data.max_damage.toFixed(4) : data.max_damage;

    // Official Table: file_id, prediction
    predictionTableTitle.textContent = 'shm_predictions';
    officialTableHead.innerHTML = `
      <tr>
        <th style="width: 50px;">#</th>
        <th>file_id</th>
        <th>prediction</th>
      </tr>
    `;

    officialTableBody.innerHTML = '';
    const predictions = data.predictions || [];
    predictions.forEach((row, idx) => {
      const predVal = typeof row.prediction === 'number' ? row.prediction.toFixed(4) : row.prediction;
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td class="row-num">${idx + 1}</td>
        <td><strong>${escapeHtml(row.file_id)}</strong></td>
        <td style="font-family: monospace; font-weight: 600; color: #1e293b;">${predVal}</td>
      </tr>
    `;
      officialTableBody.appendChild(tr);
    });

    // Hide secondary panel for SHM to keep minimal clean presentation
    if (secondaryPanel) {
      secondaryPanel.style.display = 'none';
    }
  }

  // CSV Export
  function exportResultsCsv() {
    if (!inferenceResult || !inferenceResult.csv_data) {
      alert('No prediction results available to export.');
      return;
    }

    const sub = SUBSYSTEMS[currentSubsystemId];
    let exportFilename = sub.defaultCsvName;
    if (currentSubsystemId === 'acv' && inferenceResult.file_id) {
      exportFilename = `${inferenceResult.file_id.replace(/\.[^/.]+$/, '')}_predictions.csv`;
    }

    const blob = new Blob([inferenceResult.csv_data], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement('a');
    const url = URL.createObjectURL(blob);
    link.setAttribute('href', url);
    link.setAttribute('download', exportFilename);
    link.style.visibility = 'hidden';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  }

  function escapeHtml(str) {
    if (str === null || str === undefined) return '';
    return String(str)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#039;');
  }

  // Boot
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
