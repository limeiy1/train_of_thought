
(function () {
  'use strict';

  // ============================================================================
  // 1. Subsystem Definitions & Configurations
  // ============================================================================
  const SUBSYSTEMS = {
    door: {
      id: 'door',
      name: 'Door Subsystem',
      task: 'Temporal Segment Detection',
      icon: '🚪',
      fileExt: '.csv',
      description: 'Detects each door open/close cycle from continuous telemetry and classifies cycles as Normal vs Abnormal-Resistance (jamming/wear).'
    },
    acv: {
      id: 'acv',
      name: 'ACV (HVAC) Subsystem',
      task: 'Fault Diagnosis / Localisation',
      icon: '❄️',
      fileExt: '.xlsx',
      description: 'Multi-car thermodynamic comparison to identify and localize the specific rail car suffering from a refrigerant leak.'
    },
    corrugation: {
      id: 'corrugation',
      name: 'Rail Corrugation Subsystem',
      task: 'Multi-Class Classification',
      icon: '🛤️',
      fileExt: '.csv',
      description: 'Multi-channel axle-box vibration and shock analysis to classify track condition into Normal, Side I, or Side II Corrugation.'
    },
    shm: {
      id: 'shm',
      name: 'SHM (Structural Health)',
      task: 'Fatigue Damage Regression',
      icon: '🏗️',
      fileExt: '.csv',
      description: 'ASTM E1049 Rainflow cycle counting and Miner\'s-rule cumulative fatigue damage estimation from dynamic stress time series.'
    }
  };

  // State
  let currentSubsystemId = 'acv';
  let inferenceResult = null;
  let subsystemStatusCache = null;

  // Per-tab state so switching subsystems doesn't lose an uploaded file or result --
  // each subsystem keeps its own independent upload + prediction.
  const tabState = { door: null, acv: null, corrugation: null, shm: null };

  // DOM Elements
  const subsystemCards = document.querySelectorAll('.subsystem-card');
  const dropzone = document.getElementById('dropzone');
  const fileInput = document.getElementById('file-input');
  const fileInspector = document.getElementById('file-inspector');
  const fileNameDisplay = document.getElementById('file-name-display');
  const fileMetaDisplay = document.getElementById('file-meta-display');
  const btnRunInference = document.getElementById('btn-run-inference');
  const resultsContainer = document.getElementById('results-container');

  const btnExportCsv = document.getElementById('btn-export-csv');
  const btnReset = document.getElementById('btn-reset');
  const modalHelp = document.getElementById('modal-help');
  const btnHelp = document.getElementById('btn-help');
  const btnCloseModal = document.getElementById('btn-close-modal');

  // New generic / methodology elements
  const ingestionTitle = document.getElementById('ingestion-title');
  const ingestionHint = document.getElementById('ingestion-hint');
  const dropzonePrompt = document.getElementById('dropzone-prompt');
  const dropzoneBadge = document.getElementById('dropzone-badge');
  const methodologySummary = document.getElementById('methodology-summary');
  const methodologyValidation = document.getElementById('methodology-validation');
  const exportTableHead = document.getElementById('export-table-head');
  const exportTableBody = document.getElementById('export-table-body');
  const detailTablePanel = document.getElementById('detail-table-panel');
  const detailTableTitle = document.getElementById('detail-table-title');
  const detailTableSub = document.getElementById('detail-table-sub');
  const scoresTableHead = document.getElementById('scores-table-head');
  const verdictIcon = document.getElementById('verdict-icon');
  const verdictLabel = document.getElementById('verdict-label');
  const scoreTitle = document.getElementById('score-title');

  // Strict Mode Diagnostic Modal elements
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

  // ============================================================================
  // 2. Initialization & Navigation
  // ============================================================================
  function init() {
    setupEventListeners();
    checkBackendStatus();
  }

  async function checkBackendStatus() {
    try {
      const res = await fetch('/api/status');
      if (res.ok) {
        const data = await res.json();
        subsystemStatusCache = data.subsystems || null;
        if (backendStatusText) {
          const readyCount = Object.values(subsystemStatusCache || {}).filter(s => s.available).length;
          backendStatusText.textContent = `${readyCount}/4 subsystem models active`;
        }
        if (backendStatusIndicator) {
          backendStatusIndicator.title = 'Live connection to the Python backend (app/app.py)';
        }
        renderMethodologyPanel(currentSubsystemId);
      }
    } catch (e) {
      if (backendStatusText) {
        backendStatusText.textContent = 'Server Offline (Run python3 app/app.py)';
      }
      if (backendStatusIndicator) {
        backendStatusIndicator.title = 'Please start the backend server with python3 app/app.py';
        const dot = backendStatusIndicator.querySelector ? backendStatusIndicator.querySelector('.pulse-dot') : null;
        if (dot) dot.style.background = '#f59e0b';
      }
    }
  }

  function renderMethodologyPanel(subsystemId) {
    if (!subsystemStatusCache) return;
    const info = subsystemStatusCache[subsystemId];
    if (!info) return;

    if (subsystemId === 'acv') {
      if (methodologySummary) methodologySummary.textContent = 'Compares thermodynamic cooling pull-down behaviour across every car in the uploaded file: cabin temperature, ambient temperature, target setpoint, and compressor power are tracked over time for each car. A healthy car reaches its target temperature and its compressor load tapers off; a car with a refrigerant leak keeps drawing power without ever closing the gap to target. The car with the largest sustained cooling deficit relative to target, despite continuous compressor operation, is ranked most likely to have a refrigerant leak, with the remaining cars ranked by the same deficit measure. This is a deterministic rule rather than a fitted model, so it is validated directly against labelled cases rather than by cross-validation.';
      if (methodologyValidation) methodologyValidation.textContent = info.available ? 'Training validation: 0.833 rank-decay (5/6 cases perfect)' : info.status;
      return;
    }

    if (!info.methodology) {
      if (methodologySummary) methodologySummary.textContent = info.status || 'Methodology unavailable.';
      if (methodologyValidation) methodologyValidation.textContent = '';
      return;
    }

    if (methodologySummary) methodologySummary.textContent = info.methodology.summary;
    if (methodologyValidation) methodologyValidation.textContent = info.methodology.validation_result || '';
  }

  function setupEventListeners() {
    // Subsystem Card selection
    subsystemCards.forEach(card => {
      card.addEventListener('click', () => {
        const id = card.getAttribute('data-subsystem');
        if (!SUBSYSTEMS[id] && id !== 'acv') return;
        selectSubsystem(id);
      });
    });

    // Drag and Drop
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
      if (dt.files && dt.files.length > 0) {
        handleFiles(dt.files);
      }
    });

    dropzone.addEventListener('click', () => {
      fileInput.click();
    });

    fileInput.addEventListener('change', (e) => {
      if (e.target.files && e.target.files.length > 0) {
        handleFiles(e.target.files);
      }
    });

    // Run Inference
    btnRunInference.addEventListener('click', () => {
      executeInference();
    });

    // Export & Action Buttons
    if (btnExportCsv) {
      btnExportCsv.addEventListener('click', exportResultsCsv);
    }
    const btnExportJson = document.getElementById('btn-export-json');
    if (btnExportJson) {
      btnExportJson.addEventListener('click', exportResultsJson);
    }
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

    // Strict Mode Diagnostic Modal
    if (modalDiagnostic) {
      if (btnCloseDiagnostic) btnCloseDiagnostic.addEventListener('click', () => modalDiagnostic.classList.remove('open'));
      if (btnDismissDiagnostic) btnDismissDiagnostic.addEventListener('click', () => modalDiagnostic.classList.remove('open'));
      modalDiagnostic.addEventListener('click', (e) => {
        if (e.target === modalHelp) modalHelp.classList.remove('open');
      });
    }

    // Strict Mode Diagnostic Modal
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

    // Filter input in table
    const tableFilter = document.getElementById('table-filter');
    if (tableFilter) {
      tableFilter.addEventListener('input', (e) => {
        filterTableRows(e.target.value.toLowerCase());
      });
    }

    // Handle window resize for charts
    window.addEventListener('resize', debounce(() => {
      if (primaryChart) primaryChart.render();
      if (secondaryChart) secondaryChart.render();
    }, 150));
  }

  function selectSubsystem(subsystemId) {
    currentSubsystemId = subsystemId;
    const sub = subsystemId === 'acv'
      ? { name: 'ACV (HVAC) Subsystem', fileExt: '.xlsx' }
      : SUBSYSTEMS[subsystemId];

    // Update active card styling
    subsystemCards.forEach(card => {
      if (card.getAttribute('data-subsystem') === subsystemId) {
        card.classList.add('active');
      } else {
        card.classList.remove('active');
      }
    });

    // Update ingestion panel copy + dropzone accept type for this subsystem
    if (ingestionTitle) ingestionTitle.textContent = `2. Ingest ${sub.name} Data`;
    if (ingestionHint) ingestionHint.textContent = `Upload your ${sub.fileExt} file to run inference:`;
    if (dropzonePrompt) dropzonePrompt.textContent = `Drag & Drop File (${sub.fileExt}) Here`;
    if (dropzoneBadge) dropzoneBadge.textContent = sub.fileExt.replace('.', '').toUpperCase();
    if (fileInput) fileInput.setAttribute('accept', sub.fileExt);

    renderMethodologyPanel(subsystemId);

    // Restore this tab's own uploaded file / result (each subsystem keeps its own)
    restoreTabState(subsystemId);
  }

  function restoreTabState(subsystemId) {
    const saved = tabState[subsystemId];
    fileInput.value = '';
    const fileListDisplay = document.getElementById('file-list-display');

    if (!saved || !saved.files || saved.files.length === 0) {
      inferenceResult = null;
      fileInspector.style.display = 'none';
      btnRunInference.disabled = true;
      resultsContainer.style.display = 'none';
      if (fileListDisplay) fileListDisplay.innerHTML = '';
      return;
    }

    inferenceResult = saved.results ? saved.results : null;

    const n = saved.files.length;
    fileNameDisplay.textContent = n === 1 ? saved.files[0].fileName : `${n} files selected`;
    fileMetaDisplay.textContent = saved.results ? `Ready for ${subsystemId.toUpperCase()} inference (prediction already run)` : `Ready for ${subsystemId.toUpperCase()} inference`;
    if (fileListDisplay) {
      fileListDisplay.innerHTML = saved.files.map(f => `<li>${escapeHtml(f.fileName)}</li>`).join('');
    }
    fileInspector.style.display = 'flex';
    btnRunInference.disabled = false;

    if (saved.results) {
      renderBatchResults(subsystemId, saved.results);
    } else {
      resultsContainer.style.display = 'none';
    }
  }

  function resetSession() {
    tabState[currentSubsystemId] = null;
    restoreTabState(currentSubsystemId);
  }

  // ============================================================================
  // 3. File Ingestion & Parsing
  // ============================================================================
  function readFileAsDataUrl(file) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = (e) => resolve(e.target.result);
      reader.onerror = () => reject(new Error(`Failed to read ${file.name}`));
      reader.readAsDataURL(file);
    });
  }

  async function handleFiles(fileList) {
    const files = Array.from(fileList);
    if (files.length === 0) return;

    const expectedExt = currentSubsystemId === 'acv' ? '.xlsx' : (SUBSYSTEMS[currentSubsystemId] || {}).fileExt || '.csv';
    const invalid = files.filter(f => !f.name.toLowerCase().endsWith(expectedExt));
    if (invalid.length > 0) {
      alert(`Please upload only ${expectedExt} files for the ${currentSubsystemId.toUpperCase()} subsystem.\nRejected: ${invalid.map(f => f.name).join(', ')}`);
      return;
    }

    try {
      const readFiles = await Promise.all(files.map(async f => ({
        fileName: f.name,
        rawData: await readFileAsDataUrl(f)
      })));

      tabState[currentSubsystemId] = { files: readFiles, results: null };
      inferenceResult = null;
      restoreTabState(currentSubsystemId);
    } catch (err) {
      alert('Error reading one or more selected files. Please try again.');
    }
  }

  function parseCsvOrJson(text) {
    text = text.trim();
    // Try JSON first
    if (text.startsWith('[') || text.startsWith('{')) {
      try {
        const parsed = JSON.parse(text);
        if (Array.isArray(parsed)) return parsed;
        if (typeof parsed === 'object') {
          // Columnar format check
          const keys = Object.keys(parsed);
          if (keys.length > 0 && Array.isArray(parsed[keys[0]])) {
            const len = parsed[keys[0]].length;
            const rows = [];
            for (let i = 0; i < len; i++) {
              const row = {};
              keys.forEach(k => row[k] = parsed[k][i]);
              rows.push(row);
            }
            return rows;
          }
        }
      } catch (err) {
        // Fall back to CSV parsing
      }
    }

    // Robust CSV parser
    const lines = text.split(/\r?\n/).filter(line => line.trim().length > 0);
    if (lines.length < 2) return [];

    // Detect delimiter
    const firstLine = lines[0];
    let delimiter = ',';
    if (firstLine.includes('\t')) delimiter = '\t';
    else if (firstLine.includes(';') && !firstLine.includes(',')) delimiter = ';';

    const headers = splitCsvLine(firstLine, delimiter).map(h => h.trim().toLowerCase());
    const records = [];

    for (let i = 1; i < lines.length; i++) {
      const vals = splitCsvLine(lines[i], delimiter);
      if (vals.length < headers.length) continue;
      const row = {};
      headers.forEach((h, idx) => {
        let val = vals[idx] !== undefined ? vals[idx].trim() : '';
        const num = Number(val);
        row[h] = (!isNaN(num) && val !== '') ? num : val;
      });
      records.push(row);
    }

    return records;
  }

  function splitCsvLine(line, delimiter) {
    const pattern = new RegExp(
      "(\\" + delimiter + "|\\r?\\n|\\r|^)" +
      "(?:\"([^\"]*(?:\"\"[^\"]*)*)\"|" +
      "([^\"\\" + delimiter + "\\r\\n]*))",
      "gi"
    );
    const result = [];
    let match = null;
    while ((match = pattern.exec(line))) {
      let matchedValue;
      if (match[2]) {
        matchedValue = match[2].replace(new RegExp('""', 'g'), '"');
      } else {
        matchedValue = match[3];
      }
      result.push(matchedValue);
    }
    return result;
  }

  // ============================================================================
  // 5. Condition Monitoring Inference Engines
  // ============================================================================
  async function predictOneFile(subsystemId, fileName, rawData) {
    const response = await fetch('/api/predict', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-Subsystem': subsystemId,
        'X-Filename': fileName
      },
      body: JSON.stringify({ subsystem: subsystemId, filename: fileName, file_content: rawData })
    });
    const result = await response.json();
    if (!response.ok || !result.success) {
      const err = new Error(result.error || 'Prediction failed');
      err.payload = result;
      throw err;
    }
    return result;
  }

  async function executeInference() {
    const saved = tabState[currentSubsystemId];
    if (!saved || !saved.files || saved.files.length === 0) return;

    const subsystemId = currentSubsystemId;
    const files = saved.files;
    btnRunInference.disabled = true;

    const entries = [];
    try {
      for (let i = 0; i < files.length; i++) {
        btnRunInference.innerHTML = `<span class="spinner"></span> Running ${i + 1}/${files.length}...`;
        try {
          const result = await predictOneFile(subsystemId, files[i].fileName, files[i].rawData);
          entries.push({ fileName: files[i].fileName, result, error: null });
        } catch (err) {
          entries.push({ fileName: files[i].fileName, result: null, error: err.payload || { error: err.message, error_type: 'PredictionError' } });
        }
      }

      const allFailed = entries.every(e => e.error);
      if (allFailed) {
        showDiagnosticModal({ subsystem: subsystemId, ...entries[0].error });
      } else {
        if (tabState[currentSubsystemId]) tabState[currentSubsystemId].results = entries;
        inferenceResult = entries;
        renderBatchResults(subsystemId, entries);
      }
    } catch (netErr) {
      console.error('Backend connection error:', netErr);
      showDiagnosticModal({
        subsystem: subsystemId,
        error_type: 'ConnectionError',
        error: 'Unable to reach the Python backend server at /api/predict. Please ensure python3 app/app.py is running.',
        hint: 'Start the backend using: python3 app/app.py',
        traceback: netErr.stack || netErr.toString()
      });
    } finally {
      btnRunInference.disabled = false;
      btnRunInference.innerHTML = '⚡ Run Prediction';
    }
  }

  const DIAGNOSTIC_MODULE_PATH = {
    acv: 'Optional_Items/ACV/code/src/pipeline.py',
    door: 'src/Door/predict.py (via app/subsystems.py)',
    shm: 'src/SHM/predict.py (via app/subsystems.py)',
    corrugation: 'src/Rail_Corrugation/predict.py (via app/subsystems.py)'
  };

  function showDiagnosticModal(errData) {
    if (!modalDiagnostic) return;
    if (diagnosticModule) diagnosticModule.textContent = DIAGNOSTIC_MODULE_PATH[errData.subsystem] || 'app/subsystems.py';
    if (diagnosticErrorMsg) diagnosticErrorMsg.textContent = `${errData.error_type || 'ModelError'}: ${errData.error || 'Execution failed'}`;
    if (diagnosticHint) diagnosticHint.textContent = errData.hint || `Ensure the input file matches the expected schema for this subsystem.`;
    if (diagnosticTraceback) diagnosticTraceback.textContent = errData.traceback || 'No traceback provided.';
    modalDiagnostic.classList.add('open');
  }

  // ============================================================================
  // 6. UI Rendering & Dashboard Population
  // ============================================================================
  // Shared helpers for all subsystem renderers
  function setVerdict(icon, label, title, summary, scoreLabel, scoreValue) {
    if (verdictIcon) verdictIcon.textContent = icon;
    if (verdictLabel) verdictLabel.textContent = label;
    const verdictTitle = document.getElementById('verdict-title');
    const verdictSummary = document.getElementById('verdict-summary');
    if (verdictTitle) verdictTitle.textContent = title;
    if (verdictSummary) verdictSummary.textContent = summary;
    if (scoreTitle) scoreTitle.textContent = scoreLabel;
    const scoreVal = document.getElementById('score-val');
    if (scoreVal) scoreVal.textContent = scoreValue;
  }

  function buildExportTable(headers, rows) {
    if (!exportTableHead || !exportTableBody) return;
    exportTableHead.innerHTML = '<tr>' + headers.map(h => `<th>${escapeHtml(h)}</th>`).join('') + '</tr>';
    exportTableBody.innerHTML = rows.map(row =>
      '<tr>' + row.map(cell => `<td>${cell}</td>`).join('') + '</tr>'
    ).join('');
  }

  function statusBadge(isBad, text) {
    return `<span class="status-badge ${isBad ? 'badge-red' : 'badge-green'}">${isBad ? '⚠️ ' : '✓ '}${escapeHtml(text)}</span>`;
  }

  function revealResults() {
    resultsContainer.style.display = 'block';
    resultsContainer.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  // Combines each successful entry's already-correctly-shaped csv_data into one
  // multi-row CSV: header taken from the first entry, then just the data lines
  // from every entry (skipping their individual headers).
  function combineCsv(okEntries) {
    if (okEntries.length === 0) return '';
    const first = okEntries[0].result.csv_data.trim().split(/\r?\n/);
    const header = first[0];
    const lines = [header];
    okEntries.forEach(e => {
      const rows = e.result.csv_data.trim().split(/\r?\n/).slice(1);
      lines.push(...rows);
    });
    return lines.join('\n');
  }

  function failedFilesNote(failed) {
    if (failed.length === 0) return '';
    return `<p style="margin-top:10px; font-size:0.8rem; color:var(--pastel-rose-text);">⚠️ ${failed.length} file(s) failed: ${failed.map(e => escapeHtml(e.fileName)).join(', ')}</p>`;
  }

  function renderBatchResults(subsystemId, entries) {
    if (!entries || entries.length === 0) return;
    const ok = entries.filter(e => !e.error);
    const failed = entries.filter(e => e.error);

    if (subsystemId === 'acv') renderAcvBatch(ok, failed);
    else if (subsystemId === 'door') renderDoorBatch(ok, failed);
    else if (subsystemId === 'shm') renderShmBatch(ok, failed);
    else if (subsystemId === 'corrugation') renderCorrugationBatch(ok, failed);
  }

  function renderAcvBatch(ok, failed) {
    if (ok.length === 0) return;
    const n = ok.length;
    const single = n === 1 ? ok[0].result : null;

    setVerdict('❄️', 'ACV REFRIGERANT LEAK LOCALISATION',
      single ? `Suspected Faulty Car: Car ${single.faulty_car}` : `${n} files processed`,
      single ? `Full Ranking: ${single.ranking_string}` : `Faulty car per file: ${ok.map(e => `${e.fileName} → Car ${e.result.faulty_car}`).join('; ')}`,
      'Most Likely Faulty Car', single ? `Car ${single.faulty_car}` : `${n} files`);

    buildExportTable(['#', 'file_id', 'ranked_cars'],
      ok.map((e, i) => [i + 1, escapeHtml(e.result.file_id), `<span class="mono-ranking">${escapeHtml(e.result.ranking_string)}</span>`]));

    if (single && Array.isArray(single.scores)) {
      if (detailTablePanel) detailTablePanel.style.display = 'block';
      if (detailTableTitle) detailTableTitle.textContent = 'Car Fault Ranking Details';
      if (detailTableSub) detailTableSub.textContent = 'Fault scores based on mean cooling error (°C)';
      if (scoresTableHead) scoresTableHead.innerHTML = '<tr><th>Rank</th><th>Car ID</th><th>Fault Score (Mean Cooling Error)</th><th>Status</th></tr>';
      const scoresTableBody = document.getElementById('scores-table-body');
      if (scoresTableBody) {
        scoresTableBody.innerHTML = single.scores.map(item => {
          const isTopFault = (String(item.car_id) === String(single.faulty_car) || item.rank === 1);
          const scoreFormatted = (typeof item.fault_score === 'number') ? item.fault_score.toFixed(4) : item.fault_score;
          return `<tr style="${isTopFault ? 'background:#fef2f2;' : ''}">
            <td><strong>#${item.rank}</strong></td>
            <td><strong>Car ${item.car_id}</strong></td>
            <td>${scoreFormatted} °C</td>
            <td>${statusBadge(isTopFault, isTopFault ? 'Suspected Leak' : 'Normal')}</td>
          </tr>`;
        }).join('');
      }
    } else if (detailTablePanel) {
      detailTablePanel.style.display = 'none';
    }
    finishBatchRender(ok, failed);
  }

  function renderDoorBatch(ok, failed) {
    if (ok.length === 0) return;
    const allSegments = [];
    ok.forEach(e => e.result.segments.forEach(s => allSegments.push({ ...s, sourceFile: e.fileName })));
    const totalAbnormal = allSegments.filter(s => s.prediction === 'Abnormal resistance').length;
    const abnormalPct = allSegments.length ? ((totalAbnormal / allSegments.length) * 100).toFixed(1) : '0.0';
    const isHealthy = totalAbnormal === 0;
    const multiFile = ok.length > 1;

    setVerdict(isHealthy ? '✅' : '🚪', 'DOOR CYCLE SEGMENTATION & CLASSIFICATION',
      isHealthy ? 'All Door Cycles Normal' : `${totalAbnormal} Abnormal-Resistance Cycle(s) Detected`,
      `${multiFile ? `Across ${ok.length} files: found` : 'Found'} ${allSegments.length} door cycle(s) (segmented by timestamp gaps); ${totalAbnormal} classified Abnormal-resistance.`,
      'Abnormal Rate', `${abnormalPct}%`);

    buildExportTable(
      ['#', 'start_time', 'end_time', 'prediction'],
      allSegments.map((s, i) => [i + 1, escapeHtml(s.start_time), escapeHtml(s.end_time), statusBadge(s.prediction === 'Abnormal resistance', s.prediction)])
    );

    if (detailTablePanel) detailTablePanel.style.display = 'block';
    if (detailTableTitle) detailTableTitle.textContent = 'Per-Cycle Diagnostic Detail';
    if (detailTableSub) detailTableSub.textContent = 'Operation type, mean motor current, and model probability per detected cycle';
    if (scoresTableHead) scoresTableHead.innerHTML = `<tr><th>#</th>${multiFile ? '<th>Source File</th>' : ''}<th>Operation</th><th>Mean Current (mA)</th><th>P(Abnormal)</th><th>Prediction</th></tr>`;

    const scoresTableBody = document.getElementById('scores-table-body');
    if (scoresTableBody) {
      scoresTableBody.innerHTML = allSegments.map((s, i) => `
        <tr style="${s.prediction === 'Abnormal resistance' ? 'background:#fef2f2;' : ''}">
          <td>#${i + 1}</td>
          ${multiFile ? `<td>${escapeHtml(s.sourceFile)}</td>` : ''}
          <td>${escapeHtml(s.operation)}</td>
          <td>${s.current_mean_mA}</td>
          <td>${(s.probability_abnormal * 100).toFixed(1)}%</td>
          <td>${statusBadge(s.prediction === 'Abnormal resistance', s.prediction)}</td>
        </tr>`).join('');
    }
    finishBatchRender(ok, failed);
  }

  function renderShmBatch(ok, failed) {
    if (ok.length === 0) return;
    const n = ok.length;
    const damages = ok.map(e => e.result.predicted_damage);
    const maxD = Math.max(...damages);
    const maxEntry = ok[damages.indexOf(maxD)];
    const avgD = damages.reduce((a, b) => a + b, 0) / n;
    const isCritical = maxD >= 1.0;
    const isWarning = maxD >= 0.5 && maxD < 1.0;

    setVerdict(isCritical ? '🛑' : (isWarning ? '⚠️' : '✅'), 'SHM CUMULATIVE FATIGUE DAMAGE ESTIMATE',
      n === 1 ? `Predicted Cumulative Damage D = ${damages[0].toFixed(6)}` : `${n} files processed — highest damage: ${maxEntry.fileName} (D=${maxD.toFixed(6)})`,
      `Estimated via rainflow counting + Miner's rule (exponent m=${ok[0].result.exponent_m}). Safe design limit: D < 1.0.${n > 1 ? ` Average across files: D=${avgD.toFixed(6)}.` : ''}`,
      "Miner's Index (D)", n === 1 ? damages[0].toFixed(4) : maxD.toFixed(4));

    buildExportTable(['#', 'file_id', 'prediction'], ok.map((e, i) => [i + 1, escapeHtml(e.result.file_id), e.result.predicted_damage.toFixed(6)]));

    if (detailTablePanel) detailTablePanel.style.display = 'none';
    finishBatchRender(ok, failed);
  }

  function renderCorrugationBatch(ok, failed) {
    if (ok.length === 0) return;
    const n = ok.length;
    const single = n === 1 ? ok[0].result : null;
    const counts = { Normal: 0, 'Side I': 0, 'Side II': 0 };
    ok.forEach(e => { counts[e.result.prediction] = (counts[e.result.prediction] || 0) + 1; });
    const isDefect = single ? single.prediction !== 'Normal' : (counts['Side I'] + counts['Side II']) > 0;

    setVerdict(isDefect ? '🛤️' : '✅', 'RAIL CORRUGATION CLASSIFICATION',
      single ? (isDefect ? `Track Anomaly: ${single.prediction}` : 'Normal Track Condition') : `${n} files: ${counts.Normal} Normal, ${counts['Side I']} Side I, ${counts['Side II']} Side II`,
      `Classified from 129-channel axle-box vibration/shock recording (RandomForest, LOO macro F1 = 0.7048).`,
      'Predicted Class', single ? single.prediction : `${n} files`);

    buildExportTable(['#', 'file_id', 'prediction'], ok.map((e, i) => [i + 1, escapeHtml(e.result.file_id), statusBadge(e.result.prediction !== 'Normal', e.result.prediction)]));

    if (single && single.probabilities) {
      if (detailTablePanel) detailTablePanel.style.display = 'block';
      if (detailTableTitle) detailTableTitle.textContent = 'Class Probability Breakdown';
      if (detailTableSub) detailTableSub.textContent = 'RandomForest predicted probability per class';
      if (scoresTableHead) scoresTableHead.innerHTML = '<tr><th>Class</th><th>Probability</th><th>Status</th></tr>';
      const scoresTableBody = document.getElementById('scores-table-body');
      if (scoresTableBody) {
        scoresTableBody.innerHTML = Object.entries(single.probabilities)
          .sort((a, b) => b[1] - a[1])
          .map(([cls, prob]) => `
            <tr style="${cls === single.prediction ? 'background:#fef2f2;' : ''}">
              <td><strong>${escapeHtml(cls)}</strong></td>
              <td>${(prob * 100).toFixed(1)}%</td>
              <td>${statusBadge(cls === single.prediction, cls === single.prediction ? 'Predicted' : '')}</td>
            </tr>`).join('');
      }
    } else if (detailTablePanel) {
      detailTablePanel.style.display = 'none';
    }
    finishBatchRender(ok, failed);
  }

  function finishBatchRender(ok, failed) {
    const failedNote = document.getElementById('failed-files-note');
    if (failedNote) failedNote.innerHTML = failedFilesNote(failed);
    revealResults();
  }

  // ============================================================================
  // 7. Zero-Dependency HTML5 Canvas Chart Engine
  // ============================================================================
  function renderCanvasChart(canvasId, config) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.parentElement.getBoundingClientRect();
    const width = rect.width || 500;
    const height = 280;

    canvas.width = width * dpr;
    canvas.height = height * dpr;
    canvas.style.width = width + 'px';
    canvas.style.height = height + 'px';
    ctx.scale(dpr, dpr);

    // Padding
    const padLeft = 55;
    const padRight = (config.type === 'dual-line') ? 50 : 25;
    const padTop = 30;
    const padBottom = 40;
    const plotW = width - padLeft - padRight;
    const plotH = height - padTop - padBottom;

    // Clear background (Clean modern white canvas)
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, width, height);

    if (config.type === 'bar') {
      renderBarChart(ctx, config, padLeft, padTop, plotW, plotH, width, height);
    } else {
      renderLineChart(ctx, config, padLeft, padTop, plotW, plotH, width, height);
    }
  }

  function renderLineChart(ctx, config, padLeft, padTop, plotW, plotH, width, height) {
    const isDual = config.type === 'dual-line' && config.datasets.length === 2;

    // Compute min and max
    let minY = Infinity, maxY = -Infinity;
    let minY2 = Infinity, maxY2 = -Infinity;

    config.datasets[0].data.forEach(v => {
      if (v < minY) minY = v;
      if (v > maxY) maxY = v;
    });

    if (isDual) {
      config.datasets[1].data.forEach(v => {
        if (v < minY2) minY2 = v;
        if (v > maxY2) maxY2 = v;
      });
    } else {
      config.datasets.forEach(ds => {
        ds.data.forEach(v => {
          if (v < minY) minY = v;
          if (v > maxY) maxY = v;
        });
      });
    }

    if (minY === maxY) { minY -= 1; maxY += 1; }
    const spanY = maxY - minY;
    minY -= spanY * 0.05;
    maxY += spanY * 0.08;

    if (isDual) {
      if (minY2 === maxY2) { minY2 -= 1; maxY2 += 1; }
      const span2 = maxY2 - minY2;
      minY2 -= span2 * 0.05;
      maxY2 += span2 * 0.08;
    }

    // Gridlines & Axis
    ctx.strokeStyle = '#f1f5f9';
    ctx.lineWidth = 1;
    ctx.fillStyle = '#64748b';
    ctx.font = '11px -apple-system, BlinkMacSystemFont, sans-serif';
    ctx.textAlign = 'right';

    const gridLines = 4;
    for (let g = 0; g <= gridLines; g++) {
      const yVal = minY + ((maxY - minY) * (g / gridLines));
      const py = padTop + plotH - (plotH * (g / gridLines));

      ctx.beginPath();
      ctx.moveTo(padLeft, py);
      ctx.lineTo(padLeft + plotW, py);
      ctx.stroke();

      ctx.fillText(yVal.toFixed(yVal > 10 ? 0 : 1), padLeft - 8, py + 4);

      if (isDual) {
        const yVal2 = minY2 + ((maxY2 - minY2) * (g / gridLines));
        ctx.textAlign = 'left';
        ctx.fillText(yVal2.toFixed(1), padLeft + plotW + 8, py + 4);
        ctx.textAlign = 'right';
      }
    }

    // Shaded anomaly zones if specified (Calm pastel rose band)
    if (config.anomalyBands && config.anomalyBands.length > 0) {
      config.anomalyBands.forEach(band => {
        const startX = padLeft + (plotW * 0.50); // Highlight zone
        const bandW = plotW * 0.22;
        ctx.fillStyle = 'rgba(254, 205, 211, 0.45)';
        ctx.fillRect(startX, padTop, bandW, plotH);

        ctx.strokeStyle = '#fca5a5';
        ctx.strokeRect(startX, padTop, bandW, plotH);

        ctx.fillStyle = '#e11d48';
        ctx.font = '600 11px -apple-system, BlinkMacSystemFont, sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText(band.label, startX + bandW / 2, padTop + 16);
      });
    }

    // Draw datasets
    config.datasets.forEach((ds, dIdx) => {
      const useY2 = isDual && dIdx === 1;
      const cMin = useY2 ? minY2 : minY;
      const cMax = useY2 ? maxY2 : maxY;
      const cSpan = cMax - cMin;

      ctx.strokeStyle = ds.color || '#60a5fa';
      ctx.lineWidth = ds.lineWidth || 2.2;
      ctx.beginPath();

      const n = ds.data.length;
      for (let i = 0; i < n; i++) {
        const px = padLeft + (plotW * (i / (n - 1)));
        const py = padTop + plotH - (plotH * ((ds.data[i] - cMin) / cSpan));
        if (i === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
      }
      ctx.stroke();
    });

    // Legend
    let legendX = padLeft;
    ctx.textAlign = 'left';
    ctx.font = '500 11px -apple-system, BlinkMacSystemFont, sans-serif';
    config.datasets.forEach(ds => {
      ctx.fillStyle = ds.color;
      ctx.fillRect(legendX, 10, 12, 4);
      ctx.fillStyle = '#475569';
      ctx.fillText(ds.label, legendX + 16, 15);
      legendX += ctx.measureText(ds.label).width + 35;
    });

    // X-axis label
    ctx.fillStyle = '#64748b';
    ctx.textAlign = 'center';
    ctx.font = '11px -apple-system, BlinkMacSystemFont, sans-serif';
    ctx.fillText(config.unitX || 'Time Progression', padLeft + plotW / 2, height - 12);
  }

  function renderBarChart(ctx, config, padLeft, padTop, plotW, plotH, width, height) {
    const ds = config.datasets[0];
    const data = ds.data;
    const n = data.length;
    let maxVal = Math.max(...data.map(v => parseFloat(v) || 0), config.thresholdLine || 0);
    maxVal = maxVal * 1.15 || 10;

    // Gridlines
    ctx.strokeStyle = '#f1f5f9';
    ctx.lineWidth = 1;
    ctx.fillStyle = '#64748b';
    ctx.font = '11px -apple-system, BlinkMacSystemFont, sans-serif';
    ctx.textAlign = 'right';

    for (let g = 0; g <= 4; g++) {
      const yVal = (maxVal * (g / 4));
      const py = padTop + plotH - (plotH * (g / 4));
      ctx.beginPath();
      ctx.moveTo(padLeft, py);
      ctx.lineTo(padLeft + plotW, py);
      ctx.stroke();
      ctx.fillText(yVal.toFixed(yVal > 10 ? 0 : 1), padLeft - 8, py + 4);
    }

    // Threshold Line
    if (config.thresholdLine !== undefined) {
      const thY = padTop + plotH - (plotH * (config.thresholdLine / maxVal));
      ctx.strokeStyle = '#f87171';
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(padLeft, thY);
      ctx.lineTo(padLeft + plotW, thY);
      ctx.stroke();
      ctx.setLineDash([]);

      ctx.fillStyle = '#e11d48';
      ctx.textAlign = 'left';
      ctx.font = '600 10px -apple-system, BlinkMacSystemFont, sans-serif';
      ctx.fillText(`Threshold (${config.thresholdLine})`, padLeft + 6, thY - 4);
    }

    // Bars
    const barWidth = Math.min(48, (plotW / n) * 0.65);
    const gap = plotW / n;

    data.forEach((val, i) => {
      const v = parseFloat(val) || 0;
      const bH = (v / maxVal) * plotH;
      const bx = padLeft + (gap * i) + (gap - barWidth) / 2;
      const by = padTop + plotH - bH;
      const col = (ds.colors && ds.colors[i]) ? ds.colors[i] : '#60a5fa';

      // Bar gradient (Pastel to clean white)
      const grad = ctx.createLinearGradient(bx, by, bx, padTop + plotH);
      grad.addColorStop(0, col);
      grad.addColorStop(1, '#ffffff');

      ctx.fillStyle = grad;
      ctx.fillRect(bx, by, barWidth, bH);

      ctx.strokeStyle = col;
      ctx.lineWidth = 1.5;
      ctx.strokeRect(bx, by, barWidth, bH);

      // Value label on top (Dark charcoal for clear readability)
      ctx.fillStyle = '#1e293b';
      ctx.textAlign = 'center';
      ctx.font = '600 10px -apple-system, BlinkMacSystemFont, monospace';
      ctx.fillText(v.toFixed(1), bx + barWidth / 2, by - 6);

      // X label
      ctx.fillStyle = '#64748b';
      ctx.font = '11px -apple-system, BlinkMacSystemFont, sans-serif';
      const xLab = config.xLabels[i] || '';
      ctx.fillText(xLab, bx + barWidth / 2, padTop + plotH + 20);
    });
  }

  // ============================================================================
  // 8. Prediction Export Handlers
  // ============================================================================
  function downloadBlob(content, mimeType, filename) {
    try {
      const blob = new Blob([content], { type: mimeType });
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.setAttribute('href', url);
      link.setAttribute('download', filename);
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      setTimeout(() => URL.revokeObjectURL(url), 100);
    } catch (e) {
      console.error('Download error:', e);
    }
  }

  function exportResultsCsv() {
    const entries = Array.isArray(inferenceResult) ? inferenceResult : null;
    if (!entries || entries.length === 0) {
      alert('No prediction results available to export.');
      return;
    }
    const ok = entries.filter(e => !e.error);
    if (ok.length === 0) {
      alert('No successful predictions available to export.');
      return;
    }
    const csvString = combineCsv(ok);
    downloadBlob(csvString, 'text/csv;charset=utf-8;', `${currentSubsystemId}_predictions.csv`);
  }

  function exportResultsJson() {
    const entries = Array.isArray(inferenceResult) ? inferenceResult : null;
    if (!entries || entries.length === 0) return;
    const jsonString = JSON.stringify(entries.map(e => e.error ? { fileName: e.fileName, error: e.error } : { fileName: e.fileName, ...e.result }), null, 2);
    downloadBlob(jsonString, 'application/json;charset=utf-8;', `${currentSubsystemId}_predictions.json`);
  }

  function filterTableRows(query) {
    const tableBody = document.getElementById('table-body');
    const rows = tableBody.getElementsByTagName('tr');
    for (let i = 0; i < rows.length; i++) {
      const text = rows[i].textContent.toLowerCase();
      rows[i].style.display = text.includes(query) ? '' : 'none';
    }
  }

  // Utilities
  function escapeHtml(str) {
    return String(str)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#039;');
  }

  function debounce(func, wait) {
    let timeout;
    return function (...args) {
      clearTimeout(timeout);
      timeout = setTimeout(() => func.apply(this, args), wait);
    };
  }

  // Start on DOM ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
