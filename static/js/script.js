// ── Tab switching (app page only) ─────────────────────────────────
function switchTab(name, btn) {
  document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.nav-item[onclick]').forEach(b => b.classList.remove('active'));
  document.getElementById(name + '-tab').classList.add('active');
  if (btn) btn.classList.add('active');
  const titles = { single: 'Single Prediction', batch: 'Batch Upload' };
  const el = document.getElementById('topbar-title');
  if (el) el.textContent = titles[name] || '';
}

// ── Prediction form ───────────────────────────────────────────────
let lastResult = null;

const form = document.getElementById('prediction-form');
if (form) {
  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    const btnText = document.getElementById('predict-btn-text');
    const spinner = document.getElementById('predict-spinner');
    btnText.style.display = 'none';
    spinner.style.display = 'inline-block';
    document.getElementById('results').style.display = 'none';

    const payload = {
      gene:              document.getElementById('gene').value,
      position:          document.getElementById('position').value,
      mutation_type:     document.getElementById('mutation_type').value,
      mutation_length:   1,
      aa_change:         document.getElementById('aa_change').value,
      conservation_score: document.getElementById('conservation_score').value,
      allele_frequency:  document.getElementById('allele_frequency').value,
    };

    try {
      const res  = await fetch('/predict', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(payload) });
      const data = await res.json();
      if (data.success) { lastResult = data; displayResults(data); }
      else alert('Prediction error: ' + data.error);
    } catch (err) {
      alert('Network error: ' + err.message);
    } finally {
      btnText.style.display = 'inline';
      spinner.style.display = 'none';
    }
  });
}

function displayResults(data) {
  const resultsEl = document.getElementById('results');
  resultsEl.style.display = 'block';
  const isPath = data.prediction === 'PATHOGENIC';

  // Verdict
  const verdict = document.getElementById('result-verdict');
  verdict.className = 'result-verdict ' + data.prediction.toLowerCase();
  const predText = document.getElementById('prediction-text');
  predText.textContent = data.prediction;
  predText.className   = 'verdict-text ' + data.prediction.toLowerCase();
  document.getElementById('confidence-value').textContent = data.confidence.toFixed(1) + '%';

  // Gauge
  const arc = document.getElementById('gauge-arc');
  const pct = data.confidence / 100;
  arc.style.strokeDashoffset = 157 - pct * 157;
  arc.style.stroke = isPath ? '#c0392b' : '#1a7340';
  document.getElementById('gauge-label').textContent = data.confidence.toFixed(0) + '%';

  // Bars
  const p = data.probabilities;
  setBar('bar-rf',           'rf-score',       p.classical.random_forest);
  setBar('bar-svm',          'svm-score',      p.classical.svm);
  setBar('bar-classical-avg','classical-avg',  p.classical.average);
  setBar('bar-qsvm',         'qsvm-score',     p.quantum.qsvm);

  document.getElementById('hybrid-score').textContent = p.hybrid.toFixed(1) + '%';
  document.getElementById('w-classical').textContent  = Math.round(p.weights.classical * 100) + '%';
  document.getElementById('w-quantum').textContent    = Math.round(p.weights.quantum * 100) + '%';

  displaySHAP(data.explanation);
  resultsEl.scrollIntoView({ behavior:'smooth', block:'start' });
}

function setBar(barId, numId, value) {
  const bar = document.getElementById(barId);
  const num = document.getElementById(numId);
  if (bar) setTimeout(() => { bar.style.width = value + '%'; }, 60);
  if (num) num.textContent = value.toFixed(1) + '%';
}

function displaySHAP(explanation) {
  const container = document.getElementById('explanation-list');
  container.innerHTML = '';
  const maxAbs = Math.max(...explanation.map(e => Math.abs(e.contribution)), 0.001);
  explanation.forEach(item => {
    const isPos = item.contribution >= 0;
    const barW  = Math.round((Math.abs(item.contribution) / maxAbs) * 100);
    const div   = document.createElement('div');
    div.className = 'explanation-item';
    div.innerHTML = `
      <span class="explanation-feature">${item.feature}</span>
      <span class="explanation-value">${Number(item.value).toFixed(2)}</span>
      <div class="explanation-bar-wrap">
        <div class="explanation-bar ${isPos ? 'bar-pos' : 'bar-neg'}" style="width:${barW}%"></div>
      </div>
      <span class="explanation-contrib" style="color:${isPos ? '#c0392b' : '#1a7340'}">
        ${isPos ? '+' : ''}${(item.contribution * 100).toFixed(1)}%
      </span>`;
    container.appendChild(div);
  });
}

// ── PDF Report ────────────────────────────────────────────────────
function downloadReport() {
  if (!lastResult) return;
  const { jsPDF } = window.jspdf;
  const doc = new jsPDF({ unit:'mm', format:'a4' });
  const d   = lastResult;
  const isPath = d.prediction === 'PATHOGENIC';

  // Header bar
  doc.setFillColor(17, 34, 80);
  doc.rect(0, 0, 210, 28, 'F');
  doc.setTextColor(224, 197, 143);
  doc.setFontSize(20);
  doc.setFont('times', 'bold');
  doc.text('QGene', 14, 17);
  doc.setTextColor(180, 200, 230);
  doc.setFontSize(9);
  doc.setFont('helvetica', 'normal');
  doc.text('Quantum-Classical Hybrid Variant Analysis Report', 42, 13);
  doc.text('Generated: ' + new Date(d.timestamp).toLocaleString(), 42, 19);

  let y = 38;

  // Result box
  const boxColor = isPath ? [253, 240, 239] : [237, 247, 241];
  const textColor = isPath ? [192, 57, 43] : [26, 115, 64];
  doc.setFillColor(...boxColor);
  doc.setDrawColor(...textColor);
  doc.roundedRect(14, y, 182, 24, 3, 3, 'FD');
  doc.setTextColor(...textColor);
  doc.setFontSize(16);
  doc.setFont('times', 'bold');
  doc.text(d.prediction, 20, y + 10);
  doc.setFontSize(10);
  doc.setFont('helvetica', 'normal');
  doc.text(`Confidence: ${d.confidence.toFixed(1)}%`, 20, y + 18);
  doc.setTextColor(80, 80, 80);
  doc.setFontSize(9);
  doc.text(`Hybrid Ensemble (Classical ${Math.round(d.probabilities.weights.classical*100)}% + Quantum ${Math.round(d.probabilities.weights.quantum*100)}%)`, 100, y + 18);

  y += 32;

  // Input variant
  sectionHeader(doc, 'Input Variant', y);
  y += 8;
  const inp = d.input;
  const inpRows = [
    ['Gene', inp.gene || '—'],
    ['Position', inp.position || '—'],
    ['Mutation Type', inp.mutation_type || '—'],
    ['Amino Acid Change', inp.aa_change || '—'],
    ['Conservation Score', inp.conservation_score || '—'],
    ['Allele Frequency', inp.allele_frequency || '—'],
  ];
  inpRows.forEach(([label, val], i) => {
    if (i % 2 === 0) { doc.setFillColor(247, 245, 242); doc.rect(14, y-1, 182, 7, 'F'); }
    doc.setTextColor(100, 100, 120);
    doc.setFontSize(8.5);
    doc.setFont('helvetica', 'normal');
    doc.text(label, 18, y + 4);
    doc.setTextColor(30, 30, 50);
    doc.setFont('helvetica', 'bold');
    doc.text(String(val), 80, y + 4);
    y += 7;
  });

  y += 6;

  // Model breakdown
  sectionHeader(doc, 'Model Breakdown', y);
  y += 8;
  const p = d.probabilities;
  const models = [
    ['Random Forest', `${p.classical.random_forest.toFixed(1)}%`, 'Classical'],
    ['SVM', `${p.classical.svm.toFixed(1)}%`, 'Classical'],
    ['QSVM (Quantum Kernel)', `${p.quantum.qsvm.toFixed(1)}%`, 'Quantum'],
    ['Hybrid Ensemble', `${p.hybrid.toFixed(1)}%`, 'Hybrid'],
  ];
  models.forEach(([name, val, type], i) => {
    if (i % 2 === 0) { doc.setFillColor(247, 245, 242); doc.rect(14, y-1, 182, 7, 'F'); }
    const typeColors = { Classical:[59,111,212], Quantum:[124,58,237], Hybrid:[196,149,42] };
    doc.setFillColor(...typeColors[type]);
    doc.roundedRect(14, y+1, 18, 4, 1, 1, 'F');
    doc.setTextColor(255,255,255);
    doc.setFontSize(6.5);
    doc.text(type, 23, y + 4, { align:'center' });
    doc.setTextColor(30,30,50);
    doc.setFontSize(8.5);
    doc.setFont('helvetica', 'normal');
    doc.text(name, 36, y + 4);
    doc.setFont('helvetica', 'bold');
    doc.text(val, 160, y + 4);
    // Mini bar
    const barW = parseFloat(val) / 100 * 60;
    doc.setFillColor(230,230,235);
    doc.rect(90, y+1, 60, 4, 'F');
    doc.setFillColor(...typeColors[type]);
    doc.rect(90, y+1, barW, 4, 'F');
    y += 7;
  });

  y += 6;

  // SHAP
  sectionHeader(doc, 'Feature Importance (SHAP)', y);
  y += 8;
  d.explanation.forEach((item, i) => {
    if (i % 2 === 0) { doc.setFillColor(247,245,242); doc.rect(14, y-1, 182, 7, 'F'); }
    doc.setTextColor(100,100,120);
    doc.setFontSize(8.5);
    doc.setFont('helvetica','normal');
    doc.text(item.feature, 18, y + 4);
    doc.text(`Value: ${Number(item.value).toFixed(3)}`, 80, y + 4);
    const contrib = item.contribution * 100;
    const isPos = contrib >= 0;
    doc.setTextColor(isPos ? 192 : 26, isPos ? 57 : 115, isPos ? 43 : 64);
    doc.setFont('helvetica','bold');
    doc.text(`${isPos?'+':''}${contrib.toFixed(1)}%  ${item.impact}`, 120, y + 4);
    y += 7;
  });

  y += 6;

  // What this means
  sectionHeader(doc, 'Interpretation', y);
  y += 8;
  doc.setFillColor(...(isPath ? [253,240,239] : [237,247,241]));
  doc.rect(14, y, 182, 28, 'F');
  doc.setTextColor(50, 50, 70);
  doc.setFontSize(8.5);
  doc.setFont('helvetica','normal');
  const interp = isPath
    ? 'A PATHOGENIC classification indicates this variant is predicted to disrupt BRCA gene function,\npotentially increasing hereditary cancer risk. This result is for research purposes only.\nPlease consult a certified genetic counselor for clinical interpretation.'
    : 'A BENIGN classification indicates this variant is not predicted to significantly disrupt BRCA gene\nfunction. This does not rule out cancer risk from other factors. Routine screening is\nrecommended. Consult a physician for personalised advice.';
  doc.text(interp, 18, y + 7);

  y += 36;

  // Footer
  doc.setFillColor(240,237,232);
  doc.rect(0, 280, 210, 17, 'F');
  doc.setTextColor(130,130,140);
  doc.setFontSize(7.5);
  doc.text('QGene v1.0  ·  Quantum-Classical Hybrid ML  ·  Built on ClinVar data  ·  Qiskit 2.3', 14, 287);
  doc.setTextColor(192,100,60);
  doc.text('⚠  For research purposes only. Not for clinical diagnosis. Not a substitute for professional medical advice.', 14, 293);

  doc.save(`QGene_Report_${d.input.gene || 'BRCA'}_${Date.now()}.pdf`);
}

function sectionHeader(doc, title, y) {
  doc.setFillColor(17,34,80);
  doc.rect(14, y, 182, 6, 'F');
  doc.setTextColor(224, 197, 143);
  doc.setFontSize(8.5);
  doc.setFont('helvetica','bold');
  doc.text(title.toUpperCase(), 18, y + 4.2);
}

// ── File upload ───────────────────────────────────────────────────
const fileInput  = document.getElementById('csv-file');
const fileNameEl = document.getElementById('file-name');
const uploadArea = document.getElementById('file-upload-area');

if (fileInput) {
  fileInput.addEventListener('change', e => {
    if (e.target.files.length) fileNameEl.textContent = e.target.files[0].name;
  });
}
if (uploadArea) {
  uploadArea.addEventListener('dragover', e => { e.preventDefault(); uploadArea.style.borderColor = 'var(--sapphire)'; });
  uploadArea.addEventListener('dragleave', () => { uploadArea.style.borderColor = ''; });
  uploadArea.addEventListener('drop', e => {
    e.preventDefault(); uploadArea.style.borderColor = '';
    if (e.dataTransfer.files.length) { fileInput.files = e.dataTransfer.files; fileNameEl.textContent = e.dataTransfer.files[0].name; }
  });
}

const batchForm = document.getElementById('batch-form');
if (batchForm) {
  batchForm.addEventListener('submit', async e => {
    e.preventDefault();
    if (!fileInput.files.length) { alert('Please select a CSV file.'); return; }
    const fd = new FormData(); fd.append('file', fileInput.files[0]);
    try {
      const res  = await fetch('/batch-predict', { method:'POST', body: fd });
      const data = await res.json();
      if (data.success) displayBatchResults(data);
      else alert('Batch error: ' + data.error);
    } catch(err) { alert('Network error: ' + err.message); }
  });
}

function displayBatchResults(data) {
  const div = document.getElementById('batch-results');
  div.style.display = 'block';
  const nP = data.results.filter(r => r.prediction === 'PATHOGENIC').length;
  document.getElementById('batch-summary').innerHTML = `
    <span><strong style="color:var(--text-primary)">${data.total_processed}</strong> mutations processed</span>
    <span><strong style="color:var(--pathogenic)">${nP}</strong> Pathogenic</span>
    <span><strong style="color:var(--benign)">${data.total_processed - nP}</strong> Benign</span>`;
  const table = document.getElementById('batch-table');
  let html = `<thead><tr><th>#</th><th>Gene</th><th>Position</th><th>Prediction</th><th>Confidence</th></tr></thead><tbody>`;
  data.results.forEach(r => {
    const cls = r.prediction === 'PATHOGENIC' ? 'pred-pathogenic' : 'pred-benign';
    html += `<tr><td>${r.index+1}</td><td>${r.gene}</td><td style="font-family:var(--font-mono)">${r.position}</td><td class="${cls}">${r.prediction}</td><td style="font-family:var(--font-mono)">${r.confidence.toFixed(1)}%</td></tr>`;
  });
  html += '</tbody>';
  table.innerHTML = html;
  div.scrollIntoView({ behavior:'smooth' });
}
