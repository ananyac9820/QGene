// QGene JavaScript

// Tab switching
function switchTab(tabName) {
    // Hide all tabs
    document.querySelectorAll('.tab-content').forEach(tab => {
        tab.classList.remove('active');
    });
    
    // Remove active from all buttons
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    
    // Show selected tab
    document.getElementById(tabName + '-tab').classList.add('active');
    
    // Activate button
    event.target.classList.add('active');
}

// Single Prediction Form
document.getElementById('prediction-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    // Show loading state
    document.getElementById('predict-btn-text').style.display = 'none';
    document.getElementById('predict-spinner').style.display = 'inline-block';
    document.getElementById('results').style.display = 'none';
    
    // Collect form data
    const formData = {
        gene: document.getElementById('gene').value,
        position: document.getElementById('position').value,
        mutation_type: document.getElementById('mutation_type').value,
        aa_change: document.getElementById('aa_change').value,
        conservation_score: document.getElementById('conservation_score').value,
        allele_frequency: document.getElementById('allele_frequency').value
    };
    
    try {
        // Call API
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(formData)
        });
        
        const data = await response.json();
        
        if (data.success) {
            displayResults(data);
        } else {
            alert('Error: ' + data.error);
        }
    } catch (error) {
        alert('Network error: ' + error.message);
    } finally {
        // Hide loading state
        document.getElementById('predict-btn-text').style.display = 'inline';
        document.getElementById('predict-spinner').style.display = 'none';
    }
});

// Display single prediction results
function displayResults(data) {
    // Show results section
    document.getElementById('results').style.display = 'block';
    
    // Main prediction
    const badge = document.getElementById('prediction-badge');
    const predText = document.getElementById('prediction-text');
    predText.textContent = data.prediction;
    
    // Set badge color
    badge.className = 'prediction-badge ' + data.prediction.toLowerCase();
    
    // Confidence
    document.getElementById('confidence-value').textContent = data.confidence.toFixed(1) + '%';
    
    // Model scores
    document.getElementById('rf-score').textContent = data.probabilities.classical.random_forest.toFixed(1) + '%';
    document.getElementById('svm-score').textContent = data.probabilities.classical.svm.toFixed(1) + '%';
    document.getElementById('classical-avg').textContent = data.probabilities.classical.average.toFixed(1) + '%';
    
    document.getElementById('vqc-score').textContent = data.probabilities.quantum.vqc.toFixed(1) + '%';
    document.getElementById('qsvm-score').textContent = data.probabilities.quantum.qsvm.toFixed(1) + '%';
    document.getElementById('quantum-avg').textContent = data.probabilities.quantum.average.toFixed(1) + '%';
    
    document.getElementById('hybrid-score').textContent = data.probabilities.hybrid.toFixed(1) + '%';
    
    // Explanation
    displayExplanation(data.explanation);
    
    // Scroll to results
    document.getElementById('results').scrollIntoView({ behavior: 'smooth' });
}

// Display SHAP explanation
function displayExplanation(explanation) {
    const container = document.getElementById('explanation-list');
    container.innerHTML = '';
    
    explanation.forEach(item => {
        const div = document.createElement('div');
        div.className = 'explanation-item';
        
        const featureDiv = document.createElement('div');
        featureDiv.innerHTML = `
            <div class="explanation-feature">${item.feature}</div>
            <div style="font-size: 14px; color: #6b7280;">Value: ${item.value.toFixed(3)}</div>
        `;
        
        const contributionDiv = document.createElement('div');
        contributionDiv.className = 'explanation-contribution';
        
        const barWidth = Math.abs(item.contribution) * 200;
        const barClass = item.contribution > 0 ? 'contribution-positive' : 'contribution-negative';
        
        contributionDiv.innerHTML = `
            <div class="contribution-bar ${barClass}" style="width: ${barWidth}px;"></div>
            <span style="font-weight: 600; color: ${item.contribution > 0 ? '#ef4444' : '#10b981'};">
                ${item.contribution > 0 ? '+' : ''}${(item.contribution * 100).toFixed(1)}%
            </span>
        `;
        
        div.appendChild(featureDiv);
        div.appendChild(contributionDiv);
        container.appendChild(div);
    });
}

// File upload handling
const fileInput = document.getElementById('csv-file');
const fileName = document.getElementById('file-name');

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        fileName.textContent = e.target.files[0].name;
    }
});

// Drag and drop
const uploadArea = document.getElementById('file-upload-area');

uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.style.borderColor = 'var(--primary)';
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.style.borderColor = 'var(--border)';
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.style.borderColor = 'var(--border)';
    
    if (e.dataTransfer.files.length > 0) {
        fileInput.files = e.dataTransfer.files;
        fileName.textContent = e.dataTransfer.files[0].name;
    }
});

// Batch prediction form
document.getElementById('batch-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const formData = new FormData();
    formData.append('file', fileInput.files[0]);
    
    try {
        const response = await fetch('/batch-predict', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (data.success) {
            displayBatchResults(data);
        } else {
            alert('Error: ' + data.error);
        }
    } catch (error) {
        alert('Network error: ' + error.message);
    }
});

// Display batch results
function displayBatchResults(data) {
    const resultsDiv = document.getElementById('batch-results');
    resultsDiv.style.display = 'block';
    
    // Summary
    const pathogenicCount = data.results.filter(r => r.prediction === 'PATHOGENIC').length;
    const benignCount = data.results.filter(r => r.prediction === 'BENIGN').length;
    
    document.getElementById('batch-summary').innerHTML = `
        <strong>Total Processed:</strong> ${data.total_processed} mutations<br>
        <strong>Pathogenic:</strong> ${pathogenicCount} | 
        <strong>Benign:</strong> ${benignCount}
    `;
    
    // Table
    const table = document.getElementById('batch-table');
    let tableHTML = `
        <thead>
            <tr>
                <th>#</th>
                <th>Gene</th>
                <th>Position</th>
                <th>Prediction</th>
                <th>Confidence</th>
            </tr>
        </thead>
        <tbody>
    `;
    
    data.results.forEach(result => {
        const predClass = result.prediction === 'PATHOGENIC' ? 'color: #ef4444;' : 'color: #10b981;';
        tableHTML += `
            <tr>
                <td>${result.index + 1}</td>
                <td>${result.gene}</td>
                <td>${result.position}</td>
                <td style="${predClass} font-weight: 600;">${result.prediction}</td>
                <td>${result.confidence.toFixed(1)}%</td>
            </tr>
        `;
    });
    
    tableHTML += '</tbody>';
    table.innerHTML = tableHTML;
    
    // Scroll to results
    resultsDiv.scrollIntoView({ behavior: 'smooth' });
}

// Pre-fill example on page load (optional demo)
window.addEventListener('load', () => {
    // You can uncomment this to pre-fill an example
    /*
    document.getElementById('position').value = '5382';
    document.getElementById('mutation_type').value = 'Frameshift';
    document.getElementById('aa_change').value = 'C61G';
    document.getElementById('conservation_score').value = '0.95';
    document.getElementById('allele_frequency').value = '0.0012';
    */
});
