/* ==============================================================================
   📊 APPLICATION STATE & REAL-TIME ANALYTICS COUNTERS
   ============================================================================== */
let stats = { total: 0, bullying: 0, safe: 0 };

/**
 * Sends text payloads to the Flask API endpoint and rendering prediction responses
 */
async function analyze() {
    const text = document.getElementById('inputText').value;
    
    // Boundary check for empty text input strings
    if (!text.trim()) {
        alert('Please enter text to analyze');
        return;
    }
    
    const resultDiv = document.getElementById('result');
    resultDiv.style.display = 'block';
    resultDiv.innerHTML = 'Analyzing with AI...';
    
    try {
        // Dispatches asynchronous POST request targeting the local machine learning pipeline
        const response = await fetch('http://127.0.0.1:5000/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: text })
        });
        
        const data = await response.json();
        
        // Evaluate prediction status to verify presence of cyberbullying features
        const isBullying = data.prediction && data.prediction !== 'not_cyberbullying';
        
        // Dynamic operational update for the real-time analytics model state
        stats.total++;
        if (isBullying) {
            stats.bullying++;
        } else {
            stats.safe++;
        }
        
        // Renders updated computational tracking indicators back to the DOM nodes
        document.getElementById('totalCount').innerText = stats.total;
        document.getElementById('threatCount').innerText = stats.bullying;
        document.getElementById('safeCount').innerText = stats.safe;
        
        // Formats UI container design conditionally based on prediction output classes
        if (isBullying) {
            resultDiv.className = 'result result-bullying';
            resultDiv.innerHTML = '<h3>⚠️ Cyberbullying Detected!</h3><p><strong>Confidence:</strong> ' + data.confidence + '%</p><p><strong>🚨 Recommendation:</strong> Report this content immediately.</p>';
        } else {
            resultDiv.className = 'result result-safe';
            resultDiv.innerHTML = '<h3>✅ Safe Content</h3><p><strong>Confidence:</strong> ' + data.confidence + '%</p><p><strong>✅ Great!</strong> This content appears to be safe.</p>';
        }
    } catch (error) {
        // Exception handler to securely log connection errors or service downtime anomalies
        resultDiv.innerHTML = 'Error occurred: ' + error.message;
    }
}

/**
 * Resets workspace components and suppresses the output display interface
 */
function clearText() {
    document.getElementById('inputText').value = '';
    const resultDiv = document.getElementById('result');
    if (resultDiv) {
        resultDiv.style.display = 'none';
    }
}

const inputEl = document.getElementById('inputText');
if (inputEl) {
    // Listens for 'Ctrl + Enter' shortcut combinations to maximize processing efficiency
    inputEl.addEventListener('keydown', function(e) {
        if (e.ctrlKey && e.key === 'Enter') {
            analyze();
        }
    });
}