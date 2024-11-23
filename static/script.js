document.getElementById('prediction-form').addEventListener('submit', function(e) {
    e.preventDefault();
    const content = document.getElementById('content').value;

    fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ content: content })
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById('result').innerHTML = `<div class="alert alert-info">Prediction: ${data.prediction} (Confidence: ${data.confidence * 100}%)</div>`;
    })
    .catch(error => {
        console.error('Error:', error);
        document.getElementById('result').innerHTML = `<div class="alert alert-danger">Error: ${error}</div>`;
    });
});
