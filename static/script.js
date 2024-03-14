document.getElementById('dataset-select').addEventListener('change', function() {
    const pdfUpload = document.getElementById('pdf-upload');
    if (this.value === 'pdf') {
        pdfUpload.classList.remove('hidden');
    } else {
        pdfUpload.classList.add('hidden');
    }
});

function processDataset() {
    const dataset = document.getElementById('dataset-select').value;
    if (dataset === 'pdf') {
        // Handle PDF upload or selection
        // This part requires backend integration to actually upload files and call the preprocess script
        alert('PDF processing is selected. Please integrate with backend to handle file uploads.');
    } else {
        // For Wikipedia and NYT, call the preprocess script with the selected dataset
        // This also requires backend integration
        alert(`${dataset} processing is selected. Please integrate with backend to trigger the process.`);
    }
}
