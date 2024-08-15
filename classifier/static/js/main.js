// Get elements
const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const captureButton = document.getElementById('capture');
const predictionElement = document.getElementById('prediction');
const confidenceElement = document.getElementById('confidence');
const capturedImageElement = document.getElementById('captured-image');

// Set up the video stream
navigator.mediaDevices.getUserMedia({ video: true })
    .then(stream => {
        video.srcObject = stream;
    })
    .catch(err => {
        console.error('Error accessing camera: ', err);
        alert('Error accessing camera: ' + err.message);
    });

// Capture an image from the video stream
captureButton.addEventListener('click', () => {
    const context = canvas.getContext('2d');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    context.drawImage(video, 0, 0, canvas.width, canvas.height);

    // Convert canvas image to a data URL
    const dataUrl = canvas.toDataURL('image/jpeg');
    sendImageToAPI(dataUrl);
});

// Send the captured image to the API
function sendImageToAPI(dataUrl) {
    axios.post('http://192.168.2.60:1000/api/image-classification/make_prediction/', { image: dataUrl })
        .then(response => {

            console.log('API Response:', response);

            const { prediction, confidence } = response.data;
            predictionElement.textContent = 'Prediction: ', prediction;
            confidenceElement.textContent = `Confidence: ${confidence.toFixed(2)}`;
            capturedImageElement.src = dataUrl;
            capturedImageElement.style.display = 'block';
        })
        .catch(error => {
            // console.error('Error sending image to API: ', error);
            predictionElement.textContent = 'Error processing the image.';
            confidenceElement.textContent = '';
            // capturedImageElement.style.display = 'none';
        });
}