// Get elements
const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const captureButton = document.getElementById('capture');
const predictionElement = document.getElementById('prediction');
const confidenceElement = document.getElementById('confidence');
const loaderElement = document.getElementById('loader'); // Loader element

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
    
    // Show loader
    loaderElement.style.display = 'block';

    // Send the captured image to the API
    sendImageToAPI(dataUrl);
});

// Send the captured image to the API
function sendImageToAPI(dataUrl) {
    axios.post('http://192.168.0.164:1000/api/image-classification/make_prediction/', { image: dataUrl })
        .then(response => {
            // Hide loader
            loaderElement.style.display = 'none';

            console.log('API Response:', response);

            const { prediction, confidence } = response.data;
            predictionElement.textContent = `Prediction:  ${prediction}`;
            confidenceElement.textContent = `Confidence: ${confidence.toFixed(2)}`;
        })
        .catch(error => {
            // Hide loader
            loaderElement.style.display = 'none';

            console.error('Error sending image to API: ', error);
            predictionElement.textContent = 'Error processing the image.';
            confidenceElement.textContent = '';
        });
}

document.addEventListener("DOMContentLoaded", function() {
    const header = document.querySelector("h1");
    header.addEventListener("animationend", function() {
        header.classList.add("finished-typing");
    });
});

 // Collapsible navbar logic
 const coll = document.getElementsByClassName("collapsible");
 for (let i = 0; i < coll.length; i++) {
     coll[i].addEventListener("click", function() {
         this.classList.toggle("active");
         const content = this.nextElementSibling;
         if (content.style.display === "block") {
             content.style.display = "none";
         } else {
             content.style.display = "block";
         }
     });
 }

 const navLinks = document.querySelectorAll('.collapsible-navbar a');

navLinks.forEach(link => {
    link.addEventListener('click', () => {
        navLinks.forEach(nav => nav.classList.remove('active'));
        link.classList.add('active');
    });
});
