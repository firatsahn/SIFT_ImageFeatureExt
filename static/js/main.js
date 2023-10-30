var startButton = document.getElementById('startButton');
startButton.addEventListener('click', function() {
  fetch('/start_stream')
    .then(response => response.text())
    .then(data => {
      console.log(data);
      document.getElementById('cameraFeed').src = "{{ url_for('video_feed_detect') }}";
      document.getElementById('matchingImage').src = "{{ url_for('video_feed') }}";
      // matchingImage.style.display = 'inline-block';
      // cameraFeed.style.display = 'inline-block';
    });

    

});


var closeButton = document.getElementById('closeButton');
var matchingImage = document.getElementById('matchingImage');
var cameraFeed = document.getElementById('cameraFeed');

closeButton.addEventListener('click', function() {
  matchingImage.src = '';
  cameraFeed.src = '';

  fetch("/close")  
          .then(response => response.text())
          .then(message => {
              console.log(message);  
          });

  
});

var imageUpload = document.getElementById('imageUpload');
imageUpload.addEventListener('change', function() {
  var selectedImage = imageUpload.files[0]; 

  if (selectedImage) {
    var formData = new FormData();
    formData.append('image', selectedImage);

    fetch('/upload_image', {
      method: 'POST',
      body: formData
    })
    .then(response => response.text())
    .then(data => {
      console.log('Success', data);

    });
  }
});