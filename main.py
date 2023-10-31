from flask import Flask, render_template, Response, request
import cv2
import numpy as np
import tempfile
import time
import os

app = Flask(__name__)


sift = cv2.xfeatures2d.SIFT_create() #Initialize SIFT algorithm
MIN_MATCH_COUNT = 30  #Minimum number of matches

trainingImg_path = "img/2.jpeg"
trainingImg = cv2.imread(trainingImg_path, 0)

keypoints_1, descriptors_1 = sift.detectAndCompute(trainingImg, None) #detect and compute for the training img


#flags
frame = None
img_matches = None
streaming = False

def gen_frames(detect_frame): # Generate frames, detect keypoints, and compute descriptors. Return the result.
    global frame, img_matches
    while True:
        if streaming:
            ret, frame = video_capture.read()
            if frame is not None:
                Query = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                keypoints_2, descriptors_2 = sift.detectAndCompute(Query, None) #training for real-time frame
                #todo change matchers method
                if descriptors_2 is not None and descriptors_1 is not None:
                    matcher = cv2.BFMatcher() # Initialize a Brute-Force Matcher
                    matches = matcher.knnMatch(descriptors_1, descriptors_2, k=2) #Find k-nearest matches


                    good_matches = []
                    for m, n in matches:
                        if m.distance < 0.75 * n.distance: # Filter and keep good matches
                            good_matches.append(m)
                    
                    if len(good_matches) > MIN_MATCH_COUNT:

                        tp = np.float32([keypoints_1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2) #Transform points from the training image
                        qp = np.float32([keypoints_2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2) #Transform points from the real-time frame

                        H, _ = cv2.findHomography(tp, qp, cv2.RANSAC) #perspective transformation matrix

                        s = trainingImg.shape
                        rows, cols = s[0], s[1]

                        trainingBorder = np.array([[[0, 0], [cols - 1, 0], [cols - 1, rows - 1], [0, rows - 1]]], dtype='float32') #Border of the training image
                        QueryBorder = cv2.perspectiveTransform(trainingBorder, H) #Transform the training image border using the found perspective transformation
                        cv2.polylines(frame, [np.int32(QueryBorder)], True, (0, 255, 0), 4) # Draw a polygon
                    

                        img_matches = cv2.drawMatches(trainingImg, keypoints_1, frame, keypoints_2, good_matches, None)
                
                if detect_frame == False:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + cv2.imencode('.jpg', img_matches)[1].tobytes() + b'\r\n')
                else :
                    yield (b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + cv2.imencode('.jpg', frame)[1].tobytes() + b'\r\n')
            else:
                print("Break1")
                pass


@app.route('/') #main/template endpoint
def index():
    return render_template('index.html')


@app.route('/upload_image', methods=['POST']) #image upload endpoint
def upload_image():
    global trainingImg
    if 'image' in request.files:
        image = request.files['image']

        temp_file = tempfile.NamedTemporaryFile(delete=False)
        image.save(temp_file) #save file temporarily
        temp_file_path = temp_file.name
        
        trainingImg = cv2.imread(temp_file_path, 0)
        return ''

    else:
        trainingImg = cv2.imread(trainingImg_path, 0)
        return ''


@app.route('/video_feed') #for the image matching frame
def video_feed():
    if streaming:  
        return Response(gen_frames(detect_frame=False), mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        return ''
    

@app.route('/video_feed_detect')
def video_feed_detect(): #for the real time detection frame
    if streaming:
        return Response(gen_frames(detect_frame=True), mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        return ''

@app.route('/start_stream')
def start_stream():
    global video_capture
    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        return "Camera Error!", 500
    video_capture.set(3, 640) 
    video_capture.set(4, 480)
    time.sleep(0.1)  # Delay for 0.1 seconds
    global streaming
    streaming = True
    return "Success"
    

@app.route('/close')
def close():
    global streaming,trainingImg
    streaming = False
    try:
        video_capture.release()
    except : pass

    trainingImg = cv2.imread(trainingImg_path, 0)
    return ''

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=4242,debug=True)
