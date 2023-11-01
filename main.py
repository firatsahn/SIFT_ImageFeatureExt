from flask import Flask, render_template, Response, request
import cv2
import numpy as np
import tempfile
import time
import os
import logging

app = Flask(__name__)

logger = logging.getLogger(__name__)
logging.basicConfig(filename='error.log', level=logging.ERROR)

MIN_MATCH_COUNT = 30  #Minimum number of matches
ratio_thresh = 0.75

default_trainingImg_path = "img/2.jpeg"
trainingImg = cv2.imread(default_trainingImg_path, 0)   

sift = cv2.SIFT_create() #Initialize SIFT algorithm
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
    
                matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
                if descriptors_2 is not None and descriptors_1 is not None:
                    knn_matches = matcher.knnMatch(descriptors_1, descriptors_2, 2)

                    # matcher = cv2.BFMatcher() # Initialize a Brute-Force Matcher
                    # matches = matcher.knnMatch(descriptors_1, descriptors_2, k=2) #Find k-nearest matches

                    img_matches = np.empty((0, 0))
                    good_matches = []

                    # Filter and keep good matches
                    for i in range(len(knn_matches)):
                        if knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance:
                            good_matches.append(knn_matches[i][0])
                    
                    if len(good_matches) > MIN_MATCH_COUNT:
                        tp = []
                        qp = []
                        for k in range(len(good_matches)):
                            qp.append(keypoints_2[good_matches[k].trainIdx].pt)
                            tp.append(keypoints_1[good_matches[k].queryIdx].pt)

                        H, _ = cv2.findHomography(np.array(tp), np.array(qp), cv2.RANSAC) #perspective transformation matrix

                        s = trainingImg.shape
                        rows = s[0]
                        cols = s[1]

                        trainingBorder = np.array([[0, 0], [cols - 1, 0], [cols - 1, rows - 1], [0, rows - 1]], dtype=np.float32) #Border of the training image
                        QueryBorder = cv2.perspectiveTransform(trainingBorder.reshape(1, -1, 2), H).reshape(-1, 2) #Transform the training image border using the found perspective transformation
                        cv2.polylines(frame, [np.int32(QueryBorder)], True, (0, 255, 0), 4) # Draw a polygon
                        
                        # img_matches = cv2.drawMatches(trainingImg, keypoints_1, frame, keypoints_2,good_matches,None)
                        

                        if detect_frame == False:
                            img_matches = cv2.drawMatches(trainingImg, keypoints_1, Query, keypoints_2,
                                    good_matches, img_matches, (0, 0, 255), (0, 0, 255),
                                    [], cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                                                
                            yield (b'--frame\r\n'
                                b'Content-Type: image/jpeg\r\n\r\n' + cv2.imencode('.jpg', img_matches)[1].tobytes() + b'\r\n')

                if detect_frame == True:
                    yield (b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + cv2.imencode('.jpg', frame)[1].tobytes() + b'\r\n')

            else:
                pass
            


@app.route('/') #main/template endpoint
def index():
    return render_template('index.html')


@app.route('/upload_image', methods=['POST']) #image upload endpoint
def upload_image():
    global trainingImg,keypoints_1, descriptors_1 
    if 'image' in request.files:
        image = request.files['image']

        temp_file = tempfile.NamedTemporaryFile(delete=False)
        image.save(temp_file) #save file temporarily
        temp_file_path = temp_file.name
       
        trainingImg = cv2.imread(temp_file_path, 0)
        keypoints_1, descriptors_1 = sift.detectAndCompute(trainingImg, None) #detect and compute for the training img
        return ''

    else:
        trainingImg = cv2.imread(default_trainingImg_path, 0)
        keypoints_1, descriptors_1 = sift.detectAndCompute(trainingImg, None) #detect and compute for the training img
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
    # video_capture.set(3, 640) 
    # video_capture.set(4, 480)
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
    
    return ''

if __name__ == '__main__':
    app.run(host='127.0.0.1',port=4242,debug=True)
