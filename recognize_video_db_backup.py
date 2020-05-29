# USAGE
# python recognize_video.py --detector face_detection_model \
#   --embedding-model openface_nn4.small2.v1.t7 \
#   --recognizer output/recognizer.pickle \
#   --le output/le.pickle

# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import pickle
import time
import cv2
import os
import mysql.connector as mysqlc
from datetime import datetime
import matplotlib.pyplot as plt
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--detector", required=True,
    help="path to OpenCV's deep learning face detector")
ap.add_argument("-m", "--embedding-model", required=True,
    help="path to OpenCV's deep learning face embedding model")
ap.add_argument("-r1", "--recognizer1", required=True,
    help="path to model trained to recognize faces")
ap.add_argument("-r2", "--recognizer2", required=True,
    help="path to model trained to recognize expressions")
ap.add_argument("-l1", "--let1", required=True,
    help="path to label encoder for faces")
ap.add_argument("-l2", "--let2", required=True,
    help="path to label encoder for expressions")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
    help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# load our serialized face detector from disk
print("[INFO] Loading face detector...")
protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
modelPath = os.path.sep.join([args["detector"],
    "res10_300x300_ssd_iter_140000.caffemodel"])
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# load our serialized face embedding model from disk
print("[INFO] Loading face recognizer...")
embedder = cv2.dnn.readNetFromTorch(args["embedding_model"])

# load the actual face recognition model along with the label encoder
recognizer = pickle.loads(open(args["recognizer1"], "rb").read())
le = pickle.loads(open(args["let1"], "rb").read())
recognizer_exp=pickle.loads(open(args["recognizer2"], "rb").read())
le_exp = pickle.loads(open(args["let2"], "rb").read())

# initialize the video stream, then allow the camera sensor to warm up
print("[INFO] Starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

# start the FPS throughput estimator
fps = FPS().start()
count = 0
maxval=0
neutralcount=0
frowncount=0
happycount=0
totalcount=1
numoffaces=0
text3=""
def currentDateAndTime():
        now = datetime.now()
        dt_string=now.strftime("%d/%m/%Y %H:%M:%S")
        return dt_string
def dbMethod(faceno, happypercentage, frownpercentage, neutralpercentage):
        conn = mysqlc.connect(host="localhost",
                             user="root",
                             passwd="",
                             db="finalproject")
        mycursor=conn.cursor()
        query="insert into faces values(%s, %s, %s, %s, %s)"
        dateandtime=currentDateAndTime()
        vals=(faceno, dateandtime, happypercentage, frownpercentage, neutralpercentage)
        mycursor.execute(query, vals)
        conn.commit()
        
# loop over frames from the video file stream
while True:
    # grab the frame from the threaded video stream
    frame = vs.read()
    count=count+1
    # resize the frame to have a width of 600 pixels (while
    # maintaining the aspect ratio), and then grab the image
    # dimensions
    frame = imutils.resize(frame, width=600)
    (h, w) = frame.shape[:2]

    # construct a blob from the image
    imageBlob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)), 1.0, (300, 300),
        (104.0, 177.0, 123.0), swapRB=False, crop=False)

    # apply OpenCV's deep learning-based face detector to localize
    # faces in the input image
    detector.setInput(imageBlob)
    detections = detector.forward()

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the prediction
        confidence = detections[0, 0, i, 2]
        # filter out weak detections
        if confidence > args["confidence"]:
            # compute the (x, y)-coordinates of the bounding box for
            # the face
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # extract the face ROI
            face = frame[startY:endY, startX:endX]
            (fH, fW) = face.shape[:2]

            # ensure the face width and height are sufficiently large
            if fW < 20 or fH < 20:
                continue

            # construct a blob for the face ROI, then pass the blob
            # through our face embedding model to obtain the 128-d
            # quantification of the face
            faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
                (96, 96), (0, 0, 0), swapRB=True, crop=False)
            embedder.setInput(faceBlob)
            vec = embedder.forward()

            # perform classification to recognize the face
            preds = recognizer.predict_proba(vec)[0]
            j = np.argmax(preds)
            proba = preds[j]
            name = le.classes_[j]

            #perform classificiation to recognize the expression    
            preds_exp = recognizer_exp.predict_proba(vec)[0]
            j_exp = np.argmax(preds_exp)
            proba_exp = preds_exp[j_exp]
            name_exp = le_exp.classes_[j_exp]

            if(name_exp=="neutral"):
                                neutralcount=neutralcount+1
            elif(name_exp=="frowning"):
                                frowncount=frowncount+1
            elif(name_exp=="happy"):
                                happycount=happycount+1
                                
            if(maxval<i+1):
                                maxval=i+1
                
            if proba*100>25.0:        
                    # draw the bounding box of the face along with the
                    # associated probability
                    text = "{} {} {:.2f}%".format(name, name_exp, proba * 100)
                    y = startY - 10 if startY - 10 > 10 else startY + 10
                    cv2.rectangle(frame, (startX, startY), (endX, endY),
                        (0, 255, 0), 2)
                    cv2.putText(frame, text+" ID: "+str(i+1), (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
                    #text1 = cv2.putText(frame, "Number of faces: "+str(i+1), (0, frame.shape[0] - 10), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 255, 0), 1)
                    numoffaces=i+1
                    text2 = cv2.putText(frame, "Frame count: "+str(count), (frame.shape[0], frame.shape[0] - 400), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 255, 0), 1)
                    text3 = cv2.putText(frame, "Maximum number of faces counted: "+str(maxval), (0, frame.shape[0]-25), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 255, 0), 1)
                    text4 = cv2.putText(frame, "Neutral Count: "+str(neutralcount)+" Frown Count: "+str(frowncount)+" Happy Count: "+str(happycount), (0, frame.shape[0]-40), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 255, 0), 1)
            else:
                    # draw the bounding box of the face along with the
                    # associated probability
                    text = "Unknown "+name_exp
                    y = startY - 10 if startY - 10 > 10 else startY + 10
                    cv2.rectangle(frame, (startX, startY), (endX, endY),
                        (0, 255, 0), 2)
                    cv2.putText(frame, text+" ID: "+str(i+1), (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
                    #text1 = cv2.putText(frame, "Number of faces: "+str(i+1), (0, frame.shape[0] - 10), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 255, 0), 1)
                    numoffaces=i+1
                    text2 = cv2.putText(frame, "Frame count: "+str(count), (frame.shape[0], frame.shape[0] - 400), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 255, 0), 1)
                    text3 = cv2.putText(frame, "Maximum number of faces counted: "+str(maxval), (0, frame.shape[0]-25), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 255, 0), 1)
                    text4 = cv2.putText(frame, "Neutral Count: "+str(neutralcount)+" Frown Count: "+str(frowncount)+" Happy Count: "+str(happycount), (0, frame.shape[0]-40), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 255, 0), 1)
            totalcount=neutralcount+happycount+frowncount
            neutralpercentage=(neutralcount/totalcount)*100
            happypercentage=(happycount/totalcount)*100
            frownpercentage=(frowncount/totalcount)*100
            if(count%100==0):
                    count=0
                    dbMethod(numoffaces, happypercentage, frownpercentage, neutralpercentage)


    # update the FPS counter
    fps.update()
    text1=cv2.putText(frame, "Number of faces: "+str(numoffaces), (0, frame.shape[0] - 10), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 255, 0), 1)
    # show the output frame
    cv2.imshow("frame", frame)
    #cv2.imshow("frame", text3)
    cv2.imshow("frame", text1)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# stop the timer and display FPS information
fps.stop()
print("[INFO] Elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] Approx. FPS: {:.2f}".format(fps.fps()))
print("[INFO] Happy percentage: "+str(happypercentage))
print("[INFO] Neutral percentage: "+str(neutralpercentage))
print("[INFO] Frown percentage: "+str(frownpercentage))
datetimestr=currentDateAndTime()
print("[INFO] Date and Time: "+datetimestr)

# Data to plot
labels = 'Happy', 'Neutral', 'Frowning'
sizes = [happypercentage, neutralpercentage, frownpercentage]
colors = ['gold', 'yellowgreen', 'lightskyblue']

# Plot
plt.pie(sizes, labels=labels, colors=colors,
autopct='%1.1f%%', shadow=False, startangle=140)

plt.axis('equal')
plt.show()

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
