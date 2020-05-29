# USAGE
# python build_face_dataset.py --cascade haarcascade_frontalface_default.xml --output dataset/adrian

# import the necessary packages
from imutils.video import VideoStream
import argparse
import imutils
import time
import cv2
import os
import mysql.connector as mysqlc
from datetime import datetime

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--cascade", required=True,
	help = "path to where the haar cascade xml file resides")
ap.add_argument("-o", "--output", required=True,
	help="path to output directory")
args = vars(ap.parse_args())

# load OpenCV's Haar cascade for face detection from disk
detector = cv2.CascadeClassifier(args["cascade"])

# initialize the video stream, allow the camera sensor to warm up,
# and initialize the total number of example faces written to disk
# thus far
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
#vs = VideoStream(usePiCamera=True).start()
time.sleep(2.0)
total = 0
count = 0
maxval = 0

def currentDateAndTime():
        now = datetime.now()
        dt_string=now.strftime("%d/%m/%Y %H:%M:%S")
        return dt_string
def dbMethod(faceno):
        conn = mysqlc.connect(host="localhost",
                             user="root",
                             passwd="",
                             db="FinalProject")
        mycursor=conn.cursor()
        query="insert into faces values(%s, %s)"
        dateandtime=currentDateAndTime()
        vals=(faceno, dateandtime)
        mycursor.execute(query, vals)
        conn.commit()

# loop over the frames from the video stream
while True:
	# grab the frame from the threaded video stream, clone it, (just
	# in case we want to write it to disk), and then resize the frame
	# so we can apply face detection faster
	frame = vs.read()
	count=count+1
	orig = frame.copy()
	frame = imutils.resize(frame, width=400)

	grayScale=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# detect faces in the grayscale frame
	rects = detector.detectMultiScale(grayScale)


	# loop over the face detections and draw them on the frame
	for (x, y, w, h) in rects:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                text1 = cv2.putText(frame, "Number of faces detected: " + str(rects.shape[0]), (0, frame.shape[0] - 10), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 255, 0), 1)
                text2 = cv2.putText(frame, "Number of frames: " + str(count), (0, frame.shape[0] - 25), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 255, 0), 1)
                if(maxval<rects.shape[0]):
                        maxval=rects.shape[0]
                text3=cv2.putText(frame, "Maximum number of faces counted: " + str(maxval), (0, frame.shape[0] - 45), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 255, 0), 1)
                numoffaces=str(rects.shape[0])
                #if(count%100==0):
                 #       dbMethod(numoffaces)
                  #      count=0
                for i in range(0, rects.shape[0]):
                            cv2.putText(frame, "ID: " + str(i), (x, y), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 255, 0), 1)



	# show the output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
	# if the `k` key was pressed, write the *original* frame to disk
	# so we can later process it and use it for face recognition
	if key == ord("k"):
		p = os.path.sep.join([args["output"], "{}.png".format(
			str(total).zfill(5))])
		cv2.imwrite(p, orig)
		total += 1

	# if the `q` key was pressed, break from the loop
	elif key == ord("q"):
		break

# do a bit of cleanup
print("[INFO] {} face images stored".format(total))
print("[INFO] cleaning up...")
cv2.destroyAllWindows()
vs.stop()
