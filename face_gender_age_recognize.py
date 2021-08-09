# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import pickle
import imutils
import time
import math
import cv2
import os

class RecognizeFaceGenderAge(object):
	def __init__(self):
		# construct the argument parser and parse the arguments
		self.ap = argparse.ArgumentParser()
		self.ap.add_argument("-d", "--detector", required=True,
			help="path to OpenCV's deep learning face detector")
		self.ap.add_argument("-m", "--embedding-model", required=True,
			help="path to OpenCV's deep learning face embedding model")
		self.ap.add_argument("-r", "--recognizer", required=True,
			help="path to model trained to recognize faces")
		self.ap.add_argument("-l", "--le", required=True,
			help="path to label encoder")
		self.ap.add_argument("-c", "--confidence", type=float, default=0.5,
			help="minimum probability to filter weak detections")
		self.ap.add_argument('--input', help='Path to input image or video file.')
		self.ap.add_argument("--device", default="cpu", help="Device to inference on")
		self.args = vars(self.ap.parse_args())


		self.faceProto = "opencv_face_detector.pbtxt"
		self.faceModel = "opencv_face_detector_uint8.pb"

		self.ageProto = "age_deploy.prototxt"
		self.ageModel = "age_net.caffemodel"

		self.genderProto = "gender_deploy.prototxt"
		self.genderModel = "gender_net.caffemodel"

		self.MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
		self.ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
		self.genderList = ['Male', 'Female']


		# load our serialized face detector from disk
		print("[INFO] loading face detector...")
		self.protoPath = os.path.sep.join([self.args["detector"], "deploy.prototxt"])
		self.modelPath = os.path.sep.join([self.args["detector"],
			"res10_300x300_ssd_iter_140000.caffemodel"])
		self.detector = cv2.dnn.readNetFromCaffe(self.protoPath, self.modelPath)
		# load our serialized face embedding model from disk
		print("[INFO] loading face recognizer...")
		self.embedder = cv2.dnn.readNetFromTorch(self.args["embedding_model"])
		# load the actual face recognition model along with the label encoder
		self.recognizer = pickle.loads(open(self.args["recognizer"], "rb").read())
		self.le = pickle.loads(open(self.args["le"], "rb").read())

		# Load network
		self.ageNet = cv2.dnn.readNet(self.ageModel, self.ageProto)
		self.genderNet = cv2.dnn.readNet(self.genderModel, self.genderProto)
		self.faceNet = cv2.dnn.readNet(self.faceModel, self.faceProto)



	def setBackend(self):
		if self.args["device"] == "cpu":
			self.ageNet.setPreferableBackend(cv2.dnn.DNN_TARGET_CPU)

			self.genderNet.setPreferableBackend(cv2.dnn.DNN_TARGET_CPU)
			
			self.faceNet.setPreferableBackend(cv2.dnn.DNN_TARGET_CPU)

			print("Using CPU device")
		elif self.args["device"] == "gpu":
			self.ageNet.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
			self.ageNet.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

			self.genderNet.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
			self.genderNet.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

			self.faceNet.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
			self.faceNet.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
			print("Using GPU device")
	

	def videostream(self):
		# initialize the video stream, then allow the camera sensor to warm up
		print("[INFO] starting video stream...")
		vs = VideoStream(src=0).start()
		time.sleep(2.0)
		# start the FPS throughput estimator
		fps = FPS().start()
		while True:
			# grab the frame from the threaded video stream
			frame = vs.read()
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
			self.detector.setInput(imageBlob)
			detections = self.detector.forward()
			
			# loop over the detections
			for i in range(0, detections.shape[2]):
				# extract the confidence (i.e., probability) associated with
				# the prediction
				confidence = detections[0, 0, i, 2]
				# filter out weak detections
				if confidence > self.args["confidence"]:
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
					blob =  cv2.dnn.blobFromImage(face, 1.0, (227, 227),
						self.MODEL_MEAN_VALUES, swapRB=False)
					self.embedder.setInput(faceBlob)

					self.genderNet.setInput(blob)

					vec = self.embedder.forward()

					genderPreds = self.genderNet.forward()

					gender = self.genderList[genderPreds[0].argmax()]

					self.ageNet.setInput(blob)

					agePreds = self.ageNet.forward()

					age = self.ageList[agePreds[0].argmax()]

					label = "{}, age:{}".format(gender, age)
					print("Gender : {}, conf = {:.3f}".format(gender, genderPreds[0].max()))
					print("Age Output : {}".format(agePreds))
					print("Age : {}, conf = {:.3f}".format(age, agePreds[0].max()))
					# perform classification to recognize the face
					preds = self.recognizer.predict_proba(vec)[0]
					j = np.argmax(preds)
					proba = preds[j]
					name = self.le.classes_[j]
					# draw the bounding box of the face along with the
					# associated probability
					if proba > 0.6:
						text = "{}: {:.2f}%".format(name, proba * 100)
						y = startY - 10 if startY - 10 > 10 else startY + 10
						cv2.rectangle(frame, (startX, startY), (endX, endY),
							(0, 255, 0), 2)
						cv2.putText(frame, text, (startX, y),
							cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
					else:
						text = "Unknown Face Detected"
						y = startY - 10 if startY - 10 > 10 else startY + 10
						cv2.rectangle(frame, (startX, startY), (endX, endY),
							(0, 0, 255), 2)
						cv2.putText(frame, text, (startX, y),
							cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
					cv2.putText(frame, label, (startX, y-30),
					cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

			# update the FPS counter
			fps.update()
			# show the output frame
			cv2.imshow("Frame", frame)
			key = cv2.waitKey(1) & 0xFF
			# if the `q` key was pressed, break from the loop
			if key == ord("q"):
				break

		# stop the timer and display FPS information
		fps.stop()
		print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
		print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
		# do a bit of cleanup
		cv2.destroyAllWindows()
		vs.stop()

if __name__ == '__main__':
	try:
		r = RecognizeFaceGenderAge()
		r.setBackend()
		r.videostream()
	except cv2.error as e:
		print(e)













