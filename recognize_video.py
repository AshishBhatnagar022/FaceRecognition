import imutils
import pickle
import cv2
import os
import numpy as np
import pandas as pd
from imutils import paths
from extract_embeddings import load_model
from keras.models import Model
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import pickle
import time
import cv2
import os
import pickle
img_size=224
epsilon=0.25

# data = pickle.loads(open('output/embeddings.pickle', "rb").read())
def findEuclideanDistance(source_representation, test_representation):
    euclidean_distance = source_representation - test_representation
    euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
    euclidean_distance = np.sqrt(euclidean_distance)
    return euclidean_distance
def findCosineDistance(source_representation, test_representation):
    a = np.matmul(np.transpose(source_representation), test_representation)
    b = np.sum(np.multiply(source_representation, source_representation))
    c = np.sum(np.multiply(test_representation, test_representation))
    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))
data=pickle.loads(open('output/embeddings.pickle','rb').read())
print('ss',len(data))
print(data['final_names'])
database=data['final_embed']


print("[INFO] loading face detector...")
protoPath = os.path.sep.join(['face_detection_model', "deploy.prototxt"])
modelPath = os.path.sep.join(['face_detection_model',"res10_300x300_ssd_iter_140000.caffemodel"])
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
print('Loading face recognizer...')

# Loading  vgg face model

model=load_model()
model.load_weights('weights/vgg_face_weights.h5')
vgg_face_descriptor = Model(inputs=model.layers[0].input,outputs=model.layers[-2].output)
# vs = VideoStream(src=0).start()

#for Webcam
vs = cv2.VideoCapture(0)

# time.sleep(2.0)
 
# start the FPS throughput estimator
fps = FPS().start()

while True:
	frame = vs.read()
	success,frame = vs.read()
	if (success !=True):
		print('empty frame')
		break

		# frame = imutils.resize(frame, width=600)

	else:

		frame = imutils.resize(frame, width=600)
 
	# resize the frame to have a width of 600 pixels (while
	# maintaining the aspect ratio), and then grab the image
	# dimensions
	# frame = imutils.resize(frame, width=600)
	(h, w) = frame.shape[:2]
 
	# construct a blob from the image
	imageBlob = cv2.dnn.blobFromImage(
		cv2.resize(frame, (300, 300)), 1.0, (300, 300),
		(104.0, 177.0, 123.0), swapRB=False, crop=False)
 
	# apply OpenCV's deep learning-based face detector to localize
	# faces in the input image
	detector.setInput(imageBlob)
	detections = detector.forward()


	count=0
	for i in range(0, detections.shape[2]):


		if len(detections) > 0:
		# i=np.argmax(detections[0,0,:,2])
			confidence= detections[0,0,i,2]


			if confidence>0.5:
				name='unknown'

				print('confidence',confidence)
				print('count',count)
				count+=1
				box=detections[0,0,i,3:7]*np.array([w,h,w,h])
				(startX,startY,endX,endY)=box.astype('int')
				face=frame[startY:endY,startX:endX]
				(fH,fW)=face.shape[:2]
				new_array=cv2.resize(face,(img_size,img_size))
			  
				we=new_array.reshape(-1,img_size,img_size,3)

						
						# ensure the face width and height are sufficiently large
				if fW < 20 or fH < 20:
					continue
				embedding=vgg_face_descriptor.predict(we)[0,:]
						# print('hdhdgggggggggg',type(embedding))
						# print('hdhdgggggggggg',(embedding.shape))


				for (name1, db_enc) in zip(data['final_names'],data['final_embed']):
					cosine_similarity = findCosineDistance(embedding, db_enc[0:])

					if(cosine_similarity < epsilon):
						print('cosine{} name {}'.format(cosine_similarity,name1))

						name=name1


							# pass
							# name=name1
				text = "{}".format(name)
				y = startY - 10 if startY - 10 > 10 else startY + 10
				cv2.rectangle(frame, (startX, startY), (endX, endY),
				(0, 0, 255), 2)

							
							
				cv2.putText(frame, text, (startX, y),
				cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
				# key = cv2.waitKey(1) & 0xFF
)
	
	fps.update()
		# show the output frame
	cv2.imshow("Frame", frame)
	# key = cv2.waitKey(1) & 0xFF ==ord('q')
	 
	if cv2.waitKey(1) & 0xFF ==ord('q'):
		print('q is pressed')
		break
	 
	
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
 
# do a bit of cleanup
vs.stop()

cv2.destroyAllWindows()
 

