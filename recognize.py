import imutils
import pickle
import cv2
import os
import numpy as np
import pandas as pd
from imutils import paths
from extract_embeddings import load_model
from keras.models import Model
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
# print(database.shape)
for (name1, db_enc) in zip(data['final_names'],data['final_embed']):
	print('HI')
	print('naam',name1)

	

print("[INFO] loading face detector...")
protoPath = os.path.sep.join(['face_detection_model', "deploy.prototxt"])
modelPath = os.path.sep.join(['face_detection_model',"res10_300x300_ssd_iter_140000.caffemodel"])
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
print('Loading face recognizer...')
model=load_model()
model.load_weights('weights/vgg_face_weights.h5')
vgg_face_descriptor = Model(inputs=model.layers[0].input,outputs=model.layers[-2].output)
image = cv2.imread('rm.jpg')
image = imutils.resize(image, width=600)
(h, w) = image.shape[:2]
imageBlob=cv2.dnn.blobFromImage(cv2.resize(image,(300,300)),1.0,(300,300),(104.0, 177.0, 123.0), swapRB=False, crop=False)
detector.setInput(imageBlob)
detections=detector.forward()
count=0
for i in range(0, detections.shape[2]):


	# if len(detections) > 0:
		# i=np.argmax(detections[0,0,:,2])
	confidence= detections[0,0,i,2]


	if confidence>0.5:
		name='unknown'

		print('confidence',confidence)
		print('count',count)
		count+=1
		box=detections[0,0,i,3:7]*np.array([w,h,w,h])
		(startX,startY,endX,endY)=box.astype('int')
		face=image[startY:endY,startX:endX]
		(fH,fW)=face.shape[:2]
		new_array=cv2.resize(face,(img_size,img_size))
  
		we=new_array.reshape(-1,img_size,img_size,3)

			# ensure the face width and height are sufficiently large
		if fW < 20 or fH < 20:
			continue
		embedding=vgg_face_descriptor.predict(we)[0,:]
			


		for (name1, db_enc) in zip(data['final_names'],data['final_embed']):
			cosine_similarity = findCosineDistance(embedding, db_enc[0:])

			if(cosine_similarity < epsilon):
				print('cosine{} name {}'.format(cosine_similarity,name1))
				name=name1


			
		text = "{}".format(name)
		y = startY - 10 if startY - 10 > 10 else startY + 10
		cv2.rectangle(image, (startX, startY), (endX, endY),
		(0, 0, 255), 2)

				
				
		cv2.putText(image, text, (startX, y),
		cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)


cv2.imshow("Image", image)
cv2.imwrite('detection.jpg',image)
cv2.waitKey(0)			
