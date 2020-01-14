
import numpy as np
import pandas as pd
import cv2
import imutils
import os
from imutils import paths
from extract_embeddings import load_model
from keras.models import Model
import pickle

# from keras.models import Sequential,Model
# from keras.layers import Dense,Convolution2D,ZeroPadding2D,MaxPooling2D,Dropout,Flatten,Activation



# def preprocess_image(image_path):
# 	img_size=224
#   	img_array=cv2.imread(image_path,cv2.IMREAD_COLOR)
# #   
#   	new_array=cv2.resize(img_array,(img_size,img_size))
  
#   	we=new_array.reshape(-1,img_size,img_size,3)
# #   print(we.shape)
  
  
#   	return we
img_size=224

protoPath=os.path.sep.join(['face_detection_model','deploy.prototxt'])
modelPath=os.path.sep.join(['face_detection_model','res10_300x300_ssd_iter_140000.caffemodel'])

detector=cv2.dnn.readNetFromCaffe(protoPath,modelPath)

print('Loading face recognizer...')
model=load_model()
model.load_weights('weights/vgg_face_weights.h5')
vgg_face_descriptor = Model(inputs=model.layers[0].input,outputs=model.layers[-2].output)
imagePaths=list(paths.list_images('dataset'))
cosine_similarity=0.40

embeddings=[]
names=[]


total=0
for (i,imagePath) in enumerate(imagePaths):
	# print("[INFO] processing image {}/{}".format(i + 1,
		# len(imagePaths)))
	name = imagePath.split(os.path.sep)[-2]
	print('processing_image {}/{}'.format(i+1,len(imagePaths)))
	image=cv2.imread(imagePath)
	image=imutils.resize(image,width=600)

	(h, w) = image.shape[:2]
	imageBlob=cv2.dnn.blobFromImage(cv2.resize(image,(300,300)),1.0,(300,300),(104.0, 177.0, 123.0), swapRB=False, crop=False)
	detector.setInput(imageBlob)
	detections=detector.forward()

	if len(detections) > 0:
		i=np.argmax(detections[0,0,:,2])
		confidence= detections[0,0,i,2]

		if confidence>0.5:
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
			embeddings.append(embedding)
			names.append(name)
			# 

print('Total Images',total)

data={'final_embed':embeddings,'final_names':names}
f=open('output/embeddings.pickle','wb')
f.write(pickle.dumps(data))
f.close()







