import os
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models
from tensorflow.keras
.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing import image 
from tensorflow.keras.models import load_model
import numpy as np


camera = cv2.VideoCapture(0)


face_detect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
emotion = load_model('C:\\Users\\user\\model_emotion.h5')
class_label = ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']
camera = cv2.VideoCapture(0)

while True:
    
    ret , frame = camera.read()
    label =[]
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_detect.detectMultiScale(gray,1.3,5)
 
    for (x,y,w,h) in faces :
        cv2.rectangle(frame,(x,y),(x+w,y+h),(200,100,230),2)
        roi_gray = gray[y:y+h,x:x+w]
        roi_gray = cv2.resize(gray,(48,48),interpolation=cv2.INTER_AREA)
        
   
        if np.sum([gray])!=0 :
            roi = roi_gray.astype('float')/255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi,axis = 0)
            
            pred = emotion.predict(roi)[0]
            label = class_label[pred.argmax()]
            label_position =(x,y)
            cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),2,cv2.LINE_AA)
        
        
    
        else:
            cv2.putText(frame,'No Face Found',(20,60),cv2.FONT_HERSHEY_SIMPLEX,2,(100,100,255),2)
    cv2.imshow("Emotion Detect",frame)
    
    if cv2.waitKey(3) &0xff==27:
        break
camera.release()
cv2.destroyAllWindows()