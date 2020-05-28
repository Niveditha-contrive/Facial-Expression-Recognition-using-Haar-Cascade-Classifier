
import cv2
from model import FacialExpressionModel
import numpy as np
from statistics import mode

import cv2
from keras.models import load_model
import numpy as np
video_capture = cv2.VideoCapture(0)
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
video_capture.set(3, 480) #set width of the frame
video_capture.set(4, 640) #set height of the frame
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
age_list = ['(0, 2)', '(4, 6)', '(8, 12)', '(15, 20)', '(25, 32)', '(38, 43)', '(48, 53)', '(60, 100)']
gender_list = ['Male', 'Female']

def load_caffe_models():
    age_net = cv2.dnn.readNetFromCaffe('deploy_age.prototxt', 'age_net.caffemodel')
    gender_net = cv2.dnn.readNetFromCaffe('deploy_gender.prototxt', 'gender_net.caffemodel')
    return(age_net, gender_net)


rgb = cv2.VideoCapture(0)
facec = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
font = cv2.FONT_HERSHEY_SIMPLEX

import time

def __get_data__():
    """
    __get_data__: Gets data from the VideoCapture object and classifies them
    to a face or no face. 
    
    returns: tuple (faces in image, frame read, grayscale frame)
    """
    _, fr = rgb.read()
    gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
    faces = facec.detectMultiScale(gray, 1.3, 5)

    return faces, fr, gray
def start_app(cnn,age_net,gender_net):
    skip_frame = 10
    data = []
    flag = False
    ix = 0

    while True:
        ix += 1
        check, frame = video_capture.read()
        #frame = cv2.flip(frame, 1)
        # converted our Webcam feed to Grayscale.**< Most of the operations in OpenCV are done in grayscale>**
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facec.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
            # Get Face as Matrix and copy it
            face_img = frame[y:y + h, h:h + w].copy()
            blob = cv2.dnn.blobFromImage(face_img, 2, (245, 245), MODEL_MEAN_VALUES, swapRB=True)
            # Predict Gender
            gender_net.setInput(blob)
            gender_preds = gender_net.forward()
            gender = gender_list[gender_preds[0].argmax()]
            # Predict Age
            age_net.setInput(blob)
            age_preds = age_net.forward()
            age = age_list[age_preds[0].argmax()]
            overlay_text = "%s %s" % (gender, age)
            cv2.putText(frame, overlay_text, (10,200), font, 1, (18,255,255), 2, cv2.LINE_AA)
            #cv2.imshow('frame', frame)

            for (x, y, w, h) in faces:
                fc = gray[y:y+h, x:x+w]
                roi = cv2.resize(fc, (48, 48))
                pred = cnn.predict_emotion(roi[np.newaxis, :, :, np.newaxis])
                print(pred)
                cv2.putText(frame, pred, (x, y), font, 1, (255, 255, 0), 2)
                cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        if cv2.waitKey(1) == 27:
            break
        cv2.imshow('Filter', frame)

    cv2.destroyAllWindows()


if __name__ == '__main__':
    model = FacialExpressionModel("face_model.json", "face_model.h5")
    age_net, gender_net = load_caffe_models()  # load caffe models (age & gender)
    start_app(model,age_net,gender_net)

    import urllib.request
    webUrl = urllib.request.urlopen(": https://web-chat.global.assistant.watson.cloud.ibm.com/preview.html?region=eu-gb&integrationID=431485a4-3152-45a3-961c-cce99a6ead95&serviceInstanceID=a83ed73b-189b-4336-a098-65cf38969af6")
    print(" feedback:" + str(webUrl.getcode()))
