import cv2
import joblib
import numpy as np
import pickle
import sqlite3
from keras.models import load_model


class VideoCamera():
    def __init__(self):
        self.name = None
        self.model = cv2.face.LBPHFaceRecognizer_create()
        # self.model = cv2.face.FisherFaceRecognizer_create()
        # self.model.read("facemodel.pkl")
        self.model_2 = load_model('family_vgg.h5')
        self.name = joblib.load("names.pkl")
        self.name = {v:k for k,v in self.name.items()}
        print(self.name)

        haar_file = 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(haar_file)

        self.cap = cv2.VideoCapture(0)
        self.start_time = 0
        self.present_name = None

    def __del__(self):
        self.cap.release()

    def get_frame(self):
        face = False
        ret, frame = self.cap.read()
        frames = frame.copy()
        width, height, c = frame.shape
        gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)

        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(frames, (x, y), (x + w, y + h), (255, 255, 0), 2)
            face = True

        frame_flip = cv2.flip(frames, 1)
        ret, frames = cv2.imencode('.jpg', frame_flip)
        return frame, frames.tobytes(), face

    # def get_name(self):
    #     name=0
    #
    #     ret, frame = self.cap.read()
    #     frame = cv2.flip(frame, 1)
    #     width, height, c = frame.shape
    #     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #
    #     faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
    #     for (x,y,w,h) in faces:
    #
    #         face = gray[y:y + h, x:x + w]
    #         face_resize = cv2.resize(face, (width, height))
    #         prediction = self.model.predict(face_resize)
    #         cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 3)
    #         if prediction[1]<80:
    #             name = self.name[prediction[0]]
    #             self.present_name = name
    #             # print(name)
    #             cv2.putText(frame,'%s - %.0f' % (self.name[prediction[0]],prediction[1]),(x-10, y-10), cv2.FONT_HERSHEY_PLAIN,1,(0, 255, 0))
    #
    #     cv2.putText(frame, self.present_name, (10, 450), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2, cv2.LINE_AA)

    #
    # ret, frame = cv2.imencode('.jpg', frame)
    # return frame.tobytes(), name

    def face_detector(self, img, size=0.5):

        # Convert image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        if faces == ():
            return img, []

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
            roi = img[y:y + h, x:x + w]
            roi = cv2.resize(roi, (224, 224))
        return img, roi

    def get_name(self):

        ret, frame = self.cap.read()
        image, face = self.face_detector(frame)

        face = np.array(face)
        face = np.expand_dims(face, axis=0)
        final_name = None
        if not face.shape == (1, 0):
            result = self.model_2.predict(face)
            final_name=self.name[result.argmax()]
            cv2.putText(image, str(final_name), (100, 120), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 120, 150), 2)
        ret, frame = cv2.imencode('.jpg', frame)

        return frame.tobytes(), final_name
