import cv2
import os
from keras.models import load_model
import numpy as np
from pygame import mixer
import time
import math
from keras.preprocessing import image
from keras.models import Sequential
from keras.layers import Dropout, Conv2D, Flatten, Dense, MaxPooling2D, BatchNormalization
from keras.layers import Input
from keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Function to train model if not already trained
def train_model():
    def generator(dir, gen=ImageDataGenerator(rescale=1./255), shuffle=True, batch_size=32, target_size=(24, 24), class_mode='categorical'):
        return gen.flow_from_directory(dir, batch_size=batch_size, shuffle=shuffle, color_mode='grayscale', class_mode=class_mode, target_size=target_size)

    BS = 32
    TS = (24, 24)

    train_batch = generator('./dataset_new/train', shuffle=True, batch_size=BS, target_size=TS)
    valid_batch = generator('dataset_new/test', shuffle=True, batch_size=BS, target_size=TS)

    SPE = math.ceil(len(train_batch.classes) / BS)
    VS = math.ceil(len(valid_batch.classes) / BS)
    print(SPE, VS)

    model = Sequential([
        Input(shape=(24, 24, 1)),
        Conv2D(32, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(1, 1)),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(1, 1)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(1, 1)),
        Dropout(0.25),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(4, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_batch, validation_data=valid_batch, epochs=15)
    model.save('./models/cnncat2.h5', overwrite=True)
    print('Model trained and saved successfully.')

# Check if model exists; if not, train the model
if not os.path.exists('./models/cnncat2.h5'):
    print("Model not found, starting training...")
    train_model()

# Load the model after training (or if it exists)
model = load_model('./models/cnncat2.h5')

# Initialize sound and cascade classifiers
mixer.init()
sound = mixer.Sound('alarm.wav')

face = cv2.CascadeClassifier('haar_cascade_files/haarcascade_frontalface_alt.xml')
leye = cv2.CascadeClassifier('haar_cascade_files/haarcascade_lefteye_2splits.xml')
reye = cv2.CascadeClassifier('haar_cascade_files/haarcascade_righteye_2splits.xml')

lbl = ['Close', 'Open']

# Video capture
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
count = 0
score = 0
thicc = 2
rpred = [99]
lpred = [99]

while(True):
    ret, frame = cap.read()
    height, width = frame.shape[:2]

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face.detectMultiScale(gray, minNeighbors=5, scaleFactor=1.1, minSize=(25, 25))
    left_eye = leye.detectMultiScale(gray)
    right_eye = reye.detectMultiScale(gray)

    cv2.rectangle(frame, (0, height - 50), (200, height), (0, 0, 0), thickness=cv2.FILLED)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 100, 100), 1)

    for (x, y, w, h) in right_eye:
        r_eye = frame[y:y + h, x:x + w]
        count += 1
        r_eye = cv2.cvtColor(r_eye, cv2.COLOR_BGR2GRAY)
        r_eye = cv2.resize(r_eye, (24, 24))
        r_eye = r_eye / 255
        r_eye = r_eye.reshape(24, 24, -1)
        r_eye = np.expand_dims(r_eye, axis=0)
        rpred = np.argmax(model.predict(r_eye), axis=1)

        if rpred[0] == 1:
            lbl = 'Open'
        if rpred[0] == 0:
            lbl = 'Closed'
        break

    for (x, y, w, h) in left_eye:
        l_eye = frame[y:y + h, x:x + w]
        count += 1
        l_eye = cv2.cvtColor(l_eye, cv2.COLOR_BGR2GRAY)
        l_eye = cv2.resize(l_eye, (24, 24))
        l_eye = l_eye / 255
        l_eye = l_eye.reshape(24, 24, -1)
        l_eye = np.expand_dims(l_eye, axis=0)
        lpred = np.argmax(model.predict(l_eye), axis=1)

        if lpred[0] == 1:
            lbl = 'Open'
        if lpred[0] == 0:
            lbl = 'Closed'
        break

    if rpred[0] == 0 and lpred[0] == 0:
        score += 1
        cv2.putText(frame, "Closed", (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
    else:
        score -= 1
        cv2.putText(frame, "Open", (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

    if score < 0:
        score = 0
    cv2.putText(frame, 'Score:' + str(score), (100, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

    if score > 15:
        cv2.imwrite(os.path.join(os.getcwd(), 'image.jpg'), frame)
        try:
            sound.play()
        except:
            pass

        if thicc < 16:
            thicc += 2
        else:
            thicc -= 2
            if thicc < 2:
                thicc = 2
        cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), thicc)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
