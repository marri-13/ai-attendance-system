import cv2
import numpy as np
import pandas as pd
from datetime import datetime

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")

label_map = np.load("labels.npy", allow_pickle=True).item()

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

attendance = []

cam = cv2.VideoCapture(0)

while True:
    ret, frame = cam.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face_img = gray[y:y+h, x:x+w]

        label, confidence = recognizer.predict(face_img)

        name = label_map[label]

        if name not in [a[0] for a in attendance]:
            time_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            attendance.append([name, time_now])

        cv2.putText(frame, name, (x,y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)

    cv2.imshow("Recognition", frame)

    if cv2.waitKey(1) == 27:
        break

df = pd.DataFrame(attendance, columns=["Name","Time"])
df.to_csv("attendance.csv", index=False)

cam.release()
cv2.destroyAllWindows()