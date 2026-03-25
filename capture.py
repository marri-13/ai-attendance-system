import cv2
import os

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)

user_id = input("Enter your ID: ")

count = 0

os.makedirs(f"dataset/{user_id}", exist_ok=True)

while True:
    ret, img = cam.read()

    if not ret:
        print("Camera error")
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.1, 3)

    print("Faces:", len(faces))

    for (x, y, w, h) in faces:
        count += 1
        face_img = gray[y:y+h, x:x+w]

        cv2.imwrite(f"dataset/{user_id}/{count}.jpg", face_img)

        cv2.rectangle(img, (x,y),(x+w,y+h),(255,0,0),2)

    cv2.imshow('Capturing Faces', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    if count >= 30:
        break

cam.release()
cv2.destroyAllWindows()

print("Done")