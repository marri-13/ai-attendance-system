import streamlit as st
import cv2
import numpy as np
import os
from datetime import datetime
import pandas as pd

# ================= UI =================
st.set_page_config(page_title="AI Attendance System", layout="centered")

st.title("🤖 AI Face Recognition Attendance")
st.markdown("### 🚀 Smart Attendance using Computer Vision")

st.warning("⚠️ Camera works only on local machine. Upload image for demo here.")

# ================= FILE PATH FIX =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

trainer_path = os.path.join(BASE_DIR, "trainer.yml")
labels_path = os.path.join(BASE_DIR, "labels.npy")

# ================= LOAD MODEL =================
try:
    if not os.path.exists(trainer_path) or not os.path.exists(labels_path):
        st.error("❌ Model files not found. Please train model locally.")
        st.stop()

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(trainer_path)

    label_map = np.load(labels_path, allow_pickle=True).item()

except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# ================= FACE DETECTOR =================
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

attendance = []

# ================= INPUT =================
name = st.text_input("👤 Enter your name")

# ================= CAPTURE (LOCAL ONLY) =================
if st.button("📸 Capture Faces (Local Only)"):
    cam = cv2.VideoCapture(0)

    if not cam.isOpened():
        st.error("🚫 Camera not accessible. Run locally.")
    else:
        count = 0
        os.makedirs(f"dataset/{name}", exist_ok=True)

        st.info("📸 Capturing faces... Look at camera")

        while True:
            ret, img = cam.read()

            if not ret or img is None:
                st.error("🚫 Camera error")
                break

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                count += 1
                face_img = gray[y:y+h, x:x+w]
                cv2.imwrite(f"dataset/{name}/{count}.jpg", face_img)

                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

            cv2.imshow("Capturing Faces", img)

            if cv2.waitKey(1) == 27 or count >= 30:
                break

        cam.release()
        cv2.destroyAllWindows()
        st.success("✅ Faces captured successfully!")

# ================= IMAGE UPLOAD (WORKS ONLINE) =================
st.markdown("### 🖼️ Upload Image for Recognition")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face_img = gray[y:y+h, x:x+w]

        try:
            label, confidence = recognizer.predict(face_img)
            person_name = label_map.get(label, "Unknown")
        except:
            person_name = "Unknown"

        # Attendance logic
        if person_name not in [a[0] for a in attendance]:
            time_now = datetime.now().strftime("%H:%M:%S")
            attendance.append([person_name, time_now])

        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(img, person_name, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    st.image(img, channels="BGR", caption="🔍 Recognition Result")

# ================= SHOW ATTENDANCE =================
if attendance:
    df = pd.DataFrame(attendance, columns=["Name", "Time"])
    st.markdown("### 📋 Attendance Records")
    st.dataframe(df)

    st.download_button(
        label="📥 Download Attendance CSV",
        data=df.to_csv(index=False),
        file_name="attendance.csv",
        mime="text/csv"
    )
