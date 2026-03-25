import streamlit as st
import cv2
import numpy as np
import os
from datetime import datetime
import pandas as pd

# ================= UI CONFIG =================
st.set_page_config(page_title="AI Attendance System", layout="wide")

# ================= CUSTOM STYLE =================
st.markdown("""
    <style>
    .main {
        background: linear-gradient(to right, #0f2027, #203a43, #2c5364);
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# ================= HEADER =================
st.title("🤖 AI Face Recognition Attendance")
st.markdown("### 🚀 Smart Attendance using Computer Vision")

st.warning("⚠️ Camera works only locally. Use image upload for demo.")

# ================= PATHS =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

trainer_path = os.path.join(BASE_DIR, "trainer.yml")
labels_path = os.path.join(BASE_DIR, "labels.npy")
attendance_path = os.path.join(BASE_DIR, "attendance.csv")

# ================= LOAD MODEL =================
try:
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(trainer_path)
    label_map = np.load(labels_path, allow_pickle=True).item()
except:
    st.error("❌ Model files missing or error loading.")
    st.stop()

# ================= FACE DETECTOR =================
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# ================= LOAD DATA =================
if os.path.exists(attendance_path):
    attendance_df = pd.read_csv(attendance_path)
else:
    attendance_df = pd.DataFrame(columns=["Name", "Time"])

# ================= FUNCTION =================
def mark_attendance(name):
    global attendance_df
    if name not in attendance_df["Name"].values:
        time_now = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
        new_entry = pd.DataFrame([[name, time_now]], columns=["Name", "Time"])
        attendance_df = pd.concat([attendance_df, new_entry], ignore_index=True)
        attendance_df.to_csv(attendance_path, index=False)

# ================= IMAGE UPLOAD =================
st.subheader("🖼️ Upload Image for Recognition")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        st.warning("⚠️ No face detected")
    else:
        for (x, y, w, h) in faces:
            face_img = gray[y:y+h, x:x+w]

            try:
                label, confidence = recognizer.predict(face_img)
                person_name = label_map.get(label, "Unknown")
            except:
                person_name = "Unknown"

            if person_name != "Unknown":
                mark_attendance(person_name)

            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(img, person_name, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        st.image(img, channels="BGR", caption="🔍 Recognition Result")

# ================= ATTENDANCE =================
st.subheader("📋 Attendance Records")

if not attendance_df.empty:
    st.dataframe(attendance_df)

    # Stats
    st.markdown("### 📊 Insights")
    st.metric("Total Present", len(attendance_df))

    st.download_button(
        "📥 Download CSV",
        attendance_df.to_csv(index=False),
        "attendance.csv"
    )
else:
    st.info("No attendance recorded yet.")

# ================= AI ASSISTANT =================
st.subheader("🤖 AI Assistant")

query = st.text_input("Ask anything about the system...")

if query:
    q = query.lower()

    if "attendance" in q:
        st.success("📊 Attendance is tracked using face recognition.")
    elif "how" in q:
        st.success("🧠 Uses OpenCV LBPH algorithm for recognition.")
    elif "accuracy" in q:
        st.success("🎯 Accuracy depends on training data quality.")
    elif "model" in q:
        st.success("🤖 Model is trained using captured face images.")
    else:
        st.success("🤖 Ask about attendance, model, or recognition!")
