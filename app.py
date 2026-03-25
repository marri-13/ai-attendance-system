import streamlit as st
import cv2
import os
import numpy as np
import pandas as pd
from datetime import datetime

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Attendance System",
    page_icon="🤖",
    layout="centered"
)

# ---------------- CUSTOM UI ----------------
st.markdown("""
<style>
body {
    background-color: #0E1117;
}
.title {
    text-align: center;
    font-size: 40px;
    font-weight: bold;
    color: #00FFC6;
}
.subtitle {
    text-align: center;
    font-size: 18px;
    color: #AAAAAA;
}
.stButton>button {
    width: 100%;
    border-radius: 12px;
    height: 3em;
    font-size: 16px;
    background: linear-gradient(90deg, #00FFC6, #007CF0);
    color: black;
    font-weight: bold;
}
.block {
    padding: 15px;
    border-radius: 12px;
    background-color: #1A1D24;
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown('<div class="title">🤖 AI Smart Attendance</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">🚀 Face Recognition System | OpenCV + Streamlit</div>', unsafe_allow_html=True)

st.write("---")

# ---------------- USER INPUT ----------------
st.markdown("### 👤 User Setup")
user_name = st.text_input("Enter your name")

# ---------------- BUTTON LAYOUT ----------------
col1, col2 = st.columns(2)

# ---------------- CAPTURE ----------------
with col1:
    if st.button("📸 Capture Faces"):
        if user_name == "":
            st.warning("⚠️ Enter your name first!")
        else:
            cam = cv2.VideoCapture(0)
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )

            os.makedirs(f"dataset/{user_name}", exist_ok=True)
            count = 0

            FRAME_WINDOW = st.image([])
            st.info("📷 Capturing faces... Please look at camera")

            while count < 30:
                ret, img = cam.read()
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.1, 3)

                for (x, y, w, h) in faces:
                    count += 1
                    face_img = gray[y:y+h, x:x+w]
                    cv2.imwrite(f"dataset/{user_name}/{count}.jpg", face_img)

                    cv2.rectangle(img, (x,y),(x+w,y+h),(0,255,0),2)

                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                FRAME_WINDOW.image(img)

            cam.release()
            st.success("✅ Face data captured successfully!")

# ---------------- TRAIN ----------------
with col2:
    if st.button("🧠 Train Model"):
        with st.spinner("⚙️ Training AI model..."):
            dataset_path = "dataset"
            faces = []
            labels = []
            label_map = {}
            current_label = 0

            for person in os.listdir(dataset_path):
                label_map[current_label] = person
                person_path = os.path.join(dataset_path, person)

                for img_name in os.listdir(person_path):
                    img_path = os.path.join(person_path, img_name)
                    img = cv2.imread(img_path, 0)
                    img = cv2.resize(img, (200, 200))

                    faces.append(img)
                    labels.append(current_label)

                current_label += 1

            faces = np.array(faces)
            labels = np.array(labels)

            recognizer = cv2.face.LBPHFaceRecognizer_create()
            recognizer.train(faces, labels)

            recognizer.save("trainer.yml")
            np.save("labels.npy", label_map)

        st.success("🚀 Model trained successfully!")

st.write("---")

# ---------------- RECOGNITION ----------------
if st.button("🎥 Start Recognition"):
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("trainer.yml")

    label_map = np.load("labels.npy", allow_pickle=True).item()

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )

    cam = cv2.VideoCapture(0)
    attendance = []

    FRAME_WINDOW = st.image([])
    st.info("🔍 Detecting face...")

    while True:
        ret, frame = cam.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face_img = gray[y:y+h, x:x+w]
            label, _ = recognizer.predict(face_img)
            name = label_map[label]

            cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
            cv2.putText(frame, name, (x,y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

            if name not in [a[0] for a in attendance]:
                time_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                attendance.append([name, time_now])

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(frame)

        if len(attendance) > 0:
            break

    df = pd.DataFrame(attendance, columns=["Name", "Time"])
    df.to_csv("attendance.csv", index=False)

    cam.release()

    st.success(f"✅ Attendance marked for {attendance[0][0]}")

st.write("---")

# ---------------- ATTENDANCE ----------------
if st.button("📊 View Attendance"):
    if os.path.exists("attendance.csv"):
        df = pd.read_csv("attendance.csv")
        st.dataframe(df)
    else:
        st.warning("⚠️ No attendance data found")

st.write("---")

# ---------- AI ASSITANT -----------------

st.write("---")
st.markdown("### 🤖 AI Assistant")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
user_input = st.chat_input("Ask something about the system...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    # Simple AI responses (rule-based)
    if "attendance" in user_input.lower():
        response = "📊 The system records attendance using face recognition and stores it in a CSV file."

    elif "face" in user_input.lower():
        response = "🎥 Face recognition uses OpenCV to detect and identify faces based on trained data."

    elif "how" in user_input.lower():
        response = "⚙️ The system captures faces, trains a model, and recognizes users in real-time."

    elif "project" in user_input.lower():
        response = "🚀 This is an AI-powered attendance system using OpenCV and Streamlit."

    else:
        response = "🤖 I can help you understand the AI attendance system. Try asking about face recognition or attendance!"

    st.session_state.messages.append({"role": "assistant", "content": response})

    with st.chat_message("assistant"):
        st.markdown(response)

# ---------------- FOOTER ----------------
st.markdown(
    "<center>💡 Built with ❤️ using OpenCV + Streamlit | Hackathon Ready 🚀</center>",
    unsafe_allow_html=True
)