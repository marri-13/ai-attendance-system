import cv2
import os
import numpy as np

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

        # ✅ FIX: Resize all images to same size
        img = cv2.resize(img, (200, 200))

        faces.append(img)
        labels.append(current_label)

    current_label += 1

faces = np.array(faces)
labels = np.array(labels)

# Train recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(faces, labels)

recognizer.save("trainer.yml")

np.save("labels.npy", label_map)

print("Training Completed")