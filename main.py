import cv2
import numpy as np
import os
import random

haar_path = os.path.join(
    "haarcascade",
    "haarcascade_frontalface_default.xml"
)

haar_cascade = cv2.CascadeClassifier(haar_path)

if haar_cascade.empty():
    print("Failed to load Haar Cascade")
else:
    print("Haar Cascade ready")

recognizer = cv2.face.LBPHFaceRecognizer_create()

def split_dataset():
    train_data = []
    test_data = []

    # set random seed supaya hasil suffle konsisten
    random.seed(42)
    
    # Ambil semua folder person di dalam dataset
    persons = sorted(os.listdir("dataset"))

    for person in persons:
        person_path = os.path.join("dataset", person)

        if not os.path.isdir(person_path):
            continue

        # Ambil semua gambar milik person
        images = []

        for filename in sorted(os.listdir(person_path)):
            if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                image_path = os.path.join(person_path, filename)
                images.append(image_path)

        # Randomize gambar untuk setiap person
        random.shuffle(images)

        # Hitung 80%
        split_index = int(len(images) * 0.8)

        # Array slicing (80% train, 20% test)
        train_images = images[:split_index]
        test_images = images[split_index:]

        # Simpan path + nama person
        for image in train_images:
            train_data.append((image, person))

        for image in test_images:
            test_data.append((image, person))

        print(f"{person}:")
        print(f"  Total : {len(images)}")
        print(f"  Train : {len(train_images)}")
        print(f"  Test  : {len(test_images)}")

    print("\nDATASET SPLIT")
    print(f"Total training data : {len(train_data)}")
    print(f"Total testing data  : {len(test_data)}")

    return train_data, test_data

def detect_face(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = haar_cascade.detectMultiScale(gray, scaleFactor=1.03, minNeighbors=4)

    if len(faces) == 0:
        return None

    largest_face = max(faces, key=lambda rect: rect[2] * rect[3])

    x, y, w, h = largest_face
    face_region = gray[y:y+h, x:x+w]

    return face_region

# Train Model
def train_model(face_region_train):
    #TODO: Implement the training logic for the face recognizer using the face_region_train data
    pass


def menu_one():
    train_data, test_data = split_dataset()

    face_region_train = []
    face_region_test = []
    failed_detections = []

    for image_path, person in train_data:
        face_region = detect_face(image_path)
        if face_region is not None:
            face_region_train.append((face_region, person))
        else:
            failed_detections.append((image_path, person))

    for image_path, person in test_data:
        face_region = detect_face(image_path)
        if face_region is not None:
            face_region_test.append((face_region, person))
        else:
            failed_detections.append((image_path, person))

    for image_path, person in failed_detections:
        print(f"Failed to detect face in {image_path} for person {person}")

    train_model(face_region_train)


menu_one()