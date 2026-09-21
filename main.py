import cv2
import numpy as np
import os

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

train_data, test_data = split_dataset()