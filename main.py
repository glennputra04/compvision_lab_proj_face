import sys
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
    # Skip gambar jika wajah tidak terdeteksi
    if len(faces) == 0:
        return None

    largest_face = max(faces, key=lambda rect: rect[2] * rect[3])

    x, y, w, h = largest_face
    face_region = gray[y:y+h, x:x+w]

    return face_region

# Train Model
def train_model(face_region_train):
    if len(face_region_train) == 0:
        print("No training faces available.")
        return False

    # Ubah nama person ke label numerik
    person_names = sorted(set(person for _, person in face_region_train))
    label_map = {
        person: label for label, person in enumerate(person_names)
    }

    faces = []
    labels = []

    for face_region, person in face_region_train:
        # Pastikan gambar dalam format grayscale
        if len(face_region.shape) == 3:
            face_region = cv2.cvtColor(
                face_region,
                cv2.COLOR_BGR2GRAY
            )
        # LBPH perlu gambar grayscale berukuran konsisten
        face = cv2.resize(face_region, (200, 200))
        faces.append(face)
        labels.append(label_map[person])

    recognizer.train(
        faces,
        np.array(labels, dtype=np.int32)
    )

    os.makedirs("model", exist_ok=True)

    # Simpan trained model
    recognizer.save(os.path.join("model", "nakama_model.yml"))

    # Simpan mapping label
    np.save(
        os.path.join("model", "nakama_labels.npy"),
        np.array(person_names)
    )

    print("Model training completed.")
    print(f"Total training faces: {len(faces)}")
    print(f"Total people recognized: {len(person_names)}")
    print("Model saved to model/nakama_model.yml")
    return True

def predict_test(face_region_test):
    if len(face_region_test) == 0:
        print("No testing faces available.")
        return False

    recognizer.read(os.path.join("model", "nakama_model.yml"))
    person_names = np.load(os.path.join("model", "nakama_labels.npy"), allow_pickle=True)

    correct_dictionary = {}
    total_dictionary = {}

    for _, person in face_region_test:
        correct_dictionary[person] = 0
        total_dictionary[person] = 0

    for face_region, actual_person in face_region_test:
        if len(face_region.shape) == 3:
            face_region = cv2.cvtColor(
                face_region,
                cv2.COLOR_BGR2GRAY
            )

        face = cv2.resize(face_region, (200, 200))

        label, _ = recognizer.predict(face)
        predicted_person = person_names[label]
        total_dictionary[actual_person] += 1

        if predicted_person == actual_person:
            correct_dictionary[actual_person] += 1

    total_test = sum(total_dictionary.values())
    total_acc = 0
    total_person = 0
    for person, count in correct_dictionary.items():
        average_person = count / total_dictionary[person] if total_dictionary[person] > 0 else 0
        total_acc += average_person
        total_person += 1

    macro_accuracy = total_acc / total_person if total_person > 0 else 0

    print(f"Test images evaluated: {total_test}")
    print(f"Average accuracy: {macro_accuracy:.2%}")
    return True

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

    if train_model(face_region_train):
        predict_test(face_region_test)
    else :
        print("Training failed. Cannot proceed to testing.")
def menu_two() :
    # TO DO: Implement menu_two functionality
    pass

def menu_three() :
    print("Nakama-tachi, sayonara.")
    sys.exit()
    
def main() :
    while True:
        print("")
        print("Menu:")
        print("1. Train and test model")
        print("2. Predict")
        print("3. Exit")

        choice = input("Input your choice: ")

        if choice == '1':
            menu_one()
        elif choice == '2':
            menu_two()
        elif choice == '3':
            menu_three()
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()