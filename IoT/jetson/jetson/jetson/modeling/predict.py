import cv2
import pickle
import numpy as np
import time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import os

from jetson.config import DATABASE_FILE_PATH, FACE_MODEL_PATH, SIMILARITY_THRESHOLD, INTERIM_DATA_DIR, MODELS_DIR
from jetson.face_embedder import FaceEmbedder


def load_database():
    """Loads vector database"""
    try:
        with open(DATABASE_FILE_PATH, 'rb') as f:
            return pickle.load(f)
        
    except FileNotFoundError:
        print(f"error, there is no {DATABASE_FILE_PATH}")
        print("Run as first script features.py!")
        exit()


def recognize_face(emedding, database):
    """It compares vector with base and returns name and assuredness"""
    best_match_name = "unknown"
    highest_similarity = -1.0

    for name, db_embedding in database.items():
        # Cosinus distance 
        similarity = np.dot(emedding, db_embedding)

        if similarity > highest_similarity:
            highest_similarity = similarity
            best_match_name = name

    if highest_similarity >= SIMILARITY_THRESHOLD:
        return best_match_name, highest_similarity
    else:
        return "unknown", highest_similarity
    

def run_live_recognition():
    print("Loading database...")
    database = load_database()

    print("Initialization of features extraction..")
    embedder = FaceEmbedder(model_path=str(FACE_MODEL_PATH))

    print("Initalization of detection face..")
    base_options = python.BaseOptions(model_asset_path=str(MODELS_DIR / 'blaze_face_short_range.tflite'))
    options = vision.FaceDetectorOptions(
        base_options=base_options, 
        running_mode=vision.RunningMode.VIDEO,
        min_detection_confidence=SIMILARITY_THRESHOLD
    )
    detector = vision.FaceDetector.create_from_options(options)

    print("Run camera")
    ########## Version with test sample offline ##################
    #sample_path = os.path.join(INTERIM_DATA_DIR, "test_face_recognition.mp4")
    #cap = cv2.VideoCapture(sample_path)
    ##################################################

    ####### Version for live camera ###################
    cap = cv2.VideoCapture(0)
    ##################################################

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("There was a problem with capturing frame from camera")
            break

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Mediapipe deamnds own format of an image - mp.Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

        # Counting timestamp because mode Video need it
        timestamp_ms = int(time.time() * 1000)

        # Detection of face
        results = detector.detect_for_video(mp_image, timestamp_ms)

        if results.detections:
            for detection in results.detections:
                bbox = detection.bounding_box

                x, y, w, h = bbox.origin_x, bbox.origin_y, bbox.width, bbox.height
                ih, iw, _ = image.shape
                
                # Padding
                pad_x, pad_y = int(w * 0.1), int(h * 0.1)

                x1 = max(0, x - pad_x)
                y1 = max(0, y - pad_y)
                x2 = min(iw, x + w + pad_x)
                y2 = min(ih, y + h + pad_y)

                face_roi = image[y:y+h, x:x+w]

                if face_roi.size > 0:
                    # Calculating embedding of cropeed face
                    embedding = embedder.get_embedding(face_roi)

                    # Compare vector with base
                    name, confidence = recognize_face(embedding, database)

                    # Draw frame with name 
                    color = (0, 255, 0) if name != "unknown" else (0, 0, 255)
                    cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)
                    text = f"{name} ({confidence:.2f})"
                    cv2.putText(image, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.imshow('Face recognition - jetson nano', image)

        # Press 'ESC" to quit
        if cv2.waitKey(300) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_live_recognition()

