import os 
import cv2
import pickle
import numpy as np
import sys
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from pathlib import Path

from jetson.face_embedder import FaceEmbedder
from jetson.config import DATABASE_FILE_PATH, PROCESSED_DATA_DIR, SIMILARITY_THRESHOLD, MODELS_DIR


def extract_features():
    print("Initialization of extraction of features")
    embedder = FaceEmbedder(model_path=os.path.join(MODELS_DIR, "mobilefacenet.onnx"))
    database = {}

    # Initialization of MediaPipe Tasks
    base_options = python.BaseOptions(model_asset_path=str(MODELS_DIR / 'blaze_face_short_range.tflite'))
    options = vision.FaceDetectorOptions(base_options=base_options, min_detection_confidence=SIMILARITY_THRESHOLD)
    detector = vision.FaceDetector.create_from_options(options)


    print(f"Searching directory: {PROCESSED_DATA_DIR}")

    for person_name in os.listdir(PROCESSED_DATA_DIR):
        person_dir = os.path.join(PROCESSED_DATA_DIR, person_name)

        # Checking if person_dir is directory
        if not os.path.isdir(person_dir):
            continue
        
        face_dir = os.path.join(person_dir, "Face")

        embedding_list = []

        # Going only through directory with faces
        for image_name in os.listdir(face_dir):
            img_path = os.path.join(face_dir, image_name)
            img = cv2.imread(img_path)

            if img is None:
                print(f"There was a problem with reading image {img}")

            # Face detection
            image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # MediaPipe tasks demands conversion to own format of image - mp.Image
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

            # Detection in new format
            detection_result = detector.detect(mp_image)


            if detection_result.detections:
                # We take first detected face
                bbox = detection_result.detections[0].bounding_box
                
                x, y, w, h = bbox.origin_x, bbox.origin_y, bbox.width, bbox.height
                ih, iw, _ = img.shape

                # Padding
                pad_x, pad_y = int(w * 0.1), int(h * 0.1)

                # To avoid situation when bounding box is over frame of an image
                x1 = max(0, x - pad_x)
                y1 = max(0, y - pad_y)
                x2 = min(iw, x + w + pad_x)
                y2 = min(ih, y + h + pad_y)

                # Cropping face
                face_crop = img[y:y+h, x:x+w]

                if face_crop.size > 0:
    
                    emb = embedder.get_embedding(img)
                    embedding_list.append(emb)
            else:
                print(f"mediapipe doesn't recognize any face on an image {image_name}")

        if embedding_list:

            # We calculate mean vector for each person
            mean_embedding = np.mean(embedding_list, axis=0)
            database[person_name] = mean_embedding / np.linalg.norm(mean_embedding)
            print(f"There is {len(embedding_list)} images of {person_name}")


    # Save
    os.makedirs(os.path.dirname(PROCESSED_DATA_DIR), exist_ok=True)
    with open(DATABASE_FILE_PATH, 'wb') as f:
        pickle.dump(database, f)


if __name__ == "__main__":
    extract_features()         

