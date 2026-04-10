import os 
import requests
from pathlib import Path

from jetson.config import MODELS_DIR

def download_file(url, destination):
    print(f"Downloading model from : {url}..")
    response = requests.get(url, stream=True)
    if response.status_code == 200: # What does it mean status code == 200? Full success
        with open(destination, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192): # Why 8192? Its a limitation for downloading 8 KB
                f.write(chunk)
        print(f"Download complete! File save in: {destination}")

    else:
        print(f"error during downloading: {response.status_code}")


def setup_models():
    
    mobilenet_model_url = "https://github.com/onnx/models/raw/refs/heads/main/Computer_Vision/mobilenetv3_small_050_Opset16_timm/mobilenetv3_small_050_Opset16.onnx"
    mobilenet_model_path = MODELS_DIR / "mobilefacenet.onnx"

    blazeface_url = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
    blazeface_path = MODELS_DIR / "blaze_face_short_range.tflite"

    if not mobilenet_model_path.exists():
        download_file(mobilenet_model_url, mobilenet_model_path)
    else:
        print("Model mobilenet already exsits")
    
    if not blazeface_path.exists():
        download_file(blazeface_url, blazeface_path)
    else:
        print("Model blazeface already exsits")


if __name__ == "__main__":
    setup_models()
