import onnxruntime as ort
import numpy as np
import sounddevice as sd
import os
import json
import queue


from voice_commands.config import MODELS_DIR, PROCESSED_DATA_DIR, DATA_DIR, ONNX_DIR


def calculate_softmax(logits):
    """Converts raw logits into probabilitites (0 to 1) that sum up to to 1"""
    exp_values = np.exp(logits - np.max(logits))
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)


def load_classes(json_path):
    """Function which returns classes from file"""
    if not os.path.exists(json_path):
        print(f"Error, there is no file with classes, your given path: {json_path}")
        return []
    
    with open(json_path, 'r', encoding='utf-8') as f:
        classes = json.load(f)
        return classes
    
    
def print_confidence_bars(probabilities, class_names):
    """Visualisation of confeince of the prediction for each word"""
    print("\n--- Probability of each command ---")
    # Load indices sorted from the lowest to high probability 
    sorted_indices = np.argsort(probabilities)[::-1]
    
    for idx in sorted_indices:
        word = class_names[idx]
        prob = probabilities[idx]
        bar_length = int(prob * 30) 
        bar = '█' * bar_length + '-' * (30 - bar_length)
        print(f"{word:<15} | [{bar}] {prob*100:>5.1f}%")
    print("----------------------------------------------\n")


def run_live_inference(onnx_file_path, json_classes_path):
    print(f"Loading model: {onnx_file_path}")

    print(f"Loading classes from file: {json_classes_path}")
    command_words = list(load_classes(json_classes_path))
    
    # If there is no labels
    if not command_words:
        return

    try:
        session = ort.InferenceSession(onnx_file_path)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    input_name = session.get_inputs()[0].name

    # Configuration for constant watchdog
    SAMPLE_RATE = 16000
    WINDOW_SIZE = 16000          # Complete windows (1 sec)
    CHUNK_SIZE = 4000            # Checking (0.25 s interval) 
    VOLUME_THRESHOLD = 0.05      
    COOLDOWN_CHUNKS = 4          # Freeze detection after threshold activated

    q = queue.Queue()
    audio_buffer = np.zeros(WINDOW_SIZE, dtype=np.float32)

    def audio_callback(indata, frames, time_info, status):
        """Function to call automated detection of speech"""
        if status:
            print(status, flush=True)
        q.put(indata[:, 0].copy())

    print("\n--- Live Audio Classifier Started ---")
    print(f"Say command to launch detection, threshold RMS: {VOLUME_THRESHOLD}).")
    print("Push Ctrl+C to end.\n")

    try:
        # Open a stream recording
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='float32', blocksize=CHUNK_SIZE, callback=audio_callback):
            cooldown = 0
            
            while True:
                # Download the newest chhunk from microphone 
                chunk = q.get()
                
                # Shift 1 sec buffer 
                audio_buffer = np.roll(audio_buffer, -CHUNK_SIZE) # Argument -chunk size is shiftting values to the left

                # Add new chank at the end of the list 
                audio_buffer[-CHUNK_SIZE:] = chunk
                
                if cooldown > 0:
                    cooldown -= 1
                    continue
                
                # Counter the RMS of the singnal
                rms = np.sqrt(np.mean(chunk**2))
                
                # If RMS was higher than threshold
                if rms > VOLUME_THRESHOLD:
                    print(f"\n[Sound detected, volume:{rms:.4f}] - processing command...")
                    
                    # Download to chunks more to detect full command 
                    for _ in range(2):
                        extra_chunk = q.get()
                        audio_buffer = np.roll(audio_buffer, -CHUNK_SIZE)
                        audio_buffer[-CHUNK_SIZE:] = extra_chunk
                    
                    # Extension of the dimension to size: (batch_size=1, time=16000)
                    audio_input = np.expand_dims(audio_buffer, axis=0)

                    # Launc onnx
                    outputs = session.run(None, {input_name: audio_input})
                    logits = outputs[0]

                    probabilities = calculate_softmax(logits)[0]

                    predicted_class_id = np.argmax(probabilities)
                    predicted_word = command_words[predicted_class_id]
                    highest_confidence = probabilities[predicted_class_id]

                    print(f"Predicted word: {predicted_word} with confidence: {highest_confidence:.3f}")
                    print_confidence_bars(probabilities, command_words)
                    
                    print("Listening..")
                    
                    # Setting delay
                    cooldown = COOLDOWN_CHUNKS 

    except KeyboardInterrupt:
        print("Exiting live inference")

if __name__ == "__main__":
    # Wybierz model ONNX do przetestowania
    ONNX_MODEL_PATH = ONNX_DIR / "mobilenet_frozen.onnx"
    # ONNX_MODEL_PATH = MODELS_DIR / "command_classifier_cnn.onnx"
    
    # Wybierz odpowiadający mu plik JSON z klasami z folderu processed
    # Upewnij się, że nazwa zgadza się z eksperymentem, z którego korzystasz!
    CLASSES_JSON_PATH = PROCESSED_DATA_DIR / "classes_mobilenet_frozen.json"
    # CLASSES_JSON_PATH = PROCESSED_DATA_DIR / "classes_best_audio_cnn.json"

    run_live_inference(ONNX_MODEL_PATH, CLASSES_JSON_PATH)
