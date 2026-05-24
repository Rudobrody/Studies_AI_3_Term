import onnxruntime as ort
import numpy as np
import json
import os

from voice_commands.config import MODELS_DIR, PROCESSED_DATA_DIR

def load_classes(json_path=PROCESSED_DATA_DIR / "classes_mobilenet.json"):
    if not os.path.exists(json_path):
        print(f"Error, there is no file with classes, your given path: {json_path}")
        return []
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def print_confidence_bars(probabilities, class_names):
    """Wizualizuje pewność predykcji za pomocą prostych pasków w konsoli."""
    print("\n--- Rozkład pewności komend ---")
    sorted_indices = np.argsort(probabilities)[::-1]
    
    for idx in sorted_indices:
        word = class_names[idx] if idx < len(class_names) else f"Class {idx}"
        prob = probabilities[idx]
        bar_length = int(prob * 30)
        bar = '█' * bar_length + '-' * (30 - bar_length)
        print(f"{word:<15} | [{bar}] {prob*100:>5.1f}%")
    print("-------------------------------\n")

def test_onnx_locally(onnx_file_path: str):
    """
    Loads an ONNX Model and test it with a simulated 1-second audio input
    """
    print(f"Loading oonx model from: {onnx_file_path}")

    # Initialize the ONNX Runtime session
    session = ort.InferenceSession(onnx_file_path)

    # Get the input and output names
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    # Create dummy input
    dummy_audio_data = np.random.randn(1, 16000).astype(np.float32)

    # Running inference
    print("Running inference..")

    # The run method takes (outputs_names, input_dict)
    # Passing None for output_names asks it to return all outputs
    outputs = session.run(None, {input_name: dummy_audio_data})

    # Extract and print results
    logits = outputs[0]

    # Obliczenie softmax do uzyskania prawdopodobieństw
    exp_values = np.exp(logits - np.max(logits))
    probabilities = (exp_values / np.sum(exp_values, axis=1, keepdims=True))[0]

    command_words = list(load_classes())
    predicted_class_id = np.argmax(probabilities)
    predicted_word = command_words[predicted_class_id] if command_words else str(predicted_class_id)

    print(f"Predicted word: {predicted_word} (ID: {predicted_class_id})")
    if command_words:
        print_confidence_bars(probabilities, command_words)


if __name__ == "__main__":
    test_onnx_locally(MODELS_DIR / "command_classifier.onnx")
