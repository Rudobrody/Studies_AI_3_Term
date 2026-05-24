import torch
import json
import onnx
from pathlib import Path

from voice_commands.modeling.architecture import AudioCNN, AudioMobileNetV3
from voice_commands.config import MODELS_DIR, PROCESSED_DATA_DIR
from voice_commands.wrapper import AudioONNXWrapper

def get_num_classes(experiment_name: str) -> int:
    """Wczytuje dynamicznie liczbę klas z pliku JSON dla danego eksperymentu."""
    json_path = PROCESSED_DATA_DIR / f"classes_{experiment_name}.json"
    if json_path.exists():
        with open(json_path, 'r', encoding='utf-8') as f:
            return len(json.load(f))
    print(f"Warning: Nie znaleziono pliku {json_path}. Używam domyślnych 34 klas.")
    return 34

def export_to_onnx(cnn_name, mobilenet_name):
    cnn_name = cnn_name
    mobilenet_name = mobilenet_name # Jeśli chcesz wyeksportować zamrożony model zmień na "best_audio_model_mobilenet_frozen"

    # Inicjalizacja modeli z dynamicznie pobraną liczbą klas
    base_audio_cnn = AudioCNN(num_classes=get_num_classes(cnn_name))
    base_mobile_net = AudioMobileNetV3(num_classes=get_num_classes(mobilenet_name), pretrained=False)

    # Loading weights
    base_audio_cnn.load_state_dict(torch.load(MODELS_DIR / f"{cnn_name}.pth", map_location=torch.device('cpu')))
    base_mobile_net.load_state_dict(torch.load(MODELS_DIR / f"{mobilenet_name}.pth", map_location=torch.device('cpu')))

    wrapped_audio_cnn = AudioONNXWrapper(base_audio_cnn)
    wrapped_mobile_net = AudioONNXWrapper(base_mobile_net)

    # Eval mode 
    wrapped_audio_cnn.eval()
    wrapped_mobile_net.eval()

    dummy_input = torch.randn(1, 16000) 
    
    def export_and_downgrade(model, filename, input_name='input', output_name='logits'):
        onnx_path = MODELS_DIR / "onnx_dir" / filename
        print(f"Exporting to {onnx_path}...")
        
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            input_names=[input_name],
            output_names=[output_name],
            # Obniżamy opset z powrotem do 11, by parser Protobuf w aplikacji mógł go odczytać
            opset_version=11,
            export_params=True
        )

        # Wymuszenie osadzenia wag wewnątrz głównego pliku ONNX
        # Dzięki temu aplikacja C++ nie będzie szukać zewnętrznego pliku (.onnx.data)
        onnx_model = onnx.load(str(onnx_path), load_external_data=True)
        onnx.save_model(onnx_model, str(onnx_path))

        # Skrypt na wszelki wypadek usuwa resztki zewnętrznego pliku z danymi (aby nie mieszał nam w głowie)
        external_data_path = Path(str(onnx_path) + ".data")
        if external_data_path.exists():
            external_data_path.unlink()

    export_and_downgrade(wrapped_audio_cnn, f"{cnn_name}.onnx")
    export_and_downgrade(wrapped_mobile_net, f"{mobilenet_name}.onnx")

    print("All models were exported successfully!")

if __name__ == "__main__":
    # export_to_onnx(cnn_name="cnn", mobilenet_name="mobilenet_frozen")
    # export_to_onnx(cnn_name="cnn_micr_splitted", mobilenet_name="mobilenet_frozen_micr_splitted")
    # export_to_onnx(cnn_name="cnn_2nd_group", mobilenet_name="mobilenet_frozen_2nd_group")
    # export_to_onnx(cnn_name="cnn_combined", mobilenet_name="mobilenet_frozen_combined")
    export_to_onnx(cnn_name="cnn", mobilenet_name="mobilenet_pretrained_tuning")