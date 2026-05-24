import json
from pathlib import Path
from voice_commands.config import PROCESSED_DATA_DIR, MODELS_DIR

def export_json_to_txt():
    # Znajdź wszystkie pliki z etykietami w folderze przetworzonych danych
    json_files = list(PROCESSED_DATA_DIR.glob("classes_*.json"))
    
    if not json_files:
        print(f"Nie znaleziono plików JSON w {PROCESSED_DATA_DIR}")
        return

    # Upewniamy się, że folder docelowy na ONNX i etykiety istnieje
    output_dir = MODELS_DIR / "onnx_dir"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for json_path in json_files:
        # Wczytanie listy z pliku JSON
        with open(json_path, 'r', encoding='utf-8') as f:
            classes = json.load(f)
        
        # Budowanie nowej nazwy pliku (np. classes_cnn.json -> labels_cnn.txt)
        txt_filename = json_path.stem.replace("classes_", "labels_") + ".txt"
        txt_path = output_dir / txt_filename
        
        # Zapis do pliku tekstowego (każda klasa w nowej linii)
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(classes) + "\n")
        
        print(f"Zapisano: {txt_path}")

if __name__ == "__main__":
    export_json_to_txt()
