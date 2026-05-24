import torch
import random
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader
import json
from collections import defaultdict

from voice_commands.config import DATA_DIR, TARGET_SAMPLE_RATE, BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS, MODELS_DIR, PROCESSED_DATA_DIR, REPORTS_DIR, PROCESSED_2ND_DATA_DIR
from voice_commands.dataset import parse_audio_directory, CommandDataset
from voice_commands.modeling.architecture import AudioCNN, AudioMobileNetV3, AudioWav2Letter
from voice_commands.modeling.train import train_model
from voice_commands.plots import plot_confusion_matrix , plot_training_history, visualize_mel_spectrogram

def run_experiment(
    experiment_name: str,
    data_dirs: list[Path] | Path,
    model_type: str = 'AudioCNN',
    split_strategy: str = 'random',  # 'random', 'microphone', 'none'
    target_val_mic: str = 'mikr',
    num_epochs: int = NUM_EPOCHS,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
    pretrained: bool = False,
    freeze_features: bool = False,
    weight_decay: float = 1e-2
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*50}")
    print(f"Starting Experiment: {experiment_name}")
    print(f"{'='*50}")
    print(f"Using computation device: {device}")

    print("Parsing dataset directory...")
    file_paths, labels, metadata, label_mapping = parse_audio_directory(data_dirs)
    num_classes = len(label_mapping)

    # Prevent crashing if the directory is empty or path is wrong
    if len(file_paths) == 0:
        raise ValueError(f"No audio files found in {data_dirs}. Please check the path.")

    # Save unique labels to JSON
    unique_commands_sorted = [cmd for cmd, idx in sorted(label_mapping.items(), key=lambda item: item[1])]
    classes_path = PROCESSED_DATA_DIR / f"classes_{experiment_name}.json"
    with open(classes_path, "w", encoding='utf-8') as f:
        json.dump(unique_commands_sorted, f, indent=4, ensure_ascii=False)

    # Splitting logic
    train_paths, train_labels = [], []
    val_paths, val_labels = [], []

    if split_strategy == 'random':
        print("Splitting data into 80% Training and 20% Validation (Grouped Stratified Split)...")
        
        # Grupowanie po klasach (stratified) i autorach (grouped) aby uniknąć data leakage
        class_to_author_to_paths = defaultdict(lambda: defaultdict(list))
        
        for path, label, meta in zip(file_paths, labels, metadata):
            author = meta["author"]
            class_to_author_to_paths[label][author].append(path)
            
        for label, author_dict in class_to_author_to_paths.items():
            authors = list(author_dict.keys())
            random.shuffle(authors)
            
            total_paths = sum(len(paths) for paths in author_dict.values())
            target_train_count = int(total_paths * 0.8)
            
            current_train_count = 0
            label_train_paths = []
            label_val_paths = []
            
            for author in authors:
                paths = author_dict[author]
                # Przydzielamy paczkę próbek autora do train, jeśli nie przekroczymy limitu
                # lub jeśli train jest wciąż puste (musimy dać coś do train)
                if current_train_count + len(paths) <= target_train_count or current_train_count == 0:
                    label_train_paths.extend(paths)
                    current_train_count += len(paths)
                else:
                    label_val_paths.extend(paths)
            
            # Zabezpieczenie: jeśli algorytm wrzucił wszystko do train, a mamy więcej niż 1 autora
            if len(label_val_paths) == 0 and len(authors) > 1:
                # Przenosimy ostatniego wrzuconego autora do val
                last_author = authors[-1]
                paths_to_move = author_dict[last_author]
                label_train_paths = label_train_paths[:-len(paths_to_move)]
                label_val_paths.extend(paths_to_move)

            train_paths.extend(label_train_paths)
            train_labels.extend([label] * len(label_train_paths))
            val_paths.extend(label_val_paths)
            val_labels.extend([label] * len(label_val_paths))
            
        # Przetasowanie całych zbiorów po podziale, żeby zachować losowość batchy
        combined_train = list(zip(train_paths, train_labels))
        random.shuffle(combined_train)
        train_paths, train_labels = map(list, zip(*combined_train) if combined_train else ([], []))
        
        combined_val = list(zip(val_paths, val_labels))
        random.shuffle(combined_val)
        val_paths, val_labels = map(list, zip(*combined_val) if combined_val else ([], []))

    elif split_strategy == 'microphone':
        print(f"Splitting data based on microphone. Validation mic: {target_val_mic}")
        for path, label, meta in zip(file_paths, labels, metadata):
            if meta["microphone"] == target_val_mic:
                val_paths.append(path)
                val_labels.append(label)
            else:
                train_paths.append(path)
                train_labels.append(label)
        
        combined_train = list(zip(train_paths, train_labels))
        random.shuffle(combined_train)
        train_paths, train_labels = map(list, zip(*combined_train) if combined_train else ([], []))
        
        combined_val = list(zip(val_paths, val_labels))
        random.shuffle(combined_val)
        val_paths, val_labels = map(list, zip(*combined_val) if combined_val else ([], []))

    elif split_strategy == 'none':
        print("Using all data for training (no validation set).")
        combined_train = list(zip(file_paths, labels))
        random.shuffle(combined_train)
        train_paths, train_labels = map(list, zip(*combined_train))
        # Hack: Pass train data as val data so `train.py` doesn't crash on ZeroDivisionError
        val_paths, val_labels = list(train_paths), list(train_labels)  
        print("Note: Validation metrics will reflect training data.")
    
    print(f"Training on {len(train_paths)} files, Validating on {len(val_paths)} files.")

    train_dataset = CommandDataset(train_paths, train_labels, TARGET_SAMPLE_RATE, augment=True)
    val_dataset = CommandDataset(val_paths, val_labels, TARGET_SAMPLE_RATE, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Model Initialization
    if model_type == 'AudioCNN':
        model = AudioCNN(num_classes=num_classes)
    elif model_type == 'MobileNetV3':
        model = AudioMobileNetV3(num_classes=num_classes, pretrained=pretrained, freeze_features=freeze_features)
    elif model_type == 'Wav2Letter':
        model = AudioWav2Letter(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    criterion = nn.CrossEntropyLoss()
    
    # 1. Differential Learning Rates (tylko dla fine-tuningu MobileNetV3)
    if model_type == 'MobileNetV3' and pretrained and not freeze_features:
        print("Using Differential Learning Rates for fine-tuning...")
        optimizer = torch.optim.AdamW([
            {'params': model.model.features.parameters(), 'lr': learning_rate * 0.1},  # 10x mniejszy dla bazy
            {'params': model.model.classifier.parameters(), 'lr': learning_rate}       # Normalny (np. 1e-3) dla nowej głowy
        ], weight_decay=weight_decay)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # 2. Adaptive Learning Rate Scheduler
    # Zmniejsza LR o połowę (factor=0.5) jeśli val_acc (mode='max') nie poprawi się przez 3 epoki (patience=3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, min_lr=1e-6
    )

    save_path = MODELS_DIR / f"{experiment_name}.pth"

    print("\nStarting training pipeline..")
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=num_epochs,
        device=device,
        save_path=save_path,
        scheduler=scheduler
    )
    print("\nPipeline execution finished. Best model saved to disk.")

    print("\nGenerating visualizations..")
    visualize_mel_spectrogram(train_dataset, index=0, label_mapping=label_mapping, title=f"mel_spectrogram_{experiment_name}")
    plot_training_history(history, title=f"loss_acc_{experiment_name}")
        
    # Reload best model and plot confusion matrix
    model.load_state_dict(torch.load(save_path, map_location=device))
    plot_confusion_matrix(model, val_loader, device, label_mapping, title=f"confusion_matrix_{experiment_name}")


if __name__ == '__main__':
    # 1. Base AudioCNN
    # run_experiment(
    #     experiment_name="cnn",
    #     data_dirs=PROCESSED_DATA_DIR,
    #     model_type="AudioCNN",
    #     split_strategy="random"
    # )

    # 2. AudioCNN with Microphone Split
    # run_experiment(
    #     experiment_name="cnn_micr_splitted",
    #     data_dirs=PROCESSED_DATA_DIR,
    #     model_type="AudioCNN",
    #     split_strategy="microphone",
    #     target_val_mic="mikr"
    # )

    # 3. AudioCNN on 2nd Group (Train only)
    # run_experiment(
    #     experiment_name="cnn_2nd_group",
    #     data_dirs=PROCESSED_2ND_DATA_DIR,
    #     model_type="AudioCNN",
    #     split_strategy="random"
    # )

    # 4. AudioCNN on Combined Data
    # run_experiment(
    #     experiment_name="cnn_combined",
    #     data_dirs=[PROCESSED_DATA_DIR, PROCESSED_2ND_DATA_DIR],
    #     model_type="AudioCNN",
    #     split_strategy="random"
    # )

    # 5. MobileNetV3 - Transfer Learning (Pretrained + Frozen features)
    run_experiment(
        experiment_name="mobilenet_frozen",
        data_dirs=PROCESSED_DATA_DIR,
        model_type="MobileNetV3",
        split_strategy="random",
        pretrained=True,
        freeze_features=True
    )

    # 6. MobileNetV3 - Transfer Learning (Pretrained + Frozen) with Microphone Split
    # run_experiment(
    #     experiment_name="mobilenet_frozen_micr_splitted",
    #     data_dirs=PROCESSED_DATA_DIR,
    #     model_type="MobileNetV3",
    #     split_strategy="microphone",
    #     target_val_mic="mikr",
    #     pretrained=True,
    #     freeze_features=True
    # )

    # 7. MobileNetV3 - Transfer Learning (Pretrained + Frozen features) on 2nd Group (Train only)
    # run_experiment(
    #     experiment_name="mobilenet_frozen_2nd_group",
    #     data_dirs=PROCESSED_2ND_DATA_DIR,
    #     model_type="MobileNetV3",
    #     split_strategy="random",
    #     pretrained=True,
    #     freeze_features=True
    # )

    # 8. MobileNetV3 - Transfer Learning (Pretrained + Frozen features) combined
    # run_experiment(
    #     experiment_name="mobilenet_frozen_combined",
    #     data_dirs=[PROCESSED_DATA_DIR, PROCESSED_2ND_DATA_DIR],
    #     model_type="MobileNetV3",
    #     split_strategy="random",
    #     pretrained=True,
    #     freeze_features=True
    # )

    # 8. MobileNetV3 - Training from scratch (No pretrained weights)
    # run_experiment(
    #     experiment_name="mobilenet_scratch",
    #     data_dirs=PROCESSED_DATA_DIR,
    #     model_type="MobileNetV3",
    #     split_strategy="random",
    #     pretrained=False,
    #     freeze_features=False
    # )

    # 9. MobileNetV3 - Training from scratch with Microphone Split
    # run_experiment(
    #     experiment_name="mobilenet_scratch_micr_splitted",
    #     data_dirs=PROCESSED_DATA_DIR,
    #     model_type="MobileNetV3",
    #     split_strategy="microphone",
    #     target_val_mic="mikr",
    #     pretrained=False,
    #     freeze_features=False
    # )

    # 10. MobileNetV3 - Training from scratch (No pretrained weights) on 2nd Group (Train only)
    # run_experiment(
    #     experiment_name="mobilenet_scratch_2nd_group",
    #     data_dirs=PROCESSED_2ND_DATA_DIR,
    #     model_type="MobileNetV3",
    #     split_strategy="random",
    #     pretrained=False,
    #     freeze_features=False
    # )

    # 11. MobileNetV3 - Training from scratch (No pretrained weights) combined
    # run_experiment(
    #     experiment_name="mobilenet_scratch_combined",
    #     data_dirs=[PROCESSED_DATA_DIR,PROCESSED_2ND_DATA_DIR],
    #     model_type="MobileNetV3",
    #     split_strategy="random",
    #     pretrained=False,
    #     freeze_features=False
    # )

    # 12. MobileNetV3 - Transfer Learning (Pretrained) + tuning
    # run_experiment(
    #     experiment_name="mobilenet_pretrained_tuning",
    #     data_dirs=PROCESSED_DATA_DIR,
    #     model_type="MobileNetV3",
    #     split_strategy="random",
    #     pretrained=True,
    #     freeze_features=False,
    # )
    
    # 13. Wav2Letter - Specjalistyczny model do Speech Recognition
    # run_experiment(
    #     experiment_name="wav2letter_scratch",
    #     data_dirs=PROCESSED_DATA_DIR,
    #     model_type="Wav2Letter",
    #     split_strategy="random",
    # )