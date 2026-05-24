# from pathlib import Path

# from loguru import logger
# from tqdm import tqdm
# import typer

# from voice_commands.config import FIGURES_DIR, PROCESSED_DATA_DIR

# app = typer.Typer()


# @app.command()
# def main(
#     # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
#     input_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
#     output_path: Path = FIGURES_DIR / "plot.png",
#     # -----------------------------------------
# ):
#     # ---- REPLACE THIS WITH YOUR OWN CODE ----
#     logger.info("Generating plot from data...")
#     for i in tqdm(range(10), total=10):
#         if i == 5:
#             logger.info("Something happened for iteration 5.")
#     logger.success("Plot generation complete.")
#     # -----------------------------------------


# if __name__ == "__main__":
#     app()

import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import math
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
import seaborn as sns
from pathlib import Path
import os

from voice_commands.config import FIGURES_DIR

def _get_save_path(title: str) -> Path:
    """Bezpieczne generowanie ścieżki: tworzy foldery i zapewnia rozszerzenie .png"""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    filename = os.path.basename(title) # Ignoruje wcześniejsze ścieżki (np. "reports/figures/...")
    if not filename.endswith('.png'):
        filename += '.png'
    return FIGURES_DIR / filename


def plot_training_history(history: dict[str, list[float]], title="loss_acc_our_voices") -> None:
    """Plots the training and validation loss and accuracy curves."""
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(14, 5))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], label='Train Loss', marker='o')
    plt.plot(epochs, history["val_loss"], label='Validation Loss', marker='o')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_acc"], label='Train Accuracy', marker='o')
    plt.plot(epochs, history["val_acc"], label='Validation Accuracy', marker='o')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(_get_save_path(title))
    plt.close()


def visualize_mel_spectrogram(dataset: Dataset, index: int = 0, label_mapping: dict = None, title="mel_spectrogram_our_voices") -> None:
    """
    Pulls a specific audio sample of each class from the dataset, converts the raw Mel-spectrogram 
    power to a logarithmic Decibel (dB) scale, and plots it.
    """
    num_classes = len(label_mapping)

    # Setup grid dimensions
    cols = 6
    rows = math.ceil(num_classes / cols)

   # Reverse the mapping so we can look up the text name by its integer ID
    id_to_name = {idx: name for name, idx in label_mapping.items()}
    
    # Dictionary to hold one Mel-spectrogram tensor per class
    examples = {}
    
    print("Scanning dataset for class examples..")
    for index, label_id in enumerate(dataset.labels):
        # If we haven't found an example for this class yet, extract it
        if label_id not in examples:
            mel_spec, _ = dataset[index]
            examples[label_id] = mel_spec
            
        # Stop searching once we have 1 example for every class
        if len(examples) == num_classes:
            break

    print("Generating spectrogram matrix..")
    fig, axes = plt.subplots(rows, cols, figsize=(20, 3 * rows))
    axes = axes.flatten() # Flatten the 2D array of axes for easy iteration

    for i in range(rows * cols):
        ax = axes[i]
        
        if i < num_classes:
            if i in examples:
                # We have a class for this grid square
                mel_spec = examples[i].squeeze().numpy()
                mel_spec_db = 10 * np.log10(mel_spec + 1e-9)
                
                ax.imshow(mel_spec_db, origin='lower', aspect='auto', cmap='magma')
                ax.set_title(f"ID: {i} | {id_to_name[i]}", fontsize=10)
            else:
                ax.set_title(f"ID: {i} | {id_to_name[i]}\n(Brak danych)", fontsize=10)
            
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            # Hide the empty grid squares at the end
            ax.axis('off')

    plt.tight_layout()
    plt.savefig(_get_save_path(title))
    plt.close()


def plot_confusion_matrix(
    model: nn.Module, 
    val_loader: DataLoader, 
    device: torch.device, 
    label_mapping: dict[str, int],
    title="confusion_matrix_our_voices"
) -> None:
    """Evaluates the model and plots a confusion matrix heatmap."""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            # Move tensors back to CPU and convert to python lists
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    # Generate the matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Invert the dictionary to map integers back to text names ( 0 -> "Start")
    # We sort them by index to ensure the axis labels match the matrix order
    class_names = [name for name, idx in sorted(label_mapping.items(), key=lambda item: item[1])]

    # Plotting
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix on Validation Set')
    plt.xlabel('Predicted Command')
    plt.ylabel('True Command')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(_get_save_path(title))
    plt.close()
