# %%
import os
from typing import List, Tuple
from pathlib import Path

import torch
import random
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math
import ffmpeg

import torchaudio
import torch.nn as nn
import torchaudio
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix


# %%
class CommandDataset(Dataset):
    """
    This dataset reads .wav files, resamples them to a target sample rate, 
    and converts the raw waveform into a Mel-spectrogram.

    Args:
        file_paths (List[str]): List of absolute paths to the audio files.
        labels (List[int]): List of integer command labels.
        target_sample_rate (int): The sample rate to which all audio will be converted.
    """
    def __init__(self, file_paths: List[str], labels: List[int], target_sample_rate: int = 16000):
        self.file_paths = file_paths
        self.labels = labels
        self.target_sample_rate = target_sample_rate

        self.transformation = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.target_sample_rate, 
            n_mels=64, 
            n_fft=1024, 
            hop_length=512 # Hop length is like a stride for audio
        )


    def __len__(self) -> int:
        """Returns the total number of samples in the dataset."""
        return len(self.file_paths)
    


    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        """
        Retrieves an audio sample and its label, applying necessary transformations.
        
        Args:
            index (int): The index of the item to retrieve.

        Returns:
            Tuple[torch.Tensor, int]: A tuple containing the Mel-spectrogram tensor and its label.
        """
        # Load the sample 
        waveform, sample_rate = torchaudio.load(self.file_paths[index])

        # If there are more channels lets average them 
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Resample 
        if sample_rate != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.target_sample_rate)
            waveform = resampler(waveform)

        # Standarize the length, so we set as one second so 32 000
        num_samples = self.target_sample_rate * 2
        
        # If the sample of the audio is longer than 1 sec so we trim
        if waveform.shape[1] > num_samples:
            waveform[:, :num_samples] 
        
        # If it is shorter we add some zeros
        elif waveform.shape[1] < num_samples:
            pad_amount = num_samples - waveform.shape[1]
            waveform = F.pad(waveform, (0, pad_amount)) # why (0, pad amount?) because it is padding (left, right), we don't

        # Aplying trasnformation 
        mel_spec = self.transformation(waveform)

        return mel_spec, self.labels[index]


class AudioCNN(nn.Module):
    """
    A simple Convolutional Neural Network for classifying Mel-spectrograms.
    
    Expects input tensor of shape (batch_size, channels, mel_bins, time_frames).
    """
    def __init__(self, num_classes: int = 22):
        super().__init__()

        self.model = nn.Sequential( 
            # In 1 because we have mono audio
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
            
            # Batch Norm with number of features 
            nn.BatchNorm2d(32),
            nn.ReLU(),

            # Maxpooling with stride 2 to downsample the image 
            nn.MaxPool2d(kernel_size=2, stride=2), 

            nn.Conv2d(in_channels=32, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # AdaptiveAvgPool ensures the output spatial dimensions are exactly (4, 4)
            # This is crucial because audio files have varying lengths, which means varying time_frames.
            nn.AdaptiveAvgPool2d((4, 4)),
            
            # After AdaptiveA Avg Pooling we have dimensions (Batch, Channels, Height, Width) so
            # we just skip dimension of batch thats why we have 2D vector
            nn.Flatten(),
            
            nn.Dropout(p=0.5),

            # 128 channels * 4 height * 4 width = 2048 input features
            nn.Linear(in_features=128 * 4 * 4, out_features=num_classes),
        )
    

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the network.
        
        Args:
            x (torch.Tensor): Input batch of Mel-spectrograms.
        
        Returns:
            torch.Tensor: The unnormalized raw predictions (logits).
        """
        return self.model(x)
    

def train_model(
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        num_epochs: int,
        device: torch.device,
        save_path: str = "best_audio_model.pth" 
) -> None:
    """
    Executes the training and validation loop for the neural network
    Saves the model weights
    """

    # Dict to keep metrics over time
    history = {
        "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    
    model.to(device)
    
    # Track the best accuracy to know when to save the model
    best_val_accuracy = 0.0

    for epoch in range(num_epochs):
        
        # Training mode
        model.train() 
        train_running_loss = 0.0
        train_correct = 0
        train_total = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_running_loss += loss.item()
            _, predicted = torch.max(outputs, dim=1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        train_loss = train_running_loss / len(train_loader)
        train_acc = (train_correct / train_total) * 100

        # Validation phase
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_running_loss += loss.item()
                _, predicted = torch.max(outputs, dim=1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_loss = val_running_loss / len(val_loader)
        val_acc = (val_correct / val_total) * 100

        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"| Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% "
              f"| Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        # If the model improved, save the state dictionary
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            print(f"--> Validation accuracy improved! Saving model to {save_path}")
            torch.save(model.state_dict(), save_path)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

    return history


def parse_audio_directory(directory_path: str) -> tuple[list[str], list[int], dict[str, int]]:
    """
    Parses a directory of .wav files, extracting file paths and mapping text command to integer labels

    Formatt of the files: {microphone}_{author}_{command}_{sample_number}.wav
    """
    file_paths = []
    text_commands = []
    metadata = []

    path_obj = Path(directory_path)

    # Iterate through directory
    for file_path in path_obj.rglob("*.wav"):
        file_name = file_path.name
        name_without_ext = file_name.replace(".wav", "")
        
        # Split by _
        parts = name_without_ext.split("_")

        # Extracting data by using list indexing
        try:
            # Microphone is the first item
            microphone = parts[0] 

            # Sample number is always the last item
            sample_number = int(parts[-1])

            command = parts[-2].lower()

            author = parts[1]

            # Store the extracted data
            file_paths.append(str(file_path))
            text_commands.append(command)

            metadata.append({
                "microphone": microphone,
                "author": author,
                "command": command,
                "sample_number": sample_number
            })
        except (IndexError, ValueError):
            print(f"Warning file {file_name} does not match with name conventtion ")

    # Create a mapping of the commands to int
    unique_commands = sorted(list(set(text_commands)))
    label_mapping = {command: idx for idx, command in enumerate(unique_commands)}

    # Convert all text commands to their integer equivalents
    integer_labels = [label_mapping[cmd] for cmd in text_commands]

    print(f"successfully parsed {len(file_paths)} audio files")
    print(f"Identified {len(label_mapping)} unique_commands: {label_mapping}")

    return file_paths, integer_labels, metadata, label_mapping


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
    plt.savefig(title)
    plt.show()


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
            # We have a class for this grid square
            mel_spec = examples[i].squeeze().numpy()
            mel_spec_db = 10 * np.log10(mel_spec + 1e-9)
            
            ax.imshow(mel_spec_db, origin='lower', aspect='auto', cmap='magma')
            ax.set_title(f"ID: {i} | {id_to_name[i]}", fontsize=10)
            
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            # Hide the empty grid squares at the end
            ax.axis('off')

    plt.tight_layout()
    plt.savefig(title)
    plt.show()


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
    plt.savefig(title)
    plt.show()


# %%
# Config
# Path to the dir where .wav files are stored 
DATA_DIR = "processed" 
BATCH_SIZE = 32
LEARNING_RATE = 1e-3  
NUM_EPOCHS = 20
TARGET_SAMPLE_RATE = 16000


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device computation: {device}")


print("Parsing dataset directory...")
file_paths, labels, metadata, label_mapping = parse_audio_directory(DATA_DIR)
num_classes = len(label_mapping)

# Prevent crashing if the directory is empty or path is wrong
if len(file_paths) == 0:
    raise ValueError(f"No audio files found in {DATA_DIR}. Please check the path.")

print("Splitting data into 80% Training and 20% Validation...")
# Why we are using zip? Because we want shuffle them
combined_data = list(zip(file_paths, labels))
random.shuffle(combined_data)

# Unzip them back into separate tuples
shuffled_paths, shuffled_labels = zip(*combined_data)

# Calculate the cutoff index for 80%
split_index = int(len(shuffled_paths) * 0.8)

train_paths = shuffled_paths[:split_index]
train_labels = shuffled_labels[:split_index]

val_paths = shuffled_paths[split_index:]
val_labels = shuffled_labels[split_index:]

# Creating splitted datasets
train_dataset = CommandDataset(list(train_paths), list(train_labels), TARGET_SAMPLE_RATE)
val_dataset = CommandDataset(list(val_paths), list(val_labels), TARGET_SAMPLE_RATE)

# shuffle=True for Train allows the network to learn robustly, but there is no sense to shuffle for
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Initialize the mdoel
# Pass the dynamic number of classes we found during parsing
model = AudioCNN(num_classes=num_classes)

# CrossEntropyLoss expects raw, unnormalized logits
criterion = nn.CrossEntropyLoss()

# AdamW as an optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# Exe 
print("\nStarting training pipeline..")
history = train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=optimizer,
    num_epochs=NUM_EPOCHS,
    device=device,
)

print("\nPipeline execution finished. Best model saved to disk.")

# %%
print("\nGenerating visualizations..")

visualize_mel_spectrogram(train_dataset, index=0, label_mapping=label_mapping)

# Plot the learning curves
plot_training_history(history)
    
# %%
# Plot the confusion matrix using the best saved weights
# First, load the best weights we saved during training
model.load_state_dict(torch.load("best_audio_model.pth"))
plot_confusion_matrix(model, val_loader, device, label_mapping)

#######################################################################
######################################################################
# %%
# Experiment - split dataset based on microphones
print("Parsing dataset directory...")
file_paths, labels, metadata, label_mapping = parse_audio_directory(DATA_DIR)
num_classes = len(label_mapping)

# Prevent crashing if the directory is empty or path is wrong
if len(file_paths) == 0:
    raise ValueError(f"No audio files found in {DATA_DIR}. Please check the path.")

print("Splitting data into 80% Training and 20% Validation based no microphone")

train_paths, train_labels = [], []
val_paths, val_labels = [], []

# Define which microphone is the unseen for test data
target_val_mic = "mikr" 

# Iterate through all the data simultaneously
for path, label, meta in zip(file_paths, labels, metadata):
    
    # If the file came from the studio mic, put it in the validation set
    if meta["microphone"] == target_val_mic:
        val_paths.append(path)
        val_labels.append(label)
        
    # If the file came from Phone 1 or Phone 2, put it in the training set
    else:
        train_paths.append(path)
        train_labels.append(label)

print(f"Training on {len(train_paths)} files (phones).")
print(f"Validating on {len(val_paths)} files (studio mic).")

# Why we are using zip? Because we want shuffle them
combined_train = list(zip(train_paths, train_labels))
random.shuffle(combined_train)

# Unzip them back into separate tuples
train_paths, train_labels = zip(*combined_train)

combined_val = list(zip(val_paths, val_labels))
random.shuffle(combined_val)

# Unzip them back into separate tuples
val_paths, val_labels = zip(*combined_val)

# Creating splitted datasets
train_dataset = CommandDataset(list(train_paths), list(train_labels), TARGET_SAMPLE_RATE)
val_dataset = CommandDataset(list(val_paths), list(val_labels), TARGET_SAMPLE_RATE)

# shuffle=True for Train allows the network to learn robustly, but there is no sense to shuffle for
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Initialize the mdoel
# Pass the dynamic number of classes we found during parsing
model = AudioCNN(num_classes=num_classes)

# CrossEntropyLoss expects raw, unnormalized logits
criterion = nn.CrossEntropyLoss()

# AdamW as an optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# Exe 
print("\nStarting training pipeline..")
history = train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=optimizer,
    num_epochs=NUM_EPOCHS,
    device=device,
    save_path="best_audio_model_micr_splitted.pth"
)

print("\nPipeline execution finished. Best model based on micr splitted saved to disk.")

# %%
print("\nGenerating visualizations..")

visualize_mel_spectrogram(train_dataset, index=0, label_mapping=label_mapping, title="mel_spectrogram_our_voices_micr_splitted")

# Plot the learning curves
plot_training_history(history, title="loss_acc_our_voices_micr_splitted")
    
# %%
# Plot the confusion matrix using the best saved weights
# First, load the best weights we saved during training
model.load_state_dict(torch.load("best_audio_model_micr_splitted.pth"))
plot_confusion_matrix(model, val_loader, device, label_mapping, title="confusion_matrix_our_voices_micr_splitted")