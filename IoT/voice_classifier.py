import os
from typing import List, Tuple

import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import Dataset, DataLoader

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
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        num_epochs: int,
        device: torch.device
) -> None:
    """
    Executes the training loop for the neural network.

    Args:
        model (nn.Module): The PyTorch model to be trained.
        train_loader (DataLoader): The DataLoader containing the training dataset.
        criterion (nn.Module): The loss function (e.g., CrossEntropyLoss).
        optimizer (torch.optim.Optimizer): The optimization algorithm.
        num_epochs (int): The number of complete passes through the training dataset.
        device (torch.device): The computation device ('cpu' or 'cuda').
    """
    # Transfer model to device 
    model.to(device)

    for epoch in range(num_epochs):
        
        # Set a training mode
        model.train()

        # Reset of some metrics
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        for inputs, labels in train_loader:

            # Transfer vectors to device
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            # Predict
            outputs = model(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(outputs, dim=1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = (correct_predictions / total_samples) * 100
        print(f"Epoch [{epoch+1}/{num_epochs}] | Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.2f}%")