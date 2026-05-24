import torch
import torch.nn as nn
import torch.nn.functional as F
from nnAudio.features.mel import MelSpectrogram

from voice_commands.config import TARGET_SAMPLE_RATE

class AudioONNXWrapper(nn.Module):
    def __init__(self, original_model: nn.Module):
        super().__init__()

        # Transform raw audio into spectrogram using nnAudio
        # It uses 1D convolutions instead of complex numbers, making it ONNX-friendly
        self.spectrogram_transform = MelSpectrogram(
            sr=16000, 
            n_mels=64, 
            n_fft=1024, 
            hop_length=512,
            window='hann',
            center=True,
            pad_mode='reflect',
            htk=False,
            fmin=0.0,
            fmax=None
        )

        self.classifier = original_model


    def forward(self, waveform:torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveform (torch.Tensor): (batch_size, time) like (1, 16000)
        """
        
        # Transformation, in result we will get (batch_size, n_mels, time) -> (1, 64, 63)
        mel_spec = self.spectrogram_transform(waveform)

        # changing to dB
        mel_spec = 10.0 * torch.log10(torch.clamp(mel_spec, min=1e-10))
        top_db = 80.0
        max_db = mel_spec.amax(dim=(-2, -1), keepdim=True)
        mel_spec = torch.clamp(mel_spec, min=max_db - top_db)

        # Standarization per sample
        mean = mel_spec.mean()
        std = mel_spec.std() + 1e-8
        mel_spec = (mel_spec - mean) / std
        
        # Add channel dimension to match expected (batch_size, channels, n_mels, time)
        # Models like AudioCNN strictly expect 4D inputs.
        if mel_spec.dim() == 3:
            mel_spec = mel_spec.unsqueeze(1)

        # Classification
        logits = self.classifier(mel_spec)

        return logits
