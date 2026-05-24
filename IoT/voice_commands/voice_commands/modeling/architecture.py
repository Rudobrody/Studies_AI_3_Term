import torch.nn as nn
import torch

from torchaudio.models import Wav2Letter
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

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
    

class AudioMobileNetV3(nn.Module):
    """
    Architecture based on MobileNetV3.

    Args:
    num_classes (int): Number of classes
    pretrained (bool): Using pretrained model (True) or take only architecture (false) 
    freeze_features (bool): 
    """
    def __init__(self, num_classes: int, pretrained: bool=True, freeze_features: bool=False):
        
        super().__init__()

        # Loading architecture of the model
        # With weights
        if pretrained:
            print("Loading model MobileNetV3 with pretrained weights..")
            weights = MobileNet_V3_Small_Weights.DEFAULT
            self.model = mobilenet_v3_small(weights=weights)
        # Without weights - only architecture
        else:
            print("Loading only architecture of the MobileNetV3")
            self.model = mobilenet_v3_small(weights=None)

        # Modification of the input (3 channels -> 1 channel)
        original_first_layer = self.model.features[0][0]

        new_first_layer = nn.Conv2d(
            in_channels=1, # here we force 1 channel
            out_channels=original_first_layer.out_channels,
            kernel_size=original_first_layer.kernel_size,
            stride=original_first_layer.stride,
            padding=original_first_layer.padding,
            bias=original_first_layer.bias is not None
        )

        # weights also has to be changed because of different number of channels so we have to mean
        if pretrained:
            with torch.no_grad():
                new_first_layer.weight[:] = original_first_layer.weight.mean(dim=1, keepdim=True)

        
        # Exchange of layers
        self.model.features[0][0] = new_first_layer

        # Freezing of weights and training only classifier head
        if pretrained and freeze_features:
            print("Freezing all feature extractor layers.")
            for param in self.model.features.parameters():
                param.requires_grad = False
        elif pretrained and not freeze_features:
            # Freeze
            for param in self.model.features.parameters():
                param.requires_grad = False
            # Unfreeze last 3 blocks
            print("Fine-tuning: Unfreezing last blocks of the model.")
            # Odmrażamy ostatnie 3 bloki `InvertedResidual` w `features`
            for i in range(len(self.model.features) - 3, len(self.model.features)):
                for param in self.model.features[i].parameters():
                    param.requires_grad = True
                    
        # Modification of head classifier to adjust model to what we wanna predict
        in_features = self.model.classifier[3].in_features
        self.model.classifier[3] = nn.Linear(in_features, num_classes)


    def forward(self, x):
        # MobileNet expect (batch, channels, height, width)
        # My dataset return (batch, frequency, time) so we have to add dimension of channel
        if x.dim() == 3:
            x = x.unsqueeze(1)
        return self.model(x)


class AudioWav2Letter(nn.Module):
    """
    Architecture based on Facebook's Wav2Letter designed specifically for speech recognition.
    Uses 1D Convolutions over time, treating Mel bins as channels.
    """
    def __init__(self, num_classes: int):
        super().__init__()
        
        # input_type='mfcc' simply means it expects features on the frequency axis 
        # num_features=64 because we are using 64 mel bins in our spectrograms
        self.model = Wav2Letter(num_classes=num_classes, input_type='mfcc', num_features=64)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The DataLoader outputs shape: (batch_size, channels=1, n_mels=64, time)
        # Wav2Letter expects: (batch_size, num_features=64, time)
        if x.dim() == 4:
            x = x.squeeze(1)
            
        # output shape is (batch_size, num_classes, time)
        logits = self.model(x)
        
        pooled_logits, _ = logits.max(dim=2)
        
        return pooled_logits