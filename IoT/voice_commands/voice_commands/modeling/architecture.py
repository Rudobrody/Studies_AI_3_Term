import torch.nn as nn
import torch


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
    
