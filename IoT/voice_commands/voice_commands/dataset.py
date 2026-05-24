import torchaudio
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from pathlib import Path

class CommandDataset(Dataset):
    """
    This dataset reads .wav files, resamples them to a target sample rate, 
    and converts the raw waveform into a Mel-spectrogram.

    Args:
        file_paths (List[str]): List of absolute paths to the audio files.
        labels (List[int]): List of integer command labels.
        target_sample_rate (int): The sample rate to which all audio will be converted.
        augment (bool): Whether to apply data augmentation (SpecAugment) to the spectrograms.
    """
    def __init__(self, file_paths: list[str], labels: list[int], target_sample_rate: int = 16000, augment: bool = False):
        self.file_paths = file_paths
        self.labels = labels
        self.target_sample_rate = target_sample_rate
        self.augment = augment

        self.transformation = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.target_sample_rate, 
            n_mels=64, 
            n_fft=1024, 
            hop_length=512 # Hop length is like a stride for audio
        )
        
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(stype='power', top_db=80)
        
        if self.augment:
            # freq_mask_param means cutting max 15 of 64 channels (max ~23%)
            self.freq_masking = torchaudio.transforms.FrequencyMasking(freq_mask_param=15)
            # time_mask_param means cutting max 15 time steps 
            self.time_masking = torchaudio.transforms.TimeMasking(time_mask_param=15)


    def __len__(self) -> int:
        """Returns the total number of samples in the dataset."""
        return len(self.file_paths)
    


    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        """
        Retrieves an audio sample and its label, applying necessary transformations.
        
        Args:
            index (int): The index of the item to retrieve.

        Returns:
            Tuple[torch.Tensor, int]: A tuple containing the Mel-spectrogram tensor and its label.
        """
        # Load the sample 
        waveform, sample_rate = torchaudio.load(self.file_paths[index], backend="soundfile")

        # If there are more channels lets average them 
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Resample 
        if sample_rate != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.target_sample_rate)
            waveform = resampler(waveform)

        # Standarize the length, so we set as one second so 16 000
        num_samples = self.target_sample_rate
        
        # If the sample of the audio is longer than 1 sec so we trim
        if waveform.shape[1] > num_samples:
            waveform = waveform[:, :num_samples] 
        
        # If it is shorter we add some zeros
        elif waveform.shape[1] < num_samples:
            pad_amount = num_samples - waveform.shape[1]
            waveform = F.pad(waveform, (0, pad_amount)) # why (0, pad amount?) because it is padding (left, right)

        # Waveform augmentations before we make mel spectrogram 
        if self.augment:
            # Adding  Random White Noise which simulates background or bad microphone 
            # We use small amplitude
            noise_level = torch.empty(1).uniform_(0.001, 0.02).item()
            noise = torch.randn_like(waveform) * noise_level
            waveform = waveform + noise

            # Random Gain which simulates different distance from the microphone
            # From 70% to 130% of volume
            gain = torch.empty(1).uniform_(0.7, 1.3).item()
            waveform = waveform * gain
            
            # Clipping values to range [-1.0, 1.0], to avoid sizzle sounds
            waveform = torch.clamp(waveform, min=-1.0, max=1.0)

        # Aplying trasnformation 
        mel_spec = self.transformation(waveform)
        
        # Changing regular spectrogram to LOG Spectrogram 
        mel_spec = self.amplitude_to_db(mel_spec)
        
        # Standarization per sample
        mean = mel_spec.mean()
        std = mel_spec.std() + 1e-8
        mel_spec = (mel_spec - mean) / std

        # Spectrogrma augmentations (only if augment=True)
        if self.augment:
            mel_spec = self.freq_masking(mel_spec)
            mel_spec = self.time_masking(mel_spec)

        return mel_spec, self.labels[index]
    


def parse_audio_directory(directory_paths) -> tuple[list[str], list[int], dict[str, int]]:
    """
    Parses a directory of .wav files, extracting file paths and mapping text command to integer labels

    Formatt of the files: {microphone}_{author}_{command}_{sample_number}.wav
    """
    # If only one folder was given lets convert it to a list
    if not isinstance(directory_paths, list):
        directory_paths = [directory_paths]

    file_paths = []
    text_commands = []
    metadata = []

    for directory_path in directory_paths:

        path_obj = Path(directory_path)

        if not path_obj.exists():
            continue

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


def add_prefix_to_files(folder_path: str, prefix:str) -> None:
    """
    Function iterates through all files in given folder and add prefix to its name
    """
    target_dir = Path(folder_path)

    # Checking if folder actually exsits
    if not target_dir.exists() or not target_dir.is_dir():
        print(f"There is no path: {folder_path} or its not a folder")
        return
    
    for file_path in target_dir.iterdir():

        # We wanna add prefix only for files so we have to check its not a directory or sth else
        if file_path.is_file():

            # We create new name 
            new_name = prefix + file_path.name

            # We build new path
            new_path = file_path.with_name(new_name)

            # we have to physically change the name of the file
            file_path.rename(new_path)
            
