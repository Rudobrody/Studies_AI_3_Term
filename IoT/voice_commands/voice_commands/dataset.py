# from pathlib import Path

# from loguru import logger
# from tqdm import tqdm
# import typer

# from voice_commands.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

# app = typer.Typer()


# @app.command()
# def main(
#     # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
#     input_path: Path = RAW_DATA_DIR / "dataset.csv",
#     output_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
#     # ----------------------------------------------
# ):
#     # ---- REPLACE THIS WITH YOUR OWN CODE ----
#     logger.info("Processing dataset...")
#     for i in tqdm(range(10), total=10):
#         if i == 5:
#             logger.info("Something happened for iteration 5.")
#     logger.success("Processing dataset complete.")
#     # -----------------------------------------


# if __name__ == "__main__":
#     app()
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
    """
    def __init__(self, file_paths: list[str], labels: list[int], target_sample_rate: int = 16000):
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
            
