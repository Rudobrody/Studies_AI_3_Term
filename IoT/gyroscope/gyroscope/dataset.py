import pandas as pd 
import numpy as np
import glob
import os
from pathlib import Path

from gyroscope.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

def convert_dataset_to_arrays(raw_data_dir, output_dir):
    """
    Read all good/wrong .txt windows and compiles them into .npy arrays
    """
    # Find all subfolders
    subfolders = [f.path for f in os.scandir(raw_data_dir) if f.is_dir()]
    
    if not subfolders:
        print(f"No subfolders were found in {raw_data_dir}")
    else:
        print(f"{len(subfolders)} directories were found in {raw_data_dir}")
    
    for folder_path in subfolders:
        folder_name = os.path.basename(folder_path)
        print(f"Processing folder: {folder_name}..")

        # Grab all text files in the directory
        files_in_folder = glob.glob(os.path.join(folder_path, '**', "*.txt"), recursive=True)

        if not files_in_folder:
            print("There is no .txt files, check path..")
            return None, None
    
        signals = []
        labels = []

        print(f"Found {len(files_in_folder)} files. Processing..")

        for file_path in files_in_folder:
            # Load the text file
            df = pd.read_csv(file_path, header=None)

            # Convert to a numpy and add to our list
            signals.append(df.values)

            # Extract the label from the filename
            filename = os.path.basename(file_path).lower()

            if filename.startswith('good'):
                labels.append(0) # Class 0: normal
            elif filename.startswith('wrong'):
                labels.append(1) # Class 1: anomaly
            else:
                print(f"Warning, could not classify file {filename}")

        # Convert lists into numpy arrays
        X = np.array(signals)
        y = np.array(labels)

        print(f"Signals shape (x): {X.shape}")
        print(f"Labels shape (y): {y.shape}")

        # Save to processed folder 
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        np.save(os.path.join(output_dir, f'X_{folder_name}.npy'), X)
        np.save(os.path.join(output_dir, f'y_{folder_name}.npy'), y)

        print(f"Saved succesfully to: {output_dir}")


convert_dataset_to_arrays(raw_data_dir=RAW_DATA_DIR, output_dir=PROCESSED_DATA_DIR)