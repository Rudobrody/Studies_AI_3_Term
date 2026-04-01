# %%
import torch
import random
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader

from voice_commands.config import DATA_DIR, TARGET_SAMPLE_RATE, BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS, MODELS_DIR, PROCESSED_DATA_DIR, REPORTS_DIR, PROCESSED_2ND_DATA_DIR
from voice_commands.dataset import parse_audio_directory, CommandDataset
from voice_commands.modeling.architecture import AudioCNN
from voice_commands.modeling.train import train_model
from voice_commands.plots import plot_confusion_matrix , plot_training_history, visualize_mel_spectrogram

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device computation: {device}")

# %%
print("Parsing dataset directory...")
file_paths, labels, metadata, label_mapping = parse_audio_directory(PROCESSED_DATA_DIR)
num_classes = len(label_mapping)

# Prevent crashing if the directory is empty or path is wrong
if len(file_paths) == 0:
    raise ValueError(f"No audio files found in {PROCESSED_DATA_DIR}. Please check the path.")

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

# %%
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
file_paths, labels, metadata, label_mapping = parse_audio_directory(PROCESSED_DATA_DIR)
num_classes = len(label_mapping)

# Prevent crashing if the directory is empty or path is wrong
if len(file_paths) == 0:
    raise ValueError(f"No audio files found in {PROCESSED_DATA_DIR}. Please check the path.")

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
    save_path=Path.joinpath(MODELS_DIR,"best_audio_model_micr_splitted.pth")
)

print("\nPipeline execution finished. Best model based on micr splitted saved to disk.")

# %%
print("\nGenerating visualizations..")

visualize_mel_spectrogram(train_dataset, index=0, label_mapping=label_mapping, title="reports/figures/mel_spectrogram_our_voices_micr_splitted")

# Plot the learning curves
plot_training_history(history, title="loss_acc_our_voices_micr_splitted")
    
# %%
# Plot the confusion matrix using the best saved weights
# First, load the best weights we saved during training
model.load_state_dict(torch.load(Path.joinpath(MODELS_DIR,"best_audio_model_micr_splitted.pth")))
plot_confusion_matrix(model, val_loader, device, label_mapping, title="reports/figures/confusion_matrix_our_voices_micr_splitted")

#######################################################################
######################################################################
# %%
# Experiment - using data of 2nd group

print("Parsing 2nd dataset directory...")
file_paths, labels, metadata, label_mapping = parse_audio_directory(PROCESSED_2ND_DATA_DIR)
num_classes = len(label_mapping)

# Prevent crashing if the directory is empty or path is wrong
if len(file_paths) == 0:
    raise ValueError(f"No audio files found in {PROCESSED_2ND_DATA_DIR}. Please check the path.")

print("In data of 2nd group we have only one microphone so we will use all data as training data without validate data")

train_paths, train_labels = [], []

# Iterate through all the data simultaneously
for path, label, meta in zip(file_paths, labels, metadata):

    train_paths.append(path)
    train_labels.append(label)

print(f"Training on {len(train_paths)} files")

# Why we are using zip? Because we want shuffle them
combined_train = list(zip(train_paths, train_labels))
random.shuffle(combined_train)

# Unzip them back into separate tuples
train_paths, train_labels = zip(*combined_train)

# Creating training dataset
train_dataset = CommandDataset(list(train_paths), list(train_labels), TARGET_SAMPLE_RATE)

# shuffle=True for Train allows the network to learn robustly
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)


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
    save_path=Path.joinpath(MODELS_DIR,"best_audio_model_2nd_group.pth")
)

print("\nPipeline execution finished. Best model based on micr splitted saved to disk.")

# %%
print("\nGenerating visualizations..")

visualize_mel_spectrogram(train_dataset, index=0, label_mapping=label_mapping, title="mel_spectrogram_2nd_group_voices")

# Plot the learning curves
plot_training_history(history, title="loss_acc_2nd_group_voices")
    
# %%
# Plot the confusion matrix using the best saved weights
# First, load the best weights we saved during training
model.load_state_dict(torch.load(Path.joinpath(MODELS_DIR,"best_audio_model_2nd_group.pth")))
plot_confusion_matrix(model, val_loader, device, label_mapping, title="confusion_matrix_2nd_group_voices")

# %%

#######################################################################
######################################################################
# %%
# Experiment - lets combine the data

print("Parsing combined dataset directory...")
file_paths, labels, metadata, label_mapping = parse_audio_directory([PROCESSED_DATA_DIR,PROCESSED_2ND_DATA_DIR])
num_classes = len(label_mapping)

# Prevent crashing if the directory is empty or path is wrong
if len(file_paths) == 0:
    raise ValueError(f"No audio files found in {PROCESSED_2ND_DATA_DIR}. Please check the path.")

print("In data of 2nd group we have only one microphone so we will use all data as training data without validate data")

train_paths, train_labels = [], []

# Iterate through all the data simultaneously
for path, label, meta in zip(file_paths, labels, metadata):

    train_paths.append(path)
    train_labels.append(label)

print(f"Training on {len(train_paths)} files")

# Why we are using zip? Because we want shuffle them
combined_train = list(zip(train_paths, train_labels))
random.shuffle(combined_train)

# Unzip them back into separate tuples
train_paths, train_labels = zip(*combined_train)

# Creating training dataset
train_dataset = CommandDataset(list(train_paths), list(train_labels), TARGET_SAMPLE_RATE)

# shuffle=True for Train allows the network to learn robustly
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)


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
    save_path=Path.joinpath(MODELS_DIR,"best_audio_model_combined.pth")
)

print("\nPipeline execution finished. Best model based on micr splitted saved to disk.")

# %%
print("\nGenerating visualizations..")

visualize_mel_spectrogram(train_dataset, index=0, label_mapping=label_mapping, title="mel_spectrogram_combined_voices")

# Plot the learning curves
plot_training_history(history, title="loss_acc_combined_voices")
    
# %%
# Plot the confusion matrix using the best saved weights
# First, load the best weights we saved during training
model.load_state_dict(torch.load(Path.joinpath(MODELS_DIR,"best_audio_model_combined.pth")))
plot_confusion_matrix(model, val_loader, device, label_mapping, title="confusion_matrix_combined_voices")