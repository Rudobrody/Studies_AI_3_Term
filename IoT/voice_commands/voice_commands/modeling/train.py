# from pathlib import Path

# from loguru import logger
# from tqdm import tqdm
# import typer

# from voice_commands.config import MODELS_DIR, PROCESSED_DATA_DIR

# app = typer.Typer()


# @app.command()
# def main(
#     # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
#     features_path: Path = PROCESSED_DATA_DIR / "features.csv",
#     labels_path: Path = PROCESSED_DATA_DIR / "labels.csv",
#     model_path: Path = MODELS_DIR / "model.pkl",
#     # -----------------------------------------
# ):
#     # ---- REPLACE THIS WITH YOUR OWN CODE ----
#     logger.info("Training some model...")
#     for i in tqdm(range(10), total=10):
#         if i == 5:
#             logger.info("Something happened for iteration 5.")
#     logger.success("Modeling training complete.")
#     # -----------------------------------------


# if __name__ == "__main__":
#     app()
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
from voice_commands.config import MODELS_DIR
from pathlib import Path

def train_model(
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        num_epochs: int,
        device: torch.device,
        save_path: str = Path.joinpath(MODELS_DIR,"best_audio_model.pth"),
        scheduler = None
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

        # Current Learning Rate tracking (takes LR from the classifier group)
        current_lr = optimizer.param_groups[-1]['lr']

        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"| Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% "
              f"| Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% "
              f"| LR: {current_lr:.6f}")

        if scheduler is not None:
            scheduler.step(val_acc)

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