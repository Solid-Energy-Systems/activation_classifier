import os
import torch

def train_model(
    model,
    train_loader,
    test_loader,
    criterion,
    optimizer,
    num_epochs,
    device,
    checkpoint_dir,
    verbose=True
):
    """
    Trains the neural network model, evaluates it on the validation dataset, 
    and saves the best and last checkpoints during training.

    Args:
        model (torch.nn.Module): The neural network model to train.
        train_loader (torch.utils.data.DataLoader): DataLoader for the training data.
        test_loader (torch.utils.data.DataLoader): DataLoader for the validation data.
        criterion (torch.nn.Module): Loss function to minimize (e.g., `torch.nn.BCELoss`).
        optimizer (torch.optim.Optimizer): Optimization algorithm (e.g., `torch.optim.Adam`).
        num_epochs (int): Number of epochs to train the model.
        device (str): Device to train on ('cuda' or 'cpu').
        checkpoint_dir (str): Directory to save checkpoints.
        verbose (bool): If True, prints training and validation progress.

    Returns:
        tuple: (train_losses, val_accuracies)
            - train_losses (list of floats): Training losses over epochs.
            - val_accuracies (list of floats): Validation accuracies over epochs.

    Saves:
        - `best_model.pth`: The model with the highest validation accuracy.
        - `last_model.pth`: The model after the final epoch.
    """
    best_accuracy = 0.0
    os.makedirs(checkpoint_dir, exist_ok=True)

    train_losses = []
    val_accuracies = []

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        total_samples = 0
        for batch in train_loader:
            activations = batch['activation'].to(device)  # Shape: [batch_size, num_layers, sequence_length, hidden_size]
            labels = batch['label'].to(device).unsqueeze(1)  # Shape: [batch_size, 1]

            optimizer.zero_grad()
            outputs = model(activations)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * activations.size(0)
            total_samples += activations.size(0)

        avg_loss = total_loss / total_samples
        train_losses.append(avg_loss)
        if verbose:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_loss:.4f}")

        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in test_loader:
                activations = batch['activation'].to(device)  # Shape: [batch_size, num_layers, sequence_length, hidden_size]
                labels = batch['label'].to(device).unsqueeze(1)

                outputs = model(activations)
                predicted = (outputs > 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total * 100
        val_accuracies.append(accuracy)
        if verbose:
            print(f"Validation Accuracy: {accuracy:.2f}%\n")

        # Save the best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_path = os.path.join(checkpoint_dir, 'best_model.pth')
            torch.save(model.state_dict(), best_model_path)
            if verbose:
                print(f"Best model saved with accuracy: {best_accuracy:.2f}%")

    # Save the last model
    last_model_path = os.path.join(checkpoint_dir, 'last_model.pth')
    torch.save(model.state_dict(), last_model_path)
    if verbose:
        print(f"Last model saved at epoch {num_epochs}")

    return train_losses, val_accuracies