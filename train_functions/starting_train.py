import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

def starting_train(train_dataset, val_dataset, model, hyperparameters, n_eval, device):
    """
    Trains and evaluates a model.

    Args:
        train_dataset:   PyTorch dataset containing training data.
        val_dataset:     PyTorch dataset containing validation data.
        model:           PyTorch model to be trained.
        hyperparameters: Dictionary containing hyperparameters.
        n_eval:          Interval at which we evaluate our model.
    """

    # Get keyword arguments
    batch_size, epochs = hyperparameters["batch_size"], hyperparameters["epochs"]

    # Initialize dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=True
    )

    # Initalize optimizer (for gradient descent) and loss function
    optimizer = optim.Adam(model.parameters())
    loss_fn = nn.CrossEntropyLoss()
    
    model.to(device)
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1} of {epochs}")
        # Loop over each batch in the dataset
        for step, (x, y) in tqdm(enumerate(train_loader)):
            # TODO: Backpropagation and gradient descent
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()
            # Periodically evaluate our model + log to Tensorboard
            if step % n_eval == 0:
                # TODO:
                # Compute training loss and accuracy.
                # Log the results to Tensorboard.
                writer.add_scalar("Loss", loss, step)
                acc = compute_accuracy(logits, y)
                writer.add_scalar("Accuracy", acc, step)
                # TODO:
                # Compute validation loss and accuracy.
                # Log the results to Tensorboard.
                # Don't forget to turn off gradient calculations!
                dev_loss, dev_acc = evaluate(val_loader, model, loss_fn, device)
                writer.add_scalar("Dev Loss", dev_loss, step)
                writer.add_scalar("Dev Acc", dev_acc, step)
                writer.flush()
        print()


def compute_accuracy(outputs, labels):
    """
    Computes the accuracy of a model's predictions.

    Example input:
        outputs: [0.7, 0.9, 0.3, 0.2]
        labels:  [1, 1, 0, 1]

    Example output:
        0.75
    """

    n_correct = (torch.argmax(outputs, dim=-1) == labels).sum().item()
    n_total = len(outputs)
    return n_correct / n_total


def evaluate(val_loader, model, loss_fn, device):
    """
    Computes the loss and accuracy of a model on the validation dataset.
    
    TODO!
    """
    sz = 0
    total_loss = 0.
    total_correct = 0.
    with torch.no_grad():
        for (x, y) in tqdm(val_loader):
            # TODO: Backpropagation and gradient descent
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)
            batch_sz = x.shape[0]
            total_loss += loss*batch_sz
            sz += batch_sz
            total_correct += (torch.argmax(logits) == y).sum().item()
    return total_loss / sz, total_correct / sz
