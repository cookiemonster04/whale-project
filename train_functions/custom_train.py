import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()
from datetime import datetime
import os

def siamese_train(encode_train, pred_train, pred_val, model, hyperparameters, n_eval, device):
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
    start_stage, saves = hyperparameters["stage"], hyperparameters["save_stages"]
    start_time = hyperparameters["start_time"]
    save_path = hyperparameters["save_path"]

    # Initialize dataloaders
    etl = torch.utils.data.DataLoader(
        encode_train, batch_size=batch_size, shuffle=True
    )
    ptl = torch.utils.data.DataLoader(
        pred_train, batch_size=batch_size, shuffle=True
    )
    pvl = torch.utils.data.DataLoader(
        pred_val, batch_size=batch_size
    )

    # Initalize optimizer (for gradient descent) and loss function
    if start_stage <= 1:
        optimizer = optim.Adam(model.parameters())
        loss_fn = model.loss
        model = model.to(device)
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1} of {epochs}")
            # Loop over each batch in the dataset
            pbar = tqdm(enumerate(etl), total=len(etl))
            for step, (x1, x2, x3) in pbar:
                # TODO: Backpropagation and gradient descent
                x1 = x1.to(device); v1 = model(x1)
                x2 = x2.to(device); v2 = model(x2)
                x3 = x3.to(device); v3 = model(x3)
                loss = loss_fn(v1, v2, v3)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                pbar.set_description(f"iter {step} Train encode loss: {loss}")
                writer.add_scalar("Encode Loss", loss, step)
                writer.flush()
            print()
        if 1 in saves:
            dt_string = datetime.now().strftime("%Y_%b_%d-%H_%M_%S")
            torch.save(model.state_dict(), os.path.join(save_path, f'{dt_string}_stage1_{start_time}.state'))

    loss_fn = nn.CrossEntropyLoss()
    if start_stage <= 2:
        model.init_classify_params()
        model = model.to(device)
        with torch.no_grad():
            for step, (x, y) in tqdm(enumerate(ptl)):
                x = x.to(device)
                model.init_classify(x, y)
            for step, (x, y) in tqdm(enumerate(pvl)):
                x = x.to(device)
                model.init_classify(x, y)
            model.norm_embed()
        if 2 in saves:
            dt_string = datetime.now().strftime("%Y_%b_%d-%H_%M_%S")
            torch.save(model.state_dict(), os.path.join(save_path, f'{dt_string}_stage2_{start_time}'))
    if start_stage >= 3:
        model = model.to(device)
    if start_stage <= 3:
        optimizer = optim.Adam(model.parameters())
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1} of {epochs}")
            # Loop over each batch in the dataset
            pbar = tqdm(enumerate(ptl), total=len(ptl))
            for step, (x, y) in pbar:
                # TODO: Backpropagation and gradient descent
                x = x.to(device)
                y = y.to(device)
                logits = model.classify(x)
                torch.set_printoptions(profile="full")
                # print(logits)
                # print(torch.argmax(logits, dim=-1))
                # print(y)
                loss = loss_fn(logits, y)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                pbar.set_description(f"iter {step} Train predict loss: {loss}")
                # Periodically evaluate our model + log to Tensorboard
            if (epoch+1) % n_eval == 0 or (epoch+1) == epochs:# (epoch+1) == len(ptl):
                # TODO:
                # Compute training loss and accuracy.
                # Log the results to Tensorboard.
                writer.add_scalar("Predict Loss", loss, step+1)
                acc = compute_accuracy(logits, y)
                writer.add_scalar("Predict Accuracy", acc, step+1)
                print(f"Predict train acc: {acc}")
                # TODO:
                # Compute validation loss and accuracy.
                # Log the results to Tensorboard.
                # Don't forget to turn off gradient calculations!
                dev_loss, dev_acc = pred_evaluate(pvl, model, loss_fn, device)
                writer.add_scalar("Predict Dev Loss", dev_loss, step+1)
                writer.add_scalar("Predict Dev Acc", dev_acc, step+1)
                print(f"Predict dev acc: {dev_acc}")
                writer.flush()
            print()
        if 3 in saves:
            dt_string = datetime.now().strftime("%Y_%b_%d-%H_%M_%S")
            torch.save(model.state_dict(), os.path.join(save_path, f'{dt_string}_stage3_{start_time}'))

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

def pred_evaluate(val_loader, model, loss_fn, device):
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
            logits = model.classify(x)
            ans = torch.argmax(logits, dim=-1)
            np.set_printoptions(threshold=1000000)
            # print(ans.cpu().numpy())
            loss = loss_fn(logits, y)
            batch_sz = x.shape[0]
            total_loss += loss*batch_sz
            sz += batch_sz
            total_correct += (ans == y).sum().item()
    return total_loss / sz, total_correct / sz
