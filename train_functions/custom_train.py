import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()
from datetime import datetime
import os
import math
# import gc
# from pytorch_memlab import MemReporter

def siamese_train(encode_easy, encode_hard, pred_train, pred_val, model, hyperparameters, n_eval, device):
    """
    Trains and evaluates a model.

    Args:
        train_dataset:   PyTorch dataset containing training data.
        val_dataset:     PyTorch dataset containing validation data.
        model:           PyTorch model to be trained.
        hyperparameters: Dictionary containing hyperparameters.
        n_eval:          Interval at which we evaluate our model.
    """

    start_epoch = hyperparameters["start_epoch"] if 'start_epoch' in hyperparameters else None
    easy_epochs, hard_epochs = hyperparameters["easy_epochs"], hyperparameters["hard_epochs"]
    warmup_steps, final_steps = hyperparameters["warmup"], hyperparameters["final"]
    batch_size, epochs = hyperparameters["batch_size"], hyperparameters["epochs"]
    start_stage, saves = hyperparameters["stage"], hyperparameters["save_stages"]
    start_time = hyperparameters["start_time"]
    save_path = hyperparameters["save_path"]
    save_int = hyperparameters["save_int"]

    if start_stage == 1:
        easy_epochs -= start_epoch
    elif start_stage == 2:
        hard_epochs -= start_epoch
    elif start_stage == 3:
        assert(start_epoch == None or start_epoch == 0)
    elif start_stage == 4:
        epochs -= start_epoch
    else:
        assert(start_epoch == None)
    eel = torch.utils.data.DataLoader(
        encode_easy, batch_size=batch_size, shuffle=True
    ) if encode_easy is not None else None
    ehl = torch.utils.data.DataLoader(
        encode_hard, batch_size=batch_size, shuffle=False if hasattr(encode_hard, "diff_ordered") and encode_hard.diff_ordered else True
    ) if encode_hard is not None else None
    ptl = torch.utils.data.DataLoader(
        pred_train, batch_size=batch_size, shuffle=True
    )
    pvl = torch.utils.data.DataLoader(
        pred_val, batch_size=batch_size
    )
    embed_lr = 3e-3
    optimizer = optim.Adam(model.parameters(), lr=embed_lr)
    # optimizer = optim.Adam(model.fc.parameters() if model.pretrain else model.parameters(), lr=embed_lr)
    loss_fn = model.loss
    model = model.to(device)
    
    def update_lr():
        return
        if model.steps.item() < warmup_steps:
            lr_mult = model.steps.item() / warmup_steps
        else:
            progress = (model.steps.item() - warmup_steps) / max(1, final_steps - warmup_steps)
            lr_mult = max(0.1, 0.5*(1.0+math.cos(math.pi * progress)))
        cur_lr = embed_lr * lr_mult
        for g in optimizer.param_groups:
            g['lr'] = cur_lr

    if start_stage <= 1:
        for epoch in range(easy_epochs):
            print(f"Epoch {epoch + 1} of {easy_epochs}")
            pbar = tqdm(enumerate(eel), total=len(eel))
            for step, (x1, x2, x3) in pbar:
                update_lr()
                x1 = x1.to(device); v1 = model(x1)
                x2 = x2.to(device); v2 = model(x2)
                x3 = x3.to(device); v3 = model(x3)
                loss = loss_fn(v1, v2, v3)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                model.steps += 1
                pbar.set_description(f"iter {step} Easy encode loss: {loss}")
                writer.add_scalar("Encode Loss", loss, step)
                writer.flush()
            print()
        if 1 in saves:
            dt_string = datetime.now().strftime("%Y_%b_%d-%H_%M_%S")
            save_name = f'{dt_string}_stage1_{start_time}.state'
            torch.save(model.state_dict(), os.path.join(save_path, save_name))
            print(f"Saved: {save_name}")
    if start_stage <= 2:
        encode_hard.gen_matrix(model)
        for epoch in range(hard_epochs):
            print(f"Epoch {epoch + 1} of {hard_epochs}")
            pbar = tqdm(enumerate(ehl), total=len(ehl))
            for step, (x1, x2, x3, i1, i2, i3) in pbar:
                update_lr()
                if hasattr(encode_hard, "diff_ordered"):
                    x1 = x1.to(device); v1 = model(x1)
                    x2 = x2.to(device); v2 = model(x2)
                    x3 = x3.to(device); v3 = model(x3)
                else:
                    x1 = x1.to(device); v1 = model(x1); encode_hard.embed_matrix[i1] = v1.detach()
                    x2 = x2.to(device); v2 = model(x2); encode_hard.embed_matrix[i2] = v2.detach()
                    x3 = x3.to(device); v3 = model(x3); encode_hard.embed_matrix[i3] = v3.detach()
                loss = loss_fn(v1, v2, v3)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                model.steps += 1
                pbar.set_description(f"iter {step} Hard encode loss: {loss}")
                writer.add_scalar("Encode Loss", loss, step)
                writer.flush()
                encode_hard.regen_handler(model, loss)
            print()
            if 2 in saves and save_int[2] != 0 and (epoch+1) % save_int[2] == 0:
                dt_string = datetime.now().strftime("%Y_%b_%d-%H_%M_%S")
                save_name = f'{dt_string}_stage2_e{epoch+1}_{start_time}.state'
                torch.save(model.state_dict(), os.path.join(save_path, save_name))
                print(f"Saved: {save_name}")
        if 2 in saves:
            dt_string = datetime.now().strftime("%Y_%b_%d-%H_%M_%S")
            save_name = f'{dt_string}_stage2_{start_time}.state'
            torch.save(model.state_dict(), os.path.join(save_path, save_name))
            print(f"Saved: {save_name}")
    loss_fn = nn.CrossEntropyLoss()
    if start_stage <= 3:
        model.init_classify_params()
        model = model.to(device)
        with torch.no_grad():
            for step, (x, y) in tqdm(enumerate(ptl), total=len(ptl)):
                x = x.to(device)
                model.init_classify(x, y)
            print()
            for step, (x, y) in tqdm(enumerate(pvl), total=len(pvl)):
                x = x.to(device)
                model.init_classify(x, y)
            print()
            model.norm_embed()
        if 3 in saves:
            dt_string = datetime.now().strftime("%Y_%b_%d-%H_%M_%S")
            save_name = f'{dt_string}_stage3_{start_time}.state'
            torch.save(model.state_dict(), os.path.join(save_path, save_name))
            print(f"Saved: {save_name}")
    if start_stage >= 4:
        model = model.to(device)
        with torch.no_grad():
            model.eval()
            for step, (x, y) in tqdm(enumerate(ptl), total=len(ptl)):
                x = x.to(device)
                model.test_classify(x, y)
            print()
            for step, (x, y) in tqdm(enumerate(pvl), total=len(pvl)):
                x = x.to(device)
                model.test_classify(x, y)
            print()
        return
    if start_stage <= 4:
        optimizer = optim.Adam(model.parameters())
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1} of {epochs}")
            pbar = tqdm(enumerate(ptl), total=len(ptl))
            for step, (x, y) in pbar:
                x = x.to(device)
                print(f"Expected: {y}")
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
            print()
            if (epoch+1) % n_eval == 0 or (epoch+1) == epochs:# (epoch+1) == len(ptl):
                writer.add_scalar("Predict Loss", loss, step+1)
                acc = compute_accuracy(logits, y)
                writer.add_scalar("Predict Accuracy", acc, step+1)
                print(f"Predict train acc: {acc}")
                model.eval()
                dev_loss, dev_acc = pred_evaluate(pvl, model, loss_fn, device)
                model.train()
                writer.add_scalar("Predict Dev Loss", dev_loss, step+1)
                writer.add_scalar("Predict Dev Acc", dev_acc, step+1)
                print(f"Predict dev acc: {dev_acc}")
                writer.flush()
        if 4 in saves:
            dt_string = datetime.now().strftime("%Y_%b_%d-%H_%M_%S")
            save_name = f'{dt_string}_stage4_{start_time}.state'
            torch.save(model.state_dict(), os.path.join(save_path, save_name))
            print(f"Saved: {save_name}")

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
        print()
    return total_loss / sz, total_correct / sz
