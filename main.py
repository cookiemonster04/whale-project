import torch
import constants
from data.datasets import StartingDataset, SiameseDataset, HardSiameseDataset, GlobalHardSiameseDataset
from networks.StartingNetwork import StartingNetwork
from networks.SiameseNetwork import SiameseNetwork
from train_functions.starting_train import starting_train
from train_functions.custom_train import siamese_train
from datetime import datetime
import os
import logging

def main():
    # Get command line arguments
    hyperparameters = {"epochs": constants.EPOCHS, "start_epoch": constants.START_EPOCH,
                       "easy_epochs": constants.EASY_EPOCHS, "hard_epochs": constants.HARD_EPOCHS,
                       "warmup": constants.WARMUP_STEPS, "final": constants.FINAL_STEPS,
                       "batch_size": constants.BATCH_SIZE,
                       "stage": constants.STAGE if hasattr(constants, "STAGE") else 0,
                       "save_path": constants.CKPT_PATH if hasattr(constants, "CKPT_PATH") else "",
                       "save_stages": constants.SAVE_STAGES if hasattr(constants, "SAVE_STAGES") else set(),
                       "save_int": constants.N_SAVE if hasattr(constants, "N_SAVE") else [None, 0, 0, 0, 0],
                       "start_time": datetime.now().strftime("%Y_%b_%d-%H_%M_%S")}
    # Set up logging
    logging.basicConfig(filename=os.path.join(hyperparameters["save_path"], f"{hyperparameters['start_time']}.log"),
                        filemode='a',
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.DEBUG)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if constants.ENV != "local" else torch.device('cpu')
    print("Epochs:", constants.EPOCHS)
    print("Batch size:", constants.BATCH_SIZE)

    model = SiameseNetwork(constants.STAGE, True) if constants.SIAMESE else StartingNetwork()
    if hasattr(constants, "LOAD_FILE") and constants.LOAD_FILE != "":
        model.load_state_dict(torch.load(os.path.join(constants.CKPT_PATH, constants.LOAD_FILE), map_location=device), strict=False)
    if constants.SIAMESE:
        siamese_train(
            encode_easy=SiameseDataset() if constants.STAGE <= 1 else None,
            encode_hard=HardSiameseDataset(device) if constants.STAGE <= 2 else None,
            pred_train=StartingDataset(True), pred_val=StartingDataset(False),
            model=model,
            hyperparameters=hyperparameters,
            n_eval=constants.N_EVAL,
            device=device
        )
    else:
        starting_train(
            train_dataset=StartingDataset(True),
            val_dataset=StartingDataset(False),
            model=model,
            hyperparameters=hyperparameters,
            n_eval=constants.N_EVAL,
            device=device
        )

if __name__ == "__main__":
    main()
