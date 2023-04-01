import torch
import constants
from data.datasets import StartingDataset, SiameseDataset
from networks.StartingNetwork import StartingNetwork
from networks.SiameseNetwork import SiameseNetwork
from train_functions.starting_train import starting_train
from train_functions.custom_train import siamese_train
from datetime import datetime
import os

def main():
    # Get command line arguments
    hyperparameters = {"epochs": constants.EPOCHS,
                       "batch_size": constants.BATCH_SIZE,
                       "stage": constants.STAGE if hasattr(constants, "STAGE") else 0,
                       "save_path": constants.CKPT_PATH if hasattr(constants, "CKPT_PATH") else "",
                       "save_stages": constants.SAVE_STAGES if hasattr(constants, "SAVE_STAGES") else set(),
                       "start_time": datetime.now().strftime("%Y_%b_%d-%H_%M_%S")}

    # TODO: Add GPU support. This line of code might be helpful.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Epochs:", constants.EPOCHS)
    print("Batch size:", constants.BATCH_SIZE)

    # Initalize dataset and model. Then train the model!
    model = SiameseNetwork(constants.STAGE) if constants.SIAMESE else StartingNetwork()
    if hasattr(constants, "LOAD_FILE") and constants.LOAD_FILE != "":
        model.load_state_dict(torch.load(os.path.join(constants.CKPT_PATH, constants.LOAD_FILE)))
    if constants.SIAMESE:
        siamese_train(
            encode_train=SiameseDataset(),
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
