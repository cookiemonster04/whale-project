ENV = "local"
# ENV = "kaggle"
# ENV = "colab"
EASY_EPOCHS = 5
# EASY_EPOCHS = 1
HARD_EPOCHS = 10
EPOCHS = 10
BATCH_SIZE = 32 if ENV == "kaggle" else 32
N_EVAL = 5
WARMUP_STEPS = 500
FINAL_STEPS = 5000
SIAMESE = True
# TRAIN_PATH = "/content/data/train"
TRAIN_PATH = "/kaggle/input/whale-dataset" if ENV == "kaggle" else (
    "./data/train" if ENV == "local" else "/content/data/train")
CKPT_PATH = "/kaggle/working/ckpt" if ENV == "kaggle" else "./ckpt"
# LOAD_FILE = "2023_Apr_01-00_04_31_stage2_2023_Mar_31-23_57_28"
# LOAD_FILE = "2023_Apr_04-15_32_09_stage2_e10_2023_Apr_04-13_09_23.state"
# LOAD_FILE = "2023_Apr_04-13_50_18_stage1_2023_Apr_04-13_09_23.state"
# LOAD_FILE = "2023_Apr_05-07_36_49_stage1_2023_Apr_05-06_14_36.state"
# STAGE = 2
STAGE = 1
# START_EPOCH = 5
START_EPOCH = 0
SAVE_STAGES = set((1, 2, 3, 4))
N_SAVE = [None, 0, 5, 0, 0]