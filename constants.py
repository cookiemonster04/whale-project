# ENV = "local"
# ENV = "kaggle"
# ENV = "colab"
import os
ENV = "kaggle" if os.getcwd() == "/code" else ("colab" if os.getcwd().find("shortcut") != -1 else "local")
EASY_EPOCHS = 8
# EASY_EPOCHS = 1
HARD_EPOCHS = 15
# HARD_EPOCHS = 20
EPOCHS = 10
BATCH_SIZE = 32 if ENV != "local" else 32
N_EVAL = 5
WARMUP_STEPS = 600
FINAL_STEPS = 7500
SIAMESE = True
# TRAIN_PATH = "/content/data/train"
TRAIN_PATH = "/kaggle/input/whale-dataset" if ENV == "kaggle" else (
    "./data/train" if ENV == "local" else "/content/data/train")
CKPT_PATH = "/kaggle/working/ckpt" if ENV == "kaggle" else "./ckpt"
# LOAD_FILE = "2023_Apr_01-00_04_31_stage2_2023_Mar_31-23_57_28"
# LOAD_FILE = "2023_Apr_04-15_32_09_stage2_e10_2023_Apr_04-13_09_23.state"
# LOAD_FILE = "2023_Apr_04-13_50_18_stage1_2023_Apr_04-13_09_23.state"
# LOAD_FILE = "2023_Apr_05-07_36_49_stage1_2023_Apr_05-06_14_36.state"
# LOAD_FILE = "2023_Apr_27-19_51_10_stage4_2023_Apr_27-16_17_05.state"
# LOAD_FILE = "2023_Apr_27-18_27_06_stage2_2023_Apr_27-16_17_05.state"
# LOAD_FILE = "2023_Apr_27-18_34_44_stage3_2023_Apr_27-16_17_05.state"
# LOAD_FILE = "2023_Apr_28-04_27_11_stage3_2023_Apr_28-02_20_19.state"
# LOAD_FILE = "2023_Apr_29-17_25_47_stage1_2023_Apr_29-16_50_16.state"
# LOAD_FILE = "2023_Apr_29-19_24_27_stage1_2023_Apr_29-18_50_17.state"
# LOAD_FILE = "2023_May_02-01_18_11_stage3_2023_May_01-23_16_24.state"
LOAD_FILE = "2023_May_02-06_05_42_stage1_2023_May_02-05_25_19.state"
# STAGE = 2
# STAGE = 1
STAGE = 2
# START_EPOCH = 5
START_EPOCH = 0
SAVE_STAGES = set((1, 2, 3, 4))
N_SAVE = [None, 0, 5, 0, 0]