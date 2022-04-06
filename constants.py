"""FILES PATH"""
PKL_TRAIN_PATH = "/opt/ml/code/pickled_data/preprocessed_train_data"
PKL_TEST_PATH = "/opt/ml/code/pickled_data/preprocessed_test" 
LOG_DIR = "/opt/ml/code/logs"
SAVE_DIR = "/opt/ml/code/results"
BEST_MODEL_DIR = "/opt/ml/code/best_model"

"""DATASET"""
TRAIN_DIR = "/opt/ml/dataset/train/train.csv"
TEST_DIR = "/opt/ml/dataset/test/test_data.csv"
GENERATE_DATA = "/opt/ml/dataset/generate/generated.csv"

"""OPTIONS"""
ONLY_ORIGINAL = 0
ONLY_GENERATED = 1
CONCAT_ALL = 2

"""WANDB"""
WANDB_ENT = "boostcamp_nlp_04"