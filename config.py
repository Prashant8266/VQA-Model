import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 5e-5
MAX_LENGTH = 32
IMAGE_SIZE = 224
CLIP_MODEL_ID = "openai/clip-vit-base-patch32"
TEXT_MODEL_ID = "distilgpt2"
DATASET_PATH = "data/vqa_dataset.csv"
IMAGES_DIR = "data/images"
MODEL_SAVE_PATH = "vqa_model.pth"
USE_MIXED_PRECISION = True