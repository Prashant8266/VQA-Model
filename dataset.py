import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
from transformers import CLIPImageProcessor, GPT2Tokenizer
import config
from utils import clean_text, format_prompt

class VQADataset(Dataset):
    def __init__(self, csv_file, img_dir, split="train"):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.processor = CLIPImageProcessor.from_pretrained(config.CLIP_MODEL_ID)
        self.tokenizer = GPT2Tokenizer.from_pretrained(config.TEXT_MODEL_ID)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_path = os.path.join(self.img_dir, row["image_id"])
        image = Image.open(image_path).convert("RGB")
        
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.squeeze(0)
        
        question = clean_text(row["question"])
        answer = clean_text(row["answer"])
        
        text_input = format_prompt(question, answer)
        
        encoding = self.tokenizer(
            text_input,
            truncation=True,
            padding="max_length",
            max_length=config.MAX_LENGTH,
            return_tensors="pt"
        )

        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "answers": answer,
            "questions": question
        }