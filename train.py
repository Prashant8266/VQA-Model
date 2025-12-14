import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
import config
from dataset import VQADataset
from model import VQAModel

def train():
    dataset = VQADataset(config.DATASET_PATH, config.IMAGES_DIR)
    dataloader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    
    model = VQAModel().to(config.DEVICE)
    optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE)
    scaler = torch.amp.GradScaler("cuda") if config.USE_MIXED_PRECISION else None

    model.train()
    
    for epoch in range(config.EPOCHS):
        total_loss = 0
        loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.EPOCHS}")
        
        for batch in loop:
            pixel_values = batch["pixel_values"].to(config.DEVICE)
            input_ids = batch["input_ids"].to(config.DEVICE)
            attention_mask = batch["attention_mask"].to(config.DEVICE)
            
            optimizer.zero_grad()
            
            if config.USE_MIXED_PRECISION:
                with torch.amp.autocast("cuda"):
                    loss = model(pixel_values, input_ids, attention_mask)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss = model(pixel_values, input_ids, attention_mask)
                loss.backward()
                optimizer.step()
                
            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

    torch.save(model.state_dict(), config.MODEL_SAVE_PATH)

if __name__ == "__main__":
    train()