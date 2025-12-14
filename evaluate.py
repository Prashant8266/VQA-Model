import torch
from torch.utils.data import DataLoader
from bert_score import score
from tqdm import tqdm
import config
from dataset import VQADataset
from model import VQAModel

def evaluate():
    dataset = VQADataset(config.DATASET_PATH, config.IMAGES_DIR)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    model = VQAModel().to(config.DEVICE)
    model.load_state_dict(torch.load(config.MODEL_SAVE_PATH))
    model.eval()
    
    tokenizer = dataset.tokenizer
    references = []
    candidates = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            pixel_values = batch["pixel_values"].to(config.DEVICE)
            true_answer = batch["answers"][0]
            
            generated_ids = model.generate(pixel_values, tokenizer)
            pred_answer = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            
            references.append(true_answer)
            candidates.append(pred_answer)

    P, R, F1 = score(candidates, references, lang="en", verbose=True)
    
    print(f"BERTScore Precision: {P.mean():.4f}")
    print(f"BERTScore Recall: {R.mean():.4f}")
    print(f"BERTScore F1: {F1.mean():.4f}")

if __name__ == "__main__":
    evaluate()