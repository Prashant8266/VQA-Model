import torch
import torch.nn as nn
from transformers import CLIPVisionModel, GPT2LMHeadModel
import config

class VQAModel(nn.Module):
    def __init__(self):
        super(VQAModel, self).__init__()
        self.vision_encoder = CLIPVisionModel.from_pretrained(config.CLIP_MODEL_ID)
        self.text_decoder = GPT2LMHeadModel.from_pretrained(config.TEXT_MODEL_ID)
        
        self.vision_encoder.requires_grad_(False)
        
        self.visual_projection = nn.Linear(
            self.vision_encoder.config.hidden_size,
            self.text_decoder.config.n_embd
        )

    def forward(self, pixel_values, input_ids, attention_mask):
        vision_outputs = self.vision_encoder(pixel_values=pixel_values)
        image_embeds = vision_outputs.pooler_output
        
        projected_visuals = self.visual_projection(image_embeds).unsqueeze(1)
        
        inputs_embeds = self.text_decoder.transformer.wte(input_ids)
        combined_embeds = torch.cat((projected_visuals, inputs_embeds), dim=1)
        
        extended_attention_mask = torch.cat(
            (torch.ones((attention_mask.shape[0], 1), device=attention_mask.device), attention_mask),
            dim=1
        )

        outputs = self.text_decoder(
            inputs_embeds=combined_embeds,
            attention_mask=extended_attention_mask,
            labels=input_ids
        )
        
        return outputs.loss

    def generate(self, pixel_values, tokenizer, max_new_tokens=10):
        vision_outputs = self.vision_encoder(pixel_values=pixel_values)
        image_embeds = vision_outputs.pooler_output
        projected_visuals = self.visual_projection(image_embeds).unsqueeze(1)
        
        generated_ids = []
        current_embeds = projected_visuals

        for _ in range(max_new_tokens):
            outputs = self.text_decoder(inputs_embeds=current_embeds)
            next_token_logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
            generated_ids.append(next_token)
            
            next_token_embeds = self.text_decoder.transformer.wte(next_token)
            current_embeds = torch.cat((current_embeds, next_token_embeds), dim=1)
            
            if next_token.item() == tokenizer.eos_token_id:
                break
                
        return torch.cat(generated_ids, dim=1)