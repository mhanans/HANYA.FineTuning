import torch
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("./fine_tuned_model_merged")
for name, param in model.named_parameters():
    if torch.isnan(param).any() or torch.isinf(param).any():
        print(f"Parameter {name} contains nan or inf")