import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from peft import get_peft_model, LoraConfig, TaskType

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Use a CPU/GPU-compatible model
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
print(f"Loading model from Hugging Face: {model_name}")

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32 if device.type == "cpu" else None)
model.to(device)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    print("Pad token set to EOS token.")

# Configure LoRA
lora_config = LoraConfig(
    r=8,                     # Lower rank for stability
    lora_alpha=16,           # Adjust alpha
    target_modules=["q_proj", "k_proj", "v_proj"],
    lora_dropout=0.1,        # Higher dropout for regularization
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
print("LoRA configuration applied successfully.")

# Load dataset
dataset_path = "custom_knowledge.jsonl"
dataset = load_dataset('json', data_files=dataset_path, split='train')
print(f"Dataset loaded with {len(dataset)} examples.")

# Preprocess dataset
def preprocess_function(examples):
    texts = [f"{p.strip()} [SEP] {c.strip()}" for p, c in zip(examples["prompt"], examples["completion"])]
    tokenized = tokenizer(
        texts,
        truncation=True,
        max_length=512,
        padding="max_length",
        return_tensors="pt",
    )
    tokenized["labels"] = tokenized["input_ids"].clone()
    return tokenized

tokenized_dataset = dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=dataset.column_names,
    desc="Tokenizing dataset",
)

# Data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Training arguments
training_args = TrainingArguments(
    output_dir="./fine_tuned_model",
    overwrite_output_dir=True,
    num_train_epochs=3,                  # Reduced epochs to prevent overfitting
    per_device_train_batch_size=1 if device.type == "cpu" else 2,  # Adjust for CPU/GPU
    gradient_accumulation_steps=4,
    save_steps=200,
    save_total_limit=2,
    logging_steps=20,
    learning_rate=2e-5,                  # Lower learning rate
    warmup_steps=100,
    fp16=torch.cuda.is_available(),      # Enable FP16 only on GPU
    max_grad_norm=1.0,                   # Add gradient clipping
    optim="adamw_torch",
    evaluation_strategy="no",
    report_to="none",
)

# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

# Check for nan/inf during training
def check_params_for_nan_inf(model):
    for name, param in model.named_parameters():
        if torch.isnan(param).any() or torch.isinf(param).any():
            print(f"Parameter {name} contains nan or inf")
            raise ValueError("Training stopped due to nan/inf in parameters")

trainer.add_callback(lambda trainer: check_params_for_nan_inf(trainer.model))

# Start fine-tuning
print("Starting fine-tuning...")
trainer.train()

# Save the model
model.save_pretrained("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")
model = model.merge_and_unload()
model.save_pretrained("./fine_tuned_model_merged")
tokenizer.save_pretrained("./fine_tuned_model_merged")
print("Fine-tuning completed. Model and tokenizer saved to ./fine_tuned_model_merged")