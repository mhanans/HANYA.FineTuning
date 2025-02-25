import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from peft import get_peft_model, LoraConfig, TaskType

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model and tokenizer
model_name = "deepseek-ai/DeepSeek-R1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model.to(device)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Configure LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, lora_config)

# Load dataset
dataset = load_dataset('json', data_files="custom_knowledge.jsonl", split='train')

# Preprocess dataset
def preprocess_function(examples):
    texts = [f"{p.strip()} [SEP] {c.strip()}" for p, c in zip(examples["prompt"], examples["completion"])]
    tokenized = tokenizer(texts, truncation=True, max_length=512, padding="max_length", return_tensors="pt")
    tokenized["labels"] = tokenized["input_ids"].clone()
    return tokenized

tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=dataset.column_names)

# Data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Training arguments
training_args = TrainingArguments(
    output_dir="./fine_tuned_model",
    overwrite_output_dir=True,
    num_train_epochs=5,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    save_steps=200,
    save_total_limit=2,
    logging_steps=20,
    learning_rate=5e-5,
    warmup_steps=100,
    fp16=torch.cuda.is_available(),
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

# Start fine-tuning
trainer.train()

# Save the model
model.save_pretrained("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")
model = model.merge_and_unload()
model.save_pretrained("./fine_tuned_model_merged")
tokenizer.save_pretrained("./fine_tuned_model_merged")