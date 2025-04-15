import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from peft import get_peft_model, LoraConfig, TaskType

# Paksa penggunaan CPU-only
device = torch.device("cpu")

# Load model dan tokenizer dengan attn_implementation="eager"
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    attn_implementation="eager"  # Gunakan eager untuk stabilitas
)
model.to(device)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Konfigurasi LoRA untuk efisiensi
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, lora_config)

# Load dataset
dataset = load_dataset('json', data_files="custom_knowledge.jsonl", split='train')

# Preprocessing dataset
def preprocess_function(examples):
    texts = [f"{p.strip()} [SEP] {c.strip()}" for p, c in zip(examples["prompt"], examples["completion"])]
    tokenized = tokenizer(texts, truncation=True, max_length=256, padding="max_length", return_tensors="pt")
    tokenized["labels"] = tokenized["input_ids"].clone()
    return tokenized

tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=dataset.column_names)

# Data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Training arguments yang dioptimalkan untuk CPU
training_args = TrainingArguments(
    output_dir="./fine_tuned_model",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    save_steps=100,
    save_total_limit=2,
    logging_steps=20,
    learning_rate=5e-5,
    warmup_steps=50,
    fp16=False,
    optim="adamw_torch",
    eval_strategy="no",
    report_to="none",
)

# Inisialisasi trainer dengan label_names
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
    label_names=["labels"],  # Tentukan label_names secara eksplisit
)

# Mulai fine-tuning
trainer.train()

# Simpan model
model.save_pretrained("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")
# Gabungkan adapter LoRA dan simpan versi merged
model = model.merge_and_unload()
model.save_pretrained("./fine_tuned_model_merged")
tokenizer.save_pretrained("./fine_tuned_model_merged")