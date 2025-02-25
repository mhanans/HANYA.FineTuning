from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
import torch
from peft import get_peft_model, LoraConfig, TaskType


# Replace with your model's path or identifier.
model_name_or_path = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

# Load the tokenizer and model.
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = AutoModelForCausalLM.from_pretrained(model_name_or_path)

# Configure LoRA (tweak parameters based on your model and dataset)
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],  # Adjust according to your model architecture
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

# Wrap your model with the PEFT model.
model = get_peft_model(model, lora_config)
print("LoRA configuration applied.")



# Load the custom fine-tuning dataset (JSONL format).
dataset = load_dataset('json', data_files='custom_knowledge.jsonl', split='train')

# Preprocessing: Concatenate prompt and completion.
def preprocess_function(examples):
    # Use zip to iterate over the paired values from the lists
    texts = [p + " " + c for p, c in zip(examples["prompt"], examples["completion"])]
    return tokenizer(texts, truncation=True, max_length=256)


tokenized_dataset = dataset.map(preprocess_function, batched=True)

# Create a data collator for causal language modeling (no masked LM).
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Define training arguments.
training_args = TrainingArguments(
    output_dir="./fine_tuned_model",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=1,  # Lower batch size to reduce memory usage
    gradient_accumulation_steps=2,    # Optionally, use gradient accumulation to simulate a larger batch size
    save_steps=100,
    save_total_limit=2,
    logging_steps=10,
    learning_rate=2e-5,
    fp16=torch.cuda.is_available(),  # Mixed precision may help on GPU, but if using CPU, you might disable it.
)


# Initialize the Trainer.
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

# Start fine-tuning.
trainer.train()

# Save the fine-tuned model.
trainer.save_model("./fine_tuned_model")
print("Fine-tuning complete. Model saved to ./fine_tuned_model")
