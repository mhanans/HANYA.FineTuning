import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from transformers.trainer_callback import TrainerCallback

# Disable Torch Dynamo globally to avoid potential issues (set as environment variable or in code)
# export TORCH_DYNAMO_DISABLE=1  # Uncomment and set in terminal before running, or use below
torch._dynamo.config.suppress_errors = True

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Use a CPU/GPU-compatible model
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
print(f"Loading model from Hugging Face: {model_name}")

# Load tokenizer and model
try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32 if device.type == "cpu" else None)
    model.to(device)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("Pad token set to EOS token.")
except Exception as e:
    print(f"Error loading model or tokenizer: {str(e)}")
    raise

# Load dataset
dataset_path = "custom_knowledge.jsonl"
try:
    dataset = load_dataset('json', data_files=dataset_path, split='train')
    print(f"Dataset loaded with {len(dataset)} examples.")
except Exception as e:
    print(f"Error loading dataset: {str(e)}")
    raise

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

try:
    tokenized_dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing dataset",
    )
except Exception as e:
    print(f"Error tokenizing dataset: {str(e)}")
    raise

# Data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Custom callback to check for nan/inf
class CheckNanInfCallback(TrainerCallback):
    def on_train_begin(self, args, state, control, **kwargs):
        print("Training begins. Checking model parameters for nan/inf...")
        self.check_params(kwargs["model"])

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % 10 == 0:  # Check every 10 steps
            self.check_params(kwargs["model"])

    def check_params(self, model):
        has_nan_inf = False
        for name, param in model.named_parameters():
            if torch.isnan(param).any() or torch.isinf(param).any():
                print(f"Warning: Parameter {name} contains nan or inf")
                has_nan_inf = True
        if has_nan_inf:
            print("Continuing training despite nan/inf. Investigate hyperparameters or dataset.")
            torch.save(model.state_dict(), f"model_state_step_{state.global_step}.pt")

# Training arguments with adjusted save settings
training_args = TrainingArguments(
    output_dir="/home/apoc/Finetune/fine_tuned_model",  # Ensure this path has enough space and permissions
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=1 if device.type == "cpu" else 2,
    gradient_accumulation_steps=2,
    save_steps=1000,  # Reduced save frequency to minimize I/O issues
    save_total_limit=1,  # Limit number of saved checkpoints to save space
    logging_steps=20,
    learning_rate=1e-5,
    warmup_steps=100,
    fp16=torch.cuda.is_available(),
    max_grad_norm=0.5,
    optim="adamw_torch",
    evaluation_strategy="no",
    report_to="none",
)

# Initialize trainer with the custom callback
try:
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
        callbacks=[CheckNanInfCallback()],
    )
except Exception as e:
    print(f"Error initializing trainer: {str(e)}")
    raise

# Start fine-tuning
print("Starting fine-tuning...")
try:
    trainer.train()
except Exception as e:
    print(f"Error during training: {str(e)}")
    raise

# Save the model manually after training to ensure success
try:
    model.save_pretrained("./fine_tuned_model")
    tokenizer.save_pretrained("./fine_tuned_model")
    print("Fine-tuning completed. Model and tokenizer saved to ./fine_tuned_model")
except Exception as e:
    print(f"Error saving model: {str(e)}")
    raise