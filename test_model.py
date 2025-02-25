from transformers import pipeline,AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("./fine_tuned_model")
tokenizer = AutoTokenizer.from_pretrained("./fine_tuned_model")

generator = pipeline("text-generation", model, tokenizer)

prompt = "Where is the missing MSI Accelist Laptop?"
output = generator(prompt, max_length=50)
print("Generated Output:\n", output[0]['generated_text'])
