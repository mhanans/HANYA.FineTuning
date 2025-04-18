import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, LogitsProcessorList

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model and tokenizer
model_path = "./fine_tuned_model_merged"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)
model.to(device)
model.eval()

# Custom logits processor for stability
class SafeLogitsProcessor(LogitsProcessor):
    def __call__(self, input_ids, scores):
        scores = torch.where(
            torch.isnan(scores) | torch.isinf(scores),
            torch.full_like(scores, -1e9),
            scores
        )
        return scores

# Generate response function
def generate_response(prompt, max_new_tokens=100, temperature=0.7):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {key: val.to(device) for key, val in inputs.items()}
    input_length = inputs["input_ids"].shape[1]
    logits_processor = LogitsProcessorList([SafeLogitsProcessor()])
    
    try:
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                logits_processor=logits_processor,
            )
        # Hanya ambil token yang baru digenerate (setelah prompt)
        generated_tokens = outputs[0][input_length:]
        response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return response.strip()
    except Exception as e:
        print(f"Error: {str(e)}")
        return "Sorry, an error occurred."

# Chat interface
def chat_interface(user_input, history):
    if history:
        # Tambahkan marker "Bot:" untuk memandu model menghasilkan respons
        prompt = "\n".join([f"User: {h[0]}\nBot: {h[1]}" for h in history]) + f"\nUser: {user_input}\nBot:"
    else:
        prompt = f"User: {user_input}\nBot:"
    return generate_response(prompt)

# Gradio UI
with gr.Blocks(title="HANYA.FT") as demo:
    gr.Markdown("# HANYA.FT\nAsk anything!")
    chatbot = gr.Chatbot(label="Conversation")
    user_input = gr.Textbox(label="Your Question", placeholder="Type here...")
    submit_btn = gr.Button("Send")
    state = gr.State([])

    def submit_message(user_input, history):
        if not user_input.strip():
            return "", history
        bot_response = chat_interface(user_input, history)
        history.append((user_input, bot_response))
        return "", history

    submit_btn.click(fn=submit_message, inputs=[user_input, state], outputs=[user_input, chatbot])
    user_input.submit(fn=submit_message, inputs=[user_input, state], outputs=[user_input, chatbot])

demo.launch(share=True)
