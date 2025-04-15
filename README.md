## Chatbot Fine-Tuned Model

Welcome to the Chatbot Fine-Tuned Model project! This repository contains a Python application that uses a fine-tuned language model (based on deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B) to power a chatbot interface via Gradio. Whether you're fine-tuning a model with your own dataset or deploying an interactive chatbot, this guide will walk you through the process step-by-step.

## Features
Fine-tune a pre-trained model with your custom dataset.
Interactive chatbot interface powered by Gradio.
Handles common issues like invalid logits (nan/inf) for stable text generation.
Deployable locally or on platforms like Hugging Face Spaces.
Prerequisites
Before you begin, ensure you have the following installed:

- Python 3.8+ (3.12 recommended for compatibility).
- Git (optional, for cloning the repository).
- A compatible GPU (optional, for faster fine-tuning/inference; CPU works too).

## Setup Instructions
1. Clone the Repository
If you have a Git repository, clone it to your local machine:

    '''bash
    git clone https://github.com/your-username/your-repo-name.git
    cd your-repo-name

Alternatively, download the project files manually and navigate to the project directory.

2. Create a Virtual Environment
Using a virtual environment (venv) keeps dependencies isolated and avoids conflicts. Here’s how to set it up:

On Linux/MacOS:

python3 -m venv venv
source venv/bin/activate
On Windows:

python -m venv venv
venv\Scripts\activate
Once activated, your terminal prompt should show (venv) to indicate you’re in the virtual environment.

3. Install Dependencies
Install the required Python packages using pip. Create a requirements.txt file with the following content:

torch
transformers
gradio
peft
datasets

Then run:
pip install -r requirements.txt
Alternatively, install them individually:

torch: For tensor computations and GPU support.
transformers: For pre-trained models and tokenizers.
gradio: For the web-based chatbot interface.
peft: For efficient fine-tuning with LoRA.
datasets: For loading and processing your custom dataset.
4. Prepare Your Model
You can either:

Use a pre-trained model from Hugging Face (e.g., deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B).
Fine-tune your own model (instructions below).
If you’ve already fine-tuned a model and saved it (e.g., to ./fine_tuned_model_merged), skip to the "Run the Chatbot" section.

Fine-Tuning the Model
1. Prepare Your Dataset
Create a JSONL file (e.g., custom_knowledge.jsonl) with your training data. Each line should be a JSON object with prompt and completion fields:

{"prompt": "What is AI?", "completion": "AI stands for Artificial Intelligence, a field of computer science focused on creating systems that can perform tasks requiring human intelligence."}
{"prompt": "How does a chatbot work?", "completion": "A chatbot uses natural language processing and machine learning to understand and generate human-like responses based on input text."}
2. Fine-Tune the Model
Use the provided fine-tuning script (fine_tune.py). Here’s an example:
Run the script: python fine_tune.py
This will save your fine-tuned model to ./fine_tuned_model_merged.

Run the Chatbot
1. Use the Provided Script
Use the app.py script below to launch the Gradio interface:
2. Launch the Application
Run the script: python test_model.py

A Gradio interface will launch in your browser.
Use the public URL (if share=True) to share it with others.

Deploy:
Commit and push your changes. Hugging Face will build and host the app automatically.
Troubleshooting

Contributing
Feel free to submit issues or pull requests to improve this project!

License
This project is licensed under the MIT License.