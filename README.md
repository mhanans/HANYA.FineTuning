## HANYA.FT

Welcome to the HANYA.FT project! This repository contains a Python application that uses a fine-tuned language model (using non GPU version & LoRa) to power a chatbot interface via Gradio. Whether you're fine-tuning a model with your own dataset or deploying an interactive chatbot, this guide will walk you through the process step-by-step.

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

    ''' bash
    git clone https://github.com/mhanans/HANYA.FineTuning.git
    cd your-repo-name

Alternatively, download the project files manually and navigate to the project directory.

2. Run the all in one shell
Easily just run the all in one shell, provided by HANYA.FT that will do all the hardwork.

- On Linux/MacOS/ Windows (Use Linux Env)
    ''' bash
    bash run_finetune.sh
Once activated, your terminal prompt should show (venv) to indicate you’re in the virtual environment.

3. Installed Dependencies
The on in one installation will the following content:

torch: For tensor computations and GPU support.
transformers: For pre-trained models and tokenizers.
gradio: For the web-based chatbot interface.
peft: For efficient fine-tuning with LoRA.
datasets: For loading and processing your custom dataset(default dataset on custom_knowledge.jsonl).

4. Prepare Your Model
You can either:

Use a pre-trained model from Hugging Face (e.g., unsloth unsloth/gemma-3-4b-it).
Fine-tune your own model (instructions below).
If you’ve already fine-tuned a model and saved it (e.g., to ./fine_tuned_model_merged), skip to the "Run the Chatbot" section.

Fine-Tuning the Model
1. Prepare Your Dataset
Create a JSONL file (e.g., custom_knowledge.jsonl) with your training data. Each line should be a JSON object with prompt and completion fields:
    ''' bash
    {"prompt": "What is AI?", "completion": "AI stands for Artificial Intelligence, a field of computer science focused on creating systems that can perform tasks requiring human intelligence."}
    {"prompt": "How does a chatbot work?", "completion": "A chatbot uses natural language processing and machine learning to understand and generate human-like responses based on input text."}

2. Auto Load from PDF
Just easily put your pdf on pdfs folder and let the system create .jsonl format from it

## Use the model
A Gradio interface will launch in your browser after the process done, and you can use your own finetuned model for whatever you want.