🗣️ lute-slm: A Small Language Model Framework (Version 0)

Welcome to lute-slm, a project dedicated to building and understanding Small Language Models (SLMs) from the ground up using PyTorch. This framework is currently in Version 0, which serves as a robust Proof-of-Concept for the core Transformer architecture and the training pipeline.

🌟 Project Description (Version 0)

lute-slm V0 implements a foundational decoder-only Transformer model, inspired by modern architectures like Qwen and Llama. It uses key techniques such as Multi-Head Attention, RMS Normalization, and the SiLU activation function.

The primary goal of Version 0 is to provide a clear, readable, and functional foundation for:

Data Preparation: Training a custom Byte-Pair Encoding (BPE) tokenizer from scratch.

Model Definition: Defining a complete Transformer block and model in PyTorch.

Training: Implementing a simple, synchronous training loop with checkpointing.

Inference: Enabling basic text generation (sampling).

🛠️ Installation and Dependencies

To run lute-slm, you will need Python 3.8+ and the libraries listed in requirements.txt.

1. Clone the Repository

git clone [https://github.com/your-username/lute-slm.git](https://github.com/Joseph-Ambakederimo/Lute-slm.git)
cd lute-slm


2. Install Dependencies

pip install -r requirements.txt


3. Prepare Your Training Data

Place your raw text files (e.g., .txt files containing books, articles, or other text data) inside the data/local_texts/ directory.

🚀 Usage Guide

The workflow involves three key phases: Tokenization, Training, and Generation.

Phase 1: Train the Tokenizer

Before training the model, you must train the BPE tokenizer on your data. This creates the vocabulary files the model needs.

Navigate to the tokenizer directory:

cd tokenizer


Run the tokenizer training script:

python tokenizer.py


This script will read all files in ../data/local_texts, train the tokenizer, and save the resulting files (vocab.json and merges.txt) in the current directory.

Return to the root directory:

cd ..


Phase 2: Train the Model

The main entry point for training is main.py. This script handles loading the tokenizer, preparing the dataset, initializing the model, and starting the training loop (defined in training/trainer.py).

python main.py


Training parameters (like batch size, learning rate, and model dimensions) can be adjusted in model/config.py. Checkpoints will be saved automatically to data/checkpoints/.

Phase 3: Generation and Inference

Once you have a checkpoint saved, you can use the inference.py script (or generate.py for a simpler test) to interact with your trained model.

Using inference.py (Recommended for deployment):

python inference.py


The inference.py file contains the SLMPredictor class, which handles loading the model and tokenizer and provides a clean predict method for programmatic generation.

Using the SLMPredictor class:

from inference import SLMPredictor

# Initialize the predictor (loads the latest checkpoint)
predictor = SLMPredictor()

# Generate text
prompt = "The most important thing I learned today was"
response = predictor.predict(prompt, max_new_tokens=80, temperature=0.8)

print(response)


🤝 Collaboration and Contribution

We welcome contributions! As this project moves toward more advanced versions, any help with code cleanup, documentation, or implementation of new features is greatly appreciated.

Current Roadmap (Moving to V2 & V3)

Target Version

Key Goal

Focus Area

V1.0

Stable Foundation

Complete this initial release (V0) with robust error handling and comprehensive documentation.

V2.0

High Performance & Scale

Architectural Upgrade: Implement Rotary Positional Embeddings (RoPE) and SwiGLU. Data Efficiency: Implement dataset streaming for training on massive public datasets (e.g., Hugging Face datasets).

V3.0

Chatbot Alignment

Alignment Training: Implement Supervised Fine-Tuning (SFT) and Direct Preference Optimization (DPO) to make the model follow instructions and act as a conversational assistant.

How to Contribute

Fork the repository.

Create a branch for your feature or fix (e.g., feat/add-rope-v2 or fix/checkpoint-bug).

Commit your changes clearly.

Open a Pull Request against the main branch, explaining your changes and the problem they solve.

🏗️ Project Structure

The project is modularized for clarity and maintainability:

lute-slm/
│
├── data/                       # Training data and saved model checkpoints
│   ├── local_texts/            # Your raw .txt files go here
│   └── checkpoints/            # Saved model weights (.pt files)
│
├── model/                      # Core Transformer architecture definition
│   ├── config.py               # Model hyperparameters (layers, size, vocab)
│   ├── layers.py               # Transformer components (Attention, Norm, FFN)
│   └── model.py                # Main model class (QwenModel)
│
├── tokenizer/                  # Tokenizer training and saved files
│   ├── tokenizer.py            # Script to train BPE
│   ├── vocab.json              # Trained vocabulary
│   └── merges.txt              # BPE merge rules
│
├── training/                   # Training logic and utilities
│   ├── dataset.py              # Data loading, preprocessing, and batching
│   └── trainer.py              # The core training/evaluation loop
│
├── main.py                     # Entry script to start training
├── inference.py                # Entry script for production generation/prediction
└── requirements.txt            # Python dependencies
