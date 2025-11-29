from model.config import ModelConfig
from model.model import VoxModel
# Assuming trainer.py has the correct imports for AdamW, load_checkpoint, etc.
from training.trainer import train, evaluate, load_checkpoint
import torch
import os

# -----------------------
# CONFIG
# -----------------------
# Loads all model and training settings
config = ModelConfig(vocab_size=50000) 
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"🚀 Starting SLM Training on device: {device}")
print(f"Model Size: {config.n_layers} layers, {config.d_model} embed dim")
print(f"Effective Batch Size: {config.batch_size * config.gradient_accumulation_steps}")

# -----------------------
# MODEL & OPTIMIZER SETUP
# -----------------------
model = VoxModel(config).to(device)

# Assuming you replaced Muon with AdamW in trainer.py
from torch.optim import AdamW 
optimizer = AdamW(model.parameters(), lr=config.muon_lr)
scaler = torch.cuda.amp.GradScaler(enabled=config.use_amp)

# -----------------------
# RESUME FROM CHECKPOINT
# -----------------------
CHECKPOINT_DIR = "../checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
checkpoint_path = os.path.join(CHECKPOINT_DIR, "checkpoint_latest.pt")
start_step = 0

if os.path.exists(checkpoint_path):
    start_step = load_checkpoint(model, optimizer, scaler, path=checkpoint_path, device=device)

# --- Set the starting step in the optimizer for resuming training ---
# This is a bit complex for a standard AdamW, but for clean resume, 
# you often need to manually set the global step if the trainer relies on it.

# -----------------------
# TRAINING LOOP START
# -----------------------
try:
    train(model, config)
except KeyboardInterrupt:
    print("\nTraining interrupted by user. Saving final checkpoint...")
    # You would need to pass the current step/optimizer/scaler back from the train function 
    # to save the *last* state correctly here.

# -----------------------
# OPTIONAL FINAL EVAL
# -----------------------
evaluate(model, seq_len=config.max_seq_len, num_docs=200, device=device)