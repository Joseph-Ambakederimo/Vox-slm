import os
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW # <<-- NEW
from training.dataset import TextDataset
from model.model import VoxModel
from model.config import ModelConfig
# from main import Muon # <<-- REMOVED

# Checkpointing helper functions (keep as you provided)
# ... (save_checkpoint and load_checkpoint)

CHECKPOINT_DIR = "../checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def save_checkpoint(model, optimizer, scaler, step, path=None):
    path = path or os.path.join(CHECKPOINT_DIR, f"checkpoint_{step}.pt")
    torch.save({
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scaler_state": scaler.state_dict(),
        "step": step
    }, path)
    print(f"💾 checkpoint saved at step {step} -> {path}")

def load_checkpoint(model, optimizer=None, scaler=None, path=None, device="cuda"):
    if path is None or not os.path.exists(path):
        print("⚠️ checkpoint not found")
        return 0
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    if optimizer:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    if scaler:
        scaler.load_state_dict(ckpt["scaler_state"])
    print(f"♻️ checkpoint loaded from {path}")
    return ckpt["step"]

# Evaluation helper function (keep as you provided)
# ... (evaluate)

def evaluate(model: torch.nn.Module, tokenizer_path="../tokenizer", seq_len=512, num_docs=200, device="cuda"):
    from training.dataset import TextDataset
    import torch
    model.eval()

    # small eval dataset
    dataset = TextDataset(seq_len=seq_len, num_docs=num_docs)
    loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False)

    criterion = torch.nn.CrossEntropyLoss()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            total_loss += loss.item() * x.numel()
            total_tokens += x.numel()

    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss))
    print(f"📊 evaluation | loss: {avg_loss:.4f}, ppl: {perplexity:.2f}")
    model.train()


def train(model: VoxModel, config: ModelConfig):
    dataset = TextDataset(seq_len=config.max_seq_len, num_docs=config.num_documents)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Optimizer change: AdamW is standard for LLMs
    optimizer = AdamW(model.parameters(), lr=config.muon_lr) 
    scaler = GradScaler(enabled=config.use_amp)
    criterion = torch.nn.CrossEntropyLoss()

    # --- Checkpoint Loading ---
    start_step = 0
    # Assuming the checkpoint loading logic is handled in main.py before calling train()
    # If not, you'd integrate it here.

    step = start_step
    model.train()
    for epoch in range(1000):  # dummy large number, will stop at max_steps
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            with autocast(enabled=config.use_amp):
                logits = model(x)
                # Reshape logits (B*T, V) and targets (B*T) for CrossEntropyLoss
                loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            
            # Normalize loss by accumulation steps before scaling (best practice)
            loss = loss / config.gradient_accumulation_steps 

            scaler.scale(loss).backward()

            if (step + 1) % config.gradient_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
                # Checkpoint and Evaluation
                if step % config.eval_every == 0 and step > 0:
                    evaluate(model, seq_len=config.max_seq_len, num_docs=200, device=device)
                    save_checkpoint(model, optimizer, scaler, step)
                
                if step % 50 == 0:
                    print(f"step {step} | loss: {loss.item() * config.gradient_accumulation_steps:.4f}")
            
            step += 1

            if step >= config.max_steps:
                print("✅ reached max steps, stopping")
                # Save final checkpoint
                save_checkpoint(model, optimizer, scaler, step, path=os.path.join(CHECKPOINT_DIR, "checkpoint_latest.pt"))
                return