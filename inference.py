import torch
import os
from model.model import LuteModel
from model.config import ModelConfig
from tokenizers import ByteLevelBPETokenizer

class SLMPredictor:
    def __init__(self, model_dir="./checkpoints", tokenizer_dir="./tokenize"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🚀 Initializing Inference on {self.device}...")

        # 1. Load Configuration
        # In V2, we will load this from a config.json file
        self.config = ModelConfig(vocab_size=50000)

        # 2. Load Tokenizer
        vocab_path = os.path.join(tokenizer_dir, "vocab.json")
        merges_path = os.path.join(tokenizer_dir, "merges.txt")
        
        if not os.path.exists(vocab_path):
            raise FileNotFoundError(f"Tokenizer files not found in {tokenizer_dir}")
            
        self.tokenizer = ByteLevelBPETokenizer(vocab_path, merges_path)

        # 3. Load Model Structure
        self.model = LuteModel(self.config).to(self.device)

        # 4. Load Weights (Checkpoint)
        checkpoint_path = os.path.join(model_dir, "checkpoint_latest.pt")
        if os.path.exists(checkpoint_path):
            print(f"📦 Loading weights from {checkpoint_path}...")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            # Handle potential key mismatches if saved differently
            state_dict = checkpoint.get("model_state", checkpoint)
            self.model.load_state_dict(state_dict)
            self.model.eval()
        else:
            print(f"⚠️ No checkpoint found at {checkpoint_path}. Using random weights!")

    def predict(self, prompt, max_new_tokens=100, temperature=0.7, top_k=50):
        """
        Takes a text prompt and returns the completed text.
        """
        # Encode
        input_ids = self.tokenizer.encode(prompt).ids
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)

        # Generate
        # (This is a simplified generation loop. V2 should use KV-Caching)
        for _ in range(max_new_tokens):
            with torch.no_grad():
                logits = self.model(input_tensor)
                
                # Get last token logits
                last_logits = logits[0, -1, :] / temperature
                
                # Top-K Filtering
                if top_k > 0:
                    v, _ = torch.topk(last_logits, top_k)
                    last_logits[last_logits < v[-1]] = -float('Inf')

                # Softmax & Sample
                probs = torch.nn.functional.softmax(last_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Stop if EOS (End of Sequence) - assuming standard ID, adjust if needed
                # if next_token.item() == self.tokenizer.token_to_id("</s>"):
                #     break

                # Append
                input_tensor = torch.cat((input_tensor, next_token.unsqueeze(0)), dim=1)

        # Decode
        generated_ids = input_tensor[0].tolist()
        return self.tokenizer.decode(generated_ids)

# --- Test Block (Runs only if you run 'python inference.py') ---
if __name__ == "__main__":
    predictor = SLMPredictor()
    response = predictor.predict("The future of AI is", max_new_tokens=50)
    print("\n🤖 Model Output:\n" + response)