import torch
import json
from model import BigramLanguageModel

# -------- Device --------
device = "cuda" if torch.cuda.is_available() else "cpu"

# -------- Load Vocabulary --------
with open("vocab.json", "r") as f:
    vocab = json.load(f)

stoi = vocab["stoi"]
itos = {int(k): v for k, v in vocab["itos"].items()}
vocab_size = vocab["vocab_size"]

def encode(s):
    return [stoi[c] for c in s if c in stoi]

def decode(l):
    return "".join([itos[i] for i in l])


# -------- Load Model Config --------
with open("models/models.json", "r") as f:
    MODEL_CONFIG = json.load(f)

# Cache models to avoid reloading every request
MODEL_CACHE = {}


def load_model(model_name):
    """
    Loads model from models.json configuration.
    Uses caching to prevent reloading every time.
    """

    if model_name in MODEL_CACHE:
        return MODEL_CACHE[model_name]

    if model_name not in MODEL_CONFIG:
        raise ValueError(f"Model '{model_name}' not found in models.json")

    config = MODEL_CONFIG[model_name]

    model = BigramLanguageModel(
        vocab_size=vocab_size,
        n_embd=config["n_embd"],
        n_head=config["n_head"],
        n_layer=config["n_layer"],
        block_size=config["block_size"],
        dropout=config["dropout"]
    ).to(device)

    model.load_state_dict(
        torch.load(config["file"], map_location=device)
    )

    model.eval()

    MODEL_CACHE[model_name] = model
    return model


@torch.no_grad()
def generate_poem(model_name,
                  prompt,
                  max_tokens=200,
                  temperature=1.0):
    """
    Main function used by Flask route.
    """

    model = load_model(model_name)

    idx = torch.tensor(
        [encode(prompt)],
        dtype=torch.long
    ).to(device)

    for _ in range(max_tokens):

        idx_cond = idx[:, -model.block_size:]
        logits, _ = model(idx_cond)

        logits = logits[:, -1, :] / temperature
        probs = torch.nn.functional.softmax(logits, dim=-1)

        next_id = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, next_id), dim=1)

    return decode(idx[0].tolist())