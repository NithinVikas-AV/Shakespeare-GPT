import torch
import json
from model import BigramLanguageModel

device = "cuda" if torch.cuda.is_available() else "cpu"

# -------- Load Dataset --------
with open("tinyShakespeare.txt", "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


def get_batch(split, batch_size, block_size):
    data_source = train_data if split == "train" else val_data
    ix = torch.randint(len(data_source) - block_size, (batch_size,))
    x = torch.stack([data_source[i:i+block_size] for i in ix])
    y = torch.stack([data_source[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, batch_size, block_size):
    model.eval()
    losses = {}
    for split in ["train", "val"]:
        loss_list = []
        for _ in range(200):
            X, Y = get_batch(split, batch_size, block_size)
            _, loss = model(X, Y)
            loss_list.append(loss.item())
        losses[split] = sum(loss_list) / len(loss_list)
    model.train()
    return losses


# -------- Load Model Config --------
with open("models/models.json") as f:
    MODEL_CONFIG = json.load(f)


def train_model(name, config):

    print(f"\nTraining {name.upper()}")

    model = BigramLanguageModel(
        vocab_size=vocab_size,
        n_embd=config["n_embd"],
        n_head=config["n_head"],
        n_layer=config["n_layer"],
        block_size=config["block_size"],
        dropout=config["dropout"]
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"]
    )

    for iter in range(config["max_iters"]):

        if iter % 500 == 0:
            losses = estimate_loss(model,
                                   config["batch_size"],
                                   config["block_size"])
            print(
                f"Step {iter} | "
                f"Train: {losses['train']:.4f} | "
                f"Val: {losses['val']:.4f}"
            )

        xb, yb = get_batch("train",
                           config["batch_size"],
                           config["block_size"])

        logits, loss = model(xb, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    torch.save(model.state_dict(), config["file"])
    print(f"Saved {name}")


if __name__ == "__main__":

    for name, config in MODEL_CONFIG.items():
        train_model(name, config)

    print("\nAll models trained.")