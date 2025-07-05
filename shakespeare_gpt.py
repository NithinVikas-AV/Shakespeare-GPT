import torch
import torch.nn as nn
from torch.nn import functional as F

# Hyperparameters
batch_size = 32
block_size = 128
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 256
n_head = 4
n_layer = 4
dropout = 0.2
# ---------------

torch.manual_seed(1377)

with open('tinyShakespeare.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Here are all the unique characters that occur in this text.
chars = sorted(list(set(text)))
vocab_size = len(chars)
# Create a mapping from characters to integers.
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s] # Encoder: Takes a string, outputs a list of integers.
decode = lambda l: ''.join([itos[i] for i in l]) # Decoder: Takes a list of integers, outputs a string.

# Train and Test splits.
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# Data loading
def get_batch(split):
    # Generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

# The function we used before will calculate the loss for many batches.
# So that is a noisy measurement of the current loss because every batch will be more or less lucky.

@torch.no_grad() # This context manager will let know Pytroch that everything that happens inside this function will not call backward() function.  
                 # So Pytorch can be lot more efficent with its memory because it doesnt want to store all the intermediate variables happening inside because we will never call backward.
def estimate_loss(): # This function averages up the loss of multiple batches. This is a very less noisy measurement of loss.
    out = {}  # This function will return a pretty accurate loss on train and validation loss.
    model.eval() # Evaluation mode
    for split in  ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train() # Training mode. We just use this for practice. Right now nothing is gonna change. But some layer will have different behaviour like when inference time or training time.
    return out

class Head(nn.Module):
    """ one head of self attention"""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x) # (B, T, C)
        q = self.query(x) # (B, T, C)
        # compute attention scores 'affinities'.
        wei = q @ k.transpose(-2, -1) * C**-0.5  # (B, T, C) @ (B, C, T) ---> (B, T, T)
        wei = wei.masked_fill(self.tril[:T,:T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x) # (B, T, C)
        out = wei @ v # (B, T, T) @ (B, T, C) ---> (B, T, C)
        return out

class MultilHeadAttention(nn.Module):
    """ mutiple heads of self attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList(Head(head_size) for _ in range(num_heads))
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x)for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    """ a simple linear layer followed by non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )

    def forward(self, x): # This is a token level and all token will do independently.
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation"""

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of head we'd like.
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultilHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        
    def forward(self, x):
        x = x + self.sa(self.ln1(x)) # Fork-off and do some computation and come back.
        x = x + self.ffwd(self.ln2(x)) # Fork-off and do some computation and come back.
        return x

class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table.
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd) # vocal size means the identity of the character.
        self.position_embedding_table = nn.Embedding(block_size, n_embd) # block size means the position of the character.
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # Final layer norm.
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers.
        tok_emb = self.token_embedding_table(idx) # (B, T, n_embd)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B, T, C) This holds not just the token identity but also the positions at which these token occur.
        x = self.blocks(x) # (B, T, C)
        x = self.ln_f(x) # (B, T, C)
        logits = self.lm_head(x) # (B, T, vocal_size)

        if targets == None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C) 
            targets = targets.view(B*T)  
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        # idx is (B,T) array of indices in the current context.
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens.
            idx_cond = idx[:, -block_size:] 
            # Get the Predictions.
            logits, loss = self(idx_cond)
            # Focus only on the last time step.
            logits = logits[:, -1, :] # Becomes (B,C).
            # Apply softmax to obtain the proabability.
            probs = F.softmax(logits, dim=-1) # (B,C)
            # Sample from the distribution.
            idx_next = torch.multinomial(probs, num_samples=1) # (B,1)
            # Append sampled index to the running sequence.
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

model = BigramLanguageModel()
m = model.to(device)

# Create a Pytorch Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss 
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
with open('generated.txt', 'w') as f:
    context = torch.zeros((1,1), dtype=torch.long, device=device)
    f.write(decode(m.generate(context, max_new_tokens=10000)[0].tolist()))

# OUTPUT: 

# step 0: train loss 4.3041, val loss 4.3118
# step 500: train loss 2.2091, val loss 2.2435
# step 1000: train loss 1.8828, val loss 1.9817
# step 1500: train loss 1.7023, val loss 1.8669
# step 2000: train loss 1.6057, val loss 1.7984
# step 2500: train loss 1.5378, val loss 1.7335
# step 3000: train loss 1.4900, val loss 1.6984
# step 3500: train loss 1.4592, val loss 1.6580
# step 4000: train loss 1.4297, val loss 1.6253
# step 4500: train loss 1.4076, val loss 1.6102