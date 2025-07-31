import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter

class CausalTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=8, num_layers=4, max_len=64):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, d_model))
        self.dropout = nn.Dropout(0.3)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.embed(x) + self.pos_embed[:, :x.size(1)]
        x = self.dropout(x)
        x = self.transformer(x)
        return self.fc(x)

def encode(text, stoi, max_len=64):
    tokens = text.lower().split()
    ids = [stoi.get(t, stoi["<unk>"]) for t in tokens][:max_len]
    return ids + [stoi["<pad>"]] * (max_len - len(ids))

def build_vocab(texts, min_freq=1):
    counter = Counter()
    for text in texts:
        counter.update(text.lower().split())
    vocab = ["<pad>", "<unk>"] + sorted([w for w, f in counter.items() if f >= min_freq])
    stoi = {w: i for i, w in enumerate(vocab)}
    return stoi