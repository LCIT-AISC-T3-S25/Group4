import torch
import torch.nn as nn
import math

# Define SinusoidalPosEmb
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = emb.unsqueeze(0)
        t = t.unsqueeze(-1)
        emb = t * emb
        return torch.cat([emb.sin(), emb.cos()], dim=-1)

# Actual Model
class MyTextToImageModel(nn.Module):
    def __init__(self, text_embed_dim=512, time_embed_dim=64):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_embed_dim),
            nn.Linear(time_embed_dim, 128),
            nn.ReLU()
        )
        self.text_proj = nn.Linear(text_embed_dim, 128)
        self.encoder = nn.Sequential(
            nn.Conv2d(131, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.middle = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 3, 3, padding=1)
        )

    def forward(self, x, t, text_emb):
        t_emb = self.time_mlp(t)
        txt_emb = self.text_proj(text_emb)
        cond = t_emb + txt_emb
        cond = cond[:, :, None, None].expand(-1, -1, x.shape[2], x.shape[3])
        x = torch.cat([x, cond], dim=1)
        h = self.encoder(x)
        h = self.middle(h)
        return self.decoder(h)
