import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, emb_dim: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, emb_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, emb_dim, 2).float() * (-math.log(10000.0) / emb_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class TransformerLanguageModel(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int = 64, nhead: int = 4, hidden_dim: int = 128, num_layers: int = 2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.pos_encoding = PositionalEncoding(emb_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(emb_dim, vocab_size)

    def forward(self, x):
        emb = self.embedding(x)
        emb = self.pos_encoding(emb)
        seq_len = x.size(1)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        pad_mask = x.eq(0)
        out = self.transformer(emb, mask=causal_mask, src_key_padding_mask=pad_mask)
        return self.fc(out)
