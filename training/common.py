import json
import os
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.io import load_jsonl_pairs
from utils.vocab import Vocabulary
from utils.dataset import LanguageModelingDataset
from utils.generation import generate_chat_response


def format_chat_example(user_text: str, assistant_text: str) -> str:
    return f"usuario: {user_text} asistente: {assistant_text}"


def prepare_data(data_path: str, seq_len: int = 24, min_freq: int = 1):
    pairs = load_jsonl_pairs(data_path)
    texts = [format_chat_example(row['input'], row['target']) for row in pairs]
    vocab = Vocabulary(min_freq=min_freq)
    vocab.build(texts)
    encoded = [vocab.encode(text) for text in texts]
    dataset = LanguageModelingDataset(encoded, seq_len=seq_len, pad_id=vocab.pad_id)
    return pairs, vocab, dataset


def train_model(model, dataset, vocab, save_prefix: str, epochs: int = 30, batch_size: int = 16, lr: float = 1e-3, device: str = "cpu"):
    model.to(device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_id)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())

        avg_loss = total_loss / max(1, len(loader))
        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            print(f"Epoch {epoch}/{epochs} - loss: {avg_loss:.4f}")

    out_dir = ROOT / "data" / "processed"
    os.makedirs(out_dir, exist_ok=True)
    model_path = out_dir / f"{save_prefix}.pt"
    vocab_path = out_dir / f"{save_prefix}_vocab.json"
    meta_path = out_dir / f"{save_prefix}_meta.json"
    torch.save(model.state_dict(), model_path)
    vocab.save(str(vocab_path))
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump({"seq_len": dataset.seq_len}, f, ensure_ascii=False, indent=2)
    print(f"Modelo guardado en: {model_path}")
    print(f"Vocabulario guardado en: {vocab_path}")
    print(f"Metadatos guardados en: {meta_path}")



def quick_test(model, vocab, prompt: str, seq_len: int = 24, device: str = "cpu"):
    model.eval()
    response_text = generate_chat_response(
        model=model,
        vocab=vocab,
        message=prompt,
        history=None,
        seq_len=seq_len,
        max_new_tokens=20,
        device=device,
    )
    print(f"Prompt: {prompt}")
    print(f"Respuesta: {response_text}")
