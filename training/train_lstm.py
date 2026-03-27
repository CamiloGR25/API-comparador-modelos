import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.lstm_model import LSTMLanguageModel
from training.common import prepare_data, train_model, quick_test



def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_path = ROOT / "data" / "raw" / "train.jsonl"
    seq_len = 24
    _, vocab, dataset = prepare_data(str(data_path), seq_len=seq_len)
    model = LSTMLanguageModel(vocab_size=len(vocab), emb_dim=64, hidden_dim=128)
    train_model(model, dataset, vocab, save_prefix="lstm", epochs=20, batch_size=16, lr=1e-3, device=device)
    quick_test(model, vocab, "natalia llamó a sergio para confirmar la reunión. él solicitó moverla para el jueves. ¿quién solicitó mover la reunión?", seq_len=seq_len, device=device)


if __name__ == "__main__":
    main()
