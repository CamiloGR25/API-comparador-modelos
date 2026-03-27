import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.rnn_model import RNNLanguageModel
from models.lstm_model import LSTMLanguageModel
from models.transformer_model import TransformerLanguageModel
from utils.io import load_jsonl_pairs
from utils.vocab import Vocabulary
from utils.generation import generate_chat_response
from utils.text import normalize_text

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_PATH = ROOT / "data" / "raw" / "test.jsonl"
OUT_DIR = ROOT / "evaluation" / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CONFIGS = {
    "rnn": {
        "weights": ROOT / "data" / "processed" / "rnn.pt",
        "vocab": ROOT / "data" / "processed" / "rnn_vocab.json",
        "meta": ROOT / "data" / "processed" / "rnn_meta.json",
        "builder": lambda vocab_size: RNNLanguageModel(vocab_size=vocab_size, emb_dim=64, hidden_dim=128),
    },
    "lstm": {
        "weights": ROOT / "data" / "processed" / "lstm.pt",
        "vocab": ROOT / "data" / "processed" / "lstm_vocab.json",
        "meta": ROOT / "data" / "processed" / "lstm_meta.json",
        "builder": lambda vocab_size: LSTMLanguageModel(vocab_size=vocab_size, emb_dim=64, hidden_dim=128),
    },
    "transformer": {
        "weights": ROOT / "data" / "processed" / "transformer.pt",
        "vocab": ROOT / "data" / "processed" / "transformer_vocab.json",
        "meta": ROOT / "data" / "processed" / "transformer_meta.json",
        "builder": lambda vocab_size: TransformerLanguageModel(vocab_size=vocab_size, emb_dim=64, nhead=4, hidden_dim=128, num_layers=2),
    },
}


def load_model(name: str):
    config = CONFIGS[name]
    missing = [p for p in [config["weights"], config["vocab"], config["meta"]] if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Faltan archivos del modelo {name}: {missing}")

    vocab = Vocabulary.load(str(config["vocab"]))
    with config["meta"].open("r", encoding="utf-8") as f:
        meta = json.load(f)

    model = config["builder"](len(vocab))
    state = torch.load(config["weights"], map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model, vocab, int(meta.get("seq_len", 24))


def token_f1(pred: str, gold: str) -> float:
    pred_tokens = pred.split()
    gold_tokens = gold.split()
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    pred_counter = Counter(pred_tokens)
    gold_counter = Counter(gold_tokens)
    common = sum((pred_counter & gold_counter).values())
    if common == 0:
        return 0.0
    precision = common / max(1, len(pred_tokens))
    recall = common / max(1, len(gold_tokens))
    return 2 * precision * recall / (precision + recall)


def evaluate_model(name: str, rows):
    model, vocab, seq_len = load_model(name)
    per_example = []
    total_exact = 0
    total_f1 = 0.0
    category_stats = defaultdict(lambda: {"count": 0, "exact": 0, "f1_sum": 0.0})

    for i, row in enumerate(rows, start=1):
        pred = generate_chat_response(
            model=model,
            vocab=vocab,
            message=row["input"],
            history=None,
            seq_len=seq_len,
            max_new_tokens=20,
            device=DEVICE,
        )
        pred = normalize_text(pred)
        gold = normalize_text(row["target"])
        exact = int(pred == gold)
        f1 = token_f1(pred, gold)

        total_exact += exact
        total_f1 += f1
        cat = row["category"]
        category_stats[cat]["count"] += 1
        category_stats[cat]["exact"] += exact
        category_stats[cat]["f1_sum"] += f1

        per_example.append({
            "id": i,
            "category": cat,
            "input": row["input"],
            "target": gold,
            "prediction": pred,
            "exact_match": exact,
            "token_f1": round(f1, 4),
        })

    n = max(1, len(rows))
    summary = {
        "model": name,
        "examples": len(rows),
        "exact_match": round(total_exact / n, 4),
        "avg_token_f1": round(total_f1 / n, 4),
        "by_category": {
            cat: {
                "count": stats["count"],
                "exact_match": round(stats["exact"] / max(1, stats["count"]), 4),
                "avg_token_f1": round(stats["f1_sum"] / max(1, stats["count"]), 4),
            }
            for cat, stats in sorted(category_stats.items())
        },
    }
    return summary, per_example


def save_outputs(name: str, summary: dict, per_example: list[dict]):
    with (OUT_DIR / f"{name}_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with (OUT_DIR / f"{name}_predictions.json").open("w", encoding="utf-8") as f:
        json.dump(per_example, f, ensure_ascii=False, indent=2)
    with (OUT_DIR / f"{name}_predictions.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_example[0].keys()) if per_example else ["id"])
        writer.writeheader()
        for row in per_example:
            writer.writerow(row)


def create_comparison_report(all_summaries: dict):
    report_path = OUT_DIR / "comparison_report.md"
    lines = [
        "# Reporte comparativo de modelos\n",
        "\n",
        "## Resumen global\n",
        "\n",
        "| Modelo | Exact Match | Token F1 | Ejemplos |\n",
        "|---|---:|---:|---:|\n",
    ]
    for name in ["rnn", "lstm", "transformer"]:
        summary = all_summaries.get(name, {})
        lines.append(
            f"| {name.upper()} | {summary.get('exact_match', 0):.4f} | {summary.get('avg_token_f1', 0):.4f} | {summary.get('examples', 0)} |\n"
        )

    lines.append("\n## Métricas por categoría\n\n")
    categories = sorted({cat for s in all_summaries.values() for cat in s.get("by_category", {}).keys()})
    for cat in categories:
        lines.append(f"### {cat}\n\n")
        lines.append("| Modelo | Exact Match | Token F1 | Casos |\n")
        lines.append("|---|---:|---:|---:|\n")
        for name in ["rnn", "lstm", "transformer"]:
            stats = all_summaries.get(name, {}).get("by_category", {}).get(cat, {})
            lines.append(
                f"| {name.upper()} | {stats.get('exact_match', 0):.4f} | {stats.get('avg_token_f1', 0):.4f} | {stats.get('count', 0)} |\n"
            )
        lines.append("\n")

    with report_path.open("w", encoding="utf-8") as f:
        f.writelines(lines)


def main():
    rows = load_jsonl_pairs(str(DATA_PATH))
    all_summaries = {}
    for name in ["rnn", "lstm", "transformer"]:
        print(f"Evaluando {name}...")
        summary, per_example = evaluate_model(name, rows)
        save_outputs(name, summary, per_example)
        all_summaries[name] = summary
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    create_comparison_report(all_summaries)
    print(f"\nResultados guardados en: {OUT_DIR}")


if __name__ == "__main__":
    main()
