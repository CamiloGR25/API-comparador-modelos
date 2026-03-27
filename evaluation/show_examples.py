import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "evaluation" / "results"


def load_predictions(name: str):
    path = RESULTS / f"{name}_predictions.json"
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main(limit: int = 5):
    preds = {name: load_predictions(name) for name in ["rnn", "lstm", "transformer"]}
    total = len(preds["rnn"])
    shown = 0
    for idx in range(total):
        rows = {name: preds[name][idx] for name in preds}
        different = len({rows[name]["prediction"] for name in rows}) > 1
        if not different:
            continue
        print("=" * 80)
        print(f"Caso #{rows['rnn']['id']} | Categoría: {rows['rnn']['category']}")
        print(f"Entrada: {rows['rnn']['input']}")
        print(f"Objetivo: {rows['rnn']['target']}")
        for name in ["rnn", "lstm", "transformer"]:
            print(f"{name.upper():12} -> {rows[name]['prediction']} | EM={rows[name]['exact_match']} | F1={rows[name]['token_f1']}")
        shown += 1
        if shown >= limit:
            break
    if shown == 0:
        print("No se encontraron ejemplos divergentes en los resultados actuales.")


if __name__ == "__main__":
    main()
