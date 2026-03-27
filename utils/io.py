import json
from pathlib import Path
from utils.text import normalize_text


def load_text_lines(path: str):
    path_obj = Path(path)
    lines = []
    with path_obj.open("r", encoding="utf-8") as f:
        for line in f:
            line = normalize_text(line)
            if line:
                lines.append(line)
    return lines


def load_jsonl_pairs(path: str):
    path_obj = Path(path)
    rows = []
    with path_obj.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            rows.append(
                {
                    "category": normalize_text(obj.get("category", "general")),
                    "input": normalize_text(obj["input"]),
                    "target": normalize_text(obj["target"]),
                }
            )
    return rows
