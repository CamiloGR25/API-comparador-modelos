from collections import Counter
import json


class Vocabulary:
    def __init__(self, min_freq: int = 1):
        self.min_freq = min_freq
        self.special_tokens = ["<pad>", "<unk>", "<bos>", "<eos>"]
        self.stoi = {}
        self.itos = {}

    def build(self, texts):
        counter = Counter()
        for text in texts:
            counter.update(text.split())

        tokens = self.special_tokens[:]
        for token, freq in counter.items():
            if freq >= self.min_freq and token not in tokens:
                tokens.append(token)

        self.stoi = {token: idx for idx, token in enumerate(tokens)}
        self.itos = {idx: token for token, idx in self.stoi.items()}

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = True):
        tokens = text.split()
        ids = []
        if add_bos:
            ids.append(self.stoi["<bos>"])
        ids.extend(self.stoi.get(token, self.stoi["<unk>"]) for token in tokens)
        if add_eos:
            ids.append(self.stoi["<eos>"])
        return ids

    def decode(self, ids, skip_special: bool = True):
        special = set(self.special_tokens)
        tokens = []
        for idx in ids:
            token = self.itos.get(int(idx), "<unk>")
            if skip_special and token in special:
                continue
            tokens.append(token)
        return " ".join(tokens)

    @property
    def pad_id(self):
        return self.stoi["<pad>"]

    @property
    def bos_id(self):
        return self.stoi["<bos>"]

    @property
    def eos_id(self):
        return self.stoi["<eos>"]

    def __len__(self):
        return len(self.stoi)

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"min_freq": self.min_freq, "stoi": self.stoi}, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        vocab = cls(min_freq=data.get("min_freq", 1))
        vocab.stoi = data["stoi"]
        vocab.itos = {idx: token for token, idx in vocab.stoi.items()}
        return vocab
