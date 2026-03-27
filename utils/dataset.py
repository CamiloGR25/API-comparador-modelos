import torch
from torch.utils.data import Dataset


class LanguageModelingDataset(Dataset):
    def __init__(self, encoded_sequences, seq_len: int = 24, pad_id: int = 0):
        self.samples = []
        self.seq_len = seq_len
        self.pad_id = pad_id

        for seq in encoded_sequences:
            if len(seq) < 2:
                continue

            for i in range(1, len(seq)):
                x = seq[:i]
                y = seq[1:i + 1]

                x = x[-seq_len:]
                y = y[-seq_len:]

                x = self._left_pad(x)
                y = self._left_pad(y)
                self.samples.append((x, y))

    def _left_pad(self, seq):
        if len(seq) < self.seq_len:
            seq = [self.pad_id] * (self.seq_len - len(seq)) + seq
        return seq

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)
