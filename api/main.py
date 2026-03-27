import json
import sys
from pathlib import Path
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.rnn_model import RNNLanguageModel
from models.lstm_model import LSTMLanguageModel
from models.transformer_model import TransformerLanguageModel
from utils.vocab import Vocabulary
from utils.text import normalize_text
from utils.generation import generate_chat_response, build_chat_prompt

app = FastAPI(title="Chat comparativo IA profesional")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODELS = {}
VOCABS = {}
METAS = {}
HISTORY = {}


class ChatRequest(BaseModel):
    model: str
    message: str
    session_id: Optional[str] = "default"


class ChatResponse(BaseModel):
    model: str
    session_id: str
    prompt: str
    response: str


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
    if name in MODELS:
        return MODELS[name], VOCABS[name], METAS[name]

    config = CONFIGS[name]
    required = [config["weights"], config["vocab"], config["meta"]]
    if not all(path.exists() for path in required):
        raise FileNotFoundError(f"Faltan archivos para {name}. Entrena primero el modelo.")

    vocab = Vocabulary.load(str(config["vocab"]))
    with config["meta"].open("r", encoding="utf-8") as f:
        meta = json.load(f)
    model = config["builder"](len(vocab))
    state = torch.load(config["weights"], map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()

    MODELS[name] = model
    VOCABS[name] = vocab
    METAS[name] = meta
    return model, vocab, meta



def get_history(session_id: str):
    return HISTORY.get(session_id, [])[-2:]


@app.get("/")
def root():
    return {"message": "API para chat comparativo profesional con RNN, LSTM y Transformer"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    name = normalize_text(req.model)
    if name not in CONFIGS:
        raise HTTPException(status_code=400, detail="Modelo inválido. Usa rnn, lstm o transformer.")

    try:
        model, vocab, meta = load_model(name)
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))

    session_id = req.session_id or "default"
    message = normalize_text(req.message)
    history = get_history(session_id)
    prompt = build_chat_prompt(message=message, history=history)
    response_text = generate_chat_response(
        model=model,
        vocab=vocab,
        message=message,
        history=history,
        seq_len=int(meta.get("seq_len", 24)),
        max_new_tokens=20,
        device=DEVICE,
    )

    if not response_text:
        response_text = "no tengo una respuesta clara"

    HISTORY.setdefault(session_id, []).append({"user": message, "assistant": response_text})
    HISTORY[session_id] = HISTORY[session_id][-4:]

    return ChatResponse(model=name, session_id=session_id, prompt=prompt, response=response_text)
