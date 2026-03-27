import torch


def generate_text(model, vocab, prompt_ids, seq_len: int = 24, max_new_tokens: int = 20, device: str = "cpu"):
    model.eval()
    generated = prompt_ids[:]

    with torch.no_grad():
        for _ in range(max_new_tokens):
            x_ids = generated[-seq_len:]
            x = torch.tensor([x_ids], dtype=torch.long, device=device)
            logits = model(x)
            next_token_logits = logits[0, -1]
            next_id = int(torch.argmax(next_token_logits).item())
            generated.append(next_id)
            if next_id == vocab.eos_id:
                break

    return generated



def build_chat_prompt(message: str, history=None):
    turns = []
    history = history or []
    for item in history[-2:]:
        user = item.get("user", "").strip()
        assistant = item.get("assistant", "").strip()
        if user and assistant:
            turns.append(f"usuario: {user} asistente: {assistant}")
    turns.append(f"usuario: {message.strip()} asistente:")
    return " ".join(turns).strip()



def extract_assistant_reply(full_decoded: str):
    if "asistente:" not in full_decoded:
        return full_decoded.strip()
    reply = full_decoded.rsplit("asistente:", 1)[-1].strip()
    if "usuario:" in reply:
        reply = reply.split("usuario:", 1)[0].strip()
    return reply.strip()



def generate_chat_response(model, vocab, message: str, history=None, seq_len: int = 24, max_new_tokens: int = 20, device: str = "cpu"):
    prompt = build_chat_prompt(message=message, history=history)
    prompt_ids = vocab.encode(prompt, add_bos=True, add_eos=False)
    output_ids = generate_text(
        model=model,
        vocab=vocab,
        prompt_ids=prompt_ids,
        seq_len=seq_len,
        max_new_tokens=max_new_tokens,
        device=device,
    )
    decoded = vocab.decode(output_ids)
    return extract_assistant_reply(decoded)
