import os
import random
import torch
import numpy as np

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"[Utils] Seed setada para {seed}")

def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"[Utils] Diretório criado: {path}")

def log(msg: str, prefix: str = "[LOG]"):
    print(f"{prefix} {msg}")

def clean_text(text: str) -> str:
    return " ".join(text.strip().split())

def tokens_to_text(tokens: list, tokenizer):
    return tokenizer.decode(tokens)


def text_to_tokens(text: str, tokenizer):
    return tokenizer.encode(text).ids

if __name__ == "__main__":
    set_seed(123)
    ensure_dir("./checkpoints")
    ensure_dir("./cache")
    log("Teste de logging")
    print(clean_text("  Olá   mundo!  "))
