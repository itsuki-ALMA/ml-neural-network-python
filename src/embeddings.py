import torch
import torch.nn as nn

class TokenAndPositionEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model
        self.max_len = max_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = x.size()
        if seq_len > self.max_len:
            raise ValueError(f"seq_len={seq_len} maior que max_len={self.max_len}")

        token_embeddings = self.token_emb(x)

        positions = torch.arange(0, seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        pos_embeddings = self.pos_emb(positions)

        embeddings = token_embeddings + pos_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings

if __name__ == "__main__":
    batch_size = 2
    seq_len = 10
    vocab_size = 5000
    d_model = 128

    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    emb_layer = TokenAndPositionEmbedding(vocab_size, d_model)
    out = emb_layer(x)
    print(out.shape)
