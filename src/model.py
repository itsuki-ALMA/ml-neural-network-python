import torch
import torch.nn as nn
from src.embeddings import TokenAndPositionEmbedding
from src.transformer_block import TransformerBlock

class GenerativeTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        d_ff: int = 2048,
        max_len: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size

        self.embeddings = TokenAndPositionEmbedding(vocab_size=vocab_size, d_model=d_model, max_len=max_len, dropout=dropout)

        self.layers = nn.ModuleList([
            TransformerBlock(d_model=d_model, num_heads=num_heads, d_ff=d_ff, dropout=dropout)
            for _ in range(num_layers)
        ])

        self.ln_f = nn.LayerNorm(d_model)

        self.head = nn.Linear(d_model, vocab_size, bias=False)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embeddings(x)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits


if __name__ == "__main__":
    batch_size = 2
    seq_len = 16
    vocab_size = 5000
    d_model = 128
    num_layers = 2
    num_heads = 4
    d_ff = 512

    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    model = GenerativeTransformer(vocab_size=vocab_size, d_model=d_model, num_layers=num_layers,
                                  num_heads=num_heads, d_ff=d_ff, max_len=seq_len)
    out = model(x)
    print(out.shape)
