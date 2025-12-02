import torch
import torch.nn as nn
from src.attention import MultiHeadAttention
from src.feedforward import FeedForward

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads, dropout=dropout)
        self.ff = FeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_res = x
        x = self.ln1(x)
        x = self.mha(x)
        x = self.dropout(x)
        x = x + x_res

        x_res = x
        x = self.ln2(x)
        x = self.ff(x)
        x = x + x_res

        return x

if __name__ == "__main__":
    batch_size = 2
    seq_len = 8
    d_model = 64
    num_heads = 4
    d_ff = 256

    x = torch.randn(batch_size, seq_len, d_model)
    block = TransformerBlock(d_model=d_model, num_heads=num_heads, d_ff=d_ff)
    out = block(x)
    print(out.shape)
