import torch
import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


if __name__ == "__main__":
    batch_size = 2
    seq_len = 8
    d_model = 64
    d_ff = 256

    x = torch.randn(batch_size, seq_len, d_model)
    ff = FeedForward(d_model=d_model, d_ff=d_ff)
    out = ff(x)
    print(out.shape)
