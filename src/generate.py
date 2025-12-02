import torch
import torch.nn.functional as F
from src.model import GenerativeTransformer
from src.dataset import MovieDialogsDataset

class TextGenerator:
    def __init__(self, model_path: str, tokenizer_path: str = None, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        if tokenizer_path is None:
            raise ValueError("tokenizer_path é obrigatório")
        self.tokenizer = MovieDialogsDataset(data_path=None, tokenizer=None).tokenizer
        self.tokenizer = self.tokenizer.from_file(tokenizer_path)

        checkpoint = torch.load(model_path, map_location=self.device)
        vocab_size = checkpoint.get("vocab_size", len(self.tokenizer.get_vocab()))
        model_params = checkpoint.get("model_params", {"d_model":512,"num_layers":6,"num_heads":8,"d_ff":2048})
        self.model = GenerativeTransformer(vocab_size=vocab_size, **model_params).to(self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95
    ) -> str:
        x = self.tokenizer.encode(prompt).ids
        x = torch.tensor(x, dtype=torch.long, device=self.device).unsqueeze(0)

        generated = x.clone()

        for _ in range(max_new_tokens):
            logits = self.model(generated)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)

            if top_k > 0:
                topk_probs, topk_indices = torch.topk(probs, top_k)
                probs_zero = torch.zeros_like(probs)
                probs_zero.scatter_(1, topk_indices, topk_probs)
                probs = probs_zero
                probs = probs / probs.sum(dim=-1, keepdim=True)

            if top_p < 1.0:
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                mask = cumulative_probs > top_p
                mask[..., 1:] = mask[..., :-1].clone()
                mask[..., 0] = 0
                sorted_probs[mask] = 0.0
                sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
                probs = torch.zeros_like(probs).scatter_(1, sorted_indices, sorted_probs)

            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)

        generated_text = self.tokenizer.decode(generated[0].tolist())
        return generated_text

if __name__ == "__main__":
    generator = TextGenerator(
        model_path="./checkpoints/best_model.pt",
        tokenizer_path="./cache/tokenizer.json"
    )
    prompt = "Olá, como você está"
    out = generator.generate(prompt, max_new_tokens=50, temperature=0.8, top_k=40, top_p=0.9)
    print("=== Geração ===")
    print(out)
