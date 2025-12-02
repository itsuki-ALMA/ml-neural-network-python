import argparse
from src.train import Trainer
from src.generate import TextGenerator
from src.utils import set_seed, ensure_dir, log
import os

def main():
    parser = argparse.ArgumentParser(description="IA Generativa - Treino e Geração")

    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "generate"],
        required=True,
        help="Modo: train ou generate"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/",
        help="Pasta contendo train.txt"
    )
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints", help="Diretório de checkpoints")
    parser.add_argument("--cache_dir", type=str, default="./cache", help="Diretório de cache/tokenizer")
    parser.add_argument("--seed", type=int, default=42, help="Seed para reprodutibilidade")

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--d_ff", type=int, default=2048)

    parser.add_argument("--prompt", type=str, default="Olá, como você está")
    parser.add_argument("--max_new_tokens", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=40)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--model_path", type=str, default="./checkpoints/best_model.pt")
    parser.add_argument("--tokenizer_path", type=str, default="./cache/tokenizer.json")

    args = parser.parse_args()

    set_seed(args.seed)
    ensure_dir(args.checkpoint_dir)
    ensure_dir(args.cache_dir)

    if args.mode == "train":
        log("Iniciando treinamento...")
        log(f"Usando dataset: {os.path.join(args.data_path, 'train.txt')}")
        model_params = {
            "d_model": args.d_model,
            "num_layers": args.num_layers,
            "num_heads": args.num_heads,
            "d_ff": args.d_ff
        }
        trainer = Trainer(
            data_path=args.data_path,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            epochs=args.epochs,
            cache_dir=args.cache_dir,
            checkpoint_dir=args.checkpoint_dir,
            model_params=model_params
        )
        trainer.train()

    elif args.mode == "generate":
        log("Iniciando geração de texto...")
        generator = TextGenerator(
            model_path=args.model_path,
            tokenizer_path=args.tokenizer_path
        )
        out = generator.generate(
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p
        )
        print("\n=== Geração ===\n")
        print(out)

if __name__ == "__main__":
    main()
