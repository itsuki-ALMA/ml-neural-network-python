import os
import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from tqdm import tqdm
from src.dataset import MovieDialogsDataset
from src.model import GenerativeTransformer
from torch.cuda.amp import GradScaler, autocast
import multiprocessing


class Trainer:
    def __init__(
        self,
        data_path,
        tokenizer=None,
        model_params=None,
        batch_size=16,
        seq_len=128,
        epochs=10,
        lr=3e-4,
        device=None,
        cache_dir="./cache",
        checkpoint_dir="./checkpoints",
        mixed_precision=True,
        train_ratio=0.7,
        num_workers=None
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.cache_dir = cache_dir
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.mixed_precision = mixed_precision

        if num_workers is None:
            try:
                num_cpus = multiprocessing.cpu_count()
                self.num_workers = min(4, max(0, num_cpus - 1))
            except Exception:
                self.num_workers = 0
        else:
            self.num_workers = num_workers

        print("Preparando dataset...")
        train_file = os.path.join(data_path, "train.txt")
        if not os.path.exists(train_file):
            raise RuntimeError(f"Arquivo {train_file} não encontrado.")

        full_dataset = MovieDialogsDataset(
            train_file=train_file,
            tokenizer=tokenizer,
            max_length=seq_len,
            cache_dir=cache_dir
        )

        total_examples = len(full_dataset)
        if total_examples == 0:
            raise RuntimeError(
                "Dataset vazio (len==0). Verifique se train.txt existe e não está vazio, "
                "ou apague cache/ para reprocessar."
            )

        tokenizer_obj = getattr(full_dataset, "tokenizer", None)
        if tokenizer_obj is None:
            raise RuntimeError("Tokenizer não encontrado no dataset.")

        if hasattr(tokenizer_obj, "get_vocab_size"):
            self.vocab_size = tokenizer_obj.get_vocab_size()
        elif hasattr(tokenizer_obj, "get_vocab"):
            self.vocab_size = len(tokenizer_obj.get_vocab())
        else:
            try:
                vocab = tokenizer_obj.get_vocab()
                self.vocab_size = len(vocab)
            except Exception:
                raise RuntimeError("Não foi possível determinar vocab_size do tokenizer.")

        train_ds, val_ds = full_dataset.split(train_ratio=train_ratio)

        if len(train_ds) == 0 or len(val_ds) == 0:
            raise RuntimeError(
                f"Split inválido: train={len(train_ds)}, val={len(val_ds)}. "
                "Verifique se o dataset contém exemplos suficientes e o train_ratio."
            )

        self.train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=(self.device.startswith("cuda")),
            num_workers=self.num_workers
        )
        self.val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=(self.device.startswith("cuda")),
            num_workers=self.num_workers
        )

        print("Criando modelo...")
        model_params = model_params or {}
        self.model = GenerativeTransformer(vocab_size=self.vocab_size, **model_params).to(self.device)

        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr)

        pad_id = None
        try:
            pad_id = tokenizer_obj.token_to_id("[PAD]")
        except Exception:
            try:
                pad_id = tokenizer_obj.get_vocab().get("[PAD]", None)
            except Exception:
                pad_id = None

        if pad_id is None:
            self.loss_fn = nn.CrossEntropyLoss()
        else:
            self.loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id)

        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs)

        self.scaler = GradScaler(enabled=self.mixed_precision)

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        progress = tqdm(self.train_loader, desc=f"Treino Epoch {epoch+1}")
        for batch in progress:
            x = batch["input_ids"].to(self.device)
            y = batch["labels"].to(self.device)

            self.optimizer.zero_grad()
            with autocast(enabled=self.mixed_precision):
                logits = self.model(x)
                loss = self.loss_fn(logits.view(-1, self.vocab_size), y.view(-1))

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            avg = total_loss / (progress.n if progress.n > 0 else 1)
            progress.set_postfix({"loss": f"{avg:.4f}"})
        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def validate(self, epoch):
        self.model.eval()
        total_loss = 0.0
        for batch in tqdm(self.val_loader, desc=f"Validação Epoch {epoch+1}"):
            x = batch["input_ids"].to(self.device)
            y = batch["labels"].to(self.device)
            logits = self.model(x)
            loss = self.loss_fn(logits.view(-1, self.vocab_size), y.view(-1))
            total_loss += loss.item()
        avg_loss = total_loss / len(self.val_loader)
        print(f"Validação Loss: {avg_loss:.4f}")
        return avg_loss

    def save_checkpoint(self, epoch, best=False):
        path = os.path.join(self.checkpoint_dir, f"checkpoint_epoch{epoch+1}.pt")
        if best:
            path = os.path.join(self.checkpoint_dir, "best_model.pt")
        torch.save({
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "vocab_size": self.vocab_size
        }, path)
        print(f"Checkpoint salvo em {path}")

    def train(self):
        best_val_loss = float("inf")
        progress_epochs = tqdm(range(self.epochs), desc="Treinamento geral", unit="epoch")
        
        for epoch in progress_epochs:
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate(epoch)
            self.scheduler.step()
            self.save_checkpoint(epoch)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint(epoch, best=True)
            
            progress_epochs.set_postfix({
                "train_loss": f"{train_loss:.4f}",
                "val_loss": f"{val_loss:.4f}"
            })
            
        print("Treinamento concluído!")

if __name__ == "__main__":
    trainer = Trainer(
        data_path="data/",
        batch_size=16,
        seq_len=128,
        epochs=5,
        model_params={"d_model": 512, "num_layers": 6, "num_heads": 8, "d_ff": 2048}
    )
    trainer.train()
