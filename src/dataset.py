import os
import torch
from torch.utils.data import Dataset, random_split
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors
from tqdm import tqdm

class MovieDialogsDataset(Dataset):
    def __init__(self, train_file="data/train.txt", tokenizer=None, max_length=128, cache_dir="cache"):
        self.train_file = train_file
        self.max_length = max_length
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

        print(f"📖 Lendo {train_file} ...")
        self.pairs = self._load_pairs()
        self.texts = [f"{q} {a}" for q, a in self.pairs]

        if tokenizer is None:
            print("🧠 Treinando novo tokenizer...")
            self.tokenizer = self._train_tokenizer(self.texts)
        else:
            self.tokenizer = tokenizer

        print("💾 Tokenizando e criando cache...")
        self.encoded = self._encode_texts()

    def _load_pairs(self):
        pairs = []
        with open(self.train_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) == 2:
                    pairs.append((parts[0], parts[1]))
        print(f"✅ {len(pairs)} pares carregados.")
        return pairs

    def _train_tokenizer(self, texts):
        tokenizer = Tokenizer(models.BPE())
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        trainer = trainers.BpeTrainer(
            vocab_size=30000,
            special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]"]
        )
        tokenizer.train_from_iterator(texts, trainer=trainer)

        tokenizer.post_processor = processors.TemplateProcessing(
            single="[BOS] $A [EOS]",
            special_tokens=[
                ("[BOS]", tokenizer.token_to_id("[BOS]")),
                ("[EOS]", tokenizer.token_to_id("[EOS]")),
            ],
        )

        tokenizer.save(os.path.join(self.cache_dir, "tokenizer.json"))
        return tokenizer

    def _encode_texts(self):
        cache_file = os.path.join(self.cache_dir, "encoded.pt")

        if os.path.exists(cache_file):
            print("⚡ Cache encontrado, carregando dataset...")
            return torch.load(cache_file)

        encoded = []
        for text in tqdm(self.texts, desc="🔠 Tokenizando textos"):
            ids = self.tokenizer.encode(text).ids[: self.max_length]
            pad_id = self.tokenizer.token_to_id("[PAD]")
            ids += [pad_id] * (self.max_length - len(ids))
            encoded.append(ids)

        encoded = torch.tensor(encoded, dtype=torch.long)
        torch.save(encoded, cache_file)
        return encoded

    def __len__(self):
        return len(self.encoded)

    def __getitem__(self, idx):
        x = self.encoded[idx]
        y = x.clone()
        return {"input_ids": x, "labels": y}

    def split(self, train_ratio=0.7):
        total_len = len(self)
        train_size = int(train_ratio * total_len)
        val_size = total_len - train_size
        return random_split(self, [train_size, val_size])
