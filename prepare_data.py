import os
from tqdm import tqdm

DATA_DIR = "data"
OUTPUT_FILE = os.path.join(DATA_DIR, "train.txt")

def load_movie_lines(path):
    lines = {}
    print("Lendo movie_lines.tsv ...")
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 5:
                line_id = parts[0]
                text = parts[-1]
                lines[line_id] = text
    print(f"{len(lines)} falas carregadas.")
    return lines

def parse_line_ids(s):
    s = s.strip()[1:-1]
    s = s.replace("'", "").replace('"', "")
    ids = s.split()
    return ids

def load_conversations(path):
    conversations = []
    print("💬 Lendo movie_conversations.tsv ...")
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 4:
                conv_str = parts[-1]
                conv_ids = parse_line_ids(conv_str)
                conversations.append(conv_ids)
    print(f"{len(conversations)} conversas carregadas.")
    return conversations


def build_pairs(lines_dict, conversations):
    pairs = []
    print("Gerando pares de diálogo ...")
    for conv in tqdm(conversations):
        for i in range(len(conv) - 1):
            q = lines_dict.get(conv[i])
            a = lines_dict.get(conv[i + 1])
            if q and a:
                pairs.append((q, a))
    print(f"{len(pairs)} pares criados.")
    return pairs


def save_pairs(pairs, output_file):
    print(f"Salvando dataset em {output_file} ...")
    with open(output_file, "w", encoding="utf-8") as f:
        for q, a in pairs:
            f.write(f"{q}\t{a}\n")
    print("Arquivo de treino criado com sucesso!")


def main():
    movie_lines_path = os.path.join(DATA_DIR, "movie_lines.tsv")
    movie_conversations_path = os.path.join(DATA_DIR, "movie_conversations.tsv")

    if not os.path.exists(movie_lines_path) or not os.path.exists(movie_conversations_path):
        print("Arquivos do dataset não encontrados em ./data/")
        return

    lines_dict = load_movie_lines(movie_lines_path)
    conversations = load_conversations(movie_conversations_path)
    pairs = build_pairs(lines_dict, conversations)
    save_pairs(pairs, OUTPUT_FILE)

    print(f"Pronto! Você pode treinar usando: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
