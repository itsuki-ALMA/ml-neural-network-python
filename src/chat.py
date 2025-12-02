from src.generate import TextGenerator
from src.utils import set_seed, log

MODEL_PATH = "./checkpoints/best_model.pt"
TOKENIZER_PATH = "./cache/tokenizer.json"
MAX_NEW_TOKENS = 50
TEMPERATURE = 0.8
TOP_K = 40
TOP_P = 0.9
SEED = 42
MAX_TURNS = 5

set_seed(SEED)

log("🔧 Carregando modelo e tokenizer...")
generator = TextGenerator(model_path=MODEL_PATH, tokenizer_path=TOKENIZER_PATH)

conversation_history = []

log(f"Chat pronto! Digite 'sair' para encerrar. (Máx {MAX_TURNS} turnos mantidos)\n")

while True:
    user_input = input("Você: ")
    if user_input.strip().lower() in ["sair", "exit", "quit"]:
        print("Encerrando chat...")
        break

    conversation_history.append(f"Usuário: {user_input}")

    if len(conversation_history) > MAX_TURNS * 2:
        conversation_history = conversation_history[-MAX_TURNS*2:]

    prompt_text = "\n".join(conversation_history)

    resposta = generator.generate(
        prompt=prompt_text,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        top_k=TOP_K,
        top_p=TOP_P
    )

    print(f"Bot: {resposta}\n")
    conversation_history.append(f"Bot: {resposta}")
