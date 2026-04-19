from models.transformer_loader import get_models

text = "Artificial intelligence is a branch of computer science that aims to create intelligent machines. It has become an essential part of the technology industry. Research associated with artificial intelligence is highly technical and specialized. The core problems of artificial intelligence include programming computers for certain traits such as knowledge, reasoning, problem solving, perception, learning, planning, and the ability to manipulate and move objects."

models = get_models()
res = models.rewrite_text(text, 20)
print("--- OUTPUT ---")
print(res)
