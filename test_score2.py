import torch
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-MiniLM-L6-v2')
q = model.encode("bat was flying in sky", convert_to_tensor=True)

def test(m1, m2):
    s1 = util.cos_sim(q, model.encode(m1, convert_to_tensor=True)).item()
    s2 = util.cos_sim(q, model.encode(m2, convert_to_tensor=True)).item()
    print(f"'{m1[:30]}...' -> {s1:.4f}")
    print(f"'{m2[:30]}...' -> {s2:.4f}")
    print("Winner:", "animal" if s1 > s2 else "sports")
    print("-" * 30)

test("animal", "sports equipment")
test("animal (fly, night, wings, cave, blood, vampire)", "sports equipment (baseball, hit, ball, swing, game, wood, player, strike)")
test("bat means animal", "bat means sports equipment")
test("The word bat means animal", "The word bat means sports equipment")
