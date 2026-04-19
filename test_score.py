import torch
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-MiniLM-L6-v2')

sentence = "bat was flying in sky"
word = "bat"

def test_prompt(meaning1, meaning2):
    query = model.encode(sentence, convert_to_tensor=True)
    
    p1 = f'Context Sentence: "{sentence}"\nAmbiguous Word: "{word}"\nCandidate Meaning: "{meaning1}"'
    p2 = f'Context Sentence: "{sentence}"\nAmbiguous Word: "{word}"\nCandidate Meaning: "{meaning2}"'
    
    s1 = util.cos_sim(query, model.encode(p1, convert_to_tensor=True)).item()
    s2 = util.cos_sim(query, model.encode(p2, convert_to_tensor=True)).item()
    
    print(f"'{meaning1[:30]}...' -> {s1:.4f}")
    print(f"'{meaning2[:30]}...' -> {s2:.4f}")
    print("Winner:", "1" if s1 > s2 else "2")
    print("-" * 30)

print("1. Just the manual keys:")
test_prompt("animal", "sports equipment")

print("2. Parenthesis keywords:")
test_prompt("animal (fly, night, wings, cave, blood, vampire)", "sports equipment (baseball, hit, ball, swing, game, wood, player, strike)")

print("3. Sentence keywords:")
test_prompt("animal. Keywords: fly, night, wings, cave, blood, vampire.", "sports equipment. Keywords: baseball, hit, ball, swing, game, wood, player, strike.")

print("4. Descriptive:")
test_prompt("a flying animal", "a wooden bat used in sports")

print("5. Just meanings (no context format):")
q = model.encode(sentence, convert_to_tensor=True)
s1 = util.cos_sim(q, model.encode("animal", convert_to_tensor=True)).item()
s2 = util.cos_sim(q, model.encode("sports equipment", convert_to_tensor=True)).item()
print(f"Raw 'animal': {s1:.4f} vs 'sports equipment': {s2:.4f}")
