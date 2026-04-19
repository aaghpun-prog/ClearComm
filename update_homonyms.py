import os
import re

file_path = r'd:\College Work\Sem 4\SE\ClearComm-main\ClearComm-main\modules\homonym_detector.py'

with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Replace HOMONYM_DICT
new_dict = """HOMONYM_DICT = {
    "bank": {
        "financial": {"definition": "A financial institution that accepts deposits.", "keywords": ["financial", "institution", "money", "cash", "deposit", "loan"], "example": "He deposited a check at the bank."},
        "river": {"definition": "Sloping land, especially beside a river.", "keywords": ["river", "edge", "water", "stream", "fishing", "mud", "shore", "sat"], "example": "The river bank was muddy."}
    },
    "bat": {
        "animal": {"definition": "A mouselike flying mammal.", "keywords": ["animal", "mammal", "fly", "flying", "night", "wings", "cave"], "example": "A bat flew at night."},
        "sports": {"definition": "A club used for hitting a ball in sports.", "keywords": ["sports", "equipment", "baseball", "hit", "ball", "swing", "game"], "example": "He hit the ball with a bat."}
    },
    "match": {
        "contest": {"definition": "A game or contest.", "keywords": ["game", "contest", "win", "lose", "football", "tennis", "score"], "example": "The football match was tied."},
        "fire": {"definition": "A stick tipped with combustible chemical for starting fire.", "keywords": ["fire", "starter", "burn", "light", "strike", "flame"], "example": "He struck a match to light the fire."}
    },
    "spring": {
        "season": {"definition": "The season of growth following winter.", "keywords": ["season", "summer", "bloom", "flowers", "warm"], "example": "Flowers bloom in the spring."},
        "coil": {"definition": "An elastic device, typically a metal coil.", "keywords": ["coil", "metal", "bounce", "bed", "mattress", "jump"], "example": "The mattress springs are broken."},
        "water": {"definition": "A natural flow of ground water.", "keywords": ["water", "source", "drink", "fresh", "mountain"], "example": "Fresh water from the spring."}
    },
    "light": {
        "illumination": {"definition": "Natural agent that makes things visible.", "keywords": ["bright", "sun", "shine", "dark", "see", "lamp", "bulb"], "example": "Turn on the light."},
        "weight": {"definition": "Of little weight; not heavy.", "keywords": ["heavy", "feather", "carry", "weight", "easy"], "example": "The box is light to carry."}
    },
    "left": {
        "direction": {"definition": "On or towards the side of the body to the west when facing north.", "keywords": ["right", "turn", "direction", "side", "hand"], "example": "Turn left at the signal."},
        "departed": {"definition": "Went away from a place.", "keywords": ["go", "leave", "depart", "office", "home", "early", "went"], "example": "She left the office early."}
    },
    "watch": {
        "observe": {"definition": "Look at or observe attentively.", "keywords": ["look", "see", "movie", "bird", "screen", "observe"], "example": "Watch the movie carefully."},
        "timepiece": {"definition": "A small timepiece worn typically on a strap on one's wrist.", "keywords": ["wrist", "time", "clock", "wear", "gold", "strap"], "example": "I wore my new watch today."}
    },
    "ring": {
        "jewelry": {"definition": "A small circular band, typically of precious metal, worn on a finger.", "keywords": ["finger", "gold", "diamond", "wear", "jewelry", "wedding"], "example": "He wore a gold ring."},
        "sound": {"definition": "Make a clear resonant or vibrating sound.", "keywords": ["bell", "phone", "sound", "hear", "doorbell", "call"], "example": "I heard the phone ring."}
    },
    "file": {
        "document": {"definition": "A folder or box for holding loose papers arranged in order.", "keywords": ["paper", "document", "cabinet", "folder", "computer", "data"], "example": "Please file the papers in the cabinet."},
        "tool": {"definition": "A tool with a roughened surface used for smoothing.", "keywords": ["tool", "metal", "smooth", "edge", "nail", "wood", "shape"], "example": "Use a file to smooth the sharp edge."}
    },
    "seal": {
        "animal": {"definition": "A fish-eating aquatic mammal.", "keywords": ["animal", "ocean", "swim", "fish", "bark", "water", "ice"], "example": "The seal was clapping its flippers."},
        "closure": {"definition": "A device or substance used to join two things to prevent coming apart.", "keywords": ["close", "envelope", "wax", "tight", "door", "container", "leak"], "example": "Please break the seal to open the letter."}
    },
    "key": {
        "lock": {"definition": "A shaped piece of metal used to open a lock.", "keywords": ["door", "open", "lock", "metal", "car", "start"], "example": "I lost the house key."},
        "crucial": {"definition": "Of crucial importance.", "keywords": ["important", "crucial", "success", "element", "factor", "main"], "example": "Communication is key to winning."}
    },
    "park": {
        "recreation": {"definition": "A large public green area in a town.", "keywords": ["green", "grass", "play", "children", "tree", "walk", "picnic"], "example": "Children played in the park."},
        "vehicle": {"definition": "Bring a vehicle to a halt and leave it temporarily.", "keywords": ["car", "vehicle", "drive", "lot", "garage", "outside", "space"], "example": "Please park the car outside."}
    },
    "duck": {
        "animal": {"definition": "A waterbird with a broad blunt bill.", "keywords": ["bird", "water", "quack", "pond", "feather", "animal"], "example": "The duck swam in the pond."},
        "action": {"definition": "Lower the head or the body quickly to avoid a blow.", "keywords": ["dodge", "head", "avoid", "low", "hit", "hide", "down"], "example": "You need to duck down to fit through the door."}
    },
    "current": {
        "time": {"definition": "Belonging to the present time.", "keywords": ["now", "present", "event", "affair", "today", "modern"], "example": "Current events are complicated."},
        "flow": {"definition": "A body of water or air moving in a definite direction.", "keywords": ["water", "river", "ocean", "flow", "electric", "air", "strong"], "example": "The ocean current was very strong today."}
    },
    "crane": {
        "bird": {"definition": "A tall, long-legged, long-necked bird.", "keywords": ["bird", "fly", "animal", "neck", "water", "tall"], "example": "The crane flew over the lake."},
        "machine": {"definition": "A large, tall machine used for moving heavy objects.", "keywords": ["machine", "heavy", "lift", "build", "construction", "raise", "weight"], "example": "The construction crane lifted the steel beams."}
    },
    "right": {
        "direction": {"definition": "On or toward the side of the body to the east when facing north.", "keywords": ["left", "turn", "direction", "side", "hand"], "example": "Take a right turn at the intersection."},
        "correct": {"definition": "Morally good, justified, or acceptable; true or correct.", "keywords": ["correct", "true", "wrong", "answer", "moral", "just"], "example": "You found the right answer."}
    }
}"""

content = re.sub(r'HOMONYM_DICT = \{.*?\n\}\n', new_dict + '\n', content, flags=re.DOTALL)

# 2. Replace Thresholds and Blacklist
old_settings = """# Precision settings
VERB_BLACKLIST = {"go", "do", "make", "get", "set", "run", "went", "is", "was", "are", "were"}

# Dual-threshold Strategy for Precision vs Demo Reliability
CURATED_THRESHOLD = 0.35
CURATED_GAP = 0.02

GENERAL_THRESHOLD = 0.55
GENERAL_GAP = 0.12"""

new_settings = """# Precision settings
VERB_BLACKLIST = {"go", "do", "make", "get", "set", "run", "went", "is", "was", "are", "were", "has", "have", "had", "be", "been", "being", "will", "would", "shall", "should", "can", "could", "did", "does"}

# Dual-threshold Strategy for Precision vs Demo Reliability
CURATED_THRESHOLD = 0.25
CURATED_GAP = 0.01

GENERAL_THRESHOLD = 0.40
GENERAL_GAP = 0.04"""
content = content.replace(old_settings, new_settings)

# 3. Replace POS Filtering in analyze_homonyms_sbert_pipeline
old_loop = """    for item in tokens_to_process:
        word = item["word"]
        pos = item["pos"]
        
        # 1. POS Filtering: Only Nouns and Adjectives
        if pos not in ["NOUN", "PROPN", "ADJ"]:
            continue
            
        # 2. Verb Blacklist (and common short words)
        if word in VERB_BLACKLIST or len(word) < 3 or word in seen_words:
            continue
            
        # 3. Use WordNet to check for lexical ambiguity
        synsets = wn.synsets(word)
        is_curated = word in HOMONYM_DICT
        
        if is_curated or len(synsets) > 1:"""

new_loop = """    for item in tokens_to_process:
        word = item["word"]
        pos = item["pos"]
        is_curated = word in HOMONYM_DICT
        
        # 1. POS Filtering: Allow NOUN, PROPN, ADJ, VERB. Always allow Curated words to bypass POS filter if missed.
        if pos not in ["NOUN", "PROPN", "ADJ", "VERB"] and not is_curated:
            continue
            
        # 2. Verb Blacklist (and common short words)
        if word in VERB_BLACKLIST or len(word) < 3 or word in seen_words:
            continue
            
        # 3. Use WordNet to check for lexical ambiguity
        synsets = wn.synsets(word)
        
        if is_curated or len(synsets) > 1:"""
content = content.replace(old_loop, new_loop)

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)

print("Updates written successfully.")
