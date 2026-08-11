import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
backend_dir = os.path.join(project_root, "backend")
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from app.services.intent_classifier import IntentClassifier

ic = IntentClassifier()
res = ic.classify_intent("Design a distributed Python ML pipeline...")

print(f"Text             : {res.text}")
print(f"Intent           : {res.intent}")
print(f"Top Similarity   : {res.top_similarity:.4f}")
print(f"Second Similarity: {res.second_similarity:.4f}")
print(f"Margin           : {res.margin:.4f}")
print(f"Margin Threshold : {ic.margin_threshold}")
print(f"Is Ambiguous     : {res.is_ambiguous}")
print("\nRanked Intents:")
for r in res.ranked_intents:
    print(f"  - {r.intent:<15}: {r.score:.4f}")
