import json

with open("ai_studio_code.txt") as f:
    data = json.load(f)

reviews = json.loads(data["response"])

with open("reviews_v1.json", "w", encoding="utf-8") as f:
    json.dump(reviews, f, ensure_ascii=False, indent=2)