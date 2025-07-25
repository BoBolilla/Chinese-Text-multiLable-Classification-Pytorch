import json

def load_texts(path):
    with open(path, 'r', encoding='utf-8') as f:
        return set(json.loads(line.strip())['content'] for line in f)

train_texts = load_texts("train.jsonl")
dev_texts = load_texts("dev.jsonl")

overlap = train_texts.intersection(dev_texts)
print(f"重复样本数量: {len(overlap)}")
