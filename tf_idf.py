import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import make_pipeline
from metrics import multi_class_evaluate

# 1. 读取数据
def load_data(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

dataset = './data/diaa'

train_data = load_data(dataset+'/data/train.jsonl')
dev_data = load_data(dataset+'/data/dev.jsonl')
test_data = load_data(dataset+'/data/test.jsonl')

# 2. 提取文本和标签
X_train = [item['content'] for item in train_data]
y_train = [item['label'] for item in train_data]

train_data.extend(dev_data)

X_test = [item['content'] for item in test_data]
y_test = [item['label'] for item in test_data]

# 3. 标签二值化
mlb = MultiLabelBinarizer()
y_train_bin = mlb.fit_transform(y_train)
y_test_bin = mlb.transform(y_test)

# 4. 构建TF-IDF + 多标签分类模型（以逻辑回归为例）
model = make_pipeline(
    TfidfVectorizer(max_features=10000),
    OneVsRestClassifier(LogisticRegression(solver='liblinear'))
)

# 5. 训练模型
model.fit(X_train, y_train_bin)

# 6. 预测概率得分
y_scores = model.predict_proba(X_test)

# 7. 指标评估（top@k）
k_metrics = multi_class_evaluate(y_test_bin, y_scores, max_k=3)

# 8. 输出结果
def print_metrics(metrics):
    for i, k in enumerate([1, 2, 3]):
        print(f"\n=== Metrics @ {k} ===")
        print(f"Accuracy@{k}: {metrics['acc'][i]*100:.2f}%")
        print(f"Precision@{k}: {metrics['precision'][i]*100:.2f}%")
        print(f"Recall@{k}: {metrics['recall'][i]*100:.2f}%")
        print(f"F1@{k}: {metrics['f1'][i]*100:.2f}%")

print_metrics(k_metrics)
