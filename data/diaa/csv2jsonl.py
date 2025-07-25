import csv
import json
import re
from bs4 import BeautifulSoup


def clean_text(text):
    # 使用BeautifulSoup去除HTML标签
    soup = BeautifulSoup(text, 'html.parser')
    text = soup.get_text()

    # 去除$$符号及其内容
    text = re.sub(r'\$\$', '', text)
    text = re.sub(r'\\\\', '', text)
    text = re.sub(r'\\\\', '', text)

    # 去除其他特殊字符和多余空格
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def csv_to_jsonl(input_csv, output_jsonl):
    with open(input_csv, 'r', encoding='utf-8') as csv_file, \
            open(output_jsonl, 'w', encoding='utf-8') as jsonl_file:
        reader = csv.DictReader(csv_file)

        for row in reader:
            content = clean_text(row['content'])
            label = row['label'].strip()

            # 如果label是逗号分隔的多个标签，可以拆分为列表
            labels = [l.strip() for l in label.split(',')]

            json_record = {
                "content": content,
                "label": labels
            }

            jsonl_file.write(json.dumps(json_record, ensure_ascii=False) + '\n')


# 使用示例
input_csv = 'data/test.csv'  # 替换为你的CSV文件路径
output_jsonl = 'data/test.jsonl'  # 输出JSONL文件路径
input_csv1 = 'cut_data/test.csv'  # 替换为你的CSV文件路径
output_jsonl1= 'cut_data/test.jsonl'  # 输出JSONL文件路径
csv_to_jsonl(input_csv, output_jsonl)
csv_to_jsonl(input_csv1, output_jsonl1)