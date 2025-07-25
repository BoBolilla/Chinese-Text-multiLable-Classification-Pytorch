import os
import torch
import numpy as np
import pickle as pkl
from tqdm import tqdm
import time
from datetime import timedelta
import json
from torch.utils.data import Dataset, DataLoader

MAX_VOCAB_SIZE = 10000  # 词表长度限制
UNK, PAD = '<UNK>', '<PAD>'  # 未知字，padding符号


def build_vocab(file_path, tokenizer, max_size, min_freq):
    vocab_dic = {}
    with open(file_path, 'r', encoding='UTF-8') as f:
        for line in tqdm(f):
            lin = line.strip()
            if not lin:
                continue
            content = json.loads(lin).get("content", "")
            for word in tokenizer(content):
                vocab_dic[word] = vocab_dic.get(word, 0) + 1
        vocab_list = sorted([_ for _ in vocab_dic.items() if _[1] >= min_freq],
                            key=lambda x: x[1], reverse=True)[:max_size]
        vocab_dic = {word_count[0]: idx for idx, word_count in enumerate(vocab_list)}
        vocab_dic.update({UNK: len(vocab_dic), PAD: len(vocab_dic) + 1})
    return vocab_dic


class TextDataset(Dataset):
    def __init__(self, path, vocab, config, tokenizer):
        self.pad_size = config.pad_size
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.num_classes = config.num_classes
        self.class_list = config.class_list
        self.samples = []

        with open(path, 'r', encoding='UTF-8') as f:
            for line in tqdm(f):
                data = json.loads(line.strip())
                if not data:
                    continue
                content = data['content']
                labels = data['label']

                token = self.tokenizer(content)
                seq_len = len(token)
                if len(token) < self.pad_size:
                    token.extend([PAD] * (self.pad_size - len(token)))
                else:
                    token = token[:self.pad_size]
                    seq_len = self.pad_size

                words_line = [vocab.get(word, vocab.get(UNK)) for word in token]
                label_vec = [0] * self.num_classes
                for label in labels:
                    if label in self.class_list:
                        label_vec[self.class_list.index(label)] = 1

                self.samples.append((words_line, label_vec, seq_len))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        words, label, seq_len = self.samples[index]
        return torch.LongTensor(words), torch.FloatTensor(label), torch.LongTensor(seq_len)


def tokenizer_word(text):
    return text.split(' ')

def tokenizer_char(text):
    return [ch for ch in text]


def build_dataset(config, use_word):
    tokenizer = tokenizer_word if use_word else tokenizer_char
    if os.path.exists(config.vocab_path):
        vocab = pkl.load(open(config.vocab_path, 'rb'))
    else:
        vocab = build_vocab(config.train_path, tokenizer=tokenizer, max_size=MAX_VOCAB_SIZE, min_freq=1)
        pkl.dump(vocab, open(config.vocab_path, 'wb'))
    print(f"Vocab size: {len(vocab)}")

    train_data = TextDataset(config.train_path, vocab, config, tokenizer)
    dev_data = TextDataset(config.dev_path, vocab, config, tokenizer)
    test_data = TextDataset(config.test_path, vocab, config, tokenizer)
    return vocab, train_data, dev_data, test_data


def build_iterator(dataset, config):
    return DataLoader(dataset=dataset, batch_size=config.batch_size, shuffle=True,
                      num_workers=4, collate_fn=collate_fn)


def collate_fn(batch):
    x = torch.stack([item[0] for item in batch])       # shape: (batch, seq_len)
    y = torch.stack([item[1] for item in batch])       # shape: (batch, num_classes)
    seq_len = torch.cat([item[2] for item in batch])   # shape: (batch,)
    return (x, seq_len), y


def get_time_dif(start_time):
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))
