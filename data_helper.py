# -*- coding: utf-8 -*-
# @Time    :2022/5/9 19:05
# @Author  :Sang Jiajun
# @Email   :jiajun_sang@outlook.com
# @File    :data_helper.py
from config import train_data_path, label2id, pretrained_model_path
from transformers import AutoTokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def load_data(path):
    """

    :param path:
    :return:
    """
    all_sentences, all_labels = [], []
    with open(path, "r", encoding="utf-8") as f:
        sentences = f.read().split("\n\n")
        for sentence in sentences:
            tokens, tags = [], []
            for item in sentence.split("\n"):
                token, tag = item.split("\t")
                tokens.append(token)
                tags.append(label2id[tag])
            all_sentences.append("".join(tokens))
            all_labels.append(tags)
    return all_sentences, all_labels

def get_features(data, max_seq_len=128):
    """
    将text转换成toke id。这里通过transformers中的tokenizer来实现
    :param data:
    :return:
    """
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path)
    sentences, labels = data
    all_input_ids, all_attention_mask, all_labels = [], [], []
    max_len = 0
    for i in range(len(sentences)):
        label = labels[i]
        if len(sentences[i]) > max_len:
            max_len = len(sentences[i])
        max_len = min(max_len, max_seq_len)
        tokens = tokenizer.encode_plus(sentences[i])
        all_input_ids.append(tokens["input_ids"])
        all_attention_mask.append(tokens["attention_mask"])
        all_labels.append([0] + label + [0]) # 在label首尾各添加一个O标签，对应CLS和SEP符号
    all_input_ids = pad_sequences(all_input_ids, padding='post')
    all_attention_mask = pad_sequences(all_attention_mask, padding="post")
    all_labels = pad_sequences(all_labels, padding="post")
    return all_input_ids, all_attention_mask, all_labels


train_data = load_data(train_data_path)
train_data = get_features(train_data)

