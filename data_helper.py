# -*- coding: utf-8 -*-
# @Time    :2022/5/9 19:05
# @Author  :Sang Jiajun
# @Email   :jiajun_sang@outlook.com
# @File    :data_helper.py
from config import train_data_path, label2id, pretrained_model_path
from transformers import AutoTokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf


def load_data(path, max_len = 500):
    """

    :param path:
    :return:
    """
    all_sentences, all_labels = [], []
    with open(path, "r", encoding="utf-8") as f:
        sentences = f.read().split("\n\n")
        for sentence in sentences:
            sent, labels = "", []
            for item in sentence.split("\n"):
                char, tag = item.split("\t")
                sent += char
                labels.append(label2id[tag])
                assert len(labels) == len(sent)
            if len(sent) > max_len:
                sent = sent[:max_len]
                labels = labels[:max_len]
            all_sentences.append(sent)
            all_labels.append(labels)
    return all_sentences, all_labels


def get_features(data):
    """
    将text转换成toke id。这里通过transformers中的tokenizer来实现
    :param data:
    :return:
    """
    sentences, labels = data
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path)
    token_ids = tokenizer(sentences)
    input_ids, token_type_ids, attention_mask = token_ids["input_ids"], token_ids["token_type_ids"], token_ids["attention_mask"]
    labels = [[0]+_+[0] for _ in labels]
    input_ids = pad_sequences(input_ids, padding="post")
    token_type_ids = pad_sequences(token_type_ids, padding="post")
    attention_mask = pad_sequences(attention_mask, padding="post")
    # input_lengths = tf.math.count_nonzero(attention_mask, 1) # 记录每个句子的长度
    input_lengths = attention_mask.sum(axis=(1))
    labels = pad_sequences(labels, padding="post")
    print(type(input_ids))
    print(type(attention_mask))
    print(type(input_lengths))
    print(type(labels))
    data = tf.data.Dataset.from_tensor_slices((input_ids, attention_mask, input_lengths, labels))

    return data


train_data = load_data(train_data_path)
train_data = get_features(train_data)


