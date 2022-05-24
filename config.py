# -*- coding: utf-8 -*-
# @Time    :2022/5/9 19:05
# @Author  :Sang Jiajun
# @Email   :jiajun_sang@outlook.com
# @File    :config.py
train_data_path = "example_data/train.txt"
pretrained_model_path = "/Users/sang/workhome/pretrained_models/tf/chinese_roberta_L-2_H-128"
label2id = {"O": 0, "B-PER": 1, "I-PER": 2, "B-LOC": 3, "I-LOC": 4, "B-ORG": 5, "I-ORG": 6}

vocab_size = 21128
embedding_size = 128
hidden_size = 256
dropout_rate = 0.2
num_labels = 7

epochs = 5
batch_size = 8
learning_rate = 0.0001
