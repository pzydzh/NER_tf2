# -*- coding: utf-8 -*-
# @Time    :2022/5/9 19:44
# @Author  :Sang Jiajun
# @Email   :jiajun_sang@outlook.com
# @File    :create_model.py
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow.keras.backend as K
from tensorflow_addons.text import crf_log_likelihood, crf_decode


class MyNER(tf.keras.Model):
    def __init__(self, vocab_size, embedding_size, hidden_size, num_labels, dropout_rate):
        super(MyNER, self).__init__()
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_size, mask_zero=True)
        self.bi_lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(hidden_size, return_sequences=True))
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.dense = tf.keras.layers.Dense(num_labels)
        self.transition_params = tf.Variable(tf.random.uniform(shape=(num_labels, num_labels)))

    @tf.function
    def call(self, inputs, attention_mask, label, training=None):
        output = self.embedding(inputs)
        output = self.dropout(output, training=training)
        output = self.bi_lstm(output)
        logits = self.dense(output)

        target_label = tf.convert_to_tensor(label, dtype=tf.int32)
        input_length = tf.math.count_nonzero(attention_mask, 1)
        log_likelihood, self.transition_params = crf_log_likelihood(logits,
                                                                    target_label,
                                                                    input_length,
                                                                    transition_params=self.transition_params)
        return logits, log_likelihood, self.transition_params

