# -*- coding: utf-8 -*-
# @Time    :2022/5/9 19:59
# @Author  :Sang Jiajun
# @Email   :jiajun_sang@outlook.com
# @File    :train.py
from config import epochs, batch_size, learning_rate
import tensorflow as tf
from tqdm import tqdm
import tensorflow.keras.backend as K
from create_model import MyNER
from data_helper import train_data
from config import vocab_size, embedding_size, hidden_size, num_labels, dropout_rate
from tensorflow_addons.text.crf import crf_decode


model = MyNER(vocab_size, embedding_size, hidden_size, num_labels, dropout_rate)

class MyCRFMetric(tf.keras.metrics.Metric):
    def __init__(self, **kwargs):
        super(MyCRFMetric, self).__init__(name="my_crf_metric", **kwargs)
        self.TP = 0
        self.TP_FP = 0
        self.TP_TN = 0

    def update_state(self, y_true, y_pred, sample_weight=None):
        # 将y_true和p_pred展平，方便计算
        y_true = tf.cast(y_true, tf.int32)
        y_pred = tf.cast(y_pred, tf.int32)
        y_true = tf.reshape(y_true, (-1,))
        y_pred = tf.reshape(y_pred, (-1,))
        self.TP_TN += K.sum(tf.cast(K.greater(y_true, 0), tf.int32)).numpy()
        self.TP_FP += K.sum(tf.cast(K.greater(y_pred, 0), tf.int32)).numpy()
        self.TP += K.sum(tf.cast(tf.equal((y_true>0)|(y_pred>0), tf.equal(y_true, y_pred)), tf.int32)).numpy()# 这里不需要计算标签都为O的实体

    def result(self):
        precision = self.TP / self.TP_FP
        recall = self.TP / self.TP_TN
        if precision + recall:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0
        return precision, recall, f1

    def reset_states(self):
        self.TP = 0
        self.TP_FP = 0
        self.TP_TN = 0

optimizer = tf.keras.optimizers.Adagrad(learning_rate=learning_rate)
metric = MyCRFMetric()
for epoch in range(epochs):
    for step, batch in tqdm(train_data.shuffle(5000).batch(batch_size).enumerate()):
        inputs, attention_mask, input_lengths, label = batch
        with tf.GradientTape() as tape:
            logits, log_likelihood, transition_params = model(inputs,
                                                              input_lengths,
                                                              label,
                                                              training=True)
            loss = -tf.reduce_mean(log_likelihood)
            print(f"loss: {loss}")
        variable = model.trainable_variables
        # variable = [var for var in variable if "pooler" not in var.name]
        gradients = tape.gradient(loss, variable)
        optimizer.apply_gradients(zip(gradients, variable))
        # if step % 100 == 0 and step != 0:
        batch_pred_sequence, _ = crf_decode(logits, transition_params, input_lengths)
        metric.update_state(label, batch_pred_sequence)
        precision, recall, f1 = metric.result()
        print(f"precision: {precision}, recall: {recall}, f1: {f1}")
    # 这里添加验证数据进行验证
    metric.reset_states()



