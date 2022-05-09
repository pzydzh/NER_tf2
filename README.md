# NER_tf2

A simple implementation of NER(Named-entity recognition) with tensorflow2 and tensorflow_addons.

在第一个版本中，我将会采用Bi-LSTM+CRF模型实现一个简单的CRF，并在MSRA命名实体识别数据集上运行代码，并给出实验结果。

在之后的版本中，我将会逐渐升级模型。例如会增加BERT层，即BERT+BiLSTM+CRF。同样的，我也会给出模型在MSRA数据集上的运行结果。

在模型不断优化的过程中，我会尝试添加一些数据增强的方法，添加对抗训练模块，以追求指标的提升。
