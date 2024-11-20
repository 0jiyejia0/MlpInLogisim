import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 加载训练好的模型
model = tf.keras.models.load_model('mnist_binary_mlp_model_2.h5')

# 加载 MNIST 数据集
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

# 获取第一个样本并进行预处理
input_image = train_images[0]

# 将灰度图像转换为黑白图像
threshold = 128
binary_image = (input_image > threshold).astype(int)

# 展平并归一化处理
input_image_flattened = binary_image.reshape((1, 28 * 28)).astype('float32')

# 打印第一个样本的标签
print("第一个样本的标签:", train_labels[0])

# 获取模型的权重和偏置
weights = model.get_weights()

# 定义比例因子
scale_factor = 100

# 将权重和偏置乘以比例因子并转换为整数
weights_int = [np.round(w * scale_factor).astype(int) for w in weights]

# 选择一个隐藏层神经元计算其输出
input_image_int = np.round(input_image_flattened * scale_factor).astype(int)
hidden_neuron_index = 1  # 第二个隐藏层神经元

# 获取对应的权重和偏置
W_1_1 = weights_int[0][:, hidden_neuron_index]
b_1_1 = weights_int[1][hidden_neuron_index]

# 初始化加权和
Z_1_1 = b_1_1
print(f"初始偏置: {Z_1_1}")

# 逐步计算加权和，并输出每次加法后的结果
for i in range(len(input_image_int[0])):
    if input_image_int[0][i] == 100:
        input_image_int[0][i] = 1
    product = input_image_int[0][i] * W_1_1[i]
    Z_1_1 += product
    print(f"加法 {i + 1}: 输入值 = {input_image_int[0][i]}, 权重 = {W_1_1[i]}, 乘积 = {product}, 加权和 = {Z_1_1}")

# 应用 ReLU 激活函数
A_1_1 = max(0, Z_1_1)

# 打印第二个隐藏层神经元的数值，以十进制和十六进制表示
print(f"第二个隐藏层神经元的数值（十进制）: {A_1_1}")
print(f"第二个隐藏层神经元的数值（十六进制）: {int(A_1_1):X}")

# 输出层计算
output_neuron_index = 0  # 第二个输出层神经元

# 获取对应的权重和偏置
W_2_1 = weights_int[2][:, output_neuron_index]
b_2_1 = weights_int[3][output_neuron_index]

# 初始化加权和
Z_2_1 = b_2_1
print(f"初始偏置（输出层神经元）: {Z_2_1}")

# 逐步计算加权和，并输出每次加法后的结果
for i in range(len(W_2_1)):
    if input_image_int[0][i] == 100:
        input_image_int[0][i] = 1
    hidden_layer_output = A_1_1 if i == hidden_neuron_index else max(0, np.dot(input_image_int, weights_int[0][:, i]) + weights_int[1][i])
    product = hidden_layer_output * W_2_1[i]
    Z_2_1 += product
    print(f"加法 {i + 1}: 隐藏层输出 = {hidden_layer_output}, 权重 = {W_2_1[i]}, 乘积 = {product}, 加权和 = {Z_2_1}")

# 应用 ReLU 激活函数
A_2_1 = max(0, Z_2_1)

# 打印第二个输出层神经元的数值，以十进制和十六进制表示
print(f"第二个输出层神经元的数值（十进制）: {A_2_1}")
print(f"第二个输出层神经元的数值（十六进制）: {int(A_2_1):X}")
