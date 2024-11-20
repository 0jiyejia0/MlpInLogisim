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

# 显示第一个样本的黑白图片
plt.imshow(binary_image, cmap='gray')
plt.title(f'第一个样本的标签: {train_labels[0]}')
plt.show()

# 获取模型的权重和偏置
weights = model.get_weights()

# 定义比例因子
scale_factor = 100

# 将权重和偏置乘以比例因子并转换为整数
weights_int = [np.round(w * scale_factor).astype(int) for w in weights]

# 创建一个函数用于写入ROM文件
def write_rom_file(filename, data):
    with open(filename, 'w') as f:
        f.write('v2.0 raw\n')  # 写入文件头部
        for value in data:
            f.write(f'{value & 0xFFFF:04X}\n')

# 写入输入层到每个隐藏层神经元的权重
input_count = 784
hidden_neurons = 10
output_neurons = 10

for j in range(hidden_neurons):
    w1_data = weights_int[0][:, j].flatten()
    write_rom_file(f'rom_w1_{j}.txt', w1_data)

# 写入隐藏层的偏置
b1_data = weights_int[1].flatten()
write_rom_file('rom_b1.txt', b1_data)

# 写入隐藏层到每个输出层神经元的权重
for j in range(output_neurons):
    w2_data = weights_int[2][:, j].flatten()
    write_rom_file(f'rom_w2_{j}.txt', w2_data)

# 写入输出层的偏置
b2_data = weights_int[3].flatten()
write_rom_file('rom_b2.txt', b2_data)

print("ROM 初始化文件已生成：rom_w1_*.txt, rom_b1.txt, rom_w2_*.txt, rom_b2.txt")
