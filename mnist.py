import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

# 加载MNIST数据集
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 选择第一个数字图片
first_image = x_train[3]

# 将灰度图像转换为纯黑白图像
threshold = 128
binary_image = (first_image > threshold).astype(np.uint8)

# 将二进制图像数据保存为Logisim格式的文本文件，每行28位
for i in range(10):
    with open(f'mnist_binary_image_for_logisim_{i}.txt', 'w') as f:
        binary_image = (x_train[i] > threshold).astype(np.uint8)
        f.write('v2.0 raw\n')
        for row in binary_image:
            # 将每行的28个二进制数据拼接成一个整数
            int_value = 0
            for bit in row:
                int_value = (int_value << 1) | bit
            # 将整数写入文件，每个整数表示28位
            f.write(f"{int_value:07x}\n")

print("Binary image data saved to 'mnist_binary_image_for_logisim.txt'")
