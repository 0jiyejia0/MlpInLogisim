import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import cv2
import numpy as np

# 加载MNIST数据集
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# 定义一个函数将灰度图像转换为黑白图像
def convert_to_binary(images, threshold=128):
    binary_images = []
    for image in images:
        _, binary_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
        binary_images.append(binary_image)
    return np.array(binary_images)

# 将训练集和测试集的灰度图像转换为黑白图像
train_images_binary = convert_to_binary(train_images)
test_images_binary = convert_to_binary(test_images)

# 预处理数据
train_images_binary = train_images_binary.reshape((60000, 28 * 28)).astype('float32') / 255
test_images_binary = test_images_binary.reshape((10000, 28 * 28)).astype('float32') / 255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# 构建模型
model = models.Sequential()
model.add(layers.Dense(10, activation='relu', input_shape=(28 * 28,)))
model.add(layers.Dense(10, activation='softmax'))

# 编译模型
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 训练模型
model.fit(train_images_binary, train_labels, epochs=20, batch_size=128, validation_split=0.2)

# 评估模型
test_loss, test_acc = model.evaluate(test_images_binary, test_labels)
print('Test accuracy:', test_acc)

# 保存模型
model.save('mnist_binary_mlp_model_2.h5')
print('Model saved to mnist_binary_mlp_model_2.h5')

for i in range(28):
    for j in range(28):
        print(f'train_images_binary[{i}][{j}] = {train_images_binary[0][i*28+j]}',end=' ')
    print()