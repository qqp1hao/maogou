import os # 导入os模块，用于操作文件和目录。
import streamlit as st # 导入streamlit模块，用于展示训练过程和结果。
import tensorflow as tf # 导入tensorflow模块，用于搭建和训练神经网络。
from keras.preprocessing.image import ImageDataGenerator # 导入ImageDataGenerator类，用于读取图像并进行数据增强。
from keras.models import Sequential # 导入Sequential类，用于搭建序列模型。
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense # 导入卷积层、池化层、随机失活层、全连接层等类。

# 设置路径和超参数
train_dir = 'D:/games/cats_and_dogs_small/train' # 训练集路径。
val_dir = 'D:/games/cats_and_dogs_small/validation' # 验证集路径。
input_shape = (150, 150, 3) # 输入形状，即图像大小和通道数。
batch_size = 20 # 批次大小。
epochs = 100 # 训练轮数。

# 用ImageDataGenerator读取图像
train_datagen = ImageDataGenerator( # 定义训练集数据增强器。
    rescale=1./255, # 将像素值缩放到0~1之间。
    rotation_range=40, # 随机旋转40度以内。
    width_shift_range=0.2, # 随机水平平移0.2倍以内。
    height_shift_range=0.2, # 随机垂直平移0.2倍以内。
    shear_range=0.2, # 随机错切变换0.2倍以内。
    zoom_range=0.2, # 随机缩放0.2倍以内。
    horizontal_flip=True) # 随机水平翻转。

val_datagen = ImageDataGenerator(rescale=1./255) # 定义验证集数据增强器。

train_gen = train_datagen.flow_from_directory( # 从目录中读取训练集图像并进行数据增强。
    train_dir,
    target_size=input_shape[:2], # 图像大小。
    batch_size=batch_size, # 批次大小。
    class_mode='binary') # 分类方式。

val_gen = val_datagen.flow_from_directory( # 从目录中读取验证集图像并进行数据增强。
    val_dir,
    target_size=input_shape[:2], # 图像大小。
    batch_size=batch_size, # 批次大小。
    class_mode='binary') # 分类方式。

# 导入 Sequential 类
from keras.models import Sequential
# 导入 Conv2D、MaxPooling2D、Dropout 和 Dense 层
from keras.layers import Conv2D, MaxPooling2D, Dropout, Dense

# 创建一个 Sequential 对象
model = Sequential([
    # 添加一个 2D 卷积层，32 个滤波器，每个滤波器大小为 3x3，使用 relu 激活函数，输入形状为 input_shape
    Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
    # 添加一个 2D 最大池化层，大小为 2x2
    MaxPooling2D((2,2)),
    # 添加一个 2D 卷积层，64 个滤波器，每个滤波器大小为 3x3，使用 relu 激活函数
    Conv2D(64, (3,3), activation='relu'),
    # 添加一个 2D 最大池化层，大小为 2x2
    MaxPooling2D((2,2)),
    # 添加一个 2D 卷积层，128 个滤波器，每个滤波器大小为 3x3，使用 relu 激活函数
    Conv2D(128, (3,3), activation='relu'),
    # 添加一个 2D 最大池化层，大小为 2x2
    MaxPooling2D((2,2)),
    # 添加一个 2D 卷积层，128 个滤波器，每个滤波器大小为 3x3，使用 relu 激活函数
    Conv2D(128, (3,3), activation='relu'),
    # 添加一个 2D 最大池化层，大小为 2x2
    MaxPooling2D((2,2)),
    # 将前面的卷积层输出的多维数据展平成一维
    Flatten(),
    # 添加一个 Dropout 层，随机失活率为 0.5
    Dropout(0.5),
    # 添加一个全连接层，512 个神经元，使用 relu 激活函数
    Dense(512, activation='relu'),
    # 添加一个全连接层，1 个神经元，使用 sigmoid 激活函数
    Dense(1, activation='sigmoid')
])
# 编译模型，指定损失函数、优化器和评估指标
model.compile(loss='binary_crossentropy',
              optimizer=tf.keras.optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])

# 在Streamlit中展示训练过程和结果
st.write('开始训练模型...')
# 使用 fit_generator() 方法训练模型，设置训练参数和数据来源
history = model.fit_generator(
    train_gen,
    steps_per_epoch=train_gen.samples // batch_size, # 每个 epoch 的迭代次数
    epochs=epochs, # 训练轮数
    validation_data=val_gen, # 验证集
    validation_steps=val_gen.samples // batch_size # 每个 epoch 验证集的迭代次数
)

st.write('模型训练完成！')

st.write('展示训练结果...')
# 在Streamlit中展示训练结果图
import matplotlib.pyplot as plt

acc = history.history['acc'] # 训练集准确率
val_acc = history.history['val_acc'] # 验证集准确率
loss = history.history['loss'] # 训练集损失
val_loss = history.history['val_loss'] # 验证集损失

epochs_range = range(epochs) # 训练轮数

# 创建一个包含两个子图的图形
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
# 在第一个子图中绘制训练集和验证集的准确率曲线
ax1.plot(epochs_range, acc, label='Training Accuracy')
ax1.plot(epochs_range, val_acc, label='Validation Accuracy')
ax1.legend(loc='lower right')
ax1.set_title('Training and Validation Accuracy')
# 在第二个子图中绘制训练集和验证集的损失曲线
ax2.plot(epochs_range, loss, label='Training Loss')
ax2.plot(epochs_range, val_loss, label='Validation Loss')
ax2.legend(loc='upper right')
ax2.set_title('Training and Validation Loss')

# 在 Streamlit 中展示图形
st.pyplot(fig)

# 保存模型
model.save('maogou.h5')
