import streamlit as st
from tensorflow.keras.utils import load_img
from keras.models import load_model
import numpy as np

# 加载训练好的模型
model = load_model('C:/Users/qqp/Documents/zuoye/maogou.h5')

# 设置页面标题
st.title('猫狗分类器')

# 添加文件上传组件，用户可以上传待分类的图像
uploaded_file = st.file_uploader("请选择一张猫或狗的图片", type=["jpg", "jpeg", "png"])

# 如果用户上传了图像
if uploaded_file is not None:
    # 加载图像
    img = load_img(uploaded_file, target_size=(150, 150))
    # 将图像转化为numpy数组
    x = np.expand_dims(img, axis=0)
    # 对图像进行预处理
    x = x / 255.0
    # 使用模型进行预测
    prediction = model.predict(x)
    # 根据预测结果显示对应的标签
    if prediction < 0.5:
        st.write("这是一张猫的图片")
    else:
        st.write("这是一张狗的图片")
    # 显示上传的图像
    st.image(img, caption='Uploaded Image', use_column_width=True)
