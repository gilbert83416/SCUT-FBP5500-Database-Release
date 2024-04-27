import tensorflow as tf
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image 
import face_recognition
from math import sqrt
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import img_to_array




# 將圖片根據眼距比例縮放
def resize_image(image_path, scale_factor):
    # 打開圖片
    img = Image.open(image_path)
    
    # 計算新的尺寸
    new_width = int(img.width * scale_factor)
    new_height = int(img.height * scale_factor)
    
    # 重設圖片尺寸
    resized_img = img.resize((new_width, new_height))
    
    return resized_img


# 裁切成350*350並補白底
def resize_and_pad(img_path, new_height, color=(255, 255, 255)):
    # 打開圖片
    img = Image.open(img_path)
    
    # 計算新寬度以保持比例
    original_height = img.height
    original_width = img.width
    aspect_ratio = original_width / original_height
    new_width = int(new_height * aspect_ratio)
    
    # 調整圖片大小
    resized_img = img.resize((new_width, new_height))
    
    # 計算新圖片的寬度以補全白底
    if new_width < 350:
        # 創建一個新的圖片，寬度為350，高度不變，背景為白色
        final_img = Image.new("RGB", (350, new_height), color)
        # 計算放置調整後圖片的起始x坐標（使其居中）
        x_offset = (350 - new_width) // 2
        # 將調整後的圖片貼到新的白色背景圖片上
        final_img.paste(resized_img, (x_offset, 0))
    else:
        final_img = resized_img

    return final_img



def preprocess_image(image_path, i):

    standard_eye_dist = 87.8405
    image = face_recognition.load_image_file(image_path)
    face_landmarks_list = face_recognition.face_landmarks(image)
    left_eye_x = [i[0] for i in face_landmarks_list[0]['left_eye']]
    left_eye_y = [i[1] for i in face_landmarks_list[0]['left_eye']]
    left_eye_x_avg = round(sum(left_eye_x)/len(left_eye_x))
    left_eye_y_avg = round(sum(left_eye_y)/len(left_eye_y))
    right_eye_x = [i[0] for i in face_landmarks_list[0]['right_eye']]
    right_eye_y = [i[1] for i in face_landmarks_list[0]['right_eye']]
    right_eye_x_avg = round(sum(right_eye_x)/len(right_eye_x))
    right_eye_y_avg = round(sum(right_eye_y)/len(right_eye_y))
    original_dist = sqrt((right_eye_x_avg - left_eye_x_avg)**2 + (right_eye_y_avg - left_eye_y_avg)**2)
    print('eye_dist: ', round(original_dist,2))
    scale_factor = standard_eye_dist/original_dist

    # 縮放圖片
    resized_image = resize_image(image_path, scale_factor)
    Preprocessed_path = path + 'Preprocessed' + folder_name + i
    resized_image.save(Preprocessed_path)

    # 再一次使用face_recognition套件找出臉部位置

    image = face_recognition.load_image_file(Preprocessed_path)
    face_locations = face_recognition.face_locations(image)
    upper, right, lower, left = face_locations[0]
    # print(upper, right, lower, left)


    image = Image.open(Preprocessed_path)
    width, height = image.size

    # print(f"Width: {width} pixels")
    # print(f"Height: {height} pixels")

    revised_width = width /6
    revised_height = height /6

    left = left - revised_width
    right = right + revised_width
    upper = upper - revised_height
    lower = lower + revised_height

    crop_area = (left, upper, right, lower)
    cropped_image = image.crop(crop_area)
    cropped_image.save(Preprocessed_path)

    # 裁切成350*350並補白底
    resized_padded_image = resize_and_pad(Preprocessed_path, 350)
    resized_padded_image.save(Preprocessed_path) 



def img_scoring(path, i, model):
    img = load_img(path+ 'Preprocessed'+folder_name + i)
    plt.imshow(img)

    img_width, img_height, channels = 350, 350, 3
    # 使用Pillow的resize方法调整图像大小
    img_resized = img.resize((img_width, img_height))

    # 将图像转换为数组并重新整理形状，归一化像素值
    img_array = img_to_array(img_resized)
    img_array = img_array.reshape((1, img_height, img_width, channels))
    img_array = img_array / 255.0

    # 使用模型进行预测
    # model = tf.keras.models.load_model('./model/AllData/25-0.12.h5')
    predict = model.predict(img_array)
    print('predict, ', predict[0][0])
    score = round(predict[0][0], 2)
    img.save(path + 'Scored'+ folder_name + str(score) +'_' + i)



path = './IMG/Original/'
imgs = [f for f in os.listdir(path+'IMG/')]
folder_name = 'AllData_std14/'

if not os.path.exists(path + 'Preprocessed'+ folder_name):
    os.makedirs(path + 'Preprocessed'+folder_name)
if not os.path.exists(path + 'Scored'+ folder_name):
    os.makedirs(path + 'Scored'+folder_name)


# load model
# MyModel = tf.keras.models.load_model('./model/AllData/25-0.12.h5') #AllData
# MyModel = tf.keras.models.load_model('./model/OnlyAsian/18-0.11.h5') #OnlyAsian
# MyModel = tf.keras.models.load_model('./model/AllData_std/26-0.12.h5') #AllData_std 26
# MyModel = tf.keras.models.load_model('./model/AllData_std/24-0.16.h5') #AllData_std 24
MyModel = tf.keras.models.load_model('./model/AllData_std/14-0.16.h5') #AllData_std 14
for i in imgs:
    print(i)
    try:
        preprocess_image(path+'IMG/' + i, i)
        print(i, 'is preprocessed')
        img_scoring(path , i, MyModel)
        print(i, 'is scored')
    except IndexError:
        print(i, 'is not preprocessed')
        continue
