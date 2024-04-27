import tensorflow as tf
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image 
import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import load_img
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.models import load_model
from keras.models import Sequential
from keras.applications import ResNet50
from keras.layers import Dense



def stand(df_colx):
    x = list(df_colx)
    x.remove(max(x))
    x.remove(min(x))
    mean = sum(x)/len(x)
    return mean




# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


ratings = pd.read_excel('./SCUT-FBP5500_v2/All_RatingsOnlyAsian.xlsx')

filenames = ratings.groupby('Filename').size().index.tolist()

labels = []

for filename in filenames:
    df = ratings[ratings['Filename'] == filename]
    # score = round(df['Rating'].mean(), 2)
    score = round(stand(df['Rating']), 2) #去除最高最低再平均
    labels.append({'Filename': filename, 'score': score})

labels_df = pd.DataFrame(labels)




img_width, img_height, channels = 350, 350, 3
sample_dir = './SCUT-FBP5500_v2/Images_onlyAsian/'
nb_samples = len(os.listdir(sample_dir))
input_shape = (img_width, img_height, channels)

x_total = np.empty((nb_samples, img_width, img_height, channels), dtype=np.float32)
y_total = np.empty((nb_samples, 1), dtype=np.float32)

for i, fn in enumerate(os.listdir(sample_dir)):
    img = load_img('%s/%s' % (sample_dir, fn))
    # img = tf.keras.utils.load_img('%s/%s' % (sample_dir, fn))
    x = tf.keras.utils.img_to_array(img).reshape(img_height, img_width, channels)
    x = x.astype('float32') / 255.
    y = labels_df[labels_df.Filename == fn].score.values
    y = y.astype('float32')
    x_total[i] = x
    y_total[i] = y


seed = 42
x_train_all, x_test, y_train_all, y_test = train_test_split(x_total, y_total, test_size=0.2, random_state=seed)
x_train, x_val, y_train, y_val = train_test_split(x_train_all, y_train_all, test_size=0.2, random_state=seed)


resnet = ResNet50(include_top=False, pooling='avg', input_shape=input_shape)
model = Sequential()
model.add(resnet)
model.add(Dense(1))
model.layers[0].trainable = False






filepath="./model/Only_Asian_std/{epoch:02d}-{val_loss:.2f}.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
reduce_learning_rate = ReduceLROnPlateau(monitor='loss',
                                         factor=0.1,
                                         patience=2,
                                         cooldown=2,
                                         min_lr=0.00001,
                                         verbose=1)
callback_list = [checkpoint, reduce_learning_rate]

model.layers[0].trainable = True
model.compile(loss='mse', optimizer='adam')


history = model.fit(x=x_train, 
                    y=y_train, 
                    batch_size=8,
                    epochs=30,
                    validation_data=(x_val, y_val),
                    callbacks=callback_list)
                    
