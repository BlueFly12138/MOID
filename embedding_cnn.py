import numpy as np
import pandas as pd
import tensorflow as tf
import csv
import re
import keras
import torch
import matplotlib.pyplot as plt
#写入csv文件中纪录
import random
import pandas as pd
from datetime import datetime

from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Dropout, Activation, Flatten, MaxPool1D, Conv1D
from keras.layers.embeddings import Embedding
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models.doc2vec import Doc2Vec
from sklearn.model_selection import StratifiedKFold

from tensorflow.python.keras.layers import Dense,Dropout,Activation,Flatten,Conv1D,MaxPooling1D,Conv2D,MaxPooling2D


""

train_pad = []
train_target = []
# 读取txt中的文本向量
with open('test_Embedding.txt', 'r') as f:
    # vectors = np.array([list(map(float, line.strip().split())) for line in f])
    for line in f:
        vector = np.array([[float(x)] for x in line.split()[:-1]])
        flag = line.split()[-1]
        # print(flag)
        # print(vector)
        # print(vector.shape)
        train_pad.append(vector)
        train_target.append(np.int64(flag))
train_target = np.array(train_target)
print(train_target)
input()
print("开始训练")
# 90%的样本用来训练，剩余10%用来测试
# X_train	划分的训练集数据
# X_test	划分的测试集数据
# y_train	划分的训练集标签
# y_test	划分的测试集标签
x_train, x_test, y_train, y_test = train_test_split(train_pad,
                                                  train_target,
                                                  test_size=0.1,
                                                  random_state=30)


def CNN():
    model = tf.keras.Sequential()
    model.add(Conv1D(filters=1, kernel_size=16, input_shape=(100, 1)))
    model.add(MaxPooling1D(pool_size=4))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(200, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    model.build(input_shape=(None,100, 1))
    # model.build(input_shape=(None,100, 1))
    '''	
    model.add(Conv2D(filters=16,
                     kernel_size=(2,2),
                     padding = 'same',
                     input_shape = (2,800,1),
                     activation = 'relu'))

    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Flatten())
    #建立隐蔽层
    model.add(Dense(128,activation='relu'))
    model.add(Dropout(0.5))
    #建立输出层
    model.add(Dense(4,activation='softmax'))
    '''
    print(model.summary())

    # 优化
    # sgd = tf.keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

    # 整合模型
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model

model = CNN()
x_train = np.stack([i.tolist() for i in x_train])
y_train = np.stack([i.tolist() for i in y_train])

history = model.fit(x_train,y_train, epochs=500,validation_split=0.2)# 训练集的20%用作验证集
# history = model.fit(np.array(x_train),np.array(y_train), epochs=100,validation_split=0.2)# 训练集的20%用作验证集
model.save('model_CNN_text')  # 生成模型文件 'my_model'


"CNN预测"
model = load_model('model_CNN_text')
x_test = np.stack([i.tolist() for i in x_test])

result = model.predict(x_test)
print(result)
print(np.argmax(result,axis=1))
score= model.evaluate(x_test,y_test,batch_size=30)
print(score)

# 绘制训练 & 验证的准确率值
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Valid'], loc='upper left')
plt.savefig('Valid_acc.png')
plt.show()

# 绘制训练 & 验证的损失值
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Valid'], loc='upper left')
plt.savefig('Valid_loss.png')
plt.show()

#创建record.csv纪录Epoch   Time  loss   accuracy  val_loss   val_accuracy
df = pd.DataFrame(columns=['epoch','time','loss','accuracy','val_loss','val_accuracy'])#列名
df.to_csv("./out/record.csv",index=False) #路径可以根据需要更改
for Epoch in range(500):#迭代500次
    time = "%s" % datetime.now()  # 获取当前时间
    epoch = Epoch
    # loss = "%f"% history.history('Loss')
    # accuracy = "%f" % accuracy
    # val_loss = "%f" % val_loss
    # val_accuracy = "%f" % val_accuracy

    loss = "%f" % history.history['loss'][epoch]
    accuracy = "%f" %  history.history['accuracy'][epoch]
    val_loss = "%f" %  history.history['val_loss'][epoch]
    val_accuracy = "%f" % history.history['val_accuracy'][epoch]
    list = [epoch, time, loss, accuracy, val_loss, val_accuracy]
    # 由于DataFrame是Pandas库中的一种数据结构，它类似excel，是一种二维表，所以需要将list以二维列表的形式转化为DataFrame
    data = pd.DataFrame([list])
    data.to_csv('./out/record.csv', mode='a', header=False, index=False)  # mode设为a,就可以向csv文件追加数据了

