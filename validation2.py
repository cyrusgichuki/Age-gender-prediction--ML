import tensorflow as tf
import numpy as np 

import seaborn as sebrn
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as atlas

# data
(x_train, y_train), (_, _) = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype('float32') / 255
y_train = tf.keras.utils.to_categorical(y_train , num_classes=10)

# model 
input = tf.keras.Input(shape=(28, 28))
x = tf.keras.layers.Flatten(input_shape=(28, 28))(input)
x = tf.keras.layers.Dense(128, activation='relu')(x)
output = tf.keras.layers.Dense(units=10, activation='softmax')(x) 
func_model = tf.keras.Model(input, output)

# compile 
func_model.compile(
          loss      = tf.keras.losses.CategoricalCrossentropy(),
          metrics   = tf.keras.metrics.CategoricalAccuracy(),
          optimizer = tf.keras.optimizers.Adam())

def model_train(model):
    # history is now a local variable 
    history = model.fit(x_train, y_train, 
                    batch_size=512, epochs=1, verbose = 2)
    print(history.history.keys()) # will print

# run the model 
model_train(func_model)

# try to access from outside
# but will get error 


func_model.fit(x_train, y_train, 
               batch_size=256, 
               epochs=5, verbose = 2, 
               callbacks=[tf.keras.callbacks.CSVLogger('age gender train.csv')])

import pandas
his = pandas.read_csv('age gender train.csv') 
his.head()

import matplotlib.pyplot as plt

plt.figure(figsize=(19,6))



y_true = ["16", "23", "15", "45", "4", "6"]
y_pred = ["F", "F", "M", "M", "M", "F"]
conf_matrix = (confusion_matrix(y_true, y_pred, labels=["age", "gender"]))

# Using Seaborn heatmap to create the plot
fx = sebrn.heatmap(conf_matrix, annot=True, cmap='turbo')

# labels the title and x, y axis of plot
fx.set_title('gender age Confusion Matrix using Seaborn\n\n');
fx.set_xlabel('Predicted gender age')
fx.set_ylabel('Actual gender age ');

# labels the boxes
fx.xaxis.set_ticklabels(['False','True'])
fx.yaxis.set_ticklabels(['False','True'])

atlas.show()
