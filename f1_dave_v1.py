#Requirement Library
#import cv2
#import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import tensorflow as tf
import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import rosbag
import math
import os
import time


#========================================================
# Functions
def wait_for_key():
    input("Press any key to continue...")
    print("Continuing...")

wait_for_key()
#========================================================
#Get Data
data_path = '../out.bag'
if(not os.path.exists(data_path)):
    print(f"out.bag doesn't exists in {data_path}")

#Preprocess
bag = rosbag.Bag(data_path)
lidar = []
servo = []

temp_cnt = 1
for topic, msg, t in bag.read_messages():
    if topic == 'Lidar':
        #ranges = [int(r*1000) for r in msg.ranges]
        ranges = msg.ranges
        #temp_cnt+=1

        # Remove quandrant of LIDAR directly behind us
        eighth = int(len(ranges)/8)
        ranges = np.array(ranges[eighth:-eighth])
        #print(len(ranges))
        
        lidar.append(ranges)
        #print(f'Topic: {topic}, type(msg): {type(ranges)}')

        #Augmentation
        #lidar.append(ranges[:-1]

    if topic == 'Ackermann':
        data = msg.drive.steering_angle
        servo.append(data)
        #print(f'Topic: {topic}, type(msg): {type(data)}')
        #print(f'Topic: {topic}, type(msg): {data}')
        #temp_cnt+=1
        #servo.append(-data)

#print(temp_cnt)
#print(f'Lidar 0th idx: {lidar[0]}')
#print(f'Servo 0th idx: {servo[0]}')

lidar = np.asarray(lidar)
servo = np.asarray(servo)
assert len(lidar) == len(servo)
print(f'Loaded {len(lidar)} samples')

wait_for_key()
#======================================================
# Split Dataset
print('Spliting Data to Train/Test')

x_train, x_test, y_train, y_test = train_test_split(lidar, servo, test_size = 0.35)
print(f'Train Size: {len(x_train)}')
print(f'Test Size: {len(x_test)}')
wait_for_key()
#======================================================
# DNN Arch
num_lidar_range_values = len(lidar[0])
print(f'num_lidar_range_values: {num_lidar_range_values}')

#Mess around with strides, kernel_size, Max Pooling
model = tf.keras.Sequential([
    tf.keras.layers.Conv1D(filters=24, kernel_size=5, strides=2, activation='relu', input_shape=(num_lidar_range_values, 1)),
    tf.keras.layers.Conv1D(filters=36, kernel_size=5, strides=2, activation='relu'),
    tf.keras.layers.Conv1D(filters=48, kernel_size=5, strides=2, activation='relu'),
    tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu'),
    tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu'),
    #tf.keras.layers.MaxPooling1D(pool_size=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dense(50, activation='relu'),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(1, activation='tanh')
])
#print(model.summary())


#======================================================
# Model Compilation
#lr = 3e-4
lr = 5e-5

optimizer = tf.keras.optimizers.Adam(lr)
model.compile(optimizer=optimizer, loss='huber')#, metrics = [r2]) #huber is noisy data else 'mean_squared_error'

print(model.summary())
wait_for_key()

#======================================================
# Model Fit
##See Data Balance in DeepPiCar

batch_size = 64
num_epochs = 20

start_time = time.time()
history = model.fit(x_train, y_train, epochs=num_epochs, batch_size=batch_size, validation_data=(x_test, y_test))

# Plot training and validation losses 
print(history.history.keys())
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

print(f'=============>{int(time.time() - start_time)} seconds<=============')

wait_for_key()


#======================================================
# Model Evaluation
test_loss = model.evaluate(x_test, y_test)
print(f'test_loss = {test_loss}')

y_pred = model.predict(x_test)
#accuracy = np.mean(pred_angle == y_test)
accuracy = r2_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.3f}')


#======================================================
# Save Model
model_file = 'f1_tenth_model'
model.save(model_file+'.h5')
print("Model Saved")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

quantized_tflite_model = converter.convert()

with open(model_file+".tflite", 'wb') as f:
    f.write(quantized_tflite_model)
    print (model_file+".tflite is saved. copy this file to the robot")
print('Tf_lite Model also saved')

#End
print('End')

