import cv2
import os
import time
import numpy as np
import sys
import math
import rospy
import message_filters
import tensorflow as tf
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped



lid = '/scan_filtered' #'/scan'
#===================================================
def callback(l):
    global lidar_data
    ldata = l.ranges
    eighth = int(len(ldata)/8)
    ldata = np.array(ldata[eighth:-eighth]).astype(np.float32)
    ldata = np.expand_dims(ldata, axis=-1)
    ldata = np.expand_dims(ldata, axis=0)
    lidar_data = ldata

#===================================================
def load_model():
    global interpreter
    global input_index
    global output_index
    global model

    print("Model")    
    model_name = './f1_tenth_model'
    model = tf.keras.models.load_model(model_name+'.h5')
    #try:
    interpreter = tf.lite.Interpreter(model_path=model_name+'.tflite')#,num_threads = args.ncpu)
    #except ImportError:
    #    print(f'Error in importing model: {ImportError}')
    interpreter.allocate_tensors()
    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]

def dnn_output():
    global lidar_data
    if lidar_data is None:
        return 0.
    ##lidar_data = np.expand_dims(lidar_data).astype(np.float32)
    interpreter.set_tensor(input_index,lidar_data)
    interpreter.invoke()
    servo = interpreter.get_tensor(output_index)[0][0]
    
    #servo = model.predict(lidar_data)
    #dur = time.time() - ts
    #if dur > period:
    #    print("%.3f: took %d ms - deadline miss."% (ts - start_ts, int(dur * 1000)))
    #else:
    #    print("%.3f: took %d ms" % (ts - start_ts, int(dur * 1000)))

    #if servo < 0.15:
    #    servo = 0.15
    #if servo > 0.85:
    #    servo = 0.85
    print(servo)

    return servo
#===================================================
rospy.init_node('Autonomous')
servo_pub = rospy.Publisher('/vesc/low_level/ackermann_cmd_mux/input/teleop',AckermannDriveStamped, queue_size=10)
rospy.Subscriber(lid, LaserScan, callback)
hz = 50
rate = rospy.Rate(hz)
period = 1.0/hz
start_ts = time.time()
load_model()
#lid_sub = message_filters.Subscriber(lid,LaserScan)
while not rospy.is_shutdown():
    ts = time.time()
    msg = AckermannDriveStamped()
    msg.header.stamp = rospy.Time.now()
    msg.header.frame_id = "base_link"
    msg.drive.speed = 5.0 # set speed to 1 m/s
    msg.drive.steering_angle = dnn_output() # set steering angle to 0.5 radians
    dur = time.time() - ts
    if dur > period:
        print("%.3f: took %d ms - deadline miss."% (ts - start_ts, int(dur * 1000)))
    else:
        print("%.3f: took %d ms" % (ts - start_ts, int(dur * 1000)))

    servo_pub.publish(msg)
    print(msg)
    rate.sleep()
