#------------------------------------------------------------
# Name: Jill Kapchan
# CSE 494: Intro to Robotics
# Description: Using a pre-trained model, this code detects
# people, bicycles, traffic lights, and stop signs and signals
# an Arduino car, through bluetooth, to stop all movement
# while at least one of the specified objects is in the frame.
#------------------------------------------------------------

import cv2 as cv
import numpy as np
import os
import tensorflow as tf

# From the Tensorflow API
from object_detection.utils import label_map_util as label
from object_detection.utils import visualization_utils as visualization

# from enum import Enum
#{ 1: {'id': 1, 'name': 'person'}
#  2: {'id': 2, 'name': 'bicycle'}
#  10: {'id': 10, 'name': 'traffic light'}
#  13: {'id': 13, 'name': 'stop sign'} }
#class Road(Enum):
#  PERSON = 1
#  BICYCLE = 2
#  TRAFFIC_LIGHT = 10
#  STOP_SIGN = 13
#objects_to_detect = [Road.PERSON, Road.BICYCLE, Road.TRAFFIC_LIGHT, Road.STOP_SIGN]

def real_time_inference(model, image):
  #Convert image from the video into an array
  image = np.asarray(image)
  #print(image)

  in_tensor = tf.convert_to_tensor(image)
  #print(in_tensor)

  # The model is expecting a batch of images
  in_tensor = in_tensor[tf.newaxis, ...]
  #print(in_tensor)

  # Get results for the specific frame
  # print(list(model.signatures.keys())) #'serving_default'
  model_output = model.signatures['serving_default']
  result_dict = model_output(in_tensor)
  # print(result_dictresult_dict)

  # Represents the count of the objects that the model recognizes 
  num_detections = int(result_dict.pop('num_detections'))
  #print(num_detections)

  # Each entry for the dictionary keys has 100 values
  # in the respective array, hence the use of num_detections
  # to signal the end of the list. The keys are:
  #     detection_scores
  #     detection_boxes
  #     detection_classes
  for key, val in result_dict.items():
    result_dict[key] = val[0, :num_detections].numpy()  

  # The values in key='detection_classes' are of type float32
  # The visualization method requires this to be of type int and not numpy.float64
  result_dict['detection_classes'] = result_dict['detection_classes'].astype(int)
  return result_dict


def draw_box_around_object(model, frame):
  #Convert image from the video into an array
  image_array = np.array(frame)

  # Run the image through the object detection model to 
  # identify what type of objects are in the frame
  output = real_time_inference(model, image_array)

  boxes = output['detection_boxes']
  classes = output['detection_classes']
  scores = output['detection_scores']

  # Visualization of the results of a detection.
  visualization.visualize_boxes_and_labels_on_image_array(
      image_array,
      boxes,
      classes,
      scores,
      label.create_category_index_from_labelmap('./data/mscoco_label_map.pbtxt', use_display_name=True),
      instance_masks=output.get('detection_masks_reframed', None),
      use_normalized_coordinates=True,
      line_thickness=7)

  #------------------------------------------------------------
  # Check and see that the following objects are detected
  #{ 1: {'id': 1, 'name': 'person'}
  #  2: {'id': 2, 'name': 'bicycle'}
  #  10: {'id': 10, 'name': 'traffic light'}
  #  13: {'id': 13, 'name': 'stop sign'} }
  #------------------------------------------------------------
  # Demand at least 75% confidence in object classification
  s_classes = classes[scores > 0.75]
  found = False

  #if len(s_classes) == 1 and s_classes in objects_to_detect:
  if s_classes in [1, 2, 10, 13].any():
    #print(s_classes)
    # Trigger a bool to notify the Arduino that the car needs to stop
    return {"bounding_box": image_array, "found_obj": True}
  else:
    return {"bounding_box": image_array, "found_obj": False}


#------------------------------------------------------------
# Log results to an external text file
# Tail -f 
#------------------------------------------------------------
logFile = open("MyFile.txt","a")
#------------------------------------------------------------
# Try to connect this Python script to the bluetooth module
# on the Arduino
#------------------------------------------------------------
import serial
import time

print("Started Bluetooth connection")

# Ports on the PC are COM5 and COM6
port = 'COM6'

# Baud rate set on the HC-05 module is 38400
bluetooth = serial.Serial(port, 38400)

print("Connected")
bluetooth.flushInput()
#------------------------------------------------------------
# Load the pretrained model
#------------------------------------------------------------
pretrained_model = tf.compat.v2.saved_model.load("./models/ssd_inception_v2_coco_2017_11_17/saved_model", None)
# print(list(pretrained_model.signatures.keys()))
#------------------------------------------------------------
# Open the webcam to start detecting objects
#------------------------------------------------------------
print("Sleeping for 15s to get a Bluetooth connection")
time.sleep(15)
video = cv.VideoCapture(0)

while True:
    # Capture frame-by-frame
    re, frame = video.read()
    img_results = draw_box_around_object(pretrained_model, frame)
    cv.imshow('Stop sign detection', cv.resize(img_results["bounding_box"], (1280, 960)))

    # One of the specified objects were detected by the model
    # Need to send a signal to the Arduino to stop movement
    if(img_results["found_obj"]):
      # Clean Bluetooth buffer
      bluetooth.flushInput()

      # The Arduino is expecting bytes to come in through Bluetooth
      bluetooth.write(b"detect")
      print("Sent a signal to the Arduino")

    # Break out of object detection loop
    if cv.waitKey(10) & 0xFF == ord('q'): #or img_results["found_obj"] == True:
        break

# Release connections
video.release()
cv.destroyAllWindows()
bluetooth.close()
print("Disconnected from Bluetooth")
logFile.close()