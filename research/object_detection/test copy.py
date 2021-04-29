import cv2 as cv
import numpy as np
import os
import tensorflow as tf

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util as label
from object_detection.utils import visualization_utils as visualization

from enum import Enum

#{ 1: {'id': 1, 'name': 'person'}
#  2: {'id': 2, 'name': 'bicycle'}
#  10: {'id': 10, 'name': 'traffic light'}
#  13: {'id': 13, 'name': 'stop sign'} }
class Road(Enum):
  PERSON = 1
  BICYCLE = 2
  TRAFFIC_LIGHT = 10
  STOP_SIGN = 13

#objects_to_detect = [Road.PERSON, Road.BICYCLE, Road.TRAFFIC_LIGHT, Road.STOP_SIGN]

def real_time_inference(model, image):
  #Convert image from the video into an array
  image = np.asarray(image)
  #print(image)

  # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
  in_tensor = tf.convert_to_tensor(image)
  #print(in_tensor)

  # The model expects a batch of images, so add an axis with `tf.newaxis`.
  in_tensor = in_tensor[tf.newaxis, ...]
  #print(in_tensor)

  # Get results for the specific frame
  # print(list(model.signatures.keys())) #'serving_default'
  model_output = model.signatures['serving_default']
  output_dict = model_output(in_tensor)
  # print(output_dict)

  # Represents the count of the objects that the model 
  # recognizes 
  num_detections = int(output_dict.pop('num_detections'))
  #print(num_detections)

  # Each entry for the dictionary keys has 100 values
  # in the respective array, hence the use of num_detections
  # to signal the end of the list. The keys are:
  #     detection_scores
  #     detection_boxes
  #     detection_classes
  for key, value in output_dict.items():
    output_dict[key] = value[0, :num_detections].numpy()  

  # The values in key='detection_classes' are of type float32
  # The visualization method requires this to be of type int and not numpy.float64
  output_dict['detection_classes'] = output_dict['detection_classes'].astype(int)
  return output_dict


def draw_box_around_object(model, frame):
  #Convert image from the video into an array
  image_array = np.array(frame)

  # Run the image through the object detection model to 
  # identify what type of objects are in the frame
  output_dict = real_time_inference(model, image_array)

  boxes = output_dict['detection_boxes']
  classes = output_dict['detection_classes']
  scores = output_dict['detection_scores']

  # Visualization of the results of a detection.
  visualization.visualize_boxes_and_labels_on_image_array(
      image_array,
      boxes,
      classes,
      scores,
      label.create_category_index_from_labelmap('./data/mscoco_label_map.pbtxt', use_display_name=True),
      instance_masks=output_dict.get('detection_masks_reframed', None),
      use_normalized_coordinates=True,
      line_thickness=5)

  #------------------------------------------------------------
  # Check and see that the following objects are detected
  #{ 1: {'id': 1, 'name': 'person'}
  #  2: {'id': 2, 'name': 'bicycle'}
  #  10: {'id': 10, 'name': 'traffic light'}
  #  13: {'id': 13, 'name': 'stop sign'} }
  #------------------------------------------------------------
  # Demand at leasat 75% confidence
  s_classes = classes[scores > 0.75]
  found = False

  #if len(s_classes) == 1 and s_classes in objects_to_detect:
  if s_classes in [1, 2, 10, 13]:
    #print(s_classes)
    # Trigger a bool to notify the Arduino that the car needs to stop
    return {"bounding_box": image_array, "found_obj": True}
  else:
    return {"bounding_box": image_array, "found_obj": False}


# A stop sign was detected so I will need to send a signal to the Arduino
# to STOP the car's movement for 3 seconds
def sendSignaltoArduino():
  print("Signal")
  return True


#------------------------------------------------------------
# Try to connect this Python script to the bluetooth module
# on the Arduino
#------------------------------------------------------------


#------------------------------------------------------------
# Load the pretrained model
#------------------------------------------------------------
pretrained_model = tf.compat.v2.saved_model.load("./models/ssd_inception_v2_coco_2017_11_17/saved_model", None)
# print(list(pretrained_model.signatures.keys()))
#------------------------------------------------------------
# Open the webcam to start detecting objects
#------------------------------------------------------------
video = cv.VideoCapture(0)

while True:
    # Capture frame-by-frame
    re, frame = video.read()
    img_results = draw_box_around_object(pretrained_model, frame)
    cv.imshow('Stop sign detection', cv.resize(img_results["bounding_box"], (1280, 960)))

    # If a object was detected, send a signal to the Arduino
    if(img_results["found_obj"]):
      sendSignaltoArduino()

    if cv.waitKey(10) & 0xFF == ord('q') or img_results["found_obj"] == True:
        break

video.release()
cv.destroyAllWindows()