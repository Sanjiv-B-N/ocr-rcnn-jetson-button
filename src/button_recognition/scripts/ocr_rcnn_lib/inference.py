#!/usr/bin/env python
from __future__ import print_function
import os
import cv2
import imageio.v2 as imageio
import PIL.Image
import io
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from button_detection import ButtonDetector
from character_recognition import CharacterRecognizer

def button_candidates(boxes, scores, image):
  img_height = image.shape[0]
  img_width = image.shape[1]

  button_scores = [] #stores the score of each button (confidence)
  button_patches = [] #stores the cropped image that encloses the button
  button_positions = [] #stores the coordinates of the bounding box on buttons

  for box, score in zip(boxes, scores):
    if score < 0.5: continue

    y_min = int(box[0] * img_height)
    x_min = int(box[1] * img_width)
    y_max = int(box[2] * img_height)
    x_max = int(box[3] * img_width)

    button_patch = image[y_min: y_max, x_min: x_max]
    button_patch = cv2.resize(button_patch, (180, 180))

    button_scores.append(score)
    button_patches.append(button_patch)
    button_positions.append([x_min, y_min, x_max, y_max])
  return button_patches, button_positions, button_scores

def get_image_name_list(target_path):
    assert os.path.exists(target_path)
    image_name_list = [] #stores names of all files in target_path
    file_set = os.walk(target_path)
    for root, dirs, files in file_set:
      for image_name in files:
        image_name_list.append(image_name.split('.')[0])
    return image_name_list

def warm_up(detector, recognizer):
  image = imageio.imread('./test_panels/1.jpg')
  button = imageio.imread('./test_buttons/0_0.png')
  detector.predict(image)
  recognizer.predict(button)

if __name__ == '__main__':
    data_dir = './test_panels'
    data_list = get_image_name_list(data_dir)
    detector = ButtonDetector()
    recognizer = CharacterRecognizer(verbose=False)
    warm_up(detector, recognizer)
    overall_time = 0
    for data in data_list:
      img_path = os.path.join(data_dir, data+'.jpg')
      with open(img_path, 'rb') as f:
          img_np = np.asarray(PIL.Image.open(io.BytesIO(f.read())))
      # img_np = np.asarray(PIL.Image.open(tf.gfile.GFile(img_path)))
      t0 = cv2.getTickCount()
      boxes, scores, _ = detector.predict(img_np) #get boxes and scores
      button_patches, button_positions, _ = button_candidates(boxes, scores, img_np) #pass boxes and scores to this function and get button_patches and button positions
      for button_img in button_patches:
        button_text, button_score, _ =recognizer.predict(button_img) #get button text and button_score for each of the images in button_patches
      t1 = cv2.getTickCount()
          
      overall_time += time
      print('Time elapsed: {}'.format(time))

    average_time = overall_time / len(data_list)
    print('Average_used:{}'.format(average_time))
    detector.clear_session()
    recognizer.clear_session()

