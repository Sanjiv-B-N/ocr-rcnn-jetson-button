import os
import PIL.Image
import imageio
import numpy as np
import tensorflow as tf
from utils import label_map_util
from utils import visualization_utils as vis_util

class ButtonDetector:
  def __init__(self, model_path=None, label_path=None, verbose=False):
    self.model_path = model_path #stores path to saved model
    self.loaded_model = None
    self.label_path = label_path #stores path to label map 
    self.category_index = None #required for working with label_map and visualization
    self.input = None #None for now, image will be passed in later
    self.output = [] #inference output is stored here
    self.class_num = 1
    self.verbose = verbose 
    self.image_show = None

  def init_detector(self):

    #set paths  
    self.model_path = './d0/saved_model/'
    self.label_path = './frozen_model/button_label_map.pbtxt'

    # check existence of the two files
    if not os.path.exists(self.model_path):
      raise IOError('Invalid detector_graph path! {}'.format(self.model_path))
    if not os.path.exists(self.label_path):
      raise IOError('Invalid label path! {}'.format(self.label_path))
    
    try:
        self.loaded_model = tf.keras.models.load_model(self.model_path)
    except Exception as e:
        print("Error loading the model:", e)

    # Load label map
    label_map = label_map_util.load_labelmap(self.label_path)
    categories = label_map_util.convert_label_map_to_categories(
      label_map, max_num_classes=self.class_num, use_display_name=True)
    self.category_index = label_map_util.create_category_index(categories)

  def predict(self, image_np, draw=False):
    img_in = np.expand_dims(image_np, axis=0)

    predict_fn = self.loaded_model.signatures["serving_default"]
    preds =  predict_fn(img_in) #get predictions from model
    print(preds)
    # boxes, scores, classes, num = [np.squeeze(x) for x in [boxes, scores, classes, num]]

    # if self.verbose:
    #   self.visualize_detection_result(image_np, boxes, classes, scores, self.category_index) #if verbose is true, results are visualized
    # if draw:
    #   self.image_show = np.copy(image_np)
    #   self.draw_result(self.image_show, boxes, classes, scores, self.category_index)   

    return preds 

if __name__ == '__main__':
  detector = ButtonDetector(verbose=True)
  image = imageio.imread('./test_panels/26.jpg')
  detector.predict(image)

      
