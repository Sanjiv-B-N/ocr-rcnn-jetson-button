import os
from PIL import Image
import imageio
import cv2
import numpy as np
import tensorflow as tf
import sys
from utils import visualization_utils as vis_util
from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
path2scripts = './models/research'
sys.path.insert(0, path2scripts) # making scripts in models/research available for import
class ButtonDetector:
  def __init__(self, config_path=None, label_path=None, model_path=None, verbose=False):
    self.config_path = config_path #stores path to saved model
    self.model_path = model_path
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
    self.config_path = '.exported_models/d0/pipeline.config'
    self.label_path = './frozen_model/button_label_map.pbtxt'
    self.model_path = './exported_models/d0/checkpoint/' 

    # check existence of the files
    if not os.path.exists(self.config_path):
      raise IOError('Invalid config path! {}'.format(self.config_path))
    if not os.path.exists(self.label_path):
      raise IOError('Invalid label path! {}'.format(self.label_path))
    if not os.path.exists(self.model_path):
      raise IOError('Invalid model path! {}'.format(self.model_path))
    
    configs = config_util.get_configs_from_pipeline_file(self.config_path) # importing config
    model_config = configs['model'] # recreating model config
    self.loaded_model = model_builder.build(model_config=model_config, is_training=False) # importing model

    ckpt = tf.compat.v2.train.Checkpoint(model=self.loaded_model)
    ckpt.restore(os.path.join(self.model_path, 'ckpt-0')).expect_partial()

    self.label_path = './data/button_label_map.pbtxt' # TODO: provide a path to the label map file
    self.category_index = label_map_util.create_category_index_from_labelmap(self.label_path,use_display_name=True)

    def detect_fn(image):
      """
      Detect objects in image.
      
      Args:
        image: (tf.tensor): 4D input image
        
      Returs:
        detections (dict): predictions that model made
      """

      image, shapes = self.loaded_model.preprocess(image)
      prediction_dict = self.loaded_model.predict(image, shapes)
      detections = self.loaded_model.postprocess(prediction_dict, shapes)

      return detections
    

  def predict(self, image_np, draw=False):
      image = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)

      image, shapes = self.loaded_model.preprocess(image)
      prediction_dict = self.loaded_model.predict(image, shapes)
      detections = self.loaded_model.postprocess(prediction_dict, shapes)

      return detections

if __name__ == '__main__':
  detector = ButtonDetector(verbose=True)
  image = imageio.imread('./test_panels/26.jpg')
  t0 = cv2.getTickCount()
  detector.predict(image)
  t1 = cv2.getTickCount()
  time = (t1-t0)/cv2.getTickFrequency()
  print(time)


      
