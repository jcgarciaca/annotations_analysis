# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

r"""Convert the Oxford pet dataset to TFRecord for object_detection.

See: O. M. Parkhi, A. Vedaldi, A. Zisserman, C. V. Jawahar
     Cats and Dogs
     IEEE Conference on Computer Vision and Pattern Recognition, 2012
     http://www.robots.ox.ac.uk/~vgg/data/pets/

Example usage:
    python object_detection/dataset_tools/create_pet_tf_record.py \
        --data_dir=/home/user/pet \
        --output_dir=/home/user/pet/output
"""

import hashlib
import io
import logging
import os
import random
import re

import contextlib2
from lxml import etree
import numpy as np
import PIL.Image
import tensorflow as tf

from object_detection.dataset_tools import tf_record_creation_util
from object_detection.utils import dataset_util
from object_detection.utils import label_map_util

import cv2

flags = tf.app.flags
flags.DEFINE_string('data_dir', '', 'Root directory to raw pet dataset.')
flags.DEFINE_string('output_dir', '', 'Path to directory to output TFRecords.')
flags.DEFINE_string('label_map_path', 'data/label_map.pbtxt',
                    'Path to label map proto')
flags.DEFINE_string('mask_type', 'png', 'How to represent instance '
                    'segmentation masks. Options are "png" or "numerical".')
flags.DEFINE_integer('num_shards', 1, 'Number of TFRecord shards')

FLAGS = flags.FLAGS


# colors
colors = ['negro', 'amarillo', 'rojo', 'blanco', 'azul', 'morado']

# color limits
lower_black = np.array([-10, -10, 65])
upper_black = np.array([10, 10, 145])
lower_yellow = np.array([20, 245, 215])
upper_yellow = np.array([40, 265, 295])
lower_red = np.array([-1, 123, 215])
upper_red = np.array([19, 143, 295])
lower_white = np.array([-10, -10, 215])
upper_white = np.array([10, 10, 295])
lower_blue = np.array([110, 245, 215])
upper_blue = np.array([130, 265, 295])
lower_purple = np.array([110, 188, 72])
upper_purple = np.array([130, 208, 152])


contador_negro = 0
contador_amarillo = 0
contador_rojo = 0
contador_blanco = 0
contador_azul = 0
contador_morado = 0


def reset_counters():
  global contador_negro, contador_amarillo, contador_rojo, contador_blanco, contador_azul, contador_morado
  contador_negro = 0
  contador_amarillo = 0
  contador_rojo = 0
  contador_blanco = 0
  contador_azul = 0
  contador_morado = 0


def count_element(class_name):
  global contador_negro, contador_amarillo, contador_rojo, contador_blanco, contador_azul, contador_morado
  if class_name == 'negro':
    contador_negro += 1
  elif class_name == 'amarillo':
    contador_amarillo += 1
  elif class_name == 'rojo':
    contador_rojo += 1
  elif class_name == 'blanco':
    contador_blanco += 1
  elif class_name == 'azul':
    contador_azul += 1
  elif class_name == 'morado':
    contador_morado += 1


def print_counters(section):
  print('****************************************************')
  print('*********************** ' + section + ' ***********************')
  print('****************************************************')
  print('negro tiene {} elementos'.format(contador_negro))
  print('amarillo tiene {} elementos'.format(contador_amarillo))
  print('rojo tiene {} elementos'.format(contador_rojo))
  print('blanco tiene {} elementos'.format(contador_blanco))
  print('azul tiene {} elementos'.format(contador_azul))
  print('morado tiene {} elementos'.format(contador_morado))
  print('****************************************************')
  print('****************************************************')
  print('****************************************************')


def get_limits(index):
  if index == 0:
    lower_color = lower_black
    upper_color = upper_black
  elif index == 1:
    lower_color = lower_yellow
    upper_color = upper_yellow
  elif index == 2:
    lower_color = lower_red
    upper_color = upper_red
  elif index == 3:
    lower_color = lower_white
    upper_color = upper_white
  elif index == 4:
    lower_color = lower_blue
    upper_color = upper_blue
  elif index == 5:
    lower_color = lower_purple
    upper_color = upper_purple
  return lower_color, upper_color


def analize_mask(img):
  hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
  
  masks = []
  bndboxes = []
  colors_list = []
  
  for index, color in enumerate(colors):
    # define range of color in HSV
    lower_color, upper_color = get_limits(index)
    mask = cv2.inRange(hsv, lower_color, upper_color)
    # cv2.imwrite('/home/JulioCesar/segmentacion_cintas/mask_filter.png', mask)
    _, contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # mask = cv2.bitwise_not(mask)
  
  
    if len(contours) > 0:
      print('--color ' + color + ' has {} contours'.format(len(contours)))
      
    for i, contour in enumerate(contours):
      if(cv2.contourArea(contour) > 600):
        mask_tmp = np.zeros(mask.shape, np.uint8)
        xmin, ymin, w, h = cv2.boundingRect(contour)
        xmax = xmin + w
        ymax = ymin + h
        mask_tmp[ymin:ymax, xmin:xmax] = mask[ymin:ymax, xmin:xmax]
        # cv2.imwrite('/home/JulioCesar/segmentacion_cintas/mask_filter_1.png', mask_tmp)
        mask_tmp = cv2.bitwise_not(mask_tmp)
        
        # cv2.imwrite('/home/JulioCesar/segmentacion_cintas/mask_filter_2.png', mask_tmp)
        
        mask_tmp = mask_tmp / 255
        mask_tmp = mask_tmp + 1
        
        # print('(0, 0)', mask_tmp[0, 0])
        # print('(281, 209)', mask_tmp[209, 281])
        
        # cv2.imwrite('/home/JulioCesar/segmentacion_cintas/mask_filter_3.png', mask_tmp)
        
        mask_remapped = (mask_tmp != 2).astype(np.uint8)
        masks.append(mask_remapped)
        bndbox = [xmin, xmax, ymin, ymax]
        bndboxes.append(bndbox)
        colors_list.append(color)
    
  return bndboxes, masks, colors_list




def get_class_name_from_filename(file_name):
  """Gets the class name from a file.

  Args:
    file_name: The file name to get the class name from.
               ie. "american_pit_bull_terrier_105.jpg"

  Returns:
    A string of the class name.
  """
  match = re.match(r'([A-Za-z_]+)(_[0-9]+\.jpg)', file_name, re.I)
  return match.groups()[0]


def dict_to_tf_example(filename,
                       mask_path,
                       label_map_dict,
                       image_subdirectory,
                       ignore_difficult_instances=False,
                       mask_type='png'):
  """Convert XML derived dict to tf.Example proto.

  Notice that this function normalizes the bounding box coordinates provided
  by the raw data.

  Args:
    data: dict holding PASCAL XML fields for a single image (obtained by
      running dataset_util.recursive_parse_xml_to_dict)
    mask_path: String path to PNG encoded mask.
    label_map_dict: A map from string label names to integers ids.
    image_subdirectory: String specifying subdirectory within the
      Pascal dataset directory holding the actual image data.
    ignore_difficult_instances: Whether to skip difficult instances in the
      dataset  (default: False).    
    mask_type: 'numerical' or 'png'. 'png' is recommended because it leads to
      smaller file sizes.

  Returns:
    example: The converted tf.Example.

  Raises:
    ValueError: if the image pointed to by data['filename'] is not a valid JPEG
  """
  img_path = os.path.join(image_subdirectory, filename + '.jpg')
  with tf.gfile.GFile(img_path, 'rb') as fid:
    encoded_jpg = fid.read()
  encoded_jpg_io = io.BytesIO(encoded_jpg)
  image = PIL.Image.open(encoded_jpg_io)
  if image.format != 'JPEG':
    raise ValueError('Image format not JPEG')
  key = hashlib.sha256(encoded_jpg).hexdigest()

  mask_cv = cv2.imread(mask_path)
  height, width, channels = mask_cv.shape  
  
  
  
  
  
  '''
  with tf.gfile.GFile(mask_path, 'rb') as fid:
    encoded_mask_png = fid.read()
  encoded_png_io = io.BytesIO(encoded_mask_png)
  mask = PIL.Image.open(encoded_png_io)

  # print("mask pixel", mask[400][200])
  if mask.format != 'PNG':
    raise ValueError('Mask format not PNG')

  print("img_path:", img_path)
  print("mask_path:", mask_path)  
  mask_np = np.asarray(mask)

  width = int(data['size']['width'])
  height = int(data['size']['height'])
  '''

  xmins = []
  ymins = []
  xmaxs = []
  ymaxs = []
  classes = []
  classes_text = []
  truncated = []
  poses = []
  difficult_obj = []
  # masks = []
  
  print('-filename', filename)
  bndboxes, masks, colors_list = analize_mask(mask_cv)
  print('--bndboxes length is {}, masks length is {} and colors_list length is {}'.format(len(bndboxes), len(masks), len(colors_list)))  
    
  for index, bndbox in enumerate(bndboxes):
    xmin, xmax, ymin, ymax = bndbox
  
    difficult = bool(0)
    if ignore_difficult_instances and difficult:
      continue
    difficult_obj.append(int(difficult))
      
    xmins.append(xmin / width)
    ymins.append(ymin / height)
    xmaxs.append(xmax / width)
    ymaxs.append(ymax / height)
      
    # classes_text.append(obj['name'].encode('utf8'))
    # classes.append(label_map_dict[obj['name']])
    # print("label_map_dict[obj['name']]", label_map_dict[obj['name']])
      
    class_name = colors_list[index]
    print('---box ' + str(index + 1) + ' is class_name ' + class_name + ' with label_map_dict[class_name] # ' + str(label_map_dict[class_name]) + ' xmin: ' + str(xmin) + ', ymin: ' + str(ymin) + ', xmax: ' + str(xmax) + ', ymax: ' + str(ymax))
    count_element(class_name)
    classes_text.append(class_name.encode('utf8'))
    classes.append(label_map_dict[class_name])
      
    truncated.append(0)
    poses.append('Unspecified'.encode('utf8'))
     
    #  mask_remapped = (mask_np != 2).astype(np.uint8)
    #  masks.append(mask_remapped)

  feature_dict = {
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(filename.encode('utf8')),
      'image/source_id': dataset_util.bytes_feature(filename.encode('utf8')),
      'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
      'image/encoded': dataset_util.bytes_feature(encoded_jpg),
      'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
      'image/object/difficult': dataset_util.int64_list_feature(difficult_obj),
      'image/object/truncated': dataset_util.int64_list_feature(truncated),
      'image/object/view': dataset_util.bytes_list_feature(poses),
  }
  
  if mask_type == 'png':
    encoded_mask_png_list = []
    for mask in masks:
      img = PIL.Image.fromarray(mask)
      output = io.BytesIO()
      img.save(output, format='PNG')
      encoded_mask_png_list.append(output.getvalue())
    feature_dict['image/object/mask'] = (
        dataset_util.bytes_list_feature(encoded_mask_png_list))

  example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
  return example


def create_tf_record(output_filename,
                     num_shards,
                     label_map_dict,
                     annotations_dir,
                     image_dir,
                     examples,
                     mask_type='png'):
  """Creates a TFRecord file from examples.

  Args:
    output_filename: Path to where output file is saved.
    num_shards: Number of shards for output file.
    label_map_dict: The label map dictionary.
    annotations_dir: Directory where annotation files are stored.
    image_dir: Directory where image files are stored.
    examples: Examples to parse and save to tf record.
    mask_type: 'numerical' or 'png'. 'png' is recommended because it leads to
      smaller file sizes.
  """
  with contextlib2.ExitStack() as tf_record_close_stack:
    output_tfrecords = tf_record_creation_util.open_sharded_output_tfrecords(
        tf_record_close_stack, output_filename, num_shards)
    for idx, f_example in enumerate(examples):
      if idx % 100 == 0:
        logging.info('On image %d of %d', idx, len(examples))
      example = os.path.splitext(f_example)[0]
      mask_path = os.path.join(annotations_dir, example + '.png')
      # print('example', example)
      # print('mask_path', mask_path)
            
      try:
        tf_example = dict_to_tf_example(
            example,
            mask_path,
            label_map_dict,
            image_dir,
            mask_type=mask_type)
        if tf_example:
          shard_idx = idx % num_shards
          output_tfrecords[shard_idx].write(tf_example.SerializeToString())
      except ValueError:
        logging.warning('Invalid example: %s, ignoring.', xml_path)
      print('-----------------------------')

# TODO(derekjchow): Add test for pet/PASCAL main files.
def main(_):
  data_dir = FLAGS.data_dir
  label_map_dict = label_map_util.get_label_map_dict(FLAGS.label_map_path)

  logging.info('Reading dataset.')
  image_dir = os.path.join(data_dir, 'images')
  annotations_dir = os.path.join(data_dir, 'annotation_masks')
  
  examples_list = os.listdir(annotations_dir)
  # examples_list = examples_list[:100]

  # Test images are not included in the downloaded data set, so we shall perform
  # our own split.
  random.seed(42)
  random.shuffle(examples_list)
  num_examples = len(examples_list)
  num_train = int(0.7 * num_examples)
  train_examples = examples_list[:num_train]
  val_examples = examples_list[num_train:]
  logging.info('%d training and %d validation examples.',
               len(train_examples), len(val_examples))

  train_name = 'output_train_tf.record'
  val_name = 'output_val_tf.record'
  train_output_path = os.path.join(FLAGS.output_dir, train_name)
  val_output_path = os.path.join(FLAGS.output_dir, val_name)
  
  reset_counters()
  create_tf_record(
      train_output_path,
      FLAGS.num_shards,
      label_map_dict,
      annotations_dir,
      image_dir,
      train_examples,
      mask_type=FLAGS.mask_type)
  print_counters("train")
  reset_counters()
  create_tf_record(
      val_output_path,
      FLAGS.num_shards,
      label_map_dict,
      annotations_dir,
      image_dir,
      val_examples,
      mask_type=FLAGS.mask_type)
  print_counters("eval")

if __name__ == '__main__':
  tf.app.run()
