"""
Example usage:
python extract_images_from_annotation.py /home/msdc/jcgarciaca/projects/flores/MIA2/data/initial_data
"""

import sys
import os
import cv2
import pandas as pd
from lxml import etree
import xml.etree.ElementTree as ET
from copy import deepcopy

class ExtractImages:
    def __init__(self, root_path):
        self.annotations_dir = os.path.join(root_path, 'Annotations')
        self.images_src = os.path.join(root_path, 'Images')
        self.images_tgt = os.path.join(root_path, 'Split')
        self.csv_path = os.path.join(root_path, 'data.csv')
        if not os.path.exists(self.images_tgt):
            os.mkdir(self.images_tgt)
        self.img_format = '.JPG'
        self.use_subfolder = True
        self.run()

    def run(self):
        xml_list = [id for id in os.listdir(self.annotations_dir) if id.endswith('.xml')]
        print('Found {} annotations'.format(len(xml_list)))

        data_csv = list()
        for idx, xml_sample in enumerate(xml_list):
            # check if it is xml file
            if not xml_sample.endswith('xml'):
                continue
            print('{}/{}: {}'.format(idx + 1, len(xml_list), xml_sample))
            img = cv2.imread(os.path.join(self.images_src, xml_sample.split('.xml')[0] + self.img_format))
            ann = os.path.join(self.annotations_dir, xml_sample)
            tree = ET.parse(ann)
            root = tree.getroot()            

            for elem in tree.iter():
                if 'object' in elem.tag:
                    class_name = None
                    coords = dict()
                    for attr in list(elem):
                        if 'name' in attr.tag:
                            class_name = attr.text
                        if 'bndbox' in attr.tag:
                            for data in list(attr):
                                coords[data.tag] = int(data.text)
                    if not class_name is None and len(coords.keys()) == 4:
                        if self.use_subfolder:
                            sub_folder = os.path.join(self.images_tgt, class_name)
                            if not os.path.exists(sub_folder):
                                os.mkdir(sub_folder)
                            filename = 'IMG_' + str(len(os.listdir(sub_folder)) + 1) + self.img_format
                            file_path = os.path.join(sub_folder, filename)
                        else:
                            filename = 'IMG_' + str(len(os.listdir(self.images_tgt)) + 1) + self.img_format
                            file_path = os.path.join(self.images_tgt, filename)
                        # get roi
                        sub_img = img[coords['ymin']:coords['ymax'], coords['xmin']:coords['xmax']]
                        
                        cv2.imwrite(file_path, sub_img)
                        dict_ = {'Filename': filename, 'Original': xml_sample.split('.xml')[0] + self.img_format, 'Class': class_name, 
                                'xmin': coords['xmin'], 'xmax': coords['xmax'], 'ymin': coords['ymin'], 'ymax': coords['ymax']
                                }
                        data_csv.append(deepcopy(dict_))
        df = pd.DataFrame(data_csv)
        df.to_csv(self.csv_path, index=False)

if __name__ == "__main__":
    ExtractImages(sys.argv[1])
