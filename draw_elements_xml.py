"""
Example usage:
python draw_elements_xml.py /home/msdc/jcgarciaca/projects/thalon/sku_detection
"""

import numpy as np
import sys
import os
import cv2

from lxml import etree

import xml.etree.ElementTree as ET
import random


def main():
	data_dir = sys.argv[1]
	annotations_dir = os.path.join(data_dir, "Annotations")
	images_dir = os.path.join(data_dir, "Images")
	
	xml_list = os.listdir(annotations_dir)
	print('length files: {}'.format(len(xml_list)))

	color_dict = {'SKUCOCACOLA1.5': (0, 0, 255), 'SKUCOCACOLA2.5': (0, 255, 0), 
		'SKUBRISA600': (255, 0, 0), 'SKUSALTINNOELINTEGRALX3TC': (0, 255, 255),
		'SKUMANANTIAL600': (255, 0, 255), 'SKUCRISTAL600EP': (255, 255, 0)}
	
	for idx, example_xml in enumerate(xml_list):
		if not example_xml.endswith('.xml'):
			continue
		print('{}/{}: {}'.format(idx + 1, len(xml_list), example_xml))
		img = cv2.imread(os.path.join(images_dir, example_xml.split('.xml')[0] + '.JPEG'))
		path = os.path.join(annotations_dir, example_xml)
		tree = ET.parse(path)
		root = tree.getroot()
		
		for elem in tree.iter():
			if 'object' in elem.tag:
				name_enable = False
				bndbox_enable = False
				xmin = []
				ymin = []
				xmax = []
				ymax = []
				for attr in list(elem):					
					if 'name' in attr.tag:
						if attr.text in color_dict.keys():
							COLOR = color_dict[attr.text]
							name_enable = True
						
					if 'bndbox' in attr.tag:
						bndbox_enable = True
						for data in list(attr):
							if data.tag == 'xmin':
								xmin.append(int(data.text))
							if data.tag == 'ymin':
								ymin.append(int(data.text))
							if data.tag == 'xmax':
								xmax.append(int(data.text))
							if data.tag == 'ymax':
								ymax.append(int(data.text))
						
							
				if name_enable and bndbox_enable:
					for i in range(len(xmin)):
						cv2.rectangle(img, (xmin[i], ymin[i]), (xmax[i], ymax[i]), COLOR, 2)
		
		cv2.namedWindow(example_xml, cv2.WINDOW_NORMAL)
		cv2.resizeWindow(example_xml, 840, 680)
		cv2.imshow(example_xml, img)
		k = cv2.waitKey(0)
		cv2.destroyAllWindows()
		if k % 256 == 115:
			print('Error in image:', example_xml)
	
if __name__ == "__main__":
	main()
