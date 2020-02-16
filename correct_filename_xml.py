"""
Example usage:
python correct_filename_xml.py /home/juliocesar/Downloads/calle100_elevator
"""

import numpy as np
import sys
import os

from lxml import etree

import xml.etree.ElementTree as ET
import random

def main():
	data_dir = sys.argv[1]
	annotations_dir = os.path.join(data_dir, 'Annotations')
	dst_folder = os.path.join(data_dir, 'Mod_Annotations')
	xml_list = os.listdir(annotations_dir)

	if not os.path.exists(dst_folder):
		os.mkdir(dst_folder)
	
	for idx, example_xml in enumerate(xml_list):
		path = os.path.join(annotations_dir, example_xml)
		tree = ET.parse(path)
		root = tree.getroot()
		# print('example_xml: ', example_xml)
		overwrite = False
		for elem in tree.iter():
			if 'filename' in elem.tag:
				if not elem.text.endswith('.jpg'):
					elem.text += '.jpg'
					overwrite = True
			if 'path' in elem.tag:
				if not elem.text.endswith('.jpg'):
					elem.text += '.jpg'
					overwrite = True

		if overwrite:
			print('Created new file: ' + os.path.join(dst_folder, example_xml))
			tree.write(os.path.join(dst_folder, example_xml))


if __name__ == "__main__":
	main()
