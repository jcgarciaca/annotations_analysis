"""
Example usage:
python count_elements_xml.py /home/msdc/jcgarciaca/projects/thalon/sku_detection
"""

import numpy as np
import sys
import os

from lxml import etree

import xml.etree.ElementTree as ET
import random
import operator


def main():
	data_dir = sys.argv[1]
	annotations_dir = os.path.join(data_dir, "Annotations")
	xml_list = os.listdir(annotations_dir)
	print('length files:', len(xml_list))

	sku_dict = {}	
	for idx, example_xml in enumerate(xml_list):
		if example_xml.endswith('.txt'):
			continue
		path = os.path.join(annotations_dir, example_xml)
		print('path:', path)
		tree = ET.parse(path)
		root = tree.getroot()
		for elem in tree.iter():
			if 'object' in elem.tag:
				for attr in list(elem):
					if 'name' in attr.tag:
						if attr.text in sku_dict.keys():
							sku_dict[attr.text] += 1
						else:
							sku_dict[attr.text] = 1
	
	print(sku_dict)
	print('------------------------------------')
	print(sorted(sku_dict.items(), key=operator.itemgetter(1)))

if __name__ == "__main__":
	main()
