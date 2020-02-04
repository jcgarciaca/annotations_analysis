"""
Example usage:
python merge_sku_xml.py /home/msdc/jcgarciaca/projects/thalon/sku_detection
"""

import numpy as np
import sys
import os

from lxml import etree

import xml.etree.ElementTree as ET
import random



def main():
	data_dir = sys.argv[1]
	annotations_dir = os.path.join(data_dir, "Annotations")
	mod_anns_dir = os.path.join(data_dir, "Annotations_modified", "Annotations")
	xml_list = os.listdir(annotations_dir)

	if not os.path.exists(mod_anns_dir):
		print('Created ' + mod_anns_dir)
		os.makedirs(mod_anns_dir)
	
	for idx, example_xml in enumerate(xml_list):
		path = os.path.join(annotations_dir, example_xml)
		tree = ET.parse(path)
		root = tree.getroot()
		print('example_xml: ', example_xml)
		overwrite = False
		for elem in tree.iter():
			if 'object' in elem.tag:
				for attr in list(elem):
					if 'name' in attr.tag:
						if attr.text == "SKUCOCACOLA1.5LZERO" or attr.text == "SKUCOCACOLA1.5LORIGINAL":
							attr.text = "SKUCOCACOLA1.5"
							overwrite = True
						elif attr.text == "SKUCOCACOLA2.5LZERO" or attr.text == "SKUCOCACOLA2.5LORIGINAL":
							attr.text = "SKUCOCACOLA2.5"
							overwrite = True

		if overwrite:
			print("save new")
			tree.write(os.path.join(mod_anns_dir, example_xml))
	
if __name__ == "__main__":
	main()
