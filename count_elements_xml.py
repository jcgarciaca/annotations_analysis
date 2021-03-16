"""
Example usage:
python count_elements_xml.py /home/maquinavirtualnlp/jcgarciaca/flores/MIA1/golden_dataset/modified/Edtd_Annotations
"""

import numpy as np
import sys
import os
from lxml import etree
import xml.etree.ElementTree as ET
import operator


def main():
	annotations_dir = sys.argv[1]
	xml_list = [xml_file for xml_file in os.listdir(annotations_dir) if xml_file.endswith('.xml')]
	print('length files:', len(xml_list))
	
	count_dict = {}	
	for idx, example_xml in enumerate(xml_list):
		print('{}/{}: {}'.format(idx + 1, len(xml_list), example_xml))
		path = os.path.join(annotations_dir, example_xml)
		tree = ET.parse(path)
		root = tree.getroot()
		for elem in tree.iter():
			if 'object' in elem.tag:
				for attr in list(elem):
					if 'name' in attr.tag:
						count_dict[attr.text] = count_dict.get(attr.text, 0) + 1
	
	print(count_dict)
	# print('------------------------------------')
	# print(sorted(sku_dict.items(), key=operator.itemgetter(1)))
	

if __name__ == "__main__":
	main()
