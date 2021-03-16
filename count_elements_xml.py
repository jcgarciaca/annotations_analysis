"""
Example usage:
python count_elements_xml.py /home/maquinavirtualnlp/jcgarciaca/flores/MIA1/golden_dataset/modified/Edtd_Annotations csv_filename
"""

import numpy as np
import sys
import os
from lxml import etree
import xml.etree.ElementTree as ET
import operator
import pandas as pd


def main():
	annotations_dir = sys.argv[1]
	root_folder = '/'.join(section for section in annotations_dir.split('/')[:-1])
	csv_file = os.path.join(root_folder, '{}.csv'.format(sys.argv[2]))
	xml_list = [xml_file for xml_file in os.listdir(annotations_dir) if xml_file.endswith('.xml')]
	print('length files:', len(xml_list))
	
	count_dict = {}	
	for idx, example_xml in enumerate(xml_list):		
		path = os.path.join(annotations_dir, example_xml)
		print('{}/{}: {}'.format(idx + 1, len(xml_list), path))
		tree = ET.parse(path)
		root = tree.getroot()
		for elem in tree.iter():
			if 'object' in elem.tag:
				for attr in list(elem):
					if 'name' in attr.tag:
						count_dict[attr.text] = count_dict.get(attr.text, 0) + 1
	
	df = pd.DataFrame.from_dict(count_dict, orient='index', columns=['Count'])
	df.sort_index(inplace=True)	
	print(df)
	df.to_csv(csv_file)
	print('{} saved'.format(csv_file))
	

if __name__ == "__main__":
	main()
