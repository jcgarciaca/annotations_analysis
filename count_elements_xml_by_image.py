"""
Example usage:
python count_elements_xml_by_image.py /home/msdc/jcgarciaca/projects/thalon/sku_detection
"""

import numpy as np
import sys
import os
from lxml import etree
import xml.etree.ElementTree as ET
import pandas as pd
from copy import deepcopy

def main():
	data_dir = sys.argv[1]
	annotations_dir = os.path.join(data_dir, "Edtd_Annotations")
	print('Found {} files:'.format(len(os.listdir(annotations_dir))))

	data = {}
	for idx in range(10, 68):
		objects_dict = {}
		example_xml = 'DJI_02{}.xml'.format(idx)
		path = os.path.join(annotations_dir, example_xml)
		print('path:', path)
		tree = ET.parse(path)
		root = tree.getroot()
		for elem in tree.iter():
			if 'object' in elem.tag:
				for attr in list(elem):
					if 'name' in attr.tag:
						objects_dict[attr.text] = objects_dict.get(attr.text, 0) + 1
		data[example_xml] = deepcopy(objects_dict)
	
	df = pd.DataFrame.from_dict(data, orient='index')
	# print(df)
	df = df.reindex(sorted(df.columns), axis=1)
	csv_path = sys.argv[1].split('/')[-1] + '.csv'
	df.to_csv(csv_path)
	print('{} saved'.format(csv_path))

if __name__ == "__main__":
	main()
