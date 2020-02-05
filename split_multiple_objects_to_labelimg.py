"""
Example usage:
python split_multiple_objects_to_labelimg.py /home/msdc/jcgarciaca/projects/thalon/sku_detection
"""

import os
import sys
import cv2
from lxml import etree
import xml.etree.ElementTree as ET


def main():
    data_dir = sys.argv[1]
    org_annotations_dir = os.path.join(data_dir, "Annotations")
    images_dir = os.path.join(data_dir, "Images")
    mod_annotations_dir = os.path.join(data_dir, "Split_annotations")

    if not os.path.exists(mod_annotations_dir):
        print('Created ' + mod_annotations_dir)
        os.makedirs(mod_annotations_dir)

    xml_list = os.listdir(org_annotations_dir)

    for idx, example_xml in enumerate(xml_list):
        if not example_xml.endswith('.xml'):
            continue
        print('{}/{}: {}'.format(idx + 1, len(xml_list), example_xml))

        # original path
        path = os.path.join(org_annotations_dir, example_xml)
        tree = ET.parse(path)
        root = tree.getroot()

        img_name = example_xml.split('.xml')[0] + '.JPEG'
        img_path = os.path.join(images_dir, img_name)

        img = cv2.imread(img_path)
        height, width, channels = img.shape
        
        annotation = ET.Element("annotation")
        ET.SubElement(annotation, "folder").text = "Images"
        ET.SubElement(annotation, "filename").text = img_name
        ET.SubElement(annotation, "path").text = img_path

        source = ET.SubElement(annotation, "source")
        ET.SubElement(source, "database").text = "Unknown"

        size = ET.SubElement(annotation, "size")
        ET.SubElement(size, "width").text = str(width)
        ET.SubElement(size, "height").text = str(height)
        ET.SubElement(size, "depth").text = str(channels)
        ET.SubElement(annotation, "segmented").text = "0"

        for elem in tree.iter():
            if 'object' in elem.tag:
                name_enable = False
                bndbox_enable = False
                name_object = ''
                xmin = []
                ymin = []
                xmax = []
                ymax = []                
                for attr in list(elem):
                    if 'name' in attr.tag:
                        name_object = attr.text
                        name_enable = True

                    if 'bndbox' in attr.tag:
                        bndbox_enable = True
                        for data in list(attr):
							if data.tag == 'xmin':
								xmin.append(data.text)
							if data.tag == 'ymin':
								ymin.append(data.text)
							if data.tag == 'xmax':
								xmax.append(data.text)
							if data.tag == 'ymax':
								ymax.append(data.text)

                if name_enable and bndbox_enable:
                    for i in range(len(xmin)):
                        # add object to modified xml
                        object_annotation = ET.SubElement(annotation, "object")
                        ET.SubElement(object_annotation, "name").text = name_object
                        ET.SubElement(object_annotation, "pose").text = "Unspecified"
                        ET.SubElement(object_annotation, "truncated").text = "0"
                        ET.SubElement(object_annotation, "difficult").text = "0"
                        bndbox = ET.SubElement(object_annotation, "bndbox")
                        ET.SubElement(bndbox, "xmin").text = xmin[i]
                        ET.SubElement(bndbox, "ymin").text = ymin[i]
                        ET.SubElement(bndbox, "xmax").text = xmax[i]
                        ET.SubElement(bndbox, "ymax").text = ymax[i]

        tree = ET.ElementTree(annotation)
        root = tree.getroot()
        xmlstr = ET.tostring(root, encoding='utf8', method='xml')
        
        tree.write(os.path.join(mod_annotations_dir, example_xml))


if __name__ == "__main__":
    main()