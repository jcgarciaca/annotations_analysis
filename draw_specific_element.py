"""
Example usage:
python3 draw_specific_element.py /media/juliocesar/04968D53968D4660/Millenium/projects/flores/DJI_0289_alto IMG_1
"""

import sys
import os
import cv2
import pandas as pd

class DrawElement:
    def __init__(self, root_path, id_):
        self.images_src = os.path.join(root_path, 'Images')
        self.csv_path = os.path.join(root_path, 'data.csv')
        self.id = id_ + '.jpg'
        self.run()

    def run(self):
        df = pd.read_csv(self.csv_path, index_col='Filename')
        id_info = df.loc[self.id]
        org_img = id_info['Original']
        xmin, xmax = id_info['xmin'], id_info['xmax']
        ymin, ymax = id_info['ymin'], id_info['ymax']
        img = cv2.imread(os.path.join(self.images_src, org_img))
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 10)
        window_name = '{} - {}'.format(self.id, id_info['Class'])
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(window_name, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    DrawElement(sys.argv[1], sys.argv[2])