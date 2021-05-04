import os
import shutil

annotations_folder = '/home/msdc/jcgarciaca/projects/flores/MIA2/landmark_detection/Marcas_2/Anotaciones'
images_folder = '/home/msdc/Desktop/videos_marcas_definitivas/frames'
target_folder = '/home/msdc/jcgarciaca/projects/flores/MIA2/landmark_detection/Marcas_2/Images'
if not os.path.exists(target_folder):
    os.makedirs(target_folder)

annotations = os.listdir(annotations_folder)
print('Found {} annotations'.format(len(annotations)))
cnt = 0
for annotation in annotations:
    img_name = annotation.split('.')[0] + '.jpg'
    if not os.path.exists(os.path.join(target_folder, img_name)) and os.path.exists(os.path.join(images_folder, img_name)):
        # copy image from source folder
        shutil.copy2(os.path.join(images_folder, img_name), os.path.join(target_folder, img_name))
        cnt += 1

print('Copied {} files'.format(cnt))