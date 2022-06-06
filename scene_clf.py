import os
from PIL import Image
import shutil

import torch
from prediction import make_prediction, model


origin_folder = 'Test_images'
destination_folder = 'classified_images'
os.mkdir(destination_folder)

image_pths = os.listdir('Test_images')

# get image file names
image_names = os.listdir(origin_folder)
# print(image_names)
classes = torch.load('classes.pth')

for name in image_names:
    # get prediction
    img_src_pth = os.path.join(origin_folder, name)
    img = Image.open(img_src_pth)
    pred = make_prediction(model, img, classes)

    # Destination subfolder (based on pred) and image pth names
    dst_sub_folder = os.path.join(destination_folder, pred)
    img_dst_pth = os.path.join(dst_sub_folder, name)

    # If subfolder doesn't exist, create it and copy in image
    if not os.path.isdir(dst_sub_folder):
        os.mkdir(dst_sub_folder)
        shutil.copy(img_src_pth, img_dst_pth)

    else:
        shutil.copy(img_src_pth, img_dst_pth)
