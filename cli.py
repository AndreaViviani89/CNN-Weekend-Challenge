import torch
from model_setup import model
import time
import argparse


import matplotlib.pyplot as plt
from torchvision import transforms 
from PIL import Image


# Loading model
state_dict = torch.load('model.pth')

model.load_state_dict(state_dict)


# Setting up the CLI
parser = argparse.ArgumentParser(
    description='Predict the images in Intel dataset')

parser.add_argument('file_path', type=str,
                    help='Path to samples images')

args = parser.parse_args()

# path_to_images = args.file_path

path_to_images = 'C:/Users/andre/Documents/Strive_repository/CNN-Weekend-Challenge/Test_images'

# Loading records
images = torch.load(path_to_images)

# from int to article names
class_img = {0: 'buildings',
                1: 'forest',
                2: 'glacier',
                3: 'mountain',
                4: 'sea',
                5: 'street'}

# Predictions
for img in images:

    pred = torch.argmax(model(img.view(1, -1)), dim=1)
    print(class_img[pred.item()])

    # Break (to simulate readings every 3 secondes)
    time.sleep(3)



# state_dict = torch.load('model.pth')

# model.load_state_dict(state_dict)

# model_state = state_dict['model_state']

# for i in range(9, -1, -1):
#      image_path_arg = 'C:/Users/Omistaja/Desktop/New folder/img_' +  str(i) + "_1.jpg"
#      image_path = image_path_arg
#      make_predict(image_path, model)