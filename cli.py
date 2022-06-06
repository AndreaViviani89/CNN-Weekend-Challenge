import torch
from model_setup import model
import time
import argparse

import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import models, transforms
from IPython.display import clear_output
import os
import numpy as np



# Loading model
state_dict = torch.load('model_state.pth')

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


# ---------------------------------------------------------------------------------------------------------

# model = models.resnext50_32x4d(pretrained = True)

# inputs = model.fc.in_features
# outputs = 6

# model.fc = nn.Linear(inputs, outputs) 

# model.load_state_dict( torch.load('C:/Users/andre/Documents/Strive_repository/CNN-Weekend-Challenge/model.pth') )

# # To visualize predictions
# def view_classify(img, ps):
#     labels = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

#     class_pred = labels[np.argmax(ps, axis=1)]
#     ps = ps.data.numpy().squeeze()

#     fig, (ax1, ax2) = plt.subplots(figsize=(15,15), ncols=2)
#     ax1.imshow(img)
#     ax1.axis('off')
#     ax2.barh(np.arange(6), ps)
#     ax2.set_aspect(0.1)
#     ax2.set_yticks(np.arange(6))
#     ax2.set_yticklabels(np.arange(6))
#     ax2.set_title(class_pred)
#     ax2.set_xlim(0, 1.1)

# test_transform = transforms.Compose([
#                                         transforms.Resize((150,150)),
#                                         # transforms.CenterCrop(124),
#                                         transforms.ToTensor(),
#                                         transforms.Normalize(
#                                             mean=[0.485, 0.456, 0.406],
#                                             std=[0.229, 0.224, 0.225])
#                                     ])

# # Test on self downloaded images
# root_dir = 'C:/Users/andre/Documents/dataset/Intel Image Classification/seg_pred/seg_pred'
# for img_pth in os.listdir(root_dir):
#     img = Image.open(img_pth)
#     img_transf = test_transform(img)
#     model.eval()
#     with torch.no_grad():
#         logit = model(img_transf.unsqueeze(0))
#     ps = nn.functional.softmax(logit, dim=1)
#     view_classify(img, ps)
#     plt.show()
#     time.sleep(3)
#     clear_output(wait=True)