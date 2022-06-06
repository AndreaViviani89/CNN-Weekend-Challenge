from torchvision import models
import torch.nn as nn
from data_handler import n_classes


model = models.resnext50_32x4d(pretrained=True)

inputs = model.fc.in_features
outputs = n_classes

clf = nn.Sequential(
    #               nn.Dropout(0.30),
    nn.Linear(inputs, outputs)
)

model.fc = clf
