import cv2
import numpy as np
import torch
from torch import nn


class Network(nn.Module):
    def __init__(self):
        super().__init__()

        conv0 = nn.Conv2d(3, 128, 3, stride=1, padding=0)
        act0 = nn.LeakyReLU(0.2)
        maxpool0 = nn.MaxPool2d(2, 2)

        conv1 = nn.Conv2d(128, 128, 3, stride=1, padding=0)
        act1 = nn.LeakyReLU(0.2)
        maxpool1 = nn.MaxPool2d(2, 2)

        conv2 = nn.Conv2d(128, 128, 3, stride=1, padding=0)
        act2 = nn.LeakyReLU(0.2)

        conv3 = nn.Conv2d(128, 256, 3, stride=1, padding=0)
        act3 = nn.LeakyReLU(0.2)
        adaptivepool = nn.AdaptiveAvgPool2d((1, 1))
        flatten = nn.Flatten()
        linear1 = nn.Linear(256, 20)
        linear2 = nn.Linear(20, 2)

        self.model = nn.Sequential(
            conv0, act0, maxpool0,
            conv1, act1, maxpool1,
            conv2, act2,
            conv3, act3, adaptivepool,
            flatten, linear1, act0, linear2
        )

    def forward(self, x):
        x = self.model(x)

        return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = torch.load("model.pth", map_location=device)

file_name = "img.png"

img = cv2.imread(file_name, cv2.IMREAD_COLOR)

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

img = img.astype(np.float32)
img = img / 255.0

img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)
img = img.transpose((2, 0, 1))

t_img = torch.from_numpy(img).to(device)
t_img = t_img.unsqueeze(0)

with torch.no_grad():
    output = model(t_img).squeeze(0)

    if output[0] > 0:
        print("Most likely dog is on the picture")
    else:
        print("Most likely cat is on the picture")
