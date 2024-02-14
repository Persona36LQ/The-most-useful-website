import os

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class DatasetClass(Dataset):
    def __init__(self, path_dir1: str, path_dir2: str):
        super().__init__()

        self.path_dir1 = path_dir1
        self.path_dir2 = path_dir2

        self.dir1_list = sorted(os.listdir(path_dir1))
        self.dir2_list = sorted(os.listdir(path_dir2))

    def __len__(self):
        return len(self.dir1_list) + len(self.dir2_list)

    def __getitem__(self, idx):

        if idx < len(self.dir1_list):
            class_id = 0
            img_path = os.path.join(self.path_dir1, self.dir1_list[idx])
        else:
            class_id = 1
            idx -= len(self.dir1_list)
            img_path = os.path.join(self.path_dir2, self.dir2_list[idx])

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)

        if img is None:
            print(img_path)

        try:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(e)

        img = img.astype(np.float32)
        img = img / 255.0

        img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)
        img = img.transpose((2, 0, 1))

        t_img = torch.from_numpy(img)
        t_class_id = torch.tensor(class_id)

        return {'img': t_img, 'class_id': t_class_id}


train_dataset = DatasetClass("./PetImages/Dog", "./PetImages/Cat")
train_dataloader = DataLoader(train_dataset, batch_size=16,
                              shuffle=True, drop_last=True)


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


def train(epochs: int):
    device = torch.device('cuda')
    model = Network().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999))

    def accuracy(pred, label):
        pred = F.softmax(pred, dim=1).cpu().detach().numpy().argmax(1)
        label = label.cpu().detach().numpy().argmax(1)
        answer = pred == label
        return answer.mean()

    for epoch in range(epochs):
        loss_val = 0
        acc_val = 0

        for sample in tqdm(train_dataloader):
            img, class_id = sample['img'].to(device), sample['class_id'].to(device)
            optimizer.zero_grad()

            class_id = F.one_hot(class_id, 2).float()
            output = model(img)

            loss = criterion(output, class_id)

            loss.backward()

            loss_item = loss.item()
            loss_val += loss_item

            optimizer.step()

            acc_current = accuracy(output, class_id)
            acc_val += acc_current

        print("\nLoss value: ", loss_val / len(train_dataloader))
        print("Accurate value: ", acc_val / len(train_dataloader))
        print(f"Epoch {epoch} ended\n<{"=" * 40}>")

    torch.save(model, "model.pth")


train(1)
