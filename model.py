#!/usr/bin/env python

import sys
from PIL import Image
import matplotlib.pyplot as plt
import sklearn.datasets
import sklearn.model_selection
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils as utils
import torchvision as tv
import torchvision.transforms as transforms

DEVICE = t.device("cuda" if t.cuda.is_available() else "cpu")
SIZE = 128

transform_test = transforms.Compose(
    [
        transforms.Resize(SIZE),
        transforms.CenterCrop(SIZE),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

transform_train = transforms.Compose(
    [
        transforms.ColorJitter(brightness=0.4, contrast=0.2, saturation=0.1, hue=0.1),
        transforms.RandomRotation(10),
        transforms.RandomResizedCrop(SIZE, scale=(0.8, 1.0), ratio=(0.95, 1.05)),
        transform_test,
    ]
)


def imshow(img):
    img = img / 2 + 0.5
    img = img.numpy()
    plt.imshow(img.transpose((1, 2, 0)))
    plt.show()


def imload(path):
    return Image.open(path).convert("RGB")


class Dataset(utils.data.Dataset):
    def __init__(self, paths, targets, *, transform=lambda x: x):
        self.paths = paths
        self.targets = targets
        self.transform = transform

    def __getitem__(self, i):
        return self.transform(imload(self.paths[i])), self.targets[i]

    def __len__(self):
        return len(self.targets)


class Data:
    def __init__(self, path, *, seed=0, device=DEVICE):
        d = sklearn.datasets.load_files(
            "data", load_content=False, shuffle=True, random_state=seed
        )

        self.classes = d.target_names

        x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(
            d.filenames, d.target, test_size=0.2, shuffle=True, random_state=seed
        )

        self.train = Dataset(x_train, y_train, transform=transform_train)
        self.test = Dataset(x_test, y_test, transform=transform_test)

    def loaders(self, *, train_batch=32, test_batch=4, num_workers=0, device=DEVICE):
        return tuple(
            utils.data.DataLoader(
                d,
                batch_size=s,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=True,
            )
            for d, s in [(self.train, train_batch), (self.test, test_batch)]
        )


class Net(nn.Module):
    def __init__(self, dout, *, device=DEVICE):
        super(Net, self).__init__()
        self.to(device=device)

        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 29 * 29, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, dout)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 29 * 29)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def model_train(n: Net, dl: utils.data.DataLoader, *, device=DEVICE):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(n.parameters(), lr=0.001, momentum=0.9)

    epochs = 40
    for epoch in range(epochs):

        running_loss = 0.0
        for i, (x, y) in enumerate(dl, 1):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            y_preds = n(x)
            loss = criterion(y_preds, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 0:
                print(
                    "[{:2}, {:5}, {:3.0f}%] loss: {:5.2f}".format(
                        epoch + 1,
                        i,
                        100.0 * (i / len(dl) + epoch) / epochs,
                        running_loss,
                    )
                )
                running_loss = 0.0


def model_test(n: Net, dl: utils.data.DataLoader, classes, *, device=DEVICE):
    correct = 0
    total = 0
    class_correct = [0] * len(classes)
    class_total = [0] * len(classes)

    for x, y in dl:
        x, y = x.to(device), y.to(device)

        y_preds = n(x)
        _, y_pred = t.max(y_preds, 1)

        total += y.size(0)
        correct += (y_pred == y).sum().item()

        c = y_pred == y
        for i, l in enumerate(y.tolist()):
            class_total[l] += 1
            class_correct[l] += c[i].item()

    print(f"Accuracy: {100. * correct / total:4.1f}% on {total}")

    for cat, cor, tot in zip(classes, class_correct, class_total):
        print(f" {cat}: {100. * cor / tot:6.2f}% on {tot:4}")


data = Data("data", seed=0)
train_l, test_l = data.loaders(train_batch=32, test_batch=256)
net = Net(len(data.classes)).to(DEVICE)


def model_eval(path: str):
    global data, net

    img = imload(path)
    img = transform_test(img).unsqueeze(0).to(DEVICE)

    outputs = net(img)
    print(outputs)
    _, y_pred = t.max(outputs.data, 1)
    return data.classes[y_pred[0].item()]


if __name__ == "__main__":
    if sys.argv[1] == "train":
        model_train(net, train_l)
        t.save(net.state_dict(), "./build/model.pth")
        print("Model saved to ./build/model.pth")
    elif sys.argv[1] == "test":
        net.load_state_dict(t.load("./build/model.pth"))
        model_test(net, test_l, data.classes)
    elif sys.argv[1] == "eval":
        net.load_state_dict(t.load("./build/model.pth"))
        # imshow(transform_test(imload(sys.argv[2])))
        print(model_eval(sys.argv[2]))
    elif sys.argv[1] == "show":
        for x, y in train_l:
            for i, j in zip(x, y):
                print(data.classes[j.item()])
                imshow(i)
