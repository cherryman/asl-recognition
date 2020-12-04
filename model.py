#!/usr/bin/env python

import sys
import os
from datetime import datetime
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets
import sklearn.model_selection
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils as utils
import torchvision
import torchvision.transforms as transforms

DEVICE = t.device("cuda" if t.cuda.is_available() else "cpu")
SIZE = 224

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
        transforms.ColorJitter(brightness=0.4, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.RandomRotation(15),
        transforms.RandomResizedCrop(SIZE, scale=(0.8, 1.0), ratio=(0.90, 1.10)),
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

    def loaders(self, *, train_batch=32, test_batch=32, num_workers=0, device=DEVICE):
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


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


def model_train(n: nn.Module, dl: utils.data.DataLoader, *, device=DEVICE):
    n.to(device)
    n.train()
    n.apply(init_weights)

    save_dir = f"./build/{datetime.now().isoformat(timespec='seconds')}/"

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(n.parameters(), lr=0.001, momentum=0.9)

    epochs = 150
    for epoch in range(1, epochs + 1):

        running_loss = 0.0
        for i, (x, y) in enumerate(dl, 1):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            y_preds = n(x)
            loss = criterion(y_preds, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 200 == 0:
                print(
                    "[{:2}, {:5}, {:3.0f}%] loss: {:5.2f}".format(
                        epoch,
                        i,
                        100.0 * (i / len(dl) + epoch - 1) / epochs,
                        running_loss,
                    )
                )
                running_loss = 0.0

        if epoch % 5 == 0:
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)

            save_file = os.path.join(save_dir, f"epoch-{epoch:03d}.pt")
            t.save(n.state_dict(), save_file)


def model_test(n: nn.Module, dl: utils.data.DataLoader, classes, *, device=DEVICE):
    n.to(device)
    n.eval()

    correct = 0
    total = 0
    class_correct = [0] * len(classes)
    class_total = [0] * len(classes)
    confusion = np.zeros((len(classes), len(classes)), dtype=np.int32)

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
            confusion[y_pred[i]][y[i]] += 1

    print(f"Accuracy: {100. * correct / total:4.1f}% on {total}")

    for cat, cor, tot in zip(classes, class_correct, class_total):
        print(f" {cat}: {100. * cor / tot:6.2f}% on {tot:4}")

    # Print confusion matrix
    print("")
    print("Prediction \\ Actual")
    print(f"    {'   '.join(classes)}")
    for i, l in enumerate(classes):
        print(f"{l} ", end="")
        print(" ".join(f"{x:3d}" for x in confusion[i]))


def new_net():
    net = torchvision.models.vgg19(pretrained=True)
    for param in net.parameters():
        param.requires_grad = False
    net.classifier = nn.Sequential(
        nn.Linear(512 * 7 ** 2, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, len(data.classes)),
    )
    net.to(DEVICE)
    return net


def model_eval_prep(net, *, path="./build/model.pth"):
    net.load_state_dict(t.load(path))
    net.to(DEVICE)
    net.eval()
    return net


def model_eval(net, img: Image):
    img = transform_test(img).unsqueeze(0).to(DEVICE)
    outputs = net(img)[0]
    _, idx = t.topk(outputs.data, 5, sorted=True)
    return [data.classes[i] for i in idx]


data = None
if data == None:
    data = Data("data", seed=0)

if __name__ == "__main__":
    print(f"Device: {DEVICE}")

    net = new_net()
    train_l, test_l = data.loaders()

    if sys.argv[1] == "train":
        model_train(net, train_l)
        t.save(net.state_dict(), "./build/model.pth")
        print("Model saved to ./build/model.pth")
    elif sys.argv[1] == "test":
        path = "./build/model.pth" if len(sys.argv) < 3 else sys.argv[2]
        print(path)
        net.load_state_dict(t.load(path))
        model_test(net, test_l, data.classes)
    elif sys.argv[1] == "eval":
        path = "./build/model.pth" if len(sys.argv) < 4 else sys.argv[3]
        print(path)
        net.load_state_dict(t.load(path))
        img = imload(sys.argv[2])
        print(model_eval(img))
        imshow(transform_test(imload(sys.argv[2])))
    elif sys.argv[1] == "show":
        for x, y in train_l:
            for i, j in zip(x, y):
                print(data.classes[j.item()])
                imshow(i)
