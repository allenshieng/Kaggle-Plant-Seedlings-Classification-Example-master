import torch
import torch.nn as nn
from models import VGG11
from dataset import PlantSeedlingDataset
#from utils import parse_args
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from pathlib import Path
import copy
import os
import math
import matplotlib.pyplot as plt


# base_path = os.path.dirname(os.path.abspath(__file__))
# train_path = os.path.join(base_path, "train")
# print (train_path)
# exit()
#args = parse_args()

CUDA_DEVICES = torch.cuda.is_available()
DATASET_ROOT = 'C:/Users/allen/Desktop/Kaggle-Plant-Seedlings-Classification-Example-master/train'


def train():
    loss_list=[]
    accurate_list=[]
    data_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_set = PlantSeedlingDataset(DATASET_ROOT, data_transform)
    data_set_size = len(train_set)
    print(len(train_set))
    data_loader = DataLoader(dataset=train_set, batch_size=32, shuffle=True, num_workers=1)
    
    model = VGG11(num_classes=train_set.num_classes)
    if CUDA_DEVICES:
        model = model.cuda()
    model.train()

    best_model_params = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    num_epochs = 70
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params=model.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(num_epochs):
        print(f'Epoch: {epoch + 1}/{num_epochs}')
        print('-' * len(f'Epoch: {epoch + 1}/{num_epochs}'))

        training_loss = 0.0
        training_corrects = 0

        for i, (inputs, labels) in enumerate(data_loader):
            if CUDA_DEVICES:
                inputs = Variable(inputs.cuda())
                labels = Variable(labels.cuda())
            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            training_loss += loss.data * inputs.size(0)
            training_corrects += torch.sum(preds == labels.data)

        training_loss = float(training_loss)/(data_set_size)
        loss_list.append(training_loss)
        training_acc = float(training_corrects)/(data_set_size)
        accurate_list.append(training_corrects)

        print(f'Training loss: {training_loss:.4f}\taccuracy: {training_acc:.4f}\n')

        if training_acc > best_acc:
            best_acc = training_acc
            best_model_params = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_model_params)
    torch.save(model, f'model-{best_acc:.02f}-best_train_acc.pth')
    x1 = range(0,len(loss_list))
    x2 = range(0,len(accurate_list))
    y1 = accurate_list
    y2 = loss_list
    plt.subplot(2,1,1)
    plt.plot(x1,y1,'o-')
    plt.title('Test accuracy vs epoches')
    plt.ylabel('Test accuracy')
    plt.subplot(2,1,2)
    plt.plot(x2, y2, '.-')
    plt.xlabel('Test loss vs epoches')
    plt.ylabel('Test loss')
    plt.savefig("accuracy_loss.jpg")
    plt.show()
    





if __name__ == '__main__':
    train()
