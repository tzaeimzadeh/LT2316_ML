import os, argparse, torch
from PIL import Image
from skimage import io, transform
import numpy as np
import pandas as pd
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import torchvision
import csv
import random, math


Image.MAX_IMAGE_PIXELS = None


countries = {"Monaco": "195030", "Liechtenstein": "194027", "Luxembourg" : "197025", "Switzerland" : "195027", "Macao" : "122045", "Norway" : "198018", "Ireland" : "206023",
             "Malaysia" : "127057", "Grenada" : "001052" , "Kazakhstan" : "157027", "Lebanon" : "174036", "Cuba" : "012045", "Romania" : "184028",
             "Burundi" : "172062", "South Sudan" : "173055", "Somalia" : "163056", "Malawi" : "168069", "Niger" : "189048", "Mozambique" : "167073", "Madagascar" : "159073"}


parser = argparse.ArgumentParser()
parser.add_argument('f', metavar='foldername', help='takes the name of the root folder to be opened')
parser.add_argument('l', metavar='labeldata',
                    help='takes the name of the file containing information to be used for labelling')
args = parser.parse_args()

def open_image_files():
    """
    takes root folder name as terminal argument, open root/subfolder/files
    :return:
    """

    image = []
    country_data = {}
    img_size = []

    with open(args.l) as label_data:
        gdp_csv = pd.read_csv(label_data, sep=',', skiprows=3, usecols=['Country Name', '2016'])
        gdp_csv.set_index(keys=["Country Name"], inplace=True)
        gdp_csv.fillna(0)

    #gdp_csv["label"] = ["low" if x <= 3500.00 else "high" if x > 13000.00 else "medium" for x in gdp_csv["2016"]]
    gdp_csv["label"] = [0 if x <= 3500.00 else 2 if x > 13000.00 else 1 for x in gdp_csv["2016"]]

    for files in os.listdir(args.f):
        for k, v in countries.items():
            if files != ".DS_Store" and files[10:16] in v:
                img = io.imread(args.f+files)   #imread creates arrays
                country_data[k] = torch.from_numpy(img), torch.LongTensor(gdp_csv.loc[k])      #{'country' : (tensor([[[]]], dtype), tensor([gdp value])}
                #img_size.append(img.shape)  # height,width?

    #smallest_img_size = sorted(img_size)[0]     #(6921, 7991, 3)

    return country_data


def transform_data(Dataset):

    images = [x[0].permute(2, 0, 1) for x in Dataset.values()]  # list of tensors of shape 3*6000*7000 (after resizing)
    truths = [x[1] for x in Dataset.values()]
    #print(images[0].view())
    return images, truths


class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 3, kernel_size=5)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(5)
        #self.conv2 = nn.Conv2d()
        self.linear1 = nn.Linear(3*250*250, 3*250*250) #???? start with something small, print x in the forward step and work out what the actual sizes should be
        self.linear2 = nn.Linear(3*250*250, 3)

    def forward(self, x):
        x = self.conv1(x)   #compute output of convultion layer 1
        x = self.relu(x)   #pass output of conv1 to a non-linear activation function - ReLU
        x = self.pool(x)    #pass output of previous layer to MaxPool layer
        print(len(x))
        print(x)
        print(x.shape)
        # if decide to use more conv layers, do the previous steps again on conv2
        x = x.view(-1, 3*250*250) #to transform into a 1-dimensional layer
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))

    #TODO: do i need more conv/activation/linear layers??

        return x


def split_data(dataset):

    images = dataset[0]

    total_indices = list(range(len(images)))
    random.shuffle(total_indices)
    split_index = math.floor(len(images)*0.7)

    train_indices = total_indices[:split_index]
    test_indices = total_indices[split_index:]

    train_sample = SubsetRandomSampler(train_indices)
    test_sample = SubsetRandomSampler(test_indices)

    train_dataloader = DataLoader(dataset, batch_size=20, sampler=train_sample) #, pin_memory=True)
    test_dataloader = DataLoader(dataset, sampler=test_sample)

    return train_dataloader, test_dataloader


#TODO: training
def train(dataloader, epochs=2):

    model = CNNModel()
    optimizer = torch.optim.Adam(model.parameters())
    loss_criterion = nn.CrossEntropyLoss #??

    for epoch in range(epochs):
        total_loss = 0
        batches = 0
        #for c, data in enumerate(dataloader):
        """
            images, truths = data
            optimizer.zero_grad()
            output = model(images.float())
            loss = loss_criterion(output, truths)
            total_loss += loss
            batches += 1.0
            loss.backward()
            optimizer.step()
            """
        #print("epoch = {}, loss = {}".format(epoch, total_loss/batches))

    #return model


def test(model, dataloader):
    model.eval()
    loss_criterion = nn.CrossEntropyLoss()
    total_loss = 0
    items = 0



#TODO: send to correct device
"""
is_cuda = torch.cuda.is_available()
if is_cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
"""
if __name__ == "__main__":
    country_dataset = open_image_files()
    dataset = transform_data(country_dataset)
    train_dataloader, test_dataloader = split_data(dataset)
    #train(train_dataloader)

    #print(len(dataset[0]), len(dataset[1])) # = 4, 4

