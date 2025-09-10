import os
import math
import random
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

import models
import baseClass

def load_data():
    # load MNIST and DataLoader
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)
    return train_loader, test_loader

def load_paint_data(img_dir="./paintNumbers"):
    # gleiche Normalisierung wie MNIST
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),   # falls farbig gemalt
        transforms.Resize((28, 28)),                   # auf MNIST-Größe
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    dataset = baseClass.PaintDigitsDataset(img_dir=img_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    return loader


def main():
    train_loader, test_loader = load_data()
    loss_fn = nn.CrossEntropyLoss()
    #TODO: move to each model if they differ from each other
    base_class_models = []

    # comment out models you would not like to include

    framework = models.MLP()
    optimizer = optim.Adam(framework.parameters(), lr=0.001)
    base_class_models.append(baseClass.Base(framework, train_loader, test_loader, optimizer, loss_fn, "MLP Network"))

    #framework = models.CNN()
    #optimizer = optim.Adam(framework.parameters(), lr=0.001)
    #base_class_models.append(baseClass.Base(framework, train_loader, test_loader, optimizer, loss_fn, "CNN"))
    
    cont = True 
    while(cont):
        answer = input("Do you want to train model? [y/N]: ")
        if(answer == "y" or answer == "N"):
            cont = False
    if(answer == "y"):
        print("Start training all models...")
        for model in base_class_models:
            print(f"Training {model.title}:")
            for epoch in range(1,2):
                print(f'Epoch {epoch}')
                model.train(epoch)
                model.test() 
        print("Training comlete.")

        #TODO: save models (append titles by converting Big letters to small ones and spaces to underscores)

    else:
        print("Not yet implemented")

        #TODO: retrieve models 
        
    print("Visualizing some results")
    for model in base_class_models:

        #paint_loader = load_paint_data()
        #model.visualize(paint_loader, n = 10)

        model.visualize(test_loader)
        
if __name__ == "__main__":
    main()
