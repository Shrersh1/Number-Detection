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

    framework = models.CNN()
    optimizer = optim.Adam(framework.parameters(), lr=0.001)
    base_class_models.append(baseClass.Base(framework, train_loader, test_loader, optimizer, loss_fn, "CNN"))

    framework = models.CNN2()
    optimizer = optim.Adam(framework.parameters(), lr=0.002)
    base_class_models.append(baseClass.Base(framework, train_loader, test_loader, optimizer, loss_fn, "CNN2"))
    
    cont = True 
    while(cont):
        answer = input("Do you want to train the model/s? [y/N]: ")
        if(answer == "y" or answer == "N"):
            cont = False
    if(answer == "y"):
        print("Start training all models...")
        for base_model in base_class_models:
            print(f"Training {base_model.title}:")
            for epoch in range(1,2):
                print(f'Epoch {epoch}')
                base_model.train(epoch)
                base_model.test() 
        print("Training comlete. Saving models...")
        
        for base_model in base_class_models:
            if not os.path.exists('models'):
                os.makedirs('models')
            model_name_formatted = base_model.title.replace(" ", "_").lower()
            torch.save(base_model.model.state_dict(), f'models/{model_name_formatted}.pth')
        print("Saving complete.")

    else:
        print("Retrieving all active and trained models...")
        found_models = []
        # finding all .pth-models "active" (= not commented out) in code
        for base_model in base_class_models:
            model_name_formatted = base_model.title.replace(" ", "_").lower()
            
            if(os.path.exists(f"models/{model_name_formatted}.pth")):
                print(f"{model_name_formatted}.pth found")
                base_model.model.load_state_dict(torch.load(f"models/{model_name_formatted}.pth"))
                base_model.model.eval()
                found_models.append(base_model)
        
        base_class_models = found_models # making sure only trained models will be used to visualization
        print("Loading complete.")
        
    print("Visualizing results")

    cont = True 
    seed = random.randint(0, 2**32 - 1)
    while(cont):
        answer = input("Do you want to use test_loader [t] or paint_data [p]?: ")
        if(answer == "t" or answer == "p"):
            cont = False
    for base_model in base_class_models:
        if(answer == "t"):
            print("Test data chosen")
            base_model.visualize(test_loader, seed=seed)
        elif(answer == "p"):
            print("Paint data chosen")
            paint_loader = load_paint_data()
            base_model.visualize(paint_loader, n = 20, padding=0.25, size=1.75)
        else:
            print("Congratulations you reaching unreachable code!")
        
        
if __name__ == "__main__":
    main()
