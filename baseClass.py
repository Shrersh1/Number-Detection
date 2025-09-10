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
import matplotlib.patches as patches
from torchvision import datasets, transforms

class Base:
    def __init__(self, model, train_loader, test_loader, optimizer, loss_fn, title):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.title = title

    def train(self, epoch):
        self.model.train()
        for batch_idx, (data, target) in enumerate(self.train_loader):
            self.optimizer.zero_grad()  # zero the gradients
            output = self.model(data)  # forward pass
            self.loss = self.loss_fn(output, target)  # compute loss
            self.loss.backward()  # backward pass
            self.optimizer.step()  # update weights

            if batch_idx % 100 == 0:
                print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(self.train_loader.dataset)} '
                      f'({100. * batch_idx / len(self.train_loader):.0f}%)]\tLoss: {self.loss.item():.6f}')
    
    def test(self):
        self.model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                output = self.model(data)
                test_loss += self.loss_fn(output, target).item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        test_loss /= len(self.test_loader)  # average loss over all batches
        accuracy = correct / len(self.test_loader.dataset)
        print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {accuracy*100:.2f}%\n')
        return test_loss, accuracy

    def visualize(self, loader, n = 60, seed=None, padding=0.15, size= 1.5):
        # loader could either be test_loader or paint data
        self.model.eval()
        data_iter = iter(loader)
        images, labels = next(data_iter)
        if len(images) < n:
            print(f"Not enough images in the batch to visualize {n} images.")
            return
        
        if seed is None:
            seed = random.randint(0, 2**32 - 1)

        g = torch.Generator().manual_seed(seed)  
        indices = torch.randperm(len(images), generator=g)[:n]

        images = images[indices]
        labels = labels[indices]

        with torch.no_grad():
            output = self.model(images)
            predictions = output.argmax(dim=1) # get predicted labels

        cols = min(10, n)
        rows = math.ceil(n / cols)

        fig, axes = plt.subplots(rows, cols, figsize=(cols * size, rows * size))
        if rows == 1:
            axes = list(axes) if cols > 1 else [axes]
        elif cols == 1:
            axes = [ax for ax in axes]  # jede Zeile
        else:
            axes = [ax for row in axes for ax in row]
        correct = 0

        for i in range(n):
            axes[i].imshow(images[i].squeeze(), cmap='gray')
            
            axes[i].axis('off')
            # Farbe je nach Vorhersage
            color = 'green' if predictions[i] == labels[i] else 'red'

            # Rechteck-Rahmen um das Bild
            rect = patches.Rectangle((0, 0), images[i].shape[1]-1, images[i].shape[0]-1,
                                    linewidth=2, edgecolor=color, facecolor='none')

            axes[i].add_patch(rect)
            axes[i].set_title(f'Predicted: {predictions[i].item()}\nExpected: {labels[i].item()}')
            
            if(predictions[i].eq(labels[i])):
                correct += 1

        accuracy = correct / n * 100

        # Hide any unused axes
        for i in range(n, len(axes)):
            axes[i].axis('off')
        plt.tight_layout()

        # set title for the whole figure
        plt.suptitle(f"Labeled data by {self.title}", fontsize=16)
        plt.subplots_adjust(top=1-padding)  # adjust title position
        plt.figtext(0.5, 0.01, f"Accuracy: {accuracy:.2f}%", ha="center", fontsize=12)
        plt.subplots_adjust(bottom=padding) 
        plt.show()

        # save picture?

class PaintDigitsDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        # retrieve all images in directory
        self.img_files = [f for f in os.listdir(img_dir) if f.endswith(".png")]

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_name = self.img_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path)

        # extract label of data name
        label = int("".join([c for c in img_name if c.isdigit()]))

        if self.transform:
            image = self.transform(image)

        return image, label

def load_paint_data(img_dir="./paintNumbers"):
    # same normalisation as MNIST
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),   # if coloured
        transforms.Resize((28, 28)),                   # to MNIST-size
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    dataset = PaintDigitsDataset(img_dir=img_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    return loader