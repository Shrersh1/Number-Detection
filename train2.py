# main file to run the neural network number prediction
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

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # define layers 
        #self.flatten = nn.Flatten()
        #self.fc1 = nn.Linear(28 * 28, 128)
        #self.fc2 = nn.Linear(128, 64)
        #self.fc3 = nn.Linear(64, 10)
        #self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)  # 28x28x1 -> 28x28x32
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1) # 28x28x32 -> 28x28x64
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0) # 28x28x64 -> 14x14x64
        self.fc1 = nn.Linear(14 * 14 * 64, 128) # 14*14*64 -> 128
        self.fc2 = nn.Linear(128, 10) # 128 -> 10   
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        # forward pass
        #x = self.flatten(x)
        #x = self.relu(self.fc1(x))
        #x = self.relu(self.fc2(x))
        #x = self.fc3(x)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))    
        x = self.pool(x)
        x = x.view(-1, 14 * 14 * 64)  # flatten
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class PaintDigitsDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        # alle Dateien in Ordner
        self.img_files = [f for f in os.listdir(img_dir) if f.endswith(".png")]

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_name = self.img_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path)

        # Label aus Dateinamen extrahieren (z.B. Picture3.png -> 3)
        label = int("".join([c for c in img_name if c.isdigit()]))

        if self.transform:
            image = self.transform(image)

        return image, label

def load_paint_data(img_dir="./paintNumbers"):
    # gleiche Normalisierung wie MNIST
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),   # falls farbig gemalt
        transforms.Resize((28, 28)),                   # auf MNIST-Größe
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    dataset = PaintDigitsDataset(img_dir=img_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    return loader

def train(model, loader, optimizer, loss_fn, epoch):
    # train epoch
    model.train()
    for batch_idx, (data, target) in enumerate(loader):
        optimizer.zero_grad()  # zero the gradients
        output = model(data)  # forward pass
        loss = loss_fn(output, target)  # compute loss
        loss.backward()  # backward pass
        optimizer.step()  # update weights

        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(loader.dataset)} '
                  f'({100. * batch_idx / len(loader):.0f}%)]\tLoss: {loss.item():.6f}')

def test(model, loader, loss_fn):
    # Accuracy and loss on test set
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in loader:
            output = model(data)
            test_loss += loss_fn(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(loader)  # average loss over all batches
    accuracy = correct / len(loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {accuracy*100:.2f}%\n')
    return test_loss, accuracy

def visualize_labeled_data(model, loader, n = 20): 
    # visualize some labeled data form Neural Network
    # change n to visualize more or fewer images
    model.eval()  # set model to evaluation mode
    data_iter = iter(loader)
    images, labels = next(data_iter)
    if len(images) < n:
        print(f"Not enough images in the batch to visualize {n} images.")
        return
    
    #(if you want to run model on n random images)
    #indices = random.sample(range(len(images)), n)
    #images = images[indices]
    #labels = labels[indices]

    images = images[:n]  # take first n images
    labels = labels[:n]  # take first n labels

    with torch.no_grad():
        output = model(images)
        predictions = output.argmax(dim=1) # get predicted labels

    
    cols = min(5, n)
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5))
    axes = axes.flatten()  

    for i in range(n):
        axes[i].imshow(images[i].squeeze(), cmap='gray')
        axes[i].set_title(f'Predicted: {predictions[i].item()}\nExpected: {labels[i].item()}')
        axes[i].axis('off')

    # Hide any unused axes
    for i in range(n, len(axes)):
        axes[i].axis('off')
    plt.tight_layout()

    # set title for the whole figure
    plt.suptitle('Visualizing Labeled Data from Neural Network using convolution', fontsize=16)
    plt.subplots_adjust(top=0.9)  # adjust title position
    if(is_docker()):
        os.makedirs("/app/outputs", exist_ok=True)
        plt.savefig("/app/outputs/visualization.png")
        print("Running in Docker, saving visualization to /tmp/visualization.png")
    else:
        print("Displaying visualization...")
        plt.show()

def is_docker():
    # Checks, if /proc/1/cgroup contains "docker" or "kubepods"
    # to determine if running in a container
    try:
        with open('/proc/1/cgroup', 'rt') as f:
            return 'docker' in f.read() or 'kubepods' in f.read()
    except FileNotFoundError:
        return False


def main():
    # connect everything
    train_loader, test_loader = load_data()
    model = Net()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    load_existing = True # <- switch to load existing model
    if load_existing:
        model.load_state_dict(torch.load('models/mnist_model2.pth'))
        model.eval()
        print("Loaded existing model. Testing...")

        test(model, test_loader, loss_fn)

        print("Visualizing self made data from Neural Network...")
        # deinen normalen MNIST-Loader
        train_loader, test_loader = load_data()

        # jetzt dein Paint-Dataset als "test_loader"
        paint_loader = load_paint_data("./paintNumbers")
        visualize_labeled_data(model, paint_loader, n=10)

    else:
        print("Starting training...")
        for epoch in range(1, 4):  # train for 1 epoch to keep it quick and show errors
            print(f'Epoch {epoch}')
            train(model, train_loader, optimizer, loss_fn, epoch)
            test(model, test_loader, loss_fn)
        print("Training complete.")

        print("Visualizing some labeled data from Neural Network...")
        visualize_labeled_data(model, test_loader)

        # Save the model
        if not os.path.exists('models'):
            os.makedirs('models')
        torch.save(model.state_dict(), 'models/mnist_model2.pth')
        print("Model saved as models/mnist_model2.pth")

    # TODO: add feature to draw numbers and predict them with the model

if __name__ == "__main__":
    main()

