# main file to run the neural network number prediction
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
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
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        # forward pass
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

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

def visualize_data(loader):
    # visualize some data from the dataset
    data_iter = iter(loader)
    images, labels = next(data_iter)
    fig, axes = plt.subplots(1, 5, figsize=(10, 2))
    for i in range(5):
        axes[i].imshow(images[i].squeeze(), cmap='gray')
        axes[i].set_title(f'Label: {labels[i].item()}')
        axes[i].axis('off')
    plt.show()

def main():
    # connect everything
    train_loader, test_loader = load_data()
    model = Net()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    load_existing = False # <- switch to load existing model
    if load_existing:
        model.load_state_dict(torch.load('model/mnist_model.pth'))
        model.eval()
        print("Loaded existing model. Testing...")
        test(model, test_loader, loss_fn)
    else:
        print("Starting training...")
        for epoch in range(1, 6):  # train for 5 epochs
            print(f'Epoch {epoch}')
            train(model, train_loader, optimizer, loss_fn, epoch)
            test(model, test_loader, loss_fn)
        print("Training complete.")

        # Save the model
        if not os.path.exists('models'):
            os.makedirs('models')
        torch.save(model.state_dict(), 'models/mnist_model.pth')
        print("Model saved as models/mnist_model.pth")

if __name__ == "__main__":
    main()

