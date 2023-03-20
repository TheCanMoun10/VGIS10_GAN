import numpy as np
from datetime import datetime

import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import DataLoader
# from torchvision import datasets, transforms

import matplotlib.pyplot as plt
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(DEVICE)

def plot_losses(train_losses):
    '''
    Function for plotting training and validation losses
    Takes as input train_losses and valid_losses as matrices and plots the losses.
    '''
    plt.style.use('seaborn')
    
    train_losses = np.array(train_losses)
    
    fig, ax = plt.subplots(figsize = (8, 4.5))
    
    ax.plot(train_losses, color='blue', label='Training loss')
    ax.set(title='Loss over epochs', xlabel='Epochs', ylabel='Loss')
    ax.legend()
    fig.show()
    
    plt.style.use('default')
    

def train(train_loader, model, cost, optimizer, num_epochs, device):
    '''
    Function for the training step of the training loop.
    '''
    total_step = len(train_loader)
    run_loss = 0
    
    print(f'Using device: ', device)
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))
    
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):

            images = images.to(device)
            labels = labels.to(device)
            
            # Forward Pass:
            outputs = model(images)
            loss = cost(outputs, labels)
            run_loss += loss.item() * images.size(0)
            
            # Backward pass (learning step) and optimization:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (i+1) % 3000 == 0:
                print(f'{datetime.now().time().replace(microsecond=0)} --- ' 
                    f'Epoch [{epoch+1}/{num_epochs}]\t'
                    f'Step [{i+1}/{total_step}]\t'
                    f'Training Loss: {loss.item():.4f}'
                    )
     
    train_loss = run_loss / total_step
    return model, optimizer, train_loss

def test(test_loader, model, device):
    '''
    Test function for the training loop.
    '''
    
    with torch.no_grad():
        correct = 0
        total = 0
        
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return total, correct

def training_loop(model, cost, optimizer, train_loader, test_loader, num_epochs, device):
    '''
    Function defining the training loop
    '''
    
    train_losses = []

    # Iniate training on the model:
    model, optimizer, train_loss = train(train_loader, model, cost, optimizer, num_epochs, device)
    train_losses.append(train_loss)
        
    # Testing
    total_predictions, correct_predictions = test(test_loader, model, device)
    model_accuracy = 100 * correct_predictions / total_predictions

    
    print(f'Accuracy of the network on the 10000 test images: {model_accuracy} %')
    
    return model, optimizer, train_losses, model_accuracy


    
